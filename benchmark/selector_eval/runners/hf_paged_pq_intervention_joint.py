#!/usr/bin/env python3
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from benchmark.selector_eval.gpu.run_gpu_paged_pq_eval import (
    _sync_if_cuda,
    load_selector_paged_pq_ext,
    rank_paged_pq_batched_with_scores,
    selector_bytes_fullscan,
)
from benchmark.selector_eval.runners.hf_paged_pq_intervention_common import (
    MB,
    _choose_joint_kv_action,
    _env_int,
    _env_truthy,
    _joint_kv_policy_id,
    _rel_l2_torch,
)
from benchmark.selector_eval.runners.hf_paged_pq_intervention_joint_budget import joint_kv_budget_schedule_for
from benchmark.selector_eval.runners.hf_paged_pq_intervention_joint_workspace import (
    JointExactLogitHelper,
    JointKVWorkspace,
)
from benchmark.selector_eval.runners.hf_paged_pq_intervention_joint_head_group import (
    JointKVHeadGroupRuntime,
    process_joint_kv_head_groups,
)
from benchmark.selector_eval.runners.hf_paged_pq_intervention_joint_grouped_risk import (
    JointGroupedRiskRuntime,
    process_grouped_risk_records,
)
from benchmark.selector_eval.runners.hf_paged_pq_intervention_forward_state import PagedPQForwardState
from benchmark.selector_eval.runners.hf_paged_pq_intervention_stats import ApproxStats


def _quantize_e4m3_torch(x: torch.Tensor) -> torch.Tensor:
    """Round fp32 values to the OCP fp8-e4m3 grid (round-nearest-even),
    saturating to +-448, subnormal floor 2^-9. Torch mirror of
    _quantize_e4m3 in run_joint_kv_budget_policy_eval.py (frozen logit
    buffer format, issue #6) — keep the two in sync."""
    out = x.to(dtype=torch.float32).clamp(-448.0, 448.0)
    absx = out.abs()
    e = torch.floor(torch.log2(absx.clamp_min(1e-45))).clamp_(-6.0, 8.0)
    step = torch.exp2(e - 3.0)
    q = torch.round(out / step) * step
    return q.clamp_(-448.0, 448.0)


@dataclass(frozen=True)
class JointKVDecodeContext:
    args: Any
    model: Any
    module: Any
    layer_id: int
    stats: dict[int, ApproxStats]
    device: torch.device
    hidden_states: torch.Tensor
    q_all: torch.Tensor
    keys_all: torch.Tensor
    torch_k_cache: dict[int, torch.Tensor]
    torch_v_cache: dict[int, torch.Tensor]
    dense_decode_key_t_float_cache: Any
    num_heads: int
    num_kv_heads: int
    group_size: int
    nprobes: list[int]
    online_confidence_rule: str
    key_bytes: int
    value_bytes: int
    wall_profile_enabled: bool
    forward_state: PagedPQForwardState


def approximate_joint_kv_all_heads(
    ctx: JointKVDecodeContext,
    local_qpos: int,
    query_context_len: int,
) -> torch.Tensor | None:
    args = ctx.args
    model = ctx.model
    self = ctx.module
    layer_id = ctx.layer_id
    stats = ctx.stats
    device = ctx.device
    hidden_states = ctx.hidden_states
    q_all = ctx.q_all
    keys_all = ctx.keys_all
    torch_k_cache = ctx.torch_k_cache
    torch_v_cache = ctx.torch_v_cache
    dense_decode_key_t_float_cache = ctx.dense_decode_key_t_float_cache
    num_heads = ctx.num_heads
    num_kv_heads = ctx.num_kv_heads
    group_size = ctx.group_size
    nprobes = ctx.nprobes
    online_confidence_rule = ctx.online_confidence_rule
    key_bytes = ctx.key_bytes
    value_bytes = ctx.value_bytes
    wall_profile_enabled = ctx.wall_profile_enabled
    prefix_index_for = ctx.forward_state.prefix_index_for
    gqa_native_fullscan_pack = ctx.forward_state.gqa_native_fullscan_pack
    joint_vpq_sidecars_for = ctx.forward_state.joint_vpq_sidecars_for
    grouped_vpq_residual_sidecars_for = ctx.forward_state.grouped_vpq_residual_sidecars_for
    grouped_vpq_compact_sidecars_for = ctx.forward_state.grouped_vpq_compact_sidecars_for
    joint_vpq_pack_and_fallback_for = ctx.forward_state.joint_vpq_pack_and_fallback_for

    if online_confidence_rule != "joint_kv_stability":
        return None
    if not _env_truthy("SELECTOR_PQ_JOINT_GQA_BATCHED", "0"):
        return None
    if str(args.selector_mode) != "fullscan" or str(args.selector_backend) not in {"cuda_ext", "auto"}:
        return None
    if str(args.tail_score_calibration) not in {"none", "affine_selected"}:
        return None

    policy_name = str(getattr(args, "joint_kv_policy", "k_first_alternating"))
    policy_id = int(_joint_kv_policy_id(policy_name))
    policy_uses_mb = policy_name == "sensitivity_greedy"
    needs_logical_accounting = not bool(getattr(args, "disable_cost_stats", False))
    needs_budget_mb_vectors = bool(policy_uses_mb or needs_logical_accounting)

    budget_schedule = joint_kv_budget_schedule_for(
        args=args,
        device=device,
        context_len=int(query_context_len),
    )
    joint_k_budgets = budget_schedule.k_budgets
    joint_v_budgets = budget_schedule.v_budgets
    joint_v_budgets_t = budget_schedule.v_budgets_t
    if not joint_k_budgets or not joint_v_budgets:
        return None

    context_len_i = int(query_context_len)
    sqrt_dim = float(math.sqrt(float(self.head_dim)))
    prob_dtype = torch.float32 if _env_truthy("SELECTOR_PQ_JOINT_FP32_PROBS", "1") else torch.float64
    outputs_all = torch.empty((num_heads, int(self.head_dim)), dtype=torch.float32, device=device)
    allhead_precompute = _env_truthy("SELECTOR_PQ_JOINT_ALLHEAD_PRECOMPUTE", "1")
    allhead_exact_precompute = _env_truthy("SELECTOR_PQ_JOINT_ALLHEAD_EXACT_PRECOMPUTE", "0")
    use_grouped_risk_prefix = (
        _env_truthy("SELECTOR_PQ_JOINT_GROUPED_RISK_PREFIX", "0")
        and _env_truthy("SELECTOR_PQ_JOINT_GRID_ARTIFACTS", "1")
        and _env_truthy("SELECTOR_PQ_JOINT_NATIVE_RISK_PREFIX", "0")
        and not _env_truthy("SELECTOR_PQ_JOINT_INCREMENTAL_V_GRID", "0")
        and not _env_truthy("SELECTOR_PQ_JOINT_ONDEMAND_V_PREFIX", "0")
    )
    staged_kv_prefix_enabled = _env_truthy("SELECTOR_PQ_JOINT_STAGED_KV_PREFIX", "0")
    if staged_kv_prefix_enabled:
        if not use_grouped_risk_prefix:
            raise RuntimeError("SELECTOR_PQ_JOINT_STAGED_KV_PREFIX requires grouped risk-prefix mode")
        if policy_uses_mb:
            raise RuntimeError("SELECTOR_PQ_JOINT_STAGED_KV_PREFIX currently supports non-MB policies only")
        if allhead_exact_precompute:
            raise RuntimeError(
                "SELECTOR_PQ_JOINT_STAGED_KV_PREFIX is incompatible with all-head exact-logit precompute"
            )
    stage_k_steps = min(
        len(joint_k_budgets),
        max(2, _env_int("SELECTOR_PQ_JOINT_STAGED_KV_K_STEPS", 2)),
    )
    stage_v_steps = min(
        len(joint_v_budgets),
        max(2, _env_int("SELECTOR_PQ_JOINT_STAGED_KV_V_STEPS", 3)),
    )
    staged_kv_active = bool(
        staged_kv_prefix_enabled
        and (stage_k_steps < len(joint_k_budgets) or stage_v_steps < len(joint_v_budgets))
    )
    grouped_risk_records: list[dict[str, object]] = []
    joint_total_wall_t0 = time.perf_counter() if wall_profile_enabled else 0.0
    native_exact_logits_enabled = _env_truthy("SELECTOR_PQ_JOINT_NATIVE_EXACT_LOGITS", "0")
    native_exact_logits_backend = str(
        os.environ.get("SELECTOR_PQ_JOINT_NATIVE_EXACT_LOGITS_BACKEND", "cublas_t")
    ).strip().lower()
    if native_exact_logits_backend not in {"cublas_t", "custom", "grouped"}:
        raise RuntimeError(
            "SELECTOR_PQ_JOINT_NATIVE_EXACT_LOGITS_BACKEND must be 'cublas_t', 'custom', or 'grouped'"
        )
    allhead_indexes: list[GPUIndex] | None = None
    allhead_dense_pq_scores_t: torch.Tensor | None = None
    allhead_selector_mb: float | None = None
    allhead_exact_scores_t: torch.Tensor | None = None
    allhead_selector_rank_prefix_t: torch.Tensor | None = None
    use_unsorted_k_prefix = _env_truthy("SELECTOR_PQ_JOINT_UNSORTED_K_PREFIX", "0")
    exact_logit_helper = JointExactLogitHelper(
        args=args,
        device=device,
        layer_id=int(layer_id),
        context_len=context_len_i,
        group_size=int(group_size),
        sqrt_dim=sqrt_dim,
        keys_all=keys_all,
        dense_decode_key_t_float_cache=dense_decode_key_t_float_cache,
        backend=native_exact_logits_backend,
    )
    native_full_exact_logits = exact_logit_helper.full_exact_logits
    joint_precompute_wall_t0 = time.perf_counter() if wall_profile_enabled else 0.0
    if bool(getattr(args, "profile_native_ops", False)):
        _sync_if_cuda(device)
        joint_precompute_t0 = time.perf_counter()
    else:
        joint_precompute_t0 = 0.0
    if allhead_precompute:
        candidate_indexes = [prefix_index_for(int(kv_head), context_len_i) for kv_head in range(num_kv_heads)]
        if candidate_indexes and all(index.pages for index in candidate_indexes):
            allhead_indexes = candidate_indexes
            try:
                native = load_selector_paged_pq_ext()
                codebooks, codes, page_starts = gqa_native_fullscan_pack(allhead_indexes)
                selector_rank_budget = 0
                if (
                    _env_truthy("SELECTOR_PQ_JOINT_SELECTOR_TOPK_PREFIX", "0")
                    and not use_unsorted_k_prefix
                ):
                    ranked_count_upper = int(codes.shape[1]) * int(codes.shape[2])
                    active_budget_upper = (
                        list(joint_k_budgets[:stage_k_steps])
                        if staged_kv_active
                        else list(joint_k_budgets)
                    )
                    if _env_truthy("SELECTOR_PQ_JOINT_COLLAPSE_DUP_K_ROWS", "0"):
                        collapsed_upper: list[int] = []
                        seen_upper: set[int] = set()
                        for k_budget in active_budget_upper:
                            take_i = max(0, min(int(k_budget), ranked_count_upper))
                            if int(take_i) in seen_upper:
                                continue
                            seen_upper.add(int(take_i))
                            collapsed_upper.append(int(k_budget))
                        if collapsed_upper:
                            active_budget_upper = collapsed_upper
                    avoid_full_budget_rank_upper = bool(
                        _env_truthy("SELECTOR_PQ_JOINT_SKIP_FULL_BUDGET_SORT", "0")
                        or (
                            _env_truthy("SELECTOR_PQ_JOINT_COLLAPSE_DUP_K_ROWS", "0")
                            and _env_truthy("SELECTOR_PQ_JOINT_EXACT_FULL_BUDGET_GRID", "1")
                        )
                    )
                    if avoid_full_budget_rank_upper:
                        selector_rank_budget = max(
                            [
                                max(0, min(int(v), ranked_count_upper))
                                for v in active_budget_upper
                                if max(0, min(int(v), ranked_count_upper)) < ranked_count_upper
                            ],
                            default=0,
                        )
                    else:
                        selector_rank_budget = max(
                            [max(0, min(int(v), ranked_count_upper)) for v in active_budget_upper],
                            default=0,
                        )
                selector_wall_t0 = time.perf_counter() if wall_profile_enabled else 0.0
                if bool(getattr(args, "profile_native_ops", False)):
                    _sync_if_cuda(device)
                    selector_t0 = time.perf_counter()
                _top_tokens_t, _top_scores_t, allhead_dense_pq_scores_t = native.gqa_fullscan_pq_topk_scores(
                    q_all[:, int(local_qpos), :].to(device=device, dtype=torch.float32).contiguous(),
                    codebooks,
                    codes,
                    page_starts,
                    int(group_size),
                    int(selector_rank_budget),
                )
                allhead_dense_pq_scores_t = allhead_dense_pq_scores_t.to(device=device, dtype=torch.float32)
                if str(getattr(args, "logit_buffer_format", "fp")) == "e4m3":
                    # Frozen-sim: the ranking/decision domain is the e4m3
                    # logit buffer (issue #6). Quantize the PQ score row at
                    # the source so start indices, softmax base, risk, and
                    # tail all inherit the buffer format — and re-derive
                    # the rank prefix from the QUANTIZED values (stable
                    # sort => ties in token order, matching the CPU golden
                    # ranked_cpu ordering).
                    allhead_dense_pq_scores_t = _quantize_e4m3_torch(allhead_dense_pq_scores_t)
                    if int(selector_rank_budget) > 0:
                        _sorted = torch.sort(
                            allhead_dense_pq_scores_t, dim=-1, descending=True, stable=True
                        )
                        _top_tokens_t = _sorted.indices[:, : int(selector_rank_budget)]
                if int(selector_rank_budget) > 0:
                    allhead_selector_rank_prefix_t = _top_tokens_t.to(device=device, dtype=torch.long).contiguous()
                allhead_selector_mb = (
                    selector_bytes_fullscan(
                        allhead_indexes[0],
                        key_bytes=int(key_bytes),
                        subbits=int(args.subbits),
                    )
                    / MB
                )
                if bool(getattr(args, "profile_native_ops", False)):
                    _sync_if_cuda(device)
                    stats[layer_id].add_native_timing(
                        selector_seconds=float(time.perf_counter() - selector_t0)
                    )
                if wall_profile_enabled:
                    stats[layer_id].add_joint_wall_timing(
                        selector_seconds=float(time.perf_counter() - selector_wall_t0)
                    )
            except Exception:
                if str(args.selector_backend) == "cuda_ext":
                    raise
                allhead_dense_pq_scores_t = None
                allhead_selector_mb = None
        if allhead_indexes is not None and allhead_exact_precompute:
            try:
                exact_wall_t0 = time.perf_counter() if wall_profile_enabled else 0.0
                if bool(getattr(args, "profile_native_ops", False)):
                    _sync_if_cuda(device)
                    exact_t0 = time.perf_counter()
                key_count_i = min(max(0, context_len_i), int(keys_all.shape[1]))
                if key_count_i > 0:
                    if native_exact_logits_enabled:
                        allhead_exact_scores_t = native_full_exact_logits(
                            q_all[:num_heads, int(local_qpos), :].to(
                                device=device,
                                dtype=torch.float32,
                            ),
                            keys_all,
                        )
                    else:
                        keys_t_cache = dense_decode_key_t_float_cache(
                            layer_id=int(layer_id),
                            keys_all=keys_all,
                            key_count=key_count_i,
                        )
                        allhead_exact_scores_t = torch.empty(
                            (num_heads, key_count_i),
                            dtype=torch.float32,
                            device=device,
                        )
                        dim_i = int(self.head_dim)
                        group_i = max(1, int(group_size))
                        covered_heads = min(int(num_heads), int(num_kv_heads) * group_i)
                        aligned_heads = (covered_heads // group_i) * group_i
                        if aligned_heads > 0:
                            aligned_kv_heads = aligned_heads // group_i
                            queries_grouped_t = q_all[:aligned_heads, int(local_qpos), :].to(
                                device=device,
                                dtype=torch.float32,
                            ).reshape(aligned_kv_heads, group_i, dim_i)
                            if keys_t_cache is not None:
                                keys_grouped_t = keys_t_cache[:aligned_kv_heads, :, :key_count_i]
                            else:
                                keys_grouped_t = keys_all[:aligned_kv_heads, :key_count_i, :].to(
                                    device=device,
                                    dtype=torch.float32,
                                ).transpose(1, 2).contiguous()
                            allhead_exact_scores_t[:aligned_heads] = (
                                torch.bmm(queries_grouped_t, keys_grouped_t).reshape(aligned_heads, key_count_i)
                                / sqrt_dim
                            )
                        for tail_kv_head in range(aligned_heads // group_i, int(num_kv_heads)):
                            head_start_tail = int(tail_kv_head) * group_i
                            head_end_tail = min(int(num_heads), head_start_tail + group_i)
                            if head_start_tail >= head_end_tail:
                                continue
                            queries_tail_t = q_all[head_start_tail:head_end_tail, int(local_qpos), :].to(
                                device=device,
                                dtype=torch.float32,
                            )
                            if keys_t_cache is not None:
                                keys_tail_t = keys_t_cache[int(tail_kv_head), :, :key_count_i]
                            else:
                                keys_tail_t = keys_all[int(tail_kv_head), :key_count_i, :].to(
                                    device=device,
                                    dtype=torch.float32,
                                ).transpose(0, 1).contiguous()
                            allhead_exact_scores_t[head_start_tail:head_end_tail] = (
                                torch.matmul(queries_tail_t, keys_tail_t) / sqrt_dim
                            )
                if bool(getattr(args, "profile_native_ops", False)):
                    _sync_if_cuda(device)
                    stats[layer_id].add_native_detail_timing(
                        exact_logit_seconds=float(time.perf_counter() - exact_t0)
                    )
                if wall_profile_enabled:
                    stats[layer_id].add_joint_wall_timing(
                        exact_logit_seconds=float(time.perf_counter() - exact_wall_t0)
                    )
            except Exception:
                if str(args.selector_backend) == "cuda_ext":
                    raise
                allhead_exact_scores_t = None
    if bool(getattr(args, "profile_native_ops", False)):
        _sync_if_cuda(device)
        stats[layer_id].add_joint_detail_timing(
            precompute_seconds=float(time.perf_counter() - joint_precompute_t0)
        )
    if wall_profile_enabled:
        stats[layer_id].add_joint_wall_timing(
            precompute_seconds=float(time.perf_counter() - joint_precompute_wall_t0)
        )

    grouped_strided_output_workspace_enabled = _env_truthy(
        "SELECTOR_PQ_JOINT_STRIDED_OUTPUT_WORKSPACE",
        "0",
    )
    joint_workspace = JointKVWorkspace(
        args=args,
        model=model,
        module=self,
        device=device,
        context_len=context_len_i,
        grouped_strided_output_workspace_enabled=grouped_strided_output_workspace_enabled,
    )
    score_grid_workspace_for = joint_workspace.score_grid_workspace_for
    grouped_score_grid_workspace_for = joint_workspace.grouped_score_grid_workspace_for
    softmax_base_workspace_for = joint_workspace.softmax_base_workspace_for
    grouped_output_workspace_for = joint_workspace.grouped_output_workspace_for
    native_rank_prefix_tokens = joint_workspace.native_rank_prefix_tokens
    grouped_risk_prefix_workspace_for = joint_workspace.grouped_risk_prefix_workspace_for
    grouped_score_direct_workspace_for = joint_workspace.grouped_score_direct_workspace_for
    nocalib_score_grid_workspace_for = joint_workspace.nocalib_score_grid_workspace_for
    nocalib_scatter_score_grid_workspace_for = joint_workspace.nocalib_scatter_score_grid_workspace_for
    token_layout_for = joint_workspace.token_layout_for
    allhead_rank_prefix_cache: dict[tuple[int, int, int, int], torch.Tensor] = {}

    grouped_vpq_vhat_groups_t: torch.Tensor | None = None
    grouped_vpq_residual_groups_t: torch.Tensor | None = None
    grouped_vpq_code_error_groups_t: torch.Tensor | None = None
    grouped_vpq_value_codebooks_t: torch.Tensor | None = None
    grouped_vpq_value_codes_t: torch.Tensor | None = None
    grouped_vpq_value_page_starts_t: torch.Tensor | None = None
    grouped_vpq_value_page_size: int | None = None
    grouped_vpq_values_t: torch.Tensor | None = None
    grouped_vpq_actual_subbits: int | None = None
    grouped_output_workspace_enabled = (
        use_grouped_risk_prefix
        and _env_truthy("SELECTOR_PQ_JOINT_GROUPED_OUTPUT_WORKSPACE", "0")
        and int(group_size) > 0
        and int(num_heads) == int(num_kv_heads) * int(group_size)
    )
    if grouped_strided_output_workspace_enabled:
        if not grouped_output_workspace_enabled:
            raise RuntimeError(
                "SELECTOR_PQ_JOINT_STRIDED_OUTPUT_WORKSPACE requires SELECTOR_PQ_JOINT_GROUPED_OUTPUT_WORKSPACE"
            )
        if not _env_truthy("SELECTOR_PQ_JOINT_RISK_PREFIX_WORKSPACE", "0"):
            raise RuntimeError(
                "SELECTOR_PQ_JOINT_STRIDED_OUTPUT_WORKSPACE requires SELECTOR_PQ_JOINT_RISK_PREFIX_WORKSPACE"
            )
        if _env_truthy("SELECTOR_PQ_JOINT_SOFTMAX_BASE_CUBLAS", "0") or _env_truthy(
            "SELECTOR_PQ_JOINT_GROUPED_SOFTMAX_BASE_CUBLAS",
            "0",
        ):
            raise RuntimeError(
                "SELECTOR_PQ_JOINT_STRIDED_OUTPUT_WORKSPACE does not support CUBLAS softmax/base"
            )
        if _env_truthy("SELECTOR_PQ_JOINT_FUSED_RISK_POLICY", "0") or _env_truthy(
            "SELECTOR_PQ_JOINT_INTERVAL_RISK_POLICY",
            "0",
        ):
            raise RuntimeError(
                "SELECTOR_PQ_JOINT_STRIDED_OUTPUT_WORKSPACE does not support fused/interval risk policy paths"
            )
    grouped_score_workspace_enabled = (
        use_grouped_risk_prefix
        and _env_truthy("SELECTOR_PQ_JOINT_GROUPED_SCORE_WORKSPACE", "0")
        and int(group_size) > 0
        and int(num_heads) == int(num_kv_heads) * int(group_size)
    )
    grouped_probs_workspace_t: torch.Tensor | None = None
    grouped_base_workspace_t: torch.Tensor | None = None
    if use_grouped_risk_prefix:
        gqa_indexes_for_grouped = (
            allhead_indexes
            if allhead_indexes is not None
            else [prefix_index_for(int(kv_head), context_len_i) for kv_head in range(num_kv_heads)]
        )
        if gqa_indexes_for_grouped and all(index.pages for index in gqa_indexes_for_grouped):
            grouped_vpq_wall_t0 = time.perf_counter() if wall_profile_enabled else 0.0
            if bool(getattr(args, "profile_native_ops", False)):
                _sync_if_cuda(device)
                grouped_vpq_t0 = time.perf_counter()
            else:
                grouped_vpq_t0 = 0.0
            if _env_truthy("SELECTOR_PQ_JOINT_COMPACT_VPQ_RISK_PREFIX", "0"):
                grouped_compact_vpq = grouped_vpq_compact_sidecars_for(
                    gqa_indexes_for_grouped,
                    context_len_i=context_len_i,
                )
                if grouped_compact_vpq is not None:
                    (
                        grouped_vpq_value_codebooks_t,
                        grouped_vpq_value_codes_t,
                        grouped_vpq_value_page_starts_t,
                        grouped_vpq_value_page_size,
                        grouped_vpq_code_error_groups_t,
                        grouped_vpq_actual_subbits,
                    ) = grouped_compact_vpq
                    grouped_vpq_values_t = ctx.forward_state.values_all[:, :context_len_i, :]
            else:
                grouped_vpq = grouped_vpq_residual_sidecars_for(
                    gqa_indexes_for_grouped,
                    context_len_i=context_len_i,
                )
                if grouped_vpq is not None:
                    (
                        grouped_vpq_vhat_groups_t,
                        grouped_vpq_residual_groups_t,
                        grouped_vpq_code_error_groups_t,
                        grouped_vpq_actual_subbits,
                    ) = grouped_vpq
            if bool(getattr(args, "profile_native_ops", False)):
                _sync_if_cuda(device)
                stats[layer_id].add_native_detail_timing(
                    output_seconds=float(time.perf_counter() - grouped_vpq_t0)
                )
            if wall_profile_enabled:
                stats[layer_id].add_joint_wall_timing(
                    vpq_sidecar_seconds=float(time.perf_counter() - grouped_vpq_wall_t0)
                )

    threshold_value = float(getattr(args, "joint_kv_stability_threshold", 0.001))

    def make_head_group_runtime(
        *,
        k_budgets: list[int],
        v_budgets: list[int],
        v_budgets_t: torch.Tensor,
        records: list[dict[str, object]],
        kv_head_indices: list[int] | None = None,
    ) -> JointKVHeadGroupRuntime:
        return JointKVHeadGroupRuntime(
            args=args,
            self=self,
            layer_id=int(layer_id),
            stats=stats,
            device=device,
            q_all=q_all,
            torch_k_cache=torch_k_cache,
            torch_v_cache=torch_v_cache,
            context_len_i=int(context_len_i),
            num_heads=int(num_heads),
            num_kv_heads=int(num_kv_heads),
            group_size=int(group_size),
            nprobes=nprobes,
            key_bytes=int(key_bytes),
            value_bytes=int(value_bytes),
            local_qpos=int(local_qpos),
            sqrt_dim=float(sqrt_dim),
            prob_dtype=prob_dtype,
            policy_id=int(policy_id),
            policy_uses_mb=bool(policy_uses_mb),
            needs_logical_accounting=bool(needs_logical_accounting),
            needs_budget_mb_vectors=bool(needs_budget_mb_vectors),
            joint_k_budgets=k_budgets,
            joint_v_budgets=v_budgets,
            joint_v_budgets_t=v_budgets_t,
            allhead_indexes=allhead_indexes,
            allhead_dense_pq_scores_t=allhead_dense_pq_scores_t,
            allhead_selector_mb=allhead_selector_mb,
            allhead_exact_scores_t=allhead_exact_scores_t,
            allhead_selector_rank_prefix_t=allhead_selector_rank_prefix_t,
            allhead_rank_prefix_cache=allhead_rank_prefix_cache,
            use_unsorted_k_prefix=bool(use_unsorted_k_prefix),
            native_exact_logits_enabled=bool(native_exact_logits_enabled),
            native_full_exact_logits=native_full_exact_logits,
            use_grouped_risk_prefix=bool(use_grouped_risk_prefix),
            grouped_output_workspace_enabled=bool(grouped_output_workspace_enabled),
            grouped_strided_output_workspace_enabled=bool(grouped_strided_output_workspace_enabled),
            grouped_score_workspace_enabled=bool(grouped_score_workspace_enabled),
            grouped_vpq_vhat_groups_t=grouped_vpq_vhat_groups_t,
            grouped_vpq_residual_groups_t=grouped_vpq_residual_groups_t,
            grouped_vpq_code_error_groups_t=grouped_vpq_code_error_groups_t,
            grouped_vpq_value_codebooks_t=grouped_vpq_value_codebooks_t,
            grouped_vpq_value_codes_t=grouped_vpq_value_codes_t,
            grouped_vpq_value_page_starts_t=grouped_vpq_value_page_starts_t,
            grouped_vpq_value_page_size=grouped_vpq_value_page_size,
            grouped_vpq_values_t=grouped_vpq_values_t,
            grouped_vpq_actual_subbits=grouped_vpq_actual_subbits,
            grouped_risk_records=records,
            outputs_all=outputs_all,
            prefix_index_for=prefix_index_for,
            joint_vpq_sidecars_for=joint_vpq_sidecars_for,
            joint_vpq_pack_and_fallback_for=joint_vpq_pack_and_fallback_for,
            token_layout_for=token_layout_for,
            nocalib_score_grid_workspace_for=nocalib_score_grid_workspace_for,
            nocalib_scatter_score_grid_workspace_for=nocalib_scatter_score_grid_workspace_for,
            score_grid_workspace_for=score_grid_workspace_for,
            grouped_score_grid_workspace_for=grouped_score_grid_workspace_for,
            grouped_output_workspace_for=grouped_output_workspace_for,
            softmax_base_workspace_for=softmax_base_workspace_for,
            native_rank_prefix_tokens=native_rank_prefix_tokens,
            wall_profile_enabled=bool(wall_profile_enabled),
            kv_head_indices=kv_head_indices,
        )

    def make_grouped_risk_runtime(
        *,
        records: list[dict[str, object]],
        v_budgets: list[int],
        v_budgets_t: torch.Tensor,
        head_group_runtime: JointKVHeadGroupRuntime,
        staged_prefix_pass: bool = False,
    ) -> JointGroupedRiskRuntime:
        return JointGroupedRiskRuntime(
            args=args,
            self=self,
            layer_id=int(layer_id),
            stats=stats,
            device=device,
            wall_profile_enabled=bool(wall_profile_enabled),
            grouped_risk_records=records,
            grouped_strided_output_workspace_enabled=bool(grouped_strided_output_workspace_enabled),
            grouped_vpq_vhat_groups_t=grouped_vpq_vhat_groups_t,
            grouped_vpq_residual_groups_t=grouped_vpq_residual_groups_t,
            grouped_vpq_code_error_groups_t=grouped_vpq_code_error_groups_t,
            grouped_vpq_value_codebooks_t=grouped_vpq_value_codebooks_t,
            grouped_vpq_value_codes_t=grouped_vpq_value_codes_t,
            grouped_vpq_value_page_starts_t=grouped_vpq_value_page_starts_t,
            grouped_vpq_value_page_size=grouped_vpq_value_page_size,
            grouped_vpq_values_t=grouped_vpq_values_t,
            grouped_risk_prefix_workspace_for=grouped_risk_prefix_workspace_for,
            grouped_score_direct_workspace_for=grouped_score_direct_workspace_for,
            joint_v_budgets=v_budgets,
            joint_v_budgets_t=v_budgets_t,
            key_bytes=int(key_bytes),
            value_bytes=int(value_bytes),
            policy_id=int(policy_id),
            policy_uses_mb=bool(policy_uses_mb),
            threshold_value=float(threshold_value),
            outputs_all=outputs_all,
            head_group_runtime=head_group_runtime,
            staged_prefix_pass=bool(staged_prefix_pass),
        )

    if staged_kv_active:
        stage_records: list[dict[str, object]] = []
        stage_k_budgets = list(joint_k_budgets[:stage_k_steps])
        stage_v_budgets = list(joint_v_budgets[:stage_v_steps])
        stage_v_budgets_t = joint_v_budgets_t[:stage_v_steps].contiguous()
        stage_runtime = make_head_group_runtime(
            k_budgets=stage_k_budgets,
            v_budgets=stage_v_budgets,
            v_budgets_t=stage_v_budgets_t,
            records=stage_records,
        )
        if not process_joint_kv_head_groups(stage_runtime):
            return None
        boundary_kv_heads = process_grouped_risk_records(
            make_grouped_risk_runtime(
                records=stage_records,
                v_budgets=stage_v_budgets,
                v_budgets_t=stage_v_budgets_t,
                head_group_runtime=stage_runtime,
                staged_prefix_pass=True,
            )
        )
        if boundary_kv_heads:
            full_records: list[dict[str, object]] = []
            full_runtime = make_head_group_runtime(
                k_budgets=joint_k_budgets,
                v_budgets=joint_v_budgets,
                v_budgets_t=joint_v_budgets_t,
                records=full_records,
                kv_head_indices=sorted(int(v) for v in boundary_kv_heads),
            )
            if not process_joint_kv_head_groups(full_runtime):
                return None
            process_grouped_risk_records(
                make_grouped_risk_runtime(
                    records=full_records,
                    v_budgets=joint_v_budgets,
                    v_budgets_t=joint_v_budgets_t,
                    head_group_runtime=full_runtime,
                )
            )
    else:
        head_group_runtime = make_head_group_runtime(
            k_budgets=joint_k_budgets,
            v_budgets=joint_v_budgets,
            v_budgets_t=joint_v_budgets_t,
            records=grouped_risk_records,
        )
        if not process_joint_kv_head_groups(head_group_runtime):
            return None
        process_grouped_risk_records(
            make_grouped_risk_runtime(
                records=grouped_risk_records,
                v_budgets=joint_v_budgets,
                v_budgets_t=joint_v_budgets_t,
                head_group_runtime=head_group_runtime,
            )
        )

    if wall_profile_enabled:
        stats[layer_id].add_joint_wall_timing(
            total_seconds=float(time.perf_counter() - joint_total_wall_t0)
        )
    return outputs_all.to(dtype=hidden_states.dtype, device=device)
