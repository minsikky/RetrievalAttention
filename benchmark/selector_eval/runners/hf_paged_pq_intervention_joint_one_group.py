#!/usr/bin/env python3
from __future__ import annotations

import math
import time

import torch

from benchmark.selector_eval.gpu.run_gpu_paged_pq_eval import (
    _sync_if_cuda,
    load_selector_paged_pq_ext,
    rank_paged_pq_batched_with_scores,
)
from benchmark.selector_eval.runners.hf_paged_pq_intervention_common import (
    MB,
    _env_truthy,
)
from benchmark.selector_eval.runners.hf_paged_pq_intervention_joint_budget import (
    joint_value_cost_for,
)
from benchmark.selector_eval.runners.hf_paged_pq_intervention_joint_finalize import (
    JointFinalizeRuntime,
    finalize_joint_head_outputs,
)
from benchmark.selector_eval.runners.hf_paged_pq_intervention_joint_fused_policy import (
    JointFusedPolicyRuntime,
    try_process_fused_mixed_policy,
)
from benchmark.selector_eval.runners.hf_paged_pq_intervention_joint_policy import (
    JointPolicyRuntime,
    finish_prepared_torch_policy,
    select_joint_kv_budgets,
)
from benchmark.selector_eval.runners.hf_paged_pq_intervention_joint_vprefix import (
    JointVPrefixGridRuntime,
    build_joint_vprefix_grid,
)

# Frozen progressive-precision tiers (--joint_kv_precision_tiers, spec
# OPEN-2/M6): the top 10% of the K ranked prefix / V risk-ranked exact set
# reads full precision; the rest reads the per-row absmax-int8 plane, with
# V lo reads additionally gated by the per-token commit test
# (fp16(int8_err) < fp16(code_error)). Mirrors --precision_k_hi_frac 0.1
# --precision_v_hi_frac 0.1 --precision_lo_mode int8 --precision_lo_bits 8
# in run_joint_kv_budget_policy_eval.py — keep in sync.
_FROZEN_PRECISION_K_HI_FRAC = 0.1
_FROZEN_PRECISION_V_HI_FRAC = 0.1
_FROZEN_PRECISION_LO_BYTES = 1


def _rowwise_int8_qdq(x32_t: torch.Tensor) -> torch.Tensor:
    """Per-row symmetric absmax int8 quantize-dequantize (the MSB-plane
    read). Torch mirror of _quantize_rows_symmetric(x, 8) in
    run_joint_kv_budget_policy_eval.py: scale = absmax/127 clamped to
    1e-12, round-half-even codes."""
    scale_t = (x32_t.abs().amax(dim=1, keepdim=True) / 127.0).clamp_min_(1e-12)
    return torch.round(x32_t / scale_t) * scale_t


def _cached_precision_tier_value_error(
    *,
    module,
    kv_head: int,
    values32_t: torch.Tensor,
    values_lo_t: torch.Tensor,
) -> torch.Tensor:
    """Cache the row-local fp16 V commit error for append-only CUDA KV."""

    context_len = int(values32_t.shape[0])
    cache = getattr(module, "_pagedpq_precision_tier_row_cache", None)
    if not isinstance(cache, dict):
        cache = {}
        setattr(module, "_pagedpq_precision_tier_row_cache", cache)
    entry = cache.get(int(kv_head))
    dim = int(values32_t.shape[1])
    if (
        entry is None
        or int(entry.get("dim", -1)) != dim
        or str(entry.get("device", "")) != str(values32_t.device)
        or int(entry.get("filled", 0)) > context_len
    ):
        entry = None

    filled = int(entry.get("filled", 0)) if entry is not None else 0
    capacity = int(entry.get("capacity", 0)) if entry is not None else 0
    if entry is None or capacity < context_len:
        new_capacity = context_len + 256
        value_error16 = torch.empty((new_capacity,), dtype=torch.float16, device=values32_t.device)
        if entry is not None and filled > 0:
            value_error16[:filled].copy_(entry["value_error16"][:filled])
        entry = {
            "dim": dim,
            "device": str(values32_t.device),
            "capacity": int(new_capacity),
            "filled": int(filled),
            "value_error16": value_error16,
        }
        cache[int(kv_head)] = entry

    if context_len > filled:
        value_rows = values32_t[filled:context_len]
        value_lo_new = values_lo_t[filled:context_len]
        value_error16_new = (
            (value_rows - value_lo_new).pow(2).sum(dim=1, dtype=torch.float64).to(dtype=torch.float16)
        )
        entry["value_error16"][filled:context_len].copy_(value_error16_new)
        entry["filled"] = int(context_len)

    return entry["value_error16"][:context_len]


def _budget_index_at_least(budgets: list[int], target: float) -> int:
    for idx, budget in enumerate(budgets):
        if int(budget) >= float(target):
            return int(idx)
    return max(0, len(budgets) - 1)


def _fraction_suffix(name: str, marker: str, default: float) -> float:
    tail = str(name).split(str(marker), 1)[-1]
    token = tail.split("_", 1)[0].replace("p", ".")
    try:
        return float(token)
    except ValueError:
        return float(default)


def _softmax_prefix_counts_for_mass(sorted_logits: torch.Tensor, mass: float) -> torch.Tensor:
    if int(sorted_logits.shape[-1]) == 0:
        return torch.zeros((int(sorted_logits.shape[0]),), dtype=torch.long, device=sorted_logits.device)
    weights = torch.softmax(sorted_logits.to(dtype=torch.float32), dim=1)
    prefix = torch.cumsum(weights, dim=1)
    counts = torch.sum(prefix < float(mass), dim=1).to(dtype=torch.long) + 1
    return torch.clamp(counts, min=1, max=int(sorted_logits.shape[1]))


def _joint_start_indices_for_heads(
    *,
    strategy: str,
    context_len: int,
    dense_score_rows_t: torch.Tensor,
    sqrt_dim: float,
    k_budgets: list[int],
    v_budgets: list[int],
    sorted_dense_score_rows_t: torch.Tensor | None = None,
    budget_tensors: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[list[int] | torch.Tensor, list[int] | torch.Tensor]:
    name = str(strategy).strip().lower()
    group_heads = int(dense_score_rows_t.shape[0])
    if name in {"", "min", "zero"}:
        return [0 for _ in range(group_heads)], [0 for _ in range(group_heads)]
    if name.startswith("fixed_f"):
        frac = _fraction_suffix(name, "f", 0.05)
        k_target = max(float(k_budgets[0]), float(context_len) * float(frac))
        v_target = max(float(v_budgets[0]), k_target * 0.25)
        ki = _budget_index_at_least(k_budgets, k_target)
        vi = _budget_index_at_least(v_budgets, v_target)
        return [int(ki) for _ in range(group_heads)], [int(vi) for _ in range(group_heads)]
    if name.startswith("proxy_mass_m"):
        mass = min(max(_fraction_suffix(name, "m", 0.5), 0.0), 0.999999)
        sorted_logits = (
            sorted_dense_score_rows_t.to(dtype=torch.float32) / float(sqrt_dim)
            if sorted_dense_score_rows_t is not None
            else torch.sort(
                dense_score_rows_t.to(dtype=torch.float32) / float(sqrt_dim),
                dim=1,
                descending=True,
            ).values
        )
        counts_t = _softmax_prefix_counts_for_mass(sorted_logits, float(mass))
        if dense_score_rows_t.device.type == "cuda":
            # Keep the start state on device so it can share the policy's one
            # packed D2H transfer instead of synchronizing here.
            if budget_tensors is None:
                k_budgets_t = torch.as_tensor(k_budgets, dtype=torch.long, device=counts_t.device)
                v_budgets_t = torch.as_tensor(v_budgets, dtype=torch.float32, device=counts_t.device)
            else:
                k_budgets_t, v_budgets_t = budget_tensors
            k_targets_t = torch.maximum(counts_t, k_budgets_t[0])
            k_indices_t = torch.searchsorted(k_budgets_t, k_targets_t).clamp_max_(len(k_budgets) - 1)
            v_targets_t = torch.maximum(
                v_budgets_t[0],
                k_targets_t.to(dtype=torch.float32) * 0.25,
            )
            v_indices_t = torch.searchsorted(v_budgets_t, v_targets_t).clamp_max_(len(v_budgets) - 1)
            return k_indices_t, v_indices_t
        counts = counts_t.detach().cpu().tolist()
        k_indices: list[int] = []
        v_indices: list[int] = []
        for count in counts:
            k_target = max(float(k_budgets[0]), float(count))
            v_target = max(float(v_budgets[0]), k_target * 0.25)
            k_indices.append(_budget_index_at_least(k_budgets, k_target))
            v_indices.append(_budget_index_at_least(v_budgets, v_target))
        return k_indices, v_indices
    raise ValueError(f"unknown joint_kv_start_strategy: {strategy}")


def process_one_joint_kv_head(runtime, kv_head_i: int) -> bool:
    args = runtime.args
    self = runtime.self
    layer_id = runtime.layer_id
    stats = runtime.stats
    device = runtime.device
    q_all = runtime.q_all
    torch_k_cache = runtime.torch_k_cache
    torch_v_cache = runtime.torch_v_cache
    context_len_i = runtime.context_len_i
    num_heads = runtime.num_heads
    num_kv_heads = runtime.num_kv_heads
    group_size = runtime.group_size
    nprobes = runtime.nprobes
    key_bytes = runtime.key_bytes
    value_bytes = runtime.value_bytes
    local_qpos = runtime.local_qpos
    sqrt_dim = runtime.sqrt_dim
    prob_dtype = runtime.prob_dtype
    policy_id = runtime.policy_id
    policy_uses_mb = runtime.policy_uses_mb
    needs_logical_accounting = runtime.needs_logical_accounting
    needs_budget_mb_vectors = runtime.needs_budget_mb_vectors
    joint_k_budgets = runtime.joint_k_budgets
    joint_v_budgets = runtime.joint_v_budgets
    joint_v_budgets_t = runtime.joint_v_budgets_t
    allhead_indexes = runtime.allhead_indexes
    allhead_dense_pq_scores_t = runtime.allhead_dense_pq_scores_t
    allhead_selector_mb = runtime.allhead_selector_mb
    allhead_exact_scores_t = runtime.allhead_exact_scores_t
    allhead_selector_rank_prefix_t = runtime.allhead_selector_rank_prefix_t
    allhead_rank_prefix_cache = runtime.allhead_rank_prefix_cache
    use_unsorted_k_prefix = runtime.use_unsorted_k_prefix
    native_exact_logits_enabled = runtime.native_exact_logits_enabled
    native_full_exact_logits = runtime.native_full_exact_logits
    use_grouped_risk_prefix = runtime.use_grouped_risk_prefix
    grouped_output_workspace_enabled = runtime.grouped_output_workspace_enabled
    grouped_strided_output_workspace_enabled = runtime.grouped_strided_output_workspace_enabled
    grouped_score_workspace_enabled = runtime.grouped_score_workspace_enabled
    grouped_vpq_vhat_groups_t = runtime.grouped_vpq_vhat_groups_t
    grouped_vpq_residual_groups_t = runtime.grouped_vpq_residual_groups_t
    grouped_vpq_code_error_groups_t = runtime.grouped_vpq_code_error_groups_t
    compact_grouped_vpq_enabled = (
        runtime.grouped_vpq_value_codebooks_t is not None
        and runtime.grouped_vpq_value_codes_t is not None
        and runtime.grouped_vpq_value_page_starts_t is not None
        and runtime.grouped_vpq_values_t is not None
        and grouped_vpq_code_error_groups_t is not None
    )
    grouped_vpq_actual_subbits = runtime.grouped_vpq_actual_subbits
    grouped_risk_records = runtime.grouped_risk_records
    outputs_all = runtime.outputs_all
    prefix_index_for = runtime.prefix_index_for
    joint_vpq_sidecars_for = runtime.joint_vpq_sidecars_for
    joint_vpq_pack_and_fallback_for = runtime.joint_vpq_pack_and_fallback_for
    token_layout_for = runtime.token_layout_for
    nocalib_score_grid_workspace_for = runtime.nocalib_score_grid_workspace_for
    nocalib_scatter_score_grid_workspace_for = runtime.nocalib_scatter_score_grid_workspace_for
    score_grid_workspace_for = runtime.score_grid_workspace_for
    torch_score_grid_workspace_for = runtime.torch_score_grid_workspace_for
    grouped_score_grid_workspace_for = runtime.grouped_score_grid_workspace_for
    grouped_output_workspace_for = runtime.grouped_output_workspace_for
    softmax_base_workspace_for = runtime.softmax_base_workspace_for
    native_rank_prefix_tokens = runtime.native_rank_prefix_tokens
    wall_profile_enabled = runtime.wall_profile_enabled
    time_trace = getattr(self, "_pagedpq_joint_time_trace", None)
    grouped_probs_workspace_t: torch.Tensor | None = None
    grouped_base_workspace_t: torch.Tensor | None = None

    index = (
        allhead_indexes[int(kv_head_i)]
        if allhead_indexes is not None
        else prefix_index_for(int(kv_head_i), context_len_i)
    )
    if not index.pages:
        return False
    head_start_i = int(kv_head_i) * int(group_size)
    head_end_i = min(int(num_heads), head_start_i + int(group_size))
    if head_start_i >= head_end_i:
        return True
    group_heads_i = int(head_end_i - head_start_i)
    queries_h = q_all[head_start_i:head_end_i, int(local_qpos), :].to(
        device=device,
        dtype=torch.float32,
    )

    layout_wall_t0 = time.perf_counter() if wall_profile_enabled else 0.0
    if bool(getattr(args, "profile_native_ops", False)):
        _sync_if_cuda(device)
        layout_t0 = time.perf_counter()
    else:
        layout_t0 = 0.0
    indexed_tokens_t, base_t, nonbase_mask_t, layout_covers_context = token_layout_for(index)
    if bool(getattr(args, "profile_native_ops", False)):
        _sync_if_cuda(device)
        stats[layer_id].add_joint_detail_timing(
            layout_seconds=float(time.perf_counter() - layout_t0)
        )
    if wall_profile_enabled:
        stats[layer_id].add_joint_wall_timing(
            layout_seconds=float(time.perf_counter() - layout_wall_t0)
        )
    if int(indexed_tokens_t.numel()) == 0:
        return False

    if allhead_dense_pq_scores_t is not None and allhead_selector_mb is not None:
        dense_score_rows_t = allhead_dense_pq_scores_t[head_start_i:head_end_i]
        selector_mb = float(allhead_selector_mb)
    else:
        if str(getattr(args, "logit_buffer_format", "fp")) in ("e4m3", "absmax_int"):
            # Logit-buffer quantization + rank-prefix rebuild only happens on
            # the allhead precompute path; running the fallback would silently
            # rank on fp scores.
            raise RuntimeError(
                "logit_buffer_format="
                + str(getattr(args, "logit_buffer_format", "fp"))
                + " requires the allhead PQ score precompute path "
                "(SELECTOR_PQ_JOINT_ALLHEAD_PRECOMPUTE=1)"
            )
        selector_wall_t0 = time.perf_counter() if wall_profile_enabled else 0.0
        if bool(getattr(args, "profile_native_ops", False)):
            _sync_if_cuda(device)
            selector_t0 = time.perf_counter()
        else:
            selector_t0 = 0.0
        _ranked_rows_t, _ranked_score_rows_t, dense_score_rows_t, _seconds, selector_mb, _nprobe = (
            rank_paged_pq_batched_with_scores(
                queries_h.contiguous(),
                index,
                mode=str(args.selector_mode),
                selector_backend=str(args.selector_backend),
                nprobes=nprobes,
                budget=1,
                key_bytes=key_bytes,
                subbits=int(args.subbits),
            )
        )
        if bool(getattr(args, "profile_native_ops", False)):
            _sync_if_cuda(device)
            stats[layer_id].add_native_timing(selector_seconds=float(time.perf_counter() - selector_t0))
        if wall_profile_enabled:
            stats[layer_id].add_joint_wall_timing(
                selector_seconds=float(time.perf_counter() - selector_wall_t0)
            )

    dense_score_rows_t = dense_score_rows_t.to(device=device, dtype=torch.float32)
    indexed_count = min(int(indexed_tokens_t.numel()), int(dense_score_rows_t.shape[1]))
    indexed_tokens_t = indexed_tokens_t[:indexed_count]
    dense_score_rows_t = dense_score_rows_t[:, :indexed_count]
    token_to_indexed_pos_t: torch.Tensor | None = None
    if _env_truthy("SELECTOR_PQ_JOINT_FAST_AFFINE_SELECTED", "0"):
        token_to_indexed_pos_t = torch.full(
            (context_len_i,),
            -1,
            dtype=torch.long,
            device=device,
        )
        token_to_indexed_pos_t.index_copy_(
            0,
            indexed_tokens_t,
            torch.arange(int(indexed_tokens_t.numel()), dtype=torch.long, device=device),
        )

    if nonbase_mask_t is None:
        ranked_nonbase_t = indexed_tokens_t
        ranked_nonbase_scores_t = dense_score_rows_t
    else:
        ranked_nonbase_t = indexed_tokens_t[nonbase_mask_t]
        ranked_nonbase_scores_t = dense_score_rows_t[:, nonbase_mask_t]

    use_sparse_exact_score_grid = _env_truthy("SELECTOR_PQ_JOINT_SPARSE_EXACT_SCORE_GRID", "0")
    use_sparse_direct_score_grid = (
        use_sparse_exact_score_grid
        and _env_truthy("SELECTOR_PQ_JOINT_SPARSE_DIRECT_SCORE_GRID", "0")
    )
    exact_scores_h: torch.Tensor | None
    exact_keys32_t: torch.Tensor | None = None
    sparse_base_logits_t: torch.Tensor | None = None
    sparse_ranked_tokens_t: torch.Tensor | None = None
    sparse_ranked_logits_t: torch.Tensor | None = None
    if allhead_exact_scores_t is not None and int(allhead_exact_scores_t.shape[1]) >= context_len_i:
        exact_scores_h = allhead_exact_scores_t[head_start_i:head_end_i, :context_len_i]
    elif use_sparse_exact_score_grid:
        exact_scores_h = None
    else:
        exact_wall_t0 = time.perf_counter() if wall_profile_enabled else 0.0
        exact_trace_start = (
            time_trace.cuda_start("exact_logit") if time_trace is not None else None
        )
        if bool(getattr(args, "profile_native_ops", False)):
            _sync_if_cuda(device)
            exact_t0 = time.perf_counter()
        else:
            exact_t0 = 0.0
        if native_exact_logits_enabled:
            exact_scores_h = native_full_exact_logits(
                queries_h,
                torch_k_cache[int(kv_head_i)].unsqueeze(0),
                kv_head_i=int(kv_head_i),
            )
        else:
            keys_t = torch_k_cache[int(kv_head_i)][:context_len_i].to(device=device, dtype=torch.float32)
            exact_keys32_t = keys_t
            exact_scores_h = (queries_h @ keys_t.transpose(0, 1)) / sqrt_dim
        if bool(getattr(args, "profile_native_ops", False)):
            _sync_if_cuda(device)
            stats[layer_id].add_native_detail_timing(
                exact_logit_seconds=float(time.perf_counter() - exact_t0)
            )
        if wall_profile_enabled:
            stats[layer_id].add_joint_wall_timing(
                exact_logit_seconds=float(time.perf_counter() - exact_wall_t0)
            )
        if time_trace is not None:
            time_trace.cuda_end("exact_logit", exact_trace_start)

    values_t = torch_v_cache[int(kv_head_i)][:context_len_i]
    vsidecar_t0 = 0.0
    vsidecar_wall_t0 = time.perf_counter() if wall_profile_enabled else 0.0
    if bool(getattr(args, "profile_native_ops", False)):
        _sync_if_cuda(device)
        vsidecar_t0 = time.perf_counter()
    if compact_grouped_vpq_enabled and int(grouped_vpq_code_error_groups_t.shape[0]) > int(kv_head_i):
        vhat_all_t = torch.empty(
            (0, int(self.head_dim)),
            dtype=torch.float32,
            device=device,
        )
        residual_t = torch.empty(
            (0, int(self.head_dim)),
            dtype=torch.float32,
            device=device,
        )
        code_error_t = grouped_vpq_code_error_groups_t[int(kv_head_i)]
        actual_value_subbits_for_cost = (
            int(grouped_vpq_actual_subbits)
            if grouped_vpq_actual_subbits is not None
            else int(args.value_subbits)
            if int(args.value_subbits) > 0
            else int(args.subbits)
        )
    elif (
        grouped_vpq_vhat_groups_t is not None
        and grouped_vpq_residual_groups_t is not None
        and grouped_vpq_code_error_groups_t is not None
        and int(grouped_vpq_residual_groups_t.shape[0]) > int(kv_head_i)
    ):
        vhat_all_t = grouped_vpq_vhat_groups_t[int(kv_head_i)]
        residual_t = grouped_vpq_residual_groups_t[int(kv_head_i)]
        code_error_t = grouped_vpq_code_error_groups_t[int(kv_head_i)]
        if grouped_vpq_actual_subbits is not None:
            actual_value_subbits_for_cost = int(grouped_vpq_actual_subbits)
    else:
        vhat_all_t, residual_t, code_error_t, actual_value_subbits_for_cost = joint_vpq_sidecars_for(
            kv_head=int(kv_head_i),
            index=index,
            values_t=values_t,
            context_len_i=context_len_i,
        )
    if bool(getattr(args, "profile_native_ops", False)):
        _sync_if_cuda(device)
        stats[layer_id].add_native_detail_timing(
            output_seconds=float(time.perf_counter() - vsidecar_t0)
        )
    if wall_profile_enabled:
        stats[layer_id].add_joint_wall_timing(
            vpq_sidecar_seconds=float(time.perf_counter() - vsidecar_wall_t0)
        )

    selected_batch_by_take: dict[int, torch.Tensor] = {}
    ranked_nonbase_count = int(ranked_nonbase_t.numel())
    active_joint_k_budgets = joint_k_budgets
    collapse_duplicate_k_rows = _env_truthy("SELECTOR_PQ_JOINT_COLLAPSE_DUP_K_ROWS", "0")
    if collapse_duplicate_k_rows:
        collapsed_k_budgets: list[int] = []
        seen_take_counts: set[int] = set()
        for k_budget in joint_k_budgets:
            take_i = max(0, min(int(k_budget), ranked_nonbase_count))
            if int(take_i) in seen_take_counts:
                continue
            seen_take_counts.add(int(take_i))
            collapsed_k_budgets.append(int(k_budget))
        if collapsed_k_budgets:
            active_joint_k_budgets = collapsed_k_budgets
    skip_full_budget_sort = _env_truthy("SELECTOR_PQ_JOINT_SKIP_FULL_BUDGET_SORT", "0")
    avoid_full_budget_rank = bool(
        skip_full_budget_sort
        or (
            collapse_duplicate_k_rows
            and _env_truthy("SELECTOR_PQ_JOINT_EXACT_FULL_BUDGET_GRID", "1")
        )
    )
    if avoid_full_budget_rank:
        # A K budget that reaches every non-base token is a dense
        # exact-logit row.  The native score-grid path handles that
        # row from indexed tokens directly, so the rank-prefix path
        # only needs the largest partial budget.
        partial_rank_takes = [
            max(0, min(int(v), ranked_nonbase_count))
            for v in active_joint_k_budgets
            if max(0, min(int(v), ranked_nonbase_count)) < ranked_nonbase_count
        ]
        max_rank_take = max(partial_rank_takes, default=0)
    else:
        partial_rank_takes = [
            max(0, min(int(v), ranked_nonbase_count))
            for v in active_joint_k_budgets
        ]
        max_rank_take = max(
            0,
            min(max(int(v) for v in active_joint_k_budgets), ranked_nonbase_count),
        )
    partial_rank_takes_for_prefix = sorted({int(v) for v in partial_rank_takes if int(v) > 0})
    partial_rank_takes_t = (
        torch.as_tensor(partial_rank_takes_for_prefix, dtype=torch.long, device=device)
        if partial_rank_takes_for_prefix and _env_truthy("SELECTOR_PQ_JOINT_NATIVE_RANK_PREFIX", "0")
        else None
    )
    ranked_prefix_tokens_t: torch.Tensor | None = None
    sorted_ranked_score_rows_t: torch.Tensor | None = None
    if (
        max_rank_take > 0
        and _env_truthy("SELECTOR_PQ_JOINT_REUSE_MAX_TOPK", "1")
        and not use_unsorted_k_prefix
    ):
        use_allhead_rank_prefix = (
            _env_truthy("SELECTOR_PQ_JOINT_ALLHEAD_RANK_PREFIX", "0")
            and allhead_dense_pq_scores_t is not None
            and nonbase_mask_t is None
            and int(allhead_dense_pq_scores_t.shape[0]) >= int(num_heads)
            and int(allhead_dense_pq_scores_t.shape[1]) >= int(indexed_count)
        )
        if use_allhead_rank_prefix:
            rank_prefix_key = (
                int(allhead_dense_pq_scores_t.data_ptr()),
                int(indexed_tokens_t.data_ptr()),
                int(indexed_count),
                int(max_rank_take),
            )
            cached_rank_prefix = allhead_rank_prefix_cache.get(rank_prefix_key)
            if cached_rank_prefix is None:
                allhead_ranked_prefix_t = None
                allhead_sorted_scores_t = None
                if (
                    allhead_selector_rank_prefix_t is not None
                    and int(allhead_selector_rank_prefix_t.shape[0]) >= int(num_heads)
                    and int(allhead_selector_rank_prefix_t.shape[1]) >= int(max_rank_take)
                ):
                    allhead_ranked_prefix_t = allhead_selector_rank_prefix_t[:num_heads, : int(max_rank_take)]
                else:
                    rank_prefix_wall_t0 = time.perf_counter() if wall_profile_enabled else 0.0
                    if bool(getattr(args, "profile_native_ops", False)):
                        _sync_if_cuda(device)
                        rank_prefix_t0 = time.perf_counter()
                    else:
                        rank_prefix_t0 = 0.0
                    if _env_truthy("SELECTOR_PQ_JOINT_NATIVE_RANK_PREFIX", "0"):
                        allhead_ranked_prefix_t = native_rank_prefix_tokens(
                            allhead_dense_pq_scores_t[:num_heads, :indexed_count]
                            .to(dtype=torch.float32),
                            indexed_tokens_t,
                            int(max_rank_take),
                            partial_rank_takes_t,
                        )
                    else:
                        allhead_topk_t = torch.topk(
                            allhead_dense_pq_scores_t[:num_heads, :indexed_count],
                            k=int(max_rank_take),
                            dim=1,
                            largest=True,
                            sorted=True,
                        )
                        allhead_order_t = allhead_topk_t.indices
                        allhead_sorted_scores_t = allhead_topk_t.values
                        allhead_ranked_prefix_t = indexed_tokens_t.index_select(
                            0,
                            allhead_order_t.reshape(-1),
                        ).reshape(
                            int(num_heads),
                            int(max_rank_take),
                        )
                    if bool(getattr(args, "profile_native_ops", False)):
                        _sync_if_cuda(device)
                        stats[layer_id].add_joint_detail_timing(
                            rank_prefix_seconds=float(time.perf_counter() - rank_prefix_t0)
                        )
                    if wall_profile_enabled:
                        stats[layer_id].add_joint_wall_timing(
                            rank_prefix_seconds=float(time.perf_counter() - rank_prefix_wall_t0)
                        )
                cached_rank_prefix = (allhead_ranked_prefix_t, allhead_sorted_scores_t)
                allhead_rank_prefix_cache[rank_prefix_key] = cached_rank_prefix
            allhead_ranked_prefix_t, allhead_sorted_scores_t = cached_rank_prefix
            ranked_prefix_tokens_t = allhead_ranked_prefix_t[head_start_i:head_end_i]
            if (
                device.type == "cuda"
                and bool(getattr(args, "joint_kv_precision_tiers", False))
                and allhead_sorted_scores_t is not None
                and nonbase_mask_t is None
                and int(allhead_sorted_scores_t.shape[1]) >= ranked_nonbase_count
            ):
                sorted_ranked_score_rows_t = allhead_sorted_scores_t[
                    head_start_i:head_end_i, :ranked_nonbase_count
                ]
            if _env_truthy("SELECTOR_PQ_JOINT_ALLHEAD_RANK_AUDIT", "0"):
                audit_order_t = torch.topk(
                    ranked_nonbase_scores_t,
                    k=int(max_rank_take),
                    dim=1,
                    largest=True,
                    sorted=True,
                ).indices
                audit_ranked_prefix_t = ranked_nonbase_t.index_select(
                    0,
                    audit_order_t.reshape(-1),
                ).reshape(
                    group_heads_i,
                    int(max_rank_take),
                )
                if not torch.equal(ranked_prefix_tokens_t, audit_ranked_prefix_t):
                    mismatch_t = ranked_prefix_tokens_t.ne(audit_ranked_prefix_t)
                    mismatch_count = int(mismatch_t.sum().item())
                    first = int(mismatch_t.flatten().nonzero()[0].item())
                    raise RuntimeError(
                        "all-head rank-prefix audit failed: "
                        f"{mismatch_count} mismatched token positions; first_flat={first}"
                    )
        else:
            rank_prefix_wall_t0 = time.perf_counter() if wall_profile_enabled else 0.0
            if bool(getattr(args, "profile_native_ops", False)):
                _sync_if_cuda(device)
                rank_prefix_t0 = time.perf_counter()
            else:
                rank_prefix_t0 = 0.0
            if _env_truthy("SELECTOR_PQ_JOINT_NATIVE_RANK_PREFIX", "0"):
                ranked_prefix_tokens_t = native_rank_prefix_tokens(
                    ranked_nonbase_scores_t,
                    ranked_nonbase_t,
                    int(max_rank_take),
                    partial_rank_takes_t,
                )
            else:
                max_order_t = torch.topk(
                    ranked_nonbase_scores_t,
                    k=int(max_rank_take),
                    dim=1,
                    largest=True,
                    sorted=True,
                ).indices
                ranked_prefix_tokens_t = ranked_nonbase_t.index_select(0, max_order_t.reshape(-1)).reshape(
                    group_heads_i,
                    int(max_rank_take),
                )
            if bool(getattr(args, "profile_native_ops", False)):
                _sync_if_cuda(device)
                stats[layer_id].add_joint_detail_timing(
                    rank_prefix_seconds=float(time.perf_counter() - rank_prefix_t0)
                )
            if wall_profile_enabled:
                stats[layer_id].add_joint_wall_timing(
                    rank_prefix_seconds=float(time.perf_counter() - rank_prefix_wall_t0)
                )

    if exact_scores_h is None:
        full_exact_row_requested = any(
            max(0, min(int(k_budget), ranked_nonbase_count)) > int(max_rank_take)
            for k_budget in active_joint_k_budgets
        )
        exact_wall_t0 = time.perf_counter() if wall_profile_enabled else 0.0
        exact_trace_start = (
            time_trace.cuda_start("exact_logit") if time_trace is not None else None
        )
        if bool(getattr(args, "profile_native_ops", False)):
            _sync_if_cuda(device)
            exact_t0 = time.perf_counter()
        else:
            exact_t0 = 0.0
        if full_exact_row_requested:
            if native_exact_logits_enabled:
                exact_scores_h = native_full_exact_logits(
                    queries_h,
                    torch_k_cache[int(kv_head_i)].unsqueeze(0),
                    kv_head_i=int(kv_head_i),
                )
            else:
                keys_t = torch_k_cache[int(kv_head_i)][:context_len_i].to(device=device, dtype=torch.float32)
                exact_scores_h = (queries_h @ keys_t.transpose(0, 1)) / sqrt_dim
        else:
            if not _env_truthy("SELECTOR_PQ_JOINT_NATIVE_SCORE_GRID", "0"):
                raise RuntimeError(
                    "SELECTOR_PQ_JOINT_SPARSE_EXACT_SCORE_GRID requires native score-grid mode"
                )
            if not _env_truthy("SELECTOR_PQ_JOINT_SCORE_GRID_NO_EXACT_FILL", "0"):
                raise RuntimeError(
                    "SELECTOR_PQ_JOINT_SPARSE_EXACT_SCORE_GRID requires "
                    "SELECTOR_PQ_JOINT_SCORE_GRID_NO_EXACT_FILL=1"
                )
            if not _env_truthy("SELECTOR_PQ_JOINT_TOKENFIT_SCORE_GRID", "0"):
                raise RuntimeError(
                    "SELECTOR_PQ_JOINT_SPARSE_EXACT_SCORE_GRID requires tokenfit score-grid mode"
                )
            if not bool(layout_covers_context):
                raise RuntimeError(
                    "SELECTOR_PQ_JOINT_SPARSE_EXACT_SCORE_GRID requires indexed tokens plus base "
                    "tokens to cover the full context"
                )
            native = load_selector_paged_pq_ext()
            if not (
                hasattr(native, "gqa_decode_token_exact_logits")
                and hasattr(native, "joint_sparse_exact_score_table")
            ):
                raise RuntimeError(
                    "SELECTOR_PQ_JOINT_SPARSE_EXACT_SCORE_GRID requires sparse exact-logit CUDA helpers"
                )
            base_rows_t = (
                base_t.reshape(1, -1).expand(group_heads_i, -1)
                if int(base_t.numel()) > 0
                else torch.empty((group_heads_i, 0), dtype=torch.long, device=device)
            )
            ranked_for_sparse_t = (
                ranked_prefix_tokens_t
                if ranked_prefix_tokens_t is not None
                else torch.empty((group_heads_i, 0), dtype=torch.long, device=device)
            )
            key_one_t = torch_k_cache[int(kv_head_i)].unsqueeze(0)
            combined_sparse_tokens_t = torch.cat(
                (
                    base_rows_t.to(dtype=torch.long),
                    ranked_for_sparse_t.to(dtype=torch.long),
                ),
                dim=1,
            ).contiguous()
            combined_sparse_logits_t = native.gqa_decode_token_exact_logits(
                queries_h.contiguous(),
                key_one_t,
                combined_sparse_tokens_t,
                int(group_size),
                int(context_len_i),
                float(1.0 / sqrt_dim),
            )
            base_count_for_sparse_i = int(base_rows_t.shape[1])
            base_logits_t = combined_sparse_logits_t[:, :base_count_for_sparse_i].contiguous()
            ranked_logits_t = combined_sparse_logits_t[:, base_count_for_sparse_i:].contiguous()
            if use_sparse_direct_score_grid:
                sparse_base_logits_t = base_logits_t.contiguous()
                sparse_ranked_tokens_t = ranked_for_sparse_t.to(dtype=torch.long).contiguous()
                sparse_ranked_logits_t = ranked_logits_t.contiguous()
                exact_scores_h = None
            else:
                exact_scores_h = native.joint_sparse_exact_score_table(
                    base_t.to(dtype=torch.long).contiguous(),
                    base_logits_t.contiguous(),
                    ranked_for_sparse_t.to(dtype=torch.long).contiguous(),
                    ranked_logits_t.contiguous(),
                    int(context_len_i),
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
        if time_trace is not None:
            time_trace.cuda_end("exact_logit", exact_trace_start)

    if exact_scores_h is None and not (
        use_sparse_direct_score_grid
        and sparse_base_logits_t is not None
        and sparse_ranked_tokens_t is not None
        and sparse_ranked_logits_t is not None
    ):
        raise RuntimeError("missing exact-score source for joint K/V score-grid construction")
    exact_scores_prob_t = exact_scores_h.to(dtype=prob_dtype) if exact_scores_h is not None else None
    native_pq_scale_in_kernel = (
        _env_truthy("SELECTOR_PQ_JOINT_NATIVE_PQ_SCALE_IN_KERNEL", "0")
        and prob_dtype == torch.float32
    )
    pq_logit_scale = float(1.0 / sqrt_dim) if native_pq_scale_in_kernel else 1.0
    pq_logits_t = (
        dense_score_rows_t.to(dtype=prob_dtype)
        if native_pq_scale_in_kernel
        else dense_score_rows_t.to(dtype=prob_dtype) / sqrt_dim
    )
    y_indexed_prob_t = (
        exact_scores_h.index_select(1, indexed_tokens_t).to(prob_dtype)
        if str(args.tail_score_calibration) == "affine_selected" and exact_scores_h is not None
        else None
    )

    def y_indexed_for_native_score_grid() -> torch.Tensor:
        if y_indexed_prob_t is not None:
            return y_indexed_prob_t.to(dtype=torch.float32)
        # Native score-grid wrappers require a shape-compatible y_indexed
        # tensor even when raw/no-calibration mode ignores it.
        return pq_logits_t.to(dtype=torch.float32)

    def selected_for_budget_batch(k_budget: int) -> torch.Tensor:
        take = max(0, min(int(k_budget), int(ranked_nonbase_t.numel())))
        cached_selected = selected_batch_by_take.get(int(take))
        if cached_selected is not None:
            return cached_selected
        if avoid_full_budget_rank and take >= ranked_nonbase_count and ranked_nonbase_count > 0:
            add_t = ranked_nonbase_t.reshape(1, -1).expand(group_heads_i, -1)
        elif take > 0 and ranked_prefix_tokens_t is not None and take <= int(ranked_prefix_tokens_t.shape[1]):
            add_t = ranked_prefix_tokens_t[:, : int(take)]
        elif take > 0:
            rank_prefix_wall_t0 = time.perf_counter() if wall_profile_enabled else 0.0
            if bool(getattr(args, "profile_native_ops", False)):
                _sync_if_cuda(device)
                rank_prefix_t0 = time.perf_counter()
            else:
                rank_prefix_t0 = 0.0
            order_t = torch.topk(
                ranked_nonbase_scores_t,
                k=int(take),
                dim=1,
                largest=True,
                sorted=not use_unsorted_k_prefix,
            ).indices
            add_t = ranked_nonbase_t.index_select(0, order_t.reshape(-1)).reshape(group_heads_i, take)
            if bool(getattr(args, "profile_native_ops", False)):
                _sync_if_cuda(device)
                stats[layer_id].add_joint_detail_timing(
                    rank_prefix_seconds=float(time.perf_counter() - rank_prefix_t0)
                )
            if wall_profile_enabled:
                stats[layer_id].add_joint_wall_timing(
                    rank_prefix_seconds=float(time.perf_counter() - rank_prefix_wall_t0)
                )
        else:
            add_t = torch.empty((group_heads_i, 0), dtype=torch.long, device=device)
        if int(base_t.numel()) == 0:
            selected_out = add_t
            selected_batch_by_take[int(take)] = selected_out
            return selected_out
        base_rows_t = base_t.reshape(1, -1).expand(group_heads_i, -1)
        if int(add_t.numel()) == 0:
            selected_out = base_rows_t
            selected_batch_by_take[int(take)] = selected_out
            return selected_out
        selected_out = torch.cat((base_rows_t, add_t), dim=1)
        selected_batch_by_take[int(take)] = selected_out
        return selected_out

    def mixed_scores_for_selected_batch(
        selected_t_i: torch.Tensor,
        out_t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        selected_t_i = selected_t_i.to(device=device, dtype=torch.long)
        selected_t_i = torch.clamp(selected_t_i, min=0, max=max(0, context_len_i - 1))
        if (
            _env_truthy("SELECTOR_PQ_JOINT_NATIVE_SCORE_GRID", "0")
            and not use_unsorted_k_prefix
            and selected_t_i.ndim == 2
            and prob_dtype == torch.float32
        ):
            if exact_scores_h is None:
                raise RuntimeError(
                    "sparse-direct exact logits are only supported by the grid-artifact native score-grid path"
                )
            native = load_selector_paged_pq_ext()
            take_i = max(0, int(selected_t_i.shape[1]) - int(base_t.numel()))
            if (
                _env_truthy("SELECTOR_PQ_JOINT_EXACT_FULL_BUDGET_GRID", "1")
                and take_i >= ranked_nonbase_count
            ):
                return exact_scores_h.to(dtype=prob_dtype)
            if ranked_prefix_tokens_t is None:
                ranked_prefix_tokens_for_grid_t = torch.empty(
                    (group_heads_i, 0),
                    dtype=torch.long,
                    device=device,
                )
            else:
                ranked_prefix_tokens_for_grid_t = ranked_prefix_tokens_t
            y_for_grid_t = (
                y_indexed_for_native_score_grid()
            )
            if _env_truthy("SELECTOR_PQ_JOINT_TOKENFIT_SCORE_GRID", "0"):
                if not hasattr(native, "joint_mixed_score_grid_tokenfit_scaled"):
                    raise RuntimeError(
                        "SELECTOR_PQ_JOINT_TOKENFIT_SCORE_GRID requires joint_mixed_score_grid_tokenfit_scaled"
                    )
                score_grid_one_t = native.joint_mixed_score_grid_tokenfit_scaled(
                    exact_scores_h.to(dtype=torch.float32).contiguous(),
                    pq_logits_t.to(dtype=torch.float32).contiguous(),
                    y_for_grid_t.contiguous(),
                    indexed_tokens_t.to(dtype=torch.long).contiguous(),
                    base_t.to(dtype=torch.long).contiguous(),
                    ranked_prefix_tokens_for_grid_t.to(dtype=torch.long).contiguous(),
                    torch.as_tensor([int(take_i)], dtype=torch.long, device=device),
                    bool(str(args.tail_score_calibration) == "affine_selected"),
                    float(pq_logit_scale),
                )
            elif native_pq_scale_in_kernel:
                if not hasattr(native, "joint_mixed_score_grid_scaled"):
                    raise RuntimeError(
                        "SELECTOR_PQ_JOINT_NATIVE_PQ_SCALE_IN_KERNEL requires joint_mixed_score_grid_scaled"
                    )
                score_grid_one_t = native.joint_mixed_score_grid_scaled(
                    exact_scores_h.to(dtype=torch.float32).contiguous(),
                    pq_logits_t.to(dtype=torch.float32).contiguous(),
                    y_for_grid_t.contiguous(),
                    indexed_tokens_t.to(dtype=torch.long).contiguous(),
                    base_t.to(dtype=torch.long).contiguous(),
                    ranked_prefix_tokens_for_grid_t.to(dtype=torch.long).contiguous(),
                    torch.as_tensor([int(take_i)], dtype=torch.long, device=device),
                    bool(str(args.tail_score_calibration) == "affine_selected"),
                    float(pq_logit_scale),
                )
            else:
                score_grid_one_t = native.joint_mixed_score_grid(
                    exact_scores_h.to(dtype=torch.float32).contiguous(),
                    pq_logits_t.to(dtype=torch.float32).contiguous(),
                    y_for_grid_t.contiguous(),
                    indexed_tokens_t.to(dtype=torch.long).contiguous(),
                    base_t.to(dtype=torch.long).contiguous(),
                    ranked_prefix_tokens_for_grid_t.to(dtype=torch.long).contiguous(),
                    torch.as_tensor([int(take_i)], dtype=torch.long, device=device),
                    bool(str(args.tail_score_calibration) == "affine_selected"),
                )
            return score_grid_one_t[0].to(dtype=prob_dtype)
        if exact_scores_prob_t is None:
            raise RuntimeError("missing exact score table for non-native mixed-score path")
        if out_t is None:
            score_vec = exact_scores_prob_t.clone()
        else:
            score_vec = out_t
            score_vec.copy_(exact_scores_prob_t)
        pq_logits = pq_logits_t if pq_logit_scale == 1.0 else pq_logits_t * pq_logit_scale
        if str(args.tail_score_calibration) == "affine_selected":
            if y_indexed_prob_t is None:
                raise RuntimeError("missing indexed exact logits for affine selected calibration")
            y_indexed_t = y_indexed_prob_t
            if _env_truthy("SELECTOR_PQ_JOINT_FAST_AFFINE_SELECTED", "0"):
                if token_to_indexed_pos_t is None:
                    raise RuntimeError("missing token_to_indexed_pos_t for fast affine selected calibration")
                selected_index_pos_t = token_to_indexed_pos_t.index_select(
                    0,
                    selected_t_i.reshape(-1),
                ).reshape_as(selected_t_i)
                selected_index_valid_t = selected_index_pos_t >= 0
                selected_index_pos_safe_t = torch.clamp(selected_index_pos_t, min=0)
                mask_f = selected_index_valid_t.to(dtype=prob_dtype)
                counts_t = torch.sum(mask_f, dim=1)
                safe_counts_t = torch.clamp_min(counts_t, 1.0)
                x_selected_t = torch.gather(pq_logits, 1, selected_index_pos_safe_t)
                y_selected_t = torch.gather(y_indexed_t, 1, selected_index_pos_safe_t)
                x_sum_t = torch.sum(x_selected_t * mask_f, dim=1)
                y_sum_t = torch.sum(y_selected_t * mask_f, dim=1)
                x_mean_t = x_sum_t / safe_counts_t
                y_mean_t = y_sum_t / safe_counts_t
                x_centered_selected_t = (x_selected_t - x_mean_t.reshape(-1, 1)) * mask_f
                y_centered_selected_t = (y_selected_t - y_mean_t.reshape(-1, 1)) * mask_f
                x_var_t = torch.sum(x_centered_selected_t * x_centered_selected_t, dim=1) / safe_counts_t
                cov_t = torch.sum(x_centered_selected_t * y_centered_selected_t, dim=1) / safe_counts_t
            else:
                selected_index_mask = torch.zeros(
                    (group_heads_i, context_len_i),
                    dtype=torch.bool,
                    device=device,
                )
                if int(selected_t_i.numel()) > 0:
                    selected_index_mask.scatter_(1, selected_t_i, True)
                selected_index_mask = selected_index_mask.index_select(1, indexed_tokens_t)
                mask_f = selected_index_mask.to(dtype=prob_dtype)
                counts_t = torch.sum(mask_f, dim=1)
                safe_counts_t = torch.clamp_min(counts_t, 1.0)
                x_mean_t = torch.sum(mask_f * pq_logits, dim=1) / safe_counts_t
                y_mean_t = torch.sum(mask_f * y_indexed_t, dim=1) / safe_counts_t
                x_centered_t = (pq_logits - x_mean_t.reshape(-1, 1)) * mask_f
                y_centered_t = (y_indexed_t - y_mean_t.reshape(-1, 1)) * mask_f
                x_var_t = torch.sum(x_centered_t * x_centered_t, dim=1) / safe_counts_t
                cov_t = torch.sum(x_centered_t * y_centered_t, dim=1) / safe_counts_t
            fitted_scale_t = cov_t / torch.clamp_min(x_var_t, 1e-20)
            fitted_bias_t = y_mean_t - fitted_scale_t * x_mean_t
            fit_valid_t = (
                (counts_t >= 2.0)
                & (x_var_t > 1e-20)
                & torch.isfinite(fitted_scale_t)
                & (fitted_scale_t > 0.0)
            )
            zero_var_t = (counts_t >= 2.0) & (x_var_t <= 1e-20)
            scale_t = torch.where(
                zero_var_t,
                torch.zeros_like(fitted_scale_t),
                torch.where(fit_valid_t, fitted_scale_t, torch.ones_like(fitted_scale_t)),
            )
            bias_t = torch.where(
                zero_var_t,
                y_mean_t,
                torch.where(fit_valid_t, fitted_bias_t, torch.zeros_like(fitted_bias_t)),
            )
            calibrated_scores_t = scale_t.reshape(-1, 1) * pq_logits + bias_t.reshape(-1, 1)
        else:
            calibrated_scores_t = pq_logits
        score_vec[:, indexed_tokens_t] = calibrated_scores_t
        if int(selected_t_i.numel()) > 0:
            exact_selected_scores_t = exact_scores_h.gather(1, selected_t_i).to(prob_dtype)
            score_vec.scatter_(1, selected_t_i, exact_selected_scores_t)
        if scores_lo_h_t is not None:
            # Frozen K lo tier: ranked-prefix tokens beyond the hi-precision
            # fraction (base excluded) read the int8-plane logit instead of
            # the exact one. Mirrors _precision_lo_tokens + the
            # score_vec[lo_tokens] = scores_lo_np substitution in
            # run_joint_kv_budget_policy_eval.py.
            selected_len_local = (
                int(selected_t_i.shape[1]) if selected_t_i.ndim == 2 else int(selected_t_i.numel())
            )
            take_local = max(0, selected_len_local - int(base_t.numel()))
            if take_local > 0:
                hi_local = int(math.ceil(float(take_local) * _FROZEN_PRECISION_K_HI_FRAC))
                if hi_local < take_local:
                    if (
                        ranked_prefix_tokens_t is None
                        or int(ranked_prefix_tokens_t.shape[1]) < take_local
                    ):
                        raise RuntimeError(
                            "joint_kv_precision_tiers requires the sorted ranked-prefix "
                            "token table covering every K budget"
                        )
                    lo_pos_t = ranked_prefix_tokens_t[:, hi_local:take_local]
                    score_vec.scatter_(
                        1,
                        lo_pos_t,
                        scores_lo_h_t.gather(1, lo_pos_t).to(prob_dtype),
                    )
        return score_vec

    def mixed_probs_for_selected_batch(selected_t_i: torch.Tensor) -> torch.Tensor:
        return torch.softmax(mixed_scores_for_selected_batch(selected_t_i), dim=1)

    value_cost = joint_value_cost_for(
        args=args,
        index=index,
        context_len=int(context_len_i),
        head_dim=int(self.head_dim),
        value_bytes=int(value_bytes),
        joint_v_budgets=joint_v_budgets,
        needs_budget_mb_vectors=bool(needs_budget_mb_vectors),
        actual_value_subbits_for_cost=int(actual_value_subbits_for_cost),
    )
    actual_value_subbits = int(value_cost.actual_value_subbits)
    actual_value_subvecs = int(value_cost.actual_value_subvecs)
    code_bytes = int(value_cost.code_bytes)
    metadata_mb = float(value_cost.metadata_mb)
    v_pq_codebook_mb = float(value_cost.v_pq_codebook_mb)
    v_mb_by_idx = value_cost.v_mb_by_idx
    max_exact_v_count = int(value_cost.max_exact_v_count)
    use_ondemand_v_prefix = _env_truthy("SELECTOR_PQ_JOINT_ONDEMAND_V_PREFIX", "0")
    precision_tiers_enabled = bool(getattr(args, "joint_kv_precision_tiers", False))
    scores_lo_h_t: torch.Tensor | None = None
    residual_lo_commit_t: torch.Tensor | None = None
    v_commit_mask_t: torch.Tensor | None = None
    if precision_tiers_enabled:
        commit_trace_start = time_trace.cuda_start("commit") if time_trace is not None else None
        # The frozen tiers are implemented on the canonical torch grid
        # path only; fail loudly instead of silently running the old
        # single-tier composition on an optimized path.
        unsupported = [
            name
            for name, active in (
                ("SELECTOR_PQ_JOINT_GROUPED_RISK_PREFIX", use_grouped_risk_prefix),
                ("SELECTOR_PQ_JOINT_COMPACT_VPQ_RISK_PREFIX", compact_grouped_vpq_enabled),
                ("SELECTOR_PQ_JOINT_ONDEMAND_V_PREFIX", use_ondemand_v_prefix),
                ("SELECTOR_PQ_JOINT_UNSORTED_K_PREFIX", use_unsorted_k_prefix),
                (
                    "SELECTOR_PQ_JOINT_NATIVE_SCORE_GRID",
                    _env_truthy("SELECTOR_PQ_JOINT_NATIVE_SCORE_GRID", "0"),
                ),
                (
                    "SELECTOR_PQ_JOINT_GRID_ARTIFACTS=0",
                    not _env_truthy("SELECTOR_PQ_JOINT_GRID_ARTIFACTS", "1"),
                ),
            )
            if active
        ]
        if unsupported:
            raise RuntimeError(
                "joint_kv_precision_tiers requires the canonical torch grid path; "
                f"incompatible with: {', '.join(unsupported)}"
            )
        if int(residual_t.shape[0]) < context_len_i or int(vhat_all_t.shape[0]) < context_len_i:
            raise RuntimeError(
                "joint_kv_precision_tiers requires full V-PQ vhat/residual sidecars"
            )
        # K lo tier: per-row absmax-int8 QDQ of the cached keys, one extra
        # (ctx, dim) pass + one GEMM per head group per decode step.  The fp16
        # V commit error is an append-only row artifact on CUDA; cache old rows
        # and compute only the newly appended token without retaining another
        # context-by-head_dim plane.
        if device.type == "cuda":
            keys32_t = (
                exact_keys32_t
                if exact_keys32_t is not None
                else torch_k_cache[int(kv_head_i)][:context_len_i].to(
                    device=device, dtype=torch.float32
                )
            )
            values32_t = values_t.to(device=device, dtype=torch.float32)
            keys_lo_t = _rowwise_int8_qdq(keys32_t)
            values_lo_t = _rowwise_int8_qdq(values32_t)
            int8_err16_t = _cached_precision_tier_value_error(
                module=self,
                kv_head=int(kv_head_i),
                values32_t=values32_t,
                values_lo_t=values_lo_t,
            )
            v_commit_mask_t = int8_err16_t < code_error_t.to(dtype=torch.float16)
            scores_lo_h_t = (queries_h @ keys_lo_t.transpose(0, 1)) / sqrt_dim
            del keys32_t, keys_lo_t, values32_t
        else:
            # Keep the blessed CPU operation sequence unchanged byte-for-byte.
            keys32_t = torch_k_cache[int(kv_head_i)][:context_len_i].to(
                device=device, dtype=torch.float32
            )
            scores_lo_h_t = (queries_h @ _rowwise_int8_qdq(keys32_t).transpose(0, 1)) / sqrt_dim
            del keys32_t
            values32_t = values_t.to(device=device, dtype=torch.float32)
            values_lo_t = _rowwise_int8_qdq(values32_t)
            int8_err_t = (values32_t - values_lo_t).pow(2).sum(dim=1, dtype=torch.float64)
            del values32_t
            v_commit_mask_t = int8_err_t.to(dtype=torch.float16) < code_error_t.to(dtype=torch.float16)
            del int8_err_t
        # V lo tier: int8-plane values, per-token commit test against the
        # V-PQ code-error stat in the 2-byte sidecar domain (fp16 int8-error
        # vs fp16 code-error, matching the CPU reference compare), and the
        # committed lo residual pre-zeroed on failed commits so the V grid
        # can fold it into one cumsum.
        residual_lo_commit_t = torch.where(
            v_commit_mask_t.reshape(-1, 1),
            values_lo_t - vhat_all_t.to(dtype=torch.float32),
            torch.zeros((), dtype=torch.float32, device=device),
        )
        del values_lo_t
        if time_trace is not None:
            time_trace.cuda_end("commit", commit_trace_start)
    k_core_by_idx: dict[
        int,
        tuple[torch.Tensor, float, torch.Tensor, torch.Tensor, torch.Tensor],
    ] = {}
    k_artifacts_by_idx: dict[int, tuple[torch.Tensor, float, torch.Tensor, torch.Tensor | None]] = {}
    k_artifacts_by_selected_len: dict[int, tuple[torch.Tensor, float, torch.Tensor, torch.Tensor | None]] = {}
    outputs_by_budget: dict[tuple[int, int], torch.Tensor] = {}
    v_outputs_by_count: dict[tuple[int, int], torch.Tensor] = {}
    native_v_grid_by_ki: dict[int, torch.Tensor] = {}
    if try_process_fused_mixed_policy(
        JointFusedPolicyRuntime(
            args=args,
            module=self,
            layer_id=int(layer_id),
            stats=stats,
            device=device,
            outputs_all=outputs_all,
            head_start=int(head_start_i),
            head_end=int(head_end_i),
            group_heads=int(group_heads_i),
            context_len=int(context_len_i),
            prob_dtype=prob_dtype,
            policy_id=int(policy_id),
            policy_uses_mb=bool(policy_uses_mb),
            use_grouped_risk_prefix=bool(use_grouped_risk_prefix),
            use_unsorted_k_prefix=bool(use_unsorted_k_prefix),
            active_k_budgets=active_joint_k_budgets,
            ranked_nonbase_count=int(ranked_nonbase_t.numel()),
            base_t=base_t,
            ranked_prefix_tokens_t=ranked_prefix_tokens_t,
            exact_scores_h=exact_scores_h,
            pq_logits_t=pq_logits_t,
            y_indexed_prob_t=y_indexed_prob_t,
            indexed_tokens_t=indexed_tokens_t,
            vhat_all_t=vhat_all_t,
            residual_t=residual_t,
            code_error_t=code_error_t,
            joint_v_budgets_t=joint_v_budgets_t,
            key_bytes=int(key_bytes),
            value_bytes=int(value_bytes),
            selector_mb=float(selector_mb),
            v_pq_codebook_mb=float(v_pq_codebook_mb),
            metadata_mb=float(metadata_mb),
            actual_value_subvecs=int(actual_value_subvecs),
            code_bytes=int(code_bytes),
            wall_profile_enabled=bool(wall_profile_enabled),
            pq_logit_scale=float(pq_logit_scale),
        )
    ):
        return True

    def k_core_batch(
        ki_i: int,
    ) -> tuple[torch.Tensor, float, torch.Tensor, torch.Tensor, torch.Tensor]:
        cached = k_core_by_idx.get(int(ki_i))
        if cached is not None:
            return cached
        selected_t_i = selected_for_budget_batch(int(active_joint_k_budgets[int(ki_i)]))
        selected_len_i = int(selected_t_i.shape[1]) if selected_t_i.ndim == 2 else int(selected_t_i.numel())
        probs_t = mixed_probs_for_selected_batch(selected_t_i)
        exact_key_mb_i = float(selected_len_i * int(self.head_dim) * key_bytes) / MB
        risk_t = (probs_t * probs_t) * code_error_t.to(dtype=prob_dtype).reshape(1, -1)
        base_output_t = probs_t.to(torch.float32) @ vhat_all_t.float()
        out = (
            selected_t_i,
            float(selector_mb) + exact_key_mb_i,
            probs_t,
            base_output_t,
            risk_t,
        )
        k_core_by_idx[int(ki_i)] = out
        return out

    def k_artifacts_batch(ki_i: int) -> tuple[torch.Tensor, float, torch.Tensor, torch.Tensor | None]:
        if precision_tiers_enabled:
            raise RuntimeError(
                "joint_kv_precision_tiers is only implemented for the grid V-prefix "
                "path; the lazy k_artifacts V composition is single-tier"
            )
        cached = k_artifacts_by_idx.get(int(ki_i))
        if cached is not None:
            return cached
        if use_ondemand_v_prefix:
            selected_t_i, k_mb_i, _probs_t, base_output_t, _risk_t = k_core_batch(int(ki_i))
            out = (selected_t_i, float(k_mb_i), base_output_t, None)
            k_artifacts_by_idx[int(ki_i)] = out
            return out
        selected_t_i = selected_for_budget_batch(int(active_joint_k_budgets[int(ki_i)]))
        selected_len_i = int(selected_t_i.shape[1]) if selected_t_i.ndim == 2 else int(selected_t_i.numel())
        cached_by_len = k_artifacts_by_selected_len.get(selected_len_i)
        if cached_by_len is not None:
            k_artifacts_by_idx[int(ki_i)] = cached_by_len
            return cached_by_len
        probs_t = mixed_probs_for_selected_batch(selected_t_i)
        exact_key_mb_i = float(selected_len_i * int(self.head_dim) * key_bytes) / MB
        risk_t = (probs_t * probs_t) * code_error_t.to(dtype=prob_dtype).reshape(1, -1)
        base_output_t = probs_t.to(torch.float32) @ vhat_all_t.float()
        prefix_delta_t: torch.Tensor | None = None
        if int(max_exact_v_count) > 0:
            if int(max_exact_v_count) >= context_len_i:
                exact_order_t = torch.argsort(risk_t, dim=1, descending=True, stable=True)
            else:
                exact_order_t = torch.topk(
                    risk_t,
                    k=int(max_exact_v_count),
                    dim=1,
                    largest=True,
                    sorted=True,
                ).indices
            gathered_probs_t = torch.gather(probs_t.to(torch.float32), 1, exact_order_t)
            gathered_residual_t = residual_t.index_select(0, exact_order_t.reshape(-1)).reshape(
                group_heads_i,
                int(exact_order_t.shape[1]),
                int(self.head_dim),
            )
            prefix_delta_t = torch.cumsum(
                gathered_probs_t.reshape(group_heads_i, -1, 1) * gathered_residual_t.float(),
                dim=1,
            )
        out = (selected_t_i, float(selector_mb) + exact_key_mb_i, base_output_t, prefix_delta_t)
        k_artifacts_by_idx[int(ki_i)] = out
        k_artifacts_by_selected_len[selected_len_i] = out
        return out

    def output_for_budget_batch(ki_i: int, vi_i: int) -> torch.Tensor:
        if precision_tiers_enabled:
            raise RuntimeError(
                "joint_kv_precision_tiers is only implemented for the grid V-prefix "
                "path; the lazy output_for_budget V composition is single-tier"
            )
        key = (int(ki_i), int(vi_i))
        cached = outputs_by_budget.get(key)
        if cached is not None:
            return cached
        if use_ondemand_v_prefix:
            _selected_t_i, _k_mb_i, probs_t, base_output_t, risk_t = k_core_batch(int(ki_i))
            if _env_truthy("SELECTOR_PQ_JOINT_NATIVE_RISK_PREFIX", "0"):
                cached_grid_t = native_v_grid_by_ki.get(int(ki_i))
                if cached_grid_t is None:
                    native = load_selector_paged_pq_ext()
                    cached_grid_t = native.joint_vprefix_outputs_from_risk(
                        base_output_t.to(dtype=torch.float32).reshape(1, group_heads_i, int(self.head_dim)).contiguous(),
                        probs_t.to(dtype=torch.float32).reshape(1, group_heads_i, context_len_i).contiguous(),
                        residual_t.to(dtype=torch.float32).contiguous(),
                        code_error_t.to(dtype=torch.float32).contiguous(),
                        joint_v_budgets_t,
                    )[0]
                    native_v_grid_by_ki[int(ki_i)] = cached_grid_t
                out = cached_grid_t[int(vi_i)]
                outputs_by_budget[key] = out
                return out
            exact_count = max(0, min(int(joint_v_budgets[int(vi_i)]), context_len_i))
            count_key = (int(ki_i), int(exact_count))
            cached_count = v_outputs_by_count.get(count_key)
            if cached_count is not None:
                outputs_by_budget[key] = cached_count
                return cached_count
            if exact_count <= 0:
                out = base_output_t
            elif exact_count >= context_len_i:
                delta_t = probs_t.to(torch.float32) @ residual_t.float()
                out = base_output_t + delta_t
            else:
                exact_order_t = torch.topk(
                    risk_t,
                    k=int(exact_count),
                    dim=1,
                    largest=True,
                    sorted=True,
                ).indices
                gathered_probs_t = torch.gather(probs_t.to(torch.float32), 1, exact_order_t)
                gathered_residual_t = residual_t.index_select(0, exact_order_t.reshape(-1)).reshape(
                    group_heads_i,
                    int(exact_order_t.shape[1]),
                    int(self.head_dim),
                )
                delta_t = torch.sum(
                    gathered_probs_t.reshape(group_heads_i, -1, 1) * gathered_residual_t.float(),
                    dim=1,
                )
                out = base_output_t + delta_t
            v_outputs_by_count[count_key] = out
            outputs_by_budget[key] = out
            return out
        _selected_t_i, _k_mb_i, base_output_t, prefix_delta_t = k_artifacts_batch(int(ki_i))
        exact_count = max(0, min(int(joint_v_budgets[int(vi_i)]), context_len_i))
        if exact_count > 0 and prefix_delta_t is not None:
            out = base_output_t + prefix_delta_t[:, int(exact_count) - 1, :]
        else:
            out = base_output_t
        outputs_by_budget[key] = out
        return out

    sim_t0 = 0.0
    if bool(getattr(args, "profile_native_ops", False)):
        _sync_if_cuda(device)
        sim_t0 = time.perf_counter()
    policy_name = str(getattr(args, "joint_kv_policy", "k_first_alternating"))
    threshold_value = float(getattr(args, "joint_kv_stability_threshold", 0.001))
    start_budget_tensors = None
    if device.type == "cuda":
        start_budget_cache = getattr(self, "_pagedpq_start_budget_tensor_cache", None)
        if not isinstance(start_budget_cache, dict):
            start_budget_cache = {}
            setattr(self, "_pagedpq_start_budget_tensor_cache", start_budget_cache)
        start_budget_key = (
            str(device),
            tuple(int(v) for v in active_joint_k_budgets),
            tuple(int(v) for v in joint_v_budgets),
        )
        start_budget_tensors = start_budget_cache.get(start_budget_key)
        if start_budget_tensors is None:
            start_budget_tensors = (
                torch.as_tensor(active_joint_k_budgets, dtype=torch.long, device=device),
                torch.as_tensor(joint_v_budgets, dtype=torch.float32, device=device),
            )
            start_budget_cache[start_budget_key] = start_budget_tensors
    start_ki_by_head, start_vi_by_head = _joint_start_indices_for_heads(
        strategy=str(getattr(args, "joint_kv_start_strategy", "min")),
        context_len=int(context_len_i),
        dense_score_rows_t=ranked_nonbase_scores_t,
        sqrt_dim=float(sqrt_dim),
        k_budgets=active_joint_k_budgets,
        v_budgets=joint_v_budgets,
        sorted_dense_score_rows_t=sorted_ranked_score_rows_t,
        budget_tensors=start_budget_tensors,
    )
    final_ki_by_head: list[int] = []
    final_vi_by_head: list[int] = []
    final_idx_t_for_output: torch.Tensor | None = None
    final_output_grid_t: torch.Tensor | None = None
    grid_outputs_t: torch.Tensor | None = None
    incremental_grid_outputs_by_v_idx: dict[int, torch.Tensor] | None = None
    grid_outputs_for_v_idx = None
    grid_selected_by_ki: list[torch.Tensor | None] | None = None
    grid_selected_counts_by_ki: list[int] | None = None
    grid_k_mb_by_idx: list[float] | None = None
    grid_k_lo_counts_by_ki: list[int] | None = None
    grid_v_lo_reads_t: torch.Tensor | None = None
    use_incremental_v_grid = _env_truthy("SELECTOR_PQ_JOINT_INCREMENTAL_V_GRID", "0")
    select_trace_start = time_trace.cuda_start("select") if time_trace is not None else None
    if _env_truthy("SELECTOR_PQ_JOINT_GRID_ARTIFACTS", "1"):
        joint_score_wall_t0 = time.perf_counter() if wall_profile_enabled else 0.0
        if bool(getattr(args, "profile_native_ops", False)):
            _sync_if_cuda(device)
            joint_score_t0 = time.perf_counter()
        else:
            joint_score_t0 = 0.0
        grid_selected_by_ki = []
        grid_selected_counts_by_ki = [] if needs_logical_accounting else None
        grid_score_rows: list[torch.Tensor] = []
        grid_k_mb_by_idx = [] if needs_budget_mb_vectors else None
        grid_k_lo_counts_by_ki = [] if precision_tiers_enabled else None
        grid_take_counts: list[int] = []
        exact_full_budget_grid_flag = _env_truthy("SELECTOR_PQ_JOINT_EXACT_FULL_BUDGET_GRID", "1")
        native_score_grid_enabled = (
            _env_truthy("SELECTOR_PQ_JOINT_NATIVE_SCORE_GRID", "0")
            and not use_unsorted_k_prefix
        )
        torch_score_grid_t = (
            torch_score_grid_workspace_for(
                k_count=len(active_joint_k_budgets),
                heads=int(group_heads_i),
                context_len=int(context_len_i),
                dtype=prob_dtype,
            )
            if device.type == "cuda" and not native_score_grid_enabled
            else None
        )
        fused_mixed_softmax_base_enabled = _env_truthy(
            "SELECTOR_PQ_JOINT_FUSED_SOFTMAX_BASE",
            "0",
        )
        fused_sparse_direct_softmax_enabled = (
            fused_mixed_softmax_base_enabled
            and _env_truthy("SELECTOR_PQ_JOINT_FUSED_TOKENFIT_SOFTMAX_BASE", "0")
            and use_sparse_direct_score_grid
            and exact_scores_h is None
            and sparse_base_logits_t is not None
            and sparse_ranked_tokens_t is not None
            and sparse_ranked_logits_t is not None
            and use_score_grid_no_fill
        )
        if fused_mixed_softmax_base_enabled and not native_score_grid_enabled:
            raise RuntimeError("SELECTOR_PQ_JOINT_FUSED_SOFTMAX_BASE requires native score-grid mode")
        if fused_mixed_softmax_base_enabled and not _env_truthy(
            "SELECTOR_PQ_JOINT_NATIVE_SOFTMAX_BASE",
            "0",
        ):
            raise RuntimeError("SELECTOR_PQ_JOINT_FUSED_SOFTMAX_BASE requires native softmax/base mode")
        if exact_scores_h is None and not (
            use_sparse_direct_score_grid
            and native_score_grid_enabled
            and _env_truthy("SELECTOR_PQ_JOINT_TOKENFIT_SCORE_GRID", "0")
            and _env_truthy("SELECTOR_PQ_JOINT_SCORE_GRID_NO_EXACT_FILL", "0")
        ):
            raise RuntimeError(
                "SELECTOR_PQ_JOINT_SPARSE_DIRECT_SCORE_GRID requires native tokenfit no-fill score grid"
            )
        fused_tokenfit_softmax_base_enabled = _env_truthy(
            "SELECTOR_PQ_JOINT_FUSED_TOKENFIT_SOFTMAX_BASE",
            "0",
        )
        for ki_i, k_budget in enumerate(active_joint_k_budgets):
            take_i = max(0, min(int(k_budget), int(ranked_nonbase_t.numel())))
            grid_take_counts.append(int(take_i))
            if exact_full_budget_grid_flag and int(take_i) >= ranked_nonbase_count:
                # This K-budget row is exact over the full context, so avoid
                # materializing the full selected-token tensor on the hot path.
                selected_t_i = None
                selected_len_i = int(base_t.numel()) + int(ranked_nonbase_count)
            elif native_score_grid_enabled:
                # The native score-grid path only needs the take count, base
                # tokens, and ranked prefix. Avoid allocating base+prefix
                # selected-token tensors for every K budget.
                selected_t_i = None
                selected_len_i = int(base_t.numel()) + int(take_i)
            else:
                selected_t_i = selected_for_budget_batch(int(k_budget))
                selected_len_i = int(selected_t_i.shape[1]) if selected_t_i.ndim == 2 else int(selected_t_i.numel())
            grid_selected_by_ki.append(selected_t_i)
            if grid_selected_counts_by_ki is not None:
                grid_selected_counts_by_ki.append(int(selected_len_i))
            k_lo_count_i = 0
            if precision_tiers_enabled and int(take_i) > 0:
                k_lo_count_i = int(take_i) - int(
                    math.ceil(float(take_i) * _FROZEN_PRECISION_K_HI_FRAC)
                )
            if grid_k_lo_counts_by_ki is not None:
                grid_k_lo_counts_by_ki.append(int(k_lo_count_i))
            if grid_k_mb_by_idx is not None:
                grid_k_mb_by_idx.append(
                    float(selector_mb)
                    + float(selected_len_i * int(self.head_dim) * key_bytes) / MB
                    - float(
                        k_lo_count_i
                        * int(self.head_dim)
                        * (key_bytes - _FROZEN_PRECISION_LO_BYTES)
                    )
                    / MB
                )
            if not native_score_grid_enabled:
                if selected_t_i is None:
                    selected_t_i = selected_for_budget_batch(int(k_budget))
                    grid_selected_by_ki[-1] = selected_t_i
                if torch_score_grid_t is None:
                    grid_score_rows.append(mixed_scores_for_selected_batch(selected_t_i))
                else:
                    mixed_scores_for_selected_batch(
                        selected_t_i,
                        out_t=torch_score_grid_t[int(ki_i)],
                    )
        probs_grid_t: torch.Tensor | None = None
        base_output_grid_t: torch.Tensor | None = None
        score_grid_t: torch.Tensor | None = None
        grouped_score_grid_workspace_record_t: torch.Tensor | None = None
        if native_score_grid_enabled:
            if prob_dtype != torch.float32:
                raise RuntimeError("native joint score-grid currently requires fp32 probabilities")
            if ranked_prefix_tokens_t is None:
                ranked_prefix_tokens_for_grid_t = torch.empty(
                    (group_heads_i, 0),
                    dtype=torch.long,
                    device=device,
                )
            else:
                ranked_prefix_tokens_for_grid_t = ranked_prefix_tokens_t
            native = load_selector_paged_pq_ext()
            k_take_counts_t = torch.as_tensor(
                grid_take_counts,
                dtype=torch.long,
                device=device,
            )
            y_for_grid_t = (
                y_indexed_for_native_score_grid()
            )
            if _env_truthy("SELECTOR_PQ_JOINT_MERGE_RISK_POLICY", "0"):
                if policy_uses_mb:
                    raise RuntimeError(
                        "SELECTOR_PQ_JOINT_MERGE_RISK_POLICY only supports non-MB joint policies"
                    )
                if str(args.tail_score_calibration) == "affine_selected":
                    raise RuntimeError(
                        "SELECTOR_PQ_JOINT_MERGE_RISK_POLICY requires raw/no-calibration K-PQ tail logits"
                    )
                if exact_scores_h is None:
                    raise RuntimeError("SELECTOR_PQ_JOINT_MERGE_RISK_POLICY requires exact score table")
                if not hasattr(native, "joint_mixed_select_policy_merge_rankpos_no_calib_no_mb"):
                    raise RuntimeError(
                        "SELECTOR_PQ_JOINT_MERGE_RISK_POLICY requires updated CUDA extension"
                    )
                merge_wall_t0 = time.perf_counter() if wall_profile_enabled else 0.0
                if bool(getattr(args, "profile_native_ops", False)):
                    _sync_if_cuda(device)
                    merge_t0 = time.perf_counter()
                else:
                    merge_t0 = 0.0
                final_outputs_t, final_idx_t = native.joint_mixed_select_policy_merge_rankpos_no_calib_no_mb(
                    exact_scores_h.to(dtype=torch.float32).contiguous(),
                    pq_logits_t.to(dtype=torch.float32).contiguous(),
                    indexed_tokens_t.to(dtype=torch.long).contiguous(),
                    base_t.to(dtype=torch.long).contiguous(),
                    ranked_prefix_tokens_for_grid_t.to(dtype=torch.long).contiguous(),
                    k_take_counts_t,
                    vhat_all_t.to(dtype=torch.float32).contiguous(),
                    residual_t.to(dtype=torch.float32).contiguous(),
                    code_error_t.to(dtype=torch.float32).contiguous(),
                    joint_v_budgets_t,
                    float(pq_logit_scale),
                    float(threshold_value),
                    int(policy_id),
                )
                if bool(getattr(args, "profile_native_ops", False)):
                    _sync_if_cuda(device)
                    elapsed = float(time.perf_counter() - merge_t0)
                    stats[layer_id].add_joint_detail_timing(risk_prefix_seconds=elapsed)
                    stats[layer_id].add_native_detail_timing(
                        geometric_seconds=float(time.perf_counter() - sim_t0)
                    )
                if wall_profile_enabled:
                    stats[layer_id].add_joint_wall_timing(
                        risk_prefix_seconds=float(time.perf_counter() - merge_wall_t0)
                    )
                if bool(getattr(args, "disable_cost_stats", False)):
                    outputs_all[head_start_i:head_end_i] = final_outputs_t[:group_heads_i]
                    return True
                if grid_selected_counts_by_ki is None:
                    raise RuntimeError("merge-risk policy requires selected-count metadata for accounting")
                final_idx_rows = final_idx_t.detach().cpu().tolist()
                for local_head_i, row in enumerate(final_idx_rows):
                    ki = int(row[0])
                    vi = int(row[1])
                    selected_count_i = int(grid_selected_counts_by_ki[int(ki)])
                    exact_v_count = max(0, min(int(joint_v_budgets[int(vi)]), context_len_i))
                    exact_key_mb = float(selected_count_i * int(self.head_dim) * key_bytes) / MB
                    exact_v_mb = float(exact_v_count * int(self.head_dim) * value_bytes) / MB
                    compressed_v_codes_mb = (
                        float(
                            max(0, context_len_i - int(exact_v_count))
                            * int(actual_value_subvecs)
                            * int(code_bytes)
                        )
                        / MB
                    )
                    tail_mb_override = (
                        float(v_pq_codebook_mb)
                        + compressed_v_codes_mb
                        + float(metadata_mb)
                    )
                    dense_physical_key_mb = float(context_len_i * int(self.head_dim) * key_bytes) / MB
                    stats[layer_id].add_count(
                        int(selected_count_i),
                        max(0, context_len_i - int(exact_v_count)),
                        float(selector_mb),
                        int(self.head_dim),
                        key_bytes,
                        value_bytes,
                        tail_mb_override=tail_mb_override,
                        exact_kv_mb_override=float(exact_key_mb + exact_v_mb),
                        confidence_mb_override=0.0,
                        physical_gpu_exact_kv_mb_override=float(dense_physical_key_mb + exact_v_mb),
                        physical_gpu_confidence_mb_override=0.0,
                    )
                    outputs_all[int(head_start_i) + int(local_head_i)] = final_outputs_t[int(local_head_i)]
                return True
            use_score_grid_no_fill = _env_truthy("SELECTOR_PQ_JOINT_SCORE_GRID_NO_EXACT_FILL", "0")
            if use_score_grid_no_fill:
                if not bool(layout_covers_context):
                    raise RuntimeError(
                        "SELECTOR_PQ_JOINT_SCORE_GRID_NO_EXACT_FILL requires indexed tokens plus base "
                        "tokens to cover the full context"
                    )
            if fused_mixed_softmax_base_enabled:
                if use_score_grid_no_fill and not (
                    fused_sparse_direct_softmax_enabled or fused_tokenfit_softmax_base_enabled
                ):
                    raise RuntimeError(
                        "SELECTOR_PQ_JOINT_FUSED_SOFTMAX_BASE does not support no-fill diagnostic mode"
                    )
                if fused_sparse_direct_softmax_enabled:
                    fused_score_fn_name = "joint_mixed_softmax_base_outputs_sparse_exact_tokenfit_scaled"
                    if not hasattr(native, fused_score_fn_name):
                        raise RuntimeError(
                            "SELECTOR_PQ_JOINT_FUSED_SOFTMAX_BASE requires updated CUDA extension: "
                            f"{fused_score_fn_name}"
                        )
                    probs_grid_t, base_output_grid_t = getattr(native, fused_score_fn_name)(
                        pq_logits_t.to(dtype=torch.float32).contiguous(),
                        indexed_tokens_t.to(dtype=torch.long).contiguous(),
                        base_t.to(dtype=torch.long).contiguous(),
                        sparse_base_logits_t.to(dtype=torch.float32).contiguous(),
                        sparse_ranked_tokens_t.to(dtype=torch.long).contiguous(),
                        sparse_ranked_logits_t.to(dtype=torch.float32).contiguous(),
                        k_take_counts_t,
                        vhat_all_t.to(dtype=torch.float32).contiguous(),
                        int(context_len_i),
                        bool(str(args.tail_score_calibration) == "affine_selected"),
                        float(pq_logit_scale),
                    )
                else:
                    use_rankpos_score_grid = _env_truthy("SELECTOR_PQ_JOINT_RANKPOS_SCORE_GRID", "0")
                    if fused_tokenfit_softmax_base_enabled:
                        fused_score_fn_name = "joint_mixed_softmax_base_outputs_tokenfit_scaled"
                    else:
                        if _env_truthy("SELECTOR_PQ_JOINT_TOKENFIT_SCORE_GRID", "0"):
                            raise RuntimeError(
                                "SELECTOR_PQ_JOINT_TOKENFIT_SCORE_GRID requires "
                                "SELECTOR_PQ_JOINT_FUSED_TOKENFIT_SOFTMAX_BASE with fused softmax/base"
                            )
                        if native_pq_scale_in_kernel:
                            raise RuntimeError(
                                "SELECTOR_PQ_JOINT_NATIVE_PQ_SCALE_IN_KERNEL requires "
                                "SELECTOR_PQ_JOINT_FUSED_TOKENFIT_SOFTMAX_BASE with fused softmax/base"
                            )
                        fused_score_fn_name = (
                            "joint_mixed_softmax_base_outputs_rankpos"
                            if use_rankpos_score_grid
                            else "joint_mixed_softmax_base_outputs"
                        )
                    if not hasattr(native, fused_score_fn_name):
                        raise RuntimeError(
                            f"SELECTOR_PQ_JOINT_FUSED_SOFTMAX_BASE requires updated CUDA extension: {fused_score_fn_name}"
                        )
                    fused_args = (
                        exact_scores_h.to(dtype=torch.float32).contiguous(),
                        pq_logits_t.to(dtype=torch.float32).contiguous(),
                        y_for_grid_t.contiguous(),
                        indexed_tokens_t.to(dtype=torch.long).contiguous(),
                        base_t.to(dtype=torch.long).contiguous(),
                        ranked_prefix_tokens_for_grid_t.to(dtype=torch.long).contiguous(),
                        k_take_counts_t,
                        vhat_all_t.to(dtype=torch.float32).contiguous(),
                        bool(str(args.tail_score_calibration) == "affine_selected"),
                    )
                    if fused_tokenfit_softmax_base_enabled:
                        probs_grid_t, base_output_grid_t = getattr(native, fused_score_fn_name)(
                            *fused_args,
                            float(pq_logit_scale),
                        )
                    else:
                        probs_grid_t, base_output_grid_t = getattr(native, fused_score_fn_name)(
                            *fused_args,
                        )
                score_grid_t = None
            else:
                use_rankpos_score_grid = _env_truthy("SELECTOR_PQ_JOINT_RANKPOS_SCORE_GRID", "0")
                use_tokenfit_score_grid = _env_truthy(
                    "SELECTOR_PQ_JOINT_TOKENFIT_SCORE_GRID",
                    "0",
                )
                if use_tokenfit_score_grid and use_rankpos_score_grid:
                    raise RuntimeError(
                        "SELECTOR_PQ_JOINT_TOKENFIT_SCORE_GRID does not support rank-position score grid"
                    )
                if native_pq_scale_in_kernel and use_rankpos_score_grid:
                    raise RuntimeError(
                        "SELECTOR_PQ_JOINT_NATIVE_PQ_SCALE_IN_KERNEL does not support rank-position score grid yet"
                    )
                if use_rankpos_score_grid and use_score_grid_no_fill:
                    raise RuntimeError(
                        "SELECTOR_PQ_JOINT_RANKPOS_SCORE_GRID does not support no-fill diagnostic mode"
                    )
                use_score_grid_workspace = _env_truthy(
                    "SELECTOR_PQ_JOINT_SCORE_GRID_WORKSPACE",
                    "0",
                )
                use_grouped_score_grid_workspace = (
                    grouped_score_workspace_enabled
                    and hasattr(native, "joint_mixed_score_grid_workspace")
                    and exact_scores_h is not None
                    and int(group_heads_i) == int(group_size)
                )
                if use_score_grid_workspace and (
                    use_rankpos_score_grid
                    or use_tokenfit_score_grid
                    or use_score_grid_no_fill
                    or native_pq_scale_in_kernel
                ):
                    raise RuntimeError(
                        "SELECTOR_PQ_JOINT_SCORE_GRID_WORKSPACE only supports the canonical "
                        "non-rankpos, non-tokenfit, exact-fill score-grid path"
                    )
                if use_grouped_score_grid_workspace and (
                    use_rankpos_score_grid
                    or use_tokenfit_score_grid
                    or use_score_grid_no_fill
                    or native_pq_scale_in_kernel
                    or _env_truthy("SELECTOR_PQ_JOINT_SCORE_DIRECT_VPREFIX", "0")
                    or fused_mixed_softmax_base_enabled
                ):
                    raise RuntimeError(
                        "SELECTOR_PQ_JOINT_GROUPED_SCORE_WORKSPACE only supports the canonical "
                        "non-rankpos, non-tokenfit, exact-fill, non-fused score-grid path"
                    )
                use_nocalib_score_grid_workspace = (
                    _env_truthy("SELECTOR_PQ_JOINT_NOCALIB_SCORE_GRID_WORKSPACE", "0")
                    and str(args.tail_score_calibration) != "affine_selected"
                    and not use_rankpos_score_grid
                    and not use_tokenfit_score_grid
                    and not use_score_grid_no_fill
                    and not native_pq_scale_in_kernel
                    and hasattr(native, "joint_mixed_score_grid_rankpos_nocalib_workspace")
                )
                use_nocalib_scatter_score_grid = (
                    _env_truthy("SELECTOR_PQ_JOINT_NOCALIB_SCATTER_SCORE_GRID", "0")
                    and str(args.tail_score_calibration) != "affine_selected"
                    and not use_rankpos_score_grid
                    and not use_tokenfit_score_grid
                    and not use_score_grid_no_fill
                    and not native_pq_scale_in_kernel
                    and hasattr(native, "joint_mixed_score_grid_nocalib_scatter_workspace")
                )
                if use_nocalib_scatter_score_grid:
                    score_workspace_t, token_to_indexed_workspace_t = nocalib_scatter_score_grid_workspace_for(
                        k_count=int(k_take_counts_t.numel()),
                        heads=int(exact_scores_h.shape[0]),
                        context_len=int(context_len_i),
                    )
                    score_grid_t = native.joint_mixed_score_grid_nocalib_scatter_workspace(
                        score_workspace_t,
                        token_to_indexed_workspace_t,
                        exact_scores_h.to(dtype=torch.float32).contiguous(),
                        pq_logits_t.to(dtype=torch.float32).contiguous(),
                        indexed_tokens_t.to(dtype=torch.long).contiguous(),
                        base_t.to(dtype=torch.long).contiguous(),
                        ranked_prefix_tokens_for_grid_t.to(dtype=torch.long).contiguous(),
                        k_take_counts_t,
                        float(pq_logit_scale),
                    )
                elif use_nocalib_score_grid_workspace:
                    (
                        score_workspace_t,
                        token_to_indexed_workspace_t,
                        base_mask_workspace_t,
                        rank_pos_workspace_t,
                    ) = nocalib_score_grid_workspace_for(
                        k_count=int(k_take_counts_t.numel()),
                        heads=int(exact_scores_h.shape[0]),
                        context_len=int(context_len_i),
                    )
                    score_grid_t = native.joint_mixed_score_grid_rankpos_nocalib_workspace(
                        score_workspace_t,
                        token_to_indexed_workspace_t,
                        base_mask_workspace_t,
                        rank_pos_workspace_t,
                        exact_scores_h.to(dtype=torch.float32).contiguous(),
                        pq_logits_t.to(dtype=torch.float32).contiguous(),
                        indexed_tokens_t.to(dtype=torch.long).contiguous(),
                        base_t.to(dtype=torch.long).contiguous(),
                        ranked_prefix_tokens_for_grid_t.to(dtype=torch.long).contiguous(),
                        k_take_counts_t,
                        float(pq_logit_scale),
                    )
                elif use_grouped_score_grid_workspace:
                    score_workspace_t, mask_workspace_t, fit_scale_workspace_t, fit_bias_workspace_t = (
                        grouped_score_grid_workspace_for(
                            kv_heads=int(num_kv_heads),
                            k_count=int(k_take_counts_t.numel()),
                            heads=int(exact_scores_h.shape[0]),
                            context_len=int(context_len_i),
                        )
                    )
                    grouped_score_grid_workspace_record_t = score_workspace_t
                    score_grid_t = native.joint_mixed_score_grid_workspace(
                        score_workspace_t[int(kv_head_i)],
                        mask_workspace_t[int(kv_head_i)],
                        fit_scale_workspace_t[int(kv_head_i)],
                        fit_bias_workspace_t[int(kv_head_i)],
                        exact_scores_h.to(dtype=torch.float32).contiguous(),
                        pq_logits_t.to(dtype=torch.float32).contiguous(),
                        y_for_grid_t.contiguous(),
                        indexed_tokens_t.to(dtype=torch.long).contiguous(),
                        base_t.to(dtype=torch.long).contiguous(),
                        ranked_prefix_tokens_for_grid_t.to(dtype=torch.long).contiguous(),
                        k_take_counts_t,
                        bool(str(args.tail_score_calibration) == "affine_selected"),
                    )
                elif use_score_grid_workspace:
                    if not hasattr(native, "joint_mixed_score_grid_workspace"):
                        raise RuntimeError(
                            "SELECTOR_PQ_JOINT_SCORE_GRID_WORKSPACE requires updated CUDA extension"
                        )
                    score_workspace_t, mask_workspace_t, fit_scale_workspace_t, fit_bias_workspace_t = (
                        score_grid_workspace_for(
                            k_count=int(k_take_counts_t.numel()),
                            heads=int(exact_scores_h.shape[0]),
                            context_len=int(context_len_i),
                        )
                    )
                    score_grid_t = native.joint_mixed_score_grid_workspace(
                        score_workspace_t,
                        mask_workspace_t,
                        fit_scale_workspace_t,
                        fit_bias_workspace_t,
                        exact_scores_h.to(dtype=torch.float32).contiguous(),
                        pq_logits_t.to(dtype=torch.float32).contiguous(),
                        y_for_grid_t.contiguous(),
                        indexed_tokens_t.to(dtype=torch.long).contiguous(),
                        base_t.to(dtype=torch.long).contiguous(),
                        ranked_prefix_tokens_for_grid_t.to(dtype=torch.long).contiguous(),
                        k_take_counts_t,
                        bool(str(args.tail_score_calibration) == "affine_selected"),
                    )
                elif use_rankpos_score_grid:
                    if not hasattr(native, "joint_mixed_score_grid_rankpos"):
                        raise RuntimeError(
                            "SELECTOR_PQ_JOINT_RANKPOS_SCORE_GRID requires updated CUDA extension"
                        )
                    score_grid_fn = native.joint_mixed_score_grid_rankpos
                elif use_tokenfit_score_grid:
                    scaled_fn_name = (
                        "joint_mixed_score_grid_sparse_exact_tokenfit_scaled"
                        if use_sparse_direct_score_grid and exact_scores_h is None
                        else "joint_mixed_score_grid_no_exact_fill_tokenfit_scaled"
                        if use_score_grid_no_fill
                        else "joint_mixed_score_grid_tokenfit_scaled"
                    )
                    if not hasattr(native, scaled_fn_name):
                        raise RuntimeError(
                            "SELECTOR_PQ_JOINT_TOKENFIT_SCORE_GRID requires updated CUDA extension: "
                            f"{scaled_fn_name}"
                        )
                    if use_sparse_direct_score_grid and exact_scores_h is None:
                        if sparse_base_logits_t is None or sparse_ranked_tokens_t is None or sparse_ranked_logits_t is None:
                            raise RuntimeError("missing sparse exact logits for direct sparse score-grid")
                        score_grid_t = getattr(native, scaled_fn_name)(
                            pq_logits_t.to(dtype=torch.float32).contiguous(),
                            indexed_tokens_t.to(dtype=torch.long).contiguous(),
                            base_t.to(dtype=torch.long).contiguous(),
                            sparse_base_logits_t.to(dtype=torch.float32).contiguous(),
                            sparse_ranked_tokens_t.to(dtype=torch.long).contiguous(),
                            sparse_ranked_logits_t.to(dtype=torch.float32).contiguous(),
                            k_take_counts_t,
                            int(context_len_i),
                            bool(str(args.tail_score_calibration) == "affine_selected"),
                            float(pq_logit_scale),
                        )
                    else:
                        if exact_scores_h is None:
                            raise RuntimeError("missing exact score table for tokenfit score-grid")
                        score_grid_t = getattr(native, scaled_fn_name)(
                            exact_scores_h.to(dtype=torch.float32).contiguous(),
                            pq_logits_t.to(dtype=torch.float32).contiguous(),
                            y_for_grid_t.contiguous(),
                            indexed_tokens_t.to(dtype=torch.long).contiguous(),
                            base_t.to(dtype=torch.long).contiguous(),
                            ranked_prefix_tokens_for_grid_t.to(dtype=torch.long).contiguous(),
                            k_take_counts_t,
                            bool(str(args.tail_score_calibration) == "affine_selected"),
                            float(pq_logit_scale),
                        )
                else:
                    score_grid_fn = (
                        getattr(native, "joint_mixed_score_grid_no_exact_fill")
                        if use_score_grid_no_fill
                        and hasattr(native, "joint_mixed_score_grid_no_exact_fill")
                        else native.joint_mixed_score_grid
                    )
                if score_grid_t is not None:
                    pass
                elif native_pq_scale_in_kernel:
                    if exact_scores_h is None:
                        raise RuntimeError("missing exact score table for scaled score-grid")
                    scaled_fn_name = (
                        "joint_mixed_score_grid_no_exact_fill_scaled"
                        if use_score_grid_no_fill
                        else "joint_mixed_score_grid_scaled"
                    )
                    if not hasattr(native, scaled_fn_name):
                        raise RuntimeError(
                            "SELECTOR_PQ_JOINT_NATIVE_PQ_SCALE_IN_KERNEL requires updated CUDA extension: "
                            f"{scaled_fn_name}"
                        )
                    score_grid_t = getattr(native, scaled_fn_name)(
                        exact_scores_h.to(dtype=torch.float32).contiguous(),
                        pq_logits_t.to(dtype=torch.float32).contiguous(),
                        y_for_grid_t.contiguous(),
                        indexed_tokens_t.to(dtype=torch.long).contiguous(),
                        base_t.to(dtype=torch.long).contiguous(),
                        ranked_prefix_tokens_for_grid_t.to(dtype=torch.long).contiguous(),
                        k_take_counts_t,
                        bool(str(args.tail_score_calibration) == "affine_selected"),
                        float(pq_logit_scale),
                    )
                else:
                    if exact_scores_h is None:
                        raise RuntimeError("missing exact score table for score-grid")
                    score_grid_t = score_grid_fn(
                        exact_scores_h.to(dtype=torch.float32).contiguous(),
                        pq_logits_t.to(dtype=torch.float32).contiguous(),
                        y_for_grid_t.contiguous(),
                        indexed_tokens_t.to(dtype=torch.long).contiguous(),
                        base_t.to(dtype=torch.long).contiguous(),
                        ranked_prefix_tokens_for_grid_t.to(dtype=torch.long).contiguous(),
                        k_take_counts_t,
                        bool(str(args.tail_score_calibration) == "affine_selected"),
                    )
        else:
            score_grid_t = (
                torch_score_grid_t
                if torch_score_grid_t is not None
                else torch.stack(grid_score_rows, dim=0)
            )
        if bool(getattr(args, "profile_native_ops", False)):
            _sync_if_cuda(device)
            stats[layer_id].add_joint_detail_timing(
                score_grid_seconds=float(time.perf_counter() - joint_score_t0)
            )
            joint_prob_t0 = time.perf_counter()
        else:
            joint_prob_t0 = 0.0
        if wall_profile_enabled:
            stats[layer_id].add_joint_wall_timing(
                score_grid_seconds=float(time.perf_counter() - joint_score_wall_t0)
            )
        joint_prob_wall_t0 = time.perf_counter() if wall_profile_enabled else 0.0
        k_count_i = len(active_joint_k_budgets)
        use_score_direct_vprefix = (
            use_grouped_risk_prefix
            and _env_truthy("SELECTOR_PQ_JOINT_SCORE_DIRECT_VPREFIX", "0")
        )
        use_grouped_softmax_base_cublas = (
            use_grouped_risk_prefix
            and _env_truthy("SELECTOR_PQ_JOINT_GROUPED_SOFTMAX_BASE_CUBLAS", "0")
        )
        use_grouped_softmax_base = (
            use_grouped_risk_prefix
            and _env_truthy("SELECTOR_PQ_JOINT_GROUPED_SOFTMAX_BASE", "0")
        )
        if use_grouped_softmax_base and use_grouped_softmax_base_cublas:
            raise RuntimeError(
                "SELECTOR_PQ_JOINT_GROUPED_SOFTMAX_BASE and "
                "SELECTOR_PQ_JOINT_GROUPED_SOFTMAX_BASE_CUBLAS are mutually exclusive"
            )
        if use_grouped_softmax_base or use_grouped_softmax_base_cublas:
            if score_grid_t is None:
                raise RuntimeError("grouped softmax/base requires score_grid_t")
            native = load_selector_paged_pq_ext()
            grouped_softmax_fn_name = (
                "joint_softmax_base_outputs_grouped_cublas"
                if use_grouped_softmax_base_cublas
                else "joint_softmax_base_outputs_grouped"
            )
            if not hasattr(native, grouped_softmax_fn_name):
                raise RuntimeError(
                    "grouped softmax/base requires updated CUDA extension"
                )
        elif use_score_direct_vprefix:
            if score_grid_t is None:
                raise RuntimeError("SELECTOR_PQ_JOINT_SCORE_DIRECT_VPREFIX requires score_grid_t")
            native = load_selector_paged_pq_ext()
            if not hasattr(native, "joint_vprefix_outputs_from_grouped_scores_batched"):
                raise RuntimeError(
                    "SELECTOR_PQ_JOINT_SCORE_DIRECT_VPREFIX requires updated CUDA extension"
                )
        elif (
            probs_grid_t is None
            and _env_truthy("SELECTOR_PQ_JOINT_NATIVE_SOFTMAX_BASE", "0")
            and not compact_grouped_vpq_enabled
        ):
            native = load_selector_paged_pq_ext()
            if not hasattr(native, "joint_softmax_base_outputs"):
                raise RuntimeError(
                    "SELECTOR_PQ_JOINT_NATIVE_SOFTMAX_BASE requires updated CUDA extension"
                )
            if score_grid_t is None:
                raise RuntimeError("missing score grid for native softmax/base")
            score_grid_f32_t = score_grid_t.to(dtype=torch.float32).contiguous()
            vhat_f32_t = vhat_all_t.to(dtype=torch.float32).contiguous()
            use_softmax_base_cublas = _env_truthy("SELECTOR_PQ_JOINT_SOFTMAX_BASE_CUBLAS", "0")
            softmax_base_fn = (
                getattr(native, "joint_softmax_base_outputs_cublas")
                if use_softmax_base_cublas
                and hasattr(native, "joint_softmax_base_outputs_cublas")
                else native.joint_softmax_base_outputs
            )
            softmax_base_workspace_fn = (
                getattr(native, "joint_softmax_base_outputs_workspace_cublas")
                if use_softmax_base_cublas
                and hasattr(native, "joint_softmax_base_outputs_workspace_cublas")
                else native.joint_softmax_base_outputs_workspace
            )
            if grouped_strided_output_workspace_enabled:
                if use_softmax_base_cublas:
                    raise RuntimeError(
                        "SELECTOR_PQ_JOINT_STRIDED_OUTPUT_WORKSPACE does not support CUBLAS softmax/base"
                    )
                if not hasattr(native, "joint_softmax_base_outputs_strided_workspace"):
                    raise RuntimeError(
                        "SELECTOR_PQ_JOINT_STRIDED_OUTPUT_WORKSPACE requires updated CUDA extension"
                    )
                softmax_base_workspace_fn = native.joint_softmax_base_outputs_strided_workspace
            if (
                grouped_output_workspace_enabled
                and hasattr(native, "joint_softmax_base_outputs_workspace")
                and int(group_heads_i) == int(group_size)
            ):
                grouped_probs_workspace_t, grouped_base_workspace_t = grouped_output_workspace_for(
                    kv_heads=int(num_kv_heads),
                    k_count=int(score_grid_f32_t.shape[0]),
                    heads=int(group_heads_i),
                    context_len=int(score_grid_f32_t.shape[2]),
                    dim=int(self.head_dim),
                )
                probs_out_t = grouped_probs_workspace_t[int(kv_head_i)]
                base_out_t = grouped_base_workspace_t[int(kv_head_i)]
                probs_grid_t, base_output_grid_t = softmax_base_workspace_fn(
                    probs_out_t,
                    base_out_t,
                    score_grid_f32_t,
                    vhat_f32_t,
                )
            elif (
                _env_truthy("SELECTOR_PQ_JOINT_SOFTMAX_BASE_WORKSPACE", "0")
                and hasattr(native, "joint_softmax_base_outputs_workspace")
            ):
                probs_out_t, base_out_t = softmax_base_workspace_for(
                    slot=int(kv_head_i),
                    k_count=int(score_grid_f32_t.shape[0]),
                    heads=int(score_grid_f32_t.shape[1]),
                    context_len=int(score_grid_f32_t.shape[2]),
                    dim=int(vhat_f32_t.shape[1]),
                )
                probs_grid_t, base_output_grid_t = softmax_base_workspace_fn(
                    probs_out_t,
                    base_out_t,
                    score_grid_f32_t,
                    vhat_f32_t,
                )
            else:
                probs_grid_t, base_output_grid_t = softmax_base_fn(
                    score_grid_f32_t,
                    vhat_f32_t,
                )
        elif probs_grid_t is None:
            if score_grid_t is None:
                raise RuntimeError("missing score grid for Torch softmax/base")
            probs_grid_t = torch.softmax(score_grid_t, dim=2)
        if compact_grouped_vpq_enabled and not _env_truthy("SELECTOR_PQ_JOINT_NATIVE_VPQ_BASE", "0"):
            raise RuntimeError(
                "SELECTOR_PQ_JOINT_COMPACT_VPQ_RISK_PREFIX requires "
                "SELECTOR_PQ_JOINT_NATIVE_VPQ_BASE=1"
            )
        if (
            not use_score_direct_vprefix
            and not use_grouped_softmax_base
            and not use_grouped_softmax_base_cublas
            and base_output_grid_t is None
            and _env_truthy("SELECTOR_PQ_JOINT_NATIVE_VPQ_BASE", "0")
        ):
            native = load_selector_paged_pq_ext()
            if not hasattr(native, "joint_vpq_base_outputs_from_probs"):
                raise RuntimeError("SELECTOR_PQ_JOINT_NATIVE_VPQ_BASE requires updated CUDA extension")
            vpq_pack = joint_vpq_pack_and_fallback_for(
                index=index,
                values_t=values_t,
                context_len_i=context_len_i,
            )
            if vpq_pack is None:
                raise RuntimeError(
                    "SELECTOR_PQ_JOINT_NATIVE_VPQ_BASE requires VALUE_SUBVECS=1 V-PQ pack"
                )
            value_codebooks_t, value_codes_t, value_page_starts_t, _value_page_size_i, _value_subbits_i, fallback_tokens_t = vpq_pack
            base_output_grid_t = native.joint_vpq_base_outputs_from_probs(
                probs_grid_t.to(torch.float32).contiguous(),
                values_t.contiguous(),
                value_codebooks_t.to(dtype=torch.float32).contiguous(),
                value_codes_t.contiguous(),
                value_page_starts_t.to(dtype=torch.long).contiguous(),
                fallback_tokens_t.to(dtype=torch.long).contiguous(),
            )
        if (
            not use_score_direct_vprefix
            and not use_grouped_softmax_base
            and not use_grouped_softmax_base_cublas
            and base_output_grid_t is None
        ):
            base_output_grid_t = (
                probs_grid_t.to(torch.float32).reshape(k_count_i * group_heads_i, context_len_i)
                @ vhat_all_t.float()
            ).reshape(k_count_i, group_heads_i, int(self.head_dim))
        if bool(getattr(args, "profile_native_ops", False)):
            _sync_if_cuda(device)
            stats[layer_id].add_joint_detail_timing(
                prob_base_seconds=float(time.perf_counter() - joint_prob_t0)
            )
        if wall_profile_enabled:
            stats[layer_id].add_joint_wall_timing(
                prob_base_seconds=float(time.perf_counter() - joint_prob_wall_t0)
            )
        if use_grouped_risk_prefix:
            if runtime.grouped_geo_t0 == 0.0 and bool(getattr(args, "profile_native_ops", False)):
                runtime.grouped_geo_t0 = sim_t0
            record_i: dict[str, object] = {
                "head_start": int(head_start_i),
                "head_end": int(head_end_i),
                "kv_head": int(kv_head_i),
                "group_heads": int(group_heads_i),
                "context_len": int(context_len_i),
                "selector_mb": float(selector_mb),
                "v_pq_codebook_mb": float(v_pq_codebook_mb),
                "actual_value_subvecs": int(actual_value_subvecs),
                "code_bytes": int(code_bytes),
                "metadata_mb": float(metadata_mb),
                "grid_selected_by_ki": grid_selected_by_ki,
                "grid_selected_counts_by_ki": grid_selected_counts_by_ki,
                "grid_k_mb_by_idx": grid_k_mb_by_idx,
                "v_mb_by_idx": v_mb_by_idx,
                "code_error": code_error_t.to(dtype=torch.float32).contiguous(),
            }
            if not compact_grouped_vpq_enabled:
                record_i["residual"] = residual_t.to(dtype=torch.float32).contiguous()
            if _env_truthy("SELECTOR_PQ_JOINT_MERGE_RISK_PREFIX", "0"):
                if str(args.tail_score_calibration) == "affine_selected":
                    raise RuntimeError(
                        "SELECTOR_PQ_JOINT_MERGE_RISK_PREFIX requires raw/no-calibration K-PQ tail logits"
                    )
                if exact_scores_h is None:
                    raise RuntimeError("SELECTOR_PQ_JOINT_MERGE_RISK_PREFIX requires exact score table")
                if not native_score_grid_enabled:
                    raise RuntimeError(
                        "SELECTOR_PQ_JOINT_MERGE_RISK_PREFIX requires native score-grid rank-prefix metadata"
                    )
                record_i["merge_exact_scores"] = exact_scores_h.to(dtype=torch.float32).contiguous()
                record_i["merge_pq_logits"] = pq_logits_t.to(dtype=torch.float32).contiguous()
                record_i["merge_indexed_tokens"] = indexed_tokens_t.to(dtype=torch.long).contiguous()
                record_i["merge_base_tokens"] = base_t.to(dtype=torch.long).contiguous()
                record_i["merge_ranked_prefix_tokens"] = ranked_prefix_tokens_for_grid_t.to(
                    dtype=torch.long
                ).contiguous()
                record_i["merge_k_take_counts"] = k_take_counts_t
                record_i["merge_pq_scale"] = float(pq_logit_scale)
            if use_score_direct_vprefix or use_grouped_softmax_base or use_grouped_softmax_base_cublas:
                if score_grid_t is None:
                    raise RuntimeError("missing score grid for grouped deferred softmax/base")
                record_i["vhat"] = vhat_all_t.to(dtype=torch.float32).contiguous()
                if grouped_score_grid_workspace_record_t is not None:
                    # Keep the per-KV-head view backed by the grouped
                    # workspace. Copying it here defeats the later
                    # grouped no-stack path.
                    record_i["score_grid"] = score_grid_t
                    record_i["grouped_score_grid_workspace"] = grouped_score_grid_workspace_record_t
                else:
                    record_i["score_grid"] = score_grid_t.to(dtype=torch.float32).contiguous()
            else:
                record_i["base_output_grid"] = base_output_grid_t.to(dtype=torch.float32).contiguous()
                if (
                    grouped_strided_output_workspace_enabled
                    and grouped_probs_workspace_t is not None
                    and probs_grid_t.dtype == torch.float32
                ):
                    record_i["probs_grid"] = probs_grid_t
                    record_i["probs_context_len"] = int(context_len_i)
                else:
                    record_i["probs_grid"] = probs_grid_t.to(dtype=torch.float32).contiguous()
                record_i["grouped_probs_workspace"] = grouped_probs_workspace_t
                record_i["grouped_base_workspace"] = grouped_base_workspace_t
            grouped_risk_records.append(record_i)
            return True
        vprefix_result = build_joint_vprefix_grid(
            JointVPrefixGridRuntime(
                args=args,
                layer_id=int(layer_id),
                stats=stats,
                device=device,
                wall_profile_enabled=bool(wall_profile_enabled),
                use_incremental_v_grid=bool(use_incremental_v_grid),
                max_exact_v_count=int(max_exact_v_count),
                context_len=int(context_len_i),
                k_count=int(k_count_i),
                group_heads=int(group_heads_i),
                head_dim=int(self.head_dim),
                prob_dtype=prob_dtype,
                probs_grid=probs_grid_t,
                base_output_grid=base_output_grid_t,
                residual=residual_t,
                code_error=code_error_t,
                joint_v_budgets=joint_v_budgets,
                joint_v_budgets_t=joint_v_budgets_t,
                residual_lo_commit=residual_lo_commit_t,
                v_commit_mask=v_commit_mask_t,
                v_hi_frac=_FROZEN_PRECISION_V_HI_FRAC,
            )
        )
        grid_outputs_t = vprefix_result.grid_outputs
        grid_outputs_for_v_idx = vprefix_result.grid_outputs_for_v_idx
        grid_v_lo_reads_t = vprefix_result.v_lo_reads_grid
    policy_result = select_joint_kv_budgets(
        JointPolicyRuntime(
            args=args,
            layer_id=int(layer_id),
            stats=stats,
            device=device,
            wall_profile_enabled=bool(wall_profile_enabled),
            group_heads=int(group_heads_i),
            active_k_budgets=active_joint_k_budgets,
            v_budgets=joint_v_budgets,
            policy_name=policy_name,
            policy_id=int(policy_id),
            policy_uses_mb=bool(policy_uses_mb),
            threshold=float(threshold_value),
            context_len=int(context_len_i),
            threshold_mode=str(getattr(args, "joint_kv_threshold_mode", "fixed")),
            threshold_reference_frac=float(getattr(args, "joint_kv_threshold_reference_frac", 0.2)),
            threshold_scale_shape=str(getattr(args, "joint_kv_threshold_scale_shape", "linear")),
            threshold_min_scale=float(getattr(args, "joint_kv_threshold_min_scale", 0.0)),
            threshold_max_scale=float(getattr(args, "joint_kv_threshold_max_scale", 1.0)),
            start_ki_by_head=start_ki_by_head,
            start_vi_by_head=start_vi_by_head,
            use_incremental_v_grid=bool(use_incremental_v_grid),
            grid_outputs=grid_outputs_t,
            grid_outputs_for_v_idx=grid_outputs_for_v_idx,
            output_for_budget=output_for_budget_batch,
            k_artifacts=k_artifacts_batch,
            grid_k_mb_by_idx=grid_k_mb_by_idx,
            v_mb_by_idx=v_mb_by_idx,
            sim_start_seconds=float(sim_t0),
            v_lo_reads_grid=grid_v_lo_reads_t,
            time_trace=time_trace,
            defer_torch_policy=bool(
                runtime.defer_torch_policy
                and grid_outputs_t is not None
                and grid_selected_counts_by_ki is not None
                and grid_k_mb_by_idx is not None
            ),
            d2h_owner=self,
            d2h_slot=int(kv_head_i),
        )
    )
    final_ki_by_head = policy_result.final_ki_by_head
    final_vi_by_head = policy_result.final_vi_by_head
    final_idx_t_for_output = policy_result.final_idx_for_output
    final_output_grid_t = policy_result.final_output_grid

    finalize_runtime = JointFinalizeRuntime(
        args=args,
        module=self,
        layer_id=int(layer_id),
        stats=stats,
        device=device,
        outputs_all=outputs_all,
        head_start=int(head_start_i),
        head_end=int(head_end_i),
        group_heads=int(group_heads_i),
        context_len=int(context_len_i),
        key_bytes=int(key_bytes),
        value_bytes=int(value_bytes),
        selector_mb=float(selector_mb),
        actual_value_subvecs=int(actual_value_subvecs),
        code_bytes=int(code_bytes),
        v_pq_codebook_mb=float(v_pq_codebook_mb),
        metadata_mb=float(metadata_mb),
        joint_v_budgets=joint_v_budgets,
        final_ki_by_head=final_ki_by_head,
        final_vi_by_head=final_vi_by_head,
        final_idx_for_output=final_idx_t_for_output,
        final_output_grid=final_output_grid_t,
        grid_outputs=grid_outputs_t,
        grid_outputs_for_v_idx=grid_outputs_for_v_idx,
        grid_selected_counts_by_ki=grid_selected_counts_by_ki,
        grid_selected_by_ki=grid_selected_by_ki,
        k_artifacts=k_artifacts_batch,
        output_for_budget=output_for_budget_batch,
        precision_tiers_enabled=bool(precision_tiers_enabled),
        precision_v_hi_frac=float(_FROZEN_PRECISION_V_HI_FRAC),
        precision_lo_bytes=int(_FROZEN_PRECISION_LO_BYTES),
        k_lo_counts_by_ki=grid_k_lo_counts_by_ki,
        v_lo_reads_grid=grid_v_lo_reads_t,
        v_lo_reads_rows=policy_result.v_lo_reads_rows,
    )
    if policy_result.deferred_torch_policy is not None:
        if runtime.deferred_policy_records is None:
            raise RuntimeError("missing deferred policy record list")
        # The packed host record contains every CPU policy input.  Drop the
        # lazy-output closures before retaining eight groups until the single
        # layer-level wait; otherwise those closures keep context-sized
        # temporaries alive and defeat the memory bound.
        policy_result.deferred_torch_policy.runtime.output_for_budget = None
        policy_result.deferred_torch_policy.runtime.k_artifacts = None
        policy_result.deferred_torch_policy.runtime.grid_outputs = None
        policy_result.deferred_torch_policy.runtime.v_lo_reads_grid = None
        finalize_runtime.grid_selected_by_ki = None
        finalize_runtime.k_artifacts = None
        finalize_runtime.output_for_budget = None
        runtime.deferred_policy_records.append(
            (policy_result.deferred_torch_policy, finalize_runtime)
        )
        if time_trace is not None:
            time_trace.cuda_end("select", select_trace_start)
        return True
    finalized = finalize_joint_head_outputs(finalize_runtime)
    if time_trace is not None:
        time_trace.cuda_end("select", select_trace_start)
    return finalized


def finish_deferred_joint_kv_heads(runtime) -> bool:
    records = runtime.deferred_policy_records
    if not records:
        return True
    time_trace = getattr(runtime.self, "_pagedpq_joint_time_trace", None)
    wait_t0 = time.perf_counter() if time_trace is not None else 0.0
    ready = torch.cuda.Event()
    ready.record(torch.cuda.current_stream(runtime.device))
    ready.synchronize()
    if time_trace is not None:
        time_trace.add_cpu("sync_wait", time.perf_counter() - wait_t0)
    for prepared, finalize_runtime in records:
        selected, v_lo_reads_rows = finish_prepared_torch_policy(prepared)
        finalize_runtime.final_ki_by_head = [int(ki) for ki, _vi in selected]
        finalize_runtime.final_vi_by_head = [int(vi) for _ki, vi in selected]
        finalize_runtime.v_lo_reads_rows = v_lo_reads_rows
        if not finalize_joint_head_outputs(finalize_runtime):
            return False
    records.clear()
    return True
