#!/usr/bin/env python3
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch

from benchmark.selector_eval.gpu.run_gpu_paged_pq_eval import _sync_if_cuda, load_selector_paged_pq_ext
from benchmark.selector_eval.runners.hf_paged_pq_intervention_common import _env_truthy


@dataclass(frozen=True)
class JointFusedPolicyRuntime:
    args: Any
    module: Any
    layer_id: int
    stats: dict
    device: torch.device
    outputs_all: torch.Tensor
    head_start: int
    head_end: int
    group_heads: int
    context_len: int
    prob_dtype: torch.dtype
    policy_id: int
    policy_uses_mb: bool
    use_grouped_risk_prefix: bool
    use_unsorted_k_prefix: bool
    active_k_budgets: list[int]
    ranked_nonbase_count: int
    base_t: torch.Tensor
    ranked_prefix_tokens_t: torch.Tensor | None
    exact_scores_h: torch.Tensor | None
    pq_logits_t: torch.Tensor
    y_indexed_prob_t: torch.Tensor | None
    indexed_tokens_t: torch.Tensor
    vhat_all_t: torch.Tensor
    residual_t: torch.Tensor
    code_error_t: torch.Tensor
    joint_v_budgets_t: torch.Tensor
    key_bytes: int
    value_bytes: int
    selector_mb: float
    v_pq_codebook_mb: float
    metadata_mb: float
    actual_value_subvecs: int
    code_bytes: int
    wall_profile_enabled: bool
    pq_logit_scale: float


def try_process_fused_mixed_policy(runtime: JointFusedPolicyRuntime) -> bool:
    if not _env_truthy("SELECTOR_PQ_JOINT_FUSED_MIXED_POLICY", "0"):
        return False

    args = runtime.args
    device = runtime.device
    if not runtime.use_grouped_risk_prefix:
        raise RuntimeError("SELECTOR_PQ_JOINT_FUSED_MIXED_POLICY requires grouped risk-prefix mode")
    if runtime.policy_uses_mb:
        raise RuntimeError("SELECTOR_PQ_JOINT_FUSED_MIXED_POLICY only supports non-MB policies")
    if runtime.use_unsorted_k_prefix:
        raise RuntimeError("SELECTOR_PQ_JOINT_FUSED_MIXED_POLICY requires sorted prefix semantics")
    if runtime.prob_dtype != torch.float32:
        raise RuntimeError("SELECTOR_PQ_JOINT_FUSED_MIXED_POLICY requires fp32 probability logits")
    if runtime.exact_scores_h is None:
        raise RuntimeError("SELECTOR_PQ_JOINT_FUSED_MIXED_POLICY requires exact score rows")

    native = load_selector_paged_pq_ext()
    use_rankpos_nocalib = (
        str(args.tail_score_calibration) == "none"
        and hasattr(native, "joint_mixed_select_policy_intervals_rankpos_no_calib_no_mb")
    )
    if not use_rankpos_nocalib and not hasattr(native, "joint_mixed_select_policy_intervals_no_mb"):
        raise RuntimeError(
            "SELECTOR_PQ_JOINT_FUSED_MIXED_POLICY requires joint_mixed_select_policy_intervals_no_mb"
        )

    grid_take_counts = [
        max(0, min(int(k_budget), int(runtime.ranked_nonbase_count)))
        for k_budget in runtime.active_k_budgets
    ]
    grid_selected_counts_by_ki = [
        int(runtime.base_t.numel()) + int(take_i)
        for take_i in grid_take_counts
    ]
    ranked_prefix_tokens_for_fused_t = (
        runtime.ranked_prefix_tokens_t
        if runtime.ranked_prefix_tokens_t is not None
        else torch.empty((runtime.group_heads, 0), dtype=torch.long, device=device)
    )
    k_take_counts_t = torch.as_tensor(grid_take_counts, dtype=torch.long, device=device)
    fused_policy_wall_t0 = time.perf_counter() if runtime.wall_profile_enabled else 0.0
    if bool(getattr(args, "profile_native_ops", False)):
        _sync_if_cuda(device)
        fused_policy_t0 = time.perf_counter()
    else:
        fused_policy_t0 = 0.0
    if use_rankpos_nocalib:
        final_output_t, final_idx_t = native.joint_mixed_select_policy_intervals_rankpos_no_calib_no_mb(
            runtime.exact_scores_h.to(dtype=torch.float32).contiguous(),
            runtime.pq_logits_t.to(dtype=torch.float32).contiguous(),
            runtime.indexed_tokens_t.to(dtype=torch.long).contiguous(),
            runtime.base_t.to(dtype=torch.long).contiguous(),
            ranked_prefix_tokens_for_fused_t.to(dtype=torch.long).contiguous(),
            k_take_counts_t,
            runtime.vhat_all_t.to(dtype=torch.float32).contiguous(),
            runtime.residual_t.to(dtype=torch.float32).contiguous(),
            runtime.code_error_t.to(dtype=torch.float32).contiguous(),
            runtime.joint_v_budgets_t,
            float(runtime.pq_logit_scale),
            float(getattr(args, "joint_kv_stability_threshold", 0.001)),
            int(runtime.policy_id),
        )
    else:
        y_for_grid_t = (
            runtime.y_indexed_prob_t.to(dtype=torch.float32)
            if runtime.y_indexed_prob_t is not None
            else torch.empty_like(runtime.pq_logits_t, dtype=torch.float32)
        )
        final_output_t, final_idx_t = native.joint_mixed_select_policy_intervals_no_mb(
            runtime.exact_scores_h.to(dtype=torch.float32).contiguous(),
            runtime.pq_logits_t.to(dtype=torch.float32).contiguous(),
            y_for_grid_t.contiguous(),
            runtime.indexed_tokens_t.to(dtype=torch.long).contiguous(),
            runtime.base_t.to(dtype=torch.long).contiguous(),
            ranked_prefix_tokens_for_fused_t.to(dtype=torch.long).contiguous(),
            k_take_counts_t,
            runtime.vhat_all_t.to(dtype=torch.float32).contiguous(),
            runtime.residual_t.to(dtype=torch.float32).contiguous(),
            runtime.code_error_t.to(dtype=torch.float32).contiguous(),
            runtime.joint_v_budgets_t,
            bool(str(args.tail_score_calibration) == "affine_selected"),
            float(runtime.pq_logit_scale),
            float(getattr(args, "joint_kv_stability_threshold", 0.001)),
            int(runtime.policy_id),
        )
    if bool(getattr(args, "profile_native_ops", False)):
        _sync_if_cuda(device)
        runtime.stats[runtime.layer_id].add_joint_detail_timing(
            score_grid_seconds=float(time.perf_counter() - fused_policy_t0)
        )
    if runtime.wall_profile_enabled:
        runtime.stats[runtime.layer_id].add_joint_wall_timing(
            score_grid_seconds=float(time.perf_counter() - fused_policy_wall_t0)
        )
    if bool(getattr(args, "disable_cost_stats", False)):
        runtime.outputs_all[runtime.head_start: runtime.head_end] = final_output_t[: runtime.group_heads]
        return True

    fused_accounting_wall_t0 = time.perf_counter() if runtime.wall_profile_enabled else 0.0
    if bool(getattr(args, "profile_native_ops", False)):
        _sync_if_cuda(device)
        fused_accounting_t0 = time.perf_counter()
    else:
        fused_accounting_t0 = 0.0
    selected_counts_t = torch.as_tensor(
        [int(x) for x in grid_selected_counts_by_ki],
        dtype=torch.long,
        device=device,
    )
    accounting_sums_t = native.joint_grouped_accounting_sums(
        final_idx_t,
        selected_counts_t,
        runtime.joint_v_budgets_t,
        int(runtime.context_len),
        int(runtime.module.head_dim),
        int(runtime.key_bytes),
        int(runtime.value_bytes),
        float(runtime.selector_mb),
        float(runtime.v_pq_codebook_mb),
        float(runtime.metadata_mb),
        int(runtime.actual_value_subvecs),
        int(runtime.code_bytes),
    )
    runtime.stats[runtime.layer_id].add_count_sums_device(
        int(runtime.group_heads),
        accounting_sums_t,
    )
    runtime.outputs_all[runtime.head_start: runtime.head_end] = final_output_t[: runtime.group_heads]
    if bool(getattr(args, "profile_native_ops", False)):
        _sync_if_cuda(device)
        runtime.stats[runtime.layer_id].add_joint_detail_timing(
            accounting_seconds=float(time.perf_counter() - fused_accounting_t0)
        )
    if runtime.wall_profile_enabled:
        runtime.stats[runtime.layer_id].add_joint_wall_timing(
            accounting_seconds=float(time.perf_counter() - fused_accounting_wall_t0)
        )
    return True
