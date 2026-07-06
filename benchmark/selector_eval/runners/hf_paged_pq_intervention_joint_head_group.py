#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch

from benchmark.selector_eval.runners.hf_paged_pq_intervention_joint_one_group import process_one_joint_kv_head


@dataclass
class JointKVHeadGroupRuntime:
    args: Any
    self: Any
    layer_id: int
    stats: dict
    device: torch.device
    q_all: torch.Tensor
    torch_k_cache: dict[int, torch.Tensor]
    torch_v_cache: dict[int, torch.Tensor]
    context_len_i: int
    num_heads: int
    num_kv_heads: int
    group_size: int
    nprobes: list[int]
    key_bytes: int
    value_bytes: int
    local_qpos: int
    sqrt_dim: float
    prob_dtype: torch.dtype
    policy_id: int
    policy_uses_mb: bool
    needs_logical_accounting: bool
    needs_budget_mb_vectors: bool
    joint_k_budgets: list[int]
    joint_v_budgets: list[int]
    joint_v_budgets_t: torch.Tensor
    allhead_indexes: Any
    allhead_dense_pq_scores_t: torch.Tensor | None
    allhead_selector_mb: float | None
    allhead_exact_scores_t: torch.Tensor | None
    allhead_selector_rank_prefix_t: torch.Tensor | None
    allhead_rank_prefix_cache: dict
    use_unsorted_k_prefix: bool
    native_exact_logits_enabled: bool
    native_full_exact_logits: Callable
    use_grouped_risk_prefix: bool
    grouped_output_workspace_enabled: bool
    grouped_strided_output_workspace_enabled: bool
    grouped_score_workspace_enabled: bool
    grouped_vpq_vhat_groups_t: torch.Tensor | None
    grouped_vpq_residual_groups_t: torch.Tensor | None
    grouped_vpq_code_error_groups_t: torch.Tensor | None
    grouped_vpq_value_codebooks_t: torch.Tensor | None
    grouped_vpq_value_codes_t: torch.Tensor | None
    grouped_vpq_value_page_starts_t: torch.Tensor | None
    grouped_vpq_value_page_size: int | None
    grouped_vpq_values_t: torch.Tensor | None
    grouped_vpq_actual_subbits: int | None
    grouped_risk_records: list[dict[str, object]]
    outputs_all: torch.Tensor
    prefix_index_for: Callable
    joint_vpq_sidecars_for: Callable
    joint_vpq_pack_and_fallback_for: Callable
    token_layout_for: Callable
    nocalib_score_grid_workspace_for: Callable
    nocalib_scatter_score_grid_workspace_for: Callable
    score_grid_workspace_for: Callable
    grouped_score_grid_workspace_for: Callable
    grouped_output_workspace_for: Callable
    softmax_base_workspace_for: Callable
    native_rank_prefix_tokens: Callable
    wall_profile_enabled: bool
    kv_head_indices: list[int] | None = None
    grouped_geo_t0: float = 0.0


def process_joint_kv_head_groups(runtime: JointKVHeadGroupRuntime) -> bool:
    kv_head_indices = (
        list(runtime.kv_head_indices)
        if runtime.kv_head_indices is not None
        else list(range(runtime.num_kv_heads))
    )
    for kv_head_i in kv_head_indices:
        if not process_one_joint_kv_head(runtime, int(kv_head_i)):
            return False
    return True
