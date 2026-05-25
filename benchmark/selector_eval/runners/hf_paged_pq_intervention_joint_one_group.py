#!/usr/bin/env python3
from __future__ import annotations

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
    select_joint_kv_budgets,
)
from benchmark.selector_eval.runners.hf_paged_pq_intervention_joint_vprefix import (
    JointVPrefixGridRuntime,
    build_joint_vprefix_grid,
)


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
    grouped_vpq_actual_subbits = runtime.grouped_vpq_actual_subbits
    grouped_risk_records = runtime.grouped_risk_records
    outputs_all = runtime.outputs_all
    prefix_index_for = runtime.prefix_index_for
    joint_vpq_sidecars_for = runtime.joint_vpq_sidecars_for
    joint_vpq_pack_and_fallback_for = runtime.joint_vpq_pack_and_fallback_for
    token_layout_for = runtime.token_layout_for
    score_grid_workspace_for = runtime.score_grid_workspace_for
    grouped_score_grid_workspace_for = runtime.grouped_score_grid_workspace_for
    grouped_output_workspace_for = runtime.grouped_output_workspace_for
    softmax_base_workspace_for = runtime.softmax_base_workspace_for
    native_rank_prefix_tokens = runtime.native_rank_prefix_tokens
    wall_profile_enabled = runtime.wall_profile_enabled
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
    sparse_base_logits_t: torch.Tensor | None = None
    sparse_ranked_tokens_t: torch.Tensor | None = None
    sparse_ranked_logits_t: torch.Tensor | None = None
    if allhead_exact_scores_t is not None and int(allhead_exact_scores_t.shape[1]) >= context_len_i:
        exact_scores_h = allhead_exact_scores_t[head_start_i:head_end_i, :context_len_i]
    elif use_sparse_exact_score_grid:
        exact_scores_h = None
    else:
        exact_wall_t0 = time.perf_counter() if wall_profile_enabled else 0.0
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

    values_t = torch_v_cache[int(kv_head_i)][:context_len_i]
    vsidecar_t0 = 0.0
    vsidecar_wall_t0 = time.perf_counter() if wall_profile_enabled else 0.0
    if bool(getattr(args, "profile_native_ops", False)):
        _sync_if_cuda(device)
        vsidecar_t0 = time.perf_counter()
    if (
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
        if partial_rank_takes_for_prefix
        else None
    )
    ranked_prefix_tokens_t: torch.Tensor | None = None
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
            allhead_ranked_prefix_t = allhead_rank_prefix_cache.get(rank_prefix_key)
            if allhead_ranked_prefix_t is None:
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
                        allhead_order_t = torch.topk(
                            allhead_dense_pq_scores_t[:num_heads, :indexed_count],
                            k=int(max_rank_take),
                            dim=1,
                            largest=True,
                            sorted=True,
                        ).indices
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
                allhead_rank_prefix_cache[rank_prefix_key] = allhead_ranked_prefix_t
            ranked_prefix_tokens_t = allhead_ranked_prefix_t[head_start_i:head_end_i]
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

    def mixed_scores_for_selected_batch(selected_t_i: torch.Tensor) -> torch.Tensor:
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
                y_indexed_prob_t.to(dtype=torch.float32)
                if y_indexed_prob_t is not None
                else torch.empty_like(pq_logits_t, dtype=torch.float32)
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
        score_vec = exact_scores_prob_t.clone()
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
    use_incremental_v_grid = _env_truthy("SELECTOR_PQ_JOINT_INCREMENTAL_V_GRID", "0")
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
        grid_take_counts: list[int] = []
        exact_full_budget_grid_flag = _env_truthy("SELECTOR_PQ_JOINT_EXACT_FULL_BUDGET_GRID", "1")
        native_score_grid_enabled = (
            _env_truthy("SELECTOR_PQ_JOINT_NATIVE_SCORE_GRID", "0")
            and not use_unsorted_k_prefix
        )
        fused_mixed_softmax_base_enabled = _env_truthy(
            "SELECTOR_PQ_JOINT_FUSED_SOFTMAX_BASE",
            "0",
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
            if grid_k_mb_by_idx is not None:
                grid_k_mb_by_idx.append(
                    float(selector_mb) + float(selected_len_i * int(self.head_dim) * key_bytes) / MB
                )
            if not native_score_grid_enabled:
                if selected_t_i is None:
                    selected_t_i = selected_for_budget_batch(int(k_budget))
                    grid_selected_by_ki[-1] = selected_t_i
                grid_score_rows.append(mixed_scores_for_selected_batch(selected_t_i))
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
                y_indexed_prob_t.to(dtype=torch.float32)
                if y_indexed_prob_t is not None
                else torch.empty_like(pq_logits_t, dtype=torch.float32)
            )
            use_score_grid_no_fill = _env_truthy("SELECTOR_PQ_JOINT_SCORE_GRID_NO_EXACT_FILL", "0")
            if use_score_grid_no_fill:
                if not bool(layout_covers_context):
                    raise RuntimeError(
                        "SELECTOR_PQ_JOINT_SCORE_GRID_NO_EXACT_FILL requires indexed tokens plus base "
                        "tokens to cover the full context"
                    )
            if fused_mixed_softmax_base_enabled:
                if use_score_grid_no_fill:
                    raise RuntimeError(
                        "SELECTOR_PQ_JOINT_FUSED_SOFTMAX_BASE does not support no-fill diagnostic mode"
                    )
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
                if use_grouped_score_grid_workspace:
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
            score_grid_t = torch.stack(grid_score_rows, dim=0)
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
        if use_grouped_softmax_base_cublas:
            if score_grid_t is None:
                raise RuntimeError("SELECTOR_PQ_JOINT_GROUPED_SOFTMAX_BASE_CUBLAS requires score_grid_t")
            native = load_selector_paged_pq_ext()
            if not hasattr(native, "joint_softmax_base_outputs_grouped_cublas"):
                raise RuntimeError(
                    "SELECTOR_PQ_JOINT_GROUPED_SOFTMAX_BASE_CUBLAS requires updated CUDA extension"
                )
        elif use_score_direct_vprefix:
            if score_grid_t is None:
                raise RuntimeError("SELECTOR_PQ_JOINT_SCORE_DIRECT_VPREFIX requires score_grid_t")
            native = load_selector_paged_pq_ext()
            if not hasattr(native, "joint_vprefix_outputs_from_grouped_scores_batched"):
                raise RuntimeError(
                    "SELECTOR_PQ_JOINT_SCORE_DIRECT_VPREFIX requires updated CUDA extension"
                )
        elif probs_grid_t is None and _env_truthy("SELECTOR_PQ_JOINT_NATIVE_SOFTMAX_BASE", "0"):
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
        if (
            not use_score_direct_vprefix
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
        if not use_score_direct_vprefix and not use_grouped_softmax_base_cublas and base_output_grid_t is None:
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
                "residual": residual_t.to(dtype=torch.float32).contiguous(),
                "code_error": code_error_t.to(dtype=torch.float32).contiguous(),
            }
            if use_score_direct_vprefix or use_grouped_softmax_base_cublas:
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
            )
        )
        grid_outputs_t = vprefix_result.grid_outputs
        grid_outputs_for_v_idx = vprefix_result.grid_outputs_for_v_idx
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
            use_incremental_v_grid=bool(use_incremental_v_grid),
            grid_outputs=grid_outputs_t,
            grid_outputs_for_v_idx=grid_outputs_for_v_idx,
            output_for_budget=output_for_budget_batch,
            k_artifacts=k_artifacts_batch,
            grid_k_mb_by_idx=grid_k_mb_by_idx,
            v_mb_by_idx=v_mb_by_idx,
            sim_start_seconds=float(sim_t0),
        )
    )
    final_ki_by_head = policy_result.final_ki_by_head
    final_vi_by_head = policy_result.final_vi_by_head
    final_idx_t_for_output = policy_result.final_idx_for_output
    final_output_grid_t = policy_result.final_output_grid

    return finalize_joint_head_outputs(
        JointFinalizeRuntime(
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
        )
    )
