#!/usr/bin/env python3
from __future__ import annotations

import sys
import os
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.selector_eval.gpu.run_gpu_paged_pq_eval import build_page_pq_gpu  # noqa: E402
from benchmark.selector_eval.runners.hf_paged_pq_intervention_common import (  # noqa: E402
    _choose_joint_kv_action,
)
from benchmark.selector_eval.runners.hf_paged_pq_intervention_geometric import (  # noqa: E402
    _gpu_gqa_base_logsumexp_decode,
    _gpu_gqa_dense_decode_ranked_logits_and_base_lse,
    _gpu_gqa_ranked_exact_logits,
    geometric_budget_pairs,
    select_thresholds_for_budget_counts_gpu,
    selected_mass_thresholds_from_logits_gpu,
)
from benchmark.selector_eval.runners.hf_paged_pq_intervention_value import (  # noqa: E402
    reconstruct_all_vpq_values_gpu,
    value_vpq_code_stat_risk_torch,
    value_vpq_pack_torch,
    vpq_values_for_tokens_gpu,
)
from benchmark.selector_eval.runners.run_layer_quality_eval import _vpq_values_for_tokens  # noqa: E402


def _test_native_exact_value_counts() -> None:
    from selector_paged_pq import (  # noqa: PLC0415
        gqa_causal_geometric_accept_counts,
        gqa_causal_geometric_accept_counts_vpq,
        gqa_causal_vpq_selected_tail_from_scores,
        gqa_causal_vpq_selected_tail_from_scores_counts,
        gqa_causal_vpq_selected_tail_from_scores_mass,
        gqa_causal_vpq_selected_tail_from_scores_mass_min,
        gqa_causal_vpq_selected_tail_attention,
        gqa_causal_vpq_tail_from_scores,
        gqa_causal_fullscan_pq_topk_fused_force,
        gqa_decode_vpq_selected_tail_agg_from_scores,
        gqa_decode_vpq_selected_tail_agg_from_scores_counts,
        gqa_decode_vpq_selected_tail_agg_from_scores_mass,
        gqa_decode_vpq_selected_tail_agg_from_scores_mass_min,
        gqa_decode_fullscan_vpq_selected_tail_agg,
        gqa_decode_fullscan_vpq_selected_tail_agg_mass_min,
        gqa_decode_scoreless_fullscan_vpq_tail,
        gqa_decode_geometric_accept_counts,
        gqa_decode_geometric_accept_counts_vpq,
        gqa_decode_geometric_accept_counts_vpq_tail_stability,
        gqa_decode_geometric_accept_counts_vpq_proxy,
        gqa_decode_geometric_accept_counts_vpq_mass_min_proxy_from_logits_thresholds,
        gqa_decode_geometric_output_vpq_mass_min_proxy_from_logits_thresholds,
        gqa_decode_ranked_exact_logits_with_base_lse,
        gqa_decode_vpq_selected_from_logits_mass_min,
        gqa_decode_vpq_selected_tail_agg_from_logits_mass_min,
        gqa_decode_vpq_selected_tail_agg_from_logits_mass_min_thresholds,
        gqa_fullscan_pq_topk_scores,
        joint_mixed_softmax_base_outputs,
        joint_mixed_softmax_base_outputs_rankpos,
        joint_mixed_softmax_base_outputs_tokenfit_scaled,
        joint_mixed_softmax_base_outputs_sparse_exact_tokenfit_scaled,
        joint_mixed_score_grid,
        joint_mixed_score_grid_workspace,
        joint_mixed_score_grid_nocalib_scatter_workspace,
        joint_mixed_score_grid_scaled,
        joint_mixed_score_grid_rankpos,
        joint_mixed_score_grid_no_exact_fill,
        joint_mixed_score_grid_no_exact_fill_scaled,
        joint_mixed_score_grid_tokenfit_scaled,
        joint_mixed_score_grid_no_exact_fill_tokenfit_scaled,
        joint_mixed_score_grid_sparse_exact_tokenfit_scaled,
        joint_sparse_exact_score_table,
        joint_grouped_accounting_sums,
        joint_grouped_accounting_accumulate,
        joint_rank_prefix_tokens,
        joint_rank_prefix_sort_temp_bytes,
        joint_rank_prefix_tokens_workspace,
        joint_budget_prefix_tokens,
        joint_select_policy,
        joint_select_policy_grouped_flat,
        joint_select_policy_grouped_flat_no_mb,
        joint_select_policy_grouped_flat_no_mb_accounting,
        joint_select_policy_grouped_flat_staged_no_mb,
        joint_select_policy_from_grouped_risk,
        joint_select_policy_from_grouped_risk_batched,
        joint_select_policy_from_grouped_risk_no_mb,
        joint_select_policy_from_grouped_risk_batched_no_mb,
        joint_select_policy_from_grouped_risk_intervals_batched_no_mb,
        joint_select_policy_from_grouped_scores_intervals_batched_no_mb,
        joint_select_policy_from_grouped_scores_probs_intervals_batched_no_mb,
        joint_select_policy_from_grouped_scores_topk_intervals_batched_no_mb,
        joint_mixed_select_policy_intervals_no_mb,
        joint_mixed_select_policy_intervals_rankpos_no_calib_no_mb,
        joint_mixed_select_policy_merge_rankpos_no_calib_no_mb,
        joint_vprefix_outputs,
        joint_vprefix_outputs_precision,
        joint_vprefix_outputs_precision_from_risk,
        joint_vprefix_outputs_from_grouped_risk,
        joint_vprefix_outputs_from_grouped_merge_risk_batched,
        joint_vprefix_outputs_from_grouped_risk_batched,
        joint_vprefix_outputs_from_grouped_risk_batched_strided_workspace,
        joint_vprefix_outputs_from_grouped_risk_batched_workspace,
        joint_vprefix_outputs_from_grouped_risk_topk_batched,
        joint_vprefix_outputs_from_grouped_scores_batched,
        joint_vprefix_outputs_from_grouped_scores_batched_workspace,
        joint_vprefix_outputs_from_risk,
        joint_grouped_risk_sort_temp_bytes,
        joint_softmax_base_outputs,
        joint_softmax_base_outputs_cublas,
        joint_softmax_base_outputs_strided_workspace,
        joint_softmax_base_outputs_workspace,
        joint_softmax_base_outputs_workspace_cublas,
        joint_softmax_base_outputs_grouped,
        joint_softmax_base_outputs_grouped_cublas,
        joint_vpq_append_exact_suffix,
        joint_vpq_append_exact_suffix_grouped,
        joint_vpq_sidecars_from_pack,
        joint_vpq_base_outputs_from_probs,
        selected_mass_thresholds_from_topk,
    )

    torch.manual_seed(20260517)
    device = torch.device("cuda")
    positions = 3
    heads = 2
    kv_heads = 1
    dim = 16
    pages = 3
    page_size = 8
    ranked = 4
    value_subvecs = 2
    value_centroids = 4
    value_subdim = dim // value_subvecs
    total_tokens = pages * page_size + 4
    query_start = 20
    query_context_len = 23
    static_prefix = 1
    static_suffix = 1
    policy_ids = {
        "k_first_priority": 0,
        "v_first_priority": 1,
        "k_first_alternating": 2,
        "v_first_alternating": 3,
        "sensitivity_greedy": 4,
    }

    queries = torch.randn((positions, heads, dim), device=device, dtype=torch.float32)
    keys = torch.randn((kv_heads, total_tokens, dim), device=device, dtype=torch.float16)
    values = torch.randn((kv_heads, total_tokens, dim), device=device, dtype=torch.float16)
    dense_pq_scores = torch.randn((positions, heads, pages * page_size), device=device, dtype=torch.float32)
    value_codebooks = torch.randn(
        (kv_heads, pages, value_subvecs, value_centroids, value_subdim),
        device=device,
        dtype=torch.float32,
    )
    value_codes = torch.randint(
        0,
        value_centroids,
        (kv_heads, pages, page_size, value_subvecs),
        device=device,
        dtype=torch.uint8,
    )
    page_starts = torch.tensor([1, 9, 17], device=device, dtype=torch.long)
    ranked_tokens = torch.tensor(
        [
            [[2, 9, 18, 20], [3, 10, 19, 21]],
            [[2, 9, 18, 20], [3, 10, 19, 21]],
            [[2, 9, 18, 20], [3, 10, 19, 21]],
        ],
        device=device,
        dtype=torch.long,
    )
    ranked_scores = torch.randn((positions, heads, ranked), device=device, dtype=torch.float32)
    counts = torch.ones((positions, heads), device=device, dtype=torch.long)
    scale = float(dim) ** -0.5

    k_count = 3
    v_steps = 4
    max_exact = 6
    base_outputs = torch.randn((k_count, heads, dim), device=device, dtype=torch.float32)
    probs = torch.softmax(torch.randn((k_count, heads, query_context_len), device=device, dtype=torch.float32), dim=2)
    residual = torch.randn((query_context_len, dim), device=device, dtype=torch.float32)
    risk = torch.randn((k_count, heads, query_context_len), device=device, dtype=torch.float32)
    code_error = torch.rand((query_context_len,), device=device, dtype=torch.float32)
    exact_order = torch.topk(risk, k=max_exact, dim=2, largest=True, sorted=True).indices
    v_budgets = torch.tensor([0, 1, 3, max_exact], device=device, dtype=torch.long)
    got_prefix = joint_vprefix_outputs(base_outputs, probs, residual, exact_order, v_budgets)
    gathered_probs = torch.gather(probs, 2, exact_order)
    gathered_residual = residual.index_select(0, exact_order.reshape(-1)).reshape(k_count, heads, max_exact, dim)
    prefix = torch.cumsum(gathered_probs.reshape(k_count, heads, max_exact, 1) * gathered_residual, dim=2)
    ref_by_v = []
    for budget in v_budgets.detach().cpu().tolist():
        exact = max(0, min(int(budget), max_exact, query_context_len))
        if exact > 0:
            ref_by_v.append(base_outputs + prefix[:, :, exact - 1, :])
        else:
            ref_by_v.append(base_outputs)
    ref_prefix = torch.stack(ref_by_v, dim=1)
    if not torch.allclose(got_prefix, ref_prefix, atol=2e-5, rtol=2e-5):
        raise AssertionError("native joint V-prefix output grid mismatches Torch reference")
    residual_lo = torch.randn_like(residual)
    commit_mask = torch.rand((query_context_len,), device=device) > 0.35
    v_hi_counts = torch.tensor([0, 1, 1, 2], device=device, dtype=torch.long)
    got_precision, got_lo_reads = joint_vprefix_outputs_precision(
        base_outputs,
        probs,
        residual,
        residual_lo,
        commit_mask,
        exact_order,
        v_budgets,
        v_hi_counts,
    )
    gathered_lo = residual_lo.index_select(0, exact_order.reshape(-1)).reshape(
        k_count, heads, max_exact, dim
    )
    hi_prefix = torch.cumsum(
        gathered_probs.reshape(k_count, heads, max_exact, 1) * gathered_residual,
        dim=2,
    )
    lo_prefix = torch.cumsum(
        gathered_probs.reshape(k_count, heads, max_exact, 1) * gathered_lo,
        dim=2,
    )
    commit_prefix = torch.cumsum(
        commit_mask.to(torch.int32).index_select(0, exact_order.reshape(-1)).reshape(
            k_count, heads, max_exact
        ),
        dim=2,
    )
    precision_ref = []
    reads_ref = []
    for budget, hi in zip(
        v_budgets.detach().cpu().tolist(),
        v_hi_counts.detach().cpu().tolist(),
        strict=True,
    ):
        exact = max(0, min(int(budget), max_exact, query_context_len))
        hi = max(0, min(int(hi), exact))
        if exact <= 0:
            precision_ref.append(base_outputs)
            reads_ref.append(torch.zeros((k_count, heads), device=device, dtype=torch.int32))
            continue
        delta = hi_prefix[:, :, hi - 1, :]
        reads = torch.zeros((k_count, heads), device=device, dtype=torch.int32)
        if exact > hi:
            delta = (delta + lo_prefix[:, :, exact - 1, :]) - lo_prefix[:, :, hi - 1, :]
            reads = commit_prefix[:, :, exact - 1] - commit_prefix[:, :, hi - 1]
        precision_ref.append(base_outputs + delta)
        reads_ref.append(reads)
    precision_ref_t = torch.stack(precision_ref, dim=1)
    reads_ref_t = torch.stack(reads_ref, dim=1)
    if not torch.allclose(got_precision, precision_ref_t, atol=5e-5, rtol=5e-5):
        raise AssertionError("native progressive-precision V-prefix grid mismatches Torch reference")
    if not torch.equal(got_lo_reads, reads_ref_t):
        raise AssertionError("native progressive-precision V-prefix read counts mismatch")
    got_prefix_from_risk = joint_vprefix_outputs_from_risk(
        base_outputs,
        probs,
        residual,
        code_error,
        v_budgets,
    )
    got_precision_from_risk, got_precision_reads_from_risk = (
        joint_vprefix_outputs_precision_from_risk(
            base_outputs,
            probs,
            residual,
            residual_lo,
            commit_mask,
            code_error,
            v_budgets,
            v_hi_counts,
        )
    )
    risk_for_sort = (probs * probs) * code_error.reshape(1, 1, -1)
    exact_order_for_sort = torch.topk(
        risk_for_sort,
        k=query_context_len,
        dim=2,
        largest=True,
        sorted=True,
    ).indices
    gathered_probs_for_sort = torch.gather(probs, 2, exact_order_for_sort)
    gathered_residual_for_sort = residual.index_select(0, exact_order_for_sort.reshape(-1)).reshape(
        k_count,
        heads,
        query_context_len,
        dim,
    )
    prefix_for_sort = torch.cumsum(
        gathered_probs_for_sort.reshape(k_count, heads, query_context_len, 1) * gathered_residual_for_sort,
        dim=2,
    )
    ref_by_v_from_risk = []
    for budget in v_budgets.detach().cpu().tolist():
        exact = max(0, min(int(budget), query_context_len))
        if exact > 0:
            ref_by_v_from_risk.append(base_outputs + prefix_for_sort[:, :, exact - 1, :])
        else:
            ref_by_v_from_risk.append(base_outputs)
    ref_prefix_from_risk = torch.stack(ref_by_v_from_risk, dim=1)
    if not torch.allclose(got_prefix_from_risk, ref_prefix_from_risk, atol=2e-5, rtol=2e-5):
        raise AssertionError("native joint V-prefix-from-risk output grid mismatches Torch reference")
    precision_risk_ref, precision_risk_reads_ref = joint_vprefix_outputs_precision(
        base_outputs,
        probs,
        residual,
        residual_lo,
        commit_mask,
        exact_order_for_sort,
        v_budgets,
        v_hi_counts,
    )
    if not torch.allclose(
        got_precision_from_risk,
        precision_risk_ref,
        atol=5e-5,
        rtol=5e-5,
    ):
        raise AssertionError("native progressive-precision risk sort output mismatch")
    if not torch.equal(got_precision_reads_from_risk, precision_risk_reads_ref):
        raise AssertionError("native progressive-precision risk sort read-count mismatch")

    base_context = 10
    base_pages = 2
    base_page_size = 4
    base_codes_n = 4
    base_dim = dim
    base_probs = torch.softmax(
        torch.randn((k_count, heads, base_context), device=device, dtype=torch.float32),
        dim=2,
    )
    base_values = torch.randn((base_context, base_dim), device=device, dtype=torch.float16)
    base_codebooks = torch.randn((base_pages, 1, base_codes_n, base_dim), device=device, dtype=torch.float32)
    base_codes = torch.randint(0, base_codes_n, (base_pages, base_page_size, 1), device=device, dtype=torch.uint8)
    base_page_starts = torch.tensor([1, 5], device=device, dtype=torch.long)
    fallback_tokens = torch.tensor([0, 9], device=device, dtype=torch.long)
    got_vpq_base = joint_vpq_base_outputs_from_probs(
        base_probs,
        base_values,
        base_codebooks,
        base_codes,
        base_page_starts,
        fallback_tokens,
    )
    vhat_ref = base_values.float().clone()
    for page in range(base_pages):
        start = int(base_page_starts[page].item())
        for row in range(base_page_size):
            token = start + row
            code = int(base_codes[page, row, 0].item())
            vhat_ref[token] = base_codebooks[page, 0, code]
    ref_vpq_base = (
        base_probs.reshape(k_count * heads, base_context) @ vhat_ref
    ).reshape(k_count, heads, base_dim)
    if not torch.allclose(got_vpq_base, ref_vpq_base, atol=3e-5, rtol=3e-5):
        raise AssertionError("native V-PQ base output aggregation mismatches Torch reference")

    softmax_score_grid = torch.randn((k_count, heads, base_context), device=device, dtype=torch.float32)
    softmax_values = torch.randn((base_context, base_dim), device=device, dtype=torch.float32)
    got_softmax_probs, got_softmax_base = joint_softmax_base_outputs(softmax_score_grid, softmax_values)
    got_softmax_probs_cb, got_softmax_base_cb = joint_softmax_base_outputs_cublas(softmax_score_grid, softmax_values)
    probs_workspace = torch.empty_like(got_softmax_probs)
    base_workspace = torch.empty_like(got_softmax_base)
    got_softmax_probs_ws, got_softmax_base_ws = joint_softmax_base_outputs_workspace(
        probs_workspace,
        base_workspace,
        softmax_score_grid,
        softmax_values,
    )
    probs_workspace_strided = torch.empty(
        (k_count, heads, base_context + 7),
        device=device,
        dtype=torch.float32,
    )
    base_workspace_strided = torch.empty_like(got_softmax_base)
    got_softmax_probs_strided, got_softmax_base_strided = joint_softmax_base_outputs_strided_workspace(
        probs_workspace_strided,
        base_workspace_strided,
        softmax_score_grid,
        softmax_values,
    )
    probs_workspace_cb = torch.empty_like(got_softmax_probs)
    base_workspace_cb = torch.empty_like(got_softmax_base)
    got_softmax_probs_ws_cb, got_softmax_base_ws_cb = joint_softmax_base_outputs_workspace_cublas(
        probs_workspace_cb,
        base_workspace_cb,
        softmax_score_grid,
        softmax_values,
    )
    ref_softmax_probs = torch.softmax(softmax_score_grid, dim=2)
    ref_softmax_base = (
        ref_softmax_probs.reshape(k_count * heads, base_context) @ softmax_values
    ).reshape(k_count, heads, base_dim)
    if not torch.allclose(got_softmax_probs, ref_softmax_probs, atol=3e-6, rtol=3e-6):
        raise AssertionError("native joint softmax probabilities mismatch Torch reference")
    if not torch.allclose(got_softmax_base, ref_softmax_base, atol=2e-5, rtol=2e-5):
        raise AssertionError("native joint softmax base output mismatch Torch reference")
    if not torch.allclose(got_softmax_probs_cb, ref_softmax_probs, atol=3e-6, rtol=3e-6):
        raise AssertionError("cuBLAS joint softmax probabilities mismatch Torch reference")
    if not torch.allclose(got_softmax_base_cb, ref_softmax_base, atol=2e-5, rtol=2e-5):
        raise AssertionError("cuBLAS joint softmax base output mismatch Torch reference")
    if got_softmax_probs_ws.data_ptr() != probs_workspace.data_ptr():
        raise AssertionError("workspace softmax probabilities did not reuse caller workspace")
    if got_softmax_base_ws.data_ptr() != base_workspace.data_ptr():
        raise AssertionError("workspace softmax base output did not reuse caller workspace")
    if not torch.allclose(got_softmax_probs_ws, ref_softmax_probs, atol=3e-6, rtol=3e-6):
        raise AssertionError("workspace joint softmax probabilities mismatch Torch reference")
    if not torch.allclose(got_softmax_base_ws, ref_softmax_base, atol=2e-5, rtol=2e-5):
        raise AssertionError("workspace joint softmax base output mismatch Torch reference")
    if got_softmax_probs_strided.data_ptr() != probs_workspace_strided.data_ptr():
        raise AssertionError("strided workspace softmax probabilities did not reuse caller workspace")
    if got_softmax_base_strided.data_ptr() != base_workspace_strided.data_ptr():
        raise AssertionError("strided workspace softmax base output did not reuse caller workspace")
    if not torch.allclose(got_softmax_probs_strided, ref_softmax_probs, atol=3e-6, rtol=3e-6):
        raise AssertionError("strided workspace joint softmax probabilities mismatch Torch reference")
    if not torch.allclose(got_softmax_base_strided, ref_softmax_base, atol=2e-5, rtol=2e-5):
        raise AssertionError("strided workspace joint softmax base output mismatch Torch reference")
    if got_softmax_probs_ws_cb.data_ptr() != probs_workspace_cb.data_ptr():
        raise AssertionError("cuBLAS workspace softmax probabilities did not reuse caller workspace")
    if got_softmax_base_ws_cb.data_ptr() != base_workspace_cb.data_ptr():
        raise AssertionError("cuBLAS workspace softmax base output did not reuse caller workspace")
    if not torch.allclose(got_softmax_probs_ws_cb, ref_softmax_probs, atol=3e-6, rtol=3e-6):
        raise AssertionError("cuBLAS workspace joint softmax probabilities mismatch Torch reference")
    if not torch.allclose(got_softmax_base_ws_cb, ref_softmax_base, atol=2e-5, rtol=2e-5):
        raise AssertionError("cuBLAS workspace joint softmax base output mismatch Torch reference")
    grouped_softmax_scores = torch.randn((kv_heads + 1, k_count, heads, base_context), device=device, dtype=torch.float32)
    grouped_softmax_values = torch.randn((kv_heads + 1, base_context + 3, base_dim), device=device, dtype=torch.float32)
    got_grouped_probs, got_grouped_base = joint_softmax_base_outputs_grouped(
        grouped_softmax_scores,
        grouped_softmax_values,
    )
    got_grouped_probs_cb, got_grouped_base_cb = joint_softmax_base_outputs_grouped_cublas(
        grouped_softmax_scores,
        grouped_softmax_values,
    )
    ref_grouped_probs = torch.softmax(grouped_softmax_scores, dim=3)
    ref_grouped_base = torch.stack(
        [
            (
                ref_grouped_probs[g].reshape(k_count * heads, base_context)
                @ grouped_softmax_values[g, :base_context]
            ).reshape(k_count, heads, base_dim)
            for g in range(kv_heads + 1)
        ],
        dim=0,
    )
    if not torch.allclose(got_grouped_probs, ref_grouped_probs, atol=3e-6, rtol=3e-6):
        raise AssertionError("grouped native joint softmax probabilities mismatch Torch reference")
    if not torch.allclose(got_grouped_base, ref_grouped_base, atol=2e-5, rtol=2e-5):
        raise AssertionError("grouped native joint softmax base output mismatch Torch reference")
    if not torch.allclose(got_grouped_probs_cb, ref_grouped_probs, atol=3e-6, rtol=3e-6):
        raise AssertionError("grouped cuBLAS joint softmax probabilities mismatch Torch reference")
    if not torch.allclose(got_grouped_base_cb, ref_grouped_base, atol=2e-5, rtol=2e-5):
        raise AssertionError("grouped cuBLAS joint softmax base output mismatch Torch reference")

    append_values = torch.randn((18, dim), device=device, dtype=torch.float16).contiguous()
    append_vhat = torch.randn((24, dim), device=device, dtype=torch.float32)
    append_residual = torch.randn((24, dim), device=device, dtype=torch.float32)
    append_error = torch.rand((24,), device=device, dtype=torch.float64)
    ref_vhat = append_vhat.clone()
    ref_residual = append_residual.clone()
    ref_error = append_error.clone()
    append_start, append_end = 7, 15
    ref_vhat[append_start:append_end].copy_(append_values[append_start:append_end].float())
    ref_residual[append_start:append_end].zero_()
    ref_error[append_start:append_end].zero_()
    joint_vpq_append_exact_suffix(
        append_vhat,
        append_residual,
        append_error,
        append_values,
        append_start,
        append_end,
    )
    if not torch.equal(append_vhat, ref_vhat):
        raise AssertionError("joint_vpq_append_exact_suffix vhat mismatch")
    if not torch.equal(append_residual, ref_residual):
        raise AssertionError("joint_vpq_append_exact_suffix residual mismatch")
    if not torch.equal(append_error, ref_error):
        raise AssertionError("joint_vpq_append_exact_suffix code_error mismatch")

    grouped_values = torch.randn((kv_heads + 1, 18, dim), device=device, dtype=torch.float16).contiguous()
    grouped_vhat = torch.randn((kv_heads + 1, 24, dim), device=device, dtype=torch.float32)
    grouped_residual_append = torch.randn((kv_heads + 1, 24, dim), device=device, dtype=torch.float32)
    grouped_error = torch.rand((kv_heads + 1, 24), device=device, dtype=torch.float32)
    ref_grouped_vhat = grouped_vhat.clone()
    ref_grouped_residual = grouped_residual_append.clone()
    ref_grouped_error = grouped_error.clone()
    ref_grouped_vhat[:, append_start:append_end, :].copy_(
        grouped_values[:, append_start:append_end, :].float()
    )
    ref_grouped_residual[:, append_start:append_end, :].zero_()
    ref_grouped_error[:, append_start:append_end].zero_()
    joint_vpq_append_exact_suffix_grouped(
        grouped_vhat,
        grouped_residual_append,
        grouped_error,
        grouped_values,
        append_start,
        append_end,
    )
    if not torch.equal(grouped_vhat, ref_grouped_vhat):
        raise AssertionError("joint_vpq_append_exact_suffix_grouped vhat mismatch")
    if not torch.equal(grouped_residual_append, ref_grouped_residual):
        raise AssertionError("joint_vpq_append_exact_suffix_grouped residual mismatch")
    if not torch.equal(grouped_error, ref_grouped_error):
        raise AssertionError("joint_vpq_append_exact_suffix_grouped code_error mismatch")

    rank_scores = torch.tensor(
        [[0.2, 1.5, -0.1, 0.7], [3.0, 2.0, 4.0, 1.0]],
        device=device,
        dtype=torch.float32,
    )
    rank_tokens = torch.tensor([10, 11, 12, 13], device=device, dtype=torch.long)
    got_rank_prefix = joint_rank_prefix_tokens(rank_scores, rank_tokens, 3)
    ref_rank_prefix = rank_tokens.index_select(
        0,
        torch.topk(rank_scores, k=3, dim=1, largest=True, sorted=True).indices.reshape(-1),
    ).reshape(2, 3)
    if not torch.equal(got_rank_prefix, ref_rank_prefix):
        raise AssertionError("native joint rank-prefix tokens mismatch Torch topk reference")
    rank_rows = int(rank_scores.shape[0])
    rank_count = int(rank_scores.shape[1])
    rank_take = 3
    temp_bytes = max(1, int(joint_rank_prefix_sort_temp_bytes(rank_rows, rank_count)))
    workspace_rank_prefix = joint_rank_prefix_tokens_workspace(
        rank_scores,
        rank_tokens,
        rank_take,
        torch.empty((rank_rows * rank_count,), device=device, dtype=torch.float32),
        torch.empty((rank_rows * rank_count,), device=device, dtype=torch.float32),
        torch.empty((rank_rows * rank_count,), device=device, dtype=torch.int32),
        torch.empty((rank_rows * rank_count,), device=device, dtype=torch.int32),
        torch.empty((rank_rows + 1,), device=device, dtype=torch.long),
        torch.empty((temp_bytes,), device=device, dtype=torch.uint8),
        torch.empty((rank_rows * rank_take,), device=device, dtype=torch.long),
    )
    if not torch.equal(workspace_rank_prefix, ref_rank_prefix):
        raise AssertionError("workspace native joint rank-prefix tokens mismatch Torch topk reference")
    budget_prefix = joint_budget_prefix_tokens(
        rank_scores,
        rank_tokens,
        torch.tensor([1, 3], device=device, dtype=torch.long),
        3,
    )
    ref_top1 = set(
        rank_tokens.index_select(
            0,
            torch.topk(rank_scores, k=1, dim=1, largest=True, sorted=True).indices[0],
        ).detach().cpu().tolist()
    )
    got_top1 = set(budget_prefix[0, :1].detach().cpu().tolist())
    if got_top1 != ref_top1:
        raise AssertionError(f"budget-prefix top1 mismatch: got={got_top1} ref={ref_top1}")
    ref_top3 = set(ref_rank_prefix[0, :3].detach().cpu().tolist())
    got_top3 = set(budget_prefix[0, :3].detach().cpu().tolist())
    if got_top3 != ref_top3:
        raise AssertionError(f"budget-prefix top3 mismatch: got={got_top3} ref={ref_top3}")
    unique_scores = (torch.randn((3, 257), device=device, dtype=torch.float32) * 0.01) + torch.arange(
        257,
        device=device,
        dtype=torch.float32,
    ).reshape(1, -1)
    unique_tokens = torch.arange(1000, 1257, device=device, dtype=torch.long)
    unique_budgets = torch.tensor([7, 31, 113], device=device, dtype=torch.long)
    got_unique_budget_prefix = joint_budget_prefix_tokens(unique_scores, unique_tokens, unique_budgets, 113)
    for row_i in range(int(unique_scores.shape[0])):
        for take_i in unique_budgets.detach().cpu().tolist():
            ref_set = set(
                unique_tokens.index_select(
                    0,
                    torch.topk(unique_scores[row_i], k=int(take_i), largest=True, sorted=True).indices,
                ).detach().cpu().tolist()
            )
            got_set = set(got_unique_budget_prefix[row_i, : int(take_i)].detach().cpu().tolist())
            if got_set != ref_set:
                raise AssertionError(
                    f"budget-prefix unique top{take_i} set mismatch row={row_i}: "
                    f"missing={sorted(ref_set - got_set)[:5]} extra={sorted(got_set - ref_set)[:5]}"
                )

    grouped_rows = 7
    grouped_groups = 3
    grouped_context = 11
    grouped_dim = 12
    grouped_v_budgets = torch.tensor([0, 1, 4, grouped_context], device=device, dtype=torch.long)
    grouped_base = torch.randn((grouped_rows, grouped_dim), device=device, dtype=torch.float32)
    grouped_probs = torch.softmax(
        torch.randn((grouped_rows, grouped_context), device=device, dtype=torch.float32),
        dim=1,
    )
    grouped_residual = torch.randn((grouped_groups, grouped_context, grouped_dim), device=device, dtype=torch.float32)
    grouped_error = torch.rand((grouped_groups, grouped_context), device=device, dtype=torch.float32)
    row_group_ids = torch.tensor([0, 1, 1, 2, 0, 2, 1], device=device, dtype=torch.long)
    got_grouped = joint_vprefix_outputs_from_grouped_risk(
        grouped_base,
        grouped_probs,
        grouped_residual,
        grouped_error,
        row_group_ids,
        grouped_v_budgets,
    )
    grouped_ref_rows = []
    for row in range(grouped_rows):
        group = int(row_group_ids[row].item())
        row_risk = grouped_probs[row] * grouped_probs[row] * grouped_error[group]
        order = torch.topk(row_risk, k=grouped_context, largest=True, sorted=True).indices
        row_prefix = torch.cumsum(
            grouped_probs[row, order].reshape(grouped_context, 1) * grouped_residual[group, order],
            dim=0,
        )
        row_by_v = []
        for budget in grouped_v_budgets.detach().cpu().tolist():
            exact = max(0, min(int(budget), grouped_context))
            if exact > 0:
                row_by_v.append(grouped_base[row] + row_prefix[exact - 1])
            else:
                row_by_v.append(grouped_base[row])
        grouped_ref_rows.append(torch.stack(row_by_v, dim=0))
    ref_grouped = torch.stack(grouped_ref_rows, dim=0)
    if not torch.allclose(got_grouped, ref_grouped, atol=2e-5, rtol=2e-5):
        raise AssertionError("native grouped joint V-prefix-from-risk output grid mismatches Torch reference")

    fused_groups = 3
    fused_k = 4
    fused_heads = 2
    fused_context = 17
    fused_dim = 16
    fused_v_budgets = torch.tensor([0, 2, 5, 9], device=device, dtype=torch.long)
    fused_rows = fused_groups * fused_k * fused_heads
    fused_base = torch.randn((fused_rows, fused_dim), device=device, dtype=torch.float32)
    fused_probs = torch.softmax(
        torch.randn((fused_rows, fused_context), device=device, dtype=torch.float32),
        dim=1,
    )
    fused_residual = torch.randn((fused_groups, fused_context, fused_dim), device=device, dtype=torch.float32)
    fused_error = torch.rand((fused_groups, fused_context), device=device, dtype=torch.float32)
    fused_row_group_ids = torch.arange(fused_groups, dtype=torch.long, device=device).repeat_interleave(
        fused_k * fused_heads
    )
    fused_grid_flat = joint_vprefix_outputs_from_grouped_risk(
        fused_base,
        fused_probs,
        fused_residual,
        fused_error,
        fused_row_group_ids,
        fused_v_budgets,
    )
    fused_grid_batched = joint_vprefix_outputs_from_grouped_risk_batched(
        fused_base.reshape(fused_groups, fused_k, fused_heads, fused_dim).contiguous(),
        fused_probs.reshape(fused_groups, fused_k, fused_heads, fused_context).contiguous(),
        fused_residual,
        fused_error,
        fused_v_budgets,
    )
    if not torch.allclose(fused_grid_batched, fused_grid_flat, atol=2e-5, rtol=2e-5):
        raise AssertionError("native batched grouped-risk V-prefix output grid mismatches flat reference")
    score_fused_groups = 3
    score_fused_k = 4
    score_fused_heads = 2
    score_fused_context = 17
    score_fused_dim = 16
    score_fused_v_budgets = torch.tensor([0, 2, 5, 9], device=device, dtype=torch.long)
    score_fused_scores = torch.randn(
        (score_fused_groups, score_fused_k, score_fused_heads, score_fused_context),
        device=device,
        dtype=torch.float32,
    )
    score_fused_probs = torch.softmax(score_fused_scores, dim=3)
    score_fused_vhat = torch.randn(
        (score_fused_groups, score_fused_context, score_fused_dim),
        device=device,
        dtype=torch.float32,
    )
    score_fused_residual = torch.randn_like(score_fused_vhat)
    score_fused_error = torch.rand(
        (score_fused_groups, score_fused_context),
        device=device,
        dtype=torch.float32,
    )
    score_fused_base = torch.einsum("gkhc,gcd->gkhd", score_fused_probs, score_fused_vhat).contiguous()
    score_ref = joint_vprefix_outputs_from_grouped_risk_batched(
        score_fused_base,
        score_fused_probs.contiguous(),
        score_fused_residual,
        score_fused_error,
        score_fused_v_budgets,
    )
    score_got = joint_vprefix_outputs_from_grouped_scores_batched(
        score_fused_scores,
        score_fused_vhat,
        score_fused_residual,
        score_fused_error,
        score_fused_v_budgets,
    )
    if not torch.allclose(score_got, score_ref, atol=5e-5, rtol=5e-5):
        max_diff = float((score_got - score_ref).abs().max().item())
        raise AssertionError(
            "native score-direct grouped-risk V-prefix output grid mismatches "
            f"probability-materialized reference; max_diff={max_diff}"
        )
    score_rows = score_fused_groups * score_fused_k * score_fused_heads
    score_temp_bytes = max(1, int(joint_grouped_risk_sort_temp_bytes(score_rows, score_fused_context)))
    score_workspace_got = joint_vprefix_outputs_from_grouped_scores_batched_workspace(
        score_fused_scores,
        score_fused_vhat,
        score_fused_residual,
        score_fused_error,
        score_fused_v_budgets,
        torch.empty((score_rows,), device=device, dtype=torch.float32),
        torch.empty((score_rows,), device=device, dtype=torch.float32),
        torch.empty((score_rows, score_fused_dim), device=device, dtype=torch.float32),
        torch.empty((score_rows, score_fused_context), device=device, dtype=torch.float32),
        torch.empty((score_rows, score_fused_context), device=device, dtype=torch.float32),
        torch.empty((score_rows, score_fused_context), device=device, dtype=torch.int32),
        torch.empty((score_rows, score_fused_context), device=device, dtype=torch.int32),
        torch.empty((score_rows + 1,), device=device, dtype=torch.long),
        torch.empty((score_temp_bytes,), device=device, dtype=torch.uint8),
        torch.empty((score_rows, int(score_fused_v_budgets.numel()), score_fused_dim), device=device, dtype=torch.float32),
        torch.empty((score_rows, int(score_fused_v_budgets.numel()), score_fused_dim), device=device, dtype=torch.float32),
    )
    if not torch.allclose(score_workspace_got, score_ref, atol=5e-5, rtol=5e-5):
        max_diff = float((score_workspace_got - score_ref).abs().max().item())
        raise AssertionError(
            "workspace native score-direct grouped-risk V-prefix output grid mismatches "
            f"probability-materialized reference; max_diff={max_diff}"
        )
    fused_grid_topk_batched = joint_vprefix_outputs_from_grouped_risk_topk_batched(
        fused_base.reshape(fused_groups, fused_k, fused_heads, fused_dim).contiguous(),
        fused_probs.reshape(fused_groups, fused_k, fused_heads, fused_context).contiguous(),
        fused_residual,
        fused_error,
        fused_v_budgets,
        int(fused_v_budgets.max().item()),
    )
    if not torch.allclose(fused_grid_topk_batched, fused_grid_flat, atol=2e-5, rtol=2e-5):
        raise AssertionError("native top-k grouped-risk V-prefix output grid mismatches full-sort reference")
    fused_k_mb = torch.stack(
        [
            torch.linspace(0.25 + 0.01 * group, 1.75 + 0.01 * group, fused_k, device=device)
            for group in range(fused_groups)
        ],
        dim=0,
    ).to(torch.float32)
    fused_v_mb = torch.stack(
        [
            torch.linspace(0.10 + 0.02 * group, 1.10 + 0.02 * group, int(fused_v_budgets.numel()), device=device)
            for group in range(fused_groups)
        ],
        dim=0,
    ).to(torch.float32)
    for policy_name, policy_id in policy_ids.items():
        ref_outputs, ref_indices = joint_select_policy_grouped_flat(
            fused_grid_flat,
            fused_k_mb,
            fused_v_mb,
            fused_k,
            fused_heads,
            0.25,
            policy_id,
        )
        got_outputs, got_indices = joint_select_policy_from_grouped_risk(
            fused_base,
            fused_probs,
            fused_residual,
            fused_error,
            fused_v_budgets,
            fused_k_mb,
            fused_v_mb,
            fused_k,
            fused_heads,
            0.25,
            policy_id,
        )
        if not torch.equal(got_indices, ref_indices):
            raise AssertionError(f"native fused grouped-risk policy indices mismatch; policy={policy_name}")
        if not torch.allclose(got_outputs, ref_outputs, atol=2e-5, rtol=2e-5):
            raise AssertionError(f"native fused grouped-risk policy outputs mismatch; policy={policy_name}")
        got_batched_outputs, got_batched_indices = joint_select_policy_from_grouped_risk_batched(
            fused_base.reshape(fused_groups, fused_k, fused_heads, fused_dim).contiguous(),
            fused_probs.reshape(fused_groups, fused_k, fused_heads, fused_context).contiguous(),
            fused_residual,
            fused_error,
            fused_v_budgets,
            fused_k_mb,
            fused_v_mb,
            0.25,
            policy_id,
        )
        if not torch.equal(got_batched_indices, ref_indices):
            raise AssertionError(f"native batched grouped-risk policy indices mismatch; policy={policy_name}")
        if not torch.allclose(got_batched_outputs, ref_outputs, atol=2e-5, rtol=2e-5):
            raise AssertionError(f"native batched grouped-risk policy outputs mismatch; policy={policy_name}")
        if policy_name != "sensitivity_greedy":
            got_no_mb_outputs, got_no_mb_indices = joint_select_policy_from_grouped_risk_no_mb(
                fused_base,
                fused_probs,
                fused_residual,
                fused_error,
                fused_v_budgets,
                fused_k,
                fused_heads,
                0.25,
                policy_id,
            )
            if not torch.equal(got_no_mb_indices, ref_indices):
                raise AssertionError(f"native no-MB grouped-risk policy indices mismatch; policy={policy_name}")
            if not torch.allclose(got_no_mb_outputs, ref_outputs, atol=2e-5, rtol=2e-5):
                raise AssertionError(f"native no-MB grouped-risk policy outputs mismatch; policy={policy_name}")
            got_batched_no_mb_outputs, got_batched_no_mb_indices = joint_select_policy_from_grouped_risk_batched_no_mb(
                fused_base.reshape(fused_groups, fused_k, fused_heads, fused_dim).contiguous(),
                fused_probs.reshape(fused_groups, fused_k, fused_heads, fused_context).contiguous(),
                fused_residual,
                fused_error,
                fused_v_budgets,
                0.25,
                policy_id,
            )
            if not torch.equal(got_batched_no_mb_indices, ref_indices):
                raise AssertionError(f"native batched no-MB grouped-risk policy indices mismatch; policy={policy_name}")
            if not torch.allclose(got_batched_no_mb_outputs, ref_outputs, atol=2e-5, rtol=2e-5):
                raise AssertionError(f"native batched no-MB grouped-risk policy outputs mismatch; policy={policy_name}")
            got_interval_outputs, got_interval_indices = (
                joint_select_policy_from_grouped_risk_intervals_batched_no_mb(
                    fused_base.reshape(fused_groups, fused_k, fused_heads, fused_dim).contiguous(),
                    fused_probs.reshape(fused_groups, fused_k, fused_heads, fused_context).contiguous(),
                    fused_residual,
                    fused_error,
                    fused_v_budgets,
                    0.25,
                    policy_id,
                )
            )
            if not torch.equal(got_interval_indices, ref_indices):
                raise AssertionError(f"native interval no-MB grouped-risk policy indices mismatch; policy={policy_name}")
            if not torch.allclose(got_interval_outputs, ref_outputs, atol=2e-5, rtol=2e-5):
                raise AssertionError(f"native interval no-MB grouped-risk policy outputs mismatch; policy={policy_name}")
            score_ref_outputs, score_ref_indices = joint_select_policy_grouped_flat_no_mb(
                score_ref,
                score_fused_k,
                score_fused_heads,
                0.25,
                policy_id,
            )
            got_score_interval_outputs, got_score_interval_indices = (
                joint_select_policy_from_grouped_scores_intervals_batched_no_mb(
                    score_fused_scores,
                    score_fused_vhat,
                    score_fused_residual,
                    score_fused_error,
                    score_fused_v_budgets,
                    0.25,
                    policy_id,
                )
            )
            if not torch.equal(got_score_interval_indices, score_ref_indices):
                raise AssertionError(
                    f"native score-interval no-MB grouped-risk policy indices mismatch; policy={policy_name}"
                )
            if not torch.allclose(got_score_interval_outputs, score_ref_outputs, atol=5e-5, rtol=5e-5):
                max_diff = float((got_score_interval_outputs - score_ref_outputs).abs().max().item())
                raise AssertionError(
                    "native score-interval no-MB grouped-risk policy outputs mismatch; "
                    f"policy={policy_name} max_diff={max_diff}"
                )
            got_score_prob_outputs, got_score_prob_indices = (
                joint_select_policy_from_grouped_scores_probs_intervals_batched_no_mb(
                    score_fused_scores,
                    score_fused_vhat,
                    score_fused_residual,
                    score_fused_error,
                    score_fused_v_budgets,
                    0.25,
                    policy_id,
                )
            )
            if not torch.equal(got_score_prob_indices, score_ref_indices):
                raise AssertionError(
                    f"native score-prob interval no-MB grouped-risk policy indices mismatch; policy={policy_name}"
                )
            if not torch.allclose(got_score_prob_outputs, score_ref_outputs, atol=5e-5, rtol=5e-5):
                max_diff = float((got_score_prob_outputs - score_ref_outputs).abs().max().item())
                raise AssertionError(
                    "native score-prob interval no-MB grouped-risk policy outputs mismatch; "
                    f"policy={policy_name} max_diff={max_diff}"
                )
            got_score_topk_outputs, got_score_topk_indices = (
                joint_select_policy_from_grouped_scores_topk_intervals_batched_no_mb(
                    score_fused_scores,
                    score_fused_vhat,
                    score_fused_residual,
                    score_fused_error,
                    score_fused_v_budgets,
                    int(score_fused_v_budgets.max().item()),
                    0.25,
                    policy_id,
                )
            )
            if not torch.equal(got_score_topk_indices, score_ref_indices):
                raise AssertionError(
                    f"native score-topk interval no-MB grouped-risk policy indices mismatch; policy={policy_name}"
                )
            if not torch.allclose(got_score_topk_outputs, score_ref_outputs, atol=5e-5, rtol=5e-5):
                max_diff = float((got_score_topk_outputs - score_ref_outputs).abs().max().item())
                raise AssertionError(
                    "native score-topk interval no-MB grouped-risk policy outputs mismatch; "
                    f"policy={policy_name} max_diff={max_diff}"
                )

    exact_scores = torch.randn((heads, query_context_len), device=device, dtype=torch.float32)
    indexed_tokens_score = torch.tensor([1, 2, 4, 7, 10, 13, 16, 19, 21], device=device, dtype=torch.long)
    pq_logits = torch.randn((heads, int(indexed_tokens_score.numel())), device=device, dtype=torch.float32)
    y_indexed = exact_scores.index_select(1, indexed_tokens_score)
    base_tokens_score = torch.tensor([0, 22], device=device, dtype=torch.long)
    ranked_prefix_score = torch.tensor(
        [
            [2, 5, 8, 11, 14, 17],
            [3, 6, 9, 12, 15, 18],
        ],
        device=device,
        dtype=torch.long,
    )
    k_take_counts = torch.tensor([0, 2, 5], device=device, dtype=torch.long)

    def _torch_mixed_score_grid(calibrate: bool) -> torch.Tensor:
        rows = []
        for take in k_take_counts.detach().cpu().tolist():
            add_t = ranked_prefix_score[:, : int(take)]
            base_rows_t = base_tokens_score.reshape(1, -1).expand(heads, -1)
            selected_t = torch.cat((base_rows_t, add_t), dim=1)
            score = exact_scores.clone()
            if calibrate:
                selected_mask = torch.zeros((heads, query_context_len), dtype=torch.bool, device=device)
                selected_mask.scatter_(1, selected_t, True)
                selected_index_mask = selected_mask.index_select(1, indexed_tokens_score)
                mask_f = selected_index_mask.to(dtype=torch.float32)
                counts_t = torch.sum(mask_f, dim=1)
                safe_counts_t = torch.clamp_min(counts_t, 1.0)
                x_mean_t = torch.sum(mask_f * pq_logits, dim=1) / safe_counts_t
                y_mean_t = torch.sum(mask_f * y_indexed, dim=1) / safe_counts_t
                x_centered_t = (pq_logits - x_mean_t.reshape(-1, 1)) * mask_f
                y_centered_t = (y_indexed - y_mean_t.reshape(-1, 1)) * mask_f
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
                calibrated = scale_t.reshape(-1, 1) * pq_logits + bias_t.reshape(-1, 1)
            else:
                calibrated = pq_logits
            score[:, indexed_tokens_score] = calibrated
            score.scatter_(1, selected_t, exact_scores.gather(1, selected_t))
            rows.append(score)
        return torch.stack(rows, dim=0)

    for calibrate in (False, True):
        got_score_grid = joint_mixed_score_grid(
            exact_scores,
            pq_logits,
            y_indexed,
            indexed_tokens_score,
            base_tokens_score,
            ranked_prefix_score,
            k_take_counts,
            calibrate,
        )
        ref_score_grid = _torch_mixed_score_grid(calibrate)
        if not torch.allclose(got_score_grid, ref_score_grid, atol=2e-5, rtol=2e-5):
            raise AssertionError(f"native joint mixed score grid mismatches Torch reference; calibrate={calibrate}")
        workspace_score = torch.empty_like(got_score_grid)
        workspace_mask = torch.empty(got_score_grid.shape, device=device, dtype=torch.uint8)
        workspace_fit_scale = torch.empty(
            (int(k_take_counts.numel()), int(exact_scores.shape[0])),
            device=device,
            dtype=torch.float32,
        )
        workspace_fit_bias = torch.empty_like(workspace_fit_scale)
        got_score_grid_workspace = joint_mixed_score_grid_workspace(
            workspace_score,
            workspace_mask,
            workspace_fit_scale,
            workspace_fit_bias,
            exact_scores,
            pq_logits,
            y_indexed,
            indexed_tokens_score,
            base_tokens_score,
            ranked_prefix_score,
            k_take_counts,
            calibrate,
        )
        if got_score_grid_workspace.data_ptr() != workspace_score.data_ptr():
            raise AssertionError("workspace score-grid helper did not return the caller-provided output tensor")
        if not torch.allclose(got_score_grid_workspace, ref_score_grid, atol=2e-5, rtol=2e-5):
            raise AssertionError(f"workspace joint mixed score grid mismatches Torch reference; calibrate={calibrate}")
        if not calibrate:
            scatter_score = torch.empty_like(got_score_grid)
            scatter_token_to_indexed = torch.empty((query_context_len,), device=device, dtype=torch.int32)
            got_scatter_score_grid = joint_mixed_score_grid_nocalib_scatter_workspace(
                scatter_score,
                scatter_token_to_indexed,
                exact_scores,
                pq_logits,
                indexed_tokens_score,
                base_tokens_score,
                ranked_prefix_score,
                k_take_counts,
                1.0,
            )
            if got_scatter_score_grid.data_ptr() != scatter_score.data_ptr():
                raise AssertionError("scatter score-grid helper did not return the caller-provided output tensor")
            if not torch.allclose(got_scatter_score_grid, ref_score_grid, atol=2e-5, rtol=2e-5):
                raise AssertionError("scatter no-calib joint mixed score grid mismatches Torch reference")
        pq_scale = 0.125
        got_score_grid_scaled = joint_mixed_score_grid_scaled(
            exact_scores,
            pq_logits,
            y_indexed,
            indexed_tokens_score,
            base_tokens_score,
            ranked_prefix_score,
            k_take_counts,
            calibrate,
            pq_scale,
        )
        pq_logits_saved = pq_logits
        pq_logits = pq_logits_saved * pq_scale
        ref_score_grid_scaled = _torch_mixed_score_grid(calibrate)
        pq_logits = pq_logits_saved
        if not torch.allclose(got_score_grid_scaled, ref_score_grid_scaled, atol=2e-5, rtol=2e-5):
            raise AssertionError(f"native scaled joint mixed score grid mismatches Torch reference; calibrate={calibrate}")
        if not calibrate:
            scatter_scaled = torch.empty_like(got_score_grid)
            scatter_scaled_token_to_indexed = torch.empty((query_context_len,), device=device, dtype=torch.int32)
            got_scatter_score_grid_scaled = joint_mixed_score_grid_nocalib_scatter_workspace(
                scatter_scaled,
                scatter_scaled_token_to_indexed,
                exact_scores,
                pq_logits,
                indexed_tokens_score,
                base_tokens_score,
                ranked_prefix_score,
                k_take_counts,
                pq_scale,
            )
            if not torch.allclose(got_scatter_score_grid_scaled, ref_score_grid_scaled, atol=2e-5, rtol=2e-5):
                raise AssertionError("scaled scatter no-calib joint mixed score grid mismatches Torch reference")
        got_score_grid_tokenfit = joint_mixed_score_grid_tokenfit_scaled(
            exact_scores,
            pq_logits,
            y_indexed,
            indexed_tokens_score,
            base_tokens_score,
            ranked_prefix_score,
            k_take_counts,
            calibrate,
            1.0,
        )
        if not torch.allclose(got_score_grid_tokenfit, ref_score_grid, atol=2e-5, rtol=2e-5):
            raise AssertionError(
                f"native tokenfit joint mixed score grid mismatches Torch reference; calibrate={calibrate}"
            )
        got_score_grid_tokenfit_scaled = joint_mixed_score_grid_tokenfit_scaled(
            exact_scores,
            pq_logits,
            y_indexed,
            indexed_tokens_score,
            base_tokens_score,
            ranked_prefix_score,
            k_take_counts,
            calibrate,
            pq_scale,
        )
        if not torch.allclose(got_score_grid_tokenfit_scaled, ref_score_grid_scaled, atol=2e-5, rtol=2e-5):
            raise AssertionError(
                f"native scaled tokenfit joint mixed score grid mismatches Torch reference; calibrate={calibrate}"
            )
        got_score_grid_rankpos = joint_mixed_score_grid_rankpos(
            exact_scores,
            pq_logits,
            y_indexed,
            indexed_tokens_score,
            base_tokens_score,
            ranked_prefix_score,
            k_take_counts,
            calibrate,
        )
        if not torch.allclose(got_score_grid_rankpos, ref_score_grid, atol=2e-5, rtol=2e-5):
            raise AssertionError(
                f"native rank-position joint mixed score grid mismatches Torch reference; calibrate={calibrate}"
            )
        values_for_mixed = torch.randn((query_context_len, dim), device=device, dtype=torch.float32)
        ref_probs, ref_base = joint_softmax_base_outputs(ref_score_grid.contiguous(), values_for_mixed)
        got_probs, got_base = joint_mixed_softmax_base_outputs(
            exact_scores,
            pq_logits,
            y_indexed,
            indexed_tokens_score,
            base_tokens_score,
            ranked_prefix_score,
            k_take_counts,
            values_for_mixed,
            calibrate,
        )
        if not torch.allclose(got_probs, ref_probs, atol=2e-5, rtol=2e-5):
            raise AssertionError(
                f"native fused mixed softmax probabilities mismatch score-grid reference; calibrate={calibrate}"
            )
        if not torch.allclose(got_base, ref_base, atol=2e-5, rtol=2e-5):
            raise AssertionError(
                f"native fused mixed softmax base output mismatch score-grid reference; calibrate={calibrate}"
            )
        got_probs_rankpos, got_base_rankpos = joint_mixed_softmax_base_outputs_rankpos(
            exact_scores,
            pq_logits,
            y_indexed,
            indexed_tokens_score,
            base_tokens_score,
            ranked_prefix_score,
            k_take_counts,
            values_for_mixed,
            calibrate,
        )
        if not torch.allclose(got_probs_rankpos, ref_probs, atol=2e-5, rtol=2e-5):
            raise AssertionError(
                f"native rank-position fused mixed softmax probabilities mismatch score-grid reference; calibrate={calibrate}"
            )
        if not torch.allclose(got_base_rankpos, ref_base, atol=2e-5, rtol=2e-5):
            raise AssertionError(
                f"native rank-position fused mixed softmax base output mismatch score-grid reference; calibrate={calibrate}"
            )
        got_probs_tokenfit, got_base_tokenfit = joint_mixed_softmax_base_outputs_tokenfit_scaled(
            exact_scores,
            pq_logits,
            y_indexed,
            indexed_tokens_score,
            base_tokens_score,
            ranked_prefix_score,
            k_take_counts,
            values_for_mixed,
            calibrate,
            1.0,
        )
        if not torch.allclose(got_probs_tokenfit, ref_probs, atol=2e-5, rtol=2e-5):
            raise AssertionError(
                f"native tokenfit fused mixed softmax probabilities mismatch score-grid reference; calibrate={calibrate}"
            )
        if not torch.allclose(got_base_tokenfit, ref_base, atol=2e-5, rtol=2e-5):
            raise AssertionError(
                f"native tokenfit fused mixed softmax base output mismatch score-grid reference; calibrate={calibrate}"
            )
        ref_probs_scaled, ref_base_scaled = joint_softmax_base_outputs(
            ref_score_grid_scaled.contiguous(),
            values_for_mixed,
        )
        got_probs_tokenfit_scaled, got_base_tokenfit_scaled = joint_mixed_softmax_base_outputs_tokenfit_scaled(
            exact_scores,
            pq_logits,
            y_indexed,
            indexed_tokens_score,
            base_tokens_score,
            ranked_prefix_score,
            k_take_counts,
            values_for_mixed,
            calibrate,
            pq_scale,
        )
        if not torch.allclose(got_probs_tokenfit_scaled, ref_probs_scaled, atol=2e-5, rtol=2e-5):
            raise AssertionError(
                f"native scaled tokenfit fused mixed softmax probabilities mismatch score-grid reference; calibrate={calibrate}"
            )
        if not torch.allclose(got_base_tokenfit_scaled, ref_base_scaled, atol=2e-5, rtol=2e-5):
            raise AssertionError(
                f"native scaled tokenfit fused mixed softmax base output mismatch score-grid reference; calibrate={calibrate}"
            )
        mixed_policy_v_budgets = torch.tensor([0, 3, 8, 15], device=device, dtype=torch.long)
        mixed_policy_residual = torch.randn((query_context_len, dim), device=device, dtype=torch.float32)
        mixed_policy_error = torch.rand((query_context_len,), device=device, dtype=torch.float32)
        mixed_policy_grid = joint_vprefix_outputs_from_grouped_risk_batched(
            ref_base.reshape(1, int(k_take_counts.numel()), heads, dim).contiguous(),
            ref_probs.reshape(1, int(k_take_counts.numel()), heads, query_context_len).contiguous(),
            mixed_policy_residual.reshape(1, query_context_len, dim).contiguous(),
            mixed_policy_error.reshape(1, query_context_len).contiguous(),
            mixed_policy_v_budgets,
        )
        if not calibrate:
            mixed_policy_merge_grid = joint_vprefix_outputs_from_grouped_merge_risk_batched(
                ref_base.reshape(1, int(k_take_counts.numel()), heads, dim).contiguous(),
                ref_probs.reshape(1, int(k_take_counts.numel()), heads, query_context_len).contiguous(),
                mixed_policy_residual.reshape(1, query_context_len, dim).contiguous(),
                mixed_policy_error.reshape(1, query_context_len).contiguous(),
                exact_scores.reshape(1, heads, query_context_len).contiguous(),
                pq_logits.reshape(1, heads, int(pq_logits.shape[1])).contiguous(),
                indexed_tokens_score.reshape(1, int(indexed_tokens_score.numel())).contiguous(),
                base_tokens_score.reshape(1, int(base_tokens_score.numel())).contiguous(),
                ranked_prefix_score.reshape(1, heads, int(ranked_prefix_score.shape[1])).contiguous(),
                k_take_counts,
                mixed_policy_v_budgets,
                1.0,
            )
            if not torch.allclose(mixed_policy_merge_grid, mixed_policy_grid, atol=8e-5, rtol=8e-5):
                max_diff = float((mixed_policy_merge_grid - mixed_policy_grid).abs().max().item())
                raise AssertionError(f"grouped merge-risk V-prefix grid mismatch; max_diff={max_diff}")
        for policy_name, policy_id in policy_ids.items():
            if policy_name == "sensitivity_greedy":
                continue
            ref_policy_outputs, ref_policy_indices = joint_select_policy_grouped_flat_no_mb(
                mixed_policy_grid,
                int(k_take_counts.numel()),
                heads,
                0.25,
                policy_id,
            )
            got_policy_outputs, got_policy_indices = joint_mixed_select_policy_intervals_no_mb(
                exact_scores,
                pq_logits,
                y_indexed,
                indexed_tokens_score,
                base_tokens_score,
                ranked_prefix_score,
                k_take_counts,
                values_for_mixed,
                mixed_policy_residual,
                mixed_policy_error,
                mixed_policy_v_budgets,
                calibrate,
                1.0,
                0.25,
                policy_id,
            )
            if not torch.equal(got_policy_indices, ref_policy_indices[0]):
                raise AssertionError(
                    "native mixed-score fused interval policy indices mismatch; "
                    f"policy={policy_name} calibrate={calibrate}"
                )
            if not torch.allclose(got_policy_outputs, ref_policy_outputs[0], atol=8e-5, rtol=8e-5):
                max_diff = float((got_policy_outputs - ref_policy_outputs[0]).abs().max().item())
                raise AssertionError(
                    "native mixed-score fused interval policy outputs mismatch; "
                    f"policy={policy_name} calibrate={calibrate} max_diff={max_diff}"
                )
            if not calibrate:
                got_rankpos_outputs, got_rankpos_indices = (
                    joint_mixed_select_policy_intervals_rankpos_no_calib_no_mb(
                        exact_scores,
                        pq_logits,
                        indexed_tokens_score,
                        base_tokens_score,
                        ranked_prefix_score,
                        k_take_counts,
                        values_for_mixed,
                        mixed_policy_residual,
                        mixed_policy_error,
                        mixed_policy_v_budgets,
                        1.0,
                        0.25,
                        policy_id,
                    )
                )
                if not torch.equal(got_rankpos_indices, ref_policy_indices[0]):
                    raise AssertionError(
                        "native no-calib rankpos mixed-score fused interval policy indices mismatch; "
                        f"policy={policy_name}"
                    )
                if not torch.allclose(got_rankpos_outputs, ref_policy_outputs[0], atol=8e-5, rtol=8e-5):
                    max_diff = float((got_rankpos_outputs - ref_policy_outputs[0]).abs().max().item())
                    raise AssertionError(
                        "native no-calib rankpos mixed-score fused interval policy outputs mismatch; "
                        f"policy={policy_name} max_diff={max_diff}"
                    )
                got_merge_outputs, got_merge_indices = (
                    joint_mixed_select_policy_merge_rankpos_no_calib_no_mb(
                        exact_scores,
                        pq_logits,
                        indexed_tokens_score,
                        base_tokens_score,
                        ranked_prefix_score,
                        k_take_counts,
                        values_for_mixed,
                        mixed_policy_residual,
                        mixed_policy_error,
                        mixed_policy_v_budgets,
                        1.0,
                        0.25,
                        policy_id,
                    )
                )
                if not torch.equal(got_merge_indices, ref_policy_indices[0]):
                    raise AssertionError(
                        "native merge-risk no-calib rankpos policy indices mismatch; "
                        f"policy={policy_name}"
                    )
                if not torch.allclose(got_merge_outputs, ref_policy_outputs[0], atol=8e-5, rtol=8e-5):
                    max_diff = float((got_merge_outputs - ref_policy_outputs[0]).abs().max().item())
                    raise AssertionError(
                        "native merge-risk no-calib rankpos policy outputs mismatch; "
                        f"policy={policy_name} max_diff={max_diff}"
                    )

    indexed_tokens_covered = torch.arange(1, query_context_len - 1, device=device, dtype=torch.long)
    pq_logits_covered = torch.randn(
        (heads, int(indexed_tokens_covered.numel())),
        device=device,
        dtype=torch.float32,
    )
    y_indexed_covered = exact_scores.index_select(1, indexed_tokens_covered)

    def _torch_mixed_score_grid_covered(calibrate: bool) -> torch.Tensor:
        rows = []
        for take in k_take_counts.detach().cpu().tolist():
            add_t = ranked_prefix_score[:, : int(take)]
            base_rows_t = base_tokens_score.reshape(1, -1).expand(heads, -1)
            selected_t = torch.cat((base_rows_t, add_t), dim=1)
            score = exact_scores.clone()
            if calibrate:
                selected_mask = torch.zeros((heads, query_context_len), dtype=torch.bool, device=device)
                selected_mask.scatter_(1, selected_t, True)
                selected_index_mask = selected_mask.index_select(1, indexed_tokens_covered)
                mask_f = selected_index_mask.to(dtype=torch.float32)
                counts_t = torch.sum(mask_f, dim=1)
                safe_counts_t = torch.clamp_min(counts_t, 1.0)
                x_mean_t = torch.sum(mask_f * pq_logits_covered, dim=1) / safe_counts_t
                y_mean_t = torch.sum(mask_f * y_indexed_covered, dim=1) / safe_counts_t
                x_centered_t = (pq_logits_covered - x_mean_t.reshape(-1, 1)) * mask_f
                y_centered_t = (y_indexed_covered - y_mean_t.reshape(-1, 1)) * mask_f
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
                calibrated = scale_t.reshape(-1, 1) * pq_logits_covered + bias_t.reshape(-1, 1)
            else:
                calibrated = pq_logits_covered
            score[:, indexed_tokens_covered] = calibrated
            score.scatter_(1, selected_t, exact_scores.gather(1, selected_t))
            rows.append(score)
        return torch.stack(rows, dim=0)

    for calibrate in (False, True):
        got_score_grid_no_fill = joint_mixed_score_grid_no_exact_fill(
            exact_scores,
            pq_logits_covered,
            y_indexed_covered,
            indexed_tokens_covered,
            base_tokens_score,
            ranked_prefix_score,
            k_take_counts,
            calibrate,
        )
        ref_score_grid_no_fill = _torch_mixed_score_grid_covered(calibrate)
        if not torch.allclose(got_score_grid_no_fill, ref_score_grid_no_fill, atol=2e-5, rtol=2e-5):
            raise AssertionError(
                f"native no-fill joint mixed score grid mismatches covered Torch reference; calibrate={calibrate}"
            )
        pq_scale = 0.125
        got_score_grid_no_fill_scaled = joint_mixed_score_grid_no_exact_fill_scaled(
            exact_scores,
            pq_logits_covered,
            y_indexed_covered,
            indexed_tokens_covered,
            base_tokens_score,
            ranked_prefix_score,
            k_take_counts,
            calibrate,
            pq_scale,
        )
        pq_logits_covered_saved = pq_logits_covered
        pq_logits_covered = pq_logits_covered_saved * pq_scale
        ref_score_grid_no_fill_scaled = _torch_mixed_score_grid_covered(calibrate)
        pq_logits_covered = pq_logits_covered_saved
        if not torch.allclose(got_score_grid_no_fill_scaled, ref_score_grid_no_fill_scaled, atol=2e-5, rtol=2e-5):
            raise AssertionError(
                f"native scaled no-fill joint mixed score grid mismatches covered Torch reference; calibrate={calibrate}"
            )
        got_score_grid_no_fill_tokenfit = joint_mixed_score_grid_no_exact_fill_tokenfit_scaled(
            exact_scores,
            pq_logits_covered,
            y_indexed_covered,
            indexed_tokens_covered,
            base_tokens_score,
            ranked_prefix_score,
            k_take_counts,
            calibrate,
            1.0,
        )
        if not torch.allclose(got_score_grid_no_fill_tokenfit, ref_score_grid_no_fill, atol=2e-5, rtol=2e-5):
            raise AssertionError(
                f"native no-fill tokenfit joint mixed score grid mismatches covered Torch reference; calibrate={calibrate}"
            )
        got_score_grid_no_fill_tokenfit_scaled = joint_mixed_score_grid_no_exact_fill_tokenfit_scaled(
            exact_scores,
            pq_logits_covered,
            y_indexed_covered,
            indexed_tokens_covered,
            base_tokens_score,
            ranked_prefix_score,
            k_take_counts,
            calibrate,
            pq_scale,
        )
        if not torch.allclose(
            got_score_grid_no_fill_tokenfit_scaled,
            ref_score_grid_no_fill_scaled,
            atol=2e-5,
            rtol=2e-5,
        ):
            raise AssertionError(
                f"native scaled no-fill tokenfit joint mixed score grid mismatches covered Torch reference; calibrate={calibrate}"
            )
        sparse_base_logits = exact_scores.gather(
            1,
            base_tokens_score.reshape(1, -1).expand(heads, -1),
        ).contiguous()
        sparse_ranked_logits = exact_scores.gather(1, ranked_prefix_score).contiguous()
        got_score_grid_sparse_direct = joint_mixed_score_grid_sparse_exact_tokenfit_scaled(
            pq_logits_covered,
            indexed_tokens_covered,
            base_tokens_score,
            sparse_base_logits,
            ranked_prefix_score,
            sparse_ranked_logits,
            k_take_counts,
            query_context_len,
            calibrate,
            pq_scale,
        )
        if not torch.allclose(
            got_score_grid_sparse_direct,
            ref_score_grid_no_fill_scaled,
            atol=2e-5,
            rtol=2e-5,
        ):
            max_diff = float((got_score_grid_sparse_direct - ref_score_grid_no_fill_scaled).abs().max().item())
            raise AssertionError(
                "native sparse-exact direct tokenfit score grid mismatches covered Torch reference; "
                f"calibrate={calibrate} max_diff={max_diff}"
            )
        ref_sparse_probs, ref_sparse_base = joint_softmax_base_outputs(
            ref_score_grid_no_fill_scaled.contiguous(),
            values_for_mixed,
        )
        got_sparse_probs, got_sparse_base = joint_mixed_softmax_base_outputs_sparse_exact_tokenfit_scaled(
            pq_logits_covered,
            indexed_tokens_covered,
            base_tokens_score,
            sparse_base_logits,
            ranked_prefix_score,
            sparse_ranked_logits,
            k_take_counts,
            values_for_mixed,
            query_context_len,
            calibrate,
            pq_scale,
        )
        if not torch.allclose(got_sparse_probs, ref_sparse_probs, atol=2e-5, rtol=2e-5):
            max_diff = float((got_sparse_probs - ref_sparse_probs).abs().max().item())
            raise AssertionError(
                "native sparse-exact direct tokenfit fused softmax probs mismatch; "
                f"calibrate={calibrate} max_diff={max_diff}"
            )
        if not torch.allclose(got_sparse_base, ref_sparse_base, atol=2e-5, rtol=2e-5):
            max_diff = float((got_sparse_base - ref_sparse_base).abs().max().item())
            raise AssertionError(
                "native sparse-exact direct tokenfit fused base output mismatch; "
                f"calibrate={calibrate} max_diff={max_diff}"
            )

    k_take_counts_with_full_row = torch.tensor([2, 99], device=device, dtype=torch.long)
    for calibrate in (False, True):
        got_full_row_grid = joint_mixed_score_grid(
            exact_scores,
            pq_logits,
            y_indexed,
            indexed_tokens_score,
            base_tokens_score,
            ranked_prefix_score[:, :2],
            k_take_counts_with_full_row,
            calibrate,
        )
        ref_first_row = _torch_mixed_score_grid(calibrate)[1]
        if not torch.allclose(got_full_row_grid[0], ref_first_row, atol=2e-5, rtol=2e-5):
            raise AssertionError(
                f"native joint mixed score grid truncated-prefix row mismatches Torch reference; calibrate={calibrate}"
            )
        if not torch.allclose(got_full_row_grid[1], exact_scores, atol=2e-5, rtol=2e-5):
            raise AssertionError(
                f"native joint mixed score grid full row is not exact; calibrate={calibrate}"
            )
        got_full_row_grid_rankpos = joint_mixed_score_grid_rankpos(
            exact_scores,
            pq_logits,
            y_indexed,
            indexed_tokens_score,
            base_tokens_score,
            ranked_prefix_score[:, :2],
            k_take_counts_with_full_row,
            calibrate,
        )
        if not torch.allclose(got_full_row_grid_rankpos[0], ref_first_row, atol=2e-5, rtol=2e-5):
            raise AssertionError(
                f"native rank-position joint mixed score grid truncated-prefix row mismatches Torch reference; calibrate={calibrate}"
            )
        if not torch.allclose(got_full_row_grid_rankpos[1], exact_scores, atol=2e-5, rtol=2e-5):
            raise AssertionError(
                f"native rank-position joint mixed score grid full row is not exact; calibrate={calibrate}"
            )

        got_full_row_grid_no_fill = joint_mixed_score_grid_no_exact_fill(
            exact_scores,
            pq_logits_covered,
            y_indexed_covered,
            indexed_tokens_covered,
            base_tokens_score,
            ranked_prefix_score[:, :2],
            k_take_counts_with_full_row,
            calibrate,
        )
        if not torch.allclose(got_full_row_grid_no_fill[0], _torch_mixed_score_grid_covered(calibrate)[1], atol=2e-5, rtol=2e-5):
            raise AssertionError(
                f"native no-fill joint mixed score grid truncated-prefix row mismatches Torch reference; calibrate={calibrate}"
            )
        if not torch.allclose(got_full_row_grid_no_fill[1], exact_scores, atol=2e-5, rtol=2e-5):
            raise AssertionError(
                f"native no-fill joint mixed score grid full row is not exact; calibrate={calibrate}"
            )

    policy_output_grid = torch.randn((4, 5, heads, dim), device=device, dtype=torch.float32)
    k_mb = torch.tensor([1.0, 1.5, 2.2, 3.0], device=device, dtype=torch.float32)
    v_mb = torch.tensor([0.5, 0.7, 1.1, 1.8, 2.6], device=device, dtype=torch.float32)

    def _policy_ref(policy: str, head_i: int, threshold: float) -> tuple[int, int]:
        ki = 0
        vi = 0
        steps = 0
        while steps < (policy_output_grid.shape[0] + policy_output_grid.shape[1] + 4):
            cur = policy_output_grid[ki, vi, head_i].double()
            k_can = ki + 1 < policy_output_grid.shape[0]
            v_can = vi + 1 < policy_output_grid.shape[1]
            k_delta = (
                float(torch.linalg.vector_norm(cur - policy_output_grid[ki + 1, vi, head_i].double()))
                / max(float(torch.linalg.vector_norm(policy_output_grid[ki + 1, vi, head_i].double())), 1e-20)
                if k_can
                else 0.0
            )
            v_delta = (
                float(torch.linalg.vector_norm(cur - policy_output_grid[ki, vi + 1, head_i].double()))
                / max(float(torch.linalg.vector_norm(policy_output_grid[ki, vi + 1, head_i].double())), 1e-20)
                if v_can
                else 0.0
            )
            extra_k = float(k_mb[ki + 1] - k_mb[ki]) if k_can else float("inf")
            extra_v = float(v_mb[vi + 1] - v_mb[vi]) if v_can else float("inf")
            action = _choose_joint_kv_action(
                policy=policy,
                k_delta=k_delta,
                v_delta=v_delta,
                k_can=k_can,
                v_can=v_can,
                threshold=threshold,
                turn=steps,
                extra_k_mb=extra_k,
                extra_v_mb=extra_v,
            )
            if action == "stop":
                break
            if action == "k":
                ki += 1
            elif action == "v":
                vi += 1
            else:
                raise AssertionError(action)
            steps += 1
        return int(ki), int(vi)

    for policy_name, policy_id in policy_ids.items():
        got_policy = joint_select_policy(policy_output_grid, k_mb, v_mb, 0.75, policy_id)
        ref_policy = torch.tensor(
            [_policy_ref(policy_name, head_i, 0.75) for head_i in range(heads)],
            dtype=torch.long,
            device=device,
        )
        if not torch.equal(got_policy, ref_policy):
            raise AssertionError(f"native joint policy selector mismatches Torch reference; policy={policy_name}")

    grouped_policy_grid = torch.stack(
        [
            policy_output_grid,
            policy_output_grid * 0.75 + 0.1,
            policy_output_grid * 1.25 - 0.2,
        ],
        dim=0,
    )
    grouped_policy_flat = grouped_policy_grid.permute(0, 1, 3, 2, 4).reshape(
        int(grouped_policy_grid.shape[0]) * int(grouped_policy_grid.shape[1]) * int(grouped_policy_grid.shape[3]),
        int(grouped_policy_grid.shape[2]),
        int(grouped_policy_grid.shape[4]),
    )
    grouped_k_mb = torch.stack([k_mb, k_mb + 0.05, k_mb + 0.10], dim=0).contiguous()
    grouped_v_mb = torch.stack([v_mb, v_mb + 0.03, v_mb + 0.06], dim=0).contiguous()
    for policy_name, policy_id in policy_ids.items():
        got_outputs, got_indices = joint_select_policy_grouped_flat(
            grouped_policy_flat,
            grouped_k_mb,
            grouped_v_mb,
            int(policy_output_grid.shape[0]),
            int(heads),
            0.75,
            policy_id,
        )
        ref_indices = []
        ref_outputs = []
        for group_i in range(int(grouped_policy_grid.shape[0])):
            group_indices = []
            group_outputs = []
            for head_i in range(heads):
                ki = 0
                vi = 0
                steps = 0
                while steps < (policy_output_grid.shape[0] + policy_output_grid.shape[1] + 4):
                    cur = grouped_policy_grid[group_i, ki, vi, head_i].double()
                    k_can = ki + 1 < policy_output_grid.shape[0]
                    v_can = vi + 1 < policy_output_grid.shape[1]
                    k_delta = (
                        float(torch.linalg.vector_norm(cur - grouped_policy_grid[group_i, ki + 1, vi, head_i].double()))
                        / max(float(torch.linalg.vector_norm(grouped_policy_grid[group_i, ki + 1, vi, head_i].double())), 1e-20)
                        if k_can
                        else 0.0
                    )
                    v_delta = (
                        float(torch.linalg.vector_norm(cur - grouped_policy_grid[group_i, ki, vi + 1, head_i].double()))
                        / max(float(torch.linalg.vector_norm(grouped_policy_grid[group_i, ki, vi + 1, head_i].double())), 1e-20)
                        if v_can
                        else 0.0
                    )
                    extra_k = float(grouped_k_mb[group_i, ki + 1] - grouped_k_mb[group_i, ki]) if k_can else float("inf")
                    extra_v = float(grouped_v_mb[group_i, vi + 1] - grouped_v_mb[group_i, vi]) if v_can else float("inf")
                    action = _choose_joint_kv_action(
                        policy=policy_name,
                        k_delta=k_delta,
                        v_delta=v_delta,
                        k_can=k_can,
                        v_can=v_can,
                        threshold=0.75,
                        turn=steps,
                        extra_k_mb=extra_k,
                        extra_v_mb=extra_v,
                    )
                    if action == "stop":
                        break
                    if action == "k":
                        ki += 1
                    elif action == "v":
                        vi += 1
                    else:
                        raise AssertionError(action)
                    steps += 1
                group_indices.append((ki, vi))
                group_outputs.append(grouped_policy_grid[group_i, ki, vi, head_i])
            ref_indices.append(group_indices)
            ref_outputs.append(torch.stack(group_outputs, dim=0))
        ref_indices_t = torch.tensor(ref_indices, dtype=torch.long, device=device)
        ref_outputs_t = torch.stack(ref_outputs, dim=0)
        if not torch.equal(got_indices, ref_indices_t):
            raise AssertionError(f"native grouped flat joint policy indices mismatch; policy={policy_name}")
        if not torch.allclose(got_outputs, ref_outputs_t, atol=2e-5, rtol=2e-5):
            raise AssertionError(f"native grouped flat joint policy outputs mismatch; policy={policy_name}")
        if policy_name != "sensitivity_greedy":
            got_outputs_no_mb, got_indices_no_mb = joint_select_policy_grouped_flat_no_mb(
                grouped_policy_flat,
                int(policy_output_grid.shape[0]),
                int(heads),
                0.75,
                policy_id,
            )
            if not torch.equal(got_indices_no_mb, ref_indices_t):
                raise AssertionError(f"native grouped flat no-MB policy indices mismatch; policy={policy_name}")
            if not torch.allclose(got_outputs_no_mb, ref_outputs_t, atol=2e-5, rtol=2e-5):
                raise AssertionError(f"native grouped flat no-MB policy outputs mismatch; policy={policy_name}")
            accounting_counts = torch.tensor([64, 128, 256, 384], dtype=torch.long, device=device)
            accounting_v_budgets_for_grid = torch.tensor([32, 96, 160, 192, 224], dtype=torch.long, device=device)
            (
                got_outputs_accounting,
                got_indices_accounting,
                got_sums_accounting,
            ) = joint_select_policy_grouped_flat_no_mb_accounting(
                grouped_policy_flat,
                accounting_counts,
                accounting_v_budgets_for_grid,
                int(policy_output_grid.shape[0]),
                int(heads),
                200,
                int(dim),
                2,
                2,
                0.25,
                0.03125,
                0.001,
                4,
                1,
                0.75,
                policy_id,
            )
            if not torch.equal(got_indices_accounting, ref_indices_t):
                raise AssertionError(f"native grouped flat fused-accounting indices mismatch; policy={policy_name}")
            if not torch.allclose(got_outputs_accounting, ref_outputs_t, atol=2e-5, rtol=2e-5):
                raise AssertionError(f"native grouped flat fused-accounting outputs mismatch; policy={policy_name}")
            ref_sums_accounting = np.zeros((11,), dtype=np.float64)
            for group_rows in ref_indices_t.detach().cpu().tolist():
                for ki, vi in group_rows:
                    selected_count = int(accounting_counts[int(ki)].item())
                    exact_v = max(0, min(int(accounting_v_budgets_for_grid[int(vi)].item()), 200))
                    tail_count = max(0, 200 - exact_v)
                    exact_key_mb = float(selected_count * dim * 2) / (1024.0 * 1024.0)
                    exact_v_mb = float(exact_v * dim * 2) / (1024.0 * 1024.0)
                    tail_mb = 0.03125 + float(tail_count * 4) / (1024.0 * 1024.0) + 0.001
                    dense_physical_key_mb = float(200 * dim * 2) / (1024.0 * 1024.0)
                    ref_sums_accounting[0] += selected_count
                    ref_sums_accounting[1] += tail_count
                    ref_sums_accounting[2] += 0.25
                    ref_sums_accounting[3] += exact_key_mb + exact_v_mb
                    ref_sums_accounting[4] += tail_mb
                    ref_sums_accounting[6] += dense_physical_key_mb + exact_v_mb
                    ref_sums_accounting[8] += 1
                    ref_sums_accounting[9] += 1
            got_sums_accounting_np = np.asarray(got_sums_accounting.detach().cpu().numpy(), dtype=np.float64)
            fused_accounting_diff = float(np.max(np.abs(got_sums_accounting_np - ref_sums_accounting)))
            if fused_accounting_diff > 1e-10:
                raise AssertionError(
                    "native grouped flat fused-accounting sums mismatch: "
                    f"max_diff={fused_accounting_diff} got={got_sums_accounting_np.tolist()} "
                    f"ref={ref_sums_accounting.tolist()}"
                )
            got_outputs_staged, got_indices_staged, got_boundary_staged = (
                joint_select_policy_grouped_flat_staged_no_mb(
                    grouped_policy_flat,
                    int(policy_output_grid.shape[0]),
                    int(heads),
                    0.75,
                    policy_id,
                )
            )
            ref_boundary = torch.any(
                (ref_indices_t[:, :, 0] >= int(policy_output_grid.shape[0] - 1))
                | (ref_indices_t[:, :, 1] >= int(policy_output_grid.shape[1] - 1)),
                dim=1,
            )
            if not torch.equal(got_indices_staged, ref_indices_t):
                raise AssertionError(f"native grouped staged no-MB policy indices mismatch; policy={policy_name}")
            if not torch.allclose(got_outputs_staged, ref_outputs_t, atol=2e-5, rtol=2e-5):
                raise AssertionError(f"native grouped staged no-MB policy outputs mismatch; policy={policy_name}")
            if not torch.equal(got_boundary_staged, ref_boundary):
                raise AssertionError(f"native grouped staged no-MB boundary mask mismatch; policy={policy_name}")

    final_indices = torch.tensor(
        [[0, 0], [1, 2], [3, 1], [2, 3]],
        dtype=torch.long,
        device=device,
    )
    selected_counts = torch.tensor([128, 256, 512, 1024], dtype=torch.long, device=device)
    accounting_v_budgets = torch.tensor([0, 64, 256, 2048], dtype=torch.long, device=device)
    context_len = 1500
    selector_mb = 0.125
    codebook_mb = 0.03125
    metadata_mb = 0.001
    value_subvecs = 4
    code_bytes = 1
    got_accounting = joint_grouped_accounting_sums(
        final_indices,
        selected_counts,
        accounting_v_budgets,
        context_len,
        dim,
        2,
        2,
        selector_mb,
        codebook_mb,
        metadata_mb,
        value_subvecs,
        code_bytes,
    ).detach().cpu().numpy()
    got_accounting = np.asarray(got_accounting, dtype=np.float64)
    ref_accounting = np.zeros((11,), dtype=np.float64)
    for ki, vi in final_indices.detach().cpu().tolist():
        selected_count = int(selected_counts[int(ki)].item())
        exact_v = max(0, min(int(accounting_v_budgets[int(vi)].item()), context_len))
        tail_count = max(0, context_len - exact_v)
        exact_key_mb = float(selected_count * dim * 2) / (1024.0 * 1024.0)
        exact_v_mb = float(exact_v * dim * 2) / (1024.0 * 1024.0)
        tail_mb = codebook_mb + float(tail_count * value_subvecs * code_bytes) / (1024.0 * 1024.0) + metadata_mb
        dense_physical_key_mb = float(context_len * dim * 2) / (1024.0 * 1024.0)
        ref_accounting[0] += selected_count
        ref_accounting[1] += tail_count
        ref_accounting[2] += selector_mb
        ref_accounting[3] += exact_key_mb + exact_v_mb
        ref_accounting[4] += tail_mb
        ref_accounting[6] += dense_physical_key_mb + exact_v_mb
        ref_accounting[8] += 1 if selector_mb > 0.0 else 0
        ref_accounting[9] += 1 if tail_mb > 0.0 else 0
    accounting_max_diff = float(np.max(np.abs(got_accounting - ref_accounting)))
    if accounting_max_diff > 1e-10:
        raise AssertionError(
            "native grouped accounting sums mismatch: "
            f"max_diff={accounting_max_diff} got={got_accounting.tolist()} ref={ref_accounting.tolist()}"
        )
    accounting_accum = torch.zeros((11,), dtype=torch.float64, device=device)
    joint_grouped_accounting_accumulate(
        accounting_accum,
        final_indices[:2],
        selected_counts,
        accounting_v_budgets,
        context_len,
        dim,
        2,
        2,
        selector_mb,
        codebook_mb,
        metadata_mb,
        value_subvecs,
        code_bytes,
    )
    joint_grouped_accounting_accumulate(
        accounting_accum,
        final_indices[2:],
        selected_counts,
        accounting_v_budgets,
        context_len,
        dim,
        2,
        2,
        selector_mb,
        codebook_mb,
        metadata_mb,
        value_subvecs,
        code_bytes,
    )
    got_accum = np.asarray(accounting_accum.detach().cpu().numpy(), dtype=np.float64)
    accum_max_diff = float(np.max(np.abs(got_accum - ref_accounting)))
    if accum_max_diff > 1e-10:
        raise AssertionError(
            "native grouped accounting accumulator mismatch: "
            f"max_diff={accum_max_diff} got={got_accum.tolist()} ref={ref_accounting.tolist()}"
        )

    ref = gqa_causal_vpq_selected_tail_from_scores(
        queries,
        keys,
        values,
        dense_pq_scores,
        value_codebooks,
        value_codes,
        page_starts,
        ranked_tokens,
        ranked_scores,
        2,
        query_start,
        static_prefix,
        static_suffix,
        page_size,
        1,
        scale,
        1.0,
    )
    got = gqa_causal_vpq_selected_tail_from_scores_counts(
        queries,
        keys,
        values,
        dense_pq_scores,
        value_codebooks,
        value_codes,
        page_starts,
        ranked_tokens,
        ranked_scores,
        counts,
        2,
        query_start,
        static_prefix,
        static_suffix,
        page_size,
        0,
        scale,
        1.0,
    )
    if not torch.allclose(ref, got, atol=2e-3, rtol=2e-3):
        raise AssertionError("prefill per-row exact count path mismatches fixed exact top")
    ref_all = gqa_causal_vpq_selected_tail_from_scores(
        queries,
        keys,
        values,
        dense_pq_scores,
        value_codebooks,
        value_codes,
        page_starts,
        ranked_tokens,
        ranked_scores,
        2,
        query_start,
        static_prefix,
        static_suffix,
        page_size,
        ranked,
        scale,
        1.0,
    )
    got_mass_all = gqa_causal_vpq_selected_tail_from_scores_mass(
        queries,
        keys,
        values,
        dense_pq_scores,
        value_codebooks,
        value_codes,
        page_starts,
        ranked_tokens,
        ranked_scores,
        1.0,
        2,
        query_start,
        static_prefix,
        static_suffix,
        page_size,
        scale,
        1.0,
    )
    if not torch.allclose(ref_all, got_mass_all, atol=2e-3, rtol=2e-3):
        raise AssertionError("prefill in-kernel mass exactness mismatches exact-all selected V")
    got_mass_min = gqa_causal_vpq_selected_tail_from_scores_mass_min(
        queries,
        keys,
        values,
        dense_pq_scores,
        value_codebooks,
        value_codes,
        page_starts,
        ranked_tokens,
        ranked_scores,
        0.0,
        1,
        2,
        query_start,
        static_prefix,
        static_suffix,
        page_size,
        scale,
        1.0,
    )
    if not torch.allclose(ref, got_mass_min, atol=2e-3, rtol=2e-3):
        raise AssertionError("prefill in-kernel mass+min exactness mismatches fixed exact top")
    ref_top2 = gqa_causal_vpq_selected_tail_from_scores(
        queries,
        keys,
        values,
        dense_pq_scores,
        value_codebooks,
        value_codes,
        page_starts,
        ranked_tokens,
        ranked_scores,
        2,
        query_start,
        static_prefix,
        static_suffix,
        page_size,
        2,
        scale,
        1.0,
    )
    got_mass_min2 = gqa_causal_vpq_selected_tail_from_scores_mass_min(
        queries,
        keys,
        values,
        dense_pq_scores,
        value_codebooks,
        value_codes,
        page_starts,
        ranked_tokens,
        ranked_scores,
        0.0,
        2,
        2,
        query_start,
        static_prefix,
        static_suffix,
        page_size,
        scale,
        1.0,
    )
    if not torch.allclose(ref_top2, got_mass_min2, atol=2e-3, rtol=2e-3):
        raise AssertionError("prefill in-kernel mass+min exactness mismatches fixed exact top=2")

    q_decode = queries[-1].contiguous()
    dense_decode = dense_pq_scores[-1].contiguous()
    ranked_decode = ranked_tokens[-1].contiguous()
    ranked_scores_decode = ranked_scores[-1].contiguous()
    counts_decode = torch.ones((heads,), device=device, dtype=torch.long)
    ref_decode = gqa_decode_vpq_selected_tail_agg_from_scores(
        q_decode,
        keys,
        values,
        dense_decode,
        value_codebooks,
        value_codes,
        page_starts,
        ranked_decode,
        ranked_scores_decode,
        2,
        query_context_len,
        static_prefix,
        static_suffix,
        page_size,
        1,
        scale,
        1.0,
    )
    got_decode = gqa_decode_vpq_selected_tail_agg_from_scores_counts(
        q_decode,
        keys,
        values,
        dense_decode,
        value_codebooks,
        value_codes,
        page_starts,
        ranked_decode,
        ranked_scores_decode,
        counts_decode,
        2,
        query_context_len,
        static_prefix,
        static_suffix,
        page_size,
        0,
        scale,
        1.0,
    )
    if not torch.allclose(ref_decode, got_decode, atol=2e-3, rtol=2e-3):
        raise AssertionError("decode per-head exact count path mismatches fixed exact top")
    ref_decode_all = gqa_decode_vpq_selected_tail_agg_from_scores(
        q_decode,
        keys,
        values,
        dense_decode,
        value_codebooks,
        value_codes,
        page_starts,
        ranked_decode,
        ranked_scores_decode,
        2,
        query_context_len,
        static_prefix,
        static_suffix,
        page_size,
        ranked,
        scale,
        1.0,
    )
    got_decode_mass_all = gqa_decode_vpq_selected_tail_agg_from_scores_mass(
        q_decode,
        keys,
        values,
        dense_decode,
        value_codebooks,
        value_codes,
        page_starts,
        ranked_decode,
        ranked_scores_decode,
        1.0,
        2,
        query_context_len,
        static_prefix,
        static_suffix,
        page_size,
        scale,
        1.0,
    )
    if not torch.allclose(ref_decode_all, got_decode_mass_all, atol=2e-3, rtol=2e-3):
        raise AssertionError("decode in-kernel mass exactness mismatches exact-all selected V")
    got_decode_mass_min = gqa_decode_vpq_selected_tail_agg_from_scores_mass_min(
        q_decode,
        keys,
        values,
        dense_decode,
        value_codebooks,
        value_codes,
        page_starts,
        ranked_decode,
        ranked_scores_decode,
        0.0,
        1,
        2,
        query_context_len,
        static_prefix,
        static_suffix,
        page_size,
        scale,
        1.0,
    )
    if not torch.allclose(ref_decode, got_decode_mass_min, atol=2e-3, rtol=2e-3):
        raise AssertionError("decode in-kernel mass+min exactness mismatches fixed exact top")
    key_subvecs = 2
    key_centroids = 4
    key_subdim = dim // key_subvecs
    key_codebooks = torch.randn(
        (kv_heads, pages, key_subvecs, key_centroids, key_subdim),
        device=device,
        dtype=torch.float32,
    )
    key_codes = torch.randint(
        0,
        key_centroids,
        (kv_heads, pages, page_size, key_subvecs),
        device=device,
        dtype=torch.uint8,
    )
    fused_ranked_tokens, fused_ranked_scores, fused_dense_scores = gqa_fullscan_pq_topk_scores(
        q_decode,
        key_codebooks,
        key_codes,
        page_starts,
        2,
        ranked,
    )
    ref_fused = gqa_decode_vpq_selected_tail_agg_from_scores(
        q_decode,
        keys,
        values,
        fused_dense_scores,
        value_codebooks,
        value_codes,
        page_starts,
        fused_ranked_tokens,
        fused_ranked_scores,
        2,
        query_context_len,
        static_prefix,
        static_suffix,
        page_size,
        1,
        scale,
        1.0,
    )
    got_fused = gqa_decode_fullscan_vpq_selected_tail_agg(
        q_decode,
        key_codebooks,
        key_codes,
        keys,
        values,
        value_codebooks,
        value_codes,
        page_starts,
        2,
        ranked,
        query_context_len,
        static_prefix,
        static_suffix,
        page_size,
        1,
        scale,
        1.0,
    )
    if not torch.allclose(ref_fused, got_fused, atol=2e-3, rtol=2e-3):
        raise AssertionError("decode fullscan fused fixed-exact path mismatches separate selector+attention")
    got_fused_mass_min = gqa_decode_fullscan_vpq_selected_tail_agg_mass_min(
        q_decode,
        key_codebooks,
        key_codes,
        keys,
        values,
        value_codebooks,
        value_codes,
        page_starts,
        0.0,
        1,
        2,
        ranked,
        query_context_len,
        static_prefix,
        static_suffix,
        page_size,
        scale,
        1.0,
    )
    if not torch.allclose(ref_fused, got_fused_mass_min, atol=2e-3, rtol=2e-3):
        raise AssertionError("decode fullscan fused mass+min path mismatches separate selector+attention")
    scoreless_ranked_tokens, scoreless_ranked_scores = gqa_causal_fullscan_pq_topk_fused_force(
        q_decode.reshape(1, heads, dim).contiguous(),
        key_codebooks,
        key_codes,
        page_starts,
        2,
        ranked,
        query_context_len - 1,
        static_prefix,
        static_suffix,
        2,
    )
    ref_scoreless = gqa_causal_vpq_selected_tail_attention(
        q_decode.reshape(1, heads, dim).contiguous(),
        keys.contiguous(),
        values.contiguous(),
        key_codebooks,
        key_codes,
        value_codebooks,
        value_codes,
        page_starts,
        scoreless_ranked_tokens,
        scoreless_ranked_scores,
        2,
        query_context_len - 1,
        static_prefix,
        static_suffix,
        page_size,
        1,
        scale,
        1.0,
    ).reshape(heads, dim)
    got_scoreless = gqa_decode_scoreless_fullscan_vpq_tail(
        q_decode,
        key_codebooks,
        key_codes,
        keys,
        values,
        value_codebooks,
        value_codes,
        page_starts,
        2,
        ranked,
        query_context_len,
        static_prefix,
        static_suffix,
        page_size,
        1,
        scale,
        1.0,
        2,
    )
    if not torch.allclose(ref_scoreless, got_scoreless, atol=2e-3, rtol=2e-3):
        raise AssertionError("decode scoreless fused wrapper mismatches separate fused top-k + tail attention")
    ranked_logits_decode = torch.empty_like(ranked_scores_decode)
    for head in range(heads):
        kv_head = min(head // 2, kv_heads - 1)
        toks = ranked_decode[head].clamp(min=0, max=total_tokens - 1)
        ranked_logits_decode[head] = (
            torch.sum(q_decode[head].float().unsqueeze(0) * keys[kv_head, toks].float(), dim=-1) * scale
        )
    got_decode_from_logits = gqa_decode_vpq_selected_from_logits_mass_min(
        q_decode,
        keys,
        values,
        value_codebooks,
        value_codes,
        page_starts,
        ranked_decode,
        ranked_scores_decode,
        ranked_logits_decode,
        0.0,
        1,
        2,
        query_context_len,
        static_prefix,
        static_suffix,
        page_size,
        scale,
    )
    ref_decode_selected_only = gqa_decode_vpq_selected_tail_agg_from_scores_mass_min(
        q_decode,
        keys,
        values,
        dense_decode,
        value_codebooks,
        value_codes,
        page_starts,
        ranked_decode,
        ranked_scores_decode,
        0.0,
        1,
        2,
        query_context_len,
        static_prefix,
        static_suffix,
        page_size,
        scale,
        0.0,
    )
    if not torch.allclose(ref_decode_selected_only, got_decode_from_logits, atol=2e-3, rtol=2e-3):
        raise AssertionError("decode selected-from-logits mass+min path mismatches selected-only reference")
    got_decode_tail_from_logits = gqa_decode_vpq_selected_tail_agg_from_logits_mass_min(
        q_decode,
        keys,
        values,
        dense_decode,
        value_codebooks,
        value_codes,
        page_starts,
        ranked_decode,
        ranked_scores_decode,
        ranked_logits_decode,
        0.0,
        1,
        2,
        query_context_len,
        static_prefix,
        static_suffix,
        page_size,
        scale,
        1.0,
    )
    if not torch.allclose(ref_decode, got_decode_tail_from_logits, atol=2e-3, rtol=2e-3):
        raise AssertionError("decode tail-from-logits mass+min path mismatches tail reference")
    base_lse_decode, _ = _gpu_gqa_base_logsumexp_decode(
        queries=q_decode,
        keys_all=keys,
        group_size=2,
        query_context_len=query_context_len,
        static_prefix=static_prefix,
        static_suffix=static_suffix,
        page_size=page_size,
        scale=scale,
    )
    prefix_end = min(max(0, static_prefix), query_context_len)
    indexed_end = max(prefix_end, query_context_len - max(0, static_suffix))
    sealed_end = prefix_end + (max(0, indexed_end - prefix_end) // page_size) * page_size
    base_rank_mask = ((ranked_decode >= 0) & (ranked_decode < prefix_end)) | (
        (ranked_decode >= sealed_end) & (ranked_decode < query_context_len)
    )
    ranked_logits_decode_dynamic = ranked_logits_decode.masked_fill(base_rank_mask, float("-inf"))

    def native_thresholds_from_topk(
        ranked_logits: torch.Tensor,
        ranked_scores: torch.Tensor,
        base_lse: torch.Tensor,
        budgets: list[int],
        exact_mass: float,
        min_top: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        valid = torch.isfinite(ranked_scores) & torch.isfinite(ranked_logits)
        logits = torch.where(valid, ranked_logits.float(), torch.full_like(ranked_logits.float(), float("-inf")))
        prefix_lse = torch.logcumsumexp(logits, dim=-1)
        prefix_valid = torch.cumsum(valid.to(torch.long), dim=-1)
        top_logits, top_order = torch.topk(logits, k=int(logits.shape[-1]), dim=-1, largest=True, sorted=True)
        budget_tensor = torch.tensor(budgets, dtype=torch.long, device=ranked_logits.device)
        thresholds, threshold_sels, sufficient = selected_mass_thresholds_from_topk(
            top_logits.contiguous(),
            top_order.contiguous(),
            prefix_lse.contiguous(),
            prefix_valid.contiguous(),
            base_lse.float().contiguous(),
            budget_tensor,
            float(exact_mass),
            int(min_top),
        )
        if not bool(torch.all(sufficient.to(torch.bool))):
            raise AssertionError("native top-k threshold helper reported insufficient full top-k")
        return thresholds, threshold_sels

    threshold_mass = 0.5
    ref_decode_tail_mass = gqa_decode_vpq_selected_tail_agg_from_logits_mass_min(
        q_decode,
        keys,
        values,
        dense_decode,
        value_codebooks,
        value_codes,
        page_starts,
        ranked_decode,
        ranked_scores_decode,
        ranked_logits_decode,
        threshold_mass,
        1,
        2,
        query_context_len,
        static_prefix,
        static_suffix,
        page_size,
        scale,
        1.0,
    )
    threshold_decode, threshold_sel_decode = selected_mass_thresholds_from_logits_gpu(
        ranked_logits=ranked_logits_decode_dynamic,
        ranked_scores=ranked_scores_decode,
        base_logsumexp=base_lse_decode,
        budgets=[ranked],
        exact_mass=threshold_mass,
        min_top=1,
    )
    native_threshold_decode, native_threshold_sel_decode = native_thresholds_from_topk(
        ranked_logits_decode_dynamic,
        ranked_scores_decode,
        base_lse_decode,
        [ranked],
        threshold_mass,
        1,
    )
    if not torch.equal(threshold_sel_decode, native_threshold_sel_decode) or not torch.allclose(
        threshold_decode,
        native_threshold_decode,
        equal_nan=True,
    ):
        raise AssertionError("native top-k threshold helper mismatches Python threshold helper")
    got_decode_tail_from_thresholds = gqa_decode_vpq_selected_tail_agg_from_logits_mass_min_thresholds(
        q_decode,
        keys,
        values,
        dense_decode,
        value_codebooks,
        value_codes,
        page_starts,
        ranked_decode,
        ranked_scores_decode,
        ranked_logits_decode,
        threshold_decode[:, 0].contiguous(),
        threshold_sel_decode[:, 0].contiguous(),
        threshold_mass,
        1,
        2,
        query_context_len,
        static_prefix,
        static_suffix,
        page_size,
        scale,
        1.0,
    )
    if not torch.allclose(ref_decode_tail_mass, got_decode_tail_from_thresholds, atol=2e-3, rtol=2e-3):
        raise AssertionError("decode from-logits threshold path mismatches mass+min path")

    geo_tail_budgets, geo_probe_budgets = geometric_budget_pairs(
        min_budget=1,
        max_budget=ranked,
        granularity=1,
        growth=1.5,
        probe_scale=1.5,
    )
    geo_approx_thresholds, geo_approx_threshold_sels = selected_mass_thresholds_from_logits_gpu(
        ranked_logits=ranked_logits_decode_dynamic,
        ranked_scores=ranked_scores_decode,
        base_logsumexp=base_lse_decode,
        budgets=geo_tail_budgets,
        exact_mass=threshold_mass,
        min_top=1,
    )
    geo_probe_thresholds, geo_probe_threshold_sels = selected_mass_thresholds_from_logits_gpu(
        ranked_logits=ranked_logits_decode_dynamic,
        ranked_scores=ranked_scores_decode,
        base_logsumexp=base_lse_decode,
        budgets=geo_probe_budgets,
        exact_mass=threshold_mass,
        min_top=1,
    )
    fused_counts, fused_output = gqa_decode_geometric_output_vpq_mass_min_proxy_from_logits_thresholds(
        q_decode,
        keys,
        values,
        dense_decode,
        value_codebooks,
        value_codes,
        page_starts,
        ranked_decode,
        ranked_scores_decode,
        ranked_logits_decode,
        geo_approx_thresholds,
        geo_approx_threshold_sels,
        geo_probe_thresholds,
        geo_probe_threshold_sels,
        2,
        query_context_len,
        static_prefix,
        static_suffix,
        page_size,
        1,
        ranked,
        1,
        1.5,
        1.5,
        0.35,
        threshold_mass,
        1,
        scale,
        0.0,
        1.0,
        -1.0,
        float("inf"),
        True,
        False,
        1.0,
    )
    ref_fused_counts = gqa_decode_geometric_accept_counts_vpq_mass_min_proxy_from_logits_thresholds(
        q_decode,
        keys,
        values,
        dense_decode,
        value_codebooks,
        value_codes,
        page_starts,
        ranked_decode,
        ranked_scores_decode,
        ranked_logits_decode,
        geo_approx_thresholds,
        geo_approx_threshold_sels,
        geo_probe_thresholds,
        geo_probe_threshold_sels,
        2,
        query_context_len,
        static_prefix,
        static_suffix,
        page_size,
        1,
        ranked,
        1,
        1.5,
        1.5,
        0.35,
        threshold_mass,
        1,
        scale,
        0.0,
        1.0,
        -1.0,
        float("inf"),
        True,
        False,
    )
    if fused_counts.detach().cpu().tolist() != ref_fused_counts.detach().cpu().tolist():
        raise AssertionError(
            f"fused geometric output counts mismatch: got={fused_counts.detach().cpu().tolist()} "
            f"expected={ref_fused_counts.detach().cpu().tolist()}"
        )
    fused_thresholds, fused_threshold_sels = select_thresholds_for_budget_counts_gpu(
        thresholds=geo_probe_thresholds,
        threshold_sels=geo_probe_threshold_sels,
        budgets=geo_probe_budgets,
        counts=fused_counts,
    )
    rank_ids = torch.arange(ranked, dtype=torch.long, device=device).reshape(1, ranked)
    fused_rank_mask = rank_ids >= fused_counts.unsqueeze(-1)
    ref_fused_output = gqa_decode_vpq_selected_tail_agg_from_logits_mass_min_thresholds(
        q_decode,
        keys,
        values,
        dense_decode,
        value_codebooks,
        value_codes,
        page_starts,
        ranked_decode,
        ranked_scores_decode.masked_fill(fused_rank_mask, float("-inf")).contiguous(),
        ranked_logits_decode,
        fused_thresholds,
        fused_threshold_sels,
        threshold_mass,
        1,
        2,
        query_context_len,
        static_prefix,
        static_suffix,
        page_size,
        scale,
        1.0,
    )
    if not torch.allclose(ref_fused_output, fused_output, atol=3e-3, rtol=3e-3):
        max_diff = float(torch.max(torch.abs(ref_fused_output - fused_output)).detach().cpu().item())
        raise AssertionError(f"fused geometric output mismatch: max_diff={max_diff}")

    env_keys = [
        "SELECTOR_PQ_FUSED_DIM_SCAN_OUTPUT",
        "SELECTOR_PQ_PRECOMPUTE_RANK_WEIGHTS",
        "SELECTOR_PQ_SELECTED_CODEWEIGHT_DELTAS",
        "SELECTOR_PQ_SELECTED_EXACT_LISTS",
        "SELECTOR_PQ_GEOMETRIC_TWO_PASS_OUTPUT",
    ]
    old_env = {key: os.environ.get(key) for key in env_keys}
    try:
        os.environ.update(
            {
                "SELECTOR_PQ_FUSED_DIM_SCAN_OUTPUT": "1",
                "SELECTOR_PQ_PRECOMPUTE_RANK_WEIGHTS": "1",
                "SELECTOR_PQ_SELECTED_CODEWEIGHT_DELTAS": "1",
                "SELECTOR_PQ_SELECTED_EXACT_LISTS": "1",
                "SELECTOR_PQ_GEOMETRIC_TWO_PASS_OUTPUT": "0",
            }
        )
        dim_counts, dim_output = gqa_decode_geometric_output_vpq_mass_min_proxy_from_logits_thresholds(
            q_decode,
            keys,
            values,
            dense_decode,
            value_codebooks,
            value_codes,
            page_starts,
            ranked_decode,
            ranked_scores_decode,
            ranked_logits_decode,
            geo_approx_thresholds,
            geo_approx_threshold_sels,
            geo_probe_thresholds,
            geo_probe_threshold_sels,
            2,
            query_context_len,
            static_prefix,
            static_suffix,
            page_size,
            1,
            ranked,
            1,
            1.5,
            1.5,
            0.35,
            threshold_mass,
            1,
            scale,
            0.0,
            1.0,
            -1.0,
            float("inf"),
            True,
            False,
            1.0,
        )
        os.environ["SELECTOR_PQ_GEOMETRIC_TWO_PASS_OUTPUT"] = "1"
        two_counts, two_output = gqa_decode_geometric_output_vpq_mass_min_proxy_from_logits_thresholds(
            q_decode,
            keys,
            values,
            dense_decode,
            value_codebooks,
            value_codes,
            page_starts,
            ranked_decode,
            ranked_scores_decode,
            ranked_logits_decode,
            geo_approx_thresholds,
            geo_approx_threshold_sels,
            geo_probe_thresholds,
            geo_probe_threshold_sels,
            2,
            query_context_len,
            static_prefix,
            static_suffix,
            page_size,
            1,
            ranked,
            1,
            1.5,
            1.5,
            0.35,
            threshold_mass,
            1,
            scale,
            0.0,
            1.0,
            -1.0,
            float("inf"),
            True,
            False,
            1.0,
        )
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    if dim_counts.detach().cpu().tolist() != two_counts.detach().cpu().tolist():
        raise AssertionError(
            f"two-pass fused counts mismatch: got={two_counts.detach().cpu().tolist()} "
            f"expected={dim_counts.detach().cpu().tolist()}"
        )
    if not torch.allclose(dim_output, two_output, atol=3e-3, rtol=3e-3):
        max_diff = float(torch.max(torch.abs(dim_output - two_output)).detach().cpu().item())
        raise AssertionError(f"two-pass fused output mismatch: max_diff={max_diff}")
    if not torch.allclose(ref_fused_output, two_output, atol=3e-3, rtol=3e-3):
        max_diff = float(torch.max(torch.abs(ref_fused_output - two_output)).detach().cpu().item())
        raise AssertionError(f"two-pass fused output/reference mismatch: max_diff={max_diff}")

    keys_pad = torch.empty((kv_heads, total_tokens, dim + 1), device=device, dtype=keys.dtype)
    values_pad = torch.empty((kv_heads, total_tokens, dim + 1), device=device, dtype=values.dtype)
    keys_strided = keys_pad[..., :dim]
    values_strided = values_pad[..., :dim]
    keys_strided.copy_(keys)
    values_strided.copy_(values)
    if keys_strided.is_contiguous() or values_strided.is_contiguous():
        raise AssertionError("test setup failed to create strided K/V views")
    got_decode_strided = gqa_decode_vpq_selected_tail_agg_from_scores(
        q_decode,
        keys_strided,
        values_strided,
        dense_decode,
        value_codebooks,
        value_codes,
        page_starts,
        ranked_decode,
        ranked_scores_decode,
        2,
        query_context_len,
        static_prefix,
        static_suffix,
        page_size,
        1,
        scale,
        1.0,
    )
    if not torch.allclose(ref_decode, got_decode_strided, atol=2e-3, rtol=2e-3):
        raise AssertionError("decode from-scores path mismatches with strided K/V")
    got_decode_tail_from_logits_strided = gqa_decode_vpq_selected_tail_agg_from_logits_mass_min(
        q_decode,
        keys_strided,
        values_strided,
        dense_decode,
        value_codebooks,
        value_codes,
        page_starts,
        ranked_decode,
        ranked_scores_decode,
        ranked_logits_decode,
        0.0,
        1,
        2,
        query_context_len,
        static_prefix,
        static_suffix,
        page_size,
        scale,
        1.0,
    )
    if not torch.allclose(ref_decode, got_decode_tail_from_logits_strided, atol=2e-3, rtol=2e-3):
        raise AssertionError("decode from-logits path mismatches with strided K/V")

    def round_up_budget(value: int, granularity: int, max_budget: int) -> int:
        if granularity <= 1:
            return min(value, max_budget)
        return min(((value + granularity - 1) // granularity) * granularity, max_budget)

    def mask_scores(keep: int) -> torch.Tensor:
        keep_i = max(0, min(ranked, int(keep)))
        return ranked_scores_decode.masked_fill(rank_ids >= keep_i, float("-inf")).contiguous()

    min_budget = 1
    max_budget = ranked
    granularity = 1
    growth = 1.5
    probe_scale = 1.5
    rel_l2_max = 0.35
    expected_counts = torch.full((heads,), max_budget, dtype=torch.long, device=device)
    unresolved = torch.ones((heads,), dtype=torch.bool, device=device)
    k = round_up_budget(min_budget, granularity, max_budget)
    while True:
        tail_budget = min(max_budget, int(k))
        probe_budget = round_up_budget(
            int(np.ceil(max(float(tail_budget + granularity), probe_scale * float(tail_budget)))),
            granularity,
            max_budget,
        )
        probe_budget = max(tail_budget, int(probe_budget))
        approx_tail = gqa_decode_vpq_selected_tail_agg_from_scores(
            q_decode,
            keys,
            values,
            dense_decode,
            value_codebooks,
            value_codes,
            page_starts,
            ranked_decode,
            mask_scores(tail_budget),
            2,
            query_context_len,
            static_prefix,
            static_suffix,
            page_size,
            ranked,
            scale,
            1.0,
        )
        probe_only = gqa_decode_vpq_selected_tail_agg_from_scores(
            q_decode,
            keys,
            values,
            dense_decode,
            value_codebooks,
            value_codes,
            page_starts,
            ranked_decode,
            mask_scores(probe_budget),
            2,
            query_context_len,
            static_prefix,
            static_suffix,
            page_size,
            ranked,
            scale,
            0.0,
        )
        rel = torch.linalg.vector_norm(approx_tail - probe_only, dim=-1) / torch.clamp(
            torch.linalg.vector_norm(probe_only, dim=-1),
            min=1e-20,
        )
        passed = (rel <= rel_l2_max) & unresolved
        expected_counts = torch.where(passed, torch.full_like(expected_counts, probe_budget), expected_counts)
        unresolved = unresolved & ~passed
        if not bool(torch.any(unresolved)) or probe_budget >= max_budget:
            break
        next_k = round_up_budget(
            int(np.ceil(max(float(probe_budget + granularity), growth * float(probe_budget)))),
            granularity,
            max_budget,
        )
        if next_k <= probe_budget:
            break
        k = next_k
    got_counts = gqa_decode_geometric_accept_counts(
        q_decode,
        keys,
        values,
        dense_decode,
        value_codebooks,
        value_codes,
        page_starts,
        ranked_decode,
        ranked_scores_decode,
        2,
        query_context_len,
        static_prefix,
        static_suffix,
        page_size,
        min_budget,
        max_budget,
        granularity,
        growth,
        probe_scale,
        rel_l2_max,
        scale,
    )
    if got_counts.detach().cpu().tolist() != expected_counts.detach().cpu().tolist():
        raise AssertionError(
            f"decode geometric accept counts mismatch: got={got_counts.detach().cpu().tolist()} "
            f"expected={expected_counts.detach().cpu().tolist()}"
        )
    for vpq_exact_top in (-2, 0):
        expected_counts_vpq = torch.full((heads,), max_budget, dtype=torch.long, device=device)
        unresolved_vpq = torch.ones((heads,), dtype=torch.bool, device=device)
        k = round_up_budget(min_budget, granularity, max_budget)
        while True:
            tail_budget = min(max_budget, int(k))
            probe_budget = round_up_budget(
                int(np.ceil(max(float(tail_budget + granularity), probe_scale * float(tail_budget)))),
                granularity,
                max_budget,
            )
            probe_budget = max(tail_budget, int(probe_budget))
            approx_tail = gqa_decode_vpq_selected_tail_agg_from_scores(
                q_decode,
                keys,
                values,
                dense_decode,
                value_codebooks,
                value_codes,
                page_starts,
                ranked_decode,
                mask_scores(tail_budget),
                2,
                query_context_len,
                static_prefix,
                static_suffix,
                page_size,
                vpq_exact_top,
                scale,
                1.0,
            )
            probe_only = gqa_decode_vpq_selected_tail_agg_from_scores(
                q_decode,
                keys,
                values,
                dense_decode,
                value_codebooks,
                value_codes,
                page_starts,
                ranked_decode,
                mask_scores(probe_budget),
                2,
                query_context_len,
                static_prefix,
                static_suffix,
                page_size,
                vpq_exact_top,
                scale,
                0.0,
            )
            rel = torch.linalg.vector_norm(approx_tail - probe_only, dim=-1) / torch.clamp(
                torch.linalg.vector_norm(probe_only, dim=-1),
                min=1e-20,
            )
            passed = (rel <= rel_l2_max) & unresolved_vpq
            expected_counts_vpq = torch.where(
                passed,
                torch.full_like(expected_counts_vpq, probe_budget),
                expected_counts_vpq,
            )
            unresolved_vpq = unresolved_vpq & ~passed
            if not bool(torch.any(unresolved_vpq)) or probe_budget >= max_budget:
                break
            next_k = round_up_budget(
                int(np.ceil(max(float(probe_budget + granularity), growth * float(probe_budget)))),
                granularity,
                max_budget,
            )
            if next_k <= probe_budget:
                break
            k = next_k
        got_counts_vpq = gqa_decode_geometric_accept_counts_vpq(
            q_decode,
            keys,
            values,
            dense_decode,
            value_codebooks,
            value_codes,
            page_starts,
            ranked_decode,
            ranked_scores_decode,
            2,
            query_context_len,
            static_prefix,
            static_suffix,
            page_size,
            min_budget,
            max_budget,
            granularity,
            growth,
            probe_scale,
            rel_l2_max,
            vpq_exact_top,
            scale,
        )
        if got_counts_vpq.detach().cpu().tolist() != expected_counts_vpq.detach().cpu().tolist():
            raise AssertionError(
                f"decode VPQ geometric counts mismatch exact_top={vpq_exact_top}: "
                f"got={got_counts_vpq.detach().cpu().tolist()} "
                f"expected={expected_counts_vpq.detach().cpu().tolist()}"
            )
        expected_tail_stability = torch.full((heads,), max_budget, dtype=torch.long, device=device)
        unresolved_tail_stability = torch.ones((heads,), dtype=torch.bool, device=device)
        k = round_up_budget(min_budget, granularity, max_budget)
        while True:
            tail_budget = min(max_budget, int(k))
            probe_budget = round_up_budget(
                int(np.ceil(max(float(tail_budget + granularity), probe_scale * float(tail_budget)))),
                granularity,
                max_budget,
            )
            probe_budget = max(tail_budget, int(probe_budget))
            approx_tail = gqa_decode_vpq_selected_tail_agg_from_scores(
                q_decode,
                keys,
                values,
                dense_decode,
                value_codebooks,
                value_codes,
                page_starts,
                ranked_decode,
                mask_scores(tail_budget),
                2,
                query_context_len,
                static_prefix,
                static_suffix,
                page_size,
                vpq_exact_top,
                scale,
                1.0,
            )
            probe_tail = gqa_decode_vpq_selected_tail_agg_from_scores(
                q_decode,
                keys,
                values,
                dense_decode,
                value_codebooks,
                value_codes,
                page_starts,
                ranked_decode,
                mask_scores(probe_budget),
                2,
                query_context_len,
                static_prefix,
                static_suffix,
                page_size,
                vpq_exact_top,
                scale,
                1.0,
            )
            rel = torch.linalg.vector_norm(approx_tail - probe_tail, dim=-1) / torch.clamp(
                torch.linalg.vector_norm(probe_tail, dim=-1),
                min=1e-20,
            )
            passed = (rel <= rel_l2_max) & unresolved_tail_stability
            expected_tail_stability = torch.where(
                passed,
                torch.full_like(expected_tail_stability, probe_budget),
                expected_tail_stability,
            )
            unresolved_tail_stability = unresolved_tail_stability & ~passed
            if not bool(torch.any(unresolved_tail_stability)) or probe_budget >= max_budget:
                break
            next_k = round_up_budget(
                int(np.ceil(max(float(probe_budget + granularity), growth * float(probe_budget)))),
                granularity,
                max_budget,
            )
            if next_k <= probe_budget:
                break
            k = next_k
        got_tail_stability = gqa_decode_geometric_accept_counts_vpq_tail_stability(
            q_decode,
            keys,
            values,
            dense_decode,
            value_codebooks,
            value_codes,
            page_starts,
            ranked_decode,
            ranked_scores_decode,
            2,
            query_context_len,
            static_prefix,
            static_suffix,
            page_size,
            min_budget,
            max_budget,
            granularity,
            growth,
            probe_scale,
            rel_l2_max,
            vpq_exact_top,
            scale,
        )
        if got_tail_stability.detach().cpu().tolist() != expected_tail_stability.detach().cpu().tolist():
            raise AssertionError(
                f"decode VPQ tail-stability counts mismatch exact_top={vpq_exact_top}: "
                f"got={got_tail_stability.detach().cpu().tolist()} "
                f"expected={expected_tail_stability.detach().cpu().tolist()}"
            )
        got_proxy_inactive = gqa_decode_geometric_accept_counts_vpq_proxy(
            q_decode,
            keys,
            values,
            dense_decode,
            value_codebooks,
            value_codes,
            page_starts,
            ranked_decode,
            ranked_scores_decode,
            2,
            query_context_len,
            static_prefix,
            static_suffix,
            page_size,
            min_budget,
            max_budget,
            granularity,
            growth,
            probe_scale,
            rel_l2_max,
            vpq_exact_top,
            scale,
            0.0,
            1.0,
            -1.0,
            float("inf"),
            True,
            False,
        )
        if got_proxy_inactive.detach().cpu().tolist() != expected_counts_vpq.detach().cpu().tolist():
            raise AssertionError(
                f"decode VPQ proxy inactive counts mismatch exact_top={vpq_exact_top}: "
                f"got={got_proxy_inactive.detach().cpu().tolist()} "
                f"expected={expected_counts_vpq.detach().cpu().tolist()}"
            )
        got_proxy_forced_fail = gqa_decode_geometric_accept_counts_vpq_proxy(
            q_decode,
            keys,
            values,
            dense_decode,
            value_codebooks,
            value_codes,
            page_starts,
            ranked_decode,
            ranked_scores_decode,
            2,
            query_context_len,
            static_prefix,
            static_suffix,
            page_size,
            min_budget,
            max_budget,
            granularity,
            growth,
            probe_scale,
            rel_l2_max,
            vpq_exact_top,
            scale,
            0.0,
            1.0,
            2.0,
            float("inf"),
            True,
            False,
        )
        if got_proxy_forced_fail.detach().cpu().tolist() != [max_budget] * heads:
            raise AssertionError(
                f"decode VPQ proxy forced-fail counts mismatch exact_top={vpq_exact_top}: "
                f"got={got_proxy_forced_fail.detach().cpu().tolist()}"
            )

    prefill_rank_ids = torch.arange(ranked, dtype=torch.long, device=device).reshape(1, 1, ranked)

    def mask_prefill_scores(keep: int) -> torch.Tensor:
        keep_i = max(0, min(ranked, int(keep)))
        return ranked_scores.masked_fill(prefill_rank_ids >= keep_i, float("-inf")).contiguous()

    expected_prefill_counts = torch.full((positions, heads), max_budget, dtype=torch.long, device=device)
    unresolved_prefill = torch.ones((positions, heads), dtype=torch.bool, device=device)
    k = round_up_budget(min_budget, granularity, max_budget)
    while True:
        tail_budget = min(max_budget, int(k))
        probe_budget = round_up_budget(
            int(np.ceil(max(float(tail_budget + granularity), probe_scale * float(tail_budget)))),
            granularity,
            max_budget,
        )
        probe_budget = max(tail_budget, int(probe_budget))
        approx_tail = gqa_causal_vpq_tail_from_scores(
            queries,
            keys,
            values,
            dense_pq_scores,
            value_codebooks,
            value_codes,
            page_starts,
            ranked_tokens,
            mask_prefill_scores(tail_budget),
            2,
            query_start,
            static_prefix,
            static_suffix,
            page_size,
            scale,
            1.0,
        )
        probe_only = gqa_causal_vpq_tail_from_scores(
            queries,
            keys,
            values,
            dense_pq_scores,
            value_codebooks,
            value_codes,
            page_starts,
            ranked_tokens,
            mask_prefill_scores(probe_budget),
            2,
            query_start,
            static_prefix,
            static_suffix,
            page_size,
            scale,
            0.0,
        )
        rel = torch.linalg.vector_norm(approx_tail - probe_only, dim=-1) / torch.clamp(
            torch.linalg.vector_norm(probe_only, dim=-1),
            min=1e-20,
        )
        passed = (rel <= rel_l2_max) & unresolved_prefill
        expected_prefill_counts = torch.where(
            passed,
            torch.full_like(expected_prefill_counts, probe_budget),
            expected_prefill_counts,
        )
        unresolved_prefill = unresolved_prefill & ~passed
        if not bool(torch.any(unresolved_prefill)) or probe_budget >= max_budget:
            break
        next_k = round_up_budget(
            int(np.ceil(max(float(probe_budget + granularity), growth * float(probe_budget)))),
            granularity,
            max_budget,
        )
        if next_k <= probe_budget:
            break
        k = next_k
    got_prefill_counts = gqa_causal_geometric_accept_counts(
        queries,
        keys,
        values,
        dense_pq_scores,
        value_codebooks,
        value_codes,
        page_starts,
        ranked_tokens,
        ranked_scores,
        2,
        query_start,
        static_prefix,
        static_suffix,
        page_size,
        min_budget,
        max_budget,
        granularity,
        growth,
        probe_scale,
        rel_l2_max,
        scale,
    )
    if got_prefill_counts.detach().cpu().tolist() != expected_prefill_counts.detach().cpu().tolist():
        raise AssertionError(
            f"prefill geometric accept counts mismatch: got={got_prefill_counts.detach().cpu().tolist()} "
            f"expected={expected_prefill_counts.detach().cpu().tolist()}"
        )
    for vpq_exact_top in (-2, 0):
        expected_prefill_counts_vpq = torch.full((positions, heads), max_budget, dtype=torch.long, device=device)
        unresolved_prefill_vpq = torch.ones((positions, heads), dtype=torch.bool, device=device)
        k = round_up_budget(min_budget, granularity, max_budget)
        while True:
            tail_budget = min(max_budget, int(k))
            probe_budget = round_up_budget(
                int(np.ceil(max(float(tail_budget + granularity), probe_scale * float(tail_budget)))),
                granularity,
                max_budget,
            )
            probe_budget = max(tail_budget, int(probe_budget))
            approx_tail = gqa_causal_vpq_selected_tail_from_scores(
                queries,
                keys,
                values,
                dense_pq_scores,
                value_codebooks,
                value_codes,
                page_starts,
                ranked_tokens,
                mask_prefill_scores(tail_budget),
                2,
                query_start,
                static_prefix,
                static_suffix,
                page_size,
                vpq_exact_top,
                scale,
                1.0,
            )
            probe_only = gqa_causal_vpq_selected_tail_from_scores(
                queries,
                keys,
                values,
                dense_pq_scores,
                value_codebooks,
                value_codes,
                page_starts,
                ranked_tokens,
                mask_prefill_scores(probe_budget),
                2,
                query_start,
                static_prefix,
                static_suffix,
                page_size,
                vpq_exact_top,
                scale,
                0.0,
            )
            rel = torch.linalg.vector_norm(approx_tail - probe_only, dim=-1) / torch.clamp(
                torch.linalg.vector_norm(probe_only, dim=-1),
                min=1e-20,
            )
            passed = (rel <= rel_l2_max) & unresolved_prefill_vpq
            expected_prefill_counts_vpq = torch.where(
                passed,
                torch.full_like(expected_prefill_counts_vpq, probe_budget),
                expected_prefill_counts_vpq,
            )
            unresolved_prefill_vpq = unresolved_prefill_vpq & ~passed
            if not bool(torch.any(unresolved_prefill_vpq)) or probe_budget >= max_budget:
                break
            next_k = round_up_budget(
                int(np.ceil(max(float(probe_budget + granularity), growth * float(probe_budget)))),
                granularity,
                max_budget,
            )
            if next_k <= probe_budget:
                break
            k = next_k
        got_prefill_counts_vpq = gqa_causal_geometric_accept_counts_vpq(
            queries,
            keys,
            values,
            dense_pq_scores,
            value_codebooks,
            value_codes,
            page_starts,
            ranked_tokens,
            ranked_scores,
            2,
            query_start,
            static_prefix,
            static_suffix,
            page_size,
            min_budget,
            max_budget,
            granularity,
            growth,
            probe_scale,
            rel_l2_max,
            vpq_exact_top,
            scale,
        )
        if got_prefill_counts_vpq.detach().cpu().tolist() != expected_prefill_counts_vpq.detach().cpu().tolist():
            raise AssertionError(
                f"prefill VPQ geometric counts mismatch exact_top={vpq_exact_top}: "
                f"got={got_prefill_counts_vpq.detach().cpu().tolist()} "
                f"expected={expected_prefill_counts_vpq.detach().cpu().tolist()}"
            )


def _test_dense_decode_ranked_logits_simulator() -> None:
    torch.manual_seed(20260520)
    device = torch.device("cuda")
    heads = 4
    kv_heads = 2
    group_size = 2
    dim = 16
    total_tokens = 48
    query_context_len = 37
    ranked = 13
    static_prefix = 3
    static_suffix = 5
    page_size = 8
    scale = float(dim) ** -0.5
    queries = torch.randn((heads, dim), device=device, dtype=torch.float32)
    keys = torch.randn((kv_heads, total_tokens, dim), device=device, dtype=torch.float16)
    ranked_tokens = torch.randint(0, query_context_len, (heads, ranked), device=device, dtype=torch.long)

    ref_ranked = _gpu_gqa_ranked_exact_logits(
        queries=queries,
        keys_all=keys,
        ranked_tokens=ranked_tokens,
        group_size=group_size,
        scale=scale,
        max_rank=ranked,
    )
    ref_base, ref_base_count = _gpu_gqa_base_logsumexp_decode(
        queries=queries,
        keys_all=keys,
        group_size=group_size,
        query_context_len=query_context_len,
        static_prefix=static_prefix,
        static_suffix=static_suffix,
        page_size=page_size,
        scale=scale,
    )
    got_ranked, got_base, got_base_count, got_key_count = _gpu_gqa_dense_decode_ranked_logits_and_base_lse(
        queries=queries,
        keys_all=keys,
        keys_all_t_float=None,
        ranked_tokens=ranked_tokens,
        group_size=group_size,
        scale=scale,
        max_rank=ranked,
        query_context_len=query_context_len,
        static_prefix=static_prefix,
        static_suffix=static_suffix,
        page_size=page_size,
        need_base_lse=True,
    )
    if int(got_key_count) != int(query_context_len):
        raise AssertionError(f"dense simulator key_count mismatch: got={got_key_count} expected={query_context_len}")
    if int(got_base_count) != int(ref_base_count):
        raise AssertionError(f"dense simulator base_count mismatch: got={got_base_count} expected={ref_base_count}")
    if got_base is None:
        raise AssertionError("dense simulator did not return requested base logsumexp")
    if not torch.allclose(got_ranked, ref_ranked, atol=2e-3, rtol=2e-3):
        max_diff = float((got_ranked - ref_ranked).abs().max().item())
        raise AssertionError(f"dense simulator ranked logits mismatch: max_diff={max_diff}")
    if not torch.allclose(got_base, ref_base, atol=2e-3, rtol=2e-3):
        max_diff = float((got_base - ref_base).abs().max().item())
        raise AssertionError(f"dense simulator base lse mismatch: max_diff={max_diff}")
    keys_t_float = keys.float().transpose(1, 2).contiguous()
    got_ranked_cached, got_base_cached, got_base_count_cached, _ = _gpu_gqa_dense_decode_ranked_logits_and_base_lse(
        queries=queries,
        keys_all=keys,
        keys_all_t_float=keys_t_float,
        ranked_tokens=ranked_tokens,
        group_size=group_size,
        scale=scale,
        max_rank=ranked,
        query_context_len=query_context_len,
        static_prefix=static_prefix,
        static_suffix=static_suffix,
        page_size=page_size,
        need_base_lse=True,
    )
    if int(got_base_count_cached) != int(ref_base_count):
        raise AssertionError("cached dense simulator base_count mismatch")
    if not torch.allclose(got_ranked_cached, ref_ranked, atol=2e-3, rtol=2e-3):
        max_diff = float((got_ranked_cached - ref_ranked).abs().max().item())
        raise AssertionError(f"cached dense simulator ranked logits mismatch: max_diff={max_diff}")
    if got_base_cached is None or not torch.allclose(got_base_cached, ref_base, atol=2e-3, rtol=2e-3):
        max_diff = float((got_base_cached - ref_base).abs().max().item()) if got_base_cached is not None else float("inf")
        raise AssertionError(f"cached dense simulator base lse mismatch: max_diff={max_diff}")
    from selector_paged_pq import (  # noqa: PLC0415
        gqa_decode_full_exact_logits,
        gqa_decode_full_exact_logits_grouped,
        gqa_decode_full_exact_logits_t_cublas,
        gqa_decode_token_exact_logits,
        gqa_decode_ranked_exact_logits,
        gqa_decode_ranked_exact_logits_with_base_lse,
        joint_sparse_exact_score_table,
    )

    ref_full = torch.empty((heads, query_context_len), device=device, dtype=torch.float32)
    for head in range(heads):
        kv_head = min(kv_heads - 1, head // group_size)
        ref_full[head] = (queries[head].float().reshape(1, dim) @ keys[kv_head, :query_context_len].float().T).reshape(-1) * scale
    got_full = gqa_decode_full_exact_logits(
        queries,
        keys,
        group_size,
        query_context_len,
        scale,
    )
    if not torch.allclose(got_full, ref_full, atol=2e-3, rtol=2e-3):
        max_diff = float((got_full - ref_full).abs().max().item())
        raise AssertionError(f"native full exact logits mismatch: max_diff={max_diff}")
    got_full_grouped = gqa_decode_full_exact_logits_grouped(
        queries,
        keys,
        group_size,
        query_context_len,
        scale,
    )
    if not torch.allclose(got_full_grouped, ref_full, atol=2e-3, rtol=2e-3):
        max_diff = float((got_full_grouped - ref_full).abs().max().item())
        raise AssertionError(f"native grouped full exact logits mismatch: max_diff={max_diff}")
    keys_t = keys.float().transpose(1, 2).contiguous()
    got_full_cublas = gqa_decode_full_exact_logits_t_cublas(
        queries,
        keys_t,
        group_size,
        query_context_len,
        scale,
    )
    if not torch.allclose(got_full_cublas, ref_full, atol=2e-3, rtol=2e-3):
        max_diff = float((got_full_cublas - ref_full).abs().max().item())
        raise AssertionError(f"native cuBLAS full exact logits mismatch: max_diff={max_diff}")
    padded_keys_t = torch.empty(
        (kv_heads, dim, query_context_len + 7),
        device=device,
        dtype=torch.float32,
    )
    padded_keys_t[:, :, :query_context_len].copy_(keys_t[:, :, :query_context_len])
    got_full_cublas_strided = gqa_decode_full_exact_logits_t_cublas(
        queries,
        padded_keys_t[:, :, :query_context_len],
        group_size,
        query_context_len,
        scale,
    )
    if not torch.allclose(got_full_cublas_strided, ref_full, atol=2e-3, rtol=2e-3):
        max_diff = float((got_full_cublas_strided - ref_full).abs().max().item())
        raise AssertionError(f"native cuBLAS strided full exact logits mismatch: max_diff={max_diff}")
    token_list = torch.tensor(
        [
            [0, 7, 11, query_context_len - 1, query_context_len, -1],
            [2, 5, 13, query_context_len - 2, total_tokens - 1, -4],
            [3, 9, 17, query_context_len - 3, query_context_len + 4, -5],
            [4, 10, 19, query_context_len - 4, total_tokens + 3, -6],
        ],
        device=device,
        dtype=torch.long,
    )
    got_token_exact = gqa_decode_token_exact_logits(
        queries,
        keys,
        token_list,
        group_size,
        query_context_len,
        scale,
    )
    ref_token_exact = torch.full_like(got_token_exact, -float("inf"))
    for head in range(heads):
        kv_head = min(kv_heads - 1, head // group_size)
        for sel, token in enumerate(token_list[head].detach().cpu().tolist()):
            if 0 <= int(token) < int(query_context_len):
                ref_token_exact[head, sel] = (
                    queries[head].float().reshape(1, dim)
                    @ keys[kv_head, int(token)].float().reshape(dim, 1)
                ).reshape(()) * scale
    finite = torch.isfinite(ref_token_exact)
    if not torch.equal(torch.isfinite(got_token_exact), finite):
        raise AssertionError("native arbitrary-token exact logits finite mask mismatch")
    if bool(torch.any(finite)) and not torch.allclose(
        got_token_exact[finite],
        ref_token_exact[finite],
        atol=2e-3,
        rtol=2e-3,
    ):
        max_diff = float((got_token_exact[finite] - ref_token_exact[finite]).abs().max().item())
        raise AssertionError(f"native arbitrary-token exact logits mismatch: max_diff={max_diff}")
    base_tokens = torch.tensor([0, 1, query_context_len - 3, query_context_len - 1], device=device, dtype=torch.long)
    base_token_rows = base_tokens.reshape(1, -1).expand(heads, -1).contiguous()
    base_logits = gqa_decode_token_exact_logits(
        queries,
        keys,
        base_token_rows,
        group_size,
        query_context_len,
        scale,
    )
    table_ranked_tokens = torch.tensor(
        [
            [5, 9, 17, 23],
            [6, 10, 18, 24],
            [7, 11, 19, 25],
            [8, 12, 20, 26],
        ],
        device=device,
        dtype=torch.long,
    )
    ranked_logits_for_table = gqa_decode_token_exact_logits(
        queries,
        keys,
        table_ranked_tokens,
        group_size,
        query_context_len,
        scale,
    )
    sparse_table = joint_sparse_exact_score_table(
        base_tokens,
        base_logits,
        table_ranked_tokens,
        ranked_logits_for_table,
        query_context_len,
    )
    ref_sparse_table = torch.zeros((heads, query_context_len), device=device, dtype=torch.float32)
    ref_sparse_table.scatter_(1, base_token_rows, base_logits)
    ref_sparse_table.scatter_(1, table_ranked_tokens, ranked_logits_for_table)
    if not torch.allclose(sparse_table, ref_sparse_table, atol=2e-3, rtol=2e-3):
        max_diff = float((sparse_table - ref_sparse_table).abs().max().item())
        raise AssertionError(f"native sparse exact score table mismatch: max_diff={max_diff}")

    ranked_scores = torch.ones_like(ref_ranked)
    ref_native_ranked = gqa_decode_ranked_exact_logits(
        queries,
        keys,
        ranked_tokens,
        ranked_scores,
        group_size,
        query_context_len,
        static_prefix,
        static_suffix,
        page_size,
        scale,
    )
    got_native_ranked, got_native_base = gqa_decode_ranked_exact_logits_with_base_lse(
        queries,
        keys,
        ranked_tokens,
        ranked_scores,
        group_size,
        query_context_len,
        static_prefix,
        static_suffix,
        page_size,
        scale,
    )
    if not torch.allclose(got_native_ranked, ref_native_ranked, atol=2e-3, rtol=2e-3, equal_nan=True):
        finite = torch.isfinite(got_native_ranked) & torch.isfinite(ref_native_ranked)
        max_diff = float((got_native_ranked[finite] - ref_native_ranked[finite]).abs().max().item()) if bool(torch.any(finite)) else float("inf")
        raise AssertionError(f"native ranked logits + base lse ranked mismatch: max_diff={max_diff}")
    if not torch.allclose(got_native_base, ref_base, atol=2e-3, rtol=2e-3):
        max_diff = float((got_native_base - ref_base).abs().max().item())
        raise AssertionError(f"native ranked logits + base lse base mismatch: max_diff={max_diff}")


def _test_grouped_risk_prefix_workspace() -> None:
    from selector_paged_pq import (  # noqa: PLC0415
        joint_grouped_risk_sort_temp_bytes,
        joint_vprefix_outputs_from_grouped_risk_batched,
        joint_vprefix_outputs_from_grouped_risk_batched_strided_workspace,
        joint_vprefix_outputs_from_grouped_risk_batched_workspace,
    )

    torch.manual_seed(20260524)
    device = torch.device("cuda")
    groups = 2
    k_count = 3
    heads = 2
    context_len = 37
    dim = 16
    v_steps = 4
    base_outputs = torch.randn((groups, k_count, heads, dim), device=device, dtype=torch.float32)
    logits = torch.randn((groups, k_count, heads, context_len), device=device, dtype=torch.float32)
    probs = torch.softmax(logits, dim=-1).contiguous()
    residual_groups = torch.randn((groups, context_len, dim), device=device, dtype=torch.float32)
    code_error_groups = torch.rand((groups, context_len), device=device, dtype=torch.float32)
    v_budgets = torch.tensor([0, 5, 17, context_len], device=device, dtype=torch.long)
    ref = joint_vprefix_outputs_from_grouped_risk_batched(
        base_outputs.contiguous(),
        probs,
        residual_groups.contiguous(),
        code_error_groups.contiguous(),
        v_budgets,
    )
    rows = groups * k_count * heads
    total = rows * context_len
    temp_bytes = int(joint_grouped_risk_sort_temp_bytes(rows, context_len))
    risk_in = torch.empty((total + 11,), device=device, dtype=torch.float32)
    risk_out = torch.empty_like(risk_in)
    ids_in = torch.empty((total + 11,), device=device, dtype=torch.int32)
    ids_out = torch.empty_like(ids_in)
    offsets = torch.empty((rows + 8,), device=device, dtype=torch.long)
    temp = torch.empty((temp_bytes + 17,), device=device, dtype=torch.uint8)
    interval_sums = torch.empty((rows, v_steps, dim), device=device, dtype=torch.float32)
    outputs = torch.empty_like(interval_sums)
    got = joint_vprefix_outputs_from_grouped_risk_batched_workspace(
        base_outputs.contiguous(),
        probs,
        residual_groups.contiguous(),
        code_error_groups.contiguous(),
        v_budgets,
        risk_in,
        risk_out,
        ids_in,
        ids_out,
        offsets,
        temp,
        interval_sums,
        outputs,
    )
    if int(got.data_ptr()) != int(outputs.data_ptr()):
        raise AssertionError("workspace grouped risk-prefix did not return the provided output buffer")
    if not torch.allclose(got, ref, atol=2e-5, rtol=2e-5):
        max_diff = float((got - ref).abs().max().item())
        raise AssertionError(f"workspace grouped risk-prefix mismatch: max_diff={max_diff}")

    prob_capacity = context_len + 9
    strided_probs = torch.empty((groups, k_count, heads, prob_capacity), device=device, dtype=torch.float32)
    strided_probs[..., :context_len].copy_(probs)
    strided_probs[..., context_len:].fill_(float("nan"))
    strided_got = joint_vprefix_outputs_from_grouped_risk_batched_strided_workspace(
        base_outputs.contiguous(),
        strided_probs.contiguous(),
        residual_groups.contiguous(),
        code_error_groups.contiguous(),
        v_budgets,
        int(context_len),
        risk_in,
        risk_out,
        ids_in,
        ids_out,
        offsets,
        temp,
        interval_sums,
        outputs,
    )
    if not torch.allclose(strided_got, ref, atol=2e-5, rtol=2e-5):
        max_diff = float((strided_got - ref).abs().max().item())
        raise AssertionError(f"strided workspace grouped risk-prefix mismatch: max_diff={max_diff}")


def _test_grouped_risk_prefix_compact_vpq() -> None:
    from selector_paged_pq import (  # noqa: PLC0415
        joint_vprefix_outputs_from_grouped_risk_batched,
        joint_vprefix_outputs_from_grouped_risk_batched_vpq,
    )

    torch.manual_seed(20260526)
    device = torch.device("cuda")
    groups = 2
    k_count = 3
    heads = 2
    context_len = 29
    dim = 16
    pages = 3
    page_size = 8
    codes_n = 16
    v_steps = 4
    page_starts = torch.tensor([2, 10, 18], device=device, dtype=torch.long)
    base_outputs = torch.randn((groups, k_count, heads, dim), device=device, dtype=torch.float32)
    probs = torch.softmax(
        torch.randn((groups, k_count, heads, context_len), device=device, dtype=torch.float32),
        dim=-1,
    ).contiguous()
    values = torch.randn((groups, context_len, dim), device=device, dtype=torch.float32)
    value_codebooks = torch.randn((groups, pages, 1, codes_n, dim), device=device, dtype=torch.float32)
    value_codes = torch.randint(
        0,
        codes_n,
        (groups, pages, page_size, 1),
        device=device,
        dtype=torch.uint8,
    )
    residual_groups = torch.zeros((groups, context_len, dim), device=device, dtype=torch.float32)
    for group in range(groups):
        for page in range(pages):
            start = int(page_starts[page].item())
            for row in range(page_size):
                token = start + row
                if token >= context_len:
                    continue
                code = int(value_codes[group, page, row, 0].item())
                residual_groups[group, token] = values[group, token] - value_codebooks[group, page, 0, code]
    code_error_groups = torch.rand((groups, context_len), device=device, dtype=torch.float32)
    v_budgets = torch.tensor([0, 3, 11, 23], device=device, dtype=torch.long)
    ref = joint_vprefix_outputs_from_grouped_risk_batched(
        base_outputs.contiguous(),
        probs,
        residual_groups.contiguous(),
        code_error_groups.contiguous(),
        v_budgets,
    )
    got = joint_vprefix_outputs_from_grouped_risk_batched_vpq(
        base_outputs.contiguous(),
        probs,
        values.contiguous(),
        value_codebooks.contiguous(),
        value_codes.contiguous(),
        page_starts,
        code_error_groups.contiguous(),
        v_budgets,
        int(page_size),
    )
    if not torch.allclose(got, ref, atol=2e-5, rtol=2e-5):
        max_diff = float((got - ref).abs().max().item())
        raise AssertionError(f"compact V-PQ grouped risk-prefix mismatch: max_diff={max_diff}")


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    from selector_paged_pq import joint_vpq_sidecars_from_pack  # noqa: PLC0415

    _test_dense_decode_ranked_logits_simulator()
    _test_grouped_risk_prefix_workspace()
    _test_grouped_risk_prefix_compact_vpq()
    rng = np.random.default_rng(20260514)
    page_size = 32
    pages = 3
    dim = 16
    total_values = page_size * pages + 9
    keys_np = rng.normal(size=(total_values, dim)).astype(np.float32)
    values_np = rng.normal(size=(total_values, dim)).astype(np.float32)
    index = build_page_pq_gpu(
        keys_np,
        dynamic_start=0,
        indexed_end=page_size * pages,
        page_size=page_size,
        subvecs=4,
        subbits=4,
        kmeans_iters=2,
        seed=123,
        key_bytes=2,
        router_enabled=False,
        router_prototypes=1,
        router_merge_rel=0.0,
        router_merge_var=0.0,
        router_max_groups=0,
        device=torch.device("cuda"),
    )
    tokens_np = np.asarray(
        [
            [0, 7, 31, 32, 63, 64, 80, 95],
            [2, 5, 40, 71, 90, 96, 97, 100],
        ],
        dtype=np.int64,
    )
    values = torch.as_tensor(values_np, dtype=torch.float32, device="cuda")
    tokens = torch.as_tensor(tokens_np, dtype=torch.long, device="cuda")
    got, valid, _page_ids, _actual_bits = vpq_values_for_tokens_gpu(
        index=index,
        values=values,
        values_np=values_np,
        tokens=tokens,
        subbits=4,
        value_subvecs=2,
        value_subbits=3,
    )
    ref, _compressed_mb, _fallback_mb = _vpq_values_for_tokens(
        index=index,
        values_np=values_np,
        tokens=tokens_np.reshape(-1),
        subbits=4,
        value_subvecs=2,
        value_subbits=3,
        value_bytes=2,
    )
    got_np = np.asarray(got.detach().cpu().numpy().reshape(-1, dim), dtype=np.float32)
    ref_np = np.asarray(ref, dtype=np.float32)
    max_diff = float(np.max(np.abs(got_np - ref_np))) if got_np.size else 0.0
    if max_diff > 1e-5:
        raise AssertionError(f"V-PQ GPU reconstruction mismatch: max_diff={max_diff}")
    expected_valid = tokens_np < page_size * pages
    if valid.detach().cpu().tolist() != expected_valid.tolist():
        raise AssertionError("V-PQ valid mask mismatch")
    all_got = reconstruct_all_vpq_values_gpu(
        index=index,
        values_np=values_np,
        subbits=4,
        value_subvecs=2,
        value_subbits=3,
        device=torch.device("cuda"),
    )
    if all_got is None:
        raise AssertionError("all-token V-PQ reconstruction unexpectedly unavailable")
    all_values, actual_bits = all_got
    if int(actual_bits) != 3:
        raise AssertionError(f"unexpected actual value subbits: {actual_bits}")
    all_tokens_np = np.arange(page_size * pages, dtype=np.int64)
    all_ref, _all_compressed_mb, _all_fallback_mb = _vpq_values_for_tokens(
        index=index,
        values_np=values_np,
        tokens=all_tokens_np,
        subbits=4,
        value_subvecs=2,
        value_subbits=3,
        value_bytes=2,
    )
    all_values_np = np.asarray(all_values.detach().cpu().numpy(), dtype=np.float32)
    all_ref_np = np.asarray(all_ref, dtype=np.float32)
    all_max_diff = float(np.max(np.abs(all_values_np - all_ref_np))) if all_values_np.size else 0.0
    if all_max_diff > 1e-5:
        raise AssertionError(f"all-token V-PQ reconstruction mismatch: max_diff={all_max_diff}")
    pack = value_vpq_pack_torch(
        index=index,
        values=values,
        value_subvecs=2,
        value_subbits=3,
        key_bytes=2,
        device=torch.device("cuda"),
    )
    if pack is None:
        raise AssertionError("torch V-PQ pack unexpectedly unavailable")
    codebooks, codes, page_starts, _page_size, _actual_value_subbits = pack
    native_vhat, native_residual, native_code_error = joint_vpq_sidecars_from_pack(
        values,
        codebooks,
        codes,
        page_starts,
        int(total_values),
    )
    all_tokens = torch.arange(int(total_values), dtype=torch.long, device="cuda")
    ref_vhat, ref_valid, ref_page_ids, _ref_bits = vpq_values_for_tokens_gpu(
        index=index,
        values=values,
        values_np=None,
        tokens=all_tokens,
        subbits=4,
        value_subvecs=2,
        value_subbits=3,
        prefer_torch=True,
        value_bytes=2,
    )
    ref_residual = values.float() - ref_vhat.float()
    ref_code_error, _ = value_vpq_code_stat_risk_torch(
        index=index,
        values=values,
        vhat_all=ref_vhat,
        residual_all=ref_residual,
        valid=ref_valid,
        page_ids=ref_page_ids,
        subbits=4,
        value_subvecs=2,
        value_subbits=3,
        value_bytes=2,
    )
    if not torch.allclose(native_vhat, ref_vhat.float(), atol=1e-5, rtol=1e-5):
        max_diff = float((native_vhat - ref_vhat.float()).abs().max().item())
        raise AssertionError(f"native V-PQ sidecar vhat mismatch: max_diff={max_diff}")
    if not torch.allclose(native_residual, ref_residual.float(), atol=1e-5, rtol=1e-5):
        max_diff = float((native_residual - ref_residual.float()).abs().max().item())
        raise AssertionError(f"native V-PQ sidecar residual mismatch: max_diff={max_diff}")
    if not torch.allclose(native_code_error, ref_code_error, atol=1e-6, rtol=1e-6):
        max_diff = float((native_code_error - ref_code_error).abs().max().item())
        raise AssertionError(f"native V-PQ sidecar code-error mismatch: max_diff={max_diff}")
    _test_native_exact_value_counts()
    print("GPU V-PQ helper matches CPU/reference reconstruction")


if __name__ == "__main__":
    main()
