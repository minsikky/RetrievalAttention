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
from benchmark.selector_eval.runners.run_hf_paged_pq_intervention_eval import (  # noqa: E402
    _gpu_gqa_base_logsumexp_decode,
    _gpu_gqa_dense_decode_ranked_logits_and_base_lse,
    _gpu_gqa_ranked_exact_logits,
    geometric_budget_pairs,
    reconstruct_all_vpq_values_gpu,
    select_thresholds_for_budget_counts_gpu,
    selected_mass_thresholds_from_logits_gpu,
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
        gqa_decode_ranked_exact_logits,
        gqa_decode_ranked_exact_logits_with_base_lse,
    )

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


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    _test_dense_decode_ranked_logits_simulator()
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
    _test_native_exact_value_counts()
    print("GPU V-PQ helper matches CPU/reference reconstruction")


if __name__ == "__main__":
    main()
