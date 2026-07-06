#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from typing import Any

from benchmark.selector_eval.runners.hf_paged_pq_intervention_stats import ApproxStats


def build_hf_paged_pq_intervention_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HF decode/logit/task-style eval for routed paged-PQ + stratified tail intervention.")
    parser.add_argument("--model_name", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--cache_dir", default=".hf_cache")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--layers", default="16")
    parser.add_argument("--filler_repeats", type=int, default=128)
    parser.add_argument("--target", default="ZEBRA-4729")
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument(
        "--approx_prefill",
        action="store_true",
        help="apply the paged-PQ approximation during batched prefill as well as token-by-token decode",
    )
    parser.add_argument(
        "--skip_prefill_index_build",
        action="store_true",
        help="with dense prefill, defer decode sidecar construction to first decode token instead of building after prefill",
    )
    parser.add_argument("--selector_mode", choices=["fullscan", "routed", "oracle"], default="fullscan")
    parser.add_argument("--budget", type=int, default=4096)
    parser.add_argument("--budget_by_head", default="")
    parser.add_argument("--rerank_candidates", type=int, default=0)
    parser.add_argument("--tail_samples", type=int, default=16384)
    parser.add_argument("--tail_bands", type=int, default=8)
    parser.add_argument("--tail_seed", type=int, default=0)
    parser.add_argument("--tail_sampling", choices=["random", "linspace", "systematic"], default="systematic")
    parser.add_argument("--tail_mode", choices=["sample", "pq_value", "vpq_value", "page_mean"], default="vpq_value")
    parser.add_argument(
        "--online_confidence_rule",
        choices=[
            "none",
            "geometric_probe_tail_switch",
            "geometric_tail_stability_switch",
            "geometric_exact_delta",
            "joint_kv_stability",
            "pq_proxy_mass_budget",
            "pq_ranked_mass_budget",
        ],
        default="joint_kv_stability",
        help="deployable online budget/confidence rule; non-none disables fixed-budget native fast paths",
    )
    parser.add_argument("--tail_score_calibration", choices=["none", "affine_selected"], default="none")
    parser.add_argument("--tail_probe_rel_l2_max", type=float, default=float("inf"))
    parser.add_argument("--tail_proxy_mass_min", type=float, default=0.0)
    parser.add_argument("--tail_proxy_mass_max", type=float, default=1.0)
    parser.add_argument("--tail_pq_corr_min", type=float, default=-1.0)
    parser.add_argument("--tail_pq_relrmse_max", type=float, default=float("inf"))
    parser.add_argument(
        "--ranked_confidence_cost_mode",
        choices=["exact", "upper_bound"],
        default="exact",
        help=(
            "Cost accounting for adaptive ranked/geometric confidence. exact reports accepted budgets; "
            "upper_bound avoids runtime syncs and reports conservative max-budget cost."
        ),
    )
    parser.add_argument(
        "--exact_logit_backend",
        choices=["auto", "ranked_gather", "dense_sim"],
        default=os.environ.get("FRONTIER_EXACT_LOGIT_BACKEND", "auto"),
        help=(
            "GPU simulator backend for exact ranked logits used by confidence/selected-mass checks. "
            "dense_sim computes dense QK and gathers ranked logits for speed, while logical MB still "
            "uses the frontier/custom-hardware cost model."
        ),
    )
    parser.add_argument("--geometric_min_budget", type=int, default=8192)
    parser.add_argument("--geometric_max_budget", type=int, default=65536)
    parser.add_argument("--geometric_growth", type=float, default=1.5)
    parser.add_argument("--geometric_probe_scale", type=float, default=1.5)
    parser.add_argument("--geometric_budget_granularity", type=int, default=1024)
    parser.add_argument(
        "--joint_kv_policy",
        choices=[
            "k_first_priority",
            "v_first_priority",
            "k_first_alternating",
            "v_first_alternating",
            "sensitivity_greedy",
        ],
        default="k_first_alternating",
    )
    parser.add_argument("--joint_kv_k_budgets", default="4096,8192,14336,32768")
    parser.add_argument("--joint_kv_v_budgets", default="1024,2048,4096,6144,8192,12288,16384")
    parser.add_argument(
        "--joint_kv_k_budget_fracs",
        default="0.10,0.30,0.50,0.70,0.90,1.0",
        help="optional comma-separated K budget fractions, e.g. 10%,30%,50%; overrides joint_kv_k_budgets",
    )
    parser.add_argument(
        "--joint_kv_v_budget_fracs",
        default="0.05,0.10,0.20,0.40,0.60,0.80,1.0",
        help="optional comma-separated V budget fractions; overrides joint_kv_v_budgets",
    )
    parser.add_argument("--joint_kv_stability_threshold", type=float, default=0.002)
    parser.add_argument("--joint_kv_threshold_mode", choices=["fixed", "budget_delta_frac"], default="budget_delta_frac")
    parser.add_argument("--joint_kv_threshold_reference_frac", type=float, default=0.2)
    parser.add_argument("--joint_kv_threshold_scale_shape", choices=["linear", "sqrt", "log"], default="sqrt")
    parser.add_argument("--joint_kv_threshold_min_scale", type=float, default=0.0)
    parser.add_argument("--joint_kv_threshold_max_scale", type=float, default=1.5)
    parser.add_argument("--joint_kv_start_strategy", default="proxy_mass_m0p9")
    parser.add_argument("--selected_value_mode", choices=["exact", "vpq_value"], default="vpq_value")
    parser.add_argument(
        "--selected_value_exact_rule",
        choices=[
            "fixed",
            "selector_rank",
            "selected_mass",
            "selected_risk_mass",
            "selected_mass_or_risk",
            "global_residual_risk",
        ],
        default="global_residual_risk",
    )
    parser.add_argument("--selected_value_exact_top", type=int, default=0)
    parser.add_argument("--selected_value_exact_mass", type=float, default=0.0)
    parser.add_argument("--selected_value_exact_risk_mass", type=float, default=0.0)
    parser.add_argument("--selected_value_min_exact_top", type=int, default=0)
    parser.add_argument("--selected_value_max_exact_top", type=int, default=0)
    parser.add_argument("--selected_value_exact_all_context_max", type=int, default=0)
    parser.add_argument("--selected_value_exact_all_fraction_min", type=float, default=0.0)
    parser.add_argument("--selected_value_residual_norm_bytes", type=int, default=2)
    parser.add_argument("--value_code_stat_bytes", type=int, default=2)
    parser.add_argument("--value_subvecs", type=int, default=0)
    parser.add_argument("--value_subbits", type=int, default=0)
    parser.add_argument("--value_pq_group_pages", type=int, default=1)
    parser.add_argument("--tail_blend", type=float, default=1.0)
    parser.add_argument("--prefill_tail_blend", type=float, default=None)
    parser.add_argument("--decode_tail_blend", type=float, default=None)
    parser.add_argument("--tail_off_heads", default="")
    parser.add_argument("--static_prefix", type=int, default=128)
    parser.add_argument("--static_suffix", type=int, default=128)
    parser.add_argument("--page_size", type=int, default=5632)
    parser.add_argument(
        "--prefill_chunk_size",
        type=int,
        default=0,
        help="optional exact-selected native prefill chunk size; 0 keeps the one-shot prefill path",
    )
    parser.add_argument(
        "--prefill_selector_backend",
        choices=[
            "native",
            "native_fused",
            "native_page_max",
            "torch_lut",
            "torch_lut_fp16",
            "torch_lut_streaming",
            "torch_lut_batched",
            "torch_matmul",
        ],
        default="native",
        help="selector implementation for exact-selected batched prefill; torch_lut uses PQ LUT scoring",
    )
    parser.add_argument(
        "--prefill_selector_stride",
        type=int,
        default=1,
        help="reuse exact-prefill selector results within blocks of this many query positions",
    )
    parser.add_argument(
        "--prefill_selector_tile_size",
        type=int,
        default=0,
        help="tile query positions for torch_lut_batched prefill selector to bound temporary score/gather memory",
    )
    parser.add_argument(
        "--prefill_rank_buffer_limit_mb",
        type=float,
        default=4096.0,
        help=(
            "Maximum estimated [query, head, rank] output buffer for the batched prefill fast path. "
            "High-budget confidence runs above this limit fall back instead of OOMing."
        ),
    )
    parser.add_argument(
        "--prefill_selector_page_block_size",
        type=int,
        default=0,
        help="process this many PQ pages at a time for torch_lut_batched prefill selector and maintain a running top-k",
    )
    parser.add_argument(
        "--prefill_tail_score_reuse",
        action="store_true",
        help="for V-PQ tail prefill, materialize causal PQ selector scores once and reuse them in the tail attention kernel",
    )
    parser.add_argument(
        "--prefill_attention_backend",
        choices=["native", "flashinfer_blocksparse", "flashinfer_page_blocks"],
        default="native",
        help="attention implementation for exact-selected batched prefill; flashinfer uses page-block sparse attention",
    )
    parser.add_argument("--subvecs", type=int, default=4)
    parser.add_argument("--subbits", type=int, default=6)
    parser.add_argument("--kmeans_iters", type=int, default=3)
    parser.add_argument(
        "--index_build_backend",
        choices=["numpy", "torch_gpu"],
        default=os.environ.get("PAGEDPQ_INDEX_BUILD_BACKEND", "torch_gpu"),
        help="page-PQ index construction backend; torch_gpu keeps fullscan PQ state construction on device",
    )
    parser.add_argument("--nprobes", default="16,32,64,128,256,512")
    parser.add_argument("--router_prototypes", type=int, default=16)
    parser.add_argument("--router_merge_rel", type=float, default=0.05)
    parser.add_argument("--router_merge_var", type=float, default=0.0)
    parser.add_argument("--router_max_groups", type=int, default=512)
    parser.add_argument("--key_bytes", type=int, default=2)
    parser.add_argument("--value_bytes", type=int, default=2)
    parser.add_argument(
        "--selector_backend",
        choices=["torch", "cuda_ext", "auto"],
        default=os.environ.get("SELECTOR_PAGED_PQ_BACKEND", "cuda_ext"),
        help="fullscan selector backend; cuda_ext requires benchmark/selector_eval/cuda_ext to be built",
    )
    parser.add_argument(
        "--native_decode_tail",
        action="store_true",
        help="experimental: use native compressed-tail attention for decode; currently slower than the torch tail path",
    )
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--cpu_then_to_device", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--profile_native_ops", action="store_true")
    parser.add_argument("--disable_cost_stats", action="store_true")
    parser.add_argument("--disable_native_decode_fused", dest="disable_native_decode_fused", action="store_true", default=True)
    parser.add_argument("--enable_native_decode_fused", dest="disable_native_decode_fused", action="store_false")
    parser.add_argument("--native_decode_scoreless_fused", action="store_true")
    parser.add_argument("--native_decode_scoreless_force_mode", type=int, default=2)
    parser.add_argument(
        "--debug_empty_cache_native",
        action="store_true",
        help="debug-only: call torch.cuda.empty_cache() when native decode pack caches change",
    )
    return parser


def approx_stats_payload(approx_stats: dict[int, ApproxStats]) -> dict[str, dict[str, Any]]:
    for s in approx_stats.values():
        s.flush_device_count_sums()
    stats_payload = {}
    for layer, s in sorted(approx_stats.items()):
        update_mb = float(s.index_build_read_mb + s.index_build_write_mb)
        update_mb_per_head_query = update_mb / max(1, int(s.calls))
        stats_payload[str(layer)] = {
            "calls": s.calls,
            "approx_attention_calls": s.approx_attention_calls,
            "passthrough_attention_calls": s.passthrough_attention_calls,
            "mean_selected_tokens": s.mean_selected,
            "mean_tail_samples": s.mean_tail_samples,
            "mean_selector_MB_per_head_query": s.mean_selector_mb,
            "mean_logical_frontier_selector_MB_per_head_query": s.mean_selector_mb,
            "mean_exact_KV_MB_per_head_query": s.mean_exact_kv_mb,
            "mean_logical_frontier_exact_KV_MB_per_head_query": s.mean_exact_kv_mb,
            "mean_tail_estimator_MB_per_head_query": s.mean_tail_mb,
            "mean_logical_frontier_tail_estimator_MB_per_head_query": s.mean_tail_mb,
            "mean_confidence_MB_per_head_query": s.mean_confidence_mb,
            "mean_logical_frontier_confidence_MB_per_head_query": s.mean_confidence_mb,
            "mean_step_MB_per_head_query": s.mean_step_mb,
            "mean_logical_frontier_step_MB_per_head_query": s.mean_step_mb,
            "mean_physical_gpu_exact_KV_MB_per_head_query": s.mean_physical_gpu_exact_kv_mb,
            "mean_physical_gpu_confidence_MB_per_head_query": s.mean_physical_gpu_confidence_mb,
            "mean_physical_gpu_step_MB_per_head_query": s.mean_physical_gpu_step_mb,
            "selector_active_fraction": float(s.selector_active_calls) / max(1, int(s.calls)),
            "tail_active_fraction": float(s.tail_active_calls) / max(1, int(s.calls)),
            "confidence_active_fraction": float(s.confidence_active_calls) / max(1, int(s.calls)),
            "mean_update_MB_per_head_query": update_mb_per_head_query,
            "mean_total_MB_per_head_query": s.mean_step_mb + update_mb_per_head_query,
            "mean_logical_frontier_total_MB_per_head_query": s.mean_step_mb + update_mb_per_head_query,
            "mean_physical_gpu_total_MB_per_head_query": s.mean_physical_gpu_step_mb + update_mb_per_head_query,
            "index_build_calls": s.index_build_calls,
            "index_build_seconds": s.index_build_seconds,
            "index_build_read_MB": s.index_build_read_mb,
            "index_build_write_MB": s.index_build_write_mb,
            "index_build_total_MB": update_mb,
            "online_update_MB_per_attention_call": update_mb / max(1, int(s.approx_attention_calls)),
            "cache_cast_seconds": s.cache_cast_seconds,
            "patched_attention_seconds": s.patched_attention_seconds,
            "qkv_cache_seconds": s.qkv_cache_seconds,
            "index_sidecar_seconds": s.index_sidecar_seconds,
            "native_pack_seconds": s.native_pack_seconds,
            "native_selector_seconds": s.native_selector_seconds,
            "native_attention_seconds": s.native_attention_seconds,
            "native_exact_logit_seconds": s.native_exact_logit_seconds,
            "native_threshold_seconds": s.native_threshold_seconds,
            "native_geometric_seconds": s.native_geometric_seconds,
            "native_output_seconds": s.native_output_seconds,
            "native_joint_rank_prefix_seconds": s.native_joint_rank_prefix_seconds,
            "native_joint_score_grid_seconds": s.native_joint_score_grid_seconds,
            "native_joint_prob_base_seconds": s.native_joint_prob_base_seconds,
            "native_joint_risk_prefix_seconds": s.native_joint_risk_prefix_seconds,
            "native_joint_policy_seconds": s.native_joint_policy_seconds,
            "native_joint_precompute_seconds": s.native_joint_precompute_seconds,
            "native_joint_layout_seconds": s.native_joint_layout_seconds,
            "native_joint_group_pack_seconds": s.native_joint_group_pack_seconds,
            "native_joint_accounting_seconds": s.native_joint_accounting_seconds,
            "joint_staged_kv_groups": s.joint_staged_kv_groups,
            "joint_staged_kv_accepted_groups": s.joint_staged_kv_accepted_groups,
            "joint_staged_kv_boundary_groups": s.joint_staged_kv_boundary_groups,
            "joint_staged_kv_accept_fraction": (
                float(s.joint_staged_kv_accepted_groups) / max(1, int(s.joint_staged_kv_groups))
            ),
            "native_vpq_append_seconds": s.native_vpq_append_seconds,
            "native_vpq_append_calls": s.native_vpq_append_calls,
            "native_vpq_append_grouped_calls": s.native_vpq_append_grouped_calls,
            "native_vpq_append_fallback_calls": s.native_vpq_append_fallback_calls,
            "wall_patched_attention_seconds": s.wall_patched_attention_seconds,
            "wall_qkv_cache_seconds": s.wall_qkv_cache_seconds,
            "wall_index_sidecar_seconds": s.wall_index_sidecar_seconds,
            "wall_output_projection_seconds": s.wall_output_projection_seconds,
            "wall_joint_total_seconds": s.wall_joint_total_seconds,
            "wall_joint_precompute_seconds": s.wall_joint_precompute_seconds,
            "wall_joint_selector_seconds": s.wall_joint_selector_seconds,
            "wall_joint_exact_logit_seconds": s.wall_joint_exact_logit_seconds,
            "wall_joint_vpq_sidecar_seconds": s.wall_joint_vpq_sidecar_seconds,
            "wall_joint_layout_seconds": s.wall_joint_layout_seconds,
            "wall_joint_rank_prefix_seconds": s.wall_joint_rank_prefix_seconds,
            "wall_joint_score_grid_seconds": s.wall_joint_score_grid_seconds,
            "wall_joint_prob_base_seconds": s.wall_joint_prob_base_seconds,
            "wall_joint_risk_prefix_seconds": s.wall_joint_risk_prefix_seconds,
            "wall_joint_policy_seconds": s.wall_joint_policy_seconds,
            "wall_joint_group_pack_seconds": s.wall_joint_group_pack_seconds,
            "wall_joint_accounting_seconds": s.wall_joint_accounting_seconds,
            "output_projection_seconds": s.output_projection_seconds,
        }
    return stats_payload
