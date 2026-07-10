#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaForCausalLM

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.ruler.pred.dense_kv_offload import (
    DenseKVOffloadCache,
    patched_qwen2_dense_kv_offload,
)
from benchmark.ruler.pred.utils import load_data
from benchmark.selector_eval.gpu.run_gpu_paged_pq_eval import parse_csv_ints
from benchmark.selector_eval.runners.hf_attn_output_noise import (
    maybe_attn_output_noise_patch,
)
from benchmark.selector_eval.runners.hf_paged_pq_intervention_api import (
    ApproxStats,
    patched_paged_pq_attention,
    reset_paged_pq_attention_state,
)


def log(msg: str) -> None:
    print(f"[pagedpq_stream_ruler] {time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


def env_truthy(name: str, default: str = "0") -> bool:
    return str(os.environ.get(name, default)).lower() not in {"0", "false", "no", "off", ""}


def log_cuda_memory(label: str, device: torch.device, *, reset_peak: bool = False) -> None:
    if not env_truthy("SELECTOR_PQ_JOINT_MEMORY_TRACE", "0"):
        return
    if device.type != "cuda" or not torch.cuda.is_available():
        return
    if reset_peak:
        torch.cuda.reset_peak_memory_stats(device)
    allocated = float(torch.cuda.memory_allocated(device)) / (1024.0 * 1024.0)
    reserved = float(torch.cuda.memory_reserved(device)) / (1024.0 * 1024.0)
    peak = float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0)
    peak_reserved = float(torch.cuda.max_memory_reserved(device)) / (1024.0 * 1024.0)
    free, total = torch.cuda.mem_get_info(device)
    print(
        f"[pagedpq-memory] {label}: allocated={allocated:.1f}MiB "
        f"reserved={reserved:.1f}MiB peak={peak:.1f}MiB "
        f"peak_reserved={peak_reserved:.1f}MiB "
        f"free={float(free) / (1024.0 * 1024.0):.1f}MiB "
        f"total={float(total) / (1024.0 * 1024.0):.1f}MiB",
        flush=True,
    )


def joint_cuda_flags_config() -> dict[str, bool | int]:
    return {
        "selector_pq_joint_gqa_batched": env_truthy("SELECTOR_PQ_JOINT_GQA_BATCHED", "0"),
        "selector_pq_joint_vector_policy": env_truthy("SELECTOR_PQ_JOINT_VECTOR_POLICY", "0"),
        "selector_pq_joint_reuse_max_topk": env_truthy("SELECTOR_PQ_JOINT_REUSE_MAX_TOPK", "0"),
        "selector_pq_joint_grid_artifacts": env_truthy("SELECTOR_PQ_JOINT_GRID_ARTIFACTS", "0"),
        "selector_pq_joint_allhead_precompute": env_truthy("SELECTOR_PQ_JOINT_ALLHEAD_PRECOMPUTE", "0"),
        "selector_pq_joint_grouped_risk_prefix": env_truthy("SELECTOR_PQ_JOINT_GROUPED_RISK_PREFIX", "0"),
        "selector_pq_joint_native_v_prefix": env_truthy("SELECTOR_PQ_JOINT_NATIVE_V_PREFIX", "0"),
        "selector_pq_joint_native_risk_prefix": env_truthy("SELECTOR_PQ_JOINT_NATIVE_RISK_PREFIX", "0"),
        "selector_pq_joint_native_score_grid": env_truthy("SELECTOR_PQ_JOINT_NATIVE_SCORE_GRID", "0"),
        "selector_pq_joint_native_policy": env_truthy("SELECTOR_PQ_JOINT_NATIVE_POLICY", "0"),
        "selector_pq_joint_prewarm_vpq_sidecars": env_truthy("SELECTOR_PQ_JOINT_PREWARM_VPQ_SIDECARS", "0"),
        "selector_pq_joint_persistent_vpq_cache": env_truthy("SELECTOR_PQ_JOINT_PERSISTENT_VPQ_CACHE", "0"),
        "selector_pq_joint_persistent_vpq_cache_grow_pad": int(
            os.environ.get("SELECTOR_PQ_JOINT_PERSISTENT_VPQ_CACHE_GROW_PAD", "256")
        ),
        "selector_pq_joint_exact_full_budget_grid": env_truthy("SELECTOR_PQ_JOINT_EXACT_FULL_BUDGET_GRID", "1"),
        "selector_pq_joint_segmented_v_prefix": env_truthy("SELECTOR_PQ_JOINT_SEGMENTED_V_PREFIX", "0"),
        "selector_pq_joint_unsorted_v_prefix": env_truthy("SELECTOR_PQ_JOINT_UNSORTED_V_PREFIX", "0"),
        "selector_pq_joint_fast_affine_selected": env_truthy("SELECTOR_PQ_JOINT_FAST_AFFINE_SELECTED", "0"),
        "selector_pq_joint_ondemand_v_prefix": env_truthy("SELECTOR_PQ_JOINT_ONDEMAND_V_PREFIX", "0"),
        "selector_pq_joint_incremental_v_grid": env_truthy("SELECTOR_PQ_JOINT_INCREMENTAL_V_GRID", "0"),
        "selector_pq_joint_incremental_vpq_sidecar": env_truthy(
            "SELECTOR_PQ_JOINT_INCREMENTAL_VPQ_SIDECAR",
            "0",
        ),
        "selector_pq_joint_native_lazy_policy": env_truthy("SELECTOR_PQ_JOINT_NATIVE_LAZY_POLICY", "0"),
        "selector_pq_joint_allhead_exact_precompute": env_truthy("SELECTOR_PQ_JOINT_ALLHEAD_EXACT_PRECOMPUTE", "0"),
        "selector_pq_joint_native_exact_logits": env_truthy("SELECTOR_PQ_JOINT_NATIVE_EXACT_LOGITS", "0"),
        "selector_pq_joint_sparse_exact_score_grid": env_truthy(
            "SELECTOR_PQ_JOINT_SPARSE_EXACT_SCORE_GRID",
            "0",
        ),
        "selector_pq_joint_sparse_direct_score_grid": env_truthy(
            "SELECTOR_PQ_JOINT_SPARSE_DIRECT_SCORE_GRID",
            "0",
        ),
        "selector_pq_joint_native_exact_logits_backend": os.environ.get(
            "SELECTOR_PQ_JOINT_NATIVE_EXACT_LOGITS_BACKEND",
            "cublas_t",
        ),
        "selector_pq_native_exact_logits_tf32": env_truthy("SELECTOR_PQ_NATIVE_EXACT_LOGITS_TF32", "1"),
        "selector_pq_joint_allhead_rank_prefix": env_truthy("SELECTOR_PQ_JOINT_ALLHEAD_RANK_PREFIX", "0"),
        "selector_pq_joint_native_rank_prefix": env_truthy("SELECTOR_PQ_JOINT_NATIVE_RANK_PREFIX", "0"),
        "selector_pq_joint_native_budget_prefix": env_truthy("SELECTOR_PQ_JOINT_NATIVE_BUDGET_PREFIX", "0"),
        "selector_pq_joint_rank_prefix_workspace": env_truthy("SELECTOR_PQ_JOINT_RANK_PREFIX_WORKSPACE", "0"),
        "selector_pq_joint_selector_topk_prefix": env_truthy("SELECTOR_PQ_JOINT_SELECTOR_TOPK_PREFIX", "0"),
        "selector_pq_joint_unsorted_k_prefix": env_truthy("SELECTOR_PQ_JOINT_UNSORTED_K_PREFIX", "0"),
        "selector_pq_joint_skip_full_budget_sort": env_truthy("SELECTOR_PQ_JOINT_SKIP_FULL_BUDGET_SORT", "0"),
        "selector_pq_joint_collapse_dup_k_rows": env_truthy("SELECTOR_PQ_JOINT_COLLAPSE_DUP_K_ROWS", "0"),
        "selector_pq_joint_collapse_dup_v_rows": env_truthy("SELECTOR_PQ_JOINT_COLLAPSE_DUP_V_ROWS", "0"),
        "selector_pq_joint_score_grid_no_exact_fill": env_truthy("SELECTOR_PQ_JOINT_SCORE_GRID_NO_EXACT_FILL", "0"),
        "selector_pq_joint_score_grid_workspace": env_truthy("SELECTOR_PQ_JOINT_SCORE_GRID_WORKSPACE", "0"),
        "selector_pq_joint_grouped_score_workspace": env_truthy("SELECTOR_PQ_JOINT_GROUPED_SCORE_WORKSPACE", "0"),
        "selector_pq_joint_nocalib_scatter_score_grid": env_truthy(
            "SELECTOR_PQ_JOINT_NOCALIB_SCATTER_SCORE_GRID",
            "0",
        ),
        "selector_pq_joint_rankpos_score_grid": env_truthy("SELECTOR_PQ_JOINT_RANKPOS_SCORE_GRID", "0"),
        "selector_pq_joint_grouped_vpq_cache": env_truthy("SELECTOR_PQ_JOINT_GROUPED_VPQ_CACHE", "0"),
        "selector_pq_joint_score_direct_vprefix": env_truthy("SELECTOR_PQ_JOINT_SCORE_DIRECT_VPREFIX", "0"),
        "selector_pq_joint_score_direct_interval_policy": env_truthy(
            "SELECTOR_PQ_JOINT_SCORE_DIRECT_INTERVAL_POLICY",
            "0",
        ),
        "selector_pq_joint_score_prob_interval_policy": env_truthy(
            "SELECTOR_PQ_JOINT_SCORE_PROB_INTERVAL_POLICY",
            "0",
        ),
        "selector_pq_joint_score_direct_topk_interval_policy": env_truthy(
            "SELECTOR_PQ_JOINT_SCORE_DIRECT_TOPK_INTERVAL_POLICY",
            "0",
        ),
        "selector_pq_joint_score_direct_workspace": env_truthy("SELECTOR_PQ_JOINT_SCORE_DIRECT_WORKSPACE", "0"),
        "selector_pq_joint_grouped_output_workspace": env_truthy("SELECTOR_PQ_JOINT_GROUPED_OUTPUT_WORKSPACE", "0"),
        "selector_pq_joint_fused_risk_policy": env_truthy("SELECTOR_PQ_JOINT_FUSED_RISK_POLICY", "0"),
        "selector_pq_joint_interval_risk_policy": env_truthy("SELECTOR_PQ_JOINT_INTERVAL_RISK_POLICY", "0"),
        "selector_pq_joint_risk_prefix_topk": env_truthy("SELECTOR_PQ_JOINT_RISK_PREFIX_TOPK", "0"),
        "selector_pq_joint_risk_prefix_workspace": env_truthy("SELECTOR_PQ_JOINT_RISK_PREFIX_WORKSPACE", "0"),
        "selector_pq_joint_fast_token_layout": env_truthy("SELECTOR_PQ_JOINT_FAST_TOKEN_LAYOUT", "0"),
        "selector_pq_joint_compact_vpq_risk_prefix": env_truthy(
            "SELECTOR_PQ_JOINT_COMPACT_VPQ_RISK_PREFIX",
            "0",
        ),
        "selector_pq_joint_native_vpq_base": env_truthy("SELECTOR_PQ_JOINT_NATIVE_VPQ_BASE", "0"),
        "selector_pq_joint_native_vpq_append": env_truthy("SELECTOR_PQ_JOINT_NATIVE_VPQ_APPEND", "0"),
        "selector_pq_joint_native_vpq_sidecar": env_truthy("SELECTOR_PQ_JOINT_NATIVE_VPQ_SIDECAR", "0"),
        "selector_pq_joint_native_softmax_base": env_truthy("SELECTOR_PQ_JOINT_NATIVE_SOFTMAX_BASE", "0"),
        "selector_pq_joint_grouped_softmax_base": env_truthy("SELECTOR_PQ_JOINT_GROUPED_SOFTMAX_BASE", "0"),
        "selector_pq_joint_native_pq_scale_in_kernel": env_truthy(
            "SELECTOR_PQ_JOINT_NATIVE_PQ_SCALE_IN_KERNEL",
            "0",
        ),
        "selector_pq_joint_native_accounting": env_truthy("SELECTOR_PQ_JOINT_NATIVE_ACCOUNTING", "0"),
        "selector_pq_joint_fused_policy_accounting": env_truthy(
            "SELECTOR_PQ_JOINT_FUSED_POLICY_ACCOUNTING",
            "0",
        ),
        "selector_pq_joint_native_accounting_verify": env_truthy(
            "SELECTOR_PQ_JOINT_NATIVE_ACCOUNTING_VERIFY",
            "0",
        ),
        "selector_pq_joint_tokenfit_score_grid": env_truthy("SELECTOR_PQ_JOINT_TOKENFIT_SCORE_GRID", "0"),
        "selector_pq_joint_fused_softmax_base": env_truthy("SELECTOR_PQ_JOINT_FUSED_SOFTMAX_BASE", "0"),
        "selector_pq_joint_fused_tokenfit_softmax_base": env_truthy(
            "SELECTOR_PQ_JOINT_FUSED_TOKENFIT_SOFTMAX_BASE",
            "0",
        ),
        "selector_pq_joint_wall_profile": env_truthy("SELECTOR_PQ_JOINT_WALL_PROFILE", "0"),
    }


def task_tokens_to_generate(task: str) -> int:
    yaml_path = PROJECT_ROOT / "benchmark" / "ruler" / "synthetic.yaml"
    constants_path = PROJECT_ROOT / "benchmark" / "ruler" / "data" / "synthetic" / "constants.py"
    namespace: dict[str, object] = {}
    exec(constants_path.read_text(encoding="utf-8"), namespace)
    base = namespace["TASKS"]
    custom = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    cfg = dict(custom[task])
    cfg.update(base[cfg["task"]])
    return int(cfg["tokens_to_generate"])


def stats_payload(stats: dict[int, ApproxStats]) -> dict[str, dict[str, float | int]]:
    for s in stats.values():
        s.flush_device_count_sums()
    payload = {}
    for layer, s in sorted(stats.items()):
        update_mb = float(s.index_build_read_mb + s.index_build_write_mb)
        update_mb_per_head_query = update_mb / max(1, int(s.calls))
        update_mb_per_attention_call = update_mb / max(1, int(s.approx_attention_calls))
        approx_calls = int(s.approx_attention_calls)
        passthrough_calls = int(s.passthrough_attention_calls)
        approx_path_fraction = float(approx_calls / max(1, approx_calls + passthrough_calls))
        selector_active_calls = int(getattr(s, "selector_active_calls", 0))
        tail_active_calls = int(getattr(s, "tail_active_calls", 0))
        selector_active_fraction = (
            float(selector_active_calls) / max(1, int(s.calls))
            if selector_active_calls > 0
            else approx_path_fraction
        )
        tail_active_fraction = (
            float(tail_active_calls) / max(1, int(s.calls)) if tail_active_calls > 0 else approx_path_fraction
        )
        payload[str(layer)] = {
            "head_query_calls": int(s.calls),
            "approx_attention_calls": approx_calls,
            "passthrough_attention_calls": passthrough_calls,
            "mean_selected_tokens": float(s.mean_selected),
            "mean_tail_samples": float(s.mean_tail_samples),
            "mean_selector_MB_per_head_query": float(s.mean_selector_mb),
            "mean_logical_frontier_selector_MB_per_head_query": float(s.mean_selector_mb),
            "mean_exact_KV_MB_per_head_query": float(s.mean_exact_kv_mb),
            "mean_logical_frontier_exact_KV_MB_per_head_query": float(s.mean_exact_kv_mb),
            "mean_tail_estimator_MB_per_head_query": float(s.mean_tail_mb),
            "mean_logical_frontier_tail_estimator_MB_per_head_query": float(s.mean_tail_mb),
            "mean_confidence_MB_per_head_query": float(s.mean_confidence_mb),
            "mean_logical_frontier_confidence_MB_per_head_query": float(s.mean_confidence_mb),
            "mean_step_MB_per_head_query": float(s.mean_step_mb),
            "mean_logical_frontier_step_MB_per_head_query": float(s.mean_step_mb),
            "mean_physical_gpu_exact_KV_MB_per_head_query": float(s.mean_physical_gpu_exact_kv_mb),
            "mean_physical_gpu_confidence_MB_per_head_query": float(s.mean_physical_gpu_confidence_mb),
            "mean_physical_gpu_step_MB_per_head_query": float(s.mean_physical_gpu_step_mb),
            "approx_path_active_fraction": approx_path_fraction,
            "selector_active_fraction": selector_active_fraction,
            "tail_active_fraction": tail_active_fraction,
            "confidence_active_fraction": float(getattr(s, "confidence_active_calls", 0)) / max(1, int(s.calls)),
            "mean_update_MB_per_head_query": float(update_mb_per_head_query),
            "mean_total_MB_per_head_query": float(s.mean_step_mb + update_mb_per_head_query),
            "mean_logical_frontier_total_MB_per_head_query": float(s.mean_step_mb + update_mb_per_head_query),
            "mean_physical_gpu_total_MB_per_head_query": float(
                s.mean_physical_gpu_step_mb + update_mb_per_head_query
            ),
            "index_build_calls": int(s.index_build_calls),
            "index_build_seconds": float(s.index_build_seconds),
            "index_build_read_MB": float(s.index_build_read_mb),
            "index_build_write_MB": float(s.index_build_write_mb),
            "index_build_total_MB": float(update_mb),
            "online_update_MB_per_attention_call": float(update_mb_per_attention_call),
            "cache_cast_seconds": float(s.cache_cast_seconds),
            "patched_attention_seconds": float(s.patched_attention_seconds),
            "qkv_cache_seconds": float(s.qkv_cache_seconds),
            "index_sidecar_seconds": float(s.index_sidecar_seconds),
            "native_pack_seconds": float(s.native_pack_seconds),
            "native_selector_seconds": float(s.native_selector_seconds),
            "native_attention_seconds": float(s.native_attention_seconds),
            "native_exact_logit_seconds": float(getattr(s, "native_exact_logit_seconds", 0.0)),
            "native_threshold_seconds": float(getattr(s, "native_threshold_seconds", 0.0)),
            "native_geometric_seconds": float(getattr(s, "native_geometric_seconds", 0.0)),
            "native_output_seconds": float(getattr(s, "native_output_seconds", 0.0)),
            "native_joint_rank_prefix_seconds": float(getattr(s, "native_joint_rank_prefix_seconds", 0.0)),
            "native_joint_score_grid_seconds": float(getattr(s, "native_joint_score_grid_seconds", 0.0)),
            "native_joint_prob_base_seconds": float(getattr(s, "native_joint_prob_base_seconds", 0.0)),
            "native_joint_risk_prefix_seconds": float(getattr(s, "native_joint_risk_prefix_seconds", 0.0)),
            "native_joint_policy_seconds": float(getattr(s, "native_joint_policy_seconds", 0.0)),
            "native_joint_precompute_seconds": float(getattr(s, "native_joint_precompute_seconds", 0.0)),
            "native_joint_layout_seconds": float(getattr(s, "native_joint_layout_seconds", 0.0)),
            "native_joint_group_pack_seconds": float(getattr(s, "native_joint_group_pack_seconds", 0.0)),
            "native_joint_accounting_seconds": float(getattr(s, "native_joint_accounting_seconds", 0.0)),
            "joint_staged_kv_groups": int(getattr(s, "joint_staged_kv_groups", 0)),
            "joint_staged_kv_accepted_groups": int(getattr(s, "joint_staged_kv_accepted_groups", 0)),
            "joint_staged_kv_boundary_groups": int(getattr(s, "joint_staged_kv_boundary_groups", 0)),
            "joint_staged_kv_accept_fraction": float(
                int(getattr(s, "joint_staged_kv_accepted_groups", 0))
                / max(1, int(getattr(s, "joint_staged_kv_groups", 0)))
            ),
            "wall_patched_attention_seconds": float(getattr(s, "wall_patched_attention_seconds", 0.0)),
            "wall_qkv_cache_seconds": float(getattr(s, "wall_qkv_cache_seconds", 0.0)),
            "wall_index_sidecar_seconds": float(getattr(s, "wall_index_sidecar_seconds", 0.0)),
            "wall_output_projection_seconds": float(getattr(s, "wall_output_projection_seconds", 0.0)),
            "wall_joint_total_seconds": float(getattr(s, "wall_joint_total_seconds", 0.0)),
            "wall_joint_precompute_seconds": float(getattr(s, "wall_joint_precompute_seconds", 0.0)),
            "wall_joint_selector_seconds": float(getattr(s, "wall_joint_selector_seconds", 0.0)),
            "wall_joint_exact_logit_seconds": float(getattr(s, "wall_joint_exact_logit_seconds", 0.0)),
            "wall_joint_vpq_sidecar_seconds": float(getattr(s, "wall_joint_vpq_sidecar_seconds", 0.0)),
            "wall_joint_layout_seconds": float(getattr(s, "wall_joint_layout_seconds", 0.0)),
            "wall_joint_rank_prefix_seconds": float(getattr(s, "wall_joint_rank_prefix_seconds", 0.0)),
            "wall_joint_score_grid_seconds": float(getattr(s, "wall_joint_score_grid_seconds", 0.0)),
            "wall_joint_prob_base_seconds": float(getattr(s, "wall_joint_prob_base_seconds", 0.0)),
            "wall_joint_risk_prefix_seconds": float(getattr(s, "wall_joint_risk_prefix_seconds", 0.0)),
            "wall_joint_policy_seconds": float(getattr(s, "wall_joint_policy_seconds", 0.0)),
            "wall_joint_group_pack_seconds": float(getattr(s, "wall_joint_group_pack_seconds", 0.0)),
            "wall_joint_accounting_seconds": float(getattr(s, "wall_joint_accounting_seconds", 0.0)),
            "output_projection_seconds": float(s.output_projection_seconds),
        }
    return payload


def aggregate_stats(stats: dict[int, ApproxStats]) -> dict[str, float | int]:
    for s in stats.values():
        s.flush_device_count_sums()
    if not stats:
        return {}
    layers = list(stats.values())
    total_approx_calls = int(sum(s.approx_attention_calls for s in layers))
    total_passthrough_calls = int(sum(s.passthrough_attention_calls for s in layers))
    approx_path_fraction = float(total_approx_calls / max(1, total_approx_calls + total_passthrough_calls))
    selector_active_calls = int(sum(int(getattr(s, "selector_active_calls", 0)) for s in layers))
    tail_active_calls = int(sum(int(getattr(s, "tail_active_calls", 0)) for s in layers))
    update_mbs = [float(s.index_build_read_mb + s.index_build_write_mb) for s in layers]
    update_per_head_query = [mb / max(1, int(s.calls)) for mb, s in zip(update_mbs, layers, strict=True)]
    total_per_head_query = [float(s.mean_step_mb) + upd for s, upd in zip(layers, update_per_head_query, strict=True)]
    physical_gpu_total_per_head_query = [
        float(s.mean_physical_gpu_step_mb) + upd for s, upd in zip(layers, update_per_head_query, strict=True)
    ]
    return {
        "layers": int(len(layers)),
        "head_query_calls_total": int(sum(s.calls for s in layers)),
        "approx_attention_calls_total": total_approx_calls,
        "passthrough_attention_calls_total": total_passthrough_calls,
        "mean_step_MB_per_head_query": float(np.mean([s.mean_step_mb for s in layers])),
        "mean_logical_frontier_step_MB_per_head_query": float(np.mean([s.mean_step_mb for s in layers])),
        "mean_update_MB_per_head_query": float(np.mean(update_per_head_query)),
        "mean_total_MB_per_head_query": float(np.mean(total_per_head_query)),
        "mean_logical_frontier_total_MB_per_head_query": float(np.mean(total_per_head_query)),
        "mean_physical_gpu_total_MB_per_head_query": float(np.mean(physical_gpu_total_per_head_query)),
        "max_total_MB_per_head_query": float(np.max(total_per_head_query)),
        "max_physical_gpu_total_MB_per_head_query": float(np.max(physical_gpu_total_per_head_query)),
        "max_step_MB_per_head_query": float(np.max([s.mean_step_mb for s in layers])),
        "max_physical_gpu_step_MB_per_head_query": float(np.max([s.mean_physical_gpu_step_mb for s in layers])),
        "mean_selector_MB_per_head_query": float(np.mean([s.mean_selector_mb for s in layers])),
        "mean_logical_frontier_selector_MB_per_head_query": float(np.mean([s.mean_selector_mb for s in layers])),
        "mean_exact_KV_MB_per_head_query": float(np.mean([s.mean_exact_kv_mb for s in layers])),
        "mean_logical_frontier_exact_KV_MB_per_head_query": float(np.mean([s.mean_exact_kv_mb for s in layers])),
        "mean_tail_estimator_MB_per_head_query": float(np.mean([s.mean_tail_mb for s in layers])),
        "mean_logical_frontier_tail_estimator_MB_per_head_query": float(np.mean([s.mean_tail_mb for s in layers])),
        "mean_confidence_MB_per_head_query": float(np.mean([s.mean_confidence_mb for s in layers])),
        "mean_logical_frontier_confidence_MB_per_head_query": float(np.mean([s.mean_confidence_mb for s in layers])),
        "mean_physical_gpu_exact_KV_MB_per_head_query": float(
            np.mean([s.mean_physical_gpu_exact_kv_mb for s in layers])
        ),
        "mean_physical_gpu_confidence_MB_per_head_query": float(
            np.mean([s.mean_physical_gpu_confidence_mb for s in layers])
        ),
        "mean_physical_gpu_step_MB_per_head_query": float(np.mean([s.mean_physical_gpu_step_mb for s in layers])),
        "mean_selected_tokens": float(np.mean([s.mean_selected for s in layers])),
        "approx_path_active_fraction": approx_path_fraction,
        "selector_active_fraction": float(
            selector_active_calls / max(1, sum(int(s.calls) for s in layers))
            if selector_active_calls > 0
            else approx_path_fraction
        ),
        "tail_active_fraction": float(
            tail_active_calls / max(1, sum(int(s.calls) for s in layers))
            if tail_active_calls > 0
            else approx_path_fraction
        ),
        "confidence_active_fraction": float(
            sum(int(getattr(s, "confidence_active_calls", 0)) for s in layers)
            / max(1, sum(int(s.calls) for s in layers))
        ),
        "index_build_calls_total": int(sum(s.index_build_calls for s in layers)),
        "index_build_seconds_total": float(sum(s.index_build_seconds for s in layers)),
        "index_build_read_MB_total": float(sum(s.index_build_read_mb for s in layers)),
        "index_build_write_MB_total": float(sum(s.index_build_write_mb for s in layers)),
        "index_build_total_MB": float(sum(update_mbs)),
        "mean_index_build_seconds_per_layer": float(np.mean([s.index_build_seconds for s in layers])),
        "cache_cast_seconds_total": float(sum(s.cache_cast_seconds for s in layers)),
        "patched_attention_seconds_total": float(sum(s.patched_attention_seconds for s in layers)),
        "qkv_cache_seconds_total": float(sum(s.qkv_cache_seconds for s in layers)),
        "index_sidecar_seconds_total": float(sum(s.index_sidecar_seconds for s in layers)),
        "native_pack_seconds_total": float(sum(s.native_pack_seconds for s in layers)),
        "native_selector_seconds_total": float(sum(s.native_selector_seconds for s in layers)),
        "native_attention_seconds_total": float(sum(s.native_attention_seconds for s in layers)),
        "native_exact_logit_seconds_total": float(sum(float(getattr(s, "native_exact_logit_seconds", 0.0)) for s in layers)),
        "native_threshold_seconds_total": float(sum(float(getattr(s, "native_threshold_seconds", 0.0)) for s in layers)),
        "native_geometric_seconds_total": float(sum(float(getattr(s, "native_geometric_seconds", 0.0)) for s in layers)),
        "native_output_seconds_total": float(sum(float(getattr(s, "native_output_seconds", 0.0)) for s in layers)),
        "native_joint_rank_prefix_seconds_total": float(
            sum(float(getattr(s, "native_joint_rank_prefix_seconds", 0.0)) for s in layers)
        ),
        "native_joint_score_grid_seconds_total": float(
            sum(float(getattr(s, "native_joint_score_grid_seconds", 0.0)) for s in layers)
        ),
        "native_joint_prob_base_seconds_total": float(
            sum(float(getattr(s, "native_joint_prob_base_seconds", 0.0)) for s in layers)
        ),
        "native_joint_risk_prefix_seconds_total": float(
            sum(float(getattr(s, "native_joint_risk_prefix_seconds", 0.0)) for s in layers)
        ),
        "native_joint_policy_seconds_total": float(
            sum(float(getattr(s, "native_joint_policy_seconds", 0.0)) for s in layers)
        ),
        "native_joint_precompute_seconds_total": float(
            sum(float(getattr(s, "native_joint_precompute_seconds", 0.0)) for s in layers)
        ),
        "native_joint_layout_seconds_total": float(
            sum(float(getattr(s, "native_joint_layout_seconds", 0.0)) for s in layers)
        ),
        "native_joint_group_pack_seconds_total": float(
            sum(float(getattr(s, "native_joint_group_pack_seconds", 0.0)) for s in layers)
        ),
        "native_joint_accounting_seconds_total": float(
            sum(float(getattr(s, "native_joint_accounting_seconds", 0.0)) for s in layers)
        ),
        "joint_staged_kv_groups_total": int(sum(int(getattr(s, "joint_staged_kv_groups", 0)) for s in layers)),
        "joint_staged_kv_accepted_groups_total": int(
            sum(int(getattr(s, "joint_staged_kv_accepted_groups", 0)) for s in layers)
        ),
        "joint_staged_kv_boundary_groups_total": int(
            sum(int(getattr(s, "joint_staged_kv_boundary_groups", 0)) for s in layers)
        ),
        "joint_staged_kv_accept_fraction": float(
            sum(int(getattr(s, "joint_staged_kv_accepted_groups", 0)) for s in layers)
            / max(1, sum(int(getattr(s, "joint_staged_kv_groups", 0)) for s in layers))
        ),
        "wall_patched_attention_seconds_total": float(
            sum(float(getattr(s, "wall_patched_attention_seconds", 0.0)) for s in layers)
        ),
        "wall_qkv_cache_seconds_total": float(
            sum(float(getattr(s, "wall_qkv_cache_seconds", 0.0)) for s in layers)
        ),
        "wall_index_sidecar_seconds_total": float(
            sum(float(getattr(s, "wall_index_sidecar_seconds", 0.0)) for s in layers)
        ),
        "wall_output_projection_seconds_total": float(
            sum(float(getattr(s, "wall_output_projection_seconds", 0.0)) for s in layers)
        ),
        "wall_joint_total_seconds_total": float(
            sum(float(getattr(s, "wall_joint_total_seconds", 0.0)) for s in layers)
        ),
        "wall_joint_precompute_seconds_total": float(
            sum(float(getattr(s, "wall_joint_precompute_seconds", 0.0)) for s in layers)
        ),
        "wall_joint_selector_seconds_total": float(
            sum(float(getattr(s, "wall_joint_selector_seconds", 0.0)) for s in layers)
        ),
        "wall_joint_exact_logit_seconds_total": float(
            sum(float(getattr(s, "wall_joint_exact_logit_seconds", 0.0)) for s in layers)
        ),
        "wall_joint_vpq_sidecar_seconds_total": float(
            sum(float(getattr(s, "wall_joint_vpq_sidecar_seconds", 0.0)) for s in layers)
        ),
        "wall_joint_layout_seconds_total": float(
            sum(float(getattr(s, "wall_joint_layout_seconds", 0.0)) for s in layers)
        ),
        "wall_joint_rank_prefix_seconds_total": float(
            sum(float(getattr(s, "wall_joint_rank_prefix_seconds", 0.0)) for s in layers)
        ),
        "wall_joint_score_grid_seconds_total": float(
            sum(float(getattr(s, "wall_joint_score_grid_seconds", 0.0)) for s in layers)
        ),
        "wall_joint_prob_base_seconds_total": float(
            sum(float(getattr(s, "wall_joint_prob_base_seconds", 0.0)) for s in layers)
        ),
        "wall_joint_risk_prefix_seconds_total": float(
            sum(float(getattr(s, "wall_joint_risk_prefix_seconds", 0.0)) for s in layers)
        ),
        "wall_joint_policy_seconds_total": float(
            sum(float(getattr(s, "wall_joint_policy_seconds", 0.0)) for s in layers)
        ),
        "wall_joint_group_pack_seconds_total": float(
            sum(float(getattr(s, "wall_joint_group_pack_seconds", 0.0)) for s in layers)
        ),
        "wall_joint_accounting_seconds_total": float(
            sum(float(getattr(s, "wall_joint_accounting_seconds", 0.0)) for s in layers)
        ),
        "output_projection_seconds_total": float(sum(s.output_projection_seconds for s in layers)),
    }


def model_forward_last_logits(model: LlamaForCausalLM, input_ids: torch.Tensor, **kwargs):
    """Avoid materializing full-sequence logits for long-context prefill."""
    try:
        return model(input_ids, logits_to_keep=1, **kwargs)
    except TypeError as exc:
        if "logits_to_keep" not in str(exc):
            raise
        return model(input_ids, **kwargs)


def sync_cuda_if_needed(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def set_pagedpq_prefill_sidecar_warm_deferred(model: LlamaForCausalLM, enabled: bool) -> None:
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        return
    for layer in layers:
        attn = getattr(layer, "self_attn", None)
        if attn is not None:
            setattr(attn, "_pagedpq_defer_prefill_sidecar_warm", bool(enabled))


def warm_pagedpq_decode_sidecars(model: LlamaForCausalLM, past_key_values, device: torch.device) -> None:
    """Build paged-PQ decode sidecars after dense prefill, before decode timing."""
    if past_key_values is None:
        return
    log_cuda_memory("sidecar_warm.start", device)
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        return
    for layer in layers:
        attn = getattr(layer, "self_attn", None)
        warm = getattr(attn, "_pagedpq_warm_decode_sidecars", None)
        if callable(warm):
            warm(past_key_values)
    sync_cuda_if_needed(device)
    log_cuda_memory("sidecar_warm.done", device)


def maybe_empty_cache_after_prefill(device: torch.device) -> None:
    if not env_truthy("FRONTIER_EMPTY_CACHE_AFTER_PREFILL", "0"):
        return
    if device.type != "cuda" or not torch.cuda.is_available():
        return
    torch.cuda.empty_cache()
    log_cuda_memory("prefill.empty_cache.done", device)


def maybe_empty_cache_after_prefill_chunk(device: torch.device, label: str) -> None:
    if not env_truthy("FRONTIER_EMPTY_CACHE_AFTER_PREFILL_CHUNK", "0"):
        return
    if device.type != "cuda" or not torch.cuda.is_available():
        return
    torch.cuda.empty_cache()
    log_cuda_memory(label, device)


def prefill_batched(
    model: LlamaForCausalLM,
    input_ids: torch.Tensor,
    *,
    device: torch.device,
    prefill_chunk_size: int,
    past_key_values=None,
    dense_kv_offload: bool = False,
):
    prompt_len = int(input_ids.shape[1])
    chunk_size = int(prefill_chunk_size)
    if chunk_size <= 0 or chunk_size >= prompt_len:
        label = "offload.prefill.oneshot" if dense_kv_offload else "prefill.oneshot"
        log_cuda_memory(f"{label}.start", device, reset_peak=True)
        if dense_kv_offload:
            out = model_forward_last_logits(
                model,
                input_ids.to(device),
                past_key_values=past_key_values,
                use_cache=True,
            )
        else:
            out = model_forward_last_logits(model, input_ids.to(device), use_cache=True)
        sync_cuda_if_needed(device)
        log_cuda_memory(f"{label}.forward_done", device)
        return out

    log_cuda_memory(f"prefill.chunked.start.chunk{chunk_size}", device, reset_peak=True)
    out = None
    set_pagedpq_prefill_sidecar_warm_deferred(model, True)
    try:
        for start in range(0, prompt_len, chunk_size):
            stop = min(prompt_len, start + chunk_size)
            chunk = input_ids[:, start:stop].to(device)
            if dense_kv_offload:
                log_cuda_memory(f"offload.prefill.chunk.{start}_{stop}.start", device)
            out = model_forward_last_logits(
                model,
                chunk,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = out.past_key_values
            sync_cuda_if_needed(device)
            memory_label = (
                f"offload.prefill.chunk.{start}_{stop}.done"
                if dense_kv_offload
                else f"prefill.chunk.{start}_{stop}.done"
            )
            log_cuda_memory(memory_label, device)
            maybe_empty_cache_after_prefill_chunk(
                device,
                f"prefill.chunk.{start}_{stop}.empty_cache.done",
            )
    finally:
        set_pagedpq_prefill_sidecar_warm_deferred(model, False)
    if out is None:
        raise RuntimeError("empty prompt")
    log_cuda_memory("prefill.chunked.forward_done", device)
    return out


@torch.inference_mode()
def generate_batched(
    model: LlamaForCausalLM,
    tokenizer: AutoTokenizer,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    *,
    device: torch.device,
    prefill_chunk_size: int | None = None,
    dense_kv_offload: bool = False,
    dense_kv_block_tokens: int = 8192,
    dense_kv_staging_buffers: int = 2,
    dense_kv_query_block_tokens: int = 2048,
    greedy_logit_trace: list[torch.Tensor] | None = None,
    forced_token_ids: list[int] | None = None,
) -> tuple[list[int], dict[str, float | int]]:
    if input_ids.ndim != 2 or int(input_ids.shape[0]) != 1:
        raise ValueError("batched runner currently expects batch size 1")
    prompt_len = int(input_ids.shape[1])
    prefill_chunk_size_i = (
        int(os.environ.get("PREFILL_CHUNK_SIZE", "0") or "0")
        if prefill_chunk_size is None
        else int(prefill_chunk_size)
    )
    offload_cache = None
    if dense_kv_offload:
        if prefill_chunk_size_i <= 0:
            raise ValueError("dense KV offload requires a positive prefill_chunk_size")
        offload_cache = DenseKVOffloadCache(
            num_layers=len(model.model.layers),
            max_cache_len=prompt_len + int(max_new_tokens),
            kv_block_tokens=int(dense_kv_block_tokens),
            staging_buffers=int(dense_kv_staging_buffers),
            query_block_tokens=int(dense_kv_query_block_tokens),
            device=device,
        )
    forced_tokens = None
    if forced_token_ids is not None:
        if len(forced_token_ids) != int(max_new_tokens):
            raise ValueError(
                "forced token count must equal max_new_tokens: "
                f"{len(forced_token_ids)} != {int(max_new_tokens)}"
            )
        forced_tokens = torch.tensor(forced_token_ids, dtype=torch.long, device=device).view(1, -1)
    sync_cuda_if_needed(device)
    prompt_start = time.perf_counter()
    out = prefill_batched(
        model,
        input_ids,
        device=device,
        prefill_chunk_size=prefill_chunk_size_i,
        past_key_values=offload_cache,
        dense_kv_offload=dense_kv_offload,
    )
    sync_cuda_if_needed(device)
    past_key_values = out.past_key_values
    warm_pagedpq_decode_sidecars(model, past_key_values, device)
    maybe_empty_cache_after_prefill(device)
    next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
    if greedy_logit_trace is not None and int(max_new_tokens) > 0:
        greedy_logit_trace.append(out.logits[:, -1, :].detach().cpu())
    prompt_seconds = time.perf_counter() - prompt_start

    generated: list[int] = []
    log_cuda_memory("decode.start", device, reset_peak=True)
    sync_cuda_if_needed(device)
    decode_start = time.perf_counter()
    for step in range(int(max_new_tokens)):
        input_token = next_token if forced_tokens is None else forced_tokens[:, step : step + 1]
        token_id = int(input_token.item())
        generated.append(token_id)
        if dense_kv_offload:
            log_cuda_memory(f"offload.decode.step.{step}.start", device)
        out = model_forward_last_logits(model, input_token, past_key_values=past_key_values, use_cache=True)
        past_key_values = out.past_key_values
        next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        if greedy_logit_trace is not None and step + 1 < int(max_new_tokens):
            greedy_logit_trace.append(out.logits[:, -1, :].detach().cpu())
        if dense_kv_offload:
            log_cuda_memory(f"offload.decode.step.{step}.done", device)
    sync_cuda_if_needed(device)
    decode_seconds = time.perf_counter() - decode_start
    log_cuda_memory("decode.done", device)
    timing = {
        "prompt_tokens": int(prompt_len),
        "generated_tokens": int(len(generated)),
        "stream_prefill_seconds": float(prompt_seconds),
        "stream_decode_seconds": float(decode_seconds),
        "stream_total_seconds": float(prompt_seconds + decode_seconds),
        "stream_prefill_tokens_per_second": float(prompt_len / max(prompt_seconds, 1e-9)),
        "stream_decode_tokens_per_second": float(len(generated) / max(decode_seconds, 1e-9)),
    }
    if offload_cache is not None:
        timing["dense_kv_offload_h2d_bytes"] = int(offload_cache.h2d_bytes)
        log(
            "dense KV offload streamed "
            f"{offload_cache.h2d_bytes} H2D bytes "
            f"({offload_cache.h2d_bytes / (1024.0 ** 4):.3f} TiB)"
        )
    return generated, timing


@torch.inference_mode()
def generate_streaming(
    model: LlamaForCausalLM,
    tokenizer: AutoTokenizer,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    *,
    device: torch.device,
    prefill_chunk_size: int | None = None,
) -> tuple[list[int], dict[str, float | int]]:
    if input_ids.ndim != 2 or int(input_ids.shape[0]) != 1:
        raise ValueError("streaming runner currently expects batch size 1")
    past_key_values = None
    next_token = None
    prompt_len = int(input_ids.shape[1])
    sync_cuda_if_needed(device)
    prompt_start = time.perf_counter()
    for pos in range(prompt_len):
        out = model_forward_last_logits(
            model,
            input_ids[:, pos : pos + 1].to(device),
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = out.past_key_values
        next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
    sync_cuda_if_needed(device)
    warm_pagedpq_decode_sidecars(model, past_key_values, device)
    prompt_seconds = time.perf_counter() - prompt_start
    if next_token is None:
        raise RuntimeError("empty prompt")

    generated: list[int] = []
    sync_cuda_if_needed(device)
    decode_start = time.perf_counter()
    for _ in range(int(max_new_tokens)):
        token_id = int(next_token.item())
        generated.append(token_id)
        out = model_forward_last_logits(model, next_token.to(device), past_key_values=past_key_values, use_cache=True)
        past_key_values = out.past_key_values
        next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
    sync_cuda_if_needed(device)
    decode_seconds = time.perf_counter() - decode_start
    return generated, {
        "prompt_tokens": int(prompt_len),
        "generated_tokens": int(len(generated)),
        "stream_prefill_seconds": float(prompt_seconds),
        "stream_decode_seconds": float(decode_seconds),
        "stream_total_seconds": float(prompt_seconds + decode_seconds),
        "stream_prefill_tokens_per_second": float(prompt_len / max(prompt_seconds, 1e-9)),
        "stream_decode_tokens_per_second": float(len(generated) / max(decode_seconds, 1e-9)),
    }


def run() -> None:
    parser = argparse.ArgumentParser(
        description="Streaming RULER prediction with paged-PQ decode approximation and dense prefill by default."
    )
    parser.add_argument("--model_name", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--cache_dir", default=".hf_cache")
    parser.add_argument("--data_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--summary_file", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--num_samples", type=int, default=0)
    parser.add_argument(
        "--mode",
        choices=["dense_stream", "pagedpq_stream", "dense_batched", "pagedpq_batched"],
        default="pagedpq_batched",
    )
    parser.add_argument(
        "--approx_prefill",
        action="store_true",
        help="also apply paged-PQ attention during batched prefill; default is dense prefill + approximate decode",
    )
    parser.add_argument("--layers", default="all")
    parser.add_argument("--max_new_tokens", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--cpu_then_to_device", action="store_true")
    parser.add_argument("--selector_mode", choices=["fullscan", "routed", "oracle"], default="fullscan")
    parser.add_argument(
        "--selector_backend",
        choices=["torch", "cuda_ext", "auto"],
        default=os.environ.get("SELECTOR_PAGED_PQ_BACKEND", "torch"),
    )
    parser.add_argument("--budget", type=int, default=4096)
    parser.add_argument("--budget_by_head", default="")
    parser.add_argument("--rerank_candidates", type=int, default=0)
    parser.add_argument("--tail_samples", type=int, default=0)
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
        help="GPU simulator backend for exact logits used by frontier confidence checks.",
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
    parser.add_argument("--joint_kv_k_budget_fracs", default="0.10,0.30,0.50,0.70,0.90,1.0")
    parser.add_argument("--joint_kv_v_budget_fracs", default="0.05,0.10,0.20,0.40,0.60,0.80,1.0")
    parser.add_argument("--joint_kv_stability_threshold", type=float, default=0.002)
    parser.add_argument("--joint_kv_threshold_mode", choices=["fixed", "budget_delta_frac"], default="budget_delta_frac")
    parser.add_argument("--joint_kv_threshold_reference_frac", type=float, default=0.2)
    parser.add_argument("--joint_kv_threshold_scale_shape", choices=["linear", "sqrt", "log"], default="sqrt")
    parser.add_argument("--joint_kv_threshold_min_scale", type=float, default=0.0)
    parser.add_argument("--joint_kv_threshold_max_scale", type=float, default=1.5)
    parser.add_argument("--joint_kv_start_strategy", default="proxy_mass_m0p9")
    parser.add_argument("--joint_kv_deescalate", action="store_true",
                        help="frozen-spec de-escalation walk after the escalation stop")
    parser.add_argument("--logit_buffer_format", choices=["fp", "e4m3", "absmax_int"], default="fp",
                        help="e4m3 = frozen-sim: quantize the PQ score row to the fp8-e4m3 buffer grid before ranking/decisions (issue #6); "
                             "absmax_int = per-row symmetric absmax int8 buffer (the M4 arm, bits hardwired to 8 to match the CPU reference default)")
    parser.add_argument("--joint_kv_precision_tiers", action="store_true",
                        help="frozen-sim: int8 lo-tier K logits beyond the top-10%% ranked prefix + "
                             "V hi/lo split with the per-token int8 commit test (spec OPEN-2/M6)")
    parser.add_argument("--tail_blend", type=float, default=1.0)
    parser.add_argument("--prefill_tail_blend", type=float, default=None)
    parser.add_argument("--decode_tail_blend", type=float, default=None)
    parser.add_argument("--tail_off_heads", default="")
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
    parser.add_argument("--selected_value_exact_mass", type=float, default=0.98)
    parser.add_argument("--selected_value_exact_risk_mass", type=float, default=0.0)
    parser.add_argument("--selected_value_min_exact_top", type=int, default=0)
    parser.add_argument("--selected_value_max_exact_top", type=int, default=0)
    parser.add_argument("--selected_value_exact_all_context_max", type=int, default=0)
    parser.add_argument("--selected_value_exact_all_fraction_min", type=float, default=0.0)
    parser.add_argument("--selected_value_residual_norm_bytes", type=int, default=2)
    parser.add_argument("--value_code_stat_bytes", type=int, default=2)
    parser.add_argument("--static_prefix", type=int, default=128)
    parser.add_argument("--static_suffix", type=int, default=128)
    parser.add_argument("--page_size", type=int, default=5632)
    parser.add_argument("--prefill_chunk_size", type=int, default=0)
    parser.add_argument(
        "--dense_kv_offload",
        action="store_true",
        help="opt in to exact Qwen2 attention with a pinned-CPU bf16 KV cache",
    )
    parser.add_argument("--dense_kv_block_tokens", type=int, default=8192)
    parser.add_argument("--dense_kv_staging_buffers", type=int, default=2)
    parser.add_argument("--dense_kv_query_block_tokens", type=int, default=2048)
    parser.add_argument(
        "--greedy_logit_trace_file",
        default="",
        help="optional torch-save path for per-step logits used by greedy A/B validation",
    )
    parser.add_argument(
        "--forced_token_trace_file",
        default="",
        help="optional prior logit-trace file whose token_ids are teacher-forced during decode",
    )
    parser.add_argument(
        "--prefill_selector_backend",
        choices=["native", "native_fused", "torch_lut", "torch_lut_fp16", "torch_lut_streaming", "torch_lut_batched", "torch_matmul"],
        default="native",
    )
    parser.add_argument("--prefill_selector_stride", type=int, default=1)
    parser.add_argument("--prefill_selector_tile_size", type=int, default=0)
    parser.add_argument("--prefill_rank_buffer_limit_mb", type=float, default=4096.0)
    parser.add_argument("--prefill_selector_page_block_size", type=int, default=0)
    parser.add_argument("--prefill_tail_score_reuse", action="store_true")
    parser.add_argument(
        "--prefill_attention_backend",
        choices=["native", "flashinfer_blocksparse", "flashinfer_page_blocks"],
        default="native",
    )
    parser.add_argument("--subvecs", type=int, default=4)
    parser.add_argument("--subbits", type=int, default=8)
    parser.add_argument("--value_subvecs", type=int, default=1)
    parser.add_argument("--value_subbits", type=int, default=4)
    parser.add_argument("--value_pq_group_pages", type=int, default=1)
    parser.add_argument("--kmeans_iters", type=int, default=3)
    parser.add_argument(
        "--index_build_backend",
        choices=["numpy", "torch_gpu"],
        default=os.environ.get("PAGEDPQ_INDEX_BUILD_BACKEND", "torch_gpu"),
    )
    parser.add_argument("--nprobes", default="512")
    parser.add_argument("--router_prototypes", type=int, default=16)
    parser.add_argument("--router_merge_rel", type=float, default=0.05)
    parser.add_argument("--router_merge_var", type=float, default=0.0)
    parser.add_argument("--router_max_groups", type=int, default=512)
    parser.add_argument("--key_bytes", type=int, default=2)
    parser.add_argument("--value_bytes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--profile_native_ops", action="store_true")
    parser.add_argument("--disable_cost_stats", action="store_true")
    parser.add_argument("--disable_native_decode_fused", dest="disable_native_decode_fused", action="store_true", default=True)
    parser.add_argument("--enable_native_decode_fused", dest="disable_native_decode_fused", action="store_false")
    parser.add_argument("--native_decode_scoreless_fused", action="store_true")
    parser.add_argument("--native_decode_scoreless_force_mode", type=int, default=2)
    parser.add_argument("--allow_tf32_selector", action="store_true")
    parser.add_argument(
        "--native_decode_tail",
        action="store_true",
        help="experimental: use native compressed-tail attention for decode; default keeps the faster torch tail path",
    )
    args = parser.parse_args()
    setattr(args, "approx_prefill", bool(args.approx_prefill) and str(args.mode) == "pagedpq_batched")
    if bool(args.dense_kv_offload):
        if str(args.mode) != "dense_batched":
            parser.error("--dense_kv_offload requires --mode dense_batched")
        if int(args.prefill_chunk_size) <= 0:
            parser.error("--dense_kv_offload requires --prefill_chunk_size > 0")
        if int(args.dense_kv_block_tokens) <= 0 or int(args.dense_kv_query_block_tokens) <= 0:
            parser.error("dense KV block sizes must be positive")
        if int(args.dense_kv_staging_buffers) < 2:
            parser.error("--dense_kv_staging_buffers must be at least 2")

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    device = torch.device(args.device)
    if bool(args.allow_tf32_selector):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary_file)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    max_new_tokens = int(args.max_new_tokens) if int(args.max_new_tokens) > 0 else task_tokens_to_generate(str(args.task))
    rows = load_data(args.data_file)
    if int(args.num_samples) > 0:
        rows = rows[: int(args.num_samples)]
    forced_token_records = None
    if str(args.forced_token_trace_file):
        loaded_forced_records = torch.load(
            args.forced_token_trace_file,
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(loaded_forced_records, list) or len(loaded_forced_records) != len(rows):
            raise ValueError(
                "forced token trace must contain exactly one record per selected input row"
            )
        forced_token_records = []
        for record in loaded_forced_records:
            if not isinstance(record, dict) or "index" not in record or "token_ids" not in record:
                raise ValueError("forced token trace records require index and token_ids")
            forced_token_records.append(
                {
                    "index": record["index"],
                    "token_ids": [int(token_id) for token_id in record["token_ids"]],
                }
            )
        del loaded_forced_records
        log(f"loaded forced tokens from {args.forced_token_trace_file}")

    load_start = time.perf_counter()
    log("loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir, local_files_only=bool(args.local_files_only))
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer_load_seconds = time.perf_counter() - load_start
    model_load_start = time.perf_counter()
    log("loading model")
    kwargs = {
        "cache_dir": args.cache_dir,
        "local_files_only": bool(args.local_files_only),
        "torch_dtype": torch.bfloat16,
        "attn_implementation": str(args.attn_implementation),
    }
    if not bool(args.cpu_then_to_device):
        kwargs["device_map"] = {"": str(device)}
    # AutoModel resolves the architecture from config (Llama, Qwen2, ...).
    # The paged-PQ patch only requires the Llama-style module layout
    # (model.model.layers[i].self_attn with q/k/v/o_proj + head_dim),
    # which Qwen2 shares; Qwen2.5-1M additionally needs no rope_scaling.
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **kwargs)
    # Qwen2.5(-1M) configs carry sliding_window=32768 with
    # use_sliding_window=False. The attention layers gate on
    # use_sliding_window, but the transformers 4.49 causal-mask builder
    # gates on sliding_window alone and would band-mask everything
    # beyond 32k once kv_len > sliding_window — silently truncating the
    # dense baseline. Null it so the mask path matches the layers.
    if getattr(model.config, "use_sliding_window", None) is False and getattr(model.config, "sliding_window", None):
        log(f"disabling config.sliding_window={model.config.sliding_window} (use_sliding_window=False)")
        model.config.sliding_window = None
    if bool(args.cpu_then_to_device):
        model = model.to(device)
    model.eval()
    model_load_seconds = time.perf_counter() - model_load_start
    log_cuda_memory("model.load.done", device, reset_peak=True)

    if str(args.layers) == "all":
        layer_ids = list(range(len(model.model.layers)))
    else:
        layer_ids = parse_csv_ints(str(args.layers))
    approx_stats: dict[int, ApproxStats] = {}
    results = []
    start_all = time.perf_counter()

    attn_noise_counters = None
    attn_noise_config = None
    if str(args.mode) in {"pagedpq_stream", "pagedpq_batched"}:
        context = patched_paged_pq_attention(model, layer_ids, args, approx_stats)
    elif bool(args.dense_kv_offload):
        context = patched_qwen2_dense_kv_offload(model)
    else:
        context, attn_noise_config = maybe_attn_output_noise_patch(model)
        if attn_noise_config is not None:
            log(f"attn_output_noise active: {attn_noise_config}")

    logit_trace_records = [] if str(args.greedy_logit_trace_file) else None
    with context as context_value:
        if attn_noise_config is not None:
            attn_noise_counters = context_value
        with out_path.open("w", encoding="utf-8", buffering=1) as fout:
            for sample_idx, sample in enumerate(tqdm(rows, desc=f"{args.mode}:{args.task}")):
                if str(args.mode) in {"pagedpq_stream", "pagedpq_batched"}:
                    reset_paged_pq_attention_state(model)
                input_ids = tokenizer(str(sample["input"]), return_tensors="pt").input_ids
                generate_fn = generate_batched if str(args.mode) in {"dense_batched", "pagedpq_batched"} else generate_streaming
                generate_kwargs = {
                    "model": model,
                    "tokenizer": tokenizer,
                    "input_ids": input_ids,
                    "max_new_tokens": max_new_tokens,
                    "device": device,
                    "prefill_chunk_size": int(args.prefill_chunk_size),
                }
                sample_logit_trace = None
                if bool(args.dense_kv_offload):
                    generate_kwargs.update(
                        dense_kv_offload=True,
                        dense_kv_block_tokens=int(args.dense_kv_block_tokens),
                        dense_kv_staging_buffers=int(args.dense_kv_staging_buffers),
                        dense_kv_query_block_tokens=int(args.dense_kv_query_block_tokens),
                    )
                if logit_trace_records is not None:
                    if generate_fn is not generate_batched:
                        raise ValueError("greedy logit tracing is supported only by batched generation")
                    sample_logit_trace = []
                    generate_kwargs["greedy_logit_trace"] = sample_logit_trace
                if forced_token_records is not None:
                    if generate_fn is not generate_batched:
                        raise ValueError("forced tokens are supported only by batched generation")
                    forced_record = forced_token_records[sample_idx]
                    if forced_record.get("index") != sample["index"]:
                        raise ValueError(
                            "forced token trace/input index mismatch at row "
                            f"{sample_idx}: {forced_record.get('index')} != {sample['index']}"
                        )
                    generate_kwargs["forced_token_ids"] = forced_record["token_ids"]
                generated, timing = generate_fn(**generate_kwargs)
                if logit_trace_records is not None:
                    logit_trace_records.append(
                        {
                            "index": sample["index"],
                            "prompt_tokens": int(input_ids.shape[1]),
                            "token_ids": generated,
                            "logits": torch.stack(sample_logit_trace, dim=0),
                        }
                    )
                pred = tokenizer.decode(generated, skip_special_tokens=True)
                item = {
                    "index": sample["index"],
                    "pred": pred,
                    "input": sample["input"],
                    "outputs": sample["outputs"],
                    "others": sample.get("others", {}),
                    "truncation": sample.get("truncation", -1),
                    "length": sample.get("length", -1),
                    "timing": timing,
                }
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                results.append(item)

    elapsed = time.perf_counter() - start_all
    timing_rows = [r["timing"] for r in results]
    total_prefill_seconds = float(sum(float(r["stream_prefill_seconds"]) for r in timing_rows))
    total_decode_seconds = float(sum(float(r["stream_decode_seconds"]) for r in timing_rows))
    total_stream_seconds = float(sum(float(r["stream_total_seconds"]) for r in timing_rows))
    total_generated_tokens = int(sum(int(r["generated_tokens"]) for r in timing_rows))
    summary = {
        "mode": str(args.mode),
        "approx_prefill": bool(args.approx_prefill),
        "attn_output_noise": (
            {**attn_noise_config, **(attn_noise_counters or {})}
            if attn_noise_config is not None
            else None
        ),
        "task": str(args.task),
        "samples": int(len(results)),
        "layers": layer_ids,
        "max_new_tokens": int(max_new_tokens),
        "elapsed_seconds": float(elapsed),
        "tokenizer_load_seconds": float(tokenizer_load_seconds),
        "model_load_seconds": float(model_load_seconds),
        "mean_prompt_tokens": float(np.mean([r["prompt_tokens"] for r in timing_rows])) if timing_rows else 0.0,
        "mean_generated_tokens": float(np.mean([r["generated_tokens"] for r in timing_rows])) if timing_rows else 0.0,
        "mean_stream_prefill_seconds": float(np.mean([r["stream_prefill_seconds"] for r in timing_rows])) if timing_rows else 0.0,
        "mean_stream_decode_seconds": float(np.mean([r["stream_decode_seconds"] for r in timing_rows])) if timing_rows else 0.0,
        "mean_stream_total_seconds": float(np.mean([r["stream_total_seconds"] for r in timing_rows])) if timing_rows else 0.0,
        "total_stream_prefill_seconds": float(total_prefill_seconds),
        "total_stream_decode_seconds": float(total_decode_seconds),
        "total_stream_seconds": float(total_stream_seconds),
        "prefill_fraction_of_stream_time": float(total_prefill_seconds / max(total_stream_seconds, 1e-9)),
        "decode_fraction_of_stream_time": float(total_decode_seconds / max(total_stream_seconds, 1e-9)),
        "decode_ms_per_generated_token": float(1000.0 * total_decode_seconds / max(1, total_generated_tokens)),
        "pagedpq_config": {
            "approx_prefill": bool(args.approx_prefill),
            "frontier_canonical_gpu": str(os.environ.get("FRONTIER_CANONICAL_GPU", "0")),
            "disable_cost_stats": bool(args.disable_cost_stats),
            "disable_native_decode_fused": bool(args.disable_native_decode_fused),
            "native_decode_scoreless_fused": bool(args.native_decode_scoreless_fused),
            "native_decode_scoreless_force_mode": int(args.native_decode_scoreless_force_mode),
            "native_decode_tail": bool(args.native_decode_tail),
            "budget": int(args.budget),
            "online_confidence_rule": str(args.online_confidence_rule),
            "tail_score_calibration": str(args.tail_score_calibration),
            "tail_probe_rel_l2_max": float(args.tail_probe_rel_l2_max),
            "tail_proxy_mass_min": float(args.tail_proxy_mass_min),
            "tail_proxy_mass_max": float(args.tail_proxy_mass_max),
            "tail_pq_corr_min": float(args.tail_pq_corr_min),
            "tail_pq_relrmse_max": float(args.tail_pq_relrmse_max),
            "ranked_confidence_cost_mode": str(args.ranked_confidence_cost_mode),
            "exact_logit_backend": str(args.exact_logit_backend),
            "geometric_min_budget": int(args.geometric_min_budget),
            "geometric_max_budget": int(args.geometric_max_budget),
            "geometric_budget_granularity": int(args.geometric_budget_granularity),
            "joint_kv_policy": str(getattr(args, "joint_kv_policy", "")),
            "joint_kv_k_budgets": str(getattr(args, "joint_kv_k_budgets", "")),
            "joint_kv_v_budgets": str(getattr(args, "joint_kv_v_budgets", "")),
            "joint_kv_k_budget_fracs": str(getattr(args, "joint_kv_k_budget_fracs", "")),
            "joint_kv_v_budget_fracs": str(getattr(args, "joint_kv_v_budget_fracs", "")),
            "joint_kv_stability_threshold": float(getattr(args, "joint_kv_stability_threshold", 0.0)),
            "joint_kv_threshold_mode": str(getattr(args, "joint_kv_threshold_mode", "")),
            "joint_kv_threshold_reference_frac": float(getattr(args, "joint_kv_threshold_reference_frac", 0.0)),
            "joint_kv_threshold_scale_shape": str(getattr(args, "joint_kv_threshold_scale_shape", "")),
            "joint_kv_threshold_min_scale": float(getattr(args, "joint_kv_threshold_min_scale", 0.0)),
            "joint_kv_threshold_max_scale": float(getattr(args, "joint_kv_threshold_max_scale", 0.0)),
            "joint_kv_start_strategy": str(getattr(args, "joint_kv_start_strategy", "")),
            "joint_kv_deescalate": bool(getattr(args, "joint_kv_deescalate", False)),
            "logit_buffer_format": str(getattr(args, "logit_buffer_format", "fp")),
            "joint_kv_precision_tiers": bool(getattr(args, "joint_kv_precision_tiers", False)),
            "tail_blend": float(args.tail_blend),
            "selected_value_mode": str(args.selected_value_mode),
            "selected_value_exact_rule": str(args.selected_value_exact_rule),
            "selected_value_exact_top": int(args.selected_value_exact_top),
            "selected_value_exact_mass": float(args.selected_value_exact_mass),
            "selected_value_exact_risk_mass": float(args.selected_value_exact_risk_mass),
            "selected_value_min_exact_top": int(args.selected_value_min_exact_top),
            "selected_value_max_exact_top": int(args.selected_value_max_exact_top),
            "selector_mode": str(args.selector_mode),
            "selector_backend": str(args.selector_backend),
            "page_size": int(args.page_size),
            "subvecs": int(args.subvecs),
            "subbits": int(args.subbits),
            "value_subvecs": int(args.value_subvecs),
            "value_subbits": int(args.value_subbits),
            "value_pq_group_pages": int(args.value_pq_group_pages),
            "kmeans_iters": int(args.kmeans_iters),
            "nprobes": str(args.nprobes),
            "prefill_chunk_size": int(args.prefill_chunk_size),
            "prefill_selector_backend": str(args.prefill_selector_backend),
            "prefill_selector_tile_size": int(args.prefill_selector_tile_size),
            "prefill_rank_buffer_limit_mb": float(args.prefill_rank_buffer_limit_mb),
            "prefill_tail_score_reuse": bool(args.prefill_tail_score_reuse),
            "value_code_stat_bytes": int(getattr(args, "value_code_stat_bytes", 0)),
            "index_build_backend": str(args.index_build_backend),
            "allow_tf32_selector": bool(args.allow_tf32_selector),
            **joint_cuda_flags_config(),
        },
        "cost_proxy": stats_payload(approx_stats),
        "cost_proxy_aggregate": aggregate_stats(approx_stats),
    }
    if bool(args.dense_kv_offload):
        total_h2d_bytes = int(sum(int(r.get("dense_kv_offload_h2d_bytes", 0)) for r in timing_rows))
        summary["dense_kv_offload"] = {
            "enabled": True,
            "kv_block_tokens": int(args.dense_kv_block_tokens),
            "query_block_tokens": int(args.dense_kv_query_block_tokens),
            "staging_buffers": int(args.dense_kv_staging_buffers),
            "qk_compute": (
                "tf32_bf16_exact_fp32_output" if device.type == "cuda" else "fp32"
            ),
            "av_compute": "fp32",
            "h2d_bytes": total_h2d_bytes,
            "h2d_tib": float(total_h2d_bytes / (1024.0 ** 4)),
        }
    if forced_token_records is not None:
        summary["teacher_forced"] = True
    if logit_trace_records is not None:
        trace_path = Path(args.greedy_logit_trace_file)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(logit_trace_records, trace_path)
        log(f"wrote greedy logit trace {trace_path}")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    log(f"wrote {out_path} and {summary_path}")


if __name__ == "__main__":
    run()
