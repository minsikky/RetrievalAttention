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
from transformers import AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaForCausalLM

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.ruler.pred.utils import load_data
from benchmark.selector_eval.gpu.run_gpu_paged_pq_eval import parse_csv_ints
from benchmark.selector_eval.runners.run_hf_paged_pq_intervention_eval import (
    ApproxStats,
    patched_paged_pq_attention,
    reset_paged_pq_attention_state,
)


def log(msg: str) -> None:
    print(f"[pagedpq_stream_ruler] {time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


def env_truthy(name: str, default: str = "0") -> bool:
    return str(os.environ.get(name, default)).lower() not in {"0", "false", "no", "off", ""}


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
        "selector_pq_joint_allhead_rank_prefix": env_truthy("SELECTOR_PQ_JOINT_ALLHEAD_RANK_PREFIX", "0"),
        "selector_pq_joint_skip_full_budget_sort": env_truthy("SELECTOR_PQ_JOINT_SKIP_FULL_BUDGET_SORT", "0"),
        "selector_pq_joint_collapse_dup_k_rows": env_truthy("SELECTOR_PQ_JOINT_COLLAPSE_DUP_K_ROWS", "0"),
        "selector_pq_joint_collapse_dup_v_rows": env_truthy("SELECTOR_PQ_JOINT_COLLAPSE_DUP_V_ROWS", "0"),
        "selector_pq_joint_score_grid_no_exact_fill": env_truthy("SELECTOR_PQ_JOINT_SCORE_GRID_NO_EXACT_FILL", "0"),
        "selector_pq_joint_rankpos_score_grid": env_truthy("SELECTOR_PQ_JOINT_RANKPOS_SCORE_GRID", "0"),
        "selector_pq_joint_grouped_vpq_cache": env_truthy("SELECTOR_PQ_JOINT_GROUPED_VPQ_CACHE", "0"),
        "selector_pq_joint_fused_risk_policy": env_truthy("SELECTOR_PQ_JOINT_FUSED_RISK_POLICY", "0"),
        "selector_pq_joint_risk_prefix_topk": env_truthy("SELECTOR_PQ_JOINT_RISK_PREFIX_TOPK", "0"),
        "selector_pq_joint_fast_token_layout": env_truthy("SELECTOR_PQ_JOINT_FAST_TOKEN_LAYOUT", "0"),
        "selector_pq_joint_native_vpq_base": env_truthy("SELECTOR_PQ_JOINT_NATIVE_VPQ_BASE", "0"),
        "selector_pq_joint_native_softmax_base": env_truthy("SELECTOR_PQ_JOINT_NATIVE_SOFTMAX_BASE", "0"),
        "selector_pq_joint_fused_softmax_base": env_truthy("SELECTOR_PQ_JOINT_FUSED_SOFTMAX_BASE", "0"),
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


def warm_pagedpq_decode_sidecars(model: LlamaForCausalLM, past_key_values, device: torch.device) -> None:
    """Build paged-PQ decode sidecars after dense prefill, before decode timing."""
    if past_key_values is None:
        return
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        return
    for layer in layers:
        attn = getattr(layer, "self_attn", None)
        warm = getattr(attn, "_pagedpq_warm_decode_sidecars", None)
        if callable(warm):
            warm(past_key_values)
    sync_cuda_if_needed(device)


@torch.inference_mode()
def generate_batched(
    model: LlamaForCausalLM,
    tokenizer: AutoTokenizer,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    *,
    device: torch.device,
) -> tuple[list[int], dict[str, float | int]]:
    if input_ids.ndim != 2 or int(input_ids.shape[0]) != 1:
        raise ValueError("batched runner currently expects batch size 1")
    prompt_len = int(input_ids.shape[1])
    sync_cuda_if_needed(device)
    prompt_start = time.perf_counter()
    out = model_forward_last_logits(model, input_ids.to(device), use_cache=True)
    sync_cuda_if_needed(device)
    past_key_values = out.past_key_values
    warm_pagedpq_decode_sidecars(model, past_key_values, device)
    next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
    prompt_seconds = time.perf_counter() - prompt_start

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


@torch.inference_mode()
def generate_streaming(
    model: LlamaForCausalLM,
    tokenizer: AutoTokenizer,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    *,
    device: torch.device,
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
        default="none",
    )
    parser.add_argument("--tail_score_calibration", choices=["none", "affine_selected"], default="affine_selected")
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
    parser.add_argument("--joint_kv_stability_threshold", type=float, default=0.001)
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
        default="selected_mass",
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
        default=os.environ.get("PAGEDPQ_INDEX_BUILD_BACKEND", "numpy"),
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
    model = LlamaForCausalLM.from_pretrained(args.model_name, **kwargs)
    if bool(args.cpu_then_to_device):
        model = model.to(device)
    model.eval()
    model_load_seconds = time.perf_counter() - model_load_start

    if str(args.layers) == "all":
        layer_ids = list(range(len(model.model.layers)))
    else:
        layer_ids = parse_csv_ints(str(args.layers))
    approx_stats: dict[int, ApproxStats] = {}
    results = []
    start_all = time.perf_counter()

    context = (
        patched_paged_pq_attention(model, layer_ids, args, approx_stats)
        if str(args.mode) in {"pagedpq_stream", "pagedpq_batched"}
        else None
    )
    if context is None:
        class NullContext:
            def __enter__(self):
                return None
            def __exit__(self, exc_type, exc, tb):
                return False
        context = NullContext()

    with context:
        with out_path.open("w", encoding="utf-8", buffering=1) as fout:
            for sample in tqdm(rows, desc=f"{args.mode}:{args.task}"):
                if str(args.mode) in {"pagedpq_stream", "pagedpq_batched"}:
                    reset_paged_pq_attention_state(model)
                input_ids = tokenizer(str(sample["input"]), return_tensors="pt").input_ids
                generate_fn = generate_batched if str(args.mode) in {"dense_batched", "pagedpq_batched"} else generate_streaming
                generated, timing = generate_fn(
                    model=model,
                    tokenizer=tokenizer,
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    device=device,
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
            "joint_kv_stability_threshold": float(getattr(args, "joint_kv_stability_threshold", 0.0)),
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
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    log(f"wrote {out_path} and {summary_path}")


if __name__ == "__main__":
    run()
