#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaForCausalLM, apply_rotary_pos_emb

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.selector_eval.data.trace import static_tokens, unique_tokens
from benchmark.selector_eval.gpu.run_gpu_paged_pq_eval import (
    GPUIndex,
    _sync_if_cuda,
    build_page_pq_gpu,
    build_page_pq_torch,
    load_selector_paged_pq_ext,
    parse_csv_ints,
    rank_paged_pq_batched,
    rank_paged_pq_batched_with_scores,
    rank_paged_pq,
    selector_bytes_fullscan,
    selected_plus_tail_output,
    ensure_native_fullscan_pack,
)
from benchmark.selector_eval.runners.diagnose_layer_heads import _build_value_vpq_sidecars, _compressed_tail_output
from benchmark.selector_eval.runners.run_layer_quality_eval import (
    _fit_selected_pq_logit_uncertainty,
    _proxy_selected_mass,
    _round_budget_up,
    _selected_for_budget,
    _vpq_values_for_tokens,
)
from benchmark.selector_eval.runners.run_value_exact_strategy_eval import (
    mixed_scores as frontier_mixed_scores,
    output_from_exact_mask as frontier_output_from_exact_mask,
    top_mask as frontier_top_mask,
)
from benchmark.selector_eval.metrics.attention import _output_error_metrics

MB = 1024.0 * 1024.0


def log(msg: str) -> None:
    print(f"[hf_paged_pq_intervention_eval] {time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


def _env_truthy(name: str, default: str = "0") -> bool:
    return str(os.environ.get(name, default)).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = str(os.environ.get(name, str(default))).strip()
    try:
        return int(raw)
    except ValueError:
        return int(default)


def _canonical_gpu_frontier_mismatches(args) -> list[str]:
    """Return config fields that would change CPU-frontier algorithm semantics.

    This is intentionally about the algorithmic contract, not every tunable
    threshold. Budgets and thresholds may be swept; selector/value/tail
    semantics must not silently become a faster diagnostic variant.
    """
    mismatches: list[str] = []
    if str(getattr(args, "selector_mode", "")) != "fullscan":
        mismatches.append("selector_mode must be fullscan")
    if str(getattr(args, "selector_backend", "")) not in {"cuda_ext", "auto"}:
        mismatches.append("selector_backend must be cuda_ext or auto")
    if bool(getattr(args, "approx_prefill", False)):
        mismatches.append("approx_prefill must be disabled; canonical frontier is decode-only")
    if str(getattr(args, "index_build_backend", "")) != "torch_gpu":
        mismatches.append("index_build_backend must be torch_gpu")
    if str(getattr(args, "online_confidence_rule", "")) != "joint_kv_stability":
        mismatches.append("online_confidence_rule must be joint_kv_stability")
    if str(getattr(args, "tail_mode", "")) != "vpq_value":
        mismatches.append("tail_mode must be vpq_value")
    if str(getattr(args, "tail_score_calibration", "")) != "affine_selected":
        mismatches.append("tail_score_calibration must be affine_selected")
    if str(getattr(args, "selected_value_mode", "")) != "vpq_value":
        mismatches.append("selected_value_mode must be vpq_value")
    if str(getattr(args, "selected_value_exact_rule", "")) != "global_residual_risk":
        mismatches.append("selected_value_exact_rule must be global_residual_risk")
    if str(getattr(args, "ranked_confidence_cost_mode", "")) != "exact":
        mismatches.append("ranked_confidence_cost_mode must be exact")
    if _env_truthy("SELECTOR_PQ_JOINT_SEGMENTED_V_PREFIX", "0"):
        mismatches.append("SELECTOR_PQ_JOINT_SEGMENTED_V_PREFIX must be disabled; diagnostic changes accepted budgets")
    if _env_truthy("SELECTOR_PQ_JOINT_UNSORTED_V_PREFIX", "0"):
        mismatches.append("SELECTOR_PQ_JOINT_UNSORTED_V_PREFIX must be disabled until parity/profile validation promotes it")
    if _env_truthy("SELECTOR_PQ_JOINT_FAST_AFFINE_SELECTED", "0"):
        mismatches.append("SELECTOR_PQ_JOINT_FAST_AFFINE_SELECTED must be disabled until parity/profile validation promotes it")
    if _env_truthy("SELECTOR_PQ_JOINT_ONDEMAND_V_PREFIX", "0"):
        mismatches.append("SELECTOR_PQ_JOINT_ONDEMAND_V_PREFIX must be disabled until parity/profile validation promotes it")
    if _env_truthy("SELECTOR_PQ_JOINT_INCREMENTAL_V_GRID", "0"):
        mismatches.append("SELECTOR_PQ_JOINT_INCREMENTAL_V_GRID must be disabled until parity/profile validation promotes it")
    if _env_truthy("SELECTOR_PQ_JOINT_INCREMENTAL_VPQ_SIDECAR", "0"):
        mismatches.append(
            "SELECTOR_PQ_JOINT_INCREMENTAL_VPQ_SIDECAR must be disabled; long-decode validation did not improve runtime"
        )
    if not _env_truthy("SELECTOR_PQ_JOINT_GROUPED_RISK_PREFIX", "0"):
        mismatches.append("SELECTOR_PQ_JOINT_GROUPED_RISK_PREFIX must be enabled; it is the validated canonical grouped risk-prefix path")
    if _env_truthy("SELECTOR_PQ_JOINT_NATIVE_LAZY_POLICY", "0"):
        mismatches.append("SELECTOR_PQ_JOINT_NATIVE_LAZY_POLICY must be disabled until parity/profile validation promotes it")
    if not _env_truthy("SELECTOR_PQ_JOINT_NATIVE_RISK_PREFIX", "0"):
        mismatches.append("SELECTOR_PQ_JOINT_NATIVE_RISK_PREFIX must be enabled; it is the validated canonical risk-prefix path")
    if not _env_truthy("SELECTOR_PQ_JOINT_NATIVE_SCORE_GRID", "0"):
        mismatches.append("SELECTOR_PQ_JOINT_NATIVE_SCORE_GRID must be enabled; it is the validated canonical score-grid path")
    if not _env_truthy("SELECTOR_PQ_JOINT_NATIVE_POLICY", "0"):
        mismatches.append("SELECTOR_PQ_JOINT_NATIVE_POLICY must be enabled; it is the validated canonical policy path")
    if not _env_truthy("SELECTOR_PQ_JOINT_NATIVE_V_PREFIX", "0"):
        mismatches.append("SELECTOR_PQ_JOINT_NATIVE_V_PREFIX must be enabled; it is the validated canonical V-prefix path")
    if not _env_truthy("SELECTOR_PQ_JOINT_PREWARM_VPQ_SIDECARS", "0"):
        mismatches.append("SELECTOR_PQ_JOINT_PREWARM_VPQ_SIDECARS must be enabled for benchmark-ready decode runtime")
    if not _env_truthy("SELECTOR_PQ_JOINT_PERSISTENT_VPQ_CACHE", "0"):
        mismatches.append("SELECTOR_PQ_JOINT_PERSISTENT_VPQ_CACHE must be enabled for benchmark-ready decode runtime")
    if _env_truthy("SELECTOR_PQ_JOINT_ALLHEAD_EXACT_PRECOMPUTE", "0"):
        mismatches.append("SELECTOR_PQ_JOINT_ALLHEAD_EXACT_PRECOMPUTE must be disabled; diagnostic changed accepted budgets")
    if _env_truthy("SELECTOR_PQ_JOINT_ALLHEAD_RANK_PREFIX", "0"):
        mismatches.append("SELECTOR_PQ_JOINT_ALLHEAD_RANK_PREFIX must be disabled; diagnostic changed accepted budgets")
    if _env_truthy("SELECTOR_PQ_JOINT_NATIVE_RANK_PREFIX", "0"):
        mismatches.append("SELECTOR_PQ_JOINT_NATIVE_RANK_PREFIX must be disabled until parity/profile validation promotes it")
    if _env_truthy("SELECTOR_PQ_JOINT_SKIP_FULL_BUDGET_SORT", "0"):
        mismatches.append("SELECTOR_PQ_JOINT_SKIP_FULL_BUDGET_SORT must be disabled; diagnostic changed accepted budgets")
    if not _env_truthy("SELECTOR_PQ_JOINT_EXACT_FULL_BUDGET_GRID", "1"):
        mismatches.append("SELECTOR_PQ_JOINT_EXACT_FULL_BUDGET_GRID must be enabled; it is the validated canonical score-grid shortcut")
    if _env_truthy("SELECTOR_PQ_JOINT_SCORE_GRID_NO_EXACT_FILL", "0"):
        mismatches.append("SELECTOR_PQ_JOINT_SCORE_GRID_NO_EXACT_FILL must be disabled until parity/profile validation promotes it")
    if _env_truthy("SELECTOR_PQ_JOINT_RANKPOS_SCORE_GRID", "0"):
        mismatches.append("SELECTOR_PQ_JOINT_RANKPOS_SCORE_GRID must be disabled until parity/profile validation promotes it")
    if _env_truthy("SELECTOR_PQ_JOINT_GROUPED_VPQ_CACHE", "0"):
        mismatches.append("SELECTOR_PQ_JOINT_GROUPED_VPQ_CACHE must be disabled; diagnostic worsened runtime and is not canonical")
    if _env_truthy("SELECTOR_PQ_JOINT_FUSED_RISK_POLICY", "0"):
        mismatches.append("SELECTOR_PQ_JOINT_FUSED_RISK_POLICY must be disabled until parity/profile validation promotes it")
    if _env_truthy("SELECTOR_PQ_JOINT_RISK_PREFIX_TOPK", "0"):
        mismatches.append("SELECTOR_PQ_JOINT_RISK_PREFIX_TOPK must be disabled until parity/profile validation promotes it")
    if not _env_truthy("SELECTOR_PQ_JOINT_FAST_TOKEN_LAYOUT", "0"):
        mismatches.append("SELECTOR_PQ_JOINT_FAST_TOKEN_LAYOUT must be enabled; it is the validated canonical token-layout path")
    if _env_truthy("SELECTOR_PQ_JOINT_NATIVE_VPQ_BASE", "0"):
        mismatches.append("SELECTOR_PQ_JOINT_NATIVE_VPQ_BASE must be disabled until parity/profile validation promotes it")
    if not _env_truthy("SELECTOR_PQ_JOINT_NATIVE_SOFTMAX_BASE", "0"):
        mismatches.append("SELECTOR_PQ_JOINT_NATIVE_SOFTMAX_BASE must be enabled; it is the validated canonical prob/base path")
    if _env_truthy("SELECTOR_PQ_JOINT_FUSED_SOFTMAX_BASE", "0"):
        mismatches.append("SELECTOR_PQ_JOINT_FUSED_SOFTMAX_BASE must be disabled until parity/profile validation promotes it")
    if not _env_truthy("SELECTOR_PQ_JOINT_COLLAPSE_DUP_K_ROWS", "0"):
        mismatches.append("SELECTOR_PQ_JOINT_COLLAPSE_DUP_K_ROWS must be enabled; it is the validated duplicate-budget collapse path")
    if _env_truthy("SELECTOR_PQ_JOINT_COLLAPSE_DUP_V_ROWS", "0"):
        mismatches.append("SELECTOR_PQ_JOINT_COLLAPSE_DUP_V_ROWS must be disabled until parity/profile validation promotes it")
    threshold = float(getattr(args, "joint_kv_stability_threshold", float("inf")))
    if not math.isfinite(threshold) or threshold <= 0.0:
        mismatches.append("joint_kv_stability_threshold must be finite and positive")
    return mismatches


def _parse_budget_schedule(text: str, *, name: str) -> list[int]:
    values = sorted({int(x) for x in parse_csv_ints(text) if int(x) > 0})
    if not values:
        raise ValueError(f"{name} must contain at least one positive integer budget")
    return values


def _rel_l2_np(a: np.ndarray, b: np.ndarray) -> float:
    aa = a.astype(np.float64, copy=False)
    bb = b.astype(np.float64, copy=False)
    return float(np.linalg.norm(aa - bb)) / max(float(np.linalg.norm(bb)), 1e-20)


def _rel_l2_torch(a: torch.Tensor, b: torch.Tensor) -> float:
    aa = a.to(dtype=torch.float64)
    bb = b.to(dtype=torch.float64)
    return float(torch.linalg.vector_norm(aa - bb).item()) / max(float(torch.linalg.vector_norm(bb).item()), 1e-20)


def _choose_joint_kv_action(
    *,
    policy: str,
    k_delta: float,
    v_delta: float,
    k_can: bool,
    v_can: bool,
    threshold: float,
    turn: int,
    extra_k_mb: float,
    extra_v_mb: float,
) -> str:
    k_bad = bool(k_can and float(k_delta) > float(threshold))
    v_bad = bool(v_can and float(v_delta) > float(threshold))
    if not k_bad and not v_bad:
        return "stop"
    if str(policy) == "k_first_priority":
        return "k" if k_bad else "v"
    if str(policy) == "v_first_priority":
        return "v" if v_bad else "k"
    if str(policy) == "k_first_alternating":
        preferred = "k" if int(turn) % 2 == 0 else "v"
        if preferred == "k" and k_bad:
            return "k"
        if preferred == "v" and v_bad:
            return "v"
        return "v" if v_bad else "k"
    if str(policy) == "v_first_alternating":
        preferred = "v" if int(turn) % 2 == 0 else "k"
        if preferred == "v" and v_bad:
            return "v"
        if preferred == "k" and k_bad:
            return "k"
        return "k" if k_bad else "v"
    if str(policy) == "sensitivity_greedy":
        k_gain = float(k_delta) / max(float(extra_k_mb), 1e-9) if k_bad else -1.0
        v_gain = float(v_delta) / max(float(extra_v_mb), 1e-9) if v_bad else -1.0
        return "k" if k_gain >= v_gain else "v"
    raise ValueError(f"unknown joint_kv_policy: {policy}")


def _joint_kv_policy_id(policy: str) -> int:
    policy_map = {
        "k_first_priority": 0,
        "v_first_priority": 1,
        "k_first_alternating": 2,
        "v_first_alternating": 3,
        "sensitivity_greedy": 4,
    }
    try:
        return policy_map[str(policy)]
    except KeyError as exc:
        raise ValueError(f"unknown joint_kv_policy: {policy}") from exc


def _simulate_joint_kv_policy(
    *,
    outputs: dict[tuple[int, int], np.ndarray],
    k_budgets: list[int],
    v_budgets: list[int],
    policy: str,
    threshold: float,
    k_mb_by_idx: list[float],
    v_mb_by_idx: list[float],
) -> tuple[int, int, int, float, float]:
    ki = 0
    vi = 0
    steps = 0
    while steps < (len(k_budgets) + len(v_budgets) + 4):
        cur = outputs[(ki, vi)]
        k_can = ki + 1 < len(k_budgets)
        v_can = vi + 1 < len(v_budgets)
        k_delta = _rel_l2_np(cur, outputs[(ki + 1, vi)]) if k_can else 0.0
        v_delta = _rel_l2_np(cur, outputs[(ki, vi + 1)]) if v_can else 0.0
        extra_k_mb = float(k_mb_by_idx[ki + 1] - k_mb_by_idx[ki]) if k_can else float("inf")
        extra_v_mb = float(v_mb_by_idx[vi + 1] - v_mb_by_idx[vi]) if v_can else float("inf")
        action = _choose_joint_kv_action(
            policy=str(policy),
            k_delta=float(k_delta),
            v_delta=float(v_delta),
            k_can=bool(k_can),
            v_can=bool(v_can),
            threshold=float(threshold),
            turn=int(steps),
            extra_k_mb=float(extra_k_mb),
            extra_v_mb=float(extra_v_mb),
        )
        if action == "stop":
            return ki, vi, steps, float(k_delta), float(v_delta)
        if action == "k":
            ki += 1
        elif action == "v":
            vi += 1
        else:
            raise AssertionError(action)
        steps += 1
    return ki, vi, steps, 0.0, 0.0


def _simulate_joint_kv_policy_torch(
    *,
    outputs: dict[tuple[int, int], torch.Tensor],
    k_budgets: list[int],
    v_budgets: list[int],
    policy: str,
    threshold: float,
    k_mb_by_idx: list[float],
    v_mb_by_idx: list[float],
) -> tuple[int, int, int, float, float]:
    ki = 0
    vi = 0
    steps = 0
    while steps < (len(k_budgets) + len(v_budgets) + 4):
        cur = outputs[(ki, vi)]
        k_can = ki + 1 < len(k_budgets)
        v_can = vi + 1 < len(v_budgets)
        k_delta = _rel_l2_torch(cur, outputs[(ki + 1, vi)]) if k_can else 0.0
        v_delta = _rel_l2_torch(cur, outputs[(ki, vi + 1)]) if v_can else 0.0
        extra_k_mb = float(k_mb_by_idx[ki + 1] - k_mb_by_idx[ki]) if k_can else float("inf")
        extra_v_mb = float(v_mb_by_idx[vi + 1] - v_mb_by_idx[vi]) if v_can else float("inf")
        action = _choose_joint_kv_action(
            policy=str(policy),
            k_delta=float(k_delta),
            v_delta=float(v_delta),
            k_can=bool(k_can),
            v_can=bool(v_can),
            threshold=float(threshold),
            turn=int(steps),
            extra_k_mb=float(extra_k_mb),
            extra_v_mb=float(extra_v_mb),
        )
        if action == "stop":
            return ki, vi, steps, float(k_delta), float(v_delta)
        if action == "k":
            ki += 1
        elif action == "v":
            vi += 1
        else:
            raise AssertionError(action)
        steps += 1
    return ki, vi, steps, 0.0, 0.0


def _require_canonical_gpu_frontier(args) -> None:
    if not _env_truthy("FRONTIER_CANONICAL_GPU", "0"):
        return
    mismatches = _canonical_gpu_frontier_mismatches(args)
    if mismatches:
        raise ValueError(
            "FRONTIER_CANONICAL_GPU=1 requires the CPU-frontier-matching GPU path; "
            + "; ".join(mismatches)
        )


def reset_paged_pq_attention_state(model) -> None:
    """Clear data-dependent paged-PQ caches between independent sequences.

    The page/PQ and packed-GQA caches are valid within one prefill+decode stream,
    where decode appends to the same KV history. They are not valid across
    independent benchmark samples that reuse the same model object.
    """

    for module in model.modules():
        for attr in (
            "_pagedpq_page_cache",
            "_pagedpq_gqa_native_pack_cache",
            "_pagedpq_gqa_native_pack_fast_cache",
            "_pagedpq_gqa_value_vpq_pack_cache",
            "_pagedpq_gqa_value_vpq_pack_fast_cache",
            "_pagedpq_fast_decode_index_cache",
        ):
            if hasattr(module, attr):
                delattr(module, attr)


def cache_sequence_length(cache_obj, layer_idx: int) -> int | None:
    if cache_obj is None:
        return None
    getter = getattr(cache_obj, "get_seq_length", None)
    if callable(getter):
        for call_args in ((int(layer_idx),), ()):
            try:
                value = getter(*call_args)
            except TypeError:
                continue
            except Exception:
                break
            if value is not None:
                try:
                    return int(value)
                except Exception:
                    pass
    key_cache = getattr(cache_obj, "key_cache", None)
    if isinstance(key_cache, (list, tuple)) and int(layer_idx) < len(key_cache):
        layer_key = key_cache[int(layer_idx)]
        if isinstance(layer_key, torch.Tensor) and layer_key.ndim >= 2:
            return int(layer_key.shape[-2])
    layers = getattr(cache_obj, "layers", None)
    if isinstance(layers, (list, tuple)) and int(layer_idx) < len(layers):
        layer = layers[int(layer_idx)]
        for attr in ("keys", "key", "k_cache"):
            layer_key = getattr(layer, attr, None)
            if isinstance(layer_key, torch.Tensor) and layer_key.ndim >= 2:
                return int(layer_key.shape[-2])
    return None


def cache_layer_kv_tensors(
    cache_obj,
    layer_idx: int,
    *,
    num_kv_heads: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return layer KV tensors as [kv_heads, seq, head_dim] when HF cache exposes them."""

    def normalize(tensor: torch.Tensor) -> torch.Tensor | None:
        if not isinstance(tensor, torch.Tensor):
            return None
        if tensor.ndim == 4:
            # HF DynamicCache convention is [batch, kv_heads, seq, head_dim].
            if int(tensor.shape[1]) == int(num_kv_heads):
                return tensor[0]
            if int(tensor.shape[2]) == int(num_kv_heads):
                return tensor[0].transpose(0, 1)
        if tensor.ndim == 3:
            if int(tensor.shape[0]) == int(num_kv_heads):
                return tensor
            if int(tensor.shape[1]) == int(num_kv_heads):
                return tensor.transpose(0, 1)
        return None

    key_cache = getattr(cache_obj, "key_cache", None)
    value_cache = getattr(cache_obj, "value_cache", None)
    if (
        isinstance(key_cache, (list, tuple))
        and isinstance(value_cache, (list, tuple))
        and int(layer_idx) < len(key_cache)
        and int(layer_idx) < len(value_cache)
    ):
        keys = normalize(key_cache[int(layer_idx)])
        values = normalize(value_cache[int(layer_idx)])
        if keys is not None and values is not None:
            return keys, values

    layers = getattr(cache_obj, "layers", None)
    if isinstance(layers, (list, tuple)) and int(layer_idx) < len(layers):
        layer = layers[int(layer_idx)]
        layer_key = None
        layer_value = None
        for attr in ("keys", "key", "k_cache"):
            candidate = getattr(layer, attr, None)
            if isinstance(candidate, torch.Tensor):
                layer_key = candidate
                break
        for attr in ("values", "value", "v_cache"):
            candidate = getattr(layer, attr, None)
            if isinstance(candidate, torch.Tensor):
                layer_value = candidate
                break
        keys = normalize(layer_key)
        values = normalize(layer_value)
        if keys is not None and values is not None:
            return keys, values
    return None


def tensor_read_mb(tensor: torch.Tensor, elem_bytes: int) -> float:
    return float(tensor.numel() * int(elem_bytes)) / MB


def build_page_pq_from_keys(
    keys: torch.Tensor,
    *,
    args,
    kv_head: int,
    dynamic_start: int,
    indexed_end: int,
    key_bytes: int,
    router_enabled: bool,
    device: torch.device,
    page_id_offset: int = 0,
) -> tuple[GPUIndex, float, float, float]:
    backend = str(getattr(args, "index_build_backend", "numpy"))
    seed = int(args.seed) + 2027 * int(kv_head)
    if backend == "torch_gpu":
        index = build_page_pq_torch(
            keys,
            dynamic_start=dynamic_start,
            indexed_end=indexed_end,
            page_size=int(args.page_size),
            subvecs=int(args.subvecs),
            subbits=int(args.subbits),
            kmeans_iters=int(args.kmeans_iters),
            seed=seed,
            key_bytes=key_bytes,
            router_enabled=router_enabled,
            router_prototypes=int(args.router_prototypes),
            router_merge_rel=float(args.router_merge_rel),
            router_merge_var=float(args.router_merge_var),
            router_max_groups=int(args.router_max_groups),
            device=device,
            page_id_offset=page_id_offset,
        )
        return index, float(index.build_seconds), float(index.build_read_mb), float(index.build_write_mb)

    if backend != "numpy":
        raise ValueError(f"unknown index_build_backend={backend!r}")
    index_t0 = time.perf_counter()
    transfer_read_mb = tensor_read_mb(keys, key_bytes)
    keys_np = keys.detach().to(torch.float32).cpu().numpy()
    index = build_page_pq_gpu(
        keys_np,
        dynamic_start=dynamic_start,
        indexed_end=indexed_end,
        page_size=int(args.page_size),
        subvecs=int(args.subvecs),
        subbits=int(args.subbits),
        kmeans_iters=int(args.kmeans_iters),
        seed=seed,
        key_bytes=key_bytes,
        router_enabled=router_enabled,
        router_prototypes=int(args.router_prototypes),
        router_merge_rel=float(args.router_merge_rel),
        router_merge_var=float(args.router_merge_var),
        router_max_groups=int(args.router_max_groups),
        device=device,
        page_id_offset=page_id_offset,
    )
    _sync_if_cuda(device)
    return (
        index,
        float(time.perf_counter() - index_t0),
        float(transfer_read_mb + index.build_read_mb),
        float(index.build_write_mb),
    )


@dataclass
class ApproxStats:
    calls: int = 0
    approx_attention_calls: int = 0
    passthrough_attention_calls: int = 0
    mean_selected: float = 0.0
    mean_tail_samples: float = 0.0
    mean_selector_mb: float = 0.0
    mean_exact_kv_mb: float = 0.0
    mean_tail_mb: float = 0.0
    mean_confidence_mb: float = 0.0
    mean_step_mb: float = 0.0
    mean_physical_gpu_exact_kv_mb: float = 0.0
    mean_physical_gpu_confidence_mb: float = 0.0
    mean_physical_gpu_step_mb: float = 0.0
    selector_active_calls: int = 0
    tail_active_calls: int = 0
    confidence_active_calls: int = 0
    index_build_calls: int = 0
    index_build_seconds: float = 0.0
    index_build_read_mb: float = 0.0
    index_build_write_mb: float = 0.0
    cache_cast_seconds: float = 0.0
    patched_attention_seconds: float = 0.0
    qkv_cache_seconds: float = 0.0
    index_sidecar_seconds: float = 0.0
    native_pack_seconds: float = 0.0
    native_selector_seconds: float = 0.0
    native_attention_seconds: float = 0.0
    native_exact_logit_seconds: float = 0.0
    native_threshold_seconds: float = 0.0
    native_geometric_seconds: float = 0.0
    native_output_seconds: float = 0.0
    native_joint_rank_prefix_seconds: float = 0.0
    native_joint_score_grid_seconds: float = 0.0
    native_joint_prob_base_seconds: float = 0.0
    native_joint_risk_prefix_seconds: float = 0.0
    native_joint_policy_seconds: float = 0.0
    native_joint_precompute_seconds: float = 0.0
    native_joint_layout_seconds: float = 0.0
    native_joint_group_pack_seconds: float = 0.0
    native_joint_accounting_seconds: float = 0.0
    output_projection_seconds: float = 0.0
    wall_patched_attention_seconds: float = 0.0
    wall_qkv_cache_seconds: float = 0.0
    wall_index_sidecar_seconds: float = 0.0
    wall_output_projection_seconds: float = 0.0
    wall_joint_total_seconds: float = 0.0
    wall_joint_precompute_seconds: float = 0.0
    wall_joint_selector_seconds: float = 0.0
    wall_joint_exact_logit_seconds: float = 0.0
    wall_joint_vpq_sidecar_seconds: float = 0.0
    wall_joint_layout_seconds: float = 0.0
    wall_joint_rank_prefix_seconds: float = 0.0
    wall_joint_score_grid_seconds: float = 0.0
    wall_joint_prob_base_seconds: float = 0.0
    wall_joint_risk_prefix_seconds: float = 0.0
    wall_joint_policy_seconds: float = 0.0
    wall_joint_group_pack_seconds: float = 0.0
    wall_joint_accounting_seconds: float = 0.0

    def add_count(
        self,
        selected_count: int,
        tail_count: int,
        selector_mb: float,
        head_dim: int,
        key_bytes: int,
        value_bytes: int,
        tail_mb_override: float | None = None,
        exact_kv_mb_override: float | None = None,
        confidence_mb_override: float = 0.0,
        physical_gpu_exact_kv_mb_override: float | None = None,
        physical_gpu_confidence_mb_override: float | None = None,
    ) -> None:
        self.add_count_repeated(
            1,
            selected_count,
            tail_count,
            selector_mb,
            head_dim,
            key_bytes,
            value_bytes,
            tail_mb_override=tail_mb_override,
            exact_kv_mb_override=exact_kv_mb_override,
            confidence_mb_override=confidence_mb_override,
            physical_gpu_exact_kv_mb_override=physical_gpu_exact_kv_mb_override,
            physical_gpu_confidence_mb_override=physical_gpu_confidence_mb_override,
        )

    def add_count_repeated(
        self,
        repeats: int,
        selected_count: int,
        tail_count: int,
        selector_mb: float,
        head_dim: int,
        key_bytes: int,
        value_bytes: int,
        tail_mb_override: float | None = None,
        exact_kv_mb_override: float | None = None,
        confidence_mb_override: float = 0.0,
        physical_gpu_exact_kv_mb_override: float | None = None,
        physical_gpu_confidence_mb_override: float | None = None,
    ) -> None:
        repeats = int(repeats)
        if repeats <= 0:
            return
        exact_kv_mb = (
            float(exact_kv_mb_override)
            if exact_kv_mb_override is not None
            else float(int(selected_count) * head_dim * (key_bytes + value_bytes)) / MB
        )
        tail_mb = (
            float(tail_mb_override)
            if tail_mb_override is not None
            else float(tail_count * head_dim * (key_bytes + value_bytes)) / MB
        )
        confidence_mb = float(confidence_mb_override)
        step_mb = float(selector_mb) + exact_kv_mb + tail_mb + confidence_mb
        physical_gpu_exact_kv_mb = (
            float(physical_gpu_exact_kv_mb_override)
            if physical_gpu_exact_kv_mb_override is not None
            else exact_kv_mb
        )
        physical_gpu_confidence_mb = (
            float(physical_gpu_confidence_mb_override)
            if physical_gpu_confidence_mb_override is not None
            else confidence_mb
        )
        physical_gpu_step_mb = float(selector_mb) + physical_gpu_exact_kv_mb + tail_mb + physical_gpu_confidence_mb
        next_calls = self.calls + repeats
        alpha = float(repeats) / float(next_calls)
        self.mean_selected += alpha * (float(selected_count) - self.mean_selected)
        self.mean_tail_samples += alpha * (float(tail_count) - self.mean_tail_samples)
        self.mean_selector_mb += alpha * (float(selector_mb) - self.mean_selector_mb)
        self.mean_exact_kv_mb += alpha * (exact_kv_mb - self.mean_exact_kv_mb)
        self.mean_tail_mb += alpha * (tail_mb - self.mean_tail_mb)
        self.mean_confidence_mb += alpha * (confidence_mb - self.mean_confidence_mb)
        self.mean_step_mb += alpha * (step_mb - self.mean_step_mb)
        self.mean_physical_gpu_exact_kv_mb += alpha * (
            physical_gpu_exact_kv_mb - self.mean_physical_gpu_exact_kv_mb
        )
        self.mean_physical_gpu_confidence_mb += alpha * (
            physical_gpu_confidence_mb - self.mean_physical_gpu_confidence_mb
        )
        self.mean_physical_gpu_step_mb += alpha * (physical_gpu_step_mb - self.mean_physical_gpu_step_mb)
        if float(selector_mb) > 0.0:
            self.selector_active_calls += repeats
        if float(tail_mb) > 0.0:
            self.tail_active_calls += repeats
        if float(confidence_mb) > 0.0:
            self.confidence_active_calls += repeats
        self.calls = next_calls

    def add(
        self,
        selected: list[int],
        tail_count: int,
        selector_mb: float,
        head_dim: int,
        key_bytes: int,
        value_bytes: int,
        tail_mb_override: float | None = None,
        exact_kv_mb_override: float | None = None,
        confidence_mb_override: float = 0.0,
        physical_gpu_exact_kv_mb_override: float | None = None,
        physical_gpu_confidence_mb_override: float | None = None,
    ) -> None:
        self.add_count(
            len(selected),
            tail_count,
            selector_mb,
            head_dim,
            key_bytes,
            value_bytes,
            tail_mb_override=tail_mb_override,
            exact_kv_mb_override=exact_kv_mb_override,
            confidence_mb_override=confidence_mb_override,
            physical_gpu_exact_kv_mb_override=physical_gpu_exact_kv_mb_override,
            physical_gpu_confidence_mb_override=physical_gpu_confidence_mb_override,
        )

    def add_approx_attention_call(self) -> None:
        self.approx_attention_calls += 1

    def add_passthrough_attention_call(self) -> None:
        self.passthrough_attention_calls += 1

    def add_index_build(self, seconds: float, read_mb: float, write_mb: float) -> None:
        self.index_build_calls += 1
        self.index_build_seconds += float(seconds)
        self.index_build_read_mb += float(read_mb)
        self.index_build_write_mb += float(write_mb)

    def add_cache_cast_timing(self, seconds: float) -> None:
        self.cache_cast_seconds += float(seconds)

    def add_patched_attention_timing(self, seconds: float) -> None:
        self.patched_attention_seconds += float(seconds)

    def add_qkv_cache_timing(self, seconds: float) -> None:
        self.qkv_cache_seconds += float(seconds)

    def add_index_sidecar_timing(self, seconds: float) -> None:
        self.index_sidecar_seconds += float(seconds)

    def add_native_pack_timing(self, seconds: float) -> None:
        self.native_pack_seconds += float(seconds)

    def add_native_timing(self, selector_seconds: float = 0.0, attention_seconds: float = 0.0) -> None:
        self.native_selector_seconds += float(selector_seconds)
        self.native_attention_seconds += float(attention_seconds)

    def add_native_detail_timing(
        self,
        *,
        exact_logit_seconds: float = 0.0,
        threshold_seconds: float = 0.0,
        geometric_seconds: float = 0.0,
        output_seconds: float = 0.0,
    ) -> None:
        self.native_exact_logit_seconds += float(exact_logit_seconds)
        self.native_threshold_seconds += float(threshold_seconds)
        self.native_geometric_seconds += float(geometric_seconds)
        self.native_output_seconds += float(output_seconds)

    def add_joint_detail_timing(
        self,
        *,
        rank_prefix_seconds: float = 0.0,
        score_grid_seconds: float = 0.0,
        prob_base_seconds: float = 0.0,
        risk_prefix_seconds: float = 0.0,
        policy_seconds: float = 0.0,
        precompute_seconds: float = 0.0,
        layout_seconds: float = 0.0,
        group_pack_seconds: float = 0.0,
        accounting_seconds: float = 0.0,
    ) -> None:
        self.native_joint_rank_prefix_seconds += float(rank_prefix_seconds)
        self.native_joint_score_grid_seconds += float(score_grid_seconds)
        self.native_joint_prob_base_seconds += float(prob_base_seconds)
        self.native_joint_risk_prefix_seconds += float(risk_prefix_seconds)
        self.native_joint_policy_seconds += float(policy_seconds)
        self.native_joint_precompute_seconds += float(precompute_seconds)
        self.native_joint_layout_seconds += float(layout_seconds)
        self.native_joint_group_pack_seconds += float(group_pack_seconds)
        self.native_joint_accounting_seconds += float(accounting_seconds)

    def add_output_projection_timing(self, seconds: float) -> None:
        self.output_projection_seconds += float(seconds)

    def add_wall_patched_attention_timing(self, seconds: float) -> None:
        self.wall_patched_attention_seconds += float(seconds)

    def add_wall_qkv_cache_timing(self, seconds: float) -> None:
        self.wall_qkv_cache_seconds += float(seconds)

    def add_wall_index_sidecar_timing(self, seconds: float) -> None:
        self.wall_index_sidecar_seconds += float(seconds)

    def add_wall_output_projection_timing(self, seconds: float) -> None:
        self.wall_output_projection_seconds += float(seconds)

    def add_joint_wall_timing(
        self,
        *,
        total_seconds: float = 0.0,
        precompute_seconds: float = 0.0,
        selector_seconds: float = 0.0,
        exact_logit_seconds: float = 0.0,
        vpq_sidecar_seconds: float = 0.0,
        layout_seconds: float = 0.0,
        rank_prefix_seconds: float = 0.0,
        score_grid_seconds: float = 0.0,
        prob_base_seconds: float = 0.0,
        risk_prefix_seconds: float = 0.0,
        policy_seconds: float = 0.0,
        group_pack_seconds: float = 0.0,
        accounting_seconds: float = 0.0,
    ) -> None:
        self.wall_joint_total_seconds += float(total_seconds)
        self.wall_joint_precompute_seconds += float(precompute_seconds)
        self.wall_joint_selector_seconds += float(selector_seconds)
        self.wall_joint_exact_logit_seconds += float(exact_logit_seconds)
        self.wall_joint_vpq_sidecar_seconds += float(vpq_sidecar_seconds)
        self.wall_joint_layout_seconds += float(layout_seconds)
        self.wall_joint_rank_prefix_seconds += float(rank_prefix_seconds)
        self.wall_joint_score_grid_seconds += float(score_grid_seconds)
        self.wall_joint_prob_base_seconds += float(prob_base_seconds)
        self.wall_joint_risk_prefix_seconds += float(risk_prefix_seconds)
        self.wall_joint_policy_seconds += float(policy_seconds)
        self.wall_joint_group_pack_seconds += float(group_pack_seconds)
        self.wall_joint_accounting_seconds += float(accounting_seconds)


def parse_head_budget_map(text: str) -> dict[int, int]:
    out: dict[int, int] = {}
    for part in str(text or "").split(","):
        part = part.strip()
        if not part:
            continue
        head, budget = part.split(":", 1)
        out[int(head)] = int(budget)
    return out


def parse_int_set(text: str) -> set[int]:
    return {int(part.strip()) for part in str(text or "").split(",") if part.strip()}


def output_metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    return _output_error_metrics(
        a.detach().float().cpu().numpy().reshape(-1),
        b.detach().float().cpu().numpy().reshape(-1),
    )


def selected_value_exact_mask(
    *,
    selected_scores: np.ndarray,
    values_exact: np.ndarray,
    values_approx: np.ndarray | None,
    rule: str,
    exact_top: int,
    exact_mass: float,
    exact_risk_mass: float,
    min_top: int,
    max_top: int,
) -> tuple[np.ndarray, float]:
    count = int(selected_scores.shape[0])
    mask = np.zeros((count,), dtype=bool)
    if count <= 0:
        return mask, 0.0
    scores = selected_scores.astype(np.float64, copy=False)
    if str(rule) == "fixed":
        order = np.argsort(-scores, kind="stable")
        exact_count = int(exact_top)
        if exact_count > 0:
            mask[order[: min(count, exact_count)]] = True
        return mask, 0.0
    elif str(rule) == "selector_rank":
        exact_count = int(exact_top)
        if exact_count > 0:
            mask[: min(count, exact_count)] = True
        return mask, 0.0

    shifted = scores - float(scores.max())
    probs = np.exp(shifted)
    probs /= max(float(probs.sum()), 1e-20)
    if str(rule) == "selected_mass":
        order = np.argsort(-scores, kind="stable")
        target = float(max(0.0, min(1.0, exact_mass)))
        cumulative = np.cumsum(probs[order])
        exact_count = int(np.searchsorted(cumulative, target, side="left") + 1) if target > 0.0 else 0
        if exact_count > 0:
            mask[order[: min(count, exact_count)]] = True
    elif str(rule) in {"selected_risk_mass", "selected_mass_or_risk"}:
        if values_approx is None:
            raise ValueError(f"{rule} requires approximate selected values")
        residual_norm = np.linalg.norm(
            values_exact.astype(np.float32, copy=False) - values_approx.astype(np.float32, copy=False),
            axis=1,
        ) / float(np.sqrt(float(values_exact.shape[-1])))
        risk = probs * residual_norm.astype(np.float64, copy=False)
        risk_order = np.argsort(-risk, kind="stable")
        target = float(exact_risk_mass) if float(exact_risk_mass) > 0.0 else float(exact_mass)
        total_risk = float(risk.sum())
        if total_risk > 1e-20 and target > 0.0:
            cumulative = np.cumsum(risk[risk_order]) / total_risk
            exact_count = int(np.searchsorted(cumulative, float(max(0.0, min(1.0, target))), side="left") + 1)
        else:
            exact_count = int(exact_top)
        if exact_count > 0:
            mask[risk_order[: min(count, exact_count)]] = True
        if str(rule) == "selected_mass_or_risk":
            prob_order = np.argsort(-scores, kind="stable")
            mass_target = float(max(0.0, min(1.0, exact_mass)))
            if mass_target > 0.0:
                cumulative = np.cumsum(probs[prob_order])
                mass_count = int(np.searchsorted(cumulative, mass_target, side="left") + 1)
                mask[prob_order[: min(count, mass_count)]] = True
    else:
        raise ValueError(f"unknown selected_value_exact_rule: {rule}")
    if int(min_top) > 0 and int(np.sum(mask)) < int(min_top):
        order = np.argsort(-scores, kind="stable")
        mask[order[: min(count, int(min_top))]] = True
    if int(max_top) > 0 and int(np.sum(mask)) > int(max_top):
        order = np.argsort(-(probs * (1.0 + np.arange(count, 0, -1) / max(1, count))), kind="stable")
        limited = np.zeros((count,), dtype=bool)
        limited[order[: min(count, int(max_top))]] = True
        mask = limited
    return mask, float(probs[mask].sum()) if bool(np.any(mask)) else 0.0


def selected_value_exact_top_positive(args: Any) -> int:
    return max(0, int(args.selected_value_exact_top))


def native_selected_value_exact_top_arg(args: Any) -> int:
    exact_top = selected_value_exact_top_positive(args)
    if str(args.selected_value_exact_rule) == "selector_rank" and exact_top > 0:
        return -exact_top
    return exact_top


def value_vpq_pack_gpu(
    *,
    index: GPUIndex,
    values_np: np.ndarray,
    subbits: int,
    value_subvecs: int,
    value_subbits: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int] | None:
    if not index.pages:
        return None
    page_size = int(index.pages[0].size)
    actual_value_subbits = int(value_subbits) if int(value_subbits) > 0 else int(subbits)
    cache_key = (str(device), int(value_subvecs), int(actual_value_subbits))
    cached = getattr(index, "_value_vpq_gpu_pack_by_params", None)
    if isinstance(cached, dict) and cache_key in cached:
        return cached[cache_key]
    if any(int(page.size) != page_size for page in index.pages):
        return None
    sidecars = _build_value_vpq_sidecars(
        index,
        values_np,
        int(subbits),
        value_subvecs=int(value_subvecs),
        value_subbits=int(actual_value_subbits),
    )
    if not sidecars or any(codebook.size == 0 or codes.size == 0 for codebook, codes in sidecars):
        return None
    codebooks_np = np.stack([codebook.astype(np.float32, copy=False) for codebook, _codes in sidecars], axis=0)
    codes_np = np.stack([codes.astype(np.int64, copy=False) for _codebook, codes in sidecars], axis=0)
    codebooks = torch.as_tensor(codebooks_np, dtype=torch.float32, device=device)
    codes_dtype = torch.uint8 if int(actual_value_subbits) <= 8 else torch.long
    codes = torch.as_tensor(codes_np, dtype=codes_dtype, device=device)
    page_starts = torch.as_tensor([int(page.start) for page in index.pages], dtype=torch.long, device=device)
    packed = (codebooks, codes, page_starts, int(page_size), int(actual_value_subbits))
    if not isinstance(cached, dict):
        cached = {}
    cached[cache_key] = packed
    setattr(index, "_value_vpq_gpu_pack_by_params", cached)
    return packed


def value_vpq_pack_torch(
    *,
    index: GPUIndex,
    values: torch.Tensor,
    value_subvecs: int,
    value_subbits: int,
    key_bytes: int,
    device: torch.device,
    value_group_pages: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int] | None:
    if not index.pages:
        return None
    selection_page_size = int(index.pages[0].size)
    if selection_page_size <= 0:
        return None
    group_pages = max(1, int(value_group_pages))
    page_size = int(selection_page_size * group_pages)
    actual_value_subvecs = int(value_subvecs) if int(value_subvecs) > 0 else int(index.pages[0].codes.shape[1])
    actual_value_subbits = int(value_subbits) if int(value_subbits) > 0 else 8
    cache_key = (str(device), int(actual_value_subvecs), int(actual_value_subbits), int(group_pages), "torch")
    cached = getattr(index, "_value_vpq_gpu_pack_by_params", None)
    if isinstance(cached, dict) and cache_key in cached:
        setattr(index, "_last_value_vpq_build_stats", None)
        return cached[cache_key]
    if any(int(page.size) != selection_page_size for page in index.pages):
        return None
    dynamic_start = int(index.pages[0].start)
    indexed_end = int(index.pages[-1].start) + int(index.pages[-1].size)
    v_index = build_page_pq_torch(
        values,
        dynamic_start=dynamic_start,
        indexed_end=indexed_end,
        page_size=page_size,
        subvecs=int(actual_value_subvecs),
        subbits=int(actual_value_subbits),
        kmeans_iters=3,
        seed=90210 + int(dynamic_start) + 1000003 * int(group_pages),
        key_bytes=int(key_bytes),
        router_enabled=False,
        router_prototypes=0,
        router_merge_rel=0.0,
        router_merge_var=0.0,
        router_max_groups=0,
        device=device,
    )
    if not v_index.pages:
        setattr(index, "_last_value_vpq_build_stats", None)
        return None
    codebooks, codes, page_starts = ensure_native_fullscan_pack(v_index, subbits=int(actual_value_subbits))
    packed = (codebooks, codes, page_starts, int(page_size), int(actual_value_subbits))
    setattr(
        index,
        "_last_value_vpq_build_stats",
        (
            float(v_index.build_seconds),
            float(v_index.build_read_mb),
            float(v_index.build_write_mb),
        ),
    )
    if not isinstance(cached, dict):
        cached = {}
    cached[cache_key] = packed
    setattr(index, "_value_vpq_gpu_pack_by_params", cached)
    return packed


def vpq_values_for_tokens_gpu(
    *,
    index: GPUIndex,
    values: torch.Tensor,
    values_np: np.ndarray | None,
    tokens: torch.Tensor,
    subbits: int,
    value_subvecs: int,
    value_subbits: int,
    prefer_torch: bool = False,
    value_bytes: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    if bool(prefer_torch):
        pack = value_vpq_pack_torch(
            index=index,
            values=values,
            value_subvecs=int(value_subvecs),
            value_subbits=int(value_subbits) if int(value_subbits) > 0 else int(subbits),
            key_bytes=int(value_bytes),
            device=values.device,
        )
    else:
        if values_np is None:
            raise ValueError("values_np is required for CPU-built V-PQ sidecars")
        pack = value_vpq_pack_gpu(
            index=index,
            values_np=values_np,
            subbits=int(subbits),
            value_subvecs=int(value_subvecs),
            value_subbits=int(value_subbits),
            device=values.device,
        )
    if pack is None or tokens.numel() == 0:
        exact_values = values.index_select(0, tokens.reshape(-1)).reshape(*tokens.shape, int(values.shape[-1])).float()
        return exact_values, torch.zeros_like(tokens, dtype=torch.bool), torch.full_like(tokens, -1), int(value_subbits)
    codebooks, codes, page_starts, page_size, actual_value_subbits = pack
    first_start = int(page_starts[0].item())
    page_ids = torch.div(tokens - first_start, int(page_size), rounding_mode="floor")
    in_range = (tokens >= first_start) & (page_ids >= 0) & (page_ids < int(page_starts.numel()))
    clamped_page_ids = torch.clamp(page_ids, min=0, max=max(0, int(page_starts.numel()) - 1))
    rows = tokens - page_starts.index_select(0, clamped_page_ids.reshape(-1)).reshape_as(tokens)
    valid = in_range & (rows >= 0) & (rows < int(page_size))
    if not bool(torch.any(valid)):
        exact_values = values.index_select(0, tokens.reshape(-1)).reshape(*tokens.shape, int(values.shape[-1])).float()
        return exact_values, valid, page_ids, int(actual_value_subbits)
    flat_valid = valid.reshape(-1)
    flat_page_ids = clamped_page_ids.reshape(-1).index_select(0, torch.nonzero(flat_valid, as_tuple=False).reshape(-1))
    flat_rows = rows.reshape(-1).index_select(0, torch.nonzero(flat_valid, as_tuple=False).reshape(-1)).to(torch.long)
    selected_codes = codes[flat_page_ids, flat_rows].to(torch.long)
    subvecs = int(codebooks.shape[1])
    subdim = int(codebooks.shape[-1])
    approx_flat = torch.empty((int(selected_codes.shape[0]), subvecs * subdim), dtype=torch.float32, device=values.device)
    sub_ids = torch.arange(subvecs, dtype=torch.long, device=values.device)
    for sub in range(subvecs):
        approx_flat[:, sub * subdim : (sub + 1) * subdim] = codebooks[
            flat_page_ids,
            sub_ids[sub].expand_as(flat_page_ids),
            selected_codes[:, sub],
        ]
    out = torch.empty((int(tokens.numel()), int(values.shape[-1])), dtype=torch.float32, device=values.device)
    out[flat_valid] = approx_flat
    if int(flat_valid.numel()) != int(approx_flat.shape[0]):
        invalid_flat = ~flat_valid
        if bool(torch.any(invalid_flat)):
            invalid_tokens = tokens.reshape(-1).index_select(
                0,
                torch.nonzero(invalid_flat, as_tuple=False).reshape(-1),
            )
            out[invalid_flat] = values.index_select(0, invalid_tokens).float()
    return out.reshape(*tokens.shape, int(values.shape[-1])), valid, page_ids, int(actual_value_subbits)


def reconstruct_all_vpq_values_gpu(
    *,
    index: GPUIndex,
    values_np: np.ndarray | None,
    values: torch.Tensor | None = None,
    subbits: int,
    value_subvecs: int,
    value_subbits: int,
    device: torch.device,
    prefer_torch: bool = False,
    value_bytes: int = 2,
) -> tuple[torch.Tensor, int] | None:
    actual_value_subbits_for_key = int(value_subbits) if int(value_subbits) > 0 else int(subbits)
    cache_key = (str(device), int(value_subvecs), int(actual_value_subbits_for_key), "torch" if bool(prefer_torch) else "numpy")
    cached = getattr(index, "_all_value_vpq_gpu_by_params", None)
    if isinstance(cached, dict) and cache_key in cached:
        return cached[cache_key]
    if bool(prefer_torch):
        if values is None:
            raise ValueError("values tensor is required for torch-built V-PQ sidecars")
        pack = value_vpq_pack_torch(
            index=index,
            values=values,
            value_subvecs=int(value_subvecs),
            value_subbits=int(value_subbits) if int(value_subbits) > 0 else int(subbits),
            key_bytes=int(value_bytes),
            device=device,
        )
    else:
        if values_np is None:
            raise ValueError("values_np is required for CPU-built V-PQ sidecars")
        pack = value_vpq_pack_gpu(
            index=index,
            values_np=values_np,
            subbits=int(subbits),
            value_subvecs=int(value_subvecs),
            value_subbits=int(value_subbits),
            device=device,
        )
    if pack is None:
        return None
    codebooks, codes, _page_starts, _page_size, actual_value_subbits = pack
    pages = int(codebooks.shape[0])
    page_size = int(codes.shape[1])
    subvecs = int(codebooks.shape[1])
    subdim = int(codebooks.shape[-1])
    flat_codes = codes.reshape(pages * page_size, subvecs).to(torch.long)
    page_ids = torch.arange(pages, dtype=torch.long, device=device).repeat_interleave(page_size)
    out = torch.empty((pages * page_size, subvecs * subdim), dtype=torch.float32, device=device)
    for sub in range(subvecs):
        out[:, sub * subdim : (sub + 1) * subdim] = codebooks[
            page_ids,
            torch.full_like(page_ids, int(sub)),
            flat_codes[:, sub],
        ]
    if not isinstance(cached, dict):
        cached = {}
    packed = (out, int(actual_value_subbits))
    cached[cache_key] = packed
    setattr(index, "_all_value_vpq_gpu_by_params", cached)
    return packed


def value_vpq_code_stat_risk_torch(
    *,
    index: GPUIndex,
    values: torch.Tensor,
    vhat_all: torch.Tensor,
    residual_all: torch.Tensor | None = None,
    valid: torch.Tensor,
    page_ids: torch.Tensor,
    subbits: int,
    value_subvecs: int,
    value_subbits: int,
    value_bytes: int,
) -> tuple[torch.Tensor, int]:
    """Per-token deployable V-PQ residual-risk sidecar using torch-built V-PQ.

    This mirrors the CPU reference's page/code mean residual statistic without
    invoking the CPU NumPy k-means sidecar path during HF benchmark decode.
    Invalid/non-indexed tokens use exact V fallback in `vhat_all`, so their
    residual risk is zero.
    """

    out = torch.zeros((int(values.shape[0]),), dtype=torch.float64, device=values.device)
    pack = value_vpq_pack_torch(
        index=index,
        values=values,
        value_subvecs=int(value_subvecs),
        value_subbits=int(value_subbits) if int(value_subbits) > 0 else int(subbits),
        key_bytes=int(value_bytes),
        device=values.device,
    )
    if pack is None or values.numel() == 0 or not bool(torch.any(valid)):
        actual_value_subbits = int(value_subbits) if int(value_subbits) > 0 else int(subbits)
        return out, int(actual_value_subbits)
    codebooks, codes, page_starts, page_size, actual_value_subbits = pack
    tokens = torch.arange(int(values.shape[0]), dtype=torch.long, device=values.device)
    valid_flat = valid.reshape(-1)
    valid_idx = torch.nonzero(valid_flat, as_tuple=False).reshape(-1)
    valid_pages = page_ids.reshape(-1).index_select(0, valid_idx).to(torch.long)
    valid_pages = torch.clamp(valid_pages, min=0, max=max(0, int(page_starts.numel()) - 1))
    rows = tokens.index_select(0, valid_idx) - page_starts.index_select(0, valid_pages)
    row_mask = (rows >= 0) & (rows < int(page_size))
    if not bool(torch.any(row_mask)):
        return out, int(actual_value_subbits)
    valid_idx = valid_idx.index_select(0, torch.nonzero(row_mask, as_tuple=False).reshape(-1))
    valid_pages = valid_pages.index_select(0, torch.nonzero(row_mask, as_tuple=False).reshape(-1))
    rows = rows.index_select(0, torch.nonzero(row_mask, as_tuple=False).reshape(-1)).to(torch.long)
    selected_codes = codes[valid_pages, rows].to(torch.long)
    if residual_all is None:
        residual_valid = (values.float() - vhat_all.float()).index_select(0, valid_idx).to(torch.float64)
    else:
        residual_valid = residual_all.index_select(0, valid_idx).to(torch.float64)
    subvecs = int(selected_codes.shape[1])
    subdim = int(codebooks.shape[-1])
    risk_valid = torch.zeros((int(valid_idx.numel()),), dtype=torch.float64, device=values.device)
    num_codes = 1 << int(actual_value_subbits)
    num_pages = int(page_starts.numel())
    bucket_count = int(max(1, num_pages * num_codes))
    for sub in range(subvecs):
        lo = int(sub) * subdim
        hi = lo + subdim
        per_token = torch.sum(residual_valid[:, lo:hi] * residual_valid[:, lo:hi], dim=1)
        bucket_ids = valid_pages * int(num_codes) + selected_codes[:, int(sub)]
        bucket_ids = torch.clamp(bucket_ids, min=0, max=bucket_count - 1)
        sums = torch.bincount(bucket_ids, weights=per_token, minlength=bucket_count)
        counts = torch.bincount(bucket_ids, minlength=bucket_count).to(dtype=torch.float64)
        means = sums / torch.clamp_min(counts, 1.0)
        risk_valid += means.index_select(0, bucket_ids)
    out.index_copy_(0, valid_idx, risk_valid)
    return out, int(actual_value_subbits)


def value_vpq_code_stat_risk_subset_torch(
    *,
    index: GPUIndex,
    values: torch.Tensor,
    tokens: torch.Tensor,
    residual_subset: torch.Tensor,
    valid: torch.Tensor,
    page_ids: torch.Tensor,
    subbits: int,
    value_subvecs: int,
    value_subbits: int,
    value_bytes: int,
) -> tuple[torch.Tensor, int]:
    """Per-token V-PQ residual-risk stats for a sealed page subset.

    The full risk statistic is page/code-local, so a newly sealed page can be
    refreshed without rereading/recomputing older pages.
    """

    out = torch.zeros((int(tokens.numel()),), dtype=torch.float64, device=values.device)
    pack = value_vpq_pack_torch(
        index=index,
        values=values,
        value_subvecs=int(value_subvecs),
        value_subbits=int(value_subbits) if int(value_subbits) > 0 else int(subbits),
        key_bytes=int(value_bytes),
        device=values.device,
    )
    if pack is None or tokens.numel() == 0 or not bool(torch.any(valid)):
        actual_value_subbits = int(value_subbits) if int(value_subbits) > 0 else int(subbits)
        return out.reshape(tokens.shape), int(actual_value_subbits)
    codebooks, codes, page_starts, page_size, actual_value_subbits = pack
    flat_tokens = tokens.reshape(-1)
    valid_flat = valid.reshape(-1)
    valid_idx = torch.nonzero(valid_flat, as_tuple=False).reshape(-1)
    valid_pages = page_ids.reshape(-1).index_select(0, valid_idx).to(torch.long)
    valid_pages = torch.clamp(valid_pages, min=0, max=max(0, int(page_starts.numel()) - 1))
    token_values = flat_tokens.index_select(0, valid_idx)
    rows = token_values - page_starts.index_select(0, valid_pages)
    row_mask = (rows >= 0) & (rows < int(page_size))
    if not bool(torch.any(row_mask)):
        return out.reshape(tokens.shape), int(actual_value_subbits)
    row_idx = torch.nonzero(row_mask, as_tuple=False).reshape(-1)
    valid_idx = valid_idx.index_select(0, row_idx)
    valid_pages = valid_pages.index_select(0, row_idx)
    rows = rows.index_select(0, row_idx).to(torch.long)
    selected_codes = codes[valid_pages, rows].to(torch.long)
    residual_valid = residual_subset.reshape(-1, int(values.shape[-1])).index_select(0, valid_idx).to(torch.float64)
    subvecs = int(selected_codes.shape[1])
    subdim = int(codebooks.shape[-1])
    risk_valid = torch.zeros((int(valid_idx.numel()),), dtype=torch.float64, device=values.device)
    num_codes = 1 << int(actual_value_subbits)
    num_pages = int(page_starts.numel())
    bucket_count = int(max(1, num_pages * num_codes))
    for sub in range(subvecs):
        lo = int(sub) * subdim
        hi = lo + subdim
        per_token = torch.sum(residual_valid[:, lo:hi] * residual_valid[:, lo:hi], dim=1)
        bucket_ids = valid_pages * int(num_codes) + selected_codes[:, int(sub)]
        bucket_ids = torch.clamp(bucket_ids, min=0, max=bucket_count - 1)
        sums = torch.bincount(bucket_ids, weights=per_token, minlength=bucket_count)
        counts = torch.bincount(bucket_ids, minlength=bucket_count).to(dtype=torch.float64)
        means = sums / torch.clamp_min(counts, 1.0)
        risk_valid += means.index_select(0, bucket_ids)
    out.index_copy_(0, valid_idx, risk_valid)
    return out.reshape(tokens.shape), int(actual_value_subbits)


def selected_value_exact_mask_gpu(
    *,
    selected_logits: torch.Tensor,
    rule: str,
    exact_top: int,
    exact_mass: float,
    min_top: int,
    max_top: int,
) -> torch.Tensor:
    heads, count = selected_logits.shape
    mask = torch.zeros((heads, count), dtype=torch.bool, device=selected_logits.device)
    if count == 0:
        return mask
    if str(rule) == "selector_rank":
        order = torch.arange(count, dtype=torch.long, device=selected_logits.device).reshape(1, count).expand(heads, -1)
        exact_counts = torch.full((heads,), max(0, min(count, int(exact_top))), dtype=torch.long, device=selected_logits.device)
    else:
        order = torch.argsort(selected_logits.float(), dim=1, descending=True, stable=True)
    if str(rule) == "fixed":
        exact_counts = torch.full((heads,), max(0, min(count, int(exact_top))), dtype=torch.long, device=selected_logits.device)
    elif str(rule) == "selector_rank":
        pass
    elif str(rule) == "selected_mass":
        probs = torch.softmax(selected_logits.float(), dim=1)
        cumulative = torch.cumsum(torch.gather(probs, 1, order), dim=1)
        target = max(0.0, min(1.0, float(exact_mass)))
        exact_counts = torch.sum(cumulative < target, dim=1).to(torch.long) + (1 if target > 0.0 else 0)
        exact_counts = torch.clamp(exact_counts, min=0, max=count)
    else:
        raise ValueError(f"GPU fast path does not support selected_value_exact_rule={rule}")
    if int(min_top) > 0:
        exact_counts = torch.maximum(
            exact_counts,
            torch.full_like(exact_counts, min(count, int(min_top))),
        )
    if int(max_top) > 0:
        exact_counts = torch.minimum(
            exact_counts,
            torch.full_like(exact_counts, min(count, int(max_top))),
        )
    ranks = torch.arange(count, dtype=torch.long, device=selected_logits.device).reshape(1, count)
    sorted_mask = ranks < exact_counts.reshape(heads, 1)
    return mask.scatter(1, order, sorted_mask)


def selected_value_exact_counts_from_mass_gpu(
    *,
    ranked_logits: torch.Tensor,
    ranked_scores: torch.Tensor,
    base_logsumexp: torch.Tensor | None,
    exact_mass: float,
    min_top: int,
    max_top: int,
) -> torch.Tensor:
    """Per row, count ranked tokens whose exact V should be kept.

    Static prefix/suffix and pending-page tokens are always exact in the native
    kernels. This count therefore applies only to the ranked dynamic tokens and
    chooses the smallest exact-logit prefix that reaches the requested mass
    inside the selected set.
    """

    if ranked_logits.dim() not in {2, 3}:
        raise ValueError(f"ranked_logits must be [heads, rank] or [positions, heads, rank], got {tuple(ranked_logits.shape)}")
    rank = int(ranked_logits.shape[-1])
    leading = ranked_logits.shape[:-1]
    if rank <= 0:
        return torch.zeros(leading, dtype=torch.long, device=ranked_logits.device)
    target = float(max(0.0, min(1.0, float(exact_mass))))
    if target <= 0.0:
        counts = torch.zeros(leading, dtype=torch.long, device=ranked_logits.device)
    else:
        valid = torch.isfinite(ranked_scores[..., :rank]) & torch.isfinite(ranked_logits[..., :rank])
        logits = torch.where(valid, ranked_logits[..., :rank].float(), torch.full_like(ranked_logits[..., :rank].float(), float("-inf")))
        sorted_logits, _ = torch.sort(logits, dim=-1, descending=True, stable=True)
        ranked_lse = torch.logsumexp(logits, dim=-1)
        if base_logsumexp is None:
            base_lse = torch.full_like(ranked_lse, float("-inf"))
        else:
            base_lse = base_logsumexp.float()
        total_lse = torch.logaddexp(base_lse, ranked_lse)
        base_mass = torch.where(
            torch.isfinite(total_lse),
            torch.exp(base_lse - total_lse),
            torch.zeros_like(total_lse),
        )
        cum_ranked_lse = torch.logcumsumexp(sorted_logits, dim=-1)
        cum_lse = torch.logaddexp(base_lse.unsqueeze(-1), cum_ranked_lse)
        cum_mass = torch.where(
            torch.isfinite(total_lse).unsqueeze(-1),
            torch.exp(cum_lse - total_lse.unsqueeze(-1)),
            torch.zeros_like(cum_lse),
        )
        hit = cum_mass >= min(float(target), 1.0 - 1.0e-7)
        has_hit = torch.any(hit, dim=-1)
        first_hit = torch.argmax(hit.to(torch.int32), dim=-1).to(torch.long) + 1
        counts = torch.where(
            base_mass >= float(target),
            torch.zeros_like(first_hit),
            torch.where(has_hit, first_hit, valid.sum(dim=-1).to(torch.long)),
        )
    if int(min_top) > 0:
        counts = torch.maximum(counts, torch.full_like(counts, min(rank, int(min_top))))
    if int(max_top) > 0:
        counts = torch.minimum(counts, torch.full_like(counts, min(rank, int(max_top))))
    return torch.clamp(counts, min=0, max=rank).to(torch.long)


def geometric_budget_pairs(
    *,
    min_budget: int,
    max_budget: int,
    granularity: int,
    growth: float,
    probe_scale: float,
) -> tuple[list[int], list[int]]:
    tails: list[int] = []
    probes: list[int] = []
    max_budget = max(0, int(max_budget))
    granularity = max(1, int(granularity))
    tail_budget = _round_budget_up(
        int(min_budget),
        granularity=granularity,
        max_budget=max_budget,
    )
    while tail_budget < max_budget:
        probe_budget = _round_budget_up(
            int(max(float(tail_budget + granularity), float(probe_scale) * float(tail_budget))),
            granularity=granularity,
            max_budget=max_budget,
        )
        probe_budget = max(tail_budget, int(probe_budget))
        tails.append(int(tail_budget))
        probes.append(int(probe_budget))
        if probe_budget >= max_budget:
            break
        next_budget = _round_budget_up(
            int(max(float(probe_budget + granularity), float(growth) * float(probe_budget))),
            granularity=granularity,
            max_budget=max_budget,
        )
        if next_budget <= probe_budget:
            break
        tail_budget = int(next_budget)
    return tails, probes


def selected_mass_thresholds_from_logits_gpu(
    *,
    ranked_logits: torch.Tensor,
    ranked_scores: torch.Tensor,
    base_logsumexp: torch.Tensor | None,
    budgets: list[int],
    exact_mass: float,
    min_top: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    heads = int(ranked_logits.shape[0])
    steps = len(budgets)
    device = ranked_logits.device
    thresholds = torch.empty((heads, steps), dtype=torch.float32, device=device)
    threshold_sels = torch.empty((heads, steps), dtype=torch.long, device=device)
    if steps == 0:
        return thresholds, threshold_sels
    rank = int(ranked_logits.shape[-1])
    target = float(max(0.0, min(1.0, float(exact_mass))))
    base_lse = (
        base_logsumexp.float()
        if base_logsumexp is not None
        else torch.full((heads,), float("-inf"), dtype=torch.float32, device=device)
    )
    valid_all = torch.isfinite(ranked_scores[:, :rank]) & torch.isfinite(ranked_logits[:, :rank])
    logits_all = torch.where(
        valid_all,
        ranked_logits[:, :rank].float(),
        torch.full((heads, rank), float("-inf"), dtype=torch.float32, device=device),
    )
    prefix_lse_all = torch.logcumsumexp(logits_all, dim=-1)
    prefix_valid_counts = torch.cumsum(valid_all.to(torch.long), dim=-1)
    budgets_tensor = torch.tensor(
        [max(0, min(rank, int(budget))) for budget in budgets],
        dtype=torch.long,
        device=device,
    )
    positive_steps = budgets_tensor > 0
    if not bool(torch.any(positive_steps)):
        thresholds.fill_(float("inf"))
        threshold_sels.fill_(-1)
        return thresholds.contiguous(), threshold_sels.contiguous()

    if _env_truthy("SELECTOR_PQ_THRESHOLD_MIN_TOP_FAST") and int(min_top) > 0:
        k_min = max(1, min(rank, int(min_top)))
        top_logits_min, top_order_min = torch.topk(logits_all, k=k_min, dim=-1, largest=True, sorted=True)
        top_valid_min = torch.isfinite(top_logits_min)
        lse_idx = torch.clamp(budgets_tensor - 1, min=0)
        ranked_lse = prefix_lse_all.index_select(1, lse_idx)
        total_lse = torch.logaddexp(base_lse.reshape(heads, 1), ranked_lse)
        valid_count = prefix_valid_counts.index_select(1, lse_idx)
        valid_count = torch.where(positive_steps.reshape(1, steps), valid_count, torch.zeros_like(valid_count))
        min_counts = torch.minimum(
            budgets_tensor.reshape(1, steps),
            torch.full((heads, steps), int(min_top), dtype=torch.long, device=device),
        )
        min_counts = torch.where(positive_steps.reshape(1, steps), min_counts, torch.zeros_like(min_counts))
        budget_view = budgets_tensor.reshape(1, steps, 1)
        in_budget_any = top_order_min.unsqueeze(1) < budget_view
        in_budget_sorted = in_budget_any & top_valid_min.unsqueeze(1)
        cum_selected_count = torch.cumsum(in_budget_sorted.to(torch.long), dim=-1)
        top_valid_count = cum_selected_count[..., -1]
        has_exact = min_counts > 0
        kth_valid_mask = in_budget_sorted & (cum_selected_count >= min_counts.unsqueeze(-1))
        kth_pos = torch.argmax(kth_valid_mask.to(torch.int32), dim=-1).to(torch.long)
        if target <= 0.0:
            mass_ok = torch.ones((heads, steps), dtype=torch.bool, device=device)
        else:
            sorted_logits_min = top_logits_min.unsqueeze(1).expand(-1, steps, -1).masked_fill(
                ~in_budget_sorted,
                float("-inf"),
            )
            cum_ranked_lse = torch.logcumsumexp(sorted_logits_min, dim=-1)
            cum_lse = torch.logaddexp(base_lse.reshape(heads, 1, 1), cum_ranked_lse)
            cum_mass = torch.where(
                torch.isfinite(total_lse).unsqueeze(-1),
                torch.exp(cum_lse - total_lse.unsqueeze(-1)),
                torch.zeros_like(cum_lse),
            )
            mass_at_min = torch.gather(cum_mass, 2, kth_pos.unsqueeze(-1)).squeeze(-1)
            mass_ok = mass_at_min >= min(target, 1.0 - 1.0e-7)
        sufficient = (
            (~positive_steps.reshape(1, steps))
            | (~has_exact)
            | ((top_valid_count >= min_counts) & (min_counts <= valid_count) & mass_ok)
        )
        if bool(torch.all(sufficient)):
            threshold_vals = torch.gather(
                top_logits_min.unsqueeze(1).expand(-1, steps, -1),
                2,
                kth_pos.unsqueeze(-1),
            ).squeeze(-1)
            threshold_idx = torch.gather(
                top_order_min.unsqueeze(1).expand(-1, steps, -1),
                2,
                kth_pos.unsqueeze(-1),
            ).squeeze(-1).to(torch.long)
            thresholds[:, :] = torch.where(
                has_exact,
                threshold_vals,
                torch.full_like(threshold_vals, float("inf")),
            )
            threshold_sels[:, :] = torch.where(
                has_exact,
                threshold_idx,
                torch.full_like(threshold_idx, -1),
            )
            return thresholds.contiguous(), threshold_sels.contiguous()

    topk_limit = 0
    try:
        topk_limit = int(os.environ.get("SELECTOR_PQ_THRESHOLD_TOPK", "0"))
    except ValueError:
        topk_limit = 0
    if 0 < topk_limit <= rank:
        k_top = max(1, min(rank, int(topk_limit)))
        top_logits_all, top_order_all = torch.topk(logits_all, k=k_top, dim=-1, largest=True, sorted=True)
        top_valid_all = torch.isfinite(top_logits_all)
        if _env_truthy("SELECTOR_PQ_THRESHOLD_NATIVE_TOPK"):
            try:
                native = load_selector_paged_pq_ext()
                native_threshold_fn = getattr(native, "selected_mass_thresholds_from_topk", None)
            except Exception:
                native_threshold_fn = None
            if native_threshold_fn is not None:
                native_thresholds, native_threshold_sels, native_sufficient = native_threshold_fn(
                    top_logits_all.contiguous(),
                    top_order_all.contiguous(),
                    prefix_lse_all.contiguous(),
                    prefix_valid_counts.contiguous(),
                    base_lse.contiguous(),
                    budgets_tensor.contiguous(),
                    float(target),
                    int(min_top),
                )
                if k_top >= rank or _env_truthy("SELECTOR_PQ_THRESHOLD_TOPK_ASSUME_SUFFICIENT") or bool(
                    torch.all(native_sufficient.to(torch.bool))
                ):
                    return native_thresholds.contiguous(), native_threshold_sels.contiguous()
        lse_idx = torch.clamp(budgets_tensor - 1, min=0)
        ranked_lse = prefix_lse_all.index_select(1, lse_idx)
        total_lse = torch.logaddexp(base_lse.reshape(heads, 1), ranked_lse)
        valid_count = prefix_valid_counts.index_select(1, lse_idx)
        valid_count = torch.where(positive_steps.reshape(1, steps), valid_count, torch.zeros_like(valid_count))
        budget_view = budgets_tensor.reshape(1, steps, 1)
        in_budget_any = top_order_all.unsqueeze(1) < budget_view
        in_budget_sorted = in_budget_any & top_valid_all.unsqueeze(1)
        sorted_logits = top_logits_all.unsqueeze(1).expand(-1, steps, -1).masked_fill(
            ~in_budget_sorted,
            float("-inf"),
        )
        cum_selected_count = torch.cumsum(in_budget_sorted.to(torch.long), dim=-1)
        top_valid_count = cum_selected_count[..., -1]
        if target <= 0.0:
            counts = torch.zeros((heads, steps), dtype=torch.long, device=device)
            has_hit = torch.ones((heads, steps), dtype=torch.bool, device=device)
            base_mass = torch.ones((heads, steps), dtype=torch.float32, device=device)
        else:
            base_mass = torch.where(
                torch.isfinite(total_lse),
                torch.exp(base_lse.reshape(heads, 1) - total_lse),
                torch.zeros_like(total_lse),
            )
            cum_ranked_lse = torch.logcumsumexp(sorted_logits, dim=-1)
            cum_lse = torch.logaddexp(base_lse.reshape(heads, 1, 1), cum_ranked_lse)
            cum_mass = torch.where(
                torch.isfinite(total_lse).unsqueeze(-1),
                torch.exp(cum_lse - total_lse.unsqueeze(-1)),
                torch.zeros_like(cum_lse),
            )
            hit = (cum_mass >= min(target, 1.0 - 1.0e-7)) & in_budget_sorted
            has_hit = torch.any(hit, dim=-1)
            first_hit_pos = torch.argmax(hit.to(torch.int32), dim=-1).to(torch.long)
            first_hit_count = torch.gather(cum_selected_count, 2, first_hit_pos.unsqueeze(-1)).squeeze(-1)
            counts = torch.where(
                base_mass >= target,
                torch.zeros_like(first_hit_count),
                torch.where(has_hit, first_hit_count, valid_count),
            )
        if int(min_top) > 0:
            min_counts = torch.minimum(
                budgets_tensor.reshape(1, steps),
                torch.full((heads, steps), int(min_top), dtype=torch.long, device=device),
            )
            counts = torch.maximum(counts, min_counts)
        counts = torch.minimum(torch.clamp(counts, min=0), budgets_tensor.reshape(1, steps))
        counts = torch.where(positive_steps.reshape(1, steps), counts, torch.zeros_like(counts))
        has_exact = counts > 0
        if target <= 0.0:
            enough_for_target = torch.ones((heads, steps), dtype=torch.bool, device=device)
        else:
            enough_for_target = (base_mass >= target) | has_hit
        if int(min_top) > 0:
            enough_for_target = enough_for_target | (counts <= int(min_top))
        topk_sufficient = torch.all((~has_exact) | (enough_for_target & (counts <= top_valid_count)))
        # Diagnostic-only fast path: this removes the per-call GPU->CPU sync used
        # to prove fallback safety. Do not enable by default unless a separate
        # correctness check proves the configured topk is always sufficient.
        if _env_truthy("SELECTOR_PQ_THRESHOLD_TOPK_ASSUME_SUFFICIENT") or bool(topk_sufficient):
            cum_budget_count = torch.cumsum(in_budget_any.to(torch.long), dim=-1)
            kth_valid_mask = in_budget_sorted & (cum_selected_count >= counts.unsqueeze(-1))
            kth_any_mask = in_budget_any & (cum_budget_count >= counts.unsqueeze(-1))
            kth_mask = torch.where((counts <= valid_count).unsqueeze(-1), kth_valid_mask, kth_any_mask)
            kth_pos = torch.argmax(kth_mask.to(torch.int32), dim=-1).to(torch.long)
            threshold_vals = torch.gather(
                top_logits_all.unsqueeze(1).expand(-1, steps, -1),
                2,
                kth_pos.unsqueeze(-1),
            ).squeeze(-1)
            threshold_idx = torch.gather(
                top_order_all.unsqueeze(1).expand(-1, steps, -1),
                2,
                kth_pos.unsqueeze(-1),
            ).squeeze(-1).to(torch.long)
            thresholds[:, :] = torch.where(
                has_exact,
                threshold_vals,
                torch.full_like(threshold_vals, float("inf")),
            )
            threshold_sels[:, :] = torch.where(
                has_exact,
                threshold_idx,
                torch.full_like(threshold_idx, -1),
            )
            return thresholds.contiguous(), threshold_sels.contiguous()

    # Sort once across the full ranked prefix. Per-budget exact-V thresholds are
    # then computed by masking this sorted order to the active prefix. This keeps
    # frontier semantics identical while avoiding one O(rank log rank) sort per
    # geometric budget.
    sorted_logits_all, sorted_order_all = torch.sort(logits_all, dim=-1, descending=True, stable=True)
    sorted_valid_all = torch.gather(valid_all, 1, sorted_order_all)

    if _env_truthy("SELECTOR_PQ_THRESHOLD_LOOP"):
        base_lse_1d = base_lse.reshape(heads)
        thresholds.fill_(float("inf"))
        threshold_sels.fill_(-1)
        for step_idx, budget in enumerate([int(v) for v in budgets_tensor.detach().cpu().tolist()]):
            if budget <= 0:
                continue
            lse_idx = max(0, min(rank - 1, int(budget) - 1))
            ranked_lse = prefix_lse_all[:, lse_idx]
            total_lse = torch.logaddexp(base_lse_1d, ranked_lse)
            valid_count = prefix_valid_counts[:, lse_idx]
            in_budget_any = sorted_order_all < int(budget)
            in_budget_sorted = in_budget_any & sorted_valid_all
            sorted_logits = sorted_logits_all.masked_fill(~in_budget_sorted, float("-inf"))
            cum_selected_count = torch.cumsum(in_budget_sorted.to(torch.long), dim=-1)
            if target <= 0.0:
                counts = torch.zeros((heads,), dtype=torch.long, device=device)
            else:
                base_mass = torch.where(
                    torch.isfinite(total_lse),
                    torch.exp(base_lse_1d - total_lse),
                    torch.zeros_like(total_lse),
                )
                cum_ranked_lse = torch.logcumsumexp(sorted_logits, dim=-1)
                cum_lse = torch.logaddexp(base_lse_1d.reshape(heads, 1), cum_ranked_lse)
                cum_mass = torch.where(
                    torch.isfinite(total_lse).reshape(heads, 1),
                    torch.exp(cum_lse - total_lse.reshape(heads, 1)),
                    torch.zeros_like(cum_lse),
                )
                hit = (cum_mass >= min(target, 1.0 - 1.0e-7)) & in_budget_sorted
                has_hit = torch.any(hit, dim=-1)
                first_hit_pos = torch.argmax(hit.to(torch.int32), dim=-1).to(torch.long)
                first_hit_count = torch.gather(cum_selected_count, 1, first_hit_pos.reshape(heads, 1)).squeeze(1)
                counts = torch.where(
                    base_mass >= target,
                    torch.zeros_like(first_hit_count),
                    torch.where(has_hit, first_hit_count, valid_count),
                )
            if int(min_top) > 0:
                min_counts = torch.minimum(
                    torch.full((heads,), int(budget), dtype=torch.long, device=device),
                    torch.full((heads,), int(min_top), dtype=torch.long, device=device),
                )
                counts = torch.maximum(counts, min_counts)
            counts = torch.minimum(torch.clamp(counts, min=0), torch.full_like(counts, int(budget)))
            has_exact = counts > 0
            cum_budget_count = torch.cumsum(in_budget_any.to(torch.long), dim=-1)
            kth_valid_mask = in_budget_sorted & (cum_selected_count >= counts.reshape(heads, 1))
            kth_any_mask = in_budget_any & (cum_budget_count >= counts.reshape(heads, 1))
            kth_mask = torch.where((counts <= valid_count).reshape(heads, 1), kth_valid_mask, kth_any_mask)
            kth_pos = torch.argmax(kth_mask.to(torch.int32), dim=-1).to(torch.long)
            threshold_vals = torch.gather(sorted_logits_all, 1, kth_pos.reshape(heads, 1)).squeeze(1)
            threshold_idx = torch.gather(sorted_order_all, 1, kth_pos.reshape(heads, 1)).squeeze(1).to(torch.long)
            thresholds[:, step_idx] = torch.where(
                has_exact,
                threshold_vals,
                torch.full_like(threshold_vals, float("inf")),
            )
            threshold_sels[:, step_idx] = torch.where(
                has_exact,
                threshold_idx,
                torch.full_like(threshold_idx, -1),
            )
        return thresholds.contiguous(), threshold_sels.contiguous()

    lse_idx = torch.clamp(budgets_tensor - 1, min=0)
    ranked_lse = prefix_lse_all.index_select(1, lse_idx)
    total_lse = torch.logaddexp(base_lse.reshape(heads, 1), ranked_lse)
    valid_count = prefix_valid_counts.index_select(1, lse_idx)
    valid_count = torch.where(positive_steps.reshape(1, steps), valid_count, torch.zeros_like(valid_count))

    budget_view = budgets_tensor.reshape(1, steps, 1)
    in_budget_any = sorted_order_all.unsqueeze(1) < budget_view
    in_budget_sorted = in_budget_any & sorted_valid_all.unsqueeze(1)
    sorted_logits = sorted_logits_all.unsqueeze(1).expand(-1, steps, -1).masked_fill(
        ~in_budget_sorted,
        float("-inf"),
    )
    cum_selected_count = torch.cumsum(in_budget_sorted.to(torch.long), dim=-1)
    if target <= 0.0:
        counts = torch.zeros((heads, steps), dtype=torch.long, device=device)
    else:
        base_mass = torch.where(
            torch.isfinite(total_lse),
            torch.exp(base_lse.reshape(heads, 1) - total_lse),
            torch.zeros_like(total_lse),
        )
        cum_ranked_lse = torch.logcumsumexp(sorted_logits, dim=-1)
        cum_lse = torch.logaddexp(base_lse.reshape(heads, 1, 1), cum_ranked_lse)
        cum_mass = torch.where(
            torch.isfinite(total_lse).unsqueeze(-1),
            torch.exp(cum_lse - total_lse.unsqueeze(-1)),
            torch.zeros_like(cum_lse),
        )
        hit = (cum_mass >= min(target, 1.0 - 1.0e-7)) & in_budget_sorted
        has_hit = torch.any(hit, dim=-1)
        first_hit_pos = torch.argmax(hit.to(torch.int32), dim=-1).to(torch.long)
        first_hit_count = torch.gather(cum_selected_count, 2, first_hit_pos.unsqueeze(-1)).squeeze(-1)
        counts = torch.where(
            base_mass >= target,
            torch.zeros_like(first_hit_count),
            torch.where(has_hit, first_hit_count, valid_count),
        )
    if int(min_top) > 0:
        min_counts = torch.minimum(
            budgets_tensor.reshape(1, steps),
            torch.full((heads, steps), int(min_top), dtype=torch.long, device=device),
        )
        counts = torch.maximum(counts, min_counts)
    counts = torch.minimum(torch.clamp(counts, min=0), budgets_tensor.reshape(1, steps))
    counts = torch.where(positive_steps.reshape(1, steps), counts, torch.zeros_like(counts))
    has_exact = counts > 0
    cum_budget_count = torch.cumsum(in_budget_any.to(torch.long), dim=-1)
    kth_valid_mask = in_budget_sorted & (cum_selected_count >= counts.unsqueeze(-1))
    kth_any_mask = in_budget_any & (cum_budget_count >= counts.unsqueeze(-1))
    kth_mask = torch.where((counts <= valid_count).unsqueeze(-1), kth_valid_mask, kth_any_mask)
    kth_pos = torch.argmax(kth_mask.to(torch.int32), dim=-1).to(torch.long)
    threshold_vals = torch.gather(
        sorted_logits_all.unsqueeze(1).expand(-1, steps, -1),
        2,
        kth_pos.unsqueeze(-1),
    ).squeeze(-1)
    threshold_idx = torch.gather(
        sorted_order_all.unsqueeze(1).expand(-1, steps, -1),
        2,
        kth_pos.unsqueeze(-1),
    ).squeeze(-1).to(torch.long)
    thresholds[:, :] = torch.where(
        has_exact,
        threshold_vals,
        torch.full_like(threshold_vals, float("inf")),
    )
    threshold_sels[:, :] = torch.where(
        has_exact,
        threshold_idx,
        torch.full_like(threshold_idx, -1),
    )
    return thresholds.contiguous(), threshold_sels.contiguous()


def select_thresholds_for_budget_counts_gpu(
    *,
    thresholds: torch.Tensor,
    threshold_sels: torch.Tensor,
    budgets: list[int],
    counts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather per-head threshold rows for accepted geometric budgets."""

    heads = int(counts.shape[0])
    device = counts.device
    if len(budgets) == 0 or thresholds.numel() == 0:
        return (
            torch.full((heads,), float("inf"), dtype=torch.float32, device=device),
            torch.full((heads,), -1, dtype=torch.long, device=device),
        )
    budget_tensor = torch.tensor([int(v) for v in budgets], dtype=torch.long, device=device)
    idx = torch.searchsorted(budget_tensor, counts.to(torch.long), right=False)
    idx = torch.clamp(idx, min=0, max=len(budgets) - 1)
    row = torch.arange(heads, dtype=torch.long, device=device)
    return thresholds[row, idx].contiguous(), threshold_sels[row, idx].contiguous()


def _gpu_gqa_ranked_exact_logits(
    *,
    queries: torch.Tensor,
    keys_all: torch.Tensor,
    ranked_tokens: torch.Tensor,
    group_size: int,
    scale: float,
    max_rank: int,
    rank_chunk: int = 32,
) -> torch.Tensor:
    """Exact QK logits for ranked dynamic candidates only.

    The selector scores are PQ-domain approximations. Confidence gates need the
    exact logits for already selected candidates, which is deployable because
    those K vectors are on the exact-attention path. This helper keeps the
    operation on GPU and chunks over rank to bound peak memory.
    """

    if ranked_tokens.dim() not in {2, 3}:
        raise ValueError(f"ranked_tokens must be [heads, rank] or [queries, heads, rank], got {tuple(ranked_tokens.shape)}")
    rank = min(max(0, int(max_rank)), int(ranked_tokens.shape[-1]))
    if rank <= 0:
        return torch.empty((*ranked_tokens.shape[:-1], 0), dtype=torch.float32, device=ranked_tokens.device)
    out = torch.empty((*ranked_tokens.shape[:-1], rank), dtype=torch.float32, device=ranked_tokens.device)
    dim = int(queries.shape[-1])
    kv_heads = int(keys_all.shape[0])
    group = max(1, int(group_size))
    rank_chunk = max(1, int(rank_chunk))
    keys_token_count = int(keys_all.shape[1])

    if ranked_tokens.dim() == 3:
        positions = int(ranked_tokens.shape[0])
        heads = int(ranked_tokens.shape[1])
        for kv_head in range(kv_heads):
            head_start = int(kv_head * group)
            head_end = min(heads, head_start + group)
            if head_start >= head_end:
                continue
            q = queries[:, head_start:head_end, :].float()
            for rank_start in range(0, rank, rank_chunk):
                rank_end = min(rank, rank_start + rank_chunk)
                toks = ranked_tokens[:, head_start:head_end, rank_start:rank_end].to(torch.long)
                toks = toks.clamp(min=0, max=max(0, keys_token_count - 1))
                gathered = keys_all[int(kv_head)].index_select(0, toks.reshape(-1)).reshape(
                    positions,
                    head_end - head_start,
                    rank_end - rank_start,
                    dim,
                )
                logits = torch.sum(q.unsqueeze(2) * gathered.float(), dim=-1) * float(scale)
                out[:, head_start:head_end, rank_start:rank_end] = logits
    else:
        heads = int(ranked_tokens.shape[0])
        for kv_head in range(kv_heads):
            head_start = int(kv_head * group)
            head_end = min(heads, head_start + group)
            if head_start >= head_end:
                continue
            q = queries[head_start:head_end, :].float()
            for rank_start in range(0, rank, rank_chunk):
                rank_end = min(rank, rank_start + rank_chunk)
                toks = ranked_tokens[head_start:head_end, rank_start:rank_end].to(torch.long)
                toks = toks.clamp(min=0, max=max(0, keys_token_count - 1))
                gathered = keys_all[int(kv_head)].index_select(0, toks.reshape(-1)).reshape(
                    head_end - head_start,
                    rank_end - rank_start,
                    dim,
                )
                logits = torch.sum(q.unsqueeze(1) * gathered.float(), dim=-1) * float(scale)
                out[head_start:head_end, rank_start:rank_end] = logits
    return out


def _gpu_gqa_dense_decode_ranked_logits_and_base_lse(
    *,
    queries: torch.Tensor,
    keys_all: torch.Tensor,
    keys_all_t_float: torch.Tensor | None,
    ranked_tokens: torch.Tensor,
    group_size: int,
    scale: float,
    max_rank: int,
    query_context_len: int,
    static_prefix: int,
    static_suffix: int,
    page_size: int,
    need_base_lse: bool,
) -> tuple[torch.Tensor, torch.Tensor | None, int, int]:
    """GPU simulator exact logits via dense QK then ranked gather.

    This deliberately favors GPU throughput over physical access fidelity. The
    output is still the exact ranked logits used by the frontier algorithm, but
    the GPU host may read more K than the custom-hardware logical model.
    """

    if ranked_tokens.dim() != 2:
        raise ValueError(f"dense decode exact logits expects [heads, rank], got {tuple(ranked_tokens.shape)}")
    rank = min(max(0, int(max_rank)), int(ranked_tokens.shape[-1]))
    heads = int(ranked_tokens.shape[0])
    device = ranked_tokens.device
    if rank <= 0:
        empty = torch.empty((heads, 0), dtype=torch.float32, device=device)
        base = torch.full((heads,), float("-inf"), dtype=torch.float32, device=device) if need_base_lse else None
        return empty, base, 0, 0

    key_count = min(max(0, int(query_context_len)), int(keys_all.shape[1]))
    if key_count <= 0:
        out = torch.full((heads, rank), float("-inf"), dtype=torch.float32, device=device)
        base = torch.full((heads,), float("-inf"), dtype=torch.float32, device=device) if need_base_lse else None
        return out, base, 0, 0

    out = torch.empty((heads, rank), dtype=torch.float32, device=device)
    base_out = (
        torch.full((heads,), float("-inf"), dtype=torch.float32, device=device) if bool(need_base_lse) else None
    )
    base_toks: torch.Tensor | None = None
    base_mask: torch.Tensor | None = None
    total_base = 0
    if bool(need_base_lse):
        token_rows, mask_rows, total_base = _prefill_base_token_rows(
            query_len=1,
            query_start=int(query_context_len) - 1,
            static_prefix=int(static_prefix),
            static_suffix=int(static_suffix),
            page_size=int(page_size),
            device=device,
        )
        if token_rows.numel() > 0:
            base_toks = token_rows[0].clamp(min=0, max=max(0, key_count - 1)).to(torch.long)
            base_mask = mask_rows[0]

    dim = int(queries.shape[-1])
    kv_heads = int(keys_all.shape[0])
    group = max(1, int(group_size))
    covered_heads = min(heads, kv_heads * group)
    aligned_heads = (covered_heads // group) * group
    if aligned_heads > 0:
        aligned_kv_heads = aligned_heads // group
        q_grouped = queries[:aligned_heads, :].reshape(aligned_kv_heads, group, dim).float()
        if keys_all_t_float is not None:
            key_t_grouped = keys_all_t_float[:aligned_kv_heads, :, :key_count]
        else:
            key_t_grouped = keys_all[:aligned_kv_heads, :key_count, :].float().transpose(1, 2).contiguous()
        dense_grouped = torch.bmm(
            q_grouped,
            key_t_grouped,
        ) * float(scale)
        dense_logits = dense_grouped.reshape(aligned_heads, key_count)
        toks = ranked_tokens[:aligned_heads, :rank].to(torch.long).clamp(min=0, max=max(0, key_count - 1))
        out[:aligned_heads, :rank] = torch.gather(dense_logits, 1, toks)
        if base_out is not None:
            if base_toks is None or base_toks.numel() == 0:
                base_out[:aligned_heads] = float("-inf")
            else:
                base_logits = dense_logits.index_select(1, base_toks)
                if base_mask is not None:
                    base_logits = base_logits.masked_fill(~base_mask.reshape(1, -1), float("-inf"))
                base_out[:aligned_heads] = torch.logsumexp(base_logits, dim=-1)

    for kv_head in range(aligned_heads // group, kv_heads):
        head_start = int(kv_head * group)
        head_end = min(heads, head_start + group)
        if head_start >= head_end:
            continue
        q = queries[head_start:head_end, :].float()
        # Dense GEMM is usually much faster on GPU than irregular ranked-K gathers.
        if keys_all_t_float is not None:
            key_t = keys_all_t_float[int(kv_head), :, :key_count]
        else:
            key_t = keys_all[int(kv_head), :key_count, :].float().t()
        dense_logits = torch.matmul(q, key_t) * float(scale)
        toks = ranked_tokens[head_start:head_end, :rank].to(torch.long).clamp(min=0, max=max(0, key_count - 1))
        out[head_start:head_end, :rank] = torch.gather(dense_logits, 1, toks)
        if base_out is not None:
            if base_toks is None or base_toks.numel() == 0:
                base_out[head_start:head_end] = float("-inf")
            else:
                base_logits = dense_logits.index_select(1, base_toks)
                if base_mask is not None:
                    base_logits = base_logits.masked_fill(~base_mask.reshape(1, -1), float("-inf"))
                base_out[head_start:head_end] = torch.logsumexp(base_logits, dim=-1)
    return out.contiguous(), base_out.contiguous() if base_out is not None else None, int(total_base), int(key_count)


def _prefill_base_token_rows(
    *,
    query_len: int,
    query_start: int,
    static_prefix: int,
    static_suffix: int,
    page_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    q_lens = [int(query_start) + i + 1 for i in range(int(query_len))]
    rows: list[list[int]] = []
    max_base = 0
    total_base = 0
    page_size_i = max(1, int(page_size))
    for q_len in q_lens:
        prefix_end = min(max(0, int(static_prefix)), int(q_len))
        indexed_end = max(prefix_end, int(q_len) - max(0, int(static_suffix)))
        sealed_end = prefix_end + ((max(0, indexed_end - prefix_end) // page_size_i) * page_size_i)
        suffix_start = max(sealed_end, prefix_end)
        toks = list(range(prefix_end))
        if suffix_start < q_len:
            toks.extend(range(suffix_start, q_len))
        rows.append(toks)
        max_base = max(max_base, len(toks))
        total_base += len(toks)
    if max_base <= 0:
        return (
            torch.empty((int(query_len), 0), dtype=torch.long, device=device),
            torch.empty((int(query_len), 0), dtype=torch.bool, device=device),
            0,
        )
    token_rows_cpu = torch.zeros((int(query_len), max_base), dtype=torch.long)
    mask_rows_cpu = torch.zeros((int(query_len), max_base), dtype=torch.bool)
    for row_idx, toks in enumerate(rows):
        if toks:
            token_rows_cpu[row_idx, : len(toks)] = torch.as_tensor(toks, dtype=torch.long)
            mask_rows_cpu[row_idx, : len(toks)] = True
    return token_rows_cpu.to(device=device), mask_rows_cpu.to(device=device), int(total_base)


def _gpu_gqa_base_logsumexp_prefill(
    *,
    queries: torch.Tensor,
    keys_all: torch.Tensor,
    group_size: int,
    query_start: int,
    static_prefix: int,
    static_suffix: int,
    page_size: int,
    scale: float,
    position_chunk: int = 64,
) -> tuple[torch.Tensor, int]:
    token_rows, mask_rows, total_base = _prefill_base_token_rows(
        query_len=int(queries.shape[0]),
        query_start=int(query_start),
        static_prefix=int(static_prefix),
        static_suffix=int(static_suffix),
        page_size=int(page_size),
        device=queries.device,
    )
    positions = int(queries.shape[0])
    heads = int(queries.shape[1])
    if token_rows.numel() == 0:
        return torch.full((positions, heads), float("-inf"), dtype=torch.float32, device=queries.device), int(total_base)
    out = torch.full((positions, heads), float("-inf"), dtype=torch.float32, device=queries.device)
    dim = int(queries.shape[-1])
    kv_heads = int(keys_all.shape[0])
    group = max(1, int(group_size))
    position_chunk = max(1, int(position_chunk))
    keys_token_count = int(keys_all.shape[1])
    for pos_start in range(0, positions, position_chunk):
        pos_end = min(positions, pos_start + position_chunk)
        toks_chunk = token_rows[pos_start:pos_end].clamp(min=0, max=max(0, keys_token_count - 1))
        mask_chunk = mask_rows[pos_start:pos_end]
        base_count = int(toks_chunk.shape[1])
        for kv_head in range(kv_heads):
            head_start = int(kv_head * group)
            head_end = min(heads, head_start + group)
            if head_start >= head_end:
                continue
            q = queries[pos_start:pos_end, head_start:head_end, :].float()
            gathered = keys_all[int(kv_head)].index_select(0, toks_chunk.reshape(-1)).reshape(
                pos_end - pos_start,
                base_count,
                dim,
            )
            logits = torch.einsum("pgd,pbd->pgb", q, gathered.float()) * float(scale)
            logits = logits.masked_fill(~mask_chunk.reshape(pos_end - pos_start, 1, base_count), float("-inf"))
            out[pos_start:pos_end, head_start:head_end] = torch.logsumexp(logits, dim=-1)
    return out, int(total_base)


def _gpu_gqa_base_logsumexp_decode(
    *,
    queries: torch.Tensor,
    keys_all: torch.Tensor,
    group_size: int,
    query_context_len: int,
    static_prefix: int,
    static_suffix: int,
    page_size: int,
    scale: float,
) -> tuple[torch.Tensor, int]:
    token_rows, mask_rows, total_base = _prefill_base_token_rows(
        query_len=1,
        query_start=int(query_context_len) - 1,
        static_prefix=int(static_prefix),
        static_suffix=int(static_suffix),
        page_size=int(page_size),
        device=queries.device,
    )
    heads = int(queries.shape[0])
    if token_rows.numel() == 0:
        return torch.full((heads,), float("-inf"), dtype=torch.float32, device=queries.device), int(total_base)
    toks = token_rows[0].clamp(min=0, max=max(0, int(keys_all.shape[1]) - 1))
    mask = mask_rows[0]
    out = torch.full((heads,), float("-inf"), dtype=torch.float32, device=queries.device)
    dim = int(queries.shape[-1])
    group = max(1, int(group_size))
    for kv_head in range(int(keys_all.shape[0])):
        head_start = int(kv_head * group)
        head_end = min(heads, head_start + group)
        if head_start >= head_end:
            continue
        gathered = keys_all[int(kv_head)].index_select(0, toks).float()
        logits = torch.matmul(queries[head_start:head_end, :].float(), gathered.t().contiguous()) * float(scale)
        logits = logits.masked_fill(~mask.reshape(1, -1), float("-inf"))
        out[head_start:head_end] = torch.logsumexp(logits, dim=-1)
    return out, int(total_base)


def _gpu_proxy_confidence_metrics(
    *,
    ranked_scores: torch.Tensor,
    exact_ranked_logits: torch.Tensor,
    keep_count: int | torch.Tensor,
    max_budget: int,
    query_dim: int,
    base_logsumexp: torch.Tensor | None,
    calibrate: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rank_count = int(ranked_scores.shape[-1])
    max_budget_i = max(0, min(rank_count, int(max_budget)))
    leading = ranked_scores.shape[:-1]
    if torch.is_tensor(keep_count):
        keep_i = keep_count.to(device=ranked_scores.device, dtype=torch.long)
        if tuple(keep_i.shape) != tuple(leading):
            try:
                keep_i = torch.broadcast_to(keep_i, leading)
            except RuntimeError as exc:
                raise ValueError(
                    f"keep_count shape {tuple(keep_i.shape)} cannot broadcast to {tuple(leading)}"
                ) from exc
        keep_i = torch.clamp(keep_i, min=0, max=max_budget_i)
    else:
        keep_i_scalar = max(0, min(max_budget_i, int(keep_count)))
        keep_i = torch.full(leading, keep_i_scalar, dtype=torch.long, device=ranked_scores.device)
    if max_budget_i <= 0:
        zeros = torch.zeros(leading, dtype=torch.float32, device=ranked_scores.device)
        infs = torch.full(leading, float("inf"), dtype=torch.float32, device=ranked_scores.device)
        return zeros, zeros, zeros, infs
    scores = ranked_scores[..., :max_budget_i].float()
    exact = exact_ranked_logits[..., :max_budget_i].float()
    finite = torch.isfinite(scores)
    ranks = torch.arange(max_budget_i, dtype=torch.long, device=ranked_scores.device).reshape(
        *((1,) * (scores.dim() - 1)),
        max_budget_i,
    )
    keep_i_expanded = keep_i.reshape(*leading, 1)
    selected_mask = finite & (ranks < keep_i_expanded)
    tail_mask = finite & (ranks >= keep_i_expanded)
    pq_logits = scores * (float(query_dim) ** -0.5)
    count = selected_mask.sum(dim=-1).to(torch.float32)
    selected_mask_f = selected_mask.to(torch.float32)
    safe_count = torch.clamp(count, min=1.0)

    if bool(calibrate):
        x_sum = torch.sum(torch.where(selected_mask, pq_logits, torch.zeros_like(pq_logits)), dim=-1)
        y_sum = torch.sum(torch.where(selected_mask, exact, torch.zeros_like(exact)), dim=-1)
        mean_x = x_sum / safe_count
        mean_y = y_sum / safe_count
        dx = torch.where(selected_mask, pq_logits - mean_x.unsqueeze(-1), torch.zeros_like(pq_logits))
        dy = torch.where(selected_mask, exact - mean_y.unsqueeze(-1), torch.zeros_like(exact))
        var_x = torch.sum(dx * dx, dim=-1) / safe_count
        var_y = torch.sum(dy * dy, dim=-1) / safe_count
        cov = torch.sum(dx * dy, dim=-1) / safe_count
        fit_scale = cov / torch.clamp(var_x, min=1.0e-20)
        fit_bias = mean_y - fit_scale * mean_x
        flat_case = (var_x <= 1.0e-20) & (count >= 2.0)
        fit_scale = torch.where(flat_case, torch.zeros_like(fit_scale), fit_scale)
        fit_bias = torch.where(flat_case, mean_y, fit_bias)
        bad_scale = ((fit_scale <= 0.0) | ~torch.isfinite(fit_scale)) & ~flat_case
        fit_scale = torch.where(bad_scale, torch.ones_like(fit_scale), fit_scale)
        fit_bias = torch.where(bad_scale, torch.zeros_like(fit_bias), fit_bias)
        fit_scale = torch.where(count >= 2.0, fit_scale, torch.ones_like(fit_scale))
        fit_bias = torch.where(count >= 2.0, fit_bias, torch.zeros_like(fit_bias))
        pred = fit_scale.unsqueeze(-1) * pq_logits + fit_bias.unsqueeze(-1)
        rmse = torch.sqrt(torch.sum(((pred - exact) ** 2) * selected_mask_f, dim=-1) / safe_count)
        relrmse = rmse / torch.clamp(torch.sqrt(var_y), min=1.0e-6)
        relrmse = torch.where(count >= 2.0, relrmse, torch.full_like(relrmse, float("inf")))
        corr = cov / torch.sqrt(torch.clamp(var_x * var_y, min=1.0e-20))
        corr = torch.where((count >= 2.0) & torch.isfinite(corr), corr, torch.zeros_like(corr))
    else:
        fit_scale = torch.ones(ranked_scores.shape[:-1], dtype=torch.float32, device=ranked_scores.device)
        fit_bias = torch.zeros_like(fit_scale)
        corr = torch.zeros_like(fit_scale)
        relrmse = torch.full_like(fit_scale, float("inf"))

    selected_logits = torch.where(selected_mask, exact, torch.full_like(exact, float("-inf")))
    selected_lse = torch.logsumexp(selected_logits, dim=-1)
    if base_logsumexp is not None:
        selected_lse = torch.logaddexp(selected_lse, base_logsumexp.float())
    tail_logits = fit_scale.unsqueeze(-1) * pq_logits + fit_bias.unsqueeze(-1)
    tail_logits = torch.where(tail_mask, tail_logits, torch.full_like(tail_logits, float("-inf")))
    tail_lse = torch.logsumexp(tail_logits, dim=-1)
    total_lse = torch.logaddexp(selected_lse, tail_lse)
    selected_mass = torch.where(
        torch.isfinite(total_lse),
        torch.exp(selected_lse - total_lse),
        torch.zeros_like(total_lse),
    )
    tail_mass = torch.where(
        torch.isfinite(total_lse),
        torch.exp(tail_lse - total_lse),
        torch.zeros_like(total_lse),
    )
    return selected_mass, tail_mass, corr, relrmse


def make_needle_prompt(target: str, filler_repeats: int) -> str:
    filler = "\n".join(
        f"Background line {i:04d}: this line is irrelevant context about calendars, rivers, and copper."
        for i in range(int(filler_repeats))
    )
    return (
        "You are given a long document. Find the secret code and answer with only the code.\n\n"
        f"{filler}\n\n"
        f"IMPORTANT FACT: the secret code is {target}.\n\n"
        f"{filler}\n\n"
        "Question: What is the secret code?\nAnswer:"
    )


def greedy_dense_trace(model, input_ids: torch.Tensor, max_new_tokens: int, forbidden: set[int]) -> dict[str, Any]:
    logits_trace: list[torch.Tensor] = []
    hidden_trace: list[torch.Tensor] = []
    tokens: list[int] = []
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True, output_hidden_states=True, return_dict=True)
        past = out.past_key_values
        logits = out.logits[:, -1, :]
        for step in range(int(max_new_tokens)):
            if forbidden:
                logits = logits.clone()
                logits[:, list(forbidden)] = torch.finfo(logits.dtype).min
            logits_trace.append(logits.detach().float().cpu())
            hidden_trace.append(out.hidden_states[-1][:, -1, :].detach().float().cpu())
            next_tok = int(torch.argmax(logits, dim=-1).item())
            tokens.append(next_tok)
            if step == int(max_new_tokens) - 1:
                break
            cur = torch.tensor([[next_tok]], dtype=torch.long, device=input_ids.device)
            out = model(input_ids=cur, past_key_values=past, use_cache=True, output_hidden_states=True, return_dict=True)
            past = out.past_key_values
            logits = out.logits[:, -1, :]
    return {"tokens": tokens, "logits": logits_trace, "hidden": hidden_trace}


def teacher_forced_trace(model, input_ids: torch.Tensor, forced_tokens: list[int], forbidden: set[int]) -> dict[str, Any]:
    logits_trace: list[torch.Tensor] = []
    hidden_trace: list[torch.Tensor] = []
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True, output_hidden_states=True, return_dict=True)
        past = out.past_key_values
        logits = out.logits[:, -1, :]
        for step, tok in enumerate(forced_tokens):
            if forbidden:
                logits = logits.clone()
                logits[:, list(forbidden)] = torch.finfo(logits.dtype).min
            logits_trace.append(logits.detach().float().cpu())
            hidden_trace.append(out.hidden_states[-1][:, -1, :].detach().float().cpu())
            if step == len(forced_tokens) - 1:
                break
            cur = torch.tensor([[int(tok)]], dtype=torch.long, device=input_ids.device)
            out = model(input_ids=cur, past_key_values=past, use_cache=True, output_hidden_states=True, return_dict=True)
            past = out.past_key_values
            logits = out.logits[:, -1, :]
    return {"logits": logits_trace, "hidden": hidden_trace}


def summarize_logit_trace(dense: dict[str, Any], approx: dict[str, Any], tokenizer, ignore_token_ids: set[int] | None = None) -> dict[str, Any]:
    ignore_token_ids = set(ignore_token_ids or set())
    rows = []
    for step, (dl, al, dh, ah) in enumerate(zip(dense["logits"], approx["logits"], dense["hidden"], approx["hidden"], strict=False)):
        dl = dl.reshape(-1).float()
        al = al.reshape(-1).float()
        if ignore_token_ids:
            keep = torch.ones_like(dl, dtype=torch.bool)
            valid_ids = [tok for tok in ignore_token_ids if 0 <= int(tok) < int(keep.numel())]
            if valid_ids:
                keep[torch.as_tensor(valid_ids, dtype=torch.long)] = False
                dl_metric = dl[keep]
                al_metric = al[keep]
            else:
                dl_metric = dl
                al_metric = al
        else:
            dl_metric = dl
            al_metric = al
        dh = dh.reshape(-1).float()
        ah = ah.reshape(-1).float()
        dense_top = int(torch.argmax(dl).item())
        approx_top = int(torch.argmax(al).item())
        probs_d = torch.softmax(dl, dim=-1)
        log_probs_a = torch.log_softmax(al, dim=-1)
        kl = torch.sum(probs_d * (torch.log(torch.clamp(probs_d, min=1e-30)) - log_probs_a)).item()
        logit_diff = torch.linalg.vector_norm((dl_metric - al_metric).double()).item()
        logit_norm = torch.linalg.vector_norm(dl_metric.double()).item()
        logit_dot = torch.dot(dl_metric.double(), al_metric.double()).item()
        approx_norm = torch.linalg.vector_norm(al_metric.double()).item()
        logit_cos = logit_dot / max(1e-30, logit_norm * approx_norm)
        rows.append(
            {
                "step": int(step),
                "dense_top": dense_top,
                "approx_top": approx_top,
                "top1_match": bool(dense_top == approx_top),
                "dense_top_text": tokenizer.decode([dense_top]),
                "approx_top_text": tokenizer.decode([approx_top]),
                "logit_l2": float(logit_diff),
                "logit_relative_l2": float(logit_diff / max(1e-30, logit_norm)),
                "logit_cosine": float(logit_cos),
                "dense_to_approx_kl": float(kl),
                "hidden_relative_l2": float(torch.linalg.vector_norm(dh - ah) / torch.clamp(torch.linalg.vector_norm(dh), min=1e-20)),
                "hidden_cosine": float(F.cosine_similarity(dh.unsqueeze(0), ah.unsqueeze(0), dim=-1).item()),
            }
        )
    if not rows:
        return {"steps": [], "summary": {}}
    summary = {
        "steps": int(len(rows)),
        "top1_agreement": float(np.mean([float(r["top1_match"]) for r in rows])),
        "mean_logit_relative_l2": float(np.mean([r["logit_relative_l2"] for r in rows])),
        "max_logit_relative_l2": float(np.max([r["logit_relative_l2"] for r in rows])),
        "mean_logit_cosine": float(np.mean([r["logit_cosine"] for r in rows])),
        "min_logit_cosine": float(np.min([r["logit_cosine"] for r in rows])),
        "mean_dense_to_approx_kl": float(np.mean([r["dense_to_approx_kl"] for r in rows])),
        "max_dense_to_approx_kl": float(np.max([r["dense_to_approx_kl"] for r in rows])),
        "mean_hidden_relative_l2": float(np.mean([r["hidden_relative_l2"] for r in rows])),
        "max_hidden_relative_l2": float(np.max([r["hidden_relative_l2"] for r in rows])),
        "mean_hidden_cosine": float(np.mean([r["hidden_cosine"] for r in rows])),
        "min_hidden_cosine": float(np.min([r["hidden_cosine"] for r in rows])),
    }
    affected = [r for r in rows if int(r["step"]) > 0]
    if affected:
        summary.update(
            {
                "affected_steps": int(len(affected)),
                "affected_top1_agreement": float(np.mean([float(r["top1_match"]) for r in affected])),
                "affected_mean_logit_relative_l2": float(np.mean([r["logit_relative_l2"] for r in affected])),
                "affected_max_logit_relative_l2": float(np.max([r["logit_relative_l2"] for r in affected])),
                "affected_mean_dense_to_approx_kl": float(np.mean([r["dense_to_approx_kl"] for r in affected])),
                "affected_max_dense_to_approx_kl": float(np.max([r["dense_to_approx_kl"] for r in affected])),
                "affected_mean_hidden_relative_l2": float(np.mean([r["hidden_relative_l2"] for r in affected])),
                "affected_max_hidden_relative_l2": float(np.max([r["hidden_relative_l2"] for r in affected])),
            }
        )
    else:
        summary["affected_steps"] = 0
    return {"steps": rows, "summary": summary}


@contextlib.contextmanager
def patched_paged_pq_attention(model, layer_ids: list[int], args, stats: dict[int, ApproxStats]):
    originals = {}
    device = next(model.parameters()).device
    _require_canonical_gpu_frontier(args)
    key_bytes = int(args.key_bytes)
    value_bytes = int(args.value_bytes)
    nprobes = parse_csv_ints(args.nprobes)
    budget_by_head = parse_head_budget_map(args.budget_by_head)
    tail_off_heads = parse_int_set(args.tail_off_heads)
    online_confidence_rule = str(getattr(args, "online_confidence_rule", "none"))
    if online_confidence_rule not in {
        "none",
        "geometric_probe_tail_switch",
        "geometric_tail_stability_switch",
        "geometric_exact_delta",
        "joint_kv_stability",
        "pq_proxy_mass_budget",
        "pq_ranked_mass_budget",
    }:
        raise ValueError(f"unsupported online_confidence_rule in HF runner: {online_confidence_rule}")
    ranked_confidence_cost_mode = str(getattr(args, "ranked_confidence_cost_mode", "exact"))
    if ranked_confidence_cost_mode not in {"exact", "upper_bound"}:
        raise ValueError(f"unsupported ranked_confidence_cost_mode: {ranked_confidence_cost_mode}")
    exact_logit_backend = str(getattr(args, "exact_logit_backend", "auto"))
    if exact_logit_backend not in {"auto", "ranked_gather", "dense_sim"}:
        raise ValueError(f"unsupported exact_logit_backend: {exact_logit_backend}")
    try:
        dense_sim_max_context_ratio = max(
            0.0,
            float(os.environ.get("FRONTIER_DENSE_SIM_MAX_CONTEXT_RATIO", "2.0")),
        )
    except ValueError:
        dense_sim_max_context_ratio = 2.0
    last_decode_base_key: tuple[int, int, int, int, int] | None = None
    last_decode_base_tensor: torch.Tensor | None = None
    last_decode_rank_ids_tensors: dict[tuple[str, int, int], torch.Tensor] = {}
    geometric_budget_column_tensors: dict[
        tuple[str, int, int, int, float, float],
        tuple[list[int], list[int], list[int], torch.Tensor, torch.Tensor],
    ] = {}
    dense_decode_key_t_cache: dict[int, dict[str, object]] = {}
    try:
        dense_decode_key_t_cache_max_bytes = int(
            max(0.0, float(os.environ.get("FRONTIER_DENSE_KEY_T_CACHE_MAX_GB", "12.0")))
            * 1024.0
            * 1024.0
            * 1024.0
        )
    except ValueError:
        dense_decode_key_t_cache_max_bytes = 12 * 1024 * 1024 * 1024
    dense_decode_key_t_cache_enabled = _env_truthy("FRONTIER_DENSE_KEY_T_CACHE", "1")

    def dense_decode_key_t_float_cache(
        *,
        layer_id: int,
        keys_all: torch.Tensor,
        key_count: int,
    ) -> torch.Tensor | None:
        if (
            not dense_decode_key_t_cache_enabled
            or dense_decode_key_t_cache_max_bytes <= 0
            or keys_all.device.type != "cuda"
        ):
            return None
        kv_heads = int(keys_all.shape[0])
        capacity = int(keys_all.shape[1])
        dim = int(keys_all.shape[2])
        key_count_i = min(max(0, int(key_count)), capacity)
        if kv_heads <= 0 or capacity <= 0 or dim <= 0 or key_count_i <= 0:
            return None
        # This cache is a GPU-simulator optimization only. Keep a conservative
        # cap so long-context task runs do not OOM by caching every layer's
        # float-transposed K unless the user explicitly raises the limit.
        all_layers_bytes = int(max(1, len(layer_ids))) * kv_heads * capacity * dim * 4
        if all_layers_bytes > dense_decode_key_t_cache_max_bytes:
            return None
        entry = dense_decode_key_t_cache.get(int(layer_id))
        data_ptr = int(keys_all.data_ptr())
        shape = (kv_heads, capacity, dim)
        if (
            entry is None
            or int(entry.get("data_ptr", -1)) != data_ptr
            or tuple(entry.get("shape", ())) != shape
            or str(entry.get("device", "")) != str(keys_all.device)
        ):
            entry = {
                "data_ptr": data_ptr,
                "shape": shape,
                "device": str(keys_all.device),
                "filled": 0,
                "tensor": torch.empty((kv_heads, dim, capacity), dtype=torch.float32, device=keys_all.device),
            }
            dense_decode_key_t_cache[int(layer_id)] = entry
        filled = int(entry.get("filled", 0))
        if key_count_i < filled:
            filled = 0
        if key_count_i > filled:
            cached = entry["tensor"]
            assert isinstance(cached, torch.Tensor)
            cached[:, :, filled:key_count_i].copy_(
                keys_all[:, filled:key_count_i, :].float().transpose(1, 2).contiguous()
            )
            entry["filled"] = key_count_i
        cached = entry["tensor"]
        assert isinstance(cached, torch.Tensor)
        return cached

    def decode_base_tokens_tensor(query_context_len: int, sealed_end: int, indexed_end: int) -> torch.Tensor:
        nonlocal last_decode_base_key, last_decode_base_tensor
        cache_key = (
            int(query_context_len),
            int(sealed_end),
            int(indexed_end),
            int(args.static_prefix),
            int(args.static_suffix),
        )
        if last_decode_base_key == cache_key and last_decode_base_tensor is not None:
            return last_decode_base_tensor
        base = unique_tokens(
            static_tokens(int(query_context_len) - 1, int(args.static_prefix), int(args.static_suffix))
            + list(range(max(0, int(sealed_end)), max(0, min(int(indexed_end), int(query_context_len))))),
            context_len=int(query_context_len),
        )
        last_decode_base_key = cache_key
        last_decode_base_tensor = torch.as_tensor(np.asarray(base, dtype=np.int64), dtype=torch.long, device=device)
        return last_decode_base_tensor

    def decode_rank_ids_tensor(rank_count: int, tensor_device: torch.device, *, dims: int = 2) -> torch.Tensor:
        dims_i = int(dims)
        if dims_i not in {2, 3}:
            raise ValueError(f"unsupported rank id dims: {dims}")
        cache_key = (str(tensor_device), int(rank_count), dims_i)
        cached = last_decode_rank_ids_tensors.get(cache_key)
        if cached is not None:
            return cached
        shape = (1, int(rank_count)) if dims_i == 2 else (1, 1, int(rank_count))
        tensor = torch.arange(
            int(rank_count),
            dtype=torch.long,
            device=tensor_device,
        ).reshape(*shape)
        last_decode_rank_ids_tensors[cache_key] = tensor
        return tensor

    def geometric_threshold_budget_columns(
        *,
        min_budget: int,
        max_budget: int,
        granularity: int,
        growth: float,
        probe_scale: float,
        tensor_device: torch.device,
    ) -> tuple[list[int], list[int], list[int], torch.Tensor, torch.Tensor]:
        cache_key = (
            str(tensor_device),
            int(min_budget),
            int(max_budget),
            int(granularity),
            float(growth),
            float(probe_scale),
        )
        cached = geometric_budget_column_tensors.get(cache_key)
        if cached is not None:
            return cached
        tail_budgets, probe_budgets = geometric_budget_pairs(
            min_budget=int(min_budget),
            max_budget=int(max_budget),
            granularity=int(granularity),
            growth=float(growth),
            probe_scale=float(probe_scale),
        )
        combined_budgets = sorted({int(v) for v in tail_budgets} | {int(v) for v in probe_budgets})
        budget_to_col = {int(budget): int(idx) for idx, budget in enumerate(combined_budgets)}
        approx_cols = torch.tensor(
            [budget_to_col[int(v)] for v in tail_budgets],
            dtype=torch.long,
            device=tensor_device,
        )
        probe_cols = torch.tensor(
            [budget_to_col[int(v)] for v in probe_budgets],
            dtype=torch.long,
            device=tensor_device,
        )
        cached = (tail_budgets, probe_budgets, combined_budgets, approx_cols, probe_cols)
        geometric_budget_column_tensors[cache_key] = cached
        return cached

    def decode_base_token_count(query_context_len: int, sealed_end: int) -> int:
        prefix_end = min(max(0, int(args.static_prefix)), int(query_context_len))
        base_tail_start = max(int(sealed_end), int(prefix_end))
        return int(prefix_end) + max(0, int(query_context_len) - int(base_tail_start))

    def joint_vpq_cache_key_for(kv_head: int, values_t: torch.Tensor, index: GPUIndex) -> tuple[object, ...]:
        actual_value_subbits_key = int(args.value_subbits) if int(args.value_subbits) > 0 else int(args.subbits)
        return (
            int(kv_head),
            str(values_t.device),
            int(args.subbits),
            int(args.value_subvecs),
            int(actual_value_subbits_key),
            int(value_bytes),
            int(len(index.pages)),
            int(index.pages[0].start) if index.pages else -1,
            (int(index.pages[-1].start) + int(index.pages[-1].size)) if index.pages else -1,
            int(index.pages[0].size) if index.pages else 0,
        )

    def warm_dense_prefill_decode_sidecars(layer_id: int, module, cache_obj) -> None:
        if bool(getattr(args, "skip_prefill_index_build", False)):
            return
        if str(args.selector_mode) not in {"fullscan", "routed", "oracle"}:
            return
        num_kv_heads = getattr(module, "num_key_value_heads", None)
        if num_kv_heads is None:
            config = getattr(module, "config", None)
            num_kv_heads = getattr(config, "num_key_value_heads", None)
        if num_kv_heads is None:
            return
        num_kv_heads = int(num_kv_heads)
        kv = cache_layer_kv_tensors(cache_obj, int(layer_id), num_kv_heads=num_kv_heads)
        if kv is None:
            return
        keys_all, values_all = kv
        if keys_all.ndim != 3 or values_all.ndim != 3 or int(keys_all.shape[0]) != num_kv_heads:
            return
        context_len = int(keys_all.shape[1])
        dynamic_start = min(max(0, int(args.static_prefix)), context_len)
        indexed_end = max(dynamic_start, context_len - max(0, int(args.static_suffix)))
        sealed_end = dynamic_start + (
            (max(0, indexed_end - dynamic_start) // max(1, int(args.page_size))) * max(1, int(args.page_size))
        )
        page_cache = getattr(module, "_pagedpq_page_cache", None)
        if not isinstance(page_cache, dict):
            page_cache = {}
            setattr(module, "_pagedpq_page_cache", page_cache)
        decode_tail_blend = (
            float(args.decode_tail_blend)
            if getattr(args, "decode_tail_blend", None) is not None
            else float(args.tail_blend)
        )
        needs_vpq_sidecar = (
            str(args.selected_value_mode) == "vpq_value"
            or (float(decode_tail_blend) > 0.0 and str(args.tail_mode) == "vpq_value")
        )
        warmed_indexes: list[GPUIndex | None] = [None] * num_kv_heads
        for kv_head in range(num_kv_heads):
            if str(args.selector_mode) in {"fullscan", "oracle"}:
                cache_key = (
                    "online_fullscan",
                    int(kv_head),
                    int(dynamic_start),
                    int(args.page_size),
                    int(args.subvecs),
                    int(args.subbits),
                    int(args.kmeans_iters),
                    int(args.seed),
                    int(key_bytes),
                    str(getattr(args, "index_build_backend", "numpy")),
                )
            else:
                cache_key = (
                    int(kv_head),
                    int(dynamic_start),
                    int(sealed_end),
                    int(args.page_size),
                    int(args.subvecs),
                    int(args.subbits),
                    int(args.kmeans_iters),
                    int(args.seed),
                    int(key_bytes),
                    str(getattr(args, "index_build_backend", "numpy")),
                    str(args.selector_mode),
                    int(args.router_prototypes),
                    float(args.router_merge_rel),
                    float(args.router_merge_var),
                    int(args.router_max_groups),
                )
            cached_index = page_cache.get(cache_key)
            if cached_index is None:
                cached_index, build_seconds, build_read_mb, build_write_mb = build_page_pq_from_keys(
                    keys_all[int(kv_head)].detach(),
                    args=args,
                    kv_head=int(kv_head),
                    dynamic_start=dynamic_start,
                    indexed_end=sealed_end,
                    key_bytes=key_bytes,
                    router_enabled=str(args.selector_mode) == "routed",
                    device=device,
                )
                stats[int(layer_id)].add_index_build(build_seconds, build_read_mb, build_write_mb)
                page_cache[cache_key] = cached_index
            elif str(args.selector_mode) in {"fullscan", "oracle"} and int(cached_index.pending_start) < int(sealed_end):
                new_index, build_seconds, build_read_mb, build_write_mb = build_page_pq_from_keys(
                    keys_all[int(kv_head)].detach(),
                    args=args,
                    kv_head=int(kv_head),
                    dynamic_start=int(cached_index.pending_start),
                    indexed_end=sealed_end,
                    key_bytes=key_bytes,
                    router_enabled=False,
                    device=device,
                    page_id_offset=len(cached_index.pages),
                )
                stats[int(layer_id)].add_index_build(build_seconds, build_read_mb, build_write_mb)
                cached_index.pages.extend(new_index.pages)
                cached_index.pending_start = int(sealed_end)
                cached_index.indexed_end = int(sealed_end)
                cached_index.build_seconds += float(new_index.build_seconds)
                cached_index.build_read_mb += float(new_index.build_read_mb)
                cached_index.build_write_mb += float(new_index.build_write_mb)
                cached_index.native_codebooks = None
                cached_index.native_codes = None
                cached_index.native_page_starts = None
                for attr in (
                    "_value_vpq_gpu_pack_by_params",
                    "_all_value_vpq_gpu_by_params",
                    "_value_vpq_sidecars_by_params",
                ):
                    if hasattr(cached_index, attr):
                        delattr(cached_index, attr)
            if str(args.selector_mode) in {"fullscan", "oracle"}:
                cached_index.pending_start = int(sealed_end)
                cached_index.indexed_end = int(indexed_end)
                warmed_indexes[int(kv_head)] = cached_index
            if (
                needs_vpq_sidecar
                and str(getattr(args, "index_build_backend", "numpy")) == "torch_gpu"
                and cached_index.pages
            ):
                value_vpq_pack_torch(
                    index=cached_index,
                    values=values_all[int(kv_head)].detach(),
                    value_subvecs=int(args.value_subvecs),
                    value_subbits=int(args.value_subbits) if int(args.value_subbits) > 0 else int(args.subbits),
                    key_bytes=value_bytes,
                    device=device,
                    value_group_pages=int(getattr(args, "value_pq_group_pages", 1)),
                )
                build_stats = getattr(cached_index, "_last_value_vpq_build_stats", None)
                if build_stats is not None:
                    stats[int(layer_id)].add_index_build(*build_stats)
                if _env_truthy("SELECTOR_PQ_JOINT_PREWARM_VPQ_SIDECARS", "0"):
                    persistent_cache = getattr(module, "_pagedpq_joint_vpq_sidecar_cache", None)
                    if not isinstance(persistent_cache, dict):
                        persistent_cache = {}
                        setattr(module, "_pagedpq_joint_vpq_sidecar_cache", persistent_cache)
                    values_t = values_all[int(kv_head)].detach()
                    context_len_i = int(values_t.shape[0])
                    cache_key = joint_vpq_cache_key_for(int(kv_head), values_t, cached_index)
                    if cache_key not in persistent_cache:
                        sidecar_t0 = time.perf_counter()
                        all_tokens_t = torch.arange(context_len_i, dtype=torch.long, device=values_t.device)
                        vhat_all_t, vpq_valid_t, vpq_page_ids_t, actual_value_subbits_for_cost = vpq_values_for_tokens_gpu(
                            index=cached_index,
                            values=values_t,
                            values_np=None,
                            tokens=all_tokens_t,
                            subbits=int(args.subbits),
                            value_subvecs=int(args.value_subvecs),
                            value_subbits=int(args.value_subbits),
                            prefer_torch=True,
                            value_bytes=int(value_bytes),
                        )
                        residual_t = values_t.float() - vhat_all_t.float()
                        code_error_t, actual_value_subbits_for_cost = value_vpq_code_stat_risk_torch(
                            index=cached_index,
                            values=values_t,
                            vhat_all=vhat_all_t,
                            residual_all=residual_t,
                            valid=vpq_valid_t,
                            page_ids=vpq_page_ids_t,
                            subbits=int(args.subbits),
                            value_subvecs=int(args.value_subvecs),
                            value_subbits=int(args.value_subbits),
                            value_bytes=int(value_bytes),
                        )
                        cache_len_i = int(context_len_i)
                        grow_pad_i = max(
                            0,
                            _env_int("SELECTOR_PQ_JOINT_PERSISTENT_VPQ_CACHE_GROW_PAD", 256),
                        )
                        cache_capacity_i = int(cache_len_i + grow_pad_i)
                        if cache_capacity_i > cache_len_i:
                            vhat_buf_t = torch.empty(
                                (cache_capacity_i, int(vhat_all_t.shape[1])),
                                dtype=vhat_all_t.dtype,
                                device=vhat_all_t.device,
                            )
                            residual_buf_t = torch.empty(
                                (cache_capacity_i, int(residual_t.shape[1])),
                                dtype=residual_t.dtype,
                                device=residual_t.device,
                            )
                            code_error_buf_t = torch.empty(
                                (cache_capacity_i,),
                                dtype=code_error_t.dtype,
                                device=code_error_t.device,
                            )
                            vhat_buf_t[:cache_len_i].copy_(vhat_all_t)
                            residual_buf_t[:cache_len_i].copy_(residual_t)
                            code_error_buf_t[:cache_len_i].copy_(code_error_t)
                            vhat_all_t = vhat_buf_t
                            residual_t = residual_buf_t
                            code_error_t = code_error_buf_t
                        if bool(getattr(args, "profile_native_ops", False)) and device.type == "cuda":
                            torch.cuda.synchronize(device)
                        stats[int(layer_id)].add_index_sidecar_timing(time.perf_counter() - sidecar_t0)
                        persistent_cache[cache_key] = (
                            int(cache_len_i),
                            int(cache_capacity_i),
                            vhat_all_t.detach(),
                            residual_t.detach(),
                            code_error_t.detach(),
                            int(actual_value_subbits_for_cost),
                        )
        if (
            str(args.selector_mode) in {"fullscan", "oracle"}
            and str(args.selector_backend) == "cuda_ext"
            and all(index is not None and index.pages for index in warmed_indexes)
        ):
            fast_decode_index_cache_key = (
                "fullscan_decode",
                str(online_confidence_rule),
                int(dynamic_start),
                int(sealed_end),
                int(args.page_size),
                int(args.subvecs),
                int(args.subbits),
                int(args.kmeans_iters),
                int(args.seed),
                int(key_bytes),
                str(getattr(args, "index_build_backend", "numpy")),
                str(args.selected_value_mode),
                str(args.tail_mode),
                int(args.value_subvecs),
                int(args.value_subbits),
                int(args.value_pq_group_pages),
                int(value_bytes),
                int(num_kv_heads),
            )
            fast_decode_index_cache = getattr(module, "_pagedpq_fast_decode_index_cache", None)
            if not isinstance(fast_decode_index_cache, dict):
                fast_decode_index_cache = {}
                setattr(module, "_pagedpq_fast_decode_index_cache", fast_decode_index_cache)
            fast_decode_index_cache.clear()
            fast_decode_index_cache[fast_decode_index_cache_key] = tuple(
                index for index in warmed_indexes if index is not None
            )

    def make_forward(layer_id: int, module):
        original_forward = module.forward

        def forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings,
            attention_mask,
            past_key_value=None,
            cache_position=None,
            **kwargs,
        ):
            past_key_values = kwargs.pop("past_key_values", None)
            cache_obj = past_key_value if past_key_value is not None else past_key_values

            def call_original_forward():
                original_kwargs = dict(kwargs)
                if past_key_value is not None:
                    original_kwargs["past_key_value"] = past_key_value
                if past_key_values is not None:
                    original_kwargs["past_key_values"] = past_key_values
                if cache_position is not None:
                    original_kwargs["cache_position"] = cache_position
                return original_forward(
                    hidden_states=hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                    **original_kwargs,
                )

            input_shape = hidden_states.shape[:-1]
            query_len = int(input_shape[-1])
            approx_prefill = bool(getattr(args, "approx_prefill", False))
            if cache_obj is None:
                stats[layer_id].add_passthrough_attention_call()
                return call_original_forward()
            if query_len != 1 and not approx_prefill:
                stats[layer_id].add_passthrough_attention_call()
                out = call_original_forward()
                warm_dense_prefill_decode_sidecars(int(layer_id), self, cache_obj)
                return out
            if query_len != 1 and str(args.selector_mode) != "fullscan":
                raise RuntimeError("batched prefill approximation currently supports selector_mode=fullscan only")

            if cache_position is not None and torch.numel(cache_position) > 0:
                estimated_context_len = int(cache_position.reshape(-1)[-1].item()) + 1
            else:
                past_len = cache_sequence_length(cache_obj, int(layer_id))
                estimated_context_len = (int(past_len) + query_len) if past_len is not None else query_len
            estimated_dynamic_start = min(max(0, int(args.static_prefix)), estimated_context_len)
            estimated_indexed_end = max(estimated_dynamic_start, estimated_context_len - max(0, int(args.static_suffix)))
            estimated_sealed_end = estimated_dynamic_start + (
                (max(0, estimated_indexed_end - estimated_dynamic_start) // max(1, int(args.page_size)))
                * max(1, int(args.page_size))
            )
            min_budget_est = min(
                int(budget_by_head.get(int(head), int(args.budget)))
                for head in range(int(getattr(self, "num_heads", self.config.num_attention_heads)))
            )
            sealed_indexed_tokens_est = max(0, int(estimated_sealed_end) - int(estimated_dynamic_start))
            estimated_tail_blend = (
                float(args.prefill_tail_blend)
                if query_len > 1 and getattr(args, "prefill_tail_blend", None) is not None
                else (
                    float(args.decode_tail_blend)
                    if query_len == 1 and getattr(args, "decode_tail_blend", None) is not None
                    else float(args.tail_blend)
                )
            )
            dense_equivalent = (
                str(args.selector_mode) == "fullscan"
                and (
                    int(sealed_indexed_tokens_est) <= 0
                    or (
                        int(min_budget_est) >= int(sealed_indexed_tokens_est)
                        and str(args.selected_value_mode) == "exact"
                        and float(estimated_tail_blend) <= 0.0
                    )
                )
            )
            if dense_equivalent:
                num_heads_est = int(getattr(self, "num_heads", self.config.num_attention_heads))
                query_start_est = int(estimated_context_len) - int(query_len)
                for local_qpos_est in range(query_len):
                    query_context_len_est = int(query_start_est + local_qpos_est + 1)
                    stats[layer_id].add_count_repeated(
                        num_heads_est,
                        int(query_context_len_est),
                        0,
                        0.0,
                        int(self.head_dim),
                        key_bytes,
                        value_bytes,
                    )
                stats[layer_id].add_passthrough_attention_call()
                return call_original_forward()

            stats[layer_id].add_approx_attention_call()
            wall_profile_enabled = _env_truthy("SELECTOR_PQ_JOINT_WALL_PROFILE", "0")
            patched_attention_wall_t0 = time.perf_counter() if wall_profile_enabled else 0.0
            if bool(getattr(args, "profile_native_ops", False)):
                _sync_if_cuda(device)
                patched_attention_t0 = time.perf_counter()
            else:
                patched_attention_t0 = 0.0

            qkv_cache_wall_t0 = time.perf_counter() if wall_profile_enabled else 0.0
            if bool(getattr(args, "profile_native_ops", False)):
                _sync_if_cuda(device)
                qkv_cache_t0 = time.perf_counter()
            else:
                qkv_cache_t0 = 0.0
            hidden_shape = (*input_shape, -1, self.head_dim)
            query_states = self.q_proj(hidden_states).view(hidden_shape)
            if hasattr(self, "q_norm"):
                query_states = self.q_norm(query_states)
            query_states = query_states.transpose(1, 2)
            key_states = self.k_proj(hidden_states).view(hidden_shape)
            if hasattr(self, "k_norm"):
                key_states = self.k_norm(key_states)
            key_states = key_states.transpose(1, 2)
            value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            try:
                key_states, value_states = cache_obj.update(key_states, value_states, self.layer_idx, cache_kwargs)
            except TypeError:
                key_states, value_states = cache_obj.update(key_states, value_states, self.layer_idx)
            if bool(getattr(args, "profile_native_ops", False)):
                _sync_if_cuda(device)
                stats[layer_id].add_qkv_cache_timing(time.perf_counter() - qkv_cache_t0)
            if wall_profile_enabled:
                stats[layer_id].add_wall_qkv_cache_timing(time.perf_counter() - qkv_cache_wall_t0)

            keys_all = key_states[0].detach()
            values_all = value_states[0].detach()
            q_all = query_states[0].detach().to(torch.float32)
            context_len = int(keys_all.shape[1])
            query_start = context_len - query_len
            num_heads = int(getattr(self, "num_heads", self.config.num_attention_heads))
            num_kv_heads = int(getattr(self, "num_key_value_heads", self.config.num_key_value_heads))
            group_size = int(num_heads // num_kv_heads)
            base_tail_blend = float(args.tail_blend)
            tail_blend_value = (
                float(args.prefill_tail_blend)
                if query_len > 1 and getattr(args, "prefill_tail_blend", None) is not None
                else (
                    float(args.decode_tail_blend)
                    if query_len == 1 and getattr(args, "decode_tail_blend", None) is not None
                    else base_tail_blend
                )
            )
            keys_all_f32_cache: torch.Tensor | None = None
            values_all_f32_cache: torch.Tensor | None = None

            def keys_all_float() -> torch.Tensor:
                nonlocal keys_all_f32_cache
                if keys_all.dtype == torch.float32:
                    return keys_all
                if keys_all_f32_cache is None:
                    if bool(getattr(args, "profile_native_ops", False)):
                        _sync_if_cuda(device)
                        cast_t0 = time.perf_counter()
                    keys_all_f32_cache = keys_all.to(torch.float32)
                    if bool(getattr(args, "profile_native_ops", False)):
                        _sync_if_cuda(device)
                        stats[layer_id].add_cache_cast_timing(time.perf_counter() - cast_t0)
                return keys_all_f32_cache

            def values_all_float() -> torch.Tensor:
                nonlocal values_all_f32_cache
                if values_all.dtype == torch.float32:
                    return values_all
                if values_all_f32_cache is None:
                    if bool(getattr(args, "profile_native_ops", False)):
                        _sync_if_cuda(device)
                        cast_t0 = time.perf_counter()
                    values_all_f32_cache = values_all.to(torch.float32)
                    if bool(getattr(args, "profile_native_ops", False)):
                        _sync_if_cuda(device)
                        stats[layer_id].add_cache_cast_timing(time.perf_counter() - cast_t0)
                return values_all_f32_cache

            index_cache = {}
            prefix_index_cache: dict[tuple[int, int, int], GPUIndex] = {}
            values_np_cache = {}
            torch_k_cache = {}
            torch_v_cache = {}
            dynamic_start = min(max(0, int(args.static_prefix)), context_len)
            indexed_end = max(dynamic_start, context_len - max(0, int(args.static_suffix)))
            sealed_end = dynamic_start + (
                (max(0, indexed_end - dynamic_start) // max(1, int(args.page_size))) * max(1, int(args.page_size))
            )
            page_cache = getattr(self, "_pagedpq_page_cache", None)
            if not isinstance(page_cache, dict):
                page_cache = {}
                setattr(self, "_pagedpq_page_cache", page_cache)
            selected_vpq_fixed_native = (
                str(args.selected_value_mode) == "vpq_value"
                and str(args.selected_value_exact_rule) in {"fixed", "selector_rank"}
                and int(args.selected_value_min_exact_top) == 0
                and int(args.selected_value_max_exact_top) == 0
                and int(getattr(args, "selected_value_exact_all_context_max", 0)) <= 0
                and float(getattr(args, "selected_value_exact_all_fraction_min", 0.0)) <= 0.0
                and str(getattr(args, "index_build_backend", "numpy")) == "torch_gpu"
            )
            selected_vpq_count_native = (
                str(args.selected_value_mode) == "vpq_value"
                and str(args.selected_value_exact_rule) == "selected_mass"
                and int(getattr(args, "selected_value_exact_all_context_max", 0)) <= 0
                and float(getattr(args, "selected_value_exact_all_fraction_min", 0.0)) <= 0.0
                and str(getattr(args, "index_build_backend", "numpy")) == "torch_gpu"
            )
            selected_vpq_native = selected_vpq_fixed_native or selected_vpq_count_native
            score_confidence_rule = online_confidence_rule in {"pq_proxy_mass_budget", "pq_ranked_mass_budget"}
            fast_tail_vpq_possible = (
                tail_blend_value > 0.0
                and str(args.tail_mode) == "vpq_value"
                and (str(args.selected_value_mode) == "exact" or selected_vpq_native)
                and (
                    math.isinf(float(args.tail_probe_rel_l2_max))
                    or online_confidence_rule == "geometric_probe_tail_switch"
                    or score_confidence_rule
                )
            )
            fast_native_decode_possible = (
                query_len == 1
                and str(args.selector_mode) == "fullscan"
                and str(args.selector_backend) in {"cuda_ext", "auto"}
                and (
                    (
                        tail_blend_value <= 0.0
                        and str(args.selected_value_mode) in {"exact", "vpq_value"}
                    )
                    or fast_tail_vpq_possible
                )
                and int(args.rerank_candidates) == 0
                and not budget_by_head
            )
            strict_native_exact_decode = (
                fast_native_decode_possible
                and str(args.selector_backend) == "cuda_ext"
                and str(args.selected_value_mode) == "exact"
                and not fast_tail_vpq_possible
            )
            joint_fast_decode_index_possible = (
                query_len == 1
                and online_confidence_rule == "joint_kv_stability"
                and str(args.selector_backend) in {"cuda_ext", "auto"}
                and str(args.selector_mode) == "fullscan"
                and str(getattr(args, "index_build_backend", "numpy")) == "torch_gpu"
            )
            fast_decode_index_cache_key: tuple[object, ...] | None = None
            fast_decode_cached_indexes: tuple[GPUIndex, ...] | None = None
            if (
                query_len == 1
                and (fast_native_decode_possible or joint_fast_decode_index_possible)
                and str(args.selector_backend) == "cuda_ext"
                and str(args.selector_mode) == "fullscan"
            ):
                fast_decode_index_cache_key = (
                    "fullscan_decode",
                    str(online_confidence_rule),
                    int(dynamic_start),
                    int(sealed_end),
                    int(args.page_size),
                    int(args.subvecs),
                    int(args.subbits),
                    int(args.kmeans_iters),
                    int(args.seed),
                    int(key_bytes),
                    str(getattr(args, "index_build_backend", "numpy")),
                    str(args.selected_value_mode),
                    str(args.tail_mode),
                    int(args.value_subvecs),
                    int(args.value_subbits),
                    int(args.value_pq_group_pages),
                    int(value_bytes),
                    int(num_kv_heads),
                )
                fast_decode_index_cache = getattr(self, "_pagedpq_fast_decode_index_cache", None)
                if isinstance(fast_decode_index_cache, dict):
                    cached_indexes = fast_decode_index_cache.get(fast_decode_index_cache_key)
                    if isinstance(cached_indexes, tuple) and len(cached_indexes) == int(num_kv_heads):
                        fast_decode_cached_indexes = cached_indexes
            index_sidecar_wall_t0 = time.perf_counter() if wall_profile_enabled else 0.0
            if bool(getattr(args, "profile_native_ops", False)):
                _sync_if_cuda(device)
                sidecar_t0 = time.perf_counter()
            else:
                sidecar_t0 = 0.0
            if fast_decode_cached_indexes is not None:
                index_cache = {}
                for kv_head in range(num_kv_heads):
                    cached_index = fast_decode_cached_indexes[int(kv_head)]
                    if str(args.selector_mode) in {"fullscan", "oracle"}:
                        cached_index.pending_start = int(sealed_end)
                        cached_index.indexed_end = int(indexed_end)
                    index_cache[int(kv_head)] = cached_index
                if not strict_native_exact_decode:
                    for kv_head in range(num_kv_heads):
                        torch_k_cache[int(kv_head)] = keys_all[int(kv_head)].to(device)
                        torch_v_cache[int(kv_head)] = values_all[int(kv_head)].to(device)
            else:
                for kv_head in range(num_kv_heads):
                    if str(args.selector_mode) in {"fullscan", "oracle"}:
                        cache_key = (
                            "online_fullscan",
                            int(kv_head),
                            int(dynamic_start),
                            int(args.page_size),
                            int(args.subvecs),
                            int(args.subbits),
                            int(args.kmeans_iters),
                            int(args.seed),
                            int(key_bytes),
                            str(getattr(args, "index_build_backend", "numpy")),
                        )
                        cached_index = page_cache.get(cache_key)
                        if cached_index is None or int(cached_index.pending_start) > int(sealed_end):
                            cached_index, build_seconds, build_read_mb, build_write_mb = build_page_pq_from_keys(
                                keys_all[kv_head],
                                args=args,
                                kv_head=kv_head,
                                dynamic_start=dynamic_start,
                                indexed_end=sealed_end,
                                key_bytes=key_bytes,
                                router_enabled=False,
                                device=device,
                            )
                            stats[layer_id].add_index_build(build_seconds, build_read_mb, build_write_mb)
                            page_cache[cache_key] = cached_index
                        elif int(cached_index.pending_start) < int(sealed_end):
                            new_index, build_seconds, build_read_mb, build_write_mb = build_page_pq_from_keys(
                                keys_all[kv_head],
                                args=args,
                                kv_head=kv_head,
                                dynamic_start=int(cached_index.pending_start),
                                indexed_end=sealed_end,
                                key_bytes=key_bytes,
                                router_enabled=False,
                                device=device,
                                page_id_offset=len(cached_index.pages),
                            )
                            stats[layer_id].add_index_build(build_seconds, build_read_mb, build_write_mb)
                            cached_index.pages.extend(new_index.pages)
                            cached_index.pending_start = int(sealed_end)
                            cached_index.indexed_end = int(sealed_end)
                            cached_index.build_seconds += float(new_index.build_seconds)
                            cached_index.build_read_mb += float(new_index.build_read_mb)
                            cached_index.build_write_mb += float(new_index.build_write_mb)
                            cached_index.native_codebooks = None
                            cached_index.native_codes = None
                            cached_index.native_page_starts = None
                            for attr in (
                                "_value_vpq_gpu_pack_by_params",
                                "_all_value_vpq_gpu_by_params",
                                "_value_vpq_sidecars_by_params",
                            ):
                                if hasattr(cached_index, attr):
                                    delattr(cached_index, attr)
                    else:
                        cache_key = (
                            int(kv_head),
                            int(dynamic_start),
                            int(sealed_end),
                            int(args.page_size),
                            int(args.subvecs),
                            int(args.subbits),
                            int(args.kmeans_iters),
                            int(args.seed),
                            int(key_bytes),
                            str(getattr(args, "index_build_backend", "numpy")),
                            str(args.selector_mode),
                            int(args.router_prototypes),
                            float(args.router_merge_rel),
                            float(args.router_merge_var),
                            int(args.router_max_groups),
                        )
                        cached_index = page_cache.get(cache_key)
                        if cached_index is None:
                            cached_index, build_seconds, build_read_mb, build_write_mb = build_page_pq_from_keys(
                                keys_all[kv_head],
                                args=args,
                                kv_head=kv_head,
                                dynamic_start=dynamic_start,
                                indexed_end=sealed_end,
                                key_bytes=key_bytes,
                                router_enabled=str(args.selector_mode) == "routed",
                                device=device,
                            )
                            stats[layer_id].add_index_build(build_seconds, build_read_mb, build_write_mb)
                            page_cache[cache_key] = cached_index
                    if str(args.selector_mode) in {"fullscan", "oracle"}:
                        cached_index.pending_start = int(sealed_end)
                        cached_index.indexed_end = int(indexed_end)
                        index_cache[kv_head] = cached_index
                    else:
                        index_cache[kv_head] = GPUIndex(
                            pages=cached_index.pages,
                            pending_start=int(sealed_end),
                            indexed_end=int(indexed_end),
                            build_seconds=float(cached_index.build_seconds),
                            build_read_mb=float(cached_index.build_read_mb),
                            build_write_mb=float(cached_index.build_write_mb),
                            router_group_means=cached_index.router_group_means,
                            router_group_tokens=cached_index.router_group_tokens,
                            router_group_member_refs=cached_index.router_group_member_refs,
                        )
                    prefer_torch_value_pack = str(getattr(args, "index_build_backend", "numpy")) == "torch_gpu"
                    needs_cpu_value_sidecar = (
                        not fast_native_decode_possible
                        or (str(args.selected_value_mode) == "vpq_value" and not prefer_torch_value_pack)
                        or (fast_tail_vpq_possible and not prefer_torch_value_pack)
                    )
                    if needs_cpu_value_sidecar:
                        values_np_cache[kv_head] = (
                            values_all[kv_head].detach().to(torch.float32).cpu().numpy().astype(np.float32, copy=False)
                        )
                    if not strict_native_exact_decode:
                        torch_k_cache[kv_head] = keys_all[kv_head].to(device)
                        torch_v_cache[kv_head] = values_all[kv_head].to(device)
                if fast_decode_index_cache_key is not None:
                    fast_decode_index_cache = getattr(self, "_pagedpq_fast_decode_index_cache", None)
                    if not isinstance(fast_decode_index_cache, dict):
                        fast_decode_index_cache = {}
                        setattr(self, "_pagedpq_fast_decode_index_cache", fast_decode_index_cache)
                    fast_decode_index_cache.clear()
                    fast_decode_index_cache[fast_decode_index_cache_key] = tuple(
                        index_cache[int(kv_head)] for kv_head in range(num_kv_heads)
                    )
            if bool(getattr(args, "profile_native_ops", False)):
                _sync_if_cuda(device)
                stats[layer_id].add_index_sidecar_timing(time.perf_counter() - sidecar_t0)
            if wall_profile_enabled:
                stats[layer_id].add_wall_index_sidecar_timing(time.perf_counter() - index_sidecar_wall_t0)

            def prefix_index_for(kv_head: int, query_context_len: int) -> GPUIndex:
                full_index = index_cache[int(kv_head)]
                if int(query_context_len) >= int(context_len):
                    return full_index
                dyn_start = min(max(0, int(args.static_prefix)), int(query_context_len))
                indexed_end_q = max(dyn_start, int(query_context_len) - max(0, int(args.static_suffix)))
                sealed_end_q = dyn_start + (
                    (max(0, indexed_end_q - dyn_start) // max(1, int(args.page_size))) * max(1, int(args.page_size))
                )
                key = (int(kv_head), int(indexed_end_q), int(sealed_end_q))
                cached = prefix_index_cache.get(key)
                if cached is not None:
                    return cached
                pages = [
                    page
                    for page in full_index.pages
                    if int(page.start) + int(page.size) <= int(sealed_end_q)
                ]
                native_codebooks = None
                native_codes = None
                native_page_starts = None
                if (
                    full_index.native_codebooks is not None
                    and full_index.native_codes is not None
                    and full_index.native_page_starts is not None
                ):
                    page_count = int(len(pages))
                    native_codebooks = full_index.native_codebooks[:page_count]
                    native_codes = full_index.native_codes[:page_count]
                    native_page_starts = full_index.native_page_starts[:page_count]
                view = GPUIndex(
                    pages=pages,
                    pending_start=int(sealed_end_q),
                    indexed_end=int(indexed_end_q),
                    build_seconds=0.0,
                    build_read_mb=0.0,
                    build_write_mb=0.0,
                    router_group_means=None,
                    router_group_tokens=None,
                    router_group_member_refs=None,
                    native_codebooks=native_codebooks,
                    native_codes=native_codes,
                    native_page_starts=native_page_starts,
                )
                prefix_index_cache[key] = view
                return view

            def gqa_index_pack_key(
                gqa_indexes: list[GPUIndex],
                *,
                extra: tuple[object, ...] = (),
            ) -> tuple[object, ...]:
                return (
                    tuple(
                        (
                            id(index),
                            int(len(index.pages)),
                            int(index.pending_start),
                            id(index.native_codebooks) if index.native_codebooks is not None else 0,
                            id(index.native_codes) if index.native_codes is not None else 0,
                            id(index.native_page_starts) if index.native_page_starts is not None else 0,
                        )
                        for index in gqa_indexes
                    ),
                    *extra,
                )

            def gqa_native_fullscan_pack(gqa_indexes: list[GPUIndex]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                if bool(getattr(args, "profile_native_ops", False)):
                    _sync_if_cuda(device)
                    pack_t0 = time.perf_counter()
                else:
                    pack_t0 = 0.0
                fast_key = gqa_index_pack_key(gqa_indexes, extra=("k", int(args.subbits)))
                gqa_fast_pack_cache = getattr(self, "_pagedpq_gqa_native_pack_fast_cache", None)
                if not isinstance(gqa_fast_pack_cache, dict):
                    gqa_fast_pack_cache = {}
                    setattr(self, "_pagedpq_gqa_native_pack_fast_cache", gqa_fast_pack_cache)
                cached_fast = gqa_fast_pack_cache.get(fast_key)
                if cached_fast is not None:
                    if bool(getattr(args, "profile_native_ops", False)):
                        _sync_if_cuda(device)
                        stats[layer_id].add_native_pack_timing(time.perf_counter() - pack_t0)
                    return cached_fast
                packs = [ensure_native_fullscan_pack(index, subbits=int(args.subbits)) for index in gqa_indexes]
                cache_key = tuple(
                    (
                        int(pack[0].data_ptr()),
                        tuple(int(v) for v in pack[0].shape),
                        int(pack[1].data_ptr()),
                        tuple(int(v) for v in pack[1].shape),
                        str(pack[1].dtype),
                        int(pack[2].data_ptr()),
                        int(pack[2].numel()),
                    )
                    for pack in packs
                )
                gqa_pack_cache = getattr(self, "_pagedpq_gqa_native_pack_cache", None)
                if not isinstance(gqa_pack_cache, dict):
                    gqa_pack_cache = {}
                    setattr(self, "_pagedpq_gqa_native_pack_cache", gqa_pack_cache)
                cached = gqa_pack_cache.get(cache_key)
                if cached is not None:
                    if gqa_fast_pack_cache:
                        gqa_fast_pack_cache.clear()
                    gqa_fast_pack_cache[fast_key] = cached
                    if bool(getattr(args, "profile_native_ops", False)):
                        _sync_if_cuda(device)
                        stats[layer_id].add_native_pack_timing(time.perf_counter() - pack_t0)
                    return cached
                if gqa_pack_cache:
                    gqa_pack_cache.clear()
                    if device.type == "cuda" and bool(getattr(args, "debug_empty_cache_native", False)):
                        torch.cuda.empty_cache()
                codebooks = torch.stack([pack[0] for pack in packs], dim=0).contiguous()
                codes = torch.stack([pack[1] for pack in packs], dim=0).contiguous()
                page_starts = packs[0][2]
                packed = (codebooks, codes, page_starts)
                gqa_pack_cache[cache_key] = packed
                if gqa_fast_pack_cache:
                    gqa_fast_pack_cache.clear()
                gqa_fast_pack_cache[fast_key] = packed
                if bool(getattr(args, "profile_native_ops", False)):
                    _sync_if_cuda(device)
                    stats[layer_id].add_native_pack_timing(time.perf_counter() - pack_t0)
                return packed

            def gqa_value_vpq_pack(
                gqa_indexes: list[GPUIndex],
                *,
                value_group_pages: int,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
                if bool(getattr(args, "profile_native_ops", False)):
                    _sync_if_cuda(device)
                    pack_t0 = time.perf_counter()
                else:
                    pack_t0 = 0.0
                actual_value_subbits_for_key = int(args.value_subbits) if int(args.value_subbits) > 0 else int(args.subbits)
                fast_key = gqa_index_pack_key(
                    gqa_indexes,
                    extra=(
                        "v",
                        int(args.value_subvecs),
                        int(actual_value_subbits_for_key),
                        int(value_group_pages),
                        int(value_bytes),
                    ),
                )
                gqa_value_fast_cache = getattr(self, "_pagedpq_gqa_value_vpq_pack_fast_cache", None)
                if not isinstance(gqa_value_fast_cache, dict):
                    gqa_value_fast_cache = {}
                    setattr(self, "_pagedpq_gqa_value_vpq_pack_fast_cache", gqa_value_fast_cache)
                cached_fast = gqa_value_fast_cache.get(fast_key)
                if cached_fast is not None:
                    if bool(getattr(args, "profile_native_ops", False)):
                        _sync_if_cuda(device)
                        stats[layer_id].add_native_pack_timing(time.perf_counter() - pack_t0)
                    return cached_fast
                value_packs = [
                    value_vpq_pack_torch(
                        index=index,
                        values=values_all[int(kv_head)],
                        value_subvecs=int(args.value_subvecs),
                        value_subbits=int(args.value_subbits),
                        key_bytes=int(value_bytes),
                        device=device,
                        value_group_pages=int(value_group_pages),
                    )
                    for kv_head, index in enumerate(gqa_indexes)
                ]
                if any(pack is None for pack in value_packs):
                    raise RuntimeError("missing V-PQ pack for native decode")
                for index in gqa_indexes:
                    build_stats = getattr(index, "_last_value_vpq_build_stats", None)
                    if build_stats is not None:
                        build_seconds, build_read_mb, build_write_mb = build_stats
                        stats[layer_id].add_index_build(build_seconds, build_read_mb, build_write_mb)
                        setattr(index, "_last_value_vpq_build_stats", None)
                packs = [pack for pack in value_packs if pack is not None]
                cache_key = tuple(
                    (
                        int(pack[0].data_ptr()),
                        tuple(int(v) for v in pack[0].shape),
                        int(pack[1].data_ptr()),
                        tuple(int(v) for v in pack[1].shape),
                        str(pack[1].dtype),
                        int(pack[2].data_ptr()),
                        int(pack[2].numel()),
                        int(pack[3]),
                        int(pack[4]),
                    )
                    for pack in packs
                )
                gqa_value_pack_cache = getattr(self, "_pagedpq_gqa_value_vpq_pack_cache", None)
                if not isinstance(gqa_value_pack_cache, dict):
                    gqa_value_pack_cache = {}
                    setattr(self, "_pagedpq_gqa_value_vpq_pack_cache", gqa_value_pack_cache)
                cached = gqa_value_pack_cache.get(cache_key)
                if cached is not None:
                    if gqa_value_fast_cache:
                        gqa_value_fast_cache.clear()
                    gqa_value_fast_cache[fast_key] = cached
                    if bool(getattr(args, "profile_native_ops", False)):
                        _sync_if_cuda(device)
                        stats[layer_id].add_native_pack_timing(time.perf_counter() - pack_t0)
                    return cached
                if gqa_value_pack_cache:
                    gqa_value_pack_cache.clear()
                    if device.type == "cuda" and bool(getattr(args, "debug_empty_cache_native", False)):
                        torch.cuda.empty_cache()
                value_codebooks = torch.stack([pack[0] for pack in packs], dim=0).contiguous()
                value_codes = torch.stack([pack[1] for pack in packs], dim=0).contiguous()
                value_page_starts = packs[0][2]
                value_page_size = int(packs[0][3])
                actual_value_subbits = int(packs[0][4])
                packed = (value_codebooks, value_codes, value_page_starts, value_page_size, actual_value_subbits)
                gqa_value_pack_cache[cache_key] = packed
                if gqa_value_fast_cache:
                    gqa_value_fast_cache.clear()
                gqa_value_fast_cache[fast_key] = packed
                if bool(getattr(args, "profile_native_ops", False)):
                    _sync_if_cuda(device)
                    stats[layer_id].add_native_pack_timing(time.perf_counter() - pack_t0)
                return packed

            def approximate_prefill_fast_exact() -> torch.Tensor | None:
                if query_len <= 1:
                    return None
                if str(args.selector_mode) != "fullscan" or str(args.selector_backend) not in {"cuda_ext", "auto"}:
                    return None
                confidence_prefill = online_confidence_rule in {
                    "geometric_probe_tail_switch",
                    "geometric_tail_stability_switch",
                    "pq_proxy_mass_budget",
                    "pq_ranked_mass_budget",
                }
                probe_confidence_prefill = online_confidence_rule == "geometric_probe_tail_switch"
                tail_stability_confidence_prefill = online_confidence_rule == "geometric_tail_stability_switch"
                proxy_confidence_prefill = online_confidence_rule == "pq_proxy_mass_budget"
                ranked_confidence_prefill = online_confidence_rule == "pq_ranked_mass_budget"
                selected_vpq_prefill = (
                    str(args.selected_value_mode) == "vpq_value"
                    and tail_blend_value <= 0.0
                    and str(args.selected_value_exact_rule) in {"fixed", "selector_rank"}
                    and int(args.selected_value_min_exact_top) == 0
                    and int(args.selected_value_max_exact_top) == 0
                    and int(getattr(args, "selected_value_exact_all_context_max", 0)) <= 0
                    and float(getattr(args, "selected_value_exact_all_fraction_min", 0.0)) <= 0.0
                    and str(getattr(args, "index_build_backend", "numpy")) == "torch_gpu"
                )
                tail_vpq_prefill = (
                    (str(args.selected_value_mode) == "exact" or selected_vpq_native)
                    and tail_blend_value > 0.0
                    and str(args.tail_mode) == "vpq_value"
                    and (
                        math.isinf(float(args.tail_probe_rel_l2_max))
                        or probe_confidence_prefill
                        or tail_stability_confidence_prefill
                        or proxy_confidence_prefill
                        or ranked_confidence_prefill
                    )
                    and str(getattr(args, "index_build_backend", "numpy")) == "torch_gpu"
                )
                exact_prefill = str(args.selected_value_mode) == "exact" and tail_blend_value <= 0.0
                if not exact_prefill and not selected_vpq_prefill and not tail_vpq_prefill:
                    return None
                if confidence_prefill:
                    exact_ranked_confidence_prefill = ranked_confidence_prefill and exact_prefill
                    selected_vpq_ranked_confidence_prefill = ranked_confidence_prefill and selected_vpq_prefill
                    exact_tail_confidence_prefill = (
                        tail_vpq_prefill and str(args.selected_value_mode) == "exact"
                    )
                    selected_vpq_tail_confidence_prefill = (
                        tail_vpq_prefill
                        and str(args.selected_value_mode) == "vpq_value"
                        and selected_vpq_native
                    )
                    if (
                        not exact_ranked_confidence_prefill
                        and not selected_vpq_ranked_confidence_prefill
                        and not exact_tail_confidence_prefill
                        and not selected_vpq_tail_confidence_prefill
                    ):
                            return None
                    if selected_vpq_tail_confidence_prefill:
                        if (proxy_confidence_prefill or ranked_confidence_prefill) and (
                            float(args.tail_pq_corr_min) > -1.0
                            or math.isfinite(float(args.tail_pq_relrmse_max))
                        ):
                            return None
                if int(args.rerank_candidates) > 0 or budget_by_head:
                    return None
                budget = int(args.budget)
                if confidence_prefill:
                    budget = max(
                        int(budget),
                        int(args.geometric_min_budget),
                        int(args.geometric_max_budget),
                    )
                rank_buffer_limit_mb = float(getattr(args, "prefill_rank_buffer_limit_mb", 4096.0))
                prefill_chunk_size_for_limit = int(getattr(args, "prefill_chunk_size", 0))
                rank_buffer_positions = (
                    min(int(query_len), max(1, int(prefill_chunk_size_for_limit)))
                    if int(prefill_chunk_size_for_limit) > 0
                    else int(query_len)
                )
                estimated_rank_buffer_mb = (
                    float(rank_buffer_positions) * float(num_heads) * float(max(0, int(budget))) * float(8 + 4) / MB
                )
                if rank_buffer_limit_mb > 0.0 and estimated_rank_buffer_mb > rank_buffer_limit_mb:
                    log(
                        "declining batched prefill fast path: estimated ranked-token buffer "
                        f"{estimated_rank_buffer_mb:.1f} MB exceeds limit {rank_buffer_limit_mb:.1f} MB "
                        f"(query_len={query_len}, rank_buffer_positions={rank_buffer_positions}, heads={num_heads}, budget={budget})"
                    )
                    return None
                selected_vpq_exact_all_prefill = (
                    selected_vpq_prefill
                    and not tail_vpq_prefill
                    and max(0, int(args.selected_value_exact_top)) >= max(0, int(budget))
                )
                gqa_indexes = [index_cache[int(kv_head)] for kv_head in range(num_kv_heads)]
                if not gqa_indexes or not all(index.pages for index in gqa_indexes):
                    return None
                def torch_lut_prefill_topk(
                    queries_in: torch.Tensor,
                    codebooks_in: torch.Tensor,
                    codes_in: torch.Tensor,
                    page_starts_in: torch.Tensor,
                    *,
                    local_query_start: int,
                    streaming: bool = False,
                    score_dtype: torch.dtype = torch.float32,
                ) -> tuple[torch.Tensor, torch.Tensor]:
                    positions = int(queries_in.shape[0])
                    heads = int(queries_in.shape[1])
                    pages = int(codebooks_in.shape[1])
                    subvecs = int(codebooks_in.shape[2])
                    page_size_local = int(codes_in.shape[2])
                    total_tokens = int(pages * page_size_local)
                    k = min(max(0, int(budget)), total_tokens)
                    if k <= 0 or total_tokens <= 0:
                        return (
                            torch.empty((positions, heads, 0), dtype=torch.long, device=device),
                            torch.empty((positions, heads, 0), dtype=torch.float32, device=device),
                        )
                    top_tokens_out = torch.empty((positions, heads, k), dtype=torch.long, device=device)
                    top_scores_out = torch.empty((positions, heads, k), dtype=torch.float32, device=device)
                    query_context_lens = (
                        torch.arange(positions, device=device, dtype=torch.long) + int(local_query_start) + 1
                    )
                    dyn_start_t = torch.clamp(
                        torch.full_like(query_context_lens, int(args.static_prefix)),
                        min=0,
                    )
                    dyn_start_t = torch.minimum(dyn_start_t, query_context_lens)
                    indexed_end_t = torch.maximum(
                        dyn_start_t,
                        query_context_lens - max(0, int(args.static_suffix)),
                    )
                    sealed_end_t = dyn_start_t + (
                        torch.div(
                            torch.clamp(indexed_end_t - dyn_start_t, min=0),
                            max(1, int(args.page_size)),
                            rounding_mode="floor",
                        )
                        * max(1, int(args.page_size))
                    )
                    page_starts_dev = page_starts_in.to(device=device, dtype=torch.long)
                    page_offsets = torch.arange(page_size_local, dtype=torch.long, device=device).reshape(1, 1, page_size_local)
                    for kv_head in range(num_kv_heads):
                        head_start = int(kv_head * group_size)
                        head_end = min(heads, head_start + int(group_size))
                        if head_start >= head_end:
                            continue
                        head_queries = queries_in[:, head_start:head_end, :].contiguous()
                        if streaming:
                            running_vals = torch.full(
                                (positions, head_end - head_start, k),
                                float("-inf"),
                                dtype=torch.float32,
                                device=device,
                            )
                            running_toks = torch.zeros(
                                (positions, head_end - head_start, k),
                                dtype=torch.long,
                                device=device,
                            )
                        score_pages = []
                        for page_idx in range(pages):
                            page_scores = torch.zeros(
                                (positions, head_end - head_start, page_size_local),
                                dtype=torch.float32,
                                device=device,
                            )
                            for sub in range(subvecs):
                                q_sub = head_queries[
                                    :,
                                    :,
                                    sub * int(codebooks_in.shape[-1]) : (sub + 1) * int(codebooks_in.shape[-1]),
                                ].reshape(positions * (head_end - head_start), int(codebooks_in.shape[-1]))
                                lut = q_sub @ codebooks_in[int(kv_head), int(page_idx), int(sub)].t().contiguous()
                                page_codes = codes_in[int(kv_head), int(page_idx), :, int(sub)].to(torch.long)
                                page_scores = page_scores + lut.index_select(1, page_codes).reshape(
                                    positions,
                                    head_end - head_start,
                                    page_size_local,
                                )
                            page_start = page_starts_dev[int(page_idx)]
                            valid = (page_start >= dyn_start_t) & (page_start + page_size_local <= sealed_end_t)
                            if not bool(torch.all(valid)):
                                page_scores = page_scores.masked_fill(
                                    ~valid.reshape(positions, 1, 1),
                                    float("-inf"),
                                )
                            if streaming:
                                page_toks = (page_start + page_offsets).expand(
                                    positions,
                                    head_end - head_start,
                                    page_size_local,
                                )
                                cand_vals = torch.cat([running_vals, page_scores], dim=2)
                                cand_toks = torch.cat([running_toks, page_toks], dim=2)
                                running_vals, order = torch.topk(
                                    cand_vals,
                                    k,
                                    dim=2,
                                    largest=True,
                                    sorted=True,
                                )
                                running_toks = cand_toks.gather(2, order)
                                continue
                            score_pages.append(page_scores.to(score_dtype) if score_dtype != torch.float32 else page_scores)
                        if streaming:
                            top_tokens_out[:, head_start:head_end, :] = running_toks
                            top_scores_out[:, head_start:head_end, :] = running_vals
                            continue
                        scores = torch.cat(score_pages, dim=2)
                        vals, idx = torch.topk(
                            scores.reshape(positions * (head_end - head_start), total_tokens),
                            k,
                            dim=1,
                            largest=True,
                            sorted=True,
                        )
                        idx = idx.reshape(positions, head_end - head_start, k)
                        page_ids = torch.div(idx, page_size_local, rounding_mode="floor")
                        rows = idx - page_ids * page_size_local
                        toks = page_starts_dev.index_select(0, page_ids.reshape(-1)).reshape_as(idx) + rows
                        top_tokens_out[:, head_start:head_end, :] = toks
                        top_scores_out[:, head_start:head_end, :] = vals.reshape(positions, head_end - head_start, k)
                    return top_tokens_out, top_scores_out

                def torch_lut_prefill_topk_context_lens(
                    queries_in: torch.Tensor,
                    context_lens: torch.Tensor,
                    codebooks_in: torch.Tensor,
                    codes_in: torch.Tensor,
                    page_starts_in: torch.Tensor,
                    *,
                    streaming: bool = False,
                    score_dtype: torch.dtype = torch.float32,
                ) -> tuple[torch.Tensor, torch.Tensor]:
                    positions = int(queries_in.shape[0])
                    heads = int(queries_in.shape[1])
                    pages = int(codebooks_in.shape[1])
                    subvecs = int(codebooks_in.shape[2])
                    page_size_local = int(codes_in.shape[2])
                    total_tokens = int(pages * page_size_local)
                    k = min(max(0, int(budget)), total_tokens)
                    if k <= 0 or total_tokens <= 0:
                        return (
                            torch.empty((positions, heads, 0), dtype=torch.long, device=device),
                            torch.empty((positions, heads, 0), dtype=torch.float32, device=device),
                        )
                    top_tokens_out = torch.empty((positions, heads, k), dtype=torch.long, device=device)
                    top_scores_out = torch.empty((positions, heads, k), dtype=torch.float32, device=device)
                    query_context_lens = context_lens.to(device=device, dtype=torch.long)
                    dyn_start_t = torch.minimum(
                        torch.full_like(query_context_lens, max(0, int(args.static_prefix))),
                        query_context_lens,
                    )
                    indexed_end_t = torch.maximum(
                        dyn_start_t,
                        query_context_lens - max(0, int(args.static_suffix)),
                    )
                    sealed_end_t = dyn_start_t + (
                        torch.div(
                            torch.clamp(indexed_end_t - dyn_start_t, min=0),
                            max(1, int(args.page_size)),
                            rounding_mode="floor",
                        )
                        * max(1, int(args.page_size))
                    )
                    page_starts_dev = page_starts_in.to(device=device, dtype=torch.long)
                    page_offsets = torch.arange(page_size_local, dtype=torch.long, device=device).reshape(1, 1, page_size_local)
                    for kv_head in range(num_kv_heads):
                        head_start = int(kv_head * group_size)
                        head_end = min(heads, head_start + int(group_size))
                        if head_start >= head_end:
                            continue
                        head_queries = queries_in[:, head_start:head_end, :].contiguous()
                        if streaming:
                            running_vals = torch.full(
                                (positions, head_end - head_start, k),
                                float("-inf"),
                                dtype=torch.float32,
                                device=device,
                            )
                            running_toks = torch.zeros(
                                (positions, head_end - head_start, k),
                                dtype=torch.long,
                                device=device,
                            )
                        score_pages = []
                        for page_idx in range(pages):
                            page_scores = torch.zeros(
                                (positions, head_end - head_start, page_size_local),
                                dtype=torch.float32,
                                device=device,
                            )
                            for sub in range(subvecs):
                                q_sub = head_queries[
                                    :,
                                    :,
                                    sub * int(codebooks_in.shape[-1]) : (sub + 1) * int(codebooks_in.shape[-1]),
                                ].reshape(positions * (head_end - head_start), int(codebooks_in.shape[-1]))
                                lut = q_sub @ codebooks_in[int(kv_head), int(page_idx), int(sub)].t().contiguous()
                                page_codes = codes_in[int(kv_head), int(page_idx), :, int(sub)].to(torch.long)
                                page_scores = page_scores + lut.index_select(1, page_codes).reshape(
                                    positions,
                                    head_end - head_start,
                                    page_size_local,
                                )
                            page_start = page_starts_dev[int(page_idx)]
                            valid = (page_start >= dyn_start_t) & (page_start + page_size_local <= sealed_end_t)
                            if not bool(torch.all(valid)):
                                page_scores = page_scores.masked_fill(
                                    ~valid.reshape(positions, 1, 1),
                                    float("-inf"),
                                )
                            if streaming:
                                page_toks = (page_start + page_offsets).expand(
                                    positions,
                                    head_end - head_start,
                                    page_size_local,
                                )
                                cand_vals = torch.cat([running_vals, page_scores], dim=2)
                                cand_toks = torch.cat([running_toks, page_toks], dim=2)
                                running_vals, order = torch.topk(
                                    cand_vals,
                                    k,
                                    dim=2,
                                    largest=True,
                                    sorted=True,
                                )
                                running_toks = cand_toks.gather(2, order)
                                continue
                            score_pages.append(page_scores.to(score_dtype) if score_dtype != torch.float32 else page_scores)
                        if streaming:
                            top_tokens_out[:, head_start:head_end, :] = running_toks
                            top_scores_out[:, head_start:head_end, :] = running_vals
                            continue
                        scores = torch.cat(score_pages, dim=2)
                        vals, idx = torch.topk(
                            scores.reshape(positions * (head_end - head_start), total_tokens),
                            k,
                            dim=1,
                            largest=True,
                            sorted=True,
                        )
                        idx = idx.reshape(positions, head_end - head_start, k)
                        page_ids = torch.div(idx, page_size_local, rounding_mode="floor")
                        rows = idx - page_ids * page_size_local
                        toks = page_starts_dev.index_select(0, page_ids.reshape(-1)).reshape_as(idx) + rows
                        top_tokens_out[:, head_start:head_end, :] = toks
                        top_scores_out[:, head_start:head_end, :] = vals.reshape(positions, head_end - head_start, k)
                    return top_tokens_out, top_scores_out

                def torch_lut_batched_prefill_topk(
                    queries_in: torch.Tensor,
                    codebooks_in: torch.Tensor,
                    codes_in: torch.Tensor,
                    page_starts_in: torch.Tensor,
                    *,
                    local_query_start: int,
                ) -> tuple[torch.Tensor, torch.Tensor]:
                    positions = int(queries_in.shape[0])
                    tile_size = int(getattr(args, "prefill_selector_tile_size", 0))
                    if tile_size > 0 and positions > tile_size:
                        token_chunks = []
                        score_chunks = []
                        for tile_start in range(0, positions, tile_size):
                            tile_end = min(positions, tile_start + tile_size)
                            tile_tokens, tile_scores = torch_lut_batched_prefill_topk(
                                queries_in[tile_start:tile_end],
                                codebooks_in,
                                codes_in,
                                page_starts_in,
                                local_query_start=int(local_query_start) + int(tile_start),
                            )
                            token_chunks.append(tile_tokens)
                            score_chunks.append(tile_scores)
                        return torch.cat(token_chunks, dim=0), torch.cat(score_chunks, dim=0)
                    heads = int(queries_in.shape[1])
                    pages = int(codebooks_in.shape[1])
                    subvecs = int(codebooks_in.shape[2])
                    centroids = int(codebooks_in.shape[3])
                    subdim = int(codebooks_in.shape[4])
                    page_size_local = int(codes_in.shape[2])
                    total_tokens = int(pages * page_size_local)
                    k = min(max(0, int(budget)), total_tokens)
                    if k <= 0 or total_tokens <= 0:
                        return (
                            torch.empty((positions, heads, 0), dtype=torch.long, device=device),
                            torch.empty((positions, heads, 0), dtype=torch.float32, device=device),
                        )
                    top_tokens_out = torch.empty((positions, heads, k), dtype=torch.long, device=device)
                    top_scores_out = torch.empty((positions, heads, k), dtype=torch.float32, device=device)
                    query_context_lens = (
                        torch.arange(positions, device=device, dtype=torch.long) + int(local_query_start) + 1
                    )
                    dyn_start_t = torch.minimum(
                        torch.full_like(query_context_lens, max(0, int(args.static_prefix))),
                        query_context_lens,
                    )
                    indexed_end_t = torch.maximum(
                        dyn_start_t,
                        query_context_lens - max(0, int(args.static_suffix)),
                    )
                    sealed_end_t = dyn_start_t + (
                        torch.div(
                            torch.clamp(indexed_end_t - dyn_start_t, min=0),
                            max(1, int(args.page_size)),
                            rounding_mode="floor",
                        )
                        * max(1, int(args.page_size))
                    )
                    page_starts_dev = page_starts_in.to(device=device, dtype=torch.long)
                    valid_pages = (
                        (page_starts_dev.reshape(1, pages) >= dyn_start_t.reshape(positions, 1))
                        & ((page_starts_dev.reshape(1, pages) + page_size_local) <= sealed_end_t.reshape(positions, 1))
                    )
                    page_block_size = int(getattr(args, "prefill_selector_page_block_size", 0))
                    if page_block_size > 0:
                        page_block_size = max(1, int(page_block_size))
                        for kv_head in range(num_kv_heads):
                            head_start = int(kv_head * group_size)
                            head_end = min(heads, head_start + int(group_size))
                            group_heads = int(head_end - head_start)
                            if head_start >= head_end:
                                continue
                            q_group = queries_in[:, head_start:head_end, :].reshape(
                                positions,
                                group_heads,
                                subvecs,
                                subdim,
                            )
                            running_vals = torch.full(
                                (positions, group_heads, k),
                                float("-inf"),
                                dtype=torch.float32,
                                device=device,
                            )
                            running_toks = torch.zeros((positions, group_heads, k), dtype=torch.long, device=device)
                            for page_begin in range(0, pages, page_block_size):
                                page_end = min(pages, page_begin + page_block_size)
                                block_pages = int(page_end - page_begin)
                                codebooks_block = codebooks_in[int(kv_head), page_begin:page_end]
                                # [positions, group_heads, block_pages, subvecs, centroids]
                                lut = torch.einsum("xgsd,bscd->xgbsc", q_group, codebooks_block)
                                flat_lut = lut.reshape(
                                    positions * group_heads * block_pages * subvecs,
                                    centroids,
                                )
                                code_rows = codes_in[int(kv_head), page_begin:page_end].to(torch.long)
                                code_rows = code_rows.permute(0, 2, 1).contiguous()
                                code_rows = code_rows.reshape(1, 1, block_pages, subvecs, page_size_local).expand(
                                    positions,
                                    group_heads,
                                    block_pages,
                                    subvecs,
                                    page_size_local,
                                )
                                gathered = torch.gather(
                                    flat_lut,
                                    1,
                                    code_rows.reshape(-1, page_size_local),
                                ).reshape(
                                    positions,
                                    group_heads,
                                    block_pages,
                                    subvecs,
                                    page_size_local,
                                )
                                block_scores = gathered.sum(dim=3)
                                block_valid = valid_pages[:, page_begin:page_end]
                                block_scores = block_scores.masked_fill(
                                    ~block_valid.reshape(positions, 1, block_pages, 1),
                                    float("-inf"),
                                ).reshape(positions, group_heads, block_pages * page_size_local)
                                block_offsets = torch.arange(page_size_local, dtype=torch.long, device=device).reshape(1, 1, 1, page_size_local)
                                block_starts = page_starts_dev[page_begin:page_end].reshape(1, 1, block_pages, 1)
                                block_toks = (block_starts + block_offsets).expand(
                                    positions,
                                    group_heads,
                                    block_pages,
                                    page_size_local,
                                ).reshape(positions, group_heads, block_pages * page_size_local)
                                cand_vals = torch.cat([running_vals, block_scores], dim=2)
                                cand_toks = torch.cat([running_toks, block_toks], dim=2)
                                running_vals, order = torch.topk(
                                    cand_vals,
                                    k,
                                    dim=2,
                                    largest=True,
                                    sorted=True,
                                )
                                running_toks = cand_toks.gather(2, order)
                            top_tokens_out[:, head_start:head_end, :] = running_toks
                            top_scores_out[:, head_start:head_end, :] = running_vals
                        return top_tokens_out, top_scores_out
                    for kv_head in range(num_kv_heads):
                        head_start = int(kv_head * group_size)
                        head_end = min(heads, head_start + int(group_size))
                        if head_start >= head_end:
                            continue
                        q_group = queries_in[:, head_start:head_end, :].reshape(
                            positions,
                            head_end - head_start,
                            subvecs,
                            subdim,
                        )
                        # [positions, group_heads, pages, subvecs, centroids]
                        lut = torch.einsum(
                            "xgsd,yscd->xgysc",
                            q_group,
                            codebooks_in[int(kv_head)],
                        )
                        flat_lut = lut.reshape(
                            positions * (head_end - head_start) * pages * subvecs,
                            centroids,
                        )
                        code_rows = codes_in[int(kv_head)].to(torch.long).permute(0, 2, 1).contiguous()
                        code_rows = code_rows.reshape(1, 1, pages, subvecs, page_size_local).expand(
                            positions,
                            head_end - head_start,
                            pages,
                            subvecs,
                            page_size_local,
                        )
                        gathered = torch.gather(
                            flat_lut,
                            1,
                            code_rows.reshape(-1, page_size_local),
                        ).reshape(
                            positions,
                            head_end - head_start,
                            pages,
                            subvecs,
                            page_size_local,
                        )
                        scores = gathered.sum(dim=3).reshape(
                            positions,
                            head_end - head_start,
                            total_tokens,
                        )
                        scores = scores.masked_fill(
                            ~valid_pages.reshape(positions, 1, pages, 1).expand(
                                positions,
                                head_end - head_start,
                                pages,
                                page_size_local,
                            ).reshape(positions, head_end - head_start, total_tokens),
                            float("-inf"),
                        )
                        vals, idx = torch.topk(
                            scores.reshape(positions * (head_end - head_start), total_tokens),
                            k,
                            dim=1,
                            largest=True,
                            sorted=True,
                        )
                        idx = idx.reshape(positions, head_end - head_start, k)
                        page_ids = torch.div(idx, page_size_local, rounding_mode="floor")
                        rows = idx - page_ids * page_size_local
                        toks = page_starts_dev.index_select(0, page_ids.reshape(-1)).reshape_as(idx) + rows
                        top_tokens_out[:, head_start:head_end, :] = toks
                        top_scores_out[:, head_start:head_end, :] = vals.reshape(positions, head_end - head_start, k)
                    return top_tokens_out, top_scores_out

                torch_matmul_k_approx_t_cache: dict[tuple[int, int, tuple[int, ...], tuple[int, ...], torch.dtype], torch.Tensor] = {}

                def torch_matmul_k_approx_t(
                    codebooks_in: torch.Tensor,
                    codes_in: torch.Tensor,
                    *,
                    kv_heads_local: int,
                    pages: int,
                    subvecs: int,
                    subdim: int,
                    page_size_local: int,
                    total_tokens: int,
                    dim: int,
                ) -> torch.Tensor:
                    """Return cached PQ-reconstructed K as [kv_heads, dim, tokens].

                    The chunked long-prefill path uses the same page-PQ index for
                    every query chunk. Reconstructing the approximate K matrix per
                    chunk dominates runtime at long context, so keep one layer-local
                    cache entry and evict older prefix views aggressively.
                    """

                    key = (
                        int(codebooks_in.data_ptr()),
                        int(codes_in.data_ptr()),
                        tuple(int(x) for x in codebooks_in.shape),
                        tuple(int(x) for x in codes_in.shape),
                        codes_in.dtype,
                    )
                    cached = torch_matmul_k_approx_t_cache.get(key)
                    if cached is not None:
                        return cached
                    torch_matmul_k_approx_t_cache.clear()
                    flat_page_ids = torch.arange(pages, dtype=torch.long, device=device).repeat_interleave(page_size_local)
                    flat_codes = codes_in.to(torch.long).reshape(kv_heads_local, total_tokens, subvecs)
                    kv_ids = torch.arange(kv_heads_local, dtype=torch.long, device=device).reshape(
                        kv_heads_local,
                        1,
                    ).expand(kv_heads_local, total_tokens)
                    page_ids = flat_page_ids.reshape(1, total_tokens).expand(kv_heads_local, total_tokens)
                    k_approx = torch.empty((kv_heads_local, total_tokens, dim), dtype=torch.float32, device=device)
                    for sub in range(subvecs):
                        k_approx[:, :, sub * subdim : (sub + 1) * subdim] = codebooks_in[
                            kv_ids,
                            page_ids,
                            int(sub),
                            flat_codes[:, :, int(sub)],
                        ]
                    cached = k_approx.transpose(1, 2).contiguous()
                    torch_matmul_k_approx_t_cache[key] = cached
                    return cached

                def torch_matmul_prefill_topk_scores(
                    queries_in: torch.Tensor,
                    codebooks_in: torch.Tensor,
                    codes_in: torch.Tensor,
                    page_starts_in: torch.Tensor,
                    *,
                    local_query_start: int,
                    need_dense_scores: bool,
                    need_dense_logsumexp: bool = True,
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
                    positions = int(queries_in.shape[0])
                    heads = int(queries_in.shape[1])
                    kv_heads_local = int(codebooks_in.shape[0])
                    pages = int(codebooks_in.shape[1])
                    subvecs = int(codebooks_in.shape[2])
                    subdim = int(codebooks_in.shape[4])
                    dim = int(subvecs * subdim)
                    page_size_local = int(codes_in.shape[2])
                    total_tokens = int(pages * page_size_local)
                    k = min(max(0, int(budget)), total_tokens)
                    tile_size = int(getattr(args, "prefill_selector_tile_size", 0))
                    if tile_size > 0 and positions > tile_size and not bool(need_dense_scores):
                        token_chunks = []
                        score_chunks = []
                        for tile_start in range(0, positions, tile_size):
                            tile_end = min(positions, tile_start + tile_size)
                            tile_tokens, tile_scores, _dense, _lse = torch_matmul_prefill_topk_scores(
                                queries_in[tile_start:tile_end],
                                codebooks_in,
                                codes_in,
                                page_starts_in,
                                local_query_start=int(local_query_start) + int(tile_start),
                                need_dense_scores=False,
                                need_dense_logsumexp=False,
                            )
                            token_chunks.append(tile_tokens)
                            score_chunks.append(tile_scores)
                        return torch.cat(token_chunks, dim=0), torch.cat(score_chunks, dim=0), None, None
                    top_tokens_out = torch.empty((positions, heads, k), dtype=torch.long, device=device)
                    top_scores_out = torch.empty((positions, heads, k), dtype=torch.float32, device=device)
                    dense_scores_out = (
                        torch.empty((positions, heads, total_tokens), dtype=torch.float32, device=device)
                        if bool(need_dense_scores)
                        else None
                    )
                    dense_logsumexp_out = (
                        torch.empty((positions, heads), dtype=torch.float32, device=device)
                        if bool(need_dense_scores) and bool(need_dense_logsumexp)
                        else None
                    )
                    if positions <= 0 or heads <= 0 or total_tokens <= 0:
                        return top_tokens_out, top_scores_out, dense_scores_out, dense_logsumexp_out

                    k_approx_t = torch_matmul_k_approx_t(
                        codebooks_in,
                        codes_in,
                        kv_heads_local=kv_heads_local,
                        pages=pages,
                        subvecs=subvecs,
                        subdim=subdim,
                        page_size_local=page_size_local,
                        total_tokens=total_tokens,
                        dim=dim,
                    )

                    query_context_lens = (
                        torch.arange(positions, device=device, dtype=torch.long) + int(local_query_start) + 1
                    )
                    dyn_start_t = torch.minimum(
                        torch.full_like(query_context_lens, max(0, int(args.static_prefix))),
                        query_context_lens,
                    )
                    indexed_end_t = torch.maximum(
                        dyn_start_t,
                        query_context_lens - max(0, int(args.static_suffix)),
                    )
                    sealed_end_t = dyn_start_t + (
                        torch.div(
                            torch.clamp(indexed_end_t - dyn_start_t, min=0),
                            max(1, int(args.page_size)),
                            rounding_mode="floor",
                        )
                        * max(1, int(args.page_size))
                    )
                    page_starts_dev = page_starts_in.to(device=device, dtype=torch.long)
                    valid_pages = (
                        (page_starts_dev.reshape(1, pages) >= dyn_start_t.reshape(positions, 1))
                        & ((page_starts_dev.reshape(1, pages) + page_size_local) <= sealed_end_t.reshape(positions, 1))
                    )
                    for kv_head in range(kv_heads_local):
                        head_start = int(kv_head * group_size)
                        head_end = min(heads, head_start + int(group_size))
                        if head_start >= head_end:
                            continue
                        q_group = queries_in[:, head_start:head_end, :].reshape(
                            positions * (head_end - head_start),
                            dim,
                        )
                        scores = torch.matmul(q_group, k_approx_t[int(kv_head)]).reshape(
                            positions,
                            head_end - head_start,
                            total_tokens,
                        )
                        scores = scores.masked_fill(
                            ~valid_pages.reshape(positions, 1, pages, 1).expand(
                                positions,
                                head_end - head_start,
                                pages,
                                page_size_local,
                            ).reshape(positions, head_end - head_start, total_tokens),
                            float("-inf"),
                        )
                        if dense_scores_out is not None:
                            dense_scores_out[:, head_start:head_end, :] = scores
                        if dense_logsumexp_out is not None:
                            dense_logsumexp_out[:, head_start:head_end] = torch.logsumexp(
                                scores * (float(dim) ** -0.5),
                                dim=-1,
                            )
                        if k > 0:
                            vals, idx = torch.topk(
                                scores.reshape(positions * (head_end - head_start), total_tokens),
                                k,
                                dim=1,
                                largest=True,
                                sorted=True,
                            )
                            idx = idx.reshape(positions, head_end - head_start, k)
                            top_scores_out[:, head_start:head_end, :] = vals.reshape(
                                positions,
                                head_end - head_start,
                                k,
                            )
                            page_ids_top = torch.div(idx, page_size_local, rounding_mode="floor")
                            rows = idx - page_ids_top * page_size_local
                            toks = page_starts_dev.index_select(0, page_ids_top.reshape(-1)).reshape_as(idx) + rows
                            top_tokens_out[:, head_start:head_end, :] = toks
                    return top_tokens_out, top_scores_out, dense_scores_out, dense_logsumexp_out

                prefill_selector_backend = str(getattr(args, "prefill_selector_backend", "native"))
                prefill_chunk_size = int(getattr(args, "prefill_chunk_size", 0))
                prefill_tail_score_reuse = bool(getattr(args, "prefill_tail_score_reuse", False))
                if exact_prefill and prefill_chunk_size > 0 and query_len > prefill_chunk_size:
                    try:
                        native = load_selector_paged_pq_ext()
                        outputs_chunks = []
                        selector_seconds_total = 0.0
                        attention_seconds_total = 0.0
                        for chunk_start in range(0, query_len, prefill_chunk_size):
                            chunk_end = min(query_len, chunk_start + prefill_chunk_size)
                            chunk_positions = int(chunk_end - chunk_start)
                            chunk_query_start = int(query_start + chunk_start)
                            chunk_context_end = int(query_start + chunk_end)
                            chunk_indexes = [
                                prefix_index_for(int(kv_head), int(chunk_context_end))
                                for kv_head in range(num_kv_heads)
                            ]
                            queries_chunk = q_all[:, chunk_start:chunk_end, :].transpose(0, 1).to(device).contiguous()
                            selected_page_ids = None
                            selected_page_scores = None
                            if chunk_indexes and all(index.pages for index in chunk_indexes):
                                codebooks, codes, page_starts = gqa_native_fullscan_pack(chunk_indexes)
                                if bool(getattr(args, "profile_native_ops", False)):
                                    _sync_if_cuda(device)
                                    selector_t0 = time.perf_counter()
                                if prefill_selector_backend == "native_page_max":
                                    if str(getattr(args, "prefill_attention_backend", "native")) != "flashinfer_page_blocks":
                                        return None
                                    page_budget = int(getattr(args, "prefill_selector_page_block_size", 0))
                                    if page_budget <= 0:
                                        page_budget = int(math.ceil(max(0, int(budget)) / max(1, int(args.page_size))))
                                    selected_page_ids, selected_page_scores = native.gqa_causal_fullscan_pq_top_pages(
                                        queries_chunk,
                                        codebooks,
                                        codes,
                                        page_starts,
                                        int(group_size),
                                        int(page_budget),
                                        int(chunk_query_start),
                                        int(args.static_prefix),
                                        int(args.static_suffix),
                                    )
                                    ranked_t = torch.empty((chunk_positions, num_heads, 0), dtype=torch.long, device=device)
                                    ranked_scores = torch.empty((chunk_positions, num_heads, 0), dtype=torch.float32, device=device)
                                elif prefill_selector_backend == "native_fused":
                                    ranked_t, ranked_scores = native.gqa_causal_fullscan_pq_topk_fused(
                                        queries_chunk,
                                        codebooks,
                                        codes,
                                        page_starts,
                                        int(group_size),
                                        int(budget),
                                        int(chunk_query_start),
                                        int(args.static_prefix),
                                        int(args.static_suffix),
                                    )
                                elif prefill_selector_backend == "torch_lut_batched":
                                    ranked_t, ranked_scores = torch_lut_batched_prefill_topk(
                                        queries_chunk,
                                        codebooks,
                                        codes,
                                        page_starts,
                                        local_query_start=int(chunk_query_start),
                                    )
                                elif prefill_selector_backend in {"torch_lut", "torch_lut_fp16", "torch_lut_streaming"}:
                                    ranked_t, ranked_scores = torch_lut_prefill_topk(
                                        queries_chunk,
                                        codebooks,
                                        codes,
                                        page_starts,
                                        local_query_start=int(chunk_query_start),
                                        streaming=prefill_selector_backend == "torch_lut_streaming",
                                        score_dtype=torch.float16 if prefill_selector_backend == "torch_lut_fp16" else torch.float32,
                                    )
                                elif prefill_selector_backend == "torch_matmul":
                                    ranked_t, ranked_scores, _, _ = torch_matmul_prefill_topk_scores(
                                        queries_chunk,
                                        codebooks,
                                        codes,
                                        page_starts,
                                        local_query_start=int(chunk_query_start),
                                        need_dense_scores=False,
                                        need_dense_logsumexp=False,
                                    )
                                else:
                                    ranked_t, ranked_scores = native.gqa_causal_fullscan_pq_topk(
                                        queries_chunk,
                                        codebooks,
                                        codes,
                                        page_starts,
                                        int(group_size),
                                        int(budget),
                                        int(chunk_query_start),
                                        int(args.static_prefix),
                                        int(args.static_suffix),
                                    )
                                if bool(getattr(args, "profile_native_ops", False)):
                                    _sync_if_cuda(device)
                                    selector_seconds_total += float(time.perf_counter() - selector_t0)
                            else:
                                ranked_t = torch.empty((chunk_positions, num_heads, 0), dtype=torch.long, device=device)
                                ranked_scores = torch.empty((chunk_positions, num_heads, 0), dtype=torch.float32, device=device)
                            ranked_scores_for_attention = ranked_scores.contiguous()
                            accepted_budget_mean_by_pos_cpu_chunk: list[float] | None = None
                            accepted_budget_cost_upper_chunk: float | None = None
                            if ranked_confidence_prefill and ranked_scores.numel() > 0:
                                rank_count = int(ranked_scores.shape[2])
                                max_budget = max(0, min(rank_count, int(args.geometric_max_budget)))
                                if max_budget <= 0:
                                    max_budget = rank_count
                                if max_budget > 0:
                                    granularity = max(1, int(args.geometric_budget_granularity))
                                    min_budget = _round_budget_up(
                                        int(args.geometric_min_budget),
                                        granularity=granularity,
                                        max_budget=max_budget,
                                    )
                                    proxy_target = max(
                                        float(args.tail_proxy_mass_min),
                                        1.0 - float(args.tail_proxy_mass_max),
                                    )
                                    proxy_target = min(max(float(proxy_target), 0.0), 1.0 - 1.0e-7)
                                    scale = float(self.head_dim) ** -0.5
                                    top_scores = ranked_scores[:, :, :max_budget].float() * scale
                                    denom = torch.logsumexp(top_scores, dim=-1)
                                    cum_mass = torch.exp(torch.logcumsumexp(top_scores, dim=-1) - denom.unsqueeze(-1))
                                    hit = cum_mass >= proxy_target
                                    has_hit = torch.any(hit, dim=-1)
                                    first_hit = torch.argmax(hit.to(torch.int32), dim=-1).to(torch.long) + 1
                                    accepted_budget_counts = torch.where(
                                        has_hit,
                                        first_hit,
                                        torch.full_like(first_hit, max_budget),
                                    )
                                    accepted_budget_counts = torch.clamp(
                                        accepted_budget_counts,
                                        min=int(min_budget),
                                        max=int(max_budget),
                                    )
                                    accepted_budget_counts = (
                                        torch.div(
                                            accepted_budget_counts + granularity - 1,
                                            granularity,
                                            rounding_mode="floor",
                                        )
                                        * granularity
                                    ).clamp(max=int(max_budget))
                                    accepted_budget_cost_upper_chunk = float(max_budget)
                                    if ranked_confidence_cost_mode == "exact":
                                        accepted_budget_mean_by_pos_cpu_chunk = (
                                            accepted_budget_counts.float().mean(dim=1).detach().cpu().tolist()
                                        )
                                    rank_ids = decode_rank_ids_tensor(rank_count, device, dims=3)
                                    ranked_scores_for_attention = ranked_scores.masked_fill(
                                        rank_ids >= accepted_budget_counts.unsqueeze(-1),
                                        float("-inf"),
                                    ).contiguous()
                            if bool(getattr(args, "profile_native_ops", False)):
                                _sync_if_cuda(device)
                                attention_t0 = time.perf_counter()
                            prefill_attention_backend = str(getattr(args, "prefill_attention_backend", "native"))
                            chunk_page_counts_cpu: list[int] | None = None
                            if prefill_attention_backend == "flashinfer_page_blocks":
                                try:
                                    import flashinfer  # type: ignore
                                except Exception as exc:
                                    raise RuntimeError("prefill_attention_backend=flashinfer_page_blocks requires flashinfer") from exc
                                page_size = int(args.page_size)
                                page_count = int((chunk_context_end + page_size - 1) // page_size)
                                padded_context_len = int(page_count * page_size)
                                if selected_page_ids is not None and selected_page_scores is not None:
                                    ranked_pages = selected_page_ids.to(torch.long).clamp(max=max(0, page_count - 1))
                                    ranked_valid = torch.isfinite(selected_page_scores) & (selected_page_ids >= 0)
                                else:
                                    ranked_pages = torch.div(
                                        ranked_t.to(torch.long).clamp(min=0, max=max(0, chunk_context_end - 1)),
                                        max(1, page_size),
                                        rounding_mode="floor",
                                    ).clamp(max=max(0, page_count - 1))
                                    ranked_valid = torch.isfinite(ranked_scores_for_attention)
                                page_mask = torch.zeros((chunk_positions, page_count), dtype=torch.bool, device=device)
                                if ranked_pages.numel() > 0:
                                    page_mask.scatter_(
                                        1,
                                        ranked_pages.reshape(chunk_positions, -1),
                                        ranked_valid.reshape(chunk_positions, -1),
                                    )
                                query_context_lens = torch.arange(
                                    chunk_query_start + 1,
                                    chunk_query_start + chunk_positions + 1,
                                    dtype=torch.long,
                                    device=device,
                                )
                                page_starts_grid = (
                                    torch.arange(page_count, dtype=torch.long, device=device).reshape(1, -1) * page_size
                                )
                                prefix_ends = torch.minimum(
                                    torch.full_like(query_context_lens, max(0, int(args.static_prefix))),
                                    query_context_lens,
                                )
                                indexed_ends = torch.maximum(
                                    prefix_ends,
                                    query_context_lens - max(0, int(args.static_suffix)),
                                )
                                sealed_ends = prefix_ends + (
                                    torch.div(
                                        torch.clamp(indexed_ends - prefix_ends, min=0),
                                        max(1, page_size),
                                        rounding_mode="floor",
                                    )
                                    * page_size
                                )
                                suffix_starts = torch.maximum(sealed_ends, prefix_ends)
                                prefix_mask = page_starts_grid < prefix_ends.reshape(-1, 1)
                                suffix_mask = (
                                    (page_starts_grid < query_context_lens.reshape(-1, 1))
                                    & ((page_starts_grid + page_size) > suffix_starts.reshape(-1, 1))
                                )
                                page_mask = page_mask | prefix_mask | suffix_mask
                                row_counts_t = page_mask.sum(dim=1, dtype=torch.int32)
                                indptr = torch.empty((chunk_positions + 1,), dtype=torch.int32, device=device)
                                indptr[0] = 0
                                indptr[1:] = torch.cumsum(row_counts_t, dim=0)
                                indices = page_mask.to(torch.int32).nonzero(as_tuple=False)[:, 1].to(torch.int32).contiguous()
                                q_flash = query_states[0, :, chunk_start:chunk_end, :].transpose(0, 1).contiguous()
                                k_flash = keys_all[:, :chunk_context_end, :].transpose(0, 1).contiguous()
                                v_flash = values_all[:, :chunk_context_end, :].transpose(0, 1).contiguous()
                                if int(k_flash.shape[0]) < padded_context_len:
                                    pad = padded_context_len - int(k_flash.shape[0])
                                    k_flash = torch.nn.functional.pad(k_flash, (0, 0, 0, 0, 0, pad))
                                    v_flash = torch.nn.functional.pad(v_flash, (0, 0, 0, 0, 0, pad))
                                workspace = getattr(self, "_pagedpq_flashinfer_workspace", None)
                                if not isinstance(workspace, torch.Tensor) or workspace.device != device:
                                    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
                                    setattr(self, "_pagedpq_flashinfer_workspace", workspace)
                                wrapper = flashinfer.BlockSparseAttentionWrapper(workspace)
                                wrapper.plan(
                                    indptr,
                                    indices,
                                    int(chunk_positions),
                                    int(padded_context_len),
                                    1,
                                    int(page_size),
                                    int(num_heads),
                                    int(num_kv_heads),
                                    int(self.head_dim),
                                    causal=True,
                                    sm_scale=float(self.head_dim) ** -0.5,
                                    q_data_type=q_flash.dtype,
                                    kv_data_type=k_flash.dtype,
                                )
                                outputs_chunk = wrapper.run(q_flash, k_flash, v_flash)
                                chunk_page_counts_cpu = row_counts_t.detach().cpu().tolist()
                            else:
                                outputs_chunk = native.gqa_causal_exact_selected_attention(
                                    queries_chunk,
                                    keys_all.contiguous(),
                                    values_all.contiguous(),
                                    ranked_t.contiguous(),
                                    ranked_scores_for_attention,
                                    int(group_size),
                                    int(chunk_query_start),
                                    int(args.static_prefix),
                                    int(args.static_suffix),
                                    int(args.page_size),
                                    float(self.head_dim) ** -0.5,
                                )
                            if bool(getattr(args, "profile_native_ops", False)):
                                _sync_if_cuda(device)
                                attention_seconds_total += float(time.perf_counter() - attention_t0)
                            page_costs = [
                                float(page.codebooks.numel() * int(key_bytes) + page.codes.numel() * (1 if int(args.subbits) <= 8 else 2))
                                for page in (chunk_indexes[0].pages if chunk_indexes else [])
                            ]
                            page_starts_cpu = [int(page.start) for page in (chunk_indexes[0].pages if chunk_indexes else [])]
                            page_size = int(args.page_size)
                            if (
                                accepted_budget_cost_upper_chunk is None
                                and prefill_selector_backend != "torch_matmul"
                                and chunk_positions > 0
                            ):
                                q_lens = np.arange(
                                    int(query_start + chunk_start + 1),
                                    int(query_start + chunk_end + 1),
                                    dtype=np.int64,
                                )
                                prefix_ends = np.minimum(max(0, int(args.static_prefix)), q_lens)
                                indexed_ends_q = np.maximum(
                                    prefix_ends,
                                    q_lens - max(0, int(args.static_suffix)),
                                )
                                sealed_ends_q = prefix_ends + (
                                    (np.maximum(0, indexed_ends_q - prefix_ends) // max(1, page_size))
                                    * max(1, page_size)
                                )
                                base_tail_starts = np.maximum(sealed_ends_q, prefix_ends)
                                base_counts = prefix_ends + np.maximum(0, q_lens - base_tail_starts)
                                valid_pages_np = (
                                    (np.maximum(0, sealed_ends_q - prefix_ends) // max(1, page_size))
                                    .clip(0, len(page_costs))
                                    .astype(np.int64, copy=False)
                                )
                                page_cost_prefix = np.concatenate(
                                    ([0.0], np.cumsum(np.asarray(page_costs, dtype=np.float64)))
                                )
                                selector_mbs = (
                                    np.zeros_like(valid_pages_np, dtype=np.float64)
                                    if int(budget) <= 0
                                    else page_cost_prefix[valid_pages_np] / MB
                                )
                                if chunk_page_counts_cpu is not None:
                                    selected_counts = (
                                        np.asarray(chunk_page_counts_cpu, dtype=np.float64) * float(max(1, page_size))
                                    )
                                else:
                                    ranked_counts = np.minimum(
                                        max(0, int(budget)),
                                        np.maximum(0, valid_pages_np * max(1, page_size)),
                                    ).astype(np.float64, copy=False)
                                    selected_counts = base_counts.astype(np.float64, copy=False) + ranked_counts
                                exact_kv_mbs = (
                                    selected_counts * float(int(self.head_dim) * (key_bytes + value_bytes)) / MB
                                )
                                stats[layer_id].add_count_repeated(
                                    int(num_heads * chunk_positions),
                                    float(selected_counts.mean()),
                                    0,
                                    float(selector_mbs.mean()),
                                    int(self.head_dim),
                                    key_bytes,
                                    value_bytes,
                                    tail_mb_override=0.0,
                                    exact_kv_mb_override=float(exact_kv_mbs.mean()),
                                )
                            else:
                                for local_qpos in range(chunk_start, chunk_end):
                                    query_context_len = int(query_start + local_qpos + 1)
                                    prefix_end = min(max(0, int(args.static_prefix)), query_context_len)
                                    indexed_end_q = max(prefix_end, query_context_len - max(0, int(args.static_suffix)))
                                    sealed_end_q = prefix_end + (
                                        (max(0, indexed_end_q - prefix_end) // max(1, page_size)) * max(1, page_size)
                                    )
                                    base_tail_start = max(sealed_end_q, prefix_end)
                                    base_count = int(prefix_end) + max(0, int(query_context_len) - int(base_tail_start))
                                    valid_pages = 0
                                    for page_start in page_starts_cpu:
                                        if int(page_start) >= int(prefix_end) and int(page_start) + page_size <= int(sealed_end_q):
                                            valid_pages += 1
                                    selector_mb = 0.0 if int(budget) <= 0 else float(sum(page_costs[:valid_pages])) / MB
                                    if prefill_selector_backend == "torch_matmul":
                                        selector_mb += float(valid_pages * page_size * int(self.head_dim) * 4) / MB
                                    ranked_count = min(max(0, int(budget)), max(0, valid_pages * page_size))
                                    if accepted_budget_cost_upper_chunk is not None:
                                        if accepted_budget_mean_by_pos_cpu_chunk is not None:
                                            ranked_count = float(accepted_budget_mean_by_pos_cpu_chunk[int(local_qpos - chunk_start)])
                                        else:
                                            ranked_count = float(accepted_budget_cost_upper_chunk)
                                        ranked_count = min(max(0.0, float(ranked_count)), float(max(0, valid_pages * page_size)))
                                    selected_count = int(base_count) + float(ranked_count)
                                    exact_kv_mb = float(selected_count * int(self.head_dim) * (key_bytes + value_bytes)) / MB
                                    stats[layer_id].add_count_repeated(
                                        num_heads,
                                        selected_count,
                                        0,
                                        float(selector_mb),
                                        int(self.head_dim),
                                        key_bytes,
                                        value_bytes,
                                        tail_mb_override=0.0,
                                        exact_kv_mb_override=exact_kv_mb,
                                    )
                            outputs_chunks.append(outputs_chunk)
                        if bool(getattr(args, "profile_native_ops", False)):
                            stats[layer_id].add_native_timing(
                                selector_seconds=selector_seconds_total,
                                attention_seconds=attention_seconds_total,
                            )
                        return torch.cat(outputs_chunks, dim=0).reshape(query_len, -1).to(hidden_states.dtype).contiguous()
                    except Exception:
                        if str(args.selector_backend) == "cuda_ext":
                            raise
                        return None
                if (
                    tail_vpq_prefill
                    and prefill_tail_score_reuse
                    and prefill_selector_backend in {"native", "torch_matmul"}
                    and prefill_chunk_size > 0
                    and query_len > prefill_chunk_size
                    and str(args.selected_value_mode) in {"exact", "vpq_value"}
                    and max(1, int(getattr(args, "prefill_selector_stride", 1))) == 1
                ):
                    try:
                        native = load_selector_paged_pq_ext()
                        codebooks, codes, page_starts = gqa_native_fullscan_pack(gqa_indexes)
                        value_packs = [
                            value_vpq_pack_torch(
                                index=index,
                                values=values_all[int(kv_head)],
                                value_subvecs=int(args.value_subvecs),
                                value_subbits=int(args.value_subbits),
                                key_bytes=int(value_bytes),
                                device=device,
                                value_group_pages=1,
                            )
                            for kv_head, index in enumerate(gqa_indexes)
                        ]
                        if any(pack is None for pack in value_packs):
                            return None
                        for index in gqa_indexes:
                            build_stats = getattr(index, "_last_value_vpq_build_stats", None)
                            if build_stats is not None:
                                build_seconds, build_read_mb, build_write_mb = build_stats
                                stats[layer_id].add_index_build(build_seconds, build_read_mb, build_write_mb)
                                setattr(index, "_last_value_vpq_build_stats", None)
                        value_codebooks = torch.stack([pack[0] for pack in value_packs if pack is not None], dim=0).contiguous()
                        value_codes = torch.stack([pack[1] for pack in value_packs if pack is not None], dim=0).contiguous()
                        value_page_size = int(value_packs[0][3])
                        if value_page_size != int(args.page_size):
                            return None

                        outputs_chunks = []
                        selector_seconds_total = 0.0
                        attention_seconds_total = 0.0
                        code_bytes = 1 if int(args.subbits) <= 8 else 2
                        page_costs = [
                            float(page.codebooks.numel() * int(key_bytes) + page.codes.numel() * code_bytes)
                            for page in gqa_indexes[0].pages
                        ]
                        page_starts_cpu = [int(page.start) for page in gqa_indexes[0].pages]
                        page_size = int(args.page_size)
                        actual_value_subbits = int(args.value_subbits) if int(args.value_subbits) > 0 else int(args.subbits)
                        value_code_bytes = 1 if int(actual_value_subbits) <= 8 else 2
                        value_subvecs = int(value_codebooks.shape[2])
                        value_subdim = int(value_codebooks.shape[-1])
                        value_centroids = int(value_codebooks.shape[3])
                        for chunk_start in range(0, query_len, prefill_chunk_size):
                            chunk_end = min(query_len, chunk_start + prefill_chunk_size)
                            chunk_query_start = int(query_start + chunk_start)
                            queries_chunk = q_all[:, chunk_start:chunk_end, :].transpose(0, 1).to(device).contiguous()
                            if bool(getattr(args, "profile_native_ops", False)):
                                _sync_if_cuda(device)
                                selector_t0 = time.perf_counter()
                            if prefill_selector_backend == "torch_matmul":
                                ranked_t, ranked_scores, dense_pq_scores, dense_pq_logsumexp = torch_matmul_prefill_topk_scores(
                                    queries_chunk,
                                    codebooks,
                                    codes,
                                    page_starts,
                                    local_query_start=int(chunk_query_start),
                                    need_dense_scores=True,
                                    need_dense_logsumexp=bool(proxy_confidence_prefill),
                                )
                            else:
                                ranked_t, ranked_scores, dense_pq_scores = native.gqa_causal_fullscan_pq_topk_scores(
                                    queries_chunk,
                                    codebooks,
                                    codes,
                                    page_starts,
                                    int(group_size),
                                    int(budget),
                                    int(chunk_query_start),
                                    int(args.static_prefix),
                                    int(args.static_suffix),
                                )
                                dense_pq_logsumexp = None
                            if bool(getattr(args, "profile_native_ops", False)):
                                _sync_if_cuda(device)
                                selector_seconds_total += float(time.perf_counter() - selector_t0)
                            ranked_t_eff = ranked_t.contiguous()
                            ranked_scores_eff = ranked_scores.contiguous()
                            accepted_budget_counts_chunk: torch.Tensor | None = None
                            accepted_budget_mean_by_pos_cpu_chunk: list[float] | None = None
                            accepted_budget_cost_upper_chunk: float | None = None
                            confidence_extra_attention_calls_chunk = 0
                            geometric_outputs_chunk: torch.Tensor | None = None
                            if proxy_confidence_prefill or ranked_confidence_prefill:
                                if ranked_scores.numel() == 0:
                                    return None
                                rank_count = int(ranked_scores.shape[2])
                                if rank_count <= 0:
                                    return None
                                max_budget = max(0, min(rank_count, int(args.geometric_max_budget)))
                                if max_budget <= 0:
                                    max_budget = rank_count
                                granularity = max(1, int(args.geometric_budget_granularity))
                                min_budget = _round_budget_up(
                                    int(args.geometric_min_budget),
                                    granularity=granularity,
                                    max_budget=max_budget,
                                )
                                proxy_target = max(
                                    float(args.tail_proxy_mass_min),
                                    1.0 - float(args.tail_proxy_mass_max),
                                )
                                proxy_target = min(max(float(proxy_target), 0.0), 1.0 - 1.0e-7)
                                scale = float(self.head_dim) ** -0.5
                                ranked_t_eff = ranked_t[:, :, :max_budget].contiguous()
                                ranked_scores_prefix = ranked_scores[:, :, :max_budget].contiguous()
                                top_scores = ranked_scores_prefix.float() * scale
                                denom = (
                                    torch.logsumexp(top_scores, dim=-1)
                                    if ranked_confidence_prefill
                                    else dense_pq_logsumexp
                                )
                                if denom is None:
                                    denom = torch.logsumexp(dense_pq_scores.float() * scale, dim=-1)
                                cum_mass = torch.exp(torch.logcumsumexp(top_scores, dim=-1) - denom.unsqueeze(-1))
                                hit = cum_mass >= proxy_target
                                has_hit = torch.any(hit, dim=-1)
                                first_hit = torch.argmax(hit.to(torch.int32), dim=-1).to(torch.long) + 1
                                accepted_budget_counts_chunk = torch.where(
                                    has_hit,
                                    first_hit,
                                    torch.full_like(first_hit, max_budget),
                                )
                                accepted_budget_counts_chunk = torch.clamp(
                                    accepted_budget_counts_chunk,
                                    min=int(min_budget),
                                    max=int(max_budget),
                                )
                                accepted_budget_counts_chunk = (
                                    torch.div(
                                        accepted_budget_counts_chunk + granularity - 1,
                                        granularity,
                                        rounding_mode="floor",
                                    )
                                    * granularity
                                ).clamp(max=int(max_budget))
                                accepted_budget_cost_upper_chunk = float(max_budget)
                                if ranked_confidence_cost_mode == "exact":
                                    accepted_budget_mean_by_pos_cpu_chunk = (
                                        accepted_budget_counts_chunk.float().mean(dim=1).detach().cpu().tolist()
                                    )
                                rank_ids = torch.arange(max_budget, dtype=torch.long, device=device).reshape(1, 1, max_budget)
                                ranked_scores_eff = ranked_scores_prefix.masked_fill(
                                    rank_ids >= accepted_budget_counts_chunk.unsqueeze(-1),
                                    float("-inf"),
                                ).contiguous()
                            elif probe_confidence_prefill:
                                if ranked_scores.numel() == 0:
                                    return None
                                rank_count = int(ranked_scores.shape[2])
                                if rank_count <= 0:
                                    return None
                                max_budget = max(0, min(rank_count, int(args.geometric_max_budget)))
                                if max_budget <= 0:
                                    max_budget = rank_count
                                granularity = max(1, int(args.geometric_budget_granularity))
                                growth = max(1.01, float(args.geometric_growth))
                                probe_scale = max(1.01, float(args.geometric_probe_scale))
                                k = _round_budget_up(
                                    int(args.geometric_min_budget),
                                    granularity=granularity,
                                    max_budget=max_budget,
                                )
                                ranked_t_prefix = ranked_t[:, :, :max_budget].contiguous()
                                ranked_scores_prefix = ranked_scores[:, :, :max_budget].contiguous()
                                rank_ids = torch.arange(max_budget, dtype=torch.long, device=device).reshape(1, 1, max_budget)
                                needs_proxy_gate = (
                                    float(args.tail_proxy_mass_min) > 0.0
                                    or float(args.tail_proxy_mass_max) < 1.0
                                    or float(args.tail_pq_corr_min) > -1.0
                                    or math.isfinite(float(args.tail_pq_relrmse_max))
                                )
                                selected_mass_in_kernel = (
                                    str(args.selected_value_exact_rule) == "selected_mass"
                                    and int(args.selected_value_max_exact_top) <= 0
                                )
                                exact_ranked_logits_for_conf: torch.Tensor | None = None
                                base_lse_for_conf: torch.Tensor | None = None
                                if needs_proxy_gate or (
                                    str(args.selected_value_exact_rule) == "selected_mass" and not selected_mass_in_kernel
                                ):
                                    exact_ranked_logits_for_conf = _gpu_gqa_ranked_exact_logits(
                                        queries=queries_chunk,
                                        keys_all=keys_all,
                                        ranked_tokens=ranked_t_prefix,
                                        group_size=int(group_size),
                                        scale=float(self.head_dim) ** -0.5,
                                        max_rank=int(max_budget),
                                    )
                                    base_lse_for_conf, _base_tokens_for_conf = _gpu_gqa_base_logsumexp_prefill(
                                        queries=queries_chunk,
                                        keys_all=keys_all,
                                        group_size=int(group_size),
                                        query_start=int(chunk_query_start),
                                        static_prefix=int(args.static_prefix),
                                        static_suffix=int(args.static_suffix),
                                        page_size=int(args.page_size),
                                        scale=float(self.head_dim) ** -0.5,
                                    )

                                def mask_prefill_scores(keep: int) -> torch.Tensor:
                                    keep_i = max(0, min(max_budget, int(keep)))
                                    return ranked_scores_prefix.masked_fill(rank_ids >= keep_i, float("-inf")).contiguous()

                                def prefill_selected_tail(masked_scores: torch.Tensor, blend: float) -> torch.Tensor:
                                    if str(args.selected_value_mode) == "exact":
                                        return native.gqa_causal_vpq_tail_from_scores(
                                            queries_chunk,
                                            keys_all_float().contiguous(),
                                            values_all_float().contiguous(),
                                            dense_pq_scores.contiguous(),
                                            value_codebooks,
                                            value_codes,
                                            page_starts,
                                            ranked_t_prefix,
                                            masked_scores,
                                            int(group_size),
                                            int(chunk_query_start),
                                            int(args.static_prefix),
                                            int(args.static_suffix),
                                            int(args.page_size),
                                            float(self.head_dim) ** -0.5,
                                            float(blend),
                                        )
                                    if str(args.selected_value_exact_rule) == "selected_mass":
                                        if selected_mass_in_kernel:
                                            mass_fn = (
                                                native.gqa_causal_vpq_selected_tail_from_scores_mass_min
                                                if int(args.selected_value_min_exact_top) > 0
                                                else native.gqa_causal_vpq_selected_tail_from_scores_mass
                                            )
                                            mass_args = (
                                                queries_chunk,
                                                keys_all.contiguous(),
                                                values_all.contiguous(),
                                                dense_pq_scores.contiguous(),
                                                value_codebooks,
                                                value_codes,
                                                page_starts,
                                                ranked_t_prefix,
                                                masked_scores,
                                                float(args.selected_value_exact_mass),
                                            )
                                            if int(args.selected_value_min_exact_top) > 0:
                                                mass_args = mass_args + (int(args.selected_value_min_exact_top),)
                                            return mass_fn(
                                                *mass_args,
                                                int(group_size),
                                                int(chunk_query_start),
                                                int(args.static_prefix),
                                                int(args.static_suffix),
                                                int(args.page_size),
                                                float(self.head_dim) ** -0.5,
                                                float(blend),
                                            )
                                        if exact_ranked_logits_for_conf is None:
                                            raise RuntimeError("missing exact ranked logits for prefill selected_mass exact-V counts")
                                        exact_value_counts = selected_value_exact_counts_from_mass_gpu(
                                            ranked_logits=exact_ranked_logits_for_conf,
                                            ranked_scores=masked_scores,
                                            base_logsumexp=base_lse_for_conf,
                                            exact_mass=float(args.selected_value_exact_mass),
                                            min_top=int(args.selected_value_min_exact_top),
                                            max_top=int(args.selected_value_max_exact_top),
                                        ).contiguous()
                                        return native.gqa_causal_vpq_selected_tail_from_scores_counts(
                                            queries_chunk,
                                            keys_all.contiguous(),
                                            values_all.contiguous(),
                                            dense_pq_scores.contiguous(),
                                            value_codebooks,
                                            value_codes,
                                            page_starts,
                                            ranked_t_prefix,
                                            masked_scores,
                                            exact_value_counts,
                                            int(group_size),
                                            int(chunk_query_start),
                                            int(args.static_prefix),
                                            int(args.static_suffix),
                                            int(args.page_size),
                                            0,
                                            float(self.head_dim) ** -0.5,
                                            float(blend),
                                        )
                                    return native.gqa_causal_vpq_selected_tail_from_scores(
                                        queries_chunk,
                                        keys_all.contiguous(),
                                        values_all.contiguous(),
                                        dense_pq_scores.contiguous(),
                                        value_codebooks,
                                        value_codes,
                                        page_starts,
                                        ranked_t_prefix,
                                        masked_scores,
                                        int(group_size),
                                        int(chunk_query_start),
                                        int(args.static_prefix),
                                        int(args.static_suffix),
                                        int(args.page_size),
                                        native_selected_value_exact_top_arg(args),
                                        float(self.head_dim) ** -0.5,
                                        float(blend),
                                    )

                                native_vpq_geometric_exact_top: int | None = None
                                if (
                                    str(args.selected_value_mode) == "vpq_value"
                                    and not needs_proxy_gate
                                    and hasattr(native, "gqa_causal_geometric_accept_counts_vpq")
                                ):
                                    if str(args.selected_value_exact_rule) == "selector_rank":
                                        native_vpq_geometric_exact_top = native_selected_value_exact_top_arg(args)
                                    elif (
                                        str(args.selected_value_exact_rule) == "fixed"
                                        and selected_value_exact_top_positive(args) <= 0
                                    ):
                                        native_vpq_geometric_exact_top = 0
                                if (
                                    str(args.selected_value_mode) == "exact"
                                    and not needs_proxy_gate
                                    and hasattr(native, "gqa_causal_geometric_accept_counts")
                                ):
                                    if bool(getattr(args, "profile_native_ops", False)):
                                        _sync_if_cuda(device)
                                        attention_t0 = time.perf_counter()
                                    accepted_budget_counts_chunk = native.gqa_causal_geometric_accept_counts(
                                        queries_chunk,
                                        keys_all_float().contiguous(),
                                        values_all_float().contiguous(),
                                        dense_pq_scores.contiguous(),
                                        value_codebooks,
                                        value_codes,
                                        page_starts,
                                        ranked_t_prefix,
                                        ranked_scores_prefix,
                                        int(group_size),
                                        int(chunk_query_start),
                                        int(args.static_prefix),
                                        int(args.static_suffix),
                                        int(args.page_size),
                                        int(args.geometric_min_budget),
                                        int(max_budget),
                                        int(granularity),
                                        float(growth),
                                        float(probe_scale),
                                        float(args.tail_probe_rel_l2_max),
                                        float(self.head_dim) ** -0.5,
                                    )
                                    confidence_extra_attention_calls_chunk += 2
                                    if bool(getattr(args, "profile_native_ops", False)):
                                        _sync_if_cuda(device)
                                        attention_seconds_total += float(time.perf_counter() - attention_t0)
                                elif native_vpq_geometric_exact_top is not None:
                                    if bool(getattr(args, "profile_native_ops", False)):
                                        _sync_if_cuda(device)
                                        attention_t0 = time.perf_counter()
                                    accepted_budget_counts_chunk = native.gqa_causal_geometric_accept_counts_vpq(
                                        queries_chunk,
                                        keys_all.contiguous(),
                                        values_all.contiguous(),
                                        dense_pq_scores.contiguous(),
                                        value_codebooks,
                                        value_codes,
                                        page_starts,
                                        ranked_t_prefix,
                                        ranked_scores_prefix,
                                        int(group_size),
                                        int(chunk_query_start),
                                        int(args.static_prefix),
                                        int(args.static_suffix),
                                        int(args.page_size),
                                        int(args.geometric_min_budget),
                                        int(max_budget),
                                        int(granularity),
                                        float(growth),
                                        float(probe_scale),
                                        float(args.tail_probe_rel_l2_max),
                                        int(native_vpq_geometric_exact_top),
                                        float(self.head_dim) ** -0.5,
                                    )
                                    confidence_extra_attention_calls_chunk += 2
                                    if bool(getattr(args, "profile_native_ops", False)):
                                        _sync_if_cuda(device)
                                        attention_seconds_total += float(time.perf_counter() - attention_t0)

                                if accepted_budget_counts_chunk is None:
                                    unresolved = torch.ones(
                                        (int(queries_chunk.shape[0]), int(num_heads)),
                                        dtype=torch.bool,
                                        device=device,
                                    )
                                    geometric_outputs_chunk = torch.empty(
                                        (int(queries_chunk.shape[0]), int(num_heads), int(self.head_dim)),
                                        dtype=torch.float32,
                                        device=device,
                                    )
                                    accepted_budget_counts_chunk = torch.full(
                                        (int(queries_chunk.shape[0]), int(num_heads)),
                                        int(max_budget),
                                        dtype=torch.long,
                                        device=device,
                                    )
                                    while True:
                                        tail_budget = min(max_budget, int(k))
                                        probe_budget = _round_budget_up(
                                            max(float(tail_budget + granularity), probe_scale * float(tail_budget)),
                                            granularity=granularity,
                                            max_budget=max_budget,
                                        )
                                        probe_budget = max(tail_budget, int(probe_budget))
                                        tail_ranked_scores = mask_prefill_scores(tail_budget)
                                        probe_ranked_scores = mask_prefill_scores(probe_budget)
                                        approx_tail = prefill_selected_tail(tail_ranked_scores, 1.0)
                                        probe_only = prefill_selected_tail(probe_ranked_scores, 0.0)
                                        confidence_extra_attention_calls_chunk += 2
                                        rel = torch.linalg.vector_norm(
                                            (approx_tail - probe_only).float(),
                                            dim=-1,
                                        ) / torch.clamp(
                                            torch.linalg.vector_norm(probe_only.float(), dim=-1),
                                            min=1.0e-20,
                                        )
                                        gate = rel <= float(args.tail_probe_rel_l2_max)
                                        if needs_proxy_gate:
                                            if exact_ranked_logits_for_conf is None:
                                                raise RuntimeError("missing exact ranked logits for prefill geometric proxy confidence")
                                            proxy_mass, proxy_tail_mass, tail_pq_corr, tail_pq_relrmse = _gpu_proxy_confidence_metrics(
                                                ranked_scores=ranked_scores_prefix,
                                                exact_ranked_logits=exact_ranked_logits_for_conf,
                                                keep_count=int(tail_budget),
                                                max_budget=int(max_budget),
                                                query_dim=int(self.head_dim),
                                                base_logsumexp=base_lse_for_conf,
                                                calibrate=str(args.tail_score_calibration) == "affine_selected",
                                            )
                                            gate = (
                                                gate
                                                & (proxy_mass >= float(args.tail_proxy_mass_min))
                                                & (proxy_tail_mass <= float(args.tail_proxy_mass_max))
                                                & (tail_pq_corr >= float(args.tail_pq_corr_min))
                                                & (tail_pq_relrmse <= float(args.tail_pq_relrmse_max))
                                            )
                                        passed = gate & unresolved
                                        if bool(torch.any(passed)):
                                            candidate = prefill_selected_tail(probe_ranked_scores, float(tail_blend_value))
                                            confidence_extra_attention_calls_chunk += 1
                                            geometric_outputs_chunk = torch.where(
                                                passed.unsqueeze(-1),
                                                candidate,
                                                geometric_outputs_chunk,
                                            )
                                            accepted_budget_counts_chunk = torch.where(
                                                passed,
                                                torch.full_like(accepted_budget_counts_chunk, int(probe_budget)),
                                                accepted_budget_counts_chunk,
                                            )
                                        unresolved = unresolved & ~passed
                                        if not bool(torch.any(unresolved)):
                                            break
                                        if probe_budget >= max_budget:
                                            geometric_outputs_chunk = torch.where(
                                                unresolved.unsqueeze(-1),
                                                probe_only,
                                                geometric_outputs_chunk,
                                            )
                                            break
                                        next_k = _round_budget_up(
                                            max(float(probe_budget + granularity), growth * float(probe_budget)),
                                            granularity=granularity,
                                            max_budget=max_budget,
                                        )
                                        if int(next_k) <= int(probe_budget):
                                            geometric_outputs_chunk = torch.where(
                                                unresolved.unsqueeze(-1),
                                                probe_only,
                                                geometric_outputs_chunk,
                                            )
                                            break
                                        k = int(next_k)
                                if ranked_confidence_cost_mode == "exact":
                                    accepted_budget_mean_by_pos_cpu_chunk = (
                                        accepted_budget_counts_chunk.float().mean(dim=1).detach().cpu().tolist()
                                    )
                                else:
                                    accepted_budget_cost_upper_chunk = float(max_budget)
                                ranked_t_eff = ranked_t_prefix
                                ranked_scores_eff = ranked_scores_prefix.masked_fill(
                                    rank_ids >= accepted_budget_counts_chunk.unsqueeze(-1),
                                    float("-inf"),
                                ).contiguous()
                            exact_value_counts_chunk: torch.Tensor | None = None
                            exact_value_counts_mean_by_pos_cpu_chunk: list[float] | None = None
                            selected_mass_in_kernel_chunk = (
                                str(args.selected_value_mode) == "vpq_value"
                                and str(args.selected_value_exact_rule) == "selected_mass"
                                and int(args.selected_value_max_exact_top) <= 0
                            )
                            if (
                                str(args.selected_value_mode) == "vpq_value"
                                and str(args.selected_value_exact_rule) == "selected_mass"
                                and not selected_mass_in_kernel_chunk
                            ):
                                exact_ranked_logits_chunk = _gpu_gqa_ranked_exact_logits(
                                    queries=queries_chunk,
                                    keys_all=keys_all,
                                    ranked_tokens=ranked_t_eff,
                                    group_size=int(group_size),
                                    scale=float(self.head_dim) ** -0.5,
                                    max_rank=int(ranked_scores_eff.shape[-1]),
                                )
                                base_lse_chunk, base_tokens_for_counts = _gpu_gqa_base_logsumexp_prefill(
                                    queries=queries_chunk,
                                    keys_all=keys_all,
                                    group_size=int(group_size),
                                    query_start=int(chunk_query_start),
                                    static_prefix=int(args.static_prefix),
                                    static_suffix=int(args.static_suffix),
                                    page_size=int(args.page_size),
                                    scale=float(self.head_dim) ** -0.5,
                                )
                                exact_value_counts_chunk = selected_value_exact_counts_from_mass_gpu(
                                    ranked_logits=exact_ranked_logits_chunk,
                                    ranked_scores=ranked_scores_eff,
                                    base_logsumexp=base_lse_chunk,
                                    exact_mass=float(args.selected_value_exact_mass),
                                    min_top=int(args.selected_value_min_exact_top),
                                    max_top=int(args.selected_value_max_exact_top),
                                ).contiguous()
                                exact_value_counts_mean_by_pos_cpu_chunk = (
                                    exact_value_counts_chunk.float().mean(dim=1).detach().cpu().tolist()
                                )
                            if geometric_outputs_chunk is not None:
                                outputs_chunk = geometric_outputs_chunk
                            else:
                                if bool(getattr(args, "profile_native_ops", False)):
                                    _sync_if_cuda(device)
                                    attention_t0 = time.perf_counter()
                                if selected_mass_in_kernel_chunk:
                                    mass_fn = (
                                        native.gqa_causal_vpq_selected_tail_from_scores_mass_min
                                        if int(args.selected_value_min_exact_top) > 0
                                        else native.gqa_causal_vpq_selected_tail_from_scores_mass
                                    )
                                    mass_args = (
                                        queries_chunk,
                                        keys_all.contiguous(),
                                        values_all.contiguous(),
                                        dense_pq_scores.contiguous(),
                                        value_codebooks,
                                        value_codes,
                                        page_starts,
                                        ranked_t_eff,
                                        ranked_scores_eff,
                                        float(args.selected_value_exact_mass),
                                    )
                                    if int(args.selected_value_min_exact_top) > 0:
                                        mass_args = mass_args + (int(args.selected_value_min_exact_top),)
                                    outputs_chunk = mass_fn(
                                        *mass_args,
                                        int(group_size),
                                        int(chunk_query_start),
                                        int(args.static_prefix),
                                        int(args.static_suffix),
                                        int(args.page_size),
                                        float(self.head_dim) ** -0.5,
                                        float(tail_blend_value),
                                    )
                                elif exact_value_counts_chunk is not None:
                                    outputs_chunk = native.gqa_causal_vpq_selected_tail_from_scores_counts(
                                        queries_chunk,
                                        keys_all.contiguous(),
                                        values_all.contiguous(),
                                        dense_pq_scores.contiguous(),
                                        value_codebooks,
                                        value_codes,
                                        page_starts,
                                        ranked_t_eff,
                                        ranked_scores_eff,
                                        exact_value_counts_chunk,
                                        int(group_size),
                                        int(chunk_query_start),
                                        int(args.static_prefix),
                                        int(args.static_suffix),
                                        int(args.page_size),
                                        0,
                                        float(self.head_dim) ** -0.5,
                                        float(tail_blend_value),
                                    )
                                else:
                                    if str(args.selected_value_mode) == "exact":
                                        outputs_chunk = native.gqa_causal_vpq_tail_from_scores(
                                            queries_chunk,
                                            keys_all_float().contiguous(),
                                            values_all_float().contiguous(),
                                            dense_pq_scores.contiguous(),
                                            value_codebooks,
                                            value_codes,
                                            page_starts,
                                            ranked_t_eff,
                                            ranked_scores_eff,
                                            int(group_size),
                                            int(chunk_query_start),
                                            int(args.static_prefix),
                                            int(args.static_suffix),
                                            int(args.page_size),
                                            float(self.head_dim) ** -0.5,
                                            float(tail_blend_value),
                                        )
                                    else:
                                        outputs_chunk = native.gqa_causal_vpq_selected_tail_from_scores(
                                            queries_chunk,
                                            keys_all.contiguous(),
                                            values_all.contiguous(),
                                            dense_pq_scores.contiguous(),
                                            value_codebooks,
                                            value_codes,
                                            page_starts,
                                            ranked_t_eff,
                                            ranked_scores_eff,
                                            int(group_size),
                                            int(chunk_query_start),
                                            int(args.static_prefix),
                                            int(args.static_suffix),
                                            int(args.page_size),
                                            native_selected_value_exact_top_arg(args),
                                            float(self.head_dim) ** -0.5,
                                            float(tail_blend_value),
                                        )
                                if bool(getattr(args, "profile_native_ops", False)):
                                    _sync_if_cuda(device)
                                    attention_seconds_total += float(time.perf_counter() - attention_t0)
                            for local_qpos in range(chunk_start, chunk_end):
                                query_context_len = int(query_start + local_qpos + 1)
                                prefix_end = min(max(0, int(args.static_prefix)), query_context_len)
                                indexed_end_q = max(prefix_end, query_context_len - max(0, int(args.static_suffix)))
                                sealed_end_q = prefix_end + (
                                    (max(0, indexed_end_q - prefix_end) // max(1, page_size)) * max(1, page_size)
                                )
                                base_tail_start = max(sealed_end_q, prefix_end)
                                base_count = int(prefix_end) + max(0, int(query_context_len) - int(base_tail_start))
                                valid_pages = 0
                                for page_start in page_starts_cpu:
                                    if int(page_start) >= int(prefix_end) and int(page_start) + page_size <= int(sealed_end_q):
                                        valid_pages += 1
                                selector_mb = 0.0 if int(budget) <= 0 else float(sum(page_costs[:valid_pages])) / MB
                                if prefill_selector_backend == "torch_matmul":
                                    selector_mb += float(valid_pages * page_size * int(self.head_dim) * 4) / MB
                                dense_score_write_bytes = int(len(page_starts_cpu)) * page_size * 4
                                dense_score_tail_read_bytes = int(valid_pages) * page_size * 4 * 2
                                selector_mb += float(dense_score_write_bytes + dense_score_tail_read_bytes) / MB
                                ranked_count = min(max(0, int(budget)), max(0, valid_pages * page_size))
                                if accepted_budget_counts_chunk is not None:
                                    if accepted_budget_mean_by_pos_cpu_chunk is not None:
                                        ranked_count = float(accepted_budget_mean_by_pos_cpu_chunk[int(local_qpos - chunk_start)])
                                    else:
                                        ranked_count = float(accepted_budget_cost_upper_chunk or ranked_count)
                                    ranked_count = min(max(0.0, ranked_count), float(max(0, valid_pages * page_size)))
                                selected_count = int(base_count) + float(ranked_count)
                                tail_count = max(0, int(valid_pages * page_size) - int(ranked_count))
                                exact_value_top = (
                                    float(ranked_count)
                                    if str(args.selected_value_mode) == "exact"
                                    else min(
                                        max(0.0, float(ranked_count)),
                                        float(max(0, int(args.selected_value_exact_top))),
                                    )
                                )
                                if exact_value_counts_mean_by_pos_cpu_chunk is not None:
                                    exact_value_top = min(
                                        max(0.0, float(ranked_count)),
                                        float(exact_value_counts_mean_by_pos_cpu_chunk[int(local_qpos - chunk_start)]),
                                    )
                                    selector_mb += (
                                        float((int(ranked_scores_eff.shape[-1]) + int(base_count)) * int(self.head_dim) * key_bytes)
                                        / MB
                                    )
                                elif selected_mass_in_kernel_chunk:
                                    # The in-kernel mass path avoids a separate exact-logit/count pass.
                                    # Charge an exact-V upper bound rather than hiding unknown data-dependent V reads.
                                    exact_value_top = max(0.0, float(ranked_count))
                                compressed_selected_values = max(0.0, float(ranked_count) - float(exact_value_top))
                                tail_mb = (
                                    float(valid_pages * value_subvecs * value_centroids * value_subdim * value_bytes)
                                    + float((tail_count + compressed_selected_values) * value_subvecs * value_code_bytes)
                                ) / MB
                                exact_kv_mb = (
                                    float(selected_count * int(self.head_dim) * key_bytes)
                                    + float((base_count + exact_value_top) * int(self.head_dim) * value_bytes)
                                ) / MB
                                confidence_mb = max(0, int(confidence_extra_attention_calls_chunk) - 1) * float(
                                    exact_kv_mb + tail_mb
                                )
                                stats[layer_id].add_count_repeated(
                                    num_heads,
                                    selected_count,
                                    tail_count,
                                    float(selector_mb),
                                    int(self.head_dim),
                                    key_bytes,
                                    value_bytes,
                                    tail_mb_override=tail_mb,
                                    exact_kv_mb_override=exact_kv_mb,
                                    confidence_mb_override=confidence_mb,
                                )
                            outputs_chunks.append(outputs_chunk)
                        if bool(getattr(args, "profile_native_ops", False)):
                            stats[layer_id].add_native_timing(
                                selector_seconds=selector_seconds_total,
                                attention_seconds=attention_seconds_total,
                            )
                        return torch.cat(outputs_chunks, dim=0).reshape(query_len, -1).to(hidden_states.dtype).contiguous()
                    except Exception:
                        if str(args.selector_backend) == "cuda_ext":
                            raise
                        return None
                try:
                    native = load_selector_paged_pq_ext()
                    codebooks, codes, page_starts = gqa_native_fullscan_pack(gqa_indexes)
                    queries = q_all.transpose(0, 1).to(device).contiguous()
                    if bool(getattr(args, "profile_native_ops", False)):
                        _sync_if_cuda(device)
                        selector_t0 = time.perf_counter()
                    prefill_selector_stride = max(1, int(getattr(args, "prefill_selector_stride", 1)))
                    prefill_selector_tile_size = max(0, int(getattr(args, "prefill_selector_tile_size", 0)))
                    if prefill_tail_score_reuse and not tail_vpq_prefill:
                        raise RuntimeError("prefill_tail_score_reuse requires tail_vpq_prefill")
                    if prefill_tail_score_reuse and prefill_selector_backend not in {"native", "torch_matmul"}:
                        raise RuntimeError(
                            "prefill_tail_score_reuse currently requires prefill_selector_backend=native or torch_matmul"
                        )
                    selector_mb_per_q_override: list[float] | None = None
                    dense_pq_scores = None
                    dense_pq_logsumexp = None
                    dense_score_fullscan_extra = False
                    if (exact_prefill or selected_vpq_prefill or tail_vpq_prefill) and prefill_selector_stride > 1:
                        ranked_chunks = []
                        score_chunks = []
                        selector_mb_per_q_override = [0.0 for _ in range(query_len)]
                        page_costs_for_reuse = [
                            float(
                                page.codebooks.numel() * int(key_bytes)
                                + page.codes.numel() * (1 if int(args.subbits) <= 8 else 2)
                            )
                            for page in gqa_indexes[0].pages
                        ]
                        page_starts_for_reuse = [int(page.start) for page in gqa_indexes[0].pages]
                        anchor_positions = list(range(0, query_len, prefill_selector_stride))
                        anchor_pos_t = torch.as_tensor(anchor_positions, dtype=torch.long, device=device)
                        anchor_queries = queries.index_select(0, anchor_pos_t)
                        anchor_context_lens_t = anchor_pos_t + int(query_start) + 1
                        if int(budget) <= 0:
                            anchor_ranked_all = torch.empty(
                                (len(anchor_positions), num_heads, 0), dtype=torch.long, device=device
                            )
                            anchor_scores_all = torch.empty(
                                (len(anchor_positions), num_heads, 0), dtype=torch.float32, device=device
                            )
                        elif prefill_selector_backend in {"torch_lut", "torch_lut_fp16", "torch_lut_batched", "torch_lut_streaming"}:
                            anchor_ranked_all, anchor_scores_all = torch_lut_prefill_topk_context_lens(
                                anchor_queries,
                                anchor_context_lens_t,
                                codebooks,
                                codes,
                                page_starts,
                                streaming=prefill_selector_backend == "torch_lut_streaming",
                                score_dtype=torch.float16 if prefill_selector_backend == "torch_lut_fp16" else torch.float32,
                            )
                        else:
                            anchor_rows = []
                            anchor_score_rows = []
                            for anchor_idx, block_start in enumerate(anchor_positions):
                                anchor_query_start = int(query_start + block_start)
                                anchor_ranked, anchor_scores = native.gqa_causal_fullscan_pq_topk(
                                    anchor_queries[anchor_idx : anchor_idx + 1],
                                    codebooks,
                                    codes,
                                    page_starts,
                                    int(group_size),
                                    int(budget),
                                    anchor_query_start,
                                    int(args.static_prefix),
                                    int(args.static_suffix),
                                )
                                anchor_rows.append(anchor_ranked)
                                anchor_score_rows.append(anchor_scores)
                            anchor_ranked_all = torch.cat(anchor_rows, dim=0)
                            anchor_scores_all = torch.cat(anchor_score_rows, dim=0)
                        for anchor_idx, block_start in enumerate(anchor_positions):
                            block_end = min(query_len, block_start + prefill_selector_stride)
                            block_len = int(block_end - block_start)
                            anchor_query_start = int(query_start + block_start)
                            anchor_ranked = anchor_ranked_all[anchor_idx : anchor_idx + 1]
                            anchor_scores = anchor_scores_all[anchor_idx : anchor_idx + 1]
                            ranked_chunks.append(anchor_ranked.expand(block_len, -1, -1).contiguous())
                            score_chunks.append(anchor_scores.expand(block_len, -1, -1).contiguous())
                            anchor_context_len = int(anchor_query_start + 1)
                            prefix_end = min(max(0, int(args.static_prefix)), anchor_context_len)
                            indexed_end_q = max(prefix_end, anchor_context_len - max(0, int(args.static_suffix)))
                            sealed_end_q = prefix_end + (
                                (max(0, indexed_end_q - prefix_end) // max(1, int(args.page_size)))
                                * max(1, int(args.page_size))
                            )
                            valid_pages = 0
                            for page_start in page_starts_for_reuse:
                                if int(page_start) >= int(prefix_end) and int(page_start) + int(args.page_size) <= int(sealed_end_q):
                                    valid_pages += 1
                            block_selector_mb = (
                                0.0 if int(budget) <= 0 else float(sum(page_costs_for_reuse[:valid_pages])) / MB
                            )
                            per_query_selector_mb = float(block_selector_mb) / max(1, block_len)
                            for local_qpos in range(block_start, block_end):
                                selector_mb_per_q_override[int(local_qpos)] = per_query_selector_mb
                        ranked_t = torch.cat(ranked_chunks, dim=0) if ranked_chunks else torch.empty((0, num_heads, 0), dtype=torch.long, device=device)
                        ranked_scores = (
                            torch.cat(score_chunks, dim=0)
                            if score_chunks
                            else torch.empty((0, num_heads, 0), dtype=torch.float32, device=device)
                        )
                        if prefill_tail_score_reuse:
                            _, _, dense_pq_scores = native.gqa_causal_fullscan_pq_topk_scores(
                                queries,
                                codebooks,
                                codes,
                                page_starts,
                                int(group_size),
                                0,
                                int(query_start),
                                int(args.static_prefix),
                                int(args.static_suffix),
                            )
                            dense_score_fullscan_extra = True
                    elif prefill_selector_backend == "native_fused":
                        ranked_t, ranked_scores = native.gqa_causal_fullscan_pq_topk_fused(
                            queries,
                            codebooks,
                            codes,
                            page_starts,
                            int(group_size),
                            int(budget),
                            int(query_start),
                            int(args.static_prefix),
                            int(args.static_suffix),
                        )
                    elif (
                        prefill_selector_backend == "native"
                        and prefill_selector_tile_size > 0
                        and not prefill_tail_score_reuse
                    ):
                        ranked_chunks = []
                        score_chunks = []
                        for tile_start in range(0, int(query_len), int(prefill_selector_tile_size)):
                            tile_end = min(int(query_len), tile_start + int(prefill_selector_tile_size))
                            ranked_chunk, score_chunk = native.gqa_causal_fullscan_pq_topk(
                                queries[tile_start:tile_end],
                                codebooks,
                                codes,
                                page_starts,
                                int(group_size),
                                int(budget),
                                int(query_start + tile_start),
                                int(args.static_prefix),
                                int(args.static_suffix),
                            )
                            ranked_chunks.append(ranked_chunk)
                            score_chunks.append(score_chunk)
                        ranked_t = (
                            torch.cat(ranked_chunks, dim=0)
                            if ranked_chunks
                            else torch.empty((0, num_heads, 0), dtype=torch.long, device=device)
                        )
                        ranked_scores = (
                            torch.cat(score_chunks, dim=0)
                            if score_chunks
                            else torch.empty((0, num_heads, 0), dtype=torch.float32, device=device)
                        )
                    elif prefill_selector_backend == "torch_lut_batched":
                        ranked_t, ranked_scores = torch_lut_batched_prefill_topk(
                            queries,
                            codebooks,
                            codes,
                            page_starts,
                            local_query_start=int(query_start),
                        )
                    elif prefill_selector_backend in {"torch_lut", "torch_lut_fp16", "torch_lut_streaming"}:
                        ranked_t, ranked_scores = torch_lut_prefill_topk(
                            queries,
                            codebooks,
                            codes,
                            page_starts,
                            local_query_start=int(query_start),
                            streaming=prefill_selector_backend == "torch_lut_streaming",
                            score_dtype=torch.float16 if prefill_selector_backend == "torch_lut_fp16" else torch.float32,
                        )
                    elif prefill_selector_backend == "torch_matmul":
                        ranked_t, ranked_scores, dense_pq_scores, dense_pq_logsumexp = torch_matmul_prefill_topk_scores(
                            queries,
                            codebooks,
                            codes,
                            page_starts,
                            local_query_start=int(query_start),
                            need_dense_scores=bool(prefill_tail_score_reuse),
                            need_dense_logsumexp=bool(proxy_confidence_prefill),
                        )
                    elif prefill_tail_score_reuse:
                        ranked_t, ranked_scores, dense_pq_scores = native.gqa_causal_fullscan_pq_topk_scores(
                            queries,
                            codebooks,
                            codes,
                            page_starts,
                            int(group_size),
                            int(budget),
                            int(query_start),
                            int(args.static_prefix),
                            int(args.static_suffix),
                        )
                    else:
                        ranked_t, ranked_scores = native.gqa_causal_fullscan_pq_topk(
                            queries,
                            codebooks,
                            codes,
                            page_starts,
                            int(group_size),
                            int(budget),
                            int(query_start),
                            int(args.static_prefix),
                            int(args.static_suffix),
                        )
                    selector_seconds = 0.0
                    if bool(getattr(args, "profile_native_ops", False)):
                        _sync_if_cuda(device)
                        selector_seconds = float(time.perf_counter() - selector_t0)
                        attention_t0 = time.perf_counter()
                    prefill_attention_backend = str(getattr(args, "prefill_attention_backend", "native"))
                    if exact_prefill and prefill_attention_backend in {"flashinfer_blocksparse", "flashinfer_page_blocks"}:
                        try:
                            import flashinfer  # type: ignore
                        except Exception as exc:
                            raise RuntimeError(f"prefill_attention_backend={prefill_attention_backend} requires flashinfer") from exc
                        page_size = int(args.page_size)
                        if context_len <= 0:
                            return None
                        if prefill_attention_backend == "flashinfer_page_blocks":
                            page_count = int((context_len + page_size - 1) // page_size)
                            padded_context_len = int(page_count * page_size)
                            ranked_pages = torch.div(
                                ranked_t.to(torch.long).clamp(min=0, max=max(0, context_len - 1)),
                                max(1, page_size),
                                rounding_mode="floor",
                            ).clamp(max=max(0, page_count - 1))
                            ranked_valid = torch.isfinite(ranked_scores)
                            page_mask = torch.zeros((query_len, page_count), dtype=torch.bool, device=device)
                            if ranked_pages.numel() > 0:
                                page_mask.scatter_(
                                    1,
                                    ranked_pages.reshape(query_len, -1),
                                    ranked_valid.reshape(query_len, -1),
                                )
                            query_context_lens = torch.arange(
                                query_start + 1,
                                query_start + query_len + 1,
                                dtype=torch.long,
                                device=device,
                            )
                            page_starts_grid = (
                                torch.arange(page_count, dtype=torch.long, device=device).reshape(1, -1) * page_size
                            )
                            prefix_ends = torch.minimum(
                                torch.full_like(query_context_lens, max(0, int(args.static_prefix))),
                                query_context_lens,
                            )
                            indexed_ends = torch.maximum(
                                prefix_ends,
                                query_context_lens - max(0, int(args.static_suffix)),
                            )
                            sealed_ends = prefix_ends + (
                                torch.div(
                                    torch.clamp(indexed_ends - prefix_ends, min=0),
                                    max(1, page_size),
                                    rounding_mode="floor",
                                )
                                * page_size
                            )
                            suffix_starts = torch.maximum(sealed_ends, prefix_ends)
                            prefix_mask = page_starts_grid < prefix_ends.reshape(-1, 1)
                            suffix_mask = (
                                (page_starts_grid < query_context_lens.reshape(-1, 1))
                                & ((page_starts_grid + page_size) > suffix_starts.reshape(-1, 1))
                            )
                            page_mask = page_mask | prefix_mask | suffix_mask
                            row_counts_t = page_mask.sum(dim=1, dtype=torch.int32)
                            indptr = torch.empty((query_len + 1,), dtype=torch.int32, device=device)
                            indptr[0] = 0
                            indptr[1:] = torch.cumsum(row_counts_t, dim=0)
                            indices = page_mask.to(torch.int32).nonzero(as_tuple=False)[:, 1].to(torch.int32).contiguous()
                            q_flash = query_states[0].transpose(0, 1).contiguous()
                            k_flash = keys_all.transpose(0, 1).contiguous()
                            v_flash = values_all.transpose(0, 1).contiguous()
                            if int(k_flash.shape[0]) < padded_context_len:
                                pad = padded_context_len - int(k_flash.shape[0])
                                k_flash = torch.nn.functional.pad(k_flash, (0, 0, 0, 0, 0, pad))
                                v_flash = torch.nn.functional.pad(v_flash, (0, 0, 0, 0, 0, pad))
                            workspace = getattr(self, "_pagedpq_flashinfer_workspace", None)
                            if not isinstance(workspace, torch.Tensor) or workspace.device != device:
                                workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
                                setattr(self, "_pagedpq_flashinfer_workspace", workspace)
                            wrapper = flashinfer.BlockSparseAttentionWrapper(workspace)
                            wrapper.plan(
                                indptr,
                                indices,
                                int(query_len),
                                int(padded_context_len),
                                1,
                                int(page_size),
                                int(num_heads),
                                int(num_kv_heads),
                                int(self.head_dim),
                                causal=True,
                                sm_scale=float(self.head_dim) ** -0.5,
                                q_data_type=q_flash.dtype,
                                kv_data_type=k_flash.dtype,
                            )
                            outputs = wrapper.run(q_flash, k_flash, v_flash)
                            if bool(getattr(args, "profile_native_ops", False)):
                                _sync_if_cuda(device)
                                stats[layer_id].add_native_timing(
                                    selector_seconds=selector_seconds,
                                    attention_seconds=float(time.perf_counter() - attention_t0),
                                )
                            code_bytes = 1 if int(args.subbits) <= 8 else 2
                            page_costs = [
                                float(page.codebooks.numel() * int(key_bytes) + page.codes.numel() * code_bytes)
                                for page in gqa_indexes[0].pages
                            ]
                            page_counts_cpu = row_counts_t.detach().cpu().tolist()
                            page_starts_cpu = [int(page.start) for page in gqa_indexes[0].pages]
                            for local_qpos in range(query_len):
                                query_context_len = int(query_start + local_qpos + 1)
                                prefix_end = min(max(0, int(args.static_prefix)), query_context_len)
                                indexed_end_q = max(prefix_end, query_context_len - max(0, int(args.static_suffix)))
                                sealed_end_q = prefix_end + (
                                    (max(0, indexed_end_q - prefix_end) // max(1, page_size)) * max(1, page_size)
                                )
                                valid_pages = 0
                                for page_start in page_starts_cpu:
                                    if int(page_start) >= int(prefix_end) and int(page_start) + page_size <= int(sealed_end_q):
                                        valid_pages += 1
                                selector_mb = 0.0 if int(budget) <= 0 else float(sum(page_costs[:valid_pages])) / MB
                                selected_count = int(page_counts_cpu[local_qpos] * page_size)
                                exact_kv_mb = float(selected_count * int(self.head_dim) * (key_bytes + value_bytes)) / MB
                                stats[layer_id].add_count_repeated(
                                    num_heads,
                                    selected_count,
                                    0,
                                    float(selector_mb),
                                    int(self.head_dim),
                                    key_bytes,
                                    value_bytes,
                                    tail_mb_override=0.0,
                                    exact_kv_mb_override=exact_kv_mb,
                                )
                            return outputs.reshape(query_len, -1).to(hidden_states.dtype).contiguous()
                        ranked_valid = torch.isfinite(ranked_scores)
                        token_mask = torch.zeros(
                            (query_len, context_len),
                            dtype=torch.bool,
                            device=device,
                        )
                        if ranked_t.numel() > 0:
                            safe_tokens = ranked_t.to(torch.long).clamp(min=0, max=max(0, context_len - 1))
                            token_mask.scatter_(
                                1,
                                safe_tokens.reshape(query_len, -1),
                                ranked_valid.reshape(query_len, -1),
                            )
                        query_context_lens = torch.arange(
                            query_start + 1,
                            query_start + query_len + 1,
                            dtype=torch.long,
                            device=device,
                        )
                        token_positions = torch.arange(context_len, dtype=torch.long, device=device).reshape(1, -1)
                        prefix_ends = torch.minimum(
                            torch.full_like(query_context_lens, max(0, int(args.static_prefix))),
                            query_context_lens,
                        )
                        indexed_ends = torch.maximum(
                            prefix_ends,
                            query_context_lens - max(0, int(args.static_suffix)),
                        )
                        sealed_ends = prefix_ends + (
                            torch.div(
                                torch.clamp(indexed_ends - prefix_ends, min=0),
                                max(1, page_size),
                                rounding_mode="floor",
                            )
                            * page_size
                        )
                        suffix_starts = torch.maximum(sealed_ends, prefix_ends)
                        prefix_mask = token_positions < prefix_ends.reshape(-1, 1)
                        suffix_mask = (
                            (token_positions < query_context_lens.reshape(-1, 1))
                            & (token_positions >= suffix_starts.reshape(-1, 1))
                        )
                        token_mask = token_mask | prefix_mask | suffix_mask
                        row_counts_t = token_mask.sum(dim=1, dtype=torch.int32)
                        indptr = torch.empty((query_len + 1,), dtype=torch.int32, device=device)
                        indptr[0] = 0
                        indptr[1:] = torch.cumsum(row_counts_t, dim=0)
                        indices = token_mask.to(torch.int32).nonzero(as_tuple=False)[:, 1].to(torch.int32).contiguous()
                        q_flash = query_states[0].transpose(0, 1).contiguous()
                        k_flash = keys_all.transpose(0, 1).contiguous()
                        v_flash = values_all.transpose(0, 1).contiguous()
                        workspace = getattr(self, "_pagedpq_flashinfer_workspace", None)
                        if not isinstance(workspace, torch.Tensor) or workspace.device != device:
                            workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
                            setattr(self, "_pagedpq_flashinfer_workspace", workspace)
                        wrapper = flashinfer.BlockSparseAttentionWrapper(workspace)
                        wrapper.plan(
                            indptr,
                            indices,
                            int(query_len),
                            int(context_len),
                            1,
                            1,
                            int(num_heads),
                            int(num_kv_heads),
                            int(self.head_dim),
                            causal=True,
                            sm_scale=float(self.head_dim) ** -0.5,
                            q_data_type=q_flash.dtype,
                            kv_data_type=k_flash.dtype,
                        )
                        outputs = wrapper.run(q_flash, k_flash, v_flash)
                        if bool(getattr(args, "profile_native_ops", False)):
                            _sync_if_cuda(device)
                            stats[layer_id].add_native_timing(
                                selector_seconds=selector_seconds,
                                attention_seconds=float(time.perf_counter() - attention_t0),
                            )
                        code_bytes = 1 if int(args.subbits) <= 8 else 2
                        page_costs = [
                            float(page.codebooks.numel() * int(key_bytes) + page.codes.numel() * code_bytes)
                            for page in gqa_indexes[0].pages
                        ]
                        token_counts_cpu = row_counts_t.detach().cpu().tolist()
                        page_starts_cpu = [int(page.start) for page in gqa_indexes[0].pages]
                        for local_qpos in range(query_len):
                            query_context_len = int(query_start + local_qpos + 1)
                            prefix_end = min(max(0, int(args.static_prefix)), query_context_len)
                            indexed_end_q = max(prefix_end, query_context_len - max(0, int(args.static_suffix)))
                            sealed_end_q = prefix_end + (
                                (max(0, indexed_end_q - prefix_end) // max(1, page_size)) * max(1, page_size)
                            )
                            valid_pages = 0
                            for page_start in page_starts_cpu:
                                if int(page_start) >= int(prefix_end) and int(page_start) + page_size <= int(sealed_end_q):
                                    valid_pages += 1
                            selector_mb = 0.0 if int(budget) <= 0 else float(sum(page_costs[:valid_pages])) / MB
                            selected_count = int(token_counts_cpu[local_qpos])
                            exact_kv_mb = float(selected_count * int(self.head_dim) * (key_bytes + value_bytes)) / MB
                            stats[layer_id].add_count_repeated(
                                num_heads,
                                selected_count,
                                0,
                                float(selector_mb),
                                int(self.head_dim),
                                key_bytes,
                                value_bytes,
                                tail_mb_override=0.0,
                                exact_kv_mb_override=exact_kv_mb,
                            )
                        return outputs.reshape(query_len, -1).to(hidden_states.dtype).contiguous()
                    value_codebooks = None
                    value_codes = None
                    value_page_starts = None
                    value_page_size = int(args.page_size)
                    if (selected_vpq_prefill and not selected_vpq_exact_all_prefill) or tail_vpq_prefill:
                        value_group_pages = int(args.value_pq_group_pages) if selected_vpq_prefill else 1
                        value_packs = [
                            value_vpq_pack_torch(
                                index=index,
                                values=values_all[int(kv_head)],
                                value_subvecs=int(args.value_subvecs),
                                value_subbits=int(args.value_subbits),
                                key_bytes=int(value_bytes),
                                device=device,
                                value_group_pages=value_group_pages,
                            )
                            for kv_head, index in enumerate(gqa_indexes)
                        ]
                        if any(pack is None for pack in value_packs):
                            return None
                        for index in gqa_indexes:
                            build_stats = getattr(index, "_last_value_vpq_build_stats", None)
                            if build_stats is not None:
                                build_seconds, build_read_mb, build_write_mb = build_stats
                                stats[layer_id].add_index_build(build_seconds, build_read_mb, build_write_mb)
                                setattr(index, "_last_value_vpq_build_stats", None)
                        value_codebooks = torch.stack([pack[0] for pack in value_packs if pack is not None], dim=0).contiguous()
                        value_codes = torch.stack([pack[1] for pack in value_packs if pack is not None], dim=0).contiguous()
                        value_page_starts = value_packs[0][2]
                        value_page_size = int(value_packs[0][3])
                    confidence_extra_attention_calls = 0
                    confidence_calibration_key_mb = 0.0
                    proxy_confidence_score_read_passes = 0
                    accepted_budget_counts: torch.Tensor | None = None
                    accepted_budget_cost_upper: float | None = None
                    accepted_budget_mean_by_pos_cpu: list[float] | None = None
                    if ranked_confidence_prefill and (exact_prefill or selected_vpq_prefill) and not tail_vpq_prefill:
                        if ranked_scores.numel() == 0:
                            return None
                        rank_count = int(ranked_scores.shape[2])
                        if rank_count <= 0:
                            return None
                        max_budget = max(0, min(rank_count, int(args.geometric_max_budget)))
                        if max_budget <= 0:
                            max_budget = rank_count
                        ranked_t_eff = ranked_t[:, :, :max_budget].contiguous()
                        ranked_scores_eff = ranked_scores[:, :, :max_budget].contiguous()
                        granularity = max(1, int(args.geometric_budget_granularity))
                        min_budget = _round_budget_up(
                            int(args.geometric_min_budget),
                            granularity=granularity,
                            max_budget=max_budget,
                        )
                        proxy_target = max(
                            float(args.tail_proxy_mass_min),
                            1.0 - float(args.tail_proxy_mass_max),
                        )
                        proxy_target = min(max(float(proxy_target), 0.0), 1.0 - 1.0e-7)
                        scale = float(self.head_dim) ** -0.5
                        top_scores = ranked_scores_eff.float() * scale
                        denom = torch.logsumexp(top_scores, dim=-1)
                        cum_mass = torch.exp(torch.logcumsumexp(top_scores, dim=-1) - denom.unsqueeze(-1))
                        hit = cum_mass >= proxy_target
                        has_hit = torch.any(hit, dim=-1)
                        first_hit = torch.argmax(hit.to(torch.int32), dim=-1).to(torch.long) + 1
                        accepted_budget_counts = torch.where(
                            has_hit,
                            first_hit,
                            torch.full_like(first_hit, max_budget),
                        )
                        accepted_budget_counts = torch.clamp(accepted_budget_counts, min=int(min_budget), max=int(max_budget))
                        accepted_budget_counts = (
                            torch.div(accepted_budget_counts + granularity - 1, granularity, rounding_mode="floor")
                            * granularity
                        ).clamp(max=int(max_budget))
                        accepted_budget_cost_upper = float(max_budget)
                        rank_ids = torch.arange(max_budget, dtype=torch.long, device=device).reshape(1, 1, max_budget)
                        proxy_ranked_scores = ranked_scores_eff.masked_fill(
                            rank_ids >= accepted_budget_counts.unsqueeze(-1),
                            float("-inf"),
                        ).contiguous()
                        if exact_prefill or selected_vpq_exact_all_prefill:
                            outputs = native.gqa_causal_exact_selected_attention(
                                queries,
                                keys_all.contiguous(),
                                values_all.contiguous(),
                                ranked_t_eff,
                                proxy_ranked_scores,
                                int(group_size),
                                int(query_start),
                                int(args.static_prefix),
                                int(args.static_suffix),
                                int(args.page_size),
                                float(self.head_dim) ** -0.5,
                            )
                        else:
                            exact_value_top = native_selected_value_exact_top_arg(args)
                            if selected_value_exact_top_positive(args) > 0:
                                outputs = native.gqa_causal_vpq_selected_attention_mixed_vpagesize(
                                    queries,
                                    keys_all.contiguous(),
                                    values_all.contiguous(),
                                    value_codebooks,
                                    value_codes,
                                    value_page_starts,
                                    ranked_t_eff,
                                    proxy_ranked_scores,
                                    int(group_size),
                                    int(query_start),
                                    int(args.static_prefix),
                                    int(args.static_suffix),
                                    int(args.page_size),
                                    int(value_page_size),
                                    int(exact_value_top),
                                    float(self.head_dim) ** -0.5,
                                )
                            else:
                                outputs = native.gqa_causal_vpq_selected_attention_vpagesize(
                                    queries,
                                    keys_all.contiguous(),
                                    values_all.contiguous(),
                                    value_codebooks,
                                    value_codes,
                                    value_page_starts,
                                    ranked_t_eff,
                                    proxy_ranked_scores,
                                    int(group_size),
                                    int(query_start),
                                    int(args.static_prefix),
                                    int(args.static_suffix),
                                    int(args.page_size),
                                    int(value_page_size),
                                    float(self.head_dim) ** -0.5,
                                )
                    elif (
                        (proxy_confidence_prefill or ranked_confidence_prefill)
                        and tail_vpq_prefill
                        and (dense_pq_scores is not None or ranked_confidence_prefill)
                    ):
                        if ranked_scores.numel() == 0:
                            return None
                        rank_count = int(ranked_scores.shape[2])
                        if rank_count <= 0:
                            return None
                        max_budget = max(0, min(rank_count, int(args.geometric_max_budget)))
                        if max_budget <= 0:
                            max_budget = rank_count
                        ranked_t_eff = ranked_t[:, :, :max_budget].contiguous()
                        ranked_scores_eff = ranked_scores[:, :, :max_budget].contiguous()
                        granularity = max(1, int(args.geometric_budget_granularity))
                        min_budget = _round_budget_up(
                            int(args.geometric_min_budget),
                            granularity=granularity,
                            max_budget=max_budget,
                        )
                        proxy_target = max(
                            float(args.tail_proxy_mass_min),
                            1.0 - float(args.tail_proxy_mass_max),
                        )
                        proxy_target = min(max(float(proxy_target), 0.0), 1.0 - 1.0e-7)
                        scale = float(self.head_dim) ** -0.5
                        top_scores = ranked_scores_eff.float() * scale
                        if ranked_confidence_prefill:
                            denom = torch.logsumexp(top_scores, dim=-1)
                        elif dense_pq_logsumexp is not None:
                            denom = dense_pq_logsumexp
                        else:
                            if dense_pq_scores is None:
                                raise RuntimeError("pq_proxy_mass_budget prefill requires dense PQ scores")
                            denom = torch.empty(dense_pq_scores.shape[:2], dtype=torch.float32, device=device)
                            # Avoid materializing a full [queries, heads, tokens]
                            # scaled-score temporary at long prefill lengths.
                            denom_chunk = 128
                            for denom_start in range(0, int(dense_pq_scores.shape[0]), denom_chunk):
                                denom_end = min(int(dense_pq_scores.shape[0]), denom_start + denom_chunk)
                                denom[denom_start:denom_end] = torch.logsumexp(
                                    dense_pq_scores[denom_start:denom_end].float() * scale,
                                    dim=-1,
                                )
                        cum_mass = torch.exp(torch.logcumsumexp(top_scores, dim=-1) - denom.unsqueeze(-1))
                        hit = cum_mass >= proxy_target
                        has_hit = torch.any(hit, dim=-1)
                        first_hit = torch.argmax(hit.to(torch.int32), dim=-1).to(torch.long) + 1
                        accepted_budget_counts = torch.where(
                            has_hit,
                            first_hit,
                            torch.full_like(first_hit, max_budget),
                        )
                        accepted_budget_counts = torch.clamp(accepted_budget_counts, min=int(min_budget), max=int(max_budget))
                        accepted_budget_counts = (
                            torch.div(accepted_budget_counts + granularity - 1, granularity, rounding_mode="floor")
                            * granularity
                        ).clamp(max=int(max_budget))
                        accepted_budget_cost_upper = float(max_budget)
                        rank_ids = torch.arange(max_budget, dtype=torch.long, device=device).reshape(1, 1, max_budget)
                        proxy_ranked_scores = ranked_scores_eff.masked_fill(
                            rank_ids >= accepted_budget_counts.unsqueeze(-1),
                            float("-inf"),
                        ).contiguous()
                        if dense_pq_scores is not None and str(args.selected_value_mode) == "vpq_value":
                            outputs = native.gqa_causal_vpq_selected_tail_from_scores(
                                queries,
                                keys_all.contiguous(),
                                values_all.contiguous(),
                                dense_pq_scores.contiguous(),
                                value_codebooks,
                                value_codes,
                                page_starts,
                                ranked_t_eff,
                                proxy_ranked_scores,
                                int(group_size),
                                int(query_start),
                                int(args.static_prefix),
                                int(args.static_suffix),
                                int(args.page_size),
                                native_selected_value_exact_top_arg(args),
                                float(self.head_dim) ** -0.5,
                                float(tail_blend_value),
                            )
                        elif dense_pq_scores is not None:
                            outputs = native.gqa_causal_vpq_selected_tail_from_scores(
                                queries,
                                keys_all.contiguous(),
                                values_all.contiguous(),
                                dense_pq_scores.contiguous(),
                                value_codebooks,
                                value_codes,
                                page_starts,
                                ranked_t_eff,
                                proxy_ranked_scores,
                                int(group_size),
                                int(query_start),
                                int(args.static_prefix),
                                int(args.static_suffix),
                                int(args.page_size),
                                int(ranked_t_eff.shape[-1]),
                                float(self.head_dim) ** -0.5,
                                float(tail_blend_value),
                            )
                        elif str(args.selected_value_mode) == "vpq_value":
                            outputs = native.gqa_causal_vpq_selected_tail_attention(
                                queries,
                                keys_all.contiguous(),
                                values_all.contiguous(),
                                codebooks,
                                codes,
                                value_codebooks,
                                value_codes,
                                page_starts,
                                ranked_t_eff,
                                proxy_ranked_scores,
                                int(group_size),
                                int(query_start),
                                int(args.static_prefix),
                                int(args.static_suffix),
                                int(args.page_size),
                                native_selected_value_exact_top_arg(args),
                                float(self.head_dim) ** -0.5,
                                float(tail_blend_value),
                            )
                        else:
                            outputs = native.gqa_causal_vpq_tail_attention(
                                queries,
                                keys_all_float().contiguous(),
                                values_all_float().contiguous(),
                                codebooks,
                                codes,
                                value_codebooks,
                                value_codes,
                                page_starts,
                                ranked_t_eff,
                                proxy_ranked_scores,
                                int(group_size),
                                int(query_start),
                                int(args.static_prefix),
                                int(args.static_suffix),
                                int(args.page_size),
                                float(self.head_dim) ** -0.5,
                                float(tail_blend_value),
                            )
                        proxy_confidence_score_read_passes = (
                            0
                            if (ranked_confidence_prefill or dense_pq_logsumexp is not None)
                            else 1
                        )
                    elif probe_confidence_prefill and tail_vpq_prefill:
                        if ranked_scores.numel() == 0:
                            return None
                        rank_count = int(ranked_scores.shape[2])
                        if rank_count <= 0:
                            return None
                        rank_ids = decode_rank_ids_tensor(rank_count, device, dims=3)

                        def mask_ranked_scores(keep: int) -> torch.Tensor:
                            keep_i = max(0, min(rank_count, int(keep)))
                            return ranked_scores.masked_fill(rank_ids >= keep_i, float("-inf")).contiguous()

                        def selected_tail_prefill_from_ranked(masked_scores: torch.Tensor, blend: float) -> torch.Tensor:
                            if dense_pq_scores is not None and str(args.selected_value_mode) == "vpq_value":
                                return native.gqa_causal_vpq_selected_tail_from_scores(
                                    queries,
                                    keys_all.contiguous(),
                                    values_all.contiguous(),
                                    dense_pq_scores.contiguous(),
                                    value_codebooks,
                                    value_codes,
                                    page_starts,
                                    ranked_t.contiguous(),
                                    masked_scores,
                                    int(group_size),
                                    int(query_start),
                                    int(args.static_prefix),
                                    int(args.static_suffix),
                                    int(args.page_size),
                                    native_selected_value_exact_top_arg(args),
                                    float(self.head_dim) ** -0.5,
                                    float(blend),
                                )
                            if dense_pq_scores is not None:
                                return native.gqa_causal_vpq_selected_tail_from_scores(
                                    queries,
                                    keys_all.contiguous(),
                                    values_all.contiguous(),
                                    dense_pq_scores.contiguous(),
                                    value_codebooks,
                                    value_codes,
                                    page_starts,
                                    ranked_t.contiguous(),
                                    masked_scores,
                                    int(group_size),
                                    int(query_start),
                                    int(args.static_prefix),
                                    int(args.static_suffix),
                                    int(args.page_size),
                                    int(ranked_t.shape[-1]),
                                    float(self.head_dim) ** -0.5,
                                    float(blend),
                                )
                            if str(args.selected_value_mode) == "vpq_value":
                                return native.gqa_causal_vpq_selected_tail_attention(
                                    queries,
                                    keys_all.contiguous(),
                                    values_all.contiguous(),
                                    codebooks,
                                    codes,
                                    value_codebooks,
                                    value_codes,
                                    page_starts,
                                    ranked_t.contiguous(),
                                    masked_scores,
                                    int(group_size),
                                    int(query_start),
                                    int(args.static_prefix),
                                    int(args.static_suffix),
                                    int(args.page_size),
                                    native_selected_value_exact_top_arg(args),
                                    float(self.head_dim) ** -0.5,
                                    float(blend),
                                )
                            return native.gqa_causal_vpq_tail_attention(
                                queries,
                                keys_all_float().contiguous(),
                                values_all_float().contiguous(),
                                codebooks,
                                codes,
                                value_codebooks,
                                value_codes,
                                page_starts,
                                ranked_t.contiguous(),
                                masked_scores,
                                int(group_size),
                                int(query_start),
                                int(args.static_prefix),
                                int(args.static_suffix),
                                int(args.page_size),
                                float(self.head_dim) ** -0.5,
                                float(blend),
                            )

                        max_budget = max(0, min(rank_count, int(args.geometric_max_budget)))
                        if max_budget <= 0:
                            max_budget = rank_count
                        granularity = max(1, int(args.geometric_budget_granularity))
                        growth = max(1.01, float(args.geometric_growth))
                        probe_scale = max(1.01, float(args.geometric_probe_scale))
                        k = _round_budget_up(
                            int(args.geometric_min_budget),
                            granularity=granularity,
                            max_budget=max_budget,
                        )
                        needs_proxy_gate = (
                            float(args.tail_proxy_mass_min) > 0.0
                            or float(args.tail_proxy_mass_max) < 1.0
                            or float(args.tail_pq_corr_min) > -1.0
                            or math.isfinite(float(args.tail_pq_relrmse_max))
                        )
                        exact_ranked_logits_for_conf: torch.Tensor | None = None
                        base_lse_for_conf: torch.Tensor | None = None
                        if needs_proxy_gate:
                            exact_ranked_logits_for_conf = _gpu_gqa_ranked_exact_logits(
                                queries=queries,
                                keys_all=keys_all,
                                ranked_tokens=ranked_t,
                                group_size=int(group_size),
                                scale=float(self.head_dim) ** -0.5,
                                max_rank=int(max_budget),
                            )
                            base_lse_for_conf, base_tokens_for_conf = _gpu_gqa_base_logsumexp_prefill(
                                queries=queries,
                                keys_all=keys_all,
                                group_size=int(group_size),
                                query_start=int(query_start),
                                static_prefix=int(args.static_prefix),
                                static_suffix=int(args.static_suffix),
                                page_size=int(args.page_size),
                                scale=float(self.head_dim) ** -0.5,
                            )
                            confidence_calibration_key_mb += (
                                float(query_len * num_heads * int(max_budget) * int(self.head_dim) * key_bytes)
                                + float(base_tokens_for_conf * num_heads * int(self.head_dim) * key_bytes)
                            ) / MB
                        unresolved = torch.ones((query_len, num_heads), dtype=torch.bool, device=device)
                        outputs = torch.empty((query_len, num_heads, int(self.head_dim)), dtype=torch.float32, device=device)
                        while True:
                            tail_budget = min(max_budget, int(k))
                            probe_budget = _round_budget_up(
                                max(float(tail_budget + granularity), probe_scale * float(tail_budget)),
                                granularity=granularity,
                                max_budget=max_budget,
                            )
                            probe_budget = max(tail_budget, int(probe_budget))
                            tail_ranked_scores = mask_ranked_scores(tail_budget)
                            probe_ranked_scores = mask_ranked_scores(probe_budget)
                            approx_tail = selected_tail_prefill_from_ranked(tail_ranked_scores, 1.0)
                            probe_only = selected_tail_prefill_from_ranked(probe_ranked_scores, 0.0)
                            confidence_extra_attention_calls += 2
                            rel = torch.linalg.vector_norm((approx_tail - probe_only).float(), dim=-1) / torch.clamp(
                                torch.linalg.vector_norm(probe_only.float(), dim=-1),
                                min=1.0e-20,
                            )
                            gate = rel <= float(args.tail_probe_rel_l2_max)
                            if needs_proxy_gate:
                                if exact_ranked_logits_for_conf is None:
                                    raise RuntimeError("missing exact ranked logits for geometric proxy confidence")
                                proxy_mass, proxy_tail_mass, tail_pq_corr, tail_pq_relrmse = _gpu_proxy_confidence_metrics(
                                    ranked_scores=ranked_scores,
                                    exact_ranked_logits=exact_ranked_logits_for_conf,
                                    keep_count=int(tail_budget),
                                    max_budget=int(max_budget),
                                    query_dim=int(self.head_dim),
                                    base_logsumexp=base_lse_for_conf,
                                    calibrate=str(args.tail_score_calibration) == "affine_selected",
                                )
                                gate = (
                                    gate
                                    & (proxy_mass >= float(args.tail_proxy_mass_min))
                                    & (proxy_tail_mass <= float(args.tail_proxy_mass_max))
                                    & (tail_pq_corr >= float(args.tail_pq_corr_min))
                                    & (tail_pq_relrmse <= float(args.tail_pq_relrmse_max))
                                )
                            passed = gate & unresolved
                            if bool(torch.any(passed)):
                                candidate = selected_tail_prefill_from_ranked(probe_ranked_scores, float(tail_blend_value))
                                confidence_extra_attention_calls += 1
                                outputs = torch.where(passed.unsqueeze(-1), candidate, outputs)
                            unresolved = unresolved & ~passed
                            if not bool(torch.any(unresolved)):
                                break
                            if probe_budget >= max_budget:
                                outputs = torch.where(unresolved.unsqueeze(-1), probe_only, outputs)
                                break
                            next_k = _round_budget_up(
                                max(float(probe_budget + granularity), growth * float(probe_budget)),
                                granularity=granularity,
                                max_budget=max_budget,
                            )
                            if int(next_k) <= int(probe_budget):
                                outputs = torch.where(unresolved.unsqueeze(-1), probe_only, outputs)
                                break
                            k = int(next_k)
                    elif tail_vpq_prefill:
                        if dense_pq_scores is not None and str(args.selected_value_mode) == "vpq_value":
                            outputs = native.gqa_causal_vpq_selected_tail_from_scores(
                                queries,
                                keys_all.contiguous(),
                                values_all.contiguous(),
                                dense_pq_scores.contiguous(),
                                value_codebooks,
                                value_codes,
                                page_starts,
                                ranked_t.contiguous(),
                                ranked_scores.contiguous(),
                                int(group_size),
                                int(query_start),
                                int(args.static_prefix),
                                int(args.static_suffix),
                                int(args.page_size),
                                native_selected_value_exact_top_arg(args),
                                float(self.head_dim) ** -0.5,
                                float(tail_blend_value),
                            )
                        elif dense_pq_scores is not None:
                            outputs = native.gqa_causal_vpq_tail_from_scores(
                                queries,
                                keys_all_float().contiguous(),
                                values_all_float().contiguous(),
                                dense_pq_scores.contiguous(),
                                value_codebooks,
                                value_codes,
                                page_starts,
                                ranked_t.contiguous(),
                                ranked_scores.contiguous(),
                                int(group_size),
                                int(query_start),
                                int(args.static_prefix),
                                int(args.static_suffix),
                                int(args.page_size),
                                float(self.head_dim) ** -0.5,
                                float(tail_blend_value),
                            )
                        elif str(args.selected_value_mode) == "vpq_value":
                            outputs = native.gqa_causal_vpq_selected_tail_attention(
                                queries,
                                keys_all.contiguous(),
                                values_all.contiguous(),
                                codebooks,
                                codes,
                                value_codebooks,
                                value_codes,
                                page_starts,
                                ranked_t.contiguous(),
                                ranked_scores.contiguous(),
                                int(group_size),
                                int(query_start),
                                int(args.static_prefix),
                                int(args.static_suffix),
                                int(args.page_size),
                                native_selected_value_exact_top_arg(args),
                                float(self.head_dim) ** -0.5,
                                float(tail_blend_value),
                            )
                        else:
                            outputs = native.gqa_causal_vpq_tail_attention(
                                queries,
                                keys_all_float().contiguous(),
                                values_all_float().contiguous(),
                                codebooks,
                                codes,
                                value_codebooks,
                                value_codes,
                                page_starts,
                                ranked_t.contiguous(),
                                ranked_scores.contiguous(),
                                int(group_size),
                                int(query_start),
                                int(args.static_prefix),
                                int(args.static_suffix),
                                int(args.page_size),
                                float(self.head_dim) ** -0.5,
                                float(tail_blend_value),
                            )
                    elif selected_vpq_prefill and not selected_vpq_exact_all_prefill:
                        exact_value_top = native_selected_value_exact_top_arg(args)
                        if selected_value_exact_top_positive(args) > 0:
                            outputs = native.gqa_causal_vpq_selected_attention_mixed_vpagesize(
                                queries,
                                keys_all.contiguous(),
                                values_all.contiguous(),
                                value_codebooks,
                                value_codes,
                                value_page_starts,
                                ranked_t.contiguous(),
                                ranked_scores.contiguous(),
                                int(group_size),
                                int(query_start),
                                int(args.static_prefix),
                                int(args.static_suffix),
                                int(args.page_size),
                                int(value_page_size),
                                int(exact_value_top),
                                float(self.head_dim) ** -0.5,
                            )
                        else:
                            outputs = native.gqa_causal_vpq_selected_attention_vpagesize(
                                queries,
                                keys_all.contiguous(),
                                values_all.contiguous(),
                                value_codebooks,
                                value_codes,
                                value_page_starts,
                                ranked_t.contiguous(),
                                ranked_scores.contiguous(),
                                int(group_size),
                                int(query_start),
                                int(args.static_prefix),
                                int(args.static_suffix),
                                int(args.page_size),
                                int(value_page_size),
                                float(self.head_dim) ** -0.5,
                            )
                    else:
                        outputs = native.gqa_causal_exact_selected_attention(
                            queries,
                            keys_all.contiguous(),
                            values_all.contiguous(),
                            ranked_t.contiguous(),
                            ranked_scores.contiguous(),
                            int(group_size),
                            int(query_start),
                            int(args.static_prefix),
                            int(args.static_suffix),
                            int(args.page_size),
                            float(self.head_dim) ** -0.5,
                        )
                    if bool(getattr(args, "profile_native_ops", False)):
                        _sync_if_cuda(device)
                        stats[layer_id].add_native_timing(
                            selector_seconds=selector_seconds,
                            attention_seconds=float(time.perf_counter() - attention_t0),
                        )
                except Exception:
                    if str(args.selector_backend) == "cuda_ext":
                        raise
                    return None

                code_bytes = 1 if int(args.subbits) <= 8 else 2
                page_costs = [
                    float(page.codebooks.numel() * int(key_bytes) + page.codes.numel() * code_bytes)
                    for page in gqa_indexes[0].pages
                ]
                page_starts_cpu = [int(page.start) for page in gqa_indexes[0].pages]
                value_page_starts_cpu = (
                    [int(x) for x in value_page_starts.detach().cpu().tolist()]
                    if value_page_starts is not None
                    else []
                )
                if accepted_budget_counts is not None and ranked_confidence_cost_mode == "exact":
                    # One sync per prefill call is acceptable for accounting-only runs.
                    # Avoid the previous per-query sync, which dominated benchmark runtime.
                    accepted_budget_mean_by_pos_cpu = (
                        accepted_budget_counts.float().mean(dim=1).detach().cpu().tolist()
                    )
                page_size = int(args.page_size)
                for local_qpos in range(query_len):
                    query_context_len = int(query_start + local_qpos + 1)
                    prefix_end = min(max(0, int(args.static_prefix)), query_context_len)
                    indexed_end_q = max(prefix_end, query_context_len - max(0, int(args.static_suffix)))
                    sealed_end_q = prefix_end + (
                        (max(0, indexed_end_q - prefix_end) // max(1, page_size)) * max(1, page_size)
                    )
                    base_tail_start = max(sealed_end_q, prefix_end)
                    base_count = int(prefix_end) + max(0, int(query_context_len) - int(base_tail_start))
                    valid_pages = 0
                    for page_start in page_starts_cpu:
                        if int(page_start) >= int(prefix_end) and int(page_start) + page_size <= int(sealed_end_q):
                            valid_pages += 1
                    selector_mb = (
                        float(selector_mb_per_q_override[int(local_qpos)])
                        if selector_mb_per_q_override is not None
                        else (0.0 if int(budget) <= 0 else float(sum(page_costs[:valid_pages])) / MB)
                    )
                    if prefill_selector_backend == "torch_matmul":
                        selector_mb += float(valid_pages * page_size * int(self.head_dim) * 4) / MB
                    if dense_pq_scores is not None:
                        dense_score_write_bytes = int(len(page_starts_cpu)) * page_size * 4
                        dense_score_tail_read_bytes = int(valid_pages) * page_size * 4 * 2
                        selector_mb += float(dense_score_write_bytes + dense_score_tail_read_bytes) / MB
                        if dense_score_fullscan_extra and int(budget) > 0:
                            selector_mb += float(sum(page_costs[:valid_pages])) / MB
                    ranked_count = min(max(0, int(budget)), max(0, valid_pages * page_size))
                    if accepted_budget_counts is not None:
                        if accepted_budget_mean_by_pos_cpu is not None:
                            ranked_count = float(accepted_budget_mean_by_pos_cpu[int(local_qpos)])
                        else:
                            ranked_count = float(accepted_budget_cost_upper or ranked_count)
                        ranked_count = min(max(0.0, ranked_count), float(max(0, valid_pages * page_size)))
                    selected_count = int(base_count) + float(ranked_count)
                    if tail_vpq_prefill and value_codebooks is not None and value_codes is not None:
                        actual_value_subbits = int(args.value_subbits) if int(args.value_subbits) > 0 else int(args.subbits)
                        code_bytes = 1 if int(actual_value_subbits) <= 8 else 2
                        value_subvecs = int(value_codebooks.shape[2])
                        value_subdim = int(value_codebooks.shape[-1])
                        value_centroids = int(value_codebooks.shape[3])
                        tail_count = max(0, int(valid_pages * page_size) - int(ranked_count))
                        exact_value_top = (
                            float(ranked_count)
                            if str(args.selected_value_mode) == "exact"
                            else min(max(0.0, float(ranked_count)), float(max(0, int(args.selected_value_exact_top))))
                        )
                        compressed_selected_values = max(0.0, float(ranked_count) - float(exact_value_top))
                        tail_mb = (
                            float(valid_pages * value_subvecs * value_centroids * value_subdim * value_bytes)
                            + float((tail_count + compressed_selected_values) * value_subvecs * code_bytes)
                        ) / MB
                        exact_kv_mb = (
                            float(selected_count * int(self.head_dim) * key_bytes)
                            + float((base_count + exact_value_top) * int(self.head_dim) * value_bytes)
                        ) / MB
                    elif selected_vpq_prefill and value_codebooks is not None and value_codes is not None:
                        actual_value_subbits = int(args.value_subbits) if int(args.value_subbits) > 0 else int(args.subbits)
                        code_bytes = 1 if int(actual_value_subbits) <= 8 else 2
                        value_subvecs = int(value_codebooks.shape[2])
                        value_subdim = int(value_codebooks.shape[-1])
                        value_centroids = int(value_codebooks.shape[3])
                        value_valid_pages = 0
                        for value_page_start in value_page_starts_cpu:
                            if (
                                int(value_page_start) >= int(prefix_end)
                                and int(value_page_start) + int(value_page_size) <= int(sealed_end_q)
                            ):
                                value_valid_pages += 1
                        compressed_v_mb = (
                            float(value_valid_pages * value_subvecs * value_centroids * value_subdim * value_bytes)
                            + float(ranked_count * value_subvecs * code_bytes)
                        ) / MB
                        exact_value_top = max(0, int(args.selected_value_exact_top))
                        extra_exact_value_count = min(max(0, int(ranked_count)), int(exact_value_top))
                        exact_kv_mb = (
                            float(selected_count * int(self.head_dim) * key_bytes)
                            + float(base_count * int(self.head_dim) * value_bytes)
                            + float(extra_exact_value_count * int(self.head_dim) * value_bytes)
                        ) / MB + compressed_v_mb
                        tail_count = 0
                        tail_mb = 0.0
                    else:
                        exact_kv_mb = float(selected_count * int(self.head_dim) * (key_bytes + value_bytes)) / MB
                        tail_count = 0
                        tail_mb = 0.0
                    confidence_mb = (
                        max(0, int(confidence_extra_attention_calls) - 1) * float(exact_kv_mb + tail_mb)
                        + float(confidence_calibration_key_mb) / float(max(1, query_len * num_heads))
                    )
                    if proxy_confidence_score_read_passes > 0:
                        confidence_mb += (
                            float(proxy_confidence_score_read_passes) * float(valid_pages * page_size * 4) / MB
                        )
                    stats[layer_id].add_count_repeated(
                        num_heads,
                        selected_count,
                        tail_count,
                        selector_mb,
                        int(self.head_dim),
                        key_bytes,
                        value_bytes,
                        tail_mb_override=tail_mb,
                        exact_kv_mb_override=exact_kv_mb,
                        confidence_mb_override=confidence_mb,
                    )
                return outputs.reshape(query_len, -1).to(hidden_states.dtype).contiguous()

            def approximate_decode_fast_exact(local_qpos: int = 0, query_context_len: int | None = None) -> torch.Tensor | None:
                query_context_len = int(context_len if query_context_len is None else query_context_len)
                if str(args.selector_mode) != "fullscan" or str(args.selector_backend) not in {"cuda_ext", "auto"}:
                    return None
                confidence_decode = online_confidence_rule in {
                    "geometric_probe_tail_switch",
                    "geometric_tail_stability_switch",
                    "pq_proxy_mass_budget",
                    "pq_ranked_mass_budget",
                }
                probe_confidence_decode = online_confidence_rule == "geometric_probe_tail_switch"
                tail_stability_confidence_decode = online_confidence_rule == "geometric_tail_stability_switch"
                geometric_confidence_decode = probe_confidence_decode or tail_stability_confidence_decode
                proxy_confidence_decode = online_confidence_rule == "pq_proxy_mass_budget"
                ranked_confidence_decode = online_confidence_rule == "pq_ranked_mass_budget"
                fixed_confidence_budget_decode = (
                    confidence_decode
                    and int(args.geometric_min_budget) == int(args.geometric_max_budget)
                    and int(args.geometric_min_budget) == int(args.budget)
                )
                if fixed_confidence_budget_decode:
                    confidence_decode = False
                    probe_confidence_decode = False
                    tail_stability_confidence_decode = False
                    geometric_confidence_decode = False
                    proxy_confidence_decode = False
                    ranked_confidence_decode = False
                tail_vpq_enabled = (
                    tail_blend_value > 0.0
                    and str(args.tail_mode) == "vpq_value"
                    and (str(args.selected_value_mode) == "exact" or selected_vpq_native)
                    and (
                        math.isinf(float(args.tail_probe_rel_l2_max))
                        or geometric_confidence_decode
                        or proxy_confidence_decode
                        or ranked_confidence_decode
                    )
                )
                if str(args.selected_value_mode) not in {"exact", "vpq_value"}:
                    return None
                if tail_blend_value > 0.0 and not tail_vpq_enabled:
                    return None
                if confidence_decode:
                    exact_ranked_confidence_decode = (
                        ranked_confidence_decode
                        and str(args.selected_value_mode) == "exact"
                        and tail_blend_value <= 0.0
                    )
                    exact_tail_confidence_decode = (
                        tail_vpq_enabled and str(args.selected_value_mode) == "exact"
                    )
                    selected_vpq_tail_confidence_decode = (
                        tail_vpq_enabled
                        and str(args.selected_value_mode) == "vpq_value"
                        and selected_vpq_native
                    )
                    if (
                        not exact_ranked_confidence_decode
                        and not exact_tail_confidence_decode
                        and not selected_vpq_tail_confidence_decode
                    ):
                        return None
                    if tail_stability_confidence_decode and not selected_vpq_tail_confidence_decode:
                        return None
                    if selected_vpq_tail_confidence_decode:
                        if (proxy_confidence_decode or ranked_confidence_decode) and (
                            float(args.tail_pq_corr_min) > -1.0
                            or math.isfinite(float(args.tail_pq_relrmse_max))
                        ):
                            return None
                if str(args.selected_value_mode) == "vpq_value" and str(args.selected_value_exact_rule) not in {"fixed", "selector_rank", "selected_mass"}:
                    return None
                if int(args.rerank_candidates) > 0 or budget_by_head:
                    return None
                budget = int(args.budget)
                if confidence_decode:
                    budget = max(
                        int(budget),
                        int(args.geometric_min_budget),
                        int(args.geometric_max_budget),
                    )
                pos_indexed_end = max(
                    min(max(0, int(args.static_prefix)), int(query_context_len)),
                    int(query_context_len) - max(0, int(args.static_suffix)),
                )
                pos_sealed_end = min(
                    int(query_context_len),
                    min(max(0, int(args.static_prefix)), int(query_context_len))
                    + (
                        (
                            max(
                                0,
                                pos_indexed_end
                                - min(max(0, int(args.static_prefix)), int(query_context_len)),
                            )
                            // max(1, int(args.page_size))
                        )
                        * max(1, int(args.page_size))
                    ),
                )
                base_count = decode_base_token_count(int(query_context_len), int(pos_sealed_end))
                base_t: torch.Tensor | None = None
                if (
                    tail_vpq_enabled
                    and bool(getattr(args, "native_decode_tail", False))
                    and str(getattr(args, "index_build_backend", "numpy")) == "torch_gpu"
                ):
                    try:
                        gqa_indexes = [prefix_index_for(int(kv_head), int(query_context_len)) for kv_head in range(num_kv_heads)]
                        if gqa_indexes and all(index.pages for index in gqa_indexes):
                            if device.type == "cuda" and bool(getattr(args, "debug_empty_cache_native", False)):
                                torch.cuda.empty_cache()
                            native = load_selector_paged_pq_ext()
                            codebooks, codes, page_starts = gqa_native_fullscan_pack(gqa_indexes)
                            value_codebooks, value_codes, value_page_starts, value_page_size, _actual_value_subbits = gqa_value_vpq_pack(
                                gqa_indexes,
                                value_group_pages=1,
                            )
                            queries2 = q_all[:, int(local_qpos), :].to(device).contiguous()
                            if (
                                not bool(getattr(args, "profile_native_ops", False))
                                and (
                                    not bool(getattr(args, "disable_native_decode_fused", True))
                                    or bool(getattr(args, "native_decode_scoreless_fused", False))
                                )
                                and not confidence_decode
                                and str(args.selected_value_mode) == "vpq_value"
                                and hasattr(native, "gqa_decode_fullscan_vpq_selected_tail_agg")
                            ):
                                outputs_native: torch.Tensor | None = None
                                selected_mass_cost_upper_bound = False
                                scoreless_tail_key_rescan = False
                                if (
                                    bool(getattr(args, "native_decode_scoreless_fused", False))
                                    and str(args.selected_value_exact_rule) in {"fixed", "selector_rank"}
                                    and hasattr(native, "gqa_decode_scoreless_fullscan_vpq_tail")
                                ):
                                    scoreless_tail_key_rescan = True
                                    outputs_native = native.gqa_decode_scoreless_fullscan_vpq_tail(
                                        queries2,
                                        codebooks,
                                        codes,
                                        keys_all,
                                        values_all,
                                        value_codebooks,
                                        value_codes,
                                        page_starts,
                                        int(group_size),
                                        int(budget),
                                        int(query_context_len),
                                        int(args.static_prefix),
                                        int(args.static_suffix),
                                        int(args.page_size),
                                        native_selected_value_exact_top_arg(args),
                                        float(self.head_dim) ** -0.5,
                                        float(tail_blend_value),
                                        int(getattr(args, "native_decode_scoreless_force_mode", 2)),
                                    )
                                elif str(args.selected_value_exact_rule) in {"fixed", "selector_rank"}:
                                    outputs_native = native.gqa_decode_fullscan_vpq_selected_tail_agg(
                                        queries2,
                                        codebooks,
                                        codes,
                                        keys_all,
                                        values_all,
                                        value_codebooks,
                                        value_codes,
                                        page_starts,
                                        int(group_size),
                                        int(budget),
                                        int(query_context_len),
                                        int(args.static_prefix),
                                        int(args.static_suffix),
                                        int(args.page_size),
                                        native_selected_value_exact_top_arg(args),
                                        float(self.head_dim) ** -0.5,
                                        float(tail_blend_value),
                                    )
                                elif (
                                    str(args.selected_value_exact_rule) == "selected_mass"
                                    and int(args.selected_value_max_exact_top) <= 0
                                    and hasattr(native, "gqa_decode_fullscan_vpq_selected_tail_agg_mass_min")
                                ):
                                    selected_mass_cost_upper_bound = True
                                    outputs_native = native.gqa_decode_fullscan_vpq_selected_tail_agg_mass_min(
                                        queries2,
                                        codebooks,
                                        codes,
                                        keys_all,
                                        values_all,
                                        value_codebooks,
                                        value_codes,
                                        page_starts,
                                        float(args.selected_value_exact_mass),
                                        int(args.selected_value_min_exact_top),
                                        int(group_size),
                                        int(budget),
                                        int(query_context_len),
                                        int(args.static_prefix),
                                        int(args.static_suffix),
                                        int(args.page_size),
                                        float(self.head_dim) ** -0.5,
                                        float(tail_blend_value),
                                    )
                                if outputs_native is not None:
                                    if bool(getattr(args, "disable_cost_stats", False)):
                                        return outputs_native.reshape(num_heads, int(self.head_dim))
                                    selector_mb = (
                                        0.0
                                        if int(budget) <= 0
                                        else selector_bytes_fullscan(
                                            gqa_indexes[0],
                                            key_bytes=int(key_bytes),
                                            subbits=int(args.subbits),
                                        )
                                        / MB
                                    )
                                    page_count = int(len(gqa_indexes[0].pages))
                                    page_size = int(gqa_indexes[0].pages[0].size)
                                    ranked_count = min(max(0, int(budget)), max(0, page_count * page_size))
                                    selected_count = int(base_count) + float(ranked_count)
                                    actual_value_subbits = (
                                        int(args.value_subbits) if int(args.value_subbits) > 0 else int(args.subbits)
                                    )
                                    code_bytes = 1 if int(actual_value_subbits) <= 8 else 2
                                    value_subvecs = int(value_codebooks.shape[2])
                                    value_subdim = int(value_codebooks.shape[-1])
                                    value_centroids = int(value_codebooks.shape[3])
                                    tail_count = max(0, int(page_count * page_size) - int(ranked_count))
                                    exact_value_top = min(
                                        max(0.0, float(ranked_count)),
                                        float(max(0, int(args.selected_value_exact_top))),
                                    )
                                    if selected_mass_cost_upper_bound:
                                        exact_value_top = max(0.0, float(ranked_count))
                                    compressed_selected_values = max(0.0, float(ranked_count) - float(exact_value_top))
                                    dense_score_io_mb = (
                                        float(selector_mb)
                                        if scoreless_tail_key_rescan
                                        else float(page_count * page_size * 4 * 2) / MB
                                    )
                                    tail_mb_for_cost = (
                                        float(page_count * value_subvecs * value_centroids * value_subdim * value_bytes)
                                        + float((tail_count + compressed_selected_values) * value_subvecs * code_bytes)
                                    ) / MB + dense_score_io_mb
                                    exact_kv_mb = (
                                        float(selected_count * int(self.head_dim) * key_bytes)
                                        + float((base_count + exact_value_top) * int(self.head_dim) * value_bytes)
                                    ) / MB
                                    stats[layer_id].add_count_repeated(
                                        num_heads,
                                        selected_count,
                                        tail_count,
                                        float(selector_mb),
                                        int(self.head_dim),
                                        key_bytes,
                                        value_bytes,
                                        tail_mb_override=tail_mb_for_cost,
                                        exact_kv_mb_override=exact_kv_mb,
                                        confidence_mb_override=0.0,
                                    )
                                    return outputs_native.reshape(num_heads, int(self.head_dim))
                            if bool(getattr(args, "profile_native_ops", False)):
                                _sync_if_cuda(device)
                                selector_t0 = time.perf_counter()
                            ranked_t, ranked_scores, dense_pq_scores = native.gqa_fullscan_pq_topk_scores(
                                queries2,
                                codebooks,
                                codes,
                                page_starts,
                                int(group_size),
                                int(budget),
                            )
                            selector_seconds = 0.0
                            if bool(getattr(args, "profile_native_ops", False)):
                                _sync_if_cuda(device)
                                selector_seconds = float(time.perf_counter() - selector_t0)
                                attention_t0 = time.perf_counter()
                            confidence_extra_attention_calls = 0
                            confidence_calibration_key_mb = 0.0
                            proxy_confidence_score_read_passes = 0
                            native_exact_logit_seconds = 0.0
                            native_threshold_seconds = 0.0
                            native_geometric_seconds = 0.0
                            native_output_seconds = 0.0
                            accepted_budget_counts: torch.Tensor | None = None
                            selected_mass_exact_value_counts_for_cost: torch.Tensor | None = None
                            selected_mass_cost_upper_bound = False
                            final_k_logits_reused_for_cost = False
                            exact_ranked_logits_for_conf: torch.Tensor | None = None
                            base_lse_for_conf: torch.Tensor | None = None
                            if proxy_confidence_decode or ranked_confidence_decode:
                                if ranked_scores.numel() == 0:
                                    raise RuntimeError("proxy confidence decode requires non-empty ranked scores")
                                rank_count = int(ranked_scores.shape[1])
                                if rank_count <= 0:
                                    raise RuntimeError("proxy confidence decode requires rank_count > 0")
                                max_budget = max(0, min(rank_count, int(args.geometric_max_budget)))
                                if max_budget <= 0:
                                    max_budget = rank_count
                                granularity = max(1, int(args.geometric_budget_granularity))
                                min_budget = _round_budget_up(
                                    int(args.geometric_min_budget),
                                    granularity=granularity,
                                    max_budget=max_budget,
                                )
                                proxy_target = max(
                                    float(args.tail_proxy_mass_min),
                                    1.0 - float(args.tail_proxy_mass_max),
                                )
                                proxy_target = min(max(float(proxy_target), 0.0), 1.0 - 1.0e-7)
                                scale = float(self.head_dim) ** -0.5
                                top_scores = ranked_scores[:, :max_budget].float() * scale
                                denom = (
                                    torch.logsumexp(top_scores, dim=-1)
                                    if ranked_confidence_decode
                                    else torch.logsumexp(dense_pq_scores.float() * scale, dim=-1)
                                )
                                cum_mass = torch.exp(torch.logcumsumexp(top_scores, dim=-1) - denom.unsqueeze(-1))
                                hit = cum_mass >= proxy_target
                                has_hit = torch.any(hit, dim=-1)
                                first_hit = torch.argmax(hit.to(torch.int32), dim=-1).to(torch.long) + 1
                                accepted_budget_counts = torch.where(
                                    has_hit,
                                    first_hit,
                                    torch.full_like(first_hit, max_budget),
                                )
                                accepted_budget_counts = torch.clamp(accepted_budget_counts, min=int(min_budget), max=int(max_budget))
                                accepted_budget_counts = (
                                    torch.div(accepted_budget_counts + granularity - 1, granularity, rounding_mode="floor")
                                    * granularity
                                ).clamp(max=int(max_budget))
                                rank_ids = decode_rank_ids_tensor(rank_count, device, dims=2)
                                proxy_ranked_scores = ranked_scores.masked_fill(
                                    rank_ids >= accepted_budget_counts.unsqueeze(-1),
                                    float("-inf"),
                                ).contiguous()
                                exact_value_counts_decode: torch.Tensor | None = None
                                selected_mass_in_kernel = (
                                    str(args.selected_value_mode) == "vpq_value"
                                    and
                                    str(args.selected_value_exact_rule) == "selected_mass"
                                    and int(args.selected_value_max_exact_top) <= 0
                                )
                                if (
                                    str(args.selected_value_mode) == "vpq_value"
                                    and str(args.selected_value_exact_rule) == "selected_mass"
                                    and not selected_mass_in_kernel
                                ):
                                    exact_ranked_logits_decode = _gpu_gqa_ranked_exact_logits(
                                        queries=queries2,
                                        keys_all=keys_all,
                                        ranked_tokens=ranked_t,
                                        group_size=int(group_size),
                                        scale=float(self.head_dim) ** -0.5,
                                        max_rank=int(proxy_ranked_scores.shape[-1]),
                                    )
                                    base_lse_decode, base_tokens_for_counts = _gpu_gqa_base_logsumexp_decode(
                                        queries=queries2,
                                        keys_all=keys_all,
                                        group_size=int(group_size),
                                        query_context_len=int(query_context_len),
                                        static_prefix=int(args.static_prefix),
                                        static_suffix=int(args.static_suffix),
                                        page_size=int(args.page_size),
                                        scale=float(self.head_dim) ** -0.5,
                                    )
                                    confidence_calibration_key_mb += (
                                        float(num_heads * int(proxy_ranked_scores.shape[-1]) * int(self.head_dim) * key_bytes)
                                        + float(base_tokens_for_counts * num_heads * int(self.head_dim) * key_bytes)
                                    ) / MB
                                    exact_value_counts_decode = selected_value_exact_counts_from_mass_gpu(
                                        ranked_logits=exact_ranked_logits_decode,
                                        ranked_scores=proxy_ranked_scores,
                                        base_logsumexp=base_lse_decode,
                                        exact_mass=float(args.selected_value_exact_mass),
                                        min_top=int(args.selected_value_min_exact_top),
                                        max_top=int(args.selected_value_max_exact_top),
                                    ).contiguous()
                                    selected_mass_exact_value_counts_for_cost = exact_value_counts_decode
                                if selected_mass_in_kernel:
                                    selected_mass_cost_upper_bound = True
                                    from_logits_fn = getattr(
                                        native,
                                        "gqa_decode_vpq_selected_tail_agg_from_logits_mass_min",
                                        None,
                                    )
                                    if from_logits_fn is not None:
                                        exact_ranked_logits_decode = _gpu_gqa_ranked_exact_logits(
                                            queries=queries2,
                                            keys_all=keys_all,
                                            ranked_tokens=ranked_t,
                                            group_size=int(group_size),
                                            scale=float(self.head_dim) ** -0.5,
                                            max_rank=int(proxy_ranked_scores.shape[-1]),
                                        ).contiguous()
                                        prefix_end_for_cost = min(max(0, int(args.static_prefix)), int(query_context_len))
                                        indexed_end_for_cost = max(
                                            prefix_end_for_cost,
                                            int(query_context_len) - max(0, int(args.static_suffix)),
                                        )
                                        sealed_end_for_cost = prefix_end_for_cost + (
                                            (
                                                max(0, indexed_end_for_cost - prefix_end_for_cost)
                                                // max(1, int(args.page_size))
                                            )
                                            * max(1, int(args.page_size))
                                        )
                                        base_tokens_for_counts = prefix_end_for_cost + max(
                                            0,
                                            int(query_context_len) - sealed_end_for_cost,
                                        )
                                        confidence_calibration_key_mb += (
                                            float(num_heads * int(proxy_ranked_scores.shape[-1]) * int(self.head_dim) * key_bytes)
                                            + float(base_tokens_for_counts * num_heads * int(self.head_dim) * key_bytes)
                                        ) / MB
                                        outputs_native = from_logits_fn(
                                            queries2,
                                            keys_all,
                                            values_all,
                                            dense_pq_scores.contiguous(),
                                            value_codebooks,
                                            value_codes,
                                            page_starts,
                                            ranked_t.contiguous(),
                                            proxy_ranked_scores,
                                            exact_ranked_logits_decode,
                                            float(args.selected_value_exact_mass),
                                            int(args.selected_value_min_exact_top),
                                            int(group_size),
                                            int(query_context_len),
                                            int(args.static_prefix),
                                            int(args.static_suffix),
                                            int(args.page_size),
                                            float(self.head_dim) ** -0.5,
                                            float(tail_blend_value),
                                        )
                                    else:
                                        mass_fn = (
                                            native.gqa_decode_vpq_selected_tail_agg_from_scores_mass_min
                                            if int(args.selected_value_min_exact_top) > 0
                                            else native.gqa_decode_vpq_selected_tail_agg_from_scores_mass
                                        )
                                        mass_args = (
                                            queries2,
                                            keys_all,
                                            values_all,
                                            dense_pq_scores.contiguous(),
                                            value_codebooks,
                                            value_codes,
                                            page_starts,
                                            ranked_t.contiguous(),
                                            proxy_ranked_scores,
                                            float(args.selected_value_exact_mass),
                                        )
                                        if int(args.selected_value_min_exact_top) > 0:
                                            mass_args = mass_args + (int(args.selected_value_min_exact_top),)
                                        outputs_native = mass_fn(
                                            *mass_args,
                                            int(group_size),
                                            int(query_context_len),
                                            int(args.static_prefix),
                                            int(args.static_suffix),
                                            int(args.page_size),
                                            float(self.head_dim) ** -0.5,
                                            float(tail_blend_value),
                                        )
                                elif exact_value_counts_decode is not None:
                                    outputs_native = native.gqa_decode_vpq_selected_tail_agg_from_scores_counts(
                                        queries2,
                                        keys_all,
                                        values_all,
                                        dense_pq_scores.contiguous(),
                                        value_codebooks,
                                        value_codes,
                                        page_starts,
                                        ranked_t.contiguous(),
                                        proxy_ranked_scores,
                                        exact_value_counts_decode,
                                        int(group_size),
                                        int(query_context_len),
                                        int(args.static_prefix),
                                        int(args.static_suffix),
                                        int(args.page_size),
                                        0,
                                        float(self.head_dim) ** -0.5,
                                        float(tail_blend_value),
                                    )
                                elif str(args.selected_value_mode) == "vpq_value":
                                    outputs_native = native.gqa_decode_vpq_selected_tail_agg_from_scores(
                                        queries2,
                                        keys_all,
                                        values_all,
                                        dense_pq_scores.contiguous(),
                                        value_codebooks,
                                        value_codes,
                                        page_starts,
                                        ranked_t.contiguous(),
                                        proxy_ranked_scores,
                                        int(group_size),
                                        int(query_context_len),
                                        int(args.static_prefix),
                                        int(args.static_suffix),
                                        int(args.page_size),
                                        native_selected_value_exact_top_arg(args),
                                        float(self.head_dim) ** -0.5,
                                        float(tail_blend_value),
                                    )
                                else:
                                    outputs_native = native.gqa_decode_vpq_selected_tail_agg_from_scores(
                                        queries2,
                                        keys_all,
                                        values_all,
                                        dense_pq_scores.contiguous(),
                                        value_codebooks,
                                        value_codes,
                                        page_starts,
                                        ranked_t.contiguous(),
                                        proxy_ranked_scores,
                                        int(group_size),
                                        int(query_context_len),
                                        int(args.static_prefix),
                                        int(args.static_suffix),
                                        int(args.page_size),
                                        int(proxy_ranked_scores.shape[-1]),
                                        float(self.head_dim) ** -0.5,
                                        float(tail_blend_value),
                                    )
                                proxy_confidence_score_read_passes = 0 if ranked_confidence_decode else 1
                            elif geometric_confidence_decode:
                                if ranked_scores.numel() == 0:
                                    raise RuntimeError("confidence decode requires non-empty ranked scores")
                                rank_count = int(ranked_scores.shape[1])
                                if rank_count <= 0:
                                    raise RuntimeError("confidence decode requires rank_count > 0")
                                rank_ids = decode_rank_ids_tensor(rank_count, device, dims=2)

                                def mask_decode_scores(keep: int) -> torch.Tensor:
                                    keep_i = max(0, min(rank_count, int(keep)))
                                    return ranked_scores.masked_fill(rank_ids >= keep_i, float("-inf")).contiguous()

                                max_budget = max(0, min(rank_count, int(args.geometric_max_budget)))
                                if max_budget <= 0:
                                    max_budget = rank_count
                                ranked_t_prefix = ranked_t[:, :max_budget].contiguous()
                                ranked_scores_prefix = ranked_scores[:, :max_budget].contiguous()
                                granularity = max(1, int(args.geometric_budget_granularity))
                                growth = max(1.01, float(args.geometric_growth))
                                probe_scale = max(1.01, float(args.geometric_probe_scale))
                                k = _round_budget_up(
                                    int(args.geometric_min_budget),
                                    granularity=granularity,
                                    max_budget=max_budget,
                                )
                                needs_proxy_gate = (
                                    float(args.tail_proxy_mass_min) > 0.0
                                    or float(args.tail_proxy_mass_max) < 1.0
                                    or float(args.tail_pq_corr_min) > -1.0
                                    or math.isfinite(float(args.tail_pq_relrmse_max))
                                )
                                selected_mass_in_kernel = (
                                    str(args.selected_value_mode) == "vpq_value"
                                    and str(args.selected_value_exact_rule) == "selected_mass"
                                    and int(args.selected_value_max_exact_top) <= 0
                                )
                                # The CPU frontier uses selected-mass exact-V allocation.
                                # Only the dedicated mass-min native path may bypass the
                                # repeated Python output loop for canonical runs.
                                native_proxy_gate_possible = (
                                    needs_proxy_gate
                                    and not selected_mass_in_kernel
                                    and str(args.selected_value_mode) == "vpq_value"
                                    and str(args.selected_value_exact_rule) in {"fixed", "selector_rank"}
                                    and hasattr(native, "gqa_decode_geometric_accept_counts_vpq_proxy")
                                )
                                native_selected_mass_proxy_possible = (
                                    selected_mass_in_kernel
                                    and str(args.selected_value_mode) == "vpq_value"
                                    and hasattr(native, "gqa_decode_geometric_accept_counts_vpq_mass_min_proxy")
                                )
                                native_selected_mass_threshold_proxy_possible = (
                                    selected_mass_in_kernel
                                    and str(args.selected_value_mode) == "vpq_value"
                                    and hasattr(native, "gqa_decode_geometric_accept_counts_vpq_mass_min_proxy_from_logits_thresholds")
                                )
                                needs_python_proxy_gate = (
                                    needs_proxy_gate
                                    and not native_proxy_gate_possible
                                    and not native_selected_mass_proxy_possible
                                )
                                if (
                                    needs_python_proxy_gate
                                    or (
                                        str(args.selected_value_mode) == "vpq_value"
                                        and str(args.selected_value_exact_rule) == "selected_mass"
                                        and not selected_mass_in_kernel
                                    )
                                    or (geometric_confidence_decode and selected_mass_in_kernel)
                                ):
                                    need_base_lse_for_conf = (
                                        needs_proxy_gate
                                        or native_selected_mass_threshold_proxy_possible
                                        or (
                                            str(args.selected_value_mode) == "vpq_value"
                                            and str(args.selected_value_exact_rule) == "selected_mass"
                                            and not selected_mass_in_kernel
                                        )
                                    )
                                    use_dense_exact_logit_sim = (
                                        str(exact_logit_backend) == "dense_sim"
                                        or (
                                            str(exact_logit_backend) == "auto"
                                            and device.type == "cuda"
                                            and min(int(query_context_len), int(keys_all.shape[1]))
                                            <= int(max(1.0, float(max_budget) * float(dense_sim_max_context_ratio)))
                                        )
                                    )
                                    if bool(getattr(args, "profile_native_ops", False)):
                                        _sync_if_cuda(device)
                                        native_exact_logit_t0 = time.perf_counter()
                                    if use_dense_exact_logit_sim:
                                        (
                                            exact_ranked_logits_for_conf,
                                            base_lse_for_conf,
                                            base_tokens_for_conf,
                                            dense_key_tokens_for_conf,
                                        ) = _gpu_gqa_dense_decode_ranked_logits_and_base_lse(
                                            queries=queries2,
                                            keys_all=keys_all,
                                            keys_all_t_float=dense_decode_key_t_float_cache(
                                                layer_id=int(layer_id),
                                                keys_all=keys_all,
                                                key_count=int(query_context_len),
                                            ),
                                            ranked_tokens=ranked_t_prefix,
                                            group_size=int(group_size),
                                            scale=float(self.head_dim) ** -0.5,
                                            max_rank=int(max_budget),
                                            query_context_len=int(query_context_len),
                                            static_prefix=int(args.static_prefix),
                                            static_suffix=int(args.static_suffix),
                                            page_size=int(args.page_size),
                                            need_base_lse=bool(need_base_lse_for_conf),
                                        )
                                        confidence_calibration_key_mb += (
                                            float(num_heads * int(dense_key_tokens_for_conf) * int(self.head_dim) * key_bytes)
                                        ) / MB
                                    elif (
                                        need_base_lse_for_conf
                                        and hasattr(native, "gqa_decode_ranked_exact_logits_with_base_lse")
                                    ):
                                        exact_ranked_logits_for_conf, base_lse_for_conf = (
                                            native.gqa_decode_ranked_exact_logits_with_base_lse(
                                                queries2,
                                                keys_all,
                                                ranked_t_prefix,
                                                ranked_scores_prefix,
                                                int(group_size),
                                                int(query_context_len),
                                                int(args.static_prefix),
                                                int(args.static_suffix),
                                                int(args.page_size),
                                                float(self.head_dim) ** -0.5,
                                            )
                                        )
                                        exact_ranked_logits_for_conf = exact_ranked_logits_for_conf.contiguous()
                                        base_lse_for_conf = base_lse_for_conf.contiguous()
                                        prefix_end_for_base = min(
                                            max(0, int(args.static_prefix)),
                                            int(query_context_len),
                                        )
                                        indexed_end_for_base = max(
                                            int(prefix_end_for_base),
                                            int(query_context_len) - max(0, int(args.static_suffix)),
                                        )
                                        sealed_end_for_base = int(prefix_end_for_base) + (
                                            (
                                                max(0, int(indexed_end_for_base) - int(prefix_end_for_base))
                                                // max(1, int(args.page_size))
                                            )
                                            * max(1, int(args.page_size))
                                        )
                                        base_tokens_for_conf = decode_base_token_count(
                                            int(query_context_len),
                                            int(sealed_end_for_base),
                                        )
                                        confidence_calibration_key_mb += (
                                            float(base_tokens_for_conf * num_heads * int(self.head_dim) * key_bytes)
                                        ) / MB
                                    elif hasattr(native, "gqa_decode_ranked_exact_logits"):
                                        exact_ranked_logits_for_conf = native.gqa_decode_ranked_exact_logits(
                                            queries2,
                                            keys_all,
                                            ranked_t_prefix,
                                            ranked_scores_prefix,
                                            int(group_size),
                                            int(query_context_len),
                                            int(args.static_prefix),
                                            int(args.static_suffix),
                                            int(args.page_size),
                                            float(self.head_dim) ** -0.5,
                                        ).contiguous()
                                    else:
                                        exact_ranked_logits_for_conf = _gpu_gqa_ranked_exact_logits(
                                            queries=queries2,
                                            keys_all=keys_all,
                                            ranked_tokens=ranked_t,
                                            group_size=int(group_size),
                                            scale=float(self.head_dim) ** -0.5,
                                            max_rank=int(max_budget),
                                        )
                                    confidence_calibration_key_mb += (
                                        0.0
                                        if use_dense_exact_logit_sim
                                        else float(num_heads * int(max_budget) * int(self.head_dim) * key_bytes)
                                    ) / MB
                                    if need_base_lse_for_conf and base_lse_for_conf is None and not use_dense_exact_logit_sim:
                                        base_lse_for_conf, base_tokens_for_conf = _gpu_gqa_base_logsumexp_decode(
                                            queries=queries2,
                                            keys_all=keys_all,
                                            group_size=int(group_size),
                                            query_context_len=int(query_context_len),
                                            static_prefix=int(args.static_prefix),
                                            static_suffix=int(args.static_suffix),
                                            page_size=int(args.page_size),
                                            scale=float(self.head_dim) ** -0.5,
                                        )
                                        confidence_calibration_key_mb += (
                                            float(base_tokens_for_conf * num_heads * int(self.head_dim) * key_bytes)
                                        ) / MB
                                    if bool(getattr(args, "profile_native_ops", False)):
                                        _sync_if_cuda(device)
                                        native_exact_logit_seconds += float(time.perf_counter() - native_exact_logit_t0)
                                if selected_mass_in_kernel:
                                    selected_mass_cost_upper_bound = True
                                selected_mass_output_thresholds = None
                                selected_mass_output_threshold_sels = None
                                def decode_selected_tail(masked_scores: torch.Tensor, blend: float) -> torch.Tensor:
                                    if (
                                        str(args.selected_value_mode) == "vpq_value"
                                        and str(args.selected_value_exact_rule) == "selected_mass"
                                    ):
                                        if selected_mass_in_kernel:
                                            masked_prefix = masked_scores[:, :max_budget].contiguous()
                                            if (
                                                int(args.selected_value_min_exact_top) > 0
                                                and exact_ranked_logits_for_conf is not None
                                                and selected_mass_output_thresholds is not None
                                                and selected_mass_output_threshold_sels is not None
                                                and hasattr(native, "gqa_decode_vpq_selected_tail_agg_from_logits_mass_min_thresholds")
                                            ):
                                                return native.gqa_decode_vpq_selected_tail_agg_from_logits_mass_min_thresholds(
                                                    queries2,
                                                    keys_all,
                                                    values_all,
                                                    dense_pq_scores.contiguous(),
                                                    value_codebooks,
                                                    value_codes,
                                                    page_starts,
                                                    ranked_t_prefix,
                                                    masked_prefix,
                                                    exact_ranked_logits_for_conf[:, :max_budget].contiguous(),
                                                    selected_mass_output_thresholds.contiguous(),
                                                    selected_mass_output_threshold_sels.contiguous(),
                                                    float(args.selected_value_exact_mass),
                                                    int(args.selected_value_min_exact_top),
                                                    int(group_size),
                                                    int(query_context_len),
                                                    int(args.static_prefix),
                                                    int(args.static_suffix),
                                                    int(args.page_size),
                                                    float(self.head_dim) ** -0.5,
                                                    float(blend),
                                                )
                                            if (
                                                int(args.selected_value_min_exact_top) > 0
                                                and exact_ranked_logits_for_conf is not None
                                                and hasattr(native, "gqa_decode_vpq_selected_tail_agg_from_logits_mass_min")
                                            ):
                                                return native.gqa_decode_vpq_selected_tail_agg_from_logits_mass_min(
                                                    queries2,
                                                    keys_all,
                                                    values_all,
                                                    dense_pq_scores.contiguous(),
                                                    value_codebooks,
                                                    value_codes,
                                                    page_starts,
                                                    ranked_t_prefix,
                                                    masked_prefix,
                                                    exact_ranked_logits_for_conf[:, :max_budget].contiguous(),
                                                    float(args.selected_value_exact_mass),
                                                    int(args.selected_value_min_exact_top),
                                                    int(group_size),
                                                    int(query_context_len),
                                                    int(args.static_prefix),
                                                    int(args.static_suffix),
                                                    int(args.page_size),
                                                    float(self.head_dim) ** -0.5,
                                                    float(blend),
                                                )
                                            mass_fn = (
                                                native.gqa_decode_vpq_selected_tail_agg_from_scores_mass_min
                                                if int(args.selected_value_min_exact_top) > 0
                                                else native.gqa_decode_vpq_selected_tail_agg_from_scores_mass
                                            )
                                            mass_args = (
                                                queries2,
                                                keys_all,
                                                values_all,
                                                dense_pq_scores.contiguous(),
                                                value_codebooks,
                                                value_codes,
                                                page_starts,
                                                ranked_t_prefix,
                                                masked_prefix,
                                                float(args.selected_value_exact_mass),
                                            )
                                            if int(args.selected_value_min_exact_top) > 0:
                                                mass_args = mass_args + (int(args.selected_value_min_exact_top),)
                                            return mass_fn(
                                                *mass_args,
                                                int(group_size),
                                                int(query_context_len),
                                                int(args.static_prefix),
                                                int(args.static_suffix),
                                                int(args.page_size),
                                                float(self.head_dim) ** -0.5,
                                                float(blend),
                                            )
                                        if exact_ranked_logits_for_conf is None:
                                            raise RuntimeError("missing exact ranked logits for selected_mass exact-V counts")
                                        masked_prefix = masked_scores[:, :max_budget].contiguous()
                                        exact_value_counts = selected_value_exact_counts_from_mass_gpu(
                                            ranked_logits=exact_ranked_logits_for_conf,
                                            ranked_scores=masked_prefix,
                                            base_logsumexp=base_lse_for_conf,
                                            exact_mass=float(args.selected_value_exact_mass),
                                            min_top=int(args.selected_value_min_exact_top),
                                            max_top=int(args.selected_value_max_exact_top),
                                        ).contiguous()
                                        return native.gqa_decode_vpq_selected_tail_agg_from_scores_counts(
                                            queries2,
                                            keys_all,
                                            values_all,
                                            dense_pq_scores.contiguous(),
                                            value_codebooks,
                                            value_codes,
                                            page_starts,
                                            ranked_t_prefix,
                                            masked_prefix,
                                            exact_value_counts,
                                            int(group_size),
                                            int(query_context_len),
                                            int(args.static_prefix),
                                            int(args.static_suffix),
                                            int(args.page_size),
                                            0,
                                            float(self.head_dim) ** -0.5,
                                            float(blend),
                                        )
                                    if str(args.selected_value_mode) != "vpq_value":
                                        return native.gqa_decode_vpq_selected_tail_agg_from_scores(
                                            queries2,
                                            keys_all,
                                            values_all,
                                            dense_pq_scores.contiguous(),
                                            value_codebooks,
                                            value_codes,
                                            page_starts,
                                            ranked_t.contiguous(),
                                            masked_scores,
                                            int(group_size),
                                            int(query_context_len),
                                            int(args.static_prefix),
                                            int(args.static_suffix),
                                            int(args.page_size),
                                            int(masked_scores.shape[-1]),
                                            float(self.head_dim) ** -0.5,
                                            float(blend),
                                        )
                                    return native.gqa_decode_vpq_selected_tail_agg_from_scores(
                                        queries2,
                                        keys_all,
                                        values_all,
                                        dense_pq_scores.contiguous(),
                                        value_codebooks,
                                        value_codes,
                                        page_starts,
                                        ranked_t.contiguous(),
                                        masked_scores,
                                        int(group_size),
                                        int(query_context_len),
                                        int(args.static_prefix),
                                        int(args.static_suffix),
                                        int(args.page_size),
                                        native_selected_value_exact_top_arg(args),
                                        float(self.head_dim) ** -0.5,
                                        float(blend),
                                    )
                                outputs_native = None
                                fused_geometric_output = False
                                native_vpq_geometric_exact_top: int | None = None
                                if (
                                    str(args.selected_value_mode) == "vpq_value"
                                    and (
                                        (
                                            tail_stability_confidence_decode
                                            and hasattr(native, "gqa_decode_geometric_accept_counts_vpq_tail_stability")
                                        )
                                        or (
                                            not tail_stability_confidence_decode
                                            and hasattr(native, "gqa_decode_geometric_accept_counts_vpq")
                                        )
                                    )
                                ):
                                    if str(args.selected_value_exact_rule) == "selector_rank":
                                        native_vpq_geometric_exact_top = native_selected_value_exact_top_arg(args)
                                    elif (
                                        str(args.selected_value_exact_rule) == "fixed"
                                        and selected_value_exact_top_positive(args) <= 0
                                    ):
                                        native_vpq_geometric_exact_top = 0
                                if native_selected_mass_proxy_possible:
                                    if (
                                        exact_ranked_logits_for_conf is not None
                                        and base_lse_for_conf is not None
                                        and native_selected_mass_threshold_proxy_possible
                                    ):
                                        if bool(getattr(args, "profile_native_ops", False)):
                                            _sync_if_cuda(device)
                                            native_threshold_t0 = time.perf_counter()
                                        (
                                            tail_budgets,
                                            probe_budgets,
                                            combined_threshold_budgets,
                                            approx_cols,
                                            probe_cols,
                                        ) = geometric_threshold_budget_columns(
                                            min_budget=int(k),
                                            max_budget=int(max_budget),
                                            granularity=int(granularity),
                                            growth=float(growth),
                                            probe_scale=float(probe_scale),
                                            tensor_device=device,
                                        )
                                        combined_thresholds, combined_threshold_sels = selected_mass_thresholds_from_logits_gpu(
                                            ranked_logits=exact_ranked_logits_for_conf[:, :max_budget].contiguous(),
                                            ranked_scores=ranked_scores_prefix,
                                            base_logsumexp=base_lse_for_conf,
                                            budgets=combined_threshold_budgets,
                                            exact_mass=float(args.selected_value_exact_mass),
                                            min_top=int(args.selected_value_min_exact_top),
                                        )
                                        approx_thresholds = combined_thresholds.index_select(1, approx_cols).contiguous()
                                        approx_threshold_sels = combined_threshold_sels.index_select(1, approx_cols).contiguous()
                                        probe_thresholds = combined_thresholds.index_select(1, probe_cols).contiguous()
                                        probe_threshold_sels = combined_threshold_sels.index_select(1, probe_cols).contiguous()
                                        if bool(getattr(args, "profile_native_ops", False)):
                                            _sync_if_cuda(device)
                                            native_threshold_seconds += float(time.perf_counter() - native_threshold_t0)
                                        # This fused final-output path is parity-tested, but the first
                                        # implementation is slower than the existing canonical path on
                                        # 32k decode. Keep it available for targeted optimization without
                                        # making benchmark runs pay the regression by default.
                                        output_from_logits_fn = (
                                            getattr(
                                                native,
                                                "gqa_decode_geometric_output_vpq_mass_min_proxy_from_logits_thresholds",
                                                None,
                                            )
                                            if _env_truthy("ENABLE_FUSED_GEOMETRIC_OUTPUT")
                                            and not _env_truthy("DISABLE_FUSED_GEOMETRIC_OUTPUT")
                                            else None
                                        )
                                        if bool(getattr(args, "profile_native_ops", False)):
                                            _sync_if_cuda(device)
                                            native_geometric_t0 = time.perf_counter()
                                        if output_from_logits_fn is not None:
                                            accepted_budget_counts, outputs_native = output_from_logits_fn(
                                                queries2,
                                                keys_all,
                                                values_all,
                                                dense_pq_scores.contiguous(),
                                                value_codebooks,
                                                value_codes,
                                                page_starts,
                                                ranked_t_prefix,
                                                ranked_scores_prefix,
                                                exact_ranked_logits_for_conf[:, :max_budget].contiguous(),
                                                approx_thresholds,
                                                approx_threshold_sels,
                                                probe_thresholds,
                                                probe_threshold_sels,
                                                int(group_size),
                                                int(query_context_len),
                                                int(args.static_prefix),
                                                int(args.static_suffix),
                                                int(args.page_size),
                                                int(k),
                                                int(max_budget),
                                                int(granularity),
                                                float(growth),
                                                float(probe_scale),
                                                float(args.tail_probe_rel_l2_max),
                                                float(args.selected_value_exact_mass),
                                                int(args.selected_value_min_exact_top),
                                                float(self.head_dim) ** -0.5,
                                                float(args.tail_proxy_mass_min),
                                                float(args.tail_proxy_mass_max),
                                                float(args.tail_pq_corr_min),
                                                float(args.tail_pq_relrmse_max),
                                                str(args.tail_score_calibration) == "affine_selected",
                                                bool(tail_stability_confidence_decode),
                                                float(tail_blend_value),
                                            )
                                            accepted_budget_counts = accepted_budget_counts.contiguous()
                                            outputs_native = outputs_native.contiguous()
                                            fused_geometric_output = True
                                            final_k_logits_reused_for_cost = True
                                        else:
                                            accepted_budget_counts = (
                                                native.gqa_decode_geometric_accept_counts_vpq_mass_min_proxy_from_logits_thresholds(
                                                    queries2,
                                                    keys_all,
                                                    values_all,
                                                    dense_pq_scores.contiguous(),
                                                    value_codebooks,
                                                    value_codes,
                                                    page_starts,
                                                    ranked_t_prefix,
                                                    ranked_scores_prefix,
                                                    exact_ranked_logits_for_conf[:, :max_budget].contiguous(),
                                                    approx_thresholds,
                                                    approx_threshold_sels,
                                                    probe_thresholds,
                                                    probe_threshold_sels,
                                                    int(group_size),
                                                    int(query_context_len),
                                                    int(args.static_prefix),
                                                    int(args.static_suffix),
                                                    int(args.page_size),
                                                    int(k),
                                                    int(max_budget),
                                                    int(granularity),
                                                    float(growth),
                                                    float(probe_scale),
                                                    float(args.tail_probe_rel_l2_max),
                                                    float(args.selected_value_exact_mass),
                                                    int(args.selected_value_min_exact_top),
                                                    float(self.head_dim) ** -0.5,
                                                    float(args.tail_proxy_mass_min),
                                                    float(args.tail_proxy_mass_max),
                                                    float(args.tail_pq_corr_min),
                                                    float(args.tail_pq_relrmse_max),
                                                    str(args.tail_score_calibration) == "affine_selected",
                                                    bool(tail_stability_confidence_decode),
                                                ).contiguous()
                                            )
                                            final_k_logits_reused_for_cost = True
                                        if bool(getattr(args, "profile_native_ops", False)):
                                            _sync_if_cuda(device)
                                            native_geometric_seconds += float(time.perf_counter() - native_geometric_t0)
                                        if probe_budgets and not fused_geometric_output:
                                            selected_mass_output_thresholds, selected_mass_output_threshold_sels = (
                                                select_thresholds_for_budget_counts_gpu(
                                                    thresholds=probe_thresholds,
                                                    threshold_sels=probe_threshold_sels,
                                                    budgets=probe_budgets,
                                                    counts=accepted_budget_counts,
                                                )
                                            )
                                    elif (
                                        exact_ranked_logits_for_conf is not None
                                        and hasattr(native, "gqa_decode_geometric_accept_counts_vpq_mass_min_proxy_from_logits")
                                    ):
                                        accepted_budget_counts = (
                                            native.gqa_decode_geometric_accept_counts_vpq_mass_min_proxy_from_logits(
                                                queries2,
                                                keys_all,
                                                values_all,
                                                dense_pq_scores.contiguous(),
                                                value_codebooks,
                                                value_codes,
                                                page_starts,
                                                ranked_t_prefix,
                                                ranked_scores_prefix,
                                                exact_ranked_logits_for_conf[:, :max_budget].contiguous(),
                                                int(group_size),
                                                int(query_context_len),
                                                int(args.static_prefix),
                                                int(args.static_suffix),
                                                int(args.page_size),
                                                int(k),
                                                int(max_budget),
                                                int(granularity),
                                                float(growth),
                                                float(probe_scale),
                                                float(args.tail_probe_rel_l2_max),
                                                float(args.selected_value_exact_mass),
                                                int(args.selected_value_min_exact_top),
                                                float(self.head_dim) ** -0.5,
                                                float(args.tail_proxy_mass_min),
                                                float(args.tail_proxy_mass_max),
                                                float(args.tail_pq_corr_min),
                                                float(args.tail_pq_relrmse_max),
                                                str(args.tail_score_calibration) == "affine_selected",
                                                bool(tail_stability_confidence_decode),
                                            ).contiguous()
                                        )
                                    else:
                                        accepted_budget_counts = native.gqa_decode_geometric_accept_counts_vpq_mass_min_proxy(
                                            queries2,
                                            keys_all,
                                            values_all,
                                            dense_pq_scores.contiguous(),
                                            value_codebooks,
                                            value_codes,
                                            page_starts,
                                            ranked_t.contiguous(),
                                            ranked_scores.contiguous(),
                                            int(group_size),
                                            int(query_context_len),
                                            int(args.static_prefix),
                                            int(args.static_suffix),
                                            int(args.page_size),
                                            int(k),
                                            int(max_budget),
                                            int(granularity),
                                            float(growth),
                                            float(probe_scale),
                                            float(args.tail_probe_rel_l2_max),
                                            float(args.selected_value_exact_mass),
                                            int(args.selected_value_min_exact_top),
                                            float(self.head_dim) ** -0.5,
                                            float(args.tail_proxy_mass_min),
                                            float(args.tail_proxy_mass_max),
                                            float(args.tail_pq_corr_min),
                                            float(args.tail_pq_relrmse_max),
                                            str(args.tail_score_calibration) == "affine_selected",
                                            bool(tail_stability_confidence_decode),
                                        ).contiguous()
                                    if outputs_native is None:
                                        proxy_rank_ids = decode_rank_ids_tensor(max_budget, device, dims=2)
                                        proxy_ranked_scores = ranked_scores_prefix.masked_fill(
                                            proxy_rank_ids >= accepted_budget_counts.unsqueeze(-1),
                                            float("-inf"),
                                        ).contiguous()
                                        if bool(getattr(args, "profile_native_ops", False)):
                                            _sync_if_cuda(device)
                                            native_output_t0 = time.perf_counter()
                                        outputs_native = decode_selected_tail(
                                            proxy_ranked_scores,
                                            float(tail_blend_value),
                                        )
                                        if bool(getattr(args, "profile_native_ops", False)):
                                            _sync_if_cuda(device)
                                            native_output_seconds += float(time.perf_counter() - native_output_t0)
                                    # One native confidence pass plus one final canonical output.
                                    confidence_extra_attention_calls += 2
                                elif (
                                    str(args.selected_value_mode) != "vpq_value"
                                    and hasattr(native, "gqa_decode_geometric_accept_counts")
                                ):
                                    accepted_budget_counts = native.gqa_decode_geometric_accept_counts(
                                        queries2,
                                        keys_all,
                                        values_all,
                                        dense_pq_scores.contiguous(),
                                        value_codebooks,
                                        value_codes,
                                        page_starts,
                                        ranked_t.contiguous(),
                                        ranked_scores.contiguous(),
                                        int(group_size),
                                        int(query_context_len),
                                        int(args.static_prefix),
                                        int(args.static_suffix),
                                        int(args.page_size),
                                        int(k),
                                        int(max_budget),
                                        int(granularity),
                                        float(growth),
                                        float(probe_scale),
                                        float(args.tail_probe_rel_l2_max),
                                        float(self.head_dim) ** -0.5,
                                    ).contiguous()
                                    if needs_proxy_gate:
                                        if exact_ranked_logits_for_conf is None:
                                            raise RuntimeError("missing exact ranked logits for geometric proxy confidence")
                                        proxy_mass, proxy_tail_mass, tail_pq_corr, tail_pq_relrmse = _gpu_proxy_confidence_metrics(
                                            ranked_scores=ranked_scores,
                                            exact_ranked_logits=exact_ranked_logits_for_conf,
                                            keep_count=accepted_budget_counts,
                                            max_budget=int(max_budget),
                                            query_dim=int(self.head_dim),
                                            base_logsumexp=base_lse_for_conf,
                                            calibrate=str(args.tail_score_calibration) == "affine_selected",
                                        )
                                        gate = (
                                            (proxy_mass >= float(args.tail_proxy_mass_min))
                                            & (proxy_tail_mass <= float(args.tail_proxy_mass_max))
                                            & (tail_pq_corr >= float(args.tail_pq_corr_min))
                                            & (tail_pq_relrmse <= float(args.tail_pq_relrmse_max))
                                        )
                                        accepted_budget_counts = torch.where(
                                            gate,
                                            accepted_budget_counts,
                                            torch.full_like(accepted_budget_counts, int(max_budget)),
                                        )
                                    proxy_ranked_scores = ranked_scores.masked_fill(
                                        rank_ids >= accepted_budget_counts.unsqueeze(-1),
                                        float("-inf"),
                                    ).contiguous()
                                    outputs_native = decode_selected_tail(
                                        proxy_ranked_scores,
                                        float(tail_blend_value),
                                    )
                                    # Charge this as one final attention plus one fused confidence pass.
                                    confidence_extra_attention_calls += 2
                                elif native_vpq_geometric_exact_top is not None:
                                    if native_proxy_gate_possible:
                                        accepted_budget_counts = native.gqa_decode_geometric_accept_counts_vpq_proxy(
                                            queries2,
                                            keys_all,
                                            values_all,
                                            dense_pq_scores.contiguous(),
                                            value_codebooks,
                                            value_codes,
                                            page_starts,
                                            ranked_t.contiguous(),
                                            ranked_scores.contiguous(),
                                            int(group_size),
                                            int(query_context_len),
                                            int(args.static_prefix),
                                            int(args.static_suffix),
                                            int(args.page_size),
                                            int(k),
                                            int(max_budget),
                                            int(granularity),
                                            float(growth),
                                            float(probe_scale),
                                            float(args.tail_probe_rel_l2_max),
                                            int(native_vpq_geometric_exact_top),
                                            float(self.head_dim) ** -0.5,
                                            float(args.tail_proxy_mass_min),
                                            float(args.tail_proxy_mass_max),
                                            float(args.tail_pq_corr_min),
                                            float(args.tail_pq_relrmse_max),
                                            str(args.tail_score_calibration) == "affine_selected",
                                            bool(tail_stability_confidence_decode),
                                        ).contiguous()
                                    else:
                                        vpq_accept_counts_fn = (
                                            native.gqa_decode_geometric_accept_counts_vpq_tail_stability
                                            if tail_stability_confidence_decode
                                            else native.gqa_decode_geometric_accept_counts_vpq
                                        )
                                        accepted_budget_counts = vpq_accept_counts_fn(
                                            queries2,
                                            keys_all,
                                            values_all,
                                            dense_pq_scores.contiguous(),
                                            value_codebooks,
                                            value_codes,
                                            page_starts,
                                            ranked_t.contiguous(),
                                            ranked_scores.contiguous(),
                                            int(group_size),
                                            int(query_context_len),
                                            int(args.static_prefix),
                                            int(args.static_suffix),
                                            int(args.page_size),
                                            int(k),
                                            int(max_budget),
                                            int(granularity),
                                            float(growth),
                                            float(probe_scale),
                                            float(args.tail_probe_rel_l2_max),
                                            int(native_vpq_geometric_exact_top),
                                            float(self.head_dim) ** -0.5,
                                        ).contiguous()
                                    if needs_python_proxy_gate:
                                        if exact_ranked_logits_for_conf is None:
                                            raise RuntimeError("missing exact ranked logits for geometric proxy confidence")
                                        proxy_mass, proxy_tail_mass, tail_pq_corr, tail_pq_relrmse = _gpu_proxy_confidence_metrics(
                                            ranked_scores=ranked_scores,
                                            exact_ranked_logits=exact_ranked_logits_for_conf,
                                            keep_count=accepted_budget_counts,
                                            max_budget=int(max_budget),
                                            query_dim=int(self.head_dim),
                                            base_logsumexp=base_lse_for_conf,
                                            calibrate=str(args.tail_score_calibration) == "affine_selected",
                                        )
                                        gate = (
                                            (proxy_mass >= float(args.tail_proxy_mass_min))
                                            & (proxy_tail_mass <= float(args.tail_proxy_mass_max))
                                            & (tail_pq_corr >= float(args.tail_pq_corr_min))
                                            & (tail_pq_relrmse <= float(args.tail_pq_relrmse_max))
                                        )
                                        accepted_budget_counts = torch.where(
                                            gate,
                                            accepted_budget_counts,
                                            torch.full_like(accepted_budget_counts, int(max_budget)),
                                        )
                                    proxy_ranked_scores = ranked_scores.masked_fill(
                                        rank_ids >= accepted_budget_counts.unsqueeze(-1),
                                        float("-inf"),
                                    ).contiguous()
                                    outputs_native = decode_selected_tail(
                                        proxy_ranked_scores,
                                        float(tail_blend_value),
                                    )
                                    confidence_extra_attention_calls += 2
                                if outputs_native is None:
                                    unresolved = torch.ones((num_heads,), dtype=torch.bool, device=device)
                                    outputs_native = torch.empty((num_heads, int(self.head_dim)), dtype=torch.float32, device=device)
                                    accepted_budget_counts = torch.full(
                                        (num_heads,),
                                        int(max_budget),
                                        dtype=torch.long,
                                        device=device,
                                    )
                                    while True:
                                        tail_budget = min(max_budget, int(k))
                                        probe_budget = _round_budget_up(
                                            max(float(tail_budget + granularity), probe_scale * float(tail_budget)),
                                            granularity=granularity,
                                            max_budget=max_budget,
                                        )
                                        probe_budget = max(tail_budget, int(probe_budget))
                                        tail_ranked_scores = mask_decode_scores(tail_budget)
                                        probe_ranked_scores = mask_decode_scores(probe_budget)
                                        approx_tail = decode_selected_tail(tail_ranked_scores, 1.0)
                                        probe_blend = 1.0 if tail_stability_confidence_decode else 0.0
                                        probe_only = decode_selected_tail(probe_ranked_scores, probe_blend)
                                        confidence_extra_attention_calls += 2
                                        rel = torch.linalg.vector_norm((approx_tail - probe_only).float(), dim=-1) / torch.clamp(
                                            torch.linalg.vector_norm(probe_only.float(), dim=-1),
                                            min=1.0e-20,
                                        )
                                        gate = rel <= float(args.tail_probe_rel_l2_max)
                                        if needs_proxy_gate:
                                            if exact_ranked_logits_for_conf is None:
                                                raise RuntimeError("missing exact ranked logits for geometric proxy confidence")
                                            proxy_mass, proxy_tail_mass, tail_pq_corr, tail_pq_relrmse = _gpu_proxy_confidence_metrics(
                                                ranked_scores=ranked_scores,
                                                exact_ranked_logits=exact_ranked_logits_for_conf,
                                                keep_count=int(tail_budget),
                                                max_budget=int(max_budget),
                                                query_dim=int(self.head_dim),
                                                base_logsumexp=base_lse_for_conf,
                                                calibrate=str(args.tail_score_calibration) == "affine_selected",
                                            )
                                            gate = (
                                                gate
                                                & (proxy_mass >= float(args.tail_proxy_mass_min))
                                                & (proxy_tail_mass <= float(args.tail_proxy_mass_max))
                                                & (tail_pq_corr >= float(args.tail_pq_corr_min))
                                                & (tail_pq_relrmse <= float(args.tail_pq_relrmse_max))
                                            )
                                        passed = gate & unresolved
                                        if bool(torch.any(passed)):
                                            candidate = decode_selected_tail(probe_ranked_scores, float(tail_blend_value))
                                            confidence_extra_attention_calls += 1
                                            outputs_native = torch.where(passed.unsqueeze(-1), candidate, outputs_native)
                                            accepted_budget_counts = torch.where(
                                                passed,
                                                torch.full_like(accepted_budget_counts, int(probe_budget)),
                                                accepted_budget_counts,
                                            )
                                        unresolved = unresolved & ~passed
                                        if not bool(torch.any(unresolved)):
                                            break
                                        if probe_budget >= max_budget:
                                            outputs_native = torch.where(
                                                unresolved.unsqueeze(-1),
                                                decode_selected_tail(probe_ranked_scores, float(tail_blend_value)),
                                                outputs_native,
                                            )
                                            break
                                        next_k = _round_budget_up(
                                            max(float(probe_budget + granularity), growth * float(probe_budget)),
                                            granularity=granularity,
                                            max_budget=max_budget,
                                        )
                                        if int(next_k) <= int(probe_budget):
                                            outputs_native = torch.where(unresolved.unsqueeze(-1), probe_only, outputs_native)
                                            accepted_budget_counts = torch.where(
                                                unresolved,
                                                torch.full_like(accepted_budget_counts, int(probe_budget)),
                                                accepted_budget_counts,
                                            )
                                            break
                                        k = int(next_k)
                            elif str(args.selected_value_mode) == "vpq_value":
                                if str(args.selected_value_exact_rule) == "selected_mass":
                                    selected_mass_in_kernel = (
                                        int(args.selected_value_max_exact_top) <= 0
                                    )
                                    if selected_mass_in_kernel:
                                        selected_mass_cost_upper_bound = True
                                        mass_fn = (
                                            native.gqa_decode_vpq_selected_tail_agg_from_scores_mass_min
                                            if int(args.selected_value_min_exact_top) > 0
                                            else native.gqa_decode_vpq_selected_tail_agg_from_scores_mass
                                        )
                                        mass_args = (
                                            queries2,
                                            keys_all,
                                            values_all,
                                            dense_pq_scores.contiguous(),
                                            value_codebooks,
                                            value_codes,
                                            page_starts,
                                            ranked_t.contiguous(),
                                            ranked_scores.contiguous(),
                                            float(args.selected_value_exact_mass),
                                        )
                                        if int(args.selected_value_min_exact_top) > 0:
                                            mass_args = mass_args + (int(args.selected_value_min_exact_top),)
                                        outputs_native = mass_fn(
                                            *mass_args,
                                            int(group_size),
                                            int(query_context_len),
                                            int(args.static_prefix),
                                            int(args.static_suffix),
                                            int(args.page_size),
                                            float(self.head_dim) ** -0.5,
                                            float(tail_blend_value),
                                        )
                                    else:
                                        exact_ranked_logits_decode = _gpu_gqa_ranked_exact_logits(
                                            queries=queries2,
                                            keys_all=keys_all,
                                            ranked_tokens=ranked_t,
                                            group_size=int(group_size),
                                            scale=float(self.head_dim) ** -0.5,
                                            max_rank=int(ranked_scores.shape[-1]),
                                        )
                                        base_lse_decode, base_tokens_for_counts = _gpu_gqa_base_logsumexp_decode(
                                            queries=queries2,
                                            keys_all=keys_all,
                                            group_size=int(group_size),
                                            query_context_len=int(query_context_len),
                                            static_prefix=int(args.static_prefix),
                                            static_suffix=int(args.static_suffix),
                                            page_size=int(args.page_size),
                                            scale=float(self.head_dim) ** -0.5,
                                        )
                                        confidence_calibration_key_mb += (
                                            float(num_heads * int(ranked_scores.shape[-1]) * int(self.head_dim) * key_bytes)
                                            + float(base_tokens_for_counts * num_heads * int(self.head_dim) * key_bytes)
                                        ) / MB
                                        exact_value_counts_decode = selected_value_exact_counts_from_mass_gpu(
                                            ranked_logits=exact_ranked_logits_decode,
                                            ranked_scores=ranked_scores,
                                            base_logsumexp=base_lse_decode,
                                            exact_mass=float(args.selected_value_exact_mass),
                                            min_top=int(args.selected_value_min_exact_top),
                                            max_top=int(args.selected_value_max_exact_top),
                                        ).contiguous()
                                        selected_mass_exact_value_counts_for_cost = exact_value_counts_decode
                                        outputs_native = native.gqa_decode_vpq_selected_tail_agg_from_scores_counts(
                                            queries2,
                                            keys_all,
                                            values_all,
                                            dense_pq_scores.contiguous(),
                                            value_codebooks,
                                            value_codes,
                                            page_starts,
                                            ranked_t.contiguous(),
                                            ranked_scores.contiguous(),
                                            exact_value_counts_decode,
                                            int(group_size),
                                            int(query_context_len),
                                            int(args.static_prefix),
                                            int(args.static_suffix),
                                            int(args.page_size),
                                            0,
                                            float(self.head_dim) ** -0.5,
                                            float(tail_blend_value),
                                        )
                                else:
                                    outputs_native = native.gqa_decode_vpq_selected_tail_agg_from_scores(
                                        queries2,
                                        keys_all,
                                        values_all,
                                        dense_pq_scores.contiguous(),
                                        value_codebooks,
                                        value_codes,
                                        page_starts,
                                        ranked_t.contiguous(),
                                        ranked_scores.contiguous(),
                                        int(group_size),
                                        int(query_context_len),
                                        int(args.static_prefix),
                                        int(args.static_suffix),
                                        int(args.page_size),
                                        native_selected_value_exact_top_arg(args),
                                        float(self.head_dim) ** -0.5,
                                        float(tail_blend_value),
                                    )
                            else:
                                outputs_native = native.gqa_decode_vpq_tail_from_scores(
                                    queries2,
                                    keys_all_float().contiguous(),
                                    values_all_float().contiguous(),
                                    dense_pq_scores.contiguous(),
                                    value_codebooks,
                                    value_codes,
                                    page_starts,
                                    ranked_t.contiguous(),
                                    ranked_scores.contiguous(),
                                    int(group_size),
                                    int(query_context_len),
                                    int(args.static_prefix),
                                    int(args.static_suffix),
                                    int(args.page_size),
                                    float(self.head_dim) ** -0.5,
                                    float(tail_blend_value),
                                )
                            if bool(getattr(args, "profile_native_ops", False)):
                                _sync_if_cuda(device)
                                stats[layer_id].add_native_timing(
                                    selector_seconds=selector_seconds,
                                    attention_seconds=float(time.perf_counter() - attention_t0),
                                )
                                stats[layer_id].add_native_detail_timing(
                                    exact_logit_seconds=native_exact_logit_seconds,
                                    threshold_seconds=native_threshold_seconds,
                                    geometric_seconds=native_geometric_seconds,
                                    output_seconds=native_output_seconds,
                                )

                            if bool(getattr(args, "disable_cost_stats", False)):
                                return outputs_native.reshape(num_heads, int(self.head_dim))

                            if (
                                selected_mass_cost_upper_bound
                                and selected_mass_exact_value_counts_for_cost is None
                                and exact_ranked_logits_for_conf is not None
                            ):
                                cost_rank = int(exact_ranked_logits_for_conf.shape[-1])
                                cost_ranked_scores = ranked_scores[:, :cost_rank]
                                if accepted_budget_counts is not None:
                                    cost_rank_ids = decode_rank_ids_tensor(cost_rank, device, dims=2)
                                    cost_ranked_scores = cost_ranked_scores.masked_fill(
                                        cost_rank_ids >= accepted_budget_counts.unsqueeze(-1),
                                        float("-inf"),
                                    )
                                selected_mass_exact_value_counts_for_cost = selected_value_exact_counts_from_mass_gpu(
                                    ranked_logits=exact_ranked_logits_for_conf[:, :cost_rank],
                                    ranked_scores=cost_ranked_scores,
                                    base_logsumexp=base_lse_for_conf,
                                    exact_mass=float(args.selected_value_exact_mass),
                                    min_top=int(args.selected_value_min_exact_top),
                                    max_top=int(args.selected_value_max_exact_top),
                                ).contiguous()
                                selected_mass_cost_upper_bound = False

                            selector_mb = (
                                0.0
                                if int(budget) <= 0
                                else selector_bytes_fullscan(
                                    gqa_indexes[0],
                                    key_bytes=int(key_bytes),
                                    subbits=int(args.subbits),
                                )
                                / MB
                            )
                            page_count = int(len(gqa_indexes[0].pages))
                            page_size = int(gqa_indexes[0].pages[0].size)
                            ranked_count = min(max(0, int(budget)), max(0, page_count * page_size))
                            if accepted_budget_counts is not None:
                                if ranked_confidence_cost_mode == "exact":
                                    ranked_count = float(accepted_budget_counts.float().mean().detach().cpu().item())
                                else:
                                    ranked_count = float(max_budget)
                                ranked_count = min(max(0.0, ranked_count), float(max(0, page_count * page_size)))
                            selected_count = int(base_count) + float(ranked_count)
                            actual_value_subbits = int(args.value_subbits) if int(args.value_subbits) > 0 else int(args.subbits)
                            code_bytes = 1 if int(actual_value_subbits) <= 8 else 2
                            value_subvecs = int(value_codebooks.shape[2])
                            value_subdim = int(value_codebooks.shape[-1])
                            value_centroids = int(value_codebooks.shape[3])
                            tail_count = max(0, int(page_count * page_size) - int(ranked_count))
                            exact_value_top = (
                                float(ranked_count)
                                if str(args.selected_value_mode) == "exact"
                                else min(max(0.0, float(ranked_count)), float(max(0, int(args.selected_value_exact_top))))
                            )
                            if selected_mass_exact_value_counts_for_cost is not None:
                                exact_value_top = min(
                                    max(0.0, float(ranked_count)),
                                    float(selected_mass_exact_value_counts_for_cost.float().mean().detach().cpu().item()),
                                )
                            elif selected_mass_cost_upper_bound:
                                # In-kernel selected-mass avoids a separate count pass. Charge all ranked selected V
                                # as exact to keep accounting conservative until the kernel exports counts.
                                exact_value_top = max(0.0, float(ranked_count))
                            compressed_selected_values = max(0.0, float(ranked_count) - float(exact_value_top))
                            dense_score_io_mb = float(page_count * page_size * 4 * 2) / MB
                            tail_mb_for_cost = (
                                float(page_count * value_subvecs * value_centroids * value_subdim * value_bytes)
                                + float((tail_count + compressed_selected_values) * value_subvecs * code_bytes)
                            ) / MB + dense_score_io_mb
                            physical_gpu_exact_key_mb = (
                                0.0
                                if final_k_logits_reused_for_cost
                                else float(selected_count * int(self.head_dim) * key_bytes) / MB
                            )
                            exact_value_mb = float((base_count + exact_value_top) * int(self.head_dim) * value_bytes) / MB
                            physical_gpu_exact_kv_mb = physical_gpu_exact_key_mb + exact_value_mb
                            physical_gpu_confidence_mb = (
                                max(0, int(confidence_extra_attention_calls) - 1)
                                * float(physical_gpu_exact_kv_mb + tail_mb_for_cost)
                                + float(confidence_calibration_key_mb) / float(max(1, num_heads))
                            )
                            if proxy_confidence_score_read_passes > 0:
                                physical_gpu_confidence_mb += (
                                    float(proxy_confidence_score_read_passes) * float(page_count * page_size * 4) / MB
                                )
                            exact_key_mb = physical_gpu_exact_key_mb
                            exact_kv_mb = physical_gpu_exact_kv_mb
                            confidence_mb = physical_gpu_confidence_mb
                            if final_k_logits_reused_for_cost:
                                # The GPU host may precompute exact logits up to max_budget to emulate
                                # the adaptive rule efficiently. The logical frontier/custom-hardware
                                # cost is incremental: read each accepted K once, cache its logit, and
                                # reuse it for the final attention. Therefore selected/probed K bytes
                                # belong to exact_KV, not confidence overhead.
                                exact_key_mb = float(selected_count * int(self.head_dim) * key_bytes) / MB
                                exact_kv_mb = exact_key_mb + exact_value_mb
                                confidence_mb = 0.0
                                if proxy_confidence_score_read_passes > 0:
                                    confidence_mb += (
                                        float(proxy_confidence_score_read_passes) * float(page_count * page_size * 4) / MB
                                    )
                            stats[layer_id].add_count_repeated(
                                num_heads,
                                selected_count,
                                tail_count,
                                float(selector_mb),
                                int(self.head_dim),
                                key_bytes,
                                value_bytes,
                                tail_mb_override=tail_mb_for_cost,
                                exact_kv_mb_override=exact_kv_mb,
                                confidence_mb_override=confidence_mb,
                                physical_gpu_exact_kv_mb_override=physical_gpu_exact_kv_mb,
                                physical_gpu_confidence_mb_override=physical_gpu_confidence_mb,
                            )
                            return outputs_native.reshape(num_heads, int(self.head_dim))
                    except Exception:
                        if str(args.selector_backend) == "cuda_ext":
                            raise
                selected_vpq_native_decode = (
                    str(args.selected_value_mode) == "vpq_value"
                    and not tail_vpq_enabled
                    and str(args.selected_value_exact_rule) == "fixed"
                    and int(args.selected_value_min_exact_top) <= 0
                    and int(args.selected_value_max_exact_top) <= 0
                    and int(getattr(args, "selected_value_exact_all_context_max", 0)) <= 0
                    and float(getattr(args, "selected_value_exact_all_fraction_min", 0.0)) <= 0.0
                    and str(getattr(args, "index_build_backend", "numpy")) == "torch_gpu"
                    and str(args.selector_backend) in {"cuda_ext", "auto"}
                    and str(args.selector_mode) == "fullscan"
                )
                if selected_vpq_native_decode:
                    try:
                        gqa_indexes = [prefix_index_for(int(kv_head), int(query_context_len)) for kv_head in range(num_kv_heads)]
                        if gqa_indexes and all(index.pages for index in gqa_indexes):
                            native = load_selector_paged_pq_ext()
                            codebooks, codes, page_starts = gqa_native_fullscan_pack(gqa_indexes)
                            value_codebooks, value_codes, value_page_starts, value_page_size, _actual_value_subbits = gqa_value_vpq_pack(
                                gqa_indexes,
                                value_group_pages=int(args.value_pq_group_pages),
                            )
                            queries2 = q_all[:, int(local_qpos), :].to(device).contiguous()
                            if bool(getattr(args, "profile_native_ops", False)):
                                _sync_if_cuda(device)
                                selector_t0 = time.perf_counter()
                            ranked_t, ranked_scores = native.gqa_fullscan_pq_topk(
                                queries2,
                                codebooks,
                                codes,
                                page_starts,
                                int(group_size),
                                int(budget),
                            )
                            selector_seconds = 0.0
                            if bool(getattr(args, "profile_native_ops", False)):
                                _sync_if_cuda(device)
                                selector_seconds = float(time.perf_counter() - selector_t0)
                                attention_t0 = time.perf_counter()
                            exact_value_top = native_selected_value_exact_top_arg(args)
                            if selected_value_exact_top_positive(args) > 0:
                                outputs_native = native.gqa_causal_vpq_selected_attention_mixed_vpagesize(
                                    queries2.reshape(1, num_heads, int(self.head_dim)).contiguous(),
                                    keys_all.contiguous(),
                                    values_all.contiguous(),
                                    value_codebooks,
                                    value_codes,
                                    value_page_starts,
                                    ranked_t.reshape(1, num_heads, -1).contiguous(),
                                    ranked_scores.reshape(1, num_heads, -1).contiguous(),
                                    int(group_size),
                                    int(query_context_len) - 1,
                                    int(args.static_prefix),
                                    int(args.static_suffix),
                                    int(args.page_size),
                                    int(value_page_size),
                                    int(exact_value_top),
                                    float(self.head_dim) ** -0.5,
                                )
                            else:
                                outputs_native = native.gqa_causal_vpq_selected_attention_vpagesize(
                                    queries2.reshape(1, num_heads, int(self.head_dim)).contiguous(),
                                    keys_all.contiguous(),
                                    values_all.contiguous(),
                                    value_codebooks,
                                    value_codes,
                                    value_page_starts,
                                    ranked_t.reshape(1, num_heads, -1).contiguous(),
                                    ranked_scores.reshape(1, num_heads, -1).contiguous(),
                                    int(group_size),
                                    int(query_context_len) - 1,
                                    int(args.static_prefix),
                                    int(args.static_suffix),
                                    int(args.page_size),
                                    int(value_page_size),
                                    float(self.head_dim) ** -0.5,
                                )
                            if bool(getattr(args, "profile_native_ops", False)):
                                _sync_if_cuda(device)
                                stats[layer_id].add_native_timing(
                                    selector_seconds=selector_seconds,
                                    attention_seconds=float(time.perf_counter() - attention_t0),
                                )
                            selector_mb = (
                                0.0
                                if int(budget) <= 0
                                else selector_bytes_fullscan(
                                    gqa_indexes[0],
                                    key_bytes=int(key_bytes),
                                    subbits=int(args.subbits),
                                )
                                / MB
                            )
                            page_count = int(len(gqa_indexes[0].pages))
                            page_size = int(gqa_indexes[0].pages[0].size)
                            value_page_count = int(value_codebooks.shape[1])
                            ranked_count = min(max(0, int(budget)), max(0, page_count * page_size))
                            selected_count = int(base_count) + int(ranked_count)
                            actual_value_subbits = int(args.value_subbits) if int(args.value_subbits) > 0 else int(args.subbits)
                            code_bytes = 1 if int(actual_value_subbits) <= 8 else 2
                            value_subvecs = int(value_codebooks.shape[2])
                            value_subdim = int(value_codebooks.shape[-1])
                            value_centroids = int(value_codebooks.shape[3])
                            compressed_v_mb = (
                                float(value_page_count * value_subvecs * value_centroids * value_subdim * value_bytes)
                                + float(ranked_count * value_subvecs * code_bytes)
                            ) / MB
                            extra_exact_value_count = min(max(0, int(ranked_count)), max(0, int(args.selected_value_exact_top)))
                            exact_kv_mb = (
                                float(selected_count * int(self.head_dim) * key_bytes)
                                + float(base_count * int(self.head_dim) * value_bytes)
                                + float(extra_exact_value_count * int(self.head_dim) * value_bytes)
                            ) / MB + compressed_v_mb
                            stats[layer_id].add_count_repeated(
                                num_heads,
                                selected_count,
                                0,
                                float(selector_mb),
                                int(self.head_dim),
                                key_bytes,
                                value_bytes,
                                exact_kv_mb_override=exact_kv_mb,
                            )
                            return outputs_native.reshape(num_heads, int(self.head_dim))
                    except Exception:
                        if str(args.selector_backend) == "cuda_ext":
                            raise
                outputs = torch.empty((num_heads, int(self.head_dim)), dtype=torch.float32, device=device)
                gqa_exact_enabled = str(args.selected_value_mode) == "exact" and not tail_vpq_enabled
                gqa_selector_enabled = (
                    not tail_vpq_enabled
                    and str(args.selector_mode) == "fullscan"
                    and str(args.selector_backend) in {"cuda_ext", "auto"}
                )
                gqa_selected_rows: list[torch.Tensor] = []
                gqa_ranked_t: torch.Tensor | None = None
                gqa_ranked_scores: torch.Tensor | None = None
                gqa_selector_mb = 0.0
                if gqa_selector_enabled:
                    try:
                        gqa_indexes = [prefix_index_for(int(kv_head), int(query_context_len)) for kv_head in range(num_kv_heads)]
                        if gqa_indexes and all(index.pages for index in gqa_indexes):
                            native = load_selector_paged_pq_ext()
                            codebooks, codes, page_starts = gqa_native_fullscan_pack(gqa_indexes)
                            if bool(getattr(args, "profile_native_ops", False)):
                                _sync_if_cuda(device)
                                selector_t0 = time.perf_counter()
                            gqa_ranked_t, gqa_ranked_scores = native.gqa_fullscan_pq_topk(
                                q_all[:, int(local_qpos), :].to(device).contiguous(),
                                codebooks,
                                codes,
                                page_starts,
                                int(group_size),
                                int(budget),
                            )
                            if bool(getattr(args, "profile_native_ops", False)):
                                _sync_if_cuda(device)
                                selector_seconds = float(time.perf_counter() - selector_t0)
                                stats[layer_id].add_native_timing(selector_seconds=selector_seconds)
                            gqa_selector_mb = (
                                0.0
                                if int(budget) <= 0
                                else selector_bytes_fullscan(
                                    gqa_indexes[0],
                                    key_bytes=int(key_bytes),
                                    subbits=int(args.subbits),
                                )
                                / MB
                            )
                        else:
                            gqa_ranked_t = torch.empty((num_heads, 0), dtype=torch.long, device=device)
                            gqa_ranked_scores = torch.empty((num_heads, 0), dtype=torch.float32, device=device)
                            gqa_selector_mb = 0.0
                    except Exception:
                        if str(args.selector_backend) == "cuda_ext":
                            raise
                        gqa_ranked_t = None
                if gqa_exact_enabled and gqa_ranked_t is not None:
                    if ranked_confidence_decode:
                        if gqa_ranked_scores is None:
                            return None
                        rank_count = int(gqa_ranked_scores.shape[1])
                        if rank_count <= 0:
                            accepted_budget_counts = torch.zeros((num_heads,), dtype=torch.long, device=device)
                            proxy_ranked_scores = gqa_ranked_scores.reshape(1, num_heads, 0).contiguous()
                        else:
                            max_budget = max(0, min(rank_count, int(args.geometric_max_budget)))
                            if max_budget <= 0:
                                max_budget = rank_count
                            granularity = max(1, int(args.geometric_budget_granularity))
                            min_budget = _round_budget_up(
                                int(args.geometric_min_budget),
                                granularity=granularity,
                                max_budget=max_budget,
                            )
                            proxy_target = max(
                                float(args.tail_proxy_mass_min),
                                1.0 - float(args.tail_proxy_mass_max),
                            )
                            proxy_target = min(max(float(proxy_target), 0.0), 1.0 - 1.0e-7)
                            scale = float(self.head_dim) ** -0.5
                            top_scores = gqa_ranked_scores[:, :max_budget].float() * scale
                            denom = torch.logsumexp(top_scores, dim=-1)
                            cum_mass = torch.exp(torch.logcumsumexp(top_scores, dim=-1) - denom.unsqueeze(-1))
                            hit = cum_mass >= proxy_target
                            has_hit = torch.any(hit, dim=-1)
                            first_hit = torch.argmax(hit.to(torch.int32), dim=-1).to(torch.long) + 1
                            accepted_budget_counts = torch.where(
                                has_hit,
                                first_hit,
                                torch.full_like(first_hit, max_budget),
                            )
                            accepted_budget_counts = torch.clamp(
                                accepted_budget_counts,
                                min=int(min_budget),
                                max=int(max_budget),
                            )
                            accepted_budget_counts = (
                                torch.div(accepted_budget_counts + granularity - 1, granularity, rounding_mode="floor")
                                * granularity
                            ).clamp(max=int(max_budget))
                            rank_ids = decode_rank_ids_tensor(rank_count, device, dims=2)
                            masked_scores = gqa_ranked_scores.masked_fill(
                                rank_ids >= accepted_budget_counts.unsqueeze(-1),
                                float("-inf"),
                            )
                            proxy_ranked_scores = masked_scores.reshape(1, num_heads, rank_count).contiguous()
                        if ranked_confidence_cost_mode == "exact":
                            accepted_budget_mean = float(accepted_budget_counts.float().mean().detach().cpu().item())
                        else:
                            accepted_budget_mean = float(max_budget)
                        selected_count = int(base_count) + accepted_budget_mean
                        exact_kv_mb = float(selected_count * int(self.head_dim) * (key_bytes + value_bytes)) / MB
                        stats[layer_id].add_count_repeated(
                            num_heads,
                            selected_count,
                            0,
                            float(gqa_selector_mb),
                            int(self.head_dim),
                            key_bytes,
                            value_bytes,
                            exact_kv_mb_override=exact_kv_mb,
                        )
                        try:
                            native = load_selector_paged_pq_ext()
                            if bool(getattr(args, "profile_native_ops", False)):
                                _sync_if_cuda(device)
                                attention_t0 = time.perf_counter()
                            outputs_native = native.gqa_causal_exact_selected_attention(
                                q_all[:, int(local_qpos), :].to(device).reshape(1, num_heads, int(self.head_dim)).contiguous(),
                                keys_all.contiguous(),
                                values_all.contiguous(),
                                gqa_ranked_t.reshape(1, num_heads, -1).contiguous(),
                                proxy_ranked_scores,
                                int(group_size),
                                int(query_context_len) - 1,
                                int(args.static_prefix),
                                int(args.static_suffix),
                                int(args.page_size),
                                float(self.head_dim) ** -0.5,
                            )
                            if bool(getattr(args, "profile_native_ops", False)):
                                _sync_if_cuda(device)
                                stats[layer_id].add_native_timing(
                                    attention_seconds=float(time.perf_counter() - attention_t0),
                                )
                            return outputs_native.reshape(num_heads, int(self.head_dim))
                        except Exception:
                            if str(args.selector_backend) == "cuda_ext":
                                raise
                    if base_t is None:
                        base_t = decode_base_tokens_tensor(
                            int(query_context_len),
                            int(pos_sealed_end),
                            int(pos_indexed_end),
                        )
                    if base_t.numel():
                        selected_all = torch.cat(
                            [base_t.reshape(1, -1).expand(num_heads, -1), gqa_ranked_t],
                            dim=1,
                        )
                    else:
                        selected_all = gqa_ranked_t
                    selected_count = int(selected_all.shape[1]) if selected_all.dim() == 2 else 0
                    exact_kv_mb = float(selected_count * int(self.head_dim) * (key_bytes + value_bytes)) / MB
                    stats[layer_id].add_count_repeated(
                        num_heads,
                        selected_count,
                        0,
                        float(gqa_selector_mb),
                        int(self.head_dim),
                        key_bytes,
                        value_bytes,
                        exact_kv_mb_override=exact_kv_mb,
                    )
                    if selected_all.numel() == 0:
                        return torch.zeros((num_heads, int(self.head_dim)), dtype=torch.float32, device=device)
                    try:
                        native = load_selector_paged_pq_ext()
                        if bool(getattr(args, "profile_native_ops", False)):
                            _sync_if_cuda(device)
                            attention_t0 = time.perf_counter()
                        outputs_native = native.gqa_exact_selected_attention(
                            q_all[:, int(local_qpos), :].to(device).contiguous(),
                            keys_all.contiguous(),
                            values_all.contiguous(),
                            selected_all.contiguous(),
                            int(group_size),
                            float(self.head_dim) ** -0.5,
                        )
                        if bool(getattr(args, "profile_native_ops", False)):
                            _sync_if_cuda(device)
                            stats[layer_id].add_native_timing(
                                attention_seconds=float(time.perf_counter() - attention_t0),
                            )
                        return outputs_native.reshape(num_heads, int(self.head_dim))
                    except Exception:
                        if str(args.selector_backend) == "cuda_ext":
                            raise
                for kv_head in range(num_kv_heads):
                    head_start = int(kv_head) * int(group_size)
                    head_end = min(num_heads, head_start + int(group_size))
                    if head_start >= head_end:
                        continue
                    queries = q_all[head_start:head_end, int(local_qpos), :].to(device)
                    index = prefix_index_for(int(kv_head), int(query_context_len))
                    if gqa_ranked_t is not None:
                        ranked_t = gqa_ranked_t[head_start:head_end]
                        dense_pq_scores = None
                        selector_mb = float(gqa_selector_mb)
                    elif index.pages:
                        if tail_vpq_enabled:
                            ranked_t, _ranked_scores, dense_pq_scores, _seconds, selector_mb, _nprobe = rank_paged_pq_batched_with_scores(
                                queries,
                                index,
                                mode=str(args.selector_mode),
                                selector_backend=str(args.selector_backend),
                                nprobes=nprobes,
                                budget=budget,
                                key_bytes=key_bytes,
                                subbits=int(args.subbits),
                                sync_for_timing=False,
                            )
                        else:
                            ranked_t, _ranked_scores, _seconds, selector_mb, _nprobe = rank_paged_pq_batched(
                                queries,
                                index,
                                mode=str(args.selector_mode),
                                selector_backend=str(args.selector_backend),
                                nprobes=nprobes,
                                budget=budget,
                                key_bytes=key_bytes,
                                subbits=int(args.subbits),
                                sync_for_timing=False,
                            )
                            dense_pq_scores = None
                    else:
                        ranked_t = torch.empty((head_end - head_start, 0), dtype=torch.long, device=device)
                        dense_pq_scores = torch.empty((head_end - head_start, 0), dtype=torch.float32, device=device)
                        selector_mb = 0.0
                    selected = ranked_t
                    if base_t is None:
                        base_t = decode_base_tokens_tensor(
                            int(query_context_len),
                            int(pos_sealed_end),
                            int(pos_indexed_end),
                        )
                    if base_t.numel():
                        selected = torch.cat(
                            [base_t.reshape(1, -1).expand(head_end - head_start, -1), ranked_t],
                            dim=1,
                        )
                    if selected.numel() == 0:
                        if gqa_exact_enabled:
                            gqa_selected_rows.append(selected)
                        else:
                            outputs[head_start:head_end].zero_()
                        selected_count = 0
                        exact_mask = torch.zeros((head_end - head_start, 0), dtype=torch.bool, device=device)
                        vpq_valid = torch.zeros((head_end - head_start, 0), dtype=torch.bool, device=device)
                        vpq_page_ids = torch.empty((head_end - head_start, 0), dtype=torch.long, device=device)
                        actual_value_subbits = int(args.value_subbits) if int(args.value_subbits) > 0 else int(args.subbits)
                    else:
                        selected_count = int(selected.shape[1])
                        flat = selected.reshape(-1)
                        if gqa_exact_enabled:
                            gqa_selected_rows.append(selected)
                            actual_value_subbits = int(args.value_subbits) if int(args.value_subbits) > 0 else int(args.subbits)
                        else:
                            selected_keys = torch_k_cache[int(kv_head)].index_select(0, flat).float().reshape(
                                head_end - head_start,
                                selected_count,
                                int(self.head_dim),
                            )
                            logits = torch.sum(
                                selected_keys * queries.reshape(head_end - head_start, 1, int(self.head_dim)),
                                dim=-1,
                            )
                            logits = logits / math.sqrt(float(self.head_dim))
                            weights = torch.softmax(logits.float(), dim=-1)
                            if str(args.selected_value_mode) == "vpq_value":
                                approx_values, vpq_valid, vpq_page_ids, actual_value_subbits = vpq_values_for_tokens_gpu(
                                    index=index,
                                    values=torch_v_cache[int(kv_head)],
                                    values_np=values_np_cache.get(int(kv_head)),
                                    tokens=selected,
                                    subbits=int(args.subbits),
                                    value_subvecs=int(args.value_subvecs),
                                    value_subbits=int(args.value_subbits),
                                    prefer_torch=str(getattr(args, "index_build_backend", "numpy")) == "torch_gpu",
                                    value_bytes=int(value_bytes),
                                )
                                exact_mask = selected_value_exact_mask_gpu(
                                    selected_logits=logits,
                                    rule=str(args.selected_value_exact_rule),
                                    exact_top=int(args.selected_value_exact_top),
                                    exact_mass=float(args.selected_value_exact_mass),
                                    min_top=int(args.selected_value_min_exact_top),
                                    max_top=int(args.selected_value_max_exact_top),
                                )
                                exact_v_is_empty = (
                                    str(args.selected_value_exact_rule) == "fixed"
                                    and int(args.selected_value_exact_top) <= 0
                                    and int(args.selected_value_min_exact_top) <= 0
                                    and int(args.selected_value_max_exact_top) <= 0
                                )
                                if exact_v_is_empty:
                                    selected_values = approx_values
                                else:
                                    exact_selected_values = torch_v_cache[int(kv_head)].index_select(0, flat).float().reshape(
                                        head_end - head_start,
                                        selected_count,
                                        int(self.head_dim),
                                    ).float()
                                    selected_values = torch.where(exact_mask.unsqueeze(-1), exact_selected_values, approx_values)
                            else:
                                exact_mask = torch.ones((head_end - head_start, selected_count), dtype=torch.bool, device=device)
                                vpq_valid = torch.zeros((head_end - head_start, selected_count), dtype=torch.bool, device=device)
                                vpq_page_ids = torch.full(
                                    (head_end - head_start, selected_count),
                                    -1,
                                    dtype=torch.long,
                                    device=device,
                                )
                                actual_value_subbits = int(args.value_subbits) if int(args.value_subbits) > 0 else int(args.subbits)
                                exact_selected_values = torch_v_cache[int(kv_head)].index_select(0, flat).float().reshape(
                                    head_end - head_start,
                                    selected_count,
                                    int(self.head_dim),
                                ).float()
                                selected_values = exact_selected_values
                            selected_only = torch.bmm(weights.unsqueeze(1), selected_values.float()).squeeze(1)
                            if tail_vpq_enabled and dense_pq_scores is not None and dense_pq_scores.numel() > 0:
                                all_vpq = reconstruct_all_vpq_values_gpu(
                                    index=index,
                                    values_np=values_np_cache.get(int(kv_head)),
                                    values=torch_v_cache[int(kv_head)],
                                    subbits=int(args.subbits),
                                    value_subvecs=int(args.value_subvecs),
                                    value_subbits=int(args.value_subbits),
                                    device=device,
                                    prefer_torch=str(getattr(args, "index_build_backend", "numpy")) == "torch_gpu",
                                    value_bytes=int(value_bytes),
                                )
                                if all_vpq is not None:
                                    all_values_approx, actual_value_subbits = all_vpq
                                    tail_logits = dense_pq_scores.float() / math.sqrt(float(self.head_dim))
                                    first_start = int(index.pages[0].start)
                                    page_size = int(index.pages[0].size)
                                    top_ordinals = torch.clamp(
                                        ranked_t - int(first_start),
                                        min=0,
                                        max=max(0, int(tail_logits.shape[1]) - 1),
                                    )
                                    if page_size > 0:
                                        top_page_ids = torch.div(top_ordinals, int(page_size), rounding_mode="floor")
                                        page_starts = torch.as_tensor(
                                            [int(page.start) for page in index.pages],
                                            dtype=torch.long,
                                            device=device,
                                        )
                                        top_rows = ranked_t - page_starts.index_select(0, top_page_ids.reshape(-1)).reshape_as(ranked_t)
                                        top_ordinals = top_page_ids * int(page_size) + top_rows
                                        top_ordinals = torch.clamp(
                                            top_ordinals,
                                            min=0,
                                            max=max(0, int(tail_logits.shape[1]) - 1),
                                        )
                                    tail_mask = torch.ones_like(tail_logits, dtype=torch.bool)
                                    if ranked_t.numel():
                                        tail_mask.scatter_(1, top_ordinals.to(torch.long), False)
                                    masked_tail_logits = tail_logits.masked_fill(~tail_mask, float("-inf"))
                                    max_score = torch.maximum(
                                        torch.max(logits.float(), dim=1).values,
                                        torch.max(masked_tail_logits, dim=1).values,
                                    )
                                    selected_weights = torch.exp(logits.float() - max_score.reshape(-1, 1))
                                    tail_weights = torch.exp(masked_tail_logits - max_score.reshape(-1, 1))
                                    selected_num = torch.bmm(
                                        selected_weights.unsqueeze(1),
                                        selected_values.float(),
                                    ).squeeze(1)
                                    tail_num = tail_weights @ all_values_approx.float()
                                    denom = torch.clamp(
                                        selected_weights.sum(dim=1) + tail_weights.sum(dim=1),
                                        min=1e-20,
                                    )
                                    tail_out = (selected_num + tail_num) / denom.reshape(-1, 1)
                                    blend = float(max(0.0, min(1.0, float(tail_blend_value))))
                                    outputs[head_start:head_end] = selected_only + blend * (tail_out - selected_only)
                                else:
                                    outputs[head_start:head_end] = selected_only
                            else:
                                outputs[head_start:head_end] = selected_only
                    code_bytes = 1 if int(actual_value_subbits) <= 8 else 2
                    subvecs_for_cost = int(args.value_subvecs) if int(args.value_subvecs) > 0 else int(args.subvecs)
                    subdim_for_cost = int(self.head_dim) // max(1, int(subvecs_for_cost))
                    if str(args.selected_value_mode) == "exact":
                        exact_kv_mb = float(selected_count * int(self.head_dim) * (key_bytes + value_bytes)) / MB
                        tail_count_for_cost = 0
                        tail_mb_for_cost = 0.0
                        if tail_vpq_enabled and index.pages:
                            dynamic_count = int(sum(int(page.size) for page in index.pages))
                            tail_count_for_cost = max(0, dynamic_count - int(ranked_t.shape[1]))
                            pages_read = int(len(index.pages)) if tail_count_for_cost > 0 else 0
                            dense_score_io_mb = float(dynamic_count * 4 * 2) / MB if tail_count_for_cost > 0 else 0.0
                            tail_mb_for_cost = (
                                float(
                                    pages_read
                                    * subvecs_for_cost
                                    * (1 << int(actual_value_subbits))
                                    * subdim_for_cost
                                    * value_bytes
                                )
                                + float(tail_count_for_cost * subvecs_for_cost * code_bytes)
                            ) / MB + dense_score_io_mb
                        stats[layer_id].add_count_repeated(
                            int(head_end) - int(head_start),
                            selected_count,
                            tail_count_for_cost,
                            float(selector_mb),
                            int(self.head_dim),
                            key_bytes,
                            value_bytes,
                            tail_mb_override=tail_mb_for_cost,
                            exact_kv_mb_override=exact_kv_mb,
                        )
                    else:
                        exact_mask_cpu = exact_mask.detach().cpu().numpy().astype(bool, copy=False)
                        vpq_valid_cpu = vpq_valid.detach().cpu().numpy().astype(bool, copy=False)
                        vpq_page_ids_cpu = vpq_page_ids.detach().cpu().numpy().astype(np.int64, copy=False)
                        for offset, head in enumerate(range(head_start, head_end)):
                            exact_value_count = int(np.count_nonzero(exact_mask_cpu[offset]))
                            compressed_mask = (~exact_mask_cpu[offset]) & vpq_valid_cpu[offset]
                            compressed_count = int(np.count_nonzero(compressed_mask))
                            pages_read = int(np.unique(vpq_page_ids_cpu[offset][compressed_mask]).size) if compressed_count else 0
                            fallback_count = int(selected_count - exact_value_count - compressed_count)
                            selected_value_mb = (
                                float(exact_value_count * int(self.head_dim) * value_bytes)
                                + float(fallback_count * int(self.head_dim) * value_bytes)
                                + float(
                                    pages_read
                                    * subvecs_for_cost
                                    * (1 << int(actual_value_subbits))
                                    * subdim_for_cost
                                    * value_bytes
                                )
                                + float(compressed_count * subvecs_for_cost * code_bytes)
                            ) / MB
                            exact_kv_mb = float(selected_count * int(self.head_dim) * key_bytes) / MB + selected_value_mb
                            stats[layer_id].add_count(
                                selected_count,
                                0,
                                float(selector_mb),
                                int(self.head_dim),
                                key_bytes,
                                value_bytes,
                                exact_kv_mb_override=exact_kv_mb,
                            )
                if gqa_exact_enabled:
                    if gqa_selected_rows:
                        selected_all = torch.cat(gqa_selected_rows, dim=0)
                    else:
                        selected_all = torch.empty((num_heads, 0), dtype=torch.long, device=device)
                    if selected_all.numel() == 0:
                        outputs.zero_()
                    else:
                        try:
                            native = load_selector_paged_pq_ext()
                            if bool(getattr(args, "profile_native_ops", False)):
                                _sync_if_cuda(device)
                                attention_t0 = time.perf_counter()
                            outputs = native.gqa_exact_selected_attention(
                                q_all[:, int(local_qpos), :].to(device).contiguous(),
                                keys_all.contiguous(),
                                values_all.contiguous(),
                                selected_all.contiguous(),
                                int(group_size),
                                float(self.head_dim) ** -0.5,
                            )
                            if bool(getattr(args, "profile_native_ops", False)):
                                _sync_if_cuda(device)
                                stats[layer_id].add_native_timing(
                                    attention_seconds=float(time.perf_counter() - attention_t0),
                                )
                        except Exception:
                            if str(args.selector_backend) == "cuda_ext":
                                raise
                            for head in range(num_heads):
                                kv_head = int(head // group_size)
                                selected_h = selected_all[head]
                                selected_keys_h = torch_k_cache[kv_head].index_select(0, selected_h).float()
                                logits_h = (selected_keys_h @ q_all[head, int(local_qpos), :].to(device)) / math.sqrt(float(self.head_dim))
                                weights_h = torch.softmax(logits_h.float(), dim=0)
                                values_h = torch_v_cache[kv_head].index_select(0, selected_h).float()
                                outputs[head] = weights_h @ values_h
                return outputs

            joint_gqa_ranked_t: torch.Tensor | None = None
            joint_gqa_ranked_scores: torch.Tensor | None = None
            joint_exact_scores_t: torch.Tensor | None = None
            joint_gqa_selector_mb = 0.0
            joint_vpq_runtime_cache: dict[tuple[object, ...], tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]] = {}

            def joint_vpq_sidecars_for(
                *,
                kv_head: int,
                index: GPUIndex,
                values_t: torch.Tensor,
                context_len_i: int,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
                use_joint_vpq_cache = _env_truthy("SELECTOR_PQ_JOINT_VPQ_CACHE", "1")
                use_persistent_vpq_cache = use_joint_vpq_cache and _env_truthy(
                    "SELECTOR_PQ_JOINT_PERSISTENT_VPQ_CACHE",
                    "0",
                )
                cache_key = joint_vpq_cache_key_for(int(kv_head), values_t, index)
                persistent_cache = getattr(self, "_pagedpq_joint_vpq_sidecar_cache", None)
                if use_persistent_vpq_cache and not isinstance(persistent_cache, dict):
                    persistent_cache = {}
                    setattr(self, "_pagedpq_joint_vpq_sidecar_cache", persistent_cache)
                cached = (
                    persistent_cache.get(cache_key)
                    if use_persistent_vpq_cache and isinstance(persistent_cache, dict)
                    else None
                )
                if cached is not None:
                    if len(cached) == 6:
                        (
                            cached_len,
                            cached_capacity,
                            vhat_cached,
                            residual_cached,
                            code_error_cached,
                            cached_subbits,
                        ) = cached
                        cached_capacity_i = int(cached_capacity)
                    else:
                        cached_len, vhat_cached, residual_cached, code_error_cached, cached_subbits = cached
                        cached_capacity_i = int(vhat_cached.shape[0])
                    cached_len_i = int(cached_len)
                    if cached_len_i >= int(context_len_i):
                        return (
                            vhat_cached[:context_len_i],
                            residual_cached[:context_len_i],
                            code_error_cached[:context_len_i],
                            int(cached_subbits),
                        )
                    if cached_len_i >= 0 and cached_len_i < int(context_len_i):
                        context_len_target_i = int(context_len_i)
                        if cached_capacity_i < context_len_target_i:
                            grow_pad_i = max(
                                0,
                                _env_int("SELECTOR_PQ_JOINT_PERSISTENT_VPQ_CACHE_GROW_PAD", 256),
                            )
                            new_capacity_i = max(
                                context_len_target_i,
                                cached_capacity_i + max(grow_pad_i, context_len_target_i - cached_capacity_i),
                            )
                            vhat_buf_t = torch.empty(
                                (new_capacity_i, int(vhat_cached.shape[1])),
                                dtype=vhat_cached.dtype,
                                device=vhat_cached.device,
                            )
                            residual_buf_t = torch.empty(
                                (new_capacity_i, int(residual_cached.shape[1])),
                                dtype=residual_cached.dtype,
                                device=residual_cached.device,
                            )
                            code_error_buf_t = torch.empty(
                                (new_capacity_i,),
                                dtype=code_error_cached.dtype,
                                device=code_error_cached.device,
                            )
                            if cached_len_i > 0:
                                vhat_buf_t[:cached_len_i].copy_(vhat_cached[:cached_len_i])
                                residual_buf_t[:cached_len_i].copy_(residual_cached[:cached_len_i])
                                code_error_buf_t[:cached_len_i].copy_(code_error_cached[:cached_len_i])
                            vhat_cached = vhat_buf_t
                            residual_cached = residual_buf_t
                            code_error_cached = code_error_buf_t
                            cached_capacity_i = int(new_capacity_i)
                        extra_values = values_t[cached_len_i:context_len_target_i].float()
                        if int(extra_values.numel()) > 0:
                            vhat_cached[cached_len_i:context_len_target_i].copy_(
                                extra_values.to(dtype=vhat_cached.dtype)
                            )
                            residual_cached[cached_len_i:context_len_target_i].zero_()
                            code_error_cached[cached_len_i:context_len_target_i].zero_()
                        persistent_cache[cache_key] = (
                            int(context_len_target_i),
                            int(cached_capacity_i),
                            vhat_cached,
                            residual_cached,
                            code_error_cached,
                            int(cached_subbits),
                        )
                        return (
                            vhat_cached[:context_len_target_i],
                            residual_cached[:context_len_target_i],
                            code_error_cached[:context_len_target_i],
                            int(cached_subbits),
                        )

                use_incremental_vpq_sidecar = _env_truthy(
                    "SELECTOR_PQ_JOINT_INCREMENTAL_VPQ_SIDECAR",
                    "0",
                )
                if (
                    use_incremental_vpq_sidecar
                    and use_persistent_vpq_cache
                    and isinstance(persistent_cache, dict)
                    and persistent_cache
                ):
                    best_old_key = None
                    best_old_end = -1
                    for old_key in persistent_cache:
                        if not isinstance(old_key, tuple) or len(old_key) != len(cache_key):
                            continue
                        if old_key[:6] != cache_key[:6]:
                            continue
                        if int(old_key[7]) != int(cache_key[7]) or int(old_key[9]) != int(cache_key[9]):
                            continue
                        old_end_i = int(old_key[8])
                        new_end_i = int(cache_key[8])
                        if old_end_i < 0 or old_end_i > new_end_i:
                            continue
                        if old_end_i > best_old_end:
                            best_old_end = old_end_i
                            best_old_key = old_key
                    if best_old_key is not None and best_old_key != cache_key:
                        old_cached = persistent_cache.get(best_old_key)
                        if old_cached is not None:
                            if len(old_cached) == 6:
                                (
                                    cached_len,
                                    cached_capacity,
                                    vhat_cached,
                                    residual_cached,
                                    code_error_cached,
                                    cached_subbits,
                                ) = old_cached
                                cached_capacity_i = int(cached_capacity)
                            else:
                                cached_len, vhat_cached, residual_cached, code_error_cached, cached_subbits = old_cached
                                cached_capacity_i = int(vhat_cached.shape[0])
                            cached_len_i = int(cached_len)
                            context_len_target_i = int(context_len_i)
                            if cached_capacity_i < context_len_target_i:
                                grow_pad_i = max(
                                    0,
                                    _env_int("SELECTOR_PQ_JOINT_PERSISTENT_VPQ_CACHE_GROW_PAD", 256),
                                )
                                new_capacity_i = max(
                                    context_len_target_i,
                                    cached_capacity_i + max(grow_pad_i, context_len_target_i - cached_capacity_i),
                                )
                                vhat_buf_t = torch.empty(
                                    (new_capacity_i, int(vhat_cached.shape[1])),
                                    dtype=vhat_cached.dtype,
                                    device=vhat_cached.device,
                                )
                                residual_buf_t = torch.empty(
                                    (new_capacity_i, int(residual_cached.shape[1])),
                                    dtype=residual_cached.dtype,
                                    device=residual_cached.device,
                                )
                                code_error_buf_t = torch.empty(
                                    (new_capacity_i,),
                                    dtype=code_error_cached.dtype,
                                    device=code_error_cached.device,
                                )
                                copy_len_i = min(cached_len_i, int(vhat_cached.shape[0]))
                                if copy_len_i > 0:
                                    vhat_buf_t[:copy_len_i].copy_(vhat_cached[:copy_len_i])
                                    residual_buf_t[:copy_len_i].copy_(residual_cached[:copy_len_i])
                                    code_error_buf_t[:copy_len_i].copy_(code_error_cached[:copy_len_i])
                                vhat_cached = vhat_buf_t
                                residual_cached = residual_buf_t
                                code_error_cached = code_error_buf_t
                                cached_capacity_i = int(new_capacity_i)
                            old_sealed_end_i = max(0, min(int(best_old_end), context_len_target_i))
                            new_sealed_end_i = max(0, min(int(cache_key[8]), context_len_target_i))
                            if new_sealed_end_i > old_sealed_end_i:
                                update_tokens_t = torch.arange(
                                    old_sealed_end_i,
                                    new_sealed_end_i,
                                    dtype=torch.long,
                                    device=values_t.device,
                                )
                                vhat_update_t, valid_update_t, page_ids_update_t, actual_subbits_i = vpq_values_for_tokens_gpu(
                                    index=index,
                                    values=values_t,
                                    values_np=None,
                                    tokens=update_tokens_t,
                                    subbits=int(args.subbits),
                                    value_subvecs=int(args.value_subvecs),
                                    value_subbits=int(args.value_subbits),
                                    prefer_torch=True,
                                    value_bytes=int(value_bytes),
                                )
                                residual_update_t = values_t.index_select(0, update_tokens_t).float() - vhat_update_t.float()
                                code_error_update_t, actual_subbits_i = value_vpq_code_stat_risk_subset_torch(
                                    index=index,
                                    values=values_t,
                                    tokens=update_tokens_t,
                                    residual_subset=residual_update_t,
                                    valid=valid_update_t,
                                    page_ids=page_ids_update_t,
                                    subbits=int(args.subbits),
                                    value_subvecs=int(args.value_subvecs),
                                    value_subbits=int(args.value_subbits),
                                    value_bytes=int(value_bytes),
                                )
                                vhat_cached[old_sealed_end_i:new_sealed_end_i].copy_(
                                    vhat_update_t.to(dtype=vhat_cached.dtype)
                                )
                                residual_cached[old_sealed_end_i:new_sealed_end_i].copy_(
                                    residual_update_t.to(dtype=residual_cached.dtype)
                                )
                                code_error_cached[old_sealed_end_i:new_sealed_end_i].copy_(
                                    code_error_update_t.to(dtype=code_error_cached.dtype)
                                )
                                cached_subbits = int(actual_subbits_i)
                            if context_len_target_i > new_sealed_end_i:
                                extra_values = values_t[new_sealed_end_i:context_len_target_i].float()
                                if int(extra_values.numel()) > 0:
                                    vhat_cached[new_sealed_end_i:context_len_target_i].copy_(
                                        extra_values.to(dtype=vhat_cached.dtype)
                                    )
                                    residual_cached[new_sealed_end_i:context_len_target_i].zero_()
                                    code_error_cached[new_sealed_end_i:context_len_target_i].zero_()
                            persistent_cache[cache_key] = (
                                int(context_len_target_i),
                                int(cached_capacity_i),
                                vhat_cached,
                                residual_cached,
                                code_error_cached,
                                int(cached_subbits),
                            )
                            max_entries = max(
                                1,
                                int(
                                    os.environ.get(
                                        "SELECTOR_PQ_JOINT_PERSISTENT_VPQ_CACHE_MAX_ENTRIES",
                                        str(max(1, num_kv_heads)),
                                    )
                                ),
                            )
                            while len(persistent_cache) > max_entries:
                                oldest_key = next(iter(persistent_cache))
                                if oldest_key == cache_key and len(persistent_cache) == 1:
                                    break
                                persistent_cache.pop(oldest_key, None)
                            return (
                                vhat_cached[:context_len_target_i],
                                residual_cached[:context_len_target_i],
                                code_error_cached[:context_len_target_i],
                                int(cached_subbits),
                            )

                runtime_key = (*cache_key, int(context_len_i))
                if use_joint_vpq_cache and runtime_key in joint_vpq_runtime_cache:
                    return joint_vpq_runtime_cache[runtime_key]

                all_tokens_t = torch.arange(int(context_len_i), dtype=torch.long, device=values_t.device)
                vhat_all_t, vpq_valid_t, vpq_page_ids_t, actual_value_subbits_for_cost = vpq_values_for_tokens_gpu(
                    index=index,
                    values=values_t,
                    values_np=None,
                    tokens=all_tokens_t,
                    subbits=int(args.subbits),
                    value_subvecs=int(args.value_subvecs),
                    value_subbits=int(args.value_subbits),
                    prefer_torch=True,
                    value_bytes=int(value_bytes),
                )
                residual_t = values_t.float() - vhat_all_t.float()
                code_error_t, actual_value_subbits_for_cost = value_vpq_code_stat_risk_torch(
                    index=index,
                    values=values_t,
                    vhat_all=vhat_all_t,
                    residual_all=residual_t,
                    valid=vpq_valid_t,
                    page_ids=vpq_page_ids_t,
                    subbits=int(args.subbits),
                    value_subvecs=int(args.value_subvecs),
                    value_subbits=int(args.value_subbits),
                    value_bytes=int(value_bytes),
                )
                out = (
                    vhat_all_t.detach(),
                    residual_t.detach(),
                    code_error_t.detach(),
                    int(actual_value_subbits_for_cost),
                )
                if use_joint_vpq_cache:
                    joint_vpq_runtime_cache[runtime_key] = out
                    if use_persistent_vpq_cache and isinstance(persistent_cache, dict):
                        cache_len_i = int(context_len_i)
                        grow_pad_i = max(
                            0,
                            _env_int("SELECTOR_PQ_JOINT_PERSISTENT_VPQ_CACHE_GROW_PAD", 256),
                        )
                        cache_capacity_i = int(cache_len_i + grow_pad_i)
                        vhat_cached = out[0]
                        residual_cached = out[1]
                        code_error_cached = out[2]
                        if cache_capacity_i > cache_len_i:
                            vhat_buf_t = torch.empty(
                                (cache_capacity_i, int(vhat_cached.shape[1])),
                                dtype=vhat_cached.dtype,
                                device=vhat_cached.device,
                            )
                            residual_buf_t = torch.empty(
                                (cache_capacity_i, int(residual_cached.shape[1])),
                                dtype=residual_cached.dtype,
                                device=residual_cached.device,
                            )
                            code_error_buf_t = torch.empty(
                                (cache_capacity_i,),
                                dtype=code_error_cached.dtype,
                                device=code_error_cached.device,
                            )
                            vhat_buf_t[:cache_len_i].copy_(vhat_cached)
                            residual_buf_t[:cache_len_i].copy_(residual_cached)
                            code_error_buf_t[:cache_len_i].copy_(code_error_cached)
                            vhat_cached = vhat_buf_t
                            residual_cached = residual_buf_t
                            code_error_cached = code_error_buf_t
                        persistent_cache[cache_key] = (
                            int(cache_len_i),
                            int(cache_capacity_i),
                            vhat_cached.detach(),
                            residual_cached.detach(),
                            code_error_cached.detach(),
                            int(out[3]),
                        )
                        max_entries = max(
                            1,
                            int(
                                os.environ.get(
                                    "SELECTOR_PQ_JOINT_PERSISTENT_VPQ_CACHE_MAX_ENTRIES",
                                    str(max(1, num_kv_heads)),
                                )
                            ),
                        )
                        while len(persistent_cache) > max_entries:
                            oldest_key = next(iter(persistent_cache))
                            if oldest_key == cache_key and len(persistent_cache) == 1:
                                break
                            persistent_cache.pop(oldest_key, None)
                return out

            def joint_vpq_pack_and_fallback_for(
                *,
                index: GPUIndex,
                values_t: torch.Tensor,
                context_len_i: int,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, torch.Tensor] | None:
                if int(args.value_subvecs) != 1:
                    return None
                pack = value_vpq_pack_torch(
                    index=index,
                    values=values_t,
                    value_subvecs=int(args.value_subvecs),
                    value_subbits=int(args.value_subbits),
                    key_bytes=int(value_bytes),
                    device=values_t.device,
                )
                if pack is None or not index.pages:
                    return None
                codebooks, codes, page_starts, page_size, actual_value_subbits = pack
                if int(codebooks.shape[1]) != 1:
                    return None
                fallback_parts: list[torch.Tensor] = []
                cursor_i = 0
                for page in sorted(index.pages, key=lambda p: int(p.start)):
                    start_i = max(0, min(int(page.start), int(context_len_i)))
                    end_i = max(start_i, min(int(page.start) + int(page.size), int(context_len_i)))
                    if start_i > cursor_i:
                        fallback_parts.append(
                            torch.arange(cursor_i, start_i, dtype=torch.long, device=values_t.device)
                        )
                    cursor_i = max(cursor_i, end_i)
                if cursor_i < int(context_len_i):
                    fallback_parts.append(
                        torch.arange(cursor_i, int(context_len_i), dtype=torch.long, device=values_t.device)
                    )
                if fallback_parts:
                    fallback_tokens = torch.cat(fallback_parts, dim=0).contiguous()
                else:
                    fallback_tokens = torch.empty((0,), dtype=torch.long, device=values_t.device)
                return codebooks, codes, page_starts, int(page_size), int(actual_value_subbits), fallback_tokens

            if (
                online_confidence_rule == "joint_kv_stability"
                and str(args.selector_mode) == "fullscan"
                and str(args.selector_backend) in {"cuda_ext", "auto"}
            ):
                if _env_truthy("SELECTOR_PQ_JOINT_GQA_RANK", "0"):
                    try:
                        gqa_indexes = [prefix_index_for(int(kv_head), int(context_len)) for kv_head in range(num_kv_heads)]
                        if gqa_indexes and all(index.pages for index in gqa_indexes):
                            max_joint_rank_budget = max(
                                int(sum(int(page.size) for page in index.pages))
                                for index in gqa_indexes
                            )
                            if max_joint_rank_budget > 0:
                                native = load_selector_paged_pq_ext()
                                codebooks, codes, page_starts = gqa_native_fullscan_pack(gqa_indexes)
                                if bool(getattr(args, "profile_native_ops", False)):
                                    _sync_if_cuda(device)
                                    selector_t0 = time.perf_counter()
                                joint_gqa_ranked_t, joint_gqa_ranked_scores = native.gqa_fullscan_pq_topk(
                                    q_all[:, 0, :].to(device).contiguous(),
                                    codebooks,
                                    codes,
                                    page_starts,
                                    int(group_size),
                                    int(max_joint_rank_budget),
                                )
                                if bool(getattr(args, "profile_native_ops", False)):
                                    _sync_if_cuda(device)
                                    stats[layer_id].add_native_timing(
                                        selector_seconds=float(time.perf_counter() - selector_t0)
                                    )
                                joint_gqa_selector_mb = selector_bytes_fullscan(
                                    gqa_indexes[0],
                                    key_bytes=int(key_bytes),
                                    subbits=int(args.subbits),
                                ) / MB
                    except Exception:
                        if str(args.selector_backend) == "cuda_ext":
                            raise
                        joint_gqa_ranked_t = None
                        joint_gqa_ranked_scores = None
                if _env_truthy("SELECTOR_PQ_JOINT_EXACT_GEMM", "0"):
                    try:
                        if bool(getattr(args, "profile_native_ops", False)):
                            _sync_if_cuda(device)
                            exact_t0 = time.perf_counter()
                        joint_exact_scores_t = torch.empty(
                            (num_heads, int(context_len)),
                            dtype=torch.float32,
                            device=device,
                        )
                        score_scale = 1.0 / math.sqrt(float(self.head_dim))
                        for kv_head in range(num_kv_heads):
                            head_start = int(kv_head) * int(group_size)
                            head_end = min(num_heads, head_start + int(group_size))
                            if head_start >= head_end:
                                continue
                            queries_h = q_all[head_start:head_end, 0, :].to(device=device, dtype=torch.float32)
                            keys_h = torch_k_cache[int(kv_head)][: int(context_len)].to(device=device, dtype=torch.float32)
                            joint_exact_scores_t[head_start:head_end] = (queries_h @ keys_h.transpose(0, 1)) * score_scale
                        if bool(getattr(args, "profile_native_ops", False)):
                            _sync_if_cuda(device)
                            stats[layer_id].add_native_detail_timing(
                                exact_logit_seconds=float(time.perf_counter() - exact_t0)
                            )
                    except Exception:
                        if str(args.selector_backend) == "cuda_ext":
                            raise
                        joint_exact_scores_t = None

            def approximate_one_head(head: int, local_qpos: int, query_context_len: int) -> torch.Tensor:
                budget = int(budget_by_head.get(int(head), int(args.budget)))
                rank_budget = int(budget)
                joint_kv_enabled = online_confidence_rule == "joint_kv_stability"
                joint_k_budgets = (
                    _parse_budget_schedule(str(getattr(args, "joint_kv_k_budgets", "")), name="joint_kv_k_budgets")
                    if joint_kv_enabled
                    else []
                )
                joint_v_budgets = (
                    _parse_budget_schedule(str(getattr(args, "joint_kv_v_budgets", "")), name="joint_kv_v_budgets")
                    if joint_kv_enabled
                    else []
                )
                if online_confidence_rule == "geometric_tail_stability_switch":
                    raise RuntimeError("geometric_tail_stability_switch requires the native batched fast path")
                if online_confidence_rule == "geometric_probe_tail_switch":
                    rank_budget = max(
                        int(rank_budget),
                        int(getattr(args, "geometric_min_budget", 0)),
                        int(getattr(args, "geometric_max_budget", 0)),
                    )
                elif joint_kv_enabled:
                    rank_budget = max(int(rank_budget), int(max(joint_k_budgets)))
                elif online_confidence_rule in {"pq_proxy_mass_budget", "pq_ranked_mass_budget"}:
                    raise RuntimeError(f"{online_confidence_rule} requires the native batched fast path")
                kv_head = int(head // group_size)
                query = q_all[head, int(local_qpos), :].to(device)
                index = prefix_index_for(kv_head, int(query_context_len))
                pending = list(
                    range(
                        max(0, int(index.pending_start)),
                        max(0, min(int(index.indexed_end), int(query_context_len))),
                    )
                )
                base = unique_tokens(
                    static_tokens(int(query_context_len) - 1, int(args.static_prefix), int(args.static_suffix)) + pending,
                    context_len=int(query_context_len),
                )
                joint_tail_tokens_t: torch.Tensor | None = None
                joint_tail_scores_t: torch.Tensor | None = None
                if joint_kv_enabled and joint_gqa_ranked_t is not None and joint_gqa_ranked_scores is not None:
                    ranked_t = joint_gqa_ranked_t[int(head)].to(device=device, dtype=torch.long)
                    ranked_scores_t = joint_gqa_ranked_scores[int(head)].to(device=device, dtype=torch.float32)
                    if joint_kv_enabled:
                        ranked_cpu = np.empty((0,), dtype=np.int64)
                        ranked_scores_cpu = np.empty((0,), dtype=np.float32)
                    else:
                        ranked_cpu = np.asarray(ranked_t.detach().cpu().tolist(), dtype=np.int64)
                        ranked_scores_cpu = np.asarray(ranked_scores_t.detach().cpu().tolist(), dtype=np.float32)
                    selector_mb = float(joint_gqa_selector_mb)
                elif index.pages and str(args.selector_mode) == "oracle":
                    token_parts = [
                        torch.arange(
                            int(page.start),
                            min(int(page.start) + int(page.size), int(query_context_len)),
                            dtype=torch.long,
                            device=device,
                        )
                        for page in index.pages
                        if int(page.start) < int(query_context_len) and int(page.size) > 0
                    ]
                    if token_parts:
                        tokens_all = torch.cat(token_parts)
                        exact_scores = torch_k_cache[kv_head].index_select(0, tokens_all).float() @ query
                        order = torch.argsort(exact_scores, descending=True, stable=True)
                        take = min(int(rank_budget), int(order.numel()))
                        keep = order[:take]
                        ranked_t = tokens_all.index_select(0, keep)
                        ranked_scores_t = exact_scores.index_select(0, keep)
                        selector_mb = float(int(query_context_len) * int(self.head_dim) * key_bytes) / MB
                    else:
                        ranked_t = torch.empty((0,), dtype=torch.long, device=device)
                        ranked_scores_t = torch.empty((0,), dtype=torch.float32, device=device)
                        selector_mb = 0.0
                    ranked_cpu = ranked_t.detach().cpu().numpy().astype(np.int64, copy=False)
                    ranked_scores_cpu = ranked_scores_t.detach().cpu().numpy().astype(np.float32, copy=False)
                elif index.pages:
                    selector_t0 = 0.0
                    if bool(getattr(args, "profile_native_ops", False)):
                        _sync_if_cuda(device)
                        selector_t0 = time.perf_counter()
                    if joint_kv_enabled and str(args.selector_mode) == "fullscan":
                        selector_topk_budget = 1
                        ranked_rows_t, ranked_score_rows_t, dense_score_rows_t, _seconds, selector_mb, _nprobe = (
                            rank_paged_pq_batched_with_scores(
                                query.reshape(1, -1).contiguous(),
                                index,
                                mode=str(args.selector_mode),
                                selector_backend=str(args.selector_backend),
                                nprobes=nprobes,
                                budget=int(selector_topk_budget),
                                key_bytes=key_bytes,
                                subbits=int(args.subbits),
                            )
                        )
                        ranked_t = ranked_rows_t[0].to(device=device, dtype=torch.long)
                        ranked_scores_t = ranked_score_rows_t[0].to(device=device, dtype=torch.float32)
                        token_parts = [
                            torch.arange(
                                int(page.start),
                                min(int(page.start) + int(page.size), int(query_context_len)),
                                dtype=torch.long,
                                device=device,
                            )
                            for page in index.pages
                            if int(page.start) < int(query_context_len) and int(page.size) > 0
                        ]
                        joint_tail_tokens_t = torch.cat(token_parts) if token_parts else torch.empty((0,), dtype=torch.long, device=device)
                        joint_tail_scores_t = dense_score_rows_t[0].to(device=device, dtype=torch.float32)
                    else:
                        ranked_t, ranked_scores_t, _seconds, selector_mb, _nprobe = rank_paged_pq(
                            query,
                            index,
                            mode=str(args.selector_mode),
                            selector_backend=str(args.selector_backend),
                            nprobes=nprobes,
                            budget=rank_budget,
                            key_bytes=key_bytes,
                            subbits=int(args.subbits),
                        )
                    if bool(getattr(args, "profile_native_ops", False)):
                        _sync_if_cuda(device)
                        stats[layer_id].add_native_timing(
                            selector_seconds=float(time.perf_counter() - selector_t0)
                        )
                    if joint_kv_enabled:
                        ranked_cpu = np.empty((0,), dtype=np.int64)
                        ranked_scores_cpu = np.empty((0,), dtype=np.float32)
                    else:
                        ranked_cpu = ranked_t.detach().cpu().numpy().astype(np.int64, copy=False)
                        ranked_scores_cpu = ranked_scores_t.detach().cpu().numpy().astype(np.float32, copy=False)
                else:
                    ranked_t = torch.empty((0,), dtype=torch.long, device=device)
                    ranked_scores_t = torch.empty((0,), dtype=torch.float32, device=device)
                    ranked_cpu = np.empty((0,), dtype=np.int64)
                    ranked_scores_cpu = np.empty((0,), dtype=np.float32)
                    selector_mb = 0.0
                rerank_key_mb = 0.0
                if joint_kv_enabled and int(args.rerank_candidates) > 0:
                    raise RuntimeError("joint_kv_stability does not support rerank_candidates; rerank changes CPU-reference semantics")
                if int(args.rerank_candidates) > 0 and ranked_cpu.size:
                    rerank_count = min(int(args.rerank_candidates), int(ranked_cpu.size))
                    rerank_tokens = torch.as_tensor(ranked_cpu[:rerank_count], dtype=torch.long, device=device)
                    rerank_scores = torch_k_cache[kv_head].index_select(0, rerank_tokens).float() @ query
                    rerank_order = torch.argsort(rerank_scores, descending=True, stable=True).detach().cpu().numpy()
                    reranked = ranked_cpu[:rerank_count][rerank_order].astype(np.int64, copy=False)
                    reranked_set = set(int(tok) for tok in reranked.tolist())
                    rest = np.asarray([int(tok) for tok in ranked_cpu.tolist() if int(tok) not in reranked_set], dtype=np.int64)
                    ranked_cpu = np.concatenate([reranked, rest]) if rest.size else reranked
                    rerank_key_mb = float(rerank_count * int(self.head_dim) * key_bytes) / MB
                if joint_kv_enabled:
                    if str(args.selector_mode) != "fullscan":
                        raise RuntimeError("joint_kv_stability requires fullscan paged K-PQ ranking")
                    if str(args.tail_score_calibration) not in {"none", "affine_selected"}:
                        raise RuntimeError(f"unsupported tail_score_calibration for joint_kv_stability: {args.tail_score_calibration}")
                    values_t = torch_v_cache[kv_head][: int(query_context_len)].float()
                    vsidecar_t0 = 0.0
                    if bool(getattr(args, "profile_native_ops", False)):
                        _sync_if_cuda(device)
                        vsidecar_t0 = time.perf_counter()
                    vhat_all_t, residual_t, code_error_t, actual_value_subbits_for_cost = joint_vpq_sidecars_for(
                        kv_head=int(kv_head),
                        index=index,
                        values_t=values_t,
                        context_len_i=int(query_context_len),
                    )
                    if bool(getattr(args, "profile_native_ops", False)):
                        _sync_if_cuda(device)
                        stats[layer_id].add_native_detail_timing(
                            output_seconds=float(time.perf_counter() - vsidecar_t0)
                        )
                    if joint_exact_scores_t is not None and int(query_context_len) == int(context_len):
                        exact_scores_t = joint_exact_scores_t[int(head)]
                    else:
                        exact_t0 = 0.0
                        if bool(getattr(args, "profile_native_ops", False)):
                            _sync_if_cuda(device)
                            exact_t0 = time.perf_counter()
                        exact_scores_t = (
                            torch_k_cache[kv_head][: int(query_context_len)].float()
                            @ query.float()
                        ) / math.sqrt(float(self.head_dim))
                        if bool(getattr(args, "profile_native_ops", False)):
                            _sync_if_cuda(device)
                            stats[layer_id].add_native_detail_timing(
                                exact_logit_seconds=float(time.perf_counter() - exact_t0)
                            )

                    context_len_i = int(query_context_len)
                    sqrt_dim = float(math.sqrt(float(self.head_dim)))
                    prob_dtype = torch.float32 if _env_truthy("SELECTOR_PQ_JOINT_FP32_PROBS", "1") else torch.float64
                    ranked_select_t = ranked_t.to(device=device, dtype=torch.long)
                    ranked_t_full = (
                        joint_tail_tokens_t.to(device=device, dtype=torch.long)
                        if joint_tail_tokens_t is not None
                        else ranked_select_t
                    )
                    ranked_scores_t_full = (
                        joint_tail_scores_t.to(device=device, dtype=torch.float32)
                        if joint_tail_scores_t is not None
                        else ranked_scores_t.to(device=device, dtype=torch.float32)
                    )

                    def mixed_probs_for_selected(selected_t_i: torch.Tensor) -> torch.Tensor:
                        score_vec = torch.full(
                            (context_len_i,),
                            -float("inf"),
                            dtype=prob_dtype,
                            device=device,
                        )
                        selected_t_i = selected_t_i.to(device=device, dtype=torch.long)
                        selected_mask = torch.zeros((context_len_i,), dtype=torch.bool, device=device)
                        if int(selected_t_i.numel()) > 0:
                            selected_t_i = selected_t_i[(selected_t_i >= 0) & (selected_t_i < context_len_i)]
                            if int(selected_t_i.numel()) > 0:
                                selected_mask.index_fill_(0, selected_t_i, True)
                                score_vec.index_copy_(0, selected_t_i, exact_scores_t.index_select(0, selected_t_i).to(prob_dtype))

                        scale = 1.0
                        bias = 0.0
                        valid_rank = (ranked_t_full >= 0) & (ranked_t_full < context_len_i)
                        if (
                            str(args.tail_score_calibration) == "affine_selected"
                            and int(selected_t_i.numel()) > 0
                            and int(torch.count_nonzero(valid_rank).item()) > 0
                        ):
                            valid_tokens = ranked_t_full[valid_rank]
                            selected_rank = selected_mask.index_select(0, valid_tokens)
                            if int(torch.count_nonzero(selected_rank).item()) >= 2:
                                x = ranked_scores_t_full[valid_rank][selected_rank].to(prob_dtype) / sqrt_dim
                                y = exact_scores_t.index_select(0, valid_tokens[selected_rank]).to(prob_dtype)
                                x_mean = torch.mean(x)
                                y_mean = torch.mean(y)
                                x_var = torch.mean((x - x_mean) * (x - x_mean))
                                if float(x_var.item()) <= 1e-20:
                                    scale = 0.0
                                    bias = float(y_mean.item())
                                else:
                                    cov = torch.mean((x - x_mean) * (y - y_mean))
                                    fitted_scale = float((cov / x_var).item())
                                    fitted_bias = float((y_mean - fitted_scale * x_mean).item())
                                    if fitted_scale > 0.0 and math.isfinite(fitted_scale):
                                        scale = fitted_scale
                                        bias = fitted_bias

                        if int(ranked_t_full.numel()) > 0 and int(torch.count_nonzero(valid_rank).item()) > 0:
                            valid_tokens = ranked_t_full[valid_rank]
                            tail_rank = ~selected_mask.index_select(0, valid_tokens)
                            if int(torch.count_nonzero(tail_rank).item()) > 0:
                                tail_tokens = valid_tokens[tail_rank]
                                tail_scores = (
                                    float(scale)
                                    * (ranked_scores_t_full[valid_rank][tail_rank].to(prob_dtype) / sqrt_dim)
                                    + float(bias)
                                )
                                score_vec.index_copy_(0, tail_tokens, tail_scores)

                        missing = ~torch.isfinite(score_vec)
                        if bool(torch.any(missing)):
                            missing_idx = torch.nonzero(missing, as_tuple=False).reshape(-1)
                            score_vec.index_copy_(0, missing_idx, exact_scores_t.index_select(0, missing_idx).to(prob_dtype))
                        return torch.softmax(score_vec, dim=0)

                    v_mb_by_idx: list[float] = []
                    base_t = torch.as_tensor(base, dtype=torch.long, device=device)
                    if int(base_t.numel()) > 0:
                        base_t = base_t[(base_t >= 0) & (base_t < int(query_context_len))]
                    base_rank_mask_t = torch.zeros((context_len_i,), dtype=torch.bool, device=device)
                    if int(base_t.numel()) > 0:
                        base_rank_mask_t.index_fill_(0, base_t, True)
                    if joint_tail_tokens_t is not None and joint_tail_scores_t is not None:
                        budget_source_tokens_t = ranked_t_full
                        budget_source_scores_t = ranked_scores_t_full
                    else:
                        budget_source_tokens_t = ranked_select_t
                        budget_source_scores_t = ranked_scores_t.to(device=device, dtype=torch.float32)
                    valid_rank_for_budget_t = (budget_source_tokens_t >= 0) & (budget_source_tokens_t < context_len_i)
                    ranked_valid_for_budget_t = budget_source_tokens_t[valid_rank_for_budget_t]
                    ranked_valid_scores_for_budget_t = budget_source_scores_t[valid_rank_for_budget_t]
                    if int(ranked_valid_for_budget_t.numel()) > 0:
                        nonbase_mask_t = ~base_rank_mask_t.index_select(0, ranked_valid_for_budget_t)
                        ranked_nonbase_t = ranked_valid_for_budget_t[nonbase_mask_t]
                        ranked_nonbase_scores_t = ranked_valid_scores_for_budget_t[nonbase_mask_t]
                    else:
                        ranked_nonbase_t = ranked_valid_for_budget_t
                        ranked_nonbase_scores_t = ranked_valid_scores_for_budget_t

                    selected_by_take_torch: dict[int, torch.Tensor] = {}

                    def selected_for_budget_torch(k_budget: int) -> torch.Tensor:
                        take = max(0, min(int(k_budget), int(ranked_nonbase_t.numel())))
                        cached_selected = selected_by_take_torch.get(int(take))
                        if cached_selected is not None:
                            return cached_selected
                        if take > 0 and joint_tail_tokens_t is not None and joint_tail_scores_t is not None:
                            order_t = torch.topk(
                                ranked_nonbase_scores_t,
                                k=int(take),
                                largest=True,
                                sorted=True,
                            ).indices
                            add_t = ranked_nonbase_t.index_select(0, order_t)
                        else:
                            add_t = ranked_nonbase_t[: int(take)]
                        if int(base_t.numel()) == 0:
                            selected_out = add_t
                            selected_by_take_torch[int(take)] = selected_out
                            return selected_out
                        if int(add_t.numel()) == 0:
                            selected_out = base_t
                            selected_by_take_torch[int(take)] = selected_out
                            return selected_out
                        selected_out = torch.cat((base_t, add_t), dim=0)
                        selected_by_take_torch[int(take)] = selected_out
                        return selected_out

                    actual_value_subbits = int(actual_value_subbits_for_cost)
                    actual_value_subvecs = int(args.value_subvecs) if int(args.value_subvecs) > 0 else int(args.subvecs)
                    code_bytes = 1 if int(actual_value_subbits) <= 8 else 2
                    metadata_mb = (
                        float(int(query_context_len) * actual_value_subvecs * code_bytes)
                        + float(
                            len(index.pages)
                            * actual_value_subvecs
                            * (1 << int(actual_value_subbits))
                            * int(getattr(args, "value_code_stat_bytes", getattr(args, "selected_value_residual_norm_bytes", 2)))
                        )
                    ) / MB
                    v_pq_codebook_mb = float(
                        len(index.pages)
                        * actual_value_subvecs
                        * (1 << int(actual_value_subbits))
                        * (int(self.head_dim) // max(1, actual_value_subvecs))
                        * value_bytes
                    ) / MB
                    for v_budget in joint_v_budgets:
                        exact_count = max(0, min(int(v_budget), int(query_context_len)))
                        exact_v_mb = float(exact_count * int(self.head_dim) * value_bytes) / MB
                        compressed_v_codes_mb = (
                            float(max(0, int(query_context_len) - exact_count) * actual_value_subvecs * code_bytes) / MB
                        )
                        v_mb_by_idx.append(exact_v_mb + v_pq_codebook_mb + compressed_v_codes_mb + metadata_mb)

                    max_exact_v_count = max(
                        [max(0, min(int(v_budget), int(query_context_len))) for v_budget in joint_v_budgets],
                        default=0,
                    )
                    k_cache: dict[int, tuple[torch.Tensor, float, torch.Tensor, torch.Tensor | None]] = {}
                    k_cache_by_selected_len: dict[int, tuple[torch.Tensor, float, torch.Tensor, torch.Tensor | None]] = {}
                    output_cache: dict[tuple[int, int], torch.Tensor] = {}

                    def k_artifacts(ki_i: int) -> tuple[torch.Tensor, float, torch.Tensor, torch.Tensor | None]:
                        cached = k_cache.get(int(ki_i))
                        if cached is not None:
                            return cached
                        selected_t_i = selected_for_budget_torch(int(joint_k_budgets[int(ki_i)]))
                        selected_len_i = int(selected_t_i.numel())
                        cached_by_len = k_cache_by_selected_len.get(selected_len_i)
                        if cached_by_len is not None:
                            k_cache[int(ki_i)] = cached_by_len
                            return cached_by_len
                        probs_t = mixed_probs_for_selected(selected_t_i)
                        exact_key_mb_i = float(int(selected_t_i.numel()) * int(self.head_dim) * key_bytes) / MB
                        risk_t = (probs_t * probs_t) * code_error_t.to(dtype=prob_dtype)
                        base_output_t = torch.sum(probs_t.to(torch.float32).reshape(-1, 1) * vhat_all_t.float(), dim=0)
                        prefix_delta_t: torch.Tensor | None = None
                        if int(max_exact_v_count) > 0:
                            if int(max_exact_v_count) >= int(query_context_len):
                                exact_order_t = torch.argsort(risk_t, descending=True, stable=True)
                            else:
                                exact_order_t = torch.topk(
                                    risk_t,
                                    k=int(max_exact_v_count),
                                    largest=True,
                                    sorted=True,
                                ).indices
                            exact_weighted_residuals_t = (
                                probs_t.index_select(0, exact_order_t).to(torch.float32).reshape(-1, 1)
                                * residual_t.index_select(0, exact_order_t).float()
                            )
                            prefix_delta_t = torch.cumsum(exact_weighted_residuals_t, dim=0)
                        out = (selected_t_i, float(selector_mb) + exact_key_mb_i, base_output_t, prefix_delta_t)
                        k_cache[int(ki_i)] = out
                        k_cache_by_selected_len[selected_len_i] = out
                        return out

                    def output_for_budget(ki_i: int, vi_i: int) -> torch.Tensor:
                        key = (int(ki_i), int(vi_i))
                        cached = output_cache.get(key)
                        if cached is not None:
                            return cached
                        _selected_t_i, _k_mb_i, base_output_t, prefix_delta_t = k_artifacts(int(ki_i))
                        exact_count = max(0, min(int(joint_v_budgets[int(vi_i)]), int(query_context_len)))
                        if exact_count > 0 and prefix_delta_t is not None:
                            out = base_output_t + prefix_delta_t[int(exact_count) - 1]
                        else:
                            out = base_output_t
                        output_cache[key] = out
                        return out

                    sim_t0 = 0.0
                    if bool(getattr(args, "profile_native_ops", False)):
                        _sync_if_cuda(device)
                        sim_t0 = time.perf_counter()
                    ki = 0
                    vi = 0
                    _steps = 0
                    _final_k_delta = 0.0
                    _final_v_delta = 0.0
                    threshold_value = float(getattr(args, "joint_kv_stability_threshold", 0.001))
                    while _steps < (len(joint_k_budgets) + len(joint_v_budgets) + 4):
                        cur_output_t = output_for_budget(int(ki), int(vi))
                        k_can = int(ki) + 1 < len(joint_k_budgets)
                        v_can = int(vi) + 1 < len(joint_v_budgets)
                        _final_k_delta = (
                            _rel_l2_torch(cur_output_t, output_for_budget(int(ki) + 1, int(vi))) if k_can else 0.0
                        )
                        _final_v_delta = (
                            _rel_l2_torch(cur_output_t, output_for_budget(int(ki), int(vi) + 1)) if v_can else 0.0
                        )
                        extra_k_mb = (
                            float(k_artifacts(int(ki) + 1)[1] - k_artifacts(int(ki))[1]) if k_can else float("inf")
                        )
                        extra_v_mb = (
                            float(v_mb_by_idx[int(vi) + 1] - v_mb_by_idx[int(vi)]) if v_can else float("inf")
                        )
                        action = _choose_joint_kv_action(
                            policy=policy_name,
                            k_delta=float(_final_k_delta),
                            v_delta=float(_final_v_delta),
                            k_can=bool(k_can),
                            v_can=bool(v_can),
                            threshold=threshold_value,
                            turn=int(_steps),
                            extra_k_mb=float(extra_k_mb),
                            extra_v_mb=float(extra_v_mb),
                        )
                        if action == "stop":
                            break
                        if action == "k":
                            ki += 1
                        elif action == "v":
                            vi += 1
                        else:
                            raise AssertionError(action)
                        _steps += 1
                    if bool(getattr(args, "profile_native_ops", False)):
                        _sync_if_cuda(device)
                        stats[layer_id].add_native_detail_timing(
                            geometric_seconds=float(time.perf_counter() - sim_t0)
                        )
                    selected_t = k_artifacts(int(ki))[0]
                    selected_cpu = selected_t.detach().cpu().numpy().astype(np.int64, copy=False)
                    exact_v_count = max(0, min(int(joint_v_budgets[int(vi)]), int(query_context_len)))
                    exact_key_mb = float(selected_cpu.size * int(self.head_dim) * key_bytes) / MB
                    exact_v_mb = float(exact_v_count * int(self.head_dim) * value_bytes) / MB
                    compressed_v_codes_mb = (
                        float(max(0, int(query_context_len) - exact_v_count) * actual_value_subvecs * code_bytes) / MB
                    )
                    tail_mb_override = float(v_pq_codebook_mb + compressed_v_codes_mb + metadata_mb)
                    dense_physical_key_mb = float(int(query_context_len) * int(self.head_dim) * key_bytes) / MB
                    stats[layer_id].add(
                        selected_cpu.tolist(),
                        max(0, int(query_context_len) - int(exact_v_count)),
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
                    return output_for_budget(int(ki), int(vi)).to(dtype=hidden_states.dtype, device=device)
                selected_cache: dict[bytes, tuple[torch.Tensor, np.ndarray, float, np.ndarray]] = {}

                def selected_output_for(selected_arr: np.ndarray) -> tuple[torch.Tensor, np.ndarray, float, np.ndarray]:
                    selected_arr = selected_arr.astype(np.int64, copy=False)
                    cache_key = selected_arr.tobytes()
                    cached = selected_cache.get(cache_key)
                    if cached is not None:
                        return cached
                    selected_t = torch.as_tensor(selected_arr, dtype=torch.long, device=device)
                    if selected_t.numel() == 0:
                        out_t = torch.zeros((int(self.head_dim),), dtype=torch.float32, device=device)
                        values_out = np.zeros((0, int(self.head_dim)), dtype=np.float32)
                        logits_np = np.zeros((0,), dtype=np.float32)
                        result = (out_t, values_out, 0.0, logits_np)
                        selected_cache[cache_key] = result
                        return result
                    selected_keys = torch_k_cache[kv_head].index_select(0, selected_t).float()
                    selected_logits = (selected_keys @ query) / math.sqrt(float(self.head_dim))
                    values_np = values_np_cache[kv_head]
                    selected_scores_np = selected_logits.detach().cpu().numpy().astype(np.float32, copy=False)
                    if str(args.selected_value_mode) == "vpq_value":
                        exact_values_np = values_np[selected_arr].astype(np.float32, copy=False)
                        exact_all_by_context = (
                            int(getattr(args, "selected_value_exact_all_context_max", 0)) > 0
                            and int(query_context_len) <= int(getattr(args, "selected_value_exact_all_context_max", 0))
                        )
                        exact_all_by_fraction = (
                            float(getattr(args, "selected_value_exact_all_fraction_min", 0.0)) > 0.0
                            and (float(selected_arr.size) / max(1.0, float(query_context_len)))
                            >= float(getattr(args, "selected_value_exact_all_fraction_min", 0.0))
                        )
                        if exact_all_by_context or exact_all_by_fraction:
                            exact_mask = np.ones((selected_arr.size,), dtype=bool)
                            selected_values_np = exact_values_np.astype(np.float32, copy=True)
                            compressed_v_mb = 0.0
                            fallback_v_mb = 0.0
                        elif str(args.selected_value_exact_rule) in {"selected_risk_mass", "selected_mass_or_risk"}:
                            approx_values_all, compressed_v_mb, fallback_v_mb = _vpq_values_for_tokens(
                                index=index,
                                values_np=values_np,
                                tokens=selected_arr.astype(np.int64, copy=False),
                                subbits=int(args.subbits),
                                value_subvecs=int(args.value_subvecs),
                                value_subbits=int(args.value_subbits),
                                value_bytes=value_bytes,
                            )
                            compressed_v_mb += float(
                                selected_arr.size * max(0, int(args.selected_value_residual_norm_bytes))
                            ) / MB
                            exact_mask, _exact_selected_mass = selected_value_exact_mask(
                                selected_scores=selected_scores_np,
                                values_exact=exact_values_np,
                                values_approx=approx_values_all,
                                rule=str(args.selected_value_exact_rule),
                                exact_top=int(args.selected_value_exact_top),
                                exact_mass=float(args.selected_value_exact_mass),
                                exact_risk_mass=float(args.selected_value_exact_risk_mass),
                                min_top=int(args.selected_value_min_exact_top),
                                max_top=int(args.selected_value_max_exact_top),
                            )
                            selected_values_np = approx_values_all.astype(np.float32, copy=True)
                            selected_values_np[exact_mask] = exact_values_np[exact_mask]
                        else:
                            exact_mask, _exact_selected_mass = selected_value_exact_mask(
                                selected_scores=selected_scores_np,
                                values_exact=exact_values_np,
                                values_approx=None,
                                rule=str(args.selected_value_exact_rule),
                                exact_top=int(args.selected_value_exact_top),
                                exact_mass=float(args.selected_value_exact_mass),
                                exact_risk_mass=float(args.selected_value_exact_risk_mass),
                                min_top=int(args.selected_value_min_exact_top),
                                max_top=int(args.selected_value_max_exact_top),
                            )
                            compressed_mask = ~exact_mask
                            selected_values_np = np.empty_like(exact_values_np)
                            selected_values_np[exact_mask] = exact_values_np[exact_mask]
                            compressed_v_mb = 0.0
                            fallback_v_mb = 0.0
                            if bool(np.any(compressed_mask)):
                                approx_values, compressed_v_mb, fallback_v_mb = _vpq_values_for_tokens(
                                    index=index,
                                    values_np=values_np,
                                    tokens=selected_arr[compressed_mask].astype(np.int64, copy=False),
                                    subbits=int(args.subbits),
                                    value_subvecs=int(args.value_subvecs),
                                    value_subbits=int(args.value_subbits),
                                    value_bytes=value_bytes,
                                )
                                selected_values_np[compressed_mask] = approx_values
                        exact_value_mb = float(np.sum(exact_mask) * int(self.head_dim) * value_bytes) / MB
                        selected_value_mb_local = float(compressed_v_mb) + float(fallback_v_mb) + exact_value_mb
                        selected_values = torch.as_tensor(selected_values_np, dtype=torch.float32, device=device)
                    else:
                        selected_values = torch_v_cache[kv_head].index_select(0, selected_t).float()
                        selected_values_np = selected_values.detach().float().cpu().numpy().astype(np.float32, copy=False)
                        selected_value_mb_local = float(selected_arr.size * int(self.head_dim) * value_bytes) / MB
                    selected_weights = torch.softmax(selected_logits.float(), dim=0).to(selected_values.dtype)
                    out_t = (selected_weights.unsqueeze(0) @ selected_values).squeeze(0).float()
                    result = (out_t, selected_values_np, float(selected_value_mb_local), selected_scores_np)
                    selected_cache[cache_key] = result
                    return result

                selected_cpu = _selected_for_budget(
                    base=base,
                    ranked_cpu=ranked_cpu,
                    budget=budget,
                    context_len=int(query_context_len),
                )
                confidence_mb = 0.0
                tail_confidence_pass = False
                final_tail_score_scale = 1.0
                final_tail_score_bias = 0.0
                if online_confidence_rule == "geometric_probe_tail_switch":
                    if str(args.tail_mode) not in {"pq_value", "vpq_value", "page_mean"}:
                        raise RuntimeError("geometric_probe_tail_switch requires a compressed tail mode")
                    max_budget = max(0, int(args.geometric_max_budget))
                    if max_budget <= 0:
                        max_budget = max(0, int(rank_budget))
                    granularity = max(1, int(args.geometric_budget_granularity))
                    growth = max(1.01, float(args.geometric_growth))
                    probe_scale = max(1.01, float(args.geometric_probe_scale))
                    k = _round_budget_up(
                        int(args.geometric_min_budget),
                        granularity=granularity,
                        max_budget=max_budget,
                    )
                    while True:
                        tail_budget = min(int(k), int(max_budget))
                        tail_selected = _selected_for_budget(
                            base=base,
                            ranked_cpu=ranked_cpu,
                            budget=tail_budget,
                            context_len=int(query_context_len),
                        )
                        tail_selected_only, tail_selected_values_np, tail_value_mb, tail_scores_np = selected_output_for(tail_selected)
                        scores_np = np.zeros((int(context_len),), dtype=np.float32)
                        if tail_selected.size:
                            scores_np[tail_selected] = tail_scores_np
                        tail_score_scale = 1.0
                        tail_score_bias = 0.0
                        tail_pq_relrmse = float("inf")
                        tail_pq_corr = 0.0
                        if str(args.tail_score_calibration) == "affine_selected":
                            (
                                tail_score_scale,
                                tail_score_bias,
                                _tail_calibration_tokens,
                                tail_pq_relrmse,
                                tail_pq_corr,
                                _tail_pq_residual_std,
                            ) = _fit_selected_pq_logit_uncertainty(
                                selected_cpu=tail_selected,
                                ranked_cpu=ranked_cpu,
                                ranked_scores_cpu=ranked_scores_cpu,
                                scores_np=scores_np,
                                query_dim=int(self.head_dim),
                            )
                        final_tail_score_scale = float(tail_score_scale)
                        final_tail_score_bias = float(tail_score_bias)
                        proxy_mass, proxy_tail_mass = _proxy_selected_mass(
                            selected_cpu=tail_selected,
                            ranked_cpu=ranked_cpu,
                            ranked_scores_cpu=ranked_scores_cpu,
                            scores_np=scores_np,
                            query_dim=int(self.head_dim),
                            tail_score_scale=tail_score_scale,
                            tail_score_bias=tail_score_bias,
                        )
                        approx_tail_np, _tail_count_candidate, _tail_population_candidate, tail_mb_candidate = _compressed_tail_output(
                            index=index,
                            values_np=values_np_cache[kv_head],
                            scores_np=scores_np,
                            ranked_cpu=ranked_cpu,
                            ranked_scores_cpu=ranked_scores_cpu,
                            selected_cpu=tail_selected,
                            query_dim=int(self.head_dim),
                            subbits=int(args.subbits),
                            value_bytes=value_bytes,
                            mode=str(args.tail_mode),
                            value_subvecs=int(args.value_subvecs),
                            value_subbits=int(args.value_subbits),
                            selected_values_np=tail_selected_values_np,
                            tail_score_scale=tail_score_scale,
                            tail_score_bias=tail_score_bias,
                        )
                        probe_budget = _round_budget_up(
                            max(float(tail_budget + granularity), probe_scale * float(tail_budget)),
                            granularity=granularity,
                            max_budget=max_budget,
                        )
                        probe_budget = max(tail_budget, int(probe_budget))
                        probe_selected = _selected_for_budget(
                            base=base,
                            ranked_cpu=ranked_cpu,
                            budget=probe_budget,
                            context_len=int(query_context_len),
                        )
                        probe_selected_only, _probe_values_np, probe_value_mb, _probe_scores_np = selected_output_for(probe_selected)
                        tail_probe_rel_l2 = float(
                            np.linalg.norm(
                                approx_tail_np.astype(np.float64, copy=False)
                                - probe_selected_only.detach().cpu().numpy().astype(np.float64, copy=False)
                            )
                        ) / max(float(torch.linalg.vector_norm(probe_selected_only.float()).item()), 1e-20)
                        # Charge the confidence check explicitly. This is conservative because
                        # the current Python path recomputes candidate outputs instead of retaining
                        # all fetched data across candidate budgets.
                        confidence_mb += float(tail_mb_candidate) + float(tail_value_mb) + float(probe_value_mb)
                        selected_cpu = probe_selected
                        budget = int(probe_budget)
                        tail_confidence_pass = (
                            tail_probe_rel_l2 <= float(args.tail_probe_rel_l2_max)
                            and float(proxy_mass) >= float(args.tail_proxy_mass_min)
                            and float(proxy_tail_mass) <= float(args.tail_proxy_mass_max)
                            and float(tail_pq_corr) >= float(args.tail_pq_corr_min)
                            and float(tail_pq_relrmse) <= float(args.tail_pq_relrmse_max)
                        )
                        if tail_confidence_pass or int(probe_budget) >= int(max_budget):
                            break
                        next_k = _round_budget_up(
                            max(float(probe_budget + granularity), growth * float(probe_budget)),
                            granularity=granularity,
                            max_budget=max_budget,
                        )
                        if int(next_k) <= int(probe_budget):
                            break
                        k = int(next_k)
                elif online_confidence_rule == "geometric_exact_delta":
                    max_budget = max(0, int(args.geometric_max_budget))
                    if max_budget <= 0:
                        max_budget = max(0, int(rank_budget))
                    granularity = max(1, int(args.geometric_budget_granularity))
                    growth = max(1.01, float(args.geometric_growth))
                    probe_scale = max(1.01, float(args.geometric_probe_scale))
                    k = _round_budget_up(
                        int(args.geometric_min_budget),
                        granularity=granularity,
                        max_budget=max_budget,
                    )
                    while True:
                        tail_budget = min(int(k), int(max_budget))
                        tail_selected = _selected_for_budget(
                            base=base,
                            ranked_cpu=ranked_cpu,
                            budget=tail_budget,
                            context_len=int(query_context_len),
                        )
                        tail_selected_only, _tail_values_np, tail_value_mb, _tail_scores_np = selected_output_for(tail_selected)
                        probe_budget = _round_budget_up(
                            max(float(tail_budget + granularity), probe_scale * float(tail_budget)),
                            granularity=granularity,
                            max_budget=max_budget,
                        )
                        probe_budget = max(tail_budget, int(probe_budget))
                        probe_selected = _selected_for_budget(
                            base=base,
                            ranked_cpu=ranked_cpu,
                            budget=probe_budget,
                            context_len=int(query_context_len),
                        )
                        probe_selected_only, _probe_values_np, probe_value_mb, _probe_scores_np = selected_output_for(probe_selected)
                        exact_delta_rel_l2 = float(
                            torch.linalg.vector_norm((probe_selected_only - tail_selected_only).float()).item()
                        ) / max(float(torch.linalg.vector_norm(probe_selected_only.float()).item()), 1e-20)
                        confidence_mb += float(tail_value_mb) + float(probe_value_mb)
                        selected_cpu = probe_selected
                        budget = int(probe_budget)
                        tail_confidence_pass = exact_delta_rel_l2 <= float(args.tail_probe_rel_l2_max)
                        if tail_confidence_pass or int(probe_budget) >= int(max_budget):
                            break
                        next_k = _round_budget_up(
                            max(float(probe_budget + granularity), growth * float(probe_budget)),
                            granularity=granularity,
                            max_budget=max_budget,
                        )
                        if int(next_k) <= int(probe_budget):
                            break
                        k = int(next_k)
                selected = torch.as_tensor(selected_cpu, dtype=torch.long, device=device)
                selected_only, selected_values_np, selected_value_mb, selected_scores_np = selected_output_for(selected_cpu)
                tail_blend = float(tail_blend_value)
                effective_tail_blend = 0.0 if int(head) in tail_off_heads else tail_blend
                if online_confidence_rule == "geometric_probe_tail_switch" and not tail_confidence_pass:
                    effective_tail_blend = 0.0
                tail_mb_override = None
                if effective_tail_blend <= 0.0:
                    approx_head = selected_only
                    tail_count = 0
                elif str(args.tail_mode) in {"pq_value", "vpq_value", "page_mean"}:
                    scores_np = np.zeros((context_len,), dtype=np.float32)
                    if selected_cpu.size:
                        scores_np[selected_cpu] = selected_scores_np
                    values_np = values_np_cache[kv_head]
                    approx_tail_np, tail_count, _tail_population, tail_mb_override = _compressed_tail_output(
                        index=index,
                        values_np=values_np,
                        scores_np=scores_np,
                        ranked_cpu=ranked_cpu,
                        ranked_scores_cpu=ranked_scores_cpu,
                        selected_cpu=selected_cpu,
                        query_dim=int(self.head_dim),
                        subbits=int(args.subbits),
                        value_bytes=value_bytes,
                        mode=str(args.tail_mode),
                        value_subvecs=int(args.value_subvecs),
                        value_subbits=int(args.value_subbits),
                        selected_values_np=selected_values_np,
                        tail_score_scale=final_tail_score_scale,
                        tail_score_bias=final_tail_score_bias,
                    )
                    tail_probe_rel_l2 = float(
                        np.linalg.norm(approx_tail_np.astype(np.float64, copy=False) - selected_only.detach().cpu().numpy().astype(np.float64, copy=False))
                    ) / max(float(torch.linalg.vector_norm(selected_only.float()).item()), 1e-20)
                    approx_tail = torch.as_tensor(approx_tail_np, dtype=torch.float32, device=device)
                    if tail_probe_rel_l2 > float(args.tail_probe_rel_l2_max):
                        approx_head = selected_only
                    else:
                        approx_head = (
                            approx_tail
                            if effective_tail_blend >= 1.0
                            else selected_only + effective_tail_blend * (approx_tail.float() - selected_only)
                        )
                else:
                    approx_tail, tail_count, _tail_population, _attn_seconds = selected_plus_tail_output(
                        torch_k_cache[kv_head].float(),
                        torch_v_cache[kv_head].float(),
                        query,
                        selected,
                        ranked_cpu,
                        np.zeros((int(query_context_len),), dtype=np.float32),
                        context_len=int(query_context_len),
                        samples=int(args.tail_samples),
                        bands=int(args.tail_bands),
                        seed=int(args.tail_seed),
                        qidx=int(query_context_len),
                        head=head,
                        sampling=str(args.tail_sampling),
                    )
                    approx_head = (
                        approx_tail
                        if effective_tail_blend >= 1.0
                        else selected_only + effective_tail_blend * (approx_tail.float() - selected_only)
                    )
                stats[layer_id].add(
                    selected_cpu.tolist(),
                    tail_count,
                    float(selector_mb) + float(rerank_key_mb),
                    int(self.head_dim),
                    key_bytes,
                    value_bytes,
                    tail_mb_override=tail_mb_override,
                    exact_kv_mb_override=float(selected_cpu.size * int(self.head_dim) * key_bytes) / MB + float(selected_value_mb),
                    confidence_mb_override=float(confidence_mb),
                )
                return approx_head.to(hidden_states.dtype)

            def approximate_joint_kv_all_heads(
                local_qpos: int,
                query_context_len: int,
            ) -> torch.Tensor | None:
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

                k_budget_text = str(getattr(args, "joint_kv_k_budgets", ""))
                v_budget_text = str(getattr(args, "joint_kv_v_budgets", ""))
                budget_cache_key = (
                    k_budget_text,
                    v_budget_text,
                    str(device.type),
                    int(device.index) if device.index is not None else -1,
                )
                budget_cache = getattr(args, "_pagedpq_joint_budget_cache", None)
                if not isinstance(budget_cache, dict):
                    budget_cache = {}
                    setattr(args, "_pagedpq_joint_budget_cache", budget_cache)
                cached_budgets = budget_cache.get(budget_cache_key)
                if cached_budgets is None:
                    parsed_k_budgets = _parse_budget_schedule(k_budget_text, name="joint_kv_k_budgets")
                    parsed_v_budgets = _parse_budget_schedule(v_budget_text, name="joint_kv_v_budgets")
                    cached_budgets = (
                        tuple(int(v) for v in parsed_k_budgets),
                        tuple(int(v) for v in parsed_v_budgets),
                        torch.as_tensor(parsed_v_budgets, dtype=torch.long, device=device),
                    )
                    budget_cache[budget_cache_key] = cached_budgets
                joint_k_budgets = list(cached_budgets[0])
                joint_v_budgets = list(cached_budgets[1])
                joint_v_budgets_t = cached_budgets[2]
                if not joint_k_budgets or not joint_v_budgets:
                    return None

                context_len_i = int(query_context_len)
                if _env_truthy("SELECTOR_PQ_JOINT_COLLAPSE_DUP_V_ROWS", "0"):
                    collapsed_v_budgets: list[int] = []
                    seen_v_counts: set[int] = set()
                    for v_budget in joint_v_budgets:
                        exact_count_i = max(0, min(int(v_budget), context_len_i))
                        if int(exact_count_i) in seen_v_counts:
                            continue
                        seen_v_counts.add(int(exact_count_i))
                        collapsed_v_budgets.append(int(v_budget))
                    if collapsed_v_budgets:
                        joint_v_budgets = collapsed_v_budgets
                        joint_v_budgets_t = torch.as_tensor(
                            joint_v_budgets,
                            dtype=torch.long,
                            device=device,
                        )
                sqrt_dim = float(math.sqrt(float(self.head_dim)))
                prob_dtype = torch.float32 if _env_truthy("SELECTOR_PQ_JOINT_FP32_PROBS", "1") else torch.float64
                outputs_all = torch.empty((num_heads, int(self.head_dim)), dtype=torch.float32, device=device)
                use_grouped_risk_prefix = (
                    _env_truthy("SELECTOR_PQ_JOINT_GROUPED_RISK_PREFIX", "0")
                    and _env_truthy("SELECTOR_PQ_JOINT_GRID_ARTIFACTS", "1")
                    and _env_truthy("SELECTOR_PQ_JOINT_NATIVE_RISK_PREFIX", "0")
                    and not _env_truthy("SELECTOR_PQ_JOINT_INCREMENTAL_V_GRID", "0")
                    and not _env_truthy("SELECTOR_PQ_JOINT_ONDEMAND_V_PREFIX", "0")
                )
                grouped_risk_records: list[dict[str, object]] = []
                grouped_geo_t0 = 0.0
                joint_total_wall_t0 = time.perf_counter() if wall_profile_enabled else 0.0
                allhead_precompute = _env_truthy("SELECTOR_PQ_JOINT_ALLHEAD_PRECOMPUTE", "1")
                allhead_exact_precompute = _env_truthy("SELECTOR_PQ_JOINT_ALLHEAD_EXACT_PRECOMPUTE", "0")
                allhead_indexes: list[GPUIndex] | None = None
                allhead_dense_pq_scores_t: torch.Tensor | None = None
                allhead_selector_mb: float | None = None
                allhead_exact_scores_t: torch.Tensor | None = None
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
                                0,
                            )
                            allhead_dense_pq_scores_t = allhead_dense_pq_scores_t.to(device=device, dtype=torch.float32)
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

                token_layout_cache: dict[
                    tuple[object, ...],
                    tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, bool],
                ] = {}
                allhead_rank_prefix_cache: dict[tuple[int, int, int, int], torch.Tensor] = {}

                def grouped_vpq_residual_sidecars_for(
                    gqa_indexes: list[GPUIndex],
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int] | None:
                    if not _env_truthy("SELECTOR_PQ_JOINT_GROUPED_VPQ_CACHE", "0"):
                        return None
                    if len(gqa_indexes) != int(num_kv_heads):
                        return None
                    group_key = (
                        tuple(
                            joint_vpq_cache_key_for(
                                int(kv_head),
                                torch_v_cache[int(kv_head)][:context_len_i],
                                gqa_indexes[int(kv_head)],
                            )
                            for kv_head in range(int(num_kv_heads))
                        ),
                        int(num_kv_heads),
                    )
                    grouped_cache = getattr(self, "_pagedpq_joint_grouped_vpq_sidecar_cache", None)
                    if not isinstance(grouped_cache, dict):
                        grouped_cache = {}
                        setattr(self, "_pagedpq_joint_grouped_vpq_sidecar_cache", grouped_cache)
                    cached = grouped_cache.get(group_key)
                    if cached is not None:
                        if len(cached) != 6:
                            grouped_cache.pop(group_key, None)
                        else:
                            (
                                cached_len,
                                cached_capacity,
                                vhat_groups_t,
                                residual_groups_t,
                                code_error_groups_t,
                                cached_subbits,
                            ) = cached
                            cached_capacity_i = int(cached_capacity)
                            cached_len_i = int(cached_len)
                            if cached_len_i >= int(context_len_i):
                                return (
                                    vhat_groups_t[:, :context_len_i, :],
                                    residual_groups_t[:, :context_len_i, :],
                                    code_error_groups_t[:, :context_len_i],
                                    int(cached_subbits),
                                )
                            if cached_len_i >= 0 and cached_len_i < int(context_len_i):
                                context_len_target_i = int(context_len_i)
                                if cached_capacity_i < context_len_target_i:
                                    grow_pad_i = max(
                                        0,
                                        _env_int("SELECTOR_PQ_JOINT_PERSISTENT_VPQ_CACHE_GROW_PAD", 256),
                                    )
                                    new_capacity_i = max(
                                        context_len_target_i,
                                        cached_capacity_i + max(grow_pad_i, context_len_target_i - cached_capacity_i),
                                    )
                                    vhat_buf_t = torch.empty(
                                        (
                                            int(num_kv_heads),
                                            int(new_capacity_i),
                                            int(vhat_groups_t.shape[2]),
                                        ),
                                        dtype=vhat_groups_t.dtype,
                                        device=vhat_groups_t.device,
                                    )
                                    residual_buf_t = torch.empty(
                                        (
                                            int(num_kv_heads),
                                            int(new_capacity_i),
                                            int(residual_groups_t.shape[2]),
                                        ),
                                        dtype=residual_groups_t.dtype,
                                        device=residual_groups_t.device,
                                    )
                                    code_error_buf_t = torch.empty(
                                        (int(num_kv_heads), int(new_capacity_i)),
                                        dtype=code_error_groups_t.dtype,
                                        device=code_error_groups_t.device,
                                    )
                                    if cached_len_i > 0:
                                        vhat_buf_t[:, :cached_len_i, :].copy_(vhat_groups_t[:, :cached_len_i, :])
                                        residual_buf_t[:, :cached_len_i, :].copy_(residual_groups_t[:, :cached_len_i, :])
                                        code_error_buf_t[:, :cached_len_i].copy_(code_error_groups_t[:, :cached_len_i])
                                    vhat_groups_t = vhat_buf_t
                                    residual_groups_t = residual_buf_t
                                    code_error_groups_t = code_error_buf_t
                                    cached_capacity_i = int(new_capacity_i)
                                for kv_head in range(int(num_kv_heads)):
                                    extra_values = torch_v_cache[int(kv_head)][cached_len_i:context_len_target_i]
                                    if int(extra_values.numel()) > 0:
                                        vhat_groups_t[
                                            int(kv_head),
                                            cached_len_i:context_len_target_i,
                                            :,
                                        ].copy_(extra_values.to(dtype=vhat_groups_t.dtype))
                                residual_groups_t[:, cached_len_i:context_len_target_i, :].zero_()
                                code_error_groups_t[:, cached_len_i:context_len_target_i].zero_()
                                grouped_cache[group_key] = (
                                    int(context_len_target_i),
                                    int(cached_capacity_i),
                                    vhat_groups_t,
                                    residual_groups_t,
                                    code_error_groups_t,
                                    int(cached_subbits),
                                )
                                return (
                                    vhat_groups_t[:, :context_len_target_i, :],
                                    residual_groups_t[:, :context_len_target_i, :],
                                    code_error_groups_t[:, :context_len_target_i],
                                    int(cached_subbits),
                                )

                    vhat_parts: list[torch.Tensor] = []
                    residual_parts: list[torch.Tensor] = []
                    code_error_parts: list[torch.Tensor] = []
                    actual_subbits = int(args.value_subbits) if int(args.value_subbits) > 0 else int(args.subbits)
                    for kv_head in range(int(num_kv_heads)):
                        _vhat_t, residual_t, code_error_t, actual_subbits_i = joint_vpq_sidecars_for(
                            kv_head=int(kv_head),
                            index=gqa_indexes[int(kv_head)],
                            values_t=torch_v_cache[int(kv_head)][:context_len_i],
                            context_len_i=context_len_i,
                        )
                        vhat_parts.append(_vhat_t.to(dtype=torch.float32))
                        residual_parts.append(residual_t.to(dtype=torch.float32))
                        code_error_parts.append(code_error_t.to(dtype=torch.float32))
                        actual_subbits = int(actual_subbits_i)
                    vhat_groups_t = torch.stack(vhat_parts, dim=0).contiguous()
                    residual_groups_t = torch.stack(residual_parts, dim=0).contiguous()
                    code_error_groups_t = torch.stack(code_error_parts, dim=0).contiguous()
                    grow_pad_i = max(
                        0,
                        _env_int("SELECTOR_PQ_JOINT_PERSISTENT_VPQ_CACHE_GROW_PAD", 256),
                    )
                    cache_capacity_i = int(context_len_i + grow_pad_i)
                    if cache_capacity_i > int(context_len_i):
                        vhat_buf_t = torch.empty(
                            (
                                int(num_kv_heads),
                                int(cache_capacity_i),
                                int(vhat_groups_t.shape[2]),
                            ),
                            dtype=vhat_groups_t.dtype,
                            device=vhat_groups_t.device,
                        )
                        residual_buf_t = torch.empty(
                            (
                                int(num_kv_heads),
                                int(cache_capacity_i),
                                int(residual_groups_t.shape[2]),
                            ),
                            dtype=residual_groups_t.dtype,
                            device=residual_groups_t.device,
                        )
                        code_error_buf_t = torch.empty(
                            (int(num_kv_heads), int(cache_capacity_i)),
                            dtype=code_error_groups_t.dtype,
                            device=code_error_groups_t.device,
                        )
                        vhat_buf_t[:, :context_len_i, :].copy_(vhat_groups_t)
                        residual_buf_t[:, :context_len_i, :].copy_(residual_groups_t)
                        code_error_buf_t[:, :context_len_i].copy_(code_error_groups_t)
                        vhat_groups_t = vhat_buf_t
                        residual_groups_t = residual_buf_t
                        code_error_groups_t = code_error_buf_t
                    if grouped_cache:
                        grouped_cache.clear()
                    grouped_cache[group_key] = (
                        int(context_len_i),
                        int(cache_capacity_i),
                        vhat_groups_t,
                        residual_groups_t,
                        code_error_groups_t,
                        int(actual_subbits),
                    )
                    return (
                        vhat_groups_t[:, :context_len_i, :],
                        residual_groups_t[:, :context_len_i, :],
                        code_error_groups_t[:, :context_len_i],
                        int(actual_subbits),
                    )

                def token_layout_for(index: GPUIndex) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, bool]:
                    page_ranges = tuple(
                        (
                            int(page.start),
                            min(int(page.start) + int(page.size), context_len_i),
                        )
                        for page in index.pages
                        if int(page.start) < context_len_i and int(page.size) > 0
                    )
                    layout_key = (
                        int(context_len_i),
                        int(index.pending_start),
                        int(index.indexed_end),
                        int(args.static_prefix),
                        int(args.static_suffix),
                        page_ranges,
                    )
                    cached_layout = token_layout_cache.get(layout_key)
                    if cached_layout is not None:
                        return cached_layout

                    prefix_end_i = min(max(0, int(args.static_prefix)), context_len_i)
                    suffix_start_i = max(0, context_len_i - max(0, int(args.static_suffix)))
                    sealed_end_i = max((end for _, end in page_ranges), default=prefix_end_i)
                    pages_contiguous = True
                    expected_start_i = prefix_end_i
                    for start, end in page_ranges:
                        if int(start) != int(expected_start_i) or int(end) < int(start):
                            pages_contiguous = False
                            break
                        expected_start_i = int(end)
                    pages_contiguous = pages_contiguous and int(expected_start_i) == int(sealed_end_i)
                    indexed_end_i = max(0, min(int(index.indexed_end), context_len_i))
                    pending_start_i = max(prefix_end_i, int(index.pending_start))
                    pending_end_i = max(pending_start_i, min(indexed_end_i, suffix_start_i))

                    if (
                        _env_truthy("SELECTOR_PQ_JOINT_FAST_TOKEN_LAYOUT", "0")
                        and pages_contiguous
                        and int(index.pending_start) == int(sealed_end_i)
                        and int(sealed_end_i) <= int(suffix_start_i)
                        and int(indexed_end_i) >= int(suffix_start_i)
                    ):
                        # Canonical fullscan pages are contiguous. Avoid building
                        # thousands of Python token IDs every decode step; preserve
                        # unique_tokens(static_tokens + pending) order:
                        # prefix, suffix, then pending.
                        indexed_tokens_layout_t = (
                            torch.arange(prefix_end_i, sealed_end_i, dtype=torch.long, device=device)
                            if sealed_end_i > prefix_end_i
                            else torch.empty((0,), dtype=torch.long, device=device)
                        )
                        base_parts_t = []
                        if prefix_end_i > 0:
                            base_parts_t.append(torch.arange(0, prefix_end_i, dtype=torch.long, device=device))
                        if context_len_i > suffix_start_i:
                            base_parts_t.append(torch.arange(suffix_start_i, context_len_i, dtype=torch.long, device=device))
                        if pending_end_i > pending_start_i:
                            base_parts_t.append(torch.arange(pending_start_i, pending_end_i, dtype=torch.long, device=device))
                        base_layout_t = (
                            torch.cat(base_parts_t)
                            if base_parts_t
                            else torch.empty((0,), dtype=torch.long, device=device)
                        )
                        out = (indexed_tokens_layout_t, base_layout_t, None, True)
                        token_layout_cache[layout_key] = out
                        return out

                    token_parts = [
                        torch.arange(start, end, dtype=torch.long, device=device)
                        for start, end in page_ranges
                        if end > start
                    ]
                    indexed_tokens_layout_t = (
                        torch.cat(token_parts)
                        if token_parts
                        else torch.empty((0,), dtype=torch.long, device=device)
                    )

                    pending_layout = list(
                        range(
                            max(0, int(index.pending_start)),
                            max(0, min(int(index.indexed_end), context_len_i)),
                        )
                    )
                    base_layout = unique_tokens(
                        static_tokens(context_len_i - 1, int(args.static_prefix), int(args.static_suffix))
                        + pending_layout,
                        context_len=context_len_i,
                    )
                    coverage_intervals: list[tuple[int, int]] = [
                        (max(0, int(start)), min(context_len_i, int(end)))
                        for start, end in page_ranges
                        if int(end) > int(start)
                    ]
                    base_tokens_sorted = sorted(
                        int(token)
                        for token in base_layout
                        if 0 <= int(token) < context_len_i
                    )
                    if base_tokens_sorted:
                        run_start = base_tokens_sorted[0]
                        prev = base_tokens_sorted[0]
                        for token in base_tokens_sorted[1:]:
                            if token == prev + 1:
                                prev = token
                                continue
                            coverage_intervals.append((run_start, prev + 1))
                            run_start = token
                            prev = token
                        coverage_intervals.append((run_start, prev + 1))
                    coverage_end = 0
                    layout_covers_context = context_len_i <= 0
                    for start, end in sorted(coverage_intervals):
                        if end <= coverage_end:
                            continue
                        if start > coverage_end:
                            break
                        coverage_end = max(coverage_end, end)
                        if coverage_end >= context_len_i:
                            layout_covers_context = True
                            break
                    base_layout_t = torch.as_tensor(base_layout, dtype=torch.long, device=device)
                    if int(base_layout_t.numel()) > 0:
                        base_layout_t = base_layout_t[(base_layout_t >= 0) & (base_layout_t < context_len_i)]

                    indexed_end_without_suffix = context_len_i - max(0, int(args.static_suffix))
                    nonbase_all = all(
                        start >= min(max(0, int(args.static_prefix)), context_len_i)
                        and end <= int(index.pending_start)
                        and end <= indexed_end_without_suffix
                        for start, end in page_ranges
                    )
                    if nonbase_all:
                        nonbase_mask_layout_t = None
                    else:
                        base_rank_mask_t = torch.zeros((context_len_i,), dtype=torch.bool, device=device)
                        if int(base_layout_t.numel()) > 0:
                            base_rank_mask_t.index_fill_(0, base_layout_t, True)
                        nonbase_mask_layout_t = ~base_rank_mask_t.index_select(0, indexed_tokens_layout_t)

                    out = (indexed_tokens_layout_t, base_layout_t, nonbase_mask_layout_t, bool(layout_covers_context))
                    token_layout_cache[layout_key] = out
                    return out

                grouped_vpq_vhat_groups_t: torch.Tensor | None = None
                grouped_vpq_residual_groups_t: torch.Tensor | None = None
                grouped_vpq_code_error_groups_t: torch.Tensor | None = None
                grouped_vpq_actual_subbits: int | None = None
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
                        grouped_vpq = grouped_vpq_residual_sidecars_for(gqa_indexes_for_grouped)
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

                for kv_head_i in range(num_kv_heads):
                    index = (
                        allhead_indexes[int(kv_head_i)]
                        if allhead_indexes is not None
                        else prefix_index_for(int(kv_head_i), context_len_i)
                    )
                    if not index.pages:
                        return None
                    head_start_i = int(kv_head_i) * int(group_size)
                    head_end_i = min(int(num_heads), head_start_i + int(group_size))
                    if head_start_i >= head_end_i:
                        continue
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
                        return None

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

                    if allhead_exact_scores_t is not None and int(allhead_exact_scores_t.shape[1]) >= context_len_i:
                        exact_scores_h = allhead_exact_scores_t[head_start_i:head_end_i, :context_len_i]
                    else:
                        exact_wall_t0 = time.perf_counter() if wall_profile_enabled else 0.0
                        if bool(getattr(args, "profile_native_ops", False)):
                            _sync_if_cuda(device)
                            exact_t0 = time.perf_counter()
                        else:
                            exact_t0 = 0.0
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
                    exact_scores_prob_t = exact_scores_h.to(dtype=prob_dtype)
                    pq_logits_t = dense_score_rows_t.to(dtype=prob_dtype) / sqrt_dim
                    y_indexed_prob_t = (
                        exact_scores_h.index_select(1, indexed_tokens_t).to(prob_dtype)
                        if str(args.tail_score_calibration) == "affine_selected"
                        else None
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
                        # Diagnostic only: a K budget that reaches every non-base token is
                        # semantically a dense exact-logit row, but skipping the full sort can
                        # perturb tie handling. Keep canonical default on the conservative path.
                        partial_rank_takes = [
                            max(0, min(int(v), ranked_nonbase_count))
                            for v in active_joint_k_budgets
                            if max(0, min(int(v), ranked_nonbase_count)) < ranked_nonbase_count
                        ]
                        max_rank_take = max(partial_rank_takes, default=0)
                    else:
                        max_rank_take = max(
                            0,
                            min(max(int(v) for v in active_joint_k_budgets), ranked_nonbase_count),
                        )
                    ranked_prefix_tokens_t: torch.Tensor | None = None
                    if (
                        max_rank_take > 0
                        and _env_truthy("SELECTOR_PQ_JOINT_REUSE_MAX_TOPK", "1")
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
                                rank_prefix_wall_t0 = time.perf_counter() if wall_profile_enabled else 0.0
                                if bool(getattr(args, "profile_native_ops", False)):
                                    _sync_if_cuda(device)
                                    rank_prefix_t0 = time.perf_counter()
                                else:
                                    rank_prefix_t0 = 0.0
                                if _env_truthy("SELECTOR_PQ_JOINT_NATIVE_RANK_PREFIX", "0"):
                                    native = load_selector_paged_pq_ext()
                                    allhead_ranked_prefix_t = native.joint_rank_prefix_tokens(
                                        allhead_dense_pq_scores_t[:num_heads, :indexed_count]
                                        .to(dtype=torch.float32)
                                        .contiguous(),
                                        indexed_tokens_t.to(dtype=torch.long).contiguous(),
                                        int(max_rank_take),
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
                                allhead_rank_prefix_cache[rank_prefix_key] = allhead_ranked_prefix_t
                                if bool(getattr(args, "profile_native_ops", False)):
                                    _sync_if_cuda(device)
                                    stats[layer_id].add_joint_detail_timing(
                                        rank_prefix_seconds=float(time.perf_counter() - rank_prefix_t0)
                                    )
                                if wall_profile_enabled:
                                    stats[layer_id].add_joint_wall_timing(
                                        rank_prefix_seconds=float(time.perf_counter() - rank_prefix_wall_t0)
                                    )
                            ranked_prefix_tokens_t = allhead_ranked_prefix_t[head_start_i:head_end_i]
                        else:
                            rank_prefix_wall_t0 = time.perf_counter() if wall_profile_enabled else 0.0
                            if bool(getattr(args, "profile_native_ops", False)):
                                _sync_if_cuda(device)
                                rank_prefix_t0 = time.perf_counter()
                            else:
                                rank_prefix_t0 = 0.0
                            if _env_truthy("SELECTOR_PQ_JOINT_NATIVE_RANK_PREFIX", "0"):
                                native = load_selector_paged_pq_ext()
                                ranked_prefix_tokens_t = native.joint_rank_prefix_tokens(
                                    ranked_nonbase_scores_t.to(dtype=torch.float32).contiguous(),
                                    ranked_nonbase_t.to(dtype=torch.long).contiguous(),
                                    int(max_rank_take),
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
                            order_t = torch.topk(
                                ranked_nonbase_scores_t,
                                k=int(take),
                                dim=1,
                                largest=True,
                                sorted=True,
                            ).indices
                            add_t = ranked_nonbase_t.index_select(0, order_t.reshape(-1)).reshape(group_heads_i, take)
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
                            and selected_t_i.ndim == 2
                            and prob_dtype == torch.float32
                        ):
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
                        score_vec = exact_scores_prob_t.clone()
                        pq_logits = pq_logits_t
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

                    actual_value_subbits = int(actual_value_subbits_for_cost)
                    actual_value_subvecs = int(args.value_subvecs) if int(args.value_subvecs) > 0 else int(args.subvecs)
                    code_bytes = 1 if int(actual_value_subbits) <= 8 else 2
                    metadata_mb = (
                        float(context_len_i * actual_value_subvecs * code_bytes)
                        + float(
                            len(index.pages)
                            * actual_value_subvecs
                            * (1 << int(actual_value_subbits))
                            * int(getattr(args, "value_code_stat_bytes", getattr(args, "selected_value_residual_norm_bytes", 2)))
                        )
                    ) / MB
                    v_pq_codebook_mb = float(
                        len(index.pages)
                        * actual_value_subvecs
                        * (1 << int(actual_value_subbits))
                        * (int(self.head_dim) // max(1, actual_value_subvecs))
                        * value_bytes
                    ) / MB
                    v_mb_by_idx: list[float] | None = None
                    if needs_budget_mb_vectors:
                        v_mb_by_idx = []
                        for v_budget in joint_v_budgets:
                            exact_count = max(0, min(int(v_budget), context_len_i))
                            exact_v_mb = float(exact_count * int(self.head_dim) * value_bytes) / MB
                            compressed_v_codes_mb = (
                                float(max(0, context_len_i - exact_count) * actual_value_subvecs * code_bytes) / MB
                            )
                            v_mb_by_idx.append(exact_v_mb + v_pq_codebook_mb + compressed_v_codes_mb + metadata_mb)

                    max_exact_v_count = max(
                        [max(0, min(int(v_budget), context_len_i)) for v_budget in joint_v_budgets],
                        default=0,
                    )
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
                        native_score_grid_enabled = _env_truthy("SELECTOR_PQ_JOINT_NATIVE_SCORE_GRID", "0")
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
                                fused_score_fn_name = (
                                    "joint_mixed_softmax_base_outputs_rankpos"
                                    if use_rankpos_score_grid
                                    else "joint_mixed_softmax_base_outputs"
                                )
                                if not hasattr(native, fused_score_fn_name):
                                    raise RuntimeError(
                                        f"SELECTOR_PQ_JOINT_FUSED_SOFTMAX_BASE requires updated CUDA extension: {fused_score_fn_name}"
                                    )
                                probs_grid_t, base_output_grid_t = getattr(native, fused_score_fn_name)(
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
                                score_grid_t = None
                            else:
                                use_rankpos_score_grid = _env_truthy("SELECTOR_PQ_JOINT_RANKPOS_SCORE_GRID", "0")
                                if use_rankpos_score_grid and use_score_grid_no_fill:
                                    raise RuntimeError(
                                        "SELECTOR_PQ_JOINT_RANKPOS_SCORE_GRID does not support no-fill diagnostic mode"
                                    )
                                if use_rankpos_score_grid:
                                    if not hasattr(native, "joint_mixed_score_grid_rankpos"):
                                        raise RuntimeError(
                                            "SELECTOR_PQ_JOINT_RANKPOS_SCORE_GRID requires updated CUDA extension"
                                        )
                                    score_grid_fn = native.joint_mixed_score_grid_rankpos
                                else:
                                    score_grid_fn = (
                                        getattr(native, "joint_mixed_score_grid_no_exact_fill")
                                        if use_score_grid_no_fill
                                        and hasattr(native, "joint_mixed_score_grid_no_exact_fill")
                                        else native.joint_mixed_score_grid
                                    )
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
                        if probs_grid_t is None and _env_truthy("SELECTOR_PQ_JOINT_NATIVE_SOFTMAX_BASE", "0"):
                            native = load_selector_paged_pq_ext()
                            if not hasattr(native, "joint_softmax_base_outputs"):
                                raise RuntimeError(
                                    "SELECTOR_PQ_JOINT_NATIVE_SOFTMAX_BASE requires updated CUDA extension"
                                )
                            if score_grid_t is None:
                                raise RuntimeError("missing score grid for native softmax/base")
                            probs_grid_t, base_output_grid_t = native.joint_softmax_base_outputs(
                                score_grid_t.to(dtype=torch.float32).contiguous(),
                                vhat_all_t.to(dtype=torch.float32).contiguous(),
                            )
                        elif probs_grid_t is None:
                            if score_grid_t is None:
                                raise RuntimeError("missing score grid for Torch softmax/base")
                            probs_grid_t = torch.softmax(score_grid_t, dim=2)
                        if base_output_grid_t is None and _env_truthy("SELECTOR_PQ_JOINT_NATIVE_VPQ_BASE", "0"):
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
                        if base_output_grid_t is None:
                            base_output_grid_t = (
                                probs_grid_t.to(torch.float32).reshape(k_count_i * group_heads_i, context_len_i)
                                @ vhat_all_t.float()
                            ).reshape(k_count_i, group_heads_i, int(self.head_dim))
                        if bool(getattr(args, "profile_native_ops", False)):
                            _sync_if_cuda(device)
                            stats[layer_id].add_joint_detail_timing(
                                prob_base_seconds=float(time.perf_counter() - joint_prob_t0)
                            )
                            joint_risk_t0 = time.perf_counter()
                        else:
                            joint_risk_t0 = 0.0
                        if wall_profile_enabled:
                            stats[layer_id].add_joint_wall_timing(
                                prob_base_seconds=float(time.perf_counter() - joint_prob_wall_t0)
                            )
                        if use_grouped_risk_prefix:
                            if grouped_geo_t0 == 0.0 and bool(getattr(args, "profile_native_ops", False)):
                                grouped_geo_t0 = sim_t0
                            grouped_risk_records.append(
                                {
                                    "head_start": int(head_start_i),
                                    "head_end": int(head_end_i),
                                    "group_heads": int(group_heads_i),
                                    "context_len": int(context_len_i),
                                    "selector_mb": float(selector_mb),
                                    "v_pq_codebook_mb": float(v_pq_codebook_mb),
                                    "actual_value_subvecs": int(actual_value_subvecs),
                                    "grid_selected_by_ki": grid_selected_by_ki,
                                    "grid_selected_counts_by_ki": grid_selected_counts_by_ki,
                                    "grid_k_mb_by_idx": grid_k_mb_by_idx,
                                    "v_mb_by_idx": v_mb_by_idx,
                                    "base_output_grid": base_output_grid_t.to(dtype=torch.float32).contiguous(),
                                    "probs_grid": probs_grid_t.to(dtype=torch.float32).contiguous(),
                                    "residual": residual_t.to(dtype=torch.float32).contiguous(),
                                    "code_error": code_error_t.to(dtype=torch.float32).contiguous(),
                                }
                            )
                            continue
                        prefix_delta_grid_t: torch.Tensor | None = None
                        prefix_delta_by_count: dict[int, torch.Tensor] | None = None
                        joint_risk_wall_t0 = time.perf_counter() if wall_profile_enabled else 0.0
                        if use_incremental_v_grid:
                            incremental_grid_outputs_by_v_idx = {}
                            risk_grid_incremental_t = (
                                (probs_grid_t * probs_grid_t) * code_error_t.to(dtype=prob_dtype).reshape(1, 1, -1)
                                if int(max_exact_v_count) > 0
                                else None
                            )

                            def grid_outputs_for_v_idx_fn(vi_i: int) -> torch.Tensor:
                                cached_v = incremental_grid_outputs_by_v_idx.get(int(vi_i))
                                if cached_v is not None:
                                    return cached_v
                                exact_count = max(
                                    0,
                                    min(
                                        int(joint_v_budgets[int(vi_i)]),
                                        context_len_i,
                                        int(max_exact_v_count),
                                    ),
                                )
                                if exact_count <= 0:
                                    out_v = base_output_grid_t
                                elif exact_count >= context_len_i:
                                    delta_t = (
                                        probs_grid_t.to(torch.float32).reshape(k_count_i * group_heads_i, context_len_i)
                                        @ residual_t.float()
                                    ).reshape(k_count_i, group_heads_i, int(self.head_dim))
                                    out_v = base_output_grid_t + delta_t
                                else:
                                    if risk_grid_incremental_t is None:
                                        raise RuntimeError("missing residual-risk grid for incremental V-grid")
                                    exact_order_local_t = torch.topk(
                                        risk_grid_incremental_t,
                                        k=int(exact_count),
                                        dim=2,
                                        largest=True,
                                        sorted=True,
                                    ).indices
                                    gathered_probs_local_t = torch.gather(
                                        probs_grid_t.to(torch.float32),
                                        2,
                                        exact_order_local_t,
                                    )
                                    gathered_residual_local_t = residual_t.index_select(
                                        0,
                                        exact_order_local_t.reshape(-1),
                                    ).reshape(
                                        k_count_i,
                                        group_heads_i,
                                        int(exact_order_local_t.shape[2]),
                                        int(self.head_dim),
                                    )
                                    delta_t = torch.sum(
                                        gathered_probs_local_t.reshape(k_count_i, group_heads_i, -1, 1)
                                        * gathered_residual_local_t.float(),
                                        dim=2,
                                    )
                                    out_v = base_output_grid_t + delta_t
                                incremental_grid_outputs_by_v_idx[int(vi_i)] = out_v
                                return out_v

                            grid_outputs_for_v_idx = grid_outputs_for_v_idx_fn
                        elif int(max_exact_v_count) > 0:
                            risk_grid_t = (probs_grid_t * probs_grid_t) * code_error_t.to(dtype=prob_dtype).reshape(1, 1, -1)
                            if _env_truthy("SELECTOR_PQ_JOINT_NATIVE_RISK_PREFIX", "0"):
                                native = load_selector_paged_pq_ext()
                                grid_outputs_t = native.joint_vprefix_outputs_from_risk(
                                    base_output_grid_t.to(dtype=torch.float32).contiguous(),
                                    probs_grid_t.to(dtype=torch.float32).contiguous(),
                                    residual_t.to(dtype=torch.float32).contiguous(),
                                    code_error_t.to(dtype=torch.float32).contiguous(),
                                    joint_v_budgets_t,
                                )
                            elif _env_truthy("SELECTOR_PQ_JOINT_UNSORTED_V_PREFIX", "0"):
                                grid_outputs_by_v = []
                                for v_budget in joint_v_budgets:
                                    exact_count = max(0, min(int(v_budget), context_len_i, int(max_exact_v_count)))
                                    if exact_count <= 0:
                                        grid_outputs_by_v.append(base_output_grid_t)
                                    elif exact_count >= context_len_i:
                                        delta_t = (
                                            probs_grid_t.to(torch.float32).reshape(k_count_i * group_heads_i, context_len_i)
                                            @ residual_t.float()
                                        ).reshape(k_count_i, group_heads_i, int(self.head_dim))
                                        grid_outputs_by_v.append(base_output_grid_t + delta_t)
                                    else:
                                        exact_order_i_t = torch.topk(
                                            risk_grid_t,
                                            k=int(exact_count),
                                            dim=2,
                                            largest=True,
                                            sorted=False,
                                        ).indices
                                        gathered_probs_i_t = torch.gather(
                                            probs_grid_t.to(torch.float32),
                                            2,
                                            exact_order_i_t,
                                        )
                                        gathered_residual_i_t = residual_t.index_select(
                                            0,
                                            exact_order_i_t.reshape(-1),
                                        ).reshape(
                                            k_count_i,
                                            group_heads_i,
                                            int(exact_count),
                                            int(self.head_dim),
                                        )
                                        delta_t = torch.sum(
                                            gathered_probs_i_t.reshape(k_count_i, group_heads_i, int(exact_count), 1)
                                            * gathered_residual_i_t.float(),
                                            dim=2,
                                        )
                                        grid_outputs_by_v.append(base_output_grid_t + delta_t)
                                grid_outputs_t = torch.stack(grid_outputs_by_v, dim=1)
                            else:
                                if int(max_exact_v_count) >= context_len_i:
                                    exact_order_grid_t = torch.argsort(risk_grid_t, dim=2, descending=True, stable=True)
                                else:
                                    exact_order_grid_t = torch.topk(
                                        risk_grid_t,
                                        k=int(max_exact_v_count),
                                        dim=2,
                                        largest=True,
                                        sorted=True,
                                    ).indices
                                if _env_truthy("SELECTOR_PQ_JOINT_NATIVE_V_PREFIX", "0"):
                                    native = load_selector_paged_pq_ext()
                                    grid_outputs_t = native.joint_vprefix_outputs(
                                        base_output_grid_t.to(dtype=torch.float32).contiguous(),
                                        probs_grid_t.to(dtype=torch.float32).contiguous(),
                                        residual_t.to(dtype=torch.float32).contiguous(),
                                        exact_order_grid_t.to(dtype=torch.long).contiguous(),
                                        joint_v_budgets_t,
                                    )
                                else:
                                    gathered_probs_grid_t = torch.gather(
                                        probs_grid_t.to(torch.float32),
                                        2,
                                        exact_order_grid_t,
                                    )
                                    gathered_residual_grid_t = residual_t.index_select(0, exact_order_grid_t.reshape(-1)).reshape(
                                        k_count_i,
                                        group_heads_i,
                                        int(exact_order_grid_t.shape[2]),
                                        int(self.head_dim),
                                    )
                                    weighted_residual_grid_t = (
                                        gathered_probs_grid_t.reshape(k_count_i, group_heads_i, -1, 1)
                                        * gathered_residual_grid_t.float()
                                    )
                            if grid_outputs_t is None and _env_truthy("SELECTOR_PQ_JOINT_SEGMENTED_V_PREFIX", "0"):
                                prefix_delta_by_count = {}
                                running_delta_t = torch.zeros_like(base_output_grid_t, dtype=torch.float32)
                                prev_count = 0
                                exact_counts_sorted = sorted(
                                    {
                                        max(0, min(int(v_budget), context_len_i, int(max_exact_v_count)))
                                        for v_budget in joint_v_budgets
                                        if max(0, min(int(v_budget), context_len_i, int(max_exact_v_count))) > 0
                                    }
                                )
                                for exact_count in exact_counts_sorted:
                                    if int(exact_count) > int(prev_count):
                                        running_delta_t = running_delta_t + torch.sum(
                                            weighted_residual_grid_t[:, :, int(prev_count): int(exact_count), :],
                                        dim=2,
                                    )
                                prefix_delta_by_count[int(exact_count)] = running_delta_t.clone()
                                prev_count = int(exact_count)
                            elif grid_outputs_t is None:
                                prefix_delta_grid_t = torch.cumsum(
                                    weighted_residual_grid_t,
                                    dim=2,
                                )
                        if grid_outputs_t is None and not use_incremental_v_grid:
                            grid_outputs_by_v: list[torch.Tensor] = []
                            for v_budget in joint_v_budgets:
                                exact_count = max(0, min(int(v_budget), context_len_i))
                                if exact_count > 0 and prefix_delta_by_count is not None:
                                    grid_outputs_by_v.append(
                                        base_output_grid_t
                                        + prefix_delta_by_count[
                                            max(0, min(int(exact_count), int(max_exact_v_count)))
                                        ]
                                    )
                                elif exact_count > 0 and prefix_delta_grid_t is not None:
                                    grid_outputs_by_v.append(base_output_grid_t + prefix_delta_grid_t[:, :, int(exact_count) - 1, :])
                                else:
                                    grid_outputs_by_v.append(base_output_grid_t)
                            grid_outputs_t = torch.stack(grid_outputs_by_v, dim=1)
                        if bool(getattr(args, "profile_native_ops", False)):
                            _sync_if_cuda(device)
                            stats[layer_id].add_joint_detail_timing(
                                risk_prefix_seconds=float(time.perf_counter() - joint_risk_t0)
                            )
                        if wall_profile_enabled and int(max_exact_v_count) > 0:
                            stats[layer_id].add_joint_wall_timing(
                                risk_prefix_seconds=float(time.perf_counter() - joint_risk_wall_t0)
                            )
                    joint_policy_wall_t0 = time.perf_counter() if wall_profile_enabled else 0.0
                    if bool(getattr(args, "profile_native_ops", False)):
                        _sync_if_cuda(device)
                        joint_policy_t0 = time.perf_counter()
                    else:
                        joint_policy_t0 = 0.0
                    if use_incremental_v_grid and grid_outputs_for_v_idx is not None:
                        k_mb_by_idx = (
                            grid_k_mb_by_idx
                            if grid_k_mb_by_idx is not None
                            else [float(k_artifacts_batch(int(ki_i))[1]) for ki_i in range(len(active_joint_k_budgets))]
                            if policy_uses_mb
                            else [0.0 for _ in active_joint_k_budgets]
                        )
                        for local_head_i in range(group_heads_i):
                            ki = 0
                            vi = 0
                            steps = 0
                            while steps < (len(active_joint_k_budgets) + len(joint_v_budgets) + 4):
                                cur_output_t = grid_outputs_for_v_idx(int(vi))[int(ki), int(local_head_i)]
                                k_can = int(ki) + 1 < len(active_joint_k_budgets)
                                v_can = int(vi) + 1 < len(joint_v_budgets)
                                k_delta = (
                                    _rel_l2_torch(
                                        cur_output_t,
                                        grid_outputs_for_v_idx(int(vi))[int(ki) + 1, int(local_head_i)],
                                    )
                                    if k_can
                                    else 0.0
                                )
                                v_delta = (
                                    _rel_l2_torch(
                                        cur_output_t,
                                        grid_outputs_for_v_idx(int(vi) + 1)[int(ki), int(local_head_i)],
                                    )
                                    if v_can
                                    else 0.0
                                )
                                extra_k_mb = (
                                    float(k_mb_by_idx[int(ki) + 1] - k_mb_by_idx[int(ki)])
                                    if k_can
                                    else float("inf")
                                )
                                extra_v_mb = (
                                    float(v_mb_by_idx[int(vi) + 1] - v_mb_by_idx[int(vi)])
                                    if v_can and v_mb_by_idx is not None
                                    else 0.0
                                    if v_can
                                    else float("inf")
                                )
                                action = _choose_joint_kv_action(
                                    policy=policy_name,
                                    k_delta=float(k_delta),
                                    v_delta=float(v_delta),
                                    k_can=bool(k_can),
                                    v_can=bool(v_can),
                                    threshold=threshold_value,
                                    turn=int(steps),
                                    extra_k_mb=float(extra_k_mb),
                                    extra_v_mb=float(extra_v_mb),
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
                            final_ki_by_head.append(int(ki))
                            final_vi_by_head.append(int(vi))
                    elif (
                        _env_truthy("SELECTOR_PQ_JOINT_VECTOR_POLICY", "1")
                        and not _env_truthy("SELECTOR_PQ_JOINT_NATIVE_LAZY_POLICY", "0")
                    ):
                        output_grid_t = (
                            grid_outputs_t
                            if grid_outputs_t is not None
                            else torch.stack(
                                [
                                    torch.stack(
                                        [
                                            output_for_budget_batch(int(ki_i), int(vi_i))
                                            for vi_i in range(len(joint_v_budgets))
                                        ],
                                        dim=0,
                                    )
                                    for ki_i in range(len(active_joint_k_budgets))
                                ],
                                dim=0,
                            )
                        )
                        if _env_truthy("SELECTOR_PQ_JOINT_NATIVE_POLICY", "0"):
                            native = load_selector_paged_pq_ext()
                            if policy_uses_mb:
                                k_mb_by_idx = (
                                    grid_k_mb_by_idx
                                    if grid_k_mb_by_idx is not None
                                    else [float(k_artifacts_batch(int(ki_i))[1]) for ki_i in range(len(active_joint_k_budgets))]
                                )
                                k_mb_t = torch.as_tensor(k_mb_by_idx, dtype=torch.float32, device=device)
                                v_mb_t = torch.as_tensor(v_mb_by_idx, dtype=torch.float32, device=device)
                            else:
                                k_mb_t = torch.empty((len(active_joint_k_budgets),), dtype=torch.float32, device=device)
                                v_mb_t = torch.empty((len(joint_v_budgets),), dtype=torch.float32, device=device)
                            final_idx_t = native.joint_select_policy(
                                output_grid_t.to(dtype=torch.float32).contiguous(),
                                k_mb_t,
                                v_mb_t,
                                float(threshold_value),
                                policy_id,
                            )
                            if bool(getattr(args, "disable_cost_stats", False)):
                                final_idx_t_for_output = final_idx_t
                                final_output_grid_t = output_grid_t
                            else:
                                final_idx_rows = final_idx_t.detach().cpu().tolist()
                                for row in final_idx_rows:
                                    final_ki_by_head.append(int(row[0]))
                                    final_vi_by_head.append(int(row[1]))
                        else:
                            output_grid64_t = output_grid_t.to(dtype=torch.float64)
                            if len(active_joint_k_budgets) > 1:
                                k_cur_t = output_grid64_t[:-1]
                                k_next_t = output_grid64_t[1:]
                                k_delta_t = torch.linalg.vector_norm(k_cur_t - k_next_t, dim=-1) / torch.clamp_min(
                                    torch.linalg.vector_norm(k_next_t, dim=-1),
                                    1e-20,
                                )
                                k_delta_np = k_delta_t.detach().cpu().numpy()
                            else:
                                k_delta_np = np.empty((0, len(joint_v_budgets), group_heads_i), dtype=np.float64)
                            if len(joint_v_budgets) > 1:
                                v_cur_t = output_grid64_t[:, :-1]
                                v_next_t = output_grid64_t[:, 1:]
                                v_delta_t = torch.linalg.vector_norm(v_cur_t - v_next_t, dim=-1) / torch.clamp_min(
                                    torch.linalg.vector_norm(v_next_t, dim=-1),
                                    1e-20,
                                )
                                v_delta_np = v_delta_t.detach().cpu().numpy()
                            else:
                                v_delta_np = np.empty((len(active_joint_k_budgets), 0, group_heads_i), dtype=np.float64)
                            k_mb_by_idx = (
                                grid_k_mb_by_idx
                                if grid_k_mb_by_idx is not None
                                else [float(k_artifacts_batch(int(ki_i))[1]) for ki_i in range(len(active_joint_k_budgets))]
                                if policy_uses_mb
                                else [0.0 for _ in active_joint_k_budgets]
                            )
                            for local_head_i in range(group_heads_i):
                                ki = 0
                                vi = 0
                                steps = 0
                                while steps < (len(active_joint_k_budgets) + len(joint_v_budgets) + 4):
                                    k_can = int(ki) + 1 < len(active_joint_k_budgets)
                                    v_can = int(vi) + 1 < len(joint_v_budgets)
                                    k_delta = float(k_delta_np[int(ki), int(vi), int(local_head_i)]) if k_can else 0.0
                                    v_delta = float(v_delta_np[int(ki), int(vi), int(local_head_i)]) if v_can else 0.0
                                    extra_k_mb = (
                                        float(k_mb_by_idx[int(ki) + 1] - k_mb_by_idx[int(ki)])
                                        if k_can
                                        else float("inf")
                                    )
                                    extra_v_mb = (
                                        float(v_mb_by_idx[int(vi) + 1] - v_mb_by_idx[int(vi)])
                                        if v_can and v_mb_by_idx is not None
                                        else 0.0
                                        if v_can
                                        else float("inf")
                                    )
                                    action = _choose_joint_kv_action(
                                        policy=policy_name,
                                        k_delta=k_delta,
                                        v_delta=v_delta,
                                        k_can=bool(k_can),
                                        v_can=bool(v_can),
                                        threshold=threshold_value,
                                        turn=int(steps),
                                        extra_k_mb=float(extra_k_mb),
                                        extra_v_mb=float(extra_v_mb),
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
                                final_ki_by_head.append(int(ki))
                                final_vi_by_head.append(int(vi))
                    else:
                        for local_head_i in range(group_heads_i):
                            ki = 0
                            vi = 0
                            steps = 0
                            while steps < (len(active_joint_k_budgets) + len(joint_v_budgets) + 4):
                                cur_output_t = output_for_budget_batch(int(ki), int(vi))[int(local_head_i)]
                                k_can = int(ki) + 1 < len(active_joint_k_budgets)
                                v_can = int(vi) + 1 < len(joint_v_budgets)
                                k_delta = (
                                    _rel_l2_torch(
                                        cur_output_t,
                                        output_for_budget_batch(int(ki) + 1, int(vi))[int(local_head_i)],
                                    )
                                    if k_can
                                    else 0.0
                                )
                                v_delta = (
                                    _rel_l2_torch(
                                        cur_output_t,
                                        output_for_budget_batch(int(ki), int(vi) + 1)[int(local_head_i)],
                                    )
                                    if v_can
                                    else 0.0
                                )
                                extra_k_mb = (
                                    float(k_artifacts_batch(int(ki) + 1)[1] - k_artifacts_batch(int(ki))[1])
                                    if k_can
                                    else float("inf")
                                )
                                extra_v_mb = (
                                    float(v_mb_by_idx[int(vi) + 1] - v_mb_by_idx[int(vi)])
                                    if v_can and v_mb_by_idx is not None
                                    else 0.0
                                    if v_can
                                    else float("inf")
                                )
                                action = _choose_joint_kv_action(
                                    policy=policy_name,
                                    k_delta=float(k_delta),
                                    v_delta=float(v_delta),
                                    k_can=bool(k_can),
                                    v_can=bool(v_can),
                                    threshold=threshold_value,
                                    turn=int(steps),
                                    extra_k_mb=float(extra_k_mb),
                                    extra_v_mb=float(extra_v_mb),
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
                            final_ki_by_head.append(int(ki))
                            final_vi_by_head.append(int(vi))
                    if bool(getattr(args, "profile_native_ops", False)):
                        _sync_if_cuda(device)
                        stats[layer_id].add_joint_detail_timing(
                            policy_seconds=float(time.perf_counter() - joint_policy_t0)
                        )
                    if wall_profile_enabled:
                        stats[layer_id].add_joint_wall_timing(
                            policy_seconds=float(time.perf_counter() - joint_policy_wall_t0)
                        )
                    if bool(getattr(args, "profile_native_ops", False)):
                        _sync_if_cuda(device)
                        stats[layer_id].add_native_detail_timing(
                            geometric_seconds=float(time.perf_counter() - sim_t0)
                        )

                    if (
                        final_idx_t_for_output is not None
                        and final_output_grid_t is not None
                        and bool(getattr(args, "disable_cost_stats", False))
                    ):
                        head_idx_t = torch.arange(group_heads_i, dtype=torch.long, device=device)
                        final_idx_t_for_output = final_idx_t_for_output.to(device=device, dtype=torch.long)
                        outputs_all[head_start_i:head_end_i] = final_output_grid_t[
                            final_idx_t_for_output[:, 0],
                            final_idx_t_for_output[:, 1],
                            head_idx_t,
                        ]
                        continue

                    for local_head_i, (ki, vi) in enumerate(zip(final_ki_by_head, final_vi_by_head, strict=True)):
                        global_head_i = int(head_start_i + local_head_i)
                        if not bool(getattr(args, "disable_cost_stats", False)):
                            if grid_selected_counts_by_ki is not None:
                                selected_count_i = int(grid_selected_counts_by_ki[int(ki)])
                            else:
                                selected_t_i = (
                                    grid_selected_by_ki[int(ki)][int(local_head_i)]
                                    if grid_selected_by_ki is not None
                                    else k_artifacts_batch(int(ki))[0][int(local_head_i)]
                                )
                                selected_count_i = int(selected_t_i.numel())
                            exact_v_count = max(0, min(int(joint_v_budgets[int(vi)]), context_len_i))
                            exact_key_mb = float(selected_count_i * int(self.head_dim) * key_bytes) / MB
                            exact_v_mb = float(exact_v_count * int(self.head_dim) * value_bytes) / MB
                            compressed_v_codes_mb = (
                                float(max(0, context_len_i - exact_v_count) * actual_value_subvecs * code_bytes) / MB
                            )
                            tail_mb_override = float(v_pq_codebook_mb + compressed_v_codes_mb + metadata_mb)
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
                        outputs_all[global_head_i] = (
                            grid_outputs_t[int(ki), int(vi), int(local_head_i)]
                            if grid_outputs_t is not None
                            else grid_outputs_for_v_idx(int(vi))[int(ki), int(local_head_i)]
                            if grid_outputs_for_v_idx is not None
                            else output_for_budget_batch(int(ki), int(vi))[int(local_head_i)]
                        )

                if grouped_risk_records:
                    if not _env_truthy("SELECTOR_PQ_JOINT_NATIVE_POLICY", "0"):
                        raise RuntimeError("SELECTOR_PQ_JOINT_GROUPED_RISK_PREFIX requires native joint policy")
                    native = load_selector_paged_pq_ext()
                    grouped_by_shape: dict[tuple[int, int, int, int], list[dict[str, object]]] = {}
                    for record in grouped_risk_records:
                        base_grid = record["base_output_grid"]
                        probs_grid = record["probs_grid"]
                        if not isinstance(base_grid, torch.Tensor) or not isinstance(probs_grid, torch.Tensor):
                            raise RuntimeError("invalid grouped risk-prefix record")
                        shape_key = (
                            int(base_grid.shape[0]),
                            int(base_grid.shape[1]),
                            int(probs_grid.shape[2]),
                            int(base_grid.shape[2]),
                        )
                        grouped_by_shape.setdefault(shape_key, []).append(record)

                    grouped_risk_wall_t0 = time.perf_counter() if wall_profile_enabled else 0.0
                    if bool(getattr(args, "profile_native_ops", False)):
                        _sync_if_cuda(device)
                        grouped_risk_t0 = time.perf_counter()
                    else:
                        grouped_risk_t0 = 0.0
                    grouped_policy_batches: list[
                        tuple[list[dict[str, object]], int, int, int, torch.Tensor]
                    ] = []
                    use_fused_grouped_risk_policy = _env_truthy("SELECTOR_PQ_JOINT_FUSED_RISK_POLICY", "0")

                    def grouped_policy_mb_tensors(
                        records_i: list[dict[str, object]],
                        k_count_i: int,
                    ) -> tuple[torch.Tensor, torch.Tensor]:
                        if not policy_uses_mb:
                            return (
                                torch.empty((len(records_i), int(k_count_i)), dtype=torch.float32, device=device),
                                torch.empty((len(records_i), len(joint_v_budgets)), dtype=torch.float32, device=device),
                            )
                        return (
                            torch.stack(
                                [
                                    torch.as_tensor(record["grid_k_mb_by_idx"], dtype=torch.float32, device=device)
                                    for record in records_i
                                ],
                                dim=0,
                            ).contiguous(),
                            torch.stack(
                                [
                                    torch.as_tensor(record["v_mb_by_idx"], dtype=torch.float32, device=device)
                                    for record in records_i
                                ],
                                dim=0,
                            ).contiguous(),
                        )

                    for (k_count_i, group_heads_i, context_len_bucket, dim_i), records in grouped_by_shape.items():
                        rows_per_group = int(k_count_i) * int(group_heads_i)
                        group_pack_wall_t0 = time.perf_counter() if wall_profile_enabled else 0.0
                        if bool(getattr(args, "profile_native_ops", False)):
                            _sync_if_cuda(device)
                            group_pack_t0 = time.perf_counter()
                        else:
                            group_pack_t0 = 0.0
                        base_grouped_t = torch.stack(
                            [
                                record["base_output_grid"]
                                for record in records
                                if isinstance(record["base_output_grid"], torch.Tensor)
                            ],
                            dim=0,
                        ).contiguous()
                        probs_grouped_t = torch.stack(
                            [
                                record["probs_grid"]
                                for record in records
                                if isinstance(record["probs_grid"], torch.Tensor)
                            ],
                            dim=0,
                        ).contiguous()
                        if (
                            grouped_vpq_residual_groups_t is not None
                            and grouped_vpq_code_error_groups_t is not None
                            and len(records) == int(grouped_vpq_residual_groups_t.shape[0])
                        ):
                            residual_groups_t = grouped_vpq_residual_groups_t.contiguous()
                            code_error_groups_t = grouped_vpq_code_error_groups_t.contiguous()
                        else:
                            residual_groups_t = torch.stack(
                                [
                                    record["residual"]
                                    for record in records
                                    if isinstance(record["residual"], torch.Tensor)
                                ],
                                dim=0,
                            ).contiguous()
                            code_error_groups_t = torch.stack(
                                [
                                    record["code_error"]
                                    for record in records
                                    if isinstance(record["code_error"], torch.Tensor)
                                ],
                                dim=0,
                            ).contiguous()
                        if bool(getattr(args, "profile_native_ops", False)):
                            _sync_if_cuda(device)
                            stats[layer_id].add_joint_detail_timing(
                                group_pack_seconds=float(time.perf_counter() - group_pack_t0)
                            )
                        if wall_profile_enabled:
                            stats[layer_id].add_joint_wall_timing(
                                group_pack_seconds=float(time.perf_counter() - group_pack_wall_t0)
                            )
                        if use_fused_grouped_risk_policy:
                            k_mb_groups_t, v_mb_groups_t = grouped_policy_mb_tensors(records, int(k_count_i))
                            if hasattr(native, "joint_select_policy_from_grouped_risk_batched"):
                                final_outputs_grouped_t, final_idx_grouped_t = (
                                    native.joint_select_policy_from_grouped_risk_batched(
                                        base_grouped_t,
                                        probs_grouped_t,
                                        residual_groups_t,
                                        code_error_groups_t,
                                        joint_v_budgets_t,
                                        k_mb_groups_t,
                                        v_mb_groups_t,
                                        float(threshold_value),
                                        policy_id,
                                    )
                                )
                            else:
                                final_outputs_grouped_t, final_idx_grouped_t = native.joint_select_policy_from_grouped_risk(
                                    base_grouped_t.reshape(len(records) * rows_per_group, dim_i),
                                    probs_grouped_t.reshape(len(records) * rows_per_group, context_len_bucket),
                                    residual_groups_t,
                                    code_error_groups_t,
                                    joint_v_budgets_t,
                                    k_mb_groups_t,
                                    v_mb_groups_t,
                                    int(k_count_i),
                                    int(group_heads_i),
                                    float(threshold_value),
                                    policy_id,
                                )
                            for record_i, record in enumerate(records):
                                record["final_outputs"] = final_outputs_grouped_t[int(record_i)]
                                record["final_indices"] = final_idx_grouped_t[int(record_i)]
                            continue
                        if (
                            _env_truthy("SELECTOR_PQ_JOINT_RISK_PREFIX_TOPK", "0")
                            and hasattr(native, "joint_vprefix_outputs_from_grouped_risk_topk_batched")
                        ):
                            max_exact_v_count_i = max(
                                0,
                                min(
                                    max((int(v) for v in joint_v_budgets), default=0),
                                    int(context_len_bucket),
                                ),
                            )
                            grouped_outputs_flat_t = native.joint_vprefix_outputs_from_grouped_risk_topk_batched(
                                base_grouped_t,
                                probs_grouped_t,
                                residual_groups_t,
                                code_error_groups_t,
                                joint_v_budgets_t,
                                int(max_exact_v_count_i),
                            )
                        elif hasattr(native, "joint_vprefix_outputs_from_grouped_risk_batched"):
                            grouped_outputs_flat_t = native.joint_vprefix_outputs_from_grouped_risk_batched(
                                base_grouped_t,
                                probs_grouped_t,
                                residual_groups_t,
                                code_error_groups_t,
                                joint_v_budgets_t,
                            )
                        else:
                            row_group_ids_t = torch.arange(
                                len(records),
                                dtype=torch.long,
                                device=device,
                            ).repeat_interleave(rows_per_group)
                            grouped_outputs_flat_t = native.joint_vprefix_outputs_from_grouped_risk(
                                base_grouped_t.reshape(len(records) * rows_per_group, dim_i),
                                probs_grouped_t.reshape(len(records) * rows_per_group, context_len_bucket),
                                residual_groups_t,
                                code_error_groups_t,
                                row_group_ids_t,
                                joint_v_budgets_t,
                            )
                        grouped_policy_batches.append(
                            (records, int(k_count_i), int(group_heads_i), int(dim_i), grouped_outputs_flat_t)
                        )
                    if bool(getattr(args, "profile_native_ops", False)):
                        _sync_if_cuda(device)
                        stats[layer_id].add_joint_detail_timing(
                            risk_prefix_seconds=float(time.perf_counter() - grouped_risk_t0)
                        )
                        grouped_policy_t0 = time.perf_counter()
                    else:
                        grouped_policy_t0 = 0.0
                    if wall_profile_enabled:
                        stats[layer_id].add_joint_wall_timing(
                            risk_prefix_seconds=float(time.perf_counter() - grouped_risk_wall_t0)
                        )
                    grouped_policy_wall_t0 = time.perf_counter() if wall_profile_enabled else 0.0

                    for records, k_count_i, group_heads_i, dim_i, grouped_outputs_flat_t in grouped_policy_batches:
                        if (not policy_uses_mb) and hasattr(native, "joint_select_policy_grouped_flat_no_mb"):
                            final_outputs_grouped_t, final_idx_grouped_t = native.joint_select_policy_grouped_flat_no_mb(
                                grouped_outputs_flat_t,
                                int(k_count_i),
                                int(group_heads_i),
                                float(threshold_value),
                                policy_id,
                            )
                        else:
                            k_mb_groups_t, v_mb_groups_t = grouped_policy_mb_tensors(records, int(k_count_i))
                            final_outputs_grouped_t, final_idx_grouped_t = native.joint_select_policy_grouped_flat(
                                grouped_outputs_flat_t,
                                k_mb_groups_t,
                                v_mb_groups_t,
                                int(k_count_i),
                                int(group_heads_i),
                                float(threshold_value),
                                policy_id,
                            )
                        for record_i, record in enumerate(records):
                            record["final_outputs"] = final_outputs_grouped_t[int(record_i)]
                            record["final_indices"] = final_idx_grouped_t[int(record_i)]

                    if bool(getattr(args, "profile_native_ops", False)):
                        _sync_if_cuda(device)
                        stats[layer_id].add_joint_detail_timing(
                            policy_seconds=float(time.perf_counter() - grouped_policy_t0)
                        )
                        grouped_accounting_t0 = time.perf_counter()
                    else:
                        grouped_accounting_t0 = 0.0
                    if wall_profile_enabled:
                        stats[layer_id].add_joint_wall_timing(
                            policy_seconds=float(time.perf_counter() - grouped_policy_wall_t0)
                        )
                    grouped_accounting_wall_t0 = time.perf_counter() if wall_profile_enabled else 0.0

                    for record in grouped_risk_records:
                        final_output_t = record["final_outputs"]
                        final_idx_t = record["final_indices"]
                        if not isinstance(final_output_t, torch.Tensor) or not isinstance(final_idx_t, torch.Tensor):
                            raise RuntimeError("missing grouped risk-prefix final output")
                        group_heads_i = int(record["group_heads"])
                        head_start_i = int(record["head_start"])
                        head_end_i = int(record["head_end"])
                        context_len_i = int(record["context_len"])
                        if bool(getattr(args, "disable_cost_stats", False)):
                            outputs_all[head_start_i:head_end_i] = final_output_t[:group_heads_i]
                            continue

                        grid_selected_by_ki = record["grid_selected_by_ki"]
                        if not isinstance(grid_selected_by_ki, list):
                            raise RuntimeError("invalid grouped risk-prefix selected-token metadata")
                        grid_selected_counts_by_ki = record.get("grid_selected_counts_by_ki")
                        if grid_selected_counts_by_ki is not None and not isinstance(grid_selected_counts_by_ki, list):
                            raise RuntimeError("invalid grouped risk-prefix selected-count metadata")
                        final_idx_rows = final_idx_t.detach().cpu().tolist()
                        for local_head_i, row in enumerate(final_idx_rows):
                            ki = int(row[0])
                            vi = int(row[1])
                            global_head_i = int(head_start_i + local_head_i)
                            if grid_selected_counts_by_ki is not None:
                                selected_count_i = int(grid_selected_counts_by_ki[int(ki)])
                            else:
                                selected_t_i = grid_selected_by_ki[int(ki)]
                                if selected_t_i is None:
                                    raise RuntimeError("missing selected-token tensor for grouped risk-prefix accounting")
                                selected_count_i = int(selected_t_i[int(local_head_i)].numel())
                            exact_v_count = max(0, min(int(joint_v_budgets[int(vi)]), context_len_i))
                            exact_key_mb = float(selected_count_i * int(self.head_dim) * key_bytes) / MB
                            exact_v_mb = float(exact_v_count * int(self.head_dim) * value_bytes) / MB
                            compressed_v_codes_mb = (
                                float(max(0, context_len_i - exact_v_count) * int(record["actual_value_subvecs"]) * code_bytes) / MB
                            )
                            tail_mb_override = (
                                float(record["v_pq_codebook_mb"]) + compressed_v_codes_mb + metadata_mb
                            )
                            dense_physical_key_mb = float(context_len_i * int(self.head_dim) * key_bytes) / MB
                            stats[layer_id].add_count(
                                int(selected_count_i),
                                max(0, context_len_i - int(exact_v_count)),
                                float(record["selector_mb"]),
                                int(self.head_dim),
                                key_bytes,
                                value_bytes,
                                tail_mb_override=tail_mb_override,
                                exact_kv_mb_override=float(exact_key_mb + exact_v_mb),
                                confidence_mb_override=0.0,
                                physical_gpu_exact_kv_mb_override=float(dense_physical_key_mb + exact_v_mb),
                                physical_gpu_confidence_mb_override=0.0,
                            )
                            outputs_all[global_head_i] = final_output_t[int(local_head_i)]

                    if bool(getattr(args, "profile_native_ops", False)):
                        _sync_if_cuda(device)
                        stats[layer_id].add_joint_detail_timing(
                            accounting_seconds=float(time.perf_counter() - grouped_accounting_t0)
                        )
                        if grouped_geo_t0 > 0.0:
                            stats[layer_id].add_native_detail_timing(
                                geometric_seconds=float(time.perf_counter() - grouped_geo_t0)
                            )
                    if wall_profile_enabled:
                        stats[layer_id].add_joint_wall_timing(
                            accounting_seconds=float(time.perf_counter() - grouped_accounting_wall_t0)
                        )

                if wall_profile_enabled:
                    stats[layer_id].add_joint_wall_timing(
                        total_seconds=float(time.perf_counter() - joint_total_wall_t0)
                    )
                return outputs_all.to(dtype=hidden_states.dtype, device=device)

            fast_outputs = approximate_decode_fast_exact(0, context_len) if query_len == 1 else None
            if fast_outputs is not None:
                attn_output = fast_outputs.reshape(1, 1, -1).to(hidden_states.dtype).contiguous()
            elif query_len == 1:
                joint_outputs = approximate_joint_kv_all_heads(0, context_len)
                if joint_outputs is not None:
                    attn_output = joint_outputs.reshape(1, 1, -1).contiguous()
                else:
                    outputs = [approximate_one_head(head, 0, context_len) for head in range(num_heads)]
                    attn_output = torch.stack(outputs, dim=0).reshape(1, 1, -1).contiguous()
            else:
                fast_prefill_outputs = approximate_prefill_fast_exact()
                if fast_prefill_outputs is not None:
                    attn_output = fast_prefill_outputs.reshape(1, query_len, -1).contiguous()
                else:
                    per_pos_outputs = []
                    for local_qpos in range(query_len):
                        query_context_len = int(query_start + local_qpos + 1)
                        fast_pos_outputs = approximate_decode_fast_exact(local_qpos, query_context_len)
                        if fast_pos_outputs is not None:
                            per_pos_outputs.append(fast_pos_outputs.reshape(-1).to(hidden_states.dtype))
                            continue
                        outputs = [
                            approximate_one_head(head, local_qpos, query_context_len)
                            for head in range(num_heads)
                        ]
                        per_pos_outputs.append(torch.stack(outputs, dim=0).reshape(-1))
                    attn_output = torch.stack(per_pos_outputs, dim=0).reshape(1, query_len, -1).contiguous()

            oproj_wall_t0 = time.perf_counter() if wall_profile_enabled else 0.0
            if bool(getattr(args, "profile_native_ops", False)):
                _sync_if_cuda(device)
                oproj_t0 = time.perf_counter()
            else:
                oproj_t0 = 0.0
            attn_output = self.o_proj(attn_output)
            if bool(getattr(args, "profile_native_ops", False)):
                _sync_if_cuda(device)
                stats[layer_id].add_output_projection_timing(time.perf_counter() - oproj_t0)
            if wall_profile_enabled:
                stats[layer_id].add_wall_output_projection_timing(time.perf_counter() - oproj_wall_t0)
            if bool(getattr(args, "profile_native_ops", False)):
                _sync_if_cuda(device)
                stats[layer_id].add_patched_attention_timing(time.perf_counter() - patched_attention_t0)
            if wall_profile_enabled:
                stats[layer_id].add_wall_patched_attention_timing(
                    time.perf_counter() - patched_attention_wall_t0
                )
            return attn_output, None

        return types.MethodType(forward, module)

    try:
        for layer_id in layer_ids:
            module = model.model.layers[int(layer_id)].self_attn
            originals[int(layer_id)] = module.forward
            stats[int(layer_id)] = ApproxStats()
            module.forward = make_forward(int(layer_id), module)
            setattr(
                module,
                "_pagedpq_warm_decode_sidecars",
                lambda cache_obj, lid=int(layer_id), mod=module: warm_dense_prefill_decode_sidecars(
                    int(lid),
                    mod,
                    cache_obj,
                ),
            )
        yield
    finally:
        for layer_id, forward in originals.items():
            module = model.model.layers[int(layer_id)].self_attn
            module.forward = forward
            if hasattr(module, "_pagedpq_warm_decode_sidecars"):
                delattr(module, "_pagedpq_warm_decode_sidecars")
            if hasattr(module, "_pagedpq_joint_vpq_sidecar_cache"):
                delattr(module, "_pagedpq_joint_vpq_sidecar_cache")


def run() -> None:
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
    parser.add_argument("--selector_mode", choices=["fullscan", "routed", "oracle"], default="routed")
    parser.add_argument("--budget", type=int, default=4096)
    parser.add_argument("--budget_by_head", default="")
    parser.add_argument("--rerank_candidates", type=int, default=0)
    parser.add_argument("--tail_samples", type=int, default=16384)
    parser.add_argument("--tail_bands", type=int, default=8)
    parser.add_argument("--tail_seed", type=int, default=0)
    parser.add_argument("--tail_sampling", choices=["random", "linspace", "systematic"], default="systematic")
    parser.add_argument("--tail_mode", choices=["sample", "pq_value", "vpq_value", "page_mean"], default="sample")
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
        help="deployable online budget/confidence rule; non-none disables fixed-budget native fast paths",
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
    parser.add_argument("--joint_kv_stability_threshold", type=float, default=0.001)
    parser.add_argument("--selected_value_mode", choices=["exact", "vpq_value"], default="exact")
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
        default="fixed",
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
        default=os.environ.get("PAGEDPQ_INDEX_BUILD_BACKEND", "numpy"),
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
        default=os.environ.get("SELECTOR_PAGED_PQ_BACKEND", "torch"),
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
    args = parser.parse_args()

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True), encoding="utf-8")

    device = torch.device(args.device)
    log("loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir, local_files_only=bool(args.local_files_only))
    tokenizer.pad_token = tokenizer.eos_token
    log("loading model")
    load_kwargs = {
        "cache_dir": args.cache_dir,
        "local_files_only": bool(args.local_files_only),
        "torch_dtype": torch.bfloat16,
        "attn_implementation": str(args.attn_implementation),
    }
    if not bool(args.cpu_then_to_device):
        load_kwargs["device_map"] = {"": str(device)}
    model = LlamaForCausalLM.from_pretrained(args.model_name, **load_kwargs)
    if bool(args.cpu_then_to_device):
        log(f"moving model to {device}")
        model = model.to(device)
    model.eval()
    forbidden = {tok for tok in [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")] if isinstance(tok, int) and tok >= 0}

    prompt = make_needle_prompt(str(args.target), int(args.filler_repeats))
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    log(f"prompt_tokens={int(input_ids.shape[1])} dense_trace_start")
    start = time.perf_counter()
    dense = greedy_dense_trace(model, input_ids, int(args.max_new_tokens), forbidden)
    dense_time = time.perf_counter() - start
    log(f"dense_trace_done seconds={dense_time:.3f}")
    dense_text = tokenizer.decode(dense["tokens"], skip_special_tokens=True)

    layer_ids = parse_csv_ints(args.layers)
    approx_stats: dict[int, ApproxStats] = {}
    start = time.perf_counter()
    log(f"approx_trace_start layers={layer_ids}")
    with patched_paged_pq_attention(model, layer_ids, args, approx_stats):
        approx_teacher = teacher_forced_trace(model, input_ids, dense["tokens"], forbidden)
        approx_free = greedy_dense_trace(model, input_ids, int(args.max_new_tokens), forbidden)
    approx_time = time.perf_counter() - start
    log(f"approx_trace_done seconds={approx_time:.3f}")
    approx_text = tokenizer.decode(approx_free["tokens"], skip_special_tokens=True)

    comparison = summarize_logit_trace(dense, approx_teacher, tokenizer, ignore_token_ids=forbidden)
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
    task = {
        "target": str(args.target),
        "prompt_tokens": int(input_ids.shape[1]),
        "dense_text": dense_text,
        "approx_free_text": approx_text,
        "dense_contains_target": str(args.target).lower() in dense_text.lower(),
        "approx_free_contains_target": str(args.target).lower() in approx_text.lower(),
        "free_run_exact_text_match": dense_text == approx_text,
        "dense_tokens": [int(x) for x in dense["tokens"]],
        "approx_free_tokens": [int(x) for x in approx_free["tokens"]],
    }
    summary = {
        "algorithm": (
            f"hf_routed_paged_pq_k{int(args.budget)}"
            f"_{args.selector_mode}"
            f"_rerank{int(args.rerank_candidates)}"
            f"+{args.tail_mode}_tail_b{int(args.tail_bands)}_s{int(args.tail_samples)}"
            f"_blend{float(args.tail_blend):g}"
        ),
        "layers": layer_ids,
        "dense_seconds": float(dense_time),
        "approx_seconds": float(approx_time),
        "logit_trace": comparison["summary"],
        "task": task,
        "cost_proxy": stats_payload,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "logit_steps.json").write_text(json.dumps(comparison["steps"], indent=2, sort_keys=True), encoding="utf-8")
    log(f"wrote {out_dir}")


if __name__ == "__main__":
    run()
