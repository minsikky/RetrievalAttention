#!/usr/bin/env python3
from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.selector_eval.frontier_config import (
    canonical_gpu_frontier_mismatches as shared_canonical_gpu_frontier_mismatches,
)
from benchmark.selector_eval.gpu.run_gpu_paged_pq_eval import (
    GPUIndex,
    _sync_if_cuda,
    build_page_pq_gpu,
    build_page_pq_torch,
    parse_csv_ints,
)

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
    """Return config fields that would change CPU-frontier algorithm semantics."""
    return shared_canonical_gpu_frontier_mismatches(args, env=os.environ)


def _parse_budget_schedule(text: str, *, name: str) -> list[int]:
    values = sorted({int(x) for x in parse_csv_ints(text) if int(x) > 0})
    if not values:
        raise ValueError(f"{name} must contain at least one positive integer budget")
    return values


def _parse_ratio_schedule(text: str, *, name: str) -> list[float]:
    ratios: list[float] = []
    for part in str(text).split(","):
        token = part.strip().lower()
        if not token:
            continue
        is_percent = token.endswith("%")
        if is_percent:
            token = token[:-1]
        value = float(token.replace("p", "."))
        ratios.append(value / 100.0 if is_percent else value)
    if not ratios:
        raise ValueError(f"{name} must contain at least one positive ratio")
    return ratios


def _budgets_from_fraction_schedule(text: str, *, name: str, context_len: int) -> list[int]:
    budgets = {
        max(1, min(int(context_len), int(math.ceil(float(context_len) * float(frac)))))
        for frac in _parse_ratio_schedule(text, name=name)
        if float(frac) > 0.0
    }
    if not budgets:
        raise ValueError(f"{name} produced no positive budgets")
    return sorted(budgets)


def _scaled_budget_delta_threshold(
    *,
    base_threshold: float,
    budget_delta_frac: float,
    reference_frac: float,
    shape: str,
    min_scale: float,
    max_scale: float,
) -> float:
    ref = max(float(reference_frac), 1e-12)
    ratio = max(float(budget_delta_frac), 0.0) / ref
    mode = str(shape).strip().lower()
    if mode == "linear":
        scale = ratio
    elif mode == "sqrt":
        scale = math.sqrt(ratio)
    elif mode == "log":
        scale = math.log1p(ratio) / math.log(2.0)
    else:
        raise ValueError(f"unknown joint_kv_threshold_scale_shape: {shape}")
    scale = min(max(float(scale), float(min_scale)), float(max_scale))
    return float(base_threshold) * scale


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
    k_threshold: float | None = None,
    v_threshold: float | None = None,
    turn: int,
    extra_k_mb: float,
    extra_v_mb: float,
) -> str:
    effective_k_threshold = float(threshold if k_threshold is None else k_threshold)
    effective_v_threshold = float(threshold if v_threshold is None else v_threshold)
    k_bad = bool(k_can and float(k_delta) > effective_k_threshold)
    v_bad = bool(v_can and float(v_delta) > effective_v_threshold)
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
