#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import safe_open

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.attention_efficiency_threeway_eval import build_pq_index
from benchmark.selector_eval.data.trace import attention_probs, load_trace
from benchmark.selector_eval.gpu.run_gpu_paged_pq_eval import parse_csv_ints
from benchmark.selector_eval.metrics.attention import (
    _logit_error_metrics,
    _output_error_metrics,
    _softmax_from_scores,
    attention_distribution_error_metrics,
)
from benchmark.selector_eval.metrics.tail_estimators import (
    _paper_tq_reconstruct,
    _paper_tq_scores,
    _tq_reconstruct,
    _tq_reconstruct_product,
)

MB = 1024.0 * 1024.0


@dataclass(frozen=True)
class CompressedKV:
    method: str
    keys_hat: np.ndarray
    values_hat: np.ndarray
    active_mask: np.ndarray | None
    score_source_keys: np.ndarray | None
    score_mask: np.ndarray | None
    query_read_mb: float
    online_update_mb_per_token: float
    metadata: dict[str, float | int | str | bool]
    position_weights: np.ndarray | None = None


def load_weight_index(model_dir: Path) -> dict[str, str]:
    index_path = model_dir / "model.safetensors.index.json"
    data = json.loads(index_path.read_text())
    return {str(k): str(v) for k, v in data["weight_map"].items()}


def load_safetensor_weight(model_dir: Path, weight_map: dict[str, str], name: str, device: torch.device) -> torch.Tensor:
    shard = model_dir / weight_map[name]
    with safe_open(shard, framework="pt", device="cpu") as f:
        return f.get_tensor(name).to(device=device, dtype=torch.float32, non_blocking=True)


def dense_attention_output(keys_np: np.ndarray, values_np: np.ndarray, query_np: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scores_np, probs_np = attention_probs(keys_np, query_np)
    out = probs_np.astype(np.float64, copy=False) @ values_np.astype(np.float64, copy=False)
    return scores_np.astype(np.float32, copy=False), probs_np.astype(np.float64, copy=False), out.astype(np.float32, copy=False)


def attention_outputs(
    keys_np: np.ndarray,
    values_np: np.ndarray,
    queries_np: np.ndarray,
    position_weights: np.ndarray | None = None,
) -> np.ndarray:
    keys32 = keys_np.astype(np.float32, copy=False)
    values32 = values_np.astype(np.float32, copy=False)
    queries32 = queries_np.astype(np.float32, copy=False)
    scores = (queries32 @ keys32.T) * (1.0 / np.sqrt(float(keys32.shape[-1])))
    scores = scores.astype(np.float64, copy=False)
    if position_weights is not None:
        weights = np.maximum(np.asarray(position_weights, dtype=np.float64), 1e-20)
        scores += np.log(weights)[None, :]
    scores -= np.max(scores, axis=1, keepdims=True)
    probs = np.exp(scores)
    probs /= np.maximum(np.sum(probs, axis=1, keepdims=True), 1e-20)
    return (probs @ values32.astype(np.float64, copy=False)).astype(np.float32, copy=False)


def scores_probs_for_keys(
    keys_np: np.ndarray,
    query_np: np.ndarray,
    position_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    keys32 = keys_np.astype(np.float32, copy=False)
    query32 = query_np.astype(np.float32, copy=False)
    scores = (keys32 @ query32) * (1.0 / np.sqrt(float(keys32.shape[-1])))
    effective_scores = scores.astype(np.float64, copy=False)
    if position_weights is not None:
        weights = np.maximum(np.asarray(position_weights, dtype=np.float64), 1e-20)
        effective_scores = effective_scores + np.log(weights)
    probs = _softmax_from_scores(effective_scores)
    return effective_scores.astype(np.float32, copy=False), probs


def compressed_scores_probs(
    comp: CompressedKV,
    query_np: np.ndarray,
    *,
    context_len: int,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Return full-context approximate scores/probs plus common-support scores when possible."""
    query32 = query_np.astype(np.float32, copy=False)
    score_family = str(comp.metadata.get("score_family", ""))
    if comp.active_mask is not None:
        mask = np.asarray(comp.active_mask, dtype=bool)
        active_scores, active_probs = scores_probs_for_keys(comp.keys_hat[mask], query32)
        full_probs = np.zeros((int(context_len),), dtype=np.float64)
        full_probs[mask] = active_probs
        return None, full_probs, active_scores, mask
    if comp.keys_hat.shape[0] != int(context_len):
        return None, None, None, None
    if score_family == "tq_paper":
        source_keys = comp.score_source_keys if comp.score_source_keys is not None else comp.keys_hat
        key_bits = int(comp.metadata.get("key_bits", 4))
        product_residual = bool(comp.metadata.get("product_residual", False))
        if product_residual and comp.score_mask is not None:
            scores = comp.keys_hat.astype(np.float64, copy=False) @ query32.astype(np.float64, copy=False)
            corrected = _paper_tq_scores(source_keys[comp.score_mask], query32, key_bits, product_residual=True)
            scores[np.asarray(comp.score_mask, dtype=bool)] = corrected
            scores = scores * (1.0 / np.sqrt(float(query32.shape[-1])))
            probs = _softmax_from_scores(scores)
            return scores.astype(np.float32, copy=False), probs, None, None
        raw = _paper_tq_scores(source_keys, query32, key_bits, product_residual=product_residual)
        scores = raw.astype(np.float64, copy=False) * (1.0 / np.sqrt(float(query32.shape[-1])))
        probs = _softmax_from_scores(scores)
        return scores.astype(np.float32, copy=False), probs, None, None
    scores, probs = scores_probs_for_keys(comp.keys_hat, query32, comp.position_weights)
    return scores, probs, None, None


def compressed_attention_outputs(comp: CompressedKV, queries_np: np.ndarray) -> np.ndarray:
    if comp.active_mask is not None:
        return attention_outputs(comp.keys_hat[comp.active_mask], comp.values_hat[comp.active_mask], queries_np)
    score_family = str(comp.metadata.get("score_family", ""))
    if score_family == "tq_paper":
        source_keys = comp.score_source_keys
        if source_keys is None:
            source_keys = comp.keys_hat
        key_bits = int(comp.metadata.get("key_bits", 4))
        product_residual = bool(comp.metadata.get("product_residual", False))
        values32 = comp.values_hat.astype(np.float32, copy=False)
        keys32 = comp.keys_hat.astype(np.float32, copy=False)
        score_mask = comp.score_mask
        outs = []
        scale = 1.0 / np.sqrt(float(values32.shape[-1]))
        for query in queries_np.astype(np.float32, copy=False):
            if product_residual and score_mask is not None:
                scores = keys32.astype(np.float64, copy=False) @ query.astype(np.float64, copy=False)
                corrected = _paper_tq_scores(source_keys[score_mask], query, key_bits, product_residual=True)
                scores[score_mask] = corrected
            else:
                scores = _paper_tq_scores(source_keys, query, key_bits, product_residual=product_residual)
            scores = scores * scale
            scores = scores.astype(np.float64, copy=False)
            scores -= float(np.max(scores))
            probs = np.exp(scores)
            probs /= max(float(np.sum(probs)), 1e-20)
            outs.append((probs @ values32.astype(np.float64, copy=False)).astype(np.float32, copy=False))
        return np.stack(outs, axis=0)
    return attention_outputs(comp.keys_hat, comp.values_hat, queries_np, position_weights=comp.position_weights)


def project_full(concat: np.ndarray, wo: torch.Tensor, device: torch.device) -> np.ndarray:
    x = torch.as_tensor(concat, dtype=torch.float32, device=device).reshape(1, -1)
    y = F.linear(x, wo)
    return np.asarray(y.reshape(-1).detach().cpu().tolist(), dtype=np.float32)


def exact_window_mask(context_len: int, *, static_prefix: int, residual_window: int) -> np.ndarray:
    mask = np.zeros((int(context_len),), dtype=bool)
    prefix = min(max(0, int(static_prefix)), int(context_len))
    if prefix:
        mask[:prefix] = True
    suffix = min(max(0, int(residual_window)), int(context_len))
    if suffix:
        mask[int(context_len) - suffix :] = True
    return mask


def _packed_code_bytes(count: int, elements: int, bits: int) -> float:
    return float(int(count) * int(elements) * int(bits)) / 8.0


def _scale_zero_bytes(count: int, groups: int, bytes_per_scalar: int) -> float:
    return float(int(count) * int(groups) * 2 * int(bytes_per_scalar))


def _parse_window(text: str, default: int) -> int:
    match = re.search(r"(?:^|_)w(\d+)(?:_|$)", text)
    return int(match.group(1)) if match else int(default)


def _parse_bits(text: str, *, prefix: str = "b", default: int = 4) -> int:
    match = re.search(rf"(?:^|_){re.escape(prefix)}(\d+)(?:_|$)", text)
    return int(match.group(1)) if match else int(default)


def _parse_tq_bits(text: str) -> tuple[int, int]:
    match = re.search(r"(?:^|_)k(\d+)v(\d+)(?:_|$)", text)
    if not match:
        return 4, 4
    return int(match.group(1)), int(match.group(2))


def _parse_pq(text: str) -> tuple[int, int]:
    match = re.search(r"(?:^|_)s(\d+)b(\d+)(?:_|$)", text)
    if not match:
        return 4, 4
    return int(match.group(1)), int(match.group(2))


def _parse_count(text: str, *, prefix: str, default: int) -> int:
    match = re.search(rf"(?:^|_){re.escape(prefix)}(\d+)(?:_|$)", text)
    return int(match.group(1)) if match else int(default)


def _parse_pair(text: str, *, first: str, second: str, defaults: tuple[int, int]) -> tuple[int, int]:
    pattern = rf"(?:^|_){re.escape(first)}(\d+){re.escape(second)}(\d+)(?:_|$)"
    match = re.search(pattern, text)
    if match:
        return int(match.group(1)), int(match.group(2))
    return int(defaults[0]), int(defaults[1])


def _parse_clip(text: str, default: float = 0.1) -> float:
    match = re.search(r"(?:^|_)clip([0-9p.]+)(?:_|$)", text)
    if not match:
        return float(default)
    return float(match.group(1).replace("p", "."))


def _parse_float_param(text: str, *, prefix: str, default: float) -> float:
    match = re.search(rf"(?:^|_){re.escape(prefix)}([0-9p.]+)(?:_|$)", text)
    if not match:
        return float(default)
    return float(match.group(1).replace("p", "."))


def _parse_binary_param(text: str, *, prefix: str, default: bool) -> bool:
    match = re.search(rf"(?:^|_){re.escape(prefix)}([01])(?:_|$)", text)
    if not match:
        return bool(default)
    return bool(int(match.group(1)))


def _parse_stage_bits(text: str, *, default_stages: int = 4, default_bits: int = 6) -> tuple[int, int]:
    match = re.search(r"(?:^|_)m(\d+)b(\d+)(?:_|$)", text)
    if not match:
        return int(default_stages), int(default_bits)
    return int(match.group(1)), int(match.group(2))


def _parse_dictionary_active(text: str, *, default_atoms: int = 256, default_active: int = 4) -> tuple[int, int]:
    match = re.search(r"(?:^|_)d(\d+)a(\d+)(?:_|$)", text)
    if not match:
        return int(default_atoms), int(default_active)
    return int(match.group(1)), int(match.group(2))


def _parse_observation_window(text: str, default: int = 64) -> int:
    match = re.search(r"(?:^|_)obs(\d+)(?:_|$)", text)
    return int(match.group(1)) if match else int(default)


def _parse_kernel_size(text: str, default: int = 5) -> int:
    match = re.search(r"(?:^|_)ker(\d+)(?:_|$)", text)
    return int(match.group(1)) if match else int(default)


def _quantize_per_token(x: np.ndarray, bits: int) -> np.ndarray:
    x32 = x.astype(np.float32, copy=False)
    if x32.size == 0:
        return x32.copy()
    levels = float((1 << int(bits)) - 1)
    lo = x32.min(axis=1, keepdims=True)
    hi = x32.max(axis=1, keepdims=True)
    scale = np.maximum((hi - lo) / max(levels, 1.0), np.float32(1e-8))
    codes = np.clip(np.rint((x32 - lo) / scale), 0.0, levels)
    return (codes * scale + lo).astype(np.float32, copy=False)


def _quantize_per_token_with_cost(x: np.ndarray, bits: int, metadata_bytes: int) -> tuple[np.ndarray, float, float]:
    x_hat = _quantize_per_token(x, bits)
    if x.shape[0] == 0:
        return x_hat, 0.0, 0.0
    dim = int(x.shape[1])
    total = _packed_code_bytes(x.shape[0], dim, bits) + _scale_zero_bytes(x.shape[0], 1, int(metadata_bytes))
    per_token = _packed_code_bytes(1, dim, bits) + _scale_zero_bytes(1, 1, int(metadata_bytes))
    return x_hat, float(total), float(per_token)


def _quantize_groupwise_lastdim(x: np.ndarray, group_size: int, bits: int) -> np.ndarray:
    x32 = x.astype(np.float32, copy=False)
    if x32.size == 0 or int(bits) >= 16:
        return x32.copy()
    dim = int(x32.shape[-1])
    group_size = max(1, min(int(group_size), dim))
    if dim % group_size != 0:
        group_size = dim
    groups = dim // group_size
    view = x32.reshape(-1, groups, group_size)
    lo = view.min(axis=-1, keepdims=True)
    hi = view.max(axis=-1, keepdims=True)
    levels = float((1 << int(bits)) - 1)
    scale = np.maximum((hi - lo) / max(levels, 1.0), np.float32(1e-8))
    codes = np.clip(np.rint((view - lo) / scale), 0.0, levels)
    return (codes * scale + lo).reshape(x32.shape).astype(np.float32, copy=False)


def _groupwise_lastdim_cost(count: int, dim: int, group_size: int, bits: int, metadata_bytes: int) -> float:
    if int(bits) >= 16:
        return float(int(count) * int(dim) * 2)
    group_size = max(1, min(int(group_size), int(dim)))
    if int(dim) % group_size != 0:
        group_size = int(dim)
    groups = int(dim) // group_size
    return _packed_code_bytes(count, dim, bits) + _scale_zero_bytes(count, groups, int(metadata_bytes))


def _pmkvq_reconstruct_with_cost(
    x: np.ndarray,
    *,
    bits: int,
    group_size: int,
    metadata_bytes: int,
) -> tuple[np.ndarray, float, float]:
    x_hat = _quantize_groupwise_lastdim(x, group_size, bits)
    if x.shape[0] == 0:
        return x_hat, 0.0, 0.0
    cost = _groupwise_lastdim_cost(x.shape[0], x.shape[1], group_size, bits, metadata_bytes)
    per_token = _groupwise_lastdim_cost(1, x.shape[1], group_size, bits, metadata_bytes)
    return x_hat, float(cost), float(per_token)


def _mean_centered_group_quantize_with_cost(
    x: np.ndarray,
    *,
    bits: int,
    group_size: int,
    metadata_bytes: int,
) -> tuple[np.ndarray, float, float]:
    x32 = x.astype(np.float32, copy=False)
    if x32.shape[0] == 0:
        return x32.copy(), 0.0, 0.0
    mean = np.mean(x32, axis=0, keepdims=True, dtype=np.float64).astype(np.float32, copy=False)
    centered_hat = _quantize_groupwise_lastdim(x32 - mean, group_size, bits)
    out = (centered_hat + mean).astype(np.float32, copy=False)
    mean_bytes = float(x32.shape[1] * 2)
    cost = _groupwise_lastdim_cost(x32.shape[0], x32.shape[1], group_size, bits, metadata_bytes) + mean_bytes
    per_token = _groupwise_lastdim_cost(1, x32.shape[1], group_size, bits, metadata_bytes)
    return out, float(cost), float(per_token)


def _quantize_per_channel_with_cost(
    x: np.ndarray,
    bits: int,
    *,
    metadata_bytes: int,
) -> tuple[np.ndarray, float, float]:
    x32 = x.astype(np.float32, copy=False)
    if x32.shape[0] == 0:
        return x32.copy(), 0.0, 0.0
    if int(bits) >= 16:
        cost = float(x32.shape[0] * x32.shape[1] * 2)
        return x32.copy(), cost, float(x32.shape[1] * 2)
    out = _quantize_per_channel(x32, int(bits))
    cost = _packed_code_bytes(x32.shape[0], x32.shape[1], int(bits)) + _scale_zero_bytes(1, x32.shape[1], int(metadata_bytes))
    per_token = _packed_code_bytes(1, x32.shape[1], int(bits))
    return out, float(cost), float(per_token)


def _kivi_quantize_key_time_groups(x: np.ndarray, group_size: int, bits: int) -> np.ndarray:
    """KIVI key quantization: per channel, grouped along token/time."""
    x32 = x.astype(np.float32, copy=False)
    if x32.size == 0:
        return x32.copy()
    n, dim = int(x32.shape[0]), int(x32.shape[1])
    group_size = int(group_size)
    if group_size <= 0 or n % group_size != 0:
        raise ValueError(f"KIVI key quantized length {n} must be divisible by group_size {group_size}")
    levels = float((1 << int(bits)) - 1)
    view = x32.reshape(n // group_size, group_size, dim)
    lo = view.min(axis=1, keepdims=True)
    hi = view.max(axis=1, keepdims=True)
    scale = np.maximum((hi - lo) / max(levels, 1.0), np.float32(1e-8))
    codes = np.clip(np.rint((view - lo) / scale), 0.0, levels)
    return (codes * scale + lo).reshape(x32.shape).astype(np.float32, copy=False)


def _kivi_quantize_value_channel_groups(x: np.ndarray, group_size: int, bits: int) -> np.ndarray:
    """KIVI value quantization: per token, grouped along channel/head-dim."""
    return _quantize_groupwise_lastdim(x, group_size=int(group_size), bits=int(bits))


def _build_kivi_with_cost(
    keys: np.ndarray,
    values: np.ndarray,
    *,
    bits: int,
    group_size: int,
    residual_window: int,
    key_bytes: int,
    value_bytes: int,
    metadata_bytes: int,
) -> tuple[np.ndarray, np.ndarray, float, float, dict[str, int | float]]:
    """KIVI cache state for a single KV head snapshot.

    The official implementation stores quantized K as [tokens grouped along time] per
    key channel, while V is quantized per token along channel groups. K keeps only a
    partially-filled residual chunk exact; V keeps the most recent residual window exact.
    """
    if int(bits) not in (2, 4):
        raise ValueError("KIVI supports the official 2/4-bit settings")
    n, dim = int(keys.shape[0]), int(keys.shape[1])
    group = int(group_size)
    residual = int(residual_window)
    if residual <= 0:
        raise ValueError("KIVI requires a positive residual window")
    if group <= 0 or dim % group != 0 or residual % group != 0:
        raise ValueError(
            f"KIVI requires head_dim and residual_window divisible by group_size; "
            f"got dim={dim}, residual={residual}, group={group}"
        )

    keys32 = keys.astype(np.float32, copy=False)
    values32 = values.astype(np.float32, copy=False)
    keys_hat = keys32.copy()
    values_hat = values32.copy()

    key_quant_len = n - (n % residual)
    key_exact_count = n - key_quant_len
    if key_quant_len > 0:
        keys_hat[:key_quant_len] = _kivi_quantize_key_time_groups(keys32[:key_quant_len], group, bits)

    value_exact_count = min(residual, n)
    value_quant_len = max(0, n - value_exact_count)
    if value_quant_len > 0:
        values_hat[:value_quant_len] = _kivi_quantize_value_channel_groups(values32[:value_quant_len], group, bits)

    key_groups = key_quant_len // group
    value_groups = dim // group
    key_code_bytes = _packed_code_bytes(key_quant_len, dim, bits)
    key_meta_bytes = _scale_zero_bytes(dim, key_groups, int(metadata_bytes))
    value_code_bytes = _packed_code_bytes(value_quant_len, dim, bits)
    value_meta_bytes = _scale_zero_bytes(value_quant_len, value_groups, int(metadata_bytes))
    exact_key_bytes = float(key_exact_count * dim * int(key_bytes))
    exact_value_bytes = float(value_exact_count * dim * int(value_bytes))

    key_sidecar_per_token = _packed_code_bytes(1, dim, bits) + float(dim * 2 * int(metadata_bytes)) / float(group)
    value_sidecar_per_token = _packed_code_bytes(1, dim, bits) + _scale_zero_bytes(1, value_groups, int(metadata_bytes))
    exact_append_per_token = float(dim * (int(key_bytes) + int(value_bytes)))
    update_bytes_per_token = exact_append_per_token + key_sidecar_per_token + value_sidecar_per_token

    query_bytes = (
        exact_key_bytes
        + exact_value_bytes
        + key_code_bytes
        + key_meta_bytes
        + value_code_bytes
        + value_meta_bytes
    )
    metadata = {
        "key_quantized_tokens": int(key_quant_len),
        "key_exact_tokens": int(key_exact_count),
        "key_groups": int(key_groups),
        "value_quantized_tokens": int(value_quant_len),
        "value_exact_tokens": int(value_exact_count),
        "value_groups_per_token": int(value_groups),
        "exact_key_MB": exact_key_bytes / MB,
        "exact_value_MB": exact_value_bytes / MB,
        "compressed_key_MB": float(key_code_bytes + key_meta_bytes) / MB,
        "compressed_value_MB": float(value_code_bytes + value_meta_bytes) / MB,
        "key_sidecar_update_MB_per_token": float(key_sidecar_per_token) / MB,
        "value_sidecar_update_MB_per_token": float(value_sidecar_per_token) / MB,
    }
    return keys_hat, values_hat, float(query_bytes), float(update_bytes_per_token), metadata


def _tiered_quantize_with_cost(
    x: np.ndarray,
    scores: np.ndarray,
    *,
    low_bits: int,
    mid_bits: int,
    high_bits: int,
    high_count: int,
    mid_count: int,
    metadata_bytes: int,
) -> tuple[np.ndarray, float, float, dict[str, int]]:
    """Mixed-precision token quantization: high-score tokens get more bits."""
    x32 = x.astype(np.float32, copy=False)
    n = int(x32.shape[0])
    if n == 0:
        return x32.copy(), 0.0, 0.0, {"high_tokens": 0, "mid_tokens": 0, "low_tokens": 0}
    score_arr = np.asarray(scores, dtype=np.float32)
    order = np.argsort(score_arr)[::-1]
    high_n = max(0, min(int(high_count), n))
    mid_n = max(0, min(int(mid_count), n - high_n))
    high_idx = order[:high_n]
    mid_idx = order[high_n : high_n + mid_n]
    low_idx = order[high_n + mid_n :]
    out = np.empty_like(x32, dtype=np.float32)
    total_cost = 0.0
    total_per_token = 0.0
    for idx, bits in ((high_idx, int(high_bits)), (mid_idx, int(mid_bits)), (low_idx, int(low_bits))):
        if idx.size == 0:
            continue
        quant, cost, per_token = _quantize_per_token_with_cost(x32[idx], bits, metadata_bytes)
        out[idx] = quant
        total_cost += float(cost)
        total_per_token += float(idx.size) * float(per_token)
    # Store two compact cut points / tier IDs. The per-token tier sidecar is two bits.
    tier_cost = _packed_code_bytes(n, 1, 2)
    total_cost += tier_cost
    total_per_token += tier_cost / max(1, n)
    meta = {"high_tokens": int(high_n), "mid_tokens": int(mid_n), "low_tokens": int(low_idx.size)}
    return out.astype(np.float32, copy=False), float(total_cost), float(total_per_token / max(1, n)), meta


def _sparse_channel_reconstruct_with_cost(
    x: np.ndarray,
    *,
    keep_frac: float,
    bits: int,
    metadata_bytes: int,
    keep_mean: bool,
) -> tuple[np.ndarray, float, float, int]:
    """LOOKAT-style unstructured per-token channel sparsity proxy."""
    x32 = x.astype(np.float32, copy=False)
    n, dim = int(x32.shape[0]), int(x32.shape[1]) if x32.ndim == 2 else 0
    if n == 0 or dim == 0:
        return x32.copy(), 0.0, 0.0, 0
    keep = max(1, min(dim, int(round(float(keep_frac) * dim))))
    baseline = (
        np.mean(x32, axis=0, keepdims=True, dtype=np.float64).astype(np.float32, copy=False)
        if bool(keep_mean)
        else np.zeros((1, dim), dtype=np.float32)
    )
    out = np.broadcast_to(baseline, x32.shape).copy()
    idx = np.argpartition(np.abs(x32 - baseline), -keep, axis=1)[:, -keep:]
    rows = np.arange(n)[:, None]
    selected = x32[rows, idx].reshape(n, keep)
    if int(bits) < 16:
        selected_hat = _quantize_per_token(selected, int(bits))
    else:
        selected_hat = selected.astype(np.float32, copy=False)
    out[rows, idx] = selected_hat
    bitmap_bytes = _packed_code_bytes(n, dim, 1)
    value_bytes = float(n * keep * (2 if int(bits) >= 16 else float(bits) / 8.0))
    meta_bytes = float(n * 2 * int(metadata_bytes)) if int(bits) < 16 else 0.0
    mean_bytes = float(dim * 2) if bool(keep_mean) else 0.0
    total = bitmap_bytes + value_bytes + meta_bytes + mean_bytes
    per_token = _packed_code_bytes(1, dim, 1) + float(keep * (2 if int(bits) >= 16 else float(bits) / 8.0))
    if int(bits) < 16:
        per_token += float(2 * int(metadata_bytes))
    return out.astype(np.float32, copy=False), float(total), float(per_token), int(keep)


def _pca_transform_quantize_with_cost(
    x: np.ndarray,
    *,
    rank: int,
    bits: int,
    metadata_bytes: int,
) -> tuple[np.ndarray, float, float]:
    """KVTC-style PCA decorrelation, coefficient truncation, and scalar coding."""
    x32 = x.astype(np.float32, copy=False)
    if x32.shape[0] == 0:
        return x32.copy(), 0.0, 0.0
    n, dim = int(x32.shape[0]), int(x32.shape[1])
    r = max(1, min(int(rank), dim))
    mean = np.mean(x32, axis=0, keepdims=True, dtype=np.float64).astype(np.float32, copy=False)
    centered = x32 - mean
    cov = (centered.T.astype(np.float64, copy=False) @ centered.astype(np.float64, copy=False)) / max(1, n)
    eigvals, eigvecs = np.linalg.eigh(cov)
    basis = eigvecs[:, np.argsort(eigvals)[-r:]].astype(np.float32, copy=False)
    coeff = (centered @ basis).astype(np.float32, copy=False)
    coeff_hat, coeff_cost, coeff_per_token = _quantize_per_channel_with_cost(
        coeff,
        int(bits),
        metadata_bytes=int(metadata_bytes),
    )
    out = (coeff_hat @ basis.T + mean).astype(np.float32, copy=False)
    basis_bytes = float((dim * r + dim) * 2)
    return out, float(coeff_cost + basis_bytes), float(coeff_per_token)


def _dct_matrix(dim: int) -> np.ndarray:
    n = np.arange(int(dim), dtype=np.float64)
    k = np.arange(int(dim), dtype=np.float64)[:, None]
    mat = np.cos(np.pi / float(dim) * (n + 0.5) * k)
    mat[0, :] *= np.sqrt(1.0 / float(dim))
    if dim > 1:
        mat[1:, :] *= np.sqrt(2.0 / float(dim))
    return mat.astype(np.float32)


def _dct_transform_quantize_with_cost(
    x: np.ndarray,
    *,
    rank: int,
    bits: int,
    metadata_bytes: int,
) -> tuple[np.ndarray, float, float]:
    """FreqKV-style fixed transform coding over channel dimension."""
    x32 = x.astype(np.float32, copy=False)
    if x32.shape[0] == 0:
        return x32.copy(), 0.0, 0.0
    n, dim = int(x32.shape[0]), int(x32.shape[1])
    r = max(1, min(int(rank), dim))
    dct = _dct_matrix(dim)
    coeff = (x32 @ dct.T).astype(np.float32, copy=False)
    kept = coeff[:, :r]
    kept_hat, coeff_cost, coeff_per_token = _quantize_per_channel_with_cost(
        kept,
        int(bits),
        metadata_bytes=int(metadata_bytes),
    )
    coeff_hat = np.zeros_like(coeff, dtype=np.float32)
    coeff_hat[:, :r] = kept_hat
    out = (coeff_hat @ dct).astype(np.float32, copy=False)
    return out, float(coeff_cost), float(coeff_per_token)


def _kitty_key_quantize_pages(
    x: np.ndarray,
    *,
    buffer_length: int,
    group_size: int,
    bits: int,
    promote_ratio: float,
    promote_bit: int,
    metadata_bytes: int,
) -> tuple[np.ndarray, float, float, int]:
    """Kitty quantizes K as [channel, token] groups and promotes high-magnitude channels."""
    x32 = x.astype(np.float32, copy=False)
    if x32.shape[0] == 0 or int(bits) >= 16:
        return x32.copy(), float(x32.shape[0] * x32.shape[1] * 2), float(x32.shape[1] * 2), 0
    out = x32.copy()
    n, dim = int(x32.shape[0]), int(x32.shape[1])
    page = max(1, int(buffer_length))
    group_size = max(1, int(group_size))
    promote_count = max(0, min(dim, int(dim * float(promote_ratio) + 1e-6)))
    total_bytes = 0.0
    total_promoted = 0
    for start in range(0, n, page):
        stop = min(n, start + page)
        block = x32[start:stop]
        if block.shape[0] == 0:
            continue
        promote = np.zeros((dim,), dtype=bool)
        if promote_count > 0:
            channel_score = np.mean(np.abs(block), axis=0)
            chosen = np.argpartition(channel_score, -promote_count)[-promote_count:]
            promote[chosen] = True
            total_promoted += int(chosen.size)
        block_t = block.T  # [D, T], groupwise over tokens.
        quant_t = block_t.copy()
        page_len = int(block_t.shape[1])
        g = group_size if page_len % group_size == 0 else page_len
        groups = page_len // g
        for bit_width, mask in ((int(promote_bit), promote), (int(bits), ~promote)):
            if not np.any(mask):
                continue
            if bit_width >= 16:
                continue
            view = block_t[mask].reshape(-1, groups, g)
            lo = view.min(axis=-1, keepdims=True)
            hi = view.max(axis=-1, keepdims=True)
            levels = float((1 << bit_width) - 1)
            scale = np.maximum((hi - lo) / max(levels, 1.0), np.float32(1e-8))
            codes = np.clip(np.rint((view - lo) / scale), 0.0, levels)
            quant_t[mask] = (codes * scale + lo).reshape(int(np.count_nonzero(mask)), page_len)
        out[start:stop] = quant_t.T
        low_channels = dim - promote_count
        total_bytes += float(page_len * (low_channels * int(bits) + promote_count * int(promote_bit))) / 8.0
        total_bytes += float(dim * groups * 2 * int(metadata_bytes))
    per_token = total_bytes / max(1, n)
    return out.astype(np.float32, copy=False), float(total_bytes), float(per_token), int(total_promoted)


def _zeromerge_snapshot(
    keys: np.ndarray,
    values: np.ndarray,
    obs_queries: np.ndarray | None,
    base_mask: np.ndarray,
    *,
    budget: int,
    tail: int,
    dense: int,
    obs_window: int,
    kernel_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, dict[str, int | float]]:
    """Snapshot version of ZeroMerge's top-cache + averaged residual buckets + recent tail."""
    keys32 = keys.astype(np.float32, copy=False)
    values32 = values.astype(np.float32, copy=False)
    n, dim = int(keys32.shape[0]), int(keys32.shape[1])
    budget = max(1, min(int(budget), n))
    tail = max(0, min(int(tail), budget - 1))
    dense = max(0, min(int(dense), budget - tail))
    top_budget = max(0, budget - tail - dense)
    active_tail = np.zeros((n,), dtype=bool)
    if tail:
        active_tail[max(0, n - tail) :] = True
    active_base = np.asarray(base_mask, dtype=bool).copy()
    active_base |= active_tail
    scores = _attention_observation_scores(keys32, obs_queries, mode="mean", kernel_size=kernel_size)
    if int(obs_window) > 0:
        scores[max(0, n - int(obs_window)) :] += 1.0

    candidates = np.flatnonzero(~active_base)
    top_idx = np.empty((0,), dtype=np.int64)
    if top_budget > 0 and candidates.size > 0:
        take = min(top_budget, int(candidates.size))
        top_idx = candidates[np.argpartition(scores[candidates], -take)[-take:]]
    top_idx.sort()
    used = active_base.copy()
    used[top_idx] = True

    remaining = np.flatnonzero(~used)
    merged_keys: list[np.ndarray] = []
    merged_values: list[np.ndarray] = []
    merged_weights: list[float] = []
    merged_source = 0
    if dense > 0 and remaining.size > 0:
        tail_max = min(int(remaining.size), max(dense, dense * 10))
        chosen = remaining[np.argpartition(scores[remaining], -tail_max)[-tail_max:]]
        chosen.sort()
        splits = np.array_split(chosen, dense)
        for group in splits:
            if group.size == 0:
                continue
            weights = np.ones((group.size,), dtype=np.float32)
            merged_keys.append(np.average(keys32[group], axis=0, weights=weights).astype(np.float32, copy=False))
            merged_values.append(np.average(values32[group], axis=0, weights=weights).astype(np.float32, copy=False))
            merged_weights.append(float(group.size))
            merged_source += int(group.size)

    exact_idx = np.flatnonzero(active_base | np.isin(np.arange(n), top_idx))
    exact_idx.sort()
    out_keys = [arr for arr in merged_keys]
    out_values = [arr for arr in merged_values]
    out_weights = [float(w) for w in merged_weights]
    if exact_idx.size:
        out_keys.extend([row for row in keys32[exact_idx]])
        out_values.extend([row for row in values32[exact_idx]])
        out_weights.extend([1.0] * int(exact_idx.size))
    if out_keys:
        key_out = np.stack(out_keys, axis=0).astype(np.float32, copy=False)
        value_out = np.stack(out_values, axis=0).astype(np.float32, copy=False)
        weights_out = np.asarray(out_weights, dtype=np.float32)
    else:
        key_out = keys32[-1:].copy()
        value_out = values32[-1:].copy()
        weights_out = np.ones((1,), dtype=np.float32)
    metadata = {
        "retained_tokens": int(key_out.shape[0]),
        "exact_tokens": int(exact_idx.size),
        "merged_clusters": int(len(merged_keys)),
        "merged_source_tokens": int(merged_source),
        "dropped_tokens": int(n - exact_idx.size - merged_source),
    }
    weight_bytes = float(key_out.shape[0] * 2)
    read_bytes = float(key_out.shape[0] * dim * 4 + weight_bytes)
    return key_out, value_out, weights_out, read_bytes, metadata


def _cam_snapshot(
    keys: np.ndarray,
    values: np.ndarray,
    obs_queries: np.ndarray | None,
    base_mask: np.ndarray,
    *,
    budget: int,
    merge_budget: int,
    obs_window: int,
    kernel_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, dict[str, int | float]]:
    """Deterministic snapshot proxy for CaM's merge-before-prune idea."""
    keys32 = keys.astype(np.float32, copy=False)
    values32 = values.astype(np.float32, copy=False)
    n, dim = int(keys32.shape[0]), int(keys32.shape[1])
    scores = _attention_observation_scores(keys32, obs_queries, mode="mean", kernel_size=int(kernel_size))
    if int(obs_window) > 0:
        scores[max(0, n - int(obs_window)) :] += 1.0
    active = _scores_retention_mask(scores, base_mask, budget=int(budget))
    kept = np.flatnonzero(active)
    evicted = np.flatnonzero(~active)
    if kept.size == 0:
        kept = np.asarray([n - 1], dtype=np.int64)
        active[kept] = True
        evicted = np.flatnonzero(~active)
    kept.sort()
    values_mod = values32.copy()
    merge_budget = max(1, int(merge_budget))
    merged = 0
    total_edges = 0
    if evicted.size:
        # CaM merges relatively important evicted tokens into sequential kept neighbors.
        merge_count = min(int(obs_window) if int(obs_window) > 0 else evicted.size, int(evicted.size))
        merge_src = evicted[np.argsort(scores[evicted])[-merge_count:]]
        merge_src.sort()
        for src in merge_src:
            pos = int(np.searchsorted(kept, src, side="right"))
            targets = kept[pos : pos + merge_budget]
            if targets.size == 0:
                targets = kept[max(0, pos - merge_budget) : pos]
            if targets.size == 0:
                continue
            values_mod[targets] += values32[src][None, :] / float(targets.size)
            merged += 1
            total_edges += int(targets.size)
    weights = np.ones((kept.size,), dtype=np.float32)
    read_bytes = float(kept.size * dim * (2 + 2))
    metadata = {
        "retained_tokens": int(kept.size),
        "exact_tokens": int(kept.size),
        "merged_source_tokens": int(merged),
        "merge_edges": int(total_edges),
        "dropped_tokens": int(n - kept.size - merged),
    }
    return keys32[kept], values_mod[kept], weights, read_bytes, metadata


def _quantize_per_channel(x: np.ndarray, bits: int, *, clip_percent: float = 0.0) -> np.ndarray:
    x32 = x.astype(np.float32, copy=False)
    if x32.size == 0:
        return x32.copy()
    if float(clip_percent) > 0.0:
        pct = max(0.0, min(float(clip_percent), 49.0))
        lo = np.percentile(x32, pct, axis=0, keepdims=True).astype(np.float32, copy=False)
        hi = np.percentile(x32, 100.0 - pct, axis=0, keepdims=True).astype(np.float32, copy=False)
    else:
        lo = x32.min(axis=0, keepdims=True)
        hi = x32.max(axis=0, keepdims=True)
    levels = float((1 << int(bits)) - 1)
    scale = np.maximum((hi - lo) / max(levels, 1.0), np.float32(1e-8))
    clipped = np.minimum(np.maximum(x32, lo), hi)
    codes = np.clip(np.rint((clipped - lo) / scale), 0.0, levels)
    return (codes * scale + lo).astype(np.float32, copy=False)


def _pq_reconstruct(x: np.ndarray, *, subvecs: int, subbits: int, seed: int, max_iter: int) -> tuple[np.ndarray, float]:
    x32 = x.astype(np.float32, copy=False)
    if x32.shape[0] == 0:
        return x32.copy(), 0.0
    codebooks, codes, actual_subvecs, centroids = build_pq_index(
        x32,
        0,
        x32.shape[0],
        subvecs=int(subvecs),
        subbits=int(subbits),
        seed=int(seed),
        max_iter=int(max_iter),
    )
    subdim = int(x32.shape[1]) // int(actual_subvecs)
    out = np.zeros_like(x32, dtype=np.float32)
    for sub in range(int(actual_subvecs)):
        out[:, sub * subdim : (sub + 1) * subdim] = codebooks[sub, codes[:, sub].astype(np.int64, copy=False)]
    codebook_bytes = float(int(actual_subvecs) * int(centroids) * int(subdim) * 2)
    code_bytes = _packed_code_bytes(x32.shape[0], int(actual_subvecs), int(subbits))
    return out, codebook_bytes + code_bytes


def _lowrank_reconstruct(x: np.ndarray, *, rank: int) -> tuple[np.ndarray, float]:
    x32 = x.astype(np.float32, copy=False)
    if x32.shape[0] == 0:
        return x32.copy(), 0.0
    dim = int(x32.shape[1])
    r = max(1, min(int(rank), dim))
    if r >= dim:
        return x32.copy(), float(x32.shape[0] * dim * 2)

    mean = x32.mean(axis=0, keepdims=True, dtype=np.float64).astype(np.float32, copy=False)
    centered = x32 - mean
    # Compute the small covariance eigenproblem instead of a large token-token SVD.
    cov = (centered.T.astype(np.float64, copy=False) @ centered.astype(np.float64, copy=False)) / max(1, x32.shape[0])
    eigvals, eigvecs = np.linalg.eigh(cov)
    basis = eigvecs[:, np.argsort(eigvals)[-r:]].astype(np.float32, copy=False)
    coeff = centered @ basis
    out = (coeff @ basis.T + mean).astype(np.float32, copy=False)
    basis_bytes = float((dim * r + dim) * 2)
    coeff_bytes = float(x32.shape[0] * r * 2)
    return out, basis_bytes + coeff_bytes


def _gear_reconstruct(
    x: np.ndarray,
    *,
    bits: int,
    rank: int,
    sparse_frac: float,
    metadata_bytes: int,
    value_bytes: int,
) -> tuple[np.ndarray, float, float, int]:
    x32 = x.astype(np.float32, copy=False)
    if x32.shape[0] == 0:
        return x32.copy(), 0.0, 0.0, 0
    quant, quant_cost, quant_per_token = _quantize_per_token_with_cost(x32, bits, metadata_bytes)
    residual = x32 - quant
    lowrank = np.zeros_like(x32, dtype=np.float32)
    lowrank_cost = 0.0
    if int(rank) > 0:
        lowrank, lowrank_cost = _lowrank_reconstruct(residual, rank=int(rank))
    out = quant + lowrank
    remaining = x32 - out
    sparse_frac = max(0.0, float(sparse_frac))
    nnz = min(int(np.ceil(float(remaining.size) * sparse_frac)), int(remaining.size))
    sparse_cost = 0.0
    if nnz > 0:
        flat = np.abs(remaining).reshape(-1)
        chosen = np.argpartition(flat, -nnz)[-nnz:]
        out_flat = out.reshape(-1)
        rem_flat = remaining.reshape(-1)
        out_flat[chosen] += rem_flat[chosen]
        # GEAR stores sparse error outliers. Charge value plus compact row/dim index.
        sparse_cost = float(nnz * (int(value_bytes) + 4))
    total_cost = float(quant_cost + lowrank_cost + sparse_cost)
    per_token = float(quant_per_token + (2 * max(0, int(rank)) * 2) + sparse_frac * x32.shape[1] * (int(value_bytes) + 4))
    return out.astype(np.float32, copy=False), total_cost, per_token, int(nnz)


def _residual_vq_reconstruct(
    x: np.ndarray,
    *,
    stages: int,
    bits: int,
    seed: int,
    max_iter: int,
) -> tuple[np.ndarray, float]:
    x32 = x.astype(np.float32, copy=False)
    if x32.shape[0] == 0:
        return x32.copy(), 0.0
    out = np.zeros_like(x32, dtype=np.float32)
    residual = x32.copy()
    total_cost = 0.0
    for stage in range(max(1, int(stages))):
        codebooks, codes, actual_subvecs, centroids = build_pq_index(
            residual,
            0,
            residual.shape[0],
            subvecs=1,
            subbits=int(bits),
            seed=int(seed) + 104729 * stage,
            max_iter=int(max_iter),
        )
        stage_hat = codebooks[0, codes[:, 0].astype(np.int64, copy=False)].astype(np.float32, copy=False)
        out += stage_hat
        residual -= stage_hat
        total_cost += float(int(actual_subvecs) * int(centroids) * int(x32.shape[1]) * 2)
        total_cost += _packed_code_bytes(x32.shape[0], 1, int(bits))
    return out.astype(np.float32, copy=False), float(total_cost)


def _sparse_code_reconstruct(
    x: np.ndarray,
    *,
    atoms: int,
    active: int,
    seed: int,
    max_iter: int,
) -> tuple[np.ndarray, float]:
    x32 = x.astype(np.float32, copy=False)
    if x32.shape[0] == 0:
        return x32.copy(), 0.0
    atoms_pow2 = 1 << int(np.ceil(np.log2(max(2, int(atoms)))))
    bits = int(np.log2(atoms_pow2))
    codebooks, _, _, centroids = build_pq_index(
        x32,
        0,
        x32.shape[0],
        subvecs=1,
        subbits=bits,
        seed=int(seed),
        max_iter=int(max_iter),
    )
    dictionary = codebooks[0].astype(np.float32, copy=False)
    dictionary = dictionary[: int(atoms)]
    norms = np.linalg.norm(dictionary, axis=1, keepdims=True).astype(np.float32)
    dictionary = dictionary / np.maximum(norms, np.float32(1e-8))
    residual = x32.copy()
    out = np.zeros_like(x32, dtype=np.float32)
    for _ in range(max(1, int(active))):
        scores = residual.astype(np.float32, copy=False) @ dictionary.T
        chosen = np.argmax(np.abs(scores), axis=1).astype(np.int64, copy=False)
        coeff = scores[np.arange(scores.shape[0]), chosen].astype(np.float32, copy=False)
        update = coeff[:, None] * dictionary[chosen]
        out += update
        residual -= update
    dictionary_bytes = float(dictionary.shape[0] * dictionary.shape[1] * 2)
    # Lexico stores CSR int16 atom indices and float8 coefficients, plus row pointers.
    code_bytes = float(x32.shape[0] * int(active) * (2 + 1) + (x32.shape[0] + 1) * 4)
    return out.astype(np.float32, copy=False), dictionary_bytes + code_bytes


def _pool_scores_1d(scores: np.ndarray, kernel_size: int) -> np.ndarray:
    k = max(1, int(kernel_size))
    if k <= 1 or scores.size == 0:
        return scores.astype(np.float32, copy=False)
    pad = k // 2
    padded = np.pad(scores.astype(np.float32, copy=False), (pad, pad), mode="edge")
    out = np.empty_like(scores, dtype=np.float32)
    for idx in range(scores.shape[0]):
        out[idx] = float(np.mean(padded[idx : idx + k]))
    return out


def _max_pool_scores_1d(scores: np.ndarray, kernel_size: int) -> np.ndarray:
    k = max(1, int(kernel_size))
    if k <= 1 or scores.size == 0:
        return scores.astype(np.float32, copy=False)
    pad = k // 2
    padded = np.pad(scores.astype(np.float32, copy=False), (pad, pad), mode="edge")
    out = np.empty_like(scores, dtype=np.float32)
    for idx in range(scores.shape[0]):
        out[idx] = float(np.max(padded[idx : idx + k]))
    return out


def _zscore(scores: np.ndarray) -> np.ndarray:
    x = scores.astype(np.float32, copy=False)
    if x.size == 0:
        return x.copy()
    mean = float(np.mean(x))
    std = float(np.std(x))
    return ((x - mean) / max(std, 1e-6)).astype(np.float32, copy=False)


def _scores_retention_mask(scores: np.ndarray, base_mask: np.ndarray, budget: int) -> np.ndarray:
    active = np.asarray(base_mask, dtype=bool).copy()
    budget = min(max(0, int(budget)), int(active.shape[0]))
    remaining_budget = max(0, budget - int(np.count_nonzero(active)))
    if remaining_budget <= 0:
        return active
    candidates = np.flatnonzero(~active)
    if candidates.size == 0:
        return active
    score_arr = np.asarray(scores, dtype=np.float32)
    take = min(remaining_budget, int(candidates.size))
    chosen = candidates[np.argpartition(score_arr[candidates], -take)[-take:]]
    active[chosen] = True
    return active


def _attention_observation_scores(
    keys: np.ndarray,
    obs_queries: np.ndarray | None,
    *,
    mode: str,
    kernel_size: int,
) -> np.ndarray:
    keys32 = keys.astype(np.float32, copy=False)
    if obs_queries is None or obs_queries.size == 0 or keys32.shape[0] == 0:
        return np.zeros((keys32.shape[0],), dtype=np.float32)
    queries = obs_queries.reshape(-1, obs_queries.shape[-1]).astype(np.float32, copy=False)
    scores = (queries @ keys32.T) * (1.0 / np.sqrt(float(keys32.shape[-1])))
    scores = scores.astype(np.float64, copy=False)
    scores -= np.max(scores, axis=1, keepdims=True)
    probs = np.exp(scores)
    probs /= np.maximum(np.sum(probs, axis=1, keepdims=True), 1e-20)
    if str(mode) == "max":
        token_scores = np.max(probs, axis=0)
    elif str(mode) == "sum":
        token_scores = np.sum(probs, axis=0)
    else:
        token_scores = np.mean(probs, axis=0)
    return _pool_scores_1d(token_scores.astype(np.float32, copy=False), kernel_size)


def _keydiff_scores(keys: np.ndarray) -> np.ndarray:
    keys32 = keys.astype(np.float32, copy=False)
    if keys32.shape[0] == 0:
        return np.empty((0,), dtype=np.float32)
    normed = keys32 / np.maximum(np.linalg.norm(keys32, axis=1, keepdims=True), np.float32(1e-8))
    anchor = np.mean(normed, axis=0, keepdims=True)
    anchor /= np.maximum(np.linalg.norm(anchor, axis=1, keepdims=True), np.float32(1e-8))
    # KVPress KeyDiff keeps distinctive keys by pruning keys most similar to the anchor.
    return (-(normed @ anchor.reshape(-1))).astype(np.float32, copy=False)


def _leverage_scores(keys: np.ndarray, *, sketch_dim: int, seed: int) -> np.ndarray:
    keys32 = keys.astype(np.float32, copy=False)
    n, dim = int(keys32.shape[0]), int(keys32.shape[1])
    if n == 0:
        return np.empty((0,), dtype=np.float32)
    k = max(1, min(int(sketch_dim), dim, n))
    rng = np.random.default_rng(int(seed))
    phi = (rng.standard_normal((dim, k)).astype(np.float32) / np.sqrt(float(k))).astype(np.float32)
    x = (keys32 - np.mean(keys32, axis=0, keepdims=True, dtype=np.float64).astype(np.float32)) @ phi
    xtx = x.T.astype(np.float64, copy=False) @ x.astype(np.float64, copy=False)
    xtx += np.eye(k, dtype=np.float64) * 1e-2
    inv_xt = np.linalg.solve(xtx, x.T.astype(np.float64, copy=False))
    scores = np.sum(x.astype(np.float64, copy=False) * inv_xt.T, axis=1)
    return _zscore(np.maximum(scores, 0.0).astype(np.float32, copy=False))


def _expected_attention_scores(
    keys: np.ndarray,
    values: np.ndarray,
    obs_queries: np.ndarray | None,
    *,
    use_covariance: bool,
    use_vnorm: bool,
    epsilon: float,
    kernel_size: int,
) -> np.ndarray:
    keys32 = keys.astype(np.float32, copy=False)
    if obs_queries is None or obs_queries.size == 0 or keys32.shape[0] == 0:
        return np.zeros((keys32.shape[0],), dtype=np.float32)
    queries = obs_queries.reshape(-1, obs_queries.shape[-1]).astype(np.float32, copy=False)
    mu = np.mean(queries, axis=0).astype(np.float32, copy=False)
    logits = (keys32 @ mu) * (1.0 / np.sqrt(float(keys32.shape[-1])))
    if bool(use_covariance) and queries.shape[0] > 1:
        centered = queries - mu[None, :]
        cov = (centered.T.astype(np.float64, copy=False) @ centered.astype(np.float64, copy=False)) / float(queries.shape[0])
        logits = logits.astype(np.float64, copy=False) + 0.5 * np.einsum("nd,df,nf->n", keys32, cov, keys32) / float(keys32.shape[-1])
    logits = logits.astype(np.float64, copy=False)
    logits -= float(np.max(logits))
    probs = np.exp(logits)
    probs /= max(float(np.sum(probs)), 1e-20)
    scores = probs.astype(np.float32, copy=False)
    if bool(use_vnorm):
        scores = (scores + float(epsilon)) * np.linalg.norm(values.astype(np.float32, copy=False), axis=1)
    return _pool_scores_1d(scores.astype(np.float32, copy=False), int(kernel_size))


def _lagkv_scores(
    keys: np.ndarray,
    values: np.ndarray,
    *,
    sink: int,
    lag_size: int,
    cross_scoring: bool,
) -> np.ndarray:
    keys32 = keys.astype(np.float32, copy=False)
    values32 = values.astype(np.float32, copy=False)
    n, dim = int(keys32.shape[0]), int(keys32.shape[1])
    sink = max(0, min(int(sink), n))
    lag = max(1, int(lag_size))
    scores = np.zeros((n,), dtype=np.float32)
    if n < sink + 2 * lag:
        if n > sink:
            scores[sink:] = np.linspace(0.0, 1.0, n - sink, dtype=np.float32)
        scores[:sink] = 1.0
        return scores

    end_idx = sink + ((n - sink) // lag) * lag
    tail_len = lag + n - end_idx
    body_stop = max(sink, n - tail_len)
    body = slice(sink, body_stop)
    chunks = (body_stop - sink) // lag
    if chunks <= 1:
        scores[:sink] = 1.0
        scores[body_stop:] = 1.0
        return scores

    def _state_scores(x: np.ndarray) -> np.ndarray:
        arr = x[body].reshape(chunks, lag, dim)
        ref = arr[1:]
        target = arr[:-1]
        min_ref = ref.min(axis=1, keepdims=True)
        max_ref = ref.max(axis=1, keepdims=True)
        normed = (target - min_ref) / np.maximum(max_ref - min_ref, np.float32(1e-8))
        raw = np.std(normed, axis=-1)
        raw = raw - np.max(raw, axis=-1, keepdims=True)
        exp = np.exp(raw)
        return (exp / np.maximum(np.sum(exp, axis=-1, keepdims=True), 1e-20)).astype(np.float32, copy=False)

    chunk_scores = 0.5 * (_state_scores(keys32) + _state_scores(values32))
    if not bool(cross_scoring):
        ranks = np.argsort(np.argsort(chunk_scores, axis=-1), axis=-1).astype(np.float32)
        chunk_scores = ranks / float(max(1, lag - 1))
    scores[sink : sink + chunk_scores.size] = chunk_scores.reshape(-1)
    scores[:sink] = 1.0
    scores[body_stop:] = 1.0
    return scores.astype(np.float32, copy=False)


def _cur_scores(
    keys: np.ndarray,
    values: np.ndarray,
    *,
    leverage_type: str,
    local_window_size: int,
    sink: int,
) -> np.ndarray:
    keys32 = keys.astype(np.float32, copy=False)
    values32 = values.astype(np.float32, copy=False)
    n = int(keys32.shape[0])
    k2 = np.sum(keys32 * keys32, axis=1).astype(np.float32, copy=False)
    v2 = np.sum(values32 * values32, axis=1).astype(np.float32, copy=False)
    w = max(1, int(local_window_size))
    if w > 1 and n > 0:
        pad = (w - (n % w)) % w
        k_pad = np.pad(k2, (0, pad), mode="constant").reshape(-1, w)
        v_pad = np.pad(v2, (0, pad), mode="constant").reshape(-1, w)
        k2 = (k_pad / np.maximum(np.sum(k_pad, axis=1, keepdims=True), np.float32(1e-20))).reshape(-1)[:n]
        v2 = (v_pad / np.maximum(np.sum(v_pad, axis=1, keepdims=True), np.float32(1e-20))).reshape(-1)[:n]
    if leverage_type == "key":
        scores = k2
    elif leverage_type == "value":
        scores = v2
    elif leverage_type == "kv_avg":
        scores = 0.5 * (k2 + v2)
    else:
        scores = k2 * v2
    scores = scores / max(float(np.sum(scores)), 1e-20)
    scores[: max(0, min(int(sink), n))] = 1.0
    return scores.astype(np.float32, copy=False)


def _chunk_retention_mask(
    scores: np.ndarray,
    base_mask: np.ndarray,
    *,
    budget: int,
    chunk_length: int,
) -> np.ndarray:
    active = np.asarray(base_mask, dtype=bool).copy()
    n = int(active.shape[0])
    budget = min(max(0, int(budget)), n)
    remaining_budget = max(0, budget - int(np.count_nonzero(active)))
    if remaining_budget <= 0:
        return active
    chunk = max(1, int(chunk_length))
    chunk_scores: list[tuple[float, int, int]] = []
    score_arr = np.asarray(scores, dtype=np.float32)
    for start in range(0, n, chunk):
        stop = min(n, start + chunk)
        candidates = np.flatnonzero(~active[start:stop]) + start
        if candidates.size == 0:
            continue
        chunk_scores.append((float(np.mean(score_arr[candidates])), start, stop))
    if not chunk_scores:
        return active
    chunks_to_keep = max(1, int(np.ceil(float(remaining_budget) / float(chunk))))
    chosen_chunks = sorted(chunk_scores, key=lambda item: item[0], reverse=True)[:chunks_to_keep]
    for _, start, stop in chosen_chunks:
        active[start:stop] = True
    return active


def _attention_retention_mask(
    keys: np.ndarray,
    obs_queries: np.ndarray | None,
    base_mask: np.ndarray,
    *,
    budget: int,
    obs_window: int,
    kernel_size: int,
    score_mode: str,
) -> np.ndarray:
    context_len = int(keys.shape[0])
    active = np.asarray(base_mask, dtype=bool).copy()
    if int(obs_window) > 0:
        active[max(0, context_len - int(obs_window)) :] = True
    budget = min(max(0, int(budget)), context_len)
    already = int(np.count_nonzero(active))
    remaining_budget = max(0, budget - already)
    if remaining_budget <= 0:
        return active
    candidates = np.flatnonzero(~active)
    if candidates.size == 0:
        return active
    scores = _attention_observation_scores(keys, obs_queries, mode=score_mode, kernel_size=kernel_size)
    take = min(remaining_budget, int(candidates.size))
    chosen = candidates[np.argpartition(scores[candidates], -take)[-take:]]
    active[chosen] = True
    return active


def _select_top_norm_mask(
    keys: np.ndarray,
    values: np.ndarray,
    base_mask: np.ndarray,
    keep_count: int,
) -> np.ndarray:
    out = np.asarray(base_mask, dtype=bool).copy()
    remaining = np.flatnonzero(~out)
    if int(keep_count) <= 0 or remaining.size == 0:
        return out
    take = min(int(keep_count), int(remaining.size))
    scores = np.linalg.norm(keys[remaining].astype(np.float32, copy=False), axis=1)
    scores += np.linalg.norm(values[remaining].astype(np.float32, copy=False), axis=1)
    chosen = remaining[np.argpartition(scores, -take)[-take:]]
    out[chosen] = True
    return out


def _copy_exact_window(keys_hat: np.ndarray, values_hat: np.ndarray, keys: np.ndarray, values: np.ndarray, exact_mask: np.ndarray) -> None:
    if np.any(exact_mask):
        keys_hat[exact_mask] = keys[exact_mask]
        values_hat[exact_mask] = values[exact_mask]


def build_compressed_kv(
    method: str,
    *,
    keys: np.ndarray,
    values: np.ndarray,
    static_prefix: int,
    default_residual_window: int,
    key_bytes: int,
    value_bytes: int,
    metadata_bytes: int,
    pq_iters: int,
    seed: int,
    obs_queries: np.ndarray | None = None,
) -> CompressedKV:
    name = str(method).strip().lower()
    context_len = int(keys.shape[0])
    dim = int(keys.shape[1])
    residual_window = _parse_window(name, int(default_residual_window))
    exact_mask = exact_window_mask(context_len, static_prefix=int(static_prefix), residual_window=residual_window)
    comp_mask = ~exact_mask
    comp_count = int(np.count_nonzero(comp_mask))
    exact_count = int(np.count_nonzero(exact_mask))
    exact_bytes = float(exact_count * dim * (int(key_bytes) + int(value_bytes)))
    update_exact_window = int(key_bytes + value_bytes) * dim

    if name == "dense":
        dense_mb = float(context_len * dim * (int(key_bytes) + int(value_bytes))) / MB
        return CompressedKV(
            method="dense",
            keys_hat=keys.astype(np.float32, copy=False),
            values_hat=values.astype(np.float32, copy=False),
            active_mask=None,
            score_source_keys=None,
            score_mask=None,
            query_read_mb=dense_mb,
            online_update_mb_per_token=float(update_exact_window) / MB,
            metadata={"family": "dense", "bits": 16, "exact_tokens": context_len, "compressed_tokens": 0},
        )

    keys32 = keys.astype(np.float32, copy=False)
    values32 = values.astype(np.float32, copy=False)

    if name.startswith("recent_k"):
        keep = _parse_count(name, prefix="k", default=2048)
        active = np.zeros((context_len,), dtype=bool)
        active[max(0, context_len - int(keep)) :] = True
        active_count = int(np.count_nonzero(active))
        read_mb = float(active_count * dim * (int(key_bytes) + int(value_bytes))) / MB
        return CompressedKV(
            method=f"recent_k{keep}",
            keys_hat=keys32,
            values_hat=values32,
            active_mask=active,
            score_source_keys=None,
            score_mask=None,
            query_read_mb=read_mb,
            online_update_mb_per_token=float(update_exact_window) / MB,
            metadata={
                "family": "retention_recent",
                "context_len": context_len,
                "head_dim": dim,
                "retained_tokens": active_count,
                "exact_tokens": active_count,
                "compressed_tokens": context_len - active_count,
                "proxy_not_faithful_paper_impl": True,
            },
        )

    if name.startswith("sink_recent"):
        sink, recent = _parse_pair(name, first="s", second="r", defaults=(128, 2048))
        active = np.zeros((context_len,), dtype=bool)
        if sink > 0:
            active[: min(int(sink), context_len)] = True
        if recent > 0:
            active[max(0, context_len - int(recent)) :] = True
        active_count = int(np.count_nonzero(active))
        read_mb = float(active_count * dim * (int(key_bytes) + int(value_bytes))) / MB
        return CompressedKV(
            method=f"sink_recent_s{sink}r{recent}",
            keys_hat=keys32,
            values_hat=values32,
            active_mask=active,
            score_source_keys=None,
            score_mask=None,
            query_read_mb=read_mb,
            online_update_mb_per_token=float(update_exact_window) / MB,
            metadata={
                "family": "retention_sink_recent",
                "context_len": context_len,
                "head_dim": dim,
                "sink_tokens": int(sink),
                "recent_tokens": int(recent),
                "retained_tokens": active_count,
                "exact_tokens": active_count,
                "compressed_tokens": context_len - active_count,
                "proxy_not_faithful_paper_impl": True,
            },
        )

    if name.startswith("l2ret_k"):
        keep = _parse_count(name, prefix="k", default=2048)
        active = _select_top_norm_mask(keys32, values32, exact_mask, keep)
        active_count = int(np.count_nonzero(active))
        read_mb = float(active_count * dim * (int(key_bytes) + int(value_bytes))) / MB
        return CompressedKV(
            method=f"l2ret_k{keep}_w{residual_window}",
            keys_hat=keys32,
            values_hat=values32,
            active_mask=active,
            score_source_keys=None,
            score_mask=None,
            query_read_mb=read_mb,
            online_update_mb_per_token=float(update_exact_window) / MB,
            metadata={
                "family": "retention_l2norm",
                "context_len": context_len,
                "head_dim": dim,
                "residual_window": int(residual_window),
                "norm_retained_tokens": int(keep),
                "retained_tokens": active_count,
                "exact_tokens": active_count,
                "compressed_tokens": context_len - active_count,
                "proxy_not_faithful_paper_impl": True,
            },
        )

    if (
        name.startswith("keydiff_k")
        or name.startswith("leverage_k")
        or name.startswith("expected_attn_k")
        or name.startswith("critical_snap_k")
        or name.startswith("chunk_snap_k")
        or name.startswith("lagkv_k")
        or name.startswith("compactor_k")
        or name.startswith("knorm_k")
        or name.startswith("cur_k")
        or name.startswith("tova_k")
    ):
        keep = _parse_count(name, prefix="k", default=2048)
        kernel_size = _parse_kernel_size(name, default=5)
        if name.startswith("tova_k"):
            scores = _attention_observation_scores(keys32, obs_queries, mode="mean", kernel_size=1)
            family = "tova_trace"
            label = f"tova_k{keep}_w{residual_window}"
            meta_extra = {"source_impl": "KVPress TOVA-style last-query attention scorer"}
            active = _scores_retention_mask(scores, exact_mask, budget=keep)
        elif name.startswith("cur_k"):
            local_window = _parse_count(name, prefix="lw", default=16)
            sink = _parse_count(name, prefix="s", default=4)
            if "_key" in name:
                leverage_type = "key"
            elif "_value" in name:
                leverage_type = "value"
            elif "_avg" in name:
                leverage_type = "kv_avg"
            else:
                leverage_type = "kv_product"
            scores = _cur_scores(keys32, values32, leverage_type=leverage_type, local_window_size=int(local_window), sink=int(sink))
            family = "cur_trace"
            label = f"cur_k{keep}_lw{local_window}_s{sink}_{leverage_type}_w{residual_window}"
            meta_extra = {
                "local_window_size": int(local_window),
                "sink_tokens": int(sink),
                "leverage_type": leverage_type,
                "source_impl": "KVPress CUR local leverage scorer on saved K/V trace",
            }
            active = _scores_retention_mask(scores, exact_mask, budget=keep)
        elif name.startswith("knorm_k"):
            scores = (-np.linalg.norm(keys32, axis=1)).astype(np.float32, copy=False)
            family = "knorm_trace"
            label = f"knorm_k{keep}_w{residual_window}"
            meta_extra = {"source_impl": "KVPress Knorm score on saved keys"}
            active = _scores_retention_mask(scores, exact_mask, budget=keep)
        elif name.startswith("keydiff_k"):
            scores = _keydiff_scores(keys32)
            family = "keydiff_trace"
            label = f"keydiff_k{keep}_w{residual_window}"
            meta_extra = {"source_impl": "KVPress KeyDiff score on saved post-RoPE keys"}
            active = _scores_retention_mask(scores, exact_mask, budget=keep)
        elif name.startswith("leverage_k"):
            sketch_dim = _parse_count(name, prefix="sk", default=48)
            scores = _leverage_scores(keys32, sketch_dim=int(sketch_dim), seed=int(seed) + 83)
            family = "leverage_trace"
            label = f"leverage_k{keep}_sk{sketch_dim}_w{residual_window}"
            meta_extra = {"sketch_dim": int(sketch_dim), "source_impl": "KVPress leverage-score scorer on saved post-RoPE keys"}
            active = _scores_retention_mask(scores, exact_mask, budget=keep)
        elif name.startswith("expected_attn_k"):
            obs_window = _parse_observation_window(name, default=288)
            use_cov = _parse_binary_param(name, prefix="cov", default=True)
            use_vnorm = _parse_binary_param(name, prefix="v", default=True)
            epsilon = _parse_float_param(name, prefix="eps", default=0.0)
            scores = _expected_attention_scores(
                keys32,
                values32,
                obs_queries,
                use_covariance=use_cov,
                use_vnorm=use_vnorm,
                epsilon=float(epsilon),
                kernel_size=int(kernel_size),
            )
            family = "expected_attention_trace"
            label = f"expected_attn_k{keep}_obs{obs_window}_cov{int(use_cov)}_v{int(use_vnorm)}_ker{kernel_size}_w{residual_window}"
            meta_extra = {
                "obs_window": int(obs_window),
                "use_covariance": bool(use_cov),
                "use_vnorm": bool(use_vnorm),
                "epsilon": float(epsilon),
                "source_impl": "KVPress ExpectedAttention-style post-RoPE trace scorer",
            }
            active = _scores_retention_mask(scores, exact_mask, budget=keep)
        elif name.startswith("critical_snap_k"):
            obs_window = _parse_observation_window(name, default=288)
            first_stage_ratio = _parse_float_param(name, prefix="fs", default=0.5)
            epsilon = _parse_float_param(name, prefix="eps", default=1e-4)
            base_scores = _attention_observation_scores(keys32, obs_queries, mode="mean", kernel_size=kernel_size)
            value_norm = np.linalg.norm(values32, axis=1).astype(np.float32, copy=False)
            stage1 = int(max(0, min(context_len, keep)) * max(0.0, min(float(first_stage_ratio), 1.0)))
            scores = (base_scores + float(epsilon)) * value_norm
            if stage1 > 0:
                top1 = np.argpartition(base_scores, -min(stage1, context_len))[-min(stage1, context_len) :]
                scores[top1] = np.finfo(np.float32).max
            family = "critical_snap_trace"
            label = f"critical_snap_k{keep}_obs{obs_window}_fs{first_stage_ratio:g}_ker{kernel_size}_w{residual_window}"
            meta_extra = {
                "obs_window": int(obs_window),
                "first_stage_ratio": float(first_stage_ratio),
                "epsilon": float(epsilon),
                "source_impl": "KVPress CriticalKV-style two-stage SnapKV scorer using V norm proxy",
                "wo_norm_proxy": "value_l2_norm",
            }
            active = _scores_retention_mask(scores, exact_mask, budget=keep)
        elif name.startswith("chunk_snap_k"):
            obs_window = _parse_observation_window(name, default=288)
            chunk_length = _parse_count(name, prefix="c", default=256)
            scores = _attention_observation_scores(keys32, obs_queries, mode="mean", kernel_size=kernel_size)
            family = "chunkkv_snap_trace"
            label = f"chunk_snap_k{keep}_c{chunk_length}_obs{obs_window}_ker{kernel_size}_w{residual_window}"
            meta_extra = {
                "chunk_length": int(chunk_length),
                "obs_window": int(obs_window),
                "source_impl": "KVPress ChunkKV wrapper over SnapKV-style trace scorer",
            }
            active = _chunk_retention_mask(scores, exact_mask, budget=keep, chunk_length=int(chunk_length))
        elif name.startswith("lagkv_k"):
            lag_size = _parse_count(name, prefix="lag", default=128)
            sink = _parse_count(name, prefix="s", default=4)
            cross = _parse_binary_param(name, prefix="cross", default=False)
            lag_base = exact_window_mask(context_len, static_prefix=int(sink), residual_window=int(lag_size))
            scores = _lagkv_scores(keys32, values32, sink=int(sink), lag_size=int(lag_size), cross_scoring=bool(cross))
            family = "lagkv_trace"
            label = f"lagkv_k{keep}_lag{lag_size}_s{sink}_cross{int(cross)}"
            meta_extra = {
                "lag_size": int(lag_size),
                "sink_tokens": int(sink),
                "cross_scoring": bool(cross),
                "source_impl": "KVPress LagKV score on saved K/V trace",
            }
            active = _scores_retention_mask(scores, lag_base, budget=keep)
        else:
            sketch_dim = _parse_count(name, prefix="sk", default=48)
            obs_window = _parse_observation_window(name, default=288)
            blend = _parse_float_param(name, prefix="bl", default=0.35)
            lev = _leverage_scores(keys32, sketch_dim=int(sketch_dim), seed=int(seed) + 89)
            attn = _zscore(_attention_observation_scores(keys32, obs_queries, mode="sum", kernel_size=kernel_size))
            scores = float(blend) * lev + attn
            family = "compactor_trace"
            label = f"compactor_k{keep}_sk{sketch_dim}_obs{obs_window}_bl{blend:g}_ker{kernel_size}_w{residual_window}"
            meta_extra = {
                "sketch_dim": int(sketch_dim),
                "obs_window": int(obs_window),
                "blending": float(blend),
                "source_impl": "Compactor-style blend of leverage score and trace attention score",
            }
            active = _scores_retention_mask(scores, exact_mask, budget=keep)

        active_count = int(np.count_nonzero(active))
        read_mb = float(active_count * dim * (int(key_bytes) + int(value_bytes))) / MB
        return CompressedKV(
            method=label,
            keys_hat=keys32,
            values_hat=values32,
            active_mask=active,
            score_source_keys=None,
            score_mask=None,
            query_read_mb=read_mb,
            online_update_mb_per_token=float(update_exact_window) / MB,
            metadata={
                "family": family,
                "context_len": context_len,
                "head_dim": dim,
                "budget": int(keep),
                "kernel_size": int(kernel_size),
                "retained_tokens": active_count,
                "exact_tokens": active_count,
                "compressed_tokens": context_len - active_count,
                "trace_observation_queries": obs_queries is not None,
                "proxy_not_faithful_paper_impl": bool(family in {"expected_attention_trace", "critical_snap_trace", "compactor_trace"}),
                **meta_extra,
            },
        )

    if name.startswith("snapkv_k") or name.startswith("kvzip_k") or name.startswith("rocket_snap_k") or name.startswith("h2o_k"):
        keep = _parse_count(name, prefix="k", default=2048)
        obs_window = _parse_observation_window(name, default=64)
        kernel_size = _parse_kernel_size(name, default=5)
        if name.startswith("kvzip_k"):
            score_mode = "max"
        elif name.startswith("h2o_k"):
            score_mode = "sum"
        else:
            score_mode = "mean"
        active = _attention_retention_mask(
            keys32,
            obs_queries,
            exact_mask,
            budget=keep,
            obs_window=obs_window,
            kernel_size=kernel_size,
            score_mode=score_mode,
        )
        active_count = int(np.count_nonzero(active))
        read_mb = float(active_count * dim * (int(key_bytes) + int(value_bytes))) / MB
        if name.startswith("kvzip_k"):
            family = "kvzip_trace"
            label_prefix = "kvzip"
        elif name.startswith("h2o_k"):
            family = "h2o_trace"
            label_prefix = "h2o"
        else:
            family = "snapkv_trace"
            label_prefix = "rocket_snap" if name.startswith("rocket_snap_k") else "snapkv"
        return CompressedKV(
            method=f"{label_prefix}_k{keep}_obs{obs_window}_ker{kernel_size}",
            keys_hat=keys32,
            values_hat=values32,
            active_mask=active,
            score_source_keys=None,
            score_mask=None,
            query_read_mb=read_mb,
            online_update_mb_per_token=float(update_exact_window) / MB,
            metadata={
                "family": family,
                "context_len": context_len,
                "head_dim": dim,
                "budget": int(keep),
                "obs_window": int(obs_window),
                "kernel_size": int(kernel_size),
                "score_mode": score_mode,
                "retained_tokens": active_count,
                "exact_tokens": active_count,
                "compressed_tokens": context_len - active_count,
                "proxy_not_faithful_paper_impl": True,
                "trace_observation_queries": True,
            },
        )

    keys_hat = keys32.copy()
    values_hat = values32.copy()
    query_read_override: float | None = None
    update_override: float | None = None

    if name.startswith("pmkvq_like"):
        bits = _parse_bits(name, default=4)
        sink = _parse_count(name, prefix="s", default=1)
        group_size = _parse_count(name, prefix="g", default=128)
        exact_mask = exact_window_mask(context_len, static_prefix=int(sink), residual_window=residual_window)
        comp_mask = ~exact_mask
        comp_count = int(np.count_nonzero(comp_mask))
        exact_count = int(np.count_nonzero(exact_mask))
        exact_bytes = float(exact_count * dim * (int(key_bytes) + int(value_bytes)))
        key_bytes_total = 0.0
        value_bytes_total = 0.0
        key_per_token = 0.0
        value_per_token = 0.0
        if comp_count:
            key_hat, key_bytes_total, key_per_token = _pmkvq_reconstruct_with_cost(
                keys32[comp_mask],
                bits=bits,
                group_size=group_size,
                metadata_bytes=int(metadata_bytes),
            )
            value_hat, value_bytes_total, value_per_token = _pmkvq_reconstruct_with_cost(
                values32[comp_mask],
                bits=bits,
                group_size=group_size,
                metadata_bytes=int(metadata_bytes),
            )
            keys_hat[comp_mask] = key_hat
            values_hat[comp_mask] = value_hat
        sidecar_per_token = float(key_per_token + value_per_token)
        label = f"pmkvq_like_b{bits}_s{sink}_g{group_size}_w{residual_window}"
        meta = {
            "family": "pmkvq_like",
            "bits": int(bits),
            "sink_tokens": int(sink),
            "window_tokens": int(residual_window),
            "group_size": int(group_size),
            "source_impl": "PM-KVQ progressive per-group quantized cache",
        }
    elif name.startswith("kitty_like"):
        key_bits, value_bits = _parse_tq_bits(name)
        buffer_length = _parse_count(name, prefix="buf", default=128)
        sink = _parse_count(name, prefix="s", default=32)
        group_size = _parse_count(name, prefix="g", default=128)
        promote_ratio = _parse_float_param(name, prefix="p", default=0.1)
        promote_bit = _parse_count(name, prefix="pb", default=4)
        exact_mask = exact_window_mask(context_len, static_prefix=int(sink), residual_window=int(buffer_length))
        comp_mask = ~exact_mask
        comp_count = int(np.count_nonzero(comp_mask))
        exact_count = int(np.count_nonzero(exact_mask))
        exact_bytes = float(exact_count * dim * (int(key_bytes) + int(value_bytes)))
        key_bytes_total = 0.0
        value_bytes_total = 0.0
        key_per_token = 0.0
        value_per_token = 0.0
        promoted_total = 0
        if comp_count:
            key_hat, key_bytes_total, key_per_token, promoted_total = _kitty_key_quantize_pages(
                keys32[comp_mask],
                buffer_length=int(buffer_length),
                group_size=int(group_size),
                bits=int(key_bits),
                promote_ratio=float(promote_ratio),
                promote_bit=int(promote_bit),
                metadata_bytes=int(metadata_bytes),
            )
            value_hat, value_bytes_total, value_per_token = _pmkvq_reconstruct_with_cost(
                values32[comp_mask],
                bits=int(value_bits),
                group_size=int(group_size),
                metadata_bytes=int(metadata_bytes),
            )
            keys_hat[comp_mask] = key_hat
            values_hat[comp_mask] = value_hat
        sidecar_per_token = float(key_per_token + value_per_token)
        label = f"kitty_like_k{key_bits}v{value_bits}_p{promote_ratio:g}_pb{promote_bit}_buf{buffer_length}_s{sink}"
        meta = {
            "family": "kitty_like",
            "key_bits": int(key_bits),
            "value_bits": int(value_bits),
            "sink_tokens": int(sink),
            "buffer_length": int(buffer_length),
            "group_size": int(group_size),
            "promote_ratio": float(promote_ratio),
            "promote_bit": int(promote_bit),
            "promoted_channel_pages": int(promoted_total),
            "source_impl": "Kitty dynamic channel-wise precision boost",
        }
    elif name.startswith("tada_like"):
        bits = _parse_bits(name, default=4)
        group_size = _parse_count(name, prefix="g", default=128)
        key_bytes_total = 0.0
        value_bytes_total = 0.0
        key_per_token = 0.0
        value_per_token = 0.0
        if comp_count:
            key_hat, key_bytes_total, key_per_token = _mean_centered_group_quantize_with_cost(
                keys32[comp_mask],
                bits=int(bits),
                group_size=int(group_size),
                metadata_bytes=int(metadata_bytes),
            )
            value_hat, value_bytes_total, value_per_token = _mean_centered_group_quantize_with_cost(
                values32[comp_mask],
                bits=int(bits),
                group_size=int(group_size),
                metadata_bytes=int(metadata_bytes),
            )
            keys_hat[comp_mask] = key_hat
            values_hat[comp_mask] = value_hat
        sidecar_per_token = float(key_per_token + value_per_token)
        label = f"tada_like_b{bits}_g{group_size}_w{residual_window}"
        meta = {
            "family": "tada_like",
            "bits": int(bits),
            "group_size": int(group_size),
            "source_impl": "TaDA-style mean-centered groupwise KV quantization proxy",
        }
    elif name.startswith("tiered_quant"):
        low_bits = _parse_bits(name, prefix="l", default=2)
        mid_bits = _parse_bits(name, prefix="m", default=4)
        high_bits = _parse_bits(name, prefix="h", default=8)
        high_count = _parse_count(name, prefix="hi", default=4096)
        mid_count = _parse_count(name, prefix="mid", default=16384)
        score_mode = "norm"
        if "_attn" in name:
            token_scores = _attention_observation_scores(keys32, obs_queries, mode="mean", kernel_size=1)
            score_mode = "observed_attention"
        else:
            token_scores = np.linalg.norm(keys32, axis=1) + np.linalg.norm(values32, axis=1)
        key_bytes_total = 0.0
        value_bytes_total = 0.0
        key_per_token = 0.0
        value_per_token = 0.0
        key_meta = {"high_tokens": 0, "mid_tokens": 0, "low_tokens": 0}
        if comp_count:
            comp_scores = token_scores[comp_mask]
            key_hat, key_bytes_total, key_per_token, key_meta = _tiered_quantize_with_cost(
                keys32[comp_mask],
                comp_scores,
                low_bits=int(low_bits),
                mid_bits=int(mid_bits),
                high_bits=int(high_bits),
                high_count=int(high_count),
                mid_count=int(mid_count),
                metadata_bytes=int(metadata_bytes),
            )
            value_hat, value_bytes_total, value_per_token, _ = _tiered_quantize_with_cost(
                values32[comp_mask],
                comp_scores,
                low_bits=int(low_bits),
                mid_bits=int(mid_bits),
                high_bits=int(high_bits),
                high_count=int(high_count),
                mid_count=int(mid_count),
                metadata_bytes=int(metadata_bytes),
            )
            keys_hat[comp_mask] = key_hat
            values_hat[comp_mask] = value_hat
        sidecar_per_token = float(key_per_token + value_per_token)
        suffix = "_attn" if score_mode == "observed_attention" else ""
        label = f"tiered_quant_l{low_bits}_m{mid_bits}_h{high_bits}_hi{high_count}_mid{mid_count}_w{residual_window}{suffix}"
        meta = {
            "family": "tiered_mixed_precision",
            "low_bits": int(low_bits),
            "mid_bits": int(mid_bits),
            "high_bits": int(high_bits),
            "high_budget": int(high_count),
            "mid_budget": int(mid_count),
            "score_mode": score_mode,
            **{f"key_{k}": int(v) for k, v in key_meta.items()},
            "source_impl": "KVTuner/MiniKV/TailorKV-style token-tiered mixed precision proxy",
        }
    elif name.startswith("lookat_like"):
        keep_frac = _parse_float_param(name, prefix="p", default=0.25)
        bits = _parse_bits(name, default=8)
        keep_mean = _parse_binary_param(name, prefix="mean", default=True)
        key_bytes_total = 0.0
        value_bytes_total = 0.0
        key_per_token = 0.0
        value_per_token = 0.0
        kept_channels = 0
        if comp_count:
            key_hat, key_bytes_total, key_per_token, kept_channels = _sparse_channel_reconstruct_with_cost(
                keys32[comp_mask],
                keep_frac=float(keep_frac),
                bits=int(bits),
                metadata_bytes=int(metadata_bytes),
                keep_mean=bool(keep_mean),
            )
            value_hat, value_bytes_total, value_per_token, _ = _sparse_channel_reconstruct_with_cost(
                values32[comp_mask],
                keep_frac=float(keep_frac),
                bits=int(bits),
                metadata_bytes=int(metadata_bytes),
                keep_mean=bool(keep_mean),
            )
            keys_hat[comp_mask] = key_hat
            values_hat[comp_mask] = value_hat
        sidecar_per_token = float(key_per_token + value_per_token)
        label = f"lookat_like_p{keep_frac:g}_b{bits}_mean{int(keep_mean)}_w{residual_window}"
        meta = {
            "family": "lookat_sparse_channel",
            "keep_fraction": float(keep_frac),
            "bits": int(bits),
            "kept_channels": int(kept_channels),
            "source_impl": "LOOKAT-style per-token sparse channel KV proxy",
        }
    elif name.startswith("kvtc_like") or name.startswith("freqkv_like"):
        bits = _parse_bits(name, default=4)
        rank = _parse_count(name, prefix="r", default=32)
        key_bytes_total = 0.0
        value_bytes_total = 0.0
        key_per_token = 0.0
        value_per_token = 0.0
        if comp_count:
            if name.startswith("freqkv_like"):
                key_hat, key_bytes_total, key_per_token = _dct_transform_quantize_with_cost(
                    keys32[comp_mask],
                    rank=int(rank),
                    bits=int(bits),
                    metadata_bytes=int(metadata_bytes),
                )
                value_hat, value_bytes_total, value_per_token = _dct_transform_quantize_with_cost(
                    values32[comp_mask],
                    rank=int(rank),
                    bits=int(bits),
                    metadata_bytes=int(metadata_bytes),
                )
            else:
                key_hat, key_bytes_total, key_per_token = _pca_transform_quantize_with_cost(
                    keys32[comp_mask],
                    rank=int(rank),
                    bits=int(bits),
                    metadata_bytes=int(metadata_bytes),
                )
                value_hat, value_bytes_total, value_per_token = _pca_transform_quantize_with_cost(
                    values32[comp_mask],
                    rank=int(rank),
                    bits=int(bits),
                    metadata_bytes=int(metadata_bytes),
                )
            keys_hat[comp_mask] = key_hat
            values_hat[comp_mask] = value_hat
        sidecar_per_token = float(key_per_token + value_per_token)
        family = "freqkv_transform_coding" if name.startswith("freqkv_like") else "kvtc_transform_coding"
        label = f"{'freqkv_like' if name.startswith('freqkv_like') else 'kvtc_like'}_b{bits}_r{rank}_w{residual_window}"
        meta = {
            "family": family,
            "bits": int(bits),
            "rank": int(rank),
            "source_impl": "KVTC/FreqKV-style transform coding proxy",
        }
    elif name.startswith("million_like"):
        subvecs, subbits = _parse_stage_bits(name, default_stages=64, default_bits=8)
        sink = _parse_count(name, prefix="s", default=0)
        exact_mask = exact_window_mask(context_len, static_prefix=int(sink), residual_window=residual_window)
        comp_mask = ~exact_mask
        comp_count = int(np.count_nonzero(comp_mask))
        exact_count = int(np.count_nonzero(exact_mask))
        exact_bytes = float(exact_count * dim * (int(key_bytes) + int(value_bytes)))
        key_bytes_total = 0.0
        value_bytes_total = 0.0
        if comp_count:
            key_hat, key_cost = _pq_reconstruct(
                keys32[comp_mask],
                subvecs=subvecs,
                subbits=subbits,
                seed=int(seed) + 61,
                max_iter=int(pq_iters),
            )
            value_hat, value_cost = _pq_reconstruct(
                values32[comp_mask],
                subvecs=subvecs,
                subbits=subbits,
                seed=int(seed) + 67,
                max_iter=int(pq_iters),
            )
            keys_hat[comp_mask] = key_hat
            values_hat[comp_mask] = value_hat
            key_bytes_total = float(key_cost)
            value_bytes_total = float(value_cost)
        sidecar_per_token = 2.0 * _packed_code_bytes(1, subvecs, subbits)
        label = f"million_like_m{subvecs}b{subbits}_s{sink}_w{residual_window}"
        meta = {
            "family": "million_like",
            "subvecs": int(subvecs),
            "subbits": int(subbits),
            "sink_tokens": int(sink),
            "residual_tokens": int(residual_window),
            "source_impl": "MILLION DynamicPQCache PQ codes plus exact residual cache",
        }
    elif name.startswith("cam_like"):
        keep = _parse_count(name, prefix="k", default=8192)
        merge_budget = _parse_count(name, prefix="merge", default=32)
        obs_window = _parse_observation_window(name, default=288)
        kernel_size = _parse_kernel_size(name, default=5)
        base_mask = exact_window_mask(context_len, static_prefix=static_prefix, residual_window=residual_window)
        merged_keys, merged_values, pos_weights, read_bytes, cam_meta = _cam_snapshot(
            keys32,
            values32,
            obs_queries,
            base_mask,
            budget=int(keep),
            merge_budget=int(merge_budget),
            obs_window=int(obs_window),
            kernel_size=int(kernel_size),
        )
        read_mb = float(read_bytes) / MB
        return CompressedKV(
            method=f"cam_like_k{keep}_merge{merge_budget}_obs{obs_window}_ker{kernel_size}_w{residual_window}",
            keys_hat=merged_keys,
            values_hat=merged_values,
            active_mask=None,
            score_source_keys=None,
            score_mask=None,
            query_read_mb=read_mb,
            online_update_mb_per_token=float(update_exact_window) / MB,
            metadata={
                "family": "cam_like",
                "context_len": context_len,
                "head_dim": dim,
                "budget": int(keep),
                "merge_budget": int(merge_budget),
                "obs_window": int(obs_window),
                "kernel_size": int(kernel_size),
                "proxy_not_faithful_paper_impl": True,
                "source_impl": "CaM merge-before-prune snapshot proxy with deterministic value merging",
                **cam_meta,
            },
            position_weights=pos_weights,
        )
    elif name.startswith("zeromerge_like"):
        keep = _parse_count(name, prefix="k", default=8192)
        tail = _parse_count(name, prefix="tail", default=2048)
        dense_clusters = _parse_count(name, prefix="dense", default=512)
        sink = _parse_count(name, prefix="s", default=0)
        obs_window = _parse_observation_window(name, default=64)
        kernel_size = _parse_kernel_size(name, default=5)
        base_mask = exact_window_mask(context_len, static_prefix=int(sink), residual_window=0)
        merged_keys, merged_values, pos_weights, read_bytes, zm_meta = _zeromerge_snapshot(
            keys32,
            values32,
            obs_queries,
            base_mask,
            budget=int(keep),
            tail=int(tail),
            dense=int(dense_clusters),
            obs_window=int(obs_window),
            kernel_size=int(kernel_size),
        )
        read_mb = float(read_bytes) / MB
        return CompressedKV(
            method=f"zeromerge_like_k{keep}_tail{tail}_dense{dense_clusters}_s{sink}_obs{obs_window}_ker{kernel_size}",
            keys_hat=merged_keys,
            values_hat=merged_values,
            active_mask=None,
            score_source_keys=None,
            score_mask=None,
            query_read_mb=read_mb,
            online_update_mb_per_token=float(update_exact_window) / MB,
            metadata={
                "family": "zeromerge_like",
                "context_len": context_len,
                "head_dim": dim,
                "budget": int(keep),
                "tail_tokens": int(tail),
                "dense_clusters": int(dense_clusters),
                "sink_tokens": int(sink),
                "obs_window": int(obs_window),
                "kernel_size": int(kernel_size),
                "proxy_not_faithful_paper_impl": False,
                "source_impl": "ZeroMerge cache_init snapshot with weighted merged tokens",
                **zm_meta,
            },
            position_weights=pos_weights,
        )
    elif name.startswith("kivi_b"):
        bits = _parse_bits(name, default=2)
        group_size = _parse_count(name, prefix="g", default=32)
        keys_hat, values_hat, query_bytes, update_bytes_per_token, kivi_meta = _build_kivi_with_cost(
            keys32,
            values32,
            bits=int(bits),
            group_size=int(group_size),
            residual_window=int(residual_window),
            key_bytes=int(key_bytes),
            value_bytes=int(value_bytes),
            metadata_bytes=int(metadata_bytes),
        )
        key_bytes_total = float(kivi_meta["compressed_key_MB"]) * MB
        value_bytes_total = float(kivi_meta["compressed_value_MB"]) * MB
        exact_bytes = float(kivi_meta["exact_key_MB"] + kivi_meta["exact_value_MB"]) * MB
        sidecar_per_token = float(kivi_meta["key_sidecar_update_MB_per_token"] + kivi_meta["value_sidecar_update_MB_per_token"]) * MB
        label = f"kivi_b{bits}_g{group_size}_w{residual_window}"
        meta = {
            "family": "kivi",
            "bits": int(bits),
            "group_size": int(group_size),
            "source_impl": "official jy-yuan/KIVI quantization layout: K time-groups, V channel-groups",
            "proxy_not_faithful_paper_impl": False,
            **kivi_meta,
        }
        query_read_override = float(query_bytes) / MB
        update_override = float(update_bytes_per_token) / MB
        exact_count = max(int(kivi_meta["key_exact_tokens"]), int(kivi_meta["value_exact_tokens"]))
        comp_count = context_len - exact_count
    elif name.startswith("per_token_kv"):
        bits = _parse_bits(name, default=4)
        keys_hat[comp_mask] = _quantize_per_token(keys32[comp_mask], bits)
        values_hat[comp_mask] = _quantize_per_token(values32[comp_mask], bits)
        key_bytes_total = _packed_code_bytes(comp_count, dim, bits) + _scale_zero_bytes(comp_count, 1, int(metadata_bytes))
        value_bytes_total = _packed_code_bytes(comp_count, dim, bits) + _scale_zero_bytes(comp_count, 1, int(metadata_bytes))
        sidecar_per_token = 2.0 * (_packed_code_bytes(1, dim, bits) + _scale_zero_bytes(1, 1, int(metadata_bytes)))
        label = f"per_token_kv_b{bits}_w{residual_window}"
        meta = {"family": "per_token_kv", "bits": bits}
    elif name.startswith("kvquant_like"):
        bits = _parse_bits(name, default=3)
        clip = _parse_clip(name, default=0.1)
        keys_hat[comp_mask] = _quantize_per_channel(keys32[comp_mask], bits, clip_percent=clip)
        values_hat[comp_mask] = _quantize_per_channel(values32[comp_mask], bits, clip_percent=clip)
        key_bytes_total = _packed_code_bytes(comp_count, dim, bits) + _scale_zero_bytes(1, dim, int(metadata_bytes))
        value_bytes_total = _packed_code_bytes(comp_count, dim, bits) + _scale_zero_bytes(1, dim, int(metadata_bytes))
        sidecar_per_token = 2.0 * _packed_code_bytes(1, dim, bits)
        label = f"kvquant_like_b{bits}_clip{clip:g}_w{residual_window}"
        meta = {"family": "kvquant_like", "bits": bits, "clip_percent": clip}
    elif name.startswith("tqprod") or name.startswith("tq_"):
        key_bits, value_bits = _parse_tq_bits(name)
        product = name.startswith("tqprod")
        if comp_count:
            keys_hat[comp_mask] = (
                _tq_reconstruct_product(keys32[comp_mask], key_bits)
                if product
                else _tq_reconstruct(keys32[comp_mask], key_bits)[0]
            )
            values_hat[comp_mask] = _tq_reconstruct(values32[comp_mask], value_bits)[0]
        key_bytes_total = _packed_code_bytes(comp_count, dim, key_bits) + float(comp_count * int(metadata_bytes))
        if product:
            key_bytes_total += _packed_code_bytes(comp_count, dim, 1) + float(comp_count * int(metadata_bytes))
        value_bytes_total = _packed_code_bytes(comp_count, dim, value_bits) + float(comp_count * int(metadata_bytes))
        sidecar_per_token = _packed_code_bytes(1, dim, key_bits) + float(metadata_bytes)
        if product:
            sidecar_per_token += _packed_code_bytes(1, dim, 1) + float(metadata_bytes)
        sidecar_per_token += _packed_code_bytes(1, dim, value_bits) + float(metadata_bytes)
        label = f"{'tqprod' if product else 'tq'}_k{key_bits}v{value_bits}_w{residual_window}"
        meta = {"family": "turboquant_proxy", "key_bits": key_bits, "value_bits": value_bits, "product_residual": product}
    elif name.startswith("tqpaperprod") or name.startswith("tqpaper"):
        key_bits, value_bits = _parse_tq_bits(name)
        product = name.startswith("tqpaperprod")
        score_source_keys = None
        score_mask = None
        if comp_count:
            key_hat, _, _ = _paper_tq_reconstruct(keys32[comp_mask], key_bits)
            value_hat, _, _ = _paper_tq_reconstruct(values32[comp_mask], value_bits)
            keys_hat[comp_mask] = key_hat
            values_hat[comp_mask] = value_hat
            if product:
                score_source_keys = keys32.copy()
                score_mask = comp_mask.copy()
        key_bytes_total = _packed_code_bytes(comp_count, dim, key_bits) + float(comp_count * int(metadata_bytes))
        if product:
            key_bytes_total += _packed_code_bytes(comp_count, dim, 1) + float(comp_count * int(metadata_bytes))
        value_bytes_total = _packed_code_bytes(comp_count, dim, value_bits) + float(comp_count * int(metadata_bytes))
        sidecar_per_token = _packed_code_bytes(1, dim, key_bits) + float(metadata_bytes)
        if product:
            sidecar_per_token += _packed_code_bytes(1, dim, 1) + float(metadata_bytes)
        sidecar_per_token += _packed_code_bytes(1, dim, value_bits) + float(metadata_bytes)
        label = f"{'tqpaperprod' if product else 'tqpaper'}_k{key_bits}v{value_bits}_w{residual_window}"
        meta = {
            "family": "turboquant_paper_proxy",
            "score_family": "tq_paper",
            "key_bits": key_bits,
            "value_bits": value_bits,
            "product_residual": product,
        }
    elif name.startswith("think_like"):
        ratio = max(0.0, min(_parse_float_param(name, prefix="r", default=0.5), 0.99))
        obs_window = _parse_observation_window(name, default=32)
        keep_dims = max(1, int(round(float(dim) * (1.0 - ratio))))
        if obs_queries is not None and obs_queries.size:
            q_score = np.mean(obs_queries.astype(np.float32, copy=False) ** 2, axis=(0, 1))
        else:
            q_score = np.ones((dim,), dtype=np.float32)
        k_score = np.mean(keys32[comp_mask].astype(np.float32, copy=False) ** 2, axis=0) if comp_count else np.zeros((dim,), dtype=np.float32)
        channel_scores = q_score.astype(np.float32, copy=False) * k_score.astype(np.float32, copy=False)
        keep_idx = np.argpartition(channel_scores, -keep_dims)[-keep_dims:]
        keep_mask = np.zeros((dim,), dtype=bool)
        keep_mask[keep_idx] = True
        if comp_count:
            comp_idx = np.flatnonzero(comp_mask)
            keys_hat[np.ix_(comp_idx, np.flatnonzero(~keep_mask))] = 0.0
        key_bytes_total = float(comp_count * keep_dims * int(key_bytes))
        value_bytes_total = float(comp_count * dim * int(value_bytes))
        sidecar_per_token = float(keep_dims * int(key_bytes) + dim * int(value_bytes))
        label = f"think_like_r{ratio:g}_obs{obs_window}_w{residual_window}"
        meta = {
            "family": "think_like",
            "key_channel_compression_ratio": float(ratio),
            "kept_key_dims": int(keep_dims),
            "obs_window": int(obs_window),
            "source_impl": "KVPress ThinK-style query/key channel scoring on saved trace",
        }
    elif name.startswith("pq_like"):
        subvecs, subbits = _parse_pq(name)
        key_bytes_total = 0.0
        value_bytes_total = 0.0
        if comp_count:
            key_hat, key_cost = _pq_reconstruct(
                keys32[comp_mask],
                subvecs=subvecs,
                subbits=subbits,
                seed=int(seed) + 17,
                max_iter=int(pq_iters),
            )
            value_hat, value_cost = _pq_reconstruct(
                values32[comp_mask],
                subvecs=subvecs,
                subbits=subbits,
                seed=int(seed) + 29,
                max_iter=int(pq_iters),
            )
            keys_hat[comp_mask] = key_hat
            values_hat[comp_mask] = value_hat
            key_bytes_total = float(key_cost)
            value_bytes_total = float(value_cost)
        sidecar_per_token = 2.0 * _packed_code_bytes(1, subvecs, subbits)
        label = f"pq_like_s{subvecs}b{subbits}_w{residual_window}"
        meta = {"family": "pq_like", "subvecs": subvecs, "subbits": subbits}
    elif name.startswith("commvq_like") or name.startswith("rq_like"):
        stages, bits = _parse_stage_bits(name, default_stages=4, default_bits=6)
        key_bytes_total = 0.0
        value_bytes_total = 0.0
        if comp_count:
            key_hat, key_cost = _residual_vq_reconstruct(
                keys32[comp_mask],
                stages=stages,
                bits=bits,
                seed=int(seed) + 37,
                max_iter=int(pq_iters),
            )
            value_hat, value_cost = _residual_vq_reconstruct(
                values32[comp_mask],
                stages=stages,
                bits=bits,
                seed=int(seed) + 41,
                max_iter=int(pq_iters),
            )
            keys_hat[comp_mask] = key_hat
            values_hat[comp_mask] = value_hat
            key_bytes_total = float(key_cost) + float(comp_count * int(metadata_bytes))
            value_bytes_total = float(value_cost) + float(comp_count * int(metadata_bytes))
        sidecar_per_token = 2.0 * (_packed_code_bytes(1, stages, bits) + float(metadata_bytes))
        label = f"commvq_like_m{stages}b{bits}_w{residual_window}"
        meta = {"family": "commvq_like", "stages": int(stages), "bits": int(bits)}
    elif name.startswith("gear_like"):
        bits = _parse_bits(name, default=2)
        rank = _parse_count(name, prefix="r", default=4)
        sparse_frac = _parse_float_param(name, prefix="sp", default=0.0)
        key_bytes_total = 0.0
        value_bytes_total = 0.0
        key_nnz = 0
        value_nnz = 0
        key_per_token = 0.0
        value_per_token = 0.0
        if comp_count:
            key_hat, key_cost, key_per_token, key_nnz = _gear_reconstruct(
                keys32[comp_mask],
                bits=bits,
                rank=rank,
                sparse_frac=sparse_frac,
                metadata_bytes=int(metadata_bytes),
                value_bytes=int(key_bytes),
            )
            value_hat, value_cost, value_per_token, value_nnz = _gear_reconstruct(
                values32[comp_mask],
                bits=bits,
                rank=rank,
                sparse_frac=sparse_frac,
                metadata_bytes=int(metadata_bytes),
                value_bytes=int(value_bytes),
            )
            keys_hat[comp_mask] = key_hat
            values_hat[comp_mask] = value_hat
            key_bytes_total = float(key_cost)
            value_bytes_total = float(value_cost)
        sidecar_per_token = float(key_per_token + value_per_token)
        label = f"gear_like_b{bits}_r{rank}_sp{sparse_frac:g}_w{residual_window}"
        meta = {
            "family": "gear_like",
            "bits": int(bits),
            "rank": int(rank),
            "sparse_frac": float(sparse_frac),
            "key_sparse_nnz": int(key_nnz),
            "value_sparse_nnz": int(value_nnz),
        }
    elif name.startswith("lowrank_svd"):
        rank = _parse_count(name, prefix="r", default=16)
        key_bytes_total = 0.0
        value_bytes_total = 0.0
        if comp_count:
            key_hat, key_cost = _lowrank_reconstruct(keys32[comp_mask], rank=rank)
            value_hat, value_cost = _lowrank_reconstruct(values32[comp_mask], rank=rank)
            keys_hat[comp_mask] = key_hat
            values_hat[comp_mask] = value_hat
            key_bytes_total = float(key_cost)
            value_bytes_total = float(value_cost)
        sidecar_per_token = float(2 * int(rank) * 2)
        label = f"lowrank_svd_r{rank}_w{residual_window}"
        meta = {"family": "lowrank_proxy", "rank": int(rank)}
    elif name.startswith("lexico_like"):
        atoms, active = _parse_dictionary_active(name, default_atoms=256, default_active=4)
        key_bytes_total = 0.0
        value_bytes_total = 0.0
        if comp_count:
            key_hat, key_cost = _sparse_code_reconstruct(
                keys32[comp_mask],
                atoms=atoms,
                active=active,
                seed=int(seed) + 53,
                max_iter=int(pq_iters),
            )
            value_hat, value_cost = _sparse_code_reconstruct(
                values32[comp_mask],
                atoms=atoms,
                active=active,
                seed=int(seed) + 59,
                max_iter=int(pq_iters),
            )
            keys_hat[comp_mask] = key_hat
            values_hat[comp_mask] = value_hat
            key_bytes_total = float(key_cost)
            value_bytes_total = float(value_cost)
        sidecar_per_token = float(2 * (int(active) * (2 + 1) + 4))
        label = f"lexico_like_d{atoms}a{active}_w{residual_window}"
        meta = {"family": "lexico_like", "dictionary_atoms": int(atoms), "active_atoms": int(active)}
    elif name.startswith("salient_quant"):
        bits = _parse_bits(name, default=3)
        keep = _parse_count(name, prefix="k", default=2048)
        salient_mask = _select_top_norm_mask(keys32, values32, exact_mask, keep)
        quant_mask = ~salient_mask
        quant_count = int(np.count_nonzero(quant_mask))
        exact_count = int(np.count_nonzero(salient_mask))
        exact_bytes = float(exact_count * dim * (int(key_bytes) + int(value_bytes)))
        if quant_count:
            keys_hat[quant_mask] = _quantize_per_token(keys32[quant_mask], bits)
            values_hat[quant_mask] = _quantize_per_token(values32[quant_mask], bits)
        key_bytes_total = _packed_code_bytes(quant_count, dim, bits) + _scale_zero_bytes(quant_count, 1, int(metadata_bytes))
        value_bytes_total = _packed_code_bytes(quant_count, dim, bits) + _scale_zero_bytes(quant_count, 1, int(metadata_bytes))
        sidecar_per_token = 2.0 * (_packed_code_bytes(1, dim, bits) + _scale_zero_bytes(1, 1, int(metadata_bytes)))
        label = f"salient_quant_b{bits}_k{keep}_w{residual_window}"
        meta = {
            "family": "salient_mixed_precision",
            "bits": int(bits),
            "salient_tokens": int(keep),
            "exact_tokens": int(exact_count),
            "compressed_tokens": int(quant_count),
        }
    else:
        raise ValueError(f"unknown compression method: {method}")

    _copy_exact_window(keys_hat, values_hat, keys32, values32, exact_mask)
    query_read_mb = (
        float(query_read_override)
        if query_read_override is not None
        else float(exact_bytes + key_bytes_total + value_bytes_total) / MB
    )
    update_mb_per_token = (
        float(update_override)
        if update_override is not None
        else float(update_exact_window if residual_window > 0 else sidecar_per_token) / MB
    )
    meta.update(
        {
            "method": label,
            "context_len": context_len,
            "head_dim": dim,
            "static_prefix": int(static_prefix),
            "residual_window": int(residual_window),
            "exact_tokens": exact_count,
            "compressed_tokens": comp_count,
            "exact_window_MB": exact_bytes / MB,
            "compressed_key_MB": float(key_bytes_total) / MB,
            "compressed_value_MB": float(value_bytes_total) / MB,
            "online_update_MB_per_token": update_mb_per_token,
            "proxy_not_faithful_paper_impl": bool(meta.get("proxy_not_faithful_paper_impl", True)),
        }
    )
    return CompressedKV(
        method=label,
        keys_hat=keys_hat,
        values_hat=values_hat,
        active_mask=None,
        score_source_keys=locals().get("score_source_keys", None),
        score_mask=locals().get("score_mask", None),
        query_read_mb=query_read_mb,
        online_update_mb_per_token=update_mb_per_token,
        metadata=meta,
    )


def method_display_name(method: str) -> str:
    mapping = [
        ("dense", "Dense fp16"),
        ("kivi", "KIVI"),
        ("kvquant_like", "KVQuant-like clipped scalar"),
        ("per_token_kv", "Per-token scalar"),
        ("pmkvq_like", "PM-KVQ-like progressive scalar"),
        ("kitty_like", "Kitty-like channel-promoted scalar"),
        ("tada_like", "TaDA-like mean-centered scalar"),
        ("tiered_quant", "Tiered mixed precision"),
        ("lookat_like", "LOOKAT-like sparse channels"),
        ("kvtc_like", "KVTC-like transform coding"),
        ("freqkv_like", "FreqKV-like transform coding"),
        ("million_like", "MILLION-like PQ"),
        ("cam_like", "CaM-like merge/prune"),
        ("zeromerge_like", "ZeroMerge-like merging"),
        ("tqprod", "TurboQuant product proxy"),
        ("tqpaperprod", "TurboQuant paper QJL proxy"),
        ("tqpaper", "TurboQuant paper proxy"),
        ("tq", "TurboQuant proxy"),
        ("think_like", "ThinK-like key-channel compression"),
        ("pq_like", "PQ/VQ-like"),
        ("commvq_like", "CommVQ/additive VQ-like"),
        ("rq_like", "Residual VQ-like"),
        ("gear_like", "GEAR-like"),
        ("lowrank_svd", "Low-rank proxy"),
        ("lexico_like", "Lexico sparse coding-like"),
        ("salient_quant", "Salient mixed precision"),
        ("tova", "TOVA trace retention"),
        ("cur", "CUR trace retention"),
        ("knorm", "Knorm trace retention"),
        ("keydiff", "KeyDiff trace retention"),
        ("leverage", "Leverage-score trace retention"),
        ("expected_attn", "Expected-attention trace retention"),
        ("critical_snap", "Critical SnapKV trace retention"),
        ("chunk_snap", "ChunkKV SnapKV trace retention"),
        ("lagkv", "LagKV trace retention"),
        ("compactor", "Compactor trace retention"),
        ("sink_recent", "Sink + recent retention"),
        ("recent", "Recent retention"),
        ("l2ret", "L2-norm retention"),
        ("h2o", "H2O trace heavy hitter"),
        ("snapkv", "SnapKV trace proxy"),
        ("kvzip", "KVzip trace proxy"),
        ("rocket_snap", "RocketKV SnapKV trace proxy"),
    ]
    for prefix, label in mapping:
        if method.startswith(prefix):
            return label
    return method


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_method: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_method.setdefault(str(row["method"]), []).append(row)
    out = []
    for method, group in sorted(by_method.items()):
        vals = {
            "method": method,
            "family": str(group[0].get("family", "")),
            "n": len(group),
        }
        summary_cols = set(
            (
            "query_read_MB_per_head",
            "online_update_MB_per_token",
            "step_MB_per_head_query",
            "attention_relL2",
            "attention_cosine",
            "o_proj_relL2",
            "o_proj_cosine",
            "exact_tokens",
            "compressed_tokens",
            "distribution_comparable",
            "token_probability_comparable",
            )
        )
        for row in group:
            for key in row:
                if key.startswith(("logit_", "prob_")):
                    summary_cols.add(key)
        for col in sorted(summary_cols):
            arr = np.asarray(
                [
                    float(r[col])
                    for r in group
                    if col in r and str(r.get(col, "")).strip() and str(r.get(col, "")).lower() != "nan"
                ],
                dtype=np.float64,
            )
            if arr.size:
                vals[f"mean_{col}"] = float(np.mean(arr))
                vals[f"max_{col}"] = float(np.max(arr))
                vals[f"min_{col}"] = float(np.min(arr))
                vals[f"p95_{col}"] = float(np.quantile(arr, 0.95))
                vals[f"p99_{col}"] = float(np.quantile(arr, 0.99))
        out.append(vals)
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run() -> None:
    parser = argparse.ArgumentParser(description="Compare KV-cache compression proxies on saved real Q/K/V traces.")
    parser.add_argument("--qkv_trace", required=True)
    parser.add_argument("--x_trace", default="")
    parser.add_argument(
        "--model_snapshot",
        default=".hf_cache/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--decode_lengths", default="500,1000,2000,4000,8000,16000,32000,64000,128000")
    parser.add_argument("--max_qidx_per_decode", type=int, default=1)
    parser.add_argument("--heads", default="", help="Comma-separated query heads. Empty means all heads.")
    parser.add_argument(
        "--methods",
        default=(
            "dense,"
            "kivi_b2_g32_w128,kivi_b4_g32_w128,kivi_b2_g32_w2048,kivi_b4_g32_w2048,"
            "kvquant_like_b3_clip0p1_w128,kvquant_like_b4_clip0p1_w128,"
            "per_token_kv_b3_w128,per_token_kv_b4_w128,"
            "tq_k3v3_w128,tqprod_k3v3_w128,tqprod_k4v4_w128,"
            "pq_like_s4b4_w128,pq_like_s4b6_w128"
        ),
    )
    parser.add_argument("--static_prefix", type=int, default=128)
    parser.add_argument("--residual_window", type=int, default=128)
    parser.add_argument("--key_bytes", type=int, default=2)
    parser.add_argument("--value_bytes", type=int, default=2)
    parser.add_argument("--metadata_bytes", type=int, default=2)
    parser.add_argument("--pq_iters", type=int, default=3)
    parser.add_argument("--prob_topk_sizes", default="64,512,2048")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip_post_proj", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True), encoding="utf-8")

    device = torch.device(args.device)
    torch.set_grad_enabled(False)
    trace = load_trace(args.qkv_trace)
    q_indices = trace.q_indices_for_decodes(parse_csv_ints(args.decode_lengths))
    if int(args.max_qidx_per_decode) > 0:
        limited: list[int] = []
        counts: dict[int, int] = {}
        for qidx in q_indices:
            decode = int(trace.decode_tokens_for_qidx(int(qidx)))
            seen = counts.get(decode, 0)
            if seen >= int(args.max_qidx_per_decode):
                continue
            limited.append(int(qidx))
            counts[decode] = seen + 1
        q_indices = limited
    if not q_indices:
        raise ValueError("no query indices selected")
    heads = parse_csv_ints(args.heads) if str(args.heads).strip() else list(range(int(trace.num_heads)))
    methods = [part.strip() for part in str(args.methods).split(",") if part.strip()]
    prob_topk_sizes = tuple(parse_csv_ints(args.prob_topk_sizes))

    wo = None
    if not bool(args.skip_post_proj):
        if not str(args.x_trace).strip():
            raise ValueError("--x_trace is required unless --skip_post_proj is set")
        x_data = np.load(args.x_trace, mmap_mode="r")
        x_meta = json.loads(str(x_data["metadata"].item()))
        layer_idx = int(x_meta["layer_idx"])
        model_dir = PROJECT_ROOT / args.model_snapshot
        weight_map = load_weight_index(model_dir)
        wo = load_safetensor_weight(model_dir, weight_map, f"model.layers.{layer_idx}.self_attn.o_proj.weight", device)
    else:
        layer_idx = int(trace.metadata.get("layer_idx", -1))

    per_head_rows: list[dict[str, object]] = []
    layer_rows: list[dict[str, object]] = []
    kv_cache: dict[tuple[int, int, str], CompressedKV] = {}

    for qidx in q_indices:
        position = int(trace.positions[int(qidx)])
        context_len = position + 1
        decode_tokens = int(trace.decode_tokens_for_qidx(int(qidx)))
        dense_by_head: dict[int, np.ndarray] = {}
        approx_by_method: dict[str, dict[int, np.ndarray]] = {m: {} for m in methods}

        heads_by_kv: dict[int, list[int]] = {}
        for head in heads:
            heads_by_kv.setdefault(int(trace.kv_head_for(head)), []).append(int(head))

        for kv_head in sorted(heads_by_kv):
            keys_np = trace.keys[int(kv_head), :context_len].astype(np.float32, copy=False)
            values_np = trace.values[int(kv_head), :context_len].astype(np.float32, copy=False)
            for method in methods:
                cache_key = (int(kv_head), int(context_len), str(method))
                if cache_key not in kv_cache:
                    obs_queries = None
                    if trace.graph_queries is not None and trace.graph_positions is not None:
                        graph_positions = np.asarray(trace.graph_positions)
                        obs_window = _parse_observation_window(str(method), default=64)
                        graph_idx = np.flatnonzero(graph_positions < int(context_len))
                        if graph_idx.size:
                            graph_idx = graph_idx[-int(obs_window) :]
                            obs_queries = np.stack(
                                [trace.graph_queries[int(head), graph_idx].astype(np.float32, copy=False) for head in heads_by_kv[int(kv_head)]],
                                axis=0,
                            )
                    kv_cache[cache_key] = build_compressed_kv(
                        method,
                        keys=keys_np,
                        values=values_np,
                        static_prefix=int(args.static_prefix),
                        default_residual_window=int(args.residual_window),
                        key_bytes=int(args.key_bytes),
                        value_bytes=int(args.value_bytes),
                        metadata_bytes=int(args.metadata_bytes),
                        pq_iters=int(args.pq_iters),
                        seed=2025 + 1009 * int(kv_head) + int(context_len),
                        obs_queries=obs_queries,
                    )

        for kv_head, kv_heads in sorted(heads_by_kv.items()):
            keys_np = trace.keys[kv_head, :context_len].astype(np.float32, copy=False)
            values_np = trace.values[kv_head, :context_len].astype(np.float32, copy=False)
            queries_np = np.stack(
                [trace.queries[int(head), int(qidx)].astype(np.float32, copy=False) for head in kv_heads],
                axis=0,
            )
            dense_group = attention_outputs(keys_np, values_np, queries_np)
            dense_scores_probs = [scores_probs_for_keys(keys_np, queries_np[int(local_idx)]) for local_idx in range(len(kv_heads))]
            for local_idx, head in enumerate(kv_heads):
                dense_by_head[int(head)] = dense_group[int(local_idx)]

            for requested_method in methods:
                comp = kv_cache[(int(kv_head), int(context_len), requested_method)]
                if comp.method == "dense":
                    approx_group = dense_group
                else:
                    approx_group = compressed_attention_outputs(comp, queries_np)
                for local_idx, head in enumerate(kv_heads):
                    approx_head = approx_group[int(local_idx)]
                    approx_by_method[requested_method][int(head)] = approx_head
                    metrics = _output_error_metrics(dense_group[int(local_idx)], approx_head)
                    dense_scores, dense_probs = dense_scores_probs[int(local_idx)]
                    approx_scores, approx_probs, common_scores, common_mask = compressed_scores_probs(
                        comp,
                        queries_np[int(local_idx)],
                        context_len=int(context_len),
                    )
                    dist_metrics = attention_distribution_error_metrics(
                        dense_scores,
                        dense_probs,
                        approx_scores,
                        approx_probs,
                        topk_sizes=prob_topk_sizes,
                    )
                    if common_scores is not None and common_mask is not None and np.any(common_mask):
                        dist_metrics.update(
                            _logit_error_metrics(
                                dense_scores[np.asarray(common_mask, dtype=bool)],
                                common_scores,
                                prefix="logit_common",
                            )
                        )
                    row = {
                        "requested_method": requested_method,
                        "method": comp.method,
                        "display_name": method_display_name(comp.method),
                        "family": comp.metadata.get("family", ""),
                        "qidx": int(qidx),
                        "decode_tokens": int(decode_tokens),
                        "context_len": int(context_len),
                        "head": int(head),
                        "kv_head": int(kv_head),
                        "layer_idx": int(layer_idx),
                        "query_read_MB_per_head": float(comp.query_read_mb),
                        "online_update_MB_per_token": float(comp.online_update_mb_per_token),
                        "step_MB_per_head_query": float(comp.query_read_mb + comp.online_update_mb_per_token),
                        "attention_relL2": float(metrics["output_relative_l2"]),
                        "attention_cosine": float(metrics["output_cosine"]),
                        "distribution_comparable": int(approx_scores is not None and approx_probs is not None),
                        "token_probability_comparable": int(approx_probs is not None),
                        "exact_tokens": int(comp.metadata.get("exact_tokens", 0)),
                        "compressed_tokens": int(comp.metadata.get("compressed_tokens", 0)),
                    }
                    row.update({key: float(value) for key, value in dist_metrics.items()})
                    row.update({f"meta_{k}": v for k, v in comp.metadata.items() if isinstance(v, (str, int, float, bool))})
                    per_head_rows.append(row)

        if wo is not None:
            dense_concat = np.zeros((int(trace.num_heads) * int(trace.head_dim),), dtype=np.float32)
            for head, out in dense_by_head.items():
                dense_concat[int(head) * int(trace.head_dim) : (int(head) + 1) * int(trace.head_dim)] = out
            dense_proj = project_full(dense_concat, wo, device)
            for requested_method in methods:
                approx_concat = np.zeros_like(dense_concat)
                for head, out in approx_by_method[requested_method].items():
                    approx_concat[int(head) * int(trace.head_dim) : (int(head) + 1) * int(trace.head_dim)] = out
                approx_proj = project_full(approx_concat, wo, device)
                metrics = _output_error_metrics(dense_proj, approx_proj)
                comp0 = kv_cache[(int(trace.kv_head_for(heads[0])), context_len, requested_method)]
                layer_rows.append(
                    {
                        "requested_method": requested_method,
                        "method": comp0.method,
                        "display_name": method_display_name(comp0.method),
                        "family": comp0.metadata.get("family", ""),
                        "qidx": int(qidx),
                        "decode_tokens": int(decode_tokens),
                        "context_len": int(context_len),
                        "layer_idx": int(layer_idx),
                        "heads_evaluated": len(heads),
                        "query_read_MB_per_head": float(comp0.query_read_mb),
                        "online_update_MB_per_token": float(comp0.online_update_mb_per_token),
                        "step_MB_per_head_query": float(comp0.query_read_mb + comp0.online_update_mb_per_token),
                        "o_proj_relL2": float(metrics["output_relative_l2"]),
                        "o_proj_cosine": float(metrics["output_cosine"]),
                    }
                )

    summary_rows = summarize(per_head_rows)
    if layer_rows:
        layer_summary = summarize(layer_rows)
        by_method = {str(row["method"]): row for row in summary_rows}
        for layer_row in layer_summary:
            target = by_method.setdefault(str(layer_row["method"]), {"method": layer_row["method"], "family": layer_row.get("family", "")})
            for key, value in layer_row.items():
                if key in {"method", "family", "n"}:
                    continue
                target[key] = value
        summary_rows = list(by_method.values())

    write_csv(out_dir / "per_head_kv_compression.csv", per_head_rows)
    write_csv(out_dir / "layer_kv_compression.csv", layer_rows)
    write_csv(out_dir / "summary.csv", summary_rows)
    summary = {
        "elapsed_sec": float(time.perf_counter() - t0),
        "qkv_trace": str(args.qkv_trace),
        "x_trace": str(args.x_trace),
        "decode_lengths": [int(trace.decode_tokens_for_qidx(q)) for q in q_indices],
        "heads": heads,
        "methods": methods,
        "summary": summary_rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    run()
