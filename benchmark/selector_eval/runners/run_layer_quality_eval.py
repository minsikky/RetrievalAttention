#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import safe_open

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.selector_eval.data.trace import attention_probs, load_trace, static_tokens, unique_tokens
from benchmark.selector_eval.gpu.run_gpu_paged_pq_eval import (
    build_page_pq_gpu,
    parse_csv_ints,
    pq_page_scores,
    rank_paged_pq,
    selected_plus_tail_output,
)
from benchmark.selector_eval.runners.diagnose_layer_heads import _build_value_vpq_sidecars, _compressed_tail_output
from benchmark.selector_eval.metrics.attention import _output_error_metrics


MB = 1024.0 * 1024.0


def load_weight_index(model_dir: Path) -> dict[str, str]:
    index_path = model_dir / "model.safetensors.index.json"
    data = json.loads(index_path.read_text())
    return {str(k): str(v) for k, v in data["weight_map"].items()}


def load_safetensor_weight(model_dir: Path, weight_map: dict[str, str], name: str, device: torch.device) -> torch.Tensor:
    shard = model_dir / weight_map[name]
    with safe_open(shard, framework="pt", device="cpu") as f:
        return f.get_tensor(name).to(device=device, dtype=torch.float32, non_blocking=True)


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    y = x.float() * torch.rsqrt(torch.mean(x.float() * x.float(), dim=-1, keepdim=True) + float(eps))
    return y * weight.float()


def mlp(x: torch.Tensor, gate_proj: torch.Tensor, up_proj: torch.Tensor, down_proj: torch.Tensor) -> torch.Tensor:
    return F.linear(F.silu(F.linear(x, gate_proj)) * F.linear(x, up_proj), down_proj)


def dense_attention_output(keys_np: np.ndarray, values_np: np.ndarray, query_np: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scores_np, probs_np = attention_probs(keys_np, query_np)
    out = probs_np.astype(np.float32) @ values_np.astype(np.float32, copy=False)
    return scores_np, probs_np, out.astype(np.float32, copy=False)


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


def _page_tokens(index) -> np.ndarray:
    parts = [
        np.arange(int(page.start), int(page.start) + int(page.size), dtype=np.int64)
        for page in index.pages
        if int(page.size) > 0
    ]
    return np.concatenate(parts) if parts else np.empty((0,), dtype=np.int64)


def _sparq_rank_tokens(
    *,
    keys_np: np.ndarray,
    query_np: np.ndarray,
    index,
    rank: int,
    key_bytes: int,
    index_bytes: int,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    tokens = _page_tokens(index)
    if tokens.size == 0:
        return tokens, np.empty((0,), dtype=np.float32), 0.0, 0.0
    rank = min(max(1, int(rank)), int(query_np.shape[0]))
    dims = np.argsort(-np.abs(query_np), kind="stable")[:rank]
    coverage = max(float(np.abs(query_np[dims]).sum()) / max(float(np.abs(query_np).sum()), 1e-20), 1e-6)
    # Scale partial-dot scores back to the full-dot scale expected by the
    # shared confidence calibration. Calibration can still correct query-local
    # bias, but this keeps initial ranking comparable to K-PQ scores.
    scores = (keys_np[tokens[:, None], dims] @ query_np[dims]).astype(np.float32, copy=False) / float(coverage)
    order = np.argsort(-scores, kind="stable")
    selector_mb = float(rank * int(index_bytes) + tokens.size * rank * int(key_bytes)) / MB
    return tokens[order].astype(np.int64, copy=False), scores[order].astype(np.float32, copy=False), selector_mb, float(coverage)


def _quest_page_scores(
    *,
    keys_np: np.ndarray,
    query_np: np.ndarray,
    index,
    rank: int,
) -> tuple[list[tuple[int, float]], float]:
    if not index.pages:
        return [], 0.0
    rank = min(max(1, int(rank)), int(query_np.shape[0]))
    dims = np.argsort(-np.abs(query_np), kind="stable")[:rank]
    q = query_np[dims].astype(np.float32, copy=False)
    coverage = max(float(np.abs(q).sum()) / max(float(np.abs(query_np).sum()), 1e-20), 1e-6)
    page_scores: list[tuple[int, float]] = []
    for page_id, page in enumerate(index.pages):
        start = int(page.start)
        end = min(start + int(page.size), int(keys_np.shape[0]))
        if end <= start:
            page_scores.append((int(page_id), float("-inf")))
            continue
        vals = keys_np[start:end, :][:, dims].astype(np.float32, copy=False)
        mins = vals.min(axis=0)
        maxs = vals.max(axis=0)
        bound = np.where(q >= 0.0, maxs, mins)
        page_scores.append((int(page_id), float(np.dot(bound, q) / coverage)))
    page_scores.sort(key=lambda item: (-item[1], item[0]))
    return page_scores, float(coverage)


def _rank_quest_pages(
    *,
    keys_np: np.ndarray,
    query_np: np.ndarray,
    index,
    rank: int,
    key_bytes: int,
    index_bytes: int,
) -> tuple[np.ndarray, np.ndarray, float, int, float]:
    page_scores, coverage = _quest_page_scores(keys_np=keys_np, query_np=query_np, index=index, rank=rank)
    tokens: list[int] = []
    scores: list[float] = []
    for page_id, score in page_scores:
        page = index.pages[int(page_id)]
        page_tokens = range(int(page.start), int(page.start) + int(page.size))
        tokens.extend(int(tok) for tok in page_tokens)
        scores.extend(float(score) for _ in range(int(page.size)))
    rank = min(max(1, int(rank)), int(query_np.shape[0]))
    selector_mb = float(rank * int(index_bytes) + len(index.pages) * rank * 2 * int(key_bytes)) / MB
    return (
        np.asarray(tokens, dtype=np.int64),
        np.asarray(scores, dtype=np.float32),
        selector_mb,
        int(len(index.pages)),
        float(coverage),
    )


def _rank_quest_pq(
    *,
    query: torch.Tensor,
    keys_np: np.ndarray,
    query_np: np.ndarray,
    index,
    rank: int,
    nprobes: list[int],
    budget: int,
    key_bytes: int,
    subbits: int,
    index_bytes: int,
) -> tuple[np.ndarray, np.ndarray, float, int, float]:
    page_scores, coverage = _quest_page_scores(keys_np=keys_np, query_np=query_np, index=index, rank=rank)
    if not page_scores:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float32), 0.0, 0, float(coverage)
    probe_choices = sorted(set(max(1, int(p)) for p in nprobes))
    chosen_nprobe = min(probe_choices[-1], len(page_scores))
    for nprobe in probe_choices:
        chosen_nprobe = min(int(nprobe), len(page_scores))
        candidate_count = sum(int(index.pages[pid].size) for pid, _score in page_scores[:chosen_nprobe])
        if candidate_count >= int(budget):
            break

    scanned_page_ids = {int(pid) for pid, _score in page_scores[:chosen_nprobe]}
    scanned_tokens = []
    scanned_scores = []
    for page_id, _page_score in page_scores[:chosen_nprobe]:
        tokens_t, scores_t = pq_page_scores(query, index.pages[int(page_id)])
        scanned_tokens.append(tokens_t.detach().cpu().numpy().astype(np.int64, copy=False))
        scanned_scores.append(scores_t.detach().cpu().numpy().astype(np.float32, copy=False))
    if scanned_tokens:
        scanned_tok = np.concatenate(scanned_tokens)
        scanned_score = np.concatenate(scanned_scores)
        order = np.argsort(-scanned_score, kind="stable")
        ranked_tokens = [int(tok) for tok in scanned_tok[order].tolist()]
        ranked_scores = [float(score) for score in scanned_score[order].tolist()]
    else:
        ranked_tokens = []
        ranked_scores = []

    # Append unscanned pages by QUEST page score so tail confidence still sees
    # an explicit score proxy for the full indexed tail. Selected tokens from
    # this suffix are allowed, but they are selected by page-bound score only.
    for page_id, page_score in page_scores:
        if int(page_id) in scanned_page_ids:
            continue
        page = index.pages[int(page_id)]
        for tok in range(int(page.start), int(page.start) + int(page.size)):
            ranked_tokens.append(int(tok))
            ranked_scores.append(float(page_score))

    rank = min(max(1, int(rank)), int(query_np.shape[0]))
    code_bytes = 1 if int(subbits) <= 8 else 2
    selector_bytes = float(rank * int(index_bytes) + len(index.pages) * rank * 2 * int(key_bytes))
    for page_id in scanned_page_ids:
        page = index.pages[int(page_id)]
        selector_bytes += float(page.codebooks.numel() * int(key_bytes) + page.codes.numel() * code_bytes)
    return (
        np.asarray(ranked_tokens, dtype=np.int64),
        np.asarray(ranked_scores, dtype=np.float32),
        selector_bytes / MB,
        int(chosen_nprobe),
        float(coverage),
    )


def _selected_for_budget(
    *,
    base: list[int],
    ranked_cpu: np.ndarray,
    budget: int,
    context_len: int,
) -> np.ndarray:
    base_set = set(int(tok) for tok in base)
    add = [int(tok) for tok in ranked_cpu.tolist() if int(tok) < int(context_len) and int(tok) not in base_set][: int(budget)]
    return np.asarray(unique_tokens(base + add, context_len=int(context_len)), dtype=np.int64)


def _nan_output_metrics() -> dict[str, float]:
    return {
        "output_cosine": float("nan"),
        "output_relative_l2": float("nan"),
        "output_rmsnorm_relative_l2": float("nan"),
        "output_centered_cosine": float("nan"),
        "output_mean_abs_relative_error": float("nan"),
        "output_p95_abs_relative_error": float("nan"),
        "output_p99_abs_relative_error": float("nan"),
        "output_max_abs_relative_error": float("nan"),
        "output_linf_relative": float("nan"),
    }


def _proxy_selected_mass(
    *,
    selected_cpu: np.ndarray,
    ranked_cpu: np.ndarray,
    ranked_scores_cpu: np.ndarray,
    scores_np: np.ndarray,
    query_dim: int,
    tail_score_scale: float = 1.0,
    tail_score_bias: float = 0.0,
) -> tuple[float, float]:
    """Selector-only mass proxy.

    Selected tokens use exact scores because exact K has already been fetched.
    Unselected indexed tokens use PQ selector scores. This deliberately does
    not read unselected exact K or true dense probabilities.
    """

    selected_set = set(int(tok) for tok in selected_cpu.tolist())
    exact_scores = scores_np[selected_cpu].astype(np.float64, copy=False) if selected_cpu.size else np.asarray([], dtype=np.float64)
    tail_scores = np.asarray(
        [
            float(tail_score_scale) * (float(score) / float(np.sqrt(float(query_dim)))) + float(tail_score_bias)
            for tok, score in zip(ranked_cpu.tolist(), ranked_scores_cpu.tolist(), strict=False)
            if int(tok) not in selected_set
        ],
        dtype=np.float64,
    )
    if exact_scores.size == 0 and tail_scores.size == 0:
        return 0.0, 0.0
    max_score = max(
        float(np.max(exact_scores)) if exact_scores.size else -np.inf,
        float(np.max(tail_scores)) if tail_scores.size else -np.inf,
    )
    exact_w = float(np.exp(exact_scores - max_score).sum()) if exact_scores.size else 0.0
    tail_w = float(np.exp(tail_scores - max_score).sum()) if tail_scores.size else 0.0
    total = max(exact_w + tail_w, 1e-20)
    return exact_w / total, tail_w / total


def _selected_only_output(
    keys_np: np.ndarray,
    values_np: np.ndarray,
    query_np: np.ndarray,
    selected_cpu: np.ndarray,
    *,
    values_override: np.ndarray | None = None,
) -> np.ndarray:
    if selected_cpu.size == 0:
        return np.zeros((values_np.shape[-1],), dtype=np.float32)
    selected_values = values_override if values_override is not None else values_np[selected_cpu]
    scores, probs = attention_probs(keys_np[selected_cpu], query_np)
    return (probs.astype(np.float32) @ selected_values.astype(np.float32, copy=False)).astype(np.float32, copy=False)


def _selected_output_from_scores(
    values_np: np.ndarray,
    selected_cpu: np.ndarray,
    selected_scores: np.ndarray,
    *,
    values_override: np.ndarray | None = None,
) -> np.ndarray:
    if selected_cpu.size == 0:
        return np.zeros((values_np.shape[-1],), dtype=np.float32)
    selected_values = values_override if values_override is not None else values_np[selected_cpu]
    scores = selected_scores.astype(np.float64, copy=False)
    weights = np.exp(scores - float(np.max(scores)))
    weights /= max(float(weights.sum()), 1e-20)
    return (weights.astype(np.float32, copy=False) @ selected_values.astype(np.float32, copy=False)).astype(
        np.float32,
        copy=False,
    )


def _ranked_score_map(ranked_cpu: np.ndarray, ranked_scores_cpu: np.ndarray, query_dim: int) -> dict[int, float]:
    scale = float(np.sqrt(float(query_dim)))
    return {
        int(tok): float(score) / scale
        for tok, score in zip(ranked_cpu.tolist(), ranked_scores_cpu.tolist(), strict=False)
    }


def _band_calibrated_selected_scores(
    *,
    selected_cpu: np.ndarray,
    ranked_cpu: np.ndarray,
    ranked_scores_cpu: np.ndarray,
    exact_scores_np: np.ndarray,
    query_dim: int,
    probes: int,
    bands: int,
    exact_selector_mass: float = 0.0,
    min_exact_top: int = 0,
    max_exact_top: int = 0,
) -> tuple[np.ndarray, int, int, int]:
    """Calibrate selector logits for selected tokens using rank-band probes.

    Exact fallback is used for selected tokens that do not have a selector score
    proxy, e.g. static/pending tokens. Probe tokens are also kept exact.
    """

    if selected_cpu.size == 0:
        return np.empty((0,), dtype=np.float64), 0, 0, 0
    approx_by_token = _ranked_score_map(ranked_cpu, ranked_scores_cpu, query_dim)
    selected_set = set(int(tok) for tok in selected_cpu.tolist())
    ordered = [int(tok) for tok in ranked_cpu.tolist() if int(tok) in selected_set]
    seen = set(ordered)
    ordered.extend(int(tok) for tok in selected_cpu.tolist() if int(tok) not in seen)
    token_to_pos = {int(tok): idx for idx, tok in enumerate(selected_cpu.tolist())}
    scores = exact_scores_np[selected_cpu].astype(np.float64, copy=True)
    compressible = np.asarray([int(tok) in approx_by_token for tok in selected_cpu.tolist()], dtype=bool)
    compressed_count = int(np.count_nonzero(compressible))
    if compressed_count == 0:
        return scores, int(selected_cpu.size), 0, 0

    probe_count = min(max(0, int(probes)), compressed_count)
    exact_mask = ~compressible
    approx_scores_for_selected = np.asarray(
        [
            float(approx_by_token[int(tok)]) if int(tok) in approx_by_token else -float("inf")
            for tok in selected_cpu.tolist()
        ],
        dtype=np.float64,
    )
    exact_selector_count = 0
    if float(exact_selector_mass) > 0.0 and bool(np.any(compressible)):
        compressible_pos = np.nonzero(compressible)[0]
        selector_scores = approx_scores_for_selected[compressible_pos]
        order_local = np.argsort(-selector_scores, kind="stable")
        ordered_pos = compressible_pos[order_local]
        shifted = selector_scores[order_local] - float(np.max(selector_scores[order_local]))
        probs = np.exp(shifted)
        probs /= max(float(probs.sum()), 1e-20)
        cumulative = np.cumsum(probs)
        target = float(max(0.0, min(1.0, exact_selector_mass)))
        exact_selector_count = int(np.searchsorted(cumulative, target, side="left") + 1)
        exact_selector_count = max(int(min_exact_top), exact_selector_count)
        if int(max_exact_top) > 0:
            exact_selector_count = min(int(max_exact_top), exact_selector_count)
        exact_selector_count = max(0, min(int(ordered_pos.size), exact_selector_count))
        if exact_selector_count > 0:
            exact_mask[ordered_pos[:exact_selector_count]] = True
    if probe_count > 0:
        ordered_compressible = np.asarray([tok for tok in ordered if int(tok) in approx_by_token], dtype=np.int64)
        band_count = max(1, min(int(bands), int(ordered_compressible.size)))
        ordered_bands = [arr.astype(np.int64, copy=False) for arr in np.array_split(ordered_compressible, band_count)]
        per_band = max(1, int(np.ceil(float(probe_count) / float(band_count))))
        for band in ordered_bands:
            if band.size == 0:
                continue
            if per_band >= int(band.size):
                probe_tokens = band
            else:
                positions = np.unique(np.linspace(0, int(band.size) - 1, num=per_band, dtype=np.int64))
                probe_tokens = band[positions]
            exact_mask[[token_to_pos[int(tok)] for tok in probe_tokens.tolist()]] = True
            x = np.asarray([approx_by_token[int(tok)] for tok in probe_tokens.tolist()], dtype=np.float64)
            y = exact_scores_np[probe_tokens].astype(np.float64, copy=False)
            x_mean = float(np.mean(x))
            y_mean = float(np.mean(y))
            x_var = float(np.mean((x - x_mean) * (x - x_mean)))
            if x_var > 1e-12:
                slope = float(np.mean((x - x_mean) * (y - y_mean)) / x_var)
                intercept = y_mean - slope * x_mean
            else:
                slope = 1.0
                intercept = y_mean - x_mean
            for tok in band.tolist():
                pos = token_to_pos[int(tok)]
                scores[pos] = slope * float(approx_by_token[int(tok)]) + intercept
            for tok in probe_tokens.tolist():
                scores[token_to_pos[int(tok)]] = float(exact_scores_np[int(tok)])
    else:
        for tok in ordered:
            if int(tok) in approx_by_token:
                scores[token_to_pos[int(tok)]] = float(approx_by_token[int(tok)])

    if bool(np.any(exact_mask)):
        scores[exact_mask] = exact_scores_np[selected_cpu[exact_mask]].astype(np.float64, copy=False)
    exact_count = int(np.count_nonzero(exact_mask))
    compressed_logits = int(selected_cpu.size) - exact_count
    return scores, exact_count, compressed_logits, probe_count


def _vpq_values_for_tokens(
    *,
    index,
    values_np: np.ndarray,
    tokens: np.ndarray,
    subbits: int,
    value_subvecs: int,
    value_subbits: int,
    value_bytes: int,
) -> tuple[np.ndarray, float, float]:
    if tokens.size == 0:
        return np.zeros((0, values_np.shape[-1]), dtype=np.float32), 0.0, 0.0
    actual_value_subbits = int(value_subbits) if int(value_subbits) > 0 else int(subbits)
    value_sidecars = _build_value_vpq_sidecars(
        index,
        values_np,
        int(subbits),
        value_subvecs=int(value_subvecs),
        value_subbits=actual_value_subbits,
    )
    starts = np.asarray([int(page.start) for page in index.pages], dtype=np.int64)
    sizes = np.asarray([int(page.size) for page in index.pages], dtype=np.int64)
    out = np.zeros((tokens.size, values_np.shape[-1]), dtype=np.float32)
    if starts.size == 0:
        return values_np[tokens].astype(np.float32, copy=False), 0.0, float(tokens.size * values_np.shape[-1] * int(value_bytes)) / MB
    page_ids = np.searchsorted(starts, tokens, side="right") - 1
    valid = (page_ids >= 0) & (page_ids < len(index.pages))
    valid &= tokens < (starts[np.maximum(page_ids, 0)] + sizes[np.maximum(page_ids, 0)])
    fallback = ~valid
    fallback_mb = float(np.sum(fallback) * values_np.shape[-1] * int(value_bytes)) / MB
    if np.any(fallback):
        out[fallback] = values_np[tokens[fallback]].astype(np.float32, copy=False)
    code_bytes = 1 if actual_value_subbits <= 8 else 2
    pages_read = 0
    compressed_count = 0
    subvecs = 0
    for page_id in np.unique(page_ids[valid]).astype(np.int64, copy=False).tolist():
        positions = np.nonzero(valid & (page_ids == int(page_id)))[0]
        page = index.pages[int(page_id)]
        rows = (tokens[positions] - int(page.start)).astype(np.int64, copy=False)
        codebook, page_codes = value_sidecars[int(page_id)]
        codes = page_codes[rows].astype(np.int64, copy=False)
        subvecs = int(codes.shape[1]) if codes.ndim == 2 else 0
        subdim = int(codebook.shape[-1]) if codebook.ndim == 3 else 0
        approx_values = np.zeros((positions.size, subvecs * subdim), dtype=np.float32)
        for sub in range(subvecs):
            approx_values[:, sub * subdim : (sub + 1) * subdim] = codebook[sub, codes[:, sub]]
        out[positions] = approx_values
        pages_read += 1
        compressed_count += int(positions.size)
    codebook_bytes = pages_read * subvecs * (1 << actual_value_subbits) * (
        values_np.shape[-1] // max(1, subvecs)
    ) * int(value_bytes)
    code_bytes_total = compressed_count * subvecs * code_bytes
    return out, float(codebook_bytes + code_bytes_total) / MB, fallback_mb


def _vpq_residual_norms_for_index(
    *,
    index,
    values_np: np.ndarray,
    subbits: int,
    value_subvecs: int,
    value_subbits: int,
) -> np.ndarray:
    """Residual-norm sidecar used by selected-V risk rules.

    This models a deployable page-seal sidecar: each token stores a small norm
    of (exact V - VPQ-reconstructed V). Query time reads only that scalar to
    decide which selected V vectors must remain exact.
    """

    actual_value_subbits = int(value_subbits) if int(value_subbits) > 0 else int(subbits)
    cache_key = (int(subbits), int(value_subvecs), actual_value_subbits)
    cached_by_key = getattr(index, "_value_vpq_residual_norms_by_params", None)
    if isinstance(cached_by_key, dict) and cache_key in cached_by_key:
        return cached_by_key[cache_key]

    norms = np.zeros((int(values_np.shape[0]),), dtype=np.float32)
    value_sidecars = _build_value_vpq_sidecars(
        index,
        values_np,
        int(subbits),
        value_subvecs=int(value_subvecs),
        value_subbits=actual_value_subbits,
    )
    for page_id, page in enumerate(index.pages):
        start = int(page.start)
        size = int(page.size)
        if size <= 0:
            continue
        codebook, page_codes = value_sidecars[int(page_id)]
        if codebook.size == 0 or page_codes.size == 0:
            continue
        codes = page_codes.astype(np.int64, copy=False)
        subvecs = int(codes.shape[1])
        subdim = int(codebook.shape[-1])
        approx_values = np.zeros((size, subvecs * subdim), dtype=np.float32)
        for sub in range(subvecs):
            approx_values[:, sub * subdim : (sub + 1) * subdim] = codebook[sub, codes[:, sub]]
        exact_values = values_np[start : start + size].astype(np.float32, copy=False)
        norms[start : start + size] = (
            np.linalg.norm(exact_values - approx_values, axis=1).astype(np.float32, copy=False)
            / float(np.sqrt(float(values_np.shape[-1])))
        )

    if not isinstance(cached_by_key, dict):
        cached_by_key = {}
    cached_by_key[cache_key] = norms
    setattr(index, "_value_vpq_residual_norms_by_params", cached_by_key)
    return norms


def _selected_value_exact_mask(
    *,
    selected_arr: np.ndarray,
    selected_scores: np.ndarray,
    rule: str,
    fixed_top: int,
    mass_target: float,
    min_top: int,
    max_top: int,
) -> tuple[np.ndarray, int, float]:
    """Choose selected tokens whose V should remain exact.

    The adaptive rule uses only exact logits for already selected tokens. Those
    K vectors have already been read for sparse attention, so this is an online
    deployment signal rather than an oracle over unselected/dense attention.
    """

    count = int(selected_arr.size)
    exact_mask = np.zeros((count,), dtype=bool)
    if count <= 0:
        return exact_mask, 0, 0.0

    selector_rank_order = str(rule) == "selector_rank"
    order = (
        np.arange(count, dtype=np.int64)
        if selector_rank_order
        else np.argsort(-selected_scores.astype(np.float64, copy=False), kind="stable")
    )
    shifted = selected_scores.astype(np.float64, copy=False) - float(np.max(selected_scores))
    probs = np.exp(shifted)
    probs /= max(float(probs.sum()), 1e-20)
    cumulative = np.cumsum(probs[order])
    if str(rule) in {"fixed", "selector_rank"}:
        exact_count = int(fixed_top)
        exact_count_for_mass = max(0, min(count, exact_count))
        achieved_mass = float(cumulative[exact_count_for_mass - 1]) if exact_count_for_mass > 0 else 0.0
    elif str(rule) == "selected_mass":
        target = float(max(0.0, min(1.0, mass_target)))
        exact_count = int(np.searchsorted(cumulative, target, side="left") + 1) if target > 0.0 else 0
        exact_count = min(count, exact_count)
        achieved_mass = float(cumulative[exact_count - 1]) if exact_count > 0 else 0.0
    else:
        raise ValueError(f"unknown selected_value_exact_rule: {rule}")

    exact_count = max(int(min_top), int(exact_count))
    if int(max_top) > 0:
        exact_count = min(int(max_top), int(exact_count))
    exact_count = max(0, min(count, int(exact_count)))
    if exact_count > 0:
        exact_mask[order[:exact_count]] = True
    return exact_mask, exact_count, achieved_mass


def _modeled_online_update_mb(
    *,
    index,
    online_start: int,
    head_dim: int,
    key_bytes: int,
    value_bytes: int,
    subbits: int,
    value_subvecs: int,
    value_subbits: int,
    tail_mode: str,
    selected_value_mode: str,
    selected_value_exact_rule: str,
    selected_value_residual_norm_bytes: int,
) -> float:
    """Estimate append-time sidecar traffic for pages sealed after prefill."""

    code_bytes = 1 if int(subbits) <= 8 else 2
    key_update_bytes = 0.0
    tail_update_bytes = 0.0
    for page in index.pages:
        if int(page.start) < int(online_start):
            continue
        size = int(page.size)
        subvecs = int(page.codes.shape[1])
        subdim = int(head_dim) // max(1, subvecs)
        codebook_entries = subvecs * (1 << int(subbits)) * subdim
        key_update_bytes += float(size * int(head_dim) * int(key_bytes))
        key_update_bytes += float(codebook_entries * int(key_bytes))
        key_update_bytes += float(size * subvecs * code_bytes)
        if str(tail_mode) == "vpq_value":
            vpq_subvecs = int(value_subvecs) if int(value_subvecs) > 0 else subvecs
            vpq_subbits = int(value_subbits) if int(value_subbits) > 0 else int(subbits)
            vpq_code_bytes = 1 if vpq_subbits <= 8 else 2
            vpq_subdim = int(head_dim) // max(1, vpq_subvecs)
            vpq_codebook_entries = vpq_subvecs * (1 << vpq_subbits) * vpq_subdim
            tail_update_bytes += float(size * int(head_dim) * int(value_bytes))
            tail_update_bytes += float(vpq_codebook_entries * int(value_bytes))
            tail_update_bytes += float(size * vpq_subvecs * vpq_code_bytes)
        elif str(tail_mode) == "pq_value":
            tail_update_bytes += float(size * int(head_dim) * int(value_bytes))
            tail_update_bytes += float(size * subvecs * code_bytes)
            tail_update_bytes += float(codebook_entries * int(value_bytes))
        elif str(tail_mode) == "page_mean":
            tail_update_bytes += float(size * int(head_dim) * int(value_bytes))
            tail_update_bytes += float(int(head_dim) * int(value_bytes))
        if str(selected_value_mode) == "vpq_value" and str(selected_value_exact_rule) in {
            "selected_risk_mass",
            "selected_mass_or_risk",
        }:
            tail_update_bytes += float(size * max(0, int(selected_value_residual_norm_bytes)))
    return float(key_update_bytes + tail_update_bytes) / MB


def _selected_exact_marginal_metrics(
    *,
    selected_cpu: np.ndarray,
    previous_selected_cpu: np.ndarray,
    scores_np: np.ndarray,
) -> tuple[float, float]:
    if selected_cpu.size == 0:
        return 1.0, 0.0
    prev = set(int(tok) for tok in previous_selected_cpu.tolist())
    new_mask = np.asarray([int(tok) not in prev for tok in selected_cpu.tolist()], dtype=bool)
    selected_scores = scores_np[selected_cpu].astype(np.float64, copy=False)
    max_score = float(np.max(selected_scores))
    weights = np.exp(selected_scores - max_score)
    weights /= max(float(weights.sum()), 1e-20)
    marginal_mass = float(weights[new_mask].sum()) if bool(np.any(new_mask)) else 0.0
    if bool(np.any(new_mask)):
        marginal_gap = float(np.max(selected_scores[new_mask]) - max_score)
    else:
        marginal_gap = -float("inf")
    return marginal_mass, marginal_gap


def _optimal_probe_blend(base_np: np.ndarray, tail_np: np.ndarray, probe_np: np.ndarray) -> float:
    """Least-squares blend toward a paid exact probe, clipped to [0, 1]."""

    direction = tail_np.astype(np.float64, copy=False) - base_np.astype(np.float64, copy=False)
    denom = float(np.dot(direction, direction))
    if denom <= 1e-20 or not np.isfinite(denom):
        return 0.0
    target = probe_np.astype(np.float64, copy=False) - base_np.astype(np.float64, copy=False)
    blend = float(np.dot(target, direction) / denom)
    if not np.isfinite(blend):
        return 0.0
    return float(max(0.0, min(1.0, blend)))


def _blend_from_probe_rule(
    args: argparse.Namespace,
    base_np: np.ndarray,
    tail_np: np.ndarray,
    probe_np: np.ndarray,
    *,
    selected_proxy_mass: float | None = None,
    probe_proxy_mass: float | None = None,
) -> float:
    fixed = float(max(0.0, min(1.0, float(args.tail_blend))))
    if str(args.tail_blend_rule) != "probe_optimal":
        if str(args.tail_blend_rule) != "probe_extrapolated":
            return fixed
    opt = _optimal_probe_blend(base_np, tail_np, probe_np)
    if str(args.tail_blend_rule) == "probe_optimal":
        return fixed * opt
    if selected_proxy_mass is None or probe_proxy_mass is None:
        return fixed * opt
    selected_proxy_mass = float(max(0.0, min(1.0, selected_proxy_mass)))
    probe_proxy_mass = float(max(selected_proxy_mass, min(1.0, probe_proxy_mass)))
    probed_tail_mass = max(probe_proxy_mass - selected_proxy_mass, 1e-6)
    total_tail_mass = max(1.0 - selected_proxy_mass, 1e-6)
    scale = min(float(args.tail_blend_extrap_max), total_tail_mass / probed_tail_mass)
    return float(max(0.0, min(fixed, opt * scale)))


def _fit_selected_pq_logit_calibration(
    *,
    selected_cpu: np.ndarray,
    ranked_cpu: np.ndarray,
    ranked_scores_cpu: np.ndarray,
    scores_np: np.ndarray,
    query_dim: int,
) -> tuple[float, float, int, float, float]:
    """Fit exact_logit ~= scale * pq_logit + bias on already fetched tokens.

    This is deployable because it only uses selected tokens whose exact K has
    already been read for exact attention. The fitted scale is then applied to
    unselected PQ logits before using them as a tail-distribution proxy.
    """

    if selected_cpu.size == 0 or ranked_cpu.size == 0:
        return 1.0, 0.0, 0, float("inf"), 0.0
    selected_set = set(int(tok) for tok in selected_cpu.tolist())
    pq_logits = []
    exact_logits = []
    scale = float(np.sqrt(float(query_dim)))
    for tok, score in zip(ranked_cpu.tolist(), ranked_scores_cpu.tolist(), strict=False):
        tok_i = int(tok)
        if tok_i not in selected_set:
            continue
        pq_logits.append(float(score) / scale)
        exact_logits.append(float(scores_np[tok_i]))
    if len(pq_logits) < 2:
        return 1.0, 0.0, len(pq_logits), float("inf"), 0.0
    x = np.asarray(pq_logits, dtype=np.float64)
    y = np.asarray(exact_logits, dtype=np.float64)
    x_var = float(np.var(x))
    if x_var <= 1e-20:
        pred = np.full_like(y, float(np.mean(y)))
        rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
        rel_rmse = rmse / max(float(np.std(y)), 1e-6)
        return 0.0, float(np.mean(y)), int(x.size), rel_rmse, 0.0
    cov = float(np.mean((x - float(np.mean(x))) * (y - float(np.mean(y)))))
    fitted_scale = cov / x_var
    fitted_bias = float(np.mean(y)) - fitted_scale * float(np.mean(x))
    if fitted_scale <= 0.0 or not np.isfinite(fitted_scale):
        fitted_scale = 1.0
        fitted_bias = 0.0
    pred = fitted_scale * x + fitted_bias
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    rel_rmse = rmse / max(float(np.std(y)), 1e-6)
    corr = float(np.corrcoef(x, y)[0, 1]) if x.size >= 2 and float(np.std(x)) > 1e-20 and float(np.std(y)) > 1e-20 else 0.0
    if not np.isfinite(corr):
        corr = 0.0
    return float(fitted_scale), float(fitted_bias), int(x.size), float(rel_rmse), corr


def _fit_selected_pq_logit_uncertainty(
    *,
    selected_cpu: np.ndarray,
    ranked_cpu: np.ndarray,
    ranked_scores_cpu: np.ndarray,
    scores_np: np.ndarray,
    query_dim: int,
) -> tuple[float, float, int, float, float, float]:
    tail_score_scale, tail_score_bias, n, rel_rmse, corr = _fit_selected_pq_logit_calibration(
        selected_cpu=selected_cpu,
        ranked_cpu=ranked_cpu,
        ranked_scores_cpu=ranked_scores_cpu,
        scores_np=scores_np,
        query_dim=query_dim,
    )
    if n < 2:
        return tail_score_scale, tail_score_bias, n, rel_rmse, corr, 0.0
    selected_set = set(int(tok) for tok in selected_cpu.tolist())
    pq_logits = []
    exact_logits = []
    score_scale = float(np.sqrt(float(query_dim)))
    for tok, score in zip(ranked_cpu.tolist(), ranked_scores_cpu.tolist(), strict=False):
        tok_i = int(tok)
        if tok_i not in selected_set:
            continue
        pq_logits.append(float(score) / score_scale)
        exact_logits.append(float(scores_np[tok_i]))
    if len(pq_logits) < 2:
        return tail_score_scale, tail_score_bias, n, rel_rmse, corr, 0.0
    x = np.asarray(pq_logits, dtype=np.float64)
    y = np.asarray(exact_logits, dtype=np.float64)
    residual = y - (float(tail_score_scale) * x + float(tail_score_bias))
    residual_std = float(np.std(residual))
    if not np.isfinite(residual_std):
        residual_std = 0.0
    return tail_score_scale, tail_score_bias, n, rel_rmse, corr, residual_std


def _proxy_distribution_stats(
    *,
    selected_cpu: np.ndarray,
    ranked_cpu: np.ndarray,
    ranked_scores_cpu: np.ndarray,
    scores_np: np.ndarray,
    query_dim: int,
    tail_score_scale: float,
    tail_score_bias: float,
    tail_logit_bonus: float,
) -> dict[str, float]:
    selected_set = set(int(tok) for tok in selected_cpu.tolist())
    exact_scores = scores_np[selected_cpu].astype(np.float64, copy=False) if selected_cpu.size else np.asarray([], dtype=np.float64)
    score_scale = float(np.sqrt(float(query_dim)))
    tail_scores = np.asarray(
        [
            float(tail_score_scale) * (float(score) / score_scale) + float(tail_score_bias) + float(tail_logit_bonus)
            for tok, score in zip(ranked_cpu.tolist(), ranked_scores_cpu.tolist(), strict=False)
            if int(tok) not in selected_set
        ],
        dtype=np.float64,
    )
    if exact_scores.size == 0 and tail_scores.size == 0:
        return {
            "selected_mass": 0.0,
            "tail_mass": 0.0,
            "entropy": 0.0,
            "effective_support": 0.0,
            "tail_entropy": 0.0,
            "tail_effective_support": 0.0,
        }
    max_score = max(
        float(np.max(exact_scores)) if exact_scores.size else -np.inf,
        float(np.max(tail_scores)) if tail_scores.size else -np.inf,
    )
    exact_w = np.exp(exact_scores - max_score) if exact_scores.size else np.asarray([], dtype=np.float64)
    tail_w = np.exp(tail_scores - max_score) if tail_scores.size else np.asarray([], dtype=np.float64)
    total = max(float(exact_w.sum()) + float(tail_w.sum()), 1e-20)
    probs = np.concatenate([exact_w, tail_w]) / total
    entropy = float(-np.sum(probs * np.log(np.maximum(probs, 1e-30)))) if probs.size else 0.0
    tail_entropy = 0.0
    tail_eff = 0.0
    if tail_w.size and float(tail_w.sum()) > 0.0:
        tail_probs = tail_w / max(float(tail_w.sum()), 1e-20)
        tail_entropy = float(-np.sum(tail_probs * np.log(np.maximum(tail_probs, 1e-30))))
        tail_eff = float(np.exp(tail_entropy))
    return {
        "selected_mass": float(exact_w.sum()) / total,
        "tail_mass": float(tail_w.sum()) / total,
        "entropy": entropy,
        "effective_support": float(np.exp(entropy)),
        "tail_entropy": tail_entropy,
        "tail_effective_support": tail_eff,
    }


def _round_budget_up(value: float, *, granularity: int, max_budget: int) -> int:
    gran = max(1, int(granularity))
    rounded = int(math.ceil(float(value) / float(gran)) * gran)
    return min(max(0, rounded), max(0, int(max_budget)))


def _sample_tail_audit(
    *,
    selected_cpu: np.ndarray,
    ranked_cpu: np.ndarray | None = None,
    scores_np: np.ndarray,
    context_len: int,
    samples: int,
    seed: int,
    qidx: int,
    head: int,
    mode: str = "uniform",
    bands: int = 8,
) -> tuple[float, float, int, int]:
    selected_set = set(int(tok) for tok in selected_cpu.tolist())
    mode = str(mode)
    if ranked_cpu is not None and mode in {"rank_prefix", "rank_stratified"}:
        ranked_tail = [int(tok) for tok in ranked_cpu.tolist() if int(tok) not in selected_set and int(tok) < int(context_len)]
        ranked_seen = set(ranked_tail)
        # Include static/pending tokens that may not be present in ranked_cpu so
        # the population denominator remains honest.
        tail_list = ranked_tail + [
            int(tok) for tok in range(int(context_len)) if int(tok) not in selected_set and int(tok) not in ranked_seen
        ]
        tail = np.asarray(tail_list, dtype=np.int64)
    else:
        tail = np.asarray([tok for tok in range(int(context_len)) if int(tok) not in selected_set], dtype=np.int64)
    if selected_cpu.size == 0 or tail.size == 0 or int(samples) <= 0:
        return 0.0, -float("inf"), 0, int(tail.size)
    count = min(int(samples), int(tail.size))
    rng = np.random.default_rng(int(seed) + 1000003 * int(qidx) + 65537 * int(head))
    if mode == "rank_prefix":
        sample = tail[:count]
    elif mode == "rank_stratified":
        band_count = max(1, int(bands))
        chunks = np.array_split(tail, band_count)
        per_band = max(1, int(math.ceil(float(count) / float(band_count))))
        picked: list[np.ndarray] = []
        for chunk in chunks:
            if chunk.size == 0:
                continue
            take = min(per_band, int(chunk.size))
            if take >= int(chunk.size):
                picked.append(chunk)
            else:
                # Systematic positions are deterministic and avoid adding a
                # noisy seed dependency to the confidence rule.
                idx = np.linspace(0, int(chunk.size) - 1, num=take, dtype=np.int64)
                picked.append(chunk[idx])
        sample = np.concatenate(picked)[:count] if picked else tail[:0]
    elif mode == "uniform":
        sample = rng.choice(tail, size=count, replace=False) if count < int(tail.size) else tail
    else:
        raise ValueError(f"unknown audit_tail_mode: {mode}")
    selected_scores = scores_np[selected_cpu].astype(np.float64, copy=False)
    sample_scores = scores_np[sample].astype(np.float64, copy=False)
    max_score = max(float(np.max(selected_scores)), float(np.max(sample_scores)))
    selected_w = float(np.exp(selected_scores - max_score).sum())
    sample_w_mean = float(np.exp(sample_scores - max_score).mean()) if sample_scores.size else 0.0
    tail_w_hat = sample_w_mean * float(tail.size)
    tail_mass_hat = tail_w_hat / max(selected_w + tail_w_hat, 1e-20)
    max_gap = float(np.max(sample_scores) - np.max(selected_scores)) if sample_scores.size else -float("inf")
    return float(tail_mass_hat), max_gap, int(count), int(tail.size)


def _sparq_audit_candidates(
    *,
    keys_np: np.ndarray,
    query_np: np.ndarray,
    base: list[int],
    context_len: int,
    rank: int,
    candidates: int,
    key_bytes: int,
    index_bytes: int,
) -> tuple[list[int], float, float, int]:
    """Return a deployable SparQ-style global audit candidate list.

    This scans only the largest-magnitude query channels over all non-base
    tokens. It deliberately does not inspect oracle probabilities or dense
    rankings; the only exact K traffic charged is the queried channels.
    """

    rank = min(max(0, int(rank)), int(query_np.shape[0]))
    candidates = max(0, int(candidates))
    if rank <= 0 or candidates <= 0 or context_len <= 0:
        return [], 0.0, 0.0, 0
    base_set = set(int(tok) for tok in base)
    dynamic = np.asarray([tok for tok in range(int(context_len)) if int(tok) not in base_set], dtype=np.int64)
    if dynamic.size == 0:
        return [], 0.0, 0.0, 0
    dims = np.argsort(-np.abs(query_np), kind="stable")[:rank]
    q_abs_sum = max(float(np.abs(query_np).sum()), 1e-20)
    coverage = max(float(np.abs(query_np[dims]).sum() / q_abs_sum), 1e-6)
    scale = 1.0 / np.sqrt(float(query_np.shape[0]) * coverage)
    partial_keys = keys_np[dynamic[:, None], dims[None, :]].astype(np.float32, copy=False)
    approx = (partial_keys @ query_np[dims].astype(np.float32, copy=False)).astype(np.float32, copy=False) * scale
    take = min(candidates, int(dynamic.size))
    if take < int(dynamic.size):
        top = np.argpartition(-approx, take - 1)[:take]
        top = top[np.argsort(-approx[top], kind="stable")]
    else:
        top = np.argsort(-approx, kind="stable")
    tokens = dynamic[top].astype(np.int64, copy=False).tolist()
    bytes_read = int(rank) * int(index_bytes) + int(dynamic.size) * int(rank) * int(key_bytes)
    return [int(tok) for tok in tokens], float(bytes_read) / MB, float(coverage), int(dynamic.size)


def _sparq_rerank_prefix(
    *,
    keys_np: np.ndarray,
    query_np: np.ndarray,
    ranked_cpu: np.ndarray,
    ranked_scores_cpu: np.ndarray,
    rank: int,
    candidates: int,
    key_bytes: int,
    index_bytes: int,
) -> tuple[np.ndarray, np.ndarray, float, float, int]:
    """Rerank the top PQ shortlist using SparQ-style partial K channels."""

    rank = min(max(0, int(rank)), int(query_np.shape[0]))
    count = min(max(0, int(candidates)), int(ranked_cpu.size))
    if rank <= 0 or count <= 1:
        return ranked_cpu, ranked_scores_cpu, 0.0, 0.0, 0
    dims = np.argsort(-np.abs(query_np), kind="stable")[:rank]
    q_abs_sum = max(float(np.abs(query_np).sum()), 1e-20)
    coverage = max(float(np.abs(query_np[dims]).sum() / q_abs_sum), 1e-6)
    scale = 1.0 / np.sqrt(float(query_np.shape[0]) * coverage)
    top_tokens = ranked_cpu[:count].astype(np.int64, copy=False)
    top_scores = ranked_scores_cpu[:count].astype(np.float32, copy=False)
    partial_keys = keys_np[top_tokens[:, None], dims[None, :]].astype(np.float32, copy=False)
    approx = (partial_keys @ query_np[dims].astype(np.float32, copy=False)).astype(np.float32, copy=False) * scale
    order = np.argsort(-approx, kind="stable")
    reranked_tokens = top_tokens[order].astype(np.int64, copy=False)
    reranked_scores = top_scores[order].astype(np.float32, copy=False)
    if count < int(ranked_cpu.size):
        reranked_tokens = np.concatenate([reranked_tokens, ranked_cpu[count:]])
        reranked_scores = np.concatenate([reranked_scores, ranked_scores_cpu[count:]])
    bytes_read = int(rank) * int(index_bytes) + int(count) * int(rank) * int(key_bytes)
    return reranked_tokens, reranked_scores, float(bytes_read) / MB, float(coverage), int(count)


def run() -> None:
    parser = argparse.ArgumentParser(description="Layer-level quality eval for routed/fullscan paged-PQ selector.")
    parser.add_argument("--qkv_trace", required=True)
    parser.add_argument("--x_trace", required=True)
    parser.add_argument("--model_snapshot", default=".hf_cache/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--decode_lengths", default="500,8000,32000,128000")
    parser.add_argument("--max_qidx_per_decode", type=int, default=1)
    parser.add_argument(
        "--selector_mode",
        choices=["fullscan", "routed", "sparq", "quest", "quest_pq", "oracle"],
        default="routed",
    )
    parser.add_argument("--selector_sparq_rank", type=int, default=16)
    parser.add_argument("--quest_rank", type=int, default=16)
    parser.add_argument("--selector_index_bytes", type=int, default=4)
    parser.add_argument("--budgets", default="4096")
    parser.add_argument("--budget_by_head", default="", help="Optional comma map like 0:16384,1:24576 overriding --budgets per head.")
    parser.add_argument(
        "--online_confidence_rule",
        choices=[
            "none",
            "proxy_mass_exact",
            "proxy_tail_delta",
            "proxy_mass_marginal_exact",
            "pq_proxy_mass_budget",
            "pq_ranked_mass_budget",
            "probe_tail_switch",
            "entropy_probe_tail_switch",
            "adaptive_entropy_probe_tail_switch",
            "geometric_probe_tail_switch",
            "geometric_stable_tail_switch",
            "geometric_slope_stability",
            "geometric_exact_delta",
        ],
        default="none",
    )
    parser.add_argument("--confidence_budgets", default="", help="Candidate exact budgets for online confidence rules.")
    parser.add_argument("--proxy_mass_target", type=float, default=0.98)
    parser.add_argument("--tail_confidence_budget", type=int, default=16384)
    parser.add_argument("--tail_delta_min", type=float, default=0.0)
    parser.add_argument("--tail_delta_max", type=float, default=1.0)
    parser.add_argument("--tail_proxy_mass_min", type=float, default=0.0)
    parser.add_argument("--tail_proxy_mass_max", type=float, default=1.0)
    parser.add_argument("--tail_score_calibration", choices=["none", "affine_selected"], default="none")
    parser.add_argument("--tail_pq_corr_min", type=float, default=-1.0)
    parser.add_argument("--tail_pq_relrmse_max", type=float, default=float("inf"))
    parser.add_argument("--marginal_mass_max", type=float, default=0.01)
    parser.add_argument("--marginal_score_gap_max", type=float, default=float("inf"))
    parser.add_argument("--marginal_min_budget", type=int, default=0)
    parser.add_argument("--tail_probe_budget", type=int, default=20480)
    parser.add_argument("--tail_probe_rel_l2_max", type=float, default=0.02)
    parser.add_argument("--entropy_ucb_z", type=float, default=1.0)
    parser.add_argument("--entropy_budget_scale", type=float, default=1.0)
    parser.add_argument("--entropy_probe_scale", type=float, default=1.125)
    parser.add_argument("--entropy_budget_growth", type=float, default=1.5)
    parser.add_argument("--entropy_min_budget", type=int, default=4096)
    parser.add_argument("--entropy_max_budget", type=int, default=32768)
    parser.add_argument("--entropy_budget_granularity", type=int, default=1024)
    parser.add_argument("--entropy_tail_mass_max", type=float, default=0.01)
    parser.add_argument("--geometric_min_budget", type=int, default=4096)
    parser.add_argument("--geometric_max_budget", type=int, default=32768)
    parser.add_argument(
        "--geometric_max_budget_by_head",
        default="",
        help="Optional comma map like 0:120000,1:120000 overriding --geometric_max_budget per query head.",
    )
    parser.add_argument(
        "--long_context_threshold",
        type=int,
        default=0,
        help="If >0, enables long-context overrides when decode_length is at or above this threshold.",
    )
    parser.add_argument(
        "--long_geometric_max_budget",
        type=int,
        default=0,
        help="If >0 with long_context_threshold, overrides geometric_max_budget at/above threshold.",
    )
    parser.add_argument(
        "--long_geometric_max_budget_by_head",
        default="",
        help="Optional long-context comma map like 0:120000,1:120000 overriding geometric max budget per head.",
    )
    parser.add_argument("--geometric_growth", type=float, default=1.5)
    parser.add_argument("--geometric_probe_scale", type=float, default=1.125)
    parser.add_argument("--geometric_budget_granularity", type=int, default=1024)
    parser.add_argument("--stable_tail_probe_rel_l2_max", type=float, default=0.05)
    parser.add_argument("--slope_forward_rel_l2_max", type=float, default=0.05)
    parser.add_argument("--slope_backward_rel_l2_max", type=float, default=0.10)
    parser.add_argument("--slope_ratio_max", type=float, default=1.0)
    parser.add_argument("--slope_curvature_rel_l2_max", type=float, default=0.05)
    parser.add_argument("--exact_delta_rel_l2_max", type=float, default=0.01)
    parser.add_argument("--audit_tail_samples", type=int, default=0)
    parser.add_argument("--audit_tail_mode", choices=["uniform", "rank_prefix", "rank_stratified"], default="uniform")
    parser.add_argument("--audit_tail_bands", type=int, default=8)
    parser.add_argument("--audit_tail_mass_max", type=float, default=1.0)
    parser.add_argument("--audit_tail_logit_gap_max", type=float, default=float("inf"))
    parser.add_argument("--sparq_audit_rank", type=int, default=0)
    parser.add_argument("--sparq_audit_candidates", type=int, default=0)
    parser.add_argument("--sparq_audit_index_bytes", type=int, default=4)
    parser.add_argument("--sparq_rerank_rank", type=int, default=0)
    parser.add_argument("--sparq_rerank_candidates", type=int, default=0)
    parser.add_argument("--sparq_rerank_index_bytes", type=int, default=4)
    parser.add_argument("--rerank_candidates", type=int, default=0)
    parser.add_argument("--tail_samples", type=int, default=12288)
    parser.add_argument("--tail_bands", type=int, default=8)
    parser.add_argument("--tail_seed", type=int, default=0)
    parser.add_argument("--tail_sampling", choices=["random", "linspace", "systematic"], default="systematic")
    parser.add_argument("--tail_mode", choices=["sample", "pq_value", "vpq_value", "page_mean"], default="sample")
    parser.add_argument("--selected_value_mode", choices=["exact", "vpq_value"], default="exact")
    parser.add_argument(
        "--selected_value_exact_rule",
        choices=["fixed", "selector_rank", "selected_mass", "selected_risk_mass", "selected_mass_or_risk"],
        default="fixed",
    )
    parser.add_argument("--selected_value_exact_top", type=int, default=0)
    parser.add_argument("--selected_value_exact_mass", type=float, default=0.0)
    parser.add_argument("--selected_value_exact_risk_mass", type=float, default=0.0)
    parser.add_argument("--selected_value_min_exact_top", type=int, default=0)
    parser.add_argument("--selected_value_max_exact_top", type=int, default=0)
    parser.add_argument(
        "--selected_value_max_exact_top_by_head",
        default="",
        help="Optional comma map like 0:0,1:0 overriding --selected_value_max_exact_top per query head.",
    )
    parser.add_argument(
        "--selected_value_exact_all_context_max",
        type=int,
        default=0,
        help="If >0, keep all selected V exact when context_len is at or below this threshold.",
    )
    parser.add_argument(
        "--selected_value_exact_all_fraction_min",
        type=float,
        default=0.0,
        help=(
            "If >0, keep all selected V exact when selected_tokens/context_len is at least this fraction. "
            "This is a deployable compression-confidence rule; it uses only selected-set size."
        ),
    )
    parser.add_argument(
        "--long_selected_value_exact_mass",
        type=float,
        default=-1.0,
        help="If >=0 with long_context_threshold, overrides selected_value_exact_mass at/above threshold.",
    )
    parser.add_argument(
        "--long_selected_value_max_exact_top",
        type=int,
        default=-1,
        help="If >=0 with long_context_threshold, overrides selected_value_max_exact_top at/above threshold.",
    )
    parser.add_argument(
        "--long_selected_value_max_exact_top_by_head",
        default="",
        help="Optional long-context comma map like 0:0,1:0 overriding selected_value_max_exact_top per head.",
    )
    parser.add_argument("--selected_value_residual_correction", choices=["none", "exact_mean"], default="none")
    parser.add_argument("--selected_value_residual_norm_bytes", type=int, default=2)
    parser.add_argument("--selected_key_mode", choices=["exact", "band_calibrated_selector"], default="exact")
    parser.add_argument("--selected_key_calibration_probes", type=int, default=0)
    parser.add_argument("--selected_key_calibration_bands", type=int, default=8)
    parser.add_argument(
        "--selected_key_exact_selector_mass",
        type=float,
        default=0.0,
        help="If >0 with compressed selected K, keep exact K for top selected tokens by selector-logit softmax mass.",
    )
    parser.add_argument("--selected_key_min_exact_top", type=int, default=0)
    parser.add_argument("--selected_key_max_exact_top", type=int, default=0)
    parser.add_argument(
        "--selected_key_min_context",
        type=int,
        default=0,
        help="If >0, use exact selected K below this context length even when selected_key_mode is compressed.",
    )
    parser.add_argument("--tail_blend", type=float, default=1.0)
    parser.add_argument("--tail_blend_rule", choices=["fixed", "probe_optimal", "probe_extrapolated"], default="fixed")
    parser.add_argument("--tail_blend_extrap_max", type=float, default=4.0)
    parser.add_argument("--tail_off_heads", default="", help="Comma-separated heads that use selected-only exact attention.")
    parser.add_argument("--static_prefix", type=int, default=128)
    parser.add_argument("--static_suffix", type=int, default=128)
    parser.add_argument("--page_size", type=int, default=5632)
    parser.add_argument("--subvecs", type=int, default=4)
    parser.add_argument("--subbits", type=int, default=6)
    parser.add_argument("--value_subvecs", type=int, default=0)
    parser.add_argument("--value_subbits", type=int, default=0)
    parser.add_argument("--kmeans_iters", type=int, default=3)
    parser.add_argument("--key_bytes", type=int, default=2)
    parser.add_argument("--value_bytes", type=int, default=2)
    parser.add_argument("--nprobes", default="16,32,64,128,256,512")
    parser.add_argument("--router_prototypes", type=int, default=16)
    parser.add_argument("--router_merge_rel", type=float, default=0.05)
    parser.add_argument("--router_merge_var", type=float, default=0.0)
    parser.add_argument("--router_max_groups", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--head_only",
        action="store_true",
        help="Skip o_proj/residual/MLP metrics and write only per-head attention diagnostics.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise RuntimeError("CUDA requested but not available")
    device = torch.device(args.device)
    torch.set_grad_enabled(False)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True), encoding="utf-8")

    trace = load_trace(args.qkv_trace)
    x_data = np.load(args.x_trace, mmap_mode="r")
    x_meta = json.loads(str(x_data["metadata"].item()))
    layer_inputs = x_data["layer_inputs"]
    layer_idx = int(x_meta["layer_idx"])
    norm_eps = float(x_meta.get("norm_eps", 1e-5))
    if int(trace.head_dim) * int(trace.num_heads) != int(x_meta["hidden_size"]):
        raise ValueError("trace head dimensions do not match x_trace hidden_size")

    model_dir = PROJECT_ROOT / args.model_snapshot
    weight_map = load_weight_index(model_dir)
    prefix = f"model.layers.{layer_idx}"
    wo = load_safetensor_weight(model_dir, weight_map, f"{prefix}.self_attn.o_proj.weight", device)
    post_ln = load_safetensor_weight(model_dir, weight_map, f"{prefix}.post_attention_layernorm.weight", device)
    gate_proj = load_safetensor_weight(model_dir, weight_map, f"{prefix}.mlp.gate_proj.weight", device)
    up_proj = load_safetensor_weight(model_dir, weight_map, f"{prefix}.mlp.up_proj.weight", device)
    down_proj = load_safetensor_weight(model_dir, weight_map, f"{prefix}.mlp.down_proj.weight", device)

    q_indices = trace.q_indices_for_decodes(parse_csv_ints(args.decode_lengths))
    if int(args.max_qidx_per_decode) > 0:
        limited = []
        counts: dict[int, int] = {}
        for qidx in q_indices:
            decode = trace.decode_tokens_for_qidx(int(qidx))
            seen = counts.get(int(decode), 0)
            if seen >= int(args.max_qidx_per_decode):
                continue
            limited.append(int(qidx))
            counts[int(decode)] = seen + 1
        q_indices = limited

    budgets = parse_csv_ints(args.budgets)
    if len(budgets) != 1:
        raise ValueError("layer eval expects exactly one budget for now")
    default_budget = int(budgets[0])
    budget_by_head = parse_head_budget_map(args.budget_by_head)
    geometric_max_budget_by_head = parse_head_budget_map(args.geometric_max_budget_by_head)
    long_geometric_max_budget_by_head = parse_head_budget_map(args.long_geometric_max_budget_by_head)
    selected_value_max_exact_top_by_head = parse_head_budget_map(args.selected_value_max_exact_top_by_head)
    long_selected_value_max_exact_top_by_head = parse_head_budget_map(args.long_selected_value_max_exact_top_by_head)
    confidence_budgets = (
        sorted(set(parse_csv_ints(args.confidence_budgets)))
        if str(args.confidence_budgets).strip()
        else [default_budget]
    )
    nprobes = parse_csv_ints(args.nprobes)
    tail_off_heads = parse_int_set(args.tail_off_heads)
    kv_fanout = {
        int(kv_head): max(
            1,
            sum(1 for head in range(int(trace.num_heads)) if int(trace.kv_head_for(int(head))) == int(kv_head)),
        )
        for kv_head in range(int(trace.kv_heads))
    }
    rows = []
    per_head_rows = []

    for qidx in q_indices:
        position = int(trace.positions[int(qidx)])
        decode_tokens = int(trace.decode_tokens_for_qidx(int(qidx)))
        context_len = int(position) + 1
        if position >= int(layer_inputs.shape[0]):
            raise ValueError(f"position {position} not present in layer_inputs with shape {layer_inputs.shape}")
        indexed_end = max(
            min(max(0, int(args.static_prefix)), int(trace.input_len)),
            min(context_len - max(0, int(args.static_suffix)), trace.keys.shape[1]),
        )

        index_cache = {}
        online_update_mb_by_kv = {}
        torch_k_cache = {}
        torch_v_cache = {}
        dense_heads = []
        approx_heads = []
        head_rows = []
        q_start = time.perf_counter()
        dynamic_start = min(max(0, int(args.static_prefix)), int(trace.input_len))
        init_dynamic_end = max(dynamic_start, int(trace.input_len) - max(0, int(args.static_suffix)))
        online_start = dynamic_start + (
            (max(0, init_dynamic_end - dynamic_start) // max(1, int(args.page_size))) * max(1, int(args.page_size))
        )

        for kv_head in range(trace.kv_heads):
            keys_np = trace.keys[kv_head, :context_len].astype(np.float32, copy=False)
            index_cache[kv_head] = build_page_pq_gpu(
                keys_np,
                dynamic_start=dynamic_start,
                indexed_end=indexed_end,
                page_size=int(args.page_size),
                subvecs=int(args.subvecs),
                subbits=int(args.subbits),
                kmeans_iters=int(args.kmeans_iters),
                seed=2025 + 2027 * int(kv_head),
                key_bytes=int(args.key_bytes),
                router_enabled=str(args.selector_mode) == "routed",
                router_prototypes=int(args.router_prototypes),
                router_merge_rel=float(args.router_merge_rel),
                router_merge_var=float(args.router_merge_var),
                router_max_groups=int(args.router_max_groups),
                device=device,
            )
            online_update_mb_by_kv[kv_head] = _modeled_online_update_mb(
                index=index_cache[kv_head],
                online_start=online_start,
                head_dim=int(trace.head_dim),
                key_bytes=int(args.key_bytes),
                value_bytes=int(args.value_bytes),
                subbits=int(args.subbits),
                value_subvecs=int(args.value_subvecs),
                value_subbits=int(args.value_subbits),
                tail_mode=str(args.tail_mode),
                selected_value_mode=str(args.selected_value_mode),
                selected_value_exact_rule=str(args.selected_value_exact_rule),
                selected_value_residual_norm_bytes=int(args.selected_value_residual_norm_bytes),
            )
            torch_k_cache[kv_head] = torch.as_tensor(keys_np, dtype=torch.float32, device=device)
            torch_v_cache[kv_head] = torch.as_tensor(
                trace.values[kv_head, :context_len].astype(np.float32, copy=False),
                dtype=torch.float32,
                device=device,
            )

        for head in range(trace.num_heads):
            long_context_active = (
                int(args.long_context_threshold) > 0
                and int(decode_tokens) >= int(args.long_context_threshold)
            )
            effective_selected_value_exact_mass = float(args.selected_value_exact_mass)
            effective_selected_value_max_exact_top = int(
                selected_value_max_exact_top_by_head.get(int(head), int(args.selected_value_max_exact_top))
            )
            effective_geometric_max_budget = int(
                geometric_max_budget_by_head.get(int(head), int(args.geometric_max_budget))
            )
            if long_context_active:
                if float(args.long_selected_value_exact_mass) >= 0.0:
                    effective_selected_value_exact_mass = float(args.long_selected_value_exact_mass)
                if int(args.long_selected_value_max_exact_top) >= 0:
                    effective_selected_value_max_exact_top = int(args.long_selected_value_max_exact_top)
                if int(head) in long_selected_value_max_exact_top_by_head:
                    effective_selected_value_max_exact_top = int(long_selected_value_max_exact_top_by_head[int(head)])
                if int(args.long_geometric_max_budget) > 0:
                    effective_geometric_max_budget = int(args.long_geometric_max_budget)
                if int(head) in long_geometric_max_budget_by_head:
                    effective_geometric_max_budget = int(long_geometric_max_budget_by_head[int(head)])
            if str(args.online_confidence_rule) == "none":
                budget = int(budget_by_head.get(int(head), default_budget))
                rank_budget = budget
            elif str(args.online_confidence_rule) in {"entropy_probe_tail_switch", "adaptive_entropy_probe_tail_switch"}:
                budget = int(max(confidence_budgets + [int(args.tail_confidence_budget), int(args.entropy_max_budget)]))
                rank_budget = budget
            elif str(args.online_confidence_rule) in {
                "geometric_probe_tail_switch",
                "geometric_stable_tail_switch",
                "geometric_slope_stability",
                "geometric_exact_delta",
            }:
                budget = int(effective_geometric_max_budget)
                rank_budget = budget
            else:
                budget = int(max(confidence_budgets + [int(args.tail_confidence_budget)]))
                rank_budget = budget
            kv_head = trace.kv_head_for(int(head))
            keys_np = trace.keys[kv_head, :context_len].astype(np.float32, copy=False)
            values_np = trace.values[kv_head, :context_len].astype(np.float32, copy=False)
            query_np = trace.queries[int(head), int(qidx)].astype(np.float32, copy=False)
            scores_np, probs_np, dense_head = dense_attention_output(keys_np, values_np, query_np)
            dense_heads.append(dense_head)

            index = index_cache[kv_head]
            pending = list(range(max(0, int(index.pending_start)), max(0, min(int(index.indexed_end), context_len))))
            base = unique_tokens(
                static_tokens(position, args.static_prefix, args.static_suffix) + pending,
                context_len=context_len,
            )
            selected_value_cache: dict[bytes, tuple[np.ndarray, np.ndarray | None, float, float, int, float]] = {}

            def selected_output(selected_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
                key = selected_arr.astype(np.int64, copy=False).tobytes()
                cached = selected_value_cache.get(key)
                if cached is not None:
                    return cached[0], cached[1]
                if str(args.selected_value_mode) == "vpq_value":
                    selected_values = np.empty((selected_arr.size, values_np.shape[-1]), dtype=np.float32)
                    exact_value_mb = 0.0
                    compressed_v_mb = 0.0
                    fallback_v_mb = 0.0
                    exact_all_by_context = (
                        int(args.selected_value_exact_all_context_max) > 0
                        and int(context_len) <= int(args.selected_value_exact_all_context_max)
                    )
                    selected_fraction = float(selected_arr.size) / max(1.0, float(context_len))
                    exact_all_by_fraction = (
                        float(args.selected_value_exact_all_fraction_min) > 0.0
                        and selected_fraction >= float(args.selected_value_exact_all_fraction_min)
                    )
                    if exact_all_by_context or exact_all_by_fraction:
                        selected_values[:] = values_np[selected_arr].astype(np.float32, copy=False)
                        exact_count = int(selected_arr.size)
                        exact_selected_mass = 1.0 if exact_count > 0 else 0.0
                        exact_value_mb = float(exact_count * trace.head_dim * int(args.value_bytes)) / MB
                    elif str(args.selected_value_exact_rule) in {"selected_risk_mass", "selected_mass_or_risk"}:
                        residual_norm_all = _vpq_residual_norms_for_index(
                            index=index,
                            values_np=values_np,
                            subbits=int(args.subbits),
                            value_subvecs=int(args.value_subvecs),
                            value_subbits=int(args.value_subbits),
                        )
                        residual_norm_mb = float(
                            selected_arr.size * max(0, int(args.selected_value_residual_norm_bytes))
                        ) / MB
                        compressed_v_mb += residual_norm_mb
                        selected_scores = scores_np[selected_arr].astype(np.float64, copy=False)
                        shifted = selected_scores - float(np.max(selected_scores))
                        probs = np.exp(shifted)
                        probs /= max(float(probs.sum()), 1e-20)
                        residual_norm = residual_norm_all[selected_arr].astype(np.float64, copy=False)
                        risk = probs * residual_norm.astype(np.float64, copy=False)
                        risk_order = np.argsort(-risk, kind="stable")
                        total_risk = float(risk.sum())
                        exact_mask = np.zeros((selected_arr.size,), dtype=bool)
                        risk_target = (
                            float(args.selected_value_exact_risk_mass)
                            if float(args.selected_value_exact_risk_mass) > 0.0
                            else float(effective_selected_value_exact_mass)
                        )
                        if total_risk > 1e-20 and risk_target > 0.0:
                            cumulative = np.cumsum(risk[risk_order]) / total_risk
                            risk_count = int(
                                np.searchsorted(
                                    cumulative,
                                    float(max(0.0, min(1.0, risk_target))),
                                    side="left",
                                )
                                + 1
                            )
                        else:
                            risk_count = int(args.selected_value_exact_top)
                        risk_count = max(0, min(int(selected_arr.size), int(risk_count)))
                        if risk_count > 0:
                            exact_mask[risk_order[:risk_count]] = True
                        if str(args.selected_value_exact_rule) == "selected_mass_or_risk":
                            prob_order = np.argsort(-selected_scores, kind="stable")
                            mass_target = float(max(0.0, min(1.0, effective_selected_value_exact_mass)))
                            if mass_target > 0.0:
                                prob_cumulative = np.cumsum(probs[prob_order])
                                mass_count = int(np.searchsorted(prob_cumulative, mass_target, side="left") + 1)
                            else:
                                mass_count = 0
                            mass_count = max(0, min(int(selected_arr.size), int(mass_count)))
                            if mass_count > 0:
                                exact_mask[prob_order[:mass_count]] = True
                        exact_count = int(np.sum(exact_mask))
                        min_top = int(args.selected_value_min_exact_top)
                        if exact_count < min_top:
                            fill_order = np.argsort(-selected_scores, kind="stable")
                            exact_mask[fill_order[: min(int(selected_arr.size), min_top)]] = True
                            exact_count = int(np.sum(exact_mask))
                        if int(effective_selected_value_max_exact_top) > 0:
                            max_top = int(effective_selected_value_max_exact_top)
                            if exact_count > max_top:
                                keep_order = np.argsort(-(probs * (1.0 + residual_norm)), kind="stable")[:max_top]
                                limited_mask = np.zeros((selected_arr.size,), dtype=bool)
                                limited_mask[keep_order] = True
                                exact_mask = limited_mask
                                exact_count = int(np.sum(exact_mask))
                        exact_selected_mass = float(probs[exact_mask].sum()) if exact_count > 0 else 0.0
                        if exact_count > 0:
                            selected_values[exact_mask] = values_np[selected_arr[exact_mask]].astype(np.float32, copy=False)
                            exact_value_mb = float(np.sum(exact_mask) * trace.head_dim * int(args.value_bytes)) / MB
                        compressed_mask = ~exact_mask
                        if np.any(compressed_mask):
                            approx_values, compressed_values_mb, fallback_values_mb = _vpq_values_for_tokens(
                                index=index,
                                values_np=values_np,
                                tokens=selected_arr[compressed_mask].astype(np.int64, copy=False),
                                subbits=int(args.subbits),
                                value_subvecs=int(args.value_subvecs),
                                value_subbits=int(args.value_subbits),
                                value_bytes=int(args.value_bytes),
                            )
                            compressed_v_mb += float(compressed_values_mb)
                            fallback_v_mb += float(fallback_values_mb)
                            selected_values[compressed_mask] = approx_values
                    else:
                        exact_mask, exact_count, exact_selected_mass = _selected_value_exact_mask(
                            selected_arr=selected_arr,
                            selected_scores=scores_np[selected_arr],
                            rule=str(args.selected_value_exact_rule),
                            fixed_top=int(args.selected_value_exact_top),
                            mass_target=float(effective_selected_value_exact_mass),
                            min_top=int(args.selected_value_min_exact_top),
                            max_top=int(effective_selected_value_max_exact_top),
                        )
                        if exact_count > 0:
                            selected_values[exact_mask] = values_np[selected_arr[exact_mask]].astype(np.float32, copy=False)
                            exact_value_mb = float(np.sum(exact_mask) * trace.head_dim * int(args.value_bytes)) / MB
                        compressed_mask = ~exact_mask
                        if (
                            str(args.selected_value_residual_correction) == "exact_mean"
                            and exact_count > 0
                            and np.any(compressed_mask)
                        ):
                            approx_values_all, compressed_v_mb, fallback_v_mb = _vpq_values_for_tokens(
                                index=index,
                                values_np=values_np,
                                tokens=selected_arr.astype(np.int64, copy=False),
                                subbits=int(args.subbits),
                                value_subvecs=int(args.value_subvecs),
                                value_subbits=int(args.value_subbits),
                                value_bytes=int(args.value_bytes),
                            )
                            exact_scores = scores_np[selected_arr[exact_mask]].astype(np.float64, copy=False)
                            exact_weights = np.exp(exact_scores - float(np.max(exact_scores)))
                            exact_weights /= max(float(exact_weights.sum()), 1e-20)
                            exact_residual = (
                                values_np[selected_arr[exact_mask]].astype(np.float64, copy=False)
                                - approx_values_all[exact_mask].astype(np.float64, copy=False)
                            )
                            residual_bias = (exact_weights @ exact_residual).astype(np.float32, copy=False)
                            selected_values[compressed_mask] = (
                                approx_values_all[compressed_mask].astype(np.float32, copy=False) + residual_bias
                            )
                        elif np.any(compressed_mask):
                            approx_values, compressed_v_mb, fallback_v_mb = _vpq_values_for_tokens(
                                index=index,
                                values_np=values_np,
                                tokens=selected_arr[compressed_mask].astype(np.int64, copy=False),
                                subbits=int(args.subbits),
                                value_subvecs=int(args.value_subvecs),
                                value_subbits=int(args.value_subbits),
                                value_bytes=int(args.value_bytes),
                            )
                            selected_values[compressed_mask] = approx_values
                    out_np = _selected_only_output(
                        keys_np,
                        values_np,
                        query_np,
                        selected_arr,
                        values_override=selected_values,
                    )
                    selected_value_cache[key] = (
                        out_np,
                        selected_values,
                        float(compressed_v_mb),
                        float(fallback_v_mb) + float(exact_value_mb),
                        int(exact_count),
                        float(exact_selected_mass),
                    )
                    return out_np, selected_values
                out_np = _selected_only_output(keys_np, values_np, query_np, selected_arr)
                exact_v_mb = float(selected_arr.size * trace.head_dim * int(args.value_bytes)) / MB
                selected_value_cache[key] = (out_np, None, 0.0, exact_v_mb, int(selected_arr.size), 1.0 if selected_arr.size else 0.0)
                return out_np, None

            query = torch.as_tensor(query_np, dtype=torch.float32, device=device)
            selector_coverage = 0.0
            if str(args.selector_mode) in {"fullscan", "routed"}:
                ranked_t, ranked_scores_t, selector_seconds, selector_mb, chosen_nprobe = rank_paged_pq(
                    query,
                    index,
                    mode=str(args.selector_mode),
                    selector_backend="torch",
                    nprobes=nprobes,
                    budget=rank_budget,
                    key_bytes=int(args.key_bytes),
                    subbits=int(args.subbits),
                )
                ranked_cpu = ranked_t.detach().cpu().numpy().astype(np.int64, copy=False)
                ranked_scores_cpu = ranked_scores_t.detach().cpu().numpy().astype(np.float32, copy=False)
            elif str(args.selector_mode) == "sparq":
                t0 = time.perf_counter()
                ranked_cpu, ranked_scores_cpu, selector_mb, selector_coverage = _sparq_rank_tokens(
                    keys_np=keys_np,
                    query_np=query_np,
                    index=index,
                    rank=int(args.selector_sparq_rank),
                    key_bytes=int(args.key_bytes),
                    index_bytes=int(args.selector_index_bytes),
                )
                selector_seconds = time.perf_counter() - t0
                chosen_nprobe = 0
            elif str(args.selector_mode) == "quest":
                t0 = time.perf_counter()
                ranked_cpu, ranked_scores_cpu, selector_mb, chosen_nprobe, selector_coverage = _rank_quest_pages(
                    keys_np=keys_np,
                    query_np=query_np,
                    index=index,
                    rank=int(args.quest_rank),
                    key_bytes=int(args.key_bytes),
                    index_bytes=int(args.selector_index_bytes),
                )
                selector_seconds = time.perf_counter() - t0
            elif str(args.selector_mode) == "quest_pq":
                t0 = time.perf_counter()
                ranked_cpu, ranked_scores_cpu, selector_mb, chosen_nprobe, selector_coverage = _rank_quest_pq(
                    query=query,
                    keys_np=keys_np,
                    query_np=query_np,
                    index=index,
                    rank=int(args.quest_rank),
                    nprobes=nprobes,
                    budget=rank_budget,
                    key_bytes=int(args.key_bytes),
                    subbits=int(args.subbits),
                    index_bytes=int(args.selector_index_bytes),
                )
                selector_seconds = time.perf_counter() - t0
            elif str(args.selector_mode) == "oracle":
                t0 = time.perf_counter()
                base_set = set(int(tok) for tok in base)
                order = np.argsort(-scores_np, kind="stable")
                ranked_cpu = np.asarray(
                    [int(tok) for tok in order.tolist() if int(tok) not in base_set],
                    dtype=np.int64,
                )
                ranked_scores_cpu = (scores_np[ranked_cpu].astype(np.float32, copy=False) * np.sqrt(float(trace.head_dim))).astype(
                    np.float32,
                    copy=False,
                )
                selector_mb = float(context_len * trace.head_dim * int(args.key_bytes)) / MB
                selector_coverage = 1.0
                chosen_nprobe = 0
                selector_seconds = time.perf_counter() - t0
            else:
                raise ValueError(f"unknown selector_mode: {args.selector_mode}")
            rerank_count = 0
            rerank_key_mb = 0.0
            sparq_rerank_count = 0
            sparq_rerank_mb = 0.0
            sparq_rerank_coverage = 0.0
            if int(args.rerank_candidates) > 0 and ranked_cpu.size:
                rerank_count = min(int(args.rerank_candidates), int(ranked_cpu.size))
                rerank_tokens = ranked_cpu[:rerank_count]
                rerank_pq_scores = ranked_scores_cpu[:rerank_count]
                rerank_scores = scores_np[rerank_tokens]
                rerank_order = np.argsort(-rerank_scores, kind="stable")
                reranked = rerank_tokens[rerank_order].astype(np.int64, copy=False)
                reranked_pq_scores = rerank_pq_scores[rerank_order].astype(np.float32, copy=False)
                reranked_set = set(int(tok) for tok in reranked.tolist())
                rest_pairs = [
                    (int(tok), float(score))
                    for tok, score in zip(ranked_cpu.tolist(), ranked_scores_cpu.tolist(), strict=False)
                    if int(tok) not in reranked_set
                ]
                rest = np.asarray([tok for tok, _score in rest_pairs], dtype=np.int64)
                rest_scores = np.asarray([score for _tok, score in rest_pairs], dtype=np.float32)
                ranked_cpu = np.concatenate([reranked, rest]) if rest.size else reranked
                ranked_scores_cpu = np.concatenate([reranked_pq_scores, rest_scores]) if rest_scores.size else reranked_pq_scores
                rerank_key_mb = float(rerank_count * trace.head_dim * int(args.key_bytes)) / MB
            if int(args.sparq_rerank_rank) > 0 and int(args.sparq_rerank_candidates) > 0 and ranked_cpu.size:
                ranked_cpu, ranked_scores_cpu, sparq_rerank_mb, sparq_rerank_coverage, sparq_rerank_count = _sparq_rerank_prefix(
                    keys_np=keys_np,
                    query_np=query_np,
                    ranked_cpu=ranked_cpu,
                    ranked_scores_cpu=ranked_scores_cpu,
                    rank=int(args.sparq_rerank_rank),
                    candidates=int(args.sparq_rerank_candidates),
                    key_bytes=int(args.key_bytes),
                    index_bytes=int(args.sparq_rerank_index_bytes),
                )
                rerank_key_mb += float(sparq_rerank_mb)
            sparq_audit_tokens: list[int] = []
            sparq_audit_mb = 0.0
            sparq_audit_coverage = 0.0
            sparq_audit_population = 0
            if int(args.sparq_audit_rank) > 0 and int(args.sparq_audit_candidates) > 0:
                sparq_audit_tokens, sparq_audit_mb, sparq_audit_coverage, sparq_audit_population = _sparq_audit_candidates(
                    keys_np=keys_np,
                    query_np=query_np,
                    base=base,
                    context_len=context_len,
                    rank=int(args.sparq_audit_rank),
                    candidates=int(args.sparq_audit_candidates),
                    key_bytes=int(args.key_bytes),
                    index_bytes=int(args.sparq_audit_index_bytes),
                )
                if sparq_audit_tokens:
                    base = unique_tokens(base + sparq_audit_tokens, context_len=context_len)
            online_rule = str(args.online_confidence_rule)
            chosen_proxy_mass = 0.0
            chosen_proxy_tail_mass = 0.0
            tail_delta_ratio = 0.0
            tail_confidence_pass = False
            tail_score_scale = 1.0
            tail_score_bias = 0.0
            tail_calibration_tokens = 0
            tail_pq_relrmse = float("inf")
            tail_pq_corr = 0.0
            tail_pq_residual_std = 0.0
            confidence_mb = float(sparq_audit_mb)
            marginal_exact_mass = 0.0
            marginal_score_gap = -float("inf")
            tail_probe_rel_l2 = float("inf")
            stable_tail_probe_rel_l2 = float("inf")
            slope_forward_rel_l2 = float("inf")
            slope_backward_rel_l2 = float("inf")
            slope_ratio = float("inf")
            slope_curvature_rel_l2 = float("inf")
            slope_minus_budget = 0
            slope_center_budget = 0
            slope_plus_budget = 0
            audit_tail_mass = 0.0
            audit_tail_logit_gap = -float("inf")
            audit_tail_count = 0
            audit_tail_population = 0
            entropy_effective_support = 0.0
            entropy_tail_effective_support = 0.0
            entropy_tail_mass = 0.0
            entropy_required_budget = 0
            exact_delta_rel_l2 = float("inf")
            precomputed_tail_np = None
            precomputed_tail_count = 0
            precomputed_tail_population = 0
            precomputed_tail_mb = 0.0
            tail_estimator_reused = False
            if online_rule == "none":
                selected_cpu = _selected_for_budget(base=base, ranked_cpu=ranked_cpu, budget=budget, context_len=context_len)
                selected_only_np, _selected_values_np = selected_output(selected_cpu)
                if str(args.tail_score_calibration) == "affine_selected":
                    tail_score_scale, tail_score_bias, tail_calibration_tokens, tail_pq_relrmse, tail_pq_corr = (
                        _fit_selected_pq_logit_calibration(
                            selected_cpu=selected_cpu,
                            ranked_cpu=ranked_cpu,
                            ranked_scores_cpu=ranked_scores_cpu,
                            scores_np=scores_np,
                            query_dim=int(trace.head_dim),
                        )
                    )
                chosen_proxy_mass, chosen_proxy_tail_mass = _proxy_selected_mass(
                    selected_cpu=selected_cpu,
                    ranked_cpu=ranked_cpu,
                    ranked_scores_cpu=ranked_scores_cpu,
                    scores_np=scores_np,
                    query_dim=int(trace.head_dim),
                    tail_score_scale=tail_score_scale,
                    tail_score_bias=tail_score_bias,
                )
                effective_tail_blend = 0.0 if int(head) in tail_off_heads else float(args.tail_blend)
            elif online_rule == "proxy_mass_exact":
                selected_cpu = np.empty((0,), dtype=np.int64)
                selected_only_np = np.zeros((trace.head_dim,), dtype=np.float32)
                effective_tail_blend = 0.0
                for candidate_budget in confidence_budgets:
                    candidate_selected = _selected_for_budget(
                        base=base,
                        ranked_cpu=ranked_cpu,
                        budget=int(candidate_budget),
                        context_len=context_len,
                    )
                    if str(args.tail_score_calibration) == "affine_selected":
                        (
                            tail_score_scale,
                            tail_score_bias,
                            tail_calibration_tokens,
                            tail_pq_relrmse,
                            tail_pq_corr,
                        ) = _fit_selected_pq_logit_calibration(
                            selected_cpu=candidate_selected,
                            ranked_cpu=ranked_cpu,
                            ranked_scores_cpu=ranked_scores_cpu,
                            scores_np=scores_np,
                            query_dim=int(trace.head_dim),
                        )
                    proxy_mass, proxy_tail_mass = _proxy_selected_mass(
                        selected_cpu=candidate_selected,
                        ranked_cpu=ranked_cpu,
                        ranked_scores_cpu=ranked_scores_cpu,
                        scores_np=scores_np,
                        query_dim=int(trace.head_dim),
                        tail_score_scale=tail_score_scale,
                        tail_score_bias=tail_score_bias,
                    )
                    selected_cpu = candidate_selected
                    chosen_proxy_mass = proxy_mass
                    chosen_proxy_tail_mass = proxy_tail_mass
                    budget = int(candidate_budget)
                    if proxy_mass >= float(args.proxy_mass_target):
                        break
                selected_only_np, _selected_values_np = selected_output(selected_cpu)
            elif online_rule == "proxy_mass_marginal_exact":
                selected_cpu = _selected_for_budget(base=base, ranked_cpu=ranked_cpu, budget=0, context_len=context_len)
                selected_only_np = np.zeros((trace.head_dim,), dtype=np.float32)
                effective_tail_blend = 0.0
                previous_selected = selected_cpu
                for candidate_budget in confidence_budgets:
                    candidate_selected = _selected_for_budget(
                        base=base,
                        ranked_cpu=ranked_cpu,
                        budget=int(candidate_budget),
                        context_len=context_len,
                    )
                    if str(args.tail_score_calibration) == "affine_selected":
                        (
                            tail_score_scale,
                            tail_score_bias,
                            tail_calibration_tokens,
                            tail_pq_relrmse,
                            tail_pq_corr,
                        ) = _fit_selected_pq_logit_calibration(
                            selected_cpu=candidate_selected,
                            ranked_cpu=ranked_cpu,
                            ranked_scores_cpu=ranked_scores_cpu,
                            scores_np=scores_np,
                            query_dim=int(trace.head_dim),
                        )
                    proxy_mass, proxy_tail_mass = _proxy_selected_mass(
                        selected_cpu=candidate_selected,
                        ranked_cpu=ranked_cpu,
                        ranked_scores_cpu=ranked_scores_cpu,
                        scores_np=scores_np,
                        query_dim=int(trace.head_dim),
                        tail_score_scale=tail_score_scale,
                        tail_score_bias=tail_score_bias,
                    )
                    marginal_exact_mass, marginal_score_gap = _selected_exact_marginal_metrics(
                        selected_cpu=candidate_selected,
                        previous_selected_cpu=previous_selected,
                        scores_np=scores_np,
                    )
                    selected_cpu = candidate_selected
                    chosen_proxy_mass = proxy_mass
                    chosen_proxy_tail_mass = proxy_tail_mass
                    budget = int(candidate_budget)
                    previous_selected = candidate_selected
                    can_stop = (
                        budget >= int(args.marginal_min_budget)
                        and proxy_mass >= float(args.proxy_mass_target)
                        and marginal_exact_mass <= float(args.marginal_mass_max)
                        and marginal_score_gap <= float(args.marginal_score_gap_max)
                    )
                    if can_stop:
                        break
                selected_only_np, _selected_values_np = selected_output(selected_cpu)
            elif online_rule == "proxy_tail_delta":
                tail_budget = int(args.tail_confidence_budget)
                tail_selected = _selected_for_budget(base=base, ranked_cpu=ranked_cpu, budget=tail_budget, context_len=context_len)
                tail_selected_only, tail_selected_values_np = selected_output(tail_selected)
                if str(args.tail_score_calibration) == "affine_selected":
                    tail_score_scale, tail_score_bias, tail_calibration_tokens, tail_pq_relrmse, tail_pq_corr = (
                        _fit_selected_pq_logit_calibration(
                            selected_cpu=tail_selected,
                            ranked_cpu=ranked_cpu,
                            ranked_scores_cpu=ranked_scores_cpu,
                            scores_np=scores_np,
                            query_dim=int(trace.head_dim),
                        )
                    )
                approx_tail_np, tail_count_candidate, tail_population_candidate, tail_mb_candidate = _compressed_tail_output(
                    index=index,
                    values_np=values_np,
                    scores_np=scores_np,
                    ranked_cpu=ranked_cpu,
                    ranked_scores_cpu=ranked_scores_cpu,
                    selected_cpu=tail_selected,
                    query_dim=int(trace.head_dim),
                    subbits=int(args.subbits),
                    value_bytes=int(args.value_bytes),
                    mode=str(args.tail_mode),
                    value_subvecs=int(args.value_subvecs),
                    value_subbits=int(args.value_subbits),
                    selected_values_np=tail_selected_values_np,
                    tail_score_scale=tail_score_scale,
                    tail_score_bias=tail_score_bias,
                )
                correction = approx_tail_np.astype(np.float64, copy=False) - tail_selected_only.astype(np.float64, copy=False)
                tail_delta_ratio = float(np.linalg.norm(correction)) / max(
                    float(np.linalg.norm(tail_selected_only.astype(np.float64, copy=False))),
                    1e-20,
                )
                proxy_mass, proxy_tail_mass = _proxy_selected_mass(
                    selected_cpu=tail_selected,
                    ranked_cpu=ranked_cpu,
                    ranked_scores_cpu=ranked_scores_cpu,
                    scores_np=scores_np,
                    query_dim=int(trace.head_dim),
                    tail_score_scale=tail_score_scale,
                    tail_score_bias=tail_score_bias,
                )
                tail_confidence_pass = (
                    float(args.tail_delta_min) <= tail_delta_ratio <= float(args.tail_delta_max)
                    and float(args.tail_proxy_mass_min) <= proxy_tail_mass <= float(args.tail_proxy_mass_max)
                    and tail_pq_corr >= float(args.tail_pq_corr_min)
                    and tail_pq_relrmse <= float(args.tail_pq_relrmse_max)
                )
                if tail_confidence_pass:
                    budget = tail_budget
                    selected_cpu = tail_selected
                    selected_only_np = tail_selected_only
                    chosen_proxy_mass = proxy_mass
                    chosen_proxy_tail_mass = proxy_tail_mass
                    effective_tail_blend = float(args.tail_blend)
                    approx_head_np = (
                        approx_tail_np
                        if effective_tail_blend >= 1.0
                        else (tail_selected_only + effective_tail_blend * (approx_tail_np - tail_selected_only)).astype(
                            np.float32,
                            copy=False,
                        )
                    )
                    tail_count = int(tail_count_candidate)
                    tail_population = int(tail_population_candidate)
                    attn_seconds = 0.0
                    tail_mb = float(tail_mb_candidate)
                else:
                    confidence_mb += float(tail_mb_candidate)
                    effective_tail_blend = 0.0
                    selected_cpu = tail_selected
                    chosen_proxy_mass = proxy_mass
                    chosen_proxy_tail_mass = proxy_tail_mass
                    budget = tail_budget
                    fallback_budgets = sorted(
                        set([tail_budget] + [int(b) for b in confidence_budgets if int(b) >= tail_budget])
                    )
                    for candidate_budget in fallback_budgets:
                        candidate_selected = _selected_for_budget(
                            base=base,
                            ranked_cpu=ranked_cpu,
                            budget=int(candidate_budget),
                            context_len=context_len,
                        )
                        if str(args.tail_score_calibration) == "affine_selected":
                            (
                                tail_score_scale,
                                tail_score_bias,
                                tail_calibration_tokens,
                                tail_pq_relrmse,
                                tail_pq_corr,
                            ) = _fit_selected_pq_logit_calibration(
                                selected_cpu=candidate_selected,
                                ranked_cpu=ranked_cpu,
                                ranked_scores_cpu=ranked_scores_cpu,
                                scores_np=scores_np,
                                query_dim=int(trace.head_dim),
                            )
                        proxy_mass, proxy_tail_mass = _proxy_selected_mass(
                            selected_cpu=candidate_selected,
                            ranked_cpu=ranked_cpu,
                            ranked_scores_cpu=ranked_scores_cpu,
                            scores_np=scores_np,
                            query_dim=int(trace.head_dim),
                            tail_score_scale=tail_score_scale,
                            tail_score_bias=tail_score_bias,
                        )
                        selected_cpu = candidate_selected
                        chosen_proxy_mass = proxy_mass
                        chosen_proxy_tail_mass = proxy_tail_mass
                        budget = int(candidate_budget)
                        if proxy_mass >= float(args.proxy_mass_target):
                            break
                    selected_only_np, _selected_values_np = selected_output(selected_cpu)
            elif online_rule == "probe_tail_switch":
                tail_budget = int(args.tail_confidence_budget)
                probe_budget = max(tail_budget, int(args.tail_probe_budget))
                selected_cpu = _selected_for_budget(base=base, ranked_cpu=ranked_cpu, budget=0, context_len=context_len)
                selected_only_np = np.zeros((trace.head_dim,), dtype=np.float32)
                effective_tail_blend = 0.0
                previous_selected = selected_cpu
                early_stop = False
                for candidate_budget in [int(b) for b in confidence_budgets if int(b) < tail_budget]:
                    candidate_selected = _selected_for_budget(
                        base=base,
                        ranked_cpu=ranked_cpu,
                        budget=int(candidate_budget),
                        context_len=context_len,
                    )
                    if str(args.tail_score_calibration) == "affine_selected":
                        (
                            tail_score_scale,
                            tail_score_bias,
                            tail_calibration_tokens,
                            tail_pq_relrmse,
                            tail_pq_corr,
                        ) = _fit_selected_pq_logit_calibration(
                            selected_cpu=candidate_selected,
                            ranked_cpu=ranked_cpu,
                            ranked_scores_cpu=ranked_scores_cpu,
                            scores_np=scores_np,
                            query_dim=int(trace.head_dim),
                        )
                    proxy_mass, proxy_tail_mass = _proxy_selected_mass(
                        selected_cpu=candidate_selected,
                        ranked_cpu=ranked_cpu,
                        ranked_scores_cpu=ranked_scores_cpu,
                        scores_np=scores_np,
                        query_dim=int(trace.head_dim),
                        tail_score_scale=tail_score_scale,
                        tail_score_bias=tail_score_bias,
                    )
                    marginal_exact_mass, marginal_score_gap = _selected_exact_marginal_metrics(
                        selected_cpu=candidate_selected,
                        previous_selected_cpu=previous_selected,
                        scores_np=scores_np,
                    )
                    selected_cpu = candidate_selected
                    chosen_proxy_mass = proxy_mass
                    chosen_proxy_tail_mass = proxy_tail_mass
                    budget = int(candidate_budget)
                    previous_selected = candidate_selected
                    if (
                        budget >= int(args.marginal_min_budget)
                        and proxy_mass >= float(args.proxy_mass_target)
                        and marginal_exact_mass <= float(args.marginal_mass_max)
                        and marginal_score_gap <= float(args.marginal_score_gap_max)
                    ):
                        early_stop = True
                        break
                if early_stop:
                    selected_only_np, _selected_values_np = selected_output(selected_cpu)
                else:
                    tail_selected = _selected_for_budget(
                        base=base,
                        ranked_cpu=ranked_cpu,
                        budget=tail_budget,
                        context_len=context_len,
                    )
                    probe_selected = _selected_for_budget(
                        base=base,
                        ranked_cpu=ranked_cpu,
                        budget=probe_budget,
                        context_len=context_len,
                    )
                    tail_selected_only, tail_selected_values_np = selected_output(tail_selected)
                    if np.array_equal(tail_selected, probe_selected):
                        probe_selected_only = tail_selected_only
                        probe_selected_values_np = tail_selected_values_np
                    else:
                        probe_selected_only, probe_selected_values_np = selected_output(probe_selected)
                    if str(args.tail_score_calibration) == "affine_selected":
                        (
                            tail_score_scale,
                            tail_score_bias,
                            tail_calibration_tokens,
                            tail_pq_relrmse,
                            tail_pq_corr,
                        ) = _fit_selected_pq_logit_calibration(
                            selected_cpu=tail_selected,
                            ranked_cpu=ranked_cpu,
                            ranked_scores_cpu=ranked_scores_cpu,
                            scores_np=scores_np,
                            query_dim=int(trace.head_dim),
                        )
                    approx_tail_np, tail_count_candidate, tail_population_candidate, tail_mb_candidate = _compressed_tail_output(
                        index=index,
                        values_np=values_np,
                        scores_np=scores_np,
                        ranked_cpu=ranked_cpu,
                        ranked_scores_cpu=ranked_scores_cpu,
                        selected_cpu=tail_selected,
                        query_dim=int(trace.head_dim),
                        subbits=int(args.subbits),
                        value_bytes=int(args.value_bytes),
                        mode=str(args.tail_mode),
                        value_subvecs=int(args.value_subvecs),
                        value_subbits=int(args.value_subbits),
                        selected_values_np=tail_selected_values_np,
                        tail_score_scale=tail_score_scale,
                        tail_score_bias=tail_score_bias,
                    )
                    tail_probe_rel_l2 = float(
                        np.linalg.norm(approx_tail_np.astype(np.float64, copy=False) - probe_selected_only.astype(np.float64, copy=False))
                    ) / max(float(np.linalg.norm(probe_selected_only.astype(np.float64, copy=False))), 1e-20)
                    proxy_mass, proxy_tail_mass = _proxy_selected_mass(
                        selected_cpu=tail_selected,
                        ranked_cpu=ranked_cpu,
                        ranked_scores_cpu=ranked_scores_cpu,
                        scores_np=scores_np,
                        query_dim=int(trace.head_dim),
                        tail_score_scale=tail_score_scale,
                        tail_score_bias=tail_score_bias,
                    )
                    probe_proxy_mass, _probe_proxy_tail_mass = _proxy_selected_mass(
                        selected_cpu=probe_selected,
                        ranked_cpu=ranked_cpu,
                        ranked_scores_cpu=ranked_scores_cpu,
                        scores_np=scores_np,
                        query_dim=int(trace.head_dim),
                        tail_score_scale=tail_score_scale,
                        tail_score_bias=tail_score_bias,
                    )
                    tail_confidence_pass = (
                        tail_probe_rel_l2 <= float(args.tail_probe_rel_l2_max)
                        and proxy_mass >= float(args.tail_proxy_mass_min)
                        and proxy_tail_mass <= float(args.tail_proxy_mass_max)
                        and tail_pq_corr >= float(args.tail_pq_corr_min)
                        and tail_pq_relrmse <= float(args.tail_pq_relrmse_max)
                    )
                    chosen_proxy_mass = proxy_mass
                    chosen_proxy_tail_mass = proxy_tail_mass
                    if tail_confidence_pass:
                        selected_cpu = probe_selected
                        selected_only_np = probe_selected_only
                        budget = probe_budget
                        effective_tail_blend = _blend_from_probe_rule(
                            args,
                            tail_selected_only,
                            approx_tail_np,
                            probe_selected_only,
                            selected_proxy_mass=proxy_mass,
                            probe_proxy_mass=probe_proxy_mass,
                        )
                        if np.array_equal(tail_selected, probe_selected):
                            precomputed_tail_np = approx_tail_np
                            precomputed_tail_count = int(tail_count_candidate)
                            precomputed_tail_population = int(tail_population_candidate)
                            precomputed_tail_mb = float(tail_mb_candidate)
                            tail_estimator_reused = True
                        else:
                            confidence_mb += float(tail_mb_candidate)
                    else:
                        confidence_mb += float(tail_mb_candidate)
                        effective_tail_blend = 0.0
                        selected_cpu = probe_selected
                        selected_only_np = probe_selected_only
                        budget = probe_budget
                        previous_selected = probe_selected
                        for candidate_budget in sorted(
                            set([probe_budget] + [int(b) for b in confidence_budgets if int(b) >= probe_budget])
                        ):
                            candidate_selected = _selected_for_budget(
                                base=base,
                                ranked_cpu=ranked_cpu,
                                budget=int(candidate_budget),
                                context_len=context_len,
                            )
                            if str(args.tail_score_calibration) == "affine_selected":
                                (
                                    tail_score_scale,
                                    tail_score_bias,
                                    tail_calibration_tokens,
                                    tail_pq_relrmse,
                                    tail_pq_corr,
                                ) = _fit_selected_pq_logit_calibration(
                                    selected_cpu=candidate_selected,
                                    ranked_cpu=ranked_cpu,
                                    ranked_scores_cpu=ranked_scores_cpu,
                                    scores_np=scores_np,
                                    query_dim=int(trace.head_dim),
                                )
                            proxy_mass, proxy_tail_mass = _proxy_selected_mass(
                                selected_cpu=candidate_selected,
                                ranked_cpu=ranked_cpu,
                                ranked_scores_cpu=ranked_scores_cpu,
                                scores_np=scores_np,
                                query_dim=int(trace.head_dim),
                                tail_score_scale=tail_score_scale,
                                tail_score_bias=tail_score_bias,
                            )
                            marginal_exact_mass, marginal_score_gap = _selected_exact_marginal_metrics(
                                selected_cpu=candidate_selected,
                                previous_selected_cpu=previous_selected,
                                scores_np=scores_np,
                            )
                            selected_cpu = candidate_selected
                            chosen_proxy_mass = proxy_mass
                            chosen_proxy_tail_mass = proxy_tail_mass
                            budget = int(candidate_budget)
                            previous_selected = candidate_selected
                            if (
                                budget >= int(args.marginal_min_budget)
                                and proxy_mass >= float(args.proxy_mass_target)
                                and marginal_exact_mass <= float(args.marginal_mass_max)
                                and marginal_score_gap <= float(args.marginal_score_gap_max)
                            ):
                                break
                        selected_only_np, _selected_values_np = selected_output(selected_cpu)
            elif online_rule == "entropy_probe_tail_switch":
                max_budget = int(args.entropy_max_budget)
                granularity = int(args.entropy_budget_granularity)
                candidate_budgets = sorted(
                    set([int(b) for b in confidence_budgets if int(b) <= max_budget] + [max_budget])
                )
                selected_cpu = _selected_for_budget(base=base, ranked_cpu=ranked_cpu, budget=0, context_len=context_len)
                selected_only_np = np.zeros((trace.head_dim,), dtype=np.float32)
                effective_tail_blend = 0.0
                previous_selected = selected_cpu
                early_stop = False
                tail_budget = max_budget
                for candidate_budget in candidate_budgets:
                    if int(candidate_budget) < int(args.entropy_min_budget):
                        continue
                    candidate_selected = _selected_for_budget(
                        base=base,
                        ranked_cpu=ranked_cpu,
                        budget=int(candidate_budget),
                        context_len=context_len,
                    )
                    if str(args.tail_score_calibration) == "affine_selected":
                        (
                            tail_score_scale,
                            tail_score_bias,
                            tail_calibration_tokens,
                            tail_pq_relrmse,
                            tail_pq_corr,
                            tail_pq_residual_std,
                        ) = _fit_selected_pq_logit_uncertainty(
                            selected_cpu=candidate_selected,
                            ranked_cpu=ranked_cpu,
                            ranked_scores_cpu=ranked_scores_cpu,
                            scores_np=scores_np,
                            query_dim=int(trace.head_dim),
                        )
                    entropy_stats = _proxy_distribution_stats(
                        selected_cpu=candidate_selected,
                        ranked_cpu=ranked_cpu,
                        ranked_scores_cpu=ranked_scores_cpu,
                        scores_np=scores_np,
                        query_dim=int(trace.head_dim),
                        tail_score_scale=tail_score_scale,
                        tail_score_bias=tail_score_bias,
                        tail_logit_bonus=float(args.entropy_ucb_z) * float(tail_pq_residual_std),
                    )
                    entropy_effective_support = float(entropy_stats["effective_support"])
                    entropy_tail_effective_support = float(entropy_stats["tail_effective_support"])
                    entropy_tail_mass = float(entropy_stats["tail_mass"])
                    entropy_required_budget = max(
                        int(args.entropy_min_budget),
                        _round_budget_up(
                            float(args.entropy_budget_scale) * entropy_effective_support,
                            granularity=granularity,
                            max_budget=max_budget,
                        ),
                    )
                    proxy_mass = float(entropy_stats["selected_mass"])
                    proxy_tail_mass = float(entropy_stats["tail_mass"])
                    marginal_exact_mass, marginal_score_gap = _selected_exact_marginal_metrics(
                        selected_cpu=candidate_selected,
                        previous_selected_cpu=previous_selected,
                        scores_np=scores_np,
                    )
                    selected_cpu = candidate_selected
                    chosen_proxy_mass = proxy_mass
                    chosen_proxy_tail_mass = proxy_tail_mass
                    budget = int(candidate_budget)
                    previous_selected = candidate_selected
                    can_stop = (
                        int(selected_cpu.size) >= int(entropy_required_budget)
                        and proxy_mass >= float(args.proxy_mass_target)
                        and marginal_exact_mass <= float(args.marginal_mass_max)
                        and marginal_score_gap <= float(args.marginal_score_gap_max)
                    )
                    if can_stop:
                        tail_budget = int(candidate_budget)
                        if proxy_tail_mass <= float(args.entropy_tail_mass_max):
                            early_stop = True
                        break
                if early_stop:
                    selected_only_np = _selected_only_output(keys_np, values_np, query_np, selected_cpu)
                else:
                    tail_budget = max(int(args.entropy_min_budget), min(max_budget, int(tail_budget)))
                    probe_budget = _round_budget_up(
                        max(float(tail_budget + max(1, granularity)), float(args.entropy_probe_scale) * float(tail_budget)),
                        granularity=granularity,
                        max_budget=max_budget,
                    )
                    probe_budget = max(tail_budget, int(probe_budget))
                    tail_selected = _selected_for_budget(
                        base=base,
                        ranked_cpu=ranked_cpu,
                        budget=tail_budget,
                        context_len=context_len,
                    )
                    probe_selected = _selected_for_budget(
                        base=base,
                        ranked_cpu=ranked_cpu,
                        budget=probe_budget,
                        context_len=context_len,
                    )
                    tail_selected_only = _selected_only_output(keys_np, values_np, query_np, tail_selected)
                    probe_selected_only = _selected_only_output(keys_np, values_np, query_np, probe_selected)
                    if str(args.tail_score_calibration) == "affine_selected":
                        (
                            tail_score_scale,
                            tail_score_bias,
                            tail_calibration_tokens,
                            tail_pq_relrmse,
                            tail_pq_corr,
                            tail_pq_residual_std,
                        ) = _fit_selected_pq_logit_uncertainty(
                            selected_cpu=tail_selected,
                            ranked_cpu=ranked_cpu,
                            ranked_scores_cpu=ranked_scores_cpu,
                            scores_np=scores_np,
                            query_dim=int(trace.head_dim),
                        )
                    approx_tail_np, tail_count_candidate, tail_population_candidate, tail_mb_candidate = _compressed_tail_output(
                        index=index,
                        values_np=values_np,
                        scores_np=scores_np,
                        ranked_cpu=ranked_cpu,
                        ranked_scores_cpu=ranked_scores_cpu,
                        selected_cpu=tail_selected,
                        query_dim=int(trace.head_dim),
                        subbits=int(args.subbits),
                        value_bytes=int(args.value_bytes),
                        mode=str(args.tail_mode),
                        value_subvecs=int(args.value_subvecs),
                        value_subbits=int(args.value_subbits),
                        tail_score_scale=tail_score_scale,
                        tail_score_bias=tail_score_bias,
                    )
                    tail_probe_rel_l2 = float(
                        np.linalg.norm(approx_tail_np.astype(np.float64, copy=False) - probe_selected_only.astype(np.float64, copy=False))
                    ) / max(float(np.linalg.norm(probe_selected_only.astype(np.float64, copy=False))), 1e-20)
                    entropy_stats = _proxy_distribution_stats(
                        selected_cpu=tail_selected,
                        ranked_cpu=ranked_cpu,
                        ranked_scores_cpu=ranked_scores_cpu,
                        scores_np=scores_np,
                        query_dim=int(trace.head_dim),
                        tail_score_scale=tail_score_scale,
                        tail_score_bias=tail_score_bias,
                        tail_logit_bonus=float(args.entropy_ucb_z) * float(tail_pq_residual_std),
                    )
                    entropy_effective_support = float(entropy_stats["effective_support"])
                    entropy_tail_effective_support = float(entropy_stats["tail_effective_support"])
                    entropy_tail_mass = float(entropy_stats["tail_mass"])
                    chosen_proxy_mass = float(entropy_stats["selected_mass"])
                    chosen_proxy_tail_mass = float(entropy_stats["tail_mass"])
                    tail_confidence_pass = (
                        tail_probe_rel_l2 <= float(args.tail_probe_rel_l2_max)
                        and float(entropy_stats["selected_mass"]) >= float(args.tail_proxy_mass_min)
                        and float(entropy_stats["tail_mass"]) <= float(args.tail_proxy_mass_max)
                        and tail_pq_corr >= float(args.tail_pq_corr_min)
                        and tail_pq_relrmse <= float(args.tail_pq_relrmse_max)
                    )
                    confidence_mb += float(tail_mb_candidate)
                    if tail_confidence_pass:
                        selected_cpu = probe_selected
                        selected_only_np = probe_selected_only
                        budget = int(probe_budget)
                        effective_tail_blend = _blend_from_probe_rule(
                            args,
                            tail_selected_only,
                            approx_tail_np,
                            probe_selected_only,
                        )
                    else:
                        effective_tail_blend = 0.0
                        selected_cpu = probe_selected
                        selected_only_np = probe_selected_only
                        budget = int(probe_budget)
                        previous_selected = probe_selected
                        for candidate_budget in [int(b) for b in candidate_budgets if int(b) >= int(probe_budget)]:
                            candidate_selected = _selected_for_budget(
                                base=base,
                                ranked_cpu=ranked_cpu,
                                budget=int(candidate_budget),
                                context_len=context_len,
                            )
                            if str(args.tail_score_calibration) == "affine_selected":
                                (
                                    tail_score_scale,
                                    tail_score_bias,
                                    tail_calibration_tokens,
                                    tail_pq_relrmse,
                                    tail_pq_corr,
                                    tail_pq_residual_std,
                                ) = _fit_selected_pq_logit_uncertainty(
                                    selected_cpu=candidate_selected,
                                    ranked_cpu=ranked_cpu,
                                    ranked_scores_cpu=ranked_scores_cpu,
                                    scores_np=scores_np,
                                    query_dim=int(trace.head_dim),
                                )
                            entropy_stats = _proxy_distribution_stats(
                                selected_cpu=candidate_selected,
                                ranked_cpu=ranked_cpu,
                                ranked_scores_cpu=ranked_scores_cpu,
                                scores_np=scores_np,
                                query_dim=int(trace.head_dim),
                                tail_score_scale=tail_score_scale,
                                tail_score_bias=tail_score_bias,
                                tail_logit_bonus=float(args.entropy_ucb_z) * float(tail_pq_residual_std),
                            )
                            entropy_effective_support = float(entropy_stats["effective_support"])
                            entropy_tail_effective_support = float(entropy_stats["tail_effective_support"])
                            entropy_tail_mass = float(entropy_stats["tail_mass"])
                            entropy_required_budget = max(
                                int(args.entropy_min_budget),
                                _round_budget_up(
                                    float(args.entropy_budget_scale) * entropy_effective_support,
                                    granularity=granularity,
                                    max_budget=max_budget,
                                ),
                            )
                            marginal_exact_mass, marginal_score_gap = _selected_exact_marginal_metrics(
                                selected_cpu=candidate_selected,
                                previous_selected_cpu=previous_selected,
                                scores_np=scores_np,
                            )
                            selected_cpu = candidate_selected
                            chosen_proxy_mass = float(entropy_stats["selected_mass"])
                            chosen_proxy_tail_mass = float(entropy_stats["tail_mass"])
                            budget = int(candidate_budget)
                            previous_selected = candidate_selected
                            if (
                                int(selected_cpu.size) >= int(entropy_required_budget)
                                and chosen_proxy_mass >= float(args.proxy_mass_target)
                                and marginal_exact_mass <= float(args.marginal_mass_max)
                                and marginal_score_gap <= float(args.marginal_score_gap_max)
                            ):
                                break
                        selected_only_np = _selected_only_output(keys_np, values_np, query_np, selected_cpu)
            elif online_rule == "adaptive_entropy_probe_tail_switch":
                max_budget = int(args.entropy_max_budget)
                granularity = max(1, int(args.entropy_budget_granularity))
                growth = max(1.01, float(args.entropy_budget_growth))
                selected_cpu = _selected_for_budget(base=base, ranked_cpu=ranked_cpu, budget=0, context_len=context_len)
                selected_only_np = np.zeros((trace.head_dim,), dtype=np.float32)
                effective_tail_blend = 0.0
                previous_selected = selected_cpu
                k = _round_budget_up(
                    int(args.entropy_min_budget),
                    granularity=granularity,
                    max_budget=max_budget,
                )
                while True:
                    candidate_selected = _selected_for_budget(
                        base=base,
                        ranked_cpu=ranked_cpu,
                        budget=int(k),
                        context_len=context_len,
                    )
                    if str(args.tail_score_calibration) == "affine_selected":
                        (
                            tail_score_scale,
                            tail_score_bias,
                            tail_calibration_tokens,
                            tail_pq_relrmse,
                            tail_pq_corr,
                            tail_pq_residual_std,
                        ) = _fit_selected_pq_logit_uncertainty(
                            selected_cpu=candidate_selected,
                            ranked_cpu=ranked_cpu,
                            ranked_scores_cpu=ranked_scores_cpu,
                            scores_np=scores_np,
                            query_dim=int(trace.head_dim),
                        )
                    entropy_stats = _proxy_distribution_stats(
                        selected_cpu=candidate_selected,
                        ranked_cpu=ranked_cpu,
                        ranked_scores_cpu=ranked_scores_cpu,
                        scores_np=scores_np,
                        query_dim=int(trace.head_dim),
                        tail_score_scale=tail_score_scale,
                        tail_score_bias=tail_score_bias,
                        tail_logit_bonus=float(args.entropy_ucb_z) * float(tail_pq_residual_std),
                    )
                    entropy_effective_support = float(entropy_stats["effective_support"])
                    entropy_tail_effective_support = float(entropy_stats["tail_effective_support"])
                    entropy_tail_mass = float(entropy_stats["tail_mass"])
                    entropy_required_budget = max(
                        int(args.entropy_min_budget),
                        _round_budget_up(
                            float(args.entropy_budget_scale) * entropy_effective_support,
                            granularity=granularity,
                            max_budget=max_budget,
                        ),
                    )
                    marginal_exact_mass, marginal_score_gap = _selected_exact_marginal_metrics(
                        selected_cpu=candidate_selected,
                        previous_selected_cpu=previous_selected,
                        scores_np=scores_np,
                    )
                    selected_cpu = candidate_selected
                    chosen_proxy_mass = float(entropy_stats["selected_mass"])
                    chosen_proxy_tail_mass = float(entropy_stats["tail_mass"])
                    budget = int(k)
                    entropy_budget_reached = int(selected_cpu.size) >= int(entropy_required_budget)
                    exact_stop = (
                        entropy_budget_reached
                        and chosen_proxy_mass >= float(args.proxy_mass_target)
                        and entropy_tail_mass <= float(args.entropy_tail_mass_max)
                        and marginal_exact_mass <= float(args.marginal_mass_max)
                        and marginal_score_gap <= float(args.marginal_score_gap_max)
                    )
                    if exact_stop:
                        selected_only_np = _selected_only_output(keys_np, values_np, query_np, selected_cpu)
                        break
                    if entropy_budget_reached or int(k) >= max_budget:
                        tail_budget = int(k)
                        probe_budget = _round_budget_up(
                            max(float(tail_budget + granularity), float(args.entropy_probe_scale) * float(tail_budget)),
                            granularity=granularity,
                            max_budget=max_budget,
                        )
                        probe_budget = max(tail_budget, int(probe_budget))
                        if probe_budget <= tail_budget and tail_budget >= max_budget:
                            selected_only_np = _selected_only_output(keys_np, values_np, query_np, selected_cpu)
                            break
                        tail_selected = selected_cpu
                        tail_selected_only = _selected_only_output(keys_np, values_np, query_np, tail_selected)
                        probe_selected = _selected_for_budget(
                            base=base,
                            ranked_cpu=ranked_cpu,
                            budget=probe_budget,
                            context_len=context_len,
                        )
                        probe_selected_only = _selected_only_output(keys_np, values_np, query_np, probe_selected)
                        approx_tail_np, tail_count_candidate, tail_population_candidate, tail_mb_candidate = _compressed_tail_output(
                            index=index,
                            values_np=values_np,
                            scores_np=scores_np,
                            ranked_cpu=ranked_cpu,
                            ranked_scores_cpu=ranked_scores_cpu,
                            selected_cpu=tail_selected,
                            query_dim=int(trace.head_dim),
                            subbits=int(args.subbits),
                            value_bytes=int(args.value_bytes),
                            mode=str(args.tail_mode),
                            value_subvecs=int(args.value_subvecs),
                            value_subbits=int(args.value_subbits),
                            tail_score_scale=tail_score_scale,
                            tail_score_bias=tail_score_bias,
                        )
                        confidence_mb += float(tail_mb_candidate)
                        tail_probe_rel_l2 = float(
                            np.linalg.norm(
                                approx_tail_np.astype(np.float64, copy=False)
                                - probe_selected_only.astype(np.float64, copy=False)
                            )
                        ) / max(float(np.linalg.norm(probe_selected_only.astype(np.float64, copy=False))), 1e-20)
                        tail_confidence_pass = (
                            tail_probe_rel_l2 <= float(args.tail_probe_rel_l2_max)
                            and chosen_proxy_mass >= float(args.tail_proxy_mass_min)
                            and chosen_proxy_tail_mass <= float(args.tail_proxy_mass_max)
                            and tail_pq_corr >= float(args.tail_pq_corr_min)
                            and tail_pq_relrmse <= float(args.tail_pq_relrmse_max)
                        )
                        if tail_confidence_pass:
                            selected_cpu = probe_selected
                            selected_only_np = probe_selected_only
                            budget = int(probe_budget)
                            effective_tail_blend = _blend_from_probe_rule(
                                args,
                                tail_selected_only,
                                approx_tail_np,
                                probe_selected_only,
                            )
                            break
                        selected_cpu = probe_selected
                        previous_selected = probe_selected
                        budget = int(probe_budget)
                        if probe_budget >= max_budget:
                            selected_only_np = probe_selected_only
                            break
                        k = _round_budget_up(
                            max(float(probe_budget + granularity), float(growth) * float(probe_budget)),
                            granularity=granularity,
                            max_budget=max_budget,
                        )
                        continue
                    previous_selected = candidate_selected
                    next_k = _round_budget_up(
                        max(
                            float(k + granularity),
                            float(growth) * float(k),
                            float(entropy_required_budget),
                        ),
                        granularity=granularity,
                        max_budget=max_budget,
                    )
                    if int(k) >= max_budget or int(next_k) <= int(k):
                        selected_only_np = _selected_only_output(keys_np, values_np, query_np, selected_cpu)
                        break
                    k = int(next_k)
            elif online_rule in {"pq_proxy_mass_budget", "pq_ranked_mass_budget"}:
                max_budget = int(effective_geometric_max_budget)
                if max_budget <= 0:
                    max_budget = int(rank_budget)
                max_budget = max(0, min(max_budget, int(ranked_cpu.size)))
                granularity = max(1, int(args.geometric_budget_granularity))
                min_budget = _round_budget_up(
                    int(args.geometric_min_budget),
                    granularity=granularity,
                    max_budget=max_budget,
                )
                proxy_target = max(float(args.tail_proxy_mass_min), 1.0 - float(args.tail_proxy_mass_max))
                proxy_target = min(max(float(proxy_target), 0.0), 1.0 - 1.0e-7)
                if max_budget > 0 and ranked_scores_cpu.size > 0:
                    denom_scores = (
                        ranked_scores_cpu[:max_budget]
                        if online_rule == "pq_ranked_mass_budget"
                        else ranked_scores_cpu
                    )
                    denom_scaled = denom_scores.astype(np.float64, copy=False) / math.sqrt(float(trace.head_dim))
                    top_scaled = ranked_scores_cpu[:max_budget].astype(np.float64, copy=False) / math.sqrt(float(trace.head_dim))
                    shift = float(np.max(denom_scaled))
                    top_weights = np.exp(top_scaled - shift)
                    denom = max(float(np.exp(denom_scaled - shift).sum()), 1e-20)
                    cumulative = np.cumsum(top_weights) / denom
                    if proxy_target > 0.0:
                        hit = np.flatnonzero(cumulative >= proxy_target)
                        chosen_budget = int(hit[0] + 1) if hit.size else int(max_budget)
                    else:
                        chosen_budget = int(min_budget)
                    chosen_budget = max(int(min_budget), min(int(max_budget), int(chosen_budget)))
                    chosen_budget = int(math.ceil(float(chosen_budget) / float(granularity)) * granularity)
                    budget = min(int(max_budget), max(0, int(chosen_budget)))
                    chosen_proxy_mass = float(cumulative[max(0, budget - 1)]) if budget > 0 else 0.0
                    chosen_proxy_tail_mass = max(0.0, 1.0 - float(chosen_proxy_mass))
                else:
                    budget = 0
                    chosen_proxy_mass = 0.0
                    chosen_proxy_tail_mass = 1.0
                selected_cpu = _selected_for_budget(
                    base=base,
                    ranked_cpu=ranked_cpu,
                    budget=budget,
                    context_len=context_len,
                )
                selected_only_np, _selected_values_np = selected_output(selected_cpu)
                effective_tail_blend = 0.0 if int(head) in tail_off_heads else float(args.tail_blend)
            elif online_rule in {"geometric_probe_tail_switch", "geometric_stable_tail_switch", "geometric_slope_stability"}:
                max_budget = int(effective_geometric_max_budget)
                granularity = max(1, int(args.geometric_budget_granularity))
                growth = max(1.01, float(args.geometric_growth))
                probe_scale = max(1.01, float(args.geometric_probe_scale))
                k = _round_budget_up(
                    int(args.geometric_min_budget),
                    granularity=granularity,
                    max_budget=max_budget,
                )
                selected_cpu = _selected_for_budget(base=base, ranked_cpu=ranked_cpu, budget=0, context_len=context_len)
                selected_only_np = np.zeros((trace.head_dim,), dtype=np.float32)
                effective_tail_blend = 0.0
                while True:
                    tail_budget = min(int(k), max_budget)
                    tail_selected = _selected_for_budget(
                        base=base,
                        ranked_cpu=ranked_cpu,
                        budget=tail_budget,
                        context_len=context_len,
                    )
                    if str(args.tail_score_calibration) == "affine_selected":
                        (
                            tail_score_scale,
                            tail_score_bias,
                            tail_calibration_tokens,
                            tail_pq_relrmse,
                            tail_pq_corr,
                            tail_pq_residual_std,
                        ) = _fit_selected_pq_logit_uncertainty(
                            selected_cpu=tail_selected,
                            ranked_cpu=ranked_cpu,
                            ranked_scores_cpu=ranked_scores_cpu,
                            scores_np=scores_np,
                            query_dim=int(trace.head_dim),
                        )
                    proxy_mass, proxy_tail_mass = _proxy_selected_mass(
                        selected_cpu=tail_selected,
                        ranked_cpu=ranked_cpu,
                        ranked_scores_cpu=ranked_scores_cpu,
                        scores_np=scores_np,
                        query_dim=int(trace.head_dim),
                        tail_score_scale=tail_score_scale,
                        tail_score_bias=tail_score_bias,
                    )
                    chosen_proxy_mass = float(proxy_mass)
                    chosen_proxy_tail_mass = float(proxy_tail_mass)
                    approx_tail_np, tail_count_candidate, tail_population_candidate, tail_mb_candidate = _compressed_tail_output(
                        index=index,
                        values_np=values_np,
                        scores_np=scores_np,
                        ranked_cpu=ranked_cpu,
                        ranked_scores_cpu=ranked_scores_cpu,
                        selected_cpu=tail_selected,
                        query_dim=int(trace.head_dim),
                        subbits=int(args.subbits),
                        value_bytes=int(args.value_bytes),
                        mode=str(args.tail_mode),
                        value_subvecs=int(args.value_subvecs),
                        value_subbits=int(args.value_subbits),
                        tail_score_scale=tail_score_scale,
                        tail_score_bias=tail_score_bias,
                    )
                    confidence_mb += float(tail_mb_candidate)
                    tail_selected_only = _selected_only_output(keys_np, values_np, query_np, tail_selected)
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
                        context_len=context_len,
                    )
                    probe_selected_only = _selected_only_output(keys_np, values_np, query_np, probe_selected)
                    if online_rule == "geometric_slope_stability":
                        delta_budget = max(granularity, int(probe_budget) - int(tail_budget))
                        minus_budget = max(0, int(tail_budget) - int(delta_budget))
                        minus_selected = _selected_for_budget(
                            base=base,
                            ranked_cpu=ranked_cpu,
                            budget=minus_budget,
                            context_len=context_len,
                        )
                        minus_selected_only = _selected_only_output(keys_np, values_np, query_np, minus_selected)
                        minus64 = minus_selected_only.astype(np.float64, copy=False)
                        tail64 = tail_selected_only.astype(np.float64, copy=False)
                        probe64 = probe_selected_only.astype(np.float64, copy=False)
                        slope_forward_rel_l2 = float(np.linalg.norm(probe64 - tail64)) / max(
                            float(np.linalg.norm(probe64)),
                            1e-20,
                        )
                        slope_backward_rel_l2 = float(np.linalg.norm(tail64 - minus64)) / max(
                            float(np.linalg.norm(tail64)),
                            1e-20,
                        )
                        if slope_backward_rel_l2 <= 1e-20:
                            slope_ratio = 0.0 if slope_forward_rel_l2 <= 1e-20 else float("inf")
                        else:
                            slope_ratio = float(slope_forward_rel_l2) / float(slope_backward_rel_l2)
                        slope_curvature_rel_l2 = float(np.linalg.norm(probe64 - 2.0 * tail64 + minus64)) / max(
                            float(np.linalg.norm(probe64)),
                            1e-20,
                        )
                        slope_minus_budget = int(minus_budget)
                        slope_center_budget = int(tail_budget)
                        slope_plus_budget = int(probe_budget)
                    tail_probe_rel_l2 = float(
                        np.linalg.norm(
                            approx_tail_np.astype(np.float64, copy=False)
                            - probe_selected_only.astype(np.float64, copy=False)
                        )
                    ) / max(float(np.linalg.norm(probe_selected_only.astype(np.float64, copy=False))), 1e-20)
                    marginal_exact_mass, marginal_score_gap = _selected_exact_marginal_metrics(
                        selected_cpu=probe_selected,
                        previous_selected_cpu=tail_selected,
                        scores_np=scores_np,
                    )
                    if int(args.audit_tail_samples) > 0:
                        audit_tail_mass, audit_tail_logit_gap, audit_tail_count, audit_tail_population = _sample_tail_audit(
                            selected_cpu=probe_selected,
                            ranked_cpu=ranked_cpu,
                            scores_np=scores_np,
                            context_len=context_len,
                            samples=int(args.audit_tail_samples),
                            seed=int(args.tail_seed),
                            qidx=int(qidx),
                            head=int(head),
                            mode=str(args.audit_tail_mode),
                            bands=int(args.audit_tail_bands),
                        )
                        confidence_mb += float(audit_tail_count * trace.head_dim * int(args.key_bytes)) / MB
                    stable_tail_probe_rel_l2 = 0.0
                    if online_rule == "geometric_stable_tail_switch":
                        if str(args.tail_score_calibration) == "affine_selected":
                            (
                                tail_score_scale,
                                tail_score_bias,
                                tail_calibration_tokens,
                                tail_pq_relrmse,
                                tail_pq_corr,
                                tail_pq_residual_std,
                            ) = _fit_selected_pq_logit_uncertainty(
                                selected_cpu=probe_selected,
                                ranked_cpu=ranked_cpu,
                                ranked_scores_cpu=ranked_scores_cpu,
                                scores_np=scores_np,
                                query_dim=int(trace.head_dim),
                            )
                        approx_probe_tail_np, _probe_tail_count, _probe_tail_population, probe_tail_mb_candidate = _compressed_tail_output(
                            index=index,
                            values_np=values_np,
                            scores_np=scores_np,
                            ranked_cpu=ranked_cpu,
                            ranked_scores_cpu=ranked_scores_cpu,
                            selected_cpu=probe_selected,
                            query_dim=int(trace.head_dim),
                            subbits=int(args.subbits),
                            value_bytes=int(args.value_bytes),
                            mode=str(args.tail_mode),
                            value_subvecs=int(args.value_subvecs),
                            value_subbits=int(args.value_subbits),
                            tail_score_scale=tail_score_scale,
                            tail_score_bias=tail_score_bias,
                        )
                        confidence_mb += float(probe_tail_mb_candidate)
                        stable_tail_probe_rel_l2 = float(
                            np.linalg.norm(
                                approx_probe_tail_np.astype(np.float64, copy=False)
                                - probe_selected_only.astype(np.float64, copy=False)
                            )
                        ) / max(float(np.linalg.norm(probe_selected_only.astype(np.float64, copy=False))), 1e-20)
                    selected_cpu = probe_selected
                    selected_only_np = probe_selected_only
                    budget = int(probe_budget)
                    slope_stability_pass = True
                    if online_rule == "geometric_slope_stability":
                        slope_stability_pass = (
                            slope_forward_rel_l2 <= float(args.slope_forward_rel_l2_max)
                            and slope_backward_rel_l2 <= float(args.slope_backward_rel_l2_max)
                            and slope_ratio <= float(args.slope_ratio_max)
                            and slope_curvature_rel_l2 <= float(args.slope_curvature_rel_l2_max)
                        )
                    tail_confidence_pass = (
                        tail_probe_rel_l2 <= float(args.tail_probe_rel_l2_max)
                        and stable_tail_probe_rel_l2 <= float(args.stable_tail_probe_rel_l2_max)
                        and slope_stability_pass
                        and audit_tail_mass <= float(args.audit_tail_mass_max)
                        and audit_tail_logit_gap <= float(args.audit_tail_logit_gap_max)
                        and chosen_proxy_mass >= float(args.tail_proxy_mass_min)
                        and chosen_proxy_tail_mass <= float(args.tail_proxy_mass_max)
                        and tail_pq_corr >= float(args.tail_pq_corr_min)
                        and tail_pq_relrmse <= float(args.tail_pq_relrmse_max)
                    )
                    if tail_confidence_pass:
                        effective_tail_blend = _blend_from_probe_rule(
                            args,
                            tail_selected_only,
                            approx_tail_np,
                            probe_selected_only,
                        )
                        break
                    effective_tail_blend = 0.0
                    if probe_budget >= max_budget:
                        break
                    next_k = _round_budget_up(
                        max(float(probe_budget + granularity), growth * float(probe_budget)),
                        granularity=granularity,
                        max_budget=max_budget,
                    )
                    if int(next_k) <= int(probe_budget):
                        break
                    k = int(next_k)
            elif online_rule == "geometric_exact_delta":
                max_budget = int(effective_geometric_max_budget)
                granularity = max(1, int(args.geometric_budget_granularity))
                growth = max(1.01, float(args.geometric_growth))
                probe_scale = max(1.01, float(args.geometric_probe_scale))
                k = _round_budget_up(
                    int(args.geometric_min_budget),
                    granularity=granularity,
                    max_budget=max_budget,
                )
                selected_cpu = _selected_for_budget(base=base, ranked_cpu=ranked_cpu, budget=0, context_len=context_len)
                selected_only_np = np.zeros((trace.head_dim,), dtype=np.float32)
                effective_tail_blend = 0.0
                while True:
                    tail_budget = min(int(k), max_budget)
                    tail_selected = _selected_for_budget(
                        base=base,
                        ranked_cpu=ranked_cpu,
                        budget=tail_budget,
                        context_len=context_len,
                    )
                    tail_selected_only, _tail_selected_values_np = selected_output(tail_selected)
                    if str(args.tail_score_calibration) == "affine_selected":
                        (
                            tail_score_scale,
                            tail_score_bias,
                            tail_calibration_tokens,
                            tail_pq_relrmse,
                            tail_pq_corr,
                            tail_pq_residual_std,
                        ) = _fit_selected_pq_logit_uncertainty(
                            selected_cpu=tail_selected,
                            ranked_cpu=ranked_cpu,
                            ranked_scores_cpu=ranked_scores_cpu,
                            scores_np=scores_np,
                            query_dim=int(trace.head_dim),
                        )
                    proxy_mass, proxy_tail_mass = _proxy_selected_mass(
                        selected_cpu=tail_selected,
                        ranked_cpu=ranked_cpu,
                        ranked_scores_cpu=ranked_scores_cpu,
                        scores_np=scores_np,
                        query_dim=int(trace.head_dim),
                        tail_score_scale=tail_score_scale,
                        tail_score_bias=tail_score_bias,
                    )
                    chosen_proxy_mass = float(proxy_mass)
                    chosen_proxy_tail_mass = float(proxy_tail_mass)
                    probe_budget = _round_budget_up(
                        max(float(tail_budget + granularity), probe_scale * float(tail_budget)),
                        granularity=granularity,
                        max_budget=max_budget,
                    )
                    probe_budget = max(tail_budget, int(probe_budget))
                    if probe_budget <= tail_budget:
                        selected_cpu = tail_selected
                        selected_only_np = tail_selected_only
                        budget = int(tail_budget)
                        break
                    probe_selected = _selected_for_budget(
                        base=base,
                        ranked_cpu=ranked_cpu,
                        budget=probe_budget,
                        context_len=context_len,
                    )
                    probe_selected_only, _probe_selected_values_np = selected_output(probe_selected)
                    exact_delta_rel_l2 = float(
                        np.linalg.norm(
                            probe_selected_only.astype(np.float64, copy=False)
                            - tail_selected_only.astype(np.float64, copy=False)
                        )
                    ) / max(float(np.linalg.norm(probe_selected_only.astype(np.float64, copy=False))), 1e-20)
                    marginal_exact_mass, marginal_score_gap = _selected_exact_marginal_metrics(
                        selected_cpu=probe_selected,
                        previous_selected_cpu=tail_selected,
                        scores_np=scores_np,
                    )
                    probe_proxy_mass, probe_proxy_tail_mass = _proxy_selected_mass(
                        selected_cpu=probe_selected,
                        ranked_cpu=ranked_cpu,
                        ranked_scores_cpu=ranked_scores_cpu,
                        scores_np=scores_np,
                        query_dim=int(trace.head_dim),
                        tail_score_scale=tail_score_scale,
                        tail_score_bias=tail_score_bias,
                    )
                    selected_cpu = probe_selected
                    selected_only_np = probe_selected_only
                    budget = int(probe_budget)
                    chosen_proxy_mass = float(probe_proxy_mass)
                    chosen_proxy_tail_mass = float(probe_proxy_tail_mass)
                    tail_confidence_pass = (
                        exact_delta_rel_l2 <= float(args.exact_delta_rel_l2_max)
                        and chosen_proxy_mass >= float(args.tail_proxy_mass_min)
                        and chosen_proxy_tail_mass <= float(args.tail_proxy_mass_max)
                        and marginal_exact_mass <= float(args.marginal_mass_max)
                        and marginal_score_gap <= float(args.marginal_score_gap_max)
                    )
                    if tail_confidence_pass or probe_budget >= max_budget:
                        break
                    next_k = _round_budget_up(
                        max(float(probe_budget + granularity), growth * float(probe_budget)),
                        granularity=granularity,
                        max_budget=max_budget,
                    )
                    if int(next_k) <= int(probe_budget):
                        break
                    k = int(next_k)
            else:
                raise ValueError(f"unknown online_confidence_rule: {online_rule}")

            selected_key_scores_override = None
            selected_key_exact_tokens = int(selected_cpu.size)
            selected_key_compressed_tokens = 0
            selected_key_calibration_probe_count = 0
            selected_key_active = (
                str(args.selected_key_mode) != "exact"
                and (
                    int(args.selected_key_min_context) <= 0
                    or int(context_len) >= int(args.selected_key_min_context)
                )
            )
            if selected_key_active:
                _selected_output_np, final_selected_values_for_key = selected_output(selected_cpu)
                (
                    selected_key_scores_override,
                    selected_key_exact_tokens,
                    selected_key_compressed_tokens,
                    selected_key_calibration_probe_count,
                ) = _band_calibrated_selected_scores(
                    selected_cpu=selected_cpu,
                    ranked_cpu=ranked_cpu,
                    ranked_scores_cpu=ranked_scores_cpu,
                    exact_scores_np=scores_np,
                    query_dim=int(trace.head_dim),
                    probes=int(args.selected_key_calibration_probes),
                    bands=int(args.selected_key_calibration_bands),
                    exact_selector_mass=float(args.selected_key_exact_selector_mass),
                    min_exact_top=int(args.selected_key_min_exact_top),
                    max_exact_top=int(args.selected_key_max_exact_top),
                )
                selected_only_np = _selected_output_from_scores(
                    values_np,
                    selected_cpu,
                    selected_key_scores_override,
                    values_override=final_selected_values_for_key,
                )

            selected = torch.as_tensor(selected_cpu, dtype=torch.long, device=device)
            if online_rule == "proxy_tail_delta" and tail_confidence_pass:
                pass
            elif precomputed_tail_np is not None and effective_tail_blend > 0.0 and selected_key_scores_override is None:
                attn_seconds = 0.0
                tail_count = int(precomputed_tail_count)
                tail_population = int(precomputed_tail_population)
                tail_mb = float(precomputed_tail_mb)
                if effective_tail_blend >= 1.0:
                    approx_head_np = precomputed_tail_np.astype(np.float32, copy=False)
                else:
                    approx_head_np = (
                        selected_only_np + effective_tail_blend * (precomputed_tail_np - selected_only_np)
                    ).astype(np.float32, copy=False)
            elif effective_tail_blend <= 0.0:
                approx_head_np = selected_only_np.astype(np.float32, copy=False)
                tail_count = 0
                tail_population = max(0, int(context_len) - int(selected_cpu.size))
                attn_seconds = 0.0
                tail_mb = 0.0
            elif str(args.tail_mode) in {"pq_value", "vpq_value", "page_mean"}:
                t0 = time.perf_counter()
                _selected_output_np, final_selected_values_np = selected_output(selected_cpu)
                approx_tail_np, tail_count, tail_population, tail_mb = _compressed_tail_output(
                    index=index,
                    values_np=values_np,
                    scores_np=scores_np,
                    ranked_cpu=ranked_cpu,
                    ranked_scores_cpu=ranked_scores_cpu,
                    selected_cpu=selected_cpu,
                    query_dim=int(trace.head_dim),
                    subbits=int(args.subbits),
                    value_bytes=int(args.value_bytes),
                    mode=str(args.tail_mode),
                    value_subvecs=int(args.value_subvecs),
                    value_subbits=int(args.value_subbits),
                    selected_values_np=final_selected_values_np,
                    selected_scores_override=selected_key_scores_override,
                    tail_score_scale=tail_score_scale,
                    tail_score_bias=tail_score_bias,
                )
                attn_seconds = time.perf_counter() - t0
                if effective_tail_blend >= 1.0:
                    approx_head_np = approx_tail_np
                else:
                    approx_head_np = (selected_only_np + effective_tail_blend * (approx_tail_np - selected_only_np)).astype(
                        np.float32,
                        copy=False,
                    )
            else:
                approx_head, tail_count, tail_population, attn_seconds = selected_plus_tail_output(
                    torch_k_cache[kv_head],
                    torch_v_cache[kv_head],
                    query,
                    selected,
                    ranked_cpu,
                    scores_np,
                    context_len=context_len,
                    samples=int(args.tail_samples),
                    bands=int(args.tail_bands),
                    seed=int(args.tail_seed),
                    qidx=int(qidx),
                    head=int(head),
                    sampling=str(args.tail_sampling),
                )
                approx_tail_np = approx_head.detach().cpu().numpy().astype(np.float32, copy=False)
                if effective_tail_blend >= 1.0:
                    approx_head_np = approx_tail_np
                else:
                    approx_head_np = (selected_only_np + effective_tail_blend * (approx_tail_np - selected_only_np)).astype(
                        np.float32,
                        copy=False,
                    )
                tail_mb = float(tail_count * trace.head_dim * (int(args.key_bytes) + int(args.value_bytes))) / MB
            approx_heads.append(approx_head_np)
            head_metric = _output_error_metrics(dense_head, approx_head_np)
            mass = float(probs_np[selected_cpu].sum()) if selected_cpu.size else 0.0
            exact_key_mb = float(selected_key_exact_tokens * trace.head_dim * int(args.key_bytes)) / MB
            if str(args.selected_value_mode) == "vpq_value":
                final_key = selected_cpu.astype(np.int64, copy=False).tobytes()
                final_cached = selected_value_cache.get(final_key)
                if final_cached is None:
                    selected_only_np, _selected_values_np = selected_output(selected_cpu)
                    final_cached = selected_value_cache.get(final_key)
                final_compressed_v_mb = float(final_cached[2]) if final_cached is not None else 0.0
                final_exact_v_mb = float(final_cached[3]) if final_cached is not None else 0.0
                selected_value_exact_tokens = int(final_cached[4]) if final_cached is not None else 0
                selected_value_exact_selected_mass = float(final_cached[5]) if final_cached is not None else 0.0
                selected_value_mb = final_compressed_v_mb + final_exact_v_mb
                exact_kv_mb = exact_key_mb + final_exact_v_mb
                confidence_selected_value_mb = 0.0
                for key_bytes_, cached in selected_value_cache.items():
                    if key_bytes_ == final_key:
                        continue
                    confidence_selected_value_mb += float(cached[2]) + float(cached[3])
                if online_rule == "geometric_exact_delta":
                    # Exact-delta probes are nested selected prefixes. A real
                    # implementation can retain fetched selected values until
                    # the final prefix decision, so the final selected-value
                    # traffic covers the probe prefixes.
                    confidence_selected_value_mb = 0.0
                confidence_mb += confidence_selected_value_mb
            else:
                selected_value_mb = float(selected_cpu.size * trace.head_dim * int(args.value_bytes)) / MB
                exact_kv_mb = exact_key_mb + selected_value_mb
                confidence_selected_value_mb = 0.0
                selected_value_exact_tokens = int(selected_cpu.size)
                selected_value_exact_selected_mass = 1.0 if selected_cpu.size else 0.0
            online_update_cumulative_mb = float(online_update_mb_by_kv.get(int(kv_head), 0.0))
            online_update_mb_per_token = online_update_cumulative_mb / max(1, int(decode_tokens)) / float(
                kv_fanout.get(int(kv_head), 1)
            )
            step_mb = (
                float(selector_mb)
                + float(rerank_key_mb)
                + confidence_mb
                + exact_key_mb
                + selected_value_mb
                + tail_mb
                + online_update_mb_per_token
            )
            head_rows.append(
                {
                    "decode_length": decode_tokens,
                    "qidx": int(qidx),
                    "position": position,
                    "head": int(head),
                    "kv_head": int(kv_head),
                    "budget": int(budget),
                    "selected_tokens": int(selected_cpu.size),
                    "candidate_tokens": int(ranked_cpu.size),
                    "selector_sparq_rank": int(args.selector_sparq_rank),
                    "quest_rank": int(args.quest_rank),
                    "selector_index_bytes": int(args.selector_index_bytes),
                    "selector_coverage": float(selector_coverage),
                    "rerank_candidates": int(rerank_count),
                    "sparq_rerank_rank": int(args.sparq_rerank_rank),
                    "sparq_rerank_candidates": int(args.sparq_rerank_candidates),
                    "sparq_rerank_count": int(sparq_rerank_count),
                    "sparq_rerank_coverage": float(sparq_rerank_coverage),
                    "sparq_rerank_MB_per_query": float(sparq_rerank_mb),
                    "sparq_audit_rank": int(args.sparq_audit_rank),
                    "sparq_audit_candidates": int(args.sparq_audit_candidates),
                    "sparq_audit_selected": int(len(sparq_audit_tokens)),
                    "sparq_audit_population": int(sparq_audit_population),
                    "sparq_audit_coverage": float(sparq_audit_coverage),
                    "sparq_audit_MB_per_query": float(sparq_audit_mb),
                    "tail_samples": int(tail_count),
                    "tail_population": int(tail_population),
                    "tail_mode": str(args.tail_mode),
                    "long_context_active": bool(long_context_active),
                    "long_context_threshold": int(args.long_context_threshold),
                    "selected_value_mode": str(args.selected_value_mode),
                    "selected_value_exact_rule": str(args.selected_value_exact_rule),
                    "selected_value_exact_top": int(args.selected_value_exact_top),
                    "selected_value_exact_mass": float(args.selected_value_exact_mass),
                    "selected_value_exact_mass_effective": float(effective_selected_value_exact_mass),
                    "long_selected_value_exact_mass": float(args.long_selected_value_exact_mass),
                    "selected_value_exact_risk_mass": float(args.selected_value_exact_risk_mass),
                    "selected_value_min_exact_top": int(args.selected_value_min_exact_top),
                    "selected_value_max_exact_top": int(args.selected_value_max_exact_top),
                    "selected_value_max_exact_top_effective": int(effective_selected_value_max_exact_top),
                    "selected_value_max_exact_top_by_head": str(args.selected_value_max_exact_top_by_head),
                    "long_selected_value_max_exact_top": int(args.long_selected_value_max_exact_top),
                    "long_selected_value_max_exact_top_by_head": str(args.long_selected_value_max_exact_top_by_head),
                    "selected_value_exact_all_context_max": int(args.selected_value_exact_all_context_max),
                    "selected_value_exact_all_fraction_min": float(args.selected_value_exact_all_fraction_min),
                    "selected_value_residual_correction": str(args.selected_value_residual_correction),
                    "selected_value_residual_norm_bytes": int(args.selected_value_residual_norm_bytes),
                    "selected_value_exact_tokens": int(selected_value_exact_tokens),
                    "selected_value_exact_selected_mass": float(selected_value_exact_selected_mass),
                    "selected_key_mode": str(args.selected_key_mode),
                    "selected_key_active": bool(selected_key_active),
                    "selected_key_calibration_probes": int(args.selected_key_calibration_probes),
                    "selected_key_calibration_bands": int(args.selected_key_calibration_bands),
                    "selected_key_exact_selector_mass": float(args.selected_key_exact_selector_mass),
                    "selected_key_min_exact_top": int(args.selected_key_min_exact_top),
                    "selected_key_max_exact_top": int(args.selected_key_max_exact_top),
                    "selected_key_min_context": int(args.selected_key_min_context),
                    "selected_key_exact_tokens": int(selected_key_exact_tokens),
                    "selected_key_compressed_tokens": int(selected_key_compressed_tokens),
                    "selected_key_calibration_probe_count": int(selected_key_calibration_probe_count),
                    "value_subvecs": int(args.value_subvecs),
                    "value_subbits": int(args.value_subbits),
                    "tail_estimator_reused": bool(tail_estimator_reused),
                    "tail_blend": float(max(0.0, min(1.0, effective_tail_blend))),
                    "tail_blend_rule": str(args.tail_blend_rule),
                    "tail_blend_extrap_max": float(args.tail_blend_extrap_max),
                    "tail_off_head": bool(int(head) in tail_off_heads),
                    "online_confidence_rule": online_rule,
                    "confidence_budgets": str(args.confidence_budgets),
                    "rank_budget": int(rank_budget),
                    "proxy_mass_target": float(args.proxy_mass_target),
                    "chosen_proxy_mass": float(chosen_proxy_mass),
                    "chosen_proxy_tail_mass": float(chosen_proxy_tail_mass),
                    "tail_confidence_budget": int(args.tail_confidence_budget),
                    "tail_delta_ratio": float(tail_delta_ratio),
                    "tail_confidence_pass": bool(tail_confidence_pass),
                    "tail_score_calibration": str(args.tail_score_calibration),
                    "tail_score_scale": float(tail_score_scale),
                    "tail_score_bias": float(tail_score_bias),
                    "tail_calibration_tokens": int(tail_calibration_tokens),
                    "tail_pq_relrmse": float(tail_pq_relrmse),
                    "tail_pq_corr": float(tail_pq_corr),
                    "tail_pq_residual_std": float(tail_pq_residual_std),
                    "marginal_exact_mass": float(marginal_exact_mass),
                    "marginal_score_gap": float(marginal_score_gap),
                    "tail_probe_budget": int(args.tail_probe_budget),
                    "tail_probe_rel_l2": float(tail_probe_rel_l2),
                    "stable_tail_probe_rel_l2": float(stable_tail_probe_rel_l2),
                    "slope_forward_rel_l2": float(slope_forward_rel_l2),
                    "slope_backward_rel_l2": float(slope_backward_rel_l2),
                    "slope_ratio": float(slope_ratio),
                    "slope_curvature_rel_l2": float(slope_curvature_rel_l2),
                    "slope_minus_budget": int(slope_minus_budget),
                    "slope_center_budget": int(slope_center_budget),
                    "slope_plus_budget": int(slope_plus_budget),
                    "audit_tail_mode": str(args.audit_tail_mode),
                    "audit_tail_bands": int(args.audit_tail_bands),
                    "audit_tail_mass": float(audit_tail_mass),
                    "audit_tail_logit_gap": float(audit_tail_logit_gap),
                    "audit_tail_count": int(audit_tail_count),
                    "audit_tail_population": int(audit_tail_population),
                    "entropy_ucb_z": float(args.entropy_ucb_z),
                    "entropy_budget_scale": float(args.entropy_budget_scale),
                    "entropy_effective_support": float(entropy_effective_support),
                    "entropy_tail_effective_support": float(entropy_tail_effective_support),
                    "entropy_tail_mass": float(entropy_tail_mass),
                    "entropy_required_budget": int(entropy_required_budget),
                    "geometric_min_budget": int(args.geometric_min_budget),
                    "geometric_max_budget": int(args.geometric_max_budget),
                    "geometric_max_budget_effective": int(effective_geometric_max_budget),
                    "geometric_max_budget_by_head": str(args.geometric_max_budget_by_head),
                    "long_geometric_max_budget": int(args.long_geometric_max_budget),
                    "long_geometric_max_budget_by_head": str(args.long_geometric_max_budget_by_head),
                    "geometric_growth": float(args.geometric_growth),
                    "geometric_probe_scale": float(args.geometric_probe_scale),
                    "stable_tail_probe_rel_l2_max": float(args.stable_tail_probe_rel_l2_max),
                    "slope_forward_rel_l2_max": float(args.slope_forward_rel_l2_max),
                    "slope_backward_rel_l2_max": float(args.slope_backward_rel_l2_max),
                    "slope_ratio_max": float(args.slope_ratio_max),
                    "slope_curvature_rel_l2_max": float(args.slope_curvature_rel_l2_max),
                    "exact_delta_rel_l2": float(exact_delta_rel_l2),
                    "exact_delta_rel_l2_max": float(args.exact_delta_rel_l2_max),
                    "attention_mass": mass,
                    "head_attention_relative_L2": head_metric["output_relative_l2"],
                    "head_attention_cosine": head_metric["output_cosine"],
                    "selector_MB_per_query": float(selector_mb) + float(rerank_key_mb),
                    "pq_selector_MB_per_query": float(selector_mb),
                    "rerank_key_MB_per_query": float(rerank_key_mb),
                    "confidence_MB_per_query": float(confidence_mb),
                    "exact_KV_MB_per_query": exact_kv_mb,
                    "exact_key_MB_per_query": exact_key_mb,
                    "selected_value_MB_per_query": selected_value_mb,
                    "confidence_selected_value_MB_per_query": confidence_selected_value_mb,
                    "tail_estimator_MB_per_query": tail_mb,
                    "online_update_cumulative_MB": online_update_cumulative_mb,
                    "online_update_MB_per_token": online_update_mb_per_token,
                    "online_update_MB_per_query": online_update_mb_per_token,
                    "step_MB_per_query": step_mb,
                    "selector_seconds": float(selector_seconds),
                    "attention_seconds": float(attn_seconds),
                    "nprobe": int(chosen_nprobe),
                }
            )

        for row in head_rows:
            row = dict(row)
            if str(args.online_confidence_rule) != "none":
                row["algorithm"] = (
                    f"{args.selector_mode}_paged_pq_{args.online_confidence_rule}"
                    f"_rank{int(row.get('rank_budget', default_budget))}+{args.tail_mode}_tail"
                )
            elif budget_by_head:
                row["algorithm"] = f"{args.selector_mode}_paged_pq_head_budget+{args.tail_mode}_tail"
            else:
                row["algorithm"] = f"{args.selector_mode}_paged_pq_k{int(row.get('budget', default_budget))}+{args.tail_mode}_tail"
            row["tail_sampling"] = str(args.tail_sampling)
            row["tail_mode"] = str(args.tail_mode)
            row["selector_mode"] = str(args.selector_mode)
            if str(args.selected_key_mode) != "exact":
                row["algorithm"] = f"{row['algorithm']}+{args.selected_key_mode}_k"
            row["budget"] = int(row.get("budget", default_budget))
            row["budget_by_head_enabled"] = bool(budget_by_head)
            row["tail_samples_requested"] = int(args.tail_samples)
            per_head_rows.append(row)

        dense_concat_np = np.concatenate(dense_heads, axis=0).astype(np.float32, copy=False)
        approx_concat_np = np.concatenate(approx_heads, axis=0).astype(np.float32, copy=False)
        dense_concat = torch.as_tensor(dense_concat_np, dtype=torch.float32, device=device)
        approx_concat = torch.as_tensor(approx_concat_np, dtype=torch.float32, device=device)
        layer_input = torch.as_tensor(np.asarray(layer_inputs[position], dtype=np.float32), dtype=torch.float32, device=device)

        concat_metrics = _output_error_metrics(dense_concat_np, approx_concat_np)
        if bool(args.head_only):
            proj_metrics = _nan_output_metrics()
            post_attn_metrics = _nan_output_metrics()
            layer_metrics = _nan_output_metrics()
        else:
            dense_attn_proj = F.linear(dense_concat, wo)
            approx_attn_proj = F.linear(approx_concat, wo)
            dense_post_attn = layer_input + dense_attn_proj
            approx_post_attn = layer_input + approx_attn_proj
            dense_layer_out = dense_post_attn + mlp(rmsnorm(dense_post_attn, post_ln, norm_eps), gate_proj, up_proj, down_proj)
            approx_layer_out = approx_post_attn + mlp(rmsnorm(approx_post_attn, post_ln, norm_eps), gate_proj, up_proj, down_proj)
            torch.cuda.synchronize() if device.type == "cuda" else None
            proj_metrics = _output_error_metrics(
                dense_attn_proj.detach().cpu().numpy().astype(np.float32, copy=False),
                approx_attn_proj.detach().cpu().numpy().astype(np.float32, copy=False),
            )
            post_attn_metrics = _output_error_metrics(
                dense_post_attn.detach().cpu().numpy().astype(np.float32, copy=False),
                approx_post_attn.detach().cpu().numpy().astype(np.float32, copy=False),
            )
            layer_metrics = _output_error_metrics(
                dense_layer_out.detach().cpu().numpy().astype(np.float32, copy=False),
                approx_layer_out.detach().cpu().numpy().astype(np.float32, copy=False),
            )
        elapsed = time.perf_counter() - q_start
        if str(args.online_confidence_rule) != "none":
            max_rank_budget = max([int(r.get("rank_budget", default_budget)) for r in head_rows] or [default_budget])
            algorithm = f"{args.selector_mode}_paged_pq_{args.online_confidence_rule}_rank{max_rank_budget}+{args.tail_mode}_tail"
        elif budget_by_head:
            algorithm = f"{args.selector_mode}_paged_pq_head_budget+{args.tail_mode}_tail"
        else:
            algorithm = f"{args.selector_mode}_paged_pq_k{default_budget}+{args.tail_mode}_tail"
        if str(args.selected_key_mode) != "exact":
            algorithm = f"{algorithm}+{args.selected_key_mode}_k"
        common = {
            "algorithm": algorithm,
            "decode_length": decode_tokens,
            "qidx": int(qidx),
            "position": position,
            "context_len": context_len,
            "budget": default_budget,
            "budget_by_head_enabled": bool(budget_by_head),
            "budget_by_head": str(args.budget_by_head),
            "online_confidence_rule": str(args.online_confidence_rule),
            "confidence_budgets": str(args.confidence_budgets),
            "proxy_mass_target": float(args.proxy_mass_target),
            "tail_confidence_budget": int(args.tail_confidence_budget),
            "tail_delta_min": float(args.tail_delta_min),
            "tail_delta_max": float(args.tail_delta_max),
            "tail_proxy_mass_min": float(args.tail_proxy_mass_min),
            "tail_proxy_mass_max": float(args.tail_proxy_mass_max),
            "tail_score_calibration": str(args.tail_score_calibration),
            "tail_pq_corr_min": float(args.tail_pq_corr_min),
            "tail_pq_relrmse_max": float(args.tail_pq_relrmse_max),
            "marginal_mass_max": float(args.marginal_mass_max),
            "marginal_score_gap_max": float(args.marginal_score_gap_max),
            "marginal_min_budget": int(args.marginal_min_budget),
            "tail_probe_budget": int(args.tail_probe_budget),
            "tail_probe_rel_l2_max": float(args.tail_probe_rel_l2_max),
            "entropy_ucb_z": float(args.entropy_ucb_z),
            "entropy_budget_scale": float(args.entropy_budget_scale),
            "entropy_probe_scale": float(args.entropy_probe_scale),
            "entropy_min_budget": int(args.entropy_min_budget),
            "entropy_max_budget": int(args.entropy_max_budget),
            "entropy_budget_granularity": int(args.entropy_budget_granularity),
            "entropy_tail_mass_max": float(args.entropy_tail_mass_max),
            "geometric_min_budget": int(args.geometric_min_budget),
            "geometric_max_budget": int(args.geometric_max_budget),
            "long_context_threshold": int(args.long_context_threshold),
            "long_geometric_max_budget": int(args.long_geometric_max_budget),
            "long_geometric_max_budget_by_head": str(args.long_geometric_max_budget_by_head),
            "geometric_growth": float(args.geometric_growth),
            "geometric_probe_scale": float(args.geometric_probe_scale),
            "geometric_budget_granularity": int(args.geometric_budget_granularity),
            "stable_tail_probe_rel_l2_max": float(args.stable_tail_probe_rel_l2_max),
            "slope_forward_rel_l2_max": float(args.slope_forward_rel_l2_max),
            "slope_backward_rel_l2_max": float(args.slope_backward_rel_l2_max),
            "slope_ratio_max": float(args.slope_ratio_max),
            "slope_curvature_rel_l2_max": float(args.slope_curvature_rel_l2_max),
            "exact_delta_rel_l2_max": float(args.exact_delta_rel_l2_max),
            "audit_tail_samples": int(args.audit_tail_samples),
            "audit_tail_mode": str(args.audit_tail_mode),
            "audit_tail_bands": int(args.audit_tail_bands),
            "audit_tail_mass_max": float(args.audit_tail_mass_max),
            "audit_tail_logit_gap_max": float(args.audit_tail_logit_gap_max),
            "tail_samples_requested": int(args.tail_samples),
            "tail_sampling": str(args.tail_sampling),
            "tail_mode": str(args.tail_mode),
            "selected_value_mode": str(args.selected_value_mode),
            "selected_value_exact_rule": str(args.selected_value_exact_rule),
            "selected_value_exact_top": int(args.selected_value_exact_top),
            "selected_value_exact_mass": float(args.selected_value_exact_mass),
            "long_selected_value_exact_mass": float(args.long_selected_value_exact_mass),
            "selected_value_exact_risk_mass": float(args.selected_value_exact_risk_mass),
            "selected_value_min_exact_top": int(args.selected_value_min_exact_top),
            "selected_value_max_exact_top": int(args.selected_value_max_exact_top),
            "long_selected_value_max_exact_top": int(args.long_selected_value_max_exact_top),
            "long_selected_value_max_exact_top_by_head": str(args.long_selected_value_max_exact_top_by_head),
            "selected_value_exact_all_context_max": int(args.selected_value_exact_all_context_max),
            "selected_value_exact_all_fraction_min": float(args.selected_value_exact_all_fraction_min),
            "selected_value_residual_correction": str(args.selected_value_residual_correction),
            "selected_value_residual_norm_bytes": int(args.selected_value_residual_norm_bytes),
            "selected_key_mode": str(args.selected_key_mode),
            "selected_key_calibration_probes": int(args.selected_key_calibration_probes),
            "selected_key_calibration_bands": int(args.selected_key_calibration_bands),
            "selected_key_exact_selector_mass": float(args.selected_key_exact_selector_mass),
            "selected_key_min_exact_top": int(args.selected_key_min_exact_top),
            "selected_key_max_exact_top": int(args.selected_key_max_exact_top),
            "selected_key_min_context": int(args.selected_key_min_context),
            "value_subvecs": int(args.value_subvecs),
            "value_subbits": int(args.value_subbits),
            "tail_blend": float(max(0.0, min(1.0, float(args.tail_blend)))),
            "tail_blend_rule": str(args.tail_blend_rule),
            "tail_blend_extrap_max": float(args.tail_blend_extrap_max),
            "tail_off_heads": str(args.tail_off_heads),
            "selector_mode": str(args.selector_mode),
            "selector_sparq_rank": int(args.selector_sparq_rank),
            "quest_rank": int(args.quest_rank),
            "selector_index_bytes": int(args.selector_index_bytes),
            "rerank_candidates": int(args.rerank_candidates),
            "sparq_rerank_rank": int(args.sparq_rerank_rank),
            "sparq_rerank_candidates": int(args.sparq_rerank_candidates),
            "sparq_audit_rank": int(args.sparq_audit_rank),
            "sparq_audit_candidates": int(args.sparq_audit_candidates),
            "layer_idx": layer_idx,
            "query_seconds": float(elapsed),
            "mean_head_attention_mass": float(np.mean([r["attention_mass"] for r in head_rows])),
            "min_head_attention_mass": float(np.min([r["attention_mass"] for r in head_rows])),
            "mean_head_attention_relative_L2": float(np.mean([r["head_attention_relative_L2"] for r in head_rows])),
            "max_head_attention_relative_L2": float(np.max([r["head_attention_relative_L2"] for r in head_rows])),
            "mean_selector_coverage": float(np.mean([r["selector_coverage"] for r in head_rows])),
            "mean_confidence_MB_per_head": float(np.mean([r["confidence_MB_per_query"] for r in head_rows])),
            "mean_sparq_rerank_MB_per_head": float(np.mean([r["sparq_rerank_MB_per_query"] for r in head_rows])),
            "mean_sparq_rerank_count": float(np.mean([r["sparq_rerank_count"] for r in head_rows])),
            "mean_sparq_rerank_coverage": float(np.mean([r["sparq_rerank_coverage"] for r in head_rows])),
            "mean_sparq_audit_MB_per_head": float(np.mean([r["sparq_audit_MB_per_query"] for r in head_rows])),
            "mean_sparq_audit_selected": float(np.mean([r["sparq_audit_selected"] for r in head_rows])),
            "mean_sparq_audit_coverage": float(np.mean([r["sparq_audit_coverage"] for r in head_rows])),
            "mean_exact_KV_MB_per_head": float(np.mean([r["exact_KV_MB_per_query"] for r in head_rows])),
            "mean_exact_key_MB_per_head": float(np.mean([r["exact_key_MB_per_query"] for r in head_rows])),
            "mean_selected_value_MB_per_head": float(np.mean([r["selected_value_MB_per_query"] for r in head_rows])),
            "mean_selected_value_exact_tokens": float(np.mean([r["selected_value_exact_tokens"] for r in head_rows])),
            "mean_selected_value_exact_selected_mass": float(
                np.mean([r["selected_value_exact_selected_mass"] for r in head_rows])
            ),
            "mean_selected_key_exact_tokens": float(np.mean([r["selected_key_exact_tokens"] for r in head_rows])),
            "mean_selected_key_compressed_tokens": float(np.mean([r["selected_key_compressed_tokens"] for r in head_rows])),
            "mean_selected_key_calibration_probe_count": float(
                np.mean([r["selected_key_calibration_probe_count"] for r in head_rows])
            ),
            "mean_confidence_selected_value_MB_per_head": float(
                np.mean([r["confidence_selected_value_MB_per_query"] for r in head_rows])
            ),
            "mean_tail_estimator_MB_per_head": float(np.mean([r["tail_estimator_MB_per_query"] for r in head_rows])),
            "mean_online_update_MB_per_token_per_head": float(
                np.mean([r["online_update_MB_per_token"] for r in head_rows])
            ),
            "mean_online_update_cumulative_MB_per_kv_head": float(
                np.mean([r["online_update_cumulative_MB"] for r in head_rows])
            ),
            "mean_effective_tail_blend": float(np.mean([r["tail_blend"] for r in head_rows])),
            "tail_confidence_pass_heads": int(np.sum([bool(r["tail_confidence_pass"]) for r in head_rows])),
            "mean_tail_delta_ratio": float(np.mean([r["tail_delta_ratio"] for r in head_rows])),
            "mean_chosen_proxy_mass": float(np.mean([r["chosen_proxy_mass"] for r in head_rows])),
            "mean_chosen_proxy_tail_mass": float(np.mean([r["chosen_proxy_tail_mass"] for r in head_rows])),
            "mean_tail_pq_relrmse": float(np.mean([r["tail_pq_relrmse"] for r in head_rows if np.isfinite(float(r["tail_pq_relrmse"]))])) if any(np.isfinite(float(r["tail_pq_relrmse"])) for r in head_rows) else float("inf"),
            "mean_tail_pq_corr": float(np.mean([r["tail_pq_corr"] for r in head_rows])),
            "mean_tail_pq_residual_std": float(np.mean([r["tail_pq_residual_std"] for r in head_rows])),
            "mean_marginal_exact_mass": float(np.mean([r["marginal_exact_mass"] for r in head_rows])),
            "max_marginal_exact_mass": float(np.max([r["marginal_exact_mass"] for r in head_rows])),
            "mean_tail_probe_rel_l2": float(np.mean([r["tail_probe_rel_l2"] for r in head_rows if np.isfinite(float(r["tail_probe_rel_l2"]))])) if any(np.isfinite(float(r["tail_probe_rel_l2"])) for r in head_rows) else float("inf"),
            "mean_stable_tail_probe_rel_l2": float(np.mean([r["stable_tail_probe_rel_l2"] for r in head_rows if np.isfinite(float(r["stable_tail_probe_rel_l2"]))])) if any(np.isfinite(float(r["stable_tail_probe_rel_l2"])) for r in head_rows) else float("inf"),
            "mean_slope_forward_rel_l2": float(np.mean([r["slope_forward_rel_l2"] for r in head_rows if np.isfinite(float(r["slope_forward_rel_l2"]))])) if any(np.isfinite(float(r["slope_forward_rel_l2"])) for r in head_rows) else float("inf"),
            "max_slope_forward_rel_l2": float(np.max([r["slope_forward_rel_l2"] for r in head_rows if np.isfinite(float(r["slope_forward_rel_l2"]))])) if any(np.isfinite(float(r["slope_forward_rel_l2"])) for r in head_rows) else float("inf"),
            "mean_slope_backward_rel_l2": float(np.mean([r["slope_backward_rel_l2"] for r in head_rows if np.isfinite(float(r["slope_backward_rel_l2"]))])) if any(np.isfinite(float(r["slope_backward_rel_l2"])) for r in head_rows) else float("inf"),
            "mean_slope_ratio": float(np.mean([r["slope_ratio"] for r in head_rows if np.isfinite(float(r["slope_ratio"]))])) if any(np.isfinite(float(r["slope_ratio"])) for r in head_rows) else float("inf"),
            "max_slope_ratio": float(np.max([r["slope_ratio"] for r in head_rows if np.isfinite(float(r["slope_ratio"]))])) if any(np.isfinite(float(r["slope_ratio"])) for r in head_rows) else float("inf"),
            "mean_slope_curvature_rel_l2": float(np.mean([r["slope_curvature_rel_l2"] for r in head_rows if np.isfinite(float(r["slope_curvature_rel_l2"]))])) if any(np.isfinite(float(r["slope_curvature_rel_l2"])) for r in head_rows) else float("inf"),
            "max_slope_curvature_rel_l2": float(np.max([r["slope_curvature_rel_l2"] for r in head_rows if np.isfinite(float(r["slope_curvature_rel_l2"]))])) if any(np.isfinite(float(r["slope_curvature_rel_l2"])) for r in head_rows) else float("inf"),
            "mean_slope_plus_budget": float(np.mean([r["slope_plus_budget"] for r in head_rows])),
            "mean_audit_tail_mass": float(np.mean([r["audit_tail_mass"] for r in head_rows])),
            "max_audit_tail_mass": float(np.max([r["audit_tail_mass"] for r in head_rows])),
            "mean_audit_tail_count": float(np.mean([r["audit_tail_count"] for r in head_rows])),
            "mean_entropy_effective_support": float(np.mean([r["entropy_effective_support"] for r in head_rows])),
            "mean_entropy_tail_mass": float(np.mean([r["entropy_tail_mass"] for r in head_rows])),
            "mean_entropy_required_budget": float(np.mean([r["entropy_required_budget"] for r in head_rows])),
            "mean_exact_delta_rel_l2": float(np.mean([r["exact_delta_rel_l2"] for r in head_rows if np.isfinite(float(r["exact_delta_rel_l2"]))])) if any(np.isfinite(float(r["exact_delta_rel_l2"])) for r in head_rows) else float("inf"),
            "max_exact_delta_rel_l2": float(np.max([r["exact_delta_rel_l2"] for r in head_rows if np.isfinite(float(r["exact_delta_rel_l2"]))])) if any(np.isfinite(float(r["exact_delta_rel_l2"])) for r in head_rows) else float("inf"),
            "mean_step_MB_per_head": float(np.mean([r["step_MB_per_query"] for r in head_rows])),
            "max_step_MB_per_head": float(np.max([r["step_MB_per_query"] for r in head_rows])),
        }
        for prefix_name, metrics in [
            ("attn_concat", concat_metrics),
            ("attn_o_proj", proj_metrics),
            ("post_attn_residual", post_attn_metrics),
            ("layer_output", layer_metrics),
        ]:
            for key, value in metrics.items():
                common[f"{prefix_name}_{key}"] = float(value)
        rows.append(common)

    with (out_dir / "layer_quality.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for row in rows for k in row}))
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "layer_quality.json").write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    if per_head_rows:
        with (out_dir / "per_head_quality.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=sorted({k for row in per_head_rows for k in row}))
            writer.writeheader()
            writer.writerows(per_head_rows)
        (out_dir / "per_head_quality.json").write_text(
            json.dumps(per_head_rows, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    summary = {}
    if rows:
        for key in rows[0]:
            if key in {
                "algorithm",
                "tail_sampling",
                "tail_mode",
                "selector_mode",
                "online_confidence_rule",
                "tail_score_calibration",
                "tail_blend_rule",
                "selected_value_mode",
                "selected_value_exact_rule",
                "selected_value_residual_correction",
                "selected_key_mode",
            }:
                summary[key] = rows[0][key]
        for metric in [
            "attn_concat_output_relative_l2",
            "attn_o_proj_output_relative_l2",
            "post_attn_residual_output_relative_l2",
            "layer_output_output_relative_l2",
            "attn_concat_output_cosine",
            "attn_o_proj_output_cosine",
            "post_attn_residual_output_cosine",
            "layer_output_output_cosine",
            "mean_head_attention_mass",
            "min_head_attention_mass",
            "mean_confidence_MB_per_head",
            "mean_sparq_rerank_MB_per_head",
            "mean_sparq_rerank_count",
            "mean_sparq_rerank_coverage",
            "mean_sparq_audit_MB_per_head",
            "mean_sparq_audit_selected",
            "mean_sparq_audit_coverage",
            "mean_exact_KV_MB_per_head",
            "mean_exact_key_MB_per_head",
            "mean_selected_value_MB_per_head",
            "mean_selected_value_exact_tokens",
            "mean_selected_value_exact_selected_mass",
            "mean_selected_key_exact_tokens",
            "mean_selected_key_compressed_tokens",
            "mean_selected_key_calibration_probe_count",
            "mean_confidence_selected_value_MB_per_head",
            "mean_tail_estimator_MB_per_head",
            "mean_online_update_MB_per_token_per_head",
            "mean_online_update_cumulative_MB_per_kv_head",
            "mean_effective_tail_blend",
            "tail_confidence_pass_heads",
            "mean_tail_delta_ratio",
            "mean_chosen_proxy_mass",
            "mean_chosen_proxy_tail_mass",
            "mean_tail_pq_corr",
            "mean_tail_pq_residual_std",
            "mean_marginal_exact_mass",
            "max_marginal_exact_mass",
            "mean_tail_probe_rel_l2",
            "mean_stable_tail_probe_rel_l2",
            "mean_slope_forward_rel_l2",
            "max_slope_forward_rel_l2",
            "mean_slope_backward_rel_l2",
            "mean_slope_ratio",
            "max_slope_ratio",
            "mean_slope_curvature_rel_l2",
            "max_slope_curvature_rel_l2",
            "mean_slope_plus_budget",
            "mean_audit_tail_mass",
            "max_audit_tail_mass",
            "mean_audit_tail_count",
            "mean_entropy_effective_support",
            "mean_entropy_tail_mass",
            "mean_entropy_required_budget",
            "mean_exact_delta_rel_l2",
            "max_exact_delta_rel_l2",
            "mean_step_MB_per_head",
            "max_step_MB_per_head",
            "query_seconds",
        ]:
            vals = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
            summary[f"{metric}_mean"] = float(vals.mean())
            summary[f"{metric}_max"] = float(vals.max())
            summary[f"{metric}_min"] = float(vals.min())
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[layer_quality_eval] wrote {out_dir}")


if __name__ == "__main__":
    run()
