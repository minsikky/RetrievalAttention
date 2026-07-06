#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
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
)
from benchmark.selector_eval.metrics.attention import _output_error_metrics, attention_distribution_error_metrics
from benchmark.selector_eval.runners.run_value_exact_strategy_eval import (
    dense_attention_output,
    mixed_scores,
    project_head_subset,
    top_mask,
    value_vpq_code_stat_risk,
)
from benchmark.selector_eval.runners.run_layer_quality_eval import _selected_for_budget, _vpq_values_for_tokens
from benchmark.selector_eval.runners.run_layer_quality_eval import _rank_quest_pages, _rank_quest_pq


MB = 1024.0 * 1024.0

# Lookahead-bound diagnostic variants: `charge_all` models naive hardware that
# pays exact reads for every confidence lookahead; `cs` is a strict
# Cauchy-Schwarz bound on the K-logit upgrade error; `rms<lambda>` are
# calibrated typical-case estimates (|q.r| ~ ||q||*||r||/sqrt(d) scaled by
# lambda) with a shared strict L1 V-band bound. `var<lambda>` are
# second-moment concentration estimates, sqrt(sum(p^2 * err^2)) * lambda:
# L1-style bounds scale with band token count while the true delta cancels
# like sqrt(n), so at long contexts only the variance form can certify.
LOOKAHEAD_VARIANTS = ("charge_all", "cs", "rms1", "rms2", "rms4", "var1", "var2", "var4")


def _lookahead_x_factors(*, q_norm: float, k_resid_norm: np.ndarray, head_dim: int) -> dict[str, np.ndarray]:
    sqrt_d = math.sqrt(max(1.0, float(head_dim)))
    base = float(q_norm) * np.asarray(k_resid_norm, dtype=np.float64)
    return {
        "cs": base / sqrt_d,
        "rms1": 1.0 * base / float(head_dim),
        "rms2": 2.0 * base / float(head_dim),
        "rms4": 4.0 * base / float(head_dim),
    }


def parse_csv_floats(text: str) -> list[float]:
    return [float(part.strip()) for part in str(text).split(",") if part.strip()]


def parse_csv_ratios(text: str) -> list[float]:
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
    return ratios


def budgets_from_fracs(context_len: int, fracs: list[float]) -> list[int]:
    budgets = {
        max(1, min(int(context_len), int(math.ceil(float(context_len) * float(frac)))))
        for frac in fracs
        if float(frac) > 0.0
    }
    if not budgets:
        raise ValueError("relative budget fractions produced no positive budgets")
    return sorted(budgets)


def load_weight_index(model_dir: Path) -> dict[str, str]:
    index_path = model_dir / "model.safetensors.index.json"
    data = json.loads(index_path.read_text())
    return {str(k): str(v) for k, v in data["weight_map"].items()}


def load_safetensor_weight(model_dir: Path, weight_map: dict[str, str], name: str, device: torch.device) -> torch.Tensor:
    shard = model_dir / weight_map[name]
    with safe_open(shard, framework="pt", device="cpu") as f:
        return f.get_tensor(name).to(device=device, dtype=torch.float32, non_blocking=True)


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    aa = a.astype(np.float64, copy=False)
    bb = b.astype(np.float64, copy=False)
    return float(np.linalg.norm(aa - bb)) / max(float(np.linalg.norm(bb)), 1e-20)


def parse_csv_names(text: str) -> list[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def _as_1d_numpy(x, *, dtype: np.dtype) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=dtype).reshape(-1)


def _parse_variant_count(variant: str, prefix: str, default: int) -> int:
    match = re.search(rf"(?:^|_){re.escape(prefix)}(\d+)", str(variant))
    return int(match.group(1)) if match else int(default)


def _parse_variant_float(variant: str, prefix: str, default: float) -> float:
    match = re.search(rf"(?:^|_){re.escape(prefix)}([0-9p.]+)", str(variant))
    if not match:
        return float(default)
    return float(match.group(1).replace("p", "."))


def _page_full_reconstruct(page) -> np.ndarray:
    codebooks = page.codebooks.detach().to(device="cpu", dtype=torch.float32).numpy()
    codes = page.codes.detach().to(device="cpu", dtype=torch.long).numpy().astype(np.int64, copy=False)
    subvecs = int(codebooks.shape[0])
    subdim = int(codebooks.shape[-1])
    out = np.empty((int(codes.shape[0]), subvecs * subdim), dtype=np.float32)
    for sub in range(subvecs):
        sub_codes = np.asarray(codes[:, sub], dtype=np.intp)
        out[:, sub * subdim : (sub + 1) * subdim] = np.take(codebooks[sub], sub_codes, axis=0)
    return out


def _page_dim_reconstruct(page, dims: np.ndarray) -> np.ndarray:
    codebooks = page.codebooks.detach().to(device="cpu", dtype=torch.float32).numpy()
    codes = page.codes.detach().to(device="cpu", dtype=torch.long).numpy().astype(np.int64, copy=False)
    subvecs = int(codebooks.shape[0])
    subdim = int(codebooks.shape[-1])
    out = np.empty((int(codes.shape[0]), int(dims.size)), dtype=np.float32)
    for out_col, dim in enumerate(dims.astype(np.int64, copy=False).tolist()):
        sub = int(dim) // subdim
        off = int(dim) % subdim
        if sub < 0 or sub >= subvecs:
            raise ValueError(f"dimension {dim} is outside page PQ shape")
        sub_codes = np.asarray(codes[:, sub], dtype=np.intp)
        out[:, out_col] = np.take(codebooks[sub, :, off], sub_codes, axis=0)
    return out


def _rank_scores_from_context_scores(
    *,
    ranked_cpu: np.ndarray,
    scores_by_token: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    ranked_cpu = _as_1d_numpy(ranked_cpu, dtype=np.int64)
    scores = scores_by_token[ranked_cpu].astype(np.float32, copy=False)
    order = np.argsort(-scores.astype(np.float64, copy=False), kind="stable")
    return ranked_cpu[order].astype(np.int64, copy=False), scores[order].astype(np.float32, copy=False)


def _apply_sparq_channel_correction(
    *,
    index,
    keys_np: np.ndarray,
    query_np: np.ndarray,
    ranked_cpu: np.ndarray,
    ranked_scores_cpu: np.ndarray,
    rank: int,
    key_bytes: int,
) -> tuple[np.ndarray, np.ndarray, float, dict[str, float | int | str]]:
    ranked_cpu = _as_1d_numpy(ranked_cpu, dtype=np.int64)
    ranked_scores_cpu = _as_1d_numpy(ranked_scores_cpu, dtype=np.float32)
    rank = min(max(0, int(rank)), int(query_np.shape[0]))
    if rank <= 0 or ranked_cpu.size == 0:
        return ranked_cpu, ranked_scores_cpu, 0.0, {"score_proxy_detail": "sparq_channel_r0"}
    dims = np.argsort(-np.abs(query_np).astype(np.float64, copy=False), kind="stable")[:rank].astype(np.int64, copy=False)
    scores_by_token = np.full((int(keys_np.shape[0]),), np.nan, dtype=np.float64)
    scores_by_token[ranked_cpu] = ranked_scores_cpu.astype(np.float64, copy=False)
    for page in index.pages:
        start = int(page.start)
        size = int(page.size)
        tokens = start + np.arange(size, dtype=np.int64)
        pq_dims = _page_dim_reconstruct(page, dims).astype(np.float64, copy=False)
        exact_dims = keys_np[tokens[:, None], dims[None, :]].astype(np.float64, copy=False)
        delta = (exact_dims - pq_dims) @ query_np[dims].astype(np.float64, copy=False)
        scores_by_token[tokens] += delta
    extra_mb = float(ranked_cpu.size * rank * int(key_bytes)) / MB
    new_ranked, new_scores = _rank_scores_from_context_scores(ranked_cpu=ranked_cpu, scores_by_token=scores_by_token)
    return new_ranked, new_scores, extra_mb, {"score_proxy_detail": f"sparq_channel_r{rank}", "sparq_rank": int(rank)}


def _quantize_matrix_per_column(x: np.ndarray, bits: int) -> tuple[np.ndarray, float]:
    if x.size == 0:
        return x.astype(np.float32, copy=True), 0.0
    if int(bits) >= 16:
        return x.astype(np.float32, copy=True), 0.0
    x32 = x.astype(np.float32, copy=False)
    lo = np.min(x32, axis=0, keepdims=True)
    hi = np.max(x32, axis=0, keepdims=True)
    levels = float((1 << int(bits)) - 1)
    scale = np.maximum((hi - lo) / max(levels, 1.0), np.float32(1e-8))
    codes = np.clip(np.rint((x32 - lo) / scale), 0.0, levels)
    return (codes * scale + lo).astype(np.float32, copy=False), float(codes.size * int(bits)) / 8.0


def _apply_promoted_residual_correction(
    *,
    index,
    keys_np: np.ndarray,
    query_np: np.ndarray,
    ranked_cpu: np.ndarray,
    ranked_scores_cpu: np.ndarray,
    promote_ratio: float,
    promote_bits: int,
    key_bytes: int,
    metadata_bytes: int,
) -> tuple[np.ndarray, np.ndarray, float, dict[str, float | int | str]]:
    ranked_cpu = _as_1d_numpy(ranked_cpu, dtype=np.int64)
    ranked_scores_cpu = _as_1d_numpy(ranked_scores_cpu, dtype=np.float32)
    dim = int(query_np.shape[0])
    promote_count = max(0, min(dim, int(round(dim * float(promote_ratio)))))
    if promote_count <= 0 or ranked_cpu.size == 0:
        return ranked_cpu, ranked_scores_cpu, 0.0, {"score_proxy_detail": "promoted_residual_p0"}
    scores_by_token = np.full((int(keys_np.shape[0]),), np.nan, dtype=np.float64)
    scores_by_token[ranked_cpu] = ranked_scores_cpu.astype(np.float64, copy=False)
    extra_bytes = 0.0
    promoted_total = 0
    for page in index.pages:
        start = int(page.start)
        size = int(page.size)
        tokens = start + np.arange(size, dtype=np.int64)
        khat = _page_full_reconstruct(page)
        residual = keys_np[tokens].astype(np.float32, copy=False) - khat
        channel_score = np.mean(residual.astype(np.float64, copy=False) * residual.astype(np.float64, copy=False), axis=0)
        dims = np.argpartition(channel_score, -promote_count)[-promote_count:].astype(np.int64, copy=False)
        dims.sort()
        residual_hat, code_bytes = _quantize_matrix_per_column(residual[:, dims], int(promote_bits))
        scores_by_token[tokens] += residual_hat.astype(np.float64, copy=False) @ query_np[dims].astype(np.float64, copy=False)
        if int(promote_bits) >= 16:
            extra_bytes += float(size * promote_count * int(key_bytes))
        else:
            extra_bytes += float(code_bytes)
            extra_bytes += float(promote_count * 2 * int(metadata_bytes))
        promoted_total += int(promote_count)
    new_ranked, new_scores = _rank_scores_from_context_scores(ranked_cpu=ranked_cpu, scores_by_token=scores_by_token)
    return (
        new_ranked,
        new_scores,
        float(extra_bytes) / MB,
        {
            "score_proxy_detail": f"promoted_residual_p{float(promote_ratio):g}_b{int(promote_bits)}",
            "promote_ratio": float(promote_ratio),
            "promote_bits": int(promote_bits),
            "promoted_dims_per_page": int(promote_count),
            "promoted_dims_total": int(promoted_total),
        },
    )


def _pq_scores_np(query_np: np.ndarray, codebooks: np.ndarray, codes: np.ndarray) -> np.ndarray:
    subvecs = int(codebooks.shape[0])
    subdim = int(codebooks.shape[-1])
    q_parts = query_np.astype(np.float32, copy=False).reshape(subvecs, subdim)
    out = np.zeros((int(codes.shape[0]),), dtype=np.float32)
    for sub in range(subvecs):
        lut = np.sum(codebooks[sub] * q_parts[sub][None, :], axis=1)
        out += np.take(lut, np.asarray(codes[:, sub], dtype=np.intp), axis=0)
    return out


def _apply_residual_pq_correction(
    *,
    index,
    keys_np: np.ndarray,
    query_np: np.ndarray,
    ranked_cpu: np.ndarray,
    ranked_scores_cpu: np.ndarray,
    stages: int,
    subbits: int,
    subvecs: int,
    kmeans_iters: int,
    key_bytes: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float, dict[str, float | int | str]]:
    ranked_cpu = _as_1d_numpy(ranked_cpu, dtype=np.int64)
    ranked_scores_cpu = _as_1d_numpy(ranked_scores_cpu, dtype=np.float32)
    stages = max(1, int(stages))
    subbits = max(1, int(subbits))
    dim = int(query_np.shape[0])
    subvecs = max(1, min(int(subvecs), dim))
    if dim % subvecs != 0:
        subvecs = 1
    scores_by_token = np.full((int(keys_np.shape[0]),), np.nan, dtype=np.float64)
    scores_by_token[ranked_cpu] = ranked_scores_cpu.astype(np.float64, copy=False)
    extra_bytes = 0.0
    code_bytes = 1 if int(subbits) <= 8 else 2
    for page_id, page in enumerate(index.pages):
        start = int(page.start)
        size = int(page.size)
        tokens = start + np.arange(size, dtype=np.int64)
        residual = keys_np[tokens].astype(np.float32, copy=False) - _page_full_reconstruct(page)
        for stage in range(stages):
            codebooks, codes, actual_subvecs, _centroids = build_page_residual_pq(
                residual,
                subvecs=subvecs,
                subbits=subbits,
                seed=int(seed) + 65537 * int(stage) + 4099 * int(page_id),
                max_iter=int(kmeans_iters),
            )
            scores_by_token[tokens] += _pq_scores_np(query_np, codebooks, codes).astype(np.float64, copy=False)
            residual -= _reconstruct_pq_np(codebooks, codes)
            extra_bytes += float(codebooks.size * int(key_bytes) + codes.size * code_bytes)
            if int(actual_subvecs) != int(subvecs):
                subvecs = int(actual_subvecs)
    new_ranked, new_scores = _rank_scores_from_context_scores(ranked_cpu=ranked_cpu, scores_by_token=scores_by_token)
    return (
        new_ranked,
        new_scores,
        float(extra_bytes) / MB,
        {
            "score_proxy_detail": f"residual_pq_m{stages}b{subbits}_s{subvecs}",
            "residual_pq_stages": int(stages),
            "residual_pq_subbits": int(subbits),
            "residual_pq_subvecs": int(subvecs),
        },
    )


def _reconstruct_pq_np(codebooks: np.ndarray, codes: np.ndarray) -> np.ndarray:
    subvecs = int(codebooks.shape[0])
    subdim = int(codebooks.shape[-1])
    out = np.empty((int(codes.shape[0]), subvecs * subdim), dtype=np.float32)
    for sub in range(subvecs):
        sub_codes = np.asarray(codes[:, sub], dtype=np.intp)
        out[:, sub * subdim : (sub + 1) * subdim] = np.take(codebooks[sub], sub_codes, axis=0)
    return out


def build_page_residual_pq(
    residual: np.ndarray,
    *,
    subvecs: int,
    subbits: int,
    seed: int,
    max_iter: int,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    from benchmark.attention_efficiency_threeway_eval import build_pq_index

    return build_pq_index(
        residual.astype(np.float32, copy=False),
        0,
        int(residual.shape[0]),
        subvecs=int(subvecs),
        subbits=int(subbits),
        seed=int(seed),
        max_iter=int(max_iter),
    )


def apply_score_proxy_variant(
    *,
    variant: str,
    index,
    keys_np: np.ndarray,
    query_np: np.ndarray,
    ranked_cpu: np.ndarray,
    ranked_scores_cpu: np.ndarray,
    key_bytes: int,
    metadata_bytes: int,
    kmeans_iters: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float, dict[str, float | int | str]]:
    name = str(variant).strip().lower()
    if name in {"", "baseline", "paged_pq", "pq"}:
        return ranked_cpu, ranked_scores_cpu, 0.0, {"score_proxy_detail": "baseline"}
    if name.startswith("sparq"):
        rank = _parse_variant_count(name, "r", default=4)
        return _apply_sparq_channel_correction(
            index=index,
            keys_np=keys_np,
            query_np=query_np,
            ranked_cpu=ranked_cpu,
            ranked_scores_cpu=ranked_scores_cpu,
            rank=rank,
            key_bytes=int(key_bytes),
        )
    if name.startswith("promoted") or name.startswith("kitty"):
        promote_ratio = _parse_variant_float(name, "p", default=0.1)
        promote_bits = _parse_variant_count(name, "b", default=8)
        return _apply_promoted_residual_correction(
            index=index,
            keys_np=keys_np,
            query_np=query_np,
            ranked_cpu=ranked_cpu,
            ranked_scores_cpu=ranked_scores_cpu,
            promote_ratio=promote_ratio,
            promote_bits=promote_bits,
            key_bytes=int(key_bytes),
            metadata_bytes=int(metadata_bytes),
        )
    if name.startswith("residual_pq") or name.startswith("respq"):
        stages = _parse_variant_count(name, "m", default=1)
        subbits = _parse_variant_count(name, "b", default=4)
        subvecs = _parse_variant_count(name, "s", default=4)
        return _apply_residual_pq_correction(
            index=index,
            keys_np=keys_np,
            query_np=query_np,
            ranked_cpu=ranked_cpu,
            ranked_scores_cpu=ranked_scores_cpu,
            stages=stages,
            subbits=subbits,
            subvecs=subvecs,
            kmeans_iters=int(kmeans_iters),
            key_bytes=int(key_bytes),
            seed=int(seed),
        )
    if name.startswith("bandcal"):
        return ranked_cpu, ranked_scores_cpu, 0.0, {"score_proxy_detail": name}
    raise ValueError(f"unknown score_proxy_variant: {variant}")


def mixed_scores_for_variant(
    *,
    variant: str,
    context_len: int,
    selected_cpu: np.ndarray,
    ranked_cpu: np.ndarray,
    ranked_scores_cpu: np.ndarray,
    exact_scores_np: np.ndarray,
    query_dim: int,
    calibrate: bool,
    key_bytes: int,
) -> tuple[np.ndarray, int, float, float, float, int]:
    name = str(variant).strip().lower()
    selected_cpu = _as_1d_numpy(selected_cpu, dtype=np.int64)
    ranked_cpu = _as_1d_numpy(ranked_cpu, dtype=np.int64)
    ranked_scores_cpu = _as_1d_numpy(ranked_scores_cpu, dtype=np.float32)
    if not name.startswith("bandcal"):
        score_vec, missing, scale, bias = mixed_scores(
            context_len=context_len,
            selected_cpu=selected_cpu,
            ranked_cpu=ranked_cpu,
            ranked_scores_cpu=ranked_scores_cpu,
            exact_scores_np=exact_scores_np,
            query_dim=query_dim,
            calibrate=calibrate,
        )
        return score_vec, int(missing), float(scale), float(bias), 0.0, 0

    bands = max(1, _parse_variant_count(name, "b", default=8))
    probes_per_band = max(0, _parse_variant_count(name, "p", default=0))
    selected_set = set(int(tok) for tok in selected_cpu.tolist())
    sqrt_dim = float(np.sqrt(float(query_dim)))
    pq_logits = ranked_scores_cpu.astype(np.float64, copy=False) / sqrt_dim
    global_score, missing, global_scale, global_bias = mixed_scores(
        context_len=context_len,
        selected_cpu=selected_cpu,
        ranked_cpu=ranked_cpu,
        ranked_scores_cpu=ranked_scores_cpu,
        exact_scores_np=exact_scores_np,
        query_dim=query_dim,
        calibrate=calibrate,
    )
    out = np.full((int(context_len),), -np.inf, dtype=np.float64)
    if selected_cpu.size:
        out[selected_cpu] = exact_scores_np[selected_cpu].astype(np.float64, copy=False)
    extra_probe_tokens: set[int] = set()
    rank_positions = np.arange(int(ranked_cpu.size), dtype=np.int64)
    splits = np.array_split(rank_positions, bands)
    for positions in splits:
        if positions.size == 0:
            continue
        band_tokens = ranked_cpu[positions].astype(np.int64, copy=False)
        fit_tokens = [int(tok) for tok in band_tokens.tolist() if int(tok) in selected_set]
        if probes_per_band > 0:
            candidates = [int(tok) for tok in band_tokens.tolist() if int(tok) not in selected_set]
            if candidates:
                if len(candidates) <= probes_per_band:
                    probes = candidates
                else:
                    probe_pos = np.linspace(0, len(candidates) - 1, probes_per_band).round().astype(np.int64)
                    probes = [candidates[int(pos)] for pos in np.unique(probe_pos)]
                fit_tokens.extend(probes)
                extra_probe_tokens.update(int(tok) for tok in probes)
        if len(fit_tokens) >= 2:
            fit_set = set(fit_tokens)
            x = np.asarray([float(pq_logits[int(pos)]) for pos in positions if int(ranked_cpu[int(pos)]) in fit_set], dtype=np.float64)
            y = np.asarray([float(exact_scores_np[int(ranked_cpu[int(pos)])]) for pos in positions if int(ranked_cpu[int(pos)]) in fit_set], dtype=np.float64)
            x_var = float(np.var(x))
            if x.size >= 2 and x_var > 1e-20:
                cov = float(np.mean((x - float(np.mean(x))) * (y - float(np.mean(y)))))
                scale = cov / x_var
                bias = float(np.mean(y)) - scale * float(np.mean(x))
                if scale <= 0.0 or not np.isfinite(scale):
                    scale = float(global_scale)
                    bias = float(global_bias)
            else:
                scale = float(global_scale)
                bias = float(global_bias)
        else:
            scale = float(global_scale)
            bias = float(global_bias)
        out[band_tokens] = scale * pq_logits[positions] + bias
    if extra_probe_tokens:
        probe_arr = np.asarray(sorted(extra_probe_tokens), dtype=np.int64)
        out[probe_arr] = exact_scores_np[probe_arr].astype(np.float64, copy=False)
    if selected_cpu.size:
        out[selected_cpu] = exact_scores_np[selected_cpu].astype(np.float64, copy=False)
    missing_mask = ~np.isfinite(out)
    missing_count = int(np.count_nonzero(missing_mask))
    if missing_count:
        out[missing_mask] = global_score[missing_mask]
    extra_mb = float(len(extra_probe_tokens) * int(query_dim) * int(key_bytes)) / MB
    return out, missing_count, float(global_scale), float(global_bias), float(extra_mb), int(len(extra_probe_tokens))


def choose_action(
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
    k_limit = float(threshold) if k_threshold is None else float(k_threshold)
    v_limit = float(threshold) if v_threshold is None else float(v_threshold)
    k_bad = bool(k_can and k_delta > k_limit)
    v_bad = bool(v_can and v_delta > v_limit)
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
        k_gain = (float(k_delta) / max(float(extra_k_mb), 1e-9)) if k_bad else -1.0
        v_gain = (float(v_delta) / max(float(extra_v_mb), 1e-9)) if v_bad else -1.0
        return "k" if k_gain >= v_gain else "v"
    raise ValueError(f"unknown policy: {policy}")


def simulate_policy(
    *,
    outputs: dict[tuple[int, int], np.ndarray],
    k_budgets: list[int],
    v_budgets: list[int],
    policy: str,
    threshold: float,
    k_mb_by_idx: list[float],
    v_mb_by_idx: list[float],
    context_len: int,
    threshold_mode: str = "fixed",
    threshold_reference_frac: float = 0.2,
    threshold_scale_shape: str = "linear",
    threshold_min_scale: float = 0.0,
    threshold_max_scale: float = 1.0,
    start_ki: int = 0,
    start_vi: int = 0,
    step_mb_by_idx: dict[tuple[int, int], float] | None = None,
    test_log: list[tuple[int, int, bool, bool, float, float, float, float]] | None = None,
    k_bound_by_pair: dict[tuple[int, int], float] | None = None,
    v_bound_by_pair: dict[tuple[int, int], float] | None = None,
    deescalate: bool = False,
) -> tuple[int, int, int, float, float, list[str]]:
    ki = min(max(0, int(start_ki)), max(0, len(k_budgets) - 1))
    vi = min(max(0, int(start_vi)), max(0, len(v_budgets) - 1))
    steps = 0
    trace: list[str] = []
    while steps < (len(k_budgets) + len(v_budgets) + 4):
        cur = outputs[(ki, vi)]
        k_can = ki + 1 < len(k_budgets)
        v_can = vi + 1 < len(v_budgets)
        k_delta = rel_l2(cur, outputs[(ki + 1, vi)]) if k_can else 0.0
        v_delta = rel_l2(cur, outputs[(ki, vi + 1)]) if v_can else 0.0
        if step_mb_by_idx is None:
            extra_k_mb = float(k_mb_by_idx[ki + 1] - k_mb_by_idx[ki]) if k_can else float("inf")
            extra_v_mb = float(v_mb_by_idx[vi + 1] - v_mb_by_idx[vi]) if v_can else float("inf")
        else:
            cur_mb = float(step_mb_by_idx[(ki, vi)])
            extra_k_mb = float(step_mb_by_idx[(ki + 1, vi)] - cur_mb) if k_can else float("inf")
            extra_v_mb = float(step_mb_by_idx[(ki, vi + 1)] - cur_mb) if v_can else float("inf")
        k_threshold = float(threshold)
        v_threshold = float(threshold)
        if str(threshold_mode) == "budget_delta_frac":
            if k_can:
                k_frac = float(max(0, int(k_budgets[ki + 1]) - int(k_budgets[ki]))) / max(float(context_len), 1.0)
                k_threshold = scaled_threshold(
                    base_threshold=float(threshold),
                    budget_delta_frac=k_frac,
                    reference_frac=float(threshold_reference_frac),
                    shape=str(threshold_scale_shape),
                    min_scale=float(threshold_min_scale),
                    max_scale=float(threshold_max_scale),
                )
            if v_can:
                v_frac = float(max(0, int(v_budgets[vi + 1]) - int(v_budgets[vi]))) / max(float(context_len), 1.0)
                v_threshold = scaled_threshold(
                    base_threshold=float(threshold),
                    budget_delta_frac=v_frac,
                    reference_frac=float(threshold_reference_frac),
                    shape=str(threshold_scale_shape),
                    min_scale=float(threshold_min_scale),
                    max_scale=float(threshold_max_scale),
                )
        elif str(threshold_mode) != "fixed":
            raise ValueError(f"unknown threshold_mode: {threshold_mode}")
        if test_log is not None:
            test_log.append(
                (int(ki), int(vi), bool(k_can), bool(v_can), float(k_delta), float(v_delta), float(k_threshold), float(v_threshold))
            )
        # Decision mode: a certified axis is treated as stable without reading
        # the lookahead band; the recorded raw deltas above still allow
        # false-certify auditing.
        k_delta_used = k_delta
        v_delta_used = v_delta
        if k_bound_by_pair is not None and k_can and float(k_bound_by_pair.get((ki, vi), float("inf"))) <= k_threshold:
            k_delta_used = 0.0
        if v_bound_by_pair is not None and v_can and float(v_bound_by_pair.get((ki, vi), float("inf"))) <= v_threshold:
            v_delta_used = 0.0
        action = choose_action(
            policy=policy,
            k_delta=k_delta_used,
            v_delta=v_delta_used,
            k_can=k_can,
            v_can=v_can,
            threshold=float(threshold),
            k_threshold=k_threshold,
            v_threshold=v_threshold,
            turn=steps,
            extra_k_mb=extra_k_mb,
            extra_v_mb=extra_v_mb,
        )
        trace.append(
            f"{action}:k{ki}/v{vi}:dk={k_delta:.4g}:dv={v_delta:.4g}:"
            f"tk={k_threshold:.4g}:tv={v_threshold:.4g}"
        )
        if action == "stop":
            break
        if action == "k" and k_can:
            ki += 1
        elif action == "v" and v_can:
            vi += 1
        else:
            break
        steps += 1
    if deescalate:
        # Down-walk any axis whose adjacent-band delta is within its scaled
        # threshold. The same pair-delta governs escalation across a band and
        # de-escalation back across it, so this cannot oscillate: a band that
        # forced an escalation above will fail this probe.
        def _band_threshold(lo_budget: int, hi_budget: int) -> float:
            if str(threshold_mode) != "budget_delta_frac":
                return float(threshold)
            frac = float(max(0, int(hi_budget) - int(lo_budget))) / max(float(context_len), 1.0)
            return scaled_threshold(
                base_threshold=float(threshold),
                budget_delta_frac=frac,
                reference_frac=float(threshold_reference_frac),
                shape=str(threshold_scale_shape),
                min_scale=float(threshold_min_scale),
                max_scale=float(threshold_max_scale),
            )

        while True:
            moved = False
            if ki > 0:
                d = rel_l2(outputs[(ki - 1, vi)], outputs[(ki, vi)])
                thr_k = _band_threshold(k_budgets[ki - 1], k_budgets[ki])
                if d <= thr_k:
                    trace.append(f"kd:k{ki}->k{ki - 1}:d={d:.4g}:t={thr_k:.4g}")
                    ki -= 1
                    steps += 1
                    moved = True
            if vi > 0:
                d = rel_l2(outputs[(ki, vi - 1)], outputs[(ki, vi)])
                thr_v = _band_threshold(v_budgets[vi - 1], v_budgets[vi])
                if d <= thr_v:
                    trace.append(f"vd:v{vi}->v{vi - 1}:d={d:.4g}:t={thr_v:.4g}")
                    vi -= 1
                    steps += 1
                    moved = True
            if not moved:
                break
    return ki, vi, steps, float(k_delta), float(v_delta), trace


def find_oracle_budget(
    *,
    outputs: dict[tuple[int, int], np.ndarray],
    dense: np.ndarray,
    k_budgets: list[int],
    v_budgets: list[int],
    k_mb_by_idx: list[float],
    v_mb_by_idx: list[float],
    target_rel_l2: float,
    step_mb_by_idx: dict[tuple[int, int], float] | None = None,
) -> dict[str, object]:
    best_satisfied: dict[str, object] | None = None
    best_error: dict[str, object] | None = None
    for ki, k_budget in enumerate(k_budgets):
        for vi, v_budget in enumerate(v_budgets):
            err = rel_l2(dense, outputs[(ki, vi)])
            total_mb = (
                float(k_mb_by_idx[ki] + v_mb_by_idx[vi])
                if step_mb_by_idx is None
                else float(step_mb_by_idx[(ki, vi)])
            )
            row = {
                "oracle_ki": int(ki),
                "oracle_vi": int(vi),
                "oracle_k_budget": int(k_budget),
                "oracle_v_budget": int(v_budget),
                "oracle_step_MB_per_head": float(total_mb),
                "oracle_head_attention_relative_L2": float(err),
                "oracle_target_satisfied": bool(err <= float(target_rel_l2)),
            }
            if best_error is None or float(err) < float(best_error["oracle_head_attention_relative_L2"]):
                best_error = row
            if err <= float(target_rel_l2):
                if best_satisfied is None or total_mb < float(best_satisfied["oracle_step_MB_per_head"]):
                    best_satisfied = row
    return best_satisfied if best_satisfied is not None else dict(best_error or {})


def _budget_index_at_least(budgets: list[int], target: float) -> int:
    if not budgets:
        return 0
    for idx, budget in enumerate(budgets):
        if int(budget) >= float(target):
            return int(idx)
    return len(budgets) - 1


def _parse_fraction_suffix(name: str, prefix: str, default: float) -> float:
    match = re.search(rf"(?:^|_){re.escape(prefix)}([0-9p.]+)", str(name))
    if not match:
        return float(default)
    return float(match.group(1).replace("p", "."))


def _softmax_prefix_count(scores: np.ndarray, *, mass: float, scale: float) -> int:
    if scores.size == 0:
        return 0
    # Some Slurm nodes expose a brittle NumPy reduction path in this venv. This
    # initializer is not a hot path, so keep it on plain Python reductions.
    logits = [float(score) * float(scale) for score in scores.reshape(-1).tolist()]
    max_logit = max(logits)
    weights = [math.exp(logit - max_logit) for logit in logits]
    total = float(sum(weights))
    if total <= 0.0 or not np.isfinite(total):
        return int(scores.size)
    running = 0.0
    target = float(mass) * total
    for idx, weight in enumerate(weights):
        running += float(weight)
        if running >= target:
            return int(idx + 1)
    return int(scores.size)


def _softmax_normalized_entropy(scores: np.ndarray, *, scale: float) -> float:
    if scores.size <= 1:
        return 0.0
    logits = [float(score) * float(scale) for score in scores.reshape(-1).tolist()]
    max_logit = max(logits)
    weights = [math.exp(logit - max_logit) for logit in logits]
    total = float(sum(weights))
    if total <= 0.0 or not np.isfinite(total):
        return 1.0
    entropy = 0.0
    for weight in weights:
        prob = float(weight) / total
        if prob > 0.0:
            entropy -= prob * math.log(prob)
    return min(max(float(entropy / math.log(float(len(weights)))), 0.0), 1.0)


def scaled_threshold(
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
        raise ValueError(f"unknown threshold scaling shape: {shape}")
    scale = min(max(float(scale), float(min_scale)), float(max_scale))
    return float(base_threshold) * scale


def _v_selection_block_size(rule: str, default: int) -> int:
    name = str(rule).strip().lower()
    match = re.search(r"(?:^|_)b(\d+)", name)
    if match:
        return max(1, int(match.group(1)))
    return max(1, int(default))


def exact_v_mask_for_rule(
    *,
    rule: str,
    risk_scores: np.ndarray,
    value_scores: np.ndarray,
    exact_count: int,
    block_size: int,
) -> tuple[np.ndarray, dict[str, object]]:
    """Return exact-V mask for probability-weighted or V-error-only selection."""

    name = str(rule).strip().lower()
    risk_scores = np.asarray(risk_scores, dtype=np.float64).reshape(-1)
    value_scores = np.asarray(value_scores, dtype=np.float64).reshape(-1)
    context_len = int(risk_scores.shape[0])
    if int(value_scores.shape[0]) != context_len:
        raise ValueError("value_scores must have the same length as risk_scores")
    count = max(0, min(int(exact_count), context_len))
    if name in {"", "global", "global_residual_risk", "residual_risk"}:
        return top_mask(risk_scores, count), {
            "v_selection_rule": "global_residual_risk",
            "v_selection_block_size": 0,
            "v_selection_exact_target": int(count),
        }
    if name in {"v_error_only", "global_v_error", "value_error", "code_error", "v_code_error"}:
        return top_mask(value_scores, count), {
            "v_selection_rule": "v_error_only",
            "v_selection_block_size": 0,
            "v_selection_exact_target": int(count),
        }
    if name.startswith("local_block"):
        block = _v_selection_block_size(name, int(block_size))
        mask = np.zeros((context_len,), dtype=bool)
        for start in range(0, context_len, block):
            end = min(context_len, start + block)
            # Proportional deterministic quota: each block commits immediately,
            # while the row-level exact-V count remains exactly `count`.
            local_start = int(math.floor(float(count) * float(start) / max(float(context_len), 1.0)))
            local_end = int(math.floor(float(count) * float(end) / max(float(context_len), 1.0)))
            local_count = max(0, min(end - start, local_end - local_start))
            if local_count <= 0:
                continue
            local_mask = top_mask(risk_scores[start:end], local_count)
            mask[start:end] |= local_mask
        return mask, {
            "v_selection_rule": f"local_block_b{block}",
            "v_selection_block_size": int(block),
            "v_selection_exact_target": int(count),
        }
    if name.startswith("local_v_error") or name.startswith("local_value_error") or name.startswith("local_code_error"):
        block = _v_selection_block_size(name, int(block_size))
        mask = np.zeros((context_len,), dtype=bool)
        for start in range(0, context_len, block):
            end = min(context_len, start + block)
            local_start = int(math.floor(float(count) * float(start) / max(float(context_len), 1.0)))
            local_end = int(math.floor(float(count) * float(end) / max(float(context_len), 1.0)))
            local_count = max(0, min(end - start, local_end - local_start))
            if local_count <= 0:
                continue
            local_mask = top_mask(value_scores[start:end], local_count)
            mask[start:end] |= local_mask
        return mask, {
            "v_selection_rule": f"local_v_error_b{block}",
            "v_selection_block_size": int(block),
            "v_selection_exact_target": int(count),
        }
    if name.startswith("streaming_global_risk"):
        block = _v_selection_block_size(name, int(block_size))
        mask, exact_reads = streaming_topk_mask_and_reads(risk_scores, count=count, block_size=block)
        return mask, {
            "v_selection_rule": f"streaming_global_risk_b{block}",
            "v_selection_block_size": int(block),
            "v_selection_exact_target": int(count),
            "v_selection_exact_reads": int(exact_reads),
        }
    raise ValueError(f"unknown V selection rule: {rule}")


def streaming_topk_mask_and_reads(
    scores: np.ndarray,
    *,
    count: int,
    block_size: int,
) -> tuple[np.ndarray, int]:
    return streaming_topk_masks_and_reads_for_counts(
        scores,
        counts=[int(count)],
        block_size=int(block_size),
    )[max(0, min(int(count), int(np.asarray(scores).reshape(-1).shape[0])))]


def streaming_topk_masks_and_reads_for_counts(
    scores: np.ndarray,
    *,
    counts: list[int],
    block_size: int,
) -> dict[int, tuple[np.ndarray, int]]:
    """Exact-read union masks for FlashAttention-style running top-k V risk.

    For a fixed V budget k, a token is read exactly if it belongs to the
    top-k risk set at the end of its block. The exact correction is retained
    even if a later block evicts the token from the running top-k set.
    """

    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    context_len = int(scores.shape[0])
    clipped_counts = [max(0, min(int(count), context_len)) for count in counts]
    unique_counts = sorted(set(clipped_counts))
    results: dict[int, tuple[np.ndarray, int]] = {}
    for count in unique_counts:
        if count <= 0:
            results[count] = (np.zeros((context_len,), dtype=bool), 0)
        elif count >= context_len:
            mask = np.ones((context_len,), dtype=bool)
            results[count] = (mask, context_len)

    active_counts = [count for count in unique_counts if 0 < count < context_len]
    if not active_counts:
        return results
    block = max(1, int(block_size))
    n_blocks = int(math.ceil(float(context_len) / float(block)))
    token_idx = np.arange(context_len, dtype=np.int64)
    block_ids = (token_idx // block).astype(np.int32, copy=False)
    order = np.argsort(-scores)
    inverse_order = np.empty((context_len,), dtype=np.int64)
    inverse_order[order] = np.arange(context_len, dtype=np.int64)
    ordered_blocks = block_ids[order]
    prefix_rank_at_block = np.empty((context_len,), dtype=np.int32)
    for block_id in range(n_blocks):
        start = int(block_id * block)
        end = min(context_len, start + block)
        ranks_for_prefix = np.cumsum(ordered_blocks <= block_id, dtype=np.int32)
        prefix_rank_at_block[start:end] = ranks_for_prefix[inverse_order[start:end]]
    for count in active_counts:
        mask = prefix_rank_at_block <= int(count)
        results[count] = (mask, int(np.count_nonzero(mask)))
    return results


def output_from_base_and_exact_mask(
    *,
    base_output: np.ndarray,
    probs: np.ndarray,
    residual: np.ndarray,
    exact_mask: np.ndarray,
) -> np.ndarray:
    out = base_output.astype(np.float64, copy=True)
    if bool(np.any(exact_mask)):
        out += probs[exact_mask].astype(np.float64, copy=False) @ residual[exact_mask].astype(np.float64, copy=False)
    return out.astype(np.float32, copy=False)


def _apply_global_pq_codebook(
    index,
    keys_np: np.ndarray,
    *,
    dynamic_start: int,
    subvecs: int,
    subbits: int,
    kmeans_iters: int,
    seed: int,
    sample_rows: int = 16384,
) -> None:
    """Replace per-page PQ codebooks with one shared codebook.

    Models a chip keeping a single SRAM-resident codebook: the scan then reads
    only per-token codes (4B/token) instead of re-reading 64KB of codebooks
    per page. Codes are re-assigned by nearest-centroid against the shared
    codebook; ranking fidelity loss is what the experiment measures.
    """
    if not index.pages:
        return
    from benchmark.attention_efficiency_threeway_eval import build_pq_index

    sealed_end = max(int(p.start) + int(p.size) for p in index.pages)
    block = keys_np[int(dynamic_start):sealed_end].astype(np.float32, copy=False)
    rng = np.random.default_rng(int(seed))
    if block.shape[0] > int(sample_rows):
        sample = block[rng.choice(block.shape[0], size=int(sample_rows), replace=False)]
    else:
        sample = block
    codebooks_np, _codes, _sv, _c = build_pq_index(
        sample, 0, sample.shape[0], subvecs=int(subvecs), subbits=int(subbits), seed=int(seed), max_iter=int(kmeans_iters)
    )
    device = index.pages[0].codebooks.device
    shared = torch.as_tensor(np.ascontiguousarray(codebooks_np), dtype=torch.float32, device=device)
    subdim = codebooks_np.shape[-1]
    c_sq = np.sum(codebooks_np.astype(np.float64) ** 2, axis=2)
    for page in index.pages:
        pb = keys_np[int(page.start): int(page.start) + int(page.size)].astype(np.float64, copy=False)
        codes = np.zeros((pb.shape[0], int(subvecs)), dtype=np.uint16)
        for sub in range(int(subvecs)):
            part = pb[:, sub * subdim: (sub + 1) * subdim]
            dists = -2.0 * (part @ codebooks_np[sub].astype(np.float64).T) + c_sq[sub][None, :]
            codes[:, sub] = np.argmin(dists, axis=1).astype(np.uint16)
        page.codebooks = shared
        page.codes = torch.as_tensor(codes.astype(np.int64), dtype=torch.long, device=device)
    index.native_codebooks = None
    index.native_codes = None
    index.native_page_starts = None


def _quantize_rows_symmetric(x: np.ndarray, bits: int) -> np.ndarray:
    """Per-row symmetric absmax quantization: MSB-plane read proxy."""
    levels = float((1 << (max(2, int(bits)) - 1)) - 1)
    scale = np.max(np.abs(x), axis=1, keepdims=True) / levels
    scale = np.maximum(scale, 1e-12)
    return (np.round(x / scale) * scale).astype(np.float32, copy=False)


def _precision_lo_tokens(
    base: list[int],
    ranked_cpu: np.ndarray,
    budget: int,
    context_len: int,
    hi_frac: float,
) -> np.ndarray:
    """Ranked-prefix tokens beyond the hi-precision fraction (base excluded)."""
    base_set = set(int(tok) for tok in base)
    add = [int(tok) for tok in ranked_cpu.tolist() if int(tok) < int(context_len) and int(tok) not in base_set][: int(budget)]
    hi_count = int(math.ceil(len(add) * float(hi_frac)))
    return np.asarray(add[hi_count:], dtype=np.int64)


def output_from_base_and_split_masks(
    *,
    base_output: np.ndarray,
    probs: np.ndarray,
    residual: np.ndarray,
    residual_lo: np.ndarray,
    hi_mask: np.ndarray,
    lo_mask: np.ndarray,
) -> np.ndarray:
    out = base_output.astype(np.float64, copy=True)
    if bool(np.any(hi_mask)):
        out += probs[hi_mask].astype(np.float64, copy=False) @ residual[hi_mask].astype(np.float64, copy=False)
    if bool(np.any(lo_mask)):
        out += probs[lo_mask].astype(np.float64, copy=False) @ residual_lo[lo_mask].astype(np.float64, copy=False)
    return out.astype(np.float32, copy=False)


def v_selection_state_mb(
    *,
    rule: str,
    exact_count: int,
    index_bytes: int,
    logit_bytes: int,
    include_state: bool,
) -> float:
    if not bool(include_state):
        return 0.0
    name = str(rule).strip().lower()
    if name.startswith("two_pass_risk"):
        # Pass 1 keeps only a running cutoff heap of log-risk values; pass 2
        # commits tile-locally against the scalar cutoff, so no survivor
        # index/logit list is retained across blocks.
        mult = _parse_fraction_suffix(name, "f", 1.0)
        heap = int(round(float(max(0, int(exact_count))) * float(mult)))
        return float(heap * int(logit_bytes)) / MB
    if (
        name.startswith("local_block")
        or name in {"v_error_only", "global_v_error", "value_error", "code_error", "v_code_error"}
        or name.startswith("local_v_error")
        or name.startswith("local_value_error")
        or name.startswith("local_code_error")
        or name.startswith("streaming_global_risk")
    ):
        return 0.0
    count = max(0, int(exact_count))
    return float(count * (int(index_bytes) + int(logit_bytes))) / MB


def initial_budget_indices(
    *,
    strategy: str,
    context_len: int,
    ranked_scores_cpu: np.ndarray,
    query_dim: int,
    k_budgets: list[int],
    v_budgets: list[int],
    previous_fraction: tuple[float, float] | None,
) -> tuple[int, int]:
    name = str(strategy).strip().lower()
    if name in {"", "min", "zero"}:
        return 0, 0
    if name.startswith("fixed_f"):
        frac = _parse_fraction_suffix(name, "f", 0.05)
        k_target = max(float(k_budgets[0]), float(context_len) * float(frac))
        v_target = max(float(v_budgets[0]), k_target * 0.25)
        return _budget_index_at_least(k_budgets, k_target), _budget_index_at_least(v_budgets, v_target)
    if name.startswith("proxy_mass_m"):
        mass = _parse_fraction_suffix(name, "m", 0.5)
        count = _softmax_prefix_count(
            ranked_scores_cpu,
            mass=min(max(float(mass), 0.0), 0.999999),
            scale=1.0 / math.sqrt(max(1.0, float(query_dim))),
        )
        k_target = max(float(k_budgets[0]), float(count))
        v_target = max(float(v_budgets[0]), k_target * 0.25)
        return _budget_index_at_least(k_budgets, k_target), _budget_index_at_least(v_budgets, v_target)
    if name.startswith("proxy_entropy"):
        max_frac = _parse_fraction_suffix(name, "f", 0.25)
        entropy = _softmax_normalized_entropy(
            ranked_scores_cpu,
            scale=1.0 / math.sqrt(max(1.0, float(query_dim))),
        )
        k_target = max(float(k_budgets[0]), float(context_len) * float(max_frac) * float(entropy))
        v_target = max(float(v_budgets[0]), k_target * 0.25)
        return _budget_index_at_least(k_budgets, k_target), _budget_index_at_least(v_budgets, v_target)
    if name.startswith("temporal_prev"):
        if previous_fraction is None:
            return 0, 0
        scale = 0.5 if name.endswith("_low") else 1.0
        k_frac, v_frac = previous_fraction
        k_target = max(float(k_budgets[0]), float(context_len) * float(k_frac) * float(scale))
        v_target = max(float(v_budgets[0]), float(context_len) * float(v_frac) * float(scale))
        return _budget_index_at_least(k_budgets, k_target), _budget_index_at_least(v_budgets, v_target)
    raise ValueError(f"unknown start strategy: {strategy}")


def run() -> None:
    parser = argparse.ArgumentParser(description="Joint K/V budget policy simulation on saved QKV traces.")
    parser.add_argument("--qkv_trace", required=True)
    parser.add_argument("--x_trace", required=True)
    parser.add_argument(
        "--model_snapshot",
        default=".hf_cache/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--decode_lengths", default="500,1000,2000,4000,8000,16000,32000,64000,128000")
    parser.add_argument("--max_qidx_per_decode", type=int, default=1)
    parser.add_argument("--heads", default="")
    parser.add_argument("--k_budgets", default="4096,8192,14336,32768")
    parser.add_argument("--v_budgets", default="1024,2048,4096,6144,8192,12288,16384")
    parser.add_argument(
        "--k_budget_fracs",
        default="",
        help="Optional comma-separated K budget fractions, e.g. 0.005,0.01,0.02 or 0.5%,1%,2%. Overrides --k_budgets per query.",
    )
    parser.add_argument(
        "--v_budget_fracs",
        default="",
        help="Optional comma-separated V budget fractions. Overrides --v_budgets per query.",
    )
    parser.add_argument("--stability_thresholds", default="0.0005,0.001,0.002")
    parser.add_argument(
        "--oracle_rel_l2_targets",
        default="",
        help="Optional offline-only head-output relL2 targets for cheapest-grid oracle budget diagnostics.",
    )
    parser.add_argument("--threshold_mode", choices=["fixed", "budget_delta_frac"], default="fixed")
    parser.add_argument("--threshold_reference_frac", type=float, default=0.2)
    parser.add_argument("--threshold_scale_shape", choices=["linear", "sqrt", "log"], default="linear")
    parser.add_argument("--threshold_min_scale", type=float, default=0.0)
    parser.add_argument("--threshold_max_scale", type=float, default=1.0)
    parser.add_argument(
        "--policies",
        default="k_first_priority,v_first_priority,k_first_alternating,v_first_alternating,sensitivity_greedy",
    )
    parser.add_argument(
        "--score_proxy_variants",
        default="baseline",
        help=(
            "Comma-separated selector score proxy variants: baseline, sparq_r4, "
            "promoted_p0p1_b8, residual_pq_m1b4_s4, bandcal_b8_p16."
        ),
    )
    parser.add_argument(
        "--start_strategies",
        default="min",
        help=(
            "Comma-separated initial budget strategies before adaptive confidence: "
            "min, fixed_f0p05, proxy_mass_m0p7, temporal_prev, temporal_prev_low."
        ),
    )
    parser.add_argument(
        "--v_selection_rules",
        default="global_residual_risk",
        help=(
            "Comma-separated exact-V selection rules. Use global_residual_risk "
            "for p^2*V-error ranking, v_error_only for query-independent V-error "
            "ranking, local_block_b<size> for local p^2*V-error block commit, "
            "local_v_error_b<size> for local query-independent V-error block commit, "
            "streaming_global_risk_b<size> for block-streaming global top-risk "
            "with immediate exact-V reads and eviction waste accounting, "
            "or two_pass_risk[_f<mult>] for pass-1 PQ-domain risk-cutoff estimation "
            "with tile-local pass-2 threshold commits (f scales the cutoff rank, "
            "e.g. two_pass_risk_f1p25)."
        ),
    )
    parser.add_argument("--v_local_block_size", type=int, default=1024)
    parser.add_argument(
        "--lookahead_diagnostic",
        action="store_true",
        help=(
            "Record hardware-style confidence-lookahead accounting alongside the "
            "canonical trajectory: per-test certification by compressed-domain "
            "delta bounds (charge_all/cs/rms<lambda>), false-certify counts, and "
            "wasted lookahead-band MB. Does not change policy decisions. Only "
            "active for the global_residual_risk V rule."
        ),
    )
    parser.add_argument(
        "--lookahead_decision_variants",
        default="",
        help=(
            "Comma-separated lookahead bound variants (e.g. var4) that also run "
            "in decision mode: the bound gates escalation instead of only being "
            "audited. Emits extra result rows with v_selection_rule suffixed "
            "'+la_<variant>'. Requires --lookahead_diagnostic."
        ),
    )
    parser.add_argument(
        "--temporal_reuse_max_stale",
        type=int,
        default=0,
        help=(
            "Frozen-selection temporal reuse: while the current position is at "
            "most this many tokens past the last full selector rescan for a "
            "head, reuse that rescan's ranked list, PQ logits, and page index "
            "unchanged. Newly appended tokens stay resident (exact) in an "
            "extended base until the next rescan. Reuse steps charge a stale "
            "logit-row reread (2B/token) plus a ranked-index reread "
            "(4B/candidate) instead of the PQ fullscan. 0 disables (canonical "
            "rescan every step)."
        ),
    )
    parser.add_argument(
        "--temporal_reuse_mode",
        choices=["frozen", "incremental"],
        default="frozen",
        help=(
            "frozen: reuse the last rescan's page index unchanged; tokens "
            "appended since stay resident (costs extra exact-K on page-seal "
            "boundary crossings). incremental: keep the current page index, "
            "PQ-score only pages sealed since the last rescan with the "
            "current query, and merge them into the stale ranking - no "
            "resident growth beyond canonical pending."
        ),
    )
    parser.add_argument(
        "--gqa_union_stats",
        action="store_true",
        help=(
            "Per (qidx, kv_head, trajectory) record the UNION of accepted "
            "exact-K/exact-V token sets across the q heads sharing that kv "
            "head, vs the sum of per-head set sizes. union/sum is the GQA "
            "row-read sharing factor a chip gather engine would see (1.0 = "
            "no overlap, 1/group_size = perfect overlap). Run with all q "
            "heads of at least one kv group (e.g. --heads 0,1,2,3). Writes "
            "gqa_union_stats.csv; does not change behavior."
        ),
    )
    parser.add_argument(
        "--budget_deescalate",
        action="store_true",
        help=(
            "After the escalate-only walk stops, greedily step DOWN any axis "
            "whose adjacent-band output delta is within its scaled threshold. "
            "With proxy_mass start this is a predict-then-verify controller "
            "(settle near the predicted rung, correct over-prediction); with "
            "temporal_prev start it is a warm-start hysteresis controller "
            "(carry the previous budget but re-earn it via down-probes)."
        ),
    )
    parser.add_argument(
        "--temporal_reuse_budget",
        choices=["ladder", "frozen"],
        default="ladder",
        help=(
            "ladder: run the stability ladder on reuse steps as usual (it "
            "over-escalates K on stale rankings). frozen: on reuse steps skip "
            "the ladder and reuse the (k,v) budget rungs settled at this "
            "head's last rescan step; quality on reuse steps is unguarded "
            "until the next rescan."
        ),
    )
    parser.add_argument(
        "--temporal_cache_stats",
        action="store_true",
        help=(
            "Record accepted-set overlap between consecutive same-head queries "
            "(whatever their position gap in the trace): new-token counts for "
            "the accepted exact-K and exact-V sets. Measures the hit rate a "
            "token-keyed on-chip K/V row cache would see; does not change "
            "behavior."
        ),
    )
    parser.add_argument(
        "--precision_k_hi_frac",
        type=float,
        default=1.0,
        help=(
            "Progressive-precision exact-K reads: this fraction of the ranked "
            "selection prefix (by PQ-score rank; base/resident tokens always "
            "full precision) is read at full key_bytes, the rest at the "
            "precision_lo_bits MSB plane. 1.0 disables."
        ),
    )
    parser.add_argument(
        "--precision_v_hi_frac",
        type=float,
        default=1.0,
        help=(
            "Progressive-precision exact-V reads: this fraction of the exact-V "
            "set (by residual-risk rank) is read at full value_bytes, the rest "
            "at the precision_lo_bits MSB plane. 1.0 disables."
        ),
    )
    parser.add_argument("--precision_lo_bits", type=int, default=8)
    parser.add_argument("--global_pq_subbits", type=int, default=0, help="Bits for the shared codebook (0 = same as --subbits). Larger shared codebooks recover ranking fidelity at negligible amortized read cost.")
    parser.add_argument("--global_pq_sample_rows", type=int, default=16384)
    parser.add_argument(
        "--global_pq_codebook",
        action="store_true",
        help=(
            "Replace per-page K-PQ codebooks with one shared codebook trained "
            "on a sample of the sealed region (SRAM-resident on a chip). The "
            "selector scan then charges the codebook once per head-query "
            "instead of per page; codes are re-assigned to the shared "
            "codebook. Measures the ranking-fidelity cost of dropping "
            "page-local codebooks."
        ),
    )
    parser.add_argument(
        "--page_coarse_block",
        type=int,
        default=512,
        help=(
            "Block size for the coarse tier inside unscanned pages: tokens get "
            "their block's mean reconstructed logit and mean V-PQ value "
            "(stored block stats, 2*d*2B per block). Page-level means (block = "
            "page size) were strongly negative at 32k."
        ),
    )
    parser.add_argument(
        "--page_scan_frac",
        type=float,
        default=1.0,
        help=(
            "Page-bound scan pruning: process sealed pages best-first by their "
            "max PQ score (oracle page order = tight envelope of any stored "
            "per-page bound) and PQ-scan only enough pages to cover this "
            "fraction of indexed tokens. Unscanned pages contribute at page "
            "granularity: every token gets its page's centroid logit "
            "(q . mean_khat == mean member PQ logit, exact by linearity) and "
            "the page-mean V-PQ value in the base output; their tokens cannot "
            "be selected for exact-K or exact-V. Charges per-page stat reads "
            "(MBR + mean_khat + mean_vhat, 4*d*2B/page) plus codes+codebooks "
            "for scanned pages only. 1.0 disables."
        ),
    )
    parser.add_argument("--include_v_selection_state_in_step_mb", action="store_true")
    parser.add_argument("--survivor_logit_bytes", type=int, default=2)
    parser.add_argument("--selector_mode", choices=["fullscan", "routed", "quest", "quest_pq"], default="fullscan")
    parser.add_argument("--quest_rank", type=int, default=16)
    parser.add_argument("--selector_index_bytes", type=int, default=4)
    parser.add_argument("--tail_score_calibration", choices=["none", "affine_selected"], default="none")
    parser.add_argument("--static_prefix", type=int, default=128)
    parser.add_argument("--static_suffix", type=int, default=128)
    parser.add_argument("--page_size", type=int, default=5632)
    parser.add_argument("--subvecs", type=int, default=4)
    parser.add_argument("--subbits", type=int, default=8)
    parser.add_argument("--value_subvecs", type=int, default=1)
    parser.add_argument("--value_subbits", type=int, default=4)
    parser.add_argument("--kmeans_iters", type=int, default=3)
    parser.add_argument("--key_bytes", type=int, default=2)
    parser.add_argument("--value_bytes", type=int, default=2)
    parser.add_argument("--code_stat_bytes", type=int, default=2)
    parser.add_argument("--nprobes", default="512")
    parser.add_argument("--router_prototypes", type=int, default=16)
    parser.add_argument("--router_merge_rel", type=float, default=0.05)
    parser.add_argument("--router_merge_var", type=float, default=0.0)
    parser.add_argument("--router_max_groups", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.set_grad_enabled(False)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True), encoding="utf-8")

    trace = load_trace(args.qkv_trace)
    if str(args.decode_lengths).strip().lower() == "all":
        q_indices = list(range(int(trace.positions.shape[0])))
    else:
        q_indices = trace.q_indices_for_decodes(parse_csv_ints(args.decode_lengths))
    if int(args.max_qidx_per_decode) > 0:
        limited: list[int] = []
        counts: dict[int, int] = {}
        for qidx in q_indices:
            decode = trace.decode_tokens_for_qidx(int(qidx))
            if counts.get(int(decode), 0) >= int(args.max_qidx_per_decode):
                continue
            limited.append(int(qidx))
            counts[int(decode)] = counts.get(int(decode), 0) + 1
        q_indices = limited
    if not q_indices:
        raise ValueError("no query indices selected")

    heads = parse_csv_ints(args.heads) if str(args.heads).strip() else list(range(int(trace.num_heads)))
    base_k_budgets = sorted(set(parse_csv_ints(args.k_budgets)))
    base_v_budgets = sorted(set(parse_csv_ints(args.v_budgets)))
    k_budget_fracs = parse_csv_ratios(args.k_budget_fracs)
    v_budget_fracs = parse_csv_ratios(args.v_budget_fracs)
    if bool(k_budget_fracs) != bool(v_budget_fracs):
        raise ValueError("--k_budget_fracs and --v_budget_fracs must be provided together")
    thresholds = parse_csv_floats(args.stability_thresholds)
    oracle_rel_l2_targets = parse_csv_floats(args.oracle_rel_l2_targets)
    policies = [part.strip() for part in str(args.policies).split(",") if part.strip()]
    score_proxy_variants = parse_csv_names(args.score_proxy_variants)
    start_strategies = parse_csv_names(args.start_strategies)
    v_selection_rules = parse_csv_names(args.v_selection_rules)
    lookahead_decision_variants = parse_csv_names(args.lookahead_decision_variants)
    for dv in lookahead_decision_variants:
        if dv not in LOOKAHEAD_VARIANTS or dv == "charge_all":
            raise ValueError(f"unknown lookahead decision variant: {dv}")
    if lookahead_decision_variants and not bool(args.lookahead_diagnostic):
        raise ValueError("--lookahead_decision_variants requires --lookahead_diagnostic")
    precision_k_hi_frac = float(args.precision_k_hi_frac)
    precision_v_hi_frac = float(args.precision_v_hi_frac)
    precision_lo_bits = int(args.precision_lo_bits)
    precision_active = precision_k_hi_frac < 1.0 or precision_v_hi_frac < 1.0
    if not (0.0 <= precision_k_hi_frac <= 1.0 and 0.0 <= precision_v_hi_frac <= 1.0):
        raise ValueError("--precision_k_hi_frac/--precision_v_hi_frac must be in [0, 1]")
    precision_lo_bytes = 0.5 if precision_lo_bits <= 4 else (1 if precision_lo_bits <= 8 else 2)
    if precision_active and bool(args.lookahead_diagnostic):
        raise ValueError("progressive precision is incompatible with --lookahead_diagnostic")
    page_scan_frac = float(args.page_scan_frac)
    if not (0.0 < page_scan_frac <= 1.0):
        raise ValueError("--page_scan_frac must be in (0, 1]")
    if page_scan_frac < 1.0:
        if str(args.selector_mode) != "fullscan":
            raise ValueError("--page_scan_frac requires selector_mode=fullscan")
        if bool(args.lookahead_diagnostic):
            raise ValueError("--page_scan_frac is incompatible with --lookahead_diagnostic")
        if int(args.temporal_reuse_max_stale) > 0:
            raise ValueError("--page_scan_frac and --temporal_reuse_max_stale are separate experiments")
    temporal_reuse_max_stale = int(args.temporal_reuse_max_stale)
    temporal_cache_stats = bool(args.temporal_cache_stats)
    temporal_budget_frozen = str(args.temporal_reuse_budget) == "frozen"
    if temporal_budget_frozen and temporal_reuse_max_stale <= 0:
        raise ValueError("--temporal_reuse_budget frozen requires --temporal_reuse_max_stale > 0")
    if temporal_reuse_max_stale > 0:
        if str(args.selector_mode) != "fullscan":
            raise ValueError("--temporal_reuse_max_stale requires selector_mode=fullscan")
        if bool(args.lookahead_diagnostic):
            raise ValueError("--temporal_reuse_max_stale is incompatible with --lookahead_diagnostic")
        if score_proxy_variants != ["baseline"]:
            raise ValueError("--temporal_reuse_max_stale requires score_proxy_variants=baseline")
        for rule in v_selection_rules:
            if str(rule).strip().lower() not in {"", "global", "global_residual_risk", "residual_risk"}:
                raise ValueError(
                    "--temporal_reuse_max_stale only supports the global_residual_risk "
                    "V rule (two-pass pass-1 rides the fullscan that reuse skips)"
                )
    nprobes = parse_csv_ints(args.nprobes)

    x_data = np.load(args.x_trace, mmap_mode="r")
    x_meta = json.loads(str(x_data["metadata"].item()))
    layer_idx = int(x_meta["layer_idx"])
    model_dir = PROJECT_ROOT / args.model_snapshot
    weight_map = load_weight_index(model_dir)
    wo = load_safetensor_weight(model_dir, weight_map, f"model.layers.{layer_idx}.self_attn.o_proj.weight", device)

    head_rows: list[dict[str, object]] = []
    layer_rows: list[dict[str, object]] = []
    oracle_rows: list[dict[str, object]] = []
    previous_fraction: dict[tuple[str, str, str, float, str, int], tuple[float, float]] = {}
    # Temporal reuse/cache state across the qidx loop. Ranked-list reuse is
    # per q-head (query-dependent); the frozen page index is per kv-head; the
    # sealed-page build cache avoids re-running k-means when the sealed page
    # set is unchanged between steps (bit-exact: same key prefix, same seed).
    temporal_ranked_cache: dict[int, dict[str, object]] = {}
    temporal_index_cache: dict[int, dict[str, object]] = {}
    temporal_prev_sets: dict[tuple, dict[str, object]] = {}
    # Settled (ki, vi) rung indices from each head's last rescan step, keyed
    # like previous_fraction; consumed on reuse steps in frozen-budget mode.
    temporal_budget_cache: dict[tuple[str, str, str, float, str, int], tuple[int, int]] = {}
    gqa_union_stats = bool(args.gqa_union_stats)
    gqa_union_rows: list[dict[str, object]] = []
    page_index_build_cache: dict[int, dict[str, object]] = {}
    t0 = time.perf_counter()

    for qidx in q_indices:
        position = int(trace.positions[int(qidx)])
        decode_tokens = int(trace.decode_tokens_for_qidx(int(qidx)))
        context_len = int(position) + 1
        k_budgets = budgets_from_fracs(context_len, k_budget_fracs) if k_budget_fracs else base_k_budgets
        v_budgets = budgets_from_fracs(context_len, v_budget_fracs) if v_budget_fracs else base_v_budgets
        dynamic_start = min(max(0, int(args.static_prefix)), int(trace.input_len))
        indexed_end = max(
            min(max(0, int(args.static_prefix)), int(trace.input_len)),
            min(context_len - max(0, int(args.static_suffix)), trace.keys.shape[1]),
        )
        needed_kv_heads = sorted({int(trace.kv_head_for(h)) for h in heads})
        gqa_union_acc: dict[tuple, dict[str, object]] = {}
        temporal_reuse_now = False
        temporal_stale_tokens = 0
        if temporal_reuse_max_stale > 0 and all(kv in temporal_index_cache for kv in needed_kv_heads):
            cache_positions = {int(temporal_index_cache[kv]["position"]) for kv in needed_kv_heads}
            if len(cache_positions) == 1:
                stale = int(position) - cache_positions.pop()
                if 0 < stale <= temporal_reuse_max_stale:
                    temporal_reuse_now = True
                    temporal_stale_tokens = int(stale)
        index_cache = {}
        k_resid_norm_cache: dict[int, np.ndarray] = {}
        sealed_end_geom = int(dynamic_start) + (
            (max(0, int(indexed_end) - int(dynamic_start)) // max(1, int(args.page_size))) * max(1, int(args.page_size))
        )
        temporal_incremental = str(args.temporal_reuse_mode) == "incremental"
        for kv_head in needed_kv_heads:
            keys_np = trace.keys[kv_head, :context_len].astype(np.float32, copy=False)
            if temporal_reuse_now and not temporal_incremental:
                # Frozen between rescans: same pages, same pending metadata.
                index_cache[kv_head] = temporal_index_cache[kv_head]["index"]
                continue
            build_entry = page_index_build_cache.get(int(kv_head))
            if (
                build_entry is not None
                and int(build_entry["dynamic_start"]) == int(dynamic_start)
                and int(build_entry["sealed_end"]) == int(sealed_end_geom)
                and str(args.selector_mode) != "routed"
                and not bool(args.lookahead_diagnostic)
            ):
                # Same sealed prefix and seed => bit-identical pages; only the
                # pending/indexed extent moved with the context.
                index_cache[kv_head] = dataclasses.replace(
                    build_entry["index"], indexed_end=int(indexed_end)
                )
                if temporal_reuse_max_stale > 0 and not temporal_reuse_now:
                    temporal_index_cache[kv_head] = {
                        "position": int(position),
                        "index": index_cache[kv_head],
                    }
                continue
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
            if bool(args.global_pq_codebook):
                _apply_global_pq_codebook(
                    index_cache[kv_head],
                    keys_np,
                    dynamic_start=int(dynamic_start),
                    subvecs=int(args.subvecs),
                    subbits=int(args.global_pq_subbits) or int(args.subbits),
                    kmeans_iters=int(args.kmeans_iters),
                    seed=4099 + 2027 * int(kv_head),
                    sample_rows=int(args.global_pq_sample_rows),
                )
            if str(args.selector_mode) != "routed":
                page_index_build_cache[int(kv_head)] = {
                    "dynamic_start": int(dynamic_start),
                    "sealed_end": int(sealed_end_geom),
                    "index": index_cache[kv_head],
                }
            if temporal_reuse_max_stale > 0 and not temporal_reuse_now:
                temporal_index_cache[kv_head] = {
                    "position": int(position),
                    "index": index_cache[kv_head],
                }
            if bool(args.lookahead_diagnostic):
                # Query-independent K-PQ reconstruction residual norms, the
                # sidecar stat the hardware K-delta bound would store per token.
                resid_norm = np.zeros((context_len,), dtype=np.float64)
                for page in index_cache[kv_head].pages:
                    start = int(page.start)
                    size = int(page.size)
                    khat = _page_full_reconstruct(page)
                    diff = keys_np[start : start + size].astype(np.float64, copy=False) - khat.astype(np.float64, copy=False)
                    resid_norm[start : start + size] = np.linalg.norm(diff, axis=1)
                k_resid_norm_cache[kv_head] = resid_norm

        dense_heads: dict[int, np.ndarray] = {}
        selected_heads: dict[tuple[str, str, str, float, str], dict[int, np.ndarray]] = defaultdict(dict)
        head_choices: dict[tuple[str, str, str, float, str], list[dict[str, object]]] = defaultdict(list)

        for head in heads:
            kv_head = int(trace.kv_head_for(int(head)))
            index = index_cache[kv_head]
            keys_np = trace.keys[kv_head, :context_len].astype(np.float32, copy=False)
            values_np = trace.values[kv_head, :context_len].astype(np.float32, copy=False)
            query_np = trace.queries[int(head), int(qidx)].astype(np.float32, copy=False)
            scores_np, _true_probs, dense_head = dense_attention_output(keys_np, values_np, query_np)
            dense_heads[int(head)] = dense_head

            pending = list(range(max(0, int(index.pending_start)), max(0, min(int(index.indexed_end), context_len))))
            if temporal_reuse_now and not temporal_incremental:
                # Frozen mode: everything unsealed at the last rescan (old
                # pending, old suffix, and tokens appended since) stays
                # resident until the next rescan; exact-K reads for it are
                # charged via the selected count as usual.
                pending = list(range(max(0, int(index.pending_start)), context_len))
            base = unique_tokens(
                static_tokens(position, int(args.static_prefix), int(args.static_suffix)) + pending,
                context_len=context_len,
            )
            max_k_budget = max(k_budgets)
            query_t = torch.as_tensor(query_np, dtype=torch.float32, device=device)
            selector_coverage = 1.0
            if temporal_reuse_now and int(head) in temporal_ranked_cache:
                cached_rank = temporal_ranked_cache[int(head)]
                ranked_cpu = cached_rank["ranked_cpu"]
                ranked_scores_cpu = cached_rank["ranked_scores_cpu"]
                # Reuse charges rereading the stored stale PQ logit row
                # (2B/token over the scanned extent) plus the ranked candidate
                # index list (4B each) instead of the codebook+code fullscan.
                selector_mb = float(
                    int(ranked_scores_cpu.shape[0]) * 2 + int(ranked_cpu.shape[0]) * 4
                ) / MB
                chosen_nprobe = 0
                if temporal_incremental:
                    # Score pages sealed since the last rescan with the
                    # current query and merge them into the stale ranking;
                    # charge their codes+codebooks only.
                    cached_sealed_end = int(cached_rank.get("sealed_end", 0))
                    new_tok_chunks: list[np.ndarray] = []
                    new_score_chunks: list[np.ndarray] = []
                    for page in index.pages:
                        if int(page.start) < cached_sealed_end:
                            continue
                        p_tokens, p_scores = pq_page_scores(query_t, page)
                        # tensor.numpy() arrays come from torch's bundled numpy
                        # in this venv and cannot mix with native numpy ops;
                        # round-trip through tolist.
                        new_tok_chunks.append(
                            np.fromiter(
                                (int(x) for x in p_tokens.detach().cpu().tolist()),
                                dtype=np.int64,
                                count=int(p_tokens.numel()),
                            )
                        )
                        new_score_chunks.append(
                            np.fromiter(
                                (float(x) for x in p_scores.detach().cpu().tolist()),
                                dtype=np.float32,
                                count=int(p_scores.numel()),
                            )
                        )
                        selector_mb += float(
                            int(page.codebooks.numel()) * int(args.key_bytes)
                            + int(page.codes.numel()) * (1 if int(args.subbits) <= 8 else 2)
                        ) / MB
                    if new_tok_chunks:
                        merged_tokens = np.concatenate([np.asarray(ranked_cpu, dtype=np.int64)] + new_tok_chunks)
                        merged_scores = np.concatenate(
                            [np.asarray(ranked_scores_cpu, dtype=np.float32)] + new_score_chunks
                        )
                        merge_order = np.argsort(-merged_scores.astype(np.float64, copy=False), kind="stable")
                        ranked_cpu = merged_tokens[merge_order]
                        ranked_scores_cpu = merged_scores[merge_order]
                        cached_rank["ranked_cpu"] = ranked_cpu
                        cached_rank["ranked_scores_cpu"] = ranked_scores_cpu
                        cached_rank["sealed_end"] = int(sealed_end_geom)
            elif str(args.selector_mode) in {"fullscan", "routed"}:
                ranked_t, ranked_scores_t, _selector_seconds, selector_mb, chosen_nprobe = rank_paged_pq(
                    query_t,
                    index,
                    mode=str(args.selector_mode),
                    selector_backend="torch",
                    nprobes=nprobes,
                    budget=int(max_k_budget),
                    key_bytes=int(args.key_bytes),
                    subbits=int(args.subbits),
                )
                ranked_cpu = ranked_t.detach().cpu().numpy().astype(np.int64, copy=False)
                ranked_scores_cpu = ranked_scores_t.detach().cpu().numpy().astype(np.float32, copy=False)
                if bool(args.global_pq_codebook) and index.pages:
                    # One shared codebook read per head-query + per-token codes.
                    global_bits = int(args.global_pq_subbits) or int(args.subbits)
                    selector_mb = float(
                        int(index.pages[0].codebooks.numel()) * int(args.key_bytes)
                        + sum(
                            int(p.codes.numel()) * (1 if global_bits <= 8 else 2)
                            for p in index.pages
                        )
                    ) / MB
                if temporal_reuse_max_stale > 0:
                    temporal_ranked_cache[int(head)] = {
                        "ranked_cpu": np.fromiter(
                            (int(x) for x in ranked_cpu.tolist()), dtype=np.int64, count=len(ranked_cpu)
                        ),
                        "ranked_scores_cpu": np.fromiter(
                            (float(x) for x in ranked_scores_cpu.tolist()),
                            dtype=np.float32,
                            count=len(ranked_scores_cpu),
                        ),
                        "sealed_end": int(sealed_end_geom),
                    }
            elif str(args.selector_mode) == "quest":
                ranked_cpu, ranked_scores_cpu, selector_mb, chosen_nprobe, selector_coverage = _rank_quest_pages(
                    keys_np=keys_np,
                    query_np=query_np,
                    index=index,
                    rank=int(args.quest_rank),
                    key_bytes=int(args.key_bytes),
                    index_bytes=int(args.selector_index_bytes),
                )
            elif str(args.selector_mode) == "quest_pq":
                ranked_cpu, ranked_scores_cpu, selector_mb, chosen_nprobe, selector_coverage = _rank_quest_pq(
                    query=query_t,
                    keys_np=keys_np,
                    query_np=query_np,
                    index=index,
                    rank=int(args.quest_rank),
                    nprobes=nprobes,
                    budget=int(max_k_budget),
                    key_bytes=int(args.key_bytes),
                    subbits=int(args.subbits),
                    index_bytes=int(args.selector_index_bytes),
                )
            else:
                raise ValueError(f"unknown selector_mode: {args.selector_mode}")

            page_unscanned_mask: np.ndarray | None = None
            page_centroid_logit: np.ndarray | None = None
            pages_scanned_count = len(index.pages)
            if page_scan_frac < 1.0 and len(index.pages) > 1:
                # ranked_cpu can arrive as an object-dtype array in this venv
                # (torch->numpy interop); coerce before fancy indexing.
                ranked_cpu = np.fromiter((int(x) for x in ranked_cpu.tolist()), dtype=np.int64, count=len(ranked_cpu))
                ranked_scores_cpu = np.fromiter(
                    (float(x) for x in ranked_scores_cpu.tolist()), dtype=np.float32, count=len(ranked_scores_cpu)
                )
                scores_full = np.full((context_len,), -np.inf, dtype=np.float64)
                scores_full[ranked_cpu] = ranked_scores_cpu.astype(np.float64, copy=False)
                page_spans = [(int(p.start), int(p.start) + int(p.size)) for p in index.pages]
                page_max = np.asarray([float(np.max(scores_full[s:e])) for s, e in page_spans])
                page_mean = np.asarray([float(np.mean(scores_full[s:e])) for s, e in page_spans])
                page_sizes = np.asarray([e - s for s, e in page_spans], dtype=np.int64)
                indexed_total = int(page_sizes.sum())
                order_pages = np.argsort(-page_max, kind="stable")
                target_tokens = float(page_scan_frac) * float(indexed_total)
                cum = np.cumsum(page_sizes[order_pages])
                pages_scanned_count = int(np.searchsorted(cum, target_tokens, side="left")) + 1
                pages_scanned_count = min(pages_scanned_count, len(index.pages))
                scanned_pages = set(order_pages[:pages_scanned_count].tolist())
                coarse_block = max(1, int(args.page_coarse_block))
                page_unscanned_mask = np.zeros((context_len,), dtype=bool)
                page_centroid_logit = np.zeros((context_len,), dtype=np.float32)
                coarse_block_count = 0
                for pid, (s, e) in enumerate(page_spans):
                    if pid not in scanned_pages:
                        page_unscanned_mask[s:e] = True
                        for bs in range(s, e, coarse_block):
                            be = min(e, bs + coarse_block)
                            # Match mixed_scores' tail logit scaling (raw/sqrt(d)).
                            page_centroid_logit[bs:be] = np.float32(
                                np.mean(scores_full[bs:be]) / math.sqrt(float(trace.head_dim))
                            )
                            coarse_block_count += 1
                if not bool(np.any(page_unscanned_mask)):
                    page_unscanned_mask = None
                    page_centroid_logit = None
                else:
                    keep = ~page_unscanned_mask[ranked_cpu]
                    ranked_cpu = ranked_cpu[keep]
                    ranked_scores_cpu = ranked_scores_cpu[keep]
                    # Page-level MBR bounds for ordering + block-level mean
                    # khat/vhat stats for the coarse tier of unscanned pages.
                    stats_bytes = float(len(index.pages) * 2 * int(trace.head_dim) * int(args.key_bytes))
                    stats_bytes += float(coarse_block_count * 2 * int(trace.head_dim) * int(args.key_bytes))
                    scanned_bytes = float(
                        sum(
                            int(index.pages[pid].codebooks.numel()) * int(args.key_bytes)
                            + int(index.pages[pid].codes.numel()) * (1 if int(args.subbits) <= 8 else 2)
                            for pid in scanned_pages
                        )
                    )
                    selector_mb = float(stats_bytes + scanned_bytes) / MB
            all_tokens = np.arange(context_len, dtype=np.int64)
            vhat_all, _compressed_v_mb, _fallback_v_mb = _vpq_values_for_tokens(
                index=index,
                values_np=values_np,
                tokens=all_tokens,
                subbits=int(args.subbits),
                value_subvecs=int(args.value_subvecs),
                value_subbits=int(args.value_subbits),
                value_bytes=int(args.value_bytes),
            )
            residual = values_np.astype(np.float32, copy=False) - vhat_all.astype(np.float32, copy=False)
            scores_lo_np: np.ndarray | None = None
            residual_lo: np.ndarray | None = None
            if precision_k_hi_frac < 1.0:
                keys_lo = _quantize_rows_symmetric(keys_np, precision_lo_bits)
                scores_lo_np = (
                    (keys_lo @ query_np.astype(np.float32, copy=False))
                    / math.sqrt(float(trace.head_dim))
                ).astype(np.float32, copy=False)
            precision_v_lo_err: np.ndarray | None = None
            if precision_v_hi_frac < 1.0:
                values_lo = _quantize_rows_symmetric(values_np, precision_lo_bits)
                residual_lo = (values_lo - vhat_all.astype(np.float32, copy=False)).astype(np.float32, copy=False)
                # Per-token squared error of the MSB-plane read, comparable to
                # code_error: an int8 exact read only pays off where it beats
                # the V-PQ reconstruction it would replace.
                precision_v_lo_err = np.sum(
                    np.square(values_np.astype(np.float64, copy=False) - values_lo.astype(np.float64, copy=False)),
                    axis=1,
                )
            la_x_factors: dict[str, np.ndarray] | None = None
            la_v_norm: np.ndarray | None = None
            la_v_resid_norm: np.ndarray | None = None
            if bool(args.lookahead_diagnostic):
                la_x_factors = _lookahead_x_factors(
                    q_norm=float(np.linalg.norm(query_np.astype(np.float64, copy=False))),
                    k_resid_norm=k_resid_norm_cache[kv_head],
                    head_dim=int(trace.head_dim),
                )
                la_v_norm = np.maximum(
                    np.linalg.norm(values_np.astype(np.float64, copy=False), axis=1),
                    np.linalg.norm(vhat_all.astype(np.float64, copy=False), axis=1),
                )
                la_v_resid_norm = np.linalg.norm(residual.astype(np.float64, copy=False), axis=1)
            code_error = value_vpq_code_stat_risk(
                index=index,
                values_np=values_np,
                residual=residual,
                subbits=int(args.subbits),
                value_subvecs=int(args.value_subvecs),
                value_subbits=int(args.value_subbits),
                sensitivity=None,
            )
            # Page pruning coarse-tiers only the tail LOGITS of unscanned
            # pages (block-mean khat logits). Per-token V-PQ values and the
            # exact-V rule are untouched: V codes are ~1B/token, not the scan
            # cost; collapsing V directions to block means was strongly
            # negative (relL2 ~0.13-0.18 at 32k).
            vhat_for_base = vhat_all
            actual_value_subbits = int(args.value_subbits) if int(args.value_subbits) > 0 else int(args.subbits)
            actual_value_subvecs = int(args.value_subvecs) if int(args.value_subvecs) > 0 else int(args.subvecs)
            code_bytes = 1 if actual_value_subbits <= 8 else 2
            metadata_mb = (
                float(context_len * actual_value_subvecs * code_bytes)
                + float(len(index.pages) * actual_value_subvecs * (1 << actual_value_subbits) * int(args.code_stat_bytes))
            ) / MB
            v_pq_codebook_mb = float(
                len(index.pages)
                * actual_value_subvecs
                * (1 << actual_value_subbits)
                * (int(trace.head_dim) // max(1, actual_value_subvecs))
                * int(args.value_bytes)
            ) / MB

            v_mb_by_idx: list[float] = []
            for v_budget in v_budgets:
                exact_count = max(0, min(int(v_budget), int(context_len)))
                exact_v_mb = float(exact_count * int(trace.head_dim) * int(args.value_bytes)) / MB
                compressed_v_codes_mb = float(max(0, context_len - exact_count) * actual_value_subvecs * code_bytes) / MB
                v_mb_by_idx.append(exact_v_mb + v_pq_codebook_mb + compressed_v_codes_mb + metadata_mb)

            def v_path_mb_for_exact_reads(exact_reads: int) -> float:
                reads = max(0, min(int(exact_reads), int(context_len)))
                exact_v_mb = float(reads * int(trace.head_dim) * int(args.value_bytes)) / MB
                compressed_v_codes_mb = float(max(0, context_len - reads) * actual_value_subvecs * code_bytes) / MB
                return float(exact_v_mb + v_pq_codebook_mb + compressed_v_codes_mb + metadata_mb)

            for score_proxy_variant in score_proxy_variants:
                variant_ranked_cpu, variant_ranked_scores_cpu, score_proxy_extra_mb, score_proxy_meta = apply_score_proxy_variant(
                    variant=str(score_proxy_variant),
                    index=index,
                    keys_np=keys_np,
                    query_np=query_np,
                    ranked_cpu=ranked_cpu,
                    ranked_scores_cpu=ranked_scores_cpu,
                    key_bytes=int(args.key_bytes),
                    metadata_bytes=int(args.code_stat_bytes),
                    kmeans_iters=int(args.kmeans_iters),
                    seed=2025 + 4093 * int(kv_head) + 31 * int(head) + int(context_len),
                )

                k_mb_by_idx: list[float] = []
                selected_counts_by_idx: list[int] = []
                k_lo_counts_by_idx: list[int] = []
                selected_by_k: dict[int, np.ndarray] = {}
                probs_by_k: dict[int, np.ndarray] = {}
                scores_by_k: dict[int, np.ndarray] = {}
                base_output_by_k: dict[int, np.ndarray] = {}
                calibration_extra_mb_by_idx: list[float] = []
                calibration_probe_count_by_idx: list[int] = []
                for ki, k_budget in enumerate(k_budgets):
                    selected_cpu = _selected_for_budget(
                        base=base,
                        ranked_cpu=variant_ranked_cpu,
                        budget=int(k_budget),
                        context_len=context_len,
                    )
                    selected_counts_by_idx.append(int(selected_cpu.size))
                    selected_by_k[ki] = selected_cpu
                    score_vec, _missing, _scale, _bias, calibration_extra_mb, calibration_probe_count = mixed_scores_for_variant(
                        variant=str(score_proxy_variant),
                        context_len=context_len,
                        selected_cpu=selected_cpu,
                        ranked_cpu=variant_ranked_cpu,
                        ranked_scores_cpu=variant_ranked_scores_cpu,
                        exact_scores_np=scores_np,
                        query_dim=int(trace.head_dim),
                        calibrate=str(args.tail_score_calibration) == "affine_selected",
                        key_bytes=int(args.key_bytes),
                    )
                    k_lo_tokens_count = 0
                    if precision_k_hi_frac < 1.0 and scores_lo_np is not None:
                        lo_tokens = _precision_lo_tokens(
                            base=base,
                            ranked_cpu=variant_ranked_cpu,
                            budget=int(k_budget),
                            context_len=context_len,
                            hi_frac=precision_k_hi_frac,
                        )
                        if lo_tokens.size:
                            score_vec = score_vec.astype(np.float32, copy=True)
                            score_vec[lo_tokens] = scores_lo_np[lo_tokens]
                        k_lo_tokens_count = int(lo_tokens.size)
                    k_lo_counts_by_idx.append(int(k_lo_tokens_count))
                    if page_unscanned_mask is not None and page_centroid_logit is not None:
                        score_vec = score_vec.astype(np.float32, copy=True)
                        score_vec[page_unscanned_mask] = page_centroid_logit[page_unscanned_mask]
                    probs = np.exp(score_vec - float(np.max(score_vec)))
                    probs /= max(float(probs.sum()), 1e-20)
                    scores_by_k[ki] = score_vec.astype(np.float32, copy=False)
                    probs_by_k[ki] = probs.astype(np.float64, copy=False)
                    base_output_by_k[ki] = (
                        probs.astype(np.float64, copy=False) @ vhat_for_base.astype(np.float64, copy=False)
                    ).astype(np.float32, copy=False)
                    exact_key_mb = float(selected_cpu.size * int(trace.head_dim) * int(args.key_bytes)) / MB
                    if k_lo_tokens_count > 0:
                        exact_key_mb -= float(
                            k_lo_tokens_count
                            * int(trace.head_dim)
                            * (int(args.key_bytes) - int(precision_lo_bytes))
                        ) / MB
                    calibration_extra_mb_by_idx.append(float(calibration_extra_mb))
                    calibration_probe_count_by_idx.append(int(calibration_probe_count))
                    k_mb_by_idx.append(float(selector_mb) + float(score_proxy_extra_mb) + exact_key_mb + float(calibration_extra_mb))

                for v_selection_rule_raw in v_selection_rules:
                    outputs: dict[tuple[int, int], np.ndarray] = {}
                    exact_mask_by_pair: dict[tuple[int, int], np.ndarray] = {}
                    v_lo_by_pair: dict[tuple[int, int], int] = {}
                    v_dropped_by_pair: dict[tuple[int, int], int] = {}
                    v_state_mb_by_idx: list[float] = []
                    v_rule_meta_by_idx: list[dict[str, object]] = []
                    v_rule_meta_by_pair: dict[tuple[int, int], dict[str, object]] = {}
                    v_mb_by_pair: dict[tuple[int, int], float] = {}
                    v_state_mb_by_pair: dict[tuple[int, int], float] = {}
                    step_mb_by_pair: dict[tuple[int, int], float] = {}
                    for vi, v_budget in enumerate(v_budgets):
                        exact_count = max(0, min(int(v_budget), int(context_len)))
                        v_state_mb_by_idx.append(
                            v_selection_state_mb(
                                rule=str(v_selection_rule_raw),
                                exact_count=int(exact_count),
                                index_bytes=int(args.selector_index_bytes),
                                logit_bytes=int(args.survivor_logit_bytes),
                                include_state=bool(args.include_v_selection_state_in_step_mb),
                            )
                        )
                        v_rule_meta_by_idx.append({})
                    rule_name_lower = str(v_selection_rule_raw).strip().lower()
                    two_pass_active = rule_name_lower.startswith("two_pass_risk")
                    two_pass_cutoffs: list[float] = []
                    two_pass_cutoff_ranks: list[int] = []
                    log_code_error: np.ndarray | None = None
                    if two_pass_active:
                        # Pass 1 models the selector fullscan stream: PQ logits for
                        # non-resident tokens plus exact logits for resident base
                        # tokens, ranked in unnormalized log-risk space
                        # (2*logit + log V-error). Only the per-V-budget cutoff
                        # value survives to pass 2, so tile-local commits need no
                        # survivor list and no stored probability row.
                        two_pass_rank_mult = _parse_fraction_suffix(rule_name_lower, "f", 1.0)
                        pass1_scores, _p1_missing, _p1_scale, _p1_bias = mixed_scores(
                            context_len=context_len,
                            selected_cpu=np.asarray(base, dtype=np.int64),
                            ranked_cpu=variant_ranked_cpu,
                            ranked_scores_cpu=variant_ranked_scores_cpu,
                            exact_scores_np=scores_np,
                            query_dim=int(trace.head_dim),
                            calibrate=str(args.tail_score_calibration) == "affine_selected",
                        )
                        code_error_f64 = np.asarray(code_error, dtype=np.float64).reshape(-1)
                        log_code_error = np.full((context_len,), -np.inf, dtype=np.float64)
                        positive_err = code_error_f64 > 0.0
                        log_code_error[positive_err] = np.log(code_error_f64[positive_err])
                        pass1_log_risk = 2.0 * np.asarray(pass1_scores, dtype=np.float64) + log_code_error
                        pass1_finite_sorted = np.sort(pass1_log_risk[np.isfinite(pass1_log_risk)])[::-1]
                        for v_budget in v_budgets:
                            exact_count = max(0, min(int(v_budget), int(context_len)))
                            cutoff_rank = min(
                                int(context_len),
                                int(round(float(exact_count) * float(two_pass_rank_mult))),
                            )
                            if exact_count <= 0 or cutoff_rank <= 0 or pass1_finite_sorted.size == 0:
                                two_pass_cutoffs.append(float("inf"))
                            else:
                                two_pass_cutoffs.append(
                                    float(pass1_finite_sorted[min(int(cutoff_rank), int(pass1_finite_sorted.size)) - 1])
                                )
                            two_pass_cutoff_ranks.append(int(cutoff_rank))
                    la_active = bool(args.lookahead_diagnostic) and rule_name_lower in {
                        "",
                        "global",
                        "global_residual_risk",
                        "residual_risk",
                    }
                    la_k_s0: dict[str, dict[int, float]] = {v: {} for v in ("cs", "rms1", "rms2", "rms4")}
                    la_k_s1: dict[str, dict[int, float]] = {v: {} for v in ("cs", "rms1", "rms2", "rms4")}
                    la_k_t0: dict[int, float] = {}
                    la_k_t1: dict[int, float] = {}
                    la_k_band_tokens: dict[int, int] = {}
                    la_v_band_abs: dict[tuple[int, int], float] = {}
                    la_v_band_sq: dict[tuple[int, int], float] = {}
                    la_v_band_tokens: dict[tuple[int, int], int] = {}
                    for ki in range(len(k_budgets)):
                        probs = probs_by_k[ki]
                        risk_scores = (probs * probs) * code_error
                        la_masks: list[np.ndarray] = []
                        if la_active and ki + 1 < len(k_budgets):
                            # Tokens whose logits upgrade from PQ to exact when the
                            # K budget escalates one step.
                            band = np.setdiff1d(selected_by_k[ki + 1], selected_by_k[ki], assume_unique=False)
                            la_k_band_tokens[ki] = int(band.size)
                            if band.size:
                                p_band = probs[band]
                                vn_band = la_v_norm[band]
                                for variant, x_all in la_x_factors.items():
                                    f_band = np.expm1(np.minimum(x_all[band], 50.0))
                                    la_k_s0[variant][ki] = float(np.sum(p_band * f_band))
                                    la_k_s1[variant][ki] = float(np.sum(p_band * f_band * vn_band))
                                px_band = p_band * la_x_factors["rms1"][band]
                                la_k_t0[ki] = float(np.sum(px_band * px_band))
                                la_k_t1[ki] = float(np.sum(px_band * px_band * vn_band * vn_band))
                            else:
                                for variant in la_k_s0:
                                    la_k_s0[variant][ki] = 0.0
                                    la_k_s1[variant][ki] = 0.0
                                la_k_t0[ki] = 0.0
                                la_k_t1[ki] = 0.0
                        two_pass_true_log_risk: np.ndarray | None = None
                        if two_pass_active:
                            two_pass_true_log_risk = (
                                2.0 * scores_by_k[ki].astype(np.float64, copy=False) + log_code_error
                            )
                        streaming_v_results: dict[int, tuple[np.ndarray, int]] | None = None
                        streaming_v_block = _v_selection_block_size(str(v_selection_rule_raw), int(args.v_local_block_size))
                        if str(v_selection_rule_raw).strip().lower().startswith("streaming_global_risk"):
                            streaming_v_results = streaming_topk_masks_and_reads_for_counts(
                                risk_scores,
                                counts=[
                                    max(0, min(int(v_budget), int(context_len)))
                                    for v_budget in v_budgets
                                ],
                                block_size=int(streaming_v_block),
                            )
                        for vi, v_budget in enumerate(v_budgets):
                            exact_count = max(0, min(int(v_budget), int(context_len)))
                            if two_pass_active:
                                if exact_count >= int(context_len):
                                    exact_mask = np.ones((int(context_len),), dtype=bool)
                                else:
                                    exact_mask = np.isfinite(two_pass_true_log_risk) & (
                                        two_pass_true_log_risk >= float(two_pass_cutoffs[vi])
                                    )
                                rule_meta = {
                                    "v_selection_rule": str(rule_name_lower),
                                    "v_selection_block_size": 0,
                                    "v_selection_exact_target": int(exact_count),
                                    "v_selection_exact_reads": int(np.count_nonzero(exact_mask)),
                                    "v_selection_cutoff_rank": int(two_pass_cutoff_ranks[vi]),
                                }
                            elif streaming_v_results is None:
                                exact_mask, rule_meta = exact_v_mask_for_rule(
                                    rule=str(v_selection_rule_raw),
                                    risk_scores=risk_scores,
                                    value_scores=code_error,
                                    exact_count=int(exact_count),
                                    block_size=int(args.v_local_block_size),
                                )
                            else:
                                exact_mask, exact_reads = streaming_v_results[int(exact_count)]
                                rule_meta = {
                                    "v_selection_rule": f"streaming_global_risk_b{int(streaming_v_block)}",
                                    "v_selection_block_size": int(streaming_v_block),
                                    "v_selection_exact_target": int(exact_count),
                                    "v_selection_exact_reads": int(exact_reads),
                                }
                            if ki == 0:
                                v_rule_meta_by_idx[vi] = rule_meta
                            actual_exact_reads = int(rule_meta.get("v_selection_exact_reads", int(np.count_nonzero(exact_mask))))
                            actual_v_mb = v_path_mb_for_exact_reads(actual_exact_reads)
                            if two_pass_active:
                                # Pass 1 re-reads per-token V-PQ error metadata to
                                # estimate the risk cutoffs before pass 2 commits.
                                actual_v_mb += float(metadata_mb)
                            v_lo_reads = 0
                            v_dropped_reads = 0
                            v_hi_mask: np.ndarray | None = None
                            v_lo_mask: np.ndarray | None = None
                            if precision_v_hi_frac < 1.0 and residual_lo is not None and bool(np.any(exact_mask)):
                                exact_idx = np.flatnonzero(exact_mask)
                                hi_target = int(math.ceil(float(exact_idx.size) * precision_v_hi_frac))
                                risk_order = exact_idx[
                                    np.argsort(-risk_scores[exact_idx].astype(np.float64, copy=False), kind="stable")
                                ]
                                lo_candidates = risk_order[hi_target:]
                                # Commit the MSB-plane read only where it beats the
                                # V-PQ reconstruction it replaces; otherwise skip
                                # the read and keep the V-PQ value (per-token
                                # stored int8-error stat vs code-error stat).
                                commit = lo_candidates[
                                    precision_v_lo_err[lo_candidates]
                                    < code_error[lo_candidates].astype(np.float64, copy=False)
                                ]
                                v_hi_mask = np.zeros((int(context_len),), dtype=bool)
                                v_lo_mask = np.zeros((int(context_len),), dtype=bool)
                                v_hi_mask[risk_order[:hi_target]] = True
                                v_lo_mask[commit] = True
                                v_lo_reads = int(commit.size)
                                v_dropped_reads = int(lo_candidates.size - commit.size)
                                effective_reads = int(hi_target + v_lo_reads)
                                actual_v_mb = v_path_mb_for_exact_reads(effective_reads)
                                if two_pass_active:
                                    actual_v_mb += float(metadata_mb)
                                actual_v_mb -= float(
                                    v_lo_reads
                                    * int(trace.head_dim)
                                    * (int(args.value_bytes) - int(precision_lo_bytes))
                                ) / MB
                                exact_mask = v_hi_mask | v_lo_mask
                            v_lo_by_pair[(ki, vi)] = int(v_lo_reads)
                            v_dropped_by_pair[(ki, vi)] = int(v_dropped_reads)
                            state_mb = float(v_state_mb_by_idx[vi])
                            v_rule_meta_by_pair[(ki, vi)] = rule_meta
                            v_mb_by_pair[(ki, vi)] = float(actual_v_mb)
                            v_state_mb_by_pair[(ki, vi)] = float(state_mb)
                            step_mb_by_pair[(ki, vi)] = float(k_mb_by_idx[ki] + actual_v_mb + state_mb)
                            if v_hi_mask is not None and v_lo_mask is not None:
                                outputs[(ki, vi)] = output_from_base_and_split_masks(
                                    base_output=base_output_by_k[ki],
                                    probs=probs,
                                    residual=residual,
                                    residual_lo=residual_lo,
                                    hi_mask=v_hi_mask,
                                    lo_mask=v_lo_mask,
                                )
                            else:
                                outputs[(ki, vi)] = output_from_base_and_exact_mask(
                                    base_output=base_output_by_k[ki],
                                    probs=probs,
                                    residual=residual,
                                    exact_mask=exact_mask,
                                )
                            if temporal_cache_stats or gqa_union_stats:
                                exact_mask_by_pair[(ki, vi)] = np.asarray(exact_mask, dtype=bool)
                            if la_active:
                                la_masks.append(np.asarray(exact_mask, dtype=bool))
                        if la_active:
                            for vi in range(len(v_budgets) - 1):
                                band_mask = la_masks[vi + 1] & ~la_masks[vi]
                                la_v_band_tokens[(ki, vi)] = int(np.count_nonzero(band_mask))
                                pr_band = probs[band_mask] * la_v_resid_norm[band_mask]
                                la_v_band_abs[(ki, vi)] = float(np.sum(pr_band))
                                la_v_band_sq[(ki, vi)] = float(np.sum(pr_band * pr_band))
                    canonical_v_rule = str(v_rule_meta_by_idx[0].get("v_selection_rule", v_selection_rule_raw))
                    la_k_bound: dict[str, dict[tuple[int, int], float]] = {}
                    la_v_bound: dict[str, dict[tuple[int, int], float]] = {}
                    if la_active:
                        out_norm = {
                            pair: float(np.linalg.norm(out.astype(np.float64, copy=False)))
                            for pair, out in outputs.items()
                        }
                        for variant in la_k_s0:
                            la_k_bound[variant] = {}
                            for ki in la_k_band_tokens:
                                s0 = la_k_s0[variant][ki]
                                s1 = la_k_s1[variant][ki]
                                for vi in range(len(v_budgets)):
                                    norm_o = out_norm[(ki, vi)]
                                    if s0 >= 1.0:
                                        la_k_bound[variant][(ki, vi)] = float("inf")
                                        continue
                                    abs_bound = (s1 + s0 * norm_o) / max(1.0 - s0, 1e-9)
                                    la_k_bound[variant][(ki, vi)] = float(
                                        abs_bound / max(norm_o - abs_bound, 1e-20)
                                    )
                        la_v_strict: dict[tuple[int, int], float] = {}
                        for (ki, vi), abs_bound in la_v_band_abs.items():
                            norm_o = out_norm[(ki, vi)]
                            la_v_strict[(ki, vi)] = float(abs_bound / max(norm_o - abs_bound, 1e-20))
                        for variant in ("cs", "rms1", "rms2", "rms4"):
                            la_v_bound[variant] = la_v_strict
                        for variant in ("var1", "var2", "var4"):
                            lam = float(variant[3:])
                            la_k_bound[variant] = {}
                            la_v_bound[variant] = {}
                            for ki in la_k_band_tokens:
                                t0_root = math.sqrt(max(la_k_t0[ki], 0.0))
                                t1_root = math.sqrt(max(la_k_t1[ki], 0.0))
                                for vi in range(len(v_budgets)):
                                    norm_o = out_norm[(ki, vi)]
                                    est = lam * (t1_root + t0_root * norm_o)
                                    la_k_bound[variant][(ki, vi)] = float(est / max(norm_o - est, 1e-20))
                            for (ki, vi), sq_sum in la_v_band_sq.items():
                                norm_o = out_norm[(ki, vi)]
                                est = lam * math.sqrt(max(sq_sum, 0.0))
                                la_v_bound[variant][(ki, vi)] = float(est / max(norm_o - est, 1e-20))

                    if oracle_rel_l2_targets:
                        for oracle_target in oracle_rel_l2_targets:
                            oracle = find_oracle_budget(
                                outputs=outputs,
                                dense=dense_head,
                                k_budgets=k_budgets,
                                v_budgets=v_budgets,
                                k_mb_by_idx=k_mb_by_idx,
                                v_mb_by_idx=v_mb_by_idx,
                                target_rel_l2=float(oracle_target),
                                step_mb_by_idx=step_mb_by_pair,
                            )
                            if not oracle:
                                continue
                            oracle_ki = int(oracle["oracle_ki"])
                            oracle_vi = int(oracle["oracle_vi"])
                            oracle_mb = float(oracle["oracle_step_MB_per_head"])
                            for start_strategy in start_strategies:
                                if str(start_strategy).startswith("temporal_prev"):
                                    continue
                                start_ki, start_vi = initial_budget_indices(
                                    strategy=str(start_strategy),
                                    context_len=int(context_len),
                                    ranked_scores_cpu=variant_ranked_scores_cpu,
                                    query_dim=int(trace.head_dim),
                                    k_budgets=k_budgets,
                                    v_budgets=v_budgets,
                                    previous_fraction=None,
                                )
                                start_mb = float(step_mb_by_pair[(start_ki, start_vi)])
                                oracle_rows.append(
                                    {
                                        "qidx": int(qidx),
                                        "position": int(position),
                                        "decode_length": int(decode_tokens),
                                        "head": int(head),
                                        "kv_head": int(kv_head),
                                        "selector_mode": str(args.selector_mode),
                                        "score_proxy_variant": str(score_proxy_variant),
                                        "score_proxy_detail": str(score_proxy_meta.get("score_proxy_detail", "")),
                                        "v_selection_rule": str(canonical_v_rule),
                                        "threshold_mode": str(args.threshold_mode),
                                        "threshold_reference_frac": float(args.threshold_reference_frac),
                                        "threshold_scale_shape": str(args.threshold_scale_shape),
                                        "threshold_min_scale": float(args.threshold_min_scale),
                                        "threshold_max_scale": float(args.threshold_max_scale),
                                        "start_strategy": str(start_strategy),
                                        "budget_mode": "relative" if k_budget_fracs else "absolute",
                                        "oracle_target_rel_l2": float(oracle_target),
                                        "start_ki": int(start_ki),
                                        "start_vi": int(start_vi),
                                        "start_k_budget": int(k_budgets[start_ki]),
                                        "start_v_budget": int(v_budgets[start_vi]),
                                        "start_selected_k_tokens": int(selected_counts_by_idx[start_ki]),
                                        "start_step_MB_per_head": float(start_mb),
                                        **oracle,
                                        "oracle_selected_k_tokens": int(selected_counts_by_idx[oracle_ki]),
                                        "start_minus_oracle_MB": float(start_mb - oracle_mb),
                                        "start_over_oracle_MB_ratio": float(start_mb / max(oracle_mb, 1e-12)),
                                        "start_k_over_oracle_ratio": float(
                                            int(k_budgets[start_ki]) / max(float(oracle["oracle_k_budget"]), 1.0)
                                        ),
                                        "start_v_over_oracle_ratio": float(
                                            int(v_budgets[start_vi]) / max(float(oracle["oracle_v_budget"]), 1.0)
                                        ),
                                        "start_covers_oracle_k": bool(int(start_ki) >= oracle_ki),
                                        "start_covers_oracle_v": bool(int(start_vi) >= oracle_vi),
                                        "start_covers_oracle_both": bool(int(start_ki) >= oracle_ki and int(start_vi) >= oracle_vi),
                                    }
                                )

                    for threshold in thresholds:
                        for policy in policies:
                            for start_strategy in start_strategies:
                                prev_key = (
                                    str(score_proxy_variant),
                                    str(canonical_v_rule),
                                    str(policy),
                                    float(threshold),
                                    str(start_strategy),
                                    int(head),
                                )
                                start_ki, start_vi = initial_budget_indices(
                                    strategy=str(start_strategy),
                                    context_len=int(context_len),
                                    ranked_scores_cpu=variant_ranked_scores_cpu,
                                    query_dim=int(trace.head_dim),
                                    k_budgets=k_budgets,
                                    v_budgets=v_budgets,
                                    previous_fraction=previous_fraction.get(prev_key),
                                )
                                la_test_log: list[tuple[int, int, bool, bool, float, float, float, float]] | None = (
                                    [] if la_active else None
                                )
                                frozen_budget = (
                                    temporal_budget_cache.get(prev_key)
                                    if temporal_budget_frozen and temporal_reuse_now
                                    else None
                                )
                                if frozen_budget is not None:
                                    ki = min(int(frozen_budget[0]), len(k_budgets) - 1)
                                    vi = min(int(frozen_budget[1]), len(v_budgets) - 1)
                                    steps = 0
                                    final_k_delta = 0.0
                                    final_v_delta = 0.0
                                    policy_trace = ["frozen_budget"]
                                else:
                                    ki, vi, steps, final_k_delta, final_v_delta, policy_trace = simulate_policy(
                                        outputs=outputs,
                                        k_budgets=k_budgets,
                                        v_budgets=v_budgets,
                                        policy=str(policy),
                                        threshold=float(threshold),
                                        k_mb_by_idx=k_mb_by_idx,
                                        v_mb_by_idx=v_mb_by_idx,
                                        context_len=int(context_len),
                                        threshold_mode=str(args.threshold_mode),
                                        threshold_reference_frac=float(args.threshold_reference_frac),
                                        threshold_scale_shape=str(args.threshold_scale_shape),
                                        threshold_min_scale=float(args.threshold_min_scale),
                                        threshold_max_scale=float(args.threshold_max_scale),
                                        start_ki=int(start_ki),
                                        start_vi=int(start_vi),
                                        step_mb_by_idx=step_mb_by_pair,
                                        test_log=la_test_log,
                                        deescalate=bool(args.budget_deescalate),
                                    )
                                if not temporal_reuse_now:
                                    temporal_budget_cache[prev_key] = (int(ki), int(vi))
                                previous_fraction[prev_key] = (
                                    float(k_budgets[ki]) / max(float(context_len), 1.0),
                                    float(v_budgets[vi]) / max(float(context_len), 1.0),
                                )
                                approx = outputs[(ki, vi)]
                                key = (str(score_proxy_variant), str(canonical_v_rule), policy, float(threshold), str(start_strategy))
                                selected_heads[key][int(head)] = approx
                                metric = _output_error_metrics(dense_head, approx)
                                dist_metric = attention_distribution_error_metrics(
                                    scores_np,
                                    _true_probs,
                                    scores_by_k[ki],
                                    probs_by_k[ki],
                                )
                                v_path_mb = float(v_mb_by_pair[(ki, vi)])
                                v_state_mb = float(v_state_mb_by_pair[(ki, vi)])
                                total_mb_no_state = float(k_mb_by_idx[ki] + v_path_mb)
                                total_mb = float(step_mb_by_pair[(ki, vi)])
                                v_rule_meta = v_rule_meta_by_pair[(ki, vi)]
                                la_fields: dict[str, object] = {}
                                if la_active and la_test_log is not None:
                                    # Hardware-style lookahead accounting along the
                                    # canonical trajectory: a confidence test either
                                    # certifies via a compressed-domain bound (free)
                                    # or pays the marginal band's exact reads; band
                                    # reads beyond the final accepted budget are
                                    # wasted lookahead.
                                    la_stat_mb = float(context_len * 2) / MB
                                    la_fields["la_stat_MB"] = la_stat_mb
                                    head_row_bytes = float(int(trace.head_dim))
                                    for variant in LOOKAHEAD_VARIANTS:
                                        k_tests = k_cert = k_false = 0
                                        v_tests = v_cert = v_false = 0
                                        k_charged: set[int] = set()
                                        v_charged: set[tuple[int, int]] = set()
                                        for tki, tvi, k_can, v_can, k_delta, v_delta, k_thr, v_thr in la_test_log:
                                            if k_can:
                                                k_tests += 1
                                                bound = (
                                                    float("inf")
                                                    if variant == "charge_all"
                                                    else la_k_bound[variant].get((tki, tvi), float("inf"))
                                                )
                                                if bound <= k_thr:
                                                    k_cert += 1
                                                    if k_delta > k_thr:
                                                        k_false += 1
                                                else:
                                                    k_charged.add(tki)
                                            if v_can:
                                                v_tests += 1
                                                bound = (
                                                    float("inf")
                                                    if variant == "charge_all"
                                                    else la_v_bound[variant].get((tki, tvi), float("inf"))
                                                )
                                                if bound <= v_thr:
                                                    v_cert += 1
                                                    if v_delta > v_thr:
                                                        v_false += 1
                                                else:
                                                    v_charged.add((tki, tvi))
                                        wasted_mb = 0.0
                                        for band_ki in k_charged:
                                            if band_ki + 1 > ki:
                                                wasted_mb += (
                                                    float(la_k_band_tokens.get(band_ki, 0))
                                                    * head_row_bytes
                                                    * float(int(args.key_bytes))
                                                ) / MB
                                        for band_ki, band_vi in v_charged:
                                            if not (band_ki == ki and band_vi + 1 <= vi):
                                                wasted_mb += (
                                                    float(la_v_band_tokens.get((band_ki, band_vi), 0))
                                                    * head_row_bytes
                                                    * float(int(args.value_bytes))
                                                ) / MB
                                        stat_extra = 0.0 if variant == "charge_all" else la_stat_mb
                                        la_fields[f"la_{variant}_wasted_MB"] = float(wasted_mb)
                                        la_fields[f"la_{variant}_hw_step_MB"] = float(total_mb + wasted_mb + stat_extra)
                                        la_fields[f"la_{variant}_k_tests"] = int(k_tests)
                                        la_fields[f"la_{variant}_k_certified"] = int(k_cert)
                                        la_fields[f"la_{variant}_k_false_certified"] = int(k_false)
                                        la_fields[f"la_{variant}_v_tests"] = int(v_tests)
                                        la_fields[f"la_{variant}_v_certified"] = int(v_cert)
                                        la_fields[f"la_{variant}_v_false_certified"] = int(v_false)
                                temporal_fields: dict[str, object] = {
                                    "temporal_reuse_max_stale": int(temporal_reuse_max_stale),
                                    "temporal_rescan": bool(not temporal_reuse_now),
                                    "temporal_stale_tokens": int(temporal_stale_tokens),
                                    "temporal_budget_frozen": bool(frozen_budget is not None),
                                }
                                if temporal_cache_stats:
                                    tkey = (
                                        int(head),
                                        str(score_proxy_variant),
                                        str(canonical_v_rule),
                                        str(policy),
                                        float(threshold),
                                        str(start_strategy),
                                    )
                                    t_k_set = np.asarray(selected_by_k[ki], dtype=np.int64)
                                    t_v_set = np.flatnonzero(exact_mask_by_pair[(ki, vi)]).astype(np.int64)
                                    t_prev = temporal_prev_sets.get(tkey)
                                    if t_prev is not None:
                                        t_gap = int(position) - int(t_prev["position"])
                                        t_k_new = int(np.setdiff1d(t_k_set, t_prev["k_set"]).size)
                                        t_v_new = int(np.setdiff1d(t_v_set, t_prev["v_set"]).size)
                                        temporal_fields.update(
                                            {
                                                "temporal_gap_tokens": int(t_gap),
                                                "temporal_k_tokens": int(t_k_set.size),
                                                "temporal_k_new_tokens": int(t_k_new),
                                                "temporal_k_prev_overlap_frac": float(
                                                    1.0 - float(t_k_new) / max(float(t_k_set.size), 1.0)
                                                ),
                                                "temporal_v_tokens": int(t_v_set.size),
                                                "temporal_v_new_tokens": int(t_v_new),
                                                "temporal_v_prev_overlap_frac": float(
                                                    1.0 - float(t_v_new) / max(float(t_v_set.size), 1.0)
                                                ),
                                            }
                                        )
                                    else:
                                        temporal_fields["temporal_gap_tokens"] = -1
                                    temporal_prev_sets[tkey] = {
                                        "position": int(position),
                                        "k_set": t_k_set,
                                        "v_set": t_v_set,
                                    }
                                if gqa_union_stats:
                                    gkey = (
                                        int(kv_head),
                                        str(score_proxy_variant),
                                        str(canonical_v_rule),
                                        str(policy),
                                        float(threshold),
                                        str(start_strategy),
                                    )
                                    g_k = np.asarray(selected_by_k[ki], dtype=np.int64)
                                    g_v = np.flatnonzero(exact_mask_by_pair[(ki, vi)]).astype(np.int64)
                                    acc = gqa_union_acc.setdefault(
                                        gkey,
                                        {"heads": 0, "k_sum": 0, "v_sum": 0, "k_sets": [], "v_sets": []},
                                    )
                                    acc["heads"] += 1
                                    acc["k_sum"] += int(g_k.size)
                                    acc["v_sum"] += int(g_v.size)
                                    acc["k_sets"].append(g_k)
                                    acc["v_sets"].append(g_v)
                                row = {
                                    "qidx": int(qidx),
                                    "position": int(position),
                                    "decode_length": int(decode_tokens),
                                    "head": int(head),
                                    "kv_head": int(kv_head),
                                    "selector_mode": str(args.selector_mode),
                                    "quest_rank": int(args.quest_rank),
                                    "chosen_nprobe": int(chosen_nprobe),
                                    "selector_coverage": float(selector_coverage),
                                    "score_proxy_variant": str(score_proxy_variant),
                                    "score_proxy_extra_MB": float(score_proxy_extra_mb),
                                    "score_proxy_detail": str(score_proxy_meta.get("score_proxy_detail", "")),
                                    "v_selection_rule": str(canonical_v_rule),
                                    "v_selection_block_size": int(v_rule_meta.get("v_selection_block_size", 0)),
                                    "v_exact_reads": int(v_rule_meta.get("v_selection_exact_reads", int(v_rule_meta.get("v_selection_exact_target", v_budgets[vi])))),
                                    "v_selection_state_MB": float(v_state_mb),
                                    "calibration_extra_MB": float(calibration_extra_mb_by_idx[ki]),
                                    "calibration_probe_tokens": int(calibration_probe_count_by_idx[ki]),
                                    "policy": str(policy),
                                    "threshold": float(threshold),
                                    "threshold_mode": str(args.threshold_mode),
                                    "threshold_reference_frac": float(args.threshold_reference_frac),
                                    "threshold_scale_shape": str(args.threshold_scale_shape),
                                    "threshold_min_scale": float(args.threshold_min_scale),
                                    "threshold_max_scale": float(args.threshold_max_scale),
                                    "start_strategy": str(start_strategy),
                                    "budget_mode": "relative" if k_budget_fracs else "absolute",
                                    "start_k_budget": int(k_budgets[start_ki]),
                                    "start_v_budget": int(v_budgets[start_vi]),
                                    "k_budget": int(k_budgets[ki]),
                                    "v_budget": int(v_budgets[vi]),
                                    "selected_k_tokens": int(selected_counts_by_idx[ki]),
                                    "precision_k_hi_frac": float(precision_k_hi_frac),
                                    "precision_v_hi_frac": float(precision_v_hi_frac),
                                    "precision_lo_bits": int(precision_lo_bits),
                                    "precision_k_lo_tokens": int(k_lo_counts_by_idx[ki]),
                                    "precision_v_lo_reads": int(v_lo_by_pair.get((ki, vi), 0)),
                                    "precision_v_dropped_reads": int(v_dropped_by_pair.get((ki, vi), 0)),
                                    "page_scan_frac": float(page_scan_frac),
                                    "pages_total": int(len(index.pages)),
                                    "pages_scanned": int(pages_scanned_count),
                                    "page_unscanned_tokens": int(
                                        np.count_nonzero(page_unscanned_mask)
                                        if page_unscanned_mask is not None
                                        else 0
                                    ),
                                    "iterations": int(steps),
                                    "final_k_delta": float(final_k_delta),
                                    "final_v_delta": float(final_v_delta),
                                    "selector_plus_exact_k_MB": float(k_mb_by_idx[ki]),
                                    "v_path_MB": float(v_path_mb),
                                    "step_MB_no_v_state_per_head": float(total_mb_no_state),
                                    "step_MB_with_v_state_per_head": float(total_mb),
                                    "step_MB_per_head": float(total_mb),
                                    "head_attention_relative_L2": float(metric["output_relative_l2"]),
                                    "head_attention_cosine": float(metric["output_cosine"]),
                                    "policy_trace": " | ".join(policy_trace),
                                }
                                row.update({f"score_proxy_meta_{k}": v for k, v in score_proxy_meta.items() if isinstance(v, (str, int, float, bool))})
                                row.update({key: float(value) for key, value in dist_metric.items()})
                                row.update(la_fields)
                                row.update(temporal_fields)
                                head_rows.append(row)
                                head_choices[key].append(row)

                                if la_active:
                                    for dv in lookahead_decision_variants:
                                        dv_test_log: list[tuple[int, int, bool, bool, float, float, float, float]] = []
                                        dki, dvi, dsteps, d_k_delta, d_v_delta, d_trace = simulate_policy(
                                            outputs=outputs,
                                            k_budgets=k_budgets,
                                            v_budgets=v_budgets,
                                            policy=str(policy),
                                            threshold=float(threshold),
                                            k_mb_by_idx=k_mb_by_idx,
                                            v_mb_by_idx=v_mb_by_idx,
                                            context_len=int(context_len),
                                            threshold_mode=str(args.threshold_mode),
                                            threshold_reference_frac=float(args.threshold_reference_frac),
                                            threshold_scale_shape=str(args.threshold_scale_shape),
                                            threshold_min_scale=float(args.threshold_min_scale),
                                            threshold_max_scale=float(args.threshold_max_scale),
                                            start_ki=int(start_ki),
                                            start_vi=int(start_vi),
                                            step_mb_by_idx=step_mb_by_pair,
                                            test_log=dv_test_log,
                                            k_bound_by_pair=la_k_bound[dv],
                                            v_bound_by_pair=la_v_bound[dv],
                                        )
                                        d_approx = outputs[(dki, dvi)]
                                        d_rule_label = f"{canonical_v_rule}+la_{dv}"
                                        d_key = (str(score_proxy_variant), d_rule_label, policy, float(threshold), str(start_strategy))
                                        selected_heads[d_key][int(head)] = d_approx
                                        d_metric = _output_error_metrics(dense_head, d_approx)
                                        d_dist = attention_distribution_error_metrics(
                                            scores_np, _true_probs, scores_by_k[dki], probs_by_k[dki]
                                        )
                                        d_k_charged: set[int] = set()
                                        d_v_charged: set[tuple[int, int]] = set()
                                        d_k_tests = d_k_cert = d_k_false = 0
                                        d_v_tests = d_v_cert = d_v_false = 0
                                        for tki, tvi, k_can, v_can, kd, vd, kthr, vthr in dv_test_log:
                                            if k_can:
                                                d_k_tests += 1
                                                if la_k_bound[dv].get((tki, tvi), float("inf")) <= kthr:
                                                    d_k_cert += 1
                                                    if kd > kthr:
                                                        d_k_false += 1
                                                else:
                                                    d_k_charged.add(tki)
                                            if v_can:
                                                d_v_tests += 1
                                                if la_v_bound[dv].get((tki, tvi), float("inf")) <= vthr:
                                                    d_v_cert += 1
                                                    if vd > vthr:
                                                        d_v_false += 1
                                                else:
                                                    d_v_charged.add((tki, tvi))
                                        d_wasted = 0.0
                                        for band_ki in d_k_charged:
                                            if band_ki + 1 > dki:
                                                d_wasted += (
                                                    float(la_k_band_tokens.get(band_ki, 0))
                                                    * float(int(trace.head_dim))
                                                    * float(int(args.key_bytes))
                                                ) / MB
                                        for band_ki, band_vi in d_v_charged:
                                            if not (band_ki == dki and band_vi + 1 <= dvi):
                                                d_wasted += (
                                                    float(la_v_band_tokens.get((band_ki, band_vi), 0))
                                                    * float(int(trace.head_dim))
                                                    * float(int(args.value_bytes))
                                                ) / MB
                                        d_stat_mb = float(context_len * 2) / MB
                                        d_total_mb = float(step_mb_by_pair[(dki, dvi)])
                                        d_v_rule_meta = v_rule_meta_by_pair[(dki, dvi)]
                                        d_row = dict(row)
                                        d_row.update(
                                            {
                                                "v_selection_rule": d_rule_label,
                                                "v_selection_block_size": int(d_v_rule_meta.get("v_selection_block_size", 0)),
                                                "v_exact_reads": int(
                                                    d_v_rule_meta.get(
                                                        "v_selection_exact_reads",
                                                        int(d_v_rule_meta.get("v_selection_exact_target", v_budgets[dvi])),
                                                    )
                                                ),
                                                "v_selection_state_MB": float(v_state_mb_by_pair[(dki, dvi)]),
                                                "calibration_extra_MB": float(calibration_extra_mb_by_idx[dki]),
                                                "calibration_probe_tokens": int(calibration_probe_count_by_idx[dki]),
                                                "k_budget": int(k_budgets[dki]),
                                                "v_budget": int(v_budgets[dvi]),
                                                "selected_k_tokens": int(selected_counts_by_idx[dki]),
                                                "iterations": int(dsteps),
                                                "final_k_delta": float(d_k_delta),
                                                "final_v_delta": float(d_v_delta),
                                                "selector_plus_exact_k_MB": float(k_mb_by_idx[dki]),
                                                "v_path_MB": float(v_mb_by_pair[(dki, dvi)]),
                                                "step_MB_no_v_state_per_head": float(k_mb_by_idx[dki] + v_mb_by_pair[(dki, dvi)]),
                                                "step_MB_with_v_state_per_head": float(d_total_mb),
                                                "step_MB_per_head": float(d_total_mb),
                                                "head_attention_relative_L2": float(d_metric["output_relative_l2"]),
                                                "head_attention_cosine": float(d_metric["output_cosine"]),
                                                "policy_trace": " | ".join(d_trace),
                                                "la_decision_variant": str(dv),
                                                "la_decision_wasted_MB": float(d_wasted),
                                                "la_decision_hw_step_MB": float(d_total_mb + d_wasted + d_stat_mb),
                                                "la_decision_k_tests": int(d_k_tests),
                                                "la_decision_k_certified": int(d_k_cert),
                                                "la_decision_k_false_certified": int(d_k_false),
                                                "la_decision_v_tests": int(d_v_tests),
                                                "la_decision_v_certified": int(d_v_cert),
                                                "la_decision_v_false_certified": int(d_v_false),
                                            }
                                        )
                                        d_row.update({key2: float(value) for key2, value in d_dist.items()})
                                        head_rows.append(d_row)
                                        head_choices[d_key].append(d_row)

        if gqa_union_stats:
            for gkey, acc in gqa_union_acc.items():
                k_sets = acc["k_sets"]
                v_sets = acc["v_sets"]
                k_union = int(np.unique(np.concatenate(k_sets)).size) if k_sets else 0
                v_union = int(np.unique(np.concatenate(v_sets)).size) if v_sets else 0
                gqa_union_rows.append(
                    {
                        "qidx": int(qidx),
                        "position": int(position),
                        "decode_length": int(decode_tokens),
                        "kv_head": int(gkey[0]),
                        "score_proxy_variant": str(gkey[1]),
                        "v_selection_rule": str(gkey[2]),
                        "policy": str(gkey[3]),
                        "threshold": float(gkey[4]),
                        "start_strategy": str(gkey[5]),
                        "group_heads": int(acc["heads"]),
                        "k_sum_tokens": int(acc["k_sum"]),
                        "k_union_tokens": int(k_union),
                        "k_union_over_sum": float(k_union) / max(float(acc["k_sum"]), 1.0),
                        "v_sum_tokens": int(acc["v_sum"]),
                        "v_union_tokens": int(v_union),
                        "v_union_over_sum": float(v_union) / max(float(acc["v_sum"]), 1.0),
                    }
                )
        dense_concat = np.concatenate([dense_heads[int(head)] for head in heads], axis=0).astype(np.float32, copy=False)
        dense_proj = project_head_subset(
            concat_subset=dense_concat,
            heads=[int(h) for h in heads],
            num_heads=int(trace.num_heads),
            head_dim=int(trace.head_dim),
            wo=wo,
            device=device,
        )
        for (score_proxy_variant, v_selection_rule, policy, threshold, start_strategy), by_head in selected_heads.items():
            approx_concat = np.concatenate([by_head[int(head)] for head in heads], axis=0).astype(np.float32, copy=False)
            approx_proj = project_head_subset(
                concat_subset=approx_concat,
                heads=[int(h) for h in heads],
                num_heads=int(trace.num_heads),
                head_dim=int(trace.head_dim),
                wo=wo,
                device=device,
            )
            concat_metric = _output_error_metrics(dense_concat, approx_concat)
            proj_metric = _output_error_metrics(dense_proj, approx_proj)
            choices = head_choices[(score_proxy_variant, v_selection_rule, policy, threshold, start_strategy)]
            layer_rows.append(
                {
                    "qidx": int(qidx),
                    "position": int(position),
                    "decode_length": int(decode_tokens),
                    "selector_mode": str(args.selector_mode),
                    "score_proxy_variant": str(score_proxy_variant),
                    "v_selection_rule": str(v_selection_rule),
                    "quest_rank": int(args.quest_rank),
                    "policy": str(policy),
                    "threshold": float(threshold),
                    "threshold_mode": str(args.threshold_mode),
                    "threshold_reference_frac": float(args.threshold_reference_frac),
                    "threshold_scale_shape": str(args.threshold_scale_shape),
                    "threshold_min_scale": float(args.threshold_min_scale),
                    "threshold_max_scale": float(args.threshold_max_scale),
                    "start_strategy": str(start_strategy),
                    "attn_concat_relative_L2": float(concat_metric["output_relative_l2"]),
                    "attn_o_proj_relative_L2": float(proj_metric["output_relative_l2"]),
                    "attn_o_proj_cosine": float(proj_metric["output_cosine"]),
                    "mean_head_attention_relative_L2": float(np.mean([float(r["head_attention_relative_L2"]) for r in choices])),
                    "max_head_attention_relative_L2": float(np.max([float(r["head_attention_relative_L2"]) for r in choices])),
                    "mean_logit_relL2": float(np.mean([float(r["logit_relL2"]) for r in choices])),
                    "max_logit_relL2": float(np.max([float(r["logit_relL2"]) for r in choices])),
                    "mean_prob_KL_dense_to_approx": float(np.mean([float(r["prob_KL_dense_to_approx"]) for r in choices])),
                    "max_prob_KL_dense_to_approx": float(np.max([float(r["prob_KL_dense_to_approx"]) for r in choices])),
                    "mean_prob_JS": float(np.mean([float(r["prob_JS"]) for r in choices])),
                    "max_prob_JS": float(np.max([float(r["prob_JS"]) for r in choices])),
                    "mean_prob_TV": float(np.mean([float(r["prob_TV"]) for r in choices])),
                    "max_prob_TV": float(np.max([float(r["prob_TV"]) for r in choices])),
                    "mean_prob_top512_overlap": float(np.mean([float(r["prob_top512_overlap"]) for r in choices])),
                    "min_prob_top512_overlap": float(np.min([float(r["prob_top512_overlap"]) for r in choices])),
                    "mean_prob_top512_mass_recall": float(np.mean([float(r["prob_top512_mass_recall"]) for r in choices])),
                    "min_prob_top512_mass_recall": float(np.min([float(r["prob_top512_mass_recall"]) for r in choices])),
                    "mean_k_budget": float(np.mean([float(r["k_budget"]) for r in choices])),
                    "mean_v_budget": float(np.mean([float(r["v_budget"]) for r in choices])),
                    "mean_v_exact_reads": float(np.mean([float(r["v_exact_reads"]) for r in choices])),
                    "mean_selected_k_tokens": float(np.mean([float(r["selected_k_tokens"]) for r in choices])),
                    "mean_selector_coverage": float(np.mean([float(r["selector_coverage"]) for r in choices])),
                    "mean_chosen_nprobe": float(np.mean([float(r["chosen_nprobe"]) for r in choices])),
                    "mean_score_proxy_extra_MB": float(np.mean([float(r["score_proxy_extra_MB"]) for r in choices])),
                    "mean_calibration_extra_MB": float(np.mean([float(r["calibration_extra_MB"]) for r in choices])),
                    "mean_calibration_probe_tokens": float(np.mean([float(r["calibration_probe_tokens"]) for r in choices])),
                    "mean_iterations": float(np.mean([float(r["iterations"]) for r in choices])),
                    "mean_start_k_budget": float(np.mean([float(r["start_k_budget"]) for r in choices])),
                    "mean_start_v_budget": float(np.mean([float(r["start_v_budget"]) for r in choices])),
                    "mean_v_selection_state_MB": float(np.mean([float(r["v_selection_state_MB"]) for r in choices])),
                    "mean_step_MB_no_v_state_per_head": float(np.mean([float(r["step_MB_no_v_state_per_head"]) for r in choices])),
                    "mean_step_MB_with_v_state_per_head": float(np.mean([float(r["step_MB_with_v_state_per_head"]) for r in choices])),
                    "mean_step_MB_per_head": float(np.mean([float(r["step_MB_per_head"]) for r in choices])),
                    "max_step_MB_per_head": float(np.max([float(r["step_MB_per_head"]) for r in choices])),
                }
            )

    output_tables = [
        ("per_head_joint_policy.csv", head_rows),
        ("layer_joint_policy.csv", layer_rows),
    ]
    if oracle_rows:
        output_tables.append(("oracle_budget_diagnostic.csv", oracle_rows))
    if gqa_union_rows:
        output_tables.append(("gqa_union_stats.csv", gqa_union_rows))
    for filename, rows in output_tables:
        with (out_dir / filename).open("w", newline="", encoding="utf-8") as f:
            fieldnames = list(rows[0].keys())
            seen = set(fieldnames)
            for row in rows[1:]:
                for key in row.keys():
                    if key not in seen:
                        fieldnames.append(key)
                        seen.add(key)
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    grouped: dict[tuple[str, str, str, float, str], list[dict[str, object]]] = defaultdict(list)
    for row in layer_rows:
        grouped[
            (
                str(row["score_proxy_variant"]),
                str(row["v_selection_rule"]),
                str(row["policy"]),
                float(row["threshold"]),
                str(row["start_strategy"]),
            )
        ].append(row)
    summary_rows = []
    for (score_proxy_variant, v_selection_rule, policy, threshold, start_strategy), rows in sorted(
        grouped.items(), key=lambda item: (item[0][3], item[0][0], item[0][1], item[0][2], item[0][4])
    ):
        summary_rows.append(
            {
                "score_proxy_variant": str(score_proxy_variant),
                "v_selection_rule": str(v_selection_rule),
                "selector_mode": str(rows[0].get("selector_mode", "")),
                "quest_rank": int(rows[0].get("quest_rank", 0)),
                "policy": str(policy),
                "threshold": float(threshold),
                "start_strategy": str(start_strategy),
                "queries": int(len(rows)),
                "attn_o_proj_relative_L2_mean": float(np.mean([float(r["attn_o_proj_relative_L2"]) for r in rows])),
                "attn_o_proj_relative_L2_max": float(np.max([float(r["attn_o_proj_relative_L2"]) for r in rows])),
                "attn_o_proj_relative_L2_p95": float(np.quantile([float(r["attn_o_proj_relative_L2"]) for r in rows], 0.95)),
                "attn_o_proj_relative_L2_p99": float(np.quantile([float(r["attn_o_proj_relative_L2"]) for r in rows], 0.99)),
                "attn_concat_relative_L2_mean": float(np.mean([float(r["attn_concat_relative_L2"]) for r in rows])),
                "mean_logit_relL2": float(np.mean([float(r["mean_logit_relL2"]) for r in rows])),
                "max_logit_relL2": float(np.max([float(r["max_logit_relL2"]) for r in rows])),
                "p95_logit_relL2": float(np.quantile([float(r["max_logit_relL2"]) for r in rows], 0.95)),
                "p99_logit_relL2": float(np.quantile([float(r["max_logit_relL2"]) for r in rows], 0.99)),
                "mean_prob_KL_dense_to_approx": float(np.mean([float(r["mean_prob_KL_dense_to_approx"]) for r in rows])),
                "max_prob_KL_dense_to_approx": float(np.max([float(r["max_prob_KL_dense_to_approx"]) for r in rows])),
                "p95_prob_KL_dense_to_approx": float(np.quantile([float(r["max_prob_KL_dense_to_approx"]) for r in rows], 0.95)),
                "p99_prob_KL_dense_to_approx": float(np.quantile([float(r["max_prob_KL_dense_to_approx"]) for r in rows], 0.99)),
                "mean_prob_JS": float(np.mean([float(r["mean_prob_JS"]) for r in rows])),
                "max_prob_JS": float(np.max([float(r["max_prob_JS"]) for r in rows])),
                "p95_prob_JS": float(np.quantile([float(r["max_prob_JS"]) for r in rows], 0.95)),
                "p99_prob_JS": float(np.quantile([float(r["max_prob_JS"]) for r in rows], 0.99)),
                "mean_prob_TV": float(np.mean([float(r["mean_prob_TV"]) for r in rows])),
                "max_prob_TV": float(np.max([float(r["max_prob_TV"]) for r in rows])),
                "p95_prob_TV": float(np.quantile([float(r["max_prob_TV"]) for r in rows], 0.95)),
                "p99_prob_TV": float(np.quantile([float(r["max_prob_TV"]) for r in rows], 0.99)),
                "mean_prob_top512_overlap": float(np.mean([float(r["mean_prob_top512_overlap"]) for r in rows])),
                "min_prob_top512_overlap": float(np.min([float(r["min_prob_top512_overlap"]) for r in rows])),
                "mean_prob_top512_mass_recall": float(np.mean([float(r["mean_prob_top512_mass_recall"]) for r in rows])),
                "min_prob_top512_mass_recall": float(np.min([float(r["min_prob_top512_mass_recall"]) for r in rows])),
                "mean_k_budget": float(np.mean([float(r["mean_k_budget"]) for r in rows])),
                "mean_v_budget": float(np.mean([float(r["mean_v_budget"]) for r in rows])),
                "mean_v_exact_reads": float(np.mean([float(r["mean_v_exact_reads"]) for r in rows])),
                "mean_selector_coverage": float(np.mean([float(r["mean_selector_coverage"]) for r in rows])),
                "mean_chosen_nprobe": float(np.mean([float(r["mean_chosen_nprobe"]) for r in rows])),
                "mean_score_proxy_extra_MB": float(np.mean([float(r["mean_score_proxy_extra_MB"]) for r in rows])),
                "mean_calibration_extra_MB": float(np.mean([float(r["mean_calibration_extra_MB"]) for r in rows])),
                "mean_calibration_probe_tokens": float(np.mean([float(r["mean_calibration_probe_tokens"]) for r in rows])),
                "mean_iterations": float(np.mean([float(r["mean_iterations"]) for r in rows])),
                "mean_start_k_budget": float(np.mean([float(r["mean_start_k_budget"]) for r in rows])),
                "mean_start_v_budget": float(np.mean([float(r["mean_start_v_budget"]) for r in rows])),
                "mean_v_selection_state_MB": float(np.mean([float(r["mean_v_selection_state_MB"]) for r in rows])),
                "mean_step_MB_no_v_state_per_head": float(np.mean([float(r["mean_step_MB_no_v_state_per_head"]) for r in rows])),
                "mean_step_MB_with_v_state_per_head": float(np.mean([float(r["mean_step_MB_with_v_state_per_head"]) for r in rows])),
                "mean_step_MB_per_head": float(np.mean([float(r["mean_step_MB_per_head"]) for r in rows])),
                "max_step_MB_per_head": float(np.max([float(r["max_step_MB_per_head"]) for r in rows])),
            }
        )
    summary = {
        "elapsed_seconds": float(time.perf_counter() - t0),
        "decode_lengths": str(args.decode_lengths),
        "heads": [int(h) for h in heads],
        "budget_mode": "relative" if k_budget_fracs else "absolute",
        "k_budgets": [int(x) for x in base_k_budgets],
        "v_budgets": [int(x) for x in base_v_budgets],
        "k_budget_fracs": [float(x) for x in k_budget_fracs],
        "v_budget_fracs": [float(x) for x in v_budget_fracs],
        "thresholds": [float(x) for x in thresholds],
        "oracle_rel_l2_targets": [float(x) for x in oracle_rel_l2_targets],
        "threshold_mode": str(args.threshold_mode),
        "threshold_reference_frac": float(args.threshold_reference_frac),
        "threshold_scale_shape": str(args.threshold_scale_shape),
        "threshold_min_scale": float(args.threshold_min_scale),
        "threshold_max_scale": float(args.threshold_max_scale),
        "start_strategies": [str(x) for x in start_strategies],
        "v_selection_rules": [str(x) for x in v_selection_rules],
        "v_local_block_size": int(args.v_local_block_size),
        "include_v_selection_state_in_step_mb": bool(args.include_v_selection_state_in_step_mb),
        "survivor_logit_bytes": int(args.survivor_logit_bytes),
        "summary": summary_rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[joint_kv_budget_policy_eval] wrote {out_dir}")


if __name__ == "__main__":
    run()
