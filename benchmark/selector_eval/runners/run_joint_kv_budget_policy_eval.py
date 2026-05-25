#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
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
from benchmark.selector_eval.gpu.run_gpu_paged_pq_eval import build_page_pq_gpu, parse_csv_ints, rank_paged_pq
from benchmark.selector_eval.metrics.attention import _output_error_metrics, attention_distribution_error_metrics
from benchmark.selector_eval.runners.run_value_exact_strategy_eval import (
    dense_attention_output,
    mixed_scores,
    output_from_exact_mask,
    project_head_subset,
    top_mask,
    value_vpq_code_stat_risk,
)
from benchmark.selector_eval.runners.run_layer_quality_eval import _selected_for_budget, _vpq_values_for_tokens
from benchmark.selector_eval.runners.run_layer_quality_eval import _rank_quest_pages, _rank_quest_pq


MB = 1024.0 * 1024.0


def parse_csv_floats(text: str) -> list[float]:
    return [float(part.strip()) for part in str(text).split(",") if part.strip()]


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
    turn: int,
    extra_k_mb: float,
    extra_v_mb: float,
) -> str:
    k_bad = bool(k_can and k_delta > threshold)
    v_bad = bool(v_can and v_delta > threshold)
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
) -> tuple[int, int, int, float, float, list[str]]:
    ki = 0
    vi = 0
    steps = 0
    trace: list[str] = []
    while steps < (len(k_budgets) + len(v_budgets) + 4):
        cur = outputs[(ki, vi)]
        k_can = ki + 1 < len(k_budgets)
        v_can = vi + 1 < len(v_budgets)
        k_delta = rel_l2(cur, outputs[(ki + 1, vi)]) if k_can else 0.0
        v_delta = rel_l2(cur, outputs[(ki, vi + 1)]) if v_can else 0.0
        extra_k_mb = float(k_mb_by_idx[ki + 1] - k_mb_by_idx[ki]) if k_can else float("inf")
        extra_v_mb = float(v_mb_by_idx[vi + 1] - v_mb_by_idx[vi]) if v_can else float("inf")
        action = choose_action(
            policy=policy,
            k_delta=k_delta,
            v_delta=v_delta,
            k_can=k_can,
            v_can=v_can,
            threshold=float(threshold),
            turn=steps,
            extra_k_mb=extra_k_mb,
            extra_v_mb=extra_v_mb,
        )
        trace.append(f"{action}:k{ki}/v{vi}:dk={k_delta:.4g}:dv={v_delta:.4g}")
        if action == "stop":
            break
        if action == "k" and k_can:
            ki += 1
        elif action == "v" and v_can:
            vi += 1
        else:
            break
        steps += 1
    return ki, vi, steps, float(k_delta), float(v_delta), trace


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
    parser.add_argument("--stability_thresholds", default="0.0005,0.001,0.002")
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
    parser.add_argument("--selector_mode", choices=["fullscan", "routed", "quest", "quest_pq"], default="fullscan")
    parser.add_argument("--quest_rank", type=int, default=16)
    parser.add_argument("--selector_index_bytes", type=int, default=4)
    parser.add_argument("--tail_score_calibration", choices=["none", "affine_selected"], default="affine_selected")
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
    k_budgets = sorted(set(parse_csv_ints(args.k_budgets)))
    v_budgets = sorted(set(parse_csv_ints(args.v_budgets)))
    thresholds = parse_csv_floats(args.stability_thresholds)
    policies = [part.strip() for part in str(args.policies).split(",") if part.strip()]
    score_proxy_variants = parse_csv_names(args.score_proxy_variants)
    nprobes = parse_csv_ints(args.nprobes)

    x_data = np.load(args.x_trace, mmap_mode="r")
    x_meta = json.loads(str(x_data["metadata"].item()))
    layer_idx = int(x_meta["layer_idx"])
    model_dir = PROJECT_ROOT / args.model_snapshot
    weight_map = load_weight_index(model_dir)
    wo = load_safetensor_weight(model_dir, weight_map, f"model.layers.{layer_idx}.self_attn.o_proj.weight", device)

    head_rows: list[dict[str, object]] = []
    layer_rows: list[dict[str, object]] = []
    t0 = time.perf_counter()

    for qidx in q_indices:
        position = int(trace.positions[int(qidx)])
        decode_tokens = int(trace.decode_tokens_for_qidx(int(qidx)))
        context_len = int(position) + 1
        dynamic_start = min(max(0, int(args.static_prefix)), int(trace.input_len))
        indexed_end = max(
            min(max(0, int(args.static_prefix)), int(trace.input_len)),
            min(context_len - max(0, int(args.static_suffix)), trace.keys.shape[1]),
        )
        needed_kv_heads = sorted({int(trace.kv_head_for(h)) for h in heads})
        index_cache = {}
        for kv_head in needed_kv_heads:
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

        dense_heads: dict[int, np.ndarray] = {}
        selected_heads: dict[tuple[str, str, float], dict[int, np.ndarray]] = defaultdict(dict)
        head_choices: dict[tuple[str, str, float], list[dict[str, object]]] = defaultdict(list)

        for head in heads:
            kv_head = int(trace.kv_head_for(int(head)))
            index = index_cache[kv_head]
            keys_np = trace.keys[kv_head, :context_len].astype(np.float32, copy=False)
            values_np = trace.values[kv_head, :context_len].astype(np.float32, copy=False)
            query_np = trace.queries[int(head), int(qidx)].astype(np.float32, copy=False)
            scores_np, _true_probs, dense_head = dense_attention_output(keys_np, values_np, query_np)
            dense_heads[int(head)] = dense_head

            pending = list(range(max(0, int(index.pending_start)), max(0, min(int(index.indexed_end), context_len))))
            base = unique_tokens(
                static_tokens(position, int(args.static_prefix), int(args.static_suffix)) + pending,
                context_len=context_len,
            )
            max_k_budget = max(k_budgets)
            query_t = torch.as_tensor(query_np, dtype=torch.float32, device=device)
            selector_coverage = 1.0
            if str(args.selector_mode) in {"fullscan", "routed"}:
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
            code_error = value_vpq_code_stat_risk(
                index=index,
                values_np=values_np,
                residual=residual,
                subbits=int(args.subbits),
                value_subvecs=int(args.value_subvecs),
                value_subbits=int(args.value_subbits),
                sensitivity=None,
            )
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

                outputs: dict[tuple[int, int], np.ndarray] = {}
                k_mb_by_idx: list[float] = []
                selected_counts_by_idx: list[int] = []
                probs_by_k: dict[int, np.ndarray] = {}
                scores_by_k: dict[int, np.ndarray] = {}
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
                    probs = np.exp(score_vec - float(np.max(score_vec)))
                    probs /= max(float(probs.sum()), 1e-20)
                    scores_by_k[ki] = score_vec.astype(np.float32, copy=False)
                    probs_by_k[ki] = probs.astype(np.float64, copy=False)
                    exact_key_mb = float(selected_cpu.size * int(trace.head_dim) * int(args.key_bytes)) / MB
                    calibration_extra_mb_by_idx.append(float(calibration_extra_mb))
                    calibration_probe_count_by_idx.append(int(calibration_probe_count))
                    k_mb_by_idx.append(float(selector_mb) + float(score_proxy_extra_mb) + exact_key_mb + float(calibration_extra_mb))
                    for vi, v_budget in enumerate(v_budgets):
                        exact_count = max(0, min(int(v_budget), int(context_len)))
                        exact_mask = top_mask((probs * probs) * code_error, exact_count)
                        outputs[(ki, vi)] = output_from_exact_mask(
                            probs=probs,
                            vhat_all=vhat_all,
                            residual=residual,
                            exact_mask=exact_mask,
                        )

                for threshold in thresholds:
                    for policy in policies:
                        ki, vi, steps, final_k_delta, final_v_delta, policy_trace = simulate_policy(
                            outputs=outputs,
                            k_budgets=k_budgets,
                            v_budgets=v_budgets,
                            policy=str(policy),
                            threshold=float(threshold),
                            k_mb_by_idx=k_mb_by_idx,
                            v_mb_by_idx=v_mb_by_idx,
                        )
                        approx = outputs[(ki, vi)]
                        selected_heads[(str(score_proxy_variant), policy, float(threshold))][int(head)] = approx
                        metric = _output_error_metrics(dense_head, approx)
                        dist_metric = attention_distribution_error_metrics(
                            scores_np,
                            _true_probs,
                            scores_by_k[ki],
                            probs_by_k[ki],
                        )
                        total_mb = float(k_mb_by_idx[ki] + v_mb_by_idx[vi])
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
                            "calibration_extra_MB": float(calibration_extra_mb_by_idx[ki]),
                            "calibration_probe_tokens": int(calibration_probe_count_by_idx[ki]),
                            "policy": str(policy),
                            "threshold": float(threshold),
                            "k_budget": int(k_budgets[ki]),
                            "v_budget": int(v_budgets[vi]),
                            "selected_k_tokens": int(selected_counts_by_idx[ki]),
                            "iterations": int(steps),
                            "final_k_delta": float(final_k_delta),
                            "final_v_delta": float(final_v_delta),
                            "selector_plus_exact_k_MB": float(k_mb_by_idx[ki]),
                            "v_path_MB": float(v_mb_by_idx[vi]),
                            "step_MB_per_head": float(total_mb),
                            "head_attention_relative_L2": float(metric["output_relative_l2"]),
                            "head_attention_cosine": float(metric["output_cosine"]),
                            "policy_trace": " | ".join(policy_trace),
                        }
                        row.update({f"score_proxy_meta_{k}": v for k, v in score_proxy_meta.items() if isinstance(v, (str, int, float, bool))})
                        row.update({key: float(value) for key, value in dist_metric.items()})
                        head_rows.append(row)
                        head_choices[(str(score_proxy_variant), policy, float(threshold))].append(row)

        dense_concat = np.concatenate([dense_heads[int(head)] for head in heads], axis=0).astype(np.float32, copy=False)
        dense_proj = project_head_subset(
            concat_subset=dense_concat,
            heads=[int(h) for h in heads],
            num_heads=int(trace.num_heads),
            head_dim=int(trace.head_dim),
            wo=wo,
            device=device,
        )
        for (score_proxy_variant, policy, threshold), by_head in selected_heads.items():
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
            choices = head_choices[(score_proxy_variant, policy, threshold)]
            layer_rows.append(
                {
                    "qidx": int(qidx),
                    "position": int(position),
                    "decode_length": int(decode_tokens),
                    "selector_mode": str(args.selector_mode),
                    "score_proxy_variant": str(score_proxy_variant),
                    "quest_rank": int(args.quest_rank),
                    "policy": str(policy),
                    "threshold": float(threshold),
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
                    "mean_selected_k_tokens": float(np.mean([float(r["selected_k_tokens"]) for r in choices])),
                    "mean_selector_coverage": float(np.mean([float(r["selector_coverage"]) for r in choices])),
                    "mean_chosen_nprobe": float(np.mean([float(r["chosen_nprobe"]) for r in choices])),
                    "mean_score_proxy_extra_MB": float(np.mean([float(r["score_proxy_extra_MB"]) for r in choices])),
                    "mean_calibration_extra_MB": float(np.mean([float(r["calibration_extra_MB"]) for r in choices])),
                    "mean_calibration_probe_tokens": float(np.mean([float(r["calibration_probe_tokens"]) for r in choices])),
                    "mean_iterations": float(np.mean([float(r["iterations"]) for r in choices])),
                    "mean_step_MB_per_head": float(np.mean([float(r["step_MB_per_head"]) for r in choices])),
                    "max_step_MB_per_head": float(np.max([float(r["step_MB_per_head"]) for r in choices])),
                }
            )

    for filename, rows in [
        ("per_head_joint_policy.csv", head_rows),
        ("layer_joint_policy.csv", layer_rows),
    ]:
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

    grouped: dict[tuple[str, str, float], list[dict[str, object]]] = defaultdict(list)
    for row in layer_rows:
        grouped[(str(row["score_proxy_variant"]), str(row["policy"]), float(row["threshold"]))].append(row)
    summary_rows = []
    for (score_proxy_variant, policy, threshold), rows in sorted(grouped.items(), key=lambda item: (item[0][2], item[0][0], item[0][1])):
        summary_rows.append(
            {
                "score_proxy_variant": str(score_proxy_variant),
                "selector_mode": str(rows[0].get("selector_mode", "")),
                "quest_rank": int(rows[0].get("quest_rank", 0)),
                "policy": str(policy),
                "threshold": float(threshold),
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
                "mean_selector_coverage": float(np.mean([float(r["mean_selector_coverage"]) for r in rows])),
                "mean_chosen_nprobe": float(np.mean([float(r["mean_chosen_nprobe"]) for r in rows])),
                "mean_score_proxy_extra_MB": float(np.mean([float(r["mean_score_proxy_extra_MB"]) for r in rows])),
                "mean_calibration_extra_MB": float(np.mean([float(r["mean_calibration_extra_MB"]) for r in rows])),
                "mean_calibration_probe_tokens": float(np.mean([float(r["mean_calibration_probe_tokens"]) for r in rows])),
                "mean_iterations": float(np.mean([float(r["mean_iterations"]) for r in rows])),
                "mean_step_MB_per_head": float(np.mean([float(r["mean_step_MB_per_head"]) for r in rows])),
                "max_step_MB_per_head": float(np.max([float(r["max_step_MB_per_head"]) for r in rows])),
            }
        )
    summary = {
        "elapsed_seconds": float(time.perf_counter() - t0),
        "decode_lengths": str(args.decode_lengths),
        "heads": [int(h) for h in heads],
        "k_budgets": [int(x) for x in k_budgets],
        "v_budgets": [int(x) for x in v_budgets],
        "thresholds": [float(x) for x in thresholds],
        "summary": summary_rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[joint_kv_budget_policy_eval] wrote {out_dir}")


if __name__ == "__main__":
    run()
