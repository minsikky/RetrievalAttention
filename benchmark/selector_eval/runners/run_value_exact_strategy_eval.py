#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import safe_open

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.selector_eval.data.trace import attention_probs, load_trace, static_tokens, unique_tokens
from benchmark.selector_eval.gpu.run_gpu_paged_pq_eval import build_page_pq_gpu, parse_csv_ints, rank_paged_pq
from benchmark.selector_eval.metrics.attention import _output_error_metrics
from benchmark.selector_eval.runners.run_layer_quality_eval import (
    _fit_selected_pq_logit_calibration,
    _selected_for_budget,
    _selected_value_exact_mask,
    _vpq_values_for_tokens,
)
from benchmark.selector_eval.runners.diagnose_layer_heads import _build_value_vpq_sidecars


MB = 1024.0 * 1024.0


def load_weight_index(model_dir: Path) -> dict[str, str]:
    index_path = model_dir / "model.safetensors.index.json"
    data = json.loads(index_path.read_text())
    return {str(k): str(v) for k, v in data["weight_map"].items()}


def load_safetensor_weight(model_dir: Path, weight_map: dict[str, str], name: str, device: torch.device) -> torch.Tensor:
    shard = model_dir / weight_map[name]
    with safe_open(shard, framework="pt", device="cpu") as f:
        return f.get_tensor(name).to(device=device, dtype=torch.float32, non_blocking=True)


def softmax(scores: np.ndarray) -> np.ndarray:
    scores64 = scores.astype(np.float64, copy=False)
    shifted = scores64 - float(np.max(scores64))
    weights = np.exp(shifted)
    weights /= max(float(weights.sum()), 1e-20)
    return weights.astype(np.float64, copy=False)


def dense_attention_output(keys_np: np.ndarray, values_np: np.ndarray, query_np: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scores_np, probs_np = attention_probs(keys_np, query_np)
    out = probs_np.astype(np.float64) @ values_np.astype(np.float64, copy=False)
    return scores_np.astype(np.float32, copy=False), probs_np.astype(np.float64, copy=False), out.astype(np.float32, copy=False)


def top_mask(scores: np.ndarray, budget: int) -> np.ndarray:
    mask = np.zeros((scores.shape[0],), dtype=bool)
    count = max(0, min(int(budget), int(scores.shape[0])))
    if count <= 0:
        return mask
    order = np.argsort(-scores.astype(np.float64, copy=False), kind="stable")
    mask[order[:count]] = True
    return mask


def selected_mass_exact_mask(
    *,
    selected_cpu: np.ndarray,
    scores_np: np.ndarray,
    context_len: int,
    mass_target: float,
    min_top: int,
    max_top: int,
) -> tuple[np.ndarray, int, float]:
    local_mask, exact_count, achieved_mass = _selected_value_exact_mask(
        selected_arr=selected_cpu,
        selected_scores=scores_np[selected_cpu],
        rule="selected_mass",
        fixed_top=0,
        mass_target=float(mass_target),
        min_top=int(min_top),
        max_top=int(max_top),
    )
    global_mask = np.zeros((int(context_len),), dtype=bool)
    if selected_cpu.size:
        global_mask[selected_cpu[local_mask]] = True
    return global_mask, int(exact_count), float(achieved_mass)


def mixed_scores(
    *,
    context_len: int,
    selected_cpu: np.ndarray,
    ranked_cpu: np.ndarray,
    ranked_scores_cpu: np.ndarray,
    exact_scores_np: np.ndarray,
    query_dim: int,
    calibrate: bool,
) -> tuple[np.ndarray, int, float, float]:
    selected_set = set(int(tok) for tok in selected_cpu.tolist())
    scale = 1.0
    bias = 0.0
    calibration_tokens = 0
    if calibrate and selected_cpu.size:
        fit = _fit_selected_pq_logit_calibration(
            selected_cpu=selected_cpu,
            ranked_cpu=ranked_cpu,
            ranked_scores_cpu=ranked_scores_cpu,
            scores_np=exact_scores_np,
            query_dim=int(query_dim),
        )
        scale = float(fit[0])
        bias = float(fit[1])
        calibration_tokens = int(fit[2])

    out = np.full((int(context_len),), -np.inf, dtype=np.float64)
    if selected_cpu.size:
        out[selected_cpu] = exact_scores_np[selected_cpu].astype(np.float64, copy=False)
    sqrt_dim = float(np.sqrt(float(query_dim)))
    for tok, score in zip(ranked_cpu.tolist(), ranked_scores_cpu.tolist(), strict=False):
        tok_i = int(tok)
        if tok_i in selected_set or tok_i < 0 or tok_i >= int(context_len):
            continue
        out[tok_i] = scale * (float(score) / sqrt_dim) + bias

    # Missing tokens are unindexed/static edge cases. They are not selector
    # wins, so exact fallback here is only to keep the probability vector valid.
    missing = ~np.isfinite(out)
    missing_count = int(np.count_nonzero(missing))
    if missing_count:
        out[missing] = exact_scores_np[missing].astype(np.float64, copy=False)
    return out, missing_count, float(scale), float(bias)


def projected_residual_diag_scores(
    *,
    residual: np.ndarray,
    probs: np.ndarray,
    wo: torch.Tensor | None,
    head: int,
    head_dim: int,
) -> np.ndarray:
    if wo is None:
        return (probs * probs) * np.sum(residual.astype(np.float64) * residual.astype(np.float64), axis=1)
    start = int(head) * int(head_dim)
    end = start + int(head_dim)
    # Full projected norm would use W_h^T W_h. The diagonal is much cheaper and
    # gives a post-projection-aware ranking signal without a dense 128x128 Gram
    # per head.
    sensitivity_t = torch.sum(wo[:, start:end].float() * wo[:, start:end].float(), dim=0)
    # Some cluster torch/numpy combinations expose torch-created NumPy arrays
    # with broken dtype metadata. Converting through a Python list avoids that
    # interop edge case; this vector has only head_dim elements.
    sensitivity = np.asarray(sensitivity_t.detach().cpu().tolist(), dtype="float64")
    if int(sensitivity.shape[0]) != int(residual.shape[1]):
        return (probs * probs) * np.sum(residual.astype(np.float64) * residual.astype(np.float64), axis=1)
    weighted = residual.astype(np.float64, copy=False) * sensitivity.reshape(1, -1)
    return (probs * probs) * np.sum(residual.astype(np.float64, copy=False) * weighted, axis=1)


def projected_residual_diag_norms(
    *,
    residual: np.ndarray,
    wo: torch.Tensor | None,
    head: int,
    head_dim: int,
) -> np.ndarray:
    if wo is None:
        return np.sum(residual.astype(np.float64) * residual.astype(np.float64), axis=1)
    start = int(head) * int(head_dim)
    end = start + int(head_dim)
    sensitivity_t = torch.sum(wo[:, start:end].float() * wo[:, start:end].float(), dim=0)
    sensitivity = np.asarray(sensitivity_t.detach().cpu().tolist(), dtype="float64")
    if int(sensitivity.shape[0]) != int(residual.shape[1]):
        return np.sum(residual.astype(np.float64) * residual.astype(np.float64), axis=1)
    weighted = residual.astype(np.float64, copy=False) * sensitivity.reshape(1, -1)
    return np.sum(residual.astype(np.float64, copy=False) * weighted, axis=1)


def weighted_residual_norms(*, residual: np.ndarray, sensitivity: np.ndarray | None) -> np.ndarray:
    residual64 = residual.astype(np.float64, copy=False)
    if sensitivity is None:
        return np.sum(residual64 * residual64, axis=1)
    sens = sensitivity.astype(np.float64, copy=False).reshape(1, -1)
    return np.sum(residual64 * residual64 * sens, axis=1)


def value_vpq_code_stat_risk(
    *,
    index,
    values_np: np.ndarray,
    residual: np.ndarray,
    subbits: int,
    value_subvecs: int,
    value_subbits: int,
    sensitivity: np.ndarray | None,
) -> np.ndarray:
    """Deployable per-page/code residual-risk estimate for every token.

    This models storing a small residual-risk scalar per V-PQ code when a page
    is sealed. Query time reads token codes plus the page/code risk table, not
    exact V rows.
    """

    actual_value_subbits = int(value_subbits) if int(value_subbits) > 0 else int(subbits)
    sidecars = _build_value_vpq_sidecars(
        index,
        values_np,
        int(subbits),
        value_subvecs=int(value_subvecs),
        value_subbits=actual_value_subbits,
    )
    out = np.zeros((int(values_np.shape[0]),), dtype=np.float64)
    for page_id, page in enumerate(index.pages):
        start = int(page.start)
        size = int(page.size)
        if size <= 0:
            continue
        codebook, page_codes = sidecars[int(page_id)]
        if codebook.size == 0 or page_codes.size == 0:
            continue
        codes = page_codes.astype(np.int64, copy=False)
        subvecs = int(codes.shape[1])
        subdim = int(codebook.shape[-1])
        block_residual = residual[start : start + size].astype(np.float64, copy=False)
        for sub in range(subvecs):
            lo = int(sub) * subdim
            hi = lo + subdim
            sub_residual = block_residual[:, lo:hi]
            if sensitivity is None:
                per_token = np.sum(sub_residual * sub_residual, axis=1)
            else:
                sens = sensitivity[lo:hi].astype(np.float64, copy=False).reshape(1, -1)
                per_token = np.sum(sub_residual * sub_residual * sens, axis=1)
            sub_codes = codes[:, sub]
            table = np.zeros((1 << actual_value_subbits,), dtype=np.float64)
            for code in np.unique(sub_codes).astype(np.int64, copy=False).tolist():
                mask = sub_codes == int(code)
                table[int(code)] = float(np.mean(per_token[mask])) if np.any(mask) else 0.0
            out[start : start + size] += table[sub_codes]
    return out


def projected_sensitivity(
    *,
    wo: torch.Tensor | None,
    head: int,
    head_dim: int,
) -> np.ndarray | None:
    if wo is None:
        return None
    start = int(head) * int(head_dim)
    end = start + int(head_dim)
    sensitivity_t = torch.sum(wo[:, start:end].float() * wo[:, start:end].float(), dim=0)
    sensitivity = np.asarray(sensitivity_t.detach().cpu().tolist(), dtype="float64")
    return sensitivity if int(sensitivity.shape[0]) == int(head_dim) else None


def topm_projected_sensitivity(
    *,
    sensitivity: np.ndarray | None,
    top_m: int,
) -> np.ndarray | None:
    if sensitivity is None:
        return None
    top_m = max(0, min(int(top_m), int(sensitivity.shape[0])))
    if top_m <= 0:
        return None
    out = np.zeros_like(sensitivity, dtype=np.float64)
    order = np.argsort(-sensitivity.astype(np.float64, copy=False), kind="stable")[:top_m]
    out[order] = sensitivity[order]
    return out


def output_from_exact_mask(
    *,
    probs: np.ndarray,
    vhat_all: np.ndarray,
    residual: np.ndarray,
    exact_mask: np.ndarray,
) -> np.ndarray:
    base = probs.astype(np.float64, copy=False) @ vhat_all.astype(np.float64, copy=False)
    if bool(np.any(exact_mask)):
        base += probs[exact_mask].astype(np.float64, copy=False) @ residual[exact_mask].astype(np.float64, copy=False)
    return base.astype(np.float32, copy=False)


def project_head_subset(
    *,
    concat_subset: np.ndarray,
    heads: list[int],
    num_heads: int,
    head_dim: int,
    wo: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    full = np.zeros((int(num_heads) * int(head_dim),), dtype=np.float32)
    cursor = 0
    for head in heads:
        start = int(head) * int(head_dim)
        end = start + int(head_dim)
        full[start:end] = concat_subset[cursor : cursor + int(head_dim)]
        cursor += int(head_dim)
    projected = F.linear(torch.as_tensor(full, dtype=torch.float32, device=device).reshape(1, -1), wo)
    return np.asarray(projected.reshape(-1).detach().cpu().tolist(), dtype="float32")


def run() -> None:
    parser = argparse.ArgumentParser(description="Compare exact-V set strategies at the same exact-V budget.")
    parser.add_argument("--qkv_trace", required=True)
    parser.add_argument("--x_trace", default="")
    parser.add_argument(
        "--model_snapshot",
        default=".hf_cache/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--decode_lengths", default="500,1000,2000,4000,8000,16000,32000,64000,128000")
    parser.add_argument("--max_qidx_per_decode", type=int, default=1)
    parser.add_argument("--heads", default="", help="Optional comma-separated query heads. Empty means all heads.")
    parser.add_argument("--selector_mode", choices=["fullscan", "routed"], default="fullscan")
    parser.add_argument("--k_budget", type=int, default=14336)
    parser.add_argument("--prob_sources", default="dense,mixed")
    parser.add_argument("--tail_score_calibration", choices=["none", "affine_selected"], default="none")
    parser.add_argument("--selected_value_exact_mass", type=float, default=0.99)
    parser.add_argument("--selected_value_min_exact_top", type=int, default=0)
    parser.add_argument("--selected_value_max_exact_top", type=int, default=0)
    parser.add_argument("--static_prefix", type=int, default=128)
    parser.add_argument("--static_suffix", type=int, default=128)
    parser.add_argument("--page_size", type=int, default=5632)
    parser.add_argument("--subvecs", type=int, default=4)
    parser.add_argument("--subbits", type=int, default=8)
    parser.add_argument("--value_subvecs", type=int, default=1)
    parser.add_argument("--value_subbits", type=int, default=4)
    parser.add_argument("--residual_metadata_bytes", type=int, default=2)
    parser.add_argument("--code_stat_bytes", type=int, default=2)
    parser.add_argument("--topm_channels", default="1,4,8,16,32")
    parser.add_argument("--kmeans_iters", type=int, default=3)
    parser.add_argument("--key_bytes", type=int, default=2)
    parser.add_argument("--value_bytes", type=int, default=2)
    parser.add_argument("--nprobes", default="512")
    parser.add_argument("--router_prototypes", type=int, default=16)
    parser.add_argument("--router_merge_rel", type=float, default=0.05)
    parser.add_argument("--router_merge_var", type=float, default=0.0)
    parser.add_argument("--router_max_groups", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip_post_proj", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
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
            seen = counts.get(int(decode), 0)
            if seen >= int(args.max_qidx_per_decode):
                continue
            limited.append(int(qidx))
            counts[int(decode)] = seen + 1
        q_indices = limited
    if not q_indices:
        raise ValueError("no query indices selected")

    heads = parse_csv_ints(args.heads) if str(args.heads).strip() else list(range(int(trace.num_heads)))
    prob_sources = [part.strip() for part in str(args.prob_sources).split(",") if part.strip()]
    valid_prob_sources = {"dense", "mixed"}
    unknown = sorted(set(prob_sources) - valid_prob_sources)
    if unknown:
        raise ValueError(f"unknown prob_sources: {unknown}")
    nprobes = parse_csv_ints(args.nprobes)
    topm_channels = parse_csv_ints(args.topm_channels) if str(args.topm_channels).strip() else []

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

    per_head_rows: list[dict[str, object]] = []
    layer_rows: list[dict[str, object]] = []
    t_start = time.perf_counter()

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
        approx_heads: dict[tuple[str, str], dict[int, np.ndarray]] = defaultdict(dict)

        for head in heads:
            kv_head = int(trace.kv_head_for(int(head)))
            index = index_cache[kv_head]
            keys_np = trace.keys[kv_head, :context_len].astype(np.float32, copy=False)
            values_np = trace.values[kv_head, :context_len].astype(np.float32, copy=False)
            query_np = trace.queries[int(head), int(qidx)].astype(np.float32, copy=False)
            scores_np, true_probs, dense_head = dense_attention_output(keys_np, values_np, query_np)
            dense_heads[int(head)] = dense_head

            pending = list(range(max(0, int(index.pending_start)), max(0, min(int(index.indexed_end), context_len))))
            base = unique_tokens(
                static_tokens(position, int(args.static_prefix), int(args.static_suffix)) + pending,
                context_len=context_len,
            )
            query_t = torch.as_tensor(query_np, dtype=torch.float32, device=device)
            ranked_t, ranked_scores_t, _selector_seconds, selector_mb, chosen_nprobe = rank_paged_pq(
                query_t,
                index,
                mode=str(args.selector_mode),
                selector_backend="torch",
                nprobes=nprobes,
                budget=int(args.k_budget),
                key_bytes=int(args.key_bytes),
                subbits=int(args.subbits),
            )
            ranked_cpu = ranked_t.detach().cpu().numpy().astype(np.int64, copy=False)
            ranked_scores_cpu = ranked_scores_t.detach().cpu().numpy().astype(np.float32, copy=False)
            selected_cpu = _selected_for_budget(
                base=base,
                ranked_cpu=ranked_cpu,
                budget=int(args.k_budget),
                context_len=context_len,
            )
            selected_set = np.zeros((context_len,), dtype=bool)
            selected_set[selected_cpu] = True

            current_mask, current_budget, current_selected_mass = selected_mass_exact_mask(
                selected_cpu=selected_cpu,
                scores_np=scores_np,
                context_len=context_len,
                mass_target=float(args.selected_value_exact_mass),
                min_top=int(args.selected_value_min_exact_top),
                max_top=int(args.selected_value_max_exact_top),
            )

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
            residual_l2_scores = np.sum(residual.astype(np.float64) * residual.astype(np.float64), axis=1)
            sensitivity = projected_sensitivity(wo=wo, head=int(head), head_dim=int(trace.head_dim))
            residual_postproj_scores = projected_residual_diag_norms(
                residual=residual,
                wo=wo,
                head=int(head),
                head_dim=int(trace.head_dim),
            )
            code_l2_scores = value_vpq_code_stat_risk(
                index=index,
                values_np=values_np,
                residual=residual,
                subbits=int(args.subbits),
                value_subvecs=int(args.value_subvecs),
                value_subbits=int(args.value_subbits),
                sensitivity=None,
            )
            code_postproj_scores = value_vpq_code_stat_risk(
                index=index,
                values_np=values_np,
                residual=residual,
                subbits=int(args.subbits),
                value_subvecs=int(args.value_subvecs),
                value_subbits=int(args.value_subbits),
                sensitivity=sensitivity,
            )
            topm_scalar_scores: dict[int, np.ndarray] = {}
            topm_code_scores: dict[int, np.ndarray] = {}
            for top_m in topm_channels:
                top_sens = topm_projected_sensitivity(sensitivity=sensitivity, top_m=int(top_m))
                if top_sens is None:
                    continue
                topm_scalar_scores[int(top_m)] = weighted_residual_norms(
                    residual=residual,
                    sensitivity=top_sens,
                )
                topm_code_scores[int(top_m)] = value_vpq_code_stat_risk(
                    index=index,
                    values_np=values_np,
                    residual=residual,
                    subbits=int(args.subbits),
                    value_subvecs=int(args.value_subvecs),
                    value_subbits=int(args.value_subbits),
                    sensitivity=top_sens,
                )

            score_vectors: dict[str, tuple[np.ndarray, int, float, float]] = {
                "dense": (scores_np.astype(np.float64, copy=False), 0, 1.0, 0.0)
            }
            if "mixed" in prob_sources:
                score_vectors["mixed"] = mixed_scores(
                    context_len=context_len,
                    selected_cpu=selected_cpu,
                    ranked_cpu=ranked_cpu,
                    ranked_scores_cpu=ranked_scores_cpu,
                    exact_scores_np=scores_np,
                    query_dim=int(trace.head_dim),
                    calibrate=str(args.tail_score_calibration) == "affine_selected",
                )

            for prob_source in prob_sources:
                prob_scores, missing_scores, score_scale, score_bias = score_vectors[prob_source]
                probs = softmax(prob_scores)
                projected_diag = projected_residual_diag_scores(
                    residual=residual,
                    probs=probs,
                    wo=wo,
                    head=int(head),
                    head_dim=int(trace.head_dim),
                )
                prob2 = probs * probs

                strategies: dict[str, np.ndarray] = {}
                metadata_mb: dict[str, float] = {}
                selected_meta_mb = float(selected_cpu.size * max(0, int(args.residual_metadata_bytes))) / MB
                global_meta_mb = float(context_len * max(0, int(args.residual_metadata_bytes))) / MB
                actual_value_subbits = int(args.value_subbits) if int(args.value_subbits) > 0 else int(args.subbits)
                actual_value_subvecs = int(args.value_subvecs) if int(args.value_subvecs) > 0 else int(args.subvecs)
                code_bytes = 1 if int(actual_value_subbits) <= 8 else 2
                selected_code_stat_mb = (
                    float(selected_cpu.size * int(actual_value_subvecs) * code_bytes)
                    + float(
                        len(index.pages)
                        * int(actual_value_subvecs)
                        * (1 << int(actual_value_subbits))
                        * max(0, int(args.code_stat_bytes))
                    )
                ) / MB
                global_code_stat_mb = (
                    float(context_len * int(actual_value_subvecs) * code_bytes)
                    + float(
                        len(index.pages)
                        * int(actual_value_subvecs)
                        * (1 << int(actual_value_subbits))
                        * max(0, int(args.code_stat_bytes))
                    )
                ) / MB

                strategies["current_selected_mass"] = current_mask.copy()
                metadata_mb["current_selected_mass"] = 0.0
                strategies["selected_residual_scalar_meta"] = np.zeros((context_len,), dtype=bool)
                local = selected_cpu[top_mask(prob2[selected_cpu] * residual_l2_scores[selected_cpu], current_budget)]
                strategies["selected_residual_scalar_meta"][local] = True
                metadata_mb["selected_residual_scalar_meta"] = selected_meta_mb
                strategies["selected_postproj_scalar_meta"] = np.zeros((context_len,), dtype=bool)
                local = selected_cpu[top_mask(prob2[selected_cpu] * residual_postproj_scores[selected_cpu], current_budget)]
                strategies["selected_postproj_scalar_meta"][local] = True
                metadata_mb["selected_postproj_scalar_meta"] = selected_meta_mb
                strategies["selected_residual_code_stat"] = np.zeros((context_len,), dtype=bool)
                local = selected_cpu[top_mask(prob2[selected_cpu] * code_l2_scores[selected_cpu], current_budget)]
                strategies["selected_residual_code_stat"][local] = True
                metadata_mb["selected_residual_code_stat"] = selected_code_stat_mb
                strategies["selected_postproj_code_stat"] = np.zeros((context_len,), dtype=bool)
                local = selected_cpu[top_mask(prob2[selected_cpu] * code_postproj_scores[selected_cpu], current_budget)]
                strategies["selected_postproj_code_stat"][local] = True
                metadata_mb["selected_postproj_code_stat"] = selected_code_stat_mb
                strategies["global_prob_oracle"] = top_mask(probs, current_budget)
                metadata_mb["global_prob_oracle"] = 0.0
                strategies["global_residual_scalar_meta"] = top_mask(prob2 * residual_l2_scores, current_budget)
                metadata_mb["global_residual_scalar_meta"] = global_meta_mb
                strategies["global_postproj_scalar_meta"] = top_mask(prob2 * residual_postproj_scores, current_budget)
                metadata_mb["global_postproj_scalar_meta"] = global_meta_mb
                strategies["global_residual_code_stat"] = top_mask(prob2 * code_l2_scores, current_budget)
                metadata_mb["global_residual_code_stat"] = global_code_stat_mb
                strategies["global_postproj_code_stat"] = top_mask(prob2 * code_postproj_scores, current_budget)
                metadata_mb["global_postproj_code_stat"] = global_code_stat_mb
                for top_m in topm_channels:
                    top_m = int(top_m)
                    if top_m not in topm_code_scores or top_m not in topm_scalar_scores:
                        continue
                    selected_code_name = f"selected_postproj_top{top_m}_code_stat"
                    strategies[selected_code_name] = np.zeros((context_len,), dtype=bool)
                    local = selected_cpu[
                        top_mask(prob2[selected_cpu] * topm_code_scores[top_m][selected_cpu], current_budget)
                    ]
                    strategies[selected_code_name][local] = True
                    metadata_mb[selected_code_name] = selected_code_stat_mb

                    global_code_name = f"global_postproj_top{top_m}_code_stat"
                    strategies[global_code_name] = top_mask(prob2 * topm_code_scores[top_m], current_budget)
                    metadata_mb[global_code_name] = global_code_stat_mb

                    selected_scalar_name = f"selected_postproj_top{top_m}_scalar_meta"
                    strategies[selected_scalar_name] = np.zeros((context_len,), dtype=bool)
                    local = selected_cpu[
                        top_mask(prob2[selected_cpu] * topm_scalar_scores[top_m][selected_cpu], current_budget)
                    ]
                    strategies[selected_scalar_name][local] = True
                    metadata_mb[selected_scalar_name] = selected_meta_mb

                    global_scalar_name = f"global_postproj_top{top_m}_scalar_meta"
                    strategies[global_scalar_name] = top_mask(prob2 * topm_scalar_scores[top_m], current_budget)
                    metadata_mb[global_scalar_name] = global_meta_mb
                strategies["none_exact_v"] = np.zeros((context_len,), dtype=bool)
                metadata_mb["none_exact_v"] = 0.0
                strategies["all_selected_exact_v"] = selected_set.copy()
                metadata_mb["all_selected_exact_v"] = 0.0

                for strategy, exact_mask in strategies.items():
                    approx = output_from_exact_mask(
                        probs=probs,
                        vhat_all=vhat_all,
                        residual=residual,
                        exact_mask=exact_mask,
                    )
                    approx_heads[(prob_source, strategy)][int(head)] = approx
                    metric = _output_error_metrics(dense_head, approx)
                    exact_count = int(np.count_nonzero(exact_mask))
                    outside_k = int(np.count_nonzero(exact_mask & ~selected_set))
                    per_head_rows.append(
                        {
                            "qidx": int(qidx),
                            "position": int(position),
                            "decode_length": int(decode_tokens),
                            "head": int(head),
                            "kv_head": int(kv_head),
                            "prob_source": str(prob_source),
                            "strategy": str(strategy),
                            "context_len": int(context_len),
                            "selected_k_tokens": int(selected_cpu.size),
                            "baseline_exact_v_budget": int(current_budget),
                            "baseline_selected_mass": float(current_selected_mass),
                            "exact_v_tokens": int(exact_count),
                            "exact_v_outside_k_tokens": int(outside_k),
                            "exact_v_outside_k_frac": float(outside_k) / max(1.0, float(exact_count)),
                            "exact_v_MB_per_head": float(exact_count * int(trace.head_dim) * int(args.value_bytes)) / MB,
                            "metadata_MB_per_head": float(metadata_mb.get(strategy, 0.0)),
                            "selector_MB_per_head": float(selector_mb),
                            "chosen_nprobe": int(chosen_nprobe),
                            "mixed_missing_scores": int(missing_scores),
                            "mixed_score_scale": float(score_scale),
                            "mixed_score_bias": float(score_bias),
                            "head_attention_relative_L2": float(metric["output_relative_l2"]),
                            "head_attention_cosine": float(metric["output_cosine"]),
                        }
                    )

        dense_concat = np.concatenate([dense_heads[int(head)] for head in heads], axis=0).astype(np.float32, copy=False)
        dense_proj = None
        if wo is not None:
            dense_proj = project_head_subset(
                concat_subset=dense_concat,
                heads=[int(h) for h in heads],
                num_heads=int(trace.num_heads),
                head_dim=int(trace.head_dim),
                wo=wo,
                device=device,
            )

        for key, by_head in approx_heads.items():
            prob_source, strategy = key
            approx_concat = np.concatenate([by_head[int(head)] for head in heads], axis=0).astype(np.float32, copy=False)
            concat_metric = _output_error_metrics(dense_concat, approx_concat)
            proj_metric = {"output_relative_l2": float("nan"), "output_cosine": float("nan")}
            if wo is not None and dense_proj is not None:
                approx_proj = project_head_subset(
                    concat_subset=approx_concat,
                    heads=[int(h) for h in heads],
                    num_heads=int(trace.num_heads),
                    head_dim=int(trace.head_dim),
                    wo=wo,
                    device=device,
                )
                proj_metric = _output_error_metrics(dense_proj, approx_proj)
            matching = [
                row
                for row in per_head_rows
                if int(row["qidx"]) == int(qidx)
                and str(row["prob_source"]) == str(prob_source)
                and str(row["strategy"]) == str(strategy)
            ]
            layer_rows.append(
                {
                    "qidx": int(qidx),
                    "position": int(position),
                    "decode_length": int(decode_tokens),
                    "prob_source": str(prob_source),
                    "strategy": str(strategy),
                    "heads": int(len(heads)),
                    "attn_concat_relative_L2": float(concat_metric["output_relative_l2"]),
                    "attn_concat_cosine": float(concat_metric["output_cosine"]),
                    "attn_o_proj_relative_L2": float(proj_metric["output_relative_l2"]),
                    "attn_o_proj_cosine": float(proj_metric["output_cosine"]),
                    "mean_head_attention_relative_L2": float(np.mean([float(r["head_attention_relative_L2"]) for r in matching])),
                    "max_head_attention_relative_L2": float(np.max([float(r["head_attention_relative_L2"]) for r in matching])),
                    "mean_selected_k_tokens": float(np.mean([float(r["selected_k_tokens"]) for r in matching])),
                    "mean_baseline_exact_v_budget": float(np.mean([float(r["baseline_exact_v_budget"]) for r in matching])),
                    "mean_exact_v_tokens": float(np.mean([float(r["exact_v_tokens"]) for r in matching])),
                    "mean_exact_v_outside_k_frac": float(np.mean([float(r["exact_v_outside_k_frac"]) for r in matching])),
                    "mean_exact_v_MB_per_head": float(np.mean([float(r["exact_v_MB_per_head"]) for r in matching])),
                    "mean_metadata_MB_per_head": float(np.mean([float(r["metadata_MB_per_head"]) for r in matching])),
                }
            )

    per_head_path = out_dir / "per_head_value_strategy.csv"
    with per_head_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_head_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_head_rows)
    layer_path = out_dir / "layer_value_strategy.csv"
    with layer_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(layer_rows[0].keys()))
        writer.writeheader()
        writer.writerows(layer_rows)

    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in layer_rows:
        groups[(str(row["prob_source"]), str(row["strategy"]))].append(row)
    summary_rows = []
    for (prob_source, strategy), rows in sorted(groups.items()):
        summary_rows.append(
            {
                "prob_source": prob_source,
                "strategy": strategy,
                "queries": int(len(rows)),
                "attn_concat_relative_L2_mean": float(np.mean([float(r["attn_concat_relative_L2"]) for r in rows])),
                "attn_concat_relative_L2_max": float(np.max([float(r["attn_concat_relative_L2"]) for r in rows])),
                "attn_o_proj_relative_L2_mean": float(np.mean([float(r["attn_o_proj_relative_L2"]) for r in rows])),
                "attn_o_proj_relative_L2_max": float(np.max([float(r["attn_o_proj_relative_L2"]) for r in rows])),
                "mean_head_attention_relative_L2_mean": float(
                    np.mean([float(r["mean_head_attention_relative_L2"]) for r in rows])
                ),
                "max_head_attention_relative_L2_max": float(np.max([float(r["max_head_attention_relative_L2"]) for r in rows])),
                "mean_selected_k_tokens": float(np.mean([float(r["mean_selected_k_tokens"]) for r in rows])),
                "mean_exact_v_tokens": float(np.mean([float(r["mean_exact_v_tokens"]) for r in rows])),
                "mean_exact_v_outside_k_frac": float(np.mean([float(r["mean_exact_v_outside_k_frac"]) for r in rows])),
                "mean_exact_v_MB_per_head": float(np.mean([float(r["mean_exact_v_MB_per_head"]) for r in rows])),
                "mean_metadata_MB_per_head": float(np.mean([float(r["mean_metadata_MB_per_head"]) for r in rows])),
            }
        )
    summary = {
        "elapsed_seconds": float(time.perf_counter() - t_start),
        "qkv_trace": str(args.qkv_trace),
        "x_trace": str(args.x_trace),
        "decode_lengths": str(args.decode_lengths),
        "max_qidx_per_decode": int(args.max_qidx_per_decode),
        "heads": [int(h) for h in heads],
        "k_budget": int(args.k_budget),
        "selected_value_exact_mass": float(args.selected_value_exact_mass),
        "topm_channels": [int(x) for x in topm_channels],
        "summary": summary_rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[value_exact_strategy_eval] wrote {out_dir}")


if __name__ == "__main__":
    run()
