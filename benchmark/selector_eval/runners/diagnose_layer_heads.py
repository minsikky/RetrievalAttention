#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.selector_eval.data.trace import attention_probs, load_trace, static_tokens, unique_tokens
from benchmark.selector_eval.gpu.run_gpu_paged_pq_eval import (
    build_page_pq_gpu,
    parse_csv_ints,
    rank_paged_pq,
    selected_plus_tail_output,
)
from benchmark.selector_eval.metrics.attention import _output_error_metrics

MB = 1024.0 * 1024.0


def _codes_cpu(page) -> np.ndarray:
    return page.codes.detach().cpu().numpy().astype(np.int64, copy=False)


def _build_value_codebooks_for_key_codes(index, values_np: np.ndarray, subbits: int) -> list[np.ndarray]:
    value_codebooks: list[np.ndarray] = []
    centroids = 1 << int(subbits)
    for page in index.pages:
        codes = _codes_cpu(page)
        values = values_np[int(page.start) : int(page.start) + int(page.size)].astype(np.float32, copy=False)
        subvecs = int(codes.shape[1])
        subdim = int(values.shape[1]) // max(1, subvecs)
        codebook = np.zeros((subvecs, centroids, subdim), dtype=np.float32)
        for sub in range(subvecs):
            part = values[:, sub * subdim : (sub + 1) * subdim]
            fallback = part.mean(axis=0) if part.shape[0] else np.zeros((subdim,), dtype=np.float32)
            for code in range(centroids):
                mask = codes[:, sub] == code
                codebook[sub, code] = part[mask].mean(axis=0) if np.any(mask) else fallback
        value_codebooks.append(codebook)
    return value_codebooks


def _build_value_vpq_sidecars(
    index,
    values_np: np.ndarray,
    subbits: int,
    *,
    value_subvecs: int = 0,
    value_subbits: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    default_subvecs = int(index.pages[0].codes.shape[1]) if index.pages else 0
    actual_subvecs = int(value_subvecs) if int(value_subvecs) > 0 else default_subvecs
    actual_subbits = int(value_subbits) if int(value_subbits) > 0 else int(subbits)
    cache_key = (actual_subvecs, actual_subbits)
    cached_by_key = getattr(index, "_value_vpq_sidecars_by_params", None)
    if isinstance(cached_by_key, dict) and cache_key in cached_by_key:
        return cached_by_key[cache_key]
    from benchmark.attention_efficiency_threeway_eval import build_pq_index

    sidecars: list[tuple[np.ndarray, np.ndarray]] = []
    for page_id, page in enumerate(index.pages):
        values = values_np[int(page.start) : int(page.start) + int(page.size)].astype(np.float32, copy=False)
        if values.shape[0] == 0 or actual_subvecs <= 0:
            sidecars.append(
                (
                    np.zeros((0, 0, 0), dtype=np.float32),
                    np.zeros((0, 0), dtype=np.uint16),
                )
            )
            continue
        codebooks, codes, _subvecs, _centroids = build_pq_index(
            values,
            0,
            values.shape[0],
            subvecs=actual_subvecs,
            subbits=actual_subbits,
            seed=90210 + 1009 * int(page_id) + int(page.start),
            max_iter=3,
        )
        sidecars.append((codebooks.astype(np.float32, copy=False), codes.astype(np.uint16, copy=False)))
    if not isinstance(cached_by_key, dict):
        cached_by_key = {}
    cached_by_key[cache_key] = sidecars
    setattr(index, "_value_vpq_sidecars_by_params", cached_by_key)
    return sidecars


def _compressed_tail_output(
    *,
    index,
    values_np: np.ndarray,
    scores_np: np.ndarray,
    ranked_cpu: np.ndarray,
    ranked_scores_cpu: np.ndarray,
    selected_cpu: np.ndarray,
    query_dim: int,
    subbits: int,
    value_bytes: int,
    mode: str,
    value_subvecs: int = 0,
    value_subbits: int = 0,
    selected_values_np: np.ndarray | None = None,
    selected_scores_override: np.ndarray | None = None,
    tail_score_scale: float = 1.0,
    tail_score_bias: float = 0.0,
) -> tuple[np.ndarray, int, int, float]:
    selected_set = set(int(tok) for tok in selected_cpu.tolist())
    tail_tokens = np.asarray([int(tok) for tok in ranked_cpu.tolist() if int(tok) not in selected_set], dtype=np.int64)
    if tail_tokens.size == 0 and selected_cpu.size == 0:
        return np.zeros((values_np.shape[-1],), dtype=np.float32), 0, 0, 0.0
    score_by_token = {
        int(tok): float(tail_score_scale) * (float(score) / float(np.sqrt(float(query_dim)))) + float(tail_score_bias)
        for tok, score in zip(ranked_cpu.tolist(), ranked_scores_cpu.tolist(), strict=False)
    }
    selected_scores = (
        selected_scores_override.astype(np.float64, copy=False)
        if selected_scores_override is not None and selected_cpu.size
        else scores_np[selected_cpu].astype(np.float64, copy=False)
        if selected_cpu.size
        else np.asarray([], dtype=np.float64)
    )
    tail_scores = np.asarray([score_by_token[int(tok)] for tok in tail_tokens], dtype=np.float64) if tail_tokens.size else np.asarray([], dtype=np.float64)
    max_score = max(
        float(np.max(selected_scores)) if selected_scores.size else -np.inf,
        float(np.max(tail_scores)) if tail_scores.size else -np.inf,
    )
    num = np.zeros((values_np.shape[-1],), dtype=np.float64)
    den = 0.0
    if selected_cpu.size:
        weights = np.exp(selected_scores - max_score)
        selected_values = (
            selected_values_np.astype(np.float64, copy=False)
            if selected_values_np is not None
            else values_np[selected_cpu].astype(np.float64, copy=False)
        )
        num += weights @ selected_values
        den += float(weights.sum())
    compressed_mb = 0.0
    compressed_count = 0
    if tail_tokens.size:
        tail_weights = np.exp(tail_scores - max_score)
        if str(mode) == "page_mean":
            starts = np.asarray([int(page.start) for page in index.pages], dtype=np.int64)
            sizes = np.asarray([int(page.size) for page in index.pages], dtype=np.int64)
            page_ids = np.searchsorted(starts, tail_tokens, side="right") - 1
            valid = (page_ids >= 0) & (page_ids < len(index.pages))
            valid &= tail_tokens < (starts[np.maximum(page_ids, 0)] + sizes[np.maximum(page_ids, 0)])
            compressed_count = int(np.sum(valid))
            for page_id in np.unique(page_ids[valid]).astype(np.int64, copy=False).tolist():
                positions = np.nonzero(valid & (page_ids == int(page_id)))[0]
                page = index.pages[int(page_id)]
                mean_v = values_np[int(page.start) : int(page.start) + int(page.size)].mean(axis=0).astype(np.float64, copy=False)
                num += float(tail_weights[positions].sum()) * mean_v
            den += float(tail_weights[valid].sum())
            compressed_mb = float(len(np.unique(page_ids[valid])) * values_np.shape[-1] * int(value_bytes)) / MB
        elif str(mode) == "vpq_value":
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
            page_ids = np.searchsorted(starts, tail_tokens, side="right") - 1
            valid = (page_ids >= 0) & (page_ids < len(index.pages))
            valid &= tail_tokens < (starts[np.maximum(page_ids, 0)] + sizes[np.maximum(page_ids, 0)])
            compressed_count = int(np.sum(valid))
            code_bytes = 1 if actual_value_subbits <= 8 else 2
            subvecs = int(value_sidecars[0][1].shape[1]) if value_sidecars else 0
            for page_id in np.unique(page_ids[valid]).astype(np.int64, copy=False).tolist():
                positions = np.nonzero(valid & (page_ids == int(page_id)))[0]
                page = index.pages[int(page_id)]
                rows = (tail_tokens[positions] - int(page.start)).astype(np.int64, copy=False)
                codebook, page_codes = value_sidecars[int(page_id)]
                codebook = codebook.astype(np.float64, copy=False)
                codes = page_codes[rows].astype(np.int64, copy=False)
                subdim = int(codebook.shape[-1])
                approx_values = np.zeros((positions.size, subvecs * subdim), dtype=np.float64)
                for sub in range(subvecs):
                    approx_values[:, sub * subdim : (sub + 1) * subdim] = codebook[sub, codes[:, sub]]
                num += tail_weights[positions] @ approx_values
            den += float(tail_weights[valid].sum())
            pages_read = int(len(np.unique(page_ids[valid])))
            codebook_bytes = (
                pages_read
                * subvecs
                * (1 << actual_value_subbits)
                * (values_np.shape[-1] // max(1, subvecs))
                * int(value_bytes)
            )
            code_bytes_total = compressed_count * subvecs * code_bytes
            compressed_mb = float(codebook_bytes + code_bytes_total) / MB
        else:
            value_codebooks = _build_value_codebooks_for_key_codes(index, values_np, int(subbits))
            starts = np.asarray([int(page.start) for page in index.pages], dtype=np.int64)
            sizes = np.asarray([int(page.size) for page in index.pages], dtype=np.int64)
            page_ids = np.searchsorted(starts, tail_tokens, side="right") - 1
            valid = (page_ids >= 0) & (page_ids < len(index.pages))
            valid &= tail_tokens < (starts[np.maximum(page_ids, 0)] + sizes[np.maximum(page_ids, 0)])
            compressed_count = int(np.sum(valid))
            code_bytes = 1 if int(subbits) <= 8 else 2
            subvecs = int(index.pages[0].codes.shape[1]) if index.pages else 0
            for page_id in np.unique(page_ids[valid]).astype(np.int64, copy=False).tolist():
                positions = np.nonzero(valid & (page_ids == int(page_id)))[0]
                page = index.pages[int(page_id)]
                rows = (tail_tokens[positions] - int(page.start)).astype(np.int64, copy=False)
                codes = _codes_cpu(page)[rows]
                codebook = value_codebooks[int(page_id)].astype(np.float64, copy=False)
                subdim = int(codebook.shape[-1])
                approx_values = np.zeros((positions.size, subvecs * subdim), dtype=np.float64)
                for sub in range(subvecs):
                    approx_values[:, sub * subdim : (sub + 1) * subdim] = codebook[sub, codes[:, sub]]
                num += tail_weights[positions] @ approx_values
            den += float(tail_weights[valid].sum())
            pages_read = int(len(np.unique(page_ids[valid])))
            codebook_bytes = pages_read * subvecs * (1 << int(subbits)) * (values_np.shape[-1] // max(1, subvecs)) * int(value_bytes)
            code_bytes_total = compressed_count * subvecs * code_bytes
            compressed_mb = float(codebook_bytes + code_bytes_total) / MB
    return (num / max(den, 1e-20)).astype(np.float32, copy=False), compressed_count, int(tail_tokens.size), float(compressed_mb)


def softmax_output(keys: np.ndarray, values: np.ndarray, query: np.ndarray, tokens: np.ndarray) -> np.ndarray:
    if tokens.size == 0:
        return np.zeros((values.shape[-1],), dtype=np.float32)
    scores, _ = attention_probs(keys[tokens], query)
    weights = np.exp(scores - float(np.max(scores))).astype(np.float32)
    weights /= max(float(weights.sum()), 1e-20)
    return (weights @ values[tokens]).astype(np.float32, copy=False)


def run() -> None:
    parser = argparse.ArgumentParser(description="Diagnose bad per-head layer-quality outliers.")
    parser.add_argument("--trace", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--decode_length", type=int, default=128000)
    parser.add_argument("--heads", default="0,8,9,23")
    parser.add_argument("--selector_mode", choices=["fullscan", "routed"], default="routed")
    parser.add_argument("--budget", type=int, default=4096)
    parser.add_argument("--rerank_candidates", type=int, default=0)
    parser.add_argument("--rerank_mode", choices=["full", "partial"], default="full")
    parser.add_argument("--partial_rerank_dims", type=int, default=32)
    parser.add_argument("--tail_samples", type=int, default=4096)
    parser.add_argument("--tail_bands", type=int, default=8)
    parser.add_argument("--tail_seed", type=int, default=0)
    parser.add_argument("--tail_sampling", choices=["random", "linspace", "systematic"], default="random")
    parser.add_argument("--tail_mode", choices=["sample", "pq_value", "vpq_value", "page_mean"], default="sample")
    parser.add_argument("--tail_blend", type=float, default=1.0)
    parser.add_argument("--tail_repeats", type=int, default=1)
    parser.add_argument(
        "--tail_correction_cap",
        type=float,
        default=0.0,
        help="If >0, cap ||tail_estimate - selected_only|| to this multiple of ||selected_only||.",
    )
    parser.add_argument(
        "--tail_variance_gate",
        type=float,
        default=0.0,
        help="If >0 and repeated tail estimates disagree above this relative threshold, shrink or disable tail correction.",
    )
    parser.add_argument("--tail_variance_action", choices=["shrink", "fallback"], default="shrink")
    parser.add_argument("--static_prefix", type=int, default=128)
    parser.add_argument("--static_suffix", type=int, default=128)
    parser.add_argument("--page_size", type=int, default=5632)
    parser.add_argument("--subvecs", type=int, default=4)
    parser.add_argument("--subbits", type=int, default=6)
    parser.add_argument("--kmeans_iters", type=int, default=3)
    parser.add_argument("--key_bytes", type=int, default=2)
    parser.add_argument("--value_bytes", type=int, default=2)
    parser.add_argument("--nprobes", default="16,32,64,128,256,512")
    parser.add_argument("--router_prototypes", type=int, default=16)
    parser.add_argument("--router_merge_rel", type=float, default=0.05)
    parser.add_argument("--router_merge_var", type=float, default=0.0)
    parser.add_argument("--router_max_groups", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    trace = load_trace(args.trace)
    q_indices = trace.q_indices_for_decodes([int(args.decode_length)])
    if not q_indices:
        raise ValueError(f"no qidx for decode length {args.decode_length}")
    qidx = int(q_indices[0])
    position = int(trace.positions[qidx])
    context_len = position + 1
    dynamic_start = min(max(0, int(args.static_prefix)), int(trace.input_len))
    indexed_end = max(
        dynamic_start,
        min(context_len - max(0, int(args.static_suffix)), trace.keys.shape[1]),
    )
    nprobes = parse_csv_ints(args.nprobes)
    heads = parse_csv_ints(args.heads)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True), encoding="utf-8")

    needed_kv_heads = sorted({trace.kv_head_for(h) for h in heads})
    index_cache = {}
    torch_k_cache = {}
    torch_v_cache = {}
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
        torch_k_cache[kv_head] = torch.as_tensor(keys_np, dtype=torch.float32, device=device)
        torch_v_cache[kv_head] = torch.as_tensor(
            trace.values[kv_head, :context_len].astype(np.float32, copy=False),
            dtype=torch.float32,
            device=device,
        )

    rows = []
    for head in heads:
        kv_head = trace.kv_head_for(head)
        keys_np = trace.keys[kv_head, :context_len].astype(np.float32, copy=False)
        values_np = trace.values[kv_head, :context_len].astype(np.float32, copy=False)
        query_np = trace.queries[head, qidx].astype(np.float32, copy=False)
        scores_np, probs_np = attention_probs(keys_np, query_np)
        dense = (probs_np.astype(np.float32) @ values_np).astype(np.float32, copy=False)
        query = torch.as_tensor(query_np, dtype=torch.float32, device=device)
        index = index_cache[kv_head]
        pending = list(range(max(0, int(index.pending_start)), max(0, min(int(index.indexed_end), context_len))))
        base = unique_tokens(static_tokens(position, args.static_prefix, args.static_suffix) + pending, context_len=context_len)
        ranked_t, ranked_scores_t, _selector_seconds, selector_mb, chosen_nprobe = rank_paged_pq(
            query,
            index,
            mode=str(args.selector_mode),
            selector_backend="torch",
            nprobes=nprobes,
            budget=int(args.budget),
            key_bytes=int(args.key_bytes),
            subbits=int(args.subbits),
        )
        ranked_cpu = ranked_t.detach().cpu().numpy().astype(np.int64, copy=False)
        ranked_scores_cpu = ranked_scores_t.detach().cpu().numpy().astype(np.float32, copy=False)
        rerank_count = 0
        rerank_key_mb = 0.0
        if int(args.rerank_candidates) > 0 and ranked_cpu.size:
            rerank_count = min(int(args.rerank_candidates), int(ranked_cpu.size))
            rerank_tokens = ranked_cpu[:rerank_count]
            if str(args.rerank_mode) == "partial":
                dim_count = max(1, min(int(args.partial_rerank_dims), int(query_np.shape[0])))
                dims = np.argsort(-np.abs(query_np), kind="stable")[:dim_count]
                rerank_scores = keys_np[rerank_tokens[:, None], dims.reshape(1, -1)].astype(np.float32, copy=False) @ query_np[
                    dims
                ].astype(np.float32, copy=False)
                rerank_key_mb = float(rerank_count * dim_count * int(args.key_bytes)) / MB
            else:
                rerank_scores = scores_np[rerank_tokens]
                rerank_key_mb = float(rerank_count * trace.head_dim * int(args.key_bytes)) / MB
            rerank_order = np.argsort(-rerank_scores, kind="stable")
            reranked = rerank_tokens[rerank_order].astype(np.int64, copy=False)
            reranked_set = set(int(tok) for tok in reranked.tolist())
            rest = np.asarray([int(tok) for tok in ranked_cpu.tolist() if int(tok) not in reranked_set], dtype=np.int64)
            ranked_cpu = np.concatenate([reranked, rest]) if rest.size else reranked
        base_set = set(int(tok) for tok in base)
        add = [int(tok) for tok in ranked_cpu.tolist() if int(tok) < context_len and int(tok) not in base_set][: int(args.budget)]
        selected_cpu = np.asarray(unique_tokens(base + add, context_len=context_len), dtype=np.int64)
        selected = torch.as_tensor(selected_cpu, dtype=torch.long, device=device)
        selected_only = softmax_output(keys_np, values_np, query_np, selected_cpu)
        tail_blend = float(args.tail_blend)
        if tail_blend <= 0.0:
            approx = selected_only.astype(np.float32, copy=False)
            tail_count = 0
            tail_population = max(0, int(context_len) - int(selected_cpu.size))
            tail_disagreement = 0.0
            tail_gate_scale = 0.0
            applied_cap_scale = 0.0
            tail_estimator_mb = 0.0
        elif str(args.tail_mode) in {"pq_value", "page_mean"}:
            approx_tail, tail_count, tail_population, tail_estimator_mb = _compressed_tail_output(
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
            )
            tail_disagreement = 0.0
            tail_gate_scale = 1.0
            applied_cap_scale = 1.0
            if tail_blend >= 1.0:
                approx = approx_tail
            else:
                approx = (selected_only.astype(np.float32, copy=False) + tail_blend * (approx_tail - selected_only)).astype(
                    np.float32,
                    copy=False,
                )
        else:
            tail_estimates = []
            tail_count = 0
            tail_population = 0
            for repeat in range(max(1, int(args.tail_repeats))):
                approx_t, one_tail_count, tail_population, _attn_seconds = selected_plus_tail_output(
                    torch_k_cache[kv_head],
                    torch_v_cache[kv_head],
                    query,
                    selected,
                    ranked_cpu,
                    scores_np,
                    context_len=context_len,
                    samples=int(args.tail_samples),
                    bands=int(args.tail_bands),
                    seed=int(args.tail_seed) + 104729 * repeat,
                    qidx=qidx,
                    head=head,
                    sampling=str(args.tail_sampling),
                )
                tail_count += int(one_tail_count)
                tail_estimates.append(approx_t.detach().cpu().numpy().astype(np.float32, copy=False))
            approx_tail = np.mean(np.stack(tail_estimates, axis=0), axis=0).astype(np.float32, copy=False)
            tail_disagreement = 0.0
            tail_gate_scale = 1.0
            if len(tail_estimates) > 1:
                denom = max(float(np.linalg.norm(approx_tail.astype(np.float64))), 1e-20)
                tail_disagreement = max(
                    float(np.linalg.norm((estimate - approx_tail).astype(np.float64))) / denom for estimate in tail_estimates
                )
                if float(args.tail_variance_gate) > 0.0 and tail_disagreement > float(args.tail_variance_gate):
                    if str(args.tail_variance_action) == "fallback":
                        tail_gate_scale = 0.0
                    else:
                        tail_gate_scale = max(0.0, min(1.0, float(args.tail_variance_gate) / tail_disagreement))
                    approx_tail = (selected_only + tail_gate_scale * (approx_tail - selected_only)).astype(np.float32, copy=False)
            correction = approx_tail - selected_only
            correction_norm = float(np.linalg.norm(correction.astype(np.float64)))
            selected_norm_for_cap = float(np.linalg.norm(selected_only.astype(np.float64)))
            applied_cap_scale = 1.0
            if float(args.tail_correction_cap) > 0.0 and correction_norm > 0.0:
                max_correction = float(args.tail_correction_cap) * max(selected_norm_for_cap, 1e-20)
                applied_cap_scale = min(1.0, max_correction / correction_norm)
                approx_tail = (selected_only + applied_cap_scale * correction).astype(np.float32, copy=False)
            if tail_blend >= 1.0:
                approx = approx_tail
            else:
                approx = (selected_only.astype(np.float32, copy=False) + tail_blend * (approx_tail - selected_only)).astype(
                    np.float32,
                    copy=False,
                )
            tail_estimator_mb = float(tail_count * trace.head_dim * (int(args.key_bytes) + int(args.value_bytes))) / MB
        oracle_k = int(selected_cpu.size)
        oracle_tokens = np.argsort(-probs_np, kind="stable")[:oracle_k].astype(np.int64, copy=False)
        oracle_same_k = softmax_output(keys_np, values_np, query_np, oracle_tokens)
        selected_set = set(int(x) for x in selected_cpu.tolist())
        oracle_set = set(int(x) for x in oracle_tokens.tolist())
        top1024 = np.argsort(-probs_np, kind="stable")[:1024].astype(np.int64, copy=False)
        top4096 = np.argsort(-probs_np, kind="stable")[:4096].astype(np.int64, copy=False)
        dense_norm = float(np.linalg.norm(dense.astype(np.float64)))
        err = approx.astype(np.float64) - dense.astype(np.float64)
        rows.append(
            {
                "head": head,
                "kv_head": kv_head,
                "qidx": qidx,
                "context_len": context_len,
                "nprobe": int(chosen_nprobe),
                "selected_tokens": int(selected_cpu.size),
                "tail_samples": int(tail_count),
                "tail_mode": str(args.tail_mode),
                "tail_blend": float(max(0.0, min(1.0, tail_blend))),
                "tail_repeats": int(max(1, int(args.tail_repeats))) if tail_blend > 0.0 else 0,
                "tail_correction_cap": float(args.tail_correction_cap),
                "tail_correction_cap_scale": float(applied_cap_scale) if tail_blend > 0.0 else 0.0,
                "tail_disagreement": float(tail_disagreement) if tail_blend > 0.0 else 0.0,
                "tail_variance_gate": float(args.tail_variance_gate),
                "tail_variance_gate_scale": float(tail_gate_scale) if tail_blend > 0.0 else 0.0,
                "tail_variance_action": str(args.tail_variance_action),
                "tail_population": int(tail_population),
                "candidate_tokens": int(ranked_cpu.size),
                "rerank_candidates": int(rerank_count),
                "rerank_mode": str(args.rerank_mode),
                "partial_rerank_dims": int(args.partial_rerank_dims) if str(args.rerank_mode) == "partial" else 0,
                "mass_selected": float(probs_np[selected_cpu].sum()),
                "mass_oracle_same_k": float(probs_np[oracle_tokens].sum()),
                "top1024_recall": float(len(selected_set.intersection(int(x) for x in top1024.tolist())) / max(1, top1024.size)),
                "top4096_recall": float(len(selected_set.intersection(int(x) for x in top4096.tolist())) / max(1, top4096.size)),
                "selected_oracle_jaccard": float(len(selected_set & oracle_set) / max(1, len(selected_set | oracle_set))),
                "dense_norm": dense_norm,
                "approx_norm": float(np.linalg.norm(approx.astype(np.float64))),
                "abs_l2": float(np.linalg.norm(err)),
                "rel_l2": float(np.linalg.norm(err) / max(dense_norm, 1e-20)),
                "cosine": _output_error_metrics(dense, approx)["output_cosine"],
                "selected_only_rel_l2": _output_error_metrics(dense, selected_only)["output_relative_l2"],
                "oracle_same_k_rel_l2": _output_error_metrics(dense, oracle_same_k)["output_relative_l2"],
                "selector_MB_per_query": float(selector_mb) + float(rerank_key_mb),
                "pq_selector_MB_per_query": float(selector_mb),
                "rerank_key_MB_per_query": float(rerank_key_mb),
                "exact_KV_MB_per_query": float(selected_cpu.size * trace.head_dim * (int(args.key_bytes) + int(args.value_bytes))) / MB,
                "tail_estimator_MB_per_query": float(tail_estimator_mb),
            }
        )

    with (out_dir / "head_diagnostics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for row in rows for k in row}))
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "head_diagnostics.json").write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[diagnose_layer_heads] wrote {out_dir}")


if __name__ == "__main__":
    run()
