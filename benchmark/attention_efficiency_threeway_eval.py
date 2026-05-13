#!/usr/bin/env python3
"""Three-way attention-retrieval proxy on real QKV traces.

This evaluator is intentionally narrower than attention_efficiency_eval.py. It
compares:

* RetroInfer-style centroid routing: score all centroids, then consume top
  clusters until a mass target is reached.
* RetrievalAttention-style Roar graph: build a Q-K projected token graph from
  prefill queries and traverse it at decode.
* Hybrid centroid graph: build a Q-C projected graph over RetroInfer centroids
  and traverse centroids instead of scoring all centroids.

The objective is algorithmic cost, not GPU latency. Graph construction is
treated as prefill/offline cost; per-query decode cost is byte-accounted.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

POPCOUNT_U8 = np.asarray([int(x).bit_count() for x in range(256)], dtype=np.uint8)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from cache_hub.roargraph_cpp_backend import build_roar_graph_csr_cpp, roargraph_cpp_available
except Exception:  # pragma: no cover - optional extension path
    build_roar_graph_csr_cpp = None

    def roargraph_cpp_available() -> bool:
        return False


@dataclass
class MethodResult:
    method: str
    target_mass: float
    reached: bool
    mass: float
    output_cos: float
    exact_tokens: int
    represented_tokens: int
    estimated_mb: float
    score_reads: int
    score_elements: float
    index_bytes: int
    final_kv_reads: int
    value_sum_reads: int
    edge_reads: int
    offset_reads: int
    nodes_visited: int
    clusters_scored: int
    clusters_selected: int


class CsrGraph:
    def __init__(self, vectors: np.ndarray, offsets: np.ndarray, neighbors: np.ndarray, dynamic_start: int, dynamic_end: int):
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.offsets = np.ascontiguousarray(offsets, dtype=np.uint32)
        self.neighbors_arr = np.ascontiguousarray(neighbors, dtype=np.int32)
        self.dynamic_start = int(dynamic_start)
        self.dynamic_end = int(dynamic_end)
        self.indegree = np.zeros((self.vectors.shape[0],), dtype=np.int64)
        valid = self.neighbors_arr[(self.neighbors_arr >= 0) & (self.neighbors_arr < self.vectors.shape[0])]
        if valid.size:
            self.indegree += np.bincount(valid, minlength=self.vectors.shape[0]).astype(np.int64)

    def neighbors(self, node: int) -> list[int]:
        node = int(node)
        if node < 0 or node + 1 >= self.offsets.shape[0]:
            return []
        start = int(self.offsets[node])
        end = int(self.offsets[node + 1])
        return [int(x) for x in self.neighbors_arr[start:end].tolist()]

    def seeds(self, count: int, tail_end: int | None = None) -> list[int]:
        count = max(1, int(count))
        out: list[int] = []
        if self.dynamic_end > self.dynamic_start:
            valid = np.arange(self.dynamic_start, self.dynamic_end, dtype=np.int64)
            deg = self.indegree[valid]
            if deg.size:
                order = np.argsort(-deg, kind="stable")[:count]
                out.extend(int(valid[i]) for i in order)
        if tail_end is not None:
            hi = min(int(tail_end), self.dynamic_end)
            lo = max(self.dynamic_start, hi - count)
            out.extend(range(lo, hi))
        if self.dynamic_end > self.dynamic_start:
            anchors = np.linspace(self.dynamic_start, self.dynamic_end - 1, num=min(count, self.dynamic_end - self.dynamic_start))
            out.extend(int(round(float(x))) for x in anchors)
        return unique(out, count * 3, self.dynamic_start, self.dynamic_end)


def parse_float_list(text: str) -> list[float]:
    vals = []
    for part in str(text).replace(";", ",").replace(":", ",").split(","):
        part = part.strip()
        if part:
            vals.append(float(part))
    if not vals:
        raise ValueError(f"empty float list: {text!r}")
    return vals


def parse_int_list(text: str) -> list[int]:
    vals = []
    for part in re.split(r"[,;:\s]+", str(text)):
        part = part.strip()
        if part:
            vals.append(int(part))
    if not vals:
        raise ValueError(f"empty int list: {text!r}")
    return vals


def sorted_unique_ints(text: str) -> list[int]:
    return sorted(set(parse_int_list(text)))


def parse_name_set(text: str) -> set[str]:
    vals = {
        part.strip().lower()
        for part in re.split(r"[,;:\s]+", str(text))
        if part.strip()
    }
    return vals


def parse_magicpig_configs(text: str) -> list[tuple[int, int]]:
    vals: list[tuple[int, int]] = []
    for part in re.split(r"[,;]+", str(text)):
        part = part.strip().lower()
        if not part:
            continue
        if ":" in part:
            left, right = part.split(":", 1)
        elif "x" in part:
            left, right = part.split("x", 1)
        else:
            raise ValueError(f"MagicPIG config must be K:L, got {part!r}")
        vals.append((int(left), int(right)))
    if not vals:
        raise ValueError(f"empty MagicPIG config list: {text!r}")
    return vals


def parse_magicpig_ladder(text: str) -> list[tuple[int, int, int]]:
    vals: list[tuple[int, int, int]] = []
    for part in re.split(r"[,;]+", str(text)):
        part = part.strip().lower()
        if not part:
            continue
        chunks = re.split(r"[:x]", part)
        if len(chunks) == 2:
            vals.append((int(chunks[0]), int(chunks[1]), 2))
        elif len(chunks) == 3:
            vals.append((int(chunks[0]), int(chunks[1]), int(chunks[2])))
        else:
            raise ValueError(f"MagicPIG ladder config must be K:L[:threshold], got {part!r}")
    if not vals:
        raise ValueError(f"empty MagicPIG ladder config list: {text!r}")
    return vals


def parse_pariskv_configs(text: str) -> list[tuple[int, int, float]]:
    vals: list[tuple[int, int, float]] = []
    for part in re.split(r"[,;]+", str(text)):
        part = part.strip().lower()
        if not part:
            continue
        chunks = part.split(":")
        if len(chunks) != 3:
            raise ValueError(f"ParisKV config must be bits:tables:ratio, got {part!r}")
        vals.append((int(chunks[0]), int(chunks[1]), float(chunks[2])))
    if not vals:
        raise ValueError(f"empty ParisKV config list: {text!r}")
    return vals


def parse_pariskv_ladder(text: str) -> list[tuple[int, int, float, int]]:
    vals: list[tuple[int, int, float, int]] = []
    for part in re.split(r"[,;]+", str(text)):
        part = part.strip().lower()
        if not part:
            continue
        chunks = part.split(":")
        if len(chunks) == 3:
            vals.append((int(chunks[0]), int(chunks[1]), float(chunks[2]), 16))
        elif len(chunks) == 4:
            vals.append((int(chunks[0]), int(chunks[1]), float(chunks[2]), int(chunks[3])))
        else:
            raise ValueError(f"ParisKV ladder config must be bits:tables:ratio[:rerank_dims], got {part!r}")
    if not vals:
        raise ValueError(f"empty ParisKV ladder config list: {text!r}")
    return vals


def unique(tokens: Iterable[int], limit: int, lo: int, hi: int) -> list[int]:
    out = []
    seen = set()
    for tok in tokens:
        tok = int(tok)
        if tok < int(lo) or tok >= int(hi) or tok in seen:
            continue
        out.append(tok)
        seen.add(tok)
        if len(out) >= int(limit):
            break
    return out


def static_tokens(position: int, prefix: int, suffix: int) -> list[int]:
    max_tok = int(position)
    return unique(
        list(range(0, min(int(prefix), max_tok + 1)))
        + list(range(max(0, max_tok - int(suffix) + 1), max_tok + 1)),
        max_tok + 1,
        0,
        max_tok + 1,
    )


def byte_cost(
    args,
    *,
    score_reads: int,
    final_kv_reads: int,
    value_sum_reads: int,
    edge_reads: int,
    offset_reads: int,
    score_elements: float | None = None,
    index_bytes: int = 0,
) -> float:
    head_dim = int(args.head_dim)
    total = 0
    if score_elements is None:
        score_elements = float(int(score_reads) * head_dim)
    total += float(score_elements) * int(args.score_key_bytes_per_element)
    total += int(final_kv_reads) * head_dim * (
        int(args.attn_key_bytes_per_element) + int(args.value_bytes_per_element)
    )
    total += int(value_sum_reads) * head_dim * int(args.value_bytes_per_element)
    total += int(edge_reads) * int(args.edge_index_bytes)
    total += int(offset_reads) * int(args.graph_offset_bytes)
    total += int(index_bytes)
    return float(total / (1024.0 * 1024.0))


def packed_bit_bytes(items: int, bits_per_item: int) -> int:
    return int(math.ceil(max(0, int(items)) * max(0, int(bits_per_item)) / 8.0))


def pq_code_bytes(args) -> int:
    return 1 if int(args.pqcache_subbits) <= 8 else 2


def ivfpq_method_name(args, centroids: np.ndarray) -> str:
    mode = str(getattr(args, "ivfpq_online_mode", "snapshot")).strip().lower()
    base = f"ivfpq_global_pq_oracle_c{centroids.shape[0]}"
    return base if mode == "snapshot" else f"{base}_{mode}"


def ivfpq_fixed_method_name(args, centroids: np.ndarray, nprobe: int, *, pq_logits: bool) -> str:
    mode = str(getattr(args, "ivfpq_online_mode", "snapshot")).strip().lower()
    suffix = "" if mode == "snapshot" else f"_{mode}"
    kind = "pqlogit" if pq_logits else "exactkv"
    return f"ivfpq_global_pq_fixed_{kind}_c{centroids.shape[0]}_n{int(nprobe)}{suffix}"


def ivfpq_online_update_cost(
    args,
    *,
    dynamic_count: int,
    centroids: np.ndarray,
    subvecs: int,
    centroids_per_subvec: int,
) -> tuple[float, int]:
    """Amortized per-query online-index maintenance cost for IVF+global-PQ.

    The selector quality is still evaluated with a snapshot index at the cutoff.
    This cost term makes the byte model less optimistic for long decode by
    charging the work needed to encode/append newly generated tokens.
    """
    mode = str(getattr(args, "ivfpq_online_mode", "snapshot")).strip().lower()
    if mode == "snapshot":
        return 0.0, 0

    head_dim = int(args.head_dim)
    subdim = head_dim // max(1, int(subvecs))
    amortize = int(getattr(args, "ivfpq_update_amortize_queries", 0) or getattr(args, "kv_group_size", 1) or 1)
    amortize = max(1, amortize)

    # Per generated token leaving the static suffix: read the key once, score
    # all coarse centroids for assignment, and encode against global PQ
    # codebooks before appending (posting id, PQ code).
    update_score_elements = float(
        head_dim
        + int(centroids.shape[0]) * head_dim
        + int(subvecs) * int(centroids_per_subvec) * subdim
    )
    update_index_bytes = int(subvecs) * pq_code_bytes(args) + int(args.edge_index_bytes)

    if mode in {"online_centroid", "periodic_rebuild"}:
        # Simple streaming centroid update: read + write centroid vector and
        # touch a count/metadata word. This is approximate but prevents
        # centroid maintenance from being free.
        update_index_bytes += (
            2 * head_dim * int(args.score_key_bytes_per_element)
            + int(args.graph_offset_bytes)
        )

    if mode == "periodic_rebuild":
        interval = max(1, int(getattr(args, "ivfpq_rebuild_interval", 0) or 1))
        coarse_iters = max(1, int(getattr(args, "ivfpq_coarse_iters", 1)))
        pq_iters = max(1, int(getattr(args, "pqcache_kmeans_iters", 1)))
        rebuild_score_elements = float(max(0, int(dynamic_count))) * float(
            coarse_iters * int(centroids.shape[0]) * head_dim
            + pq_iters * int(subvecs) * int(centroids_per_subvec) * subdim
        )
        update_score_elements += rebuild_score_elements / float(interval)

    return update_score_elements / float(amortize), int(math.ceil(update_index_bytes / float(amortize)))


def maybe_record_candidate_frontier(
    args,
    *,
    method: str,
    budget_kind: str,
    budget_value: int,
    candidate_tokens: np.ndarray,
    base: list[int],
    probs: np.ndarray,
    score_reads: int,
    score_elements: float | None = None,
    index_bytes: int = 0,
    edge_reads: int = 0,
    offset_reads: int = 0,
) -> None:
    rows = getattr(args, "candidate_frontier_rows", None)
    if rows is None:
        return
    base_u = unique(base, len(base), 0, probs.shape[0])
    cand = unique([int(x) for x in np.asarray(candidate_tokens).tolist()], probs.shape[0], 0, probs.shape[0])
    represented = unique(base_u + cand, len(base_u) + len(cand), 0, probs.shape[0])
    mass = float(probs[np.asarray(represented, dtype=np.int64)].sum()) if represented else 0.0
    ctx = getattr(args, "current_eval_context", {})
    rows.append(
        {
            **ctx,
            "method": str(method),
            "budget_kind": str(budget_kind),
            "budget_value": int(budget_value),
            "candidate_tokens": int(len(cand)),
            "represented_tokens": int(len(represented)),
            "oracle_mass": mass,
            "estimated_mb_pre_pq": byte_cost(
                args,
                score_reads=int(score_reads),
                score_elements=score_elements,
                index_bytes=int(index_bytes),
                final_kv_reads=0,
                value_sum_reads=0,
                edge_reads=int(edge_reads),
                offset_reads=int(offset_reads),
            ),
        }
    )


def build_retro_clusters(keys: np.ndarray, dynamic_start: int, dynamic_end: int, cluster_size: int):
    ranges: list[tuple[int, int]] = []
    centroids = []
    value_indices = []
    for start in range(int(dynamic_start), int(dynamic_end), int(cluster_size)):
        end = min(int(dynamic_end), start + int(cluster_size))
        if end <= start:
            continue
        block = keys[start:end].astype(np.float32, copy=False)
        centroid = block.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        centroids.append(centroid.astype(np.float32, copy=False))
        ranges.append((start, end))
        value_indices.append(np.arange(start, end, dtype=np.int64))
    if not centroids:
        return np.empty((0, keys.shape[-1]), dtype=np.float32), [], []
    return np.stack(centroids, axis=0), ranges, value_indices


def exact_topk_rows(
    queries: np.ndarray,
    query_positions: np.ndarray,
    keys: np.ndarray,
    *,
    k: int,
    dynamic_start: int,
    dynamic_end: int,
    score_scale: float,
    chunk_rows: int,
) -> np.ndarray:
    rows = []
    key_block = keys[int(dynamic_start):int(dynamic_end)].astype(np.float32, copy=False)
    key_abs = np.arange(int(dynamic_start), int(dynamic_end), dtype=np.int64)
    if key_block.shape[0] <= 0:
        return np.empty((0, int(k)), dtype=np.int32)
    for start in range(0, queries.shape[0], int(chunk_rows)):
        end = min(queries.shape[0], start + int(chunk_rows))
        q = queries[start:end].astype(np.float32, copy=False)
        scores = q @ key_block.T
        scores *= float(score_scale)
        pos = query_positions[start:end].reshape(-1, 1)
        scores[key_abs.reshape(1, -1) > pos] = -np.inf
        take = min(int(k), key_block.shape[0])
        if take <= 0:
            continue
        idx = np.argpartition(-scores, kth=take - 1, axis=1)[:, :take]
        part = np.take_along_axis(scores, idx, axis=1)
        order = np.argsort(-part, axis=1)
        top = np.take_along_axis(idx, order, axis=1)
        top_scores = np.take_along_axis(part, order, axis=1)
        toks = key_abs[top].astype(np.int32, copy=False)
        toks[~np.isfinite(top_scores)] = -1
        if take < int(k):
            pad = np.full((toks.shape[0], int(k) - take), -1, dtype=np.int32)
            toks = np.concatenate([toks, pad], axis=1)
        rows.append(toks)
    return np.ascontiguousarray(np.concatenate(rows, axis=0), dtype=np.int32) if rows else np.empty((0, int(k)), dtype=np.int32)


def exact_topk_centroid_rows(
    queries: np.ndarray,
    query_positions: np.ndarray,
    centroids: np.ndarray,
    ranges: list[tuple[int, int]],
    *,
    k: int,
    score_scale: float,
    chunk_rows: int,
) -> np.ndarray:
    if centroids.shape[0] <= 0:
        return np.empty((0, int(k)), dtype=np.int32)
    starts = np.asarray([s for s, _e in ranges], dtype=np.int64)
    rows = []
    for start in range(0, queries.shape[0], int(chunk_rows)):
        end = min(queries.shape[0], start + int(chunk_rows))
        q = queries[start:end].astype(np.float32, copy=False)
        scores = q @ centroids.T
        scores *= float(score_scale)
        scores[starts.reshape(1, -1) > query_positions[start:end].reshape(-1, 1)] = -np.inf
        take = min(int(k), centroids.shape[0])
        idx = np.argpartition(-scores, kth=take - 1, axis=1)[:, :take]
        part = np.take_along_axis(scores, idx, axis=1)
        order = np.argsort(-part, axis=1)
        top = np.take_along_axis(idx, order, axis=1).astype(np.int32, copy=False)
        top_scores = np.take_along_axis(part, order, axis=1)
        top[~np.isfinite(top_scores)] = -1
        rows.append(top)
    return np.ascontiguousarray(np.concatenate(rows, axis=0), dtype=np.int32) if rows else np.empty((0, int(k)), dtype=np.int32)


def build_projected_graph(knn: np.ndarray, vectors: np.ndarray, args, *, dynamic_start: int, dynamic_end: int) -> CsrGraph:
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    if str(args.roar_backend) == "cpp" and build_roar_graph_csr_cpp is not None and roargraph_cpp_available():
        offsets, neighbors, _meta = build_roar_graph_csr_cpp(
            np.ascontiguousarray(knn, dtype=np.int32),
            vectors,
            dynamic_start=int(dynamic_start),
            dynamic_end=int(dynamic_end),
            nq=int(args.roar_nq),
            degree_cap=int(args.graph_degree),
            cand_limit=int(args.roar_l),
            enable_enhance=bool(args.roar_enhance),
            enhance_limit=int(args.roar_enhance_l),
            entry_mode=str(args.roar_entry),
            max_query_per_pivot=int(args.roar_max_query_per_pivot),
            num_threads=int(args.roar_threads),
        )
        return CsrGraph(vectors, np.asarray(offsets), np.asarray(neighbors), dynamic_start, dynamic_end)

    buckets: dict[int, list[int]] = {}
    for row in np.asarray(knn, dtype=np.int32):
        valid = [int(x) for x in row.tolist() if int(dynamic_start) <= int(x) < int(dynamic_end)]
        if len(valid) < 2:
            continue
        pivot = valid[0]
        cur = buckets.setdefault(pivot, [])
        for tok in valid[1:int(args.roar_nq)]:
            if tok != pivot and tok not in cur:
                cur.append(tok)
    adjacency: list[list[int]] = [[] for _ in range(vectors.shape[0])]
    for src, cands in buckets.items():
        if not cands:
            continue
        src_vec = vectors[int(src)]
        cand_arr = np.asarray(cands, dtype=np.int64)
        sims = vectors[cand_arr] @ src_vec
        order = np.argsort(-sims, kind="stable")[: int(args.graph_degree)]
        for dst in cand_arr[order].tolist():
            if dst not in adjacency[src]:
                adjacency[src].append(int(dst))
            if src not in adjacency[int(dst)] and len(adjacency[int(dst)]) < int(args.graph_degree):
                adjacency[int(dst)].append(int(src))
    offsets = np.zeros((vectors.shape[0] + 1,), dtype=np.uint32)
    flat = []
    for i, nbrs in enumerate(adjacency):
        clean = unique(nbrs, int(args.graph_degree), dynamic_start, dynamic_end)
        flat.extend(clean)
        offsets[i + 1] = len(flat)
    return CsrGraph(vectors, offsets, np.asarray(flat, dtype=np.int32), dynamic_start, dynamic_end)


def approximate_output(
    scores: np.ndarray,
    values: np.ndarray,
    exact_tokens: list[int],
    approx_logits: list[float],
    approx_value_sums: list[np.ndarray],
    approx_sizes: list[int],
) -> np.ndarray:
    if not exact_tokens and not approx_logits:
        return np.zeros((values.shape[-1],), dtype=np.float32)
    parts = []
    vals = []
    if exact_tokens:
        idx = np.asarray(exact_tokens, dtype=np.int64)
        parts.append(scores[idx].astype(np.float32, copy=False))
        vals.append(values[idx].astype(np.float32, copy=False))
    if approx_logits:
        logits = np.asarray(approx_logits, dtype=np.float32)
        sizes = np.asarray(approx_sizes, dtype=np.float32)
        parts.append(logits + np.log(np.maximum(sizes, 1.0)))
        vals.append(np.stack(approx_value_sums, axis=0).astype(np.float32, copy=False) / np.maximum(sizes[:, None], 1.0))
    logits_all = np.concatenate(parts, axis=0)
    vals_all = np.concatenate(vals, axis=0)
    logits_all = logits_all - np.max(logits_all)
    w = np.exp(logits_all).astype(np.float32)
    return (w[:, None] * vals_all).sum(axis=0) / max(float(w.sum()), 1e-20)


def evaluate_candidate(
    args,
    method: str,
    target: float,
    scores: np.ndarray,
    values: np.ndarray,
    probs: np.ndarray,
    dense_out: np.ndarray,
    exact_tokens: list[int],
    represented_tokens: list[int],
    *,
    score_reads: int,
    score_elements: float | None = None,
    index_bytes: int = 0,
    value_sum_reads: int = 0,
    edge_reads: int = 0,
    offset_reads: int = 0,
    nodes_visited: int = 0,
    clusters_scored: int = 0,
    clusters_selected: int = 0,
    approx_logits: list[float] | None = None,
    approx_value_sums: list[np.ndarray] | None = None,
    approx_sizes: list[int] | None = None,
) -> MethodResult:
    exact_tokens = unique(exact_tokens, len(exact_tokens), 0, scores.shape[0])
    represented_tokens = unique(represented_tokens, len(represented_tokens), 0, scores.shape[0])
    mass = float(probs[np.asarray(represented_tokens, dtype=np.int64)].sum()) if represented_tokens else 0.0
    approx_logits = approx_logits or []
    approx_value_sums = approx_value_sums or []
    approx_sizes = approx_sizes or []
    if approx_logits:
        sparse_out = approximate_output(scores, values, exact_tokens, approx_logits, approx_value_sums, approx_sizes)
    elif exact_tokens:
        idx = np.asarray(exact_tokens, dtype=np.int64)
        logits = scores[idx].astype(np.float32)
        w = np.exp(logits - np.max(logits))
        sparse_out = (w[:, None] * values[idx].astype(np.float32)).sum(axis=0) / max(float(w.sum()), 1e-20)
    else:
        sparse_out = np.zeros_like(dense_out)
    denom = max(float(np.linalg.norm(sparse_out) * np.linalg.norm(dense_out)), 1e-20)
    cos = float(np.dot(sparse_out, dense_out) / denom)
    return MethodResult(
        method=method,
        target_mass=float(target),
        reached=bool(mass >= float(target)),
        mass=mass,
        output_cos=cos,
        exact_tokens=len(exact_tokens),
        represented_tokens=len(represented_tokens),
        estimated_mb=byte_cost(
            args,
            score_reads=score_reads,
            score_elements=score_elements,
            index_bytes=index_bytes,
            final_kv_reads=len(exact_tokens),
            value_sum_reads=value_sum_reads,
            edge_reads=edge_reads,
            offset_reads=offset_reads,
        ),
        score_reads=int(score_reads),
        score_elements=float(score_elements if score_elements is not None else int(score_reads) * int(args.head_dim)),
        index_bytes=int(index_bytes),
        final_kv_reads=len(exact_tokens),
        value_sum_reads=int(value_sum_reads),
        edge_reads=int(edge_reads),
        offset_reads=int(offset_reads),
        nodes_visited=int(nodes_visited),
        clusters_scored=int(clusters_scored),
        clusters_selected=int(clusters_selected),
    )


def evaluate_candidate_with_logits(
    args,
    method: str,
    target: float,
    scores: np.ndarray,
    values: np.ndarray,
    probs: np.ndarray,
    dense_out: np.ndarray,
    exact_tokens: list[int],
    represented_tokens: list[int],
    candidate_logits: dict[int, float],
    *,
    estimated_mb: float,
    score_reads: int,
    score_elements: float,
    index_bytes: int,
    final_kv_reads: int,
    nodes_visited: int = 0,
) -> MethodResult:
    exact_tokens = unique(exact_tokens, len(exact_tokens), 0, scores.shape[0])
    represented_tokens = unique(represented_tokens, len(represented_tokens), 0, scores.shape[0])
    mass = float(probs[np.asarray(represented_tokens, dtype=np.int64)].sum()) if represented_tokens else 0.0
    if represented_tokens:
        logits = np.asarray([candidate_logits.get(int(tok), float(scores[int(tok)])) for tok in represented_tokens], dtype=np.float32)
        vals = values[np.asarray(represented_tokens, dtype=np.int64)].astype(np.float32, copy=False)
        w = np.exp(logits - np.max(logits))
        sparse_out = (w[:, None] * vals).sum(axis=0) / max(float(w.sum()), 1e-20)
    else:
        sparse_out = np.zeros_like(dense_out)
    denom = max(float(np.linalg.norm(sparse_out) * np.linalg.norm(dense_out)), 1e-20)
    cos = float(np.dot(sparse_out, dense_out) / denom)
    return MethodResult(
        method=method,
        target_mass=float(target),
        reached=bool(mass >= float(target)),
        mass=mass,
        output_cos=cos,
        exact_tokens=len(exact_tokens),
        represented_tokens=len(represented_tokens),
        estimated_mb=float(estimated_mb),
        score_reads=int(score_reads),
        score_elements=float(score_elements),
        index_bytes=int(index_bytes),
        final_kv_reads=int(final_kv_reads),
        value_sum_reads=0,
        edge_reads=0,
        offset_reads=0,
        nodes_visited=int(nodes_visited),
        clusters_scored=0,
        clusters_selected=0,
    )


def dense_oracle_results(args, scores, values, probs, dense_out, base, target_masses):
    """Perfect dynamic-token discovery lower bound for each mass target."""
    base = unique(base, len(base), 0, scores.shape[0])
    base_set = set(base)
    dynamic = [tok for tok in range(scores.shape[0]) if tok not in base_set]
    order = sorted(dynamic, key=lambda tok: float(probs[int(tok)]), reverse=True)
    represented = list(base)
    mass = float(probs[np.asarray(represented, dtype=np.int64)].sum()) if represented else 0.0
    results = []
    cursor = 0
    for target in sorted(float(x) for x in target_masses):
        while mass < target and cursor < len(order):
            tok = int(order[cursor])
            cursor += 1
            represented.append(tok)
            mass += float(probs[tok])
        results.append(
            evaluate_candidate(
                args,
                "dense_oracle",
                target,
                scores,
                values,
                probs,
                dense_out,
                represented,
                represented,
                score_reads=0,
            )
        )
    return results


def ranked_token_results(
    args,
    method: str,
    scores: np.ndarray,
    values: np.ndarray,
    probs: np.ndarray,
    dense_out: np.ndarray,
    base: list[int],
    ranked_tokens: Iterable[int],
    target_masses,
    *,
    score_reads: int,
    score_elements: float | None = None,
    index_bytes: int = 0,
    offset_reads: int = 0,
    exact_all_candidates: bool = False,
    nodes_visited: int = 0,
) -> list[MethodResult]:
    base = unique(base, len(base), 0, scores.shape[0])
    base_set = set(base)
    ranked = unique(ranked_tokens, scores.shape[0], 0, scores.shape[0])
    ranked = [tok for tok in ranked if tok not in base_set]
    results: list[MethodResult] = []

    if exact_all_candidates:
        represented = unique(list(base) + ranked, len(base) + len(ranked), 0, scores.shape[0])
        for target in sorted(float(x) for x in target_masses):
            results.append(
                evaluate_candidate(
                    args,
                    method,
                    target,
                    scores,
                    values,
                    probs,
                    dense_out,
                    represented,
                    represented,
                    score_reads=score_reads,
                    score_elements=score_elements,
                    index_bytes=index_bytes,
                    offset_reads=offset_reads,
                    nodes_visited=nodes_visited,
                )
            )
        return results

    represented = list(base)
    mass = float(probs[np.asarray(represented, dtype=np.int64)].sum()) if represented else 0.0
    cursor = 0
    for target in sorted(float(x) for x in target_masses):
        while mass < target and cursor < len(ranked):
            tok = int(ranked[cursor])
            cursor += 1
            represented.append(tok)
            mass += float(probs[tok])
        results.append(
            evaluate_candidate(
                args,
                method,
                target,
                scores,
                values,
                probs,
                dense_out,
                represented,
                represented,
                score_reads=score_reads,
                score_elements=score_elements,
                index_bytes=index_bytes,
                offset_reads=offset_reads,
                nodes_visited=nodes_visited,
            )
        )
    return results


def quest_results(args, scores, values, probs, dense_out, base, keys, target_masses, score_scale):
    results: list[MethodResult] = []
    base_set = set(base)
    limit = scores.shape[0]
    q = args.current_query.astype(np.float32)
    for page_size in parse_int_list(args.quest_page_sizes):
        page_size = max(1, int(page_size))
        page_scores: list[tuple[float, list[int]]] = []
        pages_scored = 0
        for start in range(0, limit, page_size):
            toks = [tok for tok in range(start, min(limit, start + page_size)) if tok not in base_set]
            if not toks:
                continue
            block = keys[np.asarray(toks, dtype=np.int64)].astype(np.float32, copy=False)
            page_max = block.max(axis=0)
            page_min = block.min(axis=0)
            bound_key = np.where(q >= 0, page_max, page_min)
            page_scores.append((float((bound_key @ q) * float(score_scale)), toks))
            pages_scored += 1
        page_scores.sort(reverse=True, key=lambda x: x[0])
        ranked_tokens: list[int] = []
        for _score, toks in page_scores:
            ranked_tokens.extend(toks)
        results.extend(
            ranked_token_results(
                args,
                f"quest_page_p{page_size}",
                scores,
                values,
                probs,
                dense_out,
                base,
                ranked_tokens,
                target_masses,
                score_reads=pages_scored * 2,
                score_elements=float(pages_scored * 2 * int(args.head_dim)),
                index_bytes=pages_scored * int(args.edge_index_bytes),
                nodes_visited=pages_scored,
            )
        )
    return results


def sparq_results(args, scores, values, probs, dense_out, base, keys, target_masses):
    results: list[MethodResult] = []
    base_set = set(base)
    dynamic = [tok for tok in range(scores.shape[0]) if tok not in base_set]
    if not dynamic:
        return results
    dyn_arr = np.asarray(dynamic, dtype=np.int64)
    q = args.current_query.astype(np.float32)
    q_abs_sum = max(float(np.abs(q).sum()), 1e-20)
    for rank in parse_int_list(args.sparq_ranks):
        rank = min(max(1, int(rank)), q.shape[0])
        dims = np.argsort(-np.abs(q), kind="stable")[:rank]
        coverage = max(float(np.abs(q[dims]).sum() / q_abs_sum), 1e-6)
        scale = 1.0 / math.sqrt(float(args.head_dim) * coverage)
        approx = (keys[dyn_arr[:, None], dims] @ q[dims]).astype(np.float32) * scale
        order = np.argsort(-approx, kind="stable")
        ranked = dyn_arr[order].tolist()
        results.extend(
            ranked_token_results(
                args,
                f"sparq_r{rank}",
                scores,
                values,
                probs,
                dense_out,
                base,
                ranked,
                target_masses,
                score_reads=len(dynamic),
                score_elements=float(len(dynamic) * rank),
                index_bytes=rank * int(args.edge_index_bytes),
            )
        )
    return results


def compute_pca_basis(keys: np.ndarray, dynamic_start: int, dynamic_end: int, max_rank: int) -> tuple[np.ndarray, np.ndarray]:
    block = keys[int(dynamic_start):int(dynamic_end)].astype(np.float32, copy=False)
    if block.shape[0] == 0:
        return np.empty((keys.shape[-1], 0), dtype=np.float32), np.zeros((keys.shape[-1],), dtype=np.float32)
    mean = block.mean(axis=0).astype(np.float32)
    centered = block - mean
    rank = min(int(max_rank), centered.shape[1], max(1, centered.shape[0]))
    _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
    basis = vh[:rank].T.astype(np.float32, copy=False)
    return basis, mean


def lloyd_kmeans(block: np.ndarray, clusters: int, *, seed: int, max_iter: int) -> tuple[np.ndarray, np.ndarray]:
    block = block.astype(np.float32, copy=False)
    if block.shape[0] == 0:
        return np.empty((0, block.shape[-1]), dtype=np.float32), np.empty((0,), dtype=np.int32)
    clusters = max(1, min(int(clusters), block.shape[0]))
    rng = np.random.default_rng(int(seed))
    init_idx = rng.choice(block.shape[0], size=clusters, replace=False)
    centers = block[init_idx].copy()
    assign = np.zeros((block.shape[0],), dtype=np.int32)
    iters = max(1, int(max_iter))
    block_norm = np.sum(block * block, axis=1, keepdims=True)
    for _ in range(iters):
        dist = block_norm + np.sum(centers * centers, axis=1, keepdims=True).T - 2.0 * (block @ centers.T)
        assign = np.argmin(dist, axis=1).astype(np.int32, copy=False)
        for cid in range(clusters):
            mask = assign == cid
            if np.any(mask):
                centers[cid] = block[mask].mean(axis=0)
    dist = block_norm + np.sum(centers * centers, axis=1, keepdims=True).T - 2.0 * (block @ centers.T)
    assign = np.argmin(dist, axis=1).astype(np.int32, copy=False)
    return centers.astype(np.float32, copy=False), assign


def loki_results(args, scores, values, probs, dense_out, base, keys, target_masses, basis: np.ndarray, mean: np.ndarray, dynamic_start: int, dynamic_end: int):
    results: list[MethodResult] = []
    base_set = set(base)
    dynamic = [tok for tok in range(scores.shape[0]) if tok not in base_set and int(dynamic_start) <= tok < int(dynamic_end)]
    if not dynamic or basis.shape[1] == 0:
        return results
    dyn_arr = np.asarray(dynamic, dtype=np.int64)
    q = args.current_query.astype(np.float32)
    centered_keys = keys[dyn_arr].astype(np.float32, copy=False) - mean.reshape(1, -1)
    centered_q = q - mean
    for rank in parse_int_list(args.loki_ranks):
        rank = min(max(1, int(rank)), basis.shape[1])
        b = basis[:, :rank]
        k_proj = centered_keys @ b
        q_proj = centered_q @ b
        approx = (k_proj @ q_proj).astype(np.float32) / math.sqrt(float(args.head_dim))
        order = np.argsort(-approx, kind="stable")
        ranked = dyn_arr[order].tolist()
        results.extend(
            ranked_token_results(
                args,
                f"loki_r{rank}",
                scores,
                values,
                probs,
                dense_out,
                base,
                ranked,
                target_masses,
                score_reads=len(dynamic),
                score_elements=float(len(dynamic) * rank),
                index_bytes=rank * int(args.head_dim) * int(args.score_key_bytes_per_element),
            )
        )
    return results


def build_pq_index(
    keys: np.ndarray,
    dynamic_start: int,
    dynamic_end: int,
    *,
    subvecs: int,
    subbits: int,
    seed: int,
    max_iter: int,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    block = keys[int(dynamic_start):int(dynamic_end)].astype(np.float32, copy=False)
    dim = int(keys.shape[-1])
    subvecs = max(1, min(int(subvecs), dim))
    if dim % subvecs != 0:
        raise ValueError(f"PQ subvecs must divide head_dim: dim={dim} subvecs={subvecs}")
    subdim = dim // subvecs
    centroids_per_subvec = 1 << int(subbits)
    if block.shape[0] == 0:
        return (
            np.empty((subvecs, centroids_per_subvec, subdim), dtype=np.float32),
            np.empty((0, subvecs), dtype=np.uint16),
            subvecs,
            centroids_per_subvec,
        )

    rng = np.random.default_rng(int(seed))
    codebooks = np.zeros((subvecs, centroids_per_subvec, subdim), dtype=np.float32)
    codes = np.zeros((block.shape[0], subvecs), dtype=np.uint16)
    for sub in range(subvecs):
        part = block[:, sub * subdim : (sub + 1) * subdim].astype(np.float32, copy=False)
        centers, assign = lloyd_kmeans(part, centroids_per_subvec, seed=int(rng.integers(0, 2**31 - 1)), max_iter=int(max_iter))
        if centers.shape[0] < centroids_per_subvec:
            k = centers.shape[0]
            pad_idx = rng.choice(part.shape[0], size=centroids_per_subvec - k, replace=True)
            centers = np.concatenate([centers, part[pad_idx].copy()], axis=0)
            dist = (
                np.sum(part * part, axis=1, keepdims=True)
                + np.sum(centers * centers, axis=1, keepdims=True).T
                - 2.0 * (part @ centers.T)
            )
            assign = np.argmin(dist, axis=1).astype(np.int32, copy=False)
        codes[:, sub] = assign.astype(np.uint16, copy=False)
        codebooks[sub] = centers.astype(np.float32, copy=False)
    return codebooks, codes, subvecs, centroids_per_subvec


def pqcache_results(
    args,
    scores,
    values,
    probs,
    dense_out,
    base,
    keys,
    target_masses,
    pq_index: tuple[np.ndarray, np.ndarray, int, int],
    dynamic_start: int,
    dynamic_end: int,
):
    codebooks, codes, subvecs, centroids_per_subvec = pq_index
    if codes.shape[0] == 0:
        return []
    base_set = set(base)
    q = args.current_query.astype(np.float32)
    subdim = q.shape[0] // int(subvecs)
    q_parts = q.reshape(int(subvecs), subdim)
    table = np.einsum("ms, mcs -> mc", q_parts, codebooks.astype(np.float32, copy=False), optimize=True)
    approx = np.zeros((codes.shape[0],), dtype=np.float32)
    for sub in range(int(subvecs)):
        approx += table[sub, codes[:, sub]]
    token_ids = np.arange(int(dynamic_start), int(dynamic_end), dtype=np.int64)
    visible = token_ids < scores.shape[0]
    if not np.any(visible):
        return []
    token_ids = token_ids[visible]
    approx = approx[visible]
    keep = np.asarray([int(tok) not in base_set for tok in token_ids], dtype=bool)
    token_ids = token_ids[keep]
    approx = approx[keep]
    order = np.argsort(-approx, kind="stable")
    ranked = token_ids[order].tolist()
    code_bytes = 1 if int(args.pqcache_subbits) <= 8 else 2
    return ranked_token_results(
        args,
        f"pqcache_m{subvecs}_b{int(args.pqcache_subbits)}",
        scores,
        values,
        probs,
        dense_out,
        base,
        ranked,
        target_masses,
        score_reads=len(token_ids),
        score_elements=float(int(subvecs) * int(centroids_per_subvec) * int(subdim)),
        index_bytes=int(codes.shape[0]) * int(subvecs) * int(code_bytes),
    )


def pqcache_quantized_k_results(
    args,
    scores,
    values,
    probs,
    dense_out,
    base,
    target_masses,
    pq_index: tuple[np.ndarray, np.ndarray, int, int],
    dynamic_start: int,
    dynamic_end: int,
    score_scale: float,
):
    codebooks, codes, subvecs, centroids_per_subvec = pq_index
    if codes.shape[0] == 0:
        return []
    base_u = unique(base, len(base), 0, scores.shape[0])
    base_set = set(base_u)
    token_ids = np.arange(int(dynamic_start), int(dynamic_end), dtype=np.int64)
    visible = token_ids < scores.shape[0]
    token_ids = token_ids[visible]
    if token_ids.size == 0:
        return []
    row_ids = token_ids - int(dynamic_start)
    keep = np.asarray([int(tok) not in base_set for tok in token_ids], dtype=bool)
    token_ids = token_ids[keep]
    row_ids = row_ids[keep]
    if token_ids.size == 0:
        return []
    q = args.current_query.astype(np.float32)
    approx_raw = pq_scores_for_rows(q, pq_index, row_ids)
    approx_logits = approx_raw.astype(np.float32) * float(score_scale)
    order = np.argsort(-approx_logits, kind="stable")
    ranked = token_ids[order].tolist()
    approx_by_token = {int(tok): float(logit) for tok, logit in zip(token_ids.tolist(), approx_logits.tolist())}
    code_bytes = 1 if int(args.pqcache_subbits) <= 8 else 2
    score_elements = float(int(subvecs) * int(centroids_per_subvec) * (int(args.head_dim) // int(subvecs)))
    index_bytes = int(codes.shape[0]) * int(subvecs) * int(code_bytes)
    score_mb = byte_cost(
        args,
        score_reads=len(token_ids),
        score_elements=score_elements,
        index_bytes=index_bytes,
        final_kv_reads=0,
        value_sum_reads=0,
        edge_reads=0,
        offset_reads=0,
    )
    base_kv_mb = len(base_u) * int(args.head_dim) * (int(args.attn_key_bytes_per_element) + int(args.value_bytes_per_element)) / (1024.0 * 1024.0)
    represented = list(base_u)
    mass = float(probs[np.asarray(represented, dtype=np.int64)].sum()) if represented else 0.0
    cursor = 0
    results = []
    for target in sorted(float(x) for x in target_masses):
        while mass < target and cursor < len(ranked):
            tok = int(ranked[cursor])
            cursor += 1
            represented.append(tok)
            mass += float(probs[tok])
        dyn_selected = [tok for tok in represented if tok not in base_set]
        dyn_v_mb = len(dyn_selected) * int(args.head_dim) * int(args.value_bytes_per_element) / (1024.0 * 1024.0)
        results.append(
            evaluate_candidate_with_logits(
                args,
                f"pqcache_quantized_k_m{subvecs}_b{int(args.pqcache_subbits)}",
                target,
                scores,
                values,
                probs,
                dense_out,
                base_u,
                represented,
                approx_by_token,
                estimated_mb=score_mb + base_kv_mb + dyn_v_mb,
                score_reads=len(token_ids),
                score_elements=score_elements,
                index_bytes=index_bytes,
                final_kv_reads=len(base_u) + len(dyn_selected),
            )
        )
    return results


def build_sign_vq_index(
    keys: np.ndarray,
    dynamic_start: int,
    dynamic_end: int,
    *,
    group_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    block = keys[int(dynamic_start):int(dynamic_end)].astype(np.float32, copy=False)
    dim = int(keys.shape[-1])
    group_size = max(1, int(group_size))
    if group_size > 8:
        raise ValueError(f"sign VQ group_size must be <= 8 for uint8 codes, got {group_size}")
    if dim % group_size != 0:
        raise ValueError(f"sign VQ group_size must divide head_dim: dim={dim} group_size={group_size}")
    groups = dim // group_size
    patterns = 1 << group_size
    if block.shape[0] == 0:
        return (
            np.zeros((dim,), dtype=np.float32),
            np.zeros((groups, patterns, group_size), dtype=np.float32),
            np.empty((0, groups), dtype=np.uint8),
            group_size,
        )
    mean = block.mean(axis=0).astype(np.float32)
    centered = block - mean.reshape(1, -1)
    grouped = centered.reshape(block.shape[0], groups, group_size)
    bits = grouped > 0
    weights = (1 << np.arange(group_size, dtype=np.uint16)).reshape(1, 1, group_size)
    codes = np.sum(bits.astype(np.uint16) * weights, axis=-1).astype(np.uint8, copy=False)
    centroids = np.zeros((groups, patterns, group_size), dtype=np.float32)
    for g in range(groups):
        for code in range(patterns):
            mask = codes[:, g] == code
            if np.any(mask):
                centroids[g, code] = grouped[mask, g].mean(axis=0)
            else:
                sign = np.asarray([1.0 if code & (1 << bit) else -1.0 for bit in range(group_size)], dtype=np.float32)
                scale = float(np.mean(np.abs(grouped[:, g]))) if grouped.shape[0] else 0.0
                centroids[g, code] = sign * scale
    return mean, centroids, codes, group_size


def sign_vq_lut_scores(q: np.ndarray, sign_index: tuple[np.ndarray, np.ndarray, np.ndarray, int], row_ids: np.ndarray) -> np.ndarray:
    _mean, centroids, codes, group_size = sign_index
    if row_ids.size == 0:
        return np.empty((0,), dtype=np.float32)
    groups = centroids.shape[0]
    q_groups = q.astype(np.float32, copy=False).reshape(groups, int(group_size))
    lut = np.einsum("gd,gcd->gc", q_groups, centroids, optimize=True)
    row_codes = codes[row_ids]
    out = np.zeros((row_ids.shape[0],), dtype=np.float32)
    for g in range(groups):
        out += lut[g, row_codes[:, g]]
    return out


def sign_vq_lut_pqcache_adaptive_results(
    args,
    scores,
    values,
    probs,
    dense_out,
    base,
    target_masses,
    pq_index: tuple[np.ndarray, np.ndarray, int, int],
    sign_index: tuple[np.ndarray, np.ndarray, np.ndarray, int],
    dynamic_start: int,
    dynamic_end: int,
):
    _mean, centroids, codes, group_size = sign_index
    if codes.shape[0] == 0:
        return []
    base_set = set(base)
    visible_hi = min(int(dynamic_end), scores.shape[0])
    token_ids = np.arange(int(dynamic_start), int(dynamic_end), dtype=np.int64)
    keep = (token_ids < visible_hi) & np.asarray([int(tok) not in base_set for tok in token_ids], dtype=bool)
    token_ids = token_ids[keep]
    if token_ids.size == 0:
        return []
    row_ids = token_ids - int(dynamic_start)
    q = args.current_query.astype(np.float32)
    selector_scores = sign_vq_lut_scores(q, sign_index, row_ids)
    selector_order = np.argsort(-selector_scores, kind="stable")
    codebooks, _pq_codes, subvecs, centroids_per_subvec = pq_index
    pq_code_bytes = 1 if int(args.pqcache_subbits) <= 8 else 2
    sign_scan_bytes = packed_bit_bytes(int(codes.shape[0]) * int(codes.shape[1]), int(group_size))
    sign_lut_elements = int(centroids.shape[0]) * int(centroids.shape[1]) * int(group_size)
    pq_lut_elements = int(subvecs) * int(centroids_per_subvec) * (int(args.head_dim) // int(subvecs))
    targets = sorted(float(x) for x in target_masses)
    results_by_target: dict[float, MethodResult] = {}
    best_by_target: dict[float, MethodResult] = {}

    for budget in sorted_unique_ints(args.self_indexing_candidate_budgets):
        budget = max(1, min(int(budget), token_ids.size))
        idx = selector_order[:budget]
        cand_tokens = token_ids[idx]
        cand_rows = row_ids[idx]
        maybe_record_candidate_frontier(
            args,
            method=f"sign_vq_lut_pqcache_oracle_g{int(group_size)}",
            budget_kind="candidate_budget",
            budget_value=budget,
            candidate_tokens=cand_tokens,
            base=base,
            probs=probs,
            score_reads=int(token_ids.size),
            score_elements=float(sign_lut_elements),
            index_bytes=int(sign_scan_bytes),
        )
        approx = pq_scores_for_rows(q, pq_index, cand_rows)
        order = np.argsort(-approx, kind="stable")
        ranked = cand_tokens[order].tolist()
        row_by_target = ranked_token_results(
            args,
            f"sign_vq_lut_pqcache_oracle_g{int(group_size)}",
            scores,
            values,
            probs,
            dense_out,
            base,
            ranked,
            targets,
            score_reads=int(token_ids.size) + int(cand_tokens.size),
            score_elements=float(sign_lut_elements + pq_lut_elements),
            index_bytes=int(sign_scan_bytes) + int(cand_tokens.size) * int(subvecs) * int(pq_code_bytes),
            nodes_visited=int(cand_tokens.size),
        )
        for row in row_by_target:
            target = float(row.target_mass)
            if target in results_by_target:
                continue
            prev = best_by_target.get(target)
            if prev is None or row.mass > prev.mass or (row.mass == prev.mass and row.estimated_mb < prev.estimated_mb):
                best_by_target[target] = row
            if row.reached:
                results_by_target[target] = row
        if len(results_by_target) == len(targets):
            break
    return [results_by_target.get(target, best_by_target[target]) for target in targets if target in results_by_target or target in best_by_target]


def build_weighted_hamming_index(
    keys: np.ndarray,
    dynamic_start: int,
    dynamic_end: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    block = keys[int(dynamic_start):int(dynamic_end)].astype(np.float32, copy=False)
    if block.shape[0] == 0:
        return np.zeros((keys.shape[-1],), dtype=np.float32), np.empty((0, 0), dtype=np.uint8), int(keys.shape[-1])
    mean = block.mean(axis=0).astype(np.float32)
    signs = (block - mean.reshape(1, -1)) > 0
    return mean, np.packbits(signs.astype(np.uint8), axis=1, bitorder="little"), int(keys.shape[-1])


def weighted_hamming_pqcache_adaptive_results(
    args,
    scores,
    values,
    probs,
    dense_out,
    base,
    target_masses,
    pq_index: tuple[np.ndarray, np.ndarray, int, int],
    sign_index: tuple[np.ndarray, np.ndarray, int],
    dynamic_start: int,
    dynamic_end: int,
):
    base_set = set(base)
    visible_hi = min(int(dynamic_end), scores.shape[0])
    token_ids = np.arange(int(dynamic_start), int(dynamic_end), dtype=np.int64)
    keep = (token_ids < visible_hi) & np.asarray([int(tok) not in base_set for tok in token_ids], dtype=bool)
    token_ids = token_ids[keep]
    if token_ids.size == 0:
        return []
    row_ids = token_ids - int(dynamic_start)
    _mean, key_sign_packed, sign_dim = sign_index
    q = args.current_query.astype(np.float32)
    key_sign = np.unpackbits(key_sign_packed[row_ids], axis=1, count=int(sign_dim), bitorder="little").astype(bool, copy=False)
    query_sign = q > 0
    weights = np.abs(q).astype(np.float32)
    selector_scores = (key_sign == query_sign.reshape(1, -1)).astype(np.float32) @ weights
    selector_order = np.argsort(-selector_scores, kind="stable")
    _codebooks, _pq_codes, subvecs, centroids_per_subvec = pq_index
    pq_code_bytes = 1 if int(args.pqcache_subbits) <= 8 else 2
    sign_scan_bytes = packed_bit_bytes(token_ids.size, int(args.head_dim))
    pq_lut_elements = int(subvecs) * int(centroids_per_subvec) * (int(args.head_dim) // int(subvecs))
    targets = sorted(float(x) for x in target_masses)
    results_by_target: dict[float, MethodResult] = {}
    best_by_target: dict[float, MethodResult] = {}
    for budget in sorted_unique_ints(args.self_indexing_candidate_budgets):
        budget = max(1, min(int(budget), token_ids.size))
        idx = selector_order[:budget]
        cand_tokens = token_ids[idx]
        cand_rows = row_ids[idx]
        maybe_record_candidate_frontier(
            args,
            method="weighted_hamming_pqcache_oracle",
            budget_kind="candidate_budget",
            budget_value=budget,
            candidate_tokens=cand_tokens,
            base=base,
            probs=probs,
            score_reads=int(token_ids.size),
            score_elements=0.0,
            index_bytes=int(sign_scan_bytes),
        )
        approx = pq_scores_for_rows(q, pq_index, cand_rows)
        order = np.argsort(-approx, kind="stable")
        ranked = cand_tokens[order].tolist()
        row_by_target = ranked_token_results(
            args,
            "weighted_hamming_pqcache_oracle",
            scores,
            values,
            probs,
            dense_out,
            base,
            ranked,
            targets,
            score_reads=int(token_ids.size) + int(cand_tokens.size),
            score_elements=float(pq_lut_elements),
            index_bytes=int(sign_scan_bytes) + int(cand_tokens.size) * int(subvecs) * int(pq_code_bytes),
            nodes_visited=int(cand_tokens.size),
        )
        for row in row_by_target:
            target = float(row.target_mass)
            if target in results_by_target:
                continue
            prev = best_by_target.get(target)
            if prev is None or row.mass > prev.mass or (row.mass == prev.mass and row.estimated_mb < prev.estimated_mb):
                best_by_target[target] = row
            if row.reached:
                results_by_target[target] = row
        if len(results_by_target) == len(targets):
            break
    return [results_by_target.get(target, best_by_target[target]) for target in targets if target in results_by_target or target in best_by_target]


def pq_scores_for_rows(q: np.ndarray, pq_index: tuple[np.ndarray, np.ndarray, int, int], row_ids: np.ndarray) -> np.ndarray:
    codebooks, codes, subvecs, _centroids_per_subvec = pq_index
    if row_ids.size == 0:
        return np.empty((0,), dtype=np.float32)
    subdim = q.shape[0] // int(subvecs)
    q_parts = q.astype(np.float32, copy=False).reshape(int(subvecs), subdim)
    table = np.einsum("ms, mcs -> mc", q_parts, codebooks.astype(np.float32, copy=False), optimize=True)
    row_codes = codes[row_ids]
    approx = np.zeros((row_ids.shape[0],), dtype=np.float32)
    for sub in range(int(subvecs)):
        approx += table[sub, row_codes[:, sub]]
    return approx


def build_ivfpq_index(
    keys: np.ndarray,
    dynamic_start: int,
    dynamic_end: int,
    *,
    coarse_clusters: int,
    coarse_iters: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    block = keys[int(dynamic_start):int(dynamic_end)].astype(np.float32, copy=False)
    if block.shape[0] == 0:
        return np.empty((0, keys.shape[-1]), dtype=np.float32), np.empty((0,), dtype=np.int32), []
    centroids, assign = lloyd_kmeans(block, int(coarse_clusters), seed=int(seed), max_iter=int(coarse_iters))
    buckets = [np.where(assign == cid)[0].astype(np.int64, copy=False) for cid in range(centroids.shape[0])]
    return centroids, assign, buckets


def ivfpq_adaptive_results(
    args,
    scores,
    values,
    probs,
    dense_out,
    base,
    target_masses,
    ivfpq_index: tuple[np.ndarray, np.ndarray, list[np.ndarray], tuple[np.ndarray, np.ndarray, int, int]],
    dynamic_start: int,
    dynamic_end: int,
    score_scale: float,
):
    centroids, _assign, buckets, pq_index = ivfpq_index
    if centroids.shape[0] == 0:
        return []
    codebooks, codes, subvecs, centroids_per_subvec = pq_index
    q = args.current_query.astype(np.float32)
    base_set = set(base)
    visible_hi = min(int(dynamic_end), scores.shape[0])
    targets = sorted(float(x) for x in target_masses)
    results_by_target: dict[float, MethodResult] = {}
    best_by_target: dict[float, MethodResult] = {}
    coarse_scores = (centroids @ q) * float(score_scale)
    coarse_order = np.argsort(-coarse_scores, kind="stable")
    method_name = ivfpq_method_name(args, centroids)
    code_bytes = pq_code_bytes(args)
    update_score_elements, update_index_bytes = ivfpq_online_update_cost(
        args,
        dynamic_count=max(0, int(dynamic_end) - int(dynamic_start)),
        centroids=centroids,
        subvecs=int(subvecs),
        centroids_per_subvec=int(centroids_per_subvec),
    )
    fixed_nprobes = set(sorted_unique_ints(args.ivfpq_fixed_nprobes)) if str(getattr(args, "ivfpq_fixed_nprobes", "")).strip() else set()
    loop_nprobes = sorted(set(sorted_unique_ints(args.ivfpq_nprobes)) | fixed_nprobes)
    fixed_results: list[MethodResult] = []

    for nprobe in loop_nprobes:
        nprobe = max(1, min(int(nprobe), centroids.shape[0]))
        selected_bucket_ids = coarse_order[:nprobe]
        if selected_bucket_ids.size == 0:
            continue
        row_ids = np.concatenate([buckets[int(cid)] for cid in selected_bucket_ids if buckets[int(cid)].size > 0])
        if row_ids.size == 0:
            continue
        token_ids = row_ids + int(dynamic_start)
        keep = (token_ids < visible_hi) & np.asarray([int(tok) not in base_set for tok in token_ids], dtype=bool)
        row_ids = row_ids[keep]
        token_ids = token_ids[keep]
        if token_ids.size == 0:
            continue
        maybe_record_candidate_frontier(
            args,
            method=method_name,
            budget_kind="nprobe",
            budget_value=nprobe,
            candidate_tokens=token_ids,
            base=base,
            probs=probs,
            score_reads=int(centroids.shape[0]),
            score_elements=float(int(centroids.shape[0]) * int(args.head_dim)) + float(update_score_elements),
            index_bytes=int(token_ids.size) * int(args.edge_index_bytes) + int(update_index_bytes),
            offset_reads=int(nprobe),
        )
        approx = pq_scores_for_rows(q, pq_index, row_ids)
        order = np.argsort(-approx, kind="stable")
        ranked = token_ids[order].tolist()
        row_by_target = ranked_token_results(
            args,
            method_name,
            scores,
            values,
            probs,
            dense_out,
            base,
            ranked,
            targets,
            score_reads=int(centroids.shape[0]) + int(token_ids.size),
            score_elements=float(
                int(centroids.shape[0]) * int(args.head_dim)
                + int(subvecs) * int(centroids_per_subvec) * (int(args.head_dim) // int(subvecs))
            ) + float(update_score_elements),
            index_bytes=int(token_ids.size) * (int(subvecs) * int(code_bytes) + int(args.edge_index_bytes)) + int(update_index_bytes),
            offset_reads=int(nprobe),
            nodes_visited=int(token_ids.size),
        )
        for row in row_by_target:
            target = float(row.target_mass)
            if target in results_by_target:
                continue
            prev = best_by_target.get(target)
            if prev is None or row.mass > prev.mass or (row.mass == prev.mass and row.estimated_mb < prev.estimated_mb):
                best_by_target[target] = row
            if row.reached:
                results_by_target[target] = row
        if nprobe in fixed_nprobes:
            represented = unique(list(base) + token_ids.tolist(), len(base) + int(token_ids.size), 0, scores.shape[0])
            exact_score_elements = float(
                int(centroids.shape[0]) * int(args.head_dim)
                + int(subvecs) * int(centroids_per_subvec) * (int(args.head_dim) // int(subvecs))
            ) + float(update_score_elements)
            exact_index_bytes = int(token_ids.size) * (
                int(subvecs) * int(code_bytes) + int(args.edge_index_bytes)
            ) + int(update_index_bytes)
            for target in targets:
                fixed_results.append(
                    evaluate_candidate(
                        args,
                        ivfpq_fixed_method_name(args, centroids, nprobe, pq_logits=False),
                        target,
                        scores,
                        values,
                        probs,
                        dense_out,
                        represented,
                        represented,
                        score_reads=int(centroids.shape[0]) + int(token_ids.size),
                        score_elements=exact_score_elements,
                        index_bytes=exact_index_bytes,
                        offset_reads=int(nprobe),
                        nodes_visited=int(token_ids.size),
                    )
                )
            if bool(getattr(args, "ivfpq_emit_pq_logits", False)):
                approx_logits = (approx.astype(np.float32) * float(score_scale)).tolist()
                approx_by_token = {int(tok): float(logit) for tok, logit in zip(token_ids.tolist(), approx_logits)}
                score_mb = byte_cost(
                    args,
                    score_reads=int(centroids.shape[0]) + int(token_ids.size),
                    score_elements=exact_score_elements,
                    index_bytes=exact_index_bytes,
                    final_kv_reads=0,
                    value_sum_reads=0,
                    edge_reads=0,
                    offset_reads=int(nprobe),
                )
                base_u = unique(base, len(base), 0, scores.shape[0])
                base_set = set(base_u)
                dyn_selected = [tok for tok in represented if tok not in base_set]
                base_kv_mb = len(base_u) * int(args.head_dim) * (
                    int(args.attn_key_bytes_per_element) + int(args.value_bytes_per_element)
                ) / (1024.0 * 1024.0)
                dyn_v_mb = len(dyn_selected) * int(args.head_dim) * int(args.value_bytes_per_element) / (1024.0 * 1024.0)
                for target in targets:
                    fixed_results.append(
                        evaluate_candidate_with_logits(
                            args,
                            ivfpq_fixed_method_name(args, centroids, nprobe, pq_logits=True),
                            target,
                            scores,
                            values,
                            probs,
                            dense_out,
                            base_u,
                            represented,
                            approx_by_token,
                            estimated_mb=score_mb + base_kv_mb + dyn_v_mb,
                            score_reads=int(centroids.shape[0]) + int(token_ids.size),
                            score_elements=exact_score_elements,
                            index_bytes=exact_index_bytes,
                            final_kv_reads=len(base_u) + len(dyn_selected),
                            nodes_visited=int(token_ids.size),
                        )
                    )
        if len(results_by_target) == len(targets) and all(int(x) <= int(nprobe) for x in fixed_nprobes):
            break

    adaptive_results = [
        results_by_target.get(target, best_by_target[target])
        for target in targets
        if target in results_by_target or target in best_by_target
    ]
    return adaptive_results + fixed_results


def build_binary_gated_index(
    keys: np.ndarray,
    dynamic_start: int,
    dynamic_end: int,
    *,
    bits: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    block = keys[int(dynamic_start):int(dynamic_end)].astype(np.float32, copy=False)
    bits = max(1, int(bits))
    rng = np.random.default_rng(int(seed))
    projection = rng.standard_normal((int(keys.shape[-1]), bits)).astype(np.float32)
    if block.shape[0] == 0:
        return (
            projection,
            np.zeros((keys.shape[-1],), dtype=np.float32),
            np.empty((0, 0), dtype=np.uint8),
            bits,
        )
    mean = block.mean(axis=0).astype(np.float32)
    key_bits = (block - mean.reshape(1, -1)) @ projection > 0
    return projection, mean, np.packbits(key_bits.astype(np.uint8), axis=1, bitorder="little"), bits


def binary_gated_pqcache_adaptive_results(
    args,
    scores,
    values,
    probs,
    dense_out,
    base,
    target_masses,
    pq_index: tuple[np.ndarray, np.ndarray, int, int],
    binary_index: tuple[np.ndarray, np.ndarray, np.ndarray, int],
    dynamic_start: int,
    dynamic_end: int,
):
    codebooks, codes, subvecs, centroids_per_subvec = pq_index
    if codes.shape[0] == 0:
        return []
    base_set = set(base)
    visible_hi = min(int(dynamic_end), scores.shape[0])
    token_ids = np.arange(int(dynamic_start), int(dynamic_end), dtype=np.int64)
    keep = (token_ids < visible_hi) & np.asarray([int(tok) not in base_set for tok in token_ids], dtype=bool)
    token_ids = token_ids[keep]
    if token_ids.size == 0:
        return []
    row_ids = token_ids - int(dynamic_start)
    bits = max(1, int(args.binary_gated_bits))
    projection, _mean, key_bits_packed, packed_bits = binary_index
    q = args.current_query.astype(np.float32)
    q_norm = q / max(float(np.linalg.norm(q)), 1e-20)
    query_bits = (q_norm.reshape(1, -1) @ projection > 0).reshape(-1)
    query_packed = np.packbits(query_bits.astype(np.uint8), bitorder="little")
    xor = np.bitwise_xor(key_bits_packed[row_ids], query_packed.reshape(1, -1))
    mismatches = POPCOUNT_U8[xor].sum(axis=1)
    matches = int(packed_bits) - mismatches
    targets = sorted(float(x) for x in target_masses)
    results_by_target: dict[float, MethodResult] = {}
    best_by_target: dict[float, MethodResult] = {}
    code_bytes = 1 if int(args.pqcache_subbits) <= 8 else 2
    binary_scan_bytes = packed_bit_bytes(token_ids.size, bits)

    for budget in sorted_unique_ints(args.binary_gated_candidate_budgets):
        budget = max(1, min(int(budget), token_ids.size))
        cand_idx = np.argsort(-matches, kind="stable")[:budget]
        cand_rows = row_ids[cand_idx]
        cand_tokens = token_ids[cand_idx]
        maybe_record_candidate_frontier(
            args,
            method=f"binary_gated_pqcache_oracle_b{bits}",
            budget_kind="candidate_budget",
            budget_value=budget,
            candidate_tokens=cand_tokens,
            base=base,
            probs=probs,
            score_reads=int(token_ids.size),
            score_elements=0.0,
            index_bytes=int(binary_scan_bytes),
        )
        approx = pq_scores_for_rows(q, pq_index, cand_rows)
        order = np.argsort(-approx, kind="stable")
        ranked = cand_tokens[order].tolist()
        row_by_target = ranked_token_results(
            args,
            f"binary_gated_pqcache_oracle_b{bits}",
            scores,
            values,
            probs,
            dense_out,
            base,
            ranked,
            targets,
            score_reads=int(token_ids.size) + int(cand_tokens.size),
            score_elements=float(int(subvecs) * int(centroids_per_subvec) * (int(args.head_dim) // int(subvecs))),
            index_bytes=int(binary_scan_bytes) + int(cand_tokens.size) * int(subvecs) * int(code_bytes),
            nodes_visited=int(cand_tokens.size),
        )
        for row in row_by_target:
            target = float(row.target_mass)
            if target in results_by_target:
                continue
            prev = best_by_target.get(target)
            if prev is None or row.mass > prev.mass or (row.mass == prev.mass and row.estimated_mb < prev.estimated_mb):
                best_by_target[target] = row
            if row.reached:
                results_by_target[target] = row
        if len(results_by_target) == len(targets):
            break

    return [results_by_target.get(target, best_by_target[target]) for target in targets if target in results_by_target or target in best_by_target]


def simhash_codes(vectors: np.ndarray, projection: np.ndarray, bits: int, tables: int) -> np.ndarray:
    raw = (vectors.astype(np.float32, copy=False) @ projection.astype(np.float32, copy=False)) > 0
    raw = raw.reshape(vectors.shape[0], int(tables), int(bits))
    weights = (1 << np.arange(int(bits), dtype=np.int64)).reshape(1, 1, -1)
    return np.sum(raw.astype(np.int64) * weights, axis=-1).astype(np.int32, copy=False)


def magicpig_results(args, scores, values, probs, dense_out, base, keys, target_masses, projection_cache: dict[tuple, np.ndarray], cache_key: tuple):
    results: list[MethodResult] = []
    base_set = set(base)
    dynamic = [tok for tok in range(scores.shape[0]) if tok not in base_set]
    if not dynamic:
        return results
    dyn_arr = np.asarray(dynamic, dtype=np.int64)
    centered = keys[dyn_arr].astype(np.float32, copy=False)
    centered = centered - centered.mean(axis=0, keepdims=True)
    q = args.current_query.astype(np.float32)
    q_norm = q / max(float(np.linalg.norm(q)), 1e-20)
    for k_bits, tables in parse_magicpig_configs(args.magicpig_configs):
        key = (*cache_key, "magicpig", int(k_bits), int(tables))
        if key not in projection_cache:
            rng = np.random.default_rng(zlib.crc32(repr(key).encode("utf-8")))
            projection_cache[key] = rng.standard_normal((int(args.head_dim), int(k_bits) * int(tables))).astype(np.float32)
        proj = projection_cache[key]
        codes = simhash_codes(centered, proj, int(k_bits), int(tables))
        q_codes = simhash_codes(q_norm.reshape(1, -1), proj, int(k_bits), int(tables))[0]
        collisions = (codes == q_codes.reshape(1, -1)).sum(axis=1)
        mask = collisions >= int(args.magicpig_min_collisions)
        candidates = dyn_arr[mask]
        if candidates.size:
            order = np.argsort(-scores[candidates], kind="stable")
            ranked = candidates[order].tolist()
        else:
            ranked = []
        table_hits = int(collisions[mask].sum()) if candidates.size else 0
        results.extend(
            ranked_token_results(
                args,
                f"magicpig_k{k_bits}_l{tables}",
                scores,
                values,
                probs,
                dense_out,
                base,
                ranked,
                target_masses,
                score_reads=0,
                score_elements=0.0,
                index_bytes=int(tables) * 4 + table_hits * int(args.edge_index_bytes),
                exact_all_candidates=True,
                nodes_visited=int(candidates.size),
            )
        )
    return results


def magicpig_adaptive_results(args, scores, values, probs, dense_out, base, keys, target_masses, projection_cache: dict[tuple, np.ndarray], cache_key: tuple):
    results_by_target: dict[float, MethodResult] = {}
    best_by_target: dict[float, MethodResult] = {}
    base_set = set(base)
    dynamic = [tok for tok in range(scores.shape[0]) if tok not in base_set]
    if not dynamic:
        return []
    dyn_arr = np.asarray(dynamic, dtype=np.int64)
    centered = keys[dyn_arr].astype(np.float32, copy=False)
    centered = centered - centered.mean(axis=0, keepdims=True)
    q = args.current_query.astype(np.float32)
    q_norm = q / max(float(np.linalg.norm(q)), 1e-20)
    targets = sorted(float(x) for x in target_masses)

    for k_bits, tables, threshold in parse_magicpig_ladder(args.magicpig_adaptive_ladder):
        key = (*cache_key, "magicpig_adaptive", int(k_bits), int(tables))
        if key not in projection_cache:
            rng = np.random.default_rng(zlib.crc32(repr(key).encode("utf-8")))
            projection_cache[key] = rng.standard_normal((int(args.head_dim), int(k_bits) * int(tables))).astype(np.float32)
        proj = projection_cache[key]
        codes = simhash_codes(centered, proj, int(k_bits), int(tables))
        q_codes = simhash_codes(q_norm.reshape(1, -1), proj, int(k_bits), int(tables))[0]
        collisions = (codes == q_codes.reshape(1, -1)).sum(axis=1)
        mask = collisions >= int(threshold)
        candidates = dyn_arr[mask]
        candidate_collisions = collisions[mask]
        table_hits = int(collisions[mask].sum()) if candidates.size else 0
        if candidates.size:
            ranked_order = np.lexsort((candidates, -candidate_collisions))
            ranked_candidates = candidates[ranked_order].tolist()
        else:
            ranked_candidates = []
        base_u = unique(base, len(base), 0, scores.shape[0])
        base_mass = float(probs[np.asarray(base_u, dtype=np.int64)].sum()) if base_u else 0.0
        for target in targets:
            if target in results_by_target:
                continue
            represented = list(base_u)
            mass = base_mass
            for tok in ranked_candidates:
                if mass >= target:
                    break
                represented.append(int(tok))
                mass += float(probs[int(tok)])
            row = evaluate_candidate(
                args,
                "magicpig_adaptive",
                target,
                scores,
                values,
                probs,
                dense_out,
                represented,
                represented,
                score_reads=0,
                score_elements=0.0,
                index_bytes=int(tables) * 4 + table_hits * int(args.edge_index_bytes),
                nodes_visited=int(candidates.size),
            )
            prev = best_by_target.get(target)
            if prev is None or row.mass > prev.mass or (row.mass == prev.mass and row.estimated_mb < prev.estimated_mb):
                best_by_target[target] = row
            if row.reached:
                results_by_target[target] = row

    return [results_by_target.get(target, best_by_target[target]) for target in targets if target in results_by_target or target in best_by_target]


def pariskv_results(args, scores, values, probs, dense_out, base, keys, target_masses, projection_cache: dict[tuple, np.ndarray], cache_key: tuple):
    results: list[MethodResult] = []
    base_set = set(base)
    dynamic = [tok for tok in range(scores.shape[0]) if tok not in base_set]
    if not dynamic:
        return results
    dyn_arr = np.asarray(dynamic, dtype=np.int64)
    q = args.current_query.astype(np.float32)
    for bits, tables, ratio in parse_pariskv_configs(args.pariskv_configs):
        key = (*cache_key, "pariskv", int(bits), int(tables))
        if key not in projection_cache:
            rng = np.random.default_rng(zlib.crc32(repr(key).encode("utf-8")))
            projection_cache[key] = rng.standard_normal((int(args.head_dim), int(bits) * int(tables))).astype(np.float32)
        proj = projection_cache[key]
        codes = simhash_codes(keys[dyn_arr], proj, int(bits), int(tables))
        q_codes = simhash_codes(q.reshape(1, -1), proj, int(bits), int(tables))[0]
        collisions = (codes == q_codes.reshape(1, -1)).sum(axis=1)
        candidate_count = max(1, min(len(dynamic), int(math.ceil(float(ratio) * len(dynamic)))))
        candidate_idx = np.argsort(-collisions, kind="stable")[:candidate_count]
        candidates = dyn_arr[candidate_idx]
        dims = np.argsort(-np.abs(q), kind="stable")[: min(int(args.pariskv_rerank_dims), q.shape[0])]
        approx = keys[candidates[:, None], dims] @ q[dims]
        order = np.argsort(-approx, kind="stable")
        ranked = candidates[order].tolist()
        results.extend(
            ranked_token_results(
                args,
                f"pariskv_b{bits}_t{tables}_c{ratio:g}",
                scores,
                values,
                probs,
                dense_out,
                base,
                ranked,
                target_masses,
                score_reads=len(dynamic),
                score_elements=float(len(dynamic) * int(tables) + len(candidates) * len(dims)),
                index_bytes=int(tables) * int(args.edge_index_bytes) + len(candidates) * int(args.edge_index_bytes),
                nodes_visited=len(candidates),
            )
        )
    return results


def pariskv_adaptive_results(args, scores, values, probs, dense_out, base, keys, target_masses, projection_cache: dict[tuple, np.ndarray], cache_key: tuple):
    results_by_target: dict[float, MethodResult] = {}
    best_by_target: dict[float, MethodResult] = {}
    base_set = set(base)
    dynamic = [tok for tok in range(scores.shape[0]) if tok not in base_set]
    if not dynamic:
        return []
    dyn_arr = np.asarray(dynamic, dtype=np.int64)
    q = args.current_query.astype(np.float32)
    targets = sorted(float(x) for x in target_masses)

    for bits, tables, ratio, rerank_dims in parse_pariskv_ladder(args.pariskv_adaptive_ladder):
        key = (*cache_key, "pariskv_adaptive", int(bits), int(tables))
        if key not in projection_cache:
            rng = np.random.default_rng(zlib.crc32(repr(key).encode("utf-8")))
            projection_cache[key] = rng.standard_normal((int(args.head_dim), int(bits) * int(tables))).astype(np.float32)
        proj = projection_cache[key]
        codes = simhash_codes(keys[dyn_arr], proj, int(bits), int(tables))
        q_codes = simhash_codes(q.reshape(1, -1), proj, int(bits), int(tables))[0]
        collisions = (codes == q_codes.reshape(1, -1)).sum(axis=1)
        candidate_count = max(1, min(len(dynamic), int(math.ceil(float(ratio) * len(dynamic)))))
        candidate_idx = np.argsort(-collisions, kind="stable")[:candidate_count]
        candidates = dyn_arr[candidate_idx]
        dims = np.argsort(-np.abs(q), kind="stable")[: min(int(rerank_dims), q.shape[0])]
        approx = keys[candidates[:, None], dims] @ q[dims]
        order = np.argsort(-approx, kind="stable")
        ranked = candidates[order].tolist()
        row_by_target = ranked_token_results(
            args,
            "pariskv_adaptive",
            scores,
            values,
            probs,
            dense_out,
            base,
            ranked,
            targets,
            score_reads=len(dynamic),
            score_elements=float(len(dynamic) * int(tables) + len(candidates) * len(dims)),
            index_bytes=int(tables) * int(args.edge_index_bytes) + len(candidates) * int(args.edge_index_bytes),
            nodes_visited=len(candidates),
        )
        for row in row_by_target:
            target = float(row.target_mass)
            if target in results_by_target:
                continue
            prev = best_by_target.get(target)
            if prev is None or row.mass > prev.mass or (row.mass == prev.mass and row.estimated_mb < prev.estimated_mb):
                best_by_target[target] = row
            if row.reached:
                results_by_target[target] = row

    return [results_by_target.get(target, best_by_target[target]) for target in targets if target in results_by_target or target in best_by_target]


def retro_results(args, scores, values, probs, dense_out, base, centroids, ranges, target_masses, score_scale):
    if centroids.shape[0] == 0:
        return []
    cluster_scores = (centroids @ args.current_query.astype(np.float32)) * float(score_scale)
    order = np.argsort(-cluster_scores, kind="stable")
    base_set = set(base)
    results = []
    represented = list(base)
    exact = list(base)
    approx_logits: list[float] = []
    approx_value_sums: list[np.ndarray] = []
    approx_sizes: list[int] = []
    pending = set(float(x) for x in target_masses)
    exact_cluster_cap = max(0, int(args.retro_exact_clusters))
    selected_clusters = 0
    for cid in order.tolist():
        start, end = ranges[int(cid)]
        toks = [tok for tok in range(start, min(end, scores.shape[0])) if tok not in base_set]
        if not toks:
            continue
        selected_clusters += 1
        represented.extend(toks)
        if exact_cluster_cap <= 0 or selected_clusters <= exact_cluster_cap:
            exact.extend(toks)
        else:
            approx_logits.append(float(cluster_scores[int(cid)]))
            approx_value_sums.append(values[np.asarray(toks, dtype=np.int64)].astype(np.float32).sum(axis=0))
            approx_sizes.append(len(toks))
        mass = float(probs[np.asarray(unique(represented, len(represented), 0, scores.shape[0]), dtype=np.int64)].sum())
        for target in sorted(list(pending)):
            if mass >= target:
                results.append(
                    evaluate_candidate(
                        args,
                        "retroinfer",
                        target,
                        scores,
                        values,
                        probs,
                        dense_out,
                        exact,
                        represented,
                        score_reads=int(centroids.shape[0]),
                        value_sum_reads=len(approx_logits),
                        clusters_scored=int(centroids.shape[0]),
                        clusters_selected=selected_clusters,
                        approx_logits=approx_logits,
                        approx_value_sums=approx_value_sums,
                        approx_sizes=approx_sizes,
                    )
                )
                pending.remove(target)
        if not pending:
            break
    for target in sorted(pending):
        results.append(
            evaluate_candidate(
                args,
                "retroinfer",
                target,
                scores,
                values,
                probs,
                dense_out,
                exact,
                represented,
                score_reads=int(centroids.shape[0]),
                value_sum_reads=len(approx_logits),
                clusters_scored=int(centroids.shape[0]),
                clusters_selected=selected_clusters,
                approx_logits=approx_logits,
                approx_value_sums=approx_value_sums,
                approx_sizes=approx_sizes,
            )
        )
    return results


def oracle_seed_tokens(args, scores: np.ndarray, base: list[int], *, count: int) -> tuple[list[int], int]:
    base_set = set(base)
    dynamic = [tok for tok in range(scores.shape[0]) if tok not in base_set]
    dynamic.sort(key=lambda tok: float(scores[int(tok)]), reverse=True)
    return dynamic[: max(0, int(count))], 0


def retro_seed_tokens(
    args,
    scores: np.ndarray,
    base: list[int],
    centroids: np.ndarray,
    ranges: list[tuple[int, int]],
    score_scale: float,
    *,
    count: int,
) -> tuple[list[int], int]:
    if centroids.shape[0] == 0:
        return [], 0
    q = args.current_query.astype(np.float32)
    cluster_scores = (centroids @ q) * float(score_scale)
    order = np.argsort(-cluster_scores, kind="stable")
    base_set = set(base)
    seeds: list[int] = []
    scanned_tokens = 0
    for cid in order.tolist():
        start, end = ranges[int(cid)]
        toks = [tok for tok in range(start, min(end, scores.shape[0])) if tok not in base_set]
        if not toks:
            continue
        scanned_tokens += len(toks)
        toks.sort(key=lambda tok: float(scores[int(tok)]), reverse=True)
        for tok in toks:
            if tok not in seeds:
                seeds.append(int(tok))
                break
        if len(seeds) >= int(count):
            break
    # Count centroid scores plus token reads used to pick one strong entry per cluster.
    return seeds[: max(0, int(count))], int(centroids.shape[0]) + int(scanned_tokens)


def graph_token_results(
    args,
    method,
    graph: CsrGraph,
    scores,
    values,
    probs,
    dense_out,
    base,
    target_masses,
    *,
    tail_end: int,
    seed_override: list[int] | None = None,
    seed_extra_score_reads: int = 0,
):
    import heapq

    base_set = set(base)
    seeds = seed_override if seed_override is not None else graph.seeds(int(args.seed_count), tail_end=tail_end)
    frontier = []
    scored = set()
    for tok in seeds:
        if 0 <= int(tok) < scores.shape[0]:
            scored.add(int(tok))
            heapq.heappush(frontier, (-float(scores[int(tok)]), int(tok)))
    visited = set()
    candidates: list[tuple[float, int]] = []
    represented = list(base)
    exact = list(base)
    edge_reads = 0
    offset_reads = 0
    pending = set(float(x) for x in target_masses)
    results = []
    max_visits = max(1, int(args.max_visits))
    while frontier and len(visited) < max_visits and pending:
        neg, node = heapq.heappop(frontier)
        if node in visited:
            continue
        visited.add(node)
        offset_reads += 1
        if node not in base_set:
            candidates.append((-float(neg), int(node)))
        for nb in graph.neighbors(node):
            edge_reads += 1
            if nb in visited or nb in base_set or nb >= scores.shape[0]:
                continue
            if nb not in scored:
                scored.add(nb)
                heapq.heappush(frontier, (-float(scores[nb]), nb))
        ranked = [tok for _s, tok in sorted(candidates, reverse=True, key=lambda x: x[0])]
        represented = unique(list(base) + ranked, len(base) + len(ranked), 0, scores.shape[0])
        exact = represented
        mass = float(probs[np.asarray(represented, dtype=np.int64)].sum()) if represented else 0.0
        for target in sorted(list(pending)):
            if mass >= target:
                results.append(
                    evaluate_candidate(
                        args,
                        method,
                        target,
                        scores,
                        values,
                        probs,
                        dense_out,
                        exact,
                        represented,
                        score_reads=len(scored) + int(seed_extra_score_reads),
                        edge_reads=edge_reads,
                        offset_reads=offset_reads,
                        nodes_visited=len(visited),
                    )
                )
                pending.remove(target)
    for target in sorted(pending):
        results.append(
            evaluate_candidate(
                args,
                method,
                target,
                scores,
                values,
                probs,
                dense_out,
                exact,
                represented,
                score_reads=len(scored) + int(seed_extra_score_reads),
                edge_reads=edge_reads,
                offset_reads=offset_reads,
                nodes_visited=len(visited),
            )
        )
    return results


def hybrid_results(args, graph: CsrGraph, scores, values, probs, dense_out, base, centroids, ranges, target_masses, score_scale):
    import heapq

    q = args.current_query.astype(np.float32)
    centroid_scores = (centroids @ q) * float(score_scale)
    seeds = graph.seeds(int(args.seed_count))
    frontier = []
    scored = set()
    for cid in seeds:
        if 0 <= int(cid) < centroids.shape[0]:
            scored.add(int(cid))
            heapq.heappush(frontier, (-float(centroid_scores[int(cid)]), int(cid)))
    visited = set()
    selected_clusters: list[int] = []
    represented = list(base)
    exact = list(base)
    edge_reads = 0
    offset_reads = 0
    pending = set(float(x) for x in target_masses)
    results = []
    exact_cluster_cap = max(0, int(args.retro_exact_clusters))
    approx_logits: list[float] = []
    approx_value_sums: list[np.ndarray] = []
    approx_sizes: list[int] = []
    max_visits = max(1, int(args.max_visits))
    while frontier and len(visited) < max_visits and pending:
        neg, cid = heapq.heappop(frontier)
        if cid in visited:
            continue
        visited.add(cid)
        offset_reads += 1
        selected_clusters.append(cid)
        start, end = ranges[int(cid)]
        toks = [tok for tok in range(start, min(end, scores.shape[0])) if tok not in set(base)]
        represented.extend(toks)
        if exact_cluster_cap <= 0 or len(selected_clusters) <= exact_cluster_cap:
            exact.extend(toks)
        else:
            approx_logits.append(float(-neg))
            approx_value_sums.append(values[np.asarray(toks, dtype=np.int64)].astype(np.float32).sum(axis=0))
            approx_sizes.append(len(toks))
        for nb in graph.neighbors(cid):
            edge_reads += 1
            if nb in visited:
                continue
            if nb not in scored:
                scored.add(nb)
                heapq.heappush(frontier, (-float(centroid_scores[nb]), nb))
        represented_u = unique(represented, len(represented), 0, scores.shape[0])
        mass = float(probs[np.asarray(represented_u, dtype=np.int64)].sum()) if represented_u else 0.0
        for target in sorted(list(pending)):
            if mass >= target:
                results.append(
                    evaluate_candidate(
                        args,
                        "hybrid_centroid_graph",
                        target,
                        scores,
                        values,
                        probs,
                        dense_out,
                        exact,
                        represented,
                        score_reads=len(scored),
                        value_sum_reads=len(approx_logits),
                        edge_reads=edge_reads,
                        offset_reads=offset_reads,
                        nodes_visited=len(visited),
                        clusters_scored=len(scored),
                        clusters_selected=len(selected_clusters),
                        approx_logits=approx_logits,
                        approx_value_sums=approx_value_sums,
                        approx_sizes=approx_sizes,
                    )
                )
                pending.remove(target)
    for target in sorted(pending):
        results.append(
            evaluate_candidate(
                args,
                "hybrid_centroid_graph",
                target,
                scores,
                values,
                probs,
                dense_out,
                exact,
                represented,
                score_reads=len(scored),
                value_sum_reads=len(approx_logits),
                edge_reads=edge_reads,
                offset_reads=offset_reads,
                nodes_visited=len(visited),
                clusters_scored=len(scored),
                clusters_selected=len(selected_clusters),
                approx_logits=approx_logits,
                approx_value_sums=approx_value_sums,
                approx_sizes=approx_sizes,
            )
        )
    return results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Three-way RetroInfer/RA/hybrid mass-target proxy.")
    p.add_argument("--source_npz", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--num_queries", type=int, default=24)
    p.add_argument(
        "--query_selection",
        choices=("random", "first", "even"),
        default="random",
        help="How to choose evaluation query columns from the NPZ.",
    )
    p.add_argument("--mass_targets", default="0.2,0.4,0.6,0.8,0.9")
    p.add_argument(
        "--decode_tokens_filter",
        default="",
        help="Optional comma-separated decode lengths to evaluate from the NPZ positions.",
    )
    p.add_argument("--static_prefix", type=int, default=128)
    p.add_argument("--static_suffix", type=int, default=512)
    p.add_argument("--retro_cluster_size", type=int, default=128)
    p.add_argument("--retro_exact_clusters", type=int, default=0, help="0 means all represented clusters are exact-read.")
    p.add_argument("--q_knn", type=int, default=8)
    p.add_argument("--graph_degree", type=int, default=8)
    p.add_argument("--seed_count", type=int, default=32)
    p.add_argument("--max_visits", type=int, default=2048)
    p.add_argument("--roar_backend", choices=("cpp", "python"), default="cpp")
    p.add_argument("--roar_nq", type=int, default=8)
    p.add_argument("--roar_l", type=int, default=256)
    p.add_argument("--roar_enhance", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--roar_enhance_l", type=int, default=256)
    p.add_argument("--roar_entry", choices=("hub", "max_degree", "self"), default="hub")
    p.add_argument("--roar_max_query_per_pivot", type=int, default=0)
    p.add_argument("--roar_threads", type=int, default=0)
    p.add_argument("--knn_chunk_rows", type=int, default=256)
    p.add_argument("--include_graph_methods", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--enable_extra_baselines", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--extra_baseline_families",
        default="quest,sparq,loki,pqcache,magicpig,pariskv,ivfpq,binary_gated_pqcache,weighted_hamming_pqcache,sign_vq_lut_pqcache",
        help="Comma-separated extra baseline families to run.",
    )
    p.add_argument("--quest_page_sizes", default="16,32,64")
    p.add_argument("--sparq_ranks", default="8,16,32")
    p.add_argument("--loki_ranks", default="8,16,32")
    p.add_argument("--magicpig_configs", default="10:150,10:170")
    p.add_argument("--magicpig_min_collisions", type=int, default=2)
    p.add_argument(
        "--magicpig_adaptive_ladder",
        default="10:150:2,10:300:2,10:150:1,10:300:1,8:150:2,8:300:2,8:150:1,8:300:1,6:150:1,6:300:1,4:150:1",
    )
    p.add_argument("--pariskv_configs", default="8:32:0.01,8:64:0.02,10:64:0.05")
    p.add_argument("--pariskv_rerank_dims", type=int, default=16)
    p.add_argument(
        "--pariskv_adaptive_ladder",
        default="8:32:0.01:16,8:64:0.02:16,10:64:0.05:16,10:96:0.10:32,10:128:0.20:32,12:128:0.50:64,12:128:1.00:64",
    )
    p.add_argument("--pqcache_subvecs", type=int, default=2, help="PQCache-style number of subvectors per key head.")
    p.add_argument("--pqcache_subbits", type=int, default=6, help="PQCache-style codebook bits per subvector.")
    p.add_argument("--pqcache_kmeans_iters", type=int, default=3, help="K-means iterations for the PQCache proxy codebooks.")
    p.add_argument("--ivfpq_coarse_clusters", type=int, default=128)
    p.add_argument("--ivfpq_coarse_iters", type=int, default=3)
    p.add_argument("--ivfpq_nprobes", default="1,2,4,8,16,32,64,128")
    p.add_argument(
        "--ivfpq_online_mode",
        choices=("snapshot", "frozen_append", "online_centroid", "periodic_rebuild"),
        default="snapshot",
        help=(
            "IVF cost-model mode. snapshot is the old static-cutoff model; "
            "frozen_append charges per-token assignment/PQ-code append; "
            "online_centroid also charges centroid read/write updates; "
            "periodic_rebuild additionally amortizes full index rebuilds."
        ),
    )
    p.add_argument(
        "--ivfpq_update_amortize_queries",
        type=int,
        default=0,
        help="Divide per-token IVF update cost by this many query heads; 0 uses q_heads_per_kv_head.",
    )
    p.add_argument(
        "--ivfpq_rebuild_interval",
        type=int,
        default=8192,
        help="Decode-token interval used to amortize periodic_rebuild index refresh cost.",
    )
    p.add_argument(
        "--ivfpq_fixed_nprobes",
        default="",
        help="Optional nprobe values to emit as fixed-policy rows that use all candidates without oracle mass stopping.",
    )
    p.add_argument(
        "--ivfpq_emit_pq_logits",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="For fixed nprobe rows, also emit a PQ-logit variant using approximate QK logits and exact selected V.",
    )
    p.add_argument("--binary_gated_bits", type=int, default=128)
    p.add_argument("--binary_gated_candidate_budgets", default="512,1024,2048,4096,8192,16384,32768")
    p.add_argument("--self_indexing_group_size", type=int, default=4)
    p.add_argument("--self_indexing_candidate_budgets", default="512,1024,2048,4096,8192,16384,32768")
    p.add_argument("--score_key_bytes_per_element", type=int, default=4)
    p.add_argument("--attn_key_bytes_per_element", type=int, default=2)
    p.add_argument("--value_bytes_per_element", type=int, default=2)
    p.add_argument("--edge_index_bytes", type=int, default=4)
    p.add_argument("--graph_offset_bytes", type=int, default=4)
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--head_dim", type=int, default=128)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True))

    data = np.load(args.source_npz)
    keys = np.asarray(data["keys"], dtype=np.float32)
    values = np.asarray(data["values"], dtype=np.float32)
    queries = np.asarray(data["queries"], dtype=np.float32)
    positions = np.asarray(data["positions"], dtype=np.int64)
    if "graph_queries" not in data or "graph_positions" not in data:
        raise RuntimeError("source_npz must contain graph_queries and graph_positions for faithful Roar proxy")
    graph_queries = np.asarray(data["graph_queries"], dtype=np.float32)
    graph_positions = np.asarray(data["graph_positions"], dtype=np.int64)
    meta = json.loads(str(data["metadata"].item())) if "metadata" in data else {}
    input_len = int(meta.get("input_len", int(graph_positions.max()) + 1))
    num_heads, q_count, dim = queries.shape
    kv_heads = keys.shape[0]
    args.head_dim = int(dim)
    group_size = max(1, num_heads // kv_heads)
    args.kv_group_size = int(group_size)
    score_scale = 1.0 / math.sqrt(float(dim))
    target_masses = parse_float_list(args.mass_targets)
    extra_families = parse_name_set(args.extra_baseline_families)

    dynamic_start = min(max(0, int(args.static_prefix)), input_len)
    dynamic_end = max(dynamic_start, input_len - max(0, int(args.static_suffix)))

    graph_cache = {}
    centroid_cache = {}
    loki_cache = {}
    pq_cache = {}
    ivfpq_cache = {}
    binary_gate_cache = {}
    weighted_hamming_cache = {}
    sign_vq_cache = {}
    projection_cache = {}
    rows = []
    candidate_frontier_rows = []
    args.candidate_frontier_rows = candidate_frontier_rows
    rng = np.random.default_rng(int(args.seed))
    q_indices = np.arange(q_count)
    if str(args.decode_tokens_filter).strip():
        keep_decodes = set(parse_int_list(args.decode_tokens_filter))
        decode_by_q = np.asarray([int(pos) - input_len + 1 for pos in positions], dtype=np.int64)
        q_indices = q_indices[np.asarray([int(x) in keep_decodes for x in decode_by_q], dtype=bool)]
        if q_indices.size == 0:
            raise RuntimeError(f"decode_tokens_filter matched no query positions: {args.decode_tokens_filter}")
    if q_indices.shape[0] > int(args.num_queries):
        if args.query_selection == "first":
            q_indices = q_indices[: int(args.num_queries)]
        elif args.query_selection == "even":
            take = np.linspace(0, q_indices.shape[0] - 1, num=int(args.num_queries), dtype=np.int64)
            q_indices = q_indices[take]
        else:
            q_indices = np.sort(rng.choice(q_indices, size=int(args.num_queries), replace=False))

    for qidx in q_indices.tolist():
        pos = int(positions[int(qidx)])
        for head in range(num_heads):
            kv_h = min(kv_heads - 1, int(head * kv_heads // num_heads))
            key = (kv_h, pos)
            if key not in centroid_cache:
                cur_dynamic_end = max(dynamic_start, min(pos + 1 - max(0, int(args.static_suffix)), keys.shape[1]))
                centroids, ranges, _ = build_retro_clusters(
                    keys[kv_h],
                    dynamic_start=dynamic_start,
                    dynamic_end=cur_dynamic_end,
                    cluster_size=int(args.retro_cluster_size),
                )
                centroid_cache[key] = (centroids, ranges, cur_dynamic_end)
            centroids, ranges, cur_dynamic_end = centroid_cache[key]

            token_graph = None
            centroid_graph = None
            if bool(args.include_graph_methods):
                graph_key = (kv_h, cur_dynamic_end)
                if graph_key not in graph_cache:
                    qh_start = kv_h * group_size
                    qh_end = min(num_heads, qh_start + group_size)
                    visible_graph_rows = graph_positions < int(cur_dynamic_end)
                    graph_pos_visible = graph_positions[visible_graph_rows]
                    if graph_pos_visible.size == 0:
                        graph_pos_visible = graph_positions[:1]
                        visible_graph_rows = np.zeros_like(graph_positions, dtype=bool)
                        visible_graph_rows[:1] = True
                    gq = graph_queries[qh_start:qh_end][:, visible_graph_rows, :].reshape(-1, dim)
                    gp = np.tile(graph_pos_visible, qh_end - qh_start)
                    token_knn = exact_topk_rows(
                        gq,
                        gp,
                        keys[kv_h],
                        k=int(args.q_knn),
                        dynamic_start=dynamic_start,
                        dynamic_end=cur_dynamic_end,
                        score_scale=score_scale,
                        chunk_rows=int(args.knn_chunk_rows),
                    )
                    token_graph = build_projected_graph(
                        token_knn,
                        keys[kv_h],
                        args,
                        dynamic_start=dynamic_start,
                        dynamic_end=cur_dynamic_end,
                    )
                    if centroids.shape[0] > 0:
                        centroid_knn = exact_topk_centroid_rows(
                            gq,
                            gp,
                            centroids,
                            ranges,
                            k=int(args.q_knn),
                            score_scale=score_scale,
                            chunk_rows=int(args.knn_chunk_rows),
                        )
                        centroid_graph = build_projected_graph(
                            centroid_knn,
                            centroids,
                            args,
                            dynamic_start=0,
                            dynamic_end=centroids.shape[0],
                        )
                    else:
                        centroid_graph = CsrGraph(
                            np.empty((0, dim), dtype=np.float32),
                            np.zeros((1,), dtype=np.uint32),
                            np.empty((0,), dtype=np.int32),
                            0,
                            0,
                        )
                    graph_cache[graph_key] = (token_graph, centroid_graph)
                token_graph, centroid_graph = graph_cache[graph_key]

            q = queries[head, int(qidx)].astype(np.float32)
            args.current_query = q
            args.current_eval_context = {
                "qidx": int(qidx),
                "position": int(pos),
                "decode_tokens": max(0, int(pos) - input_len + 1),
                "head": int(head),
                "kv_head": int(kv_h),
            }
            usable_keys = keys[kv_h, : pos + 1].astype(np.float32, copy=False)
            scores = (usable_keys @ q) * score_scale
            vals = values[kv_h, : pos + 1].astype(np.float32, copy=False)
            logits = scores - np.max(scores)
            probs = np.exp(logits).astype(np.float32)
            probs /= max(float(probs.sum()), 1e-20)
            dense_out = probs @ vals
            base = static_tokens(pos, args.static_prefix, args.static_suffix)

            method_results = []
            method_results.extend(dense_oracle_results(args, scores, vals, probs, dense_out, base, target_masses))
            method_results.extend(retro_results(args, scores, vals, probs, dense_out, base, centroids, ranges, target_masses, score_scale))
            if bool(args.enable_extra_baselines):
                if "quest" in extra_families:
                    method_results.extend(quest_results(args, scores, vals, probs, dense_out, base, usable_keys, target_masses, score_scale))
                if "sparq" in extra_families:
                    method_results.extend(sparq_results(args, scores, vals, probs, dense_out, base, usable_keys, target_masses))
                if "loki" in extra_families:
                    max_loki_rank = max(parse_int_list(args.loki_ranks))
                    loki_key = (kv_h, cur_dynamic_end, max_loki_rank)
                    if loki_key not in loki_cache:
                        loki_cache[loki_key] = compute_pca_basis(
                            keys[kv_h],
                            dynamic_start=dynamic_start,
                            dynamic_end=cur_dynamic_end,
                            max_rank=max_loki_rank,
                        )
                    loki_basis, loki_mean = loki_cache[loki_key]
                    method_results.extend(
                        loki_results(
                            args,
                            scores,
                            vals,
                            probs,
                            dense_out,
                            base,
                            usable_keys,
                            target_masses,
                            loki_basis,
                            loki_mean,
                            dynamic_start,
                            min(cur_dynamic_end, scores.shape[0]),
                        )
                    )
                if extra_families & {
                    "pqcache",
                    "pqcache_quantized_k",
                    "ivfpq",
                    "binary_gated_pqcache",
                    "weighted_hamming_pqcache",
                    "sign_vq_lut_pqcache",
                }:
                    pq_key = (
                        kv_h,
                        cur_dynamic_end,
                        int(args.pqcache_subvecs),
                        int(args.pqcache_subbits),
                        int(args.pqcache_kmeans_iters),
                    )
                    if pq_key not in pq_cache:
                        pq_cache[pq_key] = build_pq_index(
                            keys[kv_h],
                            dynamic_start=dynamic_start,
                            dynamic_end=cur_dynamic_end,
                            subvecs=int(args.pqcache_subvecs),
                            subbits=int(args.pqcache_subbits),
                            seed=int(args.seed) + 1009 * int(kv_h) + 9176 * int(cur_dynamic_end),
                            max_iter=int(args.pqcache_kmeans_iters),
                        )
                    if "pqcache" in extra_families:
                        method_results.extend(
                            pqcache_results(
                                args,
                                scores,
                                vals,
                                probs,
                                dense_out,
                                base,
                                usable_keys,
                                target_masses,
                                pq_cache[pq_key],
                                dynamic_start,
                                cur_dynamic_end,
                            )
                        )
                    if "pqcache_quantized_k" in extra_families:
                        method_results.extend(
                            pqcache_quantized_k_results(
                                args,
                                scores,
                                vals,
                                probs,
                                dense_out,
                                base,
                                target_masses,
                                pq_cache[pq_key],
                                dynamic_start,
                                cur_dynamic_end,
                                score_scale,
                            )
                        )
                    if "ivfpq" in extra_families:
                        ivfpq_key = (
                            kv_h,
                            cur_dynamic_end,
                            int(args.ivfpq_coarse_clusters),
                            int(args.ivfpq_coarse_iters),
                        )
                        if ivfpq_key not in ivfpq_cache:
                            ivfpq_cache[ivfpq_key] = build_ivfpq_index(
                                keys[kv_h],
                                dynamic_start=dynamic_start,
                                dynamic_end=cur_dynamic_end,
                                coarse_clusters=int(args.ivfpq_coarse_clusters),
                                coarse_iters=int(args.ivfpq_coarse_iters),
                                seed=int(args.seed) + 2027 * int(kv_h) + 7919 * int(cur_dynamic_end),
                            )
                        method_results.extend(
                            ivfpq_adaptive_results(
                                args,
                                scores,
                                vals,
                                probs,
                                dense_out,
                                base,
                                target_masses,
                                (*ivfpq_cache[ivfpq_key], pq_cache[pq_key]),
                                dynamic_start,
                                cur_dynamic_end,
                                score_scale,
                            )
                        )
                    if "binary_gated_pqcache" in extra_families:
                        binary_key = (kv_h, cur_dynamic_end, int(args.binary_gated_bits))
                        if binary_key not in binary_gate_cache:
                            binary_gate_cache[binary_key] = build_binary_gated_index(
                                keys[kv_h],
                                dynamic_start=dynamic_start,
                                dynamic_end=cur_dynamic_end,
                                bits=int(args.binary_gated_bits),
                                seed=int(args.seed) + 3137 * int(kv_h) + 7919 * int(cur_dynamic_end) + int(args.binary_gated_bits),
                            )
                        method_results.extend(
                            binary_gated_pqcache_adaptive_results(
                                args,
                                scores,
                                vals,
                                probs,
                                dense_out,
                                base,
                                target_masses,
                                pq_cache[pq_key],
                                binary_gate_cache[binary_key],
                                dynamic_start,
                                cur_dynamic_end,
                            )
                        )
                    if "weighted_hamming_pqcache" in extra_families:
                        weighted_key = (kv_h, cur_dynamic_end)
                        if weighted_key not in weighted_hamming_cache:
                            weighted_hamming_cache[weighted_key] = build_weighted_hamming_index(
                                keys[kv_h],
                                dynamic_start=dynamic_start,
                                dynamic_end=cur_dynamic_end,
                            )
                        method_results.extend(
                            weighted_hamming_pqcache_adaptive_results(
                                args,
                                scores,
                                vals,
                                probs,
                                dense_out,
                                base,
                                target_masses,
                                pq_cache[pq_key],
                                weighted_hamming_cache[weighted_key],
                                dynamic_start,
                                cur_dynamic_end,
                            )
                        )
                    if "sign_vq_lut_pqcache" in extra_families:
                        sign_key = (kv_h, cur_dynamic_end, int(args.self_indexing_group_size))
                        if sign_key not in sign_vq_cache:
                            sign_vq_cache[sign_key] = build_sign_vq_index(
                                keys[kv_h],
                                dynamic_start=dynamic_start,
                                dynamic_end=cur_dynamic_end,
                                group_size=int(args.self_indexing_group_size),
                            )
                        method_results.extend(
                            sign_vq_lut_pqcache_adaptive_results(
                                args,
                                scores,
                                vals,
                                probs,
                                dense_out,
                                base,
                                target_masses,
                                pq_cache[pq_key],
                                sign_vq_cache[sign_key],
                                dynamic_start,
                                cur_dynamic_end,
                            )
                        )
                if "magicpig" in extra_families:
                    method_results.extend(
                        magicpig_results(
                            args,
                            scores,
                            vals,
                            probs,
                            dense_out,
                            base,
                            usable_keys,
                            target_masses,
                            projection_cache,
                            (kv_h, cur_dynamic_end),
                        )
                    )
                    method_results.extend(
                        magicpig_adaptive_results(
                            args,
                            scores,
                            vals,
                            probs,
                            dense_out,
                            base,
                            usable_keys,
                            target_masses,
                            projection_cache,
                            (kv_h, cur_dynamic_end),
                        )
                    )
                if "pariskv" in extra_families:
                    method_results.extend(
                        pariskv_results(
                            args,
                            scores,
                            vals,
                            probs,
                            dense_out,
                            base,
                            usable_keys,
                            target_masses,
                            projection_cache,
                            (kv_h, cur_dynamic_end),
                        )
                    )
                    method_results.extend(
                        pariskv_adaptive_results(
                            args,
                            scores,
                            vals,
                            probs,
                            dense_out,
                            base,
                            usable_keys,
                            target_masses,
                            projection_cache,
                            (kv_h, cur_dynamic_end),
                        )
                    )
            if bool(args.include_graph_methods):
                method_results.extend(
                    graph_token_results(
                        args,
                        "retrievalattention",
                        token_graph,
                        scores,
                        vals,
                        probs,
                        dense_out,
                        base,
                        target_masses,
                        tail_end=min(pos + 1, cur_dynamic_end),
                    )
                )
                oracle_seeds, oracle_seed_reads = oracle_seed_tokens(args, scores, base, count=int(args.seed_count))
                method_results.extend(
                    graph_token_results(
                        args,
                        "retrievalattention_oracle_seed",
                        token_graph,
                        scores,
                        vals,
                        probs,
                        dense_out,
                        base,
                        target_masses,
                        tail_end=min(pos + 1, cur_dynamic_end),
                        seed_override=oracle_seeds,
                        seed_extra_score_reads=oracle_seed_reads,
                    )
                )
                retro_seeds, retro_seed_reads = retro_seed_tokens(
                    args,
                    scores,
                    base,
                    centroids,
                    ranges,
                    score_scale,
                    count=int(args.seed_count),
                )
                method_results.extend(
                    graph_token_results(
                        args,
                        "retrievalattention_retro_seed",
                        token_graph,
                        scores,
                        vals,
                        probs,
                        dense_out,
                        base,
                        target_masses,
                        tail_end=min(pos + 1, cur_dynamic_end),
                        seed_override=retro_seeds,
                        seed_extra_score_reads=retro_seed_reads,
                    )
                )
                method_results.extend(
                    hybrid_results(args, centroid_graph, scores, vals, probs, dense_out, base, centroids, ranges, target_masses, score_scale)
                )
            for res in method_results:
                row = {
                    "qidx": int(qidx),
                    "position": int(pos),
                    "decode_tokens": max(0, int(pos) - input_len + 1),
                    "head": int(head),
                    "kv_head": int(kv_h),
                    **res.__dict__,
                }
                rows.append(row)

    sample_path = out_dir / "samples.jsonl"
    with sample_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    if candidate_frontier_rows:
        frontier_path = out_dir / "candidate_frontier.jsonl"
        with frontier_path.open("w", encoding="utf-8") as f:
            for row in candidate_frontier_rows:
                f.write(json.dumps(row, sort_keys=True) + "\n")
        frontier_grouped: dict[tuple, list[dict]] = {}
        for row in candidate_frontier_rows:
            frontier_grouped.setdefault((row["decode_tokens"], row["method"], row["budget_kind"], row["budget_value"]), []).append(row)
        frontier_summary = []
        for (decode_tokens, method, budget_kind, budget_value), items in sorted(frontier_grouped.items()):
            frontier_summary.append(
                {
                    "decode_tokens": int(decode_tokens),
                    "method": str(method),
                    "budget_kind": str(budget_kind),
                    "budget_value": int(budget_value),
                    "samples": len(items),
                    "oracle_mass_mean": float(np.mean([x["oracle_mass"] for x in items])),
                    "estimated_mb_pre_pq_mean": float(np.mean([x["estimated_mb_pre_pq"] for x in items])),
                    "candidate_tokens_mean": float(np.mean([x["candidate_tokens"] for x in items])),
                    "represented_tokens_mean": float(np.mean([x["represented_tokens"] for x in items])),
                }
            )
        (out_dir / "candidate_frontier_summary.json").write_text(json.dumps(frontier_summary, indent=2, sort_keys=True))
        with (out_dir / "candidate_frontier_summary.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(frontier_summary[0].keys()) if frontier_summary else ["empty"])
            writer.writeheader()
            writer.writerows(frontier_summary)

    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["decode_tokens"], row["method"], row["target_mass"]), []).append(row)
    summary = []
    for (decode_tokens, method, target), items in sorted(grouped.items()):
        summary.append(
            {
                "decode_tokens": int(decode_tokens),
                "method": method,
                "target_mass": float(target),
                "samples": len(items),
                "reached_rate": float(np.mean([float(x["reached"]) for x in items])),
                "mass_mean": float(np.mean([x["mass"] for x in items])),
                "output_cos_mean": float(np.mean([x["output_cos"] for x in items])),
                "estimated_mb_mean": float(np.mean([x["estimated_mb"] for x in items])),
                "exact_tokens_mean": float(np.mean([x["exact_tokens"] for x in items])),
                "represented_tokens_mean": float(np.mean([x["represented_tokens"] for x in items])),
                "score_reads_mean": float(np.mean([x["score_reads"] for x in items])),
                "score_elements_mean": float(np.mean([x["score_elements"] for x in items])),
                "index_bytes_mean": float(np.mean([x["index_bytes"] for x in items])),
                "edge_reads_mean": float(np.mean([x["edge_reads"] for x in items])),
                "nodes_visited_mean": float(np.mean([x["nodes_visited"] for x in items])),
                "clusters_scored_mean": float(np.mean([x["clusters_scored"] for x in items])),
                "clusters_selected_mean": float(np.mean([x["clusters_selected"] for x in items])),
            }
        )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    with (out_dir / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()) if summary else ["empty"])
        writer.writeheader()
        writer.writerows(summary)
    print(f"[threeway] wrote {out_dir}")
    print(f"[threeway] rows={len(rows)} summary={len(summary)}")


if __name__ == "__main__":
    main()
