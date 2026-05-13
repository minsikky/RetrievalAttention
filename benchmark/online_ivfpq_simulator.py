#!/usr/bin/env python3
"""True-online IVF-PQ selector simulator on saved real QKV traces.

This script is intentionally narrower than attention_efficiency_threeway_eval.py.
It replays generated K vectors into a stateful IVF-PQ index, evaluates sampled
decode queries, and records logical global-memory read/write bytes by event
category. The attention computation remains exact K/V over selected tokens; PQ
is used only for selector scoring.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.attention_efficiency_threeway_eval import (  # noqa: E402
    build_pq_index,
    dense_oracle_results,
    evaluate_candidate,
    lloyd_kmeans,
    parse_float_list,
    sorted_unique_ints,
    static_tokens,
    unique,
)

def _load_cpp_backend():
    wrapper = PROJECT_ROOT / "cache_hub" / "online_ivfpq_cpp_backend.py"
    spec = importlib.util.spec_from_file_location("online_ivfpq_cpp_backend_local", wrapper)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {wrapper}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


try:
    _cpp_backend = _load_cpp_backend()
    assign_encode_batch_cpp = _cpp_backend.assign_encode_batch_cpp
    online_ivfpq_cpp_available = _cpp_backend.online_ivfpq_cpp_available
    rank_nprobes_cpp = _cpp_backend.rank_nprobes_cpp
except Exception:  # pragma: no cover
    assign_encode_batch_cpp = None
    rank_nprobes_cpp = None

    def online_ivfpq_cpp_available() -> bool:
        return False


@dataclass
class EventBytes:
    reads: dict[str, float] = field(default_factory=dict)
    writes: dict[str, float] = field(default_factory=dict)

    def read(self, category: str, bytes_: float) -> None:
        self.reads[str(category)] = self.reads.get(str(category), 0.0) + float(bytes_)

    def write(self, category: str, bytes_: float) -> None:
        self.writes[str(category)] = self.writes.get(str(category), 0.0) + float(bytes_)

    def add(self, other: "EventBytes", *, scale: float = 1.0) -> None:
        for key, value in other.reads.items():
            self.read(key, float(value) * float(scale))
        for key, value in other.writes.items():
            self.write(key, float(value) * float(scale))

    @property
    def read_bytes(self) -> float:
        return float(sum(self.reads.values()))

    @property
    def write_bytes(self) -> float:
        return float(sum(self.writes.values()))

    @property
    def total_bytes(self) -> float:
        return self.read_bytes + self.write_bytes

    def mb(self) -> float:
        return self.total_bytes / (1024.0 * 1024.0)

    def prefixed_flat(self, prefix: str) -> dict[str, float]:
        out: dict[str, float] = {
            f"{prefix}_read_mb": self.read_bytes / (1024.0 * 1024.0),
            f"{prefix}_write_mb": self.write_bytes / (1024.0 * 1024.0),
            f"{prefix}_total_mb": self.mb(),
        }
        for key, value in sorted(self.reads.items()):
            out[f"{prefix}_read_{key}_mb"] = float(value) / (1024.0 * 1024.0)
        for key, value in sorted(self.writes.items()):
            out[f"{prefix}_write_{key}_mb"] = float(value) / (1024.0 * 1024.0)
        return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="True-online IVF-PQ selector simulator.")
    p.add_argument("--source_npz", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--policies", default="frozen_append,online_centroid,periodic_rebuild")
    p.add_argument("--mass_targets", default="0.95,0.98")
    p.add_argument("--decode_tokens_filter", default="")
    p.add_argument("--query_selection", choices=("first", "even", "random", "all"), default="all")
    p.add_argument("--num_queries", type=int, default=0, help="0 means all selected query positions.")
    p.add_argument("--static_prefix", type=int, default=128)
    p.add_argument("--static_suffix", type=int, default=128)
    p.add_argument("--ivfpq_coarse_clusters", type=int, default=128)
    p.add_argument("--ivfpq_coarse_iters", type=int, default=3)
    p.add_argument("--ivfpq_nprobes", default="1,2,4,8,16,32,64,128")
    p.add_argument("--ivfpq_final_ks", default="512,1024,2048,4096,8192,16384,32768,65536")
    p.add_argument("--skip_fixedk", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--ivfpq_rebuild_interval", type=int, default=8192)
    p.add_argument("--paged_pq_page_size", type=int, default=0, help="0 means use static_suffix as the page size.")
    p.add_argument("--paged_router_prototypes", type=int, default=16)
    p.add_argument("--paged_router_merge_rel", type=float, default=0.5)
    p.add_argument("--paged_router_merge_var", type=float, default=0.0, help="0 disables absolute merged-variance merge criterion.")
    p.add_argument("--paged_router_max_groups", type=int, default=0, help="0 leaves routed prototype groups uncapped.")
    p.add_argument("--pqcache_subvecs", type=int, default=2)
    p.add_argument("--pqcache_subbits", type=int, default=6)
    p.add_argument("--pqcache_kmeans_iters", type=int, default=3)
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--score_key_bytes_per_element", type=int, default=4)
    p.add_argument("--attn_key_bytes_per_element", type=int, default=2)
    p.add_argument("--value_bytes_per_element", type=int, default=2)
    p.add_argument("--edge_index_bytes", type=int, default=4)
    p.add_argument("--graph_offset_bytes", type=int, default=4)
    p.add_argument("--head_dim", type=int, default=128)
    p.add_argument("--emit_samples", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--compute_output_cos", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--progress_every", type=int, default=8, help="Print progress after this many query positions per policy.")
    p.add_argument("--backend", choices=("auto", "python", "cpp"), default="auto")
    p.add_argument("--backend_threads", type=int, default=0)
    return p.parse_args()


def parse_name_list(text: str) -> list[str]:
    return [part.strip().lower() for part in str(text).replace(";", ",").split(",") if part.strip()]


def pq_code_bytes(args: argparse.Namespace) -> int:
    return 1 if int(args.pqcache_subbits) <= 8 else 2


def cosine_output(scores: np.ndarray, values: np.ndarray, dense_out: np.ndarray, tokens: list[int]) -> float:
    tokens = unique(tokens, len(tokens), 0, scores.shape[0])
    if not tokens:
        sparse_out = np.zeros_like(dense_out)
    else:
        idx = np.asarray(tokens, dtype=np.int64)
        logits = scores[idx].astype(np.float32)
        w = np.exp(logits - np.max(logits)).astype(np.float32)
        sparse_out = (w[:, None] * values[idx].astype(np.float32)).sum(axis=0) / max(float(w.sum()), 1e-20)
    denom = max(float(np.linalg.norm(sparse_out) * np.linalg.norm(dense_out)), 1e-20)
    return float(np.dot(sparse_out, dense_out) / denom)


def encode_pq_rows(block: np.ndarray, codebooks: np.ndarray) -> np.ndarray:
    block = block.astype(np.float32, copy=False)
    subvecs = int(codebooks.shape[0])
    subdim = int(codebooks.shape[-1])
    codes = np.zeros((block.shape[0], subvecs), dtype=np.uint16)
    for sub in range(subvecs):
        part = block[:, sub * subdim : (sub + 1) * subdim]
        centers = codebooks[sub].astype(np.float32, copy=False)
        dist = (
            np.sum(part * part, axis=1, keepdims=True)
            + np.sum(centers * centers, axis=1, keepdims=True).T
            - 2.0 * (part @ centers.T)
        )
        codes[:, sub] = np.argmin(dist, axis=1).astype(np.uint16, copy=False)
    return codes


def pq_scores(q: np.ndarray, codebooks: np.ndarray, codes: np.ndarray) -> np.ndarray:
    if codes.shape[0] == 0:
        return np.empty((0,), dtype=np.float32)
    q = q.astype(np.float32, copy=False)
    subvecs = int(codebooks.shape[0])
    subdim = q.shape[0] // subvecs
    q_parts = q.reshape(subvecs, subdim)
    table = np.einsum("ms,mcs->mc", q_parts, codebooks.astype(np.float32, copy=False), optimize=True)
    approx = np.zeros((codes.shape[0],), dtype=np.float32)
    for sub in range(subvecs):
        approx += table[sub, codes[:, sub]]
    return approx


def pq_score_error_bounds(q: np.ndarray, radii: np.ndarray, codes: np.ndarray) -> np.ndarray:
    if codes.shape[0] == 0:
        return np.empty((0,), dtype=np.float32)
    q = q.astype(np.float32, copy=False)
    subvecs = int(radii.shape[0])
    subdim = q.shape[0] // subvecs
    q_norms = np.linalg.norm(q.reshape(subvecs, subdim), axis=1).astype(np.float32, copy=False)
    bounds = np.zeros((codes.shape[0],), dtype=np.float32)
    for sub in range(subvecs):
        bounds += q_norms[sub] * radii[sub, codes[:, sub]]
    return bounds


class OnlineIVFPQIndex:
    def __init__(
        self,
        *,
        keys: np.ndarray,
        init_start: int,
        init_end: int,
        policy: str,
        args: argparse.Namespace,
        seed: int,
        router_enabled: bool = False,
    ) -> None:
        self.keys = keys.astype(np.float32, copy=False)
        self.policy = str(policy)
        self.args = args
        self.seed = int(seed)
        self.dim = int(keys.shape[-1])
        self.subvecs = int(args.pqcache_subvecs)
        self.subbits = int(args.pqcache_subbits)
        self.centroids_per_subvec = 1 << self.subbits
        self.rebuild_interval = max(1, int(args.ivfpq_rebuild_interval))
        self.token_start = int(init_start)
        self.size = max(0, int(init_end) - int(init_start))
        self.capacity = max(1, int(keys.shape[0]) - self.token_start)
        self.codes = np.zeros((self.capacity, self.subvecs), dtype=np.uint16)
        self.assign = np.full((self.capacity,), -1, dtype=np.int32)
        self.update_events_total = EventBytes()
        self.init_events = EventBytes()
        self.appended_since_rebuild = 0
        self.total_update_steps = 0
        self.use_cpp = str(getattr(args, "backend", "auto")) != "python" and online_ivfpq_cpp_available()
        if str(getattr(args, "backend", "auto")) == "cpp" and not self.use_cpp:
            raise RuntimeError("requested --backend cpp, but online_ivfpq_ext is unavailable")
        self._rebuild(seed_offset=0, events=self.init_events)

    @property
    def token_ids(self) -> np.ndarray:
        return np.arange(self.token_start, self.token_start + self.size, dtype=np.int64)

    def _key_bytes(self, count: int) -> int:
        return int(count) * self.dim * int(self.args.attn_key_bytes_per_element)

    def _centroid_bytes(self, count: int | None = None) -> int:
        count = int(self.centroids.shape[0] if count is None else count)
        return count * self.dim * int(self.args.score_key_bytes_per_element)

    def _pq_codebook_bytes(self) -> int:
        return int(self.subvecs) * int(self.centroids_per_subvec) * (self.dim // int(self.subvecs)) * int(
            self.args.score_key_bytes_per_element
        )

    def _pq_code_bytes(self, count: int) -> int:
        return int(count) * int(self.subvecs) * pq_code_bytes(self.args)

    def _rebuild(self, *, seed_offset: int, events: EventBytes) -> None:
        if self.size == 0:
            self.centroids = np.empty((0, self.dim), dtype=np.float32)
            self.buckets = []
            self.codebooks = np.empty((self.subvecs, self.centroids_per_subvec, self.dim // self.subvecs), dtype=np.float32)
            self.counts = np.empty((0,), dtype=np.int64)
            return
        token_ids = self.token_ids
        block = self.keys[token_ids].astype(np.float32, copy=False)
        events.read("rebuild_keys", self._key_bytes(block.shape[0]))
        events.read(
            "rebuild_coarse_work",
            float(int(self.args.ivfpq_coarse_iters))
            * float(block.shape[0])
            * float(max(1, int(self.args.ivfpq_coarse_clusters)))
            * float(self.dim)
            * float(int(self.args.score_key_bytes_per_element)),
        )
        centroids, assign = lloyd_kmeans(
            block,
            int(self.args.ivfpq_coarse_clusters),
            seed=self.seed + int(seed_offset),
            max_iter=int(self.args.ivfpq_coarse_iters),
        )
        self.centroids = centroids
        self.assign[: self.size] = assign
        self.buckets = [
            np.where(assign == cid)[0].astype(np.int64, copy=False).tolist() for cid in range(self.centroids.shape[0])
        ]
        # build_pq_index cannot build arbitrary token ids, so train codebooks
        # directly on this block using the same helper on a temporary contiguous view.
        tmp_index = build_pq_index(
            block,
            0,
            block.shape[0],
            subvecs=self.subvecs,
            subbits=self.subbits,
            seed=self.seed + 1009 + int(seed_offset),
            max_iter=int(self.args.pqcache_kmeans_iters),
        )
        self.codebooks, codes, self.subvecs, self.centroids_per_subvec = tmp_index
        self.codes[: self.size] = codes
        self.counts = np.bincount(assign, minlength=self.centroids.shape[0]).astype(np.int64)
        self.centroid_sums = np.zeros_like(self.centroids, dtype=np.float32)
        for cid in range(self.centroids.shape[0]):
            mask = assign == cid
            if np.any(mask):
                self.centroid_sums[cid] = block[mask].sum(axis=0)
        events.write("coarse_centroids", self._centroid_bytes())
        events.write("pq_codebooks", self._pq_codebook_bytes())
        events.write("pq_codes", self._pq_code_bytes(self.size))
        events.write("postings", int(self.size) * int(self.args.edge_index_bytes))
        events.write("offsets", (len(self.buckets) + 1) * int(self.args.graph_offset_bytes))
        self.appended_since_rebuild = 0

    def advance_to(self, indexed_hi: int) -> None:
        indexed_hi = min(max(0, int(indexed_hi)), self.keys.shape[0])
        next_tok = self.token_start + self.size
        if next_tok < indexed_hi and self.use_cpp:
            self._append_batch_cpp(next_tok, indexed_hi)
            return
        while next_tok < indexed_hi:
            self._append_one(next_tok)
            next_tok += 1

    def _append_batch_cpp(self, start: int, end: int) -> None:
        count = max(0, int(end) - int(start))
        if count <= 0:
            return
        if self.centroids.shape[0] == 0:
            self.token_start = int(start)
            self.size = 0
            self._append_one(start)
            if start + 1 < end:
                self._append_batch_cpp(start + 1, end)
            return
        if self.policy == "periodic_rebuild":
            cursor = int(start)
            while cursor < int(end):
                remaining = max(1, self.rebuild_interval - int(self.appended_since_rebuild))
                chunk_end = min(int(end), cursor + remaining)
                self._append_batch_cpp_no_rebuild(cursor, chunk_end)
                if self.appended_since_rebuild >= self.rebuild_interval:
                    events = EventBytes()
                    self._rebuild(seed_offset=chunk_end - 1, events=events)
                    self.update_events_total.add(events)
                cursor = chunk_end
            return
        self._append_batch_cpp_no_rebuild(int(start), int(end))

    def _append_batch_cpp_no_rebuild(self, start: int, end: int) -> None:
        count = max(0, int(end) - int(start))
        if count <= 0:
            return
        events = EventBytes()
        batch = self.keys[int(start):int(end)].astype(np.float32, copy=False)
        events.read("append_key", self._key_bytes(count))
        events.read("coarse_centroids", count * self._centroid_bytes())
        events.read("pq_codebooks", count * self._pq_codebook_bytes())
        update_centroids = self.policy in {"online_centroid", "periodic_rebuild"}
        assign, codes = assign_encode_batch_cpp(
            batch,
            self.centroids,
            self.codebooks,
            self.centroid_sums,
            self.counts,
            update_centroids=update_centroids,
            num_threads=int(getattr(self.args, "backend_threads", 0)),
        )
        assign = np.asarray(assign, dtype=np.int32)
        codes = np.asarray(codes, dtype=np.uint16)
        row_start = int(self.size)
        row_end = row_start + count
        self.assign[row_start:row_end] = assign
        self.codes[row_start:row_end] = codes
        for local, cid in enumerate(assign.tolist()):
            self.buckets[int(cid)].append(row_start + int(local))
        self.size += count
        events.write("pq_codes", self._pq_code_bytes(count))
        events.write("postings", count * int(self.args.edge_index_bytes))
        if update_centroids:
            events.read("centroid_update", count * (self.dim * int(self.args.score_key_bytes_per_element) + 4))
            events.write("centroid_update", count * (self.dim * int(self.args.score_key_bytes_per_element) + 4))
        self.appended_since_rebuild += count
        self.update_events_total.add(events)
        self.total_update_steps += count

    def _append_one(self, token_id: int) -> None:
        events = EventBytes()
        key = self.keys[int(token_id)].astype(np.float32, copy=False)
        events.read("append_key", self._key_bytes(1))
        if self.centroids.shape[0] == 0:
            self.token_start = int(token_id)
            self.size = 1
            self._rebuild(seed_offset=int(token_id), events=events)
            self.update_events_total.add(events)
            self.total_update_steps += 1
            return
        events.read("coarse_centroids", self._centroid_bytes())
        scores = self.centroids.astype(np.float32, copy=False) @ key
        cid = int(np.argmax(scores))
        events.read("pq_codebooks", self._pq_codebook_bytes())
        code = encode_pq_rows(key.reshape(1, -1), self.codebooks)
        row_id = int(self.size)
        self.assign[row_id] = int(cid)
        self.codes[row_id] = code[0].astype(np.uint16, copy=False)
        self.buckets[cid].append(row_id)
        self.size += 1
        events.write("pq_codes", self._pq_code_bytes(1))
        events.write("postings", int(self.args.edge_index_bytes))
        if self.policy in {"online_centroid", "periodic_rebuild"}:
            events.read("centroid_update", self.dim * int(self.args.score_key_bytes_per_element) + 4)
            events.write("centroid_update", self.dim * int(self.args.score_key_bytes_per_element) + 4)
            old_count = int(self.counts[cid])
            self.centroids[cid] = (self.centroids[cid] * float(old_count) + key) / float(old_count + 1)
            self.counts[cid] = old_count + 1
        else:
            self.counts[cid] += 1
        self.appended_since_rebuild += 1
        if self.policy == "periodic_rebuild" and self.appended_since_rebuild >= self.rebuild_interval:
            self._rebuild(seed_offset=int(token_id), events=events)
        self.update_events_total.add(events)
        self.total_update_steps += 1

    def average_update_events_per_query(self, group_size: int) -> EventBytes:
        out = EventBytes()
        if self.total_update_steps <= 0:
            return out
        out.add(self.update_events_total, scale=1.0 / float(self.total_update_steps * max(1, int(group_size))))
        return out

    def selection(self, q: np.ndarray, nprobe: int) -> tuple[np.ndarray, np.ndarray, EventBytes]:
        events = EventBytes()
        if self.size == 0 or self.centroids.shape[0] == 0:
            return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float32), events
        q = q.astype(np.float32, copy=False)
        nprobes_arr = np.asarray([nprobe], dtype=np.int32)
        if self.use_cpp:
            ranked = rank_nprobes_cpp(
                q,
                self.centroids,
                self.codebooks,
                self.codes[: self.size],
                self.assign[: self.size],
                nprobes_arr,
                size=int(self.size),
                token_start=int(self.token_start),
                num_threads=int(getattr(self.args, "backend_threads", 0)),
            )[int(nprobe)]
            events.read("coarse_centroids", self._centroid_bytes())
            events.read("offsets", int(nprobe) * int(self.args.graph_offset_bytes))
            events.read("postings", int(len(ranked)) * int(self.args.edge_index_bytes))
            events.read("pq_codebooks", self._pq_codebook_bytes())
            events.read("pq_codes", self._pq_code_bytes(int(len(ranked))))
            return np.asarray(ranked, dtype=np.int64), np.empty((len(ranked),), dtype=np.float32), events
        nprobe = max(1, min(int(nprobe), self.centroids.shape[0]))
        events.read("coarse_centroids", self._centroid_bytes())
        coarse_scores = self.centroids.astype(np.float32, copy=False) @ q
        order = np.argsort(-coarse_scores, kind="stable")[:nprobe]
        events.read("offsets", int(nprobe) * int(self.args.graph_offset_bytes))
        rows = [self.buckets[int(cid)] for cid in order if len(self.buckets[int(cid)]) > 0]
        if not rows:
            return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float32), events
        row_ids = np.asarray([row for bucket in rows for row in bucket], dtype=np.int64)
        events.read("postings", row_ids.size * int(self.args.edge_index_bytes))
        events.read("pq_codebooks", self._pq_codebook_bytes())
        events.read("pq_codes", self._pq_code_bytes(row_ids.size))
        approx = pq_scores(q, self.codebooks, self.codes[row_ids])
        score_order = np.argsort(-approx, kind="stable")
        ranked_rows = row_ids[score_order]
        ranked_tokens = ranked_rows + int(self.token_start)
        return ranked_tokens.astype(np.int64, copy=False), approx[score_order], events

    def selection_many(self, q: np.ndarray, nprobes: list[int]) -> dict[int, tuple[np.ndarray, EventBytes]]:
        out: dict[int, tuple[np.ndarray, EventBytes]] = {}
        if self.size == 0 or self.centroids.shape[0] == 0:
            for nprobe in nprobes:
                out[int(nprobe)] = (np.empty((0,), dtype=np.int64), EventBytes())
            return out
        q = q.astype(np.float32, copy=False)
        if self.use_cpp:
            nprobes_arr = np.asarray(nprobes, dtype=np.int32)
            ranked_by_probe = rank_nprobes_cpp(
                q,
                self.centroids,
                self.codebooks,
                self.codes[: self.size],
                self.assign[: self.size],
                nprobes_arr,
                size=int(self.size),
                token_start=int(self.token_start),
                num_threads=int(getattr(self.args, "backend_threads", 0)),
            )
            for nprobe in nprobes:
                ranked = np.asarray(ranked_by_probe[int(nprobe)], dtype=np.int64)
                events = EventBytes()
                events.read("coarse_centroids", self._centroid_bytes())
                events.read("offsets", int(nprobe) * int(self.args.graph_offset_bytes))
                events.read("postings", int(len(ranked)) * int(self.args.edge_index_bytes))
                events.read("pq_codebooks", self._pq_codebook_bytes())
                events.read("pq_codes", self._pq_code_bytes(int(len(ranked))))
                out[int(nprobe)] = (ranked, events)
            return out
        for nprobe in nprobes:
            ranked, _approx, events = self.selection(q, int(nprobe))
            out[int(nprobe)] = (ranked, events)
        return out


class PagedLocalPQIndex:
    """Append-only page-local PQ selector.

    Full pages are sealed into independent PQ indexes. The not-yet-full page is
    left exact and should be included in the represented set by the caller.
    """

    def __init__(
        self,
        *,
        keys: np.ndarray,
        init_start: int,
        init_end: int,
        args: argparse.Namespace,
        seed: int,
        router_enabled: bool = False,
    ) -> None:
        self.keys = keys.astype(np.float32, copy=False)
        self.args = args
        self.seed = int(seed)
        self.dim = int(keys.shape[-1])
        self.subvecs = int(args.pqcache_subvecs)
        self.subbits = int(args.pqcache_subbits)
        self.centroids_per_subvec = 1 << self.subbits
        default_page = max(1, int(args.static_suffix))
        self.page_size = max(1, int(args.paged_pq_page_size) or default_page)
        self.router_enabled = bool(router_enabled)
        self.router_prototypes = max(1, int(args.paged_router_prototypes))
        self.router_merge_rel = max(0.0, float(args.paged_router_merge_rel))
        self.router_merge_var = max(0.0, float(args.paged_router_merge_var))
        self.router_max_groups = max(0, int(args.paged_router_max_groups))
        self.pq_permutation = str(getattr(args, "paged_pq_permutation", "none")).strip().lower()
        self.verify_proj_dim = max(0, int(getattr(args, "paged_verify_proj_dim", 0)))
        if self.verify_proj_dim > 0:
            rng = np.random.default_rng(self.seed + 104729)
            signs = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=(self.dim, self.verify_proj_dim))
            self.verify_proj_matrix = (signs / np.sqrt(float(self.verify_proj_dim))).astype(np.float32, copy=False)
        else:
            self.verify_proj_matrix = np.empty((self.dim, 0), dtype=np.float32)
        self.token_start = int(init_start)
        self.pending_start = int(init_start)
        self.indexed_hi = int(init_start)
        self.pages: list[dict] = []
        self.groups: list[dict] = []
        self.update_events_total = EventBytes()
        self.init_events = EventBytes()
        self.total_update_steps = 0
        self.advance_to(int(init_end), events=self.init_events, count_as_update=False)

    @property
    def size(self) -> int:
        return int(sum(int(page["size"]) for page in self.pages))

    def _key_bytes(self, count: int) -> int:
        return int(count) * self.dim * int(self.args.attn_key_bytes_per_element)

    def _pq_codebook_bytes_per_page(self) -> int:
        return int(self.subvecs) * int(self.centroids_per_subvec) * (self.dim // int(self.subvecs)) * int(
            self.args.score_key_bytes_per_element
        )

    def _pq_radius_bytes_per_page(self) -> int:
        return int(self.subvecs) * int(self.centroids_per_subvec) * int(self.args.score_key_bytes_per_element)

    def _pq_code_bytes(self, count: int) -> int:
        return int(count) * int(self.subvecs) * pq_code_bytes(self.args)

    def _prototype_bytes(self, count: int) -> int:
        return int(count) * self.dim * int(self.args.score_key_bytes_per_element)

    def _ref_bytes(self, count: int) -> int:
        return int(count) * 2 * int(self.args.edge_index_bytes)

    def _perm_bytes(self) -> int:
        return int(self.dim) * 2

    def _verify_proj_bytes(self, count: int) -> int:
        return int(count) * int(self.verify_proj_dim) * int(self.args.attn_key_bytes_per_element)

    def _verify_proj_matrix_bytes(self) -> int:
        return int(self.dim) * int(self.verify_proj_dim) * int(self.args.score_key_bytes_per_element)

    def _page_permutation(self, block: np.ndarray) -> np.ndarray:
        if self.pq_permutation in {"", "none", "identity"}:
            return np.arange(self.dim, dtype=np.int64)
        if self.pq_permutation == "interleave":
            subdim = self.dim // int(self.subvecs)
            return np.asarray(
                [int(offset * int(self.subvecs) + sub) for sub in range(int(self.subvecs)) for offset in range(subdim)],
                dtype=np.int64,
            )
        if self.pq_permutation in {"variance_balanced", "var_balanced", "balanced"}:
            var = np.var(block.astype(np.float32, copy=False), axis=0)
            order = np.argsort(-var, kind="stable")
            buckets = [[] for _ in range(int(self.subvecs))]
            loads = np.zeros((int(self.subvecs),), dtype=np.float64)
            for dim in order.tolist():
                bucket = int(np.argmin(loads))
                buckets[bucket].append(int(dim))
                loads[bucket] += float(var[int(dim)])
            for bucket in buckets:
                bucket.sort()
            return np.asarray([dim for bucket in buckets for dim in bucket], dtype=np.int64)
        raise ValueError(f"unknown paged_pq_permutation: {self.pq_permutation}")

    def _page_query(self, q: np.ndarray, page: dict, events: EventBytes) -> np.ndarray:
        perm = page.get("perm")
        if perm is None:
            return q
        events.read("page_pq_permutation", self._perm_bytes())
        return q[np.asarray(perm, dtype=np.int64)]

    def _seal_one_page(self, start: int, events: EventBytes) -> None:
        end = min(int(start) + self.page_size, self.keys.shape[0])
        if end <= int(start):
            return
        block = self.keys[int(start):end].astype(np.float32, copy=False)
        events.read("page_build_keys", self._key_bytes(block.shape[0]))
        events.read(
            "page_pq_build_work",
            float(int(self.args.pqcache_kmeans_iters))
            * float(block.shape[0])
            * float(self.centroids_per_subvec)
            * float(self.dim)
            * float(int(self.args.score_key_bytes_per_element)),
        )
        perm = self._page_permutation(block)
        permuted_block = block[:, perm].astype(np.float32, copy=False)
        tmp_index = build_pq_index(
            permuted_block,
            0,
            permuted_block.shape[0],
            subvecs=self.subvecs,
            subbits=self.subbits,
            seed=self.seed + 7919 + int(start),
            max_iter=int(self.args.pqcache_kmeans_iters),
        )
        codebooks, codes, self.subvecs, self.centroids_per_subvec = tmp_index
        page: dict = {
            "token_start": int(start),
            "size": int(block.shape[0]),
            "codebooks": codebooks.astype(np.float32, copy=False),
            "codes": codes.astype(np.uint16, copy=False),
            "radii": self._pq_residual_radii(permuted_block, codebooks, codes),
        }
        if self.pq_permutation not in {"", "none", "identity"}:
            page["perm"] = perm.astype(np.uint16, copy=False)
        if self.verify_proj_dim > 0:
            page["verify_proj"] = (block @ self.verify_proj_matrix).astype(np.float32, copy=False)
        if self.router_enabled:
            events.read(
                "page_proto_build_work",
                float(int(self.args.pqcache_kmeans_iters))
                * float(block.shape[0])
                * float(max(1, min(self.router_prototypes, block.shape[0])))
                * float(self.dim)
                * float(int(self.args.score_key_bytes_per_element)),
            )
            proto_centers, proto_assign = lloyd_kmeans(
                block,
                self.router_prototypes,
                seed=self.seed + 1543 + int(start),
                max_iter=int(self.args.pqcache_kmeans_iters),
            )
            proto_rows: list[np.ndarray] = []
            proto_sse: list[float] = []
            for proto_id in range(proto_centers.shape[0]):
                rows = np.nonzero(proto_assign == proto_id)[0].astype(np.int64, copy=False)
                proto_rows.append(rows)
                if rows.size == 0:
                    proto_sse.append(0.0)
                else:
                    diff = block[rows] - proto_centers[proto_id].reshape(1, -1)
                    proto_sse.append(float(np.sum(diff * diff)))
            page["proto_centers"] = proto_centers.astype(np.float32, copy=False)
            page["proto_rows"] = proto_rows
            page["proto_sse"] = proto_sse
            events.write("page_prototypes", self._prototype_bytes(proto_centers.shape[0]))
            events.write("page_proto_postings", self._ref_bytes(block.shape[0]))
        page_id = len(self.pages)
        self.pages.append(page)
        if self.router_enabled:
            self._merge_page_prototypes(page_id, events)
        events.write("page_pq_codebooks", self._pq_codebook_bytes_per_page())
        events.write("page_pq_radii", self._pq_radius_bytes_per_page())
        events.write("page_pq_codes", self._pq_code_bytes(block.shape[0]))
        if "perm" in page:
            events.write("page_pq_permutation", self._perm_bytes())
        if "verify_proj" in page:
            events.write("page_verify_proj", self._verify_proj_bytes(block.shape[0]))
        events.write("page_meta", 2 * int(self.args.graph_offset_bytes))

    def _pq_residual_radii(self, block: np.ndarray, codebooks: np.ndarray, codes: np.ndarray) -> np.ndarray:
        radii = np.zeros((int(self.subvecs), int(self.centroids_per_subvec)), dtype=np.float32)
        if block.shape[0] == 0:
            return radii
        subdim = int(self.dim) // int(self.subvecs)
        for sub in range(int(self.subvecs)):
            part = block[:, sub * subdim : (sub + 1) * subdim].astype(np.float32, copy=False)
            assigned = codebooks[sub, codes[:, sub]].astype(np.float32, copy=False)
            residual = np.linalg.norm(part - assigned, axis=1).astype(np.float32, copy=False)
            for code in range(int(self.centroids_per_subvec)):
                mask = codes[:, sub] == code
                if np.any(mask):
                    radii[sub, code] = float(np.max(residual[mask]))
        return radii

    def _merge_page_prototypes(self, page_id: int, events: EventBytes) -> None:
        page = self.pages[int(page_id)]
        centers = page["proto_centers"]
        proto_rows = page["proto_rows"]
        proto_sse = page["proto_sse"]
        for proto_id in range(centers.shape[0]):
            count = int(proto_rows[proto_id].size)
            if count <= 0:
                continue
            mean = centers[proto_id].astype(np.float32, copy=True)
            sse = float(proto_sse[proto_id])
            best_gid = -1
            best_delta = float("inf")
            force_gid = -1
            force_delta = float("inf")
            for gid, group in enumerate(self.groups):
                g_count = int(group["count"])
                if g_count <= 0:
                    continue
                diff = group["mean"] - mean
                dist2 = float(np.dot(diff, diff))
                delta = float(g_count * count) / float(g_count + count) * dist2
                if delta < force_delta:
                    force_delta = delta
                    force_gid = gid
                rel = delta / max(float(group["sse"]) + sse, 1e-6)
                merged_var = (float(group["sse"]) + sse + delta) / float(g_count + count)
                if rel <= self.router_merge_rel or (self.router_merge_var > 0.0 and merged_var <= self.router_merge_var):
                    if delta < best_delta:
                        best_delta = delta
                        best_gid = gid
            can_create_group = self.router_max_groups <= 0 or len(self.groups) < self.router_max_groups
            if best_gid < 0 and can_create_group:
                self.groups.append(
                    {
                        "mean": mean,
                        "count": count,
                        "sse": sse,
                        "members": [(int(page_id), int(proto_id))],
                    }
                )
                events.write("router_group", self._prototype_bytes(1) + 3 * int(self.args.graph_offset_bytes))
                events.write("router_postings", self._ref_bytes(1))
                continue
            if best_gid < 0:
                best_gid = force_gid
                best_delta = force_delta
            if best_gid < 0:
                continue
            group = self.groups[best_gid]
            old_count = int(group["count"])
            new_count = old_count + count
            group["mean"] = (
                group["mean"].astype(np.float32, copy=False) * float(old_count) + mean * float(count)
            ) / float(new_count)
            group["sse"] = float(group["sse"]) + sse + best_delta
            group["count"] = new_count
            group["members"].append((int(page_id), int(proto_id)))
            events.read("router_group", self._prototype_bytes(1) + 3 * int(self.args.graph_offset_bytes))
            events.write("router_group", self._prototype_bytes(1) + 3 * int(self.args.graph_offset_bytes))
            events.write("router_postings", self._ref_bytes(1))

    def advance_to(self, indexed_hi: int, events: EventBytes | None = None, count_as_update: bool = True) -> None:
        indexed_hi = min(max(int(self.token_start), int(indexed_hi)), self.keys.shape[0])
        if indexed_hi <= self.indexed_hi:
            return
        local_events = events if events is not None else EventBytes()
        self.indexed_hi = indexed_hi
        sealed = 0
        while self.pending_start + self.page_size <= self.indexed_hi:
            self._seal_one_page(self.pending_start, local_events)
            self.pending_start += self.page_size
            sealed += self.page_size
        if events is None and (local_events.reads or local_events.writes):
            self.update_events_total.add(local_events)
        if count_as_update and sealed > 0:
            self.total_update_steps += int(sealed)

    def pending_tokens(self) -> list[int]:
        if self.indexed_hi <= self.pending_start:
            return []
        return list(range(int(self.pending_start), int(self.indexed_hi)))

    def average_update_events_per_query(self, group_size: int) -> EventBytes:
        out = EventBytes()
        if self.total_update_steps <= 0:
            return out
        out.add(self.update_events_total, scale=1.0 / float(self.total_update_steps * max(1, int(group_size))))
        return out

    def selection_fullscan_scored(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, EventBytes]:
        events = EventBytes()
        if not self.pages:
            return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float32), events
        q = q.astype(np.float32, copy=False)
        ranked_tokens_parts = []
        score_parts = []
        for page in self.pages:
            codebooks = page["codebooks"]
            codes = page["codes"]
            events.read("page_pq_codebooks", self._pq_codebook_bytes_per_page())
            events.read("page_pq_codes", self._pq_code_bytes(int(page["size"])))
            page_scores = pq_scores(self._page_query(q, page, events), codebooks, codes)
            page_tokens = int(page["token_start"]) + np.arange(int(page["size"]), dtype=np.int64)
            ranked_tokens_parts.append(page_tokens)
            score_parts.append(page_scores)
        tokens = np.concatenate(ranked_tokens_parts) if ranked_tokens_parts else np.empty((0,), dtype=np.int64)
        scores = np.concatenate(score_parts) if score_parts else np.empty((0,), dtype=np.float32)
        order = np.argsort(-scores, kind="stable")
        return tokens[order].astype(np.int64, copy=False), scores[order].astype(np.float32, copy=False), events

    def selection_fullscan_bounded(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, EventBytes]:
        events = EventBytes()
        if not self.pages:
            empty_i = np.empty((0,), dtype=np.int64)
            empty_f = np.empty((0,), dtype=np.float32)
            return empty_i, empty_f, empty_f, events
        q = q.astype(np.float32, copy=False)
        token_parts = []
        score_parts = []
        bound_parts = []
        for page in self.pages:
            codebooks = page["codebooks"]
            codes = page["codes"]
            radii = page["radii"]
            events.read("page_pq_codebooks", self._pq_codebook_bytes_per_page())
            events.read("page_pq_radii", self._pq_radius_bytes_per_page())
            events.read("page_pq_codes", self._pq_code_bytes(int(page["size"])))
            page_q = self._page_query(q, page, events)
            scores = pq_scores(page_q, codebooks, codes)
            bounds = pq_score_error_bounds(page_q, radii, codes)
            tokens = int(page["token_start"]) + np.arange(int(page["size"]), dtype=np.int64)
            token_parts.append(tokens)
            score_parts.append(scores)
            bound_parts.append(bounds)
        tokens = np.concatenate(token_parts)
        scores = np.concatenate(score_parts)
        bounds = np.concatenate(bound_parts)
        order = np.argsort(-(scores + bounds), kind="stable")
        return (
            tokens[order].astype(np.int64, copy=False),
            scores[order].astype(np.float32, copy=False),
            bounds[order].astype(np.float32, copy=False),
            events,
        )

    def selection_fullscan(self, q: np.ndarray) -> tuple[np.ndarray, EventBytes]:
        tokens, _scores, events = self.selection_fullscan_scored(q)
        return tokens, events

    def selection_routed_many_scored(self, q: np.ndarray, nprobes: list[int]) -> dict[int, tuple[np.ndarray, np.ndarray, EventBytes]]:
        out: dict[int, tuple[np.ndarray, np.ndarray, EventBytes]] = {}
        if not self.groups:
            for nprobe in nprobes:
                out[int(nprobe)] = (
                    np.empty((0,), dtype=np.int64),
                    np.empty((0,), dtype=np.float32),
                    EventBytes(),
                )
            return out
        q = q.astype(np.float32, copy=False)
        group_means = np.stack([group["mean"] for group in self.groups], axis=0).astype(np.float32, copy=False)
        group_scores = group_means @ q
        group_order = np.argsort(-group_scores, kind="stable")
        for nprobe in nprobes:
            events = EventBytes()
            probe = max(1, min(int(nprobe), len(self.groups)))
            events.read("router_groups", self._prototype_bytes(len(self.groups)))
            selected_groups = group_order[:probe]
            page_rows: dict[int, set[int]] = {}
            member_refs = 0
            for gid in selected_groups.tolist():
                members = self.groups[int(gid)]["members"]
                member_refs += len(members)
                for page_id, proto_id in members:
                    page = self.pages[int(page_id)]
                    rows = page["proto_rows"][int(proto_id)]
                    if rows.size == 0:
                        continue
                    slot = page_rows.setdefault(int(page_id), set())
                    slot.update(int(x) for x in rows.tolist())
            events.read("router_postings", self._ref_bytes(member_refs))
            token_parts = []
            score_parts = []
            candidate_rows = 0
            for page_id in sorted(page_rows):
                page = self.pages[int(page_id)]
                rows = np.asarray(sorted(page_rows[page_id]), dtype=np.int64)
                if rows.size == 0:
                    continue
                candidate_rows += int(rows.size)
                events.read("page_pq_codebooks", self._pq_codebook_bytes_per_page())
                events.read("page_pq_codes", self._pq_code_bytes(rows.size))
                scores = pq_scores(self._page_query(q, page, events), page["codebooks"], page["codes"][rows])
                tokens = int(page["token_start"]) + rows
                token_parts.append(tokens.astype(np.int64, copy=False))
                score_parts.append(scores.astype(np.float32, copy=False))
            if not token_parts:
                out[int(nprobe)] = (
                    np.empty((0,), dtype=np.int64),
                    np.empty((0,), dtype=np.float32),
                    events,
                )
                continue
            tokens = np.concatenate(token_parts)
            scores = np.concatenate(score_parts)
            order = np.argsort(-scores, kind="stable")
            out[int(nprobe)] = (
                tokens[order].astype(np.int64, copy=False),
                scores[order].astype(np.float32, copy=False),
                events,
            )
        return out

    def selection_routed_many(self, q: np.ndarray, nprobes: list[int]) -> dict[int, tuple[np.ndarray, EventBytes]]:
        scored = self.selection_routed_many_scored(q, nprobes)
        return {nprobe: (tokens, events) for nprobe, (tokens, _scores, events) in scored.items()}


def selected_q_indices(args: argparse.Namespace, positions: np.ndarray, input_len: int) -> np.ndarray:
    q_indices = np.arange(positions.shape[0], dtype=np.int64)
    if str(args.decode_tokens_filter).strip():
        keep = set(sorted_unique_ints(args.decode_tokens_filter))
        decodes = np.asarray([max(0, int(pos) - int(input_len) + 1) for pos in positions], dtype=np.int64)
        q_indices = q_indices[np.asarray([int(x) in keep for x in decodes], dtype=bool)]
    if int(args.num_queries) > 0 and q_indices.shape[0] > int(args.num_queries):
        if args.query_selection == "first":
            q_indices = q_indices[: int(args.num_queries)]
        elif args.query_selection == "even":
            take = np.linspace(0, q_indices.shape[0] - 1, num=int(args.num_queries), dtype=np.int64)
            q_indices = q_indices[take]
        elif args.query_selection == "random":
            rng = np.random.default_rng(int(args.seed))
            q_indices = np.sort(rng.choice(q_indices, size=int(args.num_queries), replace=False))
    return q_indices


def method_row(
    *,
    args: argparse.Namespace,
    method: str,
    policy: str,
    target: float,
    decode_tokens: int,
    position: int,
    qidx: int,
    head: int,
    kv_head: int,
    nprobe: int,
    final_k: int,
    scores: np.ndarray,
    values: np.ndarray,
    probs: np.ndarray,
    dense_out: np.ndarray,
    represented: list[int],
    events: EventBytes,
    selection_events: EventBytes,
    update_events: EventBytes,
    candidate_count: int,
) -> dict:
    represented = unique(represented, len(represented), 0, scores.shape[0])
    mass = float(probs[np.asarray(represented, dtype=np.int64)].sum()) if represented else 0.0
    cos = cosine_output(scores, values, dense_out, represented) if bool(args.compute_output_cos) else float("nan")
    row = {
        "decode_tokens": int(decode_tokens),
        "position": int(position),
        "qidx": int(qidx),
        "head": int(head),
        "kv_head": int(kv_head),
        "policy": str(policy),
        "method": str(method),
        "target_mass": float(target),
        "reached": bool(mass >= float(target)),
        "mass": mass,
        "output_cos": cos,
        "nprobe": int(nprobe),
        "final_k": int(final_k),
        "candidate_tokens": int(candidate_count),
        "exact_tokens": int(len(represented)),
        "estimated_mb": events.mb(),
        "selector_mb": selection_events.mb(),
        "online_update_mb_amortized": update_events.mb(),
    }
    row.update(events.prefixed_flat("event"))
    return row


def pqcache_average_update_events(index: OnlineIVFPQIndex, group_size: int) -> EventBytes:
    out = EventBytes()
    if index.total_update_steps <= 0:
        return out
    scale = 1.0 / float(max(1, int(group_size)))
    out.read("append_key", index._key_bytes(1) * scale)
    out.read("pq_codebooks", index._pq_codebook_bytes() * scale)
    out.write("pq_codes", index._pq_code_bytes(1) * scale)
    return out


def summarize(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    keys = ("decode_tokens", "policy", "method", "target_mass", "nprobe", "final_k")
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    out = []
    metrics = [
        "mass",
        "output_cos",
        "estimated_mb",
        "selector_mb",
        "online_update_mb_amortized",
        "candidate_tokens",
        "exact_tokens",
        "event_read_mb",
        "event_write_mb",
        "event_total_mb",
    ]
    for key, items in sorted(groups.items(), key=lambda x: x[0]):
        row = {name: value for name, value in zip(keys, key)}
        row["samples"] = len(items)
        row["reached_rate"] = float(np.mean([1.0 if item["reached"] else 0.0 for item in items]))
        for metric in metrics:
            vals = [float(item.get(metric, 0.0)) for item in items]
            row[f"{metric}_mean"] = float(np.mean(vals))
        out.append(row)
    return out


class SummaryAccumulator:
    def __init__(self) -> None:
        self.keys = ("decode_tokens", "policy", "method", "target_mass", "nprobe", "final_k")
        self.base_metrics = [
            "mass",
            "output_cos",
            "estimated_mb",
            "selector_mb",
            "online_update_mb_amortized",
            "candidate_tokens",
            "exact_tokens",
            "event_read_mb",
            "event_write_mb",
            "event_total_mb",
        ]
        self.metrics: set[str] = set(self.base_metrics)
        self.groups: dict[tuple, dict] = {}

    def add(self, row: dict) -> None:
        key = tuple(row[k] for k in self.keys)
        item = self.groups.setdefault(
            key,
            {
                "samples": 0,
                "reached": 0.0,
                "sums": {},
            },
        )
        item["samples"] += 1
        item["reached"] += 1.0 if bool(row["reached"]) else 0.0
        row_metrics = set(self.base_metrics)
        row_metrics.update(key for key in row if key.startswith("event_") and key.endswith("_mb"))
        self.metrics.update(row_metrics)
        for metric in row_metrics:
            item["sums"][metric] = float(item["sums"].get(metric, 0.0)) + float(row.get(metric, 0.0))

    def rows(self) -> list[dict]:
        out = []
        for key, item in sorted(self.groups.items(), key=lambda x: x[0]):
            samples = max(1, int(item["samples"]))
            row = {name: value for name, value in zip(self.keys, key)}
            row["samples"] = int(samples)
            row["reached_rate"] = float(item["reached"]) / float(samples)
            for metric in sorted(self.metrics):
                row[f"{metric}_mean"] = float(item["sums"].get(metric, 0.0)) / float(samples)
            out.append(row)
        return out


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def flush_outputs(out_dir: Path, summary_acc: SummaryAccumulator, init_summary: list[dict]) -> None:
    summary_rows = summary_acc.rows()
    write_csv(out_dir / "summary.partial.csv", summary_rows)
    write_csv(out_dir / "init_summary.csv", init_summary)
    (out_dir / "summary.partial.json").write_text(json.dumps(summary_rows, indent=2, sort_keys=True))
    (out_dir / "init_summary.json").write_text(json.dumps(init_summary, indent=2, sort_keys=True))


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
    meta = json.loads(str(data["metadata"].item())) if "metadata" in data else {}
    input_len = int(meta.get("input_len", int(positions.min()) + 1))
    num_heads, _q_count, dim = queries.shape
    kv_heads = keys.shape[0]
    args.head_dim = int(dim)
    group_size = max(1, num_heads // kv_heads)
    score_scale = 1.0 / math.sqrt(float(dim))
    target_masses = parse_float_list(args.mass_targets)
    nprobes = sorted_unique_ints(args.ivfpq_nprobes)
    final_ks = [] if bool(args.skip_fixedk) else sorted_unique_ints(args.ivfpq_final_ks)
    policies = parse_name_list(args.policies)

    dynamic_start = min(max(0, int(args.static_prefix)), input_len)
    init_dynamic_end = max(dynamic_start, input_len - max(0, int(args.static_suffix)))
    q_indices = selected_q_indices(args, positions, input_len)
    q_indices = np.asarray(sorted(q_indices.tolist(), key=lambda i: int(positions[int(i)])), dtype=np.int64)

    all_rows: list[dict] | None = [] if bool(args.emit_samples) else None
    summary_acc = SummaryAccumulator()
    init_summary = []
    start_time = time.monotonic()

    def record(row: dict) -> None:
        summary_acc.add(row)
        if all_rows is not None:
            all_rows.append(row)

    policy_indexes: dict[str, list[OnlineIVFPQIndex | PagedLocalPQIndex]] = {}
    policy_start_times: dict[str, float] = {}
    for policy in policies:
        if policy not in {"frozen_append", "online_centroid", "periodic_rebuild", "paged_local_pq", "paged_merged_pq"}:
            raise ValueError(f"unknown policy: {policy}")
        policy_start_times[policy] = time.monotonic()
        print(f"[online_ivfpq_simulator] policy={policy} init", flush=True)
        if policy in {"paged_local_pq", "paged_merged_pq"}:
            policy_indexes[policy] = [
                PagedLocalPQIndex(
                    keys=keys[kv_h],
                    init_start=dynamic_start,
                    init_end=init_dynamic_end,
                    args=args,
                    seed=int(args.seed) + 2027 * int(kv_h),
                    router_enabled=(policy == "paged_merged_pq"),
                )
                for kv_h in range(kv_heads)
            ]
        else:
            policy_indexes[policy] = [
                OnlineIVFPQIndex(
                    keys=keys[kv_h],
                    init_start=dynamic_start,
                    init_end=init_dynamic_end,
                    policy=policy,
                    args=args,
                    seed=int(args.seed) + 2027 * int(kv_h),
                )
                for kv_h in range(kv_heads)
            ]
        for kv_h, index in enumerate(policy_indexes[policy]):
            init_summary.append(
                {
                    "policy": policy,
                    "kv_head": int(kv_h),
                    "init_tokens": int(index.size),
                    "pending_tokens": int(len(index.pending_tokens())) if isinstance(index, PagedLocalPQIndex) else 0,
                    "page_size": int(index.page_size) if isinstance(index, PagedLocalPQIndex) else 0,
                    "pages": int(len(index.pages)) if isinstance(index, PagedLocalPQIndex) else 0,
                    "router_groups": int(len(index.groups)) if isinstance(index, PagedLocalPQIndex) else 0,
                    "init_mb": index.init_events.mb(),
                    **index.init_events.prefixed_flat("init_event"),
                }
            )

    print(f"[online_ivfpq_simulator] evaluate policies={','.join(policies)} q={len(q_indices)}", flush=True)
    for qpos_i, qidx in enumerate(q_indices.tolist()):
        pos = int(positions[int(qidx)])
        decode_tokens = max(0, pos - input_len + 1)
        indexed_hi = max(dynamic_start, min(pos + 1 - max(0, int(args.static_suffix)), keys.shape[1]))
        for indexes in policy_indexes.values():
            for index in indexes:
                index.advance_to(indexed_hi)

        for head in range(num_heads):
            kv_h = min(kv_heads - 1, int(head * kv_heads // num_heads))
            q = queries[head, int(qidx)].astype(np.float32, copy=False)
            usable_keys = keys[kv_h, : pos + 1].astype(np.float32, copy=False)
            vals = values[kv_h, : pos + 1].astype(np.float32, copy=False) if bool(args.compute_output_cos) else values[kv_h, :1].astype(np.float32, copy=False)
            scores = (usable_keys @ q) * score_scale
            logits = scores - np.max(scores)
            probs = np.exp(logits).astype(np.float32)
            probs /= max(float(probs.sum()), 1e-20)
            dense_out = probs @ vals if bool(args.compute_output_cos) else np.empty((dim,), dtype=np.float32)
            base = static_tokens(pos, int(args.static_prefix), int(args.static_suffix))
            base = unique(base, len(base), 0, scores.shape[0])
            base_set = set(base)

            static_mask = np.zeros((scores.shape[0],), dtype=bool)
            if base:
                static_mask[np.asarray(base, dtype=np.int64)] = True
            dynamic_ids = np.nonzero(~static_mask)[0].astype(np.int64, copy=False)
            oracle_order = dynamic_ids[np.argsort(-probs[dynamic_ids], kind="stable")]
            oracle_by_target: dict[float, list[int]] = {}
            oracle_represented = list(base)
            oracle_mass = float(probs[np.asarray(oracle_represented, dtype=np.int64)].sum()) if oracle_represented else 0.0
            oracle_cursor = 0
            for target in sorted(float(x) for x in target_masses):
                while oracle_mass < float(target) and oracle_cursor < oracle_order.size:
                    tok = int(oracle_order[oracle_cursor])
                    oracle_cursor += 1
                    oracle_represented.append(tok)
                    oracle_mass += float(probs[tok])
                oracle_by_target[float(target)] = list(oracle_represented)

            for policy, indexes in policy_indexes.items():
                index = indexes[kv_h]
                update_events = index.average_update_events_per_query(group_size)
                for target, represented_oracle in oracle_by_target.items():
                    exact_events = EventBytes()
                    exact_events.read(
                        "exact_kv",
                        len(unique(represented_oracle, len(represented_oracle), 0, scores.shape[0]))
                        * dim
                        * (int(args.attn_key_bytes_per_element) + int(args.value_bytes_per_element)),
                    )
                    total_events = EventBytes()
                    total_events.add(exact_events)
                    record(
                        method_row(
                            args=args,
                            method="dense_oracle",
                            policy=policy,
                            target=float(target),
                            decode_tokens=decode_tokens,
                            position=pos,
                            qidx=int(qidx),
                            head=head,
                            kv_head=kv_h,
                            nprobe=0,
                            final_k=len(unique(represented_oracle, len(represented_oracle), 0, scores.shape[0])),
                            scores=scores,
                            values=vals,
                            probs=probs,
                            dense_out=dense_out,
                            represented=list(represented_oracle),
                            events=total_events,
                            selection_events=EventBytes(),
                            update_events=EventBytes(),
                            candidate_count=0,
                        )
                    )

                if isinstance(index, PagedLocalPQIndex):
                    pending = [
                        int(tok)
                        for tok in index.pending_tokens()
                        if int(tok) < scores.shape[0] and int(tok) not in base_set
                    ]
                    pending_set = set(pending)
                    if index.router_enabled:
                        routed = index.selection_routed_many(q, nprobes)
                    else:
                        ranked_tokens, selection_events = index.selection_fullscan(q)
                        routed = {0: (ranked_tokens, selection_events)}
                    for route_probe, (ranked_tokens_raw, selection_events) in routed.items():
                        ranked_tokens = np.asarray(
                            [
                                int(tok)
                                for tok in ranked_tokens_raw.tolist()
                                if int(tok) < scores.shape[0] and int(tok) not in base_set and int(tok) not in pending_set
                            ],
                            dtype=np.int64,
                        )
                        for target in target_masses:
                            represented = list(base) + pending
                            represented = unique(represented, len(represented), 0, scores.shape[0])
                            mass = float(probs[np.asarray(represented, dtype=np.int64)].sum()) if represented else 0.0
                            cursor = 0
                            while mass < float(target) and cursor < ranked_tokens.size:
                                tok = int(ranked_tokens[cursor])
                                cursor += 1
                                represented.append(tok)
                                mass += float(probs[tok])
                            exact_events = EventBytes()
                            exact_events.read(
                                "exact_kv",
                                len(unique(represented, len(represented), 0, scores.shape[0]))
                                * dim
                                * (int(args.attn_key_bytes_per_element) + int(args.value_bytes_per_element)),
                            )
                            total_events = EventBytes()
                            total_events.add(selection_events)
                            total_events.add(exact_events)
                            total_events.add(update_events)
                            record(
                                method_row(
                                    args=args,
                                    method="paged_merged_pq_routed_oracle" if index.router_enabled else "paged_local_pq_full_scan_oracle",
                                    policy=policy,
                                    target=float(target),
                                    decode_tokens=decode_tokens,
                                    position=pos,
                                    qidx=int(qidx),
                                    head=head,
                                    kv_head=kv_h,
                                    nprobe=int(route_probe),
                                    final_k=max(0, len(represented) - len(base) - len(pending)),
                                    scores=scores,
                                    values=vals,
                                    probs=probs,
                                    dense_out=dense_out,
                                    represented=represented,
                                    events=total_events,
                                    selection_events=selection_events,
                                    update_events=update_events,
                                    candidate_count=int(ranked_tokens.size),
                                )
                            )
                    continue

                pq_update_events = pqcache_average_update_events(index, group_size)
                pq_events = EventBytes()
                pq_events.read("pq_codebooks", index._pq_codebook_bytes())
                pq_events.read("pq_codes", index._pq_code_bytes(index.size))
                pq_rank = index.token_start + np.argsort(-pq_scores(q, index.codebooks, index.codes[: index.size]), kind="stable")
                pq_rank = np.asarray([int(tok) for tok in pq_rank.tolist() if int(tok) < scores.shape[0] and int(tok) not in base_set], dtype=np.int64)
                for target in target_masses:
                    represented = list(base)
                    mass = float(probs[np.asarray(represented, dtype=np.int64)].sum()) if represented else 0.0
                    cursor = 0
                    while mass < float(target) and cursor < pq_rank.size:
                        tok = int(pq_rank[cursor])
                        cursor += 1
                        represented.append(tok)
                        mass += float(probs[tok])
                    exact_events = EventBytes()
                    exact_events.read(
                        "exact_kv",
                        len(unique(represented, len(represented), 0, scores.shape[0]))
                        * dim
                        * (int(args.attn_key_bytes_per_element) + int(args.value_bytes_per_element)),
                    )
                    total_events = EventBytes()
                    total_events.add(pq_events)
                    total_events.add(exact_events)
                    total_events.add(pq_update_events)
                    record(
                        method_row(
                            args=args,
                            method="pqcache_full_scan_oracle",
                            policy=policy,
                            target=float(target),
                            decode_tokens=decode_tokens,
                            position=pos,
                            qidx=int(qidx),
                            head=head,
                            kv_head=kv_h,
                            nprobe=0,
                            final_k=max(0, len(represented) - len(base)),
                            scores=scores,
                            values=vals,
                            probs=probs,
                            dense_out=dense_out,
                            represented=represented,
                            events=total_events,
                            selection_events=pq_events,
                            update_events=pq_update_events,
                            candidate_count=int(index.size),
                        )
                    )

                selections = index.selection_many(q, nprobes)
                for nprobe in nprobes:
                    ranked_tokens, selection_events = selections[int(nprobe)]
                    ranked_tokens = np.asarray(
                        [int(tok) for tok in ranked_tokens.tolist() if int(tok) < scores.shape[0] and int(tok) not in base_set],
                        dtype=np.int64,
                    )
                    for target in target_masses:
                        represented = list(base)
                        mass = float(probs[np.asarray(represented, dtype=np.int64)].sum()) if represented else 0.0
                        cursor = 0
                        while mass < float(target) and cursor < ranked_tokens.size:
                            tok = int(ranked_tokens[cursor])
                            cursor += 1
                            represented.append(tok)
                            mass += float(probs[tok])
                        exact_events = EventBytes()
                        exact_events.read(
                            "exact_kv",
                            len(unique(represented, len(represented), 0, scores.shape[0]))
                            * dim
                            * (int(args.attn_key_bytes_per_element) + int(args.value_bytes_per_element)),
                        )
                        total_events = EventBytes()
                        total_events.add(selection_events)
                        total_events.add(exact_events)
                        total_events.add(update_events)
                        record(
                            method_row(
                                args=args,
                                method="ivfpq_online_oracle",
                                policy=policy,
                                target=float(target),
                                decode_tokens=decode_tokens,
                                position=pos,
                                qidx=int(qidx),
                                head=head,
                                kv_head=kv_h,
                                nprobe=int(nprobe),
                                final_k=max(0, len(represented) - len(base)),
                                scores=scores,
                                values=vals,
                                probs=probs,
                                dense_out=dense_out,
                                represented=represented,
                                events=total_events,
                                selection_events=selection_events,
                                update_events=update_events,
                                candidate_count=int(ranked_tokens.size),
                            )
                        )
                    for final_k in final_ks:
                        selected = ranked_tokens[: min(int(final_k), ranked_tokens.size)].tolist()
                        represented = unique(list(base) + selected, len(base) + len(selected), 0, scores.shape[0])
                        exact_events = EventBytes()
                        exact_events.read(
                            "exact_kv",
                            len(represented)
                            * dim
                            * (int(args.attn_key_bytes_per_element) + int(args.value_bytes_per_element)),
                        )
                        total_events = EventBytes()
                        total_events.add(selection_events)
                        total_events.add(exact_events)
                        total_events.add(update_events)
                        for target in target_masses:
                            record(
                                method_row(
                                    args=args,
                                    method="ivfpq_online_fixedk",
                                    policy=policy,
                                    target=float(target),
                                    decode_tokens=decode_tokens,
                                    position=pos,
                                    qidx=int(qidx),
                                    head=head,
                                    kv_head=kv_h,
                                    nprobe=int(nprobe),
                                    final_k=int(final_k),
                                    scores=scores,
                                    values=vals,
                                    probs=probs,
                                    dense_out=dense_out,
                                    represented=represented,
                                    events=total_events,
                                    selection_events=selection_events,
                                    update_events=update_events,
                                    candidate_count=int(ranked_tokens.size),
                                )
                            )
        if int(args.progress_every) > 0 and (qpos_i + 1) % int(args.progress_every) == 0:
            elapsed = time.monotonic() - start_time
            print(
                f"[online_ivfpq_simulator] q={qpos_i + 1}/{len(q_indices)} decode={decode_tokens} elapsed={elapsed:.1f}s",
                flush=True,
            )
            flush_outputs(out_dir, summary_acc, init_summary)

    for policy in policies:
        print(
            f"[online_ivfpq_simulator] policy={policy} done elapsed={time.monotonic() - policy_start_times[policy]:.1f}s",
            flush=True,
        )
        flush_outputs(out_dir, summary_acc, init_summary)

    summary_rows = summary_acc.rows()
    if all_rows is not None:
        write_csv(out_dir / "samples.csv", all_rows)
        (out_dir / "samples.json").write_text(json.dumps(all_rows, indent=2, sort_keys=True))
    write_csv(out_dir / "summary.csv", summary_rows)
    write_csv(out_dir / "init_summary.csv", init_summary)
    (out_dir / "summary.json").write_text(json.dumps(summary_rows, indent=2, sort_keys=True))
    (out_dir / "init_summary.json").write_text(json.dumps(init_summary, indent=2, sort_keys=True))
    print(f"[online_ivfpq_simulator] wrote {out_dir}")
    sample_count = sum(int(row["samples"]) for row in summary_rows)
    print(f"[online_ivfpq_simulator] samples={sample_count} summary={len(summary_rows)}")


if __name__ == "__main__":
    main()
