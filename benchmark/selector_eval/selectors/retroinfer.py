from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from benchmark.selector_eval.costs.base import CostTrace
from benchmark.selector_eval.data.trace import unique_tokens
from benchmark.selector_eval.selectors.base import QueryState, SelectionResult


def build_contiguous_centroids(keys: np.ndarray, *, start: int, end: int, cluster_size: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    ranges: list[tuple[int, int]] = []
    centroids = []
    for lo in range(int(start), int(end), int(cluster_size)):
        hi = min(int(end), lo + int(cluster_size))
        if hi <= lo:
            continue
        block = keys[lo:hi].astype(np.float32, copy=False)
        centroid = block.mean(axis=0)
        norm = float(np.linalg.norm(centroid))
        if norm > 0.0:
            centroid = centroid / norm
        centroids.append(centroid.astype(np.float32, copy=False))
        ranges.append((int(lo), int(hi)))
    if not centroids:
        return np.empty((0, keys.shape[-1]), dtype=np.float32), []
    return np.stack(centroids, axis=0), ranges


@dataclass
class RetroInferStyleSelector:
    """RetroInfer-style contiguous chunk centroid router.

    This models the selector/routing side only: score all maintained cluster
    centroids, then read exact K/V for member tokens from the best clusters.
    Initial cluster construction is treated as pre-existing index state, matching
    the selector-eval convention that prefill/index build is not included in
    per-query total_MB.
    """

    cluster_size: int = 256
    static_prefix: int = 128
    static_suffix: int = 128
    score_key_bytes: int = 2
    attn_key_bytes: int = 2
    value_bytes: int = 2
    edge_index_bytes: int = 4
    range_bytes: int = 8
    input_len: int | None = None
    accounting_mode: str = "snapshot"
    _indexed_end_by_kv_head: dict[int, int] = field(default_factory=dict, init=False)
    _online_update_bytes_by_kv_head: dict[int, float] = field(default_factory=dict, init=False)
    _online_update_tokens_by_kv_head: dict[int, int] = field(default_factory=dict, init=False)

    name = "retroinfer_style"

    def __post_init__(self) -> None:
        if self.accounting_mode not in {"snapshot", "online_proxy"}:
            raise ValueError(f"unknown accounting_mode: {self.accounting_mode}")
        if self.accounting_mode == "online_proxy":
            self.name = "retroinfer_online_proxy"

    def _initial_indexed_end(self) -> int:
        if self.input_len is None:
            return 0
        prefix = min(max(0, int(self.static_prefix)), int(self.input_len))
        suffix_excluded = int(self.input_len) - max(0, int(self.static_suffix))
        return max(prefix, suffix_excluded)

    def _cluster_bounds(self, state: QueryState) -> tuple[int, int]:
        context_len = int(state.scores.shape[0])
        start = min(max(0, int(self.static_prefix)), context_len)
        end = max(start, context_len - max(0, int(self.static_suffix)))
        return start, end

    def select(self, state: QueryState, *, target_mass: float | None = None, budget: int | None = None) -> SelectionResult:
        target = 1.0 if target_mass is None else float(target_mass)
        selected = unique_tokens(list(state.base_tokens), context_len=state.scores.shape[0])
        selected_set = set(selected)
        mass = float(state.probs[np.asarray(selected, dtype=np.int64)].sum()) if selected else 0.0
        cost = CostTrace()

        start, end = self._cluster_bounds(state)
        if self.accounting_mode == "online_proxy":
            prev = self._indexed_end_by_kv_head.get(int(state.kv_head), self._initial_indexed_end())
            prev = max(int(start), min(int(prev), int(end)))
            if end > prev:
                count = int(end) - int(prev)
                dim = int(state.keys.shape[-1])
                chunks = int(np.ceil(count / max(1, int(self.cluster_size))))
                cost.read("online_update", "retro_segment_keys", count * dim * int(self.attn_key_bytes))
                cost.write("online_update", "retro_centroids", chunks * dim * int(self.score_key_bytes))
                cost.write("online_update", "retro_cluster_ranges", chunks * int(self.range_bytes))
                cost.write("online_update", "retro_cluster_postings", count * int(self.edge_index_bytes))
                cost.write("online_update", "retro_value_sums", chunks * dim * int(self.value_bytes))
                self._indexed_end_by_kv_head[int(state.kv_head)] = int(end)
                self._online_update_bytes_by_kv_head[int(state.kv_head)] = (
                    self._online_update_bytes_by_kv_head.get(int(state.kv_head), 0.0)
                    + cost.bytes(phase="online_update")
                )
                self._online_update_tokens_by_kv_head[int(state.kv_head)] = (
                    self._online_update_tokens_by_kv_head.get(int(state.kv_head), 0) + count
                )
        centroids, ranges = build_contiguous_centroids(
            state.keys,
            start=start,
            end=end,
            cluster_size=max(1, int(self.cluster_size)),
        )
        if centroids.shape[0] == 0:
            return SelectionResult(
                algorithm=self.name,
                selected_tokens=selected,
                candidate_tokens=[],
                cost=cost,
                metadata={
                    "target_mass": target_mass,
                    "budget": budget,
                    "cluster_size": int(self.cluster_size),
                    "accounting_mode": str(self.accounting_mode),
                    "online_update_modeled": bool(self.accounting_mode == "online_proxy"),
                    "online_update_cumulative_MB": self._online_update_bytes_by_kv_head.get(int(state.kv_head), 0.0)
                    / (1024.0 * 1024.0),
                    "online_update_indexed_tokens": int(
                        self._online_update_tokens_by_kv_head.get(int(state.kv_head), 0)
                    ),
                },
            )

        cost.read("selector", "retro_centroids", centroids.shape[0] * state.query.shape[0] * int(self.score_key_bytes))
        cost.read("selector", "retro_cluster_ranges", centroids.shape[0] * int(self.range_bytes))
        cluster_scores = centroids.astype(np.float32, copy=False) @ state.query.astype(np.float32, copy=False)
        order = np.argsort(-cluster_scores, kind="stable")

        candidates: list[int] = []
        selected_clusters = 0
        for cid in order.tolist():
            lo, hi = ranges[int(cid)]
            toks = [tok for tok in range(int(lo), min(int(hi), state.scores.shape[0])) if tok not in selected_set]
            if not toks:
                continue
            selected_clusters += 1
            candidates.extend(toks)
            for tok in toks:
                if tok in selected_set:
                    continue
                selected.append(tok)
                selected_set.add(tok)
                mass += float(state.probs[tok])
                if budget is not None and len(selected) >= int(budget):
                    break
            if mass >= target:
                break
            if budget is not None and len(selected) >= int(budget):
                break

        return SelectionResult(
            algorithm=self.name,
            selected_tokens=unique_tokens(selected, context_len=state.scores.shape[0]),
            candidate_tokens=unique_tokens(candidates, context_len=state.scores.shape[0]),
            cost=cost,
            metadata={
                "target_mass": target_mass,
                "budget": budget,
                "cluster_size": int(self.cluster_size),
                "clusters_scored": int(centroids.shape[0]),
                "clusters_selected": int(selected_clusters),
                "cluster_start": int(start),
                "cluster_end": int(end),
                "accounting_mode": str(self.accounting_mode),
                "online_update_modeled": bool(self.accounting_mode == "online_proxy"),
                "online_update_cumulative_MB": self._online_update_bytes_by_kv_head.get(int(state.kv_head), 0.0)
                / (1024.0 * 1024.0),
                "online_update_indexed_tokens": int(self._online_update_tokens_by_kv_head.get(int(state.kv_head), 0)),
            },
        )
