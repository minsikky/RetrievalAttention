from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from benchmark.online_ivfpq_simulator import EventBytes, OnlineIVFPQIndex
from benchmark.selector_eval.costs.base import CostTrace, kv_read_bytes
from benchmark.selector_eval.data.trace import QKVTrace, unique_tokens
from benchmark.selector_eval.selectors.base import QueryState, SelectionResult
from benchmark.selector_eval.selectors.paged_pq import _event_bytes_to_cost
from benchmark.selector_eval.selectors.paged_pq import _event_bytes_mb


@dataclass
class IVFPQSelector:
    """Adapter for global online IVF-PQ policies from ``online_ivfpq_simulator.py``."""

    trace: QKVTrace
    policy: str = "frozen_append"
    static_prefix: int = 128
    static_suffix: int = 128
    nprobes: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)
    coarse_clusters: int = 128
    coarse_iters: int = 3
    rebuild_interval: int = 8192
    subvecs: int = 2
    subbits: int = 6
    kmeans_iters: int = 3
    seed: int = 2025
    score_key_bytes: int = 4
    attn_key_bytes: int = 2
    value_bytes: int = 2
    edge_index_bytes: int = 4
    graph_offset_bytes: int = 4
    backend: str = "python"
    backend_threads: int = 0

    def __post_init__(self) -> None:
        if self.policy not in {"frozen_append", "online_centroid", "periodic_rebuild"}:
            raise ValueError(f"unknown IVF-PQ policy: {self.policy}")
        self.name = f"ivfpq_{self.policy}"
        self.dynamic_start = min(max(0, int(self.static_prefix)), int(self.trace.input_len))
        self.init_dynamic_end = max(self.dynamic_start, int(self.trace.input_len) - max(0, int(self.static_suffix)))
        self.args = argparse.Namespace(
            static_prefix=int(self.static_prefix),
            static_suffix=int(self.static_suffix),
            ivfpq_coarse_clusters=int(self.coarse_clusters),
            ivfpq_coarse_iters=int(self.coarse_iters),
            ivfpq_rebuild_interval=int(self.rebuild_interval),
            pqcache_subvecs=int(self.subvecs),
            pqcache_subbits=int(self.subbits),
            pqcache_kmeans_iters=int(self.kmeans_iters),
            score_key_bytes_per_element=int(self.score_key_bytes),
            attn_key_bytes_per_element=int(self.attn_key_bytes),
            value_bytes_per_element=int(self.value_bytes),
            edge_index_bytes=int(self.edge_index_bytes),
            graph_offset_bytes=int(self.graph_offset_bytes),
            head_dim=int(self.trace.head_dim),
            backend=str(self.backend),
            backend_threads=int(self.backend_threads),
        )
        self.indexes = [
            OnlineIVFPQIndex(
                keys=self.trace.keys[kv_h],
                init_start=self.dynamic_start,
                init_end=self.init_dynamic_end,
                policy=self.policy,
                args=self.args,
                seed=int(self.seed) + 2027 * int(kv_h),
            )
            for kv_h in range(self.trace.kv_heads)
        ]
        for index in self.indexes:
            index.update_events_total = EventBytes()
            index.total_update_steps = 0

    def _advance(self, state: QueryState) -> CostTrace:
        indexed_hi = max(
            self.dynamic_start,
            min(int(state.position) + 1 - max(0, int(self.static_suffix)), self.trace.keys.shape[1]),
        )
        index = self.indexes[state.kv_head]
        before_reads = dict(index.update_events_total.reads)
        before_writes = dict(index.update_events_total.writes)
        index.advance_to(indexed_hi)
        delta = EventBytes()
        for category, value in index.update_events_total.reads.items():
            diff = float(value) - float(before_reads.get(category, 0.0))
            if diff:
                delta.read(category, diff)
        for category, value in index.update_events_total.writes.items():
            diff = float(value) - float(before_writes.get(category, 0.0))
            if diff:
                delta.write(category, diff)
        return _event_bytes_to_cost(delta, phase="online_update")

    def select(self, state: QueryState, *, target_mass: float | None = None, budget: int | None = None) -> SelectionResult:
        target = 1.0 if target_mass is None else float(target_mass)
        index = self.indexes[state.kv_head]
        update_cost = self._advance(state)
        base = unique_tokens(list(state.base_tokens), context_len=state.scores.shape[0])
        base_set = set(base)

        selections = index.selection_many(state.query, list(self.nprobes))
        choices = []
        for nprobe, (raw_ranked, selection_events) in selections.items():
            ranked = [
                int(tok)
                for tok in raw_ranked.tolist()
                if int(tok) < state.scores.shape[0] and int(tok) not in base_set
            ]
            selected = list(base)
            mass = float(state.probs[np.asarray(selected, dtype=np.int64)].sum()) if selected else 0.0
            cursor = 0
            while mass < target and cursor < len(ranked):
                tok = int(ranked[cursor])
                cursor += 1
                selected.append(tok)
                mass += float(state.probs[tok])
                if budget is not None and len(selected) >= int(budget):
                    break
            selected = unique_tokens(selected, context_len=state.scores.shape[0])
            selector_cost = _event_bytes_to_cost(selection_events, phase="selector")
            exact_mb = kv_read_bytes(len(selected), self.trace.head_dim, self.attn_key_bytes, self.value_bytes) / (
                1024.0 * 1024.0
            )
            total_mb = selector_cost.mb() + update_cost.mb() + exact_mb
            choices.append(
                {
                    "reached": mass >= target,
                    "total_mb": total_mb,
                    "nprobe": int(nprobe),
                    "selected": selected,
                    "ranked": ranked,
                    "selector_cost": selector_cost,
                    "mass": mass,
                }
            )

        reachable = [choice for choice in choices if choice["reached"]]
        choice = min(reachable, key=lambda item: item["total_mb"]) if reachable else max(choices, key=lambda item: item["mass"])
        cost = CostTrace()
        cost.extend(update_cost)
        cost.extend(choice["selector_cost"])
        return SelectionResult(
            algorithm=self.name,
            selected_tokens=choice["selected"],
            candidate_tokens=choice["ranked"],
            cost=cost,
            metadata={
                "target_mass": target_mass,
                "budget": budget,
                "nprobe": int(choice["nprobe"]),
                "policy": self.policy,
                "index_size": int(index.size),
                "accounting_mode": "online_proxy",
                "online_update_modeled": True,
                "online_update_cumulative_MB": _event_bytes_mb(index.update_events_total),
                "online_update_indexed_tokens": int(index.total_update_steps),
            },
        )
