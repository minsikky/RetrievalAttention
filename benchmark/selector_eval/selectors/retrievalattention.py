from __future__ import annotations

import heapq
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from benchmark.selector_eval.costs.base import CostTrace
from benchmark.selector_eval.data.trace import QKVTrace, unique_tokens
from benchmark.selector_eval.selectors.base import QueryState, SelectionResult


@dataclass
class RetrievalAttentionGraphSelector:
    """Causal Q-K provenance graph traversal baseline.

    The replay graph is built incrementally from saved trace queries: each prior
    query contributes co-occurrence edges among its top Q-K tokens. Query-time
    selection then traverses this token graph and reranks visited tokens by the
    current exact Q-K score.
    """

    trace: QKVTrace
    static_prefix: int = 128
    static_suffix: int = 128
    provenance_topk: int = 64
    connect_window: int = 8
    degree: int = 32
    seed_count: int = 64
    max_visits: int = 2048
    min_visits: int = 64
    score_key_bytes: int = 2
    offset_bytes: int = 4
    edge_index_bytes: int = 4
    _graphs: dict[tuple[int, int], list[Counter]] = field(default_factory=dict, init=False)
    _cursor: dict[tuple[int, int], int] = field(default_factory=dict, init=False)

    name = "retrievalattention_graph"

    def _graph_for(self, head: int, kv_head: int) -> list[Counter]:
        key = (int(head), int(kv_head))
        if key not in self._graphs:
            self._graphs[key] = [Counter() for _ in range(self.trace.keys.shape[1])]
            self._cursor[key] = 0
        return self._graphs[key]

    def _add_edge(self, graph: list[Counter], a: int, b: int) -> None:
        if a == b or a < 0 or b < 0 or a >= len(graph) or b >= len(graph):
            return
        graph[int(a)][int(b)] += 1
        graph[int(b)][int(a)] += 1

    def _extend_graph(self, state: QueryState) -> None:
        key = (int(state.head), int(state.kv_head))
        graph = self._graph_for(state.head, state.kv_head)
        start_qidx = int(self._cursor[key])
        stop_qidx = min(int(state.qidx) + 1, int(self.trace.positions.shape[0]))
        if stop_qidx <= start_qidx:
            return
        keys = self.trace.keys[int(state.kv_head)]
        for qidx in range(start_qidx, stop_qidx):
            pos = int(self.trace.positions[qidx])
            dynamic_start = min(max(0, int(self.static_prefix)), pos + 1)
            dynamic_end = max(dynamic_start, pos + 1 - max(0, int(self.static_suffix)))
            if dynamic_end <= dynamic_start:
                continue
            q = self.trace.queries[int(state.head), qidx].astype(np.float32, copy=False)
            token_ids = np.arange(dynamic_start, dynamic_end, dtype=np.int64)
            scores = keys[token_ids].astype(np.float32, copy=False) @ q
            take = min(int(self.provenance_topk), int(token_ids.size))
            if take <= 1:
                continue
            part = np.argpartition(-scores, kth=take - 1)[:take]
            ordered = token_ids[part[np.argsort(-scores[part], kind="stable")]]
            width = max(1, int(self.connect_window))
            for i, src in enumerate(ordered.tolist()):
                for dst in ordered[i + 1 : i + 1 + width].tolist():
                    self._add_edge(graph, int(src), int(dst))
        self._cursor[key] = stop_qidx

    def _neighbors(self, graph: list[Counter], tok: int, position: int) -> list[int]:
        if tok < 0 or tok >= len(graph):
            return []
        out = []
        for nb, _count in graph[int(tok)].most_common(max(1, int(self.degree))):
            nb = int(nb)
            if nb <= int(position):
                out.append(nb)
        return out

    def _seeds(self, state: QueryState, graph: list[Counter], selected_set: set[int]) -> list[int]:
        context_len = int(state.scores.shape[0])
        tail_start = max(0, context_len - max(1, int(self.static_suffix)) - max(1, int(self.seed_count)))
        tail_end = max(0, context_len - max(0, int(self.static_suffix)))
        seeds = [tok for tok in range(tail_start, tail_end) if tok not in selected_set]
        if len(seeds) >= int(self.seed_count):
            return seeds[-int(self.seed_count) :]

        dynamic_start = min(max(0, int(self.static_prefix)), context_len)
        dynamic_end = max(dynamic_start, context_len - max(0, int(self.static_suffix)))
        degrees = []
        for tok in range(dynamic_start, dynamic_end):
            if tok not in selected_set and graph[tok]:
                degrees.append((len(graph[tok]), tok))
        degrees.sort(reverse=True)
        for _deg, tok in degrees:
            seeds.append(int(tok))
            if len(seeds) >= int(self.seed_count):
                break
        return unique_tokens(seeds, context_len=context_len)

    def _rank_to_target(self, state: QueryState, base: list[int], candidates: list[int], target: float, budget: int | None) -> list[int]:
        selected = unique_tokens(list(base), context_len=state.scores.shape[0])
        selected_set = set(selected)
        mass = float(state.probs[np.asarray(selected, dtype=np.int64)].sum()) if selected else 0.0
        cand = np.asarray([tok for tok in candidates if tok not in selected_set], dtype=np.int64)
        if cand.size:
            ranked = cand[np.argsort(-state.scores[cand], kind="stable")]
            for tok in ranked.tolist():
                selected.append(int(tok))
                selected_set.add(int(tok))
                mass += float(state.probs[int(tok)])
                if mass >= target:
                    break
                if budget is not None and len(selected) >= int(budget):
                    break
        return unique_tokens(selected, context_len=state.scores.shape[0])

    def select(self, state: QueryState, *, target_mass: float | None = None, budget: int | None = None) -> SelectionResult:
        target = 1.0 if target_mass is None else float(target_mass)
        self._extend_graph(state)
        graph = self._graph_for(state.head, state.kv_head)
        cost = CostTrace()
        base = unique_tokens(list(state.base_tokens), context_len=state.scores.shape[0])
        base_set = set(base)
        seeds = self._seeds(state, graph, base_set)

        heap: list[tuple[float, int]] = []
        pushed: set[int] = set()
        visited: list[int] = []
        expanded = 0
        edge_reads = 0

        def push(tok: int) -> None:
            if tok in pushed or tok in base_set or tok < 0 or tok >= state.scores.shape[0]:
                return
            pushed.add(int(tok))
            cost.read("selector", "ra_score_keys", state.query.shape[0] * int(self.score_key_bytes))
            heapq.heappush(heap, (-float(state.scores[int(tok)]), int(tok)))

        for tok in seeds:
            push(int(tok))

        while heap and len(visited) < max(1, int(self.max_visits)):
            _neg_score, tok = heapq.heappop(heap)
            if tok in visited:
                continue
            visited.append(int(tok))
            nbs = self._neighbors(graph, int(tok), int(state.position))
            expanded += 1
            edge_reads += len(nbs)
            cost.read("selector", "ra_graph_offsets", int(self.offset_bytes))
            cost.read("selector", "ra_graph_edges", len(nbs) * int(self.edge_index_bytes))
            for nb in nbs:
                push(int(nb))
            if len(visited) >= max(0, int(self.min_visits)):
                selected_now = self._rank_to_target(state, base, visited, target, budget)
                mass = float(state.probs[np.asarray(selected_now, dtype=np.int64)].sum()) if selected_now else 0.0
                if mass >= target:
                    break

        ranked_candidates = sorted(unique_tokens(visited, context_len=state.scores.shape[0]), key=lambda tok: -float(state.scores[tok]))
        selected = self._rank_to_target(state, base, ranked_candidates, target, budget)
        return SelectionResult(
            algorithm=self.name,
            selected_tokens=selected,
            candidate_tokens=ranked_candidates,
            cost=cost,
            metadata={
                "target_mass": target_mass,
                "budget": budget,
                "provenance_topk": int(self.provenance_topk),
                "degree": int(self.degree),
                "seed_count": int(self.seed_count),
                "max_visits": int(self.max_visits),
                "visited": int(len(visited)),
                "expanded": int(expanded),
                "edge_reads": int(edge_reads),
                "graph_cursor": int(self._cursor.get((int(state.head), int(state.kv_head)), 0)),
            },
        )
