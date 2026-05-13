from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from benchmark.selector_eval.costs.base import CostTrace
from benchmark.selector_eval.data.trace import unique_tokens
from benchmark.selector_eval.selectors.base import QueryState, SelectionResult


@dataclass
class SparQSelector:
    """SparQ-style query-channel selector proxy.

    Scores every dynamic token using only the largest-magnitude query channels,
    then consumes the approximate ranking until the requested mass is reached.
    """

    rank: int = 16
    score_key_bytes: int = 4
    index_bytes: int = 4

    @property
    def name(self) -> str:
        return f"sparq_r{int(self.rank)}"

    def select(self, state: QueryState, *, target_mass: float | None = None, budget: int | None = None) -> SelectionResult:
        target = 1.0 if target_mass is None else float(target_mass)
        selected = unique_tokens(list(state.base_tokens), context_len=state.scores.shape[0])
        selected_set = set(selected)
        dynamic = [tok for tok in range(state.scores.shape[0]) if tok not in selected_set]
        cost = CostTrace()
        if not dynamic:
            return SelectionResult(self.name, selected_tokens=selected, candidate_tokens=[], cost=cost)

        rank = min(max(1, int(self.rank)), state.query.shape[0])
        dims = np.argsort(-np.abs(state.query), kind="stable")[:rank]
        dyn_arr = np.asarray(dynamic, dtype=np.int64)
        q_abs_sum = max(float(np.abs(state.query).sum()), 1e-20)
        coverage = max(float(np.abs(state.query[dims]).sum() / q_abs_sum), 1e-6)
        scale = 1.0 / np.sqrt(float(state.query.shape[0]) * coverage)
        cost.read("selector", "sparq_dims", int(rank) * int(self.index_bytes))
        cost.read("selector", "sparq_key_channels", len(dynamic) * int(rank) * int(self.score_key_bytes))
        approx = (state.keys[dyn_arr[:, None], dims] @ state.query[dims]).astype(np.float32) * scale
        order = np.argsort(-approx, kind="stable")
        ranked = dyn_arr[order].astype(np.int64, copy=False)

        mass = float(state.probs[np.asarray(selected, dtype=np.int64)].sum()) if selected else 0.0
        cursor = 0
        while mass < target and cursor < ranked.size:
            tok = int(ranked[cursor])
            cursor += 1
            selected.append(tok)
            mass += float(state.probs[tok])
            if budget is not None and len(selected) >= int(budget):
                break

        return SelectionResult(
            algorithm=self.name,
            selected_tokens=unique_tokens(selected, context_len=state.scores.shape[0]),
            candidate_tokens=ranked.tolist(),
            cost=cost,
            metadata={"target_mass": target_mass, "budget": budget, "rank": int(rank), "coverage": float(coverage)},
        )

