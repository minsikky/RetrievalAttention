from __future__ import annotations

import zlib
from dataclasses import dataclass

import numpy as np

from benchmark.selector_eval.costs.base import CostTrace
from benchmark.selector_eval.data.trace import unique_tokens
from benchmark.selector_eval.selectors.base import QueryState, SelectionResult


def simhash_codes(vectors: np.ndarray, projection: np.ndarray, bits: int, tables: int) -> np.ndarray:
    raw = (vectors.astype(np.float32, copy=False) @ projection.astype(np.float32, copy=False)) > 0
    raw = raw.reshape(vectors.shape[0], int(tables), int(bits))
    weights = (1 << np.arange(int(bits), dtype=np.int64)).reshape(1, 1, -1)
    return np.sum(raw.astype(np.int64) * weights, axis=-1).astype(np.int32, copy=False)


@dataclass
class MagicPIGSelector:
    bits: int = 10
    tables: int = 150
    min_collisions: int = 2
    seed: int = 2025
    score_key_bytes: int = 4
    hash_code_bytes: int = 4
    offset_bytes: int = 4
    edge_index_bytes: int = 4

    @property
    def name(self) -> str:
        return f"magicpig_k{int(self.bits)}_l{int(self.tables)}"

    def _projection(self, dim: int) -> np.ndarray:
        key = ("selector_eval_magicpig", int(self.seed), int(dim), int(self.bits), int(self.tables))
        rng = np.random.default_rng(zlib.crc32(repr(key).encode("utf-8")))
        return rng.standard_normal((int(dim), int(self.bits) * int(self.tables))).astype(np.float32)

    def select(self, state: QueryState, *, target_mass: float | None = None, budget: int | None = None) -> SelectionResult:
        target = 1.0 if target_mass is None else float(target_mass)
        selected = unique_tokens(list(state.base_tokens), context_len=state.scores.shape[0])
        selected_set = set(selected)
        dynamic = [tok for tok in range(state.scores.shape[0]) if tok not in selected_set]
        cost = CostTrace()
        if not dynamic:
            return SelectionResult(self.name, selected_tokens=selected, candidate_tokens=[], cost=cost)

        dyn_arr = np.asarray(dynamic, dtype=np.int64)
        centered = state.keys[dyn_arr].astype(np.float32, copy=False)
        centered = centered - centered.mean(axis=0, keepdims=True)
        q_norm = state.query.astype(np.float32, copy=False) / max(float(np.linalg.norm(state.query)), 1e-20)
        proj = self._projection(state.query.shape[0])

        cost.read("selector", "hash_projection", proj.size * int(self.score_key_bytes))
        cost.read("selector", "hash_codes", len(dynamic) * int(self.tables) * int(self.hash_code_bytes))
        codes = simhash_codes(centered, proj, int(self.bits), int(self.tables))
        q_codes = simhash_codes(q_norm.reshape(1, -1), proj, int(self.bits), int(self.tables))[0]
        collisions = (codes == q_codes.reshape(1, -1)).sum(axis=1)
        mask = collisions >= int(self.min_collisions)
        candidates = dyn_arr[mask]
        table_hits = int(collisions[mask].sum()) if candidates.size else 0
        cost.read("selector", "hash_offsets", int(self.tables) * int(self.offset_bytes))
        cost.read("selector", "hash_postings", table_hits * int(self.edge_index_bytes))

        if candidates.size:
            cost.read("selector", "candidate_score_keys", candidates.size * state.query.shape[0] * int(self.score_key_bytes))
            order = np.argsort(-state.scores[candidates], kind="stable")
            ranked = candidates[order].astype(np.int64, copy=False)
        else:
            ranked = np.empty((0,), dtype=np.int64)

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
            metadata={
                "target_mass": target_mass,
                "budget": budget,
                "bits": int(self.bits),
                "tables": int(self.tables),
                "min_collisions": int(self.min_collisions),
                "table_hits": int(table_hits),
            },
        )
