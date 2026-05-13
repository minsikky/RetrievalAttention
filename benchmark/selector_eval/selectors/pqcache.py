from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from benchmark.attention_efficiency_threeway_eval import build_pq_index
from benchmark.selector_eval.costs.base import CostTrace
from benchmark.selector_eval.data.trace import unique_tokens
from benchmark.selector_eval.selectors.base import QueryState, SelectionResult


def pq_code_bytes(subbits: int) -> int:
    return 1 if int(subbits) <= 8 else 2


def pq_scores(query: np.ndarray, codebooks: np.ndarray, codes: np.ndarray) -> np.ndarray:
    if codes.shape[0] == 0:
        return np.empty((0,), dtype=np.float32)
    query = query.astype(np.float32, copy=False)
    subvecs = int(codebooks.shape[0])
    subdim = query.shape[0] // subvecs
    q_parts = query.reshape(subvecs, subdim)
    table = np.einsum("ms,mcs->mc", q_parts, codebooks.astype(np.float32, copy=False), optimize=True)
    approx = np.zeros((codes.shape[0],), dtype=np.float32)
    for sub in range(subvecs):
        approx += table[sub, codes[:, sub]]
    return approx


@dataclass
class PQCacheFullScanSelector:
    """PQ selector that scans all PQ codes in the current dynamic context.

    This is a framework-port baseline, not yet the optimized online-maintained
    implementation from ``online_ivfpq_simulator.py``. It records current-context
    PQ construction as ``online_update`` and query scan traffic as ``selector``.
    """

    subvecs: int = 2
    subbits: int = 6
    kmeans_iters: int = 3
    score_key_bytes: int = 4
    attn_key_bytes: int = 2
    seed: int = 2025
    charge_online_update: bool = True

    name: str = "pqcache_full_scan"

    def _codebook_bytes(self, head_dim: int) -> int:
        centroids = 1 << int(self.subbits)
        return int(self.subvecs) * centroids * (int(head_dim) // int(self.subvecs)) * int(self.score_key_bytes)

    def _code_bytes(self, count: int) -> int:
        return int(count) * int(self.subvecs) * pq_code_bytes(int(self.subbits))

    def select(self, state: QueryState, *, target_mass: float | None = None, budget: int | None = None) -> SelectionResult:
        target = 1.0 if target_mass is None else float(target_mass)
        selected = unique_tokens(list(state.base_tokens), context_len=state.scores.shape[0])
        selected_set = set(selected)
        dynamic_tokens = [tok for tok in range(state.scores.shape[0]) if tok not in selected_set]
        cost = CostTrace()
        if not dynamic_tokens:
            return SelectionResult(self.name, selected_tokens=selected, candidate_tokens=[], cost=cost)

        token_arr = np.asarray(dynamic_tokens, dtype=np.int64)
        block = state.keys[token_arr].astype(np.float32, copy=False)
        if self.charge_online_update:
            cost.read("online_update", "pq_build_keys", block.shape[0] * state.keys.shape[-1] * int(self.attn_key_bytes))
        codebooks, codes, subvecs, centroids_per_subvec = build_pq_index(
            block,
            0,
            block.shape[0],
            subvecs=int(self.subvecs),
            subbits=int(self.subbits),
            seed=int(self.seed) + int(state.kv_head) * 2027 + int(state.decode_tokens),
            max_iter=int(self.kmeans_iters),
        )
        self.subvecs = int(subvecs)
        self.subbits = int(np.log2(int(centroids_per_subvec)))
        if self.charge_online_update:
            cost.write("online_update", "pq_codebooks", self._codebook_bytes(state.keys.shape[-1]))
            cost.write("online_update", "pq_codes", self._code_bytes(block.shape[0]))

        cost.read("selector", "pq_codebooks", self._codebook_bytes(state.keys.shape[-1]))
        cost.read("selector", "pq_codes", self._code_bytes(block.shape[0]))
        approx = pq_scores(state.query, codebooks, codes)
        order = np.argsort(-approx, kind="stable")
        ranked = token_arr[order].astype(np.int64, copy=False)

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
                "pq_subvecs": int(self.subvecs),
                "pq_subbits": int(self.subbits),
                "accounting_mode": "online_proxy" if self.charge_online_update else "snapshot",
                "online_update_modeled": bool(self.charge_online_update),
            },
        )
