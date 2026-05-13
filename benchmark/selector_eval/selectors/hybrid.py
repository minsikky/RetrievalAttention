from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from benchmark.online_ivfpq_simulator import pq_scores
from benchmark.selector_eval.costs.base import CostTrace, kv_read_bytes
from benchmark.selector_eval.data.trace import unique_tokens
from benchmark.selector_eval.selectors.base import QueryState, SelectionResult
from benchmark.selector_eval.selectors.paged_pq import PagedPQSelector, _event_bytes_mb, _event_bytes_to_cost


@dataclass
class PagedPQSparQRerankSelector(PagedPQSelector):
    """Use page-local PQ for candidate generation, then SparQ-rerank candidates.

    This tests whether SparQ's strong query-channel signal can improve the
    token ordering without paying SparQ's full-context channel scan.
    """

    sparq_rank: int = 16
    sparq_index_bytes: int = 4
    rerank_factors: tuple[int, ...] = (1, 2, 4)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.display_name:
            self.name = self.display_name
        elif self.routed:
            self.name = "gated_paged_pq_sparq_rerank"
        else:
            self.name = "paged_local_pq_sparq_rerank"

    def _sparq_rerank(self, state: QueryState, tokens: list[int], cost: CostTrace) -> list[int]:
        if not tokens:
            return []
        rank = min(max(1, int(self.sparq_rank)), int(state.query.shape[0]))
        dims = np.argsort(-np.abs(state.query), kind="stable")[:rank]
        token_arr = np.asarray(tokens, dtype=np.int64)
        q_abs_sum = max(float(np.abs(state.query).sum()), 1e-20)
        coverage = max(float(np.abs(state.query[dims]).sum() / q_abs_sum), 1e-6)
        scale = 1.0 / np.sqrt(float(state.query.shape[0]) * coverage)
        cost.read("selector", "hybrid_sparq_dims", int(rank) * int(self.sparq_index_bytes))
        cost.read("selector", "hybrid_sparq_key_channels", len(tokens) * int(rank) * int(self.score_key_bytes))
        approx = (state.keys[token_arr[:, None], dims] @ state.query[dims]).astype(np.float32) * scale
        order = np.argsort(-approx, kind="stable")
        return token_arr[order].astype(np.int64, copy=False).tolist()

    def select(self, state: QueryState, *, target_mass: float | None = None, budget: int | None = None) -> SelectionResult:
        target = 1.0 if target_mass is None else float(target_mass)
        index = self.indexes[state.kv_head]
        update_cost = self._advance(state)
        charged_update_cost = update_cost if self.accounting_mode == "online" else CostTrace()

        base = unique_tokens(list(state.base_tokens), context_len=state.scores.shape[0])
        base_set = set(base)
        pending = [
            int(tok)
            for tok in index.pending_tokens()
            if int(tok) < state.scores.shape[0] and int(tok) not in base_set
        ]
        pending_set = set(pending)

        if self.routed:
            routed = index.selection_routed_many(state.query, list(self.nprobes))
        else:
            ranked, selection_events = index.selection_fullscan(state.query)
            routed = {0: (ranked, selection_events)}

        factors = tuple(max(1, int(factor)) for factor in self.rerank_factors)
        choices = []
        for nprobe, (raw_ranked, selection_events) in routed.items():
            ranked = [
                int(tok)
                for tok in raw_ranked.tolist()
                if int(tok) < state.scores.shape[0] and int(tok) not in base_set and int(tok) not in pending_set
            ]
            base_mass = float(state.probs[np.asarray(base + pending, dtype=np.int64)].sum()) if base or pending else 0.0
            pq_mass = base_mass
            pq_cursor = 0
            while pq_mass < target and pq_cursor < len(ranked):
                pq_mass += float(state.probs[int(ranked[pq_cursor])])
                pq_cursor += 1
                if budget is not None and len(base) + len(pending) + pq_cursor >= int(budget):
                    break
            if pq_cursor <= 0 and ranked:
                pq_cursor = 1
            for factor in factors:
                pool_size = min(len(ranked), max(pq_cursor, pq_cursor * int(factor)))
                pool = ranked[:pool_size]
                selector_cost = _event_bytes_to_cost(selection_events, phase="selector")
                reranked = self._sparq_rerank(state, pool, selector_cost)
                selected = unique_tokens(base + pending, context_len=state.scores.shape[0])
                mass = float(state.probs[np.asarray(selected, dtype=np.int64)].sum()) if selected else 0.0
                cursor = 0
                while mass < target and cursor < len(reranked):
                    tok = int(reranked[cursor])
                    cursor += 1
                    selected.append(tok)
                    mass += float(state.probs[tok])
                    if budget is not None and len(selected) >= int(budget):
                        break
                selected = unique_tokens(selected, context_len=state.scores.shape[0])
                exact_mb = kv_read_bytes(len(selected), self.trace.head_dim, self.attn_key_bytes, self.value_bytes) / (
                    1024.0 * 1024.0
                )
                total_mb = selector_cost.mb() + exact_mb
                choices.append(
                    {
                        "reached": mass >= target,
                        "total_mb": total_mb,
                        "nprobe": int(nprobe),
                        "rerank_factor": int(factor),
                        "rerank_pool": int(pool_size),
                        "selected": selected,
                        "ranked": reranked,
                        "selector_cost": selector_cost,
                        "mass": mass,
                    }
                )

        reachable = [choice for choice in choices if choice["reached"]]
        choice = min(reachable, key=lambda item: item["total_mb"]) if reachable else max(choices, key=lambda item: item["mass"])
        cost = CostTrace()
        cost.extend(charged_update_cost)
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
                "rerank_factor": int(choice["rerank_factor"]),
                "rerank_pool": int(choice["rerank_pool"]),
                "pages": int(len(index.pages)),
                "pending_tokens": int(len(pending)),
                "router_groups": int(len(index.groups)) if self.routed else 0,
                "sparq_rank": int(self.sparq_rank),
                "accounting_mode": str(self.accounting_mode),
                "online_update_modeled": bool(self.accounting_mode == "online"),
                "online_update_cumulative_MB": (
                    _event_bytes_mb(index.update_events_total) if self.accounting_mode == "online" else 0.0
                ),
                "online_update_indexed_tokens": int(index.total_update_steps) if self.accounting_mode == "online" else 0,
            },
        )


@dataclass
class PageSparQPQSelector(PagedPQSelector):
    """Route pages with SparQ-channel min/max summaries, then scan local PQ.

    This avoids token-level SparQ over the full context. Query-time routing reads
    only top-query-channel page summaries, then scans page-local PQ codes within
    selected pages.
    """

    sparq_rank: int = 16
    sparq_index_bytes: int = 4

    def __post_init__(self) -> None:
        super().__post_init__()
        self.name = self.display_name or "page_sparq_pq"

    def _summary_update_mb(self, index) -> float:
        if self.accounting_mode != "online":
            return 0.0
        bytes_ = len(index.pages) * 2 * int(self.trace.head_dim) * int(self.score_key_bytes)
        return float(bytes_) / (1024.0 * 1024.0)

    def _rank_pages(self, state: QueryState, index, cost: CostTrace) -> list[int]:
        if not index.pages:
            return []
        rank = min(max(1, int(self.sparq_rank)), int(state.query.shape[0]))
        dims = np.argsort(-np.abs(state.query), kind="stable")[:rank]
        q = state.query[dims].astype(np.float32, copy=False)
        cost.read("selector", "page_sparq_dims", int(rank) * int(self.sparq_index_bytes))
        cost.read("selector", "page_sparq_minmax", len(index.pages) * int(rank) * 2 * int(self.score_key_bytes))
        scores = []
        for page_id, page in enumerate(index.pages):
            lo = int(page["token_start"])
            hi = min(lo + int(page["size"]), state.keys.shape[0])
            if hi <= lo:
                scores.append((float("-inf"), int(page_id)))
                continue
            block = state.keys[lo:hi, :].astype(np.float32, copy=False)
            vals = block[:, dims]
            mins = vals.min(axis=0)
            maxs = vals.max(axis=0)
            chosen = np.where(q >= 0.0, maxs, mins)
            scores.append((float(np.dot(chosen, q)), int(page_id)))
        scores.sort(key=lambda item: (-item[0], item[1]))
        return [page_id for _score, page_id in scores]

    def _scan_pages(self, state: QueryState, index, page_ids: list[int], cost: CostTrace) -> list[int]:
        token_parts = []
        score_parts = []
        for page_id in page_ids:
            page = index.pages[int(page_id)]
            size = int(page["size"])
            if size <= 0:
                continue
            cost.read("selector", "page_pq_codebooks", index._pq_codebook_bytes_per_page())
            cost.read("selector", "page_pq_codes", index._pq_code_bytes(size))
            scores = pq_scores(state.query.astype(np.float32, copy=False), page["codebooks"], page["codes"])
            tokens = int(page["token_start"]) + np.arange(size, dtype=np.int64)
            token_parts.append(tokens)
            score_parts.append(scores.astype(np.float32, copy=False))
        if not token_parts:
            return []
        tokens = np.concatenate(token_parts)
        scores = np.concatenate(score_parts)
        order = np.argsort(-scores, kind="stable")
        return tokens[order].astype(np.int64, copy=False).tolist()

    def select(self, state: QueryState, *, target_mass: float | None = None, budget: int | None = None) -> SelectionResult:
        target = 1.0 if target_mass is None else float(target_mass)
        index = self.indexes[state.kv_head]
        update_cost = self._advance(state)
        charged_update_cost = update_cost if self.accounting_mode == "online" else CostTrace()

        base = unique_tokens(list(state.base_tokens), context_len=state.scores.shape[0])
        base_set = set(base)
        pending = [
            int(tok)
            for tok in index.pending_tokens()
            if int(tok) < state.scores.shape[0] and int(tok) not in base_set
        ]
        pending_set = set(pending)

        route_cost = CostTrace()
        page_order = self._rank_pages(state, index, route_cost)
        choices = []
        for nprobe in self.nprobes:
            probe = min(max(1, int(nprobe)), len(page_order)) if page_order else 0
            selector_cost = CostTrace()
            selector_cost.extend(route_cost)
            ranked = self._scan_pages(state, index, page_order[:probe], selector_cost)
            ranked = [
                int(tok)
                for tok in ranked
                if int(tok) < state.scores.shape[0] and int(tok) not in base_set and int(tok) not in pending_set
            ]
            selected = unique_tokens(base + pending, context_len=state.scores.shape[0])
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
            exact_mb = kv_read_bytes(len(selected), self.trace.head_dim, self.attn_key_bytes, self.value_bytes) / (
                1024.0 * 1024.0
            )
            choices.append(
                {
                    "reached": mass >= target,
                    "total_mb": selector_cost.mb() + exact_mb,
                    "nprobe": int(probe),
                    "selected": selected,
                    "ranked": ranked,
                    "selector_cost": selector_cost,
                    "mass": mass,
                }
            )

        reachable = [choice for choice in choices if choice["reached"]]
        choice = min(reachable, key=lambda item: item["total_mb"]) if reachable else max(choices, key=lambda item: item["mass"])
        cost = CostTrace()
        cost.extend(charged_update_cost)
        cost.extend(choice["selector_cost"])
        cumulative_update_mb = (
            _event_bytes_mb(index.update_events_total) + self._summary_update_mb(index)
            if self.accounting_mode == "online"
            else 0.0
        )
        return SelectionResult(
            algorithm=self.name,
            selected_tokens=choice["selected"],
            candidate_tokens=choice["ranked"],
            cost=cost,
            metadata={
                "target_mass": target_mass,
                "budget": budget,
                "nprobe": int(choice["nprobe"]),
                "pages": int(len(index.pages)),
                "pending_tokens": int(len(pending)),
                "sparq_rank": int(self.sparq_rank),
                "accounting_mode": str(self.accounting_mode),
                "online_update_modeled": bool(self.accounting_mode == "online"),
                "online_update_cumulative_MB": cumulative_update_mb,
                "online_update_indexed_tokens": int(index.total_update_steps) if self.accounting_mode == "online" else 0,
            },
        )


@dataclass
class PageSparQPostingsPQSelector(PageSparQPQSelector):
    """Use per-page SparQ channel postings to form a sparse PQ candidate set."""

    postings_per_dim: int = 64

    def __post_init__(self) -> None:
        super().__post_init__()
        self.name = self.display_name or "page_sparq_postings_pq"

    def _summary_update_mb(self, index) -> float:
        if self.accounting_mode != "online":
            return 0.0
        minmax_bytes = len(index.pages) * 2 * int(self.trace.head_dim) * int(self.score_key_bytes)
        postings_bytes = (
            len(index.pages)
            * int(self.trace.head_dim)
            * 2
            * min(max(1, int(self.postings_per_dim)), int(self.page_size))
            * int(self.edge_index_bytes)
        )
        return float(minmax_bytes + postings_bytes) / (1024.0 * 1024.0)

    def _candidate_rows_from_postings(self, state: QueryState, index, page_ids: list[int], cost: CostTrace) -> dict[int, list[int]]:
        out: dict[int, set[int]] = {}
        if not page_ids:
            return {}
        rank = min(max(1, int(self.sparq_rank)), int(state.query.shape[0]))
        dims = np.argsort(-np.abs(state.query), kind="stable")[:rank]
        k = max(1, int(self.postings_per_dim))
        cost.read("selector", "page_sparq_posting_heads", len(page_ids) * rank * 2 * int(self.edge_index_bytes))
        for page_id in page_ids:
            page = index.pages[int(page_id)]
            lo = int(page["token_start"])
            hi = min(lo + int(page["size"]), state.keys.shape[0])
            if hi <= lo:
                continue
            block = state.keys[lo:hi, :].astype(np.float32, copy=False)
            rows_out = out.setdefault(int(page_id), set())
            for dim in dims.tolist():
                vals = block[:, int(dim)]
                take = min(k, vals.shape[0])
                if state.query[int(dim)] >= 0.0:
                    local = np.argpartition(-vals, take - 1)[:take]
                else:
                    local = np.argpartition(vals, take - 1)[:take]
                cost.read("selector", "page_sparq_postings", take * int(self.edge_index_bytes))
                rows_out.update(int(x) for x in local.tolist())
        return {page_id: sorted(rows) for page_id, rows in out.items() if rows}

    def _scan_page_rows(self, state: QueryState, index, page_rows: dict[int, list[int]], cost: CostTrace) -> list[int]:
        token_parts = []
        score_parts = []
        for page_id in sorted(page_rows):
            page = index.pages[int(page_id)]
            rows = np.asarray(page_rows[page_id], dtype=np.int64)
            if rows.size == 0:
                continue
            cost.read("selector", "page_pq_codebooks", index._pq_codebook_bytes_per_page())
            cost.read("selector", "page_pq_codes", index._pq_code_bytes(int(rows.size)))
            scores = pq_scores(state.query.astype(np.float32, copy=False), page["codebooks"], page["codes"][rows])
            tokens = int(page["token_start"]) + rows
            token_parts.append(tokens.astype(np.int64, copy=False))
            score_parts.append(scores.astype(np.float32, copy=False))
        if not token_parts:
            return []
        tokens = np.concatenate(token_parts)
        scores = np.concatenate(score_parts)
        order = np.argsort(-scores, kind="stable")
        return tokens[order].astype(np.int64, copy=False).tolist()

    def select(self, state: QueryState, *, target_mass: float | None = None, budget: int | None = None) -> SelectionResult:
        target = 1.0 if target_mass is None else float(target_mass)
        index = self.indexes[state.kv_head]
        update_cost = self._advance(state)
        charged_update_cost = update_cost if self.accounting_mode == "online" else CostTrace()

        base = unique_tokens(list(state.base_tokens), context_len=state.scores.shape[0])
        base_set = set(base)
        pending = [
            int(tok)
            for tok in index.pending_tokens()
            if int(tok) < state.scores.shape[0] and int(tok) not in base_set
        ]
        pending_set = set(pending)

        route_cost = CostTrace()
        page_order = self._rank_pages(state, index, route_cost)
        choices = []
        for nprobe in self.nprobes:
            probe = min(max(1, int(nprobe)), len(page_order)) if page_order else 0
            selector_cost = CostTrace()
            selector_cost.extend(route_cost)
            page_rows = self._candidate_rows_from_postings(state, index, page_order[:probe], selector_cost)
            ranked = self._scan_page_rows(state, index, page_rows, selector_cost)
            ranked = [
                int(tok)
                for tok in ranked
                if int(tok) < state.scores.shape[0] and int(tok) not in base_set and int(tok) not in pending_set
            ]
            selected = unique_tokens(base + pending, context_len=state.scores.shape[0])
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
            exact_mb = kv_read_bytes(len(selected), self.trace.head_dim, self.attn_key_bytes, self.value_bytes) / (
                1024.0 * 1024.0
            )
            choices.append(
                {
                    "reached": mass >= target,
                    "total_mb": selector_cost.mb() + exact_mb,
                    "nprobe": int(probe),
                    "selected": selected,
                    "ranked": ranked,
                    "selector_cost": selector_cost,
                    "mass": mass,
                }
            )

        reachable = [choice for choice in choices if choice["reached"]]
        choice = min(reachable, key=lambda item: item["total_mb"]) if reachable else max(choices, key=lambda item: item["mass"])
        cost = CostTrace()
        cost.extend(charged_update_cost)
        cost.extend(choice["selector_cost"])
        cumulative_update_mb = (
            _event_bytes_mb(index.update_events_total) + self._summary_update_mb(index)
            if self.accounting_mode == "online"
            else 0.0
        )
        return SelectionResult(
            algorithm=self.name,
            selected_tokens=choice["selected"],
            candidate_tokens=choice["ranked"],
            cost=cost,
            metadata={
                "target_mass": target_mass,
                "budget": budget,
                "nprobe": int(choice["nprobe"]),
                "pages": int(len(index.pages)),
                "pending_tokens": int(len(pending)),
                "sparq_rank": int(self.sparq_rank),
                "postings_per_dim": int(self.postings_per_dim),
                "accounting_mode": str(self.accounting_mode),
                "online_update_modeled": bool(self.accounting_mode == "online"),
                "online_update_cumulative_MB": cumulative_update_mb,
                "online_update_indexed_tokens": int(index.total_update_steps) if self.accounting_mode == "online" else 0,
            },
        )
