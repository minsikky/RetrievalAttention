from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from benchmark.attention_efficiency_threeway_eval import lloyd_kmeans
from benchmark.selector_eval.costs.base import CostTrace
from benchmark.selector_eval.data.trace import QKVTrace, unique_tokens
from benchmark.selector_eval.metrics.tail_estimators import _paper_tq_scores, _tq_code_bytes, _tq_reconstruct, _tq_reconstruct_product
from benchmark.selector_eval.selectors.base import QueryState, SelectionResult


def _budget_from_rule(rule: str, context_len: int) -> int:
    normalized = str(rule).strip().lower()
    n = max(1, int(context_len))
    if normalized.startswith("k"):
        return max(0, int(normalized.removeprefix("k")))
    if normalized.startswith("sqrt_x"):
        mult = float(normalized.removeprefix("sqrt_x").replace("p", "."))
        return int(np.ceil(mult * np.sqrt(float(n))))
    if normalized.startswith("log_x"):
        mult = float(normalized.removeprefix("log_x").replace("p", "."))
        return int(np.ceil(mult * np.log2(float(n))))
    if normalized.startswith("n0"):
        exp_text, mult_text = normalized.split("_x", 1)
        exponent = float("0." + exp_text.removeprefix("n0"))
        mult = float(mult_text.replace("p", "."))
        return int(np.ceil(mult * (float(n) ** exponent)))
    raise ValueError(f"unknown budget rule: {rule}")


@dataclass
class IVFTurboQuantSelector:
    """IVF bucket router with TurboQuant-compressed K scoring inside buckets.

    The router scans all coarse centroids, visits the smallest tested nprobe
    that can provide the requested fixed budget, and ranks only those bucket
    members using TurboQuant-compressed K scores. It is a deployable fixed-budget
    selector: it does not use true attention probabilities or achieved mass.
    """

    trace: QKVTrace
    key_bits: int = 3
    budget_rule: str = "k4096"
    coarse_clusters: int = 256
    coarse_iters: int = 3
    assignment_replicas: int = 1
    nprobes: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)
    route_multiplier: float = 1.0
    product_residual: bool = True
    faithful_scorer: bool = False
    static_prefix: int = 128
    static_suffix: int = 128
    seed: int = 2025
    score_key_bytes: int = 4
    attn_key_bytes: int = 2
    value_bytes: int = 2
    edge_index_bytes: int = 4
    graph_offset_bytes: int = 4
    accounting_mode: str = "online"
    display_name: str | None = None

    def __post_init__(self) -> None:
        if self.faithful_scorer:
            prefix = "ivfpapertqprod" if self.product_residual else "ivfpapertqmse"
        else:
            prefix = "ivftqprod" if self.product_residual else "ivftqmse"
        self.name = self.display_name or (
            f"{prefix}_c{int(self.coarse_clusters)}_r{int(self.assignment_replicas)}_k{int(self.key_bits)}_m{self.route_multiplier:g}_budget_{self.budget_rule}"
        )
        self.dynamic_start = min(max(0, int(self.static_prefix)), int(self.trace.input_len))
        self.init_dynamic_end = max(self.dynamic_start, int(self.trace.input_len) - max(0, int(self.static_suffix)))
        self.indexes = [_IVFTQIndex(self, kv_h) for kv_h in range(self.trace.kv_heads)]

    def select(self, state: QueryState, *, target_mass: float | None = None, budget: int | None = None) -> SelectionResult:
        index = self.indexes[int(state.kv_head)]
        update_cost = index.advance_to(
            max(
                self.dynamic_start,
                min(int(state.position) + 1 - max(0, int(self.static_suffix)), self.trace.keys.shape[1]),
            )
        )
        charged_update = update_cost if self.accounting_mode == "online" else CostTrace()
        base = unique_tokens(list(state.base_tokens), context_len=state.scores.shape[0])
        base_set = set(base)
        dynamic_budget = _budget_from_rule(self.budget_rule, state.scores.shape[0])
        if budget is not None:
            dynamic_budget = min(dynamic_budget, max(0, int(budget) - len(base)))

        route_target = int(np.ceil(float(dynamic_budget) * max(1.0, float(self.route_multiplier))))
        choices = index.selection_many(state.query, list(self.nprobes), dynamic_budget=route_target)
        valid_choices = [choice for choice in choices if int(choice["ranked"].size) >= int(route_target)]
        choice = valid_choices[0] if valid_choices else choices[-1]
        ranked = [int(tok) for tok in choice["ranked"].tolist() if int(tok) < state.scores.shape[0] and int(tok) not in base_set]
        selected = unique_tokens(base + ranked[:dynamic_budget], context_len=state.scores.shape[0])

        cost = CostTrace()
        cost.extend(charged_update)
        cost.extend(choice["cost"])
        approx_scores_by_token = np.full((state.scores.shape[0],), -np.inf, dtype=np.float32)
        ranked_scores = choice.get("scores")
        if ranked_scores is not None:
            for tok, score in zip(choice["ranked"].tolist(), ranked_scores.tolist(), strict=False):
                if int(tok) < approx_scores_by_token.shape[0]:
                    approx_scores_by_token[int(tok)] = float(score)

        return SelectionResult(
            algorithm=self.name,
            selected_tokens=selected,
            candidate_tokens=ranked,
            cost=cost,
            metadata={
                "target_mass": target_mass,
                "budget": budget,
                "budget_rule": self.budget_rule,
                "dynamic_budget": int(dynamic_budget),
                "nprobe": int(choice["nprobe"]),
                "scanned_tokens": int(choice["scanned_tokens"]),
                "route_multiplier": float(self.route_multiplier),
                "route_target": int(route_target),
                "coarse_clusters": int(self.coarse_clusters),
                "assignment_replicas": int(self.assignment_replicas),
                "turboquant_key_bits": int(self.key_bits),
                "turboquant_product_residual": bool(self.product_residual),
                "turboquant_faithful_scorer": bool(self.faithful_scorer),
                "approx_scores": approx_scores_by_token,
                "accounting_mode": str(self.accounting_mode),
                "online_update_modeled": bool(self.accounting_mode == "online"),
                "online_update_cumulative_MB": index.update_cost.mb(phase="online_update") if self.accounting_mode == "online" else 0.0,
                "online_update_indexed_tokens": int(index.total_update_steps) if self.accounting_mode == "online" else 0,
            },
        )


class _IVFTQIndex:
    def __init__(self, selector: IVFTurboQuantSelector, kv_head: int) -> None:
        self.selector = selector
        self.kv_head = int(kv_head)
        self.keys = selector.trace.keys[int(kv_head)].astype(np.float32, copy=False)
        self.dim = int(selector.trace.head_dim)
        self.token_start = int(selector.dynamic_start)
        self.size = max(0, int(selector.init_dynamic_end) - int(selector.dynamic_start))
        self.capacity = max(1, int(self.keys.shape[0]) - self.token_start)
        self.assign = np.full((self.capacity,), -1, dtype=np.int32)
        self.update_cost = CostTrace()
        self.total_update_steps = 0
        self._build_initial()

    def _key_bytes(self, count: int) -> int:
        return int(count) * self.dim * int(self.selector.attn_key_bytes)

    def _centroid_bytes(self) -> int:
        return int(self.centroids.shape[0]) * self.dim * int(self.selector.score_key_bytes)

    def _tq_sidecar_bytes(self, count: int) -> float:
        out = _tq_code_bytes(count, self.dim, int(self.selector.key_bits)) + int(count) * int(self.selector.attn_key_bytes)
        if self.selector.product_residual:
            out += _tq_code_bytes(count, self.dim, 1) + int(count) * int(self.selector.attn_key_bytes)
        return float(out)

    def _build_initial(self) -> None:
        if self.size <= 0:
            self.centroids = np.zeros((0, self.dim), dtype=np.float32)
            self.buckets: list[list[int]] = []
            self.counts = np.zeros((0,), dtype=np.int64)
            return
        token_ids = np.arange(self.token_start, self.token_start + self.size, dtype=np.int64)
        block = self.keys[token_ids].astype(np.float32, copy=False)
        centroids, assign = lloyd_kmeans(
            block,
            int(self.selector.coarse_clusters),
            seed=int(self.selector.seed) + 2027 * int(self.kv_head),
            max_iter=int(self.selector.coarse_iters),
        )
        self.centroids = centroids.astype(np.float32, copy=False)
        self.assign[: self.size] = assign.astype(np.int32, copy=False)
        replicas = max(1, min(int(self.selector.assignment_replicas), int(self.centroids.shape[0])))
        scores = block.astype(np.float32, copy=False) @ self.centroids.astype(np.float32, copy=False).T
        top = np.argpartition(-scores, kth=replicas - 1, axis=1)[:, :replicas]
        self.buckets = [[] for _ in range(self.centroids.shape[0])]
        for row_id, cids in enumerate(top.tolist()):
            for cid in cids:
                self.buckets[int(cid)].append(int(row_id))
        self.counts = np.bincount(assign, minlength=self.centroids.shape[0]).astype(np.int64)

    def advance_to(self, indexed_hi: int) -> CostTrace:
        indexed_hi = min(max(0, int(indexed_hi)), self.keys.shape[0])
        next_tok = self.token_start + self.size
        delta = CostTrace()
        if self.centroids.shape[0] == 0:
            return delta
        while next_tok < indexed_hi:
            key = self.keys[int(next_tok)].astype(np.float32, copy=False)
            delta.read("online_update", "ivftq_append_key", self._key_bytes(1))
            delta.read("online_update", "ivftq_assign_centroids", self._centroid_bytes())
            centroid_scores = self.centroids.astype(np.float32, copy=False) @ key
            cid = int(np.argmax(centroid_scores))
            replicas = max(1, min(int(self.selector.assignment_replicas), int(self.centroids.shape[0])))
            replica_cids = np.argpartition(-centroid_scores, kth=replicas - 1)[:replicas]
            row_id = int(self.size)
            self.assign[row_id] = cid
            for replica_cid in replica_cids.tolist():
                self.buckets[int(replica_cid)].append(row_id)
            self.counts[cid] += 1
            self.size += 1
            self.total_update_steps += 1
            delta.write("online_update", "ivftq_key_sidecar", self._tq_sidecar_bytes(1))
            delta.write("online_update", "ivftq_postings", replicas * int(self.selector.edge_index_bytes))
            next_tok += 1
        self.update_cost.extend(delta)
        return delta

    def selection_many(self, query: np.ndarray, nprobes: list[int], *, dynamic_budget: int) -> list[dict]:
        if self.size <= 0 or self.centroids.shape[0] == 0:
            empty = np.empty((0,), dtype=np.int64)
            return [{"nprobe": int(nprobe), "ranked": empty, "scores": np.empty((0,), dtype=np.float32), "scanned_tokens": 0, "cost": CostTrace()} for nprobe in nprobes]
        q = query.astype(np.float32, copy=False)
        cost_base = CostTrace()
        cost_base.read("selector", "ivftq_coarse_centroids", self._centroid_bytes())
        coarse_scores = self.centroids.astype(np.float32, copy=False) @ q
        coarse_order = np.argsort(-coarse_scores, kind="stable")
        out = []
        max_nprobe = max(1, min(max(int(n) for n in nprobes), self.centroids.shape[0]))
        prefix_rows: list[int] = []
        seen_buckets = 0
        cached: dict[int, tuple[np.ndarray, np.ndarray, int]] = {}
        for nprobe in sorted(set(max(1, min(int(n), self.centroids.shape[0])) for n in nprobes)):
            while seen_buckets < nprobe:
                cid = int(coarse_order[seen_buckets])
                prefix_rows.extend(self.buckets[cid])
                seen_buckets += 1
            posting_count = len(prefix_rows)
            row_ids = np.unique(np.asarray(prefix_rows, dtype=np.int64)) if prefix_rows else np.empty((0,), dtype=np.int64)
            if row_ids.size:
                token_ids = row_ids + int(self.token_start)
                key_block = self.keys[token_ids].astype(np.float32, copy=False)
                if self.selector.faithful_scorer:
                    scores = _paper_tq_scores(
                        key_block,
                        q,
                        int(self.selector.key_bits),
                        product_residual=bool(self.selector.product_residual),
                    )
                else:
                    key_hat = (
                        _tq_reconstruct_product(key_block, int(self.selector.key_bits))
                        if self.selector.product_residual
                        else _tq_reconstruct(key_block, int(self.selector.key_bits))[0]
                    )
                    scores = key_hat.astype(np.float64, copy=False) @ q.astype(np.float64, copy=False)
                order = np.argsort(-scores, kind="stable")
                ranked = token_ids[order].astype(np.int64, copy=False)
                ranked_scores = scores[order].astype(np.float32, copy=False)
            else:
                ranked = np.empty((0,), dtype=np.int64)
                ranked_scores = np.empty((0,), dtype=np.float32)
            cost = CostTrace()
            cost.extend(cost_base)
            cost.read("selector", "ivftq_offsets", int(nprobe) * int(self.selector.graph_offset_bytes))
            cost.read("selector", "ivftq_postings", int(posting_count) * int(self.selector.edge_index_bytes))
            cost.read("selector", "ivftq_key_sidecar", self._tq_sidecar_bytes(int(row_ids.size)))
            out.append(
                {
                    "nprobe": int(nprobe),
                    "ranked": ranked,
                    "scores": ranked_scores,
                    "scanned_tokens": int(row_ids.size),
                    "cost": cost,
                }
            )
            if int(row_ids.size) >= int(dynamic_budget) and nprobe >= max(1, min(max_nprobe, nprobe)):
                # Still return all requested nprobes already reached in this loop;
                # caller chooses the first sufficient one.
                pass
        return out
