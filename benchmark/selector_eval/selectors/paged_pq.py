from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from benchmark.online_ivfpq_simulator import EventBytes, PagedLocalPQIndex, build_pq_index
from benchmark.selector_eval.costs.base import CostTrace, kv_read_bytes
from benchmark.selector_eval.data.trace import QKVTrace, unique_tokens
from benchmark.selector_eval.selectors.base import QueryState, SelectionResult


NON_MEMORY_EVENT_CATEGORIES = {
    "page_pq_build_work",
    "page_proto_build_work",
}


def _event_bytes_to_cost(events: EventBytes, *, phase: str) -> CostTrace:
    cost = CostTrace()
    for category, bytes_ in events.reads.items():
        if category in NON_MEMORY_EVENT_CATEGORIES:
            continue
        cost.read(phase, category, bytes_)
    for category, bytes_ in events.writes.items():
        if category in NON_MEMORY_EVENT_CATEGORIES:
            continue
        cost.write(phase, category, bytes_)
    return cost


def _event_bytes_mb(events: EventBytes) -> float:
    return _event_bytes_to_cost(events, phase="online_update").mb(phase="online_update")


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
class PagedPQSelector:
    """Adapter for page-local PQ selectors from ``online_ivfpq_simulator.py``."""

    trace: QKVTrace
    static_prefix: int = 128
    static_suffix: int = 128
    page_size: int = 2048
    routed: bool = False
    nprobes: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
    router_prototypes: int = 16
    router_merge_rel: float = 0.05
    router_merge_var: float = 0.0
    router_max_groups: int = 512
    subvecs: int = 2
    subbits: int = 6
    kmeans_iters: int = 3
    pq_permutation: str = "none"
    seed: int = 2025
    stop_policy: str = "oracle_mass"
    approx_mass_margin: float = 0.0
    approx_mass_margin_schedule: tuple[tuple[int, float], ...] = ()
    guard_tokens: int = 4096
    probe_tokens: int = 512
    probe_bands: int = 8
    probe_residual_quantile: float = 0.95
    probe_confidence_z: float = 1.64
    budget_fraction: float = 0.0
    budget_rule: str = ""
    sparq_residual_rank: int = 0
    sparq_residual_pool: bool = False
    sparq_residual_window: int = 0
    exact_verify_window: int = 0
    verify_proj_dim: int = 0
    verify_proj_window: int = 0
    sparq_index_bytes: int = 4
    score_key_bytes: int = 4
    attn_key_bytes: int = 2
    value_bytes: int = 2
    value_pq_subvecs: int = 0
    value_pq_subbits: int = 0
    edge_index_bytes: int = 4
    graph_offset_bytes: int = 4
    display_name: str | None = None
    accounting_mode: str = "online"

    def __post_init__(self) -> None:
        self.name = self.display_name or ("gated_paged_pq" if self.routed else "paged_local_pq")
        if self.accounting_mode not in {"snapshot", "online"}:
            raise ValueError(f"unknown accounting_mode: {self.accounting_mode}")
        if self.stop_policy not in {
            "oracle_mass",
            "approx_mass",
            "approx_bound",
            "approx_guard",
            "approx_probe",
            "approx_probe_ucb",
            "approx_probe_ratio",
            "approx_probe_bands",
            "fixed_fraction",
            "fixed_budget",
        }:
            raise ValueError(f"unknown stop_policy: {self.stop_policy}")
        self.dynamic_start = min(max(0, int(self.static_prefix)), int(self.trace.input_len))
        self.init_dynamic_end = max(self.dynamic_start, int(self.trace.input_len) - max(0, int(self.static_suffix)))
        self.args = argparse.Namespace(
            static_prefix=int(self.static_prefix),
            static_suffix=int(self.static_suffix),
            paged_pq_page_size=int(self.page_size),
            paged_router_prototypes=int(self.router_prototypes),
            paged_router_merge_rel=float(self.router_merge_rel),
            paged_router_merge_var=float(self.router_merge_var),
            paged_router_max_groups=int(self.router_max_groups),
            paged_pq_permutation=str(self.pq_permutation),
            paged_verify_proj_dim=int(self.verify_proj_dim),
            pqcache_subvecs=int(self.subvecs),
            pqcache_subbits=int(self.subbits),
            pqcache_kmeans_iters=int(self.kmeans_iters),
            score_key_bytes_per_element=int(self.score_key_bytes),
            attn_key_bytes_per_element=int(self.attn_key_bytes),
            value_bytes_per_element=int(self.value_bytes),
            edge_index_bytes=int(self.edge_index_bytes),
            graph_offset_bytes=int(self.graph_offset_bytes),
            head_dim=int(self.trace.head_dim),
        )
        self.indexes = [
            PagedLocalPQIndex(
                keys=self.trace.keys[kv_h],
                init_start=self.dynamic_start,
                init_end=self.init_dynamic_end,
                args=self.args,
                seed=int(self.seed) + 2027 * int(kv_h),
                router_enabled=bool(self.routed),
            )
            for kv_h in range(self.trace.kv_heads)
        ]
        self.value_pq_sidecars: list[list[np.ndarray]] = [[] for _ in range(self.trace.kv_heads)]
        self.value_pq_sidecar_update_bytes = [0.0 for _ in range(self.trace.kv_heads)]
        self.value_vpq_sidecars: list[list[tuple[np.ndarray, np.ndarray]]] = [[] for _ in range(self.trace.kv_heads)]
        self.value_vpq_sidecar_update_bytes = [0.0 for _ in range(self.trace.kv_heads)]
        self.key_residual_pq_sidecars: list[list[tuple[np.ndarray, np.ndarray]]] = [[] for _ in range(self.trace.kv_heads)]
        self.key_residual_pq_sidecar_update_bytes = [0.0 for _ in range(self.trace.kv_heads)]
        for index in self.indexes:
            index.update_events_total = EventBytes()
            index.total_update_steps = 0

    def _advance(self, state: QueryState) -> CostTrace:
        indexed_hi = max(
            self.dynamic_start,
            min(int(state.position) + 1 - max(0, int(self.static_suffix)), self.trace.keys.shape[1]),
        )
        before_reads = dict(self.indexes[state.kv_head].update_events_total.reads)
        before_writes = dict(self.indexes[state.kv_head].update_events_total.writes)
        self.indexes[state.kv_head].advance_to(indexed_hi)
        after = self.indexes[state.kv_head].update_events_total
        delta = EventBytes()
        for category, value in after.reads.items():
            delta.read(category, float(value) - float(before_reads.get(category, 0.0)))
        for category, value in after.writes.items():
            delta.write(category, float(value) - float(before_writes.get(category, 0.0)))
        return _event_bytes_to_cost(delta, phase="online_update")

    def _value_pq_codebook_bytes_per_page(self) -> int:
        return (
            int(self._value_pq_subvecs())
            * (1 << int(self._value_pq_subbits()))
            * (int(self.trace.head_dim) // int(self._value_pq_subvecs()))
            * int(self.value_bytes)
        )

    def _pq_code_bytes_per_token(self) -> int:
        return int(self.subvecs) * (1 if int(self.subbits) <= 8 else 2)

    def _value_pq_subvecs(self) -> int:
        return int(self.value_pq_subvecs) if int(self.value_pq_subvecs) > 0 else int(self.subvecs)

    def _value_pq_subbits(self) -> int:
        return int(self.value_pq_subbits) if int(self.value_pq_subbits) > 0 else int(self.subbits)

    def _value_pq_code_bytes_per_token(self) -> int:
        return int(self._value_pq_subvecs()) * (1 if int(self._value_pq_subbits()) <= 8 else 2)

    def _ensure_value_pq_sidecars(self, kv_head: int, index: PagedLocalPQIndex) -> None:
        """Build value centroids keyed by the page's existing K-PQ subcodes.

        The selector already stores one K-PQ code per token. For the tail
        control-variate baseline, we add a small V-side centroid table per
        page/subvector/subcode but reuse those existing token codes.
        """

        sidecars = self.value_pq_sidecars[int(kv_head)]
        subvecs = int(self.subvecs)
        centroids = 1 << int(self.subbits)
        subdim = int(self.trace.head_dim) // subvecs
        while len(sidecars) < len(index.pages):
            page = index.pages[len(sidecars)]
            start = int(page["token_start"])
            size = int(page["size"])
            values = self.trace.values[int(kv_head), start : start + size].astype(np.float32, copy=False)
            codes = np.asarray(page["codes"], dtype=np.int64)
            codebook = np.zeros((subvecs, centroids, subdim), dtype=np.float32)
            for sub in range(subvecs):
                part = values[:, sub * subdim : (sub + 1) * subdim]
                fallback = part.mean(axis=0) if part.shape[0] else np.zeros((subdim,), dtype=np.float32)
                for code in range(centroids):
                    mask = codes[:, sub] == code
                    codebook[sub, code] = part[mask].mean(axis=0) if np.any(mask) else fallback
            sidecars.append(codebook)
            self.value_pq_sidecar_update_bytes[int(kv_head)] += (
                size * int(self.trace.head_dim) * int(self.value_bytes)
                + index._pq_code_bytes(size)
                + self._value_pq_codebook_bytes_per_page()
            )

        vpq_sidecars = self.value_vpq_sidecars[int(kv_head)]
        value_subvecs = self._value_pq_subvecs()
        value_subbits = self._value_pq_subbits()
        while len(vpq_sidecars) < len(index.pages):
            page = index.pages[len(vpq_sidecars)]
            start = int(page["token_start"])
            size = int(page["size"])
            values = self.trace.values[int(kv_head), start : start + size].astype(np.float32, copy=False)
            codebooks, codes, _subvecs, _centroids = build_pq_index(
                values,
                0,
                values.shape[0],
                subvecs=int(value_subvecs),
                subbits=int(value_subbits),
                seed=int(self.seed) + 3571 + int(start),
                max_iter=int(self.kmeans_iters),
            )
            vpq_sidecars.append((codebooks.astype(np.float32, copy=False), codes.astype(np.uint16, copy=False)))
            self.value_vpq_sidecar_update_bytes[int(kv_head)] += (
                size * int(self.trace.head_dim) * int(self.value_bytes)
                + size * self._value_pq_code_bytes_per_token()
                + self._value_pq_codebook_bytes_per_page()
            )

        key_resid_sidecars = self.key_residual_pq_sidecars[int(kv_head)]
        while len(key_resid_sidecars) < len(index.pages):
            page = index.pages[len(key_resid_sidecars)]
            start = int(page["token_start"])
            size = int(page["size"])
            block = self.trace.keys[int(kv_head), start : start + size].astype(np.float32, copy=False)
            codebooks = np.asarray(page["codebooks"], dtype=np.float32)
            codes = np.asarray(page["codes"], dtype=np.int64)
            subdim = int(self.trace.head_dim) // int(self.subvecs)
            khat_perm = np.zeros_like(block, dtype=np.float32)
            for sub in range(int(self.subvecs)):
                khat_perm[:, sub * subdim : (sub + 1) * subdim] = codebooks[sub, codes[:, sub]]
            if "perm" in page:
                perm = np.asarray(page["perm"], dtype=np.int64)
                khat = np.zeros_like(khat_perm, dtype=np.float32)
                khat[:, perm] = khat_perm
            else:
                khat = khat_perm
            residual = block - khat
            resid_codebooks, resid_codes, _subvecs, _centroids = build_pq_index(
                residual,
                0,
                residual.shape[0],
                subvecs=int(self.subvecs),
                subbits=int(self.subbits),
                seed=int(self.seed) + 6151 + int(start),
                max_iter=int(self.kmeans_iters),
            )
            key_resid_sidecars.append((resid_codebooks.astype(np.float32, copy=False), resid_codes.astype(np.uint16, copy=False)))
            self.key_residual_pq_sidecar_update_bytes[int(kv_head)] += (
                size * int(self.trace.head_dim) * int(self.attn_key_bytes)
                + size * self._pq_code_bytes_per_token()
                + self._value_pq_codebook_bytes_per_page()
            )

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

        if self.stop_policy in {
            "approx_bound",
            "approx_guard",
            "approx_probe",
            "approx_probe_ucb",
            "approx_probe_ratio",
            "approx_probe_bands",
            "fixed_fraction",
        } and self.routed:
            raise ValueError(f"{self.stop_policy} stop policy is only implemented for full-scan paged PQ")
        if self.stop_policy == "approx_bound":
            ranked, approx_scores, approx_bounds, selection_events = index.selection_fullscan_bounded(state.query)
            routed = {0: (ranked, approx_scores, approx_bounds, selection_events)}
        elif self.routed:
            routed = {}
            for nprobe, (ranked, approx_scores, selection_events) in index.selection_routed_many_scored(
                state.query, list(self.nprobes)
            ).items():
                routed[nprobe] = (ranked, approx_scores, np.zeros_like(approx_scores, dtype=np.float32), selection_events)
        elif int(self.exact_verify_window) > 0 or int(self.verify_proj_window) > 0:
            ranked, approx_scores, selection_events = self._selection_fullscan_verify_window_scored(
                state,
                index,
                target=target,
                base_tokens=base + pending,
            )
            approx_bounds = np.zeros_like(approx_scores, dtype=np.float32)
            routed = {0: (ranked, approx_scores, approx_bounds, selection_events)}
        elif int(self.sparq_residual_rank) > 0:
            ranked, approx_scores, selection_events = self._selection_fullscan_residual_scored(
                state,
                index,
                target=target,
                base_tokens=base + pending,
            )
            approx_bounds = np.zeros_like(approx_scores, dtype=np.float32)
            routed = {0: (ranked, approx_scores, approx_bounds, selection_events)}
        else:
            ranked, approx_scores, selection_events = index.selection_fullscan_scored(state.query)
            approx_bounds = np.zeros_like(approx_scores, dtype=np.float32)
            routed = {0: (ranked, approx_scores, approx_bounds, selection_events)}

        choices = []
        for nprobe, (raw_ranked, raw_approx_scores, raw_approx_bounds, selection_events) in routed.items():
            ranked = []
            ranked_approx_scores = []
            ranked_approx_bounds = []
            approx_scores_by_token = np.full((state.scores.shape[0],), -np.inf, dtype=np.float32)
            for tok, approx_score, approx_bound in zip(
                raw_ranked.tolist(), raw_approx_scores.tolist(), raw_approx_bounds.tolist(), strict=False
            ):
                tok = int(tok)
                if tok < state.scores.shape[0] and tok not in base_set and tok not in pending_set:
                    ranked.append(tok)
                    ranked_approx_scores.append(float(approx_score))
                    ranked_approx_bounds.append(float(approx_bound))
                    approx_scores_by_token[tok] = float(approx_score)
            selected = unique_tokens(base + pending, context_len=state.scores.shape[0])
            selector_cost = _event_bytes_to_cost(selection_events, phase="selector")
            if self.stop_policy in {"approx_probe", "approx_probe_ucb", "approx_probe_ratio", "approx_probe_bands"}:
                choices.append(
                    self._probe_choice(
                        state=state,
                        selected=selected,
                        ranked=ranked,
                        ranked_approx_scores=ranked_approx_scores,
                        selector_cost=selector_cost,
                        nprobe=int(nprobe),
                        target=target,
                        approx_scores_by_token=approx_scores_by_token,
                    )
                )
                continue
            if self.stop_policy == "fixed_fraction":
                choices.append(
                    self._fixed_fraction_choice(
                        state=state,
                        selected=selected,
                        ranked=ranked,
                        selector_cost=selector_cost,
                        nprobe=int(nprobe),
                        target=target,
                        approx_scores_by_token=approx_scores_by_token,
                    )
                )
                continue
            if self.stop_policy == "fixed_budget":
                choices.append(
                    self._fixed_budget_choice(
                        state=state,
                        selected=selected,
                        ranked=ranked,
                        selector_cost=selector_cost,
                        nprobe=int(nprobe),
                        target=target,
                        approx_scores_by_token=approx_scores_by_token,
                    )
                )
                continue
            if self.stop_policy == "approx_guard":
                choices.append(
                    self._guard_choice(
                        state=state,
                        selected=selected,
                        ranked=ranked,
                        ranked_approx_scores=ranked_approx_scores,
                        selector_cost=selector_cost,
                        nprobe=int(nprobe),
                        target=target,
                        approx_scores_by_token=approx_scores_by_token,
                    )
                )
                continue
            true_mass = (
                float(state.probs[np.asarray(selected, dtype=np.int64)].sum())
                if self.stop_policy == "oracle_mass" and selected
                else 0.0
            )
            approx_mass, ranked_approx_probs = self._approx_distribution(state, selected, ranked_approx_scores)
            bound_num, bound_denom, bound_lower_weights, bound_denom_deltas = self._bound_distribution_state(
                state, selected, ranked_approx_scores, ranked_approx_bounds
            )
            cursor = 0
            routed_added = []
            effective_margin = self._effective_approx_margin(state)
            stop_target = min(1.0, target + max(0.0, float(effective_margin)))
            selector_stop_mass = self._selector_stop_mass(true_mass, approx_mass, bound_num, bound_denom)
            while selector_stop_mass < stop_target and cursor < len(ranked):
                tok = int(ranked[cursor])
                approx_prob = float(ranked_approx_probs[cursor]) if cursor < len(ranked_approx_probs) else 0.0
                cursor += 1
                selected.append(tok)
                routed_added.append(tok)
                if self.stop_policy == "oracle_mass":
                    true_mass += float(state.probs[tok])
                approx_mass += approx_prob
                if self.stop_policy == "approx_bound":
                    idx = cursor - 1
                    bound_num += float(bound_lower_weights[idx])
                    bound_denom += float(bound_denom_deltas[idx])
                selector_stop_mass = self._selector_stop_mass(true_mass, approx_mass, bound_num, bound_denom)
                if budget is not None and len(selected) >= int(budget):
                    break
            selected = unique_tokens(selected, context_len=state.scores.shape[0])
            selector_stop_mass = self._selector_stop_mass(true_mass, approx_mass, bound_num, bound_denom)
            exact_mb = kv_read_bytes(len(selected), self.trace.head_dim, self.attn_key_bytes, self.value_bytes) / (
                1024.0 * 1024.0
            )
            total_mb = selector_cost.mb() + charged_update_cost.mb() + exact_mb
            choices.append(
                {
                    "reached": selector_stop_mass >= stop_target,
                    "total_mb": total_mb,
                    "nprobe": int(nprobe),
                    "selected": selected,
                    "ranked": ranked,
                    "routed_added": routed_added,
                    "selector_cost": selector_cost,
                    "mass": true_mass,
                    "approx_mass": approx_mass,
                    "selector_stop_mass": selector_stop_mass,
                }
            )

        reachable = [choice for choice in choices if choice["reached"]]
        choice = (
            min(reachable, key=lambda item: item["total_mb"])
            if reachable
            else max(choices, key=lambda item: item["selector_stop_mass"])
        )
        self._ensure_value_pq_sidecars(state.kv_head, index)
        value_page_starts = np.asarray([int(page["token_start"]) for page in index.pages], dtype=np.int64)
        value_page_sizes = np.asarray([int(page["size"]) for page in index.pages], dtype=np.int64)
        if len(index.pages):
            value_page_means = np.stack(
                [
                    self.trace.values[state.kv_head, int(page["token_start"]) : int(page["token_start"]) + int(page["size"])].mean(axis=0)
                    for page in index.pages
                ],
                axis=0,
            ).astype(np.float32, copy=False)
        else:
            value_page_means = np.zeros((0, int(self.trace.head_dim)), dtype=np.float32)
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
                "stop_policy": str(self.stop_policy),
                "approx_mass_margin": float(self._effective_approx_margin(state)),
                "selector_approx_mass": float(choice.get("approx_mass", 0.0)),
                "selector_stop_mass": float(choice.get("selector_stop_mass", 0.0)),
                "guard_tokens": int(choice.get("guard_tokens", 0)),
                "probe_tokens": int(choice.get("probe_tokens", 0)),
                "probe_bands": int(self.probe_bands),
                "probe_residual_quantile": float(self.probe_residual_quantile),
                "probe_residual_epsilon": float(choice.get("probe_residual_epsilon", 0.0)),
                "probe_confidence_z": float(self.probe_confidence_z),
                "budget_fraction": float(self.budget_fraction),
                "budget_rule": str(self.budget_rule),
                "dynamic_budget": int(choice.get("dynamic_budget", 0)),
                "pages": int(len(index.pages)),
                "pending_tokens": int(len(pending)),
                "router_groups": int(len(index.groups)) if self.routed else 0,
                "accounting_mode": str(self.accounting_mode),
                "online_update_modeled": bool(self.accounting_mode == "online"),
                "online_update_cumulative_MB": (
                    _event_bytes_mb(index.update_events_total) if self.accounting_mode == "online" else 0.0
                ),
                "online_update_indexed_tokens": int(index.total_update_steps) if self.accounting_mode == "online" else 0,
                "approx_scores": choice.get("approx_scores_by_token"),
                "value_page_starts": value_page_starts,
                "value_page_sizes": value_page_sizes,
                "value_page_means": value_page_means,
                "key_pq_codebooks": [np.asarray(page["codebooks"], dtype=np.float32) for page in index.pages],
                "key_pq_page_codes": [np.asarray(page["codes"], dtype=np.uint16) for page in index.pages],
                "key_pq_page_perms": [np.asarray(page.get("perm", []), dtype=np.uint16) for page in index.pages],
                "key_pq_codebook_bytes_per_page": int(index._pq_codebook_bytes_per_page()),
                "key_pq_code_bytes_per_token": int(self._pq_code_bytes_per_token()),
                "key_residual_pq_codebooks": [
                    codebooks for codebooks, _codes in self.key_residual_pq_sidecars[state.kv_head]
                ],
                "key_residual_pq_page_codes": [
                    codes for _codebooks, codes in self.key_residual_pq_sidecars[state.kv_head]
                ],
                "key_residual_pq_codebook_bytes_per_page": int(self._value_pq_codebook_bytes_per_page()),
                "key_residual_pq_code_bytes_per_token": int(self._pq_code_bytes_per_token()),
                "key_residual_pq_sidecar_update_cumulative_bytes": float(
                    self.key_residual_pq_sidecar_update_bytes[state.kv_head]
                ),
                "value_pq_codebooks": list(self.value_pq_sidecars[state.kv_head]),
                "value_pq_page_codes": [np.asarray(page["codes"], dtype=np.uint16) for page in index.pages],
                "value_pq_codebook_bytes_per_page": int(self._value_pq_codebook_bytes_per_page()),
                "value_pq_code_bytes_per_token": int(self._pq_code_bytes_per_token()),
                "value_pq_sidecar_update_cumulative_bytes": float(
                    self.value_pq_sidecar_update_bytes[state.kv_head]
                ),
                "value_vpq_codebooks": [codebooks for codebooks, _codes in self.value_vpq_sidecars[state.kv_head]],
                "value_vpq_page_codes": [codes for _codebooks, codes in self.value_vpq_sidecars[state.kv_head]],
                "value_vpq_codebook_bytes_per_page": int(self._value_pq_codebook_bytes_per_page()),
                "value_vpq_code_bytes_per_token": int(self._value_pq_code_bytes_per_token()),
                "value_vpq_subvecs": int(self._value_pq_subvecs()),
                "value_vpq_subbits": int(self._value_pq_subbits()),
                "value_vpq_sidecar_update_cumulative_bytes": float(
                    self.value_vpq_sidecar_update_bytes[state.kv_head]
                ),
            },
        )

    def _selector_stop_mass(self, true_mass: float, approx_mass: float, bound_num: float = 0.0, bound_denom: float = 1.0) -> float:
        if self.stop_policy == "approx_bound":
            return float(bound_num) / max(float(bound_denom), 1e-20)
        if self.stop_policy == "approx_mass":
            return float(approx_mass)
        return float(true_mass)

    def _fixed_fraction_choice(
        self,
        *,
        state: QueryState,
        selected: list[int],
        ranked: list[int],
        selector_cost: CostTrace,
        nprobe: int,
        target: float,
        approx_scores_by_token: np.ndarray | None = None,
    ) -> dict:
        fraction = float(self.budget_fraction)
        if not (0.0 < fraction <= 1.0):
            raise ValueError(f"fixed_fraction requires budget_fraction in (0, 1], got {fraction}")
        keep = int(np.ceil(fraction * float(state.scores.shape[0])))
        selected_out = list(selected)
        remaining = max(0, int(keep) - len(selected_out))
        if remaining > 0:
            selected_out.extend(int(tok) for tok in ranked[:remaining])
        selected_out = unique_tokens(selected_out, context_len=state.scores.shape[0])
        mass = float(state.probs[np.asarray(selected_out, dtype=np.int64)].sum()) if selected_out else 0.0
        exact_mb = kv_read_bytes(len(selected_out), self.trace.head_dim, self.attn_key_bytes, self.value_bytes) / (
            1024.0 * 1024.0
        )
        return {
            "reached": True,
            "total_mb": selector_cost.mb() + exact_mb,
            "nprobe": int(nprobe),
            "selected": selected_out,
            "ranked": ranked,
            "routed_added": selected_out[len(selected) :],
            "selector_cost": selector_cost,
            "mass": mass,
            "approx_mass": 0.0,
            "selector_stop_mass": mass,
            "approx_scores_by_token": approx_scores_by_token,
        }

    def _fixed_budget_choice(
        self,
        *,
        state: QueryState,
        selected: list[int],
        ranked: list[int],
        selector_cost: CostTrace,
        nprobe: int,
        target: float,
        approx_scores_by_token: np.ndarray | None = None,
    ) -> dict:
        dynamic_budget = _budget_from_rule(self.budget_rule, state.scores.shape[0])
        selected_out = list(selected)
        added = 0
        if dynamic_budget > 0:
            added_tokens = [int(tok) for tok in ranked[:dynamic_budget]]
            selected_out.extend(added_tokens)
            added = len(added_tokens)
        selected_out = unique_tokens(selected_out, context_len=state.scores.shape[0])
        mass = float(state.probs[np.asarray(selected_out, dtype=np.int64)].sum()) if selected_out else 0.0
        exact_mb = kv_read_bytes(len(selected_out), self.trace.head_dim, self.attn_key_bytes, self.value_bytes) / (
            1024.0 * 1024.0
        )
        return {
            "reached": int(added) >= int(dynamic_budget),
            "total_mb": selector_cost.mb() + exact_mb,
            "nprobe": int(nprobe),
            "selected": selected_out,
            "ranked": ranked,
            "routed_added": selected_out[len(selected) :],
            "selector_cost": selector_cost,
            "mass": mass,
            "approx_mass": 0.0,
            "selector_stop_mass": mass,
            "dynamic_budget": int(dynamic_budget),
            "dynamic_added": int(added),
            "approx_scores_by_token": approx_scores_by_token,
        }

    def _effective_approx_margin(self, state: QueryState) -> float:
        if not self.approx_mass_margin_schedule:
            return float(self.approx_mass_margin)
        decode_tokens = int(state.decode_tokens)
        margin = float(self.approx_mass_margin_schedule[-1][1])
        for max_decode, candidate_margin in self.approx_mass_margin_schedule:
            margin = float(candidate_margin)
            if decode_tokens <= int(max_decode):
                break
        return margin

    def _selection_fullscan_verify_window_scored(
        self,
        state: QueryState,
        index: PagedLocalPQIndex,
        *,
        target: float,
        base_tokens: list[int],
    ) -> tuple[np.ndarray, np.ndarray, EventBytes]:
        ranked, ranked_scores, events = index.selection_fullscan_scored(state.query)
        if ranked.size == 0:
            return ranked, ranked_scores, events
        cutoff = self._approx_pool_count_for_residual(state, base_tokens, ranked_scores, target)
        window = max(0, int(self.exact_verify_window) or int(self.verify_proj_window))
        start_pos = max(0, int(cutoff) - window)
        end_pos = min(int(ranked.shape[0]), int(cutoff) + window)
        if end_pos <= start_pos:
            return ranked, ranked_scores, events
        verify_tokens = ranked[start_pos:end_pos].astype(np.int64, copy=False)
        scale = 1.0 / float(np.sqrt(float(state.query.shape[-1])))
        corrected = ranked_scores.astype(np.float32, copy=True)
        if int(self.exact_verify_window) > 0:
            events.read(
                "verify_exact_keys",
                int(verify_tokens.size) * int(self.trace.head_dim) * int(self.attn_key_bytes),
            )
            corrected[start_pos:end_pos] = (
                np.asarray([float(state.scores[int(tok)]) for tok in verify_tokens], dtype=np.float32) / float(scale)
            )
        else:
            if int(index.verify_proj_dim) <= 0:
                return ranked, ranked_scores, events
            events.read("verify_proj_matrix", index._verify_proj_matrix_bytes())
            q_proj = state.query.astype(np.float32, copy=False) @ index.verify_proj_matrix
            projected_scores = np.zeros((int(verify_tokens.size),), dtype=np.float32)
            rel_positions = np.arange(int(verify_tokens.size), dtype=np.int64)
            for page in index.pages:
                if "verify_proj" not in page:
                    continue
                size = int(page["size"])
                start = int(page["token_start"])
                rows_mask = (verify_tokens >= start) & (verify_tokens < start + size)
                if not np.any(rows_mask):
                    continue
                rows = (verify_tokens[rows_mask] - start).astype(np.int64, copy=False)
                events.read("verify_proj_keys", index._verify_proj_bytes(int(rows.size)))
                projected_scores[rel_positions[rows_mask]] = page["verify_proj"][rows] @ q_proj
            corrected[start_pos:end_pos] = projected_scores
        order = np.argsort(-corrected, kind="stable")
        return ranked[order].astype(np.int64, copy=False), corrected[order].astype(np.float32, copy=False), events

    def _selection_fullscan_residual_scored(
        self,
        state: QueryState,
        index: PagedLocalPQIndex,
        *,
        target: float,
        base_tokens: list[int],
    ) -> tuple[np.ndarray, np.ndarray, EventBytes]:
        """PQ full scan plus exact residual on the largest query channels.

        The selector still ranks every sealed token, but the correction reads
        only a few real-K channels. This keeps the full-scan access regular and
        explicitly charges the extra key-channel traffic.
        """
        from benchmark.online_ivfpq_simulator import pq_scores

        events = EventBytes()
        if not index.pages:
            return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float32), events
        q = state.query.astype(np.float32, copy=False)
        rank = min(max(1, int(self.sparq_residual_rank)), int(q.shape[0]))
        dims = np.argsort(-np.abs(q), kind="stable")[:rank].astype(np.int64, copy=False)
        subdim = int(index.dim) // int(index.subvecs)
        events.read("pq_residual_dims", int(rank) * int(self.sparq_index_bytes))
        token_parts = []
        score_parts = []
        for page in index.pages:
            codebooks = page["codebooks"]
            codes = page["codes"]
            size = int(page["size"])
            if size <= 0:
                continue
            events.read("page_pq_codebooks", index._pq_codebook_bytes_per_page())
            events.read("page_pq_codes", index._pq_code_bytes(size))
            scores = pq_scores(q, codebooks, codes).astype(np.float32, copy=False)
            tokens = int(page["token_start"]) + np.arange(size, dtype=np.int64)
            token_parts.append(tokens)
            score_parts.append(scores)
        tokens = np.concatenate(token_parts) if token_parts else np.empty((0,), dtype=np.int64)
        scores = np.concatenate(score_parts) if score_parts else np.empty((0,), dtype=np.float32)
        order = np.argsort(-scores, kind="stable")
        ranked = tokens[order].astype(np.int64, copy=False)
        ranked_scores = scores[order].astype(np.float32, copy=False)
        approx_cutoff = self._approx_pool_count_for_residual(state, base_tokens, ranked_scores, target)
        window = max(0, int(self.sparq_residual_window))
        if window > 0:
            start_pos = max(0, int(approx_cutoff) - window)
            end_pos = min(int(ranked.shape[0]), int(approx_cutoff) + window)
        elif self.sparq_residual_pool:
            start_pos = 0
            end_pos = int(approx_cutoff)
        else:
            start_pos = 0
            end_pos = int(ranked.shape[0])
        start_pos = min(max(0, int(start_pos)), int(ranked.shape[0]))
        end_pos = min(max(start_pos, int(end_pos)), int(ranked.shape[0]))
        if end_pos <= start_pos:
            return ranked, ranked_scores, events
        correction_tokens = ranked[start_pos:end_pos]
        rank_positions = np.arange(start_pos, end_pos, dtype=np.int64)
        corrected = ranked_scores.astype(np.float32, copy=True)
        for page in index.pages:
            size = int(page["size"])
            if size <= 0:
                continue
            start = int(page["token_start"])
            rows_mask = (correction_tokens >= start) & (correction_tokens < start + size)
            if not np.any(rows_mask):
                continue
            rows = (correction_tokens[rows_mask] - start).astype(np.int64, copy=False)
            positions = rank_positions[rows_mask]
            codebooks = page["codebooks"]
            codes = page["codes"][rows]
            events.read("pq_residual_key_channels", int(rows.size) * int(rank) * int(self.attn_key_bytes))
            correction = np.zeros((int(rows.size),), dtype=np.float32)
            for dim in dims.tolist():
                sub = int(dim) // subdim
                off = int(dim) - sub * subdim
                pq_dim = codebooks[sub, codes[:, sub], off].astype(np.float32, copy=False)
                exact_dim = state.keys[correction_tokens[rows_mask], int(dim)].astype(np.float32, copy=False)
                correction += (exact_dim - pq_dim) * float(q[int(dim)])
            corrected[positions] += correction
        corrected_order = np.argsort(-corrected, kind="stable")
        return ranked[corrected_order].astype(np.int64, copy=False), corrected[corrected_order].astype(np.float32, copy=False), events

    def _approx_pool_count_for_residual(
        self,
        state: QueryState,
        base_tokens: list[int],
        ranked_scores: np.ndarray,
        target: float,
    ) -> int:
        if ranked_scores.size == 0:
            return 0
        scale = 1.0 / float(np.sqrt(float(state.query.shape[-1])))
        base_scores = [float(state.scores[int(tok)]) for tok in unique_tokens(base_tokens, context_len=state.scores.shape[0])]
        approx_scores = (ranked_scores.astype(np.float32, copy=False) * float(scale)).astype(np.float32, copy=False)
        all_scores = base_scores + approx_scores.tolist()
        max_score = max(all_scores)
        base_weight = float(sum(np.exp(float(score) - max_score) for score in base_scores))
        approx_weights = np.exp(approx_scores - max_score).astype(np.float64, copy=False)
        denom = max(base_weight + float(np.sum(approx_weights)), 1e-20)
        cumulative = base_weight + np.cumsum(approx_weights, dtype=np.float64)
        reached = np.nonzero((cumulative / denom) >= float(target))[0]
        if reached[0:1].size:
            return int(reached[0]) + 1
        return int(ranked_scores.size)

    def _approx_distribution(
        self,
        state: QueryState,
        selected: list[int],
        ranked_approx_scores: list[float],
    ) -> tuple[float, list[float]]:
        if self.stop_policy != "approx_mass":
            return 0.0, [0.0 for _ in ranked_approx_scores]
        scale = 1.0 / float(np.sqrt(float(state.query.shape[-1])))
        exact_scores = [float(state.scores[int(tok)]) for tok in selected]
        approx_scores = [float(score) * scale for score in ranked_approx_scores]
        all_scores = exact_scores + approx_scores
        if not all_scores:
            return 0.0, []
        max_score = max(all_scores)
        exact_weights = [float(np.exp(float(score) - max_score)) for score in exact_scores]
        approx_weights = [float(np.exp(float(score) - max_score)) for score in approx_scores]
        denom = max(float(sum(exact_weights) + sum(approx_weights)), 1e-20)
        initial_mass = float(sum(exact_weights) / denom) if exact_weights else 0.0
        return initial_mass, [float(weight / denom) for weight in approx_weights]

    def _guard_choice(
        self,
        *,
        state: QueryState,
        selected: list[int],
        ranked: list[int],
        ranked_approx_scores: list[float],
        selector_cost: CostTrace,
        nprobe: int,
        target: float,
        approx_scores_by_token: np.ndarray | None = None,
    ) -> dict:
        approx_mass, ranked_approx_probs = self._approx_distribution(state, selected, ranked_approx_scores)
        cursor = 0
        while approx_mass < target and cursor < len(ranked):
            approx_mass += float(ranked_approx_probs[cursor]) if cursor < len(ranked_approx_probs) else 0.0
            cursor += 1
        guard_width = max(0, int(self.guard_tokens))
        guard_start = max(0, cursor - guard_width)
        guard_end = min(len(ranked), cursor + guard_width)
        prefix_tokens = ranked[:guard_start]
        guard_tokens = ranked[guard_start:guard_end]
        selector_cost.read(
            "selector",
            "guard_exact_keys",
            len(guard_tokens) * int(self.trace.head_dim) * int(self.attn_key_bytes),
        )

        scale = 1.0 / float(np.sqrt(float(state.query.shape[-1])))
        base_scores = [float(state.scores[int(tok)]) for tok in selected]
        prefix_scores = [float(score) * scale for score in ranked_approx_scores[:guard_start]]
        guard_scores = [float(state.scores[int(tok)]) for tok in guard_tokens]
        tail_scores = [float(score) * scale for score in ranked_approx_scores[guard_end:]]
        all_scores = base_scores + prefix_scores + guard_scores + tail_scores
        if not all_scores:
            selector_stop_mass = 0.0
            chosen = selected
        else:
            max_score = max(all_scores)
            denom = max(float(sum(np.exp(float(score) - max_score) for score in all_scores)), 1e-20)
            mass_num = float(sum(np.exp(float(score) - max_score) for score in base_scores))
            chosen = list(selected) + [int(tok) for tok in prefix_tokens]
            mass_num += float(sum(np.exp(float(score) - max_score) for score in prefix_scores))
            guard_order = sorted(guard_tokens, key=lambda tok: float(state.scores[int(tok)]), reverse=True)
            for tok in guard_order:
                if mass_num / denom >= target:
                    break
                chosen.append(int(tok))
                mass_num += float(np.exp(float(state.scores[int(tok)]) - max_score))
            if mass_num / denom < target:
                for tok, score in zip(ranked[guard_end:], tail_scores, strict=False):
                    if mass_num / denom >= target:
                        break
                    chosen.append(int(tok))
                    mass_num += float(np.exp(float(score) - max_score))
            selector_stop_mass = float(mass_num / denom)

        chosen = unique_tokens(chosen, context_len=state.scores.shape[0])
        exact_mb = kv_read_bytes(len(chosen), self.trace.head_dim, self.attn_key_bytes, self.value_bytes) / (
            1024.0 * 1024.0
        )
        return {
            "reached": selector_stop_mass >= target,
            "total_mb": selector_cost.mb() + exact_mb,
            "nprobe": int(nprobe),
            "selected": chosen,
            "ranked": ranked,
            "routed_added": chosen[len(selected):],
            "selector_cost": selector_cost,
            "mass": 0.0,
            "approx_mass": approx_mass,
            "selector_stop_mass": selector_stop_mass,
            "guard_tokens": len(guard_tokens),
            "approx_scores_by_token": approx_scores_by_token,
        }

    def _probe_choice(
        self,
        *,
        state: QueryState,
        selected: list[int],
        ranked: list[int],
        ranked_approx_scores: list[float],
        selector_cost: CostTrace,
        nprobe: int,
        target: float,
        approx_scores_by_token: np.ndarray | None = None,
    ) -> dict:
        scale = 1.0 / float(np.sqrt(float(state.query.shape[-1])))
        calibrated_scores, probe_count, residual_epsilon, probe_pos, probe_exact_scores = self._probe_calibrated_scores(
            state,
            ranked,
            ranked_approx_scores,
            selector_cost,
            scale,
        )
        if self.stop_policy == "approx_probe_ratio":
            return self._probe_ratio_choice(
                state=state,
                selected=selected,
                ranked=ranked,
                calibrated_scores=calibrated_scores,
                probe_count=probe_count,
                residual_epsilon=residual_epsilon,
                probe_pos=probe_pos,
                probe_exact_scores=probe_exact_scores,
                selector_cost=selector_cost,
                nprobe=nprobe,
                target=target,
                approx_scores_by_token=approx_scores_by_token,
            )
        if self.stop_policy == "approx_probe_bands":
            return self._probe_bands_choice(
                state=state,
                selected=selected,
                ranked=ranked,
                calibrated_scores=calibrated_scores,
                probe_count=probe_count,
                residual_epsilon=residual_epsilon,
                probe_pos=probe_pos,
                probe_exact_scores=probe_exact_scores,
                selector_cost=selector_cost,
                nprobe=nprobe,
                target=target,
            )
        base_scores = [float(state.scores[int(tok)]) for tok in selected]
        conservative = self.stop_policy == "approx_probe_ucb"
        all_scores = base_scores + [
            float(score) + (float(residual_epsilon) if conservative else 0.0)
            for score in calibrated_scores
        ]
        if not all_scores:
            chosen = selected
            selector_stop_mass = 0.0
        else:
            max_score = max(all_scores)
            base_weights = [float(np.exp(float(score) - max_score)) for score in base_scores]
            mass_num = float(sum(base_weights))
            chosen = list(selected)
            if conservative:
                upper_weights = [
                    float(np.exp(float(score) + float(residual_epsilon) - max_score))
                    for score in calibrated_scores
                ]
                lower_weights = [
                    float(np.exp(float(score) - float(residual_epsilon) - max_score))
                    for score in calibrated_scores
                ]
                denom = max(float(sum(base_weights) + sum(upper_weights)), 1e-20)
                for tok, upper_weight, lower_weight in zip(ranked, upper_weights, lower_weights, strict=False):
                    if mass_num / denom >= target:
                        break
                    chosen.append(int(tok))
                    mass_num += float(lower_weight)
                    denom += float(lower_weight) - float(upper_weight)
            else:
                ranked_weights = [float(np.exp(float(score) - max_score)) for score in calibrated_scores]
                denom = max(float(sum(base_weights) + sum(ranked_weights)), 1e-20)
                for tok, weight in zip(ranked, ranked_weights, strict=False):
                    if mass_num / denom >= target:
                        break
                    chosen.append(int(tok))
                    mass_num += float(weight)
            selector_stop_mass = float(mass_num / denom)
        chosen = unique_tokens(chosen, context_len=state.scores.shape[0])
        exact_mb = kv_read_bytes(len(chosen), self.trace.head_dim, self.attn_key_bytes, self.value_bytes) / (
            1024.0 * 1024.0
        )
        return {
            "reached": selector_stop_mass >= target,
            "total_mb": selector_cost.mb() + exact_mb,
            "nprobe": int(nprobe),
            "selected": chosen,
            "ranked": ranked,
            "routed_added": chosen[len(selected):],
            "selector_cost": selector_cost,
            "mass": 0.0,
            "approx_mass": selector_stop_mass,
            "selector_stop_mass": selector_stop_mass,
            "probe_tokens": probe_count,
            "probe_residual_epsilon": float(residual_epsilon),
            "approx_scores_by_token": approx_scores_by_token,
        }

    def _probe_ratio_choice(
        self,
        *,
        state: QueryState,
        selected: list[int],
        ranked: list[int],
        calibrated_scores: list[float],
        probe_count: int,
        residual_epsilon: float,
        probe_pos: np.ndarray,
        probe_exact_scores: np.ndarray,
        selector_cost: CostTrace,
        nprobe: int,
        target: float,
    ) -> dict:
        base_scores = [float(state.scores[int(tok)]) for tok in selected]
        if not calibrated_scores:
            chosen = unique_tokens(selected, context_len=state.scores.shape[0])
            selector_stop_mass = 0.0
        else:
            cal = np.asarray(calibrated_scores, dtype=np.float64)
            all_scores = base_scores + cal.tolist()
            max_score = max(all_scores)
            base_weight = float(sum(np.exp(float(score) - max_score) for score in base_scores))
            approx_weights = np.exp(cal - max_score).astype(np.float64, copy=False)
            approx_prefix = np.concatenate(([0.0], np.cumsum(approx_weights, dtype=np.float64)))
            total_approx = float(approx_prefix[-1])
            probe_pos = np.asarray(probe_pos, dtype=np.int64)
            valid_probe = (probe_pos >= 0) & (probe_pos < approx_weights.size)
            probe_pos = probe_pos[valid_probe]
            probe_exact_scores = np.asarray(probe_exact_scores, dtype=np.float64)[valid_probe]
            if probe_pos.size:
                probe_ratios = np.exp(probe_exact_scores - cal[probe_pos]).astype(np.float64, copy=False)
                order = np.argsort(probe_pos, kind="stable")
                probe_pos = probe_pos[order]
                probe_ratios = probe_ratios[order]
            else:
                probe_ratios = np.asarray([], dtype=np.float64)

            def ratio_bounds(n: int, sum_: float, sumsq: float, fallback: tuple[float, float]) -> tuple[float, float]:
                if n < 2:
                    return fallback
                mean = float(sum_) / float(n)
                var = max(float(sumsq) / float(n) - mean * mean, 0.0)
                se = float(np.sqrt(var / float(n)))
                z = max(0.0, float(self.probe_confidence_z))
                return max(0.0, mean - z * se), max(0.0, mean + z * se)

            global_fallback = ratio_bounds(
                int(probe_ratios.size),
                float(np.sum(probe_ratios)),
                float(np.sum(probe_ratios * probe_ratios)),
                (1.0, 1.0),
            )
            sel_n = 0
            sel_sum = 0.0
            sel_sumsq = 0.0
            tail_n = int(probe_ratios.size)
            tail_sum = float(np.sum(probe_ratios))
            tail_sumsq = float(np.sum(probe_ratios * probe_ratios))
            next_probe = 0
            chosen_count = 0
            selector_stop_mass = 0.0
            while chosen_count < len(ranked):
                while next_probe < probe_pos.size and int(probe_pos[next_probe]) < chosen_count:
                    ratio = float(probe_ratios[next_probe])
                    sel_n += 1
                    sel_sum += ratio
                    sel_sumsq += ratio * ratio
                    tail_n -= 1
                    tail_sum -= ratio
                    tail_sumsq -= ratio * ratio
                    next_probe += 1
                selected_approx = float(approx_prefix[chosen_count])
                tail_approx = total_approx - selected_approx
                sel_lower, _sel_upper = ratio_bounds(sel_n, sel_sum, sel_sumsq, global_fallback)
                _tail_lower, tail_upper = ratio_bounds(tail_n, tail_sum, tail_sumsq, global_fallback)
                numerator = base_weight + selected_approx * sel_lower
                denominator = max(numerator + tail_approx * tail_upper, 1e-20)
                selector_stop_mass = float(numerator / denominator)
                if selector_stop_mass >= target:
                    break
                chosen_count += 1
            chosen = unique_tokens(list(selected) + [int(tok) for tok in ranked[:chosen_count]], context_len=state.scores.shape[0])
        exact_mb = kv_read_bytes(len(chosen), self.trace.head_dim, self.attn_key_bytes, self.value_bytes) / (
            1024.0 * 1024.0
        )
        return {
            "reached": selector_stop_mass >= target,
            "total_mb": selector_cost.mb() + exact_mb,
            "nprobe": int(nprobe),
            "selected": chosen,
            "ranked": ranked,
            "routed_added": chosen[len(selected):],
            "selector_cost": selector_cost,
            "mass": 0.0,
            "approx_mass": selector_stop_mass,
            "selector_stop_mass": selector_stop_mass,
            "probe_tokens": probe_count,
            "probe_residual_epsilon": float(residual_epsilon),
            "approx_scores_by_token": approx_scores_by_token,
        }

    def _probe_bands_choice(
        self,
        *,
        state: QueryState,
        selected: list[int],
        ranked: list[int],
        calibrated_scores: list[float],
        probe_count: int,
        residual_epsilon: float,
        probe_pos: np.ndarray,
        probe_exact_scores: np.ndarray,
        selector_cost: CostTrace,
        nprobe: int,
        target: float,
        approx_scores_by_token: np.ndarray | None = None,
    ) -> dict:
        base_scores = [float(state.scores[int(tok)]) for tok in selected]
        if not calibrated_scores:
            chosen = unique_tokens(selected, context_len=state.scores.shape[0])
            selector_stop_mass = 0.0
        else:
            cal = np.asarray(calibrated_scores, dtype=np.float64)
            all_scores = base_scores + cal.tolist()
            max_score = max(all_scores)
            base_weight = float(sum(np.exp(float(score) - max_score) for score in base_scores))
            approx_weights = np.exp(cal - max_score).astype(np.float64, copy=False)
            n = int(approx_weights.size)
            bands = max(1, min(int(self.probe_bands), n))
            edges = np.linspace(0, n, num=bands + 1, dtype=np.int64)
            probe_pos = np.asarray(probe_pos, dtype=np.int64)
            valid_probe = (probe_pos >= 0) & (probe_pos < n)
            probe_pos = probe_pos[valid_probe]
            probe_exact_scores = np.asarray(probe_exact_scores, dtype=np.float64)[valid_probe]
            probe_ratios = (
                np.exp(probe_exact_scores - cal[probe_pos]).astype(np.float64, copy=False)
                if probe_pos.size
                else np.asarray([], dtype=np.float64)
            )
            global_lower, global_upper = self._ratio_bounds_from_values(probe_ratios, fallback=(1.0, 1.0))
            lower = np.full((bands,), global_lower, dtype=np.float64)
            upper = np.full((bands,), global_upper, dtype=np.float64)
            for band in range(bands):
                lo = int(edges[band])
                hi = int(edges[band + 1])
                mask = (probe_pos >= lo) & (probe_pos < hi)
                if np.any(mask):
                    lower[band], upper[band] = self._ratio_bounds_from_values(
                        probe_ratios[mask],
                        fallback=(global_lower, global_upper),
                    )
            band_prefixes = []
            band_totals = []
            for band in range(bands):
                lo = int(edges[band])
                hi = int(edges[band + 1])
                weights = approx_weights[lo:hi]
                band_prefixes.append(np.concatenate(([0.0], np.cumsum(weights, dtype=np.float64))))
                band_totals.append(float(np.sum(weights)))
            chosen_count = 0
            selector_stop_mass = 0.0
            while chosen_count <= n:
                selected_lower = base_weight
                tail_upper = 0.0
                for band in range(bands):
                    lo = int(edges[band])
                    hi = int(edges[band + 1])
                    in_band_selected = min(max(0, chosen_count - lo), hi - lo)
                    selected_approx = float(band_prefixes[band][in_band_selected])
                    tail_approx = float(band_totals[band] - selected_approx)
                    selected_lower += selected_approx * float(lower[band])
                    tail_upper += tail_approx * float(upper[band])
                denom = max(selected_lower + tail_upper, 1e-20)
                selector_stop_mass = float(selected_lower / denom)
                if selector_stop_mass >= target or chosen_count >= n:
                    break
                chosen_count += 1
            chosen = unique_tokens(list(selected) + [int(tok) for tok in ranked[:chosen_count]], context_len=state.scores.shape[0])
        exact_mb = kv_read_bytes(len(chosen), self.trace.head_dim, self.attn_key_bytes, self.value_bytes) / (
            1024.0 * 1024.0
        )
        return {
            "reached": selector_stop_mass >= target,
            "total_mb": selector_cost.mb() + exact_mb,
            "nprobe": int(nprobe),
            "selected": chosen,
            "ranked": ranked,
            "routed_added": chosen[len(selected):],
            "selector_cost": selector_cost,
            "mass": 0.0,
            "approx_mass": selector_stop_mass,
            "selector_stop_mass": selector_stop_mass,
            "probe_tokens": probe_count,
            "probe_residual_epsilon": float(residual_epsilon),
            "approx_scores_by_token": approx_scores_by_token,
        }

    def _ratio_bounds_from_values(self, values: np.ndarray, *, fallback: tuple[float, float]) -> tuple[float, float]:
        values = np.asarray(values, dtype=np.float64)
        if values.size < 2:
            return fallback
        mean = float(np.mean(values))
        var = max(float(np.mean((values - mean) * (values - mean))), 0.0)
        se = float(np.sqrt(var / float(values.size)))
        z = max(0.0, float(self.probe_confidence_z))
        return max(0.0, mean - z * se), max(0.0, mean + z * se)

    def _probe_calibrated_scores(
        self,
        state: QueryState,
        ranked: list[int],
        ranked_approx_scores: list[float],
        selector_cost: CostTrace,
        scale: float,
    ) -> tuple[list[float], int, float, np.ndarray, np.ndarray]:
        approx_scaled = np.asarray(ranked_approx_scores, dtype=np.float32) * float(scale)
        if approx_scaled.size == 0:
            return [], 0, 0.0, np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float64)
        probe_count = min(max(0, int(self.probe_tokens)), approx_scaled.size)
        if probe_count < 2:
            return approx_scaled.astype(np.float32, copy=False).tolist(), 0, 0.0, np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float64)
        probe_pos = np.unique(np.linspace(0, approx_scaled.size - 1, num=probe_count, dtype=np.int64))
        probe_tokens = [int(ranked[int(pos)]) for pos in probe_pos.tolist()]
        selector_cost.read(
            "selector",
            "probe_exact_keys",
            len(probe_tokens) * int(self.trace.head_dim) * int(self.attn_key_bytes),
        )
        x = approx_scaled[probe_pos].astype(np.float64, copy=False)
        y = np.asarray([float(state.scores[int(tok)]) for tok in probe_tokens], dtype=np.float64)
        x_mean = float(np.mean(x))
        y_mean = float(np.mean(y))
        x_var = float(np.mean((x - x_mean) * (x - x_mean)))
        if x_var <= 1e-12:
            slope = 1.0
        else:
            slope = float(np.mean((x - x_mean) * (y - y_mean)) / x_var)
        slope = min(4.0, max(0.25, slope))
        intercept = y_mean - slope * x_mean
        calibrated = approx_scaled.astype(np.float64, copy=False) * slope + intercept
        probe_pred = x * slope + intercept
        residual = np.abs(y - probe_pred)
        quantile = min(1.0, max(0.0, float(self.probe_residual_quantile)))
        residual_epsilon = float(np.quantile(residual, quantile)) if residual.size else 0.0
        return calibrated.astype(np.float32, copy=False).tolist(), int(len(probe_tokens)), residual_epsilon, probe_pos, y

    def _bound_distribution_state(
        self,
        state: QueryState,
        selected: list[int],
        ranked_approx_scores: list[float],
        ranked_approx_bounds: list[float],
    ) -> tuple[float, float, list[float], list[float]]:
        if self.stop_policy != "approx_bound":
            return 0.0, 1.0, [0.0 for _ in ranked_approx_scores], [0.0 for _ in ranked_approx_scores]
        scale = 1.0 / float(np.sqrt(float(state.query.shape[-1])))
        exact_scores = [float(state.scores[int(tok)]) for tok in selected]
        lower_scores = [(float(score) - float(bound)) * scale for score, bound in zip(ranked_approx_scores, ranked_approx_bounds, strict=False)]
        upper_scores = [(float(score) + float(bound)) * scale for score, bound in zip(ranked_approx_scores, ranked_approx_bounds, strict=False)]
        all_scores = exact_scores + lower_scores + upper_scores
        if not all_scores:
            return 0.0, 1.0, [], []
        max_score = max(all_scores)
        base_weight = float(sum(np.exp(float(score) - max_score) for score in exact_scores))
        lower_weights = [float(np.exp(float(score) - max_score)) for score in lower_scores]
        upper_weights = [float(np.exp(float(score) - max_score)) for score in upper_scores]
        denom = max(base_weight + float(sum(upper_weights)), 1e-20)
        denom_deltas = [float(lower - upper) for lower, upper in zip(lower_weights, upper_weights, strict=False)]
        return base_weight, denom, lower_weights, denom_deltas
