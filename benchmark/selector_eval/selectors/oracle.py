from __future__ import annotations

import numpy as np

from benchmark.selector_eval.costs.base import CostTrace, MB
from benchmark.selector_eval.data.trace import unique_tokens
from benchmark.selector_eval.metrics.tail_estimators import (
    _paper_tq_scores,
    _tq_code_bytes,
    _tq_reconstruct,
    _tq_reconstruct_product,
)
from benchmark.selector_eval.selectors.hybrid import PageSparQPostingsPQSelector, PageSparQPQSelector, PagedPQSparQRerankSelector
from benchmark.selector_eval.selectors.ivfpq import IVFPQSelector
from benchmark.selector_eval.selectors.magicpig import MagicPIGSelector
from benchmark.selector_eval.selectors.paged_pq import PagedPQSelector
from benchmark.selector_eval.selectors.pqcache import PQCacheFullScanSelector
from benchmark.selector_eval.selectors.retroinfer import RetroInferStyleSelector
from benchmark.selector_eval.selectors.retrievalattention import RetrievalAttentionGraphSelector
from benchmark.selector_eval.selectors.sparq import SparQSelector
from benchmark.selector_eval.selectors.turboquant import IVFTurboQuantSelector
from benchmark.selector_eval.selectors.base import QueryState, SelectionResult


class DenseSelector:
    name = "dense"

    def select(self, state: QueryState, *, target_mass: float | None = None, budget: int | None = None) -> SelectionResult:
        tokens = list(range(state.scores.shape[0]))
        return SelectionResult(
            algorithm=self.name,
            selected_tokens=tokens,
            candidate_tokens=tokens,
            metadata={"target_mass": target_mass, "budget": budget},
        )


class TopMassOracleSelector:
    name = "top_mass_oracle"

    def select(self, state: QueryState, *, target_mass: float | None = None, budget: int | None = None) -> SelectionResult:
        target = 1.0 if target_mass is None else float(target_mass)
        selected = unique_tokens(list(state.base_tokens), context_len=state.scores.shape[0])
        selected_set = set(selected)
        mass = float(state.probs[np.asarray(selected, dtype=np.int64)].sum()) if selected else 0.0

        dynamic = np.asarray([idx for idx in range(state.scores.shape[0]) if idx not in selected_set], dtype=np.int64)
        order = dynamic[np.argsort(-state.probs[dynamic], kind="stable")] if dynamic.size else dynamic
        cursor = 0
        while mass < target and cursor < order.size:
            tok = int(order[cursor])
            cursor += 1
            selected.append(tok)
            selected_set.add(tok)
            mass += float(state.probs[tok])
            if budget is not None and len(selected) >= int(budget):
                break

        return SelectionResult(
            algorithm=self.name,
            selected_tokens=unique_tokens(selected, context_len=state.scores.shape[0]),
            candidate_tokens=order.tolist(),
            metadata={"target_mass": target_mass, "budget": budget, "oracle_mass": mass},
        )


class TopFractionOracleSelector:
    """Diagnostic selector: exact top probability tokens at a fixed context fraction.

    This is not deployable. It is used to ask how attention-output metrics move
    when we deliberately drop the mass target and keep a fixed fraction.
    """

    def __init__(self, fraction: float, *, name: str | None = None) -> None:
        self.fraction = float(fraction)
        if not (0.0 < self.fraction <= 1.0):
            raise ValueError(f"fraction must be in (0, 1], got {fraction}")
        suffix = int(round(self.fraction * 100))
        self.name = name or f"top_fraction_oracle_f{suffix}"

    def select(self, state: QueryState, *, target_mass: float | None = None, budget: int | None = None) -> SelectionResult:
        selected = unique_tokens(list(state.base_tokens), context_len=state.scores.shape[0])
        selected_set = set(selected)
        keep = int(np.ceil(float(self.fraction) * float(state.scores.shape[0])))
        if budget is not None:
            keep = min(keep, int(budget))
        dynamic = np.asarray([idx for idx in range(state.scores.shape[0]) if idx not in selected_set], dtype=np.int64)
        order = dynamic[np.argsort(-state.probs[dynamic], kind="stable")] if dynamic.size else dynamic
        for tok in order[: max(0, keep - len(selected))].tolist():
            selected.append(int(tok))
        return SelectionResult(
            algorithm=self.name,
            selected_tokens=unique_tokens(selected, context_len=state.scores.shape[0]),
            candidate_tokens=order.tolist(),
            metadata={
                "target_mass": target_mass,
                "budget": budget,
                "fraction": self.fraction,
                "oracle_diagnostic": True,
            },
        )


def budget_from_rule(rule: str, context_len: int) -> int:
    """Parse compact budget schedules used by selector experiments.

    Rules are total dynamic-head token budgets, excluding static base tokens:
    ``sqrt_x4`` -> ``4 * sqrt(N)``, ``log_x256`` -> ``256 * log2(N)``,
    ``n067_x1`` -> ``N**0.67``, and ``k4096`` -> fixed budget.
    """
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


class TopBudgetOracleSelector:
    """Diagnostic exact-probability top-k selector with sublinear budget rules."""

    def __init__(self, rule: str, *, name: str | None = None) -> None:
        self.rule = str(rule)
        self.name = name or f"top_budget_oracle_{self.rule}"

    def select(self, state: QueryState, *, target_mass: float | None = None, budget: int | None = None) -> SelectionResult:
        selected = unique_tokens(list(state.base_tokens), context_len=state.scores.shape[0])
        selected_set = set(selected)
        dynamic_budget = budget_from_rule(self.rule, state.scores.shape[0])
        if budget is not None:
            dynamic_budget = min(dynamic_budget, max(0, int(budget) - len(selected)))
        dynamic = np.asarray([idx for idx in range(state.scores.shape[0]) if idx not in selected_set], dtype=np.int64)
        order = dynamic[np.argsort(-state.probs[dynamic], kind="stable")] if dynamic.size else dynamic
        selected.extend(int(tok) for tok in order[:dynamic_budget].tolist())
        return SelectionResult(
            algorithm=self.name,
            selected_tokens=unique_tokens(selected, context_len=state.scores.shape[0]),
            candidate_tokens=order.tolist(),
            metadata={
                "target_mass": target_mass,
                "budget": budget,
                "budget_rule": self.rule,
                "dynamic_budget": int(dynamic_budget),
                "oracle_diagnostic": True,
            },
        )


class TurboQuantScoreSelector:
    """Deployable proxy selector using TurboQuant-compressed K score ranking."""

    def __init__(
        self,
        *,
        key_bits: int,
        rule: str,
        product_residual: bool,
        key_bytes: int = 2,
        faithful: bool = False,
        name: str | None = None,
    ) -> None:
        self.key_bits = int(key_bits)
        self.rule = str(rule)
        self.product_residual = bool(product_residual)
        self.faithful = bool(faithful)
        prefix = ("papertqprod" if self.product_residual else "papertqmse") if self.faithful else ("tqprod" if self.product_residual else "tqmse")
        self.name = name or f"{prefix}_selector_k{self.key_bits}_budget_{self.rule}"
        self.key_bytes = int(key_bytes)

    def select(self, state: QueryState, *, target_mass: float | None = None, budget: int | None = None) -> SelectionResult:
        selected = unique_tokens(list(state.base_tokens), context_len=state.scores.shape[0])
        selected_set = set(selected)
        dynamic_budget = budget_from_rule(self.rule, state.scores.shape[0])
        if budget is not None:
            dynamic_budget = min(dynamic_budget, max(0, int(budget) - len(selected)))
        dynamic = np.asarray([idx for idx in range(state.scores.shape[0]) if idx not in selected_set], dtype=np.int64)
        approx_scores_by_token = np.full((state.scores.shape[0],), -np.inf, dtype=np.float32)
        if dynamic.size:
            if self.faithful:
                approx_scores = _paper_tq_scores(
                    state.keys[dynamic],
                    state.query,
                    self.key_bits,
                    product_residual=self.product_residual,
                )
            else:
                key_hat = (
                    _tq_reconstruct_product(state.keys[dynamic], self.key_bits)
                    if self.product_residual
                    else _tq_reconstruct(state.keys[dynamic], self.key_bits)[0]
                )
                approx_scores = key_hat.astype(np.float64, copy=False) @ state.query.astype(np.float64, copy=False)
            order_idx = np.argsort(-approx_scores, kind="stable")
            order = dynamic[order_idx]
            approx_scores_by_token[dynamic] = approx_scores.astype(np.float32, copy=False)
        else:
            order = dynamic
        selected.extend(int(tok) for tok in order[:dynamic_budget].tolist())

        dim = int(state.values.shape[-1])
        cost = CostTrace()
        cost.read("selector", "tq_selector_key_codes", _tq_code_bytes(dynamic.size, dim, self.key_bits))
        cost.read("selector", "tq_selector_key_norms", int(dynamic.size) * int(self.key_bytes))
        per_token_update = _tq_code_bytes(1, dim, self.key_bits) + int(self.key_bytes)
        if self.product_residual:
            cost.read("selector", "tq_selector_key_residual_signs", _tq_code_bytes(dynamic.size, dim, 1))
            cost.read("selector", "tq_selector_key_residual_scales", int(dynamic.size) * int(self.key_bytes))
            per_token_update += _tq_code_bytes(1, dim, 1) + int(self.key_bytes)
        cumulative_update_bytes = per_token_update * float(state.scores.shape[0])
        cost.write("online_update", "tq_selector_sidecar_write", cumulative_update_bytes)
        return SelectionResult(
            algorithm=self.name,
            selected_tokens=unique_tokens(selected, context_len=state.scores.shape[0]),
            candidate_tokens=order.tolist(),
            cost=cost,
            metadata={
                "target_mass": target_mass,
                "budget": budget,
                "budget_rule": self.rule,
                "dynamic_budget": int(dynamic_budget),
                "approx_scores": approx_scores_by_token,
                "accounting_mode": "online",
                "online_update_modeled": True,
                "online_update_cumulative_MB": cumulative_update_bytes / MB,
                "online_update_indexed_tokens": int(state.scores.shape[0]),
                "turboquant_selector": True,
                "turboquant_faithful_scorer": bool(self.faithful),
                "turboquant_key_bits": int(self.key_bits),
                "turboquant_product_residual": bool(self.product_residual),
            },
        )


def selector_from_name(name: str, **kwargs):
    normalized = str(name).strip().lower()
    if normalized in {"dense", "dense_oracle"}:
        return DenseSelector()
    if normalized in {"oracle", "top_mass_oracle", "dense_topmass"}:
        return TopMassOracleSelector()
    if normalized.startswith("top_fraction_oracle_f"):
        return TopFractionOracleSelector(_fraction_from_name(normalized, "top_fraction_oracle_f"), name=normalized)
    if normalized.startswith("top_budget_oracle_"):
        return TopBudgetOracleSelector(normalized.removeprefix("top_budget_oracle_"), name=normalized)
    if (
        normalized.startswith("tqmse_selector_k")
        or normalized.startswith("tqprod_selector_k")
        or normalized.startswith("papertqmse_selector_k")
        or normalized.startswith("papertqprod_selector_k")
    ):
        prefix, rule = normalized.split("_budget_", 1)
        key_bits = int(prefix.rsplit("_k", 1)[1])
        return TurboQuantScoreSelector(
            key_bits=key_bits,
            rule=rule,
            product_residual=normalized.startswith("tqprod") or normalized.startswith("papertqprod"),
            faithful=normalized.startswith("papertq"),
            key_bytes=int(kwargs.get("key_bytes", 2)),
            name=normalized,
        )
    if (
        normalized.startswith("ivftqmse_c")
        or normalized.startswith("ivftqprod_c")
        or normalized.startswith("ivfpapertqmse_c")
        or normalized.startswith("ivfpapertqprod_c")
    ):
        if kwargs.get("trace") is None:
            raise ValueError("ivftq selector requires trace=")
        prefix, rule = normalized.split("_budget_", 1)
        coarse_text, key_text = prefix.split("_k", 1)
        route_multiplier = 1.0
        if "_m" in key_text:
            key_text, route_text = key_text.split("_m", 1)
            route_multiplier = float(route_text.replace("p", "."))
        coarse_body = coarse_text.rsplit("_c", 1)[1]
        assignment_replicas = 1
        if "_r" in coarse_body:
            coarse_clusters_text, replica_text = coarse_body.split("_r", 1)
            coarse_clusters = int(coarse_clusters_text)
            assignment_replicas = int(replica_text)
        else:
            coarse_clusters = int(coarse_body)
        key_bits = int(key_text)
        opts = dict(kwargs.get("ivfpq_kwargs", {}))
        return IVFTurboQuantSelector(
            trace=kwargs["trace"],
            key_bits=key_bits,
            budget_rule=rule,
            coarse_clusters=coarse_clusters,
            coarse_iters=int(opts.get("coarse_iters", 3)),
            assignment_replicas=int(assignment_replicas),
            nprobes=tuple(opts.get("nprobes", (1, 2, 4, 8, 16, 32, 64, 128))),
            route_multiplier=float(route_multiplier),
            product_residual=normalized.startswith("ivftqprod") or normalized.startswith("ivfpapertqprod"),
            faithful_scorer=normalized.startswith("ivfpapertq"),
            static_prefix=int(opts.get("static_prefix", 128)),
            static_suffix=int(opts.get("static_suffix", 128)),
            seed=int(opts.get("seed", 2025)),
            score_key_bytes=int(opts.get("score_key_bytes", 4)),
            attn_key_bytes=int(opts.get("attn_key_bytes", 2)),
            value_bytes=int(opts.get("value_bytes", 2)),
            edge_index_bytes=int(opts.get("edge_index_bytes", 4)),
            graph_offset_bytes=int(opts.get("graph_offset_bytes", 4)),
            accounting_mode="online",
            display_name=normalized,
        )
    if normalized in {"pqcache", "pqcache_full_scan", "full_pq", "pqcache_full_scan_online", "pqcache_online_proxy"}:
        return PQCacheFullScanSelector(name="pqcache_full_scan_online_proxy", charge_online_update=True)
    if normalized in {"pqcache_snapshot", "pqcache_full_scan_snapshot", "full_pq_snapshot"}:
        return PQCacheFullScanSelector(name="pqcache_full_scan_snapshot", charge_online_update=False)
    if normalized in {"retroinfer", "retroinfer_style", "retroinfer_snapshot", "retroinfer_style_snapshot"}:
        opts = dict(kwargs.get("retroinfer_kwargs", {}))
        opts["accounting_mode"] = "snapshot"
        return RetroInferStyleSelector(**opts)
    if normalized in {"retroinfer_online", "retroinfer_online_proxy", "retroinfer_style_online_proxy"}:
        opts = dict(kwargs.get("retroinfer_kwargs", {}))
        opts["accounting_mode"] = "online_proxy"
        return RetroInferStyleSelector(**opts)
    if normalized in {"retrievalattention", "retrievalattention_style", "retrievalattention_graph", "ra_graph"}:
        if kwargs.get("trace") is None:
            raise ValueError("retrievalattention_graph requires trace=")
        return RetrievalAttentionGraphSelector(trace=kwargs["trace"], **kwargs.get("retrievalattention_kwargs", {}))
    if normalized in {"paged_local_pq", "local_paged_pq", "paged_local_pq_online"}:
        if kwargs.get("trace") is None:
            raise ValueError("paged_local_pq requires trace=")
        opts = dict(kwargs.get("paged_kwargs", {}))
        opts["accounting_mode"] = "online"
        if normalized.endswith("_online"):
            opts["display_name"] = "paged_local_pq_online"
        return PagedPQSelector(trace=kwargs["trace"], routed=False, **opts)
    if normalized in {"paged_local_pq_snapshot", "local_paged_pq_snapshot"}:
        if kwargs.get("trace") is None:
            raise ValueError("paged_local_pq_snapshot requires trace=")
        opts = dict(kwargs.get("paged_kwargs", {}))
        opts["accounting_mode"] = "snapshot"
        opts["display_name"] = "paged_local_pq_snapshot"
        return PagedPQSelector(trace=kwargs["trace"], routed=False, **opts)
    if normalized == "paged_local_pq_approx" or normalized.startswith("paged_local_pq_approx_mbp"):
        if kwargs.get("trace") is None:
            raise ValueError("paged_local_pq_approx requires trace=")
        opts = dict(kwargs.get("paged_kwargs", {}))
        opts["accounting_mode"] = "online"
        opts["stop_policy"] = "approx_mass"
        opts["approx_mass_margin"] = _margin_from_mbp_name(normalized, "paged_local_pq_approx")
        opts["display_name"] = normalized
        return PagedPQSelector(trace=kwargs["trace"], routed=False, **opts)
    if normalized == "paged_local_pq_approx_sched_v1":
        if kwargs.get("trace") is None:
            raise ValueError("paged_local_pq_approx_sched_v1 requires trace=")
        opts = dict(kwargs.get("paged_kwargs", {}))
        opts["accounting_mode"] = "online"
        opts["stop_policy"] = "approx_mass"
        opts["approx_mass_margin_schedule"] = (
            (1000, 0.0100),
            (64000, 0.0075),
            (128000, 0.0008),
        )
        opts["display_name"] = normalized
        return PagedPQSelector(trace=kwargs["trace"], routed=False, **opts)
    if normalized == "paged_local_pq_approx_sched_v2":
        if kwargs.get("trace") is None:
            raise ValueError("paged_local_pq_approx_sched_v2 requires trace=")
        opts = dict(kwargs.get("paged_kwargs", {}))
        opts["accounting_mode"] = "online"
        opts["stop_policy"] = "approx_mass"
        opts["approx_mass_margin_schedule"] = (
            (500, 0.0085),
            (1000, 0.0080),
            (4000, 0.0060),
            (64000, 0.0050),
            (128000, 0.0008),
        )
        opts["display_name"] = normalized
        return PagedPQSelector(trace=kwargs["trace"], routed=False, **opts)
    if normalized == "paged_local_pq_varperm_approx":
        if kwargs.get("trace") is None:
            raise ValueError("paged_local_pq_varperm_approx requires trace=")
        opts = dict(kwargs.get("paged_kwargs", {}))
        opts["accounting_mode"] = "online"
        opts["stop_policy"] = "approx_mass"
        opts["pq_permutation"] = "variance_balanced"
        opts["display_name"] = normalized
        return PagedPQSelector(trace=kwargs["trace"], routed=False, **opts)
    if normalized == "paged_local_pq_interleave_approx":
        if kwargs.get("trace") is None:
            raise ValueError("paged_local_pq_interleave_approx requires trace=")
        opts = dict(kwargs.get("paged_kwargs", {}))
        opts["accounting_mode"] = "online"
        opts["stop_policy"] = "approx_mass"
        opts["pq_permutation"] = "interleave"
        opts["display_name"] = normalized
        return PagedPQSelector(trace=kwargs["trace"], routed=False, **opts)
    if normalized.startswith("paged_local_pq_varperm_probe_bands_k"):
        if kwargs.get("trace") is None:
            raise ValueError("paged_local_pq_varperm_probe_bands_k<N>[_b<B>][_z<Z>] requires trace=")
        opts = dict(kwargs.get("paged_kwargs", {}))
        opts["accounting_mode"] = "online"
        opts["stop_policy"] = "approx_probe_bands"
        opts["pq_permutation"] = "variance_balanced"
        probe_text, bands_value, z_value = _split_optional_bands_z(
            normalized.removeprefix("paged_local_pq_varperm_probe_bands_k")
        )
        opts["probe_tokens"] = int(probe_text)
        if bands_value is not None:
            opts["probe_bands"] = bands_value
        if z_value is not None:
            opts["probe_confidence_z"] = z_value
        opts["display_name"] = normalized
        return PagedPQSelector(trace=kwargs["trace"], routed=False, **opts)
    if normalized.startswith("paged_local_pq_interleave_probe_bands_k"):
        if kwargs.get("trace") is None:
            raise ValueError("paged_local_pq_interleave_probe_bands_k<N>[_b<B>][_z<Z>] requires trace=")
        opts = dict(kwargs.get("paged_kwargs", {}))
        opts["accounting_mode"] = "online"
        opts["stop_policy"] = "approx_probe_bands"
        opts["pq_permutation"] = "interleave"
        probe_text, bands_value, z_value = _split_optional_bands_z(
            normalized.removeprefix("paged_local_pq_interleave_probe_bands_k")
        )
        opts["probe_tokens"] = int(probe_text)
        if bands_value is not None:
            opts["probe_bands"] = bands_value
        if z_value is not None:
            opts["probe_confidence_z"] = z_value
        opts["display_name"] = normalized
        return PagedPQSelector(trace=kwargs["trace"], routed=False, **opts)
    if normalized.startswith("paged_local_pq_fraction_f"):
        if kwargs.get("trace") is None:
            raise ValueError("paged_local_pq_fraction_f<N> requires trace=")
        opts = dict(kwargs.get("paged_kwargs", {}))
        opts["accounting_mode"] = "online"
        opts["stop_policy"] = "fixed_fraction"
        opts["budget_fraction"] = _fraction_from_name(normalized, "paged_local_pq_fraction_f")
        opts["display_name"] = normalized
        return PagedPQSelector(trace=kwargs["trace"], routed=False, **opts)
    if normalized.startswith("paged_local_pq_budget_"):
        if kwargs.get("trace") is None:
            raise ValueError("paged_local_pq_budget_<rule> requires trace=")
        opts = dict(kwargs.get("paged_kwargs", {}))
        opts["accounting_mode"] = "online"
        opts["stop_policy"] = "fixed_budget"
        opts["budget_rule"] = normalized.removeprefix("paged_local_pq_budget_")
        opts["display_name"] = normalized
        return PagedPQSelector(trace=kwargs["trace"], routed=False, **opts)
    if normalized in {"paged_local_pq_bound", "paged_local_pq_approx_bound"}:
        if kwargs.get("trace") is None:
            raise ValueError("paged_local_pq_bound requires trace=")
        opts = dict(kwargs.get("paged_kwargs", {}))
        opts["accounting_mode"] = "online"
        opts["stop_policy"] = "approx_bound"
        opts["display_name"] = normalized
        return PagedPQSelector(trace=kwargs["trace"], routed=False, **opts)
    if normalized.startswith("paged_local_pq_guard_k"):
        if kwargs.get("trace") is None:
            raise ValueError("paged_local_pq_guard_k<N> requires trace=")
        opts = dict(kwargs.get("paged_kwargs", {}))
        opts["accounting_mode"] = "online"
        opts["stop_policy"] = "approx_guard"
        opts["guard_tokens"] = int(normalized.removeprefix("paged_local_pq_guard_k"))
        opts["display_name"] = normalized
        return PagedPQSelector(trace=kwargs["trace"], routed=False, **opts)
    if normalized.startswith("paged_local_pq_probe_k"):
        if kwargs.get("trace") is None:
            raise ValueError("paged_local_pq_probe_k<N> requires trace=")
        opts = dict(kwargs.get("paged_kwargs", {}))
        opts["accounting_mode"] = "online"
        opts["stop_policy"] = "approx_probe"
        opts["probe_tokens"] = int(normalized.removeprefix("paged_local_pq_probe_k"))
        opts["display_name"] = normalized
        return PagedPQSelector(trace=kwargs["trace"], routed=False, **opts)
    if normalized.startswith("paged_local_pq_probe_ucb_k"):
        if kwargs.get("trace") is None:
            raise ValueError("paged_local_pq_probe_ucb_k<N>[_q<PCT>] requires trace=")
        opts = dict(kwargs.get("paged_kwargs", {}))
        opts["accounting_mode"] = "online"
        opts["stop_policy"] = "approx_probe_ucb"
        body = normalized.removeprefix("paged_local_pq_probe_ucb_k")
        if "_q" in body:
            tokens_text, quantile_text = body.split("_q", 1)
            opts["probe_residual_quantile"] = float(int(quantile_text)) / 100.0
        else:
            tokens_text = body
        opts["probe_tokens"] = int(tokens_text)
        opts["display_name"] = normalized
        return PagedPQSelector(trace=kwargs["trace"], routed=False, **opts)
    if normalized.startswith("paged_local_pq_probe_ratio_k"):
        if kwargs.get("trace") is None:
            raise ValueError("paged_local_pq_probe_ratio_k<N> requires trace=")
        opts = dict(kwargs.get("paged_kwargs", {}))
        opts["accounting_mode"] = "online"
        opts["stop_policy"] = "approx_probe_ratio"
        probe_text, z_value = _split_optional_z(normalized.removeprefix("paged_local_pq_probe_ratio_k"))
        opts["probe_tokens"] = int(probe_text)
        if z_value is not None:
            opts["probe_confidence_z"] = z_value
        opts["display_name"] = normalized
        return PagedPQSelector(trace=kwargs["trace"], routed=False, **opts)
    if normalized.startswith("paged_local_pq_probe_bands_k"):
        if kwargs.get("trace") is None:
            raise ValueError("paged_local_pq_probe_bands_k<N>[_b<B>][_z<Z>] requires trace=")
        opts = dict(kwargs.get("paged_kwargs", {}))
        opts["accounting_mode"] = "online"
        opts["stop_policy"] = "approx_probe_bands"
        probe_text, bands_value, z_value = _split_optional_bands_z(normalized.removeprefix("paged_local_pq_probe_bands_k"))
        opts["probe_tokens"] = int(probe_text)
        if bands_value is not None:
            opts["probe_bands"] = bands_value
        if z_value is not None:
            opts["probe_confidence_z"] = z_value
        opts["display_name"] = normalized
        return PagedPQSelector(trace=kwargs["trace"], routed=False, **opts)
    if normalized.startswith("paged_local_pq_resid_r"):
        if kwargs.get("trace") is None:
            raise ValueError("paged_local_pq_resid_r<N> requires trace=")
        opts = dict(kwargs.get("paged_kwargs", {}))
        opts["accounting_mode"] = "online"
        opts["stop_policy"] = "approx_mass"
        opts["sparq_residual_rank"] = int(normalized.removeprefix("paged_local_pq_resid_r"))
        opts["display_name"] = normalized
        return PagedPQSelector(trace=kwargs["trace"], routed=False, **opts)
    if normalized.startswith("paged_local_pq_resid_pool_r"):
        if kwargs.get("trace") is None:
            raise ValueError("paged_local_pq_resid_pool_r<N> requires trace=")
        opts = dict(kwargs.get("paged_kwargs", {}))
        opts["accounting_mode"] = "online"
        opts["stop_policy"] = "approx_mass"
        opts["sparq_residual_rank"] = int(normalized.removeprefix("paged_local_pq_resid_pool_r"))
        opts["sparq_residual_pool"] = True
        opts["display_name"] = normalized
        return PagedPQSelector(trace=kwargs["trace"], routed=False, **opts)
    if normalized.startswith("paged_local_pq_resid_window_r"):
        if kwargs.get("trace") is None:
            raise ValueError("paged_local_pq_resid_window_r<R>_w<W> requires trace=")
        opts = dict(kwargs.get("paged_kwargs", {}))
        opts["accounting_mode"] = "online"
        opts["stop_policy"] = "approx_mass"
        body = normalized.removeprefix("paged_local_pq_resid_window_r")
        rank_text, window_text = body.split("_w", 1)
        opts["sparq_residual_rank"] = int(rank_text)
        opts["sparq_residual_window"] = int(window_text)
        opts["display_name"] = normalized
        return PagedPQSelector(trace=kwargs["trace"], routed=False, **opts)
    if normalized.startswith("paged_local_pq_resid_probe_ratio_r"):
        if kwargs.get("trace") is None:
            raise ValueError("paged_local_pq_resid_probe_ratio_r<R>_k<N> requires trace=")
        opts = dict(kwargs.get("paged_kwargs", {}))
        opts["accounting_mode"] = "online"
        opts["stop_policy"] = "approx_probe_ratio"
        body = normalized.removeprefix("paged_local_pq_resid_probe_ratio_r")
        rank_text, probe_text = body.split("_k", 1)
        probe_text, z_value = _split_optional_z(probe_text)
        opts["sparq_residual_rank"] = int(rank_text)
        opts["probe_tokens"] = int(probe_text)
        if z_value is not None:
            opts["probe_confidence_z"] = z_value
        opts["display_name"] = normalized
        return PagedPQSelector(trace=kwargs["trace"], routed=False, **opts)
    if normalized.startswith("paged_local_pq_resid_probe_bands_r"):
        if kwargs.get("trace") is None:
            raise ValueError("paged_local_pq_resid_probe_bands_r<R>_k<N>[_b<B>][_z<Z>] requires trace=")
        opts = dict(kwargs.get("paged_kwargs", {}))
        opts["accounting_mode"] = "online"
        opts["stop_policy"] = "approx_probe_bands"
        body = normalized.removeprefix("paged_local_pq_resid_probe_bands_r")
        rank_text, probe_text = body.split("_k", 1)
        probe_text, bands_value, z_value = _split_optional_bands_z(probe_text)
        opts["sparq_residual_rank"] = int(rank_text)
        opts["probe_tokens"] = int(probe_text)
        if bands_value is not None:
            opts["probe_bands"] = bands_value
        if z_value is not None:
            opts["probe_confidence_z"] = z_value
        opts["display_name"] = normalized
        return PagedPQSelector(trace=kwargs["trace"], routed=False, **opts)
    if normalized.startswith("paged_local_pq_resid_pool_probe_ratio_r"):
        if kwargs.get("trace") is None:
            raise ValueError("paged_local_pq_resid_pool_probe_ratio_r<R>_k<N> requires trace=")
        opts = dict(kwargs.get("paged_kwargs", {}))
        opts["accounting_mode"] = "online"
        opts["stop_policy"] = "approx_probe_ratio"
        body = normalized.removeprefix("paged_local_pq_resid_pool_probe_ratio_r")
        rank_text, probe_text = body.split("_k", 1)
        probe_text, z_value = _split_optional_z(probe_text)
        opts["sparq_residual_rank"] = int(rank_text)
        opts["sparq_residual_pool"] = True
        opts["probe_tokens"] = int(probe_text)
        if z_value is not None:
            opts["probe_confidence_z"] = z_value
        opts["display_name"] = normalized
        return PagedPQSelector(trace=kwargs["trace"], routed=False, **opts)
    if normalized.startswith("paged_local_pq_resid_pool_probe_bands_r"):
        if kwargs.get("trace") is None:
            raise ValueError("paged_local_pq_resid_pool_probe_bands_r<R>_k<N>[_b<B>][_z<Z>] requires trace=")
        opts = dict(kwargs.get("paged_kwargs", {}))
        opts["accounting_mode"] = "online"
        opts["stop_policy"] = "approx_probe_bands"
        body = normalized.removeprefix("paged_local_pq_resid_pool_probe_bands_r")
        rank_text, probe_text = body.split("_k", 1)
        probe_text, bands_value, z_value = _split_optional_bands_z(probe_text)
        opts["sparq_residual_rank"] = int(rank_text)
        opts["sparq_residual_pool"] = True
        opts["probe_tokens"] = int(probe_text)
        if bands_value is not None:
            opts["probe_bands"] = bands_value
        if z_value is not None:
            opts["probe_confidence_z"] = z_value
        opts["display_name"] = normalized
        return PagedPQSelector(trace=kwargs["trace"], routed=False, **opts)
    if normalized.startswith("paged_local_pq_resid_window_probe_bands_r"):
        if kwargs.get("trace") is None:
            raise ValueError("paged_local_pq_resid_window_probe_bands_r<R>_w<W>_k<N>[_b<B>][_z<Z>] requires trace=")
        opts = dict(kwargs.get("paged_kwargs", {}))
        opts["accounting_mode"] = "online"
        opts["stop_policy"] = "approx_probe_bands"
        body = normalized.removeprefix("paged_local_pq_resid_window_probe_bands_r")
        rank_text, rest = body.split("_w", 1)
        window_text, probe_text = rest.split("_k", 1)
        probe_text, bands_value, z_value = _split_optional_bands_z(probe_text)
        opts["sparq_residual_rank"] = int(rank_text)
        opts["sparq_residual_window"] = int(window_text)
        opts["probe_tokens"] = int(probe_text)
        if bands_value is not None:
            opts["probe_bands"] = bands_value
        if z_value is not None:
            opts["probe_confidence_z"] = z_value
        opts["display_name"] = normalized
        return PagedPQSelector(trace=kwargs["trace"], routed=False, **opts)
    if normalized.startswith("paged_local_pq_verify_window_w"):
        if kwargs.get("trace") is None:
            raise ValueError("paged_local_pq_verify_window_w<W> requires trace=")
        opts = dict(kwargs.get("paged_kwargs", {}))
        opts["accounting_mode"] = "online"
        opts["stop_policy"] = "approx_mass"
        opts["exact_verify_window"] = int(normalized.removeprefix("paged_local_pq_verify_window_w"))
        opts["display_name"] = normalized
        return PagedPQSelector(trace=kwargs["trace"], routed=False, **opts)
    if normalized.startswith("paged_local_pq_verify_window_probe_bands_w"):
        if kwargs.get("trace") is None:
            raise ValueError("paged_local_pq_verify_window_probe_bands_w<W>_k<N>[_b<B>][_z<Z>] requires trace=")
        opts = dict(kwargs.get("paged_kwargs", {}))
        opts["accounting_mode"] = "online"
        opts["stop_policy"] = "approx_probe_bands"
        body = normalized.removeprefix("paged_local_pq_verify_window_probe_bands_w")
        window_text, probe_text = body.split("_k", 1)
        probe_text, bands_value, z_value = _split_optional_bands_z(probe_text)
        opts["exact_verify_window"] = int(window_text)
        opts["probe_tokens"] = int(probe_text)
        if bands_value is not None:
            opts["probe_bands"] = bands_value
        if z_value is not None:
            opts["probe_confidence_z"] = z_value
        opts["display_name"] = normalized
        return PagedPQSelector(trace=kwargs["trace"], routed=False, **opts)
    if normalized.startswith("paged_local_pq_proj_window_d"):
        if kwargs.get("trace") is None:
            raise ValueError("paged_local_pq_proj_window_d<D>_w<W> requires trace=")
        opts = dict(kwargs.get("paged_kwargs", {}))
        opts["accounting_mode"] = "online"
        opts["stop_policy"] = "approx_mass"
        body = normalized.removeprefix("paged_local_pq_proj_window_d")
        dim_text, window_text = body.split("_w", 1)
        opts["verify_proj_dim"] = int(dim_text)
        opts["verify_proj_window"] = int(window_text)
        opts["display_name"] = normalized
        return PagedPQSelector(trace=kwargs["trace"], routed=False, **opts)
    if normalized.startswith("paged_local_pq_proj_window_probe_bands_d"):
        if kwargs.get("trace") is None:
            raise ValueError("paged_local_pq_proj_window_probe_bands_d<D>_w<W>_k<N>[_b<B>][_z<Z>] requires trace=")
        opts = dict(kwargs.get("paged_kwargs", {}))
        opts["accounting_mode"] = "online"
        opts["stop_policy"] = "approx_probe_bands"
        body = normalized.removeprefix("paged_local_pq_proj_window_probe_bands_d")
        dim_text, rest = body.split("_w", 1)
        window_text, probe_text = rest.split("_k", 1)
        probe_text, bands_value, z_value = _split_optional_bands_z(probe_text)
        opts["verify_proj_dim"] = int(dim_text)
        opts["verify_proj_window"] = int(window_text)
        opts["probe_tokens"] = int(probe_text)
        if bands_value is not None:
            opts["probe_bands"] = bands_value
        if z_value is not None:
            opts["probe_confidence_z"] = z_value
        opts["display_name"] = normalized
        return PagedPQSelector(trace=kwargs["trace"], routed=False, **opts)
    if normalized in {"gated_paged_pq", "paged_routed_pq", "routed_paged_pq", "gated_paged_pq_online"}:
        if kwargs.get("trace") is None:
            raise ValueError("gated_paged_pq requires trace=")
        opts = dict(kwargs.get("paged_kwargs", {}))
        opts["accounting_mode"] = "online"
        if normalized.endswith("_online"):
            opts["display_name"] = "gated_paged_pq_online"
        return PagedPQSelector(trace=kwargs["trace"], routed=True, **opts)
    if normalized.startswith("gated_paged_pq_budget_"):
        if kwargs.get("trace") is None:
            raise ValueError("gated_paged_pq_budget_<rule> requires trace=")
        opts = dict(kwargs.get("paged_kwargs", {}))
        opts["accounting_mode"] = "online"
        opts["stop_policy"] = "fixed_budget"
        opts["budget_rule"] = normalized.removeprefix("gated_paged_pq_budget_")
        opts["display_name"] = normalized
        return PagedPQSelector(trace=kwargs["trace"], routed=True, **opts)
    if normalized in {"gated_paged_pq_snapshot", "paged_routed_pq_snapshot", "routed_paged_pq_snapshot"}:
        if kwargs.get("trace") is None:
            raise ValueError("gated_paged_pq_snapshot requires trace=")
        opts = dict(kwargs.get("paged_kwargs", {}))
        opts["accounting_mode"] = "snapshot"
        opts["display_name"] = "gated_paged_pq_snapshot"
        return PagedPQSelector(trace=kwargs["trace"], routed=True, **opts)
    if normalized in {"paged_local_pq_sparq_rerank", "local_paged_pq_sparq_rerank"}:
        if kwargs.get("trace") is None:
            raise ValueError("paged_local_pq_sparq_rerank requires trace=")
        opts = dict(kwargs.get("paged_kwargs", {}))
        opts["accounting_mode"] = "online"
        opts["display_name"] = "paged_local_pq_sparq_rerank"
        return PagedPQSparQRerankSelector(trace=kwargs["trace"], routed=False, **opts)
    if normalized in {"gated_paged_pq_sparq_rerank", "paged_routed_pq_sparq_rerank"}:
        if kwargs.get("trace") is None:
            raise ValueError("gated_paged_pq_sparq_rerank requires trace=")
        opts = dict(kwargs.get("paged_kwargs", {}))
        opts["accounting_mode"] = "online"
        opts["display_name"] = "gated_paged_pq_sparq_rerank"
        return PagedPQSparQRerankSelector(trace=kwargs["trace"], routed=True, **opts)
    if normalized in {"page_sparq_pq", "sparq_page_pq"}:
        if kwargs.get("trace") is None:
            raise ValueError("page_sparq_pq requires trace=")
        opts = dict(kwargs.get("paged_kwargs", {}))
        opts["accounting_mode"] = "online"
        opts["display_name"] = "page_sparq_pq"
        return PageSparQPQSelector(trace=kwargs["trace"], routed=False, **opts)
    if normalized in {"page_sparq_postings_pq", "sparq_postings_pq"}:
        if kwargs.get("trace") is None:
            raise ValueError("page_sparq_postings_pq requires trace=")
        opts = dict(kwargs.get("paged_kwargs", {}))
        opts["accounting_mode"] = "online"
        opts["display_name"] = "page_sparq_postings_pq"
        return PageSparQPostingsPQSelector(trace=kwargs["trace"], routed=False, **opts)
    if normalized.startswith("page_sparq_postings_pq_k"):
        if kwargs.get("trace") is None:
            raise ValueError("page_sparq_postings_pq_k<N> requires trace=")
        opts = dict(kwargs.get("paged_kwargs", {}))
        opts["accounting_mode"] = "online"
        opts["display_name"] = normalized
        opts["postings_per_dim"] = int(normalized.removeprefix("page_sparq_postings_pq_k"))
        return PageSparQPostingsPQSelector(trace=kwargs["trace"], routed=False, **opts)
    if normalized in {"ivfpq", "ivfpq_frozen_append"}:
        if kwargs.get("trace") is None:
            raise ValueError("ivfpq requires trace=")
        return IVFPQSelector(trace=kwargs["trace"], policy="frozen_append", **kwargs.get("ivfpq_kwargs", {}))
    if normalized in {"ivfpq_online_centroid", "ivfpq_online"}:
        if kwargs.get("trace") is None:
            raise ValueError("ivfpq_online_centroid requires trace=")
        return IVFPQSelector(trace=kwargs["trace"], policy="online_centroid", **kwargs.get("ivfpq_kwargs", {}))
    if normalized in {"ivfpq_periodic_rebuild", "ivfpq_periodic"}:
        if kwargs.get("trace") is None:
            raise ValueError("ivfpq_periodic_rebuild requires trace=")
        return IVFPQSelector(trace=kwargs["trace"], policy="periodic_rebuild", **kwargs.get("ivfpq_kwargs", {}))
    if normalized == "sparq":
        return SparQSelector(**kwargs.get("sparq_kwargs", {}))
    if normalized.startswith("sparq_r"):
        return SparQSelector(rank=int(normalized.removeprefix("sparq_r")))
    if normalized == "magicpig":
        return MagicPIGSelector(**kwargs.get("magicpig_kwargs", {}))
    if normalized.startswith("magicpig_k") and "_l" in normalized:
        bits_text, tables_text = normalized.removeprefix("magicpig_k").split("_l", 1)
        opts = dict(kwargs.get("magicpig_kwargs", {}))
        opts["bits"] = int(bits_text)
        opts["tables"] = int(tables_text)
        return MagicPIGSelector(**opts)
    raise ValueError(f"unknown selector: {name}")


def _margin_from_mbp_name(normalized: str, prefix: str) -> float:
    """Parse margin basis points from names like ``<prefix>_mbp100``."""
    if normalized == prefix:
        return 0.0
    suffix = normalized.removeprefix(prefix)
    if not suffix.startswith("_mbp"):
        raise ValueError(f"expected {prefix}_mbp<N>, got: {normalized}")
    return float(int(suffix.removeprefix("_mbp"))) / 10000.0


def _fraction_from_name(normalized: str, prefix: str) -> float:
    """Parse fraction suffixes: ``f10`` -> 0.10, ``f125`` -> 0.125."""
    suffix = normalized.removeprefix(prefix)
    if not suffix.isdigit():
        raise ValueError(f"expected {prefix}<digits>, got: {normalized}")
    return float(int(suffix)) / 100.0


def _split_optional_z(text: str) -> tuple[str, float | None]:
    if "_z" not in text:
        return text, None
    base, z_text = text.split("_z", 1)
    return base, float(z_text.replace("p", "."))


def _split_optional_bands_z(text: str) -> tuple[str, int | None, float | None]:
    z_value = None
    if "_z" in text:
        text, z_text = text.split("_z", 1)
        z_value = float(z_text.replace("p", "."))
    bands_value = None
    if "_b" in text:
        text, bands_text = text.split("_b", 1)
        bands_value = int(bands_text)
    return text, bands_value, z_value
