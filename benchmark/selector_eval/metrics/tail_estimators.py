from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from benchmark.selector_eval.costs.base import CostTrace, kv_read_bytes
from benchmark.selector_eval.metrics.attention import output_error_metrics
from benchmark.selector_eval.selectors.base import QueryState, SelectionResult


@dataclass(frozen=True)
class TailEstimate:
    name: str
    output: np.ndarray
    cost: CostTrace
    metadata: dict[str, float | int | str | bool]


def _budget_from_rule(rule: str, context_len: int) -> int:
    normalized = str(rule).strip().lower()
    n = max(1, int(context_len))
    if normalized.startswith("s") and normalized.removeprefix("s").isdigit():
        return max(0, int(normalized.removeprefix("s")))
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
    raise ValueError(f"unknown tail budget rule: {rule}")


def parse_tail_estimator_name(name: str, context_len: int) -> tuple[str, int, int]:
    normalized = str(name).strip().lower()
    seed = 0
    if "_seed" in normalized:
        normalized, seed_text = normalized.rsplit("_seed", 1)
        seed = int(seed_text)
    if normalized in {"pq_head_only", "vpq_head_only"}:
        return normalized, 0, seed
    for prefix in ("kpqv_tail_b", "krpq_tail_b"):
        if normalized.startswith(prefix):
            rest = normalized.removeprefix(prefix)
            bands_text, rule = rest.split("_", 1)
            return f"{prefix}{bands_text}", _budget_from_rule(rule, context_len), seed
    if normalized.startswith("tqmse_after_d") or normalized.startswith("tqprod_after_d"):
        family, rest = normalized.split("_tail_b", 1)
        threshold_text, spec = family.split("_k", 1)
        threshold = int(threshold_text.rsplit("_d", 1)[1])
        key_bits_text, value_bits_text = spec.split("v", 1)
        bands_text, rule = rest.split("_", 1)
        prefix = "tqmse" if family.startswith("tqmse") else "tqprod"
        return f"{prefix}_after_d{threshold}_k{key_bits_text}v{value_bits_text}_tail_b{bands_text}", _budget_from_rule(rule, context_len), seed
    if normalized.startswith("tqmse_k") or normalized.startswith("tqprod_k"):
        family, rest = normalized.split("_tail_b", 1)
        spec = family.split("_k", 1)[1]
        key_bits_text, value_bits_text = spec.split("v", 1)
        bands_text, rule = rest.split("_", 1)
        prefix = "tqmse" if family.startswith("tqmse") else "tqprod"
        return f"{prefix}_k{key_bits_text}v{value_bits_text}_tail_b{bands_text}", _budget_from_rule(rule, context_len), seed
    if normalized.startswith("kbandvmix_after_d"):
        head, rest = normalized.split("_tail_b", 1)
        threshold_text, spec = head.removeprefix("kbandvmix_after_d").split("_p", 1)
        cal_spec, exact_value_text = spec.split("_e", 1)
        probes_text, cal_bands_text = cal_spec.split("x", 1)
        bands_text, rule = rest.split("_", 1)
        return (
            f"kbandvmix_after_d{threshold_text}_p{probes_text}x{cal_bands_text}_e{exact_value_text}_tail_b{bands_text}",
            _budget_from_rule(rule, context_len),
            seed,
        )
    if normalized.startswith("kbandvpq_after_d"):
        head, rest = normalized.split("_tail_b", 1)
        threshold_text, spec = head.removeprefix("kbandvpq_after_d").split("_p", 1)
        probes_text, cal_bands_text = spec.split("x", 1)
        bands_text, rule = rest.split("_", 1)
        return f"kbandvpq_after_d{threshold_text}_p{probes_text}x{cal_bands_text}_tail_b{bands_text}", _budget_from_rule(rule, context_len), seed
    if normalized.startswith("kcalib_after_d"):
        head, rest = normalized.split("_tail_b", 1)
        threshold_text, probes_text = head.removeprefix("kcalib_after_d").split("_p", 1)
        bands_text, rule = rest.split("_", 1)
        return f"kcalib_after_d{threshold_text}_p{probes_text}_tail_b{bands_text}", _budget_from_rule(rule, context_len), seed
    if normalized.startswith("kbandcalib_after_d"):
        head, rest = normalized.split("_tail_b", 1)
        threshold_text, spec = head.removeprefix("kbandcalib_after_d").split("_p", 1)
        probes_text, cal_bands_text = spec.split("x", 1)
        bands_text, rule = rest.split("_", 1)
        return f"kbandcalib_after_d{threshold_text}_p{probes_text}x{cal_bands_text}_tail_b{bands_text}", _budget_from_rule(rule, context_len), seed
    if normalized.startswith("kbandcalib"):
        head, rest = normalized.split("_tail_b", 1)
        probes_text, cal_bands_text = head.removeprefix("kbandcalib").split("x", 1)
        bands_text, rule = rest.split("_", 1)
        return f"kbandcalib{probes_text}x{cal_bands_text}_tail_b{bands_text}", _budget_from_rule(rule, context_len), seed
    if normalized.startswith("kcalib"):
        head, rest = normalized.split("_tail_b", 1)
        probes = int(head.removeprefix("kcalib"))
        bands_text, rule = rest.split("_", 1)
        return f"kcalib{probes}_tail_b{bands_text}", _budget_from_rule(rule, context_len), seed
    if normalized.startswith("ktopr"):
        head, rest = normalized.split("_tail_b", 1)
        rank = int(head.removeprefix("ktopr"))
        bands_text, rule = rest.split("_", 1)
        return f"ktopr{rank}_tail_b{bands_text}", _budget_from_rule(rule, context_len), seed
    if normalized.startswith("vpq_after_d") and "_strat_exp_tail_b" in normalized:
        head, rest = normalized.split("_strat_exp_tail_b", 1)
        threshold = int(head.removeprefix("vpq_after_d"))
        bands_text, rule = rest.split("_", 1)
        return f"vpq_after_d{threshold}_strat_exp_tail_b{bands_text}", _budget_from_rule(rule, context_len), seed
    if normalized.startswith("vmix_after_d"):
        head, rest = normalized.split("_tail_b", 1)
        threshold_text, exact_value_text = head.removeprefix("vmix_after_d").split("_e", 1)
        bands_text, rule = rest.split("_", 1)
        return f"vmix_after_d{threshold_text}_e{exact_value_text}_tail_b{bands_text}", _budget_from_rule(rule, context_len), seed
    for prefix in (
        "pq_head_strat_exp_tail_b",
        "vpq_head_strat_exp_tail_b",
        "strat_tail_b",
        "strat_exp_tail_b",
        "strat_neyman_tail_b",
        "strat_cv_tail_b",
        "strat_exp_cv_tail_b",
        "strat_exp_pqcv_tail_b",
    ):
        if normalized.startswith(prefix):
            rest = normalized.removeprefix(prefix)
            bands_text, rule = rest.split("_", 1)
            return f"{prefix}{bands_text}", _budget_from_rule(rule, context_len), seed
    for marker in ("_sqrt_", "_log_", "_n0", "_k", "_s"):
        if marker in normalized:
            kind, rule = normalized.rsplit(marker, 1)
            return kind, _budget_from_rule(marker.removeprefix("_") + rule, context_len), seed
    raise ValueError(f"tail estimator must include budget suffix, got: {name}")


def tail_estimate_from_name(
    name: str,
    state: QueryState,
    result: SelectionResult,
    *,
    key_bytes: int,
    value_bytes: int,
) -> TailEstimate:
    kind, samples, seed = parse_tail_estimator_name(name, state.scores.shape[0])
    if kind in {"uniform_tail", "tail_uniform"}:
        return uniform_tail_estimate(
            name,
            state,
            result,
            samples=samples,
            seed=seed,
            key_bytes=key_bytes,
            value_bytes=value_bytes,
        )
    if kind in {"oracle_prob_tail", "tail_oracle_prob"}:
        return oracle_prob_tail_estimate(
            name,
            state,
            result,
            samples=samples,
            seed=seed,
            key_bytes=key_bytes,
            value_bytes=value_bytes,
        )
    if kind in {"rank_tail", "tail_rank"}:
        return rank_tail_estimate(
            name,
            state,
            result,
            samples=samples,
            seed=seed,
            key_bytes=key_bytes,
            value_bytes=value_bytes,
        )
    if kind in {"uniform_head", "head_uniform"}:
        return uniform_head_estimate(
            name,
            state,
            result,
            samples=samples,
            seed=seed,
            key_bytes=key_bytes,
            value_bytes=value_bytes,
        )
    if kind == "pq_head_only":
        return pq_head_estimate(
            name,
            state,
            result,
            samples=0,
            seed=seed,
            bands=1,
            allocation="none",
            key_bytes=key_bytes,
            value_bytes=value_bytes,
        )
    if kind == "vpq_head_only":
        return compressed_value_head_estimate(
            name,
            state,
            result,
            samples=0,
            seed=seed,
            bands=1,
            allocation="none",
            key_bytes=key_bytes,
            value_bytes=value_bytes,
        )
    if kind.startswith("kpqv_tail_b"):
        return compressed_key_exact_value_head_estimate(
            name,
            state,
            result,
            samples=samples,
            seed=seed,
            bands=int(kind.removeprefix("kpqv_tail_b")),
            mode="base_pq",
            residual_rank=0,
            key_bytes=key_bytes,
            value_bytes=value_bytes,
        )
    if kind.startswith("krpq_tail_b"):
        return compressed_key_exact_value_head_estimate(
            name,
            state,
            result,
            samples=samples,
            seed=seed,
            bands=int(kind.removeprefix("krpq_tail_b")),
            mode="residual_pq",
            residual_rank=0,
            key_bytes=key_bytes,
            value_bytes=value_bytes,
        )
    if (kind.startswith("tqmse_after_d") or kind.startswith("tqprod_after_d")) and "_tail_b" in kind:
        head, band_text = kind.split("_tail_b", 1)
        prefix, spec = head.split("_k", 1)
        threshold = int(prefix.rsplit("_d", 1)[1])
        key_bits_text, value_bits_text = spec.split("v", 1)
        if int(state.decode_tokens) <= threshold:
            return stratified_tail_estimate(
                name,
                state,
                result,
                samples=samples,
                seed=seed,
                bands=int(band_text),
                allocation="exp",
                control_variate=False,
                key_bytes=key_bytes,
                value_bytes=value_bytes,
            )
        return turboquant_head_estimate(
            name,
            state,
            result,
            samples=samples,
            seed=seed,
            bands=int(band_text),
            key_bits=int(key_bits_text),
            value_bits=int(value_bits_text),
            product_residual=kind.startswith("tqprod"),
            key_bytes=key_bytes,
            value_bytes=value_bytes,
        )
    if (kind.startswith("tqmse_k") or kind.startswith("tqprod_k")) and "_tail_b" in kind:
        head, band_text = kind.split("_tail_b", 1)
        spec = head.split("_k", 1)[1]
        key_bits_text, value_bits_text = spec.split("v", 1)
        return turboquant_head_estimate(
            name,
            state,
            result,
            samples=samples,
            seed=seed,
            bands=int(band_text),
            key_bits=int(key_bits_text),
            value_bits=int(value_bits_text),
            product_residual=kind.startswith("tqprod"),
            key_bytes=key_bytes,
            value_bytes=value_bytes,
        )
    if kind.startswith("kbandvmix_after_d") and "_tail_b" in kind:
        head, band_text = kind.split("_tail_b", 1)
        threshold_text, spec = head.removeprefix("kbandvmix_after_d").split("_p", 1)
        cal_spec, exact_value_text = spec.split("_e", 1)
        probes_text, cal_bands_text = cal_spec.split("x", 1)
        if int(state.decode_tokens) <= int(threshold_text):
            return stratified_tail_estimate(
                name,
                state,
                result,
                samples=samples,
                seed=seed,
                bands=int(band_text),
                allocation="exp",
                control_variate=False,
                key_bytes=key_bytes,
                value_bytes=value_bytes,
            )
        return compressed_key_value_head_estimate(
            name,
            state,
            result,
            samples=samples,
            seed=seed,
            bands=int(band_text),
            key_mode="band_calibrated_pq",
            value_mode="mixed_vpq",
            exact_value_top=int(exact_value_text),
            calibration_probes=int(probes_text),
            calibration_bands=int(cal_bands_text),
            key_bytes=key_bytes,
            value_bytes=value_bytes,
        )
    if kind.startswith("kbandvpq_after_d") and "_tail_b" in kind:
        head, band_text = kind.split("_tail_b", 1)
        threshold_text, spec = head.removeprefix("kbandvpq_after_d").split("_p", 1)
        probes_text, cal_bands_text = spec.split("x", 1)
        if int(state.decode_tokens) <= int(threshold_text):
            return stratified_tail_estimate(
                name,
                state,
                result,
                samples=samples,
                seed=seed,
                bands=int(band_text),
                allocation="exp",
                control_variate=False,
                key_bytes=key_bytes,
                value_bytes=value_bytes,
            )
        return compressed_key_value_head_estimate(
            name,
            state,
            result,
            samples=samples,
            seed=seed,
            bands=int(band_text),
            key_mode="band_calibrated_pq",
            value_mode="vpq",
            exact_value_top=0,
            calibration_probes=int(probes_text),
            calibration_bands=int(cal_bands_text),
            key_bytes=key_bytes,
            value_bytes=value_bytes,
        )
    if kind.startswith("kcalib_after_d") and "_tail_b" in kind:
        head, band_text = kind.split("_tail_b", 1)
        threshold_text, probes_text = head.removeprefix("kcalib_after_d").split("_p", 1)
        if int(state.decode_tokens) <= int(threshold_text):
            return stratified_tail_estimate(
                name,
                state,
                result,
                samples=samples,
                seed=seed,
                bands=int(band_text),
                allocation="exp",
                control_variate=False,
                key_bytes=key_bytes,
                value_bytes=value_bytes,
            )
        return compressed_key_exact_value_head_estimate(
            name,
            state,
            result,
            samples=samples,
            seed=seed,
            bands=int(band_text),
            mode="calibrated_pq",
            residual_rank=0,
            calibration_probes=int(probes_text),
            key_bytes=key_bytes,
            value_bytes=value_bytes,
        )
    if kind.startswith("kbandcalib_after_d") and "_tail_b" in kind:
        head, band_text = kind.split("_tail_b", 1)
        threshold_text, spec = head.removeprefix("kbandcalib_after_d").split("_p", 1)
        probes_text, cal_bands_text = spec.split("x", 1)
        if int(state.decode_tokens) <= int(threshold_text):
            return stratified_tail_estimate(
                name,
                state,
                result,
                samples=samples,
                seed=seed,
                bands=int(band_text),
                allocation="exp",
                control_variate=False,
                key_bytes=key_bytes,
                value_bytes=value_bytes,
            )
        return compressed_key_exact_value_head_estimate(
            name,
            state,
            result,
            samples=samples,
            seed=seed,
            bands=int(band_text),
            mode="band_calibrated_pq",
            residual_rank=0,
            calibration_probes=int(probes_text),
            calibration_bands=int(cal_bands_text),
            key_bytes=key_bytes,
            value_bytes=value_bytes,
        )
    if kind.startswith("kbandcalib") and "_tail_b" in kind:
        head, band_text = kind.split("_tail_b", 1)
        probes_text, cal_bands_text = head.removeprefix("kbandcalib").split("x", 1)
        return compressed_key_exact_value_head_estimate(
            name,
            state,
            result,
            samples=samples,
            seed=seed,
            bands=int(band_text),
            mode="band_calibrated_pq",
            residual_rank=0,
            calibration_probes=int(probes_text),
            calibration_bands=int(cal_bands_text),
            key_bytes=key_bytes,
            value_bytes=value_bytes,
        )
    if kind.startswith("kcalib") and "_tail_b" in kind:
        head, band_text = kind.split("_tail_b", 1)
        return compressed_key_exact_value_head_estimate(
            name,
            state,
            result,
            samples=samples,
            seed=seed,
            bands=int(band_text),
            mode="calibrated_pq",
            residual_rank=0,
            calibration_probes=int(head.removeprefix("kcalib")),
            key_bytes=key_bytes,
            value_bytes=value_bytes,
        )
    if kind.startswith("ktopr") and "_tail_b" in kind:
        head, band_text = kind.split("_tail_b", 1)
        return compressed_key_exact_value_head_estimate(
            name,
            state,
            result,
            samples=samples,
            seed=seed,
            bands=int(band_text),
            mode="top_residual",
            residual_rank=int(head.removeprefix("ktopr")),
            key_bytes=key_bytes,
            value_bytes=value_bytes,
        )
    if kind.startswith("pq_head_strat_exp_tail_b"):
        return pq_head_estimate(
            name,
            state,
            result,
            samples=samples,
            seed=seed,
            bands=int(kind.removeprefix("pq_head_strat_exp_tail_b")),
            allocation="exp",
            key_bytes=key_bytes,
            value_bytes=value_bytes,
        )
    if kind.startswith("vpq_head_strat_exp_tail_b"):
        return compressed_value_head_estimate(
            name,
            state,
            result,
            samples=samples,
            seed=seed,
            bands=int(kind.removeprefix("vpq_head_strat_exp_tail_b")),
            allocation="exp",
            key_bytes=key_bytes,
            value_bytes=value_bytes,
        )
    if kind.startswith("vmix_after_d") and "_tail_b" in kind:
        head, band_text = kind.split("_tail_b", 1)
        threshold_text, exact_value_text = head.removeprefix("vmix_after_d").split("_e", 1)
        if int(state.decode_tokens) <= int(threshold_text):
            return stratified_tail_estimate(
                name,
                state,
                result,
                samples=samples,
                seed=seed,
                bands=int(band_text),
                allocation="exp",
                control_variate=False,
                key_bytes=key_bytes,
                value_bytes=value_bytes,
            )
        return compressed_key_value_head_estimate(
            name,
            state,
            result,
            samples=samples,
            seed=seed,
            bands=int(band_text),
            key_mode="exact",
            value_mode="mixed_vpq",
            exact_value_top=int(exact_value_text),
            key_bytes=key_bytes,
            value_bytes=value_bytes,
        )
    if kind.startswith("vpq_after_d") and "_strat_exp_tail_b" in kind:
        head, band_text = kind.split("_strat_exp_tail_b", 1)
        threshold = int(head.removeprefix("vpq_after_d"))
        if int(state.decode_tokens) <= threshold:
            return stratified_tail_estimate(
                name,
                state,
                result,
                samples=samples,
                seed=seed,
                bands=int(band_text),
                allocation="exp",
                control_variate=False,
                key_bytes=key_bytes,
                value_bytes=value_bytes,
            )
        return compressed_value_head_estimate(
            name,
            state,
            result,
            samples=samples,
            seed=seed,
            bands=int(band_text),
            allocation="exp",
            key_bytes=key_bytes,
            value_bytes=value_bytes,
        )
    if kind.startswith("strat_tail_b"):
        return stratified_tail_estimate(
            name,
            state,
            result,
            samples=samples,
            seed=seed,
            bands=int(kind.removeprefix("strat_tail_b")),
            allocation="equal",
            control_variate=False,
            key_bytes=key_bytes,
            value_bytes=value_bytes,
        )
    if kind.startswith("strat_exp_tail_b"):
        return stratified_tail_estimate(
            name,
            state,
            result,
            samples=samples,
            seed=seed,
            bands=int(kind.removeprefix("strat_exp_tail_b")),
            allocation="exp",
            control_variate=False,
            key_bytes=key_bytes,
            value_bytes=value_bytes,
        )
    if kind.startswith("strat_neyman_tail_b"):
        return stratified_tail_estimate(
            name,
            state,
            result,
            samples=samples,
            seed=seed,
            bands=int(kind.removeprefix("strat_neyman_tail_b")),
            allocation="neyman_proxy",
            control_variate=False,
            key_bytes=key_bytes,
            value_bytes=value_bytes,
        )
    if kind.startswith("strat_cv_tail_b"):
        return stratified_tail_estimate(
            name,
            state,
            result,
            samples=samples,
            seed=seed,
            bands=int(kind.removeprefix("strat_cv_tail_b")),
            allocation="neyman_proxy",
            control_variate="page_mean",
            key_bytes=key_bytes,
            value_bytes=value_bytes,
        )
    if kind.startswith("strat_exp_cv_tail_b"):
        return stratified_tail_estimate(
            name,
            state,
            result,
            samples=samples,
            seed=seed,
            bands=int(kind.removeprefix("strat_exp_cv_tail_b")),
            allocation="exp",
            control_variate="page_mean",
            key_bytes=key_bytes,
            value_bytes=value_bytes,
        )
    if kind.startswith("strat_exp_pqcv_tail_b"):
        return stratified_tail_estimate(
            name,
            state,
            result,
            samples=samples,
            seed=seed,
            bands=int(kind.removeprefix("strat_exp_pqcv_tail_b")),
            allocation="exp",
            control_variate="pq_value",
            key_bytes=key_bytes,
            value_bytes=value_bytes,
        )
    raise ValueError(f"unknown tail estimator: {name}")


def _selected_and_tail(state: QueryState, result: SelectionResult) -> tuple[np.ndarray, np.ndarray]:
    selected = np.asarray(sorted(set(int(tok) for tok in result.selected_tokens)), dtype=np.int64)
    selected_mask = np.zeros((state.scores.shape[0],), dtype=bool)
    if selected.size:
        selected_mask[selected] = True
    tail = np.nonzero(~selected_mask)[0].astype(np.int64, copy=False)
    return selected, tail


def _head_terms(state: QueryState, selected: np.ndarray, max_score: float) -> tuple[np.ndarray, float]:
    if selected.size == 0:
        return np.zeros((state.values.shape[-1],), dtype=np.float64), 0.0
    weights = np.exp(state.scores[selected].astype(np.float64) - float(max_score))
    numerator = weights @ state.values[selected].astype(np.float64, copy=False)
    return numerator, float(weights.sum())


def _tail_order_from_candidates(state: QueryState, result: SelectionResult, selected: np.ndarray, tail: np.ndarray) -> np.ndarray:
    selected_set = set(int(tok) for tok in selected.tolist())
    tail_set = set(int(tok) for tok in tail.tolist())
    ordered = []
    seen = set()
    for tok in result.candidate_tokens:
        tok = int(tok)
        if tok in selected_set or tok not in tail_set or tok in seen:
            continue
        ordered.append(tok)
        seen.add(tok)
    if len(ordered) < int(tail.size):
        # Deterministic fallback for selectors that do not rank every token.
        rest = [int(tok) for tok in tail.tolist() if int(tok) not in seen]
        if rest:
            rest_arr = np.asarray(rest, dtype=np.int64)
            rest_arr = rest_arr[np.argsort(-state.scores[rest_arr], kind="stable")]
            ordered.extend(int(tok) for tok in rest_arr.tolist())
    return np.asarray(ordered, dtype=np.int64)


def _make_rng(seed: int, state: QueryState, name: str) -> np.random.Generator:
    # Stable per-query seed without relying on Python's randomized hash().
    name_code = sum((idx + 1) * ord(ch) for idx, ch in enumerate(str(name)))
    combined = int(seed) + 1000003 * int(state.qidx) + 9176 * int(state.head) + name_code
    return np.random.default_rng(combined % (2**63 - 1))


def _allocate_samples(
    *,
    strata: list[np.ndarray],
    total_samples: int,
    allocation: str,
    state: QueryState,
    result: SelectionResult,
    max_score: float,
) -> list[int]:
    total_samples = max(0, int(total_samples))
    nonempty = [idx for idx, stratum in enumerate(strata) if stratum.size > 0]
    out = [0 for _ in strata]
    if total_samples == 0 or not nonempty:
        return out
    total_samples = min(total_samples, sum(int(stratum.size) for stratum in strata))
    if total_samples >= sum(int(stratum.size) for stratum in strata):
        return [int(stratum.size) for stratum in strata]

    if allocation == "equal":
        weights = np.asarray([1.0 if stratum.size else 0.0 for stratum in strata], dtype=np.float64)
    elif allocation == "exp":
        # More resolution for high-ranked tail bands, but keep nonzero coverage.
        weights = np.asarray([2.0 ** (-idx) if stratum.size else 0.0 for idx, stratum in enumerate(strata)], dtype=np.float64)
    elif allocation == "neyman_proxy":
        weights_list = []
        approx_scores = result.metadata.get("approx_scores")
        approx_scores_arr = np.asarray(approx_scores, dtype=np.float64) if approx_scores is not None else None
        for stratum in strata:
            if stratum.size == 0:
                weights_list.append(0.0)
                continue
            if approx_scores_arr is not None and approx_scores_arr.shape[0] >= state.scores.shape[0]:
                score_source = approx_scores_arr[stratum]
            else:
                score_source = state.scores[stratum].astype(np.float64)
            token_weights = np.exp(score_source.astype(np.float64) - float(max_score))
            vals = state.values[stratum].astype(np.float64, copy=False)
            contrib_norm = token_weights * np.linalg.norm(vals, axis=1)
            sigma = float(np.std(contrib_norm)) if stratum.size > 1 else float(np.mean(np.abs(contrib_norm)))
            weights_list.append(float(stratum.size) * max(sigma, 1e-30))
        weights = np.asarray(weights_list, dtype=np.float64)
    else:
        raise ValueError(f"unknown allocation: {allocation}")

    if not np.isfinite(weights).all() or float(weights.sum()) <= 0.0:
        weights = np.asarray([float(stratum.size) for stratum in strata], dtype=np.float64)
    raw = total_samples * weights / max(float(weights.sum()), 1e-20)
    for idx, value in enumerate(np.floor(raw).astype(np.int64).tolist()):
        out[idx] = min(int(strata[idx].size), max(0, int(value)))
    for idx in nonempty:
        if sum(out) >= total_samples:
            break
        if out[idx] == 0:
            out[idx] = 1
    while sum(out) > total_samples:
        idx = max(range(len(out)), key=lambda i: out[i])
        out[idx] -= 1
    remainders = raw - np.floor(raw)
    order = sorted(range(len(strata)), key=lambda i: float(remainders[i]), reverse=True)
    cursor = 0
    while sum(out) < total_samples and order:
        idx = order[cursor % len(order)]
        if out[idx] < int(strata[idx].size):
            out[idx] += 1
        cursor += 1
        if cursor > 10 * len(order) and all(out[i] >= int(strata[i].size) for i in order):
            break
    return out


def _pq_approx_components(state: QueryState, result: SelectionResult, tokens: np.ndarray, max_score: float) -> tuple[np.ndarray, float]:
    approx_scores = result.metadata.get("approx_scores")
    approx_values = result.metadata.get("approx_values")
    if approx_scores is None or approx_values is None or tokens.size == 0:
        return np.zeros((state.values.shape[-1],), dtype=np.float64), 0.0
    approx_scores_arr = np.asarray(approx_scores, dtype=np.float64)
    approx_values_arr = np.asarray(approx_values, dtype=np.float64)
    weights = np.exp(approx_scores_arr[tokens] - float(max_score))
    numerator = weights @ approx_values_arr[tokens]
    return numerator, float(weights.sum())


def _page_mean_value_lookup(result: SelectionResult, tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    starts = result.metadata.get("value_page_starts")
    sizes = result.metadata.get("value_page_sizes")
    means = result.metadata.get("value_page_means")
    if starts is None or sizes is None or means is None or tokens.size == 0:
        return np.full((tokens.size,), -1, dtype=np.int64), np.zeros((tokens.size, 0), dtype=np.float64)
    starts_arr = np.asarray(starts, dtype=np.int64)
    sizes_arr = np.asarray(sizes, dtype=np.int64)
    means_arr = np.asarray(means, dtype=np.float64)
    if starts_arr.size == 0 or means_arr.size == 0:
        return np.full((tokens.size,), -1, dtype=np.int64), np.zeros((tokens.size, means_arr.shape[-1] if means_arr.ndim == 2 else 0), dtype=np.float64)
    page_ids = np.searchsorted(starts_arr, tokens, side="right") - 1
    valid = (page_ids >= 0) & (page_ids < starts_arr.size)
    valid &= tokens < (starts_arr[np.maximum(page_ids, 0)] + sizes_arr[np.maximum(page_ids, 0)])
    page_ids = np.where(valid, page_ids, -1).astype(np.int64, copy=False)
    out = np.zeros((tokens.size, means_arr.shape[-1]), dtype=np.float64)
    valid_pos = np.nonzero(page_ids >= 0)[0]
    if valid_pos.size:
        out[valid_pos] = means_arr[page_ids[valid_pos]]
    return page_ids, out


def _page_mean_approx_components(
    state: QueryState,
    result: SelectionResult,
    tokens: np.ndarray,
    max_score: float,
) -> tuple[np.ndarray, float, np.ndarray]:
    approx_scores = result.metadata.get("approx_scores")
    if approx_scores is None or tokens.size == 0:
        return np.zeros((state.values.shape[-1],), dtype=np.float64), 0.0, np.asarray([], dtype=np.int64)
    approx_scores_arr = np.asarray(approx_scores, dtype=np.float64)
    valid = (tokens >= 0) & (tokens < approx_scores_arr.shape[0]) & np.isfinite(approx_scores_arr[tokens])
    if not np.any(valid):
        return np.zeros((state.values.shape[-1],), dtype=np.float64), 0.0, np.asarray([], dtype=np.int64)
    valid_tokens = tokens[valid]
    page_ids, approx_values = _page_mean_value_lookup(result, valid_tokens)
    weights = np.exp(approx_scores_arr[valid_tokens] - float(max_score))
    numerator = weights @ approx_values
    return numerator, float(weights.sum()), np.unique(page_ids[page_ids >= 0]).astype(np.int64, copy=False)


def _pq_value_lookup(result: SelectionResult, tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    starts = result.metadata.get("value_page_starts")
    sizes = result.metadata.get("value_page_sizes")
    value_codebooks = result.metadata.get("value_pq_codebooks")
    page_codes = result.metadata.get("value_pq_page_codes")
    if starts is None or sizes is None or value_codebooks is None or page_codes is None or tokens.size == 0:
        return np.full((tokens.size,), -1, dtype=np.int64), np.zeros((tokens.size, 0), dtype=np.float64)
    starts_arr = np.asarray(starts, dtype=np.int64)
    sizes_arr = np.asarray(sizes, dtype=np.int64)
    if starts_arr.size == 0 or len(value_codebooks) == 0:
        return np.full((tokens.size,), -1, dtype=np.int64), np.zeros((tokens.size, 0), dtype=np.float64)

    first_codebook = np.asarray(value_codebooks[0], dtype=np.float64)
    subvecs = int(first_codebook.shape[0])
    subdim = int(first_codebook.shape[-1])
    dim = subvecs * subdim
    page_ids = np.searchsorted(starts_arr, tokens, side="right") - 1
    valid = (page_ids >= 0) & (page_ids < starts_arr.size)
    valid &= tokens < (starts_arr[np.maximum(page_ids, 0)] + sizes_arr[np.maximum(page_ids, 0)])
    page_ids = np.where(valid, page_ids, -1).astype(np.int64, copy=False)
    out = np.zeros((tokens.size, dim), dtype=np.float64)
    for page_id in np.unique(page_ids[page_ids >= 0]).astype(np.int64, copy=False).tolist():
        positions = np.nonzero(page_ids == int(page_id))[0]
        if not positions.size:
            continue
        codebook = np.asarray(value_codebooks[int(page_id)], dtype=np.float64)
        codes = np.asarray(page_codes[int(page_id)], dtype=np.int64)
        rows = tokens[positions] - int(starts_arr[int(page_id)])
        rows = rows.astype(np.int64, copy=False)
        for sub in range(subvecs):
            out[positions, sub * subdim : (sub + 1) * subdim] = codebook[sub, codes[rows, sub]]
    return page_ids, out


def _vpq_value_lookup(result: SelectionResult, tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    starts = result.metadata.get("value_page_starts")
    sizes = result.metadata.get("value_page_sizes")
    value_codebooks = result.metadata.get("value_vpq_codebooks")
    page_codes = result.metadata.get("value_vpq_page_codes")
    if starts is None or sizes is None or value_codebooks is None or page_codes is None or tokens.size == 0:
        return np.full((tokens.size,), -1, dtype=np.int64), np.zeros((tokens.size, 0), dtype=np.float64)
    starts_arr = np.asarray(starts, dtype=np.int64)
    sizes_arr = np.asarray(sizes, dtype=np.int64)
    if starts_arr.size == 0 or len(value_codebooks) == 0:
        return np.full((tokens.size,), -1, dtype=np.int64), np.zeros((tokens.size, 0), dtype=np.float64)

    first_codebook = np.asarray(value_codebooks[0], dtype=np.float64)
    subvecs = int(first_codebook.shape[0])
    subdim = int(first_codebook.shape[-1])
    dim = subvecs * subdim
    page_ids = np.searchsorted(starts_arr, tokens, side="right") - 1
    valid = (page_ids >= 0) & (page_ids < starts_arr.size)
    valid &= tokens < (starts_arr[np.maximum(page_ids, 0)] + sizes_arr[np.maximum(page_ids, 0)])
    page_ids = np.where(valid, page_ids, -1).astype(np.int64, copy=False)
    out = np.zeros((tokens.size, dim), dtype=np.float64)
    for page_id in np.unique(page_ids[page_ids >= 0]).astype(np.int64, copy=False).tolist():
        positions = np.nonzero(page_ids == int(page_id))[0]
        if not positions.size:
            continue
        codebook = np.asarray(value_codebooks[int(page_id)], dtype=np.float64)
        codes = np.asarray(page_codes[int(page_id)], dtype=np.int64)
        rows = (tokens[positions] - int(starts_arr[int(page_id)])).astype(np.int64, copy=False)
        for sub in range(subvecs):
            out[positions, sub * subdim : (sub + 1) * subdim] = codebook[sub, codes[rows, sub]]
    return page_ids, out


def _page_ids_for_tokens(result: SelectionResult, tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    starts = result.metadata.get("value_page_starts")
    sizes = result.metadata.get("value_page_sizes")
    if starts is None or sizes is None or tokens.size == 0:
        return np.full((tokens.size,), -1, dtype=np.int64), np.full((tokens.size,), -1, dtype=np.int64)
    starts_arr = np.asarray(starts, dtype=np.int64)
    sizes_arr = np.asarray(sizes, dtype=np.int64)
    if starts_arr.size == 0:
        return np.full((tokens.size,), -1, dtype=np.int64), np.full((tokens.size,), -1, dtype=np.int64)
    page_ids = np.searchsorted(starts_arr, tokens, side="right") - 1
    valid = (page_ids >= 0) & (page_ids < starts_arr.size)
    valid &= tokens < (starts_arr[np.maximum(page_ids, 0)] + sizes_arr[np.maximum(page_ids, 0)])
    page_ids = np.where(valid, page_ids, -1).astype(np.int64, copy=False)
    rows = np.where(page_ids >= 0, tokens - starts_arr[np.maximum(page_ids, 0)], -1).astype(np.int64, copy=False)
    return page_ids, rows


def _key_pq_reconstruct_lookup(result: SelectionResult, tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    codebooks_list = result.metadata.get("key_pq_codebooks")
    page_codes = result.metadata.get("key_pq_page_codes")
    perms = result.metadata.get("key_pq_page_perms")
    page_ids, rows = _page_ids_for_tokens(result, tokens)
    if codebooks_list is None or page_codes is None or tokens.size == 0 or len(codebooks_list) == 0:
        return page_ids, np.zeros((tokens.size, 0), dtype=np.float64)
    first = np.asarray(codebooks_list[0], dtype=np.float64)
    subvecs = int(first.shape[0])
    subdim = int(first.shape[-1])
    dim = subvecs * subdim
    out = np.zeros((tokens.size, dim), dtype=np.float64)
    for page_id in np.unique(page_ids[page_ids >= 0]).astype(np.int64, copy=False).tolist():
        positions = np.nonzero(page_ids == int(page_id))[0]
        if not positions.size:
            continue
        codebooks = np.asarray(codebooks_list[int(page_id)], dtype=np.float64)
        codes = np.asarray(page_codes[int(page_id)], dtype=np.int64)
        page_rows = rows[positions]
        khat_perm = np.zeros((positions.size, dim), dtype=np.float64)
        for sub in range(subvecs):
            khat_perm[:, sub * subdim : (sub + 1) * subdim] = codebooks[sub, codes[page_rows, sub]]
        perm = np.asarray(perms[int(page_id)], dtype=np.int64) if perms is not None and int(page_id) < len(perms) else np.asarray([], dtype=np.int64)
        if perm.size:
            khat = np.zeros_like(khat_perm)
            khat[:, perm] = khat_perm
            out[positions] = khat
        else:
            out[positions] = khat_perm
    return page_ids, out


def _key_residual_pq_scores(result: SelectionResult, tokens: np.ndarray, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    codebooks_list = result.metadata.get("key_residual_pq_codebooks")
    page_codes = result.metadata.get("key_residual_pq_page_codes")
    page_ids, rows = _page_ids_for_tokens(result, tokens)
    out = np.zeros((tokens.size,), dtype=np.float64)
    if codebooks_list is None or page_codes is None or tokens.size == 0 or len(codebooks_list) == 0:
        return page_ids, out
    first = np.asarray(codebooks_list[0], dtype=np.float64)
    subvecs = int(first.shape[0])
    subdim = int(first.shape[-1])
    q_parts = query.astype(np.float64, copy=False).reshape(subvecs, subdim)
    for page_id in np.unique(page_ids[page_ids >= 0]).astype(np.int64, copy=False).tolist():
        positions = np.nonzero(page_ids == int(page_id))[0]
        if not positions.size:
            continue
        codebooks = np.asarray(codebooks_list[int(page_id)], dtype=np.float64)
        codes = np.asarray(page_codes[int(page_id)], dtype=np.int64)
        page_rows = rows[positions]
        scores = np.zeros((positions.size,), dtype=np.float64)
        for sub in range(subvecs):
            table = codebooks[sub] @ q_parts[sub]
            scores += table[codes[page_rows, sub]]
        out[positions] = scores
    return page_ids, out


_TQ_CODEBOOK_CACHE: dict[int, np.ndarray] = {}
_PAPER_TQ_CODEBOOK_CACHE: dict[tuple[int, int], np.ndarray] = {}
_PAPER_TQ_JL_CACHE: dict[tuple[int, int], np.ndarray] = {}


def _tq_codebook(bits: int) -> np.ndarray:
    bits = int(bits)
    if bits <= 0:
        raise ValueError(f"TurboQuant bitwidth must be positive, got {bits}")
    cached = _TQ_CODEBOOK_CACHE.get(bits)
    if cached is not None:
        return cached
    levels = 1 << bits
    # Equal-probability scalar centroids for N(0, 1). This is a proxy for
    # TurboQuant's precomputed Lloyd-Max/Beta codebooks and keeps the path
    # dependency-free for Slurm jobs.
    samples = np.random.default_rng(1009 + bits).standard_normal(400_000)
    samples.sort()
    bins = np.array_split(samples, levels)
    codebook = np.asarray([float(np.mean(part)) for part in bins], dtype=np.float32)
    _TQ_CODEBOOK_CACHE[bits] = codebook
    return codebook


def _paper_tq_codebook(bits: int, dim: int) -> np.ndarray:
    key = (int(bits), int(dim))
    cached = _PAPER_TQ_CODEBOOK_CACHE.get(key)
    if cached is not None:
        return cached
    bits = int(bits)
    dim = int(dim)
    levels = 1 << bits
    rng = np.random.default_rng(4253 + 17 * bits + dim)
    alpha = max(0.5, float(dim - 1) / 2.0)
    samples = (2.0 * rng.beta(alpha, alpha, size=300_000) - 1.0).astype(np.float32)
    centroids = np.linspace(-1.0, 1.0, levels, dtype=np.float32)
    for _ in range(80):
        thresholds = ((centroids[:-1] + centroids[1:]) * 0.5).astype(np.float32, copy=False)
        codes = np.searchsorted(thresholds, samples, side="left")
        new = centroids.copy()
        for idx in range(levels):
            mask = codes == idx
            if np.any(mask):
                new[idx] = float(np.mean(samples[mask]))
        if np.allclose(new, centroids, rtol=1e-5, atol=1e-6):
            centroids = new
            break
        centroids = new
    _PAPER_TQ_CODEBOOK_CACHE[key] = centroids.astype(np.float32, copy=False)
    return _PAPER_TQ_CODEBOOK_CACHE[key]


def _paper_tq_jl_matrix(dim: int, seed: int = 0) -> np.ndarray:
    key = (int(dim), int(seed))
    cached = _PAPER_TQ_JL_CACHE.get(key)
    if cached is not None:
        return cached
    rng = np.random.default_rng(7919 + int(dim) + 10007 * int(seed))
    mat = rng.standard_normal((int(dim), int(dim))).astype(np.float32) / np.sqrt(float(dim))
    _PAPER_TQ_JL_CACHE[key] = mat
    return mat


def _fwht_inplace(x: np.ndarray) -> np.ndarray:
    out = np.asarray(x, dtype=np.float32).copy()
    n = int(out.shape[-1])
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError(f"Hadamard rotation requires power-of-two dimension, got {n}")
    h = 1
    while h < n:
        reshaped = out.reshape(-1, h * 2)
        left = reshaped[:, :h].copy()
        right = reshaped[:, h : h * 2].copy()
        reshaped[:, :h] = left + right
        reshaped[:, h : h * 2] = left - right
        h *= 2
    out /= np.sqrt(float(n))
    return out


def _tq_signs(dim: int) -> np.ndarray:
    # Fixed data-oblivious sign pattern; no per-token storage needed.
    rng = np.random.default_rng(1729 + int(dim))
    return rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=int(dim)).astype(np.float32, copy=False)


def _tq_rotate(x: np.ndarray, signs: np.ndarray) -> np.ndarray:
    return _fwht_inplace(np.asarray(x, dtype=np.float32) * signs)


def _tq_unrotate(x: np.ndarray, signs: np.ndarray) -> np.ndarray:
    # Normalized Hadamard is its own inverse; diagonal signs are also inverse.
    return _fwht_inplace(np.asarray(x, dtype=np.float32)) * signs


def _tq_reconstruct(values: np.ndarray, bits: int) -> tuple[np.ndarray, np.ndarray]:
    values_arr = np.asarray(values, dtype=np.float32)
    if values_arr.size == 0:
        return np.zeros_like(values_arr, dtype=np.float32), np.zeros_like(values_arr, dtype=np.float32)
    dim = int(values_arr.shape[-1])
    signs = _tq_signs(dim)
    norms = np.linalg.norm(values_arr, axis=1, keepdims=True).astype(np.float32)
    safe_norms = np.maximum(norms, np.float32(1e-12))
    unit = values_arr / safe_norms
    rotated = _tq_rotate(unit, signs)
    scaled = rotated * np.sqrt(float(dim))
    codebook = _tq_codebook(bits).astype(np.float32, copy=False)
    thresholds = ((codebook[:-1] + codebook[1:]) * 0.5).astype(np.float32, copy=False)
    codes = np.searchsorted(thresholds, scaled, side="left")
    quant_scaled = codebook[codes].astype(np.float32, copy=False)
    quant_rotated = quant_scaled / np.sqrt(float(dim))
    unit_hat = _tq_unrotate(quant_rotated, signs)
    return (unit_hat * safe_norms).astype(np.float32, copy=False), rotated.astype(np.float32, copy=False)


def _tq_reconstruct_product(values: np.ndarray, bits: int) -> np.ndarray:
    base, base_rotated = _tq_reconstruct(values, bits)
    values_arr = np.asarray(values, dtype=np.float32)
    if values_arr.size == 0:
        return base
    dim = int(values_arr.shape[-1])
    signs = _tq_signs(dim)
    norms = np.linalg.norm(values_arr, axis=1, keepdims=True).astype(np.float32)
    safe_norms = np.maximum(norms, np.float32(1e-12))
    residual_unit = values_arr / safe_norms - base / safe_norms
    residual_rotated = _tq_rotate(residual_unit, signs)
    # 1-bit residual proxy: sign plus per-vector mean magnitude in rotated
    # space. TurboQuant's QJL residual is more principled; this tests the same
    # direction while keeping the simulator compact.
    scale = np.mean(np.abs(residual_rotated), axis=1, keepdims=True).astype(np.float32)
    residual_hat_rotated = np.sign(residual_rotated + np.float32(1e-30)) * scale
    residual_hat = _tq_unrotate(residual_hat_rotated, signs) * safe_norms
    return (base + residual_hat).astype(np.float32, copy=False)


def _paper_tq_reconstruct(values: np.ndarray, bits: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values_arr = np.asarray(values, dtype=np.float32)
    if values_arr.size == 0:
        return np.zeros_like(values_arr, dtype=np.float32), np.zeros_like(values_arr, dtype=np.float32), np.zeros((values_arr.shape[0], 1), dtype=np.float32)
    dim = int(values_arr.shape[-1])
    signs = _tq_signs(dim)
    norms = np.linalg.norm(values_arr, axis=1, keepdims=True).astype(np.float32)
    safe_norms = np.maximum(norms, np.float32(1e-12))
    unit = values_arr / safe_norms
    rotated = _tq_rotate(unit, signs)
    codebook = _paper_tq_codebook(bits, dim).astype(np.float32, copy=False)
    thresholds = ((codebook[:-1] + codebook[1:]) * 0.5).astype(np.float32, copy=False)
    codes = np.searchsorted(thresholds, rotated, side="left")
    quant_rotated = codebook[codes].astype(np.float32, copy=False)
    unit_hat = _tq_unrotate(quant_rotated, signs)
    return (unit_hat * safe_norms).astype(np.float32, copy=False), unit_hat.astype(np.float32, copy=False), norms


def _paper_tq_scores(values: np.ndarray, query: np.ndarray, bits: int, *, product_residual: bool) -> np.ndarray:
    values_arr = np.asarray(values, dtype=np.float32)
    if values_arr.size == 0:
        return np.empty((0,), dtype=np.float64)
    query_arr = np.asarray(query, dtype=np.float32)
    key_hat, unit_hat, norms = _paper_tq_reconstruct(values_arr, bits)
    scores = key_hat.astype(np.float64, copy=False) @ query_arr.astype(np.float64, copy=False)
    if not product_residual:
        return scores
    safe_norms = np.maximum(norms, np.float32(1e-12))
    unit = values_arr / safe_norms
    residual = unit - unit_hat
    residual_norms = np.linalg.norm(residual, axis=1).astype(np.float32)
    active = residual_norms > 1e-12
    if not np.any(active):
        return scores
    jl = _paper_tq_jl_matrix(int(values_arr.shape[-1]), seed=0)
    signs = np.sign(residual[active].astype(np.float32, copy=False) @ jl.T)
    signs[signs == 0] = 1.0
    q_proj = query_arr.astype(np.float32, copy=False) @ jl.T
    correction_unit = (
        np.sqrt(np.pi / 2.0)
        / np.sqrt(float(jl.shape[0]))
        * residual_norms[active].astype(np.float64)
        * (signs.astype(np.float64, copy=False) @ q_proj.astype(np.float64, copy=False))
    )
    scores[active] += norms[active, 0].astype(np.float64) * correction_unit
    return scores


def _tq_dynamic_tokens(result: SelectionResult, selected: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    approx_scores = result.metadata.get("approx_scores")
    if approx_scores is None or selected.size == 0:
        return np.asarray([], dtype=np.int64), selected
    approx_scores_arr = np.asarray(approx_scores, dtype=np.float64)
    valid = (
        (selected >= 0)
        & (selected < approx_scores_arr.shape[0])
        & np.isfinite(approx_scores_arr[selected])
    )
    return selected[valid], selected[~valid]


def _tq_code_bytes(tokens: int, dim: int, bits: int) -> float:
    return float(tokens) * float(dim) * float(bits) / 8.0


def _compressed_selected_terms(
    state: QueryState,
    result: SelectionResult,
    selected: np.ndarray,
    max_score: float,
    *,
    key_bytes: int,
    value_bytes: int,
) -> tuple[np.ndarray, float, CostTrace, int, int, int]:
    """Approximate selected-token attention using selector PQ scores and V centroids.

    Tokens without finite PQ scores or V-centroid sidecar coverage fall back to
    exact K/V. This mostly covers static/pending tokens that are intentionally
    outside the paged PQ index.
    """

    cost = CostTrace()
    dim = int(state.values.shape[-1])
    if selected.size == 0:
        return np.zeros((dim,), dtype=np.float64), 0.0, cost, 0, 0, 0
    approx_scores = result.metadata.get("approx_scores")
    if approx_scores is None:
        cost.read("exact_attention", "compressed_head_exact_fallback_kv", kv_read_bytes(selected.size, dim, key_bytes, value_bytes))
        return _head_terms(state, selected, max_score) + (cost, 0, int(selected.size), 0)

    approx_scores_arr = np.asarray(approx_scores, dtype=np.float64)
    valid_score = (
        (selected >= 0)
        & (selected < approx_scores_arr.shape[0])
        & np.isfinite(approx_scores_arr[selected])
    )
    page_ids, approx_values = _pq_value_lookup(result, selected)
    valid_value = page_ids >= 0
    compressed_mask = valid_score & valid_value
    exact_tokens = selected[~compressed_mask]
    compressed_tokens = selected[compressed_mask]

    num = np.zeros((dim,), dtype=np.float64)
    den = 0.0
    if compressed_tokens.size:
        compressed_scores = approx_scores_arr[compressed_tokens]
        weights = np.exp(compressed_scores - float(max_score))
        num += weights @ approx_values[compressed_mask]
        den += float(weights.sum())
        read_pages = np.unique(page_ids[compressed_mask]).astype(np.int64, copy=False)
        cost.read(
            "exact_attention",
            "compressed_head_value_codebooks",
            len(read_pages) * int(result.metadata.get("value_pq_codebook_bytes_per_page", 0)),
        )
        sidecar_update = float(result.metadata.get("value_pq_sidecar_update_cumulative_bytes", 0.0))
        if sidecar_update > 0.0:
            cost.write(
                "exact_attention",
                "compressed_head_value_sidecar_update_amortized",
                sidecar_update / max(1, int(state.decode_tokens)),
            )
    if exact_tokens.size:
        exact_num, exact_den = _head_terms(state, exact_tokens, max_score)
        num += exact_num
        den += exact_den
        cost.read(
            "exact_attention",
            "compressed_head_exact_fallback_kv",
            kv_read_bytes(exact_tokens.size, dim, key_bytes, value_bytes),
        )
    return num, den, cost, int(compressed_tokens.size), int(exact_tokens.size), int(len(np.unique(page_ids[compressed_mask])) if compressed_tokens.size else 0)


def pq_head_estimate(
    name: str,
    state: QueryState,
    result: SelectionResult,
    *,
    samples: int,
    seed: int,
    bands: int,
    allocation: str,
    key_bytes: int,
    value_bytes: int,
) -> TailEstimate:
    selected, tail = _selected_and_tail(state, result)
    approx_scores = result.metadata.get("approx_scores")
    sample_count = min(max(0, int(samples)), int(tail.size))
    bands = max(1, int(bands))
    variance_proxy = 0.0
    sampled_tail: list[tuple[np.ndarray, int]] = []
    if sample_count > 0 and tail.size > 0:
        ordered = _tail_order_from_candidates(state, result, selected, tail)
        strata = [arr.astype(np.int64, copy=False) for arr in np.array_split(ordered, min(bands, int(ordered.size)))]
        alloc = _allocate_samples(
            strata=strata,
            total_samples=sample_count,
            allocation="exp" if allocation == "exp" else "equal",
            state=state,
            result=result,
            max_score=0.0,
        )
        rng = _make_rng(seed, state, name)
        for stratum, stratum_samples in zip(strata, alloc, strict=False):
            if stratum.size == 0 or stratum_samples <= 0:
                continue
            sample = rng.choice(stratum, size=min(int(stratum_samples), int(stratum.size)), replace=False)
            sampled_tail.append((sample.astype(np.int64, copy=False), int(stratum.size)))
        actual_samples = int(sum(sample.size for sample, _stratum_size in sampled_tail))
    else:
        actual_samples = 0

    score_max_values: list[float] = []
    if selected.size:
        if approx_scores is not None:
            approx_scores_arr = np.asarray(approx_scores, dtype=np.float64)
            valid = (
                (selected >= 0)
                & (selected < approx_scores_arr.shape[0])
                & np.isfinite(approx_scores_arr[selected])
            )
            if np.any(valid):
                score_max_values.append(float(np.max(approx_scores_arr[selected[valid]])))
            if np.any(~valid):
                score_max_values.append(float(np.max(state.scores[selected[~valid]])))
        else:
            score_max_values.append(float(np.max(state.scores[selected])))
    for sample, _stratum_size in sampled_tail:
        if sample.size:
            score_max_values.append(float(np.max(state.scores[sample])))
    max_score = max(score_max_values) if score_max_values else float(np.max(state.scores))

    head_num, head_den, cost, compressed_count, fallback_count, read_pages = _compressed_selected_terms(
        state,
        result,
        selected,
        max_score,
        key_bytes=key_bytes,
        value_bytes=value_bytes,
    )
    tail_num = np.zeros((state.values.shape[-1],), dtype=np.float64)
    tail_den = 0.0
    if sampled_tail:
        alloc_total = 0
        for sample, stratum_size in sampled_tail:
            weights = np.exp(state.scores[sample].astype(np.float64) - max_score)
            scale = float(stratum_size) / float(sample.size)
            tail_num += scale * (weights @ state.values[sample].astype(np.float64, copy=False))
            tail_den += scale * float(weights.sum())
            variance_proxy += float(np.var(weights * float(stratum_size), ddof=1) / max(1, sample.size)) if sample.size > 1 else 0.0
            alloc_total += int(sample.size)
        cost.read(
            "tail_estimator",
            "pq_head_stratified_tail_sample_exact_kv",
            kv_read_bytes(alloc_total, state.values.shape[-1], key_bytes, value_bytes),
        )

    output = (head_num + tail_num) / max(head_den + tail_den, 1e-20)
    return TailEstimate(
        name=name,
        output=output.astype(np.float32),
        cost=cost,
        metadata={
            "tail_estimator": name,
            "tail_kind": "pq_head_compression",
            "tail_samples": actual_samples,
            "tail_population": int(tail.size),
            "tail_estimator_variance": variance_proxy,
            "oracle_diagnostic": False,
            "replaces_exact_attention": True,
            "compressed_head_tokens": compressed_count,
            "compressed_head_exact_fallback_tokens": fallback_count,
            "compressed_head_pages": read_pages,
        },
    )


def _sample_stratified_tail(
    name: str,
    state: QueryState,
    result: SelectionResult,
    selected: np.ndarray,
    tail: np.ndarray,
    *,
    samples: int,
    seed: int,
    bands: int,
) -> list[tuple[np.ndarray, int]]:
    sample_count = min(max(0, int(samples)), int(tail.size))
    if sample_count <= 0 or tail.size == 0:
        return []
    ordered = _tail_order_from_candidates(state, result, selected, tail)
    strata = [arr.astype(np.int64, copy=False) for arr in np.array_split(ordered, min(max(1, int(bands)), int(ordered.size)))]
    alloc = _allocate_samples(
        strata=strata,
        total_samples=sample_count,
        allocation="exp",
        state=state,
        result=result,
        max_score=0.0,
    )
    rng = _make_rng(seed, state, name)
    sampled = []
    for stratum, stratum_samples in zip(strata, alloc, strict=False):
        if stratum.size == 0 or stratum_samples <= 0:
            continue
        sample = rng.choice(stratum, size=min(int(stratum_samples), int(stratum.size)), replace=False)
        sampled.append((sample.astype(np.int64, copy=False), int(stratum.size)))
    return sampled


def compressed_key_exact_value_head_estimate(
    name: str,
    state: QueryState,
    result: SelectionResult,
    *,
    samples: int,
    seed: int,
    bands: int,
    mode: str,
    residual_rank: int,
    key_bytes: int,
    value_bytes: int,
    calibration_probes: int = 0,
    calibration_bands: int = 1,
) -> TailEstimate:
    selected, tail = _selected_and_tail(state, result)
    sampled_tail = _sample_stratified_tail(
        name,
        state,
        result,
        selected,
        tail,
        samples=samples,
        seed=seed,
        bands=bands,
    )
    approx_scores = result.metadata.get("approx_scores")
    if approx_scores is None:
        head_num, head_den = _head_terms(state, selected, float(np.max(state.scores)))
        cost = CostTrace()
        cost.read("exact_attention", "kcomp_exact_fallback_kv", kv_read_bytes(selected.size, state.values.shape[-1], key_bytes, value_bytes))
        output = head_num / max(head_den, 1e-20)
        return TailEstimate(
            name=name,
            output=output.astype(np.float32),
            cost=cost,
            metadata={"tail_estimator": name, "tail_kind": "compressed_key_exact_value", "tail_samples": 0, "tail_population": int(tail.size), "tail_estimator_variance": 0.0, "oracle_diagnostic": False, "replaces_exact_attention": True},
        )

    approx_scores_arr = np.asarray(approx_scores, dtype=np.float64)
    valid = (
        (selected >= 0)
        & (selected < approx_scores_arr.shape[0])
        & np.isfinite(approx_scores_arr[selected])
    )
    compressed_tokens = selected[valid]
    exact_tokens = selected[~valid]
    compressed_scores = approx_scores_arr[compressed_tokens].astype(np.float64, copy=True) if compressed_tokens.size else np.empty((0,), dtype=np.float64)
    read_pages: set[int] = set()
    cost = CostTrace()
    dim = int(state.values.shape[-1])

    if compressed_tokens.size and mode == "top_residual":
        rank = max(0, min(int(residual_rank), dim))
        if rank > 0:
            top_dims = np.argsort(-np.abs(state.query.astype(np.float64)), kind="stable")[:rank]
            _page_ids, khat = _key_pq_reconstruct_lookup(result, compressed_tokens)
            residual = state.keys[compressed_tokens][:, top_dims].astype(np.float64, copy=False) - khat[:, top_dims]
            compressed_scores += residual @ state.query[top_dims].astype(np.float64, copy=False)
            cost.read("exact_attention", f"kcomp_top{rank}_residual_dims", compressed_tokens.size * rank * key_bytes)
    elif compressed_tokens.size and mode == "residual_pq":
        page_ids, residual_scores = _key_residual_pq_scores(result, compressed_tokens, state.query)
        compressed_scores += residual_scores
        read_pages.update(int(page) for page in np.unique(page_ids[page_ids >= 0]).tolist())
        cost.read(
            "exact_attention",
            "kcomp_residual_pq_codebooks",
            len(read_pages) * int(result.metadata.get("key_residual_pq_codebook_bytes_per_page", 0)),
        )
        cost.read(
            "exact_attention",
            "kcomp_residual_pq_codes",
            compressed_tokens.size * int(result.metadata.get("key_residual_pq_code_bytes_per_token", 0)),
        )
        sidecar_update = float(result.metadata.get("key_residual_pq_sidecar_update_cumulative_bytes", 0.0))
        if sidecar_update > 0.0:
            cost.write(
                "exact_attention",
                "kcomp_residual_pq_sidecar_update_amortized",
                sidecar_update / max(1, int(state.decode_tokens)),
            )
    elif compressed_tokens.size and mode in {"calibrated_pq", "band_calibrated_pq"}:
        probe_count = min(max(0, int(calibration_probes)), int(compressed_tokens.size))
        if probe_count > 0:
            selected_set = set(int(tok) for tok in compressed_tokens.tolist())
            ordered = [int(tok) for tok in result.candidate_tokens if int(tok) in selected_set]
            if len(ordered) < int(compressed_tokens.size):
                seen = set(ordered)
                ordered.extend(int(tok) for tok in compressed_tokens.tolist() if int(tok) not in seen)
            ordered_arr = np.asarray(ordered, dtype=np.int64)
            if mode == "band_calibrated_pq":
                band_count = max(1, min(int(calibration_bands), int(ordered_arr.size)))
                ordered_bands = [arr.astype(np.int64, copy=False) for arr in np.array_split(ordered_arr, band_count)]
                per_band = max(1, int(np.ceil(float(probe_count) / float(band_count))))
                probed_parts = []
                calibrated = np.array(compressed_scores, copy=True)
                token_to_pos = {int(tok): idx for idx, tok in enumerate(compressed_tokens.tolist())}
                for band in ordered_bands:
                    if band.size == 0:
                        continue
                    if per_band >= int(band.size):
                        probe_tokens = band
                    else:
                        positions = np.unique(np.linspace(0, int(band.size) - 1, num=per_band, dtype=np.int64))
                        probe_tokens = band[positions]
                    probed_parts.append(probe_tokens)
                    x = approx_scores_arr[probe_tokens].astype(np.float64, copy=False)
                    y = state.scores[probe_tokens].astype(np.float64, copy=False)
                    x_mean = float(np.mean(x))
                    y_mean = float(np.mean(y))
                    x_var = float(np.mean((x - x_mean) * (x - x_mean)))
                    if x_var > 1e-12:
                        slope = float(np.mean((x - x_mean) * (y - y_mean)) / x_var)
                        intercept = y_mean - slope * x_mean
                    else:
                        slope = 1.0
                        intercept = y_mean - x_mean
                    positions_all = np.asarray([token_to_pos[int(tok)] for tok in band.tolist()], dtype=np.int64)
                    calibrated[positions_all] = slope * approx_scores_arr[band] + intercept
                    probe_positions = np.asarray([token_to_pos[int(tok)] for tok in probe_tokens.tolist()], dtype=np.int64)
                    calibrated[probe_positions] = state.scores[probe_tokens].astype(np.float64, copy=False)
                compressed_scores = calibrated
                probe_tokens = np.unique(np.concatenate(probed_parts)) if probed_parts else np.empty((0,), dtype=np.int64)
            else:
                if probe_count >= int(ordered_arr.size):
                    probe_tokens = ordered_arr
                else:
                    positions = np.unique(np.linspace(0, int(ordered_arr.size) - 1, num=probe_count, dtype=np.int64))
                    probe_tokens = ordered_arr[positions]
                probe_mask = np.isin(compressed_tokens, probe_tokens)
                x = approx_scores_arr[probe_tokens].astype(np.float64, copy=False)
                y = state.scores[probe_tokens].astype(np.float64, copy=False)
                x_mean = float(np.mean(x))
                y_mean = float(np.mean(y))
                x_var = float(np.mean((x - x_mean) * (x - x_mean)))
                if x_var > 1e-12:
                    slope = float(np.mean((x - x_mean) * (y - y_mean)) / x_var)
                    intercept = y_mean - slope * x_mean
                else:
                    slope = 1.0
                    intercept = y_mean - x_mean
                compressed_scores = slope * compressed_scores + intercept
                if np.any(probe_mask):
                    compressed_scores[probe_mask] = state.scores[compressed_tokens[probe_mask]].astype(np.float64, copy=False)
            cost.read("exact_attention", "kcomp_calibration_probe_keys", int(probe_tokens.size) * dim * key_bytes)

    max_values = []
    if compressed_scores.size:
        max_values.append(float(np.max(compressed_scores)))
    if exact_tokens.size:
        max_values.append(float(np.max(state.scores[exact_tokens])))
    for sample, _stratum_size in sampled_tail:
        if sample.size:
            max_values.append(float(np.max(state.scores[sample])))
    max_score = max(max_values) if max_values else float(np.max(state.scores))

    head_num = np.zeros((dim,), dtype=np.float64)
    head_den = 0.0
    if compressed_tokens.size:
        weights = np.exp(compressed_scores - max_score)
        head_num += weights @ state.values[compressed_tokens].astype(np.float64, copy=False)
        head_den += float(weights.sum())
        cost.read("exact_attention", "kcomp_exact_values", compressed_tokens.size * dim * value_bytes)
    if exact_tokens.size:
        exact_num, exact_den = _head_terms(state, exact_tokens, max_score)
        head_num += exact_num
        head_den += exact_den
        cost.read("exact_attention", "kcomp_exact_fallback_kv", kv_read_bytes(exact_tokens.size, dim, key_bytes, value_bytes))

    tail_num = np.zeros((dim,), dtype=np.float64)
    tail_den = 0.0
    variance_proxy = 0.0
    alloc_total = 0
    for sample, stratum_size in sampled_tail:
        weights = np.exp(state.scores[sample].astype(np.float64) - max_score)
        scale = float(stratum_size) / float(sample.size)
        tail_num += scale * (weights @ state.values[sample].astype(np.float64, copy=False))
        tail_den += scale * float(weights.sum())
        variance_proxy += float(np.var(weights * float(stratum_size), ddof=1) / max(1, sample.size)) if sample.size > 1 else 0.0
        alloc_total += int(sample.size)
    if alloc_total:
        cost.read("tail_estimator", "kcomp_stratified_tail_sample_exact_kv", kv_read_bytes(alloc_total, dim, key_bytes, value_bytes))

    output = (head_num + tail_num) / max(head_den + tail_den, 1e-20)
    return TailEstimate(
        name=name,
        output=output.astype(np.float32),
        cost=cost,
        metadata={
            "tail_estimator": name,
            "tail_kind": "compressed_key_exact_value",
            "tail_samples": alloc_total,
            "tail_population": int(tail.size),
            "tail_estimator_variance": variance_proxy,
            "oracle_diagnostic": False,
            "replaces_exact_attention": True,
            "compressed_head_tokens": int(compressed_tokens.size),
            "compressed_head_exact_fallback_tokens": int(exact_tokens.size),
            "compressed_key_mode": str(mode),
            "compressed_key_residual_rank": int(residual_rank),
            "compressed_key_calibration_probes": int(calibration_probes),
            "compressed_key_calibration_bands": int(calibration_bands),
        },
    )


def _ordered_subset_by_candidate_rank(result: SelectionResult, tokens: np.ndarray) -> np.ndarray:
    token_set = set(int(tok) for tok in tokens.tolist())
    ordered = []
    seen = set()
    for tok in result.candidate_tokens:
        tok = int(tok)
        if tok in token_set and tok not in seen:
            ordered.append(tok)
            seen.add(tok)
    if len(ordered) < int(tokens.size):
        ordered.extend(int(tok) for tok in tokens.tolist() if int(tok) not in seen)
    return np.asarray(ordered, dtype=np.int64)


def _band_calibrated_scores(
    state: QueryState,
    result: SelectionResult,
    compressed_tokens: np.ndarray,
    approx_scores_arr: np.ndarray,
    *,
    calibration_probes: int,
    calibration_bands: int,
    key_bytes: int,
) -> tuple[np.ndarray, CostTrace, int]:
    cost = CostTrace()
    if compressed_tokens.size == 0:
        return np.empty((0,), dtype=np.float64), cost, 0
    scores = approx_scores_arr[compressed_tokens].astype(np.float64, copy=True)
    probe_count = min(max(0, int(calibration_probes)), int(compressed_tokens.size))
    if probe_count <= 0:
        return scores, cost, 0

    ordered_arr = _ordered_subset_by_candidate_rank(result, compressed_tokens)
    band_count = max(1, min(int(calibration_bands), int(ordered_arr.size)))
    ordered_bands = [arr.astype(np.int64, copy=False) for arr in np.array_split(ordered_arr, band_count)]
    per_band = max(1, int(np.ceil(float(probe_count) / float(band_count))))
    probed_parts = []
    calibrated = np.array(scores, copy=True)
    token_to_pos = {int(tok): idx for idx, tok in enumerate(compressed_tokens.tolist())}
    for band in ordered_bands:
        if band.size == 0:
            continue
        if per_band >= int(band.size):
            probe_tokens = band
        else:
            positions = np.unique(np.linspace(0, int(band.size) - 1, num=per_band, dtype=np.int64))
            probe_tokens = band[positions]
        probed_parts.append(probe_tokens)
        x = approx_scores_arr[probe_tokens].astype(np.float64, copy=False)
        y = state.scores[probe_tokens].astype(np.float64, copy=False)
        x_mean = float(np.mean(x))
        y_mean = float(np.mean(y))
        x_var = float(np.mean((x - x_mean) * (x - x_mean)))
        if x_var > 1e-12:
            slope = float(np.mean((x - x_mean) * (y - y_mean)) / x_var)
            intercept = y_mean - slope * x_mean
        else:
            slope = 1.0
            intercept = y_mean - x_mean
        positions_all = np.asarray([token_to_pos[int(tok)] for tok in band.tolist()], dtype=np.int64)
        calibrated[positions_all] = slope * approx_scores_arr[band] + intercept
        probe_positions = np.asarray([token_to_pos[int(tok)] for tok in probe_tokens.tolist()], dtype=np.int64)
        calibrated[probe_positions] = state.scores[probe_tokens].astype(np.float64, copy=False)
    probe_tokens = np.unique(np.concatenate(probed_parts)) if probed_parts else np.empty((0,), dtype=np.int64)
    cost.read("exact_attention", "kcomp_calibration_probe_keys", int(probe_tokens.size) * int(state.values.shape[-1]) * int(key_bytes))
    return calibrated, cost, int(probe_tokens.size)


def compressed_key_value_head_estimate(
    name: str,
    state: QueryState,
    result: SelectionResult,
    *,
    samples: int,
    seed: int,
    bands: int,
    key_mode: str,
    value_mode: str,
    exact_value_top: int,
    key_bytes: int,
    value_bytes: int,
    calibration_probes: int = 0,
    calibration_bands: int = 1,
) -> TailEstimate:
    """Compressed selected-head estimator with independently selectable K and V formats.

    ``key_mode=exact`` keeps exact logits and charges exact K reads.  The
    compressed-K mode currently supported here is rank-band calibrated PQ.
    ``value_mode=vpq`` reconstructs all covered selected V vectors from V-PQ;
    ``mixed_vpq`` keeps the first ``exact_value_top`` selected ranks exact and
    uses V-PQ for the remaining covered selected ranks.
    """

    selected, tail = _selected_and_tail(state, result)
    sampled_tail = _sample_stratified_tail(
        name,
        state,
        result,
        selected,
        tail,
        samples=samples,
        seed=seed,
        bands=bands,
    )
    dim = int(state.values.shape[-1])
    cost = CostTrace()
    approx_scores = result.metadata.get("approx_scores")
    approx_scores_arr = np.asarray(approx_scores, dtype=np.float64) if approx_scores is not None else None

    if key_mode == "exact" or approx_scores_arr is None:
        compressed_tokens = np.asarray([], dtype=np.int64)
        compressed_scores = np.empty((0,), dtype=np.float64)
        exact_tokens = selected
        key_probe_count = 0
    else:
        valid = (
            (selected >= 0)
            & (selected < approx_scores_arr.shape[0])
            & np.isfinite(approx_scores_arr[selected])
        )
        compressed_tokens = selected[valid]
        exact_tokens = selected[~valid]
        if key_mode != "band_calibrated_pq":
            raise ValueError(f"unsupported compressed key mode: {key_mode}")
        compressed_scores, calibration_cost, key_probe_count = _band_calibrated_scores(
            state,
            result,
            compressed_tokens,
            approx_scores_arr,
            calibration_probes=calibration_probes,
            calibration_bands=calibration_bands,
            key_bytes=key_bytes,
        )
        cost.extend(calibration_cost)

    max_values = []
    if compressed_scores.size:
        max_values.append(float(np.max(compressed_scores)))
    if exact_tokens.size:
        max_values.append(float(np.max(state.scores[exact_tokens])))
    for sample, _stratum_size in sampled_tail:
        if sample.size:
            max_values.append(float(np.max(state.scores[sample])))
    max_score = max(max_values) if max_values else float(np.max(state.scores))

    head_num = np.zeros((dim,), dtype=np.float64)
    head_den = 0.0
    vpq_count = 0
    exact_value_count = 0
    fallback_count = 0
    read_pages: set[int] = set()

    if compressed_tokens.size:
        weights = np.exp(compressed_scores - max_score)
        page_ids, vpq_values = _vpq_value_lookup(result, compressed_tokens)
        vpq_covered = page_ids >= 0
        if value_mode == "mixed_vpq" and int(exact_value_top) > 0:
            ordered = _ordered_subset_by_candidate_rank(result, compressed_tokens)
            exact_value_set = set(int(tok) for tok in ordered[: max(0, int(exact_value_top))].tolist())
            exact_value_mask = np.asarray([int(tok) in exact_value_set for tok in compressed_tokens.tolist()], dtype=bool)
        elif value_mode == "vpq":
            exact_value_mask = np.zeros((compressed_tokens.size,), dtype=bool)
        else:
            exact_value_mask = np.ones((compressed_tokens.size,), dtype=bool)
        exact_value_mask |= ~vpq_covered
        vpq_mask = ~exact_value_mask
        if np.any(vpq_mask):
            head_num += weights[vpq_mask] @ vpq_values[vpq_mask]
            head_den += float(weights[vpq_mask].sum())
            vpq_count = int(np.count_nonzero(vpq_mask))
            read_page_ids = np.unique(page_ids[vpq_mask]).astype(np.int64, copy=False)
            read_pages.update(int(page) for page in read_page_ids.tolist())
            cost.read(
                "exact_attention",
                "compressed_kv_value_codebooks",
                len(read_page_ids) * int(result.metadata.get("value_vpq_codebook_bytes_per_page", 0)),
            )
            cost.read(
                "exact_attention",
                "compressed_kv_value_codes",
                vpq_count * int(result.metadata.get("value_vpq_code_bytes_per_token", 0)),
            )
            sidecar_update = float(result.metadata.get("value_vpq_sidecar_update_cumulative_bytes", 0.0))
            if sidecar_update > 0.0:
                cost.write(
                    "exact_attention",
                    "compressed_kv_value_sidecar_update_amortized",
                    sidecar_update / max(1, int(state.decode_tokens)),
                )
        if np.any(exact_value_mask):
            exact_value_tokens = compressed_tokens[exact_value_mask]
            exact_weights = weights[exact_value_mask]
            head_num += exact_weights @ state.values[exact_value_tokens].astype(np.float64, copy=False)
            head_den += float(exact_weights.sum())
            exact_value_count = int(exact_value_tokens.size)
            if key_mode == "exact":
                cost.read("exact_attention", "compressed_kv_exact_keys", exact_value_count * dim * int(key_bytes))
            cost.read("exact_attention", "compressed_kv_exact_values", exact_value_count * dim * int(value_bytes))

    if exact_tokens.size:
        exact_num, exact_den = _head_terms(state, exact_tokens, max_score)
        head_num += exact_num
        head_den += exact_den
        fallback_count = int(exact_tokens.size)
        cost.read("exact_attention", "compressed_kv_exact_fallback_kv", kv_read_bytes(exact_tokens.size, dim, key_bytes, value_bytes))

    tail_num = np.zeros((dim,), dtype=np.float64)
    tail_den = 0.0
    variance_proxy = 0.0
    alloc_total = 0
    for sample, stratum_size in sampled_tail:
        weights = np.exp(state.scores[sample].astype(np.float64) - max_score)
        scale = float(stratum_size) / float(sample.size)
        tail_num += scale * (weights @ state.values[sample].astype(np.float64, copy=False))
        tail_den += scale * float(weights.sum())
        variance_proxy += float(np.var(weights * float(stratum_size), ddof=1) / max(1, sample.size)) if sample.size > 1 else 0.0
        alloc_total += int(sample.size)
    if alloc_total:
        cost.read("tail_estimator", "compressed_kv_stratified_tail_sample_exact_kv", kv_read_bytes(alloc_total, dim, key_bytes, value_bytes))

    output = (head_num + tail_num) / max(head_den + tail_den, 1e-20)
    return TailEstimate(
        name=name,
        output=output.astype(np.float32),
        cost=cost,
        metadata={
            "tail_estimator": name,
            "tail_kind": "compressed_key_value",
            "tail_samples": alloc_total,
            "tail_population": int(tail.size),
            "tail_estimator_variance": variance_proxy,
            "oracle_diagnostic": False,
            "replaces_exact_attention": True,
            "compressed_head_tokens": int(compressed_tokens.size),
            "compressed_head_exact_fallback_tokens": fallback_count,
            "compressed_kv_vpq_tokens": vpq_count,
            "compressed_kv_exact_value_tokens": exact_value_count,
            "compressed_head_pages": int(len(read_pages)),
            "compressed_key_mode": str(key_mode),
            "compressed_value_mode": str(value_mode),
            "compressed_value_exact_top": int(exact_value_top),
            "compressed_key_calibration_probes": int(calibration_probes),
            "compressed_key_calibration_probe_count": int(key_probe_count),
            "compressed_key_calibration_bands": int(calibration_bands),
        },
    )


def turboquant_head_estimate(
    name: str,
    state: QueryState,
    result: SelectionResult,
    *,
    samples: int,
    seed: int,
    bands: int,
    key_bits: int,
    value_bits: int,
    product_residual: bool,
    key_bytes: int,
    value_bytes: int,
) -> TailEstimate:
    """TurboQuant-inspired selected-head compression proxy.

    This is intentionally a proxy, not a faithful kernel: it uses a fixed
    sign-Hadamard rotation and scalar Gaussian codebooks. ``product_residual``
    adds a 1-bit rotated residual correction to stress-test the TurboQuant
    claim that inner-product-optimized quantization is more useful for K than
    plain MSE reconstruction.
    """

    selected, tail = _selected_and_tail(state, result)
    sampled_tail = _sample_stratified_tail(
        name,
        state,
        result,
        selected,
        tail,
        samples=samples,
        seed=seed,
        bands=bands,
    )
    compressed_tokens, exact_tokens = _tq_dynamic_tokens(result, selected)
    dim = int(state.values.shape[-1])
    cost = CostTrace()

    key_hat = (
        _tq_reconstruct_product(state.keys[compressed_tokens], key_bits)
        if product_residual
        else _tq_reconstruct(state.keys[compressed_tokens], key_bits)[0]
    )
    compressed_scores = (key_hat.astype(np.float64, copy=False) @ state.query.astype(np.float64, copy=False)) if compressed_tokens.size else np.empty((0,), dtype=np.float64)

    max_values = []
    if compressed_scores.size:
        max_values.append(float(np.max(compressed_scores)))
    if exact_tokens.size:
        max_values.append(float(np.max(state.scores[exact_tokens])))
    for sample, _stratum_size in sampled_tail:
        if sample.size:
            max_values.append(float(np.max(state.scores[sample])))
    max_score = max(max_values) if max_values else float(np.max(state.scores))

    head_num = np.zeros((dim,), dtype=np.float64)
    head_den = 0.0
    if compressed_tokens.size:
        weights = np.exp(compressed_scores - max_score)
        if int(value_bits) > 0:
            value_hat = _tq_reconstruct(state.values[compressed_tokens], value_bits)[0]
            head_num += weights @ value_hat.astype(np.float64, copy=False)
            cost.read("exact_attention", "tq_value_codes", _tq_code_bytes(compressed_tokens.size, dim, value_bits))
            cost.read("exact_attention", "tq_value_norms", int(compressed_tokens.size) * int(value_bytes))
        else:
            head_num += weights @ state.values[compressed_tokens].astype(np.float64, copy=False)
            cost.read("exact_attention", "tq_exact_values", int(compressed_tokens.size) * dim * int(value_bytes))
        head_den += float(weights.sum())
        cost.read("exact_attention", "tq_key_codes", _tq_code_bytes(compressed_tokens.size, dim, key_bits))
        cost.read("exact_attention", "tq_key_norms", int(compressed_tokens.size) * int(key_bytes))
        if product_residual:
            cost.read("exact_attention", "tq_key_residual_signs", _tq_code_bytes(compressed_tokens.size, dim, 1))
            cost.read("exact_attention", "tq_key_residual_scales", int(compressed_tokens.size) * int(key_bytes))

    if exact_tokens.size:
        exact_num, exact_den = _head_terms(state, exact_tokens, max_score)
        head_num += exact_num
        head_den += exact_den
        cost.read("exact_attention", "tq_exact_fallback_kv", kv_read_bytes(exact_tokens.size, dim, key_bytes, value_bytes))

    # Online sidecar write proxy: each new token must write compressed K and,
    # when enabled, compressed V. Global rotation/codebooks are fixed and not
    # charged per token.
    per_token_update = _tq_code_bytes(1, dim, key_bits) + int(key_bytes)
    if product_residual:
        per_token_update += _tq_code_bytes(1, dim, 1) + int(key_bytes)
    if int(value_bits) > 0:
        per_token_update += _tq_code_bytes(1, dim, value_bits) + int(value_bytes)
    cost.write("exact_attention", "tq_online_sidecar_write_per_token", per_token_update)

    tail_num = np.zeros((dim,), dtype=np.float64)
    tail_den = 0.0
    variance_proxy = 0.0
    alloc_total = 0
    for sample, stratum_size in sampled_tail:
        weights = np.exp(state.scores[sample].astype(np.float64) - max_score)
        scale = float(stratum_size) / float(sample.size)
        tail_num += scale * (weights @ state.values[sample].astype(np.float64, copy=False))
        tail_den += scale * float(weights.sum())
        variance_proxy += float(np.var(weights * float(stratum_size), ddof=1) / max(1, sample.size)) if sample.size > 1 else 0.0
        alloc_total += int(sample.size)
    if alloc_total:
        cost.read("tail_estimator", "tq_stratified_tail_sample_exact_kv", kv_read_bytes(alloc_total, dim, key_bytes, value_bytes))

    output = (head_num + tail_num) / max(head_den + tail_den, 1e-20)
    return TailEstimate(
        name=name,
        output=output.astype(np.float32),
        cost=cost,
        metadata={
            "tail_estimator": name,
            "tail_kind": "turboquant_proxy",
            "tail_samples": alloc_total,
            "tail_population": int(tail.size),
            "tail_estimator_variance": variance_proxy,
            "oracle_diagnostic": False,
            "replaces_exact_attention": True,
            "compressed_head_tokens": int(compressed_tokens.size),
            "compressed_head_exact_fallback_tokens": int(exact_tokens.size),
            "turboquant_key_bits": int(key_bits),
            "turboquant_value_bits": int(value_bits),
            "turboquant_product_residual": bool(product_residual),
            "turboquant_proxy": True,
        },
    )


def compressed_value_head_estimate(
    name: str,
    state: QueryState,
    result: SelectionResult,
    *,
    samples: int,
    seed: int,
    bands: int,
    allocation: str,
    key_bytes: int,
    value_bytes: int,
) -> TailEstimate:
    selected, tail = _selected_and_tail(state, result)
    sampled_tail = _sample_stratified_tail(
        name,
        state,
        result,
        selected,
        tail,
        samples=samples,
        seed=seed,
        bands=bands,
    )
    max_values = []
    if selected.size:
        max_values.append(float(np.max(state.scores[selected])))
    for sample, _stratum_size in sampled_tail:
        if sample.size:
            max_values.append(float(np.max(state.scores[sample])))
    max_score = max(max_values) if max_values else float(np.max(state.scores))

    cost = CostTrace()
    dim = int(state.values.shape[-1])
    head_num = np.zeros((dim,), dtype=np.float64)
    head_den = 0.0
    compressed_count = 0
    fallback_count = 0
    read_pages = 0
    if selected.size:
        page_ids, approx_values = _vpq_value_lookup(result, selected)
        compressed_mask = page_ids >= 0
        compressed_tokens = selected[compressed_mask]
        exact_tokens = selected[~compressed_mask]
        if compressed_tokens.size:
            weights = np.exp(state.scores[compressed_tokens].astype(np.float64) - max_score)
            head_num += weights @ approx_values[compressed_mask]
            head_den += float(weights.sum())
            compressed_count = int(compressed_tokens.size)
            read_page_ids = np.unique(page_ids[compressed_mask]).astype(np.int64, copy=False)
            read_pages = int(read_page_ids.size)
            cost.read(
                "exact_attention",
                "compressed_value_head_exact_keys",
                int(compressed_tokens.size) * dim * int(key_bytes),
            )
            cost.read(
                "exact_attention",
                "compressed_value_head_value_codebooks",
                read_pages * int(result.metadata.get("value_vpq_codebook_bytes_per_page", 0)),
            )
            cost.read(
                "exact_attention",
                "compressed_value_head_value_codes",
                int(compressed_tokens.size) * int(result.metadata.get("value_vpq_code_bytes_per_token", 0)),
            )
            sidecar_update = float(result.metadata.get("value_vpq_sidecar_update_cumulative_bytes", 0.0))
            if sidecar_update > 0.0:
                cost.write(
                    "exact_attention",
                    "compressed_value_head_value_sidecar_update_amortized",
                    sidecar_update / max(1, int(state.decode_tokens)),
                )
        if exact_tokens.size:
            exact_num, exact_den = _head_terms(state, exact_tokens, max_score)
            head_num += exact_num
            head_den += exact_den
            fallback_count = int(exact_tokens.size)
            cost.read(
                "exact_attention",
                "compressed_value_head_exact_fallback_kv",
                kv_read_bytes(exact_tokens.size, dim, key_bytes, value_bytes),
            )

    tail_num = np.zeros((dim,), dtype=np.float64)
    tail_den = 0.0
    variance_proxy = 0.0
    alloc_total = 0
    for sample, stratum_size in sampled_tail:
        weights = np.exp(state.scores[sample].astype(np.float64) - max_score)
        scale = float(stratum_size) / float(sample.size)
        tail_num += scale * (weights @ state.values[sample].astype(np.float64, copy=False))
        tail_den += scale * float(weights.sum())
        variance_proxy += float(np.var(weights * float(stratum_size), ddof=1) / max(1, sample.size)) if sample.size > 1 else 0.0
        alloc_total += int(sample.size)
    if alloc_total:
        cost.read(
            "tail_estimator",
            "vpq_head_stratified_tail_sample_exact_kv",
            kv_read_bytes(alloc_total, dim, key_bytes, value_bytes),
        )

    output = (head_num + tail_num) / max(head_den + tail_den, 1e-20)
    return TailEstimate(
        name=name,
        output=output.astype(np.float32),
        cost=cost,
        metadata={
            "tail_estimator": name,
            "tail_kind": "value_pq_head_compression",
            "tail_samples": alloc_total,
            "tail_population": int(tail.size),
            "tail_estimator_variance": variance_proxy,
            "oracle_diagnostic": False,
            "replaces_exact_attention": True,
            "compressed_head_tokens": compressed_count,
            "compressed_head_exact_fallback_tokens": fallback_count,
            "compressed_head_pages": read_pages,
        },
    )


def _pq_value_approx_components(
    state: QueryState,
    result: SelectionResult,
    tokens: np.ndarray,
    max_score: float,
) -> tuple[np.ndarray, float, np.ndarray]:
    approx_scores = result.metadata.get("approx_scores")
    if approx_scores is None or tokens.size == 0:
        return np.zeros((state.values.shape[-1],), dtype=np.float64), 0.0, np.asarray([], dtype=np.int64)
    approx_scores_arr = np.asarray(approx_scores, dtype=np.float64)
    valid = (tokens >= 0) & (tokens < approx_scores_arr.shape[0]) & np.isfinite(approx_scores_arr[tokens])
    if not np.any(valid):
        return np.zeros((state.values.shape[-1],), dtype=np.float64), 0.0, np.asarray([], dtype=np.int64)
    valid_tokens = tokens[valid]
    page_ids, approx_values = _pq_value_lookup(result, valid_tokens)
    weights = np.exp(approx_scores_arr[valid_tokens] - float(max_score))
    numerator = weights @ approx_values
    return numerator, float(weights.sum()), np.unique(page_ids[page_ids >= 0]).astype(np.int64, copy=False)


def _cv_approx_components(
    state: QueryState,
    result: SelectionResult,
    tokens: np.ndarray,
    max_score: float,
    cv_mode: str,
) -> tuple[np.ndarray, float, np.ndarray]:
    if cv_mode == "pq_value":
        return _pq_value_approx_components(state, result, tokens, max_score)
    return _page_mean_approx_components(state, result, tokens, max_score)


def _cv_value_lookup(result: SelectionResult, tokens: np.ndarray, cv_mode: str) -> tuple[np.ndarray, np.ndarray]:
    if cv_mode == "pq_value":
        return _pq_value_lookup(result, tokens)
    return _page_mean_value_lookup(result, tokens)


def uniform_tail_estimate(
    name: str,
    state: QueryState,
    result: SelectionResult,
    *,
    samples: int,
    seed: int,
    key_bytes: int,
    value_bytes: int,
) -> TailEstimate:
    selected, tail = _selected_and_tail(state, result)
    max_score = float(np.max(state.scores))
    head_num, head_den = _head_terms(state, selected, max_score)
    sample_count = min(max(0, int(samples)), int(tail.size))
    cost = CostTrace()
    if sample_count == 0 or tail.size == 0:
        denom = max(head_den, 1e-20)
        return TailEstimate(
            name=name,
            output=(head_num / denom).astype(np.float32),
            cost=cost,
            metadata={
                "tail_estimator": name,
                "tail_kind": "uniform_tail",
                "tail_samples": sample_count,
                "tail_population": int(tail.size),
                "tail_estimator_variance": 0.0,
                "oracle_diagnostic": False,
            },
        )
    rng = _make_rng(seed, state, name)
    sample = rng.choice(tail, size=sample_count, replace=False)
    weights = np.exp(state.scores[sample].astype(np.float64) - max_score)
    scale = float(tail.size) / float(sample_count)
    tail_num = scale * (weights @ state.values[sample].astype(np.float64, copy=False))
    tail_den = scale * float(weights.sum())
    output = (head_num + tail_num) / max(head_den + tail_den, 1e-20)
    cost.read(
        "tail_estimator",
        "tail_sample_exact_kv",
        kv_read_bytes(sample_count, state.values.shape[-1], key_bytes, value_bytes),
    )
    # Scalar diagnostic: variance proxy for the sampled denominator estimate.
    denom_samples = float(tail.size) * weights
    variance = float(np.var(denom_samples, ddof=1) / max(1, sample_count)) if sample_count > 1 else 0.0
    return TailEstimate(
        name=name,
        output=output.astype(np.float32),
        cost=cost,
        metadata={
            "tail_estimator": name,
            "tail_kind": "uniform_tail",
            "tail_samples": sample_count,
            "tail_population": int(tail.size),
            "tail_estimator_variance": variance,
            "oracle_diagnostic": False,
        },
    )


def stratified_tail_estimate(
    name: str,
    state: QueryState,
    result: SelectionResult,
    *,
    samples: int,
    seed: int,
    bands: int,
    allocation: str,
    control_variate: bool | str,
    key_bytes: int,
    value_bytes: int,
) -> TailEstimate:
    selected, tail = _selected_and_tail(state, result)
    max_score = float(np.max(state.scores))
    head_num, head_den = _head_terms(state, selected, max_score)
    sample_count = min(max(0, int(samples)), int(tail.size))
    cost = CostTrace()
    cv_mode = "none"
    if isinstance(control_variate, str):
        cv_mode = control_variate
    elif control_variate:
        cv_mode = "page_mean"
    bands = max(1, int(bands))
    if sample_count == 0 or tail.size == 0:
        denom = max(head_den, 1e-20)
        return TailEstimate(
            name=name,
            output=(head_num / denom).astype(np.float32),
            cost=cost,
            metadata={
                "tail_estimator": name,
                "tail_kind": "stratified_tail",
                "tail_samples": sample_count,
                "tail_population": int(tail.size),
                "tail_estimator_variance": 0.0,
                "oracle_diagnostic": False,
                "tail_strata": bands,
                "tail_allocation": allocation,
                "tail_control_variate": cv_mode != "none",
                "tail_control_variate_mode": cv_mode,
            },
        )

    ordered = _tail_order_from_candidates(state, result, selected, tail)
    strata = [arr.astype(np.int64, copy=False) for arr in np.array_split(ordered, min(bands, int(ordered.size)))]
    alloc = _allocate_samples(
        strata=strata,
        total_samples=sample_count,
        allocation=allocation,
        state=state,
        result=result,
        max_score=max_score,
    )
    rng = _make_rng(seed, state, name)

    tail_num = np.zeros((state.values.shape[-1],), dtype=np.float64)
    tail_den = 0.0
    variance_proxy = 0.0
    cv_num_total = np.zeros((state.values.shape[-1],), dtype=np.float64)
    cv_den_total = 0.0
    cv_read_pages: set[int] = set()
    for stratum, stratum_samples in zip(strata, alloc, strict=False):
        if stratum.size == 0:
            continue
        if cv_mode != "none":
            cv_num, cv_den, read_pages = _cv_approx_components(state, result, stratum, max_score, cv_mode)
            cv_num_total += cv_num
            cv_den_total += cv_den
            cv_read_pages.update(int(page) for page in read_pages.tolist())
        if stratum_samples <= 0:
            continue
        sample = rng.choice(stratum, size=min(int(stratum_samples), int(stratum.size)), replace=False)
        exact_w = np.exp(state.scores[sample].astype(np.float64) - max_score)
        exact_num_samples = exact_w[:, None] * state.values[sample].astype(np.float64, copy=False)
        exact_den_samples = exact_w
        if cv_mode != "none" and result.metadata.get("approx_scores") is not None:
            approx_scores = np.asarray(result.metadata["approx_scores"], dtype=np.float64)
            _sample_page_ids, approx_values = _cv_value_lookup(result, sample, cv_mode)
            finite = np.isfinite(approx_scores[sample])
            approx_w = np.where(finite, np.exp(approx_scores[sample] - max_score), 0.0)
            sample_num = exact_num_samples - approx_w[:, None] * approx_values
            sample_den = exact_den_samples - approx_w
        else:
            sample_num = exact_num_samples
            sample_den = exact_den_samples
        scale = float(stratum.size) / float(sample.size)
        tail_num += scale * sample_num.sum(axis=0)
        tail_den += scale * float(sample_den.sum())
        variance_proxy += float(np.var(sample_den * float(stratum.size), ddof=1) / max(1, sample.size)) if sample.size > 1 else 0.0

    if cv_mode != "none":
        tail_num += cv_num_total
        tail_den += cv_den_total
    output = (head_num + tail_num) / max(head_den + tail_den, 1e-20)
    cost.read(
        "tail_estimator",
        "stratified_tail_sample_exact_kv",
        kv_read_bytes(sum(alloc), state.values.shape[-1], key_bytes, value_bytes),
    )
    if cv_mode == "page_mean" and cv_read_pages:
        cost.read(
            "tail_estimator",
            "stratified_tail_page_mean_values",
            len(cv_read_pages) * state.values.shape[-1] * value_bytes,
        )
    elif cv_mode == "pq_value" and cv_read_pages:
        cost.read(
            "tail_estimator",
            "stratified_tail_pq_value_codebooks",
            len(cv_read_pages) * int(result.metadata.get("value_pq_codebook_bytes_per_page", 0)),
        )
        sidecar_update = float(result.metadata.get("value_pq_sidecar_update_cumulative_bytes", 0.0))
        if sidecar_update > 0.0:
            # Amortize page-seal V-side centroid construction over generated tokens,
            # analogous to the selector online-update MB/token term.
            cost.write(
                "tail_estimator",
                "stratified_tail_pq_value_sidecar_update_amortized",
                sidecar_update / max(1, int(state.decode_tokens)),
            )
    return TailEstimate(
        name=name,
        output=output.astype(np.float32),
        cost=cost,
        metadata={
            "tail_estimator": name,
            "tail_kind": "stratified_tail",
            "tail_samples": int(sum(alloc)),
            "tail_population": int(tail.size),
            "tail_estimator_variance": variance_proxy,
            "oracle_diagnostic": False,
            "tail_strata": bands,
            "tail_allocation": allocation,
            "tail_control_variate": cv_mode != "none",
            "tail_control_variate_mode": cv_mode,
        },
    )


def oracle_prob_tail_estimate(
    name: str,
    state: QueryState,
    result: SelectionResult,
    *,
    samples: int,
    seed: int,
    key_bytes: int,
    value_bytes: int,
) -> TailEstimate:
    selected, tail = _selected_and_tail(state, result)
    max_score = float(np.max(state.scores))
    head_num, head_den = _head_terms(state, selected, max_score)
    sample_count = min(max(0, int(samples)), int(tail.size))
    cost = CostTrace()
    if sample_count == 0 or tail.size == 0:
        denom = max(head_den, 1e-20)
        return TailEstimate(
            name=name,
            output=(head_num / denom).astype(np.float32),
            cost=cost,
            metadata={
                "tail_estimator": name,
                "tail_kind": "oracle_prob_tail",
                "tail_samples": sample_count,
                "tail_population": int(tail.size),
                "tail_estimator_variance": 0.0,
                "oracle_diagnostic": True,
            },
        )
    tail_probs = state.probs[tail].astype(np.float64, copy=False)
    tail_mass = max(float(tail_probs.sum()), 1e-20)
    proposal = tail_probs / tail_mass
    rng = _make_rng(seed, state, name)
    sample = rng.choice(tail, size=sample_count, replace=True, p=proposal)
    # This is an oracle diagnostic: true tail mass/proposal are not deployable.
    z_total = float(np.exp(state.scores.astype(np.float64) - max_score).sum())
    tail_den = tail_mass * z_total
    tail_num = tail_den * state.values[sample].astype(np.float64, copy=False).mean(axis=0)
    output = (head_num + tail_num) / max(head_den + tail_den, 1e-20)
    cost.read(
        "tail_estimator",
        "tail_sample_exact_kv",
        kv_read_bytes(sample_count, state.values.shape[-1], key_bytes, value_bytes),
    )
    value_norms = np.linalg.norm(state.values[sample].astype(np.float64, copy=False), axis=1)
    variance = float(np.var(value_norms, ddof=1) / max(1, sample_count)) if sample_count > 1 else 0.0
    return TailEstimate(
        name=name,
        output=output.astype(np.float32),
        cost=cost,
        metadata={
            "tail_estimator": name,
            "tail_kind": "oracle_prob_tail",
            "tail_samples": sample_count,
            "tail_population": int(tail.size),
            "tail_estimator_variance": variance,
            "oracle_diagnostic": True,
        },
    )


def rank_tail_estimate(
    name: str,
    state: QueryState,
    result: SelectionResult,
    *,
    samples: int,
    seed: int,
    key_bytes: int,
    value_bytes: int,
) -> TailEstimate:
    """Importance-sample tail tokens using selector candidate rank.

    This is deployable for selectors whose candidate order is produced from
    compressed selector scores. For oracle selectors it remains diagnostic
    because their candidate order is oracle-ranked.
    """
    selected, tail = _selected_and_tail(state, result)
    max_score = float(np.max(state.scores))
    head_num, head_den = _head_terms(state, selected, max_score)
    sample_count = min(max(0, int(samples)), int(tail.size))
    cost = CostTrace()
    if sample_count == 0 or tail.size == 0:
        denom = max(head_den, 1e-20)
        return TailEstimate(
            name=name,
            output=(head_num / denom).astype(np.float32),
            cost=cost,
            metadata={
                "tail_estimator": name,
                "tail_kind": "rank_tail",
                "tail_samples": sample_count,
                "tail_population": int(tail.size),
                "tail_estimator_variance": 0.0,
                "oracle_diagnostic": False,
            },
        )

    selected_set = set(int(tok) for tok in selected.tolist())
    tail_set = set(int(tok) for tok in tail.tolist())
    ordered = []
    seen = set()
    for tok in result.candidate_tokens:
        tok = int(tok)
        if tok in selected_set or tok not in tail_set or tok in seen:
            continue
        ordered.append(tok)
        seen.add(tok)
    if len(ordered) < int(tail.size):
        ordered.extend(int(tok) for tok in tail.tolist() if int(tok) not in seen)
    ordered_arr = np.asarray(ordered, dtype=np.int64)
    ranks = np.arange(1, ordered_arr.size + 1, dtype=np.float64)
    proposal = 1.0 / ranks
    proposal /= max(float(proposal.sum()), 1e-20)
    rng = _make_rng(seed, state, name)
    rel_sample = rng.choice(np.arange(ordered_arr.size), size=sample_count, replace=True, p=proposal)
    sample = ordered_arr[rel_sample]
    p_sample = proposal[rel_sample]
    weights = np.exp(state.scores[sample].astype(np.float64) - max_score)
    contrib_scale = weights / np.maximum(p_sample, 1e-30)
    tail_num = (contrib_scale[:, None] * state.values[sample].astype(np.float64, copy=False)).mean(axis=0)
    tail_den = float(contrib_scale.mean())
    output = (head_num + tail_num) / max(head_den + tail_den, 1e-20)
    cost.read(
        "tail_estimator",
        "tail_sample_exact_kv",
        kv_read_bytes(sample_count, state.values.shape[-1], key_bytes, value_bytes),
    )
    variance = float(np.var(contrib_scale, ddof=1) / max(1, sample_count)) if sample_count > 1 else 0.0
    return TailEstimate(
        name=name,
        output=output.astype(np.float32),
        cost=cost,
        metadata={
            "tail_estimator": name,
            "tail_kind": "rank_tail",
            "tail_samples": sample_count,
            "tail_population": int(tail.size),
            "tail_estimator_variance": variance,
            "oracle_diagnostic": str(result.algorithm).startswith("top_fraction_oracle"),
        },
    )


def uniform_head_estimate(
    name: str,
    state: QueryState,
    result: SelectionResult,
    *,
    samples: int,
    seed: int,
    key_bytes: int,
    value_bytes: int,
) -> TailEstimate:
    """Diagnostic flip: exact-read the unselected tail and estimate selected head.

    This is deployable as a Monte Carlo estimator, but it is expected to be a
    poor algorithm because the exact tail is usually most of the context and
    the selected head has high per-token variance.
    """
    selected, tail = _selected_and_tail(state, result)
    max_score = float(np.max(state.scores))
    cost = CostTrace()

    if tail.size:
        tail_weights = np.exp(state.scores[tail].astype(np.float64) - max_score)
        tail_num = tail_weights @ state.values[tail].astype(np.float64, copy=False)
        tail_den = float(tail_weights.sum())
        cost.read(
            "exact_attention",
            "flipped_exact_tail_kv",
            kv_read_bytes(tail.size, state.values.shape[-1], key_bytes, value_bytes),
        )
    else:
        tail_num = np.zeros((state.values.shape[-1],), dtype=np.float64)
        tail_den = 0.0

    sample_count = min(max(0, int(samples)), int(selected.size))
    if sample_count == 0 or selected.size == 0:
        output = tail_num / max(tail_den, 1e-20)
        return TailEstimate(
            name=name,
            output=output.astype(np.float32),
            cost=cost,
            metadata={
                "tail_estimator": name,
                "tail_kind": "uniform_head",
                "tail_samples": sample_count,
                "tail_population": int(selected.size),
                "tail_estimator_variance": 0.0,
                "oracle_diagnostic": False,
                "replaces_exact_attention": True,
            },
        )

    rng = _make_rng(seed, state, name)
    sample = rng.choice(selected, size=sample_count, replace=False)
    weights = np.exp(state.scores[sample].astype(np.float64) - max_score)
    scale = float(selected.size) / float(sample_count)
    head_num = scale * (weights @ state.values[sample].astype(np.float64, copy=False))
    head_den = scale * float(weights.sum())
    output = (tail_num + head_num) / max(tail_den + head_den, 1e-20)
    cost.read(
        "tail_estimator",
        "flipped_head_sample_exact_kv",
        kv_read_bytes(sample_count, state.values.shape[-1], key_bytes, value_bytes),
    )
    denom_samples = float(selected.size) * weights
    variance = float(np.var(denom_samples, ddof=1) / max(1, sample_count)) if sample_count > 1 else 0.0
    return TailEstimate(
        name=name,
        output=output.astype(np.float32),
        cost=cost,
        metadata={
            "tail_estimator": name,
            "tail_kind": "uniform_head",
            "tail_samples": sample_count,
            "tail_population": int(selected.size),
            "tail_estimator_variance": variance,
            "oracle_diagnostic": False,
            "replaces_exact_attention": True,
        },
    )


def tail_output_metrics(state: QueryState, estimate: TailEstimate) -> dict[str, float]:
    return output_error_metrics(state, estimate.output)
