from __future__ import annotations

import numpy as np

from benchmark.selector_eval.selectors.base import QueryState, SelectionResult


def _renormalized_distribution(probs: np.ndarray, tokens: list[int]) -> np.ndarray:
    out = np.zeros_like(probs, dtype=np.float64)
    if not tokens:
        return out
    idx = np.asarray(sorted(set(int(tok) for tok in tokens)), dtype=np.int64)
    mass = max(float(probs[idx].sum()), 1e-20)
    out[idx] = probs[idx].astype(np.float64) / mass
    return out


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = p.astype(np.float64, copy=False)
    q = q.astype(np.float64, copy=False)
    m = 0.5 * (p + q)
    p_mask = p > 0
    q_mask = q > 0
    js = 0.0
    js += 0.5 * float(np.sum(p[p_mask] * np.log(p[p_mask] / m[p_mask])))
    js += 0.5 * float(np.sum(q[q_mask] * np.log(q[q_mask] / m[q_mask])))
    return js


def _softmax_from_scores(scores: np.ndarray) -> np.ndarray:
    scores64 = scores.astype(np.float64, copy=False)
    shifted = scores64 - float(np.max(scores64))
    probs = np.exp(shifted)
    probs /= max(float(np.sum(probs)), 1e-20)
    return probs.astype(np.float64, copy=False)


def _safe_distribution(probs: np.ndarray) -> np.ndarray:
    out = probs.astype(np.float64, copy=False)
    total = max(float(np.sum(out)), 1e-20)
    return out / total


def _topk_indices(values: np.ndarray, k: int) -> np.ndarray:
    count = min(max(1, int(k)), int(values.shape[0]))
    if count >= int(values.shape[0]):
        return np.arange(int(values.shape[0]), dtype=np.int64)
    idx = np.argpartition(values, -count)[-count:]
    return idx.astype(np.int64, copy=False)


def _logit_error_metrics(dense_scores: np.ndarray, approx_scores: np.ndarray, *, prefix: str = "logit") -> dict[str, float]:
    dense = dense_scores.astype(np.float64, copy=False)
    approx = approx_scores.astype(np.float64, copy=False)
    err = approx - dense
    rel_denom = max(float(np.linalg.norm(dense)), 1e-20)
    centered_dense = dense - float(np.mean(dense))
    centered_approx = approx - float(np.mean(approx))
    corr_denom = max(float(np.linalg.norm(centered_dense) * np.linalg.norm(centered_approx)), 1e-20)
    return {
        f"{prefix}_relL2": float(np.linalg.norm(err) / rel_denom),
        f"{prefix}_mean_abs": float(np.mean(np.abs(err))),
        f"{prefix}_max_abs": float(np.max(np.abs(err))),
        f"{prefix}_centered_cosine": float(np.dot(centered_dense, centered_approx) / corr_denom),
    }


def _probability_error_metrics(
    dense_probs: np.ndarray,
    approx_probs: np.ndarray,
    *,
    topk_sizes: tuple[int, ...] = (64, 512, 2048),
    prefix: str = "prob",
    eps: float = 1e-12,
) -> dict[str, float]:
    dense = _safe_distribution(dense_probs)
    approx = _safe_distribution(approx_probs)
    clipped = np.maximum(approx, float(eps))
    clipped /= max(float(np.sum(clipped)), 1e-20)
    diff = approx - dense
    out = {
        f"{prefix}_KL_dense_to_approx": float(np.sum(dense * np.log(np.maximum(dense, float(eps)) / clipped))),
        f"{prefix}_JS": float(_js_divergence(dense, approx)),
        f"{prefix}_TV": float(0.5 * np.sum(np.abs(diff))),
        f"{prefix}_L1": float(np.sum(np.abs(diff))),
        f"{prefix}_L2": float(np.linalg.norm(diff)),
        f"{prefix}_max_abs": float(np.max(np.abs(diff))),
        f"{prefix}_missing_mass": float(np.sum(dense[approx <= 0.0])),
        f"{prefix}_entropy_dense": float(-np.sum(dense * np.log(np.maximum(dense, float(eps))))),
        f"{prefix}_entropy_approx": float(-np.sum(approx * np.log(np.maximum(approx, float(eps))))),
    }
    for k in topk_sizes:
        kk = min(max(1, int(k)), int(dense.shape[0]))
        dense_top = _topk_indices(dense, kk)
        approx_top = _topk_indices(approx, kk)
        overlap = np.intersect1d(dense_top, approx_top, assume_unique=False)
        dense_top_mass = max(float(np.sum(dense[dense_top])), 1e-20)
        out[f"{prefix}_top{kk}_overlap"] = float(overlap.size) / float(kk)
        out[f"{prefix}_top{kk}_mass_recall"] = float(np.sum(dense[overlap]) / dense_top_mass)
        out[f"{prefix}_approx_top{kk}_dense_mass"] = float(np.sum(dense[approx_top]))
    return out


def attention_distribution_error_metrics(
    dense_scores: np.ndarray,
    dense_probs: np.ndarray,
    approx_scores: np.ndarray | None,
    approx_probs: np.ndarray | None,
    *,
    topk_sizes: tuple[int, ...] = (64, 512, 2048),
) -> dict[str, float]:
    out: dict[str, float] = {}
    if approx_scores is not None and dense_scores.shape == approx_scores.shape:
        out.update(_logit_error_metrics(dense_scores, approx_scores, prefix="logit"))
    if approx_probs is not None and dense_probs.shape == approx_probs.shape:
        out.update(_probability_error_metrics(dense_probs, approx_probs, topk_sizes=topk_sizes, prefix="prob"))
    return out


def _attention_output(scores: np.ndarray, values: np.ndarray, tokens: list[int]) -> np.ndarray:
    if not tokens:
        return np.zeros((values.shape[-1],), dtype=np.float32)
    idx = np.asarray(sorted(set(int(tok) for tok in tokens)), dtype=np.int64)
    logits = scores[idx].astype(np.float32)
    weights = np.exp(logits - np.max(logits)).astype(np.float32)
    weights /= max(float(weights.sum()), 1e-20)
    return weights @ values[idx].astype(np.float32, copy=False)


def _rms_normalize(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x64 = x.astype(np.float64, copy=False)
    rms = np.sqrt(float(np.mean(x64 * x64)) + float(eps))
    return x64 / max(rms, 1e-20)


def _output_error_metrics(dense_out: np.ndarray, approx_out: np.ndarray) -> dict[str, float]:
    dense = dense_out.astype(np.float64, copy=False)
    approx = approx_out.astype(np.float64, copy=False)
    err = approx - dense
    denom = max(float(np.linalg.norm(dense) * np.linalg.norm(approx)), 1e-20)
    rel_denom = max(float(np.linalg.norm(dense)), 1e-20)
    dense_abs = np.abs(dense)
    channel_scale = np.maximum(dense_abs, 1e-3)
    channel_rel = np.abs(err) / channel_scale
    dense_normed = _rms_normalize(dense)
    approx_normed = _rms_normalize(approx)
    normed_denom = max(float(np.linalg.norm(dense_normed)), 1e-20)
    centered_dense = dense - float(np.mean(dense))
    centered_approx = approx - float(np.mean(approx))
    centered_denom = max(float(np.linalg.norm(centered_dense) * np.linalg.norm(centered_approx)), 1e-20)
    return {
        "output_cosine": float(np.dot(dense, approx) / denom),
        "output_relative_l2": float(np.linalg.norm(err) / rel_denom),
        "output_rmsnorm_relative_l2": float(np.linalg.norm(approx_normed - dense_normed) / normed_denom),
        "output_centered_cosine": float(np.dot(centered_dense, centered_approx) / centered_denom),
        "output_mean_abs_relative_error": float(np.mean(channel_rel)),
        "output_p95_abs_relative_error": float(np.quantile(channel_rel, 0.95)),
        "output_p99_abs_relative_error": float(np.quantile(channel_rel, 0.99)),
        "output_max_abs_relative_error": float(np.max(channel_rel)),
        "output_linf_relative": float(np.max(np.abs(err)) / max(float(np.max(dense_abs)), 1e-20)),
    }


def compute_metrics(state: QueryState, result: SelectionResult, oracle_tokens: list[int]) -> dict[str, float]:
    selected = set(int(tok) for tok in result.selected_tokens)
    oracle = set(int(tok) for tok in oracle_tokens)
    selected_idx = np.asarray(sorted(selected), dtype=np.int64) if selected else np.empty((0,), dtype=np.int64)
    attention_mass = float(state.probs[selected_idx].sum()) if selected_idx.size else 0.0

    false_pos = selected - oracle
    false_neg = oracle - selected
    fp_idx = np.asarray(sorted(false_pos), dtype=np.int64) if false_pos else np.empty((0,), dtype=np.int64)
    fn_idx = np.asarray(sorted(false_neg), dtype=np.int64) if false_neg else np.empty((0,), dtype=np.int64)

    dense_out = state.probs.astype(np.float32) @ state.values.astype(np.float32, copy=False)
    sparse_out = _attention_output(state.scores, state.values, result.selected_tokens)
    q_dist = _renormalized_distribution(state.probs, result.selected_tokens)
    out_metrics = _output_error_metrics(dense_out, sparse_out)

    return {
        "attention_mass": attention_mass,
        "false_positive_mass": float(state.probs[fp_idx].sum()) if fp_idx.size else 0.0,
        "false_negative_mass": float(state.probs[fn_idx].sum()) if fn_idx.size else 0.0,
        "false_positive_tokens": float(len(false_pos)),
        "false_negative_tokens": float(len(false_neg)),
        "distribution_js": _js_divergence(state.probs, q_dist),
        **out_metrics,
    }


def output_error_metrics(state: QueryState, approx_out: np.ndarray) -> dict[str, float]:
    dense_out = state.probs.astype(np.float32) @ state.values.astype(np.float32, copy=False)
    approx = approx_out.astype(np.float32, copy=False)
    return _output_error_metrics(dense_out, approx)
