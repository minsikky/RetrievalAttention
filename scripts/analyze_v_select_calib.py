#!/usr/bin/env python3
"""Calibrated / predicted absolute-V-threshold analyses (issues #13, #14, #16).

Offline reduction over the V-threshold lab logs (v_threshold_lab_20260711).
Builds on the #12 verdict (single global theta is non-stationary across 3+
decades). No model, no torch, no GPU.

Substrate per head-step (N = 9216 = 288 qidx x 32 heads, single layer l16):
  - scan-domain features (deployable BEFORE V gather): pq_lse, pq_max_logit,
    pq_entropy, proxy_mass_90_count, code_error_{mean,p50,p95,max}, total_risk_mass,
    finite_risk_tokens, context_len, head / kv_head identity, and cheap
    "count >= probe" counters read off the theta grid;
  - canonical (oracle global-residual-risk) targets: selected_count, relL2,
    cutoff_risk (the ideal per-head-step theta), logical V bytes;
  - the 20-point theta grid curves (count, relL2, false-neg/false-pos risk mass)
    which we log-theta-interpolate to evaluate an arbitrary per-head-step theta.

bytes(theta) = 256 * count(theta)  (head_dim 128 x fp16). Verified constant.

Splits (guarding against qidx autocorrelation, corr(lag1 log-cutoff) ~ 0.91):
  - primary "blocked": contiguous blocks of 6 qidx (context-sorted), alternate
    blocks to train/test -> both splits span the full 6.8k-135k context range and
    all four context buckets, train/test adjacency limited to block boundaries.
  - robustness "even_odd" and "contiguous" (first 60% / last 40% qidx) reported
    alongside to show conclusions are split-stable.
All splits are BY QIDX (never split the 32 heads of one decode step across
train/test; heads within a qidx share the prompt/context).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict

import numpy as np

BYTES_PER_TOKEN = 256.0
MEET_TOL = 1.05  # relL2 within 1.05x canonical == "meets"

# ---------------------------------------------------------------- percentiles
def _pctl(xs, q):
    if len(xs) == 0:
        return float("nan")
    return float(np.percentile(np.asarray(xs, dtype=float), q))


def _stats(xs):
    xs = np.asarray(xs, dtype=float)
    if xs.size == 0:
        return {"n": 0, "mean": None, "p50": None, "p95": None, "p99": None, "max": None}
    return {
        "n": int(xs.size),
        "mean": float(xs.mean()),
        "p50": float(np.percentile(xs, 50)),
        "p95": float(np.percentile(xs, 95)),
        "p99": float(np.percentile(xs, 99)),
        "max": float(xs.max()),
    }


# ---------------------------------------------------------------- data loading
class Lab:
    """All per-head-step arrays, aligned by row index i in [0, N)."""

    def __init__(self, lab_dirs):
        hs = []
        for d in lab_dirs:
            fp = os.path.join(d, "v_threshold_headstep.csv")
            with open(fp, newline="") as f:
                hs.extend(list(csv.DictReader(f)))
        # theta grid rows keyed by (qidx, head)
        grid = defaultdict(list)
        for d in lab_dirs:
            fp = os.path.join(d, "v_threshold_theta.csv")
            with open(fp, newline="") as f:
                for r in csv.DictReader(f):
                    grid[(int(r["qidx"]), int(r["head"]))].append(r)

        N = len(hs)
        self.N = N
        self.qidx = np.array([int(r["qidx"]) for r in hs])
        self.head = np.array([int(r["head"]) for r in hs])
        self.kv_head = np.array([int(r["kv_head"]) for r in hs])
        self.layer = int(hs[0]["layer"])
        self.context_len = np.array([int(r["context_len"]) for r in hs], dtype=float)
        self.context_bucket = np.array([r["context_bucket"] for r in hs])
        self.position = np.array([int(r["position"]) for r in hs])

        self.c_count = np.array([float(r["canonical_selected_count"]) for r in hs])
        self.c_relL2 = np.array([float(r["canonical_relL2"]) for r in hs])
        self.c_cutoff = np.array([float(r["canonical_cutoff_risk"]) for r in hs])
        self.c_bytes = BYTES_PER_TOKEN * self.c_count
        self.total_risk_mass = np.array([float(r["total_risk_mass"]) for r in hs])
        self.finite_risk_tokens = np.array([float(r["finite_risk_tokens"]) for r in hs])

        self.pq_lse = np.array([float(r["pq_lse"]) for r in hs])
        self.pq_max_logit = np.array([float(r["pq_max_logit"]) for r in hs])
        self.pq_entropy = np.array([float(r["pq_entropy"]) for r in hs])
        self.proxy_mass_90 = np.array([float(r["proxy_mass_90_count"]) for r in hs])
        self.code_err_mean = np.array([float(r["code_error_mean"]) for r in hs])
        self.code_err_p50 = np.array([float(r["code_error_p50"]) for r in hs])
        self.code_err_p95 = np.array([float(r["code_error_p95"]) for r in hs])
        self.code_err_max = np.array([float(r["code_error_max"]) for r in hs])

        # theta grid (shared) + per-head-step grid metric arrays
        any_key = (int(hs[0]["qidx"]), int(hs[0]["head"]))
        gr = sorted(grid[any_key], key=lambda r: float(r["theta"]))
        self.theta_grid = np.array([float(r["theta"]) for r in gr])
        self.log_grid = np.log10(self.theta_grid)
        G = len(self.theta_grid)
        self.G = G
        self.grid_count = np.zeros((N, G))
        self.grid_relL2 = np.zeros((N, G))
        self.grid_fn_mass = np.zeros((N, G))
        self.grid_fp_mass = np.zeros((N, G))
        for i, r in enumerate(hs):
            rs = sorted(grid[(int(r["qidx"]), int(r["head"]))], key=lambda x: float(x["theta"]))
            self.grid_count[i] = [float(x["exact_selected_count"]) for x in rs]
            self.grid_relL2[i] = [float(x["relL2"]) for x in rs]
            self.grid_fn_mass[i] = [float(x["false_negative_risk_mass"]) for x in rs]
            self.grid_fp_mass[i] = [float(x["false_positive_risk_mass"]) for x in rs]

        # npz risk-distribution quantiles (descending; col0 = max risk). Genuine
        # order statistics -> require a histogram/CDF barrier to compute at scan EOS.
        qmap = {}
        for d in lab_dirs:
            fp = os.path.join(d, "v_threshold_curves.npz")
            if os.path.exists(fp):
                z = np.load(fp, allow_pickle=True)
                zq = z["qidx"]; zh = z["head"]; rq = z["risk_quantiles"]
                for k in range(len(zq)):
                    qmap[(int(zq[k]), int(zh[k]))] = rq[k]
        Q = 257
        self.risk_q = np.zeros((N, Q))
        for i, r in enumerate(hs):
            key = (int(r["qidx"]), int(r["head"]))
            if key in qmap:
                self.risk_q[i] = qmap[key]

    # -- vectorized log-theta interpolation of a per-head-step grid metric --
    def _interp(self, grid_vals, theta_vec):
        """theta_vec: (N,) per-head-step query theta. Returns (N,) interpolated."""
        q = np.log10(np.clip(theta_vec, 1e-300, None))
        lg = self.log_grid
        idx = np.searchsorted(lg, q)
        idx = np.clip(idx, 1, self.G - 1)
        lo = idx - 1
        hi = idx
        denom = lg[hi] - lg[lo]
        frac = np.where(denom > 0, (q - lg[lo]) / denom, 0.0)
        frac = np.clip(frac, 0.0, 1.0)  # clamp beyond grid ends
        ar = np.arange(self.N)
        return grid_vals[ar, lo] * (1 - frac) + grid_vals[ar, hi] * frac

    def count_at(self, theta_vec):
        return self._interp(self.grid_count, theta_vec)

    def relL2_at(self, theta_vec):
        return self._interp(self.grid_relL2, theta_vec)

    def bytes_at(self, theta_vec):
        return BYTES_PER_TOKEN * self.count_at(theta_vec)

    def fn_mass_at(self, theta_vec):
        return self._interp(self.grid_fn_mass, theta_vec)

    def fp_mass_at(self, theta_vec):
        return self._interp(self.grid_fp_mass, theta_vec)


# ---------------------------------------------------------------- splits
def make_split(lab, kind, block=6, train_frac=0.6):
    qs = np.unique(lab.qidx)  # already 0..287 sorted (== context order)
    qs = np.sort(qs)
    if kind == "blocked":
        blk = (np.arange(len(qs)) // block)
        train_q = set(qs[(blk % 2) == 0].tolist())
    elif kind == "even_odd":
        train_q = set(qs[(np.arange(len(qs)) % 2) == 0].tolist())
    elif kind == "contiguous":
        cut = int(round(len(qs) * train_frac))
        train_q = set(qs[:cut].tolist())
    else:
        raise ValueError(kind)
    is_train = np.array([q in train_q for q in lab.qidx])
    return is_train, ~is_train


def adjacency_leak(lab, is_train):
    """Fraction of adjacent qidx pairs that straddle the train/test boundary."""
    qs = np.sort(np.unique(lab.qidx))
    # map qidx -> train?
    q2t = {}
    for i in range(lab.N):
        q2t[int(lab.qidx[i])] = bool(is_train[i])
    straddle = sum(1 for j in range(len(qs) - 1) if q2t[int(qs[j])] != q2t[int(qs[j + 1])])
    return straddle / (len(qs) - 1)


# ---------------------------------------------------------------- eval helpers
def eval_theta_assignment(lab, mask, theta_vec):
    """Metrics for a per-head-step theta assignment over head-steps in `mask`."""
    relL2 = lab.relL2_at(theta_vec)
    byts = lab.bytes_at(theta_vec)
    m = mask
    ratio = relL2[m] / np.maximum(lab.c_relL2[m], 1e-30)
    meets = ratio <= MEET_TOL + 1e-9
    return {
        "n": int(m.sum()),
        "mean_bytes": float(byts[m].mean()),
        "mean_bytes_ratio_vs_canonical": float(byts[m].mean() / max(lab.c_bytes[m].mean(), 1e-9)),
        "frac_within_1.05x_canonical": float(meets.mean()),
        "relL2_ratio": _stats(ratio),
        "count": _stats(lab.count_at(theta_vec)[m]),
    }


def bucket_breakdown(lab, mask, theta_vec):
    relL2 = lab.relL2_at(theta_vec)
    byts = lab.bytes_at(theta_vec)
    out = {}
    for b in ["0-16k", "16-48k", "48-96k", "96-160k"]:
        m = mask & (lab.context_bucket == b)
        if m.sum() == 0:
            continue
        ratio = relL2[m] / np.maximum(lab.c_relL2[m], 1e-30)
        out[b] = {
            "n": int(m.sum()),
            "frac_within_1.05x": float((ratio <= MEET_TOL + 1e-9).mean()),
            "relL2_ratio_mean": float(ratio.mean()),
            "relL2_ratio_p95": float(np.percentile(ratio, 95)),
            "mean_bytes_ratio_vs_canonical": float(byts[m].mean() / max(lab.c_bytes[m].mean(), 1e-9)),
        }
    return out


def coverage_ceiling(lab, mask, theta_base):
    """Sweep a multiplicative theta scale over `theta_base` (per head-step) and
    report the reachable coverage ceiling and cheapest bytes to hit 0.95 / 0.98.
    Threshold policies flood as theta->0; int8-split over-inclusion caps coverage
    below 1.0 (the flooding ceiling)."""
    scales = np.power(10.0, np.arange(-4.0, 2.01, 0.1))
    m = mask
    target_bytes = lab.c_bytes[m].mean()
    best95 = None
    best98 = None
    max_frac = 0.0
    for s in scales:
        theta = theta_base * s
        rel = lab.relL2_at(theta)[m]
        frac = float((rel / np.maximum(lab.c_relL2[m], 1e-30) <= MEET_TOL + 1e-9).mean())
        br = float(lab.bytes_at(theta)[m].mean() / max(target_bytes, 1e-9))
        max_frac = max(max_frac, frac)
        if frac >= 0.95:
            best95 = br if (best95 is None or br < best95) else best95
        if frac >= 0.98:
            best98 = br if (best98 is None or br < best98) else best98
    return {"max_frac_within_1.05x": max_frac,
            "bytes_ratio_to_cover_95pct": best95,
            "bytes_ratio_to_cover_98pct": best98}


def calibrate_knob_to_bytes(bytes_fn, knobs, target_bytes):
    """Pick knob (from sorted `knobs`) whose train mean-bytes is closest to target."""
    best = None
    for k in knobs:
        mb = bytes_fn(k)
        d = abs(mb - target_bytes)
        if best is None or d < best[0]:
            best = (d, k, mb)
    return best[1], best[2]


# ---------------------------------------------------------------- #13 static tables
def group_keys(lab, granularity):
    out = np.empty(lab.N, dtype=object)
    if granularity == "global":
        for i in range(lab.N):
            out[i] = "g"
    elif granularity == "kv_head":
        for i in range(lab.N):
            out[i] = "kv%d" % int(lab.kv_head[i])
    elif granularity == "head":
        for i in range(lab.N):
            out[i] = "h%d" % int(lab.head[i])
    elif granularity == "kv_head_ctx":
        for i in range(lab.N):
            out[i] = "kv%d|%s" % (int(lab.kv_head[i]), lab.context_bucket[i])
    else:
        raise ValueError(granularity)
    return out


def fit_static_table(lab, is_train, granularity, pct):
    """theta[group] = pct-th percentile of canonical cutoff over train head-steps."""
    keys = group_keys(lab, granularity)
    table = {}
    global_val = np.percentile(lab.c_cutoff[is_train], pct)
    for k in set(keys[is_train]):
        m = is_train & (keys == k)
        if m.sum() >= 5:
            table[k] = float(np.percentile(lab.c_cutoff[m], pct))
        else:
            table[k] = float(global_val)  # fall back for sparse cells
    return table, float(global_val)


def apply_static_table(lab, granularity, table, global_val):
    keys = group_keys(lab, granularity)
    return np.array([table.get(k, global_val) for k in keys])


def run_issue13(lab, is_train, is_test, split_name):
    target_bytes_train = lab.c_bytes[is_train].mean()
    target_bytes_test = lab.c_bytes[is_test].mean()
    pct_grid = np.arange(1.0, 100.0, 1.0)
    results = {}
    for gran in ["global", "kv_head", "kv_head_ctx", "head"]:
        # build theta assignment as fn of percentile knob
        def bytes_at_pct(p, gran=gran):
            table, gv = fit_static_table(lab, is_train, gran, p)
            theta = apply_static_table(lab, gran, table, gv)
            return lab.bytes_at(theta)[is_train].mean()

        pstar, train_mb = calibrate_knob_to_bytes(bytes_at_pct, pct_grid, target_bytes_train)
        table, gv = fit_static_table(lab, is_train, gran, pstar)
        theta = apply_static_table(lab, gran, table, gv)

        test_metrics = eval_theta_assignment(lab, is_test, theta)
        train_metrics = eval_theta_assignment(lab, is_train, theta)
        buckets = bucket_breakdown(lab, is_test, theta)

        # coverage ceiling: scale the calibrated table down toward flooding and
        # report reachable coverage + cheapest bytes for 95%/98% on test.
        cov = coverage_ceiling(lab, is_test, theta)
        results[gran] = {
            "calibrated_percentile": float(pstar),
            "table_size_entries": len(table),
            "test_matched_bytes": test_metrics,
            "train_matched_bytes": train_metrics,
            "test_bucket_breakdown": buckets,
            "coverage_ceiling_test": cov,
            "table": {str(k): v for k, v in sorted(table.items(), key=lambda kv: str(kv[0]))},
        }
    return {
        "split": split_name,
        "target_bytes_train": float(target_bytes_train),
        "target_bytes_test": float(target_bytes_test),
        "canonical_test_relL2": _stats(lab.c_relL2[is_test]),
        "granularities": results,
    }


# ---------------------------------------------------------------- #14 predictor
def build_features(lab, feature_set):
    """Return (X, names). All scan-domain / deployable-before-V-gather."""
    cols = []
    names = []

    def add(name, v):
        cols.append(np.asarray(v, dtype=float))
        names.append(name)

    log = lambda x: np.log10(np.clip(x, 1e-30, None))
    # cheap accumulator features (no histogram/CDF barrier)
    add("log_total_risk_mass", log(lab.total_risk_mass))
    add("pq_lse", lab.pq_lse)
    add("pq_max_logit", lab.pq_max_logit)
    add("pq_entropy", lab.pq_entropy)
    add("log_proxy_mass_90", log(lab.proxy_mass_90 + 1.0))
    add("log_finite_risk_tokens", log(lab.finite_risk_tokens))
    add("log_context_len", log(lab.context_len))
    add("code_err_mean", lab.code_err_mean)
    add("code_err_p95", lab.code_err_p95)
    add("code_err_max", lab.code_err_max)
    # cheap "count >= probe" counters (monotone accumulators over the scan)
    probes = [1e-11, 1e-10, 1e-9, 1e-8]
    for pr in probes:
        # count at fixed theta from the grid == deployable counter
        c = lab.count_at(np.full(lab.N, pr))
        add("log_count_ge_%.0e" % pr, log(c + 1.0))
    if feature_set == "histogram":
        # genuine risk-distribution order statistics (npz quantiles, descending;
        # col0 = max risk). Computing these needs a full histogram/CDF barrier at
        # scan EOS -- exactly the extra barrier #14 asks whether we can avoid.
        for qi in [0, 4, 16, 32, 64, 96, 128, 160, 192, 224]:
            add("log_risk_q%d" % qi, log(lab.risk_q[:, qi]))
    # kv-head one-hot (identity is deployable)
    for kv in range(8):
        add("kv_%d" % kv, (lab.kv_head == kv).astype(float))
    X = np.stack(cols, axis=1)
    return X, names


def fit_linear(X_tr, y_tr):
    A = np.hstack([X_tr, np.ones((X_tr.shape[0], 1))])
    coef, *_ = np.linalg.lstsq(A, y_tr, rcond=None)
    return coef


def predict_linear(coef, X):
    A = np.hstack([X, np.ones((X.shape[0], 1))])
    return A @ coef


def run_issue14(lab, is_train, is_test, split_name):
    y = np.log10(np.clip(lab.c_cutoff, 1e-30, None))  # target: log10 canonical cutoff
    target_bytes_train = lab.c_bytes[is_train].mean()
    target_bytes_test = lab.c_bytes[is_test].mean()

    out = {"split": split_name, "target_bytes_test": float(target_bytes_test), "models": {}}

    for fs in ["cheap", "histogram"]:
        X, names = build_features(lab, fs)
        # standardize using train stats for conditioning
        mu = X[is_train].mean(0)
        sd = X[is_train].std(0) + 1e-9
        Xs = (X - mu) / sd
        coef = fit_linear(Xs[is_train], y[is_train])
        yhat = predict_linear(coef, Xs)
        err = yhat - y  # in decades
        # deployable theta = 10^yhat, then a single global log-shift delta calibrated
        # on train to hit matched bytes (does NOT use per-step test info).
        shift_grid = np.arange(-3.0, 3.01, 0.05)

        def bytes_at_shift(delta):
            theta = np.power(10.0, yhat + delta)
            return lab.bytes_at(theta)[is_train].mean()

        dstar, _ = calibrate_knob_to_bytes(bytes_at_shift, shift_grid, target_bytes_train)
        theta = np.power(10.0, yhat + dstar)
        test_metrics = eval_theta_assignment(lab, is_test, theta)
        buckets = bucket_breakdown(lab, is_test, theta)

        # recall/precision vs canonical (nested sets on same risk array)
        sel = lab.count_at(theta)
        can = lab.c_count
        recall = np.where(theta <= lab.c_cutoff, 1.0, np.minimum(sel / np.maximum(can, 1e-9), 1.0))
        precision = np.where(theta >= lab.c_cutoff, 1.0, np.minimum(can / np.maximum(sel, 1e-9), 1.0))

        # coverage ceiling by scaling predicted theta toward flooding
        cov = coverage_ceiling(lab, is_test, np.power(10.0, yhat))

        out["models"][fs] = {
            "n_features": len(names),
            "feature_names": names,
            "pred_error_decades_test": _stats(np.abs(err[is_test])),
            "pred_error_decades_train": _stats(np.abs(err[is_train])),
            "pred_R2_test": float(1 - np.var(err[is_test]) / max(np.var(y[is_test]), 1e-30)),
            "calibrated_log_shift": float(dstar),
            "test_matched_bytes": test_metrics,
            "test_bucket_breakdown": buckets,
            "recall_vs_canonical_test": _stats(recall[is_test]),
            "precision_vs_canonical_test": _stats(precision[is_test]),
            "coverage_ceiling_test": cov,
        }

    # baseline: predict a constant (== global static #13 at same calibration) for ref
    return out


# ---------------------------------------------------------------- #16 band
def run_issue16(lab, is_train, is_test, split_name, center_theta):
    """Two-tier band centered (multiplicatively) on `center_theta` per head-step.
    band = [center/w, center*w]; sweep half-width w (decades) and cap M.
    Overflow rule (primary): if band_count > M, include the whole band (exact V) ->
    quality >= canonical whenever cutoff in band, extra bytes bounded by count(theta_lo).
    """
    target_bytes_test = lab.c_bytes[is_test].mean()
    m = is_test
    res = {"split": split_name, "target_bytes_test": float(target_bytes_test), "sweeps": []}

    half_widths = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]  # decades
    M_grid = [16, 32, 64, 128, 256, 512, 1024, 2048]

    for hw in half_widths:
        w = 10.0 ** hw
        theta_lo = center_theta / w
        theta_hi = center_theta * w
        cnt_hi = lab.count_at(theta_hi)      # forced-exact population
        cnt_lo = lab.count_at(theta_lo)      # forced + band population
        band = np.maximum(cnt_lo - cnt_hi, 0.0)
        total = lab.count_at(np.full(lab.N, lab.theta_grid[0]))  # ~ all finite tokens
        cutoff_in_band = (lab.c_cutoff >= theta_lo) & (lab.c_cutoff <= theta_hi)

        # population fractions (mean over test head-steps)
        frac_hi = float((cnt_hi[m] / np.maximum(total[m], 1e-9)).mean())
        frac_band = float((band[m] / np.maximum(total[m], 1e-9)).mean())
        frac_lo = 1.0 - frac_hi - frac_band

        # fn / fp risk mass OUTSIDE the band (interp from grid)
        fn_mass = lab.fn_mass_at(theta_lo)   # canonical-selected tokens dropped below theta_lo
        fp_mass = lab.fp_mass_at(theta_hi)   # forced-selected tokens canonical drops above theta_hi

        row = {
            "half_width_decades": hw,
            "theta_lo_median": float(np.median(theta_lo[m])),
            "theta_hi_median": float(np.median(theta_hi[m])),
            "frac_forced_exact_pop": frac_hi,
            "frac_ambiguous_band_pop": frac_band,
            "frac_forced_approx_pop": frac_lo,
            "band_count_per_headstep": _stats(band[m]),
            "M_for_99pct_no_overflow": float(np.percentile(band[m], 99)),
            "M_for_95pct_no_overflow": float(np.percentile(band[m], 95)),
            "frac_cutoff_in_band": float(cutoff_in_band[m].mean()),
            "fn_risk_mass_below_lo": _stats(fn_mass[m]),
            "fp_risk_mass_above_hi": _stats(fp_mass[m]),
            "by_M": [],
        }
        for M in M_grid:
            # overflow -> include whole band (bytes = count(theta_lo)); else exact-resolve
            # band == reproduce canonical when cutoff in band.
            overflow = band > M
            # effective theta per head-step:
            #  - cutoff in band & no overflow -> exact canonical (relL2 = canonical, bytes = canonical)
            #  - cutoff in band & overflow   -> include whole band -> theta_lo
            #  - cutoff >= theta_hi          -> select risk>=theta_hi (theta_hi), over-provision
            #  - cutoff <  theta_lo          -> select risk>=theta_lo (theta_lo), under-provision (fn)
            eff_relL2 = np.empty(lab.N)
            eff_bytes = np.empty(lab.N)
            relL2_lo = lab.relL2_at(theta_lo)
            relL2_hi = lab.relL2_at(theta_hi)
            bytes_lo = BYTES_PER_TOKEN * cnt_lo
            bytes_hi = BYTES_PER_TOKEN * cnt_hi
            above = lab.c_cutoff >= theta_hi
            below = lab.c_cutoff < theta_lo
            inband = cutoff_in_band
            # default
            eff_relL2[:] = lab.c_relL2
            eff_bytes[:] = lab.c_bytes
            # over-provision (cutoff above hi): select down to theta_hi
            eff_relL2[above] = relL2_hi[above]
            eff_bytes[above] = bytes_hi[above]
            # under-provision (cutoff below lo): select only down to theta_lo
            eff_relL2[below] = relL2_lo[below]
            eff_bytes[below] = bytes_lo[below]
            # in band + overflow: include whole band (theta_lo) -> preserves quality,
            # bytes grow. Drop variant below trades the opposite way.
            ov = inband & overflow
            eff_relL2[ov] = relL2_lo[ov]
            eff_bytes[ov] = bytes_lo[ov]
            ratio = eff_relL2[m] / np.maximum(lab.c_relL2[m], 1e-30)
            frac_meet = float((ratio <= MEET_TOL + 1e-9).mean())
            bytes_ratio = float(eff_bytes[m].mean() / max(target_bytes_test, 1e-9))

            # drop-overflow variant: overflow band tokens dropped (keep risk>=theta_hi)
            d_relL2 = eff_relL2.copy(); d_bytes = eff_bytes.copy()
            d_relL2[ov] = relL2_hi[ov]; d_bytes[ov] = bytes_hi[ov]
            dratio = d_relL2[m] / np.maximum(lab.c_relL2[m], 1e-30)
            d_meet = float((dratio <= MEET_TOL + 1e-9).mean())
            d_bratio = float(d_bytes[m].mean() / max(target_bytes_test, 1e-9))

            row["by_M"].append({
                "M": M,
                "frac_no_overflow": float((~overflow[m]).mean()),
                # overflow -> include whole band
                "frac_within_1.05x_canonical": frac_meet,
                "mean_bytes_ratio_vs_canonical": bytes_ratio,
                "relL2_ratio_p95": float(np.percentile(ratio, 95)),
                "meets_99_and_1.2x": bool(frac_meet >= 0.99 and bytes_ratio <= 1.2),
                # overflow -> drop band
                "drop_frac_within_1.05x_canonical": d_meet,
                "drop_mean_bytes_ratio_vs_canonical": d_bratio,
                "drop_meets_99_and_1.2x": bool(d_meet >= 0.99 and d_bratio <= 1.2),
            })
        res["sweeps"].append(row)
    return res


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lab_dirs", nargs="+", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--split", default="blocked", choices=["blocked", "even_odd", "contiguous"])
    args = ap.parse_args()

    lab = Lab(args.lab_dirs)

    # interpolation fidelity self-check: leave-one-grid-point-out on interior thetas
    loo = interp_fidelity(lab)

    report = {
        "meta": {
            "n_headsteps": lab.N,
            "n_qidx": int(len(np.unique(lab.qidx))),
            "n_heads": int(len(np.unique(lab.head))),
            "n_kv_heads": int(len(np.unique(lab.kv_head))),
            "layer": lab.layer,
            "context_min": float(lab.context_len.min()),
            "context_max": float(lab.context_len.max()),
            "theta_grid": lab.theta_grid.tolist(),
            "bytes_per_token": BYTES_PER_TOKEN,
            "canonical_global_topB": {
                "relL2": _stats(lab.c_relL2),
                "selected_count": _stats(lab.c_count),
                "logical_v_bytes": _stats(lab.c_bytes),
                "cutoff_risk_log10_decades_span": float(
                    np.log10(lab.c_cutoff.max()) - np.log10(lab.c_cutoff[lab.c_cutoff > 0].min())
                ),
            },
            "interp_fidelity_leave_one_out": loo,
            "variance_decomposition": variance_decomposition(lab),
        },
        "splits_reported": {},
    }

    # run all three splits so conclusions are shown split-stable; primary = args.split
    for split_name in ["blocked", "even_odd", "contiguous"]:
        is_train, is_test = make_split(lab, split_name)
        leak = adjacency_leak(lab, is_train)
        split_block = {
            "n_train": int(is_train.sum()),
            "n_test": int(is_test.sum()),
            "adjacency_leak_frac": leak,
            "train_ctx_bucket_counts": {b: int(((lab.context_bucket == b) & is_train).sum())
                                        for b in ["0-16k", "16-48k", "48-96k", "96-160k"]},
            "test_ctx_bucket_counts": {b: int(((lab.context_bucket == b) & is_test).sum())
                                       for b in ["0-16k", "16-48k", "48-96k", "96-160k"]},
            "issue13_static_tables": run_issue13(lab, is_train, is_test, split_name),
            "issue14_predictor": run_issue14(lab, is_train, is_test, split_name),
        }
        # #16 uses the #14 cheap-model prediction as band center
        X, _ = build_features(lab, "cheap")
        y = np.log10(np.clip(lab.c_cutoff, 1e-30, None))
        mu = X[is_train].mean(0); sd = X[is_train].std(0) + 1e-9
        Xs = (X - mu) / sd
        coef = fit_linear(Xs[is_train], y[is_train])
        yhat = predict_linear(coef, Xs)
        center_pred = np.power(10.0, yhat)
        split_block["issue16_band_predictor_center"] = run_issue16(
            lab, is_train, is_test, split_name, center_pred)
        # #16 fixed global band center (single constant theta) for contrast
        theta_const = np.full(lab.N, float(np.median(lab.c_cutoff[is_train])))
        split_block["issue16_band_fixed_center"] = run_issue16(
            lab, is_train, is_test, split_name + "_fixedcenter", theta_const)
        report["splits_reported"][split_name] = split_block

    report["primary_split"] = args.split
    with open(args.out_json, "w") as f:
        json.dump(report, f, indent=2, default=lambda x: None if (isinstance(x, float) and x != x) else x)
    print("wrote", args.out_json)
    _digest(report, args.split)


def variance_decomposition(lab):
    """How much of the log10-canonical-cutoff non-stationarity does each grouping
    explain (R^2 of group-mean predictor), and the robust spread of the target."""
    y = np.log10(np.clip(lab.c_cutoff, 1e-30, None))
    tot = np.var(y)

    def r2_group(g):
        pred = np.zeros_like(y)
        for k in set(g):
            m = g == k
            pred[m] = y[m].mean()
        return float(1 - np.var(y - pred) / max(tot, 1e-30))

    kv = np.array(["kv%d" % int(h) for h in lab.kv_head], dtype=object)
    bkt = lab.context_bucket
    cell = np.array(["kv%d|%s" % (int(lab.kv_head[i]), lab.context_bucket[i]) for i in range(lab.N)], dtype=object)
    hd = np.array(["h%d" % int(h) for h in lab.head], dtype=object)
    ys = np.sort(y)
    return {
        "log10_cutoff_var": float(tot),
        "log10_cutoff_std_decades": float(np.sqrt(tot)),
        "log10_cutoff_span_full_decades": float(y.max() - y.min()),
        "log10_cutoff_p10_p90_decades": float(np.percentile(y, 90) - np.percentile(y, 10)),
        "log10_cutoff_p01_p99_decades": float(np.percentile(y, 99) - np.percentile(y, 1)),
        "R2_by_kv_head": r2_group(kv),
        "R2_by_context_bucket": r2_group(bkt),
        "R2_by_kv_head_x_context_bucket": r2_group(cell),
        "R2_by_query_head": r2_group(hd),
        "note": "R2 = fraction of log10-cutoff variance explained by group means "
                "(in-sample, upper bound on a static table). #14 scan predictor R2 is "
                "reported per split under issue14_predictor.",
    }


def interp_fidelity(lab):
    """Leave-one-out log-theta interpolation error at interior grid thetas."""
    G = lab.G
    rel_err = []
    cnt_err = []
    ar = np.arange(lab.N)
    for j in range(1, G - 1):
        lg = lab.log_grid
        frac = (lg[j] - lg[j - 1]) / (lg[j + 1] - lg[j - 1])
        pred_rel = lab.grid_relL2[:, j - 1] * (1 - frac) + lab.grid_relL2[:, j + 1] * frac
        pred_cnt = lab.grid_count[:, j - 1] * (1 - frac) + lab.grid_count[:, j + 1] * frac
        true_rel = lab.grid_relL2[:, j]
        true_cnt = lab.grid_count[:, j]
        rel_err.extend(np.abs(pred_rel - true_rel) / np.maximum(true_rel, 1e-12))
        cnt_err.extend(np.abs(pred_cnt - true_cnt) / np.maximum(true_cnt, 1.0))
    return {
        "relL2_rel_err": _stats(rel_err),
        "count_rel_err": _stats(cnt_err),
        "note": "leave-one-out over interior grid thetas; operating region near cutoffs is monotone",
    }


def _digest(report, split):
    sb = report["splits_reported"][split]
    print("=== primary split:", split, "===")
    print("adjacency_leak_frac=%.3f n_train=%d n_test=%d" % (
        sb["adjacency_leak_frac"], sb["n_train"], sb["n_test"]))
    vd = report["meta"]["variance_decomposition"]
    print("var-decomp log-cutoff: std=%.2f dec p10-p90=%.2f dec | R2 kv=%.3f ctx=%.3f kvxctx=%.3f head=%.3f" % (
        vd["log10_cutoff_std_decades"], vd["log10_cutoff_p10_p90_decades"],
        vd["R2_by_kv_head"], vd["R2_by_context_bucket"], vd["R2_by_kv_head_x_context_bucket"], vd["R2_by_query_head"]))
    i13 = sb["issue13_static_tables"]["granularities"]
    print("\n#13 matched-bytes frac within 1.05x canonical (test):")
    for g in ["global", "kv_head", "kv_head_ctx", "head"]:
        r = i13[g]["test_matched_bytes"]
        cov = i13[g]["coverage_ceiling_test"]
        print("  %-12s frac=%.3f bytes_ratio=%.3f p95=%.2f | ceiling=%.3f 98%%@%.2fx" % (
            g, r["frac_within_1.05x_canonical"], r["mean_bytes_ratio_vs_canonical"],
            r["relL2_ratio"]["p95"], cov["max_frac_within_1.05x"], cov["bytes_ratio_to_cover_98pct"] or -1))
    i14 = sb["issue14_predictor"]["models"]
    print("\n#14 predictor (test):")
    for fs in ["cheap", "histogram"]:
        r = i14[fs]
        cov = r["coverage_ceiling_test"]
        print("  %-10s err mean=%.3f p95=%.3f R2=%.3f frac=%.3f bytes=%.3f recall=%.3f prec=%.3f ceil=%.3f 98%%@%.2fx" % (
            fs, r["pred_error_decades_test"]["mean"], r["pred_error_decades_test"]["p95"],
            r["pred_R2_test"], r["test_matched_bytes"]["frac_within_1.05x_canonical"],
            r["test_matched_bytes"]["mean_bytes_ratio_vs_canonical"],
            r["recall_vs_canonical_test"]["mean"], r["precision_vs_canonical_test"]["mean"],
            cov["max_frac_within_1.05x"], cov["bytes_ratio_to_cover_98pct"] or -1))
    print("\n#16 predictor-centered band (test): best (hw,M) meeting 99%/1.2x:")
    found = False
    for row in sb["issue16_band_predictor_center"]["sweeps"]:
        for mm in row["by_M"]:
            if mm["meets_99_and_1.2x"]:
                print("  hw=%.2f dec M=%d frac_within=%.3f bytes_ratio=%.3f band_p95=%.0f cutoff_in_band=%.3f" % (
                    row["half_width_decades"], mm["M"], mm["frac_within_1.05x_canonical"],
                    mm["mean_bytes_ratio_vs_canonical"], row["band_count_per_headstep"]["p95"],
                    row["frac_cutoff_in_band"]))
                found = True
                break
        if found:
            break
    if not found:
        print("  NONE meet 99% within 1.05x at <=1.2x bytes")


if __name__ == "__main__":
    main()
