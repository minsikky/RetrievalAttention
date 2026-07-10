#!/usr/bin/env python3
"""Analyze the absolute-V-threshold lab outputs (issues #12-#18).

Consumes v_threshold_theta.csv + v_threshold_headstep.csv from one or more lab
directories and emits the issue #12 report tables:
  - per-theta exact-V count distribution (mean/p95/p99/max) and logical/physical
    V bytes;
  - per-theta relL2 mean/p95/p99/max (both int8-split and full-fp16 selection),
    plus a task-quality proxy (cosine-equivalent via relL2);
  - breakdown by context-length bucket, KV head, and decode position;
  - canonical global-top-B baseline at MATCHED BYTES and MATCHED QUALITY;
  - threshold sensitivity (quality/bytes per octave of theta);
  - false-negative risk-mass summaries.

Pure offline reduction: no model, no torch.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from collections import defaultdict


def _pctl(xs, q):
    if not xs:
        return float("nan")
    ys = sorted(xs)
    if len(ys) == 1:
        return float(ys[0])
    pos = (len(ys) - 1) * (q / 100.0)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(ys[lo])
    return float(ys[lo] + (ys[hi] - ys[lo]) * (pos - lo))


def _stats(xs):
    if not xs:
        return {"n": 0, "mean": float("nan"), "p50": float("nan"), "p95": float("nan"),
                "p99": float("nan"), "max": float("nan")}
    return {
        "n": len(xs),
        "mean": float(sum(xs) / len(xs)),
        "p50": _pctl(xs, 50),
        "p95": _pctl(xs, 95),
        "p99": _pctl(xs, 99),
        "max": float(max(xs)),
    }


def load_rows(paths, fname):
    rows = []
    for p in paths:
        fp = os.path.join(p, fname)
        if os.path.exists(fp):
            with open(fp, newline="") as f:
                rows.extend(list(csv.DictReader(f)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lab_dirs", nargs="+", required=True)
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    theta_rows = load_rows(args.lab_dirs, "v_threshold_theta.csv")
    hs_rows = load_rows(args.lab_dirs, "v_threshold_headstep.csv")
    if not theta_rows or not hs_rows:
        raise SystemExit("no lab rows found in %s" % args.lab_dirs)

    # ---- canonical (global top-B) per-head-step baseline ----
    canon_relL2 = [float(r["canonical_relL2"]) for r in hs_rows]
    canon_bytes = [float(r["canonical_logical_v_bytes"]) for r in hs_rows]
    canon_count = [float(r["canonical_selected_count"]) for r in hs_rows]
    canon_eff = [float(r["canonical_effective_reads"]) for r in hs_rows]
    n_headsteps = len(hs_rows)
    contexts = sorted({int(r["context_len"]) for r in hs_rows})

    report = {
        "n_headsteps": n_headsteps,
        "n_heads": len({int(r["head"]) for r in hs_rows}),
        "n_positions": len({int(r["qidx"]) for r in hs_rows}),
        "context_min": min(contexts),
        "context_max": max(contexts),
        "layer": int(hs_rows[0]["layer"]),
        "canonical_global_topB": {
            "relL2": _stats(canon_relL2),
            "logical_v_bytes": _stats(canon_bytes),
            "selected_count": _stats(canon_count),
            "effective_reads": _stats(canon_eff),
        },
    }

    # ---- per-theta aggregate over all head-steps ----
    by_theta = defaultdict(list)
    for r in theta_rows:
        by_theta[float(r["theta"])].append(r)

    per_theta = []
    for theta in sorted(by_theta):
        rs = by_theta[theta]
        counts = [float(r["exact_selected_count"]) for r in rs]
        eff = [float(r["exact_effective_reads"]) for r in rs]
        lbytes = [float(r["logical_v_bytes"]) for r in rs]
        vmb = [float(r["v_path_MB"]) for r in rs]
        rell2 = [float(r["relL2"]) for r in rs]
        rell2_full = [float(r["relL2_full_noSplit"]) for r in rs]
        fn_mass = [float(r["false_negative_risk_mass"]) for r in rs]
        fp_mass = [float(r["false_positive_risk_mass"]) for r in rs]
        fn_cnt = [float(r["false_negative_count"]) for r in rs]
        # task-quality proxy: fraction of head-steps meeting canonical relL2
        # within a small tolerance; and mean relL2 ratio vs canonical.
        rel_ratio = []
        meets = 0
        for r in rs:
            cr = float(r["canonical_relL2"])
            tr = float(r["relL2"])
            rel_ratio.append(tr / max(cr, 1e-20))
            if tr <= cr * 1.05 + 1e-9:
                meets += 1
        per_theta.append({
            "theta": theta,
            "count": _stats(counts),
            "effective_reads": _stats(eff),
            "logical_v_bytes": _stats(lbytes),
            "v_path_MB": _stats(vmb),
            "relL2": _stats(rell2),
            "relL2_full_noSplit": _stats(rell2_full),
            "relL2_ratio_vs_canonical": _stats(rel_ratio),
            "frac_headsteps_within_1.05x_canonical": meets / max(len(rs), 1),
            "false_negative_risk_mass": _stats(fn_mass),
            "false_positive_risk_mass": _stats(fp_mass),
            "false_negative_count": _stats(fn_cnt),
        })
    report["per_theta"] = per_theta

    # ---- matched-bytes and matched-quality comparison ----
    # Mean canonical bytes and relL2 across head-steps.
    mean_canon_bytes = sum(canon_bytes) / len(canon_bytes)
    mean_canon_rell2 = sum(canon_relL2) / len(canon_relL2)
    # find theta whose mean logical bytes is closest to canonical mean bytes,
    # and theta whose mean relL2 is closest to canonical mean relL2.
    def closest(metric_key, target, sub="mean"):
        best = None
        for pt in per_theta:
            val = pt[metric_key][sub]
            if val != val:  # nan
                continue
            d = abs(val - target)
            if best is None or d < best[0]:
                best = (d, pt)
        return best[1] if best else None

    mb = closest("logical_v_bytes", mean_canon_bytes)
    mq = closest("relL2", mean_canon_rell2)
    report["matched_bytes"] = {
        "canonical_mean_logical_v_bytes": mean_canon_bytes,
        "canonical_mean_relL2": mean_canon_rell2,
        "theta_at_matched_bytes": mb["theta"] if mb else None,
        "relL2_at_matched_bytes": mb["relL2"] if mb else None,
        "count_at_matched_bytes": mb["count"] if mb else None,
    }
    report["matched_quality"] = {
        "theta_at_matched_quality": mq["theta"] if mq else None,
        "logical_v_bytes_at_matched_quality": mq["logical_v_bytes"] if mq else None,
        "count_at_matched_quality": mq["count"] if mq else None,
        "relL2_at_matched_quality": mq["relL2"] if mq else None,
    }

    # ---- threshold sensitivity: d(relL2) and d(bytes) per octave of theta ----
    sens = []
    for i in range(1, len(per_theta)):
        a, b = per_theta[i - 1], per_theta[i]
        if a["theta"] <= 0 or b["theta"] <= 0:
            continue
        octaves = math.log2(b["theta"] / a["theta"])
        if octaves == 0:
            continue
        d_rel = (b["relL2"]["mean"] - a["relL2"]["mean"]) / octaves
        d_bytes = (b["logical_v_bytes"]["mean"] - a["logical_v_bytes"]["mean"]) / octaves
        d_count = (b["count"]["mean"] - a["count"]["mean"]) / octaves
        sens.append({
            "theta_lo": a["theta"], "theta_hi": b["theta"],
            "d_relL2_per_octave": d_rel,
            "d_logical_bytes_per_octave": d_bytes,
            "d_count_per_octave": d_count,
        })
    report["threshold_sensitivity"] = sens

    # ---- count-variance infeasibility signal: for the matched-bytes theta,
    # how much does the exact-V COUNT vary head-to-head (the core #12 risk)? ----
    if mb is not None:
        rs = by_theta[mb["theta"]]
        counts = [float(r["exact_selected_count"]) for r in rs]
        cmean = sum(counts) / len(counts)
        report["count_variance_at_matched_bytes"] = {
            "theta": mb["theta"],
            "count_mean": cmean,
            "count_p95": _pctl(counts, 95),
            "count_p99": _pctl(counts, 99),
            "count_max": max(counts),
            "count_min": min(counts),
            "count_cv": (
                (sum((c - cmean) ** 2 for c in counts) / len(counts)) ** 0.5 / max(cmean, 1e-9)
            ),
            "p99_over_mean": _pctl(counts, 99) / max(cmean, 1e-9),
            "max_over_mean": max(counts) / max(cmean, 1e-9),
        }

    # ---- breakdown by context bucket / kv head / decode position ----
    def bucket_breakdown(theta, keyfn, keyname):
        rs = by_theta.get(theta, [])
        groups = defaultdict(list)
        for r in rs:
            groups[keyfn(r)].append(r)
        out = []
        for k in sorted(groups):
            g = groups[k]
            out.append({
                keyname: k,
                "count": _stats([float(r["exact_selected_count"]) for r in g]),
                "relL2": _stats([float(r["relL2"]) for r in g]),
                "canonical_relL2": _stats([float(r["canonical_relL2"]) for r in g]),
            })
        return out

    if mb is not None:
        tb = mb["theta"]
        report["breakdown_at_matched_bytes_theta"] = {
            "theta": tb,
            "by_context_bucket": bucket_breakdown(tb, lambda r: r["context_bucket"], "context_bucket"),
            "by_kv_head": bucket_breakdown(tb, lambda r: int(r["kv_head"]), "kv_head"),
            "by_context_len": bucket_breakdown(tb, lambda r: int(r["context_len"]), "context_len"),
        }

    with open(args.out_json, "w") as f:
        json.dump(report, f, indent=2, sort_keys=False, default=lambda x: None if (isinstance(x, float) and x != x) else x)
    print("wrote", args.out_json)
    # brief console digest
    print("head-steps=%d heads=%d positions=%d ctx=[%d,%d]" % (
        report["n_headsteps"], report["n_heads"], report["n_positions"],
        report["context_min"], report["context_max"]))
    print("canonical relL2 mean=%.4g p99=%.4g; mean logical V bytes=%.4g" % (
        report["canonical_global_topB"]["relL2"]["mean"],
        report["canonical_global_topB"]["relL2"]["p99"],
        mean_canon_bytes))
    if "count_variance_at_matched_bytes" in report:
        cv = report["count_variance_at_matched_bytes"]
        print("matched-bytes theta=%.3g: count mean=%.0f p99=%.0f max=%.0f CV=%.3f p99/mean=%.2f" % (
            cv["theta"], cv["count_mean"], cv["count_p99"], cv["count_max"], cv["count_cv"], cv["p99_over_mean"]))


if __name__ == "__main__":
    main()
