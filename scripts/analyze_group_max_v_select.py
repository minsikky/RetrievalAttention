#!/usr/bin/env python3
"""Issue #20 GROUP-MAX V-selection offline evaluation (measurements 1-4 + OR-theta).

Pure-numpy reduction of the ``--gqa_group_max_eval`` producer shards
(``gqa_group_max_curve.csv`` + ``gqa_group_max_ortheta.csv``, one dir per qidx)
plus the #12 V-threshold-lab theta CSVs for the sanity gate. No torch, no GPU.

Proposal under test (issue #20, RTL group-max ranking ask):
  rank tail V tokens by GROUP-MAX RISK = max over the 4 GQA group heads of the
  per-head frozen scan-domain risk (p_pq^2 * V_error), take ONE rank cutoff at a
  single group budget B_grp, commit the top-B_grp set apply-to-all (any committed
  token applies to all 4 heads). One histogram, one walk, one cutoff per group.

Measurements (all at the ratified #20 envelope protocol):
  1. Set coverage at matched bytes (B_grp = |frozen union|): recall of the frozen
     per-head top-B_h V sets by top-B_grp(group-max), per head + aggregate.
  2. Quality under apply-to-all at matched bytes (BINDING GATE): per-head relL2 vs
     the frozen per-head baseline; aggregate-monotone + per-head envelope
     (ctx<32k: <= +6e-4 abs & +15% rel; tie floor 1e-7 absolute relL2).
  3. Budget rule: B_grp from the four B_h -- (a) sum B_h, (b) |union| oracle,
     (c) a fitted single-rung rule (train/test split, residuals vs |union|).
  4. Miss structure on failure: missed frozen-committed tokens' attention-weight
     mass share (load-bearing vs tail); group-max as prefetch + commit backstop.
  OR-theta curiosity: one FIXED global theta OR'd across the group (#12 family).
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from collections import defaultdict

import numpy as np

TIE_FLOOR = 1e-7          # ratified #20 absolute relL2 tie floor
ENV_ABS = 6e-4            # per-head regression bound (abs), ctx<32k
ENV_REL = 0.15            # per-head regression bound (rel), ctx<32k
CTX_ENV = 32000          # envelope only permits regressions below this ctx


# ------------------------------------------------------------------ loaders
def load_rows(root, name):
    rows = []
    for fp in sorted(glob.glob(os.path.join(root, "run_q*", name))):
        with open(fp, newline="") as f:
            rows.extend(list(csv.DictReader(f)))
    return rows


def as_num(rows):
    ints = {"qidx", "position", "context_len", "kv_head", "head", "B_grp", "bytes_grp",
            "Bh", "B_union", "B_sum", "B_max", "missed_tokens", "B_theta", "bytes_theta"}
    for r in rows:
        for k, v in list(r.items()):
            if k in ints:
                r[k] = int(v)
            elif k in ("budget_label", "group_heads"):
                pass
            else:
                try:
                    r[k] = float(v)
                except (TypeError, ValueError):
                    pass
    return rows


def pctl(xs, q):
    xs = np.asarray(xs, dtype=float)
    return float(np.percentile(xs, q)) if xs.size else float("nan")


# ------------------------------------------------------------------ sanity gate
def sanity_12(lab_dirs, theta=2e-11):
    """Reproduce the #12 headline: global theta 2e-11 -> frac head-steps within
    1.05x canonical relL2 at ~1.05x aggregate bytes (committed value ~0.685)."""
    rel, crel, byt, cbyt = [], [], [], []
    for d in lab_dirs:
        fp = os.path.join(d, "v_threshold_theta.csv")
        if not os.path.exists(fp):
            continue
        with open(fp, newline="") as f:
            for r in csv.DictReader(f):
                if abs(float(r["theta"]) - theta) < theta * 1e-6:
                    rel.append(float(r["relL2"]))
                    crel.append(float(r["canonical_relL2"]))
                    byt.append(float(r["logical_v_bytes"]))
                    cbyt.append(256.0 * float(r["canonical_selected_count"]))
    rel = np.array(rel); crel = np.array(crel); byt = np.array(byt); cbyt = np.array(cbyt)
    if rel.size == 0:
        return None
    return {
        "n_headsteps": int(rel.size),
        "theta": theta,
        "frac_within_1.05x_canonical": float(np.mean(rel <= 1.05 * crel)),
        "aggregate_bytes_ratio": float(byt.sum() / cbyt.sum()),
    }


def machinery_gate(curve):
    """The group-max harness must reproduce the frozen baseline bit-for-bit and
    recover the frozen committed-V set as top-B_h of its own recomputed risk."""
    bl = np.array([r["baseline_relL2"] for r in curve])
    rp = np.array([r["repro_baseline_relL2"] for r in curve])
    rc = np.array([r["repro_topBh_recall"] for r in curve])
    return {
        "rows": int(len(curve)),
        "max_abs_baseline_vs_repro": float(np.abs(bl - rp).max()),
        "min_topBh_recall": float(rc.min()),
        "mean_topBh_recall": float(rc.mean()),
    }


# ------------------------------------------------------------------ helpers
def by_budget(curve, label):
    return [r for r in curve if r["budget_label"] == label]


def head_curve(curve):
    """Map (position, head) -> sorted arrays (B_grp, relL2_frozenK) over the grid."""
    d = defaultdict(list)
    for r in curve:
        d[(r["position"], r["head"])].append((r["B_grp"], r["groupmax_relL2_frozenK"]))
    out = {}
    for k, v in d.items():
        v = sorted(set(v))
        out[k] = (np.array([x[0] for x in v], float), np.array([x[1] for x in v], float))
    return out


def interp_relL2(hc, pos, head, b):
    xs, ys = hc[(pos, head)]
    b = float(min(max(b, xs.min()), xs.max()))
    return float(np.interp(b, xs, ys))


# ------------------------------------------------------------------ measurements
def meas1_coverage(curve):
    u = by_budget(curve, "union")
    tab = {}
    for pos in sorted({r["position"] for r in u}):
        rows = [r for r in u if r["position"] == pos]
        cov = np.array([r["coverage_recall"] for r in rows])
        # token-weighted micro recall (sum hits / sum Bh)
        hits = sum((r["Bh"] - r["missed_tokens"]) for r in rows)
        tot = sum(r["Bh"] for r in rows)
        tab[pos] = {
            "ctx": rows[0]["context_len"],
            "n_heads": len(rows),
            "macro_recall_mean": float(cov.mean()),
            "macro_recall_min": float(cov.min()),
            "macro_recall_p05": pctl(cov, 5),
            "micro_recall": float(hits / max(tot, 1)),
            "B_union_mean": float(np.mean([r["B_union"] for r in rows])),
        }
    return tab


def classify_reg(base, gm, ctx):
    d = gm - base
    if d <= TIE_FLOOR:
        return "improve_or_tie"
    rel = d / max(base, 1e-30)
    if ctx < CTX_ENV and d <= ENV_ABS and rel <= ENV_REL:
        return "permitted"
    return "violation"


def meas2_envelope(curve, arm="groupmax_relL2_frozenK"):
    u = by_budget(curve, "union")
    per_pos = {}
    regressions = []
    for pos in sorted({r["position"] for r in u}):
        rows = [r for r in u if r["position"] == pos]
        base = np.array([r["baseline_relL2"] for r in rows])
        gm = np.array([r[arm] for r in rows])
        for r in rows:
            cls = classify_reg(r["baseline_relL2"], r[arm], r["context_len"])
            if cls in ("permitted", "violation"):
                regressions.append({
                    "position": pos, "ctx": r["context_len"], "head": r["head"],
                    "baseline": r["baseline_relL2"], "groupmax": r[arm],
                    "delta_abs": r[arm] - r["baseline_relL2"],
                    "delta_rel": (r[arm] - r["baseline_relL2"]) / max(r["baseline_relL2"], 1e-30),
                    "class": cls,
                })
        per_pos[pos] = {
            "ctx": rows[0]["context_len"], "n_heads": len(rows),
            "baseline_mean": float(base.mean()), "groupmax_mean": float(gm.mean()),
            "baseline_p95": pctl(base, 95), "groupmax_p95": pctl(gm, 95),
            "mean_improves": bool(gm.mean() < base.mean()),
            "p95_improves": bool(pctl(gm, 95) < pctl(base, 95)),
        }
    n_viol = sum(1 for x in regressions if x["class"] == "violation")
    return {
        "arm": arm,
        "per_position": per_pos,
        "aggregate_monotone_all_positions": all(
            v["mean_improves"] and v["p95_improves"] for v in per_pos.values()),
        "n_permitted": sum(1 for x in regressions if x["class"] == "permitted"),
        "n_violations": n_viol,
        "regressions": sorted(regressions, key=lambda z: -z["delta_abs"]),
        "envelope_pass": bool(
            n_viol == 0 and all(v["mean_improves"] and v["p95_improves"] for v in per_pos.values())),
    }


def _group_table(curve):
    """Per (position, kv_head): B_max, B_sum, B_union."""
    g = {}
    for r in by_budget(curve, "union"):
        g[(r["position"], r["kv_head"])] = (r["B_max"], r["B_sum"], r["B_union"], r["context_len"])
    return g


def meas3_budget(curve, hc):
    g = _group_table(curve)
    keys = sorted(g)
    Bmax = np.array([g[k][0] for k in keys], float)
    Bsum = np.array([g[k][1] for k in keys], float)
    Buni = np.array([g[k][2] for k in keys], float)
    ctxs = np.array([g[k][3] for k in keys], float)

    # train/test split by position (both span short..long context)
    positions = sorted({k[0] for k in keys})
    train_pos = set(positions[0::2])
    test_pos = set(positions[1::2]) or set(train_pos)  # single-position guard
    tr = np.array([k[0] in train_pos for k in keys])
    te = ~tr

    def quality_at(bmap):
        """Mean/p95 groupmax relL2 across all heads when each group uses B_grp=bmap[key]."""
        vals = []
        for r in by_budget(curve, "union"):
            k = (r["position"], r["kv_head"])
            vals.append(interp_relL2(hc, r["position"], r["head"], bmap[k]))
        return float(np.mean(vals)), pctl(vals, 95)

    def bytes_of(bmap):
        return float(256 * sum(bmap.values()))

    # candidate (a) sum, (b) union oracle
    bmap_sum = {k: g[k][1] for k in keys}
    bmap_uni = {k: g[k][2] for k in keys}

    # candidate (c) fitted single-rung: linear B_grp = c0 + c1*Bmax + c2*Bsum on train
    A = np.stack([np.ones_like(Bmax), Bmax, Bsum], axis=1)
    coef, *_ = np.linalg.lstsq(A[tr], Buni[tr], rcond=None)
    pred = A @ coef
    pred = np.clip(np.round(pred), Bmax, ctxs)   # never below max B_h; never above ctx
    bmap_fit = {keys[i]: int(pred[i]) for i in range(len(keys))}

    # single-multiplier m*Bmax fit on train (interpretable rung rule)
    m = float(np.sum(Buni[tr] * Bmax[tr]) / max(np.sum(Bmax[tr] ** 2), 1e-9))
    predm = np.clip(np.round(m * Bmax), Bmax, ctxs)
    bmap_m = {keys[i]: int(predm[i]) for i in range(len(keys))}

    def resid(predarr):
        r = predarr - Buni
        return {
            "bytes_vs_union": float(256 * predarr.sum() / (256 * Buni.sum())),
            "resid_mean": float(r.mean()), "resid_p95": pctl(np.abs(r), 95),
            "underprov_frac_test": float(np.mean((predarr[te] < Buni[te]))),
            "test_resid_mean": float(r[te].mean()), "test_resid_p95": pctl(np.abs(r[te]), 95),
        }

    out = {"train_positions": sorted(train_pos), "test_positions": sorted(test_pos),
           "candidates": {}}
    for name, bmap, extra in [
        ("sum_Bh", bmap_sum, {}),
        ("union_oracle", bmap_uni, {}),
        ("fitted_linear", bmap_fit, {"coef_[1,Bmax,Bsum]": [float(c) for c in coef], **resid(pred)}),
        ("fitted_mult_Bmax", bmap_m, {"m": m, **resid(predm)}),
    ]:
        qmean, qp95 = quality_at(bmap)
        bmean, bp95 = quality_at(bmap_uni)
        out["candidates"][name] = {
            "total_bytes_MB": bytes_of(bmap) / 1e6,
            "bytes_vs_union": bytes_of(bmap) / bytes_of(bmap_uni),
            "quality_mean_relL2": qmean, "quality_p95_relL2": qp95,
            **extra,
        }
    return out


def meas4_miss(curve):
    """Missed frozen-committed tokens' attention-weight mass share at B_grp=|union|."""
    u = by_budget(curve, "union")
    per_pos = {}
    for pos in sorted({r["position"] for r in u}):
        rows = [r for r in u if r["position"] == pos]
        miss = np.array([r["missed_weight_mass"] for r in rows])
        comm = np.array([r["committed_weight_mass"] for r in rows])
        tot = np.array([r["total_weight_mass"] for r in rows])
        # share of each head's committed attention mass that is missed
        share = miss / np.maximum(comm, 1e-30)
        per_pos[pos] = {
            "ctx": rows[0]["context_len"],
            "missed_mass_share_of_committed_mean": float(share.mean()),
            "missed_mass_share_of_committed_max": float(share.max()),
            "missed_mass_share_of_total_mean": float((miss / np.maximum(tot, 1e-30)).mean()),
            "missed_mass_share_of_total_max": float((miss / np.maximum(tot, 1e-30)).max()),
        }
    # prefetch framing: coverage recall at |union| = prefetch recall; backstop
    # (per-head commit / fetch-on-miss) closes the rest.
    cov = np.array([r["coverage_recall"] for r in u])
    return {"per_position": per_pos,
            "prefetch_recall_mean": float(cov.mean()),
            "prefetch_recall_p05": pctl(cov, 5),
            "prefetch_recall_min": float(cov.min())}


def ortheta_row(ortheta, theta=2e-11):
    rows = [r for r in ortheta if abs(r["theta"] - theta) < theta * 1e-6]
    per_pos = {}
    for pos in sorted({r["position"] for r in rows}):
        s = [r for r in rows if r["position"] == pos]
        bt = np.array([r["B_theta"] for r in s], float)
        bu = np.array([r["B_union"] for r in s], float)
        base = np.array([r["baseline_relL2"] for r in s])
        gm = np.array([r["ortheta_relL2_frozenK"] for r in s])
        cov = np.array([r["coverage_recall"] for r in s])
        per_pos[pos] = {
            "ctx": s[0]["context_len"],
            "bytes_vs_union": float(bt.mean() / max(bu.mean(), 1)),
            "coverage_mean": float(cov.mean()),
            "relL2_mean": float(gm.mean()), "baseline_mean": float(base.mean()),
            "mean_improves": bool(gm.mean() < base.mean()),
        }
    return {"theta": theta, "per_position": per_pos}


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dir holding run_q*/gqa_group_max_*.csv")
    ap.add_argument("--lab_dirs", nargs="*", default=[],
                    help="v_threshold_lab */lab dirs for the #12 sanity gate")
    ap.add_argument("--out", required=True, help="output report json")
    args = ap.parse_args()

    curve = as_num(load_rows(args.root, "gqa_group_max_curve.csv"))
    ortheta = as_num(load_rows(args.root, "gqa_group_max_ortheta.csv"))
    if not curve:
        raise SystemExit(f"no gqa_group_max_curve.csv under {args.root}/run_q*/")
    hc = head_curve(curve)

    report = {
        "meta": {
            "curve_rows": len(curve), "ortheta_rows": len(ortheta),
            "positions": sorted({r["position"] for r in curve}),
            "contexts": sorted({r["context_len"] for r in curve}),
            "n_heads_per_pos": len({r["head"] for r in curve}),
            "budgets": sorted({r["budget_label"] for r in curve}),
        },
        "sanity": {
            "issue12_reproduction": sanity_12(args.lab_dirs) if args.lab_dirs else None,
            "machinery_gate": machinery_gate(curve),
        },
        "meas1_coverage_matched_bytes": meas1_coverage(curve),
        "meas2_envelope_binding_gate": meas2_envelope(curve),
        "meas3_budget_rule": meas3_budget(curve, hc),
        "meas4_miss_structure": meas4_miss(curve),
        "ortheta_curiosity": ortheta_row(ortheta) if ortheta else None,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
