#!/usr/bin/env python3
"""Validate a dependency epoch trace (issue #11, Phase 1).

Gate 2 (reconciliation): for every traced (qidx, head), the sum of per-epoch
walk-MB contributions equals the runner's own walk_step_MB_per_head for the
committed row (from per_head_joint_policy.csv), to float roundoff.

Gate 3 (GQA union reproduction): K/V union-over-sum recomputed per kv-head
group from the trace's committed token sets equals the runner's
gqa_union_stats.csv exactly.

Also reports per-head epoch counts and mean K-escalations/step (gate 4 stats).

Usage:
  validate_epoch_trace.py --trace_dir DIR --run_dir DIR [--tol 1e-9]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def _load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace_dir", required=True)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--tol", type=float, default=1e-9, help="abs MB tolerance for gate 2")
    args = ap.parse_args()

    trace_dir = Path(args.trace_dir)
    run_dir = Path(args.run_dir)

    per_head = _load_csv(run_dir / "per_head_joint_policy.csv")
    gqa = _load_csv(run_dir / "gqa_union_stats.csv")
    index_rows = _load_csv(trace_dir / "epoch_trace_index.csv")

    if not index_rows:
        print("FAIL: no epoch_trace_index.csv rows")
        return 1

    # runner walk_step_MB_per_head keyed by (qidx, head), main row only
    runner_walk: dict[tuple[int, int], float] = {}
    for r in per_head:
        vsr = str(r.get("v_selection_rule", ""))
        if "+la_" in vsr:
            continue
        key = (int(r["qidx"]), int(r["head"]))
        runner_walk[key] = float(r["walk_step_MB_per_head"])

    # ---- Gate 2: reconciliation ----
    g2_max_err = 0.0
    g2_worst = None
    g2_fail = 0
    esc_counts: list[int] = []
    epoch_counts: list[int] = []
    per_head_report: list[tuple] = []
    trace_committed: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, int]] = {}

    for row in index_rows:
        qidx = int(row["qidx"])
        head = int(row["head"])
        kv_head = int(row["kv_head"])
        fp = trace_dir / str(row["file"])
        d = np.load(fp, allow_pickle=False)
        contrib = np.asarray(d["epoch_walk_mb_contribution"], dtype=np.float64)
        recon = float(np.sum(contrib))
        walk_from_index = float(d["walk_step_MB_per_head"])
        # internal self-consistency (index vs npz field)
        assert abs(walk_from_index - float(row["walk_step_MB_per_head"])) < 1e-12
        runner_val = runner_walk.get((qidx, head))
        ref = runner_val if runner_val is not None else walk_from_index
        err = abs(recon - ref)
        if err > g2_max_err:
            g2_max_err = err
            g2_worst = (qidx, head, recon, ref)
        if err > args.tol:
            g2_fail += 1
        n_ep = int(d["n_epochs"])
        true_action = np.asarray(d["epoch_true_action_kind_code"], dtype=np.int64)
        n_kup = int(np.count_nonzero(true_action == 1))
        esc_counts.append(n_kup)
        epoch_counts.append(n_ep)
        ck = np.asarray(d["committed_k_tokens"], dtype=np.int64)
        cv = np.asarray(d["committed_v_tokens"], dtype=np.int64)
        trace_committed[(qidx, head)] = (ck, cv, kv_head)
        per_head_report.append((qidx, head, n_ep, n_kup, recon, ref, err))
        d.close()

    g2_ok = g2_fail == 0

    # ---- Gate 3: GQA union reproduction ----
    # recompute per (qidx, kv_head) group from committed token sets
    grp_k: dict[tuple[int, int], list[np.ndarray]] = defaultdict(list)
    grp_v: dict[tuple[int, int], list[np.ndarray]] = defaultdict(list)
    for (qidx, head), (ck, cv, kv_head) in trace_committed.items():
        grp_k[(qidx, kv_head)].append(ck)
        grp_v[(qidx, kv_head)].append(cv)

    trace_factor: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    for gkey in grp_k:
        ks = grp_k[gkey]
        vs = grp_v[gkey]
        k_sum = int(sum(int(a.size) for a in ks))
        v_sum = int(sum(int(a.size) for a in vs))
        k_union = int(np.unique(np.concatenate(ks)).size) if ks else 0
        v_union = int(np.unique(np.concatenate(vs)).size) if vs else 0
        trace_factor[gkey] = (k_sum, k_union, v_sum, v_union)

    g3_fail = 0
    g3_checked = 0
    g3_details: list[str] = []
    for r in gqa:
        qidx = int(r["qidx"])
        kv_head = int(r["kv_head"])
        gkey = (qidx, kv_head)
        if gkey not in trace_factor:
            continue
        g3_checked += 1
        k_sum, k_union, v_sum, v_union = trace_factor[gkey]
        exp = (int(r["k_sum_tokens"]), int(r["k_union_tokens"]), int(r["v_sum_tokens"]), int(r["v_union_tokens"]))
        got = (k_sum, k_union, v_sum, v_union)
        if exp != got:
            g3_fail += 1
            g3_details.append(f"  qidx={qidx} kv_head={kv_head} runner={exp} trace={got}")
    g3_ok = (g3_fail == 0) and (g3_checked > 0)

    # ---- Report ----
    print("=" * 70)
    print("EPOCH TRACE VALIDATION")
    print("=" * 70)
    print(f"traced files: {len(index_rows)}")
    print(f"mean epochs/head: {np.mean(epoch_counts):.4f}  (min {min(epoch_counts)}, max {max(epoch_counts)})")
    print(f"mean K-escalations/step: {np.mean(esc_counts):.4f}")
    print(f"total K-escalations: {int(sum(esc_counts))}  over {len(esc_counts)} heads")
    print("-" * 70)
    print(f"GATE 2 (reconciliation): {'PASS' if g2_ok else 'FAIL'}  "
          f"max_abs_err={g2_max_err:.3e} MB  tol={args.tol:.1e}  n_fail={g2_fail}")
    if g2_worst is not None:
        print(f"  worst: qidx={g2_worst[0]} head={g2_worst[1]} recon={g2_worst[2]:.9f} runner={g2_worst[3]:.9f}")
    print("-" * 70)
    print(f"GATE 3 (GQA union reproduction): {'PASS' if g3_ok else 'FAIL'}  "
          f"groups_checked={g3_checked}  n_fail={g3_fail}")
    for det in g3_details[:20]:
        print(det)
    print("-" * 70)
    print("per-head epoch/escalation counts (qidx, head, n_epochs, n_kup):")
    for qidx, head, n_ep, n_kup, recon, ref, err in sorted(per_head_report):
        print(f"  q{qidx} h{head:2d}  epochs={n_ep}  k_up={n_kup}  walk_recon={recon:.6f}  err={err:.2e}")

    summary = {
        "traced_files": len(index_rows),
        "mean_epochs_per_head": float(np.mean(epoch_counts)),
        "mean_k_escalations_per_step": float(np.mean(esc_counts)),
        "gate2_reconciliation_pass": bool(g2_ok),
        "gate2_max_abs_err_MB": float(g2_max_err),
        "gate2_n_fail": int(g2_fail),
        "gate3_gqa_union_pass": bool(g3_ok),
        "gate3_groups_checked": int(g3_checked),
        "gate3_n_fail": int(g3_fail),
    }
    (trace_dir / "validation_report.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print("=" * 70)
    print(json.dumps(summary, indent=2, sort_keys=True))

    return 0 if (g2_ok and g3_ok) else 2


if __name__ == "__main__":
    sys.exit(main())
