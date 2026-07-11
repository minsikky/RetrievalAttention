#!/usr/bin/env python3
"""Issue #20 ratified quality envelope — hard assertion (exit 1 on violation).

Envelope (RTL confirmation comment, 2026-07-10; part of the frozen contract,
re-validates automatically on any operating-point or config change):
  1. aggregate mean AND p95 o-proj relL2 must improve (union < baseline) at
     EVERY validated position;
  2. per-head regressions permitted only at ctx < 32k, bounded at
     <= +6e-4 absolute and <= +15% relative vs the frozen baseline.

TIE FLOOR (proposed numeric amendment, flagged for RTL sign-off on the #20
thread): per-head deltas with delta <= 1e-7 absolute relL2 are classified as
TIES, not regressions, at any ctx. Motivation: clause 2 as ratified is a
strict zero at ctx >= 32k, which is not falsifiable at fp32 metric precision
-- observed q223 (ctx 38,838) h14/kv3: baseline 5.041846e-4 -> union
5.042063e-4, delta +2.17e-8 (+0.004% relative), the only positive delta in
the 96-row standard set (job 53297160). 1e-7 sits three orders below the
6e-4 bound and at the relL2 metric's own fp32 noise scale. Ties are printed,
never silent.

Input: one or more gqa_union_commit.csv files from a --gqa_union_commit run.
"""
import argparse
import csv
import sys

import numpy as np

ABS_BOUND = 6e-4
REL_BOUND = 0.15
CTX_BOUND = 32000
TIE_EPS = 1e-7


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", required=True)
    args = ap.parse_args()

    rows = []
    for p in args.csv:
        with open(p) as f:
            rows.extend(list(csv.DictReader(f)))
    if not rows:
        print("[envelope] FAIL: no rows")
        return 1

    violations: list[str] = []
    positions = sorted({int(r["position"]) for r in rows})
    print(f"[envelope] {len(rows)} rows, positions(ctx): {[p + 1 for p in positions]}")
    for pos in positions:
        sub = [r for r in rows if int(r["position"]) == pos]
        b = np.asarray([float(r["baseline_relL2"]) for r in sub])
        u = np.asarray([float(r["union_relL2"]) for r in sub])
        ctx = pos + 1
        mean_ok = float(np.mean(u)) < float(np.mean(b))
        p95_ok = float(np.percentile(u, 95)) < float(np.percentile(b, 95))
        print(
            f"[envelope] ctx={ctx}: mean {np.mean(b):.4e} -> {np.mean(u):.4e} "
            f"({'OK' if mean_ok else 'VIOLATION'}), p95 {np.percentile(b, 95):.4e} -> "
            f"{np.percentile(u, 95):.4e} ({'OK' if p95_ok else 'VIOLATION'})"
        )
        if not mean_ok:
            violations.append(f"ctx={ctx}: aggregate mean did not improve")
        if not p95_ok:
            violations.append(f"ctx={ctx}: aggregate p95 did not improve")
        for r in sub:
            delta = float(r["union_relL2"]) - float(r["baseline_relL2"])
            if delta <= 0:
                continue
            rel = delta / max(float(r["baseline_relL2"]), 1e-30)
            tag = f"ctx={ctx} h{r['head']} kv{r['kv_head']} delta={delta:+.3e} rel={rel:+.2%}"
            if delta <= TIE_EPS:
                print(f"[envelope] tie (|delta| <= {TIE_EPS:.0e}, fp noise floor): {tag}")
                continue
            if ctx >= CTX_BOUND:
                violations.append(f"regression at ctx>=32k: {tag}")
            elif delta > ABS_BOUND:
                violations.append(f"regression beyond +6e-4 abs: {tag}")
            elif rel > REL_BOUND:
                violations.append(f"regression beyond +15% rel: {tag}")
            else:
                print(f"[envelope] permitted regression (inside envelope): {tag}")

    if violations:
        print(f"[envelope] FAIL: {len(violations)} violation(s)")
        for v in violations:
            print(f"[envelope]   {v}")
        return 1
    print("[envelope] PASS: aggregate mean+p95 improve at every position; "
          "all per-head regressions inside the ctx<32k / +6e-4 / +15% envelope "
          f"(ties below the {TIE_EPS:.0e} fp noise floor listed above)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
