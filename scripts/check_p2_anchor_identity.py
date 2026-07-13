#!/usr/bin/env python3
"""P2-specific gate (issue #24): the P2 anchor position 134,837 reuses the
golden Q/K/V verbatim, so its committed K/V sets must reproduce
epoch_trace_20260710 (q287) EXACTLY -- set-identical per head, all 32 heads.

Strongest check that the sequential-decode path did not perturb the contract
position. Exits nonzero on any mismatch.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p2_trace_dir", required=True, help="P2 trace dir (epoch_q7_h*.npz = anchor)")
    ap.add_argument("--golden_dir", required=True, help="epoch_trace_20260710 golden dir (epoch_q287_h*.npz)")
    ap.add_argument("--anchor_qidx", type=int, default=7)
    ap.add_argument("--anchor_position", type=int, default=134837)
    args = ap.parse_args()
    p2, gold = Path(args.p2_trace_dir), Path(args.golden_dir)

    bad = 0
    checked = 0
    for h in range(32):
        gf = gold / f"epoch_q287_h{h}.npz"
        nf = p2 / f"epoch_q{args.anchor_qidx}_h{h}.npz"
        if not gf.exists() or not nf.exists():
            print(f"h{h}: MISSING file ({'golden' if not gf.exists() else 'p2'})")
            bad += 1
            continue
        g = np.load(gf, allow_pickle=True)
        n = np.load(nf, allow_pickle=True)
        pos = int(n["position"])
        if pos != args.anchor_position:
            print(f"h{h}: anchor position {pos} != {args.anchor_position}")
            bad += 1
            continue
        ok = True
        for key in ("committed_k_tokens", "committed_v_tokens"):
            a = np.sort(np.asarray(n[key], dtype=np.int64))
            b = np.sort(np.asarray(g[key], dtype=np.int64))
            if a.shape != b.shape or not np.array_equal(a, b):
                inter = np.intersect1d(a, b).size
                print(f"h{h}: {key} MISMATCH p2={a.size} gold={b.size} intersect={inter}")
                ok = False
        # settled rung must also agree (same walk endpoint)
        for key in ("settled_ki", "settled_vi", "settled_k_budget", "settled_v_budget"):
            if key in g.files and key in n.files and int(n[key]) != int(g[key]):
                print(f"h{h}: {key} {int(n[key])} != golden {int(g[key])}")
                ok = False
        if not ok:
            bad += 1
        checked += 1
    print(f"[anchor-gate] heads checked={checked} mismatched={bad}")
    if bad == 0 and checked == 32:
        print("[anchor-gate] PASS: anchor 134,837 committed K/V sets set-identical to epoch_trace_20260710 for all 32 heads")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
