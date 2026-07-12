#!/usr/bin/env python3
"""Issue #21 golden-1 (draft-selection block) combiner.

Reads the three stage-2 CPU dumps produced by run_draft_verify_golden1.sbatch
(SELECTOR_PQ_JOINT_DRAFT_MODE in {off, start1, start2}) on the standard golden
positions (q159/q223/q287) x heads {0,8,16,24}, and emits one
golden1_q{qidx}_h{head}.npz per (position, head) carrying, from identical scan
state:

  - scan / start inputs: proxy_mass_c, start_ki/vi, k_budgets, v_budgets,
    risk_scores (the risk ranking over the whole context);
  - the frozen escalation-walk committed set (frozen_committed_k_tokens,
    frozen_settled_ki) -- the reference;
  - the one-shot draft committed sets at the pinned rungs start+1 and start+2
    (draft_start1_committed_k_tokens @ start_ki+1, draft_start2_* @ start_ki+2);
  - the superset / miss bitmap: for each draft, which frozen-selected K tokens
    the draft set CONTAINS vs MISSES (packed over the sorted frozen set), plus
    the miss token list and the draft-extra token list;
  - measured K/V recall of each draft set vs frozen.

Gate cross-checks (printed, and asserted unless --no-assert):
  (1) frozen_committed_k_tokens == the pre-union stage-2 goldens'
      per-head committed K span (group_committed_k_flat[own head]) for the same
      q/h -- proves the dump did not disturb frozen selection semantics.
  (2) off-mode committed set == the frozen reference (same code path; the
      draft override is not entered).  [structural: off IS the frozen dump]
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STAGE2 = HERE.parent / "stage2_union_commit_20260710"
STD = [(159, [0, 8, 16, 24]), (223, [0, 8, 16, 24]), (287, [0, 8, 16, 24])]
HEAD_DIM = 128
KEY_BYTES = 2
VALUE_BYTES = 2


def _load(dump_root: Path, mode: str, qidx: int, head: int):
    return np.load(dump_root / mode / "gold_dump" / f"golden2_q{qidx}_h{head}.npz", allow_pickle=True)


def _stage2_frozen_k(qidx: int, head: int):
    """Per-head committed K span from the checked-in union-commit goldens."""
    f = STAGE2 / f"golden2_q{qidx}_h{head}.npz"
    if not f.exists():
        return None
    z = np.load(f, allow_pickle=True)
    heads = z["union_group_heads"].tolist()
    if head not in heads:
        return None
    g = heads.index(head)
    off = z["group_committed_k_offsets"]
    flat = z["group_committed_k_flat"]
    return np.sort(flat[int(off[g]):int(off[g + 1])].astype(np.int64))


def _recall(frozen: np.ndarray, draft: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    fset = frozen
    dset = np.asarray(draft, dtype=np.int64)
    contains = np.isin(fset, dset, assume_unique=False)  # over sorted frozen
    missed = fset[~contains]
    extra = np.setdiff1d(dset, fset, assume_unique=False)
    recall = float(contains.mean()) if fset.size else float("nan")
    return recall, contains, missed, extra


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump_root", required=True,
                    help="scratch root holding {off,start1,start2}/gold_dump")
    ap.add_argument("--out_dir", default=str(HERE))
    ap.add_argument("--no_assert", action="store_true")
    args = ap.parse_args()
    dump_root = Path(args.dump_root)
    out_dir = Path(args.out_dir)

    rows = []
    gate1_all_ok = True
    for qidx, heads in STD:
        for head in heads:
            zo = _load(dump_root, "off", qidx, head)
            z1 = _load(dump_root, "start1", qidx, head)
            z2 = _load(dump_root, "start2", qidx, head)
            kf = np.sort(zo["committed_k_tokens"].astype(np.int64))
            vf = np.sort(zo["committed_v_tokens"].astype(np.int64))
            kd1 = np.sort(z1["committed_k_tokens"].astype(np.int64))
            kd2 = np.sort(z2["committed_k_tokens"].astype(np.int64))
            vd1 = np.sort(z1["committed_v_tokens"].astype(np.int64))
            vd2 = np.sort(z2["committed_v_tokens"].astype(np.int64))

            # gate 1: frozen vs stage-2 committed K span
            s2 = _stage2_frozen_k(qidx, head)
            if s2 is not None:
                gate1 = bool(kf.size == s2.size and np.array_equal(kf, s2))
            else:
                gate1 = None
            if gate1 is False:
                gate1_all_ok = False

            r1, c1, miss1, extra1 = _recall(kf, kd1)
            r2, c2, miss2, extra2 = _recall(kf, kd2)
            rv1, cv1, missv1, extrav1 = _recall(vf, vd1)
            rv2, cv2, missv2, extrav2 = _recall(vf, vd2)

            np.savez_compressed(
                out_dir / f"golden1_q{qidx}_h{head}.npz",
                qidx=int(qidx), head=int(head), kv_head=int(zo["kv_head"]),
                position=int(zo["position"]), context_len=int(zo["context_len"]),
                proxy_mass_c=int(zo["proxy_mass_c"]),
                start_ki=int(zo["start_ki"]), start_vi=int(zo["start_vi"]),
                k_budgets=zo["k_budgets"].astype(np.int64),
                v_budgets=zo["v_budgets"].astype(np.int64),
                risk_scores=zo["risk_scores"].astype(np.float64),
                frozen_settled_ki=int(zo["settled_ki"]),
                frozen_settled_vi=int(zo["settled_vi"]),
                frozen_committed_k_tokens=kf,
                frozen_committed_v_tokens=vf,
                draft_start1_settled_ki=int(z1["settled_ki"]),
                draft_start1_settled_vi=int(z1["settled_vi"]),
                draft_start1_committed_k_tokens=kd1,
                draft_start1_committed_v_tokens=vd1,
                draft_start2_settled_ki=int(z2["settled_ki"]),
                draft_start2_settled_vi=int(z2["settled_vi"]),
                draft_start2_committed_k_tokens=kd2,
                draft_start2_committed_v_tokens=vd2,
                # superset/miss bitmaps over the SORTED frozen committed K set
                draft_start1_contains_frozen_k_packed=np.packbits(c1),
                draft_start2_contains_frozen_k_packed=np.packbits(c2),
                draft_start1_missed_frozen_k_tokens=miss1,
                draft_start2_missed_frozen_k_tokens=miss2,
                draft_start1_extra_k_tokens=extra1,
                draft_start2_extra_k_tokens=extra2,
                draft_start1_contains_frozen_v_packed=np.packbits(cv1),
                draft_start2_contains_frozen_v_packed=np.packbits(cv2),
                draft_start1_k_recall=float(r1),
                draft_start2_k_recall=float(r2),
                draft_start1_v_recall=float(rv1),
                draft_start2_v_recall=float(rv2),
                frozen_matches_stage2=bool(gate1) if gate1 is not None else False,
                frozen_stage2_available=gate1 is not None,
            )
            rows.append(dict(
                qidx=qidx, head=head, kv_head=int(zo["kv_head"]),
                context_len=int(zo["context_len"]), start_ki=int(zo["start_ki"]),
                frozen_settled_ki=int(zo["settled_ki"]),
                n_frozen_k=int(kf.size), n_draft1_k=int(kd1.size), n_draft2_k=int(kd2.size),
                k_recall_start1=round(r1, 6), k_recall_start2=round(r2, 6),
                n_miss_start1=int(miss1.size), n_miss_start2=int(miss2.size),
                n_extra_start1=int(extra1.size), n_extra_start2=int(extra2.size),
                n_frozen_v=int(vf.size), v_recall_start1=round(rv1, 6), v_recall_start2=round(rv2, 6),
                gate1_frozen_eq_stage2=("" if gate1 is None else int(gate1)),
            ))
            g1 = "n/a" if gate1 is None else ("OK" if gate1 else "MISMATCH")
            print(f"q{qidx} h{head}: |Kf|={kf.size} |Kd1|={kd1.size} |Kd2|={kd2.size} "
                  f"recall1={r1:.4f} recall2={r2:.4f} miss1={miss1.size} miss2={miss2.size} "
                  f"gate1={g1}")

    csv_path = out_dir / "golden1_draft_selection.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    r1s = [r["k_recall_start1"] for r in rows]
    r2s = [r["k_recall_start2"] for r in rows]
    print(f"\nK-recall start1: mean={np.mean(r1s):.4f} min={np.min(r1s):.4f}")
    print(f"K-recall start2: mean={np.mean(r2s):.4f} min={np.min(r2s):.4f}")
    print(f"GATE1 frozen==stage2 (all q/h): {'ALL OK' if gate1_all_ok else 'MISMATCH PRESENT'}")
    print(f"wrote {csv_path.name} + {len(rows)} golden1_*.npz")
    if not args.no_assert:
        assert gate1_all_ok, "GATE1 failed: frozen committed set != stage-2 goldens"


if __name__ == "__main__":
    main()
