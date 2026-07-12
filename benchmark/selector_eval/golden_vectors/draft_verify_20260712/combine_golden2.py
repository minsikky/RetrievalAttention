#!/usr/bin/env python3
"""Issue #21 golden-2 (verify replay trace) committed-set operand combiner.

Reads the two stage-2 CPU dumps produced by run_draft_verify_golden2.sbatch
(off + start1) over a contiguous decode segment (qidx 208..223 = 16 sampled L16
decode positions) x kv-group-0 heads {0,1,2,3}, and builds the RTL verify-FSM
operands per round (a round = a window of k consecutive decode positions whose
k verify consumers share one gather):

  (a) drafted_union_k/v_tokens: the union over the round's draft positions AND
      the group's 4 heads of the start+1 one-shot committed set -- the streamed
      committed-KV list the k*4 verify consumers read once;
  (b) per-position frozen committed set (group union of the frozen walk);
  (c) per-position membership bitmap vs the drafted union + the fetch-on-miss
      token stream (frozen tokens absent from the drafted union);
  plus per-round union sizes (rows + bytes, layer 16 / kv-group 0).

Delivered start+1-only (see README): the #14 predicted-theta union component is
not reproduced here; the RTL FSM shape is identical. The accepted-prefix /
rollback ground truth (part d) is NOT derived from this single-layer trace -- it
comes from the full-model GPU acceptance runs, tabulated in
golden2_accepted_prefix_rounds.csv (produced by make_acceptance_csvs.py from
acceptance.json). See README for why the two sources are distinct.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
GROUP_HEADS = [0, 1, 2, 3]
KV_GROUP = 0
LAYER_ID = 16
HEAD_DIM = 128
KEY_BYTES = 2
VALUE_BYTES = 2
ROUND_KS = (4, 8)


def _union(dump_root: Path, mode: str, qidx: int, field: str) -> np.ndarray:
    acc = []
    for h in GROUP_HEADS:
        z = np.load(dump_root / mode / "gold_dump" / f"golden2_q{qidx}_h{h}.npz", allow_pickle=True)
        acc.append(z[field].astype(np.int64))
    return np.unique(np.concatenate(acc)) if acc else np.zeros(0, np.int64)


def _discover_positions(dump_root: Path) -> list[int]:
    qs = set()
    for p in (dump_root / "off" / "gold_dump").glob("golden2_q*_h0.npz"):
        qs.add(int(p.stem.split("_")[1][1:]))
    return sorted(qs)


def _ragged(lst):
    flat = np.concatenate(lst) if lst else np.zeros(0, np.int64)
    off = np.zeros(len(lst) + 1, np.int64)
    for i, a in enumerate(lst):
        off[i + 1] = off[i] + len(a)
    return flat.astype(np.int64), off


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump_root", required=True)
    ap.add_argument("--out_dir", default=str(HERE))
    args = ap.parse_args()
    dump_root = Path(args.dump_root)
    out_dir = Path(args.out_dir)

    positions = _discover_positions(dump_root)
    print(f"segment positions (qidx): {positions}")

    # per-position group unions (frozen + draft), cached
    froz_k, froz_v, drft_k, drft_v, ctx_of = {}, {}, {}, {}, {}
    for q in positions:
        froz_k[q] = _union(dump_root, "off", q, "committed_k_tokens")
        froz_v[q] = _union(dump_root, "off", q, "committed_v_tokens")
        drft_k[q] = _union(dump_root, "start1", q, "committed_k_tokens")
        drft_v[q] = _union(dump_root, "start1", q, "committed_v_tokens")
        z = np.load(dump_root / "off" / "gold_dump" / f"golden2_q{q}_h0.npz", allow_pickle=True)
        ctx_of[q] = int(z["context_len"])

    rows = []
    for k in ROUND_KS:
        n_rounds = len(positions) // k
        for ri in range(n_rounds):
            rpos = positions[ri * k:ri * k + k]
            union_k = np.unique(np.concatenate([drft_k[q] for q in rpos]))
            union_v = np.unique(np.concatenate([drft_v[q] for q in rpos]))
            # per-position frozen sets, membership bitmaps, miss streams
            froz_sets, memb_packed, miss_sets = [], [], []
            per_pos_recall = []
            for q in rpos:
                fk = froz_k[q]
                contains = np.isin(fk, union_k, assume_unique=False)
                froz_sets.append(fk)
                memb_packed.append(np.packbits(contains))
                miss = fk[~contains]
                miss_sets.append(miss)
                per_pos_recall.append(float(contains.mean()) if fk.size else float("nan"))
                rows.append(dict(
                    k=k, round_index=ri, qidx=q, context_len=ctx_of[q],
                    drafted_union_k_rows=int(union_k.size),
                    drafted_union_k_bytes=int(union_k.size * HEAD_DIM * KEY_BYTES),
                    drafted_union_v_rows=int(union_v.size),
                    drafted_union_v_bytes=int(union_v.size * HEAD_DIM * VALUE_BYTES),
                    frozen_k_rows=int(fk.size),
                    n_miss_fetch_on_miss=int(miss.size),
                    draft_union_recall_of_frozen=round(per_pos_recall[-1], 6),
                ))
            ff, fo = _ragged(froz_sets)
            mf, mo = _ragged(miss_sets)
            # membership packed is ragged too (bytes per position vary)
            memb_flat = np.concatenate(memb_packed) if memb_packed else np.zeros(0, np.uint8)
            memb_off = np.zeros(len(memb_packed) + 1, np.int64)
            for i, a in enumerate(memb_packed):
                memb_off[i + 1] = memb_off[i] + len(a)
            np.savez_compressed(
                out_dir / f"golden2_k{k}_round{ri}.npz",
                layer_id=LAYER_ID, kv_group=KV_GROUP, group_heads=np.asarray(GROUP_HEADS, np.int64),
                k=int(k), round_index=int(ri),
                round_positions_qidx=np.asarray(rpos, np.int64),
                round_context_lens=np.asarray([ctx_of[q] for q in rpos], np.int64),
                drafted_union_k_tokens=union_k,
                drafted_union_v_tokens=union_v,
                drafted_union_k_rows=int(union_k.size),
                drafted_union_k_bytes=int(union_k.size * HEAD_DIM * KEY_BYTES),
                drafted_union_v_rows=int(union_v.size),
                drafted_union_v_bytes=int(union_v.size * HEAD_DIM * VALUE_BYTES),
                # per-position frozen committed sets (ragged, ordered by round_positions)
                frozen_committed_k_flat=ff, frozen_committed_k_offsets=fo,
                # per-position membership bitmap over each position's sorted frozen set
                membership_packed_flat=memb_flat, membership_packed_offsets=memb_off,
                # per-position fetch-on-miss stream (frozen tokens not in drafted union)
                fetch_on_miss_flat=mf, fetch_on_miss_offsets=mo,
                per_position_draft_union_recall=np.asarray(per_pos_recall, np.float64),
            )
            print(f"k={k} round{ri} pos={rpos}: union_k={union_k.size} rows "
                  f"({union_k.size*HEAD_DIM*KEY_BYTES/1e6:.2f} MB) union_v={union_v.size} "
                  f"recall/pos={[round(x,3) for x in per_pos_recall]}")

    csv_path = out_dir / "golden2_verify_rounds.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    recs = [r["draft_union_recall_of_frozen"] for r in rows]
    print(f"\ndraft-union recall of frozen (per position): mean={np.mean(recs):.4f} min={np.min(recs):.4f}")
    print(f"wrote {csv_path.name} + golden2_k*_round*.npz")


if __name__ == "__main__":
    main()
