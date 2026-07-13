#!/usr/bin/env python3
"""Emit per-position committed-set UNION per KV (GQA) group from an epoch trace.

Issue #24 P1 item (6): cheap, RTL-optional (RTL can also compute it from the
per-head committed sets). For each traced decode position, groups the 32 query
heads by their KV head and writes the union of `committed_k_tokens` /
`committed_v_tokens` over the heads sharing each KV head, plus per-group
union/sum sizes (the GQA-sharing factor RTL A/Bs). One npz per position:
`group_union_q{qidx}_ctx{context_len}.npz`.

Pure post-processor: reads only the epoch npz files; invents nothing.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def _csr(sets: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    offs = np.zeros(len(sets) + 1, dtype=np.int64)
    for i, s in enumerate(sets):
        offs[i + 1] = offs[i] + int(np.asarray(s).size)
    flat = (
        np.concatenate([np.asarray(s, dtype=np.int64) for s in sets])
        if sets
        else np.zeros(0, dtype=np.int64)
    )
    return flat, offs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace_dir", required=True)
    ap.add_argument("--out_dir", default="")
    args = ap.parse_args()
    trace_dir = Path(args.trace_dir)
    out_dir = Path(args.out_dir) if args.out_dir else trace_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # group[(qidx)] -> {kv_head -> {"k": [sets], "v": [sets], "heads": [...]}}
    per_qidx: dict[int, dict[int, dict]] = defaultdict(lambda: defaultdict(lambda: {"k": [], "v": [], "heads": []}))
    ctx_of: dict[int, int] = {}
    for f in sorted(trace_dir.glob("epoch_q*_h*.npz")):
        d = np.load(f, allow_pickle=True)
        qidx = int(d["qidx"])
        kvh = int(d["kv_head"])
        ctx_of[qidx] = int(d["context_len"])
        g = per_qidx[qidx][kvh]
        g["k"].append(np.asarray(d["committed_k_tokens"], dtype=np.int64))
        g["v"].append(np.asarray(d["committed_v_tokens"], dtype=np.int64))
        g["heads"].append(int(d["head"]))

    summary = []
    for qidx in sorted(per_qidx):
        kv_heads = sorted(per_qidx[qidx])
        ku_sets, vu_sets, sizes = [], [], []
        for kvh in kv_heads:
            g = per_qidx[qidx][kvh]
            ku = np.unique(np.concatenate(g["k"])) if g["k"] else np.zeros(0, np.int64)
            vu = np.unique(np.concatenate(g["v"])) if g["v"] else np.zeros(0, np.int64)
            k_sum = int(sum(int(s.size) for s in g["k"]))
            v_sum = int(sum(int(s.size) for s in g["v"]))
            ku_sets.append(ku)
            vu_sets.append(vu)
            sizes.append(
                {
                    "kv_head": int(kvh),
                    "heads": sorted(int(h) for h in g["heads"]),
                    "k_union": int(ku.size),
                    "k_sum": k_sum,
                    "k_share": (float(k_sum) / float(ku.size)) if ku.size else 0.0,
                    "v_union": int(vu.size),
                    "v_sum": v_sum,
                    "v_share": (float(v_sum) / float(vu.size)) if vu.size else 0.0,
                }
            )
        gk_flat, gk_off = _csr(ku_sets)
        gv_flat, gv_off = _csr(vu_sets)
        ctx = ctx_of[qidx]
        out = out_dir / f"group_union_q{qidx}_ctx{ctx}.npz"
        np.savez_compressed(
            out,
            qidx=np.int64(qidx),
            context_len=np.int64(ctx),
            kv_heads=np.asarray(kv_heads, dtype=np.int64),
            group_union_k_tokens=gk_flat,
            group_union_k_offsets=gk_off,
            group_union_v_tokens=gv_flat,
            group_union_v_offsets=gv_off,
            sizes_json=json.dumps(sizes),
        )
        summary.append({"qidx": int(qidx), "context_len": int(ctx), "groups": sizes})
        print(f"[emit_group_union] wrote {out.name}: {len(kv_heads)} groups")
    (out_dir / "group_union_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[emit_group_union] summary -> {out_dir / 'group_union_summary.json'}")


if __name__ == "__main__":
    main()
