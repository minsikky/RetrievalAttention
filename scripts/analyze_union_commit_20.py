#!/usr/bin/env python3
"""Issue #20 union-commit analysis + validation gates.

Consumes one or more gqa_union_commit.csv shards, emits:
  - per (position, head) baseline vs union relL2 table
  - aggregates (mean/p95/max) for both arms, overall and per position
  - superset-hypothesis verdict (count + detail of union-worse rows)
  - union-size stats per group/position
  - gate-2 bytes cross-check vs the #11 epoch_trace_20260710 Phase-1 npz
    (per-head committed counts; group unions) and the epoch_replay oracle bytes
"""
import argparse
import csv
import glob
import math
import os

import numpy as np

CONTRACT = dict(head_dim=128, key_bytes=2, value_bytes=2)  # fp16 K/V rows
ORACLE_PHYSICAL_BYTES_Q287 = 174052608  # replay_sweep unlimited-window (174.1 MB)


def pct(xs, p):
    return float(np.percentile(np.asarray(xs, dtype=np.float64), p)) if xs else float("nan")


def load_rows(paths):
    rows = []
    for p in paths:
        with open(p) as f:
            rows.extend(list(csv.DictReader(f)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", required=True, help="gqa_union_commit.csv shard(s)")
    ap.add_argument("--epoch_trace_dir", default="benchmark/selector_eval/golden_vectors/epoch_trace_20260710")
    ap.add_argument("--out", default=None, help="optional path to write the per-(pos,head) table")
    args = ap.parse_args()

    rows = load_rows(args.csv)
    for r in rows:
        for k in ("qidx", "position", "head", "kv_head", "committed_k_tokens", "committed_v_tokens",
                  "union_k_tokens", "union_v_tokens", "union_k_hi_tokens", "union_k_lo_tokens",
                  "group_k_sum_tokens", "group_v_sum_tokens", "union_v_hi_reads", "union_v_lo_reads"):
            r[k] = int(r[k])
        for k in ("baseline_relL2", "union_relL2", "relL2_delta_union_minus_baseline",
                  "baseline_cosine", "union_cosine"):
            r[k] = float(r[k])
        r["union_worse_than_baseline"] = str(r["union_worse_than_baseline"]).strip().lower() in ("true", "1")

    positions = sorted({r["position"] for r in rows})
    ctx_of = {r["position"]: r["position"] + 1 for r in rows}

    lines = []
    def emit(s=""):
        lines.append(s)
        print(s)

    emit("=" * 100)
    emit("ISSUE #20 UNION-COMMIT QUALITY VALIDATION")
    emit("=" * 100)
    emit(f"rows={len(rows)}  positions(ctx)={[ (p, p+1) for p in positions ]}")

    # ---- per (position, head) table ----
    emit("")
    emit("PER (position, head) relL2:  baseline -> union   (delta = union - baseline; * = union WORSE)")
    emit(f"{'ctx':>8} {'head':>4} {'kv':>3} {'ki':>3} {'vi':>3} {'base_relL2':>12} {'union_relL2':>12} {'delta':>12} {'worse':>6}")
    for p in positions:
        for r in sorted([r for r in rows if r["position"] == p], key=lambda r: r["head"]):
            star = "*" if r["union_worse_than_baseline"] else ""
            emit(f"{p+1:>8} {r['head']:>4} {r['kv_head']:>3} {r['committed_ki']:>3} {r['committed_vi']:>3} "
                 f"{r['baseline_relL2']:>12.6e} {r['union_relL2']:>12.6e} "
                 f"{r['relL2_delta_union_minus_baseline']:>+12.3e} {star:>6}")

    # ---- aggregates ----
    def agg(subset, label):
        b = [r["baseline_relL2"] for r in subset]
        u = [r["union_relL2"] for r in subset]
        emit(f"  {label:<18} n={len(subset):>3}  "
             f"baseline mean={np.mean(b):.4e} p95={pct(b,95):.4e} max={np.max(b):.4e} | "
             f"union mean={np.mean(u):.4e} p95={pct(u,95):.4e} max={np.max(u):.4e}")

    emit("")
    emit("AGGREGATE relL2 (baseline vs union-commit):")
    agg(rows, "ALL")
    for p in positions:
        agg([r for r in rows if r["position"] == p], f"ctx={p+1}")

    # ---- superset hypothesis ----
    worse = [r for r in rows if r["union_worse_than_baseline"]]
    emit("")
    emit("SUPERSET-HYPOTHESIS VERDICT (union relL2 <= baseline for every head?):")
    emit(f"  worse rows: {len(worse)} / {len(rows)}")
    if worse:
        emit(f"  VERDICT: REFUTED -- {len(worse)} (position, head) rows are worse under union-commit.")
        emit(f"  {'ctx':>8} {'head':>4} {'kv':>3} {'base_relL2':>12} {'union_relL2':>12} {'delta':>12} {'rel_incr':>9}")
        for r in sorted(worse, key=lambda r: -r["relL2_delta_union_minus_baseline"]):
            rel = r["relL2_delta_union_minus_baseline"] / max(r["baseline_relL2"], 1e-30)
            emit(f"  {r['position']+1:>8} {r['head']:>4} {r['kv_head']:>3} {r['baseline_relL2']:>12.6e} "
                 f"{r['union_relL2']:>12.6e} {r['relL2_delta_union_minus_baseline']:>+12.3e} {rel:>+8.2%}")
        maxd = max(r["relL2_delta_union_minus_baseline"] for r in worse)
        emit(f"  worst absolute regression: {maxd:+.3e}  (max union relL2 overall: {max(r['union_relL2'] for r in rows):.4e})")
    else:
        emit("  VERDICT: CONFIRMED -- no head is worse under union-commit at any position.")

    # ---- union-size stats per group/position ----
    emit("")
    emit("UNION SIZES per (position, kv_group)   [group union is head-invariant]:")
    emit(f"{'ctx':>8} {'kv':>3} {'k_sum':>8} {'k_union':>8} {'k_u/sum':>8} {'v_sum':>8} {'v_union':>8} {'v_u/sum':>8} {'k_uni_frac_ctx':>14} {'v_uni_frac_ctx':>14}")
    group_union = {}  # (position, kv_head) -> (k_union, v_union)
    for p in positions:
        for kv in sorted({r["kv_head"] for r in rows if r["position"] == p}):
            g = [r for r in rows if r["position"] == p and r["kv_head"] == kv]
            r0 = g[0]
            group_union[(p, kv)] = (r0["union_k_tokens"], r0["union_v_tokens"])
            ctx = p + 1
            emit(f"{ctx:>8} {kv:>3} {r0['group_k_sum_tokens']:>8} {r0['union_k_tokens']:>8} "
                 f"{r0['union_k_tokens']/max(r0['group_k_sum_tokens'],1):>8.3f} "
                 f"{r0['group_v_sum_tokens']:>8} {r0['union_v_tokens']:>8} "
                 f"{r0['union_v_tokens']/max(r0['group_v_sum_tokens'],1):>8.3f} "
                 f"{r0['union_k_tokens']/ctx:>14.3f} {r0['union_v_tokens']/ctx:>14.3f}")

    # ---- gate 2: bytes/token cross-check vs #11 epoch_trace npz (q287) ----
    emit("")
    emit("GATE 2 -- token-count reconciliation vs #11 epoch_trace_20260710 (q287):")
    et = args.epoch_trace_dir
    npz_files = sorted(glob.glob(os.path.join(et, "epoch_q287_h*.npz")))
    if not npz_files:
        emit(f"  (no epoch_trace npz found under {et}; skipping)")
    else:
        per_head_ck = {}
        per_head_cv = {}
        grp_k_sets = {}
        grp_v_sets = {}
        for fp in npz_files:
            d = np.load(fp, allow_pickle=True)
            h = int(d["head"]); kv = int(d["kv_head"])
            ck = np.asarray(d["committed_k_tokens"], dtype=np.int64)
            cv = np.asarray(d["committed_v_tokens"], dtype=np.int64)
            per_head_ck[h] = ck
            per_head_cv[h] = cv
            grp_k_sets.setdefault(kv, []).append(ck)
            grp_v_sets.setdefault(kv, []).append(cv)
        # per-head committed count match (my run vs npz)
        q287_pos = 134837
        mism = 0
        for r in [r for r in rows if r["position"] == q287_pos]:
            h = r["head"]
            if h in per_head_ck:
                nk = int(np.unique(per_head_ck[h]).size)
                nv = int(np.unique(per_head_cv[h]).size)
                if nk != r["committed_k_tokens"] or nv != r["committed_v_tokens"]:
                    mism += 1
                    emit(f"    MISMATCH h{h}: mine ck={r['committed_k_tokens']} cv={r['committed_v_tokens']} "
                         f"vs npz ck={nk} cv={nv}")
        emit(f"  per-head committed count matches npz: {'ALL OK' if mism==0 else f'{mism} MISMATCHES'}"
             f"  (heads compared: {len([r for r in rows if r['position']==q287_pos])})")
        # group union match
        gmism = 0
        for kv in sorted(grp_k_sets):
            npz_ku = int(np.unique(np.concatenate(grp_k_sets[kv])).size)
            npz_vu = int(np.unique(np.concatenate(grp_v_sets[kv])).size)
            mine = group_union.get((q287_pos, kv))
            if mine is None:
                continue
            ok = (mine[0] == npz_ku and mine[1] == npz_vu)
            if not ok:
                gmism += 1
            emit(f"    kv{kv}: mine union_k={mine[0]} union_v={mine[1]} | npz union_k={npz_ku} union_v={npz_vu}  {'OK' if ok else 'MISMATCH'}")
        emit(f"  group-union token counts vs npz: {'ALL OK' if gmism==0 else f'{gmism} MISMATCHES'}")

    # ---- byte reconciliation vs epoch_replay oracle (q287) ----
    emit("")
    emit("BYTE RECONCILIATION vs #11 epoch_replay oracle (q287, unlimited-window = union-load model):")
    q287_pos = 134837
    if any(r["position"] == q287_pos for r in rows):
        kvs = sorted({r["kv_head"] for r in rows if r["position"] == q287_pos})
        k_union_total = sum(group_union[(q287_pos, kv)][0] for kv in kvs)
        v_union_total = sum(group_union[(q287_pos, kv)][1] for kv in kvs)
        # per-group hi/lo split (head-invariant within a group under max-tier).
        k_hi_total = sum([r for r in rows if r["position"] == q287_pos and r["kv_head"] == kv][0]["union_k_hi_tokens"] for kv in kvs)
        k_lo_total = sum([r for r in rows if r["position"] == q287_pos and r["kv_head"] == kv][0]["union_k_lo_tokens"] for kv in kvs)
        hd = CONTRACT["head_dim"]
        hi_row = hd * CONTRACT["key_bytes"]     # 256B fp16 / two-plane (A+B)
        lo_row = hd * 1                          # 128B int8 lo plane (A only)
        # basis (a): all union rows at full width (contract-width upper bound)
        est_full = k_union_total * hi_row + v_union_total * (hd * CONTRACT["value_bytes"])
        # basis (b): lo-aware K rows (hi=256B, lo=128B), V rows full width
        est_lo_k = k_hi_total * hi_row + k_lo_total * lo_row
        emit(f"  sum over {len(kvs)} groups: K union tokens={k_union_total} (hi={k_hi_total} lo={k_lo_total})  V union tokens={v_union_total}")
        emit(f"  (a) full-width K+V union rows            = {est_full/1e6:.2f} MB  (contract-width upper bound)")
        emit(f"  (b) lo-aware K union rows (hi 256B/lo 128B) = {est_lo_k/1e6:.2f} MB")
        # pull the replay oracle component bytes for direct comparison
        rp = None
        rpath = os.path.join(os.path.dirname(args.epoch_trace_dir.rstrip('/')), "epoch_replay_20260710", "replay", "replay_sweep.csv")
        if os.path.exists(rpath):
            for rr in csv.DictReader(open(rpath)):
                if int(rr["qidx"]) == 287 and rr["order"] == "head_serial" and rr["window"] == "unlimited":
                    rp = rr
                    break
        if rp:
            emit(f"  replay oracle components (head_serial/unlimited):")
            emit(f"     bytes_k_rows   = {int(rp['bytes_k_rows'])/1e6:6.2f} MB   (vs my lo-aware K rows {est_lo_k/1e6:.2f} MB)")
            emit(f"     bytes_k_scale  = {int(rp['bytes_k_scale'])/1e6:6.2f} MB")
            emit(f"     bytes_v_rows   = {int(rp['bytes_v_rows'])/1e6:6.2f} MB   (vs my V union rows {v_union_total*hd*CONTRACT['value_bytes']/1e6:.2f} MB)")
            emit(f"     bytes_v_sidecar= {int(rp['bytes_v_sidecar'])/1e6:6.2f} MB")
            emit(f"     bytes_scan     = {int(rp['bytes_scan'])/1e6:6.2f} MB")
            emit(f"     physical_bytes = {int(rp['physical_bytes'])/1e6:6.2f} MB (= 174.1 MB issue oracle)")
        emit("  NOTE (accounting bases): the replay physical model adds K scale sidecars, V-PQ")
        emit("  sidecars, and the selector scan stream on top of the exact-load rows, and applies the")
        emit("  frozen hi/lo precision tiers per row. Token-count reconciliation (gate 2 above) is the")
        emit("  exact check; row-byte agreement is expected only against bytes_k_rows/bytes_v_rows,")
        emit("  and only after the lo-plane discount -- documented here, not forced.")

    if args.out:
        with open(args.out, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"\n[wrote table -> {args.out}]")


if __name__ == "__main__":
    main()
