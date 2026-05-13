#!/usr/bin/env python3
"""Diagnose where paged+routed PQ false negatives come from."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SIM_PATH = PROJECT_ROOT / "benchmark" / "online_ivfpq_simulator.py"


def load_simulator():
    spec = importlib.util.spec_from_file_location("online_ivfpq_simulator_local", SIM_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load simulator from {SIM_PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def parse_ints(text: str) -> list[int]:
    return [int(x) for x in str(text).split(",") if str(x).strip()]


def pmass(probs: np.ndarray, toks: set[int]) -> float:
    if not toks:
        return 0.0
    return float(probs[np.asarray(list(toks), dtype=np.int64)].sum())


def mean(rows: list[dict], key: str) -> float:
    vals = [float(row[key]) for row in rows]
    vals = [x for x in vals if not math.isnan(x)]
    return float(np.mean(vals)) if vals else float("nan")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source_npz", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--decode_tokens", type=int, required=True)
    p.add_argument("--target_mass", type=float, default=0.98)
    p.add_argument("--heads", default="", help="Comma-separated heads. Empty means all heads.")
    p.add_argument("--static_prefix", type=int, default=128)
    p.add_argument("--static_suffix", type=int, default=128)
    p.add_argument("--paged_pq_page_size", type=int, default=2048)
    p.add_argument("--paged_router_prototypes", type=int, default=16)
    p.add_argument("--paged_router_merge_rel", type=float, default=0.05)
    p.add_argument("--paged_router_merge_var", type=float, default=0.0)
    p.add_argument("--paged_router_max_groups", type=int, default=512)
    p.add_argument("--nprobes", default="1,2,4,8,16,32,64,128,256,512")
    p.add_argument("--pqcache_subvecs", type=int, default=2)
    p.add_argument("--pqcache_subbits", type=int, default=6)
    p.add_argument("--pqcache_kmeans_iters", type=int, default=3)
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--score_key_bytes_per_element", type=int, default=4)
    p.add_argument("--attn_key_bytes_per_element", type=int, default=2)
    p.add_argument("--value_bytes_per_element", type=int, default=2)
    p.add_argument("--edge_index_bytes", type=int, default=4)
    p.add_argument("--graph_offset_bytes", type=int, default=4)
    p.add_argument("--backend", choices=("auto", "python", "cpp"), default="cpp")
    p.add_argument("--backend_threads", type=int, default=8)
    args = p.parse_args()

    sim = load_simulator()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True))

    data = np.load(args.source_npz)
    keys = np.asarray(data["keys"], dtype=np.float32)
    queries = np.asarray(data["queries"], dtype=np.float32)
    positions = np.asarray(data["positions"], dtype=np.int64)
    meta = json.loads(str(data["metadata"].item())) if "metadata" in data else {}
    input_len = int(meta.get("input_len", int(positions.min()) + 1))

    num_heads, _q_count, dim = queries.shape
    kv_heads = keys.shape[0]
    args.head_dim = int(dim)
    score_scale = 1.0 / math.sqrt(float(dim))
    nprobes = parse_ints(args.nprobes)
    head_ids = parse_ints(args.heads) if str(args.heads).strip() else list(range(num_heads))

    decode_arr = np.asarray([max(0, int(pos) - input_len + 1) for pos in positions], dtype=np.int64)
    matches = np.where(decode_arr == int(args.decode_tokens))[0]
    if matches.size == 0:
        raise ValueError(f"decode length {args.decode_tokens} not found in trace")
    qidx = int(matches[0])
    pos = int(positions[qidx])

    dynamic_start = min(max(0, int(args.static_prefix)), input_len)
    init_dynamic_end = max(dynamic_start, input_len - max(0, int(args.static_suffix)))
    indexed_hi = max(dynamic_start, min(pos + 1 - max(0, int(args.static_suffix)), keys.shape[1]))

    indexes = []
    for kv_h in range(kv_heads):
        print(f"[fn_cause] build kv_head={kv_h}", flush=True)
        index = sim.PagedLocalPQIndex(
            keys=keys[kv_h],
            init_start=dynamic_start,
            init_end=init_dynamic_end,
            args=args,
            seed=int(args.seed) + 2027 * int(kv_h),
            router_enabled=True,
        )
        index.advance_to(indexed_hi)
        indexes.append(index)
        print(
            f"[fn_cause] built kv_head={kv_h} pages={len(index.pages)} groups={len(index.groups)} pending={len(index.pending_tokens())}",
            flush=True,
        )

    def code_for_token(index, tok: int):
        tok = int(tok)
        if tok < index.token_start or tok >= index.pending_start:
            return None
        page_id = (tok - index.token_start) // index.page_size
        if page_id < 0 or page_id >= len(index.pages):
            return None
        page = index.pages[int(page_id)]
        row = tok - int(page["token_start"])
        if row < 0 or row >= int(page["size"]):
            return None
        return tuple(int(x) for x in page["codes"][int(row)].tolist())

    rows = []
    for head in head_ids:
        kv_h = min(kv_heads - 1, int(head) * kv_heads // num_heads)
        index = indexes[kv_h]
        q = queries[int(head), qidx].astype(np.float32, copy=False)
        usable_keys = keys[kv_h, : pos + 1].astype(np.float32, copy=False)
        true_scores = (usable_keys @ q) * score_scale
        logits = true_scores - np.max(true_scores)
        probs = np.exp(logits).astype(np.float32)
        probs /= max(float(probs.sum()), 1e-20)

        base = sim.unique(
            sim.static_tokens(pos, int(args.static_prefix), int(args.static_suffix)),
            1_000_000,
            0,
            true_scores.shape[0],
        )
        base_set = set(base)
        static_mask = np.zeros((true_scores.shape[0],), dtype=bool)
        if base:
            static_mask[np.asarray(base, dtype=np.int64)] = True
        dynamic_ids = np.nonzero(~static_mask)[0].astype(np.int64, copy=False)
        oracle_order = dynamic_ids[np.argsort(-probs[dynamic_ids], kind="stable")]
        oracle = list(base)
        oracle_mass = float(probs[np.asarray(oracle, dtype=np.int64)].sum()) if oracle else 0.0
        cursor = 0
        while oracle_mass < float(args.target_mass) and cursor < oracle_order.size:
            tok = int(oracle_order[cursor])
            cursor += 1
            oracle.append(tok)
            oracle_mass += float(probs[tok])
        oracle_set = set(sim.unique(oracle, len(oracle), 0, true_scores.shape[0]))

        pending = [
            int(tok)
            for tok in index.pending_tokens()
            if int(tok) < true_scores.shape[0] and int(tok) not in base_set
        ]
        pending_set = set(pending)
        routed_many = index.selection_routed_many(q, nprobes)

        choices = []
        for nprobe, (raw_ranked, selection_events) in routed_many.items():
            ranked = np.asarray(
                [
                    int(tok)
                    for tok in raw_ranked.tolist()
                    if int(tok) < true_scores.shape[0] and int(tok) not in base_set and int(tok) not in pending_set
                ],
                dtype=np.int64,
            )
            selected = sim.unique(list(base) + pending, len(base) + len(pending), 0, true_scores.shape[0])
            selected_mass = float(probs[np.asarray(selected, dtype=np.int64)].sum()) if selected else 0.0
            routed_added = []
            rank_cursor = 0
            while selected_mass < float(args.target_mass) and rank_cursor < ranked.size:
                tok = int(ranked[rank_cursor])
                rank_cursor += 1
                selected.append(tok)
                routed_added.append(tok)
                selected_mass += float(probs[tok])
            selected = sim.unique(selected, len(selected), 0, true_scores.shape[0])
            exact_mb = (
                len(selected)
                * dim
                * (int(args.attn_key_bytes_per_element) + int(args.value_bytes_per_element))
                / (1024**2)
            )
            choices.append(
                {
                    "reached": selected_mass >= float(args.target_mass),
                    "total_mb": selection_events.mb() + exact_mb,
                    "selector_mb": selection_events.mb(),
                    "exact_mb": exact_mb,
                    "nprobe": int(nprobe),
                    "selected": selected,
                    "routed_added": routed_added,
                    "ranked": ranked,
                    "selected_mass": selected_mass,
                }
            )
        reachable = [choice for choice in choices if choice["reached"]]
        choice = min(reachable, key=lambda x: x["total_mb"]) if reachable else max(choices, key=lambda x: x["selected_mass"])

        selected_set = set(choice["selected"])
        routed_set = set(choice["routed_added"])
        candidate_set = set(choice["ranked"].tolist())
        false_pos = selected_set - oracle_set
        false_neg = oracle_set - selected_set
        coarse_fn = {
            tok
            for tok in false_neg
            if tok not in candidate_set and tok not in pending_set and tok not in base_set
        }
        pq_late_fn = {tok for tok in false_neg if tok in candidate_set}

        token_to_code = {}
        code_count = {}
        code_oracle = {}
        code_score_minmax = {}
        for tok in choice["ranked"].tolist():
            tok = int(tok)
            code = code_for_token(index, tok)
            if code is None:
                continue
            token_to_code[tok] = code
            code_count[code] = code_count.get(code, 0) + 1
            code_oracle[code] = code_oracle.get(code, 0) + (1 if tok in oracle_set else 0)
            if code not in code_score_minmax:
                code_score_minmax[code] = [float(true_scores[tok]), float(true_scores[tok])]
            else:
                code_score_minmax[code][0] = min(code_score_minmax[code][0], float(true_scores[tok]))
                code_score_minmax[code][1] = max(code_score_minmax[code][1], float(true_scores[tok]))

        selected_codes = {token_to_code[tok] for tok in routed_set if tok in token_to_code}
        same_code_fn = {tok for tok in pq_late_fn if token_to_code.get(tok) in selected_codes}
        cross_code_fn = pq_late_fn - same_code_fn

        mixed = 0
        span_values = []
        for code in selected_codes:
            count = int(code_count.get(code, 0))
            if count <= 1:
                continue
            oracle_count = int(code_oracle.get(code, 0))
            if 0 < oracle_count < count:
                mixed += 1
            lo, hi = code_score_minmax[code]
            span_values.append(float(hi - lo))

        rank_pct = []
        pq_pct = []
        if choice["ranked"].size and pq_late_fn:
            true_order = np.argsort(-true_scores[choice["ranked"]], kind="stable")
            true_rank = np.empty(choice["ranked"].size, dtype=np.int64)
            true_rank[true_order] = np.arange(choice["ranked"].size)
            token_to_idx = {int(tok): i for i, tok in enumerate(choice["ranked"].tolist())}
            for tok in pq_late_fn:
                idx = int(token_to_idx[int(tok)])
                pq_pct.append(idx / max(1, choice["ranked"].size - 1))
                rank_pct.append(int(true_rank[idx]) / max(1, choice["ranked"].size - 1))

        row = {
            "decode_tokens": int(args.decode_tokens),
            "head": int(head),
            "kv_head": int(kv_h),
            "groups": int(len(index.groups)),
            "pages": int(len(index.pages)),
            "pending_tokens": int(len(pending_set)),
            "nprobe": int(choice["nprobe"]),
            "reached": bool(choice["reached"]),
            "total_mb": float(choice["total_mb"]),
            "selector_mb": float(choice["selector_mb"]),
            "exact_mb": float(choice["exact_mb"]),
            "selected_tokens": int(len(selected_set)),
            "oracle_tokens": int(len(oracle_set)),
            "candidate_tokens": int(len(candidate_set)),
            "routed_added_tokens": int(len(routed_set)),
            "false_positive_tokens": int(len(false_pos)),
            "false_negative_tokens": int(len(false_neg)),
            "coarse_fn_tokens": int(len(coarse_fn)),
            "pq_late_fn_tokens": int(len(pq_late_fn)),
            "same_code_fn_tokens": int(len(same_code_fn)),
            "cross_code_fn_tokens": int(len(cross_code_fn)),
            "false_positive_mass": pmass(probs, false_pos),
            "false_negative_mass": pmass(probs, false_neg),
            "coarse_fn_mass": pmass(probs, coarse_fn),
            "pq_late_fn_mass": pmass(probs, pq_late_fn),
            "same_code_fn_mass": pmass(probs, same_code_fn),
            "cross_code_fn_mass": pmass(probs, cross_code_fn),
            "selected_mass": pmass(probs, selected_set),
            "oracle_mass": pmass(probs, oracle_set),
            "selected_code_mixed_frac": float(mixed / max(1, len(selected_codes))),
            "selected_code_score_span": float(np.mean(span_values)) if span_values else 0.0,
            "fn_true_rank_pct": float(np.mean(rank_pct)) if rank_pct else float("nan"),
            "fn_pq_rank_pct": float(np.mean(pq_pct)) if pq_pct else float("nan"),
        }
        rows.append(row)
        print(
            f"[fn_cause] head={head} nprobe={row['nprobe']} fn={row['false_negative_tokens']} "
            f"coarse={row['coarse_fn_tokens']} pq_late={row['pq_late_fn_tokens']} same_code={row['same_code_fn_tokens']}",
            flush=True,
        )

    summary = {
        "decode_tokens": int(args.decode_tokens),
        "heads": int(len(rows)),
        "target_mass": float(args.target_mass),
    }
    for key in rows[0]:
        if key in {"decode_tokens", "head", "kv_head", "reached"}:
            continue
        summary[f"{key}_mean"] = mean(rows, key)
    summary["reached_rate"] = float(np.mean([1.0 if row["reached"] else 0.0 for row in rows]))

    with (out_dir / "per_head.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    (out_dir / "per_head.json").write_text(json.dumps(rows, indent=2, sort_keys=True))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))

    report = out_dir / "report.md"
    with report.open("w", encoding="utf-8") as f:
        f.write("# False-Negative Cause Diagnostic\n\n")
        f.write(f"Decode length: {args.decode_tokens}. Target mass: {args.target_mass}.\n\n")
        f.write("| metric | mean |\n|---|---:|\n")
        for key in [
            "false_negative_tokens",
            "coarse_fn_tokens",
            "pq_late_fn_tokens",
            "same_code_fn_tokens",
            "cross_code_fn_tokens",
            "false_positive_tokens",
            "false_negative_mass",
            "coarse_fn_mass",
            "pq_late_fn_mass",
            "same_code_fn_mass",
            "cross_code_fn_mass",
            "false_positive_mass",
            "selected_code_mixed_frac",
            "selected_code_score_span",
            "fn_true_rank_pct",
            "fn_pq_rank_pct",
        ]:
            f.write(f"| {key} | {summary[f'{key}_mean']:.6g} |\n")
    print(f"[fn_cause] wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
