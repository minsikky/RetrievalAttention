#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.selector_eval.costs.base import kv_read_bytes
from benchmark.selector_eval.data.trace import attention_probs, load_trace, static_tokens


def parse_csv_ints(text: str) -> list[int]:
    return [int(part) for part in str(text).split(",") if part.strip()]


def parse_csv_floats(text: str) -> list[float]:
    return [float(part) for part in str(text).split(",") if part.strip()]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)


def output_metrics(dense_out: np.ndarray, approx_out: np.ndarray) -> tuple[float, float]:
    dense = dense_out.astype(np.float64, copy=False)
    approx = approx_out.astype(np.float64, copy=False)
    cos_denom = max(float(np.linalg.norm(dense) * np.linalg.norm(approx)), 1e-20)
    l2_denom = max(float(np.linalg.norm(dense)), 1e-20)
    return float(np.dot(dense, approx) / cos_denom), float(np.linalg.norm(dense - approx) / l2_denom)


def sparse_output(scores: np.ndarray, values: np.ndarray, selected: np.ndarray) -> np.ndarray:
    if selected.size == 0:
        return np.zeros((values.shape[-1],), dtype=np.float64)
    logits = scores[selected].astype(np.float64)
    weights = np.exp(logits - float(np.max(logits)))
    denom = max(float(weights.sum()), 1e-20)
    return (weights @ values[selected].astype(np.float64, copy=False)) / denom


def prefix_rows(
    *,
    algorithm: str,
    decode_tokens: int,
    target_l2: float,
    order: np.ndarray,
    base: np.ndarray,
    scores: np.ndarray,
    probs: np.ndarray,
    values: np.ndarray,
    dense_out: np.ndarray,
    head_dim: int,
    key_bytes: int,
    value_bytes: int,
    budgets: list[int],
) -> list[dict]:
    rows = []
    base_set = set(int(x) for x in base.tolist())
    ordered = [int(tok) for tok in order.tolist() if int(tok) not in base_set]
    for budget in budgets:
        dynamic = np.asarray(ordered[: max(0, int(budget))], dtype=np.int64)
        selected = np.unique(np.concatenate([base, dynamic])).astype(np.int64)
        approx = sparse_output(scores, values, selected)
        cos, rel_l2 = output_metrics(dense_out, approx)
        rows.append(
            {
                "algorithm": algorithm,
                "decode_length": int(decode_tokens),
                "target_l2": float(target_l2),
                "budget_dynamic_tokens": int(budget),
                "selected_tokens": int(selected.size),
                "attention_mass": float(probs[selected].sum()) if selected.size else 0.0,
                "output_cosine": cos,
                "output_relative_L2": rel_l2,
                "exact_KV_MB": kv_read_bytes(selected.size, head_dim, key_bytes, value_bytes) / (1024.0 * 1024.0),
                "reached": bool(rel_l2 <= float(target_l2)),
            }
        )
    return rows


def first_reached(rows: list[dict], target_l2: float) -> dict | None:
    reached = [row for row in rows if row["output_relative_L2"] <= float(target_l2)]
    if not reached:
        return None
    return min(reached, key=lambda row: (row["selected_tokens"], row["exact_KV_MB"]))


def contribution_order(probs: np.ndarray, values: np.ndarray, dense_out: np.ndarray) -> np.ndarray:
    diff = values.astype(np.float32, copy=False) - dense_out.astype(np.float32, copy=False)[None, :]
    score = probs.astype(np.float32, copy=False) * np.linalg.norm(diff, axis=1)
    return np.argsort(-score, kind="stable").astype(np.int64)


def top_prob_order(probs: np.ndarray) -> np.ndarray:
    return np.argsort(-probs, kind="stable").astype(np.int64)


def greedy_batch_order(
    *,
    scores: np.ndarray,
    probs: np.ndarray,
    values: np.ndarray,
    dense_out: np.ndarray,
    base: np.ndarray,
    candidate_pool: int,
    rounds: int,
    batch_size: int,
) -> np.ndarray:
    # Candidate pool is oracle/diagnostic; it bounds offline cost while still
    # testing whether value-aware greedy can beat one-shot rankings.
    initial = contribution_order(probs, values, dense_out)
    base_set = set(int(x) for x in base.tolist())
    candidates = [int(tok) for tok in initial.tolist() if int(tok) not in base_set][: int(candidate_pool)]
    remaining = np.asarray(candidates, dtype=np.int64)
    selected = list(int(x) for x in base.tolist())
    ordered_added: list[int] = []
    w_all = np.exp(scores.astype(np.float64) - float(np.max(scores)))
    numerator = (w_all[base].astype(np.float64) @ values[base].astype(np.float64, copy=False)) if base.size else np.zeros_like(dense_out, dtype=np.float64)
    denom = float(w_all[base].sum()) if base.size else 0.0
    for _round in range(max(0, int(rounds))):
        if remaining.size == 0:
            break
        w = w_all[remaining].astype(np.float64, copy=False)
        den = denom + w
        cand_out = (numerator[None, :] + w[:, None] * values[remaining].astype(np.float64, copy=False)) / np.maximum(
            den[:, None], 1e-20
        )
        err = np.linalg.norm(cand_out - dense_out.astype(np.float64, copy=False)[None, :], axis=1)
        take_count = min(int(batch_size), int(remaining.size))
        local_take = np.argpartition(err, take_count - 1)[:take_count]
        local_take = local_take[np.argsort(err[local_take], kind="stable")]
        taken = remaining[local_take].astype(np.int64)
        ordered_added.extend(int(tok) for tok in taken.tolist())
        selected.extend(int(tok) for tok in taken.tolist())
        numerator = (w_all[np.asarray(selected, dtype=np.int64)] @ values[np.asarray(selected, dtype=np.int64)].astype(np.float64, copy=False))
        denom = float(w_all[np.asarray(selected, dtype=np.int64)].sum())
        keep_mask = np.ones((remaining.size,), dtype=bool)
        keep_mask[local_take] = False
        remaining = remaining[keep_mask]
    # Append rest of pool by contribution so prefix budgets remain defined.
    ordered_added.extend(int(tok) for tok in remaining.tolist())
    return np.asarray(ordered_added, dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline relL2 oracle diagnostics for selector-eval traces.")
    parser.add_argument("--trace", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--decode_lengths", required=True)
    parser.add_argument("--targets", default="0.031111,0.02,0.01")
    parser.add_argument("--heads", default="0")
    parser.add_argument("--static_prefix", type=int, default=128)
    parser.add_argument("--static_suffix", type=int, default=128)
    parser.add_argument("--budgets", default="512,1024,2048,4096,8192,12288,16384,24576,32768,49152,65536")
    parser.add_argument("--key_bytes", type=int, default=2)
    parser.add_argument("--value_bytes", type=int, default=2)
    parser.add_argument("--greedy_decodes", default="128000")
    parser.add_argument("--greedy_candidate_pool", type=int, default=32768)
    parser.add_argument("--greedy_rounds", type=int, default=64)
    parser.add_argument("--greedy_batch_size", type=int, default=256)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True), encoding="utf-8")

    trace = load_trace(args.trace)
    decode_lengths = parse_csv_ints(args.decode_lengths)
    targets = parse_csv_floats(args.targets)
    heads = parse_csv_ints(args.heads)
    budgets = sorted(set(parse_csv_ints(args.budgets)))
    greedy_decodes = set(parse_csv_ints(args.greedy_decodes))
    q_indices = trace.q_indices_for_decodes(decode_lengths)

    sample_rows: list[dict] = []
    frontier_rows: list[dict] = []
    for qidx in q_indices:
        position = int(trace.positions[int(qidx)])
        decode_tokens = trace.decode_tokens_for_qidx(int(qidx))
        for head in heads:
            kv_head = trace.kv_head_for(int(head))
            keys = trace.keys[kv_head, : position + 1].astype(np.float32, copy=False)
            values = trace.values[kv_head, : position + 1].astype(np.float32, copy=False)
            query = trace.queries[int(head), int(qidx)].astype(np.float32, copy=False)
            scores, probs = attention_probs(keys, query)
            dense_out = probs.astype(np.float64) @ values.astype(np.float64, copy=False)
            base = np.asarray(static_tokens(position, args.static_prefix, args.static_suffix), dtype=np.int64)

            rankings = {
                "rel_l2_contribution_oracle": contribution_order(probs, values, dense_out),
                "top_prob_oracle": top_prob_order(probs),
            }
            if int(decode_tokens) in greedy_decodes:
                rankings["rel_l2_greedy_batch_oracle"] = greedy_batch_order(
                    scores=scores,
                    probs=probs,
                    values=values,
                    dense_out=dense_out,
                    base=base,
                    candidate_pool=int(args.greedy_candidate_pool),
                    rounds=int(args.greedy_rounds),
                    batch_size=int(args.greedy_batch_size),
                )
            for algorithm, order in rankings.items():
                rows = prefix_rows(
                    algorithm=algorithm,
                    decode_tokens=int(decode_tokens),
                    target_l2=float(targets[0]),
                    order=order,
                    base=base,
                    scores=scores,
                    probs=probs,
                    values=values,
                    dense_out=dense_out,
                    head_dim=int(trace.head_dim),
                    key_bytes=int(args.key_bytes),
                    value_bytes=int(args.value_bytes),
                    budgets=budgets,
                )
                for row in rows:
                    row.update({"qidx": int(qidx), "head": int(head), "kv_head": int(kv_head)})
                sample_rows.extend(rows)
                for target in targets:
                    target_rows = [dict(row, target_l2=float(target), reached=row["output_relative_L2"] <= float(target)) for row in rows]
                    best = first_reached(target_rows, float(target))
                    if best is None:
                        best = min(target_rows, key=lambda row: row["output_relative_L2"])
                        best["reached"] = False
                    frontier_rows.append(best)

    write_csv(out_dir / "samples.csv", sample_rows)
    (out_dir / "samples.json").write_text(json.dumps(sample_rows, indent=2, sort_keys=True), encoding="utf-8")
    write_csv(out_dir / "frontier.csv", frontier_rows)
    (out_dir / "frontier.json").write_text(json.dumps(frontier_rows, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[rel_l2_oracle] wrote {out_dir}")


if __name__ == "__main__":
    main()
