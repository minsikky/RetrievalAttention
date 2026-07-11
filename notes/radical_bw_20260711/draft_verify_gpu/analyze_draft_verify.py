#!/usr/bin/env python3
"""Token-level acceptance analysis for the draft-then-verify GPU experiment.

Compares each draft arm's greedy token stream against the frozen (DRAFT_MODE=off)
arm on identical samples. Greedy decoding is deterministic, so we recover the
token stream by re-tokenizing each arm's stored ``pred`` text with the model
tokenizer; systematic detokenize/retokenize artifacts cancel because both arms
are treated identically.

Metrics per (mode x ctx):
  - per-token agreement rate (over aligned positions from 0)
  - first-divergence position per sample (and whether the answer region passed)
  - agreement run-length distribution
  - acceptance@k for k in {4, 8, 16}: fraction of length-k sliding windows that
    match fully (this is the speculative-decoding acceptance notion)
  - expected accepted-prefix length at k=8: mean over length-8 windows of the
    number of leading tokens that match before the first mismatch (0..8)
  - standalone draft correctness: does the draft pred contain the gold answer
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from transformers import AutoTokenizer


def load_rows(run_dir: Path, task: str) -> dict[int, dict]:
    path = run_dir / "pred" / f"{task}.jsonl"
    rows = {}
    for line in path.open():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        rows[int(row["index"])] = row
    return rows


def gold_answers(row: dict) -> list[str]:
    outs = row.get("outputs") or []
    if isinstance(outs, str):
        outs = [outs]
    return [str(o) for o in outs if str(o).strip()]


def answer_token_position(tok_ids: list[int], pred_text: str, golds: list[str], tokenizer) -> int | None:
    """Approx token index at which the (first) gold answer completes in the stream.

    Returns None if no gold string is found in the decoded prefix.
    """
    low = pred_text.lower()
    best = None
    for g in golds:
        gl = g.lower()
        pos = low.find(gl)
        if pos < 0:
            continue
        # character end of the gold occurrence -> approximate token index by
        # tokenizing the text prefix up to that char end.
        char_end = pos + len(gl)
        prefix = pred_text[:char_end]
        n_tok = len(tokenizer(prefix, add_special_tokens=False)["input_ids"])
        best = n_tok if best is None else min(best, n_tok)
    return best


def contains_gold(pred_text: str, golds: list[str]) -> bool:
    low = pred_text.lower()
    return any(g.lower() in low for g in golds if g.strip())


def run_lengths(matches: list[bool]) -> list[int]:
    runs = []
    cur = 0
    for m in matches:
        if m:
            cur += 1
        else:
            if cur:
                runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    return runs


def acceptance_at_k(matches: list[bool], k: int) -> float:
    n = len(matches)
    if n < k:
        return float("nan")
    windows = n - k + 1
    good = 0
    for i in range(windows):
        if all(matches[i : i + k]):
            good += 1
    return good / windows


def expected_accepted_prefix(matches: list[bool], k: int) -> float:
    """Mean over length-k windows of leading-match count before first mismatch."""
    n = len(matches)
    if n < k:
        return float("nan")
    windows = n - k + 1
    total = 0
    for i in range(windows):
        c = 0
        for j in range(k):
            if matches[i + j]:
                c += 1
            else:
                break
        total += c
    return total / windows


def analyze_pair(frozen_rows, draft_rows, task, tokenizer):
    per_sample = []
    all_matches: list[bool] = []
    for idx in sorted(set(frozen_rows) & set(draft_rows)):
        fz = frozen_rows[idx]
        dr = draft_rows[idx]
        fz_ids = tokenizer(fz["pred"], add_special_tokens=False)["input_ids"]
        dr_ids = tokenizer(dr["pred"], add_special_tokens=False)["input_ids"]
        n = min(len(fz_ids), len(dr_ids))
        matches = [fz_ids[i] == dr_ids[i] for i in range(n)]
        first_div = next((i for i, m in enumerate(matches) if not m), None)
        if first_div is None and len(fz_ids) != len(dr_ids):
            first_div = n  # streams identical up to n, then a length divergence
        golds = gold_answers(fz)
        ans_pos = answer_token_position(fz_ids, fz["pred"], golds, tokenizer)
        answer_passed_at_div = (
            None if first_div is None or ans_pos is None else bool(first_div >= ans_pos)
        )
        per_sample.append(
            {
                "index": idx,
                "frozen_len": len(fz_ids),
                "draft_len": len(dr_ids),
                "aligned_len": n,
                "agreement_rate": (sum(matches) / n) if n else float("nan"),
                "first_divergence": first_div,
                "answer_token_pos": ans_pos,
                "answer_region_passed_before_fork": answer_passed_at_div,
                "run_lengths": run_lengths(matches),
                "frozen_correct": contains_gold(fz["pred"], golds),
                "draft_correct": contains_gold(dr["pred"], golds),
                "golds": golds,
                "frozen_pred": fz["pred"],
                "draft_pred": dr["pred"],
            }
        )
        all_matches.extend(matches)

    agg = {
        "n_samples": len(per_sample),
        "pooled_aligned_tokens": len(all_matches),
        "pooled_agreement_rate": (sum(all_matches) / len(all_matches)) if all_matches else float("nan"),
        "acceptance_at_k": {str(k): acceptance_at_k(all_matches, k) for k in (4, 8, 16)},
        "expected_accepted_prefix_k8": expected_accepted_prefix(all_matches, 8),
        "mean_first_divergence": (
            statistics.mean(
                [s["first_divergence"] for s in per_sample if s["first_divergence"] is not None]
            )
            if any(s["first_divergence"] is not None for s in per_sample)
            else None
        ),
        "n_fully_identical_streams": sum(
            1 for s in per_sample if s["first_divergence"] is None
        ),
        "draft_standalone_correct": sum(1 for s in per_sample if s["draft_correct"]),
        "frozen_correct": sum(1 for s in per_sample if s["frozen_correct"]),
    }
    return {"aggregate": agg, "per_sample": per_sample}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_root", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    root = Path(args.output_root)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # (label, ctx, task, frozen_run, draft_run)
    arms = [
        ("32k_start1", "32k", "qa_1", "qa1_32k_n4_off", "qa1_32k_n4_start1"),
        ("32k_start2", "32k", "qa_1", "qa1_32k_n4_off", "qa1_32k_n4_start2"),
        ("128k_start1", "128k", "niah_multikey_3", "mk3_128k_n2_off", "mk3_128k_n2_start1"),
        ("128k_start2", "128k", "niah_multikey_3", "mk3_128k_n2_off", "mk3_128k_n2_start2"),
    ]

    results = {}
    for label, ctx, task, frozen_run, draft_run in arms:
        fz_dir = root / frozen_run
        dr_dir = root / draft_run
        if not (fz_dir / "pred" / f"{task}.jsonl").exists():
            results[label] = {"skipped": f"missing frozen {frozen_run}"}
            continue
        if not (dr_dir / "pred" / f"{task}.jsonl").exists():
            results[label] = {"skipped": f"missing draft {draft_run}"}
            continue
        frozen_rows = load_rows(fz_dir, task)
        draft_rows = load_rows(dr_dir, task)
        results[label] = {
            "ctx": ctx,
            "task": task,
            "frozen_run": frozen_run,
            "draft_run": draft_run,
            **analyze_pair(frozen_rows, draft_rows, task, tokenizer),
        }

    Path(args.out_json).write_text(json.dumps(results, indent=2))

    # compact console table
    print(f"{'arm':<14} {'agree':>7} {'acc@4':>7} {'acc@8':>7} {'acc@16':>7} {'eap@8':>7} {'draftOK':>8}")
    for label, r in results.items():
        if "aggregate" not in r:
            print(f"{label:<14} {r.get('skipped','?')}")
            continue
        a = r["aggregate"]
        acc = a["acceptance_at_k"]
        print(
            f"{label:<14} {a['pooled_agreement_rate']:>7.3f} "
            f"{acc['4']:>7.3f} {acc['8']:>7.3f} {acc['16']:>7.3f} "
            f"{a['expected_accepted_prefix_k8']:>7.3f} "
            f"{a['draft_standalone_correct']}/{a['n_samples']:<6}"
        )


if __name__ == "__main__":
    main()
