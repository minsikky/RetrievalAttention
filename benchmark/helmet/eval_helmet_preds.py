"""Score runner predictions with HELMET's own per-dataset metrics.

Reloads the HELMET dataset deterministically (same config + seed as
prepare_helmet_data.py), joins predictions by index, and applies the
dataset's post_process (falls back to HELMET's default_post_process).
Writes a summary json with the mean of every metric key.

Run under .venv.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from prepare_helmet_data import load_helmet_data  # same dir; also sets sys.path/cwd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--test_file", default="")
    parser.add_argument("--demo_file", default="")
    parser.add_argument("--shots", type=int, default=2)
    parser.add_argument("--max_test_samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pred_file", required=True)
    parser.add_argument("--summary_out", required=True)
    parser.add_argument(
        "--data_file",
        default="",
        help="converted validation.jsonl from prepare_helmet_data. When set, "
        "score directly against its rows (gold answers stored at conversion "
        "time) with HELMET's default_post_process instead of reloading the "
        "dataset — the HF reload is NOT order-stable across processes, so "
        "positional joins against it can silently score the wrong gold rows.",
    )
    parser.add_argument(
        "--stop_new_line",
        action="store_true",
        help="truncate predictions at the first newline (HELMET stop_new_line "
        "semantics; equivalent under greedy decoding to stopping generation)",
    )
    args = parser.parse_args()

    pred_path = Path(args.pred_file)
    if not pred_path.is_absolute():
        pred_path = Path.cwd() / pred_path
    summary_path = Path(args.summary_out)
    if not summary_path.is_absolute():
        summary_path = Path.cwd() / summary_path

    preds: dict[int, str] = {}
    with pred_path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            pred = str(row["pred"])
            if args.stop_new_line:
                pred = pred.split("\n")[0]
            preds[int(row["index"])] = pred

    if str(args.data_file):
        # Direct mode: gold answers come from the converted file itself, so
        # pred index i corresponds to row i by construction.
        import sys as _sys

        _sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "third_party" / "benchmarks" / "HELMET"))
        from data import default_post_process

        examples = []
        with open(args.data_file, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                examples.append({"answer": row["outputs"], **row.get("others", {})})
        data = {"data": examples, "post_process": default_post_process}
    else:
        data = load_helmet_data(args)
    post = data["post_process"]

    sums: dict[str, float] = defaultdict(float)
    per_sample = []
    n = 0
    for i, example in enumerate(data["data"]):
        if i not in preds:
            continue
        mets, others = post({"output": preds[i]}, example)
        for k, v in mets.items():
            sums[k] += float(v)
        per_sample.append({"index": i, **{k: float(v) for k, v in mets.items()}})
        n += 1

    if n == 0:
        raise SystemExit("no predictions matched the dataset indices")
    summary = {
        "dataset": args.dataset,
        "n_scored": n,
        "n_data": len(data["data"]),
        "metrics_mean": {k: sums[k] / n for k in sorted(sums)},
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps({**summary, "per_sample": per_sample}, indent=2) + "\n")
    print(f"[eval_helmet_preds] {args.dataset}: n={n} " + " ".join(f"{k}={v:.4f}" for k, v in summary["metrics_mean"].items()))


if __name__ == "__main__":
    main()
