#!/usr/bin/env python3
"""Compare LongBench-v2 prediction JSONL files by example id."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _pred_path(path: Path) -> Path:
    if path.is_dir():
        return path / "predictions.jsonl"
    return path


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _run_arg(value: str) -> tuple[str, Path]:
    if ":" not in value:
        path = Path(value)
        return path.parent.name if path.name == "predictions.jsonl" else path.name, path
    label, raw_path = value.split(":", 1)
    if not label:
        raise argparse.ArgumentTypeError("run label must be non-empty")
    return label, Path(raw_path)


def _accuracy(rows: list[dict[str, Any]]) -> tuple[int, int]:
    return sum(bool(row.get("judge")) for row in rows), len(rows)


def _id(row: dict[str, Any]) -> str:
    value = row.get("_id")
    if value is None:
        value = row.get("id")
    if value is None:
        raise ValueError("prediction row is missing _id/id")
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        type=_run_arg,
        required=True,
        help="Run as LABEL:DIR_OR_PREDICTIONS_JSONL. Provide at least two.",
    )
    parser.add_argument(
        "--ids-from",
        default=None,
        help="Restrict comparisons to ids present in this run label.",
    )
    parser.add_argument("--max-diffs", type=int, default=20)
    args = parser.parse_args()

    if len(args.run) < 2:
        raise SystemExit("provide at least two --run entries")

    rows_by_label: dict[str, list[dict[str, Any]]] = {}
    by_id: dict[str, dict[str, dict[str, Any]]] = {}
    for label, raw_path in args.run:
        path = _pred_path(raw_path)
        rows = _load_jsonl(path)
        rows_by_label[label] = rows
        by_id[label] = {_id(row): row for row in rows}
        correct, total = _accuracy(rows)
        print(f"{label}: {correct}/{total} accuracy ({100.0 * correct / max(1, total):.2f}%)")

    if args.ids_from is not None:
        if args.ids_from not in rows_by_label:
            raise SystemExit(f"--ids-from label not found: {args.ids_from}")
        ids = [_id(row) for row in rows_by_label[args.ids_from]]
    else:
        common = set.intersection(*(set(mapping) for mapping in by_id.values()))
        ids = [row_id for row_id in (_id(row) for row in next(iter(rows_by_label.values()))) if row_id in common]

    print(f"comparison_ids: {len(ids)}")
    labels = list(rows_by_label)
    for i, left in enumerate(labels):
        for right in labels[i + 1 :]:
            missing_left = [row_id for row_id in ids if row_id not in by_id[left]]
            missing_right = [row_id for row_id in ids if row_id not in by_id[right]]
            if missing_left or missing_right:
                print(
                    f"{left} vs {right}: missing ids left={len(missing_left)} right={len(missing_right)}"
                )
                continue
            pred_same = 0
            response_same = 0
            judge_same = 0
            diffs: list[tuple[str, Any, Any, Any, Any]] = []
            for row_id in ids:
                a = by_id[left][row_id]
                b = by_id[right][row_id]
                if a.get("pred") == b.get("pred"):
                    pred_same += 1
                if a.get("response") == b.get("response"):
                    response_same += 1
                if bool(a.get("judge")) == bool(b.get("judge")):
                    judge_same += 1
                if a.get("pred") != b.get("pred") or bool(a.get("judge")) != bool(b.get("judge")):
                    diffs.append((row_id, a.get("pred"), b.get("pred"), a.get("judge"), b.get("judge")))
            total = max(1, len(ids))
            print(
                f"{left} vs {right}: pred {pred_same}/{total}, "
                f"response {response_same}/{total}, judge {judge_same}/{total}"
            )
            for row_id, pred_a, pred_b, judge_a, judge_b in diffs[: max(0, int(args.max_diffs))]:
                print(f"  diff {row_id}: {left} pred={pred_a} judge={judge_a}; {right} pred={pred_b} judge={judge_b}")


if __name__ == "__main__":
    main()
