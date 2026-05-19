#!/usr/bin/env python3
"""Report LongBench-v2 task changes alongside dense-reference drift diagnostics."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _id(row: dict[str, Any]) -> str:
    value = row.get("_id", row.get("id"))
    if value is None:
        raise ValueError("prediction row missing _id/id")
    return str(value)


def _pred_path(path: Path) -> Path:
    return path / "predictions.jsonl" if path.is_dir() else path


def _load_diagnostics(patterns: list[str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for pattern in patterns:
        for raw_path in glob.glob(pattern):
            path = Path(raw_path)
            if path.is_dir():
                path = path / "summary.json"
            if not path.exists():
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            row_id = payload.get("id_filter")
            if not row_id:
                continue
            out[str(row_id)] = payload
    return out


def _fmt(value: Any, digits: int = 4) -> str:
    if isinstance(value, bool) or value is None:
        return "n/a"
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _status(dense_correct: bool, approx_correct: bool, pred_changed: bool) -> str:
    if dense_correct and approx_correct:
        return "preserved_correct" if not pred_changed else "changed_still_correct"
    if (not dense_correct) and (not approx_correct):
        return "preserved_wrong" if not pred_changed else "changed_still_wrong"
    if dense_correct and not approx_correct:
        return "lost_correct"
    return "gained_correct"


def _markdown(rows: list[dict[str, Any]]) -> str:
    headers = [
        "id",
        "status",
        "dense",
        "frontier",
        "mean logit relL2",
        "max logit relL2",
        "min logit cos",
        "mean hidden relL2",
        "max hidden relL2",
        "min hidden cos",
        "KL mean/max",
        "top1",
        "choice top",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        diag = row.get("diag") or {}
        lines.append(
            "| "
            + " | ".join(
                [
                    row["id"],
                    row["status"],
                    f"{row['dense_pred']} / {row['dense_correct']}",
                    f"{row['frontier_pred']} / {row['frontier_correct']}",
                    _fmt(diag.get("mean_logit_relative_l2")),
                    _fmt(diag.get("max_logit_relative_l2")),
                    _fmt(diag.get("min_logit_cosine")),
                    _fmt(diag.get("mean_hidden_relative_l2")),
                    _fmt(diag.get("max_hidden_relative_l2")),
                    _fmt(diag.get("min_hidden_cosine")),
                    f"{_fmt(diag.get('mean_dense_to_approx_kl'))}/{_fmt(diag.get('max_dense_to_approx_kl'))}",
                    _fmt(diag.get("top1_agreement")),
                    _fmt(diag.get("choice_top_agreement")),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense", required=True, type=Path, help="Dense run dir or predictions.jsonl")
    parser.add_argument("--frontier", required=True, type=Path, help="Frontier run dir or predictions.jsonl")
    parser.add_argument(
        "--diag-glob",
        action="append",
        default=[],
        help="Glob for diagnostic summary.json files or diagnostic dirs. Can be repeated.",
    )
    parser.add_argument("--changed-only", action="store_true")
    parser.add_argument("--diagnosed-only", action="store_true")
    parser.add_argument("--max-rows", type=int, default=100)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    dense_rows = {_id(row): row for row in _load_jsonl(_pred_path(args.dense))}
    frontier_rows = {_id(row): row for row in _load_jsonl(_pred_path(args.frontier))}
    diagnostics = _load_diagnostics(args.diag_glob)

    common_ids = [row_id for row_id in dense_rows if row_id in frontier_rows]
    rows: list[dict[str, Any]] = []
    for row_id in common_ids:
        dense = dense_rows[row_id]
        frontier = frontier_rows[row_id]
        dense_correct = bool(dense.get("judge"))
        frontier_correct = bool(frontier.get("judge"))
        pred_changed = dense.get("pred") != frontier.get("pred")
        diag = diagnostics.get(row_id)
        if args.changed_only and not (pred_changed or dense_correct != frontier_correct):
            continue
        if args.diagnosed_only and diag is None:
            continue
        rows.append(
            {
                "id": row_id,
                "status": _status(dense_correct, frontier_correct, pred_changed),
                "dense_pred": dense.get("pred"),
                "frontier_pred": frontier.get("pred"),
                "dense_correct": dense_correct,
                "frontier_correct": frontier_correct,
                "diag": diag,
            }
        )

    status_counts: dict[str, int] = {}
    for row in rows:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1

    rows = rows[: max(0, int(args.max_rows))]
    lines = ["# LongBench Drift Report", ""]
    lines.append(
        f"Compared `{_pred_path(args.dense)}` vs `{_pred_path(args.frontier)}` over `{len(common_ids)}` common rows."
    )
    lines.append("")
    lines.append("Status counts in reported set: " + json.dumps(status_counts, sort_keys=True))
    lines.append("")
    lines.append(_markdown(rows))
    text = "\n".join(lines)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
