#!/usr/bin/env python3
"""Plot LiveCodeBench dense/frontier artifact progress when a shard is partial."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [_num(row.get(key)) for row in rows]
    values = [value for value in values if value is not None]
    if not values:
        return None
    return sum(values) / len(values)


def summarize(label: str, run_dir: Path, target_count: int | None) -> dict[str, Any]:
    predictions = _read_jsonl(run_dir / "predictions.jsonl")
    scored = _read_jsonl(run_dir / "scored_predictions.jsonl")
    summary = _read_json(run_dir / "summary.json")
    pass_at_1 = None
    metrics = summary.get("livecodebench_metrics")
    if isinstance(metrics, dict):
        pass_at_1 = _num(metrics.get("pass@1"))
    if pass_at_1 is None:
        pass_at_1 = _num(summary.get("pass_at_1"))
    if pass_at_1 is not None and pass_at_1 <= 1.0:
        pass_at_1 *= 100.0
    return {
        "label": label,
        "run_dir": str(run_dir),
        "target_examples": target_count,
        "prediction_rows": len(predictions),
        "scored_rows": len(scored),
        "summary_exists": bool(summary),
        "pass_at_1_pct": pass_at_1,
        "avg_generated_tokens_from_predictions": _mean(predictions, "generated_tokens"),
        "avg_generation_sec_from_predictions": _mean(predictions, "generation_sec"),
        "ids": ",".join(str(row.get("id") or "") for row in predictions),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "label",
        "run_dir",
        "target_examples",
        "prediction_rows",
        "scored_rows",
        "summary_exists",
        "pass_at_1_pct",
        "avg_generated_tokens_from_predictions",
        "avg_generation_sec_from_predictions",
        "ids",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: Any, digits: int = 2) -> str:
    number = _num(value)
    return "" if number is None else f"{number:.{digits}f}"


def write_markdown(path: Path, rows: list[dict[str, Any]], figure_path: Path) -> None:
    lines = [
        "# LiveCodeBench Partial Progress",
        "",
        f"Figure: `{figure_path}`",
        "",
        "This is a progress/availability plot, not a frontier quality plot. Frontier pass@1 is omitted until scored predictions or summary exists.",
        "",
        "| run | predictions | scored | summary | pass@1 % | avg gen tok | avg gen sec |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        target = row["target_examples"]
        pred = str(row["prediction_rows"]) if target is None else f"{row['prediction_rows']}/{target}"
        scored = str(row["scored_rows"]) if target is None else f"{row['scored_rows']}/{target}"
        lines.append(
            "| {label} | {pred} | {scored} | {summary} | {pass1} | {tokens} | {sec} |".format(
                label=row["label"],
                pred=pred,
                scored=scored,
                summary="yes" if row["summary_exists"] else "no",
                pass1=_fmt(row["pass_at_1_pct"]),
                tokens=_fmt(row["avg_generated_tokens_from_predictions"], 1),
                sec=_fmt(row["avg_generation_sec_from_predictions"], 1),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot(path: Path, rows: list[dict[str, Any]], title: str) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    labels = [row["label"] for row in rows]
    x = np.arange(len(rows))
    width = 0.28
    target = [row["target_examples"] or max(row["prediction_rows"], row["scored_rows"], 1) for row in rows]
    pred = [row["prediction_rows"] for row in rows]
    scored = [row["scored_rows"] for row in rows]
    pass1 = [row["pass_at_1_pct"] for row in rows]

    fig, (ax_count, ax_metric) = plt.subplots(1, 2, figsize=(11.5, 4.8), dpi=180)

    ax_count.bar(x - width, target, width, label="Target examples", color="#b8b8b8")
    ax_count.bar(x, pred, width, label="Predictions written", color="#2f6f9f")
    ax_count.bar(x + width, scored, width, label="Scored rows", color="#d9822b")
    ax_count.set_xticks(x)
    ax_count.set_xticklabels(labels)
    ax_count.set_ylabel("Rows")
    ax_count.set_title("Artifact completion")
    ax_count.grid(axis="y", alpha=0.25)
    ax_count.legend(frameon=False)
    for xi, row in zip(x, rows):
        ax_count.text(xi, max(target[xi], pred[xi], scored[xi]) + 0.25, "summary" if row["summary_exists"] else "no summary", ha="center", fontsize=8)

    visible_pass1 = [value if value is not None else 0.0 for value in pass1]
    bars = ax_metric.bar(x, visible_pass1, width=0.44, color="#2f8f46")
    ax_metric.set_xticks(x)
    ax_metric.set_xticklabels(labels)
    ax_metric.set_ylim(0, 105)
    ax_metric.set_ylabel("pass@1 (%)")
    ax_metric.set_title("Quality availability")
    ax_metric.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, pass1):
        if value is None:
            ax_metric.text(bar.get_x() + bar.get_width() / 2, 3.0, "unscored", ha="center", va="bottom", fontsize=9)
        else:
            ax_metric.text(bar.get_x() + bar.get_width() / 2, value + 2.0, f"{value:.1f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    if path.suffix.lower() != ".pdf":
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense-dir", type=Path, required=True)
    parser.add_argument("--frontier-dir", type=Path, required=True)
    parser.add_argument("--target-count", type=int, default=None)
    parser.add_argument("--output-png", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--title", default="LiveCodeBench Partial Progress")
    args = parser.parse_args()

    rows = [
        summarize("Dense", args.dense_dir, args.target_count),
        summarize("Frontier", args.frontier_dir, args.target_count),
    ]
    write_csv(args.output_csv, rows)
    write_markdown(args.output_md, rows, args.output_png)
    plot(args.output_png, rows, args.title)


if __name__ == "__main__":
    main()
