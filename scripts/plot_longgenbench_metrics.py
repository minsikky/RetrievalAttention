#!/usr/bin/env python3
"""Plot LongGenBench SGT dense/frontier metrics from pair-audit JSON."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class MetricSpec:
    key: str
    label: str
    dense_field: str
    frontier_field: str


METRICS = (
    MetricSpec("once", "Once", "longgen_once_pct", "longgen_once_pct"),
    MetricSpec("range", "Range", "longgen_range_pct", "longgen_range_pct"),
    MetricSpec("periodic", "Periodic", "longgen_periodic_pct", "longgen_periodic_pct"),
    MetricSpec("completion", "Completion", "longgen_completion_pct", "longgen_completion_pct"),
)

SUITES = (
    ("longgenbench_sgt_short_", "SGT short"),
    ("longgenbench_sgt_long_", "SGT long"),
)


def _num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _load_pairs(paths: Iterable[Path]) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        pairs.extend(payload.get("pairs", []))
    return pairs


def _weight(pair: dict[str, Any]) -> float:
    dense = pair.get("dense") or {}
    frontier = pair.get("frontier") or {}
    return max(1.0, _num(frontier.get("examples")) or _num(dense.get("examples")) or 1.0)


def _weighted(rows: list[dict[str, Any]], side: str, field: str) -> float | None:
    vals: list[tuple[float, float]] = []
    for row in rows:
        value = _num((row.get(side) or {}).get(field))
        if value is not None:
            vals.append((_weight(row), value))
    if not vals:
        return None
    total = sum(weight for weight, _ in vals)
    return sum(weight * value for weight, value in vals) / total


def _complete_longgen_pair(pair: dict[str, Any]) -> bool:
    dense = pair.get("dense") or {}
    frontier = pair.get("frontier") or {}
    return (
        bool(dense)
        and bool(frontier)
        and _num(dense.get("dense_step_mb")) is not None
        and _num(frontier.get("step_mb")) is not None
        and any(_num(dense.get(metric.dense_field)) is not None for metric in METRICS)
        and any(_num(frontier.get(metric.frontier_field)) is not None for metric in METRICS)
    )


def build_rows(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for prefix, suite_label in SUITES:
        suite_pairs = [
            pair
            for pair in pairs
            if str(pair.get("key") or "").startswith(prefix) and _complete_longgen_pair(pair)
        ]
        if not suite_pairs:
            continue

        dense_mb = _weighted(suite_pairs, "dense", "dense_step_mb")
        logical_mb = _weighted(suite_pairs, "frontier", "step_mb")
        physical_mb = _weighted(suite_pairs, "frontier", "physical_step_mb")
        active_fraction = _weighted(suite_pairs, "frontier", "active_fraction")
        examples = sum(_weight(pair) for pair in suite_pairs)
        generated = _weighted(suite_pairs, "dense", "generated_tokens")

        for metric in METRICS:
            dense_value = _weighted(suite_pairs, "dense", metric.dense_field)
            frontier_value = _weighted(suite_pairs, "frontier", metric.frontier_field)
            if dense_value is None and frontier_value is None:
                continue
            rows.append(
                {
                    "suite": suite_label,
                    "metric": metric.key,
                    "metric_label": metric.label,
                    "pairs": len(suite_pairs),
                    "examples": examples,
                    "generated_tokens": generated,
                    "dense_pct": dense_value,
                    "frontier_pct": frontier_value,
                    "delta_pct": None
                    if dense_value is None or frontier_value is None
                    else frontier_value - dense_value,
                    "dense_mb_per_head_query": dense_mb,
                    "frontier_logical_mb_per_head_query": logical_mb,
                    "frontier_physical_mb_per_head_query": physical_mb,
                    "logical_mb_savings_pct": None
                    if dense_mb is None or logical_mb is None or dense_mb == 0
                    else 100.0 * (1.0 - logical_mb / dense_mb),
                    "physical_mb_savings_pct": None
                    if dense_mb is None or physical_mb is None or dense_mb == 0
                    else 100.0 * (1.0 - physical_mb / dense_mb),
                    "active_fraction_pct": None if active_fraction is None else 100.0 * active_fraction,
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "suite",
        "metric",
        "metric_label",
        "pairs",
        "examples",
        "generated_tokens",
        "dense_pct",
        "frontier_pct",
        "delta_pct",
        "dense_mb_per_head_query",
        "frontier_logical_mb_per_head_query",
        "frontier_physical_mb_per_head_query",
        "logical_mb_savings_pct",
        "physical_mb_savings_pct",
        "active_fraction_pct",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: Any, digits: int = 2) -> str:
    number = _num(value)
    if number is None:
        return ""
    return f"{number:.{digits}f}"


def write_markdown(path: Path, rows: list[dict[str, Any]], figure_path: Path) -> None:
    lines = [
        "# LongGenBench SGT Metrics",
        "",
        f"Figure: `{figure_path}`",
        "",
        "Caveat: these are substring-smoke metrics from the local SGT scorer, not the official LLM-judge evaluation.",
        "",
        "| suite | metric | examples | dense % | frontier % | delta | logical MB savings % | physical MB savings % | active % |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {suite} | {metric} | {examples:.0f} | {dense} | {frontier} | {delta} | {logical} | {physical} | {active} |".format(
                suite=row["suite"],
                metric=row["metric_label"],
                examples=row["examples"],
                dense=_fmt(row["dense_pct"]),
                frontier=_fmt(row["frontier_pct"]),
                delta=_fmt(row["delta_pct"]),
                logical=_fmt(row["logical_mb_savings_pct"], 1),
                physical=_fmt(row["physical_mb_savings_pct"], 1),
                active=_fmt(row["active_fraction_pct"], 1),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot(path: Path, rows: list[dict[str, Any]], title: str) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    suites = [label for _, label in SUITES if any(row["suite"] == label for row in rows)]
    fig, axes = plt.subplots(1, len(suites), figsize=(6.6 * len(suites), 5.2), dpi=180, sharey=True)
    if len(suites) == 1:
        axes = [axes]

    for ax, suite in zip(axes, suites, strict=False):
        suite_rows = [row for row in rows if row["suite"] == suite]
        labels = [row["metric_label"] for row in suite_rows]
        x = np.arange(len(suite_rows))
        width = 0.36
        dense = [row["dense_pct"] if row["dense_pct"] is not None else 0.0 for row in suite_rows]
        frontier = [row["frontier_pct"] if row["frontier_pct"] is not None else 0.0 for row in suite_rows]
        dense_bars = ax.bar(x - width / 2, dense, width, label="Dense", color="#2f6f9f")
        frontier_bars = ax.bar(x + width / 2, frontier, width, label="Frontier", color="#d9822b")

        ax.set_title(suite)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylim(0, 105)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)

        for bars in (dense_bars, frontier_bars):
            for bar in bars:
                value = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 1.0,
                    f"{value:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        representative = suite_rows[0]
        logical = _fmt(representative["logical_mb_savings_pct"], 1)
        physical = _fmt(representative["physical_mb_savings_pct"], 1)
        active = _fmt(representative["active_fraction_pct"], 1)
        generated = _fmt(representative["generated_tokens"], 0)
        ax.text(
            0.02,
            0.98,
            f"logical savings {logical}%\nphysical savings {physical}%\nactive {active}%\ngen {generated} tok",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9},
        )

    axes[0].set_ylabel("Metric value (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.965))
    fig.suptitle(title, y=1.02, fontsize=15, fontweight="normal")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    if path.suffix.lower() != ".pdf":
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-json", action="append", type=Path, required=True)
    parser.add_argument("--output-png", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--title", default="LongGenBench SGT: Dense vs Frontier")
    args = parser.parse_args()

    rows = build_rows(_load_pairs(args.audit_json))
    if not rows:
        raise SystemExit("no completed LongGenBench SGT pairs found")
    write_csv(args.output_csv, rows)
    write_markdown(args.output_md, rows, args.output_png)
    plot(args.output_png, rows, args.title)


if __name__ == "__main__":
    main()
