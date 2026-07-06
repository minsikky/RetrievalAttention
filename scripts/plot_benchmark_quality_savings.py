#!/usr/bin/env python3
"""Plot dense/frontier quality and logical bandwidth savings from pair audits."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class GroupSpec:
    name: str
    prefix: str
    note: str = ""


GROUPS = (
    GroupSpec("RULER 32K", "ruler_ctx32768_"),
    GroupSpec("RULER 64K", "ruler_ctx65536_"),
    GroupSpec("RULER 128K partial", "ruler_ctx131072_", "partial"),
    GroupSpec("LBv2 short easy", "lbv2_short_easy_"),
    GroupSpec("LBv2 short hard", "lbv2_short_hard_"),
    GroupSpec("GPQA", "gpqa_"),
)


def _num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _load_pairs(paths: Iterable[Path]) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        pairs.extend(payload.get("pairs", []))
    return pairs


def _complete_pair(row: dict[str, Any]) -> bool:
    dense = row.get("dense") or {}
    frontier = row.get("frontier") or {}
    return (
        bool(dense)
        and bool(frontier)
        and _num(dense.get("quality")) is not None
        and _num(frontier.get("quality")) is not None
        and _num(dense.get("dense_step_mb")) is not None
        and _num(frontier.get("step_mb")) is not None
    )


def _weight(row: dict[str, Any]) -> float:
    dense = row.get("dense") or {}
    frontier = row.get("frontier") or {}
    return max(1.0, _num(frontier.get("examples")) or _num(dense.get("examples")) or 1.0)


def _aggregate(name: str, note: str, rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    rows = [row for row in rows if _complete_pair(row)]
    if not rows:
        return None

    total_w = sum(_weight(row) for row in rows)

    def weighted(path: tuple[str, str]) -> float:
        return sum(_weight(row) * float((row.get(path[0]) or {}).get(path[1])) for row in rows) / total_w

    dense_quality = weighted(("dense", "quality"))
    frontier_quality = weighted(("frontier", "quality"))
    dense_mb = weighted(("dense", "dense_step_mb"))
    frontier_logical_mb = weighted(("frontier", "step_mb"))
    frontier_physical_mb = weighted(("frontier", "physical_step_mb"))

    logical_savings = 100.0 * (1.0 - frontier_logical_mb / dense_mb) if dense_mb else float("nan")
    physical_savings = 100.0 * (1.0 - frontier_physical_mb / dense_mb) if dense_mb else float("nan")

    return {
        "group": name,
        "note": note,
        "pairs": len(rows),
        "dense_quality_pct": dense_quality,
        "frontier_quality_pct": frontier_quality,
        "quality_delta_pct": frontier_quality - dense_quality,
        "dense_mb_per_head_query": dense_mb,
        "frontier_logical_mb_per_head_query": frontier_logical_mb,
        "frontier_physical_mb_per_head_query": frontier_physical_mb,
        "logical_mb_savings_pct": logical_savings,
        "physical_mb_savings_pct": physical_savings,
    }


def build_rows(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for group in GROUPS:
        matched = [row for row in pairs if str(row.get("key") or "").startswith(group.prefix)]
        item = _aggregate(group.name, group.note, matched)
        if item:
            rows.append(item)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "group",
        "note",
        "pairs",
        "dense_quality_pct",
        "frontier_quality_pct",
        "quality_delta_pct",
        "dense_mb_per_head_query",
        "frontier_logical_mb_per_head_query",
        "frontier_physical_mb_per_head_query",
        "logical_mb_savings_pct",
        "physical_mb_savings_pct",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]], figure_path: Path) -> None:
    lines = [
        "# Benchmark Quality vs Logical MB Savings",
        "",
        f"Figure: `{figure_path}`",
        "",
        "| group | pairs | dense quality % | frontier quality % | delta | logical MB savings % | physical MB savings % | note |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {group} | {pairs} | {dense:.2f} | {frontier:.2f} | {delta:.2f} | {logical:.1f} | {physical:.1f} | {note} |".format(
                group=row["group"],
                pairs=row["pairs"],
                dense=row["dense_quality_pct"],
                frontier=row["frontier_quality_pct"],
                delta=row["quality_delta_pct"],
                logical=row["logical_mb_savings_pct"],
                physical=row["physical_mb_savings_pct"],
                note=row["note"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot(path: Path, rows: list[dict[str, Any]], title: str) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    labels = [row["group"] for row in rows]
    x = np.arange(len(rows))
    width = 0.34
    dense = [row["dense_quality_pct"] for row in rows]
    frontier = [row["frontier_quality_pct"] for row in rows]
    savings = [row["logical_mb_savings_pct"] for row in rows]

    fig, ax_quality = plt.subplots(figsize=(13.0, 6.2), dpi=180)
    ax_savings = ax_quality.twinx()

    dense_bar = ax_quality.bar(x - width / 2, dense, width, label="Dense quality", color="#2f6f9f")
    frontier_bar = ax_quality.bar(x + width / 2, frontier, width, label="Frontier quality", color="#d9822b")
    line = ax_savings.plot(
        x,
        savings,
        color="#2f8f46",
        marker="o",
        linewidth=2.5,
        label="Logical MB savings",
    )

    ax_quality.set_ylabel("Quality (%)")
    ax_quality.set_ylim(0, max(105, max(dense + frontier) * 1.12))
    min_savings = min(savings)
    ax_savings.set_ylabel("Logical MB savings vs dense (%)")
    ax_savings.set_ylim(min(-45, math.floor(min_savings / 10.0) * 10.0 - 5.0), 95)
    ax_savings.axhline(0, color="#4a4a4a", linewidth=1.0, linestyle="--", alpha=0.6)

    ax_quality.set_xticks(x)
    ax_quality.set_xticklabels(labels, rotation=25, ha="right")
    ax_quality.set_title(title)
    ax_quality.grid(axis="y", alpha=0.25)
    ax_quality.set_axisbelow(True)

    for bars in (dense_bar, frontier_bar):
        for bar in bars:
            value = bar.get_height()
            ax_quality.text(
                bar.get_x() + bar.get_width() / 2,
                value + 1.0,
                f"{value:.0f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    for xi, value in zip(x, savings):
        ax_savings.text(
            xi,
            value + (3.0 if value >= 0 else -5.0),
            f"{value:.0f}%",
            color="#1f6f35",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=8,
        )

    handles1, labels1 = ax_quality.get_legend_handles_labels()
    handles2, labels2 = ax_savings.get_legend_handles_labels()
    ax_quality.legend(handles1 + handles2, labels1 + labels2, loc="upper center", ncol=3, frameon=False)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    if path.suffix.lower() != ".pdf":
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audit-json",
        action="append",
        type=Path,
        required=True,
        help="Pair-audit JSON. Can be passed multiple times.",
    )
    parser.add_argument("--output-png", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument(
        "--title",
        default="Dense vs Frontier Quality and Logical MB Savings",
    )
    args = parser.parse_args()

    pairs = _load_pairs(args.audit_json)
    rows = build_rows(pairs)
    if not rows:
        raise SystemExit("no completed benchmark groups found")
    write_csv(args.output_csv, rows)
    write_markdown(args.output_md, rows, args.output_png)
    plot(args.output_png, rows, args.title)


if __name__ == "__main__":
    main()
