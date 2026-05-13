#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def f(row: dict, key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    return float(value)


def summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["algorithm"]].append(row)
    out = []
    for algorithm, items in grouped.items():
        out.append(
            {
                "algorithm": algorithm,
                "max_relL2": max(f(row, "output_relative_L2_mean") for row in items),
                "min_cos": min(f(row, "output_cosine_mean") for row in items),
                "max_stepMB": max(f(row, "step_MB_per_query_mean") for row in items),
                "max_selectorMB": max(f(row, "selector_MB_per_query_mean") for row in items),
                "max_exactKVMB": max(f(row, "exact_KV_MB_per_query_mean") for row in items),
                "max_estimatorMB": max(f(row, "tail_estimator_MB_per_query_mean") for row in items),
                "min_mass": min(f(row, "attention_mass_mean") for row in items),
            }
        )
    return sorted(out, key=lambda row: (row["max_stepMB"], row["max_relL2"], row["algorithm"]))


def markdown_table(rows: list[dict]) -> str:
    cols = [
        "algorithm",
        "max_relL2",
        "min_cos",
        "max_stepMB",
        "max_selectorMB",
        "max_exactKVMB",
        "max_estimatorMB",
        "min_mass",
    ]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["algorithm"]),
                    f"{row['max_relL2']:.6f}",
                    f"{row['min_cos']:.6f}",
                    f"{row['max_stepMB']:.3f}",
                    f"{row['max_selectorMB']:.3f}",
                    f"{row['max_exactKVMB']:.3f}",
                    f"{row['max_estimatorMB']:.3f}",
                    f"{row['min_mass']:.6f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize flipped head/tail diagnostic.")
    parser.add_argument("output_dir")
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    with (out_dir / "summary.csv").open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    summary = summarize(rows)
    text = "\n".join(
        [
            "# Flipped Head/Tail Diagnostic",
            "",
            "Rows with `uniform_head_*` exact-read the unselected tail and estimate the selected head. They are diagnostics for the flipped idea.",
            "",
            markdown_table(summary),
            "",
        ]
    )
    (out_dir / "flipped_head_tail_summary.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
