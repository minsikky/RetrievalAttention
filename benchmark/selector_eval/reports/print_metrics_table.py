#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


DEFAULT_COLUMNS = [
    "decode_length",
    "attention_mass_mean",
    "FN_mass_mean",
    "FP_mass_mean",
    "distribution_JS_mean",
    "output_cosine_mean",
    "output_relative_L2_mean",
    "selected_tokens_mean",
    "candidate_tokens_mean",
    "selector_MB_per_query_mean",
    "exact_KV_MB_per_query_mean",
    "online_update_cumulative_MB_mean",
    "online_update_MB_per_token_mean",
    "step_MB_per_query_mean",
]


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fmt(value: str) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(number - round(number)) < 1e-9 and abs(number) >= 10:
        return f"{int(round(number)):,}"
    if abs(number) >= 100:
        return f"{number:.3f}"
    if abs(number) >= 1:
        return f"{number:.6f}"
    return f"{number:.6f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Print a compact Markdown metrics table from selector-eval summary.csv.")
    parser.add_argument("--summary_csv", required=True)
    parser.add_argument("--algorithm", default="", help="Optional algorithm filter.")
    parser.add_argument("--target", type=float, default=None, help="Optional target_mass filter.")
    parser.add_argument("--columns", default=",".join(DEFAULT_COLUMNS))
    args = parser.parse_args()

    rows = load_rows(Path(args.summary_csv))
    if args.algorithm:
        rows = [row for row in rows if row.get("algorithm") == args.algorithm]
    if args.target is not None:
        rows = [row for row in rows if abs(float(row.get("target_mass", "nan")) - float(args.target)) < 1e-9]
    rows = sorted(rows, key=lambda row: (row.get("algorithm", ""), int(float(row.get("decode_length", "0")))))
    columns = [col.strip() for col in str(args.columns).split(",") if col.strip()]
    columns = [col for col in columns if any(col in row for row in rows)]

    print("| " + " | ".join(columns) + " |")
    print("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        print("| " + " | ".join(fmt(row.get(col, "")) for col in columns) + " |")


if __name__ == "__main__":
    main()
