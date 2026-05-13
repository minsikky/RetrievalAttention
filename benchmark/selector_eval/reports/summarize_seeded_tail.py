#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


def f(row: dict[str, str], key: str) -> float:
    return float(row.get(key, "nan"))


def base_algorithm(name: str) -> str:
    return re.sub(r"_seed\d+$", "", name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize seeded tail-estimator robustness.")
    parser.add_argument("--summary_csv", required=True)
    parser.add_argument("--output_md", required=True)
    args = parser.parse_args()

    rows = list(csv.DictReader(open(args.summary_csv, newline="", encoding="utf-8")))
    by_alg_seed: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        alg = row["algorithm"]
        base = base_algorithm(alg)
        seed_match = re.search(r"_seed(\d+)$", alg)
        seed = seed_match.group(1) if seed_match else "none"
        by_alg_seed[(base, seed)].append(row)

    per_seed: dict[str, list[dict[str, float | str]]] = defaultdict(list)
    for (base, seed), items in by_alg_seed.items():
        per_seed[base].append(
            {
                "seed": seed,
                "max_l2": max(f(row, "output_relative_L2_mean") for row in items),
                "min_cos": min(f(row, "output_cosine_mean") for row in items),
                "max_step": max(f(row, "step_MB_per_query_mean") for row in items),
                "min_mass": min(f(row, "attention_mass_mean") for row in items),
            }
        )

    summary = []
    for base, seed_rows in per_seed.items():
        if base.endswith("+none"):
            continue
        l2 = np.asarray([float(row["max_l2"]) for row in seed_rows], dtype=np.float64)
        step = np.asarray([float(row["max_step"]) for row in seed_rows], dtype=np.float64)
        cos = np.asarray([float(row["min_cos"]) for row in seed_rows], dtype=np.float64)
        summary.append(
            {
                "algorithm": base,
                "seeds": len(seed_rows),
                "mean_max_relL2": float(l2.mean()),
                "worst_max_relL2": float(l2.max()),
                "best_max_relL2": float(l2.min()),
                "std_max_relL2": float(l2.std(ddof=0)),
                "mean_stepMB": float(step.mean()),
                "worst_min_cos": float(cos.min()),
            }
        )
    summary.sort(key=lambda row: (row["mean_max_relL2"], row["mean_stepMB"]))

    lines = ["# Seeded Tail Estimator Summary", "", f"Source: `{args.summary_csv}`", ""]
    lines.append("| algorithm | seeds | mean max relL2 | worst max relL2 | best max relL2 | std relL2 | mean stepMB | worst min cos |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in summary[:100]:
        lines.append(
            f"| {row['algorithm']} | {row['seeds']} | {row['mean_max_relL2']:.6f} | "
            f"{row['worst_max_relL2']:.6f} | {row['best_max_relL2']:.6f} | "
            f"{row['std_max_relL2']:.6f} | {row['mean_stepMB']:.3f} | {row['worst_min_cos']:.6f} |"
        )
    Path(args.output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output_md)


if __name__ == "__main__":
    main()
