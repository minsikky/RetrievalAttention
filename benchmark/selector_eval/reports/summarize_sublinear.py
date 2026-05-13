#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import numpy as np


def _f(row: dict[str, str], key: str) -> float:
    return float(row.get(key, "nan"))


def _i(row: dict[str, str], key: str) -> int:
    return int(float(row.get(key, "0")))


def _group_seed(algorithm: str) -> str:
    return re.sub(r"_seed\d+$", "", algorithm)


def _load(paths: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for text in paths:
        with Path(text).open(newline="", encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))
    return rows


def _slope(xs: list[float], ys: list[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys, strict=False) if x > 0 and y > 0 and math.isfinite(y)]
    if len(pairs) < 2:
        return float("nan")
    lx = np.log([x for x, _y in pairs])
    ly = np.log([y for _x, y in pairs])
    return float(np.polyfit(lx, ly, 1)[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize sublinear selector/tail sweeps.")
    parser.add_argument("--summary_csv", action="append", required=True)
    parser.add_argument("--output_md", required=True)
    parser.add_argument("--baseline", default="paged_local_pq_approx_sched_v2")
    parser.add_argument("--top", type=int, default=40)
    args = parser.parse_args()

    rows = _load(args.summary_csv)
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        if "oracle_prob_tail" in row.get("algorithm", ""):
            continue
        grouped.setdefault(_group_seed(row["algorithm"]), []).append(row)

    lines: list[str] = []
    lines.append("# Sublinear Selector-Eval Summary\n\n")
    lines.append("Sources:\n\n")
    for path in args.summary_csv:
        lines.append(f"- `{path}`\n")
    lines.append("\n")

    baseline_rows = grouped.get(args.baseline, [])
    baseline_max_l2 = max((_f(row, "output_relative_L2_mean") for row in baseline_rows), default=float("nan"))
    baseline_max_step = max((_f(row, "step_MB_per_query_mean") for row in baseline_rows), default=float("nan"))
    lines.append("## Baseline\n\n")
    lines.append(f"- `{args.baseline}` max relL2: `{baseline_max_l2:.6f}`\n")
    lines.append(f"- `{args.baseline}` max stepMB/query: `{baseline_max_step:.3f}`\n\n")

    summary = []
    for algorithm, items in grouped.items():
        if algorithm.startswith("top_budget_oracle") or algorithm.startswith("top_fraction_oracle"):
            deployable = False
        else:
            deployable = True
        by_len: dict[int, list[dict[str, str]]] = {}
        for row in items:
            by_len.setdefault(_i(row, "decode_length"), []).append(row)
        lengths = sorted(by_len)
        mean_selected = [float(np.mean([_f(row, "selected_tokens_mean") for row in by_len[length]])) for length in lengths]
        mean_step = [float(np.mean([_f(row, "step_MB_per_query_mean") for row in by_len[length]])) for length in lengths]
        mean_exact_tail = [
            float(
                np.mean(
                    [
                        _f(row, "exact_KV_MB_per_query_mean") + _f(row, "tail_estimator_MB_per_query_mean")
                        for row in by_len[length]
                    ]
                )
            )
            for length in lengths
        ]
        mean_selector = [
            float(np.mean([_f(row, "selector_MB_per_query_mean") for row in by_len[length]])) for length in lengths
        ]
        max_l2 = max(_f(row, "output_relative_L2_mean") for row in items)
        min_cos = min(_f(row, "output_cosine_mean") for row in items)
        max_step = max(_f(row, "step_MB_per_query_mean") for row in items)
        max_selector = max(_f(row, "selector_MB_per_query_mean") for row in items)
        min_mass = min(_f(row, "attention_mass_mean") for row in items)
        endpoint = by_len[max(lengths)]
        endpoint_l2 = float(np.mean([_f(row, "output_relative_L2_mean") for row in endpoint]))
        endpoint_step = float(np.mean([_f(row, "step_MB_per_query_mean") for row in endpoint]))
        summary.append(
            {
                "algorithm": algorithm,
                "deployable": deployable,
                "max_l2": max_l2,
                "min_cos": min_cos,
                "max_step": max_step,
                "max_selector": max_selector,
                "min_mass": min_mass,
                "endpoint_l2": endpoint_l2,
                "endpoint_step": endpoint_step,
                "selected_alpha": _slope([float(x) for x in lengths], mean_selected),
                "step_alpha": _slope([float(x) for x in lengths], mean_step),
                "exact_tail_alpha": _slope([float(x) for x in lengths], mean_exact_tail),
                "selector_alpha": _slope([float(x) for x in lengths], mean_selector),
            }
        )

    def row_key(item: dict) -> tuple[float, float]:
        beats_l2 = item["max_l2"] <= baseline_max_l2
        beats_step = item["max_step"] < baseline_max_step
        return (0.0 if beats_l2 and beats_step else 1.0, item["max_step"])

    lines.append("## Best Deployable Rows\n\n")
    lines.append(
        "| algorithm | max relL2 | min cos | max stepMB | selected alpha | exact+tail alpha | selector alpha | min mass |\n"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
    for item in sorted([x for x in summary if x["deployable"]], key=row_key)[: args.top]:
        lines.append(
            f"| {item['algorithm']} | {item['max_l2']:.6f} | {item['min_cos']:.6f} | "
            f"{item['max_step']:.3f} | {item['selected_alpha']:.3f} | "
            f"{item['exact_tail_alpha']:.3f} | {item['selector_alpha']:.3f} | {item['min_mass']:.6f} |\n"
        )

    lines.append("\n## Oracle Diagnostics\n\n")
    lines.append("| algorithm | max relL2 | min cos | max stepMB | selected alpha | exact+tail alpha | min mass |\n")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n")
    for item in sorted([x for x in summary if not x["deployable"]], key=lambda x: (x["max_l2"], x["max_step"]))[: args.top]:
        lines.append(
            f"| {item['algorithm']} | {item['max_l2']:.6f} | {item['min_cos']:.6f} | "
            f"{item['max_step']:.3f} | {item['selected_alpha']:.3f} | "
            f"{item['exact_tail_alpha']:.3f} | {item['min_mass']:.6f} |\n"
        )

    lines.append("\n## Reading The Table\n\n")
    lines.append("- `selected alpha` is the log-log slope of selected tokens vs decode length; below `1.0` is sublinear selected-token growth.\n")
    lines.append("- `exact+tail alpha` is the slope of exact KV plus tail-estimator traffic; this is the attention-read complexity excluding selector scan.\n")
    lines.append("- `selector alpha` exposes whether selector traffic is still linear even if exact attention is sublinear.\n")
    lines.append("- A credible win should beat baseline max relL2 and max stepMB across all decode lengths, not only at 128k.\n")

    Path(args.output_md).write_text("".join(lines), encoding="utf-8")
    print(args.output_md)


if __name__ == "__main__":
    main()
