#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


def f(row: dict[str, str], key: str) -> float:
    return float(row.get(key, "nan"))


def i(row: dict[str, str], key: str) -> int:
    return int(float(row.get(key, "0")))


def parse_head_budget(algorithm: str) -> str:
    base = algorithm.split("+", 1)[0]
    if "_budget_" not in base:
        return "adaptive"
    return base.rsplit("_budget_", 1)[1]


def parse_tail(algorithm: str) -> str:
    if "+" not in algorithm:
        return "none"
    return re.sub(r"_seed\d+$", "", algorithm.split("+", 1)[1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize head/tail split sweeps.")
    parser.add_argument("--summary_csv", required=True)
    parser.add_argument("--output_md", required=True)
    parser.add_argument("--endpoint", type=int, default=128000)
    args = parser.parse_args()

    rows = list(csv.DictReader(open(args.summary_csv, newline="", encoding="utf-8")))
    rows = [r for r in rows if i(r, "decode_length") == args.endpoint]
    rows = [r for r in rows if not r["algorithm"].startswith("top_")]
    baseline = next((r for r in rows if r["algorithm"] == "paged_local_pq_approx_sched_v2"), None)

    lines: list[str] = []
    lines.append("# Head/Tail Split Summary\n\n")
    lines.append(f"Source: `{args.summary_csv}`\n\n")
    if baseline:
        lines.append("## Baseline\n\n")
        lines.append("| algorithm | relL2 | stepMB | selectorMB | exactKVMB | tailMB | selected | mass |\n")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        lines.append(
            f"| {baseline['algorithm']} | {f(baseline, 'output_relative_L2_mean'):.6f} | "
            f"{f(baseline, 'step_MB_per_query_mean'):.3f} | {f(baseline, 'selector_MB_per_query_mean'):.3f} | "
            f"{f(baseline, 'exact_KV_MB_per_query_mean'):.3f} | {f(baseline, 'tail_estimator_MB_per_query_mean'):.3f} | "
            f"{f(baseline, 'selected_tokens_mean'):.0f} | {f(baseline, 'attention_mass_mean'):.6f} |\n\n"
        )

    rows = sorted(rows, key=lambda r: (f(r, "output_relative_L2_mean"), f(r, "step_MB_per_query_mean")))
    lines.append("## Best By Quality\n\n")
    lines.append("| algorithm | head | tail | relL2 | cos | stepMB | selectorMB | exactKVMB | tailMB | selected | mass |\n")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
    for r in rows[:40]:
        lines.append(
            f"| {r['algorithm']} | {parse_head_budget(r['algorithm'])} | {parse_tail(r['algorithm'])} | "
            f"{f(r, 'output_relative_L2_mean'):.6f} | {f(r, 'output_cosine_mean'):.6f} | "
            f"{f(r, 'step_MB_per_query_mean'):.3f} | {f(r, 'selector_MB_per_query_mean'):.3f} | "
            f"{f(r, 'exact_KV_MB_per_query_mean'):.3f} | {f(r, 'tail_estimator_MB_per_query_mean'):.3f} | "
            f"{f(r, 'selected_tokens_mean'):.0f} | {f(r, 'attention_mass_mean'):.6f} |\n"
        )

    bands = [(0, 5), (5, 8), (8, 12), (12, 16), (16, 24), (24, 32), (32, 1000)]
    lines.append("\n## Best Within Step-MB Bands\n\n")
    lines.append("| stepMB band | algorithm | relL2 | stepMB | selectorMB | exactKVMB | tailMB | selected | mass |\n")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
    for lo, hi in bands:
        candidates = [r for r in rows if lo <= f(r, "step_MB_per_query_mean") < hi]
        if not candidates:
            continue
        r = min(candidates, key=lambda x: f(x, "output_relative_L2_mean"))
        lines.append(
            f"| [{lo},{hi}) | {r['algorithm']} | {f(r, 'output_relative_L2_mean'):.6f} | "
            f"{f(r, 'step_MB_per_query_mean'):.3f} | {f(r, 'selector_MB_per_query_mean'):.3f} | "
            f"{f(r, 'exact_KV_MB_per_query_mean'):.3f} | {f(r, 'tail_estimator_MB_per_query_mean'):.3f} | "
            f"{f(r, 'selected_tokens_mean'):.0f} | {f(r, 'attention_mass_mean'):.6f} |\n"
        )

    Path(args.output_md).write_text("".join(lines), encoding="utf-8")
    print(args.output_md)


if __name__ == "__main__":
    main()
