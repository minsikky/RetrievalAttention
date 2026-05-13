#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _float(row: dict[str, str], key: str) -> float:
    return float(row.get(key, "nan"))


def _int(row: dict[str, str], key: str) -> int:
    return int(float(row.get(key, "0")))


def _load(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _base_algorithm(algorithm: str) -> str:
    return algorithm.split("+", 1)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize selector-eval tail-estimation sweeps.")
    parser.add_argument("--summary_csv", required=True)
    parser.add_argument("--output_md", required=True)
    parser.add_argument("--endpoint", type=int, default=128000)
    parser.add_argument("--baseline", default="paged_local_pq_approx_sched_v2")
    args = parser.parse_args()

    rows = _load(Path(args.summary_csv))
    endpoint_rows = [row for row in rows if _int(row, "decode_length") == int(args.endpoint)]
    baseline = next((row for row in endpoint_rows if row.get("algorithm") == args.baseline), None)
    lines: list[str] = []
    lines.append("# Tail-Sampling Selector-Eval Summary\n\n")
    lines.append(f"Source: `{args.summary_csv}`\n\n")
    lines.append(f"Endpoint decode length: `{args.endpoint}`\n\n")
    if baseline:
        lines.append("## Baseline\n\n")
        lines.append("| algorithm | mass | output_cosine | output_relative_L2 | selectorMB | exactKVMB | tailMB | stepMB |\n")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        lines.append(
            f"| {baseline['algorithm']} | {_float(baseline, 'attention_mass_mean'):.6f} | "
            f"{_float(baseline, 'output_cosine_mean'):.6f} | {_float(baseline, 'output_relative_L2_mean'):.6f} | "
            f"{_float(baseline, 'selector_MB_per_query_mean'):.3f} | "
            f"{_float(baseline, 'exact_KV_MB_per_query_mean'):.3f} | "
            f"{_float(baseline, 'tail_estimator_MB_per_query_mean'):.3f} | "
            f"{_float(baseline, 'step_MB_per_query_mean'):.3f} |\n\n"
        )

    deployable = [
        row
        for row in endpoint_rows
        if "oracle_prob_tail" not in row.get("algorithm", "")
        and row.get("tail_oracle_diagnostic_mean", row.get("tail_oracle_diagnostic", "False")) in {"False", "0.0", "0", ""}
        and not row.get("algorithm", "").startswith("top_fraction_oracle")
    ]
    deployable = sorted(
        deployable,
        key=lambda row: (_float(row, "output_relative_L2_mean"), _float(row, "step_MB_per_query_mean")),
    )
    lines.append("## Best Deployable Endpoint Rows By Relative L2\n\n")
    lines.append("| algorithm | mass | output_cosine | output_relative_L2 | stepMB | tailMB | selected | tail_samples |\n")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
    for row in deployable[:20]:
        lines.append(
            f"| {row['algorithm']} | {_float(row, 'attention_mass_mean'):.6f} | "
            f"{_float(row, 'output_cosine_mean'):.6f} | {_float(row, 'output_relative_L2_mean'):.6f} | "
            f"{_float(row, 'step_MB_per_query_mean'):.3f} | {_float(row, 'tail_estimator_MB_per_query_mean'):.3f} | "
            f"{_float(row, 'selected_tokens_mean'):.0f} | {_float(row, 'tail_samples_mean'):.0f} |\n"
        )

    available_lengths = sorted({_int(row, "decode_length") for row in rows})
    if len(available_lengths) > 1:
        lines.append("\n## Full-Curve Worst Case By Algorithm\n\n")
    else:
        lines.append("\n## Available-Length Worst Case By Algorithm\n\n")
        lines.append(f"Only decode length `{available_lengths[0]}` is present in this file.\n\n")
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["algorithm"], []).append(row)
    full = []
    for algorithm, items in grouped.items():
        if algorithm.startswith("top_fraction_oracle"):
            continue
        if "oracle_prob_tail" in algorithm:
            continue
        oracle_diag = any(
            item.get("tail_oracle_diagnostic_mean", item.get("tail_oracle_diagnostic", "False")) in {"True", "1.0", "1"}
            for item in items
        )
        if oracle_diag:
            continue
        full.append(
            (
                max(_float(item, "output_relative_L2_mean") for item in items),
                max(_float(item, "step_MB_per_query_mean") for item in items),
                min(_float(item, "output_cosine_mean") for item in items),
                min(_float(item, "attention_mass_mean") for item in items),
                algorithm,
            )
        )
    full.sort()
    lines.append("| algorithm | max_relL2 | min_cos | min_mass | max_stepMB |\n")
    lines.append("| --- | ---: | ---: | ---: | ---: |\n")
    for max_l2, max_step, min_cos, min_mass, algorithm in full[:30]:
        lines.append(f"| {algorithm} | {max_l2:.6f} | {min_cos:.6f} | {min_mass:.6f} | {max_step:.3f} |\n")

    lines.append("\n## Interpretation Checklist\n\n")
    lines.append("- Endpoint wins are not enough; inspect full-curve worst-case relative L2.\n")
    lines.append("- Oracle-probability tail rows are diagnostics only and must not be counted as deployable wins.\n")
    lines.append("- A useful result should reduce `step_MB/query` versus the baseline while keeping relative L2 close to the baseline.\n")

    Path(args.output_md).write_text("".join(lines), encoding="utf-8")
    print(args.output_md)


if __name__ == "__main__":
    main()
