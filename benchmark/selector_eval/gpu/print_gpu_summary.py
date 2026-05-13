#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv


def f(row: dict[str, str], key: str) -> float:
    return float(row.get(key, "nan"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Print compact GPU selector-eval summary.")
    parser.add_argument("--summary_csv", required=True)
    args = parser.parse_args()
    rows = list(csv.DictReader(open(args.summary_csv, newline="", encoding="utf-8")))
    print("| algorithm | decode | head | relL2 | cos | mass | total_ms med | selector_ms med | attn_ms med | dense_ms | stepMB | gpuPeakMB |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in sorted(rows, key=lambda r: (int(r["decode_length"]), int(r["head"]), r["algorithm"])):
        total_key = "total_query_seconds_median" if "total_query_seconds_median" in row else "total_query_seconds_mean"
        selector_key = "selector_seconds_median" if "selector_seconds_median" in row else "selector_seconds_mean"
        attention_key = "attention_seconds_median" if "attention_seconds_median" in row else "attention_seconds_mean"
        print(
            f"| {row['algorithm']} | {int(row['decode_length'])} | {int(row['head'])} | "
            f"{f(row, 'output_relative_L2_mean'):.6f} | {f(row, 'output_cosine_mean'):.6f} | "
            f"{f(row, 'attention_mass_mean'):.6f} | {1000.0 * f(row, total_key):.3f} | "
            f"{1000.0 * f(row, selector_key):.3f} | {1000.0 * f(row, attention_key):.3f} | "
            f"{1000.0 * f(row, 'dense_seconds_mean'):.3f} | {f(row, 'step_MB_per_query_mean'):.3f} | "
            f"{f(row, 'gpu_peak_MB_mean'):.1f} |"
        )


if __name__ == "__main__":
    main()
