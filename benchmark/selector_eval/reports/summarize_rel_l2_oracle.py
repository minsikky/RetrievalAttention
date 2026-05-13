#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def fmt_float(value: float, digits: int = 6) -> str:
    return f"{float(value):.{digits}f}"


def load_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["decode_length"] = int(row["decode_length"])
        row["selected_tokens"] = int(row["selected_tokens"])
        for key in ["target_l2", "attention_mass", "output_relative_L2", "output_cosine", "exact_KV_MB"]:
            row[key] = float(row[key])
        row["reached"] = str(row["reached"]).lower() in {"true", "1", "yes"}
    return rows


def markdown_table(rows: list[dict], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(str(row[col]) for col in columns) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def aggregate_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[float, int, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["target_l2"], row["decode_length"], row["algorithm"])].append(row)
    out = []
    for (target, decode_length, algorithm), items in grouped.items():
        n = len(items)
        out.append(
            {
                "target_l2": target,
                "decode_length": decode_length,
                "algorithm": algorithm,
                "samples": n,
                "all_reached": all(row["reached"] for row in items),
                "mean_selected_tokens": sum(row["selected_tokens"] for row in items) / n,
                "mean_attention_mass": sum(row["attention_mass"] for row in items) / n,
                "mean_output_relative_L2": sum(row["output_relative_L2"] for row in items) / n,
                "max_output_relative_L2": max(row["output_relative_L2"] for row in items),
                "mean_output_cosine": sum(row["output_cosine"] for row in items) / n,
                "mean_exact_KV_MB": sum(row["exact_KV_MB"] for row in items) / n,
            }
        )
    return sorted(out, key=lambda row: (row["target_l2"], row["decode_length"], row["algorithm"]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize relL2 oracle frontier rows.")
    parser.add_argument("output_dir")
    parser.add_argument("--markdown", default=None)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    frontier_path = out_dir / "frontier.csv"
    if not frontier_path.exists():
        raise FileNotFoundError(frontier_path)
    rows = aggregate_rows(load_rows(frontier_path))

    lines: list[str] = []
    lines.append("# relL2 Oracle Diagnostics")
    lines.append("")
    lines.append("First prefix budget reaching each output-relative-L2 target. These are offline diagnostics, not deployable selectors.")
    lines.append("")
    grouped: dict[float, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["target_l2"]].append(row)
    for target in sorted(grouped):
        lines.append(f"## target relL2 <= {fmt_float(target, 6)}")
        view: list[dict] = []
        for row in grouped[target]:
            view.append(
                {
                    "decode_length": row["decode_length"],
                    "algorithm": row["algorithm"],
                    "samples": row["samples"],
                    "all_reached": row["all_reached"],
                    "mean_selected": fmt_float(row["mean_selected_tokens"], 1),
                    "mean_mass": fmt_float(row["mean_attention_mass"], 6),
                    "mean_relL2": fmt_float(row["mean_output_relative_L2"], 6),
                    "max_relL2": fmt_float(row["max_output_relative_L2"], 6),
                    "mean_cos": fmt_float(row["mean_output_cosine"], 6),
                    "mean_exact_KV_MB": fmt_float(row["mean_exact_KV_MB"], 6),
                }
            )
        lines.append(
            markdown_table(
                view,
                [
                    "decode_length",
                    "algorithm",
                    "samples",
                    "all_reached",
                    "mean_selected",
                    "mean_mass",
                    "mean_relL2",
                    "max_relL2",
                    "mean_cos",
                    "mean_exact_KV_MB",
                ],
            )
        )
        lines.append("")

    text = "\n".join(lines)
    md_path = Path(args.markdown) if args.markdown else out_dir / "summary.md"
    md_path.write_text(text, encoding="utf-8")
    print(text)
    print(f"[summarize_rel_l2_oracle] wrote {md_path}")


if __name__ == "__main__":
    main()
