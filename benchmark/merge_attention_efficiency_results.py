#!/usr/bin/env python3
"""Merge split attention-efficiency proxy outputs.

Method-specific Slurm jobs write separate summary.json files. This helper
combines them into one summary table and optional comparison plots.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge attention-efficiency proxy summaries.")
    parser.add_argument("inputs", nargs="+", help="Run directories containing summary.json files.")
    parser.add_argument("--output_dir", default="attention_efficiency_result/proxy_merged")
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args()


def load_rows(inputs: list[str]) -> list[dict]:
    rows = []
    for item in inputs:
        path = Path(item)
        summary_path = path / "summary.json" if path.is_dir() else path
        if not summary_path.exists():
            print(f"[merge_attention_efficiency_results] missing: {summary_path}")
            continue
        data = json.loads(summary_path.read_text())
        if not isinstance(data, list):
            raise ValueError(f"summary is not a list: {summary_path}")
        for row in data:
            row = dict(row)
            row.setdefault("source_dir", str(path))
            rows.append(row)
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def maybe_plot(rows: list[dict], output_dir: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", f"/tmp/matplotlib-{os.getuid()}")
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[merge_attention_efficiency_results] plot skipped: {type(exc).__name__}: {exc}")
        return

    for n_tokens in sorted({int(row["n_tokens"]) for row in rows}):
        n_rows = [row for row in rows if int(row["n_tokens"]) == n_tokens]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        panels = [
            ("token_read_ratio_mean", "dense_mass_covered_mean", "Dense Mass Covered"),
            ("token_read_ratio_mean", "recall_at_budget_mean", "Recall@Budget"),
            ("token_read_ratio_mean", "relative_attention_output_l2_mean", "Relative Output L2"),
        ]
        for ax, (x_key, y_key, title) in zip(axes, panels):
            for method in sorted({row["method"] for row in n_rows}):
                method_rows = sorted(
                    [row for row in n_rows if row["method"] == method],
                    key=lambda row: int(row["budget"]),
                )
                xs = [float(row[x_key]) for row in method_rows if row.get(x_key) is not None]
                ys = [float(row[y_key]) for row in method_rows if row.get(y_key) is not None]
                if xs and ys:
                    ax.plot(xs, ys, marker="o", label=method)
            ax.set_title(title)
            ax.set_xlabel("Token Read Ratio")
            ax.grid(True, alpha=0.3)
        axes[0].set_ylabel("Higher is better")
        axes[2].set_ylabel("Lower is better")
        axes[0].legend(fontsize=8)
        fig.suptitle(f"Attention Efficiency Proxy, N={n_tokens}")
        fig.tight_layout()
        fig.savefig(output_dir / f"merged_attention_efficiency_n{n_tokens}.png", dpi=160)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.inputs)
    rows = sorted(rows, key=lambda row: (int(row["n_tokens"]), int(row["budget"]), row["method"]))
    (output_dir / "summary.json").write_text(json.dumps(rows, indent=2, sort_keys=True))
    write_csv(rows, output_dir / "summary.csv")
    if args.plot:
        maybe_plot(rows, output_dir)
    print(f"[merge_attention_efficiency_results] rows={len(rows)}")
    print(f"[merge_attention_efficiency_results] output={output_dir}")


if __name__ == "__main__":
    main()
