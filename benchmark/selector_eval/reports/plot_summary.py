#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def load_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot selector-eval summary metrics.")
    parser.add_argument("--summary_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--quality_metric", default="attention_mass_mean")
    parser.add_argument(
        "--cost_metric",
        default="query_MB_mean",
        help="Cost column for the main MB plot. Use total_MB_mean to include online update traffic.",
    )
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    rows = load_rows(Path(args.summary_csv))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = sorted({float(row["target_mass"]) for row in rows})

    for target in targets:
        target_rows = [row for row in rows if abs(float(row["target_mass"]) - target) < 1e-9]
        algorithms = sorted({row["algorithm"] for row in target_rows})

        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        for algorithm in algorithms:
            algo_rows = sorted(
                [row for row in target_rows if row["algorithm"] == algorithm],
                key=lambda row: int(row["decode_length"]),
            )
            if not algo_rows:
                continue
            x = [int(row["decode_length"]) for row in algo_rows]
            y = [float(row.get(args.cost_metric, row["total_MB_mean"])) for row in algo_rows]
            ax.plot(x, y, marker="o", label=algorithm)
        ax.set_title(f"{args.cost_metric} at target mass {target:g}")
        ax.set_xlabel("Decode length")
        ax.set_ylabel("MB/query")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"{args.cost_metric}_vs_decode_t{target:g}.png", dpi=220)
        plt.close(fig)

        if any("online_update_MB_mean" in row for row in target_rows):
            fig, ax = plt.subplots(figsize=(7.5, 4.5))
            for algorithm in algorithms:
                algo_rows = sorted(
                    [row for row in target_rows if row["algorithm"] == algorithm],
                    key=lambda row: int(row["decode_length"]),
                )
                x = [int(row["decode_length"]) for row in algo_rows]
                y = [float(row.get("online_update_MB_mean", "0")) for row in algo_rows]
                ax.plot(x, y, marker="o", label=algorithm)
            ax.set_title(f"online_update_MB_mean at target mass {target:g}")
            ax.set_xlabel("Decode length")
            ax.set_ylabel("MB/query")
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(out_dir / f"online_update_MB_mean_vs_decode_t{target:g}.png", dpi=220)
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        for algorithm in algorithms:
            algo_rows = sorted(
                [row for row in target_rows if row["algorithm"] == algorithm],
                key=lambda row: int(row["decode_length"]),
            )
            x = [int(row["decode_length"]) for row in algo_rows]
            y = [float(row.get(args.quality_metric, "nan")) for row in algo_rows]
            ax.plot(x, y, marker="o", label=algorithm)
        ax.set_title(f"{args.quality_metric} at target mass {target:g}")
        ax.set_xlabel("Decode length")
        ax.set_ylabel(args.quality_metric)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"{args.quality_metric}_vs_decode_t{target:g}.png", dpi=220)
        plt.close(fig)

    print(f"[selector_eval.plot] wrote {out_dir}")


if __name__ == "__main__":
    main()
