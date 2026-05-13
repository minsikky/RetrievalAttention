#!/usr/bin/env python3
"""Plot true-online IVF-PQ simulator summaries."""

from __future__ import annotations

import csv
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


INPUT_CSV = Path(os.environ.get("ONLINE_IVFPQ_SUMMARY", "attention_efficiency_result/online_ivfpq_simulator_v1/summary.csv"))
OUT_DIR = Path(os.environ.get("OUT_DIR", str(INPUT_CSV.parent / "plots")))

STYLE = {
    "dense_oracle": {"label": "Oracle", "color": "#222222", "marker": "P"},
    "pqcache_full_scan_oracle": {"label": "PQCache full scan", "color": "#4D908E", "marker": "p"},
    "ivfpq_online_oracle": {"label": "IVF-PQ adaptive", "color": "#2E6F9E", "marker": "o"},
}


def load_rows() -> list[dict]:
    with INPUT_CSV.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["decode_tokens"] = int(row["decode_tokens"])
        row["target_mass"] = float(row["target_mass"])
        row["estimated_mb_mean"] = float(row["estimated_mb_mean"])
        row["mass_mean"] = float(row["mass_mean"])
        row["reached_rate"] = float(row["reached_rate"])
        row["nprobe"] = int(float(row["nprobe"]))
        row["final_k"] = int(float(row["final_k"]))
    return rows


def best_rows(rows: list[dict]) -> dict[tuple[int, float, str, str], dict]:
    grouped: dict[tuple[int, float, str, str], list[dict]] = {}
    for row in rows:
        method = row["method"]
        if method not in STYLE:
            continue
        grouped.setdefault((row["decode_tokens"], row["target_mass"], row["policy"], method), []).append(row)
    out = {}
    for key, items in grouped.items():
        reached = [row for row in items if row["reached_rate"] >= 1.0]
        pool = reached if reached else items
        out[key] = min(pool, key=lambda row: (row["estimated_mb_mean"], -row["mass_mean"]))
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    chosen = best_rows(rows)
    decodes = sorted({key[0] for key in chosen})
    targets = sorted({key[1] for key in chosen})
    policies = sorted({key[2] for key in chosen})

    for policy in policies:
        fig, axes = plt.subplots(1, len(targets), figsize=(7.4 * len(targets), 5.1), sharey=True)
        if len(targets) == 1:
            axes = [axes]
        for ax, target in zip(axes, targets):
            for method, style in STYLE.items():
                xs = [d for d in decodes if (d, target, policy, method) in chosen]
                if not xs:
                    continue
                ys = [chosen[(d, target, policy, method)]["estimated_mb_mean"] for d in xs]
                reach = [chosen[(d, target, policy, method)]["reached_rate"] for d in xs]
                ax.plot(
                    xs,
                    ys,
                    color=style["color"],
                    marker=style["marker"],
                    linewidth=2.3,
                    markersize=6,
                    linestyle="--" if any(x < 1.0 for x in reach) else "-",
                    label=style["label"],
                )
            ax.set_xscale("log", base=2)
            ax.set_xticks(decodes)
            ax.set_xticklabels([str(x) for x in decodes], rotation=30)
            ax.grid(True, color="#DDDDDD", linewidth=0.8)
            ax.set_title(f"Target mass = {target:.2f}")
            ax.set_xlabel("Decode length")
            ax.set_ylabel("Estimated logical bytes/query (MB)")
        axes[0].legend(frameon=False)
        fig.suptitle(f"True-online IVF-PQ simulator: {policy}", fontsize=15, fontweight="bold")
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"online_ivfpq_mb_vs_decode_{policy}.png", dpi=240)
        fig.savefig(OUT_DIR / f"online_ivfpq_mb_vs_decode_{policy}.pdf")
        plt.close(fig)

    with (OUT_DIR / "online_ivfpq_best_rows.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "decode_tokens",
            "target_mass",
            "policy",
            "method",
            "estimated_mb_mean",
            "mass_mean",
            "reached_rate",
            "nprobe",
            "final_k",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for key in sorted(chosen):
            writer.writerow(chosen[key])
    print(f"wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
