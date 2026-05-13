#!/usr/bin/env python3
"""Plot pre-rerank candidate oracle-mass frontiers."""

from __future__ import annotations

import csv
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(os.environ.get("FRONTIER_ROOT", "attention_efficiency_result/threeway_pqcache_variants_frontier_nograph_v1"))
OUT_DIR = Path(os.environ.get("OUT_DIR", "attention_efficiency_result/plots/pqcache_variants"))

STYLE = {
    "ivfpq": {"label": "IVF + global PQ", "color": "#2E6F9E", "marker": "o"},
    "binary": {"label": "Raw binary gate", "color": "#DD8452", "marker": "s"},
    "weighted": {"label": "Weighted Hamming", "color": "#55A868", "marker": "^"},
    "sign_vq": {"label": "Sign-VQ LUT", "color": "#8172B3", "marker": "D"},
}


def family(method: str) -> str:
    if method.startswith("ivfpq"):
        return "ivfpq"
    if method.startswith("binary_gated"):
        return "binary"
    if method.startswith("weighted_hamming"):
        return "weighted"
    if method.startswith("sign_vq"):
        return "sign_vq"
    return method


def load_rows() -> list[dict]:
    rows: list[dict] = []
    for path in sorted(ROOT.glob("decode_*/candidate_frontier_summary.csv")):
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                row["decode_tokens"] = int(row["decode_tokens"])
                row["budget_value"] = int(row["budget_value"])
                row["oracle_mass_mean"] = float(row["oracle_mass_mean"])
                row["estimated_mb_pre_pq_mean"] = float(row["estimated_mb_pre_pq_mean"])
                row["candidate_tokens_mean"] = float(row["candidate_tokens_mean"])
                row["family"] = family(row["method"])
                rows.append(row)
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    if not rows:
        raise RuntimeError(f"no candidate_frontier_summary.csv files under {ROOT}")
    decodes = sorted({row["decode_tokens"] for row in rows})
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.2), sharey=True)
    axes_flat = list(axes.flat)
    for ax, decode in zip(axes_flat, decodes):
        for fam in ("ivfpq", "binary", "weighted", "sign_vq"):
            series = [row for row in rows if row["decode_tokens"] == decode and row["family"] == fam]
            series.sort(key=lambda r: r["estimated_mb_pre_pq_mean"])
            if not series:
                continue
            style = STYLE[fam]
            xs = [row["estimated_mb_pre_pq_mean"] for row in series]
            ys = [row["oracle_mass_mean"] for row in series]
            labels = [row["budget_value"] for row in series]
            ax.plot(xs, ys, color=style["color"], marker=style["marker"], linewidth=2.2, markersize=5.5, label=style["label"])
            for x, y, label in zip(xs, ys, labels):
                ax.annotate(str(label), (x, y), textcoords="offset points", xytext=(4, 3), fontsize=7, color=style["color"])
        ax.axhline(0.95, color="#888888", linestyle="--", linewidth=0.9)
        ax.axhline(0.98, color="#BBBBBB", linestyle=":", linewidth=0.9)
        ax.set_title(f"Decode {decode}")
        ax.grid(True, color="#DDDDDD", linewidth=0.8)
        ax.set_xlabel("Pre-PQ routing cost (MB)")
        ax.set_ylabel("Oracle mass inside candidates")
    for ax in axes_flat[len(decodes) :]:
        ax.axis("off")
    axes_flat[0].legend(frameon=False, loc="lower right")
    fig.suptitle("Candidate frontier before PQ rerank", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "candidate_frontier_oracle_mass_vs_cost.png", dpi=240)
    fig.savefig(OUT_DIR / "candidate_frontier_oracle_mass_vs_cost.pdf")
    plt.close(fig)

    with (OUT_DIR / "candidate_frontier_oracle_mass_vs_cost.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "decode_tokens",
            "family",
            "method",
            "budget_kind",
            "budget_value",
            "estimated_mb_pre_pq_mean",
            "oracle_mass_mean",
            "candidate_tokens_mean",
            "samples",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {OUT_DIR / 'candidate_frontier_oracle_mass_vs_cost.png'}")


if __name__ == "__main__":
    main()
