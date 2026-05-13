#!/usr/bin/env python3
"""Plot merged attention-efficiency tables.

Input is the normalized CSV produced in
attention_efficiency_result/merged_attention_efficiency_full_table.csv.
"""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path("attention_efficiency_result")
INPUT_CSV = Path(os.environ.get("MERGED_CSV", str(ROOT / "merged_attention_efficiency_full_table.csv")))
OUT_DIR = Path(os.environ.get("OUT_DIR", str(ROOT / "plots" / "full_baselines")))
TARGETS = [0.95, 0.98]

FAMILY_ORDER = [
    "oracle",
    "retroinfer",
    "hybrid",
    "ra",
    "quest",
    "sparq",
    "loki",
    "magicpig",
    "pariskv",
    "pqcache",
]

FAMILY_STYLE = {
    "oracle": {"label": "Oracle", "color": "#222222", "marker": "P"},
    "retroinfer": {"label": "RetroInfer", "color": "#2E6F9E", "marker": "o"},
    "hybrid": {"label": "Hybrid centroid graph", "color": "#55A868", "marker": "^"},
    "ra": {"label": "RetrievalAttention", "color": "#C44E52", "marker": "s"},
    "quest": {"label": "Quest best", "color": "#8172B3", "marker": "D"},
    "sparq": {"label": "SparQ best", "color": "#CCB974", "marker": "v"},
    "loki": {"label": "Loki best", "color": "#64B5CD", "marker": "X"},
    "magicpig": {"label": "MagicPIG best", "color": "#DD8452", "marker": "*"},
    "pariskv": {"label": "ParisKV best", "color": "#937860", "marker": "h"},
    "pqcache": {"label": "PQCache", "color": "#4D908E", "marker": "p"},
}

SELECTED_METHODS = [
    "dense_oracle",
    "retroinfer",
    "hybrid_centroid_graph",
    "retrievalattention",
    "sparq_r8",
    "magicpig_adaptive",
    "pqcache_m2_b6",
]

METHOD_STYLE = {
    "dense_oracle": {"label": "Oracle", "color": "#222222", "marker": "P"},
    "retroinfer": {"label": "RetroInfer", "color": "#2E6F9E", "marker": "o"},
    "hybrid_centroid_graph": {"label": "Hybrid centroid graph", "color": "#55A868", "marker": "^"},
    "retrievalattention": {"label": "RetrievalAttention", "color": "#C44E52", "marker": "s"},
    "sparq_r8": {"label": "SparQ r8", "color": "#CCB974", "marker": "v"},
    "magicpig_adaptive": {"label": "MagicPIG adaptive", "color": "#DD8452", "marker": "*"},
    "pqcache_m2_b6": {"label": "PQCache m2 b6", "color": "#4D908E", "marker": "p"},
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
    return rows


def setup_axis(ax, decodes: list[int]) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xticks(decodes)
    ax.set_xticklabels([str(x) for x in decodes])
    ax.grid(True, which="major", color="#D7D7D7", linewidth=0.8)
    ax.grid(True, which="minor", axis="y", color="#EEEEEE", linewidth=0.5)
    ax.set_xlabel("Decode length")
    ax.set_ylabel("Estimated bytes read per query (MB)")


def best_family(rows: list[dict]) -> dict[tuple[int, float, str], dict]:
    grouped: dict[tuple[int, float, str], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["decode_tokens"], row["target_mass"], row["family"]), []).append(row)

    out: dict[tuple[int, float, str], dict] = {}
    for key, items in grouped.items():
        # Keep RA as the current paper-style RA row, even when it does not reach.
        if key[2] == "ra":
            current = [x for x in items if x["method"] == "retrievalattention"]
            items = current or items
        reached = [x for x in items if x["reached_rate"] >= 1.0]
        pool = reached if reached else items
        out[key] = min(pool, key=lambda x: (x["estimated_mb_mean"], -x["mass_mean"]))
    return out


def plot_two_panel(
    rows_by_key: dict[tuple[int, float, str], dict],
    series: list[str],
    styles: dict[str, dict],
    *,
    title: str,
    outfile_stem: str,
    key_kind: str,
) -> None:
    decodes = sorted({key[0] for key in rows_by_key})
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.1), sharey=True)
    for ax, target in zip(axes, TARGETS):
        for name in series:
            xs = [d for d in decodes if (d, target, name) in rows_by_key]
            if not xs:
                continue
            ys = [rows_by_key[(d, target, name)]["estimated_mb_mean"] for d in xs]
            reach = [rows_by_key[(d, target, name)]["reached_rate"] for d in xs]
            style = styles[name]
            linestyle = "--" if any(r < 1.0 for r in reach) else "-"
            alpha = 0.70 if any(r < 1.0 for r in reach) else 0.95
            ax.plot(
                xs,
                ys,
                label=style["label"],
                color=style["color"],
                marker=style["marker"],
                linewidth=2.4,
                markersize=6.5,
                linestyle=linestyle,
                alpha=alpha,
            )
        ax.set_title(f"Target attention mass = {target:.2f}")
        setup_axis(ax, decodes)
    axes[0].legend(frameon=False, loc="upper left", fontsize=10, ncol=1)
    fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"{outfile_stem}.{suffix}", dpi=240 if suffix == "png" else None)
    plt.close(fig)

    # Raw plotting data for reproducibility.
    with (OUT_DIR / f"{outfile_stem}.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["decode_tokens", "target_mass", key_kind, "label", "estimated_mb_mean", "mass_mean", "reached_rate", "method"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for target in TARGETS:
            for decode in decodes:
                for name in series:
                    row = rows_by_key.get((decode, target, name))
                    if row is None:
                        continue
                    writer.writerow(
                        {
                            "decode_tokens": decode,
                            "target_mass": target,
                            key_kind: name,
                            "label": styles[name]["label"],
                            "estimated_mb_mean": row["estimated_mb_mean"],
                            "mass_mean": row["mass_mean"],
                            "reached_rate": row["reached_rate"],
                            "method": row["method"],
                        }
                    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    by_method = {(r["decode_tokens"], r["target_mass"], r["method"]): r for r in rows}
    family_rows = best_family(rows)

    plot_two_panel(
        family_rows,
        FAMILY_ORDER,
        FAMILY_STYLE,
        title="Full baseline cost by retrieval family",
        outfile_stem="full_family_best_mb_vs_decode",
        key_kind="family",
    )
    plot_two_panel(
        by_method,
        SELECTED_METHODS,
        METHOD_STYLE,
        title="Selected baselines: MB vs decode length",
        outfile_stem="selected_methods_mb_vs_decode",
        key_kind="method_key",
    )

    # A zoomed/practical plot drops the very expensive ParisKV line and keeps the
    # families we currently care about for slide readability.
    practical = ["oracle", "retroinfer", "hybrid", "ra", "sparq", "magicpig", "pqcache"]
    plot_two_panel(
        family_rows,
        practical,
        FAMILY_STYLE,
        title="Practical baseline cost by retrieval family",
        outfile_stem="practical_family_mb_vs_decode",
        key_kind="family",
    )

    print(f"wrote {OUT_DIR / 'full_family_best_mb_vs_decode.png'}")
    print(f"wrote {OUT_DIR / 'selected_methods_mb_vs_decode.png'}")
    print(f"wrote {OUT_DIR / 'practical_family_mb_vs_decode.png'}")


if __name__ == "__main__":
    main()
