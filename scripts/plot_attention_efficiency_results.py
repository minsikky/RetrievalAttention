#!/usr/bin/env python3
"""Generate slide-ready plots for attention-efficiency experiments."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("attention_efficiency_result")
BASELINE_DIR = Path(os.environ.get("BASELINE_DIR", str(ROOT / "threeway_cutoffs_graphall_s16_exact_v1")))
SOURCE_NPZ = Path(os.environ.get("SOURCE_NPZ", str(ROOT / "real_qkv_llama31_l16_6838_g16384_q24_graphall_s16.npz")))
OUT_DIR = Path(os.environ.get("OUT_DIR", str(ROOT / "plots")))

DECODES = [500, 1000, 2000, 4000, 8000, 16000]
TARGETS = [0.95, 0.98]
METHODS = {
    "retroinfer": {"label": "RetroInfer", "color": "#2E6F9E", "marker": "o"},
    "retrievalattention": {"label": "RetrievalAttention", "color": "#C44E52", "marker": "s"},
    "hybrid_centroid_graph": {"label": "Hybrid centroid graph", "color": "#55A868", "marker": "^"},
}
FAMILIES = {
    "dense_oracle": {"label": "Oracle", "color": "#222222", "marker": "P"},
    "retroinfer": {"label": "RetroInfer", "color": "#2E6F9E", "marker": "o"},
    "retrievalattention": {"label": "RetrievalAttention", "color": "#C44E52", "marker": "s"},
    "hybrid_centroid_graph": {"label": "Hybrid centroid graph", "color": "#55A868", "marker": "^"},
    "quest_page": {"label": "Quest-page", "color": "#8172B3", "marker": "D"},
    "sparq": {"label": "SparQ", "color": "#CCB974", "marker": "v"},
    "loki": {"label": "Loki", "color": "#64B5CD", "marker": "X"},
    "pqcache": {"label": "PQCache proxy", "color": "#4D908E", "marker": "p"},
    "magicpig": {"label": "MagicPIG", "color": "#DD8452", "marker": "*"},
    "pariskv": {"label": "ParisKV proxy", "color": "#937860", "marker": "h"},
}


def load_baseline_rows() -> dict[tuple[int, str, float], dict]:
    out: dict[tuple[int, str, float], dict] = {}
    for path in sorted(BASELINE_DIR.glob("decode_*/summary.json")):
        rows = json.loads(path.read_text())
        for row in rows:
            out[(int(row["decode_tokens"]), str(row["method"]), float(row["target_mass"]))] = row
    return out


def method_family(method: str) -> str:
    for prefix in ("quest_page", "sparq", "loki", "pqcache", "magicpig", "pariskv"):
        if method.startswith(prefix):
            return prefix
    return method


def available_decodes(rows: dict[tuple[int, str, float], dict]) -> list[int]:
    vals = sorted({decode for decode, _method, _target in rows})
    return vals or DECODES


def best_family_rows(rows: dict[tuple[int, str, float], dict], oracle: dict[tuple[int, float], dict]) -> dict[tuple[int, str, float], dict]:
    grouped: dict[tuple[int, str, float], list[dict]] = {}
    for (decode, method, target), row in rows.items():
        grouped.setdefault((decode, method_family(method), target), []).append(row)
    out: dict[tuple[int, str, float], dict] = {}
    for key, items in grouped.items():
        reached = [x for x in items if float(x.get("reached_rate", 0.0)) >= 1.0]
        pool = reached if reached else sorted(items, key=lambda x: (-float(x.get("mass_mean", 0.0)), float(x.get("estimated_mb_mean", math.inf))))
        out[key] = min(pool, key=lambda x: float(x.get("estimated_mb_mean", math.inf)))
    for (decode, target), row in oracle.items():
        out[(decode, "dense_oracle", target)] = {
            "decode_tokens": decode,
            "method": "dense_oracle",
            "target_mass": target,
            "estimated_mb_mean": row["estimated_mb"],
            "mass_mean": row["mass"],
            "reached_rate": 1.0,
        }
    return out


def static_tokens(position: int, prefix: int = 128, suffix: int = 512) -> list[int]:
    max_tok = int(position)
    seen = set()
    out = []
    for tok in list(range(0, min(prefix, max_tok + 1))) + list(range(max(0, max_tok - suffix + 1), max_tok + 1)):
        if tok not in seen:
            seen.add(tok)
            out.append(tok)
    return out


def dense_oracle_mb_by_decode(decodes: list[int]) -> dict[tuple[int, float], dict]:
    data = np.load(SOURCE_NPZ, allow_pickle=False)
    keys = data["keys"]
    queries = data["queries"]
    positions = data["positions"]
    meta = json.loads(str(data["metadata"].item()))
    input_len = int(meta["input_len"])
    num_heads = int(queries.shape[0])
    num_kv_heads = int(keys.shape[0])
    group_size = max(1, num_heads // num_kv_heads)
    dim = int(keys.shape[-1])
    score_scale = 1.0 / math.sqrt(dim)

    out: dict[tuple[int, float], dict] = {}
    for decode in decodes:
        qidxs = np.where(positions == input_len + decode - 1)[0]
        if qidxs.size == 0:
            continue
        qidxs = qidxs[:4]
        per_target_exact = {target: [] for target in TARGETS}
        per_target_mass = {target: [] for target in TARGETS}

        for qidx in qidxs.tolist():
            pos = int(positions[int(qidx)])
            base = static_tokens(pos)
            base_arr = np.asarray(base, dtype=np.int64)
            dynamic_mask = np.ones((pos + 1,), dtype=bool)
            dynamic_mask[base_arr] = False

            for head in range(num_heads):
                kv_head = min(num_kv_heads - 1, head // group_size)
                q = queries[head, int(qidx)].astype(np.float32, copy=False)
                k = keys[kv_head, : pos + 1].astype(np.float32, copy=False)
                scores = (k @ q) * score_scale
                logits = scores - np.max(scores)
                probs = np.exp(logits).astype(np.float32)
                probs /= max(float(probs.sum()), 1e-20)

                base_mass = float(probs[base_arr].sum()) if base_arr.size else 0.0
                dynamic_probs = probs[dynamic_mask]
                dynamic_sorted = np.sort(dynamic_probs)[::-1]
                dynamic_cumsum = np.cumsum(dynamic_sorted, dtype=np.float64)
                for target in TARGETS:
                    needed = max(0.0, float(target) - base_mass)
                    if needed <= 0.0:
                        dyn_count = 0
                        mass = base_mass
                    else:
                        idx = int(np.searchsorted(dynamic_cumsum, needed, side="left"))
                        dyn_count = min(idx + 1, int(dynamic_sorted.shape[0]))
                        mass = base_mass + (float(dynamic_cumsum[dyn_count - 1]) if dyn_count > 0 else 0.0)
                    per_target_exact[target].append(len(base) + dyn_count)
                    per_target_mass[target].append(mass)

        for target in TARGETS:
            exact = float(np.mean(per_target_exact[target]))
            mb = exact * dim * 4.0 / (1024.0 * 1024.0)
            out[(decode, target)] = {
                "decode": decode,
                "target": target,
                "exact_tokens": exact,
                "estimated_mb": mb,
                "mass": float(np.mean(per_target_mass[target])),
            }
    return out


def setup_axis(ax, decodes: list[int]):
    ax.set_xscale("log", base=2)
    ax.set_xticks(decodes)
    ax.set_xticklabels([str(x) for x in decodes])
    ax.grid(True, which="major", color="#D7D7D7", linewidth=0.8)
    ax.grid(True, which="minor", axis="y", color="#EEEEEE", linewidth=0.5)
    ax.set_xlabel("Decode length")
    ax.set_ylabel("Estimated bytes read per query (MB)")


def plot_baseline(baseline: dict[tuple[int, str, float], dict], decodes: list[int]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5), sharey=True)
    for ax, target in zip(axes, TARGETS):
        for method, style in METHODS.items():
            ys = [baseline[(decode, method, target)]["estimated_mb_mean"] for decode in decodes]
            ax.plot(
                decodes,
                ys,
                label=style["label"],
                color=style["color"],
                marker=style["marker"],
                linewidth=2.5,
                markersize=6,
            )
        ax.set_title(f"Target attention mass = {target:.2f}")
        setup_axis(ax, decodes)
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("Baseline cost: RetroInfer vs RetrievalAttention vs Hybrid", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "baseline_mb_vs_decode.png", dpi=220)
    plt.close(fig)


def plot_oracle_vs_ra(baseline: dict[tuple[int, str, float], dict], oracle: dict[tuple[int, float], dict], decodes: list[int]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5), sharey=True)
    for ax, target in zip(axes, TARGETS):
        oracle_y = [oracle[(decode, target)]["estimated_mb"] for decode in decodes if (decode, target) in oracle]
        xs = [decode for decode in decodes if (decode, target) in oracle]
        ra_y = [baseline[(decode, "retrievalattention", target)]["estimated_mb_mean"] for decode in xs]
        retro_y = [baseline[(decode, "retroinfer", target)]["estimated_mb_mean"] for decode in xs]
        ax.plot(xs, oracle_y, label="Dense oracle (perfect token discovery)", color="#4C72B0", marker="o", linewidth=2.8)
        ax.plot(xs, ra_y, label="RetrievalAttention current", color="#C44E52", marker="s", linewidth=2.8)
        ax.plot(xs, retro_y, label="RetroInfer", color="#8C8C8C", marker="^", linewidth=2.0, linestyle="--")
        ax.set_title(f"Target attention mass = {target:.2f}")
        setup_axis(ax, decodes)
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("RA potential vs current ANNS traversal", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "oracle_vs_ra_mb_vs_decode.png", dpi=220)
    plt.close(fig)


def plot_all_methods(family_rows: dict[tuple[int, str, float], dict], decodes: list[int]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0), sharey=True)
    for ax, target in zip(axes, TARGETS):
        for family, style in FAMILIES.items():
            xs = [decode for decode in decodes if (decode, family, target) in family_rows]
            if not xs:
                continue
            ys = [family_rows[(decode, family, target)]["estimated_mb_mean"] for decode in xs]
            ax.plot(xs, ys, label=style["label"], color=style["color"], marker=style["marker"], linewidth=2.2, markersize=6)
        ax.set_title(f"Target attention mass = {target:.2f}")
        setup_axis(ax, decodes)
    axes[0].legend(frameon=False, loc="upper left", ncol=1)
    fig.suptitle("Algorithmic memory cost by retrieval family", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "all_methods_mb_vs_decode.png", dpi=220)
    plt.close(fig)


def write_csvs(baseline: dict[tuple[int, str, float], dict], oracle: dict[tuple[int, float], dict], family_rows: dict[tuple[int, str, float], dict], decodes: list[int]) -> None:
    with (OUT_DIR / "baseline_mb_vs_decode.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["decode", "target", "method", "estimated_mb", "mass", "reach"])
        writer.writeheader()
        for target in TARGETS:
            for decode in decodes:
                for method in METHODS:
                    if (decode, method, target) not in baseline:
                        continue
                    row = baseline[(decode, method, target)]
                    writer.writerow(
                        {
                            "decode": decode,
                            "target": target,
                            "method": method,
                            "estimated_mb": row["estimated_mb_mean"],
                            "mass": row["mass_mean"],
                            "reach": row["reached_rate"],
                        }
                    )
    with (OUT_DIR / "oracle_vs_ra_mb_vs_decode.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["decode", "target", "method", "estimated_mb", "mass"])
        writer.writeheader()
        for target in TARGETS:
            for decode in decodes:
                if (decode, target) not in oracle:
                    continue
                writer.writerow(
                    {
                        "decode": decode,
                        "target": target,
                        "method": "dense_oracle",
                        "estimated_mb": oracle[(decode, target)]["estimated_mb"],
                        "mass": oracle[(decode, target)]["mass"],
                    }
                )
                row = baseline[(decode, "retrievalattention", target)]
                writer.writerow(
                    {
                        "decode": decode,
                        "target": target,
                        "method": "retrievalattention",
                        "estimated_mb": row["estimated_mb_mean"],
                        "mass": row["mass_mean"],
                    }
                )
    with (OUT_DIR / "all_methods_mb_vs_decode.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["decode", "target", "family", "selected_method", "estimated_mb", "mass", "reach"])
        writer.writeheader()
        for target in TARGETS:
            for decode in decodes:
                for family in FAMILIES:
                    if (decode, family, target) not in family_rows:
                        continue
                    row = family_rows[(decode, family, target)]
                    writer.writerow(
                        {
                            "decode": decode,
                            "target": target,
                            "family": family,
                            "selected_method": row.get("method", family),
                            "estimated_mb": row["estimated_mb_mean"],
                            "mass": row["mass_mean"],
                            "reach": row["reached_rate"],
                        }
                    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    baseline = load_baseline_rows()
    decodes = available_decodes(baseline)
    oracle = dense_oracle_mb_by_decode(decodes)
    family_rows = best_family_rows(baseline, oracle)
    plot_baseline(baseline, decodes)
    plot_oracle_vs_ra(baseline, oracle, decodes)
    plot_all_methods(family_rows, decodes)
    write_csvs(baseline, oracle, family_rows, decodes)
    print(f"wrote {OUT_DIR / 'baseline_mb_vs_decode.png'}")
    print(f"wrote {OUT_DIR / 'oracle_vs_ra_mb_vs_decode.png'}")
    print(f"wrote {OUT_DIR / 'all_methods_mb_vs_decode.png'}")


if __name__ == "__main__":
    main()
