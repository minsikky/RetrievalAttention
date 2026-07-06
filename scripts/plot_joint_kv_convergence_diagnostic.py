#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _read_csv(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def _f(row: dict[str, object], key: str, default: float = float("nan")) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def _i(row: dict[str, object], key: str, default: int = 0) -> int:
    try:
        return int(float(row.get(key, default)))
    except (TypeError, ValueError):
        return default


def _median_band(rows: list[dict[str, object]], x_key: str, y_key: str) -> tuple[list[int], list[float], list[float], list[float], list[float]]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        y = _f(row, y_key)
        if math.isfinite(y):
            grouped[_i(row, x_key)].append(y)
    xs = sorted(grouped)
    med = [float(np.median(grouped[x])) for x in xs]
    p25 = [float(np.quantile(grouped[x], 0.25)) for x in xs]
    p75 = [float(np.quantile(grouped[x], 0.75)) for x in xs]
    p95 = [float(np.quantile(grouped[x], 0.95)) for x in xs]
    return xs, med, p25, p75, p95


def _plot_policy_aggregate(policy_rows: list[dict[str, object]], out_dir: Path) -> Path:
    fig, axes = plt.subplots(3, 1, figsize=(9.0, 10.0), sharex=True)
    specs = [
        ("relL2_to_dense", "relL2 to dense"),
        ("max_next_delta", "max(next K delta, next V delta)"),
        ("delta_to_max_budget", "delta to max budget"),
    ]
    for ax, (key, ylabel) in zip(axes, specs):
        xs, med, p25, p75, p95 = _median_band(policy_rows, "step", key)
        ax.plot(xs, med, marker="o", linewidth=2.5, label="median")
        ax.fill_between(xs, p25, p75, alpha=0.22, label="p25-p75")
        ax.plot(xs, p95, linestyle="--", linewidth=1.5, label="p95")
        ax.set_yscale("log")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.25)
    threshold = _f(policy_rows[0], "threshold") if policy_rows else float("nan")
    if math.isfinite(threshold):
        axes[1].axhline(threshold, color="red", linestyle=":", linewidth=1.8, label=f"threshold={threshold:g}")
    axes[-1].set_xlabel("Policy step")
    axes[0].set_title("Policy-path convergence aggregate")
    for ax in axes:
        ax.legend(loc="best", fontsize=8, frameon=False)
    fig.tight_layout()
    path = out_dir / "policy_path_convergence_aggregate.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_reliability(policy_rows: list[dict[str, object]], out_dir: Path) -> Path:
    accepted = [r for r in policy_rows if str(r.get("accepted", "")).lower() in {"true", "1"}]
    fig, ax = plt.subplots(figsize=(8.4, 6.4))
    x = np.asarray([_f(r, "max_next_delta") for r in accepted], dtype=np.float64)
    y = np.asarray([_f(r, "delta_to_max_budget") for r in accepted], dtype=np.float64)
    c = np.asarray([_i(r, "decode_length") for r in accepted], dtype=np.float64)
    keep = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    sc = ax.scatter(x[keep], y[keep], c=c[keep], cmap="viridis", s=28, alpha=0.75, edgecolor="none")
    threshold = _f(policy_rows[0], "threshold") if policy_rows else float("nan")
    if math.isfinite(threshold):
        ax.axvline(threshold, color="red", linestyle=":", linewidth=1.8, label=f"threshold={threshold:g}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("stop-time max(next K delta, next V delta)")
    ax.set_ylabel("delta from accepted output to max-budget output")
    ax.set_title("Does small local delta predict stable final output?")
    ax.grid(True, which="both", alpha=0.25)
    if math.isfinite(threshold):
        ax.legend(frameon=False)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("decode length")
    fig.tight_layout()
    path = out_dir / "confidence_reliability_scatter.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_heatmap(layer_rows: list[dict[str, object]], policy_rows: list[dict[str, object]], out_dir: Path, decode_length: int | None, head: int | None) -> Path:
    decodes = sorted({_i(r, "decode_length") for r in layer_rows})
    target_decode = int(decode_length or decodes[-1])
    rows = [r for r in layer_rows if _i(r, "decode_length") == target_decode]
    if not rows:
        raise ValueError(f"no layer grid rows for decode length {target_decode}")
    k_budgets = sorted({_i(r, "k_budget") for r in rows})
    v_budgets = sorted({_i(r, "v_budget") for r in rows})
    z = np.full((len(v_budgets), len(k_budgets)), np.nan, dtype=np.float64)
    for row in rows:
        ki = k_budgets.index(_i(row, "k_budget"))
        vi = v_budgets.index(_i(row, "v_budget"))
        z[vi, ki] = _f(row, "attn_o_proj_relative_L2")
    fig, ax = plt.subplots(figsize=(9.0, 6.8))
    im = ax.imshow(np.log10(np.maximum(z, 1e-8)), origin="lower", aspect="auto", cmap="magma_r")
    ax.set_xticks(range(len(k_budgets)))
    ax.set_xticklabels([str(x) for x in k_budgets], rotation=45, ha="right")
    ax.set_yticks(range(len(v_budgets)))
    ax.set_yticklabels([str(x) for x in v_budgets])
    ax.set_xlabel("K budget")
    ax.set_ylabel("V budget")
    ax.set_title(f"Layer o-proj relL2 surface at decode {target_decode}")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("log10(o-proj relL2)")

    path_rows = [r for r in policy_rows if _i(r, "decode_length") == target_decode]
    if head is not None:
        path_rows = [r for r in path_rows if _i(r, "head") == int(head)]
    elif path_rows:
        first_head = _i(path_rows[0], "head")
        path_rows = [r for r in path_rows if _i(r, "head") == first_head]
    path_rows.sort(key=lambda r: _i(r, "step"))
    xs = [k_budgets.index(_i(r, "k_budget")) for r in path_rows if _i(r, "k_budget") in k_budgets and _i(r, "v_budget") in v_budgets]
    ys = [v_budgets.index(_i(r, "v_budget")) for r in path_rows if _i(r, "k_budget") in k_budgets and _i(r, "v_budget") in v_budgets]
    if xs and ys:
        ax.plot(xs, ys, color="white", marker="o", linewidth=2.0, markersize=5, label=f"policy path, head {_i(path_rows[0], 'head')}")
        ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    path = out_dir / f"heatmap_decode{target_decode}_layer_uniform.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_policy_examples(policy_rows: list[dict[str, object]], out_dir: Path, head: int | None) -> Path:
    decodes = sorted({_i(r, "decode_length") for r in policy_rows})
    targets = []
    for target in [decodes[0], 8000, 32000, decodes[-1]]:
        if target in decodes and target not in targets:
            targets.append(target)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2))
    for decode in targets:
        rows = [r for r in policy_rows if _i(r, "decode_length") == decode]
        if head is not None:
            rows = [r for r in rows if _i(r, "head") == int(head)]
        elif rows:
            rows = [r for r in rows if _i(r, "head") == _i(rows[0], "head")]
        rows.sort(key=lambda r: _i(r, "step"))
        if not rows:
            continue
        x = [_f(r, "step_MB_per_head") for r in rows]
        axes[0].plot(x, [_f(r, "relL2_to_dense") for r in rows], marker="o", linewidth=2.0, label=str(decode))
        axes[1].plot(x, [_f(r, "max_next_delta") for r in rows], marker="o", linewidth=2.0, label=str(decode))
    for ax in axes:
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.25)
        ax.set_xlabel("Step MB/head")
    axes[0].set_ylabel("relL2 to dense")
    axes[1].set_ylabel("max next-budget delta")
    axes[0].set_title("Example policy convergence")
    axes[1].set_title("Step-to-step output change")
    threshold = _f(policy_rows[0], "threshold") if policy_rows else float("nan")
    if math.isfinite(threshold):
        axes[1].axhline(threshold, color="red", linestyle=":", linewidth=1.8)
    axes[1].legend(title="decode", frameon=False)
    fig.tight_layout()
    path = out_dir / "policy_path_examples.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return path


def run() -> None:
    parser = argparse.ArgumentParser(description="Plot joint K/V convergence diagnostic outputs.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--decode_length", type=int, default=0)
    parser.add_argument("--head", type=int, default=-1)
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    layer_rows = _read_csv(input_dir / "layer_uniform_budget_grid.csv")
    policy_rows = _read_csv(input_dir / "policy_path_per_head.csv")
    per_head_rows = _read_csv(input_dir / "per_head_budget_grid.csv")
    head = None if int(args.head) < 0 else int(args.head)
    paths = [
        _plot_policy_aggregate(policy_rows, output_dir),
        _plot_reliability(policy_rows, output_dir),
        _plot_heatmap(layer_rows, policy_rows, output_dir, int(args.decode_length) or None, head),
        _plot_policy_examples(policy_rows, output_dir, head),
    ]

    accepted = [r for r in policy_rows if str(r.get("accepted", "")).lower() in {"true", "1"}]
    summary = {
        "input_dir": str(input_dir),
        "plots": [str(p) for p in paths],
        "accepted_count": len(accepted),
        "accepted_mean_step_MB_per_head": float(np.mean([_f(r, "step_MB_per_head") for r in accepted])) if accepted else float("nan"),
        "accepted_mean_relL2_to_dense": float(np.mean([_f(r, "relL2_to_dense") for r in accepted])) if accepted else float("nan"),
        "accepted_mean_delta_to_max_budget": float(np.mean([_f(r, "delta_to_max_budget") for r in accepted])) if accepted else float("nan"),
        "accepted_max_delta_to_max_budget": float(np.max([_f(r, "delta_to_max_budget") for r in accepted])) if accepted else float("nan"),
        "per_head_grid_rows": len(per_head_rows),
        "layer_grid_rows": len(layer_rows),
        "policy_path_rows": len(policy_rows),
    }
    (output_dir / "plot_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    for path in paths:
        print(path)
    print(output_dir / "plot_summary.json")


if __name__ == "__main__":
    run()
