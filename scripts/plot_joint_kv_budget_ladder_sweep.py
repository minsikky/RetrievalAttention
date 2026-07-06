#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _mean(xs: list[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else float("nan")


def collect(input_root: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    summary_rows: list[dict[str, object]] = []
    decode_rows: list[dict[str, object]] = []
    for run_dir in sorted(p for p in input_root.iterdir() if p.is_dir()):
        summary_path = run_dir / "summary.json"
        layer_path = run_dir / "layer_joint_policy.csv"
        if not summary_path.exists() or not layer_path.exists():
            continue
        label = run_dir.name
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        canonical_rows = []
        for row in summary.get("summary", []):
            if (
                str(row.get("score_proxy_variant", "")) == "baseline"
                and str(row.get("policy", "")) == "k_first_alternating"
                and abs(float(row.get("threshold", 0.0)) - 0.001) < 1e-12
            ):
                canonical_rows.append(row)
        if not canonical_rows:
            continue
        label_by_start: dict[str, str] = {}
        multi_start = len(canonical_rows) > 1
        for canonical in canonical_rows:
            start_strategy = str(canonical.get("start_strategy", "min"))
            row_label = f"{label}:{start_strategy}" if multi_start else label
            label_by_start[start_strategy] = row_label
            summary_rows.append(
                {
                    "label": row_label,
                    "run": label,
                    "start_strategy": start_strategy,
                    "queries": int(canonical["queries"]),
                    "mean_step_MB_per_head": float(canonical["mean_step_MB_per_head"]),
                    "max_step_MB_per_head": float(canonical["max_step_MB_per_head"]),
                    "attn_o_proj_relative_L2_mean": float(canonical["attn_o_proj_relative_L2_mean"]),
                    "attn_o_proj_relative_L2_max": float(canonical["attn_o_proj_relative_L2_max"]),
                    "mean_k_budget": float(canonical["mean_k_budget"]),
                    "mean_v_budget": float(canonical["mean_v_budget"]),
                    "mean_start_k_budget": float(canonical.get("mean_start_k_budget", 0.0)),
                    "mean_start_v_budget": float(canonical.get("mean_start_v_budget", 0.0)),
                    "mean_iterations": float(canonical["mean_iterations"]),
                    "elapsed_seconds": float(summary.get("elapsed_seconds", float("nan"))),
                    "source": str(run_dir),
                }
            )
        layers = _read_csv(layer_path)
        by_decode: dict[tuple[str, int], list[dict[str, str]]] = {}
        for row in layers:
            if (
                str(row.get("score_proxy_variant", "")) == "baseline"
                and str(row.get("policy", "")) == "k_first_alternating"
                and abs(float(row.get("threshold", 0.0)) - 0.001) < 1e-12
            ):
                start_strategy = str(row.get("start_strategy", "min"))
                by_decode.setdefault((start_strategy, int(float(row["decode_length"]))), []).append(row)
        for (start_strategy, decode), rows in sorted(by_decode.items(), key=lambda item: (item[0][0], item[0][1])):
            decode_rows.append(
                {
                    "label": label_by_start.get(start_strategy, label),
                    "run": label,
                    "start_strategy": start_strategy,
                    "decode_length": int(decode),
                    "mean_step_MB_per_head": _mean([float(r["mean_step_MB_per_head"]) for r in rows]),
                    "max_step_MB_per_head": max(float(r["max_step_MB_per_head"]) for r in rows),
                    "attn_o_proj_relative_L2": _mean([float(r["attn_o_proj_relative_L2"]) for r in rows]),
                    "attn_concat_relative_L2": _mean([float(r["attn_concat_relative_L2"]) for r in rows]),
                    "mean_k_budget": _mean([float(r["mean_k_budget"]) for r in rows]),
                    "mean_v_budget": _mean([float(r["mean_v_budget"]) for r in rows]),
                    "mean_start_k_budget": _mean([float(r.get("mean_start_k_budget", 0.0)) for r in rows]),
                    "mean_start_v_budget": _mean([float(r.get("mean_start_v_budget", 0.0)) for r in rows]),
                    "mean_iterations": _mean([float(r["mean_iterations"]) for r in rows]),
                    "source": str(run_dir),
                }
            )
    return summary_rows, decode_rows


def plot(output_dir: Path, summary_rows: list[dict[str, object]], decode_rows: list[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    labels = [str(r["label"]) for r in summary_rows]
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.6), dpi=180)
    axes[0].bar(labels, [float(r["mean_step_MB_per_head"]) for r in summary_rows], color="#4c78a8")
    axes[0].set_ylabel("Mean step MB / head-query")
    axes[0].set_title("Cost by Budget Ladder")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[1].bar(labels, [float(r["attn_o_proj_relative_L2_mean"]) for r in summary_rows], color="#f58518")
    axes[1].set_ylabel("Mean o-proj relative L2")
    axes[1].set_title("Output Error by Budget Ladder")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "budget_ladder_summary.png")
    fig.savefig(output_dir / "budget_ladder_summary.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.0), dpi=180)
    for label in sorted({str(r["label"]) for r in decode_rows}):
        rows = [r for r in decode_rows if str(r["label"]) == label]
        rows.sort(key=lambda r: int(r["decode_length"]))
        xs = [int(r["decode_length"]) for r in rows]
        axes[0].plot(xs, [float(r["mean_step_MB_per_head"]) for r in rows], marker="o", label=label)
        axes[1].plot(xs, [float(r["attn_o_proj_relative_L2"]) for r in rows], marker="o", label=label)
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Decode length")
    axes[0].set_ylabel("Mean step MB / head-query")
    axes[0].set_title("Cost Across Decode Length")
    axes[1].set_ylabel("Mean o-proj relative L2")
    axes[1].set_title("Output Error Across Decode Length")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "budget_ladder_decode_curves.png")
    fig.savefig(output_dir / "budget_ladder_decode_curves.pdf")
    plt.close(fig)


def write_md(output_dir: Path, summary_rows: list[dict[str, object]]) -> None:
    lines = [
        "# Joint K/V Budget Ladder Sweep",
        "",
        "| ladder | mean MB/head-q | max MB/head-q | mean o-proj relL2 | max o-proj relL2 | start K | start V | mean K | mean V | mean iterations |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(summary_rows, key=lambda r: float(r["mean_step_MB_per_head"])):
        lines.append(
            "| {label} | {mean_step_MB_per_head:.3f} | {max_step_MB_per_head:.3f} | "
            "{attn_o_proj_relative_L2_mean:.6g} | {attn_o_proj_relative_L2_max:.6g} | "
            "{mean_start_k_budget:.1f} | {mean_start_v_budget:.1f} | "
            "{mean_k_budget:.1f} | {mean_v_budget:.1f} | {mean_iterations:.2f} |".format(**row)
        )
    (output_dir / "budget_ladder_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run() -> None:
    parser = argparse.ArgumentParser(description="Aggregate and plot joint K/V budget ladder sweep outputs.")
    parser.add_argument("--input_root", required=True)
    parser.add_argument("--output_dir", default="")
    args = parser.parse_args()
    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir) if args.output_dir else input_root / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows, decode_rows = collect(input_root)
    _write_csv(output_dir / "budget_ladder_summary.csv", summary_rows)
    _write_csv(output_dir / "budget_ladder_by_decode.csv", decode_rows)
    if summary_rows:
        write_md(output_dir, summary_rows)
        plot(output_dir, summary_rows, decode_rows)
    print(json.dumps({"output_dir": str(output_dir), "ladders": [r["label"] for r in summary_rows]}, indent=2))


if __name__ == "__main__":
    run()
