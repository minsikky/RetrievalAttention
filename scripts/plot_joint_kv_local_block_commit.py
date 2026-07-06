#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
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


def f(row: dict[str, object], key: str, default: float = float("nan")) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def pareto(rows: list[dict[str, object]], *, mb_key: str, err_key: str) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    best = float("inf")
    for row in sorted(rows, key=lambda r: (f(r, mb_key), f(r, err_key))):
        err = f(row, err_key)
        if err < best:
            out.append(row)
            best = err
    return out


def collect(input_root: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    summary_rows: list[dict[str, object]] = []
    decode_rows: list[dict[str, object]] = []
    if (input_root / "summary.json").exists():
        run_dirs = [input_root]
    else:
        run_dirs = sorted(p for p in input_root.iterdir() if p.is_dir())
    for run_dir in run_dirs:
        summary_path = run_dir / "summary.json"
        layer_path = run_dir / "layer_joint_policy.csv"
        if not summary_path.exists() or not layer_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for row in summary.get("summary", []):
            if str(row.get("score_proxy_variant", "")) != "baseline":
                continue
            if str(row.get("policy", "")) != "k_first_alternating":
                continue
            summary_rows.append(
                {
                    "run": run_dir.name,
                    "v_selection_rule": str(row.get("v_selection_rule", "")),
                    "threshold": f(row, "threshold"),
                    "queries": int(row.get("queries", 0)),
                    "mean_step_MB_per_head": f(row, "mean_step_MB_per_head"),
                    "mean_step_MB_no_v_state_per_head": f(row, "mean_step_MB_no_v_state_per_head"),
                    "mean_step_MB_with_v_state_per_head": f(row, "mean_step_MB_with_v_state_per_head"),
                    "mean_v_selection_state_MB": f(row, "mean_v_selection_state_MB"),
                    "max_step_MB_per_head": f(row, "max_step_MB_per_head"),
                    "attn_o_proj_relative_L2_mean": f(row, "attn_o_proj_relative_L2_mean"),
                    "attn_o_proj_relative_L2_max": f(row, "attn_o_proj_relative_L2_max"),
                    "attn_o_proj_relative_L2_p95": f(row, "attn_o_proj_relative_L2_p95"),
                    "mean_logit_relL2": f(row, "mean_logit_relL2"),
                    "mean_prob_JS": f(row, "mean_prob_JS"),
                    "mean_k_budget": f(row, "mean_k_budget"),
                    "mean_v_budget": f(row, "mean_v_budget"),
                    "mean_v_exact_reads": f(row, "mean_v_exact_reads"),
                    "mean_iterations": f(row, "mean_iterations"),
                    "source": str(run_dir),
                }
            )
        for row in read_csv(layer_path):
            if str(row.get("score_proxy_variant", "")) != "baseline":
                continue
            if str(row.get("policy", "")) != "k_first_alternating":
                continue
            decode_rows.append(
                {
                    "run": run_dir.name,
                    "v_selection_rule": str(row.get("v_selection_rule", "")),
                    "threshold": float(row.get("threshold", "nan")),
                    "decode_length": int(float(row.get("decode_length", "0"))),
                    "mean_step_MB_per_head": float(row.get("mean_step_MB_per_head", "nan")),
                    "mean_step_MB_no_v_state_per_head": float(row.get("mean_step_MB_no_v_state_per_head", "nan")),
                    "mean_step_MB_with_v_state_per_head": float(row.get("mean_step_MB_with_v_state_per_head", "nan")),
                    "mean_v_selection_state_MB": float(row.get("mean_v_selection_state_MB", "nan")),
                    "attn_o_proj_relative_L2": float(row.get("attn_o_proj_relative_L2", "nan")),
                    "attn_concat_relative_L2": float(row.get("attn_concat_relative_L2", "nan")),
                    "mean_k_budget": float(row.get("mean_k_budget", "nan")),
                    "mean_v_budget": float(row.get("mean_v_budget", "nan")),
                    "mean_v_exact_reads": float(row.get("mean_v_exact_reads", row.get("mean_v_budget", "nan"))),
                    "mean_iterations": float(row.get("mean_iterations", "nan")),
                    "source": str(run_dir),
                }
            )
    return summary_rows, decode_rows


def plot_pareto(
    output_dir: Path,
    rows: list[dict[str, object]],
    *,
    mb_key: str,
    filename: str,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    rules = sorted({str(row["v_selection_rule"]) for row in rows})
    fig, ax = plt.subplots(figsize=(8.2, 5.4), dpi=180)
    for rule in rules:
        rule_rows = [row for row in rows if str(row["v_selection_rule"]) == rule]
        rule_rows = [row for row in rule_rows if math.isfinite(f(row, mb_key)) and math.isfinite(f(row, "attn_o_proj_relative_L2_mean"))]
        if not rule_rows:
            continue
        rule_rows.sort(key=lambda r: f(r, mb_key))
        ax.scatter(
            [f(row, mb_key) for row in rule_rows],
            [f(row, "attn_o_proj_relative_L2_mean") for row in rule_rows],
            s=18,
            alpha=0.35,
        )
        frontier = pareto(rule_rows, mb_key=mb_key, err_key="attn_o_proj_relative_L2_mean")
        ax.plot(
            [f(row, mb_key) for row in frontier],
            [f(row, "attn_o_proj_relative_L2_mean") for row in frontier],
            marker="o",
            linewidth=2.0,
            label=rule,
        )
    ax.set_xlabel("Mean step MB / head-query")
    ax.set_ylabel("Mean o-proj relative L2")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / f"{filename}.png")
    fig.savefig(output_dir / f"{filename}.pdf")
    plt.close(fig)


def plot_decode_curves(
    output_dir: Path,
    decode_rows: list[dict[str, object]],
    *,
    threshold: float,
    mb_key: str,
) -> None:
    import matplotlib.pyplot as plt

    rows = [row for row in decode_rows if abs(f(row, "threshold") - float(threshold)) < 1e-12]
    if not rows:
        return
    rules = sorted({str(row["v_selection_rule"]) for row in rows})
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), dpi=180)
    for rule in rules:
        rule_rows = [row for row in rows if str(row["v_selection_rule"]) == rule]
        rule_rows.sort(key=lambda r: int(r["decode_length"]))
        xs = [int(row["decode_length"]) for row in rule_rows]
        axes[0].plot(xs, [f(row, mb_key) for row in rule_rows], marker="o", label=rule)
        axes[1].plot(xs, [f(row, "attn_o_proj_relative_L2") for row in rule_rows], marker="o", label=rule)
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Decode length")
    axes[0].set_ylabel("Mean step MB / head-query")
    axes[0].set_title(f"Cost at threshold={threshold:g}")
    axes[1].set_ylabel("o-proj relative L2")
    axes[1].set_yscale("log")
    axes[1].set_title(f"Error at threshold={threshold:g}")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    safe_threshold = str(threshold).replace(".", "p")
    fig.savefig(output_dir / f"local_block_by_decode_tau{safe_threshold}.png")
    fig.savefig(output_dir / f"local_block_by_decode_tau{safe_threshold}.pdf")
    plt.close(fig)


def write_md(output_dir: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# Exact-V Selection Sweep",
        "",
        "Rows below are the cheapest point per rule satisfying common mean o-proj relL2 targets.",
        "",
        "| target relL2 | V rule | MB/head-q | no-state MB | state MB | mean relL2 | max relL2 | mean K | active V | exact V reads | threshold |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for target in [0.0005, 0.001, 0.0015, 0.002, 0.003, 0.005, 0.01]:
        for rule in sorted({str(row["v_selection_rule"]) for row in rows}):
            candidates = [
                row
                for row in rows
                if str(row["v_selection_rule"]) == rule
                and f(row, "attn_o_proj_relative_L2_mean") <= target
            ]
            if not candidates:
                continue
            best = min(candidates, key=lambda r: f(r, "mean_step_MB_per_head"))
            lines.append(
                "| {target:.4g} | {v_selection_rule} | {mean_step_MB_per_head:.3f} | "
                "{mean_step_MB_no_v_state_per_head:.3f} | {mean_v_selection_state_MB:.3f} | "
                "{attn_o_proj_relative_L2_mean:.6g} | {attn_o_proj_relative_L2_max:.6g} | "
                "{mean_k_budget:.1f} | {mean_v_budget:.1f} | {mean_v_exact_reads:.1f} | {threshold:.5g} |".format(
                    target=target, **best
                )
            )
    (output_dir / "local_block_commit_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot local block-commit exact-V selection sweeps.")
    parser.add_argument("--input_root", required=True)
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--decode_threshold", type=float, default=0.002)
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir) if args.output_dir else input_root / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, decode_rows = collect(input_root)
    write_csv(output_dir / "local_block_commit_summary.csv", rows)
    write_csv(output_dir / "local_block_commit_by_decode.csv", decode_rows)
    if rows:
        write_md(output_dir, rows)
        plot_pareto(
            output_dir,
            rows,
            mb_key="mean_step_MB_per_head",
            filename="local_block_commit_pareto_state_inclusive",
            title="Exact-V Selection Pareto",
        )
        plot_pareto(
            output_dir,
            rows,
            mb_key="mean_step_MB_no_v_state_per_head",
            filename="local_block_commit_pareto_no_state",
            title="Exact-V Selection Pareto: No Survivor-State Charge",
        )
        plot_decode_curves(
            output_dir,
            decode_rows,
            threshold=float(args.decode_threshold),
            mb_key="mean_step_MB_per_head",
        )
    print(json.dumps({"output_dir": str(output_dir), "rows": len(rows), "decode_rows": len(decode_rows)}, indent=2))


if __name__ == "__main__":
    main()
