#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path


def _float(row: dict[str, object], key: str, default: float = float("nan")) -> float:
    value = row.get(key, "")
    if value in {"", None}:
        return default
    return float(value)


def _int(row: dict[str, object], key: str, default: int = 0) -> int:
    value = row.get(key, "")
    if value in {"", None}:
        return default
    return int(float(value))


def _page_size(path: Path) -> int:
    args_path = path / "args.json"
    if args_path.exists():
        args = json.loads(args_path.read_text(encoding="utf-8"))
        if "page_size" in args:
            return int(args["page_size"])
    match = re.search(r"ps(\d+)", path.name)
    if not match:
        raise ValueError(f"cannot infer page size from {path}")
    return int(match.group(1))


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
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _mean(values: list[float]) -> float:
    values = [v for v in values if math.isfinite(v)]
    return float(sum(values) / len(values)) if values else float("nan")


def collect(input_root: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    summary_rows: list[dict[str, object]] = []
    by_decode_rows: list[dict[str, object]] = []
    run_dirs: list[Path] = []
    for path in input_root.iterdir():
        if not path.is_dir():
            continue
        if not (path / "summary.json").exists() or not (path / "layer_joint_policy.csv").exists():
            continue
        run_dirs.append(path)
    for run_dir in sorted(run_dirs, key=_page_size):
        layer_csv = run_dir / "layer_joint_policy.csv"
        summary_json = run_dir / "summary.json"
        if not layer_csv.exists() or not summary_json.exists():
            continue
        page_size = _page_size(run_dir)
        layer_rows = _read_csv(layer_csv)
        summary = json.loads(summary_json.read_text(encoding="utf-8"))
        if not layer_rows:
            continue
        for row in layer_rows:
            by_decode_rows.append(
                {
                    "page_size": page_size,
                    "decode_length": _int(row, "decode_length"),
                    "attn_o_proj_relative_L2": _float(row, "attn_o_proj_relative_L2"),
                    "attn_concat_relative_L2": _float(row, "attn_concat_relative_L2"),
                    "mean_head_attention_relative_L2": _float(row, "mean_head_attention_relative_L2"),
                    "mean_logit_relL2": _float(row, "mean_logit_relL2"),
                    "mean_prob_JS": _float(row, "mean_prob_JS"),
                    "mean_prob_TV": _float(row, "mean_prob_TV"),
                    "mean_k_budget": _float(row, "mean_k_budget"),
                    "mean_v_budget": _float(row, "mean_v_budget"),
                    "mean_selected_k_tokens": _float(row, "mean_selected_k_tokens"),
                    "mean_step_MB_per_head": _float(row, "mean_step_MB_per_head"),
                    "max_step_MB_per_head": _float(row, "max_step_MB_per_head"),
                    "source": str(run_dir),
                }
            )
        canonical = None
        for row in summary.get("summary", []):
            if (
                str(row.get("score_proxy_variant", "")) == "baseline"
                and str(row.get("policy", "")) == "k_first_alternating"
                and abs(float(row.get("threshold", 0.0)) - 0.001) < 1e-12
            ):
                canonical = row
                break
        if canonical is None and summary.get("summary"):
            canonical = summary["summary"][0]
        page_layers = [r for r in by_decode_rows if int(r["page_size"]) == page_size]
        summary_rows.append(
            {
                "page_size": page_size,
                "queries": int(canonical.get("queries", len(page_layers))) if canonical else len(page_layers),
                "mean_step_MB_per_head": float(canonical.get("mean_step_MB_per_head", _mean([float(r["mean_step_MB_per_head"]) for r in page_layers]))),
                "max_step_MB_per_head": float(canonical.get("max_step_MB_per_head", max(float(r["max_step_MB_per_head"]) for r in page_layers))),
                "attn_o_proj_relative_L2_mean": float(canonical.get("attn_o_proj_relative_L2_mean", _mean([float(r["attn_o_proj_relative_L2"]) for r in page_layers]))),
                "attn_o_proj_relative_L2_max": float(canonical.get("attn_o_proj_relative_L2_max", max(float(r["attn_o_proj_relative_L2"]) for r in page_layers))),
                "mean_logit_relL2": float(canonical.get("mean_logit_relL2", _mean([float(r["mean_logit_relL2"]) for r in page_layers]))),
                "mean_prob_JS": float(canonical.get("mean_prob_JS", _mean([float(r["mean_prob_JS"]) for r in page_layers]))),
                "mean_prob_TV": float(canonical.get("mean_prob_TV", _mean([float(r["mean_prob_TV"]) for r in page_layers]))),
                "mean_k_budget": float(canonical.get("mean_k_budget", _mean([float(r["mean_k_budget"]) for r in page_layers]))),
                "mean_v_budget": float(canonical.get("mean_v_budget", _mean([float(r["mean_v_budget"]) for r in page_layers]))),
                "elapsed_seconds": float(summary.get("elapsed_seconds", float("nan"))),
                "source": str(run_dir),
            }
        )
    return sorted(summary_rows, key=lambda r: int(r["page_size"])), sorted(
        by_decode_rows, key=lambda r: (int(r["page_size"]), int(r["decode_length"]))
    )


def plot(output_dir: Path, summary_rows: list[dict[str, object]], by_decode_rows: list[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    pages = [int(r["page_size"]) for r in summary_rows]
    mean_mb = [float(r["mean_step_MB_per_head"]) for r in summary_rows]
    max_mb = [float(r["max_step_MB_per_head"]) for r in summary_rows]
    mean_l2 = [float(r["attn_o_proj_relative_L2_mean"]) for r in summary_rows]
    max_l2 = [float(r["attn_o_proj_relative_L2_max"]) for r in summary_rows]

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8), dpi=180)
    axes[0].plot(pages, mean_mb, marker="o", label="mean")
    axes[0].plot(pages, max_mb, marker="s", label="max", alpha=0.75)
    axes[0].set_xlabel("Page size")
    axes[0].set_ylabel("Step MB / head-query")
    axes[0].set_title("Cost vs Page Size")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(frameon=False)
    axes[1].plot(pages, mean_l2, marker="o", label="mean")
    axes[1].plot(pages, max_l2, marker="s", label="max", alpha=0.75)
    axes[1].set_xlabel("Page size")
    axes[1].set_ylabel("o-proj relative L2")
    axes[1].set_title("Quality vs Page Size")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "page_size_vs_cost_quality.png")
    fig.savefig(output_dir / "page_size_vs_cost_quality.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 5.2), dpi=180)
    ax.scatter(mean_mb, mean_l2, s=70, color="#2f6f9f")
    for page, x, y in zip(pages, mean_mb, mean_l2):
        ax.annotate(str(page), (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Mean step MB / head-query")
    ax.set_ylabel("Mean o-proj relative L2")
    ax.set_title("Page-Size Cost/Quality Tradeoff")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "page_size_pareto.png")
    fig.savefig(output_dir / "page_size_pareto.pdf")
    plt.close(fig)

    decode_lengths = sorted({int(r["decode_length"]) for r in by_decode_rows})
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.0), dpi=180)
    cmap = plt.get_cmap("viridis")
    for idx, page in enumerate(pages):
        rows = [r for r in by_decode_rows if int(r["page_size"]) == page]
        color = cmap(idx / max(len(pages) - 1, 1))
        axes[0].plot(
            [int(r["decode_length"]) for r in rows],
            [float(r["mean_step_MB_per_head"]) for r in rows],
            marker="o",
            markersize=3,
            color=color,
            label=str(page),
        )
        axes[1].plot(
            [int(r["decode_length"]) for r in rows],
            [float(r["attn_o_proj_relative_L2"]) for r in rows],
            marker="o",
            markersize=3,
            color=color,
            label=str(page),
        )
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(decode_lengths)
        ax.set_xticklabels([str(x) for x in decode_lengths], rotation=35, ha="right")
        ax.grid(True, alpha=0.3)
    axes[0].set_xlabel("Decode length")
    axes[0].set_ylabel("Step MB / head-query")
    axes[0].set_title("Cost Across Decode Length")
    axes[1].set_xlabel("Decode length")
    axes[1].set_ylabel("o-proj relative L2")
    axes[1].set_title("Quality Across Decode Length")
    axes[1].legend(title="page", fontsize=6.5, title_fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "decode_length_curves_by_page_size.png")
    fig.savefig(output_dir / "decode_length_curves_by_page_size.pdf")
    plt.close(fig)


def write_markdown(output_dir: Path, summary_rows: list[dict[str, object]]) -> None:
    lines = [
        "# Joint K/V Page-Size Sweep",
        "",
        "Current canonical trace policy, varying only `page_size`.",
        "",
        "| page | mean MB/head-q | max MB/head-q | mean o-proj relL2 | max o-proj relL2 | mean K budget | mean V budget |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {page_size} | {mean_step_MB_per_head:.3f} | {max_step_MB_per_head:.3f} | "
            "{attn_o_proj_relative_L2_mean:.6g} | {attn_o_proj_relative_L2_max:.6g} | "
            "{mean_k_budget:.1f} | {mean_v_budget:.1f} |".format(**row)
        )
    output_dir.joinpath("page_size_sweep_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run() -> None:
    parser = argparse.ArgumentParser(description="Aggregate and plot joint K/V page-size sweep outputs.")
    parser.add_argument("--input_root", required=True)
    parser.add_argument("--output_dir", default="")
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir) if args.output_dir else input_root / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows, by_decode_rows = collect(input_root)
    _write_csv(output_dir / "page_size_sweep_summary.csv", summary_rows)
    _write_csv(output_dir / "page_size_sweep_by_decode.csv", by_decode_rows)
    if summary_rows:
        plot(output_dir, summary_rows, by_decode_rows)
        write_markdown(output_dir, summary_rows)
    print(json.dumps({"input_root": str(input_root), "output_dir": str(output_dir), "page_sizes": [r["page_size"] for r in summary_rows]}, indent=2))


if __name__ == "__main__":
    run()
