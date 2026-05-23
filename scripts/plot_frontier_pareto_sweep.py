#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METRICS = {
    "o_proj_relL2": {
        "label": "Mean o-proj relative L2",
        "title": "MB vs Output Error",
        "joint_mean": "attn_o_proj_relative_L2_mean",
        "joint_max": "attn_o_proj_relative_L2_max",
        "compression_mean": "mean_o_proj_relL2",
        "compression_max": "max_o_proj_relL2",
    },
    "prob_JS": {
        "label": "Mean attention-probability JS divergence",
        "title": "MB vs Attention Probability Divergence",
        "joint_mean": "mean_prob_JS",
        "joint_max": "max_prob_JS",
        "compression_mean": "mean_prob_JS",
        "compression_max": "max_prob_JS",
    },
    "prob_KL": {
        "label": "Mean KL(dense P || approx P)",
        "title": "MB vs Attention Probability KL",
        "joint_mean": "mean_prob_KL_dense_to_approx",
        "joint_max": "max_prob_KL_dense_to_approx",
        "compression_mean": "mean_prob_KL_dense_to_approx",
        "compression_max": "max_prob_KL_dense_to_approx",
    },
    "prob_TV": {
        "label": "Mean attention-probability total variation",
        "title": "MB vs Attention Probability TV",
        "joint_mean": "mean_prob_TV",
        "joint_max": "max_prob_TV",
        "compression_mean": "mean_prob_TV",
        "compression_max": "max_prob_TV",
    },
    "logit_relL2": {
        "label": "Mean logit relative L2",
        "title": "MB vs Logit Error",
        "joint_mean": "mean_logit_relL2",
        "joint_max": "max_logit_relL2",
        "compression_mean": "mean_logit_relL2",
        "compression_max": "max_logit_relL2",
    },
}


def _read_float(row: dict[str, object], key: str) -> float | None:
    value = row.get(key, "")
    if value in {"", None}:
        return None
    return float(value)


def read_joint_summary(path: Path, label: str, metric: str) -> list[dict[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    cfg = METRICS[metric]
    rows = []
    for row in data.get("summary", []):
        value = _read_float(row, cfg["joint_mean"])
        if value is None:
            continue
        max_value = _read_float(row, cfg["joint_max"])
        variant = str(row.get("score_proxy_variant", "baseline"))
        variant_label = "" if variant in {"", "baseline"} else f"_{variant}"
        rows.append(
            {
                "method": f"frontier_{label}{variant_label}_tau{float(row['threshold']):g}",
                "family": "frontier_sweep",
                "sweep": f"{label}{variant_label}",
                "threshold": float(row["threshold"]),
                "MB": float(row["mean_step_MB_per_head"]),
                "relL2": float(value),
                "max_relL2": float(max_value if max_value is not None else value),
                "mean_k_budget": float(row.get("mean_k_budget", 0.0)),
                "mean_v_budget": float(row.get("mean_v_budget", 0.0)),
                "source": str(path),
            }
        )
    return rows


def read_compression_points(path: Path, metric: str, keep_families: set[str] | None = None) -> list[dict[str, object]]:
    if not path.exists():
        return []
    cfg = METRICS[metric]
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = []
    for row in rows:
        family = str(row.get("family", ""))
        if keep_families is not None and family not in keep_families:
            continue
        if metric.startswith("prob_"):
            comparable = _read_float(row, "mean_token_probability_comparable")
            if comparable is not None and comparable < 0.999:
                continue
        if metric.startswith("logit_"):
            comparable = _read_float(row, "mean_distribution_comparable")
            if comparable is not None and comparable < 0.999:
                continue
        value = _read_float(row, cfg["compression_mean"])
        if value is None and metric == "o_proj_relL2":
            value = _read_float(row, "mean_attention_relL2")
        if value is None:
            continue
        max_value = _read_float(row, cfg["compression_max"])
        if max_value is None and metric == "o_proj_relL2":
            max_value = _read_float(row, "max_attention_relL2")
        out.append(
            {
                "method": str(row["method"]),
                "family": family,
                "sweep": "compression",
                "threshold": "",
                "MB": float(row["mean_step_MB_per_head_query"]),
                "relL2": float(value),
                "max_relL2": float(max_value if max_value is not None else value),
                "mean_k_budget": "",
                "mean_v_budget": "",
                "source": str(path),
            }
        )
    return out


def read_points_csv(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = []
    for row in rows:
        out.append(
            {
                "method": str(row["method"]),
                "family": str(row["family"]),
                "sweep": row.get("sweep", ""),
                "threshold": row.get("threshold", ""),
                "MB": float(row["MB"]),
                "relL2": float(row["relL2"]),
                "max_relL2": float(row["max_relL2"]),
                "mean_k_budget": row.get("mean_k_budget", ""),
                "mean_v_budget": row.get("mean_v_budget", ""),
                "source": row.get("source", str(path)),
            }
        )
    return out


def pareto(points: list[dict[str, object]]) -> list[dict[str, object]]:
    ordered = sorted(points, key=lambda p: (float(p["MB"]), float(p["relL2"])))
    out = []
    best_rel = float("inf")
    for point in ordered:
        rel = float(point["relL2"])
        if rel < best_rel:
            out.append(point)
            best_rel = rel
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = ["method", "family", "sweep", "threshold", "MB", "relL2", "max_relL2", "mean_k_budget", "mean_v_budget", "source"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_md(path: Path, rows: list[dict[str, object]], frontier_rows: list[dict[str, object]], *, metric_label: str) -> None:
    lines = [
        "# Frontier Pareto Sweep",
        "",
        f"All points use deployable knobs. Metric: `{metric_label}`. The metric is measured offline against dense attention and is not used as a stopping rule.",
        "",
        "## Non-Dominated Points",
        "",
        f"| Method | MB/head-query | {metric_label} | max metric |",
        "|---|---:|---:|---:|",
    ]
    for row in frontier_rows:
        lines.append(
            f"| {row['method']} | {float(row['MB']):.3f} | {float(row['relL2']):.6g} | {float(row['max_relL2']):.6g} |"
        )
    lines.extend(
        [
            "",
            "## All Points",
            "",
            f"| Method | Family | MB/head-query | {metric_label} | max metric |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in sorted(rows, key=lambda p: (float(p["MB"]), float(p["relL2"]))):
        lines.append(
            f"| {row['method']} | {row['family']} | {float(row['MB']):.3f} | {float(row['relL2']):.6g} | {float(row['max_relL2']):.6g} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot(
    path_png: Path,
    path_pdf: Path,
    rows: list[dict[str, object]],
    frontier_rows: list[dict[str, object]],
    *,
    xmax: float | None,
    point_labels: bool,
    metric_label: str,
    metric_title: str,
) -> None:
    import matplotlib.pyplot as plt

    families = sorted({str(row["family"]) for row in rows})
    markers = ["o", "s", "^", "D", "P", "X", "v", "*", "<", ">", "h", "8", "p", "H", "d"]
    color_maps = ["tab20", "tab20b", "tab20c"]
    palette = []
    for cmap_name in color_maps:
        cmap = plt.get_cmap(cmap_name)
        palette.extend(cmap(i) for i in range(cmap.N))
    special_styles = {
        "dense": {"color": "0.35", "marker": "X", "s": 95, "alpha": 0.9},
        "frontier_reference": {"color": "black", "marker": "*", "s": 140, "alpha": 0.95},
        "frontier_sweep": {"color": "black", "marker": "o", "s": 72, "alpha": 0.72},
    }
    fig, ax = plt.subplots(figsize=(12.8, 5.8), dpi=180)
    for idx, family in enumerate(families):
        group = [row for row in rows if str(row["family"]) == family]
        style = special_styles.get(
            family,
            {
                "color": palette[idx % len(palette)],
                "marker": markers[(idx // len(palette) + idx) % len(markers)],
                "s": 70,
                "alpha": 0.86,
            },
        )
        ax.scatter(
            [float(row["MB"]) for row in group],
            [float(row["relL2"]) for row in group],
            marker=style["marker"],
            s=style["s"],
            alpha=style["alpha"],
            color=style["color"],
            edgecolors="white",
            linewidths=0.55,
            label=family,
        )
    if frontier_rows:
        frontier_sorted = sorted(frontier_rows, key=lambda row: float(row["MB"]))
        ax.plot(
            [float(row["MB"]) for row in frontier_sorted],
            [float(row["relL2"]) for row in frontier_sorted],
            color="black",
            linewidth=1.8,
            alpha=0.8,
            label="non-dominated frontier",
        )
    if point_labels:
        for row in rows:
            mb = float(row["MB"])
            rel = float(row["relL2"])
            if xmax is not None and mb > float(xmax):
                continue
            if str(row["family"]) == "frontier_sweep" and rel > 0.15:
                continue
            if str(row["family"]) in {"dense"}:
                continue
            ax.annotate(str(row["method"]).replace("frontier_", ""), (mb, rel), xytext=(4, 5), textcoords="offset points", fontsize=6.5)
    ax.set_xlabel("Step MB per head-query")
    ax.set_ylabel(metric_label)
    ax.set_title(f"Deployable Frontier Sweep: {metric_title}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=6.6, loc="center left", bbox_to_anchor=(1.01, 0.5), ncol=2, frameon=False)
    ax.set_xlim(left=0.0, right=xmax)
    ax.set_ylim(bottom=0.0)
    fig.tight_layout()
    fig.savefig(path_png)
    fig.savefig(path_pdf)
    plt.close(fig)


def run() -> None:
    parser = argparse.ArgumentParser(description="Merge and plot frontier Pareto sweep outputs.")
    parser.add_argument("--points_csv", default=None, help="Optional existing frontier_pareto_points.csv to replot.")
    parser.add_argument("--joint_summaries", default="", help="Comma-separated label:path entries.")
    parser.add_argument("--compression_summary_csv", default="attention_efficiency_result/kv_compression_rel_l2_20260522/kvcomp_full_pq_20260522/summary.csv")
    parser.add_argument(
        "--compression_families",
        default="dense,pq_like",
        help="Comma-separated compression families to keep. Use 'all' to keep every family.",
    )
    parser.add_argument("--xmax", type=float, default=None)
    parser.add_argument("--y_metric", choices=sorted(METRICS), default="o_proj_relL2")
    parser.add_argument("--point_labels", action="store_true", help="Annotate individual points. Disabled by default to keep plots readable.")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.points_csv:
        rows = read_points_csv(Path(args.points_csv))
    else:
        rows = []
        for item in [part.strip() for part in str(args.joint_summaries).split(",") if part.strip()]:
            label, path = item.split(":", 1)
            rows.extend(read_joint_summary(Path(path), label, args.y_metric))
        families_arg = str(args.compression_families).strip().lower()
        keep_families = None if families_arg == "all" else {part.strip() for part in str(args.compression_families).split(",") if part.strip()}
        for csv_path in [part.strip() for part in str(args.compression_summary_csv).split(",") if part.strip()]:
            rows.extend(read_compression_points(Path(csv_path), args.y_metric, keep_families=keep_families))
        has_frontier_summary = any(str(row.get("family", "")) == "frontier_sweep" for row in rows)
        if args.y_metric == "o_proj_relL2" and not has_frontier_summary:
            rows.append(
                {
                    "method": "current_reference_tau0.001",
                    "family": "frontier_reference",
                    "sweep": "reference",
                    "threshold": 0.001,
                    "MB": 4.779294550418854,
                    "relL2": 0.0011178374271938424,
                    "max_relL2": 0.002082,
                    "mean_k_budget": "",
                    "mean_v_budget": "",
                    "source": "notes/current_status.md",
                }
            )
    dedup: dict[str, dict[str, object]] = {}
    for row in rows:
        dedup[str(row["method"])] = row
    rows = list(dedup.values())
    frontier_rows = pareto(rows)
    metric_cfg = METRICS[args.y_metric]
    write_csv(out_dir / "frontier_pareto_points.csv", rows)
    write_csv(out_dir / "frontier_pareto_nondominated.csv", frontier_rows)
    write_md(out_dir / "frontier_pareto_summary.md", rows, frontier_rows, metric_label=str(metric_cfg["label"]))
    plot(
        out_dir / "frontier_pareto.png",
        out_dir / "frontier_pareto.pdf",
        rows,
        frontier_rows,
        xmax=args.xmax,
        point_labels=args.point_labels,
        metric_label=str(metric_cfg["label"]),
        metric_title=str(metric_cfg["title"]),
    )
    print(json.dumps({"output_dir": str(out_dir), "points": len(rows), "nondominated": len(frontier_rows)}, indent=2))


if __name__ == "__main__":
    run()
