#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _float(row: dict[str, str], key: str, default: float = float("nan")) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    return float(value)


def read_compression_points(summary_csv: Path) -> list[dict[str, object]]:
    with summary_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    points = []
    for row in rows:
        method = str(row.get("method", ""))
        if not method:
            continue
        rel = _float(row, "mean_o_proj_relL2", _float(row, "mean_attention_relL2"))
        mb = _float(row, "mean_step_MB_per_head_query")
        points.append(
            {
                "method": method,
                "family": row.get("family", ""),
                "step_MB_per_head_query": mb,
                "relL2": rel,
                "max_relL2": _float(row, "max_o_proj_relL2", _float(row, "max_attention_relL2")),
                "source": str(summary_csv),
            }
        )
    return points


def read_existing_points(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    points = []
    for row in rows:
        label = row.get("label") or row.get("method") or row.get("name")
        mb = row.get("MB") or row.get("mb") or row.get("step_MB_per_head_query") or row.get("x")
        rel = row.get("relL2") or row.get("rell2") or row.get("o_proj_relL2") or row.get("y")
        if not label or mb in {None, ""} or rel in {None, ""}:
            continue
        points.append(
            {
                "method": str(label),
                "family": row.get("family", "selector_baseline"),
                "step_MB_per_head_query": float(mb),
                "relL2": float(rel),
                "max_relL2": float(row.get("max_relL2") or row.get("relL2") or rel),
                "source": str(path),
            }
        )
    return points


def write_points(path: Path, points: list[dict[str, object]]) -> None:
    fields = ["method", "family", "step_MB_per_head_query", "relL2", "max_relL2", "source"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for point in points:
            writer.writerow({field: point.get(field, "") for field in fields})


def write_markdown(path: Path, points: list[dict[str, object]]) -> None:
    ordered = sorted(points, key=lambda p: (float(p["step_MB_per_head_query"]), float(p["relL2"])))
    lines = [
        "# KV Compression vs Frontier",
        "",
        "| Method | Family | MB/head-query | mean o-proj relL2 | max relL2 |",
        "|---|---:|---:|---:|---:|",
    ]
    for point in ordered:
        lines.append(
            "| {method} | {family} | {mb:.3f} | {rel:.6g} | {max_rel:.6g} |".format(
                method=point["method"],
                family=point.get("family", ""),
                mb=float(point["step_MB_per_head_query"]),
                rel=float(point["relL2"]),
                max_rel=float(point.get("max_relL2", point["relL2"])),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot(path_png: Path, path_pdf: Path, points: list[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    families = sorted({str(p.get("family", "")) for p in points})
    markers = ["o", "s", "^", "D", "P", "X", "v", "*", "<", ">"]
    marker_by_family = {family: markers[idx % len(markers)] for idx, family in enumerate(families)}

    fig, ax = plt.subplots(figsize=(8.8, 5.3), dpi=180)
    for family in families:
        group = [p for p in points if str(p.get("family", "")) == family]
        xs = [float(p["step_MB_per_head_query"]) for p in group]
        ys = [float(p["relL2"]) for p in group]
        ax.scatter(xs, ys, s=70, marker=marker_by_family[family], label=family or "other", alpha=0.9)
        for point in group:
            ax.annotate(
                str(point["method"]),
                (float(point["step_MB_per_head_query"]), float(point["relL2"])),
                xytext=(4, 5),
                textcoords="offset points",
                fontsize=7.5,
            )
    ax.set_xlabel("Step MB per head-query")
    ax.set_ylabel("Mean o-proj relative L2")
    ax.set_title("KV-Cache Compression vs Current Frontier")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    ax.set_ylim(bottom=0.0)
    ax.set_xlim(left=0.0)
    fig.tight_layout()
    fig.savefig(path_png)
    fig.savefig(path_pdf)
    plt.close(fig)


def run() -> None:
    parser = argparse.ArgumentParser(description="Plot KV-compression MB-vs-relL2 points.")
    parser.add_argument("--compression_summary_csv", required=True, help="Comma-separated summary.csv paths.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--existing_points_csv", default="attention_efficiency_result/plots/mb_vs_relL2_current_20260522/mb_vs_relL2_clean_points.csv")
    parser.add_argument("--frontier_label", default="Current adaptive K/V confidence")
    parser.add_argument("--frontier_mb", type=float, default=4.779294550418854)
    parser.add_argument("--frontier_rel_l2", type=float, default=0.0011178374271938424)
    parser.add_argument("--frontier_max_rel_l2", type=float, default=0.002082)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    points = []
    if str(args.existing_points_csv).strip():
        points.extend(read_existing_points(Path(args.existing_points_csv)))
    for csv_path in [part.strip() for part in str(args.compression_summary_csv).split(",") if part.strip()]:
        points.extend(read_compression_points(Path(csv_path)))
    points.append(
        {
            "method": str(args.frontier_label),
            "family": "frontier_selector_compression",
            "step_MB_per_head_query": float(args.frontier_mb),
            "relL2": float(args.frontier_rel_l2),
            "max_relL2": float(args.frontier_max_rel_l2),
            "source": "notes/current_status.md",
        }
    )

    # Deduplicate exact duplicate labels, keeping the latest appended point.
    dedup: dict[str, dict[str, object]] = {}
    for point in points:
        dedup[str(point["method"])] = point
    points = list(dedup.values())

    write_points(out_dir / "kv_compression_vs_frontier_points.csv", points)
    write_markdown(out_dir / "kv_compression_vs_frontier.md", points)
    (out_dir / "kv_compression_vs_frontier_points.json").write_text(json.dumps(points, indent=2, sort_keys=True), encoding="utf-8")
    plot(out_dir / "kv_compression_vs_frontier.png", out_dir / "kv_compression_vs_frontier.pdf", points)
    print(json.dumps({"output_dir": str(out_dir), "points": len(points)}, indent=2))


if __name__ == "__main__":
    run()
