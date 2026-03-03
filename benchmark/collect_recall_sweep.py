#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path


SUMMARY_RE = re.compile(r"parity_summary_json=(\{.*\})")
INDEX_BUILT_RE = re.compile(
    r"index built layer=(\d+)\s+head=(\d+).*?time=([0-9.]+)s.*?proj=([0-9.]+)s.*?edges=(\d+)"
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Collect recall/traversal metrics from sweep jobs and emit CSV (+ optional plot)."
    )
    p.add_argument("--jobs_tsv", required=True, help="TSV created at submission time (job_id + config).")
    p.add_argument("--out_csv", required=True, help="Output CSV path.")
    p.add_argument("--out_png", default="", help="Optional output PNG path (requires matplotlib).")
    p.add_argument(
        "--log_pattern",
        default="slurm-{name}-{job_id}.out",
        help="Log filename pattern; supports {name} and {job_id}.",
    )
    return p.parse_args()


def load_jobs(tsv_path: Path):
    jobs = []
    with tsv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            jobs.append(row)
    return jobs


def find_summary_json(log_path: Path):
    if not log_path.exists():
        return None, f"missing_log:{log_path}", ""
    text = log_path.read_text(encoding="utf-8", errors="replace")
    matches = SUMMARY_RE.findall(text)
    if not matches:
        return None, "missing_summary_json", text
    try:
        return json.loads(matches[-1]), "", text
    except Exception as exc:
        return None, f"json_parse_error:{exc}", text


def extract_graph_build_stats(log_text: str):
    matches = INDEX_BUILT_RE.findall(log_text or "")
    if not matches:
        return {}
    rows = []
    for layer_s, head_s, time_s, proj_s, edges_s in matches:
        try:
            rows.append(
                (
                    int(layer_s),
                    int(head_s),
                    float(time_s),
                    float(proj_s),
                    int(edges_s),
                )
            )
        except Exception:
            continue
    if not rows:
        return {}

    def _mean(vals):
        return float(sum(vals) / float(len(vals))) if vals else 0.0

    all_time = [r[2] for r in rows]
    all_proj = [r[3] for r in rows]
    all_edges = [float(r[4]) for r in rows]
    layer0 = [r for r in rows if r[0] == 0]
    out = {
        "graph_build_rows": int(len(rows)),
        "graph_build_time_mean": _mean(all_time),
        "graph_build_time_sum": float(sum(all_time)),
        "graph_build_proj_mean": _mean(all_proj),
        "graph_build_proj_sum": float(sum(all_proj)),
        "graph_build_edges_mean": _mean(all_edges),
    }
    if layer0:
        out["graph_build_layer0_rows"] = int(len(layer0))
        out["graph_build_layer0_time_mean"] = _mean([r[2] for r in layer0])
        out["graph_build_layer0_proj_mean"] = _mean([r[3] for r in layer0])
        out["graph_build_layer0_edges_mean"] = _mean([float(r[4]) for r in layer0])
    return out


def extract_point(summary: dict):
    traversal = summary.get("traversal") or {}
    return {
        "trav_visit_rate": float(traversal.get("visit_rate_mean", 0.0)),
        "trav_recall": float(traversal.get("recall_mean", 0.0)),
        "trav_visited_mean": float(traversal.get("visited_mean", 0.0)),
        "trav_cand_per_visit": float(traversal.get("cand_per_visit_mean", 0.0)),
        "parity_recall": float(summary.get("recall_weighted", 0.0)),
    }


def write_csv(rows, out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    base_fieldnames = [
        "job_id",
        "name",
        "status",
        "parity_recall",
        "trav_recall",
        "trav_visit_rate",
        "trav_visited_mean",
        "trav_cand_per_visit",
        "log_file",
        "error",
    ]
    if rows:
        extra = set()
        for row in rows:
            extra.update(row.keys())
        for k in base_fieldnames:
            if k in extra:
                extra.remove(k)
        preferred = [
            "expand",
            "min_visits",
            "max_visits",
            "cand_mult",
            "trav_sample",
            "train_frac",
            "split",
            "seed",
        ]
        ordered_extra = [k for k in preferred if k in extra] + sorted(k for k in extra if k not in preferred)
        fieldnames = ["job_id", "name"] + ordered_extra + [k for k in base_fieldnames if k not in {"job_id", "name"}]
    else:
        fieldnames = list(base_fieldnames)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def maybe_plot(rows, out_png: Path):
    if not out_png:
        return
    complete = [r for r in rows if r.get("status") == "ok"]
    if not complete:
        return
    xs = [float(r["trav_visit_rate"]) for r in complete]
    ys = [float(r["trav_recall"]) for r in complete]
    labels = [r["name"] for r in complete]
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(xs, ys, marker="o")
    for x, y, lbl in zip(xs, ys, labels):
        ax.annotate(lbl, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_xlabel("Traversal Visit Rate")
    ax.set_ylabel("Traversal Recall")
    ax.set_title("Recall vs Traversal Effort")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)


def main():
    args = parse_args()
    jobs_tsv = Path(args.jobs_tsv)
    out_csv = Path(args.out_csv)
    out_png = Path(args.out_png) if args.out_png else None
    jobs = load_jobs(jobs_tsv)
    rows = []

    for job in jobs:
        job_id = str(job["job_id"]).strip()
        name = str(job["name"]).strip()
        log_file = args.log_pattern.format(name=name, job_id=job_id)
        log_path = Path(log_file)
        summary, err, log_text = find_summary_json(log_path)
        graph_stats = extract_graph_build_stats(log_text)
        out = dict(job)
        out["log_file"] = str(log_path)
        out.update(graph_stats)
        if summary is None:
            out["status"] = "pending_or_failed"
            out["error"] = err
            out["parity_recall"] = ""
            out["trav_recall"] = ""
            out["trav_visit_rate"] = ""
            out["trav_visited_mean"] = ""
            out["trav_cand_per_visit"] = ""
        else:
            point = extract_point(summary)
            out["status"] = "ok"
            out["error"] = ""
            out["parity_recall"] = point["parity_recall"]
            out["trav_recall"] = point["trav_recall"]
            out["trav_visit_rate"] = point["trav_visit_rate"]
            out["trav_visited_mean"] = point["trav_visited_mean"]
            out["trav_cand_per_visit"] = point["trav_cand_per_visit"]
        rows.append(out)

    rows_sorted = sorted(
        rows,
        key=lambda r: float(r["trav_visit_rate"]) if r.get("status") == "ok" and str(r.get("trav_visit_rate", "")).strip() else -1.0,
    )
    write_csv(rows_sorted, out_csv)
    if out_png is not None:
        maybe_plot(rows_sorted, out_png)

    ok = sum(1 for r in rows_sorted if r.get("status") == "ok")
    total = len(rows_sorted)
    print(f"[collect_recall_sweep] complete={ok}/{total}")
    for r in rows_sorted:
        print(
            f"{r['job_id']} {r['name']} status={r['status']} "
            f"visit_rate={r.get('trav_visit_rate', '')} recall={r.get('trav_recall', '')} "
            f"log={r.get('log_file', '')}"
        )


if __name__ == "__main__":
    main()
