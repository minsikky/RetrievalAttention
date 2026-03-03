#!/usr/bin/env python3
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize decode-complexity sweeps from collect_recall_sweep CSV."
    )
    parser.add_argument("--in_csv", required=True, help="CSV from benchmark/collect_recall_sweep.py")
    parser.add_argument("--target_recall", type=float, default=0.95)
    parser.add_argument("--out_frontier_csv", default="", help="Per-(regime,N) best row CSV.")
    parser.add_argument("--out_regime_csv", default="", help="Per-regime summary CSV.")
    parser.add_argument("--out_json", default="", help="Optional JSON report.")
    return parser.parse_args()


def to_float(val, default=float("nan")):
    try:
        txt = str(val).strip()
        if txt == "":
            return default
        return float(txt)
    except Exception:
        return default


def to_int(val, default=0):
    try:
        txt = str(val).strip()
        if txt == "":
            return default
        return int(float(txt))
    except Exception:
        return default


def fit_power_law(xs, ys):
    if len(xs) < 2 or len(ys) < 2:
        return None
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    mask = (x > 0) & (y > 0)
    x = x[mask]
    y = y[mask]
    if x.shape[0] < 2:
        return None
    lx = np.log(x)
    ly = np.log(y)
    slope, intercept = np.polyfit(lx, ly, 1)
    alpha = float(slope)
    c = float(math.exp(intercept))
    pred = c * np.power(x, alpha)
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot if ss_tot > 0 else 0.0)
    return {"c": c, "alpha": alpha, "r2": r2}


def load_rows(path: Path):
    rows = []
    skipped_invalid = 0
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            status = str(row.get("status", "")).strip()
            if status != "ok":
                continue
            r = dict(row)
            r["n_tokens"] = to_int(row.get("n_tokens", row.get("recall_input_tokens", 0)), 0)
            r["regime"] = str(row.get("regime", "")).strip()
            if r["n_tokens"] <= 0 or not r["regime"]:
                skipped_invalid += 1
                continue
            r["trav_recall"] = to_float(row.get("trav_recall"))
            r["trav_visit_rate"] = to_float(row.get("trav_visit_rate"))
            r["trav_visited_mean"] = to_float(row.get("trav_visited_mean"))
            r["parity_recall"] = to_float(row.get("parity_recall"))
            r["max_visits"] = to_int(row.get("max_visits", 0), 0)
            r["min_visits"] = to_int(row.get("min_visits", 0), 0)
            r["expand_width"] = to_int(row.get("expand_width", 0), 0)
            rows.append(r)
    return rows, skipped_invalid


def pick_best(rows, target_recall: float):
    by_key = defaultdict(list)
    for r in rows:
        key = (r["regime"], r["n_tokens"])
        by_key[key].append(r)

    frontier = []
    for key, group in sorted(by_key.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        regime, n_tokens = key
        feas = [r for r in group if np.isfinite(r["trav_recall"]) and r["trav_recall"] >= float(target_recall)]
        if feas:
            feas.sort(
                key=lambda r: (
                    float(r["trav_visit_rate"]),
                    -float(r["trav_recall"]),
                    float(r["max_visits"]) if r["max_visits"] > 0 else float("inf"),
                )
            )
            best = dict(feas[0])
            best["hit_target"] = 1
        else:
            # Keep best-effort row for diagnostics.
            group_sorted = sorted(
                group,
                key=lambda r: (
                    -float(r["trav_recall"]),
                    float(r["trav_visit_rate"]),
                ),
            )
            best = dict(group_sorted[0])
            best["hit_target"] = 0
        best["target_recall"] = float(target_recall)
        best["group_points"] = len(group)
        best["regime"] = regime
        best["n_tokens"] = int(n_tokens)
        frontier.append(best)
    return frontier


def summarize_regimes(frontier):
    by_regime = defaultdict(list)
    for row in frontier:
        by_regime[row["regime"]].append(row)

    out = []
    for regime, group in sorted(by_regime.items()):
        group_sorted = sorted(group, key=lambda r: int(r["n_tokens"]))
        hits = [r for r in group_sorted if int(r.get("hit_target", 0)) == 1]
        visit_mean = float(np.mean([float(r["trav_visit_rate"]) for r in hits])) if hits else float("nan")
        recall_mean = float(np.mean([float(r["trav_recall"]) for r in group_sorted])) if group_sorted else float("nan")
        n_hit = len(hits)
        n_total = len(group_sorted)

        fit_obs = None
        fit_cfg = None
        if len(hits) >= 2:
            fit_obs = fit_power_law(
                [float(r["n_tokens"]) for r in hits],
                [max(1.0, float(r["trav_visited_mean"])) for r in hits],
            )
            fit_cfg = fit_power_law(
                [float(r["n_tokens"]) for r in hits],
                [max(1.0, float(r["max_visits"])) for r in hits],
            )

        summary = {
            "regime": regime,
            "points": n_total,
            "hit_points": n_hit,
            "hit_rate": (float(n_hit) / float(n_total)) if n_total > 0 else 0.0,
            "mean_visit_rate_at_target": visit_mean,
            "mean_frontier_recall": recall_mean,
            "obs_alpha": float(fit_obs["alpha"]) if fit_obs else float("nan"),
            "obs_r2": float(fit_obs["r2"]) if fit_obs else float("nan"),
            "cfg_alpha": float(fit_cfg["alpha"]) if fit_cfg else float("nan"),
            "cfg_r2": float(fit_cfg["r2"]) if fit_cfg else float("nan"),
        }
        out.append(summary)
    return out


def write_csv(rows, path: Path):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main():
    args = parse_args()
    in_csv = Path(args.in_csv)
    rows, skipped_invalid = load_rows(in_csv)
    if not rows:
        raise RuntimeError(f"No complete rows in: {in_csv}")

    frontier = pick_best(rows=rows, target_recall=float(args.target_recall))
    regime_summary = summarize_regimes(frontier=frontier)

    print(f"[summarize_decode_complexity] input_rows={len(rows)}")
    if skipped_invalid > 0:
        print(f"[summarize_decode_complexity] skipped_invalid_rows={skipped_invalid}")
    print(f"[summarize_decode_complexity] frontier_rows={len(frontier)}")
    print("[frontier]")
    for r in sorted(frontier, key=lambda x: (x["regime"], int(x["n_tokens"]))):
        print(
            f"regime={r['regime']:<6} N={int(r['n_tokens']):>6d} "
            f"hit={int(r['hit_target'])} visit={float(r['trav_visit_rate']):.5f} "
            f"recall={float(r['trav_recall']):.5f} max_vis={int(r['max_visits'])}"
        )

    print("[regime_summary]")
    for s in regime_summary:
        print(
            f"regime={s['regime']:<6} hit={int(s['hit_points'])}/{int(s['points'])} "
            f"mean_visit@target={s['mean_visit_rate_at_target']:.5f} "
            f"obs_alpha={s['obs_alpha']:.3f} cfg_alpha={s['cfg_alpha']:.3f}"
        )

    out_frontier_csv = Path(args.out_frontier_csv) if args.out_frontier_csv else None
    out_regime_csv = Path(args.out_regime_csv) if args.out_regime_csv else None
    out_json = Path(args.out_json) if args.out_json else None

    if out_frontier_csv is not None:
        write_csv(frontier, out_frontier_csv)
        print(f"[summarize_decode_complexity] frontier_csv={out_frontier_csv}")
    if out_regime_csv is not None:
        write_csv(regime_summary, out_regime_csv)
        print(f"[summarize_decode_complexity] regime_csv={out_regime_csv}")
    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "target_recall": float(args.target_recall),
            "input_rows": len(rows),
            "frontier": frontier,
            "regime_summary": regime_summary,
        }
        out_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[summarize_decode_complexity] report_json={out_json}")


if __name__ == "__main__":
    main()
