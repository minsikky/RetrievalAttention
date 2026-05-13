#!/usr/bin/env python3
"""Regression diagnostics for dense-oracle token scaling."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("attention_efficiency_result")
IN_CSV = ROOT / "plots" / "oracle_vs_ra_mb_vs_decode.csv"
OUT_DIR = ROOT / "plots"
STATIC_TOKENS = 128 + 512
HEAD_DIM = 128
KV_BYTES_PER_TOKEN = HEAD_DIM * 4
TARGETS = [0.95, 0.98]


def r2(y: np.ndarray, pred: np.ndarray) -> float:
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-20)


def rmse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - pred) ** 2)))


def fit_line(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coef = np.polyfit(x, y, deg=1)
    return coef, np.polyval(coef, x)


def fit_power(x: np.ndarray, y: np.ndarray) -> tuple[float, float, np.ndarray]:
    logx = np.log(x)
    logy = np.log(y)
    alpha, logc = np.polyfit(logx, logy, deg=1)
    pred = math.exp(float(logc)) * np.power(x, float(alpha))
    return float(math.exp(float(logc))), float(alpha), pred


def main() -> None:
    rows = []
    with IN_CSV.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["method"] != "dense_oracle":
                continue
            estimated_mb = float(row["estimated_mb"])
            exact_tokens = estimated_mb * 1024.0 * 1024.0 / float(KV_BYTES_PER_TOKEN)
            rows.append(
                {
                    "decode": int(row["decode"]),
                    "target": float(row["target"]),
                    "exact_tokens": exact_tokens,
                    "dynamic_tokens": max(0.0, exact_tokens - STATIC_TOKENS),
                    "mass": float(row["mass"]),
                }
            )

    regression_rows = []
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for ax, target in zip(axes, TARGETS):
        items = [r for r in rows if abs(r["target"] - target) < 1e-9]
        x = np.asarray([r["decode"] for r in items], dtype=np.float64)
        y = np.asarray([r["dynamic_tokens"] for r in items], dtype=np.float64)

        lin_coef, lin_pred = fit_line(x, y)
        log_coef, log_pred = fit_line(np.log2(x), y)
        c, alpha, pow_pred = fit_power(x, y)

        fits = [
            ("linear", f"y={lin_coef[0]:.4f}N+{lin_coef[1]:.1f}", lin_pred),
            ("log2", f"y={log_coef[0]:.1f}log2(N){log_coef[1]:+.1f}", log_pred),
            ("power", f"y={c:.1f}N^{alpha:.3f}", pow_pred),
        ]
        for model, equation, pred in fits:
            regression_rows.append(
                {
                    "target": target,
                    "model": model,
                    "equation": equation,
                    "r2": r2(y, pred),
                    "rmse_dynamic_tokens": rmse(y, pred),
                }
            )

        x_dense = np.geomspace(float(x.min()), float(x.max()), 200)
        lin_dense = np.polyval(lin_coef, x_dense)
        log_dense = np.polyval(log_coef, np.log2(x_dense))
        pow_dense = c * np.power(x_dense, alpha)

        ax.scatter(x, y, color="#111111", s=55, label="Dense oracle observed", zorder=5)
        ax.plot(x_dense, lin_dense, color="#C44E52", linewidth=2.2, label=f"Linear, R2={r2(y, lin_pred):.3f}")
        ax.plot(x_dense, log_dense, color="#55A868", linewidth=2.2, label=f"Log2, R2={r2(y, log_pred):.3f}")
        ax.plot(x_dense, pow_dense, color="#4C72B0", linewidth=2.2, label=f"Power a={alpha:.2f}, R2={r2(y, pow_pred):.3f}")
        ax.set_xscale("log", base=2)
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(v)) for v in x])
        ax.set_title(f"Target mass = {target:.2f}")
        ax.set_xlabel("Decode length")
        ax.set_ylabel("Oracle dynamic tokens needed")
        ax.grid(True, which="major", color="#D7D7D7", linewidth=0.8)
        ax.legend(frameon=False, fontsize=9)

    fig.suptitle("Dense-oracle token scaling", fontsize=15, fontweight="bold")
    fig.tight_layout()
    out_png = OUT_DIR / "oracle_token_scaling_regression.png"
    fig.savefig(out_png, dpi=220)
    plt.close(fig)

    with (OUT_DIR / "oracle_token_scaling_observed.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["decode", "target", "exact_tokens", "dynamic_tokens", "mass"])
        writer.writeheader()
        writer.writerows(rows)
    with (OUT_DIR / "oracle_token_scaling_regression.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["target", "model", "equation", "r2", "rmse_dynamic_tokens"])
        writer.writeheader()
        writer.writerows(regression_rows)

    print(json.dumps(regression_rows, indent=2))
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
