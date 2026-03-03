#!/usr/bin/env python3
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np


def load_roar_ext(repo_root: Path):
    ext_dir = (repo_root / "third_party" / "RoarGraph" / "python_ext").resolve()
    if str(ext_dir) not in sys.path:
        sys.path.insert(0, str(ext_dir))
    try:
        import roargraph_builder_ext  # pylint: disable=import-error
    except Exception as exc:
        raise RuntimeError(
            "Failed to import roargraph_builder_ext.\n"
            "Build with:\n"
            "  module load python/3.10.4\n"
            "  source .venv/bin/activate\n"
            "  python third_party/RoarGraph/python_ext/setup.py build_ext --inplace\n"
            f"Import error: {exc}"
        ) from exc
    return roargraph_builder_ext


def parse_sizes(sizes_text: str):
    out = []
    for part in sizes_text.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    if not out:
        raise ValueError("No sizes parsed from --sizes")
    return out


def mean_std(values):
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=0))


def fit_power_law(sizes, times):
    x = np.log(np.asarray(sizes, dtype=np.float64))
    y = np.log(np.asarray(times, dtype=np.float64))
    slope, intercept = np.polyfit(x, y, 1)
    alpha = float(slope)
    c = float(math.exp(intercept))
    pred = c * np.power(np.asarray(sizes, dtype=np.float64), alpha)
    ss_res = float(np.sum((np.asarray(times, dtype=np.float64) - pred) ** 2))
    ss_tot = float(np.sum((np.asarray(times, dtype=np.float64) - np.mean(times)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot if ss_tot > 0 else 0.0)
    return c, alpha, r2


def main():
    parser = argparse.ArgumentParser(description="Scale test for Roar C++ graph build.")
    parser.add_argument(
        "--sizes",
        type=str,
        default="2048,4096,8192,12288,16384,24576",
        help="Comma-separated token counts.",
    )
    parser.add_argument("--head_dim", type=int, default=128)
    parser.add_argument("--nq", type=int, default=32)
    parser.add_argument("--roar_m", type=int, default=32)
    parser.add_argument("--roar_l", type=int, default=128)
    parser.add_argument("--enhance_l", type=int, default=256)
    parser.add_argument("--entry", type=str, default="hub", choices=["hub", "max_degree", "self"])
    parser.add_argument("--max_query_per_pivot", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--disable_enhance", action="store_true")
    parser.add_argument("--target_tokens", type=int, default=119647)
    args = parser.parse_args()

    sizes = parse_sizes(args.sizes)
    sizes = sorted(sizes)
    repeat = max(1, int(args.repeat))
    warmup = max(0, int(args.warmup))
    enable_enhance = not bool(args.disable_enhance)

    repo_root = Path(__file__).resolve().parent.parent
    ext = load_roar_ext(repo_root)

    print("[Config]")
    print(f"  sizes={sizes}")
    print(
        f"  head_dim={args.head_dim}, nq={args.nq}, M={args.roar_m}, L={args.roar_l}, "
        f"enhance={int(enable_enhance)}, enhance_L={args.enhance_l}, entry={args.entry}, "
        f"max_q_per_pivot={args.max_query_per_pivot}, threads={args.threads}"
    )
    print(f"  warmup={warmup}, repeat={repeat}, seed={args.seed}, target_tokens={args.target_tokens}")

    measured_sizes = []
    measured_times = []

    print("\n[Runs]")
    for n in sizes:
        dynamic_start = 128
        dynamic_end = n - 512
        if dynamic_end <= dynamic_start:
            print(f"  n={n}: skipped (dynamic range empty)")
            continue

        rng = np.random.default_rng(args.seed + n)
        keys = rng.standard_normal((n, args.head_dim)).astype(np.float32, copy=False)
        knn = rng.integers(dynamic_start, dynamic_end, size=(n, args.nq), dtype=np.int32)

        for _ in range(warmup):
            ext.build_graph_csr(
                knn,
                keys,
                dynamic_start,
                dynamic_end,
                args.nq,
                args.roar_m,
                args.roar_l,
                enable_enhance,
                args.enhance_l,
                args.entry,
                args.max_query_per_pivot,
                args.threads,
            )

        wall_times = []
        last_meta = None
        last_edges = 0
        for _ in range(repeat):
            t0 = time.perf_counter()
            offsets, neighbors, meta = ext.build_graph_csr(
                knn,
                keys,
                dynamic_start,
                dynamic_end,
                args.nq,
                args.roar_m,
                args.roar_l,
                enable_enhance,
                args.enhance_l,
                args.entry,
                args.max_query_per_pivot,
                args.threads,
            )
            wall = time.perf_counter() - t0
            wall_times.append(wall)
            last_meta = meta
            last_edges = int(offsets[-1])
            if int(offsets[-1]) != int(neighbors.shape[0]):
                raise RuntimeError(
                    f"Invalid CSR at n={n}: offsets[-1]={int(offsets[-1])} neighbors={int(neighbors.shape[0])}"
                )

        wall_mean, wall_std = mean_std(wall_times)
        measured_sizes.append(n)
        measured_times.append(wall_mean)
        print(
            f"  n={n:6d} | wall={wall_mean:8.4f}s ± {wall_std:7.4f}s | "
            f"edges={last_edges:9d} | active_q={int(last_meta.get('active_queries', 0)):6d} | "
            f"active_p={int(last_meta.get('active_pivots', 0)):6d} | "
            f"nodes={int(last_meta.get('projected_nodes', 0)):6d} | "
            f"enh_nodes={int(last_meta.get('enhanced_nodes', 0)):6d}"
        )

    if len(measured_sizes) < 2:
        print("\n[Fit] Not enough points to fit scaling trend.")
        return

    c, alpha, r2 = fit_power_law(measured_sizes, measured_times)
    target_pred = c * (float(args.target_tokens) ** alpha)

    print("\n[Fit]")
    print(f"  model: time ~= c * N^alpha")
    print(f"  c={c:.6e}, alpha={alpha:.4f}, R^2={r2:.4f}")
    print(
        f"  estimated time at N={args.target_tokens}: {target_pred:.1f}s "
        f"({target_pred / 60.0:.1f} min)"
    )


if __name__ == "__main__":
    main()
