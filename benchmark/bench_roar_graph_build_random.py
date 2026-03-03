#!/usr/bin/env python3
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np


def load_roar_extension(repo_root: Path):
    ext_dir = (repo_root / "third_party" / "RoarGraph" / "python_ext").resolve()
    if str(ext_dir) not in sys.path:
        sys.path.insert(0, str(ext_dir))
    try:
        import roargraph_builder_ext  # pylint: disable=import-error
    except Exception as exc:
        raise RuntimeError(
            "Failed to import roargraph_builder_ext. Build it with:\n"
            "  module load python/3.10.4\n"
            "  source .venv/bin/activate\n"
            "  python third_party/RoarGraph/python_ext/setup.py build_ext --inplace\n"
            f"Import error: {exc}"
        ) from exc
    return roargraph_builder_ext


def parse_args():
    parser = argparse.ArgumentParser(
        description="Random-data microbenchmark for Roar graph build (C++ backend)."
    )
    parser.add_argument("--num_tokens", type=int, default=8192)
    parser.add_argument("--head_dim", type=int, default=128)
    parser.add_argument("--num_queries", type=int, default=8192)
    parser.add_argument("--dynamic_start", type=int, default=128)
    parser.add_argument(
        "--dynamic_end",
        type=int,
        default=-1,
        help="If <0, use num_tokens-512.",
    )
    parser.add_argument("--nq", type=int, default=32)
    parser.add_argument("--roar_m", type=int, default=32)
    parser.add_argument("--roar_l", type=int, default=128)
    parser.add_argument("--enhance_l", type=int, default=256)
    parser.add_argument("--entry", type=str, default="hub", choices=["hub", "max_degree", "self"])
    parser.add_argument("--max_query_per_pivot", type=int, default=0)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--disable_enhance", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    roar_ext = load_roar_extension(repo_root)

    num_tokens = int(args.num_tokens)
    head_dim = int(args.head_dim)
    num_queries = int(args.num_queries)
    dynamic_start = max(0, int(args.dynamic_start))
    dynamic_end = int(args.dynamic_end)
    if dynamic_end < 0:
        dynamic_end = num_tokens - 512
    dynamic_end = min(num_tokens, dynamic_end)
    if dynamic_end <= dynamic_start:
        raise ValueError(
            f"Invalid dynamic range: [{dynamic_start}, {dynamic_end}) for num_tokens={num_tokens}"
        )

    nq = max(1, int(args.nq))
    roar_m = max(1, int(args.roar_m))
    roar_l = max(1, int(args.roar_l))
    enhance_l = max(1, int(args.enhance_l))
    max_query_per_pivot = max(0, int(args.max_query_per_pivot))
    threads = max(0, int(args.threads))
    warmup = max(0, int(args.warmup))
    repeat = max(1, int(args.repeat))
    enable_enhance = not bool(args.disable_enhance)

    dyn_size = dynamic_end - dynamic_start
    rng = np.random.default_rng(int(args.seed))

    print("[Config]")
    print(f"  tokens={num_tokens}, head_dim={head_dim}, queries={num_queries}")
    print(f"  dynamic=[{dynamic_start}, {dynamic_end}) (size={dyn_size})")
    print(
        f"  nq={nq}, M={roar_m}, L={roar_l}, enhance={int(enable_enhance)}, "
        f"enhance_L={enhance_l}, entry={args.entry}, max_q_per_pivot={max_query_per_pivot}"
    )
    print(f"  threads={threads}, warmup={warmup}, repeat={repeat}, seed={args.seed}")

    print("[Data] Generating random keys/knn ...")
    keys = rng.standard_normal((num_tokens, head_dim)).astype(np.float32, copy=False)
    knn = rng.integers(dynamic_start, dynamic_end, size=(num_queries, nq), dtype=np.int32)

    print("[Run] Starting benchmark ...")
    for _ in range(warmup):
        roar_ext.build_graph_csr(
            knn,
            keys,
            dynamic_start,
            dynamic_end,
            nq,
            roar_m,
            roar_l,
            enable_enhance,
            enhance_l,
            args.entry,
            max_query_per_pivot,
            threads,
        )

    wall_times = []
    meta_times = []
    edge_counts = []
    projected_nodes = []
    enhanced_nodes = []
    active_queries = []
    active_pivots = []

    for ridx in range(repeat):
        t0 = time.perf_counter()
        offsets, neighbors, meta = roar_ext.build_graph_csr(
            knn,
            keys,
            dynamic_start,
            dynamic_end,
            nq,
            roar_m,
            roar_l,
            enable_enhance,
            enhance_l,
            args.entry,
            max_query_per_pivot,
            threads,
        )
        wall = time.perf_counter() - t0
        wall_times.append(wall)
        meta_times.append(float(meta.get("total_sec", 0.0)))
        edge_counts.append(int(offsets[-1]))
        projected_nodes.append(int(meta.get("projected_nodes", 0)))
        enhanced_nodes.append(int(meta.get("enhanced_nodes", 0)))
        active_queries.append(int(meta.get("active_queries", 0)))
        active_pivots.append(int(meta.get("active_pivots", 0)))
        print(
            f"  run={ridx + 1}/{repeat} "
            f"wall={wall:.4f}s meta_total={meta_times[-1]:.4f}s "
            f"edges={edge_counts[-1]} nodes={projected_nodes[-1]} "
            f"enh_nodes={enhanced_nodes[-1]} stop={meta.get('stop_reason', 'n/a')}"
        )

        # keep refs alive to prevent linter stripping; also validate shape
        if int(offsets[-1]) != int(neighbors.shape[0]):
            raise RuntimeError(
                f"Invalid CSR: offsets[-1]={int(offsets[-1])}, neighbors={int(neighbors.shape[0])}"
            )

    def mean_std(vals):
        arr = np.asarray(vals, dtype=np.float64)
        return float(arr.mean()), float(arr.std(ddof=0))

    wall_mean, wall_std = mean_std(wall_times)
    meta_mean, meta_std = mean_std(meta_times)
    edges_mean, edges_std = mean_std(edge_counts)

    print("[Summary]")
    print(f"  wall_time: mean={wall_mean:.4f}s std={wall_std:.4f}s")
    print(f"  meta_total: mean={meta_mean:.4f}s std={meta_std:.4f}s")
    print(f"  edges: mean={edges_mean:.1f} std={edges_std:.1f}")
    print(
        f"  active_queries(mean)={np.mean(active_queries):.1f}, "
        f"active_pivots(mean)={np.mean(active_pivots):.1f}, "
        f"projected_nodes(mean)={np.mean(projected_nodes):.1f}, "
        f"enhanced_nodes(mean)={np.mean(enhanced_nodes):.1f}"
    )


if __name__ == "__main__":
    main()
