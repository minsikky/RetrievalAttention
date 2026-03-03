#!/usr/bin/env python3
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np


def parse_list_int(text: str):
    out = []
    for part in (text or "").split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    if not out:
        raise ValueError(f"empty int list: {text!r}")
    return out


def load_roar_ext(repo_root: Path):
    ext_dir = (repo_root / "third_party" / "RoarGraph" / "python_ext").resolve()
    if str(ext_dir) not in sys.path:
        sys.path.insert(0, str(ext_dir))
    import roargraph_builder_ext  # pylint: disable=import-error

    return roargraph_builder_ext


def digest_graph(offsets: np.ndarray, neighbors: np.ndarray):
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(offsets).view(np.uint8).tobytes())
    h.update(np.ascontiguousarray(neighbors).view(np.uint8).tobytes())
    return h.hexdigest()


def run_case(ext, cfg: dict):
    n = int(cfg["num_tokens"])
    head_dim = int(cfg["head_dim"])
    nq = int(cfg["nq"])
    dynamic_start = int(cfg["dynamic_start"])
    dynamic_end = int(cfg["dynamic_end"])
    roar_m = int(cfg["roar_m"])
    roar_l = int(cfg["roar_l"])
    enhance_l = int(cfg["enhance_l"])
    enable_enhance = bool(cfg["enable_enhance"])
    entry = str(cfg["entry"])
    max_query_per_pivot = int(cfg["max_query_per_pivot"])
    threads = int(cfg["threads"])
    seed = int(cfg["seed"])

    rng = np.random.default_rng(seed)
    keys = rng.standard_normal((n, head_dim), dtype=np.float32)
    knn = rng.integers(dynamic_start, dynamic_end, size=(n, nq), dtype=np.int32)

    t0 = time.perf_counter()
    offsets, neighbors, meta = ext.build_graph_csr(
        knn,
        keys,
        dynamic_start,
        dynamic_end,
        nq,
        roar_m,
        roar_l,
        enable_enhance,
        enhance_l,
        entry,
        max_query_per_pivot,
        threads,
    )
    wall = time.perf_counter() - t0
    offsets_np = np.asarray(offsets, dtype=np.uint32)
    neighbors_np = np.asarray(neighbors, dtype=np.int32)
    graph_hash = digest_graph(offsets_np, neighbors_np)
    return offsets_np, neighbors_np, meta, wall, graph_hash


def build_cases(args):
    sizes = parse_list_int(args.sizes)
    cases = []
    cid = 0
    for n in sizes:
        dynamic_start = max(0, min(int(args.static_start), n - 1))
        dynamic_end = n - int(args.static_end)
        if dynamic_end <= dynamic_start:
            raise ValueError(
                f"Invalid dynamic range for n={n}: [{dynamic_start}, {dynamic_end})"
            )
        for k in range(int(args.cases_per_size)):
            cid += 1
            cases.append(
                {
                    "case_id": f"n{n}_c{k}",
                    "num_tokens": int(n),
                    "head_dim": int(args.head_dim),
                    "nq": int(args.nq),
                    "dynamic_start": int(dynamic_start),
                    "dynamic_end": int(dynamic_end),
                    "roar_m": int(args.roar_m),
                    "roar_l": int(args.roar_l),
                    "enhance_l": int(args.enhance_l),
                    "enable_enhance": bool(not args.disable_enhance),
                    "entry": str(args.entry),
                    "max_query_per_pivot": int(args.max_query_per_pivot),
                    "threads": int(args.threads),
                    "seed": int(args.seed + cid * 7919),
                }
            )
    return cases


def save_golden(path: Path, cases_out: list):
    payload = {}
    manifest = []
    for row in cases_out:
        cid = row["case_id"]
        payload[f"offsets__{cid}"] = row["offsets"]
        payload[f"neighbors__{cid}"] = row["neighbors"]
        m = dict(row)
        m.pop("offsets", None)
        m.pop("neighbors", None)
        manifest.append(m)
    payload["manifest_json"] = np.array([json.dumps(manifest, sort_keys=True)], dtype=object)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def load_golden(path: Path):
    z = np.load(path, allow_pickle=True)
    manifest = json.loads(str(z["manifest_json"][0]))
    out = {}
    for m in manifest:
        cid = m["case_id"]
        out[cid] = {
            "meta": m,
            "offsets": np.asarray(z[f"offsets__{cid}"], dtype=np.uint32),
            "neighbors": np.asarray(z[f"neighbors__{cid}"], dtype=np.int32),
        }
    return out


def compare_graphs(ref_offsets, ref_neighbors, cur_offsets, cur_neighbors):
    ok_offsets = ref_offsets.shape == cur_offsets.shape and np.array_equal(ref_offsets, cur_offsets)
    ok_neighbors = ref_neighbors.shape == cur_neighbors.shape and np.array_equal(ref_neighbors, cur_neighbors)
    return ok_offsets and ok_neighbors, ok_offsets, ok_neighbors


def parse_args():
    p = argparse.ArgumentParser(description="CPU-only graph-builder parity harness.")
    p.add_argument("--mode", choices=["write", "check"], required=True)
    p.add_argument("--golden", required=True, help="Path to .npz golden file.")
    p.add_argument("--sizes", default="8192,16384", help="Comma-separated token sizes.")
    p.add_argument("--cases_per_size", type=int, default=2)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--head_dim", type=int, default=128)
    p.add_argument("--nq", type=int, default=32)
    p.add_argument("--static_start", type=int, default=128)
    p.add_argument("--static_end", type=int, default=512)
    p.add_argument("--roar_m", type=int, default=32)
    p.add_argument("--roar_l", type=int, default=20)
    p.add_argument("--enhance_l", type=int, default=16)
    p.add_argument("--disable_enhance", action="store_true")
    p.add_argument("--entry", choices=["hub", "max_degree", "self"], default="hub")
    p.add_argument("--max_query_per_pivot", type=int, default=0)
    p.add_argument("--threads", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    ext = load_roar_ext(repo_root)
    cases = build_cases(args)
    golden_path = Path(args.golden)

    print(
        "[CONFIG] mode={} golden={} sizes={} cases_per_size={} threads={}".format(
            args.mode, golden_path, args.sizes, args.cases_per_size, args.threads
        )
    )

    rows = []
    if args.mode == "write":
        for cfg in cases:
            offsets, neighbors, meta, wall, graph_hash = run_case(ext, cfg)
            print(
                "[WRITE] case={} N={} edges={} hash={} wall={:.3f}s proj={:.3f}s enh={:.3f}s".format(
                    cfg["case_id"],
                    cfg["num_tokens"],
                    int(offsets[-1]) if offsets.size else 0,
                    graph_hash[:16],
                    wall,
                    float(meta.get("projection_sec", 0.0)),
                    float(meta.get("enhance_sec", 0.0)),
                )
            )
            rows.append(
                {
                    **cfg,
                    "offsets": offsets,
                    "neighbors": neighbors,
                    "graph_hash": graph_hash,
                    "wall_sec": float(wall),
                    "meta": dict(meta),
                }
            )
        save_golden(golden_path, rows)
        print(f"[WRITE] saved golden -> {golden_path}")
        return

    # check mode
    ref = load_golden(golden_path)
    mismatches = 0
    for cfg in cases:
        cid = cfg["case_id"]
        if cid not in ref:
            raise RuntimeError(f"case {cid} not found in golden {golden_path}")
        cur_offsets, cur_neighbors, meta, wall, graph_hash = run_case(ext, cfg)
        ok, ok_offsets, ok_neighbors = compare_graphs(
            ref[cid]["offsets"],
            ref[cid]["neighbors"],
            cur_offsets,
            cur_neighbors,
        )
        tag = "OK" if ok else "MISMATCH"
        print(
            "[CHECK:{}] case={} N={} edges={} hash={} wall={:.3f}s proj={:.3f}s enh={:.3f}s".format(
                tag,
                cid,
                cfg["num_tokens"],
                int(cur_offsets[-1]) if cur_offsets.size else 0,
                graph_hash[:16],
                wall,
                float(meta.get("projection_sec", 0.0)),
                float(meta.get("enhance_sec", 0.0)),
            )
        )
        if not ok:
            mismatches += 1
            print(
                "  offsets_equal={} neighbors_equal={} "
                "ref_edges={} cur_edges={}".format(
                    ok_offsets,
                    ok_neighbors,
                    int(ref[cid]["offsets"][-1]) if ref[cid]["offsets"].size else 0,
                    int(cur_offsets[-1]) if cur_offsets.size else 0,
                )
            )
    if mismatches > 0:
        raise RuntimeError(f"parity failed: {mismatches}/{len(cases)} mismatched cases")
    print(f"[CHECK] all {len(cases)} cases matched exactly.")


if __name__ == "__main__":
    main()
