#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch


EXT_ROOT = Path(__file__).resolve().parent
if str(EXT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXT_ROOT))

import selector_paged_pq  # noqa: E402


def _time_ms(fn, *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end)) / max(1, iters)


def main() -> None:
    parser = argparse.ArgumentParser(description="Microbenchmark causal GQA page-PQ selector backends.")
    parser.add_argument("--positions", type=int, default=128)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--pages", type=int, default=16)
    parser.add_argument("--page-size", type=int, default=256)
    parser.add_argument(
        "--page-configs",
        default="",
        help="Optional comma/semicolon list like 16x256;8x512. Overrides --pages/--page-size.",
    )
    parser.add_argument("--subvecs", type=int, default=4)
    parser.add_argument("--centroids", type=int, default=256)
    parser.add_argument("--budgets", default="8,16,32,64")
    parser.add_argument(
        "--fused-modes",
        default="auto",
        help="Comma/semicolon list from auto,smallscan,localtopk.",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--static-prefix", type=int, default=128)
    parser.add_argument("--static-suffix", type=int, default=128)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.heads % args.kv_heads != 0:
        raise ValueError("heads must be divisible by kv_heads")
    if args.dim % args.subvecs != 0:
        raise ValueError("dim must be divisible by subvecs")
    if args.centroids > 256:
        raise ValueError("uint8 code benchmark supports at most 256 centroids")

    if args.page_configs.strip():
        page_configs = []
        for item in re.split(r"[;,]", args.page_configs):
            if not item.strip():
                continue
            pages_text, page_size_text = item.lower().split("x", 1)
            page_configs.append((int(pages_text), int(page_size_text)))
    else:
        page_configs = [(int(args.pages), int(args.page_size))]
    mode_to_id = {"auto": 0, "smallscan": 1, "localtopk": 2}
    fused_modes = []
    for item in re.split(r"[;,]", args.fused_modes):
        mode = item.strip().lower()
        if not mode:
            continue
        if mode not in mode_to_id:
            raise ValueError(f"unknown fused mode {mode!r}; expected one of {sorted(mode_to_id)}")
        fused_modes.append(mode)
    if not fused_modes:
        fused_modes = ["auto"]

    torch.manual_seed(20260515)
    device = torch.device("cuda")
    subdim = args.dim // args.subvecs
    group_size = args.heads // args.kv_heads
    queries = torch.randn((args.positions, args.heads, args.dim), device=device, dtype=torch.float32)
    results_by_config = []
    for pages, page_size in page_configs:
        codebooks = torch.randn(
            (args.kv_heads, pages, args.subvecs, args.centroids, subdim),
            device=device,
            dtype=torch.float32,
        )
        codes = torch.randint(
            0,
            args.centroids,
            (args.kv_heads, pages, page_size, args.subvecs),
            device=device,
            dtype=torch.uint8,
        )
        page_starts = torch.arange(
            args.static_prefix,
            args.static_prefix + pages * page_size,
            page_size,
            device=device,
            dtype=torch.long,
        )
        query_start = int(args.static_prefix + pages * page_size + args.static_suffix)

        results = []
        for budget_text in args.budgets.split(","):
            budget = int(budget_text)

            def native():
                return selector_paged_pq.gqa_causal_fullscan_pq_topk(
                    queries,
                    codebooks,
                    codes,
                    page_starts,
                    group_size,
                    budget,
                    query_start,
                    args.static_prefix,
                    args.static_suffix,
                )

            native_tokens, native_scores = native()
            torch.cuda.synchronize()

            native_ms = _time_ms(native, warmup=args.warmup, iters=args.iters)
            result = {"budget": budget, "native_ms": native_ms, "fused": []}
            for mode in fused_modes:
                mode_id = mode_to_id[mode]

                def fused():
                    if mode == "auto":
                        return selector_paged_pq.gqa_causal_fullscan_pq_topk_fused(
                            queries,
                            codebooks,
                            codes,
                            page_starts,
                            group_size,
                            budget,
                            query_start,
                            args.static_prefix,
                            args.static_suffix,
                        )
                    return selector_paged_pq.gqa_causal_fullscan_pq_topk_fused_force(
                        queries,
                        codebooks,
                        codes,
                        page_starts,
                        group_size,
                        budget,
                        query_start,
                        args.static_prefix,
                        args.static_suffix,
                        mode_id,
                    )

                fused_tokens, fused_scores = fused()
                torch.cuda.synchronize()
                if not torch.allclose(native_scores, fused_scores, atol=1e-4, rtol=1e-4):
                    diff = torch.max(torch.abs(native_scores - fused_scores)).item()
                    raise AssertionError(
                        f"score mismatch for mode={mode} pages={pages} page_size={page_size} budget={budget}: max_diff={diff}"
                    )
                if not torch.equal(native_tokens, fused_tokens):
                    # PQ codes can tie exactly, and torch.topk does not expose a stable tie-break guarantee.
                    # Matching the ranked approximate scores is the selector contract for this microbenchmark.
                    mismatched = int((native_tokens != fused_tokens).sum().item())
                else:
                    mismatched = 0
                fused_ms = _time_ms(fused, warmup=args.warmup, iters=args.iters)
                result["fused"].append(
                    {
                        "mode": mode,
                        "ms": fused_ms,
                        "speedup": native_ms / fused_ms if fused_ms > 0 else float("inf"),
                        "token_tie_mismatches": mismatched,
                    }
                )
            results.append(result)
        results_by_config.append({"pages": pages, "page_size": page_size, "results": results})

    payload = {
        "positions": args.positions,
        "heads": args.heads,
        "kv_heads": args.kv_heads,
        "dim": args.dim,
        "subvecs": args.subvecs,
        "centroids": args.centroids,
        "configs": results_by_config,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
