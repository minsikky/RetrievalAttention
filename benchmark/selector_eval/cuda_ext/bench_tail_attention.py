#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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


def _parse_ints(text: str) -> list[int]:
    out = []
    for item in text.replace(";", ",").split(","):
        item = item.strip()
        if item:
            out.append(int(item))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Microbenchmark causal selected+V-PQ-tail attention kernels.")
    parser.add_argument("--positions", type=int, default=128)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--pages", type=int, default=8)
    parser.add_argument("--page-size", type=int, default=512)
    parser.add_argument("--static-prefix", type=int, default=128)
    parser.add_argument("--static-suffix", type=int, default=128)
    parser.add_argument("--selected", default="512,1024,2048,4096")
    parser.add_argument("--value-centroids", type=int, default=16)
    parser.add_argument("--value-subvecs", type=int, default=1)
    parser.add_argument("--exact-value-top", type=int, default=1024)
    parser.add_argument("--tail-blend", type=float, default=1.0)
    parser.add_argument("--kv-dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.heads % args.kv_heads != 0:
        raise ValueError("heads must be divisible by kv-heads")
    if args.dim % args.value_subvecs != 0:
        raise ValueError("dim must be divisible by value-subvecs")

    torch.manual_seed(20260516)
    device = torch.device("cuda")
    kv_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.kv_dtype]
    page_size = int(args.page_size)
    pages = int(args.pages)
    static_prefix = int(args.static_prefix)
    total_tokens = static_prefix + pages * page_size + int(args.static_suffix) + int(args.positions)
    query_start = total_tokens - int(args.positions)
    group_size = int(args.heads // args.kv_heads)
    scale = float(args.dim) ** -0.5

    queries = torch.randn((args.positions, args.heads, args.dim), device=device, dtype=torch.float32)
    keys = torch.randn((args.kv_heads, total_tokens, args.dim), device=device, dtype=kv_dtype)
    values = torch.randn((args.kv_heads, total_tokens, args.dim), device=device, dtype=kv_dtype)
    page_starts = static_prefix + torch.arange(pages, device=device, dtype=torch.long) * page_size
    dense_scores = torch.randn((args.positions, args.heads, pages * page_size), device=device, dtype=torch.float32)
    value_subdim = int(args.dim // args.value_subvecs)
    value_codebooks = torch.randn(
        (args.kv_heads, pages, args.value_subvecs, args.value_centroids, value_subdim),
        device=device,
        dtype=torch.float32,
    )
    value_codes = torch.randint(
        0,
        args.value_centroids,
        (args.kv_heads, pages, page_size, args.value_subvecs),
        device=device,
        dtype=torch.uint8,
    )

    all_page_tokens = (
        page_starts.reshape(pages, 1) + torch.arange(page_size, device=device, dtype=torch.long).reshape(1, page_size)
    ).reshape(-1)
    results = []
    for selected in _parse_ints(args.selected):
        selected = max(1, int(selected))
        base = torch.arange(selected, device=device, dtype=torch.long)
        tokens_1d = all_page_tokens[torch.remainder(base, all_page_tokens.numel())]
        ranked_tokens = tokens_1d.reshape(1, 1, selected).expand(args.positions, args.heads, selected).contiguous()
        ranked_scores = torch.zeros((args.positions, args.heads, selected), device=device, dtype=torch.float32)

        def run_tail():
            return selector_paged_pq.gqa_causal_vpq_selected_tail_from_scores(
                queries,
                keys,
                values,
                dense_scores,
                value_codebooks,
                value_codes,
                page_starts,
                ranked_tokens,
                ranked_scores,
                group_size,
                query_start,
                static_prefix,
                int(args.static_suffix),
                page_size,
                int(args.exact_value_top),
                scale,
                float(args.tail_blend),
            )

        def run_exact():
            return selector_paged_pq.gqa_causal_exact_selected_attention(
                queries,
                keys,
                values,
                ranked_tokens,
                ranked_scores,
                group_size,
                query_start,
                static_prefix,
                int(args.static_suffix),
                page_size,
                scale,
            )

        tail_out = run_tail()
        exact_out = run_exact()
        torch.cuda.synchronize()
        if tail_out.shape != (args.positions, args.heads, args.dim):
            raise AssertionError(f"unexpected tail output shape {tuple(tail_out.shape)}")
        if exact_out.shape != tail_out.shape:
            raise AssertionError(f"unexpected exact output shape {tuple(exact_out.shape)}")
        tail_ms = _time_ms(run_tail, warmup=args.warmup, iters=args.iters)
        exact_ms = _time_ms(run_exact, warmup=args.warmup, iters=args.iters)
        logical_selected_kv_mb = (
            float(args.positions * args.heads * selected * args.dim * 2 * torch.finfo(kv_dtype).bits / 8)
            / (1024.0 * 1024.0)
        )
        logical_tail_rows = float(args.positions * args.heads * pages * page_size)
        results.append(
            {
                "selected": selected,
                "tail_ms": tail_ms,
                "exact_ms": exact_ms,
                "tail_over_exact": tail_ms / exact_ms if exact_ms > 0 else float("inf"),
                "logical_selected_kv_mb": logical_selected_kv_mb,
                "logical_tail_rows": logical_tail_rows,
            }
        )

    print(
        json.dumps(
            {
                "positions": args.positions,
                "heads": args.heads,
                "kv_heads": args.kv_heads,
                "dim": args.dim,
                "pages": pages,
                "page_size": page_size,
                "total_tokens": total_tokens,
                "query_start": query_start,
                "static_prefix": static_prefix,
                "static_suffix": args.static_suffix,
                "value_subvecs": args.value_subvecs,
                "value_centroids": args.value_centroids,
                "exact_value_top": args.exact_value_top,
                "tail_blend": args.tail_blend,
                "kv_dtype": args.kv_dtype,
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
