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
    parser = argparse.ArgumentParser(description="Microbenchmark causal selected-attention CUDA kernel.")
    parser.add_argument("--positions", type=int, default=2048)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--total-tokens", type=int, default=16384)
    parser.add_argument("--query-start", type=int, default=8192)
    parser.add_argument("--selected", default="512,1024,2048")
    parser.add_argument("--static-prefix", type=int, default=128)
    parser.add_argument("--static-suffix", type=int, default=128)
    parser.add_argument("--page-size", type=int, default=512)
    parser.add_argument("--kv-dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.heads % args.kv_heads != 0:
        raise ValueError("heads must be divisible by kv-heads")

    torch.manual_seed(20260516)
    device = torch.device("cuda")
    kv_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.kv_dtype]
    queries = torch.randn((args.positions, args.heads, args.dim), device=device, dtype=torch.float32)
    keys = torch.randn((args.kv_heads, args.total_tokens, args.dim), device=device, dtype=kv_dtype)
    values = torch.randn((args.kv_heads, args.total_tokens, args.dim), device=device, dtype=kv_dtype)
    group_size = args.heads // args.kv_heads
    scale = args.dim ** -0.5
    results = []
    for selected in _parse_ints(args.selected):
        selected = max(0, min(int(selected), int(args.total_tokens)))
        if selected == 0:
            continue
        # Use valid tokens from the sealed indexed region for every query in this benchmark.
        base = torch.arange(selected, device=device, dtype=torch.long)
        dyn_start = min(max(0, args.static_prefix), args.query_start + 1)
        max_valid = max(1, min(args.total_tokens, args.query_start + 1 - max(0, args.static_suffix)) - dyn_start)
        tokens_1d = dyn_start + torch.remainder(base, max_valid)
        ranked_tokens = tokens_1d.reshape(1, 1, selected).expand(args.positions, args.heads, selected).contiguous()
        ranked_scores = torch.zeros((args.positions, args.heads, selected), device=device, dtype=torch.float32)

        def run():
            return selector_paged_pq.gqa_causal_exact_selected_attention(
                queries,
                keys,
                values,
                ranked_tokens,
                ranked_scores,
                group_size,
                args.query_start,
                args.static_prefix,
                args.static_suffix,
                args.page_size,
                scale,
            )

        out = run()
        torch.cuda.synchronize()
        if out.shape != (args.positions, args.heads, args.dim):
            raise AssertionError(f"unexpected output shape {tuple(out.shape)}")
        ms = _time_ms(run, warmup=args.warmup, iters=args.iters)
        logical_kv_mb = (
            float(args.positions * args.heads * selected * args.dim * 2 * torch.finfo(kv_dtype).bits / 8)
            / (1024.0 * 1024.0)
        )
        results.append(
            {
                "selected": selected,
                "ms": ms,
                "logical_kv_mb": logical_kv_mb,
                "logical_gb_per_s": logical_kv_mb / 1024.0 / (ms / 1000.0) if ms > 0 else float("inf"),
            }
        )

    print(
        json.dumps(
            {
                "positions": args.positions,
                "heads": args.heads,
                "kv_heads": args.kv_heads,
                "dim": args.dim,
                "total_tokens": args.total_tokens,
                "query_start": args.query_start,
                "static_prefix": args.static_prefix,
                "static_suffix": args.static_suffix,
                "page_size": args.page_size,
                "kv_dtype": args.kv_dtype,
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
