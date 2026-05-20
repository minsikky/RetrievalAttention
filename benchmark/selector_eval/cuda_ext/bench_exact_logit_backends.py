#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.selector_eval.runners.run_hf_paged_pq_intervention_eval import (  # noqa: E402
    _gpu_gqa_dense_decode_ranked_logits_and_base_lse,
    _gpu_gqa_ranked_exact_logits,
    load_selector_paged_pq_ext,
)


def _time_cuda(fn, *, warmup: int, iters: int) -> tuple[torch.Tensor, float]:
    out = None
    for _ in range(max(0, int(warmup))):
        out = fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(max(1, int(iters))):
        out = fn()
    end.record()
    torch.cuda.synchronize()
    return out, float(start.elapsed_time(end)) / float(max(1, int(iters)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Microbenchmark exact-logit simulator backends.")
    parser.add_argument("--context", type=int, default=32768)
    parser.add_argument("--rank", type=int, default=32768)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--kv_heads", type=int, default=8)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--static_prefix", type=int, default=128)
    parser.add_argument("--static_suffix", type=int, default=128)
    parser.add_argument("--page_size", type=int, default=5632)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260520)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(int(args.seed))
    device = torch.device("cuda")
    heads = int(args.heads)
    kv_heads = int(args.kv_heads)
    group_size = max(1, heads // max(1, kv_heads))
    context = int(args.context)
    rank = min(int(args.rank), context)
    dim = int(args.dim)
    scale = float(dim) ** -0.5
    queries = torch.randn((heads, dim), device=device, dtype=torch.float32)
    keys = torch.randn((kv_heads, context, dim), device=device, dtype=torch.float16)
    ranked_tokens = torch.randint(0, context, (heads, rank), device=device, dtype=torch.long)
    ranked_scores = torch.randn((heads, rank), device=device, dtype=torch.float32)
    keys_t_float = keys.float().transpose(1, 2).contiguous()
    native = load_selector_paged_pq_ext()

    def ranked_gather():
        return _gpu_gqa_ranked_exact_logits(
            queries=queries,
            keys_all=keys,
            ranked_tokens=ranked_tokens,
            group_size=group_size,
            scale=scale,
            max_rank=rank,
        )

    def dense_sim():
        out, _base, _base_count, _key_count = _gpu_gqa_dense_decode_ranked_logits_and_base_lse(
            queries=queries,
            keys_all=keys,
            keys_all_t_float=None,
            ranked_tokens=ranked_tokens,
            group_size=group_size,
            scale=scale,
            max_rank=rank,
            query_context_len=context,
            static_prefix=int(args.static_prefix),
            static_suffix=int(args.static_suffix),
            page_size=int(args.page_size),
            need_base_lse=False,
        )
        return out

    def dense_sim_cached():
        out, _base, _base_count, _key_count = _gpu_gqa_dense_decode_ranked_logits_and_base_lse(
            queries=queries,
            keys_all=keys,
            keys_all_t_float=keys_t_float,
            ranked_tokens=ranked_tokens,
            group_size=group_size,
            scale=scale,
            max_rank=rank,
            query_context_len=context,
            static_prefix=int(args.static_prefix),
            static_suffix=int(args.static_suffix),
            page_size=int(args.page_size),
            need_base_lse=False,
        )
        return out

    def native_ranked_exact():
        if not hasattr(native, "gqa_decode_ranked_exact_logits"):
            raise RuntimeError("native extension does not expose gqa_decode_ranked_exact_logits")
        return native.gqa_decode_ranked_exact_logits(
            queries,
            keys,
            ranked_tokens,
            ranked_scores,
            int(group_size),
            int(context),
            int(args.static_prefix),
            int(args.static_suffix),
            int(args.page_size),
            float(scale),
        )

    def native_ranked_with_base():
        if not hasattr(native, "gqa_decode_ranked_exact_logits_with_base_lse"):
            raise RuntimeError("native extension does not expose gqa_decode_ranked_exact_logits_with_base_lse")
        out, _base_lse = native.gqa_decode_ranked_exact_logits_with_base_lse(
            queries,
            keys,
            ranked_tokens,
            ranked_scores,
            int(group_size),
            int(context),
            int(args.static_prefix),
            int(args.static_suffix),
            int(args.page_size),
            float(scale),
        )
        return out, _base_lse

    ranked_out, ranked_ms = _time_cuda(ranked_gather, warmup=int(args.warmup), iters=int(args.iters))
    dense_out, dense_ms = _time_cuda(dense_sim, warmup=int(args.warmup), iters=int(args.iters))
    cached_out, cached_dense_ms = _time_cuda(dense_sim_cached, warmup=int(args.warmup), iters=int(args.iters))
    if hasattr(native, "gqa_decode_ranked_exact_logits_with_base_lse"):
        native_ref, native_ref_ms = _time_cuda(native_ranked_exact, warmup=int(args.warmup), iters=int(args.iters))
        native_result, native_ms = _time_cuda(native_ranked_with_base, warmup=int(args.warmup), iters=int(args.iters))
        native_out, _native_base_lse = native_result
        finite = torch.isfinite(native_ref) & torch.isfinite(native_out)
        if bool(torch.any(finite)):
            native_max_diff = float((native_ref[finite] - native_out[finite]).abs().max().item())
        else:
            native_max_diff = 0.0
        native_inf_mask_mismatch = int((torch.isfinite(native_ref) != torch.isfinite(native_out)).sum().item())
    else:
        native_ref_ms = None
        native_ms = None
        native_max_diff = None
        native_inf_mask_mismatch = None
    max_diff = float((ranked_out - dense_out).abs().max().item())
    cached_max_diff = float((ranked_out - cached_out).abs().max().item())
    payload = {
        "context": context,
        "rank": rank,
        "heads": heads,
        "kv_heads": kv_heads,
        "group_size": group_size,
        "dim": dim,
        "ranked_gather_ms": ranked_ms,
        "dense_sim_ms": dense_ms,
        "dense_sim_cached_ms": cached_dense_ms,
        "native_ranked_exact_ms": native_ref_ms,
        "native_ranked_with_base_ms": native_ms,
        "speedup_dense_vs_ranked": float(ranked_ms / dense_ms) if dense_ms > 0.0 else float("inf"),
        "speedup_dense_cached_vs_ranked": float(ranked_ms / cached_dense_ms) if cached_dense_ms > 0.0 else float("inf"),
        "speedup_cached_vs_uncached_dense": float(dense_ms / cached_dense_ms) if cached_dense_ms > 0.0 else float("inf"),
        "speedup_native_with_base_vs_ranked": (
            float(ranked_ms / native_ms) if native_ms is not None and native_ms > 0.0 else None
        ),
        "max_abs_diff": max_diff,
        "cached_max_abs_diff": cached_max_diff,
        "native_max_abs_diff": native_max_diff,
        "native_inf_mask_mismatch": native_inf_mask_mismatch,
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
