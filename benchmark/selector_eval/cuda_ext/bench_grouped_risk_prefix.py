#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CUDA_EXT_DIR = PROJECT_ROOT / "benchmark" / "selector_eval" / "cuda_ext"
if str(CUDA_EXT_DIR) not in sys.path:
    sys.path.insert(0, str(CUDA_EXT_DIR))


def _sync() -> None:
    torch.cuda.synchronize()


def _time_call(fn, *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    _sync()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    _sync()
    return float(start.elapsed_time(end)) / 1000.0 / max(1, int(iters))


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark grouped vs repeated residual-risk V-prefix CUDA helper.")
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--k_count", type=int, default=4)
    parser.add_argument("--heads_per_group", type=int, default=4)
    parser.add_argument("--context_len", type=int, default=32768)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--v_budgets", default="1024,2048,4096,6144,8192,12288,16384")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260523)
    args = parser.parse_args()

    from selector_paged_pq import (  # noqa: PLC0415
        joint_vprefix_outputs_from_grouped_risk,
        joint_vprefix_outputs_from_grouped_risk_batched,
        joint_vprefix_outputs_from_risk,
    )

    torch.manual_seed(int(args.seed))
    device = torch.device("cuda")
    groups = int(args.groups)
    k_count = int(args.k_count)
    heads = int(args.heads_per_group)
    context = int(args.context_len)
    dim = int(args.dim)
    rows_per_group = k_count * heads
    rows = groups * rows_per_group
    v_budgets = torch.tensor(
        [int(x) for x in str(args.v_budgets).split(",") if x.strip()],
        dtype=torch.long,
        device=device,
    )

    base = torch.randn((groups, k_count, heads, dim), device=device, dtype=torch.float32)
    probs = torch.softmax(
        torch.randn((groups, k_count, heads, context), device=device, dtype=torch.float32),
        dim=3,
    )
    residual = torch.randn((groups, context, dim), device=device, dtype=torch.float32)
    code_error = torch.rand((groups, context), device=device, dtype=torch.float32)
    row_group_ids = torch.arange(groups, device=device, dtype=torch.long).repeat_interleave(rows_per_group)
    base_flat = base.reshape(rows, dim).contiguous()
    probs_flat = probs.reshape(rows, context).contiguous()

    def repeated() -> list[torch.Tensor]:
        return [
            joint_vprefix_outputs_from_risk(
                base[g].contiguous(),
                probs[g].contiguous(),
                residual[g].contiguous(),
                code_error[g].contiguous(),
                v_budgets,
            )
            for g in range(groups)
        ]

    def grouped() -> torch.Tensor:
        return joint_vprefix_outputs_from_grouped_risk(
            base_flat,
            probs_flat,
            residual.contiguous(),
            code_error.contiguous(),
            row_group_ids,
            v_budgets,
        )

    def grouped_batched() -> torch.Tensor:
        return joint_vprefix_outputs_from_grouped_risk_batched(
            base.contiguous(),
            probs.contiguous(),
            residual.contiguous(),
            code_error.contiguous(),
            v_budgets,
        )

    repeated_out = torch.stack(repeated(), dim=0)
    grouped_out = grouped().reshape(groups, k_count, heads, int(v_budgets.numel()), dim).permute(0, 1, 3, 2, 4)
    grouped_batched_out = grouped_batched().reshape(groups, k_count, heads, int(v_budgets.numel()), dim).permute(0, 1, 3, 2, 4)
    max_abs_diff = float(torch.max(torch.abs(repeated_out - grouped_out)).item())
    max_abs_diff_batched = float(torch.max(torch.abs(repeated_out - grouped_batched_out)).item())
    repeated_seconds = _time_call(repeated, warmup=int(args.warmup), iters=int(args.iters))
    grouped_seconds = _time_call(grouped, warmup=int(args.warmup), iters=int(args.iters))
    grouped_batched_seconds = _time_call(grouped_batched, warmup=int(args.warmup), iters=int(args.iters))
    payload = {
        "groups": groups,
        "k_count": k_count,
        "heads_per_group": heads,
        "context_len": context,
        "dim": dim,
        "v_steps": int(v_budgets.numel()),
        "rows": rows,
        "repeated_seconds_per_call": repeated_seconds,
        "grouped_seconds_per_call": grouped_seconds,
        "grouped_batched_seconds_per_call": grouped_batched_seconds,
        "speedup": repeated_seconds / grouped_seconds if grouped_seconds > 0 else float("inf"),
        "batched_speedup_vs_repeated": repeated_seconds / grouped_batched_seconds if grouped_batched_seconds > 0 else float("inf"),
        "batched_speedup_vs_grouped": grouped_seconds / grouped_batched_seconds if grouped_batched_seconds > 0 else float("inf"),
        "max_abs_diff": max_abs_diff,
        "max_abs_diff_batched": max_abs_diff_batched,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
