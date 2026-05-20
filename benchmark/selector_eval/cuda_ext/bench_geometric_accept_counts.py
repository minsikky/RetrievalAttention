#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--pages", type=int, default=128)
    parser.add_argument("--page-size", type=int, default=256)
    parser.add_argument("--ranked", type=int, default=4096)
    parser.add_argument("--value-subvecs", type=int, default=8)
    parser.add_argument("--value-centroids", type=int, default=16)
    parser.add_argument("--min-budget", type=int, default=512)
    parser.add_argument("--max-budget", type=int, default=4096)
    parser.add_argument("--granularity", type=int, default=512)
    parser.add_argument("--growth", type=float, default=1.5)
    parser.add_argument("--probe-scale", type=float, default=1.125)
    parser.add_argument("--rel-l2-max", type=float, default=0.04)
    parser.add_argument("--exact-value-top", type=int, default=-2048)
    parser.add_argument("--exact-value-mass", type=float, default=0.0)
    parser.add_argument("--exact-value-min-top", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260518)
    parser.add_argument("--output", default="")
    parser.add_argument(
        "--mode",
        choices=["strict", "tail_stability"],
        default="strict",
        help="strict compares compressed-tail output to larger exact output; tail_stability compares two compressed-tail budgets",
    )
    parser.add_argument(
        "--skip-old-loop",
        action="store_true",
        help="skip the repeated-output reference loop; useful for large 32k/128k-sized timing runs",
    )
    return parser.parse_args()


def round_up_budget(value: int, granularity: int, max_budget: int) -> int:
    if granularity <= 1:
        return min(value, max_budget)
    return min(((value + granularity - 1) // granularity) * granularity, max_budget)


def geometric_budget_pairs(
    *,
    min_budget: int,
    max_budget: int,
    granularity: int,
    growth: float,
    probe_scale: float,
) -> tuple[list[int], list[int]]:
    tails: list[int] = []
    probes: list[int] = []
    tail_budget = round_up_budget(int(min_budget), int(granularity), int(max_budget))
    while tail_budget < max_budget:
        probe_budget = round_up_budget(
            int(max(float(tail_budget + granularity), float(probe_scale) * float(tail_budget))),
            int(granularity),
            int(max_budget),
        )
        probe_budget = max(tail_budget, int(probe_budget))
        tails.append(int(tail_budget))
        probes.append(int(probe_budget))
        if probe_budget >= max_budget:
            break
        next_budget = round_up_budget(
            int(max(float(probe_budget + granularity), float(growth) * float(probe_budget))),
            int(granularity),
            int(max_budget),
        )
        if next_budget <= probe_budget:
            break
        tail_budget = int(next_budget)
    return tails, probes


def selected_mass_thresholds_from_logits(
    *,
    ranked_logits: torch.Tensor,
    ranked_scores: torch.Tensor,
    base_logsumexp: torch.Tensor,
    budgets: list[int],
    exact_mass: float,
    min_top: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    heads = int(ranked_logits.shape[0])
    steps = len(budgets)
    device = ranked_logits.device
    thresholds = torch.empty((heads, steps), dtype=torch.float32, device=device)
    threshold_sels = torch.empty((heads, steps), dtype=torch.long, device=device)
    if steps == 0:
        return thresholds, threshold_sels
    rank = int(ranked_logits.shape[-1])
    target = float(max(0.0, min(1.0, float(exact_mass))))
    valid_all = torch.isfinite(ranked_scores[:, :rank]) & torch.isfinite(ranked_logits[:, :rank])
    logits_all = torch.where(
        valid_all,
        ranked_logits[:, :rank].float(),
        torch.full((heads, rank), float("-inf"), dtype=torch.float32, device=device),
    )
    sorted_logits_all, sorted_order_all = torch.sort(logits_all, dim=-1, descending=True, stable=True)
    sorted_valid_all = torch.gather(valid_all, 1, sorted_order_all)
    prefix_lse_all = torch.logcumsumexp(logits_all, dim=-1)
    prefix_valid_counts = torch.cumsum(valid_all.to(torch.long), dim=-1)
    budgets_tensor = torch.tensor(
        [max(0, min(rank, int(budget))) for budget in budgets],
        dtype=torch.long,
        device=device,
    )
    positive_steps = budgets_tensor > 0
    if not bool(torch.any(positive_steps)):
        thresholds.fill_(float("inf"))
        threshold_sels.fill_(-1)
        return thresholds.contiguous(), threshold_sels.contiguous()

    lse_idx = torch.clamp(budgets_tensor - 1, min=0)
    ranked_lse = prefix_lse_all.index_select(1, lse_idx)
    base_lse = base_logsumexp.float().reshape(heads, 1)
    total_lse = torch.logaddexp(base_lse, ranked_lse)
    valid_count = prefix_valid_counts.index_select(1, lse_idx)
    valid_count = torch.where(positive_steps.reshape(1, steps), valid_count, torch.zeros_like(valid_count))
    budget_view = budgets_tensor.reshape(1, steps, 1)
    in_budget_any = sorted_order_all.unsqueeze(1) < budget_view
    in_budget_sorted = in_budget_any & sorted_valid_all.unsqueeze(1)
    sorted_logits = sorted_logits_all.unsqueeze(1).expand(-1, steps, -1).masked_fill(
        ~in_budget_sorted,
        float("-inf"),
    )
    cum_selected_count = torch.cumsum(in_budget_sorted.to(torch.long), dim=-1)
    if target <= 0.0:
        counts = torch.zeros((heads, steps), dtype=torch.long, device=device)
    else:
        base_mass = torch.where(
            torch.isfinite(total_lse),
            torch.exp(base_lse - total_lse),
            torch.zeros_like(total_lse),
        )
        cum_ranked_lse = torch.logcumsumexp(sorted_logits, dim=-1)
        cum_lse = torch.logaddexp(base_lse.unsqueeze(-1), cum_ranked_lse)
        cum_mass = torch.where(
            torch.isfinite(total_lse).unsqueeze(-1),
            torch.exp(cum_lse - total_lse.unsqueeze(-1)),
            torch.zeros_like(cum_lse),
        )
        hit = (cum_mass >= min(target, 1.0 - 1.0e-7)) & in_budget_sorted
        has_hit = torch.any(hit, dim=-1)
        first_hit_pos = torch.argmax(hit.to(torch.int32), dim=-1).to(torch.long)
        first_hit_count = torch.gather(cum_selected_count, 2, first_hit_pos.unsqueeze(-1)).squeeze(-1)
        counts = torch.where(
            base_mass >= target,
            torch.zeros_like(first_hit_count),
            torch.where(has_hit, first_hit_count, valid_count),
        )
    if int(min_top) > 0:
        min_counts = torch.minimum(
            budgets_tensor.reshape(1, steps),
            torch.full((heads, steps), int(min_top), dtype=torch.long, device=device),
        )
        counts = torch.maximum(counts, min_counts)
    counts = torch.minimum(torch.clamp(counts, min=0), budgets_tensor.reshape(1, steps))
    counts = torch.where(positive_steps.reshape(1, steps), counts, torch.zeros_like(counts))
    has_exact = counts > 0
    cum_budget_count = torch.cumsum(in_budget_any.to(torch.long), dim=-1)
    kth_valid_mask = in_budget_sorted & (cum_selected_count >= counts.unsqueeze(-1))
    kth_any_mask = in_budget_any & (cum_budget_count >= counts.unsqueeze(-1))
    kth_mask = torch.where((counts <= valid_count).unsqueeze(-1), kth_valid_mask, kth_any_mask)
    kth_pos = torch.argmax(kth_mask.to(torch.int32), dim=-1).to(torch.long)
    threshold_vals = torch.gather(
        sorted_logits_all.unsqueeze(1).expand(-1, steps, -1),
        2,
        kth_pos.unsqueeze(-1),
    ).squeeze(-1)
    threshold_idx = torch.gather(
        sorted_order_all.unsqueeze(1).expand(-1, steps, -1),
        2,
        kth_pos.unsqueeze(-1),
    ).squeeze(-1).to(torch.long)
    thresholds[:, :] = torch.where(
        has_exact,
        threshold_vals,
        torch.full_like(threshold_vals, float("inf")),
    )
    threshold_sels[:, :] = torch.where(
        has_exact,
        threshold_idx,
        torch.full_like(threshold_idx, -1),
    )
    return thresholds.contiguous(), threshold_sels.contiguous()


def elapsed_ms(fn, *, warmup: int, iters: int) -> tuple[float, object]:
    result = None
    for _ in range(max(0, warmup)):
        result = fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(max(1, iters)):
        result = fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end)) / float(max(1, iters)), result


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import selector_paged_pq as native  # noqa: PLC0415

    device = torch.device("cuda")
    torch.manual_seed(int(args.seed))
    heads = int(args.heads)
    kv_heads = int(args.kv_heads)
    group_size = max(1, heads // kv_heads)
    dim = int(args.dim)
    pages = int(args.pages)
    page_size = int(args.page_size)
    total_tokens = pages * page_size
    ranked = min(int(args.ranked), total_tokens)
    value_subvecs = int(args.value_subvecs)
    value_centroids = int(args.value_centroids)
    value_subdim = dim // value_subvecs
    if value_subdim * value_subvecs != dim:
        raise ValueError("dim must be divisible by value_subvecs")

    queries = torch.randn((heads, dim), device=device, dtype=torch.float32)
    keys = torch.randn((kv_heads, total_tokens, dim), device=device, dtype=torch.float16)
    values = torch.randn((kv_heads, total_tokens, dim), device=device, dtype=torch.float16)
    dense_pq_scores = torch.randn((heads, total_tokens), device=device, dtype=torch.float32)
    value_codebooks = torch.randn(
        (kv_heads, pages, value_subvecs, value_centroids, value_subdim),
        device=device,
        dtype=torch.float32,
    )
    value_codes = torch.randint(
        0,
        value_centroids,
        (kv_heads, pages, page_size, value_subvecs),
        device=device,
        dtype=torch.uint8,
    )
    page_starts = torch.arange(0, total_tokens, page_size, device=device, dtype=torch.long)
    ranked_tokens = torch.stack(
        [torch.randperm(total_tokens, device=device, dtype=torch.long)[:ranked] for _ in range(heads)],
        dim=0,
    ).contiguous()
    ranked_scores = torch.take_along_dim(dense_pq_scores, ranked_tokens, dim=1).contiguous()
    scale = float(dim) ** -0.5
    rank_ids = torch.arange(ranked, device=device, dtype=torch.long).reshape(1, ranked)
    ranked_logits = torch.empty_like(ranked_scores)
    for kv_head in range(kv_heads):
        head_start = kv_head * group_size
        head_end = min(heads, head_start + group_size)
        if head_start >= head_end:
            continue
        toks = ranked_tokens[head_start:head_end].clamp(min=0, max=total_tokens - 1)
        gathered = keys[kv_head].index_select(0, toks.reshape(-1)).reshape(head_end - head_start, ranked, dim)
        ranked_logits[head_start:head_end] = torch.sum(
            queries[head_start:head_end].unsqueeze(1) * gathered.float(),
            dim=-1,
        ) * scale
    prefix_end = min(128, total_tokens)
    indexed_end = max(prefix_end, total_tokens - 512)
    sealed_end = prefix_end + ((max(0, indexed_end - prefix_end) // page_size) * page_size)
    base_tail_start = max(sealed_end, prefix_end)
    ranked_base_mask = (ranked_tokens < prefix_end) | (
        (ranked_tokens >= base_tail_start) & (ranked_tokens < total_tokens)
    )
    ranked_logits = ranked_logits.masked_fill(ranked_base_mask, float("-inf")).contiguous()
    base_token_parts = []
    if prefix_end > 0:
        base_token_parts.append(torch.arange(0, prefix_end, device=device, dtype=torch.long))
    if base_tail_start < total_tokens:
        base_token_parts.append(torch.arange(base_tail_start, total_tokens, device=device, dtype=torch.long))
    if base_token_parts:
        base_tokens = torch.cat(base_token_parts)
        base_logits = torch.empty((heads, int(base_tokens.numel())), device=device, dtype=torch.float32)
        for kv_head in range(kv_heads):
            head_start = kv_head * group_size
            head_end = min(heads, head_start + group_size)
            if head_start >= head_end:
                continue
            gathered = keys[kv_head].index_select(0, base_tokens)
            base_logits[head_start:head_end] = torch.sum(
                queries[head_start:head_end].unsqueeze(1) * gathered.float().unsqueeze(0),
                dim=-1,
            ) * scale
        base_lse = torch.logsumexp(base_logits, dim=-1)
    else:
        base_lse = torch.full((heads,), float("-inf"), device=device, dtype=torch.float32)

    min_budget = min(int(args.min_budget), ranked)
    max_budget = min(int(args.max_budget), ranked)
    granularity = max(1, int(args.granularity))
    exact_value_top = int(args.exact_value_top)

    def mask_scores(keep: int) -> torch.Tensor:
        return ranked_scores.masked_fill(rank_ids >= int(keep), float("-inf")).contiguous()

    use_selected_mass = float(args.exact_value_mass) > 0.0

    def selected_tail(masked_scores: torch.Tensor, blend: float) -> torch.Tensor:
        if use_selected_mass:
            return native.gqa_decode_vpq_selected_tail_agg_from_logits_mass_min(
                queries,
                keys,
                values,
                dense_pq_scores,
                value_codebooks,
                value_codes,
                page_starts,
                ranked_tokens,
                masked_scores,
                ranked_logits,
                float(args.exact_value_mass),
                int(args.exact_value_min_top),
                group_size,
                total_tokens,
                128,
                512,
                page_size,
                scale,
                blend,
            )
        return native.gqa_decode_vpq_selected_tail_agg_from_scores(
            queries,
            keys,
            values,
            dense_pq_scores,
            value_codebooks,
            value_codes,
            page_starts,
            ranked_tokens,
            masked_scores,
            group_size,
            total_tokens,
            128,
            512,
            page_size,
            exact_value_top,
            scale,
            blend,
        )

    def old_loop() -> torch.Tensor:
        expected = torch.full((heads,), max_budget, dtype=torch.long, device=device)
        unresolved = torch.ones((heads,), dtype=torch.bool, device=device)
        k = round_up_budget(min_budget, granularity, max_budget)
        while True:
            tail_budget = min(max_budget, int(k))
            probe_budget = round_up_budget(
                int(max(float(tail_budget + granularity), args.probe_scale * float(tail_budget))),
                granularity,
                max_budget,
            )
            probe_budget = max(tail_budget, int(probe_budget))
            approx_tail = selected_tail(mask_scores(tail_budget), 1.0)
            probe_blend = 1.0 if str(args.mode) == "tail_stability" else 0.0
            probe_only = selected_tail(mask_scores(probe_budget), probe_blend)
            rel = torch.linalg.vector_norm(approx_tail - probe_only, dim=-1) / torch.clamp(
                torch.linalg.vector_norm(probe_only, dim=-1),
                min=1e-20,
            )
            passed = (rel <= float(args.rel_l2_max)) & unresolved
            expected = torch.where(passed, torch.full_like(expected, probe_budget), expected)
            unresolved = unresolved & ~passed
            if not bool(torch.any(unresolved)) or probe_budget >= max_budget:
                break
            next_k = round_up_budget(
                int(max(float(probe_budget + granularity), float(args.growth) * float(probe_budget))),
                granularity,
                max_budget,
            )
            if next_k <= probe_budget:
                break
            k = next_k
        return expected

    def native_counts() -> torch.Tensor:
        if (
            use_selected_mass
            and hasattr(native, "gqa_decode_geometric_accept_counts_vpq_mass_min_proxy_from_logits_thresholds")
        ):
            tail_budgets, probe_budgets = geometric_budget_pairs(
                min_budget=min_budget,
                max_budget=max_budget,
                granularity=granularity,
                growth=float(args.growth),
                probe_scale=float(args.probe_scale),
            )
            combined_budgets = sorted({int(v) for v in tail_budgets} | {int(v) for v in probe_budgets})
            combined_thresholds, combined_threshold_sels = selected_mass_thresholds_from_logits(
                ranked_logits=ranked_logits,
                ranked_scores=ranked_scores,
                base_logsumexp=base_lse,
                budgets=combined_budgets,
                exact_mass=float(args.exact_value_mass),
                min_top=int(args.exact_value_min_top),
            )
            budget_to_col = {int(budget): int(idx) for idx, budget in enumerate(combined_budgets)}
            approx_cols = torch.tensor(
                [budget_to_col[int(v)] for v in tail_budgets],
                dtype=torch.long,
                device=ranked_logits.device,
            )
            probe_cols = torch.tensor(
                [budget_to_col[int(v)] for v in probe_budgets],
                dtype=torch.long,
                device=ranked_logits.device,
            )
            approx_thresholds = combined_thresholds.index_select(1, approx_cols).contiguous()
            approx_threshold_sels = combined_threshold_sels.index_select(1, approx_cols).contiguous()
            probe_thresholds = combined_thresholds.index_select(1, probe_cols).contiguous()
            probe_threshold_sels = combined_threshold_sels.index_select(1, probe_cols).contiguous()
            return native.gqa_decode_geometric_accept_counts_vpq_mass_min_proxy_from_logits_thresholds(
                queries,
                keys,
                values,
                dense_pq_scores,
                value_codebooks,
                value_codes,
                page_starts,
                ranked_tokens,
                ranked_scores,
                ranked_logits,
                approx_thresholds,
                approx_threshold_sels,
                probe_thresholds,
                probe_threshold_sels,
                group_size,
                total_tokens,
                128,
                512,
                page_size,
                min_budget,
                max_budget,
                granularity,
                float(args.growth),
                float(args.probe_scale),
                float(args.rel_l2_max),
                float(args.exact_value_mass),
                int(args.exact_value_min_top),
                scale,
                0.0,
                1.0,
                -1.0,
                float("inf"),
                False,
                str(args.mode) == "tail_stability",
            )
        if use_selected_mass and hasattr(native, "gqa_decode_geometric_accept_counts_vpq_mass_min_proxy_from_logits"):
            return native.gqa_decode_geometric_accept_counts_vpq_mass_min_proxy_from_logits(
                queries,
                keys,
                values,
                dense_pq_scores,
                value_codebooks,
                value_codes,
                page_starts,
                ranked_tokens,
                ranked_scores,
                ranked_logits,
                group_size,
                total_tokens,
                128,
                512,
                page_size,
                min_budget,
                max_budget,
                granularity,
                float(args.growth),
                float(args.probe_scale),
                float(args.rel_l2_max),
                float(args.exact_value_mass),
                int(args.exact_value_min_top),
                scale,
                0.0,
                1.0,
                -1.0,
                float("inf"),
                False,
                str(args.mode) == "tail_stability",
            )
        if use_selected_mass:
            return native.gqa_decode_geometric_accept_counts_vpq_mass_min_proxy(
                queries,
                keys,
                values,
                dense_pq_scores,
                value_codebooks,
                value_codes,
                page_starts,
                ranked_tokens,
                ranked_scores,
                group_size,
                total_tokens,
                128,
                512,
                page_size,
                min_budget,
                max_budget,
                granularity,
                float(args.growth),
                float(args.probe_scale),
                float(args.rel_l2_max),
                float(args.exact_value_mass),
                int(args.exact_value_min_top),
                scale,
                0.0,
                1.0,
                -1.0,
                float("inf"),
                False,
                str(args.mode) == "tail_stability",
            )
        native_fn = (
            native.gqa_decode_geometric_accept_counts_vpq_tail_stability
            if str(args.mode) == "tail_stability"
            else native.gqa_decode_geometric_accept_counts_vpq
        )
        return native_fn(
            queries,
            keys,
            values,
            dense_pq_scores,
            value_codebooks,
            value_codes,
            page_starts,
            ranked_tokens,
            ranked_scores,
            group_size,
            total_tokens,
            128,
            512,
            page_size,
            min_budget,
            max_budget,
            granularity,
            float(args.growth),
            float(args.probe_scale),
            float(args.rel_l2_max),
            exact_value_top,
            scale,
        )

    old_ms = None
    old_result = None
    if not bool(args.skip_old_loop):
        old_ms, old_result = elapsed_ms(old_loop, warmup=int(args.warmup), iters=int(args.iters))
    native_ms, native_result = elapsed_ms(native_counts, warmup=int(args.warmup), iters=int(args.iters))
    native_cpu = native_result.detach().cpu()
    old_cpu = old_result.detach().cpu() if old_result is not None else None
    counts_match = bool(torch.equal(old_cpu, native_cpu)) if old_cpu is not None else None
    payload = {
        "heads": heads,
        "kv_heads": kv_heads,
        "dim": dim,
        "pages": pages,
        "page_size": page_size,
        "tokens": total_tokens,
        "ranked": ranked,
        "min_budget": min_budget,
        "max_budget": max_budget,
        "granularity": granularity,
        "exact_value_top": exact_value_top,
        "exact_value_mass": float(args.exact_value_mass),
        "exact_value_min_top": int(args.exact_value_min_top),
        "mode": str(args.mode),
        "old_loop_ms": old_ms,
        "native_ms": native_ms,
        "speedup": old_ms / native_ms if old_ms is not None and native_ms > 0.0 else None,
        "native_path": (
            "mass_min_proxy_from_logits_thresholds"
            if use_selected_mass and hasattr(native, "gqa_decode_geometric_accept_counts_vpq_mass_min_proxy_from_logits_thresholds")
            else "mass_min_proxy_from_logits"
            if use_selected_mass and hasattr(native, "gqa_decode_geometric_accept_counts_vpq_mass_min_proxy_from_logits")
            else "mass_min_proxy"
            if use_selected_mass
            else "vpq"
        ),
        "counts_match": counts_match,
        "old_counts": old_cpu.tolist() if old_cpu is not None else None,
        "native_counts": native_cpu.tolist(),
        "skip_old_loop": bool(args.skip_old_loop),
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.output:
        Path(args.output).write_text(text + "\n")
    if payload["counts_match"] is False:
        raise AssertionError("native counts did not match repeated-output reference")


if __name__ == "__main__":
    main()
