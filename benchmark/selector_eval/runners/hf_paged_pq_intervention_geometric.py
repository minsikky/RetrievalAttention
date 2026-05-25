#!/usr/bin/env python3
from __future__ import annotations

import os

import torch

from benchmark.selector_eval.gpu.run_gpu_paged_pq_eval import load_selector_paged_pq_ext
from benchmark.selector_eval.runners.hf_paged_pq_intervention_common import _env_truthy
from benchmark.selector_eval.runners.run_layer_quality_eval import _round_budget_up


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
    max_budget = max(0, int(max_budget))
    granularity = max(1, int(granularity))
    tail_budget = _round_budget_up(
        int(min_budget),
        granularity=granularity,
        max_budget=max_budget,
    )
    while tail_budget < max_budget:
        probe_budget = _round_budget_up(
            int(max(float(tail_budget + granularity), float(probe_scale) * float(tail_budget))),
            granularity=granularity,
            max_budget=max_budget,
        )
        probe_budget = max(tail_budget, int(probe_budget))
        tails.append(int(tail_budget))
        probes.append(int(probe_budget))
        if probe_budget >= max_budget:
            break
        next_budget = _round_budget_up(
            int(max(float(probe_budget + granularity), float(growth) * float(probe_budget))),
            granularity=granularity,
            max_budget=max_budget,
        )
        if next_budget <= probe_budget:
            break
        tail_budget = int(next_budget)
    return tails, probes


def selected_mass_thresholds_from_logits_gpu(
    *,
    ranked_logits: torch.Tensor,
    ranked_scores: torch.Tensor,
    base_logsumexp: torch.Tensor | None,
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
    base_lse = (
        base_logsumexp.float()
        if base_logsumexp is not None
        else torch.full((heads,), float("-inf"), dtype=torch.float32, device=device)
    )
    valid_all = torch.isfinite(ranked_scores[:, :rank]) & torch.isfinite(ranked_logits[:, :rank])
    logits_all = torch.where(
        valid_all,
        ranked_logits[:, :rank].float(),
        torch.full((heads, rank), float("-inf"), dtype=torch.float32, device=device),
    )
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

    if _env_truthy("SELECTOR_PQ_THRESHOLD_MIN_TOP_FAST") and int(min_top) > 0:
        k_min = max(1, min(rank, int(min_top)))
        top_logits_min, top_order_min = torch.topk(logits_all, k=k_min, dim=-1, largest=True, sorted=True)
        top_valid_min = torch.isfinite(top_logits_min)
        lse_idx = torch.clamp(budgets_tensor - 1, min=0)
        ranked_lse = prefix_lse_all.index_select(1, lse_idx)
        total_lse = torch.logaddexp(base_lse.reshape(heads, 1), ranked_lse)
        valid_count = prefix_valid_counts.index_select(1, lse_idx)
        valid_count = torch.where(positive_steps.reshape(1, steps), valid_count, torch.zeros_like(valid_count))
        min_counts = torch.minimum(
            budgets_tensor.reshape(1, steps),
            torch.full((heads, steps), int(min_top), dtype=torch.long, device=device),
        )
        min_counts = torch.where(positive_steps.reshape(1, steps), min_counts, torch.zeros_like(min_counts))
        budget_view = budgets_tensor.reshape(1, steps, 1)
        in_budget_any = top_order_min.unsqueeze(1) < budget_view
        in_budget_sorted = in_budget_any & top_valid_min.unsqueeze(1)
        cum_selected_count = torch.cumsum(in_budget_sorted.to(torch.long), dim=-1)
        top_valid_count = cum_selected_count[..., -1]
        has_exact = min_counts > 0
        kth_valid_mask = in_budget_sorted & (cum_selected_count >= min_counts.unsqueeze(-1))
        kth_pos = torch.argmax(kth_valid_mask.to(torch.int32), dim=-1).to(torch.long)
        if target <= 0.0:
            mass_ok = torch.ones((heads, steps), dtype=torch.bool, device=device)
        else:
            sorted_logits_min = top_logits_min.unsqueeze(1).expand(-1, steps, -1).masked_fill(
                ~in_budget_sorted,
                float("-inf"),
            )
            cum_ranked_lse = torch.logcumsumexp(sorted_logits_min, dim=-1)
            cum_lse = torch.logaddexp(base_lse.reshape(heads, 1, 1), cum_ranked_lse)
            cum_mass = torch.where(
                torch.isfinite(total_lse).unsqueeze(-1),
                torch.exp(cum_lse - total_lse.unsqueeze(-1)),
                torch.zeros_like(cum_lse),
            )
            mass_at_min = torch.gather(cum_mass, 2, kth_pos.unsqueeze(-1)).squeeze(-1)
            mass_ok = mass_at_min >= min(target, 1.0 - 1.0e-7)
        sufficient = (
            (~positive_steps.reshape(1, steps))
            | (~has_exact)
            | ((top_valid_count >= min_counts) & (min_counts <= valid_count) & mass_ok)
        )
        if bool(torch.all(sufficient)):
            threshold_vals = torch.gather(
                top_logits_min.unsqueeze(1).expand(-1, steps, -1),
                2,
                kth_pos.unsqueeze(-1),
            ).squeeze(-1)
            threshold_idx = torch.gather(
                top_order_min.unsqueeze(1).expand(-1, steps, -1),
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

    topk_limit = 0
    try:
        topk_limit = int(os.environ.get("SELECTOR_PQ_THRESHOLD_TOPK", "0"))
    except ValueError:
        topk_limit = 0
    if 0 < topk_limit <= rank:
        k_top = max(1, min(rank, int(topk_limit)))
        top_logits_all, top_order_all = torch.topk(logits_all, k=k_top, dim=-1, largest=True, sorted=True)
        top_valid_all = torch.isfinite(top_logits_all)
        if _env_truthy("SELECTOR_PQ_THRESHOLD_NATIVE_TOPK"):
            try:
                native = load_selector_paged_pq_ext()
                native_threshold_fn = getattr(native, "selected_mass_thresholds_from_topk", None)
            except Exception:
                native_threshold_fn = None
            if native_threshold_fn is not None:
                native_thresholds, native_threshold_sels, native_sufficient = native_threshold_fn(
                    top_logits_all.contiguous(),
                    top_order_all.contiguous(),
                    prefix_lse_all.contiguous(),
                    prefix_valid_counts.contiguous(),
                    base_lse.contiguous(),
                    budgets_tensor.contiguous(),
                    float(target),
                    int(min_top),
                )
                if k_top >= rank or _env_truthy("SELECTOR_PQ_THRESHOLD_TOPK_ASSUME_SUFFICIENT") or bool(
                    torch.all(native_sufficient.to(torch.bool))
                ):
                    return native_thresholds.contiguous(), native_threshold_sels.contiguous()
        lse_idx = torch.clamp(budgets_tensor - 1, min=0)
        ranked_lse = prefix_lse_all.index_select(1, lse_idx)
        total_lse = torch.logaddexp(base_lse.reshape(heads, 1), ranked_lse)
        valid_count = prefix_valid_counts.index_select(1, lse_idx)
        valid_count = torch.where(positive_steps.reshape(1, steps), valid_count, torch.zeros_like(valid_count))
        budget_view = budgets_tensor.reshape(1, steps, 1)
        in_budget_any = top_order_all.unsqueeze(1) < budget_view
        in_budget_sorted = in_budget_any & top_valid_all.unsqueeze(1)
        sorted_logits = top_logits_all.unsqueeze(1).expand(-1, steps, -1).masked_fill(
            ~in_budget_sorted,
            float("-inf"),
        )
        cum_selected_count = torch.cumsum(in_budget_sorted.to(torch.long), dim=-1)
        top_valid_count = cum_selected_count[..., -1]
        if target <= 0.0:
            counts = torch.zeros((heads, steps), dtype=torch.long, device=device)
            has_hit = torch.ones((heads, steps), dtype=torch.bool, device=device)
            base_mass = torch.ones((heads, steps), dtype=torch.float32, device=device)
        else:
            base_mass = torch.where(
                torch.isfinite(total_lse),
                torch.exp(base_lse.reshape(heads, 1) - total_lse),
                torch.zeros_like(total_lse),
            )
            cum_ranked_lse = torch.logcumsumexp(sorted_logits, dim=-1)
            cum_lse = torch.logaddexp(base_lse.reshape(heads, 1, 1), cum_ranked_lse)
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
        if target <= 0.0:
            enough_for_target = torch.ones((heads, steps), dtype=torch.bool, device=device)
        else:
            enough_for_target = (base_mass >= target) | has_hit
        if int(min_top) > 0:
            enough_for_target = enough_for_target | (counts <= int(min_top))
        topk_sufficient = torch.all((~has_exact) | (enough_for_target & (counts <= top_valid_count)))
        # Diagnostic-only fast path: this removes the per-call GPU->CPU sync used
        # to prove fallback safety. Do not enable by default unless a separate
        # correctness check proves the configured topk is always sufficient.
        if _env_truthy("SELECTOR_PQ_THRESHOLD_TOPK_ASSUME_SUFFICIENT") or bool(topk_sufficient):
            cum_budget_count = torch.cumsum(in_budget_any.to(torch.long), dim=-1)
            kth_valid_mask = in_budget_sorted & (cum_selected_count >= counts.unsqueeze(-1))
            kth_any_mask = in_budget_any & (cum_budget_count >= counts.unsqueeze(-1))
            kth_mask = torch.where((counts <= valid_count).unsqueeze(-1), kth_valid_mask, kth_any_mask)
            kth_pos = torch.argmax(kth_mask.to(torch.int32), dim=-1).to(torch.long)
            threshold_vals = torch.gather(
                top_logits_all.unsqueeze(1).expand(-1, steps, -1),
                2,
                kth_pos.unsqueeze(-1),
            ).squeeze(-1)
            threshold_idx = torch.gather(
                top_order_all.unsqueeze(1).expand(-1, steps, -1),
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

    # Sort once across the full ranked prefix. Per-budget exact-V thresholds are
    # then computed by masking this sorted order to the active prefix. This keeps
    # frontier semantics identical while avoiding one O(rank log rank) sort per
    # geometric budget.
    sorted_logits_all, sorted_order_all = torch.sort(logits_all, dim=-1, descending=True, stable=True)
    sorted_valid_all = torch.gather(valid_all, 1, sorted_order_all)

    if _env_truthy("SELECTOR_PQ_THRESHOLD_LOOP"):
        base_lse_1d = base_lse.reshape(heads)
        thresholds.fill_(float("inf"))
        threshold_sels.fill_(-1)
        for step_idx, budget in enumerate([int(v) for v in budgets_tensor.detach().cpu().tolist()]):
            if budget <= 0:
                continue
            lse_idx = max(0, min(rank - 1, int(budget) - 1))
            ranked_lse = prefix_lse_all[:, lse_idx]
            total_lse = torch.logaddexp(base_lse_1d, ranked_lse)
            valid_count = prefix_valid_counts[:, lse_idx]
            in_budget_any = sorted_order_all < int(budget)
            in_budget_sorted = in_budget_any & sorted_valid_all
            sorted_logits = sorted_logits_all.masked_fill(~in_budget_sorted, float("-inf"))
            cum_selected_count = torch.cumsum(in_budget_sorted.to(torch.long), dim=-1)
            if target <= 0.0:
                counts = torch.zeros((heads,), dtype=torch.long, device=device)
            else:
                base_mass = torch.where(
                    torch.isfinite(total_lse),
                    torch.exp(base_lse_1d - total_lse),
                    torch.zeros_like(total_lse),
                )
                cum_ranked_lse = torch.logcumsumexp(sorted_logits, dim=-1)
                cum_lse = torch.logaddexp(base_lse_1d.reshape(heads, 1), cum_ranked_lse)
                cum_mass = torch.where(
                    torch.isfinite(total_lse).reshape(heads, 1),
                    torch.exp(cum_lse - total_lse.reshape(heads, 1)),
                    torch.zeros_like(cum_lse),
                )
                hit = (cum_mass >= min(target, 1.0 - 1.0e-7)) & in_budget_sorted
                has_hit = torch.any(hit, dim=-1)
                first_hit_pos = torch.argmax(hit.to(torch.int32), dim=-1).to(torch.long)
                first_hit_count = torch.gather(cum_selected_count, 1, first_hit_pos.reshape(heads, 1)).squeeze(1)
                counts = torch.where(
                    base_mass >= target,
                    torch.zeros_like(first_hit_count),
                    torch.where(has_hit, first_hit_count, valid_count),
                )
            if int(min_top) > 0:
                min_counts = torch.minimum(
                    torch.full((heads,), int(budget), dtype=torch.long, device=device),
                    torch.full((heads,), int(min_top), dtype=torch.long, device=device),
                )
                counts = torch.maximum(counts, min_counts)
            counts = torch.minimum(torch.clamp(counts, min=0), torch.full_like(counts, int(budget)))
            has_exact = counts > 0
            cum_budget_count = torch.cumsum(in_budget_any.to(torch.long), dim=-1)
            kth_valid_mask = in_budget_sorted & (cum_selected_count >= counts.reshape(heads, 1))
            kth_any_mask = in_budget_any & (cum_budget_count >= counts.reshape(heads, 1))
            kth_mask = torch.where((counts <= valid_count).reshape(heads, 1), kth_valid_mask, kth_any_mask)
            kth_pos = torch.argmax(kth_mask.to(torch.int32), dim=-1).to(torch.long)
            threshold_vals = torch.gather(sorted_logits_all, 1, kth_pos.reshape(heads, 1)).squeeze(1)
            threshold_idx = torch.gather(sorted_order_all, 1, kth_pos.reshape(heads, 1)).squeeze(1).to(torch.long)
            thresholds[:, step_idx] = torch.where(
                has_exact,
                threshold_vals,
                torch.full_like(threshold_vals, float("inf")),
            )
            threshold_sels[:, step_idx] = torch.where(
                has_exact,
                threshold_idx,
                torch.full_like(threshold_idx, -1),
            )
        return thresholds.contiguous(), threshold_sels.contiguous()

    lse_idx = torch.clamp(budgets_tensor - 1, min=0)
    ranked_lse = prefix_lse_all.index_select(1, lse_idx)
    total_lse = torch.logaddexp(base_lse.reshape(heads, 1), ranked_lse)
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
            torch.exp(base_lse.reshape(heads, 1) - total_lse),
            torch.zeros_like(total_lse),
        )
        cum_ranked_lse = torch.logcumsumexp(sorted_logits, dim=-1)
        cum_lse = torch.logaddexp(base_lse.reshape(heads, 1, 1), cum_ranked_lse)
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


def select_thresholds_for_budget_counts_gpu(
    *,
    thresholds: torch.Tensor,
    threshold_sels: torch.Tensor,
    budgets: list[int],
    counts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather per-head threshold rows for accepted geometric budgets."""

    heads = int(counts.shape[0])
    device = counts.device
    if len(budgets) == 0 or thresholds.numel() == 0:
        return (
            torch.full((heads,), float("inf"), dtype=torch.float32, device=device),
            torch.full((heads,), -1, dtype=torch.long, device=device),
        )
    budget_tensor = torch.tensor([int(v) for v in budgets], dtype=torch.long, device=device)
    idx = torch.searchsorted(budget_tensor, counts.to(torch.long), right=False)
    idx = torch.clamp(idx, min=0, max=len(budgets) - 1)
    row = torch.arange(heads, dtype=torch.long, device=device)
    return thresholds[row, idx].contiguous(), threshold_sels[row, idx].contiguous()


def _gpu_gqa_ranked_exact_logits(
    *,
    queries: torch.Tensor,
    keys_all: torch.Tensor,
    ranked_tokens: torch.Tensor,
    group_size: int,
    scale: float,
    max_rank: int,
    rank_chunk: int = 32,
) -> torch.Tensor:
    """Exact QK logits for ranked dynamic candidates only.

    The selector scores are PQ-domain approximations. Confidence gates need the
    exact logits for already selected candidates, which is deployable because
    those K vectors are on the exact-attention path. This helper keeps the
    operation on GPU and chunks over rank to bound peak memory.
    """

    if ranked_tokens.dim() not in {2, 3}:
        raise ValueError(f"ranked_tokens must be [heads, rank] or [queries, heads, rank], got {tuple(ranked_tokens.shape)}")
    rank = min(max(0, int(max_rank)), int(ranked_tokens.shape[-1]))
    if rank <= 0:
        return torch.empty((*ranked_tokens.shape[:-1], 0), dtype=torch.float32, device=ranked_tokens.device)
    out = torch.empty((*ranked_tokens.shape[:-1], rank), dtype=torch.float32, device=ranked_tokens.device)
    dim = int(queries.shape[-1])
    kv_heads = int(keys_all.shape[0])
    group = max(1, int(group_size))
    rank_chunk = max(1, int(rank_chunk))
    keys_token_count = int(keys_all.shape[1])

    if ranked_tokens.dim() == 3:
        positions = int(ranked_tokens.shape[0])
        heads = int(ranked_tokens.shape[1])
        for kv_head in range(kv_heads):
            head_start = int(kv_head * group)
            head_end = min(heads, head_start + group)
            if head_start >= head_end:
                continue
            q = queries[:, head_start:head_end, :].float()
            for rank_start in range(0, rank, rank_chunk):
                rank_end = min(rank, rank_start + rank_chunk)
                toks = ranked_tokens[:, head_start:head_end, rank_start:rank_end].to(torch.long)
                toks = toks.clamp(min=0, max=max(0, keys_token_count - 1))
                gathered = keys_all[int(kv_head)].index_select(0, toks.reshape(-1)).reshape(
                    positions,
                    head_end - head_start,
                    rank_end - rank_start,
                    dim,
                )
                logits = torch.sum(q.unsqueeze(2) * gathered.float(), dim=-1) * float(scale)
                out[:, head_start:head_end, rank_start:rank_end] = logits
    else:
        heads = int(ranked_tokens.shape[0])
        for kv_head in range(kv_heads):
            head_start = int(kv_head * group)
            head_end = min(heads, head_start + group)
            if head_start >= head_end:
                continue
            q = queries[head_start:head_end, :].float()
            for rank_start in range(0, rank, rank_chunk):
                rank_end = min(rank, rank_start + rank_chunk)
                toks = ranked_tokens[head_start:head_end, rank_start:rank_end].to(torch.long)
                toks = toks.clamp(min=0, max=max(0, keys_token_count - 1))
                gathered = keys_all[int(kv_head)].index_select(0, toks.reshape(-1)).reshape(
                    head_end - head_start,
                    rank_end - rank_start,
                    dim,
                )
                logits = torch.sum(q.unsqueeze(1) * gathered.float(), dim=-1) * float(scale)
                out[head_start:head_end, rank_start:rank_end] = logits
    return out


def _gpu_gqa_dense_decode_ranked_logits_and_base_lse(
    *,
    queries: torch.Tensor,
    keys_all: torch.Tensor,
    keys_all_t_float: torch.Tensor | None,
    ranked_tokens: torch.Tensor,
    group_size: int,
    scale: float,
    max_rank: int,
    query_context_len: int,
    static_prefix: int,
    static_suffix: int,
    page_size: int,
    need_base_lse: bool,
) -> tuple[torch.Tensor, torch.Tensor | None, int, int]:
    """GPU simulator exact logits via dense QK then ranked gather.

    This deliberately favors GPU throughput over physical access fidelity. The
    output is still the exact ranked logits used by the frontier algorithm, but
    the GPU host may read more K than the custom-hardware logical model.
    """

    if ranked_tokens.dim() != 2:
        raise ValueError(f"dense decode exact logits expects [heads, rank], got {tuple(ranked_tokens.shape)}")
    rank = min(max(0, int(max_rank)), int(ranked_tokens.shape[-1]))
    heads = int(ranked_tokens.shape[0])
    device = ranked_tokens.device
    if rank <= 0:
        empty = torch.empty((heads, 0), dtype=torch.float32, device=device)
        base = torch.full((heads,), float("-inf"), dtype=torch.float32, device=device) if need_base_lse else None
        return empty, base, 0, 0

    key_count = min(max(0, int(query_context_len)), int(keys_all.shape[1]))
    if key_count <= 0:
        out = torch.full((heads, rank), float("-inf"), dtype=torch.float32, device=device)
        base = torch.full((heads,), float("-inf"), dtype=torch.float32, device=device) if need_base_lse else None
        return out, base, 0, 0

    out = torch.empty((heads, rank), dtype=torch.float32, device=device)
    base_out = (
        torch.full((heads,), float("-inf"), dtype=torch.float32, device=device) if bool(need_base_lse) else None
    )
    base_toks: torch.Tensor | None = None
    base_mask: torch.Tensor | None = None
    total_base = 0
    if bool(need_base_lse):
        token_rows, mask_rows, total_base = _prefill_base_token_rows(
            query_len=1,
            query_start=int(query_context_len) - 1,
            static_prefix=int(static_prefix),
            static_suffix=int(static_suffix),
            page_size=int(page_size),
            device=device,
        )
        if token_rows.numel() > 0:
            base_toks = token_rows[0].clamp(min=0, max=max(0, key_count - 1)).to(torch.long)
            base_mask = mask_rows[0]

    dim = int(queries.shape[-1])
    kv_heads = int(keys_all.shape[0])
    group = max(1, int(group_size))
    covered_heads = min(heads, kv_heads * group)
    aligned_heads = (covered_heads // group) * group
    if aligned_heads > 0:
        aligned_kv_heads = aligned_heads // group
        q_grouped = queries[:aligned_heads, :].reshape(aligned_kv_heads, group, dim).float()
        if keys_all_t_float is not None:
            key_t_grouped = keys_all_t_float[:aligned_kv_heads, :, :key_count]
        else:
            key_t_grouped = keys_all[:aligned_kv_heads, :key_count, :].float().transpose(1, 2).contiguous()
        dense_grouped = torch.bmm(
            q_grouped,
            key_t_grouped,
        ) * float(scale)
        dense_logits = dense_grouped.reshape(aligned_heads, key_count)
        toks = ranked_tokens[:aligned_heads, :rank].to(torch.long).clamp(min=0, max=max(0, key_count - 1))
        out[:aligned_heads, :rank] = torch.gather(dense_logits, 1, toks)
        if base_out is not None:
            if base_toks is None or base_toks.numel() == 0:
                base_out[:aligned_heads] = float("-inf")
            else:
                base_logits = dense_logits.index_select(1, base_toks)
                if base_mask is not None:
                    base_logits = base_logits.masked_fill(~base_mask.reshape(1, -1), float("-inf"))
                base_out[:aligned_heads] = torch.logsumexp(base_logits, dim=-1)

    for kv_head in range(aligned_heads // group, kv_heads):
        head_start = int(kv_head * group)
        head_end = min(heads, head_start + group)
        if head_start >= head_end:
            continue
        q = queries[head_start:head_end, :].float()
        # Dense GEMM is usually much faster on GPU than irregular ranked-K gathers.
        if keys_all_t_float is not None:
            key_t = keys_all_t_float[int(kv_head), :, :key_count]
        else:
            key_t = keys_all[int(kv_head), :key_count, :].float().t()
        dense_logits = torch.matmul(q, key_t) * float(scale)
        toks = ranked_tokens[head_start:head_end, :rank].to(torch.long).clamp(min=0, max=max(0, key_count - 1))
        out[head_start:head_end, :rank] = torch.gather(dense_logits, 1, toks)
        if base_out is not None:
            if base_toks is None or base_toks.numel() == 0:
                base_out[head_start:head_end] = float("-inf")
            else:
                base_logits = dense_logits.index_select(1, base_toks)
                if base_mask is not None:
                    base_logits = base_logits.masked_fill(~base_mask.reshape(1, -1), float("-inf"))
                base_out[head_start:head_end] = torch.logsumexp(base_logits, dim=-1)
    return out.contiguous(), base_out.contiguous() if base_out is not None else None, int(total_base), int(key_count)


def _prefill_base_token_rows(
    *,
    query_len: int,
    query_start: int,
    static_prefix: int,
    static_suffix: int,
    page_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    q_lens = [int(query_start) + i + 1 for i in range(int(query_len))]
    rows: list[list[int]] = []
    max_base = 0
    total_base = 0
    page_size_i = max(1, int(page_size))
    for q_len in q_lens:
        prefix_end = min(max(0, int(static_prefix)), int(q_len))
        indexed_end = max(prefix_end, int(q_len) - max(0, int(static_suffix)))
        sealed_end = prefix_end + ((max(0, indexed_end - prefix_end) // page_size_i) * page_size_i)
        suffix_start = max(sealed_end, prefix_end)
        toks = list(range(prefix_end))
        if suffix_start < q_len:
            toks.extend(range(suffix_start, q_len))
        rows.append(toks)
        max_base = max(max_base, len(toks))
        total_base += len(toks)
    if max_base <= 0:
        return (
            torch.empty((int(query_len), 0), dtype=torch.long, device=device),
            torch.empty((int(query_len), 0), dtype=torch.bool, device=device),
            0,
        )
    token_rows_cpu = torch.zeros((int(query_len), max_base), dtype=torch.long)
    mask_rows_cpu = torch.zeros((int(query_len), max_base), dtype=torch.bool)
    for row_idx, toks in enumerate(rows):
        if toks:
            token_rows_cpu[row_idx, : len(toks)] = torch.as_tensor(toks, dtype=torch.long)
            mask_rows_cpu[row_idx, : len(toks)] = True
    return token_rows_cpu.to(device=device), mask_rows_cpu.to(device=device), int(total_base)


def _gpu_gqa_base_logsumexp_prefill(
    *,
    queries: torch.Tensor,
    keys_all: torch.Tensor,
    group_size: int,
    query_start: int,
    static_prefix: int,
    static_suffix: int,
    page_size: int,
    scale: float,
    position_chunk: int = 64,
) -> tuple[torch.Tensor, int]:
    token_rows, mask_rows, total_base = _prefill_base_token_rows(
        query_len=int(queries.shape[0]),
        query_start=int(query_start),
        static_prefix=int(static_prefix),
        static_suffix=int(static_suffix),
        page_size=int(page_size),
        device=queries.device,
    )
    positions = int(queries.shape[0])
    heads = int(queries.shape[1])
    if token_rows.numel() == 0:
        return torch.full((positions, heads), float("-inf"), dtype=torch.float32, device=queries.device), int(total_base)
    out = torch.full((positions, heads), float("-inf"), dtype=torch.float32, device=queries.device)
    dim = int(queries.shape[-1])
    kv_heads = int(keys_all.shape[0])
    group = max(1, int(group_size))
    position_chunk = max(1, int(position_chunk))
    keys_token_count = int(keys_all.shape[1])
    for pos_start in range(0, positions, position_chunk):
        pos_end = min(positions, pos_start + position_chunk)
        toks_chunk = token_rows[pos_start:pos_end].clamp(min=0, max=max(0, keys_token_count - 1))
        mask_chunk = mask_rows[pos_start:pos_end]
        base_count = int(toks_chunk.shape[1])
        for kv_head in range(kv_heads):
            head_start = int(kv_head * group)
            head_end = min(heads, head_start + group)
            if head_start >= head_end:
                continue
            q = queries[pos_start:pos_end, head_start:head_end, :].float()
            gathered = keys_all[int(kv_head)].index_select(0, toks_chunk.reshape(-1)).reshape(
                pos_end - pos_start,
                base_count,
                dim,
            )
            logits = torch.einsum("pgd,pbd->pgb", q, gathered.float()) * float(scale)
            logits = logits.masked_fill(~mask_chunk.reshape(pos_end - pos_start, 1, base_count), float("-inf"))
            out[pos_start:pos_end, head_start:head_end] = torch.logsumexp(logits, dim=-1)
    return out, int(total_base)


def _gpu_gqa_base_logsumexp_decode(
    *,
    queries: torch.Tensor,
    keys_all: torch.Tensor,
    group_size: int,
    query_context_len: int,
    static_prefix: int,
    static_suffix: int,
    page_size: int,
    scale: float,
) -> tuple[torch.Tensor, int]:
    token_rows, mask_rows, total_base = _prefill_base_token_rows(
        query_len=1,
        query_start=int(query_context_len) - 1,
        static_prefix=int(static_prefix),
        static_suffix=int(static_suffix),
        page_size=int(page_size),
        device=queries.device,
    )
    heads = int(queries.shape[0])
    if token_rows.numel() == 0:
        return torch.full((heads,), float("-inf"), dtype=torch.float32, device=queries.device), int(total_base)
    toks = token_rows[0].clamp(min=0, max=max(0, int(keys_all.shape[1]) - 1))
    mask = mask_rows[0]
    out = torch.full((heads,), float("-inf"), dtype=torch.float32, device=queries.device)
    dim = int(queries.shape[-1])
    group = max(1, int(group_size))
    for kv_head in range(int(keys_all.shape[0])):
        head_start = int(kv_head * group)
        head_end = min(heads, head_start + group)
        if head_start >= head_end:
            continue
        gathered = keys_all[int(kv_head)].index_select(0, toks).float()
        logits = torch.matmul(queries[head_start:head_end, :].float(), gathered.t().contiguous()) * float(scale)
        logits = logits.masked_fill(~mask.reshape(1, -1), float("-inf"))
        out[head_start:head_end] = torch.logsumexp(logits, dim=-1)
    return out, int(total_base)


def _gpu_proxy_confidence_metrics(
    *,
    ranked_scores: torch.Tensor,
    exact_ranked_logits: torch.Tensor,
    keep_count: int | torch.Tensor,
    max_budget: int,
    query_dim: int,
    base_logsumexp: torch.Tensor | None,
    calibrate: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rank_count = int(ranked_scores.shape[-1])
    max_budget_i = max(0, min(rank_count, int(max_budget)))
    leading = ranked_scores.shape[:-1]
    if torch.is_tensor(keep_count):
        keep_i = keep_count.to(device=ranked_scores.device, dtype=torch.long)
        if tuple(keep_i.shape) != tuple(leading):
            try:
                keep_i = torch.broadcast_to(keep_i, leading)
            except RuntimeError as exc:
                raise ValueError(
                    f"keep_count shape {tuple(keep_i.shape)} cannot broadcast to {tuple(leading)}"
                ) from exc
        keep_i = torch.clamp(keep_i, min=0, max=max_budget_i)
    else:
        keep_i_scalar = max(0, min(max_budget_i, int(keep_count)))
        keep_i = torch.full(leading, keep_i_scalar, dtype=torch.long, device=ranked_scores.device)
    if max_budget_i <= 0:
        zeros = torch.zeros(leading, dtype=torch.float32, device=ranked_scores.device)
        infs = torch.full(leading, float("inf"), dtype=torch.float32, device=ranked_scores.device)
        return zeros, zeros, zeros, infs
    scores = ranked_scores[..., :max_budget_i].float()
    exact = exact_ranked_logits[..., :max_budget_i].float()
    finite = torch.isfinite(scores)
    ranks = torch.arange(max_budget_i, dtype=torch.long, device=ranked_scores.device).reshape(
        *((1,) * (scores.dim() - 1)),
        max_budget_i,
    )
    keep_i_expanded = keep_i.reshape(*leading, 1)
    selected_mask = finite & (ranks < keep_i_expanded)
    tail_mask = finite & (ranks >= keep_i_expanded)
    pq_logits = scores * (float(query_dim) ** -0.5)
    count = selected_mask.sum(dim=-1).to(torch.float32)
    selected_mask_f = selected_mask.to(torch.float32)
    safe_count = torch.clamp(count, min=1.0)

    if bool(calibrate):
        x_sum = torch.sum(torch.where(selected_mask, pq_logits, torch.zeros_like(pq_logits)), dim=-1)
        y_sum = torch.sum(torch.where(selected_mask, exact, torch.zeros_like(exact)), dim=-1)
        mean_x = x_sum / safe_count
        mean_y = y_sum / safe_count
        dx = torch.where(selected_mask, pq_logits - mean_x.unsqueeze(-1), torch.zeros_like(pq_logits))
        dy = torch.where(selected_mask, exact - mean_y.unsqueeze(-1), torch.zeros_like(exact))
        var_x = torch.sum(dx * dx, dim=-1) / safe_count
        var_y = torch.sum(dy * dy, dim=-1) / safe_count
        cov = torch.sum(dx * dy, dim=-1) / safe_count
        fit_scale = cov / torch.clamp(var_x, min=1.0e-20)
        fit_bias = mean_y - fit_scale * mean_x
        flat_case = (var_x <= 1.0e-20) & (count >= 2.0)
        fit_scale = torch.where(flat_case, torch.zeros_like(fit_scale), fit_scale)
        fit_bias = torch.where(flat_case, mean_y, fit_bias)
        bad_scale = ((fit_scale <= 0.0) | ~torch.isfinite(fit_scale)) & ~flat_case
        fit_scale = torch.where(bad_scale, torch.ones_like(fit_scale), fit_scale)
        fit_bias = torch.where(bad_scale, torch.zeros_like(fit_bias), fit_bias)
        fit_scale = torch.where(count >= 2.0, fit_scale, torch.ones_like(fit_scale))
        fit_bias = torch.where(count >= 2.0, fit_bias, torch.zeros_like(fit_bias))
        pred = fit_scale.unsqueeze(-1) * pq_logits + fit_bias.unsqueeze(-1)
        rmse = torch.sqrt(torch.sum(((pred - exact) ** 2) * selected_mask_f, dim=-1) / safe_count)
        relrmse = rmse / torch.clamp(torch.sqrt(var_y), min=1.0e-6)
        relrmse = torch.where(count >= 2.0, relrmse, torch.full_like(relrmse, float("inf")))
        corr = cov / torch.sqrt(torch.clamp(var_x * var_y, min=1.0e-20))
        corr = torch.where((count >= 2.0) & torch.isfinite(corr), corr, torch.zeros_like(corr))
    else:
        fit_scale = torch.ones(ranked_scores.shape[:-1], dtype=torch.float32, device=ranked_scores.device)
        fit_bias = torch.zeros_like(fit_scale)
        corr = torch.zeros_like(fit_scale)
        relrmse = torch.full_like(fit_scale, float("inf"))

    selected_logits = torch.where(selected_mask, exact, torch.full_like(exact, float("-inf")))
    selected_lse = torch.logsumexp(selected_logits, dim=-1)
    if base_logsumexp is not None:
        selected_lse = torch.logaddexp(selected_lse, base_logsumexp.float())
    tail_logits = fit_scale.unsqueeze(-1) * pq_logits + fit_bias.unsqueeze(-1)
    tail_logits = torch.where(tail_mask, tail_logits, torch.full_like(tail_logits, float("-inf")))
    tail_lse = torch.logsumexp(tail_logits, dim=-1)
    total_lse = torch.logaddexp(selected_lse, tail_lse)
    selected_mass = torch.where(
        torch.isfinite(total_lse),
        torch.exp(selected_lse - total_lse),
        torch.zeros_like(total_lse),
    )
    tail_mass = torch.where(
        torch.isfinite(total_lse),
        torch.exp(tail_lse - total_lse),
        torch.zeros_like(total_lse),
    )
    return selected_mass, tail_mass, corr, relrmse

