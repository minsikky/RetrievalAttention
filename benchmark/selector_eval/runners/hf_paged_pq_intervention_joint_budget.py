#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from benchmark.selector_eval.runners.hf_paged_pq_intervention_common import (
    _budgets_from_fraction_schedule,
    _env_truthy,
    _parse_budget_schedule,
)
from benchmark.selector_eval.runners.hf_paged_pq_intervention_common import MB


@dataclass(frozen=True)
class JointKVBudgetSchedule:
    k_budgets: list[int]
    v_budgets: list[int]
    v_budgets_t: torch.Tensor


@dataclass(frozen=True)
class JointValueCost:
    actual_value_subbits: int
    actual_value_subvecs: int
    code_bytes: int
    metadata_mb: float
    v_pq_codebook_mb: float
    v_mb_by_idx: list[float] | None
    max_exact_v_count: int


def joint_kv_budget_schedule_for(
    *,
    args: Any,
    device: torch.device,
    context_len: int,
) -> JointKVBudgetSchedule:
    k_budget_text = str(getattr(args, "joint_kv_k_budgets", ""))
    v_budget_text = str(getattr(args, "joint_kv_v_budgets", ""))
    k_budget_frac_text = str(getattr(args, "joint_kv_k_budget_fracs", "")).strip()
    v_budget_frac_text = str(getattr(args, "joint_kv_v_budget_fracs", "")).strip()
    if bool(k_budget_frac_text) != bool(v_budget_frac_text):
        raise ValueError("joint_kv_k_budget_fracs and joint_kv_v_budget_fracs must be provided together")
    budget_cache_key = (
        k_budget_text,
        v_budget_text,
        k_budget_frac_text,
        v_budget_frac_text,
        int(context_len),
        str(device.type),
        int(device.index) if device.index is not None else -1,
    )
    budget_cache = getattr(args, "_pagedpq_joint_budget_cache", None)
    if not isinstance(budget_cache, dict):
        budget_cache = {}
        setattr(args, "_pagedpq_joint_budget_cache", budget_cache)
    cached_budgets = budget_cache.get(budget_cache_key)
    if cached_budgets is None:
        if k_budget_frac_text:
            parsed_k_budgets = _budgets_from_fraction_schedule(
                k_budget_frac_text,
                name="joint_kv_k_budget_fracs",
                context_len=int(context_len),
            )
            parsed_v_budgets = _budgets_from_fraction_schedule(
                v_budget_frac_text,
                name="joint_kv_v_budget_fracs",
                context_len=int(context_len),
            )
        else:
            parsed_k_budgets = _parse_budget_schedule(k_budget_text, name="joint_kv_k_budgets")
            parsed_v_budgets = _parse_budget_schedule(v_budget_text, name="joint_kv_v_budgets")
        cached_budgets = (
            tuple(int(v) for v in parsed_k_budgets),
            tuple(int(v) for v in parsed_v_budgets),
            torch.as_tensor(parsed_v_budgets, dtype=torch.long, device=device),
        )
        budget_cache[budget_cache_key] = cached_budgets
    k_budgets = list(cached_budgets[0])
    v_budgets = list(cached_budgets[1])
    v_budgets_t = cached_budgets[2]

    if _env_truthy("SELECTOR_PQ_JOINT_COLLAPSE_DUP_V_ROWS", "0"):
        collapsed_v_budgets: list[int] = []
        seen_v_counts: set[int] = set()
        for v_budget in v_budgets:
            exact_count_i = max(0, min(int(v_budget), int(context_len)))
            if int(exact_count_i) in seen_v_counts:
                continue
            seen_v_counts.add(int(exact_count_i))
            collapsed_v_budgets.append(int(v_budget))
        if collapsed_v_budgets:
            v_budgets = collapsed_v_budgets
            v_budgets_t = torch.as_tensor(
                v_budgets,
                dtype=torch.long,
                device=device,
            )
    return JointKVBudgetSchedule(
        k_budgets=k_budgets,
        v_budgets=v_budgets,
        v_budgets_t=v_budgets_t,
    )


def joint_value_cost_for(
    *,
    args: Any,
    index: Any,
    context_len: int,
    head_dim: int,
    value_bytes: int,
    joint_v_budgets: list[int],
    needs_budget_mb_vectors: bool,
    actual_value_subbits_for_cost: int,
) -> JointValueCost:
    actual_value_subbits = int(actual_value_subbits_for_cost)
    actual_value_subvecs = int(args.value_subvecs) if int(args.value_subvecs) > 0 else int(args.subvecs)
    code_bytes = 1 if int(actual_value_subbits) <= 8 else 2
    metadata_mb = (
        float(context_len * actual_value_subvecs * code_bytes)
        + float(
            len(index.pages)
            * actual_value_subvecs
            * (1 << int(actual_value_subbits))
            * int(getattr(args, "value_code_stat_bytes", getattr(args, "selected_value_residual_norm_bytes", 2)))
        )
    ) / MB
    v_pq_codebook_mb = float(
        len(index.pages)
        * actual_value_subvecs
        * (1 << int(actual_value_subbits))
        * (int(head_dim) // max(1, actual_value_subvecs))
        * value_bytes
    ) / MB

    v_mb_by_idx: list[float] | None = None
    if needs_budget_mb_vectors:
        v_mb_by_idx = []
        for v_budget in joint_v_budgets:
            exact_count = max(0, min(int(v_budget), context_len))
            exact_v_mb = float(exact_count * int(head_dim) * value_bytes) / MB
            compressed_v_codes_mb = (
                float(max(0, context_len - exact_count) * actual_value_subvecs * code_bytes) / MB
            )
            v_mb_by_idx.append(exact_v_mb + v_pq_codebook_mb + compressed_v_codes_mb + metadata_mb)

    max_exact_v_count = max(
        [max(0, min(int(v_budget), context_len)) for v_budget in joint_v_budgets],
        default=0,
    )
    return JointValueCost(
        actual_value_subbits=actual_value_subbits,
        actual_value_subvecs=actual_value_subvecs,
        code_bytes=code_bytes,
        metadata_mb=float(metadata_mb),
        v_pq_codebook_mb=float(v_pq_codebook_mb),
        v_mb_by_idx=v_mb_by_idx,
        max_exact_v_count=int(max_exact_v_count),
    )
