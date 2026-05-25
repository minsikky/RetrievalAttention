#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.selector_eval.data.trace import attention_probs, load_trace, static_tokens, unique_tokens
from benchmark.selector_eval.gpu.run_gpu_paged_pq_eval import (
    MB,
    build_page_pq_gpu,
    load_selector_paged_pq_ext,
    parse_csv_ints,
    rank_paged_pq,
    rank_paged_pq_batched_with_scores,
    selector_bytes_fullscan,
)
from benchmark.selector_eval.metrics.attention import _output_error_metrics
from benchmark.selector_eval.runners.hf_paged_pq_intervention_common import (
    _joint_kv_policy_id,
    _simulate_joint_kv_policy,
    _simulate_joint_kv_policy_torch,
)
from benchmark.selector_eval.runners.run_joint_kv_budget_policy_eval import (
    load_safetensor_weight,
    load_weight_index,
)
from benchmark.selector_eval.runners.run_layer_quality_eval import _selected_for_budget, _vpq_values_for_tokens
from benchmark.selector_eval.runners.run_value_exact_strategy_eval import (
    mixed_scores,
    output_from_exact_mask,
    project_head_subset,
    top_mask,
    value_vpq_code_stat_risk,
)


def _env_truthy(name: str, default: str = "0") -> bool:
    return str(os.environ.get(name, default)).strip().lower() in {"1", "true", "yes", "on"}


def _parse_float_list(text: str) -> list[float]:
    return [float(part.strip()) for part in str(text).split(",") if part.strip()]


def _collapse_duplicate_v_budgets(v_budgets: list[int], context_len: int) -> list[int]:
    collapsed: list[int] = []
    seen_counts: set[int] = set()
    for v_budget in v_budgets:
        exact_count = max(0, min(int(v_budget), int(context_len)))
        if int(exact_count) in seen_counts:
            continue
        seen_counts.add(int(exact_count))
        collapsed.append(int(v_budget))
    return collapsed or list(v_budgets)


def _numeric_vector(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float32).reshape(-1).copy()


def _safe_output_error_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    return _output_error_metrics(_numeric_vector(reference), _numeric_vector(candidate))


def _evaluate_policy(
    *,
    query_np: np.ndarray,
    keys_np: np.ndarray,
    values_np: np.ndarray,
    position: int,
    input_len: int,
    index,
    selector_backend: str,
    k_budgets: list[int],
    v_budgets: list[int],
    policy: str,
    threshold: float,
    args: argparse.Namespace,
) -> dict[str, object]:
    context_len = int(position) + 1
    _scores_check, _probs_check = attention_probs(keys_np, query_np)
    dynamic_count = int(sum(int(page.size) for page in index.pages))
    rank_budget = dynamic_count
    ranked_t, ranked_scores_t, _seconds, selector_mb, _nprobe = rank_paged_pq(
        torch.as_tensor(query_np, dtype=torch.float32, device=args.device),
        index,
        mode="fullscan",
        selector_backend=str(selector_backend),
        nprobes=[],
        budget=int(rank_budget),
        key_bytes=int(args.key_bytes),
        subbits=int(args.subbits),
    )
    ranked_cpu = ranked_t.detach().cpu().numpy().astype(np.int64, copy=False)
    ranked_scores_cpu = ranked_scores_t.detach().cpu().numpy().astype(np.float32, copy=False)
    expected_selector_mb = selector_bytes_fullscan(index, key_bytes=int(args.key_bytes), subbits=int(args.subbits)) / MB
    selector_mb = float(expected_selector_mb if selector_mb == 0.0 and dynamic_count > 0 else selector_mb)

    pending = list(range(max(0, int(index.pending_start)), max(0, min(int(index.indexed_end), context_len))))
    base = unique_tokens(
        static_tokens(int(position), int(args.static_prefix), int(args.static_suffix)) + pending,
        context_len=context_len,
    )
    all_tokens = np.arange(context_len, dtype=np.int64)
    vhat_all, _compressed_v_mb, _fallback_v_mb = _vpq_values_for_tokens(
        index=index,
        values_np=values_np,
        tokens=all_tokens,
        subbits=int(args.subbits),
        value_subvecs=int(args.value_subvecs),
        value_subbits=int(args.value_subbits),
        value_bytes=int(args.value_bytes),
    )
    residual = values_np.astype(np.float32, copy=False) - vhat_all.astype(np.float32, copy=False)
    code_error = value_vpq_code_stat_risk(
        index=index,
        values_np=values_np,
        residual=residual,
        subbits=int(args.subbits),
        value_subvecs=int(args.value_subvecs),
        value_subbits=int(args.value_subbits),
        sensitivity=None,
    )
    exact_scores_np, _true_probs = attention_probs(keys_np, query_np)

    outputs: dict[tuple[int, int], np.ndarray] = {}
    selected_by_idx: list[np.ndarray] = []
    k_mb_by_idx: list[float] = []
    for ki, k_budget in enumerate(k_budgets):
        selected_cpu = _selected_for_budget(
            base=base,
            ranked_cpu=ranked_cpu,
            budget=int(k_budget),
            context_len=context_len,
        )
        selected_by_idx.append(selected_cpu)
        score_vec, _missing, _scale, _bias = mixed_scores(
            context_len=context_len,
            selected_cpu=selected_cpu,
            ranked_cpu=ranked_cpu,
            ranked_scores_cpu=ranked_scores_cpu,
            exact_scores_np=exact_scores_np,
            query_dim=int(args.head_dim),
            calibrate=True,
        )
        probs = np.exp(score_vec - float(np.max(score_vec)))
        probs /= max(float(probs.sum()), 1e-20)
        exact_key_mb = float(selected_cpu.size * int(args.head_dim) * int(args.key_bytes)) / MB
        k_mb_by_idx.append(float(selector_mb) + exact_key_mb)
        risk = (probs * probs) * code_error
        for vi, v_budget in enumerate(v_budgets):
            exact_count = max(0, min(int(v_budget), context_len))
            exact_mask = top_mask(risk, exact_count)
            outputs[(ki, vi)] = output_from_exact_mask(
                probs=probs,
                vhat_all=vhat_all,
                residual=residual,
                exact_mask=exact_mask,
            )

    actual_value_subbits = int(args.value_subbits) if int(args.value_subbits) > 0 else int(args.subbits)
    actual_value_subvecs = int(args.value_subvecs) if int(args.value_subvecs) > 0 else int(args.subvecs)
    code_bytes = 1 if actual_value_subbits <= 8 else 2
    metadata_mb = (
        float(context_len * actual_value_subvecs * code_bytes)
        + float(len(index.pages) * actual_value_subvecs * (1 << actual_value_subbits) * int(args.value_code_stat_bytes))
    ) / MB
    v_pq_codebook_mb = float(
        len(index.pages)
        * actual_value_subvecs
        * (1 << actual_value_subbits)
        * (int(args.head_dim) // max(1, actual_value_subvecs))
        * int(args.value_bytes)
    ) / MB
    v_mb_by_idx: list[float] = []
    for v_budget in v_budgets:
        exact_count = max(0, min(int(v_budget), context_len))
        exact_v_mb = float(exact_count * int(args.head_dim) * int(args.value_bytes)) / MB
        compressed_v_codes_mb = float(max(0, context_len - exact_count) * actual_value_subvecs * code_bytes) / MB
        v_mb_by_idx.append(exact_v_mb + v_pq_codebook_mb + compressed_v_codes_mb + metadata_mb)

    ki, vi, steps, final_k_delta, final_v_delta = _simulate_joint_kv_policy(
        outputs=outputs,
        k_budgets=k_budgets,
        v_budgets=v_budgets,
        policy=str(policy),
        threshold=float(threshold),
        k_mb_by_idx=k_mb_by_idx,
        v_mb_by_idx=v_mb_by_idx,
    )
    out = outputs[(int(ki), int(vi))]
    selected = selected_by_idx[int(ki)]
    return {
        "k_idx": int(ki),
        "v_idx": int(vi),
        "k_budget": int(k_budgets[int(ki)]),
        "v_budget": int(v_budgets[int(vi)]),
        "selected_count": int(selected.size),
        "step_MB_per_head": float(k_mb_by_idx[int(ki)] + v_mb_by_idx[int(vi)]),
        "selector_MB_per_head": float(selector_mb),
        "policy_steps": int(steps),
        "final_k_delta": float(final_k_delta),
        "final_v_delta": float(final_v_delta),
        "output": out.astype(np.float32, copy=False),
        "ranked_tokens_prefix": ranked_cpu[: min(32, ranked_cpu.size)].astype(np.int64, copy=True),
    }


def _evaluate_policy_torch_gpu(
    *,
    query_np: np.ndarray,
    keys_np: np.ndarray,
    values_np: np.ndarray,
    position: int,
    input_len: int,
    index,
    selector_backend: str,
    k_budgets: list[int],
    v_budgets: list[int],
    policy: str,
    threshold: float,
    args: argparse.Namespace,
    use_native_vprefix: bool = False,
    use_native_risk_prefix: bool = False,
    use_native_risk_prefix_topk: bool = False,
    use_native_score_grid: bool = False,
    use_native_pq_scale_in_kernel: bool = False,
    use_tokenfit_score_grid: bool = False,
    use_score_grid_workspace: bool = False,
    use_native_policy: bool = False,
    use_interval_risk_policy: bool = False,
    use_score_direct_interval_policy: bool = False,
) -> dict[str, object]:
    """Mirror the benchmark-facing Torch/GPU joint K/V policy on a saved trace.

    The existing parity path checks CPU policy logic while swapping only the
    selector backend. This path additionally exercises the GPU/Torch mixed-logit
    and residual-risk exact-V grid used by the HF benchmark wrapper.
    """

    device = torch.device(args.device)
    context_len = int(position) + 1
    dynamic_count = int(sum(int(page.size) for page in index.pages))
    query_t = torch.as_tensor(query_np, dtype=torch.float32, device=device).reshape(1, -1)
    _ranked_t, _ranked_scores_t, dense_scores_t, _seconds, selector_mb, _nprobe = rank_paged_pq_batched_with_scores(
        query_t,
        index,
        mode="fullscan",
        selector_backend=str(selector_backend),
        nprobes=[],
        budget=max(1, int(dynamic_count)),
        key_bytes=int(args.key_bytes),
        subbits=int(args.subbits),
    )
    expected_selector_mb = selector_bytes_fullscan(index, key_bytes=int(args.key_bytes), subbits=int(args.subbits)) / MB
    selector_mb = float(expected_selector_mb if selector_mb == 0.0 and dynamic_count > 0 else selector_mb)

    page_ranges = [
        (
            int(page.start),
            min(int(page.start) + int(page.size), context_len),
        )
        for page in index.pages
        if int(page.start) < context_len and int(page.size) > 0
    ]
    token_parts = [
        torch.arange(
            start,
            end,
            dtype=torch.long,
            device=device,
        )
        for start, end in page_ranges
        if end > start
    ]
    indexed_tokens_t = torch.cat(token_parts) if token_parts else torch.empty((0,), dtype=torch.long, device=device)
    indexed_count = min(int(indexed_tokens_t.numel()), int(dense_scores_t.shape[1]))
    indexed_tokens_t = indexed_tokens_t[:indexed_count]
    dense_scores_t = dense_scores_t[:, :indexed_count].to(device=device, dtype=torch.float32)

    pending = list(range(max(0, int(index.pending_start)), max(0, min(int(index.indexed_end), context_len))))
    base = unique_tokens(
        static_tokens(int(position), int(args.static_prefix), int(args.static_suffix)) + pending,
        context_len=context_len,
    )
    coverage_intervals = [(max(0, start), min(context_len, end)) for start, end in page_ranges if end > start]
    base_tokens_sorted = sorted(int(token) for token in base if 0 <= int(token) < context_len)
    if base_tokens_sorted:
        run_start = base_tokens_sorted[0]
        prev = base_tokens_sorted[0]
        for token in base_tokens_sorted[1:]:
            if token == prev + 1:
                prev = token
                continue
            coverage_intervals.append((run_start, prev + 1))
            run_start = token
            prev = token
        coverage_intervals.append((run_start, prev + 1))
    coverage_end = 0
    layout_covers_context = context_len <= 0
    for start, end in sorted(coverage_intervals):
        if end <= coverage_end:
            continue
        if start > coverage_end:
            break
        coverage_end = max(coverage_end, end)
        if coverage_end >= context_len:
            layout_covers_context = True
            break
    base_t = torch.as_tensor(base, dtype=torch.long, device=device)
    if int(base_t.numel()) > 0:
        base_t = base_t[(base_t >= 0) & (base_t < context_len)]
    base_rank_mask_t = torch.zeros((context_len,), dtype=torch.bool, device=device)
    if int(base_t.numel()) > 0:
        base_rank_mask_t.index_fill_(0, base_t, True)
    nonbase_mask_t = ~base_rank_mask_t.index_select(0, indexed_tokens_t)
    ranked_nonbase_t = indexed_tokens_t[nonbase_mask_t]
    ranked_nonbase_scores_t = dense_scores_t[:, nonbase_mask_t]

    def native_rank_prefix_tokens(
        scores_t: torch.Tensor,
        tokens_t: torch.Tensor,
        max_take_i: int,
        take_counts_t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        native = load_selector_paged_pq_ext()
        scores_c = scores_t.to(dtype=torch.float32).contiguous()
        tokens_c = tokens_t.to(dtype=torch.long).contiguous()
        if (
            _env_truthy("SELECTOR_PQ_JOINT_NATIVE_BUDGET_PREFIX", "0")
            and take_counts_t is not None
            and int(take_counts_t.numel()) > 0
            and hasattr(native, "joint_budget_prefix_tokens")
        ):
            return native.joint_budget_prefix_tokens(
                scores_c,
                tokens_c,
                take_counts_t.to(device=device, dtype=torch.long).contiguous(),
                int(max_take_i),
            )
        if (
            _env_truthy("SELECTOR_PQ_JOINT_RANK_PREFIX_WORKSPACE", "0")
            and hasattr(native, "joint_rank_prefix_tokens_workspace")
        ):
            rows_i = int(scores_c.shape[0])
            count_i = int(scores_c.shape[1])
            take_i = max(0, min(int(max_take_i), count_i))
            temp_bytes = max(1, int(native.joint_rank_prefix_sort_temp_bytes(rows_i, count_i)))
            return native.joint_rank_prefix_tokens_workspace(
                scores_c,
                tokens_c,
                take_i,
                torch.empty((max(1, rows_i * count_i),), dtype=torch.float32, device=device),
                torch.empty((max(1, rows_i * count_i),), dtype=torch.float32, device=device),
                torch.empty((max(1, rows_i * count_i),), dtype=torch.int32, device=device),
                torch.empty((max(1, rows_i * count_i),), dtype=torch.int32, device=device),
                torch.empty((max(1, rows_i + 1),), dtype=torch.long, device=device),
                torch.empty((temp_bytes,), dtype=torch.uint8, device=device),
                torch.empty((max(1, rows_i * take_i),), dtype=torch.long, device=device),
            )
        return native.joint_rank_prefix_tokens(scores_c, tokens_c, int(max_take_i))

    all_tokens = np.arange(context_len, dtype=np.int64)
    vhat_all, _compressed_v_mb, _fallback_v_mb = _vpq_values_for_tokens(
        index=index,
        values_np=values_np,
        tokens=all_tokens,
        subbits=int(args.subbits),
        value_subvecs=int(args.value_subvecs),
        value_subbits=int(args.value_subbits),
        value_bytes=int(args.value_bytes),
    )
    residual_np = values_np.astype(np.float32, copy=False) - vhat_all.astype(np.float32, copy=False)
    code_error_np = value_vpq_code_stat_risk(
        index=index,
        values_np=values_np,
        residual=residual_np,
        subbits=int(args.subbits),
        value_subvecs=int(args.value_subvecs),
        value_subbits=int(args.value_subbits),
        sensitivity=None,
    )
    exact_scores_np, _true_probs = attention_probs(keys_np, query_np)
    use_sparse_exact_score_grid = _env_truthy("SELECTOR_PQ_JOINT_SPARSE_EXACT_SCORE_GRID", "0")
    exact_scores_t: torch.Tensor | None = None

    def _dense_exact_scores_t() -> torch.Tensor:
        native = load_selector_paged_pq_ext()
        backend = str(os.environ.get("SELECTOR_PQ_JOINT_NATIVE_EXACT_LOGITS_BACKEND", "cublas_t")).strip().lower()
        key_t = torch.as_tensor(keys_np[:context_len], dtype=torch.float32, device=device).reshape(1, context_len, -1)
        scale = 1.0 / math.sqrt(float(key_t.shape[2]))
        if _env_truthy("SELECTOR_PQ_JOINT_NATIVE_EXACT_LOGITS", "0"):
            if backend == "cublas_t" and hasattr(native, "gqa_decode_full_exact_logits_t_cublas"):
                return native.gqa_decode_full_exact_logits_t_cublas(
                    query_t,
                    key_t.transpose(1, 2).contiguous(),
                    1,
                    context_len,
                    float(scale),
                )
            if backend == "grouped" and hasattr(native, "gqa_decode_full_exact_logits_grouped"):
                return native.gqa_decode_full_exact_logits_grouped(
                    query_t,
                    key_t,
                    1,
                    context_len,
                    float(scale),
                )
            if backend in {"custom", "native"} and hasattr(native, "gqa_decode_full_exact_logits"):
                return native.gqa_decode_full_exact_logits(
                    query_t,
                    key_t,
                    1,
                    context_len,
                    float(scale),
                )
            raise RuntimeError(
                "SELECTOR_PQ_JOINT_NATIVE_EXACT_LOGITS requires "
                "gqa_decode_full_exact_logits_t_cublas or gqa_decode_full_exact_logits"
            )
        return torch.as_tensor(exact_scores_np, dtype=torch.float32, device=device).reshape(1, -1)

    if not use_sparse_exact_score_grid:
        exact_scores_t = _dense_exact_scores_t()
    elif not bool(use_native_score_grid):
        raise RuntimeError("SELECTOR_PQ_JOINT_SPARSE_EXACT_SCORE_GRID requires --use_native_score_grid")
    elif not _env_truthy("SELECTOR_PQ_JOINT_SCORE_GRID_NO_EXACT_FILL", "0"):
        raise RuntimeError(
            "SELECTOR_PQ_JOINT_SPARSE_EXACT_SCORE_GRID requires SELECTOR_PQ_JOINT_SCORE_GRID_NO_EXACT_FILL=1"
        )
    elif not bool(use_tokenfit_score_grid):
        raise RuntimeError(
            "SELECTOR_PQ_JOINT_SPARSE_EXACT_SCORE_GRID requires --use_tokenfit_score_grid so affine fit only reads exact selected tokens"
        )

    vhat_all_t = torch.as_tensor(vhat_all, dtype=torch.float32, device=device)
    residual_t = torch.as_tensor(residual_np, dtype=torch.float32, device=device)
    code_error_t = torch.as_tensor(code_error_np, dtype=torch.float32, device=device)

    ranked_nonbase_count = int(ranked_nonbase_t.numel())
    active_k_budgets = k_budgets
    collapse_duplicate_k_rows = _env_truthy("SELECTOR_PQ_JOINT_COLLAPSE_DUP_K_ROWS", "0")
    if collapse_duplicate_k_rows:
        collapsed_k_budgets: list[int] = []
        seen_take_counts: set[int] = set()
        for k_budget in k_budgets:
            take_i = max(0, min(int(k_budget), ranked_nonbase_count))
            if int(take_i) in seen_take_counts:
                continue
            seen_take_counts.add(int(take_i))
            collapsed_k_budgets.append(int(k_budget))
        if collapsed_k_budgets:
            active_k_budgets = collapsed_k_budgets
    avoid_full_budget_rank = bool(
        _env_truthy("SELECTOR_PQ_JOINT_SKIP_FULL_BUDGET_SORT", "0")
        or (
            collapse_duplicate_k_rows
            and _env_truthy("SELECTOR_PQ_JOINT_EXACT_FULL_BUDGET_GRID", "1")
        )
    )
    if avoid_full_budget_rank:
        partial_rank_takes = [
            max(0, min(int(v), ranked_nonbase_count))
            for v in active_k_budgets
            if max(0, min(int(v), ranked_nonbase_count)) < ranked_nonbase_count
        ]
        max_rank_take = max(partial_rank_takes, default=0)
    else:
        partial_rank_takes = [
            max(0, min(int(v), ranked_nonbase_count))
            for v in active_k_budgets
        ]
        max_rank_take = max(0, min(max(int(v) for v in active_k_budgets), ranked_nonbase_count))
    partial_rank_takes_for_prefix = sorted({int(v) for v in partial_rank_takes if int(v) > 0})
    partial_rank_takes_t = (
        torch.as_tensor(partial_rank_takes_for_prefix, dtype=torch.long, device=device)
        if partial_rank_takes_for_prefix
        else None
    )
    use_unsorted_k_prefix = _env_truthy("SELECTOR_PQ_JOINT_UNSORTED_K_PREFIX", "0")
    ranked_prefix_tokens_t: torch.Tensor | None = None
    if max_rank_take > 0 and not use_unsorted_k_prefix:
        if _env_truthy("SELECTOR_PQ_JOINT_NATIVE_RANK_PREFIX", "0"):
            ranked_prefix_tokens_t = native_rank_prefix_tokens(
                ranked_nonbase_scores_t,
                ranked_nonbase_t,
                int(max_rank_take),
                partial_rank_takes_t,
            )
        else:
            max_order_t = torch.topk(
                ranked_nonbase_scores_t,
                k=int(max_rank_take),
                dim=1,
                largest=True,
                sorted=True,
            ).indices
            ranked_prefix_tokens_t = ranked_nonbase_t.index_select(0, max_order_t.reshape(-1)).reshape(1, int(max_rank_take))

    if exact_scores_t is None:
        full_exact_row_requested = any(
            max(0, min(int(k_budget), ranked_nonbase_count)) > int(max_rank_take)
            for k_budget in active_k_budgets
        )
        if full_exact_row_requested:
            exact_scores_t = _dense_exact_scores_t()
        else:
            if not bool(layout_covers_context):
                raise RuntimeError(
                    "SELECTOR_PQ_JOINT_SPARSE_EXACT_SCORE_GRID requires indexed tokens plus base tokens "
                    "to cover the full context"
                )
            native = load_selector_paged_pq_ext()
            if not (
                hasattr(native, "gqa_decode_token_exact_logits")
                and hasattr(native, "joint_sparse_exact_score_table")
            ):
                raise RuntimeError(
                    "SELECTOR_PQ_JOINT_SPARSE_EXACT_SCORE_GRID requires sparse exact-logit CUDA helpers"
                )
            key_t = torch.as_tensor(keys_np[:context_len], dtype=torch.float32, device=device).reshape(1, context_len, -1)
            scale = 1.0 / math.sqrt(float(key_t.shape[2]))
            base_rows_t = base_t.reshape(1, -1) if int(base_t.numel()) > 0 else torch.empty((1, 0), dtype=torch.long, device=device)
            ranked_for_sparse_t = (
                ranked_prefix_tokens_t
                if ranked_prefix_tokens_t is not None
                else torch.empty((1, 0), dtype=torch.long, device=device)
            )
            base_logits_t = native.gqa_decode_token_exact_logits(
                query_t,
                key_t,
                base_rows_t.to(dtype=torch.long).contiguous(),
                1,
                int(context_len),
                float(scale),
            )
            ranked_logits_t = native.gqa_decode_token_exact_logits(
                query_t,
                key_t,
                ranked_for_sparse_t.to(dtype=torch.long).contiguous(),
                1,
                int(context_len),
                float(scale),
            )
            exact_scores_t = native.joint_sparse_exact_score_table(
                base_t.to(dtype=torch.long).contiguous(),
                base_logits_t.contiguous(),
                ranked_for_sparse_t.to(dtype=torch.long).contiguous(),
                ranked_logits_t.contiguous(),
                int(context_len),
            )

    def selected_for_budget(k_budget: int) -> torch.Tensor:
        take = max(0, min(int(k_budget), int(ranked_nonbase_t.numel())))
        if avoid_full_budget_rank and take >= ranked_nonbase_count and ranked_nonbase_count > 0:
            add_t = ranked_nonbase_t.reshape(1, -1)
        elif take > 0 and ranked_prefix_tokens_t is not None and take <= int(ranked_prefix_tokens_t.shape[1]):
            add_t = ranked_prefix_tokens_t[:, :take]
        elif take > 0:
            if _env_truthy("SELECTOR_PQ_JOINT_NATIVE_RANK_PREFIX", "0"):
                take_counts_single_t = torch.as_tensor([int(take)], dtype=torch.long, device=device)
                add_t = native_rank_prefix_tokens(
                    ranked_nonbase_scores_t,
                    ranked_nonbase_t,
                    int(take),
                    take_counts_single_t,
                )
            else:
                order_t = torch.topk(
                    ranked_nonbase_scores_t,
                k=int(take),
                dim=1,
                largest=True,
                sorted=not use_unsorted_k_prefix,
            ).indices
            add_t = ranked_nonbase_t.index_select(0, order_t.reshape(-1)).reshape(1, take)
        else:
            add_t = torch.empty((1, 0), dtype=torch.long, device=device)
        if int(base_t.numel()) == 0:
            return add_t
        base_rows_t = base_t.reshape(1, -1)
        if int(add_t.numel()) == 0:
            return base_rows_t
        return torch.cat((base_rows_t, add_t), dim=1)

    sqrt_dim = float(math.sqrt(float(args.head_dim)))
    prob_dtype = torch.float32

    def mixed_scores_for_selected(selected_t: torch.Tensor) -> torch.Tensor:
        selected_t = selected_t.to(device=device, dtype=torch.long)
        selected_t = torch.clamp(selected_t, min=0, max=max(0, context_len - 1))
        score_vec = exact_scores_t.to(dtype=prob_dtype).clone()
        pq_logits = dense_scores_t.to(dtype=prob_dtype) / sqrt_dim
        if str(args.tail_score_calibration) == "affine_selected":
            selected_index_mask = torch.zeros((1, context_len), dtype=torch.bool, device=device)
            if int(selected_t.numel()) > 0:
                selected_index_mask.scatter_(1, selected_t, True)
            selected_index_mask = selected_index_mask.index_select(1, indexed_tokens_t)
            mask_f = selected_index_mask.to(dtype=prob_dtype)
            counts_t = torch.sum(mask_f, dim=1)
            safe_counts_t = torch.clamp_min(counts_t, 1.0)
            y_indexed_t = exact_scores_t.index_select(1, indexed_tokens_t).to(prob_dtype)
            x_mean_t = torch.sum(mask_f * pq_logits, dim=1) / safe_counts_t
            y_mean_t = torch.sum(mask_f * y_indexed_t, dim=1) / safe_counts_t
            x_centered_t = (pq_logits - x_mean_t.reshape(-1, 1)) * mask_f
            y_centered_t = (y_indexed_t - y_mean_t.reshape(-1, 1)) * mask_f
            x_var_t = torch.sum(x_centered_t * x_centered_t, dim=1) / safe_counts_t
            cov_t = torch.sum(x_centered_t * y_centered_t, dim=1) / safe_counts_t
            fitted_scale_t = cov_t / torch.clamp_min(x_var_t, 1e-20)
            fitted_bias_t = y_mean_t - fitted_scale_t * x_mean_t
            fit_valid_t = (
                (counts_t >= 2.0)
                & (x_var_t > 1e-20)
                & torch.isfinite(fitted_scale_t)
                & (fitted_scale_t > 0.0)
            )
            zero_var_t = (counts_t >= 2.0) & (x_var_t <= 1e-20)
            scale_t = torch.where(
                zero_var_t,
                torch.zeros_like(fitted_scale_t),
                torch.where(fit_valid_t, fitted_scale_t, torch.ones_like(fitted_scale_t)),
            )
            bias_t = torch.where(
                zero_var_t,
                y_mean_t,
                torch.where(fit_valid_t, fitted_bias_t, torch.zeros_like(fitted_bias_t)),
            )
            calibrated_scores_t = scale_t.reshape(-1, 1) * pq_logits + bias_t.reshape(-1, 1)
        else:
            calibrated_scores_t = pq_logits
        score_vec[:, indexed_tokens_t] = calibrated_scores_t
        if int(selected_t.numel()) > 0:
            exact_selected_scores_t = exact_scores_t.gather(1, selected_t).to(prob_dtype)
            score_vec.scatter_(1, selected_t, exact_selected_scores_t)
        return score_vec

    native_score_grid_t: torch.Tensor | None = None
    native_probs_grid_t: torch.Tensor | None = None
    native_base_output_grid_t: torch.Tensor | None = None
    if bool(use_native_score_grid) and not use_unsorted_k_prefix:
        native = load_selector_paged_pq_ext()
        pq_logit_scale = float(1.0 / sqrt_dim) if bool(use_native_pq_scale_in_kernel) else 1.0
        pq_logits_for_grid_t = (
            dense_scores_t.to(dtype=torch.float32)
            if bool(use_native_pq_scale_in_kernel)
            else dense_scores_t.to(dtype=torch.float32) / sqrt_dim
        )
        y_indexed_for_grid_t = exact_scores_t.index_select(1, indexed_tokens_t).to(dtype=torch.float32)
        ranked_prefix_for_grid_t = (
            ranked_prefix_tokens_t
            if ranked_prefix_tokens_t is not None
            else torch.empty((1, 0), dtype=torch.long, device=device)
        )
        k_take_counts_t = torch.as_tensor(
            [max(0, min(int(k_budget), int(ranked_nonbase_t.numel()))) for k_budget in active_k_budgets],
            dtype=torch.long,
            device=device,
        )
        use_score_grid_no_fill = _env_truthy("SELECTOR_PQ_JOINT_SCORE_GRID_NO_EXACT_FILL", "0")
        if use_score_grid_no_fill:
            if not bool(layout_covers_context):
                raise RuntimeError(
                    "SELECTOR_PQ_JOINT_SCORE_GRID_NO_EXACT_FILL requires indexed tokens plus base tokens "
                    "to cover the full context"
                )
        if _env_truthy("SELECTOR_PQ_JOINT_FUSED_SOFTMAX_BASE", "0"):
            if use_score_grid_no_fill:
                raise RuntimeError("SELECTOR_PQ_JOINT_FUSED_SOFTMAX_BASE does not support no-fill diagnostic mode")
            use_rankpos_score_grid = _env_truthy("SELECTOR_PQ_JOINT_RANKPOS_SCORE_GRID", "0")
            use_fused_tokenfit = _env_truthy("SELECTOR_PQ_JOINT_FUSED_TOKENFIT_SOFTMAX_BASE", "0")
            if use_fused_tokenfit:
                fused_score_fn_name = "joint_mixed_softmax_base_outputs_tokenfit_scaled"
            else:
                if bool(use_tokenfit_score_grid):
                    raise RuntimeError(
                        "tokenfit score grid requires SELECTOR_PQ_JOINT_FUSED_TOKENFIT_SOFTMAX_BASE "
                        "with fused softmax/base"
                    )
                if bool(use_native_pq_scale_in_kernel):
                    raise RuntimeError(
                        "native in-kernel PQ scaling requires SELECTOR_PQ_JOINT_FUSED_TOKENFIT_SOFTMAX_BASE "
                        "with fused softmax/base"
                    )
                fused_score_fn_name = (
                    "joint_mixed_softmax_base_outputs_rankpos"
                    if use_rankpos_score_grid
                    else "joint_mixed_softmax_base_outputs"
                )
            if not hasattr(native, fused_score_fn_name):
                raise RuntimeError(
                    f"SELECTOR_PQ_JOINT_FUSED_SOFTMAX_BASE requires updated CUDA extension: {fused_score_fn_name}"
                )
            fused_args = (
                exact_scores_t.to(dtype=torch.float32).contiguous(),
                pq_logits_for_grid_t.contiguous(),
                y_indexed_for_grid_t.contiguous(),
                indexed_tokens_t.to(dtype=torch.long).contiguous(),
                base_t.to(dtype=torch.long).contiguous(),
                ranked_prefix_for_grid_t.to(dtype=torch.long).contiguous(),
                k_take_counts_t,
                vhat_all_t.to(dtype=torch.float32).contiguous(),
                bool(str(args.tail_score_calibration) == "affine_selected"),
            )
            if use_fused_tokenfit:
                native_probs_grid_t, native_base_output_grid_t = getattr(native, fused_score_fn_name)(
                    *fused_args,
                    float(pq_logit_scale),
                )
            else:
                native_probs_grid_t, native_base_output_grid_t = getattr(native, fused_score_fn_name)(*fused_args)
        else:
            use_rankpos_score_grid = _env_truthy("SELECTOR_PQ_JOINT_RANKPOS_SCORE_GRID", "0")
            if bool(use_tokenfit_score_grid) and use_rankpos_score_grid:
                raise RuntimeError("tokenfit score grid does not support rank-position score grid")
            if bool(use_native_pq_scale_in_kernel) and use_rankpos_score_grid:
                raise RuntimeError("native in-kernel PQ scaling does not support rank-position score grid yet")
            if use_rankpos_score_grid and use_score_grid_no_fill:
                raise RuntimeError(
                    "SELECTOR_PQ_JOINT_RANKPOS_SCORE_GRID does not support no-fill diagnostic mode"
                )
            if bool(use_score_grid_workspace) and (
                use_rankpos_score_grid
                or bool(use_tokenfit_score_grid)
                or use_score_grid_no_fill
                or bool(use_native_pq_scale_in_kernel)
            ):
                raise RuntimeError(
                    "--use_score_grid_workspace only supports the canonical non-rankpos, "
                    "non-tokenfit, exact-fill score-grid path"
                )
            if bool(use_score_grid_workspace):
                if not hasattr(native, "joint_mixed_score_grid_workspace"):
                    raise RuntimeError("--use_score_grid_workspace requires updated CUDA extension")
                score_workspace_t = torch.empty(
                    (
                        int(k_take_counts_t.numel()),
                        int(exact_scores_t.shape[0]),
                        int(context_len),
                    ),
                    dtype=torch.float32,
                    device=device,
                )
                mask_workspace_t = torch.empty_like(score_workspace_t, dtype=torch.uint8)
                fit_scale_workspace_t = torch.empty(
                    (int(k_take_counts_t.numel()), int(exact_scores_t.shape[0])),
                    dtype=torch.float32,
                    device=device,
                )
                fit_bias_workspace_t = torch.empty_like(fit_scale_workspace_t)
                native_score_grid_t = native.joint_mixed_score_grid_workspace(
                    score_workspace_t,
                    mask_workspace_t,
                    fit_scale_workspace_t,
                    fit_bias_workspace_t,
                    exact_scores_t.to(dtype=torch.float32).contiguous(),
                    pq_logits_for_grid_t.contiguous(),
                    y_indexed_for_grid_t.contiguous(),
                    indexed_tokens_t.to(dtype=torch.long).contiguous(),
                    base_t.to(dtype=torch.long).contiguous(),
                    ranked_prefix_for_grid_t.to(dtype=torch.long).contiguous(),
                    k_take_counts_t,
                    bool(str(args.tail_score_calibration) == "affine_selected"),
                )
            if use_rankpos_score_grid:
                if not hasattr(native, "joint_mixed_score_grid_rankpos"):
                    raise RuntimeError("SELECTOR_PQ_JOINT_RANKPOS_SCORE_GRID requires updated CUDA extension")
                score_grid_fn = native.joint_mixed_score_grid_rankpos
            elif bool(use_tokenfit_score_grid):
                scaled_fn_name = (
                    "joint_mixed_score_grid_no_exact_fill_tokenfit_scaled"
                    if use_score_grid_no_fill
                    else "joint_mixed_score_grid_tokenfit_scaled"
                )
                if not hasattr(native, scaled_fn_name):
                    raise RuntimeError(f"tokenfit score grid requires {scaled_fn_name}")
                native_score_grid_t = getattr(native, scaled_fn_name)(
                    exact_scores_t.to(dtype=torch.float32).contiguous(),
                    pq_logits_for_grid_t.contiguous(),
                    y_indexed_for_grid_t.contiguous(),
                    indexed_tokens_t.to(dtype=torch.long).contiguous(),
                    base_t.to(dtype=torch.long).contiguous(),
                    ranked_prefix_for_grid_t.to(dtype=torch.long).contiguous(),
                    k_take_counts_t,
                    bool(str(args.tail_score_calibration) == "affine_selected"),
                    float(pq_logit_scale),
                )
            else:
                score_grid_fn = (
                    getattr(native, "joint_mixed_score_grid_no_exact_fill")
                    if use_score_grid_no_fill
                    and hasattr(native, "joint_mixed_score_grid_no_exact_fill")
                    else native.joint_mixed_score_grid
                )
            if native_score_grid_t is not None:
                pass
            elif bool(use_native_pq_scale_in_kernel):
                scaled_fn_name = (
                    "joint_mixed_score_grid_no_exact_fill_scaled"
                    if use_score_grid_no_fill
                    else "joint_mixed_score_grid_scaled"
                )
                if not hasattr(native, scaled_fn_name):
                    raise RuntimeError(f"native in-kernel PQ scaling requires {scaled_fn_name}")
                native_score_grid_t = getattr(native, scaled_fn_name)(
                    exact_scores_t.to(dtype=torch.float32).contiguous(),
                    pq_logits_for_grid_t.contiguous(),
                    y_indexed_for_grid_t.contiguous(),
                    indexed_tokens_t.to(dtype=torch.long).contiguous(),
                    base_t.to(dtype=torch.long).contiguous(),
                    ranked_prefix_for_grid_t.to(dtype=torch.long).contiguous(),
                    k_take_counts_t,
                    bool(str(args.tail_score_calibration) == "affine_selected"),
                    float(pq_logit_scale),
                )
            else:
                native_score_grid_t = score_grid_fn(
                    exact_scores_t.to(dtype=torch.float32).contiguous(),
                    pq_logits_for_grid_t.contiguous(),
                    y_indexed_for_grid_t.contiguous(),
                    indexed_tokens_t.to(dtype=torch.long).contiguous(),
                    base_t.to(dtype=torch.long).contiguous(),
                    ranked_prefix_for_grid_t.to(dtype=torch.long).contiguous(),
                    k_take_counts_t,
                    bool(str(args.tail_score_calibration) == "affine_selected"),
                )
        if native_probs_grid_t is None and _env_truthy("SELECTOR_PQ_JOINT_NATIVE_SOFTMAX_BASE", "0"):
            if not hasattr(native, "joint_softmax_base_outputs"):
                raise RuntimeError("SELECTOR_PQ_JOINT_NATIVE_SOFTMAX_BASE requires updated CUDA extension")
            if native_score_grid_t is None:
                raise RuntimeError("missing native score grid for native softmax/base")
            if _env_truthy("SELECTOR_PQ_JOINT_GROUPED_SOFTMAX_BASE_CUBLAS", "0"):
                if not hasattr(native, "joint_softmax_base_outputs_grouped_cublas"):
                    raise RuntimeError(
                        "SELECTOR_PQ_JOINT_GROUPED_SOFTMAX_BASE_CUBLAS requires updated CUDA extension"
                    )
                grouped_probs_t, grouped_base_t = native.joint_softmax_base_outputs_grouped_cublas(
                    native_score_grid_t.to(dtype=torch.float32).reshape(
                        1,
                        int(native_score_grid_t.shape[0]),
                        int(native_score_grid_t.shape[1]),
                        int(native_score_grid_t.shape[2]),
                    ).contiguous(),
                    vhat_all_t.to(dtype=torch.float32).reshape(
                        1,
                        int(vhat_all_t.shape[0]),
                        int(vhat_all_t.shape[1]),
                    ).contiguous(),
                )
                native_probs_grid_t = grouped_probs_t[0].contiguous()
                native_base_output_grid_t = grouped_base_t[0].contiguous()
            else:
                softmax_base_fn = (
                    getattr(native, "joint_softmax_base_outputs_cublas")
                    if _env_truthy("SELECTOR_PQ_JOINT_SOFTMAX_BASE_CUBLAS", "0")
                    and hasattr(native, "joint_softmax_base_outputs_cublas")
                    else native.joint_softmax_base_outputs
                )
                native_probs_grid_t, native_base_output_grid_t = softmax_base_fn(
                    native_score_grid_t.to(dtype=torch.float32).contiguous(),
                    vhat_all_t.to(dtype=torch.float32).contiguous(),
                )

    actual_value_subbits = int(args.value_subbits) if int(args.value_subbits) > 0 else int(args.subbits)
    actual_value_subvecs = int(args.value_subvecs) if int(args.value_subvecs) > 0 else int(args.subvecs)
    code_bytes = 1 if int(actual_value_subbits) <= 8 else 2
    metadata_mb = (
        float(context_len * actual_value_subvecs * code_bytes)
        + float(len(index.pages) * actual_value_subvecs * (1 << int(actual_value_subbits)) * int(args.value_code_stat_bytes))
    ) / MB
    v_pq_codebook_mb = float(
        len(index.pages)
        * actual_value_subvecs
        * (1 << int(actual_value_subbits))
        * (int(args.head_dim) // max(1, actual_value_subvecs))
        * int(args.value_bytes)
    ) / MB
    v_mb_by_idx: list[float] = []
    for v_budget in v_budgets:
        exact_count = max(0, min(int(v_budget), context_len))
        exact_v_mb = float(exact_count * int(args.head_dim) * int(args.value_bytes)) / MB
        compressed_v_codes_mb = float(max(0, context_len - exact_count) * actual_value_subvecs * code_bytes) / MB
        v_mb_by_idx.append(exact_v_mb + v_pq_codebook_mb + compressed_v_codes_mb + metadata_mb)

    if bool(use_score_direct_interval_policy):
        if not (bool(use_native_score_grid) and bool(use_native_policy)):
            raise RuntimeError("--use_score_direct_interval_policy requires --use_native_score_grid and --use_native_policy")
        if _joint_kv_policy_id(str(policy)) > 3:
            raise RuntimeError("--use_score_direct_interval_policy only supports non-MB policies")
        if native_score_grid_t is None:
            raise RuntimeError("--use_score_direct_interval_policy requires a native score grid")
        native = load_selector_paged_pq_ext()
        if not hasattr(native, "joint_select_policy_from_grouped_scores_intervals_batched_no_mb"):
            raise RuntimeError("--use_score_direct_interval_policy requires updated CUDA extension")
        selected_by_idx_direct: list[torch.Tensor] = []
        k_mb_by_idx_direct: list[float] = []
        for k_budget in active_k_budgets:
            selected_t = selected_for_budget(int(k_budget))
            selected_by_idx_direct.append(selected_t)
            exact_key_mb = float(int(selected_t.shape[1]) * int(args.head_dim) * int(args.key_bytes)) / MB
            k_mb_by_idx_direct.append(float(selector_mb) + exact_key_mb)
        final_output_grouped_t, final_idx_grouped_t = native.joint_select_policy_from_grouped_scores_intervals_batched_no_mb(
            native_score_grid_t.to(dtype=torch.float32).reshape(
                1,
                int(native_score_grid_t.shape[0]),
                int(native_score_grid_t.shape[1]),
                int(native_score_grid_t.shape[2]),
            ).contiguous(),
            vhat_all_t.to(dtype=torch.float32).reshape(1, context_len, int(args.head_dim)).contiguous(),
            residual_t.to(dtype=torch.float32).reshape(1, context_len, int(args.head_dim)).contiguous(),
            code_error_t.to(dtype=torch.float32).reshape(1, context_len).contiguous(),
            torch.as_tensor(v_budgets, dtype=torch.long, device=device),
            float(threshold),
            int(_joint_kv_policy_id(str(policy))),
        )
        ki = int(final_idx_grouped_t[0, 0, 0].item())
        vi = int(final_idx_grouped_t[0, 0, 1].item())
        selected_t = selected_by_idx_direct[int(ki)]
        out_np = final_output_grouped_t[0, 0].reshape(-1).detach().cpu().numpy().astype(np.float32, copy=False)
        return {
            "k_idx": int(ki),
            "v_idx": int(vi),
            "k_budget": int(active_k_budgets[int(ki)]),
            "v_budget": int(v_budgets[int(vi)]),
            "selected_count": int(selected_t.shape[1]),
            "step_MB_per_head": float(k_mb_by_idx_direct[int(ki)] + v_mb_by_idx[int(vi)]),
            "selector_MB_per_head": float(selector_mb),
            "policy_steps": -1,
            "final_k_delta": 0.0,
            "final_v_delta": 0.0,
            "output": out_np,
        }

    outputs: dict[tuple[int, int], torch.Tensor] = {}
    selected_by_idx: list[torch.Tensor] = []
    k_mb_by_idx: list[float] = []
    max_exact_v_count = max([max(0, min(int(v_budget), context_len)) for v_budget in v_budgets], default=0)
    output_rows: list[torch.Tensor] = []
    interval_base_rows: list[torch.Tensor] = []
    interval_prob_rows: list[torch.Tensor] = []
    interval_final_output_t: torch.Tensor | None = None
    interval_final_idx_t: torch.Tensor | None = None
    for ki, k_budget in enumerate(active_k_budgets):
        selected_t = selected_for_budget(int(k_budget))
        selected_by_idx.append(selected_t)
        selected_len = int(selected_t.shape[1])
        if native_probs_grid_t is not None:
            probs_t = native_probs_grid_t[int(ki)]
        elif native_score_grid_t is not None:
            probs_t = (
                torch.softmax(native_score_grid_t[int(ki)], dim=1)
            )
        else:
            probs_t = torch.softmax(mixed_scores_for_selected(selected_t), dim=1)
        exact_key_mb = float(selected_len * int(args.head_dim) * int(args.key_bytes)) / MB
        k_mb_by_idx.append(float(selector_mb) + exact_key_mb)
        base_output_t = (
            native_base_output_grid_t[int(ki)]
            if native_base_output_grid_t is not None
            else probs_t.to(torch.float32) @ vhat_all_t.float()
        )
        if bool(use_interval_risk_policy):
            interval_base_rows.append(base_output_t.reshape(1, -1).to(dtype=torch.float32))
            interval_prob_rows.append(probs_t.reshape(1, -1).to(dtype=torch.float32))
        if bool(use_native_vprefix):
            if bool(use_interval_risk_policy):
                continue
            if int(max_exact_v_count) > 0:
                risk_t = (probs_t * probs_t) * code_error_t.to(dtype=prob_dtype).reshape(1, -1)
                native = load_selector_paged_pq_ext()
                v_budgets_t = torch.as_tensor(v_budgets, dtype=torch.long, device=device)
                if bool(use_native_risk_prefix):
                    if bool(use_native_risk_prefix_topk):
                        if not hasattr(native, "joint_vprefix_outputs_from_grouped_risk_topk_batched"):
                            raise RuntimeError("native risk-prefix top-k helper is unavailable")
                        topk_grid_t = native.joint_vprefix_outputs_from_grouped_risk_topk_batched(
                            base_output_t.reshape(1, 1, 1, -1).contiguous(),
                            probs_t.reshape(1, 1, 1, -1).contiguous(),
                            residual_t.reshape(1, context_len, int(args.head_dim)).contiguous(),
                            code_error_t.to(dtype=torch.float32).reshape(1, context_len).contiguous(),
                            v_budgets_t,
                            int(max_exact_v_count),
                        )
                        grid_t = topk_grid_t.reshape(1, len(v_budgets), 1, int(args.head_dim)).contiguous()
                    else:
                        grid_t = native.joint_vprefix_outputs_from_risk(
                            base_output_t.reshape(1, 1, -1).contiguous(),
                            probs_t.reshape(1, 1, -1).contiguous(),
                            residual_t.contiguous(),
                            code_error_t.to(dtype=torch.float32).contiguous(),
                            v_budgets_t,
                        )
                else:
                    if int(max_exact_v_count) >= context_len:
                        exact_order_t = torch.argsort(risk_t, dim=1, descending=True, stable=True)
                    else:
                        exact_order_t = torch.topk(
                            risk_t,
                            k=int(max_exact_v_count),
                            dim=1,
                            largest=True,
                            sorted=True,
                        ).indices
                    grid_t = native.joint_vprefix_outputs(
                        base_output_t.reshape(1, 1, -1).contiguous(),
                        probs_t.reshape(1, 1, -1).contiguous(),
                        residual_t.contiguous(),
                        exact_order_t.reshape(1, 1, -1).contiguous(),
                        v_budgets_t,
                    )
                output_rows.append(grid_t[0])
            else:
                output_rows.append(base_output_t.reshape(1, 1, -1).expand(1, len(v_budgets), -1, -1)[0])
            continue
        prefix_delta_t: torch.Tensor | None = None
        if int(max_exact_v_count) > 0:
            risk_t = (probs_t * probs_t) * code_error_t.to(dtype=prob_dtype).reshape(1, -1)
            if int(max_exact_v_count) >= context_len:
                exact_order_t = torch.argsort(risk_t, dim=1, descending=True, stable=True)
            else:
                exact_order_t = torch.topk(
                    risk_t,
                    k=int(max_exact_v_count),
                    dim=1,
                    largest=True,
                    sorted=True,
                ).indices
            gathered_probs_t = torch.gather(probs_t.to(torch.float32), 1, exact_order_t)
            gathered_residual_t = residual_t.index_select(0, exact_order_t.reshape(-1)).reshape(
                1,
                int(exact_order_t.shape[1]),
                int(args.head_dim),
            )
            prefix_delta_t = torch.cumsum(gathered_probs_t.reshape(1, -1, 1) * gathered_residual_t.float(), dim=1)
        for vi, v_budget in enumerate(v_budgets):
            exact_count = max(0, min(int(v_budget), context_len))
            if exact_count > 0 and prefix_delta_t is not None:
                outputs[(ki, vi)] = base_output_t + prefix_delta_t[:, int(exact_count) - 1, :]
            else:
                outputs[(ki, vi)] = base_output_t

    if bool(use_native_vprefix):
        if bool(use_interval_risk_policy):
            if not bool(use_native_risk_prefix):
                raise RuntimeError("--use_interval_risk_policy requires --use_native_risk_prefix")
            if not bool(use_native_policy):
                raise RuntimeError("--use_interval_risk_policy requires --use_native_policy")
            if _joint_kv_policy_id(str(policy)) > 3:
                raise RuntimeError("--use_interval_risk_policy only supports non-MB policies")
            native = load_selector_paged_pq_ext()
            if not hasattr(native, "joint_select_policy_from_grouped_risk_intervals_batched_no_mb"):
                raise RuntimeError("--use_interval_risk_policy requires updated CUDA extension")
            base_grouped_t = torch.stack(
                [row.reshape(-1) for row in interval_base_rows],
                dim=0,
            ).reshape(1, len(active_k_budgets), 1, int(args.head_dim)).contiguous()
            probs_grouped_t = torch.stack(
                [row.reshape(-1) for row in interval_prob_rows],
                dim=0,
            ).reshape(1, len(active_k_budgets), 1, context_len).contiguous()
            interval_final_output_grouped_t, interval_final_idx_grouped_t = (
                native.joint_select_policy_from_grouped_risk_intervals_batched_no_mb(
                    base_grouped_t,
                    probs_grouped_t,
                    residual_t.reshape(1, context_len, int(args.head_dim)).contiguous(),
                    code_error_t.to(dtype=torch.float32).reshape(1, context_len).contiguous(),
                    torch.as_tensor(v_budgets, dtype=torch.long, device=device),
                    float(threshold),
                    int(_joint_kv_policy_id(str(policy))),
                )
            )
            interval_final_output_t = interval_final_output_grouped_t[0, 0].reshape(1, -1)
            interval_final_idx_t = interval_final_idx_grouped_t[0, 0].reshape(1, 2)
        else:
            output_grid_t = torch.stack(output_rows, dim=0)
            for ki in range(len(active_k_budgets)):
                for vi in range(len(v_budgets)):
                    outputs[(ki, vi)] = output_grid_t[int(ki), int(vi)]

    if bool(use_native_policy):
        native = load_selector_paged_pq_ext()
        if bool(use_interval_risk_policy):
            if interval_final_idx_t is None:
                raise RuntimeError("interval risk policy did not produce final indices")
            final_idx = interval_final_idx_t.detach().cpu().numpy()
        elif bool(use_native_vprefix):
            output_grid_for_policy_t = output_grid_t.to(dtype=torch.float32).contiguous()
        else:
            output_grid_for_policy_t = torch.stack(
                [
                    torch.stack([outputs[(ki_i, vi_i)] for vi_i in range(len(v_budgets))], dim=0)
                    for ki_i in range(len(active_k_budgets))
                ],
                dim=0,
            ).to(dtype=torch.float32).contiguous()
        if not bool(use_interval_risk_policy):
            k_mb_t = torch.as_tensor(k_mb_by_idx, dtype=torch.float32, device=device)
            v_mb_t = torch.as_tensor(v_mb_by_idx, dtype=torch.float32, device=device)
            final_idx_t = native.joint_select_policy(
                output_grid_for_policy_t,
                k_mb_t,
                v_mb_t,
                float(threshold),
                int(_joint_kv_policy_id(str(policy))),
            )
            final_idx = final_idx_t.detach().cpu().numpy()
        ki = int(final_idx[0, 0])
        vi = int(final_idx[0, 1])
        steps = -1
        final_k_delta = 0.0
        final_v_delta = 0.0
    else:
        ki, vi, steps, final_k_delta, final_v_delta = _simulate_joint_kv_policy_torch(
            outputs=outputs,
            k_budgets=active_k_budgets,
            v_budgets=v_budgets,
            policy=str(policy),
            threshold=float(threshold),
            k_mb_by_idx=k_mb_by_idx,
            v_mb_by_idx=v_mb_by_idx,
        )
    selected_t = selected_by_idx[int(ki)]
    if bool(use_interval_risk_policy):
        if interval_final_output_t is None:
            raise RuntimeError("interval risk policy did not produce final output")
        out_t = interval_final_output_t
    else:
        out_t = outputs[(int(ki), int(vi))]
    out_np = out_t.reshape(-1).detach().cpu().numpy().astype(np.float32, copy=False)
    return {
        "k_idx": int(ki),
        "v_idx": int(vi),
        "k_budget": int(active_k_budgets[int(ki)]),
        "v_budget": int(v_budgets[int(vi)]),
        "selected_count": int(selected_t.shape[1]),
        "step_MB_per_head": float(k_mb_by_idx[int(ki)] + v_mb_by_idx[int(vi)]),
        "selector_MB_per_head": float(selector_mb),
        "policy_steps": int(steps),
        "final_k_delta": float(final_k_delta),
        "final_v_delta": float(final_v_delta),
        "output": out_np,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace-level CPU/CUDA parity for joint adaptive K/V frontier policy.")
    parser.add_argument("--trace", required=True)
    parser.add_argument("--x_trace", default="")
    parser.add_argument("--model_snapshot", default=".hf_cache/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--decode_lengths", default="500,1000")
    parser.add_argument("--heads", default="0,8")
    parser.add_argument("--policies", default="k_first_alternating")
    parser.add_argument("--thresholds", default="0.001")
    parser.add_argument("--k_budgets", default="4096,8192,14336,32768")
    parser.add_argument("--v_budgets", default="1024,2048,4096,6144,8192,12288,16384")
    parser.add_argument("--static_prefix", type=int, default=128)
    parser.add_argument("--static_suffix", type=int, default=128)
    parser.add_argument("--page_size", type=int, default=5632)
    parser.add_argument("--subvecs", type=int, default=4)
    parser.add_argument("--subbits", type=int, default=8)
    parser.add_argument("--value_subvecs", type=int, default=1)
    parser.add_argument("--value_subbits", type=int, default=4)
    parser.add_argument("--kmeans_iters", type=int, default=3)
    parser.add_argument("--key_bytes", type=int, default=2)
    parser.add_argument("--value_bytes", type=int, default=2)
    parser.add_argument("--value_code_stat_bytes", type=int, default=2)
    parser.add_argument("--tail_score_calibration", choices=["none", "affine_selected"], default="affine_selected")
    parser.add_argument("--router_prototypes", type=int, default=16)
    parser.add_argument("--router_merge_rel", type=float, default=0.05)
    parser.add_argument("--router_merge_var", type=float, default=0.0)
    parser.add_argument("--router_max_groups", type=int, default=512)
    parser.add_argument("--budget_tolerance", type=int, default=0)
    parser.add_argument("--mb_tolerance", type=float, default=1e-5)
    parser.add_argument("--output_rel_l2_tolerance", type=float, default=5e-4)
    parser.add_argument("--oproj_rel_l2_tolerance", type=float, default=5e-4)
    parser.add_argument(
        "--compare_torch_gpu_policy",
        action="store_true",
        help="Also compare the Torch/GPU joint K/V policy-output math against the CPU reference.",
    )
    parser.add_argument(
        "--use_native_vprefix",
        action="store_true",
        help="When comparing Torch/GPU policy, use the diagnostic native residual-risk V-prefix helper.",
    )
    parser.add_argument(
        "--use_native_risk_prefix",
        action="store_true",
        help="When comparing Torch/GPU policy, have the native V-prefix helper sort residual-risk scores internally.",
    )
    parser.add_argument(
        "--use_native_risk_prefix_topk",
        action="store_true",
        help="When comparing Torch/GPU policy, use the grouped top-k residual-risk prefix helper.",
    )
    parser.add_argument(
        "--use_native_score_grid",
        action="store_true",
        help="When comparing Torch/GPU policy, build the mixed exact-K/K-PQ score grid with the native CUDA helper.",
    )
    parser.add_argument(
        "--use_native_pq_scale_in_kernel",
        action="store_true",
        help="When comparing Torch/GPU policy, pass unscaled PQ logits and apply the QK scale inside the native score-grid helper.",
    )
    parser.add_argument(
        "--use_tokenfit_score_grid",
        action="store_true",
        help="When comparing Torch/GPU policy, use the mask-free token-list affine-fit native score-grid helper.",
    )
    parser.add_argument(
        "--use_score_grid_workspace",
        action="store_true",
        help="When comparing Torch/GPU policy, pass preallocated workspace tensors to the native score-grid helper.",
    )
    parser.add_argument(
        "--use_native_policy",
        action="store_true",
        help="When comparing Torch/GPU policy, choose adaptive K/V budget indices with the native CUDA helper.",
    )
    parser.add_argument(
        "--use_interval_risk_policy",
        action="store_true",
        help=(
            "When comparing Torch/GPU policy, choose adaptive K/V budgets directly from grouped "
            "residual-risk interval sums instead of materializing the full V-prefix output grid."
        ),
    )
    parser.add_argument(
        "--use_score_direct_interval_policy",
        action="store_true",
        help=(
            "When comparing Torch/GPU policy, choose adaptive K/V budgets directly from the "
            "native mixed-score grid, avoiding materialized probability and V-prefix grids."
        ),
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or str(args.device) == "cpu" else "cpu")
    args.device = device
    trace = load_trace(args.trace)
    args.head_dim = int(trace.head_dim)
    decode_lengths = parse_csv_ints(args.decode_lengths)
    q_indices = trace.q_indices_for_decodes(decode_lengths)
    if not q_indices:
        raise ValueError(f"no q indices for decode lengths {decode_lengths}")
    heads = parse_csv_ints(args.heads) if str(args.heads).strip() else list(range(int(trace.num_heads)))
    k_budgets = sorted({int(x) for x in parse_csv_ints(args.k_budgets) if int(x) > 0})
    v_budgets = sorted({int(x) for x in parse_csv_ints(args.v_budgets) if int(x) > 0})
    policies = [part.strip() for part in str(args.policies).split(",") if part.strip()]
    thresholds = _parse_float_list(args.thresholds)

    layer_idx = None
    wo = None
    x_trace = Path(args.x_trace) if str(args.x_trace).strip() else None
    model_dir = PROJECT_ROOT / args.model_snapshot
    if x_trace is not None and x_trace.exists() and model_dir.exists():
        x_data = np.load(x_trace, mmap_mode="r")
        x_meta = json.loads(str(x_data["metadata"].item()))
        layer_idx = int(x_meta["layer_idx"])
        weight_map = load_weight_index(model_dir)
        wo = load_safetensor_weight(model_dir, weight_map, f"model.layers.{layer_idx}.self_attn.o_proj.weight", device)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    failures: list[str] = []
    t0 = time.perf_counter()
    for qidx in q_indices:
        position = int(trace.positions[int(qidx)])
        context_len = int(position) + 1
        row_v_budgets = (
            _collapse_duplicate_v_budgets(v_budgets, context_len)
            if _env_truthy("SELECTOR_PQ_JOINT_COLLAPSE_DUP_V_ROWS", "0")
            else v_budgets
        )
        dynamic_start = min(max(0, int(args.static_prefix)), int(trace.input_len))
        indexed_end = max(
            min(max(0, int(args.static_prefix)), int(trace.input_len)),
            min(context_len - max(0, int(args.static_suffix)), trace.keys.shape[1]),
        )
        indexes = {}
        for kv_head in sorted({int(trace.kv_head_for(h)) for h in heads}):
            keys_np = trace.keys[kv_head, :context_len].astype(np.float32, copy=False)
            indexes[kv_head] = build_page_pq_gpu(
                keys_np,
                dynamic_start=dynamic_start,
                indexed_end=indexed_end,
                page_size=int(args.page_size),
                subvecs=int(args.subvecs),
                subbits=int(args.subbits),
                kmeans_iters=int(args.kmeans_iters),
                seed=2025 + 2027 * int(kv_head),
                key_bytes=int(args.key_bytes),
                router_enabled=False,
                router_prototypes=int(args.router_prototypes),
                router_merge_rel=float(args.router_merge_rel),
                router_merge_var=float(args.router_merge_var),
                router_max_groups=int(args.router_max_groups),
                device=device,
            )
        for policy in policies:
            for threshold in thresholds:
                cpu_heads: dict[int, np.ndarray] = {}
                gpu_heads: dict[int, np.ndarray] = {}
                torch_gpu_heads: dict[int, np.ndarray] = {}
                for head in heads:
                    kv_head = int(trace.kv_head_for(int(head)))
                    keys_np = trace.keys[kv_head, :context_len].astype(np.float32, copy=False)
                    values_np = trace.values[kv_head, :context_len].astype(np.float32, copy=False)
                    query_np = trace.queries[int(head), int(qidx)].astype(np.float32, copy=False)
                    common = dict(
                        query_np=query_np,
                        keys_np=keys_np,
                        values_np=values_np,
                        position=position,
                        input_len=int(trace.input_len),
                        index=indexes[kv_head],
                        k_budgets=k_budgets,
                        v_budgets=row_v_budgets,
                        policy=policy,
                        threshold=float(threshold),
                        args=args,
                    )
                    cpu = _evaluate_policy(selector_backend="torch", **common)
                    gpu = _evaluate_policy(selector_backend="cuda_ext", **common)
                    torch_gpu = (
                        _evaluate_policy_torch_gpu(
                            selector_backend="cuda_ext",
                            use_native_vprefix=bool(args.use_native_vprefix),
                            use_native_risk_prefix=bool(args.use_native_risk_prefix),
                            use_native_risk_prefix_topk=bool(args.use_native_risk_prefix_topk),
                            use_native_score_grid=bool(args.use_native_score_grid),
                            use_native_pq_scale_in_kernel=bool(args.use_native_pq_scale_in_kernel),
                            use_tokenfit_score_grid=bool(args.use_tokenfit_score_grid),
                            use_score_grid_workspace=bool(args.use_score_grid_workspace),
                            use_native_policy=bool(args.use_native_policy),
                            use_interval_risk_policy=bool(args.use_interval_risk_policy),
                            use_score_direct_interval_policy=bool(args.use_score_direct_interval_policy),
                            **common,
                        )
                        if bool(args.compare_torch_gpu_policy) and str(device) != "cpu"
                        else None
                    )
                    cpu_heads[int(head)] = cpu["output"]
                    gpu_heads[int(head)] = gpu["output"]
                    metric = _safe_output_error_metrics(cpu["output"], gpu["output"])
                    torch_metric = (
                        _safe_output_error_metrics(cpu["output"], torch_gpu["output"])
                        if torch_gpu is not None
                        else None
                    )
                    row = {
                        "qidx": int(qidx),
                        "decode_tokens": int(trace.decode_tokens_for_qidx(int(qidx))),
                        "head": int(head),
                        "policy": str(policy),
                        "threshold": float(threshold),
                        "cpu_k_budget": int(cpu["k_budget"]),
                        "gpu_k_budget": int(gpu["k_budget"]),
                        "cpu_v_budget": int(cpu["v_budget"]),
                        "gpu_v_budget": int(gpu["v_budget"]),
                        "cpu_selected_count": int(cpu["selected_count"]),
                        "gpu_selected_count": int(gpu["selected_count"]),
                        "cpu_step_MB_per_head": float(cpu["step_MB_per_head"]),
                        "gpu_step_MB_per_head": float(gpu["step_MB_per_head"]),
                        "attention_relative_L2": float(metric["output_relative_l2"]),
                        "attention_cosine": float(metric["output_cosine"]),
                    }
                    if torch_gpu is not None and torch_metric is not None:
                        torch_gpu_heads[int(head)] = torch_gpu["output"]
                        row.update(
                            {
                                "torch_gpu_k_budget": int(torch_gpu["k_budget"]),
                                "torch_gpu_v_budget": int(torch_gpu["v_budget"]),
                                "torch_gpu_selected_count": int(torch_gpu["selected_count"]),
                                "torch_gpu_step_MB_per_head": float(torch_gpu["step_MB_per_head"]),
                                "torch_gpu_attention_relative_L2": float(torch_metric["output_relative_l2"]),
                                "torch_gpu_attention_cosine": float(torch_metric["output_cosine"]),
                            }
                        )
                    rows.append(row)
                    if abs(int(cpu["k_budget"]) - int(gpu["k_budget"])) > int(args.budget_tolerance):
                        failures.append(f"k_budget mismatch row={len(rows)-1}")
                    if abs(int(cpu["v_budget"]) - int(gpu["v_budget"])) > int(args.budget_tolerance):
                        failures.append(f"v_budget mismatch row={len(rows)-1}")
                    if abs(float(cpu["step_MB_per_head"]) - float(gpu["step_MB_per_head"])) > float(args.mb_tolerance):
                        failures.append(f"step MB mismatch row={len(rows)-1}")
                    if float(metric["output_relative_l2"]) > float(args.output_rel_l2_tolerance):
                        failures.append(f"attention output relL2 too high row={len(rows)-1}: {metric['output_relative_l2']}")
                    if torch_gpu is not None and torch_metric is not None:
                        duplicate_k_budget_equiv = bool(
                            _env_truthy("SELECTOR_PQ_JOINT_COLLAPSE_DUP_K_ROWS", "0")
                            and int(cpu["selected_count"]) == int(torch_gpu["selected_count"])
                            and abs(float(cpu["step_MB_per_head"]) - float(torch_gpu["step_MB_per_head"]))
                            <= float(args.mb_tolerance)
                        )
                        if (
                            abs(int(cpu["k_budget"]) - int(torch_gpu["k_budget"])) > int(args.budget_tolerance)
                            and not duplicate_k_budget_equiv
                        ):
                            failures.append(f"torch_gpu k_budget mismatch row={len(rows)-1}")
                        if abs(int(cpu["v_budget"]) - int(torch_gpu["v_budget"])) > int(args.budget_tolerance):
                            failures.append(f"torch_gpu v_budget mismatch row={len(rows)-1}")
                        if abs(float(cpu["step_MB_per_head"]) - float(torch_gpu["step_MB_per_head"])) > float(args.mb_tolerance):
                            failures.append(f"torch_gpu step MB mismatch row={len(rows)-1}")
                        if float(torch_metric["output_relative_l2"]) > float(args.output_rel_l2_tolerance):
                            failures.append(
                                f"torch_gpu attention output relL2 too high row={len(rows)-1}: "
                                f"{torch_metric['output_relative_l2']}"
                            )

                if wo is not None:
                    cpu_concat = np.concatenate([cpu_heads[int(h)] for h in heads], axis=0).astype(np.float32, copy=False)
                    gpu_concat = np.concatenate([gpu_heads[int(h)] for h in heads], axis=0).astype(np.float32, copy=False)
                    cpu_proj = project_head_subset(
                        concat_subset=cpu_concat,
                        heads=[int(h) for h in heads],
                        num_heads=int(trace.num_heads),
                        head_dim=int(trace.head_dim),
                        wo=wo,
                        device=device,
                    )
                    gpu_proj = project_head_subset(
                        concat_subset=gpu_concat,
                        heads=[int(h) for h in heads],
                        num_heads=int(trace.num_heads),
                        head_dim=int(trace.head_dim),
                        wo=wo,
                        device=device,
                    )
                    proj_metric = _safe_output_error_metrics(cpu_proj, gpu_proj)
                    rows.append(
                        {
                            "qidx": int(qidx),
                            "decode_tokens": int(trace.decode_tokens_for_qidx(int(qidx))),
                            "head": "o_proj_subset",
                            "policy": str(policy),
                            "threshold": float(threshold),
                            "oproj_relative_L2": float(proj_metric["output_relative_l2"]),
                            "oproj_cosine": float(proj_metric["output_cosine"]),
                            "layer_idx": int(layer_idx) if layer_idx is not None else None,
                        }
                    )
                    if float(proj_metric["output_relative_l2"]) > float(args.oproj_rel_l2_tolerance):
                        failures.append(f"o-proj relL2 too high qidx={qidx}: {proj_metric['output_relative_l2']}")
                    if bool(args.compare_torch_gpu_policy) and len(torch_gpu_heads) == len(heads):
                        torch_gpu_concat = np.concatenate(
                            [torch_gpu_heads[int(h)] for h in heads],
                            axis=0,
                        ).astype(np.float32, copy=False)
                        torch_gpu_proj = project_head_subset(
                            concat_subset=torch_gpu_concat,
                            heads=[int(h) for h in heads],
                            num_heads=int(trace.num_heads),
                            head_dim=int(trace.head_dim),
                            wo=wo,
                            device=device,
                        )
                        torch_proj_metric = _safe_output_error_metrics(cpu_proj, torch_gpu_proj)
                        rows.append(
                            {
                                "qidx": int(qidx),
                                "decode_tokens": int(trace.decode_tokens_for_qidx(int(qidx))),
                                "head": "torch_gpu_o_proj_subset",
                                "policy": str(policy),
                                "threshold": float(threshold),
                                "torch_gpu_oproj_relative_L2": float(torch_proj_metric["output_relative_l2"]),
                                "torch_gpu_oproj_cosine": float(torch_proj_metric["output_cosine"]),
                                "layer_idx": int(layer_idx) if layer_idx is not None else None,
                            }
                        )
                        if float(torch_proj_metric["output_relative_l2"]) > float(args.oproj_rel_l2_tolerance):
                            failures.append(
                                f"torch_gpu o-proj relL2 too high qidx={qidx}: "
                                f"{torch_proj_metric['output_relative_l2']}"
                            )

    summary = {
        "ok": not failures,
        "failures": failures,
        "rows": int(len(rows)),
        "elapsed_seconds": float(time.perf_counter() - t0),
        "max_attention_relative_L2": float(
            max([float(r.get("attention_relative_L2", 0.0)) for r in rows], default=0.0)
        ),
        "max_torch_gpu_attention_relative_L2": float(
            max([float(r.get("torch_gpu_attention_relative_L2", 0.0)) for r in rows], default=0.0)
        ),
        "max_oproj_relative_L2": float(max([float(r.get("oproj_relative_L2", 0.0)) for r in rows], default=0.0)),
        "max_torch_gpu_oproj_relative_L2": float(
            max([float(r.get("torch_gpu_oproj_relative_L2", 0.0)) for r in rows], default=0.0)
        ),
        "config": vars(args) | {"device": str(device)},
    }
    (out_dir / "rows.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
