#!/usr/bin/env python3
from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch

from benchmark.selector_eval.gpu.run_gpu_paged_pq_eval import (
    GPUIndex,
    build_page_pq_torch,
    ensure_native_fullscan_pack,
)
from benchmark.selector_eval.metrics.attention import _output_error_metrics
from benchmark.selector_eval.runners.diagnose_layer_heads import _build_value_vpq_sidecars


def parse_head_budget_map(text: str) -> dict[int, int]:
    out: dict[int, int] = {}
    for part in str(text or "").split(","):
        part = part.strip()
        if not part:
            continue
        head, budget = part.split(":", 1)
        out[int(head)] = int(budget)
    return out


def parse_int_set(text: str) -> set[int]:
    return {int(part.strip()) for part in str(text or "").split(",") if part.strip()}


def output_metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    return _output_error_metrics(
        a.detach().float().cpu().numpy().reshape(-1),
        b.detach().float().cpu().numpy().reshape(-1),
    )


def selected_value_exact_mask(
    *,
    selected_scores: np.ndarray,
    values_exact: np.ndarray,
    values_approx: np.ndarray | None,
    rule: str,
    exact_top: int,
    exact_mass: float,
    exact_risk_mass: float,
    min_top: int,
    max_top: int,
) -> tuple[np.ndarray, float]:
    count = int(selected_scores.shape[0])
    mask = np.zeros((count,), dtype=bool)
    if count <= 0:
        return mask, 0.0
    scores = selected_scores.astype(np.float64, copy=False)
    if str(rule) == "fixed":
        order = np.argsort(-scores, kind="stable")
        exact_count = int(exact_top)
        if exact_count > 0:
            mask[order[: min(count, exact_count)]] = True
        return mask, 0.0
    elif str(rule) == "selector_rank":
        exact_count = int(exact_top)
        if exact_count > 0:
            mask[: min(count, exact_count)] = True
        return mask, 0.0

    shifted = scores - float(scores.max())
    probs = np.exp(shifted)
    probs /= max(float(probs.sum()), 1e-20)
    if str(rule) == "selected_mass":
        order = np.argsort(-scores, kind="stable")
        target = float(max(0.0, min(1.0, exact_mass)))
        cumulative = np.cumsum(probs[order])
        exact_count = int(np.searchsorted(cumulative, target, side="left") + 1) if target > 0.0 else 0
        if exact_count > 0:
            mask[order[: min(count, exact_count)]] = True
    elif str(rule) in {"selected_risk_mass", "selected_mass_or_risk"}:
        if values_approx is None:
            raise ValueError(f"{rule} requires approximate selected values")
        residual_norm = np.linalg.norm(
            values_exact.astype(np.float32, copy=False) - values_approx.astype(np.float32, copy=False),
            axis=1,
        ) / float(np.sqrt(float(values_exact.shape[-1])))
        risk = probs * residual_norm.astype(np.float64, copy=False)
        risk_order = np.argsort(-risk, kind="stable")
        target = float(exact_risk_mass) if float(exact_risk_mass) > 0.0 else float(exact_mass)
        total_risk = float(risk.sum())
        if total_risk > 1e-20 and target > 0.0:
            cumulative = np.cumsum(risk[risk_order]) / total_risk
            exact_count = int(np.searchsorted(cumulative, float(max(0.0, min(1.0, target))), side="left") + 1)
        else:
            exact_count = int(exact_top)
        if exact_count > 0:
            mask[risk_order[: min(count, exact_count)]] = True
        if str(rule) == "selected_mass_or_risk":
            prob_order = np.argsort(-scores, kind="stable")
            mass_target = float(max(0.0, min(1.0, exact_mass)))
            if mass_target > 0.0:
                cumulative = np.cumsum(probs[prob_order])
                mass_count = int(np.searchsorted(cumulative, mass_target, side="left") + 1)
                mask[prob_order[: min(count, mass_count)]] = True
    else:
        raise ValueError(f"unknown selected_value_exact_rule: {rule}")
    if int(min_top) > 0 and int(np.sum(mask)) < int(min_top):
        order = np.argsort(-scores, kind="stable")
        mask[order[: min(count, int(min_top))]] = True
    if int(max_top) > 0 and int(np.sum(mask)) > int(max_top):
        order = np.argsort(-(probs * (1.0 + np.arange(count, 0, -1) / max(1, count))), kind="stable")
        limited = np.zeros((count,), dtype=bool)
        limited[order[: min(count, int(max_top))]] = True
        mask = limited
    return mask, float(probs[mask].sum()) if bool(np.any(mask)) else 0.0


def selected_value_exact_top_positive(args: Any) -> int:
    return max(0, int(args.selected_value_exact_top))


def native_selected_value_exact_top_arg(args: Any) -> int:
    exact_top = selected_value_exact_top_positive(args)
    if str(args.selected_value_exact_rule) == "selector_rank" and exact_top > 0:
        return -exact_top
    return exact_top


def value_vpq_pack_gpu(
    *,
    index: GPUIndex,
    values_np: np.ndarray,
    subbits: int,
    value_subvecs: int,
    value_subbits: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int] | None:
    if not index.pages:
        return None
    page_size = int(index.pages[0].size)
    actual_value_subbits = int(value_subbits) if int(value_subbits) > 0 else int(subbits)
    cache_key = (str(device), int(value_subvecs), int(actual_value_subbits))
    cached = getattr(index, "_value_vpq_gpu_pack_by_params", None)
    if isinstance(cached, dict) and cache_key in cached:
        return cached[cache_key]
    if any(int(page.size) != page_size for page in index.pages):
        return None
    sidecars = _build_value_vpq_sidecars(
        index,
        values_np,
        int(subbits),
        value_subvecs=int(value_subvecs),
        value_subbits=int(actual_value_subbits),
    )
    if not sidecars or any(codebook.size == 0 or codes.size == 0 for codebook, codes in sidecars):
        return None
    codebooks_np = np.stack([codebook.astype(np.float32, copy=False) for codebook, _codes in sidecars], axis=0)
    codes_np = np.stack([codes.astype(np.int64, copy=False) for _codebook, codes in sidecars], axis=0)
    codebooks = torch.as_tensor(codebooks_np, dtype=torch.float32, device=device)
    codes_dtype = torch.uint8 if int(actual_value_subbits) <= 8 else torch.long
    codes = torch.as_tensor(codes_np, dtype=codes_dtype, device=device)
    page_starts = torch.as_tensor([int(page.start) for page in index.pages], dtype=torch.long, device=device)
    packed = (codebooks, codes, page_starts, int(page_size), int(actual_value_subbits))
    if not isinstance(cached, dict):
        cached = {}
    cached[cache_key] = packed
    setattr(index, "_value_vpq_gpu_pack_by_params", cached)
    return packed


def value_vpq_pack_torch(
    *,
    index: GPUIndex,
    values: torch.Tensor,
    value_subvecs: int,
    value_subbits: int,
    key_bytes: int,
    device: torch.device,
    value_group_pages: int = 1,
    kmeans_iters: int = 3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int] | None:
    if not index.pages:
        return None
    selection_page_size = int(index.pages[0].size)
    if selection_page_size <= 0:
        return None
    group_pages = max(1, int(value_group_pages))
    page_size = int(selection_page_size * group_pages)
    actual_value_subvecs = int(value_subvecs) if int(value_subvecs) > 0 else int(index.pages[0].codes.shape[1])
    actual_value_subbits = int(value_subbits) if int(value_subbits) > 0 else 8
    cache_key = (
        str(device),
        int(actual_value_subvecs),
        int(actual_value_subbits),
        int(group_pages),
        int(kmeans_iters),
        "torch",
    )
    cached = getattr(index, "_value_vpq_gpu_pack_by_params", None)
    if isinstance(cached, dict) and cache_key in cached:
        setattr(index, "_last_value_vpq_build_stats", None)
        return cached[cache_key]
    if any(int(page.size) != selection_page_size for page in index.pages):
        return None
    dynamic_start = int(index.pages[0].start)
    indexed_end = int(index.pages[-1].start) + int(index.pages[-1].size)
    v_index = build_page_pq_torch(
        values,
        dynamic_start=dynamic_start,
        indexed_end=indexed_end,
        page_size=page_size,
        subvecs=int(actual_value_subvecs),
        subbits=int(actual_value_subbits),
        kmeans_iters=int(kmeans_iters),
        seed=90210 + int(dynamic_start) + 1000003 * int(group_pages),
        key_bytes=int(key_bytes),
        router_enabled=False,
        router_prototypes=0,
        router_merge_rel=0.0,
        router_merge_var=0.0,
        router_max_groups=0,
        device=device,
    )
    if not v_index.pages:
        setattr(index, "_last_value_vpq_build_stats", None)
        return None
    codebooks, codes, page_starts = ensure_native_fullscan_pack(v_index, subbits=int(actual_value_subbits))
    packed = (codebooks, codes, page_starts, int(page_size), int(actual_value_subbits))
    setattr(
        index,
        "_last_value_vpq_build_stats",
        (
            float(v_index.build_seconds),
            float(v_index.build_read_mb),
            float(v_index.build_write_mb),
        ),
    )
    if not isinstance(cached, dict):
        cached = {}
    cached[cache_key] = packed
    setattr(index, "_value_vpq_gpu_pack_by_params", cached)
    return packed


def vpq_values_for_tokens_gpu(
    *,
    index: GPUIndex,
    values: torch.Tensor,
    values_np: np.ndarray | None,
    tokens: torch.Tensor,
    subbits: int,
    value_subvecs: int,
    value_subbits: int,
    prefer_torch: bool = False,
    value_bytes: int = 2,
    kmeans_iters: int = 3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    if bool(prefer_torch):
        pack = value_vpq_pack_torch(
            index=index,
            values=values,
            value_subvecs=int(value_subvecs),
            value_subbits=int(value_subbits) if int(value_subbits) > 0 else int(subbits),
            key_bytes=int(value_bytes),
            device=values.device,
            kmeans_iters=int(kmeans_iters),
        )
    else:
        if values_np is None:
            raise ValueError("values_np is required for CPU-built V-PQ sidecars")
        pack = value_vpq_pack_gpu(
            index=index,
            values_np=values_np,
            subbits=int(subbits),
            value_subvecs=int(value_subvecs),
            value_subbits=int(value_subbits),
            device=values.device,
        )
    if pack is None or tokens.numel() == 0:
        exact_values = values.index_select(0, tokens.reshape(-1)).reshape(*tokens.shape, int(values.shape[-1])).float()
        return exact_values, torch.zeros_like(tokens, dtype=torch.bool), torch.full_like(tokens, -1), int(value_subbits)
    codebooks, codes, page_starts, page_size, actual_value_subbits = pack
    first_start = int(page_starts[0].item())
    page_ids = torch.div(tokens - first_start, int(page_size), rounding_mode="floor")
    in_range = (tokens >= first_start) & (page_ids >= 0) & (page_ids < int(page_starts.numel()))
    clamped_page_ids = torch.clamp(page_ids, min=0, max=max(0, int(page_starts.numel()) - 1))
    rows = tokens - page_starts.index_select(0, clamped_page_ids.reshape(-1)).reshape_as(tokens)
    valid = in_range & (rows >= 0) & (rows < int(page_size))
    if not bool(torch.any(valid)):
        exact_values = values.index_select(0, tokens.reshape(-1)).reshape(*tokens.shape, int(values.shape[-1])).float()
        return exact_values, valid, page_ids, int(actual_value_subbits)
    flat_valid = valid.reshape(-1)
    flat_page_ids = clamped_page_ids.reshape(-1).index_select(0, torch.nonzero(flat_valid, as_tuple=False).reshape(-1))
    flat_rows = rows.reshape(-1).index_select(0, torch.nonzero(flat_valid, as_tuple=False).reshape(-1)).to(torch.long)
    selected_codes = codes[flat_page_ids, flat_rows].to(torch.long)
    subvecs = int(codebooks.shape[1])
    subdim = int(codebooks.shape[-1])
    approx_flat = torch.empty((int(selected_codes.shape[0]), subvecs * subdim), dtype=torch.float32, device=values.device)
    sub_ids = torch.arange(subvecs, dtype=torch.long, device=values.device)
    for sub in range(subvecs):
        approx_flat[:, sub * subdim : (sub + 1) * subdim] = codebooks[
            flat_page_ids,
            sub_ids[sub].expand_as(flat_page_ids),
            selected_codes[:, sub],
        ]
    out = torch.empty((int(tokens.numel()), int(values.shape[-1])), dtype=torch.float32, device=values.device)
    out[flat_valid] = approx_flat
    if int(flat_valid.numel()) != int(approx_flat.shape[0]):
        invalid_flat = ~flat_valid
        if bool(torch.any(invalid_flat)):
            invalid_tokens = tokens.reshape(-1).index_select(
                0,
                torch.nonzero(invalid_flat, as_tuple=False).reshape(-1),
            )
            out[invalid_flat] = values.index_select(0, invalid_tokens).float()
    return out.reshape(*tokens.shape, int(values.shape[-1])), valid, page_ids, int(actual_value_subbits)


def reconstruct_all_vpq_values_gpu(
    *,
    index: GPUIndex,
    values_np: np.ndarray | None,
    values: torch.Tensor | None = None,
    subbits: int,
    value_subvecs: int,
    value_subbits: int,
    device: torch.device,
    prefer_torch: bool = False,
    value_bytes: int = 2,
) -> tuple[torch.Tensor, int] | None:
    actual_value_subbits_for_key = int(value_subbits) if int(value_subbits) > 0 else int(subbits)
    cache_key = (str(device), int(value_subvecs), int(actual_value_subbits_for_key), "torch" if bool(prefer_torch) else "numpy")
    cached = getattr(index, "_all_value_vpq_gpu_by_params", None)
    if isinstance(cached, dict) and cache_key in cached:
        return cached[cache_key]
    if bool(prefer_torch):
        if values is None:
            raise ValueError("values tensor is required for torch-built V-PQ sidecars")
        pack = value_vpq_pack_torch(
            index=index,
            values=values,
            value_subvecs=int(value_subvecs),
            value_subbits=int(value_subbits) if int(value_subbits) > 0 else int(subbits),
            key_bytes=int(value_bytes),
            device=device,
        )
    else:
        if values_np is None:
            raise ValueError("values_np is required for CPU-built V-PQ sidecars")
        pack = value_vpq_pack_gpu(
            index=index,
            values_np=values_np,
            subbits=int(subbits),
            value_subvecs=int(value_subvecs),
            value_subbits=int(value_subbits),
            device=device,
        )
    if pack is None:
        return None
    codebooks, codes, _page_starts, _page_size, actual_value_subbits = pack
    pages = int(codebooks.shape[0])
    page_size = int(codes.shape[1])
    subvecs = int(codebooks.shape[1])
    subdim = int(codebooks.shape[-1])
    flat_codes = codes.reshape(pages * page_size, subvecs).to(torch.long)
    page_ids = torch.arange(pages, dtype=torch.long, device=device).repeat_interleave(page_size)
    out = torch.empty((pages * page_size, subvecs * subdim), dtype=torch.float32, device=device)
    for sub in range(subvecs):
        out[:, sub * subdim : (sub + 1) * subdim] = codebooks[
            page_ids,
            torch.full_like(page_ids, int(sub)),
            flat_codes[:, sub],
        ]
    if not isinstance(cached, dict):
        cached = {}
    packed = (out, int(actual_value_subbits))
    cached[cache_key] = packed
    setattr(index, "_all_value_vpq_gpu_by_params", cached)
    return packed


def _bucket_sums_counts(
    bucket_ids: torch.Tensor,
    weights: torch.Tensor,
    *,
    bucket_count: int,
    bins_per_group: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    legacy_cuda_atomics = str(
        os.environ.get("PAGEDPQ_BUILD_DIAGNOSTIC_LEGACY_ATOMICS", "0")
    ).strip().lower() in {"1", "true", "yes", "on"}
    if bucket_ids.device.type != "cuda" or legacy_cuda_atomics:
        sums = torch.bincount(bucket_ids, weights=weights, minlength=int(bucket_count))
        counts = torch.bincount(bucket_ids, minlength=int(bucket_count)).to(dtype=weights.dtype)
        return sums, counts

    # Weighted CUDA bincount uses floating-point atomics. PQ risk buckets are
    # page-local, so reduce one page-sized group at a time with a conflict-free
    # constant scatter followed by fp64 GEMM. This bounds the one-hot workspace
    # to roughly page_size * num_codes instead of tokens * all_page_codes.
    sums = torch.zeros((int(bucket_count),), dtype=weights.dtype, device=weights.device)
    counts = torch.zeros_like(sums)
    bins_per_group = max(1, int(bins_per_group))
    for group_start in range(0, int(bucket_count), bins_per_group):
        group_end = min(int(bucket_count), group_start + bins_per_group)
        group_rows = torch.nonzero(
            (bucket_ids >= group_start) & (bucket_ids < group_end),
            as_tuple=False,
        ).reshape(-1)
        if group_rows.numel() == 0:
            continue
        local_ids = bucket_ids.index_select(0, group_rows) - int(group_start)
        onehot = torch.zeros(
            (int(group_rows.numel()), group_end - group_start),
            dtype=weights.dtype,
            device=weights.device,
        )
        onehot.scatter_(1, local_ids.unsqueeze(1), 1.0)
        group_weights = weights.index_select(0, group_rows)
        sums[group_start:group_end] = torch.mm(
            onehot.transpose(0, 1),
            group_weights.unsqueeze(1),
        ).squeeze(1)
        counts[group_start:group_end] = onehot.sum(dim=0)
    return sums, counts


def value_vpq_code_stat_risk_torch(
    *,
    index: GPUIndex,
    values: torch.Tensor,
    vhat_all: torch.Tensor,
    residual_all: torch.Tensor | None = None,
    valid: torch.Tensor,
    page_ids: torch.Tensor,
    subbits: int,
    value_subvecs: int,
    value_subbits: int,
    value_bytes: int,
    kmeans_iters: int = 3,
) -> tuple[torch.Tensor, int]:
    """Per-token deployable V-PQ residual-risk sidecar using torch-built V-PQ.

    This mirrors the CPU reference's page/code mean residual statistic without
    invoking the CPU NumPy k-means sidecar path during HF benchmark decode.
    Invalid/non-indexed tokens use exact V fallback in `vhat_all`, so their
    residual risk is zero.
    """

    out = torch.zeros((int(values.shape[0]),), dtype=torch.float64, device=values.device)
    pack = value_vpq_pack_torch(
        index=index,
        values=values,
        value_subvecs=int(value_subvecs),
        value_subbits=int(value_subbits) if int(value_subbits) > 0 else int(subbits),
        key_bytes=int(value_bytes),
        device=values.device,
        kmeans_iters=int(kmeans_iters),
    )
    if pack is None or values.numel() == 0 or not bool(torch.any(valid)):
        actual_value_subbits = int(value_subbits) if int(value_subbits) > 0 else int(subbits)
        return out, int(actual_value_subbits)
    codebooks, codes, page_starts, page_size, actual_value_subbits = pack
    tokens = torch.arange(int(values.shape[0]), dtype=torch.long, device=values.device)
    valid_flat = valid.reshape(-1)
    valid_idx = torch.nonzero(valid_flat, as_tuple=False).reshape(-1)
    valid_pages = page_ids.reshape(-1).index_select(0, valid_idx).to(torch.long)
    valid_pages = torch.clamp(valid_pages, min=0, max=max(0, int(page_starts.numel()) - 1))
    rows = tokens.index_select(0, valid_idx) - page_starts.index_select(0, valid_pages)
    row_mask = (rows >= 0) & (rows < int(page_size))
    if not bool(torch.any(row_mask)):
        return out, int(actual_value_subbits)
    valid_idx = valid_idx.index_select(0, torch.nonzero(row_mask, as_tuple=False).reshape(-1))
    valid_pages = valid_pages.index_select(0, torch.nonzero(row_mask, as_tuple=False).reshape(-1))
    rows = rows.index_select(0, torch.nonzero(row_mask, as_tuple=False).reshape(-1)).to(torch.long)
    selected_codes = codes[valid_pages, rows].to(torch.long)
    if residual_all is None:
        residual_valid = (values.float() - vhat_all.float()).index_select(0, valid_idx).to(torch.float64)
    else:
        residual_valid = residual_all.index_select(0, valid_idx).to(torch.float64)
    subvecs = int(selected_codes.shape[1])
    subdim = int(codebooks.shape[-1])
    risk_valid = torch.zeros((int(valid_idx.numel()),), dtype=torch.float64, device=values.device)
    num_codes = 1 << int(actual_value_subbits)
    num_pages = int(page_starts.numel())
    bucket_count = int(max(1, num_pages * num_codes))
    for sub in range(subvecs):
        lo = int(sub) * subdim
        hi = lo + subdim
        per_token = torch.sum(residual_valid[:, lo:hi] * residual_valid[:, lo:hi], dim=1)
        bucket_ids = valid_pages * int(num_codes) + selected_codes[:, int(sub)]
        bucket_ids = torch.clamp(bucket_ids, min=0, max=bucket_count - 1)
        sums, counts = _bucket_sums_counts(
            bucket_ids,
            per_token,
            bucket_count=bucket_count,
            bins_per_group=num_codes,
        )
        means = sums / torch.clamp_min(counts, 1.0)
        risk_valid += means.index_select(0, bucket_ids)
    out.index_copy_(0, valid_idx, risk_valid)
    return out, int(actual_value_subbits)


def value_vpq_code_stat_risk_from_pack_streaming_torch(
    *,
    values: torch.Tensor,
    pack: tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int],
    context_len: int,
) -> tuple[torch.Tensor, int]:
    """Build deployable V-PQ residual-risk stats without full vhat/residual tensors.

    The statistic is page/code-local.  Streaming one page at a time preserves
    the same bucket means as `value_vpq_code_stat_risk_torch`, but avoids
    materializing `[context, dim]` reconstructed values and residuals.
    """

    codebooks, codes, page_starts, page_size, actual_value_subbits = pack
    context_len_i = max(0, min(int(context_len), int(values.shape[0])))
    out = torch.zeros((context_len_i,), dtype=torch.float64, device=values.device)
    if context_len_i == 0 or int(page_starts.numel()) == 0:
        return out, int(actual_value_subbits)
    subvecs = int(codebooks.shape[1])
    subdim = int(codebooks.shape[-1])
    dim = int(values.shape[-1])
    if subvecs <= 0 or subdim <= 0 or dim <= 0:
        return out, int(actual_value_subbits)
    if subvecs * subdim != dim:
        raise RuntimeError(
            "V-PQ streaming risk requires subvecs * subdim to match value dim"
        )
    num_codes = 1 << int(actual_value_subbits)
    pages = int(page_starts.numel())
    for page_i in range(pages):
        start_i = int(page_starts[page_i].item())
        if start_i >= context_len_i:
            continue
        end_i = min(context_len_i, start_i + int(page_size), int(values.shape[0]))
        rows_i = end_i - start_i
        if rows_i <= 0:
            continue
        values_page = values[start_i:end_i].float()
        codes_page = codes[page_i, :rows_i].to(torch.long)
        risk_page = torch.zeros((rows_i,), dtype=torch.float64, device=values.device)
        for sub in range(subvecs):
            lo = int(sub) * subdim
            hi = lo + subdim
            code_ids = torch.clamp(codes_page[:, int(sub)], min=0, max=num_codes - 1)
            approx = codebooks[page_i, int(sub)].index_select(0, code_ids).float()
            residual = values_page[:, lo:hi] - approx
            per_token = torch.sum(residual.double() * residual.double(), dim=1)
            sums, counts = _bucket_sums_counts(
                code_ids,
                per_token,
                bucket_count=num_codes,
                bins_per_group=num_codes,
            )
            means = sums / torch.clamp_min(counts, 1.0)
            risk_page += means.index_select(0, code_ids)
        out[start_i:end_i] = risk_page
    return out, int(actual_value_subbits)


def value_vpq_code_stat_risk_subset_torch(
    *,
    index: GPUIndex,
    values: torch.Tensor,
    tokens: torch.Tensor,
    residual_subset: torch.Tensor,
    valid: torch.Tensor,
    page_ids: torch.Tensor,
    subbits: int,
    value_subvecs: int,
    value_subbits: int,
    value_bytes: int,
) -> tuple[torch.Tensor, int]:
    """Per-token V-PQ residual-risk stats for a sealed page subset.

    The full risk statistic is page/code-local, so a newly sealed page can be
    refreshed without rereading/recomputing older pages.
    """

    out = torch.zeros((int(tokens.numel()),), dtype=torch.float64, device=values.device)
    pack = value_vpq_pack_torch(
        index=index,
        values=values,
        value_subvecs=int(value_subvecs),
        value_subbits=int(value_subbits) if int(value_subbits) > 0 else int(subbits),
        key_bytes=int(value_bytes),
        device=values.device,
    )
    if pack is None or tokens.numel() == 0 or not bool(torch.any(valid)):
        actual_value_subbits = int(value_subbits) if int(value_subbits) > 0 else int(subbits)
        return out.reshape(tokens.shape), int(actual_value_subbits)
    codebooks, codes, page_starts, page_size, actual_value_subbits = pack
    flat_tokens = tokens.reshape(-1)
    valid_flat = valid.reshape(-1)
    valid_idx = torch.nonzero(valid_flat, as_tuple=False).reshape(-1)
    valid_pages = page_ids.reshape(-1).index_select(0, valid_idx).to(torch.long)
    valid_pages = torch.clamp(valid_pages, min=0, max=max(0, int(page_starts.numel()) - 1))
    token_values = flat_tokens.index_select(0, valid_idx)
    rows = token_values - page_starts.index_select(0, valid_pages)
    row_mask = (rows >= 0) & (rows < int(page_size))
    if not bool(torch.any(row_mask)):
        return out.reshape(tokens.shape), int(actual_value_subbits)
    row_idx = torch.nonzero(row_mask, as_tuple=False).reshape(-1)
    valid_idx = valid_idx.index_select(0, row_idx)
    valid_pages = valid_pages.index_select(0, row_idx)
    rows = rows.index_select(0, row_idx).to(torch.long)
    selected_codes = codes[valid_pages, rows].to(torch.long)
    residual_valid = residual_subset.reshape(-1, int(values.shape[-1])).index_select(0, valid_idx).to(torch.float64)
    subvecs = int(selected_codes.shape[1])
    subdim = int(codebooks.shape[-1])
    risk_valid = torch.zeros((int(valid_idx.numel()),), dtype=torch.float64, device=values.device)
    num_codes = 1 << int(actual_value_subbits)
    num_pages = int(page_starts.numel())
    bucket_count = int(max(1, num_pages * num_codes))
    for sub in range(subvecs):
        lo = int(sub) * subdim
        hi = lo + subdim
        per_token = torch.sum(residual_valid[:, lo:hi] * residual_valid[:, lo:hi], dim=1)
        bucket_ids = valid_pages * int(num_codes) + selected_codes[:, int(sub)]
        bucket_ids = torch.clamp(bucket_ids, min=0, max=bucket_count - 1)
        sums, counts = _bucket_sums_counts(
            bucket_ids,
            per_token,
            bucket_count=bucket_count,
            bins_per_group=num_codes,
        )
        means = sums / torch.clamp_min(counts, 1.0)
        risk_valid += means.index_select(0, bucket_ids)
    out.index_copy_(0, valid_idx, risk_valid)
    return out.reshape(tokens.shape), int(actual_value_subbits)


def selected_value_exact_mask_gpu(
    *,
    selected_logits: torch.Tensor,
    rule: str,
    exact_top: int,
    exact_mass: float,
    min_top: int,
    max_top: int,
) -> torch.Tensor:
    heads, count = selected_logits.shape
    mask = torch.zeros((heads, count), dtype=torch.bool, device=selected_logits.device)
    if count == 0:
        return mask
    if str(rule) == "selector_rank":
        order = torch.arange(count, dtype=torch.long, device=selected_logits.device).reshape(1, count).expand(heads, -1)
        exact_counts = torch.full((heads,), max(0, min(count, int(exact_top))), dtype=torch.long, device=selected_logits.device)
    else:
        order = torch.argsort(selected_logits.float(), dim=1, descending=True, stable=True)
    if str(rule) == "fixed":
        exact_counts = torch.full((heads,), max(0, min(count, int(exact_top))), dtype=torch.long, device=selected_logits.device)
    elif str(rule) == "selector_rank":
        pass
    elif str(rule) == "selected_mass":
        probs = torch.softmax(selected_logits.float(), dim=1)
        cumulative = torch.cumsum(torch.gather(probs, 1, order), dim=1)
        target = max(0.0, min(1.0, float(exact_mass)))
        exact_counts = torch.sum(cumulative < target, dim=1).to(torch.long) + (1 if target > 0.0 else 0)
        exact_counts = torch.clamp(exact_counts, min=0, max=count)
    else:
        raise ValueError(f"GPU fast path does not support selected_value_exact_rule={rule}")
    if int(min_top) > 0:
        exact_counts = torch.maximum(
            exact_counts,
            torch.full_like(exact_counts, min(count, int(min_top))),
        )
    if int(max_top) > 0:
        exact_counts = torch.minimum(
            exact_counts,
            torch.full_like(exact_counts, min(count, int(max_top))),
        )
    ranks = torch.arange(count, dtype=torch.long, device=selected_logits.device).reshape(1, count)
    sorted_mask = ranks < exact_counts.reshape(heads, 1)
    return mask.scatter(1, order, sorted_mask)


def selected_value_exact_counts_from_mass_gpu(
    *,
    ranked_logits: torch.Tensor,
    ranked_scores: torch.Tensor,
    base_logsumexp: torch.Tensor | None,
    exact_mass: float,
    min_top: int,
    max_top: int,
) -> torch.Tensor:
    """Per row, count ranked tokens whose exact V should be kept.

    Static prefix/suffix and pending-page tokens are always exact in the native
    kernels. This count therefore applies only to the ranked dynamic tokens and
    chooses the smallest exact-logit prefix that reaches the requested mass
    inside the selected set.
    """

    if ranked_logits.dim() not in {2, 3}:
        raise ValueError(f"ranked_logits must be [heads, rank] or [positions, heads, rank], got {tuple(ranked_logits.shape)}")
    rank = int(ranked_logits.shape[-1])
    leading = ranked_logits.shape[:-1]
    if rank <= 0:
        return torch.zeros(leading, dtype=torch.long, device=ranked_logits.device)
    target = float(max(0.0, min(1.0, float(exact_mass))))
    if target <= 0.0:
        counts = torch.zeros(leading, dtype=torch.long, device=ranked_logits.device)
    else:
        valid = torch.isfinite(ranked_scores[..., :rank]) & torch.isfinite(ranked_logits[..., :rank])
        logits = torch.where(valid, ranked_logits[..., :rank].float(), torch.full_like(ranked_logits[..., :rank].float(), float("-inf")))
        sorted_logits, _ = torch.sort(logits, dim=-1, descending=True, stable=True)
        ranked_lse = torch.logsumexp(logits, dim=-1)
        if base_logsumexp is None:
            base_lse = torch.full_like(ranked_lse, float("-inf"))
        else:
            base_lse = base_logsumexp.float()
        total_lse = torch.logaddexp(base_lse, ranked_lse)
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
        hit = cum_mass >= min(float(target), 1.0 - 1.0e-7)
        has_hit = torch.any(hit, dim=-1)
        first_hit = torch.argmax(hit.to(torch.int32), dim=-1).to(torch.long) + 1
        counts = torch.where(
            base_mass >= float(target),
            torch.zeros_like(first_hit),
            torch.where(has_hit, first_hit, valid.sum(dim=-1).to(torch.long)),
        )
    if int(min_top) > 0:
        counts = torch.maximum(counts, torch.full_like(counts, min(rank, int(min_top))))
    if int(max_top) > 0:
        counts = torch.minimum(counts, torch.full_like(counts, min(rank, int(max_top))))
    return torch.clamp(counts, min=0, max=rank).to(torch.long)
