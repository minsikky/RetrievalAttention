#!/usr/bin/env python3
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable

import torch

from benchmark.selector_eval.gpu.run_gpu_paged_pq_eval import _sync_if_cuda, load_selector_paged_pq_ext
from benchmark.selector_eval.runners.hf_paged_pq_intervention_common import _env_int, _env_truthy


_compiled_risk_grid = None


def _risk_grid_eager(
    probs_grid: torch.Tensor,
    code_error: torch.Tensor,
    use_bf16: bool,
) -> torch.Tensor:
    dtype = torch.bfloat16 if use_bf16 else probs_grid.dtype
    probs_t = probs_grid.to(dtype=dtype)
    return (probs_t * probs_t) * code_error.to(dtype=dtype).reshape(1, 1, -1)


def _frozensim_risk_grid(runtime: "JointVPrefixGridRuntime") -> torch.Tensor:
    use_bf16 = _env_truthy("SELECTOR_PQ_JOINT_FROZENSIM_BF16_VPREFIX", "0")
    if (
        runtime.device.type == "cuda"
        and _env_truthy("SELECTOR_PQ_JOINT_FROZENSIM_COMPILE", "0")
    ):
        global _compiled_risk_grid
        if _compiled_risk_grid is None:
            _compiled_risk_grid = torch.compile(
                _risk_grid_eager,
                dynamic=True,
                fullgraph=True,
            )
        return _compiled_risk_grid(runtime.probs_grid, runtime.code_error, use_bf16)
    return _risk_grid_eager(runtime.probs_grid, runtime.code_error, use_bf16)


def precision_grid_layout_tensors(
    *,
    owner: object,
    device: torch.device,
    kind: str,
    counts: list[int] | tuple[int, ...],
    hi_frac: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cache immutable rung counts/hi cutoffs and rank positions on device."""

    counts_key = tuple(int(v) for v in counts)
    key = (
        str(kind),
        str(device),
        counts_key,
        float(hi_frac),
    )
    cache = getattr(owner, "_pagedpq_precision_grid_layout_cache", None)
    if not isinstance(cache, dict):
        cache = {}
        setattr(owner, "_pagedpq_precision_grid_layout_cache", cache)
    cached = cache.get(key)
    if cached is None:
        hi_counts = tuple(
            min(int(count), max(1, int(math.ceil(float(count) * float(hi_frac)))))
            if int(count) > 0
            else 0
            for count in counts_key
        )
        cached = (
            torch.as_tensor(counts_key, dtype=torch.long, device=device),
            torch.as_tensor(hi_counts, dtype=torch.long, device=device),
            torch.arange(max(counts_key, default=0), dtype=torch.long, device=device),
        )
        old_keys = [cached_key for cached_key in cache if cached_key[0] == str(kind)]
        for cached_key in old_keys:
            del cache[cached_key]
        cache[key] = cached
    return cached


def compose_precision_vprefix_outputs_batched(
    *,
    base_output_grid: torch.Tensor,
    cum_hi: torch.Tensor,
    cum_lo: torch.Tensor,
    commit_cum: torch.Tensor,
    exact_counts_t: torch.Tensor,
    hi_counts_t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project every V rung in one batched chain without new reductions.

    The cumsums are unchanged.  This only gathers their existing prefix cells
    and applies the same hi + (lo_exact - lo_hi) formula across a V dimension.
    """

    exact_indices_t = torch.clamp_min(exact_counts_t, 1) - 1
    hi_indices_t = torch.clamp_min(hi_counts_t, 1) - 1
    hi_delta_t = cum_hi.index_select(2, hi_indices_t).permute(0, 2, 1, 3)
    lo_exact_t = cum_lo.index_select(2, exact_indices_t).permute(0, 2, 1, 3)
    lo_hi_t = cum_lo.index_select(2, hi_indices_t).permute(0, 2, 1, 3)
    has_exact_t = (exact_counts_t > 0).reshape(1, -1, 1, 1)
    has_lo_t = (exact_counts_t > hi_counts_t).reshape(1, -1, 1, 1)
    # Preserve the eager per-rung arithmetic order exactly:
    # (hi_prefix + lo_exact_prefix) - lo_hi_prefix.
    hi_lo_delta_t = (hi_delta_t + lo_exact_t) - lo_hi_t
    delta_t = torch.where(has_lo_t, hi_lo_delta_t, hi_delta_t)
    delta_t = torch.where(has_exact_t, delta_t, 0.0)
    outputs_t = base_output_grid[:, None, :, :] + delta_t

    commit_exact_t = commit_cum.index_select(2, exact_indices_t).permute(0, 2, 1)
    commit_hi_t = commit_cum.index_select(2, hi_indices_t).permute(0, 2, 1)
    lo_reads_t = torch.where(
        has_lo_t.reshape(1, -1, 1),
        commit_exact_t - commit_hi_t,
        0,
    )
    return outputs_t, lo_reads_t


def precision_vprefix_k_chunk_size(
    *,
    k_count: int,
    heads: int,
    order_count: int,
    head_dim: int,
) -> int:
    """Bound the four live fp32 residual/product/cumsum planes.

    A 1600 MiB default keeps all six canonical K rows in the original batch
    shape at 32k, but selects one row at 128k.  The latter caps the dominant
    grid transient near 1 GiB instead of roughly 6 GiB.
    """

    budget_mb = _env_int("SELECTOR_PQ_JOINT_VPREFIX_TRANSIENT_BUDGET_MB", 1600)
    if budget_mb <= 0:
        raise ValueError("SELECTOR_PQ_JOINT_VPREFIX_TRANSIENT_BUDGET_MB must be positive")
    bytes_per_k = (
        4
        * int(heads)
        * int(order_count)
        * int(head_dim)
        * 4
    )
    if bytes_per_k <= 0:
        return max(1, int(k_count))
    budget_bytes = int(budget_mb) * 1024 * 1024
    return max(1, min(int(k_count), int(budget_bytes // bytes_per_k)))


@dataclass(frozen=True)
class JointVPrefixGridRuntime:
    args: object
    layer_id: int
    stats: dict
    device: torch.device
    wall_profile_enabled: bool
    use_incremental_v_grid: bool
    max_exact_v_count: int
    context_len: int
    k_count: int
    group_heads: int
    head_dim: int
    prob_dtype: torch.dtype
    probs_grid: torch.Tensor
    base_output_grid: torch.Tensor
    residual: torch.Tensor
    code_error: torch.Tensor
    joint_v_budgets: list[int]
    joint_v_budgets_t: torch.Tensor
    # Frozen precision tiers (spec OPEN-2/M6, --joint_kv_precision_tiers):
    # residual_lo_commit rows are int8-QDQ(V) - vhat where the per-token
    # commit test passes (int8_err < code_error) and ZERO elsewhere, so the
    # lo band folds into one extra cumsum. v_commit_mask is the raw commit
    # bitmap (for lo-read accounting). v_hi_frac is the frozen 0.1.
    residual_lo_commit: torch.Tensor | None = None
    v_commit_mask: torch.Tensor | None = None
    v_hi_frac: float = 1.0


@dataclass(frozen=True)
class JointVPrefixGridResult:
    grid_outputs: torch.Tensor | None
    grid_outputs_for_v_idx: Callable[[int], torch.Tensor] | None
    # (k_count, v_count, heads) committed lo-tier exact reads per budget
    # pair; only populated when precision tiers are active.
    v_lo_reads_grid: torch.Tensor | None = None


def build_joint_vprefix_grid(runtime: JointVPrefixGridRuntime) -> JointVPrefixGridResult:
    prefix_delta_grid_t: torch.Tensor | None = None
    prefix_delta_by_count: dict[int, torch.Tensor] | None = None
    grid_outputs_t: torch.Tensor | None = None
    grid_outputs_for_v_idx: Callable[[int], torch.Tensor] | None = None
    v_lo_reads_grid_t: torch.Tensor | None = None
    precision_tiers_active = runtime.residual_lo_commit is not None
    joint_risk_wall_t0 = time.perf_counter() if runtime.wall_profile_enabled else 0.0
    if bool(getattr(runtime.args, "profile_native_ops", False)):
        _sync_if_cuda(runtime.device)
        joint_risk_t0 = time.perf_counter()
    else:
        joint_risk_t0 = 0.0

    if precision_tiers_active and (
        runtime.use_incremental_v_grid
        or _env_truthy("SELECTOR_PQ_JOINT_NATIVE_RISK_PREFIX", "0")
        or _env_truthy("SELECTOR_PQ_JOINT_UNSORTED_V_PREFIX", "0")
    ):
        raise RuntimeError(
            "joint_kv_precision_tiers requires the sorted torch V-prefix path "
            "(disable SELECTOR_PQ_JOINT_INCREMENTAL_V_GRID / NATIVE_RISK_PREFIX / "
            "UNSORTED_V_PREFIX)"
        )
    if runtime.use_incremental_v_grid:
        grid_outputs_for_v_idx = _incremental_v_grid_accessor(runtime)
    elif int(runtime.max_exact_v_count) > 0:
        fused_precision_risk = bool(
            precision_tiers_active
            and _env_truthy("SELECTOR_PQ_JOINT_FROZENSIM_FUSED_RISK_SORT", "0")
        )
        risk_grid_t = None if fused_precision_risk else _frozensim_risk_grid(runtime)
        if _env_truthy("SELECTOR_PQ_JOINT_NATIVE_RISK_PREFIX", "0"):
            if risk_grid_t is None:
                raise RuntimeError("native risk-prefix path is incompatible with fused frozensim risk sort")
            native = load_selector_paged_pq_ext()
            grid_outputs_t = native.joint_vprefix_outputs_from_risk(
                runtime.base_output_grid.to(dtype=torch.float32).contiguous(),
                runtime.probs_grid.to(dtype=torch.float32).contiguous(),
                runtime.residual.to(dtype=torch.float32).contiguous(),
                runtime.code_error.to(dtype=torch.float32).contiguous(),
                runtime.joint_v_budgets_t,
            )
        elif _env_truthy("SELECTOR_PQ_JOINT_UNSORTED_V_PREFIX", "0"):
            if risk_grid_t is None:
                raise RuntimeError("unsorted V-prefix path is incompatible with fused frozensim risk sort")
            grid_outputs_t = _unsorted_vprefix_outputs(runtime, risk_grid_t)
        elif precision_tiers_active:
            grid_outputs_t, v_lo_reads_grid_t = _sorted_vprefix_outputs_precision_tiers(
                runtime,
                risk_grid_t,
            )
        else:
            grid_outputs_t, prefix_delta_grid_t, prefix_delta_by_count = _sorted_vprefix_outputs(
                runtime,
                risk_grid_t,
            )
    elif precision_tiers_active:
        # No V budget reads exact values, so every grid row is the base
        # output and there is nothing for the tiers to split.
        pass

    if grid_outputs_t is None and not runtime.use_incremental_v_grid:
        grid_outputs_by_v: list[torch.Tensor] = []
        for v_budget in runtime.joint_v_budgets:
            exact_count = max(0, min(int(v_budget), runtime.context_len))
            if exact_count > 0 and prefix_delta_by_count is not None:
                grid_outputs_by_v.append(
                    runtime.base_output_grid
                    + prefix_delta_by_count[
                        max(0, min(int(exact_count), int(runtime.max_exact_v_count)))
                    ]
                )
            elif exact_count > 0 and prefix_delta_grid_t is not None:
                grid_outputs_by_v.append(
                    runtime.base_output_grid + prefix_delta_grid_t[:, :, int(exact_count) - 1, :]
                )
            else:
                grid_outputs_by_v.append(runtime.base_output_grid)
        grid_outputs_t = torch.stack(grid_outputs_by_v, dim=1)

    if bool(getattr(runtime.args, "profile_native_ops", False)):
        _sync_if_cuda(runtime.device)
        runtime.stats[runtime.layer_id].add_joint_detail_timing(
            risk_prefix_seconds=float(time.perf_counter() - joint_risk_t0)
        )
    if runtime.wall_profile_enabled and int(runtime.max_exact_v_count) > 0:
        runtime.stats[runtime.layer_id].add_joint_wall_timing(
            risk_prefix_seconds=float(time.perf_counter() - joint_risk_wall_t0)
        )
    return JointVPrefixGridResult(
        grid_outputs=grid_outputs_t,
        grid_outputs_for_v_idx=grid_outputs_for_v_idx,
        v_lo_reads_grid=v_lo_reads_grid_t,
    )

def _incremental_v_grid_accessor(runtime: JointVPrefixGridRuntime) -> Callable[[int], torch.Tensor]:
    incremental_grid_outputs_by_v_idx: dict[int, torch.Tensor] = {}
    risk_grid_incremental_t = (
        (runtime.probs_grid * runtime.probs_grid)
        * runtime.code_error.to(dtype=runtime.prob_dtype).reshape(1, 1, -1)
        if int(runtime.max_exact_v_count) > 0
        else None
    )

    def grid_outputs_for_v_idx_fn(vi_i: int) -> torch.Tensor:
        cached_v = incremental_grid_outputs_by_v_idx.get(int(vi_i))
        if cached_v is not None:
            return cached_v
        exact_count = max(
            0,
            min(
                int(runtime.joint_v_budgets[int(vi_i)]),
                runtime.context_len,
                int(runtime.max_exact_v_count),
            ),
        )
        if exact_count <= 0:
            out_v = runtime.base_output_grid
        elif exact_count >= runtime.context_len:
            delta_t = (
                runtime.probs_grid.to(torch.float32).reshape(
                    runtime.k_count * runtime.group_heads,
                    runtime.context_len,
                )
                @ runtime.residual.float()
            ).reshape(runtime.k_count, runtime.group_heads, int(runtime.head_dim))
            out_v = runtime.base_output_grid + delta_t
        else:
            if risk_grid_incremental_t is None:
                raise RuntimeError("missing residual-risk grid for incremental V-grid")
            exact_order_local_t = torch.topk(
                risk_grid_incremental_t,
                k=int(exact_count),
                dim=2,
                largest=True,
                sorted=True,
            ).indices
            gathered_probs_local_t = torch.gather(
                runtime.probs_grid.to(torch.float32),
                2,
                exact_order_local_t,
            )
            gathered_residual_local_t = runtime.residual.index_select(
                0,
                exact_order_local_t.reshape(-1),
            ).reshape(
                runtime.k_count,
                runtime.group_heads,
                int(exact_order_local_t.shape[2]),
                int(runtime.head_dim),
            )
            delta_t = torch.sum(
                gathered_probs_local_t.reshape(runtime.k_count, runtime.group_heads, -1, 1)
                * gathered_residual_local_t.float(),
                dim=2,
            )
            out_v = runtime.base_output_grid + delta_t
        incremental_grid_outputs_by_v_idx[int(vi_i)] = out_v
        return out_v

    return grid_outputs_for_v_idx_fn


def _unsorted_vprefix_outputs(
    runtime: JointVPrefixGridRuntime,
    risk_grid_t: torch.Tensor,
) -> torch.Tensor:
    grid_outputs_by_v = []
    for v_budget in runtime.joint_v_budgets:
        exact_count = max(0, min(int(v_budget), runtime.context_len, int(runtime.max_exact_v_count)))
        if exact_count <= 0:
            grid_outputs_by_v.append(runtime.base_output_grid)
        elif exact_count >= runtime.context_len:
            delta_t = (
                runtime.probs_grid.to(torch.float32).reshape(
                    runtime.k_count * runtime.group_heads,
                    runtime.context_len,
                )
                @ runtime.residual.float()
            ).reshape(runtime.k_count, runtime.group_heads, int(runtime.head_dim))
            grid_outputs_by_v.append(runtime.base_output_grid + delta_t)
        else:
            exact_order_i_t = torch.topk(
                risk_grid_t,
                k=int(exact_count),
                dim=2,
                largest=True,
                sorted=False,
            ).indices
            gathered_probs_i_t = torch.gather(
                runtime.probs_grid.to(torch.float32),
                2,
                exact_order_i_t,
            )
            gathered_residual_i_t = runtime.residual.index_select(
                0,
                exact_order_i_t.reshape(-1),
            ).reshape(
                runtime.k_count,
                runtime.group_heads,
                int(exact_count),
                int(runtime.head_dim),
            )
            delta_t = torch.sum(
                gathered_probs_i_t.reshape(runtime.k_count, runtime.group_heads, int(exact_count), 1)
                * gathered_residual_i_t.float(),
                dim=2,
            )
            grid_outputs_by_v.append(runtime.base_output_grid + delta_t)
    return torch.stack(grid_outputs_by_v, dim=1)


def _sorted_vprefix_outputs(
    runtime: JointVPrefixGridRuntime,
    risk_grid_t: torch.Tensor,
) -> tuple[torch.Tensor | None, torch.Tensor | None, dict[int, torch.Tensor] | None]:
    if int(runtime.max_exact_v_count) >= runtime.context_len:
        exact_order_grid_t = torch.argsort(risk_grid_t, dim=2, descending=True, stable=True)
    else:
        exact_order_grid_t = torch.topk(
            risk_grid_t,
            k=int(runtime.max_exact_v_count),
            dim=2,
            largest=True,
            sorted=True,
        ).indices
    if _env_truthy("SELECTOR_PQ_JOINT_NATIVE_V_PREFIX", "0"):
        native = load_selector_paged_pq_ext()
        grid_outputs_t = native.joint_vprefix_outputs(
            runtime.base_output_grid.to(dtype=torch.float32).contiguous(),
            runtime.probs_grid.to(dtype=torch.float32).contiguous(),
            runtime.residual.to(dtype=torch.float32).contiguous(),
            exact_order_grid_t.to(dtype=torch.long).contiguous(),
            runtime.joint_v_budgets_t,
        )
        return grid_outputs_t, None, None

    gathered_probs_grid_t = torch.gather(
        runtime.probs_grid.to(torch.float32),
        2,
        exact_order_grid_t,
    )
    gathered_residual_grid_t = runtime.residual.index_select(0, exact_order_grid_t.reshape(-1)).reshape(
        runtime.k_count,
        runtime.group_heads,
        int(exact_order_grid_t.shape[2]),
        int(runtime.head_dim),
    )
    weighted_residual_grid_t = (
        gathered_probs_grid_t.reshape(runtime.k_count, runtime.group_heads, -1, 1)
        * gathered_residual_grid_t.float()
    )
    if _env_truthy("SELECTOR_PQ_JOINT_SEGMENTED_V_PREFIX", "0"):
        prefix_delta_by_count: dict[int, torch.Tensor] = {}
        running_delta_t = torch.zeros_like(runtime.base_output_grid, dtype=torch.float32)
        prev_count = 0
        exact_counts_sorted = sorted(
            {
                max(0, min(int(v_budget), runtime.context_len, int(runtime.max_exact_v_count)))
                for v_budget in runtime.joint_v_budgets
                if max(0, min(int(v_budget), runtime.context_len, int(runtime.max_exact_v_count))) > 0
            }
        )
        for exact_count in exact_counts_sorted:
            if int(exact_count) > int(prev_count):
                running_delta_t = running_delta_t + torch.sum(
                    weighted_residual_grid_t[:, :, int(prev_count): int(exact_count), :],
                    dim=2,
                )
            prefix_delta_by_count[int(exact_count)] = running_delta_t.clone()
            prev_count = int(exact_count)
        return None, None, prefix_delta_by_count
    return None, torch.cumsum(weighted_residual_grid_t, dim=2), None


def _sorted_vprefix_outputs_precision_tiers(
    runtime: JointVPrefixGridRuntime,
    risk_grid_t: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Frozen progressive-precision V composition (spec OPEN-2/M6): the
    top ceil(v_hi_frac * c) of the risk-ranked exact set reads full
    precision (residual), the remainder reads the int8 plane only where
    the per-token commit test passed (residual_lo_commit rows, pre-zeroed
    on failed commits so dropped reads keep the V-PQ value). Mirrors the
    v_hi_mask/v_lo_mask split + output_from_base_and_split_masks in
    run_joint_kv_budget_policy_eval.py."""
    exact_counts = [
        max(0, min(int(v_budget), runtime.context_len, int(runtime.max_exact_v_count)))
        for v_budget in runtime.joint_v_budgets
    ]
    exact_counts_t, hi_counts_t, _positions_t = precision_grid_layout_tensors(
        owner=runtime.args,
        device=runtime.device,
        kind="v",
        counts=exact_counts,
        hi_frac=float(runtime.v_hi_frac),
    )
    fused_vprefix = _env_truthy("SELECTOR_PQ_JOINT_FROZENSIM_FUSED_VPREFIX", "0")
    fused_risk_sort = _env_truthy("SELECTOR_PQ_JOINT_FROZENSIM_FUSED_RISK_SORT", "0")
    bf16_vprefix = _env_truthy("SELECTOR_PQ_JOINT_FROZENSIM_BF16_VPREFIX", "0")
    if fused_risk_sort:
        if not fused_vprefix:
            raise RuntimeError(
                "SELECTOR_PQ_JOINT_FROZENSIM_FUSED_RISK_SORT requires "
                "SELECTOR_PQ_JOINT_FROZENSIM_FUSED_VPREFIX=1"
            )
        if bf16_vprefix:
            raise RuntimeError("fused frozensim risk sort does not support bf16 V-prefix inputs")
        native = load_selector_paged_pq_ext()
        if not hasattr(native, "joint_vprefix_outputs_precision_from_risk"):
            raise RuntimeError("fused frozensim risk sort requires an updated CUDA extension")
        return native.joint_vprefix_outputs_precision_from_risk(
            runtime.base_output_grid.to(dtype=torch.float32).contiguous(),
            runtime.probs_grid.to(dtype=torch.float32).contiguous(),
            runtime.residual.to(dtype=torch.float32).contiguous(),
            runtime.residual_lo_commit.to(dtype=torch.float32).contiguous(),
            runtime.v_commit_mask.to(dtype=torch.bool).contiguous(),
            runtime.code_error.to(dtype=torch.float32).contiguous(),
            runtime.joint_v_budgets_t,
            hi_counts_t,
        )
    if risk_grid_t is None:
        raise RuntimeError("missing risk grid for sorted precision V-prefix")
    if int(runtime.max_exact_v_count) >= runtime.context_len:
        exact_order_grid_t = torch.argsort(risk_grid_t, dim=2, descending=True, stable=True)
    else:
        exact_order_grid_t = torch.topk(
            risk_grid_t,
            k=int(runtime.max_exact_v_count),
            dim=2,
            largest=True,
            sorted=True,
        ).indices
    if fused_vprefix:
        if bf16_vprefix:
            raise RuntimeError("fused frozensim V-prefix does not support bf16 inputs")
        native = load_selector_paged_pq_ext()
        if not hasattr(native, "joint_vprefix_outputs_precision"):
            raise RuntimeError("fused frozensim V-prefix requires an updated CUDA extension")
        return native.joint_vprefix_outputs_precision(
            runtime.base_output_grid.to(dtype=torch.float32).contiguous(),
            runtime.probs_grid.to(dtype=torch.float32).contiguous(),
            runtime.residual.to(dtype=torch.float32).contiguous(),
            runtime.residual_lo_commit.to(dtype=torch.float32).contiguous(),
            runtime.v_commit_mask.to(dtype=torch.bool).contiguous(),
            exact_order_grid_t.to(dtype=torch.long).contiguous(),
            runtime.joint_v_budgets_t,
            hi_counts_t,
        )
    order_count_i = int(exact_order_grid_t.shape[2])
    prefix_dtype = torch.bfloat16 if bf16_vprefix else torch.float32
    commit_mask_i32_t = runtime.v_commit_mask.to(dtype=torch.int32)

    def torch_prefix_chunk(k_start: int, k_end: int):
        chunk_k = int(k_end - k_start)
        order_chunk_t = exact_order_grid_t[int(k_start): int(k_end)]
        gathered_probs_t = torch.gather(
            runtime.probs_grid[int(k_start): int(k_end)].to(prefix_dtype),
            2,
            order_chunk_t,
        )
        order_flat_t = order_chunk_t.reshape(-1)
        gathered_residual_t = runtime.residual.index_select(0, order_flat_t).reshape(
            chunk_k,
            runtime.group_heads,
            order_count_i,
            int(runtime.head_dim),
        )
        probs_col_t = gathered_probs_t.reshape(chunk_k, runtime.group_heads, -1, 1)
        hi_product_t = probs_col_t * gathered_residual_t.to(prefix_dtype)
        cum_hi_t = torch.cumsum(hi_product_t, dim=2, dtype=torch.float32)
        del gathered_residual_t, hi_product_t
        gathered_lo_t = runtime.residual_lo_commit.index_select(0, order_flat_t).reshape(
            chunk_k,
            runtime.group_heads,
            order_count_i,
            int(runtime.head_dim),
        )
        lo_product_t = probs_col_t * gathered_lo_t.to(prefix_dtype)
        cum_lo_t = torch.cumsum(lo_product_t, dim=2, dtype=torch.float32)
        del gathered_lo_t, lo_product_t
        commit_cum_t = torch.cumsum(
            commit_mask_i32_t.index_select(0, order_flat_t).reshape(
                chunk_k,
                runtime.group_heads,
                order_count_i,
            ),
            dim=2,
        )
        return cum_hi_t, cum_lo_t, commit_cum_t

    if runtime.device.type == "cuda":
        k_chunk = precision_vprefix_k_chunk_size(
            k_count=int(runtime.k_count),
            heads=int(runtime.group_heads),
            order_count=int(order_count_i),
            head_dim=int(runtime.head_dim),
        )
        output_chunks: list[torch.Tensor] = []
        read_chunks: list[torch.Tensor] = []
        for k_start in range(0, int(runtime.k_count), int(k_chunk)):
            k_end = min(int(runtime.k_count), int(k_start) + int(k_chunk))
            cum_hi_t, cum_lo_t, commit_cum_t = torch_prefix_chunk(k_start, k_end)
            output_chunk_t, read_chunk_t = compose_precision_vprefix_outputs_batched(
                base_output_grid=runtime.base_output_grid[int(k_start): int(k_end)],
                cum_hi=cum_hi_t,
                cum_lo=cum_lo_t,
                commit_cum=commit_cum_t,
                exact_counts_t=exact_counts_t,
                hi_counts_t=hi_counts_t,
            )
            output_chunks.append(output_chunk_t)
            read_chunks.append(read_chunk_t)
            del cum_hi_t, cum_lo_t, commit_cum_t
        if len(output_chunks) == 1:
            return output_chunks[0], read_chunks[0]
        return torch.cat(output_chunks, dim=0), torch.cat(read_chunks, dim=0)

    cum_hi_t, cum_lo_t, commit_cum_t = torch_prefix_chunk(0, int(runtime.k_count))

    # Keep the blessed CPU reference operation sequence unchanged.
    grid_outputs_by_v: list[torch.Tensor] = []
    v_lo_reads_by_v: list[torch.Tensor] = []
    zero_reads_t = torch.zeros(
        (runtime.k_count, runtime.group_heads),
        dtype=torch.int32,
        device=runtime.device,
    )
    for exact_count in exact_counts:
        if exact_count <= 0:
            grid_outputs_by_v.append(runtime.base_output_grid)
            v_lo_reads_by_v.append(zero_reads_t)
            continue
        hi_count = min(
            int(exact_count),
            max(1, int(math.ceil(float(exact_count) * float(runtime.v_hi_frac)))),
        )
        delta_t = cum_hi_t[:, :, int(hi_count) - 1, :]
        lo_reads_t = zero_reads_t
        if int(exact_count) > int(hi_count):
            delta_t = (
                delta_t
                + cum_lo_t[:, :, int(exact_count) - 1, :]
                - cum_lo_t[:, :, int(hi_count) - 1, :]
            )
            lo_reads_t = (
                commit_cum_t[:, :, int(exact_count) - 1]
                - commit_cum_t[:, :, int(hi_count) - 1]
            )
        grid_outputs_by_v.append(runtime.base_output_grid + delta_t)
        v_lo_reads_by_v.append(lo_reads_t)
    grid_outputs_t = torch.stack(grid_outputs_by_v, dim=1)
    v_lo_reads_grid_t = torch.stack(v_lo_reads_by_v, dim=1)
    return grid_outputs_t, v_lo_reads_grid_t
