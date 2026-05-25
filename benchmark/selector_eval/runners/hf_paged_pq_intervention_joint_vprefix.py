#!/usr/bin/env python3
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import torch

from benchmark.selector_eval.gpu.run_gpu_paged_pq_eval import _sync_if_cuda, load_selector_paged_pq_ext
from benchmark.selector_eval.runners.hf_paged_pq_intervention_common import _env_truthy


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


@dataclass(frozen=True)
class JointVPrefixGridResult:
    grid_outputs: torch.Tensor | None
    grid_outputs_for_v_idx: Callable[[int], torch.Tensor] | None


def build_joint_vprefix_grid(runtime: JointVPrefixGridRuntime) -> JointVPrefixGridResult:
    prefix_delta_grid_t: torch.Tensor | None = None
    prefix_delta_by_count: dict[int, torch.Tensor] | None = None
    grid_outputs_t: torch.Tensor | None = None
    grid_outputs_for_v_idx: Callable[[int], torch.Tensor] | None = None
    joint_risk_wall_t0 = time.perf_counter() if runtime.wall_profile_enabled else 0.0
    if bool(getattr(runtime.args, "profile_native_ops", False)):
        _sync_if_cuda(runtime.device)
        joint_risk_t0 = time.perf_counter()
    else:
        joint_risk_t0 = 0.0

    if runtime.use_incremental_v_grid:
        grid_outputs_for_v_idx = _incremental_v_grid_accessor(runtime)
    elif int(runtime.max_exact_v_count) > 0:
        risk_grid_t = (
            (runtime.probs_grid * runtime.probs_grid)
            * runtime.code_error.to(dtype=runtime.prob_dtype).reshape(1, 1, -1)
        )
        if _env_truthy("SELECTOR_PQ_JOINT_NATIVE_RISK_PREFIX", "0"):
            native = load_selector_paged_pq_ext()
            grid_outputs_t = native.joint_vprefix_outputs_from_risk(
                runtime.base_output_grid.to(dtype=torch.float32).contiguous(),
                runtime.probs_grid.to(dtype=torch.float32).contiguous(),
                runtime.residual.to(dtype=torch.float32).contiguous(),
                runtime.code_error.to(dtype=torch.float32).contiguous(),
                runtime.joint_v_budgets_t,
            )
        elif _env_truthy("SELECTOR_PQ_JOINT_UNSORTED_V_PREFIX", "0"):
            grid_outputs_t = _unsorted_vprefix_outputs(runtime, risk_grid_t)
        else:
            grid_outputs_t, prefix_delta_grid_t, prefix_delta_by_count = _sorted_vprefix_outputs(
                runtime,
                risk_grid_t,
            )

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
