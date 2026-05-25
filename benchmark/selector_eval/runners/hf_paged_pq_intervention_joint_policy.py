#!/usr/bin/env python3
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from benchmark.selector_eval.gpu.run_gpu_paged_pq_eval import _sync_if_cuda, load_selector_paged_pq_ext
from benchmark.selector_eval.runners.hf_paged_pq_intervention_common import (
    _choose_joint_kv_action,
    _env_truthy,
    _rel_l2_torch,
)


@dataclass
class JointPolicyResult:
    final_ki_by_head: list[int]
    final_vi_by_head: list[int]
    final_idx_for_output: torch.Tensor | None = None
    final_output_grid: torch.Tensor | None = None


@dataclass
class JointPolicyRuntime:
    args: object
    layer_id: int
    stats: dict
    device: torch.device
    wall_profile_enabled: bool
    group_heads: int
    active_k_budgets: list[int]
    v_budgets: list[int]
    policy_name: str
    policy_id: int
    policy_uses_mb: bool
    threshold: float
    use_incremental_v_grid: bool
    grid_outputs: torch.Tensor | None
    grid_outputs_for_v_idx: Callable[[int], torch.Tensor] | None
    output_for_budget: Callable[[int, int], torch.Tensor]
    k_artifacts: Callable[[int], tuple[torch.Tensor, float, torch.Tensor, torch.Tensor | None]]
    grid_k_mb_by_idx: list[float] | None
    v_mb_by_idx: list[float] | None
    sim_start_seconds: float


def select_joint_kv_budgets(runtime: JointPolicyRuntime) -> JointPolicyResult:
    joint_policy_wall_t0 = time.perf_counter() if runtime.wall_profile_enabled else 0.0
    if bool(getattr(runtime.args, "profile_native_ops", False)):
        _sync_if_cuda(runtime.device)
        joint_policy_t0 = time.perf_counter()
    else:
        joint_policy_t0 = 0.0

    final_ki_by_head: list[int] = []
    final_vi_by_head: list[int] = []
    final_idx_for_output: torch.Tensor | None = None
    final_output_grid: torch.Tensor | None = None

    if runtime.use_incremental_v_grid and runtime.grid_outputs_for_v_idx is not None:
        k_mb_by_idx = _k_mb_by_index(runtime)
        for local_head_i in range(runtime.group_heads):
            ki, vi = _walk_incremental_grid(runtime, local_head_i, k_mb_by_idx)
            final_ki_by_head.append(int(ki))
            final_vi_by_head.append(int(vi))
    elif (
        _env_truthy("SELECTOR_PQ_JOINT_VECTOR_POLICY", "1")
        and not _env_truthy("SELECTOR_PQ_JOINT_NATIVE_LAZY_POLICY", "0")
    ):
        output_grid = _materialize_output_grid(runtime)
        if _env_truthy("SELECTOR_PQ_JOINT_NATIVE_POLICY", "0"):
            final_idx_for_output, final_output_grid = _select_with_native_policy(runtime, output_grid)
            if not bool(getattr(runtime.args, "disable_cost_stats", False)):
                final_rows = final_idx_for_output.detach().cpu().tolist()
                for row in final_rows:
                    final_ki_by_head.append(int(row[0]))
                    final_vi_by_head.append(int(row[1]))
        else:
            selected = _select_with_torch_policy(runtime, output_grid)
            final_ki_by_head.extend(int(ki) for ki, _vi in selected)
            final_vi_by_head.extend(int(vi) for _ki, vi in selected)
    else:
        for local_head_i in range(runtime.group_heads):
            ki, vi = _walk_lazy_outputs(runtime, local_head_i)
            final_ki_by_head.append(int(ki))
            final_vi_by_head.append(int(vi))

    if bool(getattr(runtime.args, "profile_native_ops", False)):
        _sync_if_cuda(runtime.device)
        runtime.stats[runtime.layer_id].add_joint_detail_timing(
            policy_seconds=float(time.perf_counter() - joint_policy_t0)
        )
    if runtime.wall_profile_enabled:
        runtime.stats[runtime.layer_id].add_joint_wall_timing(
            policy_seconds=float(time.perf_counter() - joint_policy_wall_t0)
        )
    if bool(getattr(runtime.args, "profile_native_ops", False)):
        _sync_if_cuda(runtime.device)
        runtime.stats[runtime.layer_id].add_native_detail_timing(
            geometric_seconds=float(time.perf_counter() - runtime.sim_start_seconds)
        )

    return JointPolicyResult(
        final_ki_by_head=final_ki_by_head,
        final_vi_by_head=final_vi_by_head,
        final_idx_for_output=final_idx_for_output,
        final_output_grid=final_output_grid,
    )


def _k_mb_by_index(runtime: JointPolicyRuntime) -> list[float]:
    if runtime.grid_k_mb_by_idx is not None:
        return runtime.grid_k_mb_by_idx
    if runtime.policy_uses_mb:
        return [
            float(runtime.k_artifacts(int(ki_i))[1])
            for ki_i in range(len(runtime.active_k_budgets))
        ]
    return [0.0 for _ in runtime.active_k_budgets]


def _extra_v_mb(runtime: JointPolicyRuntime, vi: int, v_can: bool) -> float:
    if not v_can:
        return float("inf")
    if runtime.v_mb_by_idx is None:
        return 0.0
    return float(runtime.v_mb_by_idx[int(vi) + 1] - runtime.v_mb_by_idx[int(vi)])


def _next_action(
    runtime: JointPolicyRuntime,
    *,
    ki: int,
    vi: int,
    k_delta: float,
    v_delta: float,
    k_can: bool,
    v_can: bool,
    turn: int,
    k_mb_by_idx: list[float] | None = None,
) -> str:
    extra_k_mb = (
        float(k_mb_by_idx[int(ki) + 1] - k_mb_by_idx[int(ki)])
        if k_can and k_mb_by_idx is not None
        else float(runtime.k_artifacts(int(ki) + 1)[1] - runtime.k_artifacts(int(ki))[1])
        if k_can
        else float("inf")
    )
    return _choose_joint_kv_action(
        policy=runtime.policy_name,
        k_delta=float(k_delta),
        v_delta=float(v_delta),
        k_can=bool(k_can),
        v_can=bool(v_can),
        threshold=float(runtime.threshold),
        turn=int(turn),
        extra_k_mb=float(extra_k_mb),
        extra_v_mb=_extra_v_mb(runtime, vi, v_can),
    )


def _walk_incremental_grid(
    runtime: JointPolicyRuntime,
    local_head_i: int,
    k_mb_by_idx: list[float],
) -> tuple[int, int]:
    if runtime.grid_outputs_for_v_idx is None:
        raise RuntimeError("missing incremental V-grid output accessor")
    ki = 0
    vi = 0
    steps = 0
    max_steps = len(runtime.active_k_budgets) + len(runtime.v_budgets) + 4
    while steps < max_steps:
        cur_output = runtime.grid_outputs_for_v_idx(int(vi))[int(ki), int(local_head_i)]
        k_can = int(ki) + 1 < len(runtime.active_k_budgets)
        v_can = int(vi) + 1 < len(runtime.v_budgets)
        k_delta = (
            _rel_l2_torch(
                cur_output,
                runtime.grid_outputs_for_v_idx(int(vi))[int(ki) + 1, int(local_head_i)],
            )
            if k_can
            else 0.0
        )
        v_delta = (
            _rel_l2_torch(
                cur_output,
                runtime.grid_outputs_for_v_idx(int(vi) + 1)[int(ki), int(local_head_i)],
            )
            if v_can
            else 0.0
        )
        action = _next_action(
            runtime,
            ki=ki,
            vi=vi,
            k_delta=float(k_delta),
            v_delta=float(v_delta),
            k_can=k_can,
            v_can=v_can,
            turn=steps,
            k_mb_by_idx=k_mb_by_idx,
        )
        if action == "stop":
            break
        if action == "k":
            ki += 1
        elif action == "v":
            vi += 1
        else:
            raise AssertionError(action)
        steps += 1
    return ki, vi


def _materialize_output_grid(runtime: JointPolicyRuntime) -> torch.Tensor:
    if runtime.grid_outputs is not None:
        return runtime.grid_outputs
    return torch.stack(
        [
            torch.stack(
                [
                    runtime.output_for_budget(int(ki_i), int(vi_i))
                    for vi_i in range(len(runtime.v_budgets))
                ],
                dim=0,
            )
            for ki_i in range(len(runtime.active_k_budgets))
        ],
        dim=0,
    )


def _select_with_native_policy(
    runtime: JointPolicyRuntime,
    output_grid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    native = load_selector_paged_pq_ext()
    if runtime.policy_uses_mb:
        k_mb_t = torch.as_tensor(_k_mb_by_index(runtime), dtype=torch.float32, device=runtime.device)
        if runtime.v_mb_by_idx is None:
            raise RuntimeError("MB-aware joint policy requires V-budget MB accounting")
        v_mb_t = torch.as_tensor(runtime.v_mb_by_idx, dtype=torch.float32, device=runtime.device)
    else:
        k_mb_t = torch.empty((len(runtime.active_k_budgets),), dtype=torch.float32, device=runtime.device)
        v_mb_t = torch.empty((len(runtime.v_budgets),), dtype=torch.float32, device=runtime.device)
    final_idx = native.joint_select_policy(
        output_grid.to(dtype=torch.float32).contiguous(),
        k_mb_t,
        v_mb_t,
        float(runtime.threshold),
        runtime.policy_id,
    )
    if bool(getattr(runtime.args, "disable_cost_stats", False)):
        return final_idx, output_grid
    return final_idx, None


def _select_with_torch_policy(
    runtime: JointPolicyRuntime,
    output_grid: torch.Tensor,
) -> list[tuple[int, int]]:
    output_grid64 = output_grid.to(dtype=torch.float64)
    if len(runtime.active_k_budgets) > 1:
        k_cur = output_grid64[:-1]
        k_next = output_grid64[1:]
        k_delta_np = (
            torch.linalg.vector_norm(k_cur - k_next, dim=-1)
            / torch.clamp_min(torch.linalg.vector_norm(k_next, dim=-1), 1e-20)
        ).detach().cpu().numpy()
    else:
        k_delta_np = np.empty((0, len(runtime.v_budgets), runtime.group_heads), dtype=np.float64)
    if len(runtime.v_budgets) > 1:
        v_cur = output_grid64[:, :-1]
        v_next = output_grid64[:, 1:]
        v_delta_np = (
            torch.linalg.vector_norm(v_cur - v_next, dim=-1)
            / torch.clamp_min(torch.linalg.vector_norm(v_next, dim=-1), 1e-20)
        ).detach().cpu().numpy()
    else:
        v_delta_np = np.empty((len(runtime.active_k_budgets), 0, runtime.group_heads), dtype=np.float64)
    k_mb_by_idx = _k_mb_by_index(runtime)
    selected: list[tuple[int, int]] = []
    for local_head_i in range(runtime.group_heads):
        ki = 0
        vi = 0
        steps = 0
        max_steps = len(runtime.active_k_budgets) + len(runtime.v_budgets) + 4
        while steps < max_steps:
            k_can = int(ki) + 1 < len(runtime.active_k_budgets)
            v_can = int(vi) + 1 < len(runtime.v_budgets)
            k_delta = float(k_delta_np[int(ki), int(vi), int(local_head_i)]) if k_can else 0.0
            v_delta = float(v_delta_np[int(ki), int(vi), int(local_head_i)]) if v_can else 0.0
            action = _next_action(
                runtime,
                ki=ki,
                vi=vi,
                k_delta=k_delta,
                v_delta=v_delta,
                k_can=k_can,
                v_can=v_can,
                turn=steps,
                k_mb_by_idx=k_mb_by_idx,
            )
            if action == "stop":
                break
            if action == "k":
                ki += 1
            elif action == "v":
                vi += 1
            else:
                raise AssertionError(action)
            steps += 1
        selected.append((int(ki), int(vi)))
    return selected


def _walk_lazy_outputs(runtime: JointPolicyRuntime, local_head_i: int) -> tuple[int, int]:
    ki = 0
    vi = 0
    steps = 0
    max_steps = len(runtime.active_k_budgets) + len(runtime.v_budgets) + 4
    while steps < max_steps:
        cur_output = runtime.output_for_budget(int(ki), int(vi))[int(local_head_i)]
        k_can = int(ki) + 1 < len(runtime.active_k_budgets)
        v_can = int(vi) + 1 < len(runtime.v_budgets)
        k_delta = (
            _rel_l2_torch(
                cur_output,
                runtime.output_for_budget(int(ki) + 1, int(vi))[int(local_head_i)],
            )
            if k_can
            else 0.0
        )
        v_delta = (
            _rel_l2_torch(
                cur_output,
                runtime.output_for_budget(int(ki), int(vi) + 1)[int(local_head_i)],
            )
            if v_can
            else 0.0
        )
        action = _next_action(
            runtime,
            ki=ki,
            vi=vi,
            k_delta=float(k_delta),
            v_delta=float(v_delta),
            k_can=k_can,
            v_can=v_can,
            turn=steps,
        )
        if action == "stop":
            break
        if action == "k":
            ki += 1
        elif action == "v":
            vi += 1
        else:
            raise AssertionError(action)
        steps += 1
    return ki, vi
