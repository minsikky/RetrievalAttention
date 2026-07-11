#!/usr/bin/env python3
from __future__ import annotations

import math
import os
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
    _scaled_budget_delta_threshold,
)


_compiled_adjacent_rel_l2 = None


def _joint_draft_mode() -> str | None:
    """Draft-then-verify one-shot selection mode.

    SELECTOR_PQ_JOINT_DRAFT_MODE={off,start1,start2}. Default off leaves the
    escalation walk untouched (byte-identical to the frozen algorithm). When
    ``start1``/``start2`` is set, the controller skips the stability walk
    entirely and pins the K-budget rung to (proxy-mass start rung + 1 / + 2),
    clamped to the ladder, then applies the frozen v_target rule to that rung.
    """
    mode = str(os.environ.get("SELECTOR_PQ_JOINT_DRAFT_MODE", "off")).strip().lower()
    if mode in {"", "off", "0", "false", "none"}:
        return None
    if mode in {"start1", "start2"}:
        return mode
    raise ValueError(
        f"unknown SELECTOR_PQ_JOINT_DRAFT_MODE={mode!r}; expected off, start1, or start2"
    )


def _draft_budget_index_at_least(budgets, target: float) -> int:
    """First ladder index whose budget is >= target (mirrors the frozen rule)."""
    for idx, budget in enumerate(budgets):
        if int(budget) >= float(target):
            return int(idx)
    return max(0, len(budgets) - 1)


def _adjacent_rel_l2_eager(cur_t: torch.Tensor, next_t: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(cur_t - next_t, dim=-1) / torch.clamp_min(
        torch.linalg.vector_norm(next_t, dim=-1),
        1e-20,
    )


def _adjacent_rel_l2(cur_t: torch.Tensor, next_t: torch.Tensor) -> torch.Tensor:
    if (
        cur_t.device.type == "cuda"
        and _env_truthy("SELECTOR_PQ_JOINT_FROZENSIM_COMPILE", "0")
    ):
        global _compiled_adjacent_rel_l2
        if _compiled_adjacent_rel_l2 is None:
            _compiled_adjacent_rel_l2 = torch.compile(
                _adjacent_rel_l2_eager,
                dynamic=True,
                fullgraph=True,
            )
        return _compiled_adjacent_rel_l2(cur_t, next_t)
    return _adjacent_rel_l2_eager(cur_t, next_t)


@dataclass
class JointPolicyResult:
    final_ki_by_head: list[int]
    final_vi_by_head: list[int]
    final_idx_for_output: torch.Tensor | None = None
    final_output_grid: torch.Tensor | None = None
    v_lo_reads_rows: list[list[list[int]]] | None = None
    deferred_torch_policy: object | None = None


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
    context_len: int
    threshold_mode: str
    threshold_reference_frac: float
    threshold_scale_shape: str
    threshold_min_scale: float
    threshold_max_scale: float
    start_ki_by_head: list[int] | torch.Tensor | None
    start_vi_by_head: list[int] | torch.Tensor | None
    use_incremental_v_grid: bool
    grid_outputs: torch.Tensor | None
    grid_outputs_for_v_idx: Callable[[int], torch.Tensor] | None
    output_for_budget: Callable[[int, int], torch.Tensor]
    k_artifacts: Callable[[int], tuple[torch.Tensor, float, torch.Tensor, torch.Tensor | None]]
    grid_k_mb_by_idx: list[float] | None
    v_mb_by_idx: list[float] | None
    sim_start_seconds: float
    v_lo_reads_grid: torch.Tensor | None = None
    time_trace: object | None = None
    defer_torch_policy: bool = False
    d2h_owner: object | None = None
    d2h_slot: int = 0


@dataclass
class PreparedTorchPolicy:
    runtime: JointPolicyRuntime
    packed_host: torch.Tensor | np.ndarray
    k_delta_shape: tuple[int, ...]
    v_delta_shape: tuple[int, ...]
    tensor_starts_k: bool
    tensor_starts_v: bool
    v_lo_shape: tuple[int, ...] | None


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
    v_lo_reads_rows: list[list[list[int]]] | None = None
    deferred_torch_policy: PreparedTorchPolicy | None = None

    draft_mode = _joint_draft_mode()
    if draft_mode is not None:
        # Draft-then-verify: skip the escalation walk entirely. Pin each head's
        # K rung to (proxy-mass start rung + bump), clamped to the ladder, then
        # apply the frozen v_target rule (v = max(v0, 0.25*k_target)) to that
        # rung. finalize_joint_head_outputs reconstructs the fused output for
        # these one-shot budgets via the same grid the walk would have used.
        bump = 1 if draft_mode == "start1" else 2
        k_budgets = runtime.active_k_budgets
        v_budgets = runtime.v_budgets
        n_k = len(k_budgets)
        n_v = len(v_budgets)
        for local_head_i in range(runtime.group_heads):
            start_ki, _start_vi = _head_start_indices(runtime, local_head_i)
            ki = min(int(start_ki) + int(bump), max(0, n_k - 1))
            k_target = float(k_budgets[int(ki)])
            v_target = max(float(v_budgets[0]), k_target * 0.25)
            vi = _draft_budget_index_at_least(v_budgets, v_target)
            vi = min(max(0, int(vi)), max(0, n_v - 1))
            final_ki_by_head.append(int(ki))
            final_vi_by_head.append(int(vi))
        return JointPolicyResult(
            final_ki_by_head=final_ki_by_head,
            final_vi_by_head=final_vi_by_head,
            final_idx_for_output=None,
            final_output_grid=None,
            v_lo_reads_rows=None,
            deferred_torch_policy=None,
        )

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
        if (
            _env_truthy("SELECTOR_PQ_JOINT_NATIVE_POLICY", "0")
            and not _requires_torch_policy(runtime)
        ):
            final_idx_for_output, final_output_grid = _select_with_native_policy(runtime, output_grid)
            if not bool(getattr(runtime.args, "disable_cost_stats", False)):
                final_rows = final_idx_for_output.detach().cpu().tolist()
                for row in final_rows:
                    final_ki_by_head.append(int(row[0]))
                    final_vi_by_head.append(int(row[1]))
        else:
            if runtime.defer_torch_policy:
                deferred_torch_policy = _prepare_torch_policy(
                    runtime,
                    output_grid,
                    non_blocking=True,
                )
            else:
                selected, v_lo_reads_rows = _select_with_torch_policy(runtime, output_grid)
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
        v_lo_reads_rows=v_lo_reads_rows,
        deferred_torch_policy=deferred_torch_policy,
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


def _head_start_indices(
    runtime: JointPolicyRuntime,
    local_head_i: int,
    *,
    start_ki_by_head: list[int] | None = None,
    start_vi_by_head: list[int] | None = None,
) -> tuple[int, int]:
    ki = 0
    vi = 0
    starts_k = runtime.start_ki_by_head if start_ki_by_head is None else start_ki_by_head
    starts_v = runtime.start_vi_by_head if start_vi_by_head is None else start_vi_by_head
    if starts_k is not None and int(local_head_i) < len(starts_k):
        ki = int(starts_k[int(local_head_i)])
    if starts_v is not None and int(local_head_i) < len(starts_v):
        vi = int(starts_v[int(local_head_i)])
    ki = min(max(0, int(ki)), max(0, len(runtime.active_k_budgets) - 1))
    vi = min(max(0, int(vi)), max(0, len(runtime.v_budgets) - 1))
    return ki, vi


def _requires_torch_policy(runtime: JointPolicyRuntime) -> bool:
    if bool(getattr(runtime.args, "joint_kv_deescalate", False)):
        # The native policy kernel is escalation-only; the frozen-spec
        # down-walk lives in the torch policy.
        return True
    if str(runtime.threshold_mode) != "fixed":
        return True
    for values in (runtime.start_ki_by_head, runtime.start_vi_by_head):
        if values is not None and any(int(v) != 0 for v in values):
            return True
    return False


def _thresholds_for_step(
    runtime: JointPolicyRuntime,
    *,
    ki: int,
    vi: int,
    k_can: bool,
    v_can: bool,
) -> tuple[float, float]:
    if str(runtime.threshold_mode) == "fixed":
        return float(runtime.threshold), float(runtime.threshold)
    if str(runtime.threshold_mode) != "budget_delta_frac":
        raise ValueError(f"unknown joint_kv_threshold_mode: {runtime.threshold_mode}")
    context = max(float(runtime.context_len), 1.0)
    k_threshold = float(runtime.threshold)
    v_threshold = float(runtime.threshold)
    if k_can:
        k_frac = (
            float(max(0, int(runtime.active_k_budgets[int(ki) + 1]) - int(runtime.active_k_budgets[int(ki)])))
            / context
        )
        k_threshold = _scaled_budget_delta_threshold(
            base_threshold=float(runtime.threshold),
            budget_delta_frac=float(k_frac),
            reference_frac=float(runtime.threshold_reference_frac),
            shape=str(runtime.threshold_scale_shape),
            min_scale=float(runtime.threshold_min_scale),
            max_scale=float(runtime.threshold_max_scale),
        )
    if v_can:
        v_frac = (
            float(max(0, int(runtime.v_budgets[int(vi) + 1]) - int(runtime.v_budgets[int(vi)])))
            / context
        )
        v_threshold = _scaled_budget_delta_threshold(
            base_threshold=float(runtime.threshold),
            budget_delta_frac=float(v_frac),
            reference_frac=float(runtime.threshold_reference_frac),
            shape=str(runtime.threshold_scale_shape),
            min_scale=float(runtime.threshold_min_scale),
            max_scale=float(runtime.threshold_max_scale),
        )
    return float(k_threshold), float(v_threshold)


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
    k_threshold, v_threshold = _thresholds_for_step(
        runtime,
        ki=int(ki),
        vi=int(vi),
        k_can=bool(k_can),
        v_can=bool(v_can),
    )
    return _choose_joint_kv_action(
        policy=runtime.policy_name,
        k_delta=float(k_delta),
        v_delta=float(v_delta),
        k_can=bool(k_can),
        v_can=bool(v_can),
        threshold=float(runtime.threshold),
        k_threshold=float(k_threshold),
        v_threshold=float(v_threshold),
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
    ki, vi = _head_start_indices(runtime, local_head_i)
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
    if bool(getattr(runtime.args, "joint_kv_deescalate", False)):
        ki, vi = _deescalate_walk(runtime, local_head_i, ki, vi)
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


def _prepare_torch_policy(
    runtime: JointPolicyRuntime,
    output_grid: torch.Tensor,
    *,
    non_blocking: bool,
) -> PreparedTorchPolicy:
    output_grid64 = output_grid.to(dtype=torch.float64)
    if len(runtime.active_k_budgets) > 1:
        k_cur = output_grid64[:-1]
        k_next = output_grid64[1:]
        k_delta_t = _adjacent_rel_l2(k_cur, k_next)
    else:
        k_delta_t = torch.empty(
            (0, len(runtime.v_budgets), runtime.group_heads),
            dtype=torch.float64,
            device=runtime.device,
        )
    if len(runtime.v_budgets) > 1:
        v_cur = output_grid64[:, :-1]
        v_next = output_grid64[:, 1:]
        v_delta_t = _adjacent_rel_l2(v_cur, v_next)
    else:
        v_delta_t = torch.empty(
            (len(runtime.active_k_budgets), 0, runtime.group_heads),
            dtype=torch.float64,
            device=runtime.device,
        )

    # Start indices, policy deltas, and precision-tier accounting were four
    # separate blocking D2H reads in the standard path.  Pack their already
    # computed values into one float64 transfer; all integer fields are small
    # enough to be represented exactly.
    packed_parts = [k_delta_t.reshape(-1), v_delta_t.reshape(-1)]
    tensor_starts_k = isinstance(runtime.start_ki_by_head, torch.Tensor)
    tensor_starts_v = isinstance(runtime.start_vi_by_head, torch.Tensor)
    if tensor_starts_k:
        packed_parts.append(runtime.start_ki_by_head.to(dtype=torch.float64).reshape(-1))
    if tensor_starts_v:
        packed_parts.append(runtime.start_vi_by_head.to(dtype=torch.float64).reshape(-1))
    if runtime.v_lo_reads_grid is not None:
        packed_parts.append(runtime.v_lo_reads_grid.to(dtype=torch.float64).reshape(-1))
    packed_t = torch.cat(packed_parts).detach()
    if non_blocking and packed_t.device.type == "cuda":
        if runtime.d2h_owner is None:
            raise RuntimeError("deferred Torch policy requires a D2H buffer owner")
        cache = getattr(runtime.d2h_owner, "_pagedpq_policy_d2h_buffer_cache", None)
        if not isinstance(cache, dict):
            cache = {}
            setattr(runtime.d2h_owner, "_pagedpq_policy_d2h_buffer_cache", cache)
        cache_key = (int(runtime.d2h_slot), int(packed_t.numel()), str(packed_t.dtype))
        packed_host = cache.get(cache_key)
        if packed_host is None:
            packed_host = torch.empty(
                (int(packed_t.numel()),),
                dtype=packed_t.dtype,
                device="cpu",
                pin_memory=True,
            )
            cache[cache_key] = packed_host
        packed_host.copy_(packed_t, non_blocking=True)
    else:
        transfer_t0 = time.perf_counter() if runtime.time_trace is not None else 0.0
        packed_host = packed_t.cpu().numpy()
        if runtime.time_trace is not None:
            runtime.time_trace.add_cpu("sync_wait", time.perf_counter() - transfer_t0)
    return PreparedTorchPolicy(
        runtime=runtime,
        packed_host=packed_host,
        k_delta_shape=tuple(k_delta_t.shape),
        v_delta_shape=tuple(v_delta_t.shape),
        tensor_starts_k=bool(tensor_starts_k),
        tensor_starts_v=bool(tensor_starts_v),
        v_lo_shape=(tuple(runtime.v_lo_reads_grid.shape) if runtime.v_lo_reads_grid is not None else None),
    )


def prepare_batched_torch_policies(
    runtimes: list[JointPolicyRuntime],
    output_grids: torch.Tensor,
    v_lo_reads_grids: torch.Tensor,
) -> list[PreparedTorchPolicy]:
    """Prepare all KV-head policies with one set of tensor launches and D2H.

    Relative-L2 reductions still operate independently over the final head
    dimension, so batching only adds a leading group dimension and preserves
    each row's reduction order.
    """

    if not runtimes:
        return []
    groups = len(runtimes)
    if int(output_grids.shape[0]) != groups or int(v_lo_reads_grids.shape[0]) != groups:
        raise RuntimeError("batched policy group count mismatch")
    first = runtimes[0]
    k_count = len(first.active_k_budgets)
    v_count = len(first.v_budgets)
    heads = int(first.group_heads)
    for runtime in runtimes:
        if (
            len(runtime.active_k_budgets) != k_count
            or len(runtime.v_budgets) != v_count
            or int(runtime.group_heads) != heads
        ):
            raise RuntimeError("batched policy requires uniform grid shapes")

    output64 = output_grids.to(dtype=torch.float64)
    if k_count > 1:
        k_delta = _adjacent_rel_l2(output64[:, :-1], output64[:, 1:])
    else:
        k_delta = torch.empty(
            (groups, 0, v_count, heads), dtype=torch.float64, device=output_grids.device
        )
    if v_count > 1:
        v_delta = _adjacent_rel_l2(output64[:, :, :-1], output64[:, :, 1:])
    else:
        v_delta = torch.empty(
            (groups, k_count, 0, heads), dtype=torch.float64, device=output_grids.device
        )

    tensor_starts_k = all(isinstance(runtime.start_ki_by_head, torch.Tensor) for runtime in runtimes)
    tensor_starts_v = all(isinstance(runtime.start_vi_by_head, torch.Tensor) for runtime in runtimes)
    if tensor_starts_k != any(isinstance(runtime.start_ki_by_head, torch.Tensor) for runtime in runtimes):
        raise RuntimeError("batched policy requires uniform K-start representation")
    if tensor_starts_v != any(isinstance(runtime.start_vi_by_head, torch.Tensor) for runtime in runtimes):
        raise RuntimeError("batched policy requires uniform V-start representation")

    parts = [k_delta.reshape(groups, -1), v_delta.reshape(groups, -1)]
    if tensor_starts_k:
        parts.append(
            torch.stack([runtime.start_ki_by_head for runtime in runtimes]).to(torch.float64)
        )
    if tensor_starts_v:
        parts.append(
            torch.stack([runtime.start_vi_by_head for runtime in runtimes]).to(torch.float64)
        )
    parts.append(v_lo_reads_grids.to(dtype=torch.float64).reshape(groups, -1))
    packed = torch.cat(parts, dim=1).detach()
    owner = first.d2h_owner
    if owner is None:
        raise RuntimeError("batched deferred Torch policy requires a D2H buffer owner")
    cache = getattr(owner, "_pagedpq_policy_d2h_buffer_cache", None)
    if not isinstance(cache, dict):
        cache = {}
        setattr(owner, "_pagedpq_policy_d2h_buffer_cache", cache)
    cache_key = ("batched", tuple(packed.shape), str(packed.dtype))
    packed_host = cache.get(cache_key)
    if packed_host is None:
        packed_host = torch.empty_like(packed, device="cpu", pin_memory=True)
        cache[cache_key] = packed_host
    packed_host.copy_(packed, non_blocking=True)

    prepared = []
    for group, runtime in enumerate(runtimes):
        runtime.grid_outputs = output_grids[group]
        runtime.v_lo_reads_grid = v_lo_reads_grids[group]
        prepared.append(
            PreparedTorchPolicy(
                runtime=runtime,
                packed_host=packed_host[group],
                k_delta_shape=tuple(k_delta.shape[1:]),
                v_delta_shape=tuple(v_delta.shape[1:]),
                tensor_starts_k=bool(tensor_starts_k),
                tensor_starts_v=bool(tensor_starts_v),
                v_lo_shape=tuple(v_lo_reads_grids.shape[1:]),
            )
        )
    return prepared


def finish_prepared_torch_policy(
    prepared: PreparedTorchPolicy,
) -> tuple[list[tuple[int, int]], list[list[list[int]]] | None]:
    runtime = prepared.runtime
    packed_np = (
        prepared.packed_host.numpy()
        if isinstance(prepared.packed_host, torch.Tensor)
        else prepared.packed_host
    )
    offset = 0
    k_size = int(math.prod(prepared.k_delta_shape))
    k_delta_np = packed_np[offset: offset + k_size].reshape(prepared.k_delta_shape)
    offset += k_size
    v_size = int(math.prod(prepared.v_delta_shape))
    v_delta_np = packed_np[offset: offset + v_size].reshape(prepared.v_delta_shape)
    offset += v_size
    starts_k_cpu = runtime.start_ki_by_head if not prepared.tensor_starts_k else None
    starts_v_cpu = runtime.start_vi_by_head if not prepared.tensor_starts_v else None
    if prepared.tensor_starts_k:
        starts_k_cpu = [int(v) for v in packed_np[offset: offset + runtime.group_heads]]
        offset += runtime.group_heads
    if prepared.tensor_starts_v:
        starts_v_cpu = [int(v) for v in packed_np[offset: offset + runtime.group_heads]]
        offset += runtime.group_heads
    v_lo_reads_rows = None
    if prepared.v_lo_shape is not None:
        v_lo_size = int(math.prod(prepared.v_lo_shape))
        v_lo_reads_rows = (
            packed_np[offset: offset + v_lo_size]
            .reshape(prepared.v_lo_shape)
            .astype(np.int64, copy=False)
            .tolist()
        )
    k_mb_by_idx = _k_mb_by_index(runtime)
    selected: list[tuple[int, int]] = []
    for local_head_i in range(runtime.group_heads):
        ki, vi = _head_start_indices(
            runtime,
            local_head_i,
            start_ki_by_head=starts_k_cpu,
            start_vi_by_head=starts_v_cpu,
        )
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
        if bool(getattr(runtime.args, "joint_kv_deescalate", False)):
            # Frozen-spec down-walk (spec §4 step 4) on the materialized
            # grid: k_delta_np[ki-1, vi] IS relL2(out[ki-1,vi], out[ki,vi])
            # with the richer-rung denominator, so the down-probe reuses
            # the escalation delta arrays at zero extra GPU work. K probe
            # then V probe each round (V reads the post-K-move ki),
            # mirroring _deescalate_walk / simulate_policy(deescalate).
            while True:
                moved = False
                if int(ki) > 0:
                    d = float(k_delta_np[int(ki) - 1, int(vi), int(local_head_i)])
                    thr_k = _band_threshold_between(
                        runtime,
                        runtime.active_k_budgets[int(ki) - 1],
                        runtime.active_k_budgets[int(ki)],
                    )
                    if d <= thr_k:
                        ki -= 1
                        moved = True
                if int(vi) > 0:
                    d = float(v_delta_np[int(ki), int(vi) - 1, int(local_head_i)])
                    thr_v = _band_threshold_between(
                        runtime,
                        runtime.v_budgets[int(vi) - 1],
                        runtime.v_budgets[int(vi)],
                    )
                    if d <= thr_v:
                        vi -= 1
                        moved = True
                if not moved:
                    break
        selected.append((int(ki), int(vi)))
    return selected, v_lo_reads_rows


def _select_with_torch_policy(
    runtime: JointPolicyRuntime,
    output_grid: torch.Tensor,
) -> tuple[list[tuple[int, int]], list[list[list[int]]] | None]:
    return finish_prepared_torch_policy(
        _prepare_torch_policy(runtime, output_grid, non_blocking=False)
    )


def _band_threshold_between(runtime: JointPolicyRuntime, lo_budget: int, hi_budget: int) -> float:
    if str(runtime.threshold_mode) != "budget_delta_frac":
        return float(runtime.threshold)
    frac = float(max(0, int(hi_budget) - int(lo_budget))) / max(float(runtime.context_len), 1.0)
    return _scaled_budget_delta_threshold(
        base_threshold=float(runtime.threshold),
        budget_delta_frac=float(frac),
        reference_frac=float(runtime.threshold_reference_frac),
        shape=str(runtime.threshold_scale_shape),
        min_scale=float(runtime.threshold_min_scale),
        max_scale=float(runtime.threshold_max_scale),
    )


def _deescalate_walk(runtime: JointPolicyRuntime, local_head_i: int, ki: int, vi: int) -> tuple[int, int]:
    """Frozen-spec down-walk (spec §4 step 4): after the escalation stop,
    step DOWN any axis whose adjacent-band delta is within its scaled
    threshold. Mirrors simulate_policy(deescalate=True) in
    run_joint_kv_budget_policy_eval.py — K probe then V probe each round,
    same band-threshold formula in both directions (no oscillation)."""
    while True:
        moved = False
        if int(ki) > 0:
            d = float(
                _rel_l2_torch(
                    runtime.output_for_budget(int(ki) - 1, int(vi))[int(local_head_i)],
                    runtime.output_for_budget(int(ki), int(vi))[int(local_head_i)],
                )
            )
            thr_k = _band_threshold_between(
                runtime, runtime.active_k_budgets[int(ki) - 1], runtime.active_k_budgets[int(ki)]
            )
            if d <= thr_k:
                ki -= 1
                moved = True
        if int(vi) > 0:
            d = float(
                _rel_l2_torch(
                    runtime.output_for_budget(int(ki), int(vi) - 1)[int(local_head_i)],
                    runtime.output_for_budget(int(ki), int(vi))[int(local_head_i)],
                )
            )
            thr_v = _band_threshold_between(
                runtime, runtime.v_budgets[int(vi) - 1], runtime.v_budgets[int(vi)]
            )
            if d <= thr_v:
                vi -= 1
                moved = True
        if not moved:
            break
    return int(ki), int(vi)


def _walk_lazy_outputs(runtime: JointPolicyRuntime, local_head_i: int) -> tuple[int, int]:
    ki, vi = _head_start_indices(runtime, local_head_i)
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
    if bool(getattr(runtime.args, "joint_kv_deescalate", False)):
        ki, vi = _deescalate_walk(runtime, local_head_i, ki, vi)
    return ki, vi
