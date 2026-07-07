#!/usr/bin/env python3
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch

from benchmark.selector_eval.runners.hf_paged_pq_intervention_common import MB


@dataclass
class JointFinalizeRuntime:
    args: object
    module: object
    layer_id: int
    stats: dict
    device: torch.device
    outputs_all: torch.Tensor
    head_start: int
    head_end: int
    group_heads: int
    context_len: int
    key_bytes: int
    value_bytes: int
    selector_mb: float
    actual_value_subvecs: int
    code_bytes: int
    v_pq_codebook_mb: float
    metadata_mb: float
    joint_v_budgets: list[int]
    final_ki_by_head: list[int]
    final_vi_by_head: list[int]
    final_idx_for_output: torch.Tensor | None
    final_output_grid: torch.Tensor | None
    grid_outputs: torch.Tensor | None
    grid_outputs_for_v_idx: Callable[[int], torch.Tensor] | None
    grid_selected_counts_by_ki: list[int] | None
    grid_selected_by_ki: list[torch.Tensor | None] | None
    k_artifacts: Callable[[int], tuple[torch.Tensor, float, torch.Tensor, torch.Tensor | None]]
    output_for_budget: Callable[[int, int], torch.Tensor]
    # Frozen precision tiers (--joint_kv_precision_tiers): lo-tier reads
    # are precision_lo_bytes wide, dropped V lo reads keep the V-PQ value
    # (counted with the compressed tail instead of the exact set).
    precision_tiers_enabled: bool = False
    precision_v_hi_frac: float = 1.0
    precision_lo_bytes: int = 1
    k_lo_counts_by_ki: list[int] | None = None
    v_lo_reads_grid: torch.Tensor | None = None


def finalize_joint_head_outputs(runtime: JointFinalizeRuntime) -> bool:
    if (
        runtime.final_idx_for_output is not None
        and runtime.final_output_grid is not None
        and bool(getattr(runtime.args, "disable_cost_stats", False))
    ):
        head_idx = torch.arange(runtime.group_heads, dtype=torch.long, device=runtime.device)
        final_idx = runtime.final_idx_for_output.to(device=runtime.device, dtype=torch.long)
        runtime.outputs_all[runtime.head_start: runtime.head_end] = runtime.final_output_grid[
            final_idx[:, 0],
            final_idx[:, 1],
            head_idx,
        ]
        return True

    v_lo_reads_rows: list[list[list[int]]] | None = None
    if (
        runtime.precision_tiers_enabled
        and runtime.v_lo_reads_grid is not None
        and not bool(getattr(runtime.args, "disable_cost_stats", False))
    ):
        # One transfer for the whole group instead of a per-head sync.
        v_lo_reads_rows = runtime.v_lo_reads_grid.detach().cpu().tolist()
    for local_head_i, (ki, vi) in enumerate(
        zip(runtime.final_ki_by_head, runtime.final_vi_by_head, strict=True)
    ):
        global_head_i = int(runtime.head_start + local_head_i)
        if not bool(getattr(runtime.args, "disable_cost_stats", False)):
            selected_count = _selected_count(runtime, int(ki), int(local_head_i))
            exact_v_count = max(0, min(int(runtime.joint_v_budgets[int(vi)]), runtime.context_len))
            exact_key_mb = (
                float(selected_count * int(runtime.module.head_dim) * runtime.key_bytes) / MB
            )
            if (
                runtime.precision_tiers_enabled
                and runtime.k_lo_counts_by_ki is not None
                and int(ki) < len(runtime.k_lo_counts_by_ki)
            ):
                # Frozen K lo tier: ranked-prefix tokens beyond the hi
                # fraction read precision_lo_bytes per element.
                exact_key_mb -= (
                    float(
                        int(runtime.k_lo_counts_by_ki[int(ki)])
                        * int(runtime.module.head_dim)
                        * (runtime.key_bytes - runtime.precision_lo_bytes)
                    )
                    / MB
                )
            if v_lo_reads_rows is not None and exact_v_count > 0:
                # Frozen V tiers: hi reads full precision, committed lo
                # reads are precision_lo_bytes wide, dropped lo candidates
                # keep the V-PQ value (counted with the compressed tail).
                v_hi_count = min(
                    int(exact_v_count),
                    int(math.ceil(float(exact_v_count) * float(runtime.precision_v_hi_frac))),
                )
                v_lo_count = int(v_lo_reads_rows[int(ki)][int(vi)][int(local_head_i)])
                effective_v_reads = int(v_hi_count + v_lo_count)
                exact_v_mb = (
                    float(effective_v_reads * int(runtime.module.head_dim) * runtime.value_bytes)
                    / MB
                    - float(
                        v_lo_count
                        * int(runtime.module.head_dim)
                        * (runtime.value_bytes - runtime.precision_lo_bytes)
                    )
                    / MB
                )
                exact_v_count = effective_v_reads
            else:
                exact_v_mb = (
                    float(exact_v_count * int(runtime.module.head_dim) * runtime.value_bytes) / MB
                )
            compressed_v_codes_mb = (
                float(
                    max(0, runtime.context_len - exact_v_count)
                    * runtime.actual_value_subvecs
                    * runtime.code_bytes
                )
                / MB
            )
            tail_mb_override = float(
                runtime.v_pq_codebook_mb
                + compressed_v_codes_mb
                + runtime.metadata_mb
            )
            dense_physical_key_mb = (
                float(runtime.context_len * int(runtime.module.head_dim) * runtime.key_bytes) / MB
            )
            runtime.stats[runtime.layer_id].add_count(
                int(selected_count),
                max(0, runtime.context_len - int(exact_v_count)),
                float(runtime.selector_mb),
                int(runtime.module.head_dim),
                runtime.key_bytes,
                runtime.value_bytes,
                tail_mb_override=tail_mb_override,
                exact_kv_mb_override=float(exact_key_mb + exact_v_mb),
                confidence_mb_override=0.0,
                physical_gpu_exact_kv_mb_override=float(dense_physical_key_mb + exact_v_mb),
                physical_gpu_confidence_mb_override=0.0,
            )
        runtime.outputs_all[global_head_i] = _output_for_choice(
            runtime,
            int(ki),
            int(vi),
            int(local_head_i),
        )
    return True


def _selected_count(runtime: JointFinalizeRuntime, ki: int, local_head_i: int) -> int:
    if runtime.grid_selected_counts_by_ki is not None:
        return int(runtime.grid_selected_counts_by_ki[int(ki)])
    if runtime.grid_selected_by_ki is not None:
        selected = runtime.grid_selected_by_ki[int(ki)]
        if selected is not None:
            return int(selected[int(local_head_i)].numel())
    return int(runtime.k_artifacts(int(ki))[0][int(local_head_i)].numel())


def _output_for_choice(
    runtime: JointFinalizeRuntime,
    ki: int,
    vi: int,
    local_head_i: int,
) -> torch.Tensor:
    if runtime.grid_outputs is not None:
        return runtime.grid_outputs[int(ki), int(vi), int(local_head_i)]
    if runtime.grid_outputs_for_v_idx is not None:
        return runtime.grid_outputs_for_v_idx(int(vi))[int(ki), int(local_head_i)]
    return runtime.output_for_budget(int(ki), int(vi))[int(local_head_i)]
