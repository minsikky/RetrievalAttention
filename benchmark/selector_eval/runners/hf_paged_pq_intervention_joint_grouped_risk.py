#!/usr/bin/env python3
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import torch

from benchmark.selector_eval.gpu.run_gpu_paged_pq_eval import _sync_if_cuda, load_selector_paged_pq_ext
from benchmark.selector_eval.runners.hf_paged_pq_intervention_common import MB, _env_truthy


@dataclass(frozen=True)
class JointGroupedRiskRuntime:
    args: Any
    self: Any
    layer_id: int
    stats: dict
    device: torch.device
    wall_profile_enabled: bool
    grouped_risk_records: list[dict[str, object]]
    grouped_strided_output_workspace_enabled: bool
    grouped_vpq_vhat_groups_t: torch.Tensor | None
    grouped_vpq_residual_groups_t: torch.Tensor | None
    grouped_vpq_code_error_groups_t: torch.Tensor | None
    grouped_risk_prefix_workspace_for: Callable
    grouped_score_direct_workspace_for: Callable
    joint_v_budgets: list[int]
    joint_v_budgets_t: torch.Tensor
    key_bytes: int
    value_bytes: int
    policy_id: int
    policy_uses_mb: bool
    threshold_value: float
    outputs_all: torch.Tensor
    head_group_runtime: Any


def process_grouped_risk_records(runtime: JointGroupedRiskRuntime) -> None:
    args = runtime.args
    self = runtime.self
    layer_id = runtime.layer_id
    stats = runtime.stats
    device = runtime.device
    wall_profile_enabled = runtime.wall_profile_enabled
    grouped_risk_records = runtime.grouped_risk_records
    grouped_strided_output_workspace_enabled = runtime.grouped_strided_output_workspace_enabled
    grouped_vpq_vhat_groups_t = runtime.grouped_vpq_vhat_groups_t
    grouped_vpq_residual_groups_t = runtime.grouped_vpq_residual_groups_t
    grouped_vpq_code_error_groups_t = runtime.grouped_vpq_code_error_groups_t
    grouped_risk_prefix_workspace_for = runtime.grouped_risk_prefix_workspace_for
    grouped_score_direct_workspace_for = runtime.grouped_score_direct_workspace_for
    joint_v_budgets = runtime.joint_v_budgets
    joint_v_budgets_t = runtime.joint_v_budgets_t
    key_bytes = runtime.key_bytes
    value_bytes = runtime.value_bytes
    policy_id = runtime.policy_id
    policy_uses_mb = runtime.policy_uses_mb
    threshold_value = runtime.threshold_value
    outputs_all = runtime.outputs_all
    head_group_runtime = runtime.head_group_runtime

    if grouped_risk_records:
        if not _env_truthy("SELECTOR_PQ_JOINT_NATIVE_POLICY", "0"):
            raise RuntimeError("SELECTOR_PQ_JOINT_GROUPED_RISK_PREFIX requires native joint policy")
        native = load_selector_paged_pq_ext()
        grouped_by_shape: dict[tuple[int, int, int, int], list[dict[str, object]]] = {}
        for record in grouped_risk_records:
            score_grid = record.get("score_grid")
            if isinstance(score_grid, torch.Tensor):
                vhat_grid = record.get("vhat")
                if not isinstance(vhat_grid, torch.Tensor):
                    raise RuntimeError("invalid score-direct grouped risk-prefix record")
                shape_key = (
                    int(score_grid.shape[0]),
                    int(score_grid.shape[1]),
                    int(score_grid.shape[2]),
                    int(vhat_grid.shape[1]),
                )
            else:
                base_grid = record["base_output_grid"]
                probs_grid = record["probs_grid"]
                if not isinstance(base_grid, torch.Tensor) or not isinstance(probs_grid, torch.Tensor):
                    raise RuntimeError("invalid grouped risk-prefix record")
                shape_key = (
                    int(base_grid.shape[0]),
                    int(base_grid.shape[1]),
                    int(probs_grid.shape[2]),
                    int(base_grid.shape[2]),
                )
            grouped_by_shape.setdefault(shape_key, []).append(record)

        if _env_truthy("SELECTOR_PQ_JOINT_GROUPED_SOFTMAX_BASE_CUBLAS", "0"):
            grouped_prob_wall_t0 = time.perf_counter() if wall_profile_enabled else 0.0
            if bool(getattr(args, "profile_native_ops", False)):
                _sync_if_cuda(device)
                grouped_prob_t0 = time.perf_counter()
            else:
                grouped_prob_t0 = 0.0
            if not hasattr(native, "joint_softmax_base_outputs_grouped_cublas"):
                raise RuntimeError(
                    "SELECTOR_PQ_JOINT_GROUPED_SOFTMAX_BASE_CUBLAS requires updated CUDA extension"
                )
            for (_k_count_i, _group_heads_i, _context_len_bucket, _dim_i), records in grouped_by_shape.items():
                if not all(isinstance(record.get("score_grid"), torch.Tensor) for record in records):
                    raise RuntimeError(
                        "grouped softmax/base cuBLAS requires every grouped record to carry a score grid"
                    )
                if not all(isinstance(record.get("vhat"), torch.Tensor) for record in records):
                    raise RuntimeError(
                        "grouped softmax/base cuBLAS requires every grouped record to carry V-PQ values"
                    )
                workspace_score_t = records[0].get("grouped_score_grid_workspace")
                use_workspace_score = (
                    isinstance(workspace_score_t, torch.Tensor)
                    and int(workspace_score_t.shape[0]) >= len(records)
                    and int(workspace_score_t.shape[1]) >= int(_k_count_i)
                    and int(workspace_score_t.shape[2]) >= int(_group_heads_i)
                    and int(workspace_score_t.shape[3]) >= int(_context_len_bucket)
                    and all(
                        record.get("grouped_score_grid_workspace") is workspace_score_t
                        and int(record.get("kv_head", -1)) == int(record_i)
                        for record_i, record in enumerate(records)
                    )
                )
                if use_workspace_score:
                    score_grouped_t = workspace_score_t[
                        : len(records),
                        : int(_k_count_i),
                        : int(_group_heads_i),
                        : int(_context_len_bucket),
                    ].contiguous()
                else:
                    score_grouped_t = torch.stack(
                        [record["score_grid"] for record in records],
                        dim=0,
                    ).contiguous()
                use_grouped_vhat = (
                    grouped_vpq_vhat_groups_t is not None
                    and int(grouped_vpq_vhat_groups_t.shape[0]) >= len(records)
                    and int(grouped_vpq_vhat_groups_t.shape[1]) >= int(_context_len_bucket)
                    and int(grouped_vpq_vhat_groups_t.shape[2]) == int(_dim_i)
                    and all(int(record.get("kv_head", -1)) == int(record_i) for record_i, record in enumerate(records))
                )
                if use_grouped_vhat:
                    vhat_grouped_t = grouped_vpq_vhat_groups_t[
                        : len(records),
                        : int(_context_len_bucket),
                        : int(_dim_i),
                    ].contiguous()
                else:
                    vhat_grouped_t = torch.stack(
                        [record["vhat"] for record in records],
                        dim=0,
                    ).contiguous()
                probs_grouped_t, base_grouped_t = native.joint_softmax_base_outputs_grouped_cublas(
                    score_grouped_t,
                    vhat_grouped_t,
                )
                for record_i, record in enumerate(records):
                    record["probs_grid"] = probs_grouped_t[int(record_i)]
                    record["base_output_grid"] = base_grouped_t[int(record_i)]
                    record.pop("score_grid", None)
                    record.pop("vhat", None)
            if bool(getattr(args, "profile_native_ops", False)):
                _sync_if_cuda(device)
                stats[layer_id].add_joint_detail_timing(
                    prob_base_seconds=float(time.perf_counter() - grouped_prob_t0)
                )
            if wall_profile_enabled:
                stats[layer_id].add_joint_wall_timing(
                    prob_base_seconds=float(time.perf_counter() - grouped_prob_wall_t0)
                )

        grouped_risk_wall_t0 = time.perf_counter() if wall_profile_enabled else 0.0
        if bool(getattr(args, "profile_native_ops", False)):
            _sync_if_cuda(device)
            grouped_risk_t0 = time.perf_counter()
        else:
            grouped_risk_t0 = 0.0
        grouped_policy_batches: list[
            tuple[list[dict[str, object]], int, int, int, torch.Tensor]
        ] = []
        grouped_accounting_batches: list[
            tuple[list[dict[str, object]], int, int, torch.Tensor, torch.Tensor]
        ] = []
        use_fused_grouped_risk_policy = _env_truthy("SELECTOR_PQ_JOINT_FUSED_RISK_POLICY", "0")
        use_interval_grouped_risk_policy = _env_truthy("SELECTOR_PQ_JOINT_INTERVAL_RISK_POLICY", "0")

        def grouped_policy_mb_tensors(
            records_i: list[dict[str, object]],
            k_count_i: int,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            if not policy_uses_mb:
                return (
                    torch.empty((len(records_i), int(k_count_i)), dtype=torch.float32, device=device),
                    torch.empty((len(records_i), len(joint_v_budgets)), dtype=torch.float32, device=device),
                )
            return (
                torch.stack(
                    [
                        torch.as_tensor(record["grid_k_mb_by_idx"], dtype=torch.float32, device=device)
                        for record in records_i
                    ],
                    dim=0,
                ).contiguous(),
                torch.stack(
                    [
                        torch.as_tensor(record["v_mb_by_idx"], dtype=torch.float32, device=device)
                        for record in records_i
                    ],
                    dim=0,
                ).contiguous(),
            )

        for (k_count_i, group_heads_i, context_len_bucket, dim_i), records in grouped_by_shape.items():
            rows_per_group = int(k_count_i) * int(group_heads_i)
            group_pack_wall_t0 = time.perf_counter() if wall_profile_enabled else 0.0
            if bool(getattr(args, "profile_native_ops", False)):
                _sync_if_cuda(device)
                group_pack_t0 = time.perf_counter()
            else:
                group_pack_t0 = 0.0
            use_score_direct_records = all(
                isinstance(record.get("score_grid"), torch.Tensor) for record in records
            )
            if use_score_direct_records:
                if not hasattr(native, "joint_vprefix_outputs_from_grouped_scores_batched"):
                    raise RuntimeError(
                        "SELECTOR_PQ_JOINT_SCORE_DIRECT_VPREFIX requires updated CUDA extension"
                    )
                workspace_score_t = records[0].get("grouped_score_grid_workspace")
                use_workspace_score = (
                    isinstance(workspace_score_t, torch.Tensor)
                    and int(workspace_score_t.shape[0]) >= len(records)
                    and int(workspace_score_t.shape[1]) >= int(k_count_i)
                    and int(workspace_score_t.shape[2]) >= int(group_heads_i)
                    and int(workspace_score_t.shape[3]) >= int(context_len_bucket)
                    and all(
                        record.get("grouped_score_grid_workspace") is workspace_score_t
                        and int(record.get("kv_head", -1)) == int(record_i)
                        for record_i, record in enumerate(records)
                    )
                )
                if use_workspace_score:
                    score_grouped_t = workspace_score_t[
                        : len(records),
                        : int(k_count_i),
                        : int(group_heads_i),
                        : int(context_len_bucket),
                    ]
                else:
                    score_grouped_t = torch.stack(
                        [
                            record["score_grid"]
                            for record in records
                            if isinstance(record["score_grid"], torch.Tensor)
                        ],
                        dim=0,
                    ).contiguous()
                if (
                    grouped_vpq_vhat_groups_t is not None
                    and grouped_vpq_residual_groups_t is not None
                    and grouped_vpq_code_error_groups_t is not None
                    and len(records) == int(grouped_vpq_vhat_groups_t.shape[0])
                ):
                    vhat_groups_t = grouped_vpq_vhat_groups_t.contiguous()
                    residual_groups_t = grouped_vpq_residual_groups_t.contiguous()
                    code_error_groups_t = grouped_vpq_code_error_groups_t.contiguous()
                else:
                    vhat_groups_t = torch.stack(
                        [
                            record["vhat"]
                            for record in records
                            if isinstance(record["vhat"], torch.Tensor)
                        ],
                        dim=0,
                    ).contiguous()
                    residual_groups_t = torch.stack(
                        [
                            record["residual"]
                            for record in records
                            if isinstance(record["residual"], torch.Tensor)
                        ],
                        dim=0,
                    ).contiguous()
                    code_error_groups_t = torch.stack(
                        [
                            record["code_error"]
                            for record in records
                            if isinstance(record["code_error"], torch.Tensor)
                        ],
                        dim=0,
                    ).contiguous()
                if bool(getattr(args, "profile_native_ops", False)):
                    _sync_if_cuda(device)
                    stats[layer_id].add_joint_detail_timing(
                        group_pack_seconds=float(time.perf_counter() - group_pack_t0)
                    )
                if wall_profile_enabled:
                    stats[layer_id].add_joint_wall_timing(
                        group_pack_seconds=float(time.perf_counter() - group_pack_wall_t0)
                    )
                if _env_truthy("SELECTOR_PQ_JOINT_SCORE_DIRECT_INTERVAL_POLICY", "0"):
                    if policy_uses_mb:
                        raise RuntimeError(
                            "SELECTOR_PQ_JOINT_SCORE_DIRECT_INTERVAL_POLICY only supports non-MB policies"
                        )
                    if not hasattr(native, "joint_select_policy_from_grouped_scores_intervals_batched_no_mb"):
                        raise RuntimeError(
                            "SELECTOR_PQ_JOINT_SCORE_DIRECT_INTERVAL_POLICY requires updated CUDA extension"
                        )
                    final_outputs_grouped_t, final_idx_grouped_t = (
                        native.joint_select_policy_from_grouped_scores_intervals_batched_no_mb(
                            score_grouped_t,
                            vhat_groups_t,
                            residual_groups_t,
                            code_error_groups_t,
                            joint_v_budgets_t,
                            float(threshold_value),
                            policy_id,
                        )
                    )
                    for record_i, record in enumerate(records):
                        record["final_outputs"] = final_outputs_grouped_t[int(record_i)]
                        record["final_indices"] = final_idx_grouped_t[int(record_i)]
                    grouped_accounting_batches.append(
                        (
                            records,
                            int(k_count_i),
                            int(group_heads_i),
                            final_outputs_grouped_t,
                            final_idx_grouped_t,
                        )
                    )
                    continue
                if (
                    _env_truthy("SELECTOR_PQ_JOINT_SCORE_DIRECT_WORKSPACE", "0")
                    and hasattr(native, "joint_vprefix_outputs_from_grouped_scores_batched_workspace")
                ):
                    rows_i = int(score_grouped_t.shape[0]) * int(score_grouped_t.shape[1]) * int(score_grouped_t.shape[2])
                    workspace = grouped_score_direct_workspace_for(
                        rows=int(rows_i),
                        context_len=int(score_grouped_t.shape[3]),
                        v_steps=int(joint_v_budgets_t.numel()),
                        dim=int(vhat_groups_t.shape[2]),
                    )
                    grouped_outputs_flat_t = native.joint_vprefix_outputs_from_grouped_scores_batched_workspace(
                        score_grouped_t,
                        vhat_groups_t,
                        residual_groups_t,
                        code_error_groups_t,
                        joint_v_budgets_t,
                        *workspace,
                    )
                else:
                    grouped_outputs_flat_t = native.joint_vprefix_outputs_from_grouped_scores_batched(
                        score_grouped_t,
                        vhat_groups_t,
                        residual_groups_t,
                        code_error_groups_t,
                        joint_v_budgets_t,
                    )
                grouped_policy_batches.append(
                    (records, int(k_count_i), int(group_heads_i), int(dim_i), grouped_outputs_flat_t)
                )
                continue
            workspace_base = records[0].get("grouped_base_workspace") if records else None
            workspace_probs = records[0].get("grouped_probs_workspace") if records else None
            use_grouped_output_workspace = (
                _env_truthy("SELECTOR_PQ_JOINT_GROUPED_OUTPUT_WORKSPACE", "0")
                and isinstance(workspace_base, torch.Tensor)
                and isinstance(workspace_probs, torch.Tensor)
                and len(records) <= int(workspace_base.shape[0])
                and len(records) <= int(workspace_probs.shape[0])
                and all(
                    record.get("grouped_base_workspace") is workspace_base
                    and record.get("grouped_probs_workspace") is workspace_probs
                    and int(record.get("kv_head", -1)) == int(record_i)
                    for record_i, record in enumerate(records)
                )
            )
            if use_grouped_output_workspace:
                base_grouped_t = workspace_base[: len(records), : int(k_count_i), : int(group_heads_i), : int(dim_i)]
                if grouped_strided_output_workspace_enabled:
                    probs_grouped_t = workspace_probs[
                        : len(records),
                        : int(k_count_i),
                        : int(group_heads_i),
                        :,
                    ]
                else:
                    probs_grouped_t = workspace_probs[
                        : len(records),
                        : int(k_count_i),
                        : int(group_heads_i),
                        : int(context_len_bucket),
                    ]
            else:
                base_grouped_t = torch.stack(
                    [
                        record["base_output_grid"]
                        for record in records
                        if isinstance(record["base_output_grid"], torch.Tensor)
                    ],
                    dim=0,
                ).contiguous()
                probs_grouped_t = torch.stack(
                    [
                        record["probs_grid"]
                        for record in records
                        if isinstance(record["probs_grid"], torch.Tensor)
                    ],
                    dim=0,
                ).contiguous()
            if (
                grouped_vpq_residual_groups_t is not None
                and grouped_vpq_code_error_groups_t is not None
                and len(records) == int(grouped_vpq_residual_groups_t.shape[0])
            ):
                residual_groups_t = grouped_vpq_residual_groups_t.contiguous()
                code_error_groups_t = grouped_vpq_code_error_groups_t.contiguous()
            else:
                residual_groups_t = torch.stack(
                    [
                        record["residual"]
                        for record in records
                        if isinstance(record["residual"], torch.Tensor)
                    ],
                    dim=0,
                ).contiguous()
                code_error_groups_t = torch.stack(
                    [
                        record["code_error"]
                        for record in records
                        if isinstance(record["code_error"], torch.Tensor)
                    ],
                    dim=0,
                ).contiguous()
            if bool(getattr(args, "profile_native_ops", False)):
                _sync_if_cuda(device)
                stats[layer_id].add_joint_detail_timing(
                    group_pack_seconds=float(time.perf_counter() - group_pack_t0)
                )
            if wall_profile_enabled:
                stats[layer_id].add_joint_wall_timing(
                    group_pack_seconds=float(time.perf_counter() - group_pack_wall_t0)
                )
            if use_fused_grouped_risk_policy:
                if (
                    not policy_uses_mb
                    and hasattr(native, "joint_select_policy_from_grouped_risk_batched_no_mb")
                ):
                    final_outputs_grouped_t, final_idx_grouped_t = (
                        native.joint_select_policy_from_grouped_risk_batched_no_mb(
                            base_grouped_t,
                            probs_grouped_t,
                            residual_groups_t,
                            code_error_groups_t,
                            joint_v_budgets_t,
                            float(threshold_value),
                            policy_id,
                        )
                    )
                else:
                    k_mb_groups_t, v_mb_groups_t = grouped_policy_mb_tensors(records, int(k_count_i))
                    if hasattr(native, "joint_select_policy_from_grouped_risk_batched"):
                        final_outputs_grouped_t, final_idx_grouped_t = (
                            native.joint_select_policy_from_grouped_risk_batched(
                                base_grouped_t,
                                probs_grouped_t,
                                residual_groups_t,
                                code_error_groups_t,
                                joint_v_budgets_t,
                                k_mb_groups_t,
                                v_mb_groups_t,
                                float(threshold_value),
                                policy_id,
                            )
                        )
                    elif (
                        not policy_uses_mb
                        and hasattr(native, "joint_select_policy_from_grouped_risk_no_mb")
                    ):
                        final_outputs_grouped_t, final_idx_grouped_t = (
                            native.joint_select_policy_from_grouped_risk_no_mb(
                                base_grouped_t.reshape(len(records) * rows_per_group, dim_i),
                                probs_grouped_t.reshape(len(records) * rows_per_group, context_len_bucket),
                                residual_groups_t,
                                code_error_groups_t,
                                joint_v_budgets_t,
                                int(k_count_i),
                                int(group_heads_i),
                                float(threshold_value),
                                policy_id,
                            )
                        )
                    else:
                        final_outputs_grouped_t, final_idx_grouped_t = (
                            native.joint_select_policy_from_grouped_risk(
                                base_grouped_t.reshape(len(records) * rows_per_group, dim_i),
                                probs_grouped_t.reshape(len(records) * rows_per_group, context_len_bucket),
                                residual_groups_t,
                                code_error_groups_t,
                                joint_v_budgets_t,
                                k_mb_groups_t,
                                v_mb_groups_t,
                                int(k_count_i),
                                int(group_heads_i),
                                float(threshold_value),
                                policy_id,
                            )
                        )
                for record_i, record in enumerate(records):
                    record["final_outputs"] = final_outputs_grouped_t[int(record_i)]
                    record["final_indices"] = final_idx_grouped_t[int(record_i)]
                grouped_accounting_batches.append(
                    (
                        records,
                        int(k_count_i),
                        int(group_heads_i),
                        final_outputs_grouped_t,
                        final_idx_grouped_t,
                    )
                )
                continue
            if use_interval_grouped_risk_policy:
                if policy_uses_mb:
                    raise RuntimeError(
                        "SELECTOR_PQ_JOINT_INTERVAL_RISK_POLICY only supports non-MB policies"
                    )
                if not hasattr(native, "joint_select_policy_from_grouped_risk_intervals_batched_no_mb"):
                    raise RuntimeError(
                        "SELECTOR_PQ_JOINT_INTERVAL_RISK_POLICY requires updated CUDA extension"
                    )
                final_outputs_grouped_t, final_idx_grouped_t = (
                    native.joint_select_policy_from_grouped_risk_intervals_batched_no_mb(
                        base_grouped_t,
                        probs_grouped_t,
                        residual_groups_t,
                        code_error_groups_t,
                        joint_v_budgets_t,
                        float(threshold_value),
                        policy_id,
                    )
                )
                for record_i, record in enumerate(records):
                    record["final_outputs"] = final_outputs_grouped_t[int(record_i)]
                    record["final_indices"] = final_idx_grouped_t[int(record_i)]
                grouped_accounting_batches.append(
                    (
                        records,
                        int(k_count_i),
                        int(group_heads_i),
                        final_outputs_grouped_t,
                        final_idx_grouped_t,
                    )
                )
                continue
            if (
                _env_truthy("SELECTOR_PQ_JOINT_RISK_PREFIX_TOPK", "0")
                and hasattr(native, "joint_vprefix_outputs_from_grouped_risk_topk_batched")
            ):
                max_exact_v_count_i = max(
                    0,
                    min(
                        max((int(v) for v in joint_v_budgets), default=0),
                        int(context_len_bucket),
                    ),
                )
                grouped_outputs_flat_t = native.joint_vprefix_outputs_from_grouped_risk_topk_batched(
                    base_grouped_t,
                    probs_grouped_t,
                    residual_groups_t,
                    code_error_groups_t,
                    joint_v_budgets_t,
                    int(max_exact_v_count_i),
                )
            elif hasattr(native, "joint_vprefix_outputs_from_grouped_risk_batched"):
                if _env_truthy("SELECTOR_PQ_JOINT_RISK_PREFIX_WORKSPACE", "0"):
                    if grouped_strided_output_workspace_enabled:
                        if not hasattr(
                            native,
                            "joint_vprefix_outputs_from_grouped_risk_batched_strided_workspace",
                        ):
                            raise RuntimeError(
                                "SELECTOR_PQ_JOINT_STRIDED_OUTPUT_WORKSPACE requires updated CUDA extension"
                            )
                    elif not hasattr(native, "joint_vprefix_outputs_from_grouped_risk_batched_workspace"):
                        raise RuntimeError(
                            "SELECTOR_PQ_JOINT_RISK_PREFIX_WORKSPACE requires updated CUDA extension"
                        )
                    (
                        risk_in_t,
                        risk_out_t,
                        ids_in_t,
                        ids_out_t,
                        offsets_t,
                        temp_t,
                        interval_sums_t,
                        outputs_workspace_t,
                    ) = grouped_risk_prefix_workspace_for(
                        rows=int(len(records) * rows_per_group),
                        context_len=int(context_len_bucket),
                        v_steps=int(len(joint_v_budgets)),
                        dim=int(dim_i),
                    )
                    if grouped_strided_output_workspace_enabled:
                        grouped_outputs_flat_t = native.joint_vprefix_outputs_from_grouped_risk_batched_strided_workspace(
                            base_grouped_t.contiguous(),
                            probs_grouped_t.contiguous(),
                            residual_groups_t,
                            code_error_groups_t,
                            joint_v_budgets_t,
                            int(context_len_bucket),
                            risk_in_t,
                            risk_out_t,
                            ids_in_t,
                            ids_out_t,
                            offsets_t,
                            temp_t,
                            interval_sums_t,
                            outputs_workspace_t,
                        )
                    else:
                        grouped_outputs_flat_t = native.joint_vprefix_outputs_from_grouped_risk_batched_workspace(
                            base_grouped_t,
                            probs_grouped_t,
                            residual_groups_t,
                            code_error_groups_t,
                            joint_v_budgets_t,
                            risk_in_t,
                            risk_out_t,
                            ids_in_t,
                            ids_out_t,
                            offsets_t,
                            temp_t,
                            interval_sums_t,
                            outputs_workspace_t,
                        )
                else:
                    grouped_outputs_flat_t = native.joint_vprefix_outputs_from_grouped_risk_batched(
                        base_grouped_t,
                        probs_grouped_t,
                        residual_groups_t,
                        code_error_groups_t,
                        joint_v_budgets_t,
                    )
            else:
                row_group_ids_t = torch.arange(
                    len(records),
                    dtype=torch.long,
                    device=device,
                ).repeat_interleave(rows_per_group)
                grouped_outputs_flat_t = native.joint_vprefix_outputs_from_grouped_risk(
                    base_grouped_t.reshape(len(records) * rows_per_group, dim_i),
                    probs_grouped_t.reshape(len(records) * rows_per_group, context_len_bucket),
                    residual_groups_t,
                    code_error_groups_t,
                    row_group_ids_t,
                    joint_v_budgets_t,
                )
            grouped_policy_batches.append(
                (records, int(k_count_i), int(group_heads_i), int(dim_i), grouped_outputs_flat_t)
            )
        if bool(getattr(args, "profile_native_ops", False)):
            _sync_if_cuda(device)
            stats[layer_id].add_joint_detail_timing(
                risk_prefix_seconds=float(time.perf_counter() - grouped_risk_t0)
            )
            grouped_policy_t0 = time.perf_counter()
        else:
            grouped_policy_t0 = 0.0
        if wall_profile_enabled:
            stats[layer_id].add_joint_wall_timing(
                risk_prefix_seconds=float(time.perf_counter() - grouped_risk_wall_t0)
            )
        grouped_policy_wall_t0 = time.perf_counter() if wall_profile_enabled else 0.0

        for records, k_count_i, group_heads_i, dim_i, grouped_outputs_flat_t in grouped_policy_batches:
            if (not policy_uses_mb) and hasattr(native, "joint_select_policy_grouped_flat_no_mb"):
                final_outputs_grouped_t, final_idx_grouped_t = native.joint_select_policy_grouped_flat_no_mb(
                    grouped_outputs_flat_t,
                    int(k_count_i),
                    int(group_heads_i),
                    float(threshold_value),
                    policy_id,
                )
            else:
                k_mb_groups_t, v_mb_groups_t = grouped_policy_mb_tensors(records, int(k_count_i))
                final_outputs_grouped_t, final_idx_grouped_t = native.joint_select_policy_grouped_flat(
                    grouped_outputs_flat_t,
                    k_mb_groups_t,
                    v_mb_groups_t,
                    int(k_count_i),
                    int(group_heads_i),
                    float(threshold_value),
                    policy_id,
                )
            for record_i, record in enumerate(records):
                record["final_outputs"] = final_outputs_grouped_t[int(record_i)]
                record["final_indices"] = final_idx_grouped_t[int(record_i)]
            grouped_accounting_batches.append(
                (
                    records,
                    int(k_count_i),
                    int(group_heads_i),
                    final_outputs_grouped_t,
                    final_idx_grouped_t,
                )
            )

        if bool(getattr(args, "profile_native_ops", False)):
            _sync_if_cuda(device)
            stats[layer_id].add_joint_detail_timing(
                policy_seconds=float(time.perf_counter() - grouped_policy_t0)
            )
            grouped_accounting_t0 = time.perf_counter()
        else:
            grouped_accounting_t0 = 0.0
        if wall_profile_enabled:
            stats[layer_id].add_joint_wall_timing(
                policy_seconds=float(time.perf_counter() - grouped_policy_wall_t0)
            )
        grouped_accounting_wall_t0 = time.perf_counter() if wall_profile_enabled else 0.0

        if (
            not bool(getattr(args, "disable_cost_stats", False))
            and _env_truthy("SELECTOR_PQ_JOINT_NATIVE_ACCOUNTING", "0")
            and not _env_truthy("SELECTOR_PQ_JOINT_NATIVE_ACCOUNTING_VERIFY", "0")
            and hasattr(native, "joint_grouped_accounting_sums")
        ):
            for (
                records,
                _k_count_i,
                group_heads_i,
                final_outputs_grouped_t,
                final_idx_grouped_t,
            ) in grouped_accounting_batches:
                if not records:
                    continue
                first = records[0]
                first_counts = first.get("grid_selected_counts_by_ki")
                if not isinstance(first_counts, list):
                    continue
                selector_mb0 = float(first["selector_mb"])
                v_pq_codebook_mb0 = float(first["v_pq_codebook_mb"])
                actual_value_subvecs0 = int(first["actual_value_subvecs"])
                context_len0 = int(first["context_len"])
                compatible = True
                for record in records[1:]:
                    counts_i = record.get("grid_selected_counts_by_ki")
                    if (
                        not isinstance(counts_i, list)
                        or [int(x) for x in counts_i] != [int(x) for x in first_counts]
                        or int(record["context_len"]) != context_len0
                        or abs(float(record["selector_mb"]) - selector_mb0) > 1e-12
                        or abs(float(record["v_pq_codebook_mb"]) - v_pq_codebook_mb0) > 1e-12
                        or int(record["actual_value_subvecs"]) != actual_value_subvecs0
                    ):
                        compatible = False
                        break
                if not compatible:
                    continue
                selected_counts_t = torch.as_tensor(
                    [int(x) for x in first_counts],
                    dtype=torch.long,
                    device=device,
                )
                accounting_sums_t = native.joint_grouped_accounting_sums(
                    final_idx_grouped_t.reshape(-1, 2),
                    selected_counts_t,
                    joint_v_budgets_t,
                    int(context_len0),
                    int(self.head_dim),
                    int(key_bytes),
                    int(value_bytes),
                    selector_mb0,
                    v_pq_codebook_mb0,
                    float(records[0].get("metadata_mb", 0.0)),
                    actual_value_subvecs0,
                    int(records[0].get("code_bytes", 1)),
                )
                stats[layer_id].add_count_sums_device(
                    int(len(records) * int(group_heads_i)),
                    accounting_sums_t,
                )
                for record_i, record in enumerate(records):
                    head_start_i = int(record["head_start"])
                    head_end_i = int(record["head_end"])
                    group_heads_record = int(record["group_heads"])
                    outputs_all[head_start_i:head_end_i] = final_outputs_grouped_t[
                        int(record_i), :group_heads_record
                    ]
                    record["accounting_batched"] = True

        for record in grouped_risk_records:
            final_output_t = record["final_outputs"]
            final_idx_t = record["final_indices"]
            if not isinstance(final_output_t, torch.Tensor) or not isinstance(final_idx_t, torch.Tensor):
                raise RuntimeError("missing grouped risk-prefix final output")
            if bool(record.get("accounting_batched", False)):
                continue
            group_heads_i = int(record["group_heads"])
            head_start_i = int(record["head_start"])
            head_end_i = int(record["head_end"])
            context_len_i = int(record["context_len"])
            if bool(getattr(args, "disable_cost_stats", False)):
                outputs_all[head_start_i:head_end_i] = final_output_t[:group_heads_i]
                continue

            grid_selected_by_ki = record["grid_selected_by_ki"]
            if not isinstance(grid_selected_by_ki, list):
                raise RuntimeError("invalid grouped risk-prefix selected-token metadata")
            grid_selected_counts_by_ki = record.get("grid_selected_counts_by_ki")
            if grid_selected_counts_by_ki is not None and not isinstance(grid_selected_counts_by_ki, list):
                raise RuntimeError("invalid grouped risk-prefix selected-count metadata")
            use_native_accounting = (
                _env_truthy("SELECTOR_PQ_JOINT_NATIVE_ACCOUNTING", "0")
                and hasattr(native, "joint_grouped_accounting_sums")
                and grid_selected_counts_by_ki is not None
            )
            if use_native_accounting:
                selected_counts_t = torch.as_tensor(
                    [int(x) for x in grid_selected_counts_by_ki],
                    dtype=torch.long,
                    device=device,
                )
                accounting_sums_t = native.joint_grouped_accounting_sums(
                    final_idx_t,
                    selected_counts_t,
                    joint_v_budgets_t,
                    int(context_len_i),
                    int(self.head_dim),
                    int(key_bytes),
                    int(value_bytes),
                    float(record["selector_mb"]),
                    float(record["v_pq_codebook_mb"]),
                    float(record.get("metadata_mb", 0.0)),
                    int(record["actual_value_subvecs"]),
                    int(record.get("code_bytes", 1)),
                )
                if _env_truthy("SELECTOR_PQ_JOINT_NATIVE_ACCOUNTING_VERIFY", "0"):
                    accounting_sums = accounting_sums_t.detach().cpu().tolist()
                    ref_sums = [0.0] * 11
                    final_idx_rows = final_idx_t.detach().cpu().tolist()
                    for row in final_idx_rows:
                        ki = int(row[0])
                        vi = int(row[1])
                        selected_count_i = int(grid_selected_counts_by_ki[int(ki)])
                        exact_v_count = max(0, min(int(joint_v_budgets[int(vi)]), context_len_i))
                        tail_count_i = max(0, context_len_i - int(exact_v_count))
                        exact_key_mb = float(selected_count_i * int(self.head_dim) * key_bytes) / MB
                        exact_v_mb = float(exact_v_count * int(self.head_dim) * value_bytes) / MB
                        compressed_v_codes_mb = (
                            float(tail_count_i * int(record["actual_value_subvecs"]) * int(record.get("code_bytes", 1))) / MB
                        )
                        tail_mb_override = (
                            float(record["v_pq_codebook_mb"]) + compressed_v_codes_mb + float(record.get("metadata_mb", 0.0))
                        )
                        dense_physical_key_mb = float(context_len_i * int(self.head_dim) * key_bytes) / MB
                        ref_sums[0] += float(selected_count_i)
                        ref_sums[1] += float(tail_count_i)
                        ref_sums[2] += float(record["selector_mb"])
                        ref_sums[3] += float(exact_key_mb + exact_v_mb)
                        ref_sums[4] += float(tail_mb_override)
                        ref_sums[6] += float(dense_physical_key_mb + exact_v_mb)
                        ref_sums[8] += 1.0 if float(record["selector_mb"]) > 0.0 else 0.0
                        ref_sums[9] += 1.0 if float(tail_mb_override) > 0.0 else 0.0
                    max_diff = max(
                        abs(float(accounting_sums[i]) - float(ref_sums[i]))
                        for i in range(len(ref_sums))
                    )
                    if max_diff > 1e-8:
                        raise RuntimeError(
                            "native grouped accounting mismatch: "
                            f"max_diff={max_diff} native={accounting_sums} ref={ref_sums}"
                        )
                    stats[layer_id].add_count_sums(
                        int(group_heads_i),
                        selected_sum=float(accounting_sums[0]),
                        tail_count_sum=float(accounting_sums[1]),
                        selector_mb_sum=float(accounting_sums[2]),
                        exact_kv_mb_sum=float(accounting_sums[3]),
                        tail_mb_sum=float(accounting_sums[4]),
                        confidence_mb_sum=float(accounting_sums[5]),
                        physical_gpu_exact_kv_mb_sum=float(accounting_sums[6]),
                        physical_gpu_confidence_mb_sum=float(accounting_sums[7]),
                        selector_active_count=int(round(float(accounting_sums[8]))),
                        tail_active_count=int(round(float(accounting_sums[9]))),
                        confidence_active_count=int(round(float(accounting_sums[10]))),
                    )
                else:
                    stats[layer_id].add_count_sums_device(
                        int(group_heads_i),
                        accounting_sums_t,
                    )
                outputs_all[head_start_i:head_end_i] = final_output_t[:group_heads_i]
            else:
                final_idx_rows = final_idx_t.detach().cpu().tolist()
                for local_head_i, row in enumerate(final_idx_rows):
                    ki = int(row[0])
                    vi = int(row[1])
                    global_head_i = int(head_start_i + local_head_i)
                    if grid_selected_counts_by_ki is not None:
                        selected_count_i = int(grid_selected_counts_by_ki[int(ki)])
                    else:
                        selected_t_i = grid_selected_by_ki[int(ki)]
                        if selected_t_i is None:
                            raise RuntimeError("missing selected-token tensor for grouped risk-prefix accounting")
                        selected_count_i = int(selected_t_i[int(local_head_i)].numel())
                    exact_v_count = max(0, min(int(joint_v_budgets[int(vi)]), context_len_i))
                    exact_key_mb = float(selected_count_i * int(self.head_dim) * key_bytes) / MB
                    exact_v_mb = float(exact_v_count * int(self.head_dim) * value_bytes) / MB
                    compressed_v_codes_mb = (
                        float(
                            max(0, context_len_i - exact_v_count)
                            * int(record["actual_value_subvecs"])
                            * int(record.get("code_bytes", 1))
                        ) / MB
                    )
                    tail_mb_override = (
                        float(record["v_pq_codebook_mb"])
                        + compressed_v_codes_mb
                        + float(record.get("metadata_mb", 0.0))
                    )
                    dense_physical_key_mb = float(context_len_i * int(self.head_dim) * key_bytes) / MB
                    stats[layer_id].add_count(
                        int(selected_count_i),
                        max(0, context_len_i - int(exact_v_count)),
                        float(record["selector_mb"]),
                        int(self.head_dim),
                        key_bytes,
                        value_bytes,
                        tail_mb_override=tail_mb_override,
                        exact_kv_mb_override=float(exact_key_mb + exact_v_mb),
                        confidence_mb_override=0.0,
                        physical_gpu_exact_kv_mb_override=float(dense_physical_key_mb + exact_v_mb),
                        physical_gpu_confidence_mb_override=0.0,
                    )
                    outputs_all[global_head_i] = final_output_t[int(local_head_i)]

        if bool(getattr(args, "profile_native_ops", False)):
            _sync_if_cuda(device)
            stats[layer_id].add_joint_detail_timing(
                accounting_seconds=float(time.perf_counter() - grouped_accounting_t0)
            )
            if head_group_runtime.grouped_geo_t0 > 0.0:
                stats[layer_id].add_native_detail_timing(
                    geometric_seconds=float(time.perf_counter() - head_group_runtime.grouped_geo_t0)
                )
        if wall_profile_enabled:
            stats[layer_id].add_joint_wall_timing(
                accounting_seconds=float(time.perf_counter() - grouped_accounting_wall_t0)
            )
