#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from benchmark.selector_eval.runners.hf_paged_pq_intervention_common import MB


@dataclass
class ApproxStats:
    calls: int = 0
    approx_attention_calls: int = 0
    passthrough_attention_calls: int = 0
    mean_selected: float = 0.0
    mean_tail_samples: float = 0.0
    mean_selector_mb: float = 0.0
    mean_exact_kv_mb: float = 0.0
    mean_tail_mb: float = 0.0
    mean_confidence_mb: float = 0.0
    mean_step_mb: float = 0.0
    mean_physical_gpu_exact_kv_mb: float = 0.0
    mean_physical_gpu_confidence_mb: float = 0.0
    mean_physical_gpu_step_mb: float = 0.0
    selector_active_calls: int = 0
    tail_active_calls: int = 0
    confidence_active_calls: int = 0
    index_build_calls: int = 0
    index_build_seconds: float = 0.0
    index_build_read_mb: float = 0.0
    index_build_write_mb: float = 0.0
    cache_cast_seconds: float = 0.0
    patched_attention_seconds: float = 0.0
    qkv_cache_seconds: float = 0.0
    index_sidecar_seconds: float = 0.0
    native_pack_seconds: float = 0.0
    native_selector_seconds: float = 0.0
    native_attention_seconds: float = 0.0
    native_exact_logit_seconds: float = 0.0
    native_threshold_seconds: float = 0.0
    native_geometric_seconds: float = 0.0
    native_output_seconds: float = 0.0
    native_joint_rank_prefix_seconds: float = 0.0
    native_joint_score_grid_seconds: float = 0.0
    native_joint_prob_base_seconds: float = 0.0
    native_joint_risk_prefix_seconds: float = 0.0
    native_joint_policy_seconds: float = 0.0
    native_joint_precompute_seconds: float = 0.0
    native_joint_layout_seconds: float = 0.0
    native_joint_group_pack_seconds: float = 0.0
    native_joint_accounting_seconds: float = 0.0
    joint_staged_kv_groups: int = 0
    joint_staged_kv_accepted_groups: int = 0
    joint_staged_kv_boundary_groups: int = 0
    native_vpq_append_seconds: float = 0.0
    native_vpq_append_calls: int = 0
    native_vpq_append_grouped_calls: int = 0
    native_vpq_append_fallback_calls: int = 0
    output_projection_seconds: float = 0.0
    wall_patched_attention_seconds: float = 0.0
    wall_qkv_cache_seconds: float = 0.0
    wall_index_sidecar_seconds: float = 0.0
    wall_output_projection_seconds: float = 0.0
    wall_joint_total_seconds: float = 0.0
    wall_joint_precompute_seconds: float = 0.0
    wall_joint_selector_seconds: float = 0.0
    wall_joint_exact_logit_seconds: float = 0.0
    wall_joint_vpq_sidecar_seconds: float = 0.0
    wall_joint_layout_seconds: float = 0.0
    wall_joint_rank_prefix_seconds: float = 0.0
    wall_joint_score_grid_seconds: float = 0.0
    wall_joint_prob_base_seconds: float = 0.0
    wall_joint_risk_prefix_seconds: float = 0.0
    wall_joint_policy_seconds: float = 0.0
    wall_joint_group_pack_seconds: float = 0.0
    wall_joint_accounting_seconds: float = 0.0
    _device_count_sums: torch.Tensor | None = field(default=None, init=False, repr=False)
    _device_count_repeats: int = field(default=0, init=False, repr=False)

    def add_count(
        self,
        selected_count: int,
        tail_count: int,
        selector_mb: float,
        head_dim: int,
        key_bytes: int,
        value_bytes: int,
        tail_mb_override: float | None = None,
        exact_kv_mb_override: float | None = None,
        confidence_mb_override: float = 0.0,
        physical_gpu_exact_kv_mb_override: float | None = None,
        physical_gpu_confidence_mb_override: float | None = None,
    ) -> None:
        self.add_count_repeated(
            1,
            selected_count,
            tail_count,
            selector_mb,
            head_dim,
            key_bytes,
            value_bytes,
            tail_mb_override=tail_mb_override,
            exact_kv_mb_override=exact_kv_mb_override,
            confidence_mb_override=confidence_mb_override,
            physical_gpu_exact_kv_mb_override=physical_gpu_exact_kv_mb_override,
            physical_gpu_confidence_mb_override=physical_gpu_confidence_mb_override,
        )

    def add_count_repeated(
        self,
        repeats: int,
        selected_count: int,
        tail_count: int,
        selector_mb: float,
        head_dim: int,
        key_bytes: int,
        value_bytes: int,
        tail_mb_override: float | None = None,
        exact_kv_mb_override: float | None = None,
        confidence_mb_override: float = 0.0,
        physical_gpu_exact_kv_mb_override: float | None = None,
        physical_gpu_confidence_mb_override: float | None = None,
    ) -> None:
        repeats = int(repeats)
        if repeats <= 0:
            return
        exact_kv_mb = (
            float(exact_kv_mb_override)
            if exact_kv_mb_override is not None
            else float(int(selected_count) * head_dim * (key_bytes + value_bytes)) / MB
        )
        tail_mb = (
            float(tail_mb_override)
            if tail_mb_override is not None
            else float(tail_count * head_dim * (key_bytes + value_bytes)) / MB
        )
        confidence_mb = float(confidence_mb_override)
        step_mb = float(selector_mb) + exact_kv_mb + tail_mb + confidence_mb
        physical_gpu_exact_kv_mb = (
            float(physical_gpu_exact_kv_mb_override)
            if physical_gpu_exact_kv_mb_override is not None
            else exact_kv_mb
        )
        physical_gpu_confidence_mb = (
            float(physical_gpu_confidence_mb_override)
            if physical_gpu_confidence_mb_override is not None
            else confidence_mb
        )
        physical_gpu_step_mb = float(selector_mb) + physical_gpu_exact_kv_mb + tail_mb + physical_gpu_confidence_mb
        next_calls = self.calls + repeats
        alpha = float(repeats) / float(next_calls)
        self.mean_selected += alpha * (float(selected_count) - self.mean_selected)
        self.mean_tail_samples += alpha * (float(tail_count) - self.mean_tail_samples)
        self.mean_selector_mb += alpha * (float(selector_mb) - self.mean_selector_mb)
        self.mean_exact_kv_mb += alpha * (exact_kv_mb - self.mean_exact_kv_mb)
        self.mean_tail_mb += alpha * (tail_mb - self.mean_tail_mb)
        self.mean_confidence_mb += alpha * (confidence_mb - self.mean_confidence_mb)
        self.mean_step_mb += alpha * (step_mb - self.mean_step_mb)
        self.mean_physical_gpu_exact_kv_mb += alpha * (
            physical_gpu_exact_kv_mb - self.mean_physical_gpu_exact_kv_mb
        )
        self.mean_physical_gpu_confidence_mb += alpha * (
            physical_gpu_confidence_mb - self.mean_physical_gpu_confidence_mb
        )
        self.mean_physical_gpu_step_mb += alpha * (physical_gpu_step_mb - self.mean_physical_gpu_step_mb)
        if float(selector_mb) > 0.0:
            self.selector_active_calls += repeats
        if float(tail_mb) > 0.0:
            self.tail_active_calls += repeats
        if float(confidence_mb) > 0.0:
            self.confidence_active_calls += repeats
        self.calls = next_calls

    def add_count_sums(
        self,
        repeats: int,
        selected_sum: float,
        tail_count_sum: float,
        selector_mb_sum: float,
        exact_kv_mb_sum: float,
        tail_mb_sum: float,
        confidence_mb_sum: float,
        physical_gpu_exact_kv_mb_sum: float,
        physical_gpu_confidence_mb_sum: float,
        selector_active_count: int,
        tail_active_count: int,
        confidence_active_count: int,
    ) -> None:
        repeats = int(repeats)
        if repeats <= 0:
            return
        next_calls = self.calls + repeats
        inv_next = 1.0 / float(next_calls)
        step_mb_sum = float(selector_mb_sum) + float(exact_kv_mb_sum) + float(tail_mb_sum) + float(confidence_mb_sum)
        physical_gpu_step_mb_sum = (
            float(selector_mb_sum)
            + float(physical_gpu_exact_kv_mb_sum)
            + float(tail_mb_sum)
            + float(physical_gpu_confidence_mb_sum)
        )
        self.mean_selected = (self.mean_selected * self.calls + float(selected_sum)) * inv_next
        self.mean_tail_samples = (self.mean_tail_samples * self.calls + float(tail_count_sum)) * inv_next
        self.mean_selector_mb = (self.mean_selector_mb * self.calls + float(selector_mb_sum)) * inv_next
        self.mean_exact_kv_mb = (self.mean_exact_kv_mb * self.calls + float(exact_kv_mb_sum)) * inv_next
        self.mean_tail_mb = (self.mean_tail_mb * self.calls + float(tail_mb_sum)) * inv_next
        self.mean_confidence_mb = (self.mean_confidence_mb * self.calls + float(confidence_mb_sum)) * inv_next
        self.mean_step_mb = (self.mean_step_mb * self.calls + step_mb_sum) * inv_next
        self.mean_physical_gpu_exact_kv_mb = (
            self.mean_physical_gpu_exact_kv_mb * self.calls + float(physical_gpu_exact_kv_mb_sum)
        ) * inv_next
        self.mean_physical_gpu_confidence_mb = (
            self.mean_physical_gpu_confidence_mb * self.calls + float(physical_gpu_confidence_mb_sum)
        ) * inv_next
        self.mean_physical_gpu_step_mb = (
            self.mean_physical_gpu_step_mb * self.calls + physical_gpu_step_mb_sum
        ) * inv_next
        self.selector_active_calls += int(selector_active_count)
        self.tail_active_calls += int(tail_active_count)
        self.confidence_active_calls += int(confidence_active_count)
        self.calls = next_calls

    def add_count_sums_device(self, repeats: int, sums: torch.Tensor) -> None:
        """Accumulate native accounting sums on device and defer CPU sync to reporting."""

        repeats = int(repeats)
        if repeats <= 0:
            return
        sums_t = sums.detach()
        if sums_t.dim() != 1 or int(sums_t.numel()) != 11:
            raise RuntimeError("device accounting sums must have shape [11]")
        if self._device_count_sums is None:
            self._device_count_sums = torch.zeros_like(sums_t, dtype=torch.float64, device=sums_t.device)
        elif self._device_count_sums.device != sums_t.device:
            self.flush_device_count_sums()
            self._device_count_sums = torch.zeros_like(sums_t, dtype=torch.float64, device=sums_t.device)
        self._device_count_sums.add_(sums_t.to(dtype=torch.float64))
        self._device_count_repeats += int(repeats)

    def reserve_count_sums_device_accumulator(self, repeats: int, device: torch.device) -> torch.Tensor | None:
        """Return the deferred device accumulator for native in-place accounting."""

        repeats = int(repeats)
        if repeats <= 0:
            return None
        if self._device_count_sums is None:
            self._device_count_sums = torch.zeros((11,), dtype=torch.float64, device=device)
        elif self._device_count_sums.device != device:
            self.flush_device_count_sums()
            self._device_count_sums = torch.zeros((11,), dtype=torch.float64, device=device)
        self._device_count_repeats += int(repeats)
        return self._device_count_sums

    def flush_device_count_sums(self) -> None:
        if self._device_count_sums is None or int(self._device_count_repeats) <= 0:
            return
        repeats = int(self._device_count_repeats)
        sums = self._device_count_sums.detach().cpu().tolist()
        self._device_count_sums = None
        self._device_count_repeats = 0
        self.add_count_sums(
            repeats,
            selected_sum=float(sums[0]),
            tail_count_sum=float(sums[1]),
            selector_mb_sum=float(sums[2]),
            exact_kv_mb_sum=float(sums[3]),
            tail_mb_sum=float(sums[4]),
            confidence_mb_sum=float(sums[5]),
            physical_gpu_exact_kv_mb_sum=float(sums[6]),
            physical_gpu_confidence_mb_sum=float(sums[7]),
            selector_active_count=int(round(float(sums[8]))),
            tail_active_count=int(round(float(sums[9]))),
            confidence_active_count=int(round(float(sums[10]))),
        )

    def add(
        self,
        selected: list[int],
        tail_count: int,
        selector_mb: float,
        head_dim: int,
        key_bytes: int,
        value_bytes: int,
        tail_mb_override: float | None = None,
        exact_kv_mb_override: float | None = None,
        confidence_mb_override: float = 0.0,
        physical_gpu_exact_kv_mb_override: float | None = None,
        physical_gpu_confidence_mb_override: float | None = None,
    ) -> None:
        self.add_count(
            len(selected),
            tail_count,
            selector_mb,
            head_dim,
            key_bytes,
            value_bytes,
            tail_mb_override=tail_mb_override,
            exact_kv_mb_override=exact_kv_mb_override,
            confidence_mb_override=confidence_mb_override,
            physical_gpu_exact_kv_mb_override=physical_gpu_exact_kv_mb_override,
            physical_gpu_confidence_mb_override=physical_gpu_confidence_mb_override,
        )

    def add_approx_attention_call(self) -> None:
        self.approx_attention_calls += 1

    def add_passthrough_attention_call(self) -> None:
        self.passthrough_attention_calls += 1

    def add_index_build(self, seconds: float, read_mb: float, write_mb: float) -> None:
        self.index_build_calls += 1
        self.index_build_seconds += float(seconds)
        self.index_build_read_mb += float(read_mb)
        self.index_build_write_mb += float(write_mb)

    def add_cache_cast_timing(self, seconds: float) -> None:
        self.cache_cast_seconds += float(seconds)

    def add_patched_attention_timing(self, seconds: float) -> None:
        self.patched_attention_seconds += float(seconds)

    def add_qkv_cache_timing(self, seconds: float) -> None:
        self.qkv_cache_seconds += float(seconds)

    def add_index_sidecar_timing(self, seconds: float) -> None:
        self.index_sidecar_seconds += float(seconds)

    def add_native_pack_timing(self, seconds: float) -> None:
        self.native_pack_seconds += float(seconds)

    def add_native_timing(self, selector_seconds: float = 0.0, attention_seconds: float = 0.0) -> None:
        self.native_selector_seconds += float(selector_seconds)
        self.native_attention_seconds += float(attention_seconds)

    def add_native_detail_timing(
        self,
        *,
        exact_logit_seconds: float = 0.0,
        threshold_seconds: float = 0.0,
        geometric_seconds: float = 0.0,
        output_seconds: float = 0.0,
    ) -> None:
        self.native_exact_logit_seconds += float(exact_logit_seconds)
        self.native_threshold_seconds += float(threshold_seconds)
        self.native_geometric_seconds += float(geometric_seconds)
        self.native_output_seconds += float(output_seconds)

    def add_joint_detail_timing(
        self,
        *,
        rank_prefix_seconds: float = 0.0,
        score_grid_seconds: float = 0.0,
        prob_base_seconds: float = 0.0,
        risk_prefix_seconds: float = 0.0,
        policy_seconds: float = 0.0,
        precompute_seconds: float = 0.0,
        layout_seconds: float = 0.0,
        group_pack_seconds: float = 0.0,
        accounting_seconds: float = 0.0,
    ) -> None:
        self.native_joint_rank_prefix_seconds += float(rank_prefix_seconds)
        self.native_joint_score_grid_seconds += float(score_grid_seconds)
        self.native_joint_prob_base_seconds += float(prob_base_seconds)
        self.native_joint_risk_prefix_seconds += float(risk_prefix_seconds)
        self.native_joint_policy_seconds += float(policy_seconds)
        self.native_joint_precompute_seconds += float(precompute_seconds)
        self.native_joint_layout_seconds += float(layout_seconds)
        self.native_joint_group_pack_seconds += float(group_pack_seconds)
        self.native_joint_accounting_seconds += float(accounting_seconds)

    def add_vpq_append_timing(
        self,
        *,
        seconds: float = 0.0,
        calls: int = 0,
        grouped_calls: int = 0,
        fallback_calls: int = 0,
    ) -> None:
        self.native_vpq_append_seconds += float(seconds)
        self.native_vpq_append_calls += int(calls)
        self.native_vpq_append_grouped_calls += int(grouped_calls)
        self.native_vpq_append_fallback_calls += int(fallback_calls)

    def add_output_projection_timing(self, seconds: float) -> None:
        self.output_projection_seconds += float(seconds)

    def add_wall_patched_attention_timing(self, seconds: float) -> None:
        self.wall_patched_attention_seconds += float(seconds)

    def add_wall_qkv_cache_timing(self, seconds: float) -> None:
        self.wall_qkv_cache_seconds += float(seconds)

    def add_wall_index_sidecar_timing(self, seconds: float) -> None:
        self.wall_index_sidecar_seconds += float(seconds)

    def add_wall_output_projection_timing(self, seconds: float) -> None:
        self.wall_output_projection_seconds += float(seconds)

    def add_joint_wall_timing(
        self,
        *,
        total_seconds: float = 0.0,
        precompute_seconds: float = 0.0,
        selector_seconds: float = 0.0,
        exact_logit_seconds: float = 0.0,
        vpq_sidecar_seconds: float = 0.0,
        layout_seconds: float = 0.0,
        rank_prefix_seconds: float = 0.0,
        score_grid_seconds: float = 0.0,
        prob_base_seconds: float = 0.0,
        risk_prefix_seconds: float = 0.0,
        policy_seconds: float = 0.0,
        group_pack_seconds: float = 0.0,
        accounting_seconds: float = 0.0,
    ) -> None:
        self.wall_joint_total_seconds += float(total_seconds)
        self.wall_joint_precompute_seconds += float(precompute_seconds)
        self.wall_joint_selector_seconds += float(selector_seconds)
        self.wall_joint_exact_logit_seconds += float(exact_logit_seconds)
        self.wall_joint_vpq_sidecar_seconds += float(vpq_sidecar_seconds)
        self.wall_joint_layout_seconds += float(layout_seconds)
        self.wall_joint_rank_prefix_seconds += float(rank_prefix_seconds)
        self.wall_joint_score_grid_seconds += float(score_grid_seconds)
        self.wall_joint_prob_base_seconds += float(prob_base_seconds)
        self.wall_joint_risk_prefix_seconds += float(risk_prefix_seconds)
        self.wall_joint_policy_seconds += float(policy_seconds)
        self.wall_joint_group_pack_seconds += float(group_pack_seconds)
        self.wall_joint_accounting_seconds += float(accounting_seconds)

    def add_joint_staged_kv_groups(self, total_groups: int, boundary_groups: int) -> None:
        total_i = max(0, int(total_groups))
        boundary_i = max(0, min(int(boundary_groups), total_i))
        self.joint_staged_kv_groups += total_i
        self.joint_staged_kv_boundary_groups += boundary_i
        self.joint_staged_kv_accepted_groups += total_i - boundary_i
