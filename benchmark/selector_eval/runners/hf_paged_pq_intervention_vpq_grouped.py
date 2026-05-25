#!/usr/bin/env python3
from __future__ import annotations

from typing import Any

import torch

from benchmark.selector_eval.gpu.run_gpu_paged_pq_eval import GPUIndex
from benchmark.selector_eval.runners.hf_paged_pq_intervention_common import _env_int, _env_truthy


def grouped_vpq_residual_sidecars_for(
    state: Any,
    gqa_indexes: list[GPUIndex],
    *,
    context_len_i: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int] | None:
    if not _env_truthy("SELECTOR_PQ_JOINT_GROUPED_VPQ_CACHE", "0"):
        return None
    if len(gqa_indexes) != int(state.num_kv_heads):
        return None

    args = state.args
    context_len_i = int(context_len_i)
    num_kv_heads = int(state.num_kv_heads)
    values_all = state.values_all
    joint_vpq_cache_key_for = state.patch_state.joint_vpq_cache_key_for
    group_key = (
        tuple(
            joint_vpq_cache_key_for(
                int(kv_head),
                values_all[int(kv_head)][:context_len_i],
                gqa_indexes[int(kv_head)],
            )
            for kv_head in range(num_kv_heads)
        ),
        num_kv_heads,
    )
    grouped_cache = getattr(state.module, "_pagedpq_joint_grouped_vpq_sidecar_cache", None)
    if not isinstance(grouped_cache, dict):
        grouped_cache = {}
        setattr(state.module, "_pagedpq_joint_grouped_vpq_sidecar_cache", grouped_cache)
    cached = grouped_cache.get(group_key)
    if cached is not None:
        if len(cached) != 6:
            grouped_cache.pop(group_key, None)
        else:
            (
                cached_len,
                cached_capacity,
                vhat_groups_t,
                residual_groups_t,
                code_error_groups_t,
                cached_subbits,
            ) = cached
            cached_capacity_i = int(cached_capacity)
            cached_len_i = int(cached_len)
            if cached_len_i >= context_len_i:
                return (
                    vhat_groups_t[:, :context_len_i, :],
                    residual_groups_t[:, :context_len_i, :],
                    code_error_groups_t[:, :context_len_i],
                    int(cached_subbits),
                )
            if cached_len_i >= 0 and cached_len_i < context_len_i:
                if cached_capacity_i < context_len_i:
                    grow_pad_i = max(
                        0,
                        _env_int("SELECTOR_PQ_JOINT_PERSISTENT_VPQ_CACHE_GROW_PAD", 256),
                    )
                    new_capacity_i = max(
                        context_len_i,
                        cached_capacity_i + max(grow_pad_i, context_len_i - cached_capacity_i),
                    )
                    vhat_buf_t = torch.empty(
                        (
                            num_kv_heads,
                            int(new_capacity_i),
                            int(vhat_groups_t.shape[2]),
                        ),
                        dtype=vhat_groups_t.dtype,
                        device=vhat_groups_t.device,
                    )
                    residual_buf_t = torch.empty(
                        (
                            num_kv_heads,
                            int(new_capacity_i),
                            int(residual_groups_t.shape[2]),
                        ),
                        dtype=residual_groups_t.dtype,
                        device=residual_groups_t.device,
                    )
                    code_error_buf_t = torch.empty(
                        (num_kv_heads, int(new_capacity_i)),
                        dtype=code_error_groups_t.dtype,
                        device=code_error_groups_t.device,
                    )
                    if cached_len_i > 0:
                        vhat_buf_t[:, :cached_len_i, :].copy_(vhat_groups_t[:, :cached_len_i, :])
                        residual_buf_t[:, :cached_len_i, :].copy_(residual_groups_t[:, :cached_len_i, :])
                        code_error_buf_t[:, :cached_len_i].copy_(code_error_groups_t[:, :cached_len_i])
                    vhat_groups_t = vhat_buf_t
                    residual_groups_t = residual_buf_t
                    code_error_groups_t = code_error_buf_t
                    cached_capacity_i = int(new_capacity_i)
                used_grouped_append = state.append_exact_suffix_grouped_sidecar_inplace(
                    vhat_t=vhat_groups_t,
                    residual_t=residual_groups_t,
                    code_error_t=code_error_groups_t,
                    values_t=values_all,
                    start=cached_len_i,
                    end=context_len_i,
                )
                if not used_grouped_append:
                    for kv_head in range(num_kv_heads):
                        state.append_exact_suffix_sidecar_inplace(
                            vhat_t=vhat_groups_t[int(kv_head)],
                            residual_t=residual_groups_t[int(kv_head)],
                            code_error_t=code_error_groups_t[int(kv_head)],
                            values_t=values_all[int(kv_head)][:context_len_i],
                            start=cached_len_i,
                            end=context_len_i,
                        )
                grouped_cache[group_key] = (
                    context_len_i,
                    int(cached_capacity_i),
                    vhat_groups_t,
                    residual_groups_t,
                    code_error_groups_t,
                    int(cached_subbits),
                )
                return (
                    vhat_groups_t[:, :context_len_i, :],
                    residual_groups_t[:, :context_len_i, :],
                    code_error_groups_t[:, :context_len_i],
                    int(cached_subbits),
                )

    vhat_parts: list[torch.Tensor] = []
    residual_parts: list[torch.Tensor] = []
    code_error_parts: list[torch.Tensor] = []
    actual_subbits = int(args.value_subbits) if int(args.value_subbits) > 0 else int(args.subbits)
    for kv_head in range(num_kv_heads):
        _vhat_t, residual_t, code_error_t, actual_subbits_i = state.joint_vpq_sidecars_for(
            kv_head=int(kv_head),
            index=gqa_indexes[int(kv_head)],
            values_t=values_all[int(kv_head)][:context_len_i],
            context_len_i=context_len_i,
        )
        vhat_parts.append(_vhat_t.to(dtype=torch.float32))
        residual_parts.append(residual_t.to(dtype=torch.float32))
        code_error_parts.append(code_error_t.to(dtype=torch.float32))
        actual_subbits = int(actual_subbits_i)
    vhat_groups_t = torch.stack(vhat_parts, dim=0).contiguous()
    residual_groups_t = torch.stack(residual_parts, dim=0).contiguous()
    code_error_groups_t = torch.stack(code_error_parts, dim=0).contiguous()
    grow_pad_i = max(
        0,
        _env_int("SELECTOR_PQ_JOINT_PERSISTENT_VPQ_CACHE_GROW_PAD", 256),
    )
    cache_capacity_i = int(context_len_i + grow_pad_i)
    if cache_capacity_i > context_len_i:
        vhat_buf_t = torch.empty(
            (
                num_kv_heads,
                int(cache_capacity_i),
                int(vhat_groups_t.shape[2]),
            ),
            dtype=vhat_groups_t.dtype,
            device=vhat_groups_t.device,
        )
        residual_buf_t = torch.empty(
            (
                num_kv_heads,
                int(cache_capacity_i),
                int(residual_groups_t.shape[2]),
            ),
            dtype=residual_groups_t.dtype,
            device=residual_groups_t.device,
        )
        code_error_buf_t = torch.empty(
            (num_kv_heads, int(cache_capacity_i)),
            dtype=code_error_groups_t.dtype,
            device=code_error_groups_t.device,
        )
        vhat_buf_t[:, :context_len_i, :].copy_(vhat_groups_t)
        residual_buf_t[:, :context_len_i, :].copy_(residual_groups_t)
        code_error_buf_t[:, :context_len_i].copy_(code_error_groups_t)
        vhat_groups_t = vhat_buf_t
        residual_groups_t = residual_buf_t
        code_error_groups_t = code_error_buf_t
    if grouped_cache:
        grouped_cache.clear()
    grouped_cache[group_key] = (
        context_len_i,
        int(cache_capacity_i),
        vhat_groups_t,
        residual_groups_t,
        code_error_groups_t,
        int(actual_subbits),
    )
    return (
        vhat_groups_t[:, :context_len_i, :],
        residual_groups_t[:, :context_len_i, :],
        code_error_groups_t[:, :context_len_i],
        int(actual_subbits),
    )
