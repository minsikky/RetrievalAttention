#!/usr/bin/env python3
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from benchmark.selector_eval.gpu.run_gpu_paged_pq_eval import _sync_if_cuda
from benchmark.selector_eval.runners.hf_paged_pq_intervention_stats import ApproxStats


@dataclass(frozen=True)
class AttentionQKVState:
    query_states: torch.Tensor
    key_states: torch.Tensor
    value_states: torch.Tensor
    keys_all: torch.Tensor
    values_all: torch.Tensor
    q_all: torch.Tensor
    context_len: int
    query_start: int
    num_heads: int
    num_kv_heads: int
    group_size: int


def project_and_update_kv_cache(
    *,
    module: Any,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    cache_obj: Any,
    cache_position: torch.Tensor | None,
    input_shape: torch.Size | tuple[int, ...],
    query_len: int,
    layer_stats: ApproxStats,
    device: torch.device,
    wall_profile_enabled: bool,
    profile_native_ops: bool,
) -> AttentionQKVState:
    qkv_cache_wall_t0 = time.perf_counter() if wall_profile_enabled else 0.0
    if profile_native_ops:
        _sync_if_cuda(device)
        qkv_cache_t0 = time.perf_counter()
    else:
        qkv_cache_t0 = 0.0

    hidden_shape = (*input_shape, -1, module.head_dim)
    query_states = module.q_proj(hidden_states).view(hidden_shape)
    if hasattr(module, "q_norm"):
        query_states = module.q_norm(query_states)
    query_states = query_states.transpose(1, 2)

    key_states = module.k_proj(hidden_states).view(hidden_shape)
    if hasattr(module, "k_norm"):
        key_states = module.k_norm(key_states)
    key_states = key_states.transpose(1, 2)
    value_states = module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
    try:
        key_states, value_states = cache_obj.update(key_states, value_states, module.layer_idx, cache_kwargs)
    except TypeError:
        key_states, value_states = cache_obj.update(key_states, value_states, module.layer_idx)

    if profile_native_ops:
        _sync_if_cuda(device)
        layer_stats.add_qkv_cache_timing(time.perf_counter() - qkv_cache_t0)
    if wall_profile_enabled:
        layer_stats.add_wall_qkv_cache_timing(time.perf_counter() - qkv_cache_wall_t0)

    keys_all = key_states[0].detach()
    values_all = value_states[0].detach()
    q_all = query_states[0].detach().to(torch.float32)
    context_len = int(keys_all.shape[1])
    query_start = context_len - int(query_len)
    num_heads = int(getattr(module, "num_heads", module.config.num_attention_heads))
    num_kv_heads = int(getattr(module, "num_key_value_heads", module.config.num_key_value_heads))
    group_size = int(num_heads // num_kv_heads)

    return AttentionQKVState(
        query_states=query_states,
        key_states=key_states,
        value_states=value_states,
        keys_all=keys_all,
        values_all=values_all,
        q_all=q_all,
        context_len=context_len,
        query_start=query_start,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        group_size=group_size,
    )
