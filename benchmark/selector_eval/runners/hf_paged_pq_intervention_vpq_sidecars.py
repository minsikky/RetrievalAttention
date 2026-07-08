#!/usr/bin/env python3
from __future__ import annotations

import os
from typing import Any

import torch

from benchmark.selector_eval.gpu.run_gpu_paged_pq_eval import GPUIndex, load_selector_paged_pq_ext
from benchmark.selector_eval.runners.hf_paged_pq_intervention_common import _env_int, _env_truthy
from benchmark.selector_eval.runners.hf_paged_pq_intervention_value import (
    value_vpq_code_stat_risk_subset_torch,
    value_vpq_code_stat_risk_torch,
    value_vpq_pack_torch,
    vpq_values_for_tokens_gpu,
)


def joint_vpq_sidecars_for(
state: Any,
    *,
    kv_head: int,
    index: GPUIndex,
    values_t: torch.Tensor,
    context_len_i: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    args = state.args
    module = state.module
    device = state.device
    value_bytes = state.value_bytes
    num_kv_heads = state.num_kv_heads
    layer_id = state.layer_id
    stats = {int(layer_id): state.stats}
    joint_vpq_cache_key_for = state.patch_state.joint_vpq_cache_key_for
    append_exact_suffix_sidecar_inplace = state.append_exact_suffix_sidecar_inplace
    joint_vpq_runtime_cache = state.joint_vpq_runtime_cache
    use_joint_vpq_cache = _env_truthy("SELECTOR_PQ_JOINT_VPQ_CACHE", "1")
    use_persistent_vpq_cache = use_joint_vpq_cache and _env_truthy(
        "SELECTOR_PQ_JOINT_PERSISTENT_VPQ_CACHE",
        "0",
    )
    cache_key = joint_vpq_cache_key_for(int(kv_head), values_t, index)
    # The joint_vpq cache key is derived from page GEOMETRY (kv_head, page
    # counts/starts/size) only -- it does NOT encode the values content. When
    # several sequences share the same sealed-page geometry (e.g. RULER samples
    # of equal length), the persistent/runtime sidecar caches would return the
    # FIRST sequence's V-PQ vhat/residual/code_error for every later sequence,
    # silently garbling all but the first sample. The grouped/native sidecar
    # path avoids this by keying on the freshly-built pack tensors; mirror that
    # here by folding a content fingerprint of the per-sequence V-PQ pack
    # (built fresh per prefill index) into the cache key. Stable across decode
    # steps of one sequence (same cached pack), distinct across sequences.
    if use_joint_vpq_cache:
        try:
            _fp_pack = value_vpq_pack_torch(
                index=index,
                values=values_t,
                value_subvecs=int(args.value_subvecs),
                value_subbits=int(args.value_subbits)
                if int(args.value_subbits) > 0
                else int(args.subbits),
                key_bytes=int(value_bytes),
                device=values_t.device,
            )
        except Exception:
            _fp_pack = None
        if _fp_pack is not None:
            _cb, _cd = _fp_pack[0], _fp_pack[1]
            _sig = (
                torch.stack(
                    [
                        _cb.double().sum(),
                        _cb.double().square().sum(),
                        _cd.double().sum(),
                        _cd.double().square().sum(),
                    ]
                )
                .detach()
                .to("cpu")
                .tolist()
            )
            cache_key = (
                *cache_key,
                tuple(int(v) for v in _cb.shape),
                tuple(int(v) for v in _cd.shape),
                int(_fp_pack[3]),
                int(_fp_pack[4]),
                tuple(float(v) for v in _sig),
            )
    persistent_cache = getattr(module, "_pagedpq_joint_vpq_sidecar_cache", None)
    if use_persistent_vpq_cache and not isinstance(persistent_cache, dict):
        persistent_cache = {}
        setattr(module, "_pagedpq_joint_vpq_sidecar_cache", persistent_cache)
    cached = (
        persistent_cache.get(cache_key)
        if use_persistent_vpq_cache and isinstance(persistent_cache, dict)
        else None
    )
    if cached is not None:
        if len(cached) == 6:
            (
                cached_len,
                cached_capacity,
                vhat_cached,
                residual_cached,
                code_error_cached,
                cached_subbits,
            ) = cached
            cached_capacity_i = int(cached_capacity)
        else:
            cached_len, vhat_cached, residual_cached, code_error_cached, cached_subbits = cached
            cached_capacity_i = int(vhat_cached.shape[0])
        cached_len_i = int(cached_len)
        if cached_len_i >= int(context_len_i):
            return (
                vhat_cached[:context_len_i],
                residual_cached[:context_len_i],
                code_error_cached[:context_len_i],
                int(cached_subbits),
            )
        if cached_len_i >= 0 and cached_len_i < int(context_len_i):
            context_len_target_i = int(context_len_i)
            if cached_capacity_i < context_len_target_i:
                grow_pad_i = max(
                    0,
                    _env_int("SELECTOR_PQ_JOINT_PERSISTENT_VPQ_CACHE_GROW_PAD", 256),
                )
                new_capacity_i = max(
                    context_len_target_i,
                    cached_capacity_i + max(grow_pad_i, context_len_target_i - cached_capacity_i),
                )
                vhat_buf_t = torch.empty(
                    (new_capacity_i, int(vhat_cached.shape[1])),
                    dtype=vhat_cached.dtype,
                    device=vhat_cached.device,
                )
                residual_buf_t = torch.empty(
                    (new_capacity_i, int(residual_cached.shape[1])),
                    dtype=residual_cached.dtype,
                    device=residual_cached.device,
                )
                code_error_buf_t = torch.empty(
                    (new_capacity_i,),
                    dtype=code_error_cached.dtype,
                    device=code_error_cached.device,
                )
                if cached_len_i > 0:
                    vhat_buf_t[:cached_len_i].copy_(vhat_cached[:cached_len_i])
                    residual_buf_t[:cached_len_i].copy_(residual_cached[:cached_len_i])
                    code_error_buf_t[:cached_len_i].copy_(code_error_cached[:cached_len_i])
                vhat_cached = vhat_buf_t
                residual_cached = residual_buf_t
                code_error_cached = code_error_buf_t
                cached_capacity_i = int(new_capacity_i)
            append_exact_suffix_sidecar_inplace(
                vhat_t=vhat_cached,
                residual_t=residual_cached,
                code_error_t=code_error_cached,
                values_t=values_t,
                start=cached_len_i,
                end=context_len_target_i,
            )
            persistent_cache[cache_key] = (
                int(context_len_target_i),
                int(cached_capacity_i),
                vhat_cached,
                residual_cached,
                code_error_cached,
                int(cached_subbits),
            )
            return (
                vhat_cached[:context_len_target_i],
                residual_cached[:context_len_target_i],
                code_error_cached[:context_len_target_i],
                int(cached_subbits),
            )

    use_incremental_vpq_sidecar = _env_truthy(
        "SELECTOR_PQ_JOINT_INCREMENTAL_VPQ_SIDECAR",
        "0",
    )
    if (
        use_incremental_vpq_sidecar
        and use_persistent_vpq_cache
        and isinstance(persistent_cache, dict)
        and persistent_cache
    ):
        best_old_key = None
        best_old_end = -1
        for old_key in persistent_cache:
            if not isinstance(old_key, tuple) or len(old_key) != len(cache_key):
                continue
            if old_key[:6] != cache_key[:6]:
                continue
            if int(old_key[7]) != int(cache_key[7]) or int(old_key[9]) != int(cache_key[9]):
                continue
            old_end_i = int(old_key[8])
            new_end_i = int(cache_key[8])
            if old_end_i < 0 or old_end_i > new_end_i:
                continue
            if old_end_i > best_old_end:
                best_old_end = old_end_i
                best_old_key = old_key
        if best_old_key is not None and best_old_key != cache_key:
            old_cached = persistent_cache.get(best_old_key)
            if old_cached is not None:
                if len(old_cached) == 6:
                    (
                        cached_len,
                        cached_capacity,
                        vhat_cached,
                        residual_cached,
                        code_error_cached,
                        cached_subbits,
                    ) = old_cached
                    cached_capacity_i = int(cached_capacity)
                else:
                    cached_len, vhat_cached, residual_cached, code_error_cached, cached_subbits = old_cached
                    cached_capacity_i = int(vhat_cached.shape[0])
                cached_len_i = int(cached_len)
                context_len_target_i = int(context_len_i)
                if cached_capacity_i < context_len_target_i:
                    grow_pad_i = max(
                        0,
                        _env_int("SELECTOR_PQ_JOINT_PERSISTENT_VPQ_CACHE_GROW_PAD", 256),
                    )
                    new_capacity_i = max(
                        context_len_target_i,
                        cached_capacity_i + max(grow_pad_i, context_len_target_i - cached_capacity_i),
                    )
                    vhat_buf_t = torch.empty(
                        (new_capacity_i, int(vhat_cached.shape[1])),
                        dtype=vhat_cached.dtype,
                        device=vhat_cached.device,
                    )
                    residual_buf_t = torch.empty(
                        (new_capacity_i, int(residual_cached.shape[1])),
                        dtype=residual_cached.dtype,
                        device=residual_cached.device,
                    )
                    code_error_buf_t = torch.empty(
                        (new_capacity_i,),
                        dtype=code_error_cached.dtype,
                        device=code_error_cached.device,
                    )
                    copy_len_i = min(cached_len_i, int(vhat_cached.shape[0]))
                    if copy_len_i > 0:
                        vhat_buf_t[:copy_len_i].copy_(vhat_cached[:copy_len_i])
                        residual_buf_t[:copy_len_i].copy_(residual_cached[:copy_len_i])
                        code_error_buf_t[:copy_len_i].copy_(code_error_cached[:copy_len_i])
                    vhat_cached = vhat_buf_t
                    residual_cached = residual_buf_t
                    code_error_cached = code_error_buf_t
                    cached_capacity_i = int(new_capacity_i)
                old_sealed_end_i = max(0, min(int(best_old_end), context_len_target_i))
                new_sealed_end_i = max(0, min(int(cache_key[8]), context_len_target_i))
                if new_sealed_end_i > old_sealed_end_i:
                    update_tokens_t = torch.arange(
                        old_sealed_end_i,
                        new_sealed_end_i,
                        dtype=torch.long,
                        device=values_t.device,
                    )
                    vhat_update_t, valid_update_t, page_ids_update_t, actual_subbits_i = vpq_values_for_tokens_gpu(
                        index=index,
                        values=values_t,
                        values_np=None,
                        tokens=update_tokens_t,
                        subbits=int(args.subbits),
                        value_subvecs=int(args.value_subvecs),
                        value_subbits=int(args.value_subbits),
                        prefer_torch=True,
                        value_bytes=int(value_bytes),
                    )
                    residual_update_t = values_t.index_select(0, update_tokens_t).float() - vhat_update_t.float()
                    code_error_update_t, actual_subbits_i = value_vpq_code_stat_risk_subset_torch(
                        index=index,
                        values=values_t,
                        tokens=update_tokens_t,
                        residual_subset=residual_update_t,
                        valid=valid_update_t,
                        page_ids=page_ids_update_t,
                        subbits=int(args.subbits),
                        value_subvecs=int(args.value_subvecs),
                        value_subbits=int(args.value_subbits),
                        value_bytes=int(value_bytes),
                    )
                    vhat_cached[old_sealed_end_i:new_sealed_end_i].copy_(
                        vhat_update_t.to(dtype=vhat_cached.dtype)
                    )
                    residual_cached[old_sealed_end_i:new_sealed_end_i].copy_(
                        residual_update_t.to(dtype=residual_cached.dtype)
                    )
                    code_error_cached[old_sealed_end_i:new_sealed_end_i].copy_(
                        code_error_update_t.to(dtype=code_error_cached.dtype)
                    )
                    cached_subbits = int(actual_subbits_i)
                if context_len_target_i > new_sealed_end_i:
                    append_exact_suffix_sidecar_inplace(
                        vhat_t=vhat_cached,
                        residual_t=residual_cached,
                        code_error_t=code_error_cached,
                        values_t=values_t,
                        start=new_sealed_end_i,
                        end=context_len_target_i,
                    )
                persistent_cache[cache_key] = (
                    int(context_len_target_i),
                    int(cached_capacity_i),
                    vhat_cached,
                    residual_cached,
                    code_error_cached,
                    int(cached_subbits),
                )
                max_entries = max(
                    1,
                    int(
                        os.environ.get(
                            "SELECTOR_PQ_JOINT_PERSISTENT_VPQ_CACHE_MAX_ENTRIES",
                            str(max(1, num_kv_heads)),
                        )
                    ),
                )
                while len(persistent_cache) > max_entries:
                    oldest_key = next(iter(persistent_cache))
                    if oldest_key == cache_key and len(persistent_cache) == 1:
                        break
                    persistent_cache.pop(oldest_key, None)
                return (
                    vhat_cached[:context_len_target_i],
                    residual_cached[:context_len_target_i],
                    code_error_cached[:context_len_target_i],
                    int(cached_subbits),
                )

    runtime_key = (*cache_key, int(context_len_i))
    if use_joint_vpq_cache and runtime_key in joint_vpq_runtime_cache:
        return joint_vpq_runtime_cache[runtime_key]

    if _env_truthy("SELECTOR_PQ_JOINT_NATIVE_VPQ_SIDECAR", "0"):
        pack = value_vpq_pack_torch(
            index=index,
            values=values_t,
            value_subvecs=int(args.value_subvecs),
            value_subbits=int(args.value_subbits) if int(args.value_subbits) > 0 else int(args.subbits),
            key_bytes=int(value_bytes),
            device=values_t.device,
        )
        if pack is not None:
            native = load_selector_paged_pq_ext()
            if not hasattr(native, "joint_vpq_sidecars_from_pack"):
                raise RuntimeError(
                    "SELECTOR_PQ_JOINT_NATIVE_VPQ_SIDECAR requires joint_vpq_sidecars_from_pack"
                )
            codebooks, codes, page_starts, _page_size_i, actual_value_subbits_for_cost = pack
            vhat_all_t, residual_t, code_error_t = native.joint_vpq_sidecars_from_pack(
                values_t.contiguous(),
                codebooks.to(dtype=torch.float32).contiguous(),
                codes.contiguous(),
                page_starts.to(dtype=torch.long).contiguous(),
                int(context_len_i),
            )
            out = (
                vhat_all_t.detach(),
                residual_t.detach(),
                code_error_t.detach(),
                int(actual_value_subbits_for_cost),
            )
            if use_joint_vpq_cache:
                joint_vpq_runtime_cache[runtime_key] = out
                if use_persistent_vpq_cache and isinstance(persistent_cache, dict):
                    cache_len_i = int(context_len_i)
                    grow_pad_i = max(
                        0,
                        _env_int("SELECTOR_PQ_JOINT_PERSISTENT_VPQ_CACHE_GROW_PAD", 256),
                    )
                    cache_capacity_i = int(cache_len_i + grow_pad_i)
                    vhat_cached = out[0]
                    residual_cached = out[1]
                    code_error_cached = out[2]
                    if cache_capacity_i > cache_len_i:
                        vhat_buf_t = torch.empty(
                            (cache_capacity_i, int(vhat_cached.shape[1])),
                            dtype=vhat_cached.dtype,
                            device=vhat_cached.device,
                        )
                        residual_buf_t = torch.empty(
                            (cache_capacity_i, int(residual_cached.shape[1])),
                            dtype=residual_cached.dtype,
                            device=residual_cached.device,
                        )
                        code_error_buf_t = torch.empty(
                            (cache_capacity_i,),
                            dtype=code_error_cached.dtype,
                            device=code_error_cached.device,
                        )
                        vhat_buf_t[:cache_len_i].copy_(vhat_cached)
                        residual_buf_t[:cache_len_i].copy_(residual_cached)
                        code_error_buf_t[:cache_len_i].copy_(code_error_cached)
                        vhat_cached = vhat_buf_t
                        residual_cached = residual_buf_t
                        code_error_cached = code_error_buf_t
                    persistent_cache[cache_key] = (
                        int(cache_len_i),
                        int(cache_capacity_i),
                        vhat_cached.detach(),
                        residual_cached.detach(),
                        code_error_cached.detach(),
                        int(out[3]),
                    )
                    max_entries = max(
                        1,
                        int(
                            os.environ.get(
                                "SELECTOR_PQ_JOINT_PERSISTENT_VPQ_CACHE_MAX_ENTRIES",
                                str(max(1, num_kv_heads)),
                            )
                        ),
                    )
                    while len(persistent_cache) > max_entries:
                        oldest_key = next(iter(persistent_cache))
                        if oldest_key == cache_key and len(persistent_cache) == 1:
                            break
                        persistent_cache.pop(oldest_key, None)
            return out

    all_tokens_t = torch.arange(int(context_len_i), dtype=torch.long, device=values_t.device)
    vhat_all_t, vpq_valid_t, vpq_page_ids_t, actual_value_subbits_for_cost = vpq_values_for_tokens_gpu(
        index=index,
        values=values_t,
        values_np=None,
        tokens=all_tokens_t,
        subbits=int(args.subbits),
        value_subvecs=int(args.value_subvecs),
        value_subbits=int(args.value_subbits),
        prefer_torch=True,
        value_bytes=int(value_bytes),
    )
    residual_t = values_t.float() - vhat_all_t.float()
    code_error_t, actual_value_subbits_for_cost = value_vpq_code_stat_risk_torch(
        index=index,
        values=values_t,
        vhat_all=vhat_all_t,
        residual_all=residual_t,
        valid=vpq_valid_t,
        page_ids=vpq_page_ids_t,
        subbits=int(args.subbits),
        value_subvecs=int(args.value_subvecs),
        value_subbits=int(args.value_subbits),
        value_bytes=int(value_bytes),
    )
    out = (
        vhat_all_t.detach(),
        residual_t.detach(),
        code_error_t.detach(),
        int(actual_value_subbits_for_cost),
    )
    if use_joint_vpq_cache:
        joint_vpq_runtime_cache[runtime_key] = out
        if use_persistent_vpq_cache and isinstance(persistent_cache, dict):
            cache_len_i = int(context_len_i)
            grow_pad_i = max(
                0,
                _env_int("SELECTOR_PQ_JOINT_PERSISTENT_VPQ_CACHE_GROW_PAD", 256),
            )
            cache_capacity_i = int(cache_len_i + grow_pad_i)
            vhat_cached = out[0]
            residual_cached = out[1]
            code_error_cached = out[2]
            if cache_capacity_i > cache_len_i:
                vhat_buf_t = torch.empty(
                    (cache_capacity_i, int(vhat_cached.shape[1])),
                    dtype=vhat_cached.dtype,
                    device=vhat_cached.device,
                )
                residual_buf_t = torch.empty(
                    (cache_capacity_i, int(residual_cached.shape[1])),
                    dtype=residual_cached.dtype,
                    device=residual_cached.device,
                )
                code_error_buf_t = torch.empty(
                    (cache_capacity_i,),
                    dtype=code_error_cached.dtype,
                    device=code_error_cached.device,
                )
                vhat_buf_t[:cache_len_i].copy_(vhat_cached)
                residual_buf_t[:cache_len_i].copy_(residual_cached)
                code_error_buf_t[:cache_len_i].copy_(code_error_cached)
                vhat_cached = vhat_buf_t
                residual_cached = residual_buf_t
                code_error_cached = code_error_buf_t
            persistent_cache[cache_key] = (
                int(cache_len_i),
                int(cache_capacity_i),
                vhat_cached.detach(),
                residual_cached.detach(),
                code_error_cached.detach(),
                int(out[3]),
            )
            max_entries = max(
                1,
                int(
                    os.environ.get(
                        "SELECTOR_PQ_JOINT_PERSISTENT_VPQ_CACHE_MAX_ENTRIES",
                        str(max(1, num_kv_heads)),
                    )
                ),
            )
            while len(persistent_cache) > max_entries:
                oldest_key = next(iter(persistent_cache))
                if oldest_key == cache_key and len(persistent_cache) == 1:
                    break
                persistent_cache.pop(oldest_key, None)
    return out

