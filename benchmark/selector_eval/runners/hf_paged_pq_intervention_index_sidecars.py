#!/usr/bin/env python3
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch

from benchmark.selector_eval.gpu.run_gpu_paged_pq_eval import GPUIndex, _sync_if_cuda
from benchmark.selector_eval.runners.hf_paged_pq_intervention_common import build_page_pq_from_keys
from benchmark.selector_eval.runners.hf_paged_pq_intervention_stats import ApproxStats


@dataclass
class DecodeIndexSidecars:
    index_cache: dict[int, GPUIndex]
    prefix_index_cache: dict[tuple[int, int, int], GPUIndex] = field(default_factory=dict)
    torch_k_cache: dict[int, torch.Tensor] = field(default_factory=dict)
    torch_v_cache: dict[int, torch.Tensor] = field(default_factory=dict)
    dynamic_start: int = 0
    indexed_end: int = 0
    sealed_end: int = 0


def _clear_index_native_sidecars(index: GPUIndex) -> None:
    index.native_codebooks = None
    index.native_codes = None
    index.native_page_starts = None
    for attr in (
        "_value_vpq_gpu_pack_by_params",
        "_all_value_vpq_gpu_by_params",
        "_value_vpq_sidecars_by_params",
    ):
        if hasattr(index, attr):
            delattr(index, attr)


def build_decode_index_sidecars(
    *,
    args: Any,
    module: Any,
    layer_stats: ApproxStats,
    device: torch.device,
    keys_all: torch.Tensor,
    values_all: torch.Tensor,
    context_len: int,
    query_len: int,
    num_kv_heads: int,
    online_confidence_rule: str,
    key_bytes: int,
    wall_profile_enabled: bool,
) -> DecodeIndexSidecars:
    context_len = int(context_len)
    num_kv_heads = int(num_kv_heads)
    dynamic_start = min(max(0, int(args.static_prefix)), context_len)
    indexed_end = max(dynamic_start, context_len - max(0, int(args.static_suffix)))
    sealed_end = dynamic_start + (
        (max(0, indexed_end - dynamic_start) // max(1, int(args.page_size))) * max(1, int(args.page_size))
    )

    joint_fast_decode_index_possible = (
        int(query_len) == 1
        and online_confidence_rule == "joint_kv_stability"
        and str(args.selector_backend) in {"cuda_ext", "auto"}
        and str(args.selector_mode) == "fullscan"
        and str(getattr(args, "index_build_backend", "numpy")) == "torch_gpu"
    )

    page_cache = getattr(module, "_pagedpq_page_cache", None)
    if not isinstance(page_cache, dict):
        page_cache = {}
        setattr(module, "_pagedpq_page_cache", page_cache)

    fast_decode_index_cache_key: tuple[object, ...] | None = None
    fast_decode_cached_indexes: tuple[GPUIndex, ...] | None = None
    if (
        int(query_len) == 1
        and joint_fast_decode_index_possible
        and str(args.selector_backend) == "cuda_ext"
        and str(args.selector_mode) == "fullscan"
    ):
        fast_decode_index_cache_key = (
            "fullscan_decode",
            str(online_confidence_rule),
            int(dynamic_start),
            int(sealed_end),
            int(args.page_size),
            int(args.subvecs),
            int(args.subbits),
            int(args.kmeans_iters),
            int(args.seed),
            int(key_bytes),
            str(getattr(args, "index_build_backend", "numpy")),
            str(args.selected_value_mode),
            str(args.tail_mode),
            int(args.value_subvecs),
            int(args.value_subbits),
            int(args.value_pq_group_pages),
            int(num_kv_heads),
        )
        fast_decode_index_cache = getattr(module, "_pagedpq_fast_decode_index_cache", None)
        if isinstance(fast_decode_index_cache, dict):
            cached_indexes = fast_decode_index_cache.get(fast_decode_index_cache_key)
            if isinstance(cached_indexes, tuple) and len(cached_indexes) == int(num_kv_heads):
                fast_decode_cached_indexes = cached_indexes

    index_cache: dict[int, GPUIndex] = {}
    torch_k_cache: dict[int, torch.Tensor] = {}
    torch_v_cache: dict[int, torch.Tensor] = {}

    index_sidecar_wall_t0 = time.perf_counter() if wall_profile_enabled else 0.0
    if bool(getattr(args, "profile_native_ops", False)):
        _sync_if_cuda(device)
        sidecar_t0 = time.perf_counter()
    else:
        sidecar_t0 = 0.0

    if fast_decode_cached_indexes is not None:
        for kv_head in range(num_kv_heads):
            cached_index = fast_decode_cached_indexes[int(kv_head)]
            cached_index.pending_start = int(sealed_end)
            cached_index.indexed_end = int(indexed_end)
            index_cache[int(kv_head)] = cached_index
        for kv_head in range(num_kv_heads):
            torch_k_cache[int(kv_head)] = keys_all[int(kv_head)].to(device)
            torch_v_cache[int(kv_head)] = values_all[int(kv_head)].to(device)
    else:
        for kv_head in range(num_kv_heads):
            cache_key = (
                "online_fullscan",
                int(kv_head),
                int(dynamic_start),
                int(args.page_size),
                int(args.subvecs),
                int(args.subbits),
                int(args.kmeans_iters),
                int(args.seed),
                int(key_bytes),
                str(getattr(args, "index_build_backend", "numpy")),
            )
            cached_index = page_cache.get(cache_key)
            if cached_index is None or int(cached_index.pending_start) > int(sealed_end):
                cached_index, build_seconds, build_read_mb, build_write_mb = build_page_pq_from_keys(
                    keys_all[kv_head],
                    args=args,
                    kv_head=kv_head,
                    dynamic_start=dynamic_start,
                    indexed_end=sealed_end,
                    key_bytes=key_bytes,
                    router_enabled=False,
                    device=device,
                )
                layer_stats.add_index_build(build_seconds, build_read_mb, build_write_mb)
                page_cache[cache_key] = cached_index
            elif int(cached_index.pending_start) < int(sealed_end):
                new_index, build_seconds, build_read_mb, build_write_mb = build_page_pq_from_keys(
                    keys_all[kv_head],
                    args=args,
                    kv_head=kv_head,
                    dynamic_start=int(cached_index.pending_start),
                    indexed_end=sealed_end,
                    key_bytes=key_bytes,
                    router_enabled=False,
                    device=device,
                    page_id_offset=len(cached_index.pages),
                )
                layer_stats.add_index_build(build_seconds, build_read_mb, build_write_mb)
                cached_index.pages.extend(new_index.pages)
                cached_index.pending_start = int(sealed_end)
                cached_index.indexed_end = int(sealed_end)
                cached_index.build_seconds += float(new_index.build_seconds)
                cached_index.build_read_mb += float(new_index.build_read_mb)
                cached_index.build_write_mb += float(new_index.build_write_mb)
                _clear_index_native_sidecars(cached_index)
            cached_index.pending_start = int(sealed_end)
            cached_index.indexed_end = int(indexed_end)
            index_cache[int(kv_head)] = cached_index
            torch_k_cache[int(kv_head)] = keys_all[kv_head].to(device)
            torch_v_cache[int(kv_head)] = values_all[kv_head].to(device)
        if fast_decode_index_cache_key is not None:
            fast_decode_index_cache = getattr(module, "_pagedpq_fast_decode_index_cache", None)
            if not isinstance(fast_decode_index_cache, dict):
                fast_decode_index_cache = {}
                setattr(module, "_pagedpq_fast_decode_index_cache", fast_decode_index_cache)
            fast_decode_index_cache.clear()
            fast_decode_index_cache[fast_decode_index_cache_key] = tuple(
                index_cache[int(kv_head)] for kv_head in range(num_kv_heads)
            )

    if bool(getattr(args, "profile_native_ops", False)):
        _sync_if_cuda(device)
        layer_stats.add_index_sidecar_timing(time.perf_counter() - sidecar_t0)
    if wall_profile_enabled:
        layer_stats.add_wall_index_sidecar_timing(time.perf_counter() - index_sidecar_wall_t0)

    return DecodeIndexSidecars(
        index_cache=index_cache,
        prefix_index_cache={},
        torch_k_cache=torch_k_cache,
        torch_v_cache=torch_v_cache,
        dynamic_start=dynamic_start,
        indexed_end=indexed_end,
        sealed_end=sealed_end,
    )
