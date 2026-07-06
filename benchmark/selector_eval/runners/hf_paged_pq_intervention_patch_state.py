#!/usr/bin/env python3
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from benchmark.selector_eval.data.trace import static_tokens, unique_tokens
from benchmark.selector_eval.gpu.run_gpu_paged_pq_eval import GPUIndex
from benchmark.selector_eval.runners.hf_paged_pq_intervention_common import (
    _env_int,
    _env_truthy,
    build_page_pq_from_keys,
    cache_layer_kv_tensors,
)
from benchmark.selector_eval.runners.hf_paged_pq_intervention_geometric import geometric_budget_pairs
from benchmark.selector_eval.runners.hf_paged_pq_intervention_stats import ApproxStats
from benchmark.selector_eval.runners.hf_paged_pq_intervention_value import (
    value_vpq_code_stat_risk_torch,
    value_vpq_pack_torch,
    vpq_values_for_tokens_gpu,
)


@dataclass
class PagedPQPatchState:
    args: Any
    layer_ids: list[int]
    device: torch.device
    stats: dict[int, ApproxStats]
    key_bytes: int
    value_bytes: int
    online_confidence_rule: str
    last_decode_base_key: tuple[int, int, int, int, int] | None = None
    last_decode_base_tensor: torch.Tensor | None = None
    last_decode_rank_ids_tensors: dict[tuple[str, int, int], torch.Tensor] = field(default_factory=dict)
    geometric_budget_column_tensors: dict[
        tuple[str, int, int, int, float, float],
        tuple[list[int], list[int], list[int], torch.Tensor, torch.Tensor],
    ] = field(default_factory=dict)
    dense_decode_key_t_cache: dict[int, dict[str, object]] = field(default_factory=dict)
    dense_decode_key_t_cache_max_bytes: int = field(init=False)
    dense_decode_key_t_cache_enabled: bool = field(init=False)

    def __post_init__(self) -> None:
        try:
            self.dense_decode_key_t_cache_max_bytes = int(
                max(0.0, float(os.environ.get("FRONTIER_DENSE_KEY_T_CACHE_MAX_GB", "12.0")))
                * 1024.0
                * 1024.0
                * 1024.0
            )
        except ValueError:
            self.dense_decode_key_t_cache_max_bytes = 12 * 1024 * 1024 * 1024
        self.dense_decode_key_t_cache_enabled = _env_truthy("FRONTIER_DENSE_KEY_T_CACHE", "1")

    def dense_decode_key_t_float_cache(
        self,
        *,
        layer_id: int,
        keys_all: torch.Tensor,
        key_count: int,
    ) -> torch.Tensor | None:
        if (
            not self.dense_decode_key_t_cache_enabled
            or self.dense_decode_key_t_cache_max_bytes <= 0
            or keys_all.device.type != "cuda"
        ):
            return None
        kv_heads = int(keys_all.shape[0])
        capacity = int(keys_all.shape[1])
        dim = int(keys_all.shape[2])
        key_count_i = min(max(0, int(key_count)), capacity)
        if kv_heads <= 0 or capacity <= 0 or dim <= 0 or key_count_i <= 0:
            return None
        # This cache is a GPU-simulator optimization only. Keep a conservative
        # cap so long-context task runs do not OOM by caching every layer's
        # float-transposed K unless the user explicitly raises the limit.
        all_layers_bytes = int(max(1, len(self.layer_ids))) * kv_heads * capacity * dim * 4
        if all_layers_bytes > self.dense_decode_key_t_cache_max_bytes:
            return None
        entry = self.dense_decode_key_t_cache.get(int(layer_id))
        data_ptr = int(keys_all.data_ptr())
        shape = (kv_heads, capacity, dim)
        if (
            entry is None
            or int(entry.get("data_ptr", -1)) != data_ptr
            or tuple(entry.get("shape", ())) != shape
            or str(entry.get("device", "")) != str(keys_all.device)
        ):
            entry = {
                "data_ptr": data_ptr,
                "shape": shape,
                "device": str(keys_all.device),
                "filled": 0,
                "tensor": torch.empty((kv_heads, dim, capacity), dtype=torch.float32, device=keys_all.device),
            }
            self.dense_decode_key_t_cache[int(layer_id)] = entry
        filled = int(entry.get("filled", 0))
        if key_count_i < filled:
            filled = 0
        if key_count_i > filled:
            cached = entry["tensor"]
            assert isinstance(cached, torch.Tensor)
            cached[:, :, filled:key_count_i].copy_(
                keys_all[:, filled:key_count_i, :].float().transpose(1, 2).contiguous()
            )
            entry["filled"] = key_count_i
        cached = entry["tensor"]
        assert isinstance(cached, torch.Tensor)
        return cached

    def decode_base_tokens_tensor(self, query_context_len: int, sealed_end: int, indexed_end: int) -> torch.Tensor:
        cache_key = (
            int(query_context_len),
            int(sealed_end),
            int(indexed_end),
            int(self.args.static_prefix),
            int(self.args.static_suffix),
        )
        if self.last_decode_base_key == cache_key and self.last_decode_base_tensor is not None:
            return self.last_decode_base_tensor
        base = unique_tokens(
            static_tokens(int(query_context_len) - 1, int(self.args.static_prefix), int(self.args.static_suffix))
            + list(range(max(0, int(sealed_end)), max(0, min(int(indexed_end), int(query_context_len))))),
            context_len=int(query_context_len),
        )
        self.last_decode_base_key = cache_key
        self.last_decode_base_tensor = torch.as_tensor(
            np.asarray(base, dtype=np.int64),
            dtype=torch.long,
            device=self.device,
        )
        return self.last_decode_base_tensor

    def decode_rank_ids_tensor(self, rank_count: int, tensor_device: torch.device, *, dims: int = 2) -> torch.Tensor:
        dims_i = int(dims)
        if dims_i not in {2, 3}:
            raise ValueError(f"unsupported rank id dims: {dims}")
        cache_key = (str(tensor_device), int(rank_count), dims_i)
        cached = self.last_decode_rank_ids_tensors.get(cache_key)
        if cached is not None:
            return cached
        shape = (1, int(rank_count)) if dims_i == 2 else (1, 1, int(rank_count))
        tensor = torch.arange(
            int(rank_count),
            dtype=torch.long,
            device=tensor_device,
        ).reshape(*shape)
        self.last_decode_rank_ids_tensors[cache_key] = tensor
        return tensor

    def geometric_threshold_budget_columns(
        self,
        *,
        min_budget: int,
        max_budget: int,
        granularity: int,
        growth: float,
        probe_scale: float,
        tensor_device: torch.device,
    ) -> tuple[list[int], list[int], list[int], torch.Tensor, torch.Tensor]:
        cache_key = (
            str(tensor_device),
            int(min_budget),
            int(max_budget),
            int(granularity),
            float(growth),
            float(probe_scale),
        )
        cached = self.geometric_budget_column_tensors.get(cache_key)
        if cached is not None:
            return cached
        tail_budgets, probe_budgets = geometric_budget_pairs(
            min_budget=int(min_budget),
            max_budget=int(max_budget),
            granularity=int(granularity),
            growth=float(growth),
            probe_scale=float(probe_scale),
        )
        combined_budgets = sorted({int(v) for v in tail_budgets} | {int(v) for v in probe_budgets})
        budget_to_col = {int(budget): int(idx) for idx, budget in enumerate(combined_budgets)}
        approx_cols = torch.tensor(
            [budget_to_col[int(v)] for v in tail_budgets],
            dtype=torch.long,
            device=tensor_device,
        )
        probe_cols = torch.tensor(
            [budget_to_col[int(v)] for v in probe_budgets],
            dtype=torch.long,
            device=tensor_device,
        )
        cached = (tail_budgets, probe_budgets, combined_budgets, approx_cols, probe_cols)
        self.geometric_budget_column_tensors[cache_key] = cached
        return cached

    def decode_base_token_count(self, query_context_len: int, sealed_end: int) -> int:
        prefix_end = min(max(0, int(self.args.static_prefix)), int(query_context_len))
        base_tail_start = max(int(sealed_end), int(prefix_end))
        return int(prefix_end) + max(0, int(query_context_len) - int(base_tail_start))

    def joint_vpq_cache_key_for(self, kv_head: int, values_t: torch.Tensor, index: GPUIndex) -> tuple[object, ...]:
        actual_value_subbits_key = (
            int(self.args.value_subbits) if int(self.args.value_subbits) > 0 else int(self.args.subbits)
        )
        return (
            int(kv_head),
            str(values_t.device),
            int(self.args.subbits),
            int(self.args.value_subvecs),
            int(actual_value_subbits_key),
            int(self.value_bytes),
            int(len(index.pages)),
            int(index.pages[0].start) if index.pages else -1,
            (int(index.pages[-1].start) + int(index.pages[-1].size)) if index.pages else -1,
            int(index.pages[0].size) if index.pages else 0,
        )

    def warm_dense_prefill_decode_sidecars(self, layer_id: int, module, cache_obj) -> None:
        args = self.args
        if bool(getattr(args, "skip_prefill_index_build", False)):
            return
        if str(args.selector_mode) not in {"fullscan", "routed", "oracle"}:
            return
        num_kv_heads = getattr(module, "num_key_value_heads", None)
        if num_kv_heads is None:
            config = getattr(module, "config", None)
            num_kv_heads = getattr(config, "num_key_value_heads", None)
        if num_kv_heads is None:
            return
        num_kv_heads = int(num_kv_heads)
        kv = cache_layer_kv_tensors(cache_obj, int(layer_id), num_kv_heads=num_kv_heads)
        if kv is None:
            return
        keys_all, values_all = kv
        if keys_all.ndim != 3 or values_all.ndim != 3 or int(keys_all.shape[0]) != num_kv_heads:
            return
        context_len = int(keys_all.shape[1])
        dynamic_start = min(max(0, int(args.static_prefix)), context_len)
        indexed_end = max(dynamic_start, context_len - max(0, int(args.static_suffix)))
        sealed_end = dynamic_start + (
            (max(0, indexed_end - dynamic_start) // max(1, int(args.page_size))) * max(1, int(args.page_size))
        )
        page_cache = getattr(module, "_pagedpq_page_cache", None)
        if not isinstance(page_cache, dict):
            page_cache = {}
            setattr(module, "_pagedpq_page_cache", page_cache)
        decode_tail_blend = (
            float(args.decode_tail_blend)
            if getattr(args, "decode_tail_blend", None) is not None
            else float(args.tail_blend)
        )
        needs_vpq_sidecar = (
            str(args.selected_value_mode) == "vpq_value"
            or (float(decode_tail_blend) > 0.0 and str(args.tail_mode) == "vpq_value")
        )
        warmed_indexes: list[GPUIndex | None] = [None] * num_kv_heads
        for kv_head in range(num_kv_heads):
            if str(args.selector_mode) in {"fullscan", "oracle"}:
                cache_key = (
                    "online_fullscan",
                    int(kv_head),
                    int(dynamic_start),
                    int(args.page_size),
                    int(args.subvecs),
                    int(args.subbits),
                    int(args.kmeans_iters),
                    int(args.seed),
                    int(self.key_bytes),
                    str(getattr(args, "index_build_backend", "numpy")),
                )
            else:
                cache_key = (
                    int(kv_head),
                    int(dynamic_start),
                    int(sealed_end),
                    int(args.page_size),
                    int(args.subvecs),
                    int(args.subbits),
                    int(args.kmeans_iters),
                    int(args.seed),
                    int(self.key_bytes),
                    str(getattr(args, "index_build_backend", "numpy")),
                    str(args.selector_mode),
                    int(args.router_prototypes),
                    float(args.router_merge_rel),
                    float(args.router_merge_var),
                    int(args.router_max_groups),
                )
            cached_index = page_cache.get(cache_key)
            if cached_index is None:
                cached_index, build_seconds, build_read_mb, build_write_mb = build_page_pq_from_keys(
                    keys_all[int(kv_head)].detach(),
                    args=args,
                    kv_head=int(kv_head),
                    dynamic_start=dynamic_start,
                    indexed_end=sealed_end,
                    key_bytes=self.key_bytes,
                    router_enabled=str(args.selector_mode) == "routed",
                    device=self.device,
                )
                self.stats[int(layer_id)].add_index_build(build_seconds, build_read_mb, build_write_mb)
                page_cache[cache_key] = cached_index
            elif str(args.selector_mode) in {"fullscan", "oracle"} and int(cached_index.pending_start) < int(sealed_end):
                new_index, build_seconds, build_read_mb, build_write_mb = build_page_pq_from_keys(
                    keys_all[int(kv_head)].detach(),
                    args=args,
                    kv_head=int(kv_head),
                    dynamic_start=int(cached_index.pending_start),
                    indexed_end=sealed_end,
                    key_bytes=self.key_bytes,
                    router_enabled=False,
                    device=self.device,
                    page_id_offset=len(cached_index.pages),
                )
                self.stats[int(layer_id)].add_index_build(build_seconds, build_read_mb, build_write_mb)
                cached_index.pages.extend(new_index.pages)
                cached_index.pending_start = int(sealed_end)
                cached_index.indexed_end = int(sealed_end)
                cached_index.build_seconds += float(new_index.build_seconds)
                cached_index.build_read_mb += float(new_index.build_read_mb)
                cached_index.build_write_mb += float(new_index.build_write_mb)
                cached_index.native_codebooks = None
                cached_index.native_codes = None
                cached_index.native_page_starts = None
                for attr in (
                    "_value_vpq_gpu_pack_by_params",
                    "_all_value_vpq_gpu_by_params",
                    "_value_vpq_sidecars_by_params",
                ):
                    if hasattr(cached_index, attr):
                        delattr(cached_index, attr)
            if str(args.selector_mode) in {"fullscan", "oracle"}:
                cached_index.pending_start = int(sealed_end)
                cached_index.indexed_end = int(indexed_end)
                warmed_indexes[int(kv_head)] = cached_index
            if (
                needs_vpq_sidecar
                and str(getattr(args, "index_build_backend", "numpy")) == "torch_gpu"
                and cached_index.pages
            ):
                value_vpq_pack_torch(
                    index=cached_index,
                    values=values_all[int(kv_head)].detach(),
                    value_subvecs=int(args.value_subvecs),
                    value_subbits=int(args.value_subbits) if int(args.value_subbits) > 0 else int(args.subbits),
                    key_bytes=self.value_bytes,
                    device=self.device,
                    value_group_pages=int(getattr(args, "value_pq_group_pages", 1)),
                )
                build_stats = getattr(cached_index, "_last_value_vpq_build_stats", None)
                if build_stats is not None:
                    self.stats[int(layer_id)].add_index_build(*build_stats)
                if _env_truthy(
                    "SELECTOR_PQ_JOINT_PREWARM_VPQ_SIDECARS",
                    "0",
                ) and not _env_truthy("SELECTOR_PQ_JOINT_COMPACT_VPQ_RISK_PREFIX", "0"):
                    persistent_cache = getattr(module, "_pagedpq_joint_vpq_sidecar_cache", None)
                    if not isinstance(persistent_cache, dict):
                        persistent_cache = {}
                        setattr(module, "_pagedpq_joint_vpq_sidecar_cache", persistent_cache)
                    values_t = values_all[int(kv_head)].detach()
                    context_len_i = int(values_t.shape[0])
                    cache_key = self.joint_vpq_cache_key_for(int(kv_head), values_t, cached_index)
                    if cache_key not in persistent_cache:
                        sidecar_t0 = time.perf_counter()
                        all_tokens_t = torch.arange(context_len_i, dtype=torch.long, device=values_t.device)
                        vhat_all_t, vpq_valid_t, vpq_page_ids_t, actual_value_subbits_for_cost = vpq_values_for_tokens_gpu(
                            index=cached_index,
                            values=values_t,
                            values_np=None,
                            tokens=all_tokens_t,
                            subbits=int(args.subbits),
                            value_subvecs=int(args.value_subvecs),
                            value_subbits=int(args.value_subbits),
                            prefer_torch=True,
                            value_bytes=int(self.value_bytes),
                        )
                        residual_t = values_t.float() - vhat_all_t.float()
                        code_error_t, actual_value_subbits_for_cost = value_vpq_code_stat_risk_torch(
                            index=cached_index,
                            values=values_t,
                            vhat_all=vhat_all_t,
                            residual_all=residual_t,
                            valid=vpq_valid_t,
                            page_ids=vpq_page_ids_t,
                            subbits=int(args.subbits),
                            value_subvecs=int(args.value_subvecs),
                            value_subbits=int(args.value_subbits),
                            value_bytes=int(self.value_bytes),
                        )
                        cache_len_i = int(context_len_i)
                        grow_pad_i = max(
                            0,
                            _env_int("SELECTOR_PQ_JOINT_PERSISTENT_VPQ_CACHE_GROW_PAD", 256),
                        )
                        cache_capacity_i = int(cache_len_i + grow_pad_i)
                        if cache_capacity_i > cache_len_i:
                            vhat_buf_t = torch.empty(
                                (cache_capacity_i, int(vhat_all_t.shape[1])),
                                dtype=vhat_all_t.dtype,
                                device=vhat_all_t.device,
                            )
                            residual_buf_t = torch.empty(
                                (cache_capacity_i, int(residual_t.shape[1])),
                                dtype=residual_t.dtype,
                                device=residual_t.device,
                            )
                            code_error_buf_t = torch.empty(
                                (cache_capacity_i,),
                                dtype=code_error_t.dtype,
                                device=code_error_t.device,
                            )
                            vhat_buf_t[:cache_len_i].copy_(vhat_all_t)
                            residual_buf_t[:cache_len_i].copy_(residual_t)
                            code_error_buf_t[:cache_len_i].copy_(code_error_t)
                            vhat_all_t = vhat_buf_t
                            residual_t = residual_buf_t
                            code_error_t = code_error_buf_t
                        if bool(getattr(args, "profile_native_ops", False)) and self.device.type == "cuda":
                            torch.cuda.synchronize(self.device)
                        self.stats[int(layer_id)].add_index_sidecar_timing(time.perf_counter() - sidecar_t0)
                        persistent_cache[cache_key] = (
                            int(cache_len_i),
                            int(cache_capacity_i),
                            vhat_all_t.detach(),
                            residual_t.detach(),
                            code_error_t.detach(),
                            int(actual_value_subbits_for_cost),
                        )
        if (
            str(args.selector_mode) in {"fullscan", "oracle"}
            and str(args.selector_backend) == "cuda_ext"
            and all(index is not None and index.pages for index in warmed_indexes)
        ):
            fast_decode_index_cache_key = (
                "fullscan_decode",
                str(self.online_confidence_rule),
                int(dynamic_start),
                int(sealed_end),
                int(args.page_size),
                int(args.subvecs),
                int(args.subbits),
                int(args.kmeans_iters),
                int(args.seed),
                int(self.key_bytes),
                str(getattr(args, "index_build_backend", "numpy")),
                str(args.selected_value_mode),
                str(args.tail_mode),
                int(args.value_subvecs),
                int(args.value_subbits),
                int(args.value_pq_group_pages),
                int(self.value_bytes),
                int(num_kv_heads),
            )
            fast_decode_index_cache = getattr(module, "_pagedpq_fast_decode_index_cache", None)
            if not isinstance(fast_decode_index_cache, dict):
                fast_decode_index_cache = {}
                setattr(module, "_pagedpq_fast_decode_index_cache", fast_decode_index_cache)
            fast_decode_index_cache.clear()
            fast_decode_index_cache[fast_decode_index_cache_key] = tuple(
                index for index in warmed_indexes if index is not None
            )
