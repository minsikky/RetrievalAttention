#!/usr/bin/env python3
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch

from benchmark.selector_eval.gpu.run_gpu_paged_pq_eval import (
    GPUIndex,
    _sync_if_cuda,
    ensure_native_fullscan_pack,
    load_selector_paged_pq_ext,
)
from benchmark.selector_eval.runners.hf_paged_pq_intervention_common import _env_truthy
from benchmark.selector_eval.runners.hf_paged_pq_intervention_patch_state import PagedPQPatchState
from benchmark.selector_eval.runners.hf_paged_pq_intervention_stats import ApproxStats
from benchmark.selector_eval.runners.hf_paged_pq_intervention_value import (
    value_vpq_pack_torch,
)
from benchmark.selector_eval.runners.hf_paged_pq_intervention_vpq_grouped import (
    grouped_vpq_residual_sidecars_for,
)
from benchmark.selector_eval.runners.hf_paged_pq_intervention_vpq_sidecars import joint_vpq_sidecars_for


@dataclass
class PagedPQForwardState:
    args: Any
    module: Any
    patch_state: PagedPQPatchState
    layer_id: int
    stats: ApproxStats
    device: torch.device
    values_all: torch.Tensor
    index_cache: dict[int, GPUIndex]
    prefix_index_cache: dict[tuple[int, int, int], GPUIndex]
    context_len: int
    num_kv_heads: int
    value_bytes: int
    joint_vpq_runtime_cache: dict[tuple[object, ...], tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]] = field(
        default_factory=dict
    )

    def prefix_index_for(self, kv_head: int, query_context_len: int) -> GPUIndex:
        args = self.args
        full_index = self.index_cache[int(kv_head)]
        if int(query_context_len) >= int(self.context_len):
            return full_index
        dyn_start = min(max(0, int(args.static_prefix)), int(query_context_len))
        indexed_end_q = max(dyn_start, int(query_context_len) - max(0, int(args.static_suffix)))
        sealed_end_q = dyn_start + (
            (max(0, indexed_end_q - dyn_start) // max(1, int(args.page_size))) * max(1, int(args.page_size))
        )
        key = (int(kv_head), int(indexed_end_q), int(sealed_end_q))
        cached = self.prefix_index_cache.get(key)
        if cached is not None:
            return cached
        pages = [
            page
            for page in full_index.pages
            if int(page.start) + int(page.size) <= int(sealed_end_q)
        ]
        native_codebooks = None
        native_codes = None
        native_page_starts = None
        if (
            full_index.native_codebooks is not None
            and full_index.native_codes is not None
            and full_index.native_page_starts is not None
        ):
            page_count = int(len(pages))
            native_codebooks = full_index.native_codebooks[:page_count]
            native_codes = full_index.native_codes[:page_count]
            native_page_starts = full_index.native_page_starts[:page_count]
        view = GPUIndex(
            pages=pages,
            pending_start=int(sealed_end_q),
            indexed_end=int(indexed_end_q),
            build_seconds=0.0,
            build_read_mb=0.0,
            build_write_mb=0.0,
            router_group_means=None,
            router_group_tokens=None,
            router_group_member_refs=None,
            native_codebooks=native_codebooks,
            native_codes=native_codes,
            native_page_starts=native_page_starts,
        )
        self.prefix_index_cache[key] = view
        return view

    @staticmethod
    def gqa_index_pack_key(
        gqa_indexes: list[GPUIndex],
        *,
        extra: tuple[object, ...] = (),
    ) -> tuple[object, ...]:
        return (
            tuple(
                (
                    id(index),
                    int(len(index.pages)),
                    int(index.pending_start),
                    id(index.native_codebooks) if index.native_codebooks is not None else 0,
                    id(index.native_codes) if index.native_codes is not None else 0,
                    id(index.native_page_starts) if index.native_page_starts is not None else 0,
                )
                for index in gqa_indexes
            ),
            *extra,
        )

    def gqa_native_fullscan_pack(self, gqa_indexes: list[GPUIndex]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        args = self.args
        if bool(getattr(args, "profile_native_ops", False)):
            _sync_if_cuda(self.device)
            pack_t0 = time.perf_counter()
        else:
            pack_t0 = 0.0
        fast_key = self.gqa_index_pack_key(gqa_indexes, extra=("k", int(args.subbits)))
        gqa_fast_pack_cache = getattr(self.module, "_pagedpq_gqa_native_pack_fast_cache", None)
        if not isinstance(gqa_fast_pack_cache, dict):
            gqa_fast_pack_cache = {}
            setattr(self.module, "_pagedpq_gqa_native_pack_fast_cache", gqa_fast_pack_cache)
        cached_fast = gqa_fast_pack_cache.get(fast_key)
        if cached_fast is not None:
            if bool(getattr(args, "profile_native_ops", False)):
                _sync_if_cuda(self.device)
                self.stats.add_native_pack_timing(time.perf_counter() - pack_t0)
            return cached_fast
        packs = [ensure_native_fullscan_pack(index, subbits=int(args.subbits)) for index in gqa_indexes]
        cache_key = tuple(
            (
                int(pack[0].data_ptr()),
                tuple(int(v) for v in pack[0].shape),
                int(pack[1].data_ptr()),
                tuple(int(v) for v in pack[1].shape),
                str(pack[1].dtype),
                int(pack[2].data_ptr()),
                int(pack[2].numel()),
            )
            for pack in packs
        )
        gqa_pack_cache = getattr(self.module, "_pagedpq_gqa_native_pack_cache", None)
        if not isinstance(gqa_pack_cache, dict):
            gqa_pack_cache = {}
            setattr(self.module, "_pagedpq_gqa_native_pack_cache", gqa_pack_cache)
        cached = gqa_pack_cache.get(cache_key)
        if cached is not None:
            if gqa_fast_pack_cache:
                gqa_fast_pack_cache.clear()
            gqa_fast_pack_cache[fast_key] = cached
            if bool(getattr(args, "profile_native_ops", False)):
                _sync_if_cuda(self.device)
                self.stats.add_native_pack_timing(time.perf_counter() - pack_t0)
            return cached
        if gqa_pack_cache:
            gqa_pack_cache.clear()
            if self.device.type == "cuda" and bool(getattr(args, "debug_empty_cache_native", False)):
                torch.cuda.empty_cache()
        codebooks = torch.stack([pack[0] for pack in packs], dim=0).contiguous()
        codes = torch.stack([pack[1] for pack in packs], dim=0).contiguous()
        page_starts = packs[0][2]
        packed = (codebooks, codes, page_starts)
        gqa_pack_cache[cache_key] = packed
        if gqa_fast_pack_cache:
            gqa_fast_pack_cache.clear()
        gqa_fast_pack_cache[fast_key] = packed
        if bool(getattr(args, "profile_native_ops", False)):
            _sync_if_cuda(self.device)
            self.stats.add_native_pack_timing(time.perf_counter() - pack_t0)
        return packed

    def gqa_value_vpq_pack(
        self,
        gqa_indexes: list[GPUIndex],
        *,
        value_group_pages: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        args = self.args
        if bool(getattr(args, "profile_native_ops", False)):
            _sync_if_cuda(self.device)
            pack_t0 = time.perf_counter()
        else:
            pack_t0 = 0.0
        actual_value_subbits_for_key = int(args.value_subbits) if int(args.value_subbits) > 0 else int(args.subbits)
        fast_key = self.gqa_index_pack_key(
            gqa_indexes,
            extra=(
                "v",
                int(args.value_subvecs),
                int(actual_value_subbits_for_key),
                int(value_group_pages),
                int(self.value_bytes),
            ),
        )
        gqa_value_fast_cache = getattr(self.module, "_pagedpq_gqa_value_vpq_pack_fast_cache", None)
        if not isinstance(gqa_value_fast_cache, dict):
            gqa_value_fast_cache = {}
            setattr(self.module, "_pagedpq_gqa_value_vpq_pack_fast_cache", gqa_value_fast_cache)
        cached_fast = gqa_value_fast_cache.get(fast_key)
        if cached_fast is not None:
            if bool(getattr(args, "profile_native_ops", False)):
                _sync_if_cuda(self.device)
                self.stats.add_native_pack_timing(time.perf_counter() - pack_t0)
            return cached_fast
        value_packs = [
            value_vpq_pack_torch(
                index=index,
                values=self.values_all[int(kv_head)],
                value_subvecs=int(args.value_subvecs),
                value_subbits=int(args.value_subbits),
                key_bytes=int(self.value_bytes),
                device=self.device,
                value_group_pages=int(value_group_pages),
            )
            for kv_head, index in enumerate(gqa_indexes)
        ]
        if any(pack is None for pack in value_packs):
            raise RuntimeError("missing V-PQ pack for native decode")
        for index in gqa_indexes:
            build_stats = getattr(index, "_last_value_vpq_build_stats", None)
            if build_stats is not None:
                build_seconds, build_read_mb, build_write_mb = build_stats
                self.stats.add_index_build(build_seconds, build_read_mb, build_write_mb)
                setattr(index, "_last_value_vpq_build_stats", None)
        packs = [pack for pack in value_packs if pack is not None]
        cache_key = tuple(
            (
                int(pack[0].data_ptr()),
                tuple(int(v) for v in pack[0].shape),
                int(pack[1].data_ptr()),
                tuple(int(v) for v in pack[1].shape),
                str(pack[1].dtype),
                int(pack[2].data_ptr()),
                int(pack[2].numel()),
                int(pack[3]),
                int(pack[4]),
            )
            for pack in packs
        )
        gqa_value_pack_cache = getattr(self.module, "_pagedpq_gqa_value_vpq_pack_cache", None)
        if not isinstance(gqa_value_pack_cache, dict):
            gqa_value_pack_cache = {}
            setattr(self.module, "_pagedpq_gqa_value_vpq_pack_cache", gqa_value_pack_cache)
        cached = gqa_value_pack_cache.get(cache_key)
        if cached is not None:
            if gqa_value_fast_cache:
                gqa_value_fast_cache.clear()
            gqa_value_fast_cache[fast_key] = cached
            if bool(getattr(args, "profile_native_ops", False)):
                _sync_if_cuda(self.device)
                self.stats.add_native_pack_timing(time.perf_counter() - pack_t0)
            return cached
        if gqa_value_pack_cache:
            gqa_value_pack_cache.clear()
            if self.device.type == "cuda" and bool(getattr(args, "debug_empty_cache_native", False)):
                torch.cuda.empty_cache()
        value_codebooks = torch.stack([pack[0] for pack in packs], dim=0).contiguous()
        value_codes = torch.stack([pack[1] for pack in packs], dim=0).contiguous()
        value_page_starts = packs[0][2]
        value_page_size = int(packs[0][3])
        actual_value_subbits = int(packs[0][4])
        packed = (value_codebooks, value_codes, value_page_starts, value_page_size, actual_value_subbits)
        gqa_value_pack_cache[cache_key] = packed
        if gqa_value_fast_cache:
            gqa_value_fast_cache.clear()
        gqa_value_fast_cache[fast_key] = packed
        if bool(getattr(args, "profile_native_ops", False)):
            _sync_if_cuda(self.device)
            self.stats.add_native_pack_timing(time.perf_counter() - pack_t0)
        return packed

    def append_exact_suffix_sidecar_inplace(
        self,
        *,
        vhat_t: torch.Tensor,
        residual_t: torch.Tensor,
        code_error_t: torch.Tensor,
        values_t: torch.Tensor,
        start: int,
        end: int,
    ) -> None:
        args = self.args
        start_i = int(start)
        end_i = int(end)
        if end_i <= start_i:
            return
        if _env_truthy("SELECTOR_PQ_JOINT_NATIVE_VPQ_APPEND", "0"):
            can_use_native = (
                vhat_t.is_contiguous()
                and residual_t.is_contiguous()
                and code_error_t.is_contiguous()
                and values_t.is_contiguous()
            )
            if can_use_native:
                native = load_selector_paged_pq_ext()
                if not hasattr(native, "joint_vpq_append_exact_suffix"):
                    raise RuntimeError("SELECTOR_PQ_JOINT_NATIVE_VPQ_APPEND requires joint_vpq_append_exact_suffix")
                if bool(getattr(args, "profile_native_ops", False)):
                    _sync_if_cuda(self.device)
                    append_t0 = time.perf_counter()
                else:
                    append_t0 = 0.0
                native.joint_vpq_append_exact_suffix(
                    vhat_t,
                    residual_t,
                    code_error_t,
                    values_t,
                    start_i,
                    end_i,
                )
                if bool(getattr(args, "profile_native_ops", False)):
                    _sync_if_cuda(self.device)
                    self.stats.add_vpq_append_timing(
                        seconds=float(time.perf_counter() - append_t0),
                        calls=1,
                    )
                else:
                    self.stats.add_vpq_append_timing(calls=1)
                return
            self.stats.add_vpq_append_timing(fallback_calls=1)
        extra_values = values_t[start_i:end_i].float()
        if int(extra_values.numel()) > 0:
            vhat_t[start_i:end_i].copy_(extra_values.to(dtype=vhat_t.dtype))
            residual_t[start_i:end_i].zero_()
            code_error_t[start_i:end_i].zero_()

    def append_exact_suffix_grouped_sidecar_inplace(
        self,
        *,
        vhat_t: torch.Tensor,
        residual_t: torch.Tensor,
        code_error_t: torch.Tensor,
        values_t: torch.Tensor,
        start: int,
        end: int,
    ) -> bool:
        args = self.args
        start_i = int(start)
        end_i = int(end)
        if end_i <= start_i:
            return True
        if not _env_truthy("SELECTOR_PQ_JOINT_NATIVE_VPQ_APPEND", "0"):
            return False
        if not (vhat_t.is_contiguous() and residual_t.is_contiguous() and code_error_t.is_contiguous()):
            self.stats.add_vpq_append_timing(fallback_calls=1)
            return False
        if not values_t.is_contiguous():
            self.stats.add_vpq_append_timing(fallback_calls=1)
            return False
        native = load_selector_paged_pq_ext()
        if not hasattr(native, "joint_vpq_append_exact_suffix_grouped"):
            self.stats.add_vpq_append_timing(fallback_calls=1)
            return False
        if bool(getattr(args, "profile_native_ops", False)):
            _sync_if_cuda(self.device)
            append_t0 = time.perf_counter()
        else:
            append_t0 = 0.0
        native.joint_vpq_append_exact_suffix_grouped(
            vhat_t,
            residual_t,
            code_error_t,
            values_t,
            start_i,
            end_i,
        )
        if bool(getattr(args, "profile_native_ops", False)):
            _sync_if_cuda(self.device)
            self.stats.add_vpq_append_timing(
                seconds=float(time.perf_counter() - append_t0),
                calls=1,
                grouped_calls=1,
            )
        else:
            self.stats.add_vpq_append_timing(calls=1, grouped_calls=1)
        return True

    def joint_vpq_pack_and_fallback_for(
        self,
        *,
        index: GPUIndex,
        values_t: torch.Tensor,
        context_len_i: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, torch.Tensor] | None:
        args = self.args
        if int(args.value_subvecs) != 1:
            return None
        pack = value_vpq_pack_torch(
            index=index,
            values=values_t,
            value_subvecs=int(args.value_subvecs),
            value_subbits=int(args.value_subbits),
            key_bytes=int(self.value_bytes),
            device=values_t.device,
        )
        if pack is None or not index.pages:
            return None
        codebooks, codes, page_starts, page_size, actual_value_subbits = pack
        if int(codebooks.shape[1]) != 1:
            return None
        fallback_parts: list[torch.Tensor] = []
        cursor_i = 0
        for page in sorted(index.pages, key=lambda p: int(p.start)):
            start_i = max(0, min(int(page.start), int(context_len_i)))
            end_i = max(start_i, min(int(page.start) + int(page.size), int(context_len_i)))
            if start_i > cursor_i:
                fallback_parts.append(torch.arange(cursor_i, start_i, dtype=torch.long, device=values_t.device))
            cursor_i = max(cursor_i, end_i)
        if cursor_i < int(context_len_i):
            fallback_parts.append(torch.arange(cursor_i, int(context_len_i), dtype=torch.long, device=values_t.device))
        if fallback_parts:
            fallback_tokens = torch.cat(fallback_parts, dim=0).contiguous()
        else:
            fallback_tokens = torch.empty((0,), dtype=torch.long, device=values_t.device)
        return codebooks, codes, page_starts, int(page_size), int(actual_value_subbits), fallback_tokens

    def joint_vpq_sidecars_for(
        self,
        *,
        kv_head: int,
        index: GPUIndex,
        values_t: torch.Tensor,
        context_len_i: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        return joint_vpq_sidecars_for(
            self,
            kv_head=kv_head,
            index=index,
            values_t=values_t,
            context_len_i=context_len_i,
        )

    def grouped_vpq_residual_sidecars_for(
        self,
        gqa_indexes: list[GPUIndex],
        *,
        context_len_i: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int] | None:
        return grouped_vpq_residual_sidecars_for(
            self,
            gqa_indexes,
            context_len_i=context_len_i,
        )
