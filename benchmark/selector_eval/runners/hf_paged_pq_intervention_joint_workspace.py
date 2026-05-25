#!/usr/bin/env python3
from __future__ import annotations

import os
from dataclasses import dataclass, field
from collections.abc import Callable
from typing import Any

import torch

from benchmark.selector_eval.data.trace import static_tokens, unique_tokens
from benchmark.selector_eval.gpu.run_gpu_paged_pq_eval import GPUIndex, load_selector_paged_pq_ext
from benchmark.selector_eval.runners.hf_paged_pq_intervention_common import _env_int, _env_truthy


def _dict_attr(owner: Any, name: str) -> dict:
    cache = getattr(owner, name, None)
    if not isinstance(cache, dict):
        cache = {}
        setattr(owner, name, cache)
    return cache


@dataclass
class JointExactLogitHelper:
    args: Any
    device: torch.device
    layer_id: int
    context_len: int
    group_size: int
    sqrt_dim: float
    keys_all: torch.Tensor
    dense_decode_key_t_float_cache: Callable[..., torch.Tensor | None]
    backend: str
    _keys_t_float_t: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.backend not in {"cublas_t", "custom", "grouped"}:
            raise RuntimeError(
                "SELECTOR_PQ_JOINT_NATIVE_EXACT_LOGITS_BACKEND must be 'cublas_t', 'custom', or 'grouped'"
            )

    def keys_t_float_cache(self) -> torch.Tensor | None:
        if self._keys_t_float_t is None:
            self._keys_t_float_t = self.dense_decode_key_t_float_cache(
                layer_id=int(self.layer_id),
                keys_all=self.keys_all,
                key_count=int(self.context_len),
            )
        return self._keys_t_float_t

    def full_exact_logits(
        self,
        queries_i: torch.Tensor,
        keys_i: torch.Tensor,
        *,
        kv_head_i: int | None = None,
    ) -> torch.Tensor:
        native_i = load_selector_paged_pq_ext()
        if not hasattr(native_i, "gqa_decode_full_exact_logits"):
            raise RuntimeError("SELECTOR_PQ_JOINT_NATIVE_EXACT_LOGITS requires updated CUDA extension")
        if self.backend == "cublas_t" and hasattr(native_i, "gqa_decode_full_exact_logits_t_cublas"):
            keys_t_cache_i = self.keys_t_float_cache()
            if keys_t_cache_i is not None:
                if kv_head_i is None:
                    keys_t_i = keys_t_cache_i[:, :, : int(self.context_len)]
                else:
                    keys_t_i = keys_t_cache_i[int(kv_head_i): int(kv_head_i) + 1, :, : int(self.context_len)]
                if int(queries_i.shape[0]) == int(keys_t_i.shape[0]) * int(self.group_size):
                    return native_i.gqa_decode_full_exact_logits_t_cublas(
                        queries_i.contiguous(),
                        keys_t_i,
                        int(self.group_size),
                        int(self.context_len),
                        float(1.0 / float(self.sqrt_dim)),
                    )
        if self.backend == "grouped" and hasattr(native_i, "gqa_decode_full_exact_logits_grouped"):
            return native_i.gqa_decode_full_exact_logits_grouped(
                queries_i.contiguous(),
                keys_i,
                int(self.group_size),
                int(self.context_len),
                float(1.0 / float(self.sqrt_dim)),
            )
        return native_i.gqa_decode_full_exact_logits(
            queries_i.contiguous(),
            keys_i,
            int(self.group_size),
            int(self.context_len),
            float(1.0 / float(self.sqrt_dim)),
        )


@dataclass
class JointKVWorkspace:
    args: Any
    model: Any
    module: Any
    device: torch.device
    context_len: int
    grouped_strided_output_workspace_enabled: bool
    token_layout_cache: dict[
        tuple[object, ...],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, bool],
    ] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.score_grid_workspace_cache = _dict_attr(
            self.model,
            "_pagedpq_joint_score_grid_workspace_cache",
        )
        self.grouped_score_grid_workspace_cache = _dict_attr(
            self.model,
            "_pagedpq_joint_grouped_score_grid_workspace_cache",
        )
        self.softmax_base_workspace_cache = _dict_attr(
            self.model,
            "_pagedpq_joint_softmax_base_workspace_cache",
        )
        self.grouped_output_workspace_cache = _dict_attr(
            self.model,
            "_pagedpq_joint_grouped_output_workspace_cache",
        )
        self.rank_prefix_workspace_cache = _dict_attr(
            self.module,
            "_pagedpq_joint_rank_prefix_workspace_cache",
        )
        self.risk_prefix_workspace_cache = _dict_attr(
            self.module,
            "_pagedpq_joint_risk_prefix_workspace_cache",
        )
        self.score_direct_workspace_cache = _dict_attr(
            self.model,
            "_pagedpq_joint_score_direct_workspace_cache",
        )

    def score_grid_workspace_for(
        self,
        *,
        k_count: int,
        heads: int,
        context_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        key = (str(self.device), int(k_count), int(heads))
        cached = self.score_grid_workspace_cache.get(key)
        if cached is not None and int(cached[0].shape[2]) == int(context_len):
            return cached
        score_t = torch.empty(
            (int(k_count), int(heads), int(context_len)),
            dtype=torch.float32,
            device=self.device,
        )
        mask_t = torch.empty(
            (int(k_count), int(heads), int(context_len)),
            dtype=torch.uint8,
            device=self.device,
        )
        fit_scale_t = torch.empty((int(k_count), int(heads)), dtype=torch.float32, device=self.device)
        fit_bias_t = torch.empty_like(fit_scale_t)
        cached = (score_t, mask_t, fit_scale_t, fit_bias_t)
        self.score_grid_workspace_cache[key] = cached
        return cached

    def grouped_score_grid_workspace_for(
        self,
        *,
        kv_heads: int,
        k_count: int,
        heads: int,
        context_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        key = (str(self.device), int(kv_heads), int(k_count), int(heads))
        cached = self.grouped_score_grid_workspace_cache.get(key)
        flat_len = max(1, int(k_count) * int(heads) * int(context_len))
        if cached is not None:
            cached_flat_len, score_flat_t, mask_flat_t, fit_scale_t, fit_bias_t = cached
            if int(cached_flat_len) >= int(flat_len):
                return (
                    score_flat_t[:, :flat_len].reshape(
                        int(kv_heads),
                        int(k_count),
                        int(heads),
                        int(context_len),
                    ),
                    mask_flat_t[:, :flat_len].reshape(
                        int(kv_heads),
                        int(k_count),
                        int(heads),
                        int(context_len),
                    ),
                    fit_scale_t,
                    fit_bias_t,
                )
            grow_pad = max(
                0,
                _env_int("SELECTOR_PQ_JOINT_GROUPED_SCORE_WORKSPACE_GROW_PAD", 262144),
            )
            flat_len = max(
                int(flat_len),
                int(cached_flat_len) + max(int(grow_pad), int(flat_len) - int(cached_flat_len)),
            )
        score_flat_t = torch.empty(
            (int(kv_heads), int(flat_len)),
            dtype=torch.float32,
            device=self.device,
        )
        mask_flat_t = torch.empty(
            (int(kv_heads), int(flat_len)),
            dtype=torch.uint8,
            device=self.device,
        )
        fit_scale_t = torch.empty(
            (int(kv_heads), int(k_count), int(heads)),
            dtype=torch.float32,
            device=self.device,
        )
        fit_bias_t = torch.empty_like(fit_scale_t)
        cached = (int(flat_len), score_flat_t, mask_flat_t, fit_scale_t, fit_bias_t)
        self.grouped_score_grid_workspace_cache[key] = cached
        view_len = max(1, int(k_count) * int(heads) * int(context_len))
        return (
            score_flat_t[:, :view_len].reshape(
                int(kv_heads),
                int(k_count),
                int(heads),
                int(context_len),
            ),
            mask_flat_t[:, :view_len].reshape(
                int(kv_heads),
                int(k_count),
                int(heads),
                int(context_len),
            ),
            fit_scale_t,
            fit_bias_t,
        )

    def softmax_base_workspace_for(
        self,
        *,
        slot: int,
        k_count: int,
        heads: int,
        context_len: int,
        dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Grouped residual-risk stores records for all live KV heads, so each
        # live KV-head needs a distinct slot.
        key = (str(self.device), int(slot), int(k_count), int(heads), int(dim))
        cached = self.softmax_base_workspace_cache.get(key)
        if cached is not None and int(cached[0].shape[2]) == int(context_len):
            return cached
        probs_t = torch.empty(
            (int(k_count), int(heads), int(context_len)),
            dtype=torch.float32,
            device=self.device,
        )
        base_t_ws = torch.empty(
            (int(k_count), int(heads), int(dim)),
            dtype=torch.float32,
            device=self.device,
        )
        cached = (probs_t, base_t_ws)
        self.softmax_base_workspace_cache[key] = cached
        return cached

    def grouped_output_workspace_for(
        self,
        *,
        kv_heads: int,
        k_count: int,
        heads: int,
        context_len: int,
        dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = (str(self.device), int(kv_heads), int(k_count), int(heads), int(dim))
        cached = self.grouped_output_workspace_cache.get(key)
        if cached is not None:
            capacity_i, probs_buf_t, base_buf_t = cached
            if self.grouped_strided_output_workspace_enabled and int(capacity_i) >= int(context_len):
                return probs_buf_t, base_buf_t
            if not self.grouped_strided_output_workspace_enabled and int(capacity_i) == int(context_len):
                return probs_buf_t, base_buf_t
        if self.grouped_strided_output_workspace_enabled:
            grow_pad_i = max(
                0,
                int(os.environ.get("SELECTOR_PQ_JOINT_STRIDED_OUTPUT_WORKSPACE_GROW_PAD", "1024")),
            )
            prior_capacity_i = int(cached[0]) if cached is not None else 0
            capacity_i = max(int(context_len), prior_capacity_i * 2, int(context_len) + grow_pad_i)
        else:
            # Native non-strided softmax/base workspace kernels require
            # contiguous probability buffers, so keep exact context length.
            capacity_i = int(context_len)
        probs_buf_t = torch.empty(
            (int(kv_heads), int(k_count), int(heads), int(capacity_i)),
            dtype=torch.float32,
            device=self.device,
        )
        base_buf_t = torch.empty(
            (int(kv_heads), int(k_count), int(heads), int(dim)),
            dtype=torch.float32,
            device=self.device,
        )
        self.grouped_output_workspace_cache[key] = (
            int(capacity_i),
            probs_buf_t,
            base_buf_t,
        )
        if self.grouped_strided_output_workspace_enabled:
            return probs_buf_t, base_buf_t
        return probs_buf_t[:, :, :, : int(context_len)], base_buf_t

    def rank_prefix_workspace_for(
        self,
        *,
        rows: int,
        count: int,
        max_take: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        native = load_selector_paged_pq_ext()
        if not hasattr(native, "joint_rank_prefix_sort_temp_bytes"):
            raise RuntimeError("SELECTOR_PQ_JOINT_RANK_PREFIX_WORKSPACE requires joint_rank_prefix_sort_temp_bytes")
        grow_pad = max(0, int(os.environ.get("SELECTOR_PQ_JOINT_RANK_PREFIX_WORKSPACE_GROW_PAD", "1024")))
        key = (
            int(rows),
            int(max_take),
            str(self.device),
            int(torch.cuda.current_device()) if self.device.type == "cuda" else -1,
        )
        cached = self.rank_prefix_workspace_cache.get(key)
        count_capacity = int(count)
        if cached is not None:
            (
                cached_count_capacity,
                score_in_t,
                score_out_t,
                pos_in_t,
                pos_out_t,
                offsets_t,
                temp_t,
                out_t,
            ) = cached
            if int(cached_count_capacity) >= int(count):
                return (
                    score_in_t,
                    score_out_t,
                    pos_in_t,
                    pos_out_t,
                    offsets_t,
                    temp_t,
                    out_t,
                )
            count_capacity = max(
                int(count),
                int(cached_count_capacity) + max(int(grow_pad), int(count) - int(cached_count_capacity)),
            )
        else:
            count_capacity = int(count) + int(grow_pad)
        total_capacity = max(1, int(rows) * int(count_capacity))
        score_in_t = torch.empty((total_capacity,), dtype=torch.float32, device=self.device)
        score_out_t = torch.empty_like(score_in_t)
        pos_in_t = torch.empty((total_capacity,), dtype=torch.int32, device=self.device)
        pos_out_t = torch.empty_like(pos_in_t)
        offsets_t = torch.empty((max(1, int(rows) + 1),), dtype=torch.long, device=self.device)
        temp_bytes = max(1, int(native.joint_rank_prefix_sort_temp_bytes(int(rows), int(count_capacity))))
        temp_t = torch.empty((temp_bytes,), dtype=torch.uint8, device=self.device)
        out_t = torch.empty((max(1, int(rows) * int(max_take)),), dtype=torch.long, device=self.device)
        self.rank_prefix_workspace_cache[key] = (
            int(count_capacity),
            score_in_t,
            score_out_t,
            pos_in_t,
            pos_out_t,
            offsets_t,
            temp_t,
            out_t,
        )
        return score_in_t, score_out_t, pos_in_t, pos_out_t, offsets_t, temp_t, out_t

    def native_rank_prefix_tokens(
        self,
        scores_t: torch.Tensor,
        tokens_t: torch.Tensor,
        max_take_i: int,
        take_counts_t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        native = load_selector_paged_pq_ext()
        scores_c = scores_t.to(dtype=torch.float32).contiguous()
        tokens_c = tokens_t.to(dtype=torch.long).contiguous()
        if (
            _env_truthy("SELECTOR_PQ_JOINT_NATIVE_BUDGET_PREFIX", "0")
            and take_counts_t is not None
            and int(take_counts_t.numel()) > 0
            and hasattr(native, "joint_budget_prefix_tokens")
        ):
            return native.joint_budget_prefix_tokens(
                scores_c,
                tokens_c,
                take_counts_t.to(device=scores_c.device, dtype=torch.long).contiguous(),
                int(max_take_i),
            )
        if (
            _env_truthy("SELECTOR_PQ_JOINT_RANK_PREFIX_WORKSPACE", "0")
            and hasattr(native, "joint_rank_prefix_tokens_workspace")
        ):
            workspace = self.rank_prefix_workspace_for(
                rows=int(scores_c.shape[0]),
                count=int(scores_c.shape[1]),
                max_take=int(max_take_i),
            )
            return native.joint_rank_prefix_tokens_workspace(
                scores_c,
                tokens_c,
                int(max_take_i),
                *workspace,
            )
        return native.joint_rank_prefix_tokens(scores_c, tokens_c, int(max_take_i))

    def grouped_risk_prefix_workspace_for(
        self,
        *,
        rows: int,
        context_len: int,
        v_steps: int,
        dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        native = load_selector_paged_pq_ext()
        if not hasattr(native, "joint_grouped_risk_sort_temp_bytes"):
            raise RuntimeError("SELECTOR_PQ_JOINT_RISK_PREFIX_WORKSPACE requires joint_grouped_risk_sort_temp_bytes")
        grow_pad = max(0, int(os.environ.get("SELECTOR_PQ_JOINT_RISK_PREFIX_WORKSPACE_GROW_PAD", "1024")))
        key = (
            int(rows),
            int(v_steps),
            int(dim),
            str(self.device),
            int(torch.cuda.current_device()) if self.device.type == "cuda" else -1,
        )
        cached = self.risk_prefix_workspace_cache.get(key)
        context_capacity = int(context_len)
        if cached is not None:
            (
                cached_context_capacity,
                risk_in_t,
                risk_out_t,
                ids_in_t,
                ids_out_t,
                offsets_t,
                temp_t,
                interval_sums_t,
                outputs_t,
            ) = cached
            if int(cached_context_capacity) >= int(context_len):
                return (
                    risk_in_t,
                    risk_out_t,
                    ids_in_t,
                    ids_out_t,
                    offsets_t,
                    temp_t,
                    interval_sums_t,
                    outputs_t,
                )
            context_capacity = max(
                int(context_len),
                int(cached_context_capacity) + max(int(grow_pad), int(context_len) - int(cached_context_capacity)),
            )
        else:
            context_capacity = int(context_len) + int(grow_pad)
        total_capacity = max(1, int(rows) * int(context_capacity))
        risk_in_t = torch.empty((total_capacity,), dtype=torch.float32, device=self.device)
        risk_out_t = torch.empty_like(risk_in_t)
        ids_in_t = torch.empty((total_capacity,), dtype=torch.int32, device=self.device)
        ids_out_t = torch.empty_like(ids_in_t)
        offsets_t = torch.empty((max(1, int(rows) + 1),), dtype=torch.long, device=self.device)
        temp_bytes = max(1, int(native.joint_grouped_risk_sort_temp_bytes(int(rows), int(context_capacity))))
        temp_t = torch.empty((temp_bytes,), dtype=torch.uint8, device=self.device)
        interval_sums_t = torch.empty((int(rows), int(v_steps), int(dim)), dtype=torch.float32, device=self.device)
        outputs_t = torch.empty_like(interval_sums_t)
        self.risk_prefix_workspace_cache[key] = (
            int(context_capacity),
            risk_in_t,
            risk_out_t,
            ids_in_t,
            ids_out_t,
            offsets_t,
            temp_t,
            interval_sums_t,
            outputs_t,
        )
        return risk_in_t, risk_out_t, ids_in_t, ids_out_t, offsets_t, temp_t, interval_sums_t, outputs_t

    def grouped_score_direct_workspace_for(
        self,
        *,
        rows: int,
        context_len: int,
        v_steps: int,
        dim: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        native = load_selector_paged_pq_ext()
        if not hasattr(native, "joint_grouped_risk_sort_temp_bytes"):
            raise RuntimeError("SELECTOR_PQ_JOINT_SCORE_DIRECT_WORKSPACE requires joint_grouped_risk_sort_temp_bytes")
        grow_pad = max(0, _env_int("SELECTOR_PQ_JOINT_SCORE_DIRECT_WORKSPACE_GROW_PAD", 1024))
        key = (
            int(rows),
            int(v_steps),
            int(dim),
            str(self.device),
            int(torch.cuda.current_device()) if self.device.type == "cuda" else -1,
        )
        cached = self.score_direct_workspace_cache.get(key)
        context_capacity = int(context_len)
        if cached is not None:
            (
                cached_context_capacity,
                row_max_t,
                row_denom_t,
                base_flat_t,
                risk_in_t,
                risk_out_t,
                ids_in_t,
                ids_out_t,
                offsets_t,
                temp_t,
                interval_sums_t,
                outputs_t,
            ) = cached
            if int(cached_context_capacity) >= int(context_len):
                return (
                    row_max_t,
                    row_denom_t,
                    base_flat_t,
                    risk_in_t,
                    risk_out_t,
                    ids_in_t,
                    ids_out_t,
                    offsets_t,
                    temp_t,
                    interval_sums_t,
                    outputs_t,
                )
            context_capacity = max(
                int(context_len),
                int(cached_context_capacity) + max(int(grow_pad), int(context_len) - int(cached_context_capacity)),
            )
        else:
            context_capacity = int(context_len) + int(grow_pad)
        total_capacity = max(1, int(rows) * int(context_capacity))
        row_max_t = torch.empty((max(1, int(rows)),), dtype=torch.float32, device=self.device)
        row_denom_t = torch.empty_like(row_max_t)
        base_flat_t = torch.empty((max(1, int(rows)), int(dim)), dtype=torch.float32, device=self.device)
        risk_in_t = torch.empty((total_capacity,), dtype=torch.float32, device=self.device)
        risk_out_t = torch.empty_like(risk_in_t)
        ids_in_t = torch.empty((total_capacity,), dtype=torch.int32, device=self.device)
        ids_out_t = torch.empty_like(ids_in_t)
        offsets_t = torch.empty((max(1, int(rows) + 1),), dtype=torch.long, device=self.device)
        temp_bytes = max(1, int(native.joint_grouped_risk_sort_temp_bytes(int(rows), int(context_capacity))))
        temp_t = torch.empty((temp_bytes,), dtype=torch.uint8, device=self.device)
        interval_sums_t = torch.empty((max(1, int(rows)), int(v_steps), int(dim)), dtype=torch.float32, device=self.device)
        outputs_t = torch.empty_like(interval_sums_t)
        self.score_direct_workspace_cache[key] = (
            int(context_capacity),
            row_max_t,
            row_denom_t,
            base_flat_t,
            risk_in_t,
            risk_out_t,
            ids_in_t,
            ids_out_t,
            offsets_t,
            temp_t,
            interval_sums_t,
            outputs_t,
        )
        return (
            row_max_t,
            row_denom_t,
            base_flat_t,
            risk_in_t,
            risk_out_t,
            ids_in_t,
            ids_out_t,
            offsets_t,
            temp_t,
            interval_sums_t,
            outputs_t,
        )

    def token_layout_for(
        self,
        index: GPUIndex,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, bool]:
        args = self.args
        context_len_i = int(self.context_len)
        page_ranges = tuple(
            (
                int(page.start),
                min(int(page.start) + int(page.size), context_len_i),
            )
            for page in index.pages
            if int(page.start) < context_len_i and int(page.size) > 0
        )
        layout_key = (
            int(context_len_i),
            int(index.pending_start),
            int(index.indexed_end),
            int(args.static_prefix),
            int(args.static_suffix),
            page_ranges,
        )
        cached_layout = self.token_layout_cache.get(layout_key)
        if cached_layout is not None:
            return cached_layout

        prefix_end_i = min(max(0, int(args.static_prefix)), context_len_i)
        suffix_start_i = max(0, context_len_i - max(0, int(args.static_suffix)))
        sealed_end_i = max((end for _, end in page_ranges), default=prefix_end_i)
        pages_contiguous = True
        expected_start_i = prefix_end_i
        for start, end in page_ranges:
            if int(start) != int(expected_start_i) or int(end) < int(start):
                pages_contiguous = False
                break
            expected_start_i = int(end)
        pages_contiguous = pages_contiguous and int(expected_start_i) == int(sealed_end_i)
        indexed_end_i = max(0, min(int(index.indexed_end), context_len_i))
        pending_start_i = max(prefix_end_i, int(index.pending_start))
        pending_end_i = max(pending_start_i, min(indexed_end_i, suffix_start_i))

        if (
            _env_truthy("SELECTOR_PQ_JOINT_FAST_TOKEN_LAYOUT", "0")
            and pages_contiguous
            and int(index.pending_start) == int(sealed_end_i)
            and int(sealed_end_i) <= int(suffix_start_i)
            and int(indexed_end_i) >= int(suffix_start_i)
        ):
            # Preserve unique_tokens(static_tokens + pending) order:
            # prefix, suffix, then pending.
            indexed_tokens_layout_t = (
                torch.arange(prefix_end_i, sealed_end_i, dtype=torch.long, device=self.device)
                if sealed_end_i > prefix_end_i
                else torch.empty((0,), dtype=torch.long, device=self.device)
            )
            base_parts_t = []
            if prefix_end_i > 0:
                base_parts_t.append(torch.arange(0, prefix_end_i, dtype=torch.long, device=self.device))
            if context_len_i > suffix_start_i:
                base_parts_t.append(torch.arange(suffix_start_i, context_len_i, dtype=torch.long, device=self.device))
            pending_base_end_i = min(int(pending_end_i), int(suffix_start_i))
            if pending_base_end_i > pending_start_i:
                base_parts_t.append(
                    torch.arange(pending_start_i, pending_base_end_i, dtype=torch.long, device=self.device)
                )
            base_layout_t = (
                torch.cat(base_parts_t)
                if base_parts_t
                else torch.empty((0,), dtype=torch.long, device=self.device)
            )
            out = (indexed_tokens_layout_t, base_layout_t, None, True)
            self.token_layout_cache[layout_key] = out
            return out

        token_parts = [
            torch.arange(start, end, dtype=torch.long, device=self.device)
            for start, end in page_ranges
            if end > start
        ]
        indexed_tokens_layout_t = (
            torch.cat(token_parts)
            if token_parts
            else torch.empty((0,), dtype=torch.long, device=self.device)
        )

        pending_layout = list(
            range(
                max(0, int(index.pending_start)),
                max(0, min(int(index.indexed_end), context_len_i)),
            )
        )
        base_layout = unique_tokens(
            static_tokens(context_len_i - 1, int(args.static_prefix), int(args.static_suffix))
            + pending_layout,
            context_len=context_len_i,
        )
        coverage_intervals: list[tuple[int, int]] = [
            (max(0, int(start)), min(context_len_i, int(end)))
            for start, end in page_ranges
            if int(end) > int(start)
        ]
        base_tokens_sorted = sorted(
            int(token)
            for token in base_layout
            if 0 <= int(token) < context_len_i
        )
        if base_tokens_sorted:
            run_start = base_tokens_sorted[0]
            prev = base_tokens_sorted[0]
            for token in base_tokens_sorted[1:]:
                if token == prev + 1:
                    prev = token
                    continue
                coverage_intervals.append((run_start, prev + 1))
                run_start = token
                prev = token
            coverage_intervals.append((run_start, prev + 1))
        coverage_end = 0
        layout_covers_context = context_len_i <= 0
        for start, end in sorted(coverage_intervals):
            if end <= coverage_end:
                continue
            if start > coverage_end:
                break
            coverage_end = max(coverage_end, end)
            if coverage_end >= context_len_i:
                layout_covers_context = True
                break
        base_layout_t = torch.as_tensor(base_layout, dtype=torch.long, device=self.device)
        if int(base_layout_t.numel()) > 0:
            base_layout_t = base_layout_t[(base_layout_t >= 0) & (base_layout_t < context_len_i)]

        indexed_end_without_suffix = context_len_i - max(0, int(args.static_suffix))
        nonbase_all = all(
            start >= min(max(0, int(args.static_prefix)), context_len_i)
            and end <= int(index.pending_start)
            and end <= indexed_end_without_suffix
            for start, end in page_ranges
        )
        if nonbase_all:
            nonbase_mask_layout_t = None
        else:
            base_rank_mask_t = torch.zeros((context_len_i,), dtype=torch.bool, device=self.device)
            if int(base_layout_t.numel()) > 0:
                base_rank_mask_t.index_fill_(0, base_layout_t, True)
            nonbase_mask_layout_t = ~base_rank_mask_t.index_select(0, indexed_tokens_layout_t)

        out = (indexed_tokens_layout_t, base_layout_t, nonbase_mask_layout_t, bool(layout_covers_context))
        self.token_layout_cache[layout_key] = out
        return out
