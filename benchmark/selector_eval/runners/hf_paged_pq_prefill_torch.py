#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class PrefillTorchSelector:
    args: Any
    device: torch.device
    num_kv_heads: int
    group_size: int
    budget: int
    torch_matmul_k_approx_t_cache: dict[
        tuple[int, int, tuple[int, ...], tuple[int, ...], torch.dtype],
        torch.Tensor,
    ] = field(default_factory=dict)

    def torch_lut_prefill_topk(
        self,
        queries_in: torch.Tensor,
        codebooks_in: torch.Tensor,
        codes_in: torch.Tensor,
        page_starts_in: torch.Tensor,
        *,
        local_query_start: int,
        streaming: bool = False,
        score_dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        args = self.args
        device = self.device
        num_kv_heads = self.num_kv_heads
        group_size = self.group_size
        budget = self.budget
        positions = int(queries_in.shape[0])
        heads = int(queries_in.shape[1])
        pages = int(codebooks_in.shape[1])
        subvecs = int(codebooks_in.shape[2])
        page_size_local = int(codes_in.shape[2])
        total_tokens = int(pages * page_size_local)
        k = min(max(0, int(budget)), total_tokens)
        if k <= 0 or total_tokens <= 0:
            return (
                torch.empty((positions, heads, 0), dtype=torch.long, device=device),
                torch.empty((positions, heads, 0), dtype=torch.float32, device=device),
            )
        top_tokens_out = torch.empty((positions, heads, k), dtype=torch.long, device=device)
        top_scores_out = torch.empty((positions, heads, k), dtype=torch.float32, device=device)
        query_context_lens = (
            torch.arange(positions, device=device, dtype=torch.long) + int(local_query_start) + 1
        )
        dyn_start_t = torch.clamp(
            torch.full_like(query_context_lens, int(args.static_prefix)),
            min=0,
        )
        dyn_start_t = torch.minimum(dyn_start_t, query_context_lens)
        indexed_end_t = torch.maximum(
            dyn_start_t,
            query_context_lens - max(0, int(args.static_suffix)),
        )
        sealed_end_t = dyn_start_t + (
            torch.div(
                torch.clamp(indexed_end_t - dyn_start_t, min=0),
                max(1, int(args.page_size)),
                rounding_mode="floor",
            )
            * max(1, int(args.page_size))
        )
        page_starts_dev = page_starts_in.to(device=device, dtype=torch.long)
        page_offsets = torch.arange(page_size_local, dtype=torch.long, device=device).reshape(1, 1, page_size_local)
        for kv_head in range(num_kv_heads):
            head_start = int(kv_head * group_size)
            head_end = min(heads, head_start + int(group_size))
            if head_start >= head_end:
                continue
            head_queries = queries_in[:, head_start:head_end, :].contiguous()
            if streaming:
                running_vals = torch.full(
                    (positions, head_end - head_start, k),
                    float("-inf"),
                    dtype=torch.float32,
                    device=device,
                )
                running_toks = torch.zeros(
                    (positions, head_end - head_start, k),
                    dtype=torch.long,
                    device=device,
                )
            score_pages = []
            for page_idx in range(pages):
                page_scores = torch.zeros(
                    (positions, head_end - head_start, page_size_local),
                    dtype=torch.float32,
                    device=device,
                )
                for sub in range(subvecs):
                    q_sub = head_queries[
                        :,
                        :,
                        sub * int(codebooks_in.shape[-1]) : (sub + 1) * int(codebooks_in.shape[-1]),
                    ].reshape(positions * (head_end - head_start), int(codebooks_in.shape[-1]))
                    lut = q_sub @ codebooks_in[int(kv_head), int(page_idx), int(sub)].t().contiguous()
                    page_codes = codes_in[int(kv_head), int(page_idx), :, int(sub)].to(torch.long)
                    page_scores = page_scores + lut.index_select(1, page_codes).reshape(
                        positions,
                        head_end - head_start,
                        page_size_local,
                    )
                page_start = page_starts_dev[int(page_idx)]
                valid = (page_start >= dyn_start_t) & (page_start + page_size_local <= sealed_end_t)
                if not bool(torch.all(valid)):
                    page_scores = page_scores.masked_fill(
                        ~valid.reshape(positions, 1, 1),
                        float("-inf"),
                    )
                if streaming:
                    page_toks = (page_start + page_offsets).expand(
                        positions,
                        head_end - head_start,
                        page_size_local,
                    )
                    cand_vals = torch.cat([running_vals, page_scores], dim=2)
                    cand_toks = torch.cat([running_toks, page_toks], dim=2)
                    running_vals, order = torch.topk(
                        cand_vals,
                        k,
                        dim=2,
                        largest=True,
                        sorted=True,
                    )
                    running_toks = cand_toks.gather(2, order)
                    continue
                score_pages.append(page_scores.to(score_dtype) if score_dtype != torch.float32 else page_scores)
            if streaming:
                top_tokens_out[:, head_start:head_end, :] = running_toks
                top_scores_out[:, head_start:head_end, :] = running_vals
                continue
            scores = torch.cat(score_pages, dim=2)
            vals, idx = torch.topk(
                scores.reshape(positions * (head_end - head_start), total_tokens),
                k,
                dim=1,
                largest=True,
                sorted=True,
            )
            idx = idx.reshape(positions, head_end - head_start, k)
            page_ids = torch.div(idx, page_size_local, rounding_mode="floor")
            rows = idx - page_ids * page_size_local
            toks = page_starts_dev.index_select(0, page_ids.reshape(-1)).reshape_as(idx) + rows
            top_tokens_out[:, head_start:head_end, :] = toks
            top_scores_out[:, head_start:head_end, :] = vals.reshape(positions, head_end - head_start, k)
        return top_tokens_out, top_scores_out

    def torch_lut_prefill_topk_context_lens(
        self,
        queries_in: torch.Tensor,
        context_lens: torch.Tensor,
        codebooks_in: torch.Tensor,
        codes_in: torch.Tensor,
        page_starts_in: torch.Tensor,
        *,
        streaming: bool = False,
        score_dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        args = self.args
        device = self.device
        num_kv_heads = self.num_kv_heads
        group_size = self.group_size
        budget = self.budget
        positions = int(queries_in.shape[0])
        heads = int(queries_in.shape[1])
        pages = int(codebooks_in.shape[1])
        subvecs = int(codebooks_in.shape[2])
        page_size_local = int(codes_in.shape[2])
        total_tokens = int(pages * page_size_local)
        k = min(max(0, int(budget)), total_tokens)
        if k <= 0 or total_tokens <= 0:
            return (
                torch.empty((positions, heads, 0), dtype=torch.long, device=device),
                torch.empty((positions, heads, 0), dtype=torch.float32, device=device),
            )
        top_tokens_out = torch.empty((positions, heads, k), dtype=torch.long, device=device)
        top_scores_out = torch.empty((positions, heads, k), dtype=torch.float32, device=device)
        query_context_lens = context_lens.to(device=device, dtype=torch.long)
        dyn_start_t = torch.minimum(
            torch.full_like(query_context_lens, max(0, int(args.static_prefix))),
            query_context_lens,
        )
        indexed_end_t = torch.maximum(
            dyn_start_t,
            query_context_lens - max(0, int(args.static_suffix)),
        )
        sealed_end_t = dyn_start_t + (
            torch.div(
                torch.clamp(indexed_end_t - dyn_start_t, min=0),
                max(1, int(args.page_size)),
                rounding_mode="floor",
            )
            * max(1, int(args.page_size))
        )
        page_starts_dev = page_starts_in.to(device=device, dtype=torch.long)
        page_offsets = torch.arange(page_size_local, dtype=torch.long, device=device).reshape(1, 1, page_size_local)
        for kv_head in range(num_kv_heads):
            head_start = int(kv_head * group_size)
            head_end = min(heads, head_start + int(group_size))
            if head_start >= head_end:
                continue
            head_queries = queries_in[:, head_start:head_end, :].contiguous()
            if streaming:
                running_vals = torch.full(
                    (positions, head_end - head_start, k),
                    float("-inf"),
                    dtype=torch.float32,
                    device=device,
                )
                running_toks = torch.zeros(
                    (positions, head_end - head_start, k),
                    dtype=torch.long,
                    device=device,
                )
            score_pages = []
            for page_idx in range(pages):
                page_scores = torch.zeros(
                    (positions, head_end - head_start, page_size_local),
                    dtype=torch.float32,
                    device=device,
                )
                for sub in range(subvecs):
                    q_sub = head_queries[
                        :,
                        :,
                        sub * int(codebooks_in.shape[-1]) : (sub + 1) * int(codebooks_in.shape[-1]),
                    ].reshape(positions * (head_end - head_start), int(codebooks_in.shape[-1]))
                    lut = q_sub @ codebooks_in[int(kv_head), int(page_idx), int(sub)].t().contiguous()
                    page_codes = codes_in[int(kv_head), int(page_idx), :, int(sub)].to(torch.long)
                    page_scores = page_scores + lut.index_select(1, page_codes).reshape(
                        positions,
                        head_end - head_start,
                        page_size_local,
                    )
                page_start = page_starts_dev[int(page_idx)]
                valid = (page_start >= dyn_start_t) & (page_start + page_size_local <= sealed_end_t)
                if not bool(torch.all(valid)):
                    page_scores = page_scores.masked_fill(
                        ~valid.reshape(positions, 1, 1),
                        float("-inf"),
                    )
                if streaming:
                    page_toks = (page_start + page_offsets).expand(
                        positions,
                        head_end - head_start,
                        page_size_local,
                    )
                    cand_vals = torch.cat([running_vals, page_scores], dim=2)
                    cand_toks = torch.cat([running_toks, page_toks], dim=2)
                    running_vals, order = torch.topk(
                        cand_vals,
                        k,
                        dim=2,
                        largest=True,
                        sorted=True,
                    )
                    running_toks = cand_toks.gather(2, order)
                    continue
                score_pages.append(page_scores.to(score_dtype) if score_dtype != torch.float32 else page_scores)
            if streaming:
                top_tokens_out[:, head_start:head_end, :] = running_toks
                top_scores_out[:, head_start:head_end, :] = running_vals
                continue
            scores = torch.cat(score_pages, dim=2)
            vals, idx = torch.topk(
                scores.reshape(positions * (head_end - head_start), total_tokens),
                k,
                dim=1,
                largest=True,
                sorted=True,
            )
            idx = idx.reshape(positions, head_end - head_start, k)
            page_ids = torch.div(idx, page_size_local, rounding_mode="floor")
            rows = idx - page_ids * page_size_local
            toks = page_starts_dev.index_select(0, page_ids.reshape(-1)).reshape_as(idx) + rows
            top_tokens_out[:, head_start:head_end, :] = toks
            top_scores_out[:, head_start:head_end, :] = vals.reshape(positions, head_end - head_start, k)
        return top_tokens_out, top_scores_out

    def torch_lut_batched_prefill_topk(
        self,
        queries_in: torch.Tensor,
        codebooks_in: torch.Tensor,
        codes_in: torch.Tensor,
        page_starts_in: torch.Tensor,
        *,
        local_query_start: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        args = self.args
        device = self.device
        num_kv_heads = self.num_kv_heads
        group_size = self.group_size
        budget = self.budget
        positions = int(queries_in.shape[0])
        tile_size = int(getattr(args, "prefill_selector_tile_size", 0))
        if tile_size > 0 and positions > tile_size:
            token_chunks = []
            score_chunks = []
            for tile_start in range(0, positions, tile_size):
                tile_end = min(positions, tile_start + tile_size)
                tile_tokens, tile_scores = self.torch_lut_batched_prefill_topk(
                    queries_in[tile_start:tile_end],
                    codebooks_in,
                    codes_in,
                    page_starts_in,
                    local_query_start=int(local_query_start) + int(tile_start),
                )
                token_chunks.append(tile_tokens)
                score_chunks.append(tile_scores)
            return torch.cat(token_chunks, dim=0), torch.cat(score_chunks, dim=0)
        heads = int(queries_in.shape[1])
        pages = int(codebooks_in.shape[1])
        subvecs = int(codebooks_in.shape[2])
        centroids = int(codebooks_in.shape[3])
        subdim = int(codebooks_in.shape[4])
        page_size_local = int(codes_in.shape[2])
        total_tokens = int(pages * page_size_local)
        k = min(max(0, int(budget)), total_tokens)
        if k <= 0 or total_tokens <= 0:
            return (
                torch.empty((positions, heads, 0), dtype=torch.long, device=device),
                torch.empty((positions, heads, 0), dtype=torch.float32, device=device),
            )
        top_tokens_out = torch.empty((positions, heads, k), dtype=torch.long, device=device)
        top_scores_out = torch.empty((positions, heads, k), dtype=torch.float32, device=device)
        query_context_lens = (
            torch.arange(positions, device=device, dtype=torch.long) + int(local_query_start) + 1
        )
        dyn_start_t = torch.minimum(
            torch.full_like(query_context_lens, max(0, int(args.static_prefix))),
            query_context_lens,
        )
        indexed_end_t = torch.maximum(
            dyn_start_t,
            query_context_lens - max(0, int(args.static_suffix)),
        )
        sealed_end_t = dyn_start_t + (
            torch.div(
                torch.clamp(indexed_end_t - dyn_start_t, min=0),
                max(1, int(args.page_size)),
                rounding_mode="floor",
            )
            * max(1, int(args.page_size))
        )
        page_starts_dev = page_starts_in.to(device=device, dtype=torch.long)
        valid_pages = (
            (page_starts_dev.reshape(1, pages) >= dyn_start_t.reshape(positions, 1))
            & ((page_starts_dev.reshape(1, pages) + page_size_local) <= sealed_end_t.reshape(positions, 1))
        )
        page_block_size = int(getattr(args, "prefill_selector_page_block_size", 0))
        if page_block_size > 0:
            page_block_size = max(1, int(page_block_size))
            for kv_head in range(num_kv_heads):
                head_start = int(kv_head * group_size)
                head_end = min(heads, head_start + int(group_size))
                group_heads = int(head_end - head_start)
                if head_start >= head_end:
                    continue
                q_group = queries_in[:, head_start:head_end, :].reshape(
                    positions,
                    group_heads,
                    subvecs,
                    subdim,
                )
                running_vals = torch.full(
                    (positions, group_heads, k),
                    float("-inf"),
                    dtype=torch.float32,
                    device=device,
                )
                running_toks = torch.zeros((positions, group_heads, k), dtype=torch.long, device=device)
                for page_begin in range(0, pages, page_block_size):
                    page_end = min(pages, page_begin + page_block_size)
                    block_pages = int(page_end - page_begin)
                    codebooks_block = codebooks_in[int(kv_head), page_begin:page_end]
                    # [positions, group_heads, block_pages, subvecs, centroids]
                    lut = torch.einsum("xgsd,bscd->xgbsc", q_group, codebooks_block)
                    flat_lut = lut.reshape(
                        positions * group_heads * block_pages * subvecs,
                        centroids,
                    )
                    code_rows = codes_in[int(kv_head), page_begin:page_end].to(torch.long)
                    code_rows = code_rows.permute(0, 2, 1).contiguous()
                    code_rows = code_rows.reshape(1, 1, block_pages, subvecs, page_size_local).expand(
                        positions,
                        group_heads,
                        block_pages,
                        subvecs,
                        page_size_local,
                    )
                    gathered = torch.gather(
                        flat_lut,
                        1,
                        code_rows.reshape(-1, page_size_local),
                    ).reshape(
                        positions,
                        group_heads,
                        block_pages,
                        subvecs,
                        page_size_local,
                    )
                    block_scores = gathered.sum(dim=3)
                    block_valid = valid_pages[:, page_begin:page_end]
                    block_scores = block_scores.masked_fill(
                        ~block_valid.reshape(positions, 1, block_pages, 1),
                        float("-inf"),
                    ).reshape(positions, group_heads, block_pages * page_size_local)
                    block_offsets = torch.arange(page_size_local, dtype=torch.long, device=device).reshape(1, 1, 1, page_size_local)
                    block_starts = page_starts_dev[page_begin:page_end].reshape(1, 1, block_pages, 1)
                    block_toks = (block_starts + block_offsets).expand(
                        positions,
                        group_heads,
                        block_pages,
                        page_size_local,
                    ).reshape(positions, group_heads, block_pages * page_size_local)
                    cand_vals = torch.cat([running_vals, block_scores], dim=2)
                    cand_toks = torch.cat([running_toks, block_toks], dim=2)
                    running_vals, order = torch.topk(
                        cand_vals,
                        k,
                        dim=2,
                        largest=True,
                        sorted=True,
                    )
                    running_toks = cand_toks.gather(2, order)
                top_tokens_out[:, head_start:head_end, :] = running_toks
                top_scores_out[:, head_start:head_end, :] = running_vals
            return top_tokens_out, top_scores_out
        for kv_head in range(num_kv_heads):
            head_start = int(kv_head * group_size)
            head_end = min(heads, head_start + int(group_size))
            if head_start >= head_end:
                continue
            q_group = queries_in[:, head_start:head_end, :].reshape(
                positions,
                head_end - head_start,
                subvecs,
                subdim,
            )
            # [positions, group_heads, pages, subvecs, centroids]
            lut = torch.einsum(
                "xgsd,yscd->xgysc",
                q_group,
                codebooks_in[int(kv_head)],
            )
            flat_lut = lut.reshape(
                positions * (head_end - head_start) * pages * subvecs,
                centroids,
            )
            code_rows = codes_in[int(kv_head)].to(torch.long).permute(0, 2, 1).contiguous()
            code_rows = code_rows.reshape(1, 1, pages, subvecs, page_size_local).expand(
                positions,
                head_end - head_start,
                pages,
                subvecs,
                page_size_local,
            )
            gathered = torch.gather(
                flat_lut,
                1,
                code_rows.reshape(-1, page_size_local),
            ).reshape(
                positions,
                head_end - head_start,
                pages,
                subvecs,
                page_size_local,
            )
            scores = gathered.sum(dim=3).reshape(
                positions,
                head_end - head_start,
                total_tokens,
            )
            scores = scores.masked_fill(
                ~valid_pages.reshape(positions, 1, pages, 1).expand(
                    positions,
                    head_end - head_start,
                    pages,
                    page_size_local,
                ).reshape(positions, head_end - head_start, total_tokens),
                float("-inf"),
            )
            vals, idx = torch.topk(
                scores.reshape(positions * (head_end - head_start), total_tokens),
                k,
                dim=1,
                largest=True,
                sorted=True,
            )
            idx = idx.reshape(positions, head_end - head_start, k)
            page_ids = torch.div(idx, page_size_local, rounding_mode="floor")
            rows = idx - page_ids * page_size_local
            toks = page_starts_dev.index_select(0, page_ids.reshape(-1)).reshape_as(idx) + rows
            top_tokens_out[:, head_start:head_end, :] = toks
            top_scores_out[:, head_start:head_end, :] = vals.reshape(positions, head_end - head_start, k)
        return top_tokens_out, top_scores_out

    def torch_matmul_k_approx_t(
        self,
        codebooks_in: torch.Tensor,
        codes_in: torch.Tensor,
        *,
        kv_heads_local: int,
        pages: int,
        subvecs: int,
        subdim: int,
        page_size_local: int,
        total_tokens: int,
        dim: int,
    ) -> torch.Tensor:
        args = self.args
        device = self.device
        num_kv_heads = self.num_kv_heads
        group_size = self.group_size
        budget = self.budget
        """Return cached PQ-reconstructed K as [kv_heads, dim, tokens].

        The chunked long-prefill path uses the same page-PQ index for
        every query chunk. Reconstructing the approximate K matrix per
        chunk dominates runtime at long context, so keep one layer-local
        cache entry and evict older prefix views aggressively.
        """

        key = (
            int(codebooks_in.data_ptr()),
            int(codes_in.data_ptr()),
            tuple(int(x) for x in codebooks_in.shape),
            tuple(int(x) for x in codes_in.shape),
            codes_in.dtype,
        )
        cached = self.torch_matmul_k_approx_t_cache.get(key)
        if cached is not None:
            return cached
        self.torch_matmul_k_approx_t_cache.clear()
        flat_page_ids = torch.arange(pages, dtype=torch.long, device=device).repeat_interleave(page_size_local)
        flat_codes = codes_in.to(torch.long).reshape(kv_heads_local, total_tokens, subvecs)
        kv_ids = torch.arange(kv_heads_local, dtype=torch.long, device=device).reshape(
            kv_heads_local,
            1,
        ).expand(kv_heads_local, total_tokens)
        page_ids = flat_page_ids.reshape(1, total_tokens).expand(kv_heads_local, total_tokens)
        k_approx = torch.empty((kv_heads_local, total_tokens, dim), dtype=torch.float32, device=device)
        for sub in range(subvecs):
            k_approx[:, :, sub * subdim : (sub + 1) * subdim] = codebooks_in[
                kv_ids,
                page_ids,
                int(sub),
                flat_codes[:, :, int(sub)],
            ]
        cached = k_approx.transpose(1, 2).contiguous()
        self.torch_matmul_k_approx_t_cache[key] = cached
        return cached

    def torch_matmul_prefill_topk_scores(
        self,
        queries_in: torch.Tensor,
        codebooks_in: torch.Tensor,
        codes_in: torch.Tensor,
        page_starts_in: torch.Tensor,
        *,
        local_query_start: int,
        need_dense_scores: bool,
        need_dense_logsumexp: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        args = self.args
        device = self.device
        num_kv_heads = self.num_kv_heads
        group_size = self.group_size
        budget = self.budget
        positions = int(queries_in.shape[0])
        heads = int(queries_in.shape[1])
        kv_heads_local = int(codebooks_in.shape[0])
        pages = int(codebooks_in.shape[1])
        subvecs = int(codebooks_in.shape[2])
        subdim = int(codebooks_in.shape[4])
        dim = int(subvecs * subdim)
        page_size_local = int(codes_in.shape[2])
        total_tokens = int(pages * page_size_local)
        k = min(max(0, int(budget)), total_tokens)
        tile_size = int(getattr(args, "prefill_selector_tile_size", 0))
        if tile_size > 0 and positions > tile_size and not bool(need_dense_scores):
            token_chunks = []
            score_chunks = []
            for tile_start in range(0, positions, tile_size):
                tile_end = min(positions, tile_start + tile_size)
                tile_tokens, tile_scores, _dense, _lse = self.torch_matmul_prefill_topk_scores(
                    queries_in[tile_start:tile_end],
                    codebooks_in,
                    codes_in,
                    page_starts_in,
                    local_query_start=int(local_query_start) + int(tile_start),
                    need_dense_scores=False,
                    need_dense_logsumexp=False,
                )
                token_chunks.append(tile_tokens)
                score_chunks.append(tile_scores)
            return torch.cat(token_chunks, dim=0), torch.cat(score_chunks, dim=0), None, None
        top_tokens_out = torch.empty((positions, heads, k), dtype=torch.long, device=device)
        top_scores_out = torch.empty((positions, heads, k), dtype=torch.float32, device=device)
        dense_scores_out = (
            torch.empty((positions, heads, total_tokens), dtype=torch.float32, device=device)
            if bool(need_dense_scores)
            else None
        )
        dense_logsumexp_out = (
            torch.empty((positions, heads), dtype=torch.float32, device=device)
            if bool(need_dense_scores) and bool(need_dense_logsumexp)
            else None
        )
        if positions <= 0 or heads <= 0 or total_tokens <= 0:
            return top_tokens_out, top_scores_out, dense_scores_out, dense_logsumexp_out

        k_approx_t = self.torch_matmul_k_approx_t(
            codebooks_in,
            codes_in,
            kv_heads_local=kv_heads_local,
            pages=pages,
            subvecs=subvecs,
            subdim=subdim,
            page_size_local=page_size_local,
            total_tokens=total_tokens,
            dim=dim,
        )

        query_context_lens = (
            torch.arange(positions, device=device, dtype=torch.long) + int(local_query_start) + 1
        )
        dyn_start_t = torch.minimum(
            torch.full_like(query_context_lens, max(0, int(args.static_prefix))),
            query_context_lens,
        )
        indexed_end_t = torch.maximum(
            dyn_start_t,
            query_context_lens - max(0, int(args.static_suffix)),
        )
        sealed_end_t = dyn_start_t + (
            torch.div(
                torch.clamp(indexed_end_t - dyn_start_t, min=0),
                max(1, int(args.page_size)),
                rounding_mode="floor",
            )
            * max(1, int(args.page_size))
        )
        page_starts_dev = page_starts_in.to(device=device, dtype=torch.long)
        valid_pages = (
            (page_starts_dev.reshape(1, pages) >= dyn_start_t.reshape(positions, 1))
            & ((page_starts_dev.reshape(1, pages) + page_size_local) <= sealed_end_t.reshape(positions, 1))
        )
        for kv_head in range(kv_heads_local):
            head_start = int(kv_head * group_size)
            head_end = min(heads, head_start + int(group_size))
            if head_start >= head_end:
                continue
            q_group = queries_in[:, head_start:head_end, :].reshape(
                positions * (head_end - head_start),
                dim,
            )
            scores = torch.matmul(q_group, k_approx_t[int(kv_head)]).reshape(
                positions,
                head_end - head_start,
                total_tokens,
            )
            scores = scores.masked_fill(
                ~valid_pages.reshape(positions, 1, pages, 1).expand(
                    positions,
                    head_end - head_start,
                    pages,
                    page_size_local,
                ).reshape(positions, head_end - head_start, total_tokens),
                float("-inf"),
            )
            if dense_scores_out is not None:
                dense_scores_out[:, head_start:head_end, :] = scores
            if dense_logsumexp_out is not None:
                dense_logsumexp_out[:, head_start:head_end] = torch.logsumexp(
                    scores * (float(dim) ** -0.5),
                    dim=-1,
                )
            if k > 0:
                vals, idx = torch.topk(
                    scores.reshape(positions * (head_end - head_start), total_tokens),
                    k,
                    dim=1,
                    largest=True,
                    sorted=True,
                )
                idx = idx.reshape(positions, head_end - head_start, k)
                top_scores_out[:, head_start:head_end, :] = vals.reshape(
                    positions,
                    head_end - head_start,
                    k,
                )
                page_ids_top = torch.div(idx, page_size_local, rounding_mode="floor")
                rows = idx - page_ids_top * page_size_local
                toks = page_starts_dev.index_select(0, page_ids_top.reshape(-1)).reshape_as(idx) + rows
                top_tokens_out[:, head_start:head_end, :] = toks
        return top_tokens_out, top_scores_out, dense_scores_out, dense_logsumexp_out
