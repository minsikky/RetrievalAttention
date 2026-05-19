#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import torch


EXT_ROOT = Path(__file__).resolve().parent
if str(EXT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXT_ROOT))
PROJECT_ROOT = EXT_ROOT.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import selector_paged_pq  # noqa: E402
from benchmark.selector_eval.gpu.run_gpu_paged_pq_eval import build_page_pq_torch, ensure_native_fullscan_pack  # noqa: E402


def reference_fullscan_pq_topk(
    queries: torch.Tensor,
    codebooks: torch.Tensor,
    codes: torch.Tensor,
    page_starts: torch.Tensor,
    budget: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    heads, dim = queries.shape
    pages, subvecs, _centroids, subdim = codebooks.shape
    page_size = codes.shape[1]
    assert dim == subvecs * subdim
    k = min(max(int(budget), 0), pages * page_size)
    if k == 0:
        scores = torch.empty((heads, pages * page_size), dtype=torch.float32, device=queries.device)
        q_parts = queries.reshape(heads, subvecs, subdim)
        for page in range(pages):
            table = torch.einsum("hms,mcs->hmc", q_parts, codebooks[page])
            page_scores = torch.zeros((heads, page_size), dtype=torch.float32, device=queries.device)
            for sub in range(subvecs):
                gather_idx = codes[page, :, sub].to(torch.long).view(1, page_size).expand(heads, page_size)
                page_scores += table[:, sub, :].gather(1, gather_idx)
            scores[:, page * page_size : (page + 1) * page_size] = page_scores
        return (
            torch.empty((heads, 0), dtype=torch.long, device=queries.device),
            torch.empty((heads, 0), dtype=torch.float32, device=queries.device),
            scores,
        )

    q_parts = queries.reshape(heads, subvecs, subdim)
    scores = torch.empty((heads, pages, page_size), dtype=torch.float32, device=queries.device)
    for page in range(pages):
        table = torch.einsum("hms,mcs->hmc", q_parts, codebooks[page])
        page_scores = torch.zeros((heads, page_size), dtype=torch.float32, device=queries.device)
        for sub in range(subvecs):
            gather_idx = codes[page, :, sub].to(torch.long).view(1, page_size).expand(heads, page_size)
            page_scores += table[:, sub, :].gather(1, gather_idx)
        scores[:, page, :] = page_scores
    top_scores, top_indices = torch.topk(scores.reshape(heads, pages * page_size), k, dim=1, largest=True, sorted=True)
    page_ids = torch.div(top_indices, page_size, rounding_mode="floor")
    rows = top_indices - page_ids * page_size
    top_tokens = page_starts.index_select(0, page_ids.reshape(-1)).reshape_as(page_ids) + rows
    return top_tokens, top_scores, scores.reshape(heads, pages * page_size)


def run_case(*, dtype: torch.dtype, budget: int) -> None:
    torch.manual_seed(20260514 + int(budget))
    heads = 5
    pages = 7
    page_size = 19
    subvecs = 4
    centroids = 32
    subdim = 8
    dim = subvecs * subdim
    queries = torch.randn((heads, dim), device="cuda", dtype=torch.float32)
    codebooks = torch.randn((pages, subvecs, centroids, subdim), device="cuda", dtype=torch.float32)
    codes_i64 = torch.randint(0, centroids, (pages, page_size, subvecs), device="cuda", dtype=torch.long)
    codes = codes_i64.to(dtype) if dtype is torch.uint8 else codes_i64
    page_starts = torch.arange(11, 11 + pages * (page_size + 3), page_size + 3, device="cuda", dtype=torch.long)

    got_tokens, got_scores = selector_paged_pq.fullscan_pq_topk(queries, codebooks, codes, page_starts, budget)
    got_tokens_s, got_scores_s, got_dense_scores = selector_paged_pq.fullscan_pq_topk_scores(
        queries,
        codebooks,
        codes,
        page_starts,
        budget,
    )
    ref_tokens, ref_scores, ref_dense_scores = reference_fullscan_pq_topk(queries, codebooks, codes_i64, page_starts, budget)
    torch.cuda.synchronize()

    if got_tokens.shape != ref_tokens.shape:
        raise AssertionError(f"token shape mismatch: {got_tokens.shape} vs {ref_tokens.shape}")
    if got_scores.shape != ref_scores.shape:
        raise AssertionError(f"score shape mismatch: {got_scores.shape} vs {ref_scores.shape}")
    if got_dense_scores.shape != ref_dense_scores.shape:
        raise AssertionError(f"dense score shape mismatch: {got_dense_scores.shape} vs {ref_dense_scores.shape}")
    if got_tokens.numel() and not torch.equal(got_tokens, ref_tokens):
        raise AssertionError(f"token mismatch for dtype={dtype} budget={budget}")
    if got_scores.numel() and not torch.allclose(got_scores, ref_scores, atol=1e-4, rtol=1e-4):
        diff = torch.max(torch.abs(got_scores - ref_scores)).item()
        raise AssertionError(f"score mismatch for dtype={dtype} budget={budget}: max_diff={diff}")
    if got_tokens_s.numel() and not torch.equal(got_tokens_s, ref_tokens):
        raise AssertionError(f"scored token mismatch for dtype={dtype} budget={budget}")
    if got_scores_s.numel() and not torch.allclose(got_scores_s, ref_scores, atol=1e-4, rtol=1e-4):
        diff = torch.max(torch.abs(got_scores_s - ref_scores)).item()
        raise AssertionError(f"scored top-k score mismatch for dtype={dtype} budget={budget}: max_diff={diff}")
    if got_dense_scores.numel() and not torch.allclose(got_dense_scores, ref_dense_scores, atol=1e-4, rtol=1e-4):
        diff = torch.max(torch.abs(got_dense_scores - ref_dense_scores)).item()
        raise AssertionError(f"dense score mismatch for dtype={dtype} budget={budget}: max_diff={diff}")


def run_exact_selected_attention_case(*, heads: int, selected: int, dim: int, total_tokens: int) -> None:
    torch.manual_seed(20260514 + heads * 13 + selected * 17 + dim)
    queries = torch.randn((heads, dim), device="cuda", dtype=torch.float32)
    keys = torch.randn((total_tokens, dim), device="cuda", dtype=torch.float32)
    values = torch.randn((total_tokens, dim), device="cuda", dtype=torch.float32)
    tokens = torch.randint(0, total_tokens, (heads, selected), device="cuda", dtype=torch.long)
    scale = float(dim) ** -0.5

    got = selector_paged_pq.exact_selected_attention(queries, keys, values, tokens, scale)
    ref_rows = []
    for head in range(heads):
        selected_keys = keys.index_select(0, tokens[head])
        logits = (selected_keys @ queries[head]) * scale
        weights = torch.softmax(logits, dim=0)
        ref_rows.append(weights @ values.index_select(0, tokens[head]))
    ref = torch.stack(ref_rows, dim=0)
    torch.cuda.synchronize()

    if got.shape != ref.shape:
        raise AssertionError(f"attention output shape mismatch: {got.shape} vs {ref.shape}")
    if not torch.allclose(got, ref, atol=2e-4, rtol=2e-4):
        diff = torch.max(torch.abs(got - ref)).item()
        raise AssertionError(
            f"exact_selected_attention mismatch for heads={heads} selected={selected} dim={dim}: max_diff={diff}"
        )


def run_gqa_exact_selected_attention_case(
    *,
    heads: int,
    kv_heads: int,
    selected: int,
    dim: int,
    total_tokens: int,
    kv_dtype: torch.dtype = torch.float32,
) -> None:
    torch.manual_seed(20260515 + heads * 13 + selected * 17 + dim)
    if heads % kv_heads != 0:
        raise AssertionError("test expects integer GQA group size")
    group_size = heads // kv_heads
    queries = torch.randn((heads, dim), device="cuda", dtype=torch.float32)
    keys = torch.randn((kv_heads, total_tokens, dim), device="cuda", dtype=torch.float32).to(kv_dtype)
    values = torch.randn((kv_heads, total_tokens, dim), device="cuda", dtype=torch.float32).to(kv_dtype)
    tokens = torch.randint(0, total_tokens, (heads, selected), device="cuda", dtype=torch.long)
    scale = float(dim) ** -0.5

    got = selector_paged_pq.gqa_exact_selected_attention(queries, keys, values, tokens, group_size, scale)
    ref_rows = []
    for head in range(heads):
        kv_head = head // group_size
        selected_keys = keys[kv_head].float().index_select(0, tokens[head])
        logits = (selected_keys @ queries[head]) * scale
        weights = torch.softmax(logits, dim=0)
        ref_rows.append(weights @ values[kv_head].float().index_select(0, tokens[head]))
    ref = torch.stack(ref_rows, dim=0)
    torch.cuda.synchronize()

    if got.shape != ref.shape:
        raise AssertionError(f"GQA attention output shape mismatch: {got.shape} vs {ref.shape}")
    tol = 3e-3 if kv_dtype in {torch.float16, torch.bfloat16} else 2e-4
    if not torch.allclose(got, ref, atol=tol, rtol=tol):
        diff = torch.max(torch.abs(got - ref)).item()
        raise AssertionError(
            f"gqa_exact_selected_attention mismatch for heads={heads} selected={selected} dim={dim} kv_dtype={kv_dtype}: max_diff={diff}"
        )


def run_gqa_fullscan_pq_topk_case(*, dtype: torch.dtype, budget: int) -> None:
    torch.manual_seed(20260516 + int(budget))
    heads = 8
    kv_heads = 2
    group_size = heads // kv_heads
    pages = 5
    page_size = 23
    subvecs = 4
    centroids = 32
    subdim = 8
    dim = subvecs * subdim
    queries = torch.randn((heads, dim), device="cuda", dtype=torch.float32)
    codebooks = torch.randn((kv_heads, pages, subvecs, centroids, subdim), device="cuda", dtype=torch.float32)
    codes_i64 = torch.randint(0, centroids, (kv_heads, pages, page_size, subvecs), device="cuda", dtype=torch.long)
    codes = codes_i64.to(dtype) if dtype is torch.uint8 else codes_i64
    page_starts = torch.arange(17, 17 + pages * (page_size + 5), page_size + 5, device="cuda", dtype=torch.long)

    got_tokens, got_scores = selector_paged_pq.gqa_fullscan_pq_topk(
        queries,
        codebooks,
        codes,
        page_starts,
        group_size,
        budget,
    )
    got_tokens_s, got_scores_s, got_dense_scores = selector_paged_pq.gqa_fullscan_pq_topk_scores(
        queries,
        codebooks,
        codes,
        page_starts,
        group_size,
        budget,
    )
    ref_tokens = []
    ref_scores = []
    ref_dense_scores = []
    for head in range(heads):
        kv_head = head // group_size
        tokens, scores, dense = reference_fullscan_pq_topk(
            queries[head : head + 1],
            codebooks[kv_head],
            codes_i64[kv_head],
            page_starts,
            budget,
        )
        ref_tokens.append(tokens[0])
        ref_scores.append(scores[0])
        ref_dense_scores.append(dense[0])
    ref_tokens_t = torch.stack(ref_tokens, dim=0)
    ref_scores_t = torch.stack(ref_scores, dim=0)
    ref_dense_scores_t = torch.stack(ref_dense_scores, dim=0)
    torch.cuda.synchronize()

    if got_tokens.shape != ref_tokens_t.shape:
        raise AssertionError(f"GQA selector token shape mismatch: {got_tokens.shape} vs {ref_tokens_t.shape}")
    if got_scores.shape != ref_scores_t.shape:
        raise AssertionError(f"GQA selector score shape mismatch: {got_scores.shape} vs {ref_scores_t.shape}")
    if got_dense_scores.shape != ref_dense_scores_t.shape:
        raise AssertionError(f"GQA dense score shape mismatch: {got_dense_scores.shape} vs {ref_dense_scores_t.shape}")
    if got_tokens.numel() and not torch.equal(got_tokens, ref_tokens_t):
        raise AssertionError(f"GQA selector token mismatch for dtype={dtype} budget={budget}")
    if got_scores.numel() and not torch.allclose(got_scores, ref_scores_t, atol=1e-4, rtol=1e-4):
        diff = torch.max(torch.abs(got_scores - ref_scores_t)).item()
        raise AssertionError(f"GQA selector score mismatch for dtype={dtype} budget={budget}: max_diff={diff}")
    if got_tokens_s.numel() and not torch.equal(got_tokens_s, ref_tokens_t):
        raise AssertionError(f"GQA scored selector token mismatch for dtype={dtype} budget={budget}")
    if got_scores_s.numel() and not torch.allclose(got_scores_s, ref_scores_t, atol=1e-4, rtol=1e-4):
        diff = torch.max(torch.abs(got_scores_s - ref_scores_t)).item()
        raise AssertionError(f"GQA scored selector score mismatch for dtype={dtype} budget={budget}: max_diff={diff}")
    if got_dense_scores.numel() and not torch.allclose(got_dense_scores, ref_dense_scores_t, atol=1e-4, rtol=1e-4):
        diff = torch.max(torch.abs(got_dense_scores - ref_dense_scores_t)).item()
        raise AssertionError(f"GQA dense score mismatch for dtype={dtype} budget={budget}: max_diff={diff}")


def run_gqa_causal_prefill_case(*, dtype: torch.dtype, budget: int, kv_dtype: torch.dtype = torch.float32) -> None:
    torch.manual_seed(20260517 + int(budget))
    positions = 11
    heads = 8
    kv_heads = 2
    group_size = heads // kv_heads
    pages = 4
    page_size = 5
    subvecs = 2
    centroids = 16
    subdim = 4
    dim = subvecs * subdim
    query_start = 0
    static_prefix = 3
    static_suffix = 4
    queries = torch.randn((positions, heads, dim), device="cuda", dtype=torch.float32)
    keys = torch.randn((kv_heads, positions, dim), device="cuda", dtype=torch.float32).to(kv_dtype)
    values = torch.randn((kv_heads, positions, dim), device="cuda", dtype=torch.float32).to(kv_dtype)
    codebooks = torch.randn((kv_heads, pages, subvecs, centroids, subdim), device="cuda", dtype=torch.float32)
    codes_i64 = torch.randint(0, centroids, (kv_heads, pages, page_size, subvecs), device="cuda", dtype=torch.long)
    codes = codes_i64.to(dtype) if dtype is torch.uint8 else codes_i64
    page_starts = torch.arange(static_prefix, static_prefix + pages * page_size, page_size, device="cuda", dtype=torch.long)
    scale = float(dim) ** -0.5

    got_tokens, got_scores = selector_paged_pq.gqa_causal_fullscan_pq_topk(
        queries,
        codebooks,
        codes,
        page_starts,
        group_size,
        budget,
        query_start,
        static_prefix,
        static_suffix,
    )
    fused_tokens, fused_scores = selector_paged_pq.gqa_causal_fullscan_pq_topk_fused(
        queries,
        codebooks,
        codes,
        page_starts,
        group_size,
        budget,
        query_start,
        static_prefix,
        static_suffix,
    )
    got_out = selector_paged_pq.gqa_causal_exact_selected_attention(
        queries,
        keys,
        values,
        got_tokens,
        got_scores,
        group_size,
        query_start,
        static_prefix,
        static_suffix,
        page_size,
        scale,
    )

    ref_tokens = torch.empty_like(got_tokens)
    ref_scores = torch.empty_like(got_scores)
    ref_out = torch.empty_like(got_out)
    for pos in range(positions):
        query_context_len = query_start + pos + 1
        dyn_start = min(max(0, static_prefix), query_context_len)
        indexed_end = max(dyn_start, query_context_len - max(0, static_suffix))
        sealed_end = dyn_start + ((max(0, indexed_end - dyn_start) // page_size) * page_size)
        valid_pages = [
            page
            for page in range(pages)
            if int(page_starts[page].item()) >= dyn_start and int(page_starts[page].item()) + page_size <= sealed_end
        ]
        prefix = list(range(0, min(static_prefix, query_context_len)))
        base_tail_start = max(sealed_end, len(prefix))
        base = prefix + list(range(base_tail_start, query_context_len))
        for head in range(heads):
            kv_head = head // group_size
            if valid_pages:
                scores_by_page = []
                q_parts = queries[pos, head].reshape(subvecs, subdim)
                for page in range(pages):
                    if page not in valid_pages:
                        scores_by_page.append(torch.full((page_size,), float("-inf"), device="cuda"))
                        continue
                    table = torch.einsum("ms,mcs->mc", q_parts, codebooks[kv_head, page])
                    page_scores = torch.zeros((page_size,), dtype=torch.float32, device="cuda")
                    for sub in range(subvecs):
                        page_scores += table[sub].gather(0, codes_i64[kv_head, page, :, sub])
                    scores_by_page.append(page_scores)
                dense_scores = torch.cat(scores_by_page, dim=0)
            else:
                dense_scores = torch.full((pages * page_size,), float("-inf"), device="cuda")
            k = got_scores.shape[-1]
            top_scores, top_idx = torch.topk(dense_scores, k, dim=0, largest=True, sorted=True)
            page_ids = torch.div(top_idx, page_size, rounding_mode="floor")
            rows = top_idx - page_ids * page_size
            top_tokens = page_starts.index_select(0, page_ids) + rows
            ref_tokens[pos, head] = top_tokens
            ref_scores[pos, head] = top_scores
            selected = list(base)
            for tok, score in zip(top_tokens.detach().cpu().tolist(), top_scores.detach().cpu().tolist(), strict=True):
                if score == float("-inf"):
                    continue
                tok = int(tok)
                if tok < query_context_len and tok not in selected:
                    selected.append(tok)
            if selected:
                selected_t = torch.as_tensor(selected, dtype=torch.long, device="cuda")
                logits = (keys[kv_head].float().index_select(0, selected_t) @ queries[pos, head]) * scale
                weights = torch.softmax(logits, dim=0)
                ref_out[pos, head] = weights @ values[kv_head].float().index_select(0, selected_t)
            else:
                ref_out[pos, head].zero_()
    torch.cuda.synchronize()

    if got_tokens.shape != ref_tokens.shape:
        raise AssertionError(f"causal GQA selector token shape mismatch: {got_tokens.shape} vs {ref_tokens.shape}")
    if got_scores.shape != ref_scores.shape:
        raise AssertionError(f"causal GQA selector score shape mismatch: {got_scores.shape} vs {ref_scores.shape}")
    if got_tokens.numel() and not torch.equal(got_tokens, ref_tokens):
        raise AssertionError(f"causal GQA selector token mismatch for dtype={dtype} budget={budget}")
    if got_scores.numel() and not torch.allclose(got_scores, ref_scores, atol=1e-4, rtol=1e-4, equal_nan=True):
        diff = torch.max(torch.abs(got_scores - ref_scores).masked_fill(torch.isinf(got_scores - ref_scores), 0)).item()
        raise AssertionError(f"causal GQA selector score mismatch for dtype={dtype} budget={budget}: max_diff={diff}")
    if fused_tokens.shape != ref_tokens.shape:
        raise AssertionError(f"fused causal GQA selector token shape mismatch: {fused_tokens.shape} vs {ref_tokens.shape}")
    if fused_scores.shape != ref_scores.shape:
        raise AssertionError(f"fused causal GQA selector score shape mismatch: {fused_scores.shape} vs {ref_scores.shape}")
    if fused_scores.numel() and not torch.allclose(fused_scores, ref_scores, atol=1e-4, rtol=1e-4, equal_nan=True):
        diff = torch.max(torch.abs(fused_scores - ref_scores).masked_fill(torch.isinf(fused_scores - ref_scores), 0)).item()
        raise AssertionError(f"fused causal GQA selector score mismatch for dtype={dtype} budget={budget}: max_diff={diff}")
    for force_mode in (1, 2):
        forced_tokens, forced_scores = selector_paged_pq.gqa_causal_fullscan_pq_topk_fused_force(
            queries,
            codebooks,
            codes,
            page_starts,
            group_size,
            budget,
            query_start,
            static_prefix,
            static_suffix,
            force_mode,
        )
        if forced_tokens.shape != ref_tokens.shape:
            raise AssertionError(
                f"forced fused causal GQA selector token shape mismatch for mode={force_mode}: "
                f"{forced_tokens.shape} vs {ref_tokens.shape}"
            )
        if forced_scores.shape != ref_scores.shape:
            raise AssertionError(
                f"forced fused causal GQA selector score shape mismatch for mode={force_mode}: "
                f"{forced_scores.shape} vs {ref_scores.shape}"
            )
        finite_forced = torch.isfinite(forced_scores) & torch.isfinite(ref_scores)
        if (
            forced_tokens.numel()
            and bool(finite_forced.any())
            and not torch.equal(forced_tokens[finite_forced], ref_tokens[finite_forced])
        ):
            raise AssertionError(
                f"forced fused causal GQA selector token mismatch for dtype={dtype} "
                f"budget={budget} mode={force_mode}"
            )
        if forced_scores.numel() and not torch.allclose(forced_scores, ref_scores, atol=1e-4, rtol=1e-4, equal_nan=True):
            diff = torch.max(
                torch.abs(forced_scores - ref_scores).masked_fill(torch.isinf(forced_scores - ref_scores), 0)
            ).item()
            raise AssertionError(
                f"forced fused causal GQA selector score mismatch for dtype={dtype} "
                f"budget={budget} mode={force_mode}: max_diff={diff}"
            )
    finite_fused = torch.isfinite(fused_scores)
    if bool(finite_fused.any()):
        for pos in range(positions):
            query_context_len = query_start + pos + 1
            dyn_start = min(max(0, static_prefix), query_context_len)
            indexed_end = max(dyn_start, query_context_len - max(0, static_suffix))
            sealed_end = dyn_start + ((max(0, indexed_end - dyn_start) // page_size) * page_size)
            for head in range(heads):
                kv_head = head // group_size
                for sel in range(fused_scores.shape[-1]):
                    if not bool(finite_fused[pos, head, sel].item()):
                        continue
                    tok = int(fused_tokens[pos, head, sel].item())
                    if tok >= query_context_len:
                        raise AssertionError(
                            f"fused token is non-causal for dtype={dtype} budget={budget}: token={tok} ctx={query_context_len}"
                        )
                    page = (tok - int(page_starts[0].item())) // page_size
                    if page < 0 or page >= pages:
                        raise AssertionError(f"fused token outside indexed pages for dtype={dtype} budget={budget}: token={tok}")
                    page_start = int(page_starts[page].item())
                    if page_start < dyn_start or page_start + page_size > sealed_end:
                        raise AssertionError(f"fused token from invalid causal page for dtype={dtype} budget={budget}: token={tok}")
                    row = tok - page_start
                    q_parts = queries[pos, head].reshape(subvecs, subdim)
                    recomputed = torch.zeros((), device="cuda")
                    for sub in range(subvecs):
                        code = int(codes_i64[kv_head, page, row, sub].item())
                        recomputed = recomputed + q_parts[sub] @ codebooks[kv_head, page, sub, code]
                    got_score = fused_scores[pos, head, sel]
                    if not torch.allclose(got_score, recomputed, atol=1e-4, rtol=1e-4):
                        diff = torch.abs(got_score - recomputed).item()
                        raise AssertionError(
                            f"fused token score/token mismatch for dtype={dtype} budget={budget}: max_diff={diff}"
                        )
    if got_out.shape != ref_out.shape:
        raise AssertionError(f"causal GQA attention shape mismatch: {got_out.shape} vs {ref_out.shape}")
    tol = 3e-3 if kv_dtype in {torch.float16, torch.bfloat16} else 5e-4
    if not torch.allclose(got_out, ref_out, atol=tol, rtol=tol):
        diff = torch.max(torch.abs(got_out - ref_out)).item()
        raise AssertionError(f"causal GQA attention mismatch for dtype={dtype} budget={budget} kv_dtype={kv_dtype}: max_diff={diff}")


def run_gqa_causal_top_pages_case(*, dtype: torch.dtype, page_budget: int) -> None:
    torch.manual_seed(20260519 + int(page_budget))
    positions = 9
    heads = 6
    kv_heads = 2
    group_size = heads // kv_heads
    pages = 5
    page_size = 7
    subvecs = 2
    centroids = 16
    subdim = 4
    dim = subvecs * subdim
    query_start = 0
    static_prefix = 3
    static_suffix = 2
    queries = torch.randn((positions, heads, dim), device="cuda", dtype=torch.float32)
    codebooks = torch.randn((kv_heads, pages, subvecs, centroids, subdim), device="cuda", dtype=torch.float32)
    codes_i64 = torch.randint(0, centroids, (kv_heads, pages, page_size, subvecs), device="cuda", dtype=torch.long)
    codes = codes_i64.to(dtype) if dtype is torch.uint8 else codes_i64
    page_starts = torch.arange(static_prefix, static_prefix + pages * page_size, page_size, device="cuda", dtype=torch.long)

    got_pages, got_scores = selector_paged_pq.gqa_causal_fullscan_pq_top_pages(
        queries,
        codebooks,
        codes,
        page_starts,
        group_size,
        page_budget,
        query_start,
        static_prefix,
        static_suffix,
    )
    k = min(max(0, int(page_budget)), pages)
    ref_pages = torch.empty((positions, heads, k), dtype=torch.long, device="cuda")
    ref_scores = torch.empty((positions, heads, k), dtype=torch.float32, device="cuda")
    for pos in range(positions):
        query_context_len = query_start + pos + 1
        dyn_start = min(max(0, static_prefix), query_context_len)
        indexed_end = max(dyn_start, query_context_len - max(0, static_suffix))
        sealed_end = dyn_start + ((max(0, indexed_end - dyn_start) // page_size) * page_size)
        for head in range(heads):
            kv_head = head // group_size
            candidates: list[tuple[float, int]] = []
            q_parts = queries[pos, head].reshape(subvecs, subdim)
            for page in range(pages):
                page_start = int(page_starts[page].item())
                if not (page_start >= dyn_start and page_start + page_size <= sealed_end):
                    continue
                table = torch.einsum("ms,mcs->mc", q_parts, codebooks[kv_head, page])
                page_scores = torch.zeros((page_size,), dtype=torch.float32, device="cuda")
                for sub in range(subvecs):
                    page_scores += table[sub].gather(0, codes_i64[kv_head, page, :, sub])
                best_score, best_row = torch.max(page_scores, dim=0)
                abs_page = int((page_start + int(best_row.item())) // page_size)
                candidates.append((float(best_score.item()), abs_page))
            candidates.sort(key=lambda x: (-x[0], x[1]))
            out: list[tuple[float, int]] = []
            seen: set[int] = set()
            for score, abs_page in candidates:
                if abs_page in seen:
                    continue
                out.append((score, abs_page))
                seen.add(abs_page)
                if len(out) >= k:
                    break
            while len(out) < k:
                out.append((float("-inf"), -1))
            for rank, (score, abs_page) in enumerate(out):
                ref_scores[pos, head, rank] = score
                ref_pages[pos, head, rank] = abs_page
    torch.cuda.synchronize()

    if got_pages.shape != ref_pages.shape:
        raise AssertionError(f"causal GQA top-page shape mismatch: {got_pages.shape} vs {ref_pages.shape}")
    if got_scores.shape != ref_scores.shape:
        raise AssertionError(f"causal GQA top-page score shape mismatch: {got_scores.shape} vs {ref_scores.shape}")
    if got_pages.numel() and not torch.equal(got_pages, ref_pages):
        raise AssertionError(f"causal GQA top-page id mismatch for dtype={dtype} page_budget={page_budget}")
    if got_scores.numel() and not torch.allclose(got_scores, ref_scores, atol=1e-4, rtol=1e-4, equal_nan=True):
        diff = torch.max(torch.abs(got_scores - ref_scores).masked_fill(torch.isinf(got_scores - ref_scores), 0)).item()
        raise AssertionError(f"causal GQA top-page score mismatch for dtype={dtype} page_budget={page_budget}: {diff}")


def run_gqa_causal_vpq_prefill_case(
    *,
    dtype: torch.dtype,
    budget: int,
    kv_dtype: torch.dtype = torch.float32,
) -> None:
    torch.manual_seed(20260519 + int(budget))
    positions = 13
    heads = 8
    kv_heads = 2
    group_size = heads // kv_heads
    pages = 4
    page_size = 6
    subvecs = 2
    centroids = 16
    subdim = 4
    dim = subvecs * subdim
    query_start = 0
    static_prefix = 3
    static_suffix = 4
    queries = torch.randn((positions, heads, dim), device="cuda", dtype=torch.float32)
    keys = torch.randn((kv_heads, positions, dim), device="cuda", dtype=torch.float32).to(kv_dtype)
    values = torch.randn((kv_heads, positions, dim), device="cuda", dtype=torch.float32).to(kv_dtype)
    codebooks = torch.randn((kv_heads, pages, subvecs, centroids, subdim), device="cuda", dtype=torch.float32)
    codes_i64 = torch.randint(0, centroids, (kv_heads, pages, page_size, subvecs), device="cuda", dtype=torch.long)
    codes = codes_i64.to(dtype) if dtype is torch.uint8 else codes_i64
    value_codebooks = torch.randn((kv_heads, pages, subvecs, centroids, subdim), device="cuda", dtype=torch.float32)
    value_codes_i64 = torch.randint(0, centroids, (kv_heads, pages, page_size, subvecs), device="cuda", dtype=torch.long)
    value_codes = value_codes_i64.to(dtype) if dtype is torch.uint8 else value_codes_i64
    page_starts = torch.arange(static_prefix, static_prefix + pages * page_size, page_size, device="cuda", dtype=torch.long)
    scale = float(dim) ** -0.5

    ranked_tokens, ranked_scores = selector_paged_pq.gqa_causal_fullscan_pq_topk(
        queries,
        codebooks,
        codes,
        page_starts,
        group_size,
        budget,
        query_start,
        static_prefix,
        static_suffix,
    )
    got_out = selector_paged_pq.gqa_causal_vpq_selected_attention(
        queries,
        keys,
        values,
        value_codebooks,
        value_codes,
        page_starts,
        ranked_tokens,
        ranked_scores,
        group_size,
        query_start,
        static_prefix,
        static_suffix,
        page_size,
        scale,
    )

    ref_out = torch.empty_like(got_out)
    for pos in range(positions):
        query_context_len = query_start + pos + 1
        dyn_start = min(max(0, static_prefix), query_context_len)
        indexed_end = max(dyn_start, query_context_len - max(0, static_suffix))
        sealed_end = dyn_start + ((max(0, indexed_end - dyn_start) // page_size) * page_size)
        prefix = list(range(0, min(static_prefix, query_context_len)))
        base_tail_start = max(sealed_end, len(prefix))
        base = prefix + list(range(base_tail_start, query_context_len))
        for head in range(heads):
            kv_head = head // group_size
            selected = list(base)
            for tok, score in zip(
                ranked_tokens[pos, head].detach().cpu().tolist(),
                ranked_scores[pos, head].detach().cpu().tolist(),
                strict=True,
            ):
                if score == float("-inf"):
                    continue
                tok = int(tok)
                if tok < query_context_len and tok not in selected:
                    selected.append(tok)
            if not selected:
                ref_out[pos, head].zero_()
                continue
            selected_t = torch.as_tensor(selected, dtype=torch.long, device="cuda")
            logits = (keys[kv_head].index_select(0, selected_t).float() @ queries[pos, head]) * scale
            weights = torch.softmax(logits, dim=0)
            selected_values = values[kv_head].index_select(0, selected_t).float().clone()
            for row, tok in enumerate(selected):
                page = int((tok - int(page_starts[0].item())) // page_size)
                if tok >= int(page_starts[0].item()) and 0 <= page < pages:
                    page_start = int(page_starts[page].item())
                    page_row = tok - page_start
                    if 0 <= page_row < page_size:
                        pieces = []
                        for sub in range(subvecs):
                            code = int(value_codes_i64[kv_head, page, page_row, sub].item())
                            pieces.append(value_codebooks[kv_head, page, sub, code])
                        selected_values[row] = torch.cat(pieces, dim=0)
            ref_out[pos, head] = weights @ selected_values
    torch.cuda.synchronize()

    if got_out.shape != ref_out.shape:
        raise AssertionError(f"causal GQA V-PQ attention shape mismatch: {got_out.shape} vs {ref_out.shape}")
    tol = 3e-3 if kv_dtype in {torch.float16, torch.bfloat16} else 5e-4
    if not torch.allclose(got_out, ref_out, atol=tol, rtol=tol):
        diff = torch.max(torch.abs(got_out - ref_out)).item()
        raise AssertionError(
            f"causal GQA V-PQ attention mismatch for dtype={dtype} budget={budget} kv_dtype={kv_dtype}: max_diff={diff}"
        )

    exact_value_top = 3
    got_mixed = selector_paged_pq.gqa_causal_vpq_selected_attention_mixed_vpagesize(
        queries,
        keys,
        values,
        value_codebooks,
        value_codes,
        page_starts,
        ranked_tokens,
        ranked_scores,
        group_size,
        query_start,
        static_prefix,
        static_suffix,
        page_size,
        page_size,
        exact_value_top,
        scale,
    )
    ref_mixed = torch.empty_like(got_mixed)
    first_start = int(page_starts[0].item())
    for pos in range(positions):
        query_context_len = query_start + pos + 1
        dyn_start = min(max(0, static_prefix), query_context_len)
        indexed_end = max(dyn_start, query_context_len - max(0, static_suffix))
        sealed_end = dyn_start + ((max(0, indexed_end - dyn_start) // page_size) * page_size)
        prefix = list(range(0, min(static_prefix, query_context_len)))
        base_tail_start = max(sealed_end, len(prefix))
        base = prefix + list(range(base_tail_start, query_context_len))
        for head in range(heads):
            kv_head = head // group_size
            selected = list(base)
            for tok, score in zip(
                ranked_tokens[pos, head].detach().cpu().tolist(),
                ranked_scores[pos, head].detach().cpu().tolist(),
                strict=True,
            ):
                if score == float("-inf"):
                    continue
                tok = int(tok)
                if tok < query_context_len and tok not in selected:
                    selected.append(tok)
            if not selected:
                ref_mixed[pos, head].zero_()
                continue
            selected_t = torch.as_tensor(selected, dtype=torch.long, device="cuda")
            logits = (keys[kv_head].index_select(0, selected_t).float() @ queries[pos, head]) * scale
            weights = torch.softmax(logits, dim=0)
            order = torch.argsort(logits.float(), descending=True, stable=True)
            exact_rows = set(order[: min(len(selected), exact_value_top)].detach().cpu().tolist())
            selected_values = values[kv_head].index_select(0, selected_t).float().clone()
            for row, tok in enumerate(selected):
                if row in exact_rows:
                    continue
                page = int((tok - first_start) // page_size)
                if tok >= first_start and 0 <= page < pages:
                    page_start = int(page_starts[page].item())
                    page_row = tok - page_start
                    if 0 <= page_row < page_size:
                        pieces = []
                        for sub in range(subvecs):
                            code = int(value_codes_i64[kv_head, page, page_row, sub].item())
                            pieces.append(value_codebooks[kv_head, page, sub, code])
                        selected_values[row] = torch.cat(pieces, dim=0)
            ref_mixed[pos, head] = weights @ selected_values
    torch.cuda.synchronize()

    if got_mixed.shape != ref_mixed.shape:
        raise AssertionError(f"mixed V-PQ attention shape mismatch: {got_mixed.shape} vs {ref_mixed.shape}")
    if not torch.allclose(got_mixed, ref_mixed, atol=tol, rtol=tol):
        diff = torch.max(torch.abs(got_mixed - ref_mixed)).item()
        raise AssertionError(
            f"mixed V-PQ attention mismatch for dtype={dtype} budget={budget} kv_dtype={kv_dtype}: max_diff={diff}"
        )

    selector_rank_exact_top = -3
    got_rank_exact = selector_paged_pq.gqa_causal_vpq_selected_attention_mixed_vpagesize(
        queries,
        keys,
        values,
        value_codebooks,
        value_codes,
        page_starts,
        ranked_tokens,
        ranked_scores,
        group_size,
        query_start,
        static_prefix,
        static_suffix,
        page_size,
        page_size,
        selector_rank_exact_top,
        scale,
    )
    ref_rank_exact = torch.empty_like(got_rank_exact)
    selector_rank_limit = -selector_rank_exact_top
    for pos in range(positions):
        query_context_len = query_start + pos + 1
        dyn_start = min(max(0, static_prefix), query_context_len)
        indexed_end = max(dyn_start, query_context_len - max(0, static_suffix))
        sealed_end = dyn_start + ((max(0, indexed_end - dyn_start) // page_size) * page_size)
        prefix = list(range(0, min(static_prefix, query_context_len)))
        base_tail_start = max(sealed_end, len(prefix))
        base = prefix + list(range(base_tail_start, query_context_len))
        for head in range(heads):
            kv_head = head // group_size
            selected = list(base)
            for tok, score in zip(
                ranked_tokens[pos, head].detach().cpu().tolist(),
                ranked_scores[pos, head].detach().cpu().tolist(),
                strict=True,
            ):
                if score == float("-inf"):
                    continue
                tok = int(tok)
                if tok < query_context_len and tok not in selected:
                    selected.append(tok)
            if not selected:
                ref_rank_exact[pos, head].zero_()
                continue
            selected_t = torch.as_tensor(selected, dtype=torch.long, device="cuda")
            logits = (keys[kv_head].index_select(0, selected_t).float() @ queries[pos, head]) * scale
            weights = torch.softmax(logits, dim=0)
            # Negative exact_value_top means exact selected V follows selector order,
            # avoiding an extra exact-logit top-k over the selected set.
            exact_rows = set(range(min(len(selected), selector_rank_limit)))
            selected_values = values[kv_head].index_select(0, selected_t).float().clone()
            for row, tok in enumerate(selected):
                if row in exact_rows:
                    continue
                page = int((tok - first_start) // page_size)
                if tok >= first_start and 0 <= page < pages:
                    page_start = int(page_starts[page].item())
                    page_row = tok - page_start
                    if 0 <= page_row < page_size:
                        pieces = []
                        for sub in range(subvecs):
                            code = int(value_codes_i64[kv_head, page, page_row, sub].item())
                            pieces.append(value_codebooks[kv_head, page, sub, code])
                        selected_values[row] = torch.cat(pieces, dim=0)
            ref_rank_exact[pos, head] = weights @ selected_values
    torch.cuda.synchronize()

    if not torch.allclose(got_rank_exact, ref_rank_exact, atol=tol, rtol=tol):
        diff = torch.max(torch.abs(got_rank_exact - ref_rank_exact)).item()
        raise AssertionError(
            f"selector-rank mixed V-PQ attention mismatch for dtype={dtype} budget={budget} kv_dtype={kv_dtype}: max_diff={diff}"
        )

    value_page_size = page_size * 2
    value_pages = 2
    value_page_starts = torch.arange(
        static_prefix,
        static_prefix + value_pages * value_page_size,
        value_page_size,
        device="cuda",
        dtype=torch.long,
    )
    grouped_value_codebooks = torch.randn(
        (kv_heads, value_pages, subvecs, centroids, subdim),
        device="cuda",
        dtype=torch.float32,
    )
    grouped_value_codes_i64 = torch.randint(
        0,
        centroids,
        (kv_heads, value_pages, value_page_size, subvecs),
        device="cuda",
        dtype=torch.long,
    )
    grouped_value_codes = grouped_value_codes_i64.to(dtype) if dtype is torch.uint8 else grouped_value_codes_i64
    got_grouped = selector_paged_pq.gqa_causal_vpq_selected_attention_vpagesize(
        queries,
        keys,
        values,
        grouped_value_codebooks,
        grouped_value_codes,
        value_page_starts,
        ranked_tokens,
        ranked_scores,
        group_size,
        query_start,
        static_prefix,
        static_suffix,
        page_size,
        value_page_size,
        scale,
    )
    ref_grouped = torch.empty_like(got_grouped)
    first_value_start = int(value_page_starts[0].item())
    for pos in range(positions):
        query_context_len = query_start + pos + 1
        dyn_start = min(max(0, static_prefix), query_context_len)
        indexed_end = max(dyn_start, query_context_len - max(0, static_suffix))
        sealed_end = dyn_start + ((max(0, indexed_end - dyn_start) // page_size) * page_size)
        prefix = list(range(0, min(static_prefix, query_context_len)))
        base_tail_start = max(sealed_end, len(prefix))
        base = prefix + list(range(base_tail_start, query_context_len))
        for head in range(heads):
            kv_head = head // group_size
            selected = list(base)
            for tok, score in zip(
                ranked_tokens[pos, head].detach().cpu().tolist(),
                ranked_scores[pos, head].detach().cpu().tolist(),
                strict=True,
            ):
                if score == float("-inf"):
                    continue
                tok = int(tok)
                if tok < query_context_len and tok not in selected:
                    selected.append(tok)
            if not selected:
                ref_grouped[pos, head].zero_()
                continue
            selected_t = torch.as_tensor(selected, dtype=torch.long, device="cuda")
            logits = (keys[kv_head].index_select(0, selected_t).float() @ queries[pos, head]) * scale
            weights = torch.softmax(logits, dim=0)
            selected_values = values[kv_head].index_select(0, selected_t).float().clone()
            for row, tok in enumerate(selected):
                page = int((tok - first_value_start) // value_page_size)
                if tok >= first_value_start and 0 <= page < value_pages:
                    page_start = int(value_page_starts[page].item())
                    page_row = tok - page_start
                    if 0 <= page_row < value_page_size:
                        pieces = []
                        for sub in range(subvecs):
                            code = int(grouped_value_codes_i64[kv_head, page, page_row, sub].item())
                            pieces.append(grouped_value_codebooks[kv_head, page, sub, code])
                        selected_values[row] = torch.cat(pieces, dim=0)
            ref_grouped[pos, head] = weights @ selected_values
    torch.cuda.synchronize()

    if got_grouped.shape != ref_grouped.shape:
        raise AssertionError(
            f"grouped V-PQ attention shape mismatch: {got_grouped.shape} vs {ref_grouped.shape}"
        )
    if not torch.allclose(got_grouped, ref_grouped, atol=tol, rtol=tol):
        diff = torch.max(torch.abs(got_grouped - ref_grouped)).item()
        raise AssertionError(
            f"grouped V-PQ attention mismatch for dtype={dtype} budget={budget} kv_dtype={kv_dtype}: max_diff={diff}"
        )


def run_gqa_causal_vpq_tail_case(*, dtype: torch.dtype, budget: int) -> None:
    torch.manual_seed(20260520 + int(budget))
    positions = 9
    heads = 4
    kv_heads = 2
    group_size = heads // kv_heads
    pages = 3
    page_size = 5
    subvecs = 2
    centroids = 16
    subdim = 4
    dim = subvecs * subdim
    query_start = 0
    static_prefix = 2
    static_suffix = 3
    tail_blend = 1.0
    queries = torch.randn((positions, heads, dim), device="cuda", dtype=torch.float32)
    keys = torch.randn((kv_heads, positions, dim), device="cuda", dtype=torch.float32)
    values = torch.randn((kv_heads, positions, dim), device="cuda", dtype=torch.float32)
    key_codebooks = torch.randn((kv_heads, pages, subvecs, centroids, subdim), device="cuda", dtype=torch.float32)
    key_codes_i64 = torch.randint(0, centroids, (kv_heads, pages, page_size, subvecs), device="cuda", dtype=torch.long)
    key_codes = key_codes_i64.to(dtype) if dtype is torch.uint8 else key_codes_i64
    value_codebooks = torch.randn((kv_heads, pages, subvecs, centroids, subdim), device="cuda", dtype=torch.float32)
    value_codes_i64 = torch.randint(0, centroids, (kv_heads, pages, page_size, subvecs), device="cuda", dtype=torch.long)
    value_codes = value_codes_i64.to(dtype) if dtype is torch.uint8 else value_codes_i64
    page_starts = torch.arange(static_prefix, static_prefix + pages * page_size, page_size, device="cuda", dtype=torch.long)
    scale = float(dim) ** -0.5
    ranked_tokens, ranked_scores = selector_paged_pq.gqa_causal_fullscan_pq_topk(
        queries,
        key_codebooks,
        key_codes,
        page_starts,
        group_size,
        budget,
        query_start,
        static_prefix,
        static_suffix,
    )
    ranked_tokens_s, ranked_scores_s, dense_pq_scores = selector_paged_pq.gqa_causal_fullscan_pq_topk_scores(
        queries,
        key_codebooks,
        key_codes,
        page_starts,
        group_size,
        budget,
        query_start,
        static_prefix,
        static_suffix,
    )
    if dense_pq_scores.shape != (positions, heads, pages * page_size):
        raise AssertionError(f"causal dense score shape mismatch: {dense_pq_scores.shape}")
    if not torch.equal(ranked_tokens, ranked_tokens_s):
        raise AssertionError("causal score-returning top-k tokens differ from regular top-k")
    if not torch.allclose(ranked_scores, ranked_scores_s, atol=1e-6, rtol=1e-6):
        diff = torch.max(torch.abs(ranked_scores - ranked_scores_s)).item()
        raise AssertionError(f"causal score-returning top-k scores differ: max_diff={diff}")
    got_out = selector_paged_pq.gqa_causal_vpq_tail_attention(
        queries,
        keys,
        values,
        key_codebooks,
        key_codes,
        value_codebooks,
        value_codes,
        page_starts,
        ranked_tokens,
        ranked_scores,
        group_size,
        query_start,
        static_prefix,
        static_suffix,
        page_size,
        scale,
        tail_blend,
    )
    got_from_scores = selector_paged_pq.gqa_causal_vpq_tail_from_scores(
        queries,
        keys,
        values,
        dense_pq_scores,
        value_codebooks,
        value_codes,
        page_starts,
        ranked_tokens,
        ranked_scores,
        group_size,
        query_start,
        static_prefix,
        static_suffix,
        page_size,
        scale,
        tail_blend,
    )
    if not torch.allclose(got_from_scores, got_out, atol=8e-4, rtol=8e-4):
        diff = torch.max(torch.abs(got_from_scores - got_out)).item()
        raise AssertionError(f"causal from-scores tail mismatch with regular tail: max_diff={diff}")

    ref_out = torch.empty_like(got_out)
    for pos in range(positions):
        query_context_len = query_start + pos + 1
        dyn_start = min(max(0, static_prefix), query_context_len)
        indexed_end = max(dyn_start, query_context_len - max(0, static_suffix))
        sealed_end = dyn_start + ((max(0, indexed_end - dyn_start) // page_size) * page_size)
        prefix = list(range(0, min(static_prefix, query_context_len)))
        base_tail_start = max(sealed_end, len(prefix))
        base = prefix + list(range(base_tail_start, query_context_len))
        valid_pages = [
            page
            for page in range(pages)
            if int(page_starts[page].item()) >= dyn_start and int(page_starts[page].item()) + page_size <= sealed_end
        ]
        for head in range(heads):
            kv_head = head // group_size
            logits = []
            vals = []
            selected_set = set()
            for tok in base:
                selected_set.add(int(tok))
                logits.append((keys[kv_head, tok] @ queries[pos, head]) * scale)
                vals.append(values[kv_head, tok])
            for tok, score in zip(
                ranked_tokens[pos, head].detach().cpu().tolist(),
                ranked_scores[pos, head].detach().cpu().tolist(),
                strict=True,
            ):
                tok = int(tok)
                if score == float("-inf") or tok >= query_context_len or tok in selected_set:
                    continue
                selected_set.add(tok)
                logits.append((keys[kv_head, tok] @ queries[pos, head]) * scale)
                vals.append(values[kv_head, tok])
            for page in valid_pages:
                page_start = int(page_starts[page].item())
                for row in range(page_size):
                    tok = page_start + row
                    if tok >= query_context_len or tok in selected_set:
                        continue
                    score = torch.zeros((), device="cuda")
                    for sub in range(subvecs):
                        code = int(key_codes_i64[kv_head, page, row, sub].item())
                        score = score + queries[pos, head, sub * subdim : (sub + 1) * subdim] @ key_codebooks[kv_head, page, sub, code]
                    pieces = []
                    for sub in range(subvecs):
                        code = int(value_codes_i64[kv_head, page, row, sub].item())
                        pieces.append(value_codebooks[kv_head, page, sub, code])
                    logits.append(score * scale)
                    vals.append(torch.cat(pieces, dim=0))
            if logits:
                logits_t = torch.stack(logits, dim=0)
                vals_t = torch.stack(vals, dim=0)
                ref_out[pos, head] = torch.softmax(logits_t, dim=0) @ vals_t
            else:
                ref_out[pos, head].zero_()
    torch.cuda.synchronize()

    if got_out.shape != ref_out.shape:
        raise AssertionError(f"causal GQA V-PQ tail shape mismatch: {got_out.shape} vs {ref_out.shape}")
    if not torch.allclose(got_out, ref_out, atol=8e-4, rtol=8e-4):
        diff = torch.max(torch.abs(got_out - ref_out)).item()
        raise AssertionError(f"causal GQA V-PQ tail mismatch for dtype={dtype} budget={budget}: max_diff={diff}")

    exact_value_top = 1
    got_mixed = selector_paged_pq.gqa_causal_vpq_selected_tail_attention(
        queries,
        keys,
        values,
        key_codebooks,
        key_codes,
        value_codebooks,
        value_codes,
        page_starts,
        ranked_tokens,
        ranked_scores,
        group_size,
        query_start,
        static_prefix,
        static_suffix,
        page_size,
        exact_value_top,
        scale,
        tail_blend,
    )
    got_mixed_from_scores = selector_paged_pq.gqa_causal_vpq_selected_tail_from_scores(
        queries,
        keys,
        values,
        dense_pq_scores,
        value_codebooks,
        value_codes,
        page_starts,
        ranked_tokens,
        ranked_scores,
        group_size,
        query_start,
        static_prefix,
        static_suffix,
        page_size,
        exact_value_top,
        scale,
        tail_blend,
    )
    if not torch.allclose(got_mixed_from_scores, got_mixed, atol=8e-4, rtol=8e-4):
        diff = torch.max(torch.abs(got_mixed_from_scores - got_mixed)).item()
        raise AssertionError(f"causal from-scores mixed-tail mismatch with regular mixed-tail: max_diff={diff}")
    ref_mixed = torch.empty_like(got_mixed)
    for pos in range(positions):
        query_context_len = query_start + pos + 1
        dyn_start = min(max(0, static_prefix), query_context_len)
        indexed_end = max(dyn_start, query_context_len - max(0, static_suffix))
        sealed_end = dyn_start + ((max(0, indexed_end - dyn_start) // page_size) * page_size)
        prefix = list(range(0, min(static_prefix, query_context_len)))
        base_tail_start = max(sealed_end, len(prefix))
        base = prefix + list(range(base_tail_start, query_context_len))
        valid_pages = [
            page
            for page in range(pages)
            if int(page_starts[page].item()) >= dyn_start and int(page_starts[page].item()) + page_size <= sealed_end
        ]
        for head in range(heads):
            kv_head = head // group_size
            logits = []
            vals = []
            selected_set = set()
            ranked_entries = []
            for tok in base:
                selected_set.add(int(tok))
                logits.append((keys[kv_head, tok] @ queries[pos, head]) * scale)
                vals.append(values[kv_head, tok])
            for sel, (tok, score) in enumerate(
                zip(
                    ranked_tokens[pos, head].detach().cpu().tolist(),
                    ranked_scores[pos, head].detach().cpu().tolist(),
                    strict=True,
                )
            ):
                tok = int(tok)
                if score == float("-inf") or tok >= query_context_len or tok in selected_set:
                    continue
                selected_set.add(tok)
                logit = (keys[kv_head, tok] @ queries[pos, head]) * scale
                ranked_entries.append((sel, tok, logit))
            exact_ranked = {
                sel
                for sel, _tok, _logit in sorted(
                    ranked_entries,
                    key=lambda item: float(item[2].detach().cpu().item()),
                    reverse=True,
                )[:exact_value_top]
            }
            for sel, tok, logit in ranked_entries:
                logits.append(logit)
                if sel in exact_ranked:
                    vals.append(values[kv_head, tok])
                else:
                    page = int((tok - int(page_starts[0].item())) // page_size)
                    row = tok - int(page_starts[page].item()) if 0 <= page < pages else -1
                    if 0 <= page < pages and 0 <= row < page_size:
                        pieces = []
                        for sub in range(subvecs):
                            code = int(value_codes_i64[kv_head, page, row, sub].item())
                            pieces.append(value_codebooks[kv_head, page, sub, code])
                        vals.append(torch.cat(pieces, dim=0))
                    else:
                        vals.append(values[kv_head, tok])
            for page in valid_pages:
                page_start = int(page_starts[page].item())
                for row in range(page_size):
                    tok = page_start + row
                    if tok >= query_context_len or tok in selected_set:
                        continue
                    score = torch.zeros((), device="cuda")
                    for sub in range(subvecs):
                        code = int(key_codes_i64[kv_head, page, row, sub].item())
                        score = score + queries[pos, head, sub * subdim : (sub + 1) * subdim] @ key_codebooks[kv_head, page, sub, code]
                    pieces = []
                    for sub in range(subvecs):
                        code = int(value_codes_i64[kv_head, page, row, sub].item())
                        pieces.append(value_codebooks[kv_head, page, sub, code])
                    logits.append(score * scale)
                    vals.append(torch.cat(pieces, dim=0))
            if logits:
                logits_t = torch.stack(logits, dim=0)
                vals_t = torch.stack(vals, dim=0)
                ref_mixed[pos, head] = torch.softmax(logits_t, dim=0) @ vals_t
            else:
                ref_mixed[pos, head].zero_()
    torch.cuda.synchronize()

    if got_mixed.shape != ref_mixed.shape:
        raise AssertionError(f"causal GQA selected V-PQ tail shape mismatch: {got_mixed.shape} vs {ref_mixed.shape}")
    if not torch.allclose(got_mixed, ref_mixed, atol=8e-4, rtol=8e-4):
        diff = torch.max(torch.abs(got_mixed - ref_mixed)).item()
        raise AssertionError(f"causal GQA selected V-PQ tail mismatch for dtype={dtype} budget={budget}: max_diff={diff}")


def run_gqa_decode_vpq_tail_from_scores_case(*, dtype: torch.dtype, budget: int) -> None:
    torch.manual_seed(20260521 + int(budget))
    heads = 4
    kv_heads = 2
    group_size = heads // kv_heads
    pages = 3
    page_size = 5
    subvecs = 2
    centroids = 16
    subdim = 4
    dim = subvecs * subdim
    query_context_len = 9
    static_prefix = 2
    static_suffix = 3
    tail_blend = 1.0
    queries = torch.randn((heads, dim), device="cuda", dtype=torch.float32)
    keys = torch.randn((kv_heads, query_context_len, dim), device="cuda", dtype=torch.float32)
    values = torch.randn((kv_heads, query_context_len, dim), device="cuda", dtype=torch.float32)
    key_codebooks = torch.randn((kv_heads, pages, subvecs, centroids, subdim), device="cuda", dtype=torch.float32)
    key_codes_i64 = torch.randint(0, centroids, (kv_heads, pages, page_size, subvecs), device="cuda", dtype=torch.long)
    key_codes = key_codes_i64.to(dtype) if dtype is torch.uint8 else key_codes_i64
    value_codebooks = torch.randn((kv_heads, pages, subvecs, centroids, subdim), device="cuda", dtype=torch.float32)
    value_codes_i64 = torch.randint(0, centroids, (kv_heads, pages, page_size, subvecs), device="cuda", dtype=torch.long)
    value_codes = value_codes_i64.to(dtype) if dtype is torch.uint8 else value_codes_i64
    page_starts = torch.arange(static_prefix, static_prefix + pages * page_size, page_size, device="cuda", dtype=torch.long)
    scale = float(dim) ** -0.5
    ranked_tokens, ranked_scores = selector_paged_pq.gqa_fullscan_pq_topk(
        queries,
        key_codebooks,
        key_codes,
        page_starts,
        group_size,
        budget,
    )
    dense_rows = []
    for head in range(heads):
        kv_head = head // group_size
        _tokens, _scores, dense = reference_fullscan_pq_topk(
            queries[head : head + 1],
            key_codebooks[kv_head],
            key_codes_i64[kv_head],
            page_starts,
            budget,
        )
        dense_rows.append(dense[0])
    dense_scores = torch.stack(dense_rows, dim=0)
    got_out = selector_paged_pq.gqa_decode_vpq_tail_from_scores(
        queries,
        keys,
        values,
        dense_scores,
        value_codebooks,
        value_codes,
        page_starts,
        ranked_tokens,
        ranked_scores,
        group_size,
        query_context_len,
        static_prefix,
        static_suffix,
        page_size,
        scale,
        tail_blend,
    )

    ref_out = torch.empty_like(got_out)
    prefix = list(range(0, min(static_prefix, query_context_len)))
    indexed_end = max(len(prefix), query_context_len - max(0, static_suffix))
    sealed_end = len(prefix) + ((max(0, indexed_end - len(prefix)) // page_size) * page_size)
    base_tail_start = max(sealed_end, len(prefix))
    base = prefix + list(range(base_tail_start, query_context_len))
    valid_pages = [
        page
        for page in range(pages)
        if int(page_starts[page].item()) >= len(prefix) and int(page_starts[page].item()) + page_size <= sealed_end
    ]
    for head in range(heads):
        kv_head = head // group_size
        logits = []
        vals = []
        selected_set = set()
        for tok in base:
            selected_set.add(int(tok))
            logits.append((keys[kv_head, tok] @ queries[head]) * scale)
            vals.append(values[kv_head, tok])
        for tok, score in zip(
            ranked_tokens[head].detach().cpu().tolist(),
            ranked_scores[head].detach().cpu().tolist(),
            strict=True,
        ):
            tok = int(tok)
            if score == float("-inf") or tok >= query_context_len or tok in selected_set:
                continue
            selected_set.add(tok)
            logits.append((keys[kv_head, tok] @ queries[head]) * scale)
            vals.append(values[kv_head, tok])
        for page in valid_pages:
            page_start = int(page_starts[page].item())
            for row in range(page_size):
                tok = page_start + row
                if tok >= query_context_len or tok in selected_set:
                    continue
                pieces = []
                for sub in range(subvecs):
                    code = int(value_codes_i64[kv_head, page, row, sub].item())
                    pieces.append(value_codebooks[kv_head, page, sub, code])
                logits.append(dense_scores[head, page * page_size + row] * scale)
                vals.append(torch.cat(pieces, dim=0))
        if logits:
            ref_out[head] = torch.softmax(torch.stack(logits, dim=0), dim=0) @ torch.stack(vals, dim=0)
        else:
            ref_out[head].zero_()
    torch.cuda.synchronize()

    if got_out.shape != ref_out.shape:
        raise AssertionError(f"decode V-PQ tail shape mismatch: {got_out.shape} vs {ref_out.shape}")
    if not torch.allclose(got_out, ref_out, atol=8e-4, rtol=8e-4):
        diff = torch.max(torch.abs(got_out - ref_out)).item()
        raise AssertionError(f"decode V-PQ tail mismatch for dtype={dtype} budget={budget}: max_diff={diff}")

    exact_value_top = 1
    got_mixed = selector_paged_pq.gqa_decode_vpq_selected_tail_from_scores(
        queries,
        keys,
        values,
        dense_scores,
        value_codebooks,
        value_codes,
        page_starts,
        ranked_tokens,
        ranked_scores,
        group_size,
        query_context_len,
        static_prefix,
        static_suffix,
        page_size,
        exact_value_top,
        scale,
        tail_blend,
    )
    ref_mixed = torch.empty_like(got_mixed)
    for head in range(heads):
        kv_head = head // group_size
        logits = []
        vals = []
        selected_set = set()
        ranked_entries = []
        for tok in base:
            selected_set.add(int(tok))
            logits.append((keys[kv_head, tok] @ queries[head]) * scale)
            vals.append(values[kv_head, tok])
        for sel, (tok, score) in enumerate(
            zip(
                ranked_tokens[head].detach().cpu().tolist(),
                ranked_scores[head].detach().cpu().tolist(),
                strict=True,
            )
        ):
            tok = int(tok)
            if score == float("-inf") or tok >= query_context_len or tok in selected_set:
                continue
            selected_set.add(tok)
            logit = (keys[kv_head, tok] @ queries[head]) * scale
            ranked_entries.append((sel, tok, logit))
        exact_ranked = {
            sel
            for sel, _tok, _logit in sorted(
                ranked_entries,
                key=lambda item: float(item[2].detach().cpu().item()),
                reverse=True,
            )[:exact_value_top]
        }
        for sel, tok, logit in ranked_entries:
            logits.append(logit)
            if sel in exact_ranked:
                vals.append(values[kv_head, tok])
            else:
                page = int((tok - int(page_starts[0].item())) // page_size)
                row = tok - int(page_starts[page].item()) if 0 <= page < pages else -1
                if 0 <= page < pages and 0 <= row < page_size:
                    pieces = []
                    for sub in range(subvecs):
                        code = int(value_codes_i64[kv_head, page, row, sub].item())
                        pieces.append(value_codebooks[kv_head, page, sub, code])
                    vals.append(torch.cat(pieces, dim=0))
                else:
                    vals.append(values[kv_head, tok])
        for page in valid_pages:
            page_start = int(page_starts[page].item())
            for row in range(page_size):
                tok = page_start + row
                if tok >= query_context_len or tok in selected_set:
                    continue
                pieces = []
                for sub in range(subvecs):
                    code = int(value_codes_i64[kv_head, page, row, sub].item())
                    pieces.append(value_codebooks[kv_head, page, sub, code])
                logits.append(dense_scores[head, page * page_size + row] * scale)
                vals.append(torch.cat(pieces, dim=0))
        if logits:
            ref_mixed[head] = torch.softmax(torch.stack(logits, dim=0), dim=0) @ torch.stack(vals, dim=0)
        else:
            ref_mixed[head].zero_()
    torch.cuda.synchronize()

    if got_mixed.shape != ref_mixed.shape:
        raise AssertionError(f"decode selected V-PQ tail shape mismatch: {got_mixed.shape} vs {ref_mixed.shape}")
    if not torch.allclose(got_mixed, ref_mixed, atol=8e-4, rtol=8e-4):
        diff = torch.max(torch.abs(got_mixed - ref_mixed)).item()
        raise AssertionError(f"decode selected V-PQ tail mismatch for dtype={dtype} budget={budget}: max_diff={diff}")

    got_agg = selector_paged_pq.gqa_decode_vpq_selected_tail_agg_from_scores(
        queries,
        keys,
        values,
        dense_scores,
        value_codebooks,
        value_codes,
        page_starts,
        ranked_tokens,
        ranked_scores,
        group_size,
        query_context_len,
        static_prefix,
        static_suffix,
        page_size,
        exact_value_top,
        scale,
        tail_blend,
    )
    torch.cuda.synchronize()
    if got_agg.shape != ref_mixed.shape:
        raise AssertionError(f"decode aggregated V-PQ tail shape mismatch: {got_agg.shape} vs {ref_mixed.shape}")
    if not torch.allclose(got_agg, ref_mixed, atol=8e-4, rtol=8e-4):
        diff = torch.max(torch.abs(got_agg - ref_mixed)).item()
        raise AssertionError(f"decode aggregated V-PQ tail mismatch for dtype={dtype} budget={budget}: max_diff={diff}")


def run_gqa_decode_vpq_selected_tail_agg_kv_dtype_case(
    *, dtype: torch.dtype, budget: int, kv_dtype: torch.dtype
) -> None:
    torch.manual_seed(20260523 + int(budget))
    heads = 4
    kv_heads = 2
    group_size = heads // kv_heads
    pages = 3
    page_size = 5
    subvecs = 2
    centroids = 16
    subdim = 4
    dim = subvecs * subdim
    query_context_len = 9
    static_prefix = 2
    static_suffix = 3
    tail_blend = 1.0
    exact_value_top = 1
    queries = torch.randn((heads, dim), device="cuda", dtype=torch.float32)
    keys = torch.randn((kv_heads, query_context_len, dim), device="cuda", dtype=torch.float32).to(kv_dtype)
    values = torch.randn((kv_heads, query_context_len, dim), device="cuda", dtype=torch.float32).to(kv_dtype)
    key_codebooks = torch.randn((kv_heads, pages, subvecs, centroids, subdim), device="cuda", dtype=torch.float32)
    key_codes_i64 = torch.randint(0, centroids, (kv_heads, pages, page_size, subvecs), device="cuda", dtype=torch.long)
    key_codes = key_codes_i64.to(dtype) if dtype is torch.uint8 else key_codes_i64
    value_codebooks = torch.randn((kv_heads, pages, subvecs, centroids, subdim), device="cuda", dtype=torch.float32)
    value_codes_i64 = torch.randint(0, centroids, (kv_heads, pages, page_size, subvecs), device="cuda", dtype=torch.long)
    value_codes = value_codes_i64.to(dtype) if dtype is torch.uint8 else value_codes_i64
    page_starts = torch.arange(static_prefix, static_prefix + pages * page_size, page_size, device="cuda", dtype=torch.long)
    scale = float(dim) ** -0.5
    ranked_tokens, ranked_scores = selector_paged_pq.gqa_fullscan_pq_topk(
        queries,
        key_codebooks,
        key_codes,
        page_starts,
        group_size,
        budget,
    )
    dense_rows = []
    for head in range(heads):
        kv_head = head // group_size
        _tokens, _scores, dense = reference_fullscan_pq_topk(
            queries[head : head + 1],
            key_codebooks[kv_head],
            key_codes_i64[kv_head],
            page_starts,
            budget,
        )
        dense_rows.append(dense[0])
    dense_scores = torch.stack(dense_rows, dim=0)
    got = selector_paged_pq.gqa_decode_vpq_selected_tail_agg_from_scores(
        queries,
        keys,
        values,
        dense_scores,
        value_codebooks,
        value_codes,
        page_starts,
        ranked_tokens,
        ranked_scores,
        group_size,
        query_context_len,
        static_prefix,
        static_suffix,
        page_size,
        exact_value_top,
        scale,
        tail_blend,
    )
    prefix = list(range(0, min(static_prefix, query_context_len)))
    indexed_end = max(len(prefix), query_context_len - max(0, static_suffix))
    sealed_end = len(prefix) + ((max(0, indexed_end - len(prefix)) // page_size) * page_size)
    base_tail_start = max(sealed_end, len(prefix))
    base = prefix + list(range(base_tail_start, query_context_len))
    valid_pages = [
        page
        for page in range(pages)
        if int(page_starts[page].item()) >= len(prefix) and int(page_starts[page].item()) + page_size <= sealed_end
    ]
    ref = torch.empty_like(got)
    for head in range(heads):
        kv_head = head // group_size
        logits = []
        vals = []
        selected_set = set()
        ranked_entries = []
        for tok in base:
            selected_set.add(int(tok))
            logits.append((keys[kv_head, tok].float() @ queries[head]) * scale)
            vals.append(values[kv_head, tok].float())
        for sel, (tok, score) in enumerate(
            zip(
                ranked_tokens[head].detach().cpu().tolist(),
                ranked_scores[head].detach().cpu().tolist(),
                strict=True,
            )
        ):
            tok = int(tok)
            if score == float("-inf") or tok >= query_context_len or tok in selected_set:
                continue
            selected_set.add(tok)
            logit = (keys[kv_head, tok].float() @ queries[head]) * scale
            ranked_entries.append((sel, tok, logit))
        exact_ranked = {
            sel
            for sel, _tok, _logit in sorted(
                ranked_entries,
                key=lambda item: float(item[2].detach().cpu().item()),
                reverse=True,
            )[:exact_value_top]
        }
        for sel, tok, logit in ranked_entries:
            logits.append(logit)
            if sel in exact_ranked:
                vals.append(values[kv_head, tok].float())
            else:
                page = int((tok - int(page_starts[0].item())) // page_size)
                row = tok - int(page_starts[page].item()) if 0 <= page < pages else -1
                if 0 <= page < pages and 0 <= row < page_size:
                    pieces = []
                    for sub in range(subvecs):
                        code = int(value_codes_i64[kv_head, page, row, sub].item())
                        pieces.append(value_codebooks[kv_head, page, sub, code])
                    vals.append(torch.cat(pieces, dim=0))
                else:
                    vals.append(values[kv_head, tok].float())
        for page in valid_pages:
            page_start = int(page_starts[page].item())
            for row in range(page_size):
                tok = page_start + row
                if tok >= query_context_len or tok in selected_set:
                    continue
                pieces = []
                for sub in range(subvecs):
                    code = int(value_codes_i64[kv_head, page, row, sub].item())
                    pieces.append(value_codebooks[kv_head, page, sub, code])
                logits.append(dense_scores[head, page * page_size + row] * scale)
                vals.append(torch.cat(pieces, dim=0))
        if logits:
            ref[head] = torch.softmax(torch.stack(logits, dim=0), dim=0) @ torch.stack(vals, dim=0)
        else:
            ref[head].zero_()
    torch.cuda.synchronize()
    if not torch.allclose(got, ref, atol=2e-3, rtol=2e-3):
        diff = torch.max(torch.abs(got - ref)).item()
        raise AssertionError(
            f"decode aggregated V-PQ tail mismatch for dtype={dtype} budget={budget} kv_dtype={kv_dtype}: "
            f"max_diff={diff}"
        )


def run_torch_page_pq_builder_lossless_case() -> None:
    torch.manual_seed(20260518)
    total_tokens = 257
    dynamic_start = 17
    page_size = 64
    pages = 3
    indexed_end = dynamic_start + pages * page_size
    heads = 5
    subvecs = 4
    subbits = 6
    subdim = 8
    dim = subvecs * subdim
    budget = 37
    keys = torch.randn((total_tokens, dim), device="cuda", dtype=torch.float32)
    queries = torch.randn((heads, dim), device="cuda", dtype=torch.float32)

    index = build_page_pq_torch(
        keys,
        dynamic_start=dynamic_start,
        indexed_end=indexed_end,
        page_size=page_size,
        subvecs=subvecs,
        subbits=subbits,
        kmeans_iters=1,
        seed=20260518,
        key_bytes=2,
        router_enabled=False,
        router_prototypes=16,
        router_merge_rel=0.05,
        router_merge_var=0.0,
        router_max_groups=512,
        device=torch.device("cuda"),
    )
    codebooks, codes, page_starts = ensure_native_fullscan_pack(index, subbits=subbits)
    got_tokens, got_scores, got_dense_scores = selector_paged_pq.fullscan_pq_topk_scores(
        queries,
        codebooks,
        codes,
        page_starts,
        budget,
    )
    flat_keys = keys[dynamic_start:indexed_end].reshape(pages * page_size, dim)
    exact_scores = queries @ flat_keys.T
    ref_scores, ref_ordinals = torch.topk(exact_scores, budget, dim=1, largest=True, sorted=True)
    ref_page_ids = torch.div(ref_ordinals, page_size, rounding_mode="floor")
    ref_rows = ref_ordinals - ref_page_ids * page_size
    ref_tokens = page_starts.index_select(0, ref_page_ids.reshape(-1)).reshape_as(ref_page_ids) + ref_rows
    torch.cuda.synchronize()

    if not torch.allclose(got_dense_scores, exact_scores, atol=1e-4, rtol=1e-4):
        diff = torch.max(torch.abs(got_dense_scores - exact_scores)).item()
        raise AssertionError(f"torch page-PQ builder score mismatch: max_diff={diff}")
    if not torch.equal(got_tokens, ref_tokens):
        raise AssertionError("torch page-PQ builder top-k token mismatch")
    if not torch.allclose(got_scores, ref_scores, atol=1e-4, rtol=1e-4):
        diff = torch.max(torch.abs(got_scores - ref_scores)).item()
        raise AssertionError(f"torch page-PQ builder top-k score mismatch: max_diff={diff}")


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    for dtype in (torch.uint8, torch.long):
        for budget in (0, 1, 17, 133, 1000):
            run_case(dtype=dtype, budget=budget)
    for selected in (1, 7, 64, 257):
        run_exact_selected_attention_case(heads=4, selected=selected, dim=128, total_tokens=513)
        run_gqa_exact_selected_attention_case(heads=8, kv_heads=2, selected=selected, dim=128, total_tokens=513)
    for kv_dtype in (torch.float16, torch.bfloat16):
        run_gqa_exact_selected_attention_case(
            heads=8,
            kv_heads=2,
            selected=17,
            dim=128,
            total_tokens=513,
            kv_dtype=kv_dtype,
        )
    for dtype in (torch.uint8, torch.long):
        for budget in (0, 1, 31, 128, 1000):
            run_gqa_fullscan_pq_topk_case(dtype=dtype, budget=budget)
        for budget in (0, 1, 7, 64, 100):
            run_gqa_causal_prefill_case(dtype=dtype, budget=budget)
            run_gqa_causal_vpq_prefill_case(dtype=dtype, budget=budget)
        for page_budget in (0, 1, 3, 9):
            run_gqa_causal_top_pages_case(dtype=dtype, page_budget=page_budget)
        for kv_dtype in (torch.float16, torch.bfloat16):
            run_gqa_causal_prefill_case(dtype=dtype, budget=7, kv_dtype=kv_dtype)
            run_gqa_causal_vpq_prefill_case(dtype=dtype, budget=7, kv_dtype=kv_dtype)
        for budget in (0, 3, 20):
            run_gqa_causal_vpq_tail_case(dtype=dtype, budget=budget)
            run_gqa_decode_vpq_tail_from_scores_case(dtype=dtype, budget=budget)
        for kv_dtype in (torch.float16, torch.bfloat16):
            run_gqa_decode_vpq_selected_tail_agg_kv_dtype_case(dtype=dtype, budget=20, kv_dtype=kv_dtype)
    run_torch_page_pq_builder_lossless_case()
    print("fullscan_pq_topk CUDA extension matches torch reference")


if __name__ == "__main__":
    main()
