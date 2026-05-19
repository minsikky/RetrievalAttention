#!/usr/bin/env python3
"""Trace-level CPU/GPU parity check for the paged-PQ selector path.

This complements the CUDA extension unit tests.  The unit tests prove the
native kernels match small tensor references; this script checks the real trace
policy boundary: CPU-built page-local PQ state, static/pending token handling,
native GPU top-k, selected-token construction, exact selected attention, and
unit-explicit MB accounting.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.online_ivfpq_simulator import EventBytes, PagedLocalPQIndex
from benchmark.attention_efficiency_threeway_eval import build_pq_index
from benchmark.selector_eval.data.trace import load_trace, static_tokens, unique_tokens
from benchmark.selector_eval.gpu.run_gpu_paged_pq_eval import (
    GPUIndex,
    MB,
    PagePQ,
    load_selector_paged_pq_ext,
    parse_csv_ints,
    rank_paged_pq,
    rank_paged_pq_batched_with_scores,
    selector_bytes_fullscan,
)


def _sync_if_cuda(device: torch.device | str) -> None:
    dev = torch.device(device)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def _cpu_rank_page_pq(query: np.ndarray, index: GPUIndex, *, budget: int) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
    """Rank tokens with the same page-local PQ state used by the GPU path."""

    token_chunks: list[torch.Tensor] = []
    score_chunks: list[torch.Tensor] = []
    query_t = torch.as_tensor(query, dtype=torch.float32, device="cpu")
    for page in index.pages:
        codebooks = page.codebooks.detach().to(device="cpu", dtype=torch.float32)
        codes = page.codes.detach().to(device="cpu", dtype=torch.long)
        subvecs = int(codebooks.shape[0])
        subdim = int(codebooks.shape[-1])
        q_parts = query_t.reshape(subvecs, subdim)
        lut = torch.sum(q_parts[:, None, :] * codebooks, dim=2)
        scores = torch.zeros((int(page.size),), dtype=torch.float32, device="cpu")
        for sub in range(subvecs):
            scores += lut[sub].gather(0, codes[:, sub])
        token_chunks.append(torch.arange(int(page.start), int(page.start) + int(page.size), dtype=torch.long))
        score_chunks.append(scores)
    if not token_chunks:
        empty_i = np.empty((0,), dtype=np.int64)
        empty_f = torch.empty((0,), dtype=torch.float32, device="cpu")
        return empty_i, empty_f, empty_f
    tokens_t = torch.cat(token_chunks)
    scores_t = torch.cat(score_chunks)
    order = torch.argsort(scores_t, descending=True, stable=True)
    k = min(max(0, int(budget)), int(tokens_t.numel()))
    top = order[:k]
    return (
        tokens_t.index_select(0, top).numpy().astype(np.int64, copy=False),
        scores_t.index_select(0, top).detach().clone(),
        scores_t.detach().clone(),
    )


def _cpu_index_args(args: argparse.Namespace, head_dim: int) -> argparse.Namespace:
    return argparse.Namespace(
        static_prefix=int(args.static_prefix),
        static_suffix=int(args.static_suffix),
        paged_pq_page_size=int(args.page_size),
        paged_router_prototypes=16,
        paged_router_merge_rel=0.05,
        paged_router_merge_var=0.0,
        paged_router_max_groups=512,
        paged_pq_permutation="none",
        paged_verify_proj_dim=0,
        pqcache_subvecs=int(args.subvecs),
        pqcache_subbits=int(args.subbits),
        pqcache_kmeans_iters=int(args.kmeans_iters),
        score_key_bytes_per_element=int(args.score_key_bytes),
        attn_key_bytes_per_element=int(args.attn_key_bytes),
        value_bytes_per_element=int(args.value_bytes),
        edge_index_bytes=4,
        graph_offset_bytes=4,
        head_dim=int(head_dim),
        backend="python",
    )


def _cpu_event_total_bytes(events: EventBytes) -> float:
    return float(sum(float(v) for v in events.reads.values()) + sum(float(v) for v in events.writes.values()))


def _cpu_page_state(index: PagedLocalPQIndex) -> tuple[list[int], list[int], list[torch.Tensor], list[torch.Tensor]]:
    starts: list[int] = []
    sizes: list[int] = []
    codebooks: list[torch.Tensor] = []
    codes: list[torch.Tensor] = []
    for page in index.pages:
        starts.append(int(page["token_start"]))
        sizes.append(int(page["size"]))
        codebooks.append(torch.as_tensor(page["codebooks"], dtype=torch.float32, device="cpu"))
        codes.append(torch.as_tensor(page["codes"], dtype=torch.long, device="cpu"))
    return starts, sizes, codebooks, codes


def _gpu_index_from_cpu_index(index: PagedLocalPQIndex, *, device: torch.device) -> GPUIndex:
    pages: list[PagePQ] = []
    for page in index.pages:
        pages.append(
            PagePQ(
                start=int(page["token_start"]),
                size=int(page["size"]),
                codebooks=torch.as_tensor(
                    np.asarray(page["codebooks"], dtype=np.float32),
                    dtype=torch.float32,
                    device=device,
                ),
                codes=torch.as_tensor(
                    np.asarray(page["codes"], dtype=np.int64),
                    dtype=torch.long,
                    device=device,
                ),
                proto_rows=None,
            )
        )
    return GPUIndex(
        pages=pages,
        pending_start=int(index.pending_start),
        indexed_end=int(index.indexed_hi),
        build_seconds=0.0,
        build_read_mb=float(index.init_events.read_bytes) / MB,
        build_write_mb=float(index.init_events.write_bytes) / MB,
    )


def _gpu_page_state(index: GPUIndex) -> tuple[list[int], list[int], list[torch.Tensor], list[torch.Tensor]]:
    starts: list[int] = []
    sizes: list[int] = []
    codebooks: list[torch.Tensor] = []
    codes: list[torch.Tensor] = []
    for page in index.pages:
        starts.append(int(page.start))
        sizes.append(int(page.size))
        codebooks.append(page.codebooks.detach().to(device="cpu", dtype=torch.float32))
        codes.append(page.codes.detach().to(device="cpu", dtype=torch.long))
    return starts, sizes, codebooks, codes


def _page_state_diffs(
    cpu_index: PagedLocalPQIndex,
    gpu_index: GPUIndex,
) -> dict[str, float | bool | int]:
    cpu_starts, cpu_sizes, cpu_codebooks, cpu_codes = _cpu_page_state(cpu_index)
    gpu_starts, gpu_sizes, gpu_codebooks, gpu_codes = _gpu_page_state(gpu_index)
    starts_equal = cpu_starts == gpu_starts
    sizes_equal = cpu_sizes == gpu_sizes
    codebook_diff = 0.0
    codes_equal = len(cpu_codes) == len(gpu_codes)
    compared_pages = min(len(cpu_codes), len(gpu_codes))
    for page_id in range(compared_pages):
        if cpu_codebooks[page_id].shape != gpu_codebooks[page_id].shape:
            codebook_diff = float("inf")
        else:
            diff = torch.max(torch.abs(cpu_codebooks[page_id] - gpu_codebooks[page_id])) if cpu_codebooks[page_id].numel() else torch.tensor(0.0)
            codebook_diff = max(codebook_diff, float(diff))
        if cpu_codes[page_id].shape != gpu_codes[page_id].shape or not torch.equal(cpu_codes[page_id], gpu_codes[page_id]):
            codes_equal = False
    return {
        "page_count_equal": len(cpu_starts) == len(gpu_starts),
        "page_starts_equal": starts_equal,
        "page_sizes_equal": sizes_equal,
        "page_codes_equal": codes_equal,
        "page_codebook_max_abs_diff": codebook_diff,
        "cpu_pages": len(cpu_starts),
        "gpu_pages": len(gpu_starts),
        "cpu_pending_start": int(cpu_index.pending_start),
        "gpu_pending_start": int(gpu_index.pending_start),
        "cpu_indexed_hi": int(cpu_index.indexed_hi),
        "gpu_indexed_end": int(gpu_index.indexed_end),
        "pending_start_equal": int(cpu_index.pending_start) == int(gpu_index.pending_start),
        "indexed_end_equal": int(cpu_index.indexed_hi) == int(gpu_index.indexed_end),
    }


def _cpu_selection_fullscan_torch(
    query: np.ndarray,
    index: PagedLocalPQIndex,
) -> tuple[np.ndarray, torch.Tensor, EventBytes]:
    token_chunks: list[torch.Tensor] = []
    score_chunks: list[torch.Tensor] = []
    events = EventBytes()
    query_t = torch.as_tensor(query, dtype=torch.float32, device="cpu")
    for page in index.pages:
        codebooks = torch.as_tensor(page["codebooks"], dtype=torch.float32, device="cpu")
        codes = torch.as_tensor(page["codes"], dtype=torch.long, device="cpu")
        subvecs = int(codebooks.shape[0])
        subdim = int(codebooks.shape[-1])
        q_parts = query_t.reshape(subvecs, subdim)
        lut = torch.sum(q_parts[:, None, :] * codebooks, dim=2)
        scores = torch.zeros((int(page["size"]),), dtype=torch.float32, device="cpu")
        for sub in range(subvecs):
            scores += lut[sub].gather(0, codes[:, sub])
        token_chunks.append(torch.arange(int(page["token_start"]), int(page["token_start"]) + int(page["size"]), dtype=torch.long))
        score_chunks.append(scores)
        events.read("page_pq_codebooks", index._pq_codebook_bytes_per_page())
        events.read("page_pq_codes", index._pq_code_bytes(int(page["size"])))
    if not token_chunks:
        return np.empty((0,), dtype=np.int64), torch.empty((0,), dtype=torch.float32), events
    tokens_t = torch.cat(token_chunks)
    scores_t = torch.cat(score_chunks)
    order = torch.argsort(scores_t, descending=True, stable=True)
    return tokens_t.index_select(0, order).numpy().astype(np.int64, copy=False), scores_t.index_select(0, order), events


def _build_cpu_compatible_value_vpq(
    index: GPUIndex,
    values_np: np.ndarray,
    *,
    seed: int,
    subbits: int,
    value_subvecs: int,
    value_subbits: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Build V-PQ sidecars with the CPU PagedPQSelector seed convention."""

    if not index.pages:
        raise ValueError("cannot build V-PQ sidecars for an empty page index")
    page_size = int(index.pages[0].size)
    if any(int(page.size) != page_size for page in index.pages):
        raise ValueError("V-PQ parity check requires uniform page sizes")
    actual_subvecs = int(value_subvecs) if int(value_subvecs) > 0 else int(index.pages[0].codes.shape[1])
    actual_subbits = int(value_subbits) if int(value_subbits) > 0 else int(subbits)
    codebooks_np: list[np.ndarray] = []
    codes_np: list[np.ndarray] = []
    for page in index.pages:
        start = int(page.start)
        size = int(page.size)
        block = values_np[start : start + size].astype(np.float32, copy=False)
        codebooks, codes, _subvecs, _centroids = build_pq_index(
            block,
            0,
            block.shape[0],
            subvecs=actual_subvecs,
            subbits=actual_subbits,
            seed=int(seed) + 3571 + int(start),
            max_iter=3,
        )
        codebooks_np.append(codebooks.astype(np.float32, copy=False))
        codes_np.append(codes.astype(np.int64, copy=False))
    codebooks_t = torch.as_tensor(np.stack(codebooks_np, axis=0), dtype=torch.float32, device=device)
    code_dtype = torch.uint8 if int(actual_subbits) <= 8 else torch.long
    codes_t = torch.as_tensor(np.stack(codes_np, axis=0), dtype=code_dtype, device=device)
    page_starts_t = torch.as_tensor([int(page.start) for page in index.pages], dtype=torch.long, device=device)
    return codebooks_t, codes_t, page_starts_t, int(actual_subbits)


def _decode_base_bounds(query_context_len: int, static_prefix: int, static_suffix: int, page_size: int) -> tuple[int, int]:
    prefix_end = min(max(0, int(static_prefix)), int(query_context_len))
    dyn_start = int(prefix_end)
    indexed_end = max(dyn_start, int(query_context_len) - max(0, int(static_suffix)))
    sealed_end = dyn_start + (max(0, indexed_end - dyn_start) // int(page_size)) * int(page_size)
    return int(prefix_end), int(max(sealed_end, prefix_end))


def _decode_token_in_base(token: int, query_context_len: int, static_prefix: int, static_suffix: int, page_size: int) -> bool:
    prefix_end, base_tail_start = _decode_base_bounds(query_context_len, static_prefix, static_suffix, page_size)
    tok = int(token)
    return (0 <= tok < prefix_end) or (base_tail_start <= tok < int(query_context_len))


def _vpq_value_for_token_cpu(
    token: int,
    *,
    value_codebooks: torch.Tensor,
    value_codes: torch.Tensor,
    page_starts: torch.Tensor,
    page_size: int,
) -> torch.Tensor | None:
    if int(page_starts.numel()) == 0:
        return None
    first_start = int(page_starts[0].item())
    page = (int(token) - first_start) // int(page_size)
    if int(token) < first_start or page < 0 or page >= int(page_starts.numel()):
        return None
    row = int(token) - int(page_starts[page].item())
    if row < 0 or row >= int(page_size):
        return None
    codebooks_cpu = value_codebooks.detach().to(device="cpu", dtype=torch.float32)
    codes_cpu = value_codes.detach().to(device="cpu", dtype=torch.long)
    subvecs = int(codebooks_cpu.shape[1])
    subdim = int(codebooks_cpu.shape[-1])
    pieces = []
    for sub in range(subvecs):
        code = int(codes_cpu[page, row, sub].item())
        pieces.append(codebooks_cpu[page, sub, code])
    return torch.cat(pieces, dim=0)


def _decode_vpq_selected_tail_ref(
    *,
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    dense_pq_scores: torch.Tensor,
    value_codebooks: torch.Tensor,
    value_codes: torch.Tensor,
    page_starts: torch.Tensor,
    ranked_tokens: torch.Tensor,
    ranked_scores: torch.Tensor,
    query_context_len: int,
    static_prefix: int,
    static_suffix: int,
    page_size: int,
    exact_value_top: int,
    scale: float,
    tail_blend: float,
) -> tuple[torch.Tensor, dict[str, int]]:
    query_cpu = query.detach().to(device="cpu", dtype=torch.float32)
    keys_cpu = keys.detach().to(device="cpu", dtype=torch.float32)
    values_cpu = values.detach().to(device="cpu", dtype=torch.float32)
    dense_cpu = dense_pq_scores.detach().to(device="cpu", dtype=torch.float32)
    ranked_tokens_cpu = ranked_tokens.detach().to(device="cpu", dtype=torch.long)
    ranked_scores_cpu = ranked_scores.detach().to(device="cpu", dtype=torch.float32)
    page_starts_cpu = page_starts.detach().to(device="cpu", dtype=torch.long)
    prefix_end, base_tail_start = _decode_base_bounds(query_context_len, static_prefix, static_suffix, page_size)
    valid_pages = [
        int(page)
        for page in range(int(page_starts_cpu.numel()))
        if int(page_starts_cpu[page].item()) >= prefix_end
        and int(page_starts_cpu[page].item()) + int(page_size) <= base_tail_start
    ]
    logits: list[torch.Tensor] = []
    vals: list[torch.Tensor] = []
    selected_set: set[int] = set()
    ranked_entries: list[tuple[int, int, torch.Tensor]] = []

    for tok in list(range(0, prefix_end)) + list(range(base_tail_start, int(query_context_len))):
        selected_set.add(int(tok))
        logits.append((keys_cpu[int(tok)] @ query_cpu) * float(scale))
        vals.append(values_cpu[int(tok)])

    for sel, (tok_t, score_t) in enumerate(zip(ranked_tokens_cpu.tolist(), ranked_scores_cpu.tolist(), strict=True)):
        tok = int(tok_t)
        score = float(score_t)
        if not math.isfinite(score) or tok < 0 or tok >= int(query_context_len):
            continue
        if _decode_token_in_base(tok, query_context_len, static_prefix, static_suffix, page_size):
            continue
        if tok in selected_set:
            continue
        selected_set.add(tok)
        logit = (keys_cpu[tok] @ query_cpu) * float(scale)
        ranked_entries.append((int(sel), tok, logit))

    if int(exact_value_top) < 0:
        exact_limit = -int(exact_value_top)
        exact_ranked = {sel for sel, _tok, _logit in ranked_entries if sel < exact_limit}
    elif int(exact_value_top) > 0:
        exact_ranked = {
            sel
            for sel, _tok, _logit in sorted(
                ranked_entries,
                key=lambda item: (float(item[2]), -int(item[0])),
                reverse=True,
            )[: int(exact_value_top)]
        }
    else:
        exact_ranked = set()

    compressed_selected = 0
    exact_selected = 0
    for sel, tok, logit in ranked_entries:
        logits.append(logit)
        if sel in exact_ranked:
            vals.append(values_cpu[tok])
            exact_selected += 1
            continue
        approx = _vpq_value_for_token_cpu(
            tok,
            value_codebooks=value_codebooks,
            value_codes=value_codes,
            page_starts=page_starts_cpu,
            page_size=int(page_size),
        )
        if approx is None:
            vals.append(values_cpu[tok])
            exact_selected += 1
        else:
            vals.append(approx)
            compressed_selected += 1

    tail_count = 0
    if float(tail_blend) > 0.0:
        for page in valid_pages:
            page_start = int(page_starts_cpu[page].item())
            for row in range(int(page_size)):
                tok = page_start + row
                if tok >= int(query_context_len) or tok in selected_set:
                    continue
                approx = _vpq_value_for_token_cpu(
                    tok,
                    value_codebooks=value_codebooks,
                    value_codes=value_codes,
                    page_starts=page_starts_cpu,
                    page_size=int(page_size),
                )
                if approx is None:
                    continue
                logits.append(dense_cpu[page * int(page_size) + row] * float(scale))
                vals.append(approx)
                tail_count += 1

    if not logits:
        return torch.zeros((values_cpu.shape[-1],), dtype=torch.float32), {
            "base_count": 0,
            "ranked_count": 0,
            "tail_count": 0,
            "exact_selected_v": 0,
            "compressed_selected_v": 0,
        }
    full = torch.softmax(torch.stack(logits, dim=0), dim=0) @ torch.stack(vals, dim=0)
    return full.float(), {
        "base_count": len(list(range(0, prefix_end)) + list(range(base_tail_start, int(query_context_len)))),
        "ranked_count": len(ranked_entries),
        "tail_count": int(tail_count),
        "exact_selected_v": int(exact_selected),
        "compressed_selected_v": int(compressed_selected),
    }


def _selected_tokens(base: list[int], ranked: np.ndarray, *, context_len: int, budget: int) -> np.ndarray:
    base_set = set(int(tok) for tok in base)
    add = [int(tok) for tok in ranked.tolist() if int(tok) < int(context_len) and int(tok) not in base_set][: int(budget)]
    return np.asarray(unique_tokens(list(base) + add, context_len=int(context_len)), dtype=np.int64)


def _gpu_exact_selected_output(
    keys: torch.Tensor,
    values: torch.Tensor,
    query: torch.Tensor,
    selected: torch.Tensor,
) -> torch.Tensor:
    if int(selected.numel()) == 0:
        return torch.zeros((values.shape[-1],), dtype=torch.float32, device=values.device)
    selected_keys = keys.index_select(0, selected).float()
    logits = (selected_keys @ query.float()) / math.sqrt(float(query.numel()))
    weights = torch.softmax(logits, dim=0)
    return weights @ values.index_select(0, selected).float()


def _cpu_exact_selected_output(
    scores: torch.Tensor,
    values: torch.Tensor,
    selected: np.ndarray,
) -> torch.Tensor:
    if int(selected.size) == 0:
        return torch.zeros((values.shape[-1],), dtype=torch.float32, device="cpu")
    selected_t = torch.as_tensor(selected, dtype=torch.long, device="cpu")
    logits = scores.index_select(0, selected_t).float()
    weights = torch.softmax(logits, dim=0)
    return weights @ values.index_select(0, selected_t).float()


def _torch_output_error_metrics(dense_out: torch.Tensor, approx_out: torch.Tensor) -> dict[str, float]:
    dense = dense_out.detach().to(device="cpu", dtype=torch.float64)
    approx = approx_out.detach().to(device="cpu", dtype=torch.float64)
    err = approx - dense
    rel_denom = torch.linalg.norm(dense).clamp_min(1e-20)
    cosine_denom = (torch.linalg.norm(dense) * torch.linalg.norm(approx)).clamp_min(1e-20)
    dense_normed = dense / torch.sqrt(torch.mean(dense * dense) + 1e-6).clamp_min(1e-20)
    approx_normed = approx / torch.sqrt(torch.mean(approx * approx) + 1e-6).clamp_min(1e-20)
    normed_denom = torch.linalg.norm(dense_normed).clamp_min(1e-20)
    return {
        "output_relative_l2": float(torch.linalg.norm(err) / rel_denom),
        "output_cosine": float(torch.dot(dense, approx) / cosine_denom),
        "output_rmsnorm_relative_l2": float(torch.linalg.norm(approx_normed - dense_normed) / normed_denom),
    }


def _set_equal(left: np.ndarray, right: np.ndarray) -> bool:
    return set(int(x) for x in left.tolist()) == set(int(x) for x in right.tolist())


def _overlap(left: np.ndarray, right: np.ndarray) -> float:
    if int(left.size) == 0 and int(right.size) == 0:
        return 1.0
    if int(left.size) == 0 or int(right.size) == 0:
        return 0.0
    a = set(int(x) for x in left.tolist())
    b = set(int(x) for x in right.tolist())
    return float(len(a & b)) / float(max(len(a), len(b), 1))


def _mean(values: list[float]) -> float:
    return float(sum(float(x) for x in values) / max(1, len(values)))


def _order_equal(left: np.ndarray, right: np.ndarray) -> bool:
    return int(left.size) == int(right.size) and [int(x) for x in left.tolist()] == [int(x) for x in right.tolist()]


def run() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--decode_lengths", default="500,1000,2000")
    parser.add_argument("--heads", default="0,8")
    parser.add_argument("--budgets", default="256,1024")
    parser.add_argument("--static_prefix", type=int, default=128)
    parser.add_argument("--static_suffix", type=int, default=128)
    parser.add_argument("--page_size", type=int, default=2048)
    parser.add_argument("--subvecs", type=int, default=4)
    parser.add_argument("--subbits", type=int, default=6)
    parser.add_argument("--kmeans_iters", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--key_bytes", type=int, default=2, help="Backward-compatible alias for attn_key_bytes.")
    parser.add_argument("--attn_key_bytes", type=int, default=0)
    parser.add_argument("--score_key_bytes", type=int, default=4)
    parser.add_argument("--value_bytes", type=int, default=2)
    parser.add_argument("--check_vpq_tail", action="store_true")
    parser.add_argument("--value_subvecs", type=int, default=4)
    parser.add_argument("--value_subbits", type=int, default=6)
    parser.add_argument("--exact_value_top", type=int, default=-64)
    parser.add_argument("--tail_blend", type=float, default=1.0)
    parser.add_argument("--vpq_tail_atol", type=float, default=3e-3)
    parser.add_argument("--max_qidx_per_decode", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--score_atol", type=float, default=5e-3)
    parser.add_argument("--output_atol", type=float, default=5e-4)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    if int(args.attn_key_bytes) <= 0:
        args.attn_key_bytes = int(args.key_bytes)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for GPU/CPU parity eval")

    torch.set_grad_enabled(False)
    device = torch.device("cuda")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    trace = load_trace(args.trace)
    decode_lengths = parse_csv_ints(args.decode_lengths)
    heads = parse_csv_ints(args.heads)
    budgets = parse_csv_ints(args.budgets)
    q_indices = trace.q_indices_for_decodes(decode_lengths)
    if int(args.max_qidx_per_decode) > 0:
        limited: list[int] = []
        counts: dict[int, int] = {}
        for qidx in q_indices:
            decode = trace.decode_tokens_for_qidx(int(qidx))
            seen = counts.get(int(decode), 0)
            if seen >= int(args.max_qidx_per_decode):
                continue
            limited.append(int(qidx))
            counts[int(decode)] = seen + 1
        q_indices = limited
    if not q_indices:
        raise SystemExit(f"no q_indices found for decode_lengths={decode_lengths}")

    rows: list[dict] = []
    status = "passed"
    failures: list[str] = []

    for qidx in q_indices:
        position = int(trace.positions[int(qidx)])
        decode_tokens = trace.decode_tokens_for_qidx(int(qidx))
        context_len = position + 1
        dynamic_start = min(max(0, int(args.static_prefix)), int(trace.input_len))
        indexed_end = max(
            dynamic_start,
            min(context_len - max(0, int(args.static_suffix)), int(trace.keys.shape[1])),
        )
        index_cache: dict[int, GPUIndex] = {}
        cpu_index_cache: dict[int, PagedLocalPQIndex] = {}
        for head in heads:
            kv_head = trace.kv_head_for(int(head))
            keys_np = trace.keys[kv_head, :context_len].astype(np.float32, copy=False)
            values_np = trace.values[kv_head, :context_len].astype(np.float32, copy=False)
            query_np = trace.queries[int(head), int(qidx)].astype(np.float32, copy=False)
            keys_cpu_t = torch.as_tensor(keys_np, dtype=torch.float32, device="cpu")
            values_cpu_t = torch.as_tensor(values_np, dtype=torch.float32, device="cpu")
            query_cpu_t = torch.as_tensor(query_np, dtype=torch.float32, device="cpu")
            scores_cpu_t = (keys_cpu_t @ query_cpu_t) / math.sqrt(float(trace.head_dim))
            probs_cpu_t = torch.softmax(scores_cpu_t, dim=0)
            dense_out_cpu_t = probs_cpu_t @ values_cpu_t

            if int(kv_head) not in index_cache:
                seed = int(args.seed) + 2027 * int(kv_head)
                cpu_index_cache[int(kv_head)] = PagedLocalPQIndex(
                    keys=keys_np,
                    init_start=dynamic_start,
                    init_end=indexed_end,
                    args=_cpu_index_args(args, int(trace.head_dim)),
                    seed=seed,
                    router_enabled=False,
                )
                index_cache[int(kv_head)] = _gpu_index_from_cpu_index(
                    cpu_index_cache[int(kv_head)],
                    device=device,
                )
            index = index_cache[int(kv_head)]
            cpu_index = cpu_index_cache[int(kv_head)]
            if not index.pages:
                continue

            page_diffs = _page_state_diffs(cpu_index, index)
            gpu_pending = list(range(max(0, int(index.pending_start)), max(0, min(int(index.indexed_end), context_len))))
            cpu_pending = [
                int(tok)
                for tok in cpu_index.pending_tokens()
                if int(tok) < context_len
            ]
            pending_equal = [int(tok) for tok in cpu_pending] == [int(tok) for tok in gpu_pending]
            base = unique_tokens(
                static_tokens(position, int(args.static_prefix), int(args.static_suffix)) + gpu_pending,
                context_len=context_len,
            )
            cpu_base = unique_tokens(
                static_tokens(position, int(args.static_prefix), int(args.static_suffix)) + cpu_pending,
                context_len=context_len,
            )
            base_equal = [int(tok) for tok in cpu_base] == [int(tok) for tok in base]
            keys_t = torch.as_tensor(keys_np, dtype=torch.float32, device=device)
            values_t = torch.as_tensor(values_np, dtype=torch.float32, device=device)
            query_t = torch.as_tensor(query_np, dtype=torch.float32, device=device)

            for budget in budgets:
                cpu_t0 = time.perf_counter()
                cpu_ranked_all, cpu_scores_all, cpu_selection_events = _cpu_selection_fullscan_torch(query_np, cpu_index)
                cpu_tokens = cpu_ranked_all[: int(budget)]
                cpu_scores = cpu_scores_all[: int(budget)].detach().clone()
                cpu_seconds = time.perf_counter() - cpu_t0
                cpu_selector_mb = _cpu_event_total_bytes(cpu_selection_events) / MB

                for _ in range(max(0, int(args.warmup))):
                    _ = rank_paged_pq(
                        query_t,
                        index,
                        mode="fullscan",
                        selector_backend="cuda_ext",
                        nprobes=[],
                        budget=int(budget),
                        key_bytes=int(args.score_key_bytes),
                        subbits=int(args.subbits),
                    )
                    _ = rank_paged_pq(
                        query_t,
                        index,
                        mode="fullscan",
                        selector_backend="torch",
                        nprobes=[],
                        budget=int(budget),
                        key_bytes=int(args.score_key_bytes),
                        subbits=int(args.subbits),
                    )

                cuda_seconds_samples: list[float] = []
                torch_seconds_samples: list[float] = []
                cuda_tokens_np = np.empty((0,), dtype=np.int64)
                cuda_scores_cpu_t = torch.empty((0,), dtype=torch.float32, device="cpu")
                torch_tokens_np = np.empty((0,), dtype=np.int64)
                selector_mb = 0.0
                for _rep in range(max(1, int(args.repeat))):
                    cuda_tokens, cuda_scores, cuda_seconds, selector_mb, _ = rank_paged_pq(
                        query_t,
                        index,
                        mode="fullscan",
                        selector_backend="cuda_ext",
                        nprobes=[],
                        budget=int(budget),
                        key_bytes=int(args.score_key_bytes),
                        subbits=int(args.subbits),
                    )
                    torch_tokens, torch_scores, torch_seconds, _torch_selector_mb, _ = rank_paged_pq(
                        query_t,
                        index,
                        mode="fullscan",
                        selector_backend="torch",
                        nprobes=[],
                        budget=int(budget),
                        key_bytes=int(args.score_key_bytes),
                        subbits=int(args.subbits),
                    )
                    cuda_seconds_samples.append(float(cuda_seconds))
                    torch_seconds_samples.append(float(torch_seconds))
                    cuda_tokens_np = cuda_tokens.detach().cpu().numpy().astype(np.int64, copy=False)
                    cuda_scores_cpu_t = cuda_scores.detach().to(device="cpu", dtype=torch.float32)
                    torch_tokens_np = torch_tokens[: int(budget)].detach().cpu().numpy().astype(np.int64, copy=False)

                cpu_selected = _selected_tokens(base, cpu_tokens, context_len=context_len, budget=int(budget))
                cuda_selected = _selected_tokens(base, cuda_tokens_np, context_len=context_len, budget=int(budget))
                torch_selected = _selected_tokens(base, torch_tokens_np, context_len=context_len, budget=int(budget))

                cpu_sparse_out_t = _cpu_exact_selected_output(scores_cpu_t, values_cpu_t, cpu_selected)
                selected_t = torch.as_tensor(cuda_selected, dtype=torch.long, device=device)
                gpu_sparse_out_t = _gpu_exact_selected_output(keys_t, values_t, query_t, selected_t)
                _sync_if_cuda(device)
                gpu_sparse_out_cpu_t = gpu_sparse_out_t.detach().to(device="cpu", dtype=torch.float32)
                gpu_vs_cpu_rel = float(
                    torch.linalg.norm(gpu_sparse_out_cpu_t.double() - cpu_sparse_out_t.double())
                    / torch.linalg.norm(cpu_sparse_out_t.double()).clamp_min(1e-20)
                )
                out_metrics = _torch_output_error_metrics(dense_out_cpu_t, gpu_sparse_out_cpu_t)
                vpq_tail_checked = False
                vpq_tail_rel = 0.0
                vpq_tail_abs = 0.0
                vpq_tail_count = 0
                vpq_tail_base_count = 0
                vpq_tail_ranked_count = 0
                vpq_tail_exact_selected_v = 0
                vpq_tail_compressed_selected_v = 0
                vpq_tail_mb = 0.0
                if bool(args.check_vpq_tail):
                    native = load_selector_paged_pq_ext()
                    ranked_for_tail, ranked_scores_for_tail, dense_pq_scores, _dense_seconds, _dense_selector_mb, _ = (
                        rank_paged_pq_batched_with_scores(
                            query_t.reshape(1, -1).contiguous(),
                            index,
                            mode="fullscan",
                            selector_backend="cuda_ext",
                            nprobes=[],
                            budget=int(budget),
                            key_bytes=int(args.score_key_bytes),
                            subbits=int(args.subbits),
                        )
                    )
                    value_codebooks, value_codes, value_page_starts, actual_value_subbits = _build_cpu_compatible_value_vpq(
                        index,
                        values_np,
                        seed=int(args.seed),
                        subbits=int(args.subbits),
                        value_subvecs=int(args.value_subvecs),
                        value_subbits=int(args.value_subbits),
                        device=device,
                    )
                    native_tail_out = native.gqa_decode_vpq_selected_tail_agg_from_scores(
                        query_t.reshape(1, -1).contiguous(),
                        keys_t.reshape(1, context_len, int(trace.head_dim)).contiguous(),
                        values_t.reshape(1, context_len, int(trace.head_dim)).contiguous(),
                        dense_pq_scores.contiguous(),
                        value_codebooks.reshape(
                            1,
                            int(value_codebooks.shape[0]),
                            int(value_codebooks.shape[1]),
                            int(value_codebooks.shape[2]),
                            int(value_codebooks.shape[3]),
                        ).contiguous(),
                        value_codes.reshape(
                            1,
                            int(value_codes.shape[0]),
                            int(value_codes.shape[1]),
                            int(value_codes.shape[2]),
                        ).contiguous(),
                        value_page_starts.contiguous(),
                        ranked_for_tail.contiguous(),
                        ranked_scores_for_tail.contiguous(),
                        1,
                        int(context_len),
                        int(args.static_prefix),
                        int(args.static_suffix),
                        int(args.page_size),
                        int(args.exact_value_top),
                        float(trace.head_dim) ** -0.5,
                        float(args.tail_blend),
                    )
                    ref_tail_out, tail_counts = _decode_vpq_selected_tail_ref(
                        query=query_t,
                        keys=keys_t,
                        values=values_t,
                        dense_pq_scores=dense_pq_scores[0],
                        value_codebooks=value_codebooks,
                        value_codes=value_codes,
                        page_starts=value_page_starts,
                        ranked_tokens=ranked_for_tail[0],
                        ranked_scores=ranked_scores_for_tail[0],
                        query_context_len=int(context_len),
                        static_prefix=int(args.static_prefix),
                        static_suffix=int(args.static_suffix),
                        page_size=int(args.page_size),
                        exact_value_top=int(args.exact_value_top),
                        scale=float(trace.head_dim) ** -0.5,
                        tail_blend=float(args.tail_blend),
                    )
                    _sync_if_cuda(device)
                    native_tail_cpu = native_tail_out[0].detach().to(device="cpu", dtype=torch.float32)
                    diff = native_tail_cpu.double() - ref_tail_out.double()
                    vpq_tail_abs = float(torch.max(torch.abs(diff))) if diff.numel() else 0.0
                    vpq_tail_rel = float(torch.linalg.norm(diff) / torch.linalg.norm(ref_tail_out.double()).clamp_min(1e-20))
                    vpq_tail_checked = True
                    vpq_tail_count = int(tail_counts["tail_count"])
                    vpq_tail_base_count = int(tail_counts["base_count"])
                    vpq_tail_ranked_count = int(tail_counts["ranked_count"])
                    vpq_tail_exact_selected_v = int(tail_counts["exact_selected_v"])
                    vpq_tail_compressed_selected_v = int(tail_counts["compressed_selected_v"])
                    code_bytes = 1 if int(actual_value_subbits) <= 8 else 2
                    vpq_tail_mb = float(
                        int(value_codebooks.numel()) * int(args.value_bytes)
                        + int(vpq_tail_count + vpq_tail_compressed_selected_v) * int(value_codebooks.shape[1]) * code_bytes
                    ) / MB

                score_count = min(int(cpu_scores.numel()), int(cuda_scores_cpu_t.numel()))
                if score_count:
                    score_diff_t = cpu_scores[:score_count].float() - cuda_scores_cpu_t[:score_count].float()
                    score_max_abs_diff = float(torch.max(torch.abs(score_diff_t)))
                else:
                    score_max_abs_diff = 0.0
                cuda_order_equal = _order_equal(cpu_tokens, cuda_tokens_np)
                torch_order_equal = _order_equal(cpu_tokens, torch_tokens_np)
                cuda_selected_set_equal = _set_equal(cpu_selected, cuda_selected)
                torch_selected_set_equal = _set_equal(cpu_selected, torch_selected)
                row_status = "passed"
                reason = ""
                if not cuda_selected_set_equal:
                    row_status = "failed"
                    reason = "cuda-selected-set-mismatch"
                elif score_max_abs_diff > float(args.score_atol):
                    row_status = "failed"
                    reason = f"cuda-score-diff>{float(args.score_atol):g}"
                elif gpu_vs_cpu_rel > float(args.output_atol):
                    row_status = "failed"
                    reason = f"selected-output-diff>{float(args.output_atol):g}"
                elif not all(bool(page_diffs[key]) for key in [
                    "page_count_equal",
                    "page_starts_equal",
                    "page_sizes_equal",
                    "page_codes_equal",
                    "pending_start_equal",
                    "indexed_end_equal",
                ]):
                    row_status = "failed"
                    reason = "page-state-mismatch"
                elif float(page_diffs["page_codebook_max_abs_diff"]) > float(args.score_atol):
                    row_status = "failed"
                    reason = f"page-codebook-diff>{float(args.score_atol):g}"
                elif not pending_equal:
                    row_status = "failed"
                    reason = "pending-token-mismatch"
                elif not base_equal:
                    row_status = "failed"
                    reason = "base-token-mismatch"
                elif vpq_tail_checked and vpq_tail_rel > float(args.vpq_tail_atol):
                    row_status = "failed"
                    reason = f"vpq-tail-diff>{float(args.vpq_tail_atol):g}"
                if row_status != "passed":
                    status = "failed"
                    failures.append(
                        f"decode={decode_tokens} head={head} budget={budget}: {reason} "
                        f"score_diff={score_max_abs_diff:.6g} output_rel={gpu_vs_cpu_rel:.6g}"
                    )

                selected_count = int(cuda_selected.size)
                exact_kv_mb = float(selected_count * int(trace.head_dim) * (int(args.attn_key_bytes) + int(args.value_bytes))) / MB
                expected_selector_mb = selector_bytes_fullscan(
                    index,
                    key_bytes=int(args.score_key_bytes),
                    subbits=int(args.subbits),
                ) / MB
                rows.append(
                    {
                        "status": row_status,
                        "failure_reason": reason,
                        "decode_length": int(decode_tokens),
                        "qidx": int(qidx),
                        "head": int(head),
                        "kv_head": int(kv_head),
                        "context_len": int(context_len),
                        "budget": int(budget),
                        "pages": int(len(index.pages)),
                        "page_size": int(args.page_size),
                        "sealed_tokens": int(sum(int(page.size) for page in index.pages)),
                        "base_tokens": int(len(base)),
                        "cpu_base_tokens": int(len(cpu_base)),
                        "base_tokens_equal": base_equal,
                        "pending_tokens_equal": pending_equal,
                        "selected_tokens": selected_count,
                        "candidate_tokens_cpu_all": int(cpu_ranked_all.size),
                        "candidate_tokens_gpu": int(cuda_tokens_np.size),
                        "cpu_selector_MB_expected": float(cpu_selector_mb),
                        "gpu_selector_MB_expected": float(expected_selector_mb),
                        "cpu_cuda_topk_order_equal": cuda_order_equal,
                        "cpu_torch_topk_order_equal": torch_order_equal,
                        "cpu_cuda_topk_overlap": _overlap(cpu_tokens, cuda_tokens_np),
                        "cpu_torch_topk_overlap": _overlap(cpu_tokens, torch_tokens_np),
                        "cpu_cuda_selected_set_equal": cuda_selected_set_equal,
                        "cpu_torch_selected_set_equal": torch_selected_set_equal,
                        "cpu_cuda_score_max_abs_diff": score_max_abs_diff,
                        "selected_output_gpu_vs_cpu_relL2": gpu_vs_cpu_rel,
                        "vpq_tail_checked": vpq_tail_checked,
                        "vpq_tail_output_relL2": vpq_tail_rel,
                        "vpq_tail_output_linf": vpq_tail_abs,
                        "vpq_tail_tokens": vpq_tail_count,
                        "vpq_tail_base_tokens": vpq_tail_base_count,
                        "vpq_tail_ranked_tokens": vpq_tail_ranked_count,
                        "vpq_tail_exact_selected_v": vpq_tail_exact_selected_v,
                        "vpq_tail_compressed_selected_v": vpq_tail_compressed_selected_v,
                        "vpq_tail_MB_per_query": vpq_tail_mb,
                        "vpq_value_seed_scheme": "cpu_selector_seed_plus_3571_plus_page_start",
                        "attention_mass": float(
                            probs_cpu_t.index_select(0, torch.as_tensor(cuda_selected, dtype=torch.long)).sum()
                        )
                        if selected_count
                        else 0.0,
                        "output_relative_L2": out_metrics["output_relative_l2"],
                        "output_cosine": out_metrics["output_cosine"],
                        "output_rmsnorm_relative_L2": out_metrics["output_rmsnorm_relative_l2"],
                        "cpu_selector_seconds": cpu_seconds,
                        "cuda_selector_seconds_mean": _mean(cuda_seconds_samples),
                        "cuda_selector_seconds_min": float(min(cuda_seconds_samples)) if cuda_seconds_samples else 0.0,
                        "torch_selector_seconds_mean": _mean(torch_seconds_samples),
                        "torch_selector_seconds_min": float(min(torch_seconds_samples)) if torch_seconds_samples else 0.0,
                        "selector_speedup_vs_cpu_mean": float(cpu_seconds / max(_mean(cuda_seconds_samples), 1e-20)),
                        "selector_speedup_vs_torch_mean": float(
                            _mean(torch_seconds_samples) / max(_mean(cuda_seconds_samples), 1e-20)
                        ),
                        "selector_MB_per_query": float(selector_mb),
                        "selector_MB_expected": float(expected_selector_mb),
                        "selector_MB_cpu_gpu_abs_diff": abs(float(cpu_selector_mb) - float(expected_selector_mb)),
                        "exact_KV_MB_per_query": exact_kv_mb,
                        "tail_estimator_MB_per_query": vpq_tail_mb if vpq_tail_checked else 0.0,
                        "step_MB_per_query": float(selector_mb) + exact_kv_mb + (vpq_tail_mb if vpq_tail_checked else 0.0),
                        "pq_build_read_MB": float(index.build_read_mb),
                        "pq_build_write_MB": float(index.build_write_mb),
                        **page_diffs,
                    }
                )

    samples_path = out_dir / "samples.csv"
    with samples_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    (out_dir / "samples.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    summary = {
        "kind": "gpu_cpu_paged_pq_parity",
        "status": status,
        "failures": failures,
        "rows": int(len(rows)),
        "trace": str(args.trace),
        "decode_lengths": decode_lengths,
        "heads": heads,
        "budgets": budgets,
        "page_size": int(args.page_size),
        "subvecs": int(args.subvecs),
        "subbits": int(args.subbits),
        "kmeans_iters": int(args.kmeans_iters),
        "max_cpu_cuda_score_abs_diff": float(max((row["cpu_cuda_score_max_abs_diff"] for row in rows), default=0.0)),
        "max_selected_output_gpu_vs_cpu_relL2": float(
            max((row["selected_output_gpu_vs_cpu_relL2"] for row in rows), default=0.0)
        ),
        "vpq_tail_checked": bool(any(row.get("vpq_tail_checked", False) for row in rows)),
        "max_vpq_tail_output_relL2": float(max((row.get("vpq_tail_output_relL2", 0.0) for row in rows), default=0.0)),
        "max_vpq_tail_output_linf": float(max((row.get("vpq_tail_output_linf", 0.0) for row in rows), default=0.0)),
        "max_vpq_tail_MB_per_query": float(max((row.get("vpq_tail_MB_per_query", 0.0) for row in rows), default=0.0)),
        "min_cpu_cuda_topk_overlap": float(min((row["cpu_cuda_topk_overlap"] for row in rows), default=1.0)),
        "all_cpu_cuda_selected_sets_equal": bool(all(row["cpu_cuda_selected_set_equal"] for row in rows)),
        "all_page_counts_equal": bool(all(row["page_count_equal"] for row in rows)),
        "all_page_starts_equal": bool(all(row["page_starts_equal"] for row in rows)),
        "all_page_sizes_equal": bool(all(row["page_sizes_equal"] for row in rows)),
        "all_page_codes_equal": bool(all(row["page_codes_equal"] for row in rows)),
        "all_pending_tokens_equal": bool(all(row["pending_tokens_equal"] for row in rows)),
        "all_base_tokens_equal": bool(all(row["base_tokens_equal"] for row in rows)),
        "max_page_codebook_abs_diff": float(max((row["page_codebook_max_abs_diff"] for row in rows), default=0.0)),
        "max_selector_MB_cpu_gpu_abs_diff": float(
            max((row["selector_MB_cpu_gpu_abs_diff"] for row in rows), default=0.0)
        ),
        "cuda_selector_seconds_mean": _mean([float(row["cuda_selector_seconds_mean"]) for row in rows]) if rows else 0.0,
        "torch_selector_seconds_mean": _mean([float(row["torch_selector_seconds_mean"]) for row in rows]) if rows else 0.0,
        "cpu_selector_seconds_mean": _mean([float(row["cpu_selector_seconds"]) for row in rows]) if rows else 0.0,
        "selector_speedup_vs_torch_mean": _mean([float(row["selector_speedup_vs_torch_mean"]) for row in rows])
        if rows
        else 0.0,
        "selector_speedup_vs_cpu_mean": _mean([float(row["selector_speedup_vs_cpu_mean"]) for row in rows]) if rows else 0.0,
        "samples_csv": str(samples_path),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))

    if bool(args.strict) and status != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    run()
