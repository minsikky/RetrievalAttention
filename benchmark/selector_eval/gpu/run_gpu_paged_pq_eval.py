#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.attention_efficiency_threeway_eval import build_pq_index, lloyd_kmeans
from benchmark.selector_eval.data.trace import attention_probs, load_trace, static_tokens, unique_tokens
from benchmark.selector_eval.metrics.attention import _output_error_metrics


MB = 1024.0 * 1024.0


def _sync_if_cuda(device: torch.device | str) -> None:
    dev = torch.device(device)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def parse_csv_ints(text: str) -> list[int]:
    return [int(part.strip()) for part in str(text).split(",") if part.strip()]


def parse_csv_names(text: str) -> list[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


@dataclass
class PagePQ:
    start: int
    size: int
    codebooks: torch.Tensor
    codes: torch.Tensor
    proto_rows: list[np.ndarray] | None = None


@dataclass
class GPUIndex:
    pages: list[PagePQ]
    pending_start: int
    indexed_end: int
    build_seconds: float
    build_read_mb: float
    build_write_mb: float
    router_group_means: torch.Tensor | None = None
    router_group_tokens: list[torch.Tensor] | None = None
    router_group_member_refs: list[int] | None = None


def build_page_pq_gpu(
    keys_np: np.ndarray,
    *,
    dynamic_start: int,
    indexed_end: int,
    page_size: int,
    subvecs: int,
    subbits: int,
    kmeans_iters: int,
    seed: int,
    key_bytes: int,
    router_enabled: bool,
    router_prototypes: int,
    router_merge_rel: float,
    router_merge_var: float,
    router_max_groups: int,
    device: torch.device,
) -> GPUIndex:
    t0 = time.perf_counter()
    pages: list[PagePQ] = []
    write_bytes = 0.0
    read_bytes = 0.0
    cursor = int(dynamic_start)
    page_id = 0
    groups: list[dict] = []

    def merge_page_prototypes(page_idx: int, centers: np.ndarray, proto_rows: list[np.ndarray], proto_sse: list[float]) -> None:
        for proto_id in range(centers.shape[0]):
            count = int(proto_rows[proto_id].size)
            if count <= 0:
                continue
            mean = centers[proto_id].astype(np.float32, copy=True)
            sse = float(proto_sse[proto_id])
            best_gid = -1
            best_delta = float("inf")
            force_gid = -1
            force_delta = float("inf")
            for gid, group in enumerate(groups):
                g_count = int(group["count"])
                if g_count <= 0:
                    continue
                diff = group["mean"] - mean
                dist2 = float(np.dot(diff, diff))
                delta = float(g_count * count) / float(g_count + count) * dist2
                if delta < force_delta:
                    force_delta = delta
                    force_gid = gid
                rel = delta / max(float(group["sse"]) + sse, 1e-6)
                merged_var = (float(group["sse"]) + sse + delta) / float(g_count + count)
                if rel <= float(router_merge_rel) or (float(router_merge_var) > 0.0 and merged_var <= float(router_merge_var)):
                    if delta < best_delta:
                        best_delta = delta
                        best_gid = gid
            can_create_group = int(router_max_groups) <= 0 or len(groups) < int(router_max_groups)
            if best_gid < 0 and can_create_group:
                groups.append({"mean": mean, "count": count, "sse": sse, "members": [(int(page_idx), int(proto_id))]})
                continue
            if best_gid < 0:
                best_gid = force_gid
                best_delta = force_delta
            if best_gid < 0:
                continue
            group = groups[best_gid]
            old_count = int(group["count"])
            new_count = old_count + count
            group["mean"] = (group["mean"].astype(np.float32, copy=False) * float(old_count) + mean * float(count)) / float(new_count)
            group["sse"] = float(group["sse"]) + sse + best_delta
            group["count"] = new_count
            group["members"].append((int(page_idx), int(proto_id)))

    indexed_end = int(indexed_end)
    page_size = max(1, int(page_size))
    sealed_end = int(dynamic_start) + ((max(0, indexed_end - int(dynamic_start)) // page_size) * page_size)
    while cursor < sealed_end:
        end = min(sealed_end, cursor + page_size)
        block = keys_np[cursor:end].astype(np.float32, copy=False)
        if block.shape[0] == 0:
            break
        codebooks, codes, _subvecs, _centroids = build_pq_index(
            block,
            0,
            block.shape[0],
            subvecs=int(subvecs),
            subbits=int(subbits),
            seed=int(seed) + 1009 * page_id + int(cursor),
            max_iter=int(kmeans_iters),
        )
        codebooks_t = torch.as_tensor(codebooks, dtype=torch.float32, device=device)
        codes_t = torch.as_tensor(codes.astype(np.int64, copy=False), dtype=torch.long, device=device)
        proto_rows = None
        if bool(router_enabled):
            proto_centers, proto_assign = lloyd_kmeans(
                block,
                max(1, min(int(router_prototypes), int(block.shape[0]))),
                seed=int(seed) + 1543 + int(cursor),
                max_iter=int(kmeans_iters),
            )
            proto_rows = []
            proto_sse = []
            for proto_id in range(proto_centers.shape[0]):
                rows = np.nonzero(proto_assign == proto_id)[0].astype(np.int64, copy=False)
                proto_rows.append(rows)
                if rows.size == 0:
                    proto_sse.append(0.0)
                else:
                    diff = block[rows] - proto_centers[proto_id].reshape(1, -1)
                    proto_sse.append(float(np.sum(diff * diff)))
            merge_page_prototypes(len(pages), proto_centers.astype(np.float32, copy=False), proto_rows, proto_sse)
            read_bytes += float(int(kmeans_iters) * block.shape[0] * max(1, min(int(router_prototypes), int(block.shape[0]))) * block.shape[1] * int(key_bytes))
            write_bytes += float(proto_centers.size * int(key_bytes) + block.shape[0] * 8)
        pages.append(PagePQ(start=cursor, size=end - cursor, codebooks=codebooks_t, codes=codes_t, proto_rows=proto_rows))
        read_bytes += float(block.shape[0] * block.shape[1] * int(key_bytes))
        write_bytes += float(codebooks.size * int(key_bytes) + codes.size)
        cursor = end
        page_id += 1
    router_group_means = None
    router_group_tokens = None
    router_group_member_refs = None
    if bool(router_enabled) and groups:
        router_group_means = torch.as_tensor(
            np.stack([group["mean"] for group in groups], axis=0).astype(np.float32, copy=False),
            dtype=torch.float32,
            device=device,
        )
        router_group_tokens = []
        router_group_member_refs = []
        for group in groups:
            pieces = []
            for page_idx, proto_id in group["members"]:
                page = pages[int(page_idx)]
                rows = page.proto_rows[int(proto_id)] if page.proto_rows is not None else np.empty((0,), dtype=np.int64)
                if rows.size:
                    pieces.append((int(page.start) + rows).astype(np.int64, copy=False))
            tokens = np.unique(np.concatenate(pieces)) if pieces else np.empty((0,), dtype=np.int64)
            router_group_tokens.append(torch.as_tensor(tokens, dtype=torch.long, device=device))
            router_group_member_refs.append(int(len(group["members"])))
        write_bytes += float(router_group_means.numel() * int(key_bytes) + sum(t.numel() * 8 for t in router_group_tokens))
    return GPUIndex(
        pages=pages,
        pending_start=int(sealed_end),
        indexed_end=int(indexed_end),
        build_seconds=time.perf_counter() - t0,
        build_read_mb=read_bytes / MB,
        build_write_mb=write_bytes / MB,
        router_group_means=router_group_means,
        router_group_tokens=router_group_tokens,
        router_group_member_refs=router_group_member_refs,
    )


def pq_page_scores(query: torch.Tensor, page: PagePQ) -> tuple[torch.Tensor, torch.Tensor]:
    subvecs = int(page.codebooks.shape[0])
    subdim = int(page.codebooks.shape[-1])
    q_parts = query.reshape(subvecs, subdim)
    table = torch.einsum("ms,mcs->mc", q_parts, page.codebooks)
    scores = torch.zeros((page.size,), dtype=torch.float32, device=query.device)
    rows = torch.arange(page.size, device=query.device)
    for sub in range(subvecs):
        scores += table[sub].gather(0, page.codes[:, sub])
    tokens = torch.arange(page.start, page.start + page.size, dtype=torch.long, device=query.device)
    return tokens, scores


def pq_page_scores_rows(query: torch.Tensor, page: PagePQ, rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if rows.numel() == 0:
        return (
            torch.empty((0,), dtype=torch.long, device=query.device),
            torch.empty((0,), dtype=torch.float32, device=query.device),
        )
    subvecs = int(page.codebooks.shape[0])
    subdim = int(page.codebooks.shape[-1])
    q_parts = query.reshape(subvecs, subdim)
    table = torch.einsum("ms,mcs->mc", q_parts, page.codebooks)
    row_codes = page.codes.index_select(0, rows)
    scores = torch.zeros((rows.numel(),), dtype=torch.float32, device=query.device)
    for sub in range(subvecs):
        scores += table[sub].gather(0, row_codes[:, sub])
    return rows + int(page.start), scores


def selector_bytes_fullscan(index: GPUIndex, *, key_bytes: int, subbits: int) -> float:
    return float(
        sum(
            page.codebooks.numel() * int(key_bytes)
            + page.codes.numel() * (1 if int(subbits) <= 8 else 2)
            for page in index.pages
        )
    )


def selector_bytes_routed(
    index: GPUIndex,
    selected_group_ids: torch.Tensor,
    page_row_counts: dict[int, int],
    *,
    key_bytes: int,
    subbits: int,
) -> float:
    if index.router_group_means is None or index.router_group_member_refs is None:
        return 0.0
    bytes_ = float(index.router_group_means.numel() * int(key_bytes))
    member_refs = 0
    for gid in selected_group_ids.detach().cpu().tolist():
        member_refs += int(index.router_group_member_refs[int(gid)])
    bytes_ += float(member_refs * 8)
    code_bytes = 1 if int(subbits) <= 8 else 2
    for page_id, rows in page_row_counts.items():
        page = index.pages[int(page_id)]
        bytes_ += float(page.codebooks.numel() * int(key_bytes) + int(rows) * int(page.codes.shape[1]) * code_bytes)
    return bytes_


def rank_paged_pq(
    query: torch.Tensor,
    index: GPUIndex,
    *,
    mode: str,
    nprobes: list[int],
    budget: int,
    key_bytes: int,
    subbits: int,
) -> tuple[torch.Tensor, torch.Tensor, float, float, int]:
    _sync_if_cuda(query.device)
    t0 = time.perf_counter()
    if mode == "routed":
        if index.router_group_means is None or not index.router_group_tokens:
            empty = torch.empty((0,), dtype=torch.long, device=query.device)
            return empty, torch.empty((0,), dtype=torch.float32, device=query.device), 0.0, 0.0, 0
        group_scores = index.router_group_means @ query
        group_order = torch.argsort(group_scores, descending=True, stable=True)
        best: tuple[torch.Tensor, torch.Tensor, float, int] | None = None
        for nprobe in sorted(set(max(1, int(x)) for x in nprobes)):
            selected_groups = group_order[: min(int(nprobe), int(group_order.numel()))]
            candidate_tokens = torch.unique(torch.cat([index.router_group_tokens[int(gid)] for gid in selected_groups.detach().cpu().tolist()]))
            token_chunks = []
            score_chunks = []
            page_row_counts: dict[int, int] = {}
            for page_id, page in enumerate(index.pages):
                mask = (candidate_tokens >= int(page.start)) & (candidate_tokens < int(page.start + page.size))
                if not bool(torch.any(mask)):
                    continue
                rows = (candidate_tokens[mask] - int(page.start)).to(torch.long)
                page_row_counts[int(page_id)] = int(rows.numel())
                tokens, scores = pq_page_scores_rows(query, page, rows)
                token_chunks.append(tokens)
                score_chunks.append(scores)
            if token_chunks:
                tokens_all = torch.cat(token_chunks)
                scores_all = torch.cat(score_chunks)
                order = torch.argsort(scores_all, descending=True, stable=True)
                ranked_tokens = tokens_all[order]
                ranked_scores = scores_all[order]
            else:
                ranked_tokens = torch.empty((0,), dtype=torch.long, device=query.device)
                ranked_scores = torch.empty((0,), dtype=torch.float32, device=query.device)
            selector_bytes = selector_bytes_routed(
                index,
                selected_groups,
                page_row_counts,
                key_bytes=int(key_bytes),
                subbits=int(subbits),
            )
            best = (ranked_tokens, ranked_scores, selector_bytes, int(nprobe))
            if ranked_tokens.numel() >= int(budget):
                break
        _sync_if_cuda(query.device)
        assert best is not None
        return best[0], best[1], time.perf_counter() - t0, best[2] / MB, best[3]

    token_chunks = []
    score_chunks = []
    for page in index.pages:
        tokens, scores = pq_page_scores(query, page)
        token_chunks.append(tokens)
        score_chunks.append(scores)
    if not token_chunks:
        return (
            torch.empty((0,), dtype=torch.long, device=query.device),
            torch.empty((0,), dtype=torch.float32, device=query.device),
            0.0,
        )
    tokens_all = torch.cat(token_chunks)
    scores_all = torch.cat(score_chunks)
    order = torch.argsort(scores_all, descending=True, stable=True)
    _sync_if_cuda(query.device)
    return (
        tokens_all[order],
        scores_all[order],
        time.perf_counter() - t0,
        selector_bytes_fullscan(index, key_bytes=int(key_bytes), subbits=int(subbits)) / MB,
        0,
    )


def stratified_tail_samples(
    *,
    ranked_cpu: np.ndarray,
    selected_cpu: np.ndarray,
    scores_cpu: np.ndarray,
    context_len: int,
    samples: int,
    bands: int,
    seed: int,
    qidx: int,
    head: int,
    sampling: str,
) -> tuple[list[tuple[np.ndarray, int]], int]:
    selected_set = set(int(tok) for tok in selected_cpu.tolist())
    all_tail = np.asarray([tok for tok in range(context_len) if tok not in selected_set], dtype=np.int64)
    if samples <= 0 or all_tail.size == 0:
        return [], int(all_tail.size)
    tail_set = set(int(tok) for tok in all_tail.tolist())
    ordered = []
    seen = set()
    for tok in ranked_cpu.tolist():
        tok = int(tok)
        if tok in tail_set and tok not in seen:
            ordered.append(tok)
            seen.add(tok)
    if len(ordered) < int(all_tail.size):
        # Deployable tail sampling cannot use dense/oracle scores to rank tokens
        # that the selector never scored. Keep the remaining tail in token order.
        rest = np.asarray([tok for tok in all_tail.tolist() if int(tok) not in seen], dtype=np.int64)
        if rest.size:
            ordered.extend(int(tok) for tok in rest.tolist())
    ordered_arr = np.asarray(ordered, dtype=np.int64)
    strata = [arr.astype(np.int64, copy=False) for arr in np.array_split(ordered_arr, min(max(1, int(bands)), max(1, int(ordered_arr.size))))]
    total = min(max(0, int(samples)), int(all_tail.size))
    weights = np.asarray([2.0 ** (-idx) if s.size else 0.0 for idx, s in enumerate(strata)], dtype=np.float64)
    if float(weights.sum()) <= 0:
        weights = np.asarray([float(s.size) for s in strata], dtype=np.float64)
    raw = total * weights / max(float(weights.sum()), 1e-20)
    alloc = [min(int(s.size), int(np.floor(raw[idx]))) for idx, s in enumerate(strata)]
    for idx, s in enumerate(strata):
        if sum(alloc) >= total:
            break
        if s.size and alloc[idx] == 0:
            alloc[idx] = 1
    remainders = raw - np.floor(raw)
    for idx in np.argsort(-remainders).tolist():
        if sum(alloc) >= total:
            break
        if alloc[idx] < int(strata[idx].size):
            alloc[idx] += 1
    rng = np.random.default_rng(int(seed) + 1000003 * int(qidx) + 9176 * int(head))
    sampled: list[tuple[np.ndarray, int]] = []
    for stratum, count in zip(strata, alloc, strict=False):
        if count <= 0 or stratum.size == 0:
            continue
        count = min(int(count), int(stratum.size))
        if str(sampling) == "linspace" and count < int(stratum.size):
            positions = np.unique(np.linspace(0, int(stratum.size) - 1, num=count, dtype=np.int64))
            sample = stratum[positions]
        elif str(sampling) == "systematic" and count < int(stratum.size):
            step = float(stratum.size) / float(count)
            offset = float(rng.uniform(0.0, step))
            positions = np.floor(offset + step * np.arange(count, dtype=np.float64)).astype(np.int64)
            positions = np.clip(positions, 0, int(stratum.size) - 1)
            sample = stratum[np.unique(positions)]
        else:
            sample = rng.choice(stratum, size=count, replace=False)
        sampled.append((sample.astype(np.int64, copy=False), int(stratum.size)))
    return sampled, int(all_tail.size)


def attention_output(scores: torch.Tensor, values: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    if tokens.numel() == 0:
        return torch.zeros((values.shape[-1],), dtype=torch.float32, device=values.device)
    logits = scores.index_select(0, tokens)
    weights = torch.softmax(logits, dim=0)
    return weights @ values.index_select(0, tokens)


def selected_plus_tail_output(
    keys: torch.Tensor,
    values: torch.Tensor,
    query: torch.Tensor,
    selected: torch.Tensor,
    ranked_cpu: np.ndarray,
    scores_cpu: np.ndarray,
    *,
    context_len: int,
    samples: int,
    bands: int,
    seed: int,
    qidx: int,
    head: int,
    sampling: str,
) -> tuple[torch.Tensor, int, int, float]:
    selected_cpu = selected.detach().cpu().numpy().astype(np.int64, copy=False)
    sampled_tail, tail_population = stratified_tail_samples(
        ranked_cpu=ranked_cpu,
        selected_cpu=selected_cpu,
        scores_cpu=scores_cpu,
        context_len=context_len,
        samples=samples,
        bands=bands,
        seed=seed,
        qidx=qidx,
        head=head,
        sampling=sampling,
    )
    _sync_if_cuda(values.device)
    t0 = time.perf_counter()
    token_parts = []
    scale_parts = []
    if selected.numel():
        token_parts.append(selected)
        scale_parts.append(torch.ones((selected.numel(),), dtype=torch.float32, device=values.device))
    for sample, stratum_size in sampled_tail:
        if not sample.size:
            continue
        sample_t = torch.as_tensor(sample, dtype=torch.long, device=values.device)
        scale = float(stratum_size) / float(max(1, int(sample_t.numel())))
        token_parts.append(sample_t)
        scale_parts.append(torch.full((sample_t.numel(),), scale, dtype=torch.float32, device=values.device))
    if not token_parts:
        out = torch.zeros((values.shape[-1],), dtype=torch.float32, device=values.device)
    else:
        tokens = torch.cat(token_parts)
        scales = torch.cat(scale_parts)
        token_keys = keys.index_select(0, tokens)
        logits = (token_keys @ query) / np.sqrt(float(query.numel()))
        weights = torch.exp(logits - torch.max(logits)) * scales
        out = weights @ values.index_select(0, tokens)
        out = out / torch.clamp(weights.sum(), min=1e-20)
    _sync_if_cuda(values.device)
    tail_count = int(sum(int(sample.size) for sample, _stratum_size in sampled_tail))
    return out, tail_count, int(tail_population), time.perf_counter() - t0


def gpu_peak_mb() -> float:
    return float(torch.cuda.max_memory_allocated()) / MB


def run() -> None:
    parser = argparse.ArgumentParser(description="GPU paged-PQ selector prototype.")
    parser.add_argument("--trace", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--decode_lengths", default="128000")
    parser.add_argument("--heads", default="0")
    parser.add_argument("--budgets", default="4096")
    parser.add_argument("--tail_samples", type=int, default=4096)
    parser.add_argument("--tail_bands", type=int, default=8)
    parser.add_argument("--tail_seed", type=int, default=0)
    parser.add_argument("--tail_sampling", choices=["random", "linspace", "systematic"], default="random")
    parser.add_argument("--static_prefix", type=int, default=128)
    parser.add_argument("--static_suffix", type=int, default=128)
    parser.add_argument("--page_size", type=int, default=5632)
    parser.add_argument("--subvecs", type=int, default=4)
    parser.add_argument("--subbits", type=int, default=6)
    parser.add_argument("--kmeans_iters", type=int, default=3)
    parser.add_argument("--key_bytes", type=int, default=2)
    parser.add_argument("--value_bytes", type=int, default=2)
    parser.add_argument("--selector_mode", choices=["fullscan", "routed"], default="fullscan")
    parser.add_argument("--nprobes", default="1,2,4,8,16,32,64,128,256,512")
    parser.add_argument("--router_prototypes", type=int, default=16)
    parser.add_argument("--router_merge_rel", type=float, default=0.05)
    parser.add_argument("--router_merge_var", type=float, default=0.0)
    parser.add_argument("--router_max_groups", type=int, default=512)
    parser.add_argument("--max_qidx_per_decode", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for GPU selector eval")
    device = torch.device("cuda")
    torch.set_grad_enabled(False)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trace = load_trace(args.trace)
    decode_lengths = parse_csv_ints(args.decode_lengths)
    heads = parse_csv_ints(args.heads)
    budgets = parse_csv_ints(args.budgets)
    nprobes = parse_csv_ints(args.nprobes)
    q_indices = trace.q_indices_for_decodes(decode_lengths)
    if int(args.max_qidx_per_decode) > 0:
        limited = []
        counts: dict[int, int] = {}
        for qidx in q_indices:
            decode = trace.decode_tokens_for_qidx(int(qidx))
            seen = counts.get(int(decode), 0)
            if seen >= int(args.max_qidx_per_decode):
                continue
            limited.append(int(qidx))
            counts[int(decode)] = seen + 1
        q_indices = limited
    rows = []

    for qidx in q_indices:
        position = int(trace.positions[int(qidx)])
        decode_tokens = trace.decode_tokens_for_qidx(int(qidx))
        context_len = position + 1
        indexed_end = max(
            min(max(0, int(args.static_prefix)), int(trace.input_len)),
            min(context_len - max(0, int(args.static_suffix)), trace.keys.shape[1]),
        )
        index_cache: dict[int, GPUIndex] = {}
        for head in heads:
            kv_head = trace.kv_head_for(int(head))
            keys_np = trace.keys[kv_head, :context_len].astype(np.float32, copy=False)
            values_np = trace.values[kv_head, :context_len].astype(np.float32, copy=False)
            query_np = trace.queries[int(head), int(qidx)].astype(np.float32, copy=False)
            scores_np, probs_np = attention_probs(keys_np, query_np)
            dense_out_np = probs_np.astype(np.float32) @ values_np.astype(np.float32, copy=False)
            torch.cuda.reset_peak_memory_stats()
            keys = torch.as_tensor(keys_np, dtype=torch.float32, device=device)
            values = torch.as_tensor(values_np, dtype=torch.float32, device=device)
            query = torch.as_tensor(query_np, dtype=torch.float32, device=device)
            scores = (keys @ query) / np.sqrt(float(trace.head_dim))
            torch.cuda.synchronize()
            _warm_scores = (keys @ query) / np.sqrt(float(trace.head_dim))
            _ = torch.softmax(_warm_scores, dim=0) @ values
            torch.cuda.synchronize()
            dense_t0 = time.perf_counter()
            dense_scores = (keys @ query) / np.sqrt(float(trace.head_dim))
            dense_probs = torch.softmax(dense_scores, dim=0)
            dense_out_gpu = dense_probs @ values
            torch.cuda.synchronize()
            dense_seconds = time.perf_counter() - dense_t0
            dense_err = float(torch.linalg.norm(dense_out_gpu - torch.as_tensor(dense_out_np, device=device)) / max(float(np.linalg.norm(dense_out_np)), 1e-20))

            if int(kv_head) not in index_cache:
                index_cache[int(kv_head)] = build_page_pq_gpu(
                    keys_np,
                    dynamic_start=min(max(0, int(args.static_prefix)), int(trace.input_len)),
                    indexed_end=indexed_end,
                    page_size=int(args.page_size),
                    subvecs=int(args.subvecs),
                    subbits=int(args.subbits),
                    kmeans_iters=int(args.kmeans_iters),
                    seed=2025 + int(kv_head),
                    key_bytes=int(args.key_bytes),
                    router_enabled=str(args.selector_mode) == "routed",
                    router_prototypes=int(args.router_prototypes),
                    router_merge_rel=float(args.router_merge_rel),
                    router_merge_var=float(args.router_merge_var),
                    router_max_groups=int(args.router_max_groups),
                    device=device,
                )
            index = index_cache[int(kv_head)]
            pending = list(range(max(0, int(index.pending_start)), max(0, min(int(index.indexed_end), context_len))))
            base = unique_tokens(
                static_tokens(position, args.static_prefix, args.static_suffix) + pending,
                context_len=context_len,
            )
            base_t = torch.as_tensor(np.asarray(base, dtype=np.int64), dtype=torch.long, device=device)

            for _ in range(max(0, int(args.warmup))):
                ranked_t, _ranked_scores, _, _, _ = rank_paged_pq(
                    query,
                    index,
                    mode=str(args.selector_mode),
                    nprobes=nprobes,
                    budget=max(budgets) if budgets else 0,
                    key_bytes=int(args.key_bytes),
                    subbits=int(args.subbits),
                )
                _ = ranked_t[:1].sum()
            for budget in budgets:
                for _ in range(max(0, int(args.warmup))):
                    ranked_t, _ranked_scores, _selector_seconds, _selector_mb, _nprobe = rank_paged_pq(
                        query,
                        index,
                        mode=str(args.selector_mode),
                        nprobes=nprobes,
                        budget=int(budget),
                        key_bytes=int(args.key_bytes),
                        subbits=int(args.subbits),
                    )
                    ranked_cpu = ranked_t.detach().cpu().numpy().astype(np.int64, copy=False)
                    base_set = set(int(tok) for tok in base)
                    add = [int(tok) for tok in ranked_cpu.tolist() if int(tok) < context_len and int(tok) not in base_set][: int(budget)]
                    selected_cpu = np.asarray(unique_tokens(base + add, context_len=context_len), dtype=np.int64)
                    selected = torch.as_tensor(selected_cpu, dtype=torch.long, device=device)
                    _ = selected_plus_tail_output(
                        keys,
                        values,
                        query,
                        selected,
                        ranked_cpu,
                        scores_np,
                        context_len=context_len,
                        samples=int(args.tail_samples),
                        bands=int(args.tail_bands),
                        seed=int(args.tail_seed),
                        qidx=int(qidx),
                        head=int(head),
                        sampling=str(args.tail_sampling),
                    )
                for rep in range(max(1, int(args.repeat))):
                    ranked_t, _ranked_scores, selector_seconds, selector_mb, chosen_nprobe = rank_paged_pq(
                        query,
                        index,
                        mode=str(args.selector_mode),
                        nprobes=nprobes,
                        budget=int(budget),
                        key_bytes=int(args.key_bytes),
                        subbits=int(args.subbits),
                    )
                    ranked_cpu = ranked_t.detach().cpu().numpy().astype(np.int64, copy=False)
                    base_set = set(int(tok) for tok in base)
                    add = [int(tok) for tok in ranked_cpu.tolist() if int(tok) < context_len and int(tok) not in base_set][: int(budget)]
                    selected_cpu = np.asarray(unique_tokens(base + add, context_len=context_len), dtype=np.int64)
                    selected = torch.as_tensor(selected_cpu, dtype=torch.long, device=device)
                    sparse_out, tail_count, tail_population, attention_seconds = selected_plus_tail_output(
                        keys,
                        values,
                        query,
                        selected,
                        ranked_cpu,
                        scores_np,
                        context_len=context_len,
                        samples=int(args.tail_samples),
                        bands=int(args.tail_bands),
                        seed=int(args.tail_seed),
                        qidx=int(qidx),
                        head=int(head),
                        sampling=str(args.tail_sampling),
                    )
                    approx_np = sparse_out.detach().cpu().numpy().astype(np.float32, copy=False)
                    metrics = _output_error_metrics(dense_out_np, approx_np)
                    mass = float(probs_np[selected_cpu].sum()) if selected_cpu.size else 0.0
                    exact_kv_mb = float(selected_cpu.size * trace.head_dim * (int(args.key_bytes) + int(args.value_bytes))) / MB
                    tail_mb = float(tail_count * trace.head_dim * (int(args.key_bytes) + int(args.value_bytes))) / MB
                    step_mb = selector_mb + exact_kv_mb + tail_mb
                    rows.append(
                        {
                            "algorithm": f"gpu_{str(args.selector_mode)}_paged_pq_k{int(budget)}_tail_s{int(args.tail_samples)}",
                            "decode_length": int(decode_tokens),
                            "qidx": int(qidx),
                            "head": int(head),
                            "kv_head": int(kv_head),
                            "repeat": int(rep),
                            "context_len": int(context_len),
                            "budget": int(budget),
                            "selected_tokens": int(selected_cpu.size),
                            "candidate_tokens": int(ranked_cpu.size),
                            "tail_samples": int(tail_count),
                            "tail_population": int(tail_population),
                            "attention_mass": mass,
                            "output_relative_L2": metrics["output_relative_l2"],
                            "output_cosine": metrics["output_cosine"],
                            "output_rmsnorm_relative_L2": metrics["output_rmsnorm_relative_l2"],
                            "dense_seconds": dense_seconds,
                            "dense_gpu_relerr_vs_numpy": dense_err,
                            "pq_build_seconds": index.build_seconds,
                            "selector_seconds": selector_seconds,
                            "attention_seconds": attention_seconds,
                            "total_query_seconds": selector_seconds + attention_seconds,
                            "selector_MB_per_query": selector_mb,
                            "exact_KV_MB_per_query": exact_kv_mb,
                            "tail_estimator_MB_per_query": tail_mb,
                            "step_MB_per_query": step_mb,
                            "pq_build_read_MB": index.build_read_mb,
                            "pq_build_write_MB": index.build_write_mb,
                            "gpu_peak_MB": gpu_peak_mb(),
                            "pages": int(len(index.pages)),
                            "page_size": int(args.page_size),
                            "subvecs": int(args.subvecs),
                            "subbits": int(args.subbits),
                            "selector_mode": str(args.selector_mode),
                            "nprobe": int(chosen_nprobe),
                            "router_groups": int(index.router_group_means.shape[0]) if index.router_group_means is not None else 0,
                        }
                    )

    with (out_dir / "samples.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    (out_dir / "samples.json").write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")

    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["algorithm"], row["decode_length"], row["head"], row["budget"]), []).append(row)
    summary = []
    for key, items in grouped.items():
        out = {"algorithm": key[0], "decode_length": key[1], "head": key[2], "budget": key[3], "samples": len(items)}
        for metric in [
            "selected_tokens",
            "tail_samples",
            "attention_mass",
            "output_relative_L2",
            "output_cosine",
            "output_rmsnorm_relative_L2",
            "dense_seconds",
            "pq_build_seconds",
            "selector_seconds",
            "attention_seconds",
            "total_query_seconds",
            "selector_MB_per_query",
            "exact_KV_MB_per_query",
            "tail_estimator_MB_per_query",
            "step_MB_per_query",
            "gpu_peak_MB",
        ]:
            vals = [float(item[metric]) for item in items]
            out[f"{metric}_mean"] = float(np.mean(vals))
            out[f"{metric}_median"] = float(np.median(vals))
            out[f"{metric}_min"] = float(np.min(vals))
            out[f"{metric}_max"] = float(np.max(vals))
        summary.append(out)
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()) if summary else [])
        if summary:
            writer.writeheader()
            writer.writerows(summary)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True), encoding="utf-8")
    print(f"[gpu_paged_pq_eval] wrote {out_dir}")


if __name__ == "__main__":
    run()
