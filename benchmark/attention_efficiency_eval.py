#!/usr/bin/env python3
"""Offline sparse-attention algorithmic-efficiency evaluator.

This script compares token-set selection policies against an exact dense
attention reference on sampled query/key/value tensors. It is intentionally an
algorithmic proxy: it estimates read bytes, dense mass coverage, top-k recall,
and output error without claiming GPU latency speedup.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch


@dataclass(frozen=True)
class QuerySpec:
    qid: int
    position: int
    head: int
    kv_head: int
    query: torch.Tensor
    prefill_tokens: int | None = None
    decode_tokens: int = 0


@dataclass
class Selection:
    token_ids: list[int]
    static_tokens: int = 0
    dynamic_selected_tokens: int = 0
    dynamic_budget: int = 0
    metadata_reads: int = 0
    graph_nodes_visited: int = 0
    graph_edges_touched: int = 0
    clusters_scored: int = 0
    clusters_selected: int = 0
    key_score_reads: int = 0
    rerank_key_reads: int = 0
    edge_index_reads: int = 0
    graph_offset_reads: int = 0
    centroid_score_reads: int = 0
    target_dense_mass: float | None = None
    target_attention_output_cos: float | None = None
    target_reached: bool | None = None


def parse_int_list(text: str) -> list[int]:
    out = []
    for part in re.split(r"[,;:\s]+", str(text)):
        part = part.strip()
        if part:
            out.append(int(part))
    if not out:
        raise ValueError(f"empty integer list: {text!r}")
    return out


def parse_str_list(text: str) -> list[str]:
    out = []
    for part in re.split(r"[,;:\s]+", str(text)):
        part = part.strip()
        if part:
            out.append(part)
    if not out:
        raise ValueError(f"empty string list: {text!r}")
    return out


def parse_float_list(text: str) -> list[float]:
    out = []
    for part in re.split(r"[,;:\s]+", str(text)):
        part = part.strip()
        if part:
            out.append(float(part))
    if not out:
        raise ValueError(f"empty float list: {text!r}")
    return out


def parse_ra_absolute_target_method(method: str) -> tuple[str, float] | None:
    for prefix, metric in (
        ("retrievalattention_target_mass_", "mass"),
        ("retrievalattention_target_cos_", "cos"),
    ):
        if method.startswith(prefix):
            raw = method[len(prefix):].replace("p", ".")
            return metric, float(raw)
    return None


def resolve_budgets_for_spec(args, spec: QuerySpec) -> list[int]:
    total_tokens = int(spec.position) + 1
    if args.budget_policy == "fixed":
        return parse_int_list(args.budgets)
    if args.budget_policy == "linear":
        return [max(1, int(math.ceil(float(args.budget_ratio) * total_tokens)))]
    if args.budget_policy == "log2":
        return [max(1, int(math.ceil(math.log2(max(2, total_tokens)))))]
    if args.budget_policy == "retro_static_extension":
        prefill_tokens = int(spec.prefill_tokens or total_tokens)
        return [len(static_extension_tokens_for_decode(
            spec.position,
            prefill_tokens,
            args.static_prefix,
            args.static_suffix,
        ))]
    raise ValueError(f"unknown budget_policy: {args.budget_policy}")


def make_synthetic_qkv(
    n_tokens: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    num_queries: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[QuerySpec]]:
    """Build synthetic but clustered QKV geometry.

    Keys are drawn from latent semantic clusters plus local drift. Queries are
    anchored to real key positions with noise, which makes retrieval quality
    measurable without needing a full model forward pass.
    """

    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    cluster_count = max(32, min(2048, int(math.sqrt(max(1, n_tokens))) * 4))
    cluster_ids = torch.arange(n_tokens) % cluster_count
    cluster_ids = cluster_ids[torch.randperm(n_tokens, generator=gen)]
    centers = torch.randn(cluster_count, head_dim, generator=gen)
    centers = torch.nn.functional.normalize(centers, dim=-1)

    keys = []
    values = []
    drift = torch.linspace(-1.0, 1.0, n_tokens).unsqueeze(1)
    for kv_h in range(num_kv_heads):
        head_noise = 0.08 * torch.randn(n_tokens, head_dim, generator=gen)
        local = 0.02 * torch.randn(1, head_dim, generator=gen) * drift
        k = centers[cluster_ids] + head_noise + local
        k = torch.nn.functional.normalize(k, dim=-1)
        v = torch.randn(n_tokens, head_dim, generator=gen)
        values.append(v)
        keys.append(k)
    keys_t = torch.stack(keys, dim=0).to(device=device, dtype=torch.float32)
    values_t = torch.stack(values, dim=0).to(device=device, dtype=torch.float32)

    specs: list[QuerySpec] = []
    min_pos = max(1, min(512, n_tokens - 1))
    for qid in range(num_queries):
        frac = (qid + 0.5) / max(1, num_queries)
        position = min(n_tokens - 1, max(min_pos, int(frac * n_tokens)))
        head = qid % num_heads
        kv_head = head * num_kv_heads // num_heads
        if position <= 1:
            anchor = 0
        else:
            # Mix local and far anchors so static/chunk baselines are not
            # always sufficient.
            if qid % 3 == 0:
                anchor = max(0, position - 1 - (qid * 17) % max(1, min(2048, position)))
            else:
                anchor = int(torch.randint(0, position + 1, (1,), generator=gen).item())
        q = keys_t[kv_head, anchor].detach().cpu()
        q = q + 0.10 * torch.randn(head_dim, generator=gen)
        q = torch.nn.functional.normalize(q, dim=-1).to(device)
        specs.append(QuerySpec(qid=qid, position=position, head=head, kv_head=kv_head, query=q))
    return keys_t, values_t, specs


def make_decode_query_specs(
    keys: torch.Tensor,
    prefill_tokens: int,
    decode_tokens: int,
    num_heads: int,
    num_kv_heads: int,
    num_queries: int,
    seed: int,
    device: torch.device,
) -> list[QuerySpec]:
    """Sample decode-step queries at the end of a prefill+decode sequence."""

    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    max_pos = int(keys.shape[1]) - 1
    position = min(max_pos, max(0, int(prefill_tokens) + max(0, int(decode_tokens)) - 1))
    prefill_end = min(int(prefill_tokens), position + 1)
    generated_start = min(prefill_end, position + 1)

    specs: list[QuerySpec] = []
    for qid in range(int(num_queries)):
        head = qid % int(num_heads)
        kv_head = head * int(num_kv_heads) // int(num_heads)
        if int(decode_tokens) > 0 and qid % 3 == 0:
            anchor = int(torch.randint(generated_start, position + 1, (1,), generator=gen).item())
        elif int(decode_tokens) > 0 and qid % 3 == 1:
            aged_span = max(1, int(decode_tokens) - 1)
            anchor = generated_start + (qid * 17) % aged_span
            anchor = min(position, int(anchor))
        elif prefill_end > 0:
            anchor = int(torch.randint(0, prefill_end, (1,), generator=gen).item())
        else:
            anchor = 0
        q = keys[kv_head, anchor].detach().cpu()
        q = q + 0.10 * torch.randn(keys.shape[-1], generator=gen)
        q = torch.nn.functional.normalize(q, dim=-1).to(device)
        specs.append(
            QuerySpec(
                qid=qid,
                position=position,
                head=head,
                kv_head=kv_head,
                query=q,
                prefill_tokens=int(prefill_tokens),
                decode_tokens=int(decode_tokens),
            )
        )
    return specs


def load_npz_qkv(
    path: Path,
    num_queries: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[QuerySpec]]:
    data = np.load(path)
    keys = torch.as_tensor(data["keys"], dtype=torch.float32, device=device)
    values = torch.as_tensor(data["values"], dtype=torch.float32, device=device)
    queries = torch.as_tensor(data["queries"], dtype=torch.float32, device=device)
    metadata = {}
    if "metadata" in data:
        raw_meta = data["metadata"]
        try:
            metadata = json.loads(str(raw_meta.item() if hasattr(raw_meta, "item") else raw_meta))
        except Exception:
            metadata = {}
    if keys.ndim != 3 or values.shape != keys.shape:
        raise ValueError("NPZ expects keys and values with shape [kv_heads, seq, dim].")
    if queries.ndim != 3:
        raise ValueError("NPZ expects queries with shape [heads, q_samples, dim].")
    rng = random.Random(int(seed))
    specs = []
    heads, q_available, _ = queries.shape
    kv_heads = int(keys.shape[0])
    positions = data["positions"].tolist() if "positions" in data else list(range(q_available))
    prefill_tokens = metadata.get("input_len")
    prefill_tokens = int(prefill_tokens) if prefill_tokens is not None else None
    choices = list(range(q_available))
    rng.shuffle(choices)
    for qid, qidx in enumerate(choices[: int(num_queries)]):
        head = qid % heads
        kv_head = head * kv_heads // heads
        pos = int(positions[qidx])
        pos = max(0, min(int(keys.shape[1]) - 1, pos))
        q = torch.nn.functional.normalize(queries[head, qidx], dim=-1)
        decode_tokens = max(0, pos - int(prefill_tokens) + 1) if prefill_tokens is not None else 0
        specs.append(
            QuerySpec(
                qid=qid,
                position=pos,
                head=head,
                kv_head=kv_head,
                query=q,
                prefill_tokens=prefill_tokens,
                decode_tokens=decode_tokens,
            )
        )
    return keys, values, specs


def unique_budgeted(tokens: Iterable[int], budget: int, max_token: int) -> list[int]:
    if int(budget) <= 0:
        return []
    out = []
    seen = set()
    for tok in tokens:
        tok = int(tok)
        if tok < 0 or tok > max_token or tok in seen:
            continue
        out.append(tok)
        seen.add(tok)
        if len(out) >= int(budget):
            break
    return out


def static_tokens_for_position(position: int, prefix: int, suffix: int) -> list[int]:
    max_tok = int(position)
    prefix_tokens = range(0, min(int(prefix), max_tok + 1))
    suffix_tokens = range(max(0, max_tok - int(suffix) + 1), max_tok + 1)
    return unique_budgeted(list(prefix_tokens) + list(suffix_tokens), max_tok + 1, max_tok)


def static_extension_tokens_for_decode(
    position: int,
    prefill_tokens: int,
    prefix: int,
    suffix: int,
) -> list[int]:
    """RetroInfer static-only long-decode extension.

    Keep the normal prefill static prefix/suffix and make every generated token
    directly readable. This intentionally ignores the nominal sparse budget
    because the point of this baseline is to expose the growing read cost.
    """

    max_tok = int(position)
    prefill_end = min(max(0, int(prefill_tokens)), max_tok + 1)
    prefix_tokens = range(0, min(int(prefix), prefill_end))
    prefill_suffix = range(max(0, prefill_end - int(suffix)), prefill_end)
    generated_tokens = range(prefill_end, max_tok + 1)
    return unique_budgeted(
        list(prefix_tokens) + list(prefill_suffix) + list(generated_tokens),
        max_tok + 1,
        max_tok,
    )


def compose_selection(
    static_tokens: list[int],
    dynamic_tokens: Iterable[int],
    budget: int,
    max_token: int,
    budget_mode: str,
) -> tuple[list[int], int, int]:
    static = unique_budgeted(static_tokens, max_token + 1, max_token)
    static_set = set(static)
    if budget_mode == "dynamic":
        dynamic = unique_budgeted((tok for tok in dynamic_tokens if int(tok) not in static_set), budget, max_token)
        return unique_budgeted(static + dynamic, len(static) + len(dynamic), max_token), len(static), len(dynamic)

    selected = unique_budgeted(list(static) + list(dynamic_tokens), budget, max_token)
    selected_set = set(selected)
    static_count = len(selected_set.intersection(static_set))
    return selected, static_count, max(0, len(selected) - static_count)


def compose_static_floor_selection(
    static_tokens: list[int],
    dynamic_tokens: Iterable[int],
    budget: int,
    max_token: int,
) -> tuple[list[int], int, int]:
    """Select all mandatory static tokens, then spend remaining total budget.

    The model cache does not drop static prefix/suffix tokens when the nominal
    retrieval budget is smaller than the static pattern. For proxy accounting,
    that means static reads are a floor, not a prefix of the requested budget.
    """

    static = unique_budgeted(static_tokens, max_token + 1, max_token)
    static_set = set(static)
    dynamic_budget = max(0, int(budget) - len(static))
    dynamic = unique_budgeted(
        (tok for tok in dynamic_tokens if int(tok) not in static_set),
        dynamic_budget,
        max_token,
    )
    return (
        unique_budgeted(static + dynamic, len(static) + len(dynamic), max_token),
        len(static),
        len(dynamic),
    )


def select_dense_oracle(
    scores: torch.Tensor,
    budget: int,
    static_tokens: list[int],
    budget_mode: str,
) -> Selection:
    max_tok = int(scores.numel()) - 1
    static_set = set(static_tokens)
    k = min(int(budget), int(scores.numel()))
    if budget_mode == "dynamic":
        order = torch.argsort(scores, descending=True).detach().cpu().tolist()
    elif k > 0:
        # Total-budget oracle is the clean upper bound: pick the best K tokens
        # directly from dense attention, without receiving the static window for
        # free or being forced to spend budget on it.
        order = torch.topk(scores, k=k, largest=True, sorted=True).indices.detach().cpu().tolist()
        tokens = unique_budgeted([int(x) for x in order], budget, max_tok)
        return Selection(
            token_ids=tokens,
            static_tokens=0,
            dynamic_selected_tokens=len(tokens),
            dynamic_budget=int(budget),
            metadata_reads=int(scores.numel()),
            key_score_reads=int(scores.numel()),
        )
    else:
        order = []
    if not order:
        return Selection([], dynamic_budget=int(budget))
    tokens, static_count, dynamic_count = compose_selection(
        static_tokens=static_tokens,
        dynamic_tokens=[int(x) for x in order if int(x) not in static_set],
        budget=budget,
        max_token=max_tok,
        budget_mode=budget_mode,
    )
    return Selection(
        token_ids=tokens,
        static_tokens=static_count,
        dynamic_selected_tokens=dynamic_count,
        dynamic_budget=int(budget),
        metadata_reads=int(scores.numel()),
        key_score_reads=int(scores.numel()),
    )


def select_static_chunk(
    position: int,
    budget: int,
    prefix: int,
    suffix: int,
    chunk: int,
    budget_mode: str,
) -> Selection:
    max_tok = int(position)
    base = static_tokens_for_position(position, prefix, suffix)
    suffix_start = max(0, max_tok - int(suffix) + 1)

    chunk = max(1, int(chunk))
    used = set(base)
    candidates = []
    span_end = max(0, suffix_start)
    if span_end > int(prefix):
        num_chunks = max(1, math.ceil(int(budget) / float(chunk)))
        starts = np.linspace(int(prefix), max(int(prefix), span_end - chunk), num_chunks)
        for start in starts:
            start_i = int(round(float(start)))
            for tok in range(start_i, min(span_end, start_i + chunk)):
                if tok not in used:
                    candidates.append(tok)
    tokens, static_count, dynamic_count = compose_selection(base, candidates, budget, max_tok, budget_mode)
    return Selection(
        tokens,
        static_tokens=static_count,
        dynamic_selected_tokens=dynamic_count,
        dynamic_budget=int(budget),
    )


def select_static_only(
    position: int,
    budget: int,
    prefix: int,
    suffix: int,
    budget_mode: str,
) -> Selection:
    max_tok = int(position)
    base = static_tokens_for_position(position, prefix, suffix)
    if budget_mode == "dynamic":
        tokens = unique_budgeted(base, len(base), max_tok)
    else:
        tokens = unique_budgeted(base, budget, max_tok)
    return Selection(
        token_ids=tokens,
        static_tokens=len(tokens),
        dynamic_selected_tokens=0,
        dynamic_budget=int(budget),
    )


def select_retroinfer_static_extension(
    position: int,
    budget: int,
    prefill_tokens: int,
    prefix: int,
    suffix: int,
) -> Selection:
    tokens = static_extension_tokens_for_decode(position, prefill_tokens, prefix, suffix)
    return Selection(
        token_ids=tokens,
        static_tokens=len(tokens),
        dynamic_selected_tokens=0,
        dynamic_budget=int(budget),
    )


def build_retro_clusters(keys: torch.Tensor, cluster_size: int) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    n_tokens = int(keys.shape[0])
    if n_tokens <= 0:
        raise ValueError("cannot build RetroInfer clusters for an empty key range")
    ranges = []
    centroids = []
    for start in range(0, n_tokens, int(cluster_size)):
        end = min(n_tokens, start + int(cluster_size))
        ranges.append((start, end))
        centroid = keys[start:end].mean(dim=0)
        centroid = torch.nn.functional.normalize(centroid, dim=-1)
        centroids.append(centroid)
    return torch.stack(centroids, dim=0), ranges


def retro_cluster_limit_for_spec(args, spec: QuerySpec, n_tokens: int, total_tokens: int) -> int:
    """Return the visible key range for RetroInfer-style centroid scoring.

    In long-decode mode, the paper-style/implementation-faithful path builds
    clusters during prefill and lets generated tokens participate through the
    static suffix, rather than rebuilding centroid clusters over future decode
    tokens. ``causal`` is useful as an optimistic online-clustering diagnostic;
    ``full`` is kept only for backward-compatible/offline upper-bound checks.
    """

    current = min(int(spec.position) + 1, int(total_tokens))
    if args.retro_cluster_scope == "prefill":
        limit = int(spec.prefill_tokens) if spec.prefill_tokens is not None else current
        return max(1, min(limit, current))
    if args.retro_cluster_scope == "causal":
        return max(1, current)
    if args.retro_cluster_scope == "full":
        return max(1, int(total_tokens))
    raise ValueError(f"unknown retro_cluster_scope: {args.retro_cluster_scope}")


def select_retroinfer_style(
    query: torch.Tensor,
    position: int,
    budget: int,
    centroids: torch.Tensor,
    ranges: list[tuple[int, int]],
    prefix: int,
    suffix: int,
    budget_mode: str,
    include_static: bool = True,
) -> Selection:
    max_tok = int(position)
    base = static_tokens_for_position(position, prefix, suffix) if include_static else []

    cluster_scores = torch.matmul(centroids, query)
    order = torch.argsort(cluster_scores, descending=True).detach().cpu().tolist()
    candidates = []
    selected_clusters = 0
    base_set = set(base)
    for cid in order:
        start, end = ranges[int(cid)]
        cluster_tokens = [tok for tok in range(start, min(end, max_tok + 1)) if tok not in base_set]
        if not cluster_tokens:
            continue
        selected_clusters += 1
        candidates.extend(cluster_tokens)
        if len(candidates) >= int(budget):
            break
    tokens, static_count, dynamic_count = compose_selection(base, candidates, budget, max_tok, budget_mode)
    return Selection(
        token_ids=tokens,
        static_tokens=static_count,
        dynamic_selected_tokens=dynamic_count,
        dynamic_budget=int(budget),
        clusters_scored=int(centroids.shape[0]),
        clusters_selected=int(selected_clusters),
        metadata_reads=int(centroids.shape[0]),
        centroid_score_reads=int(centroids.shape[0]),
    )


def select_retroinfer_target_for_match(
    args,
    spec: QuerySpec,
    keys: torch.Tensor,
    query: torch.Tensor,
    position: int,
    budget: int,
    kv_h: int,
    n_tokens: int,
    retro_cache: dict,
) -> Selection:
    target = str(args.retro_target_method)
    if target == "retroinfer_static_extension":
        return select_retroinfer_static_extension(
            position=position,
            budget=budget,
            prefill_tokens=spec.prefill_tokens or n_tokens,
            prefix=args.static_prefix,
            suffix=args.static_suffix,
        )
    if target == "static_only":
        return select_static_only(
            position=position,
            budget=budget,
            prefix=args.static_prefix,
            suffix=args.static_suffix,
            budget_mode=args.budget_mode,
        )
    if target in ("retroinfer_style", "retroinfer_mixed", "retroinfer_dynamic_only"):
        cluster_limit = retro_cluster_limit_for_spec(args, spec, n_tokens, int(keys.shape[1]))
        key = (kv_h, int(args.retro_cluster_size), int(cluster_limit), str(args.retro_cluster_scope))
        if key not in retro_cache:
            retro_cache[key] = build_retro_clusters(keys[kv_h, :cluster_limit], args.retro_cluster_size)
        centroids, ranges = retro_cache[key]
        return select_retroinfer_style(
            query=query,
            position=position,
            budget=budget,
            centroids=centroids,
            ranges=ranges,
            prefix=args.static_prefix,
            suffix=args.static_suffix,
            budget_mode=args.budget_mode,
            include_static=target != "retroinfer_dynamic_only",
        )
    raise ValueError(f"unsupported retro_target_method: {target}")


class LazyKnnGraph:
    def __init__(self, keys: torch.Tensor, degree: int):
        self.keys = keys
        self.degree = max(1, int(degree))
        self.cache: dict[tuple[int, int], list[int]] = {}

    def neighbors(self, node: int, max_token: int) -> list[int]:
        node = int(node)
        max_token = int(max_token)
        cached = self.cache.get((node, max_token))
        if cached is not None:
            return cached
        scores = torch.matmul(self.keys[: max_token + 1], self.keys[node])
        scores[node] = torch.finfo(scores.dtype).min
        k = min(self.degree, int(scores.numel()) - 1)
        if k <= 0:
            out: list[int] = []
        else:
            out = torch.topk(scores, k=k, largest=True, sorted=False).indices.detach().cpu().tolist()
            out = [int(x) for x in out]
        self.cache[(node, max_token)] = out
        return out


class OnlineKnnGraph:
    """Prefill-built KNN graph plus a lazy generated-token overlay.

    This models the long-decode setup more closely than rebuilding exact KNN
    over all tokens at each evaluated position. The base graph is restricted to
    prefill tokens. Generated tokens get overlay edges only after they age out
    of the static suffix window.
    """

    def __init__(
        self,
        keys: torch.Tensor,
        degree: int,
        prefill_tokens: int,
        prefix: int,
        suffix: int,
        bidirectional: bool = True,
    ):
        self.keys = keys
        self.degree = max(1, int(degree))
        self.prefill_tokens = max(0, int(prefill_tokens))
        self.prefix = max(0, int(prefix))
        self.suffix = max(0, int(suffix))
        self.bidirectional = bool(bidirectional)
        self.base_cache: dict[int, list[int]] = {}
        self.overlay: dict[int, list[int]] = {}
        self.overlay_built_until = self.prefill_tokens - 1

    def _add_overlay_edge(self, src: int, dst: int, max_token: int) -> None:
        src = int(src)
        dst = int(dst)
        if src == dst or src < self.prefix or dst < self.prefix:
            return
        if src < 0 or dst < 0 or src > int(max_token) or dst > int(max_token):
            return
        cur = self.overlay.setdefault(src, [])
        if dst not in cur and len(cur) < self.degree:
            cur.append(dst)

    def _base_neighbors(self, node: int) -> list[int]:
        node = int(node)
        if node < 0 or node >= self.prefill_tokens:
            return []
        cached = self.base_cache.get(node)
        if cached is not None:
            return cached
        max_prefill = min(int(self.prefill_tokens), int(self.keys.shape[0]))
        if max_prefill <= 1:
            out: list[int] = []
        else:
            scores = torch.matmul(self.keys[:max_prefill], self.keys[node])
            scores[node] = torch.finfo(scores.dtype).min
            k = min(self.degree, int(scores.numel()) - 1)
            out = (
                [int(x) for x in torch.topk(scores, k=k, largest=True, sorted=False).indices.detach().cpu().tolist()]
                if k > 0 else []
            )
        self.base_cache[node] = out
        return out

    def _generated_provenance_neighbors(self, token_pos: int) -> list[int]:
        token_pos = int(token_pos)
        # At the step that creates token_pos, the current static suffix is not
        # part of dynamic graph retrieval. Use only the aged dynamic range.
        cand_start = min(self.prefix, token_pos)
        cand_end = max(cand_start, token_pos - self.suffix)
        cand_end = min(cand_end, token_pos, int(self.keys.shape[0]))
        if cand_end <= cand_start:
            return []
        cand_keys = self.keys[cand_start:cand_end]
        scores = torch.matmul(cand_keys, self.keys[token_pos])
        k = min(self.degree, int(scores.numel()))
        if k <= 0:
            return []
        idx = torch.topk(scores, k=k, largest=True, sorted=False).indices.detach().cpu().tolist()
        return [cand_start + int(x) for x in idx]

    def _ensure_overlay(self, max_token: int) -> None:
        max_token = min(int(max_token), int(self.keys.shape[0]) - 1)
        eligible_until = max(self.prefill_tokens - 1, max_token - self.suffix)
        if eligible_until <= self.overlay_built_until:
            return
        start = max(self.prefill_tokens, self.overlay_built_until + 1)
        for token_pos in range(start, eligible_until + 1):
            neighbors = self._generated_provenance_neighbors(token_pos)
            for nb in neighbors:
                self._add_overlay_edge(token_pos, nb, max_token)
                if self.bidirectional:
                    self._add_overlay_edge(nb, token_pos, max_token)
        self.overlay_built_until = eligible_until

    def neighbors(self, node: int, max_token: int) -> list[int]:
        self._ensure_overlay(max_token)
        merged = []
        if int(node) < self.prefill_tokens:
            merged.extend(self._base_neighbors(node))
        merged.extend(self.overlay.get(int(node), ()))
        return unique_budgeted(merged, len(merged), int(max_token))


class PrecomputedOnlineKnnGraph:
    """OnlineKnnGraph-compatible adjacency precomputed with batched matmuls.

    The lazy graph is faithful but expensive for offline sweeps because every
    visited node triggers a separate top-k neighbor search and a device sync.
    This backend preserves the same prefill graph and chronological generated
    overlay semantics, but pays the exact KNN cost once per KV head in chunks.
    """

    def __init__(
        self,
        keys: torch.Tensor,
        degree: int,
        prefill_tokens: int,
        prefix: int,
        suffix: int,
        chunk_size: int = 512,
        bidirectional: bool = True,
    ):
        self.keys = keys
        self.degree = max(1, int(degree))
        self.prefill_tokens = max(0, int(prefill_tokens))
        self.prefix = max(0, int(prefix))
        self.suffix = max(0, int(suffix))
        self.chunk_size = max(1, int(chunk_size))
        self.bidirectional = bool(bidirectional)
        self.total_tokens = int(keys.shape[0])
        self.base_neighbors: list[list[int]] = [[] for _ in range(self.prefill_tokens)]
        self.overlay_edges: list[list[tuple[int, int]]] = [[] for _ in range(self.total_tokens)]
        self._precompute_base()
        self._precompute_overlay()

    def _precompute_base(self) -> None:
        n = min(self.prefill_tokens, self.total_tokens)
        if n <= 1:
            return
        key_base = self.keys[:n]
        k = min(self.degree, n - 1)
        for start in range(0, n, self.chunk_size):
            end = min(n, start + self.chunk_size)
            scores = torch.matmul(key_base[start:end], key_base.T)
            rows = torch.arange(start, end, device=scores.device)
            scores[torch.arange(end - start, device=scores.device), rows] = torch.finfo(scores.dtype).min
            top = torch.topk(scores, k=k, largest=True, sorted=False).indices.detach().cpu().tolist()
            for row_offset, nbrs in enumerate(top):
                self.base_neighbors[start + row_offset] = [int(x) for x in nbrs]

    def _append_overlay_edge(self, src: int, dst: int, birth: int) -> None:
        src = int(src)
        dst = int(dst)
        birth = int(birth)
        if src == dst or src < self.prefix or dst < self.prefix:
            return
        if src < 0 or dst < 0 or src >= self.total_tokens or dst >= self.total_tokens:
            return
        cur = self.overlay_edges[src]
        if len(cur) >= self.degree:
            return
        if any(existing_dst == dst for existing_dst, _existing_birth in cur):
            return
        cur.append((dst, birth))

    def _precompute_overlay(self) -> None:
        start_token = max(self.prefill_tokens, 0)
        if start_token >= self.total_tokens:
            return
        candidates = torch.arange(self.total_tokens, device=self.keys.device)
        neg_inf = torch.finfo(self.keys.dtype).min
        for start in range(start_token, self.total_tokens, self.chunk_size):
            end = min(self.total_tokens, start + self.chunk_size)
            rows = torch.arange(start, end, device=self.keys.device)
            cand_end = torch.clamp(rows - self.suffix, min=self.prefix)
            valid = (candidates.unsqueeze(0) >= self.prefix) & (candidates.unsqueeze(0) < cand_end.unsqueeze(1))
            valid_counts = valid.sum(dim=1)
            max_valid = int(valid_counts.max().item()) if int(valid_counts.numel()) else 0
            if max_valid <= 0:
                continue
            k = min(self.degree, max_valid)
            scores = torch.matmul(self.keys[start:end], self.keys.T)
            scores = scores.masked_fill(~valid, neg_inf)
            vals, idx = torch.topk(scores, k=k, largest=True, sorted=False)
            vals_cpu = vals.detach().cpu()
            idx_cpu = idx.detach().cpu()
            for row_offset in range(end - start):
                token_pos = start + row_offset
                for val, nb in zip(vals_cpu[row_offset].tolist(), idx_cpu[row_offset].tolist()):
                    if not math.isfinite(float(val)) or float(val) == float(neg_inf):
                        continue
                    nb = int(nb)
                    self._append_overlay_edge(token_pos, nb, token_pos)
                    if self.bidirectional:
                        self._append_overlay_edge(nb, token_pos, token_pos)

    def neighbors(self, node: int, max_token: int) -> list[int]:
        node = int(node)
        max_token = min(int(max_token), self.total_tokens - 1)
        eligible_until = max(self.prefill_tokens - 1, max_token - self.suffix)
        merged: list[int] = []
        if 0 <= node < self.prefill_tokens:
            merged.extend(self.base_neighbors[node])
        if 0 <= node < self.total_tokens:
            merged.extend(dst for dst, birth in self.overlay_edges[node] if birth <= eligible_until)
        return unique_budgeted(merged, len(merged), max_token)


def _ra_dynamic_seeds(
    keys: torch.Tensor,
    max_tok: int,
    prefix: int,
    suffix: int,
    seed_count: int,
    static_set: set[int],
    dynamic_tail_only: bool,
) -> list[int]:
    seed_count = max(1, int(seed_count))
    max_tok = int(max_tok)
    if dynamic_tail_only:
        dyn_start = min(max(0, int(prefix)), max_tok + 1)
        dyn_end = max(dyn_start, max_tok - max(0, int(suffix)) + 1)
        tail_start = max(dyn_start, dyn_end - seed_count)
        tail = list(range(tail_start, dyn_end))
        hub_span = [tok for tok in range(dyn_start, dyn_end) if tok not in static_set]
    else:
        tail_start = max(0, max_tok - max(seed_count, int(suffix)) + 1)
        tail = list(range(tail_start, max_tok + 1))
        hub_span = [tok for tok in range(0, max_tok + 1) if tok not in static_set]
    if hub_span:
        hub_idx = torch.as_tensor(hub_span, dtype=torch.long, device=keys.device)
        norms = torch.linalg.vector_norm(keys.index_select(0, hub_idx), dim=-1)
        top = torch.topk(norms, k=min(seed_count, int(norms.numel())), largest=True).indices.detach().cpu().tolist()
        hubs = [hub_span[int(i)] for i in top]
    else:
        hubs = []
    anchors = []
    midpoint = max(0, max_tok // 2)
    if midpoint not in static_set:
        anchors.append(midpoint)
    if 0 not in static_set:
        anchors.append(0)
    return unique_budgeted(
        (tok for tok in tail + hubs + anchors if int(tok) not in static_set),
        seed_count * 3,
        max_tok,
    )


def select_retrievalattention_style(
    query: torch.Tensor,
    scores: torch.Tensor,
    position: int,
    budget: int,
    graph: LazyKnnGraph,
    prefix: int,
    suffix: int,
    seed_count: int,
    visit_budget: int,
    budget_mode: str,
    include_static_in_total: bool = False,
) -> Selection:
    max_tok = int(position)
    # In dynamic-budget mode, preserve the paper-style/static-pattern setup:
    # fixed prefix/suffix tokens are included and graph traversal fills extra
    # budget. In total-budget mode, use RetrievalAttention as a pure dynamic
    # selector so the full budget is spent on graph-retrieved tokens.
    base = (
        static_tokens_for_position(position, prefix, suffix)
        if budget_mode == "dynamic" or include_static_in_total
        else []
    )
    base_set = set(base)

    seed_count = max(1, int(seed_count))
    tail_start = max(0, max_tok - max(seed_count, suffix) + 1)
    tail = list(range(tail_start, max_tok + 1))
    # Use norm hubs as query-independent entry points, similar in spirit to
    # graph-only hub seeding. No dense top-k seed is used.
    norms = torch.linalg.vector_norm(graph.keys[: max_tok + 1], dim=-1)
    hubs = torch.topk(norms, k=min(seed_count, int(norms.numel())), largest=True).indices.detach().cpu().tolist()
    seeds = unique_budgeted(tail + [int(x) for x in hubs] + [0, max_tok // 2], seed_count * 3, max_tok)

    import heapq

    frontier = []
    scored_nodes = set()
    for seed in seeds:
        scored_nodes.add(int(seed))
        heapq.heappush(frontier, (-float(scores[seed]), int(seed)))
    visited = set()
    candidates = []
    edges_touched = 0
    while frontier and len(visited) < int(visit_budget):
        neg_score, node = heapq.heappop(frontier)
        if node in visited:
            continue
        visited.add(node)
        if node not in base_set:
            candidates.append((-float(neg_score), int(node)))
        for nbr in graph.neighbors(node, max_tok):
            edges_touched += 1
            if nbr not in visited:
                scored_nodes.add(int(nbr))
                heapq.heappush(frontier, (-float(scores[nbr]), int(nbr)))
    candidates.sort(reverse=True, key=lambda item: item[0])
    dynamic = [node for _score, node in candidates]
    tokens, static_count, dynamic_count = compose_selection(base, dynamic, budget, max_tok, budget_mode)
    return Selection(
        token_ids=tokens,
        static_tokens=static_count,
        dynamic_selected_tokens=dynamic_count,
        dynamic_budget=int(budget),
        metadata_reads=int(len(visited) + edges_touched),
        graph_nodes_visited=int(len(visited)),
        graph_edges_touched=int(edges_touched),
        key_score_reads=int(len(scored_nodes)),
        rerank_key_reads=int(len(candidates)),
        edge_index_reads=int(edges_touched),
        graph_offset_reads=int(len(visited)),
    )


def select_retrievalattention_online_graph(
    query: torch.Tensor,
    scores: torch.Tensor,
    position: int,
    budget: int,
    graph: OnlineKnnGraph,
    prefix: int,
    suffix: int,
    seed_count: int,
    visit_budget: int,
    budget_mode: str,
) -> Selection:
    del query  # Query scoring is represented by the precomputed scores vector.
    max_tok = int(position)
    base = static_tokens_for_position(position, prefix, suffix)
    base_set = set(base)
    if budget_mode == "total" and int(budget) <= len(base):
        tokens, static_count, dynamic_count = compose_static_floor_selection(base, [], budget, max_tok)
        return Selection(
            token_ids=tokens,
            static_tokens=static_count,
            dynamic_selected_tokens=dynamic_count,
            dynamic_budget=max(0, int(budget) - static_count),
        )

    seeds = _ra_dynamic_seeds(
        keys=graph.keys,
        max_tok=max_tok,
        prefix=prefix,
        suffix=suffix,
        seed_count=seed_count,
        static_set=base_set,
        dynamic_tail_only=True,
    )

    import heapq

    frontier = []
    scored_nodes = set()
    for seed in seeds:
        scored_nodes.add(int(seed))
        heapq.heappush(frontier, (-float(scores[seed]), int(seed)))
    visited = set()
    candidates = []
    edges_touched = 0
    max_visits = max(1, int(visit_budget))
    while frontier and len(visited) < max_visits:
        neg_score, node = heapq.heappop(frontier)
        if node in visited or node in base_set:
            continue
        visited.add(node)
        candidates.append((-float(neg_score), int(node)))
        for nbr in graph.neighbors(node, max_tok):
            edges_touched += 1
            if nbr not in visited and nbr not in base_set:
                scored_nodes.add(int(nbr))
                heapq.heappush(frontier, (-float(scores[nbr]), int(nbr)))

    candidates.sort(reverse=True, key=lambda item: item[0])
    dynamic = [node for _score, node in candidates]
    if budget_mode == "dynamic":
        tokens, static_count, dynamic_count = compose_selection(base, dynamic, budget, max_tok, budget_mode)
    else:
        tokens, static_count, dynamic_count = compose_static_floor_selection(base, dynamic, budget, max_tok)
    return Selection(
        token_ids=tokens,
        static_tokens=static_count,
        dynamic_selected_tokens=dynamic_count,
        dynamic_budget=(int(budget) if budget_mode == "dynamic" else max(0, int(budget) - static_count)),
        metadata_reads=int(len(visited) + edges_touched),
        graph_nodes_visited=int(len(visited)),
        graph_edges_touched=int(edges_touched),
        key_score_reads=int(len(scored_nodes)),
        rerank_key_reads=int(len(candidates)),
        edge_index_reads=int(edges_touched),
        graph_offset_reads=int(len(visited)),
    )


def _quality_for_tokens(
    scores: torch.Tensor,
    values: torch.Tensor,
    token_ids: list[int],
    probs: torch.Tensor,
    dense_out: torch.Tensor,
) -> tuple[float, float, float]:
    selected = unique_budgeted(token_ids, len(token_ids), int(scores.numel()) - 1)
    if not selected:
        return 0.0, 0.0, float("inf")
    idx = torch.as_tensor(selected, dtype=torch.long, device=scores.device)
    mass = float(probs.index_select(0, idx).sum().item())
    sparse_probs = torch.softmax(scores.index_select(0, idx).float(), dim=-1)
    sparse_out = torch.matmul(sparse_probs, values.index_select(0, idx).float())
    l2 = float(torch.linalg.vector_norm(sparse_out - dense_out).item())
    denom = max(1.0e-8, float(torch.linalg.vector_norm(dense_out).item()))
    rel_l2 = float(l2 / denom)
    cos = float(torch.nn.functional.cosine_similarity(sparse_out, dense_out, dim=0).item())
    return mass, cos, rel_l2


def select_retrievalattention_until_target(
    query: torch.Tensor,
    scores: torch.Tensor,
    values: torch.Tensor,
    position: int,
    budget: int,
    graph: LazyKnnGraph,
    prefix: int,
    suffix: int,
    seed_count: int,
    visit_budget: int,
    budget_mode: str,
    target_metric: str,
    target_value: float,
    target_dense_mass: float,
    target_attention_output_cos: float,
    include_static_in_total: bool = True,
    check_interval: int = 16,
    dynamic_tail_seeds: bool = False,
    static_floor_total: bool = False,
) -> Selection:
    max_tok = int(position)
    base = (
        static_tokens_for_position(position, prefix, suffix)
        if budget_mode == "dynamic" or include_static_in_total
        else []
    )
    base_set = set(base)
    seeds = _ra_dynamic_seeds(
        keys=graph.keys,
        max_tok=max_tok,
        prefix=prefix,
        suffix=suffix,
        seed_count=seed_count,
        static_set=base_set,
        dynamic_tail_only=dynamic_tail_seeds,
    )

    probs = torch.softmax(scores.float(), dim=-1)
    dense_out = torch.matmul(probs, values.float())

    if static_floor_total and budget_mode == "total" and base:
        initial_tokens, initial_static, initial_dynamic = compose_static_floor_selection(
            base,
            [],
            budget,
            max_tok,
        )
        mass, cos, _rel_l2 = _quality_for_tokens(scores, values, initial_tokens, probs, dense_out)
        observed = mass if target_metric == "mass" else cos
        if observed >= float(target_value):
            return Selection(
                token_ids=initial_tokens,
                static_tokens=initial_static,
                dynamic_selected_tokens=initial_dynamic,
                dynamic_budget=max(0, int(budget) - int(initial_static)),
                target_dense_mass=float(target_dense_mass),
                target_attention_output_cos=float(target_attention_output_cos),
                target_reached=True,
            )

    import heapq

    frontier = []
    scored_nodes = set()
    for seed in seeds:
        scored_nodes.add(int(seed))
        heapq.heappush(frontier, (-float(scores[seed]), int(seed)))
    visited = set()
    candidates = []
    edges_touched = 0
    max_visits = int(visit_budget) if int(visit_budget) > 0 else int(budget)
    max_visits = max(1, max_visits)
    check_interval = max(1, int(check_interval))
    target_reached = False
    final_tokens: list[int] = []

    while frontier and len(visited) < max_visits:
        neg_score, node = heapq.heappop(frontier)
        if node in visited:
            continue
        visited.add(node)
        if node not in base_set:
            candidates.append((-float(neg_score), int(node)))
        for nbr in graph.neighbors(node, max_tok):
            edges_touched += 1
            if nbr not in visited and (not dynamic_tail_seeds or nbr not in base_set):
                scored_nodes.add(int(nbr))
                heapq.heappush(frontier, (-float(scores[nbr]), int(nbr)))

        if len(visited) == 1 or len(visited) % check_interval == 0:
            ranked = [node for _score, node in sorted(candidates, reverse=True, key=lambda item: item[0])]
            if static_floor_total and budget_mode == "total":
                tokens, static_count, dynamic_count = compose_static_floor_selection(base, ranked, budget, max_tok)
            else:
                tokens, static_count, dynamic_count = compose_selection(base, ranked, budget, max_tok, budget_mode)
            final_tokens = tokens
            mass, cos, _rel_l2 = _quality_for_tokens(scores, values, tokens, probs, dense_out)
            observed = mass if target_metric == "mass" else cos
            if observed >= float(target_value):
                target_reached = True
                break

    if not final_tokens:
        ranked = [node for _score, node in sorted(candidates, reverse=True, key=lambda item: item[0])]
        if static_floor_total and budget_mode == "total":
            final_tokens, static_count, dynamic_count = compose_static_floor_selection(base, ranked, budget, max_tok)
        else:
            final_tokens, static_count, dynamic_count = compose_selection(base, ranked, budget, max_tok, budget_mode)
    else:
        _tokens, static_count, dynamic_count = compose_selection(base, [], budget, max_tok, budget_mode)
        final_set = set(final_tokens)
        static_count = len(final_set.intersection(base_set))
        dynamic_count = max(0, len(final_tokens) - static_count)

    return Selection(
        token_ids=final_tokens,
        static_tokens=static_count,
        dynamic_selected_tokens=dynamic_count,
        dynamic_budget=(
            max(0, int(budget) - int(static_count))
            if static_floor_total and budget_mode == "total"
            else int(budget)
        ),
        metadata_reads=int(len(visited) + edges_touched),
        graph_nodes_visited=int(len(visited)),
        graph_edges_touched=int(edges_touched),
        key_score_reads=int(len(scored_nodes)),
        rerank_key_reads=int(len(candidates)),
        edge_index_reads=int(edges_touched),
        graph_offset_reads=int(len(visited)),
        target_dense_mass=float(target_dense_mass),
        target_attention_output_cos=float(target_attention_output_cos),
        target_reached=bool(target_reached),
    )


def select_retrievalattention_target_sweep(
    scores: torch.Tensor,
    values: torch.Tensor,
    position: int,
    budget: int,
    graph: LazyKnnGraph,
    prefix: int,
    suffix: int,
    seed_count: int,
    visit_budget: int,
    budget_mode: str,
    mass_targets: list[float],
    cos_targets: list[float],
    check_interval: int = 1,
) -> dict[str, Selection]:
    max_tok = int(position)
    base = static_tokens_for_position(position, prefix, suffix)
    base_set = set(base)
    seeds = _ra_dynamic_seeds(
        keys=graph.keys,
        max_tok=max_tok,
        prefix=prefix,
        suffix=suffix,
        seed_count=seed_count,
        static_set=base_set,
        dynamic_tail_only=True,
    )

    targets: dict[str, tuple[str, float]] = {}
    for target in mass_targets:
        targets[f"retrievalattention_target_mass_{str(float(target)).replace('.', 'p')}"] = ("mass", float(target))
    for target in cos_targets:
        targets[f"retrievalattention_target_cos_{str(float(target)).replace('.', 'p')}"] = ("cos", float(target))

    probs = torch.softmax(scores.float(), dim=-1)
    dense_out = torch.matmul(probs, values.float())

    def make_selection(
        tokens: list[int],
        static_count: int,
        dynamic_count: int,
        visited_count: int,
        edge_count: int,
        scored_count: int,
        metric: str,
        target: float,
        reached: bool,
    ) -> Selection:
        return Selection(
            token_ids=list(tokens),
            static_tokens=int(static_count),
            dynamic_selected_tokens=int(dynamic_count),
            dynamic_budget=max(0, int(budget) - int(static_count)),
            metadata_reads=int(visited_count + edge_count),
            graph_nodes_visited=int(visited_count),
            graph_edges_touched=int(edge_count),
            key_score_reads=int(scored_count),
            rerank_key_reads=int(dynamic_count),
            edge_index_reads=int(edge_count),
            graph_offset_reads=int(visited_count),
            target_dense_mass=float(target) if metric == "mass" else 0.0,
            target_attention_output_cos=float(target) if metric == "cos" else 0.0,
            target_reached=bool(reached),
        )

    results: dict[str, Selection] = {}
    pending = set(targets.keys())
    final_tokens, final_static, final_dynamic = compose_static_floor_selection(base, [], budget, max_tok)
    mass, cos, _rel_l2 = _quality_for_tokens(scores, values, final_tokens, probs, dense_out)
    for label in list(pending):
        metric, target = targets[label]
        observed = mass if metric == "mass" else cos
        if observed >= target:
            results[label] = make_selection(final_tokens, final_static, final_dynamic, 0, 0, 0, metric, target, True)
            pending.remove(label)
    if not pending:
        return results

    import heapq

    frontier = []
    scored_nodes = set()
    for seed in seeds:
        scored_nodes.add(int(seed))
        heapq.heappush(frontier, (-float(scores[seed]), int(seed)))
    visited = set()
    candidates = []
    edges_touched = 0
    max_visits = int(visit_budget) if int(visit_budget) > 0 else int(budget)
    max_visits = max(1, max_visits)
    check_interval = max(1, int(check_interval))

    while frontier and len(visited) < max_visits and pending:
        neg_score, node = heapq.heappop(frontier)
        if node in visited:
            continue
        visited.add(node)
        if node not in base_set:
            candidates.append((-float(neg_score), int(node)))
        for nbr in graph.neighbors(node, max_tok):
            edges_touched += 1
            if nbr not in visited and nbr not in base_set:
                scored_nodes.add(int(nbr))
                heapq.heappush(frontier, (-float(scores[nbr]), int(nbr)))

        if len(visited) == 1 or len(visited) % check_interval == 0:
            ranked = [node for _score, node in sorted(candidates, reverse=True, key=lambda item: item[0])]
            final_tokens, final_static, final_dynamic = compose_static_floor_selection(base, ranked, budget, max_tok)
            mass, cos, _rel_l2 = _quality_for_tokens(scores, values, final_tokens, probs, dense_out)
            for label in list(pending):
                metric, target = targets[label]
                observed = mass if metric == "mass" else cos
                if observed >= target:
                    results[label] = make_selection(
                        final_tokens,
                        final_static,
                        final_dynamic,
                        len(visited),
                        edges_touched,
                        len(scored_nodes),
                        metric,
                        target,
                        True,
                    )
                    pending.remove(label)

    if pending:
        ranked = [node for _score, node in sorted(candidates, reverse=True, key=lambda item: item[0])]
        final_tokens, final_static, final_dynamic = compose_static_floor_selection(base, ranked, budget, max_tok)
        for label in pending:
            metric, target = targets[label]
            results[label] = make_selection(
                final_tokens,
                final_static,
                final_dynamic,
                len(visited),
                edges_touched,
                len(scored_nodes),
                metric,
                target,
                False,
            )
    return results


def evaluate_selection(
    scores: torch.Tensor,
    values: torch.Tensor,
    selection: Selection,
    budget: int,
    budget_mode: str,
) -> dict:
    probs = torch.softmax(scores.float(), dim=-1)
    dense_out = torch.matmul(probs, values.float())
    exact_k = min(int(budget), int(scores.numel()))
    exact = set(torch.topk(scores, k=exact_k, largest=True).indices.detach().cpu().tolist())
    selected = unique_budgeted(selection.token_ids, len(selection.token_ids), int(scores.numel()) - 1)
    selected_set = set(selected)
    if selected:
        idx = torch.as_tensor(selected, dtype=torch.long, device=scores.device)
        mass = float(probs.index_select(0, idx).sum().item())
        sparse_probs = torch.softmax(scores.index_select(0, idx).float(), dim=-1)
        sparse_out = torch.matmul(sparse_probs, values.index_select(0, idx).float())
        l2 = float(torch.linalg.vector_norm(sparse_out - dense_out).item())
        denom = max(1.0e-8, float(torch.linalg.vector_norm(dense_out).item()))
        rel_l2 = float(l2 / denom)
        cos = float(torch.nn.functional.cosine_similarity(sparse_out, dense_out, dim=0).item())
    else:
        mass = 0.0
        rel_l2 = float("inf")
        cos = 0.0
    recall = float(len(selected_set.intersection(exact)) / max(1, len(exact)))
    return {
        "selected_tokens": int(len(selected)),
        "static_tokens": int(selection.static_tokens),
        "dynamic_selected_tokens": int(selection.dynamic_selected_tokens),
        "dynamic_budget": int(selection.dynamic_budget),
        "budget_mode": budget_mode,
        "token_read_ratio": float(len(selected) / max(1, int(scores.numel()))),
        "dense_mass_covered": mass,
        "recall_at_budget": recall,
        "relative_attention_output_l2": rel_l2,
        "attention_output_cos": cos,
        "target_dense_mass": selection.target_dense_mass,
        "target_attention_output_cos": selection.target_attention_output_cos,
        "target_reached": selection.target_reached,
    }


def estimate_read_cost(args, selection: Selection, selected_tokens: int) -> dict[str, float | int]:
    """Byte-weighted read model for the proxy.

    Edges and offsets are metadata reads. K-score reads model the vectors read
    to rank frontier candidates. Final attention reads model the selected K/V
    vectors used to compute sparse attention.
    """

    head_dim = int(args.head_dim)
    score_key_reads = int(selection.key_score_reads) + int(selection.centroid_score_reads)
    rerank_key_reads = int(selection.rerank_key_reads) if bool(args.include_rerank_cost) else 0
    final_kv_tokens = int(selected_tokens)
    final_kv_bytes = int(
        final_kv_tokens
        * head_dim
        * (int(args.attn_key_bytes_per_element) + int(args.value_bytes_per_element))
    )
    score_key_bytes = int(
        (score_key_reads + rerank_key_reads)
        * head_dim
        * int(args.score_key_bytes_per_element)
    )
    edge_index_bytes = int(selection.edge_index_reads) * int(args.edge_index_bytes)
    graph_offset_bytes = int(selection.graph_offset_reads) * int(args.graph_offset_bytes)
    estimated = int(final_kv_bytes + score_key_bytes + edge_index_bytes + graph_offset_bytes)
    return {
        "estimated_read_bytes": int(estimated),
        "estimated_read_mb": float(estimated / (1024.0 * 1024.0)),
        "final_kv_read_tokens": int(final_kv_tokens),
        "key_score_reads": int(selection.key_score_reads),
        "rerank_key_reads": int(rerank_key_reads),
        "centroid_score_reads": int(selection.centroid_score_reads),
        "edge_index_reads": int(selection.edge_index_reads),
        "graph_offset_reads": int(selection.graph_offset_reads),
        "final_kv_read_bytes": int(final_kv_bytes),
        "score_key_read_bytes": int(score_key_bytes),
        "edge_index_read_bytes": int(edge_index_bytes),
        "graph_offset_read_bytes": int(graph_offset_bytes),
    }


def run_for_length(args, n_tokens: int, out_jsonl) -> list[dict]:
    device = torch.device(args.device)
    decode_lengths = parse_int_list(args.decode_lengths)
    max_decode = max(0, max(decode_lengths))
    qkv_tokens = int(n_tokens) + max_decode
    if args.source_npz:
        keys, values, query_specs = load_npz_qkv(Path(args.source_npz), args.num_queries, args.seed, device)
    else:
        keys, values, query_specs = make_synthetic_qkv(
            n_tokens=qkv_tokens,
            num_heads=args.num_heads,
            num_kv_heads=args.num_kv_heads,
            head_dim=args.head_dim,
            num_queries=args.num_queries,
            seed=args.seed + int(n_tokens),
            device=device,
        )
    if decode_lengths != [0] and not args.source_npz:
        query_specs = []
        for decode_len in decode_lengths:
            query_specs.extend(
                make_decode_query_specs(
                    keys=keys,
                    prefill_tokens=int(n_tokens),
                    decode_tokens=int(decode_len),
                    num_heads=args.num_heads,
                    num_kv_heads=args.num_kv_heads,
                    num_queries=args.num_queries,
                    seed=args.seed + int(n_tokens) * 1009 + int(decode_len),
                    device=device,
                )
            )

    methods = parse_str_list(args.methods)
    retro_cache = {}
    graph_cache = {}
    online_graph_cache = {}
    rows = []
    if float(args.score_scale) > 0.0:
        score_scale = float(args.score_scale)
    elif args.source_npz:
        score_scale = 1.0 / math.sqrt(float(keys.shape[-1]))
    else:
        # Synthetic keys and queries are L2-normalized. A larger scale gives a
        # non-uniform attention distribution, making mass coverage meaningful.
        score_scale = 16.0

    def get_online_graph(kv_h: int, prefill_tokens: int):
        backend = str(args.ra_graph_backend)
        if backend == "auto":
            backend = "precomputed" if keys.device.type == "cuda" else "lazy"
        key = (
            int(kv_h),
            int(args.graph_degree),
            int(prefill_tokens),
            int(args.static_prefix),
            int(args.static_suffix),
            backend,
            int(args.ra_precompute_chunk),
        )
        if key not in online_graph_cache:
            if backend == "precomputed":
                online_graph_cache[key] = PrecomputedOnlineKnnGraph(
                    keys[int(kv_h)],
                    args.graph_degree,
                    prefill_tokens=int(prefill_tokens),
                    prefix=args.static_prefix,
                    suffix=args.static_suffix,
                    chunk_size=args.ra_precompute_chunk,
                )
            elif backend == "lazy":
                online_graph_cache[key] = OnlineKnnGraph(
                    keys[int(kv_h)],
                    args.graph_degree,
                    prefill_tokens=int(prefill_tokens),
                    prefix=args.static_prefix,
                    suffix=args.static_suffix,
                )
            else:
                raise ValueError(f"unknown ra_graph_backend: {args.ra_graph_backend}")
        return online_graph_cache[key]

    for spec in query_specs:
        kv_h = int(spec.kv_head)
        usable_scores = torch.matmul(keys[kv_h, : spec.position + 1], spec.query) * score_scale
        usable_values = values[kv_h, : spec.position + 1]
        static_tokens = static_tokens_for_position(spec.position, args.static_prefix, args.static_suffix)
        budgets = resolve_budgets_for_spec(args, spec)
        for budget in budgets:
            retro_target_selection = None
            retro_target_metrics = None
            def record_selection(method_label: str, selection: Selection) -> None:
                metrics = evaluate_selection(usable_scores, usable_values, selection, budget, args.budget_mode)
                row = {
                    "source": "npz" if args.source_npz else "synthetic_clustered",
                    "n_tokens": int(n_tokens),
                    "prefill_tokens": int(spec.prefill_tokens or n_tokens),
                    "decode_tokens": int(spec.decode_tokens),
                    "qid": int(spec.qid),
                    "position": int(spec.position),
                    "head": int(spec.head),
                    "kv_head": int(spec.kv_head),
                    "budget": int(budget),
                    "budget_policy": args.budget_policy,
                    "budget_ratio": float(args.budget_ratio),
                    "budget_mode": args.budget_mode,
                    "score_scale": float(score_scale),
                    "method": method_label,
                    "metadata_reads": int(selection.metadata_reads),
                    "graph_nodes_visited": int(selection.graph_nodes_visited),
                    "graph_edges_touched": int(selection.graph_edges_touched),
                    "clusters_scored": int(selection.clusters_scored),
                    "clusters_selected": int(selection.clusters_selected),
                }
                row.update(metrics)
                row.update(estimate_read_cost(args, selection, int(row["selected_tokens"])))
                out_jsonl.write(json.dumps(row, sort_keys=True) + "\n")
                rows.append(row)

            for method in methods:
                absolute_target = parse_ra_absolute_target_method(method)
                if method in (
                    "retrievalattention_match_retro_mass",
                    "retrievalattention_match_retro_cos",
                ) and retro_target_metrics is None:
                    retro_target_selection = select_retroinfer_target_for_match(
                        args=args,
                        spec=spec,
                        keys=keys,
                        query=spec.query,
                        position=spec.position,
                        budget=budget,
                        kv_h=kv_h,
                        n_tokens=n_tokens,
                        retro_cache=retro_cache,
                    )
                    retro_target_metrics = evaluate_selection(
                        usable_scores,
                        usable_values,
                        retro_target_selection,
                        budget,
                        args.budget_mode,
                    )
                if method == "dense_oracle":
                    selection = select_dense_oracle(usable_scores, budget, static_tokens, args.budget_mode)
                elif method == "static_only":
                    selection = select_static_only(
                        position=spec.position,
                        budget=budget,
                        prefix=args.static_prefix,
                        suffix=args.static_suffix,
                        budget_mode=args.budget_mode,
                    )
                elif method == "retroinfer_static_extension":
                    selection = select_retroinfer_static_extension(
                        position=spec.position,
                        budget=budget,
                        prefill_tokens=spec.prefill_tokens or n_tokens,
                        prefix=args.static_prefix,
                        suffix=args.static_suffix,
                    )
                elif method == "static_chunk":
                    selection = select_static_chunk(
                        position=spec.position,
                        budget=budget,
                        prefix=args.static_prefix,
                        suffix=args.static_suffix,
                        chunk=args.chunk_size,
                        budget_mode=args.budget_mode,
                    )
                elif method in ("retroinfer_style", "retroinfer_mixed", "retroinfer_dynamic_only"):
                    cluster_limit = retro_cluster_limit_for_spec(args, spec, n_tokens, int(keys.shape[1]))
                    key = (
                        kv_h,
                        int(args.retro_cluster_size),
                        int(cluster_limit),
                        str(args.retro_cluster_scope),
                    )
                    if key not in retro_cache:
                        retro_cache[key] = build_retro_clusters(keys[kv_h, :cluster_limit], args.retro_cluster_size)
                    centroids, ranges = retro_cache[key]
                    selection = select_retroinfer_style(
                        query=spec.query,
                        position=spec.position,
                        budget=budget,
                        centroids=centroids,
                        ranges=ranges,
                        prefix=args.static_prefix,
                        suffix=args.static_suffix,
                        budget_mode=args.budget_mode,
                        include_static=method != "retroinfer_dynamic_only",
                    )
                elif method == "retrievalattention_style":
                    key = (kv_h, int(args.graph_degree))
                    if key not in graph_cache:
                        graph_cache[key] = LazyKnnGraph(keys[kv_h], args.graph_degree)
                    selection = select_retrievalattention_style(
                        query=spec.query,
                        scores=usable_scores,
                        position=spec.position,
                        budget=budget,
                        graph=graph_cache[key],
                        prefix=args.static_prefix,
                        suffix=args.static_suffix,
                        seed_count=args.ra_seed_count,
                        visit_budget=budget if int(args.ra_visit_budget) <= 0 else args.ra_visit_budget,
                        budget_mode=args.budget_mode,
                        include_static_in_total=False,
                    )
                elif method == "retrievalattention_online_graph":
                    prefill_tokens = int(spec.prefill_tokens or n_tokens)
                    graph = get_online_graph(kv_h, prefill_tokens)
                    selection = select_retrievalattention_online_graph(
                        query=spec.query,
                        scores=usable_scores,
                        position=spec.position,
                        budget=budget,
                        graph=graph,
                        prefix=args.static_prefix,
                        suffix=args.static_suffix,
                        seed_count=args.ra_seed_count,
                        visit_budget=budget if int(args.ra_visit_budget) <= 0 else args.ra_visit_budget,
                        budget_mode=args.budget_mode,
                    )
                elif method in ("retrievalattention_match_retro_mass", "retrievalattention_match_retro_cos"):
                    prefill_tokens = int(spec.prefill_tokens or n_tokens)
                    graph = get_online_graph(kv_h, prefill_tokens)
                    assert retro_target_metrics is not None
                    metric_name = "mass" if method.endswith("_mass") else "cos"
                    target_value = (
                        float(retro_target_metrics["dense_mass_covered"])
                        if metric_name == "mass"
                        else float(retro_target_metrics["attention_output_cos"])
                    )
                    selection = select_retrievalattention_until_target(
                        query=spec.query,
                        scores=usable_scores,
                        values=usable_values,
                        position=spec.position,
                        budget=budget,
                        graph=graph,
                        prefix=args.static_prefix,
                        suffix=args.static_suffix,
                        seed_count=args.ra_seed_count,
                        visit_budget=budget if int(args.ra_visit_budget) <= 0 else args.ra_visit_budget,
                        budget_mode=args.budget_mode,
                        target_metric=metric_name,
                        target_value=target_value,
                        target_dense_mass=float(retro_target_metrics["dense_mass_covered"]),
                        target_attention_output_cos=float(retro_target_metrics["attention_output_cos"]),
                        include_static_in_total=True,
                        check_interval=args.adaptive_check_interval,
                        dynamic_tail_seeds=True,
                        static_floor_total=True,
                    )
                elif method == "retrievalattention_target_sweep":
                    prefill_tokens = int(spec.prefill_tokens or n_tokens)
                    graph = get_online_graph(kv_h, prefill_tokens)
                    selections = select_retrievalattention_target_sweep(
                        scores=usable_scores,
                        values=usable_values,
                        position=spec.position,
                        budget=budget,
                        graph=graph,
                        prefix=args.static_prefix,
                        suffix=args.static_suffix,
                        seed_count=args.ra_seed_count,
                        visit_budget=budget if int(args.ra_visit_budget) <= 0 else args.ra_visit_budget,
                        budget_mode=args.budget_mode,
                        mass_targets=parse_float_list(args.ra_mass_targets),
                        cos_targets=parse_float_list(args.ra_cos_targets),
                        check_interval=args.adaptive_check_interval,
                    )
                    for method_label, selection in selections.items():
                        record_selection(method_label, selection)
                    continue
                elif absolute_target is not None:
                    prefill_tokens = int(spec.prefill_tokens or n_tokens)
                    graph = get_online_graph(kv_h, prefill_tokens)
                    metric_name, target_value = absolute_target
                    selection = select_retrievalattention_until_target(
                        query=spec.query,
                        scores=usable_scores,
                        values=usable_values,
                        position=spec.position,
                        budget=budget,
                        graph=graph,
                        prefix=args.static_prefix,
                        suffix=args.static_suffix,
                        seed_count=args.ra_seed_count,
                        visit_budget=budget if int(args.ra_visit_budget) <= 0 else args.ra_visit_budget,
                        budget_mode=args.budget_mode,
                        target_metric=metric_name,
                        target_value=target_value,
                        target_dense_mass=target_value if metric_name == "mass" else 0.0,
                        target_attention_output_cos=target_value if metric_name == "cos" else 0.0,
                        include_static_in_total=True,
                        check_interval=args.adaptive_check_interval,
                        dynamic_tail_seeds=True,
                        static_floor_total=True,
                    )
                else:
                    raise ValueError(f"unknown method: {method}")

                record_selection(method, selection)
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        grouped.setdefault(
            (
                row["n_tokens"],
                row.get("prefill_tokens", row["n_tokens"]),
                row.get("decode_tokens", 0),
                row.get("budget_policy", "fixed"),
                row["budget"],
                row["budget_mode"],
                row["method"],
            ),
            [],
        ).append(row)
    out = []
    metric_keys = [
        "selected_tokens",
        "static_tokens",
        "dynamic_selected_tokens",
        "dynamic_budget",
        "token_read_ratio",
        "estimated_read_bytes",
        "estimated_read_mb",
        "dense_mass_covered",
        "recall_at_budget",
        "relative_attention_output_l2",
        "attention_output_cos",
        "metadata_reads",
        "graph_nodes_visited",
        "graph_edges_touched",
        "clusters_scored",
        "clusters_selected",
        "target_dense_mass",
        "target_attention_output_cos",
        "target_reached",
    ]
    for (n_tokens, prefill_tokens, decode_tokens, budget_policy, budget, budget_mode, method), group in sorted(grouped.items()):
        row = {
            "n_tokens": n_tokens,
            "prefill_tokens": prefill_tokens,
            "decode_tokens": decode_tokens,
            "budget_policy": budget_policy,
            "budget": budget,
            "budget_mode": budget_mode,
            "method": method,
            "samples": len(group),
        }
        for key in metric_keys:
            vals = [float(x[key]) for x in group if x.get(key) is not None and math.isfinite(float(x[key]))]
            row[f"{key}_mean"] = float(np.mean(vals)) if vals else None
        out.append(row)
    return out


def write_summary_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def maybe_plot(summary_rows: list[dict], output_dir: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", f"/tmp/matplotlib-{os.getuid()}")
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[attention_efficiency_eval] plot skipped: {type(exc).__name__}: {exc}")
        return
    by_len = sorted({row["n_tokens"] for row in summary_rows})
    for n_tokens in by_len:
        rows = [r for r in summary_rows if r["n_tokens"] == n_tokens]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        targets = [
            ("estimated_read_mb_mean", "dense_mass_covered_mean", "Dense Mass Covered"),
            ("estimated_read_mb_mean", "recall_at_budget_mean", "Recall@Budget"),
            ("estimated_read_mb_mean", "relative_attention_output_l2_mean", "Relative Output L2"),
        ]
        methods = sorted({r["method"] for r in rows})
        for ax, (x_key, y_key, title) in zip(axes, targets):
            for method in methods:
                mr = sorted([r for r in rows if r["method"] == method], key=lambda r: r["budget"])
                xs = [r[x_key] for r in mr]
                ys = [r[y_key] for r in mr]
            ax.plot(xs, ys, marker="o", label=method)
            ax.set_title(title)
            ax.set_xlabel("Estimated Read MB")
            ax.grid(True, alpha=0.3)
        axes[0].set_ylabel("Higher is better")
        axes[2].set_ylabel("Lower is better")
        axes[0].legend(fontsize=8)
        fig.suptitle(f"Attention Efficiency Proxy, N={n_tokens}")
        fig.tight_layout()
        fig.savefig(output_dir / f"attention_efficiency_n{n_tokens}.png", dpi=160)
        plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Compare sparse-attention algorithmic efficiency.")
    parser.add_argument("--output_dir", default="attention_efficiency_result/proxy_v1")
    parser.add_argument("--source_npz", default="")
    parser.add_argument("--context_lengths", default="16384,32768")
    parser.add_argument(
        "--decode_lengths",
        default="0",
        help=(
            "Comma-separated generated decode lengths. 0 preserves the fixed-context proxy. "
            "Nonzero values treat --context_lengths as prefill lengths and evaluate queries "
            "at prefill+decode-1."
        ),
    )
    parser.add_argument("--budgets", default="64,128,256,512,1024,2048")
    parser.add_argument(
        "--budget_policy",
        choices=("fixed", "linear", "log2", "retro_static_extension"),
        default="fixed",
        help=(
            "fixed uses --budgets; linear uses ceil(--budget_ratio * current causal tokens); "
            "log2 uses ceil(log2(current causal tokens)); retro_static_extension matches "
            "the static-extension token count."
        ),
    )
    parser.add_argument("--budget_ratio", type=float, default=0.10)
    parser.add_argument(
        "--budget_mode",
        choices=("dynamic", "total"),
        default="dynamic",
        help=(
            "dynamic treats --budgets as extra retrieved tokens beyond static prefix/suffix; "
            "total treats --budgets as the full selected-token budget. In total mode, "
            "dense_oracle is pure dense top-k while sparse methods count any static tokens they use."
        ),
    )
    parser.add_argument(
        "--methods",
        default="dense_oracle,static_chunk,retroinfer_style,retrievalattention_style",
        help=(
            "Comma-separated methods. Supported: dense_oracle, static_only, static_chunk, "
            "retroinfer_style/retroinfer_mixed, retroinfer_dynamic_only, "
            "retroinfer_static_extension, retrievalattention_style, retrievalattention_online_graph, "
            "retrievalattention_match_retro_mass, retrievalattention_match_retro_cos, "
            "retrievalattention_target_mass_<value>, retrievalattention_target_cos_<value>. "
            "Use p instead of . in method names, e.g. retrievalattention_target_mass_0p4. "
            "retrievalattention_target_sweep evaluates --ra_mass_targets and --ra_cos_targets in one traversal."
        ),
    )
    parser.add_argument("--num_queries", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=32)
    parser.add_argument("--num_kv_heads", type=int, default=8)
    parser.add_argument("--head_dim", type=int, default=128)
    parser.add_argument("--static_prefix", type=int, default=128)
    parser.add_argument("--static_suffix", type=int, default=512)
    parser.add_argument("--chunk_size", type=int, default=128)
    parser.add_argument("--retro_cluster_size", type=int, default=128)
    parser.add_argument(
        "--retro_cluster_scope",
        choices=("prefill", "causal", "full"),
        default="prefill",
        help=(
            "Visible key range for RetroInfer-style centroid clusters. prefill builds clusters "
            "only over the prefill/current causal range and is the default long-decode baseline; "
            "causal is an optimistic online-clustering diagnostic; full is a non-causal legacy upper bound."
        ),
    )
    parser.add_argument(
        "--retro_target_method",
        choices=("retroinfer_style", "retroinfer_mixed", "retroinfer_dynamic_only", "retroinfer_static_extension", "static_only"),
        default="retroinfer_style",
        help=(
            "RetroInfer target used by retrievalattention_match_retro_* adaptive diagnostics. "
            "Default is the clustered RetroInfer-style mixed baseline; static-only extensions "
            "must be requested explicitly as controls."
        ),
    )
    parser.add_argument("--graph_degree", type=int, default=16)
    parser.add_argument(
        "--ra_graph_backend",
        choices=("auto", "lazy", "precomputed"),
        default="lazy",
        help=(
            "Online graph backend for RetrievalAttention diagnostics. lazy computes exact KNN "
            "neighbors on demand; precomputed builds the same prefill graph and generated-token "
            "overlay in batched matmul chunks before traversal; auto uses precomputed on CUDA."
        ),
    )
    parser.add_argument(
        "--ra_precompute_chunk",
        type=int,
        default=512,
        help="Rows per chunk when --ra_graph_backend=precomputed.",
    )
    parser.add_argument("--ra_seed_count", type=int, default=32)
    parser.add_argument(
        "--ra_visit_budget",
        type=int,
        default=2048,
        help="Traversal visit cap for RetrievalAttention proxy. Use 0 to match the resolved per-query token budget.",
    )
    parser.add_argument(
        "--adaptive_check_interval",
        type=int,
        default=16,
        help="For oracle-threshold adaptive RA diagnostics, check dense target metrics every N visited graph nodes.",
    )
    parser.add_argument(
        "--ra_mass_targets",
        default="0.1,0.2,0.4,0.6",
        help="Mass targets for retrievalattention_target_sweep.",
    )
    parser.add_argument(
        "--ra_cos_targets",
        default="0.2,0.4,0.6,0.8",
        help="Output-cosine targets for retrievalattention_target_sweep.",
    )
    parser.add_argument(
        "--score_scale",
        type=float,
        default=0.0,
        help="Score multiplier. 0 uses 16.0 for synthetic normalized QK and 1/sqrt(D) for NPZ.",
    )
    parser.add_argument(
        "--score_key_bytes_per_element",
        type=int,
        default=4,
        help="Bytes per K element read for candidate/centroid scoring in the byte-cost model.",
    )
    parser.add_argument(
        "--attn_key_bytes_per_element",
        type=int,
        default=2,
        help="Bytes per selected attention K element in the byte-cost model.",
    )
    parser.add_argument(
        "--value_bytes_per_element",
        type=int,
        default=2,
        help="Bytes per selected V element in the byte-cost model.",
    )
    parser.add_argument(
        "--edge_index_bytes",
        type=int,
        default=4,
        help="Bytes per graph neighbor ID in the byte-cost model.",
    )
    parser.add_argument(
        "--graph_offset_bytes",
        type=int,
        default=4,
        help="Bytes per graph row offset read in the byte-cost model.",
    )
    parser.add_argument(
        "--include_rerank_cost",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include an extra K-score read over candidates to model runtime reranking.",
    )
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "args.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True))

    all_rows = []
    jsonl_path = output_dir / "attention_efficiency_samples.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as fout:
        for n_tokens in parse_int_list(args.context_lengths):
            print(f"[attention_efficiency_eval] running n_tokens={n_tokens}")
            all_rows.extend(run_for_length(args, n_tokens=n_tokens, out_jsonl=fout))
            fout.flush()

    summary_rows = summarize(all_rows)
    summary_path = output_dir / "summary.csv"
    write_summary_csv(summary_rows, summary_path)
    (output_dir / "summary.json").write_text(json.dumps(summary_rows, indent=2, sort_keys=True))
    if args.plot:
        maybe_plot(summary_rows, output_dir)
    print(f"[attention_efficiency_eval] samples={jsonl_path}")
    print(f"[attention_efficiency_eval] summary={summary_path}")


if __name__ == "__main__":
    main()
