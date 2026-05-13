#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaForCausalLM, apply_rotary_pos_emb

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.selector_eval.data.trace import static_tokens, unique_tokens
from benchmark.selector_eval.gpu.run_gpu_paged_pq_eval import (
    build_page_pq_gpu,
    parse_csv_ints,
    rank_paged_pq,
    selected_plus_tail_output,
)
from benchmark.selector_eval.runners.diagnose_layer_heads import _compressed_tail_output
from benchmark.selector_eval.runners.run_layer_quality_eval import _vpq_values_for_tokens
from benchmark.selector_eval.metrics.attention import _output_error_metrics

MB = 1024.0 * 1024.0


def log(msg: str) -> None:
    print(f"[hf_paged_pq_intervention_eval] {time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


@dataclass
class ApproxStats:
    calls: int = 0
    mean_selected: float = 0.0
    mean_tail_samples: float = 0.0
    mean_selector_mb: float = 0.0
    mean_exact_kv_mb: float = 0.0
    mean_tail_mb: float = 0.0
    mean_step_mb: float = 0.0

    def add(
        self,
        selected: list[int],
        tail_count: int,
        selector_mb: float,
        head_dim: int,
        key_bytes: int,
        value_bytes: int,
        tail_mb_override: float | None = None,
        exact_kv_mb_override: float | None = None,
    ) -> None:
        exact_kv_mb = (
            float(exact_kv_mb_override)
            if exact_kv_mb_override is not None
            else float(len(selected) * head_dim * (key_bytes + value_bytes)) / MB
        )
        tail_mb = (
            float(tail_mb_override)
            if tail_mb_override is not None
            else float(tail_count * head_dim * (key_bytes + value_bytes)) / MB
        )
        step_mb = float(selector_mb) + exact_kv_mb + tail_mb
        self.calls += 1
        alpha = 1.0 / float(self.calls)
        self.mean_selected += alpha * (float(len(selected)) - self.mean_selected)
        self.mean_tail_samples += alpha * (float(tail_count) - self.mean_tail_samples)
        self.mean_selector_mb += alpha * (float(selector_mb) - self.mean_selector_mb)
        self.mean_exact_kv_mb += alpha * (exact_kv_mb - self.mean_exact_kv_mb)
        self.mean_tail_mb += alpha * (tail_mb - self.mean_tail_mb)
        self.mean_step_mb += alpha * (step_mb - self.mean_step_mb)


def parse_head_budget_map(text: str) -> dict[int, int]:
    out: dict[int, int] = {}
    for part in str(text or "").split(","):
        part = part.strip()
        if not part:
            continue
        head, budget = part.split(":", 1)
        out[int(head)] = int(budget)
    return out


def parse_int_set(text: str) -> set[int]:
    return {int(part.strip()) for part in str(text or "").split(",") if part.strip()}


def output_metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    return _output_error_metrics(
        a.detach().float().cpu().numpy().reshape(-1),
        b.detach().float().cpu().numpy().reshape(-1),
    )


def selected_value_exact_mask(
    *,
    selected_scores: np.ndarray,
    values_exact: np.ndarray,
    values_approx: np.ndarray | None,
    rule: str,
    exact_top: int,
    exact_mass: float,
    exact_risk_mass: float,
    min_top: int,
    max_top: int,
) -> tuple[np.ndarray, float]:
    count = int(selected_scores.shape[0])
    mask = np.zeros((count,), dtype=bool)
    if count <= 0:
        return mask, 0.0
    scores = selected_scores.astype(np.float64, copy=False)
    shifted = scores - float(np.max(scores))
    probs = np.exp(shifted)
    probs /= max(float(probs.sum()), 1e-20)
    if str(rule) == "fixed":
        order = np.argsort(-scores, kind="stable")
        exact_count = int(exact_top)
        if exact_count > 0:
            mask[order[: min(count, exact_count)]] = True
    elif str(rule) == "selected_mass":
        order = np.argsort(-scores, kind="stable")
        target = float(max(0.0, min(1.0, exact_mass)))
        cumulative = np.cumsum(probs[order])
        exact_count = int(np.searchsorted(cumulative, target, side="left") + 1) if target > 0.0 else 0
        if exact_count > 0:
            mask[order[: min(count, exact_count)]] = True
    elif str(rule) in {"selected_risk_mass", "selected_mass_or_risk"}:
        if values_approx is None:
            raise ValueError(f"{rule} requires approximate selected values")
        residual_norm = np.linalg.norm(
            values_exact.astype(np.float32, copy=False) - values_approx.astype(np.float32, copy=False),
            axis=1,
        ) / float(np.sqrt(float(values_exact.shape[-1])))
        risk = probs * residual_norm.astype(np.float64, copy=False)
        risk_order = np.argsort(-risk, kind="stable")
        target = float(exact_risk_mass) if float(exact_risk_mass) > 0.0 else float(exact_mass)
        total_risk = float(risk.sum())
        if total_risk > 1e-20 and target > 0.0:
            cumulative = np.cumsum(risk[risk_order]) / total_risk
            exact_count = int(np.searchsorted(cumulative, float(max(0.0, min(1.0, target))), side="left") + 1)
        else:
            exact_count = int(exact_top)
        if exact_count > 0:
            mask[risk_order[: min(count, exact_count)]] = True
        if str(rule) == "selected_mass_or_risk":
            prob_order = np.argsort(-scores, kind="stable")
            mass_target = float(max(0.0, min(1.0, exact_mass)))
            if mass_target > 0.0:
                cumulative = np.cumsum(probs[prob_order])
                mass_count = int(np.searchsorted(cumulative, mass_target, side="left") + 1)
                mask[prob_order[: min(count, mass_count)]] = True
    else:
        raise ValueError(f"unknown selected_value_exact_rule: {rule}")
    if int(min_top) > 0 and int(np.sum(mask)) < int(min_top):
        order = np.argsort(-scores, kind="stable")
        mask[order[: min(count, int(min_top))]] = True
    if int(max_top) > 0 and int(np.sum(mask)) > int(max_top):
        order = np.argsort(-(probs * (1.0 + np.arange(count, 0, -1) / max(1, count))), kind="stable")
        limited = np.zeros((count,), dtype=bool)
        limited[order[: min(count, int(max_top))]] = True
        mask = limited
    return mask, float(probs[mask].sum()) if bool(np.any(mask)) else 0.0


def make_needle_prompt(target: str, filler_repeats: int) -> str:
    filler = "\n".join(
        f"Background line {i:04d}: this line is irrelevant context about calendars, rivers, and copper."
        for i in range(int(filler_repeats))
    )
    return (
        "You are given a long document. Find the secret code and answer with only the code.\n\n"
        f"{filler}\n\n"
        f"IMPORTANT FACT: the secret code is {target}.\n\n"
        f"{filler}\n\n"
        "Question: What is the secret code?\nAnswer:"
    )


def greedy_dense_trace(model, input_ids: torch.Tensor, max_new_tokens: int, forbidden: set[int]) -> dict[str, Any]:
    logits_trace: list[torch.Tensor] = []
    hidden_trace: list[torch.Tensor] = []
    tokens: list[int] = []
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True, output_hidden_states=True, return_dict=True)
        past = out.past_key_values
        logits = out.logits[:, -1, :]
        for step in range(int(max_new_tokens)):
            if forbidden:
                logits = logits.clone()
                logits[:, list(forbidden)] = torch.finfo(logits.dtype).min
            logits_trace.append(logits.detach().float().cpu())
            hidden_trace.append(out.hidden_states[-1][:, -1, :].detach().float().cpu())
            next_tok = int(torch.argmax(logits, dim=-1).item())
            tokens.append(next_tok)
            if step == int(max_new_tokens) - 1:
                break
            cur = torch.tensor([[next_tok]], dtype=torch.long, device=input_ids.device)
            out = model(input_ids=cur, past_key_values=past, use_cache=True, output_hidden_states=True, return_dict=True)
            past = out.past_key_values
            logits = out.logits[:, -1, :]
    return {"tokens": tokens, "logits": logits_trace, "hidden": hidden_trace}


def teacher_forced_trace(model, input_ids: torch.Tensor, forced_tokens: list[int], forbidden: set[int]) -> dict[str, Any]:
    logits_trace: list[torch.Tensor] = []
    hidden_trace: list[torch.Tensor] = []
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True, output_hidden_states=True, return_dict=True)
        past = out.past_key_values
        logits = out.logits[:, -1, :]
        for step, tok in enumerate(forced_tokens):
            if forbidden:
                logits = logits.clone()
                logits[:, list(forbidden)] = torch.finfo(logits.dtype).min
            logits_trace.append(logits.detach().float().cpu())
            hidden_trace.append(out.hidden_states[-1][:, -1, :].detach().float().cpu())
            if step == len(forced_tokens) - 1:
                break
            cur = torch.tensor([[int(tok)]], dtype=torch.long, device=input_ids.device)
            out = model(input_ids=cur, past_key_values=past, use_cache=True, output_hidden_states=True, return_dict=True)
            past = out.past_key_values
            logits = out.logits[:, -1, :]
    return {"logits": logits_trace, "hidden": hidden_trace}


def summarize_logit_trace(dense: dict[str, Any], approx: dict[str, Any], tokenizer, ignore_token_ids: set[int] | None = None) -> dict[str, Any]:
    ignore_token_ids = set(ignore_token_ids or set())
    rows = []
    for step, (dl, al, dh, ah) in enumerate(zip(dense["logits"], approx["logits"], dense["hidden"], approx["hidden"], strict=False)):
        dl = dl.reshape(-1).float()
        al = al.reshape(-1).float()
        if ignore_token_ids:
            keep = torch.ones_like(dl, dtype=torch.bool)
            valid_ids = [tok for tok in ignore_token_ids if 0 <= int(tok) < int(keep.numel())]
            if valid_ids:
                keep[torch.as_tensor(valid_ids, dtype=torch.long)] = False
                dl_metric = dl[keep]
                al_metric = al[keep]
            else:
                dl_metric = dl
                al_metric = al
        else:
            dl_metric = dl
            al_metric = al
        dh = dh.reshape(-1).float()
        ah = ah.reshape(-1).float()
        dense_top = int(torch.argmax(dl).item())
        approx_top = int(torch.argmax(al).item())
        probs_d = torch.softmax(dl, dim=-1)
        log_probs_a = torch.log_softmax(al, dim=-1)
        kl = torch.sum(probs_d * (torch.log(torch.clamp(probs_d, min=1e-30)) - log_probs_a)).item()
        logit_diff = torch.linalg.vector_norm((dl_metric - al_metric).double()).item()
        logit_norm = torch.linalg.vector_norm(dl_metric.double()).item()
        logit_dot = torch.dot(dl_metric.double(), al_metric.double()).item()
        approx_norm = torch.linalg.vector_norm(al_metric.double()).item()
        logit_cos = logit_dot / max(1e-30, logit_norm * approx_norm)
        rows.append(
            {
                "step": int(step),
                "dense_top": dense_top,
                "approx_top": approx_top,
                "top1_match": bool(dense_top == approx_top),
                "dense_top_text": tokenizer.decode([dense_top]),
                "approx_top_text": tokenizer.decode([approx_top]),
                "logit_l2": float(logit_diff),
                "logit_relative_l2": float(logit_diff / max(1e-30, logit_norm)),
                "logit_cosine": float(logit_cos),
                "dense_to_approx_kl": float(kl),
                "hidden_relative_l2": float(torch.linalg.vector_norm(dh - ah) / torch.clamp(torch.linalg.vector_norm(dh), min=1e-20)),
                "hidden_cosine": float(F.cosine_similarity(dh.unsqueeze(0), ah.unsqueeze(0), dim=-1).item()),
            }
        )
    if not rows:
        return {"steps": [], "summary": {}}
    summary = {
        "steps": int(len(rows)),
        "top1_agreement": float(np.mean([float(r["top1_match"]) for r in rows])),
        "mean_logit_relative_l2": float(np.mean([r["logit_relative_l2"] for r in rows])),
        "max_logit_relative_l2": float(np.max([r["logit_relative_l2"] for r in rows])),
        "mean_logit_cosine": float(np.mean([r["logit_cosine"] for r in rows])),
        "min_logit_cosine": float(np.min([r["logit_cosine"] for r in rows])),
        "mean_dense_to_approx_kl": float(np.mean([r["dense_to_approx_kl"] for r in rows])),
        "max_dense_to_approx_kl": float(np.max([r["dense_to_approx_kl"] for r in rows])),
        "mean_hidden_relative_l2": float(np.mean([r["hidden_relative_l2"] for r in rows])),
        "max_hidden_relative_l2": float(np.max([r["hidden_relative_l2"] for r in rows])),
        "mean_hidden_cosine": float(np.mean([r["hidden_cosine"] for r in rows])),
        "min_hidden_cosine": float(np.min([r["hidden_cosine"] for r in rows])),
    }
    affected = [r for r in rows if int(r["step"]) > 0]
    if affected:
        summary.update(
            {
                "affected_steps": int(len(affected)),
                "affected_top1_agreement": float(np.mean([float(r["top1_match"]) for r in affected])),
                "affected_mean_logit_relative_l2": float(np.mean([r["logit_relative_l2"] for r in affected])),
                "affected_max_logit_relative_l2": float(np.max([r["logit_relative_l2"] for r in affected])),
                "affected_mean_dense_to_approx_kl": float(np.mean([r["dense_to_approx_kl"] for r in affected])),
                "affected_max_dense_to_approx_kl": float(np.max([r["dense_to_approx_kl"] for r in affected])),
                "affected_mean_hidden_relative_l2": float(np.mean([r["hidden_relative_l2"] for r in affected])),
                "affected_max_hidden_relative_l2": float(np.max([r["hidden_relative_l2"] for r in affected])),
            }
        )
    else:
        summary["affected_steps"] = 0
    return {"steps": rows, "summary": summary}


@contextlib.contextmanager
def patched_paged_pq_attention(model, layer_ids: list[int], args, stats: dict[int, ApproxStats]):
    originals = {}
    device = next(model.parameters()).device
    key_bytes = int(args.key_bytes)
    value_bytes = int(args.value_bytes)
    nprobes = parse_csv_ints(args.nprobes)
    budget_by_head = parse_head_budget_map(args.budget_by_head)
    tail_off_heads = parse_int_set(args.tail_off_heads)

    def make_forward(layer_id: int, module):
        original_forward = module.forward

        def forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings,
            attention_mask,
            past_key_value=None,
            cache_position=None,
            **kwargs,
        ):
            input_shape = hidden_states.shape[:-1]
            if input_shape[-1] != 1 or past_key_value is None:
                return original_forward(
                    hidden_states=hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                    past_key_value=past_key_value,
                    cache_position=cache_position,
                    **kwargs,
                )

            hidden_shape = (*input_shape, -1, self.head_dim)
            query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

            keys_all = key_states[0].detach().to(torch.float32)
            values_all = value_states[0].detach().to(torch.float32)
            q_all = query_states[0, :, 0, :].detach().to(torch.float32)
            context_len = int(keys_all.shape[1])
            num_heads = int(getattr(self, "num_heads", self.config.num_attention_heads))
            num_kv_heads = int(getattr(self, "num_key_value_heads", self.config.num_key_value_heads))
            group_size = int(num_heads // num_kv_heads)
            outputs = []

            index_cache = {}
            torch_k_cache = {}
            torch_v_cache = {}
            dynamic_start = min(max(0, int(args.static_prefix)), context_len)
            indexed_end = max(dynamic_start, context_len - max(0, int(args.static_suffix)))
            for kv_head in range(num_kv_heads):
                keys_np = keys_all[kv_head].detach().cpu().numpy().astype(np.float32, copy=False)
                index_cache[kv_head] = build_page_pq_gpu(
                    keys_np,
                    dynamic_start=dynamic_start,
                    indexed_end=indexed_end,
                    page_size=int(args.page_size),
                    subvecs=int(args.subvecs),
                    subbits=int(args.subbits),
                    kmeans_iters=int(args.kmeans_iters),
                    seed=int(args.seed) + int(kv_head) * 2025 + int(context_len),
                    key_bytes=key_bytes,
                    router_enabled=str(args.selector_mode) == "routed",
                    router_prototypes=int(args.router_prototypes),
                    router_merge_rel=float(args.router_merge_rel),
                    router_merge_var=float(args.router_merge_var),
                    router_max_groups=int(args.router_max_groups),
                    device=device,
                )
                torch_k_cache[kv_head] = keys_all[kv_head].to(device)
                torch_v_cache[kv_head] = values_all[kv_head].to(device)

            for head in range(num_heads):
                budget = int(budget_by_head.get(int(head), int(args.budget)))
                kv_head = int(head // group_size)
                query = q_all[head].to(device)
                index = index_cache[kv_head]
                base = unique_tokens(
                    static_tokens(context_len - 1, int(args.static_prefix), int(args.static_suffix)),
                    context_len=context_len,
                )
                ranked_t, ranked_scores_t, _seconds, selector_mb, _nprobe = rank_paged_pq(
                    query,
                    index,
                    mode=str(args.selector_mode),
                    nprobes=nprobes,
                    budget=budget,
                    key_bytes=key_bytes,
                    subbits=int(args.subbits),
                )
                ranked_cpu = ranked_t.detach().cpu().numpy().astype(np.int64, copy=False)
                ranked_scores_cpu = ranked_scores_t.detach().cpu().numpy().astype(np.float32, copy=False)
                rerank_key_mb = 0.0
                if int(args.rerank_candidates) > 0 and ranked_cpu.size:
                    rerank_count = min(int(args.rerank_candidates), int(ranked_cpu.size))
                    rerank_tokens = torch.as_tensor(ranked_cpu[:rerank_count], dtype=torch.long, device=device)
                    rerank_scores = torch_k_cache[kv_head].index_select(0, rerank_tokens) @ query
                    rerank_order = torch.argsort(rerank_scores, descending=True, stable=True).detach().cpu().numpy()
                    reranked = ranked_cpu[:rerank_count][rerank_order].astype(np.int64, copy=False)
                    reranked_set = set(int(tok) for tok in reranked.tolist())
                    rest = np.asarray([int(tok) for tok in ranked_cpu.tolist() if int(tok) not in reranked_set], dtype=np.int64)
                    ranked_cpu = np.concatenate([reranked, rest]) if rest.size else reranked
                    rerank_key_mb = float(rerank_count * int(self.head_dim) * key_bytes) / MB
                base_set = set(int(tok) for tok in base)
                add = [int(tok) for tok in ranked_cpu.tolist() if int(tok) < context_len and int(tok) not in base_set][:budget]
                selected_cpu = np.asarray(unique_tokens(base + add, context_len=context_len), dtype=np.int64)
                selected = torch.as_tensor(selected_cpu, dtype=torch.long, device=device)
                if selected.numel() == 0:
                    selected_only = torch.zeros((int(self.head_dim),), dtype=torch.float32, device=device)
                    selected_values_np = np.zeros((0, int(self.head_dim)), dtype=np.float32)
                    selected_value_mb = 0.0
                else:
                    selected_keys = torch_k_cache[kv_head].index_select(0, selected)
                    selected_logits = (selected_keys @ query) / math.sqrt(float(self.head_dim))
                    values_np = values_all[kv_head].detach().cpu().numpy().astype(np.float32, copy=False)
                    selected_scores_np = selected_logits.detach().cpu().numpy().astype(np.float32, copy=False)
                    if str(args.selected_value_mode) == "vpq_value":
                        exact_values_np = values_np[selected_cpu].astype(np.float32, copy=False)
                        if str(args.selected_value_exact_rule) in {"selected_risk_mass", "selected_mass_or_risk"}:
                            approx_values_all, compressed_v_mb, fallback_v_mb = _vpq_values_for_tokens(
                                index=index,
                                values_np=values_np,
                                tokens=selected_cpu.astype(np.int64, copy=False),
                                subbits=int(args.subbits),
                                value_subvecs=int(args.value_subvecs),
                                value_subbits=int(args.value_subbits),
                                value_bytes=value_bytes,
                            )
                            compressed_v_mb += float(
                                selected_cpu.size * max(0, int(args.selected_value_residual_norm_bytes))
                            ) / MB
                            exact_mask, _exact_selected_mass = selected_value_exact_mask(
                                selected_scores=selected_scores_np,
                                values_exact=exact_values_np,
                                values_approx=approx_values_all,
                                rule=str(args.selected_value_exact_rule),
                                exact_top=int(args.selected_value_exact_top),
                                exact_mass=float(args.selected_value_exact_mass),
                                exact_risk_mass=float(args.selected_value_exact_risk_mass),
                                min_top=int(args.selected_value_min_exact_top),
                                max_top=int(args.selected_value_max_exact_top),
                            )
                            selected_values_np = approx_values_all.astype(np.float32, copy=True)
                            selected_values_np[exact_mask] = exact_values_np[exact_mask]
                        else:
                            exact_mask, _exact_selected_mass = selected_value_exact_mask(
                                selected_scores=selected_scores_np,
                                values_exact=exact_values_np,
                                values_approx=None,
                                rule=str(args.selected_value_exact_rule),
                                exact_top=int(args.selected_value_exact_top),
                                exact_mass=float(args.selected_value_exact_mass),
                                exact_risk_mass=float(args.selected_value_exact_risk_mass),
                                min_top=int(args.selected_value_min_exact_top),
                                max_top=int(args.selected_value_max_exact_top),
                            )
                            compressed_mask = ~exact_mask
                            selected_values_np = np.empty_like(exact_values_np)
                            selected_values_np[exact_mask] = exact_values_np[exact_mask]
                            compressed_v_mb = 0.0
                            fallback_v_mb = 0.0
                            if bool(np.any(compressed_mask)):
                                approx_values, compressed_v_mb, fallback_v_mb = _vpq_values_for_tokens(
                                    index=index,
                                    values_np=values_np,
                                    tokens=selected_cpu[compressed_mask].astype(np.int64, copy=False),
                                    subbits=int(args.subbits),
                                    value_subvecs=int(args.value_subvecs),
                                    value_subbits=int(args.value_subbits),
                                    value_bytes=value_bytes,
                                )
                                selected_values_np[compressed_mask] = approx_values
                        exact_value_mb = float(np.sum(exact_mask) * int(self.head_dim) * value_bytes) / MB
                        selected_value_mb = float(compressed_v_mb) + float(fallback_v_mb) + exact_value_mb
                        selected_values = torch.as_tensor(selected_values_np, dtype=torch.float32, device=device)
                    else:
                        selected_values = torch_v_cache[kv_head].index_select(0, selected)
                        selected_values_np = selected_values.detach().cpu().numpy().astype(np.float32, copy=False)
                        selected_value_mb = float(selected_cpu.size * int(self.head_dim) * value_bytes) / MB
                    selected_weights = torch.softmax(selected_logits.float(), dim=0).to(selected_values.dtype)
                    selected_only = (selected_weights.unsqueeze(0) @ selected_values).squeeze(0).float()
                tail_blend = float(args.tail_blend)
                effective_tail_blend = 0.0 if int(head) in tail_off_heads else tail_blend
                tail_mb_override = None
                if effective_tail_blend <= 0.0:
                    approx_head = selected_only
                    tail_count = 0
                elif str(args.tail_mode) in {"pq_value", "vpq_value", "page_mean"}:
                    scores_np = np.zeros((context_len,), dtype=np.float32)
                    if selected_cpu.size:
                        scores_np[selected_cpu] = selected_logits.detach().cpu().numpy().astype(np.float32, copy=False)
                    values_np = values_all[kv_head].detach().cpu().numpy().astype(np.float32, copy=False)
                    approx_tail_np, tail_count, _tail_population, tail_mb_override = _compressed_tail_output(
                        index=index,
                        values_np=values_np,
                        scores_np=scores_np,
                        ranked_cpu=ranked_cpu,
                        ranked_scores_cpu=ranked_scores_cpu,
                        selected_cpu=selected_cpu,
                        query_dim=int(self.head_dim),
                        subbits=int(args.subbits),
                        value_bytes=value_bytes,
                        mode=str(args.tail_mode),
                        value_subvecs=int(args.value_subvecs),
                        value_subbits=int(args.value_subbits),
                        selected_values_np=selected_values_np,
                    )
                    tail_probe_rel_l2 = float(
                        np.linalg.norm(approx_tail_np.astype(np.float64, copy=False) - selected_only.detach().cpu().numpy().astype(np.float64, copy=False))
                    ) / max(float(torch.linalg.vector_norm(selected_only.float()).item()), 1e-20)
                    approx_tail = torch.as_tensor(approx_tail_np, dtype=torch.float32, device=device)
                    if tail_probe_rel_l2 > float(args.tail_probe_rel_l2_max):
                        approx_head = selected_only
                    else:
                        approx_head = (
                            approx_tail
                            if effective_tail_blend >= 1.0
                            else selected_only + effective_tail_blend * (approx_tail.float() - selected_only)
                        )
                else:
                    approx_tail, tail_count, _tail_population, _attn_seconds = selected_plus_tail_output(
                        torch_k_cache[kv_head],
                        torch_v_cache[kv_head],
                        query,
                        selected,
                        ranked_cpu,
                        np.zeros((context_len,), dtype=np.float32),
                        context_len=context_len,
                        samples=int(args.tail_samples),
                        bands=int(args.tail_bands),
                        seed=int(args.tail_seed),
                        qidx=context_len,
                        head=head,
                        sampling=str(args.tail_sampling),
                    )
                    approx_head = (
                        approx_tail
                        if effective_tail_blend >= 1.0
                        else selected_only + effective_tail_blend * (approx_tail.float() - selected_only)
                    )
                stats[layer_id].add(
                    selected_cpu.tolist(),
                    tail_count,
                    float(selector_mb) + float(rerank_key_mb),
                    int(self.head_dim),
                    key_bytes,
                    value_bytes,
                    tail_mb_override=tail_mb_override,
                    exact_kv_mb_override=float(selected_cpu.size * int(self.head_dim) * key_bytes) / MB + float(selected_value_mb),
                )
                outputs.append(approx_head.to(hidden_states.dtype))

            attn_output = torch.stack(outputs, dim=0).reshape(1, 1, -1).contiguous()
            attn_output = self.o_proj(attn_output)
            return attn_output, None

        return types.MethodType(forward, module)

    try:
        for layer_id in layer_ids:
            module = model.model.layers[int(layer_id)].self_attn
            originals[int(layer_id)] = module.forward
            stats[int(layer_id)] = ApproxStats()
            module.forward = make_forward(int(layer_id), module)
        yield
    finally:
        for layer_id, forward in originals.items():
            model.model.layers[int(layer_id)].self_attn.forward = forward


def run() -> None:
    parser = argparse.ArgumentParser(description="HF decode/logit/task-style eval for routed paged-PQ + stratified tail intervention.")
    parser.add_argument("--model_name", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--cache_dir", default=".hf_cache")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--layers", default="16")
    parser.add_argument("--filler_repeats", type=int, default=128)
    parser.add_argument("--target", default="ZEBRA-4729")
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--selector_mode", choices=["fullscan", "routed"], default="routed")
    parser.add_argument("--budget", type=int, default=4096)
    parser.add_argument("--budget_by_head", default="")
    parser.add_argument("--rerank_candidates", type=int, default=0)
    parser.add_argument("--tail_samples", type=int, default=16384)
    parser.add_argument("--tail_bands", type=int, default=8)
    parser.add_argument("--tail_seed", type=int, default=0)
    parser.add_argument("--tail_sampling", choices=["random", "linspace", "systematic"], default="systematic")
    parser.add_argument("--tail_mode", choices=["sample", "pq_value", "vpq_value", "page_mean"], default="sample")
    parser.add_argument("--tail_probe_rel_l2_max", type=float, default=float("inf"))
    parser.add_argument("--selected_value_mode", choices=["exact", "vpq_value"], default="exact")
    parser.add_argument(
        "--selected_value_exact_rule",
        choices=["fixed", "selected_mass", "selected_risk_mass", "selected_mass_or_risk"],
        default="fixed",
    )
    parser.add_argument("--selected_value_exact_top", type=int, default=0)
    parser.add_argument("--selected_value_exact_mass", type=float, default=0.0)
    parser.add_argument("--selected_value_exact_risk_mass", type=float, default=0.0)
    parser.add_argument("--selected_value_min_exact_top", type=int, default=0)
    parser.add_argument("--selected_value_max_exact_top", type=int, default=0)
    parser.add_argument("--selected_value_residual_norm_bytes", type=int, default=2)
    parser.add_argument("--value_subvecs", type=int, default=0)
    parser.add_argument("--value_subbits", type=int, default=0)
    parser.add_argument("--tail_blend", type=float, default=1.0)
    parser.add_argument("--tail_off_heads", default="")
    parser.add_argument("--static_prefix", type=int, default=128)
    parser.add_argument("--static_suffix", type=int, default=128)
    parser.add_argument("--page_size", type=int, default=5632)
    parser.add_argument("--subvecs", type=int, default=4)
    parser.add_argument("--subbits", type=int, default=6)
    parser.add_argument("--kmeans_iters", type=int, default=3)
    parser.add_argument("--nprobes", default="16,32,64,128,256,512")
    parser.add_argument("--router_prototypes", type=int, default=16)
    parser.add_argument("--router_merge_rel", type=float, default=0.05)
    parser.add_argument("--router_merge_var", type=float, default=0.0)
    parser.add_argument("--router_max_groups", type=int, default=512)
    parser.add_argument("--key_bytes", type=int, default=2)
    parser.add_argument("--value_bytes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--cpu_then_to_device", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True), encoding="utf-8")

    device = torch.device(args.device)
    log("loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir, local_files_only=bool(args.local_files_only))
    tokenizer.pad_token = tokenizer.eos_token
    log("loading model")
    load_kwargs = {
        "cache_dir": args.cache_dir,
        "local_files_only": bool(args.local_files_only),
        "torch_dtype": torch.bfloat16,
        "attn_implementation": str(args.attn_implementation),
    }
    if not bool(args.cpu_then_to_device):
        load_kwargs["device_map"] = {"": str(device)}
    model = LlamaForCausalLM.from_pretrained(args.model_name, **load_kwargs)
    if bool(args.cpu_then_to_device):
        log(f"moving model to {device}")
        model = model.to(device)
    model.eval()
    forbidden = {tok for tok in [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")] if isinstance(tok, int) and tok >= 0}

    prompt = make_needle_prompt(str(args.target), int(args.filler_repeats))
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    log(f"prompt_tokens={int(input_ids.shape[1])} dense_trace_start")
    start = time.perf_counter()
    dense = greedy_dense_trace(model, input_ids, int(args.max_new_tokens), forbidden)
    dense_time = time.perf_counter() - start
    log(f"dense_trace_done seconds={dense_time:.3f}")
    dense_text = tokenizer.decode(dense["tokens"], skip_special_tokens=True)

    layer_ids = parse_csv_ints(args.layers)
    approx_stats: dict[int, ApproxStats] = {}
    start = time.perf_counter()
    log(f"approx_trace_start layers={layer_ids}")
    with patched_paged_pq_attention(model, layer_ids, args, approx_stats):
        approx_teacher = teacher_forced_trace(model, input_ids, dense["tokens"], forbidden)
        approx_free = greedy_dense_trace(model, input_ids, int(args.max_new_tokens), forbidden)
    approx_time = time.perf_counter() - start
    log(f"approx_trace_done seconds={approx_time:.3f}")
    approx_text = tokenizer.decode(approx_free["tokens"], skip_special_tokens=True)

    comparison = summarize_logit_trace(dense, approx_teacher, tokenizer, ignore_token_ids=forbidden)
    stats_payload = {
        str(layer): {
            "calls": s.calls,
            "mean_selected_tokens": s.mean_selected,
            "mean_tail_samples": s.mean_tail_samples,
            "mean_selector_MB_per_head_query": s.mean_selector_mb,
            "mean_exact_KV_MB_per_head_query": s.mean_exact_kv_mb,
            "mean_tail_estimator_MB_per_head_query": s.mean_tail_mb,
            "mean_step_MB_per_head_query": s.mean_step_mb,
        }
        for layer, s in sorted(approx_stats.items())
    }
    task = {
        "target": str(args.target),
        "prompt_tokens": int(input_ids.shape[1]),
        "dense_text": dense_text,
        "approx_free_text": approx_text,
        "dense_contains_target": str(args.target).lower() in dense_text.lower(),
        "approx_free_contains_target": str(args.target).lower() in approx_text.lower(),
        "free_run_exact_text_match": dense_text == approx_text,
        "dense_tokens": [int(x) for x in dense["tokens"]],
        "approx_free_tokens": [int(x) for x in approx_free["tokens"]],
    }
    summary = {
        "algorithm": (
            f"hf_routed_paged_pq_k{int(args.budget)}"
            f"_{args.selector_mode}"
            f"_rerank{int(args.rerank_candidates)}"
            f"+{args.tail_mode}_tail_b{int(args.tail_bands)}_s{int(args.tail_samples)}"
            f"_blend{float(args.tail_blend):g}"
        ),
        "layers": layer_ids,
        "dense_seconds": float(dense_time),
        "approx_seconds": float(approx_time),
        "logit_trace": comparison["summary"],
        "task": task,
        "cost_proxy": stats_payload,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "logit_steps.json").write_text(json.dumps(comparison["steps"], indent=2, sort_keys=True), encoding="utf-8")
    log(f"wrote {out_dir}")


if __name__ == "__main__":
    run()
