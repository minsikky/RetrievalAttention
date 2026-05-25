#!/usr/bin/env python3
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


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
