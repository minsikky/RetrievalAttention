import argparse
import copy
import json
import math
import os
import random
import re
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model_hub import LlamaModel, QwenModel
from config import generate_config


CODEBOOK = [
    "red", "blue", "green", "gold", "silver", "black", "white", "orange",
    "purple", "yellow", "brown", "pink", "cyan", "lime", "coral", "navy",
]
CODEBOOK_SET = set(CODEBOOK)
ENTRY_RE = re.compile(r"^ENTRY\s+(\d+):\s+([a-z]+)\s*$", re.MULTILINE)
ANSWER_RE = re.compile(r"^ANSWER\s+(\d+):\s+([a-z]+)\s*$", re.MULTILINE)


def parse_args():
    parser = argparse.ArgumentParser(description="Generated-memory decode benchmark")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--attn_type", type=str, default="RetrievalAttention")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--num_entries", type=int, default=192)
    parser.add_argument("--num_queries", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=1600)
    parser.add_argument("--retrieval_budget", type=float, default=0.018)
    parser.add_argument("--estimation_budget", type=float, default=0.232)
    parser.add_argument("--token_budget_override", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="generated_memory_eval_result")
    parser.add_argument("--prefill_filler_repeats", type=int, default=0)
    parser.add_argument("--min_prompt_tokens", type=int, default=0)
    parser.add_argument("--generation_margin_tokens", type=int, default=64)
    parser.add_argument(
        "--teacher_ledger_attn_type",
        type=str,
        default=os.environ.get("GENERATED_MEMORY_TEACHER_LEDGER_ATTN_TYPE", ""),
    )
    parser.add_argument(
        "--teacher_drift_diag",
        action="store_true",
        help="Record dense-vs-student layerwise decode hidden-state drift under teacher-forced ledger tokens.",
    )
    parser.add_argument(
        "--teacher_drift_max_steps",
        type=int,
        default=0,
        help="Optional cap on the number of teacher-forced ledger decode steps to compare (0 = all).",
    )
    parser.add_argument(
        "--teacher_dense_kv_refresh",
        action="store_true",
        help="After each teacher-forced sparse ledger decode step, overwrite the new student KV row with the dense teacher KV row.",
    )
    parser.add_argument(
        "--replay_prefill_compare",
        action="store_true",
        help="After the normal run generates a ledger, rebuild a fresh prompt with that exact ledger text as prefill and decode answers again for comparison.",
    )
    parser.add_argument(
        "--answer_prefix_scaffold",
        action="store_true",
        help="Inject answer prefixes one by one during answer decoding instead of asking the model to emit the full answer-line format by itself.",
    )
    parser.add_argument(
        "--answer_constrained_codebook",
        action="store_true",
        help="When using answer-prefix scaffold, choose each answer from the codebook by constrained next-token selection.",
    )
    parser.add_argument(
        "--replay_question_via_decode",
        action="store_true",
        help="For replay-prefill compare, prefill only through the ledger and append the question prompt via decode, matching the online question path.",
    )
    parser.add_argument(
        "--replay_import_online_prev_seeds",
        action="store_true",
        help="For replay-prefill compare under RetrievalAttention, copy the online answer-start prev-seed state into the replay run before answering.",
    )
    parser.add_argument(
        "--replay_import_online_overlay",
        action="store_true",
        help="For replay-prefill compare under RetrievalAttention, copy the online graph overlay into the replay run before answering.",
    )
    parser.add_argument(
        "--answer_start_clear_prev_seeds",
        action="store_true",
        help="For RetrievalAttention, clear answer-start warm prev-seed state before answering.",
    )
    parser.add_argument(
        "--answer_start_clear_overlay",
        action="store_true",
        help="For RetrievalAttention, clear answer-start online overlay graph before answering.",
    )
    parser.add_argument(
        "--replay_graph_compare",
        action="store_true",
        help="For replay-prefill compare under RetrievalAttention, summarize answer-start neighbor overlap between online generated-memory state and replay-prefill state for generated ledger tokens.",
    )
    parser.add_argument(
        "--state_partition_diag",
        action="store_true",
        help="Record RetrievalAttention static/dynamic partition state at online and replay answer start.",
    )
    parser.add_argument(
        "--state_equiv_diag",
        action="store_true",
        help="Compare answer-start logits/KV/first attention step across online, replay-prefill, and teacher-forced decode states.",
    )
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(model_name, max_len, dtype, device):
    if "Llama" in model_name:
        llm = LlamaModel(
            model_name,
            max_length=max_len,
            dtype=dtype,
            device_map=device,
        )
    elif "Qwen" in model_name:
        llm = QwenModel(
            model_name,
            max_length=max_len,
            dtype=dtype,
            device_map=device,
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return llm


def choose_query_positions(num_entries: int, num_queries: int, rng: random.Random):
    anchors = [4, num_entries // 4, num_entries // 2, max(1, num_entries - 4)]
    while len(anchors) < num_queries:
        anchors.append(max(1, int(round((len(anchors) + 1) * num_entries / float(num_queries + 1)))))
    anchors = anchors[:num_queries]
    positions = []
    for pos in anchors:
        jitter = rng.randint(-3, 3)
        positions.append(min(num_entries, max(1, int(pos) + jitter)))
    positions = sorted(set(positions))
    while len(positions) < num_queries:
        candidate = rng.randint(1, num_entries)
        if candidate not in positions:
            positions.append(candidate)
    return sorted(positions[:num_queries])


def build_ledger_prompt(sample_idx: int, num_entries: int, filler_repeats: int):
    filler = ""
    if filler_repeats > 0:
        filler_line = (
            "FILLER BLOCK. Ignore this block. It exists only to make the prompt longer. "
            "Do not copy it into the answer.\n"
        )
        filler = filler_line * int(filler_repeats)
    codebook_text = ", ".join(CODEBOOK)
    prompt = (
        "You must follow the format exactly.\n"
        f"{filler}"
        f"Write exactly {num_entries} ledger lines.\n"
        "Each ledger line must have exactly this format:\n"
        "ENTRY i: red\n"
        "Value rules:\n"
        f"- the value must be exactly one word from this list: {codebook_text}\n"
        "- repeats are allowed\n"
        "- do not add punctuation or explanations\n"
        "- do not skip or renumber entries\n"
        f"After ENTRY {num_entries}, write exactly this line:\n"
        "END LEDGER\n"
        "Stop immediately after END LEDGER.\n"
        f"Begin with ENTRY 1 now.\n"
        f"# SAMPLE_ID={sample_idx}\n"
    )
    return prompt


def build_question_prompt(query_positions, answer_prefix_scaffold: bool = False):
    question_lines = [
        f"QUESTION {i + 1}: What was the value in ENTRY {int(pos)}?"
        for i, pos in enumerate(query_positions)
    ]
    if answer_prefix_scaffold:
        codebook_text = ", ".join(CODEBOOK)
        return (
            "\nNow answer questions about the ledger you already wrote.\n"
            + "\n".join(question_lines)
            + "\nI will provide each answer prefix one at a time.\n"
            + f"After each prefix, output exactly one word from this list and then stop the line: {codebook_text}\n"
            + "Do not write a sentence. Do not repeat the question. Do not add punctuation.\n"
            + "Correct example:\n"
            + "ANSWER 1: red\n"
            + "Incorrect examples:\n"
            + "ANSWER 1: The value in ENTRY 1 was red.\n"
            + "ANSWER 1: red.\n"
            + "ANSWER 1: ENTRY 1 was red\n"
        )
    answer_format = "\n".join(
        f"ANSWER {i + 1}: <value>"
        for i in range(len(query_positions))
    )
    prompt = (
        "\nNow answer questions about the ledger you already wrote.\n"
        + "\n".join(question_lines)
        + "\nRespond with exactly these lines and nothing else:\n"
        + answer_format
        + "\n"
    )
    return prompt


def build_codebook_token_map(tokenizer):
    token_map = []
    seen = set()
    for word in CODEBOOK:
        ids = tokenizer(" " + word, return_tensors="pt", add_special_tokens=False).input_ids[0].tolist()
        if len(ids) != 1:
            ids = tokenizer(word, return_tensors="pt", add_special_tokens=False).input_ids[0].tolist()
        if len(ids) != 1:
            raise RuntimeError(f"Codebook word {word!r} is not single-token under the current tokenizer.")
        tok_id = int(ids[0])
        if tok_id in seen:
            raise RuntimeError(f"Codebook token collision for id={tok_id} on word {word!r}.")
        seen.add(tok_id)
        token_map.append((tok_id, word))
    return token_map


def ensure_min_prompt_tokens(tokenizer, prompt: str, sample_idx: int, num_entries: int, filler_repeats: int, min_prompt_tokens: int):
    prompt_tokens = len(tokenizer(prompt, return_tensors="pt").input_ids[0])
    if prompt_tokens >= int(min_prompt_tokens):
        return prompt, int(prompt_tokens), int(filler_repeats)

    repeats = int(max(0, filler_repeats))
    while prompt_tokens < int(min_prompt_tokens):
        repeats += 8
        prompt = build_ledger_prompt(
            sample_idx=sample_idx,
            num_entries=num_entries,
            filler_repeats=repeats,
        )
        prompt_tokens = len(tokenizer(prompt, return_tensors="pt").input_ids[0])
    return prompt, int(prompt_tokens), int(repeats)


def extract_generated_region(text: str):
    start = text.find("ENTRY 1:")
    if start < 0:
        return ""
    return text[start:]


def build_expected_output_stub(num_entries: int, num_queries: int):
    lines = [f"ENTRY {i}: red" for i in range(1, int(num_entries) + 1)]
    lines.append("END LEDGER")
    lines.extend(f"ANSWER {i}: red" for i in range(1, int(num_queries) + 1))
    return "\n".join(lines)


def build_ledger_output_stub(num_entries: int):
    lines = [f"ENTRY {i}: red" for i in range(1, int(num_entries) + 1)]
    lines.append("END LEDGER")
    return "\n".join(lines)


def build_answer_output_stub(num_queries: int):
    return "\n".join(f"ANSWER {i}: red" for i in range(1, int(num_queries) + 1))


def aggregate_oracle_compare(records):
    if not records:
        return None
    numeric_keys = [
        "oracle_dyn_mass",
        "total_dynamic_mass",
        "omitted_dynamic_mass",
        "oracle_token_count",
        "sparse_dynamic_count",
        "dense_sparse_out_l2",
        "adaptive_mass_bound",
        "adaptive_upper_score_bound",
        "adaptive_dynamic_span",
        "adaptive_candidate_count",
        "adaptive_keep_count",
    ]
    out = {"num_records": int(len(records))}
    for key in numeric_keys:
        vals = []
        for rec in records:
            value = rec.get(key)
            if value is None:
                continue
            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                continue
            vals.append(float(value))
        if vals:
            out[f"avg_{key}"] = float(np.mean(vals))
    bound_vals = []
    violation_count = 0
    compare_count = 0
    for rec in records:
        bound = rec.get("adaptive_mass_bound")
        actual = rec.get("omitted_dynamic_mass")
        if bound is None or actual is None:
            continue
        if isinstance(bound, float) and (math.isnan(bound) or math.isinf(bound)):
            continue
        if float(bound) < 0.0:
            continue
        bound_vals.append(float(bound) - float(actual))
        compare_count += 1
        if float(actual) > float(bound) + 1e-6:
            violation_count += 1
    if compare_count > 0:
        out["bound_compare_count"] = int(compare_count)
        out["bound_violation_rate"] = float(violation_count) / float(compare_count)
        out["avg_bound_slack"] = float(np.mean(bound_vals))
        out["min_bound_slack"] = float(np.min(bound_vals))
    return out


def evaluate_output(text: str, query_positions, num_entries: int):
    region = extract_generated_region(text)
    entries = {}
    answers = {}
    for match in ENTRY_RE.finditer(region):
        idx = int(match.group(1))
        value = match.group(2)
        if idx not in entries and value in CODEBOOK_SET:
            entries[idx] = value
    for match in ANSWER_RE.finditer(region):
        idx = int(match.group(1))
        value = match.group(2)
        if idx not in answers and value in CODEBOOK_SET:
            answers[idx] = value

    entry_count = len(entries)
    answer_count = len(answers)
    expected_answers = len(query_positions)
    end_marker_present = ("END LEDGER" in region)
    format_ok = entry_count >= int(num_entries) and answer_count >= expected_answers and end_marker_present
    query_hits = []
    for i, pos in enumerate(query_positions, start=1):
        expected = entries.get(int(pos))
        got = answers.get(int(i))
        query_hits.append(bool(expected is not None and got is not None and expected == got))

    unique_entries = len(set(entries.values())) if entries else 0
    return {
        "format_ok": bool(format_ok),
        "end_marker_present": bool(end_marker_present),
        "entry_count": int(entry_count),
        "answer_count": int(answer_count),
        "query_hits": query_hits,
        "query_acc": float(sum(query_hits) / float(len(query_hits))) if query_hits else 0.0,
        "strict_acc": bool(query_hits and all(query_hits)),
        "unique_entries": int(unique_entries),
        "unique_entry_ratio": float(unique_entries / float(max(1, entry_count))),
        "entries": entries,
        "answers": answers,
    }


def compute_entry_token_spans(tokenizer, ledger_prompt: str, ledger_output: str, query_positions):
    prompt_ids = tokenizer(ledger_prompt, return_tensors="pt").input_ids[0]
    prompt_len = int(prompt_ids.shape[0])
    spans = {}
    cursor = 0
    for line in ledger_output.splitlines(keepends=True):
        token_len = len(tokenizer(line, return_tensors="pt", add_special_tokens=False).input_ids[0])
        match = ENTRY_RE.match(line.strip("\n"))
        if match is not None:
            entry_idx = int(match.group(1))
            if entry_idx in query_positions:
                start = prompt_len + int(cursor)
                spans[int(entry_idx)] = list(range(start, start + int(token_len)))
        cursor += int(token_len)
    return spans


def summarize_oracle_debug(records, entry_spans):
    summary = {}
    if not records:
        return summary
    for entry_idx, span in entry_spans.items():
        span_set = set(int(tok) for tok in span)
        hit_records = []
        hit_steps = set()
        hit_layers = set()
        hit_heads = set()
        for rec in records:
            toks = rec.get("tokens", [])
            if any(int(tok) in span_set for tok in toks):
                hit_records.append(rec)
                hit_steps.add(int(rec.get("step", -1)))
                hit_layers.add(int(rec.get("layer", -1)))
                hit_heads.add(int(rec.get("head", -1)))
        summary[str(int(entry_idx))] = {
            "span_start": int(span[0]) if span else -1,
            "span_end": int(span[-1]) if span else -1,
            "span_len": int(len(span)),
            "hit_record_count": int(len(hit_records)),
            "hit_step_count": int(len(hit_steps)),
            "hit_steps": sorted(int(x) for x in hit_steps),
            "hit_layer_count": int(len(hit_layers)),
            "hit_head_count": int(len(hit_heads)),
            "any_hit": bool(hit_records),
        }
    return summary


def init_generation_session(llm, tokenizer, prompt_text: str, attn_type: str, attn_config, total_future_tokens: int):
    inputs = tokenizer([prompt_text], return_tensors="pt", padding=True)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    llm.attention_type = str(attn_type)
    llm.batch_size = int(input_ids.shape[0])
    llm.input_length = int(input_ids.shape[1])
    llm.max_new_length = int(total_future_tokens)
    assert llm.input_length + llm.max_new_length <= llm.max_length, (
        f"input_length({llm.input_length}) + max_new_length({llm.max_new_length}) exceeds max_length({llm.max_length})"
    )
    valid_start = attention_mask.shape[1] - torch.sum(attention_mask, dim=-1).detach().cpu().numpy()
    llm.prefill_bsz = 1
    llm.prefill_method = "full"
    llm.init_kv_cache(valid_start, attn_config)

    device = llm.layers[0].device
    prefill_start = time.time()
    logits = llm.prefill_forward(inputs_ids=input_ids.to(device))
    first_token = llm.sampling(logits, do_sample=False)
    llm.move()
    torch.cuda.synchronize()
    prefill_sec = float(time.time() - prefill_start)
    return first_token.to(device), prefill_sec


def greedy_decode_until(llm, tokenizer, start_token, max_new_tokens: int, stop_substrings):
    device = llm.layers[0].device
    generated_ids = []
    current_token = start_token.to(device)
    next_logits = None
    for _ in range(max(1, int(max_new_tokens))):
        token_id = int(current_token[0, 0].item())
        generated_ids.append(token_id)
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        next_logits = llm.decode_forward(inputs_ids=current_token)
        if any(stop in text for stop in stop_substrings):
            return text, generated_ids, next_logits, True
        current_token = llm.sampling(next_logits, do_sample=False).to(device)
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return text, generated_ids, next_logits, False


def greedy_decode_answers(llm, tokenizer, start_token, max_new_tokens: int, expected_answers: int):
    device = llm.layers[0].device
    generated_ids = []
    current_token = start_token.to(device)
    next_logits = None
    for _ in range(max(1, int(max_new_tokens))):
        token_id = int(current_token[0, 0].item())
        generated_ids.append(token_id)
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        next_logits = llm.decode_forward(inputs_ids=current_token)
        if len(list(ANSWER_RE.finditer(text))) >= int(expected_answers):
            return text, generated_ids, next_logits, True
        current_token = llm.sampling(next_logits, do_sample=False).to(device)
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return text, generated_ids, next_logits, False


def greedy_decode_answers_with_prefix_scaffold(
    llm,
    tokenizer,
    fallback_logits,
    expected_answers: int,
    max_new_tokens: int,
    constrained_token_map=None,
):
    device = llm.layers[0].device
    rendered_text = ""
    generated_ids = []
    next_logits = fallback_logits
    remaining = int(max(1, max_new_tokens))
    allowed_ids_t = None
    allowed_words = None
    if constrained_token_map:
        allowed_ids_t = torch.as_tensor(
            [int(tok_id) for tok_id, _word in constrained_token_map],
            dtype=torch.long,
            device=device,
        )
        allowed_words = [str(word) for _tok_id, word in constrained_token_map]

    for answer_idx in range(int(expected_answers)):
        prefix = f"ANSWER {answer_idx + 1}: "
        if rendered_text and not rendered_text.endswith("\n"):
            prefix = "\n" + prefix
        prefix_ids = tokenizer(prefix, return_tensors="pt", add_special_tokens=False).input_ids[0]
        prefix_token_count = int(prefix_ids.shape[0])
        if remaining < prefix_token_count:
            break
        rendered_text += prefix
        next_logits = append_prompt_continuation(
            llm=llm,
            tokenizer=tokenizer,
            prompt_text=prefix,
            fallback_logits=next_logits,
        )
        remaining -= prefix_token_count
        if remaining <= 0:
            break

        if allowed_ids_t is not None:
            logits_view = next_logits
            if logits_view.dim() == 3:
                logits_view = logits_view[:, -1, :]
            elif logits_view.dim() != 2:
                raise RuntimeError(f"Unexpected logits shape for constrained answer decode: {tuple(logits_view.shape)}")
            allowed_logits = torch.index_select(logits_view, dim=-1, index=allowed_ids_t)
            best_idx = int(torch.argmax(allowed_logits, dim=-1).item())
            chosen_id = int(allowed_ids_t[best_idx].item())
            chosen_word = allowed_words[best_idx]
            generated_ids.append(chosen_id)
            rendered_text += chosen_word
            current_token = torch.tensor([[chosen_id]], dtype=torch.long, device=device)
            next_logits = llm.decode_forward(inputs_ids=current_token)
            remaining -= 1
            continue

        current_token = llm.sampling(next_logits, do_sample=False).to(device)
        value_ids = []
        for _ in range(remaining):
            token_id = int(current_token[0, 0].item())
            generated_ids.append(token_id)
            value_ids.append(token_id)
            value_text = tokenizer.decode(value_ids, skip_special_tokens=True)
            next_logits = llm.decode_forward(inputs_ids=current_token)
            remaining -= 1
            if "\n" in value_text:
                value_text = value_text.split("\n", 1)[0] + "\n"
                rendered_text += value_text
                break
            current_token = llm.sampling(next_logits, do_sample=False).to(device)
            if remaining <= 0:
                rendered_text += value_text
                break
        else:
            rendered_text += tokenizer.decode(value_ids, skip_special_tokens=True)

    parsed_answers = len(list(ANSWER_RE.finditer(rendered_text)))
    return rendered_text, generated_ids, next_logits, bool(parsed_answers >= int(expected_answers))


def teacher_force_decode_ids(llm, tokenizer, forced_ids, stop_substrings=()):
    device = llm.layers[0].device
    generated_ids = []
    next_logits = None
    for tok in forced_ids:
        current_token = torch.tensor([[int(tok)]], dtype=torch.long, device=device)
        generated_ids.append(int(tok))
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        next_logits = llm.decode_forward(inputs_ids=current_token)
        if any(stop in text for stop in stop_substrings):
            return text, generated_ids, next_logits, True
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return text, generated_ids, next_logits, False


def install_decode_trace(llm):
    original_layer_decode = llm.layer_decode
    llm._decode_trace_enabled = False
    llm._decode_trace_steps = []
    llm._decode_trace_current = None
    llm._decode_trace_limit = 0

    def traced_layer_decode(self, layer_idx, hidden_states):
        out = original_layer_decode(layer_idx, hidden_states)
        if getattr(self, "_decode_trace_enabled", False):
            limit = int(getattr(self, "_decode_trace_limit", 0))
            if limit <= 0 or len(self._decode_trace_steps) < limit:
                if int(layer_idx) == 0 or getattr(self, "_decode_trace_current", None) is None:
                    self._decode_trace_current = []
                self._decode_trace_current.append(out[:, -1, :].detach().float().cpu().clone())
                if int(layer_idx) == int(self.num_layers) - 1:
                    self._decode_trace_steps.append(self._decode_trace_current)
                    self._decode_trace_current = None
        return out

    llm.layer_decode = types.MethodType(traced_layer_decode, llm)
    return original_layer_decode


def remove_decode_trace(llm, original_layer_decode):
    llm.layer_decode = original_layer_decode
    if hasattr(llm, "_decode_trace_enabled"):
        delattr(llm, "_decode_trace_enabled")
    if hasattr(llm, "_decode_trace_steps"):
        delattr(llm, "_decode_trace_steps")
    if hasattr(llm, "_decode_trace_current"):
        delattr(llm, "_decode_trace_current")
    if hasattr(llm, "_decode_trace_limit"):
        delattr(llm, "_decode_trace_limit")


def teacher_force_decode_ids_with_trace(llm, tokenizer, forced_ids, stop_substrings=(), max_steps: int = 0):
    original_layer_decode = install_decode_trace(llm)
    llm._decode_trace_enabled = True
    llm._decode_trace_limit = int(max_steps)
    try:
        text, generated_ids, next_logits, saw_stop = teacher_force_decode_ids(
            llm=llm,
            tokenizer=tokenizer,
            forced_ids=list(forced_ids),
            stop_substrings=stop_substrings,
        )
        trace_steps = list(getattr(llm, "_decode_trace_steps", []))
    finally:
        remove_decode_trace(llm, original_layer_decode)
    return text, generated_ids, next_logits, saw_stop, trace_steps


def extract_flash_teacher_decode_kv_rows(llm, start_pos: int, token_count: int):
    rows = []
    kv_cache = llm.kv_cache
    if not hasattr(kv_cache, "key_cache") or not hasattr(kv_cache, "value_cache"):
        raise RuntimeError("Teacher KV refresh requires flash_attn_cache-style key_cache/value_cache.")
    for step in range(int(token_count)):
        pos = int(start_pos) + int(step)
        per_layer = []
        for ldx in range(int(llm.num_layers)):
            key_row = kv_cache.key_cache[ldx][0, pos, :, :].detach().cpu().clone()
            value_row = kv_cache.value_cache[ldx][0, pos, :, :].detach().cpu().clone()
            per_layer.append((key_row, value_row))
        rows.append(per_layer)
    return rows


def overwrite_student_decode_kv_rows(llm, pos: int, dense_rows):
    kv_cache = llm.kv_cache
    if not hasattr(kv_cache, "cpu_keys") or not hasattr(kv_cache, "cpu_values"):
        raise RuntimeError("Student KV refresh requires retrievalattention_cache CPU KV storage.")

    for ldx, (key_row_cpu, value_row_cpu) in enumerate(dense_rows):
        kv_cache.cpu_keys[ldx][:, pos:pos + 1, :].copy_(key_row_cpu.unsqueeze(1), non_blocking=True)
        kv_cache.cpu_values[ldx][:, pos:pos + 1, :].copy_(value_row_cpu.unsqueeze(1), non_blocking=True)

        if getattr(kv_cache, "decode_backend", "") == "roar_cuda_fullgpu":
            device = kv_cache.layer_mapping[str(ldx)]
            key_row = key_row_cpu.to(device, non_blocking=True)
            value_row = value_row_cpu.to(device, non_blocking=True)
            score_row = kv_cache._score_transform_torch(key_row.float())
            attn_norm_row = torch.linalg.vector_norm(key_row.float(), dim=-1)
            for kv_hdx in range(int(kv_cache.kv_head)):
                key_cache = kv_cache._decode_cuda_key_cache[ldx][kv_hdx]
                if key_cache is not None:
                    key_cache[pos:pos + 1, :].copy_(score_row[kv_hdx:kv_hdx + 1], non_blocking=True)
                attn_key_cache = kv_cache._decode_cuda_attn_key_cache[ldx][kv_hdx]
                old_attn_row = None
                if attn_key_cache is not None:
                    old_attn_row = attn_key_cache[pos:pos + 1, :].float().clone()
                    attn_key_cache[pos:pos + 1, :].copy_(key_row[kv_hdx:kv_hdx + 1], non_blocking=True)
                value_cache = kv_cache._decode_cuda_value_cache[ldx][kv_hdx]
                if value_cache is not None:
                    value_cache[pos:pos + 1, :].copy_(value_row[kv_hdx:kv_hdx + 1], non_blocking=True)
                norm_cache = kv_cache._decode_cuda_attn_key_norm_cache[ldx][kv_hdx]
                prefixmax_cache = kv_cache._decode_cuda_attn_key_prefixmax_cache[ldx][kv_hdx]
                if norm_cache is not None:
                    norm_cache[pos] = attn_norm_row[kv_hdx]
                if prefixmax_cache is not None:
                    if pos < int(kv_cache.dynamic_start):
                        prefixmax_cache[pos] = 0.0
                    elif pos == int(kv_cache.dynamic_start):
                        prefixmax_cache[pos] = attn_norm_row[kv_hdx]
                    else:
                        prefixmax_cache[pos] = torch.maximum(prefixmax_cache[pos - 1], attn_norm_row[kv_hdx])
                if (
                    old_attn_row is not None
                    and kv_cache._decode_cuda_attn_key_sum_cache[ldx][kv_hdx] is not None
                    and kv_cache._decode_cuda_attn_key_sumsq_cache[ldx][kv_hdx] is not None
                    and pos >= int(kv_cache.dynamic_start)
                    and pos < int(kv_cache.dynamic_end)
                ):
                    delta = key_row[kv_hdx:kv_hdx + 1].float() - old_attn_row
                    delta_sq = key_row[kv_hdx:kv_hdx + 1].float().square() - old_attn_row.square()
                    kv_cache._decode_cuda_attn_key_sum_cache[ldx][kv_hdx].add_(delta.squeeze(0))
                    kv_cache._decode_cuda_attn_key_sumsq_cache[ldx][kv_hdx].add_(delta_sq.squeeze(0))

            if getattr(kv_cache, "static_pattern_end", 0) > 0 and not getattr(kv_cache, "growing_static_suffix", False):
                suffix_slot = int(kv_cache.static_pattern_start + kv_cache.static_pattern_end - 1)
                kv_cache.static_gpu_keys[ldx][:, suffix_slot, :].copy_(key_row, non_blocking=True)
                kv_cache.static_gpu_values[ldx][:, suffix_slot, :].copy_(value_row, non_blocking=True)


def teacher_force_decode_ids_with_kv_refresh(
    llm,
    tokenizer,
    forced_ids,
    dense_kv_rows,
    stop_substrings=(),
):
    device = llm.layers[0].device
    generated_ids = []
    next_logits = None
    for step_idx, tok in enumerate(forced_ids):
        current_token = torch.tensor([[int(tok)]], dtype=torch.long, device=device)
        generated_ids.append(int(tok))
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        next_logits = llm.decode_forward(inputs_ids=current_token)
        pos = int(llm.kv_cache.context) - 1
        if 0 <= step_idx < len(dense_kv_rows):
            overwrite_student_decode_kv_rows(llm, pos=pos, dense_rows=dense_kv_rows[step_idx])
        if any(stop in text for stop in stop_substrings):
            return text, generated_ids, next_logits, True
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return text, generated_ids, next_logits, False


def teacher_force_decode_ids_with_kv_refresh_and_trace(
    llm,
    tokenizer,
    forced_ids,
    dense_kv_rows,
    stop_substrings=(),
    max_steps: int = 0,
):
    original_layer_decode = install_decode_trace(llm)
    llm._decode_trace_enabled = True
    llm._decode_trace_limit = int(max_steps)
    try:
        text, generated_ids, next_logits, saw_stop = teacher_force_decode_ids_with_kv_refresh(
            llm=llm,
            tokenizer=tokenizer,
            forced_ids=list(forced_ids),
            dense_kv_rows=dense_kv_rows,
            stop_substrings=stop_substrings,
        )
        trace_steps = list(getattr(llm, "_decode_trace_steps", []))
    finally:
        remove_decode_trace(llm, original_layer_decode)
    return text, generated_ids, next_logits, saw_stop, trace_steps


def summarize_teacher_drift(dense_trace_steps, student_trace_steps):
    step_count = min(len(dense_trace_steps), len(student_trace_steps))
    if step_count <= 0:
        return None
    layer_count = min(len(dense_trace_steps[0]), len(student_trace_steps[0]))
    if layer_count <= 0:
        return None

    l2 = np.zeros((step_count, layer_count), dtype=np.float64)
    rel = np.zeros((step_count, layer_count), dtype=np.float64)
    cos = np.zeros((step_count, layer_count), dtype=np.float64)
    for s in range(step_count):
        for l in range(layer_count):
            dense_vec = dense_trace_steps[s][l].reshape(-1).float()
            student_vec = student_trace_steps[s][l].reshape(-1).float()
            diff = torch.norm(dense_vec - student_vec, p=2).item()
            dense_norm = torch.norm(dense_vec, p=2).item()
            l2[s, l] = float(diff)
            rel[s, l] = float(diff / max(1e-8, dense_norm))
            cos_val = torch.nn.functional.cosine_similarity(
                dense_vec.unsqueeze(0), student_vec.unsqueeze(0), dim=-1
            ).item()
            cos[s, l] = float(cos_val)

    return {
        "num_steps": int(step_count),
        "num_layers": int(layer_count),
        "avg_l2_by_layer": [float(x) for x in l2.mean(axis=0)],
        "first_step_l2_by_layer": [float(x) for x in l2[0]],
        "last_step_l2_by_layer": [float(x) for x in l2[-1]],
        "avg_rel_l2_by_layer": [float(x) for x in rel.mean(axis=0)],
        "avg_cos_by_layer": [float(x) for x in cos.mean(axis=0)],
        "avg_l2_by_step": [float(x) for x in l2.mean(axis=1)],
        "last_layer_l2_by_step": [float(x) for x in l2[:, -1]],
        "first_layer_l2_by_step": [float(x) for x in l2[:, 0]],
        "last_layer_first_step_l2": float(l2[0, -1]),
        "last_layer_last_step_l2": float(l2[-1, -1]),
        "first_layer_first_step_l2": float(l2[0, 0]),
        "first_layer_last_step_l2": float(l2[-1, 0]),
    }


def aggregate_teacher_drift(records):
    records = [r for r in records if r]
    if not records:
        return None
    out = {"num_samples": int(len(records))}
    scalar_keys = [
        "num_steps",
        "num_layers",
        "last_layer_first_step_l2",
        "last_layer_last_step_l2",
        "first_layer_first_step_l2",
        "first_layer_last_step_l2",
    ]
    for key in scalar_keys:
        vals = [float(r[key]) for r in records if key in r]
        if vals:
            out[f"avg_{key}"] = float(np.mean(vals))

    array_keys = [
        "avg_l2_by_layer",
        "avg_rel_l2_by_layer",
        "avg_cos_by_layer",
        "avg_l2_by_step",
        "last_layer_l2_by_step",
        "first_layer_l2_by_step",
    ]
    for key in array_keys:
        arrays = [np.asarray(r[key], dtype=np.float64) for r in records if key in r]
        if arrays and len({a.shape for a in arrays}) == 1:
            out[f"avg_{key}"] = [float(x) for x in np.mean(np.stack(arrays, axis=0), axis=0)]
    return out


def append_prompt_continuation(llm, tokenizer, prompt_text: str, fallback_logits):
    if not prompt_text:
        return fallback_logits
    prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids
    device = llm.layers[0].device
    prompt_ids = prompt_ids.to(device)
    logits = fallback_logits
    for pos in range(prompt_ids.shape[1]):
        logits = llm.decode_forward(inputs_ids=prompt_ids[:, pos:pos + 1])
    return logits


def capture_retrievalattention_answer_start_state(llm):
    if not hasattr(llm, "kv_cache"):
        return None
    kv_cache = llm.kv_cache
    if not hasattr(kv_cache, "_decode_cuda_prev_seed_ids"):
        return None
    return {
        "dynamic_start": int(getattr(kv_cache, "dynamic_start", 0)),
        "dynamic_end": int(getattr(kv_cache, "dynamic_end", 0)),
        "growing_static_dynamic_end": int(getattr(kv_cache, "growing_static_dynamic_end", -1)),
        "prev_seed_ids": [
            None if tensor is None else tensor.detach().clone()
            for tensor in kv_cache._decode_cuda_prev_seed_ids
        ],
        "prev_seed_counts": [
            None if tensor is None else tensor.detach().clone()
            for tensor in kv_cache._decode_cuda_prev_seed_counts
        ],
        "online_graph_overlay": copy.deepcopy(getattr(kv_cache, "_online_graph_overlay", None)),
    }


def apply_retrievalattention_partition_state(llm, state):
    if state is None or not hasattr(llm, "kv_cache"):
        return
    kv_cache = llm.kv_cache
    if not bool(getattr(kv_cache, "growing_static_suffix", False)):
        return
    if "dynamic_start" not in state or "dynamic_end" not in state:
        return
    total_tokens = int(kv_cache._decode_token_limit()) if hasattr(kv_cache, "_decode_token_limit") else int(getattr(kv_cache, "context", 0))
    dynamic_start = max(0, min(int(state["dynamic_start"]), total_tokens))
    dynamic_end = max(dynamic_start, min(int(state["dynamic_end"]), total_tokens))
    kv_cache.dynamic_start = int(dynamic_start)
    kv_cache.dynamic_end = int(dynamic_end)
    kv_cache.growing_static_dynamic_end = int(dynamic_end)
    if hasattr(kv_cache, "_rebuild_static_index_set"):
        kv_cache._rebuild_static_index_set(total_tokens)


def apply_retrievalattention_answer_start_state(
    llm,
    state,
    import_prev_seeds: bool,
    import_overlay: bool,
):
    if state is None or not hasattr(llm, "kv_cache"):
        return
    kv_cache = llm.kv_cache
    if import_prev_seeds and hasattr(kv_cache, "_decode_cuda_prev_seed_ids"):
        src_ids = state.get("prev_seed_ids") or []
        src_counts = state.get("prev_seed_counts") or []
        for ldx, src in enumerate(src_ids):
            dst = kv_cache._decode_cuda_prev_seed_ids[ldx]
            if src is None or dst is None:
                continue
            dst.copy_(src.to(device=dst.device, dtype=dst.dtype))
        for ldx, src in enumerate(src_counts):
            dst = kv_cache._decode_cuda_prev_seed_counts[ldx]
            if src is None or dst is None:
                continue
            dst.copy_(src.to(device=dst.device, dtype=dst.dtype))
    if import_overlay and hasattr(kv_cache, "_online_graph_overlay"):
        kv_cache._online_graph_overlay = copy.deepcopy(state.get("online_graph_overlay"))
        kv_cache._decode_cuda_overlay_graph_cache = [[None for _ in range(kv_cache.kv_head)] for _ in range(kv_cache.layer_num)]
        kv_cache._decode_cuda_overlay_graph_device_cache = [[None for _ in range(kv_cache.kv_head)] for _ in range(kv_cache.layer_num)]


def clear_retrievalattention_answer_start_state(
    llm,
    clear_prev_seeds: bool,
    clear_overlay: bool,
):
    if not hasattr(llm, "kv_cache"):
        return
    kv_cache = llm.kv_cache
    if clear_prev_seeds and hasattr(kv_cache, "_decode_cuda_prev_seed_ids"):
        for ldx in range(len(kv_cache._decode_cuda_prev_seed_ids)):
            ids_t = kv_cache._decode_cuda_prev_seed_ids[ldx]
            counts_t = kv_cache._decode_cuda_prev_seed_counts[ldx]
            if ids_t is not None:
                ids_t.fill_(-1)
            if counts_t is not None:
                counts_t.zero_()
    if clear_overlay and hasattr(kv_cache, "_online_graph_overlay"):
        kv_cache._online_graph_overlay = [[{} for _ in range(kv_cache.kv_head)] for _ in range(kv_cache.layer_num)]
        kv_cache._decode_cuda_overlay_graph_cache = [[None for _ in range(kv_cache.kv_head)] for _ in range(kv_cache.layer_num)]
        kv_cache._decode_cuda_overlay_graph_device_cache = [[None for _ in range(kv_cache.kv_head)] for _ in range(kv_cache.layer_num)]


def _graph_row_neighbors(graph, row_idx: int):
    if graph is None or len(graph) < 2:
        return ()
    offsets = np.asarray(graph[0], dtype=np.int64)
    neighbors = np.asarray(graph[1], dtype=np.int32)
    row = int(row_idx)
    if row < 0 or row + 1 >= int(offsets.shape[0]):
        return ()
    start = int(offsets[row])
    end = int(offsets[row + 1])
    if end <= start:
        return ()
    return tuple(int(x) for x in neighbors[start:end].tolist())


def capture_retrievalattention_graph_snapshot(llm, token_positions):
    if not hasattr(llm, "kv_cache"):
        return None
    kv_cache = llm.kv_cache
    if not hasattr(kv_cache, "graphs"):
        return None
    token_positions = [int(tok) for tok in token_positions if int(tok) >= 0]
    snapshot = {
        "token_positions": list(token_positions),
        "rows": [],
        "layer_num": int(getattr(kv_cache, "layer_num", 0)),
        "kv_head": int(getattr(kv_cache, "kv_head", 0)),
    }
    overlay_all = getattr(kv_cache, "_online_graph_overlay", None)
    for ldx in range(int(snapshot["layer_num"])):
        layer_rows = []
        for hdx in range(int(snapshot["kv_head"])):
            graph = kv_cache.graphs[ldx][hdx]
            overlay = overlay_all[ldx][hdx] if overlay_all is not None else {}
            row_map = {}
            for tok in token_positions:
                merged = []
                seen = set()
                for nb in _graph_row_neighbors(graph, tok):
                    nb_i = int(nb)
                    if nb_i < 0 or nb_i == tok or nb_i in seen:
                        continue
                    seen.add(nb_i)
                    merged.append(nb_i)
                for nb in overlay.get(int(tok), ()):
                    nb_i = int(nb)
                    if nb_i < 0 or nb_i == tok or nb_i in seen:
                        continue
                    seen.add(nb_i)
                    merged.append(nb_i)
                row_map[int(tok)] = merged
            layer_rows.append(row_map)
        snapshot["rows"].append(layer_rows)
    return snapshot


def summarize_retrievalattention_graph_overlap(online_snapshot, replay_snapshot):
    if online_snapshot is None or replay_snapshot is None:
        return None
    token_positions = [int(tok) for tok in online_snapshot.get("token_positions", [])]
    layer_num = min(int(online_snapshot.get("layer_num", 0)), int(replay_snapshot.get("layer_num", 0)))
    kv_head = min(int(online_snapshot.get("kv_head", 0)), int(replay_snapshot.get("kv_head", 0)))
    if layer_num <= 0 or kv_head <= 0 or not token_positions:
        return None

    online_deg_all = []
    replay_deg_all = []
    jaccard_all = []
    recall_replay_by_online = []
    replay_nonempty_flags = []
    online_nonempty_flags = []
    replay_nonempty_online_empty_flags = []
    per_layer = []
    samples = []

    for ldx in range(layer_num):
        layer_online_deg = []
        layer_replay_deg = []
        layer_jaccard = []
        layer_recall = []
        for hdx in range(kv_head):
            online_rows = online_snapshot["rows"][ldx][hdx]
            replay_rows = replay_snapshot["rows"][ldx][hdx]
            for tok in token_positions:
                online_set = set(int(x) for x in online_rows.get(int(tok), ()))
                replay_set = set(int(x) for x in replay_rows.get(int(tok), ()))
                inter = online_set & replay_set
                union = online_set | replay_set
                online_deg = len(online_set)
                replay_deg = len(replay_set)
                online_deg_all.append(float(online_deg))
                replay_deg_all.append(float(replay_deg))
                online_nonempty = 1.0 if online_deg > 0 else 0.0
                replay_nonempty = 1.0 if replay_deg > 0 else 0.0
                online_nonempty_flags.append(online_nonempty)
                replay_nonempty_flags.append(replay_nonempty)
                replay_nonempty_online_empty_flags.append(1.0 if replay_deg > 0 and online_deg == 0 else 0.0)
                if union:
                    j = float(len(inter) / len(union))
                    jaccard_all.append(j)
                    layer_jaccard.append(j)
                if replay_set:
                    r = float(len(inter) / len(replay_set))
                    recall_replay_by_online.append(r)
                    layer_recall.append(r)
                layer_online_deg.append(float(online_deg))
                layer_replay_deg.append(float(replay_deg))
                if replay_deg >= 4 and len(samples) < 8:
                    samples.append(
                        {
                            "layer": int(ldx),
                            "head": int(hdx),
                            "token_pos": int(tok),
                            "online_degree": int(online_deg),
                            "replay_degree": int(replay_deg),
                            "intersection": int(len(inter)),
                            "jaccard": float(len(inter) / len(union)) if union else None,
                            "replay_recall": float(len(inter) / len(replay_set)) if replay_set else None,
                            "online_neighbors": sorted(int(x) for x in online_set)[:16],
                            "replay_neighbors": sorted(int(x) for x in replay_set)[:16],
                        }
                    )
        per_layer.append(
            {
                "layer": int(ldx),
                "avg_online_degree": float(np.mean(layer_online_deg)) if layer_online_deg else 0.0,
                "avg_replay_degree": float(np.mean(layer_replay_deg)) if layer_replay_deg else 0.0,
                "avg_jaccard": float(np.mean(layer_jaccard)) if layer_jaccard else 0.0,
                "avg_replay_recall": float(np.mean(layer_recall)) if layer_recall else 0.0,
            }
        )

    return {
        "generated_token_count": int(len(token_positions)),
        "layer_num": int(layer_num),
        "kv_head": int(kv_head),
        "avg_online_degree": float(np.mean(online_deg_all)) if online_deg_all else 0.0,
        "avg_replay_degree": float(np.mean(replay_deg_all)) if replay_deg_all else 0.0,
        "online_nonempty_rate": float(np.mean(online_nonempty_flags)) if online_nonempty_flags else 0.0,
        "replay_nonempty_rate": float(np.mean(replay_nonempty_flags)) if replay_nonempty_flags else 0.0,
        "replay_nonempty_online_empty_rate": float(np.mean(replay_nonempty_online_empty_flags)) if replay_nonempty_online_empty_flags else 0.0,
        "avg_jaccard": float(np.mean(jaccard_all)) if jaccard_all else 0.0,
        "avg_replay_recall": float(np.mean(recall_replay_by_online)) if recall_replay_by_online else 0.0,
        "per_layer": per_layer,
        "sample_rows": samples,
    }


def summarize_token_ranges(token_ids):
    ids = sorted(set(int(tok) for tok in token_ids))
    if not ids:
        return []
    ranges = []
    start = prev = ids[0]
    for tok in ids[1:]:
        if tok == prev + 1:
            prev = tok
            continue
        ranges.append([int(start), int(prev)])
        start = prev = tok
    ranges.append([int(start), int(prev)])
    return ranges


def summarize_retrievalattention_partition_state(llm, label: str = ""):
    if not hasattr(llm, "kv_cache"):
        return None
    kv_cache = llm.kv_cache
    total_tokens = int(kv_cache._decode_token_limit()) if hasattr(kv_cache, "_decode_token_limit") else int(getattr(kv_cache, "context", 0))
    dynamic_start = int(getattr(kv_cache, "dynamic_start", 0))
    dynamic_end = int(getattr(kv_cache, "dynamic_end", 0))
    try:
        static_ids = list(kv_cache._get_fullgpu_static_token_ids())
    except Exception:
        static_set = getattr(kv_cache, "static_index_set", set())
        static_ids = sorted(int(tok) for tok in static_set) if static_set is not None else []
    static_ids = [int(tok) for tok in static_ids if 0 <= int(tok) < total_tokens]
    static_count = int(len(static_ids))
    prefix_static = len([tok for tok in static_ids if int(tok) < dynamic_start])
    suffix_static = len([tok for tok in static_ids if int(tok) >= dynamic_end])
    return {
        "label": str(label),
        "input_length": int(getattr(kv_cache, "input_length", 0)),
        "decode_pos": int(getattr(kv_cache, "decode_pos", 0)),
        "context": int(getattr(kv_cache, "context", 0)),
        "total_tokens": int(total_tokens),
        "dynamic_start": int(dynamic_start),
        "dynamic_end": int(dynamic_end),
        "dynamic_span": int(max(0, dynamic_end - dynamic_start)),
        "static_count": int(static_count),
        "static_prefix_count": int(prefix_static),
        "static_suffix_count": int(suffix_static),
        "static_ranges": summarize_token_ranges(static_ids),
        "static_ids_head": static_ids[:64],
        "static_ids_tail": static_ids[-64:] if len(static_ids) > 64 else static_ids,
        "growing_static_suffix": bool(getattr(kv_cache, "growing_static_suffix", False)),
        "growing_static_dynamic_end": int(getattr(kv_cache, "growing_static_dynamic_end", -1)),
        "static_pattern_start": int(getattr(kv_cache, "static_pattern_start", -1)),
        "static_pattern_end": int(getattr(kv_cache, "static_pattern_end", -1)),
    }


def _last_token_logits(logits):
    logits_view = logits
    if logits_view.dim() == 3:
        logits_view = logits_view[:, -1, :]
    if logits_view.dim() != 2:
        raise RuntimeError(f"Unexpected logits shape: {tuple(logits_view.shape)}")
    return logits_view[0].detach().float()


def summarize_answer_start_logits(logits, tokenizer, constrained_token_map=None, top_k: int = 16):
    row = _last_token_logits(logits).cpu()
    top_count = min(int(top_k), int(row.numel()))
    vals_t, ids_t = torch.topk(row, k=top_count)
    top_tokens = []
    for tok_id, score in zip(ids_t.tolist(), vals_t.tolist()):
        top_tokens.append(
            {
                "token_id": int(tok_id),
                "text": tokenizer.decode([int(tok_id)]),
                "logit": float(score),
            }
        )
    codebook = []
    if constrained_token_map:
        for tok_id, word in constrained_token_map:
            score = float(row[int(tok_id)].item())
            rank = int((row > score).sum().item()) + 1
            codebook.append(
                {
                    "word": str(word),
                    "token_id": int(tok_id),
                    "logit": score,
                    "rank": rank,
                }
            )
    return {"top_tokens": top_tokens, "codebook": codebook}


def choose_state_equiv_positions(total_tokens: int, partition_state, token_ranges, entry_spans, max_positions: int = 128):
    total_tokens = int(total_tokens)
    positions = set()
    def add(pos):
        pos = int(pos)
        if 0 <= pos < total_tokens:
            positions.add(pos)

    for pos in [0, 1, 2, total_tokens - 3, total_tokens - 2, total_tokens - 1]:
        add(pos)
    if partition_state:
        ds = int(partition_state.get("dynamic_start", 0))
        de = int(partition_state.get("dynamic_end", 0))
        for pos in [ds - 2, ds - 1, ds, ds + 1, de - 2, de - 1, de, de + 1]:
            add(pos)
    for start, end in (token_ranges or {}).values():
        start = int(start)
        end = int(end)
        for pos in [start, start + 1, max(start, end - 2), end - 1]:
            add(pos)
    for span in (entry_spans or {}).values():
        for pos in span:
            add(pos)
    out = sorted(positions)
    if len(out) > int(max_positions):
        keep = set(out[:16] + out[-16:])
        middle = out[16:-16]
        stride = max(1, len(middle) // max(1, int(max_positions) - len(keep)))
        keep.update(middle[::stride])
        out = sorted(keep)[: int(max_positions)]
    return out


def capture_kv_state_signature(llm, positions):
    if not hasattr(llm, "kv_cache"):
        return None
    kv_cache = llm.kv_cache
    if not hasattr(kv_cache, "cpu_keys") or not hasattr(kv_cache, "cpu_values"):
        return None
    positions = [int(pos) for pos in positions]
    if not positions:
        return None
    pos_t = torch.as_tensor(positions, dtype=torch.long)
    layers = []
    for ldx in range(int(getattr(kv_cache, "layer_num", len(kv_cache.cpu_keys)))):
        key_src = kv_cache.cpu_keys[ldx]
        value_src = kv_cache.cpu_values[ldx]
        max_pos = int(key_src.shape[1])
        valid_positions = [pos for pos in positions if 0 <= pos < max_pos]
        if not valid_positions:
            layers.append(None)
            continue
        valid_t = torch.as_tensor(valid_positions, dtype=torch.long)
        layers.append(
            {
                "positions": valid_positions,
                "keys": torch.index_select(key_src, 1, valid_t).detach().float().cpu().clone(),
                "values": torch.index_select(value_src, 1, valid_t).detach().float().cpu().clone(),
            }
        )
    return {"positions": positions, "layers": layers}


def compare_kv_state_signatures(reference, other):
    if reference is None or other is None:
        return None
    per_layer = []
    key_l2_vals = []
    value_l2_vals = []
    key_cos_vals = []
    value_cos_vals = []
    for ldx, (ref_layer, other_layer) in enumerate(zip(reference.get("layers", []), other.get("layers", []))):
        if ref_layer is None or other_layer is None:
            continue
        if ref_layer.get("positions") != other_layer.get("positions"):
            continue
        ref_k = ref_layer["keys"].reshape(-1).float()
        other_k = other_layer["keys"].reshape(-1).float()
        ref_v = ref_layer["values"].reshape(-1).float()
        other_v = other_layer["values"].reshape(-1).float()
        key_l2 = float(torch.linalg.vector_norm(ref_k - other_k).item())
        value_l2 = float(torch.linalg.vector_norm(ref_v - other_v).item())
        key_rel = float(key_l2 / max(1e-8, float(torch.linalg.vector_norm(ref_k).item())))
        value_rel = float(value_l2 / max(1e-8, float(torch.linalg.vector_norm(ref_v).item())))
        key_cos = float(torch.nn.functional.cosine_similarity(ref_k.unsqueeze(0), other_k.unsqueeze(0), dim=-1).item())
        value_cos = float(torch.nn.functional.cosine_similarity(ref_v.unsqueeze(0), other_v.unsqueeze(0), dim=-1).item())
        row = {
            "layer": int(ldx),
            "position_count": int(len(ref_layer.get("positions", []))),
            "key_l2": key_l2,
            "key_rel_l2": key_rel,
            "key_cos": key_cos,
            "value_l2": value_l2,
            "value_rel_l2": value_rel,
            "value_cos": value_cos,
        }
        per_layer.append(row)
        key_l2_vals.append(key_l2)
        value_l2_vals.append(value_l2)
        key_cos_vals.append(key_cos)
        value_cos_vals.append(value_cos)
    if not per_layer:
        return None
    return {
        "position_count": int(len(reference.get("positions", []))),
        "layer_count": int(len(per_layer)),
        "avg_key_l2": float(np.mean(key_l2_vals)),
        "max_key_l2": float(np.max(key_l2_vals)),
        "avg_key_cos": float(np.mean(key_cos_vals)),
        "min_key_cos": float(np.min(key_cos_vals)),
        "avg_value_l2": float(np.mean(value_l2_vals)),
        "max_value_l2": float(np.max(value_l2_vals)),
        "avg_value_cos": float(np.mean(value_cos_vals)),
        "min_value_cos": float(np.min(value_cos_vals)),
        "per_layer": per_layer,
    }


def compare_attention_debug_records(reference_records, other_records):
    if not reference_records or not other_records:
        return None
    ref = reference_records[0]
    other = other_records[0]
    ref_final = set(int(tok) for tok in ref.get("final_ids", []))
    other_final = set(int(tok) for tok in other.get("final_ids", []))
    out = {
        "ref_dynamic_count": int(ref.get("dynamic_count", 0)),
        "other_dynamic_count": int(other.get("dynamic_count", 0)),
        "ref_final_count": int(ref.get("final_count", 0)),
        "other_final_count": int(other.get("final_count", 0)),
        "final_jaccard": float(len(ref_final & other_final) / float(max(1, len(ref_final | other_final)))),
        "missing_from_other": sorted(int(tok) for tok in (ref_final - other_final))[:64],
        "extra_in_other": sorted(int(tok) for tok in (other_final - ref_final))[:64],
    }
    ref_vec = ref.get("sparse_out")
    other_vec = other.get("sparse_out")
    if ref_vec is not None and other_vec is not None and len(ref_vec) == len(other_vec):
        ref_t = torch.as_tensor(ref_vec, dtype=torch.float32)
        other_t = torch.as_tensor(other_vec, dtype=torch.float32)
        diff = ref_t - other_t
        l2 = float(torch.linalg.vector_norm(diff).item())
        ref_norm = float(torch.linalg.vector_norm(ref_t).item())
        out.update(
            {
                "sparse_out_l2": l2,
                "sparse_out_rel_l2": float(l2 / max(1e-8, ref_norm)),
                "sparse_out_cos": float(torch.nn.functional.cosine_similarity(ref_t.unsqueeze(0), other_t.unsqueeze(0), dim=-1).item()),
                "sparse_out_max_abs": float(torch.max(torch.abs(diff)).item()),
            }
        )
    return out


def capture_first_answer_attention_debug(
    llm,
    tokenizer,
    answer_seed_logits,
    answer_prefix_scaffold: bool,
    oracle_retrieve_enable: bool,
):
    if not hasattr(llm, "kv_cache"):
        return []
    kv_cache = llm.kv_cache
    setattr(kv_cache, "attention_input_debug_enable", True)
    setattr(kv_cache, "attention_input_debug_answer_start_pos", int(kv_cache.decode_pos))
    setattr(kv_cache, "attention_input_debug_records", [])
    setattr(kv_cache, "oracle_retrieval_enable", bool(oracle_retrieve_enable))
    setattr(kv_cache, "oracle_debug_enable", False)
    setattr(kv_cache, "oracle_compare_enable", False)
    setattr(kv_cache, "oracle_answer_start_pos", int(kv_cache.decode_pos))
    device = llm.layers[0].device
    if answer_prefix_scaffold:
        prefix_ids = tokenizer("ANSWER 1: ", return_tensors="pt", add_special_tokens=False).input_ids[0]
        token_id = int(prefix_ids[0].item())
        current_token = torch.tensor([[token_id]], dtype=torch.long, device=device)
    else:
        current_token = llm.sampling(answer_seed_logits, do_sample=False).to(device)
    _ = llm.decode_forward(inputs_ids=current_token)
    torch.cuda.synchronize()
    records = list(getattr(kv_cache, "attention_input_debug_records", []))
    setattr(kv_cache, "attention_input_debug_enable", False)
    setattr(kv_cache, "attention_input_debug_answer_start_pos", None)
    setattr(kv_cache, "attention_input_debug_records", [])
    setattr(kv_cache, "oracle_retrieval_enable", False)
    setattr(kv_cache, "oracle_debug_enable", False)
    setattr(kv_cache, "oracle_compare_enable", False)
    setattr(kv_cache, "oracle_answer_start_pos", None)
    return records


def capture_first_color_decision_debug(
    llm,
    tokenizer,
    answer_seed_logits,
    answer_prefix_scaffold: bool,
    oracle_retrieve_enable: bool,
    constrained_token_map=None,
):
    if not hasattr(llm, "kv_cache"):
        return {
            "attention_records": [],
            "decision_logits": summarize_answer_start_logits(
                answer_seed_logits,
                tokenizer=tokenizer,
                constrained_token_map=constrained_token_map,
            ),
            "prefix_text": "",
            "prefix_token_ids": [],
        }
    kv_cache = llm.kv_cache
    device = llm.layers[0].device

    setattr(kv_cache, "oracle_retrieval_enable", bool(oracle_retrieve_enable))
    setattr(kv_cache, "oracle_debug_enable", False)
    setattr(kv_cache, "oracle_compare_enable", False)

    prefix_text = "ANSWER 1: " if answer_prefix_scaffold else ""
    prefix_token_ids = []
    next_logits = answer_seed_logits
    if answer_prefix_scaffold:
        prefix_ids = tokenizer(prefix_text, return_tensors="pt", add_special_tokens=False).input_ids[0]
        prefix_token_ids = [int(tok) for tok in prefix_ids.tolist()]
        for idx, tok_id in enumerate(prefix_token_ids):
            is_last = idx == len(prefix_token_ids) - 1
            if is_last:
                setattr(kv_cache, "attention_input_debug_enable", True)
                setattr(kv_cache, "attention_input_debug_answer_start_pos", int(kv_cache.decode_pos))
                setattr(kv_cache, "attention_input_debug_records", [])
                setattr(kv_cache, "oracle_answer_start_pos", int(kv_cache.decode_pos))
            else:
                setattr(kv_cache, "attention_input_debug_enable", False)
                setattr(kv_cache, "attention_input_debug_answer_start_pos", None)
            current_token = torch.tensor([[int(tok_id)]], dtype=torch.long, device=device)
            next_logits = llm.decode_forward(inputs_ids=current_token)
    else:
        setattr(kv_cache, "attention_input_debug_enable", True)
        setattr(kv_cache, "attention_input_debug_answer_start_pos", int(kv_cache.decode_pos))
        setattr(kv_cache, "attention_input_debug_records", [])
        setattr(kv_cache, "oracle_answer_start_pos", int(kv_cache.decode_pos))
        current_token = llm.sampling(answer_seed_logits, do_sample=False).to(device)
        next_logits = llm.decode_forward(inputs_ids=current_token)

    torch.cuda.synchronize()
    records = list(getattr(kv_cache, "attention_input_debug_records", []))
    decision_logits = summarize_answer_start_logits(
        next_logits,
        tokenizer=tokenizer,
        constrained_token_map=constrained_token_map,
    )
    chosen_codebook = None
    codebook_rows = decision_logits.get("codebook") or []
    if codebook_rows:
        chosen_codebook = min(codebook_rows, key=lambda row: int(row.get("rank", 1 << 30)))

    setattr(kv_cache, "attention_input_debug_enable", False)
    setattr(kv_cache, "attention_input_debug_answer_start_pos", None)
    setattr(kv_cache, "attention_input_debug_records", [])
    setattr(kv_cache, "oracle_retrieval_enable", False)
    setattr(kv_cache, "oracle_debug_enable", False)
    setattr(kv_cache, "oracle_compare_enable", False)
    setattr(kv_cache, "oracle_answer_start_pos", None)
    return {
        "attention_records": records,
        "decision_logits": decision_logits,
        "chosen_codebook": chosen_codebook,
        "prefix_text": prefix_text,
        "prefix_token_ids": prefix_token_ids,
    }


def summarize_state_equiv_mode(
    label,
    llm,
    tokenizer,
    answer_seed_logits,
    constrained_token_map,
    partition_state,
    kv_signature,
    reference_kv_signature,
    reference_attention_records,
    attention_records,
    color_decision=None,
):
    return {
        "label": str(label),
        "partition_state": partition_state,
        "logits": summarize_answer_start_logits(
            answer_seed_logits,
            tokenizer=tokenizer,
            constrained_token_map=constrained_token_map,
        ),
        "color_decision": color_decision,
        "kv_compare_to_online": compare_kv_state_signatures(reference_kv_signature, kv_signature),
        "attention_input_debug": attention_records,
        "attention_compare_to_online": compare_attention_debug_records(reference_attention_records, attention_records),
    }


def replay_answers_from_prefilled_ledger(
    llm,
    tokenizer,
    ledger_prompt: str,
    ledger_text: str,
    question_prompt: str,
    attn_type: str,
    retrieval_budget: float,
    estimation_budget: float,
    token_budget_override: int,
    max_answer_new_tokens: int,
    num_queries: int,
    answer_prefix_scaffold: bool,
    answer_constrained_token_map=None,
    question_via_decode: bool = False,
    online_ra_state=None,
    import_online_prev_seeds: bool = False,
    import_online_overlay: bool = False,
    clear_prev_seeds: bool = False,
    clear_overlay: bool = False,
    oracle_retrieve_enable: bool = False,
    oracle_compare_enable: bool = False,
    graph_compare_online_snapshot=None,
    state_partition_diag: bool = False,
):
    replay_prompt = ledger_prompt + ledger_text
    if not question_via_decode:
        replay_prompt = replay_prompt + question_prompt
    replay_prompt_tokens = int(
        len(tokenizer(replay_prompt, return_tensors="pt").input_ids[0])
    )
    replay_config = generate_config(
        llm.model_name,
        replay_prompt_tokens,
        attn_type,
        retrieval_budget=retrieval_budget,
        estimation_budget=estimation_budget,
        token_budget_override=token_budget_override,
    )
    inputs = tokenizer([replay_prompt], return_tensors="pt", padding=True)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    llm.attention_type = str(attn_type)
    llm.batch_size = int(input_ids.shape[0])
    llm.input_length = int(input_ids.shape[1])
    llm.max_new_length = int(max_answer_new_tokens)
    if question_via_decode:
        question_tokens = tokenizer(question_prompt, return_tensors="pt", add_special_tokens=False).input_ids[0]
        llm.max_new_length = int(max_answer_new_tokens + int(question_tokens.shape[0]))
    assert llm.input_length + llm.max_new_length <= llm.max_length, (
        f"input_length({llm.input_length}) + max_new_length({llm.max_new_length}) exceeds max_length({llm.max_length})"
    )
    valid_start = attention_mask.shape[1] - torch.sum(attention_mask, dim=-1).detach().cpu().numpy()
    llm.prefill_bsz = 1
    llm.prefill_method = "full"
    llm.init_kv_cache(valid_start, replay_config)
    device = llm.layers[0].device
    prefill_start = time.time()
    replay_logits = llm.prefill_forward(inputs_ids=input_ids.to(device))
    llm.move()
    torch.cuda.synchronize()
    prefill_sec = float(time.time() - prefill_start)
    apply_retrievalattention_partition_state(llm=llm, state=online_ra_state)
    if question_via_decode:
        replay_logits = append_prompt_continuation(
            llm=llm,
            tokenizer=tokenizer,
            prompt_text=question_prompt,
            fallback_logits=replay_logits,
        )
    replay_partition_state = (
        summarize_retrievalattention_partition_state(llm, label="replay_answer_start")
        if state_partition_diag and attn_type == "RetrievalAttention"
        else None
    )
    apply_retrievalattention_answer_start_state(
        llm=llm,
        state=online_ra_state,
        import_prev_seeds=bool(import_online_prev_seeds),
        import_overlay=bool(import_online_overlay),
    )
    clear_retrievalattention_answer_start_state(
        llm=llm,
        clear_prev_seeds=bool(clear_prev_seeds),
        clear_overlay=bool(clear_overlay),
    )
    graph_compare_summary = None
    if (
        graph_compare_online_snapshot is not None
        and attn_type == "RetrievalAttention"
    ):
        replay_snapshot = capture_retrievalattention_graph_snapshot(
            llm=llm,
            token_positions=graph_compare_online_snapshot.get("token_positions", []),
        )
        graph_compare_summary = summarize_retrievalattention_graph_overlap(
            online_snapshot=graph_compare_online_snapshot,
            replay_snapshot=replay_snapshot,
        )
    attention_input_debug_records = None
    if (
        state_partition_diag
        and attn_type == "RetrievalAttention"
        and hasattr(llm, "kv_cache")
    ):
        setattr(llm.kv_cache, "attention_input_debug_enable", True)
        setattr(llm.kv_cache, "attention_input_debug_answer_start_pos", int(llm.kv_cache.decode_pos))
        setattr(llm.kv_cache, "attention_input_debug_records", [])
    if (
        attn_type == "RetrievalAttention"
        and hasattr(llm, "kv_cache")
    ):
        setattr(llm.kv_cache, "oracle_retrieval_enable", bool(oracle_retrieve_enable))
        setattr(llm.kv_cache, "oracle_debug_enable", bool(oracle_retrieve_enable))
        setattr(llm.kv_cache, "oracle_compare_enable", bool(oracle_compare_enable))
        setattr(llm.kv_cache, "oracle_answer_start_pos", int(llm.kv_cache.decode_pos))
        setattr(llm.kv_cache, "oracle_debug_records", [])
        setattr(llm.kv_cache, "oracle_compare_records", [])
    decode_start = time.time()
    if answer_prefix_scaffold:
        answer_text, answer_ids, _after_logits, _ = greedy_decode_answers_with_prefix_scaffold(
            llm=llm,
            tokenizer=tokenizer,
            fallback_logits=replay_logits,
            expected_answers=int(num_queries),
            max_new_tokens=max_answer_new_tokens,
            constrained_token_map=answer_constrained_token_map,
        )
    else:
        first_token = llm.sampling(replay_logits, do_sample=False).to(device)
        answer_text, answer_ids, _after_logits, _ = greedy_decode_answers(
            llm=llm,
            tokenizer=tokenizer,
            start_token=first_token,
            max_new_tokens=max_answer_new_tokens,
            expected_answers=int(num_queries),
        )
    torch.cuda.synchronize()
    decode_sec = float(time.time() - decode_start)
    if (
        attn_type == "RetrievalAttention"
        and hasattr(llm, "kv_cache")
    ):
        setattr(llm.kv_cache, "oracle_retrieval_enable", False)
        setattr(llm.kv_cache, "oracle_debug_enable", False)
        setattr(llm.kv_cache, "oracle_compare_enable", False)
        attention_input_debug_records = (
            list(getattr(llm.kv_cache, "attention_input_debug_records", []))
            if state_partition_diag
            else None
        )
        setattr(llm.kv_cache, "attention_input_debug_enable", False)
        setattr(llm.kv_cache, "attention_input_debug_answer_start_pos", None)
        setattr(llm.kv_cache, "attention_input_debug_records", [])
    output_text = replay_prompt + (question_prompt if question_via_decode else "") + answer_text
    return {
        "replay_prompt_tokens": int(replay_prompt_tokens),
        "prefill_sec": float(prefill_sec),
        "decode_sec": float(decode_sec),
        "answer_generated_tokens": int(len(answer_ids)),
        "answer_output": answer_text,
        "output": output_text,
        "question_via_decode": bool(question_via_decode),
        "import_online_prev_seeds": bool(import_online_prev_seeds),
        "import_online_overlay": bool(import_online_overlay),
        "clear_prev_seeds": bool(clear_prev_seeds),
        "clear_overlay": bool(clear_overlay),
        "oracle_retrieve_enable": bool(oracle_retrieve_enable),
        "graph_compare": graph_compare_summary,
        "partition_state": replay_partition_state,
        "attention_input_debug": attention_input_debug_records,
    }


def build_replay_answer_start_state(
    llm,
    tokenizer,
    ledger_prompt: str,
    ledger_text: str,
    question_prompt: str,
    attn_type: str,
    retrieval_budget: float,
    estimation_budget: float,
    token_budget_override: int,
    online_ra_state,
    future_tokens: int,
):
    replay_prompt = ledger_prompt + ledger_text
    replay_prompt_tokens = int(len(tokenizer(replay_prompt, return_tensors="pt").input_ids[0]))
    replay_config = generate_config(
        llm.model_name,
        replay_prompt_tokens,
        attn_type,
        retrieval_budget=retrieval_budget,
        estimation_budget=estimation_budget,
        token_budget_override=token_budget_override,
    )
    _first_token, prefill_sec = init_generation_session(
        llm=llm,
        tokenizer=tokenizer,
        prompt_text=replay_prompt,
        attn_type=attn_type,
        attn_config=replay_config,
        total_future_tokens=int(future_tokens),
    )
    apply_retrievalattention_partition_state(llm=llm, state=online_ra_state)
    question_start = time.time()
    answer_seed_logits = append_prompt_continuation(
        llm=llm,
        tokenizer=tokenizer,
        prompt_text=question_prompt,
        fallback_logits=None,
    )
    torch.cuda.synchronize()
    question_sec = float(time.time() - question_start)
    partition = summarize_retrievalattention_partition_state(llm, label="state_equiv_replay_answer_start")
    return {
        "answer_seed_logits": answer_seed_logits,
        "partition_state": partition,
        "prefill_sec": float(prefill_sec),
        "question_decode_sec": float(question_sec),
        "prompt_tokens": int(replay_prompt_tokens),
    }


def build_teacher_forced_answer_start_state(
    llm,
    tokenizer,
    ledger_prompt: str,
    ledger_ids,
    question_prompt: str,
    attn_type: str,
    retrieval_budget: float,
    estimation_budget: float,
    token_budget_override: int,
    future_tokens: int,
):
    prompt_tokens = int(len(tokenizer(ledger_prompt, return_tensors="pt").input_ids[0]))
    config = generate_config(
        llm.model_name,
        prompt_tokens,
        attn_type,
        retrieval_budget=retrieval_budget,
        estimation_budget=estimation_budget,
        token_budget_override=token_budget_override,
    )
    _first_token, prefill_sec = init_generation_session(
        llm=llm,
        tokenizer=tokenizer,
        prompt_text=ledger_prompt,
        attn_type=attn_type,
        attn_config=config,
        total_future_tokens=int(future_tokens),
    )
    force_start = time.time()
    _ledger_text, forced_ids, after_ledger_logits, saw_end = teacher_force_decode_ids(
        llm=llm,
        tokenizer=tokenizer,
        forced_ids=ledger_ids,
        stop_substrings=("END LEDGER",),
    )
    answer_seed_logits = append_prompt_continuation(
        llm=llm,
        tokenizer=tokenizer,
        prompt_text=question_prompt,
        fallback_logits=after_ledger_logits,
    )
    torch.cuda.synchronize()
    force_sec = float(time.time() - force_start)
    partition = summarize_retrievalattention_partition_state(llm, label="state_equiv_teacher_forced_answer_start")
    return {
        "answer_seed_logits": answer_seed_logits,
        "partition_state": partition,
        "prefill_sec": float(prefill_sec),
        "decode_to_answer_start_sec": float(force_sec),
        "prompt_tokens": int(prompt_tokens),
        "forced_ledger_tokens": int(len(forced_ids)),
        "saw_end_ledger": bool(saw_end),
    }


def run_state_equiv_diagnostic(
    llm,
    tokenizer,
    sample,
    ledger_text: str,
    ledger_ids,
    online_ra_state,
    online_reference_kv,
    online_attention_records,
    online_partition_state,
    online_answer_seed_logits,
    state_positions,
    constrained_token_map,
    args,
    replay_future_tokens: int,
    teacher_future_tokens: int,
    answer_prefix_scaffold: bool,
):
    oracle_retrieve_enable = os.environ.get("RETRIEVALATTN_ORACLE_RETRIEVE", "0") == "1"
    out = {
        "positions": [int(pos) for pos in state_positions],
        "color_reference_source": "teacher_forced_decode_color_decision",
        "online": {
            "partition_state": online_partition_state,
            "logits": summarize_answer_start_logits(
                online_answer_seed_logits,
                tokenizer=tokenizer,
                constrained_token_map=constrained_token_map,
            ),
            "attention_input_debug": online_attention_records,
        },
    }

    teacher_state = build_teacher_forced_answer_start_state(
        llm=llm,
        tokenizer=tokenizer,
        ledger_prompt=sample["ledger_prompt"],
        ledger_ids=ledger_ids,
        question_prompt=sample["question_prompt"],
        attn_type=args.attn_type,
        retrieval_budget=args.retrieval_budget,
        estimation_budget=args.estimation_budget,
        token_budget_override=args.token_budget_override,
        future_tokens=teacher_future_tokens,
    )
    teacher_kv = capture_kv_state_signature(llm, state_positions)
    teacher_color = capture_first_color_decision_debug(
        llm=llm,
        tokenizer=tokenizer,
        answer_seed_logits=teacher_state["answer_seed_logits"],
        answer_prefix_scaffold=bool(answer_prefix_scaffold),
        oracle_retrieve_enable=bool(oracle_retrieve_enable),
        constrained_token_map=constrained_token_map,
    )
    color_reference_attention_records = teacher_color.get("attention_records", [])
    teacher_summary = summarize_state_equiv_mode(
        label="teacher_forced_decode",
        llm=llm,
        tokenizer=tokenizer,
        answer_seed_logits=teacher_state["answer_seed_logits"],
        constrained_token_map=constrained_token_map,
        partition_state=teacher_state["partition_state"],
        kv_signature=teacher_kv,
        reference_kv_signature=online_reference_kv,
        reference_attention_records=color_reference_attention_records,
        attention_records=color_reference_attention_records,
        color_decision=teacher_color,
    )
    teacher_summary.update(
        {
            "prefill_sec": float(teacher_state["prefill_sec"]),
            "decode_to_answer_start_sec": float(teacher_state["decode_to_answer_start_sec"]),
            "prompt_tokens": int(teacher_state["prompt_tokens"]),
            "forced_ledger_tokens": int(teacher_state["forced_ledger_tokens"]),
            "saw_end_ledger": bool(teacher_state["saw_end_ledger"]),
        }
    )
    out["teacher_forced_decode"] = teacher_summary

    replay_state = build_replay_answer_start_state(
        llm=llm,
        tokenizer=tokenizer,
        ledger_prompt=sample["ledger_prompt"],
        ledger_text=ledger_text,
        question_prompt=sample["question_prompt"],
        attn_type=args.attn_type,
        retrieval_budget=args.retrieval_budget,
        estimation_budget=args.estimation_budget,
        token_budget_override=args.token_budget_override,
        online_ra_state=online_ra_state,
        future_tokens=replay_future_tokens,
    )
    replay_kv = capture_kv_state_signature(llm, state_positions)
    replay_color = capture_first_color_decision_debug(
        llm=llm,
        tokenizer=tokenizer,
        answer_seed_logits=replay_state["answer_seed_logits"],
        answer_prefix_scaffold=bool(answer_prefix_scaffold),
        oracle_retrieve_enable=bool(oracle_retrieve_enable),
        constrained_token_map=constrained_token_map,
    )
    replay_summary = summarize_state_equiv_mode(
        label="replay_prefill",
        llm=llm,
        tokenizer=tokenizer,
        answer_seed_logits=replay_state["answer_seed_logits"],
        constrained_token_map=constrained_token_map,
        partition_state=replay_state["partition_state"],
        kv_signature=replay_kv,
        reference_kv_signature=online_reference_kv,
        reference_attention_records=color_reference_attention_records,
        attention_records=replay_color.get("attention_records", []),
        color_decision=replay_color,
    )
    replay_summary.update(
        {
            "prefill_sec": float(replay_state["prefill_sec"]),
            "question_decode_sec": float(replay_state["question_decode_sec"]),
            "prompt_tokens": int(replay_state["prompt_tokens"]),
        }
    )
    out["replay_prefill"] = replay_summary
    return out


def release_model(llm):
    if llm is None:
        return
    try:
        del llm.kv_cache
    except Exception:
        pass
    del llm
    import gc
    gc.collect()
    torch.cuda.empty_cache()


def bucket_name(pos: int, num_entries: int):
    frac = float(pos) / float(max(1, num_entries))
    if frac <= 0.34:
        return "far"
    if frac <= 0.67:
        return "mid"
    return "near"


def main():
    args = parse_args()
    set_seed(args.seed)
    if int(args.batch_size) != 1:
        raise RuntimeError("generated_memory_eval currently supports only batch_size=1.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    constrained_token_map = build_codebook_token_map(tokenizer) if args.answer_constrained_codebook else None

    static_start = int(os.environ.get("RETRIEVALATTN_STATIC_PATTERN_START", "128"))
    static_end = int(os.environ.get("RETRIEVALATTN_STATIC_PATTERN_END", "512"))
    min_prompt_tokens = int(args.min_prompt_tokens)
    if min_prompt_tokens <= 0:
        min_prompt_tokens = int(static_start + static_end + 64)

    rng = random.Random(args.seed)
    samples = []
    max_prompt_len = 0
    max_question_len = 0
    for sample_idx in range(int(args.num_samples)):
        query_positions = choose_query_positions(args.num_entries, args.num_queries, rng)
        prompt = build_ledger_prompt(
            sample_idx=sample_idx,
            num_entries=int(args.num_entries),
            filler_repeats=int(args.prefill_filler_repeats),
        )
        prompt, prompt_len, used_filler_repeats = ensure_min_prompt_tokens(
            tokenizer=tokenizer,
            prompt=prompt,
            sample_idx=sample_idx,
            num_entries=int(args.num_entries),
            filler_repeats=int(args.prefill_filler_repeats),
            min_prompt_tokens=int(min_prompt_tokens),
        )
        question_prompt = build_question_prompt(
            query_positions,
            answer_prefix_scaffold=bool(args.answer_prefix_scaffold),
        )
        question_len = len(tokenizer(question_prompt, return_tensors="pt", add_special_tokens=False).input_ids[0])
        max_prompt_len = max(max_prompt_len, int(prompt_len))
        max_question_len = max(max_question_len, int(question_len))
        samples.append(
            {
                "sample_idx": int(sample_idx),
                "query_positions": [int(x) for x in query_positions],
                "ledger_prompt": prompt,
                "ledger_prompt_tokens": int(prompt_len),
                "question_prompt": question_prompt,
                "question_prompt_tokens": int(question_len),
                "filler_repeats": int(used_filler_repeats),
            }
        )

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    max_ledger_new_tokens = int(args.max_new_tokens)
    if max_ledger_new_tokens <= 0:
        ledger_stub = build_ledger_output_stub(args.num_entries)
        max_ledger_new_tokens = int(
            len(tokenizer(ledger_stub, return_tensors="pt").input_ids[0])
            + max(16, int(args.generation_margin_tokens))
        )
    answer_stub = build_answer_output_stub(args.num_queries)
    max_answer_new_tokens = int(
        len(tokenizer(answer_stub, return_tensors="pt").input_ids[0])
        + max(8, int(args.generation_margin_tokens) // 2)
    )
    total_future_tokens = int(max_ledger_new_tokens + max_question_len + max_answer_new_tokens)
    max_len = int(max_prompt_len + total_future_tokens)
    print(
        f"[generated_memory] max_prompt_len={max_prompt_len} "
        f"max_ledger_new_tokens={max_ledger_new_tokens} "
        f"max_question_tokens={max_question_len} "
        f"max_answer_new_tokens={max_answer_new_tokens} "
        f"max_len={max_len}"
    )

    llm = load_model(args.model_name, max_len, dtype, args.device)

    results = []
    bucket_hits = {"far": [], "mid": [], "near": []}
    for sample in samples:
        teacher_ledger_attn_type = str(args.teacher_ledger_attn_type or "").strip()
        if (args.teacher_drift_diag or args.teacher_dense_kv_refresh) and not teacher_ledger_attn_type:
            teacher_ledger_attn_type = "Full_Flash_Attn"
        teacher_prefill_sec = 0.0
        teacher_decode_sec = 0.0
        teacher_ledger_ids = None
        teacher_saw_end = None
        teacher_drift_summary = None
        teacher_dense_kv_rows = None
        if teacher_ledger_attn_type:
            teacher_llm = load_model(args.model_name, max_len, dtype, args.device)
            teacher_config = generate_config(
                args.model_name,
                sample["ledger_prompt_tokens"],
                teacher_ledger_attn_type,
                retrieval_budget=args.retrieval_budget,
                estimation_budget=args.estimation_budget,
                token_budget_override=args.token_budget_override,
            )
            teacher_first_token, teacher_prefill_sec = init_generation_session(
                llm=teacher_llm,
                tokenizer=tokenizer,
                prompt_text=sample["ledger_prompt"],
                attn_type=teacher_ledger_attn_type,
                attn_config=teacher_config,
                total_future_tokens=int(max_ledger_new_tokens),
            )
            teacher_decode_start = time.time()
            _teacher_ledger_text, teacher_ledger_ids, _teacher_after_logits, teacher_saw_end = greedy_decode_until(
                llm=teacher_llm,
                tokenizer=tokenizer,
                start_token=teacher_first_token,
                max_new_tokens=max_ledger_new_tokens,
                stop_substrings=("END LEDGER",),
            )
            torch.cuda.synchronize()
            teacher_decode_sec = float(time.time() - teacher_decode_start)
            if args.teacher_dense_kv_refresh and teacher_ledger_ids is not None:
                teacher_dense_kv_rows = extract_flash_teacher_decode_kv_rows(
                    teacher_llm,
                    start_pos=int(sample["ledger_prompt_tokens"]),
                    token_count=len(teacher_ledger_ids),
                )
            release_model(teacher_llm)

        dense_trace = None
        if args.teacher_drift_diag and teacher_ledger_ids is not None:
            release_model(llm)
            llm = None
            teacher_diag_llm = load_model(args.model_name, max_len, dtype, args.device)
            teacher_diag_config = generate_config(
                args.model_name,
                sample["ledger_prompt_tokens"],
                teacher_ledger_attn_type,
                retrieval_budget=args.retrieval_budget,
                estimation_budget=args.estimation_budget,
                token_budget_override=args.token_budget_override,
            )
            _teacher_diag_first_token, _ = init_generation_session(
                llm=teacher_diag_llm,
                tokenizer=tokenizer,
                prompt_text=sample["ledger_prompt"],
                attn_type=teacher_ledger_attn_type,
                attn_config=teacher_diag_config,
                total_future_tokens=int(max_ledger_new_tokens),
            )
            _teacher_diag_text, _teacher_diag_ids, _teacher_diag_logits, _teacher_diag_saw_end, dense_trace = (
                teacher_force_decode_ids_with_trace(
                    llm=teacher_diag_llm,
                    tokenizer=tokenizer,
                    forced_ids=teacher_ledger_ids,
                    stop_substrings=("END LEDGER",),
                    max_steps=int(args.teacher_drift_max_steps),
                )
            )
            release_model(teacher_diag_llm)
            teacher_diag_llm = None
            llm = load_model(args.model_name, max_len, dtype, args.device)

        attn_config = generate_config(
            args.model_name,
            sample["ledger_prompt_tokens"],
            args.attn_type,
            retrieval_budget=args.retrieval_budget,
            estimation_budget=args.estimation_budget,
            token_budget_override=args.token_budget_override,
        )
        first_token, prefill_sec = init_generation_session(
            llm=llm,
            tokenizer=tokenizer,
            prompt_text=sample["ledger_prompt"],
            attn_type=args.attn_type,
            attn_config=attn_config,
            total_future_tokens=total_future_tokens,
        )
        decode_start = time.time()
        if teacher_ledger_ids is not None:
            if args.teacher_dense_kv_refresh and teacher_dense_kv_rows is not None and args.teacher_drift_diag and dense_trace is not None:
                ledger_text, ledger_ids, after_ledger_logits, saw_end, student_trace = teacher_force_decode_ids_with_kv_refresh_and_trace(
                    llm=llm,
                    tokenizer=tokenizer,
                    forced_ids=teacher_ledger_ids,
                    dense_kv_rows=teacher_dense_kv_rows,
                    stop_substrings=("END LEDGER",),
                    max_steps=int(args.teacher_drift_max_steps),
                )
                teacher_drift_summary = summarize_teacher_drift(dense_trace, student_trace)
                if teacher_drift_summary is not None:
                    teacher_drift_summary["teacher_attn_type"] = str(teacher_ledger_attn_type)
                    teacher_drift_summary["student_attn_type"] = str(args.attn_type)
                    teacher_drift_summary["forced_decode_tokens"] = int(
                        min(
                            len(teacher_ledger_ids),
                            int(args.teacher_drift_max_steps) if int(args.teacher_drift_max_steps) > 0 else len(teacher_ledger_ids),
                        )
                    )
                    teacher_drift_summary["teacher_dense_kv_refresh"] = True
            elif args.teacher_dense_kv_refresh and teacher_dense_kv_rows is not None:
                ledger_text, ledger_ids, after_ledger_logits, saw_end = teacher_force_decode_ids_with_kv_refresh(
                    llm=llm,
                    tokenizer=tokenizer,
                    forced_ids=teacher_ledger_ids,
                    dense_kv_rows=teacher_dense_kv_rows,
                    stop_substrings=("END LEDGER",),
                )
            elif args.teacher_drift_diag and dense_trace is not None:
                ledger_text, ledger_ids, after_ledger_logits, saw_end, student_trace = teacher_force_decode_ids_with_trace(
                    llm=llm,
                    tokenizer=tokenizer,
                    forced_ids=teacher_ledger_ids,
                    stop_substrings=("END LEDGER",),
                    max_steps=int(args.teacher_drift_max_steps),
                )
                teacher_drift_summary = summarize_teacher_drift(dense_trace, student_trace)
                if teacher_drift_summary is not None:
                    teacher_drift_summary["teacher_attn_type"] = str(teacher_ledger_attn_type)
                    teacher_drift_summary["student_attn_type"] = str(args.attn_type)
                    teacher_drift_summary["forced_decode_tokens"] = int(
                        min(
                            len(teacher_ledger_ids),
                            int(args.teacher_drift_max_steps) if int(args.teacher_drift_max_steps) > 0 else len(teacher_ledger_ids),
                        )
                    )
            else:
                ledger_text, ledger_ids, after_ledger_logits, saw_end = teacher_force_decode_ids(
                    llm=llm,
                    tokenizer=tokenizer,
                    forced_ids=teacher_ledger_ids,
                    stop_substrings=("END LEDGER",),
                )
        else:
            ledger_text, ledger_ids, after_ledger_logits, saw_end = greedy_decode_until(
                llm=llm,
                tokenizer=tokenizer,
                start_token=first_token,
                max_new_tokens=max_ledger_new_tokens,
                stop_substrings=("END LEDGER",),
            )
        answer_seed_logits = append_prompt_continuation(
            llm=llm,
            tokenizer=tokenizer,
            prompt_text=sample["question_prompt"],
            fallback_logits=after_ledger_logits,
        )
        ledger_start_pos = int(sample["ledger_prompt_tokens"])
        ledger_end_pos = int(ledger_start_pos + len(ledger_ids))
        question_start_pos = int(ledger_end_pos)
        question_end_pos = int(question_start_pos + sample["question_prompt_tokens"])
        token_position_ranges = {
            "ledger_prompt": [0, int(sample["ledger_prompt_tokens"])],
            "ledger_generated": [int(ledger_start_pos), int(ledger_end_pos)],
            "question": [int(question_start_pos), int(question_end_pos)],
        }
        online_ra_state = None
        online_graph_snapshot = None
        online_partition_state = (
            summarize_retrievalattention_partition_state(llm, label="online_answer_start")
            if (args.state_partition_diag or args.state_equiv_diag) and args.attn_type == "RetrievalAttention"
            else None
        )
        state_equiv_online_kv = None
        state_equiv_positions = []
        state_equiv_diag_result = None
        if args.state_equiv_diag:
            entry_spans_for_state = compute_entry_token_spans(
                tokenizer=tokenizer,
                ledger_prompt=sample["ledger_prompt"],
                ledger_output=ledger_text,
                query_positions=sample["query_positions"],
            )
            state_equiv_positions = choose_state_equiv_positions(
                total_tokens=int(question_end_pos),
                partition_state=online_partition_state,
                token_ranges=token_position_ranges,
                entry_spans=entry_spans_for_state,
            )
            state_equiv_online_kv = capture_kv_state_signature(llm, state_equiv_positions)
        if (
            (
                args.replay_prefill_compare
                or args.replay_import_online_prev_seeds
                or args.replay_import_online_overlay
                or args.state_equiv_diag
            )
            and args.attn_type == "RetrievalAttention"
        ):
            online_ra_state = capture_retrievalattention_answer_start_state(llm)
        if args.attn_type == "RetrievalAttention" and bool(args.replay_graph_compare):
            generated_token_positions = list(
                range(
                    int(sample["ledger_prompt_tokens"]),
                    int(sample["ledger_prompt_tokens"]) + int(len(ledger_ids)),
                )
            )
            online_graph_snapshot = capture_retrievalattention_graph_snapshot(
                llm=llm,
                token_positions=generated_token_positions,
            )
        clear_retrievalattention_answer_start_state(
            llm=llm,
            clear_prev_seeds=bool(args.answer_start_clear_prev_seeds),
            clear_overlay=bool(args.answer_start_clear_overlay),
        )
        if (
            args.attn_type == "RetrievalAttention"
            and hasattr(llm, "kv_cache")
        ):
            oracle_retrieve_flag = os.environ.get("RETRIEVALATTN_ORACLE_RETRIEVE", "0") == "1"
            oracle_compare_flag = os.environ.get("RETRIEVALATTN_ORACLE_COMPARE", "1") == "1"
            setattr(llm.kv_cache, "oracle_retrieval_enable", oracle_retrieve_flag)
            setattr(llm.kv_cache, "oracle_debug_enable", oracle_retrieve_flag)
            setattr(llm.kv_cache, "oracle_compare_enable", oracle_compare_flag)
            setattr(llm.kv_cache, "oracle_answer_start_pos", int(llm.kv_cache.decode_pos))
            setattr(llm.kv_cache, "oracle_debug_records", [])
            setattr(llm.kv_cache, "oracle_compare_records", [])
            if args.state_partition_diag or args.state_equiv_diag:
                setattr(llm.kv_cache, "attention_input_debug_enable", True)
                setattr(llm.kv_cache, "attention_input_debug_answer_start_pos", int(llm.kv_cache.decode_pos))
                setattr(llm.kv_cache, "attention_input_debug_records", [])
        if args.answer_prefix_scaffold:
            answer_text, answer_ids, _after_answer_logits, _ = greedy_decode_answers_with_prefix_scaffold(
                llm=llm,
                tokenizer=tokenizer,
                fallback_logits=answer_seed_logits,
                expected_answers=int(args.num_queries),
                max_new_tokens=max_answer_new_tokens,
                constrained_token_map=constrained_token_map,
            )
        else:
            first_answer = llm.sampling(answer_seed_logits, do_sample=False).to(llm.layers[0].device)
            answer_text, answer_ids, _after_answer_logits, _ = greedy_decode_answers(
                llm=llm,
                tokenizer=tokenizer,
                start_token=first_answer,
                max_new_tokens=max_answer_new_tokens,
                expected_answers=int(args.num_queries),
            )
        if (
            args.attn_type == "RetrievalAttention"
            and hasattr(llm, "kv_cache")
        ):
            setattr(llm.kv_cache, "oracle_retrieval_enable", False)
        oracle_debug_summary = None
        if (
            args.attn_type == "RetrievalAttention"
            and hasattr(llm, "kv_cache")
        ):
            records = list(getattr(llm.kv_cache, "oracle_debug_records", []))
            entry_spans = compute_entry_token_spans(
                tokenizer=tokenizer,
                ledger_prompt=sample["ledger_prompt"],
                ledger_output=ledger_text,
                query_positions=sample["query_positions"],
            )
            oracle_debug_summary = {
                "entry_spans": {str(k): v for k, v in entry_spans.items()},
                "retrieval_hits": summarize_oracle_debug(records, entry_spans),
                "record_count": int(len(records)),
            }
            oracle_compare_summary = list(getattr(llm.kv_cache, "oracle_compare_records", []))
            attention_input_debug_summary = (
                list(getattr(llm.kv_cache, "attention_input_debug_records", []))
                if (args.state_partition_diag or args.state_equiv_diag)
                else None
            )
            setattr(llm.kv_cache, "oracle_debug_enable", False)
            setattr(llm.kv_cache, "oracle_compare_enable", False)
            setattr(llm.kv_cache, "oracle_answer_start_pos", None)
            setattr(llm.kv_cache, "oracle_debug_records", [])
            setattr(llm.kv_cache, "oracle_compare_records", [])
            setattr(llm.kv_cache, "attention_input_debug_enable", False)
            setattr(llm.kv_cache, "attention_input_debug_answer_start_pos", None)
            setattr(llm.kv_cache, "attention_input_debug_records", [])
        else:
            oracle_compare_summary = None
            attention_input_debug_summary = None
        oracle_compare_agg = aggregate_oracle_compare(oracle_compare_summary)
        torch.cuda.synchronize()
        decode_sec = float(time.time() - decode_start)
        output_text = sample["ledger_prompt"] + ledger_text + sample["question_prompt"] + answer_text
        eval_result = evaluate_output(output_text, sample["query_positions"], args.num_entries)
        answer_start_pos = int(question_end_pos)
        answer_end_pos = int(answer_start_pos + len(answer_ids))
        for hit, pos in zip(eval_result["query_hits"], sample["query_positions"]):
            bucket_hits[bucket_name(int(pos), int(args.num_entries))].append(1.0 if hit else 0.0)
        decode_profile_msg = None
        if hasattr(llm, "kv_cache") and hasattr(llm.kv_cache, "report_decode_profile"):
            decode_profile_msg = llm.kv_cache.report_decode_profile(reset=True)
            if decode_profile_msg:
                print(decode_profile_msg)
        if args.state_equiv_diag:
            state_equiv_diag_result = run_state_equiv_diagnostic(
                llm=llm,
                tokenizer=tokenizer,
                sample=sample,
                ledger_text=ledger_text,
                ledger_ids=ledger_ids,
                online_ra_state=online_ra_state,
                online_reference_kv=state_equiv_online_kv,
                online_attention_records=attention_input_debug_summary,
                online_partition_state=online_partition_state,
                online_answer_seed_logits=answer_seed_logits,
                state_positions=state_equiv_positions,
                constrained_token_map=constrained_token_map,
                args=args,
                replay_future_tokens=int(sample["question_prompt_tokens"]) + 16,
                teacher_future_tokens=int(len(ledger_ids)) + int(sample["question_prompt_tokens"]) + 16,
                answer_prefix_scaffold=bool(args.answer_prefix_scaffold),
            )
        replay_prefill_result = None
        if args.replay_prefill_compare:
            replay_prefill_result = replay_answers_from_prefilled_ledger(
                llm=llm,
                tokenizer=tokenizer,
                ledger_prompt=sample["ledger_prompt"],
                ledger_text=ledger_text,
                question_prompt=sample["question_prompt"],
                attn_type=args.attn_type,
                retrieval_budget=args.retrieval_budget,
                estimation_budget=args.estimation_budget,
                token_budget_override=args.token_budget_override,
                max_answer_new_tokens=max_answer_new_tokens,
                num_queries=int(args.num_queries),
                answer_prefix_scaffold=bool(args.answer_prefix_scaffold),
                answer_constrained_token_map=constrained_token_map,
                question_via_decode=bool(args.replay_question_via_decode),
                online_ra_state=online_ra_state,
                import_online_prev_seeds=bool(args.replay_import_online_prev_seeds),
                import_online_overlay=bool(args.replay_import_online_overlay),
                clear_prev_seeds=bool(args.answer_start_clear_prev_seeds),
                clear_overlay=bool(args.answer_start_clear_overlay),
                oracle_retrieve_enable=bool(os.environ.get("RETRIEVALATTN_ORACLE_RETRIEVE", "0") == "1"),
                oracle_compare_enable=False,
                graph_compare_online_snapshot=online_graph_snapshot,
                state_partition_diag=bool(args.state_partition_diag),
            )
            replay_prefill_eval = evaluate_output(
                replay_prefill_result["output"],
                sample["query_positions"],
                args.num_entries,
            )
            replay_prefill_result.update(
                {
                    "query_acc": float(replay_prefill_eval["query_acc"]),
                    "strict_acc": bool(replay_prefill_eval["strict_acc"]),
                    "format_ok": bool(replay_prefill_eval["format_ok"]),
                    "query_hits": list(replay_prefill_eval["query_hits"]),
                    "entry_count": int(replay_prefill_eval["entry_count"]),
                    "answer_count": int(replay_prefill_eval["answer_count"]),
                    "same_answer_output": bool(replay_prefill_result["answer_output"] == answer_text),
                }
            )
        sample_result = {
            "sample_idx": sample["sample_idx"],
            "query_positions": sample["query_positions"],
            "ledger_prompt_tokens": sample["ledger_prompt_tokens"],
            "question_prompt_tokens": sample["question_prompt_tokens"],
            "token_position_ranges": {
                "ledger_prompt": [0, int(sample["ledger_prompt_tokens"])],
                "ledger_generated": [int(ledger_start_pos), int(ledger_end_pos)],
                "question": [int(question_start_pos), int(question_end_pos)],
                "answer": [int(answer_start_pos), int(answer_end_pos)],
            },
            "prefill_sec": float(prefill_sec + teacher_prefill_sec),
            "decode_sec": float(decode_sec + teacher_decode_sec),
            "student_prefill_sec": float(prefill_sec),
            "student_decode_sec": float(decode_sec),
            "teacher_prefill_sec": float(teacher_prefill_sec),
            "teacher_decode_sec": float(teacher_decode_sec),
            "ledger_generated_tokens": int(len(ledger_ids)),
            "answer_generated_tokens": int(len(answer_ids)),
            "saw_end_ledger": bool(saw_end),
            "teacher_saw_end_ledger": bool(teacher_saw_end) if teacher_saw_end is not None else None,
            "teacher_ledger_attn_type": teacher_ledger_attn_type or None,
            "teacher_dense_kv_refresh": bool(args.teacher_dense_kv_refresh),
            "answer_prefix_scaffold": bool(args.answer_prefix_scaffold),
            "answer_constrained_codebook": bool(args.answer_constrained_codebook),
            "answer_start_clear_prev_seeds": bool(args.answer_start_clear_prev_seeds),
            "answer_start_clear_overlay": bool(args.answer_start_clear_overlay),
            "replay_graph_compare": bool(args.replay_graph_compare),
            "state_partition_diag": bool(args.state_partition_diag),
            "state_equiv_diag": bool(args.state_equiv_diag),
            "partition_state": online_partition_state,
            "attention_input_debug": attention_input_debug_summary,
            "output": output_text,
            "ledger_output": ledger_text,
            "answer_output": answer_text,
            "decode_profile": decode_profile_msg,
            "oracle_debug": oracle_debug_summary,
            "oracle_compare": oracle_compare_summary,
            "oracle_compare_agg": oracle_compare_agg,
            "teacher_drift_diag": teacher_drift_summary,
            "state_equiv_diag": state_equiv_diag_result,
            "replay_prefill": replay_prefill_result,
            **eval_result,
        }
        results.append(sample_result)

    query_acc = float(np.mean([r["query_acc"] for r in results])) if results else 0.0
    strict_acc = float(np.mean([1.0 if r["strict_acc"] else 0.0 for r in results])) if results else 0.0
    format_acc = float(np.mean([1.0 if r["format_ok"] else 0.0 for r in results])) if results else 0.0
    unique_entry_ratio = float(np.mean([r["unique_entry_ratio"] for r in results])) if results else 0.0
    bucket_acc = {
        key: (float(np.mean(vals)) if vals else 0.0)
        for key, vals in bucket_hits.items()
    }

    summary = {
        "num_samples": int(len(results)),
        "num_entries": int(args.num_entries),
        "num_queries": int(args.num_queries),
        "min_prompt_tokens": int(min_prompt_tokens),
        "max_ledger_new_tokens": int(max_ledger_new_tokens),
        "max_answer_new_tokens": int(max_answer_new_tokens),
        "query_acc": float(query_acc),
        "strict_acc": float(strict_acc),
        "format_acc": float(format_acc),
        "unique_entry_ratio": float(unique_entry_ratio),
        "bucket_acc": bucket_acc,
        "avg_prefill_sec": float(np.mean([r["prefill_sec"] for r in results])) if results else 0.0,
        "avg_decode_sec": float(np.mean([r["decode_sec"] for r in results])) if results else 0.0,
        "avg_ledger_generated_tokens": float(np.mean([r["ledger_generated_tokens"] for r in results])) if results else 0.0,
        "avg_answer_generated_tokens": float(np.mean([r["answer_generated_tokens"] for r in results])) if results else 0.0,
        "avg_total_generated_tokens": float(
            np.mean([r["ledger_generated_tokens"] + r["answer_generated_tokens"] for r in results])
        ) if results else 0.0,
        "model_name": args.model_name,
        "attn_type": args.attn_type,
        "token_budget_override": int(args.token_budget_override) if args.token_budget_override is not None else None,
        "answer_prefix_scaffold": bool(args.answer_prefix_scaffold),
        "answer_constrained_codebook": bool(args.answer_constrained_codebook),
        "answer_start_clear_prev_seeds": bool(args.answer_start_clear_prev_seeds),
        "answer_start_clear_overlay": bool(args.answer_start_clear_overlay),
        "replay_graph_compare": bool(args.replay_graph_compare),
        "state_partition_diag": bool(args.state_partition_diag),
        "state_equiv_diag": bool(args.state_equiv_diag),
    }
    oracle_compare_rows = []
    for row in results:
        oracle_compare_rows.extend(row.get("oracle_compare", []) or [])
    oracle_compare_summary = aggregate_oracle_compare(oracle_compare_rows)
    if oracle_compare_summary is not None:
        summary["oracle_compare"] = oracle_compare_summary
    teacher_drift_rows = [row.get("teacher_drift_diag") for row in results if row.get("teacher_drift_diag") is not None]
    teacher_drift_summary = aggregate_teacher_drift(teacher_drift_rows)
    if teacher_drift_summary is not None:
        summary["teacher_drift_diag"] = teacher_drift_summary
    state_equiv_rows = [row.get("state_equiv_diag") for row in results if row.get("state_equiv_diag") is not None]
    if state_equiv_rows:
        def _avg_nested(path):
            vals = []
            for row in state_equiv_rows:
                cur = row
                for key in path:
                    if not isinstance(cur, dict) or key not in cur:
                        cur = None
                        break
                    cur = cur[key]
                if cur is not None:
                    vals.append(float(cur))
            return float(np.mean(vals)) if vals else None

        summary["state_equiv_diag"] = {
            "num_rows": int(len(state_equiv_rows)),
            "avg_replay_key_l2": _avg_nested(["replay_prefill", "kv_compare_to_online", "avg_key_l2"]),
            "avg_replay_value_l2": _avg_nested(["replay_prefill", "kv_compare_to_online", "avg_value_l2"]),
            "avg_replay_attention_l2": _avg_nested(["replay_prefill", "attention_compare_to_online", "sparse_out_l2"]),
            "avg_teacher_key_l2": _avg_nested(["teacher_forced_decode", "kv_compare_to_online", "avg_key_l2"]),
            "avg_teacher_value_l2": _avg_nested(["teacher_forced_decode", "kv_compare_to_online", "avg_value_l2"]),
            "avg_teacher_attention_l2": _avg_nested(["teacher_forced_decode", "attention_compare_to_online", "sparse_out_l2"]),
        }
    replay_rows = [row.get("replay_prefill") for row in results if row.get("replay_prefill") is not None]
    if replay_rows:
        summary["replay_prefill"] = {
            "query_acc": float(np.mean([float(r["query_acc"]) for r in replay_rows])),
            "strict_acc": float(np.mean([1.0 if r["strict_acc"] else 0.0 for r in replay_rows])),
            "format_acc": float(np.mean([1.0 if r["format_ok"] else 0.0 for r in replay_rows])),
            "avg_prefill_sec": float(np.mean([float(r["prefill_sec"]) for r in replay_rows])),
            "avg_decode_sec": float(np.mean([float(r["decode_sec"]) for r in replay_rows])),
            "avg_replay_prompt_tokens": float(np.mean([float(r["replay_prompt_tokens"]) for r in replay_rows])),
            "avg_answer_generated_tokens": float(np.mean([float(r["answer_generated_tokens"]) for r in replay_rows])),
            "same_answer_output_rate": float(np.mean([1.0 if r["same_answer_output"] else 0.0 for r in replay_rows])),
        }
        graph_compare_rows = [r.get("graph_compare") for r in replay_rows if r.get("graph_compare") is not None]
        if graph_compare_rows:
            summary["replay_prefill"]["graph_compare"] = {
                "avg_online_degree": float(np.mean([float(r["avg_online_degree"]) for r in graph_compare_rows])),
                "avg_replay_degree": float(np.mean([float(r["avg_replay_degree"]) for r in graph_compare_rows])),
                "online_nonempty_rate": float(np.mean([float(r["online_nonempty_rate"]) for r in graph_compare_rows])),
                "replay_nonempty_rate": float(np.mean([float(r["replay_nonempty_rate"]) for r in graph_compare_rows])),
                "replay_nonempty_online_empty_rate": float(
                    np.mean([float(r["replay_nonempty_online_empty_rate"]) for r in graph_compare_rows])
                ),
                "avg_jaccard": float(np.mean([float(r["avg_jaccard"]) for r in graph_compare_rows])),
                "avg_replay_recall": float(np.mean([float(r["avg_replay_recall"]) for r in graph_compare_rows])),
            }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "generated_memory_results.jsonl"
    summary_path = output_dir / "summary.json"
    with result_path.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("[generated_memory] summary=" + json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
