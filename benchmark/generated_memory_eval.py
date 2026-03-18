import argparse
import json
import math
import os
import random
import re
import sys
import time
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


def build_question_prompt(query_positions):
    question_lines = [
        f"QUESTION {i + 1}: What was the value in ENTRY {int(pos)}?"
        for i, pos in enumerate(query_positions)
    ]
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
        question_prompt = build_question_prompt(query_positions)
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
        teacher_prefill_sec = 0.0
        teacher_decode_sec = 0.0
        teacher_ledger_ids = None
        teacher_saw_end = None
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
            release_model(teacher_llm)

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
            setattr(llm.kv_cache, "oracle_debug_enable", False)
            setattr(llm.kv_cache, "oracle_compare_enable", False)
            setattr(llm.kv_cache, "oracle_answer_start_pos", None)
            setattr(llm.kv_cache, "oracle_debug_records", [])
            setattr(llm.kv_cache, "oracle_compare_records", [])
        else:
            oracle_compare_summary = None
        oracle_compare_agg = aggregate_oracle_compare(oracle_compare_summary)
        torch.cuda.synchronize()
        decode_sec = float(time.time() - decode_start)
        output_text = sample["ledger_prompt"] + ledger_text + sample["question_prompt"] + answer_text
        eval_result = evaluate_output(output_text, sample["query_positions"], args.num_entries)
        for hit, pos in zip(eval_result["query_hits"], sample["query_positions"]):
            bucket_hits[bucket_name(int(pos), int(args.num_entries))].append(1.0 if hit else 0.0)
        decode_profile_msg = None
        if hasattr(llm, "kv_cache") and hasattr(llm.kv_cache, "report_decode_profile"):
            decode_profile_msg = llm.kv_cache.report_decode_profile(reset=True)
            if decode_profile_msg:
                print(decode_profile_msg)
        sample_result = {
            "sample_idx": sample["sample_idx"],
            "query_positions": sample["query_positions"],
            "ledger_prompt_tokens": sample["ledger_prompt_tokens"],
            "question_prompt_tokens": sample["question_prompt_tokens"],
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
            "output": output_text,
            "ledger_output": ledger_text,
            "answer_output": answer_text,
            "decode_profile": decode_profile_msg,
            "oracle_debug": oracle_debug_summary,
            "oracle_compare": oracle_compare_summary,
            "oracle_compare_agg": oracle_compare_agg,
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
    }
    oracle_compare_rows = []
    for row in results:
        oracle_compare_rows.extend(row.get("oracle_compare", []) or [])
    oracle_compare_summary = aggregate_oracle_compare(oracle_compare_rows)
    if oracle_compare_summary is not None:
        summary["oracle_compare"] = oracle_compare_summary

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
