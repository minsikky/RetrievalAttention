import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HF_CACHE_DIR = PROJECT_ROOT / ".hf_cache"
os.environ.setdefault("HF_HOME", str(DEFAULT_HF_CACHE_DIR))
os.environ.setdefault("HF_HUB_CACHE", str(DEFAULT_HF_CACHE_DIR / "hub"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.environ["HF_HUB_CACHE"])
os.environ.setdefault("HF_DATASETS_CACHE", str(DEFAULT_HF_CACHE_DIR / "datasets"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(DEFAULT_HF_CACHE_DIR / "transformers"))
for _cache_dir in (
    os.environ["HF_HOME"],
    os.environ["HF_HUB_CACHE"],
    os.environ["HF_DATASETS_CACHE"],
    os.environ["TRANSFORMERS_CACHE"],
):
    Path(_cache_dir).mkdir(parents=True, exist_ok=True)

import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from transformers import AutoConfig

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.generated_memory_hf_eval import (
    dtype_from_name,
    load_hf_model,
    load_tokenizer,
    model_device,
)
from benchmark.selector_eval.runners.run_hf_paged_pq_intervention_eval import (
    ApproxStats,
    greedy_dense_trace,
    patched_paged_pq_attention,
    reset_paged_pq_attention_state,
    summarize_logit_trace,
    teacher_forced_trace,
)


PROMPT_0SHOT = """Please read the following text and answer the question below.

$DOC$

What is the correct answer to this question: $Q$
Choices:
(A) $C_A$
(B) $C_B$
(C) $C_C$
(D) $C_D$
Format your response as follows: "The correct answer is (insert answer here)"."""


ANSWER_PATTERNS = (
    re.compile(r"The correct answer is\s*\(([A-D])\)", re.IGNORECASE),
    re.compile(r"The correct answer is\s*([A-D])\b", re.IGNORECASE),
    re.compile(r"correct answer\s*[:\-]?\s*\(?([A-D])\)?", re.IGNORECASE),
    re.compile(r"^\s*\(?([A-D])\)?[\.\):\s]", re.IGNORECASE),
)


def parse_args():
    parser = argparse.ArgumentParser(description="HF-native LongBench v2 evaluator.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--attn_implementation", type=str, default="")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--low_cpu_mem_usage", action="store_true")
    parser.add_argument("--hf_language_model_only", action="store_true")
    parser.add_argument("--use_chat_template", action="store_true")
    parser.add_argument("--disable_thinking", action="store_true")
    parser.add_argument("--dataset_name", type=str, default="THUDM/LongBench-v2")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output_dir", type=str, default="longbench_v2_hf_result")
    parser.add_argument("--max_examples", type=int, default=16)
    parser.add_argument("--length_filter", type=str, default="", choices=["", "short", "medium", "long"])
    parser.add_argument("--difficulty_filter", type=str, default="", choices=["", "easy", "hard"])
    parser.add_argument("--domain_filter", type=str, default="")
    parser.add_argument("--id_filter", type=str, default="")
    parser.add_argument("--selection", type=str, default="first", choices=["first", "random", "shortest"])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max_input_tokens", type=int, default=120000)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--dataset_scan_limit", type=int, default=1000)
    parser.add_argument("--qwen_yarn_factor", type=float, default=0.0)
    parser.add_argument("--qwen_yarn_original_max_position_embeddings", type=int, default=32768)
    parser.add_argument("--attention_mode", choices=["dense", "pagedpq"], default="dense")
    parser.add_argument(
        "--approx_prefill",
        action="store_true",
        help="also apply paged-PQ attention during batched prefill; default is dense prefill + approximate decode",
    )
    parser.add_argument("--layers", default="all")
    parser.add_argument("--selector_mode", choices=["fullscan", "routed"], default="fullscan")
    parser.add_argument("--selector_backend", choices=["torch", "cuda_ext", "auto"], default="cuda_ext")
    parser.add_argument("--budget", type=int, default=8)
    parser.add_argument("--budget_by_head", default="")
    parser.add_argument("--rerank_candidates", type=int, default=0)
    parser.add_argument("--tail_samples", type=int, default=16384)
    parser.add_argument("--tail_bands", type=int, default=8)
    parser.add_argument("--tail_seed", type=int, default=0)
    parser.add_argument("--tail_sampling", choices=["random", "linspace", "systematic"], default="systematic")
    parser.add_argument("--tail_mode", choices=["sample", "pq_value", "vpq_value", "page_mean"], default="vpq_value")
    parser.add_argument(
        "--online_confidence_rule",
        choices=[
            "none",
            "geometric_probe_tail_switch",
            "geometric_tail_stability_switch",
            "pq_proxy_mass_budget",
            "pq_ranked_mass_budget",
        ],
        default="pq_ranked_mass_budget",
    )
    parser.add_argument("--tail_score_calibration", choices=["none", "affine_selected"], default="affine_selected")
    parser.add_argument("--tail_probe_rel_l2_max", type=float, default=float("inf"))
    parser.add_argument("--tail_proxy_mass_min", type=float, default=0.97)
    parser.add_argument("--tail_proxy_mass_max", type=float, default=1.0)
    parser.add_argument("--tail_pq_corr_min", type=float, default=-1.0)
    parser.add_argument("--tail_pq_relrmse_max", type=float, default=float("inf"))
    parser.add_argument(
        "--ranked_confidence_cost_mode",
        choices=["exact", "upper_bound"],
        default="exact",
    )
    parser.add_argument("--geometric_min_budget", type=int, default=8)
    parser.add_argument("--geometric_max_budget", type=int, default=512)
    parser.add_argument("--geometric_growth", type=float, default=1.5)
    parser.add_argument("--geometric_probe_scale", type=float, default=1.5)
    parser.add_argument("--geometric_budget_granularity", type=int, default=8)
    parser.add_argument("--selected_value_mode", choices=["exact", "vpq_value"], default="exact")
    parser.add_argument(
        "--selected_value_exact_rule",
        choices=["fixed", "selector_rank", "selected_mass", "selected_risk_mass", "selected_mass_or_risk"],
        default="fixed",
    )
    parser.add_argument("--selected_value_exact_top", type=int, default=0)
    parser.add_argument("--selected_value_exact_mass", type=float, default=0.0)
    parser.add_argument("--selected_value_exact_risk_mass", type=float, default=0.0)
    parser.add_argument("--selected_value_min_exact_top", type=int, default=0)
    parser.add_argument("--selected_value_max_exact_top", type=int, default=0)
    parser.add_argument("--selected_value_exact_all_context_max", type=int, default=0)
    parser.add_argument("--selected_value_exact_all_fraction_min", type=float, default=0.0)
    parser.add_argument("--selected_value_residual_norm_bytes", type=int, default=2)
    parser.add_argument("--tail_blend", type=float, default=0.0)
    parser.add_argument("--prefill_tail_blend", type=float, default=None)
    parser.add_argument("--decode_tail_blend", type=float, default=None)
    parser.add_argument("--tail_off_heads", default="")
    parser.add_argument("--static_prefix", type=int, default=128)
    parser.add_argument("--static_suffix", type=int, default=128)
    parser.add_argument("--page_size", type=int, default=512)
    parser.add_argument("--prefill_chunk_size", type=int, default=2048)
    parser.add_argument(
        "--prefill_selector_backend",
        choices=["native", "native_fused", "torch_lut", "torch_lut_fp16", "torch_lut_streaming", "torch_lut_batched", "torch_matmul"],
        default="torch_lut",
    )
    parser.add_argument("--prefill_selector_stride", type=int, default=1)
    parser.add_argument("--prefill_selector_tile_size", type=int, default=0)
    parser.add_argument("--prefill_rank_buffer_limit_mb", type=float, default=4096.0)
    parser.add_argument("--prefill_selector_page_block_size", type=int, default=0)
    parser.add_argument("--prefill_tail_score_reuse", action="store_true")
    parser.add_argument(
        "--prefill_attention_backend",
        choices=["native", "flashinfer_blocksparse", "flashinfer_page_blocks"],
        default="native",
    )
    parser.add_argument("--subvecs", type=int, default=4)
    parser.add_argument("--subbits", type=int, default=8)
    parser.add_argument("--value_subvecs", type=int, default=1)
    parser.add_argument("--value_subbits", type=int, default=4)
    parser.add_argument("--value_pq_group_pages", type=int, default=1)
    parser.add_argument("--kmeans_iters", type=int, default=3)
    parser.add_argument(
        "--index_build_backend",
        choices=["numpy", "torch_gpu"],
        default=os.environ.get("PAGEDPQ_INDEX_BUILD_BACKEND", "torch_gpu"),
    )
    parser.add_argument("--nprobes", default="16,32,64,128,256,512")
    parser.add_argument("--router_prototypes", type=int, default=16)
    parser.add_argument("--router_merge_rel", type=float, default=0.05)
    parser.add_argument("--router_merge_var", type=float, default=0.0)
    parser.add_argument("--router_max_groups", type=int, default=512)
    parser.add_argument("--key_bytes", type=int, default=2)
    parser.add_argument("--value_bytes", type=int, default=2)
    parser.add_argument("--profile_native_ops", action="store_true")
    parser.add_argument("--disable_cost_stats", action="store_true")
    parser.add_argument("--disable_native_decode_fused", dest="disable_native_decode_fused", action="store_true", default=True)
    parser.add_argument("--enable_native_decode_fused", dest="disable_native_decode_fused", action="store_false")
    parser.add_argument("--native_decode_scoreless_fused", action="store_true")
    parser.add_argument("--native_decode_scoreless_force_mode", type=int, default=2)
    parser.add_argument("--allow_tf32_selector", action="store_true")
    parser.add_argument("--native_decode_tail", action="store_true")
    parser.add_argument(
        "--diagnose_dense_reference",
        action="store_true",
        help="For pagedpq mode, run dense greedy first and compare paged-PQ teacher-forced logits/hidden states on the dense trajectory.",
    )
    return parser.parse_args()


def build_prompt(row):
    return (
        PROMPT_0SHOT.replace("$DOC$", str(row["context"]).strip())
        .replace("$Q$", str(row["question"]).strip())
        .replace("$C_A$", str(row["choice_A"]).strip())
        .replace("$C_B$", str(row["choice_B"]).strip())
        .replace("$C_C$", str(row["choice_C"]).strip())
        .replace("$C_D$", str(row["choice_D"]).strip())
    )


def maybe_apply_chat_template(tokenizer, prompt: str, args):
    if not bool(args.use_chat_template):
        return prompt
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if bool(args.disable_thinking):
        kwargs["enable_thinking"] = False
    try:
        return tokenizer.apply_chat_template([{"role": "user", "content": prompt}], **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template([{"role": "user", "content": prompt}], **kwargs)


def truncate_middle(input_ids, max_input_tokens: int):
    if int(max_input_tokens) <= 0 or int(input_ids.shape[1]) <= int(max_input_tokens):
        return input_ids, False
    half = int(max_input_tokens) // 2
    keep_tail = int(max_input_tokens) - half
    return torch.cat([input_ids[:, :half], input_ids[:, -keep_tail:]], dim=1), True


def maybe_apply_qwen_yarn(config, args):
    factor = float(args.qwen_yarn_factor)
    if factor <= 0.0:
        return False

    original_max = int(args.qwen_yarn_original_max_position_embeddings)
    max_position_embeddings = max(
        int(args.max_input_tokens) + int(args.max_new_tokens),
        int(round(original_max * factor)),
    )
    rope_parameters = {
        "mrope_interleaved": True,
        "mrope_section": [11, 11, 10],
        "rope_type": "yarn",
        "rope_theta": 10000000,
        "partial_rotary_factor": 0.25,
        "factor": factor,
        "original_max_position_embeddings": original_max,
    }

    targets = [config]
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        targets.append(text_config)
    for target in targets:
        setattr(target, "rope_parameters", dict(rope_parameters))
        setattr(target, "max_position_embeddings", int(max_position_embeddings))
    return True


def extract_answer(text: str):
    for pattern in ANSWER_PATTERNS:
        match = pattern.search(text or "")
        if match:
            return match.group(1).upper()
    compact = (text or "").strip().upper()
    if compact in {"A", "B", "C", "D"}:
        return compact
    return None


def row_matches(row, args):
    if args.id_filter and str(row.get("_id", "")) != str(args.id_filter):
        return False
    if args.length_filter and str(row.get("length", "")) != str(args.length_filter):
        return False
    if args.difficulty_filter and str(row.get("difficulty", "")) != str(args.difficulty_filter):
        return False
    if args.domain_filter and str(row.get("domain", "")) != str(args.domain_filter):
        return False
    return True


def iter_json_array(path: str | Path, chunk_size: int = 1 << 20):
    decoder = json.JSONDecoder()
    buffer = ""
    pos = 0
    started = False
    with Path(path).open("r", encoding="utf-8") as f:
        while True:
            if pos >= len(buffer):
                more = f.read(chunk_size)
                if not more:
                    return
                buffer = buffer[pos:] + more
                pos = 0
            while True:
                while pos < len(buffer) and buffer[pos].isspace():
                    pos += 1
                if not started:
                    if pos >= len(buffer):
                        break
                    if buffer[pos] != "[":
                        raise ValueError(f"expected JSON array in {path}")
                    started = True
                    pos += 1
                    continue
                while pos < len(buffer) and buffer[pos].isspace():
                    pos += 1
                if pos < len(buffer) and buffer[pos] == ",":
                    pos += 1
                    continue
                if pos < len(buffer) and buffer[pos] == "]":
                    return
                break
            try:
                obj, end = decoder.raw_decode(buffer, pos)
            except json.JSONDecodeError:
                more = f.read(chunk_size)
                if not more:
                    raise
                buffer = buffer[pos:] + more
                pos = 0
                continue
            yield obj
            buffer = buffer[end:]
            pos = 0


def iter_longbench_rows(args):
    if str(args.dataset_name) == "THUDM/LongBench-v2":
        path = hf_hub_download(
            repo_id=str(args.dataset_name),
            filename="data.json",
            repo_type="dataset",
        )
        return iter_json_array(path)
    return iter(load_dataset(args.dataset_name, split=args.split, streaming=bool(args.streaming)))


def select_rows(args):
    rows = []
    scanned = 0
    for row in iter_longbench_rows(args):
        scanned += 1
        if row_matches(row, args):
            rows.append(
                {
                    "_id": row["_id"],
                    "domain": row["domain"],
                    "sub_domain": row["sub_domain"],
                    "difficulty": row["difficulty"],
                    "length": row["length"],
                    "question": row["question"],
                    "choice_A": row["choice_A"],
                    "choice_B": row["choice_B"],
                    "choice_C": row["choice_C"],
                    "choice_D": row["choice_D"],
                    "answer": row["answer"],
                    "context": row["context"],
                }
            )
        if args.selection == "first" and len(rows) >= int(args.max_examples):
            break
        if int(args.dataset_scan_limit) > 0 and scanned >= int(args.dataset_scan_limit):
            break

    if args.selection == "shortest":
        rows.sort(key=lambda item: len(str(item["context"])))
    elif args.selection == "random":
        rng = random.Random(int(args.seed))
        rng.shuffle(rows)
    return rows[: int(args.max_examples)], int(scanned)


@torch.inference_mode()
def run_one(model, tokenizer, row, args):
    device = model_device(model)
    prompt = maybe_apply_chat_template(tokenizer, build_prompt(row), args)
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=not bool(args.use_chat_template))
    original_tokens = int(enc.input_ids.shape[1])
    input_ids, truncated = truncate_middle(enc.input_ids, int(args.max_input_tokens))
    attention_mask = torch.ones_like(input_ids)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    generate_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": int(args.max_new_tokens),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": bool(float(args.temperature) > 0.0),
    }
    if float(args.temperature) > 0.0:
        generate_kwargs["temperature"] = float(args.temperature)

    start = time.time()
    output_ids = model.generate(**generate_kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = float(time.time() - start)

    new_ids = output_ids[0, input_ids.shape[1] :].detach().cpu()
    response = tokenizer.decode(new_ids, skip_special_tokens=True)
    pred = extract_answer(response)
    answer = str(row["answer"]).strip().upper()
    return {
        "_id": row["_id"],
        "domain": row["domain"],
        "sub_domain": row["sub_domain"],
        "difficulty": row["difficulty"],
        "length": row["length"],
        "answer": answer,
        "pred": pred,
        "judge": bool(pred == answer),
        "response": response,
        "prompt_tokens": original_tokens,
        "used_prompt_tokens": int(input_ids.shape[1]),
        "truncated": bool(truncated),
        "context_chars": int(len(str(row["context"]))),
        "generation_sec": elapsed,
    }


def aggregate(rows):
    total = len(rows)
    correct = sum(1 for row in rows if row["judge"])
    summary = {
        "num_examples": int(total),
        "accuracy": float(correct / total) if total else 0.0,
        "accuracy_pct": float(100.0 * correct / total) if total else 0.0,
        "avg_prompt_tokens": float(sum(row["prompt_tokens"] for row in rows) / total) if total else 0.0,
        "avg_used_prompt_tokens": float(sum(row["used_prompt_tokens"] for row in rows) / total) if total else 0.0,
        "truncated_count": int(sum(1 for row in rows if row["truncated"])),
        "avg_generation_sec": float(sum(row["generation_sec"] for row in rows) / total) if total else 0.0,
    }
    for key in ("difficulty", "length", "domain"):
        values = sorted({str(row[key]) for row in rows})
        summary[f"accuracy_by_{key}"] = {
            value: float(
                sum(1 for row in rows if str(row[key]) == value and row["judge"])
                / max(1, sum(1 for row in rows if str(row[key]) == value))
            )
            for value in values
        }
    return summary


def pagedpq_config(args):
    if str(args.attention_mode) != "pagedpq":
        return None
    return {
        "approx_prefill": bool(args.approx_prefill),
        "frontier_canonical_gpu": str(os.environ.get("FRONTIER_CANONICAL_GPU", "0")),
        "disable_cost_stats": bool(args.disable_cost_stats),
        "disable_native_decode_fused": bool(args.disable_native_decode_fused),
        "native_decode_scoreless_fused": bool(args.native_decode_scoreless_fused),
        "native_decode_scoreless_force_mode": int(args.native_decode_scoreless_force_mode),
        "native_decode_tail": bool(args.native_decode_tail),
        "budget": int(args.budget),
        "online_confidence_rule": str(args.online_confidence_rule),
        "tail_score_calibration": str(args.tail_score_calibration),
        "tail_probe_rel_l2_max": float(args.tail_probe_rel_l2_max),
        "tail_proxy_mass_min": float(args.tail_proxy_mass_min),
        "tail_proxy_mass_max": float(args.tail_proxy_mass_max),
        "tail_pq_corr_min": float(args.tail_pq_corr_min),
        "tail_pq_relrmse_max": float(args.tail_pq_relrmse_max),
        "ranked_confidence_cost_mode": str(args.ranked_confidence_cost_mode),
        "geometric_min_budget": int(args.geometric_min_budget),
        "geometric_max_budget": int(args.geometric_max_budget),
        "geometric_budget_granularity": int(args.geometric_budget_granularity),
        "tail_blend": float(args.tail_blend),
        "selected_value_mode": str(args.selected_value_mode),
        "selected_value_exact_rule": str(args.selected_value_exact_rule),
        "selected_value_exact_top": int(args.selected_value_exact_top),
        "selected_value_exact_mass": float(args.selected_value_exact_mass),
        "selected_value_exact_risk_mass": float(args.selected_value_exact_risk_mass),
        "selected_value_min_exact_top": int(args.selected_value_min_exact_top),
        "selected_value_max_exact_top": int(args.selected_value_max_exact_top),
        "selector_mode": str(args.selector_mode),
        "selector_backend": str(args.selector_backend),
        "page_size": int(args.page_size),
        "prefill_chunk_size": int(args.prefill_chunk_size),
        "prefill_selector_backend": str(args.prefill_selector_backend),
        "prefill_selector_tile_size": int(args.prefill_selector_tile_size),
        "prefill_rank_buffer_limit_mb": float(args.prefill_rank_buffer_limit_mb),
        "prefill_tail_score_reuse": bool(args.prefill_tail_score_reuse),
        "allow_tf32_selector": bool(args.allow_tf32_selector),
        "subvecs": int(args.subvecs),
        "subbits": int(args.subbits),
        "value_subvecs": int(args.value_subvecs),
        "value_subbits": int(args.value_subbits),
        "value_pq_group_pages": int(args.value_pq_group_pages),
        "kmeans_iters": int(args.kmeans_iters),
        "index_build_backend": str(args.index_build_backend),
        "index_build_backend": str(args.index_build_backend),
        "nprobes": str(args.nprobes),
    }


def aggregate_diagnostics(rows):
    total = len(rows)
    dense_correct = sum(1 for row in rows if row["dense_judge"])
    approx_correct = sum(1 for row in rows if row["approx_judge"])
    summary = {
        "num_examples": int(total),
        "dense_accuracy": float(dense_correct / total) if total else 0.0,
        "dense_accuracy_pct": float(100.0 * dense_correct / total) if total else 0.0,
        "approx_accuracy": float(approx_correct / total) if total else 0.0,
        "approx_accuracy_pct": float(100.0 * approx_correct / total) if total else 0.0,
        "free_run_exact_text_match_rate": float(
            sum(1 for row in rows if row["free_run_exact_text_match"]) / total
        )
        if total
        else 0.0,
        "avg_prompt_tokens": float(sum(row["prompt_tokens"] for row in rows) / total) if total else 0.0,
        "avg_used_prompt_tokens": float(sum(row["used_prompt_tokens"] for row in rows) / total) if total else 0.0,
        "truncated_count": int(sum(1 for row in rows if row["truncated"])),
        "avg_dense_seconds": float(sum(row["dense_generation_sec"] for row in rows) / total) if total else 0.0,
        "avg_approx_seconds": float(sum(row["approx_generation_sec"] for row in rows) / total) if total else 0.0,
    }
    metric_rows = [row.get("logit_trace", {}) for row in rows if row.get("logit_trace")]
    if metric_rows:
        for key in (
            "top1_agreement",
            "affected_top1_agreement",
            "mean_logit_relative_l2",
            "mean_logit_cosine",
            "mean_dense_to_approx_kl",
            "mean_hidden_relative_l2",
            "mean_hidden_cosine",
        ):
            values = [float(item[key]) for item in metric_rows if key in item]
            if values:
                summary[key] = float(sum(values) / len(values))
        for key in (
            "max_logit_relative_l2",
            "affected_max_logit_relative_l2",
            "max_dense_to_approx_kl",
            "affected_max_dense_to_approx_kl",
            "max_hidden_relative_l2",
            "affected_max_hidden_relative_l2",
        ):
            values = [float(item[key]) for item in metric_rows if key in item]
            if values:
                summary[key] = float(max(values))
        for key in ("min_logit_cosine", "min_hidden_cosine"):
            values = [float(item[key]) for item in metric_rows if key in item]
            if values:
                summary[key] = float(min(values))
    choice_steps = [
        step
        for row in rows
        for step in row.get("choice_logit_steps", [])
        if step.get("dense_choice_top") is not None and step.get("approx_choice_top") is not None
    ]
    if choice_steps:
        summary["choice_top_agreement"] = float(
            sum(1 for step in choice_steps if step["dense_choice_top"] == step["approx_choice_top"])
            / len(choice_steps)
        )
        margin_errors = [
            abs(float(step["dense_choice_margin"]) - float(step["approx_choice_margin"]))
            for step in choice_steps
            if step.get("dense_choice_margin") is not None and step.get("approx_choice_margin") is not None
        ]
        if margin_errors:
            summary["mean_choice_margin_abs_error"] = float(sum(margin_errors) / len(margin_errors))
            summary["max_choice_margin_abs_error"] = float(max(margin_errors))
    return summary


def encode_row(model, tokenizer, row, args):
    device = model_device(model)
    prompt = maybe_apply_chat_template(tokenizer, build_prompt(row), args)
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=not bool(args.use_chat_template))
    original_tokens = int(enc.input_ids.shape[1])
    input_ids, truncated = truncate_middle(enc.input_ids, int(args.max_input_tokens))
    input_ids = input_ids.to(device)
    return input_ids, original_tokens, bool(truncated)


def summarize_choice_logits(dense: dict, approx: dict, tokenizer) -> list[dict]:
    choice_token_ids: dict[str, int] = {}
    for label in ("A", "B", "C", "D"):
        encoded = tokenizer(label, add_special_tokens=False).input_ids
        if len(encoded) == 1:
            choice_token_ids[label] = int(encoded[0])
    rows = []
    if len(choice_token_ids) != 4:
        return rows
    for step, (dense_logits, approx_logits) in enumerate(
        zip(dense["logits"], approx["logits"], strict=False)
    ):
        dl = dense_logits.reshape(-1).float()
        al = approx_logits.reshape(-1).float()
        dense_choice_logits = {
            label: float(dl[token_id].item()) for label, token_id in choice_token_ids.items()
        }
        approx_choice_logits = {
            label: float(al[token_id].item()) for label, token_id in choice_token_ids.items()
        }
        dense_order = sorted(dense_choice_logits, key=dense_choice_logits.get, reverse=True)
        approx_order = sorted(approx_choice_logits, key=approx_choice_logits.get, reverse=True)
        dense_margin = (
            float(dense_choice_logits[dense_order[0]] - dense_choice_logits[dense_order[1]])
            if len(dense_order) >= 2
            else None
        )
        approx_margin = (
            float(approx_choice_logits[approx_order[0]] - approx_choice_logits[approx_order[1]])
            if len(approx_order) >= 2
            else None
        )
        rows.append(
            {
                "step": int(step),
                "dense_choice_top": dense_order[0] if dense_order else None,
                "approx_choice_top": approx_order[0] if approx_order else None,
                "choice_top_match": bool(dense_order and approx_order and dense_order[0] == approx_order[0]),
                "dense_choice_margin": dense_margin,
                "approx_choice_margin": approx_margin,
                "choice_margin_error": float(abs(dense_margin - approx_margin))
                if dense_margin is not None and approx_margin is not None
                else None,
                "dense_choice_logits": dense_choice_logits,
                "approx_choice_logits": approx_choice_logits,
            }
        )
    return rows


@torch.inference_mode()
def run_one_diagnostic(model, tokenizer, row, args, layer_ids, approx_stats: dict[int, ApproxStats]):
    input_ids, original_tokens, truncated = encode_row(model, tokenizer, row, args)
    forbidden: set[int] = set()

    dense_start = time.time()
    dense = greedy_dense_trace(model, input_ids, int(args.max_new_tokens), forbidden)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dense_elapsed = float(time.time() - dense_start)
    dense_text = tokenizer.decode(dense["tokens"], skip_special_tokens=True)
    dense_pred = extract_answer(dense_text)

    approx_start = time.time()
    reset_paged_pq_attention_state(model)
    with patched_paged_pq_attention(model, layer_ids, args, approx_stats):
        approx_teacher = teacher_forced_trace(model, input_ids, dense["tokens"], forbidden)
        approx_free = greedy_dense_trace(model, input_ids, int(args.max_new_tokens), forbidden)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    approx_elapsed = float(time.time() - approx_start)
    approx_text = tokenizer.decode(approx_free["tokens"], skip_special_tokens=True)
    approx_pred = extract_answer(approx_text)

    comparison = summarize_logit_trace(dense, approx_teacher, tokenizer, ignore_token_ids=forbidden)
    choice_logit_steps = summarize_choice_logits(dense, approx_teacher, tokenizer)
    answer = str(row["answer"]).strip().upper()
    return {
        "_id": row["_id"],
        "domain": row["domain"],
        "sub_domain": row["sub_domain"],
        "difficulty": row["difficulty"],
        "length": row["length"],
        "answer": answer,
        "dense_pred": dense_pred,
        "dense_judge": bool(dense_pred == answer),
        "dense_response": dense_text,
        "approx_pred": approx_pred,
        "approx_judge": bool(approx_pred == answer),
        "approx_response": approx_text,
        "free_run_exact_text_match": bool(dense_text == approx_text),
        "prompt_tokens": original_tokens,
        "used_prompt_tokens": int(input_ids.shape[1]),
        "truncated": bool(truncated),
        "context_chars": int(len(str(row["context"]))),
        "dense_generation_sec": dense_elapsed,
        "approx_generation_sec": approx_elapsed,
        "dense_tokens": [int(tok) for tok in dense["tokens"]],
        "approx_free_tokens": [int(tok) for tok in approx_free["tokens"]],
        "logit_trace": comparison["summary"],
        "logit_steps": comparison["steps"],
        "choice_logit_steps": choice_logit_steps,
    }


def parse_layer_ids(text: str, model) -> list[int]:
    if str(text).strip().lower() == "all":
        return list(range(len(model.model.layers)))
    return [int(part.strip()) for part in str(text).split(",") if part.strip()]


def summarize_approx_stats(stats: dict[int, ApproxStats]) -> dict[str, dict[str, float | int]]:
    payload: dict[str, dict[str, float | int]] = {}
    for layer, s in sorted(stats.items()):
        update_mb = float(s.index_build_read_mb + s.index_build_write_mb)
        payload[str(layer)] = {
            "calls": int(s.calls),
            "approx_attention_calls": int(s.approx_attention_calls),
            "passthrough_attention_calls": int(s.passthrough_attention_calls),
            "mean_selected_tokens": float(s.mean_selected),
            "mean_tail_samples": float(s.mean_tail_samples),
            "mean_selector_MB_per_head_query": float(s.mean_selector_mb),
            "mean_exact_KV_MB_per_head_query": float(s.mean_exact_kv_mb),
            "mean_tail_estimator_MB_per_head_query": float(s.mean_tail_mb),
            "mean_confidence_MB_per_head_query": float(s.mean_confidence_mb),
            "mean_step_MB_per_head_query": float(s.mean_step_mb),
            "selector_active_fraction": float(getattr(s, "selector_active_calls", 0)) / max(1, int(s.calls)),
            "tail_active_fraction": float(getattr(s, "tail_active_calls", 0)) / max(1, int(s.calls)),
            "confidence_active_fraction": float(getattr(s, "confidence_active_calls", 0)) / max(1, int(s.calls)),
            "mean_update_MB_per_head_query": float(update_mb / max(1, int(s.calls))),
            "mean_total_MB_per_head_query": float(s.mean_step_mb + update_mb / max(1, int(s.calls))),
            "index_build_calls": int(s.index_build_calls),
            "index_build_seconds": float(s.index_build_seconds),
            "index_build_read_MB": float(s.index_build_read_mb),
            "index_build_write_MB": float(s.index_build_write_mb),
            "index_build_total_MB": float(update_mb),
            "online_update_MB_per_attention_call": float(update_mb / max(1, int(s.approx_attention_calls))),
            "cache_cast_seconds": float(s.cache_cast_seconds),
            "patched_attention_seconds": float(s.patched_attention_seconds),
            "qkv_cache_seconds": float(s.qkv_cache_seconds),
            "index_sidecar_seconds": float(s.index_sidecar_seconds),
            "native_pack_seconds": float(s.native_pack_seconds),
            "native_selector_seconds": float(s.native_selector_seconds),
            "native_attention_seconds": float(s.native_attention_seconds),
            "output_projection_seconds": float(s.output_projection_seconds),
        }
    return payload


def aggregate_approx_stats(stats: dict[int, ApproxStats]) -> dict[str, float | int]:
    if not stats:
        return {}
    rows = list(stats.values())

    def mean_attr(attr: str) -> float:
        return float(sum(float(getattr(s, attr)) for s in rows) / max(1, len(rows)))

    total_update_mb = float(sum(float(s.index_build_read_mb + s.index_build_write_mb) for s in rows))
    total_calls = int(sum(int(s.calls) for s in rows))
    total_approx_calls = int(sum(int(s.approx_attention_calls) for s in rows))
    return {
        "layers": int(len(rows)),
        "calls_total": total_calls,
        "approx_attention_calls_total": total_approx_calls,
        "passthrough_attention_calls_total": int(sum(int(s.passthrough_attention_calls) for s in rows)),
        "mean_selected_tokens": mean_attr("mean_selected"),
        "mean_tail_samples": mean_attr("mean_tail_samples"),
        "mean_selector_MB_per_head_query": mean_attr("mean_selector_mb"),
        "mean_exact_KV_MB_per_head_query": mean_attr("mean_exact_kv_mb"),
        "mean_tail_estimator_MB_per_head_query": mean_attr("mean_tail_mb"),
        "mean_confidence_MB_per_head_query": mean_attr("mean_confidence_mb"),
        "mean_step_MB_per_head_query": mean_attr("mean_step_mb"),
        "selector_active_fraction": float(
            sum(int(getattr(s, "selector_active_calls", 0)) for s in rows) / max(1, total_calls)
        ),
        "tail_active_fraction": float(
            sum(int(getattr(s, "tail_active_calls", 0)) for s in rows) / max(1, total_calls)
        ),
        "confidence_active_fraction": float(
            sum(int(getattr(s, "confidence_active_calls", 0)) for s in rows) / max(1, total_calls)
        ),
        "online_update_cumulative_MB": total_update_mb,
        "online_update_MB_per_head_query": float(total_update_mb / max(1, total_calls)),
        "online_update_MB_per_attention_call": float(total_update_mb / max(1, total_approx_calls)),
        "mean_total_MB_per_head_query": float(mean_attr("mean_step_mb") + total_update_mb / max(1, total_calls)),
        "index_build_calls_total": int(sum(int(s.index_build_calls) for s in rows)),
        "index_build_seconds_total": float(sum(float(s.index_build_seconds) for s in rows)),
        "index_build_read_MB_total": float(sum(float(s.index_build_read_mb) for s in rows)),
        "index_build_write_MB_total": float(sum(float(s.index_build_write_mb) for s in rows)),
        "index_build_total_MB": total_update_mb,
        "cache_cast_seconds_total": float(sum(float(s.cache_cast_seconds) for s in rows)),
        "patched_attention_seconds_total": float(sum(float(s.patched_attention_seconds) for s in rows)),
        "qkv_cache_seconds_total": float(sum(float(s.qkv_cache_seconds) for s in rows)),
        "index_sidecar_seconds_total": float(sum(float(s.index_sidecar_seconds) for s in rows)),
        "native_pack_seconds_total": float(sum(float(s.native_pack_seconds) for s in rows)),
        "native_selector_seconds_total": float(sum(float(s.native_selector_seconds) for s in rows)),
        "native_attention_seconds_total": float(sum(float(s.native_attention_seconds) for s in rows)),
        "output_projection_seconds_total": float(sum(float(s.output_projection_seconds) for s in rows)),
    }


def main():
    args = parse_args()
    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if bool(args.allow_tf32_selector):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "args.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True), encoding="utf-8")

    rows, scanned = select_rows(args)
    (output_dir / "selected_ids.json").write_text(
        json.dumps(
            {
                "scanned": int(scanned),
                "selected": [
                    {
                        "_id": row["_id"],
                        "domain": row["domain"],
                        "difficulty": row["difficulty"],
                        "length": row["length"],
                        "context_chars": len(str(row["context"])),
                    }
                    for row in rows
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )

    config = AutoConfig.from_pretrained(
        args.model_name,
        trust_remote_code=bool(args.trust_remote_code),
        local_files_only=bool(args.local_files_only),
    )
    yarn_enabled = maybe_apply_qwen_yarn(config, args)
    tokenizer = load_tokenizer(args)
    dtype = dtype_from_name(args.dtype)
    model, auto_class = load_hf_model(args, dtype, config)
    setattr(args, "approx_prefill", bool(args.approx_prefill) and str(args.attention_mode) == "pagedpq")
    approx_stats: dict[int, ApproxStats] = {}
    layer_ids = parse_layer_ids(str(args.layers), model) if str(args.attention_mode) == "pagedpq" else []

    if bool(args.diagnose_dense_reference):
        if str(args.attention_mode) != "pagedpq":
            raise ValueError("--diagnose_dense_reference requires --attention_mode pagedpq")
        diagnostics = []
        out_path = output_dir / "diagnostics.jsonl"
        with out_path.open("w", encoding="utf-8") as fout:
            for row in tqdm(rows):
                pred = run_one_diagnostic(model, tokenizer, row, args, layer_ids, approx_stats)
                diagnostics.append(pred)
                fout.write(json.dumps(pred, ensure_ascii=False) + "\n")
                fout.flush()
                torch.cuda.empty_cache()

        summary = aggregate_diagnostics(diagnostics)
        summary.update(
            {
                "model_name": args.model_name,
                "auto_class": auto_class,
                "dataset_name": args.dataset_name,
                "split": args.split,
                "max_examples": int(args.max_examples),
                "length_filter": args.length_filter,
                "difficulty_filter": args.difficulty_filter,
                "domain_filter": args.domain_filter,
                "id_filter": args.id_filter,
                "selection": args.selection,
                "seed": int(args.seed),
                "max_input_tokens": int(args.max_input_tokens),
                "max_new_tokens": int(args.max_new_tokens),
                "temperature": float(args.temperature),
                "use_chat_template": bool(args.use_chat_template),
                "disable_thinking": bool(args.disable_thinking),
                "streaming": bool(args.streaming),
                "dataset_scan_limit": int(args.dataset_scan_limit),
                "scanned": int(scanned),
                "qwen_yarn_enabled": bool(yarn_enabled),
                "qwen_yarn_factor": float(args.qwen_yarn_factor),
                "qwen_yarn_original_max_position_embeddings": int(
                    args.qwen_yarn_original_max_position_embeddings
                ),
                "attention_mode": str(args.attention_mode),
                "approx_prefill": bool(args.approx_prefill),
                "diagnose_dense_reference": True,
                "layers": layer_ids,
                "pagedpq_config": pagedpq_config(args),
                "cost_proxy": summarize_approx_stats(approx_stats),
                "cost_proxy_aggregate": aggregate_approx_stats(approx_stats),
            }
        )
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
        print("[longbench_v2_hf] summary=" + json.dumps(summary, sort_keys=True))
        return

    predictions = []
    out_path = output_dir / "predictions.jsonl"
    attention_context = (
        patched_paged_pq_attention(model, layer_ids, args, approx_stats)
        if str(args.attention_mode) == "pagedpq"
        else None
    )
    if attention_context is None:
        class NullContext:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, tb):
                return False

        attention_context = NullContext()
    with attention_context, out_path.open("w", encoding="utf-8") as fout:
        for row in tqdm(rows):
            if str(args.attention_mode) == "pagedpq":
                reset_paged_pq_attention_state(model)
            pred = run_one(model, tokenizer, row, args)
            predictions.append(pred)
            fout.write(json.dumps(pred, ensure_ascii=False) + "\n")
            fout.flush()
            torch.cuda.empty_cache()

    summary = aggregate(predictions)
    summary.update(
        {
            "model_name": args.model_name,
            "auto_class": auto_class,
            "dataset_name": args.dataset_name,
            "split": args.split,
            "max_examples": int(args.max_examples),
            "length_filter": args.length_filter,
            "difficulty_filter": args.difficulty_filter,
            "domain_filter": args.domain_filter,
            "id_filter": args.id_filter,
            "selection": args.selection,
            "seed": int(args.seed),
            "max_input_tokens": int(args.max_input_tokens),
            "max_new_tokens": int(args.max_new_tokens),
            "temperature": float(args.temperature),
            "use_chat_template": bool(args.use_chat_template),
            "disable_thinking": bool(args.disable_thinking),
            "streaming": bool(args.streaming),
            "dataset_scan_limit": int(args.dataset_scan_limit),
            "scanned": int(scanned),
            "qwen_yarn_enabled": bool(yarn_enabled),
            "qwen_yarn_factor": float(args.qwen_yarn_factor),
            "qwen_yarn_original_max_position_embeddings": int(
                args.qwen_yarn_original_max_position_embeddings
            ),
            "attention_mode": str(args.attention_mode),
            "approx_prefill": bool(args.approx_prefill),
            "layers": layer_ids,
            "pagedpq_config": pagedpq_config(args),
            "cost_proxy": summarize_approx_stats(approx_stats),
            "cost_proxy_aggregate": aggregate_approx_stats(approx_stats),
        }
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print("[longbench_v2_hf] summary=" + json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
