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
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3.5-9B")
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
    parser.add_argument("--dataset_scan_limit", type=int, default=200)
    parser.add_argument("--qwen_yarn_factor", type=float, default=0.0)
    parser.add_argument("--qwen_yarn_original_max_position_embeddings", type=int, default=262144)
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


def select_rows(args):
    dataset = load_dataset(args.dataset_name, split=args.split, streaming=bool(args.streaming))
    rows = []
    scanned = 0
    for row in dataset:
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


def main():
    args = parse_args()
    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    predictions = []
    out_path = output_dir / "predictions.jsonl"
    with out_path.open("w", encoding="utf-8") as fout:
        for row in tqdm(rows):
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
        }
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print("[longbench_v2_hf] summary=" + json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
