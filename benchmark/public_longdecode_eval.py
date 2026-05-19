import argparse
import json
import os
import random
import re
import sys
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from transformers import AutoConfig

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

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.generated_memory_hf_eval import (  # noqa: E402
    dtype_from_name,
    load_hf_model,
    load_tokenizer,
    model_device,
)
from benchmark.longbench_v2_hf_eval import (  # noqa: E402
    aggregate_approx_stats,
    maybe_apply_qwen_yarn,
    pagedpq_config,
    parse_layer_ids,
    summarize_approx_stats,
    truncate_middle,
)
from benchmark.selector_eval.runners.run_hf_paged_pq_intervention_eval import (  # noqa: E402
    ApproxStats,
    patched_paged_pq_attention,
    reset_paged_pq_attention_state,
)


BENCHMARK_CHOICES = (
    "aime24",
    "gpqa",
    "livecodebench_codegen",
    "longgenbench_sgt_short",
    "longgenbench_sgt_long",
    "longgenbench_gsm8k",
)

ANSWER_INT_RE = re.compile(r"(-?\d+)")
BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
CHOICE_RE = re.compile(r"(?:answer\s+is|correct\s+answer\s+is|final\s+answer\s+is)\s*\(?([A-E])\)?", re.I)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Public long-decode benchmark runner for coding, reasoning, and LongGenBench. "
            "Supports dense and paged-PQ attention through the same HF patch path used by LongBench-v2."
        )
    )
    parser.add_argument("--benchmark", choices=BENCHMARK_CHOICES, required=True)
    parser.add_argument("--output_dir", type=str, default="public_longdecode_result")
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
    parser.add_argument("--attention_mode", choices=["dense", "pagedpq"], default="dense")
    parser.add_argument(
        "--approx_prefill",
        action="store_true",
        help="also apply paged-PQ attention during batched prefill; default is dense prefill + approximate decode",
    )
    parser.add_argument("--layers", default="all")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max_examples", type=int, default=4)
    parser.add_argument("--selection", choices=["first", "random", "shortest"], default="first")
    parser.add_argument("--max_input_tokens", type=int, default=120000)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--min_new_tokens", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--force_max_new_tokens", action="store_true")
    parser.add_argument("--dry_run", action="store_true", help="Load/select tasks and write args without loading a model.")

    parser.add_argument("--qwen_yarn_factor", type=float, default=0.0)
    parser.add_argument("--qwen_yarn_original_max_position_embeddings", type=int, default=32768)

    # LiveCodeBench controls.
    parser.add_argument("--livecodebench_repo", type=str, default="third_party/benchmarks/LiveCodeBench")
    parser.add_argument("--livecodebench_release", type=str, default="release_v6")
    parser.add_argument("--livecodebench_start_date", type=str, default="")
    parser.add_argument("--livecodebench_end_date", type=str, default="")
    parser.add_argument("--evaluate_code", action="store_true")
    parser.add_argument("--code_eval_processes", type=int, default=4)
    parser.add_argument("--code_eval_timeout", type=int, default=6)

    # LongGenBench controls.
    parser.add_argument("--longgenbench_mozhu_repo", type=str, default="third_party/benchmarks/LongGenBench_mozhu")
    parser.add_argument("--longgenbench_dominic_repo", type=str, default="third_party/benchmarks/LongGenBench_dominic")
    parser.add_argument("--longgenbench_gsm8k_k", type=int, default=32)
    parser.add_argument("--longgenbench_gsm8k_question_limit", type=int, default=256)

    # Current paged-PQ frontier knobs. These mirror benchmark/longbench_v2_hf_eval.py.
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
    parser.add_argument("--ranked_confidence_cost_mode", choices=["exact", "upper_bound"], default="exact")
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
        default="torch_matmul",
    )
    parser.add_argument("--prefill_selector_stride", type=int, default=1)
    parser.add_argument("--prefill_selector_tile_size", type=int, default=0)
    parser.add_argument("--prefill_rank_buffer_limit_mb", type=float, default=4096.0)
    parser.add_argument("--prefill_selector_page_block_size", type=int, default=0)
    parser.add_argument("--prefill_tail_score_reuse", action="store_true")
    parser.add_argument("--prefill_attention_backend", choices=["native", "flashinfer_blocksparse", "flashinfer_page_blocks"], default="native")
    parser.add_argument("--subvecs", type=int, default=4)
    parser.add_argument("--subbits", type=int, default=8)
    parser.add_argument("--value_subvecs", type=int, default=1)
    parser.add_argument("--value_subbits", type=int, default=4)
    parser.add_argument("--value_pq_group_pages", type=int, default=1)
    parser.add_argument("--kmeans_iters", type=int, default=3)
    parser.add_argument("--index_build_backend", choices=["numpy", "torch_gpu"], default=os.environ.get("PAGEDPQ_INDEX_BUILD_BACKEND", "torch_gpu"))
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
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def select_tasks(tasks: list[dict[str, Any]], args) -> list[dict[str, Any]]:
    if args.selection == "random":
        rng = random.Random(int(args.seed))
        rng.shuffle(tasks)
    elif args.selection == "shortest":
        tasks.sort(key=lambda row: int(row.get("prompt_chars", len(str(row.get("prompt", ""))))))
    return tasks[: max(0, int(args.max_examples))]


def load_aime24(args) -> list[dict[str, Any]]:
    path = PROJECT_ROOT / "benchmark/reasoning/data/aime24/test.jsonl"
    tasks = []
    for row in read_jsonl(path):
        prompt = (
            "Solve the following AIME problem. Think step by step, then put the final integer answer in "
            "\\boxed{}.\n\n"
            f"Problem:\n{row.get('question') or row.get('problem')}\n\nSolution:"
        )
        tasks.append(
            {
                "id": f"aime24_{row['id']}",
                "suite": "aime24",
                "prompt": prompt,
                "answer": str(row["answer"]).strip(),
                "prompt_chars": len(prompt),
            }
        )
    return select_tasks(tasks, args)


def load_gpqa(args) -> list[dict[str, Any]]:
    path = PROJECT_ROOT / "benchmark/reasoning/data/gpqa/test.jsonl"
    tasks = []
    letters = "ABCDE"
    for idx, row in enumerate(read_jsonl(path)):
        choices = row["choices"]
        answer_idx = int(row["answer"])
        lines = [
            "Answer the following multiple-choice question. Think step by step, then end with 'The answer is (X)'.",
            "",
            f"Question: {row['question']}",
            "",
            "Choices:",
        ]
        for i, choice in enumerate(choices):
            lines.append(f"({letters[i]}) {choice}")
        prompt = "\n".join(lines) + "\n\nSolution:"
        tasks.append(
            {
                "id": f"gpqa_{idx}",
                "suite": "gpqa",
                "prompt": prompt,
                "answer": letters[answer_idx],
                "prompt_chars": len(prompt),
            }
        )
    return select_tasks(tasks, args)


def add_livecodebench_path(args):
    repo = PROJECT_ROOT / args.livecodebench_repo
    if not repo.exists():
        raise FileNotFoundError(f"LiveCodeBench repo not found: {repo}")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    return repo


def load_livecodebench(args) -> list[dict[str, Any]]:
    add_livecodebench_path(args)
    from lcb_runner.benchmarks.code_generation import CodeGenerationProblem, load_code_generation_dataset

    try:
        problems = load_code_generation_dataset(
            release_version=str(args.livecodebench_release),
            start_date=args.livecodebench_start_date or None,
            end_date=args.livecodebench_end_date or None,
        )
    except RuntimeError as exc:
        if "Dataset scripts are no longer supported" not in str(exc):
            raise
        problems = load_livecodebench_direct_jsonl(args, CodeGenerationProblem)
    tasks = []
    for problem in problems:
        prompt = build_livecodebench_prompt(problem)
        tasks.append(
            {
                "id": f"lcb_{problem.question_id}",
                "suite": "livecodebench_codegen",
                "prompt": prompt,
                "answer": None,
                "prompt_chars": len(prompt),
                "metadata": {
                    "question_title": problem.question_title,
                    "question_id": problem.question_id,
                    "platform": problem.platform.value,
                    "difficulty": problem.difficulty.value,
                    "contest_date": problem.contest_date.isoformat(),
                },
                "_lcb_problem": problem,
            }
        )
    return select_tasks(tasks, args)


def livecodebench_filename(release: str) -> str:
    release = str(release or "release_v6")
    mapping = {
        "release_latest": "test6.jsonl",
        "release_v1": "test.jsonl",
        "release_v2": "test2.jsonl",
        "release_v3": "test3.jsonl",
        "release_v4": "test4.jsonl",
        "release_v5": "test5.jsonl",
        "release_v6": "test6.jsonl",
    }
    if release in mapping:
        return mapping[release]
    match = re.fullmatch(r"(?:release_)?v?([1-6])", release)
    if match:
        idx = int(match.group(1))
        return "test.jsonl" if idx == 1 else f"test{idx}.jsonl"
    raise ValueError(f"Unsupported LiveCodeBench release: {release}")


def load_livecodebench_direct_jsonl(args, problem_cls) -> list[Any]:
    filename = livecodebench_filename(str(args.livecodebench_release))
    path = hf_hub_download(
        repo_id="livecodebench/code_generation_lite",
        filename=filename,
        repo_type="dataset",
    )
    problems = [problem_cls(**row) for row in read_jsonl(Path(path))]
    if args.livecodebench_start_date:
        start = datetime.strptime(str(args.livecodebench_start_date), "%Y-%m-%d")
        problems = [problem for problem in problems if start <= problem.contest_date]
    if args.livecodebench_end_date:
        end = datetime.strptime(str(args.livecodebench_end_date), "%Y-%m-%d")
        problems = [problem for problem in problems if problem.contest_date <= end]
    print(f"Loaded {len(problems)} LiveCodeBench problems from {filename}")
    return problems


def build_livecodebench_prompt(problem) -> str:
    prompt = (
        "You are an expert Python programmer. You will be given a programming problem. "
        "Generate a correct Python program that matches the specification and passes all tests.\n\n"
        f"### Question:\n{problem.question_content}\n\n"
    )
    if getattr(problem, "starter_code", ""):
        prompt += (
            "### Format:\nUse the following starter code and enclose the final solution in a Python code block.\n"
            f"```python\n{problem.starter_code}\n```\n\n"
        )
    else:
        prompt += (
            "### Format:\nRead from stdin and write the answer to stdout. "
            "Enclose the final solution in a Python code block.\n"
            "```python\n# YOUR CODE HERE\n```\n\n"
        )
    prompt += "### Answer:\n"
    return prompt


def load_longgenbench_sgt(args, long: bool) -> list[dict[str, Any]]:
    repo = PROJECT_ROOT / args.longgenbench_mozhu_repo
    filename = "Dataset_long.json" if long else "Dataset_short.json"
    path = repo / "Dataset" / filename
    if not path.exists():
        raise FileNotFoundError(f"LongGenBench SGT dataset not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    tasks = []
    suite = "longgenbench_sgt_long" if long else "longgenbench_sgt_short"
    for idx, row in enumerate(data):
        prompt = str(row["prompt"])
        tasks.append(
            {
                "id": f"{suite}_{idx}",
                "suite": suite,
                "prompt": prompt,
                "answer": None,
                "prompt_chars": len(prompt),
                "metadata": {
                    "type": row.get("type"),
                    "number": row.get("number"),
                    "checks_once": row.get("checks_once", {}),
                    "checks_range": row.get("checks_range", {}),
                    "checks_periodic": row.get("checks_periodic", {}),
                    "prefix": row.get("prefix", ""),
                },
            }
        )
    return select_tasks(tasks, args)


def load_longgenbench_gsm8k(args) -> list[dict[str, Any]]:
    repo = PROJECT_ROOT / args.longgenbench_dominic_repo
    prompt_path = repo / "data/LongGenBench_GSM8K_prompt/LongGenBench_prompt.txt"
    if not prompt_path.exists():
        raise FileNotFoundError(f"LongGenBench GSM8K prompt not found: {prompt_path}")
    from datasets import load_dataset

    prompt_original = prompt_path.read_text(encoding="utf-8")
    gsm8k = load_dataset("gsm8k", "main", split="test")
    limit = min(int(args.longgenbench_gsm8k_question_limit), len(gsm8k))
    rows = [gsm8k[i] for i in range(limit)]
    tasks = []
    k = max(1, int(args.longgenbench_gsm8k_k))
    for start in range(0, len(rows), k):
        batch = rows[start : start + k]
        if not batch:
            continue
        prompt = "Examples:\n" + prompt_original + "\n\nFollowing Question:\n"
        for i, row in enumerate(batch, start=9):
            prompt += f"Question_{i}:\n{row['question']}\n"
        prompt += (
            "\nAnswer each question step by step, adhering to the format shown in the examples. "
            "Start each response with 'Answer_' and introduce the final response with 'The answer is'. "
            "Do not repeat the question. Answer all questions.\n"
        )
        tasks.append(
            {
                "id": f"longgenbench_gsm8k_{start}_{start + len(batch) - 1}",
                "suite": "longgenbench_gsm8k",
                "prompt": prompt,
                "answer": [extract_gsm8k_answer(row["answer"]) for row in batch],
                "prompt_chars": len(prompt),
                "metadata": {
                    "start": start,
                    "end": start + len(batch) - 1,
                    "k": len(batch),
                    "questions": [row["question"] for row in batch],
                    "raw_answers": [row["answer"] for row in batch],
                },
            }
        )
    return select_tasks(tasks, args)


def extract_gsm8k_answer(text: str) -> str:
    if "####" in text:
        return text.split("####")[-1].strip().replace(",", "")
    matches = ANSWER_INT_RE.findall(text.replace(",", ""))
    return matches[-1] if matches else ""


def load_tasks(args) -> list[dict[str, Any]]:
    if args.benchmark == "aime24":
        return load_aime24(args)
    if args.benchmark == "gpqa":
        return load_gpqa(args)
    if args.benchmark == "livecodebench_codegen":
        return load_livecodebench(args)
    if args.benchmark == "longgenbench_sgt_short":
        return load_longgenbench_sgt(args, long=False)
    if args.benchmark == "longgenbench_sgt_long":
        return load_longgenbench_sgt(args, long=True)
    if args.benchmark == "longgenbench_gsm8k":
        return load_longgenbench_gsm8k(args)
    raise ValueError(f"unsupported benchmark: {args.benchmark}")


def maybe_apply_chat_template(tokenizer, prompt: str, args) -> str:
    if not bool(args.use_chat_template):
        return prompt
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    if bool(args.disable_thinking):
        kwargs["enable_thinking"] = False
    try:
        return tokenizer.apply_chat_template([{"role": "user", "content": prompt}], **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template([{"role": "user", "content": prompt}], **kwargs)


def generation_kwargs(tokenizer, args, input_ids):
    kwargs = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "max_new_tokens": int(args.max_new_tokens),
        "pad_token_id": tokenizer.pad_token_id,
        "do_sample": bool(float(args.temperature) > 0.0),
    }
    if int(args.min_new_tokens) > 0:
        kwargs["min_new_tokens"] = int(args.min_new_tokens)
    if bool(args.force_max_new_tokens):
        kwargs["eos_token_id"] = None
        kwargs["forced_eos_token_id"] = None
    else:
        kwargs["eos_token_id"] = tokenizer.eos_token_id
    if float(args.temperature) > 0.0:
        kwargs["temperature"] = float(args.temperature)
        kwargs["top_p"] = float(args.top_p)
    return kwargs


def generate_one(model, tokenizer, task, args) -> dict[str, Any]:
    device = model_device(model)
    prompt = maybe_apply_chat_template(tokenizer, str(task["prompt"]), args)
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=not bool(args.use_chat_template))
    original_tokens = int(enc.input_ids.shape[1])
    input_ids, truncated = truncate_middle(enc.input_ids, int(args.max_input_tokens))
    input_ids = input_ids.to(device)
    kwargs = generation_kwargs(tokenizer, args, input_ids)

    start = time.time()
    output_ids = model.generate(**kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = float(time.time() - start)

    new_ids = output_ids[0, input_ids.shape[1] :].detach().cpu()
    response = tokenizer.decode(new_ids, skip_special_tokens=True)
    return {
        "id": task["id"],
        "suite": task["suite"],
        "response": response,
        "prompt_tokens": int(original_tokens),
        "used_prompt_tokens": int(input_ids.shape[1]),
        "truncated": bool(truncated),
        "generated_tokens": int(new_ids.numel()),
        "generation_sec": elapsed,
        "metadata": task.get("metadata", {}),
    }


def extract_aime_answer(text: str) -> str | None:
    boxed = BOXED_RE.findall(text or "")
    if boxed:
        matches = ANSWER_INT_RE.findall(boxed[-1].replace(",", ""))
        if matches:
            return matches[-1]
    matches = ANSWER_INT_RE.findall((text or "").replace(",", ""))
    return matches[-1] if matches else None


def extract_choice_answer(text: str) -> str | None:
    match = CHOICE_RE.search(text or "")
    if match:
        return match.group(1).upper()
    compact = (text or "").strip().upper()
    if compact in {"A", "B", "C", "D", "E"}:
        return compact
    paren = re.findall(r"\(([A-E])\)", text or "", flags=re.I)
    return paren[-1].upper() if paren else None


def extract_numbered_answers(text: str) -> list[str]:
    answers = []
    chunks = re.split(r"(?=Answer[_\s-]*\d+\s*:)", text or "", flags=re.I)
    for chunk in chunks:
        if not re.search(r"Answer[_\s-]*\d+\s*:", chunk, flags=re.I):
            continue
        if "The answer is" in chunk:
            tail = chunk.split("The answer is")[-1]
        else:
            tail = chunk
        matches = ANSWER_INT_RE.findall(tail.replace(",", ""))
        answers.append(matches[-1] if matches else "")
    if answers:
        return answers
    return ANSWER_INT_RE.findall((text or "").replace(",", ""))


def score_longgenbench_sgt(row: dict[str, Any]) -> dict[str, Any]:
    meta = row.get("metadata", {})
    text = str(meta.get("prefix", "")) + str(row["response"])
    blocks = text.split("#*#")
    word_count = len(text.split())
    number = int(meta.get("number") or 0)
    type_name = str(meta.get("type") or "")
    seen = set()
    type_re = re.compile(rf"{re.escape(type_name)}\s+(\d+)", re.I) if type_name else None
    for block in blocks:
        if type_re:
            match = type_re.search(block)
            if match:
                seen.add(int(match.group(1)))
    completion_rate = float(len(seen & set(range(1, number + 1))) / max(1, number))

    # Lightweight non-LLM check: substring presence in the expected block. Official
    # LongGenBench uses an LLM yes/no judge; this gives a cheap smoke signal only.
    def check_group(checks: dict[str, str]) -> tuple[int, int]:
        hits = 0
        total = 0
        for idx, desc in checks.items():
            total += 1
            target = int(idx)
            block = ""
            for candidate in blocks:
                if type_re and type_re.search(candidate):
                    m = type_re.search(candidate)
                    if m and int(m.group(1)) == target:
                        block = candidate
                        break
            if str(desc).lower() in block.lower():
                hits += 1
        return hits, total

    once_hits, once_total = check_group(meta.get("checks_once", {}))
    range_hits, range_total = check_group(meta.get("checks_range", {}))
    periodic_hits, periodic_total = check_group(meta.get("checks_periodic", {}))
    return {
        "word_count": int(word_count),
        "block_count": int(max(0, len(blocks) - 1)),
        "expected_blocks": int(number),
        "completion_rate": completion_rate,
        "substring_once_acc": float(once_hits / once_total) if once_total else None,
        "substring_range_acc": float(range_hits / range_total) if range_total else None,
        "substring_periodic_acc": float(periodic_hits / periodic_total) if periodic_total else None,
        "note": "substring_* metrics are smoke checks; use LongGenBench LLM judge for paper numbers.",
    }


def score_rows(rows: list[dict[str, Any]], tasks: list[dict[str, Any]], args) -> list[dict[str, Any]]:
    task_by_id = {task["id"]: task for task in tasks}
    scored = []
    for row in rows:
        task = task_by_id[row["id"]]
        item = dict(row)
        if task["suite"] == "aime24":
            pred = extract_aime_answer(row["response"])
            item.update({"pred": pred, "answer": task["answer"], "judge": bool(pred == task["answer"])})
        elif task["suite"] == "gpqa":
            pred = extract_choice_answer(row["response"])
            item.update({"pred": pred, "answer": task["answer"], "judge": bool(pred == task["answer"])})
        elif task["suite"] == "longgenbench_gsm8k":
            preds = extract_numbered_answers(row["response"])
            answers = list(task["answer"])
            correct = sum(1 for pred, ans in zip(preds, answers, strict=False) if pred == ans)
            item.update(
                {
                    "pred": preds,
                    "answer": answers,
                    "correct_count": int(correct),
                    "question_count": int(len(answers)),
                    "judge": bool(correct == len(answers)),
                    "accuracy": float(correct / max(1, len(answers))),
                }
            )
        elif task["suite"].startswith("longgenbench_sgt_"):
            item.update(score_longgenbench_sgt({**row, "metadata": task.get("metadata", {})}))
        scored.append(item)
    if args.benchmark == "livecodebench_codegen" and bool(args.evaluate_code):
        add_livecodebench_path(args)
        if not hasattr(sys, "set_int_max_str_digits"):
            # LiveCodeBench calls this Python 3.10.7+ API during import. The
            # cluster module is 3.10.4, where the integer-string limit is absent.
            sys.set_int_max_str_digits = lambda *_args, **_kwargs: None  # type: ignore[attr-defined]
        from lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics
        from lcb_runner.lm_styles import LMStyle
        from lcb_runner.utils.extraction_utils import extract_code

        samples = []
        generations = []
        for task, row in zip(tasks, scored, strict=False):
            problem = task["_lcb_problem"]
            code = extract_code(row["response"], LMStyle.LLaMa3)
            if not code:
                code = extract_code(row["response"], LMStyle.GenericBase)
            row["code"] = code
            samples.append(problem.get_evaluation_sample())
            generations.append([code])
        metrics, results, metadata = codegen_metrics(
            samples,
            generations,
            k_list=[1],
            num_process_evaluate=int(args.code_eval_processes),
            timeout=int(args.code_eval_timeout),
            debug=False,
        )
        for idx, row in enumerate(scored):
            row["code_eval_result"] = results.get(idx)
            row["code_eval_metadata"] = metadata[idx] if idx < len(metadata) else None
        for row in scored:
            row["_livecodebench_metrics"] = metrics
    return scored


def aggregate(rows: list[dict[str, Any]], args, approx_stats: dict[int, ApproxStats] | None = None) -> dict[str, Any]:
    total = len(rows)
    summary: dict[str, Any] = {
        "benchmark": args.benchmark,
        "attention_mode": args.attention_mode,
        "approx_prefill": bool(args.approx_prefill),
        "num_examples": int(total),
        "model_name": args.model_name,
        "max_input_tokens": int(args.max_input_tokens),
        "max_new_tokens": int(args.max_new_tokens),
        "min_new_tokens": int(args.min_new_tokens),
        "force_max_new_tokens": bool(args.force_max_new_tokens),
        "temperature": float(args.temperature),
        "avg_prompt_tokens": float(sum(row["prompt_tokens"] for row in rows) / total) if total else 0.0,
        "avg_used_prompt_tokens": float(sum(row["used_prompt_tokens"] for row in rows) / total) if total else 0.0,
        "avg_generated_tokens": float(sum(row["generated_tokens"] for row in rows) / total) if total else 0.0,
        "max_generated_tokens": int(max((row["generated_tokens"] for row in rows), default=0)),
        "avg_generation_sec": float(sum(row["generation_sec"] for row in rows) / total) if total else 0.0,
        "truncated_count": int(sum(1 for row in rows if row.get("truncated"))),
    }
    judged = [row for row in rows if "judge" in row]
    if judged:
        summary["accuracy"] = float(sum(1 for row in judged if row["judge"]) / len(judged))
        summary["accuracy_pct"] = float(100.0 * summary["accuracy"])
    if args.benchmark == "longgenbench_gsm8k":
        total_q = sum(int(row.get("question_count", 0)) for row in rows)
        total_correct = sum(int(row.get("correct_count", 0)) for row in rows)
        summary["subquestion_accuracy"] = float(total_correct / max(1, total_q))
        summary["subquestion_accuracy_pct"] = float(100.0 * summary["subquestion_accuracy"])
    if args.benchmark.startswith("longgenbench_sgt_"):
        for key in ("completion_rate", "substring_once_acc", "substring_range_acc", "substring_periodic_acc"):
            vals = [row[key] for row in rows if row.get(key) is not None]
            if vals:
                summary[f"mean_{key}"] = float(sum(float(v) for v in vals) / len(vals))
    if args.benchmark == "livecodebench_codegen":
        metrics = next((row.get("_livecodebench_metrics") for row in rows if row.get("_livecodebench_metrics")), None)
        if metrics:
            summary["livecodebench_metrics"] = metrics
            if "pass@1" in metrics:
                summary["pass_at_1"] = float(metrics["pass@1"])
    if approx_stats is not None:
        summary["cost_proxy"] = summarize_approx_stats(approx_stats)
        summary["cost_proxy_aggregate"] = aggregate_approx_stats(approx_stats)
    return summary


def strip_unserializable_task(task: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in task.items() if not k.startswith("_")}


def jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if hasattr(obj, "item") and callable(obj.item):
        try:
            return obj.item()
        except Exception:
            pass
    if isinstance(obj, Path):
        return str(obj)
    return obj


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

    tasks = load_tasks(args)
    (output_dir / "selected_tasks.json").write_text(
        json.dumps(jsonable([strip_unserializable_task(task) for task in tasks]), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if bool(args.dry_run):
        summary = {
            "benchmark": args.benchmark,
            "dry_run": True,
            "num_examples": len(tasks),
            "avg_prompt_chars": float(sum(t["prompt_chars"] for t in tasks) / max(1, len(tasks))),
            "max_prompt_chars": int(max((t["prompt_chars"] for t in tasks), default=0)),
        }
        (output_dir / "summary.json").write_text(json.dumps(jsonable(summary), indent=2, sort_keys=True), encoding="utf-8")
        print("[public_longdecode_eval] summary=" + json.dumps(jsonable(summary), sort_keys=True))
        return

    config = AutoConfig.from_pretrained(
        args.model_name,
        trust_remote_code=bool(args.trust_remote_code),
        local_files_only=bool(args.local_files_only),
    )
    maybe_apply_qwen_yarn(config, args)
    tokenizer = load_tokenizer(args)
    dtype = dtype_from_name(args.dtype)
    model, auto_class = load_hf_model(args, dtype, config)
    setattr(args, "approx_prefill", bool(args.approx_prefill) and str(args.attention_mode) == "pagedpq")

    approx_stats: dict[int, ApproxStats] = {}
    layer_ids = parse_layer_ids(str(args.layers), model) if str(args.attention_mode) == "pagedpq" else []
    attention_context = (
        patched_paged_pq_attention(model, layer_ids, args, approx_stats)
        if str(args.attention_mode) == "pagedpq"
        else nullcontext()
    )

    rows = []
    out_path = output_dir / "predictions.jsonl"
    with attention_context, out_path.open("w", encoding="utf-8") as fout:
        for task in tqdm(tasks):
            if str(args.attention_mode) == "pagedpq":
                reset_paged_pq_attention_state(model)
            pred = generate_one(model, tokenizer, task, args)
            rows.append(pred)
            fout.write(json.dumps(jsonable(pred), ensure_ascii=False) + "\n")
            fout.flush()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    scored = score_rows(rows, tasks, args)
    scored_path = output_dir / "scored_predictions.jsonl"
    with scored_path.open("w", encoding="utf-8") as fout:
        for row in scored:
            clean = {k: v for k, v in row.items() if not k.startswith("_")}
            fout.write(json.dumps(jsonable(clean), ensure_ascii=False) + "\n")

    summary = aggregate(scored, args, approx_stats if str(args.attention_mode) == "pagedpq" else None)
    summary["auto_class"] = auto_class
    summary["layers"] = layer_ids
    summary["pagedpq_config"] = pagedpq_config(args)
    (output_dir / "summary.json").write_text(json.dumps(jsonable(summary), indent=2, sort_keys=True), encoding="utf-8")
    print("[public_longdecode_eval] summary=" + json.dumps(jsonable(summary), sort_keys=True))


if __name__ == "__main__":
    main()
