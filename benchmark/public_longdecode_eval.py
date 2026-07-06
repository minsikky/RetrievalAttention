from __future__ import annotations

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


BENCHMARK_CHOICES = (
    "aime24",
    "gpqa",
    "helmet_longqa",
    "helmet_rag",
    "helmet_recall",
    "livecodebench_codegen",
    "longproc_2k",
    "longproc_8k",
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
    parser.add_argument(
        "--task_offset",
        type=int,
        default=0,
        help="Skip this many selected tasks before applying max_examples; used for Slurm sharding.",
    )
    parser.add_argument("--selection", choices=["first", "random", "shortest"], default="first")
    parser.add_argument("--max_input_tokens", type=int, default=120000)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--min_new_tokens", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--force_max_new_tokens", action="store_true")
    parser.add_argument("--dry_run", action="store_true", help="Load/select tasks and write args without loading a model.")

    # HELMET controls. RAG/Recall require the official HELMET data directory
    # prepared from princeton-nlp/HELMET's data.tar.gz.
    parser.add_argument("--helmet_repo", type=str, default="third_party/benchmarks/HELMET")
    parser.add_argument("--helmet_data_dir", type=str, default="third_party/benchmarks/HELMET/data")
    parser.add_argument(
        "--helmet_dataset_filter",
        type=str,
        default="",
        help="Comma-separated HELMET dataset names to keep within a category, e.g. kilt_nq or infbench_qa_eng_130862.",
    )

    # LongProc controls.
    parser.add_argument("--longproc_repo", type=str, default="third_party/benchmarks/LongProc")
    parser.add_argument("--longproc_data_dir", type=str, default="third_party/benchmarks/LongProc/data")
    parser.add_argument(
        "--longproc_datasets",
        type=str,
        default="",
        help="Comma-separated LongProc datasets. Defaults depend on longproc_2k vs longproc_8k.",
    )

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
            "joint_kv_stability",
            "pq_proxy_mass_budget",
            "pq_ranked_mass_budget",
        ],
        default="joint_kv_stability",
    )
    parser.add_argument("--tail_score_calibration", choices=["none", "affine_selected"], default="none")
    parser.add_argument("--tail_probe_rel_l2_max", type=float, default=float("inf"))
    parser.add_argument("--tail_proxy_mass_min", type=float, default=0.97)
    parser.add_argument("--tail_proxy_mass_max", type=float, default=1.0)
    parser.add_argument("--tail_pq_corr_min", type=float, default=-1.0)
    parser.add_argument("--tail_pq_relrmse_max", type=float, default=float("inf"))
    parser.add_argument("--ranked_confidence_cost_mode", choices=["exact", "upper_bound"], default="exact")
    parser.add_argument(
        "--exact_logit_backend",
        choices=["auto", "ranked_gather", "dense_sim"],
        default=os.environ.get("FRONTIER_EXACT_LOGIT_BACKEND", "auto"),
    )
    parser.add_argument("--geometric_min_budget", type=int, default=8)
    parser.add_argument("--geometric_max_budget", type=int, default=512)
    parser.add_argument("--geometric_growth", type=float, default=1.5)
    parser.add_argument("--geometric_probe_scale", type=float, default=1.5)
    parser.add_argument("--geometric_budget_granularity", type=int, default=8)
    parser.add_argument(
        "--joint_kv_policy",
        choices=[
            "k_first_priority",
            "v_first_priority",
            "k_first_alternating",
            "v_first_alternating",
            "sensitivity_greedy",
        ],
        default="k_first_alternating",
    )
    parser.add_argument("--joint_kv_k_budgets", default="4096,8192,14336,32768")
    parser.add_argument("--joint_kv_v_budgets", default="1024,2048,4096,6144,8192,12288,16384")
    parser.add_argument("--joint_kv_k_budget_fracs", default="0.10,0.30,0.50,0.70,0.90,1.0")
    parser.add_argument("--joint_kv_v_budget_fracs", default="0.05,0.10,0.20,0.40,0.60,0.80,1.0")
    parser.add_argument("--joint_kv_stability_threshold", type=float, default=0.002)
    parser.add_argument("--joint_kv_threshold_mode", choices=["fixed", "budget_delta_frac"], default="budget_delta_frac")
    parser.add_argument("--joint_kv_threshold_reference_frac", type=float, default=0.2)
    parser.add_argument("--joint_kv_threshold_scale_shape", choices=["linear", "sqrt", "log"], default="sqrt")
    parser.add_argument("--joint_kv_threshold_min_scale", type=float, default=0.0)
    parser.add_argument("--joint_kv_threshold_max_scale", type=float, default=1.5)
    parser.add_argument("--joint_kv_start_strategy", default="proxy_mass_m0p9")
    parser.add_argument("--selected_value_mode", choices=["exact", "vpq_value"], default="vpq_value")
    parser.add_argument(
        "--selected_value_exact_rule",
        choices=[
            "fixed",
            "selector_rank",
            "selected_mass",
            "selected_risk_mass",
            "selected_mass_or_risk",
            "global_residual_risk",
        ],
        default="global_residual_risk",
    )
    parser.add_argument("--selected_value_exact_top", type=int, default=0)
    parser.add_argument("--selected_value_exact_mass", type=float, default=0.0)
    parser.add_argument("--selected_value_exact_risk_mass", type=float, default=0.0)
    parser.add_argument("--selected_value_min_exact_top", type=int, default=0)
    parser.add_argument("--selected_value_max_exact_top", type=int, default=0)
    parser.add_argument("--selected_value_exact_all_context_max", type=int, default=0)
    parser.add_argument("--selected_value_exact_all_fraction_min", type=float, default=0.0)
    parser.add_argument("--selected_value_residual_norm_bytes", type=int, default=2)
    parser.add_argument("--value_code_stat_bytes", type=int, default=2)
    parser.add_argument("--tail_blend", type=float, default=1.0)
    parser.add_argument("--prefill_tail_blend", type=float, default=None)
    parser.add_argument("--decode_tail_blend", type=float, default=None)
    parser.add_argument("--tail_off_heads", default="")
    parser.add_argument("--static_prefix", type=int, default=128)
    parser.add_argument("--static_suffix", type=int, default=128)
    parser.add_argument("--page_size", type=int, default=5632)
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
    offset = max(0, int(getattr(args, "task_offset", 0)))
    limit = max(0, int(args.max_examples))
    return tasks[offset : offset + limit]


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
    except Exception as exc:
        message = str(exc)
        fallback_markers = (
            "Dataset scripts are no longer supported",
            "trust_remote_code",
            "is not supported anymore",
        )
        if not any(marker in message for marker in fallback_markers):
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
    from huggingface_hub import hf_hub_download

    filename = livecodebench_filename(str(args.livecodebench_release))
    path = hf_hub_download(
        repo_id="livecodebench/code_generation_lite",
        filename=filename,
        repo_type="dataset",
        local_files_only=bool(args.local_files_only),
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


def split_csv(value: str) -> list[str]:
    return [part.strip() for part in str(value or "").split(",") if part.strip()]


def project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def interleave_task_groups(groups: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    tasks = []
    max_len = max((len(group) for group in groups), default=0)
    for idx in range(max_len):
        for group in groups:
            if idx < len(group):
                tasks.append(group[idx])
    return tasks


def normalize_answer(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


def qa_metrics(prediction: str, answers: Any) -> dict[str, float]:
    if isinstance(answers, str):
        answer_list = [answers]
    elif answers and isinstance(answers[0], list):
        answer_list = [item for group in answers for item in group]
    else:
        answer_list = list(answers or [])
    pred_norm = normalize_answer(prediction)
    exact = 0.0
    sub_em = 0.0
    best_f1 = 0.0
    for answer in answer_list:
        ans_norm = normalize_answer(str(answer))
        if not ans_norm:
            continue
        exact = max(exact, float(pred_norm == ans_norm))
        sub_em = max(sub_em, float(ans_norm in pred_norm))
        pred_toks = pred_norm.split()
        ans_toks = ans_norm.split()
        common = {}
        for tok in pred_toks:
            common[tok] = min(pred_toks.count(tok), ans_toks.count(tok))
        overlap = sum(common.values())
        if overlap:
            precision = overlap / max(1, len(pred_toks))
            recall = overlap / max(1, len(ans_toks))
            best_f1 = max(best_f1, (2.0 * precision * recall) / max(1e-12, precision + recall))
    return {
        "exact_match": exact,
        "f1": best_f1,
        "substring_exact_match": sub_em,
    }


def parse_answer_prefixed_output(text: str) -> str:
    match = re.search(r"(?:answer\s*:)(.*?)(?:\n|$)", text or "", flags=re.I | re.S)
    if match:
        return match.group(1).strip()
    return (text or "").strip().splitlines()[0].strip() if (text or "").strip() else ""


def helmet_path(args, configured_path: str) -> Path:
    raw = str(configured_path or "").strip().strip("'\"")
    if not raw:
        return Path("")
    path = Path(raw)
    if path.is_absolute():
        return path
    repo = project_path(args.helmet_repo)
    data_dir = project_path(args.helmet_data_dir)
    candidates = [repo / path]
    if path.parts and path.parts[0] == "data":
        candidates.append(data_dir / Path(*path.parts[1:]))
    candidates.append(data_dir / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    if path.parts and path.parts[0] == "data" and len(path.parts) >= 2:
        parent = data_dir / Path(*path.parts[1:-1])
        pattern = re.sub(r"_k\d+_", "_k*_", path.name)
        matches = sorted(parent.glob(pattern))
        if matches:
            return matches[-1]
    return candidates[0]


def read_helmet_config(args, name: str) -> dict[str, Any]:
    import yaml

    path = project_path(args.helmet_repo) / "configs" / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"HELMET config not found: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def helmet_category_config(args) -> tuple[str, dict[str, Any], list[str]]:
    if args.benchmark == "helmet_rag":
        return "rag", read_helmet_config(args, "rag"), ["kilt_nq"]
    if args.benchmark == "helmet_recall":
        return "recall", read_helmet_config(args, "recall"), ["ruler_niah_mk_2"]
    if args.benchmark == "helmet_longqa":
        # Use InfiniteBench QA as the default LongQA representative. It keeps
        # this small validation suite practical and avoids pulling the full
        # NarrativeQA path unless explicitly requested later.
        return "longqa", read_helmet_config(args, "longqa"), ["infbench_qa_eng_130862"]
    raise ValueError(f"unsupported HELMET benchmark: {args.benchmark}")


def helmet_config_rows(config: dict[str, Any]) -> list[dict[str, str]]:
    datasets = split_csv(config.get("datasets", ""))
    test_files = split_csv(config.get("test_files", ""))
    demo_files = split_csv(config.get("demo_files", ""))
    gen_lengths = split_csv(config.get("generation_max_length", ""))
    rows = []
    for idx, dataset in enumerate(datasets):
        rows.append(
            {
                "dataset": dataset,
                "test_file": test_files[idx] if idx < len(test_files) else "",
                "demo_file": demo_files[idx] if idx < len(demo_files) else "",
                "generation_max_length": gen_lengths[idx] if idx < len(gen_lengths) else "",
            }
        )
    return rows


def format_helmet_documents(ctxs: list[dict[str, Any]]) -> str:
    docs = []
    for ctx in ctxs:
        title = str(ctx.get("title", "")).strip()
        text = str(ctx.get("text", "")).strip()
        docs.append(f"Document (Title: {title}): {text}" if title else f"Document: {text}")
    return "\n\n".join(docs)


def read_helmet_json_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"HELMET data file not found: {path}. Prepare HELMET data with "
            "`bash third_party/benchmarks/HELMET/scripts/download_data.sh` from the HELMET repo, "
            "or set HELMET_DATA_DIR to an existing unpacked data directory."
        )
    if path.suffix == ".jsonl":
        return read_jsonl(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "data" in payload:
        return list(payload["data"])
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unsupported HELMET data format: {path}")


def build_helmet_qa_demo(row: dict[str, Any]) -> str:
    answer = row.get("answer")
    if answer is None:
        answers = row.get("answers") or []
        answer = answers[0] if answers else ""
    return (
        f"{format_helmet_documents(list(row.get('ctxs') or []))}\n\n"
        f"Question: {row.get('question', '')}\n"
        f"Answer: {answer}"
    )


def load_helmet_rag_dataset(args, dataset: str, test_file: str, demo_file: str) -> list[dict[str, Any]]:
    rows = read_helmet_json_rows(helmet_path(args, test_file))
    demos = []
    demo_path = helmet_path(args, demo_file)
    if str(demo_file).strip().strip("'\"") and demo_path.exists():
        demos = read_helmet_json_rows(demo_path)[: max(0, int(getattr(args, "shots", 2)))]
    demo_text = ("\n\n".join(build_helmet_qa_demo(row) for row in demos) + "\n\n") if demos else ""
    tasks = []
    for idx, row in enumerate(rows):
        context = format_helmet_documents(list(row.get("ctxs") or []))
        prompt = (
            "Use the given documents to write a concise and short answer to the question.\n"
            "Write your answer in the following format:\n"
            "Answer: [answer]\n\n"
            f"{demo_text}{context}\n\n"
            f"Question: {row.get('question', '')}\n"
            "Answer:"
        )
        tasks.append(
            {
                "id": f"helmet_rag_{dataset}_{row.get('id', idx)}",
                "suite": "helmet_rag",
                "prompt": prompt,
                "answer": row.get("answers") or row.get("answer") or [],
                "prompt_chars": len(prompt),
                "metadata": {"helmet_dataset": dataset, "metric_family": "helmet_qa"},
            }
        )
    return tasks


def load_helmet_recall_dataset(args, dataset: str, test_file: str) -> list[dict[str, Any]]:
    rows = read_helmet_json_rows(helmet_path(args, test_file))
    tasks = []
    for idx, row in enumerate(rows):
        if dataset == "json_kv":
            demos = row.get("demos") or []
            demo_text = "\n\n".join(
                f"Key: {key}\nCorresponding value: {value}" for key, value in demos[:2]
            )
            if demo_text:
                demo_text += "\n\n"
            question = row.get("question") or row.get("key") or row.get("query") or ""
            prompt = (
                f"{row.get('context', '')}\n\n"
                "Extract the value corresponding to the specified key in the JSON object below.\n\n"
                f"{demo_text}Key: {question}\n"
                "Corresponding value:"
            )
            answer = row.get("answer") or row.get("value") or []
            metric_family = "helmet_qa"
        else:
            type_needle = row.get("type_needle_v", "value")
            query = row.get("query") or row.get("question") or ""
            if "niah_mv" in dataset or "niah_mq" in dataset:
                prompt = (
                    f"Some special magic {type_needle} are hidden within the following text. "
                    f"Make sure to memorize it. I will quiz you about the {type_needle} afterwards.\n"
                    f"{row.get('context', '')}\n"
                    f"What are all the special magic {type_needle} for {query} mentioned in the provided text?\n"
                    f"The special magic {type_needle} for {query} mentioned in the provided text are"
                )
            else:
                prompt = (
                    f"A special magic {type_needle} is hidden within the following text. "
                    f"Make sure to memorize it. I will quiz you about the {type_needle} afterwards.\n"
                    f"{row.get('context', '')}\n"
                    f"What is the special magic {type_needle} for {query} mentioned in the provided text?\n"
                    f"The special magic {type_needle} for {query} mentioned in the provided text is"
                )
            answer = row.get("answer") or row.get("outputs") or []
            metric_family = "helmet_recall"
        tasks.append(
            {
                "id": f"helmet_recall_{dataset}_{row.get('id', idx)}",
                "suite": "helmet_recall",
                "prompt": prompt,
                "answer": answer,
                "prompt_chars": len(prompt),
                "metadata": {"helmet_dataset": dataset, "metric_family": metric_family},
            }
        )
    return tasks


def load_helmet_longqa_dataset(args, dataset: str) -> list[dict[str, Any]]:
    if not dataset.startswith("infbench_"):
        raise NotImplementedError(
            f"Local HELMET LongQA harness currently supports InfiniteBench QA/choice datasets, got {dataset!r}."
        )
    if "qa_eng" in dataset:
        filename = "longbook_qa_eng.jsonl"
        metric_family = "helmet_qa"
    elif "choice_eng" in dataset:
        filename = "longbook_choice_eng.jsonl"
        metric_family = "helmet_choice"
    else:
        raise NotImplementedError(f"Unsupported HELMET LongQA dataset: {dataset}")
    local_path = project_path(args.helmet_data_dir) / "infbench" / filename
    if not local_path.exists():
        from huggingface_hub import hf_hub_download

        local_path = Path(
            hf_hub_download(
                repo_id="xinrongzhang2022/InfiniteBench",
                filename=filename,
                repo_type="dataset",
                local_files_only=bool(args.local_files_only),
            )
        )
    rows = read_jsonl(local_path)
    tasks = []
    for idx, row in enumerate(rows):
        if metric_family == "helmet_choice":
            options = list(row["options"])
            labels = "ABCD"
            option_text = "\n".join(f"{labels[i]}. {option}" for i, option in enumerate(options))
            answer_idx = options.index(row["answer"][0])
            answer = labels[answer_idx]
            prompt = (
                "You are given a story and a question with multiple choices. Choose the best answer from the options provided. "
                "Only one of the following options is correct, output the answer using one single letter (A, B, C, or D). "
                "Don't say anything else.\n\n"
                f"{row['context']}\n\nQuestion: {row['input']}\nOptions:\n{option_text}\n\nAnswer:"
            )
        else:
            answer = list(row["answer"])
            prompt = (
                "You are given a story and a question. Answer the question as concisely as you can, using a single phrase if possible.\n\n"
                f"{row['context']}\n\nQuestion: {row['input']}\n\nAnswer:"
            )
        tasks.append(
            {
                "id": f"helmet_longqa_{dataset}_{row.get('id', idx)}",
                "suite": "helmet_longqa",
                "prompt": prompt,
                "answer": answer,
                "prompt_chars": len(prompt),
                "metadata": {"helmet_dataset": dataset, "metric_family": metric_family},
            }
        )
    return tasks


def load_helmet(args) -> list[dict[str, Any]]:
    category, config, default_filter = helmet_category_config(args)
    requested = set(split_csv(args.helmet_dataset_filter) or default_filter)
    groups = []
    for row in helmet_config_rows(config):
        dataset = row["dataset"]
        if dataset not in requested:
            continue
        if category == "rag":
            groups.append(load_helmet_rag_dataset(args, dataset, row["test_file"], row["demo_file"]))
        elif category == "recall":
            groups.append(load_helmet_recall_dataset(args, dataset, row["test_file"]))
        elif category == "longqa":
            groups.append(load_helmet_longqa_dataset(args, dataset))
    if not groups:
        raise ValueError(f"No HELMET datasets selected for {args.benchmark}; requested={sorted(requested)}")
    return select_tasks(interleave_task_groups(groups), args)


def default_longproc_datasets(args) -> list[str]:
    if args.longproc_datasets:
        return split_csv(args.longproc_datasets)
    if args.benchmark == "longproc_2k":
        return [
            "path_traversal_2k",
            "countdown_2k",
            "tom_tracking_2k",
            "travel_planning_2k",
        ]
    if args.benchmark == "longproc_8k":
        return [
            "path_traversal_8k",
            "countdown_8k",
            "tom_tracking_8k",
            "travel_planning_8k",
        ]
    raise ValueError(f"unsupported LongProc benchmark: {args.benchmark}")


def load_longproc(args) -> list[dict[str, Any]]:
    repo = project_path(args.longproc_repo)
    if not repo.exists():
        raise FileNotFoundError(f"LongProc repo not found: {repo}")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    data_dir = project_path(args.longproc_data_dir)
    groups = []
    for dataset in default_longproc_datasets(args):
        rows, eval_func = load_longproc_dataset_lightweight(repo, data_dir, dataset)
        group = []
        for idx, row in enumerate(rows):
            prompt = str(row["input_prompt"])
            group.append(
                {
                    "id": f"{args.benchmark}_{dataset}_{idx}",
                    "suite": args.benchmark,
                    "prompt": prompt,
                    "answer": row.get("reference_output"),
                    "prompt_chars": len(prompt),
                    "metadata": {"longproc_dataset": dataset},
                    "_longproc_eval_func": eval_func,
                    "_longproc_example": row,
                }
            )
        groups.append(group)
    return select_tasks(interleave_task_groups(groups), args)


def read_longproc_prompt(data_dir: Path, task_name: str) -> str:
    import yaml

    prompt_path = data_dir / task_name / "prompts.yaml"
    if not prompt_path.exists():
        raise FileNotFoundError(f"LongProc prompt file not found: {prompt_path}")
    with prompt_path.open("r", encoding="utf-8") as f:
        return str(yaml.safe_load(f)["USER_PROMPT"])


def extract_tagged_text(text: str, tag: str) -> str | None:
    start = text.find(f"<{tag}>")
    end = text.find(f"</{tag}>")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start + len(tag) + 2 : end].strip()


def eval_longproc_path_traversal(prediction: str, example: dict) -> tuple[dict[str, float], dict[str, Any]]:
    gt = str(example["reference_output"]).strip()
    parsed = extract_tagged_text(prediction, "Route")
    if parsed is None:
        return {"accuracy": 0.0, "partial_accuracy": 0.0, "extraction_rate": 0.0}, {"parsed_output": None, "error_report": "Parsing error"}
    parsed = parsed.strip()
    if parsed == gt:
        return {"accuracy": 1.0, "partial_accuracy": 1.0, "extraction_rate": 1.0}, {"parsed_output": parsed, "error_report": None}
    gt_lines = gt.splitlines()
    pred_lines = parsed.splitlines()
    first_diff = 0
    for first_diff, (gt_line, pred_line) in enumerate(zip(gt_lines, pred_lines, strict=False)):
        if gt_line != pred_line:
            break
    else:
        first_diff = min(len(gt_lines), len(pred_lines))
    partial = float(first_diff / max(1, len(gt_lines)))
    return {
        "accuracy": 0.0,
        "partial_accuracy": partial,
        "extraction_rate": 1.0,
    }, {"parsed_output": parsed, "error_report": {"line": first_diff}}


def load_longproc_dataset_lightweight(repo: Path, data_dir: Path, dataset: str) -> tuple[list[dict[str, Any]], Any]:
    task_name = dataset.rsplit("_", 1)[0]
    prompt = read_longproc_prompt(data_dir, task_name)
    task_dir = data_dir / task_name
    if dataset.startswith("path_traversal_"):
        rows = json.loads((task_dir / f"{dataset}.json").read_text(encoding="utf-8"))
        data = [
            {
                "input_prompt": prompt.format(
                    city_context=row["context_nl"],
                    src_city=row["question_repr"][0],
                    dst_city=row["question_repr"][1],
                ),
                "reference_output": row["answer_nl"],
                "item": row,
            }
            for row in rows
        ]
        return data, eval_longproc_path_traversal
    if dataset.startswith("tom_tracking_"):
        from longproc.tom_tracking_evaluator import evaluate_tom_trace

        def eval_tom(prediction: str, example: dict) -> tuple[dict[str, float], dict[str, Any]]:
            parsed_pred = "\n".join(line for line in prediction.splitlines() if line.strip().startswith("-"))
            parsed_gt = "\n".join(line for line in str(example["reference_output"]).splitlines() if line.strip().startswith("-"))
            strict_acc, partial_acc, error_report = evaluate_tom_trace(parsed_pred, parsed_gt)
            return {
                "accuracy": float(strict_acc),
                "partial_accuracy": float(partial_acc),
                "extraction_rate": 1.0,
            }, {"parsed_output": parsed_pred, "error_report": error_report}

        rows = json.loads((task_dir / f"{dataset}.json").read_text(encoding="utf-8"))
        data = [
            {
                "input_prompt": prompt.format(story=row["story"], question=row["question"]),
                "reference_output": row["solution"],
                "item": row,
            }
            for row in rows
        ]
        return data, eval_tom
    if dataset.startswith("countdown_"):
        from longproc.countdown_evaluator import (
            build_countdown_demonstration,
            evaluate_countdown_final_solution,
            evaluate_countdown_search_procedure,
        )

        def build_icl_demonstration() -> str:
            demos = [{"nums": [40, 19, 23, 7], "target": 29}, {"nums": [9, 16, 6, 18], "target": 12}]
            parts = []
            for demo in demos:
                _, demonstration = build_countdown_demonstration(demo["nums"], demo["target"])
                parts.append(f"# Example\nNumbers: {demo['nums']}\nTarget: {demo['target']}\n\n{demonstration}")
            return "\n\n".join(parts)

        user_prompt = prompt.format(demonstration=build_icl_demonstration(), nums="{nums}", target="{target}")

        def eval_countdown(prediction: str, example: dict) -> tuple[dict[str, float], dict[str, Any]]:
            item = example["item"]
            pred_solution = extract_tagged_text(prediction, "Solution")
            nums = list(item["nums"])
            target = int(item["target"])
            if pred_solution is not None and evaluate_countdown_final_solution(nums, target, pred_solution):
                return {"accuracy": 1.0, "partial_accuracy": 1.0, "extraction_rate": 1.0}, {"parsed_output": pred_solution}
            extraction_rate = 1.0 if pred_solution is not None else 0.0
            if "# Search Procedure" not in prediction:
                return {"accuracy": 0.0, "partial_accuracy": 0.0, "extraction_rate": extraction_rate}, {"parsed_output": pred_solution}
            pred_procedure = prediction.split("# Search Procedure")[-1].strip()
            gt_procedure = str(example["reference_output"]).split("# Search Procedure")[-1].split("Now we have found the target")[0].strip()
            partial, error_report = evaluate_countdown_search_procedure(nums, target, pred_procedure, gt_procedure)
            return {
                "accuracy": 0.0,
                "partial_accuracy": float(partial),
                "extraction_rate": extraction_rate,
            }, {"parsed_output": pred_solution, "error_report": error_report}

        rows = json.loads((task_dir / f"{dataset}.json").read_text(encoding="utf-8"))
        data = []
        for row in rows:
            solution, demonstration = build_countdown_demonstration(list(row["nums"]), int(row["target"]))
            data.append(
                {
                    "input_prompt": user_prompt.format(nums=row["nums"], target=row["target"]),
                    "reference_output": demonstration,
                    "item": {**row, "solution": solution, "reference_output": demonstration},
                }
            )
        return data, eval_countdown
    if dataset.startswith("travel_planning_"):
        from longproc.travel_planning_evaluator import (
            build_travel_plan_demonstration,
            evaluate_travel_plan_search_procedure,
            evaluate_travel_plan_solution,
        )

        def build_icl_demonstration() -> str:
            demo_path = task_dir / "travel_planning_icl_examples.json"
            demos = json.loads(demo_path.read_text(encoding="utf-8"))
            return "\n\n".join(
                f"# Example\n{row['disambig_question_text']}\n\n{build_travel_plan_demonstration(row)}"
                for row in demos
            )

        user_prompt = prompt.format(demonstration=build_icl_demonstration(), problem="{problem}")

        def eval_travel(prediction: str, example: dict) -> tuple[dict[str, float], dict[str, Any]]:
            item = example["item"]
            plan_text = extract_tagged_text(prediction, "Plan")
            extraction_rate = 1.0 if plan_text is not None else 0.0
            accuracy = (
                evaluate_travel_plan_solution(item["ground_truth_cities"], item["ground_truth_durations"], plan_text.strip())
                if plan_text is not None
                else 0.0
            )
            if accuracy == 1.0:
                partial = 1.0
                error_report = None
            else:
                partial, error_report = evaluate_travel_plan_search_procedure(item, prediction, example["reference_output"])
            return {
                "accuracy": float(accuracy),
                "partial_accuracy": float(partial),
                "extraction_rate": extraction_rate,
            }, {"parsed_output": plan_text, "error_report": error_report}

        output_range = (0, 2048) if dataset.endswith("_2k") else (4096, 8192)
        rows = json.loads((task_dir / "travel_planning_all.json").read_text(encoding="utf-8"))
        rows = [row for row in rows if output_range[0] <= int(row["estimated_output_tokens"]) < output_range[1]]
        data = []
        for row in rows:
            reference = build_travel_plan_demonstration(row)
            data.append(
                {
                    "input_prompt": user_prompt.format(problem=row["disambig_question_text"]),
                    "reference_output": reference,
                    "item": {
                        **row,
                        "problem": row["disambig_question_text"],
                        "reference_output": reference,
                    },
                }
            )
        return data, eval_travel
    raise NotImplementedError(
        f"LongProc dataset {dataset!r} is not in the lightweight default loader. "
        "Use path_traversal/countdown/tom_tracking/travel_planning, or add the task-specific loader before benchmarking it."
    )


def load_tasks(args) -> list[dict[str, Any]]:
    if args.benchmark == "aime24":
        return load_aime24(args)
    if args.benchmark == "gpqa":
        return load_gpqa(args)
    if args.benchmark.startswith("helmet_"):
        return load_helmet(args)
    if args.benchmark == "livecodebench_codegen":
        return load_livecodebench(args)
    if args.benchmark in {"longproc_2k", "longproc_8k"}:
        return load_longproc(args)
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
    import torch

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
        if int(args.top_k) > 0:
            kwargs["top_k"] = int(args.top_k)
    return kwargs


def generate_one(model, tokenizer, task, args) -> dict[str, Any]:
    import torch

    from benchmark.generated_memory_hf_eval import model_device
    from benchmark.longbench_v2_hf_eval import truncate_middle

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
        elif task["suite"].startswith("helmet_"):
            family = str(task.get("metadata", {}).get("metric_family", "helmet_qa"))
            if family == "helmet_recall":
                answers = task.get("answer") or []
                if isinstance(answers, str):
                    answers = [answers]
                hits = sum(str(answer).lower() in str(row["response"]).lower() for answer in answers)
                recall = float(hits / max(1, len(answers)))
                item.update({"ruler_recall": recall, "judge": bool(recall >= 1.0), "answer": task.get("answer")})
            elif family == "helmet_choice":
                pred = extract_choice_answer(row["response"]) or parse_answer_prefixed_output(row["response"]).strip().upper()[:1]
                item.update({"pred": pred, "answer": task["answer"], "judge": bool(pred == task["answer"])})
            else:
                parsed = parse_answer_prefixed_output(row["response"])
                mets = qa_metrics(row["response"], task.get("answer"))
                parsed_mets = qa_metrics(parsed, task.get("answer"))
                mets = {key: max(float(mets[key]), float(parsed_mets[key])) for key in mets}
                item.update({**mets, "parsed_output": parsed, "answer": task.get("answer"), "judge": bool(mets["substring_exact_match"])})
        elif task["suite"] in {"longproc_2k", "longproc_8k"}:
            eval_func = task["_longproc_eval_func"]
            metrics, extra = eval_func(str(row["response"]), task["_longproc_example"])
            item.update({**metrics, **extra})
            item["judge"] = bool(float(metrics.get("accuracy", 0.0)) >= 1.0)
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
        "use_chat_template": bool(args.use_chat_template),
        "disable_thinking": bool(args.disable_thinking),
        "hf_language_model_only": bool(args.hf_language_model_only),
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "top_k": int(args.top_k),
        "avg_prompt_tokens": float(sum(row["prompt_tokens"] for row in rows) / total) if total else 0.0,
        "avg_used_prompt_tokens": float(sum(row["used_prompt_tokens"] for row in rows) / total) if total else 0.0,
        "avg_generated_tokens": float(sum(row["generated_tokens"] for row in rows) / total) if total else 0.0,
        "max_generated_tokens": int(max((row["generated_tokens"] for row in rows), default=0)),
        "max_new_token_hit_count": int(sum(1 for row in rows if int(row["generated_tokens"]) >= int(args.max_new_tokens))),
        "max_new_token_hit_fraction": float(
            sum(1 for row in rows if int(row["generated_tokens"]) >= int(args.max_new_tokens)) / total
        )
        if total
        else 0.0,
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
    if args.benchmark.startswith("helmet_"):
        for key in ("exact_match", "f1", "substring_exact_match", "ruler_recall"):
            vals = [row[key] for row in rows if row.get(key) is not None]
            if vals:
                summary[f"mean_{key}"] = float(sum(float(v) for v in vals) / len(vals))
                summary[f"mean_{key}_pct"] = float(100.0 * summary[f"mean_{key}"])
    if args.benchmark in {"longproc_2k", "longproc_8k"}:
        metric_keys = sorted(
            {
                key
                for row in rows
                for key, value in row.items()
                if key in {"accuracy", "partial_accuracy", "extraction_rate", "f1", "precision", "recall"}
                and isinstance(value, (int, float, bool))
            }
        )
        for key in metric_keys:
            vals = [float(row[key]) for row in rows if row.get(key) is not None]
            if vals:
                summary[f"mean_{key}"] = float(sum(vals) / len(vals))
                summary[f"mean_{key}_pct"] = float(100.0 * summary[f"mean_{key}"])
    if args.benchmark == "livecodebench_codegen":
        metrics = next((row.get("_livecodebench_metrics") for row in rows if row.get("_livecodebench_metrics")), None)
        if metrics:
            summary["livecodebench_metrics"] = metrics
            if "pass@1" in metrics:
                summary["pass_at_1"] = float(metrics["pass@1"])
    if approx_stats is not None:
        from benchmark.longbench_v2_hf_eval import aggregate_approx_stats, summarize_approx_stats

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

    import torch
    from tqdm import tqdm
    from transformers import AutoConfig

    from benchmark.generated_memory_hf_eval import dtype_from_name, load_hf_model, load_tokenizer
    from benchmark.longbench_v2_hf_eval import maybe_apply_qwen_yarn, pagedpq_config, parse_layer_ids
    from benchmark.selector_eval.runners.hf_paged_pq_intervention_api import (
        ApproxStats,
        patched_paged_pq_attention,
        reset_paged_pq_attention_state,
    )

    torch.manual_seed(int(args.seed))
    if bool(args.allow_tf32_selector):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

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
