#!/usr/bin/env python3
from __future__ import annotations

import datetime as _dt
import math
import os
import shlex
import subprocess
from pathlib import Path


ROOT = Path("/gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention")
QWEN3_AIME_MAX_NEW_TOKENS = 38912
QWEN3_DEFAULT_MAX_NEW_TOKENS = 32768


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _truthy(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "on"}


def _split_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _quote_env(env: dict[str, str]) -> str:
    return " ".join(shlex.quote(f"{key}={value}") for key, value in sorted(env.items()))


def _resolved_hf_preset_env() -> dict[str, str]:
    preset = _env("HF_MODEL_PRESET", "qwen3_8b")
    script = (
        f"cd {shlex.quote(str(ROOT))} && "
        "source scripts/hf_model_presets.sh && "
        f"resolve_hf_model_preset {shlex.quote(preset)} >/dev/null && "
        "printf 'MODEL_NAME=%s\\nHF_LANGUAGE_MODEL_ONLY=%s\\nUSE_CHAT_TEMPLATE=%s\\nDISABLE_THINKING=%s\\nQWEN_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS=%s\\n' "
        '"${PRESET_MODEL_NAME}" "${PRESET_HF_LANGUAGE_MODEL_ONLY:-1}" '
        '"${PRESET_USE_CHAT_TEMPLATE:-1}" "${PRESET_DISABLE_THINKING:-1}" '
        '"${PRESET_QWEN_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS:-32768}"'
    )
    out = subprocess.check_output(["bash", "-lc", script], text=True)
    resolved: dict[str, str] = {"HF_MODEL_PRESET": preset}
    for line in out.splitlines():
        key, value = line.split("=", 1)
        resolved[key] = value
    for key in (
        "MODEL_NAME",
        "HF_LANGUAGE_MODEL_ONLY",
        "USE_CHAT_TEMPLATE",
        "DISABLE_THINKING",
        "QWEN_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS",
    ):
        if os.environ.get(key, "") != "":
            resolved[key] = os.environ[key]
    return resolved


def _is_qwen3_preset(hf_env: dict[str, str]) -> bool:
    preset = hf_env.get("HF_MODEL_PRESET", "")
    model_name = hf_env.get("MODEL_NAME", "")
    return preset.startswith("qwen3") or "Qwen3" in model_name or "Qwen--Qwen3" in model_name


def _qwen3_eval_defaults(hf_env: dict[str, str]) -> dict[str, str]:
    """Qwen3 report settings for task-quality public benchmarks.

    Qwen3 Technical Report uses thinking-mode sampling for the main reasoning
    table: temperature=0.6, top_p=0.95, top_k=20. Non-thinking mode has its own
    sampling defaults. Presence penalty is not exposed by this HF runner.
    """

    if not _truthy(_env("USE_QWEN3_OFFICIAL_EVAL_DEFAULTS", "1")) or not _is_qwen3_preset(hf_env):
        return {
            "TEMPERATURE": _env("PUBLIC_TEMPERATURE", _env("TEMPERATURE", "0.0")),
            "TOP_P": _env("PUBLIC_TOP_P", _env("TOP_P", "1.0")),
            "TOP_K": _env("PUBLIC_TOP_K", _env("TOP_K", "0")),
        }

    mode = _env("QWEN3_EVAL_MODE", "thinking").lower()
    if mode in {"thinking", "think"}:
        defaults = {"DISABLE_THINKING": "0", "TEMPERATURE": "0.6", "TOP_P": "0.95", "TOP_K": "20"}
    elif mode in {"nonthinking", "non-thinking", "no_think", "nothink"}:
        defaults = {"DISABLE_THINKING": "1", "TEMPERATURE": "0.7", "TOP_P": "0.8", "TOP_K": "20"}
    else:
        raise ValueError(f"unknown QWEN3_EVAL_MODE={mode!r}")

    for key in ("DISABLE_THINKING", "TEMPERATURE", "TOP_P", "TOP_K"):
        public_key = f"PUBLIC_{key}"
        if os.environ.get(public_key, "") != "":
            defaults[key] = os.environ[public_key]
        elif os.environ.get(key, "") != "":
            defaults[key] = os.environ[key]
    return defaults


def _public_generation_env(hf_env: dict[str, str]) -> dict[str, str]:
    return _qwen3_eval_defaults(hf_env)


def _deterministic_generation_env() -> dict[str, str]:
    return {
        "TEMPERATURE": _env("PUBLIC_TEMPERATURE", _env("TEMPERATURE", "0.0")),
        "TOP_P": _env("PUBLIC_TOP_P", _env("TOP_P", "1.0")),
        "TOP_K": _env("PUBLIC_TOP_K", _env("TOP_K", "0")),
    }


def _max_new_tokens_env(name: str, default: int, *, legacy_reasoning_fallback: bool = False) -> int:
    if os.environ.get(name, "") != "":
        return int(os.environ[name])
    if legacy_reasoning_fallback and os.environ.get("REASONING_MAX_NEW_TOKENS", "") != "":
        return int(os.environ["REASONING_MAX_NEW_TOKENS"])
    return int(default)


def _chunks(items: list[dict[str, object]], size: int) -> list[list[dict[str, object]]]:
    if size <= 0:
        raise ValueError("group size must be positive")
    return [items[idx : idx + size] for idx in range(0, len(items), size)]


def _task(
    *,
    suite: str,
    label: str,
    script: str,
    output_dir: str,
    env: dict[str, str],
) -> dict[str, object]:
    merged_env = dict(env)
    merged_env.setdefault("HF_VENV_DIR", _env("HF_VENV_DIR", ".venv_cu128"))
    merged_env.setdefault("HF_EXTRA_PYTHONPATH", _env("HF_EXTRA_PYTHONPATH", ".hf_pydeps_cu128"))
    merged_env.setdefault("TORCH_CUDA_ARCH_LIST", _env("TORCH_CUDA_ARCH_LIST", "8.0 8.6 12.0"))
    for key in (
        "PREFILL_CHUNK_SIZE",
        "FRONTIER_EMPTY_CACHE_AFTER_PREFILL_CHUNK",
        "FRONTIER_EMPTY_CACHE_AFTER_PREFILL",
        "PYTORCH_CUDA_ALLOC_CONF",
    ):
        if key in os.environ:
            merged_env.setdefault(key, os.environ[key])
    return {
        "suite": suite,
        "label": label,
        "script": script,
        "output_dir": output_dir,
        "env": merged_env,
    }


def _submit_shards(
    *,
    suite: str,
    bench: str,
    total: int,
    shard: int,
    max_new: int,
    min_new: int,
    force_max: int,
    evaluate_code: int,
    code_eval_timeout: int,
    output_root: str,
    modes: list[str],
    start_offset: int = 0,
    extra_env: dict[str, str] | None = None,
    generation_policy: str = "qwen3_official",
) -> list[dict[str, object]]:
    tasks: list[dict[str, object]] = []
    if total <= 0 or shard <= 0:
        return tasks
    hf_env = _resolved_hf_preset_env()
    if generation_policy == "qwen3_official":
        generation_env = _public_generation_env(hf_env)
    elif generation_policy == "deterministic":
        generation_env = _deterministic_generation_env()
    else:
        raise ValueError(f"unknown generation_policy={generation_policy!r}")
    end_offset = start_offset + total
    offset = start_offset
    while offset < end_offset:
        count = min(shard, end_offset - offset)
        for mode in modes:
            label = f"{mode}_{bench}_off{offset}_n{count}"
            env = {
                **hf_env,
                **generation_env,
                "BENCHMARK": bench,
                "ATTENTION_MODE": mode,
                "OUTPUT_DIR": f"{output_root}/{label}",
                "OUTPUT_ROOT": output_root,
                "RUN_NAME": label,
                "MAX_EXAMPLES": str(count),
                "TASK_OFFSET": str(offset),
                "SELECTION": _env("SELECTION", "first"),
                "MAX_INPUT_TOKENS": _env("PUBLIC_MAX_INPUT_TOKENS", _env("MAX_INPUT_TOKENS", "120000")),
                "MAX_NEW_TOKENS": str(max_new),
                "MIN_NEW_TOKENS": str(min_new),
                "FORCE_MAX_NEW_TOKENS": str(force_max),
                "LOCAL_FILES_ONLY": _env("LOCAL_FILES_ONLY", "1"),
                "LOW_CPU_MEM_USAGE": _env("LOW_CPU_MEM_USAGE", "1"),
                "EVALUATE_CODE": str(evaluate_code),
                "CODE_EVAL_TIMEOUT": str(code_eval_timeout),
                "LIVE_CODE_RELEASE": _env("LIVE_CODE_RELEASE", "release_v6"),
                "LONGGENBENCH_GSM8K_K": _env("LONGGEN_GSM8K_K", "32"),
                "LONGGENBENCH_GSM8K_QUESTION_LIMIT": _env("LONGGEN_GSM8K_QUESTION_LIMIT", "256"),
            }
            if extra_env:
                env.update(extra_env)
            tasks.append(
                _task(
                    suite=suite,
                    label=label,
                    script="benchmark/run_public_longdecode_hf.sh",
                    output_dir=f"{output_root}/{label}",
                    env=env,
                )
            )
        offset += count
    return tasks


def build_public_tasks(output_root: str) -> list[dict[str, object]]:
    modes = _split_csv(_env("PUBLIC_MODES", "dense,pagedpq"))
    tasks: list[dict[str, object]] = []
    if _truthy(_env("INCLUDE_AIME", "1")):
        tasks += _submit_shards(
            suite="public",
            bench="aime24",
            total=int(_env("AIME_TOTAL_EXAMPLES", "30")),
            shard=int(_env("AIME_SHARD_SIZE", "30")),
            max_new=_max_new_tokens_env(
                "AIME_MAX_NEW_TOKENS",
                QWEN3_AIME_MAX_NEW_TOKENS,
                legacy_reasoning_fallback=True,
            ),
            min_new=0,
            force_max=0,
            evaluate_code=0,
            code_eval_timeout=6,
            output_root=output_root,
            modes=modes,
        )
    if _truthy(_env("INCLUDE_GPQA", "1")):
        tasks += _submit_shards(
            suite="public",
            bench="gpqa",
            total=int(_env("GPQA_TOTAL_EXAMPLES", "50")),
            shard=int(_env("GPQA_SHARD_SIZE", "25")),
            max_new=_max_new_tokens_env(
                "GPQA_MAX_NEW_TOKENS",
                QWEN3_DEFAULT_MAX_NEW_TOKENS,
                legacy_reasoning_fallback=True,
            ),
            min_new=0,
            force_max=0,
            evaluate_code=0,
            code_eval_timeout=6,
            output_root=output_root,
            modes=modes,
        )
    if _truthy(_env("INCLUDE_LIVE_CODE", "1")):
        tasks += _submit_shards(
            suite="public",
            bench="livecodebench_codegen",
            total=int(_env("LIVE_CODE_TOTAL_EXAMPLES", "100")),
            shard=int(_env("LIVE_CODE_SHARD_SIZE", "10")),
            max_new=_max_new_tokens_env("LIVE_CODE_MAX_NEW_TOKENS", QWEN3_DEFAULT_MAX_NEW_TOKENS),
            min_new=int(_env("LIVE_CODE_MIN_NEW_TOKENS", "0")),
            force_max=int(_env("LIVE_CODE_FORCE_MAX_NEW_TOKENS", "0")),
            evaluate_code=int(_env("LIVE_CODE_EVALUATE_CODE", "1")),
            code_eval_timeout=int(_env("LIVE_CODE_CODE_EVAL_TIMEOUT", "6")),
            output_root=output_root,
            modes=modes,
            start_offset=int(_env("LIVE_CODE_START_OFFSET", "0")),
        )
    include_helmet = _truthy(_env("INCLUDE_HELMET", "0"))
    if _truthy(_env("INCLUDE_HELMET_RAG", "1" if include_helmet else "0")):
        tasks += _submit_shards(
            suite="public",
            bench="helmet_rag",
            total=int(_env("HELMET_RAG_TOTAL_EXAMPLES", "8")),
            shard=int(_env("HELMET_RAG_SHARD_SIZE", "4")),
            max_new=int(_env("HELMET_RAG_MAX_NEW_TOKENS", "20")),
            min_new=0,
            force_max=0,
            evaluate_code=0,
            code_eval_timeout=6,
            output_root=output_root,
            modes=modes,
            generation_policy="deterministic",
            extra_env={
                "MAX_INPUT_TOKENS": _env("HELMET_MAX_INPUT_TOKENS", "131072"),
                "HELMET_DATASET_FILTER": _env("HELMET_RAG_DATASET_FILTER", "kilt_nq"),
            },
        )
    if _truthy(_env("INCLUDE_HELMET_RECALL", "1" if include_helmet else "0")):
        tasks += _submit_shards(
            suite="public",
            bench="helmet_recall",
            total=int(_env("HELMET_RECALL_TOTAL_EXAMPLES", "8")),
            shard=int(_env("HELMET_RECALL_SHARD_SIZE", "4")),
            max_new=int(_env("HELMET_RECALL_MAX_NEW_TOKENS", "100")),
            min_new=0,
            force_max=0,
            evaluate_code=0,
            code_eval_timeout=6,
            output_root=output_root,
            modes=modes,
            generation_policy="deterministic",
            extra_env={
                "MAX_INPUT_TOKENS": _env("HELMET_MAX_INPUT_TOKENS", "131072"),
                "HELMET_DATASET_FILTER": _env("HELMET_RECALL_DATASET_FILTER", "ruler_niah_mk_2"),
            },
        )
    if _truthy(_env("INCLUDE_HELMET_LONGQA", "1" if include_helmet else "0")):
        tasks += _submit_shards(
            suite="public",
            bench="helmet_longqa",
            total=int(_env("HELMET_LONGQA_TOTAL_EXAMPLES", "8")),
            shard=int(_env("HELMET_LONGQA_SHARD_SIZE", "4")),
            max_new=int(_env("HELMET_LONGQA_MAX_NEW_TOKENS", "100")),
            min_new=0,
            force_max=0,
            evaluate_code=0,
            code_eval_timeout=6,
            output_root=output_root,
            modes=modes,
            generation_policy="deterministic",
            extra_env={
                "MAX_INPUT_TOKENS": _env("HELMET_MAX_INPUT_TOKENS", "131072"),
                "HELMET_DATASET_FILTER": _env("HELMET_LONGQA_DATASET_FILTER", "infbench_qa_eng_130862"),
            },
        )
    include_longproc = _truthy(_env("INCLUDE_LONGPROC", "0"))
    if _truthy(_env("INCLUDE_LONGPROC_2K", "1" if include_longproc else "0")):
        tasks += _submit_shards(
            suite="public",
            bench="longproc_2k",
            total=int(_env("LONGPROC_2K_TOTAL_EXAMPLES", "12")),
            shard=int(_env("LONGPROC_2K_SHARD_SIZE", "2")),
            max_new=int(_env("LONGPROC_2K_MAX_NEW_TOKENS", "3072")),
            min_new=int(_env("LONGPROC_2K_MIN_NEW_TOKENS", "2048")),
            force_max=1,
            evaluate_code=0,
            code_eval_timeout=6,
            output_root=output_root,
            modes=modes,
            generation_policy="deterministic",
            extra_env={
                "MAX_INPUT_TOKENS": _env("LONGPROC_MAX_INPUT_TOKENS", "120000"),
                "LONGPROC_DATASETS": _env("LONGPROC_2K_DATASETS", ""),
            },
        )
    if _truthy(_env("INCLUDE_LONGPROC_8K", "1" if include_longproc else "0")):
        tasks += _submit_shards(
            suite="public",
            bench="longproc_8k",
            total=int(_env("LONGPROC_8K_TOTAL_EXAMPLES", "10")),
            shard=int(_env("LONGPROC_8K_SHARD_SIZE", "2")),
            max_new=int(_env("LONGPROC_8K_MAX_NEW_TOKENS", "9216")),
            min_new=int(_env("LONGPROC_8K_MIN_NEW_TOKENS", "8192")),
            force_max=1,
            evaluate_code=0,
            code_eval_timeout=6,
            output_root=output_root,
            modes=modes,
            generation_policy="deterministic",
            extra_env={
                "MAX_INPUT_TOKENS": _env("LONGPROC_MAX_INPUT_TOKENS", "120000"),
                "LONGPROC_DATASETS": _env("LONGPROC_8K_DATASETS", ""),
            },
        )
    if _truthy(_env("INCLUDE_LONGGEN_SGT_SHORT", "1")):
        tasks += _submit_shards(
            suite="public",
            bench="longgenbench_sgt_short",
            total=int(_env("LONGGEN_SGT_SHORT_TOTAL_EXAMPLES", "32")),
            shard=int(_env("LONGGEN_SGT_SHORT_SHARD_SIZE", "4")),
            max_new=int(_env("LONGGEN_SGT_SHORT_MAX_NEW_TOKENS", "16384")),
            min_new=int(_env("LONGGEN_SGT_SHORT_MIN_NEW_TOKENS", "8192")),
            force_max=1,
            evaluate_code=0,
            code_eval_timeout=6,
            output_root=output_root,
            modes=modes,
            generation_policy="deterministic",
        )
    if _truthy(_env("INCLUDE_LONGGEN_SGT_LONG", "1")):
        tasks += _submit_shards(
            suite="public",
            bench="longgenbench_sgt_long",
            total=int(_env("LONGGEN_SGT_LONG_TOTAL_EXAMPLES", "16")),
            shard=int(_env("LONGGEN_SGT_LONG_SHARD_SIZE", "2")),
            max_new=int(_env("LONGGEN_SGT_LONG_MAX_NEW_TOKENS", "32768")),
            min_new=int(_env("LONGGEN_SGT_LONG_MIN_NEW_TOKENS", "16384")),
            force_max=1,
            evaluate_code=0,
            code_eval_timeout=6,
            output_root=output_root,
            modes=modes,
            generation_policy="deterministic",
        )
    if _truthy(_env("INCLUDE_LONGGEN_GSM8K", "1")):
        question_limit = int(_env("LONGGEN_GSM8K_QUESTION_LIMIT", "256"))
        k = int(_env("LONGGEN_GSM8K_K", "32"))
        default_total = str(int(math.ceil(float(question_limit) / float(k))))
        tasks += _submit_shards(
            suite="public",
            bench="longgenbench_gsm8k",
            total=int(_env("LONGGEN_GSM8K_TOTAL_EXAMPLES", default_total)),
            shard=int(_env("LONGGEN_GSM8K_SHARD_SIZE", "2")),
            max_new=int(_env("LONGGEN_GSM8K_MAX_NEW_TOKENS", "16384")),
            min_new=int(_env("LONGGEN_GSM8K_MIN_NEW_TOKENS", "8192")),
            force_max=1,
            evaluate_code=0,
            code_eval_timeout=6,
            output_root=output_root,
            modes=modes,
            generation_policy="deterministic",
        )
    return tasks


def build_ruler_tasks(output_root: str) -> list[dict[str, object]]:
    tasks: list[dict[str, object]] = []
    contexts = _split_csv(_env("RULER_CONTEXTS", "32768,65536,131072"))
    task_names = _split_csv(
        _env(
            "RULER_TASKS",
            "niah_single_1,niah_single_2,niah_single_3,"
            "niah_multikey_1,niah_multikey_2,niah_multikey_3,"
            "niah_multivalue,niah_multiquery,vt,cwe,fwe,qa_1,qa_2",
        )
    )
    modes = _split_csv(_env("RULER_MODES", "dense,frontier"))
    for context_len in contexts:
        for task_name in task_names:
            for mode in modes:
                if mode == "dense":
                    script = "scripts/run_dense_ruler_batched_one.sh"
                    run_mode = "dense_batched"
                elif mode in {"frontier", "pagedpq"}:
                    script = "scripts/run_frontier_ruler_batched_one.sh"
                    run_mode = "pagedpq_batched"
                else:
                    raise ValueError(f"unknown RULER mode: {mode}")
                label = f"{mode}_ruler_ctx{context_len}_n{_env('RULER_NUM_SAMPLES', '1')}_{task_name}"
                env = {
                    "OUTPUT_ROOT": output_root,
                    "RUN_NAME": label,
                    "TASK_NAME": task_name,
                    "CONTEXT_LEN": context_len,
                    "NUM_SAMPLES": _env("RULER_NUM_SAMPLES", "1"),
                    "MAX_NEW_TOKENS": _env("RULER_MAX_NEW_TOKENS", "128"),
                    "MODE": run_mode,
                    "FRONTIER_CANONICAL_GPU": "1" if mode != "dense" else "0",
                }
                tasks.append(
                    _task(
                        suite="ruler",
                        label=label,
                        script=script,
                        output_dir=f"{output_root}/{label}",
                        env=env,
                    )
                )
    return tasks


def build_lbv2_tasks(output_root: str) -> list[dict[str, object]]:
    tasks: list[dict[str, object]] = []
    hf_env = _resolved_hf_preset_env()
    lengths = _split_csv(_env("LONGBENCH_LENGTHS", "short,medium,long"))
    difficulties = _split_csv(_env("LONGBENCH_DIFFICULTIES", "easy,hard"))
    modes = _split_csv(_env("LONGBENCH_MODES", "dense,pagedpq"))
    for length in lengths:
        for difficulty in difficulties:
            for mode in modes:
                script = (
                    "scripts/run_dense_longbench_v2_one.sh"
                    if mode == "dense"
                    else "scripts/run_frontier_longbench_v2_one.sh"
                )
                label = (
                    f"{mode}_lbv2_{length}_{difficulty}"
                    f"_n{_env('LONGBENCH_MAX_EXAMPLES', '16')}"
                    f"_l{_env('LONGBENCH_MAX_INPUT_TOKENS', '120000')}"
                )
                env = {
                    **hf_env,
                    "OUTPUT_DIR": f"{output_root}/{label}",
                    "ATTENTION_MODE": mode,
                    "MAX_EXAMPLES": _env("LONGBENCH_MAX_EXAMPLES", "16"),
                    "LENGTH_FILTER": length,
                    "DIFFICULTY_FILTER": difficulty,
                    "MAX_INPUT_TOKENS": _env("LONGBENCH_MAX_INPUT_TOKENS", "120000"),
                    "MAX_NEW_TOKENS": _env("LONGBENCH_MAX_NEW_TOKENS", "128"),
                    "TEMPERATURE": _env("LONGBENCH_TEMPERATURE", "0.0"),
                    "LOCAL_FILES_ONLY": _env("LOCAL_FILES_ONLY", "1"),
                    "FRONTIER_CANONICAL_GPU": "1" if mode != "dense" else "0",
                }
                tasks.append(
                    _task(
                        suite="lbv2",
                        label=label,
                        script=script,
                        output_dir=f"{output_root}/{label}",
                        env=env,
                    )
                )
    return tasks


def write_group_script(
    *,
    suite: str,
    group_idx: int,
    tasks: list[dict[str, object]],
    job_dir: Path,
    time_limit: str,
    mem: str,
    cpus: str,
) -> Path:
    job_dir.mkdir(parents=True, exist_ok=True)
    script_path = job_dir / f"{suite}_group_{group_idx:03d}.sh"
    lines = [
        "#!/usr/bin/env bash",
        f"#SBATCH --job-name={suite}-coal-{group_idx:03d}",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks-per-node=1",
        f"#SBATCH --cpus-per-task={cpus}",
        f"#SBATCH --mem={mem}",
        f"#SBATCH --time={time_limit}",
        "#SBATCH --account=zhengya98",
        "#SBATCH --gpus-per-node=1",
        "set -euo pipefail",
        f"cd {shlex.quote(str(ROOT))}",
        "echo \"[COALESCED] start $(date) suite=${SLURM_JOB_NAME:-unknown} job=${SLURM_JOB_ID:-manual}\"",
    ]
    for task in tasks:
        env_text = _quote_env(task["env"])  # type: ignore[arg-type]
        label = str(task["label"])
        script = str(task["script"])
        lines += [
            f"echo \"[COALESCED] task_start {shlex.quote(label)} $(date)\"",
            f"mkdir -p {shlex.quote(str(task['output_dir']))}",
            f"env {env_text} bash {shlex.quote(script)}",
            f"echo \"[COALESCED] task_done {shlex.quote(label)} $(date)\"",
        ]
    lines.append("echo \"[COALESCED] done $(date)\"")
    script_path.write_text("\n".join(lines) + "\n")
    script_path.chmod(0o755)
    return script_path


def submit_groups(
    *,
    suite: str,
    tasks: list[dict[str, object]],
    group_size: int,
    job_dir: Path,
    slurm_root: Path,
    manifest_rows: list[list[str]],
    task_rows: list[list[str]],
    submit: bool,
    partitions: str,
    time_limit: str,
    mem: str,
    cpus: str,
) -> int:
    count = 0
    for group_idx, group_tasks in enumerate(_chunks(tasks, group_size)):
        group_label = f"{suite}_group_{group_idx:03d}"
        script_path = write_group_script(
            suite=suite,
            group_idx=group_idx,
            tasks=group_tasks,
            job_dir=job_dir,
            time_limit=time_limit,
            mem=mem,
            cpus=cpus,
        )
        slurm_out = slurm_root / f"{group_label}-%j.out"
        if submit:
            jobid = subprocess.check_output(
                [
                    "sbatch",
                    "--parsable",
                    f"--partition={partitions}",
                    f"--output={slurm_out}",
                    str(script_path),
                ],
                text=True,
            ).strip()
            rendered_out = str(slurm_out).replace("%j", jobid)
        else:
            jobid = "DRYRUN"
            rendered_out = str(slurm_out).replace("%j", "DRYRUN")
        task_labels = ",".join(str(task["label"]) for task in group_tasks)
        output_dirs = ",".join(str(task["output_dir"]) for task in group_tasks)
        manifest_rows.append(
            [
                group_label,
                jobid,
                output_dirs,
                rendered_out,
                suite,
                str(len(group_tasks)),
                task_labels,
            ]
        )
        for task in group_tasks:
            task_rows.append(
                [
                    str(task["label"]),
                    group_label,
                    jobid,
                    str(task["output_dir"]),
                    rendered_out,
                    str(task["suite"]),
                    str(task["script"]),
                ]
            )
        print(f"{group_label}\t{jobid}\t{len(group_tasks)}\t{task_labels}")
        count += 1
    return count


def write_tsv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("\t".join(header) + "\n")
        for row in rows:
            f.write("\t".join(row) + "\n")


def main() -> None:
    stamp = _env("STAMP", _dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
    suite_name = _env("SUITE_NAME", f"coalesced_benchmark_suite_{stamp}")
    output_root = Path(_env("OUTPUT_ROOT", f"benchmark_suite_result/{suite_name}"))
    slurm_root = Path(_env("SLURM_ROOT", f"slurm_out/{suite_name}"))
    manifest = Path(_env("MANIFEST", f"notes/slurm_manifests/{suite_name}.tsv"))
    task_manifest = Path(_env("TASK_MANIFEST", f"notes/slurm_manifests/{suite_name}_tasks.tsv"))
    job_dir = output_root / "job_scripts"
    submit = _truthy(_env("SUBMIT", "0"))
    partitions = _env("PARTITIONS", "gpu-rtx6000,spgpu,gpu_mig40")

    output_root.mkdir(parents=True, exist_ok=True)
    slurm_root.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[list[str]] = []
    task_rows: list[list[str]] = []
    total_jobs = 0
    total_tasks = 0

    if _truthy(_env("RUN_PUBLIC", "1")):
        tasks = build_public_tasks(str(output_root / "public"))
        total_tasks += len(tasks)
        total_jobs += submit_groups(
            suite="public",
            tasks=tasks,
            group_size=int(_env("PUBLIC_GROUP_SIZE", "4")),
            job_dir=job_dir,
            slurm_root=slurm_root,
            manifest_rows=manifest_rows,
            task_rows=task_rows,
            submit=submit,
            partitions=partitions,
            # AIME24 Qwen3 thinking-mode generations are long: dense+pagedPQ
            # full shards can exceed 30 hours before any coalesced neighbors.
            time_limit=_env("PUBLIC_TIME", "3-00:00:00"),
            mem=_env("PUBLIC_MEM", "128000m"),
            cpus=_env("PUBLIC_CPUS", "4"),
        )
    if _truthy(_env("RUN_RULER", "1")):
        tasks = build_ruler_tasks(str(output_root / "ruler"))
        total_tasks += len(tasks)
        total_jobs += submit_groups(
            suite="ruler",
            tasks=tasks,
            group_size=int(_env("RULER_GROUP_SIZE", "4")),
            job_dir=job_dir,
            slurm_root=slurm_root,
            manifest_rows=manifest_rows,
            task_rows=task_rows,
            submit=submit,
            partitions=partitions,
            time_limit=_env("RULER_TIME", "24:00:00"),
            mem=_env("RULER_MEM", "128000m"),
            cpus=_env("RULER_CPUS", "4"),
        )
    if _truthy(_env("RUN_LONGBENCH", "1")):
        tasks = build_lbv2_tasks(str(output_root / "longbench_v2"))
        total_tasks += len(tasks)
        total_jobs += submit_groups(
            suite="lbv2",
            tasks=tasks,
            group_size=int(_env("LONGBENCH_GROUP_SIZE", "4")),
            job_dir=job_dir,
            slurm_root=slurm_root,
            manifest_rows=manifest_rows,
            task_rows=task_rows,
            submit=submit,
            partitions=partitions,
            time_limit=_env("LONGBENCH_TIME", "12:00:00"),
            mem=_env("LONGBENCH_MEM", "128000m"),
            cpus=_env("LONGBENCH_CPUS", "4"),
        )

    write_tsv(
        manifest,
        ["label", "jobid", "output_dir", "slurm_out", "suite", "task_count", "task_labels"],
        manifest_rows,
    )
    write_tsv(
        task_manifest,
        ["task_label", "group_label", "jobid", "output_dir", "slurm_out", "suite", "script"],
        task_rows,
    )
    print(f"[INFO] SUBMIT={int(submit)}")
    print(f"[INFO] total_task_runs={total_tasks}")
    print(f"[INFO] total_slurm_jobs={total_jobs}")
    print(f"[INFO] manifest={manifest}")
    print(f"[INFO] task_manifest={task_manifest}")


if __name__ == "__main__":
    main()
