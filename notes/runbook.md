# Runbook

Keep this file to current commands only. Old command recipes are preserved in `notes/archive/status_history/runbook_2026-05-20_full.md`.

## Rules

- Use Slurm for GPU, benchmark, and compile/build jobs.
- Prefer partition `spgpu` and account `zhengya98`.
- Do not run heavy GPU workloads or extension builds on the login node.
- Use the repo `.venv` and load `python/3.10.4` when needed.
- Report logical frontier MB separately from physical GPU simulator MB.

## CUDA / Frontier Unit Validation

```bash
sbatch scripts/run_frontier_cuda_unit_tests.sh
```

## Exact-Logit Backend Benchmark

```bash
CTX_LEN=32768 sbatch scripts/run_exact_logit_backend_bench.sh
CTX_LEN=65536 sbatch scripts/run_exact_logit_backend_bench.sh
CTX_LEN=131072 sbatch scripts/run_exact_logit_backend_bench.sh
```

Use this to choose between ranked-gather and dense-sim exact-logit backends for benchmark execution. This affects physical GPU runtime, not logical frontier accounting.

## RULER Smoke

Dense reference:

```bash
TASK_NAME=niah_single_1 CONTEXT_LEN=32768 NUM_SAMPLES=1 \
OUTPUT_ROOT=ruler_eval_result/dense_smoke \
sbatch scripts/run_dense_ruler_batched_one.sh
```

Canonical frontier:

```bash
TASK_NAME=niah_single_1 CONTEXT_LEN=32768 NUM_SAMPLES=1 \
OUTPUT_ROOT=ruler_eval_result/frontier_smoke \
FRONTIER_CANONICAL_GPU=1 PROFILE_NATIVE_OPS=1 \
sbatch scripts/run_frontier_ruler_batched_one.sh
```

## LongBench-v2 Smoke

Dense reference:

```bash
MAX_EXAMPLES=2 LENGTH_FILTER=short DIFFICULTY_FILTER=easy MAX_INPUT_TOKENS=32768 \
OUTPUT_DIR=longbench_v2_hf_result/dense_smoke \
sbatch scripts/run_dense_longbench_v2_one.sh
```

Canonical frontier:

```bash
MAX_EXAMPLES=2 LENGTH_FILTER=short DIFFICULTY_FILTER=easy MAX_INPUT_TOKENS=32768 \
OUTPUT_DIR=longbench_v2_hf_result/frontier_smoke \
FRONTIER_CANONICAL_GPU=1 PROFILE_NATIVE_OPS=1 \
sbatch scripts/run_frontier_longbench_v2_one.sh
```

## Audit / Reporting

```bash
.venv/bin/python benchmark/audit_benchmark_wrappers.py \
  --output notes/archive/benchmark_audits_2026-05/wrapper_config_latest.md
```

```bash
.venv/bin/python benchmark/audit_benchmark_readiness.py \
  --manifest <manifest.tsv> \
  --output notes/archive/benchmark_audits_2026-05/readiness_audit_latest.md
```

```bash
bash scripts/check_frontier_benchmark_readiness.sh
```

## HuggingFace Cache

HF scripts default to workspace-local caches:

```bash
HF_CACHE_DIR=/gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention/.hf_cache
HF_HOME=${HF_CACHE_DIR}
HF_HUB_CACHE=${HF_CACHE_DIR}/hub
HF_DATASETS_CACHE=${HF_CACHE_DIR}/datasets
TRANSFORMERS_CACHE=${HF_CACHE_DIR}/transformers
```
