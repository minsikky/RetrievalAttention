# Runbook

Keep this file to current commands only. Old command recipes are preserved in `notes/archive/status_history/runbook_2026-05-20_full.md`.

## Rules

- Use Slurm for GPU, benchmark, and compile/build jobs.
- Prefer partition `spgpu` and account `zhengya98`.
- Do not run heavy GPU workloads or extension builds on the login node.
- Use the repo `.venv` and load `python/3.10.4` when needed.
- Report logical frontier MB separately from physical GPU simulator MB.
- Do not use `DISABLE_COST_STATS=1` / no-stats runs for optimization or promotion decisions. Use accounting/profile runs so latency, logical MB, physical MB, selected counts, and wall buckets are captured together.
- Active Slurm wrappers should reject `DISABLE_COST_STATS=1` rather than forwarding `--disable_cost_stats`.
- Canonical frontier defaults live in `benchmark/selector_eval/frontier_config.py`; regenerate `scripts/frontier_canonical_env.sh` and `scripts/frontier_direct_runtime_env.sh` from that module if the canonical contract changes.

## CUDA / Frontier Unit Validation

```bash
sbatch scripts/run_frontier_cuda_unit_tests.sh
```

## Joint K/V Trace Parity

```bash
sbatch scripts/run_joint_kv_cpu_gpu_parity_one.sh
```

This checks the current residual-risk adaptive K/V policy on a saved real Q/K/V trace. It compares the CPU-reference selector path against the CUDA selector path for accepted K/V budgets, attention outputs, optional o-proj outputs, and logical MB.

To also exercise the benchmark-style Torch/GPU mixed-logit and residual-risk V-output grid, enable:

```bash
COMPARE_TORCH_GPU_POLICY=1 sbatch scripts/run_joint_kv_cpu_gpu_parity_one.sh
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
FRONTIER_CANONICAL_GPU=1 PROFILE_NATIVE_OPS=1 SELECTOR_PQ_JOINT_WALL_PROFILE=1 \
sbatch scripts/run_frontier_ruler_batched_one.sh
```

Current canonical defaults are `ONLINE_CONFIDENCE_RULE=joint_kv_stability` and `SELECTED_VALUE_EXACT_RULE=global_residual_risk`. Older `geometric_probe_tail_switch` plus `selected_mass` runs are baselines only.

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
FRONTIER_CANONICAL_GPU=1 PROFILE_NATIVE_OPS=1 SELECTOR_PQ_JOINT_WALL_PROFILE=1 \
sbatch scripts/run_frontier_longbench_v2_one.sh
```

## Public Long-Decode Suite

Smoke matrix, launches separate dense/frontier jobs:

```bash
MAX_EXAMPLES=1 HF_MODEL_PRESET=qwen3_8b \
bash scripts/submit_public_longdecode_matrix.sh
```

Full matrix plan, dry-run by default:

```bash
HF_MODEL_PRESET=qwen3_8b \
bash scripts/submit_public_longdecode_full_matrix.sh
```

Launch the full matrix only when ready:

```bash
SUBMIT=1 HF_MODEL_PRESET=qwen3_8b \
bash scripts/submit_public_longdecode_full_matrix.sh
```

The full matrix shards AIME24, GPQA, LiveCodeBench codegen, LongGenBench SGT short/long, and LongGenBench GSM8K with dense and canonical `pagedpq` modes. Defaults are capped to fewer than 100 jobs. Override totals or shard sizes before launch, for example `LIVE_CODE_TOTAL_EXAMPLES=175`, `LONGGEN_SGT_SHORT_TOTAL_EXAMPLES=400`, or `LONGGEN_SGT_SHORT_SHARD_SIZE=8`.

## Audit / Reporting

Regenerate canonical shell fragments after editing `frontier_config.py`:

```bash
module load python/3.10.4
python -m benchmark.selector_eval.frontier_config --emit-shell > scripts/frontier_canonical_env.sh
python -m benchmark.selector_eval.frontier_config --emit-direct-runtime-shell > scripts/frontier_direct_runtime_env.sh
```

## KV-Compression Trace Comparison

```bash
RUN_NAME=kvcomp_full_scalar_YYYYMMDD \
METHODS=dense,kivi_like_b2_w128,kivi_like_b3_w128,kivi_like_b4_w128 \
sbatch scripts/run_kv_compression_rel_l2_eval_one.sh
```

Plot one or more completed `summary.csv` files:

```bash
MPLCONFIGDIR=/tmp/matplotlib-kvcomp \
.venv/bin/python scripts/plot_kv_compression_rel_l2.py \
  --compression_summary_csv <summary1.csv>,<summary2.csv> \
  --existing_points_csv '' \
  --output_dir attention_efficiency_result/plots/<name>
```

## Frontier Pareto Sweep

Run one budget ladder:

```bash
RUN_NAME=frontier_pareto_low_YYYYMMDD \
OUTPUT_ROOT=attention_efficiency_result/frontier_pareto_YYYYMMDD \
STABILITY_THRESHOLDS=0.001,0.002,0.004,0.008,0.016,0.032,0.064,0.128 \
POLICIES=k_first_alternating \
K_BUDGETS=512,1024,2048,4096,8192,14336 \
V_BUDGETS=128,256,512,1024,2048,4096,8192 \
sbatch scripts/run_joint_kv_budget_policy_eval_one.sh
```

Merge/plot completed sweep summaries:

```bash
MPLCONFIGDIR=/tmp/matplotlib-frontier-pareto \
.venv/bin/python scripts/plot_frontier_pareto_sweep.py \
  --joint_summaries low:<low/summary.json>,tiny:<tiny/summary.json> \
  --compression_summary_csv attention_efficiency_result/kv_compression_rel_l2_20260522/kvcomp_full_pq_20260522/summary.csv \
  --output_dir attention_efficiency_result/plots/<name>
```

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
