#!/usr/bin/env bash
#SBATCH --job-name=qwen1m-probe-mig40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60000m
#SBATCH --time=01:30:00
#SBATCH --account=zhengya0
#SBATCH --partition=gpu_mig40
#SBATCH --gpus-per-node=1
set -euo pipefail

# Cheap memory-model calibration probe — the trace lines tell us whether the
# sdpa path materializes GQA repeat_kv copies (per-chunk allocated jumps ~2x
# KV-chunk at 28 heads) and measures true per-chunk transients before the
# 256k/512k/1M arms. Mirrors run_qwen1m_dense_spike_one.sh at a short context.
#
# Env inputs: CONTEXT_LEN (default 32768), NUM_SAMPLES (default 2),
#             PREFILL_CHUNK_SIZE (default 4096 — bounds the chunk x kv
#             causal-mask allocation, ~8 GB at 1M), TASK_NAME.

cd /gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention

QWEN_SNAPSHOT=".hf_cache/hub/models--Qwen--Qwen2.5-7B-Instruct-1M/snapshots/e28526f7bb80e2a9c8af03b831a9af3812f18fba"

export MODEL_NAME="${QWEN_SNAPSHOT}"
export MODEL_TEMPLATE_TYPE="qwen-chat"
export MODE="dense_batched"
export TASK_NAME="${TASK_NAME:-niah_single_1}"
export CONTEXT_LEN="${CONTEXT_LEN:-32768}"
export NUM_SAMPLES="${NUM_SAMPLES:-2}"
export PREFILL_CHUNK_SIZE="${PREFILL_CHUNK_SIZE:-4096}"
export OUTPUT_ROOT="benchmark_suite_result/qwen1m_dense_probe_20260709/runs"
export RUN_NAME="dense_${TASK_NAME}_${CONTEXT_LEN}_n${NUM_SAMPLES}"

# Memory instrumentation (peak-memory trace lines) + zero numeric-impact
# hygiene (allocator config + per-chunk cache release).
export SELECTOR_PQ_JOINT_MEMORY_TRACE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FRONTIER_EMPTY_CACHE_AFTER_PREFILL_CHUNK=1

bash scripts/run_ruler_pagedpq_stream_smoke_one.sh
