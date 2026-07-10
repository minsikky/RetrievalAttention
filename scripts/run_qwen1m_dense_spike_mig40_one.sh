#!/usr/bin/env bash
#SBATCH --job-name=qwen1m-spike-mig40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120000m
#SBATCH --time=08:00:00
#SBATCH --account=zhengya0
#SBATCH --partition=gpu_mig40
#SBATCH --gpus-per-node=1
set -euo pipefail

# Small-card (40 GB MIG) replica of the 256k dense spike arm with memory
# instrumentation; predicted peak ~34 GiB. Mirrors run_qwen1m_dense_spike_one.sh
# but sized for a 40 GB MIG slice and emits the joint peak-memory trace. Also
# submittable to spgpu via `sbatch --partition=spgpu ...` override.
#
# Env inputs: CONTEXT_LEN (default 262144), NUM_SAMPLES (default 8),
#             PREFILL_CHUNK_SIZE (default 4096 — bounds the chunk x kv
#             causal-mask allocation, ~8 GB at 1M), TASK_NAME.

cd /gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention

QWEN_SNAPSHOT=".hf_cache/hub/models--Qwen--Qwen2.5-7B-Instruct-1M/snapshots/e28526f7bb80e2a9c8af03b831a9af3812f18fba"

export MODEL_NAME="${QWEN_SNAPSHOT}"
export MODEL_TEMPLATE_TYPE="qwen-chat"
export MODE="dense_batched"
export TASK_NAME="${TASK_NAME:-niah_single_1}"
export CONTEXT_LEN="${CONTEXT_LEN:-262144}"
export NUM_SAMPLES="${NUM_SAMPLES:-8}"
export PREFILL_CHUNK_SIZE="${PREFILL_CHUNK_SIZE:-4096}"
export OUTPUT_ROOT="benchmark_suite_result/qwen1m_dense_spike_smallcard_20260709/runs"
export RUN_NAME="dense_${TASK_NAME}_${CONTEXT_LEN}_n${NUM_SAMPLES}"

# Memory instrumentation (peak-memory trace lines) + zero numeric-impact
# hygiene (allocator config + per-chunk cache release).
export SELECTOR_PQ_JOINT_MEMORY_TRACE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FRONTIER_EMPTY_CACHE_AFTER_PREFILL_CHUNK=1

bash scripts/run_ruler_pagedpq_stream_smoke_one.sh
