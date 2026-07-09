#!/usr/bin/env bash
#SBATCH --job-name=qwen1m-dense-offload
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180000m
#SBATCH --time=2-00:00:00
#SBATCH --account=zhengya0
#SBATCH --partition=gpu_mig40
#SBATCH --gpus-per-node=1
set -euo pipefail

# Exact dense Qwen2.5-7B-Instruct-1M with bf16 KV in pinned host memory.
# CONTEXT_LEN selects 262144, 524288, or 1048576. The 180 GB host request
# covers the ~56 GiB 1M KV allocation plus model/data/process headroom.

WORKTREE="/gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention/worktrees/p2-kv-offload"
MAIN_REPO="/gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention"
QWEN_SNAPSHOT="${MAIN_REPO}/.hf_cache/hub/models--Qwen--Qwen2.5-7B-Instruct-1M/snapshots/e28526f7bb80e2a9c8af03b831a9af3812f18fba"
cd "${WORKTREE}"

export MODEL_NAME="${QWEN_SNAPSHOT}"
export CACHE_DIR="${MAIN_REPO}/.hf_cache"
export HF_VENV_DIR="${MAIN_REPO}/.venv"
export MODEL_TEMPLATE_TYPE="qwen-chat"
export MODE="dense_batched"
export DENSE_KV_OFFLOAD=1
export DENSE_KV_BLOCK_TOKENS="${DENSE_KV_BLOCK_TOKENS:-8192}"
export DENSE_KV_STAGING_BUFFERS="${DENSE_KV_STAGING_BUFFERS:-2}"
export DENSE_KV_QUERY_BLOCK_TOKENS="${DENSE_KV_QUERY_BLOCK_TOKENS:-2048}"
export TASK_NAME="${TASK_NAME:-niah_single_1}"
export CONTEXT_LEN="${CONTEXT_LEN:-262144}"
export NUM_SAMPLES="${NUM_SAMPLES:-8}"
export PREFILL_CHUNK_SIZE="${PREFILL_CHUNK_SIZE:-8192}"
export OUTPUT_ROOT="benchmark_suite_result/qwen1m_dense_spike_offload/runs"
export RUN_NAME="dense_offload_${TASK_NAME}_${CONTEXT_LEN}_n${NUM_SAMPLES}"
export SELECTOR_PQ_JOINT_MEMORY_TRACE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

bash scripts/run_ruler_pagedpq_stream_smoke_one.sh
