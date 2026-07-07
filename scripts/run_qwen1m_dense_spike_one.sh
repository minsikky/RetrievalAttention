#!/usr/bin/env bash
#SBATCH --job-name=qwen1m-dense-spike
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180000m
#SBATCH --time=12:00:00
#SBATCH --account=zhengya0
#SBATCH --partition=gpu-rtx6000
#SBATCH --gpus-per-node=1
set -euo pipefail

# Phase E step 2: dense feasibility spike for Qwen2.5-7B-Instruct-1M on a
# 96 GB RTX Pro 6000 Blackwell. One context length per job; RULER
# niah_single_1 through the standard smoke script in dense_batched mode.
# Purpose: prove native long-context accuracy (the model is trained to
# 256k and extended to 1M) and measure wall time / peak VRAM before any
# frontier arms. gpu-rtx6000 ONLY — spgpu A40s (44 GB) cannot hold the
# KV cache (55 GB at 1M).
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
export OUTPUT_ROOT="benchmark_suite_result/qwen1m_dense_spike_20260707/runs"
export RUN_NAME="dense_${TASK_NAME}_${CONTEXT_LEN}_n${NUM_SAMPLES}"

bash scripts/run_ruler_pagedpq_stream_smoke_one.sh
