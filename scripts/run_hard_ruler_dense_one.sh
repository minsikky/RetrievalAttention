#!/usr/bin/env bash
#SBATCH --job-name=hard-ruler-dense
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128000m
#SBATCH --time=10:00:00
#SBATCH --account=zhengya0
#SBATCH --partition=gpu-rtx6000,spgpu
#SBATCH --gpus-per-node=1
set -euo pipefail

# Dense reference for one (hard) RULER task with the cu128 env stack.
# Env inputs: TASK_NAME, CONTEXT_LEN, NUM_SAMPLES, OUTPUT_ROOT.

cd /gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention

export HF_VENV_DIR="${HF_VENV_DIR:-.venv_cu128}"
export HF_EXTRA_PYTHONPATH="${PWD}/.hf_pydeps_cu128:${PWD}/.hf_pydeps_cu128_scipy"
export LD_LIBRARY_PATH="${PWD}/.venv/lib/python3.10/site-packages/numpy.libs:${PWD}/.hf_pydeps_cu128_scipy/scipy.libs:${LD_LIBRARY_PATH:-}"
export PREFILL_CHUNK_SIZE="${PREFILL_CHUNK_SIZE:-16384}"
export FRONTIER_EMPTY_CACHE_AFTER_PREFILL_CHUNK="${FRONTIER_EMPTY_CACHE_AFTER_PREFILL_CHUNK:-1}"

export MODE=dense_batched
export OUTPUT_ROOT="${OUTPUT_ROOT:-benchmark_suite_result/hard_ruler_20260706/ruler}"
export TASK_NAME="${TASK_NAME:?TASK_NAME required}"
export CONTEXT_LEN="${CONTEXT_LEN:-131072}"
export NUM_SAMPLES="${NUM_SAMPLES:-16}"
export RUN_NAME="dense_${TASK_NAME}_ctx${CONTEXT_LEN}_n${NUM_SAMPLES}"

exec bash scripts/run_ruler_pagedpq_stream_smoke_one.sh
