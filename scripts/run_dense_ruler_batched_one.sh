#!/usr/bin/env bash
#SBATCH --job-name=dense-ruler
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128000m
#SBATCH --time=01:00:00
#SBATCH --account=zhengya98
#SBATCH --partition=spgpu
#SBATCH --gpus-per-node=1
set -euo pipefail

# Dense/reference preset for one RULER task.

cd /gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention

export MODE="${MODE:-dense_batched}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-ruler_eval_result/dense_batched}"
export TASK_NAME="${TASK_NAME:-niah_single_1}"
export CONTEXT_LEN="${CONTEXT_LEN:-8192}"
export NUM_SAMPLES="${NUM_SAMPLES:-4}"
export PROFILE_NATIVE_OPS="${PROFILE_NATIVE_OPS:-0}"

exec bash scripts/run_ruler_pagedpq_stream_smoke_one.sh
