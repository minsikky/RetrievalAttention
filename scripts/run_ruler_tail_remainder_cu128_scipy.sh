#!/usr/bin/env bash
#SBATCH --job-name=ruler-tail-rem
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128000m
#SBATCH --time=2-00:00:00
#SBATCH --account=zhengya98
#SBATCH --partition=gpu-rtx6000,spgpu
#SBATCH --gpus-per-node=1
#SBATCH --output=slurm_out/ruler_tail_remainder_cu128_scipy/%x-%j.out
set -euo pipefail

cd /gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention

CONTEXT_LEN="${CONTEXT_LEN:-131072}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-benchmark_suite_result/ruler_tail_remainder_cu128_scipy/ruler}"
TASKS_CSV="${TASKS_CSV:-fwe,qa_1,qa_2}"

mkdir -p slurm_out/ruler_tail_remainder_cu128_scipy "${OUTPUT_ROOT}"

# Keep torch/transformers on the CUDA-12.8 path, but expose the scipy wheel
# needed by RULER fwe data prep without adding the old .venv site-packages.
export HF_VENV_DIR="${HF_VENV_DIR:-.venv_cu128}"
export HF_EXTRA_PYTHONPATH="${PWD}/.hf_pydeps_cu128:${PWD}/.hf_pydeps_cu128_scipy"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;8.6;12.0}"
export LD_LIBRARY_PATH="${PWD}/.venv/lib/python3.10/site-packages/numpy.libs:${PWD}/.hf_pydeps_cu128_scipy/scipy.libs:${LD_LIBRARY_PATH:-}"

export PREFILL_CHUNK_SIZE="${PREFILL_CHUNK_SIZE:-16384}"
export FRONTIER_EMPTY_CACHE_AFTER_PREFILL_CHUNK="${FRONTIER_EMPTY_CACHE_AFTER_PREFILL_CHUNK:-1}"
export FRONTIER_EMPTY_CACHE_AFTER_PREFILL="${FRONTIER_EMPTY_CACHE_AFTER_PREFILL:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:256}"
export FRONTIER_CANONICAL_GPU="${FRONTIER_CANONICAL_GPU:-1}"
export PROFILE_NATIVE_OPS="${PROFILE_NATIVE_OPS:-0}"
export SELECTOR_PQ_JOINT_WALL_PROFILE="${SELECTOR_PQ_JOINT_WALL_PROFILE:-1}"

IFS=',' read -r -a TASKS <<< "${TASKS_CSV}"

echo "[RULER-TAIL] start $(date) job=${SLURM_JOB_ID:-manual} context=${CONTEXT_LEN} tasks=${TASKS_CSV}"
echo "[RULER-TAIL] output_root=${OUTPUT_ROOT}"

for task in "${TASKS[@]}"; do
  for mode in dense frontier; do
    if [[ "${mode}" == "dense" ]]; then
      run_name="dense_ruler_ctx${CONTEXT_LEN}_n${NUM_SAMPLES}_${task}"
      echo "[RULER-TAIL] task_start ${run_name} $(date)"
      MODE=dense_batched \
      TASK_NAME="${task}" \
      CONTEXT_LEN="${CONTEXT_LEN}" \
      NUM_SAMPLES="${NUM_SAMPLES}" \
      RUN_NAME="${run_name}" \
      OUTPUT_ROOT="${OUTPUT_ROOT}" \
      bash scripts/run_dense_ruler_batched_one.sh
    else
      run_name="frontier_ruler_ctx${CONTEXT_LEN}_n${NUM_SAMPLES}_${task}"
      echo "[RULER-TAIL] task_start ${run_name} $(date)"
      MODE=pagedpq_batched \
      TASK_NAME="${task}" \
      CONTEXT_LEN="${CONTEXT_LEN}" \
      NUM_SAMPLES="${NUM_SAMPLES}" \
      RUN_NAME="${run_name}" \
      OUTPUT_ROOT="${OUTPUT_ROOT}" \
      bash scripts/run_frontier_ruler_batched_one.sh
    fi
    echo "[RULER-TAIL] task_done ${run_name} $(date)"
  done
done

echo "[RULER-TAIL] done $(date)"
