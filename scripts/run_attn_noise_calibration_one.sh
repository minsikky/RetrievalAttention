#!/usr/bin/env bash
#SBATCH --job-name=attn-noise-calib
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128000m
#SBATCH --time=10:00:00
#SBATCH --account=zhengya0
#SBATCH --partition=gpu-rtx6000,spgpu
#SBATCH --gpus-per-node=1
#SBATCH --output=slurm_out/attn_noise_calibration/%x-%j.out
set -euo pipefail

# relL2 -> task-quality calibration: dense RULER runs with controlled Gaussian
# noise injected on every layer's decode-step attention output (post-o_proj)
# at a fixed relative L2. Sweeps noise levels for one task; data is prepared
# once per task so every noise level scores the exact same samples.
#
# Decision criteria (see notes/current_status.md 2026-07-05 calibration entry):
# - score at level 0 must match the known dense reference (harness sanity).
# - The largest level with no score drop vs level 0 is the calibrated safe
#   per-layer/per-step o-proj relL2. If >= 0.002, the frontier tau=0.002
#   operating point is task-safe; if >= 0.005, there is headroom to raise tau
#   for more MB savings; if < 0.002, relL2 is an unsafe proxy and tau must
#   tighten or the proxy must change.

cd /gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention

TASK_NAME="${TASK_NAME:-niah_single_1}"
CONTEXT_LEN="${CONTEXT_LEN:-32768}"
NUM_SAMPLES="${NUM_SAMPLES:-50}"
NOISE_LEVELS="${NOISE_LEVELS:-0,0.0005,0.001,0.002,0.005,0.01,0.02,0.05}"
NOISE_SEED="${NOISE_SEED:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-benchmark_suite_result/attn_noise_calibration_20260705/ruler}"

mkdir -p slurm_out/attn_noise_calibration "${OUTPUT_ROOT}"

export HF_VENV_DIR="${HF_VENV_DIR:-.venv_cu128}"
export HF_EXTRA_PYTHONPATH="${PWD}/.hf_pydeps_cu128:${PWD}/.hf_pydeps_cu128_scipy"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;8.6;12.0}"
export LD_LIBRARY_PATH="${PWD}/.venv/lib/python3.10/site-packages/numpy.libs:${PWD}/.hf_pydeps_cu128_scipy/scipy.libs:${LD_LIBRARY_PATH:-}"

export PREFILL_CHUNK_SIZE="${PREFILL_CHUNK_SIZE:-16384}"
export FRONTIER_EMPTY_CACHE_AFTER_PREFILL_CHUNK="${FRONTIER_EMPTY_CACHE_AFTER_PREFILL_CHUNK:-1}"
export FRONTIER_EMPTY_CACHE_AFTER_PREFILL="${FRONTIER_EMPTY_CACHE_AFTER_PREFILL:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:256}"
export PROFILE_NATIVE_OPS=0

IFS=',' read -r -a LEVELS <<< "${NOISE_LEVELS}"

echo "[NOISE-CALIB] start $(date) job=${SLURM_JOB_ID:-manual} task=${TASK_NAME} context=${CONTEXT_LEN} n=${NUM_SAMPLES} levels=${NOISE_LEVELS} seed=${NOISE_SEED}"
echo "[NOISE-CALIB] output_root=${OUTPUT_ROOT}"

shared_data_file=""
for level in "${LEVELS[@]}"; do
  run_name="dense_noise${level}_seed${NOISE_SEED}_${TASK_NAME}_ctx${CONTEXT_LEN}_n${NUM_SAMPLES}"
  echo "[NOISE-CALIB] run_start ${run_name} $(date)"
  if [[ "${level}" == "0" || "${level}" == "0.0" ]]; then
    export ATTN_OUTPUT_NOISE_REL_L2=""
  else
    export ATTN_OUTPUT_NOISE_REL_L2="${level}"
  fi
  export ATTN_OUTPUT_NOISE_SEED="${NOISE_SEED}"
  export ATTN_OUTPUT_NOISE_SCOPE="decode"
  if [[ -n "${shared_data_file}" ]]; then
    export DATA_FILE_OVERRIDE="${shared_data_file}"
  else
    unset DATA_FILE_OVERRIDE || true
  fi
  MODE=dense_batched \
  TASK_NAME="${TASK_NAME}" \
  CONTEXT_LEN="${CONTEXT_LEN}" \
  NUM_SAMPLES="${NUM_SAMPLES}" \
  RUN_NAME="${run_name}" \
  OUTPUT_ROOT="${OUTPUT_ROOT}" \
  bash scripts/run_dense_ruler_batched_one.sh
  if [[ -z "${shared_data_file}" ]]; then
    candidate="${OUTPUT_ROOT}/${run_name}/data/${TASK_NAME}/validation.jsonl"
    if [[ -s "${candidate}" ]]; then
      shared_data_file="${candidate}"
      echo "[NOISE-CALIB] shared data file: ${shared_data_file}"
    else
      echo "[NOISE-CALIB] WARNING: expected data file missing: ${candidate}" >&2
    fi
  fi
  echo "[NOISE-CALIB] run_done ${run_name} $(date)"
done

echo "[NOISE-CALIB] collecting scores"
for level in "${LEVELS[@]}"; do
  run_name="dense_noise${level}_seed${NOISE_SEED}_${TASK_NAME}_ctx${CONTEXT_LEN}_n${NUM_SAMPLES}"
  summary="${OUTPUT_ROOT}/${run_name}/summary/${TASK_NAME}.json"
  if [[ -s "${summary}" ]]; then
    score=$(grep -o '"score": *[0-9.]*' "${summary}" | head -1 || true)
    echo "[NOISE-CALIB] level=${level} ${score:-score-missing} (${summary})"
  else
    echo "[NOISE-CALIB] level=${level} summary-missing (${summary})"
  fi
done

echo "[NOISE-CALIB] done $(date)"
