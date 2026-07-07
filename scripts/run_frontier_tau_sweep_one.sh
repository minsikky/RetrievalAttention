#!/usr/bin/env bash
#SBATCH --job-name=frontier-tau-sweep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128000m
#SBATCH --time=10:00:00
#SBATCH --account=zhengya0
#SBATCH --partition=gpu-rtx6000,spgpu
#SBATCH --gpus-per-node=1
set -euo pipefail

# Stability-threshold (tau) sweep for the canonical frontier on one RULER task.
# Motivation: the 2026-07-05/06 noise-injection calibration showed decode-side
# per-step relL2 up to 0.05 (32k) / 0.01 (128k) is task-invisible, so the
# canonical tau=0.002 has real headroom. This job measures, at task level with
# the REAL (structured) frontier error, whether tau=0.004/0.008 preserves the
# score and how much the measured decode bytes drop.
#
# Env inputs: TASK_NAME, CONTEXT_LEN, NUM_SAMPLES, TAUS (comma list).
# Decision criteria per task:
#   - score at tau must match the tau=0.002 arm (same samples, deterministic
#     data prep per output dir seed) -> tau is task-safe on structured error.
#   - cost_proxy / decode byte stats in summary JSON quantify the MB saving;
#     CPU traces predict roughly -25% step MB at 0.004 with precision off.
#   - if 0.004 holds but 0.008 drops, adopt 0.004; if both drop, keep 0.002
#     (would mean structured error binds where Gaussian does not - itself a
#     finding worth recording against the calibration).

cd /gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention

export HF_VENV_DIR="${HF_VENV_DIR:-.venv_cu128}"
export HF_EXTRA_PYTHONPATH="${PWD}/.hf_pydeps_cu128:${PWD}/.hf_pydeps_cu128_scipy"
# scipy's bundled BLAS needs libgfortran from numpy.libs (same fix as the
# noise-calibration script); without it transformers' import chain dies.
export LD_LIBRARY_PATH="${PWD}/.venv/lib/python3.10/site-packages/numpy.libs:${PWD}/.hf_pydeps_cu128_scipy/scipy.libs:${LD_LIBRARY_PATH:-}"
export PREFILL_CHUNK_SIZE="${PREFILL_CHUNK_SIZE:-8192}"
# 128k on 44GB GPUs sits at the memory edge; expandable segments avoids
# the fragmentation OOM seen in jobs 52996326-28.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export FRONTIER_EMPTY_CACHE_AFTER_PREFILL_CHUNK="${FRONTIER_EMPTY_CACHE_AFTER_PREFILL_CHUNK:-1}"

TASK_NAME="${TASK_NAME:-niah_multikey_2}"
CONTEXT_LEN="${CONTEXT_LEN:-32768}"
NUM_SAMPLES="${NUM_SAMPLES:-25}"
TAUS="${TAUS:-0.002,0.004,0.008}"
SWEEP_ROOT="${SWEEP_ROOT:-benchmark_suite_result/frontier_tau_sweep_20260706/ruler}"

export TASK_NAME CONTEXT_LEN NUM_SAMPLES

echo "[TAU-SWEEP] task=${TASK_NAME} ctx=${CONTEXT_LEN} n=${NUM_SAMPLES} taus=${TAUS}"

IFS=',' read -r -a tau_arr <<< "${TAUS}"
# Seed from env to pair samples with an arm run in a previous job.
data_override="${DATA_FILE_OVERRIDE:-}"
for tau in "${tau_arr[@]}"; do
  run_name="frontier_tau${tau}_${TASK_NAME}_ctx${CONTEXT_LEN}_n${NUM_SAMPLES}"
  echo "[TAU-SWEEP] === tau=${tau} -> ${SWEEP_ROOT}/${run_name} ==="
  JOINT_KV_STABILITY_THRESHOLD="${tau}" \
  OUTPUT_ROOT="${SWEEP_ROOT}" \
  RUN_NAME="${run_name}" \
  DATA_FILE_OVERRIDE="${data_override}" \
  bash scripts/run_frontier_ruler_batched_one.sh
  # First arm prepares the data; later arms reuse it so samples are paired.
  if [[ -z "${data_override}" ]]; then
    candidate="${SWEEP_ROOT}/${run_name}/data/${TASK_NAME}/validation.jsonl"
    if [[ -s "${candidate}" ]]; then
      data_override="${candidate}"
    fi
  fi
done

echo "[TAU-SWEEP] ===== score table ====="
for tau in "${tau_arr[@]}"; do
  run_name="frontier_tau${tau}_${TASK_NAME}_ctx${CONTEXT_LEN}_n${NUM_SAMPLES}"
  f="${SWEEP_ROOT}/${run_name}/summary/${TASK_NAME}.json"
  if [[ -f "${f}" ]]; then
    score=$(python3 -c "import json;print(json.load(open('${f}')).get('score'))")
    echo "[TAU-SWEEP] tau=${tau} score=${score} summary=${f}"
  else
    echo "[TAU-SWEEP] tau=${tau} MISSING ${f}"
  fi
done
echo "[TAU-SWEEP] done $(date --iso-8601=seconds)"
