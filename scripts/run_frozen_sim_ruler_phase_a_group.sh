#!/usr/bin/env bash
#SBATCH --job-name=frozensim-ruler-phasea
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128000m
#SBATCH --time=2-00:00:00
#SBATCH --account=zhengya0
#SBATCH --partition=gpu-rtx6000,spgpu
#SBATCH --gpus-per-node=1
set -euo pipefail

# Current frozen algorithm task-quality rerun.
# Runs tasks serially in one Slurm allocation to respect the <=6 active-job cap.
# Each arm reuses the paired dense validation.jsonl from hard_ruler_20260706.

cd /gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention

export HF_VENV_DIR="${HF_VENV_DIR:-.venv_cu128}"
export HF_EXTRA_PYTHONPATH="${PWD}/.hf_pydeps_cu128:${PWD}/.hf_pydeps_cu128_scipy"
export LD_LIBRARY_PATH="${PWD}/.venv/lib/python3.10/site-packages/numpy.libs:${PWD}/.hf_pydeps_cu128_scipy/scipy.libs:${LD_LIBRARY_PATH:-}"
export PREFILL_CHUNK_SIZE="${PREFILL_CHUNK_SIZE:-8192}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export FRONTIER_EMPTY_CACHE_AFTER_PREFILL_CHUNK="${FRONTIER_EMPTY_CACHE_AFTER_PREFILL_CHUNK:-1}"

export JOINT_KV_STABILITY_THRESHOLD="${JOINT_KV_STABILITY_THRESHOLD:-0.004}"
export LOGIT_BUFFER_FORMAT="${LOGIT_BUFFER_FORMAT:-e4m3}"
export JOINT_KV_PRECISION_TIERS="${JOINT_KV_PRECISION_TIERS:-1}"
unset JOINT_KV_DEESCALATE

export OUTPUT_ROOT="${OUTPUT_ROOT:-benchmark_suite_result/frozen_sim_20260707/ruler_phase_a}"
export NUM_SAMPLES="${NUM_SAMPLES:-16}"

TASK_SPECS="${TASK_SPECS:-fwe:131072 niah_multikey_3:131072 qa_1:131072 qa_2:131072 cwe:65536}"

echo "[frozensim-phasea] start $(date --iso-8601=seconds)"
echo "[frozensim-phasea] output_root=${OUTPUT_ROOT}"
echo "[frozensim-phasea] task_specs=${TASK_SPECS}"
echo "[frozensim-phasea] tau=${JOINT_KV_STABILITY_THRESHOLD} logit=${LOGIT_BUFFER_FORMAT} precision=${JOINT_KV_PRECISION_TIERS}"

for spec in ${TASK_SPECS}; do
  task="${spec%%:*}"
  ctx="${spec##*:}"
  data_file="benchmark_suite_result/hard_ruler_20260706/ruler/dense_${task}_ctx${ctx}_n${NUM_SAMPLES}/data/${task}/validation.jsonl"
  if [[ ! -s "${data_file}" ]]; then
    echo "[frozensim-phasea] missing paired data: ${data_file}" >&2
    exit 2
  fi

  export TASK_NAME="${task}"
  export CONTEXT_LEN="${ctx}"
  export DATA_FILE_OVERRIDE="${PWD}/${data_file}"
  export RUN_NAME="frozensim_${task}_ctx${ctx}_n${NUM_SAMPLES}"

  echo "[frozensim-phasea] === ${task}@${ctx} using ${data_file} ==="
  bash scripts/run_frontier_ruler_batched_one.sh
done

echo "[frozensim-phasea] done $(date --iso-8601=seconds)"
