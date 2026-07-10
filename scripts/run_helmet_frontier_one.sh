#!/usr/bin/env bash
#SBATCH --job-name=helmet-frontier
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128000m
#SBATCH --time=10:00:00
#SBATCH --account=zhengya0
#SBATCH --partition=gpu_mig40,spgpu,gpu-rtx6000
#SBATCH --gpus-per-node=1
set -euo pipefail

# One HELMET dataset through the canonical frontier (or dense reference).
# Pipeline: HELMET prompts (its own templates/truncation, converted to
# RULER-style jsonl under .venv) -> our validated runner (cu128 env) ->
# HELMET per-dataset metrics (.venv). Phase C of the benchmark plan.
#
# Env inputs:
#   DATASET (e.g. kilt_nq)      TEST_FILE (HELMET-relative path, may be "")
#   DEMO_FILE                   INPUT_MAX_LENGTH (default 131072)
#   GENERATION_MAX_LENGTH       SHOTS (default 2)
#   MAX_TEST_SAMPLES            ARM (frontier|dense, default frontier)
#   STOP_NEW_LINE (0/1)         SEED (default 42)

cd /gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention

DATASET="${DATASET:?DATASET required}"
TEST_FILE="${TEST_FILE:-}"
DEMO_FILE="${DEMO_FILE:-}"
INPUT_MAX_LENGTH="${INPUT_MAX_LENGTH:-131072}"
GENERATION_MAX_LENGTH="${GENERATION_MAX_LENGTH:-20}"
SHOTS="${SHOTS:-2}"
MAX_TEST_SAMPLES="${MAX_TEST_SAMPLES:-16}"
SEED="${SEED:-42}"
ARM="${ARM:-frontier}"
STOP_NEW_LINE="${STOP_NEW_LINE:-0}"
HELMET_ROOT="${HELMET_ROOT:-benchmark_suite_result/helmet_20260706}"

module purge
module load python/3.10.4
export LD_LIBRARY_PATH="/sw/pkgs/arc/python/3.10.4/lib:${LD_LIBRARY_PATH:-}"

data_dir="${HELMET_ROOT}/data/${DATASET}_ctx${INPUT_MAX_LENGTH}_n${MAX_TEST_SAMPLES}_s${SEED}"
data_file="${PWD}/${data_dir}/validation.jsonl"
mkdir -p "${data_dir}"

if [ ! -s "${data_file}" ]; then
  echo "[helmet] preparing ${DATASET} -> ${data_file}"
  .venv/bin/python benchmark/helmet/prepare_helmet_data.py \
    --dataset "${DATASET}" \
    --test_file "${TEST_FILE}" \
    --demo_file "${DEMO_FILE}" \
    --input_max_length "${INPUT_MAX_LENGTH}" \
    --generation_max_length "${GENERATION_MAX_LENGTH}" \
    --shots "${SHOTS}" \
    --max_test_samples "${MAX_TEST_SAMPLES}" \
    --seed "${SEED}" \
    --tokenizer ".hf_cache/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659" \
    --output "${data_file}"
else
  echo "[helmet] reusing ${data_file}"
fi

# Runner env (cu128 stack, same fixes as the tau-sweep wrapper).
export HF_VENV_DIR="${HF_VENV_DIR:-.venv_cu128}"
export HF_EXTRA_PYTHONPATH="${PWD}/.hf_pydeps_cu128:${PWD}/.hf_pydeps_cu128_scipy"
export LD_LIBRARY_PATH="${PWD}/.venv/lib/python3.10/site-packages/numpy.libs:${PWD}/.hf_pydeps_cu128_scipy/scipy.libs:${LD_LIBRARY_PATH:-}"
export PREFILL_CHUNK_SIZE="${PREFILL_CHUNK_SIZE:-8192}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export FRONTIER_EMPTY_CACHE_AFTER_PREFILL_CHUNK="${FRONTIER_EMPTY_CACHE_AFTER_PREFILL_CHUNK:-1}"
# Standard config on all three eligible partitions (gpu_mig40/spgpu/gpu-rtx6000):
# keyT cache off inherits from run_frontier_ruler_batched_one.sh's flipped
# FRONTIER_DENSE_KEY_T_CACHE default (0); memory trace on for the ~34 GiB peak.
export SELECTOR_PQ_JOINT_MEMORY_TRACE="${SELECTOR_PQ_JOINT_MEMORY_TRACE:-1}"

export TASK_NAME="${DATASET}"
export CONTEXT_LEN="${INPUT_MAX_LENGTH}"
# The converted file may hold more rows than MAX_TEST_SAMPLES (kilt dep6
# datasets expand each question into 6 gold-depth rows). Generate ALL
# rows of the file; scoring joins by index.
export NUM_SAMPLES=0
export MAX_NEW_TOKENS="${GENERATION_MAX_LENGTH}"
export DATA_FILE_OVERRIDE="${data_file}"
export SKIP_RULER_EVAL=1
export OUTPUT_ROOT="${HELMET_ROOT}/runs"

if [ "${ARM}" = "dense" ]; then
  export MODE=dense_batched
  export RUN_NAME="dense_${DATASET}_ctx${INPUT_MAX_LENGTH}_n${MAX_TEST_SAMPLES}"
  bash scripts/run_ruler_pagedpq_stream_smoke_one.sh
else
  export JOINT_KV_STABILITY_THRESHOLD="${JOINT_KV_STABILITY_THRESHOLD:-0.004}"
  export RUN_NAME="frontier_tau${JOINT_KV_STABILITY_THRESHOLD}_${DATASET}_ctx${INPUT_MAX_LENGTH}_n${MAX_TEST_SAMPLES}"
  bash scripts/run_frontier_ruler_batched_one.sh
fi

pred_file="${OUTPUT_ROOT}/${RUN_NAME}/pred/${TASK_NAME}.jsonl"
summary_out="${OUTPUT_ROOT}/${RUN_NAME}/summary/${TASK_NAME}.helmet.json"
stop_flag=()
if [ "${STOP_NEW_LINE}" = "1" ]; then stop_flag=(--stop_new_line); fi
# Score directly against the converted data file (gold stored at
# conversion time; pred index == row index by construction). NEVER score
# via a dataset reload: the HF reload is not order-stable across
# processes, and a positional join against it can silently score the
# wrong gold rows (frontier kilt_nq scored 0.0000 on predictions
# containing the gold answers verbatim).
.venv/bin/python benchmark/helmet/eval_helmet_preds.py \
  --dataset "${DATASET}" \
  --data_file "${data_file}" \
  --pred_file "${PWD}/${pred_file}" \
  --summary_out "${PWD}/${summary_out}" \
  "${stop_flag[@]}"
echo "[helmet] done ${summary_out}"
