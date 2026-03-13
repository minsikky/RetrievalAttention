#!/bin/bash
set -euo pipefail

MODES="${MODES:-baseline dynamic online full_dense}"
ENTRY_LIST="${ENTRY_LIST:-12 24 48}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"
NUM_QUERIES="${NUM_QUERIES:-3}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-0}"
TOKEN_BUDGET_OVERRIDE="${TOKEN_BUDGET_OVERRIDE:-100}"
OUTPUT_ROOT="${OUTPUT_ROOT:-generated_memory_eval_result/length_sweep_s16_fullgpu}"
STATIC_START="${RETRIEVALATTN_STATIC_PATTERN_START:-16}"
STATIC_END="${RETRIEVALATTN_STATIC_PATTERN_END:-16}"
DECODE_BACKEND="${RETRIEVALATTN_DECODE_BACKEND:-roar_cuda_fullgpu}"
PREFILL_FILLER_REPEATS="${PREFILL_FILLER_REPEATS:-0}"
MIN_PROMPT_TOKENS="${MIN_PROMPT_TOKENS:-$((STATIC_START + STATIC_END + 64))}"

mkdir -p "${OUTPUT_ROOT}"
JOB_LOG="${OUTPUT_ROOT}/submitted_jobs.tsv"
echo -e "job_id\tmode\tnum_entries\toutput_dir" > "${JOB_LOG}"

echo "[INFO] OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "[INFO] MODES=${MODES}"
echo "[INFO] ENTRY_LIST=${ENTRY_LIST}"
echo "[INFO] NUM_SAMPLES=${NUM_SAMPLES}"
echo "[INFO] NUM_QUERIES=${NUM_QUERIES}"
echo "[INFO] STATIC_START=${STATIC_START}"
echo "[INFO] STATIC_END=${STATIC_END}"
echo "[INFO] DECODE_BACKEND=${DECODE_BACKEND}"
echo "[INFO] MIN_PROMPT_TOKENS=${MIN_PROMPT_TOKENS}"

for mode in ${MODES}; do
  for entries in ${ENTRY_LIST}; do
    output_dir="${OUTPUT_ROOT}/${mode}_e${entries}"
    job_id=$(
      ONLINE_MODE="${mode}" \
      NUM_SAMPLES="${NUM_SAMPLES}" \
      NUM_ENTRIES="${entries}" \
      NUM_QUERIES="${NUM_QUERIES}" \
      MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
      TOKEN_BUDGET_OVERRIDE="${TOKEN_BUDGET_OVERRIDE}" \
      PREFILL_FILLER_REPEATS="${PREFILL_FILLER_REPEATS}" \
      MIN_PROMPT_TOKENS="${MIN_PROMPT_TOKENS}" \
      OUTPUT_DIR="${output_dir}" \
      RETRIEVALATTN_DECODE_BACKEND="${DECODE_BACKEND}" \
      RETRIEVALATTN_STATIC_PATTERN_START="${STATIC_START}" \
      RETRIEVALATTN_STATIC_PATTERN_END="${STATIC_END}" \
      sbatch --parsable benchmark/run_generated_memory_online.sh
    )
    echo -e "${job_id}\t${mode}\t${entries}\t${output_dir}" | tee -a "${JOB_LOG}"
  done
done

echo "[INFO] Wrote submission log to ${JOB_LOG}"
