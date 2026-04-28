#!/bin/bash
set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-generated_memory_eval_result/full_dense_transition_sweep_v1}"
SEEDS="${SEEDS:-2025 2026 2027}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"
TIME_LIMIT="${TIME_LIMIT:-360:00}"

# Two coarse bands:
# - low_q: should stay mostly in the success region longer
# - high_q: intended to cross into failure earlier
LOW_Q_SETTINGS="${LOW_Q_SETTINGS:-12:3 24:6 48:12 96:24 144:36 192:48 240:60}"
HIGH_Q_SETTINGS="${HIGH_Q_SETTINGS:-24:10 48:20 96:40 144:60 192:80 240:100}"

mkdir -p "${OUTPUT_ROOT}"
JOB_LOG="${OUTPUT_ROOT}/submitted_jobs.tsv"
echo -e "job_id\tband\tnum_entries\tnum_queries\tseed\toutput_dir" > "${JOB_LOG}"

submit_setting() {
  local band="$1"
  local entries="$2"
  local queries="$3"
  local seed="$4"
  local output_dir="${OUTPUT_ROOT}/${band}_e${entries}_q${queries}_s${seed}"

  job_id=$(
    ONLINE_MODE="full_dense" \
    NUM_SAMPLES="${NUM_SAMPLES}" \
    SEED="${seed}" \
    NUM_ENTRIES="${entries}" \
    NUM_QUERIES="${queries}" \
    OUTPUT_DIR="${output_dir}" \
    sbatch --parsable --time="${TIME_LIMIT}" benchmark/run_generated_memory_online.sh
  )

  echo -e "${job_id}\t${band}\t${entries}\t${queries}\t${seed}\t${output_dir}" | tee -a "${JOB_LOG}"
}

echo "[INFO] OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "[INFO] SEEDS=${SEEDS}"
echo "[INFO] NUM_SAMPLES=${NUM_SAMPLES}"
echo "[INFO] TIME_LIMIT=${TIME_LIMIT}"
echo "[INFO] LOW_Q_SETTINGS=${LOW_Q_SETTINGS}"
echo "[INFO] HIGH_Q_SETTINGS=${HIGH_Q_SETTINGS}"

for pair in ${LOW_Q_SETTINGS}; do
  entries="${pair%%:*}"
  queries="${pair##*:}"
  for seed in ${SEEDS}; do
    submit_setting "low_q" "${entries}" "${queries}" "${seed}"
  done
done

for pair in ${HIGH_Q_SETTINGS}; do
  entries="${pair%%:*}"
  queries="${pair##*:}"
  for seed in ${SEEDS}; do
    submit_setting "high_q" "${entries}" "${queries}" "${seed}"
  done
done

echo "[INFO] Wrote submission log to ${JOB_LOG}"
