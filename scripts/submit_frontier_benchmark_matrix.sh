#!/usr/bin/env bash
set -euo pipefail

# Submit a paired dense-vs-frontier benchmark matrix using the validated wrappers.
# This script only launches jobs; inspect artifacts with benchmark/audit_benchmark_readiness.py.

cd /gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention

TAG="${TAG:-$(date +%Y%m%d_%H%M%S)}"
SLURM_ROOT="${SLURM_ROOT:-slurm_out/frontier_benchmark_matrix_${TAG}}"
MANIFEST="${MANIFEST:-notes/slurm_manifests/frontier_benchmark_matrix_${TAG}.tsv}"
DRY_RUN="${DRY_RUN:-0}"
SBATCH_DEPENDENCY="${SBATCH_DEPENDENCY:-}"

RUN_RULER="${RUN_RULER:-1}"
RULER_TASKS="${RULER_TASKS:-niah_single_1,niah_multikey_2,vt,fwe}"
RULER_CONTEXT_LEN="${RULER_CONTEXT_LEN:-8192}"
RULER_NUM_SAMPLES="${RULER_NUM_SAMPLES:-4}"
RULER_OUTPUT_ROOT="${RULER_OUTPUT_ROOT:-ruler_eval_result/frontier_benchmark_matrix_${TAG}}"
RULER_DATA_OVERRIDE_ROOT="${RULER_DATA_OVERRIDE_ROOT:-}"

RUN_LONGBENCH="${RUN_LONGBENCH:-1}"
LONGBENCH_MAX_EXAMPLES="${LONGBENCH_MAX_EXAMPLES:-64}"
LONGBENCH_LENGTH_FILTER="${LONGBENCH_LENGTH_FILTER:-short}"
LONGBENCH_DIFFICULTY_FILTER="${LONGBENCH_DIFFICULTY_FILTER:-easy}"
LONGBENCH_MAX_INPUT_TOKENS="${LONGBENCH_MAX_INPUT_TOKENS:-8192}"
LONGBENCH_OUTPUT_ROOT="${LONGBENCH_OUTPUT_ROOT:-longbench_v2_hf_result/frontier_benchmark_matrix_${TAG}}"

mkdir -p "${SLURM_ROOT}" "$(dirname "${MANIFEST}")"
printf 'label\tjobid\toutput_dir\tslurm_out\n' > "${MANIFEST}"

submit_job() {
  local label="$1"
  local output_dir="$2"
  local script="$3"
  shift 3
  local slurm_out="${SLURM_ROOT}/${label}-%j.out"
  if [ "${DRY_RUN}" = "1" ]; then
    printf '[DRY_RUN] %s -> %s via %s dependency=%s env=%s\n' "${label}" "${output_dir}" "${script}" "${SBATCH_DEPENDENCY:-none}" "$*"
    printf '%s\tDRY_RUN\t%s\t%s\n' "${label}" "${output_dir}" "${slurm_out}" >> "${MANIFEST}"
    return
  fi
  local dependency_args=()
  if [ -n "${SBATCH_DEPENDENCY}" ]; then
    dependency_args=(--dependency="${SBATCH_DEPENDENCY}")
  fi
  local jobid
  jobid=$(sbatch --parsable \
    --job-name="${label:0:40}" \
    --output="${slurm_out}" \
    "${dependency_args[@]}" \
    --export="ALL,$*" \
    "${script}")
  printf '%s\t%s\t%s\t%s\n' "${label}" "${jobid}" "${output_dir}" "${slurm_out//%j/${jobid}}" >> "${MANIFEST}"
  printf '[SUBMITTED] %s %s\n' "${label}" "${jobid}"
}

if [ "${RUN_RULER}" = "1" ]; then
  IFS=',' read -r -a tasks <<< "${RULER_TASKS}"
  for task in "${tasks[@]}"; do
    task="${task//[[:space:]]/}"
    [ -n "${task}" ] || continue
    dense_label="dense_ruler_ctx${RULER_CONTEXT_LEN}_n${RULER_NUM_SAMPLES}_${task}"
    frontier_label="frontier_ruler_ctx${RULER_CONTEXT_LEN}_n${RULER_NUM_SAMPLES}_${task}"
    data_args="REUSE_DATA=0"
    if [ -n "${RULER_DATA_OVERRIDE_ROOT}" ]; then
      data_file="${RULER_DATA_OVERRIDE_ROOT}/${task}/validation.jsonl"
      data_args="DATA_FILE_OVERRIDE=${data_file},REUSE_DATA=1"
    fi
    submit_job \
      "${dense_label}" \
      "${RULER_OUTPUT_ROOT}/${dense_label}" \
      scripts/run_dense_ruler_batched_one.sh \
      "OUTPUT_ROOT=${RULER_OUTPUT_ROOT},RUN_NAME=${dense_label},TASK_NAME=${task},CONTEXT_LEN=${RULER_CONTEXT_LEN},NUM_SAMPLES=${RULER_NUM_SAMPLES},${data_args}"
    submit_job \
      "${frontier_label}" \
      "${RULER_OUTPUT_ROOT}/${frontier_label}" \
      scripts/run_frontier_ruler_batched_one.sh \
      "OUTPUT_ROOT=${RULER_OUTPUT_ROOT},RUN_NAME=${frontier_label},TASK_NAME=${task},CONTEXT_LEN=${RULER_CONTEXT_LEN},NUM_SAMPLES=${RULER_NUM_SAMPLES},${data_args}"
  done
fi

if [ "${RUN_LONGBENCH}" = "1" ]; then
  dense_label="dense_lbv2_${LONGBENCH_LENGTH_FILTER}_${LONGBENCH_DIFFICULTY_FILTER}_n${LONGBENCH_MAX_EXAMPLES}_l${LONGBENCH_MAX_INPUT_TOKENS}"
  frontier_label="frontier_lbv2_${LONGBENCH_LENGTH_FILTER}_${LONGBENCH_DIFFICULTY_FILTER}_n${LONGBENCH_MAX_EXAMPLES}_l${LONGBENCH_MAX_INPUT_TOKENS}"
  submit_job \
    "${dense_label}" \
    "${LONGBENCH_OUTPUT_ROOT}/${dense_label}" \
    scripts/run_dense_longbench_v2_one.sh \
    "OUTPUT_DIR=${LONGBENCH_OUTPUT_ROOT}/${dense_label},MAX_EXAMPLES=${LONGBENCH_MAX_EXAMPLES},LENGTH_FILTER=${LONGBENCH_LENGTH_FILTER},DIFFICULTY_FILTER=${LONGBENCH_DIFFICULTY_FILTER},MAX_INPUT_TOKENS=${LONGBENCH_MAX_INPUT_TOKENS},TEMPERATURE=0.0"
  submit_job \
    "${frontier_label}" \
    "${LONGBENCH_OUTPUT_ROOT}/${frontier_label}" \
    scripts/run_frontier_longbench_v2_one.sh \
    "OUTPUT_DIR=${LONGBENCH_OUTPUT_ROOT}/${frontier_label},MAX_EXAMPLES=${LONGBENCH_MAX_EXAMPLES},LENGTH_FILTER=${LONGBENCH_LENGTH_FILTER},DIFFICULTY_FILTER=${LONGBENCH_DIFFICULTY_FILTER},MAX_INPUT_TOKENS=${LONGBENCH_MAX_INPUT_TOKENS},TEMPERATURE=0.0"
fi

printf '[MANIFEST] %s\n' "${MANIFEST}"
