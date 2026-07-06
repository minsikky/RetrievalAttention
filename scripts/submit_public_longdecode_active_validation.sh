#!/usr/bin/env bash
set -euo pipefail

cd /gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
SUITE_NAME="${SUITE_NAME:-public_longdecode_active_validation_${STAMP}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-public_longdecode_result/${SUITE_NAME}}"
SLURM_ROOT="${SLURM_ROOT:-slurm_out/${SUITE_NAME}}"
MANIFEST="${MANIFEST:-notes/slurm_manifests/${SUITE_NAME}.tsv}"
PLAN="${PLAN:-${OUTPUT_ROOT}/submit_plan.sh}"
SUBMIT="${SUBMIT:-0}"
PARTITIONS="${PARTITIONS:-spgpu}"
HF_VENV_DIR="${HF_VENV_DIR:-.venv_cu128}"
HF_EXTRA_PYTHONPATH="${HF_EXTRA_PYTHONPATH:-.hf_pydeps_cu128}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0 8.6 12.0}"

mkdir -p "${OUTPUT_ROOT}" "${SLURM_ROOT}" "$(dirname "${MANIFEST}")"
printf "label\tjobid\toutput_dir\tslurm_out\tbenchmark\tmode\toffset\tmax_examples\tmax_new_tokens\tmin_new_tokens\tforce_max_new_tokens\n" > "${MANIFEST}"
printf "#!/usr/bin/env bash\nset -euo pipefail\ncd /gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention\n\n" > "${PLAN}"

HF_MODEL_PRESET="${HF_MODEL_PRESET:-qwen3_8b}"
source scripts/hf_model_presets.sh
resolve_hf_model_preset "${HF_MODEL_PRESET}" || exit $?
MODEL_NAME="${MODEL_NAME:-${PRESET_MODEL_NAME}}"
HF_LANGUAGE_MODEL_ONLY="${HF_LANGUAGE_MODEL_ONLY:-${PRESET_HF_LANGUAGE_MODEL_ONLY:-1}}"
USE_CHAT_TEMPLATE="${USE_CHAT_TEMPLATE:-${PRESET_USE_CHAT_TEMPLATE:-1}}"
DISABLE_THINKING="${DISABLE_THINKING:-${PRESET_DISABLE_THINKING:-1}}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
MODES_CSV="${MODES:-dense,pagedpq}"
IFS=, read -r -a MODES <<< "${MODES_CSV}"

AIME_EXAMPLES="${AIME_EXAMPLES:-3}"
AIME_OFFSET="${AIME_OFFSET:-0}"
AIME_MAX_NEW_TOKENS="${AIME_MAX_NEW_TOKENS:-8192}"
AIME_MIN_NEW_TOKENS="${AIME_MIN_NEW_TOKENS:-8192}"
AIME_FORCE_MAX_NEW_TOKENS="${AIME_FORCE_MAX_NEW_TOKENS:-1}"

LIVE_CODE_EXAMPLES="${LIVE_CODE_EXAMPLES:-3}"
LIVE_CODE_OFFSET="${LIVE_CODE_OFFSET:-2}"
LIVE_CODE_MAX_NEW_TOKENS="${LIVE_CODE_MAX_NEW_TOKENS:-8192}"
LIVE_CODE_MIN_NEW_TOKENS="${LIVE_CODE_MIN_NEW_TOKENS:-8192}"
LIVE_CODE_FORCE_MAX_NEW_TOKENS="${LIVE_CODE_FORCE_MAX_NEW_TOKENS:-1}"
LIVE_CODE_RELEASE="${LIVE_CODE_RELEASE:-release_v6}"
LIVE_CODE_EVALUATE_CODE="${LIVE_CODE_EVALUATE_CODE:-1}"
LIVE_CODE_CODE_EVAL_TIMEOUT="${LIVE_CODE_CODE_EVAL_TIMEOUT:-6}"

LONGGEN_EXAMPLES="${LONGGEN_EXAMPLES:-1}"
LONGGEN_OFFSET="${LONGGEN_OFFSET:-0}"
LONGGEN_MAX_NEW_TOKENS="${LONGGEN_MAX_NEW_TOKENS:-8192}"
LONGGEN_MIN_NEW_TOKENS="${LONGGEN_MIN_NEW_TOKENS:-8192}"
LONGGEN_FORCE_MAX_NEW_TOKENS="${LONGGEN_FORCE_MAX_NEW_TOKENS:-1}"

submit_one() {
  local bench="$1"
  local mode="$2"
  local offset="$3"
  local count="$4"
  local max_new="$5"
  local min_new="$6"
  local force_max="$7"
  local evaluate_code="$8"
  local code_eval_timeout="$9"

  if [ "${count}" -le 0 ]; then
    return
  fi

  local label="${mode}_${bench}_off${offset}_n${count}_tok${max_new}"
  local out_dir="${OUTPUT_ROOT}/${label}"
  local slurm_out="${SLURM_ROOT}/${label}-%j.out"
  local export_args=(
    "ALL"
    "HF_MODEL_PRESET=${HF_MODEL_PRESET}"
    "BENCHMARK=${bench}"
    "ATTENTION_MODE=${mode}"
    "MODEL_NAME=${MODEL_NAME}"
    "HF_LANGUAGE_MODEL_ONLY=${HF_LANGUAGE_MODEL_ONLY}"
    "USE_CHAT_TEMPLATE=${USE_CHAT_TEMPLATE}"
    "DISABLE_THINKING=${DISABLE_THINKING}"
    "OUTPUT_DIR=${out_dir}"
    "RUN_NAME=${label}"
    "OUTPUT_ROOT=${OUTPUT_ROOT}"
    "MAX_EXAMPLES=${count}"
    "TASK_OFFSET=${offset}"
    "MAX_NEW_TOKENS=${max_new}"
    "MIN_NEW_TOKENS=${min_new}"
    "FORCE_MAX_NEW_TOKENS=${force_max}"
    "LOCAL_FILES_ONLY=${LOCAL_FILES_ONLY}"
    "EVALUATE_CODE=${evaluate_code}"
    "CODE_EVAL_TIMEOUT=${code_eval_timeout}"
    "LIVE_CODE_RELEASE=${LIVE_CODE_RELEASE}"
    "HF_VENV_DIR=${HF_VENV_DIR}"
    "HF_EXTRA_PYTHONPATH=${HF_EXTRA_PYTHONPATH}"
    "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
  )
  local export_csv
  export_csv="$(IFS=,; echo "${export_args[*]}")"
  printf 'sbatch --parsable --partition=%q --output=%q --export=%q benchmark/run_public_longdecode_hf.sh\n' \
    "${PARTITIONS}" "${slurm_out}" "${export_csv}" >> "${PLAN}"

  local jobid="DRYRUN"
  local slurm_out_rendered="${slurm_out/\%j/DRYRUN}"
  if [ "${SUBMIT}" = "1" ] || [ "${SUBMIT}" = "true" ] || [ "${SUBMIT}" = "yes" ]; then
    jobid="$(sbatch --parsable --partition="${PARTITIONS}" --output="${slurm_out}" --export="${export_csv}" benchmark/run_public_longdecode_hf.sh)"
    slurm_out_rendered="${slurm_out/\%j/${jobid}}"
  fi
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${label}" "${jobid}" "${out_dir}" "${slurm_out_rendered}" "${bench}" "${mode}" \
    "${offset}" "${count}" "${max_new}" "${min_new}" "${force_max}" | tee -a "${MANIFEST}"
}

for mode in "${MODES[@]}"; do
  submit_one "aime24" "${mode}" "${AIME_OFFSET}" "${AIME_EXAMPLES}" \
    "${AIME_MAX_NEW_TOKENS}" "${AIME_MIN_NEW_TOKENS}" "${AIME_FORCE_MAX_NEW_TOKENS}" 0 6
  submit_one "livecodebench_codegen" "${mode}" "${LIVE_CODE_OFFSET}" "${LIVE_CODE_EXAMPLES}" \
    "${LIVE_CODE_MAX_NEW_TOKENS}" "${LIVE_CODE_MIN_NEW_TOKENS}" "${LIVE_CODE_FORCE_MAX_NEW_TOKENS}" \
    "${LIVE_CODE_EVALUATE_CODE}" "${LIVE_CODE_CODE_EVAL_TIMEOUT}"
  submit_one "longgenbench_sgt_short" "${mode}" "${LONGGEN_OFFSET}" "${LONGGEN_EXAMPLES}" \
    "${LONGGEN_MAX_NEW_TOKENS}" "${LONGGEN_MIN_NEW_TOKENS}" "${LONGGEN_FORCE_MAX_NEW_TOKENS}" 0 6
done

echo "[INFO] Manifest: ${MANIFEST}"
echo "[INFO] Submit plan: ${PLAN}"
if [ "${SUBMIT}" != "1" ] && [ "${SUBMIT}" != "true" ] && [ "${SUBMIT}" != "yes" ]; then
  echo "[INFO] Dry run only. Re-run with SUBMIT=1 to launch jobs."
fi
