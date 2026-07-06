#!/usr/bin/env bash
set -euo pipefail

cd /gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention

# Full public long-decode matrix. Safe by default: set SUBMIT=1 to actually
# launch Slurm jobs. With SUBMIT=0 this writes the manifest and submit_plan.sh.
#
# Defaults intentionally cap this at O(10^2) jobs. For exhaustive coverage,
# override the *_TOTAL_EXAMPLES and *_SHARD_SIZE knobs explicitly.

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
SUITE_NAME="${SUITE_NAME:-public_longdecode_full_${STAMP}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-public_longdecode_result/${SUITE_NAME}}"
SLURM_ROOT="${SLURM_ROOT:-slurm_out/${SUITE_NAME}}"
MANIFEST="${MANIFEST:-notes/slurm_manifests/${SUITE_NAME}.tsv}"
PLAN="${PLAN:-${OUTPUT_ROOT}/submit_plan.sh}"
SUBMIT="${SUBMIT:-0}"
PARTITIONS="${PARTITIONS:-spgpu}"
HF_VENV_DIR="${HF_VENV_DIR:-}"
HF_EXTRA_PYTHONPATH="${HF_EXTRA_PYTHONPATH:-}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-}"

mkdir -p "${OUTPUT_ROOT}" "${SLURM_ROOT}" "$(dirname "${MANIFEST}")"
printf "label\tjobid\toutput_dir\tslurm_out\tbenchmark\tmode\toffset\tmax_examples\tmax_new_tokens\tmin_new_tokens\tforce_max_new_tokens\n" > "${MANIFEST}"
printf "#!/usr/bin/env bash\nset -euo pipefail\ncd /gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention\n\n" > "${PLAN}"

HF_MODEL_PRESET="${HF_MODEL_PRESET:-qwen3_8b}"
source scripts/hf_model_presets.sh
resolve_hf_model_preset "${HF_MODEL_PRESET}" || exit $?

QWEN3_AIME_MAX_NEW_TOKENS=38912
QWEN3_DEFAULT_MAX_NEW_TOKENS=32768
USE_QWEN3_OFFICIAL_EVAL_DEFAULTS="${USE_QWEN3_OFFICIAL_EVAL_DEFAULTS:-1}"
QWEN3_EVAL_MODE="${QWEN3_EVAL_MODE:-thinking}"
is_qwen3=0
if [[ "${HF_MODEL_PRESET}" == qwen3* ]] || [[ "${PRESET_MODEL_NAME}" == *Qwen3* ]] || [[ "${PRESET_MODEL_NAME}" == *Qwen--Qwen3* ]]; then
  is_qwen3=1
fi
if [ "${USE_QWEN3_OFFICIAL_EVAL_DEFAULTS}" = "1" ] && [ "${is_qwen3}" = "1" ]; then
  case "${QWEN3_EVAL_MODE}" in
    thinking|think)
      official_disable_thinking=0
      official_temperature=0.6
      official_top_p=0.95
      official_top_k=20
      ;;
    nonthinking|non-thinking|no_think|nothink)
      official_disable_thinking=1
      official_temperature=0.7
      official_top_p=0.8
      official_top_k=20
      ;;
    *)
      echo "[ERROR] Unknown QWEN3_EVAL_MODE=${QWEN3_EVAL_MODE}" >&2
      exit 2
      ;;
  esac
else
  official_disable_thinking="${PRESET_DISABLE_THINKING:-1}"
  official_temperature=0.0
  official_top_p=1.0
  official_top_k=0
fi

MODEL_NAME="${MODEL_NAME:-${PRESET_MODEL_NAME}}"
HF_LANGUAGE_MODEL_ONLY="${HF_LANGUAGE_MODEL_ONLY:-${PRESET_HF_LANGUAGE_MODEL_ONLY:-1}}"
USE_CHAT_TEMPLATE="${USE_CHAT_TEMPLATE:-${PRESET_USE_CHAT_TEMPLATE:-1}}"
DISABLE_THINKING="${DISABLE_THINKING:-${PUBLIC_DISABLE_THINKING:-${official_disable_thinking}}}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-120000}"
SELECTION="${SELECTION:-first}"
TEMPERATURE="${TEMPERATURE:-${PUBLIC_TEMPERATURE:-${official_temperature}}}"
TOP_P="${TOP_P:-${PUBLIC_TOP_P:-${official_top_p}}}"
TOP_K="${TOP_K:-${PUBLIC_TOP_K:-${official_top_k}}}"
MODES_CSV="${MODES:-dense,pagedpq}"
IFS=, read -r -a MODES <<< "${MODES_CSV}"

AIME_MAX_NEW_TOKENS="${AIME_MAX_NEW_TOKENS:-${REASONING_MAX_NEW_TOKENS:-${QWEN3_AIME_MAX_NEW_TOKENS}}}"
GPQA_MAX_NEW_TOKENS="${GPQA_MAX_NEW_TOKENS:-${REASONING_MAX_NEW_TOKENS:-${QWEN3_DEFAULT_MAX_NEW_TOKENS}}}"
AIME_TOTAL_EXAMPLES="${AIME_TOTAL_EXAMPLES:-30}"
AIME_SHARD_SIZE="${AIME_SHARD_SIZE:-30}"
GPQA_TOTAL_EXAMPLES="${GPQA_TOTAL_EXAMPLES:-50}"
GPQA_SHARD_SIZE="${GPQA_SHARD_SIZE:-25}"

LIVE_CODE_TOTAL_EXAMPLES="${LIVE_CODE_TOTAL_EXAMPLES:-100}"
LIVE_CODE_SHARD_SIZE="${LIVE_CODE_SHARD_SIZE:-10}"
LIVE_CODE_START_OFFSET="${LIVE_CODE_START_OFFSET:-0}"
LIVE_CODE_MAX_NEW_TOKENS="${LIVE_CODE_MAX_NEW_TOKENS:-${QWEN3_DEFAULT_MAX_NEW_TOKENS}}"
LIVE_CODE_MIN_NEW_TOKENS="${LIVE_CODE_MIN_NEW_TOKENS:-0}"
LIVE_CODE_FORCE_MAX_NEW_TOKENS="${LIVE_CODE_FORCE_MAX_NEW_TOKENS:-0}"
LIVE_CODE_RELEASE="${LIVE_CODE_RELEASE:-release_v6}"
LIVE_CODE_EVALUATE_CODE="${LIVE_CODE_EVALUATE_CODE:-1}"
LIVE_CODE_CODE_EVAL_TIMEOUT="${LIVE_CODE_CODE_EVAL_TIMEOUT:-6}"

LONGGEN_SGT_SHORT_TOTAL_EXAMPLES="${LONGGEN_SGT_SHORT_TOTAL_EXAMPLES:-32}"
LONGGEN_SGT_SHORT_SHARD_SIZE="${LONGGEN_SGT_SHORT_SHARD_SIZE:-4}"
LONGGEN_SGT_SHORT_MAX_NEW_TOKENS="${LONGGEN_SGT_SHORT_MAX_NEW_TOKENS:-16384}"
LONGGEN_SGT_SHORT_MIN_NEW_TOKENS="${LONGGEN_SGT_SHORT_MIN_NEW_TOKENS:-8192}"

LONGGEN_SGT_LONG_TOTAL_EXAMPLES="${LONGGEN_SGT_LONG_TOTAL_EXAMPLES:-16}"
LONGGEN_SGT_LONG_SHARD_SIZE="${LONGGEN_SGT_LONG_SHARD_SIZE:-2}"
LONGGEN_SGT_LONG_MAX_NEW_TOKENS="${LONGGEN_SGT_LONG_MAX_NEW_TOKENS:-32768}"
LONGGEN_SGT_LONG_MIN_NEW_TOKENS="${LONGGEN_SGT_LONG_MIN_NEW_TOKENS:-16384}"

LONGGEN_GSM8K_QUESTION_LIMIT="${LONGGEN_GSM8K_QUESTION_LIMIT:-256}"
LONGGEN_GSM8K_K="${LONGGEN_GSM8K_K:-32}"
LONGGEN_GSM8K_TOTAL_EXAMPLES="${LONGGEN_GSM8K_TOTAL_EXAMPLES:-$(((LONGGEN_GSM8K_QUESTION_LIMIT + LONGGEN_GSM8K_K - 1) / LONGGEN_GSM8K_K))}"
LONGGEN_GSM8K_SHARD_SIZE="${LONGGEN_GSM8K_SHARD_SIZE:-2}"
LONGGEN_GSM8K_MAX_NEW_TOKENS="${LONGGEN_GSM8K_MAX_NEW_TOKENS:-16384}"
LONGGEN_GSM8K_MIN_NEW_TOKENS="${LONGGEN_GSM8K_MIN_NEW_TOKENS:-8192}"

INCLUDE_AIME="${INCLUDE_AIME:-1}"
INCLUDE_GPQA="${INCLUDE_GPQA:-1}"
INCLUDE_LIVE_CODE="${INCLUDE_LIVE_CODE:-1}"
INCLUDE_LONGGEN_SGT_SHORT="${INCLUDE_LONGGEN_SGT_SHORT:-1}"
INCLUDE_LONGGEN_SGT_LONG="${INCLUDE_LONGGEN_SGT_LONG:-1}"
INCLUDE_LONGGEN_GSM8K="${INCLUDE_LONGGEN_GSM8K:-1}"

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
  local temperature="${10:-${TEMPERATURE}}"
  local top_p="${11:-${TOP_P}}"
  local top_k="${12:-${TOP_K}}"
  local disable_thinking="${13:-${DISABLE_THINKING}}"

  local label="${mode}_${bench}_off${offset}_n${count}"
  local out_dir="${OUTPUT_ROOT}/${label}"
  local slurm_out="${SLURM_ROOT}/${label}-%j.out"
  local slurm_out_rendered="${slurm_out/\%j/DRYRUN}"

  local export_args=(
    "ALL"
    "HF_MODEL_PRESET=${HF_MODEL_PRESET}"
    "BENCHMARK=${bench}"
    "ATTENTION_MODE=${mode}"
    "MODEL_NAME=${MODEL_NAME}"
    "HF_LANGUAGE_MODEL_ONLY=${HF_LANGUAGE_MODEL_ONLY}"
    "USE_CHAT_TEMPLATE=${USE_CHAT_TEMPLATE}"
    "DISABLE_THINKING=${disable_thinking}"
    "OUTPUT_DIR=${out_dir}"
    "RUN_NAME=${label}"
    "OUTPUT_ROOT=${OUTPUT_ROOT}"
    "MAX_EXAMPLES=${count}"
    "TASK_OFFSET=${offset}"
    "SELECTION=${SELECTION}"
    "MAX_INPUT_TOKENS=${MAX_INPUT_TOKENS}"
    "MAX_NEW_TOKENS=${max_new}"
    "MIN_NEW_TOKENS=${min_new}"
    "TEMPERATURE=${temperature}"
    "TOP_P=${top_p}"
    "TOP_K=${top_k}"
    "FORCE_MAX_NEW_TOKENS=${force_max}"
    "LOCAL_FILES_ONLY=${LOCAL_FILES_ONLY}"
    "EVALUATE_CODE=${evaluate_code}"
    "CODE_EVAL_TIMEOUT=${code_eval_timeout}"
    "LIVE_CODE_RELEASE=${LIVE_CODE_RELEASE}"
    "LONGGENBENCH_GSM8K_K=${LONGGEN_GSM8K_K}"
    "LONGGENBENCH_GSM8K_QUESTION_LIMIT=${LONGGEN_GSM8K_QUESTION_LIMIT}"
    "HF_VENV_DIR=${HF_VENV_DIR}"
    "HF_EXTRA_PYTHONPATH=${HF_EXTRA_PYTHONPATH}"
    "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
  )
  local export_csv
  export_csv="$(IFS=,; echo "${export_args[*]}")"

  printf 'sbatch --parsable --partition=%q --output=%q --export=%q benchmark/run_public_longdecode_hf.sh\n' \
    "${PARTITIONS}" "${slurm_out}" "${export_csv}" >> "${PLAN}"

  local jobid="DRYRUN"
  if [ "${SUBMIT}" = "1" ] || [ "${SUBMIT}" = "true" ] || [ "${SUBMIT}" = "yes" ]; then
    jobid="$(sbatch --parsable --partition="${PARTITIONS}" --output="${slurm_out}" --export="${export_csv}" benchmark/run_public_longdecode_hf.sh)"
    slurm_out_rendered="${slurm_out/\%j/${jobid}}"
  fi
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${label}" "${jobid}" "${out_dir}" "${slurm_out_rendered}" "${bench}" "${mode}" \
    "${offset}" "${count}" "${max_new}" "${min_new}" "${force_max}" | tee -a "${MANIFEST}"
}

submit_sharded() {
  local bench="$1"
  local total="$2"
  local shard="$3"
  local max_new="$4"
  local min_new="$5"
  local force_max="$6"
  local evaluate_code="$7"
  local code_eval_timeout="$8"
  local start_offset="${9:-0}"
  local temperature="${10:-${TEMPERATURE}}"
  local top_p="${11:-${TOP_P}}"
  local top_k="${12:-${TOP_K}}"
  local disable_thinking="${13:-${DISABLE_THINKING}}"

  if [ "${total}" -le 0 ] || [ "${shard}" -le 0 ]; then
    return
  fi
  local offset="${start_offset}"
  local end_offset=$((start_offset + total))
  while [ "${offset}" -lt "${end_offset}" ]; do
    local remaining=$((end_offset - offset))
    local count="${shard}"
    if [ "${remaining}" -lt "${count}" ]; then
      count="${remaining}"
    fi
    for mode in "${MODES[@]}"; do
      submit_one "${bench}" "${mode}" "${offset}" "${count}" "${max_new}" "${min_new}" "${force_max}" "${evaluate_code}" "${code_eval_timeout}" \
        "${temperature}" "${top_p}" "${top_k}" "${disable_thinking}"
    done
    offset=$((offset + count))
  done
}

if [ "${INCLUDE_AIME}" = "1" ]; then
  submit_sharded "aime24" "${AIME_TOTAL_EXAMPLES}" "${AIME_SHARD_SIZE}" "${AIME_MAX_NEW_TOKENS}" 0 0 0 6 0
fi
if [ "${INCLUDE_GPQA}" = "1" ]; then
  submit_sharded "gpqa" "${GPQA_TOTAL_EXAMPLES}" "${GPQA_SHARD_SIZE}" "${GPQA_MAX_NEW_TOKENS}" 0 0 0 6 0
fi
if [ "${INCLUDE_LIVE_CODE}" = "1" ]; then
  submit_sharded \
    "livecodebench_codegen" \
    "${LIVE_CODE_TOTAL_EXAMPLES}" \
    "${LIVE_CODE_SHARD_SIZE}" \
    "${LIVE_CODE_MAX_NEW_TOKENS}" \
    "${LIVE_CODE_MIN_NEW_TOKENS}" \
    "${LIVE_CODE_FORCE_MAX_NEW_TOKENS}" \
    "${LIVE_CODE_EVALUATE_CODE}" \
    "${LIVE_CODE_CODE_EVAL_TIMEOUT}" \
    "${LIVE_CODE_START_OFFSET}"
fi
if [ "${INCLUDE_LONGGEN_SGT_SHORT}" = "1" ]; then
  submit_sharded "longgenbench_sgt_short" "${LONGGEN_SGT_SHORT_TOTAL_EXAMPLES}" "${LONGGEN_SGT_SHORT_SHARD_SIZE}" "${LONGGEN_SGT_SHORT_MAX_NEW_TOKENS}" "${LONGGEN_SGT_SHORT_MIN_NEW_TOKENS}" 1 0 6 0 \
    0.0 1.0 0 "${PRESET_DISABLE_THINKING:-0}"
fi
if [ "${INCLUDE_LONGGEN_SGT_LONG}" = "1" ]; then
  submit_sharded "longgenbench_sgt_long" "${LONGGEN_SGT_LONG_TOTAL_EXAMPLES}" "${LONGGEN_SGT_LONG_SHARD_SIZE}" "${LONGGEN_SGT_LONG_MAX_NEW_TOKENS}" "${LONGGEN_SGT_LONG_MIN_NEW_TOKENS}" 1 0 6 0 \
    0.0 1.0 0 "${PRESET_DISABLE_THINKING:-0}"
fi
if [ "${INCLUDE_LONGGEN_GSM8K}" = "1" ]; then
  submit_sharded "longgenbench_gsm8k" "${LONGGEN_GSM8K_TOTAL_EXAMPLES}" "${LONGGEN_GSM8K_SHARD_SIZE}" "${LONGGEN_GSM8K_MAX_NEW_TOKENS}" "${LONGGEN_GSM8K_MIN_NEW_TOKENS}" 1 0 6 0 \
    0.0 1.0 0 "${PRESET_DISABLE_THINKING:-0}"
fi

chmod +x "${PLAN}"
echo "[INFO] SUBMIT=${SUBMIT}"
echo "[INFO] Manifest: ${MANIFEST}"
echo "[INFO] Submit plan: ${PLAN}"
if [ "${SUBMIT}" != "1" ] && [ "${SUBMIT}" != "true" ] && [ "${SUBMIT}" != "yes" ]; then
  echo "[INFO] Dry run only. Re-run with SUBMIT=1 to launch jobs."
fi
