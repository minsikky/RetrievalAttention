#!/usr/bin/env bash
set -euo pipefail

cd /gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-public_longdecode_result/${STAMP}}"
SLURM_ROOT="${SLURM_ROOT:-slurm_out/public_longdecode_${STAMP}}"
MANIFEST="${MANIFEST:-notes/slurm_manifests/public_longdecode_${STAMP}.tsv}"

mkdir -p "${OUTPUT_ROOT}" "${SLURM_ROOT}" "$(dirname "${MANIFEST}")"
printf "label\tjobid\toutput_dir\tslurm_out\n" > "${MANIFEST}"

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
TEMPERATURE="${TEMPERATURE:-${PUBLIC_TEMPERATURE:-${official_temperature}}}"
TOP_P="${TOP_P:-${PUBLIC_TOP_P:-${official_top_p}}}"
TOP_K="${TOP_K:-${PUBLIC_TOP_K:-${official_top_k}}}"
MAX_EXAMPLES="${MAX_EXAMPLES:-1}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
LIVE_CODE_EVALUATE_CODE="${LIVE_CODE_EVALUATE_CODE:-1}"
LIVE_CODE_MAX_EXAMPLES="${LIVE_CODE_MAX_EXAMPLES:-${MAX_EXAMPLES}}"
LIVE_CODE_CODE_EVAL_TIMEOUT="${LIVE_CODE_CODE_EVAL_TIMEOUT:-6}"

# Smoke defaults use Qwen3 official-style generation caps for task-quality
# benchmarks, but keep LongGenBench forced/deterministic because it is a
# long-decode stress benchmark rather than a Qwen3 report benchmark.
declare -a BENCHMARKS=(
  "aime24:${AIME_MAX_NEW_TOKENS:-${QWEN3_AIME_MAX_NEW_TOKENS}}:0:0:official"
  "livecodebench_codegen:${LIVE_CODE_MAX_NEW_TOKENS:-${QWEN3_DEFAULT_MAX_NEW_TOKENS}}:0:0:official"
  "longgenbench_sgt_short:16384:1:8192"
)

declare -a MODES=("dense" "pagedpq")

for item in "${BENCHMARKS[@]}"; do
  IFS=: read -r bench max_new force_max min_new generation_policy <<< "${item}"
  for mode in "${MODES[@]}"; do
    label="${mode}_${bench}"
    out_dir="${OUTPUT_ROOT}/${label}"
    slurm_out="${SLURM_ROOT}/${label}-%j.out"
    bench_max_examples="${MAX_EXAMPLES}"
    evaluate_code="0"
    code_eval_timeout="${CODE_EVAL_TIMEOUT:-6}"
    if [ "${bench}" = "livecodebench_codegen" ]; then
      bench_max_examples="${LIVE_CODE_MAX_EXAMPLES}"
      evaluate_code="${LIVE_CODE_EVALUATE_CODE}"
      code_eval_timeout="${LIVE_CODE_CODE_EVAL_TIMEOUT}"
    fi
    task_temperature="${TEMPERATURE}"
    task_top_p="${TOP_P}"
    task_top_k="${TOP_K}"
    task_disable_thinking="${DISABLE_THINKING}"
    if [ "${generation_policy:-deterministic}" != "official" ]; then
      task_temperature=0.0
      task_top_p=1.0
      task_top_k=0
      task_disable_thinking="${PRESET_DISABLE_THINKING:-0}"
    fi
    export_args=(
      "ALL"
      "HF_MODEL_PRESET=${HF_MODEL_PRESET}"
      "BENCHMARK=${bench}"
      "ATTENTION_MODE=${mode}"
      "MODEL_NAME=${MODEL_NAME}"
      "HF_LANGUAGE_MODEL_ONLY=${HF_LANGUAGE_MODEL_ONLY}"
      "USE_CHAT_TEMPLATE=${USE_CHAT_TEMPLATE}"
      "DISABLE_THINKING=${task_disable_thinking}"
      "OUTPUT_DIR=${out_dir}"
      "RUN_NAME=${label}"
      "OUTPUT_ROOT=${OUTPUT_ROOT}"
      "MAX_EXAMPLES=${bench_max_examples}"
      "MAX_NEW_TOKENS=${max_new}"
      "MIN_NEW_TOKENS=${min_new}"
      "FORCE_MAX_NEW_TOKENS=${force_max}"
      "TEMPERATURE=${task_temperature}"
      "TOP_P=${task_top_p}"
      "TOP_K=${task_top_k}"
      "LOCAL_FILES_ONLY=${LOCAL_FILES_ONLY}"
      "EVALUATE_CODE=${evaluate_code}"
      "CODE_EVAL_TIMEOUT=${code_eval_timeout}"
    )
    jobid="$(sbatch --parsable --output="${slurm_out}" --export="$(IFS=,; echo "${export_args[*]}")" benchmark/run_public_longdecode_hf.sh)"
    printf "%s\t%s\t%s\t%s\n" "${label}" "${jobid}" "${out_dir}" "${slurm_out/\%j/${jobid}}" | tee -a "${MANIFEST}"
  done
done

echo "[INFO] Manifest: ${MANIFEST}"
