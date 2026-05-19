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
case "${HF_MODEL_PRESET}" in
  ""|qwen3_8b)
    DEFAULT_MODEL_NAME=".hf_cache/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
    ;;
  llama31_8b|llama3_1_8b)
    DEFAULT_MODEL_NAME=".hf_cache/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
    ;;
  qwen3_5_9b)
    DEFAULT_MODEL_NAME=".hf_cache/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a"
    ;;
  *)
    echo "[ERROR] Unknown HF_MODEL_PRESET=${HF_MODEL_PRESET}"
    echo "[ERROR] Supported presets: qwen3_8b, llama31_8b, llama3_1_8b, qwen3_5_9b"
    exit 2
    ;;
esac
MODEL_NAME="${MODEL_NAME:-${DEFAULT_MODEL_NAME}}"
MAX_EXAMPLES="${MAX_EXAMPLES:-1}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
LIVE_CODE_EVALUATE_CODE="${LIVE_CODE_EVALUATE_CODE:-1}"
LIVE_CODE_MAX_EXAMPLES="${LIVE_CODE_MAX_EXAMPLES:-${MAX_EXAMPLES}}"
LIVE_CODE_CODE_EVAL_TIMEOUT="${LIVE_CODE_CODE_EVAL_TIMEOUT:-6}"

# Smoke defaults are intentionally small. Increase MAX_EXAMPLES and MAX_NEW_TOKENS
# for validation/full runs after the wrapper is confirmed on this model.
declare -a BENCHMARKS=(
  "aime24:2048:0:0"
  "livecodebench_codegen:2048:0:0"
  "longgenbench_sgt_short:16384:1:8192"
)

declare -a MODES=("dense" "pagedpq")

for item in "${BENCHMARKS[@]}"; do
  IFS=: read -r bench max_new force_max min_new <<< "${item}"
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
    export_args=(
      "ALL"
      "HF_MODEL_PRESET=${HF_MODEL_PRESET}"
      "BENCHMARK=${bench}"
      "ATTENTION_MODE=${mode}"
      "MODEL_NAME=${MODEL_NAME}"
      "OUTPUT_DIR=${out_dir}"
      "RUN_NAME=${label}"
      "OUTPUT_ROOT=${OUTPUT_ROOT}"
      "MAX_EXAMPLES=${bench_max_examples}"
      "MAX_NEW_TOKENS=${max_new}"
      "MIN_NEW_TOKENS=${min_new}"
      "FORCE_MAX_NEW_TOKENS=${force_max}"
      "LOCAL_FILES_ONLY=${LOCAL_FILES_ONLY}"
      "EVALUATE_CODE=${evaluate_code}"
      "CODE_EVAL_TIMEOUT=${code_eval_timeout}"
    )
    jobid="$(sbatch --parsable --output="${slurm_out}" --export="$(IFS=,; echo "${export_args[*]}")" benchmark/run_public_longdecode_hf.sh)"
    printf "%s\t%s\t%s\t%s\n" "${label}" "${jobid}" "${out_dir}" "${slurm_out/\%j/${jobid}}" | tee -a "${MANIFEST}"
  done
done

echo "[INFO] Manifest: ${MANIFEST}"
