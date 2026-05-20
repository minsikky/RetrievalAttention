#!/usr/bin/env bash
set -euo pipefail

cd /gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-slurm_out/frontier_gpu_validation_${STAMP}}"
RESULT_ROOT="${RESULT_ROOT:-cuda_unit_result/frontier_gpu_validation_${STAMP}}"
MANIFEST="${MANIFEST:-${OUT_ROOT}/manifest.tsv}"

mkdir -p "${OUT_ROOT}" "${RESULT_ROOT}"

if ! timeout 5s scontrol ping >/dev/null 2>&1; then
  echo "Slurm controller is not reachable; no jobs submitted." >&2
  exit 2
fi

echo -e "name\tjob_id\toutput" >"${MANIFEST}"
LAST_JOB_ID=""

submit_job() {
  local name="$1"
  local output="$2"
  local dependency="$3"
  local script="$4"
  shift 4
  mkdir -p "$(dirname "${output}")"
  local sbatch_args=(--parsable --output="${output}")
  if [[ "${dependency}" != "-" ]]; then
    sbatch_args+=(--dependency="${dependency}")
  fi
  local job_id
  job_id="$(env "$@" sbatch "${sbatch_args[@]}" "${script}")"
  LAST_JOB_ID="${job_id}"
  echo -e "${name}\t${job_id}\t${output}" | tee -a "${MANIFEST}"
}

submit_job \
  "cuda_unit" \
  "${OUT_ROOT}/unit-%j.out" \
  "-" \
  scripts/run_frontier_cuda_unit_tests.sh \
  OUTPUT_DIR="${RESULT_ROOT}/cuda_unit"
unit_dependency="afterok:${LAST_JOB_ID}"

for context in 32768 65536 131072; do
  submit_job \
    "exact_logit_${context}" \
    "${OUT_ROOT}/exact-logit-${context}-%j.out" \
    "${unit_dependency}" \
    scripts/run_exact_logit_backend_bench.sh \
    OUTPUT_DIR="${RESULT_ROOT}/exact_logit_${context}" CONTEXT="${context}" RANK="${context}" ITERS=10 WARMUP=3 BUILD_CUDA_EXT=0
done

submit_job \
  "ruler_niah_32768_profile" \
  "${OUT_ROOT}/ruler-niah-32768-%j.out" \
  "${unit_dependency}" \
  scripts/run_frontier_ruler_batched_one.sh \
  OUTPUT_ROOT="ruler_eval_result/frontier_gpu_validation_${STAMP}" RUN_NAME="frontier_niah_32768_profile" \
  TASK_NAME=niah_single_1 CONTEXT_LEN=32768 NUM_SAMPLES=1 MAX_NEW_TOKENS=128 PROFILE_NATIVE_OPS=1 \
  FRONTIER_EXACT_LOGIT_BACKEND=auto

submit_job \
  "longbench_v2_short_easy_profile" \
  "${OUT_ROOT}/lbv2-short-easy-%j.out" \
  "${unit_dependency}" \
  scripts/run_frontier_longbench_v2_one.sh \
  OUTPUT_DIR="longbench_v2_hf_result/frontier_gpu_validation_${STAMP}" MAX_EXAMPLES=4 LENGTH_FILTER=short \
  DIFFICULTY_FILTER=easy MAX_INPUT_TOKENS=8192 MAX_NEW_TOKENS=128 PROFILE_NATIVE_OPS=1 FRONTIER_EXACT_LOGIT_BACKEND=auto

echo "manifest=${MANIFEST}"
