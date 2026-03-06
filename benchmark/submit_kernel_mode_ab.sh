#!/bin/bash
set -euo pipefail

# Submit a controlled RetrievalAttention kernel-mode A/B matrix:
#   - legacy
#   - v2_local
#   - v2_splitk
#
# Optional first argument (or BUILD_JOB env) sets dependency on a build job:
#   ./benchmark/submit_kernel_mode_ab.sh 44298749
#   BUILD_JOB=44298749 ./benchmark/submit_kernel_mode_ab.sh

BUILD_JOB="${BUILD_JOB:-${1:-}}"

ACCOUNT="${ACCOUNT:-zhengya98}"
PARTITION="${PARTITION:-spgpu}"
TIME_LIMIT="${TIME_LIMIT:-90:00}"
CPUS="${CPUS:-4}"
MEM_MB="${MEM_MB:-48000}"
GPUS="${GPUS:-1}"

TEST_MODE="${TEST_MODE:-simple}"
ATTN_TYPE="${ATTN_TYPE:-RetrievalAttention}"
GEN_LEN="${GEN_LEN:-1}"
TOKEN_BUDGET_OVERRIDE="${TOKEN_BUDGET_OVERRIDE:-100}"

MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.1-8B-Instruct}"
DTYPE="${DTYPE:-bf16}"
BATCH_SIZE="${BATCH_SIZE:-1}"

RETRIEVALATTN_FA_GRAPH_FUSED="${RETRIEVALATTN_FA_GRAPH_FUSED:-1}"
RETRIEVALATTN_FA_GRAPH_FUSED_REQUIRE="${RETRIEVALATTN_FA_GRAPH_FUSED_REQUIRE:-1}"
RETRIEVALATTN_FA_GRAPH_PROFILE="${RETRIEVALATTN_FA_GRAPH_PROFILE:-1}"
RETRIEVALATTN_FA_SPLITK="${RETRIEVALATTN_FA_SPLITK:-auto}"
RETRIEVALATTN_FA_KERNEL_PROFILE="${RETRIEVALATTN_FA_KERNEL_PROFILE:-1}"
RETRIEVALATTN_FA_KERNEL_DEBUG="${RETRIEVALATTN_FA_KERNEL_DEBUG:-0}"

# Optional short debug run after the three main runs.
SUBMIT_DEBUG_SHORT="${SUBMIT_DEBUG_SHORT:-0}"
DEBUG_RECALL_INPUT_TOKENS="${DEBUG_RECALL_INPUT_TOKENS:-2048}"

dependency_opts=()
if [ -n "${BUILD_JOB}" ]; then
  dependency_opts=(--dependency="afterok:${BUILD_JOB}")
fi

common_env=(
  "TEST_MODE=${TEST_MODE}"
  "ATTN_TYPE=${ATTN_TYPE}"
  "MODEL_NAME=${MODEL_NAME}"
  "DTYPE=${DTYPE}"
  "BATCH_SIZE=${BATCH_SIZE}"
  "GEN_LEN=${GEN_LEN}"
  "TOKEN_BUDGET_OVERRIDE=${TOKEN_BUDGET_OVERRIDE}"
  "LOW_CPU_MEM_USAGE=1"
  "RETRIEVALATTN_FA_GRAPH_FUSED=${RETRIEVALATTN_FA_GRAPH_FUSED}"
  "RETRIEVALATTN_FA_GRAPH_FUSED_REQUIRE=${RETRIEVALATTN_FA_GRAPH_FUSED_REQUIRE}"
  "RETRIEVALATTN_FA_GRAPH_PROFILE=${RETRIEVALATTN_FA_GRAPH_PROFILE}"
  "RETRIEVALATTN_FA_KERNEL_PROFILE=${RETRIEVALATTN_FA_KERNEL_PROFILE}"
)

submit_case() {
  local label="$1"
  local mode="$2"
  local splitk="$3"
  local debug="$4"
  local recall_only="$5"
  local recall_tokens="$6"

  local env_items=("${common_env[@]}")
  env_items+=(
    "RETRIEVALATTN_FA_KERNEL_MODE=${mode}"
    "RETRIEVALATTN_FA_SPLITK=${splitk}"
    "RETRIEVALATTN_FA_KERNEL_DEBUG=${debug}"
    "RECALL_ONLY=${recall_only}"
    "RECALL_INPUT_TOKENS=${recall_tokens}"
  )

  local export_str="ALL"
  local x
  for x in "${env_items[@]}"; do
    export_str="${export_str},${x}"
  done

  local job_id
  job_id="$(
    sbatch --parsable \
      --account="${ACCOUNT}" \
      --partition="${PARTITION}" \
      --time="${TIME_LIMIT}" \
      --cpus-per-task="${CPUS}" \
      --mem="${MEM_MB}" \
      --gpus-per-node="${GPUS}" \
      --job-name="ra-kab-${label}" \
      --output="slurm-ra-kab-${label}-%j.out" \
      "${dependency_opts[@]}" \
      --export="${export_str}" \
      test.sh
  )"
  echo "[KAB] submitted ${label}: job_id=${job_id}"
}

echo "[KAB] account=${ACCOUNT} partition=${PARTITION} cpus=${CPUS} mem_mb=${MEM_MB} gpus=${GPUS} time=${TIME_LIMIT}"
if [ -n "${BUILD_JOB}" ]; then
  echo "[KAB] dependency=afterok:${BUILD_JOB}"
else
  echo "[KAB] dependency=none"
fi

submit_case "legacy" "legacy" "0" "0" "0" "8192"
submit_case "v2local" "v2_local" "0" "0" "0" "8192"
submit_case "v2splitk" "v2_splitk" "${RETRIEVALATTN_FA_SPLITK}" "0" "0" "8192"

if [ "${SUBMIT_DEBUG_SHORT}" = "1" ]; then
  submit_case "v2splitkdbg" "v2_splitk" "${RETRIEVALATTN_FA_SPLITK}" "${RETRIEVALATTN_FA_KERNEL_DEBUG}" "1" "${DEBUG_RECALL_INPUT_TOKENS}"
fi

