#!/usr/bin/env bash
#SBATCH --job-name=dv-wide-acc
#SBATCH --account=zhengya0
#SBATCH --partition=gpu_mig40,spgpu,gpu-rtx6000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120000m
#SBATCH --time=03:30:00
#SBATCH --output=logs/draft_verify_wide_accept_%j.log
set -euo pipefail

# Issue #21 bundle B: acceptance generalization beyond qa_1 at 32k.
#
# Widens the off-vs-dense 0.953/0.992 headline (and the start1 drafter
# numbers) across task families: qa_2 (hotpotqa multi-hop), vt (variable
# tracking), fwe (frequent-words extraction), n=4 each, 64 new tokens.
# Arms per task, sequential on one card: dense reference (dumps trace), off
# and start1 free-run, then teacher-forced passes for off and start1 on the
# dense stream. Every pass dumps its full per-position logit trace.
#
# Data is PRE-STAGED under ${OUTPUT_ROOT}/data_staging/{task}/validation.jsonl
# (generated once with data/prepare.py, seed 42) and pinned via
# DATA_FILE_OVERRIDE for every arm, so all arms of a task see identical
# inputs and no data prep runs inside the GPU allocation.

REPO=/gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention/worktrees/draft-gpu
cd "${REPO}"

OUTPUT_ROOT="${OUTPUT_ROOT:-benchmark_suite_result/draft_verify_gpu_20260712_matrix/wide_accept_32k}"
CONTEXT_LEN=32768
NUM_SAMPLES=4
MAX_NEW_TOKENS=64
TASKS="${TASKS:-qa_2 vt fwe}"
mkdir -p logs

echo "[dv-wide-acc] head=$(git rev-parse --short HEAD) node=$(hostname)"
nvidia-smi -L

run_arm() {
  local task=$1
  local prefix=$2
  local draft_mode=$3   # dense | off | start1
  local tf_ref=$4       # "" or "dense"
  local data_file="${REPO}/${OUTPUT_ROOT}/data_staging/${task}/validation.jsonl"
  [ -s "${data_file}" ] || { echo "missing staged data ${data_file}" >&2; exit 1; }
  local run_name="${prefix}_${draft_mode}"
  local forced_trace=""
  if [ -n "${tf_ref}" ]; then
    run_name="${prefix}_${draft_mode}_tf_${tf_ref}"
    forced_trace="${REPO}/${OUTPUT_ROOT}/${prefix}_${tf_ref}/logit_trace.pt"
    [ -s "${forced_trace}" ] || { echo "missing ref trace ${forced_trace}" >&2; exit 1; }
  fi
  local arm_mode=pagedpq_batched
  local arm_draft="${draft_mode}"
  if [ "${draft_mode}" = "dense" ]; then
    arm_mode=dense_batched
    arm_draft=off
  fi
  local t0=${SECONDS}
  echo "[dv-wide-acc] arm start task=${task} mode=${draft_mode} run=${run_name} exec_mode=${arm_mode} tf_ref=${tf_ref:-none}"
  env \
    MODE="${arm_mode}" \
    GREEDY_LOGIT_TRACE_FILE="${REPO}/${OUTPUT_ROOT}/${run_name}/logit_trace.pt" \
    FORCED_TOKEN_TRACE_FILE="${forced_trace}" \
    TASK_NAME="${task}" \
    CONTEXT_LEN="${CONTEXT_LEN}" \
    NUM_SAMPLES="${NUM_SAMPLES}" \
    MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
    RUN_NAME="${run_name}" \
    JOINT_KV_STABILITY_THRESHOLD=0.004 \
    LOGIT_BUFFER_FORMAT=e4m3 \
    JOINT_KV_PRECISION_TIERS=1 \
    FRONTIER_DENSE_KEY_T_CACHE=0 \
    OUTPUT_ROOT="${OUTPUT_ROOT}" \
    DATA_FILE_OVERRIDE="${data_file}" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PREFILL_CHUNK_SIZE=8192 \
    FRONTIER_EMPTY_CACHE_AFTER_PREFILL_CHUNK=1 \
    SELECTOR_PQ_JOINT_FROZENSIM_FUSED_VPREFIX=1 \
    SELECTOR_PQ_JOINT_FROZENSIM_FUSED_RISK_SORT=1 \
    SELECTOR_PQ_JOINT_FROZENSIM_FUSED_VPREFIX_TOKPAR=0 \
    SELECTOR_PQ_JOINT_MEMORY_TRACE=0 \
    SELECTOR_PQ_JOINT_TIME_TRACE=0 \
    SELECTOR_PQ_JOINT_DRAFT_MODE="${arm_draft}" \
    SKIP_RULER_EVAL=1 \
    bash scripts/run_frontier_ruler_batched_one.sh
  local wall=$(( SECONDS - t0 ))
  echo "DV-WIDE-ARM task=${task} mode=${draft_mode} run=${run_name} wall=${wall}s"
}

for task in ${TASKS}; do
  prefix="$(echo "${task}" | tr -d '_')_32k_n4"
  run_arm "${task}" "${prefix}" dense  ""
  run_arm "${task}" "${prefix}" off    ""
  run_arm "${task}" "${prefix}" start1 ""
  run_arm "${task}" "${prefix}" off    dense
  run_arm "${task}" "${prefix}" start1 dense
done

echo "DV-WIDE-ACCEPT: COMPLETED"
