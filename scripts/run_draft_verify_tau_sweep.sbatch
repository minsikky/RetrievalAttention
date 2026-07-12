#!/usr/bin/env bash
#SBATCH --job-name=dv-tau-sweep
#SBATCH --account=zhengya0
#SBATCH --partition=gpu_mig40,spgpu,gpu-rtx6000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120000m
#SBATCH --time=02:30:00
#SBATCH --output=logs/draft_verify_tau_sweep_%j.log
set -euo pipefail

# Issue #21 bundle A: aggressive-tau acceptance-vs-bytes sweep at 32k.
#
# Motivated by the dense-anchor result (off-vs-dense 0.953 free-run / 0.992
# teacher-forced at 32k): with a dense outer verify tier, the frozen sparse
# stack becomes a DRAFTER, so its selection knobs can relax. tau here is the
# escalation-walk stability threshold (JOINT_KV_STABILITY_THRESHOLD, consumed
# by run_frontier_ruler_batched_one.sh -> --joint_kv_stability_threshold).
# Larger tau lets the walk settle earlier -> smaller committed K/V budgets ->
# fewer committed bytes per token. Every arm is the PLAIN frozen algorithm
# (SELECTOR_PQ_JOINT_DRAFT_MODE=off) — pure env-config sweep, no code edits
# to selection logic; the walk itself is untouched.
#
# Arms (sequential, one card): dense reference (dumps its stream's logit
# trace), frozen at tau in {0.004 (= the matrix off baseline), 0.01, 0.02,
# 0.05} free-run, then a teacher-forced pass on the dense stream per tau arm.
# Every pass dumps its full per-position logit trace. Committed-bytes-per-
# token comes from the summary counters each arm already writes; combined
# with acceptance this yields the acceptance-vs-bytes curve for the drafter.

REPO=/gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention/worktrees/draft-gpu
cd "${REPO}"

OUTPUT_ROOT="${OUTPUT_ROOT:-benchmark_suite_result/draft_verify_gpu_20260712_matrix/tau_sweep_32k}"
TASK_NAME=qa_1
CONTEXT_LEN=32768
NUM_SAMPLES=4
MAX_NEW_TOKENS=0
DATA_FILE="${REPO}/benchmark_suite_result/ktcache_ab_20260709/qa1_32k_n4_ktON/data/qa_1/validation.jsonl"
RUN_PREFIX=qa1_32k_n4
mkdir -p logs

[ -s "${DATA_FILE}" ] || { echo "missing data ${DATA_FILE}" >&2; exit 1; }
echo "[dv-tau-sweep] head=$(git rev-parse --short HEAD) node=$(hostname)"
nvidia-smi -L

run_arm() {
  local arm_name=$1     # dense | tau0p004 | tau0p010 | ...
  local tau=$2          # JOINT_KV_STABILITY_THRESHOLD for frozen arms
  local tf_ref=$3       # "" for free-run, "dense" for teacher-forced
  local run_name="${RUN_PREFIX}_${arm_name}"
  local forced_trace=""
  if [ -n "${tf_ref}" ]; then
    run_name="${RUN_PREFIX}_${arm_name}_tf_${tf_ref}"
    forced_trace="${REPO}/${OUTPUT_ROOT}/${RUN_PREFIX}_${tf_ref}/logit_trace.pt"
    [ -s "${forced_trace}" ] || { echo "missing ref trace ${forced_trace}" >&2; exit 1; }
  fi
  local arm_mode=pagedpq_batched
  if [ "${arm_name}" = "dense" ]; then arm_mode=dense_batched; fi
  local t0=${SECONDS}
  echo "[dv-tau-sweep] arm start name=${arm_name} tau=${tau} run=${run_name} exec_mode=${arm_mode} tf_ref=${tf_ref:-none}"
  env \
    MODE="${arm_mode}" \
    GREEDY_LOGIT_TRACE_FILE="${REPO}/${OUTPUT_ROOT}/${run_name}/logit_trace.pt" \
    FORCED_TOKEN_TRACE_FILE="${forced_trace}" \
    TASK_NAME="${TASK_NAME}" \
    CONTEXT_LEN="${CONTEXT_LEN}" \
    NUM_SAMPLES="${NUM_SAMPLES}" \
    MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
    RUN_NAME="${run_name}" \
    JOINT_KV_STABILITY_THRESHOLD="${tau}" \
    LOGIT_BUFFER_FORMAT=e4m3 \
    JOINT_KV_PRECISION_TIERS=1 \
    FRONTIER_DENSE_KEY_T_CACHE=0 \
    OUTPUT_ROOT="${OUTPUT_ROOT}" \
    DATA_FILE_OVERRIDE="${DATA_FILE}" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PREFILL_CHUNK_SIZE=8192 \
    FRONTIER_EMPTY_CACHE_AFTER_PREFILL_CHUNK=1 \
    SELECTOR_PQ_JOINT_FROZENSIM_FUSED_VPREFIX=1 \
    SELECTOR_PQ_JOINT_FROZENSIM_FUSED_RISK_SORT=1 \
    SELECTOR_PQ_JOINT_FROZENSIM_FUSED_VPREFIX_TOKPAR=0 \
    SELECTOR_PQ_JOINT_MEMORY_TRACE=0 \
    SELECTOR_PQ_JOINT_TIME_TRACE=0 \
    SELECTOR_PQ_JOINT_DRAFT_MODE=off \
    SKIP_RULER_EVAL=1 \
    bash scripts/run_frontier_ruler_batched_one.sh
  local wall=$(( SECONDS - t0 ))
  echo "DV-TAU-ARM name=${arm_name} tau=${tau} run=${run_name} wall=${wall}s"
}

run_arm dense    0.004 ""
run_arm tau0p004 0.004 ""
run_arm tau0p010 0.010 ""
run_arm tau0p020 0.020 ""
run_arm tau0p050 0.050 ""
run_arm tau0p004 0.004 dense
run_arm tau0p010 0.010 dense
run_arm tau0p020 0.020 dense
run_arm tau0p050 0.050 dense

echo "DV-TAU-SWEEP: COMPLETED"
