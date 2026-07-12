#!/usr/bin/env bash
#SBATCH --job-name=acc-campaign
#SBATCH --account=zhengya0
#SBATCH --partition=gpu_mig40,spgpu,gpu-rtx6000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120000m
#SBATCH --time=04:00:00
#SBATCH --output=logs/acceptance_campaign_%j.log
set -euo pipefail

# Large-N acceptance campaign (issue #21 close / final acceptance claims).
# CONTRACT operating point: op-fix lineage, fixed parser (proxy_mass_m0p9
# parses 0.9). Per task: dense free-run (trace producer, OP-independent)
# then THREE teacher-forced arms consuming that trace: off @ tau=0.004
# (contract baseline), off @ tau=0.05 (relaxed; swap to the knee value for
# the 128k wave if the knee lands first), start1. All four passes for a
# task run inside one job on one card, so per-task TF comparisons are
# card-consistent by construction.
#
# Incremental per task: a pass whose pred file already exists is skipped,
# so a killed/resubmitted job loses at most one task's partial work.
# TASKS and CTX come from --export; jobs are sized <=4h (about 2 tasks at
# 32k) and chained afterany within the running cap.
# Target: n=8 x 64 new tokens = 512 TF positions per arm per task;
# 13 tasks -> 6,656 positions per arm per ctx. Renewal-loop statistics
# (rounds, accepted/round, Clopper-Pearson CIs) are the primary metric,
# computed offline by logit_trace_postproc.py.

REPO=/gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention/worktrees/op-fix
cd "${REPO}"
: "${TASKS:?set TASKS (space-separated)}"
CTX="${CTX:-32k}"
if [ "${CTX}" = "32k" ]; then
  CONTEXT_LEN=32768; CTXTAG=32k
else
  CONTEXT_LEN=131072; CTXTAG=128k
fi
NUM_SAMPLES="${NUM_SAMPLES:-8}"
MAX_NEW_TOKENS=64
CAMPAIGN_ROOT="$(readlink -f benchmark_suite_result)/acceptance_campaign_20260712"
DATA_ROOT="${CAMPAIGN_ROOT}/data_${CTXTAG}"
OUTPUT_ROOT="${CAMPAIGN_ROOT}/runs_${CTXTAG}"
mkdir -p logs

echo "[acc-campaign] head=$(git rev-parse --short HEAD) ctx=${CTXTAG} tasks='${TASKS}' node=$(hostname)"
nvidia-smi -L

run_pass() {
  local task=$1 arm=$2 tau=$3 draft_mode=$4 tf_ref=$5
  local prefix="${task}_${CTXTAG}_n${NUM_SAMPLES}"
  local run_name="${prefix}_${arm}"
  [ -n "${tf_ref}" ] && run_name="${prefix}_${arm}_tf_${tf_ref}"
  if [ -s "${OUTPUT_ROOT}/${run_name}/pred/${task}.jsonl" ]; then
    echo "[acc-campaign] SKIP ${run_name} (exists)"
    return 0
  fi
  local forced_trace=""
  if [ -n "${tf_ref}" ]; then
    forced_trace="${OUTPUT_ROOT}/${prefix}_${tf_ref}/logit_trace.pt"
    [ -s "${forced_trace}" ] || { echo "missing ref trace ${forced_trace}" >&2; exit 1; }
  fi
  local data_file="${DATA_ROOT}/${task}/validation.jsonl"
  [ -s "${data_file}" ] || { echo "missing staged data ${data_file}" >&2; exit 1; }
  local arm_mode=pagedpq_batched
  [ "${arm}" = "dense" ] && arm_mode=dense_batched
  local t0=${SECONDS}
  echo "[acc-campaign] pass start task=${task} arm=${arm} tau=${tau} run=${run_name}"
  env \
    MODE="${arm_mode}" \
    GREEDY_LOGIT_TRACE_FILE="${OUTPUT_ROOT}/${run_name}/logit_trace.pt" \
    FORCED_TOKEN_TRACE_FILE="${forced_trace}" \
    TASK_NAME="${task}" \
    CONTEXT_LEN="${CONTEXT_LEN}" \
    NUM_SAMPLES="${NUM_SAMPLES}" \
    MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
    RUN_NAME="${run_name}" \
    JOINT_KV_STABILITY_THRESHOLD="${tau}" \
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
    SELECTOR_PQ_JOINT_DRAFT_MODE="${draft_mode}" \
    SKIP_RULER_EVAL=1 \
    bash scripts/run_frontier_ruler_batched_one.sh
  echo "ACC-CAMPAIGN-PASS ctx=${CTXTAG} task=${task} arm=${arm} run=${run_name} wall=$(( SECONDS - t0 ))s"
}

for task in ${TASKS}; do
  run_pass "${task}" dense      0.004 off    ""
  run_pass "${task}" off_t0p004 0.004 off    dense
  run_pass "${task}" off_t0p05  0.05  off    dense
  run_pass "${task}" start1     0.004 start1 dense
done

echo "ACC-CAMPAIGN: COMPLETED ctx=${CTXTAG} tasks='${TASKS}'"
