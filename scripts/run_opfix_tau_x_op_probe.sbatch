#!/usr/bin/env bash
#SBATCH --job-name=opfix-tauxop
#SBATCH --account=zhengya0
#SBATCH --partition=spgpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120000m
#SBATCH --time=05:00:00
#SBATCH --output=logs/opfix_tau_x_op_%j.log
set -euo pipefail

# tau x OP interaction probe (STAGED; submit on coordinator order, after
# P3). P3 sweeps mass/bump/v_ratio at fixed tau=0.004; Bundle A swept tau
# at the floor OP. This probe fills the cross-term: frozen (off, walk ON)
# arms at the CONTRACT operating point (mass 0.9 via the fixed parser on
# this lineage) with tau in {0.010, 0.050}, both contexts, free-run +
# teacher-forced on the matrix dense references (spgpu-pinned for card
# class). P1 showed the walk still sets the budget at OP-0.9 (starts land
# at/below the settled point), so tau should still gate bytes there — this
# measures by how much.

REPO=/gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention/worktrees/op-fix
DGPU=/gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention/worktrees/draft-gpu
MATRIX=${DGPU}/benchmark_suite_result/draft_verify_gpu_20260712_matrix
OUTPUT_ROOT="${OUTPUT_ROOT:-${DGPU}/benchmark_suite_result/draft_verify_gpu_20260712_opfix/tau_x_op}"
cd "${REPO}"
mkdir -p logs

echo "[opfix-tauxop] head=$(git rev-parse --short HEAD) node=$(hostname)"
nvidia-smi -L

run_arm() {
  local ctxtag=$1 tau=$2 tf_ref=$3
  local task prefix ctx nsamp data max_new
  if [ "${ctxtag}" = "32k" ]; then
    task=qa_1; prefix=qa1_32k_n4; ctx=32768; nsamp=4; max_new=0
    data=${DGPU}/benchmark_suite_result/ktcache_ab_20260709/qa1_32k_n4_ktON/data/qa_1/validation.jsonl
  else
    task=niah_multikey_3; prefix=mk3_128k_n2; ctx=131072; nsamp=2; max_new=64
    data=${DGPU}/benchmark_suite_result/frozen_sim_20260707/runs/frozensim_mk3_128k_n16/data/niah_multikey_3/validation.jsonl
  fi
  local tag="tau${tau//./p}"
  local run_name="${prefix}_${tag}"
  local forced_trace=""
  if [ -n "${tf_ref}" ]; then
    run_name="${prefix}_${tag}_tf_${tf_ref}"
    forced_trace="${MATRIX}/${prefix}_dense/logit_trace.pt"
    [ -s "${forced_trace}" ] || { echo "missing dense trace" >&2; exit 1; }
  fi
  local t0=${SECONDS}
  echo "[opfix-tauxop] arm start ctx=${ctxtag} tau=${tau} run=${run_name} tf_ref=${tf_ref:-none}"
  env \
    MODE=pagedpq_batched \
    GREEDY_LOGIT_TRACE_FILE="${OUTPUT_ROOT}/${run_name}/logit_trace.pt" \
    FORCED_TOKEN_TRACE_FILE="${forced_trace}" \
    TASK_NAME="${task}" \
    CONTEXT_LEN="${ctx}" \
    NUM_SAMPLES="${nsamp}" \
    MAX_NEW_TOKENS="${max_new}" \
    RUN_NAME="${run_name}" \
    JOINT_KV_STABILITY_THRESHOLD="${tau}" \
    LOGIT_BUFFER_FORMAT=e4m3 \
    JOINT_KV_PRECISION_TIERS=1 \
    FRONTIER_DENSE_KEY_T_CACHE=0 \
    OUTPUT_ROOT="${OUTPUT_ROOT}" \
    DATA_FILE_OVERRIDE="${data}" \
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
  echo "OPFIX-TAUXOP-ARM ctx=${ctxtag} tau=${tau} run=${run_name} wall=${wall}s"
}

run_arm 32k  0.010 ""
run_arm 32k  0.010 dense
run_arm 32k  0.050 ""
run_arm 32k  0.050 dense
run_arm 128k 0.010 ""
run_arm 128k 0.010 dense
run_arm 128k 0.050 ""
run_arm 128k 0.050 dense

echo "OPFIX-TAUXOP: COMPLETED"
