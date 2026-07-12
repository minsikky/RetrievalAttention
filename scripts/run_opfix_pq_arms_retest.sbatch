#!/usr/bin/env bash
#SBATCH --job-name=opfix-pqarms
#SBATCH --account=zhengya0
#SBATCH --partition=spgpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120000m
#SBATCH --time=03:00:00
#SBATCH --output=logs/opfix_pqarms_retest_%j.log
set -euo pipefail

# P2 addendum (HELD until the implementation audit reports): contract-OP
# reruns of the PQ-approximation draft arms. middle and v_exact consumed the
# buggy 0.5-mass start rungs (middle: exact-K set from start_ki+1; v_exact:
# v_target from the start_ki+1 rung), so their floor-lineage acceptance
# numbers are OP-floor; pq_only touches no start rung and reruns here only
# as an unaffected control that makes the rerun matrix self-consistent.
# Same dense-reference reuse + spgpu pinning as the gate retest.

REPO=/gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention/worktrees/op-fix
DGPU=/gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention/worktrees/draft-gpu
MATRIX=${DGPU}/benchmark_suite_result/draft_verify_gpu_20260712_matrix
OUTPUT_ROOT="${OUTPUT_ROOT:-${DGPU}/benchmark_suite_result/draft_verify_gpu_20260712_opfix}"
cd "${REPO}"
mkdir -p logs

echo "[opfix-pqarms] head=$(git rev-parse --short HEAD) node=$(hostname)"
nvidia-smi -L

run_arm() {
  local ctxtag=$1     # 32k | 128k
  local draft_mode=$2 # off | start1
  local tf_ref=$3     # "" | dense
  local task prefix ctx nsamp data
  if [ "${ctxtag}" = "32k" ]; then
    task=qa_1; prefix=qa1_32k_n4; ctx=32768; nsamp=4
    data=${DGPU}/benchmark_suite_result/ktcache_ab_20260709/qa1_32k_n4_ktON/data/qa_1/validation.jsonl
  else
    task=niah_multikey_3; prefix=mk3_128k_n2; ctx=131072; nsamp=2
    data=${DGPU}/benchmark_suite_result/frozen_sim_20260707/runs/frozensim_mk3_128k_n16/data/niah_multikey_3/validation.jsonl
  fi
  # Match each context's matrix max_new so the reused dense trace satisfies
  # the fixed-length forced-token contract (32k task default = 32; 128k 64).
  local max_new=0
  [ "${ctxtag}" = "128k" ] && max_new=64
  local run_name="${prefix}_${draft_mode}"
  local forced_trace=""
  if [ -n "${tf_ref}" ]; then
    run_name="${prefix}_${draft_mode}_tf_${tf_ref}"
    forced_trace="${MATRIX}/${prefix}_dense/logit_trace.pt"
    [ -s "${forced_trace}" ] || { echo "missing matrix dense trace ${forced_trace}" >&2; exit 1; }
  fi
  local t0=${SECONDS}
  echo "[opfix-pqarms] arm start ctx=${ctxtag} mode=${draft_mode} run=${run_name} tf_ref=${tf_ref:-none}"
  env \
    MODE=pagedpq_batched \
    GREEDY_LOGIT_TRACE_FILE="${OUTPUT_ROOT}/${run_name}/logit_trace.pt" \
    FORCED_TOKEN_TRACE_FILE="${forced_trace}" \
    TASK_NAME="${task}" \
    CONTEXT_LEN="${ctx}" \
    NUM_SAMPLES="${nsamp}" \
    MAX_NEW_TOKENS="${max_new}" \
    RUN_NAME="${run_name}" \
    JOINT_KV_STABILITY_THRESHOLD=0.004 \
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
    SELECTOR_PQ_JOINT_DRAFT_MODE="${draft_mode}" \
    SKIP_RULER_EVAL=1 \
    bash scripts/run_frontier_ruler_batched_one.sh
  local wall=$(( SECONDS - t0 ))
  echo "OPFIX-PQARMS-ARM ctx=${ctxtag} mode=${draft_mode} run=${run_name} wall=${wall}s"
}

for ctx in 32k 128k; do
  for arm in pq_only middle v_exact; do
    run_arm "${ctx}" "${arm}" ""
    run_arm "${ctx}" "${arm}" dense
  done
done

echo "OPFIX-PQARMS-RETEST: COMPLETED"
