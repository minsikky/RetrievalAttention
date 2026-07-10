#!/usr/bin/env bash
#SBATCH --job-name=fspq-tokpar-rate
#SBATCH --account=zhengya0
#SBATCH --partition=gpu_mig40,spgpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120000m
#SBATCH --time=02:00:00
#SBATCH --output=logs/fspq_tokpar_rate_%j.log
set -euo pipefail

REPO=/gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention
cd "${REPO}"

DATA_FILE="${REPO}/benchmark_suite_result/frozen_sim_20260707/runs/frozensim_mk3_128k_n16/data/niah_multikey_3/validation.jsonl"
OUTPUT_ROOT="${REPO}/benchmark_suite_result/fspq_tokpar_round3"
[ -s "${DATA_FILE}" ] || { echo "missing data ${DATA_FILE}" >&2; exit 1; }

echo "[tokpar-rate] main=$(git rev-parse --short HEAD) gpu=$(nvidia-smi -L | head -1)"

run_arm() {
  local label=$1 tokpar=$2
  local run_name="mk3_128k_n1_8tok_${label}_${SLURM_JOB_ID}"
  local t0=${SECONDS}
  env \
    TASK_NAME=niah_multikey_3 \
    CONTEXT_LEN=131072 \
    NUM_SAMPLES=1 \
    MAX_NEW_TOKENS=8 \
    RUN_NAME="${run_name}" \
    JOINT_KV_STABILITY_THRESHOLD=0.004 \
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
    SELECTOR_PQ_JOINT_FROZENSIM_FUSED_VPREFIX_TOKPAR="${tokpar}" \
    SELECTOR_PQ_JOINT_MEMORY_TRACE=0 \
    SELECTOR_PQ_JOINT_TIME_TRACE=1 \
    SKIP_RULER_EVAL=1 \
    bash scripts/run_frontier_ruler_batched_one.sh
  local wall=$(( SECONDS - t0 ))
  echo "TOKPAR-RATE-ARM label=${label} tokpar=${tokpar} run=${run_name} wall=${wall}s"
}

run_arm fused 0
run_arm fused_tokpar 1

echo "[tokpar-rate] selector attention totals and wall clocks"
grep -E "selector-pq-joint-time|TOKPAR-RATE-ARM" \
  "${REPO}/logs/fspq_tokpar_rate_${SLURM_JOB_ID}.log" | tail -n 20 || true
echo "TOKPAR-RATE: COMPLETED"
