#!/usr/bin/env bash
#SBATCH --job-name=draft-verify-b
#SBATCH --account=zhengya0
#SBATCH --partition=gpu_mig40,spgpu,gpu-rtx6000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120000m
#SBATCH --time=06:00:00
#SBATCH --output=logs/draft_verify_bundle_%j.log
set -euo pipefail

# Draft-then-verify token-acceptance BUNDLE: all arms of one context length run
# sequentially inside a single allocation, so every arm shares the exact same
# GPU by construction (card-class consistency binds only WITHIN a comparison).
# Pattern cribbed from scripts/run_fspq_tokpar_rate_probe.sbatch's run_arm loop.
#
# Submit with BUNDLE=32k or BUNDLE=128k (sbatch --export). Partition is set on
# the submit line: 32k = gpu_mig40,spgpu,gpu-rtx6000; 128k = spgpu,gpu-rtx6000
# (mig40 excluded: mk3@128k dense-prefill SDPA OOMs the 39.25GB slice — suite
# jobs 53253685/53253684 died there even with expandable_segments + chunk 8192).

REPO=/gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention/worktrees/draft-gpu
cd "${REPO}"

: "${BUNDLE:?set BUNDLE=32k or 128k}"
# Round 2 (issue #21): PQ-approximation draft arms (pq_only, middle) vs the
# frozen off reference. New output dir so the 20260711 start1/start2 runs are
# not overwritten.
OUTPUT_ROOT="${OUTPUT_ROOT:-benchmark_suite_result/draft_verify_gpu_20260712}"
mkdir -p logs

echo "[draft-verify-bundle] head=$(git rev-parse --short HEAD) bundle=${BUNDLE} node=$(hostname)"
nvidia-smi -L

case "${BUNDLE}" in
  32k)
    TASK_NAME=qa_1
    CONTEXT_LEN=32768
    NUM_SAMPLES=4
    MAX_NEW_TOKENS=0
    DATA_FILE="${REPO}/benchmark_suite_result/ktcache_ab_20260709/qa1_32k_n4_ktON/data/qa_1/validation.jsonl"
    RUN_PREFIX=qa1_32k_n4
    ARMS="${ARMS:-off pq_only middle}"
    ;;
  128k)
    TASK_NAME=niah_multikey_3
    CONTEXT_LEN=131072
    NUM_SAMPLES=2
    MAX_NEW_TOKENS=64
    DATA_FILE="${REPO}/benchmark_suite_result/frozen_sim_20260707/runs/frozensim_mk3_128k_n16/data/niah_multikey_3/validation.jsonl"
    RUN_PREFIX=mk3_128k_n2
    ARMS="${ARMS:-off pq_only middle}"
    ;;
  *)
    echo "unknown BUNDLE=${BUNDLE}; expected 32k or 128k" >&2
    exit 2
    ;;
esac

[ -s "${DATA_FILE}" ] || { echo "missing data ${DATA_FILE}" >&2; exit 1; }

run_arm() {
  local draft_mode=$1
  local run_name="${RUN_PREFIX}_${draft_mode}"
  local t0=${SECONDS}
  echo "[draft-verify-bundle] arm start mode=${draft_mode} run=${run_name}"
  env \
    TASK_NAME="${TASK_NAME}" \
    CONTEXT_LEN="${CONTEXT_LEN}" \
    NUM_SAMPLES="${NUM_SAMPLES}" \
    MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
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
    SELECTOR_PQ_JOINT_FROZENSIM_FUSED_VPREFIX_TOKPAR=0 \
    SELECTOR_PQ_JOINT_MEMORY_TRACE=0 \
    SELECTOR_PQ_JOINT_TIME_TRACE=0 \
    SELECTOR_PQ_JOINT_DRAFT_MODE="${draft_mode}" \
    SKIP_RULER_EVAL=1 \
    bash scripts/run_frontier_ruler_batched_one.sh
  local wall=$(( SECONDS - t0 ))
  echo "DRAFT-VERIFY-ARM bundle=${BUNDLE} mode=${draft_mode} run=${run_name} wall=${wall}s"
}

for arm in ${ARMS}; do
  run_arm "${arm}"
done

echo "DRAFT-VERIFY-BUNDLE: COMPLETED bundle=${BUNDLE}"
