#!/usr/bin/env bash
#SBATCH --job-name=draft-verify
#SBATCH --account=zhengya0
#SBATCH --partition=gpu_mig40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60000m
#SBATCH --time=02:00:00
#SBATCH --output=logs/draft_verify_%j.log
set -euo pipefail

# End-to-end draft-then-verify token-acceptance arm. Cribs the tokpar identity
# gate env (scripts/run_fspq_tokpar_identity_gate.sbatch) minus the comparison
# epilogue, and adds SELECTOR_PQ_JOINT_DRAFT_MODE={off,start1,start2}. Off is
# byte-identical to the frozen algorithm; start1/start2 skip the escalation walk
# and pin K to (proxy-mass start rung + 1 / + 2). Pinned to gpu_mig40 so token
# streams are comparable against the mig40-class frozen reference.
#
# Required env (pass via sbatch --export):
#   DRAFT_MODE            off | start1 | start2
#   TASK_NAME             e.g. qa_1 | niah_multikey_3
#   CONTEXT_LEN           e.g. 32768 | 131072
#   NUM_SAMPLES           e.g. 4 | 2
#   DATA_FILE_OVERRIDE    absolute path to validation.jsonl
#   RUN_NAME              unique run dir name
#   OUTPUT_ROOT           results root
# Optional:
#   MAX_NEW_TOKENS        default 0 (task default); 128k uses 64

REPO=/gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention/worktrees/draft-budget
cd "${REPO}"

: "${DRAFT_MODE:?set DRAFT_MODE}"
: "${TASK_NAME:?set TASK_NAME}"
: "${CONTEXT_LEN:?set CONTEXT_LEN}"
: "${NUM_SAMPLES:?set NUM_SAMPLES}"
: "${DATA_FILE_OVERRIDE:?set DATA_FILE_OVERRIDE}"
: "${RUN_NAME:?set RUN_NAME}"
: "${OUTPUT_ROOT:?set OUTPUT_ROOT}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-0}"

[ -s "${DATA_FILE_OVERRIDE}" ] || { echo "missing data ${DATA_FILE_OVERRIDE}" >&2; exit 1; }
mkdir -p logs

echo "[draft-verify] head=$(git rev-parse --short HEAD) run=${RUN_NAME} draft_mode=${DRAFT_MODE} task=${TASK_NAME} ctx=${CONTEXT_LEN} n=${NUM_SAMPLES} max_new=${MAX_NEW_TOKENS}"

env \
  TASK_NAME="${TASK_NAME}" \
  CONTEXT_LEN="${CONTEXT_LEN}" \
  NUM_SAMPLES="${NUM_SAMPLES}" \
  MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
  RUN_NAME="${RUN_NAME}" \
  JOINT_KV_STABILITY_THRESHOLD=0.004 \
  LOGIT_BUFFER_FORMAT=e4m3 \
  JOINT_KV_PRECISION_TIERS=1 \
  FRONTIER_DENSE_KEY_T_CACHE=0 \
  OUTPUT_ROOT="${OUTPUT_ROOT}" \
  DATA_FILE_OVERRIDE="${DATA_FILE_OVERRIDE}" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  PREFILL_CHUNK_SIZE=8192 \
  FRONTIER_EMPTY_CACHE_AFTER_PREFILL_CHUNK=1 \
  SELECTOR_PQ_JOINT_FROZENSIM_FUSED_VPREFIX=1 \
  SELECTOR_PQ_JOINT_FROZENSIM_FUSED_RISK_SORT=1 \
  SELECTOR_PQ_JOINT_FROZENSIM_FUSED_VPREFIX_TOKPAR=0 \
  SELECTOR_PQ_JOINT_MEMORY_TRACE=0 \
  SELECTOR_PQ_JOINT_TIME_TRACE=0 \
  SELECTOR_PQ_JOINT_DRAFT_MODE="${DRAFT_MODE}" \
  SKIP_RULER_EVAL=1 \
  bash scripts/run_frontier_ruler_batched_one.sh

echo "DRAFT-VERIFY-RUN=${RUN_NAME} DRAFT_MODE=${DRAFT_MODE}"
