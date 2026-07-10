#!/usr/bin/env bash
#SBATCH --job-name=qwen1m-kvoff-ab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180000m
#SBATCH --time=12:00:00
#SBATCH --account=zhengya0
#SBATCH --partition=gpu_mig40
#SBATCH --gpus-per-node=1
set -euo pipefail

WORKTREE="/gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention/worktrees/p2-kv-offload"
MAIN_REPO="/gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention"
QWEN_SNAPSHOT="${MAIN_REPO}/.hf_cache/hub/models--Qwen--Qwen2.5-7B-Instruct-1M/snapshots/e28526f7bb80e2a9c8af03b831a9af3812f18fba"
VENV_PY="${MAIN_REPO}/.venv/bin/python"
cd "${WORKTREE}"

export MODEL_NAME="${QWEN_SNAPSHOT}"
export CACHE_DIR="${MAIN_REPO}/.hf_cache"
export HF_VENV_DIR="${MAIN_REPO}/.venv"
export MODEL_TEMPLATE_TYPE="qwen-chat"
export MODE="dense_batched"
export TASK_NAME="niah_single_1"
export CONTEXT_LEN="${CONTEXT_LEN:-32768}"
export NUM_SAMPLES="2"
export PREFILL_CHUNK_SIZE="${PREFILL_CHUNK_SIZE:-4096}"
export DENSE_KV_BLOCK_TOKENS="${DENSE_KV_BLOCK_TOKENS:-8192}"
export DENSE_KV_STAGING_BUFFERS="${DENSE_KV_STAGING_BUFFERS:-2}"
export DENSE_KV_QUERY_BLOCK_TOKENS="${DENSE_KV_QUERY_BLOCK_TOKENS:-2048}"
export SELECTOR_PQ_JOINT_MEMORY_TRACE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OUTPUT_ROOT="benchmark_suite_result/qwen1m_dense_kv_offload_ab/runs"

LOG_ROOT="${WORKTREE}/benchmark_suite_result/qwen1m_dense_kv_offload_ab/logs/ctx${CONTEXT_LEN}"
mkdir -p "${LOG_ROOT}"

unset DENSE_KV_OFFLOAD DATA_FILE_OVERRIDE REUSE_DATA FORCED_TOKEN_TRACE_FILE
export RUN_NAME="stock_${TASK_NAME}_${CONTEXT_LEN}_n${NUM_SAMPLES}"
export GREEDY_LOGIT_TRACE_FILE="${LOG_ROOT}/stock_logits.pt"
bash scripts/run_ruler_pagedpq_stream_smoke_one.sh 2>&1 | tee "${LOG_ROOT}/stock.console.log"

STOCK_DATA="${WORKTREE}/${OUTPUT_ROOT}/${RUN_NAME}/data/${TASK_NAME}/validation.jsonl"
STOCK_SUMMARY="${WORKTREE}/${OUTPUT_ROOT}/${RUN_NAME}/summary/${TASK_NAME}.json"
export DENSE_KV_OFFLOAD=1
export DATA_FILE_OVERRIDE="${STOCK_DATA}"
export RUN_NAME="offload_${TASK_NAME}_${CONTEXT_LEN}_n${NUM_SAMPLES}"
export GREEDY_LOGIT_TRACE_FILE="${LOG_ROOT}/offload_logits.pt"
bash scripts/run_ruler_pagedpq_stream_smoke_one.sh 2>&1 | tee "${LOG_ROOT}/offload.console.log"
OFFLOAD_SUMMARY="${WORKTREE}/${OUTPUT_ROOT}/${RUN_NAME}/summary/${TASK_NAME}.json"

export RUN_NAME="offload_teacher_${TASK_NAME}_${CONTEXT_LEN}_n${NUM_SAMPLES}"
export GREEDY_LOGIT_TRACE_FILE="${LOG_ROOT}/teacher_logits.pt"
export FORCED_TOKEN_TRACE_FILE="${LOG_ROOT}/stock_logits.pt"
bash scripts/run_ruler_pagedpq_stream_smoke_one.sh 2>&1 | tee "${LOG_ROOT}/teacher.console.log"
TEACHER_SUMMARY="${WORKTREE}/${OUTPUT_ROOT}/${RUN_NAME}/summary/${TASK_NAME}.json"

module load python/3.10.4
export LD_LIBRARY_PATH="/sw/pkgs/arc/python/3.10.4/lib:${LD_LIBRARY_PATH:-}"
"${VENV_PY}" benchmark/ruler/pred/compare_dense_kv_offload_ab.py \
  --stock-trace "${LOG_ROOT}/stock_logits.pt" \
  --offload-trace "${LOG_ROOT}/offload_logits.pt" \
  --teacher-trace "${LOG_ROOT}/teacher_logits.pt" \
  --stock-summary "${STOCK_SUMMARY}" \
  --offload-summary "${OFFLOAD_SUMMARY}" \
  --teacher-summary "${TEACHER_SUMMARY}" \
  --stock-console "${LOG_ROOT}/stock.console.log" \
  --offload-console "${LOG_ROOT}/offload.console.log" \
  --teacher-console "${LOG_ROOT}/teacher.console.log" \
  --max-logit-diff "${DENSE_KV_AB_MAX_LOGIT_DIFF:-0.1}"
