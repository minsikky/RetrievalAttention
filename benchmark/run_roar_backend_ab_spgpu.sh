#!/bin/bash
#SBATCH --job-name=roar_backend_ab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000m
#SBATCH --time=180:00
#SBATCH --account=zhengya98
#SBATCH --partition=spgpu
#SBATCH --gpus-per-node=1

set -euo pipefail

module purge
module load python/3.10.4
module unload pytorch 2>/dev/null || true
module load cuda/12.8.1

if [ -f ".venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
else
  echo "[ERROR] .venv/bin/activate not found."
  exit 1
fi

export PYTHONNOUSERSITE=1
unset PYTHONPATH
unset PYTHONHOME

MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.1-8B-Instruct}"
DTYPE="${DTYPE:-bf16}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GEN_LEN="${GEN_LEN:-100}"
TOKEN_BUDGET_OVERRIDE="${TOKEN_BUDGET_OVERRIDE:-100}"
RECALL_INPUT_TOKENS="${RECALL_INPUT_TOKENS:-8192}"

RETRIEVALATTN_ROAR_NQ="${RETRIEVALATTN_ROAR_NQ:-32}"
RETRIEVALATTN_ROAR_M="${RETRIEVALATTN_ROAR_M:-32}"
RETRIEVALATTN_ROAR_L="${RETRIEVALATTN_ROAR_L:-20}"
RETRIEVALATTN_ROAR_ENHANCE_L="${RETRIEVALATTN_ROAR_ENHANCE_L:-16}"
RETRIEVALATTN_ROAR_MAX_QUERY_PER_PIVOT="${RETRIEVALATTN_ROAR_MAX_QUERY_PER_PIVOT:-0}"
RETRIEVALATTN_ROAR_CPP_THREADS="${RETRIEVALATTN_ROAR_CPP_THREADS:-${SLURM_CPUS_PER_TASK:-0}}"
RETRIEVALATTN_ROAR_PY_GPU_DEVICE="${RETRIEVALATTN_ROAR_PY_GPU_DEVICE:-cuda}"
RETRIEVALATTN_ROAR_PY_GPU_BATCH="${RETRIEVALATTN_ROAR_PY_GPU_BATCH:-256}"

COMMON_ENV=(
  LOW_CPU_MEM_USAGE=1
  RETRIEVALATTN_FA_FUSED_PREFILL=1
  RETRIEVALATTN_FUSED_PREFILL_OVERLAP=1
  RETRIEVALATTN_FUSED_PREFILL_OVERLAP_WORKERS=1
  RETRIEVALATTN_DECODE_INDEX=faiss
  RETRIEVALATTN_SEED_MODE=graph_only
  RETRIEVALATTN_QUERY_MODE=per_head
  RETRIEVALATTN_SCORE_MODE=ip
  RETRIEVALATTN_ROAR_NQ="${RETRIEVALATTN_ROAR_NQ}"
  RETRIEVALATTN_ROAR_M="${RETRIEVALATTN_ROAR_M}"
  RETRIEVALATTN_ROAR_L="${RETRIEVALATTN_ROAR_L}"
  RETRIEVALATTN_ROAR_ENABLE_ENHANCE=1
  RETRIEVALATTN_ROAR_ENHANCE_L="${RETRIEVALATTN_ROAR_ENHANCE_L}"
  RETRIEVALATTN_ROAR_ENTRY=hub
  RETRIEVALATTN_ROAR_MAX_QUERY_PER_PIVOT="${RETRIEVALATTN_ROAR_MAX_QUERY_PER_PIVOT}"
  RETRIEVALATTN_ROAR_CPP_THREADS="${RETRIEVALATTN_ROAR_CPP_THREADS}"
  RETRIEVALATTN_ROAR_PY_GPU_DEVICE="${RETRIEVALATTN_ROAR_PY_GPU_DEVICE}"
  RETRIEVALATTN_ROAR_PY_GPU_BATCH="${RETRIEVALATTN_ROAR_PY_GPU_BATCH}"
  RETRIEVALATTN_ROAR_LOG=1
  RETRIEVALATTN_VALIDATE_PARITY=0
  RETRIEVALATTN_TRAVERSAL_EVAL=0
  RETRIEVALATTN_DECODE_PROFILE=1
)

BASE_CMD=(
  python -u simple_test.py
  --model_name "${MODEL_NAME}"
  --attn_type RetrievalAttention
  --dtype "${DTYPE}"
  --batch_size "${BATCH_SIZE}"
  --gen_len "${GEN_LEN}"
  --token_budget_override "${TOKEN_BUDGET_OVERRIDE}"
  --recall_only
  --recall_input_tokens "${RECALL_INPUT_TOKENS}"
)

run_case() {
  local label="$1"
  local backend="$2"
  echo "[AB] ===== ${label} (backend=${backend}) ====="
  /usr/bin/time -v env "${COMMON_ENV[@]}" \
    RETRIEVALATTN_ROAR_BACKEND="${backend}" \
    "${BASE_CMD[@]}"
}

AB_MODE="${AB_MODE:-both}" # both | cpp_only | gpu_only

echo "[INFO] Host: $(hostname)"
echo "[INFO] Python: $(which python)"
echo "[INFO] AB_MODE=${AB_MODE}"
python -V
python - <<'PY'
import torch
print('[INFO] torch:', torch.__version__)
print('[INFO] cuda_available:', torch.cuda.is_available())
print('[INFO] device_name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')
PY

if [ "${AB_MODE}" = "both" ] || [ "${AB_MODE}" = "cpp_only" ]; then
  run_case "cpp" "cpp"
fi
if [ "${AB_MODE}" = "both" ] || [ "${AB_MODE}" = "gpu_only" ]; then
  run_case "python_gpu" "python_gpu"
fi

echo "[AB] completed"
