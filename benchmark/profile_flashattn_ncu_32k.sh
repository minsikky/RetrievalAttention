#!/bin/bash
#SBATCH --job-name=ncu_ra32k
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --account=zhengya98
#SBATCH --partition=spgpu
#SBATCH --gpus-per-node=1

set -euo pipefail

module purge
module load python/3.10.4
module unload pytorch 2>/dev/null || true
module load cuda/12.8.1

cd /gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention
source .venv/bin/activate

if ! command -v ncu >/dev/null 2>&1; then
  echo "[ERROR] ncu not found on PATH after loading cuda module."
  exit 2
fi

unset PYTHONPATH
unset PYTHONHOME
export PYTHONNOUSERSITE=1

MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.1-8B-Instruct}"
ATTN_TYPE="${ATTN_TYPE:-RetrievalAttention}"
DTYPE="${DTYPE:-bf16}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GEN_LEN="${GEN_LEN:-1}"
RECALL_INPUT_TOKENS="${RECALL_INPUT_TOKENS:-32768}"
TOKEN_BUDGET_OVERRIDE="${TOKEN_BUDGET_OVERRIDE:-100}"

NCU_SET="${NCU_SET:-full}"
NCU_LAUNCH_SKIP="${NCU_LAUNCH_SKIP:-1}"
NCU_LAUNCH_COUNT="${NCU_LAUNCH_COUNT:-1}"
NCU_KERNEL_REGEX="${NCU_KERNEL_REGEX:-flash_fwd_(kernel|splitkv_kernel|splitkv_combine_kernel)}"
NCU_OUTDIR="${NCU_OUTDIR:-benchmark/ncu_reports}"
NCU_EXTRA_ARGS="${NCU_EXTRA_ARGS:-}"

mkdir -p "${NCU_OUTDIR}"
REPORT_PREFIX="${NCU_OUTDIR}/ncu_ra32k_job${SLURM_JOB_ID:-manual}"

# Keep the profiled run focused on the fused forward kernel.
export LOW_CPU_MEM_USAGE="${LOW_CPU_MEM_USAGE:-1}"
export RETRIEVALATTN_FA_FUSED_PREFILL=1
export RETRIEVALATTN_FA_GRAPH_FUSED="${RETRIEVALATTN_FA_GRAPH_FUSED:-1}"
export RETRIEVALATTN_FA_GRAPH_FUSED_REQUIRE="${RETRIEVALATTN_FA_GRAPH_FUSED_REQUIRE:-1}"
export RETRIEVALATTN_FA_GRAPH_FUSED_CHECK="${RETRIEVALATTN_FA_GRAPH_FUSED_CHECK:-0}"
export RETRIEVALATTN_FA_KERNEL_MODE="${RETRIEVALATTN_FA_KERNEL_MODE:-v2_splitk}"
export RETRIEVALATTN_FA_SPLITK="${RETRIEVALATTN_FA_SPLITK:-auto}"
export RETRIEVALATTN_FA_MERGE_KERNEL="${RETRIEVALATTN_FA_MERGE_KERNEL:-1}"
export RETRIEVALATTN_FA_KERNEL_PROFILE=0
export RETRIEVALATTN_FA_GRAPH_PROFILE=0
export RETRIEVALATTN_SCORE_MODE="${RETRIEVALATTN_SCORE_MODE:-ip}"
export RETRIEVALATTN_VALIDATE_PARITY=0
export RETRIEVALATTN_TRAVERSAL_EVAL=0
export RETRIEVALATTN_DECODE_PROFILE=0
export RETRIEVALATTN_FUSED_PREFILL_OVERLAP=0
export RETRIEVALATTN_ROAR_LOG=0

echo "[INFO] host=$(hostname)"
echo "[INFO] python=$(which python)"
python -V
echo "[INFO] ncu=$(which ncu)"
ncu --version | head -n 1
echo "[INFO] report=${REPORT_PREFIX}.ncu-rep"
echo "[INFO] NCU_SET=${NCU_SET}"
echo "[INFO] NCU_KERNEL_REGEX=${NCU_KERNEL_REGEX}"
echo "[INFO] NCU_LAUNCH_SKIP=${NCU_LAUNCH_SKIP}"
echo "[INFO] NCU_LAUNCH_COUNT=${NCU_LAUNCH_COUNT}"
echo "[INFO] RETRIEVALATTN_FA_KERNEL_MODE=${RETRIEVALATTN_FA_KERNEL_MODE}"
echo "[INFO] RETRIEVALATTN_FA_SPLITK=${RETRIEVALATTN_FA_SPLITK}"
echo "[INFO] MODEL_NAME=${MODEL_NAME}"
echo "[INFO] RECALL_INPUT_TOKENS=${RECALL_INPUT_TOKENS}"

# If source correlation is missing in the report, rebuild flash-attn with:
#   FLASH_ATTN_LINEINFO=1 sbatch install_2.sh
ncu \
  --force-overwrite \
  --target-processes all \
  --set "${NCU_SET}" \
  --kernel-name-base demangled \
  --kernel-name "regex:${NCU_KERNEL_REGEX}" \
  --launch-skip "${NCU_LAUNCH_SKIP}" \
  --launch-count "${NCU_LAUNCH_COUNT}" \
  --import-source yes \
  --export "${REPORT_PREFIX}" \
  ${NCU_EXTRA_ARGS} \
  python -u simple_test.py \
    --model_name "${MODEL_NAME}" \
    --attn_type "${ATTN_TYPE}" \
    --dtype "${DTYPE}" \
    --batch_size "${BATCH_SIZE}" \
    --gen_len "${GEN_LEN}" \
    --token_budget_override "${TOKEN_BUDGET_OVERRIDE}" \
    --recall_only \
    --recall_input_tokens "${RECALL_INPUT_TOKENS}"

echo "[INFO] ncu report written to ${REPORT_PREFIX}.ncu-rep"
