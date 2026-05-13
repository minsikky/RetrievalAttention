#!/bin/bash
# Dump one real-model QKV decode trace for offline attention-efficiency sweeps.

#SBATCH --job-name=qkv-trace
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

module load python/3.10.4 >/dev/null 2>&1 || true

export LD_LIBRARY_PATH="/sw/pkgs/arc/python/3.10.4/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.1-8B-Instruct}"
DATA_PATH="${DATA_PATH:-attention_efficiency_result/gpu_smoke_8192.json}"
OUTPUT_NPZ="${OUTPUT_NPZ:-attention_efficiency_result/real_qkv_trace_llama31_8b_layer16_8k_g128.npz}"
LAYER_IDX="${LAYER_IDX:-16}"
GEN_LEN="${GEN_LEN:-128}"
MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-8192}"
ATTENTION_TYPE="${ATTENTION_TYPE:-Full_Flash_Attn}"
DTYPE="${DTYPE:-bf16}"
DEVICE="${DEVICE:-cuda:0}"
PROMPT_INDEX="${PROMPT_INDEX:-0}"
DO_SAMPLE="${DO_SAMPLE:-0}"
TEMPERATURE="${TEMPERATURE:-0.8}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-50}"
INCLUDE_PREFILL_QUERIES="${INCLUDE_PREFILL_QUERIES:-0}"
SAVE_LAYER_INPUTS="${SAVE_LAYER_INPUTS:-0}"
SKIP_QKV="${SKIP_QKV:-0}"
MASK_STOP_TOKENS="${MASK_STOP_TOKENS:-0}"

args=(
  scripts/dump_real_qkv_trace.py
  --model_name "${MODEL_NAME}"
  --data_path "${DATA_PATH}"
  --output_npz "${OUTPUT_NPZ}"
  --layer_idx "${LAYER_IDX}"
  --gen_len "${GEN_LEN}"
  --max_input_tokens "${MAX_INPUT_TOKENS}"
  --attention_type "${ATTENTION_TYPE}"
  --dtype "${DTYPE}"
  --device "${DEVICE}"
  --prompt_index "${PROMPT_INDEX}"
  --temperature "${TEMPERATURE}"
  --top_p "${TOP_P}"
  --top_k "${TOP_K}"
)

if [[ "${DO_SAMPLE}" == "1" ]]; then
  args+=(--do_sample)
fi
if [[ "${INCLUDE_PREFILL_QUERIES}" == "1" ]]; then
  args+=(--include_prefill_queries)
fi
if [[ "${SAVE_LAYER_INPUTS}" == "1" ]]; then
  args+=(--save_layer_inputs)
fi
if [[ "${SKIP_QKV}" == "1" ]]; then
  args+=(--skip_qkv)
fi
if [[ "${MASK_STOP_TOKENS}" == "1" ]]; then
  args+=(--mask_stop_tokens)
fi

echo "[run_dump_real_qkv_trace] model=${MODEL_NAME}"
echo "[run_dump_real_qkv_trace] data=${DATA_PATH}"
echo "[run_dump_real_qkv_trace] output=${OUTPUT_NPZ}"
echo "[run_dump_real_qkv_trace] attention_type=${ATTENTION_TYPE}"
echo "[run_dump_real_qkv_trace] layer=${LAYER_IDX} input_cap=${MAX_INPUT_TOKENS} gen_len=${GEN_LEN}"
echo "[run_dump_real_qkv_trace] do_sample=${DO_SAMPLE} temperature=${TEMPERATURE} top_p=${TOP_P} top_k=${TOP_K}"
echo "[run_dump_real_qkv_trace] save_layer_inputs=${SAVE_LAYER_INPUTS} skip_qkv=${SKIP_QKV}"
echo "[run_dump_real_qkv_trace] mask_stop_tokens=${MASK_STOP_TOKENS}"

.venv/bin/python "${args[@]}"
