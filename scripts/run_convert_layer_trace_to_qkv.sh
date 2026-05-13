#!/bin/bash
# Convert a saved layer-input trace into the compact QKV NPZ used by offline sweeps.

#SBATCH --job-name=qkv-convert
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
module load python/3.10.4 >/dev/null 2>&1 || true

export LD_LIBRARY_PATH="/sw/pkgs/arc/python/3.10.4/lib:${LD_LIBRARY_PATH:-}"

INPUT_NPZ="${INPUT_NPZ:?INPUT_NPZ is required}"
OUTPUT_NPZ="${OUTPUT_NPZ:?OUTPUT_NPZ is required}"
DECODE_LENGTHS="${DECODE_LENGTHS:-500,1000,2000,4000,8000,16000,32000}"
REPEAT_POSITIONS="${REPEAT_POSITIONS:-4}"
QUERY_POSITION_MODE="${QUERY_POSITION_MODE:-repeat}"
DEVICE="${DEVICE:-cuda}"
CHUNK_TOKENS="${CHUNK_TOKENS:-512}"
DTYPE="${DTYPE:-fp16}"
GRAPH_QUERY_SCOPE="${GRAPH_QUERY_SCOPE:-all}"
GRAPH_QUERY_STRIDE="${GRAPH_QUERY_STRIDE:-16}"

echo "[run_convert_layer_trace_to_qkv] input=${INPUT_NPZ}"
echo "[run_convert_layer_trace_to_qkv] output=${OUTPUT_NPZ}"
echo "[run_convert_layer_trace_to_qkv] decode_lengths=${DECODE_LENGTHS}"
echo "[run_convert_layer_trace_to_qkv] repeat_positions=${REPEAT_POSITIONS}"
echo "[run_convert_layer_trace_to_qkv] query_position_mode=${QUERY_POSITION_MODE}"
echo "[run_convert_layer_trace_to_qkv] graph_scope=${GRAPH_QUERY_SCOPE} stride=${GRAPH_QUERY_STRIDE}"

.venv/bin/python scripts/convert_layer_trace_to_qkv_npz.py \
  --input_npz "${INPUT_NPZ}" \
  --output_npz "${OUTPUT_NPZ}" \
  --decode_lengths "${DECODE_LENGTHS}" \
  --repeat_positions "${REPEAT_POSITIONS}" \
  --query_position_mode "${QUERY_POSITION_MODE}" \
  --device "${DEVICE}" \
  --chunk_tokens "${CHUNK_TOKENS}" \
  --dtype "${DTYPE}" \
  --include_graph_prefill_queries \
  --graph_query_scope "${GRAPH_QUERY_SCOPE}" \
  --graph_query_stride "${GRAPH_QUERY_STRIDE}"
