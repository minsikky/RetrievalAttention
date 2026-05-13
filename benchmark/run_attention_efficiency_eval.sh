#!/bin/bash
# Slurm wrapper for the offline sparse-attention algorithmic-efficiency proxy.
#
# This is not a latency benchmark. It estimates algorithmic efficiency by
# comparing selected token sets against dense attention mass/output references.

#SBATCH --job-name=attn-eff-proxy
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

module load python/3.10.4 >/dev/null 2>&1 || true

export LD_LIBRARY_PATH="/sw/pkgs/arc/python/3.10.4/lib:${LD_LIBRARY_PATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER}-${SLURM_JOB_ID:-manual}}"
mkdir -p "${MPLCONFIGDIR}"

OUTPUT_DIR="${OUTPUT_DIR:-attention_efficiency_result/proxy_v1}"
SOURCE_NPZ="${SOURCE_NPZ:-}"
CONTEXT_LENGTHS="${CONTEXT_LENGTHS:-16384,32768,65536}"
DECODE_LENGTHS="${DECODE_LENGTHS:-0}"
BUDGETS="${BUDGETS:-64,128,256,512,1024,2048}"
BUDGET_POLICY="${BUDGET_POLICY:-fixed}"
BUDGET_RATIO="${BUDGET_RATIO:-0.10}"
METHODS="${METHODS:-dense_oracle,static_chunk,retroinfer_style,retrievalattention_style}"
NUM_QUERIES="${NUM_QUERIES:-128}"
NUM_HEADS="${NUM_HEADS:-32}"
NUM_KV_HEADS="${NUM_KV_HEADS:-8}"
HEAD_DIM="${HEAD_DIM:-128}"
STATIC_PREFIX="${STATIC_PREFIX:-128}"
STATIC_SUFFIX="${STATIC_SUFFIX:-512}"
CHUNK_SIZE="${CHUNK_SIZE:-128}"
RETRO_CLUSTER_SIZE="${RETRO_CLUSTER_SIZE:-128}"
RETRO_CLUSTER_SCOPE="${RETRO_CLUSTER_SCOPE:-prefill}"
RETRO_TARGET_METHOD="${RETRO_TARGET_METHOD:-retroinfer_style}"
GRAPH_DEGREE="${GRAPH_DEGREE:-16}"
RA_GRAPH_BACKEND="${RA_GRAPH_BACKEND:-auto}"
RA_PRECOMPUTE_CHUNK="${RA_PRECOMPUTE_CHUNK:-512}"
RA_SEED_COUNT="${RA_SEED_COUNT:-32}"
RA_VISIT_BUDGET="${RA_VISIT_BUDGET:-2048}"
ADAPTIVE_CHECK_INTERVAL="${ADAPTIVE_CHECK_INTERVAL:-16}"
RA_MASS_TARGETS="${RA_MASS_TARGETS:-0.1,0.2,0.4,0.6}"
RA_COS_TARGETS="${RA_COS_TARGETS:-0.2,0.4,0.6,0.8}"
BUDGET_MODE="${BUDGET_MODE:-dynamic}"
SCORE_SCALE="${SCORE_SCALE:-0}"
SCORE_KEY_BYTES_PER_ELEMENT="${SCORE_KEY_BYTES_PER_ELEMENT:-4}"
ATTN_KEY_BYTES_PER_ELEMENT="${ATTN_KEY_BYTES_PER_ELEMENT:-2}"
VALUE_BYTES_PER_ELEMENT="${VALUE_BYTES_PER_ELEMENT:-2}"
EDGE_INDEX_BYTES="${EDGE_INDEX_BYTES:-4}"
GRAPH_OFFSET_BYTES="${GRAPH_OFFSET_BYTES:-4}"
INCLUDE_RERANK_COST="${INCLUDE_RERANK_COST:-1}"
SEED="${SEED:-2025}"
DEVICE="${DEVICE:-cuda}"
PLOT="${PLOT:-1}"

args=(
  benchmark/attention_efficiency_eval.py
  --output_dir "${OUTPUT_DIR}"
  --context_lengths "${CONTEXT_LENGTHS}"
  --decode_lengths "${DECODE_LENGTHS}"
  --budgets "${BUDGETS}"
  --budget_policy "${BUDGET_POLICY}"
  --budget_ratio "${BUDGET_RATIO}"
  --budget_mode "${BUDGET_MODE}"
  --methods "${METHODS}"
  --num_queries "${NUM_QUERIES}"
  --num_heads "${NUM_HEADS}"
  --num_kv_heads "${NUM_KV_HEADS}"
  --head_dim "${HEAD_DIM}"
  --static_prefix "${STATIC_PREFIX}"
  --static_suffix "${STATIC_SUFFIX}"
  --chunk_size "${CHUNK_SIZE}"
  --retro_cluster_size "${RETRO_CLUSTER_SIZE}"
  --retro_cluster_scope "${RETRO_CLUSTER_SCOPE}"
  --retro_target_method "${RETRO_TARGET_METHOD}"
  --graph_degree "${GRAPH_DEGREE}"
  --ra_graph_backend "${RA_GRAPH_BACKEND}"
  --ra_precompute_chunk "${RA_PRECOMPUTE_CHUNK}"
  --ra_seed_count "${RA_SEED_COUNT}"
  --ra_visit_budget "${RA_VISIT_BUDGET}"
  --adaptive_check_interval "${ADAPTIVE_CHECK_INTERVAL}"
  --ra_mass_targets "${RA_MASS_TARGETS}"
  --ra_cos_targets "${RA_COS_TARGETS}"
  --score_scale "${SCORE_SCALE}"
  --score_key_bytes_per_element "${SCORE_KEY_BYTES_PER_ELEMENT}"
  --attn_key_bytes_per_element "${ATTN_KEY_BYTES_PER_ELEMENT}"
  --value_bytes_per_element "${VALUE_BYTES_PER_ELEMENT}"
  --edge_index_bytes "${EDGE_INDEX_BYTES}"
  --graph_offset_bytes "${GRAPH_OFFSET_BYTES}"
  --seed "${SEED}"
  --device "${DEVICE}"
)

if [[ "${INCLUDE_RERANK_COST}" == "0" ]]; then
  args+=(--no-include_rerank_cost)
else
  args+=(--include_rerank_cost)
fi

if [[ -n "${SOURCE_NPZ}" ]]; then
  args+=(--source_npz "${SOURCE_NPZ}")
fi

if [[ "${PLOT}" == "1" ]]; then
  args+=(--plot)
fi

echo "[run_attention_efficiency_eval] output=${OUTPUT_DIR}"
echo "[run_attention_efficiency_eval] context_lengths=${CONTEXT_LENGTHS} budgets=${BUDGETS}"
echo "[run_attention_efficiency_eval] decode_lengths=${DECODE_LENGTHS}"
echo "[run_attention_efficiency_eval] budget_policy=${BUDGET_POLICY} budget_ratio=${BUDGET_RATIO}"
echo "[run_attention_efficiency_eval] budget_mode=${BUDGET_MODE} device=${DEVICE}"
echo "[run_attention_efficiency_eval] retro_cluster_scope=${RETRO_CLUSTER_SCOPE} retro_target_method=${RETRO_TARGET_METHOD}"
echo "[run_attention_efficiency_eval] ra_graph_backend=${RA_GRAPH_BACKEND} ra_precompute_chunk=${RA_PRECOMPUTE_CHUNK}"
echo "[run_attention_efficiency_eval] ra_mass_targets=${RA_MASS_TARGETS} ra_cos_targets=${RA_COS_TARGETS}"
echo "[run_attention_efficiency_eval] byte_model score_key=${SCORE_KEY_BYTES_PER_ELEMENT} attn_key=${ATTN_KEY_BYTES_PER_ELEMENT} value=${VALUE_BYTES_PER_ELEMENT} edge=${EDGE_INDEX_BYTES} offset=${GRAPH_OFFSET_BYTES} include_rerank=${INCLUDE_RERANK_COST}"

.venv/bin/python "${args[@]}"
