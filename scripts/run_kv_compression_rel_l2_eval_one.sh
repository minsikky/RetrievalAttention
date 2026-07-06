#!/usr/bin/env bash
#SBATCH --job-name=kv-comp-l2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96000m
#SBATCH --time=04:00:00
#SBATCH --account=zhengya98
#SBATCH --partition=standard
set -euo pipefail

cd /gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention

module purge
module load python/3.10.4
source .venv/bin/activate

export LD_LIBRARY_PATH="/sw/pkgs/arc/python/3.10.4/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

name="${RUN_NAME:?RUN_NAME is required}"
out_dir="${OUTPUT_ROOT:-attention_efficiency_result/kv_compression_rel_l2_20260522}/${name}"
mkdir -p "${out_dir}"

echo "[kv_compression_l2] host=$(hostname)"
echo "[kv_compression_l2] started=$(date --iso-8601=seconds)"
echo "[kv_compression_l2] out=${out_dir}"
echo "[kv_compression_l2] decode_lengths=${DECODE_LENGTHS:-500,1000,2000,4000,8000,16000,32000,64000,128000}"
echo "[kv_compression_l2] heads=${HEADS:-all}"
echo "[kv_compression_l2] methods=${METHODS:-default}"

.venv/bin/python benchmark/selector_eval/runners/run_kv_compression_rel_l2_eval.py \
  --qkv_trace "${QKV_TRACE:-attention_efficiency_result/real_qkv_llama31_l16_6838_g131072_q288_window32_graphall_s16.npz}" \
  --x_trace "${X_TRACE:-attention_efficiency_result/real_xtrace_llama31_8b_layer16_8k_g131072_sampled_maskstop.npz}" \
  --output_dir "${out_dir}" \
  --decode_lengths "${DECODE_LENGTHS:-500,1000,2000,4000,8000,16000,32000,64000,128000}" \
  --max_qidx_per_decode "${MAX_QIDX_PER_DECODE:-1}" \
  --heads "${HEADS:-}" \
  --methods "${METHODS:-dense,kivi_b2_g32_w128,kivi_b4_g32_w128,kivi_b2_g32_w2048,kivi_b4_g32_w2048,kvquant_like_b3_clip0p1_w128,kvquant_like_b4_clip0p1_w128,per_token_kv_b3_w128,per_token_kv_b4_w128,tq_k3v3_w128,tqprod_k3v3_w128,tqprod_k4v4_w128,pq_like_s4b4_w128,pq_like_s4b6_w128}" \
  --static_prefix "${STATIC_PREFIX:-128}" \
  --residual_window "${RESIDUAL_WINDOW:-128}" \
  --key_bytes "${KEY_BYTES:-2}" \
  --value_bytes "${VALUE_BYTES:-2}" \
  --metadata_bytes "${METADATA_BYTES:-2}" \
  --pq_iters "${PQ_ITERS:-3}" \
  --device cpu

echo "[kv_compression_l2] finished=$(date --iso-8601=seconds)"
cat "${out_dir}/summary.json"
