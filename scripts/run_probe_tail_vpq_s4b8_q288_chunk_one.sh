#!/usr/bin/env bash
set -euo pipefail

cd /scratch/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention
export LD_LIBRARY_PATH="/sw/pkgs/arc/python/3.10.4/lib:${LD_LIBRARY_PATH:-}"

CHUNK_ID="${CHUNK_ID:?CHUNK_ID is required}"
CONFIG_NAME="${CONFIG_NAME:?CONFIG_NAME is required}"

source attention_efficiency_result/validation_deployable_20260512_173454/q288_chunks.env
chunk_var="chunk${CHUNK_ID}"
DECODE_LENGTHS="${!chunk_var}"
export DECODE_LENGTHS

case "${CONFIG_NAME}" in
  low_selmass090_b12288)
    export TAIL_BUDGET=12288
    export SELECTED_VALUE_EXACT_RULE=selected_mass
    export SELECTED_VALUE_EXACT_MASS=0.90
    export SELECTED_VALUE_EXACT_RISK_MASS=0.0
    ;;
  mid_risk090_b14336)
    export TAIL_BUDGET=14336
    export SELECTED_VALUE_EXACT_RULE=selected_risk_mass
    export SELECTED_VALUE_EXACT_MASS=0.0
    export SELECTED_VALUE_EXACT_RISK_MASS=0.90
    ;;
  high_selmass099_b14336)
    export TAIL_BUDGET=14336
    export SELECTED_VALUE_EXACT_RULE=selected_mass
    export SELECTED_VALUE_EXACT_MASS=0.99
    export SELECTED_VALUE_EXACT_RISK_MASS=0.0
    ;;
  *)
    echo "unknown CONFIG_NAME=${CONFIG_NAME}" >&2
    exit 2
    ;;
esac

export RUN_NAME="val_q288c${CHUNK_ID}_${CONFIG_NAME}${RUN_SUFFIX:-}_cpu_v2"
export QKV_TRACE="attention_efficiency_result/real_qkv_llama31_l16_6838_g131072_q288_window32_graphall_s16.npz"
export X_TRACE="attention_efficiency_result/real_xtrace_llama31_8b_layer16_8k_g131072_sampled_maskstop.npz"
export MAX_QIDX_PER_DECODE=1
export DEVICE=cpu
export SELECTED_VALUE_MODE=vpq_value
export VALUE_SUBVECS=1
export VALUE_SUBBITS=4

echo "[q288_chunk_one] chunk=${CHUNK_ID} config=${CONFIG_NAME}"
echo "[q288_chunk_one] decode_lengths=${DECODE_LENGTHS}"
bash scripts/run_probe_tail_vpq_s4b8_reuse_onlinecost_one.sh
