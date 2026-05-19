#!/usr/bin/env bash
set -euo pipefail

cd /scratch/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention
export LD_LIBRARY_PATH="/sw/pkgs/arc/python/3.10.4/lib:${LD_LIBRARY_PATH:-}"

CHUNK_ID="${CHUNK_ID:?CHUNK_ID is required}"
RUN_SUFFIX="${RUN_SUFFIX:?RUN_SUFFIX is required}"

source attention_efficiency_result/validation_deployable_20260512_173454/q288_chunks.env
chunk_var="chunk${CHUNK_ID}"
export DECODE_LENGTHS="${!chunk_var}"

export OUTPUT_ROOT="${OUTPUT_ROOT:-attention_efficiency_result/confidence_audit_rules_20260513}"
export RUN_NAME="q288c${CHUNK_ID}_${RUN_SUFFIX}"
export ONLINE_CONFIDENCE_RULE="${ONLINE_CONFIDENCE_RULE:-geometric_probe_tail_switch}"
export QKV_TRACE="${QKV_TRACE:-attention_efficiency_result/real_qkv_llama31_l16_6838_g131072_q288_window32_graphall_s16.npz}"
export X_TRACE="${X_TRACE:-attention_efficiency_result/real_xtrace_llama31_8b_layer16_8k_g131072_sampled_maskstop.npz}"
export MAX_QIDX_PER_DECODE=1
export DEVICE="${DEVICE:-cpu}"
export SELECTOR_MODE="${SELECTOR_MODE:-fullscan}"
export TAIL_PROBE_REL_L2_MAX="${TAIL_PROBE_REL_L2_MAX:-0.020}"
export GEOMETRIC_MIN_BUDGET="${GEOMETRIC_MIN_BUDGET:-8192}"
export GEOMETRIC_MAX_BUDGET="${GEOMETRIC_MAX_BUDGET:-65536}"
export GEOMETRIC_GROWTH="${GEOMETRIC_GROWTH:-1.5}"
export GEOMETRIC_PROBE_SCALE="${GEOMETRIC_PROBE_SCALE:-1.5}"
export GEOMETRIC_BUDGET_GRANULARITY="${GEOMETRIC_BUDGET_GRANULARITY:-1024}"
export TAIL_MODE="${TAIL_MODE:-vpq_value}"
export SELECTED_VALUE_MODE="${SELECTED_VALUE_MODE:-vpq_value}"
export SELECTED_VALUE_EXACT_RULE="${SELECTED_VALUE_EXACT_RULE:-selected_mass}"
export SELECTED_VALUE_EXACT_MASS="${SELECTED_VALUE_EXACT_MASS:-0.99}"
export SUBVECS="${SUBVECS:-4}"
export SUBBITS="${SUBBITS:-8}"
export VALUE_SUBVECS="${VALUE_SUBVECS:-1}"
export VALUE_SUBBITS="${VALUE_SUBBITS:-4}"
export AUDIT_TAIL_MODE="${AUDIT_TAIL_MODE:-rank_prefix}"
export AUDIT_TAIL_SAMPLES="${AUDIT_TAIL_SAMPLES:-2048}"
export AUDIT_TAIL_MASS_MAX="${AUDIT_TAIL_MASS_MAX:-0.10}"
export AUDIT_TAIL_LOGIT_GAP_MAX="${AUDIT_TAIL_LOGIT_GAP_MAX:-inf}"

echo "[confidence_audit_q288_chunk] chunk=${CHUNK_ID} suffix=${RUN_SUFFIX}"
echo "[confidence_audit_q288_chunk] decode_lengths=${DECODE_LENGTHS}"
bash scripts/run_confidence_budget_rule_one.sh
