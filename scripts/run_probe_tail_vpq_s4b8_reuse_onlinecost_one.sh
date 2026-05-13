#!/usr/bin/env bash
set -euo pipefail

cd /scratch/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention
export LD_LIBRARY_PATH="/sw/pkgs/arc/python/3.10.4/lib:${LD_LIBRARY_PATH:-}"

QKV_TRACE="${QKV_TRACE:-/scratch/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention/attention_efficiency_result/real_qkv_llama31_l16_6838_g131072_q36_graphall_s16.npz}"
X_TRACE="${X_TRACE:-/scratch/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention/attention_efficiency_result/real_xtrace_llama31_8b_layer16_8k_g131072_sampled_maskstop.npz}"
BASE_OUT="/scratch/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention/attention_efficiency_result"
CONF_BUDGETS="${CONF_BUDGETS:-4096,8192,10240,12288,14336,16384,17408,18432,19456,20480,24576,28672,32768}"
DECODE_LENGTHS="${DECODE_LENGTHS:-500,1000,2000,4000,8000,16000,32000,64000,128000}"

name="${RUN_NAME:?RUN_NAME is required}"
budget="${TAIL_BUDGET:?TAIL_BUDGET is required}"
threshold="${TAIL_THRESHOLD:-0.100}"
out_dir="${BASE_OUT}/${name}"
if [[ -f "${out_dir}/summary.json" ]]; then
  echo "[vpq_s4b8_reuse_onlinecost_one] skip existing ${name}"
  exit 0
fi

echo "[vpq_s4b8_reuse_onlinecost_one] run ${name}"
.venv/bin/python benchmark/selector_eval/runners/run_layer_quality_eval.py \
  --qkv_trace "${QKV_TRACE}" \
  --x_trace "${X_TRACE}" \
  --output_dir "${out_dir}" \
  --decode_lengths "${DECODE_LENGTHS}" \
  --max_qidx_per_decode "${MAX_QIDX_PER_DECODE:-1}" \
  --device "${DEVICE:-cpu}" \
  --selector_mode fullscan \
  --budgets "${BUDGETS:-16384}" \
  --online_confidence_rule probe_tail_switch \
  --tail_confidence_budget "${budget}" \
  --tail_probe_budget "${budget}" \
  --confidence_budgets "${CONF_BUDGETS}" \
  --tail_probe_rel_l2_max "${threshold}" \
  --tail_score_calibration affine_selected \
  --proxy_mass_target "${PROXY_MASS_TARGET:-0.99}" \
  --marginal_mass_max 0.010 \
  --marginal_score_gap_max -6.0 \
  --tail_mode "${TAIL_MODE:-vpq_value}" \
  --selected_value_mode "${SELECTED_VALUE_MODE:-exact}" \
  --selected_value_exact_rule "${SELECTED_VALUE_EXACT_RULE:-fixed}" \
  --selected_value_exact_top "${SELECTED_VALUE_EXACT_TOP:-0}" \
  --selected_value_exact_mass "${SELECTED_VALUE_EXACT_MASS:-0.0}" \
  --selected_value_exact_risk_mass "${SELECTED_VALUE_EXACT_RISK_MASS:-0.0}" \
  --selected_value_min_exact_top "${SELECTED_VALUE_MIN_EXACT_TOP:-0}" \
  --selected_value_max_exact_top "${SELECTED_VALUE_MAX_EXACT_TOP:-0}" \
  --selected_value_residual_correction "${SELECTED_VALUE_RESIDUAL_CORRECTION:-none}" \
  --selected_value_residual_norm_bytes "${SELECTED_VALUE_RESIDUAL_NORM_BYTES:-2}" \
  --tail_blend 1.0 \
  --tail_blend_rule fixed \
  --tail_samples 0 \
  --page_size 5632 \
  --subvecs "${SUBVECS:-4}" \
  --subbits "${SUBBITS:-8}" \
  --value_subvecs "${VALUE_SUBVECS:-0}" \
  --value_subbits "${VALUE_SUBBITS:-0}" \
  --kmeans_iters 3 \
  --key_bytes 2 \
  --value_bytes 2 \
  --nprobes 512 \
  --static_prefix 128 \
  --static_suffix 128
