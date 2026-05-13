#!/usr/bin/env bash
set -euo pipefail

cd /scratch/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention
export LD_LIBRARY_PATH="/sw/pkgs/arc/python/3.10.4/lib:${LD_LIBRARY_PATH:-}"

name="${RUN_NAME:?RUN_NAME is required}"
rule="${ONLINE_CONFIDENCE_RULE:?ONLINE_CONFIDENCE_RULE is required}"
out_dir="${OUTPUT_ROOT:-attention_efficiency_result/confidence_budget_rules_20260512}/${name}"

mkdir -p "$(dirname "${out_dir}")"
if [[ -f "${out_dir}/summary.json" ]]; then
  echo "[confidence_budget_rule_one] skip existing ${out_dir}"
  exit 0
fi

echo "[confidence_budget_rule_one] run ${name} rule=${rule}"
.venv/bin/python benchmark/selector_eval/runners/run_layer_quality_eval.py \
  --qkv_trace "${QKV_TRACE:-attention_efficiency_result/real_qkv_llama31_l16_6838_g131072_q288_window32_graphall_s16.npz}" \
  --x_trace "${X_TRACE:-attention_efficiency_result/real_xtrace_llama31_8b_layer16_8k_g131072_sampled_maskstop.npz}" \
  --output_dir "${out_dir}" \
  --decode_lengths "${DECODE_LENGTHS:-76387}" \
  --max_qidx_per_decode "${MAX_QIDX_PER_DECODE:-1}" \
  --device "${DEVICE:-cpu}" \
  --selector_mode "${SELECTOR_MODE:-fullscan}" \
  --budgets "${BUDGETS:-16384}" \
  --online_confidence_rule "${rule}" \
  --tail_score_calibration affine_selected \
  --proxy_mass_target "${PROXY_MASS_TARGET:-0.99}" \
  --marginal_mass_max "${MARGINAL_MASS_MAX:-0.010}" \
  --marginal_score_gap_max "${MARGINAL_SCORE_GAP_MAX:--6.0}" \
  --tail_mode "${TAIL_MODE:-vpq_value}" \
  --tail_confidence_budget "${TAIL_CONFIDENCE_BUDGET:-16384}" \
  --tail_probe_budget "${TAIL_PROBE_BUDGET:-18432}" \
  --tail_probe_rel_l2_max "${TAIL_PROBE_REL_L2_MAX:-0.050}" \
  --stable_tail_probe_rel_l2_max "${STABLE_TAIL_PROBE_REL_L2_MAX:-0.050}" \
  --geometric_min_budget "${GEOMETRIC_MIN_BUDGET:-8192}" \
  --geometric_max_budget "${GEOMETRIC_MAX_BUDGET:-65536}" \
  --geometric_growth "${GEOMETRIC_GROWTH:-1.5}" \
  --geometric_probe_scale "${GEOMETRIC_PROBE_SCALE:-1.5}" \
  --geometric_budget_granularity "${GEOMETRIC_BUDGET_GRANULARITY:-1024}" \
  --exact_delta_rel_l2_max "${EXACT_DELTA_REL_L2_MAX:-0.010}" \
  --tail_blend "${TAIL_BLEND:-1.0}" \
  --tail_blend_rule "${TAIL_BLEND_RULE:-fixed}" \
  --tail_samples 0 \
  --selected_value_mode "${SELECTED_VALUE_MODE:-vpq_value}" \
  --selected_value_exact_rule "${SELECTED_VALUE_EXACT_RULE:-selected_mass}" \
  --selected_value_exact_mass "${SELECTED_VALUE_EXACT_MASS:-0.99}" \
  --selected_value_exact_risk_mass "${SELECTED_VALUE_EXACT_RISK_MASS:-0.0}" \
  --selected_value_min_exact_top "${SELECTED_VALUE_MIN_EXACT_TOP:-0}" \
  --selected_value_max_exact_top "${SELECTED_VALUE_MAX_EXACT_TOP:-0}" \
  --selected_value_residual_correction none \
  --selected_value_residual_norm_bytes 2 \
  --page_size 5632 \
  --subvecs "${SUBVECS:-4}" \
  --subbits "${SUBBITS:-8}" \
  --value_subvecs "${VALUE_SUBVECS:-1}" \
  --value_subbits "${VALUE_SUBBITS:-4}" \
  --kmeans_iters 3 \
  --key_bytes 2 \
  --value_bytes 2 \
  --nprobes 512 \
  --static_prefix 128 \
  --static_suffix 128
