#!/usr/bin/env bash
set -euo pipefail

cd /scratch/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention
export LD_LIBRARY_PATH="/sw/pkgs/arc/python/3.10.4/lib:${LD_LIBRARY_PATH:-}"

QKV_TRACE="/scratch/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention/attention_efficiency_result/real_qkv_llama31_l16_6838_g131072_q36_graphall_s16.npz"
X_TRACE="/scratch/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention/attention_efficiency_result/real_xtrace_llama31_8b_layer16_8k_g131072_sampled_maskstop.npz"
BASE_OUT="/scratch/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention/attention_efficiency_result"
CONF_BUDGETS="4096,8192,12288,14336,16384,17408,18432,19456,20480,24576,28672,32768"

run_one() {
  local name="$1"
  local tail_budget="$2"
  local probe_budget="$3"
  local out_dir="${BASE_OUT}/${name}"
  if [[ -f "${out_dir}/summary.json" ]]; then
    echo "[vpq_lowbudget_sweep] skip existing ${name}"
    return
  fi
  echo "[vpq_lowbudget_sweep] run ${name}"
  .venv/bin/python benchmark/selector_eval/runners/run_layer_quality_eval.py \
    --qkv_trace "${QKV_TRACE}" \
    --x_trace "${X_TRACE}" \
    --output_dir "${out_dir}" \
    --decode_lengths 128000 \
    --max_qidx_per_decode 1 \
    --device cpu \
    --selector_mode fullscan \
    --budgets 16384 \
    --online_confidence_rule probe_tail_switch \
    --tail_confidence_budget "${tail_budget}" \
    --tail_probe_budget "${probe_budget}" \
    --confidence_budgets "${CONF_BUDGETS}" \
    --tail_probe_rel_l2_max 0.050 \
    --tail_score_calibration affine_selected \
    --proxy_mass_target 0.99 \
    --marginal_mass_max 0.010 \
    --marginal_score_gap_max -6.0 \
    --tail_mode vpq_value \
    --tail_blend 1.0 \
    --tail_blend_rule fixed \
    --tail_samples 0 \
    --page_size 5632 \
    --subvecs 4 \
    --subbits 6 \
    --kmeans_iters 3 \
    --key_bytes 2 \
    --value_bytes 2 \
    --nprobes 512 \
    --static_prefix 128 \
    --static_suffix 128
}

run_one probe_tail_b16384_p16384_l050_vpq_blend100_128k_v1 16384 16384
run_one probe_tail_b14336_p16384_l050_vpq_blend100_128k_v1 14336 16384
run_one probe_tail_b12288_p14336_l050_vpq_blend100_128k_v1 12288 14336
run_one probe_tail_b12288_p16384_l050_vpq_blend100_128k_v1 12288 16384
run_one probe_tail_b8192_p12288_l050_vpq_blend100_128k_v1 8192 12288
