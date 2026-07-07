#!/usr/bin/env bash
#SBATCH --job-name=v-risk-key-study
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000m
#SBATCH --time=12:00:00
#SBATCH --account=zhengya0
#SBATCH --partition=standard

# Risk-rank key precision study (issue #7): decision-flip rate of the
# quantized monotone risk key vs the fp64 reference at tau = 0.004,
# swept over exponent-window x mantissa configs, classified against
# eps_band = 2e-5, plus the 12-golden hard gate per config.
set -euo pipefail
cd /gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention
module purge
module load python/3.10.4
export LD_LIBRARY_PATH="/sw/pkgs/arc/python/3.10.4/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

STUDY_TAG="${STUDY_TAG:-v_risk_key_study_20260707}"
ROOT="benchmark_suite_result/${STUDY_TAG}"
# E:M pairs; "0:0" = full-precision reference.
CONFIGS="${CONFIGS:-0:0 5:10 6:6 6:8 6:10 6:12 7:10}"
POP_QIDX="${POP_QIDX:-8}"
GOLDEN_REF_DUMP="${GOLDEN_REF_DUMP:-benchmark_suite_result/selector_eval/golden_stage2_ref}"

run_eval() {
  local out_dir="$1" dump_dir="$2" heads="$3" max_qidx="$4" exp_bits="$5" man_bits="$6"
  local dump_args=()
  [ -n "$dump_dir" ] && { mkdir -p "$dump_dir"; dump_args=(--golden_dump_stage2_dir "$dump_dir"); }
  mkdir -p "$out_dir"
  .venv/bin/python benchmark/selector_eval/runners/run_joint_kv_budget_policy_eval.py \
    --qkv_trace attention_efficiency_result/real_qkv_llama31_l16_6838_g131072_q288_window32_graphall_s16.npz \
    --x_trace attention_efficiency_result/real_xtrace_llama31_8b_layer16_8k_g131072_sampled_maskstop.npz \
    --model_snapshot .hf_cache/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659 \
    --device cpu \
    --decode_lengths 8000,32000,128000 \
    --max_qidx_per_decode "$max_qidx" \
    --heads "$heads" \
    --page_size 5632 --static_prefix 128 --static_suffix 128 \
    --subvecs 4 --subbits 8 --kmeans_iters 3 \
    --value_subvecs 1 --value_subbits 4 \
    --kv_storage_format fp16 --key_bytes 2 --value_bytes 2 \
    --code_stat_bytes 2 --selector_index_bytes 4 --survivor_logit_bytes 2 \
    --selector_mode fullscan --nprobes 512 --page_scan_frac 1.0 \
    --k_budgets 4096,8192,14336,32768 \
    --v_budgets 1024,2048,4096,6144,8192,12288,16384 \
    --k_budget_fracs 0.10,0.30,0.50,0.70,0.90,1.0 \
    --v_budget_fracs 0.05,0.10,0.20,0.40,0.60,0.80,1.0 \
    --policies k_first_alternating \
    --start_strategies proxy_mass_m0p9 \
    --stability_thresholds 0.004 \
    --threshold_mode budget_delta_frac --threshold_reference_frac 0.2 \
    --threshold_scale_shape sqrt --threshold_min_scale 0.0 --threshold_max_scale 1.5 \
    --budget_deescalate \
    --precision_k_hi_frac 0.1 --precision_v_hi_frac 0.1 \
    --precision_lo_mode int8 --precision_lo_bits 8 \
    --v_selection_rules global_residual_risk --v_local_block_size 1024 \
    --logit_buffer_bits 8 --logit_buffer_format e4m3 \
    --tail_score_calibration none \
    --score_proxy_variants baseline \
    --temporal_reuse_mode frozen --temporal_reuse_budget ladder \
    --v_risk_key_exp_bits "$exp_bits" --v_risk_key_mantissa_bits "$man_bits" \
    "${dump_args[@]}" \
    --output_dir "$out_dir"
}

variant_args=()
golden_args=()
for cfg in $CONFIGS; do
  E="${cfg%%:*}"; M="${cfg##*:}"
  tag="e${E}m${M}"
  echo "[study] population run ${tag} (E=${E} M=${M})"
  run_eval "${ROOT}/pop_${tag}" "" "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31" "$POP_QIDX" "$E" "$M"
  echo "[study] golden-gate run ${tag}"
  run_eval "${ROOT}/gold_run_${tag}" "${ROOT}/gold_dump_${tag}" "0,8,16,24" 1 "$E" "$M"
  if [ "$cfg" != "0:0" ]; then
    variant_args+=(--variant "${tag}=${ROOT}/pop_${tag}")
    golden_args+=(--golden_variant "${tag}=${ROOT}/gold_dump_${tag}")
  fi
done

echo "[study] comparing"
.venv/bin/python benchmark/selector_eval/runners/compare_v_risk_key_runs.py \
  --reference "${ROOT}/pop_e0m0" \
  "${variant_args[@]}" \
  --golden_reference "${ROOT}/gold_dump_e0m0" \
  "${golden_args[@]}" \
  --eps_band 2e-5 \
  --out "${ROOT}/report.json"

# Determinism cross-check: the regenerated full-precision golden dump
# must reproduce the FROZEN stage-2 goldens bit-for-bit before it can
# serve as the gate reference.
echo "[study] frozen-dir determinism check"
.venv/bin/python benchmark/selector_eval/runners/compare_v_risk_key_runs.py \
  --reference "${ROOT}/pop_e0m0" \
  --golden_reference "benchmark/selector_eval/golden_vectors/stage2_20260707" \
  --golden_variant "regen_ref=${ROOT}/gold_dump_e0m0" \
  --eps_band 2e-5 \
  --out "${ROOT}/report_frozen_check.json"
echo "[study] done: ${ROOT}/report.json"
