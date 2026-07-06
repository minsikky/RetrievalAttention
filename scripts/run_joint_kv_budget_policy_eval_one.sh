#!/usr/bin/env bash
#SBATCH --job-name=joint-kv-policy
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
out_dir="${OUTPUT_ROOT:-attention_efficiency_result/joint_kv_budget_policy_20260522}/${name}"
mkdir -p "${out_dir}"

echo "[joint_kv_policy] host=$(hostname)"
echo "[joint_kv_policy] started=$(date --iso-8601=seconds)"
echo "[joint_kv_policy] out=${out_dir}"
echo "[joint_kv_policy] decode_lengths=${DECODE_LENGTHS:-500,1000,2000,4000,8000,16000,32000,64000,128000}"
echo "[joint_kv_policy] heads=${HEADS:-all}"
echo "[joint_kv_policy] k_budgets=${K_BUDGETS:-4096,8192,14336,32768}"
echo "[joint_kv_policy] v_budgets=${V_BUDGETS:-1024,2048,4096,6144,8192,12288,16384}"
echo "[joint_kv_policy] k_budget_fracs=${K_BUDGET_FRACS:-0.10,0.30,0.50,0.70,0.90,1.0}"
echo "[joint_kv_policy] v_budget_fracs=${V_BUDGET_FRACS:-0.05,0.10,0.20,0.40,0.60,0.80,1.0}"
echo "[joint_kv_policy] score_proxy_variants=${SCORE_PROXY_VARIANTS:-baseline}"
echo "[joint_kv_policy] start_strategies=${START_STRATEGIES:-proxy_mass_m0p9}"
echo "[joint_kv_policy] v_selection_rules=${V_SELECTION_RULES:-global_residual_risk}"
echo "[joint_kv_policy] v_local_block_size=${V_LOCAL_BLOCK_SIZE:-1024}"
echo "[joint_kv_policy] include_v_selection_state_in_step_mb=${INCLUDE_V_SELECTION_STATE_IN_STEP_MB:-0}"
echo "[joint_kv_policy] survivor_logit_bytes=${SURVIVOR_LOGIT_BYTES:-2}"
echo "[joint_kv_policy] oracle_rel_l2_targets=${ORACLE_REL_L2_TARGETS:-}"
echo "[joint_kv_policy] threshold_mode=${THRESHOLD_MODE:-budget_delta_frac}"
echo "[joint_kv_policy] threshold_scale_shape=${THRESHOLD_SCALE_SHAPE:-sqrt}"
echo "[joint_kv_policy] selector_mode=${SELECTOR_MODE:-fullscan}"
echo "[joint_kv_policy] quest_rank=${QUEST_RANK:-16}"

extra_args=()
if [[ "${INCLUDE_V_SELECTION_STATE_IN_STEP_MB:-0}" == "1" ]]; then
  extra_args+=(--include_v_selection_state_in_step_mb)
fi
if [[ "${LOOKAHEAD_DIAGNOSTIC:-0}" == "1" ]]; then
  extra_args+=(--lookahead_diagnostic)
fi
if [[ -n "${LOOKAHEAD_DECISION_VARIANTS:-}" ]]; then
  extra_args+=(--lookahead_decision_variants "${LOOKAHEAD_DECISION_VARIANTS}")
fi
if [[ "${TEMPORAL_CACHE_STATS:-0}" == "1" ]]; then
  extra_args+=(--temporal_cache_stats)
fi
if [[ -n "${TEMPORAL_REUSE_MAX_STALE:-}" ]]; then
  extra_args+=(--temporal_reuse_max_stale "${TEMPORAL_REUSE_MAX_STALE}")
fi
if [[ -n "${TEMPORAL_REUSE_MODE:-}" ]]; then
  extra_args+=(--temporal_reuse_mode "${TEMPORAL_REUSE_MODE}")
fi
if [[ -n "${TEMPORAL_REUSE_BUDGET:-}" ]]; then
  extra_args+=(--temporal_reuse_budget "${TEMPORAL_REUSE_BUDGET}")
fi
if [[ -n "${PRECISION_K_HI_FRAC:-}" ]]; then
  extra_args+=(--precision_k_hi_frac "${PRECISION_K_HI_FRAC}")
fi
if [[ -n "${PRECISION_V_HI_FRAC:-}" ]]; then
  extra_args+=(--precision_v_hi_frac "${PRECISION_V_HI_FRAC}")
fi
if [[ -n "${PRECISION_LO_BITS:-}" ]]; then
  extra_args+=(--precision_lo_bits "${PRECISION_LO_BITS}")
fi
if [[ -n "${PAGE_SCAN_FRAC:-}" ]]; then
  extra_args+=(--page_scan_frac "${PAGE_SCAN_FRAC}")
fi
if [[ "${GLOBAL_PQ_CODEBOOK:-0}" == "1" ]]; then
  extra_args+=(--global_pq_codebook)
fi
if [[ "${BUDGET_DEESCALATE:-0}" == "1" ]]; then
  extra_args+=(--budget_deescalate)
fi
if [[ "${GQA_UNION_STATS:-0}" == "1" ]]; then
  extra_args+=(--gqa_union_stats)
fi

.venv/bin/python benchmark/selector_eval/runners/run_joint_kv_budget_policy_eval.py \
  --qkv_trace "${QKV_TRACE:-attention_efficiency_result/real_qkv_llama31_l16_6838_g131072_q288_window32_graphall_s16.npz}" \
  --x_trace "${X_TRACE:-attention_efficiency_result/real_xtrace_llama31_8b_layer16_8k_g131072_sampled_maskstop.npz}" \
  --output_dir "${out_dir}" \
  --decode_lengths "${DECODE_LENGTHS:-500,1000,2000,4000,8000,16000,32000,64000,128000}" \
  --max_qidx_per_decode "${MAX_QIDX_PER_DECODE:-1}" \
  --heads "${HEADS:-}" \
  --k_budgets "${K_BUDGETS:-4096,8192,14336,32768}" \
  --v_budgets "${V_BUDGETS:-1024,2048,4096,6144,8192,12288,16384}" \
  --k_budget_fracs "${K_BUDGET_FRACS:-0.10,0.30,0.50,0.70,0.90,1.0}" \
  --v_budget_fracs "${V_BUDGET_FRACS:-0.05,0.10,0.20,0.40,0.60,0.80,1.0}" \
  --stability_thresholds "${STABILITY_THRESHOLDS:-0.002}" \
  --oracle_rel_l2_targets "${ORACLE_REL_L2_TARGETS:-}" \
  --threshold_mode "${THRESHOLD_MODE:-budget_delta_frac}" \
  --threshold_reference_frac "${THRESHOLD_REFERENCE_FRAC:-0.2}" \
  --threshold_scale_shape "${THRESHOLD_SCALE_SHAPE:-sqrt}" \
  --threshold_min_scale "${THRESHOLD_MIN_SCALE:-0.0}" \
  --threshold_max_scale "${THRESHOLD_MAX_SCALE:-1.5}" \
  --policies "${POLICIES:-k_first_priority,v_first_priority,k_first_alternating,v_first_alternating,sensitivity_greedy}" \
  --score_proxy_variants "${SCORE_PROXY_VARIANTS:-baseline}" \
  --start_strategies "${START_STRATEGIES:-proxy_mass_m0p9}" \
  --v_selection_rules "${V_SELECTION_RULES:-global_residual_risk}" \
  --v_local_block_size "${V_LOCAL_BLOCK_SIZE:-1024}" \
  --survivor_logit_bytes "${SURVIVOR_LOGIT_BYTES:-2}" \
  --selector_mode "${SELECTOR_MODE:-fullscan}" \
  --quest_rank "${QUEST_RANK:-16}" \
  --selector_index_bytes "${SELECTOR_INDEX_BYTES:-4}" \
  --tail_score_calibration "${TAIL_SCORE_CALIBRATION:-none}" \
  --page_size "${PAGE_SIZE:-5632}" \
  --subvecs "${SUBVECS:-4}" \
  --subbits "${SUBBITS:-8}" \
  --value_subvecs "${VALUE_SUBVECS:-1}" \
  --value_subbits "${VALUE_SUBBITS:-4}" \
  --kmeans_iters "${KMEANS_ITERS:-3}" \
  --key_bytes "${KEY_BYTES:-2}" \
  --value_bytes "${VALUE_BYTES:-2}" \
  --code_stat_bytes "${CODE_STAT_BYTES:-2}" \
  --nprobes "${NPROBES:-512}" \
  --static_prefix "${STATIC_PREFIX:-128}" \
  --static_suffix "${STATIC_SUFFIX:-128}" \
  "${extra_args[@]}" \
  --device cpu

echo "[joint_kv_policy] finished=$(date --iso-8601=seconds)"
cat "${out_dir}/summary.json"
