#!/usr/bin/env bash
#SBATCH --job-name=geom-rank-diag
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64000m
#SBATCH --time=02:00:00
#SBATCH --account=zhengya98
#SBATCH --partition=spgpu
#SBATCH --gpus-per-node=1
set -euo pipefail

cd /gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention

module purge
module load python/3.10.4
module load "${CUDA_MODULE:-cuda/12.8.1}"
source .venv/bin/activate

export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.10/site-packages/torch/lib:/sw/pkgs/arc/python/3.10.4/lib:${LD_LIBRARY_PATH:-}"

name="${RUN_NAME:?RUN_NAME is required}"
out_dir="${OUTPUT_ROOT:-attention_efficiency_result/geometric_rank_attribution_20260519}/${name}"
mkdir -p "${out_dir}"

echo "[geom_rank_diag] host=$(hostname)"
echo "[geom_rank_diag] started=$(date --iso-8601=seconds)"
echo "[geom_rank_diag] out=${out_dir}"
echo "[geom_rank_diag] selector=${SELECTOR_MODE:-fullscan} rule=${ONLINE_CONFIDENCE_RULE:-geometric_probe_tail_switch}"

.venv/bin/python benchmark/selector_eval/runners/run_layer_quality_eval.py \
  --qkv_trace "${QKV_TRACE:-attention_efficiency_result/real_qkv_llama31_l16_8k_g32768_q4_d32000_graphall_s16.npz}" \
  --x_trace "${X_TRACE:-attention_efficiency_result/real_xtrace_llama31_8b_layer16_8k_g32768.npz}" \
  --output_dir "${out_dir}" \
  --decode_lengths "${DECODE_LENGTHS:-32000}" \
  --max_qidx_per_decode "${MAX_QIDX_PER_DECODE:-1}" \
  --device "${DEVICE:-cuda}" \
  --head_only \
  --selector_mode "${SELECTOR_MODE:-fullscan}" \
  --budgets "${BUDGETS:-4096}" \
  --online_confidence_rule "${ONLINE_CONFIDENCE_RULE:-geometric_probe_tail_switch}" \
  --tail_score_calibration "${TAIL_SCORE_CALIBRATION:-none}" \
  --tail_mode "${TAIL_MODE:-vpq_value}" \
  --tail_blend "${TAIL_BLEND:-1.0}" \
  --tail_probe_rel_l2_max "${TAIL_PROBE_REL_L2_MAX:-0.020}" \
  --tail_proxy_mass_min "${TAIL_PROXY_MASS_MIN:-0.0}" \
  --tail_proxy_mass_max "${TAIL_PROXY_MASS_MAX:-1.0}" \
  --tail_pq_corr_min "${TAIL_PQ_CORR_MIN:--1.0}" \
  --tail_pq_relrmse_max "${TAIL_PQ_RELRMSE_MAX:-inf}" \
  --geometric_min_budget "${GEOMETRIC_MIN_BUDGET:-4096}" \
  --geometric_max_budget "${GEOMETRIC_MAX_BUDGET:-32768}" \
  --geometric_growth "${GEOMETRIC_GROWTH:-1.5}" \
  --geometric_probe_scale "${GEOMETRIC_PROBE_SCALE:-1.5}" \
  --geometric_budget_granularity "${GEOMETRIC_BUDGET_GRANULARITY:-1024}" \
  --exact_delta_rel_l2_max "${EXACT_DELTA_REL_L2_MAX:-0.020}" \
  --selected_value_mode "${SELECTED_VALUE_MODE:-vpq_value}" \
  --selected_value_exact_rule "${SELECTED_VALUE_EXACT_RULE:-selector_rank}" \
  --selected_value_exact_top "${SELECTED_VALUE_EXACT_TOP:-256}" \
  --selected_value_exact_mass "${SELECTED_VALUE_EXACT_MASS:-0.0}" \
  --selected_value_min_exact_top "${SELECTED_VALUE_MIN_EXACT_TOP:-0}" \
  --selected_value_max_exact_top "${SELECTED_VALUE_MAX_EXACT_TOP:-0}" \
  --value_subvecs "${VALUE_SUBVECS:-1}" \
  --value_subbits "${VALUE_SUBBITS:-4}" \
  --page_size "${PAGE_SIZE:-2048}" \
  --subvecs "${SUBVECS:-4}" \
  --subbits "${SUBBITS:-8}" \
  --kmeans_iters "${KMEANS_ITERS:-3}" \
  --key_bytes 2 \
  --value_bytes 2 \
  --nprobes "${NPROBES:-512}" \
  --static_prefix "${STATIC_PREFIX:-128}" \
  --static_suffix "${STATIC_SUFFIX:-128}"

echo "[geom_rank_diag] finished=$(date --iso-8601=seconds)"
cat "${out_dir}/summary.json"
