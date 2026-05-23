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
echo "[joint_kv_policy] score_proxy_variants=${SCORE_PROXY_VARIANTS:-baseline}"

.venv/bin/python benchmark/selector_eval/runners/run_joint_kv_budget_policy_eval.py \
  --qkv_trace "${QKV_TRACE:-attention_efficiency_result/real_qkv_llama31_l16_6838_g131072_q288_window32_graphall_s16.npz}" \
  --x_trace "${X_TRACE:-attention_efficiency_result/real_xtrace_llama31_8b_layer16_8k_g131072_sampled_maskstop.npz}" \
  --output_dir "${out_dir}" \
  --decode_lengths "${DECODE_LENGTHS:-500,1000,2000,4000,8000,16000,32000,64000,128000}" \
  --max_qidx_per_decode "${MAX_QIDX_PER_DECODE:-1}" \
  --heads "${HEADS:-}" \
  --k_budgets "${K_BUDGETS:-4096,8192,14336,32768}" \
  --v_budgets "${V_BUDGETS:-1024,2048,4096,6144,8192,12288,16384}" \
  --stability_thresholds "${STABILITY_THRESHOLDS:-0.0005,0.001,0.002}" \
  --policies "${POLICIES:-k_first_priority,v_first_priority,k_first_alternating,v_first_alternating,sensitivity_greedy}" \
  --score_proxy_variants "${SCORE_PROXY_VARIANTS:-baseline}" \
  --selector_mode "${SELECTOR_MODE:-fullscan}" \
  --tail_score_calibration "${TAIL_SCORE_CALIBRATION:-affine_selected}" \
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
  --device cpu

echo "[joint_kv_policy] finished=$(date --iso-8601=seconds)"
cat "${out_dir}/summary.json"
