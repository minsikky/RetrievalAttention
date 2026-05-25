#!/usr/bin/env bash
#SBATCH --job-name=jointkv-parity
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64000m
#SBATCH --time=00:30:00
#SBATCH --account=zhengya98
#SBATCH --partition=spgpu
#SBATCH --gpus-per-node=1
set -euo pipefail

cd /gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention

export LD_LIBRARY_PATH="/sw/pkgs/arc/python/3.10.4/lib:${LD_LIBRARY_PATH:-}"
module purge
module load python/3.10.4
source .venv/bin/activate

TRACE="${TRACE:-attention_efficiency_result/real_qkv_llama31_l16_6838_g131072_q288_window32_graphall_s16.npz}"
X_TRACE="${X_TRACE:-attention_efficiency_result/real_xtrace_llama31_8b_layer16_8k_g131072_sampled_maskstop.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-attention_efficiency_result/joint_kv_cpu_gpu_parity_20260522/smoke}"
case "${PARITY_PRESET:-default}" in
  default)
    DEFAULT_DECODE_LENGTHS="500,1000"
    DEFAULT_HEADS="0,8"
    ;;
  long)
    DEFAULT_DECODE_LENGTHS="32000,64000,128000"
    DEFAULT_HEADS="0,8"
    ;;
  *)
    echo "unknown PARITY_PRESET=${PARITY_PRESET}; expected default or long" >&2
    exit 2
    ;;
esac
case "${JOINT_KV_SCHEDULE_PRESET:-default}" in
  default)
    DEFAULT_JOINT_KV_K_BUDGETS="4096,8192,14336,32768"
    DEFAULT_JOINT_KV_V_BUDGETS="1024,2048,4096,6144,8192,12288,16384"
    ;;
  coarse)
    DEFAULT_JOINT_KV_K_BUDGETS="4096,8192,32768"
    DEFAULT_JOINT_KV_V_BUDGETS="2048,8192,16384"
    ;;
  coarse2)
    DEFAULT_JOINT_KV_K_BUDGETS="4096,32768"
    DEFAULT_JOINT_KV_V_BUDGETS="4096,16384"
    ;;
  coarse_v8k)
    DEFAULT_JOINT_KV_K_BUDGETS="4096,32768"
    DEFAULT_JOINT_KV_V_BUDGETS="4096,8192"
    ;;
  mid_v8k)
    DEFAULT_JOINT_KV_K_BUDGETS="4096,8192,16384"
    DEFAULT_JOINT_KV_V_BUDGETS="4096,8192"
    ;;
  *)
    echo "unknown JOINT_KV_SCHEDULE_PRESET=${JOINT_KV_SCHEDULE_PRESET}; expected default, coarse, coarse2, coarse_v8k, or mid_v8k" >&2
    exit 2
    ;;
esac

echo "[jointkv_parity] host=$(hostname)"
echo "[jointkv_parity] started=$(date --iso-8601=seconds)"
echo "[jointkv_parity] trace=${TRACE}"
echo "[jointkv_parity] x_trace=${X_TRACE}"
echo "[jointkv_parity] output=${OUTPUT_DIR}"

.venv/bin/python benchmark/selector_eval/gpu/run_joint_kv_cpu_gpu_parity_eval.py \
  --trace "${TRACE}" \
  --x_trace "${X_TRACE}" \
  --output_dir "${OUTPUT_DIR}" \
  --decode_lengths "${DECODE_LENGTHS:-${DEFAULT_DECODE_LENGTHS}}" \
  --heads "${HEADS:-${DEFAULT_HEADS}}" \
  --policies "${JOINT_KV_POLICY:-k_first_alternating}" \
  --thresholds "${JOINT_KV_STABILITY_THRESHOLD:-0.001}" \
  --k_budgets "${JOINT_KV_K_BUDGETS:-${DEFAULT_JOINT_KV_K_BUDGETS}}" \
  --v_budgets "${JOINT_KV_V_BUDGETS:-${DEFAULT_JOINT_KV_V_BUDGETS}}" \
  --page_size "${PAGE_SIZE:-5632}" \
  --subvecs "${SUBVECS:-4}" \
  --subbits "${SUBBITS:-8}" \
  --value_subvecs "${VALUE_SUBVECS:-1}" \
  --value_subbits "${VALUE_SUBBITS:-4}" \
  --kmeans_iters "${KMEANS_ITERS:-3}" \
  --key_bytes "${KEY_BYTES:-2}" \
  --value_bytes "${VALUE_BYTES:-2}" \
  --value_code_stat_bytes "${VALUE_CODE_STAT_BYTES:-2}" \
  --tail_score_calibration "${TAIL_SCORE_CALIBRATION:-affine_selected}" \
  --output_rel_l2_tolerance "${OUTPUT_REL_L2_TOLERANCE:-5e-4}" \
  --oproj_rel_l2_tolerance "${OPROJ_REL_L2_TOLERANCE:-5e-4}" \
  --device cuda \
  ${COMPARE_TORCH_GPU_POLICY:+--compare_torch_gpu_policy} \
  ${USE_NATIVE_VPREFIX:+--use_native_vprefix} \
  ${USE_NATIVE_RISK_PREFIX:+--use_native_risk_prefix} \
  ${USE_NATIVE_RISK_PREFIX_TOPK:+--use_native_risk_prefix_topk} \
  ${USE_NATIVE_SCORE_GRID:+--use_native_score_grid} \
  ${USE_NATIVE_PQ_SCALE_IN_KERNEL:+--use_native_pq_scale_in_kernel} \
  ${USE_TOKENFIT_SCORE_GRID:+--use_tokenfit_score_grid} \
  ${USE_SCORE_GRID_WORKSPACE:+--use_score_grid_workspace} \
  ${USE_NATIVE_POLICY:+--use_native_policy} \
  ${USE_INTERVAL_RISK_POLICY:+--use_interval_risk_policy} \
  ${USE_SCORE_DIRECT_INTERVAL_POLICY:+--use_score_direct_interval_policy}

echo "[jointkv_parity] finished=$(date --iso-8601=seconds)"
