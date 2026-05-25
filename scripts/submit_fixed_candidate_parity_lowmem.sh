#!/usr/bin/env bash
set -euo pipefail

MAIN_ROOT="${MAIN_ROOT:-/gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention}"
STAMP="${STAMP:-20260523_lowmem_parity}"
LOG_ROOT="${LOG_ROOT:-${MAIN_ROOT}/slurm_out/parallel_gpu_opt_fixedpaths_${STAMP}}"
MANIFEST="${MANIFEST:-${MAIN_ROOT}/notes/slurm_manifests/parallel_gpu_opt_fixedpaths_${STAMP}.tsv}"

TRACE="${TRACE:-${MAIN_ROOT}/attention_efficiency_result/real_qkv_llama31_l16_6838_g131072_q288_window32_graphall_s16.npz}"
X_TRACE="${X_TRACE:-${MAIN_ROOT}/attention_efficiency_result/real_xtrace_llama31_8b_layer16_8k_g131072_sampled_maskstop.npz}"

mkdir -p "${LOG_ROOT}" "$(dirname "${MANIFEST}")"
echo -e "candidate\tlabel\tjobid\toutput_dir\tlog" >"${MANIFEST}"

submit_parity() {
  local candidate="$1"
  local worktree="$2"
  local flag_name="$3"
  local output_dir="${MAIN_ROOT}/attention_efficiency_result/parallel_gpu_opt_fixedpaths_${STAMP}/${candidate}/parity_long"
  local log_path="${LOG_ROOT}/${candidate}_parity_long-%j.out"
  mkdir -p "${output_dir}"

  local jobid
  jobid="$(
    sbatch --parsable \
      --chdir="${worktree}" \
      --mem=32000m \
      --time=00:20:00 \
      --output="${log_path}" \
      --export="ALL,RETRIEVAL_ATTENTION_ROOT=${worktree},TRACE=${TRACE},X_TRACE=${X_TRACE},OUTPUT_DIR=${output_dir},PARITY_PRESET=long,COMPARE_TORCH_GPU_POLICY=1,USE_NATIVE_VPREFIX=1,USE_NATIVE_RISK_PREFIX=1,USE_NATIVE_SCORE_GRID=1,USE_NATIVE_POLICY=1,FRONTIER_CANONICAL_GPU=0,${flag_name}=1" \
      "${worktree}/scripts/run_joint_kv_cpu_gpu_parity_one.sh"
  )"
  echo -e "${candidate}\tparity_long\t${jobid}\t${output_dir}\t${log_path}" | tee -a "${MANIFEST}"
}

submit_parity \
  "rank_prefix_topk" \
  "${MAIN_ROOT}/worktrees/opt-rank-prefix-topk" \
  "SELECTOR_PQ_JOINT_RANK_PREFIX_TOPK"

submit_parity \
  "nofill_score_grid" \
  "${MAIN_ROOT}/worktrees/opt-nofill-score-grid" \
  "SELECTOR_PQ_JOINT_SCORE_GRID_NO_EXACT_FILL"

echo "manifest=${MANIFEST}"
