#!/usr/bin/env bash
set -euo pipefail

MAIN_ROOT="${MAIN_ROOT:-/gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention}"
STAMP="${STAMP:-20260523_fixedpaths}"
LOG_ROOT="${LOG_ROOT:-${MAIN_ROOT}/slurm_out/parallel_gpu_opt_fixedpaths_${STAMP}}"
MANIFEST="${MANIFEST:-${MAIN_ROOT}/notes/slurm_manifests/parallel_gpu_opt_fixedpaths_${STAMP}.tsv}"

TRACE="${TRACE:-${MAIN_ROOT}/attention_efficiency_result/real_qkv_llama31_l16_6838_g131072_q288_window32_graphall_s16.npz}"
X_TRACE="${X_TRACE:-${MAIN_ROOT}/attention_efficiency_result/real_xtrace_llama31_8b_layer16_8k_g131072_sampled_maskstop.npz}"
HF_CACHE_DIR="${HF_CACHE_DIR:-${MAIN_ROOT}/.hf_cache}"
HF_VENV_DIR="${HF_VENV_DIR:-${MAIN_ROOT}/.venv}"
HF_EXTRA_PYTHONPATH="${HF_EXTRA_PYTHONPATH:-${MAIN_ROOT}/.hf_pydeps}"
MODEL_NAME="${MODEL_NAME:-${HF_CACHE_DIR}/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218}"
LONGGENBENCH_MOZHU_REPO="${LONGGENBENCH_MOZHU_REPO:-${MAIN_ROOT}/third_party/benchmarks/LongGenBench_mozhu}"
LONGGENBENCH_DOMINIC_REPO="${LONGGENBENCH_DOMINIC_REPO:-${MAIN_ROOT}/third_party/benchmarks/LongGenBench_dominic}"

mkdir -p "${LOG_ROOT}" "$(dirname "${MANIFEST}")"
echo -e "candidate\tlabel\tjobid\toutput_dir\tlog" >"${MANIFEST}"

submit() {
  local candidate="$1"
  local label="$2"
  local worktree="$3"
  local script="$4"
  local output_dir="$5"
  shift 5

  mkdir -p "$(dirname "${output_dir}")"
  local log_path="${LOG_ROOT}/${candidate}_${label}-%j.out"
  local export_arg="ALL,RETRIEVAL_ATTENTION_ROOT=${worktree}"
  local kv
  for kv in "$@"; do
    export_arg+=",${kv}"
  done
  local jobid
  jobid="$(
    sbatch --parsable \
      --chdir="${worktree}" \
      --output="${log_path}" \
      --export="${export_arg}" \
      "${worktree}/${script}"
  )"
  echo -e "${candidate}\t${label}\t${jobid}\t${output_dir}\t${log_path}" | tee -a "${MANIFEST}"
}

submit_candidate() {
  local candidate="$1"
  local worktree="$2"
  local flag_name="$3"

  local result_root="${MAIN_ROOT}/attention_efficiency_result/parallel_gpu_opt_fixedpaths_${STAMP}/${candidate}"
  local long_root="${MAIN_ROOT}/public_longdecode_result/parallel_gpu_opt_fixedpaths_${STAMP}/${candidate}"

  submit "${candidate}" "parity_long" "${worktree}" "scripts/run_joint_kv_cpu_gpu_parity_one.sh" "${result_root}/parity_long" \
    TRACE="${TRACE}" \
    X_TRACE="${X_TRACE}" \
    OUTPUT_DIR="${result_root}/parity_long" \
    PARITY_PRESET=long \
    COMPARE_TORCH_GPU_POLICY=1 \
    USE_NATIVE_VPREFIX=1 \
    USE_NATIVE_RISK_PREFIX=1 \
    USE_NATIVE_SCORE_GRID=1 \
    USE_NATIVE_POLICY=1 \
    FRONTIER_CANONICAL_GPU=0 \
    "${flag_name}=1"

  submit "${candidate}" "longgen8192_accounting" "${worktree}" "benchmark/run_public_longdecode_hf.sh" "${long_root}/longgen8192_accounting" \
    HF_CACHE_DIR="${HF_CACHE_DIR}" \
    HF_VENV_DIR="${HF_VENV_DIR}" \
    HF_EXTRA_PYTHONPATH="${HF_EXTRA_PYTHONPATH}" \
    MODEL_NAME="${MODEL_NAME}" \
    LONGGENBENCH_MOZHU_REPO="${LONGGENBENCH_MOZHU_REPO}" \
    LONGGENBENCH_DOMINIC_REPO="${LONGGENBENCH_DOMINIC_REPO}" \
    OUTPUT_DIR="${long_root}/longgen8192_accounting" \
    ATTENTION_MODE=pagedpq \
    BENCHMARK=longgenbench_sgt_short \
    MAX_EXAMPLES=1 \
    MAX_NEW_TOKENS=8192 \
    MIN_NEW_TOKENS=8192 \
    FORCE_MAX_NEW_TOKENS=1 \
    FRONTIER_CANONICAL_GPU=0 \
    DISABLE_COST_STATS=0 \
    PROFILE_NATIVE_OPS=1 \
    SELECTOR_PQ_JOINT_WALL_PROFILE=1 \
    "${flag_name}=1"

  submit "${candidate}" "longgen16384_accounting" "${worktree}" "benchmark/run_public_longdecode_hf.sh" "${long_root}/longgen16384_accounting" \
    HF_CACHE_DIR="${HF_CACHE_DIR}" \
    HF_VENV_DIR="${HF_VENV_DIR}" \
    HF_EXTRA_PYTHONPATH="${HF_EXTRA_PYTHONPATH}" \
    MODEL_NAME="${MODEL_NAME}" \
    LONGGENBENCH_MOZHU_REPO="${LONGGENBENCH_MOZHU_REPO}" \
    LONGGENBENCH_DOMINIC_REPO="${LONGGENBENCH_DOMINIC_REPO}" \
    OUTPUT_DIR="${long_root}/longgen16384_accounting" \
    ATTENTION_MODE=pagedpq \
    BENCHMARK=longgenbench_sgt_short \
    MAX_EXAMPLES=1 \
    MAX_NEW_TOKENS=16384 \
    MIN_NEW_TOKENS=16384 \
    FORCE_MAX_NEW_TOKENS=1 \
    FRONTIER_CANONICAL_GPU=0 \
    DISABLE_COST_STATS=0 \
    PROFILE_NATIVE_OPS=1 \
    SELECTOR_PQ_JOINT_WALL_PROFILE=1 \
    "${flag_name}=1"
}

submit_candidate \
  "pq_score_topk_fusion" \
  "${MAIN_ROOT}/worktrees/opt-pq-score-topk-fusion" \
  "SELECTOR_PQ_JOINT_SCORE_TOPK_FUSION"

submit_candidate \
  "rank_prefix_topk" \
  "${MAIN_ROOT}/worktrees/opt-rank-prefix-topk" \
  "SELECTOR_PQ_JOINT_RANK_PREFIX_TOPK"

submit_candidate \
  "nofill_score_grid" \
  "${MAIN_ROOT}/worktrees/opt-nofill-score-grid" \
  "SELECTOR_PQ_JOINT_SCORE_GRID_NO_EXACT_FILL"

echo "manifest=${MANIFEST}"
