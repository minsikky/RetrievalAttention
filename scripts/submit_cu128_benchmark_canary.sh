#!/usr/bin/env bash
set -euo pipefail

cd /gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
SUITE_NAME="${SUITE_NAME:-cu128_benchmark_canary_${STAMP}}"
MANIFEST="${MANIFEST:-notes/slurm_manifests/${SUITE_NAME}.tsv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-benchmark_canary_result/${SUITE_NAME}}"
SLURM_ROOT="${SLURM_ROOT:-slurm_out/${SUITE_NAME}}"
PARTITIONS="${PARTITIONS:-gpu-rtx6000,spgpu,gpu_mig40}"

mkdir -p "$(dirname "${MANIFEST}")" "${OUTPUT_ROOT}" "${SLURM_ROOT}"
printf "label\tjobid\toutput_dir\tslurm_out\n" > "${MANIFEST}"

submit_job() {
  local label="$1"
  local out_dir="$2"
  local script="$3"
  shift 3
  local slurm_out="${SLURM_ROOT}/${label}-%j.out"
  local export_csv="ALL,HF_VENV_DIR=.venv_cu128,HF_EXTRA_PYTHONPATH=.hf_pydeps_cu128,TORCH_CUDA_ARCH_LIST=8.0;8.6;12.0"
  local item
  for item in "$@"; do
    export_csv+=",${item}"
  done
  local jobid
  jobid="$(
    sbatch --parsable \
      --partition="${PARTITIONS}" \
      --output="${slurm_out}" \
      --export="${export_csv}" \
      "${script}"
  )"
  printf "%s\t%s\t%s\t%s\n" "${label}" "${jobid}" "${out_dir}" "${slurm_out/\%j/${jobid}}" | tee -a "${MANIFEST}"
}

public_common=(
  HF_MODEL_PRESET=qwen3_8b
  HF_LANGUAGE_MODEL_ONLY=1
  USE_CHAT_TEMPLATE=1
  DISABLE_THINKING=1
  BENCHMARK=aime24
  MAX_EXAMPLES=1
  TASK_OFFSET=0
  MAX_INPUT_TOKENS=8192
  MAX_NEW_TOKENS=32
  MIN_NEW_TOKENS=0
  LOCAL_FILES_ONLY=1
  USE_CHAT_TEMPLATE=1
  LOW_CPU_MEM_USAGE=1
  RUN_NAME=unused
)

submit_job \
  public_dense_aime24_n1 \
  "${OUTPUT_ROOT}/public_dense_aime24_n1" \
  benchmark/run_public_longdecode_hf.sh \
  "${public_common[@]}" \
  ATTENTION_MODE=dense \
  OUTPUT_DIR="${OUTPUT_ROOT}/public_dense_aime24_n1" \
  RUN_NAME=public_dense_aime24_n1

submit_job \
  public_frontier_aime24_n1 \
  "${OUTPUT_ROOT}/public_frontier_aime24_n1" \
  benchmark/run_public_longdecode_hf.sh \
  "${public_common[@]}" \
  ATTENTION_MODE=pagedpq \
  OUTPUT_DIR="${OUTPUT_ROOT}/public_frontier_aime24_n1" \
  RUN_NAME=public_frontier_aime24_n1

submit_job \
  ruler_dense_8k_n1 \
  "${OUTPUT_ROOT}/ruler_dense_8k_n1" \
  scripts/run_dense_ruler_batched_one.sh \
  TASK_NAME=niah_single_1 \
  CONTEXT_LEN=8192 \
  NUM_SAMPLES=1 \
  MAX_NEW_TOKENS=32 \
  OUTPUT_ROOT="${OUTPUT_ROOT}/ruler_dense_8k_n1"

submit_job \
  ruler_frontier_8k_n1 \
  "${OUTPUT_ROOT}/ruler_frontier_8k_n1" \
  scripts/run_frontier_ruler_batched_one.sh \
  TASK_NAME=niah_single_1 \
  CONTEXT_LEN=8192 \
  NUM_SAMPLES=1 \
  MAX_NEW_TOKENS=32 \
  OUTPUT_ROOT="${OUTPUT_ROOT}/ruler_frontier_8k_n1" \
  FRONTIER_CANONICAL_GPU=1

lbv2_common=(
  HF_MODEL_PRESET=qwen3_8b
  HF_LANGUAGE_MODEL_ONLY=1
  USE_CHAT_TEMPLATE=1
  DISABLE_THINKING=1
  MAX_EXAMPLES=1
  LENGTH_FILTER=short
  DIFFICULTY_FILTER=easy
  MAX_INPUT_TOKENS=8192
  MAX_NEW_TOKENS=32
  LOCAL_FILES_ONLY=1
)

submit_job \
  lbv2_dense_short_n1 \
  "${OUTPUT_ROOT}/lbv2_dense_short_n1" \
  benchmark/run_longbench_v2_hf.sh \
  "${lbv2_common[@]}" \
  ATTENTION_MODE=dense \
  OUTPUT_DIR="${OUTPUT_ROOT}/lbv2_dense_short_n1"

submit_job \
  lbv2_frontier_short_n1 \
  "${OUTPUT_ROOT}/lbv2_frontier_short_n1" \
  scripts/run_frontier_longbench_v2_one.sh \
  "${lbv2_common[@]}" \
  ATTENTION_MODE=pagedpq \
  OUTPUT_DIR="${OUTPUT_ROOT}/lbv2_frontier_short_n1" \
  FRONTIER_CANONICAL_GPU=1

echo "[INFO] Manifest: ${MANIFEST}"
