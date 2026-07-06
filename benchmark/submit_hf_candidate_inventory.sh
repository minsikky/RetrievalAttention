#!/bin/bash
set -euo pipefail

TAG="${TAG:-$(date +%Y%m%d_%H%M%S)}"
MANIFEST="${MANIFEST:-/tmp/hf_candidate_inventory_${TAG}.tsv}"
SUBMIT="${SUBMIT:-0}"
PARTITIONS="${PARTITIONS:-spgpu}"

submit_inventory() {
  local label="$1"
  local preset="$2"
  local output_dir="generated_memory_hf_eval_result/inventory_${label}_${TAG}"
  local jobid="DRYRUN"
  if [ "${SUBMIT}" = "1" ] || [ "${SUBMIT}" = "true" ] || [ "${SUBMIT}" = "yes" ]; then
    jobid="$(sbatch --parsable \
      --partition="${PARTITIONS}" \
      --job-name="hf-inv-${label}" \
      --export="ALL,HF_MODEL_PRESET=${preset},OUTPUT_DIR=${output_dir},INVENTORY_ONLY=1,HF_ATTENTION_MODE=native,NUM_SAMPLES=1,NUM_ENTRIES=1,NUM_QUERIES=1" \
      benchmark/run_generated_memory_hf.sh)"
  fi
  printf '%s\t%s\t%s\n' "${label}" "${jobid}" "${output_dir}" >> "${MANIFEST}"
  printf '[submit_hf_candidate_inventory] %s job=%s output=%s\n' "${label}" "${jobid}" "${output_dir}"
}

printf 'label\tjobid\toutput_dir\n' > "${MANIFEST}"
submit_inventory "qwen3_8b" "qwen3_8b"
submit_inventory "qwen3_14b" "qwen3_14b"
submit_inventory "qwen3_5_9b" "qwen3_5_9b"
submit_inventory "mistral_nemo_12b" "mistral_nemo_12b"
submit_inventory "glm4_9b" "glm4_9b"
submit_inventory "phi4_reasoning_14b" "phi4_reasoning_14b"
printf '[submit_hf_candidate_inventory] manifest=%s\n' "${MANIFEST}"
if [ "${SUBMIT}" != "1" ] && [ "${SUBMIT}" != "true" ] && [ "${SUBMIT}" != "yes" ]; then
  printf '[submit_hf_candidate_inventory] dry run only; set SUBMIT=1 to launch jobs\n'
fi
