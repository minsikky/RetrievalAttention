#!/usr/bin/env bash
set -euo pipefail

cd /scratch/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention
export LD_LIBRARY_PATH="/sw/pkgs/arc/python/3.10.4/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

RUN_NAME="${RUN_NAME:?RUN_NAME is required}"
OUT_DIR="attention_efficiency_result/${RUN_NAME}"
if [[ -f "${OUT_DIR}/summary.json" ]]; then
  echo "[hf_selectedv_intervention_one] skip existing ${RUN_NAME}"
  exit 0
fi

args=(
  benchmark/selector_eval/runners/run_hf_paged_pq_intervention_eval.py
  --output_dir "${OUT_DIR}" \
  --layers "${LAYERS:-16}" \
  --filler_repeats "${FILLER_REPEATS:-1024}" \
  --max_new_tokens "${MAX_NEW_TOKENS:-8}" \
  --selector_mode "${SELECTOR_MODE:-fullscan}" \
  --budget "${BUDGET:?BUDGET is required}" \
  --tail_mode "${TAIL_MODE:-vpq_value}" \
  --tail_probe_rel_l2_max "${TAIL_PROBE_REL_L2_MAX:-0.1}" \
  --selected_value_mode "${SELECTED_VALUE_MODE:-vpq_value}" \
  --selected_value_exact_rule "${SELECTED_VALUE_EXACT_RULE:?SELECTED_VALUE_EXACT_RULE is required}" \
  --selected_value_exact_mass "${SELECTED_VALUE_EXACT_MASS:-0.0}" \
  --selected_value_exact_risk_mass "${SELECTED_VALUE_EXACT_RISK_MASS:-0.0}" \
  --selected_value_min_exact_top "${SELECTED_VALUE_MIN_EXACT_TOP:-0}" \
  --selected_value_max_exact_top "${SELECTED_VALUE_MAX_EXACT_TOP:-0}" \
  --selected_value_residual_norm_bytes "${SELECTED_VALUE_RESIDUAL_NORM_BYTES:-2}" \
  --page_size "${PAGE_SIZE:-5632}" \
  --subvecs "${SUBVECS:-4}" \
  --subbits "${SUBBITS:-8}" \
  --value_subvecs "${VALUE_SUBVECS:-1}" \
  --value_subbits "${VALUE_SUBBITS:-4}" \
  --kmeans_iters "${KMEANS_ITERS:-3}" \
  --nprobes "${NPROBES:-512}" \
  --static_prefix "${STATIC_PREFIX:-128}" \
  --static_suffix "${STATIC_SUFFIX:-128}" \
  --key_bytes "${KEY_BYTES:-2}" \
  --value_bytes "${VALUE_BYTES:-2}" \
  --device "${DEVICE:-cuda}"
)

if [[ "${LOCAL_FILES_ONLY:-0}" == "1" ]]; then
  args+=(--local_files_only)
fi

echo "[hf_selectedv_intervention_one] run ${RUN_NAME}"
.venv/bin/python "${args[@]}"
