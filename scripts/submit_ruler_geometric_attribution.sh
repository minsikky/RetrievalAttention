#!/usr/bin/env bash
set -euo pipefail

cd /gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention

TAG="${TAG:-20260519}"
OUTPUT_ROOT="${OUTPUT_ROOT:-ruler_eval_result/geometric_rank_attribution_${TAG}}"
SLURM_ROOT="${SLURM_ROOT:-slurm_out/ruler_geometric_rank_attribution_${TAG}}"
MANIFEST="${MANIFEST:-notes/slurm_manifests/ruler_geometric_rank_attribution_${TAG}.tsv}"
DATA_FILE="${DATA_FILE:-ruler_eval_result/geometric_fastpath_diag_20260518/frontier_ruler_ctx32768_n1_niah_single_1_geom_fastpath/data/niah_single_1/validation.jsonl}"
REL="${REL:-0.20}"

mkdir -p "${SLURM_ROOT}" "$(dirname "${MANIFEST}")"
printf 'label\tjobid\toutput_dir\tslurm_log\tselector\trule\tthreshold\n' > "${MANIFEST}"

submit_case() {
  local label="$1"
  local selector="$2"
  local selector_backend="$3"
  local rule="$4"
  local tail_blend="$5"
  local selected_value_mode="$6"
  local selected_value_rule="$7"
  local selected_value_top="$8"
  local run_name="ruler32k_n1_${label}"
  local out_dir="${OUTPUT_ROOT}/${run_name}"
  local slurm_out="${SLURM_ROOT}/${label}-%j.out"
  local export_args
  export_args="ALL,OUTPUT_ROOT=${OUTPUT_ROOT},RUN_NAME=${run_name},TASK_NAME=niah_single_1,CONTEXT_LEN=32768,NUM_SAMPLES=1,MAX_NEW_TOKENS=128,MODE=pagedpq_batched,APPROX_PREFILL=0,DATA_FILE_OVERRIDE=${DATA_FILE},BUDGET=4096,ONLINE_CONFIDENCE_RULE=${rule},TAIL_MODE=vpq_value,TAIL_SCORE_CALIBRATION=affine_selected,TAIL_BLEND=${tail_blend},TAIL_PROBE_REL_L2_MAX=${REL},TAIL_PROXY_MASS_MIN=0.0,TAIL_PROXY_MASS_MAX=1.0,TAIL_PQ_CORR_MIN=-1,TAIL_PQ_RELRMSE_MAX=inf,GEOMETRIC_MIN_BUDGET=4096,GEOMETRIC_MAX_BUDGET=32768,GEOMETRIC_GROWTH=1.5,GEOMETRIC_PROBE_SCALE=1.5,GEOMETRIC_BUDGET_GRANULARITY=1024,SELECTED_VALUE_MODE=${selected_value_mode},SELECTED_VALUE_EXACT_RULE=${selected_value_rule},SELECTED_VALUE_EXACT_TOP=${selected_value_top},SELECTED_VALUE_EXACT_MASS=0.0,SELECTED_VALUE_MIN_EXACT_TOP=0,SELECTED_VALUE_MAX_EXACT_TOP=0,SELECTOR_MODE=${selector},SELECTOR_BACKEND=${selector_backend},PAGE_SIZE=2048,PREFILL_CHUNK_SIZE=512,PREFILL_SELECTOR_BACKEND=native,PREFILL_TAIL_SCORE_REUSE=1,PREFILL_ATTENTION_BACKEND=native,SUBVECS=4,SUBBITS=8,VALUE_SUBVECS=1,VALUE_SUBBITS=4,VALUE_PQ_GROUP_PAGES=1,INDEX_BUILD_BACKEND=torch_gpu,NATIVE_DECODE_TAIL=1,DISABLE_NATIVE_DECODE_FUSED=1"
  local jobid
  jobid="$(sbatch --parsable \
    --job-name="${label}" \
    --output="${slurm_out}" \
    --time=02:00:00 \
    --cpus-per-task=4 \
    --mem=128000m \
    --gpus-per-node=1 \
    --partition=spgpu \
    --account=zhengya98 \
    --export="${export_args}" \
    scripts/run_ruler_pagedpq_stream_smoke_one.sh)"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${run_name}" "${jobid%%;*}" "${out_dir}" "${slurm_out//%j/${jobid%%;*}}" "${selector}" "${rule}" "${REL}" >> "${MANIFEST}"
  printf '[SUBMITTED] %s %s\n' "${run_name}" "${jobid%%;*}"
}

# Existing PQ strict result already showed near-full retrieval at REL=0.20.
# These isolate whether oracle ranking and exact-only convergence behave differently.
submit_case "oracle_strict_vpq_rel${REL/./p}" "oracle" "torch" "geometric_probe_tail_switch" "1.0" "vpq_value" "selector_rank" "256"
submit_case "oracle_exactdelta_exactv_rel${REL/./p}" "oracle" "torch" "geometric_exact_delta" "0.0" "exact" "fixed" "0"
submit_case "pq_exactdelta_exactv_rel${REL/./p}" "fullscan" "torch" "geometric_exact_delta" "0.0" "exact" "fixed" "0"

echo "[MANIFEST] ${MANIFEST}"
