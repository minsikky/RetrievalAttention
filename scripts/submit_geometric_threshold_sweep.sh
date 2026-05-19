#!/usr/bin/env bash
set -euo pipefail

cd /gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention

TAG="${TAG:-20260518}"
OUTPUT_ROOT="${OUTPUT_ROOT:-ruler_eval_result/geometric_rel_sweep_${TAG}}"
SLURM_ROOT="${SLURM_ROOT:-slurm_out/geometric_rel_sweep_${TAG}}"
MANIFEST="${MANIFEST:-notes/slurm_manifests/geometric_rel_sweep_${TAG}.tsv}"
DATA_FILE="${DATA_FILE:-ruler_eval_result/geometric_fastpath_diag_20260518/frontier_ruler_ctx32768_n1_niah_single_1_geom_fastpath/data/niah_single_1/validation.jsonl}"
THRESHOLDS="${THRESHOLDS:-0.08 0.12 0.20}"
CONFIDENCE_RULE="${CONFIDENCE_RULE:-geometric_probe_tail_switch}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "${SLURM_ROOT}" "$(dirname "${MANIFEST}")"
printf 'label\tjobid\toutput_dir\tslurm_log\ttail_probe_rel_l2_max\n' > "${MANIFEST}"

for rel in ${THRESHOLDS}; do
  rel_label="$(printf '%s' "${rel}" | tr -d '.')"
  rule_label="$(printf '%s' "${CONFIDENCE_RULE}" | tr '_' '-')"
  run_name="frontier_ruler32k_n1_${rule_label}_rel${rel_label}"
  out_dir="${OUTPUT_ROOT}/${run_name}"
  slurm_out="${SLURM_ROOT}/geom_rel${rel_label}-%j.out"
  export_args="ALL,OUTPUT_ROOT=${OUTPUT_ROOT},RUN_NAME=${run_name},TASK_NAME=niah_single_1,CONTEXT_LEN=32768,NUM_SAMPLES=1,MAX_NEW_TOKENS=128,MODE=pagedpq_batched,APPROX_PREFILL=0,DATA_FILE_OVERRIDE=${DATA_FILE},BUDGET=4096,ONLINE_CONFIDENCE_RULE=${CONFIDENCE_RULE},TAIL_MODE=vpq_value,TAIL_SCORE_CALIBRATION=affine_selected,TAIL_BLEND=1.0,TAIL_PROBE_REL_L2_MAX=${rel},TAIL_PROXY_MASS_MIN=0.0,TAIL_PROXY_MASS_MAX=1.0,TAIL_PQ_CORR_MIN=-1,TAIL_PQ_RELRMSE_MAX=inf,GEOMETRIC_MIN_BUDGET=4096,GEOMETRIC_MAX_BUDGET=32768,GEOMETRIC_GROWTH=1.5,GEOMETRIC_PROBE_SCALE=1.5,GEOMETRIC_BUDGET_GRANULARITY=1024,SELECTED_VALUE_MODE=vpq_value,SELECTED_VALUE_EXACT_RULE=selector_rank,SELECTED_VALUE_EXACT_TOP=256,SELECTED_VALUE_EXACT_MASS=0.0,SELECTED_VALUE_MIN_EXACT_TOP=0,SELECTED_VALUE_MAX_EXACT_TOP=0,SELECTOR_MODE=fullscan,SELECTOR_BACKEND=cuda_ext,PAGE_SIZE=2048,PREFILL_CHUNK_SIZE=512,PREFILL_SELECTOR_BACKEND=native,PREFILL_TAIL_SCORE_REUSE=1,PREFILL_ATTENTION_BACKEND=native,SUBVECS=4,SUBBITS=8,VALUE_SUBVECS=1,VALUE_SUBBITS=4,VALUE_PQ_GROUP_PAGES=1,INDEX_BUILD_BACKEND=torch_gpu,NATIVE_DECODE_TAIL=1,DISABLE_NATIVE_DECODE_FUSED=1"
  if [ "${DRY_RUN}" = "1" ]; then
    printf '[DRY_RUN] %s rel=%s -> %s\n' "${run_name}" "${rel}" "${out_dir}"
    printf '%s\tDRY_RUN\t%s\t%s\t%s\n' "${run_name}" "${out_dir}" "${slurm_out}" "${rel}" >> "${MANIFEST}"
    continue
  fi
  jobid="$(sbatch --parsable \
    --job-name="geom-rel${rel_label}" \
    --output="${slurm_out}" \
    --time=01:00:00 \
    --cpus-per-task=4 \
    --mem=128000m \
    --gpus-per-node=1 \
    --partition=spgpu \
    --account=zhengya98 \
    --export="${export_args}" \
    scripts/run_ruler_pagedpq_stream_smoke_one.sh)"
  printf '%s\t%s\t%s\t%s\t%s\n' "${run_name}" "${jobid}" "${out_dir}" "${slurm_out//%j/${jobid}}" "${rel}" >> "${MANIFEST}"
  printf '[SUBMITTED] %s %s rel=%s\n' "${run_name}" "${jobid}" "${rel}"
done

echo "[MANIFEST] ${MANIFEST}"
