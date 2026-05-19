#!/usr/bin/env bash
set -euo pipefail

cd /gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention

tag="${TAG:-20260519}"
output_root="${OUTPUT_ROOT:-attention_efficiency_result/geometric_rank_attribution_${tag}}"
slurm_root="${SLURM_ROOT:-slurm_out/geometric_rank_attribution_${tag}}"
manifest="${MANIFEST:-notes/slurm_manifests/geometric_rank_attribution_${tag}.tsv}"

mkdir -p "${output_root}" "${slurm_root}" "$(dirname "${manifest}")"
printf 'label\tjobid\toutput_dir\tslurm_log\tselector\trule\tthreshold\n' > "${manifest}"

submit_case() {
  local label="$1"
  local selector="$2"
  local rule="$3"
  local threshold="$4"
  local selected_value_mode="$5"
  local selected_value_rule="$6"
  local selected_value_top="$7"
  local tail_mode="$8"
  local tail_blend="$9"
  local out_dir="${output_root}/${label}"
  local log_path="${slurm_root}/${label}-%j.out"
  local export_args
  export_args="ALL,OUTPUT_ROOT=${output_root},RUN_NAME=${label},SELECTOR_MODE=${selector},ONLINE_CONFIDENCE_RULE=${rule},TAIL_PROBE_REL_L2_MAX=${threshold},EXACT_DELTA_REL_L2_MAX=${threshold},SELECTED_VALUE_MODE=${selected_value_mode},SELECTED_VALUE_EXACT_RULE=${selected_value_rule},SELECTED_VALUE_EXACT_TOP=${selected_value_top},TAIL_MODE=${tail_mode},TAIL_BLEND=${tail_blend},GEOMETRIC_MIN_BUDGET=4096,GEOMETRIC_MAX_BUDGET=32768,GEOMETRIC_GROWTH=1.5,GEOMETRIC_PROBE_SCALE=1.5,GEOMETRIC_BUDGET_GRANULARITY=1024,PAGE_SIZE=2048,SUBVECS=4,SUBBITS=8,VALUE_SUBVECS=1,VALUE_SUBBITS=4,NPROBES=512,DECODE_LENGTHS=32000,MAX_QIDX_PER_DECODE=1,DEVICE=cuda"
  local jobid
  jobid="$(sbatch --parsable --job-name="${label}" --output="${log_path}" --export="${export_args}" scripts/run_geometric_rank_attribution_one.sh)"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${label}" "${jobid%%;*}" "${out_dir}" "${log_path//%j/${jobid%%;*}}" "${selector}" "${rule}" "${threshold}" >> "${manifest}"
  echo "${label} ${jobid%%;*}"
}

for threshold in 0.020 0.200; do
  short="${threshold/./p}"
  submit_case "pq_strict_vpq_rel${short}" "fullscan" "geometric_probe_tail_switch" "${threshold}" "vpq_value" "selector_rank" "256" "vpq_value" "1.0"
  submit_case "oracle_strict_vpq_rel${short}" "oracle" "geometric_probe_tail_switch" "${threshold}" "vpq_value" "selector_rank" "256" "vpq_value" "1.0"
  submit_case "pq_exactdelta_exactv_rel${short}" "fullscan" "geometric_exact_delta" "${threshold}" "exact" "fixed" "0" "vpq_value" "0.0"
  submit_case "oracle_exactdelta_exactv_rel${short}" "oracle" "geometric_exact_delta" "${threshold}" "exact" "fixed" "0" "vpq_value" "0.0"
done

echo "manifest=${manifest}"
