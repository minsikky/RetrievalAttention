#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/slurm_race_submit.sh [options] -- <sbatch-script>

Options:
  --partitions CSV      Partitions to race. Default: spgpu,gpu_mig40,gpu-rtx6000
  --safe CSV            Partitions safe to trust once RUNNING. Default: spgpu,gpu_mig40
  --risky CSV           Partitions that need readiness before canceling safe jobs. Default: gpu-rtx6000
  --tag TAG             Tag for manifest/log/output suffixes. Default: timestamp
  --manifest PATH       Manifest path. Default: /tmp/slurm_race_TAG.tsv
  --output-base DIR     Base output directory for per-partition OUTPUT_DIR suffixes.
  --job-name NAME       Override Slurm job name prefix.
  --export VARS         Extra sbatch --export payload after ALL,PARTITION vars.
  --time HH:MM:SS       Optional sbatch time override.
  --mem MEM             Optional sbatch mem override.
  --gpus N              Optional --gpus-per-node override. Default: script default.
  --watch               Start detached race watcher.
  --poll-sec N          Watcher poll interval. Default: 20.
  --ready-file NAME     Relative readiness marker under output dir. Default: hf_job_ready
  --winner-file PATH    Winner file. Default: /tmp/slurm_race_TAG.winner

Example:
  scripts/slurm_race_submit.sh --output-base generated_memory_hf_eval_result/run1 \
    --export 'MODEL_NAME=...,HF_ATTENTION_MODE=graph_topk_roar' \
    --watch -- benchmark/run_generated_memory_hf.sh
EOF
}

csv_has() {
  local csv="$1"
  local value="$2"
  IFS=',' read -r -a items <<< "${csv}"
  for item in "${items[@]}"; do
    if [[ "${item}" == "${value}" ]]; then
      return 0
    fi
  done
  return 1
}

partitions="${SLURM_RACE_PARTITIONS:-spgpu,gpu_mig40,gpu-rtx6000}"
safe_partitions="${SLURM_RACE_SAFE_PARTITIONS:-spgpu,gpu_mig40}"
risky_partitions="${SLURM_RACE_RISKY_PARTITIONS:-gpu-rtx6000}"
tag="${SLURM_RACE_TAG:-$(date +%Y%m%d_%H%M%S)}"
manifest=""
output_base=""
job_name="race"
extra_export=""
time_override=""
mem_override=""
gpus_override=""
watch=0
poll_sec="${SLURM_RACE_POLL_SEC:-20}"
ready_file="${SLURM_RACE_READY_FILE:-hf_job_ready}"
winner_file=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --partitions) partitions="${2:-}"; shift 2 ;;
    --safe) safe_partitions="${2:-}"; shift 2 ;;
    --risky) risky_partitions="${2:-}"; shift 2 ;;
    --tag) tag="${2:-}"; shift 2 ;;
    --manifest) manifest="${2:-}"; shift 2 ;;
    --output-base) output_base="${2:-}"; shift 2 ;;
    --job-name) job_name="${2:-}"; shift 2 ;;
    --export) extra_export="${2:-}"; shift 2 ;;
    --time) time_override="${2:-}"; shift 2 ;;
    --mem) mem_override="${2:-}"; shift 2 ;;
    --gpus) gpus_override="${2:-}"; shift 2 ;;
    --watch) watch=1; shift ;;
    --poll-sec) poll_sec="${2:-}"; shift 2 ;;
    --ready-file) ready_file="${2:-}"; shift 2 ;;
    --winner-file) winner_file="${2:-}"; shift 2 ;;
    --) shift; break ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[race-submit] unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ $# -ne 1 ]]; then
  echo "[race-submit] expected exactly one sbatch script after --" >&2
  usage >&2
  exit 2
fi

sbatch_script="$1"
if [[ ! -f "${sbatch_script}" ]]; then
  echo "[race-submit] sbatch script not found: ${sbatch_script}" >&2
  exit 2
fi

if [[ -z "${manifest}" ]]; then
  manifest="/tmp/slurm_race_${tag}.tsv"
fi
if [[ -z "${winner_file}" ]]; then
  winner_file="/tmp/slurm_race_${tag}.winner"
fi
if [[ -z "${output_base}" ]]; then
  output_base="slurm_race_result/${tag}"
fi

mkdir -p "$(dirname "${manifest}")" "$(dirname "${winner_file}")" .codex/slurm
printf 'label\tjobid\tpartition\trisk\toutput_dir\tready_file\tlog_path\n' > "${manifest}"

IFS=',' read -r -a partition_arr <<< "${partitions}"
for partition in "${partition_arr[@]}"; do
  if [[ -z "${partition}" ]]; then
    continue
  fi
  label="${partition//[^A-Za-z0-9_]/_}"
  output_dir="${output_base}/${label}"
  risk="safe"
  if csv_has "${risky_partitions}" "${partition}"; then
    risk="risky"
  elif ! csv_has "${safe_partitions}" "${partition}"; then
    risk="unknown"
  fi
  export_payload="ALL,SLURM_RACE_TAG=${tag},SLURM_RACE_PARTITION=${partition},OUTPUT_DIR=${output_dir}"
  if [[ -n "${extra_export}" ]]; then
    export_payload="${export_payload},${extra_export}"
  fi
  sbatch_args=(
    --parsable
    --partition="${partition}"
    --job-name="${job_name}-${label}"
    --export="${export_payload}"
  )
  if [[ -n "${time_override}" ]]; then
    sbatch_args+=(--time="${time_override}")
  fi
  if [[ -n "${mem_override}" ]]; then
    sbatch_args+=(--mem="${mem_override}")
  fi
  if [[ -n "${gpus_override}" ]]; then
    sbatch_args+=(--gpus-per-node="${gpus_override}")
  fi
  job_info="$(sbatch "${sbatch_args[@]}" "${sbatch_script}")"
  jobid="${job_info%%;*}"
  log_path="slurm-${jobid}.out"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${label}" "${jobid}" "${partition}" "${risk}" "${output_dir}" "${output_dir}/${ready_file}" "${log_path}" >> "${manifest}"
  printf '[race-submit] partition=%s risk=%s job=%s output=%s\n' "${partition}" "${risk}" "${jobid}" "${output_dir}"
done

printf '[race-submit] manifest=%s\n' "${manifest}"
printf '[race-submit] winner_file=%s\n' "${winner_file}"

if [[ "${watch}" -eq 1 ]]; then
  watcher_log=".codex/slurm/race-watch-${tag}.log"
  watcher_cmd=(
    bash scripts/slurm_race_watch.sh
    --manifest "${manifest}"
    --winner-file "${winner_file}"
    --poll-sec "${poll_sec}"
  )
  if command -v setsid >/dev/null 2>&1; then
    setsid "${watcher_cmd[@]}" > "${watcher_log}" 2>&1 < /dev/null &
  else
    nohup "${watcher_cmd[@]}" > "${watcher_log}" 2>&1 < /dev/null &
  fi
  printf '[race-submit] watcher_log=%s\n' "${watcher_log}"
  printf '[race-submit] watcher_pid=%s\n' "$!"
fi
