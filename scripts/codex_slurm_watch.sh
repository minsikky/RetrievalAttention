#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  codex_slurm_watch.sh --job-id JOB_ID [options]

Options:
  --job-id JOB_ID          Slurm job id to watch.
  --session-id ID          Codex session id to resume. If omitted, uses `--last`.
  --log-path PATH          Expected Slurm output path. Default: slurm-JOB_ID.out
  --workdir DIR            Working directory to cd into before resume. Default: current directory.
  --app-session-file PATH  App-session metadata file. If set, use app-server turn/start or turn/steer instead of codex exec resume.
  --prompt TEXT            Prompt template. Tokens: __JOB_ID__, __JOB_STATE__, __LOG_PATH__, __WORKDIR__.
  --poll-sec N             Poll interval in seconds. Default: 15.
  --resume-cmd CMD         Override resume action. Runs as `bash -lc CMD`.
  --metadata PATH          Optional metadata file to update when the watcher finishes.
EOF
}

terminal_state() {
  case "${1:-}" in
    COMPLETED|FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|PREEMPTED|BOOT_FAIL|DEADLINE|NODE_FAIL)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

job_id=""
session_id=""
log_path=""
workdir="$(pwd)"
poll_sec="${CODEX_SLURM_POLL_SEC:-15}"
prompt_template="${CODEX_SLURM_PROMPT_TEMPLATE:-Slurm job __JOB_ID__ finished with state __JOB_STATE__. Read __LOG_PATH__ in __WORKDIR__, summarize the result, and continue the experiment.}"
resume_cmd="${CODEX_RESUME_CMD:-}"
metadata_path=""
app_session_file=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --job-id)
      job_id="${2:-}"
      shift 2
      ;;
    --session-id)
      session_id="${2:-}"
      shift 2
      ;;
    --log-path)
      log_path="${2:-}"
      shift 2
      ;;
    --workdir)
      workdir="${2:-}"
      shift 2
      ;;
    --app-session-file)
      app_session_file="${2:-}"
      shift 2
      ;;
    --prompt)
      prompt_template="${2:-}"
      shift 2
      ;;
    --poll-sec)
      poll_sec="${2:-}"
      shift 2
      ;;
    --resume-cmd)
      resume_cmd="${2:-}"
      shift 2
      ;;
    --metadata)
      metadata_path="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[watch] unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${job_id}" ]]; then
  echo "[watch] --job-id is required" >&2
  exit 2
fi

if [[ -z "${log_path}" ]]; then
  log_path="slurm-${job_id}.out"
fi

mkdir -p "${workdir}/.codex/slurm"
cd "${workdir}"

query_state() {
  local state=""
  state="$(sacct -j "${job_id}" --format=JobIDRaw,State --parsable2 --noheader 2>/dev/null | awk -F'|' -v id="${job_id}" '$1==id {print $2; exit}')"
  if [[ -n "${state}" ]]; then
    state="${state%% *}"
    state="${state%%+*}"
    printf '%s\n' "${state}"
    return 0
  fi
  state="$(squeue -j "${job_id}" -h -o '%T' 2>/dev/null | head -n1 || true)"
  if [[ -n "${state}" ]]; then
    case "${state}" in
      PENDING) printf 'PENDING\n' ;;
      RUNNING|COMPLETING|CONFIGURING) printf 'RUNNING\n' ;;
      *) printf '%s\n' "${state}" ;;
    esac
    return 0
  fi
  printf '\n'
}

state=""
while true; do
  state="$(query_state)"
  if terminal_state "${state}"; then
    break
  fi
  sleep "${poll_sec}"
done

prompt="${prompt_template//__JOB_ID__/${job_id}}"
prompt="${prompt//__JOB_STATE__/${state}}"
prompt="${prompt//__LOG_PATH__/${log_path}}"
prompt="${prompt//__WORKDIR__/${workdir}}"

resume_log="${workdir}/.codex/slurm/resume-${job_id}.log"
{
  echo "[watch] job_id=${job_id}"
  echo "[watch] state=${state}"
  echo "[watch] log_path=${log_path}"
  echo "[watch] session_id=${session_id:-last}"
  echo "[watch] prompt=${prompt}"
} > "${resume_log}"

if [[ -n "${metadata_path}" ]]; then
  {
    echo "state=${state}"
    echo "finished_at=$(date --iso-8601=seconds)"
  } >> "${metadata_path}"
fi

if [[ -n "${resume_cmd}" ]]; then
  export JOB_ID="${job_id}"
  export JOB_STATE="${state}"
  export JOB_LOG_PATH="${log_path}"
  export CODEX_SESSION_ID="${session_id}"
  export CODEX_RESUME_PROMPT="${prompt}"
  bash -lc "${resume_cmd}" >> "${resume_log}" 2>&1
  exit $?
fi

if [[ -n "${app_session_file}" ]]; then
  node scripts/codex_app_session_ctl.mjs send \
    --session-file "${app_session_file}" \
    --message "${prompt}" \
    --output-file "${workdir}/.codex/slurm/app-send-${job_id}.json" \
    >> "${resume_log}" 2>&1
  exit $?
fi

if [[ -n "${session_id}" ]]; then
  codex exec resume "${session_id}" "${prompt}" >> "${resume_log}" 2>&1
else
  codex exec resume --last "${prompt}" >> "${resume_log}" 2>&1
fi
