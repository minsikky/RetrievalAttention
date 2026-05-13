#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  codex_slurm_submit.sh [options] -- <sbatch args...>

Options:
  --session-id ID          Codex session id to resume. Default: latest from ~/.codex/history.jsonl.
  --app-session-file PATH  Route job-completion prompts to an app-managed Codex thread.
  --log-path PATH          Expected Slurm log path. Default: slurm-JOB_ID.out
  --prompt TEXT            Prompt template. Tokens: __JOB_ID__, __JOB_STATE__, __LOG_PATH__, __WORKDIR__.
  --poll-sec N             Watcher poll interval in seconds. Default: 15.
  --resume-cmd CMD         Override resume action for testing.
  --watcher-log PATH       Path for the detached watcher stdout/stderr.
  --metadata PATH          Metadata file path. Default: .codex/slurm/JOB_ID.env

Example:
  scripts/codex_slurm_submit.sh -- --wrap 'sleep 5; echo done'
EOF
}

infer_session_id() {
  python3 - <<'PY'
import json
from pathlib import Path
path = Path.home() / ".codex" / "history.jsonl"
last = ""
if path.exists():
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        sid = str(obj.get("session_id", "")).strip()
        if sid:
            last = sid
print(last)
PY
}

session_id="${CODEX_SESSION_ID:-}"
log_path=""
poll_sec="${CODEX_SLURM_POLL_SEC:-15}"
prompt_template="${CODEX_SLURM_PROMPT_TEMPLATE:-Slurm job __JOB_ID__ finished with state __JOB_STATE__. Read __LOG_PATH__ in __WORKDIR__, summarize the result, and continue the experiment.}"
resume_cmd="${CODEX_RESUME_CMD:-}"
watcher_log=""
metadata_path=""
app_session_file=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --session-id)
      session_id="${2:-}"
      shift 2
      ;;
    --log-path)
      log_path="${2:-}"
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
    --watcher-log)
      watcher_log="${2:-}"
      shift 2
      ;;
    --metadata)
      metadata_path="${2:-}"
      shift 2
      ;;
    --)
      shift
      break
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[submit] unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ $# -eq 0 ]]; then
  echo "[submit] missing sbatch arguments after --" >&2
  usage >&2
  exit 2
fi

if [[ -z "${session_id}" ]]; then
  session_id="$(infer_session_id)"
fi

mkdir -p .codex/slurm

job_info="$(sbatch --parsable "$@")"
job_id="${job_info%%;*}"

if [[ -z "${job_id}" ]]; then
  echo "[submit] failed to parse sbatch output: ${job_info}" >&2
  exit 1
fi

if [[ -z "${log_path}" ]]; then
  log_path="slurm-${job_id}.out"
fi
if [[ -z "${metadata_path}" ]]; then
  metadata_path=".codex/slurm/${job_id}.env"
fi
if [[ -z "${watcher_log}" ]]; then
  watcher_log=".codex/slurm/watch-${job_id}.log"
fi

cat > "${metadata_path}" <<EOF
job_id=${job_id}
job_info=${job_info}
session_id=${session_id}
log_path=${log_path}
workdir=$(pwd)
watcher_log=${watcher_log}
poll_sec=${poll_sec}
submitted_at=$(date --iso-8601=seconds)
EOF

watcher_cmd=(
  bash scripts/codex_slurm_watch.sh
  --job-id "${job_id}"
  --session-id "${session_id}"
  --log-path "${log_path}"
  --workdir "$(pwd)"
  --prompt "${prompt_template}"
  --poll-sec "${poll_sec}"
  --metadata "${metadata_path}"
)
if [[ -n "${app_session_file}" ]]; then
  watcher_cmd+=(--app-session-file "${app_session_file}")
fi
if [[ -n "${resume_cmd}" ]]; then
  watcher_cmd+=(--resume-cmd "${resume_cmd}")
fi

if command -v setsid >/dev/null 2>&1; then
  setsid "${watcher_cmd[@]}" > "${watcher_log}" 2>&1 < /dev/null &
else
  nohup "${watcher_cmd[@]}" > "${watcher_log}" 2>&1 < /dev/null &
fi

watcher_pid=$!

{
  echo "watcher_pid=${watcher_pid}"
  echo "watcher_started_at=$(date --iso-8601=seconds)"
} >> "${metadata_path}"

echo "job_id=${job_id}"
echo "job_info=${job_info}"
echo "session_id=${session_id:-last}"
echo "log_path=${log_path}"
echo "metadata=${metadata_path}"
echo "watcher_log=${watcher_log}"
echo "watcher_pid=${watcher_pid}"
