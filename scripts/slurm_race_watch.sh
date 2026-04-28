#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/slurm_race_watch.sh --manifest PATH [options]

Options:
  --manifest PATH       Race manifest from slurm_race_submit.sh.
  --winner-file PATH    File to write chosen job metadata.
  --poll-sec N          Poll interval in seconds. Default: 20.
  --dry-run             Do not cancel losers.

Policy:
  - Safe partition RUNNING wins immediately.
  - Risky/unknown partition RUNNING wins only after its ready_file exists.
  - When a winner is selected, all other non-terminal jobs are canceled.
EOF
}

manifest=""
winner_file=""
poll_sec="${SLURM_RACE_POLL_SEC:-20}"
dry_run=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest) manifest="${2:-}"; shift 2 ;;
    --winner-file) winner_file="${2:-}"; shift 2 ;;
    --poll-sec) poll_sec="${2:-}"; shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[race-watch] unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "${manifest}" ]]; then
  echo "[race-watch] --manifest is required" >&2
  exit 2
fi
if [[ ! -f "${manifest}" ]]; then
  echo "[race-watch] manifest not found: ${manifest}" >&2
  exit 2
fi
if [[ -z "${winner_file}" ]]; then
  winner_file="${manifest%.tsv}.winner"
fi

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

query_state() {
  local jobid="$1"
  local state=""
  state="$(squeue -j "${jobid}" -h -o '%T' 2>/dev/null | head -n1 || true)"
  if [[ -n "${state}" ]]; then
    case "${state}" in
      RUNNING|COMPLETING|CONFIGURING) printf 'RUNNING\n' ;;
      *) printf '%s\n' "${state}" ;;
    esac
    return 0
  fi
  state="$(sacct -j "${jobid}" --format=JobIDRaw,State --parsable2 --noheader 2>/dev/null | awk -F'|' -v id="${jobid}" '$1==id {print $2; exit}')"
  state="${state%% *}"
  state="${state%%+*}"
  printf '%s\n' "${state}"
}

choose_winner() {
  local line label jobid partition risk output_dir ready_file log_path state
  tail -n +2 "${manifest}" | while IFS=$'\t' read -r label jobid partition risk output_dir ready_file log_path; do
    [[ -z "${jobid}" ]] && continue
    state="$(query_state "${jobid}")"
    printf '[race-watch] job=%s partition=%s risk=%s state=%s ready=%s\n' \
      "${jobid}" "${partition}" "${risk}" "${state:-UNKNOWN}" "$([[ -f "${ready_file}" ]] && printf yes || printf no)" >&2
    if [[ "${state}" != "RUNNING" ]]; then
      continue
    fi
    if [[ "${risk}" == "safe" || -f "${ready_file}" ]]; then
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${label}" "${jobid}" "${partition}" "${risk}" "${output_dir}" "${ready_file}" "${log_path}" "${state}"
      return 0
    fi
  done
}

all_terminal_or_gone() {
  local label jobid partition risk output_dir ready_file log_path state
  while IFS=$'\t' read -r label jobid partition risk output_dir ready_file log_path; do
    [[ "${label}" == "label" || -z "${jobid}" ]] && continue
    state="$(query_state "${jobid}")"
    if ! terminal_state "${state}"; then
      return 1
    fi
  done < "${manifest}"
  return 0
}

winner=""
while true; do
  winner="$(choose_winner || true)"
  if [[ -n "${winner}" ]]; then
    break
  fi
  if all_terminal_or_gone; then
    {
      echo "winner="
      echo "state=no_running_winner"
      echo "decided_at=$(date --iso-8601=seconds)"
    } > "${winner_file}"
    echo "[race-watch] no running winner; all jobs terminal"
    exit 1
  fi
  sleep "${poll_sec}"
done

IFS=$'\t' read -r win_label win_jobid win_partition win_risk win_output_dir win_ready_file win_log_path win_state <<< "${winner}"
{
  echo "winner_label=${win_label}"
  echo "winner_jobid=${win_jobid}"
  echo "winner_partition=${win_partition}"
  echo "winner_risk=${win_risk}"
  echo "winner_output_dir=${win_output_dir}"
  echo "winner_ready_file=${win_ready_file}"
  echo "winner_log_path=${win_log_path}"
  echo "winner_state=${win_state}"
  echo "decided_at=$(date --iso-8601=seconds)"
} > "${winner_file}"

echo "[race-watch] winner job=${win_jobid} partition=${win_partition} risk=${win_risk}"

while IFS=$'\t' read -r label jobid partition risk output_dir ready_file log_path; do
  [[ "${label}" == "label" || -z "${jobid}" ]] && continue
  if [[ "${jobid}" == "${win_jobid}" ]]; then
    continue
  fi
  state="$(query_state "${jobid}")"
  if terminal_state "${state}"; then
    continue
  fi
  echo "[race-watch] cancel loser job=${jobid} partition=${partition} state=${state:-UNKNOWN}"
  if [[ "${dry_run}" -eq 0 ]]; then
    scancel "${jobid}" || true
  fi
done < "${manifest}"
