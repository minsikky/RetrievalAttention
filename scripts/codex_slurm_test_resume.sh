#!/usr/bin/env bash
set -euo pipefail

mkdir -p .codex/slurm
printf '%s|%s|%s|%s\n' \
  "${JOB_ID:-}" \
  "${JOB_STATE:-}" \
  "${JOB_LOG_PATH:-}" \
  "${CODEX_SESSION_ID:-}" \
  > .codex/slurm/test_resume_output.txt
