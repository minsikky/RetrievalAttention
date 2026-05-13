#!/usr/bin/env bash
set -euo pipefail

mkdir -p .codex/slurm

job_id="${JOB_ID:?JOB_ID is required}"
session_id="${CODEX_SESSION_ID:?CODEX_SESSION_ID is required}"
out_file=".codex/slurm/real_resume_last_${job_id}.txt"

codex exec resume "${session_id}" \
  "Resume test for job ${job_id}. Reply with exactly RESUME_OK_${job_id} and nothing else." \
  -o "${out_file}"
