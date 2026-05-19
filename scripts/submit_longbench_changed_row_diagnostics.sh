#!/usr/bin/env bash
set -euo pipefail

# Submit dense-reference diagnostics for changed LongBench-v2 rows in a dense/frontier pair.

cd /gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention

MATRIX_MANIFEST="${MATRIX_MANIFEST:-notes/slurm_manifests/frontier_benchmark_matrix_afterok_20260516.tsv}"
TAG="${TAG:-matrix_afterok_20260516}"
MANIFEST="${MANIFEST:-notes/slurm_manifests/longbench_changed_rows_diag_${TAG}.tsv}"
SLURM_ROOT="${SLURM_ROOT:-slurm_out/frontier_readiness_20260516}"
DIAG_ROOT="${DIAG_ROOT:-longbench_v2_hf_result/frontier_readiness_20260516_diag_${TAG}}"
DEPENDENCY="${DEPENDENCY:-}"
MAX_ROWS="${MAX_ROWS:-0}"
DRY_RUN="${DRY_RUN:-0}"

dense_lb_dir="$(awk -F'\t' 'NR>1 && $1 ~ /^dense_lbv2_/ {print $3; exit}' "${MATRIX_MANIFEST}")"
frontier_lb_dir="$(awk -F'\t' 'NR>1 && $1 ~ /^frontier_lbv2_/ {print $3; exit}' "${MATRIX_MANIFEST}")"

if [ -z "${dense_lb_dir}" ] || [ -z "${frontier_lb_dir}" ]; then
  echo "Could not find dense/frontier LongBench rows in ${MATRIX_MANIFEST}" >&2
  exit 1
fi
if [ ! -s "${dense_lb_dir}/predictions.jsonl" ] || [ ! -s "${frontier_lb_dir}/predictions.jsonl" ]; then
  echo "LongBench predictions are not ready yet." >&2
  echo "dense=${dense_lb_dir}" >&2
  echo "frontier=${frontier_lb_dir}" >&2
  exit 1
fi

mkdir -p "$(dirname "${MANIFEST}")" "${SLURM_ROOT}"
tmp_ids="$(mktemp)"
trap 'rm -f "${tmp_ids}"' EXIT

.venv/bin/python - <<PY > "${tmp_ids}"
import json
from pathlib import Path

dense_path = Path("${dense_lb_dir}") / "predictions.jsonl"
frontier_path = Path("${frontier_lb_dir}") / "predictions.jsonl"

def load(path):
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        row_id = str(row.get("_id", row.get("id")))
        out[row_id] = row
    return out

dense = load(dense_path)
frontier = load(frontier_path)
ids = []
for row_id, d in dense.items():
    f = frontier.get(row_id)
    if f is None:
        continue
    if d.get("pred") != f.get("pred") or bool(d.get("judge")) != bool(f.get("judge")):
        ids.append(row_id)
limit = int("${MAX_ROWS}")
if limit > 0:
    ids = ids[:limit]
for row_id in ids:
    print(row_id)
PY

printf 'label\tjobid\toutput_dir\tslurm_out\n' > "${MANIFEST}"
count=0
while IFS= read -r row_id; do
  [ -n "${row_id}" ] || continue
  out_dir="${DIAG_ROOT}_${row_id}"
  slurm_out="${SLURM_ROOT}/lbdiag_${row_id}-%j.out"
  label="diag_${TAG}_${row_id}"
  if [ -s "${out_dir}/summary.json" ]; then
    printf '%s\tEXISTS\t%s\t%s\n' "${label}" "${out_dir}" "${slurm_out}" >> "${MANIFEST}"
    continue
  fi
  if [ "${DRY_RUN}" = "1" ]; then
    printf '[DRY_RUN] %s -> %s dependency=%s\n' "${label}" "${out_dir}" "${DEPENDENCY:-none}"
    printf '%s\tDRY_RUN\t%s\t%s\n' "${label}" "${out_dir}" "${slurm_out}" >> "${MANIFEST}"
    count=$((count + 1))
    continue
  fi
  dep_args=()
  if [ -n "${DEPENDENCY}" ]; then
    dep_args=(--dependency="${DEPENDENCY}")
  fi
  jobid=$(sbatch --parsable \
    --job-name="${label:0:40}" \
    --output="${slurm_out}" \
    "${dep_args[@]}" \
    --export="ALL,OUTPUT_DIR=${out_dir},MAX_EXAMPLES=1,LENGTH_FILTER=short,DIFFICULTY_FILTER=easy,ID_FILTER=${row_id},MAX_INPUT_TOKENS=8192,TEMPERATURE=0.0,DIAGNOSE_DENSE_REFERENCE=1" \
    scripts/run_frontier_longbench_v2_one.sh)
  printf '%s\t%s\t%s\t%s\n' "${label}" "${jobid}" "${out_dir}" "${slurm_out//%j/${jobid}}" >> "${MANIFEST}"
  printf '[SUBMITTED] %s %s\n' "${label}" "${jobid}"
  count=$((count + 1))
done < "${tmp_ids}"

echo "[MANIFEST] ${MANIFEST}"
echo "[ROWS] ${count}"
