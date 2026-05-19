#!/usr/bin/env bash
set -euo pipefail

# Generate audit and LongBench drift reports for a completed dense/frontier matrix.

cd /gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention

export LD_LIBRARY_PATH="/sw/pkgs/arc/python/3.10.4/lib:${LD_LIBRARY_PATH:-}"

MANIFEST="${MANIFEST:-notes/slurm_manifests/frontier_benchmark_matrix_afterok_20260516.tsv}"
OUT_PREFIX="${OUT_PREFIX:-notes/frontier_benchmark_matrix_afterok_20260516}"
DIAG_GLOBS="${DIAG_GLOBS:-longbench_v2_hf_result/frontier_readiness_20260516_diag_fulltail64_temp0_*/summary.json longbench_v2_hf_result/frontier_readiness_20260516_diag_matrix_afterok_20260516_*/summary.json}"

mkdir -p "$(dirname "${OUT_PREFIX}")"

AUDIT_OUT="${OUT_PREFIX}_audit.md"
COMPARE_OUT="${OUT_PREFIX}_longbench_compare.txt"
DRIFT_OUT="${OUT_PREFIX}_longbench_drift.md"

echo "===== benchmark artifact audit ${AUDIT_OUT}"
.venv/bin/python benchmark/audit_benchmark_readiness.py \
  --manifest "${MANIFEST}" \
  --output "${AUDIT_OUT}"

dense_lb_dir="$(awk -F'\t' 'NR>1 && $1 ~ /^dense_lbv2_/ {print $3; exit}' "${MANIFEST}")"
frontier_lb_dir="$(awk -F'\t' 'NR>1 && $1 ~ /^frontier_lbv2_/ {print $3; exit}' "${MANIFEST}")"

if [ -n "${dense_lb_dir}" ] && [ -n "${frontier_lb_dir}" ] \
  && [ -s "${dense_lb_dir}/predictions.jsonl" ] \
  && [ -s "${frontier_lb_dir}/predictions.jsonl" ]; then
  echo "===== LongBench comparison ${COMPARE_OUT}"
  .venv/bin/python benchmark/compare_longbench_runs.py \
    --run "dense:${dense_lb_dir}" \
    --run "frontier:${frontier_lb_dir}" \
    > "${COMPARE_OUT}"
  cat "${COMPARE_OUT}"

  echo "===== LongBench drift ${DRIFT_OUT}"
  diag_args=()
  for pattern in ${DIAG_GLOBS}; do
    diag_args+=(--diag-glob "${pattern}")
  done
  .venv/bin/python benchmark/report_longbench_drift.py \
    --dense "${dense_lb_dir}" \
    --frontier "${frontier_lb_dir}" \
    "${diag_args[@]}" \
    --changed-only \
    --output "${DRIFT_OUT}"
else
  {
    echo "LongBench predictions are not ready yet."
    echo "dense=${dense_lb_dir:-missing}"
    echo "frontier=${frontier_lb_dir:-missing}"
  } | tee "${COMPARE_OUT}"
  {
    echo "# LongBench Drift Report"
    echo
    echo "LongBench predictions are not ready yet."
    echo
    echo "- dense: \`${dense_lb_dir:-missing}\`"
    echo "- frontier: \`${frontier_lb_dir:-missing}\`"
  } > "${DRIFT_OUT}"
fi

echo "===== outputs"
printf '%s\n' "${AUDIT_OUT}" "${COMPARE_OUT}" "${DRIFT_OUT}"
