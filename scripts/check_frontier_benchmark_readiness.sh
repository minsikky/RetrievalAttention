#!/usr/bin/env bash
set -euo pipefail

# Strict completion gate for frontier benchmark readiness.
# This intentionally fails until all smoke, unit, matrix, and drift artifacts exist.

cd /gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention

export LD_LIBRARY_PATH="/sw/pkgs/arc/python/3.10.4/lib:${LD_LIBRARY_PATH:-}"

WRAPPER_AUDIT="${WRAPPER_AUDIT:-notes/wrapper_config_audit_20260516.md}"
CUDA_AUDIT="${CUDA_AUDIT:-notes/cuda_unit_audit_20260516.md}"
SMOKE_AUDIT="${SMOKE_AUDIT:-notes/wrapper_smoke_audit_20260516.md}"
MATRIX_AUDIT="${MATRIX_AUDIT:-notes/frontier_benchmark_matrix_afterok_20260516_audit.md}"
DRIFT_REPORT="${DRIFT_REPORT:-notes/frontier_benchmark_matrix_afterok_20260516_longbench_drift.md}"

echo "===== wrapper defaults"
.venv/bin/python benchmark/audit_benchmark_wrappers.py --output "${WRAPPER_AUDIT}"

echo "===== cuda unit tests"
.venv/bin/python benchmark/audit_cuda_unit_tests.py \
  --manifest notes/slurm_manifests/frontier_cuda_unit_tests_20260516.tsv \
  --output "${CUDA_AUDIT}" \
  --strict

echo "===== wrapper smoke artifacts"
.venv/bin/python benchmark/audit_benchmark_readiness.py \
  --manifest notes/slurm_manifests/ruler_frontier_wrapper_smoke_20260516.tsv \
  --manifest notes/slurm_manifests/longbench_frontier_wrapper_smoke_20260516.tsv \
  --manifest notes/slurm_manifests/dense_wrapper_smoke_20260516.tsv \
  --output "${SMOKE_AUDIT}" \
  --strict

echo "===== benchmark matrix artifacts"
.venv/bin/python benchmark/audit_benchmark_readiness.py \
  --manifest notes/slurm_manifests/frontier_benchmark_matrix_afterok_20260516.tsv \
  --output "${MATRIX_AUDIT}" \
  --strict

echo "===== LongBench drift report"
if [ ! -s "${DRIFT_REPORT}" ]; then
  echo "missing drift report: ${DRIFT_REPORT}" >&2
  exit 1
fi
if grep -q "LongBench predictions are not ready yet" "${DRIFT_REPORT}"; then
  echo "drift report is placeholder: ${DRIFT_REPORT}" >&2
  exit 1
fi
if grep -q "n/a" "${DRIFT_REPORT}"; then
  echo "drift report still contains n/a diagnostics: ${DRIFT_REPORT}" >&2
  exit 1
fi

echo "frontier benchmark readiness gate: PASS"
