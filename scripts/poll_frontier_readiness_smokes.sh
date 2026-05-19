#!/usr/bin/env bash
set -euo pipefail

cd /gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention

export LD_LIBRARY_PATH="/sw/pkgs/arc/python/3.10.4/lib:${LD_LIBRARY_PATH:-}"

POLL="/home/minsikky/.codex/skills/slurm-submit-wait/scripts/poll_batch.py"
OUT="${OUT:-notes/wrapper_smoke_audit_20260516.md}"
WRAPPER_OUT="${WRAPPER_OUT:-notes/wrapper_config_audit_20260516.md}"

manifests=(
  "notes/slurm_manifests/ruler_frontier_wrapper_smoke_20260516.tsv"
  "notes/slurm_manifests/longbench_frontier_wrapper_smoke_20260516.tsv"
  "notes/slurm_manifests/dense_wrapper_smoke_20260516.tsv"
)

matrix_manifests=(
  "notes/slurm_manifests/frontier_benchmark_matrix_afterok_20260516.tsv"
)

diag_manifests=(
  "notes/slurm_manifests/longbench_missing_changed_rows_diag_20260516.tsv"
)

unit_manifests=(
  "notes/slurm_manifests/frontier_cuda_unit_tests_20260516.tsv"
)

echo "===== wrapper config audit ${WRAPPER_OUT}"
.venv/bin/python benchmark/audit_benchmark_wrappers.py --output "${WRAPPER_OUT}"

for manifest in "${unit_manifests[@]}"; do
  if [ -f "${manifest}" ]; then
    state_file="${manifest%.tsv}.state.json"
    echo "===== ${manifest}"
    .venv/bin/python "${POLL}" "${manifest}" --state-file "${state_file}" --all-terminal
    echo "===== cuda unit audit ${manifest}"
    .venv/bin/python benchmark/audit_cuda_unit_tests.py \
      --manifest "${manifest}" \
      --output notes/cuda_unit_audit_20260516.md
  fi
done

for manifest in "${manifests[@]}"; do
  state_file="${manifest%.tsv}.state.json"
  echo "===== ${manifest}"
  .venv/bin/python "${POLL}" "${manifest}" --state-file "${state_file}" --all-terminal
done

audit_args=()
for manifest in "${manifests[@]}"; do
  audit_args+=(--manifest "${manifest}")
done

.venv/bin/python benchmark/audit_benchmark_readiness.py \
  "${audit_args[@]}" \
  --output "${OUT}"

echo "===== audit ${OUT}"
cat "${OUT}"

for manifest in "${matrix_manifests[@]}"; do
  if [ -f "${manifest}" ]; then
    state_file="${manifest%.tsv}.state.json"
    echo "===== ${manifest}"
    .venv/bin/python "${POLL}" "${manifest}" --state-file "${state_file}" --all-terminal
  fi
done

for manifest in "${diag_manifests[@]}"; do
  if [ -f "${manifest}" ]; then
    state_file="${manifest%.tsv}.state.json"
    echo "===== ${manifest}"
    .venv/bin/python "${POLL}" "${manifest}" --state-file "${state_file}" --all-terminal
  fi
done
