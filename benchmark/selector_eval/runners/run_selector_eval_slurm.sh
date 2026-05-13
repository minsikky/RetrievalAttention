#!/bin/bash
# Slurm entrypoint for selector-eval sweeps.

#SBATCH --job-name=selector-eval
#SBATCH --partition=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%A.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

TRACE="${TRACE:?TRACE is required}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR is required}"

bash benchmark/selector_eval/runners/run_selector_eval.sh

if [[ "${PLOT:-1}" == "1" ]]; then
  LD_LIBRARY_PATH="/sw/pkgs/arc/python/3.10.4/lib:${LD_LIBRARY_PATH:-}" \
    .venv/bin/python benchmark/selector_eval/reports/plot_summary.py \
      --summary_csv "${OUTPUT_DIR}/summary.csv" \
      --output_dir "${OUTPUT_DIR}/plots"
fi

