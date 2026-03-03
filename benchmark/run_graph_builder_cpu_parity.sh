#!/bin/bash
#SBATCH --job-name=graph_cpu_parity
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000m
#SBATCH --time=120:00
#SBATCH --account=zhengya98
#SBATCH --partition=standard

set -euo pipefail

module purge
module load python/3.10.4

if [ -f ".venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
else
  echo "[ERROR] .venv/bin/activate not found."
  exit 1
fi

MODE="${MODE:-write}"            # write | check
GOLDEN="${GOLDEN:-notes/graph_builder_golden_$(date +%F).npz}"
SIZES="${SIZES:-8192,16384}"
CASES_PER_SIZE="${CASES_PER_SIZE:-2}"
THREADS="${THREADS:-${SLURM_CPUS_PER_TASK:-0}}"

ROAR_M="${ROAR_M:-32}"
ROAR_L="${ROAR_L:-20}"
ROAR_ENHANCE_L="${ROAR_ENHANCE_L:-16}"
NQ="${NQ:-32}"
ENTRY="${ENTRY:-hub}"
MAX_QUERY_PER_PIVOT="${MAX_QUERY_PER_PIVOT:-0}"
DISABLE_ENHANCE="${DISABLE_ENHANCE:-0}"

echo "[INFO] MODE=${MODE}"
echo "[INFO] GOLDEN=${GOLDEN}"
echo "[INFO] SIZES=${SIZES}"
echo "[INFO] CASES_PER_SIZE=${CASES_PER_SIZE}"
echo "[INFO] THREADS=${THREADS}"
echo "[INFO] ROAR_M=${ROAR_M} ROAR_L=${ROAR_L} ROAR_ENHANCE_L=${ROAR_ENHANCE_L} NQ=${NQ} ENTRY=${ENTRY}"

EXTRA_ARGS=()
if [ "${DISABLE_ENHANCE}" = "1" ]; then
  EXTRA_ARGS+=(--disable_enhance)
fi

python -u benchmark/graph_builder_cpu_parity.py \
  --mode "${MODE}" \
  --golden "${GOLDEN}" \
  --sizes "${SIZES}" \
  --cases_per_size "${CASES_PER_SIZE}" \
  --nq "${NQ}" \
  --roar_m "${ROAR_M}" \
  --roar_l "${ROAR_L}" \
  --enhance_l "${ROAR_ENHANCE_L}" \
  --entry "${ENTRY}" \
  --max_query_per_pivot "${MAX_QUERY_PER_PIVOT}" \
  --threads "${THREADS}" \
  "${EXTRA_ARGS[@]}"
