#!/bin/bash
# Run false-negative cause diagnostics for paged+routed PQ.

#SBATCH --job-name=fn-cause
#SBATCH --partition=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%A.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
module load python/3.10.4 >/dev/null 2>&1 || true

export LD_LIBRARY_PATH="/sw/pkgs/arc/python/3.10.4/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

SOURCE_NPZ="${SOURCE_NPZ:?SOURCE_NPZ is required}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR is required}"
DECODE_TOKENS="${DECODE_TOKENS:?DECODE_TOKENS is required}"
HEADS="${HEADS:-}"
TARGET_MASS="${TARGET_MASS:-0.98}"
NPROBES="${NPROBES:-1,2,4,8,16,32,64,128,256,512}"
STATIC_PREFIX="${STATIC_PREFIX:-128}"
STATIC_SUFFIX="${STATIC_SUFFIX:-128}"
PAGED_PQ_PAGE_SIZE="${PAGED_PQ_PAGE_SIZE:-2048}"
PAGED_ROUTER_PROTOTYPES="${PAGED_ROUTER_PROTOTYPES:-16}"
PAGED_ROUTER_MERGE_REL="${PAGED_ROUTER_MERGE_REL:-0.05}"
PAGED_ROUTER_MERGE_VAR="${PAGED_ROUTER_MERGE_VAR:-0}"
PAGED_ROUTER_MAX_GROUPS="${PAGED_ROUTER_MAX_GROUPS:-512}"
BACKEND="${BACKEND:-cpp}"
BACKEND_THREADS="${BACKEND_THREADS:-8}"

args=(
  benchmark/fn_cause_diagnostic.py
  --source_npz "${SOURCE_NPZ}"
  --output_dir "${OUTPUT_DIR}"
  --decode_tokens "${DECODE_TOKENS}"
  --target_mass "${TARGET_MASS}"
  --nprobes "${NPROBES}"
  --static_prefix "${STATIC_PREFIX}"
  --static_suffix "${STATIC_SUFFIX}"
  --paged_pq_page_size "${PAGED_PQ_PAGE_SIZE}"
  --paged_router_prototypes "${PAGED_ROUTER_PROTOTYPES}"
  --paged_router_merge_rel "${PAGED_ROUTER_MERGE_REL}"
  --paged_router_merge_var "${PAGED_ROUTER_MERGE_VAR}"
  --paged_router_max_groups "${PAGED_ROUTER_MAX_GROUPS}"
  --backend "${BACKEND}"
  --backend_threads "${BACKEND_THREADS}"
)

if [[ -n "${HEADS}" ]]; then
  args+=(--heads "${HEADS}")
fi

echo "[run_fn_cause_diagnostic] source=${SOURCE_NPZ}"
echo "[run_fn_cause_diagnostic] output=${OUTPUT_DIR}"
echo "[run_fn_cause_diagnostic] decode=${DECODE_TOKENS}"
echo "[run_fn_cause_diagnostic] heads=${HEADS:-all}"

.venv/bin/python "${args[@]}"
