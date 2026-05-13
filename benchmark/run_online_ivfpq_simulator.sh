#!/bin/bash
# Run the true-online IVF-PQ selector simulator.

#SBATCH --job-name=online-ivfpq
#SBATCH --partition=standard
#SBATCH --cpus-per-task=1
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%A.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
module load python/3.10.4 >/dev/null 2>&1 || true

export LD_LIBRARY_PATH="/sw/pkgs/arc/python/3.10.4/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

SOURCE_NPZ="${SOURCE_NPZ:?SOURCE_NPZ is required}"
OUTPUT_DIR="${OUTPUT_DIR:-attention_efficiency_result/online_ivfpq_simulator_v1}"
POLICIES="${POLICIES:-frozen_append,online_centroid,periodic_rebuild}"
DECODE_FILTER="${DECODE_FILTER:-}"
NUM_QUERIES="${NUM_QUERIES:-0}"
QUERY_SELECTION="${QUERY_SELECTION:-all}"
MASS_TARGETS="${MASS_TARGETS:-0.95,0.98}"
STATIC_PREFIX="${STATIC_PREFIX:-128}"
STATIC_SUFFIX="${STATIC_SUFFIX:-128}"
IVFPQ_NPROBES="${IVFPQ_NPROBES:-1,2,4,8,16,32,64,128}"
IVFPQ_FINAL_KS="${IVFPQ_FINAL_KS:-512,1024,2048,4096,8192,16384,32768,65536}"
SKIP_FIXEDK="${SKIP_FIXEDK:-0}"
IVFPQ_REBUILD_INTERVAL="${IVFPQ_REBUILD_INTERVAL:-8192}"
PAGED_PQ_PAGE_SIZE="${PAGED_PQ_PAGE_SIZE:-0}"
PAGED_ROUTER_PROTOTYPES="${PAGED_ROUTER_PROTOTYPES:-16}"
PAGED_ROUTER_MERGE_REL="${PAGED_ROUTER_MERGE_REL:-0.5}"
PAGED_ROUTER_MERGE_VAR="${PAGED_ROUTER_MERGE_VAR:-0}"
PAGED_ROUTER_MAX_GROUPS="${PAGED_ROUTER_MAX_GROUPS:-0}"
PROGRESS_EVERY="${PROGRESS_EVERY:-8}"
BACKEND="${BACKEND:-auto}"
BACKEND_THREADS="${BACKEND_THREADS:-0}"
COMPUTE_OUTPUT_COS="${COMPUTE_OUTPUT_COS:-1}"

args=(
  benchmark/online_ivfpq_simulator.py
  --source_npz "${SOURCE_NPZ}"
  --output_dir "${OUTPUT_DIR}"
  --policies "${POLICIES}"
  --mass_targets "${MASS_TARGETS}"
  --num_queries "${NUM_QUERIES}"
  --query_selection "${QUERY_SELECTION}"
  --static_prefix "${STATIC_PREFIX}"
  --static_suffix "${STATIC_SUFFIX}"
  --ivfpq_nprobes "${IVFPQ_NPROBES}"
  --ivfpq_final_ks "${IVFPQ_FINAL_KS}"
  --ivfpq_rebuild_interval "${IVFPQ_REBUILD_INTERVAL}"
  --paged_pq_page_size "${PAGED_PQ_PAGE_SIZE}"
  --paged_router_prototypes "${PAGED_ROUTER_PROTOTYPES}"
  --paged_router_merge_rel "${PAGED_ROUTER_MERGE_REL}"
  --paged_router_merge_var "${PAGED_ROUTER_MERGE_VAR}"
  --paged_router_max_groups "${PAGED_ROUTER_MAX_GROUPS}"
  --progress_every "${PROGRESS_EVERY}"
  --backend "${BACKEND}"
  --backend_threads "${BACKEND_THREADS}"
)

if [[ "${SKIP_FIXEDK}" == "1" ]]; then
  args+=(--skip_fixedk)
fi
if [[ "${COMPUTE_OUTPUT_COS}" == "0" ]]; then
  args+=(--no-compute_output_cos)
fi
if [[ -n "${DECODE_FILTER}" ]]; then
  args+=(--decode_tokens_filter "${DECODE_FILTER}")
fi
if [[ "${EMIT_SAMPLES:-0}" == "1" ]]; then
  args+=(--emit_samples)
fi

echo "[run_online_ivfpq_simulator] source=${SOURCE_NPZ}"
echo "[run_online_ivfpq_simulator] output=${OUTPUT_DIR}"
echo "[run_online_ivfpq_simulator] policies=${POLICIES}"
echo "[run_online_ivfpq_simulator] decode_filter=${DECODE_FILTER}"

.venv/bin/python "${args[@]}"

if [[ "${PLOT:-1}" == "1" ]]; then
  ONLINE_IVFPQ_SUMMARY="${OUTPUT_DIR}/summary.csv" \
    OUT_DIR="${OUTPUT_DIR}/plots" \
    .venv/bin/python scripts/plot_online_ivfpq_simulator.py
fi
