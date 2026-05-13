#!/bin/bash
# Shallow selector-evaluation entrypoint.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
module load python/3.10.4 >/dev/null 2>&1 || true

export LD_LIBRARY_PATH="/sw/pkgs/arc/python/3.10.4/lib:${LD_LIBRARY_PATH:-}"

TRACE="${TRACE:?TRACE is required}"
OUTPUT_DIR="${OUTPUT_DIR:-attention_efficiency_result/selector_eval_smoke}"
PRESET="${PRESET:-}"
SELECTORS="${SELECTORS:-dense,top_mass_oracle}"
DECODE_LENGTHS="${DECODE_LENGTHS:-32000,64000,128000}"
TARGETS="${TARGETS:-0.95,0.98}"
TAIL_ESTIMATORS="${TAIL_ESTIMATORS:-}"
HEADS="${HEADS:-}"
STATIC_PREFIX="${STATIC_PREFIX:-128}"
STATIC_SUFFIX="${STATIC_SUFFIX:-128}"
RETROINFER_CLUSTER_SIZE="${RETROINFER_CLUSTER_SIZE:-256}"
PAGED_PQ_PAGE_SIZE="${PAGED_PQ_PAGE_SIZE:-2048}"
PAGED_PQ_SUBVECS="${PAGED_PQ_SUBVECS:-2}"
PAGED_PQ_SUBBITS="${PAGED_PQ_SUBBITS:-6}"
PAGED_PQ_KMEANS_ITERS="${PAGED_PQ_KMEANS_ITERS:-3}"
PAGED_PQ_PERMUTATION="${PAGED_PQ_PERMUTATION:-none}"
VALUE_PQ_SUBVECS="${VALUE_PQ_SUBVECS:-0}"
VALUE_PQ_SUBBITS="${VALUE_PQ_SUBBITS:-0}"
PAGED_ROUTER_MAX_GROUPS="${PAGED_ROUTER_MAX_GROUPS:-512}"
PAGED_ROUTER_MERGE_REL="${PAGED_ROUTER_MERGE_REL:-0.05}"
PAGED_NPROBES="${PAGED_NPROBES:-1,2,4,8,16,32,64,128,256,512}"
IVFPQ_NPROBES="${IVFPQ_NPROBES:-1,2,4,8,16,32,64,128}"
IVFPQ_COARSE_CLUSTERS="${IVFPQ_COARSE_CLUSTERS:-128}"
IVFPQ_REBUILD_INTERVAL="${IVFPQ_REBUILD_INTERVAL:-8192}"
SPARQ_RANK="${SPARQ_RANK:-16}"
MAGICPIG_BITS="${MAGICPIG_BITS:-10}"
MAGICPIG_TABLES="${MAGICPIG_TABLES:-150}"
MAGICPIG_MIN_COLLISIONS="${MAGICPIG_MIN_COLLISIONS:-2}"
RA_PROVENANCE_TOPK="${RA_PROVENANCE_TOPK:-64}"
RA_CONNECT_WINDOW="${RA_CONNECT_WINDOW:-8}"
RA_DEGREE="${RA_DEGREE:-32}"
RA_SEED_COUNT="${RA_SEED_COUNT:-64}"
RA_MAX_VISITS="${RA_MAX_VISITS:-2048}"
RA_MIN_VISITS="${RA_MIN_VISITS:-64}"

args=(
  benchmark/selector_eval/runners/run_selector_eval.py
  --trace "${TRACE}"
  --output_dir "${OUTPUT_DIR}"
  --decode_lengths "${DECODE_LENGTHS}"
)

if [[ -n "${PRESET}" ]]; then
  args+=(--preset "${PRESET}")
else
  args+=(
    --selectors "${SELECTORS}"
    --targets "${TARGETS}"
    --tail_estimators "${TAIL_ESTIMATORS}"
    --static_prefix "${STATIC_PREFIX}"
    --static_suffix "${STATIC_SUFFIX}"
    --retroinfer_cluster_size "${RETROINFER_CLUSTER_SIZE}"
    --paged_pq_page_size "${PAGED_PQ_PAGE_SIZE}"
    --paged_pq_subvecs "${PAGED_PQ_SUBVECS}"
    --paged_pq_subbits "${PAGED_PQ_SUBBITS}"
    --paged_pq_kmeans_iters "${PAGED_PQ_KMEANS_ITERS}"
    --paged_pq_permutation "${PAGED_PQ_PERMUTATION}"
    --value_pq_subvecs "${VALUE_PQ_SUBVECS}"
    --value_pq_subbits "${VALUE_PQ_SUBBITS}"
    --paged_router_max_groups "${PAGED_ROUTER_MAX_GROUPS}"
    --paged_router_merge_rel "${PAGED_ROUTER_MERGE_REL}"
    --paged_nprobes "${PAGED_NPROBES}"
    --ivfpq_nprobes "${IVFPQ_NPROBES}"
    --ivfpq_coarse_clusters "${IVFPQ_COARSE_CLUSTERS}"
    --ivfpq_rebuild_interval "${IVFPQ_REBUILD_INTERVAL}"
    --sparq_rank "${SPARQ_RANK}"
    --magicpig_bits "${MAGICPIG_BITS}"
    --magicpig_tables "${MAGICPIG_TABLES}"
    --magicpig_min_collisions "${MAGICPIG_MIN_COLLISIONS}"
    --ra_provenance_topk "${RA_PROVENANCE_TOPK}"
    --ra_connect_window "${RA_CONNECT_WINDOW}"
    --ra_degree "${RA_DEGREE}"
    --ra_seed_count "${RA_SEED_COUNT}"
    --ra_max_visits "${RA_MAX_VISITS}"
    --ra_min_visits "${RA_MIN_VISITS}"
  )
fi

if [[ -n "${HEADS}" ]]; then
  args+=(--heads "${HEADS}")
fi

.venv/bin/python "${args[@]}"
