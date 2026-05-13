#!/bin/bash
# Slurm array wrapper for per-decode three-way mass-target sweeps.

#SBATCH --job-name=threeway-cut
#SBATCH --partition=standard
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%A_%a.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
module load python/3.10.4 >/dev/null 2>&1 || true

export LD_LIBRARY_PATH="/sw/pkgs/arc/python/3.10.4/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-16}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-16}}"

SOURCE_NPZ="${SOURCE_NPZ:-attention_efficiency_result/real_qkv_llama31_l16_6838_g16384_q24_graphall_s16.npz}"
OUTPUT_ROOT="${OUTPUT_ROOT:-attention_efficiency_result/threeway_cutoffs_graphall_s16_exact_v1}"
DECODE_LIST="${DECODE_LIST:-500,1000,2000,4000,8000,16000}"
MASS_TARGETS="${MASS_TARGETS:-0.8,0.9,0.95,0.98}"
QUERY_PER_CUTOFF="${QUERY_PER_CUTOFF:-4}"
STATIC_PREFIX="${STATIC_PREFIX:-128}"
STATIC_SUFFIX="${STATIC_SUFFIX:-128}"
Q_KNN="${Q_KNN:-8}"
GRAPH_DEGREE="${GRAPH_DEGREE:-8}"
SEED_COUNT="${SEED_COUNT:-32}"
MAX_VISITS="${MAX_VISITS:-8192}"
ROAR_NQ="${ROAR_NQ:-8}"
ROAR_L="${ROAR_L:-256}"
ROAR_ENHANCE_L="${ROAR_ENHANCE_L:-256}"
INCLUDE_GRAPH_METHODS="${INCLUDE_GRAPH_METHODS:-1}"
ENABLE_EXTRA_BASELINES="${ENABLE_EXTRA_BASELINES:-1}"
EXTRA_BASELINE_FAMILIES="${EXTRA_BASELINE_FAMILIES:-quest,sparq,loki,pqcache,magicpig,pariskv,ivfpq,binary_gated_pqcache,weighted_hamming_pqcache,sign_vq_lut_pqcache}"
QUEST_PAGE_SIZES="${QUEST_PAGE_SIZES:-16,32,64}"
SPARQ_RANKS="${SPARQ_RANKS:-8,16,32}"
LOKI_RANKS="${LOKI_RANKS:-8,16,32}"
MAGICPIG_CONFIGS="${MAGICPIG_CONFIGS:-10:150,10:170}"
MAGICPIG_MIN_COLLISIONS="${MAGICPIG_MIN_COLLISIONS:-2}"
MAGICPIG_ADAPTIVE_LADDER="${MAGICPIG_ADAPTIVE_LADDER:-10:150:2,10:300:2,10:150:1,10:300:1,8:150:2,8:300:2,8:150:1,8:300:1,6:150:1,6:300:1,4:150:1}"
PARISKV_CONFIGS="${PARISKV_CONFIGS:-8:32:0.01,8:64:0.02,10:64:0.05}"
PARISKV_RERANK_DIMS="${PARISKV_RERANK_DIMS:-16}"
PARISKV_ADAPTIVE_LADDER="${PARISKV_ADAPTIVE_LADDER:-8:32:0.01:16,8:64:0.02:16,10:64:0.05:16,10:96:0.10:32,10:128:0.20:32,12:128:0.50:64,12:128:1.00:64}"
PQCACHE_SUBVECS="${PQCACHE_SUBVECS:-2}"
PQCACHE_SUBBITS="${PQCACHE_SUBBITS:-6}"
PQCACHE_KMEANS_ITERS="${PQCACHE_KMEANS_ITERS:-3}"
IVFPQ_COARSE_CLUSTERS="${IVFPQ_COARSE_CLUSTERS:-128}"
IVFPQ_COARSE_ITERS="${IVFPQ_COARSE_ITERS:-3}"
IVFPQ_NPROBES="${IVFPQ_NPROBES:-1,2,4,8,16,32,64,128}"
IVFPQ_ONLINE_MODE="${IVFPQ_ONLINE_MODE:-frozen_append}"
IVFPQ_UPDATE_AMORTIZE_QUERIES="${IVFPQ_UPDATE_AMORTIZE_QUERIES:-0}"
IVFPQ_REBUILD_INTERVAL="${IVFPQ_REBUILD_INTERVAL:-8192}"
IVFPQ_FIXED_NPROBES="${IVFPQ_FIXED_NPROBES:-}"
IVFPQ_EMIT_PQ_LOGITS="${IVFPQ_EMIT_PQ_LOGITS:-0}"
BINARY_GATED_BITS="${BINARY_GATED_BITS:-128}"
BINARY_GATED_CANDIDATE_BUDGETS="${BINARY_GATED_CANDIDATE_BUDGETS:-512,1024,2048,4096,8192,16384,32768}"
SELF_INDEXING_GROUP_SIZE="${SELF_INDEXING_GROUP_SIZE:-4}"
SELF_INDEXING_CANDIDATE_BUDGETS="${SELF_INDEXING_CANDIDATE_BUDGETS:-512,1024,2048,4096,8192,16384,32768}"

IFS=',' read -r -a DECODE_VALUES <<< "${DECODE_LIST}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if (( TASK_ID < 0 || TASK_ID >= ${#DECODE_VALUES[@]} )); then
  echo "[run_threeway_cutoff_array] invalid task ${TASK_ID} for DECODE_LIST=${DECODE_LIST}" >&2
  exit 2
fi

DECODE_TOKENS="${DECODE_VALUES[$TASK_ID]}"
OUT_DIR="${OUTPUT_ROOT}/decode_${DECODE_TOKENS}"
EXTRA_BASELINE_FLAG="--enable_extra_baselines"
if [[ "${ENABLE_EXTRA_BASELINES}" != "1" ]]; then
  EXTRA_BASELINE_FLAG="--no-enable_extra_baselines"
fi
GRAPH_METHODS_FLAG="--include_graph_methods"
if [[ "${INCLUDE_GRAPH_METHODS}" != "1" ]]; then
  GRAPH_METHODS_FLAG="--no-include_graph_methods"
fi
IVFPQ_PQ_LOGITS_FLAG="--no-ivfpq_emit_pq_logits"
if [[ "${IVFPQ_EMIT_PQ_LOGITS}" == "1" ]]; then
  IVFPQ_PQ_LOGITS_FLAG="--ivfpq_emit_pq_logits"
fi

echo "[run_threeway_cutoff_array] decode=${DECODE_TOKENS} output=${OUT_DIR}"

.venv/bin/python benchmark/attention_efficiency_threeway_eval.py \
  --source_npz "${SOURCE_NPZ}" \
  --output_dir "${OUT_DIR}" \
  --decode_tokens_filter "${DECODE_TOKENS}" \
  --num_queries "${QUERY_PER_CUTOFF}" \
  --query_selection first \
  --mass_targets "${MASS_TARGETS}" \
  --static_prefix "${STATIC_PREFIX}" \
  --static_suffix "${STATIC_SUFFIX}" \
  --retro_cluster_size 128 \
  --retro_exact_clusters 0 \
  --q_knn "${Q_KNN}" \
  --graph_degree "${GRAPH_DEGREE}" \
  --seed_count "${SEED_COUNT}" \
  --max_visits "${MAX_VISITS}" \
  --roar_backend cpp \
  --roar_nq "${ROAR_NQ}" \
  --roar_l "${ROAR_L}" \
  --roar_enhance \
  --roar_enhance_l "${ROAR_ENHANCE_L}" \
  --roar_entry hub \
  --roar_threads "${SLURM_CPUS_PER_TASK:-16}" \
  --knn_chunk_rows 512 \
  "${GRAPH_METHODS_FLAG}" \
  "${EXTRA_BASELINE_FLAG}" \
  --extra_baseline_families "${EXTRA_BASELINE_FAMILIES}" \
  --quest_page_sizes "${QUEST_PAGE_SIZES}" \
  --sparq_ranks "${SPARQ_RANKS}" \
  --loki_ranks "${LOKI_RANKS}" \
  --magicpig_configs "${MAGICPIG_CONFIGS}" \
  --magicpig_min_collisions "${MAGICPIG_MIN_COLLISIONS}" \
  --magicpig_adaptive_ladder "${MAGICPIG_ADAPTIVE_LADDER}" \
  --pariskv_configs "${PARISKV_CONFIGS}" \
  --pariskv_rerank_dims "${PARISKV_RERANK_DIMS}" \
  --pariskv_adaptive_ladder "${PARISKV_ADAPTIVE_LADDER}" \
  --pqcache_subvecs "${PQCACHE_SUBVECS}" \
  --pqcache_subbits "${PQCACHE_SUBBITS}" \
  --pqcache_kmeans_iters "${PQCACHE_KMEANS_ITERS}" \
  --ivfpq_coarse_clusters "${IVFPQ_COARSE_CLUSTERS}" \
  --ivfpq_coarse_iters "${IVFPQ_COARSE_ITERS}" \
  --ivfpq_nprobes "${IVFPQ_NPROBES}" \
  --ivfpq_online_mode "${IVFPQ_ONLINE_MODE}" \
  --ivfpq_update_amortize_queries "${IVFPQ_UPDATE_AMORTIZE_QUERIES}" \
  --ivfpq_rebuild_interval "${IVFPQ_REBUILD_INTERVAL}" \
  --ivfpq_fixed_nprobes "${IVFPQ_FIXED_NPROBES}" \
  "${IVFPQ_PQ_LOGITS_FLAG}" \
  --binary_gated_bits "${BINARY_GATED_BITS}" \
  --binary_gated_candidate_budgets "${BINARY_GATED_CANDIDATE_BUDGETS}" \
  --self_indexing_group_size "${SELF_INDEXING_GROUP_SIZE}" \
  --self_indexing_candidate_budgets "${SELF_INDEXING_CANDIDATE_BUDGETS}"
