#!/bin/bash
# Small deployable approx-stop PQ selector smoke.

#SBATCH --job-name=pagedpq-approx
#SBATCH --partition=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%A.out

set -euo pipefail

export TRACE="attention_efficiency_result/real_qkv_llama31_l16_6838_g131072_q36_graphall_s16.npz"
export OUTPUT_DIR="attention_efficiency_result/selector_eval_pagedpq_approx_ps3072_s4b6_margins_smoke_t098_h0_v1"
export PLOT=0
export SELECTORS="paged_local_pq_approx,paged_local_pq_approx_mbp50,paged_local_pq_approx_mbp100,paged_local_pq_approx_mbp200,paged_local_pq_approx_mbp500"
export TARGETS="0.98"
export HEADS="0"
export DECODE_LENGTHS="128000"
export PAGED_PQ_PAGE_SIZE="3072"
export PAGED_PQ_SUBVECS="4"
export PAGED_PQ_SUBBITS="6"
export PAGED_PQ_KMEANS_ITERS="3"

bash benchmark/selector_eval/runners/run_selector_eval.sh
