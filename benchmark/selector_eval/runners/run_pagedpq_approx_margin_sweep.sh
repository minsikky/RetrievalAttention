#!/bin/bash
# Narrow deployable approx-stop PQ margin sweep at 128k.

#SBATCH --job-name=pagedpq-margin
#SBATCH --partition=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%A.out

set -euo pipefail

export TRACE="attention_efficiency_result/real_qkv_llama31_l16_6838_g131072_q36_graphall_s16.npz"
export OUTPUT_DIR="attention_efficiency_result/selector_eval_pagedpq_approx_ps3072_s4b6_margin_sweep_t098_h0_v1"
export PLOT=0
export SELECTORS="paged_local_pq_approx_mbp5,paged_local_pq_approx_mbp10,paged_local_pq_approx_mbp15,paged_local_pq_approx_mbp20,paged_local_pq_approx_mbp25,paged_local_pq_approx_mbp30,paged_local_pq_approx_mbp35,paged_local_pq_approx_mbp40,paged_local_pq_approx_mbp45,paged_local_pq_approx_mbp50"
export TARGETS="0.98"
export HEADS="0"
export DECODE_LENGTHS="128000"
export PAGED_PQ_PAGE_SIZE="3072"
export PAGED_PQ_SUBVECS="4"
export PAGED_PQ_SUBBITS="6"
export PAGED_PQ_KMEANS_ITERS="3"

bash benchmark/selector_eval/runners/run_selector_eval.sh
