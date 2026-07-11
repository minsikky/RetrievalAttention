#!/usr/bin/env bash
#SBATCH --job-name=uc20
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96000m
#SBATCH --time=24:00:00
#SBATCH --account=zhengya0
#SBATCH --partition=standard

# Issue #20 union-commit quality validation. All 32 q heads (8 GQA groups of 4)
# at the frozen operating point (mirrors run_epoch_trace_full.sbatch /
# run_stage2_key_regen.sbatch env). --gqa_union_commit records baseline vs
# union-commit per-head relL2; --gqa_union_stats gives the group union/sum
# cross-check. CPU only. Pinned worktree (union-commit-20); absolute paths.
#
# Positions passed via $DECODES (comma list of decode-token lengths).
# SUBMIT WITH THE VARS IN THE SUBMISSION ENVIRONMENT, NOT --export=ALL,VAR=...:
#   DECODES=119742,76387,5162 TAG=small sbatch --export=ALL scripts/run_union_commit_20.sbatch
# (sbatch --export splits on commas, so --export=ALL,DECODES=a,b silently
# truncates the list to "a" -- this bit the first submission.)
#
#   q137 ctx 12,000  -> decode 5162
#   q262 ctx 83,225  -> decode 76387
#   q283 ctx 126,580 -> decode 119742
#   q287 ctx 134,838 -> decode 128000
set -euo pipefail
WT=/gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention/worktrees/union-commit
cd "$WT"
module purge
module load python/3.10.4
export LD_LIBRARY_PATH="/sw/pkgs/arc/python/3.10.4/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

DECODES="${DECODES:?set DECODES=comma,list}"
TAG="${TAG:-shard}"
OUTROOT=/scratch/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention/union_commit_20
OUT="${OUTROOT}/${TAG}"
mkdir -p "$OUT"

HEADS=$(python3 -c "print(','.join(str(i) for i in range(32)))")
echo "[uc20:${TAG}] decodes=${DECODES} heads=${HEADS}"

run_one () {
  local outdir="$1"; shift
  "$WT/.venv/bin/python" "$WT/benchmark/selector_eval/runners/run_joint_kv_budget_policy_eval.py" \
    --qkv_trace "$WT/attention_efficiency_result/real_qkv_llama31_l16_6838_g131072_q288_window32_graphall_s16.npz" \
    --x_trace "$WT/attention_efficiency_result/real_xtrace_llama31_8b_layer16_8k_g131072_sampled_maskstop.npz" \
    --model_snapshot "$WT/.hf_cache/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659" \
    --device cpu \
    --decode_lengths "$DECODES" \
    --max_qidx_per_decode 1 \
    --heads "$HEADS" \
    --page_size 5632 --static_prefix 128 --static_suffix 128 \
    --subvecs 4 --subbits 8 --kmeans_iters 3 \
    --value_subvecs 1 --value_subbits 4 \
    --kv_storage_format fp16 --key_bytes 2 --value_bytes 2 \
    --code_stat_bytes 2 --selector_index_bytes 4 --survivor_logit_bytes 2 \
    --selector_mode fullscan --nprobes 512 --page_scan_frac 1.0 \
    --k_budgets 4096,8192,14336,32768 \
    --v_budgets 1024,2048,4096,6144,8192,12288,16384 \
    --k_budget_fracs 0.10,0.30,0.50,0.70,0.90,1.0 \
    --v_budget_fracs 0.05,0.10,0.20,0.40,0.60,0.80,1.0 \
    --policies k_first_alternating \
    --start_strategies proxy_mass_m0p9 \
    --stability_thresholds 0.004 \
    --threshold_mode budget_delta_frac --threshold_reference_frac 0.2 \
    --threshold_scale_shape sqrt --threshold_min_scale 0.0 --threshold_max_scale 1.5 \
    --precision_k_hi_frac 0.1 --precision_v_hi_frac 0.1 \
    --precision_lo_mode int8 --precision_lo_bits 8 \
    --precision_split_freeze start \
    --v_selection_rules global_residual_risk --v_local_block_size 1024 \
    --logit_buffer_bits 8 --logit_buffer_format e4m3 \
    --tail_score_calibration none \
    --score_proxy_variants baseline \
    --gqa_union_stats --gqa_union_commit \
    --output_dir "$outdir"
}

echo "[uc20:${TAG}] main run"
run_one "$OUT"
echo "[uc20:${TAG}] done -> $OUT"
