#!/usr/bin/env bash
#SBATCH --job-name=group_max_v
#SBATCH --account=zhengya0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96g
#SBATCH --time=16:00:00
#SBATCH --output=%x_%j.out
# Issue #20 group-max V-selection eval. Runs the additive
# --gqa_group_max_eval producer path (device=cpu, cached qkv trace) for a
# comma-separated QIDX list, one output dir per qidx. Pinned worktree so the
# edited producer is read at runtime without touching the main tree.
set -euo pipefail
WT=/gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention/worktrees/group-max-v
MAIN=/gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention
cd "$WT"
module purge
module load python/3.10.4
export LD_LIBRARY_PATH="/sw/pkgs/arc/python/3.10.4/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

QIDX="${QIDX:?set QIDX=comma,list}"
OUTROOT="${OUTROOT:-/scratch/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention/group_max_v_20260712}"
ALLHEADS=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31
mkdir -p "$OUTROOT"

COMMON=(
  --qkv_trace "$MAIN/attention_efficiency_result/real_qkv_llama31_l16_6838_g131072_q288_window32_graphall_s16.npz"
  --x_trace "$MAIN/attention_efficiency_result/real_xtrace_llama31_8b_layer16_8k_g131072_sampled_maskstop.npz"
  --model_snapshot ".hf_cache/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
  --device cpu --decode_lengths all --max_qidx_per_decode 1
  --page_size 5632 --static_prefix 128 --static_suffix 128
  --subvecs 4 --subbits 8 --kmeans_iters 3 --value_subvecs 1 --value_subbits 4
  --kv_storage_format fp16 --key_bytes 2 --value_bytes 2
  --code_stat_bytes 2 --selector_index_bytes 4 --survivor_logit_bytes 2
  --selector_mode fullscan --nprobes 512 --page_scan_frac 1.0
  --k_budgets 4096,8192,14336,32768
  --v_budgets 1024,2048,4096,6144,8192,12288,16384
  --k_budget_fracs 0.10,0.30,0.50,0.70,0.90,1.0
  --v_budget_fracs 0.05,0.10,0.20,0.40,0.60,0.80,1.0
  --policies k_first_alternating --start_strategies proxy_mass_m0p9 --stability_thresholds 0.004
  --threshold_mode budget_delta_frac --threshold_reference_frac 0.2
  --threshold_scale_shape sqrt --threshold_min_scale 0.0 --threshold_max_scale 1.5
  --precision_k_hi_frac 0.1 --precision_v_hi_frac 0.1
  --precision_lo_mode int8 --precision_lo_bits 8 --precision_split_freeze start
  --v_selection_rules global_residual_risk --v_local_block_size 1024
  --v_risk_key_exp_bits 0 --v_risk_key_mantissa_bits 8
  --heads "$ALLHEADS"
  --gqa_union_commit --gqa_group_max_eval
)

IFS=',' read -ra QS <<< "$QIDX"
for q in "${QS[@]}"; do
  OUT="$OUTROOT/run_q${q}"
  rm -rf "$OUT"; mkdir -p "$OUT"
  echo "[group_max_v] qidx=$q -> $OUT  $(date)"
  "$WT/.venv/bin/python" "$WT/benchmark/selector_eval/runners/run_joint_kv_budget_policy_eval.py" \
    "${COMMON[@]}" \
    --qidx_slice_start "$q" --qidx_slice_stop "$((q+1))" \
    --output_dir "$OUT"
  echo "[group_max_v] qidx=$q DONE $(date)"
done
echo "[group_max_v] ALL QIDX DONE -> $OUTROOT"
