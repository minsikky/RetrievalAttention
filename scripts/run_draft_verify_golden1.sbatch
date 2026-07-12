#!/usr/bin/env bash
#SBATCH --job-name=dv-golden1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96000m
#SBATCH --time=18:00:00
#SBATCH --account=zhengya0
#SBATCH --partition=standard
#SBATCH --output=logs/dv_golden1_%j.log

# Issue #21 golden-1 (draft-selection block) producer. Runs the stage-2 CPU
# dump path THREE times -- SELECTOR_PQ_JOINT_DRAFT_MODE={off,start1,start2} --
# on the standard stage-2 golden positions (q159/q223/q287) x heads {0,8,16,24}
# from the SAME scan state. `off` reproduces the frozen escalation-walk
# committed set (must match stage2_union_commit_20260710 per q/h); start1/start2
# pin the one-shot draft rung (proxy-mass start +1/+2) and dump that committed
# set. Non-union execution (committed_k_tokens / committed_v_tokens are the
# per-head selection, independent of the #20 union-commit execution policy).
# Config = the frozen stage-2 operating point (two-pass base of
# run_stage2_union_commit_regen.sbatch) MINUS --gqa_union_commit.
set -euo pipefail
WT=/gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention/worktrees/draft-verify-goldens
cd "$WT"
module purge
module load python/3.10.4
export LD_LIBRARY_PATH="/sw/pkgs/arc/python/3.10.4/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

ROOT=/scratch/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention/draft_verify_golden1_20260712
mkdir -p logs

COMMON=(
  --qkv_trace "$WT/attention_efficiency_result/real_qkv_llama31_l16_6838_g131072_q288_window32_graphall_s16.npz"
  --x_trace "$WT/attention_efficiency_result/real_xtrace_llama31_8b_layer16_8k_g131072_sampled_maskstop.npz"
  --model_snapshot "$WT/.hf_cache/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
  --device cpu
  --decode_lengths 8000,32000,128000
  --max_qidx_per_decode 1
  --heads 0,8,16,24
  --page_size 5632 --static_prefix 128 --static_suffix 128
  --subvecs 4 --subbits 8 --kmeans_iters 3
  --value_subvecs 1 --value_subbits 4
  --kv_storage_format fp16 --key_bytes 2 --value_bytes 2
  --code_stat_bytes 2 --selector_index_bytes 4 --survivor_logit_bytes 2
  --selector_mode fullscan --nprobes 512 --page_scan_frac 1.0
  --k_budgets 4096,8192,14336,32768
  --v_budgets 1024,2048,4096,6144,8192,12288,16384
  --k_budget_fracs 0.10,0.30,0.50,0.70,0.90,1.0
  --v_budget_fracs 0.05,0.10,0.20,0.40,0.60,0.80,1.0
  --policies k_first_alternating
  --start_strategies proxy_mass_m0p9
  --stability_thresholds 0.004
  --threshold_mode budget_delta_frac --threshold_reference_frac 0.2
  --threshold_scale_shape sqrt --threshold_min_scale 0.0 --threshold_max_scale 1.5
  --precision_k_hi_frac 0.1 --precision_v_hi_frac 0.1
  --precision_lo_mode int8 --precision_lo_bits 8
  --precision_split_freeze start
  --v_selection_rules global_residual_risk --v_local_block_size 1024
  --two_pass_cutoff_frac_bits 6 --two_pass_cutoff_int_bits 7
  --logit_buffer_bits 8 --logit_buffer_format e4m3
  --tail_score_calibration none
  --score_proxy_variants baseline
  --temporal_reuse_mode frozen --temporal_reuse_budget ladder
)

for mode in off start1 start2; do
  DUMP="${ROOT}/${mode}/gold_dump"
  rm -rf "${ROOT}/${mode}"
  mkdir -p "$DUMP" "${ROOT}/${mode}/gold_run"
  echo "[golden1] mode=${mode} -> ${DUMP}"
  SELECTOR_PQ_JOINT_DRAFT_MODE="${mode}" \
    "$WT/.venv/bin/python" "$WT/benchmark/selector_eval/runners/run_joint_kv_budget_policy_eval.py" \
    "${COMMON[@]}" \
    --golden_dump_stage2_dir "$DUMP" \
    --output_dir "${ROOT}/${mode}/gold_run"
done

echo "GOLDEN1-PRODUCER: COMPLETED root=${ROOT}"
