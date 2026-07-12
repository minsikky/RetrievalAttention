#!/usr/bin/env bash
#SBATCH --job-name=dv-golden2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96000m
#SBATCH --time=18:00:00
#SBATCH --account=zhengya0
#SBATCH --partition=standard
#SBATCH --output=logs/dv_golden2_%j.log

# Issue #21 golden-2 (verify replay trace) committed-set operands. Dumps, over a
# contiguous decode segment (qidx 208..223 = 16 sampled decode positions, ctx
# ~31k..39k) x kv-group-0 heads {0,1,2,3}, the per-position committed K/V sets
# for the frozen walk (off) and the start+1 draft (start1). The combiner unions
# these over rounds of k=4 and k=8 to form the drafted union committed-KV list,
# per-position membership bitmap (fetch-on-miss stream), and per-round union
# sizes; the accepted-prefix / rollback ground truth comes from the GPU
# acceptance runs (acceptance.json), NOT this trace (single-layer L16). Non-union
# execution; start+1-only (see README -- the #14 predicted-theta union component
# is not reproduced here; the RTL FSM shape is identical).
set -euo pipefail
WT=/gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention/worktrees/draft-verify-goldens
cd "$WT"
module purge
module load python/3.10.4
export LD_LIBRARY_PATH="/sw/pkgs/arc/python/3.10.4/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

ROOT=/scratch/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention/draft_verify_golden2_20260712
mkdir -p logs

COMMON=(
  --qkv_trace "$WT/attention_efficiency_result/real_qkv_llama31_l16_6838_g131072_q288_window32_graphall_s16.npz"
  --x_trace "$WT/attention_efficiency_result/real_xtrace_llama31_8b_layer16_8k_g131072_sampled_maskstop.npz"
  --model_snapshot "$WT/.hf_cache/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
  --device cpu
  --decode_lengths all
  --max_qidx_per_decode 0
  --qidx_slice_start 208 --qidx_slice_stop 224
  --heads 0,1,2,3
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

for mode in off start1; do
  DUMP="${ROOT}/${mode}/gold_dump"
  rm -rf "${ROOT}/${mode}"
  mkdir -p "$DUMP" "${ROOT}/${mode}/gold_run"
  echo "[golden2] mode=${mode} -> ${DUMP}"
  SELECTOR_PQ_JOINT_DRAFT_MODE="${mode}" \
    "$WT/.venv/bin/python" "$WT/benchmark/selector_eval/runners/run_joint_kv_budget_policy_eval.py" \
    "${COMMON[@]}" \
    --golden_dump_stage2_dir "$DUMP" \
    --output_dir "${ROOT}/${mode}/gold_run"
done

echo "GOLDEN2-PRODUCER: COMPLETED root=${ROOT}"
