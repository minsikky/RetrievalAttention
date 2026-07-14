#!/usr/bin/env bash
#SBATCH --job-name=m0-p2-128k-s8
#SBATCH --output=slurm-m0-p2-128k-streams8-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180000m
#SBATCH --time=48:00:00
#SBATCH --account=zhengya0
#SBATCH --partition=standard

# Issue #24 Priority 2 follow-up (RTL request): 128k production-gate sequence,
# IDENTICAL to run_m0_trace_p2_128k.sbatch except the pass-1 Q7.6 V-risk stream
# is dumped at ALL 8 decode positions (134,830..134,837) instead of the newest
# only, via --epoch_pass1_stream_position -1. Same schema as P1, 8 CONSECUTIVE
# decode positions 134,830..134,837 (ctx 134,831..134,838 -- ending at the
# existing q287 golden context), all 32 heads grouped by 8 KV heads, OP-0.9
# fixed parser.
set -euo pipefail
cd /gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention/worktrees/m0-trace
module purge
module load python/3.10.4
export LD_LIBRARY_PATH="/sw/pkgs/arc/python/3.10.4/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

QKV=/scratch/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention/m0_trace_work/qkv_consec8_ctx134838.npz
OUT=benchmark/selector_eval/golden_vectors/m0_trace_128k_streams8_20260714
TRACE="$OUT/trace"; RUN="$OUT/gold_run"
rm -rf "$TRACE" "$RUN"; mkdir -p "$TRACE" "$RUN"
HEADS=$(python3 -c "print(','.join(str(i) for i in range(32)))")

echo "[p2-s8] main run: 8 consecutive positions x 32 heads at 128k, pass-1 at ALL positions"
.venv/bin/python benchmark/selector_eval/runners/run_joint_kv_budget_policy_eval.py \
  --qkv_trace "$QKV" \
  --x_trace attention_efficiency_result/real_xtrace_llama31_8b_layer16_8k_g131072_sampled_maskstop.npz \
  --model_snapshot .hf_cache/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659 \
  --device cpu --decode_lengths all --max_qidx_per_decode 1 --heads "$HEADS" \
  --epoch_pass1_stream_position -1 \
  --page_size 5632 --static_prefix 128 --static_suffix 128 \
  --subvecs 4 --subbits 8 --kmeans_iters 3 --value_subvecs 1 --value_subbits 4 \
  --kv_storage_format fp16 --key_bytes 2 --value_bytes 2 --code_stat_bytes 2 --selector_index_bytes 4 --survivor_logit_bytes 2 \
  --selector_mode fullscan --nprobes 512 --page_scan_frac 1.0 \
  --k_budgets 4096,8192,14336,32768 --v_budgets 1024,2048,4096,6144,8192,12288,16384 \
  --k_budget_fracs 0.10,0.30,0.50,0.70,0.90,1.0 --v_budget_fracs 0.05,0.10,0.20,0.40,0.60,0.80,1.0 \
  --policies k_first_alternating --start_strategies proxy_mass_m0p9 --stability_thresholds 0.004 \
  --threshold_mode budget_delta_frac --threshold_reference_frac 0.2 --threshold_scale_shape sqrt --threshold_min_scale 0.0 --threshold_max_scale 1.5 \
  --precision_k_hi_frac 0.1 --precision_v_hi_frac 0.1 --precision_lo_mode int8 --precision_lo_bits 8 --precision_split_freeze start \
  --v_selection_rules global_residual_risk --v_local_block_size 1024 \
  --logit_buffer_bits 8 --logit_buffer_format e4m3 --tail_score_calibration none --score_proxy_variants baseline \
  --temporal_reuse_mode frozen --temporal_reuse_budget ladder --gqa_union_stats \
  --epoch_trace_dir "$TRACE" --output_dir "$RUN"

echo "[p2-s8] per-KV-group committed-set union"
.venv/bin/python scripts/emit_group_union.py --trace_dir "$TRACE"
echo "[p2-s8] gate 2 + gate 3 validators"
.venv/bin/python scripts/validate_epoch_trace.py --trace_dir "$TRACE" --run_dir "$RUN"
echo "[p2-s8] DONE: $OUT"
du -sh "$TRACE" "$RUN" 2>/dev/null || true
