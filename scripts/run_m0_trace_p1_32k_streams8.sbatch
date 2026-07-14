#!/usr/bin/env bash
#SBATCH --job-name=m0-p1-32k-s8
#SBATCH --output=slurm-m0-p1-32k-streams8-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96000m
#SBATCH --time=10:00:00
#SBATCH --account=zhengya0
#SBATCH --partition=standard

# Issue #24 Priority 1 follow-up 1 (RTL request): 32k-class perf-iteration
# trace, IDENTICAL to run_m0_trace_p1_32k.sbatch except the pass-1 Q7.6 V-risk
# stream is dumped at ALL 8 decode positions (38,830..38,837) instead of the
# anchor (38,837) only, via --epoch_pass1_stream_position -1. All 32 query heads
# (grouped by 8 KV heads), 8 CONSECUTIVE decode positions 38,830..38,837
# (ctx 38,831..38,838 -- the 8-step serial-decode run ENDING at the existing
# 38,838-token golden context), OP-0.9 fixed parser (proxy_mass_m0p9). Emits
# per-position committed-set union per KV group and cross-checks the pass-1
# stream against the stage-2 item-8 two_pass operand for head 0 (anchor 38,837).
# The all-position streams are purely additive; committed sets are unchanged,
# so the anchor stream + every shared field stay bit-identical to
# m0_trace_32k_20260713 (regression gate G-A/G-C).
set -euo pipefail
cd /gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention/worktrees/m0-trace
module purge
module load python/3.10.4
export LD_LIBRARY_PATH="/sw/pkgs/arc/python/3.10.4/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

QKV=/scratch/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention/m0_trace_work/qkv_consec8_ctx38838.npz
OUT=benchmark/selector_eval/golden_vectors/m0_trace_32k_streams8_20260714
TRACE="$OUT/trace"; RUN="$OUT/gold_run"; X2="$OUT/stage2_xcheck"
rm -rf "$TRACE" "$RUN" "$X2"; mkdir -p "$TRACE" "$RUN" "$X2"
HEADS=$(python3 -c "print(','.join(str(i) for i in range(32)))")

COMMON=(--x_trace attention_efficiency_result/real_xtrace_llama31_8b_layer16_8k_g131072_sampled_maskstop.npz \
  --model_snapshot .hf_cache/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659 \
  --device cpu \
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
  --temporal_reuse_mode frozen --temporal_reuse_budget ladder --gqa_union_stats)

echo "[p1-s8] main run: 8 consecutive positions x 32 heads, pass-1 stream at ALL positions"
.venv/bin/python benchmark/selector_eval/runners/run_joint_kv_budget_policy_eval.py \
  --qkv_trace "$QKV" --decode_lengths all --max_qidx_per_decode 1 --heads "$HEADS" \
  --epoch_pass1_stream_position -1 \
  --epoch_trace_dir "$TRACE" --output_dir "$RUN" "${COMMON[@]}"

echo "[p1-s8] per-KV-group committed-set union"
.venv/bin/python scripts/emit_group_union.py --trace_dir "$TRACE"

echo "[p1-s8] gate 2 + gate 3 validators"
.venv/bin/python scripts/validate_epoch_trace.py --trace_dir "$TRACE" --run_dir "$RUN"

echo "[p1-s8] stage-2 item-8 pass-1 cross-check (head 0, anchor 38837)"
.venv/bin/python benchmark/selector_eval/runners/run_joint_kv_budget_policy_eval.py \
  --qkv_trace "$QKV" --decode_lengths 32000 --max_qidx_per_decode 1 --heads 0 \
  --two_pass_cutoff_frac_bits 6 --two_pass_cutoff_int_bits 7 \
  --golden_dump_stage2_dir "$X2" --golden_dump_stage2_heads 0 --output_dir "$X2/run" "${COMMON[@]}"

.venv/bin/python - "$TRACE" "$X2" <<'PY'
import sys, glob, numpy as np, pathlib
trace, x2 = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
# epoch anchor file for head 0 (position 38837)
ep = None
for f in trace.glob("epoch_q*_h0.npz"):
    d = np.load(f, allow_pickle=True)
    if int(d["position"]) == 38837 and "pass1_vrisk_q76" in d.files:
        ep = d; break
assert ep is not None, "no anchor pass-1 stream for head 0"
g2 = sorted(x2.glob("golden2_q*_h0.npz"))
assert g2, "no stage2 golden2 dump"
s2 = np.load(g2[0], allow_pickle=True)
mine = ep["pass1_vrisk_q76"]; mine_fin = mine[np.isfinite(mine)]
ref = np.asarray(s2["two_pass_pass1_logrisk_q_fp64"], dtype=np.float64)
ok_len = mine_fin.size == ref.size
ok_val = ok_len and np.array_equal(np.sort(mine_fin), np.sort(ref))
print(f"[p1-s8-xcheck] my_finite={mine_fin.size} stage2_finite={ref.size} equal={ok_val}")
sys.exit(0 if ok_val else 1)
PY
echo "[p1-s8] DONE: $OUT"
du -sh "$TRACE" "$RUN" 2>/dev/null || true
