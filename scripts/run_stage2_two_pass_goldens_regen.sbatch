#!/usr/bin/env bash
#SBATCH --job-name=stage2-two-pass-goldens
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000m
#SBATCH --time=04:00:00
#SBATCH --account=zhengya0
#SBATCH --partition=standard

# Stage-2 golden regen adding issue #9 item-8 two_pass_risk selection goldens,
# ON TOP OF the item-7 operand superset (commit a0fcc4b). IDENTICAL config to
# run_stage2_item7_operands_regen.sbatch (escalation-only, --precision_split_freeze
# start, E6M12 keys, fp16 commit, item-7 all_*/vexact_band operands) PLUS the
# adopted two-pass cutoff fixed point (--two_pass_cutoff_frac_bits 6
# --two_pass_cutoff_int_bits 7 -> Q7.6, 13-bit total signed, clamp +/-64). The
# runner now ALSO dumps, per row, the two_pass_* block: pass-1
# quantized-log-risk cutoff, committed exact-V set (tokens + packed mask),
# pass-1/pass-2 quantized-log-risk operands, the observed log-risk range, and
# the occupancy metadata (distinct quantized-value count + 256-bin occupancy
# histogram with lo/hi anchors for RTL flat-histogram cutoff sizing). Existing
# item-1/6/7/all_* fields are additive-only
# (byte-identical superset). Verify does THREE things: (a) superset vs the
# blessed goldens, (b) G3 operand-rebuild gate (bit-tight), (c) the item-8
# two_pass rebuild gate (cutoff + committed set rebuilt from operands, exact).
set -euo pipefail
cd /gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention
module purge
module load python/3.10.4
export LD_LIBRARY_PATH="/sw/pkgs/arc/python/3.10.4/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

ROOT="benchmark_suite_result/stage2_two_pass_goldens_regen_20260709"
DUMP="${ROOT}/gold_dump"
mkdir -p "$DUMP" "${ROOT}/gold_run"

.venv/bin/python benchmark/selector_eval/runners/run_joint_kv_budget_policy_eval.py \
  --qkv_trace attention_efficiency_result/real_qkv_llama31_l16_6838_g131072_q288_window32_graphall_s16.npz \
  --x_trace attention_efficiency_result/real_xtrace_llama31_8b_layer16_8k_g131072_sampled_maskstop.npz \
  --model_snapshot .hf_cache/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659 \
  --device cpu \
  --decode_lengths 8000,32000,128000 \
  --max_qidx_per_decode 1 \
  --heads 0,8,16,24 \
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
  --two_pass_cutoff_frac_bits 6 --two_pass_cutoff_int_bits 7 \
  --logit_buffer_bits 8 --logit_buffer_format e4m3 \
  --tail_score_calibration none \
  --score_proxy_variants baseline \
  --temporal_reuse_mode frozen --temporal_reuse_budget ladder \
  --golden_dump_stage2_dir "$DUMP" \
  --output_dir "${ROOT}/gold_run"

echo "[twopass] verify: superset vs blessed goldens + G3 gate + item-8 two_pass rebuild gate"
.venv/bin/python benchmark/selector_eval/runners/verify_stage2_key_regen.py \
  --frozen benchmark/selector_eval/golden_vectors/stage2_20260707 \
  --regen "$DUMP" \
  --study_e6m12 benchmark_suite_result/v_risk_key_study_20260707/gold_dump_e6m12 \
  --g3_tol 1e-9 \
  --two_pass_tol 0.0 \
  --out "${ROOT}/verify_report.json" || echo "[twopass] verifier nonzero (inspect report: item-8 fields are +add; G3 + two_pass must be bit-tight)"

echo "[twopass] superset check: existing fields byte-identical, item-8 two_pass_* added"
OLD="benchmark/selector_eval/golden_vectors/stage2_20260707" NEW="$DUMP" .venv/bin/python - <<'EOF'
import glob, os, numpy as np
OLD = os.environ["OLD"]; NEW = os.environ["NEW"]
def summarize(pat):
    old = {os.path.basename(p): p for p in glob.glob(os.path.join(OLD, pat))}
    new = {os.path.basename(p): p for p in glob.glob(os.path.join(NEW, pat))}
    names = sorted(set(old) & set(new)); n_pure=0; problems=[]; added=set()
    for name in names:
        o=np.load(old[name],allow_pickle=True); n=np.load(new[name],allow_pickle=True)
        added.update(k for k in n.files if k not in o.files)
        removed=[k for k in o.files if k not in n.files]; moved=[]
        for k in o.files:
            if k not in n.files: continue
            a,b=o[k],n[k]
            try: same=a.shape==b.shape and np.array_equal(a,b)
            except Exception: same=str(a)==str(b)
            if not same: moved.append(k)
        if removed or moved: problems.append((name,removed,moved))
        else: n_pure+=1
    print(f"  [{pat}] {n_pure}/{len(names)} pure supersets (existing fields byte-identical)")
    if added: print(f"  [{pat}] added fields: {sorted(added)}")
    for name,r,m in problems: print(f"  [{pat}] !! {name}: removed={r} moved_existing={m}")
summarize("golden2_*.npz"); summarize("page_v_*.npz")
EOF

echo "[twopass] G3 + two_pass summary from report"
.venv/bin/python - <<'EOF'
import json
d=json.load(open("benchmark_suite_result/stage2_two_pass_goldens_regen_20260709/verify_report.json"))
g3=d.get("g3",{}); tp=d.get("two_pass",{})
print(f"  G3 checked {g3.get('checked')} max_abs_by_band_domain={g3.get('max_abs_by_band_domain')}")
print(f"  two_pass checked {tp.get('checked')} max_abs_cutoff={tp.get('max_abs_cutoff')} max_committed/mask_mismatch={tp.get('max_committed_mismatch')} worst={tp.get('worst')}")
tperrs=[r for r in tp.get('rows',[]) if r.get('error')]
print(f"  two_pass rows with error: {len(tperrs)}")
for r in tperrs[:12]: print(f"    {r['row']}: {r['error']}")
# scope of the pure->= vs all-ones caveat: any settled row with n_committed==context_len-ish?
for r in tp.get('rows',[]):
    print(f"    {r['row']}: n_committed={r.get('n_committed')} cutoff_abs={r.get('cutoff_abs')}")
print(f"  overall fatal={d.get('fatal')}")
EOF
echo "[twopass] done: ${ROOT}/verify_report.json"