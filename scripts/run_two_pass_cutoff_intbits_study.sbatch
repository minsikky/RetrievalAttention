#!/usr/bin/env bash
#SBATCH --job-name=two-pass-intbits
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000m
#SBATCH --time=08:00:00
#SBATCH --account=zhengya0
#SBATCH --partition=standard

# Issue #9 follow-up (RTL contract-width catch): empirical INTEGER-field sweep
# for the two-pass log-risk fixed point. The helper's int_bits is the integer
# field width (clamp +-2^(int_bits-1)); frac_bits is additional, so total
# storage = int_bits + frac_bits. Observed log-risk on real data is only
# [-42.2, +3.8] (scan domain), so clamps at int_bits >= 7 are dead; this sweep
# finds where the BOTTOM clamp starts moving selection (clamped ties collapse
# to one class at the floor; if the cutoff lands there, pass-2 over-commits).
# Sweep int_bits x frac_bits vs the fp64 baseline, with per-context breakdown
# (contract must hold across ctx; log-risk min deepens with ctx).
set -euo pipefail
cd /gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention
module purge
module load python/3.10.4
export LD_LIBRARY_PATH="/sw/pkgs/arc/python/3.10.4/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

STUDY_TAG="${STUDY_TAG:-two_pass_cutoff_intbits_20260709}"
ROOT="benchmark_suite_result/${STUDY_TAG}"
# (frac,int) pairs; "0:0" = fp64 baseline.
CONFIGS="${CONFIGS:-0:0 6:5 6:6 6:7 6:8 6:10 6:16 4:6 4:7 4:8 4:10}"
ALL_HEADS="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31"

run_eval() {
  local out_dir="$1" frac="$2" ibits="$3"
  mkdir -p "$out_dir"
  .venv/bin/python benchmark/selector_eval/runners/run_joint_kv_budget_policy_eval.py \
    --qkv_trace attention_efficiency_result/real_qkv_llama31_l16_6838_g131072_q288_window32_graphall_s16.npz \
    --x_trace attention_efficiency_result/real_xtrace_llama31_8b_layer16_8k_g131072_sampled_maskstop.npz \
    --model_snapshot .hf_cache/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659 \
    --device cpu \
    --decode_lengths 8000,32000,128000 \
    --max_qidx_per_decode 8 \
    --heads "$ALL_HEADS" \
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
    --stability_thresholds 0.002,0.004 \
    --threshold_mode budget_delta_frac --threshold_reference_frac 0.2 \
    --threshold_scale_shape sqrt --threshold_min_scale 0.0 --threshold_max_scale 1.5 \
    --precision_k_hi_frac 0.1 --precision_v_hi_frac 0.1 \
    --precision_lo_mode int8 --precision_lo_bits 8 \
    --v_selection_rules two_pass_risk --v_local_block_size 1024 \
    --v_risk_key_exp_bits 0 \
    --two_pass_cutoff_frac_bits "$frac" --two_pass_cutoff_int_bits "$ibits" \
    --logit_buffer_bits 8 --logit_buffer_format e4m3 \
    --tail_score_calibration none \
    --score_proxy_variants baseline \
    --temporal_reuse_mode frozen --temporal_reuse_budget ladder \
    --output_dir "$out_dir"
}

for cfg in $CONFIGS; do
  frac="${cfg%%:*}"; ibits="${cfg##*:}"
  echo "[intbits] === frac=${frac} int=${ibits} (total $((frac+ibits)) bits) ==="
  run_eval "${ROOT}/f${frac}_i${ibits}" "$frac" "$ibits"
done

echo "[intbits] aggregate"
.venv/bin/python - "$ROOT" "$CONFIGS" <<'PYEOF'
import csv, json, sys
from pathlib import Path
root = Path(sys.argv[1]); cfgs = sys.argv[2].split()
def load(tag):
    rows = []
    with open(root / tag / "per_head_joint_policy.csv") as f:
        for r in csv.DictReader(f):
            if r.get("v_selection_rule","").strip().lower().startswith("two_pass_risk"):
                rows.append(r)
    return rows
def agg(rows, thr, ctx=None):
    sel = [r for r in rows if abs(float(r["threshold"]) - thr) < 1e-12
           and (ctx is None or r["decode_length"] == ctx)]
    if not sel: return None
    l2 = [float(r["head_attention_relative_L2"]) for r in sel]
    mbcol = "walk_step_MB_per_head" if sel[0].get("walk_step_MB_per_head") else "step_MB_per_head"
    mb = [float(r[mbcol]) for r in sel]
    rd = [float(r["v_exact_reads"]) for r in sel]
    return dict(n=len(sel), l2_mean=sum(l2)/len(l2), l2_max=max(l2),
                mb_mean=sum(mb)/len(mb), reads_mean=sum(rd)/len(rd))
base_rows = load("f0_i0")
report = {}
print(f"\n{'='*100}\nTWO-PASS CUTOFF INTEGER-WIDTH SWEEP — vs fp64 baseline (f0_i0). total bits = int + frac\n{'='*100}")
for thr in (0.002, 0.004):
    print(f"\n--- threshold {thr}: OVERALL ---")
    print(f"{'cfg':>8} {'bits':>5} {'relL2_mean':>11} {'relL2_max':>11} {'dRelL2%':>9} {'dMB%':>8} {'dReads%':>9}")
    base = agg(base_rows, thr)
    for cfg in cfgs:
        frac, ib = cfg.split(":"); tag = f"f{frac}_i{ib}"
        a = agg(load(tag), thr)
        if a is None: continue
        dl2 = 100*(a['l2_mean']-base['l2_mean'])/base['l2_mean']
        dmb = 100*(a['mb_mean']-base['mb_mean'])/base['mb_mean']
        drd = 100*(a['reads_mean']-base['reads_mean'])/base['reads_mean']
        tot = (int(frac)+int(ib)) if cfg != "0:0" else 64
        print(f"{tag:>8} {tot:>5} {a['l2_mean']:>11.6f} {a['l2_max']:>11.6f} {dl2:>+9.3f} {dmb:>+8.3f} {drd:>+9.3f}")
        report[f"thr{thr}_{tag}"] = a
    print(f"--- threshold {thr}: PER-CONTEXT dReads% (ctx-dependence of the clamp) ---")
    hdr = f"{'cfg':>8}"
    ctxs = ["8000","32000","128000"]
    for c in ctxs: hdr += f" {'ctx'+c:>12}"
    print(hdr)
    for cfg in cfgs:
        if cfg == "0:0": continue
        frac, ib = cfg.split(":"); tag = f"f{frac}_i{ib}"
        rows = load(tag); line = f"{tag:>8}"
        for c in ctxs:
            a = agg(rows, thr, c); b = agg(base_rows, thr, c)
            line += f" {100*(a['reads_mean']-b['reads_mean'])/b['reads_mean']:>+12.3f}" if a and b else f" {'--':>12}"
        print(line)
(root / "sweep_report.json").write_text(json.dumps(report, indent=2))
print(f"\nwrote {root/'sweep_report.json'}")
PYEOF
echo "[intbits] done: ${ROOT}"