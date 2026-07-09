#!/usr/bin/env bash
#SBATCH --job-name=two-pass-cutoff-fp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000m
#SBATCH --time=12:00:00
#SBATCH --account=zhengya0
#SBATCH --partition=standard

# Issue #9 sweep: fixed-point precision of the two_pass_risk log-risk cutoff.
# RTL wants to adopt the scan-domain two-pass cutoff (2*logit + log err) as the
# V-selection contract; before answering we pin the smallest fixed-point grid
# that stays quality-neutral vs the fp64 two-pass baseline (the metrics used in
# the original validation: o-proj relL2, MB/head-query, exact-V reads).
#
# frac_bits=0 = fp64 baseline (byte-identical to the validated two_pass run,
# 52950302/52950295). The key-domain risk-key quantizer is left OFF
# (--v_risk_key_exp_bits 0) to ISOLATE the log-risk cutoff precision as the
# only variable. Everything else = the frozen operating point.
set -euo pipefail
cd /gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention
module purge
module load python/3.10.4
export LD_LIBRARY_PATH="/sw/pkgs/arc/python/3.10.4/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

STUDY_TAG="${STUDY_TAG:-two_pass_cutoff_fp_20260709}"
ROOT="benchmark_suite_result/${STUDY_TAG}"
FRAC_BITS_LIST="${FRAC_BITS_LIST:-0 2 3 4 5 6 8}"   # 0 = fp64 baseline
INT_BITS="${INT_BITS:-16}"
POP_QIDX="${POP_QIDX:-8}"
ALL_HEADS="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31"

run_eval() {
  local out_dir="$1" frac="$2"
  mkdir -p "$out_dir"
  .venv/bin/python benchmark/selector_eval/runners/run_joint_kv_budget_policy_eval.py \
    --qkv_trace attention_efficiency_result/real_qkv_llama31_l16_6838_g131072_q288_window32_graphall_s16.npz \
    --x_trace attention_efficiency_result/real_xtrace_llama31_8b_layer16_8k_g131072_sampled_maskstop.npz \
    --model_snapshot .hf_cache/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659 \
    --device cpu \
    --decode_lengths 8000,32000,128000 \
    --max_qidx_per_decode "$POP_QIDX" \
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
    --two_pass_cutoff_frac_bits "$frac" --two_pass_cutoff_int_bits "$INT_BITS" \
    --logit_buffer_bits 8 --logit_buffer_format e4m3 \
    --tail_score_calibration none \
    --score_proxy_variants baseline \
    --temporal_reuse_mode frozen --temporal_reuse_budget ladder \
    --output_dir "$out_dir"
}

for frac in $FRAC_BITS_LIST; do
  echo "[cutoff-fp] === frac_bits=${frac} (int_bits=${INT_BITS}) ==="
  run_eval "${ROOT}/frac_${frac}" "$frac"
done

echo "[cutoff-fp] aggregate"
.venv/bin/python - "$ROOT" "$FRAC_BITS_LIST" <<'PYEOF'
import csv, json, sys, math
from pathlib import Path
root = Path(sys.argv[1]); fracs = sys.argv[2].split()
def load(frac):
    d = root / f"frac_{frac}"
    rows = []
    with open(d / "per_head_joint_policy.csv") as f:
        for r in csv.DictReader(f):
            if r.get("v_selection_rule","").strip().lower().startswith("two_pass_risk"):
                rows.append(r)
    return rows
def agg(rows, thr):
    sel = [r for r in rows if abs(float(r["threshold"]) - thr) < 1e-12]
    if not sel: return None
    l2 = [float(r["head_attention_relative_L2"]) for r in sel]
    mbcol = "walk_step_MB_per_head" if "walk_step_MB_per_head" in sel[0] and sel[0]["walk_step_MB_per_head"] else "step_MB_per_head"
    mb = [float(r[mbcol]) for r in sel]
    rd = [float(r["v_exact_reads"]) for r in sel]
    return dict(n=len(sel), l2_mean=sum(l2)/len(l2), l2_max=max(l2),
                mb_mean=sum(mb)/len(mb), reads_mean=sum(rd)/len(rd), mbcol=mbcol)
report = {}
print(f"\n{'='*84}\nTWO-PASS CUTOFF FIXED-POINT SWEEP — quality vs fp64 baseline (frac=0)\n{'='*84}")
for thr in (0.002, 0.004):
    base = agg(load("0"), thr)
    print(f"\n--- threshold {thr} ---  (MB col = {base['mbcol'] if base else '?'})")
    print(f"{'frac':>5} {'n':>5} {'relL2_mean':>12} {'relL2_max':>12} {'MB_mean':>10} "
          f"{'dRelL2%':>9} {'dMB%':>8} {'reads_mean':>11} {'dReads%':>9}")
    for frac in fracs:
        a = agg(load(frac), thr)
        if a is None: continue
        dl2 = 100.0*(a['l2_mean']-base['l2_mean'])/base['l2_mean'] if base['l2_mean'] else 0.0
        dmb = 100.0*(a['mb_mean']-base['mb_mean'])/base['mb_mean'] if base['mb_mean'] else 0.0
        drd = 100.0*(a['reads_mean']-base['reads_mean'])/base['reads_mean'] if base['reads_mean'] else 0.0
        print(f"{frac:>5} {a['n']:>5} {a['l2_mean']:>12.6f} {a['l2_max']:>12.6f} {a['mb_mean']:>10.4f} "
              f"{dl2:>+9.3f} {dmb:>+8.3f} {a['reads_mean']:>11.1f} {drd:>+9.3f}")
        report[f"thr{thr}_frac{frac}"] = a
(root / "sweep_report.json").write_text(json.dumps(report, indent=2))
print(f"\nwrote {root/'sweep_report.json'}")
PYEOF
echo "[cutoff-fp] done: ${ROOT}"