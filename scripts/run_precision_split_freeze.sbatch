#!/usr/bin/env bash
#SBATCH --job-name=prec-split-freeze
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000m
#SBATCH --time=04:00:00
#SBATCH --account=zhengya0
#SBATCH --partition=standard

# Historical reproduction script from before de-escalation was removed from
# the frozen algorithm. Do not use this as the current canonical precision
# study without deleting --budget_deescalate.
#
# RTL issue #2: measure the quality/MB delta of freezing the K-side
# progressive-precision hi/lo split ONCE per step vs the per-rung
# growing boundary. Three arms over the historical 96-head-step
# de-escalating population at tau=0.004, precision(0.1,0.1):
# --precision_split_freeze in
# {none (current), start (freeze at proxy-mass start rung hi_count),
# kbmax (freeze at ceil(hi_frac*max(k_budgets)))}. Args are otherwise
# identical to the pop_deesc arm of run_walk_mb_requote.sbatch.
set -euo pipefail
cd /gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention
module purge
module load python/3.10.4
export LD_LIBRARY_PATH="/sw/pkgs/arc/python/3.10.4/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

ROOT="benchmark_suite_result/precision_split_freeze_20260707"

run_pop() {
  local out_dir="$1"; shift
  mkdir -p "$out_dir"
  .venv/bin/python benchmark/selector_eval/runners/run_joint_kv_budget_policy_eval.py \
    --qkv_trace attention_efficiency_result/real_qkv_llama31_l16_6838_g131072_q288_window32_graphall_s16.npz \
    --x_trace attention_efficiency_result/real_xtrace_llama31_8b_layer16_8k_g131072_sampled_maskstop.npz \
    --model_snapshot .hf_cache/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659 \
    --device cpu \
    --decode_lengths 8000,32000,128000 \
    --max_qidx_per_decode 8 \
    --heads 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31 \
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
    --v_selection_rules global_residual_risk --v_local_block_size 1024 \
    --logit_buffer_bits 8 --logit_buffer_format e4m3 \
    --tail_score_calibration none \
    --score_proxy_variants baseline \
    --temporal_reuse_mode frozen --temporal_reuse_budget ladder \
    --budget_deescalate \
    "$@" \
    --output_dir "$out_dir"
}

echo "[prec-split] arm 1/3: none (current per-rung growing boundary)"
run_pop "${ROOT}/none"  --precision_split_freeze none
echo "[prec-split] arm 2/3: start (freeze at proxy-mass start rung)"
run_pop "${ROOT}/start" --precision_split_freeze start
echo "[prec-split] arm 3/3: kbmax (freeze at ceil(hi_frac*max_k_budget))"
run_pop "${ROOT}/kbmax" --precision_split_freeze kbmax

echo "[prec-split] summary"
.venv/bin/python - <<'EOF'
import csv, math

ROOT = "benchmark_suite_result/precision_split_freeze_20260707"

def load(arm):
    with open(f"{ROOT}/{arm}/per_head_joint_policy.csv", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))

def col(rows, name):
    return [float(r[name]) for r in rows if r.get(name, "") not in ("", None)]

def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")

hdr = ("arm", "rows", "meanL2", "maxL2", "mean_step_MB", "mean_walk_MB",
       "mean_sel_k", "mean_k_lo", "mean_v_lo")
print(("{:>6} {:>5} " + " ".join(["{:>11}"]*7)).format(*hdr))
base = None
for arm in ("none", "start", "kbmax"):
    rows = load(arm)
    l2 = col(rows, "head_attention_relative_L2")
    step = col(rows, "step_MB_per_head")
    walk = col(rows, "walk_step_MB_per_head")
    selk = col(rows, "selected_k_tokens")
    klo = col(rows, "precision_k_lo_tokens")
    vlo = col(rows, "precision_v_lo_reads")
    vals = (arm, len(rows), mean(l2), max(l2), mean(step), mean(walk),
            mean(selk), mean(klo), mean(vlo))
    print(("{:>6} {:>5} " + " ".join(["{:>11.5f}"]*7)).format(*vals))
    if arm == "none":
        base = (mean(l2), mean(step), mean(walk))
    else:
        dl2 = (mean(l2) - base[0]) / base[0] * 100.0
        dstep = (mean(step) - base[1]) / base[1] * 100.0
        dwalk = (mean(walk) - base[2]) / base[2] * 100.0
        print(f"       vs none: meanL2 {dl2:+.2f}%  step_MB {dstep:+.2f}%  walk_MB {dwalk:+.2f}%")
EOF
echo "[prec-split] done"
