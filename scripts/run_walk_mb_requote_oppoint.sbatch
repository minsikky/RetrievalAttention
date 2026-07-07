#!/usr/bin/env bash
#SBATCH --job-name=walk-mb-oppoint
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000m
#SBATCH --time=08:00:00
#SBATCH --account=zhengya0
#SBATCH --partition=standard

# Historical correction script: requote the old deesc + precision(0.1,0.1)
# operating-point artifact with faithful walk accounting. This intentionally
# enables --budget_deescalate to reproduce job 52987561 and prove that
# de-escalation gives no DRAM refund. It is NOT the current frozen run recipe.
#
# Per-step DRAM = deepest band READ on each axis, no de-escalation refund.
# Args replicated from
# attention_efficiency_result/joint_kv_ladder_grid_20260706/deesc_precision_compose/args.json.
set -euo pipefail
cd /gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention
module purge
module load python/3.10.4
export LD_LIBRARY_PATH="/sw/pkgs/arc/python/3.10.4/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

OUT="benchmark_suite_result/walk_mb_requote_20260707/oppoint_deesc_precision"
mkdir -p "$OUT"

.venv/bin/python benchmark/selector_eval/runners/run_joint_kv_budget_policy_eval.py \
  --qkv_trace attention_efficiency_result/real_qkv_llama31_l16_6838_g131072_q288_window32_graphall_s16.npz \
  --x_trace attention_efficiency_result/real_xtrace_llama31_8b_layer16_8k_g131072_sampled_maskstop.npz \
  --model_snapshot .hf_cache/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659 \
  --device cpu \
  --decode_lengths all \
  --max_qidx_per_decode 1 \
  --heads 0,8,16,24 \
  --page_size 5632 --static_prefix 128 --static_suffix 128 \
  --subvecs 4 --subbits 8 --kmeans_iters 3 \
  --value_subvecs 1 --value_subbits 4 \
  --key_bytes 2 --value_bytes 2 \
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
  --precision_k_hi_frac 0.1 --precision_v_hi_frac 0.1 --precision_lo_bits 8 \
  --v_selection_rules global_residual_risk --v_local_block_size 1024 \
  --tail_score_calibration none \
  --score_proxy_variants baseline \
  --temporal_reuse_mode frozen --temporal_reuse_budget ladder \
  --budget_deescalate \
  --output_dir "$OUT"

echo "[oppoint] summary (per threshold)"
.venv/bin/python - <<'EOF'
import csv
from collections import defaultdict

acc = defaultdict(lambda: {"n": 0, "settled": 0.0, "walk": 0.0})
p = "benchmark_suite_result/walk_mb_requote_20260707/oppoint_deesc_precision/per_head_joint_policy.csv"
with open(p, newline="", encoding="utf-8") as fh:
    for row in csv.DictReader(fh):
        a = acc[row["threshold"]]
        a["n"] += 1
        a["settled"] += float(row["step_MB_per_head"])
        a["walk"] += float(row["walk_step_MB_per_head"])
for thr in sorted(acc, key=float):
    a = acc[thr]
    ms, mw = a["settled"] / a["n"], a["walk"] / a["n"]
    print(f"[oppoint thr={thr}] rows={a['n']} mean_settled_MB={ms:.4f} mean_walk_MB={mw:.4f} refund={100*(1-ms/mw):.1f}%")
EOF
echo "[oppoint] done"
