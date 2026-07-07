#!/usr/bin/env bash
#SBATCH --job-name=walk-mb-requote
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000m
#SBATCH --time=04:00:00
#SBATCH --account=zhengya0
#SBATCH --partition=standard

# Historical correction script. Requotes the frontier per-step DRAM metric
# with faithful walk accounting (no de-escalation refund): the walk's cost
# is the deepest band READ on
# each axis (escalation probes read their lookahead band), not the settled
# state. Two arms over the historical 96-head-step population at tau=0.004:
# de-escalation on vs off. Expected: walk MB identical across arms (the
# climb is the same; the down-walk reads nothing), settled MB lower on the
# deesc arm by exactly the phantom refund.
set -euo pipefail
cd /gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention
module purge
module load python/3.10.4
export LD_LIBRARY_PATH="/sw/pkgs/arc/python/3.10.4/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

ROOT="benchmark_suite_result/walk_mb_requote_20260707"

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
    "$@" \
    --output_dir "$out_dir"
}

echo "[requote] arm 1: de-escalation ON (canonical)"
run_pop "${ROOT}/pop_deesc" --budget_deescalate
echo "[requote] arm 2: de-escalation OFF (escalation-only)"
run_pop "${ROOT}/pop_nodeesc"

echo "[requote] summary"
.venv/bin/python - <<'EOF'
import csv

def stats(path):
    settled, walk = [], []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            settled.append(float(row["step_MB_per_head"]))
            walk.append(float(row["walk_step_MB_per_head"]))
    n = len(settled)
    return n, sum(settled) / n, sum(walk) / n

for arm in ("pop_deesc", "pop_nodeesc"):
    p = f"benchmark_suite_result/walk_mb_requote_20260707/{arm}/per_head_joint_policy.csv"
    n, ms, mw = stats(p)
    print(f"[{arm}] rows={n} mean_settled_MB={ms:.4f} mean_walk_MB={mw:.4f} refund={100*(1-ms/mw):.1f}%")
EOF
echo "[requote] done"
