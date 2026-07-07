#!/usr/bin/env bash
#SBATCH --job-name=stage2-frozen-split
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000m
#SBATCH --time=04:00:00
#SBATCH --account=zhengya0
#SBATCH --partition=standard

# Stage-2 golden regeneration adopting the FROZEN precision split (RTL #2 Q1):
# identical to run_stage2_key_regen.sbatch (escalation-only frozen algorithm,
# E6M12 keys, fp16 commit, item-7 fields) EXCEPT --precision_split_freeze start
# freezes the K-side hi/lo boundary once per step at the proxy-mass start rung
# instead of the per-rung growing boundary. This changes the RANKING domain
# (which tokens score with int8-QDQ logits), so a few walks resettle -> a few
# golden rows legitimately move. The verify step is mismatch-TOLERANT here (a
# frozen-split delta is expected, unlike the bit-exact key-regen); it exists to
# ENUMERATE exactly which rows/fields moved vs the current escalation-only
# goldens (commit 84d0323, dir stage2_20260707) so the RTL delivery is precise.
set -euo pipefail
cd /gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention
module purge
module load python/3.10.4
export LD_LIBRARY_PATH="/sw/pkgs/arc/python/3.10.4/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

ROOT="benchmark_suite_result/stage2_frozen_split_regen_20260707"
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
  --logit_buffer_bits 8 --logit_buffer_format e4m3 \
  --tail_score_calibration none \
  --score_proxy_variants baseline \
  --temporal_reuse_mode frozen --temporal_reuse_budget ladder \
  --golden_dump_stage2_dir "$DUMP" \
  --output_dir "${ROOT}/gold_run"

echo "[frozen-split] verify vs current escalation-only goldens (mismatch expected)"
.venv/bin/python benchmark/selector_eval/runners/verify_stage2_key_regen.py \
  --frozen benchmark/selector_eval/golden_vectors/stage2_20260707 \
  --regen "$DUMP" \
  --study_e6m12 benchmark_suite_result/v_risk_key_study_20260707/gold_dump_e6m12 \
  --out "${ROOT}/verify_report.json" || echo "[frozen-split] verifier returned nonzero (expected on frozen-split deltas)"

echo "[frozen-split] per-row diff enumeration vs escalation-only goldens"
.venv/bin/python - <<'EOF'
import glob, os, numpy as np

OLD = "benchmark/selector_eval/golden_vectors/stage2_20260707"
NEW = "benchmark_suite_result/stage2_frozen_split_regen_20260707/gold_dump"

old_files = {os.path.basename(p): p for p in glob.glob(os.path.join(OLD, "golden2_*.npz"))}
new_files = {os.path.basename(p): p for p in glob.glob(os.path.join(NEW, "golden2_*.npz"))}
names = sorted(set(old_files) & set(new_files))
print(f"[diff] comparing {len(names)} golden2 rows (old={len(old_files)} new={len(new_files)})")

n_identical = 0
moved = []
for name in names:
    o = np.load(old_files[name], allow_pickle=True)
    n = np.load(new_files[name], allow_pickle=True)
    keys = sorted(set(o.files) | set(n.files))
    diff_keys = []
    for k in keys:
        if k not in o.files or k not in n.files:
            diff_keys.append(k + ("(+new)" if k in n.files else "(-old)"))
            continue
        a, b = o[k], n[k]
        try:
            same = a.shape == b.shape and np.array_equal(a, b)
        except Exception:
            same = str(a) == str(b)
        if not same:
            diff_keys.append(k)
    if diff_keys:
        moved.append((name, diff_keys))
    else:
        n_identical += 1

print(f"[diff] bit-identical rows: {n_identical}/{len(names)}")
print(f"[diff] moved rows: {len(moved)}")
for name, dk in moved:
    print(f"  {name}: {len(dk)} fields moved -> {', '.join(dk[:12])}{' ...' if len(dk) > 12 else ''}")
EOF
echo "[frozen-split] done: ${ROOT}/verify_report.json"
