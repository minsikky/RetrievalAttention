#!/usr/bin/env bash
#SBATCH --job-name=stage2-deesc-hunt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000m
#SBATCH --time=08:00:00
#SBATCH --account=zhengya0
#SBATCH --partition=standard

# Scan heads x ctx at the frozen operating point, dumping stage-2
# goldens, to find rows whose controller trace de-escalates (kd/vd
# probes) — supplement for issue #7 (the base 12-row set had 0).
# Env overrides: HUNT_TAG, HUNT_HEADS, HUNT_START (start strategy).
set -euo pipefail
cd /gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention
module purge
module load python/3.10.4
export LD_LIBRARY_PATH="/sw/pkgs/arc/python/3.10.4/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

HUNT_TAG="${HUNT_TAG:-stage2_deesc_hunt_20260707}"
HUNT_HEADS="${HUNT_HEADS:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31}"
HUNT_START="${HUNT_START:-proxy_mass_m0p9}"
DUMP_DIR="benchmark_suite_result/${HUNT_TAG}/dump"
OUT_DIR="benchmark_suite_result/${HUNT_TAG}/run"
mkdir -p "$DUMP_DIR" "$OUT_DIR"

.venv/bin/python benchmark/selector_eval/runners/run_joint_kv_budget_policy_eval.py \
  --qkv_trace attention_efficiency_result/real_qkv_llama31_l16_6838_g131072_q288_window32_graphall_s16.npz \
  --x_trace attention_efficiency_result/real_xtrace_llama31_8b_layer16_8k_g131072_sampled_maskstop.npz \
  --model_snapshot .hf_cache/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659 \
  --device cpu \
  --decode_lengths 8000,32000,128000 \
  --max_qidx_per_decode 1 \
  --heads "$HUNT_HEADS" \
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
  --start_strategies "$HUNT_START" \
  --stability_thresholds 0.004 \
  --threshold_mode budget_delta_frac --threshold_reference_frac 0.2 \
  --threshold_scale_shape sqrt --threshold_min_scale 0.0 --threshold_max_scale 1.5 \
  --budget_deescalate \
  --precision_k_hi_frac 0.1 --precision_v_hi_frac 0.1 \
  --precision_lo_mode int8 --precision_lo_bits 8 \
  --v_selection_rules global_residual_risk --v_local_block_size 1024 \
  --logit_buffer_bits 8 --logit_buffer_format e4m3 \
  --tail_score_calibration none \
  --score_proxy_variants baseline \
  --temporal_reuse_mode frozen --temporal_reuse_budget ladder \
  --golden_dump_stage2_dir "$DUMP_DIR" \
  --output_dir "$OUT_DIR"

echo "[hunt] run complete; scanning traces for kd/vd..."
export DUMP_DIR
.venv/bin/python - <<'EOF'
import numpy as np, glob
hits = []
for f in sorted(glob.glob(__import__("os").environ["DUMP_DIR"] + "/golden2_*.npz")):
    z = np.load(f, allow_pickle=True)
    kinds = [str(k) for k in z["probe_kind"]]
    if any(k in ("kd", "vd") for k in kinds):
        hits.append((f, "->".join(kinds)))
print(f"[hunt] de-escalating rows: {len(hits)}")
for f, t in hits:
    print(f"  {f}  {t}")
EOF
