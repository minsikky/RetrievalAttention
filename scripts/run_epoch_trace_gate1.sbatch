#!/usr/bin/env bash
#SBATCH --job-name=epoch-trace-gate1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000m
#SBATCH --time=01:00:00
#SBATCH --account=zhengya0
#SBATCH --partition=standard

# Issue #11 Phase 1 -- gate 1 (observer neutrality) on a SMALL config.
# Runs the frozen operating point twice (epoch trace OFF vs ON) on one kv group
# (heads 0-3) at decode 8000, then asserts every non-trace output is
# byte-identical (args.json legitimately differs: it records the new flag).
# Also runs the reconciliation (gate 2) + GQA-union (gate 3) validators on the
# small ON run so the fast job exercises all three gates before the full run.
set -euo pipefail
cd /gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention
module purge
module load python/3.10.4
export LD_LIBRARY_PATH="/sw/pkgs/arc/python/3.10.4/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

ROOT="attention_efficiency_result/epoch_trace_gate1_20260710"
OUT_OFF="${ROOT}/run_off"
OUT_ON="${ROOT}/run_on"
TRACE="${ROOT}/trace_on"
mkdir -p "$OUT_OFF" "$OUT_ON" "$TRACE"

COMMON=(
  --qkv_trace attention_efficiency_result/real_qkv_llama31_l16_6838_g131072_q288_window32_graphall_s16.npz
  --x_trace attention_efficiency_result/real_xtrace_llama31_8b_layer16_8k_g131072_sampled_maskstop.npz
  --model_snapshot .hf_cache/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659
  --device cpu
  --decode_lengths 8000
  --max_qidx_per_decode 1
  --heads 0,1,2,3
  --page_size 5632 --static_prefix 128 --static_suffix 128
  --subvecs 4 --subbits 8 --kmeans_iters 3
  --value_subvecs 1 --value_subbits 4
  --kv_storage_format fp16 --key_bytes 2 --value_bytes 2
  --code_stat_bytes 2 --selector_index_bytes 4 --survivor_logit_bytes 2
  --selector_mode fullscan --nprobes 512 --page_scan_frac 1.0
  --k_budgets 4096,8192,14336,32768
  --v_budgets 1024,2048,4096,6144,8192,12288,16384
  --k_budget_fracs 0.10,0.30,0.50,0.70,0.90,1.0
  --v_budget_fracs 0.05,0.10,0.20,0.40,0.60,0.80,1.0
  --policies k_first_alternating
  --start_strategies proxy_mass_m0p9
  --stability_thresholds 0.004
  --threshold_mode budget_delta_frac --threshold_reference_frac 0.2
  --threshold_scale_shape sqrt --threshold_min_scale 0.0 --threshold_max_scale 1.5
  --precision_k_hi_frac 0.1 --precision_v_hi_frac 0.1
  --precision_lo_mode int8 --precision_lo_bits 8
  --precision_split_freeze start
  --v_selection_rules global_residual_risk --v_local_block_size 1024
  --logit_buffer_bits 8 --logit_buffer_format e4m3
  --tail_score_calibration none
  --score_proxy_variants baseline
  --temporal_reuse_mode frozen --temporal_reuse_budget ladder
  --gqa_union_stats
)

echo "[gate1] OFF run"
.venv/bin/python benchmark/selector_eval/runners/run_joint_kv_budget_policy_eval.py \
  "${COMMON[@]}" --output_dir "$OUT_OFF"

echo "[gate1] ON run (epoch trace enabled)"
.venv/bin/python benchmark/selector_eval/runners/run_joint_kv_budget_policy_eval.py \
  "${COMMON[@]}" --output_dir "$OUT_ON" --epoch_trace_dir "$TRACE"

echo "[gate1] byte-diff non-trace outputs (args.json excluded: records the flag)"
FAIL=0
for f in per_head_joint_policy.csv layer_joint_policy.csv gqa_union_stats.csv; do
  if ! cmp -s "$OUT_OFF/$f" "$OUT_ON/$f"; then
    echo "  MISMATCH: $f"
    FAIL=1
  else
    echo "  identical: $f"
  fi
done
# summary.json embeds wall-clock elapsed_seconds; compare field-aware with
# timing excluded (every numeric field must still match bit-exactly).
if .venv/bin/python - "$OUT_OFF/summary.json" "$OUT_ON/summary.json" <<'EOF'
import json, sys
a = json.load(open(sys.argv[1]))
b = json.load(open(sys.argv[2]))
a.pop("elapsed_seconds", None)
b.pop("elapsed_seconds", None)
sys.exit(0 if json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True) else 1)
EOF
then
  echo "  identical (excl. elapsed_seconds): summary.json"
else
  echo "  MISMATCH: summary.json (beyond elapsed_seconds)"
  FAIL=1
fi
if [ "$FAIL" -ne 0 ]; then
  echo "[gate1] GATE 1 FAIL: observer changed a numeric output"
  exit 1
fi
echo "[gate1] GATE 1 PASS: all numeric outputs byte-identical (summary.json modulo wall clock)"

echo "[gate1] gate 2 + gate 3 validators on the small ON run"
.venv/bin/python scripts/validate_epoch_trace.py --trace_dir "$TRACE" --run_dir "$OUT_ON"
echo "[gate1] done"
