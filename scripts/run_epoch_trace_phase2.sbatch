#!/usr/bin/env bash
#SBATCH --job-name=epoch-trace-phase2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96000m
#SBATCH --time=06:00:00
#SBATCH --account=zhengya0
#SBATCH --partition=standard

# Issue #11 Phase 2 -- realized epoch traces (trace_format_version 2, adds
# start_v_tokens / k_hi_tokens{,_start}) for ALL 32 heads at FOUR positions:
#   q137 ctx  12,000 (decode  5,162) -- high-escalation tail (meanK 1.25)
#   q262 ctx  83,225 (decode 76,387) -- mid-context 80-100k (meanK 0.50)
#   q283 ctx 126,580 (decode 119,742) -- near cross-position mean (meanK 0.25 ~ 0.244)
#   q287 ctx 134,838 (decode 128,000) -- MVD (Phase-1 position, regenerated with v2 fields)
# Position shortlist mined from joint_kv_ladder_grid_20260706/deesc_precision_compose
# (288 positions x 4 heads at tau=0.004). Frozen operating point, single tau.
set -euo pipefail
cd /gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention
module purge
module load python/3.10.4
export LD_LIBRARY_PATH="/sw/pkgs/arc/python/3.10.4/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

ROOT="attention_efficiency_result/epoch_trace_phase2_20260710"
RUN="${ROOT}/gold_run"
TRACE="${ROOT}/trace"
mkdir -p "$RUN" "$TRACE"

HEADS=$(python3 -c "print(','.join(str(i) for i in range(32)))")

.venv/bin/python benchmark/selector_eval/runners/run_joint_kv_budget_policy_eval.py \
  --qkv_trace attention_efficiency_result/real_qkv_llama31_l16_6838_g131072_q288_window32_graphall_s16.npz \
  --x_trace attention_efficiency_result/real_xtrace_llama31_8b_layer16_8k_g131072_sampled_maskstop.npz \
  --model_snapshot .hf_cache/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659 \
  --device cpu \
  --decode_lengths 5162,76387,119742,128000 \
  --max_qidx_per_decode 1 \
  --heads "$HEADS" \
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
  --gqa_union_stats \
  --epoch_trace_dir "$TRACE" \
  --output_dir "$RUN"

echo "[phase2] resolved positions:"
python3 -c "
import csv
rows=list(csv.DictReader(open('${RUN}/per_head_joint_policy.csv')))
qs=sorted({(int(r['qidx']),int(r['position'])+1) for r in rows})
print('  (qidx, ctx) =', qs)
"
echo "[phase2] gates 2+3"
.venv/bin/python scripts/validate_epoch_trace.py --trace_dir "$TRACE" --run_dir "$RUN"
echo "[phase2] done: ${TRACE}"
