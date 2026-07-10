#!/usr/bin/env bash
#SBATCH --job-name=vthr-sweep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96000m
#SBATCH --time=24:00:00
#SBATCH --account=zhengya0
#SBATCH --partition=standard

# Absolute-V-threshold lab (issues #12-#18) FULL SWEEP: all 32 heads across a
# contiguous slice [VTHR_START:VTHR_STOP] of the 288 decode positions
# (contexts 6.8k-134.8k), at the frozen operating point (mirrors
# run_epoch_trace_full.sbatch). Fixed absolute theta grid. Two of these jobs
# (balanced by cumulative context) cover the full spectrum.
set -euo pipefail
cd /gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention
module purge
module load python/3.10.4
export LD_LIBRARY_PATH="/sw/pkgs/arc/python/3.10.4/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

START="${VTHR_START:?set VTHR_START}"
STOP="${VTHR_STOP:?set VTHR_STOP}"
TAG="${VTHR_TAG:-s${START}_${STOP}}"
ROOT="attention_efficiency_result/v_threshold_lab_20260711/sweep_${TAG}"
LAB="${ROOT}/lab"; RUN="${ROOT}/gold_run"
mkdir -p "$LAB" "$RUN"
GRID="1e-14,3e-14,1e-13,3e-13,1e-12,3e-12,1e-11,2e-11,3e-11,6e-11,1e-10,2e-10,3e-10,1e-9,3e-9,1e-8,3e-8,1e-7,3e-7,1e-6"

HEADS=$(python3 -c "print(','.join(str(i) for i in range(32)))")
echo "[sweep ${TAG}] positions [${START}:${STOP}) heads=0..31 grid_n=20"

/usr/bin/time -v .venv/bin/python benchmark/selector_eval/runners/run_joint_kv_budget_policy_eval.py \
  --qkv_trace attention_efficiency_result/real_qkv_llama31_l16_6838_g131072_q288_window32_graphall_s16.npz \
  --x_trace attention_efficiency_result/real_xtrace_llama31_8b_layer16_8k_g131072_sampled_maskstop.npz \
  --model_snapshot .hf_cache/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659 \
  --device cpu --decode_lengths all --max_qidx_per_decode 0 \
  --qidx_slice_start "$START" --qidx_slice_stop "$STOP" --heads "$HEADS" \
  --page_size 5632 --static_prefix 128 --static_suffix 128 \
  --subvecs 4 --subbits 8 --kmeans_iters 3 --value_subvecs 1 --value_subbits 4 \
  --kv_storage_format fp16 --key_bytes 2 --value_bytes 2 \
  --code_stat_bytes 2 --selector_index_bytes 4 --survivor_logit_bytes 2 \
  --selector_mode fullscan --nprobes 512 --page_scan_frac 1.0 \
  --k_budgets 4096,8192,14336,32768 --v_budgets 1024,2048,4096,6144,8192,12288,16384 \
  --k_budget_fracs 0.10,0.30,0.50,0.70,0.90,1.0 --v_budget_fracs 0.05,0.10,0.20,0.40,0.60,0.80,1.0 \
  --policies k_first_alternating --start_strategies proxy_mass_m0p9 --stability_thresholds 0.004 \
  --threshold_mode budget_delta_frac --threshold_reference_frac 0.2 \
  --threshold_scale_shape sqrt --threshold_min_scale 0.0 --threshold_max_scale 1.5 \
  --precision_k_hi_frac 0.1 --precision_v_hi_frac 0.1 --precision_lo_mode int8 --precision_lo_bits 8 \
  --precision_split_freeze start --v_selection_rules global_residual_risk --v_local_block_size 1024 \
  --logit_buffer_bits 8 --logit_buffer_format e4m3 --tail_score_calibration none \
  --score_proxy_variants baseline --temporal_reuse_mode frozen --temporal_reuse_budget ladder \
  --v_threshold_lab_dir "$LAB" --v_threshold_grid "$GRID" --output_dir "$RUN"
echo "[sweep ${TAG}] done"
