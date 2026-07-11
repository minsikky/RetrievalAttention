#!/usr/bin/env bash
#SBATCH --job-name=s2-union-regen
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96000m
#SBATCH --time=06:00:00
#SBATCH --account=zhengya0
#SBATCH --partition=standard

# Issue #20 item 3: stage-2 golden regeneration under UNION-COMMIT execution
# (ratified 2026-07-10). Template: run_stage2_two_pass_goldens_regen.sbatch
# (escalation-only frozen walk, --precision_split_freeze start, item-7
# operands, adopted Q7.6 two-pass grid) PLUS --gqa_union_commit and the
# deferred union golden dump.
#
# Inventory: standard rows q159/q223/q287 x h{0,8,16,24} (their full GQA
# groups run so the unions exist; only the contract heads dump) PLUS the two
# honest-case rows from the #20 envelope: q137 h0 (kv0) and q137 h5 (kv1).
#
# Gates run at the end:
#   (a) verify_stage2_key_regen.py vs the pre-union blessed set (moves are
#       EXPECTED under the new contract; kept for enumeration + its G3 and
#       two-pass sub-gates which must be clean),
#   (b) union-aware classification: walk-domain fields must be bit-identical
#       to stage2_20260707; only the documented union execution fields move,
#   (c) the ratified numeric quality envelope (verify_union_envelope.py) on
#       the run's own gqa_union_commit.csv.
set -euo pipefail
WT=/gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention/worktrees/union-commit
cd "$WT"
module purge
module load python/3.10.4
export LD_LIBRARY_PATH="/sw/pkgs/arc/python/3.10.4/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

ROOT=/scratch/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention/stage2_union_commit_regen_20260710
DUMP="${ROOT}/gold_dump"
mkdir -p "$DUMP" "${ROOT}/run_std" "${ROOT}/run_q137"

COMMON=(
  --qkv_trace "$WT/attention_efficiency_result/real_qkv_llama31_l16_6838_g131072_q288_window32_graphall_s16.npz"
  --x_trace "$WT/attention_efficiency_result/real_xtrace_llama31_8b_layer16_8k_g131072_sampled_maskstop.npz"
  --model_snapshot "$WT/.hf_cache/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
  --device cpu
  --max_qidx_per_decode 1
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
  --two_pass_cutoff_frac_bits 6 --two_pass_cutoff_int_bits 7
  --logit_buffer_bits 8 --logit_buffer_format e4m3
  --tail_score_calibration none
  --score_proxy_variants baseline
  --temporal_reuse_mode frozen --temporal_reuse_budget ladder
  --gqa_union_stats --gqa_union_commit
  --golden_dump_stage2_dir "$DUMP"
)

echo "[s2-union] run A: standard rows (full groups of h0/8/16/24)"
"$WT/.venv/bin/python" "$WT/benchmark/selector_eval/runners/run_joint_kv_budget_policy_eval.py" \
  "${COMMON[@]}" \
  --decode_lengths 8000,32000,128000 \
  --heads 0,1,2,3,8,9,10,11,16,17,18,19,24,25,26,27 \
  --golden_dump_stage2_heads 0,8,16,24 \
  --output_dir "${ROOT}/run_std"

echo "[s2-union] run B: honest-case rows q137 h0 (kv0) + h5 (kv1)"
"$WT/.venv/bin/python" "$WT/benchmark/selector_eval/runners/run_joint_kv_budget_policy_eval.py" \
  "${COMMON[@]}" \
  --decode_lengths 5162 \
  --heads 0,1,2,3,4,5,6,7 \
  --golden_dump_stage2_heads 0,5 \
  --output_dir "${ROOT}/run_q137"

echo "[s2-union] gate (a): verifier vs pre-union blessed set (moves EXPECTED; G3 + two-pass must be clean)"
"$WT/.venv/bin/python" "$WT/benchmark/selector_eval/runners/verify_stage2_key_regen.py" \
  --frozen "$WT/benchmark/selector_eval/golden_vectors/stage2_20260707" \
  --regen "$DUMP" \
  --g3_tol 1e-9 \
  --two_pass_tol 0.0 \
  --out "${ROOT}/verify_report.json" || echo "[s2-union] verifier nonzero (expected: union contract moves execution fields)"

echo "[s2-union] gate (b): union-aware classification of the verifier report"
"$WT/.venv/bin/python" "$WT/scripts/classify_union_golden_moves.py" \
  --report "${ROOT}/verify_report.json"

echo "[s2-union] gate (c): ratified quality envelope"
"$WT/.venv/bin/python" "$WT/scripts/verify_union_envelope.py" \
  --csv "${ROOT}/run_std/gqa_union_commit.csv" "${ROOT}/run_q137/gqa_union_commit.csv"

echo "[s2-union] inventory:"
ls -la "$DUMP"
echo "[s2-union] ALL GATES DONE -> $ROOT"
