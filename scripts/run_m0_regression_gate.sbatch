#!/usr/bin/env bash
#SBATCH --job-name=m0-reggate
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96000m
#SBATCH --time=10:00:00
#SBATCH --account=zhengya0
#SBATCH --partition=standard

# Issue #24 regression gate: run the EXTENDED dumper (pass-1 flag default OFF)
# on the ORIGINAL golden qkv trace at qidx 287 (ctx 134,838), 32 heads, and
# diff every epoch_q287_h*.npz array-for-array against the committed golden
# benchmark/selector_eval/golden_vectors/epoch_trace_20260710/. Proves the
# extension does not drift the existing contract artifact.
set -euo pipefail
cd /gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention/worktrees/m0-trace
module purge
module load python/3.10.4
export LD_LIBRARY_PATH="/sw/pkgs/arc/python/3.10.4/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

WORK=/scratch/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention/m0_trace_work/reggate
rm -rf "$WORK"; mkdir -p "$WORK/trace" "$WORK/run"
HEADS=$(python3 -c "print(','.join(str(i) for i in range(32)))")

.venv/bin/python benchmark/selector_eval/runners/run_joint_kv_budget_policy_eval.py \
  --qkv_trace attention_efficiency_result/real_qkv_llama31_l16_6838_g131072_q288_window32_graphall_s16.npz \
  --x_trace attention_efficiency_result/real_xtrace_llama31_8b_layer16_8k_g131072_sampled_maskstop.npz \
  --model_snapshot .hf_cache/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659 \
  --device cpu --decode_lengths 128000 --max_qidx_per_decode 1 --heads "$HEADS" \
  --page_size 5632 --static_prefix 128 --static_suffix 128 \
  --subvecs 4 --subbits 8 --kmeans_iters 3 --value_subvecs 1 --value_subbits 4 \
  --kv_storage_format fp16 --key_bytes 2 --value_bytes 2 --code_stat_bytes 2 --selector_index_bytes 4 --survivor_logit_bytes 2 \
  --selector_mode fullscan --nprobes 512 --page_scan_frac 1.0 \
  --k_budgets 4096,8192,14336,32768 --v_budgets 1024,2048,4096,6144,8192,12288,16384 \
  --k_budget_fracs 0.10,0.30,0.50,0.70,0.90,1.0 --v_budget_fracs 0.05,0.10,0.20,0.40,0.60,0.80,1.0 \
  --policies k_first_alternating --start_strategies proxy_mass_m0p9 --stability_thresholds 0.004 \
  --threshold_mode budget_delta_frac --threshold_reference_frac 0.2 --threshold_scale_shape sqrt --threshold_min_scale 0.0 --threshold_max_scale 1.5 \
  --precision_k_hi_frac 0.1 --precision_v_hi_frac 0.1 --precision_lo_mode int8 --precision_lo_bits 8 --precision_split_freeze start \
  --v_selection_rules global_residual_risk --v_local_block_size 1024 \
  --logit_buffer_bits 8 --logit_buffer_format e4m3 --tail_score_calibration none --score_proxy_variants baseline \
  --temporal_reuse_mode frozen --temporal_reuse_budget ladder --gqa_union_stats \
  --epoch_trace_dir "$WORK/trace" --output_dir "$WORK/run"

echo "[reggate] diff vs committed golden"
.venv/bin/python - "$WORK/trace" benchmark/selector_eval/golden_vectors/epoch_trace_20260710 <<'PY'
import sys, numpy as np, pathlib
new, gold = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
# Documented exception list: fields op-fix HEAD (ad03525) already emits that
# postdate the Jul-10 golden (Phase-2 physical-replay fields, trace_format
# _version 2). These are NOT task-#40 changes -- the golden simply predates
# them. pass1_* is OFF in this gate (default -2), so it must not appear.
ALLOWED_EXTRA = {"trace_format_version", "start_v_tokens", "k_hi_tokens", "k_hi_tokens_start"}
bad = 0; nfiles = 0
for gf in sorted(gold.glob("epoch_q287_h*.npz")):
    nf = new / gf.name
    if not nf.exists():
        print("MISSING", gf.name); bad += 1; continue
    a = np.load(nf, allow_pickle=True); b = np.load(gf, allow_pickle=True)
    nfiles += 1
    ka, kb = set(a.files), set(b.files)
    missing = kb - ka
    extra = ka - kb
    if missing:
        print(gf.name, "MISSING GOLDEN KEYS", sorted(missing)); bad += 1; continue
    unexpected = extra - ALLOWED_EXTRA
    if unexpected:
        print(gf.name, "UNEXPECTED EXTRA KEYS", sorted(unexpected)); bad += 1; continue
    # Compare every GOLDEN (shared) key array-for-array.
    for k in b.files:
        va, vb = a[k], b[k]
        if va.dtype.kind in "US" or vb.dtype.kind in "US":
            if str(va) != str(vb):
                print(gf.name, k, "STR DIFF"); bad += 1
            continue
        if not np.array_equal(np.asarray(va), np.asarray(vb)):
            # array_equal treats NaN!=NaN; allow identical-NaN layout
            if va.shape == vb.shape and np.array_equal(np.nan_to_num(va, nan=0.0, posinf=1e30, neginf=-1e30),
                                                       np.nan_to_num(vb, nan=0.0, posinf=1e30, neginf=-1e30)) \
               and np.array_equal(np.isnan(va), np.isnan(vb)):
                continue
            print(gf.name, k, "ARRAY DIFF"); bad += 1
print(f"[reggate] compared {nfiles} files; mismatches={bad}")
sys.exit(1 if bad else 0)
PY
echo "[reggate] PASS (byte/array-identical to golden)"
