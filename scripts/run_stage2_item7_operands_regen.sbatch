#!/usr/bin/env bash
#SBATCH --job-name=stage2-item7-operands
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000m
#SBATCH --time=04:00:00
#SBATCH --account=zhengya0
#SBATCH --partition=standard

# Stage-2 golden regen adding issue #7 option-A raw Vcorr operands, ON TOP OF
# the frozen-split baseline (commit aac1593). Identical config to
# run_stage2_frozen_split_regen.sbatch (escalation-only, --precision_split_freeze
# start, E6M12 keys, fp16 commit) -- the runner now ALSO dumps the all-sealed-
# pages V-PQ block (all_*) and the band-operand union (vexact_band_*) so RTL's
# G3 gate can rebuild vcorr_acc_{marginal,hiboundary,settled}_{ref,hw} from
# operands alone. Existing item-7 fields are additive-only (byte-identical
# superset). Verify does BOTH: (a) superset/back-compat vs the frozen-split
# goldens (existing fields must stay bit-identical -> pure add), and (b) the
# G3 operand-rebuild gate (bit-tight, per row, every band/domain).
set -euo pipefail
cd /gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention
module purge
module load python/3.10.4
export LD_LIBRARY_PATH="/sw/pkgs/arc/python/3.10.4/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

ROOT="benchmark_suite_result/stage2_item7_operands_regen_20260707"
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

echo "[item7] verify: superset vs frozen-split goldens + G3 operand-rebuild gate"
.venv/bin/python benchmark/selector_eval/runners/verify_stage2_key_regen.py \
  --frozen benchmark/selector_eval/golden_vectors/stage2_20260707 \
  --regen "$DUMP" \
  --study_e6m12 benchmark_suite_result/v_risk_key_study_20260707/gold_dump_e6m12 \
  --g3_tol 1e-9 \
  --out "${ROOT}/verify_report.json" || echo "[item7] verifier nonzero (inspect report: new fields are +add; G3 must be bit-tight)"

echo "[item7] superset check: existing fields must be byte-identical, operands added"
.venv/bin/python - <<'EOF'
import glob, os, numpy as np

OLD = "benchmark/selector_eval/golden_vectors/stage2_20260707"       # frozen-split goldens (aac1593)
NEW = "benchmark_suite_result/stage2_item7_operands_regen_20260707/gold_dump"

def summarize(pat):
    old = {os.path.basename(p): p for p in glob.glob(os.path.join(OLD, pat))}
    new = {os.path.basename(p): p for p in glob.glob(os.path.join(NEW, pat))}
    names = sorted(set(old) & set(new))
    n_pure_super = 0
    problems = []
    added_fields = set()
    for name in names:
        o = np.load(old[name], allow_pickle=True)
        n = np.load(new[name], allow_pickle=True)
        added = [k for k in n.files if k not in o.files]
        removed = [k for k in o.files if k not in n.files]
        added_fields.update(added)
        moved = []
        for k in o.files:
            if k not in n.files:
                continue
            a, b = o[k], n[k]
            try:
                same = a.shape == b.shape and np.array_equal(a, b)
            except Exception:
                same = str(a) == str(b)
            if not same:
                moved.append(k)
        if removed or moved:
            problems.append((name, {"removed": removed, "moved_existing": moved}))
        else:
            n_pure_super += 1
    print(f"  [{pat}] {n_pure_super}/{len(names)} pure supersets (existing fields byte-identical)")
    if added_fields:
        print(f"  [{pat}] added fields: {sorted(added_fields)}")
    for name, d in problems:
        print(f"  [{pat}] !! {name}: removed={d['removed']} moved_existing={d['moved_existing']}")

summarize("golden2_*.npz")
summarize("page_v_*.npz")
EOF

echo "[item7] G3 summary from report"
.venv/bin/python - <<'EOF'
import json
d = json.load(open("benchmark_suite_result/stage2_item7_operands_regen_20260707/verify_report.json"))
g3 = d.get("g3", {})
print(f"  G3 checked rows: {g3.get('checked')}")
print(f"  G3 max_abs_by_band_domain: {g3.get('max_abs_by_band_domain')}")
print(f"  G3 worst: {g3.get('worst')}")
errs = [r for r in g3.get("rows", []) if r.get("error")]
print(f"  G3 rows with error (> tol): {len(errs)}")
for r in errs[:12]:
    print(f"    {r['row']}: {r['error']}")
print(f"  overall fatal={d.get('fatal')}")
EOF
echo "[item7] done: ${ROOT}/verify_report.json"
