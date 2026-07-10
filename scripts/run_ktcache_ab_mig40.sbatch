#!/usr/bin/env bash
#SBATCH --job-name=ktcache-ab
#SBATCH --account=zhengya0
#SBATCH --partition=gpu_mig40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60000m
#SBATCH --time=03:00:00
#SBATCH --output=logs/ktcache_ab_%j.log
set -euo pipefail

# A/B test for FRONTIER_DENSE_KEY_T_CACHE (transposed-float-K decode cache).
# Both arms run on the SAME GPU inside ONE job, then a byte/field-level diff
# decides whether disabling the cache changes the (algorithm-defined) frozen-sim
# selector outputs. Arm A = cache ON (production), Arm B = cache OFF.
#
# Rationale for a small-context config: cache ON must fit a 40GB MIG slice
# (weights ~16G + KV ~4.3G + K^T cache ~2.1G at these sizes), so this uses
# qa_1 @ 32768, n=4 instead of the phase-a 128k/n16 grid. Everything else
# mirrors launch_frozen_sim_phase_a.sh COMMON_ENV exactly.
#
# Submit with: sbatch scripts/run_ktcache_ab_mig40.sbatch   (do NOT bash it)

REPO=/gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention
cd "${REPO}"

TASK_NAME=qa_1
CONTEXT_LEN=32768
NUM_SAMPLES=4
OUTPUT_ROOT=benchmark_suite_result/ktcache_ab_20260709
RUN_ON=qa1_32k_n4_ktON
RUN_OFF=qa1_32k_n4_ktOFF

# --- Frozen-sim phase-a environment ---------------------------------------
# Replicated verbatim from launch_frozen_sim_phase_a.sh COMMON_ENV:
#   JOINT_KV_STABILITY_THRESHOLD=0.004  LOGIT_BUFFER_FORMAT=e4m3
#   JOINT_KV_PRECISION_TIERS=1  OUTPUT_ROOT/TASK_NAME/CONTEXT_LEN/NUM_SAMPLES
# Overridden for the A/B (reason): CONTEXT_LEN 131072->32768, NUM_SAMPLES 16->4
#   so cache-ON fits the 40GB slice; OUTPUT_ROOT -> the A/B directory.
# COMMON_ENV in the launcher sets nothing else and unsets nothing.
# Added per task spec (memory hygiene / trace; zero numeric impact):
#   PYTORCH_CUDA_ALLOC_CONF, PREFILL_CHUNK_SIZE, FRONTIER_EMPTY_CACHE_AFTER_PREFILL_CHUNK,
#   SELECTOR_PQ_JOINT_MEMORY_TRACE.
COMMON_ENV=(
  TASK_NAME="${TASK_NAME}"
  CONTEXT_LEN="${CONTEXT_LEN}"
  NUM_SAMPLES="${NUM_SAMPLES}"
  JOINT_KV_STABILITY_THRESHOLD=0.004
  LOGIT_BUFFER_FORMAT=e4m3
  JOINT_KV_PRECISION_TIERS=1
  OUTPUT_ROOT="${OUTPUT_ROOT}"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  PREFILL_CHUNK_SIZE=8192
  FRONTIER_EMPTY_CACHE_AFTER_PREFILL_CHUNK=1
  SELECTOR_PQ_JOINT_MEMORY_TRACE=1
)

echo "[ktcache-ab] repo=${REPO}"
echo "[ktcache-ab] task=${TASK_NAME} ctx=${CONTEXT_LEN} n=${NUM_SAMPLES}"
echo "[ktcache-ab] out_root=${OUTPUT_ROOT}"

# --- Arm A: cache ON (production MAX_GB) ----------------------------------
echo "[ktcache-ab] === ARM A: FRONTIER_DENSE_KEY_T_CACHE=1 -> ${RUN_ON} ==="
env "${COMMON_ENV[@]}" \
  RUN_NAME="${RUN_ON}" \
  FRONTIER_DENSE_KEY_T_CACHE=1 \
  FRONTIER_DENSE_KEY_T_CACHE_MAX_GB=12.0 \
  bash scripts/run_frontier_ruler_batched_one.sh

# --- Arm B: cache OFF -----------------------------------------------------
# Reuse Arm A's prepared data so the two arms consume byte-identical input
# tokens (isolates the K^T cache as the only variable). Data prep is itself
# deterministic, so this is belt-and-suspenders.
DATA_A="${REPO}/${OUTPUT_ROOT}/${RUN_ON}/data/${TASK_NAME}/validation.jsonl"
echo "[ktcache-ab] === ARM B: FRONTIER_DENSE_KEY_T_CACHE=0 -> ${RUN_OFF} ==="
echo "[ktcache-ab] arm B reuses data: ${DATA_A}"
env "${COMMON_ENV[@]}" \
  RUN_NAME="${RUN_OFF}" \
  DATA_FILE_OVERRIDE="${DATA_A}" \
  FRONTIER_DENSE_KEY_T_CACHE=0 \
  bash scripts/run_frontier_ruler_batched_one.sh

# --- Diff step ------------------------------------------------------------
echo "[ktcache-ab] === DIFF ==="
module purge 2>/dev/null || true
module load python/3.10.4
export LD_LIBRARY_PATH="/sw/pkgs/arc/python/3.10.4/lib:${LD_LIBRARY_PATH:-}"

"${REPO}/.venv/bin/python" - "${OUTPUT_ROOT}" "${RUN_ON}" "${RUN_OFF}" "${TASK_NAME}" <<'PY'
import csv
import json
import sys
from pathlib import Path

output_root, run_on, run_off, task = sys.argv[1:5]


def arm_paths(run_name):
    base = Path(output_root) / run_name
    return {
        "pred": base / "pred" / f"{task}.jsonl",
        "summary": base / "summary" / f"{task}.json",
        "score": base / "pred" / f"summary-{task}.csv",
    }


A = arm_paths(run_on)  # cache ON
B = arm_paths(run_off)  # cache OFF

diffs = []  # human-readable difference descriptions


def load_jsonl(path):
    rows = []
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def is_wallclock_key(key):
    k = str(key).lower()
    return (
        "second" in k
        or "per_second" in k
        or k.endswith("_time")
        or k == "time"
        or "tokens_per" in k
        or "ms_per" in k
        or k == "elapsed"
    )


def compare_values(path, a, b):
    """Recursively compare, skipping wall-clock leaves. Appends to diffs."""
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            if is_wallclock_key(key):
                continue
            if key not in a:
                diffs.append(f"{path}.{key}: missing in ON")
            elif key not in b:
                diffs.append(f"{path}.{key}: missing in OFF")
            else:
                compare_values(f"{path}.{key}", a[key], b[key])
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            diffs.append(f"{path}: list len ON={len(a)} OFF={len(b)}")
        else:
            for i, (av, bv) in enumerate(zip(a, b)):
                compare_values(f"{path}[{i}]", av, bv)
    else:
        if a != b:
            diffs.append(f"{path}: ON={a!r} OFF={b!r}")


# (a) per-sample generated text (pred) + deterministic per-sample counters
pa = load_jsonl(A["pred"])
pb = load_jsonl(B["pred"])
if pa is None or pb is None:
    diffs.append(f"pred jsonl missing: ON={A['pred']}({pa is not None}) OFF={B['pred']}({pb is not None})")
elif len(pa) != len(pb):
    diffs.append(f"pred sample count: ON={len(pa)} OFF={len(pb)}")
else:
    by_idx_b = {r.get("index"): r for r in pb}
    for ra in pa:
        idx = ra.get("index")
        rb = by_idx_b.get(idx)
        if rb is None:
            diffs.append(f"sample index {idx} present ON but missing OFF")
            continue
        # Generated text MUST be identical.
        if ra.get("pred") != rb.get("pred"):
            diffs.append(
                f"sample {idx} pred differs:\n    ON ={ra.get('pred')!r}\n    OFF={rb.get('pred')!r}"
            )
        # Deterministic per-sample counters from the timing dict (skip seconds).
        ta = ra.get("timing", {}) or {}
        tb = rb.get("timing", {}) or {}
        for key in sorted(set(ta) | set(tb)):
            if is_wallclock_key(key):
                continue
            if ta.get(key) != tb.get(key):
                diffs.append(f"sample {idx} timing.{key}: ON={ta.get(key)!r} OFF={tb.get(key)!r}")

# (b) summary MB / reads / counter fields (cost_proxy + cost_proxy_aggregate)
sa = json.loads(A["summary"].read_text()) if A["summary"].exists() else None
sb = json.loads(B["summary"].read_text()) if B["summary"].exists() else None
if sa is None or sb is None:
    diffs.append(f"summary json missing: ON={A['summary']}({sa is not None}) OFF={B['summary']}({sb is not None})")
else:
    for sub in ("cost_proxy", "cost_proxy_aggregate"):
        compare_values(f"summary.{sub}", sa.get(sub), sb.get(sub))

# (c) RULER score csv (Score / Nulls)
def load_score_csv(path):
    if not path.exists():
        return None
    rows = {}
    with path.open(newline="") as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                rows[row[0]] = row[1]
    return rows


ca = load_score_csv(A["score"])
cb = load_score_csv(B["score"])
if ca is None or cb is None:
    diffs.append(f"score csv missing: ON={A['score']}({ca is not None}) OFF={B['score']}({cb is not None})")
else:
    for key in sorted(set(ca) | set(cb)):
        if ca.get(key) != cb.get(key):
            diffs.append(f"score.{key}: ON={ca.get(key)!r} OFF={cb.get(key)!r}")

print("=" * 72)
print(f"[ktcache-ab] compared ON={run_on}  vs  OFF={run_off}")
print(f"[ktcache-ab] pred={A['pred']}")
print(f"[ktcache-ab]      {B['pred']}")
if not diffs:
    print("KTCACHE-AB: IDENTICAL")
    sys.exit(0)
else:
    print(f"KTCACHE-AB: DIFFERS ({len(diffs)} field(s))")
    for d in diffs:
        print(f"  - {d}")
    sys.exit(1)
PY

echo "[ktcache-ab] done"
