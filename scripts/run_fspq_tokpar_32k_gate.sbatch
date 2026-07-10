#!/usr/bin/env bash
#SBATCH --job-name=fspq-tokpar-32k
#SBATCH --account=zhengya0
#SBATCH --partition=gpu_mig40,spgpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60000m
#SBATCH --time=02:00:00
#SBATCH --output=logs/fspq_tokpar_32k_%j.log
set -euo pipefail

REPO=/gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention
cd "${REPO}"

: "${BASE_RUN:?submit with BASE_RUN=<passing identity run name>}"
TASK_NAME=qa_1
OUTPUT_ROOT="${REPO}/benchmark_suite_result/fspq_tokpar_round3"
RUN_TOKPAR="qa1_32k_n4_tokpar_${SLURM_JOB_ID}"
DATA_FILE="${REPO}/benchmark_suite_result/ktcache_ab_20260709/qa1_32k_n4_ktON/data/${TASK_NAME}/validation.jsonl"

[ -s "${DATA_FILE}" ] || { echo "missing data ${DATA_FILE}" >&2; exit 1; }
[ -s "${OUTPUT_ROOT}/${BASE_RUN}/pred/${TASK_NAME}.jsonl" ] || {
  echo "missing identity baseline ${OUTPUT_ROOT}/${BASE_RUN}" >&2
  exit 1
}

echo "[tokpar-32k] main=$(git rev-parse --short HEAD) base=${BASE_RUN} run=${RUN_TOKPAR}"
env \
  TASK_NAME="${TASK_NAME}" \
  CONTEXT_LEN=32768 \
  NUM_SAMPLES=4 \
  RUN_NAME="${RUN_TOKPAR}" \
  JOINT_KV_STABILITY_THRESHOLD=0.004 \
  LOGIT_BUFFER_FORMAT=e4m3 \
  JOINT_KV_PRECISION_TIERS=1 \
  FRONTIER_DENSE_KEY_T_CACHE=0 \
  OUTPUT_ROOT="${OUTPUT_ROOT}" \
  DATA_FILE_OVERRIDE="${DATA_FILE}" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  PREFILL_CHUNK_SIZE=8192 \
  FRONTIER_EMPTY_CACHE_AFTER_PREFILL_CHUNK=1 \
  SELECTOR_PQ_JOINT_FROZENSIM_FUSED_VPREFIX=1 \
  SELECTOR_PQ_JOINT_FROZENSIM_FUSED_RISK_SORT=1 \
  SELECTOR_PQ_JOINT_FROZENSIM_FUSED_VPREFIX_TOKPAR=1 \
  SELECTOR_PQ_JOINT_MEMORY_TRACE=0 \
  SELECTOR_PQ_JOINT_TIME_TRACE=0 \
  SKIP_RULER_EVAL=1 \
  bash scripts/run_frontier_ruler_batched_one.sh

module purge 2>/dev/null || true
module load python/3.10.4
export LD_LIBRARY_PATH="/sw/pkgs/arc/python/3.10.4/lib:${LD_LIBRARY_PATH:-}"

"${REPO}/.venv/bin/python" - \
  "${OUTPUT_ROOT}" "${BASE_RUN}" "${RUN_TOKPAR}" "${TASK_NAME}" "${DATA_FILE}" <<'PY'
import json
import sys
from pathlib import Path

root, run_a, run_b, task, data_path = sys.argv[1:6]
root = Path(root)
data = [json.loads(line) for line in open(data_path)]


def load(run):
    path = root / run / "pred" / f"{task}.jsonl"
    return {row["index"]: row for row in (json.loads(line) for line in path.open())}


def score(row):
    golds = data[row["index"]].get("outputs") or []
    return any(str(gold).lower() in str(row["pred"]).lower() for gold in golds)


a, b = load(run_a), load(run_b)
if set(a) != set(b):
    print("TOKPAR-32K: FAIL (sample sets differ)")
    raise SystemExit(1)
score_ok = True
for index in sorted(a):
    sa, sb = score(a[index]), score(b[index])
    print(
        f"sample {index}: default_score={sa} tokpar_score={sb} "
        f"pred_forked={a[index]['pred'] != b[index]['pred']}"
    )
    score_ok &= sa == sb

summary_a = json.loads((root / run_a / "summary" / f"{task}.json").read_text())
summary_b = json.loads((root / run_b / "summary" / f"{task}.json").read_text())
worst = (0.0, "")


def timing_key(key):
    key = str(key).lower()
    return "second" in key or key.endswith("_time") or "tokens_per" in key or "ms_per" in key


def walk(path, x, y):
    global worst
    if isinstance(x, dict) and isinstance(y, dict):
        for key in sorted(set(x) & set(y)):
            if not timing_key(key):
                walk(f"{path}.{key}", x[key], y[key])
    elif isinstance(x, list) and isinstance(y, list):
        for i, (xx, yy) in enumerate(zip(x, y)):
            walk(f"{path}[{i}]", xx, yy)
    elif (
        isinstance(x, (int, float))
        and not isinstance(x, bool)
        and isinstance(y, (int, float))
        and not isinstance(y, bool)
        and x != y
    ):
        drift = abs(float(y) - float(x)) / max(1.0e-12, abs(float(x)))
        if drift > worst[0]:
            worst = (drift, path)


for field in ("cost_proxy", "cost_proxy_aggregate"):
    walk(field, summary_a.get(field), summary_b.get(field))
print(f"worst counter drift: {worst[0] * 100:.3f}% at {worst[1] or 'none'}")
if score_ok and worst[0] < 0.02:
    print("TOKPAR-32K: PASS (scores unchanged, worst counter drift < 2%)")
    raise SystemExit(0)
print(f"TOKPAR-32K: FAIL (score_ok={score_ok}, worst_drift={worst[0] * 100:.3f}%)")
raise SystemExit(1)
PY

echo "TOKPAR-32K-RUN=${RUN_TOKPAR}"
