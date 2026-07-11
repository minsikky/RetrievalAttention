#!/usr/bin/env bash
#SBATCH --job-name=fspq-r5-id
#SBATCH --account=zhengya0
#SBATCH --partition=gpu_mig40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60000m
#SBATCH --time=02:00:00
#SBATCH --output=logs/fspq_tokpar_round5_identity_%j.log
set -euo pipefail

REPO=/gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention
cd "${REPO}"

TASK_NAME=qa_1
OUTPUT_ROOT="${REPO}/benchmark_suite_result/fspq_tokpar_round5"
BASE_ROOT="${REPO}/benchmark_suite_result/fspq_tokpar_round4"
RUN_BASE=qa1_32k_n4_tokpar_batched_53323928
RUN_CURRENT="qa1_32k_n4_tokpar_round5_${SLURM_JOB_ID}"
DATA_FILE="${REPO}/benchmark_suite_result/ktcache_ab_20260709/qa1_32k_n4_ktON/data/${TASK_NAME}/validation.jsonl"

[ -s "${DATA_FILE}" ] || { echo "missing data ${DATA_FILE}" >&2; exit 1; }
[ -s "${BASE_ROOT}/${RUN_BASE}/pred/${TASK_NAME}.jsonl" ] || {
  echo "missing cached baseline ${BASE_ROOT}/${RUN_BASE}" >&2
  exit 1
}

echo "[round5-identity] main=$(git rev-parse --short HEAD) run=${RUN_CURRENT}"
env \
  TASK_NAME="${TASK_NAME}" \
  CONTEXT_LEN=32768 \
  NUM_SAMPLES=4 \
  RUN_NAME="${RUN_CURRENT}" \
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
  SELECTOR_PQ_JOINT_FROZENSIM_FUSED_VPREFIX_TOKPAR_BATCHED=1 \
  SELECTOR_PQ_JOINT_FROZENSIM_FUSED_VPREFIX_TOKPAR_BATCHED_GLUE=${R5_GLUE:-1} \
  SELECTOR_PQ_JOINT_FROZENSIM_FUSED_VPREFIX_TOKPAR_BATCHED_PARTIALS_VEC4=${R5_VEC4:-1} \
  SELECTOR_PQ_JOINT_MEMORY_TRACE=0 \
  SELECTOR_PQ_JOINT_TIME_TRACE=0 \
  SKIP_RULER_EVAL=1 \
  bash scripts/run_frontier_ruler_batched_one.sh

module purge 2>/dev/null || true
module load python/3.10.4
export LD_LIBRARY_PATH="/sw/pkgs/arc/python/3.10.4/lib:${LD_LIBRARY_PATH:-}"

"${REPO}/.venv/bin/python" - \
  "${BASE_ROOT}/${RUN_BASE}" "${OUTPUT_ROOT}/${RUN_CURRENT}" "${TASK_NAME}" <<'PY'
import json
import sys
from pathlib import Path

base, current, task = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
diffs = []


def timing(key):
    key = str(key).lower()
    return "second" in key or key.endswith("_time") or "tokens_per" in key or "ms_per" in key or key in {"time", "elapsed"}


def compare(path, a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            if timing(key):
                continue
            if key not in a or key not in b:
                diffs.append(f"{path}.{key}: missing")
            else:
                compare(f"{path}.{key}", a[key], b[key])
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            diffs.append(f"{path}: length {len(a)} != {len(b)}")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                compare(f"{path}[{i}]", x, y)
    elif a != b:
        diffs.append(f"{path}: {a!r} != {b!r}")


def pred_rows(root):
    return [json.loads(line) for line in (root / "pred" / f"{task}.jsonl").open() if line.strip()]


compare("pred", {r["index"]: r for r in pred_rows(base)}, {r["index"]: r for r in pred_rows(current)})
base_summary = json.loads((base / "summary" / f"{task}.json").read_text())
current_summary = json.loads((current / "summary" / f"{task}.json").read_text())
for field in ("cost_proxy", "cost_proxy_aggregate"):
    compare(f"summary.{field}", base_summary.get(field), current_summary.get(field))
if diffs:
    print(f"TOKPAR-ROUND5-IDENTITY: FAIL ({len(diffs)} differences)")
    for diff in diffs[:50]:
        print(f"  {diff}")
    raise SystemExit(1)
print("TOKPAR-ROUND5-IDENTITY: PASS")
PY

echo "TOKPAR-ROUND5-IDENTITY-RUN=${RUN_CURRENT}"
