#!/usr/bin/env bash
#SBATCH --job-name=compact-vpq-128k
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=160000m
#SBATCH --time=1-00:00:00
#SBATCH --account=zhengya98
#SBATCH --partition=spgpu
#SBATCH --gpus-per-node=1
set -euo pipefail

cd /gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-benchmark_suite_result/compact_vpq_128k_profile_${RUN_STAMP}}"
mkdir -p "${OUTPUT_ROOT}"

module purge
module load python/3.10.4
module load cuda/12.8.1

export HF_VENV_DIR="${HF_VENV_DIR:-.venv_cu128}"
export HF_EXTRA_PYTHONPATH="${HF_EXTRA_PYTHONPATH:-.hf_pydeps_cu128}"
source "${HF_VENV_DIR}/bin/activate"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PWD/benchmark/selector_eval/cuda_ext:${PYTHONPATH:-}"
if [ -n "${HF_EXTRA_PYTHONPATH}" ]; then
  export PYTHONPATH="$PWD/${HF_EXTRA_PYTHONPATH}:${PYTHONPATH}"
fi
export LD_LIBRARY_PATH="$PWD/${HF_VENV_DIR}/lib/python3.10/site-packages/torch/lib:/sw/pkgs/arc/python/3.10.4/lib:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;8.6;12.0}"
export MAX_JOBS="${MAX_JOBS:-${SLURM_CPUS_PER_TASK:-6}}"
export CUDA_EXT_BUILD_LOCK="${CUDA_EXT_BUILD_LOCK:-${PWD}/.codex/selector_pq_build.lock}"
mkdir -p "$(dirname "${CUDA_EXT_BUILD_LOCK}")"

echo "[compact_vpq_128k] host=$(hostname)"
echo "[compact_vpq_128k] started=$(date --iso-8601=seconds)"
echo "[compact_vpq_128k] output_root=${OUTPUT_ROOT}"
echo "[compact_vpq_128k] python=$(which python)"
python -V
nvcc --version
nvidia-smi || true

start_ts="$(date +%s)"
status="passed"
rc=0

set +e
(
  set -e
  echo "[compact_vpq_128k] build extension"
  (
    flock 200
    cd benchmark/selector_eval/cuda_ext
    python setup.py build_ext --inplace
  ) 200>"${CUDA_EXT_BUILD_LOCK}"

  echo "[compact_vpq_128k] run compact V-PQ CUDA unit coverage"
  python benchmark/selector_eval/cuda_ext/test_gpu_vpq_helpers.py

  echo "[compact_vpq_128k] run 128K frontier RULER profile"
  export FRONTIER_CANONICAL_GPU=1
  export SELECTOR_PQ_JOINT_COMPACT_VPQ_RISK_PREFIX=1
  export SELECTOR_PQ_JOINT_NATIVE_VPQ_BASE=1
  export SELECTOR_PQ_JOINT_MEMORY_TRACE=1
  export SELECTOR_PQ_JOINT_WALL_PROFILE=1
  export PROFILE_NATIVE_OPS=0
  export TASK_NAME="${TASK_NAME:-niah_single_1}"
  export CONTEXT_LEN="${CONTEXT_LEN:-131072}"
  export NUM_SAMPLES="${NUM_SAMPLES:-1}"
  export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
  export OUTPUT_ROOT="${OUTPUT_ROOT}/ruler"
  export RUN_NAME="${RUN_NAME:-compact_vpq_${TASK_NAME}_${CONTEXT_LEN}_n${NUM_SAMPLES}}"
  export PAGE_SIZE="${PAGE_SIZE:-5632}"
  export VALUE_PQ_GROUP_PAGES="${VALUE_PQ_GROUP_PAGES:-1}"
  export PREFILL_CHUNK_SIZE="${PREFILL_CHUNK_SIZE:-0}"
  export STAGE_MODEL_TO_TMP="${STAGE_MODEL_TO_TMP:-0}"
  bash scripts/run_frontier_ruler_batched_one.sh
) >"${OUTPUT_ROOT}/compact_vpq_128k.log" 2>&1
rc=$?
set -e

if [ "${rc}" -ne 0 ]; then
  status="failed"
fi
end_ts="$(date +%s)"

python - <<PY
import json
from pathlib import Path

root = Path("${OUTPUT_ROOT}")
payload = {
    "kind": "compact_vpq_128k_profile",
    "status": "${status}",
    "return_code": int("${rc}"),
    "elapsed_seconds": int("${end_ts}") - int("${start_ts}"),
    "slurm_job_id": "${SLURM_JOB_ID:-manual}",
    "partition": "${SLURM_JOB_PARTITION:-spgpu}",
    "account": "zhengya98",
    "log": "compact_vpq_128k.log",
    "ruler_summary_glob": "ruler/*/summary/*.json",
}
root.joinpath("summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

tail -n 240 "${OUTPUT_ROOT}/compact_vpq_128k.log" || true
cat "${OUTPUT_ROOT}/summary.json"
exit "${rc}"
