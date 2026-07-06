#!/usr/bin/env bash
#SBATCH --job-name=frontier-cuda-build
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000m
#SBATCH --time=00:30:00
#SBATCH --account=zhengya98
#SBATCH --partition=standard
set -euo pipefail

# Compile the frontier CUDA extension on a Slurm compute node without running GPU tests.

cd /gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention

OUTPUT_DIR="${OUTPUT_DIR:-cuda_unit_result/frontier_cuda_ext_build_only_$(date +%Y%m%d_%H%M%S)}"
HF_VENV_DIR="${HF_VENV_DIR:-.venv}"
mkdir -p "${OUTPUT_DIR}"

module purge
module load python/3.10.4
module load cuda/12.8.1
source "${HF_VENV_DIR}/bin/activate"

export LD_LIBRARY_PATH="$PWD/${HF_VENV_DIR}/lib/python3.10/site-packages/torch/lib:/sw/pkgs/arc/python/3.10.4/lib:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
export PYTHONPATH="$PWD/benchmark/selector_eval/cuda_ext:${PYTHONPATH:-}"
export MAX_JOBS="${MAX_JOBS:-${SLURM_CPUS_PER_TASK:-4}}"
LOCK_FILE="${CUDA_EXT_BUILD_LOCK:-${PWD}/.codex/selector_pq_build.lock}"
mkdir -p "$(dirname "${LOCK_FILE}")"

start_ts="$(date +%s)"
status="passed"

set +e
(
  set -e
  echo "[cuda_build] host=$(hostname)"
  echo "[cuda_build] started=$(date --iso-8601=seconds)"
  echo "[cuda_build] output_dir=${OUTPUT_DIR}"
  echo "[cuda_build] python=$(which python)"
  python -V
  nvcc --version
  (
    flock 200
    cd benchmark/selector_eval/cuda_ext
    python setup.py build_ext --inplace
  ) 200>"${LOCK_FILE}"
) >"${OUTPUT_DIR}/build.log" 2>&1
rc=$?
set -e

if [ "${rc}" -ne 0 ]; then
  status="failed"
fi
end_ts="$(date +%s)"

"${HF_VENV_DIR}/bin/python" - <<PY
import json
from pathlib import Path

payload = {
    "kind": "cuda_ext_build_only",
    "status": "${status}",
    "return_code": int("${rc}"),
    "elapsed_seconds": int("${end_ts}") - int("${start_ts}"),
    "slurm_job_id": "${SLURM_JOB_ID:-manual}",
    "partition": "${SLURM_JOB_PARTITION:-standard}",
    "account": "zhengya98",
    "log": "build.log",
}
Path("${OUTPUT_DIR}/summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n")
PY

cat "${OUTPUT_DIR}/build.log"
cat "${OUTPUT_DIR}/summary.json"
exit "${rc}"
