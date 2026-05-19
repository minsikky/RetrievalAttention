#!/usr/bin/env bash
#SBATCH --job-name=frontier-cuda-tests
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64000m
#SBATCH --time=00:30:00
#SBATCH --account=zhengya98
#SBATCH --partition=spgpu
#SBATCH --gpus-per-node=1
set -euo pipefail

# Build and test the frontier CUDA selector/V-PQ extension on a GPU node.

cd /gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention

OUTPUT_DIR="${OUTPUT_DIR:-cuda_unit_result/frontier_cuda_unit_tests_20260516}"
mkdir -p "${OUTPUT_DIR}"

module purge
module load python/3.10.4
module load cuda/12.8.1
source .venv/bin/activate

export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.10/site-packages/torch/lib:/sw/pkgs/arc/python/3.10.4/lib:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;8.6;9.0}"
export PYTHONPATH="$PWD/benchmark/selector_eval/cuda_ext:${PYTHONPATH:-}"
export MAX_JOBS="${MAX_JOBS:-4}"

start_ts="$(date +%s)"
status="passed"

set +e
(
  set -e
  echo "[cuda_unit] host=$(hostname)"
  echo "[cuda_unit] started=$(date --iso-8601=seconds)"
  echo "[cuda_unit] output_dir=${OUTPUT_DIR}"
  echo "[cuda_unit] python=$(which python)"
  python -V
  nvidia-smi || true

  cd benchmark/selector_eval/cuda_ext
  python setup.py build_ext --inplace
  cd ../../..

  .venv/bin/python benchmark/selector_eval/cuda_ext/test_fullscan_pq_topk.py
  .venv/bin/python benchmark/selector_eval/cuda_ext/test_gpu_vpq_helpers.py
  .venv/bin/python benchmark/selector_eval/cuda_ext/test_online_page_append.py
) >"${OUTPUT_DIR}/unit_tests.log" 2>&1
rc=$?
set -e

if [ "${rc}" -ne 0 ]; then
  status="failed"
fi
end_ts="$(date +%s)"

.venv/bin/python - <<PY
import json
from pathlib import Path

payload = {
    "kind": "cuda_unit_tests",
    "status": "${status}",
    "return_code": int("${rc}"),
    "elapsed_seconds": int("${end_ts}") - int("${start_ts}"),
    "tests": [
        "benchmark/selector_eval/cuda_ext/test_fullscan_pq_topk.py",
        "benchmark/selector_eval/cuda_ext/test_gpu_vpq_helpers.py",
        "benchmark/selector_eval/cuda_ext/test_online_page_append.py",
    ],
    "slurm_job_id": "${SLURM_JOB_ID:-manual}",
    "partition": "${SLURM_JOB_PARTITION:-spgpu}",
    "account": "zhengya98",
    "log": "unit_tests.log",
}
Path("${OUTPUT_DIR}/summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n")
PY

cat "${OUTPUT_DIR}/unit_tests.log"
cat "${OUTPUT_DIR}/summary.json"
exit "${rc}"
