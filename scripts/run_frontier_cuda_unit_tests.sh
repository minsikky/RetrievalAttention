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

case "${CUDA_UNIT_TEST_SET:-all}" in
  all)
    CUDA_UNIT_TESTS=(
      "benchmark/selector_eval/cuda_ext/test_fullscan_pq_topk.py"
      "benchmark/selector_eval/cuda_ext/test_gpu_vpq_helpers.py"
      "benchmark/selector_eval/cuda_ext/test_online_page_append.py"
    )
    ;;
  vpq)
    CUDA_UNIT_TESTS=(
      "benchmark/selector_eval/cuda_ext/test_gpu_vpq_helpers.py"
    )
    ;;
  *)
    echo "unknown CUDA_UNIT_TEST_SET=${CUDA_UNIT_TEST_SET}; expected all or vpq" >&2
    exit 2
    ;;
esac

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

  (
    flock 200
    cd benchmark/selector_eval/cuda_ext
    python setup.py build_ext --inplace
  ) 200>"${LOCK_FILE}"

  for test_path in "${CUDA_UNIT_TESTS[@]}"; do
    "${HF_VENV_DIR}/bin/python" "${test_path}"
  done
) >"${OUTPUT_DIR}/unit_tests.log" 2>&1
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
    "kind": "cuda_unit_tests",
    "status": "${status}",
    "return_code": int("${rc}"),
    "elapsed_seconds": int("${end_ts}") - int("${start_ts}"),
    "test_set": "${CUDA_UNIT_TEST_SET:-all}",
    "tests": """${CUDA_UNIT_TESTS[*]}""".split(),
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
