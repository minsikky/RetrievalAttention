#!/usr/bin/env bash
#SBATCH --job-name=exact-logit-bench
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64000m
#SBATCH --time=00:15:00
#SBATCH --account=zhengya98
#SBATCH --partition=spgpu
#SBATCH --gpus-per-node=1
set -euo pipefail

cd /gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention

OUTPUT_DIR="${OUTPUT_DIR:-cuda_unit_result/exact_logit_backend_bench_20260520}"
mkdir -p "${OUTPUT_DIR}"

module purge
module load python/3.10.4
module load cuda/12.8.1
source .venv/bin/activate

export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.10/site-packages/torch/lib:/sw/pkgs/arc/python/3.10.4/lib:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;8.6;9.0}"
export PYTHONPATH="$PWD/benchmark/selector_eval/cuda_ext:${PYTHONPATH:-}"
export MAX_JOBS="${MAX_JOBS:-4}"
LOCK_FILE="${CUDA_EXT_BUILD_LOCK:-${PWD}/.codex/selector_pq_build.lock}"
mkdir -p "$(dirname "${LOCK_FILE}")"

if [[ "${BUILD_CUDA_EXT:-1}" == "1" ]]; then
  (
    flock 200
    cd benchmark/selector_eval/cuda_ext
    python setup.py build_ext --inplace
  ) 200>"${LOCK_FILE}"
fi

CONTEXT="${CONTEXT:-32768}"
RANK="${RANK:-32768}"
ITERS="${ITERS:-20}"
WARMUP="${WARMUP:-5}"

.venv/bin/python benchmark/selector_eval/cuda_ext/bench_exact_logit_backends.py \
  --context "${CONTEXT}" \
  --rank "${RANK}" \
  --iters "${ITERS}" \
  --warmup "${WARMUP}" \
  --output "${OUTPUT_DIR}/summary.json"
