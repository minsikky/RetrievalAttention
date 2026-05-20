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
source .venv/bin/activate

export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.10/site-packages/torch/lib:/sw/pkgs/arc/python/3.10.4/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD/benchmark/selector_eval/cuda_ext:${PYTHONPATH:-}"

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
