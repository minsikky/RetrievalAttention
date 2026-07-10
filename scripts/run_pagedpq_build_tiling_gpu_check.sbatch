#!/usr/bin/env bash
#SBATCH --job-name=pagedpq-build-tiling
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16000m
#SBATCH --time=00:30:00
#SBATCH --account=zhengya0
#SBATCH --partition=gpu_mig40
#SBATCH --gpus-per-node=1

set -euo pipefail

cd /gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention
module purge
module load python/3.10.4
export LD_LIBRARY_PATH="/sw/pkgs/arc/python/3.10.4/lib:${LD_LIBRARY_PATH:-}"
export PAGEDPQ_BUILD_LOG_TILE=1

status=0
echo "[pagedpq-build-check] mode=equality"
.venv/bin/python benchmark/selector_eval/runners/verify_pagedpq_build_tiling.py --device cuda || status=$?
echo "[pagedpq-build-check] mode=determinism"
.venv/bin/python benchmark/selector_eval/runners/verify_pagedpq_build_tiling.py \
  --device cuda --mode determinism || status=$?
exit "${status}"
