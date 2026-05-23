#!/usr/bin/env bash
#SBATCH --job-name=grouped-risk-bench
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24000m
#SBATCH --time=00:10:00
#SBATCH --account=zhengya98
#SBATCH --partition=spgpu
#SBATCH --gpus-per-node=1
set -euo pipefail

cd /gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention

OUTPUT_DIR="${OUTPUT_DIR:-cuda_unit_result/grouped_risk_prefix_bench_20260523}"
mkdir -p "${OUTPUT_DIR}"

module purge
module load python/3.10.4
module load cuda/12.8.1
source .venv/bin/activate

export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.10/site-packages/torch/lib:/sw/pkgs/arc/python/3.10.4/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD/benchmark/selector_eval/cuda_ext:${PYTHONPATH:-}"

.venv/bin/python benchmark/selector_eval/cuda_ext/bench_grouped_risk_prefix.py \
  --groups "${RISK_GROUPS:-8}" \
  --k_count "${RISK_K_COUNT:-4}" \
  --heads_per_group "${RISK_HEADS_PER_GROUP:-4}" \
  --context_len "${RISK_CONTEXT_LEN:-32768}" \
  --dim "${RISK_HEAD_DIM:-128}" \
  --v_budgets "${RISK_V_BUDGETS:-1024,2048,4096,6144,8192,12288,16384}" \
  --warmup "${RISK_WARMUP:-2}" \
  --iters "${RISK_ITERS:-10}" \
  >"${OUTPUT_DIR}/summary.json"

cat "${OUTPUT_DIR}/summary.json"
