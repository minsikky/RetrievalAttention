#!/usr/bin/env bash
#SBATCH --job-name=geom-accept-bench
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000m
#SBATCH --time=00:20:00
#SBATCH --account=zhengya98
#SBATCH --partition=spgpu
#SBATCH --gpus-per-node=1
set -euo pipefail

cd /gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention

OUTPUT_DIR="${OUTPUT_DIR:-cuda_unit_result/geometric_accept_counts_bench_20260518}"
mkdir -p "${OUTPUT_DIR}" slurm_out/geometric_accept_counts_bench_20260518

module purge
module load python/3.10.4
module load "${CUDA_MODULE:-cuda/12.8.1}"
source .venv/bin/activate

export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.10/site-packages/torch/lib:/sw/pkgs/arc/python/3.10.4/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD/benchmark/selector_eval/cuda_ext:${PYTHONPATH:-}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;8.6;9.0}"
export MAX_JOBS="${MAX_JOBS:-4}"

LOCK_FILE="${CUDA_EXT_BUILD_LOCK:-${PWD}/.codex/selector_pq_build.lock}"
mkdir -p "$(dirname "${LOCK_FILE}")"

echo "[geom_accept_bench] host=$(hostname)"
echo "[geom_accept_bench] started=$(date --iso-8601=seconds)"
echo "[geom_accept_bench] output_dir=${OUTPUT_DIR}"
nvidia-smi || true

(
  flock 200
  cd benchmark/selector_eval/cuda_ext
  python setup.py build_ext --inplace
) 200>"${LOCK_FILE}"

python benchmark/selector_eval/cuda_ext/bench_geometric_accept_counts.py \
  --mode "${BENCH_MODE:-strict}" \
  --heads "${BENCH_HEADS:-32}" \
  --kv-heads "${BENCH_KV_HEADS:-8}" \
  --dim "${BENCH_DIM:-128}" \
  --pages "${BENCH_PAGES:-16}" \
  --page-size "${BENCH_PAGE_SIZE:-2048}" \
  --ranked "${BENCH_RANKED:-32768}" \
  --value-subvecs "${BENCH_VALUE_SUBVECS:-1}" \
  --value-centroids "${BENCH_VALUE_CENTROIDS:-16}" \
  --min-budget "${BENCH_MIN_BUDGET:-4096}" \
  --max-budget "${BENCH_MAX_BUDGET:-32768}" \
  --granularity "${BENCH_GRANULARITY:-1024}" \
  --growth "${BENCH_GROWTH:-1.5}" \
  --probe-scale "${BENCH_PROBE_SCALE:-1.5}" \
  --rel-l2-max "${BENCH_REL_L2_MAX:-0.04}" \
  --exact-value-top "${BENCH_EXACT_VALUE_TOP:-256}" \
  --exact-value-mass "${BENCH_EXACT_VALUE_MASS:-0.0}" \
  --exact-value-min-top "${BENCH_EXACT_VALUE_MIN_TOP:-0}" \
  --warmup "${BENCH_WARMUP:-3}" \
  --iters "${BENCH_ITERS:-10}" \
  $( [ "${BENCH_SKIP_OLD_LOOP:-0}" = "1" ] && printf '%s' "--skip-old-loop" ) \
  --output "${OUTPUT_DIR}/summary.json"

echo "[geom_accept_bench] finished=$(date --iso-8601=seconds)"
cat "${OUTPUT_DIR}/summary.json"
