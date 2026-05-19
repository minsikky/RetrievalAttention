#!/usr/bin/env bash
set -euo pipefail

cd /gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention
export LD_LIBRARY_PATH="/sw/pkgs/arc/python/3.10.4/lib:${LD_LIBRARY_PATH:-}"

module purge
module load python/3.10.4
module load "${CUDA_MODULE:-cuda/12.8.1}"

source .venv/bin/activate

if ! command -v nvcc >/dev/null 2>&1; then
  echo "[ERROR] nvcc not found on PATH. Check CUDA module in this sbatch job."
  exit 1
fi
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;8.6}"
export MAX_JOBS="${MAX_JOBS:-4}"

echo "[cuda_ext_test] host=$(hostname)"
echo "[cuda_ext_test] nvcc=$(which nvcc)"
echo "[cuda_ext_test] CUDA_HOME=${CUDA_HOME}"

LOCK_FILE="${CUDA_EXT_BUILD_LOCK:-${PWD}/.codex/selector_pq_build.lock}"
mkdir -p "$(dirname "${LOCK_FILE}")"
(
  flock 200
  pushd benchmark/selector_eval/cuda_ext >/dev/null
  python setup.py build_ext --inplace
  popd >/dev/null

  if [ "${RUN_CUDA_EXT_TEST:-1}" = "1" ]; then
    python benchmark/selector_eval/cuda_ext/test_fullscan_pq_topk.py
  fi

  if [ "${RUN_FUSED_SELECTOR_BENCH:-0}" = "1" ]; then
    python benchmark/selector_eval/cuda_ext/bench_fused_selector.py \
      --positions "${BENCH_POSITIONS:-128}" \
      --heads "${BENCH_HEADS:-32}" \
      --kv-heads "${BENCH_KV_HEADS:-8}" \
      --dim "${BENCH_DIM:-128}" \
      --pages "${BENCH_PAGES:-16}" \
      --page-size "${BENCH_PAGE_SIZE:-256}" \
      --page-configs "${BENCH_PAGE_CONFIGS:-}" \
      --subvecs "${BENCH_SUBVECS:-4}" \
      --centroids "${BENCH_CENTROIDS:-256}" \
      --budgets "${BENCH_BUDGETS:-8,16,32,64}" \
      --fused-modes "${BENCH_FUSED_MODES:-auto}" \
      --warmup "${BENCH_WARMUP:-10}" \
      --iters "${BENCH_ITERS:-50}"
  fi

  if [ "${RUN_SELECTED_ATTENTION_BENCH:-0}" = "1" ]; then
    python benchmark/selector_eval/cuda_ext/bench_selected_attention.py \
      --positions "${SELECTED_BENCH_POSITIONS:-2048}" \
      --heads "${SELECTED_BENCH_HEADS:-32}" \
      --kv-heads "${SELECTED_BENCH_KV_HEADS:-8}" \
      --dim "${SELECTED_BENCH_DIM:-128}" \
      --total-tokens "${SELECTED_BENCH_TOTAL_TOKENS:-16384}" \
      --query-start "${SELECTED_BENCH_QUERY_START:-8192}" \
      --selected "${SELECTED_BENCH_SELECTED:-512,1024,2048}" \
      --static-prefix "${SELECTED_BENCH_STATIC_PREFIX:-128}" \
      --static-suffix "${SELECTED_BENCH_STATIC_SUFFIX:-128}" \
      --page-size "${SELECTED_BENCH_PAGE_SIZE:-512}" \
      --kv-dtype "${SELECTED_BENCH_KV_DTYPE:-bfloat16}" \
      --warmup "${SELECTED_BENCH_WARMUP:-5}" \
      --iters "${SELECTED_BENCH_ITERS:-20}"
  fi

  if [ "${RUN_TAIL_ATTENTION_BENCH:-0}" = "1" ]; then
    python benchmark/selector_eval/cuda_ext/bench_tail_attention.py \
      --positions "${TAIL_BENCH_POSITIONS:-128}" \
      --heads "${TAIL_BENCH_HEADS:-32}" \
      --kv-heads "${TAIL_BENCH_KV_HEADS:-8}" \
      --dim "${TAIL_BENCH_DIM:-128}" \
      --pages "${TAIL_BENCH_PAGES:-8}" \
      --page-size "${TAIL_BENCH_PAGE_SIZE:-512}" \
      --static-prefix "${TAIL_BENCH_STATIC_PREFIX:-128}" \
      --static-suffix "${TAIL_BENCH_STATIC_SUFFIX:-128}" \
      --selected "${TAIL_BENCH_SELECTED:-512,1024,2048,4096}" \
      --value-centroids "${TAIL_BENCH_VALUE_CENTROIDS:-16}" \
      --value-subvecs "${TAIL_BENCH_VALUE_SUBVECS:-1}" \
      --exact-value-top "${TAIL_BENCH_EXACT_VALUE_TOP:-1024}" \
      --tail-blend "${TAIL_BENCH_TAIL_BLEND:-1.0}" \
      --kv-dtype "${TAIL_BENCH_KV_DTYPE:-bfloat16}" \
      --warmup "${TAIL_BENCH_WARMUP:-2}" \
      --iters "${TAIL_BENCH_ITERS:-5}"
  fi
) 200>"${LOCK_FILE}"
