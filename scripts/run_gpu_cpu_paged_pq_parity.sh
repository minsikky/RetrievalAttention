#!/usr/bin/env bash
#SBATCH --job-name=gpu-cpu-pq-parity
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64000m
#SBATCH --time=00:30:00
#SBATCH --account=zhengya98
#SBATCH --partition=spgpu
#SBATCH --gpus-per-node=1
set -euo pipefail

cd /gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention

OUTPUT_DIR="${OUTPUT_DIR:-cuda_unit_result/gpu_cpu_paged_pq_parity_$(date +%Y%m%d_%H%M%S)}"
TRACE="${TRACE:-attention_efficiency_result/real_xtrace_qkv_llama31_8b_layer16_8k_g16384_full.npz}"
DECODE_LENGTHS="${DECODE_LENGTHS:-500,1000,2000}"
HEADS="${HEADS:-0,8}"
BUDGETS="${BUDGETS:-256,1024}"
PAGE_SIZE="${PAGE_SIZE:-2048}"
ATTN_KEY_BYTES="${ATTN_KEY_BYTES:-2}"
SCORE_KEY_BYTES="${SCORE_KEY_BYTES:-4}"
CHECK_VPQ_TAIL="${CHECK_VPQ_TAIL:-0}"
VALUE_SUBVECS="${VALUE_SUBVECS:-4}"
VALUE_SUBBITS="${VALUE_SUBBITS:-6}"
EXACT_VALUE_TOP="${EXACT_VALUE_TOP:--64}"
TAIL_BLEND="${TAIL_BLEND:-1.0}"
CHECK_CANONICAL_GEOMETRIC="${CHECK_CANONICAL_GEOMETRIC:-0}"
GEOMETRIC_MIN_BUDGET="${GEOMETRIC_MIN_BUDGET:-4096}"
GEOMETRIC_MAX_BUDGET="${GEOMETRIC_MAX_BUDGET:-32768}"
GEOMETRIC_BUDGET_GRANULARITY="${GEOMETRIC_BUDGET_GRANULARITY:-1024}"
GEOMETRIC_GROWTH="${GEOMETRIC_GROWTH:-1.5}"
GEOMETRIC_PROBE_SCALE="${GEOMETRIC_PROBE_SCALE:-1.5}"
TAIL_PROBE_REL_L2_MAX="${TAIL_PROBE_REL_L2_MAX:-0.020}"
SELECTED_VALUE_EXACT_MASS="${SELECTED_VALUE_EXACT_MASS:-0.99}"
SELECTED_VALUE_MIN_EXACT_TOP="${SELECTED_VALUE_MIN_EXACT_TOP:-1024}"
TAIL_PROXY_MASS_MIN="${TAIL_PROXY_MASS_MIN:-0.99}"
TAIL_PROXY_MASS_MAX="${TAIL_PROXY_MASS_MAX:-1.0}"
TAIL_PQ_CORR_MIN="${TAIL_PQ_CORR_MIN:-0.70}"
TAIL_PQ_RELRMSE_MAX="${TAIL_PQ_RELRMSE_MAX:-inf}"
CANONICAL_OUTPUT_ATOL="${CANONICAL_OUTPUT_ATOL:-0.005}"
MAX_QIDX_PER_DECODE="${MAX_QIDX_PER_DECODE:-1}"
mkdir -p "${OUTPUT_DIR}"

module purge
module load python/3.10.4
module load cuda/12.8.1
source .venv/bin/activate

export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.10/site-packages/torch/lib:/sw/pkgs/arc/python/3.10.4/lib:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;8.6;9.0}"
export PYTHONPATH="$PWD/benchmark/selector_eval/cuda_ext:${PYTHONPATH:-}"
export MAX_JOBS="${MAX_JOBS:-4}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

start_ts="$(date +%s)"
status="passed"

set +e
(
  set -e
  extra_args=()
  if [ "${CHECK_VPQ_TAIL}" = "1" ] || [ "${CHECK_VPQ_TAIL}" = "true" ] || [ "${CHECK_VPQ_TAIL}" = "yes" ]; then
    extra_args+=(--check_vpq_tail)
  fi
  if [ "${CHECK_CANONICAL_GEOMETRIC}" = "1" ] || [ "${CHECK_CANONICAL_GEOMETRIC}" = "true" ] || [ "${CHECK_CANONICAL_GEOMETRIC}" = "yes" ]; then
    extra_args+=(--check_canonical_geometric)
  fi
  echo "[gpu_cpu_parity] host=$(hostname)"
  echo "[gpu_cpu_parity] started=$(date --iso-8601=seconds)"
  echo "[gpu_cpu_parity] output_dir=${OUTPUT_DIR}"
  echo "[gpu_cpu_parity] trace=${TRACE}"
  echo "[gpu_cpu_parity] decode_lengths=${DECODE_LENGTHS}"
  echo "[gpu_cpu_parity] heads=${HEADS}"
  echo "[gpu_cpu_parity] budgets=${BUDGETS}"
  echo "[gpu_cpu_parity] page_size=${PAGE_SIZE}"
  echo "[gpu_cpu_parity] attn_key_bytes=${ATTN_KEY_BYTES}"
  echo "[gpu_cpu_parity] score_key_bytes=${SCORE_KEY_BYTES}"
  echo "[gpu_cpu_parity] check_vpq_tail=${CHECK_VPQ_TAIL}"
  echo "[gpu_cpu_parity] value_subvecs=${VALUE_SUBVECS}"
  echo "[gpu_cpu_parity] value_subbits=${VALUE_SUBBITS}"
  echo "[gpu_cpu_parity] exact_value_top=${EXACT_VALUE_TOP}"
  echo "[gpu_cpu_parity] check_canonical_geometric=${CHECK_CANONICAL_GEOMETRIC}"
  echo "[gpu_cpu_parity] geometric_min_budget=${GEOMETRIC_MIN_BUDGET}"
  echo "[gpu_cpu_parity] geometric_max_budget=${GEOMETRIC_MAX_BUDGET}"
  echo "[gpu_cpu_parity] geometric_budget_granularity=${GEOMETRIC_BUDGET_GRANULARITY}"
  echo "[gpu_cpu_parity] tail_probe_rel_l2_max=${TAIL_PROBE_REL_L2_MAX}"
  echo "[gpu_cpu_parity] selected_value_exact_mass=${SELECTED_VALUE_EXACT_MASS}"
  echo "[gpu_cpu_parity] selected_value_min_exact_top=${SELECTED_VALUE_MIN_EXACT_TOP}"
  echo "[gpu_cpu_parity] numpy_threads=OMP:${OMP_NUM_THREADS} OPENBLAS:${OPENBLAS_NUM_THREADS} MKL:${MKL_NUM_THREADS}"
  python -V
  nvidia-smi || true

  cd benchmark/selector_eval/cuda_ext
  python setup.py build_ext --inplace
  cd ../../..

  .venv/bin/python benchmark/selector_eval/gpu/run_gpu_cpu_parity_eval.py \
    --trace "${TRACE}" \
    --output_dir "${OUTPUT_DIR}" \
    --decode_lengths "${DECODE_LENGTHS}" \
    --heads "${HEADS}" \
    --budgets "${BUDGETS}" \
    --page_size "${PAGE_SIZE}" \
    --attn_key_bytes "${ATTN_KEY_BYTES}" \
    --score_key_bytes "${SCORE_KEY_BYTES}" \
    --value_subvecs "${VALUE_SUBVECS}" \
    --value_subbits "${VALUE_SUBBITS}" \
    --exact_value_top "${EXACT_VALUE_TOP}" \
    --tail_blend "${TAIL_BLEND}" \
    --geometric_min_budget "${GEOMETRIC_MIN_BUDGET}" \
    --geometric_max_budget "${GEOMETRIC_MAX_BUDGET}" \
    --geometric_budget_granularity "${GEOMETRIC_BUDGET_GRANULARITY}" \
    --geometric_growth "${GEOMETRIC_GROWTH}" \
    --geometric_probe_scale "${GEOMETRIC_PROBE_SCALE}" \
    --tail_probe_rel_l2_max "${TAIL_PROBE_REL_L2_MAX}" \
    --selected_value_exact_mass "${SELECTED_VALUE_EXACT_MASS}" \
    --selected_value_min_exact_top "${SELECTED_VALUE_MIN_EXACT_TOP}" \
    --tail_proxy_mass_min "${TAIL_PROXY_MASS_MIN}" \
    --tail_proxy_mass_max "${TAIL_PROXY_MASS_MAX}" \
    --tail_pq_corr_min "${TAIL_PQ_CORR_MIN}" \
    --tail_pq_relrmse_max "${TAIL_PQ_RELRMSE_MAX}" \
    --canonical_output_atol "${CANONICAL_OUTPUT_ATOL}" \
    --max_qidx_per_decode "${MAX_QIDX_PER_DECODE}" \
    "${extra_args[@]}" \
    --strict
) >"${OUTPUT_DIR}/parity.log" 2>&1
rc=$?
set -e

if [ "${rc}" -ne 0 ]; then
  status="failed"
fi
end_ts="$(date +%s)"

.venv/bin/python - <<PY
import json
from pathlib import Path

out = Path("${OUTPUT_DIR}")
summary_path = out / "summary.json"
payload = {}
if summary_path.exists():
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
payload.update({
    "wrapper_status": "${status}",
    "return_code": int("${rc}"),
    "elapsed_seconds": int("${end_ts}") - int("${start_ts}"),
    "slurm_job_id": "${SLURM_JOB_ID:-manual}",
    "partition": "${SLURM_JOB_PARTITION:-spgpu}",
    "account": "zhengya98",
    "log": "parity.log",
})
summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
PY

cat "${OUTPUT_DIR}/parity.log"
cat "${OUTPUT_DIR}/summary.json"
exit "${rc}"
