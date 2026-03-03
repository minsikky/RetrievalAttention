#!/bin/bash
#SBATCH --job-name=build_flashattn
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --account=zhengya98
#SBATCH --partition=standard
#SBATCH --gpus-per-node=0

set -euo pipefail

module purge
module load python/3.10.4
module unload pytorch 2>/dev/null || true
module load "${CUDA_MODULE:-cuda/12.8.1}"

cd /scratch/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention
source .venv/bin/activate
cd third_party/flash-attn-ra

# Resolve CUDA toolchain from the compute-node module environment.
if ! command -v nvcc >/dev/null 2>&1; then
  echo "[ERROR] nvcc not found on PATH. Check CUDA module in this sbatch job."
  exit 1
fi
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

# Build tuning knobs (can override via sbatch env exports).
# For fastest rebuild on a single target GPU class, set:
#   A40: FLASH_ATTN_CUDA_ARCHS=86 TORCH_CUDA_ARCH_LIST=8.6
#   A100: FLASH_ATTN_CUDA_ARCHS=80 TORCH_CUDA_ARCH_LIST=8.0
export FLASH_ATTN_CUDA_ARCHS="${FLASH_ATTN_CUDA_ARCHS:-80;86}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;8.6}"
export MAX_JOBS="${MAX_JOBS:-${SLURM_CPUS_PER_TASK:-16}}"
export NVCC_THREADS="${NVCC_THREADS:-4}"
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export FLASH_ATTN_INCREMENTAL="${FLASH_ATTN_INCREMENTAL:-1}"

echo "[INFO] host=$(hostname)"
echo "[INFO] python=$(which python)"
python -V
echo "[INFO] nvcc=$(which nvcc)"
nvcc -V
echo "[INFO] FLASH_ATTN_CUDA_ARCHS=${FLASH_ATTN_CUDA_ARCHS}"
echo "[INFO] TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
echo "[INFO] MAX_JOBS=${MAX_JOBS}"
echo "[INFO] NVCC_THREADS=${NVCC_THREADS}"
echo "[INFO] FLASH_ATTN_INCREMENTAL=${FLASH_ATTN_INCREMENTAL}"
python - <<'PY'
import os, torch
print("[INFO] torch:", torch.__version__)
print("[INFO] torch.cuda:", torch.version.cuda)
print("[INFO] CUDA_HOME:", os.environ.get("CUDA_HOME"))
PY

NEED_EDITABLE_INSTALL=0
python - <<'PY' || NEED_EDITABLE_INSTALL=1
import importlib, os, sys
repo = os.path.realpath(os.getcwd())
try:
    mod = importlib.import_module("flash_attn")
except Exception:
    sys.exit(1)
mod_path = os.path.realpath(os.path.dirname(mod.__file__))
if mod_path.startswith(repo):
    sys.exit(0)
sys.exit(1)
PY
if [ "${NEED_EDITABLE_INSTALL}" -ne 0 ]; then
  echo "[INFO] Installing editable flash_attn package once..."
  pip install --no-build-isolation --no-deps -v -e .
else
  echo "[INFO] Editable flash_attn already installed."
fi

if [ "${FLASH_ATTN_INCREMENTAL}" = "1" ]; then
  echo "[INFO] Running incremental in-place rebuild (setup.py build_ext --inplace)..."
  python setup.py build_ext --inplace
else
  echo "[INFO] Running full pip editable rebuild..."
  pip install --no-build-isolation --no-deps -v -e .
fi
