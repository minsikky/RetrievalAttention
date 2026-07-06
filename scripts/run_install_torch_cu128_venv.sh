#!/usr/bin/env bash
#SBATCH --job-name=install_torch_cu128
#SBATCH --account=zhengya98
#SBATCH --partition=standard
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=logs/slurm/%x-%j.out

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention}"
VENV_DIR="${VENV_DIR:-.venv_cu128}"
TORCH_VERSION="${TORCH_VERSION:-2.11.0+cu128}"

cd "${ROOT_DIR}"
mkdir -p logs/slurm

module load python/3.10.4 cuda/12.8.1

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  python -m venv "${VENV_DIR}"
fi

"${VENV_DIR}/bin/python" -m pip install --upgrade pip wheel
"${VENV_DIR}/bin/python" -m pip install \
  --index-url https://download.pytorch.org/whl/cu128 \
  "torch==${TORCH_VERSION}"

# Minimal build/runtime dependencies for local CUDA-extension validation.
"${VENV_DIR}/bin/python" -m pip install \
  ninja==1.11.1.4 \
  pybind11==2.12.0 \
  numpy==1.26.4 \
  packaging==25.0 \
  safetensors

"${VENV_DIR}/bin/python" - <<'PY'
import os
import torch
from torch.utils.cpp_extension import _get_cuda_arch_flags

print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
for arch in ("8.0;8.6;12.0", "8.0;8.6;12.0+PTX", "8.0;8.6;9.0+PTX"):
    os.environ["TORCH_CUDA_ARCH_LIST"] = arch
    try:
        print("arch", arch, _get_cuda_arch_flags())
    except Exception as exc:
        print("arch", arch, type(exc).__name__, exc)
PY
