#!/bin/bash
set -euo pipefail

module load python/3.10.4

HF_CACHE_DIR="${HF_CACHE_DIR:-$(pwd)/.hf_cache}"
mkdir -p "${HF_CACHE_DIR}/hub" "${HF_CACHE_DIR}/datasets" "${HF_CACHE_DIR}/transformers"
export HF_HOME="${HF_HOME:-${HF_CACHE_DIR}}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_CACHE_DIR}/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HUB_CACHE}}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_CACHE_DIR}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_CACHE_DIR}/transformers}"

HF_VENV_DIR="${HF_VENV_DIR:-.venv}"
HF_PYDEPS_DIR="${HF_PYDEPS_DIR:-.hf_pydeps}"
HF_FORCE_REINSTALL="${HF_FORCE_REINSTALL:-0}"

if [ ! -x "${HF_VENV_DIR}/bin/python" ]; then
  echo "[ERROR] ${HF_VENV_DIR}/bin/python not found."
  exit 1
fi
mkdir -p "${HF_PYDEPS_DIR}"

PIP_CMD=("${HF_VENV_DIR}/bin/python" -m pip)
if ! "${PIP_CMD[@]}" --version >/dev/null 2>&1; then
  if [ -x "${HF_VENV_DIR}/bin/pip" ] && "${HF_VENV_DIR}/bin/pip" --version >/dev/null 2>&1; then
    PIP_CMD=("${HF_VENV_DIR}/bin/pip")
  elif python -m pip --version >/dev/null 2>&1; then
    echo "[WARN] ${HF_VENV_DIR} pip is unavailable; using module Python pip to install into --target."
    PIP_CMD=(python -m pip)
  else
    echo "[ERROR] No working pip found."
    exit 1
  fi
fi

if [ "${HF_FORCE_REINSTALL}" = "1" ] || [ ! -d "${HF_PYDEPS_DIR}/transformers" ]; then
  "${PIP_CMD[@]}" install \
    --upgrade \
    --no-deps \
    --target "${HF_PYDEPS_DIR}" \
    'transformers @ git+https://github.com/huggingface/transformers.git@main'
else
  echo "[INFO] ${HF_PYDEPS_DIR}/transformers already exists; set HF_FORCE_REINSTALL=1 to refresh it."
fi

"${PIP_CMD[@]}" install \
  --upgrade \
  --no-deps \
  --target "${HF_PYDEPS_DIR}" \
  'accelerate>=1.0.0' \
  'huggingface-hub>=0.34.0' \
  'hf_xet' \
  'tokenizers' \
  'safetensors' \
  'pillow' \
  'httpx' \
  'httpcore' \
  'h11' \
  'anyio' \
  'idna' \
  'certifi' \
  'sniffio' \
  'filelock' \
  'fsspec' \
  'packaging' \
  'pyyaml' \
  'regex' \
  'tqdm' \
  'typing-extensions' \
  'jinja2' \
  'markupsafe' \
  'datasets' \
  'pyarrow' \
  'pandas' \
  'numpy' \
  'python-dateutil' \
  'pytz' \
  'tzdata' \
  'xxhash' \
  'multiprocess' \
  'dill'

cat <<EOF
[INFO] Installed Hugging Face overlay deps into ${HF_PYDEPS_DIR}
[INFO] Run HF jobs with:
  HF_EXTRA_PYTHONPATH=${HF_PYDEPS_DIR} benchmark/run_generated_memory_hf.sh
or through sbatch:
  --export=ALL,HF_EXTRA_PYTHONPATH=${HF_PYDEPS_DIR},MODEL_NAME=Qwen/Qwen3.5-9B,...
EOF
