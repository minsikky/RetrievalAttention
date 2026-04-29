#!/bin/bash
#SBATCH --job-name=lbv2-hf
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128000m
#SBATCH --time=02:00:00
#SBATCH --account=zhengya98
#SBATCH --partition=spgpu
#SBATCH --gpus-per-node=1

module purge
module load python/3.10.4
module unload pytorch 2>/dev/null || true
module load cuda/12.8.1

HF_CACHE_DIR="${HF_CACHE_DIR:-$(pwd)/.hf_cache}"
mkdir -p "${HF_CACHE_DIR}/hub" "${HF_CACHE_DIR}/datasets" "${HF_CACHE_DIR}/transformers"
export HF_HOME="${HF_HOME:-${HF_CACHE_DIR}}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_CACHE_DIR}/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HUB_CACHE}}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_CACHE_DIR}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_CACHE_DIR}/transformers}"

HF_VENV_DIR="${HF_VENV_DIR:-.venv}"
if [ -f "${HF_VENV_DIR}/bin/activate" ]; then
  # shellcheck disable=SC1090
  source "${HF_VENV_DIR}/bin/activate"
else
  echo "[ERROR] ${HF_VENV_DIR}/bin/activate not found."
  exit 1
fi

unset PYTHONPATH
unset PYTHONHOME
export PYTHONNOUSERSITE=1
HF_EXTRA_PYTHONPATH="${HF_EXTRA_PYTHONPATH:-}"
if [ -n "${HF_EXTRA_PYTHONPATH}" ]; then
  export PYTHONPATH="${HF_EXTRA_PYTHONPATH}"
  if [ -d "${HF_EXTRA_PYTHONPATH}/numpy.libs" ]; then
    export LD_LIBRARY_PATH="${HF_EXTRA_PYTHONPATH}/numpy.libs:${LD_LIBRARY_PATH:-}"
  fi
fi
export TOKENIZERS_PARALLELISM=false
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-9B}"
DTYPE="${DTYPE:-bf16}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-0}"
LOW_CPU_MEM_USAGE="${LOW_CPU_MEM_USAGE:-1}"
HF_LANGUAGE_MODEL_ONLY="${HF_LANGUAGE_MODEL_ONLY:-1}"
USE_CHAT_TEMPLATE="${USE_CHAT_TEMPLATE:-1}"
DISABLE_THINKING="${DISABLE_THINKING:-1}"
DATASET_NAME="${DATASET_NAME:-THUDM/LongBench-v2}"
SPLIT="${SPLIT:-train}"
OUTPUT_DIR="${OUTPUT_DIR:-longbench_v2_hf_result}"
MAX_EXAMPLES="${MAX_EXAMPLES:-16}"
LENGTH_FILTER="${LENGTH_FILTER:-}"
DIFFICULTY_FILTER="${DIFFICULTY_FILTER:-}"
DOMAIN_FILTER="${DOMAIN_FILTER:-}"
ID_FILTER="${ID_FILTER:-}"
SELECTION="${SELECTION:-first}"
SEED="${SEED:-2026}"
MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-120000}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
TEMPERATURE="${TEMPERATURE:-0.1}"
STREAMING="${STREAMING:-1}"
DATASET_SCAN_LIMIT="${DATASET_SCAN_LIMIT:-200}"
QWEN_YARN_FACTOR="${QWEN_YARN_FACTOR:-0}"
QWEN_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS="${QWEN_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS:-262144}"

echo "[INFO] Job started at: $(date)"
echo "[INFO] Host: $(hostname)"
echo "[INFO] MODEL_NAME=${MODEL_NAME}"
echo "[INFO] DATASET_NAME=${DATASET_NAME}"
echo "[INFO] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[INFO] HF_CACHE_DIR=${HF_CACHE_DIR}"
echo "[INFO] HF_HOME=${HF_HOME}"
echo "[INFO] HF_HUB_CACHE=${HF_HUB_CACHE}"
echo "[INFO] HF_DATASETS_CACHE=${HF_DATASETS_CACHE}"
echo "[INFO] TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE}"
echo "[INFO] MAX_EXAMPLES=${MAX_EXAMPLES}"
echo "[INFO] LENGTH_FILTER=${LENGTH_FILTER}"
echo "[INFO] DIFFICULTY_FILTER=${DIFFICULTY_FILTER}"
echo "[INFO] DOMAIN_FILTER=${DOMAIN_FILTER}"
echo "[INFO] ID_FILTER=${ID_FILTER}"
echo "[INFO] SELECTION=${SELECTION}"
echo "[INFO] MAX_INPUT_TOKENS=${MAX_INPUT_TOKENS}"
echo "[INFO] QWEN_YARN_FACTOR=${QWEN_YARN_FACTOR}"
echo "[INFO] QWEN_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS=${QWEN_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS}"
echo "[INFO] USE_CHAT_TEMPLATE=${USE_CHAT_TEMPLATE}"
echo "[INFO] DISABLE_THINKING=${DISABLE_THINKING}"
echo "[INFO] HF_EXTRA_PYTHONPATH=${HF_EXTRA_PYTHONPATH}"
echo "[INFO] Python: $(which python)"
python -V

"${HF_VENV_DIR}/bin/python" benchmark/longbench_v2_hf_eval.py \
  --model_name "${MODEL_NAME}" \
  --dtype "${DTYPE}" \
  --device_map "${DEVICE_MAP}" \
  --dataset_name "${DATASET_NAME}" \
  --split "${SPLIT}" \
  --output_dir "${OUTPUT_DIR}" \
  --max_examples "${MAX_EXAMPLES}" \
  --length_filter "${LENGTH_FILTER}" \
  --difficulty_filter "${DIFFICULTY_FILTER}" \
  --domain_filter "${DOMAIN_FILTER}" \
  --id_filter "${ID_FILTER}" \
  --selection "${SELECTION}" \
  --seed "${SEED}" \
  --max_input_tokens "${MAX_INPUT_TOKENS}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --temperature "${TEMPERATURE}" \
  --dataset_scan_limit "${DATASET_SCAN_LIMIT}" \
  --qwen_yarn_factor "${QWEN_YARN_FACTOR}" \
  --qwen_yarn_original_max_position_embeddings "${QWEN_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS}" \
  $( [ -n "${ATTN_IMPLEMENTATION}" ] && printf '%s %s' "--attn_implementation" "${ATTN_IMPLEMENTATION}" ) \
  $( [ "${TRUST_REMOTE_CODE}" = "1" ] && printf '%s' "--trust_remote_code" ) \
  $( [ "${LOCAL_FILES_ONLY}" = "1" ] && printf '%s' "--local_files_only" ) \
  $( [ "${LOW_CPU_MEM_USAGE}" = "1" ] && printf '%s' "--low_cpu_mem_usage" ) \
  $( [ "${HF_LANGUAGE_MODEL_ONLY}" = "1" ] && printf '%s' "--hf_language_model_only" ) \
  $( [ "${USE_CHAT_TEMPLATE}" = "1" ] && printf '%s' "--use_chat_template" ) \
  $( [ "${DISABLE_THINKING}" = "1" ] && printf '%s' "--disable_thinking" ) \
  $( [ "${STREAMING}" = "1" ] && printf '%s' "--streaming" )

echo "[INFO] Job finished at: $(date)"
