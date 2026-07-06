#!/bin/bash
#SBATCH --job-name=gen-mem-hf
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64000m
#SBATCH --time=120:00
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

HF_MODEL_PRESET="${HF_MODEL_PRESET:-}"
source scripts/hf_model_presets.sh
if [ -n "${HF_MODEL_PRESET}" ]; then
  resolve_hf_model_preset "${HF_MODEL_PRESET}" || exit $?
else
  PRESET_MODEL_NAME=""
  PRESET_HF_VENV_DIR=""
  PRESET_HF_EXTRA_PYTHONPATH=""
  PRESET_TRUST_REMOTE_CODE=""
  PRESET_HF_LANGUAGE_MODEL_ONLY=""
  PRESET_USE_CHAT_TEMPLATE=""
  PRESET_DISABLE_THINKING=""
fi

HF_VENV_DIR="${HF_VENV_DIR:-${PRESET_HF_VENV_DIR:-.venv}}"
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
HF_EXTRA_PYTHONPATH="${HF_EXTRA_PYTHONPATH:-${PRESET_HF_EXTRA_PYTHONPATH}}"
if [ -n "${HF_EXTRA_PYTHONPATH}" ]; then
  export PYTHONPATH="${HF_EXTRA_PYTHONPATH}"
  if [ -d "${HF_EXTRA_PYTHONPATH}/numpy.libs" ]; then
    export LD_LIBRARY_PATH="${HF_EXTRA_PYTHONPATH}/numpy.libs:${LD_LIBRARY_PATH:-}"
  fi
fi
export TOKENIZERS_PARALLELISM=false
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-${PRESET_MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}}"
DTYPE="${DTYPE:-bf16}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"
SEED="${SEED:-2025}"
NUM_ENTRIES="${NUM_ENTRIES:-24}"
NUM_QUERIES="${NUM_QUERIES:-10}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-generated_memory_hf_eval_result}"
PREFILL_FILLER_REPEATS="${PREFILL_FILLER_REPEATS:-0}"
MIN_PROMPT_TOKENS="${MIN_PROMPT_TOKENS:-0}"
GENERATION_MODE="${GENERATION_MODE:-manual_cache}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-${PRESET_TRUST_REMOTE_CODE:-0}}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-0}"
LOW_CPU_MEM_USAGE="${LOW_CPU_MEM_USAGE:-1}"
HF_LANGUAGE_MODEL_ONLY="${HF_LANGUAGE_MODEL_ONLY:-${PRESET_HF_LANGUAGE_MODEL_ONLY:-0}}"
USE_CHAT_TEMPLATE="${USE_CHAT_TEMPLATE:-${PRESET_USE_CHAT_TEMPLATE:-0}}"
DISABLE_THINKING="${DISABLE_THINKING:-${PRESET_DISABLE_THINKING:-0}}"
INVENTORY_ONLY="${INVENTORY_ONLY:-0}"
CONFIG_ONLY="${CONFIG_ONLY:-0}"
TOKENIZER_ONLY="${TOKENIZER_ONLY:-0}"
TRACE_ATTENTION="${TRACE_ATTENTION:-0}"
TRACE_PREFILL="${TRACE_PREFILL:-0}"
TRACE_DECODE_STEPS="${TRACE_DECODE_STEPS:-8}"
TRACE_MAX_RECORDS="${TRACE_MAX_RECORDS:-256}"
HF_ATTENTION_MODE="${HF_ATTENTION_MODE:-native}"
HF_SPARSE_TOPK="${HF_SPARSE_TOPK:-128}"
HF_SPARSE_STATIC_PREFIX="${HF_SPARSE_STATIC_PREFIX:-128}"
HF_SPARSE_STATIC_SUFFIX="${HF_SPARSE_STATIC_SUFFIX:-512}"
HF_GRAPH_DEGREE="${HF_GRAPH_DEGREE:-16}"
HF_GRAPH_VISIT_BUDGET="${HF_GRAPH_VISIT_BUDGET:-256}"
HF_GRAPH_SEED_COUNT="${HF_GRAPH_SEED_COUNT:-32}"
HF_GRAPH_ONLINE_EDGES="${HF_GRAPH_ONLINE_EDGES:-16}"
HF_GRAPH_SEARCH_BACKEND="${HF_GRAPH_SEARCH_BACKEND:-cuda_group}"
HF_GRAPH_CANDIDATE_TARGET="${HF_GRAPH_CANDIDATE_TARGET:-0}"
HF_GRAPH_EXPAND_WIDTH="${HF_GRAPH_EXPAND_WIDTH:-32}"
HF_GRAPH_MIN_VISITS="${HF_GRAPH_MIN_VISITS:-32}"
HF_GRAPH_FRONTIER_TOPN="${HF_GRAPH_FRONTIER_TOPN:-128}"
HF_GRAPH_STOP_PATIENCE="${HF_GRAPH_STOP_PATIENCE:-1}"
HF_GRAPH_STOP_MARGIN="${HF_GRAPH_STOP_MARGIN:-0.0}"
HF_GRAPH_ROAR_CAND_LIMIT="${HF_GRAPH_ROAR_CAND_LIMIT:-32}"
HF_GRAPH_ROAR_ENHANCE_LIMIT="${HF_GRAPH_ROAR_ENHANCE_LIMIT:-32}"
HF_GRAPH_ROAR_ENTRY="${HF_GRAPH_ROAR_ENTRY:-hub}"
HF_GRAPH_ROAR_THREADS="${HF_GRAPH_ROAR_THREADS:-0}"
ANSWER_PREFIX_SCAFFOLD="${ANSWER_PREFIX_SCAFFOLD:-0}"
ANSWER_CONSTRAINED_CODEBOOK="${ANSWER_CONSTRAINED_CODEBOOK:-0}"
FORCE_MAX_DECODE_STEPS="${FORCE_MAX_DECODE_STEPS:-0}"
HF_READY_MARKER="${HF_READY_MARKER:-hf_job_ready}"

echo "[INFO] Job started at: $(date)"
echo "[INFO] Host: $(hostname)"
echo "[INFO] HF_MODEL_PRESET=${HF_MODEL_PRESET}"
echo "[INFO] HF_CACHE_DIR=${HF_CACHE_DIR}"
echo "[INFO] HF_HOME=${HF_HOME}"
echo "[INFO] HF_HUB_CACHE=${HF_HUB_CACHE}"
echo "[INFO] HF_DATASETS_CACHE=${HF_DATASETS_CACHE}"
echo "[INFO] TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE}"
echo "[INFO] HF_VENV_DIR=${HF_VENV_DIR}"
echo "[INFO] HF_EXTRA_PYTHONPATH=${HF_EXTRA_PYTHONPATH}"
echo "[INFO] MODEL_NAME=${MODEL_NAME}"
echo "[INFO] DTYPE=${DTYPE}"
echo "[INFO] DEVICE_MAP=${DEVICE_MAP}"
echo "[INFO] ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION}"
echo "[INFO] GENERATION_MODE=${GENERATION_MODE}"
echo "[INFO] NUM_SAMPLES=${NUM_SAMPLES}"
echo "[INFO] SEED=${SEED}"
echo "[INFO] NUM_ENTRIES=${NUM_ENTRIES}"
echo "[INFO] NUM_QUERIES=${NUM_QUERIES}"
echo "[INFO] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[INFO] USE_CHAT_TEMPLATE=${USE_CHAT_TEMPLATE}"
echo "[INFO] DISABLE_THINKING=${DISABLE_THINKING}"
echo "[INFO] LOCAL_FILES_ONLY=${LOCAL_FILES_ONLY}"
echo "[INFO] LOW_CPU_MEM_USAGE=${LOW_CPU_MEM_USAGE}"
echo "[INFO] HF_LANGUAGE_MODEL_ONLY=${HF_LANGUAGE_MODEL_ONLY}"
echo "[INFO] INVENTORY_ONLY=${INVENTORY_ONLY}"
echo "[INFO] CONFIG_ONLY=${CONFIG_ONLY}"
echo "[INFO] TOKENIZER_ONLY=${TOKENIZER_ONLY}"
echo "[INFO] TRACE_ATTENTION=${TRACE_ATTENTION}"
echo "[INFO] HF_ATTENTION_MODE=${HF_ATTENTION_MODE}"
echo "[INFO] HF_SPARSE_TOPK=${HF_SPARSE_TOPK}"
echo "[INFO] HF_SPARSE_STATIC_PREFIX=${HF_SPARSE_STATIC_PREFIX}"
echo "[INFO] HF_SPARSE_STATIC_SUFFIX=${HF_SPARSE_STATIC_SUFFIX}"
echo "[INFO] HF_GRAPH_DEGREE=${HF_GRAPH_DEGREE}"
echo "[INFO] HF_GRAPH_VISIT_BUDGET=${HF_GRAPH_VISIT_BUDGET}"
echo "[INFO] HF_GRAPH_SEED_COUNT=${HF_GRAPH_SEED_COUNT}"
echo "[INFO] HF_GRAPH_ONLINE_EDGES=${HF_GRAPH_ONLINE_EDGES}"
echo "[INFO] HF_GRAPH_SEARCH_BACKEND=${HF_GRAPH_SEARCH_BACKEND}"
echo "[INFO] HF_GRAPH_CANDIDATE_TARGET=${HF_GRAPH_CANDIDATE_TARGET}"
echo "[INFO] HF_GRAPH_EXPAND_WIDTH=${HF_GRAPH_EXPAND_WIDTH}"
echo "[INFO] HF_GRAPH_MIN_VISITS=${HF_GRAPH_MIN_VISITS}"
echo "[INFO] HF_GRAPH_FRONTIER_TOPN=${HF_GRAPH_FRONTIER_TOPN}"
echo "[INFO] HF_GRAPH_STOP_PATIENCE=${HF_GRAPH_STOP_PATIENCE}"
echo "[INFO] HF_GRAPH_STOP_MARGIN=${HF_GRAPH_STOP_MARGIN}"
echo "[INFO] HF_GRAPH_ROAR_CAND_LIMIT=${HF_GRAPH_ROAR_CAND_LIMIT}"
echo "[INFO] HF_GRAPH_ROAR_ENHANCE_LIMIT=${HF_GRAPH_ROAR_ENHANCE_LIMIT}"
echo "[INFO] HF_GRAPH_ROAR_ENTRY=${HF_GRAPH_ROAR_ENTRY}"
echo "[INFO] HF_GRAPH_ROAR_THREADS=${HF_GRAPH_ROAR_THREADS}"
echo "[INFO] ANSWER_PREFIX_SCAFFOLD=${ANSWER_PREFIX_SCAFFOLD}"
echo "[INFO] ANSWER_CONSTRAINED_CODEBOOK=${ANSWER_CONSTRAINED_CODEBOOK}"
echo "[INFO] FORCE_MAX_DECODE_STEPS=${FORCE_MAX_DECODE_STEPS}"
echo "[INFO] HF_READY_MARKER=${HF_READY_MARKER}"

"${HF_VENV_DIR}/bin/python" benchmark/generated_memory_hf_eval.py \
  --model_name "${MODEL_NAME}" \
  --dtype "${DTYPE}" \
  --device_map "${DEVICE_MAP}" \
  --generation_mode "${GENERATION_MODE}" \
  --num_samples "${NUM_SAMPLES}" \
  --seed "${SEED}" \
  --num_entries "${NUM_ENTRIES}" \
  --num_queries "${NUM_QUERIES}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --prefill_filler_repeats "${PREFILL_FILLER_REPEATS}" \
  --min_prompt_tokens "${MIN_PROMPT_TOKENS}" \
  --trace_decode_steps "${TRACE_DECODE_STEPS}" \
  --trace_max_records "${TRACE_MAX_RECORDS}" \
  --hf_attention_mode "${HF_ATTENTION_MODE}" \
  --hf_sparse_topk "${HF_SPARSE_TOPK}" \
  --hf_sparse_static_prefix "${HF_SPARSE_STATIC_PREFIX}" \
  --hf_sparse_static_suffix "${HF_SPARSE_STATIC_SUFFIX}" \
  --hf_graph_degree "${HF_GRAPH_DEGREE}" \
  --hf_graph_visit_budget "${HF_GRAPH_VISIT_BUDGET}" \
  --hf_graph_seed_count "${HF_GRAPH_SEED_COUNT}" \
  --hf_graph_online_edges "${HF_GRAPH_ONLINE_EDGES}" \
  --hf_graph_search_backend "${HF_GRAPH_SEARCH_BACKEND}" \
  --hf_graph_candidate_target "${HF_GRAPH_CANDIDATE_TARGET}" \
  --hf_graph_expand_width "${HF_GRAPH_EXPAND_WIDTH}" \
  --hf_graph_min_visits "${HF_GRAPH_MIN_VISITS}" \
  --hf_graph_frontier_topn "${HF_GRAPH_FRONTIER_TOPN}" \
  --hf_graph_stop_patience "${HF_GRAPH_STOP_PATIENCE}" \
  --hf_graph_stop_margin "${HF_GRAPH_STOP_MARGIN}" \
  --hf_graph_roar_cand_limit "${HF_GRAPH_ROAR_CAND_LIMIT}" \
  --hf_graph_roar_enhance_limit "${HF_GRAPH_ROAR_ENHANCE_LIMIT}" \
  --hf_graph_roar_entry "${HF_GRAPH_ROAR_ENTRY}" \
  --hf_graph_roar_threads "${HF_GRAPH_ROAR_THREADS}" \
  $( [ -n "${ATTN_IMPLEMENTATION}" ] && printf '%s %s' "--attn_implementation" "${ATTN_IMPLEMENTATION}" ) \
  $( [ "${TRUST_REMOTE_CODE}" = "1" ] && printf '%s' "--trust_remote_code" ) \
  $( [ "${LOCAL_FILES_ONLY}" = "1" ] && printf '%s' "--local_files_only" ) \
  $( [ "${LOW_CPU_MEM_USAGE}" = "1" ] && printf '%s' "--low_cpu_mem_usage" ) \
  $( [ "${HF_LANGUAGE_MODEL_ONLY}" = "1" ] && printf '%s' "--hf_language_model_only" ) \
  $( [ "${USE_CHAT_TEMPLATE}" = "1" ] && printf '%s' "--use_chat_template" ) \
  $( [ "${DISABLE_THINKING}" = "1" ] && printf '%s' "--disable_thinking" ) \
  $( [ "${INVENTORY_ONLY}" = "1" ] && printf '%s' "--inventory_only" ) \
  $( [ "${CONFIG_ONLY}" = "1" ] && printf '%s' "--config_only" ) \
  $( [ "${TOKENIZER_ONLY}" = "1" ] && printf '%s' "--tokenizer_only" ) \
  $( [ "${TRACE_ATTENTION}" = "1" ] && printf '%s' "--trace_attention" ) \
  $( [ "${TRACE_PREFILL}" = "1" ] && printf '%s' "--trace_prefill" ) \
  $( [ "${ANSWER_PREFIX_SCAFFOLD}" = "1" ] && printf '%s' "--answer_prefix_scaffold" ) \
  $( [ "${ANSWER_CONSTRAINED_CODEBOOK}" = "1" ] && printf '%s' "--answer_constrained_codebook" ) \
  $( [ "${FORCE_MAX_DECODE_STEPS}" = "1" ] && printf '%s' "--force_max_decode_steps" ) \
  --output_dir "${OUTPUT_DIR}"
