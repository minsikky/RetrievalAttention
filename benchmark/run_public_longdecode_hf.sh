#!/bin/bash
#SBATCH --job-name=public-ldecode
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128000m
#SBATCH --time=06:00:00
#SBATCH --account=zhengya98
#SBATCH --partition=spgpu
#SBATCH --gpus-per-node=1

set -euo pipefail

cd /gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention

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

HF_MODEL_PRESET="${HF_MODEL_PRESET:-qwen3_8b}"
PRESET_MODEL_NAME=""
PRESET_HF_EXTRA_PYTHONPATH=""
PRESET_QWEN_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS="32768"
case "${HF_MODEL_PRESET}" in
  ""|qwen3_8b)
    PRESET_MODEL_NAME=".hf_cache/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
    PRESET_HF_EXTRA_PYTHONPATH=".hf_pydeps"
    PRESET_QWEN_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS="32768"
    ;;
  llama31_8b|llama3_1_8b)
    PRESET_MODEL_NAME=".hf_cache/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
    PRESET_HF_EXTRA_PYTHONPATH=""
    PRESET_QWEN_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS="32768"
    ;;
  qwen3_5_9b)
    PRESET_MODEL_NAME=".hf_cache/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a"
    PRESET_HF_EXTRA_PYTHONPATH=".hf_pydeps"
    PRESET_QWEN_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS="262144"
    ;;
  *)
    echo "[ERROR] Unknown HF_MODEL_PRESET=${HF_MODEL_PRESET}"
    echo "[ERROR] Supported presets: qwen3_8b, llama31_8b, llama3_1_8b, qwen3_5_9b"
    exit 2
    ;;
esac

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
HF_EXTRA_PYTHONPATH="${HF_EXTRA_PYTHONPATH:-${PRESET_HF_EXTRA_PYTHONPATH}}"
if [ -n "${HF_EXTRA_PYTHONPATH}" ] && [[ "${HF_EXTRA_PYTHONPATH}" != /* ]]; then
  HF_EXTRA_PYTHONPATH="$(pwd)/${HF_EXTRA_PYTHONPATH}"
fi
if [ -n "${HF_EXTRA_PYTHONPATH}" ]; then
  export PYTHONPATH="${HF_EXTRA_PYTHONPATH}"
  if [ -d "${HF_EXTRA_PYTHONPATH}/numpy.libs" ]; then
    export LD_LIBRARY_PATH="${HF_EXTRA_PYTHONPATH}/numpy.libs:${LD_LIBRARY_PATH:-}"
  fi
fi
export TOKENIZERS_PARALLELISM=false

DEFAULT_MODEL_PATH="${PRESET_MODEL_NAME}"
MODEL_NAME="${MODEL_NAME:-${DEFAULT_MODEL_PATH}}"
BENCHMARK="${BENCHMARK:-aime24}"
ATTENTION_MODE="${ATTENTION_MODE:-dense}"
APPROX_PREFILL="${APPROX_PREFILL:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-public_longdecode_result}"
RUN_NAME="${RUN_NAME:-${ATTENTION_MODE}_${BENCHMARK}_smoke}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${RUN_NAME}}"

DTYPE="${DTYPE:-bf16}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
LOW_CPU_MEM_USAGE="${LOW_CPU_MEM_USAGE:-1}"
HF_LANGUAGE_MODEL_ONLY="${HF_LANGUAGE_MODEL_ONLY:-0}"
USE_CHAT_TEMPLATE="${USE_CHAT_TEMPLATE:-1}"
DISABLE_THINKING="${DISABLE_THINKING:-0}"
SEED="${SEED:-2026}"
MAX_EXAMPLES="${MAX_EXAMPLES:-1}"
SELECTION="${SELECTION:-first}"
MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-120000}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
MIN_NEW_TOKENS="${MIN_NEW_TOKENS:-0}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
FORCE_MAX_NEW_TOKENS="${FORCE_MAX_NEW_TOKENS:-0}"
QWEN_YARN_FACTOR="${QWEN_YARN_FACTOR:-0}"
QWEN_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS="${QWEN_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS:-${PRESET_QWEN_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS}}"

LIVE_CODE_RELEASE="${LIVE_CODE_RELEASE:-release_v6}"
LIVE_CODE_START_DATE="${LIVE_CODE_START_DATE:-}"
LIVE_CODE_END_DATE="${LIVE_CODE_END_DATE:-}"
EVALUATE_CODE="${EVALUATE_CODE:-0}"
CODE_EVAL_PROCESSES="${CODE_EVAL_PROCESSES:-4}"
CODE_EVAL_TIMEOUT="${CODE_EVAL_TIMEOUT:-6}"
LONGGENBENCH_GSM8K_K="${LONGGENBENCH_GSM8K_K:-32}"
LONGGENBENCH_GSM8K_QUESTION_LIMIT="${LONGGENBENCH_GSM8K_QUESTION_LIMIT:-256}"

LAYERS="${LAYERS:-all}"
SELECTOR_MODE="${SELECTOR_MODE:-fullscan}"
SELECTOR_BACKEND="${SELECTOR_BACKEND:-cuda_ext}"
BUDGET="${BUDGET:-4096}"
ONLINE_CONFIDENCE_RULE="${ONLINE_CONFIDENCE_RULE:-geometric_probe_tail_switch}"
TAIL_MODE="${TAIL_MODE:-vpq_value}"
TAIL_SCORE_CALIBRATION="${TAIL_SCORE_CALIBRATION:-affine_selected}"
TAIL_PROBE_REL_L2_MAX="${TAIL_PROBE_REL_L2_MAX:-0.020}"
TAIL_PROXY_MASS_MIN="${TAIL_PROXY_MASS_MIN:-0.990}"
TAIL_PROXY_MASS_MAX="${TAIL_PROXY_MASS_MAX:-1.0}"
TAIL_PQ_CORR_MIN="${TAIL_PQ_CORR_MIN:-0.70}"
TAIL_PQ_RELRMSE_MAX="${TAIL_PQ_RELRMSE_MAX:-inf}"
RANKED_CONFIDENCE_COST_MODE="${RANKED_CONFIDENCE_COST_MODE:-exact}"
FRONTIER_CANONICAL_GPU="${FRONTIER_CANONICAL_GPU:-1}"
export FRONTIER_CANONICAL_GPU
GEOMETRIC_MIN_BUDGET="${GEOMETRIC_MIN_BUDGET:-4096}"
GEOMETRIC_MAX_BUDGET="${GEOMETRIC_MAX_BUDGET:-65536}"
GEOMETRIC_GROWTH="${GEOMETRIC_GROWTH:-1.5}"
GEOMETRIC_PROBE_SCALE="${GEOMETRIC_PROBE_SCALE:-1.5}"
GEOMETRIC_BUDGET_GRANULARITY="${GEOMETRIC_BUDGET_GRANULARITY:-1024}"
SELECTED_VALUE_MODE="${SELECTED_VALUE_MODE:-vpq_value}"
SELECTED_VALUE_EXACT_RULE="${SELECTED_VALUE_EXACT_RULE:-selected_mass}"
SELECTED_VALUE_EXACT_TOP="${SELECTED_VALUE_EXACT_TOP:-0}"
SELECTED_VALUE_EXACT_MASS="${SELECTED_VALUE_EXACT_MASS:-0.99}"
SELECTED_VALUE_EXACT_RISK_MASS="${SELECTED_VALUE_EXACT_RISK_MASS:-0.0}"
SELECTED_VALUE_MIN_EXACT_TOP="${SELECTED_VALUE_MIN_EXACT_TOP:-1024}"
SELECTED_VALUE_MAX_EXACT_TOP="${SELECTED_VALUE_MAX_EXACT_TOP:-0}"
SELECTED_VALUE_EXACT_ALL_CONTEXT_MAX="${SELECTED_VALUE_EXACT_ALL_CONTEXT_MAX:-0}"
SELECTED_VALUE_EXACT_ALL_FRACTION_MIN="${SELECTED_VALUE_EXACT_ALL_FRACTION_MIN:-0.0}"
TAIL_BLEND="${TAIL_BLEND:-1.0}"
PAGE_SIZE="${PAGE_SIZE:-5632}"
PREFILL_CHUNK_SIZE="${PREFILL_CHUNK_SIZE:-0}"
PREFILL_SELECTOR_BACKEND="${PREFILL_SELECTOR_BACKEND:-native}"
if [[ -z "${PREFILL_SELECTOR_TILE_SIZE+x}" ]]; then
  if [[ "${PREFILL_SELECTOR_BACKEND}" == "native" ]]; then
    PREFILL_SELECTOR_TILE_SIZE=2048
  else
    PREFILL_SELECTOR_TILE_SIZE=256
  fi
fi
PREFILL_SELECTOR_PAGE_BLOCK_SIZE="${PREFILL_SELECTOR_PAGE_BLOCK_SIZE:-0}"
PREFILL_RANK_BUFFER_LIMIT_MB="${PREFILL_RANK_BUFFER_LIMIT_MB:-4096}"
PREFILL_TAIL_SCORE_REUSE="${PREFILL_TAIL_SCORE_REUSE:-1}"
PREFILL_ATTENTION_BACKEND="${PREFILL_ATTENTION_BACKEND:-native}"
SUBVECS="${SUBVECS:-4}"
SUBBITS="${SUBBITS:-8}"
VALUE_SUBVECS="${VALUE_SUBVECS:-1}"
VALUE_SUBBITS="${VALUE_SUBBITS:-4}"
VALUE_PQ_GROUP_PAGES="${VALUE_PQ_GROUP_PAGES:-1}"
KMEANS_ITERS="${KMEANS_ITERS:-3}"
INDEX_BUILD_BACKEND="${INDEX_BUILD_BACKEND:-torch_gpu}"
NPROBES="${NPROBES:-16,32,64,128,256,512}"
PROFILE_NATIVE_OPS="${PROFILE_NATIVE_OPS:-0}"
DISABLE_COST_STATS="${DISABLE_COST_STATS:-0}"
DISABLE_NATIVE_DECODE_FUSED="${DISABLE_NATIVE_DECODE_FUSED:-1}"
ENABLE_NATIVE_DECODE_FUSED="${ENABLE_NATIVE_DECODE_FUSED:-0}"
NATIVE_DECODE_SCORELESS_FUSED="${NATIVE_DECODE_SCORELESS_FUSED:-0}"
NATIVE_DECODE_SCORELESS_FORCE_MODE="${NATIVE_DECODE_SCORELESS_FORCE_MODE:-2}"
NATIVE_DECODE_TAIL="${NATIVE_DECODE_TAIL:-1}"

echo "[INFO] Job started at: $(date)"
echo "[INFO] Host: $(hostname)"
echo "[INFO] HF_MODEL_PRESET=${HF_MODEL_PRESET}"
echo "[INFO] BENCHMARK=${BENCHMARK}"
echo "[INFO] ATTENTION_MODE=${ATTENTION_MODE}"
echo "[INFO] APPROX_PREFILL=${APPROX_PREFILL}"
echo "[INFO] MODEL_NAME=${MODEL_NAME}"
echo "[INFO] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[INFO] MAX_EXAMPLES=${MAX_EXAMPLES}"
echo "[INFO] MAX_INPUT_TOKENS=${MAX_INPUT_TOKENS}"
echo "[INFO] MAX_NEW_TOKENS=${MAX_NEW_TOKENS}"
echo "[INFO] MIN_NEW_TOKENS=${MIN_NEW_TOKENS}"
echo "[INFO] FORCE_MAX_NEW_TOKENS=${FORCE_MAX_NEW_TOKENS}"
echo "[INFO] LOCAL_FILES_ONLY=${LOCAL_FILES_ONLY}"
echo "[INFO] HF_EXTRA_PYTHONPATH=${HF_EXTRA_PYTHONPATH}"
echo "[INFO] ONLINE_CONFIDENCE_RULE=${ONLINE_CONFIDENCE_RULE}"
echo "[INFO] FRONTIER_CANONICAL_GPU=${FRONTIER_CANONICAL_GPU}"
echo "[INFO] SELECTED_VALUE_MODE=${SELECTED_VALUE_MODE}"
echo "[INFO] DISABLE_NATIVE_DECODE_FUSED=${DISABLE_NATIVE_DECODE_FUSED}"
echo "[INFO] ENABLE_NATIVE_DECODE_FUSED=${ENABLE_NATIVE_DECODE_FUSED}"
echo "[INFO] NATIVE_DECODE_SCORELESS_FUSED=${NATIVE_DECODE_SCORELESS_FUSED}"
echo "[INFO] PREFILL_SELECTOR_BACKEND=${PREFILL_SELECTOR_BACKEND}"
echo "[INFO] PREFILL_RANK_BUFFER_LIMIT_MB=${PREFILL_RANK_BUFFER_LIMIT_MB}"

"${HF_VENV_DIR}/bin/python" benchmark/public_longdecode_eval.py \
  --benchmark "${BENCHMARK}" \
  --output_dir "${OUTPUT_DIR}" \
  --model_name "${MODEL_NAME}" \
  --dtype "${DTYPE}" \
  --device_map "${DEVICE_MAP}" \
  --max_examples "${MAX_EXAMPLES}" \
  --selection "${SELECTION}" \
  --seed "${SEED}" \
  --max_input_tokens "${MAX_INPUT_TOKENS}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --min_new_tokens "${MIN_NEW_TOKENS}" \
  --temperature "${TEMPERATURE}" \
  --top_p "${TOP_P}" \
  --qwen_yarn_factor "${QWEN_YARN_FACTOR}" \
  --qwen_yarn_original_max_position_embeddings "${QWEN_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS}" \
  --attention_mode "${ATTENTION_MODE}" \
  $( [ "${APPROX_PREFILL}" = "1" ] && printf '%s' "--approx_prefill" ) \
  --layers "${LAYERS}" \
  --livecodebench_release "${LIVE_CODE_RELEASE}" \
  --livecodebench_start_date "${LIVE_CODE_START_DATE}" \
  --livecodebench_end_date "${LIVE_CODE_END_DATE}" \
  --code_eval_processes "${CODE_EVAL_PROCESSES}" \
  --code_eval_timeout "${CODE_EVAL_TIMEOUT}" \
  --longgenbench_gsm8k_k "${LONGGENBENCH_GSM8K_K}" \
  --longgenbench_gsm8k_question_limit "${LONGGENBENCH_GSM8K_QUESTION_LIMIT}" \
  --selector_mode "${SELECTOR_MODE}" \
  --selector_backend "${SELECTOR_BACKEND}" \
  --budget "${BUDGET}" \
  --online_confidence_rule "${ONLINE_CONFIDENCE_RULE}" \
  --tail_mode "${TAIL_MODE}" \
  --tail_score_calibration "${TAIL_SCORE_CALIBRATION}" \
  --tail_probe_rel_l2_max "${TAIL_PROBE_REL_L2_MAX}" \
  --tail_proxy_mass_min "${TAIL_PROXY_MASS_MIN}" \
  --tail_proxy_mass_max "${TAIL_PROXY_MASS_MAX}" \
  --tail_pq_corr_min "${TAIL_PQ_CORR_MIN}" \
  --tail_pq_relrmse_max "${TAIL_PQ_RELRMSE_MAX}" \
  --ranked_confidence_cost_mode "${RANKED_CONFIDENCE_COST_MODE}" \
  --geometric_min_budget "${GEOMETRIC_MIN_BUDGET}" \
  --geometric_max_budget "${GEOMETRIC_MAX_BUDGET}" \
  --geometric_growth "${GEOMETRIC_GROWTH}" \
  --geometric_probe_scale "${GEOMETRIC_PROBE_SCALE}" \
  --geometric_budget_granularity "${GEOMETRIC_BUDGET_GRANULARITY}" \
  --selected_value_mode "${SELECTED_VALUE_MODE}" \
  --selected_value_exact_rule "${SELECTED_VALUE_EXACT_RULE}" \
  --selected_value_exact_top "${SELECTED_VALUE_EXACT_TOP}" \
  --selected_value_exact_mass "${SELECTED_VALUE_EXACT_MASS}" \
  --selected_value_exact_risk_mass "${SELECTED_VALUE_EXACT_RISK_MASS}" \
  --selected_value_min_exact_top "${SELECTED_VALUE_MIN_EXACT_TOP}" \
  --selected_value_max_exact_top "${SELECTED_VALUE_MAX_EXACT_TOP}" \
  --selected_value_exact_all_context_max "${SELECTED_VALUE_EXACT_ALL_CONTEXT_MAX}" \
  --selected_value_exact_all_fraction_min "${SELECTED_VALUE_EXACT_ALL_FRACTION_MIN}" \
  --tail_blend "${TAIL_BLEND}" \
  --page_size "${PAGE_SIZE}" \
  --prefill_chunk_size "${PREFILL_CHUNK_SIZE}" \
  --prefill_selector_backend "${PREFILL_SELECTOR_BACKEND}" \
  --prefill_selector_tile_size "${PREFILL_SELECTOR_TILE_SIZE}" \
  --prefill_rank_buffer_limit_mb "${PREFILL_RANK_BUFFER_LIMIT_MB}" \
  --prefill_selector_page_block_size "${PREFILL_SELECTOR_PAGE_BLOCK_SIZE}" \
  --prefill_attention_backend "${PREFILL_ATTENTION_BACKEND}" \
  --subvecs "${SUBVECS}" \
  --subbits "${SUBBITS}" \
  --value_subvecs "${VALUE_SUBVECS}" \
  --value_subbits "${VALUE_SUBBITS}" \
  --value_pq_group_pages "${VALUE_PQ_GROUP_PAGES}" \
  --kmeans_iters "${KMEANS_ITERS}" \
  --index_build_backend "${INDEX_BUILD_BACKEND}" \
  --nprobes "${NPROBES}" \
  $( [ "${TRUST_REMOTE_CODE}" = "1" ] && printf '%s' "--trust_remote_code" ) \
  $( [ "${LOCAL_FILES_ONLY}" = "1" ] && printf '%s' "--local_files_only" ) \
  $( [ "${LOW_CPU_MEM_USAGE}" = "1" ] && printf '%s' "--low_cpu_mem_usage" ) \
  $( [ "${HF_LANGUAGE_MODEL_ONLY}" = "1" ] && printf '%s' "--hf_language_model_only" ) \
  $( [ "${USE_CHAT_TEMPLATE}" = "1" ] && printf '%s' "--use_chat_template" ) \
  $( [ "${DISABLE_THINKING}" = "1" ] && printf '%s' "--disable_thinking" ) \
  $( [ "${FORCE_MAX_NEW_TOKENS}" = "1" ] && printf '%s' "--force_max_new_tokens" ) \
  $( [ "${EVALUATE_CODE}" = "1" ] && printf '%s' "--evaluate_code" ) \
  $( [ "${PREFILL_TAIL_SCORE_REUSE}" = "1" ] && printf '%s' "--prefill_tail_score_reuse" ) \
  $( [ "${PROFILE_NATIVE_OPS}" = "1" ] && printf '%s' "--profile_native_ops" ) \
  $( [ "${DISABLE_COST_STATS}" = "1" ] && printf '%s' "--disable_cost_stats" ) \
  $( [ "${DISABLE_NATIVE_DECODE_FUSED}" = "1" ] && printf '%s' "--disable_native_decode_fused" ) \
  $( [ "${ENABLE_NATIVE_DECODE_FUSED}" = "1" ] && printf '%s' "--enable_native_decode_fused" ) \
  $( [ "${NATIVE_DECODE_SCORELESS_FUSED}" = "1" ] && printf '%s %s' "--native_decode_scoreless_fused --native_decode_scoreless_force_mode" "${NATIVE_DECODE_SCORELESS_FORCE_MODE}" ) \
  $( [ "${NATIVE_DECODE_TAIL}" = "1" ] && printf '%s' "--native_decode_tail" ) \
  $( [ -n "${ATTN_IMPLEMENTATION}" ] && printf '%s %s' "--attn_implementation" "${ATTN_IMPLEMENTATION}" )
