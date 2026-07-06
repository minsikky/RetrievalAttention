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
source scripts/hf_model_presets.sh
resolve_hf_model_preset "${HF_MODEL_PRESET}" || exit $?

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
if [ -n "${HF_EXTRA_PYTHONPATH}" ] && [[ "${HF_EXTRA_PYTHONPATH}" != /* ]]; then
  HF_EXTRA_PYTHONPATH="$(pwd)/${HF_EXTRA_PYTHONPATH}"
fi
if [ -n "${HF_EXTRA_PYTHONPATH}" ]; then
  export PYTHONPATH="${HF_EXTRA_PYTHONPATH}"
  if [ -d "${HF_EXTRA_PYTHONPATH}/numpy.libs" ]; then
    export LD_LIBRARY_PATH="${HF_EXTRA_PYTHONPATH}/numpy.libs:${LD_LIBRARY_PATH:-}"
  fi
fi
export LD_LIBRARY_PATH="$PWD/${HF_VENV_DIR}/lib/python3.10/site-packages/torch/lib:/sw/pkgs/arc/python/3.10.4/lib:${LD_LIBRARY_PATH:-}"
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
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-${PRESET_TRUST_REMOTE_CODE:-0}}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
LOW_CPU_MEM_USAGE="${LOW_CPU_MEM_USAGE:-1}"
HF_LANGUAGE_MODEL_ONLY="${HF_LANGUAGE_MODEL_ONLY:-${PRESET_HF_LANGUAGE_MODEL_ONLY:-0}}"
USE_CHAT_TEMPLATE="${USE_CHAT_TEMPLATE:-${PRESET_USE_CHAT_TEMPLATE:-1}}"
QWEN3_AIME_MAX_NEW_TOKENS=38912
QWEN3_DEFAULT_MAX_NEW_TOKENS=32768
USE_QWEN3_OFFICIAL_EVAL_DEFAULTS="${USE_QWEN3_OFFICIAL_EVAL_DEFAULTS:-1}"
QWEN3_EVAL_MODE="${QWEN3_EVAL_MODE:-thinking}"
is_qwen3=0
if [[ "${HF_MODEL_PRESET}" == qwen3* ]] || [[ "${MODEL_NAME}" == *Qwen3* ]] || [[ "${MODEL_NAME}" == *Qwen--Qwen3* ]]; then
  is_qwen3=1
fi
is_qwen3_report_benchmark=0
case "${BENCHMARK}" in
  aime24|gpqa|livecodebench_codegen)
    is_qwen3_report_benchmark=1
    ;;
esac
if [ "${USE_QWEN3_OFFICIAL_EVAL_DEFAULTS}" = "1" ] && [ "${is_qwen3}" = "1" ] && [ "${is_qwen3_report_benchmark}" = "1" ]; then
  case "${QWEN3_EVAL_MODE}" in
    thinking|think)
      default_disable_thinking=0
      default_temperature=0.6
      default_top_p=0.95
      default_top_k=20
      ;;
    nonthinking|non-thinking|no_think|nothink)
      default_disable_thinking=1
      default_temperature=0.7
      default_top_p=0.8
      default_top_k=20
      ;;
    *)
      echo "[ERROR] Unknown QWEN3_EVAL_MODE=${QWEN3_EVAL_MODE}" >&2
      exit 2
      ;;
  esac
else
  default_disable_thinking="${PRESET_DISABLE_THINKING:-0}"
  default_temperature=0.0
  default_top_p=1.0
  default_top_k=0
fi
case "${BENCHMARK}" in
  aime24)
    default_max_new_tokens="${QWEN3_AIME_MAX_NEW_TOKENS}"
    ;;
  gpqa|livecodebench_codegen)
    default_max_new_tokens="${QWEN3_DEFAULT_MAX_NEW_TOKENS}"
    ;;
  helmet_rag)
    default_max_new_tokens=20
    ;;
  helmet_recall)
    default_max_new_tokens=100
    ;;
  helmet_longqa)
    default_max_new_tokens=100
    ;;
  longproc_2k)
    default_max_new_tokens=3072
    ;;
  longproc_8k)
    default_max_new_tokens=9216
    ;;
  *)
    default_max_new_tokens=1024
    ;;
esac
case "${BENCHMARK}" in
  longproc_2k)
    default_min_new_tokens=2048
    default_force_max_new_tokens=1
    ;;
  longproc_8k)
    default_min_new_tokens=8192
    default_force_max_new_tokens=1
    ;;
  *)
    default_min_new_tokens=0
    default_force_max_new_tokens=0
    ;;
esac
DISABLE_THINKING="${DISABLE_THINKING:-${PUBLIC_DISABLE_THINKING:-${default_disable_thinking}}}"
SEED="${SEED:-2026}"
MAX_EXAMPLES="${MAX_EXAMPLES:-1}"
TASK_OFFSET="${TASK_OFFSET:-0}"
SELECTION="${SELECTION:-first}"
MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-120000}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-${default_max_new_tokens}}"
MIN_NEW_TOKENS="${MIN_NEW_TOKENS:-${default_min_new_tokens}}"
TEMPERATURE="${TEMPERATURE:-${PUBLIC_TEMPERATURE:-${default_temperature}}}"
TOP_P="${TOP_P:-${PUBLIC_TOP_P:-${default_top_p}}}"
TOP_K="${TOP_K:-${PUBLIC_TOP_K:-${default_top_k}}}"
FORCE_MAX_NEW_TOKENS="${FORCE_MAX_NEW_TOKENS:-${default_force_max_new_tokens}}"
DRY_RUN="${DRY_RUN:-0}"
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
HELMET_REPO="${HELMET_REPO:-third_party/benchmarks/HELMET}"
HELMET_DATA_DIR="${HELMET_DATA_DIR:-third_party/benchmarks/HELMET/data}"
HELMET_DATASET_FILTER="${HELMET_DATASET_FILTER:-}"
LONGPROC_REPO="${LONGPROC_REPO:-third_party/benchmarks/LongProc}"
LONGPROC_DATA_DIR="${LONGPROC_DATA_DIR:-third_party/benchmarks/LongProc/data}"
LONGPROC_DATASETS="${LONGPROC_DATASETS:-}"

LAYERS="${LAYERS:-all}"
SELECTOR_MODE="${SELECTOR_MODE:-fullscan}"
SELECTOR_BACKEND="${SELECTOR_BACKEND:-cuda_ext}"
BUDGET="${BUDGET:-4096}"
ONLINE_CONFIDENCE_RULE="${ONLINE_CONFIDENCE_RULE:-joint_kv_stability}"
TAIL_MODE="${TAIL_MODE:-vpq_value}"
TAIL_SCORE_CALIBRATION="${TAIL_SCORE_CALIBRATION:-none}"
TAIL_PROBE_REL_L2_MAX="${TAIL_PROBE_REL_L2_MAX:-0.020}"
TAIL_PROXY_MASS_MIN="${TAIL_PROXY_MASS_MIN:-0.990}"
TAIL_PROXY_MASS_MAX="${TAIL_PROXY_MASS_MAX:-1.0}"
TAIL_PQ_CORR_MIN="${TAIL_PQ_CORR_MIN:-0.70}"
TAIL_PQ_RELRMSE_MAX="${TAIL_PQ_RELRMSE_MAX:-inf}"
RANKED_CONFIDENCE_COST_MODE="${RANKED_CONFIDENCE_COST_MODE:-exact}"
FRONTIER_EXACT_LOGIT_BACKEND="${FRONTIER_EXACT_LOGIT_BACKEND:-auto}"
FRONTIER_CANONICAL_GPU="${FRONTIER_CANONICAL_GPU:-1}"
export FRONTIER_CANONICAL_GPU
source scripts/frontier_canonical_env.sh
source scripts/frontier_direct_runtime_env.sh
echo "[INFO] DISABLE_NATIVE_DECODE_FUSED=${DISABLE_NATIVE_DECODE_FUSED}"
echo "[INFO] ENABLE_NATIVE_DECODE_FUSED=${ENABLE_NATIVE_DECODE_FUSED}"
echo "[INFO] NATIVE_DECODE_SCORELESS_FUSED=${NATIVE_DECODE_SCORELESS_FUSED}"
echo "[INFO] PREFILL_SELECTOR_BACKEND=${PREFILL_SELECTOR_BACKEND}"
echo "[INFO] PREFILL_RANK_BUFFER_LIMIT_MB=${PREFILL_RANK_BUFFER_LIMIT_MB}"
echo "[INFO] MODEL_NAME=${MODEL_NAME}"
echo "[INFO] USE_CHAT_TEMPLATE=${USE_CHAT_TEMPLATE}"
echo "[INFO] DISABLE_THINKING=${DISABLE_THINKING}"
echo "[INFO] QWEN3_EVAL_MODE=${QWEN3_EVAL_MODE}"
echo "[INFO] MAX_NEW_TOKENS=${MAX_NEW_TOKENS}"
echo "[INFO] FORCE_MAX_NEW_TOKENS=${FORCE_MAX_NEW_TOKENS}"
echo "[INFO] TEMPERATURE=${TEMPERATURE}"
echo "[INFO] TOP_P=${TOP_P}"
echo "[INFO] TOP_K=${TOP_K}"

"${HF_VENV_DIR}/bin/python" benchmark/public_longdecode_eval.py \
  --benchmark "${BENCHMARK}" \
  --output_dir "${OUTPUT_DIR}" \
  --model_name "${MODEL_NAME}" \
  --dtype "${DTYPE}" \
  --device_map "${DEVICE_MAP}" \
  --max_examples "${MAX_EXAMPLES}" \
  --task_offset "${TASK_OFFSET}" \
  --selection "${SELECTION}" \
  --seed "${SEED}" \
  --max_input_tokens "${MAX_INPUT_TOKENS}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --min_new_tokens "${MIN_NEW_TOKENS}" \
  --temperature "${TEMPERATURE}" \
  --top_p "${TOP_P}" \
  --top_k "${TOP_K}" \
  --qwen_yarn_factor "${QWEN_YARN_FACTOR}" \
  --qwen_yarn_original_max_position_embeddings "${QWEN_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS}" \
  --attention_mode "${ATTENTION_MODE}" \
  --layers "${LAYERS}" \
  --livecodebench_release "${LIVE_CODE_RELEASE}" \
  --livecodebench_start_date "${LIVE_CODE_START_DATE}" \
  --livecodebench_end_date "${LIVE_CODE_END_DATE}" \
  --code_eval_processes "${CODE_EVAL_PROCESSES}" \
  --code_eval_timeout "${CODE_EVAL_TIMEOUT}" \
  --helmet_repo "${HELMET_REPO}" \
  --helmet_data_dir "${HELMET_DATA_DIR}" \
  --helmet_dataset_filter "${HELMET_DATASET_FILTER}" \
  --longproc_repo "${LONGPROC_REPO}" \
  --longproc_data_dir "${LONGPROC_DATA_DIR}" \
  --longproc_datasets "${LONGPROC_DATASETS}" \
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
  --exact_logit_backend "${FRONTIER_EXACT_LOGIT_BACKEND}" \
  --geometric_min_budget "${GEOMETRIC_MIN_BUDGET}" \
  --geometric_max_budget "${GEOMETRIC_MAX_BUDGET}" \
  --geometric_growth "${GEOMETRIC_GROWTH}" \
  --geometric_probe_scale "${GEOMETRIC_PROBE_SCALE}" \
  --geometric_budget_granularity "${GEOMETRIC_BUDGET_GRANULARITY}" \
  --joint_kv_policy "${JOINT_KV_POLICY}" \
  --joint_kv_k_budgets "${JOINT_KV_K_BUDGETS}" \
  --joint_kv_v_budgets "${JOINT_KV_V_BUDGETS}" \
  --joint_kv_k_budget_fracs "${JOINT_KV_K_BUDGET_FRACS}" \
  --joint_kv_v_budget_fracs "${JOINT_KV_V_BUDGET_FRACS}" \
  --joint_kv_stability_threshold "${JOINT_KV_STABILITY_THRESHOLD}" \
  --joint_kv_threshold_mode "${JOINT_KV_THRESHOLD_MODE}" \
  --joint_kv_threshold_reference_frac "${JOINT_KV_THRESHOLD_REFERENCE_FRAC}" \
  --joint_kv_threshold_scale_shape "${JOINT_KV_THRESHOLD_SCALE_SHAPE}" \
  --joint_kv_threshold_min_scale "${JOINT_KV_THRESHOLD_MIN_SCALE}" \
  --joint_kv_threshold_max_scale "${JOINT_KV_THRESHOLD_MAX_SCALE}" \
  --joint_kv_start_strategy "${JOINT_KV_START_STRATEGY}" \
  --selected_value_mode "${SELECTED_VALUE_MODE}" \
  --selected_value_exact_rule "${SELECTED_VALUE_EXACT_RULE}" \
  --selected_value_exact_top "${SELECTED_VALUE_EXACT_TOP}" \
  --selected_value_exact_mass "${SELECTED_VALUE_EXACT_MASS}" \
  --selected_value_exact_risk_mass "${SELECTED_VALUE_EXACT_RISK_MASS}" \
  --selected_value_min_exact_top "${SELECTED_VALUE_MIN_EXACT_TOP}" \
  --selected_value_max_exact_top "${SELECTED_VALUE_MAX_EXACT_TOP}" \
  --selected_value_exact_all_context_max "${SELECTED_VALUE_EXACT_ALL_CONTEXT_MAX}" \
  --selected_value_exact_all_fraction_min "${SELECTED_VALUE_EXACT_ALL_FRACTION_MIN}" \
  --value_code_stat_bytes "${VALUE_CODE_STAT_BYTES}" \
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
  $( [ "${DISABLE_NATIVE_DECODE_FUSED}" = "1" ] && printf '%s' "--disable_native_decode_fused" ) \
  $( [ "${ENABLE_NATIVE_DECODE_FUSED}" = "1" ] && printf '%s' "--enable_native_decode_fused" ) \
  $( [ "${NATIVE_DECODE_SCORELESS_FUSED}" = "1" ] && printf '%s %s' "--native_decode_scoreless_fused --native_decode_scoreless_force_mode" "${NATIVE_DECODE_SCORELESS_FORCE_MODE}" ) \
  $( [ "${NATIVE_DECODE_TAIL}" = "1" ] && printf '%s' "--native_decode_tail" ) \
  $( [ "${DRY_RUN}" = "1" ] && printf '%s' "--dry_run" ) \
  $( [ -n "${ATTN_IMPLEMENTATION}" ] && printf '%s %s' "--attn_implementation" "${ATTN_IMPLEMENTATION}" )
