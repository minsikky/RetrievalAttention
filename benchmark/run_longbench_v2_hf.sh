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
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-${PRESET_MODEL_NAME}}"
if [ -e "${MODEL_NAME}" ]; then
  MODEL_NAME="$(readlink -f "${MODEL_NAME}")"
fi
if [ "${STAGE_MODEL_TO_TMP:-0}" = "1" ]; then
  if [ ! -d "${MODEL_NAME}" ]; then
    echo "[ERROR] STAGE_MODEL_TO_TMP=1 requires MODEL_NAME to be a local directory: ${MODEL_NAME}" >&2
    exit 1
  fi
  TMP_MODEL_ROOT="${SLURM_TMPDIR:-/tmp/${USER}/longbench_model_${SLURM_JOB_ID:-manual}}"
  mkdir -p "${TMP_MODEL_ROOT}"
  echo "[INFO] Staging model to ${TMP_MODEL_ROOT}"
  if command -v rsync >/dev/null 2>&1; then
    # HF snapshot directories are symlink farms into blobs/. Dereference them
    # when staging, otherwise /tmp gets broken ../../blobs links.
    rsync -aL --delete "${MODEL_NAME}/" "${TMP_MODEL_ROOT}/"
  else
    cp -aL "${MODEL_NAME}/." "${TMP_MODEL_ROOT}/"
  fi
  MODEL_NAME="${TMP_MODEL_ROOT}"
fi
DTYPE="${DTYPE:-bf16}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-${PRESET_TRUST_REMOTE_CODE:-0}}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-0}"
LOW_CPU_MEM_USAGE="${LOW_CPU_MEM_USAGE:-1}"
HF_LANGUAGE_MODEL_ONLY="${HF_LANGUAGE_MODEL_ONLY:-${PRESET_HF_LANGUAGE_MODEL_ONLY:-1}}"
USE_CHAT_TEMPLATE="${USE_CHAT_TEMPLATE:-${PRESET_USE_CHAT_TEMPLATE:-1}}"
DISABLE_THINKING="${DISABLE_THINKING:-${PRESET_DISABLE_THINKING:-1}}"
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
PROFILE_EXECUTION="${PROFILE_EXECUTION:-0}"
DATASET_SCAN_LIMIT="${DATASET_SCAN_LIMIT:-1000}"
QWEN_YARN_FACTOR="${QWEN_YARN_FACTOR:-0}"
QWEN_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS="${QWEN_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS:-${PRESET_QWEN_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS}}"
ATTENTION_MODE="${ATTENTION_MODE:-dense}"
APPROX_PREFILL="${APPROX_PREFILL:-0}"
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
echo "[INFO] INDEX_BUILD_BACKEND=${INDEX_BUILD_BACKEND}"
echo "[INFO] DIAGNOSE_DENSE_REFERENCE=${DIAGNOSE_DENSE_REFERENCE}"
echo "[INFO] PROFILE_EXECUTION=${PROFILE_EXECUTION}"
echo "[INFO] PROFILE_NATIVE_OPS=${PROFILE_NATIVE_OPS}"
echo "[INFO] DISABLE_COST_STATS=${DISABLE_COST_STATS}"
echo "[INFO] DISABLE_NATIVE_DECODE_FUSED=${DISABLE_NATIVE_DECODE_FUSED}"
echo "[INFO] ENABLE_NATIVE_DECODE_FUSED=${ENABLE_NATIVE_DECODE_FUSED}"
echo "[INFO] NATIVE_DECODE_SCORELESS_FUSED=${NATIVE_DECODE_SCORELESS_FUSED}"
echo "[INFO] NATIVE_DECODE_TAIL=${NATIVE_DECODE_TAIL}"
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
  --attention_mode "${ATTENTION_MODE}" \
  --layers "${LAYERS}" \
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
  $( [ -n "${PREFILL_TAIL_BLEND}" ] && printf '%s %s' "--prefill_tail_blend" "${PREFILL_TAIL_BLEND}" ) \
  $( [ -n "${DECODE_TAIL_BLEND}" ] && printf '%s %s' "--decode_tail_blend" "${DECODE_TAIL_BLEND}" ) \
  --page_size "${PAGE_SIZE}" \
  --prefill_chunk_size "${PREFILL_CHUNK_SIZE}" \
  --prefill_selector_backend "${PREFILL_SELECTOR_BACKEND}" \
  --prefill_selector_tile_size "${PREFILL_SELECTOR_TILE_SIZE}" \
  --prefill_rank_buffer_limit_mb "${PREFILL_RANK_BUFFER_LIMIT_MB}" \
  --prefill_selector_page_block_size "${PREFILL_SELECTOR_PAGE_BLOCK_SIZE}" \
  $( [ "${PREFILL_TAIL_SCORE_REUSE}" = "1" ] && printf '%s' "--prefill_tail_score_reuse" ) \
  --prefill_attention_backend "${PREFILL_ATTENTION_BACKEND}" \
  --subvecs "${SUBVECS}" \
  --subbits "${SUBBITS}" \
  --value_subvecs "${VALUE_SUBVECS}" \
  --value_subbits "${VALUE_SUBBITS}" \
  --value_pq_group_pages "${VALUE_PQ_GROUP_PAGES}" \
  --kmeans_iters "${KMEANS_ITERS}" \
  --index_build_backend "${INDEX_BUILD_BACKEND}" \
  --nprobes "${NPROBES}" \
  $( [ -n "${ATTN_IMPLEMENTATION}" ] && printf '%s %s' "--attn_implementation" "${ATTN_IMPLEMENTATION}" ) \
  $( [ "${TRUST_REMOTE_CODE}" = "1" ] && printf '%s' "--trust_remote_code" ) \
  $( [ "${LOCAL_FILES_ONLY}" = "1" ] && printf '%s' "--local_files_only" ) \
  $( [ "${LOW_CPU_MEM_USAGE}" = "1" ] && printf '%s' "--low_cpu_mem_usage" ) \
  $( [ "${HF_LANGUAGE_MODEL_ONLY}" = "1" ] && printf '%s' "--hf_language_model_only" ) \
  $( [ "${USE_CHAT_TEMPLATE}" = "1" ] && printf '%s' "--use_chat_template" ) \
  $( [ "${DISABLE_THINKING}" = "1" ] && printf '%s' "--disable_thinking" ) \
  $( [ "${STREAMING}" = "1" ] && printf '%s' "--streaming" ) \
  $( [ "${PROFILE_EXECUTION}" = "1" ] && printf '%s' "--profile_execution" ) \
  $( [ "${PROFILE_NATIVE_OPS}" = "1" ] && printf '%s' "--profile_native_ops" ) \
  $( [ "${DISABLE_COST_STATS}" = "1" ] && printf '%s' "--disable_cost_stats" ) \
  $( [ "${DISABLE_NATIVE_DECODE_FUSED}" = "1" ] && printf '%s' "--disable_native_decode_fused" ) \
  $( [ "${ENABLE_NATIVE_DECODE_FUSED}" = "1" ] && printf '%s' "--enable_native_decode_fused" ) \
  $( [ "${NATIVE_DECODE_SCORELESS_FUSED}" = "1" ] && printf '%s %s' "--native_decode_scoreless_fused --native_decode_scoreless_force_mode" "${NATIVE_DECODE_SCORELESS_FORCE_MODE}" ) \
  $( [ "${ALLOW_TF32_SELECTOR:-0}" = "1" ] && printf '%s' "--allow_tf32_selector" ) \
  $( [ "${NATIVE_DECODE_TAIL}" = "1" ] && printf '%s' "--native_decode_tail" ) \
  $( [ "${DIAGNOSE_DENSE_REFERENCE}" = "1" ] && printf '%s' "--diagnose_dense_reference" )

echo "[INFO] Job finished at: $(date)"
