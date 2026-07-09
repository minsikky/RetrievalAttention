#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "${SCRIPT_DIR}/.." && pwd)"
export LD_LIBRARY_PATH="/sw/pkgs/arc/python/3.10.4/lib:${LD_LIBRARY_PATH:-}"

TASK_NAME="${TASK_NAME:-niah_single_1}"
CONTEXT_LEN="${CONTEXT_LEN:-2048}"
NUM_SAMPLES="${NUM_SAMPLES:-2}"
MODE="${MODE:-pagedpq_batched}"
APPROX_PREFILL="${APPROX_PREFILL:-0}"
RUN_NAME="${RUN_NAME:-${MODE}_${TASK_NAME}_${CONTEXT_LEN}_n${NUM_SAMPLES}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-ruler_eval_result/pagedpq_stream_smoke_20260514}"
DEFAULT_MODEL_PATH=".hf_cache/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
MODEL_PATH="${MODEL_NAME:-${DEFAULT_MODEL_PATH}}"
MODEL_PATH="$(readlink -f "${MODEL_PATH}")"
if [ "${STAGE_MODEL_TO_TMP:-0}" = "1" ]; then
  TMP_MODEL_ROOT="${SLURM_TMPDIR:-/tmp/${USER}/ruler_model_${SLURM_JOB_ID:-manual}}"
  mkdir -p "${TMP_MODEL_ROOT}"
  echo "[pagedpq_stream_smoke] staging model to ${TMP_MODEL_ROOT}"
  if command -v rsync >/dev/null 2>&1; then
    rsync -aL --delete "${MODEL_PATH}/" "${TMP_MODEL_ROOT}/"
  else
    cp -aL "${MODEL_PATH}/." "${TMP_MODEL_ROOT}/"
  fi
  MODEL_PATH="${TMP_MODEL_ROOT}"
fi
OUT_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
DATA_DIR="${OUT_DIR}/data"
PRED_DIR="${OUT_DIR}/pred"
SUMMARY_DIR="${OUT_DIR}/summary"

mkdir -p "${DATA_DIR}" "${PRED_DIR}" "${SUMMARY_DIR}"

module purge
module load python/3.10.4
module unload pytorch 2>/dev/null || true
module load cuda/12.8.1

HF_VENV_DIR="${HF_VENV_DIR:-.venv}"
if [[ "${HF_VENV_DIR}" != /* ]]; then
  HF_VENV_DIR="$(pwd)/${HF_VENV_DIR}"
fi
if [ -f "${HF_VENV_DIR}/bin/activate" ]; then
  # shellcheck disable=SC1090
  source "${HF_VENV_DIR}/bin/activate"
else
  echo "[ERROR] ${HF_VENV_DIR}/bin/activate not found." >&2
  exit 1
fi

unset PYTHONPATH
unset PYTHONHOME
export PYTHONNOUSERSITE=1
HF_EXTRA_PYTHONPATH="${HF_EXTRA_PYTHONPATH:-}"
if [ -n "${HF_EXTRA_PYTHONPATH}" ] && [[ "${HF_EXTRA_PYTHONPATH}" != /* ]]; then
  HF_EXTRA_PYTHONPATH="$(pwd)/${HF_EXTRA_PYTHONPATH}"
fi
if [ -n "${HF_EXTRA_PYTHONPATH}" ]; then
  export PYTHONPATH="${HF_EXTRA_PYTHONPATH}:$(pwd)/benchmark/selector_eval/cuda_ext"
else
  export PYTHONPATH="$(pwd)/benchmark/selector_eval/cuda_ext"
fi
if [ -d "${HF_EXTRA_PYTHONPATH:-}/numpy.libs" ]; then
  export LD_LIBRARY_PATH="${HF_EXTRA_PYTHONPATH}/numpy.libs:${LD_LIBRARY_PATH:-}"
fi
export LD_LIBRARY_PATH="${HF_VENV_DIR}/lib/python3.10/site-packages/torch/lib:/sw/pkgs/arc/python/3.10.4/lib:${LD_LIBRARY_PATH:-}"

if [ -n "${JOINT_KV_DEESCALATE:-}" ] || [ -n "${JOINT_KV_PRECISION_TIERS:-}" ]; then
  # Frozen-sim precision tiers currently require the canonical torch grid
  # path: the precision tiers live in the torch score grid + sorted
  # V-prefix, while the native score-grid/risk-prefix kernels are
  # single-tier. JOINT_KV_DEESCALATE is historical/repro-only but also
  # needs the torch policy path if explicitly enabled. These exports win
  # over frontier_canonical_env.sh, which the batched wrapper sources
  # earlier. COLLAPSE_DUP_K_ROWS must be off so the ranked prefix covers
  # every K budget for the lo-tier substitution.
  echo "[pagedpq_stream_smoke] frozen-sim flags set: forcing canonical torch grid path"
  # The canonical-GPU assertion layer requires every native kernel ON; the
  # frozen-sim arms deliberately run all-torch, so drop the assertion too.
  export FRONTIER_CANONICAL_GPU=0
  export SELECTOR_PQ_JOINT_GROUPED_RISK_PREFIX=0
  export SELECTOR_PQ_JOINT_NATIVE_RISK_PREFIX=0
  export SELECTOR_PQ_JOINT_NATIVE_SCORE_GRID=0
  export SELECTOR_PQ_JOINT_NATIVE_POLICY=0
  export SELECTOR_PQ_JOINT_NATIVE_V_PREFIX=0
  export SELECTOR_PQ_JOINT_COMPACT_VPQ_RISK_PREFIX=0
  export SELECTOR_PQ_JOINT_COLLAPSE_DUP_K_ROWS=0
  export SELECTOR_PQ_JOINT_MERGE_RISK_POLICY=0
  export SELECTOR_PQ_JOINT_FUSED_MIXED_POLICY=0
  export SELECTOR_PQ_JOINT_STAGED_KV_PREFIX=0
  # Keep softmax/base in torch as well: native softmax/base over a
  # torch-built score grid is an untested combination, and the HELMET
  # frontier runs already validated the all-torch configuration.
  export SELECTOR_PQ_JOINT_NATIVE_SOFTMAX_BASE=0
  export SELECTOR_PQ_JOINT_NATIVE_VPQ_BASE=0
  # Bisection escape hatch: FROZEN_SIM_FLAG_SET="NAME=1 NAME=0" (space- or
  # comma-separated; use spaces inside sbatch --export, whose own separator
  # is the comma) re-sets individual flags AFTER the hard zeros. A plain env
  # default here would not work: the batched wrapper sources
  # frontier_canonical_env.sh (all native flags default 1) before this
  # block, so ${VAR:-0} would silently keep the canonical 1s and turn the
  # frozen-sim override into a no-op.
  if [ -n "${FROZEN_SIM_FLAG_SET:-}" ]; then
    IFS=', ' read -ra _fs_pairs <<< "${FROZEN_SIM_FLAG_SET}"
    for _fs in "${_fs_pairs[@]}"; do
      [ -n "${_fs}" ] || continue
      export "${_fs%%=*}=${_fs#*=}"
      echo "[pagedpq_stream_smoke] frozen-sim flag override: ${_fs}"
    done
  fi
fi
export TOKENIZERS_PARALLELISM=false

PROFILE_NATIVE_OPS_ARG=()
if [ "${PROFILE_NATIVE_OPS:-0}" = "1" ]; then
  PROFILE_NATIVE_OPS_ARG=(--profile_native_ops)
fi
if [ "${DISABLE_COST_STATS:-0}" = "1" ]; then
  echo "[ERROR] DISABLE_COST_STATS=1 is deprecated for active frontier runs; use accounting/profile artifacts instead." >&2
  exit 2
fi
DISABLE_NATIVE_DECODE_FUSED_ARG=()
if [ "${DISABLE_NATIVE_DECODE_FUSED:-1}" = "1" ]; then
  DISABLE_NATIVE_DECODE_FUSED_ARG=(--disable_native_decode_fused)
fi
ENABLE_NATIVE_DECODE_FUSED_ARG=()
if [ "${ENABLE_NATIVE_DECODE_FUSED:-0}" = "1" ]; then
  ENABLE_NATIVE_DECODE_FUSED_ARG=(--enable_native_decode_fused)
fi
NATIVE_DECODE_SCORELESS_FUSED_ARG=()
if [ "${NATIVE_DECODE_SCORELESS_FUSED:-0}" = "1" ]; then
  NATIVE_DECODE_SCORELESS_FUSED_ARG=(
    --native_decode_scoreless_fused
    --native_decode_scoreless_force_mode "${NATIVE_DECODE_SCORELESS_FORCE_MODE:-2}"
  )
fi
ALLOW_TF32_SELECTOR_ARG=()
if [ "${ALLOW_TF32_SELECTOR:-0}" = "1" ]; then
  ALLOW_TF32_SELECTOR_ARG=(--allow_tf32_selector)
fi
NATIVE_DECODE_TAIL_ARG=()
if [ "${NATIVE_DECODE_TAIL:-0}" = "1" ]; then
  NATIVE_DECODE_TAIL_ARG=(--native_decode_tail)
fi
DENSE_KV_OFFLOAD_ARG=()
if [ "${DENSE_KV_OFFLOAD:-0}" = "1" ]; then
  DENSE_KV_OFFLOAD_ARG=(
    --dense_kv_offload
    --dense_kv_block_tokens "${DENSE_KV_BLOCK_TOKENS:-8192}"
    --dense_kv_staging_buffers "${DENSE_KV_STAGING_BUFFERS:-2}"
    --dense_kv_query_block_tokens "${DENSE_KV_QUERY_BLOCK_TOKENS:-2048}"
  )
fi
GREEDY_LOGIT_TRACE_ARG=()
if [ -n "${GREEDY_LOGIT_TRACE_FILE:-}" ]; then
  GREEDY_LOGIT_TRACE_ARG=(--greedy_logit_trace_file "${GREEDY_LOGIT_TRACE_FILE}")
fi
PREFILL_TAIL_BLEND_ARG=()
if [ -n "${PREFILL_TAIL_BLEND:-}" ]; then
  PREFILL_TAIL_BLEND_ARG=(--prefill_tail_blend "${PREFILL_TAIL_BLEND}")
fi
DECODE_TAIL_BLEND_ARG=()
if [ -n "${DECODE_TAIL_BLEND:-}" ]; then
  DECODE_TAIL_BLEND_ARG=(--decode_tail_blend "${DECODE_TAIL_BLEND}")
fi
PREFILL_TAIL_SCORE_REUSE_ARG=()
if [ "${PREFILL_TAIL_SCORE_REUSE:-0}" = "1" ]; then
  PREFILL_TAIL_SCORE_REUSE_ARG=(--prefill_tail_score_reuse)
fi

echo "[pagedpq_stream_smoke] mode=${MODE} approx_prefill=${APPROX_PREFILL} task=${TASK_NAME} context=${CONTEXT_LEN} samples=${NUM_SAMPLES}"
echo "[pagedpq_stream_smoke] out=${OUT_DIR}"
echo "[pagedpq_stream_smoke] budget=${BUDGET:-4096} confidence=${ONLINE_CONFIDENCE_RULE:-joint_kv_stability} target=${TAIL_PROXY_MASS_MIN:-0.0} geom_min=${GEOMETRIC_MIN_BUDGET:-8192} geom_max=${GEOMETRIC_MAX_BUDGET:-65536} page=${PAGE_SIZE:-5632} chunk=${PREFILL_CHUNK_SIZE:-0}"
echo "[pagedpq_stream_smoke] exact_logit_backend=${FRONTIER_EXACT_LOGIT_BACKEND:-auto}"
echo "[pagedpq_stream_smoke] dense_kv_offload=${DENSE_KV_OFFLOAD:-0} kv_block=${DENSE_KV_BLOCK_TOKENS:-8192} staging=${DENSE_KV_STAGING_BUFFERS:-2} query_block=${DENSE_KV_QUERY_BLOCK_TOKENS:-2048}"

DATA_FILE="${DATA_FILE_OVERRIDE:-${DATA_DIR}/${TASK_NAME}/validation.jsonl}"
if [ -n "${DATA_FILE_OVERRIDE:-}" ]; then
  if [ ! -s "${DATA_FILE}" ]; then
    echo "[pagedpq_stream_smoke] DATA_FILE_OVERRIDE does not exist or is empty: ${DATA_FILE}" >&2
    exit 1
  fi
  echo "[pagedpq_stream_smoke] using data override ${DATA_FILE}"
elif [ "${REUSE_DATA:-0}" = "1" ] && [ -s "${DATA_FILE}" ]; then
  echo "[pagedpq_stream_smoke] reusing data ${DATA_FILE}"
else
  pushd benchmark/ruler >/dev/null
  python -u data/prepare.py \
    --save_dir "../../${DATA_DIR}" \
    --benchmark synthetic \
    --task "${TASK_NAME}" \
    --tokenizer_path "${MODEL_PATH}" \
    --tokenizer_type hf \
    --max_seq_length "${CONTEXT_LEN}" \
    --model_template_type "${MODEL_TEMPLATE_TYPE:-meta-chat}" \
    --num_samples "${NUM_SAMPLES}"
  popd >/dev/null
fi

"${HF_VENV_DIR}/bin/python" benchmark/ruler/pred/call_pagedpq_streaming.py \
  --model_name "${MODEL_PATH}" \
  --cache_dir "${CACHE_DIR:-.hf_cache}" \
  --data_file "${DATA_FILE}" \
  --output_file "${PRED_DIR}/${TASK_NAME}.jsonl" \
  --summary_file "${SUMMARY_DIR}/${TASK_NAME}.json" \
  --task "${TASK_NAME}" \
  --num_samples "${NUM_SAMPLES}" \
  --max_new_tokens "${MAX_NEW_TOKENS:-0}" \
  --mode "${MODE}" \
  --layers "${LAYERS:-all}" \
  --device "${DEVICE:-cuda}" \
  --local_files_only \
  --selector_mode "${SELECTOR_MODE:-fullscan}" \
  --selector_backend "${SELECTOR_BACKEND:-${SELECTOR_PAGED_PQ_BACKEND:-cuda_ext}}" \
  --budget "${BUDGET:-4096}" \
  --budget_by_head "${BUDGET_BY_HEAD:-}" \
  --tail_mode "${TAIL_MODE:-vpq_value}" \
  --online_confidence_rule "${ONLINE_CONFIDENCE_RULE:-joint_kv_stability}" \
  --tail_score_calibration "${TAIL_SCORE_CALIBRATION:-none}" \
  --tail_blend "${TAIL_BLEND:-1.0}" \
  --tail_probe_rel_l2_max "${TAIL_PROBE_REL_L2_MAX:-0.020}" \
  --tail_proxy_mass_min "${TAIL_PROXY_MASS_MIN:-0.0}" \
  --tail_proxy_mass_max "${TAIL_PROXY_MASS_MAX:-1.0}" \
  --tail_pq_corr_min "${TAIL_PQ_CORR_MIN:--1.0}" \
  --tail_pq_relrmse_max "${TAIL_PQ_RELRMSE_MAX:-inf}" \
  --ranked_confidence_cost_mode "${RANKED_CONFIDENCE_COST_MODE:-exact}" \
  --exact_logit_backend "${FRONTIER_EXACT_LOGIT_BACKEND:-auto}" \
  --geometric_min_budget "${GEOMETRIC_MIN_BUDGET:-8192}" \
  --geometric_max_budget "${GEOMETRIC_MAX_BUDGET:-65536}" \
  --geometric_growth "${GEOMETRIC_GROWTH:-1.5}" \
  --geometric_probe_scale "${GEOMETRIC_PROBE_SCALE:-1.5}" \
  --geometric_budget_granularity "${GEOMETRIC_BUDGET_GRANULARITY:-1024}" \
  --joint_kv_policy "${JOINT_KV_POLICY:-k_first_alternating}" \
  --joint_kv_k_budgets "${JOINT_KV_K_BUDGETS:-4096,8192,14336,32768}" \
  --joint_kv_v_budgets "${JOINT_KV_V_BUDGETS:-1024,2048,4096,6144,8192,12288,16384}" \
  --joint_kv_k_budget_fracs "${JOINT_KV_K_BUDGET_FRACS:-0.10,0.30,0.50,0.70,0.90,1.0}" \
  --joint_kv_v_budget_fracs "${JOINT_KV_V_BUDGET_FRACS:-0.05,0.10,0.20,0.40,0.60,0.80,1.0}" \
  --joint_kv_stability_threshold "${JOINT_KV_STABILITY_THRESHOLD:-0.002}" \
  --joint_kv_threshold_mode "${JOINT_KV_THRESHOLD_MODE:-budget_delta_frac}" \
  --joint_kv_threshold_reference_frac "${JOINT_KV_THRESHOLD_REFERENCE_FRAC:-0.2}" \
  --joint_kv_threshold_scale_shape "${JOINT_KV_THRESHOLD_SCALE_SHAPE:-sqrt}" \
  --joint_kv_threshold_min_scale "${JOINT_KV_THRESHOLD_MIN_SCALE:-0.0}" \
  --joint_kv_threshold_max_scale "${JOINT_KV_THRESHOLD_MAX_SCALE:-1.5}" \
  --joint_kv_start_strategy "${JOINT_KV_START_STRATEGY:-proxy_mass_m0p9}" \
  --logit_buffer_format "${LOGIT_BUFFER_FORMAT:-fp}" \
  ${JOINT_KV_DEESCALATE:+--joint_kv_deescalate} \
  ${JOINT_KV_PRECISION_TIERS:+--joint_kv_precision_tiers} \
  --selected_value_mode "${SELECTED_VALUE_MODE:-vpq_value}" \
  --selected_value_exact_rule "${SELECTED_VALUE_EXACT_RULE:-global_residual_risk}" \
  --selected_value_exact_top "${SELECTED_VALUE_EXACT_TOP:-0}" \
  --selected_value_exact_mass "${SELECTED_VALUE_EXACT_MASS:-0.98}" \
  --selected_value_min_exact_top "${SELECTED_VALUE_MIN_EXACT_TOP:-0}" \
  --selected_value_max_exact_top "${SELECTED_VALUE_MAX_EXACT_TOP:-0}" \
  --selected_value_exact_all_context_max "${SELECTED_VALUE_EXACT_ALL_CONTEXT_MAX:-0}" \
  --selected_value_exact_all_fraction_min "${SELECTED_VALUE_EXACT_ALL_FRACTION_MIN:-0.0}" \
  --value_code_stat_bytes "${VALUE_CODE_STAT_BYTES:-2}" \
  --page_size "${PAGE_SIZE:-5632}" \
  --prefill_chunk_size "${PREFILL_CHUNK_SIZE:-0}" \
  --prefill_selector_backend "${PREFILL_SELECTOR_BACKEND:-native}" \
  --prefill_selector_stride "${PREFILL_SELECTOR_STRIDE:-1}" \
  --prefill_selector_tile_size "${PREFILL_SELECTOR_TILE_SIZE:-0}" \
  --prefill_rank_buffer_limit_mb "${PREFILL_RANK_BUFFER_LIMIT_MB:-4096}" \
  --prefill_selector_page_block_size "${PREFILL_SELECTOR_PAGE_BLOCK_SIZE:-0}" \
  "${PREFILL_TAIL_SCORE_REUSE_ARG[@]}" \
  --prefill_attention_backend "${PREFILL_ATTENTION_BACKEND:-native}" \
  --subvecs "${SUBVECS:-4}" \
  --subbits "${SUBBITS:-8}" \
  --value_subvecs "${VALUE_SUBVECS:-1}" \
  --value_subbits "${VALUE_SUBBITS:-4}" \
  --value_pq_group_pages "${VALUE_PQ_GROUP_PAGES:-1}" \
  --kmeans_iters "${KMEANS_ITERS:-3}" \
  --index_build_backend "${INDEX_BUILD_BACKEND:-${PAGEDPQ_INDEX_BUILD_BACKEND:-numpy}}" \
  --nprobes "${NPROBES:-512}" \
  --static_prefix "${STATIC_PREFIX:-128}" \
  --static_suffix "${STATIC_SUFFIX:-128}" \
  --key_bytes 2 \
  --value_bytes 2 \
  "${PREFILL_TAIL_BLEND_ARG[@]}" \
  "${DECODE_TAIL_BLEND_ARG[@]}" \
  "${PROFILE_NATIVE_OPS_ARG[@]}" \
  "${DISABLE_NATIVE_DECODE_FUSED_ARG[@]}" \
  "${ENABLE_NATIVE_DECODE_FUSED_ARG[@]}" \
  "${NATIVE_DECODE_SCORELESS_FUSED_ARG[@]}" \
  "${ALLOW_TF32_SELECTOR_ARG[@]}" \
  "${NATIVE_DECODE_TAIL_ARG[@]}" \
  "${DENSE_KV_OFFLOAD_ARG[@]}" \
  "${GREEDY_LOGIT_TRACE_ARG[@]}"

if [ "${SKIP_RULER_EVAL:-0}" != "1" ]; then
  pushd benchmark/ruler >/dev/null
  python -u eval/evaluate.py \
    --data_dir "../../${PRED_DIR}" \
    --benchmark synthetic
  popd >/dev/null
else
  echo "[pagedpq_stream_smoke] SKIP_RULER_EVAL=1 (non-RULER data; score externally)"
fi

"${HF_VENV_DIR}/bin/python" - <<PY
import csv
import json
from pathlib import Path

summary_path = Path("${SUMMARY_DIR}/${TASK_NAME}.json")
score_path = Path("${PRED_DIR}/summary-${TASK_NAME}.csv")
if summary_path.exists() and score_path.exists():
    rows = {}
    with score_path.open(newline="") as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                rows[row[0]] = row[1]
    payload = json.loads(summary_path.read_text())
    if "Score" in rows:
        payload["score"] = float(rows["Score"])
    if "Nulls" in rows:
        payload["nulls"] = rows["Nulls"]
    summary_path.write_text(json.dumps(payload, indent=2) + "\n")
PY

echo "[pagedpq_stream_smoke] done ${OUT_DIR}"
