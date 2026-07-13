#!/usr/bin/env bash
#SBATCH --job-name=finalist-cert
#SBATCH --account=zhengya0
#SBATCH --partition=gpu_mig40,spgpu,gpu-rtx6000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120000m
#SBATCH --time=06:00:00
#SBATCH --output=logs/finalist_cert_%j.log
set -euo pipefail

# Drafter finalist certification (issue #21/#22 down-select follow-through).
# TF-ONLY passes reusing the acceptance campaign's dense traces (same n=8
# data, same references). Finalists per the unified Pareto:
#   FINALIST=pin   : floor-rung one-shot pin — SELECTOR_PQ_JOINT_DRAFT_MODE=
#                    start-1 + SELECTOR_PQ_JOINT_DRAFT_PROXY_MASS=0.5 (the
#                    S2/S3-validated draft-only override; OP pinned by env,
#                    no parser dependence). Byte-identical to the measured
#                    contract m0p5_bm1 arm by construction (same env); floor-
#                    equivalent to floor-OP start0 wherever the floor rung
#                    binds (100% of head-steps at 128k; 98.5% at 32k, the
#                    remaining 1.5% pin one rung lower). Audited per-pass via
#                    the realized-rung histogram dump.
#   FINALIST=tau01 : relaxed-walk off arm at tau=0.1 (32k supplement; the
#                    128k campaign's off_t0p1 arm covers 128k).
# Every pass echoes its realized OP env; completeness marker is the logit
# trace (per-pass skip); missing dense trace is a HARD FAIL.

REPO=/gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention/worktrees/op-fix
cd "${REPO}"
: "${TASKS:?set TASKS}"
: "${FINALIST:?set FINALIST=pin or tau01}"
CTX="${CTX:-32k}"
if [ "${CTX}" = "32k" ]; then CONTEXT_LEN=32768; CTXTAG=32k; else CONTEXT_LEN=131072; CTXTAG=128k; fi
NUM_SAMPLES=8
MAX_NEW_TOKENS=64
if [ "${CTXTAG}" = "128k" ]; then CAMPAIGN_TOKPAR="${CAMPAIGN_TOKPAR:-1}"; else CAMPAIGN_TOKPAR="${CAMPAIGN_TOKPAR:-0}"; fi
CAMPAIGN_ROOT="$(readlink -f benchmark_suite_result)/acceptance_campaign_20260712"
DATA_ROOT="${CAMPAIGN_ROOT}/data_${CTXTAG}"
OUTPUT_ROOT="${CAMPAIGN_ROOT}/runs_${CTXTAG}"
mkdir -p logs

case "${FINALIST}" in
  pin)   ARM_NAME=finalist_pin_floor; DRAFT_MODE=start-1; DRAFT_MASS=0.5; TAU=0.004 ;;
  tau01) ARM_NAME=off_t0p1;           DRAFT_MODE=off;     DRAFT_MASS="";  TAU=0.1 ;;
  *) echo "unknown FINALIST=${FINALIST}" >&2; exit 2 ;;
esac

echo "[finalist-cert] head=$(git rev-parse --short HEAD) finalist=${FINALIST} ctx=${CTXTAG} tokpar=${CAMPAIGN_TOKPAR} tasks='${TASKS}' node=$(hostname)"
nvidia-smi -L

for task in ${TASKS}; do
  prefix="${task}_${CTXTAG}_n${NUM_SAMPLES}"
  run_name="${prefix}_${ARM_NAME}_tf_dense"
  if [ -s "${OUTPUT_ROOT}/${run_name}/logit_trace.pt" ]; then
    echo "[finalist-cert] SKIP ${run_name} (complete)"
    continue
  fi
  forced_trace="${OUTPUT_ROOT}/${prefix}_dense/logit_trace.pt"
  [ -s "${forced_trace}" ] || { echo "HARD FAIL: missing campaign dense trace ${forced_trace}" >&2; exit 1; }
  data_file="${DATA_ROOT}/${task}/validation.jsonl"
  [ -s "${data_file}" ] || { echo "HARD FAIL: missing staged data ${data_file}" >&2; exit 1; }
  t0=${SECONDS}
  echo "[finalist-cert] pass start task=${task} run=${run_name} REALIZED-OP: draft_mode=${DRAFT_MODE} proxy_mass_override=${DRAFT_MASS:-none(parser)} tau=${TAU} v_ratio=0.25 tokpar=${CAMPAIGN_TOKPAR}"
  env \
    MODE=pagedpq_batched \
    GREEDY_LOGIT_TRACE_FILE="${OUTPUT_ROOT}/${run_name}/logit_trace.pt" \
    FORCED_TOKEN_TRACE_FILE="${forced_trace}" \
    SELECTOR_PQ_JOINT_DRAFT_PROXY_MASS="${DRAFT_MASS}" \
    SELECTOR_PQ_JOINT_DRAFT_V_RATIO=0.25 \
    SELECTOR_PQ_JOINT_DRAFT_HISTO_FILE="${OUTPUT_ROOT}/${run_name}/draft_histo.json" \
    TASK_NAME="${task}" \
    CONTEXT_LEN="${CONTEXT_LEN}" \
    NUM_SAMPLES="${NUM_SAMPLES}" \
    MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
    RUN_NAME="${run_name}" \
    JOINT_KV_STABILITY_THRESHOLD="${TAU}" \
    LOGIT_BUFFER_FORMAT=e4m3 \
    JOINT_KV_PRECISION_TIERS=1 \
    FRONTIER_DENSE_KEY_T_CACHE=0 \
    OUTPUT_ROOT="${OUTPUT_ROOT}" \
    DATA_FILE_OVERRIDE="${data_file}" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PREFILL_CHUNK_SIZE=8192 \
    FRONTIER_EMPTY_CACHE_AFTER_PREFILL_CHUNK=1 \
    SELECTOR_PQ_JOINT_FROZENSIM_FUSED_VPREFIX=1 \
    SELECTOR_PQ_JOINT_FROZENSIM_FUSED_RISK_SORT=1 \
    SELECTOR_PQ_JOINT_FROZENSIM_FUSED_VPREFIX_TOKPAR="${CAMPAIGN_TOKPAR}" \
    SELECTOR_PQ_JOINT_MEMORY_TRACE=0 \
    SELECTOR_PQ_JOINT_TIME_TRACE=0 \
    SELECTOR_PQ_JOINT_DRAFT_MODE="${DRAFT_MODE}" \
    SKIP_RULER_EVAL=1 \
    bash scripts/run_frontier_ruler_batched_one.sh
  echo "ACC-CAMPAIGN-PASS ctx=${CTXTAG} task=${task} arm=${ARM_NAME} tokpar=${CAMPAIGN_TOKPAR} run=${run_name} wall=$(( SECONDS - t0 ))s"
done

echo "FINALIST-CERT: COMPLETED finalist=${FINALIST} ctx=${CTXTAG} tasks='${TASKS}'"
