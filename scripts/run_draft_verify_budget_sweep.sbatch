#!/usr/bin/env bash
#SBATCH --job-name=dv-budget-swp
#SBATCH --account=zhengya0
#SBATCH --partition=gpu_mig40,spgpu,gpu-rtx6000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120000m
#SBATCH --time=04:30:00
#SBATCH --output=logs/draft_verify_budget_sweep_%j.log
set -euo pipefail

# Issue #21 bundle C: drafter budget-sensitivity sweep (qa_1 32k n=4, 64 new
# tokens), run from the PINNED draft-budget worktree (draft-gpu stays byte-
# frozen for its in-flight jobs).
#
# GRID NOTE (dedup pass, revised on GPU counter evidence): the GPU
# lineage's de facto start mass is 0.5 (the GPU-side suffix parse always
# returns the default). On the L16 trace, mass-0.5 counts sit at/below the
# GPU floor k0=4096 — but the 32k matrix counters show qa_1 mass-0.5
# starts are NOT floor-clamped (start1 mean committed ~13.1k rows = rung
# mix well above floor), so floor collapse is data-dependent and the
# down-mass axis IS live on qa_1: m0p35/m0p2 arms are kept at 32k. The
# grid: start0 (one-rung-cheaper pin), legacy start1 anchor, down-mass
# probes, the mass-0.9 upward probe at bump 0 (= the CPU-spec operating
# point, for the divergence reconciliation), and v_ratio {0.125, 0.5}
# probes at the off-floor arms. Mass overrides use the DRAFT-ONLY env
# SELECTOR_PQ_JOINT_DRAFT_PROXY_MASS (cannot engage in off/frozen runs).
# Per-arm draft_histo.json records the realized (layer,ki,vi) mix, which
# settles the floor-clamp question empirically per arm.
#
# All draft arms are pin-rung (walk skipped) with the start1 shape: exact-K
# logits on the committed set + exact/corrected V (full frozen finalize).
# Knobs, draft-branch only:
#   SELECTOR_PQ_JOINT_DRAFT_MODE       start1|start0|start-1 (bump +1/0/-1)
#   SELECTOR_PQ_JOINT_DRAFT_PROXY_MASS coverage mass for the start rung
#   SELECTOR_PQ_JOINT_DRAFT_V_RATIO    v_target coupling (frozen rule: 0.25)
#   SELECTOR_PQ_JOINT_DRAFT_HISTO_FILE realized (layer,ki,vi) histogram, for
#                                      the ladder-floor-clamp analysis
#                                      (k0=4096 at 32k = 12.5% of ctx; flag
#                                      floor-clamped arms — the 1M ladder
#                                      floor is 0.4% so 32k understates room)
# Passes: dense reference (dumps trace; max_new=64 differs from the matrix
# bundle so its dense trace is NOT reusable) then per draft arm free-run +
# teacher-forced-on-dense, all with logit traces.

REPO=/gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention/worktrees/draft-budget
DGPU=/gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention/worktrees/draft-gpu
cd "${REPO}"

: "${BUNDLE:?set BUNDLE=32k or 128k}"
case "${BUNDLE}" in
  32k)
    TASK_NAME=qa_1
    CONTEXT_LEN=32768
    NUM_SAMPLES=4
    DATA_FILE="${DGPU}/benchmark_suite_result/ktcache_ab_20260709/qa1_32k_n4_ktON/data/qa_1/validation.jsonl"
    RUN_PREFIX=qa1_32k_n4
    OUTPUT_ROOT="${OUTPUT_ROOT:-${DGPU}/benchmark_suite_result/draft_verify_gpu_20260712_matrix/budget_sweep_32k}"
    ;;
  128k)
    # mig40 excluded on the submit line (mk3@128k dense prefill OOMs 40GB).
    TASK_NAME=niah_multikey_3
    CONTEXT_LEN=131072
    NUM_SAMPLES=2
    DATA_FILE="${DGPU}/benchmark_suite_result/frozen_sim_20260707/runs/frozensim_mk3_128k_n16/data/niah_multikey_3/validation.jsonl"
    RUN_PREFIX=mk3_128k_n2
    OUTPUT_ROOT="${OUTPUT_ROOT:-${DGPU}/benchmark_suite_result/draft_verify_gpu_20260712_matrix/budget_sweep_128k}"
    ;;
  *)
    echo "unknown BUNDLE=${BUNDLE}" >&2; exit 2 ;;
esac
MAX_NEW_TOKENS=64
mkdir -p logs

[ -s "${DATA_FILE}" ] || { echo "missing data ${DATA_FILE}" >&2; exit 1; }
echo "[dv-budget-swp] head=$(git rev-parse --short HEAD) bundle=${BUNDLE} node=$(hostname)"
nvidia-smi -L

run_arm() {
  local arm_name=$1   # dense | grid point label
  local draft_mode=$2 # off (dense) | start1 | start0 | start-1
  local mass=$3       # "" = legacy (de facto 0.5) | explicit mass
  local v_ratio=$4    # v_target coupling
  local tf_ref=$5     # "" free-run | dense
  local run_name="${RUN_PREFIX}_${arm_name}"
  local forced_trace=""
  if [ -n "${tf_ref}" ]; then
    run_name="${RUN_PREFIX}_${arm_name}_tf_${tf_ref}"
    forced_trace="${OUTPUT_ROOT}/${RUN_PREFIX}_${tf_ref}/logit_trace.pt"
    [ -s "${forced_trace}" ] || { echo "missing ref trace ${forced_trace}" >&2; exit 1; }
  fi
  local arm_mode=pagedpq_batched
  if [ "${arm_name}" = "dense" ]; then arm_mode=dense_batched; fi
  local t0=${SECONDS}
  echo "[dv-budget-swp] arm start name=${arm_name} mode=${draft_mode} mass=${mass:-legacy} vr=${v_ratio} run=${run_name} tf_ref=${tf_ref:-none}"
  env \
    MODE="${arm_mode}" \
    GREEDY_LOGIT_TRACE_FILE="${OUTPUT_ROOT}/${run_name}/logit_trace.pt" \
    FORCED_TOKEN_TRACE_FILE="${forced_trace}" \
    SELECTOR_PQ_JOINT_DRAFT_PROXY_MASS="${mass}" \
    SELECTOR_PQ_JOINT_DRAFT_V_RATIO="${v_ratio}" \
    SELECTOR_PQ_JOINT_DRAFT_HISTO_FILE="${OUTPUT_ROOT}/${run_name}/draft_histo.json" \
    TASK_NAME="${TASK_NAME}" \
    CONTEXT_LEN="${CONTEXT_LEN}" \
    NUM_SAMPLES="${NUM_SAMPLES}" \
    MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
    RUN_NAME="${run_name}" \
    JOINT_KV_STABILITY_THRESHOLD=0.004 \
    LOGIT_BUFFER_FORMAT=e4m3 \
    JOINT_KV_PRECISION_TIERS=1 \
    FRONTIER_DENSE_KEY_T_CACHE=0 \
    OUTPUT_ROOT="${OUTPUT_ROOT}" \
    DATA_FILE_OVERRIDE="${DATA_FILE}" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PREFILL_CHUNK_SIZE=8192 \
    FRONTIER_EMPTY_CACHE_AFTER_PREFILL_CHUNK=1 \
    SELECTOR_PQ_JOINT_FROZENSIM_FUSED_VPREFIX=1 \
    SELECTOR_PQ_JOINT_FROZENSIM_FUSED_RISK_SORT=1 \
    SELECTOR_PQ_JOINT_FROZENSIM_FUSED_VPREFIX_TOKPAR=0 \
    SELECTOR_PQ_JOINT_MEMORY_TRACE=0 \
    SELECTOR_PQ_JOINT_TIME_TRACE=0 \
    SELECTOR_PQ_JOINT_DRAFT_MODE="${draft_mode}" \
    SKIP_RULER_EVAL=1 \
    bash scripts/run_frontier_ruler_batched_one.sh
  local wall=$(( SECONDS - t0 ))
  echo "DV-BUDGET-ARM bundle=${BUNDLE} name=${arm_name} run=${run_name} wall=${wall}s"
}

# name           mode     mass   v_ratio
if [ "${BUNDLE}" = "32k" ]; then
  # 9 draft arms x 2 passes + dense = 19 passes (~3.2h at ~10 min/pass).
  GRID="\
start0:start0::0.25 \
anchor_b1:start1::0.25 \
m0p35_b0:start0:0.35:0.25 \
m0p2_b0:start0:0.2:0.25 \
m0p9_b0:start0:0.9:0.25 \
s1_vr0p125:start1::0.125 \
s1_vr0p5:start1::0.5 \
m0p9_vr0p125:start0:0.9:0.125 \
m0p9_vr0p5:start0:0.9:0.5"
else
  # 128k passes cost ~40 min each; drop the two m0p9 v_ratio probes to fit
  # the 12h limit (7 draft arms x 2 + dense = 15 passes).
  GRID="\
start0:start0::0.25 \
anchor_b1:start1::0.25 \
m0p35_b0:start0:0.35:0.25 \
m0p2_b0:start0:0.2:0.25 \
m0p9_b0:start0:0.9:0.25 \
s1_vr0p125:start1::0.125 \
s1_vr0p5:start1::0.5"
fi

run_arm dense off "" 0.25 ""
for spec in ${GRID}; do
  IFS=':' read -r nm md ms vr <<< "${spec}"
  run_arm "${nm}" "${md}" "${ms}" "${vr}" ""
done
for spec in ${GRID}; do
  IFS=':' read -r nm md ms vr <<< "${spec}"
  run_arm "${nm}" "${md}" "${ms}" "${vr}" dense
done

echo "DV-BUDGET-SWEEP: COMPLETED bundle=${BUNDLE}"
