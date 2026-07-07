#!/bin/bash
# Launch the frozen-sim arm (deesc + e4m3 + precision tiers, tau 0.004)
# across the Phase A task set at 128k, n=16 — the accuracy-of-the-frozen-
# algorithm reruns. Run ONLY after the frozen-sim smokes
# (53032068 deesc+e4m3, 53032533 +tiers) validate.
#
# Respects the 6-concurrent-job cap by chaining every job after the
# previous one with --dependency=afterany (they contend for the same
# account GPU pool anyway, so serialization costs little wall time).
#
# Usage: bash scripts/launch_frozen_sim_phase_a.sh [first_dependency_jobid]
set -euo pipefail
cd "$(dirname "$0")/.."

DEP="${1:-}"
COMMON_ENV=(
  CONTEXT_LEN=131072
  NUM_SAMPLES=16
  JOINT_KV_STABILITY_THRESHOLD=0.004
  JOINT_KV_DEESCALATE=1
  LOGIT_BUFFER_FORMAT=e4m3
  JOINT_KV_PRECISION_TIERS=1
  OUTPUT_ROOT=benchmark_suite_result/frozen_sim_20260707/runs
)
SBATCH_ARGS=(--account=zhengya0 --partition=gpu-rtx6000,spgpu --time=08:00:00 --export=ALL)

submit() {
  local task="$1" ctx="$2" run_name="$3"
  local dep_args=()
  [ -n "$DEP" ] && dep_args=(--dependency="afterany:${DEP}")
  local jid
  jid=$(env "${COMMON_ENV[@]}" TASK_NAME="$task" CONTEXT_LEN="$ctx" \
      RUN_NAME="$run_name" \
      sbatch "${SBATCH_ARGS[@]}" "${dep_args[@]}" \
      --job-name="frozensim-${task}" \
      --output="logs/frozensim_${task}_%j.log" \
      --parsable scripts/run_frontier_ruler_batched_one.sh)
  echo "submitted ${task}@${ctx} -> ${jid} (after: ${DEP:-none})"
  DEP="$jid"
}

submit fwe            131072 frozensim_fwe_128k_n16
submit niah_multikey_3 131072 frozensim_mk3_128k_n16
submit qa_1           131072 frozensim_qa1_128k_n16
submit qa_2           131072 frozensim_qa2_128k_n16
submit cwe            65536  frozensim_cwe_64k_n16

echo "chain tail: ${DEP}"
echo "NOTE: kilt_nq frozen-sim arm goes through scripts/run_helmet_frontier_one.sh"
echo "with the same JOINT_KV_DEESCALATE/LOGIT_BUFFER_FORMAT/JOINT_KV_PRECISION_TIERS"
echo "env once the RULER chain drains (submit separately to stay under the cap)."
