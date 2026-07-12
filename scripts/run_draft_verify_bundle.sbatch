#!/usr/bin/env bash
#SBATCH --job-name=draft-verify-b
#SBATCH --account=zhengya0
#SBATCH --partition=gpu_mig40,spgpu,gpu-rtx6000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120000m
#SBATCH --time=12:00:00
#SBATCH --output=logs/draft_verify_bundle_%j.log
set -euo pipefail

# Draft-then-verify token-acceptance BUNDLE: all arms of one context length run
# sequentially inside a single allocation, so every arm shares the exact same
# GPU by construction (card-class consistency binds only WITHIN a comparison).
# Pattern cribbed from scripts/run_fspq_tokpar_rate_probe.sbatch's run_arm loop.
#
# Submit with BUNDLE=32k or BUNDLE=128k (sbatch --export). Partition is set on
# the submit line: 32k = gpu_mig40,spgpu,gpu-rtx6000; 128k = spgpu,gpu-rtx6000
# (mig40 excluded: mk3@128k dense-prefill SDPA OOMs the 39.25GB slice — suite
# jobs 53253685/53253684 died there even with expandable_segments + chunk 8192).

REPO=/gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention/worktrees/draft-gpu
cd "${REPO}"

: "${BUNDLE:?set BUNDLE=32k or 128k}"
# Round 3 (issue #21, RTL pairwise-matrix request): all arms sequential on one
# card so the whole agreement matrix is internally consistent. The `dense` arm
# is the vanilla model (MODE=dense_batched — the same dense reference path the
# hard-RULER/HELMET dense runs use, cf. scripts/run_hard_ruler_dense_one.sh;
# no PQ patch is applied there, so the SELECTOR_PQ_JOINT_* env is inert). All
# other arms run pagedpq_batched with SELECTOR_PQ_JOINT_DRAFT_MODE per arm.
# Fresh output root; the 20260711/20260712 round-1/round-2 dirs stay untouched
# as the interim record.
OUTPUT_ROOT="${OUTPUT_ROOT:-benchmark_suite_result/draft_verify_gpu_20260712_matrix}"
mkdir -p logs

echo "[draft-verify-bundle] head=$(git rev-parse --short HEAD) bundle=${BUNDLE} node=$(hostname)"
nvidia-smi -L

case "${BUNDLE}" in
  32k)
    TASK_NAME=qa_1
    CONTEXT_LEN=32768
    NUM_SAMPLES=4
    MAX_NEW_TOKENS=0
    DATA_FILE="${REPO}/benchmark_suite_result/ktcache_ab_20260709/qa1_32k_n4_ktON/data/qa_1/validation.jsonl"
    RUN_PREFIX=qa1_32k_n4
    # Free runs (dense first: its free-run IS the dense reference and its
    # trace feeds every tf_dense pass), then teacher-forced passes on the
    # dense stream for every sparse arm, then (32k only, cheap) teacher-forced
    # passes on the frozen stream for the draft arms.
    ARMS="${ARMS:-dense off start1 pq_only middle v_exact \
      off:dense start1:dense pq_only:dense middle:dense v_exact:dense \
      start1:off pq_only:off middle:off v_exact:off}"
    ;;
  128k)
    TASK_NAME=niah_multikey_3
    CONTEXT_LEN=131072
    NUM_SAMPLES=2
    MAX_NEW_TOKENS=64
    DATA_FILE="${REPO}/benchmark_suite_result/frozen_sim_20260707/runs/frozensim_mk3_128k_n16/data/niah_multikey_3/validation.jsonl"
    RUN_PREFIX=mk3_128k_n2
    # As 32k but without the tf_off passes (128k arms are ~45 min each; the
    # tf_dense set already discharges the gray-zone rule for every pair).
    ARMS="${ARMS:-dense off start1 pq_only middle v_exact \
      off:dense start1:dense pq_only:dense middle:dense v_exact:dense}"
    ;;
  *)
    echo "unknown BUNDLE=${BUNDLE}; expected 32k or 128k" >&2
    exit 2
    ;;
esac

[ -s "${DATA_FILE}" ] || { echo "missing data ${DATA_FILE}" >&2; exit 1; }

run_arm() {
  # Arm token syntax: "<mode>" = free-running pass; "<mode>:<ref>" =
  # teacher-forced pass consuming <ref>'s free-run logit trace (the decode
  # input token is overridden with the reference stream token each step; the
  # arm's own per-position logits are what the trace records). Every pass
  # dumps the full per-position logit trace (torch.save; converted offline to
  # the top-32 + logsumexp npz per the #21 addendum).
  local arm_spec=$1
  local draft_mode="${arm_spec%%:*}"
  local tf_ref=""
  case "${arm_spec}" in *:*) tf_ref="${arm_spec#*:}" ;; esac
  local run_name="${RUN_PREFIX}_${draft_mode}"
  local forced_trace=""
  if [ -n "${tf_ref}" ]; then
    run_name="${RUN_PREFIX}_${draft_mode}_tf_${tf_ref}"
    forced_trace="${REPO}/${OUTPUT_ROOT}/${RUN_PREFIX}_${tf_ref}/logit_trace.pt"
    if [ ! -s "${forced_trace}" ]; then
      echo "missing reference trace ${forced_trace} for arm ${arm_spec}" >&2
      exit 1
    fi
  fi
  local t0=${SECONDS}
  # dense arm: vanilla model, no PQ patch; every other arm: pagedpq_batched
  # with the per-arm draft mode.
  local arm_mode=pagedpq_batched
  local arm_draft="${draft_mode}"
  if [ "${draft_mode}" = "dense" ]; then
    arm_mode=dense_batched
    arm_draft=off
  fi
  echo "[draft-verify-bundle] arm start spec=${arm_spec} run=${run_name} exec_mode=${arm_mode} tf_ref=${tf_ref:-none}"
  env \
    MODE="${arm_mode}" \
    GREEDY_LOGIT_TRACE_FILE="${REPO}/${OUTPUT_ROOT}/${run_name}/logit_trace.pt" \
    FORCED_TOKEN_TRACE_FILE="${forced_trace}" \
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
    SELECTOR_PQ_JOINT_DRAFT_MODE="${arm_draft}" \
    SKIP_RULER_EVAL=1 \
    bash scripts/run_frontier_ruler_batched_one.sh
  local wall=$(( SECONDS - t0 ))
  echo "DRAFT-VERIFY-ARM bundle=${BUNDLE} spec=${arm_spec} run=${run_name} wall=${wall}s"
}

for arm in ${ARMS}; do
  run_arm "${arm}"
done

echo "DRAFT-VERIFY-BUNDLE: COMPLETED bundle=${BUNDLE}"
