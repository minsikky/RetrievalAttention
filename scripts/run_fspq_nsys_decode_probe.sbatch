#!/usr/bin/env bash
#SBATCH --job-name=fspq-nsys-decode
#SBATCH --account=zhengya0
#SBATCH --partition=gpu_mig40,spgpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120000m
#SBATCH --time=02:00:00
#SBATCH --output=logs/fspq_nsys_decode_%j.log
set -euo pipefail

# Kernel-level attribution of the fused 128k decode step (17.5 s/token):
# is the remaining time GPU-bound math or launch/sync overhead?
# Wraps the standard fused rate config (8 new tokens) in Nsight Systems,
# then slices the decode window (the trailing attention_seconds of the
# timeline, taken from the run's own time trace) out of the kernel table:
# GPU busy fraction, top kernels, launch counts, memcpy traffic.
# Round 4: runs from the MAIN tree (tokpar-enabled ext + b41e326 tokpar-default
# python). The original round-3 probe ran from the pinned ab-cand worktree
# (pre-tokpar 4b52294) — job 53321696 profiled the OLD path by mistake.
#
# Submit with: sbatch scripts/run_fspq_nsys_decode_probe.sbatch

REPO=/gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention
WT_CAND="${REPO}"
cd "${REPO}"

DATA_128K="${REPO}/benchmark_suite_result/frozen_sim_20260707/runs/frozensim_mk3_128k_n16/data/niah_multikey_3/validation.jsonl"
OUTPUT_ROOT="${REPO}/benchmark_suite_result/fspq_rate_probes"
NSYS_DIR="${OUTPUT_ROOT}/nsys"
mkdir -p "${NSYS_DIR}"
[ -s "${DATA_128K}" ] || { echo "[nsys-probe] missing data ${DATA_128K}" >&2; exit 1; }

module load nsight-systems/2025.3.1
echo "[nsys-probe] cand=$(git -C "${WT_CAND}" rev-parse --short HEAD) gpu=$(nvidia-smi -L | head -1) nsys=$(nsys --version | head -1)"

REP="${NSYS_DIR}/decode_128k_${SLURM_JOB_ID}"
nsys profile \
  -o "${REP}" \
  --trace=cuda \
  --sample=none \
  --cpuctxsw=none \
  --force-overwrite=true \
  env \
    TASK_NAME=niah_multikey_3 \
    CONTEXT_LEN=131072 \
    NUM_SAMPLES=1 \
    MAX_NEW_TOKENS=8 \
    RUN_NAME="nsysprobe_fused_8tok" \
    JOINT_KV_STABILITY_THRESHOLD=0.004 \
    LOGIT_BUFFER_FORMAT=e4m3 \
    JOINT_KV_PRECISION_TIERS=1 \
    FRONTIER_DENSE_KEY_T_CACHE=0 \
    OUTPUT_ROOT="${OUTPUT_ROOT}" \
    DATA_FILE_OVERRIDE="${DATA_128K}" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PREFILL_CHUNK_SIZE=8192 \
    FRONTIER_EMPTY_CACHE_AFTER_PREFILL_CHUNK=1 \
    SELECTOR_PQ_JOINT_FROZENSIM_FUSED_VPREFIX=1 \
    SELECTOR_PQ_JOINT_FROZENSIM_FUSED_RISK_SORT=1 \
    SELECTOR_PQ_JOINT_FROZENSIM_FUSED_VPREFIX_TOKPAR=1 \
    SELECTOR_PQ_JOINT_MEMORY_TRACE=0 \
    SELECTOR_PQ_JOINT_TIME_TRACE=1 \
    SKIP_RULER_EVAL=1 \
    bash "${WT_CAND}/scripts/run_frontier_ruler_batched_one.sh"

echo "[nsys-probe] exporting sqlite"
nsys export --type sqlite --force-overwrite=true -o "${REP}.sqlite" "${REP}.nsys-rep"

echo "[nsys-probe] === decode-window kernel attribution ==="
module load python/3.10.4
export LD_LIBRARY_PATH="/sw/pkgs/arc/python/3.10.4/lib:${LD_LIBRARY_PATH:-}"
"${REPO}/.venv/bin/python" - "${REP}.sqlite" "${REPO}/logs/fspq_nsys_decode_${SLURM_JOB_ID}.log" <<'PY'
import json
import re
import sqlite3
import sys

db_path, log_path = sys.argv[1], sys.argv[2]

# Decode duration from the run's own time trace (attention encompasses select).
attn = None
for line in open(log_path, errors="replace"):
    m = re.search(r"\[selector-pq-joint-time\] (\{.*\})", line)
    if m:
        attn = json.loads(m.group(1))
if attn is None:
    sys.exit("no [selector-pq-joint-time] line found in log")
decode_s = attn["attention_seconds"] + attn.get("lm_head_seconds", 0.0)
print(f"time-trace: attention={attn['attention_seconds']:.1f}s prefill={attn['prefill_seconds']:.1f}s "
      f"select={attn['select_seconds']:.1f}s sync_wait={attn['sync_wait_seconds']:.1f}s")

con = sqlite3.connect(db_path)
tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}

def rows(table, name_col):
    if table not in tables:
        return []
    has_names = "StringIds" in tables
    if has_names:
        q = (f"SELECT k.start, k.end, s.value FROM {table} k "
             f"LEFT JOIN StringIds s ON k.{name_col} = s.id")
    else:
        q = f"SELECT start, end, '{table}' FROM {table}"
    try:
        return list(con.execute(q))
    except sqlite3.OperationalError as e:
        print(f"({table}: {e})")
        return []

kern = rows("CUPTI_ACTIVITY_KIND_KERNEL", "demangledName") or rows("CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL", "demangledName")
if not kern:
    sys.exit(f"no kernel table found; tables={sorted(tables)}")

t_end = max(e for _, e, _ in kern)
w0 = t_end - int((decode_s + 3.0) * 1e9)  # decode = trailing window + margin
win = [(s, e, n) for s, e, n in kern if e > w0]
span = (t_end - w0) / 1e9

# GPU busy = union of kernel intervals in the window
ivals = sorted((max(s, w0), e) for s, e, _ in win)
busy, cur_s, cur_e = 0, None, None
for s, e in ivals:
    if cur_e is None or s > cur_e:
        if cur_e is not None:
            busy += cur_e - cur_s
        cur_s, cur_e = s, e
    else:
        cur_e = max(cur_e, e)
if cur_e is not None:
    busy += cur_e - cur_s
busy_s = busy / 1e9

print(f"\ndecode window: {span:.1f}s | kernel launches: {len(win)} ({len(win)/max(1e-9,span):.0f}/s)")
print(f"GPU busy: {busy_s:.1f}s ({100*busy_s/span:.1f}%)  |  GPU idle: {span-busy_s:.1f}s ({100*(1-busy_s/span):.1f}%)")

agg = {}
for s, e, n in win:
    name = (n or "?").split("(")[0][:100]
    t, c = agg.get(name, (0, 0))
    agg[name] = (t + (e - max(s, w0)), c + 1)
print("\ntop kernels in decode window (by GPU time):")
for name, (t, c) in sorted(agg.items(), key=lambda kv: -kv[1][0])[:20]:
    print(f"  {t/1e9:8.2f}s  x{c:<7d} {name}")

for table, label in (("CUPTI_ACTIVITY_KIND_MEMCPY", "memcpy"), ("CUPTI_ACTIVITY_KIND_MEMSET", "memset")):
    if table not in tables:
        continue
    try:
        tt, cnt, byt = con.execute(
            f"SELECT COALESCE(SUM(end-start),0), COUNT(*), COALESCE(SUM(bytes),0) FROM {table} WHERE end > ?",
            (w0,),
        ).fetchone()
        print(f"\n{label} in window: {tt/1e9:.2f}s x{cnt} {byt/2**30:.2f} GiB")
    except sqlite3.OperationalError as e:
        print(f"({table}: {e})")
print("\nNSYS-DECODE-PROBE: ANALYZED")
PY

echo "NSYS-DECODE-PROBE: COMPLETED (rep: ${REP}.nsys-rep)"
