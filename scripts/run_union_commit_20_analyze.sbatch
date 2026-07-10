#!/usr/bin/env bash
#SBATCH --job-name=uc20-an
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16000m
#SBATCH --time=00:30:00
#SBATCH --account=zhengya0
#SBATCH --partition=standard

# Issue #20: runs after both union-commit shards (afterany dependency). Merges
# the big (q287,q283) + small (q262,q137) gqa_union_commit.csv shards and emits
# the validation report (aggregates, superset verdict, union sizes, gate-2 token
# reconciliation vs #11 epoch_trace, byte reconciliation vs epoch_replay oracle).
set -euo pipefail
WT=/gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention/worktrees/union-commit
cd "$WT"
module purge
module load python/3.10.4
export LD_LIBRARY_PATH="/sw/pkgs/arc/python/3.10.4/lib:${LD_LIBRARY_PATH:-}"

OUTROOT=/scratch/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention/union_commit_20
BIG="${OUTROOT}/big/gqa_union_commit.csv"
SMALL="${OUTROOT}/small/gqa_union_commit.csv"
REPORT="${OUTROOT}/union_commit_20_report.txt"

echo "[uc20-an] shards: $BIG ; $SMALL"
"$WT/.venv/bin/python" "$WT/scripts/analyze_union_commit_20.py" \
  --csv "$BIG" "$SMALL" \
  --epoch_trace_dir "$WT/benchmark/selector_eval/golden_vectors/epoch_trace_20260710" \
  --out "$REPORT"
echo "[uc20-an] report -> $REPORT"
