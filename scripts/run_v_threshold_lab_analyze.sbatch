#!/usr/bin/env bash
#SBATCH --job-name=vthr-analyze
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000m
#SBATCH --time=00:30:00
#SBATCH --account=zhengya0
#SBATCH --partition=standard

# Reduce the two absolute-V-threshold sweep shards into the issue #12 report
# JSON (matched-bytes / matched-quality / count-variance / sensitivity /
# per-context+head breakdowns). Runs afterok the two sweep jobs.
set -euo pipefail
cd /gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention
module purge
module load python/3.10.4
export LD_LIBRARY_PATH="/sw/pkgs/arc/python/3.10.4/lib:${LD_LIBRARY_PATH:-}"
ROOT="attention_efficiency_result/v_threshold_lab_20260711"
.venv/bin/python scripts/analyze_v_threshold_lab.py \
  --lab_dirs "${ROOT}/sweep_A_0_199/lab" "${ROOT}/sweep_B_199_288/lab" \
  --out_json "${ROOT}/report_12_full.json"
echo "[analyze] wrote ${ROOT}/report_12_full.json"
