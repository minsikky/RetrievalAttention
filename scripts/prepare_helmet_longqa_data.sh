#!/usr/bin/env bash
#SBATCH --job-name=helmet-longqa
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16000m
#SBATCH --time=02:00:00
#SBATCH --account=zhengya98
#SBATCH --partition=standard

set -euo pipefail

cd /gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention

module purge
module load python/3.10.4

HF_VENV_DIR="${HF_VENV_DIR:-.venv}"
HELMET_DATA_DIR="${HELMET_DATA_DIR:-third_party/benchmarks/HELMET/data}"

mkdir -p "${HELMET_DATA_DIR}/infbench"

export HELMET_DATA_DIR
"${HF_VENV_DIR}/bin/python" - <<'PY'
import os
from pathlib import Path
from shutil import copy2

from huggingface_hub import hf_hub_download

out_dir = Path(os.environ["HELMET_DATA_DIR"]) / "infbench"
out_dir.mkdir(parents=True, exist_ok=True)
for filename in ("longbook_qa_eng.jsonl", "longbook_choice_eng.jsonl"):
    target = out_dir / filename
    if target.exists():
        print(f"[INFO] {target} already exists")
        continue
    cached = Path(
        hf_hub_download(
            repo_id="xinrongzhang2022/InfiniteBench",
            filename=filename,
            repo_type="dataset",
            local_files_only=False,
        )
    )
    copy2(cached, target)
    print(f"[INFO] copied {cached} -> {target}")
PY
