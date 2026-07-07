#!/usr/bin/env bash
#SBATCH --job-name=dl-qwen-1m
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8000m
#SBATCH --time=03:00:00
#SBATCH --account=zhengya0
#SBATCH --partition=standard
set -euo pipefail
cd /gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention
module purge
module load python/3.10.4
export LD_LIBRARY_PATH="/sw/pkgs/arc/python/3.10.4/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="$PWD/.hf_cache"
.venv/bin/huggingface-cli download Qwen/Qwen2.5-7B-Instruct-1M --exclude "*.gguf"
echo "[dl] complete; incomplete blobs left:"
find .hf_cache/hub/models--Qwen--Qwen2.5-7B-Instruct-1M/blobs -name '*.incomplete' 2>/dev/null | wc -l
du -sh .hf_cache/hub/models--Qwen--Qwen2.5-7B-Instruct-1M
