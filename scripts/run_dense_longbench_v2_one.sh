#!/usr/bin/env bash
#SBATCH --job-name=dense-lbv2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128000m
#SBATCH --time=02:00:00
#SBATCH --account=zhengya98
#SBATCH --partition=spgpu
#SBATCH --gpus-per-node=1
set -euo pipefail

# Dense/reference preset for one LongBench-v2 slice.

cd /gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention

export HF_MODEL_PRESET="${HF_MODEL_PRESET:-qwen3_8b}"
export OUTPUT_DIR="${OUTPUT_DIR:-longbench_v2_hf_result/dense_lbv2}"
export ATTENTION_MODE="${ATTENTION_MODE:-dense}"
export MAX_EXAMPLES="${MAX_EXAMPLES:-64}"
export LENGTH_FILTER="${LENGTH_FILTER:-short}"
export DIFFICULTY_FILTER="${DIFFICULTY_FILTER:-easy}"
export MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-8192}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
export TEMPERATURE="${TEMPERATURE:-0.0}"
export LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
export HF_CACHE_DIR="${HF_CACHE_DIR:-$(pwd)/.hf_cache}"

exec bash benchmark/run_longbench_v2_hf.sh
