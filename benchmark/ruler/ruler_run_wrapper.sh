#!/bin/bash
#SBATCH --job-name=ruler
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=64000m
#SBATCH --time=01:00:00
#SBATCH --account=zhengya98
#SBATCH --partition=spgpu
#SBATCH --gpus-per-node=1

set -euo pipefail

echo "[INFO] Job started at: $(date)"
echo "[INFO] Host: $(hostname)"

module purge
module load python/3.10.4
module unload pytorch 2>/dev/null || true

if [ -f ".venv/bin/activate" ]; then
  # Activate local virtual environment
  # shellcheck disable=SC1091
  source .venv/bin/activate
else
  echo "[ERROR] .venv/bin/activate not found. Did you set up the venv?"
  exit 1
fi

unset PYTHONPATH
unset PYTHONHOME
export PYTHONNOUSERSITE=1
MODEL_NAME="${MODEL_NAME:-llama-3.1-8b}"
BENCHMARK_NAME="${BENCHMARK_NAME:-synthetic}"
ATTN_TYPE="${ATTN_TYPE:-RetroInfer}"
CONTEXT_LEN="${CONTEXT_LEN:-131072}"
TASK_NAME="${TASK_NAME:-vt}"
DTYPE="${DTYPE:-bf16}"
BATCH_SIZE="${BATCH_SIZE:-1}"
BUDGET_RATIO="${BUDGET_RATIO:-0.018}"
ESTIMATE_RATIO="${ESTIMATE_RATIO:-0.232}"
TOKEN_BUDGET_OVERRIDE="${TOKEN_BUDGET_OVERRIDE:-100}"
TOKEN_BUDGET_RATIO="${TOKEN_BUDGET_RATIO:-}"
NUM_SAMPLES="${NUM_SAMPLES:-20}"
LOW_CPU_MEM_USAGE="${LOW_CPU_MEM_USAGE:-1}"
REUSE_DATA="${REUSE_DATA:-1}"
FORCE_PREPARE="${FORCE_PREPARE:-0}"
ENABLE_PROFILER="${ENABLE_PROFILER:-0}"
PROFILER_DIR="${PROFILER_DIR:-ruler_eval_result/profiling}"
FORCE_PRED="${FORCE_PRED:-1}"
PROFILER_SAFE="${PROFILER_SAFE:-0}"

echo "[INFO] MODEL_NAME=${MODEL_NAME}"
echo "[INFO] BENCHMARK_NAME=${BENCHMARK_NAME}"
echo "[INFO] ATTN_TYPE=${ATTN_TYPE}"
echo "[INFO] CONTEXT_LEN=${CONTEXT_LEN}"
echo "[INFO] TASK_NAME=${TASK_NAME}"
echo "[INFO] DTYPE=${DTYPE}"
echo "[INFO] BATCH_SIZE=${BATCH_SIZE}"
echo "[INFO] BUDGET_RATIO=${BUDGET_RATIO}"
echo "[INFO] ESTIMATE_RATIO=${ESTIMATE_RATIO}"
echo "[INFO] TOKEN_BUDGET_OVERRIDE=${TOKEN_BUDGET_OVERRIDE}"
echo "[INFO] TOKEN_BUDGET_RATIO=${TOKEN_BUDGET_RATIO}"
echo "[INFO] NUM_SAMPLES=${NUM_SAMPLES}"
echo "[INFO] LOW_CPU_MEM_USAGE=${LOW_CPU_MEM_USAGE}"
echo "[INFO] REUSE_DATA=${REUSE_DATA}"
echo "[INFO] FORCE_PREPARE=${FORCE_PREPARE}"
echo "[INFO] ENABLE_PROFILER=${ENABLE_PROFILER}"
echo "[INFO] PROFILER_DIR=${PROFILER_DIR}"
echo "[INFO] FORCE_PRED=${FORCE_PRED}"
echo "[INFO] PROFILER_SAFE=${PROFILER_SAFE}"
echo "[INFO] Python: $(which python)"
python -V
python - <<'PY'
import sys
import torch
print("[INFO] torch version:", torch.__version__)
print("[INFO] torch file:", torch.__file__)
PY

# Preflight CPU memory estimate and Slurm request check (RetroInfer only)
if [ "${ATTN_TYPE}" = "RetroInfer" ]; then
  python - <<'PY'
import json
import os
import math

model_name = os.environ.get("MODEL_NAME", "llama-3.1-8b")
context_len = int(os.environ.get("CONTEXT_LEN", "131072"))
task_name = os.environ.get("TASK_NAME", "vt")
dtype = os.environ.get("DTYPE", "bf16").lower()
batch_size = int(os.environ.get("BATCH_SIZE", "1"))

# tokens_to_generate mapping for RULER synthetic tasks
TOKENS_TO_GENERATE = {
    "vt": 30,
    "cwe": 120,
    "fwe": 50,
    "qa_1": 32,
    "qa_2": 32,
    "niah_single_1": 128,
    "niah_single_2": 128,
    "niah_single_3": 128,
    "niah_multikey_1": 128,
    "niah_multikey_2": 128,
    "niah_multikey_3": 128,
    "niah_multivalue": 128,
    "niah_multiquery": 128,
}
max_new_len = TOKENS_TO_GENERATE.get(task_name, 128)
if task_name not in TOKENS_TO_GENERATE:
    print(f"[WARN] Unknown TASK_NAME={task_name}; defaulting tokens_to_generate=128")

# Load RetroInfer config (static patterns) from repo config
repo_root = os.environ.get("SLURM_SUBMIT_DIR", os.getcwd())
cfg_path = os.path.join(repo_root, "config", "Llama-3.1-8B-Instruct.json")
with open(cfg_path, "r", encoding="utf-8") as f:
    retro_cfg = json.load(f)["RetroInfer"]

static_pattern_total = int(retro_cfg["static_pattern_start"]) + int(retro_cfg["static_pattern_end"])

# Model dimensions: try HF config, fallback to known Llama-3.1-8B defaults
num_layers = 32
num_heads = 32
num_kv_heads = 8
hidden_size = 4096

try:
    from transformers import LlamaConfig
    # Prefer local cache; avoid downloads in preflight
    hf_name = "meta-llama/Llama-3.1-8B-Instruct" if model_name == "llama-3.1-8b" else model_name
    cfg = LlamaConfig.from_pretrained(hf_name, local_files_only=True)
    num_layers = int(cfg.num_hidden_layers)
    num_heads = int(cfg.num_attention_heads)
    num_kv_heads = int(cfg.num_key_value_heads)
    hidden_size = int(cfg.hidden_size)
except Exception as e:
    print(f"[WARN] Could not load HF config locally ({e}); using defaults for Llama-3.1-8B.")

head_dim = hidden_size // num_heads
dtype_bytes = 2 if dtype in ("bf16", "fp16") else 4

input_length = max(0, context_len - max_new_len)
list_stride = max(0, input_length - static_pattern_total)

# Dominant pinned CPU memory: list_keys + list_values
bytes_kv = 2 * num_layers * batch_size * num_kv_heads * list_stride * head_dim * dtype_bytes

# Add 20% overhead for other pinned buffers and metadata
estimated_bytes = int(bytes_kv * 1.2)
estimated_gib = estimated_bytes / (1024 ** 3)

# Apply headroom factor
headroom = 1.25
required_gib = estimated_gib * headroom

print(f"[INFO] Estimated CPU RAM (pinned KV only, +20% overhead): {estimated_gib:.2f} GiB")
print(f"[INFO] Required CPU RAM with headroom {headroom:.2f}x: {required_gib:.2f} GiB")

# Read Slurm memory request (MB)
mem_per_node = os.environ.get("SLURM_MEM_PER_NODE")
mem_per_cpu = os.environ.get("SLURM_MEM_PER_CPU")
cpus_on_node = os.environ.get("SLURM_CPUS_ON_NODE")

slurm_gib = None
if mem_per_node:
    slurm_gib = float(mem_per_node) / 1024.0
elif mem_per_cpu and cpus_on_node:
    slurm_gib = float(mem_per_cpu) * float(cpus_on_node) / 1024.0

if slurm_gib is None:
    print("[WARN] Could not determine Slurm memory request; continuing without guard.")
else:
    print(f"[INFO] Slurm requested memory: {slurm_gib:.2f} GiB")
    if slurm_gib < required_gib:
        print(f"[ERROR] Slurm memory too small: {slurm_gib:.2f} GiB < required {required_gib:.2f} GiB.")
        print("[ERROR] Please increase --mem in sbatch and retry.")
        raise SystemExit(2)
PY
fi

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  SCRIPT_DIR="${SLURM_SUBMIT_DIR}/benchmark/ruler"
else
  SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
fi

cd "${SCRIPT_DIR}"

LOW_CPU_MEM_USAGE="${LOW_CPU_MEM_USAGE}" \
NUM_SAMPLES="${NUM_SAMPLES}" \
REUSE_DATA="${REUSE_DATA}" \
FORCE_PREPARE="${FORCE_PREPARE}" \
ENABLE_PROFILER="${ENABLE_PROFILER}" \
PROFILER_DIR="${PROFILER_DIR}" \
FORCE_PRED="${FORCE_PRED}" \
PROFILER_SAFE="${PROFILER_SAFE}" \
TOKEN_BUDGET_OVERRIDE="${TOKEN_BUDGET_OVERRIDE}" \
TOKEN_BUDGET_RATIO="${TOKEN_BUDGET_RATIO}" \
bash ./ruler_run.sh \
  "${MODEL_NAME}" \
  "${BENCHMARK_NAME}" \
  "${ATTN_TYPE}" \
  "${CONTEXT_LEN}" \
  "${TASK_NAME}" \
  "${DTYPE}" \
  "${BUDGET_RATIO}" \
  "${ESTIMATE_RATIO}"

echo "[INFO] Job finished at: $(date)"
