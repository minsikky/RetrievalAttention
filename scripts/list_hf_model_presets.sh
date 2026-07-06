#!/usr/bin/env bash
set -euo pipefail

cd /gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention

HF_CACHE_DIR="${HF_CACHE_DIR:-$(pwd)/.hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_CACHE_DIR}/hub}"

source scripts/hf_model_presets.sh

if [ "$#" -gt 0 ]; then
  presets=("$@")
else
  mapfile -t presets < <(hf_model_preset_supported_names)
fi

printf "preset\tmodel_name\tlocal_snapshot\thf_venv_dir\thf_extra_pythonpath\ttrust_remote_code\n"
for preset in "${presets[@]}"; do
  resolve_hf_model_preset "${preset}"
  local_snapshot="0"
  if [ -d "${PRESET_MODEL_NAME}" ]; then
    local_snapshot="1"
  fi
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${preset}" \
    "${PRESET_MODEL_NAME}" \
    "${local_snapshot}" \
    "${PRESET_HF_VENV_DIR:-}" \
    "${PRESET_HF_EXTRA_PYTHONPATH:-}" \
    "${PRESET_TRUST_REMOTE_CODE:-0}"
done
