#!/usr/bin/env bash

# Shared HuggingFace model preset resolver for benchmark wrappers.
# Call resolve_hf_model_preset "$HF_MODEL_PRESET", then consume the
# PRESET_* variables it sets. The function intentionally falls back to the
# public repo id when a local snapshot is not present; LOCAL_FILES_ONLY=1 will
# then fail cleanly at model-load time instead of silently using another model.

hf_model_preset_supported_names() {
  printf '%s\n' \
    "qwen3_8b" \
    "qwen3_14b" \
    "qwen3_5_9b" \
    "llama31_8b" \
    "llama3_1_8b" \
    "mistral_nemo_12b" \
    "glm4_9b" \
    "phi4_reasoning_14b"
}

_hf_model_preset_hub_cache() {
  printf '%s\n' "${HF_HUB_CACHE:-${HF_CACHE_DIR:-.hf_cache}/hub}"
}

_hf_model_preset_snapshot_or_repo() {
  local cache_name="$1"
  local repo_id="$2"
  local snapshot_root
  local snapshot

  snapshot_root="$(_hf_model_preset_hub_cache)/${cache_name}/snapshots"
  if [ -d "${snapshot_root}" ]; then
    snapshot="$(find "${snapshot_root}" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
    if [ -n "${snapshot}" ]; then
      printf '%s\n' "${snapshot}"
      return 0
    fi
  fi
  printf '%s\n' "${repo_id}"
}

resolve_hf_model_preset() {
  local preset="${1:-qwen3_8b}"

  PRESET_MODEL_NAME=""
  PRESET_HF_VENV_DIR=""
  PRESET_HF_EXTRA_PYTHONPATH=""
  PRESET_TRUST_REMOTE_CODE=""
  PRESET_HF_LANGUAGE_MODEL_ONLY=""
  PRESET_USE_CHAT_TEMPLATE=""
  PRESET_DISABLE_THINKING=""
  PRESET_QWEN_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS="32768"

  case "${preset}" in
    ""|qwen3_8b)
      PRESET_MODEL_NAME="$(_hf_model_preset_snapshot_or_repo "models--Qwen--Qwen3-8B" "Qwen/Qwen3-8B")"
      PRESET_HF_EXTRA_PYTHONPATH=".hf_pydeps"
      PRESET_HF_LANGUAGE_MODEL_ONLY="1"
      PRESET_USE_CHAT_TEMPLATE="1"
      PRESET_DISABLE_THINKING="1"
      PRESET_QWEN_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS="32768"
      ;;
    qwen3_14b)
      PRESET_MODEL_NAME="$(_hf_model_preset_snapshot_or_repo "models--Qwen--Qwen3-14B" "Qwen/Qwen3-14B")"
      PRESET_HF_VENV_DIR=".venv_cu128"
      PRESET_HF_EXTRA_PYTHONPATH=".hf_pydeps_cu128"
      PRESET_HF_LANGUAGE_MODEL_ONLY="1"
      PRESET_USE_CHAT_TEMPLATE="1"
      PRESET_DISABLE_THINKING="1"
      PRESET_QWEN_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS="32768"
      ;;
    qwen3_5_9b)
      PRESET_MODEL_NAME="$(_hf_model_preset_snapshot_or_repo "models--Qwen--Qwen3.5-9B" "Qwen/Qwen3.5-9B")"
      PRESET_HF_EXTRA_PYTHONPATH=".hf_pydeps"
      PRESET_HF_LANGUAGE_MODEL_ONLY="1"
      PRESET_USE_CHAT_TEMPLATE="1"
      PRESET_DISABLE_THINKING="1"
      PRESET_QWEN_YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS="262144"
      ;;
    llama31_8b|llama3_1_8b)
      PRESET_MODEL_NAME="$(_hf_model_preset_snapshot_or_repo "models--meta-llama--Llama-3.1-8B-Instruct" "meta-llama/Llama-3.1-8B-Instruct")"
      PRESET_HF_LANGUAGE_MODEL_ONLY="1"
      PRESET_USE_CHAT_TEMPLATE="1"
      ;;
    mistral_nemo_12b)
      PRESET_MODEL_NAME="$(_hf_model_preset_snapshot_or_repo "models--mistralai--Mistral-Nemo-Instruct-2407" "mistralai/Mistral-Nemo-Instruct-2407")"
      PRESET_HF_EXTRA_PYTHONPATH=".hf_pydeps"
      PRESET_HF_LANGUAGE_MODEL_ONLY="1"
      PRESET_USE_CHAT_TEMPLATE="1"
      ;;
    glm4_9b)
      PRESET_MODEL_NAME="$(_hf_model_preset_snapshot_or_repo "models--zai-org--glm-4-9b-chat-hf" "zai-org/glm-4-9b-chat-hf")"
      PRESET_HF_EXTRA_PYTHONPATH=".hf_pydeps"
      PRESET_TRUST_REMOTE_CODE="1"
      PRESET_HF_LANGUAGE_MODEL_ONLY="1"
      PRESET_USE_CHAT_TEMPLATE="1"
      ;;
    phi4_reasoning_14b)
      PRESET_MODEL_NAME="$(_hf_model_preset_snapshot_or_repo "models--microsoft--Phi-4-reasoning" "microsoft/Phi-4-reasoning")"
      PRESET_HF_VENV_DIR=".venv_cu128"
      PRESET_HF_EXTRA_PYTHONPATH=".hf_pydeps_cu128"
      PRESET_HF_LANGUAGE_MODEL_ONLY="1"
      PRESET_USE_CHAT_TEMPLATE="1"
      ;;
    *)
      echo "[ERROR] Unknown HF_MODEL_PRESET=${preset}" >&2
      echo "[ERROR] Supported presets: $(hf_model_preset_supported_names | paste -sd, -)" >&2
      return 2
      ;;
  esac
}
