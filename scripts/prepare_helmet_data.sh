#!/usr/bin/env bash
#SBATCH --job-name=helmet-data
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64000m
#SBATCH --time=12:00:00
#SBATCH --account=zhengya98
#SBATCH --partition=standard

set -euo pipefail

cd /gpfs/accounts/zhengya_root/zhengya98/minsikky/long_context/RetrievalAttention

HELMET_REPO="${HELMET_REPO:-third_party/benchmarks/HELMET}"
DATA_DIR="${HELMET_DATA_DIR:-${HELMET_REPO}/data}"
ARCHIVE="${HELMET_ARCHIVE:-${HELMET_REPO}/data.tar.gz}"
URL="${HELMET_DATA_URL:-https://huggingface.co/datasets/princeton-nlp/HELMET/resolve/main/data.tar.gz}"

mkdir -p "${HELMET_REPO}"

if [ -d "${DATA_DIR}/kilt" ] && [ -d "${DATA_DIR}/ruler" ]; then
  echo "[INFO] HELMET data already present at ${DATA_DIR}"
  exit 0
fi

echo "[INFO] downloading HELMET data archive to ${ARCHIVE}"
if command -v curl >/dev/null 2>&1; then
  curl -L --continue-at - --output "${ARCHIVE}" "${URL}"
elif command -v wget >/dev/null 2>&1; then
  wget -c -O "${ARCHIVE}" "${URL}"
else
  echo "[ERROR] neither curl nor wget found" >&2
  exit 2
fi

echo "[INFO] extracting ${ARCHIVE} under ${HELMET_REPO}"
tar -xzf "${ARCHIVE}" -C "${HELMET_REPO}"

if [ ! -d "${DATA_DIR}/kilt" ] || [ ! -d "${DATA_DIR}/ruler" ]; then
  echo "[ERROR] extracted archive did not create expected HELMET data directories under ${DATA_DIR}" >&2
  exit 3
fi

echo "[INFO] HELMET data ready at ${DATA_DIR}"
