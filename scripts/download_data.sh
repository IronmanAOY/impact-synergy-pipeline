#!/usr/bin/env bash
set -e

DATASET_ID="${1:-ds003171}"
SNAPSHOT="${2:-}"
OUT_DIR="${3:-./data/scratch/${DATASET_ID}}"

mkdir -p ./data/scratch
if ! command -v openneuro >/dev/null 2>&1; then
  npm install -g @openneuro/cli@2.0.1
fi

if [ -n "${SNAPSHOT}" ]; then
  openneuro download --snapshot "${SNAPSHOT}" "${DATASET_ID}" "${OUT_DIR}"
else
  openneuro download "${DATASET_ID}" "${OUT_DIR}"
fi

# Optional independent fMRI replication dataset used by ds003171.
if [ "${DATASET_ID}" = "ds003171" ]; then
  if [ ! -d ./data/scratch/melbourne/.git ]; then
    git clone https://github.com/MelbourneHci/MelbournePropofolData.git ./data/scratch/melbourne
  fi
fi
