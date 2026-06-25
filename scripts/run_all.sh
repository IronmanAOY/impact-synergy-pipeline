#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

DATASET_ID="${1:-ds003171}"
CONDA_BIN="${CONDA_BIN:-$(command -v conda || true)}"
CONDA_ENV="${IMPACT_CONDA_ENV:-impact-synergy-clean}"

if [ -z "${CONDA_BIN}" ]; then
  echo "conda not found in PATH. Install/initialize conda first."
  exit 1
fi

run_pipeline() {
  MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}" \
  NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache}" \
  "${CONDA_BIN}" run -n "${CONDA_ENV}" python run_pipeline.py "$@"
}

bash scripts/download_data.sh "${DATASET_ID}" "${2:-}"

if [ "${DATASET_ID}" = "ds003171" ]; then
  bash scripts/download_atlases.sh
  bash scripts/fetch_fmriprep_ds003171.sh --skip-reconall

  if [ ! -d data/scratch/melbourne/derivatives ]; then
    echo "Please fetch fMRIPrep for Melbourne:"
    echo "  export FS_LICENSE=~/license.txt"
    echo "  bash scripts/fetch_fmriprep.sh"
    exit 1
  fi

  run_pipeline \
    --dataset-id ds003171 \
    --out-dir outputs/scratch \
    --run-preprocessing \
    --mpc-metrics PDI NAS IIM \
    --no-ci
elif [ "${DATASET_ID}" = "ds005620" ]; then
  run_pipeline \
    --dataset-id ds005620 \
    --out-dir outputs/scratch \
    --run-preprocessing \
    --mpc-metrics PDI NAS IIM \
    --no-ci
else
  echo "Unsupported DATASET_ID=${DATASET_ID}"
  exit 1
fi
