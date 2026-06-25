#!/usr/bin/env bash
set -e
CONDA_BIN="${CONDA_BIN:-$(command -v conda || true)}"
CONDA_ENV="${IMPACT_CONDA_ENV:-impact-synergy-clean}"

if [ -z "${CONDA_BIN}" ]; then
  echo "conda not found in PATH. Install/initialize conda first."
  exit 1
fi

"${CONDA_BIN}" run -n "${CONDA_ENV}" python - <<'PYCODE'
from nilearn import datasets
datasets.fetch_atlas_schaefer_2018(n_rois=400, data_dir='atlases', overwrite=False)
datasets.fetch_atlas_aal(data_dir='atlases', overwrite=False)
datasets.fetch_atlas_shen_2015(data_dir='atlases', n_parcels=268, overwrite=False)
PYCODE
