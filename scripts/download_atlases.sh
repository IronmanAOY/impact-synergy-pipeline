#!/usr/bin/env bash
set -e
python - <<'PYCODE'
from nilearn import datasets
datasets.fetch_atlas_schaefer_2018(n_rois=400, data_dir='atlases', overwrite=False)
datasets.fetch_atlas_aal(data_dir='atlases', overwrite=False)
datasets.fetch_atlas_shen_2015(data_dir='atlases', n_parcels=268, overwrite=False)
PYCODE
