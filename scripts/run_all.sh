#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

bash download_data.sh
bash scripts/download_atlases.sh
bash scripts/fetch_fmriprep_ds003171.sh --skip-reconall

if [ ! -d data/melbourne/derivatives ]; then
  echo "Please fetch fMRIPrep for Melbourne:"
  echo "  export FS_LICENSE=~/license.txt"
  echo "  bash scripts/fetch_fmriprep.sh"
  exit 1
fi

python run_pipeline.py --out-dir outputs
