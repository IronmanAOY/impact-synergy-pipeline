# IMPACT Synergy Pipeline

Reproduces all analyses for “Parameter Sensitivity and Pilot Simulation of Synergy Metrics.”

## Requirements

```bash
conda env create -f environment.yml
conda activate impact-synergy
```

## Host Prerequisites
- Node ≥14 + npm (for download_data.sh)

  - macOS: brew install node

  - Ubuntu: sudo apt-get update && sudo apt-get install -y nodejs npm

- Docker (for FMRIPrep containers)

## Data Download
```bash
download_data.sh
```
Or in Docker:
```bash
docker run --rm \
  -v $(pwd)/data:/workspace/data -w /workspace \
  node:14-buster bash -lc "\
    npm install -g @openneuro/cli@2.0.1 && \
    openneuro download --snapshot 2.0.1 ds003171 ./data/ds003171"
```
## fMRIPrep Derivatives
1. OpenNeuro (ds003171)
```bash
# with license:
export FS_LICENSE=~/license.txt
bash scripts/fetch_fmriprep_ds003171.sh

# or skip recon-all:
bash scripts/fetch_fmriprep_ds003171.sh --skip-reconall
```
2. Melbourne Propofol
```bash
export FS_LICENSE=~/license.txt
bash scripts/fetch_fmriprep.sh
```
Note: We call `BIDSLayout(validate=False)`; run `bids-validator` separately for compliance.
## Run the Pipeline
```bash
python run_pipeline.py --out-dir outputs
```
Outputs in `outputs/`. See `docs/metrics.md` for equations. 
License: MIT
## Cite this repository

Anthony Obiri-Yeboah (2025). *IMPACT Synergy Pipeline* (v1.0.0).  
DOI: [pending]  
