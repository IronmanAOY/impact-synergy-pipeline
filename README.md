[![DOI](https://zenodo.org/badge/975114052.svg)](https://doi.org/10.5281/zenodo.15306740)

# IMPACT Synergy Pipeline

Reproduces all analyses for “Parameter Sensitivity and Pilot Simulation of Synergy Metrics.”

## Requirements

```bash
conda env create -f environment.yml
conda activate impact-synergy
```
## Atlases
We fetch Schaefer-400 and AAL automatically, but you must manually download Shen-268 (1 mm) into `atlases/shen_1mm_268_parcellation.nii.gz`.  

## Host Prerequisites
- Node ≥14 + npm (for download_data.sh)

  - macOS: brew install node

  - Ubuntu: sudo apt-get update && sudo apt-get install -y nodejs npm

- Docker (for FMRIPrep containers)
- FreeSurfer (install via Homebrew, apt, or from https://freesurfer.net)
- you need Nilearn ≥ 0.11 (to get fetch_atlas_schaefer_2018) but note that Tedana may conflict if you bump to 0.12.

## Data Download
```bash
download_data.sh  # grabs 17-subject ds003171 snapshot
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

## KI-Ironman-Trainingscoach

Dieses Repository enthält jetzt auch einen interaktiven Trainingscoach,
der Athlet:innen Schritt für Schritt zur Ironman-Langdistanz begleitet.

```bash
python scripts/ironman_coach.py            # interaktive Eingabe
python scripts/ironman_coach.py --demo     # Beispielplan ohne Rückfragen
python scripts/ironman_coach.py --output plan.json --search Hamburg
```

Der Coach lädt auf Wunsch passende Ironman-Rennen (mit Fallback auf eine
Offline-Liste), erfasst deinen aktuellen Fitnesszustand und deine
Zielsetzung und generiert anschließend einen individuellen
Vorbereitungsplan zwischen 14 und 42 Wochen inklusive Wochenstruktur,
Legende und Hintergrundinformationen.
## Cleaning & ROI Extraction
Once fMRIPrep is done, generate cleaned time-series and meanFD by:
```bash
python - <<'PYCODE'
from preprocessing import run_preprocessing
run_preprocessing('outputs/fmriprep/fmriprep', 'test_outputs/preprocessed')
PYCODE```

## Cite this repository

Anthony Obiri-Yeboah (2025). *IMPACT Synergy Pipeline* (v1.0.0).  
DOI: 10.5281/zenodo.15306741

## Related publication

Obiri-Yeboah, A. (in prep).  
**IMPACT: Integrated Minimal Principles Accounting for Consciousness Theory**.  

