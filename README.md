[![DOI](https://zenodo.org/badge/975114052.svg)](https://doi.org/10.5281/zenodo.15306740)

# IMPaCT Synergy Pipeline

This repository contains the IMPaCT metric pipeline for EEG and fMRI data.

The codebase is organized as a Python package under `src/impact_pipeline`, with
`run_pipeline.py` as the main command-line entrypoint.

## Repository Layout

- `src/impact_pipeline/`: preprocessing, metric computation, CI orchestration, and dashboard support code
- `run_pipeline.py`: main pipeline entrypoint
- `scripts/`: command-line tools for data download, atlas download, preprocessing inputs, desktop launch, dashboard launch, and synthetic dataset generation
- `tests/`: regression and smoke tests
- `docs/metrics.md`: implementation-facing metric definitions
- `docs/synthetic_data.md`: synthetic validation dataset generation and validation notes
- `data/managed/`: small managed reference files that are safe to version

## Installation

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate impact-synergy-clean
```

## External Requirements

The Python environment covers the analysis code itself. Some workflows also need
external tools:

- Docker, for fMRIPrep from raw fMRI BIDS data
- Node.js and `npm`, for OpenNeuro CLI downloads
- A valid FreeSurfer license, for fMRIPrep with recon-all enabled

FreeSurfer license handling:

- preferred: set `FS_LICENSE=/absolute/path/to/license.txt`
- alternative: place your local license at `licenses/fs_license.txt`
- a template is included at `licenses/fs_license.txt.example`

## Downloading Data

Download OpenNeuro datasets with:

```bash
bash scripts/download_data.sh ds003171 2.0.1
bash scripts/download_data.sh ds005620
```

To regenerate the synthetic validation datasets, download the required
OpenNeuro source snapshots into a single local source root and pass that root to
the inspection and generation commands:

```bash
export IMPACT_SOURCE_ROOT=/absolute/path/to/openneuro_sources

bash scripts/download_data.sh ds003171 2.0.1 "$IMPACT_SOURCE_ROOT/ds003171"
bash scripts/download_data.sh ds005620 1.0.0 "$IMPACT_SOURCE_ROOT/ds005620"
bash scripts/download_data.sh ds002547 1.1.0 "$IMPACT_SOURCE_ROOT/ds002547"
bash scripts/download_data.sh ds005479 1.1.1 "$IMPACT_SOURCE_ROOT/ds005479"
bash scripts/download_data.sh ds004295 1.0.0 "$IMPACT_SOURCE_ROOT/ds004295"
bash scripts/download_data.sh ds002336 2.0.2 "$IMPACT_SOURCE_ROOT/ds002336"
```

Download atlas assets with:

```bash
bash scripts/download_atlases.sh
```

The package catalog defines the supported dataset metadata, local root
candidates, target MPC roles, and pipeline support flags. CI should only be
computed when RAM, PDI, NAS, IIM, and SRPI are explicitly defined for the run;
missing components are not imputed.

## Synthetic Validation Datasets

The repository can generate compact real-data-derived synthetic datasets for
software validation. They are derived from supported OpenNeuro source snapshots
and exercise the full RAM, PDI, NAS, IIM, SRPI, and CI path without
redistributing participant-level payloads.

Two setup paths are supported:

- download the open archive at https://doi.org/10.5281/zenodo.20786673
- regenerate the datasets locally from the OpenNeuro source snapshots listed above

After downloading the archive, extract it so the selected root contains
`test_objects/datasets/real_derived_synth_completed/` and
`test_objects/runs/real_derived_synth_completed/`, then run:

```bash
export IMPACT_SYNTH_ROOT=/absolute/path/that/contains/test_objects
conda run -n impact-synergy-clean python scripts/generate_real_derived_synth_completed.py \
  --validate-only
```

To regenerate the datasets, download the source snapshots listed above, then run:

```bash
export IMPACT_SYNTH_ROOT=/absolute/path/to/impact-synth-output
conda run -n impact-synergy-clean python scripts/inspect_real_sources_for_synth.py \
  --source-root "$IMPACT_SOURCE_ROOT" \
  --output-dir "$IMPACT_SYNTH_ROOT/test_objects/real_derived_synth_completed/reports"
conda run -n impact-synergy-clean python scripts/generate_real_derived_synth_completed.py \
  --source-root "$IMPACT_SOURCE_ROOT"
```

See `docs/synthetic_data.md` for workflow details.

## Running The Pipeline Locally

### fMRI Example

```bash
conda run -n impact-synergy-clean python run_pipeline.py \
  --execution-mode local \
  --dataset-id ds003171 \
  --out-dir outputs/scratch \
  --run-preprocessing \
  --mpc-metrics PDI NAS IIM \
  --no-ci
```

### EEG Example

```bash
conda run -n impact-synergy-clean python run_pipeline.py \
  --execution-mode local \
  --dataset-id ds005620 \
  --out-dir outputs/scratch \
  --run-preprocessing \
  --mpc-metrics PDI NAS IIM \
  --no-ci
```

## Hunter Execution

Hunter execution is selected with `--execution-mode hunter`. The same metric
logic is used as in local runs, while IIM can be prepared, sharded, and reduced
through Slurm jobs.

Set the site-specific Slurm/runtime values before building a Hunter campaign:

```bash
export IMPACT_HUNTER_CONDA_ENV=impact-synergy-clean
export IMPACT_HUNTER_CPU_PARTITION=<cpu-partition>
export IMPACT_HUNTER_APU_PARTITION=<apu-partition>
export IMPACT_HUNTER_SLURM_ACCOUNT=<project-account>  # if required
export IMPACT_HUNTER_SLURM_QOS=<qos>                  # if required
export IMPACT_HUNTER_SLURM_SETUP_FILE=/absolute/path/to/hunter_slurm_setup.sh
```

The optional setup file is copied into generated `.sbatch` files and can contain
site-specific `module load`, conda initialization, and temporary cache exports.

Build a Hunter campaign from an archived or generated synthetic dataset:

```bash
export IMPACT_SYNTH_ROOT=/absolute/path/that/contains/test_objects
export IMPACT_SYNTH_DATASET=ds003171

conda run -n impact-synergy-clean python run_pipeline.py \
  --execution-mode hunter \
  --hardware-target hunter-apu \
  --hunter-stage build-campaign \
  --data-origin dummy \
  --dataset-id "$IMPACT_SYNTH_DATASET" \
  --bids-root "$IMPACT_SYNTH_ROOT/test_objects/datasets/real_derived_synth_completed/$IMPACT_SYNTH_DATASET" \
  --out-dir "$IMPACT_SYNTH_ROOT/test_objects/runs/real_derived_synth_completed/$IMPACT_SYNTH_DATASET" \
  --mpc-metrics RAM PDI NAS IIM SRPI
```

Use `ds003171`, `ds002547`, or `ds005620` for `IMPACT_SYNTH_DATASET`. For raw
OpenNeuro runs, add `--run-preprocessing` and set `--bids-root` to the local
BIDS root.

The campaign is written under:

```text
<out-dir>/cache/hunter_iim_campaign/
```

Slurm submission files are written under:

```text
<out-dir>/cache/hunter_iim_campaign/slurm/
```

Submit the generated campaign with:

```bash
bash <out-dir>/cache/hunter_iim_campaign/slurm/00_submit_all.sh
```

Hardware selection is explicit. Use `--hardware-target cpu` for NumPy/SciPy CPU
execution, `--hardware-target auto` to use CuPy when available and otherwise use
CPU, `--hardware-target gpu` to require a visible CuPy GPU, or
`--hardware-target hunter-apu` to require a ROCm/HIP CuPy runtime for Hunter APU
jobs. Explicit GPU/APU modes fail early if the requested accelerator is not
visible.

The hardware target is passed through RAM, PDI, NAS, IIM, SRPI, and final CI
computation. The GPU/APU backend uses CuPy/ROCm array operations for
matrix-heavy MPC kernels while keeping CPU implementations available for normal
workstations.

## Dashboard And Desktop Launcher

### Browser Dashboard

```bash
conda run -n impact-synergy-clean python scripts/live_dashboard.py \
  --out-dir outputs/scratch \
  --dataset-id ds003171 \
  --port 8765
```

Then open:

```text
http://127.0.0.1:8765
```

The dashboard provides a three-step workflow:

1. select or upload one or more datasets
2. configure run settings and per-metric dataset mapping
3. review live metrics and control the active run

Datasets marked as synthetic are routed to the synthetic validation output area.

### Desktop Launcher

```bash
conda run -n impact-synergy-clean python scripts/impact_desktop_app.py
```

Double-click launchers:

- macOS: `scripts/start_impact_desktop.command`
- Windows: `scripts/start_impact_desktop.bat`

## fMRIPrep Inputs

Fetch fMRIPrep derivatives for ds003171 with:

```bash
export FS_LICENSE=~/license.txt
bash scripts/fetch_fmriprep_ds003171.sh
```

To skip recon-all:

```bash
bash scripts/fetch_fmriprep_ds003171.sh --skip-reconall
```

For another dataset root:

```bash
bash scripts/fetch_fmriprep.sh --bids-root /path/to/bids --dataset-id ds003171
```

## Testing

Run the full test suite:

```bash
conda run -n impact-synergy-clean pytest -q
```

The repository also includes GitHub Actions CI and container definitions for
Docker and Singularity-based setups.


## Citation

If you use the software, please cite the archived release listed in
`CITATION.cff`. The DOI badge above points to the Zenodo record for the
repository.

If you use the open synthetic validation archive, please cite:
Obiri-Yeboah, A. (2026). Synthetic validation dataset [Data set]. Zenodo.
https://doi.org/10.5281/zenodo.20786673.
