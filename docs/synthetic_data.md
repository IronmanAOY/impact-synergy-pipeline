# Real-Data-Derived Synthetic Datasets

This repository can generate compact synthetic datasets for IMPaCT pipeline
validation from supported OpenNeuro source snapshots.

## Scope

The generated package targets:

- `ds003171`: fMRI state-switch structure
- `ds002547`: fMRI self/other structure
- `ds005620`: EEG sedation/wake structure

The archived synthetic validation package is available on Zenodo:

- Obiri-Yeboah, A. (2026). Synthetic validation dataset [Data set]. Zenodo.
  https://doi.org/10.5281/zenodo.20786673
- Zenodo record: https://zenodo.org/records/20786673

Generation uses `ds003171`, `ds005620`, `ds002547`, `ds005479`, `ds004295`,
and `ds002336`.

## Source Snapshot Setup

Download complete public snapshots first and place them under one parent
folder:

```text
$IMPACT_SOURCE_ROOT/
  ds003171/
  ds005620/
  ds002547/
  ds005479/
  ds004295/
  ds002336/
```

One reproducible download sequence is:

```bash
export IMPACT_SOURCE_ROOT=/absolute/path/to/openneuro_sources

bash scripts/download_data.sh ds003171 2.0.1 "$IMPACT_SOURCE_ROOT/ds003171"
bash scripts/download_data.sh ds005620 1.0.0 "$IMPACT_SOURCE_ROOT/ds005620"
bash scripts/download_data.sh ds002547 1.1.0 "$IMPACT_SOURCE_ROOT/ds002547"
bash scripts/download_data.sh ds005479 1.1.1 "$IMPACT_SOURCE_ROOT/ds005479"
bash scripts/download_data.sh ds004295 1.0.0 "$IMPACT_SOURCE_ROOT/ds004295"
bash scripts/download_data.sh ds002336 2.0.2 "$IMPACT_SOURCE_ROOT/ds002336"
```

If the snapshots already exist elsewhere, set `IMPACT_SOURCE_ROOT` to the
parent directory or create symlinks under one source root.

## Source Requirements

The generator is source-specific. It expects complete, readable payloads for:

- `ds003171`: fMRI state/rest structure
- `ds005620`: BrainVision EEG wake/sedation structure
- `ds002547`: self/other events and fMRIPrep derivatives
- `ds005479`: MID reward/loss events
- `ds004295`: EEG reward/reversal-learning source metadata
- `ds002336`: multimodal source metadata

The source folders must contain materialized payload files. Required file
classes include `*_events.tsv`, JSON sidecars, fMRI `.nii`/`.nii.gz` files or
fMRIPrep derivatives, and BrainVision `.vhdr`, `.eeg`, and `.vmrk` files for EEG
sources.

Using another dataset requires dataset-specific subject discovery, event
mapping, signal extraction, baseline definition, and metric validation.

## Generation Basis

Generation is implemented in `scripts/generate_real_derived_synth_completed.py`.
Run source inspection before generation and use the same `IMPACT_SOURCE_ROOT`
for both commands.

The generated outputs are built from source-derived timing and signal summaries:

- fMRI signals are derived from inspected local NIfTI or fMRIPrep payloads.
- EEG signals are derived from inspected local BrainVision payloads.
- RAM event structure is based on reward timing and feedback information from
  `ds005479`, with EEG reward timing available from `ds004295`.
- SRPI event structure is based on self/other timing from `ds002547`.
- Event-locked synthetic components are added so RAM, PDI, NAS, IIM, SRPI, and
  CI can be computed during pipeline validation.

Every generated dataset root contains `dataset_description.json`,
`README_SYNTHETIC.md`, `participants.tsv`, BIDS-like event files, and
`manifest.json`.

## Running Generation

Use a local or external storage root for generated data. The default root is
`/Volumes/MPW_OT_AOY/impact-synergy-pipeline`; override it with
`IMPACT_SYNTH_ROOT`.

```bash
export IMPACT_SYNTH_ROOT=/absolute/path/to/impact-synergy-pipeline-data
export IMPACT_SOURCE_ROOT=/absolute/path/to/openneuro_sources

conda run -n impact-synergy-clean python scripts/inspect_real_sources_for_synth.py \
  --source-root "$IMPACT_SOURCE_ROOT" \
  --output-dir "$IMPACT_SYNTH_ROOT/test_objects/real_derived_synth_completed/reports"

conda run -n impact-synergy-clean python scripts/generate_real_derived_synth_completed.py \
  --source-root "$IMPACT_SOURCE_ROOT"

conda run -n impact-synergy-clean python scripts/generate_real_derived_synth_completed.py \
  --source-root "$IMPACT_SOURCE_ROOT" \
  --validate-only
```

## Validation

A generated dataset is usable only when the summary manifest reports
`all_validated: true` and the generated metric reports show finite positive RAM,
PDI, NAS, IIM, SRPI, and CI values for the intended validation rows.

For SRPI, the detailed components must also be positive:

- reactivity bias
- representational separability
- self-pattern stability
- internal-state coupling

The final criterion is successful metric computation.

## Limitations

The generated files are for software validation, not neuroscientific inference.
