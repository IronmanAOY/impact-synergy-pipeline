# Real-Data-Derived Synthetic Datasets

This repository supports compact synthetic datasets for IMPaCT pipeline
validation. They are not real patient data.

## Targets

- `ds003171`: fMRI state-switch structure
- `ds002547`: fMRI self/other structure
- `ds005620`: EEG sedation/wake structure

## Setup Paths

The open archive is available at:

```text
https://doi.org/10.5281/zenodo.20786673
```

It contains:

- `impact-synergy-real-derived-synthetic-test-objects.tar.gz`
- `release_manifest.json`
- `SHA256SUMS.txt`

After downloading it, set `IMPACT_SYNTH_ROOT` to the directory that will contain
`test_objects/`, then verify, extract, and validate:

```bash
export IMPACT_SYNTH_ROOT=/absolute/path/for/synthetic_package

mkdir -p "$IMPACT_SYNTH_ROOT"
shasum -a 256 -c SHA256SUMS.txt
tar -xzf impact-synergy-real-derived-synthetic-test-objects.tar.gz -C "$IMPACT_SYNTH_ROOT"
conda run -n impact-synergy-clean python scripts/generate_real_derived_synth_completed.py \
  --validate-only
```

To regenerate the datasets instead, download complete OpenNeuro snapshots into a
single source root:

```text
$IMPACT_SOURCE_ROOT/
  ds003171/
  ds005620/
  ds002547/
  ds005479/
  ds004295/
  ds002336/
```

One download sequence is:

```bash
export IMPACT_SOURCE_ROOT=/absolute/path/to/openneuro_sources

bash scripts/download_data.sh ds003171 2.0.1 "$IMPACT_SOURCE_ROOT/ds003171"
bash scripts/download_data.sh ds005620 1.0.0 "$IMPACT_SOURCE_ROOT/ds005620"
bash scripts/download_data.sh ds002547 1.1.0 "$IMPACT_SOURCE_ROOT/ds002547"
bash scripts/download_data.sh ds005479 1.1.1 "$IMPACT_SOURCE_ROOT/ds005479"
bash scripts/download_data.sh ds004295 1.0.0 "$IMPACT_SOURCE_ROOT/ds004295"
bash scripts/download_data.sh ds002336 2.0.2 "$IMPACT_SOURCE_ROOT/ds002336"
```

Then inspect sources, generate datasets, and validate:

```bash
export IMPACT_SYNTH_ROOT=/absolute/path/for/generated_outputs

conda run -n impact-synergy-clean python scripts/inspect_real_sources_for_synth.py \
  --source-root "$IMPACT_SOURCE_ROOT" \
  --output-dir "$IMPACT_SYNTH_ROOT/test_objects/real_derived_synth_completed/reports"

conda run -n impact-synergy-clean python scripts/generate_real_derived_synth_completed.py \
  --source-root "$IMPACT_SOURCE_ROOT"

conda run -n impact-synergy-clean python scripts/generate_real_derived_synth_completed.py \
  --validate-only
```

If `IMPACT_SYNTH_ROOT` is omitted, generated outputs are written under this
repository's `test_objects/` tree.

## Source Requirements

The generator is source-specific. It expects complete, readable payloads for:

- `ds003171`: fMRI state/rest structure
- `ds005620`: BrainVision EEG wake/sedation structure
- `ds002547`: self/other events and fMRIPrep derivatives
- `ds005479`: MID reward/loss events
- `ds004295`: EEG reward/reversal-learning source metadata
- `ds002336`: multimodal source metadata

The source folders must contain materialized payload files: `*_events.tsv`, JSON
sidecars, fMRI `.nii`/`.nii.gz` files or fMRIPrep derivatives, and BrainVision
`.vhdr`, `.eeg`, and `.vmrk` files for EEG sources.

Using another source dataset requires dataset-specific subject discovery, event
mapping, signal extraction, baseline definition, and metric validation.

## Validation

A synthetic package is usable when the summary manifest reports
`all_validated: true` and the metric reports show finite positive RAM, PDI, NAS,
IIM, SRPI, and CI values for the intended rows.

For SRPI, the detailed components must also be positive:

- reactivity bias
- representational separability
- self-pattern stability
- internal-state coupling
