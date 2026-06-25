# Real-Data-Derived Synthetic Datasets

This repository can generate compact synthetic datasets for IMPaCT pipeline
validation. These files are not empirical participant data. They are synthetic
validation data derived from inspected local OpenNeuro payload statistics,
timing structures, and event templates.

## Scope

The generated package targets:

- `ds003171`: fMRI state-switch structure
- `ds002547`: fMRI self/other structure
- `ds005620`: EEG sedation/wake structure

Generated dataset manifests record the source datasets used as structural or
statistical donors. The source set used by the generator is `ds003171`,
`ds005620`, `ds002547`, `ds005479`, `ds004295`, and `ds002336`.

## Generation Basis

Generation is implemented in `scripts/generate_real_derived_synth_completed.py`.
The generator reads source-inspection reports created from local OpenNeuro
payloads and stops if the required reports are missing.

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
conda run -n impact-synergy-clean python scripts/generate_real_derived_synth_completed.py
conda run -n impact-synergy-clean python scripts/generate_real_derived_synth_completed.py --validate-only
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

Readiness checks are useful for detecting missing files, but the final criterion
is successful metric computation.

## Limitations

The generated files are synthetic validation data. They should be used to test
pipeline behavior, file handling, metric orchestration, and CI computation.
They should not be used as evidence for neuroscientific inference.
