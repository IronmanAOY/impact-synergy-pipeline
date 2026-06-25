# Synthetic Validation Data

This folder is reserved for synthetic validation datasets and outputs derived
from them.

Use it for:

- synthetic BIDS-style datasets under `test_objects/datasets/`
- pipeline outputs for synthetic validation runs under `test_objects/runs/`
- reusable validation metric exports under `test_objects/metric_bank/`

Rules:

- synthetic validation data and derived outputs stay here
- real study data and real-study outputs stay outside this folder
- metric-bank exports stored here are selected explicitly by downstream runs
- mixed-source CI analysis must preserve dataset-origin labels

The dashboard and pipeline enforce this separation when a dataset is marked as
synthetic. See `docs/synthetic_data.md` for real-data-derived synthetic dataset
generation.
