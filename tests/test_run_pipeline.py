import json

import pandas as pd

import run_pipeline


def test_smoke_reuse_step2(tmp_path, monkeypatch):
    out_dir = tmp_path / "outputs"
    cache_dir = out_dir / "cache"
    prep_dir = out_dir / "preprocessed" / "sub-01" / "awake" / "audio"
    bids_dir = tmp_path / "bids"
    func_dir = bids_dir / "sub-01" / "func"

    cache_dir.mkdir(parents=True)
    prep_dir.mkdir(parents=True)
    func_dir.mkdir(parents=True)

    # Minimal fMRI sidecar required by step-6 TR lookup.
    (func_dir / "sub-01_task-audioawake_run-01_bold.json").write_text(
        json.dumps({"RepetitionTime": 2.0}),
        encoding="utf-8",
    )

    # Cached step-2 tables.
    pd.DataFrame(
        [
            {"subject": "01", "session": "awake", "theta": 0.6, "S": 0.2, "PDI": 0.1, "NAS": 0.3, "IIM": 0.9},
            {"subject": "01", "session": "deep", "theta": 0.6, "S": 0.1, "PDI": 0.05, "NAS": 0.31, "IIM": 0.92},
        ]
    ).to_csv(cache_dir / "step2_df.csv", index=False)
    pd.DataFrame(
        [
            {"subject": "01", "session": "awake", "S": 0.2},
            {"subject": "01", "session": "deep", "S": 0.1},
        ]
    ).to_csv(cache_dir / "step2_df_mean.csv", index=False)
    pd.DataFrame([{"theta": 0.6, "t_S": 1.0, "p_S": 0.3, "d_S": 0.5}]).to_csv(
        cache_dir / "step2_theta_stats.csv", index=False
    )

    monkeypatch.setattr(
        run_pipeline,
        "compute_baseline_metrics",
        lambda *a, **k: pd.DataFrame(
            [{"subject": "01", "session": "awake", "S": 0.2, "mean_conn": 0.1, "modularity": 0.2, "pci_fmri": 0.3},
             {"subject": "01", "session": "deep", "S": 0.1, "mean_conn": 0.1, "modularity": 0.2, "pci_fmri": 0.3}]
        ),
    )
    monkeypatch.setattr(run_pipeline, "bootstrap_ci", lambda *a, **k: (0.1, 0.9))
    monkeypatch.setattr(run_pipeline, "permutation_test_auc", lambda *a, **k: (0.6, 0.2))
    monkeypatch.setattr(run_pipeline, "motion_covariate_analysis", lambda *a, **k: {"coef_awake": [0.0], "p_awake": [1.0]})
    monkeypatch.setattr(run_pipeline, "atlas_check", lambda *a, **k: {"aal90": {"awake": 0.1, "deep": 0.1}})
    monkeypatch.setattr(run_pipeline, "compare_models", lambda *a, **k: {"mean_conn": {"delta_auc": 0.0, "p_val": 1.0}})
    monkeypatch.setattr(run_pipeline, "create_doc", lambda *a, **k: None)

    run_pipeline.main(
        str(out_dir),
        dataset_id="ds003171",
        bids_root_override=str(bids_dir),
        reuse_step2=True,
        mpc_metrics=["PDI", "NAS", "IIM"],
        compute_ci=False,
    )


def test_smoke_reuse_step2_ds005620(tmp_path, monkeypatch):
    out_dir = tmp_path / "outputs"
    cache_dir = (out_dir / "ds005620" / "cache")
    prep_dir = out_dir / "ds005620" / "preprocessed" / "sub-1010" / "awake" / "eeg"
    bids_dir = tmp_path / "bids_ds005620"

    cache_dir.mkdir(parents=True)
    prep_dir.mkdir(parents=True)
    (bids_dir / "sub-1010" / "eeg").mkdir(parents=True)

    pd.DataFrame(
        [
            {"subject": "1010", "session": "awake", "theta": 0.6, "S": 0.2, "PDI": 0.1, "NAS": 0.3, "IIM": 0.9},
            {"subject": "1010", "session": "deep", "theta": 0.6, "S": 0.1, "PDI": 0.05, "NAS": 0.31, "IIM": 0.92},
        ]
    ).to_csv(cache_dir / "step2_df.csv", index=False)
    pd.DataFrame(
        [
            {"subject": "1010", "session": "awake", "S": 0.2},
            {"subject": "1010", "session": "deep", "S": 0.1},
        ]
    ).to_csv(cache_dir / "step2_df_mean.csv", index=False)
    pd.DataFrame([{"theta": 0.6, "t_S": 1.0, "p_S": 0.3, "d_S": 0.5}]).to_csv(
        cache_dir / "step2_theta_stats.csv", index=False
    )

    monkeypatch.setattr(
        run_pipeline,
        "compute_baseline_metrics",
        lambda *a, **k: pd.DataFrame(
            [{"subject": "1010", "session": "awake", "S": 0.2, "mean_conn": 0.1, "modularity": 0.2, "pci_fmri": 0.3},
             {"subject": "1010", "session": "deep", "S": 0.1, "mean_conn": 0.1, "modularity": 0.2, "pci_fmri": 0.3}]
        ),
    )
    monkeypatch.setattr(run_pipeline, "bootstrap_ci", lambda *a, **k: (0.1, 0.9))
    monkeypatch.setattr(run_pipeline, "permutation_test_auc", lambda *a, **k: (0.6, 0.2))
    monkeypatch.setattr(run_pipeline, "compare_models", lambda *a, **k: {"mean_conn": {"delta_auc": 0.0, "p_val": 1.0}})
    monkeypatch.setattr(run_pipeline, "create_doc", lambda *a, **k: None)

    run_pipeline.main(
        str(out_dir),
        dataset_id="ds005620",
        bids_root_override=str(bids_dir),
        reuse_step2=True,
        mpc_metrics=["PDI", "NAS", "IIM"],
        compute_ci=False,
    )
