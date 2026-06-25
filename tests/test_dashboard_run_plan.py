import json

from pathlib import Path

import pandas as pd


def _write_minimal_bids_root(root: Path, name: str) -> None:
    (root / "sub-01" / "func").mkdir(parents=True, exist_ok=True)
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": name}),
        encoding="utf-8",
    )


def test_preview_run_plan_groups_metrics_by_dataset(tmp_path, monkeypatch):
    import scripts.live_dashboard as dash

    monkeypatch.setattr(dash, "REPO_ROOT", tmp_path)
    cfg = dash.DashboardConfig(
        out_dir=tmp_path / "outputs" / "scratch",
        dataset_id="ds_primary",
        data_origin="real",
        subject_filter=None,
        refresh_sec=2.0,
        history_points=120,
    )
    state = dash.DashboardState(cfg)

    real_root = tmp_path / "data" / "managed" / "ds_primary"
    dummy_root = tmp_path / "test_objects" / "datasets" / "dummy_aux"
    _write_minimal_bids_root(real_root, "Primary")
    _write_minimal_bids_root(dummy_root, "Dummy")

    state._dataset_registry = {
        "ds_primary": {
            "bids_root": str(real_root),
            "data_origin": "real",
            "modality_profile": "fmri",
        },
        "dummy_aux": {
            "bids_root": str(dummy_root),
            "data_origin": "dummy",
            "modality_profile": "fmri",
        },
    }

    plan = state.preview_run_plan(
        {
            "selected_datasets": [
                {
                    "dataset_id": "ds_primary",
                    "bids_root": str(real_root),
                    "out_dir": str(tmp_path / "outputs" / "scratch"),
                    "data_origin": "real",
                    "modality_profile": "fmri",
                },
                {
                    "dataset_id": "dummy_aux",
                    "bids_root": str(dummy_root),
                    "out_dir": str(tmp_path / "outputs" / "scratch"),
                    "data_origin": "dummy",
                    "modality_profile": "fmri",
                },
            ],
            "primary_dataset_id": "ds_primary",
            "mpc_metrics": ["RAM", "PDI", "NAS", "IIM", "SRPI"],
            "metric_dataset_map": {
                "RAM": "ds_primary",
                "PDI": "ds_primary",
                "NAS": "ds_primary",
                "IIM": "dummy_aux",
                "SRPI": "dummy_aux",
            },
            "sessions": ["awake", "deep"],
            "condition": "audio",
            "atlas": "schaefer400",
            "execution_mode": "local",
            "hardware_target": "cpu",
            "run_preprocessing": True,
            "run_fmriprep": False,
            "run_replication": False,
            "reuse_step2": False,
            "no_ci": False,
        }
    )

    assert plan["ci_requested"] is True
    assert plan["ci_allowed"] is True
    assert plan["ci_mode"] == "mixed_source"
    assert "scientifically fragile" in str(plan["ci_problem_note"])
    assert plan["subject_mapping_auto"]["dummy_aux"]["01"] == "01"
    assert len(plan["plan"]) == 2
    assert plan["plan"][0]["dataset_id"] == "ds_primary"
    assert plan["plan"][0]["hardware_target"] == "cpu"
    assert plan["plan"][0]["metrics"] == ["RAM", "PDI", "NAS"]
    assert plan["plan"][0]["compute_ci"] is False
    assert plan["plan"][1]["dataset_id"] == "dummy_aux"
    assert plan["plan"][1]["metrics"] == ["IIM", "SRPI"]
    assert plan["plan"][1]["compute_ci"] is False


def test_preview_run_plan_preserves_manual_subject_mapping(tmp_path, monkeypatch):
    import scripts.live_dashboard as dash

    monkeypatch.setattr(dash, "REPO_ROOT", tmp_path)
    cfg = dash.DashboardConfig(
        out_dir=tmp_path / "outputs" / "scratch",
        dataset_id="ds_primary",
        data_origin="real",
        subject_filter=None,
        refresh_sec=2.0,
        history_points=120,
    )
    state = dash.DashboardState(cfg)

    real_root = tmp_path / "data" / "managed" / "ds_primary"
    dummy_root = tmp_path / "test_objects" / "datasets" / "dummy_aux"
    _write_minimal_bids_root(real_root, "Primary")
    _write_minimal_bids_root(dummy_root, "Dummy")

    state._dataset_registry = {
        "ds_primary": {
            "bids_root": str(real_root),
            "data_origin": "real",
            "modality_profile": "fmri",
        },
        "dummy_aux": {
            "bids_root": str(dummy_root),
            "data_origin": "dummy",
            "modality_profile": "fmri",
        },
    }

    plan = state.preview_run_plan(
        {
            "selected_datasets": [
                {
                    "dataset_id": "ds_primary",
                    "bids_root": str(real_root),
                    "out_dir": str(tmp_path / "outputs" / "scratch"),
                    "data_origin": "real",
                    "modality_profile": "fmri",
                },
                {
                    "dataset_id": "dummy_aux",
                    "bids_root": str(dummy_root),
                    "out_dir": str(tmp_path / "outputs" / "scratch"),
                    "data_origin": "dummy",
                    "modality_profile": "fmri",
                },
            ],
            "primary_dataset_id": "ds_primary",
            "mpc_metrics": ["RAM", "PDI", "NAS", "IIM", "SRPI"],
            "metric_dataset_map": {
                "RAM": "ds_primary",
                "PDI": "ds_primary",
                "NAS": "ds_primary",
                "IIM": "dummy_aux",
                "SRPI": "dummy_aux",
            },
            "subject_mapping": {
                "dummy_aux": {
                    "01": None,
                }
            },
            "sessions": ["awake", "deep"],
            "condition": "audio",
            "atlas": "schaefer400",
            "execution_mode": "local",
            "no_ci": False,
        }
    )

    assert plan["subject_mapping_auto"]["dummy_aux"]["01"] == "01"
    assert plan["subject_mapping_effective"]["dummy_aux"]["01"] is None


def test_compose_mixed_source_ci_records_subject_metric_summary(tmp_path, monkeypatch):
    import scripts.live_dashboard as dash

    monkeypatch.setattr(dash, "REPO_ROOT", tmp_path)
    cfg = dash.DashboardConfig(
        out_dir=tmp_path / "outputs" / "scratch",
        dataset_id="ds_primary",
        data_origin="real",
        subject_filter=None,
        refresh_sec=2.0,
        history_points=120,
    )
    state = dash.DashboardState(cfg)

    primary_out = tmp_path / "outputs" / "scratch" / "ds_primary"
    aux_out = tmp_path / "test_objects" / "runs" / "dummy_aux"
    for out_dir, ram_shift in ((primary_out, 0.0), (aux_out, 0.1)):
        cache_dir = out_dir / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {
                    "subject": "01",
                    "session": "awake",
                    "theta": 0.5,
                    "S": 1.0,
                    "RAM": 1.0 + ram_shift,
                    "PDI": 1.1 + ram_shift,
                    "NAS": 1.2 + ram_shift,
                    "IIM": 1.3 + ram_shift,
                    "SRPI": 1.4 + ram_shift,
                    "IIM_defined": True,
                },
                {
                    "subject": "01",
                    "session": "deep",
                    "theta": 0.5,
                    "S": 0.8,
                    "RAM": 0.9 + ram_shift,
                    "PDI": 1.0 + ram_shift,
                    "NAS": 1.1 + ram_shift,
                    "IIM": 1.2 + ram_shift,
                    "SRPI": 1.3 + ram_shift,
                    "IIM_defined": True,
                },
            ]
        ).to_csv(cache_dir / "step2_df.csv", index=False)

    manifest = state._compose_mixed_source_ci(
        {
            "plan": [
                {
                    "dataset_id": "ds_primary",
                    "out_dir": str(primary_out),
                    "data_origin": "real",
                },
                {
                    "dataset_id": "dummy_aux",
                    "out_dir": str(aux_out),
                    "data_origin": "dummy",
                },
            ],
            "primary_dataset_id": "ds_primary",
            "anchor_dataset_id": "ds_primary",
            "metric_dataset_map": {
                "RAM": "ds_primary",
                "PDI": "ds_primary",
                "NAS": "ds_primary",
                "IIM": "dummy_aux",
                "SRPI": "dummy_aux",
            },
            "subject_mapping_effective": {
                "ds_primary": {"01": "01"},
                "dummy_aux": {"01": "01"},
            },
            "ci_problem_note": "Mixed-source CI is scientifically fragile.",
        }
    )

    summary = manifest["subject_metric_summary"]
    assert len(summary) == 1
    row = summary[0]
    assert row["anchor_subject"] == "01"
    assert row["RAM_source_dataset_id"] == "ds_primary"
    assert row["RAM_source_subject"] == "01"
    assert row["IIM_source_dataset_id"] == "dummy_aux"
    assert row["IIM_source_origin"] == "dummy"
    assert row["IIM_source_subject"] == "01"
    assert Path(manifest["step2_df_path"]).exists()
    assert Path(manifest["subject_metric_summary_path"]).exists()
    manifest_path = Path(manifest["out_dir"]) / "cache" / "mixed_source_manifest.json"
    assert manifest_path.exists()


def test_dataset_library_exposes_report_catalog_metadata(tmp_path, monkeypatch):
    import scripts.live_dashboard as dash

    monkeypatch.setattr(dash, "REPO_ROOT", tmp_path)
    cfg = dash.DashboardConfig(
        out_dir=tmp_path / "outputs" / "scratch",
        dataset_id="ds006623",
        data_origin="real",
        subject_filter=None,
        refresh_sec=2.0,
        history_points=120,
    )
    state = dash.DashboardState(cfg)

    root = tmp_path / "data" / "managed" / "ds006623"
    _write_minimal_bids_root(root, "Mirror")
    state._dataset_registry = {
        "ds006623": {
            "bids_root": str(root),
            "data_origin": "real",
            "modality_profile": "fmri",
        }
    }

    lib = {row["dataset_id"]: row for row in state.dataset_library()}
    rec = lib["ds006623"]
    assert rec["catalog_role"] == "state_switch_anchor_fmri"
    assert rec["pipeline_ready"] is False
    assert rec["target_metrics"] == ["PDI", "NAS", "IIM"]


def test_dataset_library_prefers_catalog_annex_root_without_alias_entry(tmp_path, monkeypatch):
    import scripts.live_dashboard as dash

    monkeypatch.setattr(dash, "REPO_ROOT", tmp_path)
    cfg = dash.DashboardConfig(
        out_dir=tmp_path / "outputs" / "scratch",
        dataset_id="ds005620",
        data_origin="real",
        subject_filter=None,
        refresh_sec=2.0,
        history_points=120,
    )
    state = dash.DashboardState(cfg)

    plain_root = tmp_path / "data" / "scratch" / "ds005620"
    annex_root = tmp_path / "data" / "scratch" / "ds005620_annex"
    _write_minimal_bids_root(plain_root, "Plain")
    _write_minimal_bids_root(annex_root, "Annex")

    lib = {row["dataset_id"]: row for row in state.dataset_library()}
    assert "ds005620_annex" not in lib
    assert lib["ds005620"]["bids_root"] == str(annex_root.resolve())
    assert state._detect_dataset_root_for_id("ds005620") == annex_root.resolve()


def test_guardrails_block_preprocessing_for_catalog_only_dataset(tmp_path, monkeypatch):
    import scripts.live_dashboard as dash

    monkeypatch.setattr(dash, "REPO_ROOT", tmp_path)
    cfg = dash.DashboardConfig(
        out_dir=tmp_path / "outputs" / "scratch",
        dataset_id="ds006623",
        data_origin="real",
        subject_filter=None,
        refresh_sec=2.0,
        history_points=120,
    )
    state = dash.DashboardState(cfg)

    resolved = {
        "dataset_id": "ds006623",
        "data_origin": "real",
        "out_dir": str(tmp_path / "outputs" / "scratch"),
        "run_preprocessing": True,
        "run_fmriprep": False,
        "no_ci": False,
        "mpc_metrics": ["PDI", "NAS", "IIM"],
    }

    try:
        state._apply_run_guardrails(resolved)
    except ValueError as exc:
        assert "dataset-specific preprocessing/session mappings" in str(exc)
    else:
        raise AssertionError("Expected preprocessing guardrail to reject catalog-only dataset.")
