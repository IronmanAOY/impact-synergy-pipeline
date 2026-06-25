from pathlib import Path

from impact_pipeline.provenance import resolve_dataset_provenance


def test_synthetic_provenance_uses_validation_output_root(tmp_path):
    prov = resolve_dataset_provenance(
        repo_root=tmp_path,
        out_dir=tmp_path / "outputs" / "scratch",
        dataset_id="dummy_ds",
        data_origin="dummy",
    )
    assert prov.data_origin == "dummy"
    assert prov.dataset_role == "synthetic_validation"
    assert prov.effective_out_dir == (tmp_path / "test_objects" / "runs" / "dummy_ds").resolve()
    assert prov.metric_bank_dataset_dir == (tmp_path / "test_objects" / "metric_bank" / "dummy_ds").resolve()


def test_real_provenance_keeps_standard_output_layout(tmp_path):
    prov = resolve_dataset_provenance(
        repo_root=tmp_path,
        out_dir=tmp_path / "outputs" / "scratch",
        dataset_id="ds005620",
        data_origin="real",
    )
    assert prov.data_origin == "real"
    assert prov.dataset_role == "study_data"
    assert prov.effective_out_dir == (tmp_path / "outputs" / "scratch" / "ds005620").resolve()
    assert prov.metric_bank_dataset_dir is None
