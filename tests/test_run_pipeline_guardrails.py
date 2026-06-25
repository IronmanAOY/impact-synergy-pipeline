from pathlib import Path

import pytest

import run_pipeline


def test_cataloged_non_pipeline_dataset_rejects_raw_preprocessing(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(run_pipeline, "_assert_expected_runtime_env", lambda: None)

    with pytest.raises(ValueError, match="not .*pipeline-enabled"):
        run_pipeline.main(
            out_dir=tmp_path / "outputs",
            dataset_id="ds002547",
            run_preprocessing_flag=True,
            compute_ci=False,
        )
