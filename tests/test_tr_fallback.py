import numpy as np
import logging

from impact_pipeline import preprocessing


class DummyHeader:
    def get_zooms(self):
        return (1, 1, 20)

    def copy(self):
        return self


class DummyImg:
    def __init__(self):
        self.header = DummyHeader()
        self.shape = (2, 2, 2, 4)
        self.affine = np.eye(4)

    def get_fdata(self):
        return np.zeros(self.shape)


def dummy_clean(signals, **kwargs):
    assert "t_r" in kwargs and kwargs["t_r"] == 2.0
    return signals.T


class DummyMasker:
    def fit_transform(self, img):
        return np.zeros((4, 1))


def test_tr_fallback(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.WARNING)
    bids = tmp_path / "bids"
    fmriprep_deriv = tmp_path / "fmriprep"
    func = fmriprep_deriv / "sub-01" / "func"
    func.mkdir(parents=True)
    (func / "sub-01_task-audioawake_run-1_desc-preproc_bold.nii.gz").write_bytes(b"")

    monkeypatch.setattr(preprocessing.image, "load_img", lambda p: DummyImg())
    monkeypatch.setattr(preprocessing, "clean", dummy_clean)
    monkeypatch.setattr(preprocessing, "NiftiLabelsMasker", lambda **k: DummyMasker())
    monkeypatch.setattr(
        preprocessing,
        "get_atlas_globs",
        lambda: {"schaefer400": "dummy_atlas.nii.gz"},
    )

    bf = type("BF", (), {"entities": {"task": "audioawake", "run": 1}})()
    preprocessing.preprocess_subject(
        str(bids),
        str(fmriprep_deriv),
        "01",
        bf,
        str(tmp_path / "out"),
    )
    assert "Suspicious TR" in caplog.text
    assert "using 2.0s" in caplog.text
