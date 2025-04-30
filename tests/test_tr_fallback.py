import numpy as np
import logging

import preprocessing


class DummyHeader:
    def get_zooms(self):
        return (1, 1, 20)


class DummyImg:
    def __init__(self):
        self.header = DummyHeader()
        self.shape = (2, 2, 2, 4)

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
    f = bids / "sub-01" / "ses-01"
    f.mkdir(parents=True)
    (f / "sub-01_ses-01_desc-preproc_bold.nii.gz").write_bytes(b"")
    monkeypatch.setattr(
        preprocessing,
        "BIDSLayout",
        lambda root, validate: type(
            "L", (), {"get": lambda *a, **k: [type("O", (), {"path": str(f / "xxx")})]}
        )(),
    )
    monkeypatch.setattr(preprocessing.image, "load_img", lambda p: DummyImg())
    monkeypatch.setattr(preprocessing, "clean", dummy_clean)
    monkeypatch.setattr(preprocessing, "NiftiLabelsMasker", lambda **k: DummyMasker())
    preprocessing.preprocess_subject(
        str(bids), "sub-01", "ses-01", str(tmp_path / "out")
    )
    assert "falling back to 2.0s" in caplog.text
