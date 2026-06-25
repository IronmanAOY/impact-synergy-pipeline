import importlib

import numpy as np
import pytest

from impact_pipeline.hardware_backend import (
    HardwareBackendError,
    backend_summary,
    resolve_hardware_backend,
)
from impact_pipeline import mpc_metrics as mm


def test_cpu_backend_is_default():
    backend = resolve_hardware_backend("cpu")
    assert backend.target == "cpu"
    assert backend.accelerator is False
    assert "cpu" in backend_summary(backend)


def test_auto_falls_back_to_cpu_when_cupy_missing(monkeypatch):
    real_import = importlib.import_module

    def fake_import(name, *args, **kwargs):
        if name == "cupy":
            raise ImportError("simulated missing cupy")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", fake_import)

    assert resolve_hardware_backend("auto").target == "cpu"
    with pytest.raises(HardwareBackendError):
        resolve_hardware_backend("gpu")
    with pytest.raises(HardwareBackendError):
        resolve_hardware_backend("hunter-apu")


def test_cpu_hardware_parameter_preserves_pdi_and_iim_values():
    rng = np.random.RandomState(7)
    ts = rng.normal(size=(4, 80))
    baseline = rng.normal(size=(4, 80))

    pdi_default = mm.compute_PDI(ts, baseline_ts=[baseline], normalize=False)
    pdi_cpu = mm.compute_PDI(ts, baseline_ts=[baseline], normalize=False, hardware_backend="cpu")
    assert pdi_cpu == pytest.approx(pdi_default, rel=1e-12, abs=1e-12)

    iim_default = mm.compute_IIM(
        ts,
        bins=2,
        n_parts=2,
        max_nodes=3,
        max_mechanism_size=2,
        max_purview_size=2,
        return_details=True,
        phase1_parallel_workers=None,
    )
    iim_cpu = mm.compute_IIM(
        ts,
        bins=2,
        n_parts=2,
        max_nodes=3,
        max_mechanism_size=2,
        max_purview_size=2,
        return_details=True,
        phase1_parallel_workers=None,
        hardware_backend="cpu",
    )
    assert iim_cpu["defined"] is True
    assert iim_cpu["canonical"] == pytest.approx(iim_default["canonical"], rel=1e-12, abs=1e-12)
