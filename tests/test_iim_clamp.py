import numpy as np, logging
import mpc_metrics as mm

def test_iim_clamp(caplog,monkeypatch):
    caplog.set_level(logging.WARNING)
    monkeypatch.setattr(mm,"mutual_info_score",lambda *a,**k:2.0)
    ts=np.random.RandomState(0).rand(10,5)
    v=mm.compute_IIM(ts)
    assert v==0.0
    assert "clamped" in caplog.text
