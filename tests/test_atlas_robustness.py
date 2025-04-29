import pandas as pd
import atlas_robustness

def test_atlas_robustness(monkeypatch):
    monkeypatch.setattr(atlas_robustness,"compute_synergy_ci",
                        lambda *a,**k: pd.DataFrame(columns=['session','theta','S']))
    res=atlas_robustness.atlas_check("dummy",["aal90"])
    assert isinstance(res,dict)
    assert 'aal90' in res
