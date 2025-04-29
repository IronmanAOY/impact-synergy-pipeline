import numpy as np
import pandas as pd
from synergy_ci import compute_synergy_ci

def test_synergy_ci_shape(tmp_path):
    base=tmp_path/"data"
    for subj in ("s1",):
        d=base/subj/"awake"
        d.mkdir(parents=True)
        np.save(d/"s1_awake_schaefer400_ts.npy",np.random.rand(5,3))
    df=compute_synergy_ci(str(base),"schaefer400",[0.5],sessions=("awake",))
    assert isinstance(df,pd.DataFrame)
    assert list(df.columns)==['subject','session','theta','S','CI']

def test_empty_data_dir(tmp_path):
    df=compute_synergy_ci(str(tmp_path),"schaefer400",[0.5],sessions=("awake",))
    assert df.empty
