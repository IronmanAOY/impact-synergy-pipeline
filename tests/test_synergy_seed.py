import numpy as np
from synergy_ci import compute_synergy_ci

def test_seed_variation(tmp_path):
    base=tmp_path/"data"
    for subj in ("s1","s2"):
        d=base/subj/"awake"; d.mkdir(parents=True)
        ts=np.random.RandomState(0).rand(10,5)
        np.save(d/f"{subj}_awake_schaefer400_ts.npy",ts)
    df=compute_synergy_ci(str(base),"schaefer400",[0.5],sessions=("awake",))
    vals={r.subject:r.CI for _,r in df.iterrows()}
    assert vals["s1"]!=vals["s2"]
