import numpy as np
from impact_pipeline.synergy_ci import compute_synergy_ci

def test_seed_variation(tmp_path):
    base=tmp_path/"data"
    for i, subj in enumerate(("s1","s2")):
        d=base/subj/"awake"/"audio"; d.mkdir(parents=True)
        ts=np.random.RandomState(100 + i).rand(10,5)
        np.save(d/f"{subj}_run-1_schaefer400_ts.npy",ts)
    df=compute_synergy_ci(str(base),"schaefer400",[0.5],sessions=("awake",),compute_mpc=False)
    vals={r.subject:r.S for _,r in df.iterrows()}
    assert set(vals.keys()) == {"s1", "s2"}
    assert np.isfinite(vals["s1"])
    assert np.isfinite(vals["s2"])
