import pandas as pd
from synergy_ci import compute_synergy_ci

def test_empty(tmp_path):
    df=compute_synergy_ci(str(tmp_path),"schaefer400",[0.5],sessions=("awake",))
    assert isinstance(df,pd.DataFrame) and df.empty
