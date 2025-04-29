import numpy as np
import pandas as pd
from synergy_ci import compute_synergy_ci

def atlas_check(data_dir, atlases):
    out = {}
    for atlas in atlases:
        df = compute_synergy_ci(data_dir, atlas,
                                thetas=[i*0.1 for i in range(1,10)])
        if not isinstance(df, pd.DataFrame):
            raise TypeError("compute_synergy_ci must return DataFrame")
        rough = {}
        for ses in ['awake','sedation']:
            sub = df[df.session == ses]
            s = sub.sort_values('theta')['S'].values
            rough[ses] = np.sqrt(np.mean(np.diff(s)**2)) if len(s)>1 else np.nan
        out[atlas] = rough
    return out
