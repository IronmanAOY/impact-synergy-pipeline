from sklearn.metrics import roc_auc_score
from utils import delong_roc_test

def compare_models(df, metrics=('mean_conn','modularity','pci_fmri')):
    y = (df.session == 'awake').astype(int).values
    auc0 = roc_auc_score(y, df.S)
    out = {}
    for m in metrics:
        auc_m = roc_auc_score(y, df[m])
        p = delong_roc_test(y, df.S.values, df[m].values)
        out[m] = {'delta_auc': auc0 - auc_m, 'p_val': p}
    return out
