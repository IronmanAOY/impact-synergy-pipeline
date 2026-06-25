import numpy as np
import logging
from sklearn.metrics import roc_auc_score
from impact_pipeline.utils import delong_roc_test

log = logging.getLogger(__name__)

def compare_models(df,metrics=('mean_conn','modularity','pci_fmri'), sessions=('awake','deep')):
    """
    Compare each metric against S by computing ΔAUC = AUC_S – AUC_metric,
    and a Delong test p‐value.
    """
    neg, pos = sessions
    session_vals = df.session.astype(str).to_numpy()
    uniq_sessions = set(session_vals.tolist())
    if pos not in uniq_sessions and neg in uniq_sessions and len(uniq_sessions) == 2:
        # Backward-compatible fallback for datasets that use a different
        # positive label (e.g., "sedation" instead of "deep").
        pos = next(s for s in uniq_sessions if s != neg)
        log.debug("compare_models: inferred positive session label '%s'", pos)

    y = (session_vals == pos).astype(int)
    if np.unique(y).size < 2:
        raise ValueError(
            f"compare_models requires two classes after session mapping; got labels={sorted(uniq_sessions)} "
            f"with sessions={sessions}."
        )
    
    unique, counts = np.unique(y, return_counts=True)
    log.debug("compare_models y counts: %s", dict(zip(unique, counts)))
    
    auc0 = roc_auc_score(y, df.S)
    out = {}
    for m in metrics:
        auc_m = roc_auc_score(y, df[m])
        p = delong_roc_test(y, df.S.values, df[m].values)
        out[m] = {'delta_auc': auc0 - auc_m, 'p_val': p}
    return out
