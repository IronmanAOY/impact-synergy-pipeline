import os
import numpy as np
import networkx as nx
import logging

log = logging.getLogger(__name__)

def mean_conn(ts):
    corr = np.corrcoef(ts.T)
    iu = np.triu_indices(corr.shape[0], k=1)
    return np.mean(np.abs(corr[iu]))

def modularity(ts, threshold=0.2):
    corr = np.corrcoef(ts.T)
    adj = (np.abs(corr) > threshold).astype(int)
    G = nx.from_numpy_array(adj)
    try:
        import community as louvain
        part = louvain.best_partition(G)
        return louvain.modularity(part, G)
    except ImportError:
        log.warning("python-louvain not installed; modularity=0.0")
        return 0.0

def pci_fmri(ts):
    bin_ts = (ts >= np.median(ts, axis=0)).astype(int)
    patterns = set(map(tuple, bin_ts))
    return len(patterns) / (2**ts.shape[1])

def compute_baseline_metrics(df, data_dir, atlas):
    import pandas as pd
    rows = []
    for _, r in df.iterrows():
        fn = os.path.join(data_dir, r.subject, r.session,
                          f"{r.subject}_{r.session}_{atlas}_ts.npy")
        ts = np.load(fn)
        rows.append({
            **r.to_dict(),
            'mean_conn': mean_conn(ts),
            'modularity': modularity(ts),
            'pci_fmri': pci_fmri(ts)
        })
    return pd.DataFrame(rows)
