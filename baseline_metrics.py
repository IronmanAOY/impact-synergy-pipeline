import os
import numpy as np
import networkx as nx
import logging
import glob
import pandas as pd

log = logging.getLogger(__name__)

def mean_conn(ts: np.ndarray) -> float:
    """
    Compute the average off-diagonal Pearson correlation
    across all ROIs in a time-series matrix ts
    of shape (n_timepoints, n_rois).
    """
    # build correlation matrix between ROIs
    corr = np.corrcoef(ts.T)            # shape (n_rois, n_rois)
    n = corr.shape[0]
    mask = ~np.eye(n, dtype=bool)      # off-diagonal mask
    return corr[mask].mean()

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
    rows = []
    for _, r in df.iterrows():
        pattern = os.path.join(
            data_dir, r.subject, r.session,
            f"{r.subject}_run-*_{atlas}_ts.npy"
        )
        candidates = sorted(glob.glob(pattern))
        if not candidates:
            raise FileNotFoundError(f"No TS file for {r.subject}/{r.session} under {pattern}")
        ts = np.load(candidates[0])

        rows.append({
            **r.to_dict(),
            'mean_conn': mean_conn(ts),
            'modularity': modularity(ts),
            'pci_fmri': pci_fmri(ts)
        })
    return pd.DataFrame(rows)
