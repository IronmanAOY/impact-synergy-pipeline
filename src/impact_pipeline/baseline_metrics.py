import os
import glob
import logging
import numpy as np
import pandas as pd
import networkx as nx

log = logging.getLogger(__name__)

def mean_conn(ts: np.ndarray) -> float:
    corr = np.corrcoef(ts.T)
    n = corr.shape[0]
    mask = ~np.eye(n, dtype=bool)
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

def _find_ts_files(base_dir, subject, session, atlas):
    """
    Search both directly under <base>/<sub>/<ses>/ and in the common 'rest' subfolder.
    Accept typical filename variants with/without 'task-rest'.
    """
    roots = [
        os.path.join(base_dir, subject, session, 'rest'),
        os.path.join(base_dir, subject, session),
    ]
    patterns = [
        f"{subject}_task-rest_run-*_{atlas}_ts.npy",
        f"{subject}_run-*_{atlas}_ts.npy",
        f"{subject}_*_rest_*_{atlas}_ts.npy",   # extra safety
        f"{subject}_*_{atlas}_ts.npy",          # fallback
    ]

    tried = []
    hits = []
    for root in roots:
        for pat in patterns:
            full = os.path.join(root, pat)
            tried.append(full)
            hits.extend(glob.glob(full))
        # final recursive fallback (use sparingly)
        rec = os.path.join(root, "**", f"{subject}_*_{atlas}_ts.npy")
        tried.append(rec)
        hits.extend(glob.glob(rec, recursive=True))

    hits = sorted(set(hits))
    return hits, tried

def compute_baseline_metrics(df, data_dir, atlas):
    rows = []
    for _, r in df.iterrows():
        candidates, tried = _find_ts_files(data_dir, r.subject, r.session, atlas)

        if not candidates:
            msg = (
                f"No TS file for {r.subject}/{r.session}.\n"
                "Looked in these patterns:\n  - " + "\n  - ".join(tried)
            )
            raise FileNotFoundError(msg)

        # Prefer files found under 'rest' and the lowest run number
        def pref_key(p):
            in_rest = "/rest/" in p.replace("\\", "/")
            # extract run number if present
            import re
            m = re.search(r"run-(\d+)", p)
            run = int(m.group(1)) if m else 9999
            return (0 if in_rest else 1, run, p)

        chosen = sorted(candidates, key=pref_key)[0]
        ts = np.load(chosen)

        rows.append({
            **r.to_dict(),
            'ts_path': chosen,
            'mean_conn': mean_conn(ts),
            'modularity': modularity(ts),
            'pci_fmri': pci_fmri(ts)
        })
        log.info("Baseline TS for %s/%s → %s", r.subject, r.session, chosen)

    return pd.DataFrame(rows)

