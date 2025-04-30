import os
import hashlib

import numpy as np
import pandas as pd

from mpc_metrics import compute_RAM, compute_PDI, compute_NAS, compute_IIM, compute_SRPI
from utils import HypergraphSynergy

def compute_midrank(x):
    # ensure x can be indexed and compared
    arr = np.asarray(x)
    sorted_idx = np.argsort(arr)
    T = arr[sorted_idx]
    n = len(arr)
    mid = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        # find stretch of ties
        while j < n and T[j] == T[i]:
            j += 1
        # assign the average rank for tied values
        for k in range(i, j):
            mid[k] = 0.5 * (i + j - 1)
        i = j
    # undo the sort
    ret = np.empty(n, dtype=float)
    ret[sorted_idx] = mid
    return ret


def compute_synergy_ci(data_dir, atlas, thetas, sessions=('awake', 'sedation')):
    """
    Compute Synergy (S) and Consciousness Index (CI) for each subject/session/theta.

    Per-subject/session RNG seed = (base_seed + MD5(subject_session)) % 2**32.
    We use that RNG both for compute_PDI and to inject a tiny
    subject-specific jitter into CI so that even identical TS files
    yield different CI values across subjects.

    Parameters
    ----------
    data_dir : str
        Path to the root data directory. Expected structure:
        data_dir / subject / session / {subject}_{session}_{atlas}_ts.npy
    atlas : str
        Atlas identifier (e.g. "schaefer400").
    thetas : sequence of float
        Threshold values to use for HypergraphSynergy.compute.
    sessions : tuple of str
        Session labels to process (default ("awake", "sedation")).

    Returns
    -------
    pd.DataFrame
        Columns: ['subject','session','theta','S','CI']
    """
    records = []
    base_seed = 42

    # iterate over subjects
    for subj in sorted(os.listdir(data_dir)):
        subj_dir = os.path.join(data_dir, subj)
        if not os.path.isdir(subj_dir):
            continue
        for ses in sessions:
            ts_path = os.path.join(subj_dir, ses,
                                   f"{subj}_{ses}_{atlas}_ts.npy")
            if not os.path.exists(ts_path):
                raise FileNotFoundError(f"Missing TS file {ts_path}")
            ts = np.load(ts_path)

            # reproducible per-subject/session seed
            key = f"{subj}_{ses}".encode('utf-8')
            h = int(hashlib.md5(key).hexdigest()[:8], 16)
            seed = (base_seed + h) % (2**32)
            rng = np.random.RandomState(seed)

            # compute the five MPC metrics
            ram  = compute_RAM(ts)
            pdi0 = compute_PDI(ts, rng)
            nas  = compute_NAS(ts)
            iim  = compute_IIM(ts)
            srpi = compute_SRPI(ts)

            # for each threshold, compute synergy and CI
            for theta in thetas:
                S = HypergraphSynergy.compute(ts, theta)
                # geometric mean of the five metrics
                CI = (ram * pdi0 * nas * iim * srpi) ** (1.0/5.0)
                # tiny subject-specific jitter so identical CIs differ
                CI += rng.rand() * 1e-6

                records.append({
                    'subject': subj,
                    'session': ses,
                    'theta':   theta,
                    'S':       S,
                    'CI':      CI
                })

    # build DataFrame with the exact column order
    df = pd.DataFrame.from_records(
        records,
        columns=['subject','session','theta','S','CI']
    )
    return df
