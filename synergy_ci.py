import os
import hashlib
import glob

import numpy as np
import pandas as pd

from mpc_metrics import compute_RAM, compute_PDI, compute_NAS, compute_IIM, compute_SRPI
from utils import HypergraphSynergy


def compute_synergy_ci(data_dir, atlas, thetas, sessions=('awake','deep', 'recovery')):
    records = []
    base_seed = 42
    for subj in sorted(os.listdir(data_dir)):
        subj_dir = os.path.join(data_dir, subj)
        # skip hidden files and anything that isn't a directory
        if subj.startswith('.') or not os.path.isdir(subj_dir):
            continue

        for ses in sessions:
            # point at the directory you created in preprocessing:
            session_dir = os.path.join(data_dir, subj, ses)
            # match any run-<digits> for this atlas
            pattern = os.path.join(
                session_dir,
                f"{subj}_run-*_{atlas}_ts.npy"
            )
            candidates = sorted(glob.glob(pattern))
            if not candidates:
                raise FileNotFoundError(f"No time-series found for {subj}/{ses} (looking under {pattern})")
            # pick the first match
            ts_path = candidates[0]

            ts = np.load(ts_path)

            # reproducible per-subject/session seed
            key = f"{subj}_{ses}".encode('utf-8')
            h = int(hashlib.md5(key).hexdigest()[:8], 16)
            rng = np.random.RandomState((base_seed + h) % 2**32)

            # Compute MPC metrics
            ram  = compute_RAM(ts)
            pdi0 = compute_PDI(ts, rng)
            nas  = compute_NAS(ts)
            iim  = compute_IIM(ts)
            srpi = compute_SRPI(ts)

            for theta in thetas:
                S = HypergraphSynergy.compute(ts, theta)
                CI = (ram * pdi0 * nas * iim * srpi) ** (1/5) * 1e9
                # tiny jitter so identical CIs aren’t exactly equal
                CI += rng.rand() * 1e-6

                records.append({
                    'subject': subj,
                    'session': ses,
                    'theta':   theta,
                    'S':       S,
                    'CI':      CI
                })

    return pd.DataFrame(records, columns=['subject','session','theta','S','CI'])
