import os, numpy as np, pandas as pd
from mpc_metrics import compute_RAM, compute_PDI, compute_NAS, compute_IIM, compute_SRPI
from utils import HypergraphSynergy
import hashlib

def compute_synergy_ci(data_dir, atlas, thetas, sessions=('awake','sedation')):
    """
    Per-subject RNG seed = (base_seed + MD5(subject_session) ) % 2**32
    """
    recs = []
    base_seed = 42
    dirs = sorted(os.listdir(data_dir))
    if not dirs:
        return pd.DataFrame(columns=['subject','session','theta','S','CI'])
    for subj in dirs:
        for ses in sessions:
            key = f"{subj}_{ses}".encode()
            h = int(hashlib.md5(key).hexdigest()[:8], 16)
            seed = (base_seed + h) % (2**32)
            rng = np.random.RandomState(seed)

            ts_file = os.path.join(data_dir, subj, ses,
                                   f"{subj}_{ses}_{atlas}_ts.npy")
            if not os.path.exists(ts_file):
                raise FileNotFoundError(f"Missing TS file {ts_file}")
            ts = np.load(ts_file)

            ram = compute_RAM(ts)
            pdi = compute_PDI(ts, rng)
            nas = compute_NAS(ts)
            iim = compute_IIM(ts)
            srpi = compute_SRPI(ts)

            for theta in thetas:
                S = HypergraphSynergy.compute(ts, theta)
                prod = ram * pdi * nas * iim * srpi
                prod_clamped = max(0.0, min(prod, 1.0))
                CI = prod_clamped ** (1/5)
                recs.append({
                    'subject': subj, 'session': ses,
                    'theta': theta, 'S': S, 'CI': CI
                })
    return pd.DataFrame.from_records(recs)
