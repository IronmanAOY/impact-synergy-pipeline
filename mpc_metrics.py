import numpy as np
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score
import logging

log = logging.getLogger(__name__)

def compute_RAM(ts, epsilon=1e-6):
    AC = np.var(ts)
    RT = 1.0 / (np.mean(np.std(ts, axis=1)) + epsilon)
    return AC / (RT + epsilon)

def compute_PDI(ts, rng, bins=10):
    # Per-region observed entropy
    H_obs = np.mean([
        entropy(np.histogram(ts[r], bins=bins, density=True)[0] + 1e-12, base=2)
        for r in range(ts.shape[0])
    ])
    # Per-region baseline entropy with per-region shuffle
    H_base = np.mean([
        entropy(np.histogram(rng.permutation(ts[r]), bins=bins, density=True)[0] + 1e-12, base=2)
        for r in range(ts.shape[0])
    ])
    H_max = np.log2(bins)
    val = (H_obs - H_base) / (H_max - H_base + 1e-12)
    if val < 0 or val > 1:
        log.warning(f"PDI {val:.3f} out of [0,1]; clamped")
        val = max(0.0, min(val, 1.0))
    return val

def compute_NAS(ts):
    corr = np.corrcoef(ts.T)
    iu = np.triu_indices(corr.shape[0], k=1)
    return np.mean(np.abs(corr[iu]))

def compute_IIM(ts):
    disc = np.digitize(ts, np.histogram_bin_edges(ts, bins=5))
    joint = mutual_info_score(disc.flatten()[:-1], disc.flatten()[1:])
    mid = disc.shape[1] // 2
    I_A = mutual_info_score(disc[:, :mid].flatten()[:-1], disc[:, :mid].flatten()[1:])
    I_B = mutual_info_score(disc[:, mid:].flatten()[:-1], disc[:, mid:].flatten()[1:])
    val = joint - (I_A + I_B)
    if val < 0:
        log.warning(f"IIM raw {val:.2f} below 0; clamped to 0.0")
        return 0.0
    if val > 1.0:
        log.warning(f"IIM raw {val:.2f} > 1; clamped to 1.0")
        return 1.0
    return val

def compute_SRPI(ts):
    half = ts.shape[1] // 2
    Rself = np.var(ts[:, :half])
    Rnon = np.var(ts[:, half:])
    num = Rself - Rnon; den = Rself + Rnon + 1e-12
    val = (num / den + 1) / 2
    if val < 0 or val > 1:
        log.warning(f"SRPI {val:.3f} out of [0,1]; clamped")
        val = max(0.0, min(val, 1.0))
    return val
