import numpy as np
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_regression
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


def compute_IIM(ts, bins=10):
    # 0) ensure rows=regions, cols=time
    if ts.shape[0] > ts.shape[1]:
        ts = ts.T

    # 1) discretize
    edges = np.histogram_bin_edges(ts, bins=bins)
    disc  = np.digitize(ts, edges)

    T = disc.shape[1]
    # 2) build patterns
    patterns_t   = [tuple(disc[:,   i])   for i in range(T - 1)]
    patterns_tp1 = [tuple(disc[:, i+1]) for i in range(T - 1)]

    # 3a) OBJECT‐array trick
    obj_t   = np.empty(len(patterns_t),   dtype=object); obj_t[:]   = patterns_t
    obj_tp1 = np.empty(len(patterns_tp1), dtype=object); obj_tp1[:] = patterns_tp1
    _, labels_t   = np.unique(obj_t,   return_inverse=True)
    _, labels_tp1 = np.unique(obj_tp1, return_inverse=True)

    # 4) mutual info
    I_joint = mutual_info_score(labels_t, labels_tp1)

    # 5) split A/B and repeat above
    half = disc.shape[0] // 2
    def get_labels(slice_):
        pats   = [tuple(slice_[:, i])   for i in range(T - 1)]
        pats1  = [tuple(slice_[:, i+1]) for i in range(T - 1)]
        o      = np.empty(len(pats), dtype=object)
        o[:]   = pats
        o1     = np.empty(len(pats1), dtype=object)
        o1[:]  = pats1
        return np.unique(o, return_inverse=True)[1], np.unique(o1, return_inverse=True)[1]

    labels_A_t,   labels_A_tp1 = get_labels(disc[:half, :])
    labels_B_t,   labels_B_tp1 = get_labels(disc[half:, :])

    I_A = mutual_info_score(labels_A_t,   labels_A_tp1)
    I_B = mutual_info_score(labels_B_t,   labels_B_tp1)

    raw = I_joint - (I_A + I_B)
    return max(raw, 0.0)


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
