import numpy as np
from sklearn.utils import resample  # noqa: F401
from sklearn.metrics import roc_auc_score
import warnings


def bootstrap_ci(
    df,
    score_col,
    session_col="session",
    sessions=("awake", "deep"),
    n_boot=1000,
    random_state=None,
):
    # map sessions to binary
    y = (df[session_col] == sessions[1]).astype(int).to_numpy()
    scores = df[score_col].to_numpy()
    if np.unique(y).size < 2:
        warnings.warn(
            f"bootstrap_ci({score_col}): session labels contain <2 classes for sessions={sessions}; returning NaN CI.",
            RuntimeWarning,
        )
        return np.nan, np.nan
    rng = np.random.RandomState(random_state)
    aucs = []
    attempts = 0
    max_attempts = max(10 * n_boot, 1000)
    while len(aucs) < n_boot and attempts < max_attempts:
        attempts += 1
        idx = rng.choice(len(df), size=len(df), replace=True)
        try:
            aucs.append(roc_auc_score(y[idx], scores[idx]))
        except ValueError:
            continue
    if len(aucs) == 0:
        warnings.warn(
            f"bootstrap_ci({score_col}): no valid bootstrap resamples produced; returning NaN CI.",
            RuntimeWarning,
        )
        return np.nan, np.nan
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return lo, hi


def permutation_test_auc(
    df,
    score_col,
    session_col="session",
    sessions=("awake", "deep"),
    n_perm=10000,
    random_state=None,
):
    y = (df[session_col] == sessions[1]).astype(int).to_numpy()
    scores = df[score_col].to_numpy()
    if np.unique(y).size < 2:
        warnings.warn(
            f"permutation_test_auc({score_col}): session labels contain <2 classes for sessions={sessions}; returning NaNs.",
            RuntimeWarning,
        )
        return np.nan, np.nan
    obs_auc = roc_auc_score(y, scores)
    rng = np.random.RandomState(random_state)
    greater = 0
    for _ in range(n_perm):
        perm = rng.permutation(y)
        if roc_auc_score(perm, scores) >= obs_auc:
            greater += 1
    p_val = (greater + 1) / (n_perm + 1)
    return obs_auc, p_val
