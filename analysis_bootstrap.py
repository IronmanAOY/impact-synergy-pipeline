import numpy as np
from sklearn.utils import resample
from sklearn.metrics import roc_auc_score

def bootstrap_ci(df, score_col, session_col='session', sessions=('awake','sedation'),
                 n_boot=1000, random_state=None):
    # map sessions to binary
    y = (df[session_col] == sessions[1]).astype(int).to_numpy()
    scores = df[score_col].to_numpy()
    rng = np.random.RandomState(random_state)
    aucs = []
    while len(aucs) < n_boot:
        idx = rng.choice(len(df), size=len(df), replace=True)
        try:
            aucs.append(roc_auc_score(y[idx], scores[idx]))
        except ValueError:
            continue
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return lo, hi

def permutation_test_auc(df, score_col, session_col='session', sessions=('awake','sedation'),
                         n_perm=10000, random_state=None):
    y = (df[session_col] == sessions[1]).astype(int).to_numpy()
    scores = df[score_col].to_numpy()
    obs_auc = roc_auc_score(y, scores)
    rng = np.random.RandomState(random_state)
    greater = 0
    for _ in range(n_perm):
        perm = rng.permutation(y)
        if roc_auc_score(perm, scores) >= obs_auc:
            greater += 1
    p_val = (greater + 1) / (n_perm + 1)
    return obs_auc, p_val
