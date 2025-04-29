import numpy as np
from sklearn.utils import resample
from sklearn.metrics import roc_auc_score

def bootstrap_ci(df, metric='S', n_boot=1000, seed=42):
    y = (df.session == 'awake').astype(int).values
    scores = df[metric].values
    rng = np.random.RandomState(seed)
    aucs = []
    for _ in range(n_boot):
        idx = resample(np.arange(len(y)), replace=True, random_state=rng)
        aucs.append(roc_auc_score(y[idx], scores[idx]))
    return np.percentile(aucs, [2.5, 97.5])

def permutation_test_auc(df, metric='S', n_perm=10000, seed=42):
    y = (df.session == 'awake').astype(int).values
    scores = df[metric].values
    true_auc = roc_auc_score(y, scores)
    rng = np.random.RandomState(seed)
    perm = []
    for _ in range(n_perm):
        p = rng.permutation(y)
        perm.append(roc_auc_score(p, scores))
    perm = np.array(perm)
    count = np.sum(perm >= true_auc)
    p_val = (count + 1) / (n_perm + 1)
    return true_auc, p_val
