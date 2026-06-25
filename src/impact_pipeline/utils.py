import numpy as np
import networkx as nx
from scipy.stats import norm


class HypergraphSynergy:
    @staticmethod
    def compute(ts, theta):
        # ts is expected as (time, nodes)
        corr = np.corrcoef(ts, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        abs_corr = np.abs(corr)
        np.fill_diagonal(abs_corr, 0.0)

        adj_bool = abs_corr > theta
        edge_i, edge_j = np.where(np.triu(adj_bool, k=1))
        n_local = int(edge_i.size)

        adj = adj_bool.astype(int)
        G = nx.from_numpy_array(adj)
        glob_ = [c for c in nx.connected_components(G) if len(c) > 5]
        I_measure = HypergraphSynergy._integration(ts, edge_i, edge_j)
        Bc = HypergraphSynergy._broadcast(glob_, ts.shape[1])
        Bal = HypergraphSynergy._balance(n_local, len(glob_))
        return I_measure * Bc * Bal

    @staticmethod
    def _integration(ts, edge_i, edge_j):
        if edge_i.size == 0:
            return 0.0
        # Vectorized equivalent of mean over edge-wise var(mean(ts[:, i], ts[:, j])).
        pair_means = 0.5 * (ts[:, edge_i] + ts[:, edge_j])
        return float(np.var(pair_means, axis=0).mean())

    @staticmethod
    def _broadcast(glob_, n):
        sizes = [len(e) for e in glob_]
        return float(np.mean(sizes) / n) if sizes else 0.0

    @staticmethod
    def _balance(n_local, n_global):
        Wl, Wg = int(n_local), int(n_global)
        return 1 - abs(Wl - Wg) / (Wl + Wg) if (Wl + Wg) > 0 else 0.0


def compute_midrank(x):
    x = np.asarray(x, dtype=float)
    sorted_idx = np.argsort(x)
    T = x[sorted_idx]
    n = len(x)
    mid = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and T[j] == T[i]:
            j += 1
        mid[i:j] = 0.5 * (i + j - 1)
        i = j
    ret = np.empty(n, dtype=float)
    ret[sorted_idx] = mid
    return ret


def fast_delong(y_true, y_score):
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    m, n = len(pos), len(neg)
    preds = np.concatenate([pos, neg])
    lab = np.concatenate([np.ones(m), np.zeros(n)])
    mid = compute_midrank(preds)
    auc = (np.sum(mid[lab == 1]) - m * (m + 1) / 2) / (m * n)
    v01 = (mid[lab == 1] - np.arange(m)) / n
    v10 = (mid[lab == 0] - np.arange(n)) / m
    var = np.var(v01, ddof=1) / m + np.var(v10, ddof=1) / n
    return auc, var


def delong_roc_test(y_true, y1, y2):
    auc1, var1 = fast_delong(y_true, y1)
    auc2, var2 = fast_delong(y_true, y2)
    delta = auc1 - auc2
    se = np.sqrt(var1 + var2)
    z = delta / se
    p = 2 * (1 - norm.cdf(abs(z)))
    return float(p)
