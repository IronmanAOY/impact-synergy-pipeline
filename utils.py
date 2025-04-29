import numpy as np
import networkx as nx
from scipy.stats import norm

class HypergraphSynergy:
    @staticmethod
    def compute(ts, theta):
        corr = np.corrcoef(ts.T)
        local = [(i,j) for i in range(corr.shape[0])
                 for j in range(i+1,corr.shape[0])
                 if abs(corr[i,j])>theta]
        adj = (abs(corr)>theta).astype(int)
        G = nx.from_numpy_array(adj)
        glob_ = [c for c in nx.connected_components(G) if len(c)>5]
        I = HypergraphSynergy._integration(ts, local)
        Bc = HypergraphSynergy._broadcast(glob_, ts.shape[1])
        Bal = HypergraphSynergy._balance(local, glob_)
        return I*Bc*Bal

    @staticmethod
    def _integration(ts, edges):
        phis = [np.var(ts[:,list(e)].mean(axis=1)) for e in edges] if edges else [0]
        return float(np.mean(phis))

    @staticmethod
    def _broadcast(glob_, n):
        sizes = [len(e) for e in glob_]
        return float(np.mean(sizes)/n) if sizes else 0.0

    @staticmethod
    def _balance(local, glob_):
        Wl, Wg = len(local), len(glob_)
        return 1 - abs(Wl-Wg)/(Wl+Wg) if (Wl+Wg)>0 else 0.0

def compute_midrank(x):
    sorted_idx = np.argsort(x)
    T = x[sorted_idx]
    n = len(x)
    mid = np.zeros(n)
    i = 0
    while i<n:
        j=i
        while j<n and T[j]==T[i]:
            j+=1
        for k in range(i,j):
            mid[k] = 0.5*(i+j-1)
        i=j
    ret = np.empty(n)
    ret[sorted_idx] = mid
    return ret

def fast_delong(y_true, y_score):
    pos = y_score[y_true==1]
    neg = y_score[y_true==0]
    m, n = len(pos), len(neg)
    preds = np.concatenate([pos, neg])
    lab = np.concatenate([np.ones(m), np.zeros(n)])
    mid = compute_midrank(preds)
    auc = (np.sum(mid[lab==1]) - m*(m+1)/2)/(m*n)
    v01 = (mid[lab==1] - np.arange(m)) / n
    v10 = (mid[lab==0] - np.arange(n)) / m
    var = np.var(v01, ddof=1)/m + np.var(v10, ddof=1)/n
    return auc, var

def delong_roc_test(y_true, y1, y2):
    auc1, var1 = fast_delong(y_true, y1)
    auc2, var2 = fast_delong(y_true, y2)
    delta = auc1 - auc2
    se = np.sqrt(var1+var2)
    z = delta / se
    p = 2*(1-norm.cdf(abs(z)))
    return float(p)
