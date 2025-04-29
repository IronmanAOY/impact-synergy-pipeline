import numpy as np
from baseline_metrics import mean_conn,modularity,pci_fmri

def test_mean_conn():
    ts=np.eye(3)
    assert mean_conn(ts)==0

def test_modularity_no_louvain(monkeypatch):
    monkeypatch.setitem(__import__('sys').modules,'community',None)
    ts=np.random.rand(5,5)
    assert modularity(ts)>=0

def test_pci_fmri():
    ts=np.zeros((4,4))
    val=pci_fmri(ts)
    assert 0<=val<=1
