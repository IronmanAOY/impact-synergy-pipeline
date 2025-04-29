import numpy as np
from utils import compute_midrank, fast_delong

def test_midrank():
    x=[1,2,2,3]
    mid=compute_midrank(x)
    assert len(mid)==4

def test_fast_delong():
    y_true=np.array([1,0,1,0])
    y_score=np.array([0.9,0.1,0.8,0.2])
    auc,var=fast_delong(y_true,y_score)
    assert 0<=auc<=1
