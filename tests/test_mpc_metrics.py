import numpy as np
import mpc_metrics as mm

def test_ram():
    ts=np.ones((5,10))
    assert mm.compute_RAM(ts)>=0

def test_nas():
    ts=np.random.RandomState(0).rand(5,5)
    assert 0<=mm.compute_NAS(ts)<=1
