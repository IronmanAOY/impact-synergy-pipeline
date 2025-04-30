import numpy as np
import pytest
from simulation import simulate_bold

@pytest.mark.parametrize("N,T,dt", [
    (5, 10.0, 0.5),
    (10, 5.0, 1.0)
])
def test_smoke_simulation(N, T, dt):
    C = np.eye(N)
    params = {
        'G': 0.0, 'E0': 0.34,
        'tau_s': 1.0, 'tau_f': 1.0, 'tau_0': 1.0,
        'alpha': 0.32, 'sigma': 0.01,
        'k1': 7*0.34, 'k2': 2.0, 'k3': 2*0.34 - 0.2, 'V0': 0.02
    }
    bold, times = simulate_bold(N, T, dt, C, params)
    assert bold.shape == (len(times), N)
    assert not np.any(np.isnan(bold))
    assert not np.any(np.isinf(bold))
