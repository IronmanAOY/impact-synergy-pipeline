import numpy as np
from scipy.integrate import odeint
from metrics import compute_synergy_metrics 
import os

def windkessel(state, t, u, tau_s, tau_f, tau_0, alpha, E0):
    """
    state: [s, f, v, q] for each node concatenated -> length 4N
    u: neural drive (length N)
    parameters scalar or length-N arrays
    """
    N = u.shape[0]
    s, f, v, q = np.split(state, 4)
    # Vasodilatory signal dynamics
    ds = u - s / tau_s - (f - 1) / tau_f
    # Blood inflow
    df = s
    # Volume change
    dv = (f - v ** (1/alpha)) / tau_0
    # Deoxyhemoglobin change
    dq = (f * (1 - (1 - E0) ** (1 / f)) / E0 - q * v ** (1/alpha - 1)) / tau_0
    return np.concatenate([ds, df, dv, dq])

def simulate_bold(N, T, dt, connectivity, params):
    """
    N: #nodes, T: total time (s), dt: step size
    connectivity: NxN structural matrix
    params: dict of model parameters
    """
    times = np.arange(0, T, dt)
    G = params['G']
    E0 = params['E0']
    state0 = np.zeros(4*N)
    bold = np.zeros((len(times), N))
    state = state0.copy()

    for i, t in enumerate(times):
        # neural input: coupling times last blood flow f
        f = state[N:2*N]
        u = G * connectivity.dot(f) + np.random.normal(0, params['sigma'], size=N)
        # integrate one step
        state = odeint(windkessel, state, [t, t+dt], args=(
            u, params['tau_s'], params['tau_f'],
            params['tau_0'], params['alpha'], E0
        ))[-1]
        # Balloon model to BOLD: here we take the normalized volume/oxygenation
        # Simplest BOLD proxy: y = V0 * (k1 * (1 - q) + k2 * (1 - q/v) + k3 * (1 - v))
        v = state[2*N:3*N]
        q = state[3*N:4*N]
        # Constants from Friston ’03
        k1, k2, k3, V0 = params['k1'], params['k2'], params['k3'], params['V0']
        bold[i] = V0 * (k1*(1-q) + k2*(1-q/v) + k3*(1-v))
    return bold, times
