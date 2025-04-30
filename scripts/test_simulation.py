import numpy as np
import matplotlib.pyplot as plt
from simulation import simulate_bold  # assuming your code is in simulation.py

def test_simulation():
    # 1) Tiny toy network: N=5, very short run
    N, T, dt = 5, 10.0, 0.5  # 20 time points
    # toy connectivity: identity (no coupling) or small random
    C = np.eye(N)  

    params = {
        'G': 0.0,      # no coupling => pure noise-driven
        'E0': 0.34,
        'tau_s': 1.0, 'tau_f': 1.0, 'tau_0': 1.0,
        'alpha': 0.32, 'sigma': 0.01,
        'k1': 7*0.34, 'k2': 2.0, 'k3': 2*0.34 - 0.2, 'V0': 0.02
    }

    bold, times = simulate_bold(N, T, dt, C, params)

    # 2) Shape and NaN checks
    assert bold.shape == (len(times), N), f"Expected shape {(len(times), N)}, got {bold.shape}"
    assert not np.any(np.isnan(bold)), "NaNs found in BOLD output"
    assert not np.any(np.isinf(bold)), "Infs found in BOLD output"

    print("✅ Simulation output shape and numerical sanity checks passed.")
    print(f"Time vector length: {len(times)}, N nodes: {N}")

    # 3) Plot a few nodes
    for node in range(min(3, N)):
        plt.plot(times, bold[:, node], label=f"node {node}")
    plt.xlabel("Time (s)")
    plt.ylabel("BOLD signal (arbitrary units)")
    plt.title("Smoke-test: BOLD timecourses (G=0)")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_simulation()
