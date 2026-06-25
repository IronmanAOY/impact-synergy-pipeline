import numpy as np
from impact_pipeline import mpc_metrics as mm


def _build_ram_fixture():
    rng = np.random.RandomState(7)
    n_regions, n_tp = 8, 320
    tr = 0.1
    ts = 0.05 * rng.randn(n_regions, n_tp)
    onsets = np.arange(2.0, 24.0, 2.0)
    kernel = np.exp(-np.arange(0, 12) / 3.0)

    for i, onset in enumerate(onsets):
        idx = int(round(onset / tr))
        end = min(n_tp, idx + kernel.size)
        k = kernel[: max(0, end - idx)]
        if k.size == 0:
            continue
        amp = 0.6 + 0.2 * np.sin(i)
        ts[:, idx:end] += amp * k
        # feedback-like modulation to induce trial-to-trial adaptation signal
        fb_idx = idx + int(round(0.8 / tr))
        fb_end = min(n_tp, fb_idx + 4)
        if fb_idx < n_tp:
            ts[:, fb_idx:fb_end] += (0.1 + 0.05 * ((i % 3) - 1))

    return ts, onsets.tolist(), tr


def test_ram():
    ts, onsets, tr = _build_ram_fixture()
    ram = mm.compute_RAM(
        ts,
        tr=tr,
        stimulus_onsets=onsets,
        require_explicit_feedback=False,
    )
    assert np.isfinite(ram)
    assert ram >= 0.0


def test_ram_structured_event_bundle_details():
    ts, onsets, tr = _build_ram_fixture()
    goal_onsets = [o - 0.6 for o in onsets if (o - 0.6) > 0]
    feedback_onsets = [o + 0.8 for o in onsets if (o + 0.8) < (ts.shape[1] * tr)]
    feedback_values = [(-1.0) ** i * (1.0 + (i % 4)) for i in range(len(feedback_onsets))]
    bundle = {
        "onsets": onsets,
        "goal_onsets": goal_onsets,
        "feedback_onsets": feedback_onsets,
        "feedback_values": feedback_values,
    }

    details = mm.compute_RAM(
        ts,
        tr=tr,
        stimulus_onsets=bundle,
        return_details=True,
    )
    assert np.isfinite(details["value"])
    assert details["value"] >= 0.0
    assert 0.0 <= details["quality_term"] <= 1.0
    comps = details["components"]
    assert 0.0 <= comps["goal_alignment"] <= 1.0
    assert 0.0 <= comps["feedback_integration"] <= 1.0
    assert 0.0 <= comps["adaptive_update"] <= 1.0


def test_ram_missing_events_is_undefined():
    ts, _, tr = _build_ram_fixture()
    ram = mm.compute_RAM(ts, tr=tr, stimulus_onsets={"onsets": []})
    assert np.isnan(ram)

def test_nas():
    ts = np.random.RandomState(0).rand(5, 128)
    val = mm.compute_NAS(
        ts,
        tr=0.01,
        tau=0.2,
        bands=((1.0, 20.0),),
        band_weights=(1.0,),
        window_len=64,
        step_len=32,
    )
    assert 0 <= val <= 1


def test_nas_workspace_and_baseline_boost():
    rng = np.random.RandomState(11)
    n_regions, n_tp = 10, 360
    t = np.linspace(0.0, 12.0, n_tp)
    shared = np.sin(2.0 * np.pi * 2.5 * t)
    task = 0.30 * rng.randn(n_regions, n_tp)
    task[:4] += 0.90 * shared
    base = 0.35 * rng.randn(n_regions, n_tp)

    nas_plain = mm.compute_NAS(
        task,
        tr=0.02,
        tau=0.2,
        bands=((1.0, 20.0),),
        band_weights=(1.0,),
        window_len=120,
        step_len=60,
        workspace_nodes=np.arange(4),
        directed_lag=1,
        reverberation_lags=(2, 3, 4),
        normalize=True,
    )
    nas_boost = mm.compute_NAS(
        task,
        tr=0.02,
        tau=0.2,
        bands=((1.0, 20.0),),
        band_weights=(1.0,),
        window_len=120,
        step_len=60,
        workspace_nodes=np.arange(4),
        directed_lag=1,
        reverberation_lags=(2, 3, 4),
        baseline_ts=base,
        boost_against_baseline=True,
        normalize=True,
    )

    assert 0.0 <= nas_plain <= 1.0
    assert 0.0 <= nas_boost <= 1.0
    assert nas_boost <= nas_plain + 1e-12

def test_pdi_noise_robust_behavior():
    rng = np.random.RandomState(0)
    base = 0.35 * rng.randn(8, 400)
    task = np.sin(np.linspace(0, 20, 400))[None, :] + 0.15 * rng.randn(8, 400)

    p_eq = mm.compute_PDI(base, baseline_ts=[base], normalize=False)
    p_struct = mm.compute_PDI(task, baseline_ts=[base], normalize=False)
    p_struct_norm = mm.compute_PDI(task, baseline_ts=[base], normalize=True)

    assert p_eq == 0.0
    assert p_struct >= 0.0
    assert p_struct_norm >= 0.0 and p_struct_norm <= 1.0


def test_pdi_signed_mode_exposes_negative_contrast():
    rng = np.random.RandomState(4)
    n_regions, n_tp = 8, 420
    time = np.linspace(0, 24, n_tp)
    base = np.vstack(
        [np.sin((0.25 + 0.05 * i) * time + 0.3 * i) for i in range(n_regions)]
    ) + 0.12 * rng.randn(n_regions, n_tp)
    task = 0.08 * rng.randn(n_regions, n_tp)

    p_signed = mm.compute_PDI(task, baseline_ts=[base], clip_negative=False, normalize=False)
    p_clipped = mm.compute_PDI(task, baseline_ts=[base], clip_negative=True, normalize=False)

    assert p_signed <= p_clipped
    assert p_clipped >= 0.0


def test_srpi_missing_events_is_undefined():
    ts = np.random.RandomState(3).randn(6, 240)
    val = mm.compute_SRPI(
        ts,
        tr=0.1,
        self_onsets=[],
        nonself_onsets=[],
        modality="eeg",
        pre_window_sec=0.2,
        response_lag_sec=0.1,
        response_window_sec=0.4,
        covariance_ridge=1e-3,
        component_weights=(0.35, 0.25, 0.20, 0.20),
        min_events_per_class=3,
        sample_reliability_tau=4.0,
    )
    assert np.isnan(val)


def test_srpi_self_model_signal_is_finite():
    rng = np.random.RandomState(23)
    n_regions, n_tp, tr = 6, 700, 0.1
    ts = 0.03 * rng.randn(n_regions, n_tp)

    self_onsets = [6.0, 10.0, 14.0, 18.0, 22.0, 26.0]
    non_onsets = [8.0, 12.0, 16.0, 20.0, 24.0, 28.0]
    self_pattern = np.array([1.0, 0.8, 0.6, 0.2, -0.1, -0.2], dtype=float)
    non_pattern = np.array([0.2, 0.0, 0.1, 0.5, 0.7, 0.9], dtype=float)
    state_axis = np.array([0.7, -0.4, 0.3, 0.1, -0.2, 0.5], dtype=float)

    for i, onset in enumerate(self_onsets):
        idx = int(round(onset / tr))
        latent = np.sin(0.9 * i) + 0.3 * i
        ts[:, idx - 2:idx] += (0.08 * latent) * state_axis[:, None]
        amp = 0.55 + 0.25 * latent
        ts[:, idx + 1:idx + 5] += amp * self_pattern[:, None]

    for i, onset in enumerate(non_onsets):
        idx = int(round(onset / tr))
        latent = np.cos(0.8 * i) - 0.2 * i
        ts[:, idx - 2:idx] += (0.08 * latent) * state_axis[:, None]
        amp = 0.28 + 0.18 * np.sin(1.3 * i + 0.5)
        ts[:, idx + 1:idx + 5] += amp * non_pattern[:, None]

    val = mm.compute_SRPI(
        ts,
        tr=tr,
        self_onsets=self_onsets,
        nonself_onsets=non_onsets,
        modality="eeg",
        pre_window_sec=0.2,
        response_lag_sec=0.1,
        response_window_sec=0.4,
        covariance_ridge=1e-3,
        component_weights=(0.35, 0.25, 0.20, 0.20),
        min_events_per_class=3,
        sample_reliability_tau=4.0,
    )
    assert np.isfinite(val)
    assert 0.0 <= val <= 1.0
    assert val > 0.0
