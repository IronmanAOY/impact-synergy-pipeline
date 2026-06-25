import numpy as np
import pandas as pd
from impact_pipeline.synergy_ci import compute_synergy_ci


_PDI_PARAMS = {
    "bins": 10,
    "weighted": True,
    "normalize": False,
    "clip_negative": True,
    "stability_segments": 4,
    "noise_penalty_kappa": 1.0,
    "component_weights": (0.35, 0.25, 0.20, 0.20),
    "ordinal_order": 3,
    "multiscale_max_scale": 5,
    "eps": 1e-12,
}

_NAS_PARAMS = {
    "zthr": 1.0,
    "eps": 0.2,
    "tau": 0.2,
    "lambda_phase": 0.5,
    "alpha": 0.20,
    "beta": 0.16,
    "gamma": 0.14,
    "delta": 0.12,
    "eta": 0.16,
    "zeta": 0.12,
    "rho": 0.10,
    "bands": ((0.5, 4.0),),
    "band_weights": (1.0,),
    "window_len": 32,
    "step_len": 16,
    "max_triads": 5000,
    "random_state": 0,
    "workspace_nodes": None,
    "workspace_quantile": 0.2,
    "workspace_min_size": 2,
    "directed_lag": 1,
    "reverberation_lags": (2, 3, 4),
    "baseline_ts": None,
    "boost_against_baseline": False,
    "normalize": True,
}

_SRPI_PARAMS = {
    "modality": "eeg",
    "pre_window_sec": 0.2,
    "response_lag_sec": 0.1,
    "response_window_sec": 0.4,
    "covariance_ridge": 1e-3,
    "component_weights": (0.35, 0.25, 0.20, 0.20),
    "min_events_per_class": 3,
    "sample_reliability_tau": 4.0,
    "eps": 1e-8,
}


def test_synergy_ci_shape(tmp_path):
    base=tmp_path/"data"
    for subj in ("s1",):
        d=base/subj/"awake"/"audio"
        d.mkdir(parents=True)
        np.save(d/"s1_run-1_schaefer400_ts.npy",np.random.rand(64,3))
    df=compute_synergy_ci(
        str(base),
        "schaefer400",
        [0.5],
        sessions=("awake",),
        tr=0.1,
        nas_params=_NAS_PARAMS,
        srpi_params=_SRPI_PARAMS,
        srpi_require_explicit_params=True,
    )
    assert isinstance(df,pd.DataFrame)
    assert {'subject', 'session', 'theta', 'S', 'CI'}.issubset(df.columns)

def test_empty_data_dir(tmp_path):
    df=compute_synergy_ci(
        str(tmp_path),
        "schaefer400",
        [0.5],
        sessions=("awake",),
        tr=0.1,
        nas_params=_NAS_PARAMS,
        srpi_params=_SRPI_PARAMS,
        srpi_require_explicit_params=True,
    )
    assert df.empty


def test_pdi_strict_dual_baseline_requires_measured_rest_runs(tmp_path):
    base = tmp_path / "data"
    subj = "s1"
    d = base / subj / "awake" / "audio"
    d.mkdir(parents=True)
    np.save(d / f"{subj}_run-1_schaefer400_ts.npy", np.random.RandomState(1).rand(16, 6))

    df = compute_synergy_ci(
        str(base),
        "schaefer400",
        [0.5],
        sessions=("awake",),
        mpc_metrics=("PDI",),
        compute_ci=False,
        pdi_params=_PDI_PARAMS,
        pdi_require_explicit_params=True,
        pdi_require_strict_baseline=True,
    )
    row = df.iloc[0]
    assert np.isnan(row["PDI"])
    assert row["PDI_anchor_reason"] == "missing_deep_rest_baseline"
    assert row["PDI_task_reason"] == "missing_state_rest_baseline"
