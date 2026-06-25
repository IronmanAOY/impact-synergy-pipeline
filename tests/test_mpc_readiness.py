import numpy as np
import pandas as pd

from impact_pipeline.mpc_readiness import check_mpc_readiness


def _write_events(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(path, sep="\t", index=False)


def _write_ts(path, n_time=80, n_regions=6):
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(0)
    arr = rng.randn(n_time, n_regions)
    np.save(path, arr)


def test_readiness_all_mpcs_ready(tmp_path):
    prep = tmp_path / "preprocessed"
    bids = tmp_path / "bids"
    subj = "01"
    ses = "awake"

    ts_path = prep / subj / ses / "audio" / f"{subj}_run-1_schaefer400_ts.npy"
    _write_ts(ts_path)
    _write_ts(prep / subj / ses / "rest" / f"{subj}_run-1_schaefer400_ts.npy")
    _write_ts(prep / subj / "deep" / "rest" / f"{subj}_run-1_schaefer400_ts.npy")

    ev_path = bids / f"sub-{subj}" / "func" / f"sub-{subj}_task-audioawake_run-01_events.tsv"
    _write_events(
        ev_path,
        [
            {"onset": 0.20, "duration": 0.1, "trial_type": "goal_cue", "reward": np.nan},
            {"onset": 0.80, "duration": 0.1, "trial_type": "audio_stim", "reward": np.nan},
            {"onset": 1.40, "duration": 0.1, "trial_type": "feedback", "reward": 1.0},
            {"onset": 2.00, "duration": 0.1, "trial_type": "audio_stim", "reward": np.nan},
            {"onset": 2.60, "duration": 0.1, "trial_type": "feedback", "reward": 0.0},
            {"onset": 3.20, "duration": 0.1, "trial_type": "self_name"},
            {"onset": 3.80, "duration": 0.1, "trial_type": "other_name"},
            {"onset": 4.40, "duration": 0.1, "trial_type": "self_name"},
            {"onset": 5.00, "duration": 0.1, "trial_type": "other_name"},
            {"onset": 5.60, "duration": 0.1, "trial_type": "self_name"},
            {"onset": 6.20, "duration": 0.1, "trial_type": "other_name"},
        ],
    )

    df, summary = check_mpc_readiness(
        prep_root=str(prep),
        bids_root=str(bids),
        atlas="schaefer400",
        condition="audio",
        sessions=[ses],
    )
    assert df.shape[0] == 1
    row = df.iloc[0]
    assert bool(row["RAM_ready"])
    assert bool(row["PDI_ready"])
    assert bool(row["PDI_anchor_ready"])
    assert bool(row["PDI_task_ready"])
    assert bool(row["NAS_ready"])
    assert bool(row["IIM_ready"])
    assert bool(row["SRPI_ready"])
    assert bool(row["CI_ready"])
    assert summary["metrics"]["CI"]["ready"] == 1


def test_readiness_ram_not_ready_without_feedback(tmp_path):
    prep = tmp_path / "preprocessed"
    bids = tmp_path / "bids"
    subj = "02"
    ses = "awake"

    ts_path = prep / subj / ses / "audio" / f"{subj}_run-1_schaefer400_ts.npy"
    _write_ts(ts_path)

    ev_path = bids / f"sub-{subj}" / "func" / f"sub-{subj}_task-audioawake_run-01_events.tsv"
    _write_events(
        ev_path,
        [
            {"onset": 0.30, "duration": 0.1, "trial_type": "audio_stim"},
            {"onset": 1.30, "duration": 0.1, "trial_type": "audio_stim"},
        ],
    )

    df, _ = check_mpc_readiness(
        prep_root=str(prep),
        bids_root=str(bids),
        atlas="schaefer400",
        condition="audio",
        sessions=[ses],
    )
    row = df.iloc[0]
    assert not bool(row["RAM_ready"])
    assert not bool(row["PDI_ready"])
    assert not bool(row["PDI_anchor_ready"])
    assert not bool(row["PDI_task_ready"])
    assert row["RAM_reason"] in {"missing_feedback_events", "missing_or_nonvarying_feedback_values"}
    assert not bool(row["CI_ready"])
