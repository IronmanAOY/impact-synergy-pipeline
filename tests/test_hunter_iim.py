import dataclasses

import numpy as np
import pytest

from impact_pipeline.execution_profiles import get_execution_profile
from impact_pipeline.hunter_iim import (
    collect_iim_results_by_path,
    prepare_hunter_campaign,
    run_cut_reduce,
    run_cut_shard,
    run_phase1_reduce,
    run_phase1_shard,
)
from impact_pipeline import mpc_metrics as mm


def test_hunter_iim_matches_direct_compute(tmp_path):
    data_dir = tmp_path / "prep"
    run_dir = data_dir / "s1" / "awake" / "audio"
    run_dir.mkdir(parents=True)

    rng = np.random.RandomState(5)
    ts_time_region = rng.rand(64, 4)
    ts_path = run_dir / "s1_run-1_schaefer400_ts.npy"
    np.save(ts_path, ts_time_region, allow_pickle=False)

    hunter_profile = dataclasses.replace(
        get_execution_profile("hunter"),
        hunter_phase1_shards_per_run=2,
        hunter_cut_shards_per_run=2,
        hunter_phase1_workers_per_task=1,
        hunter_shared_memory=False,
    )
    campaign_dir = tmp_path / "campaign"
    manifest = prepare_hunter_campaign(
        data_dir=data_dir,
        atlas="schaefer400",
        sessions=("awake",),
        condition="audio",
        stimulus_onsets=None,
        subjects=None,
        campaign_dir=campaign_dir,
        execution_profile=hunter_profile,
        iim_bins=2,
        iim_lag_trs=1,
        iim_n_parts=3,
        iim_max_timepoints=None,
        iim_max_nodes=4,
        iim_max_mechanism_size=2,
        iim_max_purview_size=2,
        step2_context={},
    )

    for i, _task in enumerate(manifest["phase1_tasks"]):
        run_phase1_shard(campaign_dir, i)
    run_phase1_reduce(campaign_dir, 0)
    for i, _task in enumerate(manifest["cut_tasks"]):
        run_cut_shard(campaign_dir, i)
    run_cut_reduce(campaign_dir, 0)

    hunter_map = collect_iim_results_by_path(campaign_dir)
    hunter_info = hunter_map[str(ts_path.resolve())]
    direct_info = mm.compute_IIM(
        ts_time_region.T,
        bins=2,
        lag_trs=1,
        n_parts=3,
        max_nodes=4,
        max_mechanism_size=2,
        max_purview_size=2,
        return_details=True,
        phase1_parallel_workers=None,
    )

    assert hunter_info["defined"] is True
    assert direct_info["defined"] is True
    assert hunter_info["raw"] == pytest.approx(direct_info["raw"], rel=1e-9, abs=1e-9)
    assert hunter_info["canonical"] == pytest.approx(direct_info["canonical"], rel=1e-9, abs=1e-9)
    assert hunter_info["Psi_full"] == pytest.approx(direct_info["Psi_full"], rel=1e-9, abs=1e-9)
    assert hunter_info["Psi_mip_preserved"] == pytest.approx(
        direct_info["Psi_mip_preserved"],
        rel=1e-9,
        abs=1e-9,
    )


def test_hunter_slurm_scripts_embed_handoff_runtime(monkeypatch, tmp_path):
    monkeypatch.setenv("IMPACT_HUNTER_CONDA_ENV", "impact-synergy-clean")
    monkeypatch.setenv("IMPACT_HUNTER_SLURM_ACCOUNT", "project123")
    monkeypatch.setenv("IMPACT_HUNTER_CPU_PARTITION", "cpu-test")
    monkeypatch.setenv("IMPACT_HUNTER_APU_PARTITION", "apu-test")
    monkeypatch.setenv(
        "IMPACT_HUNTER_SLURM_SETUP",
        "module load miniforge\nexport MPLCONFIGDIR=${SLURM_TMPDIR:-/tmp}/mpl",
    )

    data_dir = tmp_path / "prep"
    run_dir = data_dir / "s1" / "awake" / "audio"
    run_dir.mkdir(parents=True)
    ts_path = run_dir / "s1_run-1_schaefer400_ts.npy"
    np.save(ts_path, np.random.RandomState(11).rand(32, 3), allow_pickle=False)

    hunter_profile = dataclasses.replace(
        get_execution_profile("hunter"),
        hunter_phase1_shards_per_run=1,
        hunter_cut_shards_per_run=1,
        hunter_phase1_workers_per_task=1,
        hunter_shared_memory=False,
    )
    campaign_dir = tmp_path / "campaign"
    prepare_hunter_campaign(
        data_dir=data_dir,
        atlas="schaefer400",
        sessions=("awake",),
        condition="audio",
        stimulus_onsets=None,
        subjects=None,
        campaign_dir=campaign_dir,
        execution_profile=hunter_profile,
        iim_bins=2,
        iim_lag_trs=1,
        iim_n_parts=2,
        iim_max_timepoints=None,
        iim_max_nodes=3,
        iim_max_mechanism_size=2,
        iim_max_purview_size=2,
        step2_context={"hardware_target": "cpu"},
    )

    phase1_script = (campaign_dir / "slurm" / "01_phase1_shards.sbatch").read_text(
        encoding="utf-8"
    )
    cut_script = (campaign_dir / "slurm" / "03_cut_shards.sbatch").read_text(
        encoding="utf-8"
    )
    submit_script = (campaign_dir / "slurm" / "00_submit_all.sh").read_text(
        encoding="utf-8"
    )

    assert "#SBATCH --account=project123" in phase1_script
    assert "#SBATCH --partition=cpu-test" in phase1_script
    assert "#SBATCH --partition=cpu-test" in cut_script
    assert "module load miniforge" in phase1_script
    assert "conda run -n impact-synergy-clean python" in phase1_script
    assert 'campaign_dir="$(cd "${script_dir}/.." && pwd)"' in submit_script
