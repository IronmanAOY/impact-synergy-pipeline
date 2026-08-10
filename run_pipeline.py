#!/usr/bin/env python
import glob
import sys
import argparse
import logging
import random
import tempfile
import numpy as np
import subprocess, os
import pandas as pd
import json

from pathlib import Path
from bids import BIDSLayout

root = Path(__file__).resolve().parent
src_root = root / "src"
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

mpl_cache_dir = Path(tempfile.gettempdir()) / "impact_mpl_cache"
mpl_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache_dir))

from impact_pipeline.preprocessing_eeg import run_preprocessing_eeg
from impact_pipeline.baseline_metrics import compute_baseline_metrics
from impact_pipeline.analysis_bootstrap import bootstrap_ci, permutation_test_auc
from impact_pipeline.motion_model import motion_covariate_analysis
from impact_pipeline.atlas_robustness import atlas_check
from impact_pipeline.replication import run_replication
from impact_pipeline.model_comparison import compare_models
from impact_pipeline.generate_word_doc import create_doc
from impact_pipeline.execution_profiles import get_execution_profile
from impact_pipeline.hardware_backend import (
    HardwareBackendError,
    backend_summary,
    configure_process_for_hardware,
)
from impact_pipeline.hunter_iim import (
    collect_iim_results_by_path,
    prepare_hunter_campaign,
    run_cut_reduce,
    run_cut_shard,
    run_phase1_reduce,
    run_phase1_shard,
)
from impact_pipeline.provenance import (
    PROVENANCE_COLUMNS,
    REAL_DATA_ORIGIN,
    is_test_object_origin,
    resolve_dataset_provenance,
    write_json,
)
from impact_pipeline.dataset_catalog import get_report_dataset

# ---------------------------------------------------------------------
### ── TOGGLE FULL-RUN STEPS ──────────────────────────────────────────────
RUN_FMRIPREP      = False   # set True to run step 0 (fMRIPrep)
RUN_PREPROCESSING = False   # set True to run step 1 (preprocessing)
RUN_REPLICATION   = False   # set True to run step 7 (replication)

random.seed(42)
np.random.seed(42)
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("pipeline")
EXPECTED_CONDA_ENV = os.environ.get("IMPACT_CONDA_ENV", "impact-synergy-clean")

DATASET_CONFIGS = {
    "ds003171": {
        "modality": "fmri",
        "bids_candidates": (
            root / "data" / "scratch" / "ds003171",
            root / "data" / "ds003171",
        ),
        "atlas": "schaefer400",
        "sessions": ("awake", "deep"),
        "condition": "audio",
        "atlas_robustness": True,
        "supports_fmriprep": True,
        "supports_replication": True,
        "iim_n_parts": None,
        "iim_max_timepoints": None,
        "iim_max_nodes": None,
        "iim_max_mechanism_size": None,
        "iim_max_purview_size": None,
        "iim_parallel_workers": None,
        "iim_memory_target_ratio": 0.90,
        "iim_worker_mem_gb_estimate": 3.0,
        "iim_cpu_oversub_factor": 3.0,
        "iim_phase1_parallel_workers": None,
        "iim_phase1_chunk_size": 8,
        "iim_phase1_shared_memory": True,
        "pdi_params": {
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
        },
        "pdi_require_explicit_params": True,
        "pdi_require_strict_baseline": True,
        "pdi_primary_endpoint": "anchor",
        "nas_params": {
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
            "bands": ((0.01, 0.10),),
            "band_weights": (1.0,),
            "window_len": 30,
            "step_len": 15,
            "max_triads": 5000,
            "random_state": 0,
            "workspace_nodes": None,
            "workspace_quantile": 0.2,
            "workspace_min_size": 4,
            "directed_lag": 1,
            "reverberation_lags": (2, 3, 4),
            "baseline_ts": None,
            "boost_against_baseline": False,
            "normalize": True,
        },
        "srpi_params": {
            "modality": "fmri",
            "pre_window_sec": 2.0,
            "response_lag_sec": 4.0,
            "response_window_sec": 6.0,
            "covariance_ridge": 1e-3,
            "component_weights": (0.35, 0.25, 0.20, 0.20),
            "min_events_per_class": 3,
            "sample_reliability_tau": 4.0,
            "eps": 1e-8,
        },
        "srpi_require_explicit_params": True,
        "eeg_session_rules": None,
    },
    "ds002547": {
        "modality": "fmri",
        "bids_candidates": (
            root / "data" / "scratch" / "ds002547",
            root / "data" / "ds002547",
        ),
        "atlas": "schaefer400",
        "sessions": ("awake", "deep"),
        "condition": "selfother",
        "atlas_robustness": False,
        "supports_fmriprep": False,
        "supports_replication": False,
        "iim_n_parts": None,
        "iim_max_timepoints": None,
        "iim_max_nodes": None,
        "iim_max_mechanism_size": None,
        "iim_max_purview_size": None,
        "iim_parallel_workers": None,
        "iim_memory_target_ratio": 0.90,
        "iim_worker_mem_gb_estimate": 3.0,
        "iim_cpu_oversub_factor": 3.0,
        "iim_phase1_parallel_workers": None,
        "iim_phase1_chunk_size": 8,
        "iim_phase1_shared_memory": True,
        "pdi_params": {
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
        },
        "pdi_require_explicit_params": True,
        "pdi_require_strict_baseline": True,
        "pdi_primary_endpoint": "anchor",
        "nas_params": {
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
            "bands": ((0.01, 0.10),),
            "band_weights": (1.0,),
            "window_len": 30,
            "step_len": 15,
            "max_triads": 5000,
            "random_state": 0,
            "workspace_nodes": None,
            "workspace_quantile": 0.2,
            "workspace_min_size": 4,
            "directed_lag": 1,
            "reverberation_lags": (2, 3, 4),
            "baseline_ts": None,
            "boost_against_baseline": False,
            "normalize": True,
        },
        "srpi_params": {
            "modality": "fmri",
            "pre_window_sec": 2.0,
            "response_lag_sec": 4.0,
            "response_window_sec": 6.0,
            "covariance_ridge": 1e-3,
            "component_weights": (0.35, 0.25, 0.20, 0.20),
            "min_events_per_class": 3,
            "sample_reliability_tau": 4.0,
            "eps": 1e-8,
        },
        "srpi_require_explicit_params": True,
        "eeg_session_rules": None,
    },
    "ds005620": {
        "modality": "eeg",
        "bids_candidates": (
            root / "data" / "scratch" / "ds005620_annex",
            root / "data" / "scratch" / "ds005620",
            root / "data" / "ds005620",
        ),
        "atlas": "eeg64",
        "sessions": ("awake", "deep"),
        "condition": "eeg",
        "atlas_robustness": False,
        "supports_fmriprep": False,
        "supports_replication": False,
        "iim_n_parts": None,
        "iim_max_timepoints": None,
        "iim_max_nodes": None,
        "iim_max_mechanism_size": None,
        "iim_max_purview_size": None,
        "iim_parallel_workers": None,
        "iim_memory_target_ratio": 0.90,
        "iim_worker_mem_gb_estimate": 3.0,
        "iim_cpu_oversub_factor": 3.0,
        "iim_phase1_parallel_workers": None,
        "iim_phase1_chunk_size": 8,
        "iim_phase1_shared_memory": True,
        "pdi_params": {
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
        },
        "pdi_require_explicit_params": True,
        "pdi_require_strict_baseline": True,
        "pdi_primary_endpoint": "anchor",
        "nas_params": {
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
            "bands": ((0.5, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 45.0)),
            "band_weights": (0.2, 0.2, 0.2, 0.2, 0.2),
            "window_len": 500,
            "step_len": 250,
            "max_triads": 5000,
            "random_state": 0,
            "workspace_nodes": None,
            "workspace_quantile": 0.2,
            "workspace_min_size": 4,
            "directed_lag": 1,
            "reverberation_lags": (2, 3, 4),
            "baseline_ts": None,
            "boost_against_baseline": False,
            "normalize": True,
        },
        "srpi_params": {
            "modality": "eeg",
            "pre_window_sec": 0.20,
            "response_lag_sec": 0.05,
            "response_window_sec": 0.40,
            "covariance_ridge": 1e-3,
            "component_weights": (0.35, 0.25, 0.20, 0.20),
            "min_events_per_class": 3,
            "sample_reliability_tau": 4.0,
            "eps": 1e-8,
        },
        "srpi_require_explicit_params": True,
        "eeg_session_rules": {
            "awake": [("awake", "EC")],
            "deep": [("sed2", "rest"), ("sed", "rest")],
        },
    },
}


def _first_existing_path(*candidates):
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def _apply_metric_subset(df, df_mean, mpc_metrics=None, compute_ci=True):
    """
    Keep only S and the selected MPC metrics in run tables.
    Useful when reusing cached step-2 outputs produced with a wider metric set.
    """
    if mpc_metrics is None:
        selected = {"RAM", "PDI", "NAS", "IIM", "SRPI"}
    else:
        selected = set(mpc_metrics)

    keep_df = [c for c in PROVENANCE_COLUMNS if c in df.columns]
    keep_df.extend(['subject', 'session', 'theta', 'S'])
    for metric in ("RAM", "PDI", "NAS", "IIM", "SRPI"):
        if metric in selected and metric in df.columns:
            keep_df.append(metric)
    if "PDI" in selected:
        for extra in (
            "PDI_anchor",
            "PDI_task",
            "PDI_anchor_defined",
            "PDI_task_defined",
            "PDI_anchor_reason",
            "PDI_task_reason",
            "PDI_primary_endpoint",
            "PDI_primary_source",
            "PDI_baseline_policy",
            "PDI_anchor_baseline_n_runs",
            "PDI_task_baseline_n_runs",
            "PDI_anchor_baseline_paths",
            "PDI_task_baseline_paths",
        ):
            if extra in df.columns:
                keep_df.append(extra)
    if "IIM" in selected:
        for extra in ("IIM_raw", "IIM_raw_scaled", "IIM_defined", "IIM_undefined_reason"):
            if extra in df.columns:
                keep_df.append(extra)
    if compute_ci and "CI" in df.columns and selected.issuperset({"RAM", "PDI", "NAS", "IIM", "SRPI"}):
        keep_df.append("CI")
        for extra in ("RAM_norm", "PDI_norm", "NAS_norm", "IIM_norm", "SRPI_norm"):
            if extra in df.columns:
                keep_df.append(extra)

    df_filtered = df.loc[:, [c for c in keep_df if c in df.columns]].copy()

    # Some cached df_mean tables may miss per-metric columns; recover from df.
    desired_means = []
    for m in ("RAM", "PDI", "PDI_anchor", "PDI_task", "NAS", "IIM", "SRPI", "IIM_raw", "IIM_raw_scaled"):
        include = (m in selected) or (m.startswith("IIM") and "IIM" in selected)
        if m in {"PDI_anchor", "PDI_task"} and "PDI" in selected:
            include = True
        if include:
            desired_means.append(m)
    missing_in_mean = [m for m in desired_means if m not in df_mean.columns and m in df_filtered.columns]
    if missing_in_mean:
        recovered = (
            df_filtered
            .groupby(["subject", "session"])[missing_in_mean]
            .mean()
            .reset_index()
        )
        df_mean = df_mean.merge(recovered, on=["subject", "session"], how="left")

    keep_mean = [c for c in PROVENANCE_COLUMNS if c in df_mean.columns]
    keep_mean.extend(['subject', 'session', 'S'])
    for metric in ("RAM", "PDI", "PDI_anchor", "PDI_task", "NAS", "IIM", "SRPI", "IIM_raw", "IIM_raw_scaled"):
        include = (metric in selected) or (metric.startswith("IIM") and "IIM" in selected)
        if metric in {"PDI_anchor", "PDI_task"} and "PDI" in selected:
            include = True
        if metric in df_mean.columns and include:
            keep_mean.append(metric)
    if compute_ci and "CI" in df_mean.columns and selected.issuperset({"RAM", "PDI", "NAS", "IIM", "SRPI"}):
        keep_mean.append("CI")

    df_mean_filtered = df_mean.loc[:, [c for c in keep_mean if c in df_mean.columns]].copy()
    return df_filtered, df_mean_filtered


def _write_run_provenance_manifest(
    *,
    cache_dir: Path,
    provenance,
    modality: str,
    bids_root: Path | None,
    execution_mode: str,
    atlas: str,
    sessions,
    condition: str,
) -> None:
    payload = provenance.as_manifest_dict()
    payload.update(
        {
            "modality": str(modality),
            "bids_root": (None if bids_root is None else str(Path(bids_root).resolve())),
            "execution_mode": str(execution_mode),
            "atlas": str(atlas),
            "sessions": [str(s) for s in sessions],
            "condition": str(condition),
            "result_usage_note": (
                "Results retain explicit dataset-origin provenance. Synthetic validation outputs "
                "remain isolated from real-study outputs unless selected explicitly."
            ),
        }
    )
    write_json(cache_dir / "provenance_manifest.json", payload)


def _ensure_provenance_columns(df: pd.DataFrame, provenance) -> pd.DataFrame:
    out = df.copy()
    for key, value in provenance.as_result_metadata().items():
        if key not in out.columns:
            out[key] = value
        else:
            out[key] = out[key].replace("", pd.NA).fillna(value)
    return out


def _export_test_object_metric_bank(
    *,
    df: pd.DataFrame,
    df_mean: pd.DataFrame,
    cache_dir: Path,
    provenance,
    modality: str,
    atlas: str,
    sessions,
    condition: str,
) -> None:
    metric_bank_dir = provenance.metric_bank_dataset_dir
    if metric_bank_dir is None:
        return

    metric_bank_dir.mkdir(parents=True, exist_ok=True)
    per_metric_dir = metric_bank_dir / "per_metric"
    per_metric_dir.mkdir(parents=True, exist_ok=True)

    full_df_path = metric_bank_dir / "step2_df.csv"
    full_mean_path = metric_bank_dir / "step2_df_mean.csv"
    df.to_csv(full_df_path, index=False)
    df_mean.to_csv(full_mean_path, index=False)

    metric_files = {}
    for metric in ("RAM", "PDI", "NAS", "IIM", "SRPI", "CI"):
        if metric not in df.columns:
            continue
        keep_cols = [c for c in PROVENANCE_COLUMNS if c in df.columns]
        keep_cols.extend([c for c in ("subject", "session", "theta", "S", metric) if c in df.columns])
        if metric == "PDI":
            keep_cols.extend(
                [
                    c
                    for c in (
                        "PDI_anchor",
                        "PDI_task",
                        "PDI_anchor_defined",
                        "PDI_task_defined",
                        "PDI_anchor_reason",
                        "PDI_task_reason",
                        "PDI_primary_endpoint",
                        "PDI_primary_source",
                        "PDI_baseline_policy",
                        "PDI_anchor_baseline_n_runs",
                        "PDI_task_baseline_n_runs",
                        "PDI_anchor_baseline_paths",
                        "PDI_task_baseline_paths",
                    )
                    if c in df.columns
                ]
            )
        if metric == "IIM":
            keep_cols.extend(
                [c for c in ("IIM_raw", "IIM_raw_scaled", "IIM_defined", "IIM_undefined_reason") if c in df.columns]
            )
        out_path = per_metric_dir / f"{metric}.csv"
        df.loc[:, keep_cols].to_csv(out_path, index=False)
        metric_files[metric] = str(out_path)

    manifest = provenance.as_manifest_dict()
    manifest.update(
        {
            "source_dataset_id": str(provenance.dataset_id),
            "source_type": "synthetic_validation",
            "modality": str(modality),
            "atlas": str(atlas),
            "sessions": [str(s) for s in sessions],
            "condition": str(condition),
            "step2_cache_dir": str(cache_dir),
            "step2_df_path": str(full_df_path),
            "step2_df_mean_path": str(full_mean_path),
            "available_metrics": [m for m in ("RAM", "PDI", "NAS", "IIM", "SRPI", "CI") if m in df.columns],
            "metric_files": metric_files,
            "row_count": int(len(df)),
            "row_count_mean": int(len(df_mean)),
            "usage_note": (
                "This metric bank contains synthetic validation results. Mixed-source CI analyses "
                "must select these results explicitly and keep provenance labels intact."
            ),
        }
    )
    write_json(metric_bank_dir / "metric_bank_manifest.json", manifest)


def _assert_expected_runtime_env(expected_env: str = EXPECTED_CONDA_ENV) -> None:
    """
    Fail fast when the pipeline is launched from an unexpected Python runtime.
    """
    skip_check = os.environ.get("IMPACT_SKIP_ENV_CHECK", "").strip().lower()
    if skip_check in {"1", "true", "yes", "on"}:
        log.warning("Skipping runtime environment check (IMPACT_SKIP_ENV_CHECK=%s).", skip_check)
        return

    conda_default = os.environ.get("CONDA_DEFAULT_ENV", "")
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    exe = sys.executable
    tokens = [conda_default, conda_prefix, exe]
    ok = any(expected_env in t for t in tokens if t)

    if not ok:
        raise RuntimeError(
            "Invalid Python runtime for run_pipeline.py.\n"
            f"Expected conda env: '{expected_env}'\n"
            f"Detected CONDA_DEFAULT_ENV='{conda_default}', CONDA_PREFIX='{conda_prefix}', "
            f"sys.executable='{exe}'.\n"
            f"Run with: conda run -n {expected_env} python run_pipeline.py ..."
        )


def _write_eeg_preprocessing_summary(cache_dir: Path, summary_payload: dict) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)

    (cache_dir / "preprocessing_eeg_summary.json").write_text(
        json.dumps(summary_payload, indent=2),
        encoding="utf-8",
    )

    table_map = {
        "missing_files": (
            "preprocessing_eeg_missing_files.csv",
            ["subject", "session", "task", "acquisition", "file", "reason"],
        ),
        "skipped_subjects": (
            "preprocessing_eeg_skipped_subjects.csv",
            ["subject", "reason", "detail"],
        ),
        "written_runs": (
            "preprocessing_eeg_written_runs.csv",
            [
                "subject",
                "session",
                "source_file",
                "output_file",
                "run_index",
                "n_channels",
                "n_timepoints",
                "sfreq_hz",
            ],
        ),
    }
    for key, (name, cols) in table_map.items():
        rows = summary_payload.get(key, [])
        pd.DataFrame(rows).reindex(columns=cols).to_csv(cache_dir / name, index=False)


def _resolve_freesurfer_license_path() -> Path:
    env_path = os.environ.get("FS_LICENSE", "").strip()
    if env_path:
        lic = Path(env_path).expanduser().resolve()
        if not lic.exists():
            raise FileNotFoundError(
                f"FS_LICENSE points to '{lic}', but that file does not exist."
            )
        return lic

    local_default = root / "licenses" / "fs_license.txt"
    if local_default.exists():
        return local_default

    raise FileNotFoundError(
        "FreeSurfer license not found. Set FS_LICENSE to your local license file "
        "or place it at 'licenses/fs_license.txt' (see licenses/fs_license.txt.example)."
    )


def ensure_fmriprep(bids_dir, fmriprep_out, work_dir, fs_license, freesurf_out, subjects=None):

    layout = BIDSLayout(bids_dir, validate=False)
    all_subj = sorted(layout.get(return_type='id', target='subject'))
    if subjects:
        subjects = [s for s in all_subj if s in subjects]
    else:
        subjects = all_subj
 

    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(fmriprep_out, exist_ok=True)
    os.makedirs(freesurf_out, exist_ok=True)
    
    for sub in subjects:
        log.info("Running fMRIPrep on %s …", sub)
        cache_host = Path(work_dir) / 'bids_db'
        cache_cont = '/bids_db'
        cmd = [
          'docker','run','--rm',
          '-v', f'{bids_dir}:/data',
          '-v', f'{fmriprep_out}:/out',
          '-v', f'{work_dir}:/work',
          '-v', f'{freesurf_out}:/out_freesurfer',
          '-v', f'{Path(fs_license).parent.resolve()}:/licenses:ro',
          '-v', f'{cache_host}:{cache_cont}',
          'nipreps/fmriprep:25.1.3',
          '/data','/out','participant',
          '--participant-label', sub,
          '--fs-license-file',
          '/licenses/fs_license.txt', # make sure your freesurfer license is named and placed correctly
          '--fs-subjects-dir', '/out_freesurfer',
          '--bids-database-dir', cache_cont,
          '--work-dir','/work',
 #         '--clean-workdir',   # Enable only for final full-dataset fMRIPrep runs when intermediate work files can be discarded.
          '--skip-bids-validation',
          '--nthreads', '16', # or however many logical CPUs you have
          '--omp-nthreads', '8', # ~ half of total
          '--mem', '96000', # adjust to your RAM
        ]
        subprocess.run(cmd, check=True)


def _persist_step2_outputs(
    cache_dir: Path,
    df: pd.DataFrame,
    df_mean: pd.DataFrame,
    df_stats_by_theta: pd.DataFrame,
    *,
    persist_primary: bool,
) -> None:
    if bool(persist_primary):
        step2_df_path = cache_dir / "step2_df.csv"
        step2_df_mean_path = cache_dir / "step2_df_mean.csv"
        step2_theta_stats_path = cache_dir / "step2_theta_stats.csv"
        df.to_csv(step2_df_path, index=False)
        df_mean.to_csv(step2_df_mean_path, index=False)
        df_stats_by_theta.reset_index().to_csv(step2_theta_stats_path, index=False)
    (cache_dir / "step2_df_active.csv").write_text(df.to_csv(index=False), encoding="utf-8")
    (cache_dir / "step2_df_mean_active.csv").write_text(df_mean.to_csv(index=False), encoding="utf-8")


def _postprocess_after_step2(
    *,
    df,
    df_mean,
    prep_out,
    bids_root,
    figdir,
    atlas,
    sessions,
    compute_ci,
    modality,
    dataset_id,
    cfg,
    subjects,
    out,
    report_doc,
    pdi_params,
    pdi_require_explicit_params,
    pdi_require_strict_baseline,
    pdi_primary_endpoint,
    nas_params,
    srpi_params,
    srpi_require_explicit_params,
    run_replication_flag,
    df_stats_by_theta,
):
    df_S = df_mean.pivot(index='subject', columns='session', values='S')
    df_CI = df_mean.pivot(index='subject', columns='session', values='CI') if 'CI' in df_mean.columns else None

    log.info("3/9 Computing baseline graph metrics")
    df_baseline = compute_baseline_metrics(
        df_mean,
        data_dir=str(prep_out),
        atlas=atlas
    )
    log.debug("df_baseline columns: %s", df_baseline.columns.tolist())

    log.info("4/9 Bootstrapping CIs and permutation tests")
    boot_S = bootstrap_ci(df_mean, 'S', sessions=sessions)
    boot_CI = bootstrap_ci(df_mean, 'CI', sessions=sessions) if 'CI' in df_mean.columns else (np.nan, np.nan)

    auc_S, p_S = permutation_test_auc(df_mean, 'S', sessions=sessions) or (None, None)
    auc_CI, p_CI = (
        permutation_test_auc(df_mean, 'CI', sessions=sessions) or (None, None)
    ) if 'CI' in df_mean.columns else (None, None)
    theta_results = {}
    for theta in sorted(df['theta'].unique()):
        subdf = df[df['theta'] == theta]
        agg_theta = {'S': ('S', 'mean')}
        if 'CI' in subdf.columns:
            agg_theta['CI'] = ('CI', 'mean')
        df_theta_mean = (
            subdf.groupby(['subject', 'session'])
                 .agg(**agg_theta)
                 .reset_index()
        )
        auc_S_theta, p_S_theta = permutation_test_auc(df_theta_mean, 'S', sessions=sessions) or (None, None)
        if 'CI' in df_theta_mean.columns:
            auc_CI_theta, p_CI_theta = permutation_test_auc(df_theta_mean, 'CI', sessions=sessions) or (None, None)
        else:
            auc_CI_theta, p_CI_theta = (None, None)
        theta_results[theta] = {
            'auc_S': auc_S_theta, 'p_S': p_S_theta,
            'auc_CI': auc_CI_theta, 'p_CI': p_CI_theta
        }

    log.info("5/9 Motion covariate analysis")
    if log.isEnabledFor(logging.DEBUG):
        log.debug("df columns: %s", df.columns.tolist())
        log.debug("df_mean columns: %s", df_mean.columns.tolist())
        rows_per_group = (
            df.groupby(['subject', 'session']).size()
            .rename('rows').reset_index()
            .sort_values('rows', ascending=False)
            .head(10)
        )
        log.debug("rows per (subject, session):\n%s", rows_per_group.to_string(index=False))
        if 'theta' in df.columns:
            n_theta = (
                df.groupby(['subject', 'session'])['theta']
                .nunique().rename('n_theta').reset_index()
                .sort_values('n_theta', ascending=False)
                .head(10)
            )
            log.debug("n_theta per (subject, session):\n%s", n_theta.to_string(index=False))
        for cand in ['condition', 'task', 'run', 'acq', 'desc']:
            if cand in df.columns:
                log.debug("%s unique values: %s", cand, df[cand].unique()[:10])

    if (modality == "fmri") and ('CI' in df_mean.columns):
        motion = motion_covariate_analysis(df_mean, str(prep_out), atlas=atlas)
    else:
        motion = {'skipped': f"motion~CI omitted for modality={modality} or missing CI."}

    if cfg.get("atlas_robustness", False) and modality == "fmri":
        sidecar_path = glob.glob(str(bids_root / "sub-*/func/*_bold.json"))[0]
        with open(sidecar_path, "r", encoding="utf-8") as f:
            sidecar = json.load(f)
        real_tr = sidecar['RepetitionTime']
        log.info("6/9 Testing robustness across atlases")
        robust_metrics = [m for m in ("RAM", "PDI", "NAS", "IIM", "SRPI") if m in df.columns]
        robust_ci = bool(compute_ci and ("CI" in df.columns))
        atlas_res = atlas_check(
            str(prep_out),
            atlases=('aal90','shen268'),
            sessions=sessions,
            thetas=np.arange(0.1, 1.0, 0.1),
            tr=real_tr,
            stimulus_onsets=None,
            mpc_metrics=robust_metrics,
            compute_ci=robust_ci,
            pdi_params=pdi_params,
            pdi_require_explicit_params=pdi_require_explicit_params,
            pdi_require_strict_baseline=pdi_require_strict_baseline,
            pdi_primary_endpoint=pdi_primary_endpoint,
            nas_params=nas_params,
            srpi_params=srpi_params,
            srpi_require_explicit_params=srpi_require_explicit_params,
            subjects=subjects,
        )
    else:
        atlas_res = {
            "skipped": {
                "reason": f"atlas robustness not configured for modality={modality}",
                "dataset": dataset_id,
            }
        }

    repl = None
    if run_replication_flag and cfg.get("supports_replication", False):
        log.info("7/9 Replication (Melbourne Propofol)")
        melb_root = _first_existing_path(
            root / 'data' / 'scratch' / 'melbourne',
            root / 'data' / 'scratch' / 'melbourne_propofol',
            root / 'data' / 'melbourne',
            root / 'data' / 'melbourne_propofol',
        )
        melb_out = out / 'melbourne' / 'preprocessed'
        repl = run_replication(
            data_root=str(melb_root),
            out_dir=str(melb_out),
            atlas=atlas,
            sessions=('awake', 'deep')
        )
    elif run_replication_flag:
        log.info("7/9 Replication skipped: not configured for dataset '%s'", dataset_id)

    log.info("8/9 Comparing to baseline and PCI models")
    mc = compare_models(
        df_baseline,
        metrics=('mean_conn', 'modularity', 'pci_fmri'),
        sessions=('awake', 'deep')
    )
    log.debug("baseline session counts:\n%s", df_baseline['session'].value_counts().to_string())

    log.info("9/9 Generating Word report")
    report_path = out / report_doc
    create_doc(
        path=str(report_path),
        df=df,
        df_stats_by_theta=df_stats_by_theta,
        motion=motion,
        atlas_results=atlas_res,
        mc=mc,
        theta_results=theta_results,
        fig_dir=str(figdir),
        use_sem=True,
        S_SCALE=1e3,
        RAM_SCALE=1e3,
        CI_SCALE=1.0,
        LABEL_S="S×10³",
        LABEL_RAM="RAM×10³",
        LABEL_CI="CI (human-normalized)",
        repl=repl
    )
    log.info("Pipeline complete! Outputs in %s", out)


def _run_hunter_stage(
    *,
    hunter_stage,
    hunter_campaign_dir,
    hunter_task_index,
    hunter_run_index,
    execution_profile,
    prep_out,
    bids_root,
    figdir,
    cache_dir,
    atlas,
    sessions,
    condition,
    metric_tr,
    subjects,
    mpc_metrics,
    compute_ci,
    iim_n_parts,
    iim_max_timepoints,
    iim_max_nodes,
    iim_max_mechanism_size,
    iim_max_purview_size,
    pdi_params,
    pdi_require_explicit_params,
    pdi_require_strict_baseline,
    pdi_primary_endpoint,
    nas_params,
    srpi_params,
    srpi_require_explicit_params,
    out,
    modality,
    dataset_id,
    data_origin,
    dataset_role,
    provenance_label,
    cfg,
    report_doc,
    run_replication_flag,
    hardware_target,
):
    from impact_pipeline.run_synergy_ci import load_onsets, run_s_ci

    stage = str(hunter_stage or "build-campaign").strip().lower()
    campaign_dir = Path(
        hunter_campaign_dir if hunter_campaign_dir is not None else (cache_dir / "hunter_iim_campaign")
    ).resolve()

    if stage == "build-campaign":
        onsets = None
        needs_onsets = (mpc_metrics is None) or bool({"RAM", "SRPI"} & set(mpc_metrics))
        if needs_onsets and bids_root is not None:
            discovered_subjects = []
            if subjects is not None:
                discovered_subjects = [str(s).replace("sub-", "").strip() for s in subjects if str(s).strip()]
            else:
                layout = BIDSLayout(bids_root, validate=False)
                discovered_subjects = sorted(layout.get(return_type='id', target='subject'))
            onsets = {
                subj: {ses: load_onsets(bids_root, subj, ses) for ses in sessions}
                for subj in discovered_subjects
            }
        context = {
            "prep_out": str(prep_out),
            "bids_root": (None if bids_root is None else str(bids_root)),
            "figdir": str(figdir),
            "cache_dir": str(cache_dir),
            "out_dir": str(out),
            "atlas": str(atlas),
            "sessions": list(sessions),
            "condition": str(condition),
            "tr": (None if metric_tr is None else float(metric_tr)),
            "subjects": (None if subjects is None else list(subjects)),
            "mpc_metrics": (None if mpc_metrics is None else list(mpc_metrics)),
            "compute_ci": bool(compute_ci),
            "dataset_id": str(dataset_id),
            "data_origin": str(data_origin),
            "dataset_role": str(dataset_role),
            "provenance_label": str(provenance_label),
            "report_doc": str(report_doc),
            "run_replication_flag": bool(run_replication_flag),
            "hardware_target": str(hardware_target),
            "pdi_params": pdi_params,
            "pdi_require_explicit_params": bool(pdi_require_explicit_params),
            "pdi_require_strict_baseline": bool(pdi_require_strict_baseline),
            "pdi_primary_endpoint": str(pdi_primary_endpoint),
            "nas_params": nas_params,
            "srpi_params": srpi_params,
            "srpi_require_explicit_params": bool(srpi_require_explicit_params),
        }
        manifest = prepare_hunter_campaign(
            data_dir=prep_out,
            atlas=atlas,
            sessions=sessions,
            condition=condition,
            stimulus_onsets=onsets,
            subjects=subjects,
            campaign_dir=campaign_dir,
            execution_profile=execution_profile,
            iim_bins=3,
            iim_lag_trs=1,
            iim_n_parts=iim_n_parts,
            iim_max_timepoints=iim_max_timepoints,
            iim_max_nodes=iim_max_nodes,
            iim_max_mechanism_size=iim_max_mechanism_size,
            iim_max_purview_size=iim_max_purview_size,
            hardware_target=hardware_target,
            step2_context=context,
        )
        log.info(
            "Hunter campaign prepared: runs=%d phase1_tasks=%d cut_tasks=%d dir=%s",
            len(manifest.get("runs", [])),
            len(manifest.get("phase1_tasks", [])),
            len(manifest.get("cut_tasks", [])),
            campaign_dir,
        )
        return

    if stage == "phase1-shard":
        if hunter_task_index is None:
            raise ValueError("--hunter-task-index is required for hunter phase1-shard.")
        run_phase1_shard(campaign_dir, int(hunter_task_index))
        return
    if stage == "phase1-reduce":
        if hunter_run_index is None:
            raise ValueError("--hunter-run-index is required for hunter phase1-reduce.")
        run_phase1_reduce(campaign_dir, int(hunter_run_index))
        return
    if stage == "cut-shard":
        if hunter_task_index is None:
            raise ValueError("--hunter-task-index is required for hunter cut-shard.")
        run_cut_shard(campaign_dir, int(hunter_task_index))
        return
    if stage == "cut-reduce":
        if hunter_run_index is None:
            raise ValueError("--hunter-run-index is required for hunter cut-reduce.")
        run_cut_reduce(campaign_dir, int(hunter_run_index))
        return
    if stage == "finalize-pipeline":
        ctx = json.loads((campaign_dir / "campaign_manifest.json").read_text(encoding="utf-8"))["step2_context"]
        ctx_dataset_id = str(ctx.get("dataset_id", dataset_id))
        ctx_cfg = DATASET_CONFIGS.get(ctx_dataset_id, cfg)
        ctx_modality = ctx_cfg["modality"] if ctx_cfg is not None else modality
        iim_precomputed_by_path = collect_iim_results_by_path(campaign_dir)
        df, df_mean, _df_S, _df_CI, df_stats_by_theta = run_s_ci(
            prep_out=Path(ctx["prep_out"]),
            bids_root=(None if ctx["bids_root"] is None else Path(ctx["bids_root"])),
            figdir=Path(ctx["figdir"]),
            atlas=ctx["atlas"],
            sessions=tuple(ctx["sessions"]),
            thetas=np.arange(0.1, 1.0, 0.1),
            thetas_fine=np.arange(0.4, 0.81, 0.02),
            mpc_metrics=ctx["mpc_metrics"],
            compute_ci=bool(ctx["compute_ci"]),
            condition=ctx["condition"],
            tr=ctx["tr"],
            onsets=None,
            load_onsets_fn=load_onsets if ctx_modality in {"fmri", "eeg"} else None,
            iim_n_parts=iim_n_parts,
            iim_max_timepoints=iim_max_timepoints,
            iim_max_nodes=iim_max_nodes,
            iim_max_mechanism_size=iim_max_mechanism_size,
            iim_max_purview_size=iim_max_purview_size,
            iim_parallel_workers=None,
            iim_memory_target_ratio=0.90,
            iim_worker_mem_gb_estimate=3.0,
            iim_cpu_oversub_factor=3.0,
            iim_enable_parallel=False,
            iim_checkpoint_dir=None,
            iim_resume_checkpoint=False,
            iim_checkpoint_every_cuts=1,
            iim_progress_log_every_cuts=1,
            iim_use_shared_memory=False,
            iim_phase1_parallel_workers=None,
            iim_phase1_chunk_size=8,
            iim_phase1_shared_memory=False,
            pdi_params=ctx["pdi_params"],
            pdi_require_explicit_params=bool(ctx["pdi_require_explicit_params"]),
            pdi_require_strict_baseline=bool(ctx["pdi_require_strict_baseline"]),
            pdi_primary_endpoint=ctx["pdi_primary_endpoint"],
            nas_params=ctx["nas_params"],
            srpi_params=ctx["srpi_params"],
            srpi_require_explicit_params=bool(ctx["srpi_require_explicit_params"]),
            subjects=ctx["subjects"],
            iim_precomputed_by_path=iim_precomputed_by_path,
            dataset_id=ctx_dataset_id,
            data_origin=ctx.get("data_origin", REAL_DATA_ORIGIN),
            dataset_role=ctx.get("dataset_role"),
            provenance_label=ctx.get("provenance_label"),
            modality=ctx_modality,
            hardware_target=ctx.get("hardware_target", "cpu"),
        )
        df, df_mean = _apply_metric_subset(
            df,
            df_mean,
            mpc_metrics=ctx["mpc_metrics"],
            compute_ci=bool(ctx["compute_ci"]),
        )
        if 'subject' in df.columns:
            df['subject'] = df['subject'].astype(str)
        if 'subject' in df_mean.columns:
            df_mean['subject'] = df_mean['subject'].astype(str)
        _persist_step2_outputs(
            Path(ctx["cache_dir"]),
            df,
            df_mean,
            df_stats_by_theta,
            persist_primary=True,
        )
        provenance = resolve_dataset_provenance(
            repo_root=root,
            out_dir=ctx["out_dir"],
            dataset_id=ctx_dataset_id,
            data_origin=ctx.get("data_origin", REAL_DATA_ORIGIN),
        )
        df = _ensure_provenance_columns(df, provenance)
        df_mean = _ensure_provenance_columns(df_mean, provenance)
        _write_run_provenance_manifest(
            cache_dir=Path(ctx["cache_dir"]),
            provenance=provenance,
            modality=ctx_modality,
            bids_root=(None if ctx["bids_root"] is None else Path(ctx["bids_root"])),
            execution_mode="hunter",
            atlas=ctx["atlas"],
            sessions=tuple(ctx["sessions"]),
            condition=ctx["condition"],
        )
        if is_test_object_origin(provenance.data_origin):
            _export_test_object_metric_bank(
                df=df,
                df_mean=df_mean,
                cache_dir=Path(ctx["cache_dir"]),
                provenance=provenance,
                modality=ctx_modality,
                atlas=ctx["atlas"],
                sessions=tuple(ctx["sessions"]),
                condition=ctx["condition"],
            )
        _postprocess_after_step2(
            df=df,
            df_mean=df_mean,
            prep_out=Path(ctx["prep_out"]),
            bids_root=(None if ctx["bids_root"] is None else Path(ctx["bids_root"])),
            figdir=Path(ctx["figdir"]),
            atlas=ctx["atlas"],
            sessions=tuple(ctx["sessions"]),
            compute_ci=bool(ctx["compute_ci"]),
            modality=ctx_modality,
            dataset_id=ctx_dataset_id,
            cfg=ctx_cfg,
            subjects=ctx["subjects"],
            out=Path(ctx["out_dir"]),
            report_doc=ctx["report_doc"],
            pdi_params=ctx["pdi_params"],
            pdi_require_explicit_params=bool(ctx["pdi_require_explicit_params"]),
            pdi_require_strict_baseline=bool(ctx["pdi_require_strict_baseline"]),
            pdi_primary_endpoint=ctx["pdi_primary_endpoint"],
            nas_params=ctx["nas_params"],
            srpi_params=ctx["srpi_params"],
            srpi_require_explicit_params=bool(ctx["srpi_require_explicit_params"]),
            run_replication_flag=bool(ctx["run_replication_flag"]),
            df_stats_by_theta=df_stats_by_theta,
        )
        return

    raise ValueError(f"Unknown hunter stage '{hunter_stage}'.")


def main(
    out_dir,
    subjects=None,
    reuse_step2=False,
    mpc_metrics=None,
    compute_ci=True,
    dataset_id="ds003171",
    bids_root_override=None,
    run_fmriprep=False,
    run_preprocessing_flag=False,
    run_replication_flag=False,
    atlas_override=None,
    sessions_override=None,
    condition_override=None,
    tr_override=None,
    eeg_target_sfreq=250.0,
    eeg_l_freq=0.5,
    eeg_h_freq=45.0,
    eeg_max_duration_sec=120.0,
    iim_n_parts_override=None,
    iim_max_timepoints_override=None,
    iim_max_nodes_override=None,
    iim_max_mechanism_size_override=None,
    iim_max_purview_size_override=None,
    iim_parallel_workers_override=None,
    iim_memory_target_ratio_override=None,
    iim_worker_mem_gb_estimate_override=None,
    iim_cpu_oversub_factor_override=None,
    iim_phase1_parallel_workers_override=None,
    iim_phase1_chunk_size_override=None,
    iim_checkpoint_dir_override=None,
    disable_iim_checkpoint_resume=False,
    iim_checkpoint_every_cuts=1,
    iim_progress_log_every_cuts=1,
    disable_iim_shared_memory=False,
    disable_iim_phase1_shared_memory=False,
    disable_iim_parallel=False,
    execution_mode="local",
    hardware_target="cpu",
    data_origin=REAL_DATA_ORIGIN,
    hunter_stage=None,
    hunter_campaign_dir=None,
    hunter_task_index=None,
    hunter_run_index=None,
):
    from impact_pipeline.run_synergy_ci import load_onsets, run_s_ci

    _assert_expected_runtime_env()

    catalog_entry = get_report_dataset(dataset_id)
    cfg = DATASET_CONFIGS.get(dataset_id)
    if cfg is None:
        if catalog_entry is not None:
            raise ValueError(
                f"Dataset '{dataset_id}' is available in the local dataset catalog "
                f"('{catalog_entry.title}') but does not have full pipeline mappings. "
                "Use compatible preprocessed outputs for modular analyses or add "
                "dataset-specific support before running the full pipeline."
            )
        raise ValueError(
            f"Unknown dataset_id='{dataset_id}'. "
            f"Known datasets: {sorted(DATASET_CONFIGS.keys())}"
        )
    if (
        catalog_entry is not None
        and not bool(catalog_entry.pipeline_ready)
        and (bool(run_preprocessing_flag) or bool(run_fmriprep))
    ):
        raise ValueError(
            f"Dataset '{dataset_id}' is cataloged locally ('{catalog_entry.title}') but is not "
            "pipeline-enabled for raw preprocessing/fMRIPrep in this repository yet. "
            "Reuse existing compatible preprocessed outputs for modular analyses, or add "
            "dataset-specific preprocessing/session support first."
        )
    modality = cfg["modality"]
    execution_profile = get_execution_profile(execution_mode)
    try:
        hardware_backend = configure_process_for_hardware(hardware_target)
    except HardwareBackendError as exc:
        raise RuntimeError(f"Hardware target unavailable: {exc}") from exc
    log.info("Hardware target: %s", backend_summary(hardware_backend))

    provenance = resolve_dataset_provenance(
        repo_root=root,
        out_dir=out_dir,
        dataset_id=dataset_id,
        data_origin=data_origin,
    )
    out = provenance.effective_out_dir
    out.mkdir(parents=True, exist_ok=True)
    figdir = out / 'pipe_figures'
    figdir.mkdir(exist_ok=True)
    cache_dir = out / 'cache'
    cache_dir.mkdir(exist_ok=True)
    
    fmriprep_out = out / 'fmriprep'
    prep_out     = out / 'preprocessed'
    freesurf_out = out / 'freesurfer'
    workdir      = out / 'work'
    bids_root = (
        Path(bids_root_override)
        if bids_root_override is not None
        else _first_existing_path(*cfg["bids_candidates"])
    )
    hunter_stage_key = str(hunter_stage or "build-campaign").strip().lower()
    if execution_profile.name == "hunter" and hunter_stage_key in {
        "phase1-shard",
        "phase1-reduce",
        "cut-shard",
        "cut-reduce",
        "finalize-pipeline",
    }:
        _run_hunter_stage(
            hunter_stage=hunter_stage_key,
            hunter_campaign_dir=hunter_campaign_dir,
            hunter_task_index=hunter_task_index,
            hunter_run_index=hunter_run_index,
            execution_profile=execution_profile,
            prep_out=prep_out,
            bids_root=bids_root,
            figdir=figdir,
            cache_dir=cache_dir,
            atlas=atlas_override or cfg["atlas"],
            sessions=tuple(sessions_override) if sessions_override else tuple(cfg["sessions"]),
            condition=condition_override or cfg["condition"],
            metric_tr=tr_override,
            subjects=subjects,
            mpc_metrics=mpc_metrics,
            compute_ci=compute_ci,
            iim_n_parts=iim_n_parts_override,
            iim_max_timepoints=iim_max_timepoints_override,
            iim_max_nodes=iim_max_nodes_override,
            iim_max_mechanism_size=iim_max_mechanism_size_override,
            iim_max_purview_size=iim_max_purview_size_override,
            pdi_params=dict(cfg.get("pdi_params", {})),
            pdi_require_explicit_params=bool(cfg.get("pdi_require_explicit_params", True)),
            pdi_require_strict_baseline=bool(cfg.get("pdi_require_strict_baseline", True)),
            pdi_primary_endpoint=str(cfg.get("pdi_primary_endpoint", "anchor")),
            nas_params=dict(cfg.get("nas_params", {})),
            srpi_params=dict(cfg.get("srpi_params", {})),
            srpi_require_explicit_params=bool(cfg.get("srpi_require_explicit_params", True)),
            out=out,
            modality=modality,
            dataset_id=dataset_id,
            data_origin=provenance.data_origin,
            dataset_role=provenance.dataset_role,
            provenance_label=provenance.provenance_label,
            cfg=cfg,
            report_doc=f"IMPaCT_Empirical_Validation_{dataset_id}.docx",
            run_replication_flag=run_replication_flag,
            hardware_target=hardware_backend.requested,
        )
        return
    if not bids_root.exists():
        raise FileNotFoundError(
            f"Missing {dataset_id} dataset at '{bids_root}'. "
            "Provide --bids-root explicitly or download the dataset first."
        )
    _write_run_provenance_manifest(
        cache_dir=cache_dir,
        provenance=provenance,
        modality=modality,
        bids_root=bids_root,
        execution_mode=execution_mode,
        atlas=atlas_override or cfg["atlas"],
        sessions=tuple(sessions_override) if sessions_override else tuple(cfg["sessions"]),
        condition=condition_override or cfg["condition"],
    )

    atlas = atlas_override or cfg["atlas"]
    sessions = tuple(sessions_override) if sessions_override else tuple(cfg["sessions"])
    condition = condition_override or cfg["condition"]
    metric_tr = tr_override
    if metric_tr is None and modality == "eeg":
        if eeg_target_sfreq is None or float(eeg_target_sfreq) <= 0:
            raise ValueError("eeg_target_sfreq must be > 0 for EEG metric computation.")
        metric_tr = 1.0 / float(eeg_target_sfreq)
    def _opt_int(v):
        return None if v is None else int(v)

    iim_n_parts = (
        _opt_int(iim_n_parts_override)
        if iim_n_parts_override is not None
        else _opt_int(cfg.get("iim_n_parts", None))
    )
    iim_max_nodes = (
        _opt_int(iim_max_nodes_override)
        if iim_max_nodes_override is not None
        else _opt_int(cfg.get("iim_max_nodes", None))
    )
    iim_max_mechanism_size = (
        _opt_int(iim_max_mechanism_size_override)
        if iim_max_mechanism_size_override is not None
        else _opt_int(cfg.get("iim_max_mechanism_size", None))
    )
    iim_max_purview_size = (
        _opt_int(iim_max_purview_size_override)
        if iim_max_purview_size_override is not None
        else _opt_int(cfg.get("iim_max_purview_size", None))
    )
    iim_parallel_workers = (
        _opt_int(iim_parallel_workers_override)
        if iim_parallel_workers_override is not None
        else _opt_int(cfg.get("iim_parallel_workers", None))
    )
    iim_memory_target_ratio = (
        float(iim_memory_target_ratio_override)
        if iim_memory_target_ratio_override is not None
        else float(cfg.get("iim_memory_target_ratio", 0.90))
    )
    iim_worker_mem_gb_estimate = (
        float(iim_worker_mem_gb_estimate_override)
        if iim_worker_mem_gb_estimate_override is not None
        else float(cfg.get("iim_worker_mem_gb_estimate", 3.0))
    )
    iim_cpu_oversub_factor = (
        float(iim_cpu_oversub_factor_override)
        if iim_cpu_oversub_factor_override is not None
        else float(cfg.get("iim_cpu_oversub_factor", 3.0))
    )
    iim_phase1_parallel_workers = (
        _opt_int(iim_phase1_parallel_workers_override)
        if iim_phase1_parallel_workers_override is not None
        else _opt_int(cfg.get("iim_phase1_parallel_workers", None))
    )
    iim_phase1_chunk_size = (
        _opt_int(iim_phase1_chunk_size_override)
        if iim_phase1_chunk_size_override is not None
        else _opt_int(cfg.get("iim_phase1_chunk_size", 8))
    )
    if iim_phase1_chunk_size is None:
        iim_phase1_chunk_size = 8
    if int(iim_phase1_chunk_size) < 1:
        raise ValueError("iim_phase1_chunk_size must be >= 1")
    iim_enable_parallel = not bool(disable_iim_parallel)
    iim_use_shared_memory = not bool(disable_iim_shared_memory)
    iim_phase1_shared_memory = (
        bool(cfg.get("iim_phase1_shared_memory", True))
        and not bool(disable_iim_phase1_shared_memory)
    )
    pdi_params = dict(cfg.get("pdi_params", {}))
    pdi_require_explicit_params = bool(cfg.get("pdi_require_explicit_params", True))
    pdi_require_strict_baseline = bool(cfg.get("pdi_require_strict_baseline", True))
    pdi_primary_endpoint = str(cfg.get("pdi_primary_endpoint", "anchor"))
    nas_params = dict(cfg.get("nas_params", {}))
    srpi_params = dict(cfg.get("srpi_params", {}))
    srpi_require_explicit_params = bool(cfg.get("srpi_require_explicit_params", True))
    iim_resume_checkpoint = not bool(disable_iim_checkpoint_resume)
    iim_checkpoint_dir = (
        str(Path(iim_checkpoint_dir_override))
        if iim_checkpoint_dir_override is not None
        else str(cache_dir / "iim_checkpoints")
    )
    if int(iim_checkpoint_every_cuts) < 1:
        raise ValueError("iim_checkpoint_every_cuts must be >= 1")
    if int(iim_progress_log_every_cuts) < 1:
        raise ValueError("iim_progress_log_every_cuts must be >= 1")
    iim_max_timepoints = (
        None
        if iim_max_timepoints_override is None and cfg.get("iim_max_timepoints", None) is None
        else int(iim_max_timepoints_override if iim_max_timepoints_override is not None else cfg.get("iim_max_timepoints"))
    )
    report_doc = f"IMPaCT_Empirical_Validation_{dataset_id}.docx"
    cuts_mode = "all" if iim_n_parts is None else str(iim_n_parts)
    log.info(
        "IIM configuration: cuts=%s, max_nodes=%s, max_mechanism_size=%s, max_purview_size=%s, "
        "max_timepoints=%s, parallel=%s, workers=%s, mem_target=%.2f, worker_mem_est=%.2fGB",
        cuts_mode,
        "all" if iim_max_nodes is None else iim_max_nodes,
        "all" if iim_max_mechanism_size is None else iim_max_mechanism_size,
        "all" if iim_max_purview_size is None else iim_max_purview_size,
        "all" if iim_max_timepoints is None else iim_max_timepoints,
        iim_enable_parallel,
        "auto" if iim_parallel_workers is None else iim_parallel_workers,
        iim_memory_target_ratio,
        iim_worker_mem_gb_estimate,
    )
    log.info(
        "IIM planner oversub factor: %.2f (soft worker cap = cpu_count * factor)",
        iim_cpu_oversub_factor,
    )
    log.info(
        "IIM intra-task phase config: workers=%s chunk_size=%d shared_memory=%s",
        "off/auto" if iim_phase1_parallel_workers is None else str(int(iim_phase1_parallel_workers)),
        int(iim_phase1_chunk_size),
        bool(iim_phase1_shared_memory),
    )
    log.info(
        "IIM checkpoint config: dir=%s, resume=%s, checkpoint_every_cuts=%d, progress_every_cuts=%d, use_shared_memory=%s",
        iim_checkpoint_dir,
        iim_resume_checkpoint,
        int(iim_checkpoint_every_cuts),
        int(iim_progress_log_every_cuts),
        iim_use_shared_memory,
    )
    
    # 0. RUN FMRIPrep ON MISSING SUBJECTS
    if run_fmriprep:
        if not cfg["supports_fmriprep"]:
            raise ValueError(
                f"fMRIPrep is not applicable to dataset '{dataset_id}' "
                f"(modality={modality})."
            )
        ensure_fmriprep(
            bids_dir=str(bids_root),
            fmriprep_out=str(fmriprep_out),
            work_dir=str(workdir),
            fs_license=str(_resolve_freesurfer_license_path()),
            freesurf_out=str(freesurf_out),
            subjects=subjects
        )

    # 1. PREPROCESSING
    if run_preprocessing_flag:
        if modality == "fmri":
            from impact_pipeline.preprocessing import run_preprocessing
            log.info("1/9 Preprocessing %s (fMRI)", dataset_id)
            run_preprocessing(
                bids_root=str(bids_root),
                fmriprep_deriv=str(out/'fmriprep'),
                out_root=str(prep_out),
                subjects=subjects
            )
        elif modality == "eeg":
            log.info("1/9 Preprocessing %s (EEG)", dataset_id)
            eeg_summary = run_preprocessing_eeg(
                bids_root=str(bids_root),
                out_root=str(prep_out),
                subjects=subjects,
                session_rules=cfg.get("eeg_session_rules"),
                condition_label=condition,
                atlas_key=atlas,
                target_sfreq=float(eeg_target_sfreq),
                l_freq=float(eeg_l_freq) if eeg_l_freq is not None else None,
                h_freq=float(eeg_h_freq) if eeg_h_freq is not None else None,
                max_duration_sec=float(eeg_max_duration_sec) if eeg_max_duration_sec is not None else None,
            )
            _write_eeg_preprocessing_summary(cache_dir, eeg_summary)
            summ = eeg_summary.get("summary", {})
            log.info(
                "EEG preprocessing summary: subjects_requested=%s processed=%s skipped=%s missing_records=%s written_runs=%s",
                summ.get("subjects_requested", "na"),
                summ.get("subjects_processed", "na"),
                summ.get("subjects_skipped", "na"),
                summ.get("missing_file_records", "na"),
                summ.get("written_runs", "na"),
            )
        else:
            raise ValueError(f"Unsupported modality '{modality}'")

    # Step 2 depends on preprocessing outputs even when Step 1 is skipped.
    if not prep_out.exists() or not any(prep_out.iterdir()):
        raise FileNotFoundError(
            f"Missing preprocessing outputs at '{prep_out}'. "
            "Run with --run-preprocessing (and ensure required derivatives are available) "
            "or provide an out-dir that already contains a populated 'preprocessed' folder."
        )

    if execution_profile.name == "hunter":
        _run_hunter_stage(
            hunter_stage=hunter_stage,
            hunter_campaign_dir=hunter_campaign_dir,
            hunter_task_index=hunter_task_index,
            hunter_run_index=hunter_run_index,
            execution_profile=execution_profile,
            prep_out=prep_out,
            bids_root=bids_root,
            figdir=figdir,
            cache_dir=cache_dir,
            atlas=atlas,
            sessions=sessions,
            condition=condition,
            metric_tr=metric_tr,
            subjects=subjects,
            mpc_metrics=mpc_metrics,
            compute_ci=compute_ci,
            iim_n_parts=iim_n_parts,
            iim_max_timepoints=iim_max_timepoints,
            iim_max_nodes=iim_max_nodes,
            iim_max_mechanism_size=iim_max_mechanism_size,
            iim_max_purview_size=iim_max_purview_size,
            pdi_params=pdi_params,
            pdi_require_explicit_params=pdi_require_explicit_params,
            pdi_require_strict_baseline=pdi_require_strict_baseline,
            pdi_primary_endpoint=pdi_primary_endpoint,
            nas_params=nas_params,
            srpi_params=srpi_params,
            srpi_require_explicit_params=srpi_require_explicit_params,
            out=out,
            modality=modality,
            dataset_id=dataset_id,
            cfg=cfg,
            report_doc=report_doc,
            run_replication_flag=run_replication_flag,
            hardware_target=hardware_backend.requested,
        )
        return

    # 2. SYNERGY & CI
    thetas      = np.arange(0.1, 1.0, 0.1)
    thetas_fine = np.arange(0.4, 0.81, 0.02)
    step2_df_path = cache_dir / "step2_df.csv"
    step2_df_mean_path = cache_dir / "step2_df_mean.csv"
    step2_theta_stats_path = cache_dir / "step2_theta_stats.csv"
    if reuse_step2:
        missing = [
            str(p)
            for p in (step2_df_path, step2_df_mean_path, step2_theta_stats_path)
            if not p.exists()
        ]
        if missing:
            raise FileNotFoundError(
                "Cannot reuse step 2 because cache files are missing: "
                + ", ".join(missing)
            )
        log.info("2/9 Reusing cached step-2 metric tables")
        df = pd.read_csv(step2_df_path)
        df_mean = pd.read_csv(step2_df_mean_path)
        if 'subject' in df.columns:
            df['subject'] = df['subject'].astype(str)
        if 'subject' in df_mean.columns:
            df_mean['subject'] = df_mean['subject'].astype(str)
        df, df_mean = _apply_metric_subset(
            df,
            df_mean,
            mpc_metrics=mpc_metrics,
            compute_ci=compute_ci,
        )
        df = _ensure_provenance_columns(df, provenance)
        df_mean = _ensure_provenance_columns(df_mean, provenance)
        df_stats_by_theta = pd.read_csv(step2_theta_stats_path).set_index('theta')
        df_S = df_mean.pivot(index='subject', columns='session', values='S')
        df_CI = df_mean.pivot(index='subject', columns='session', values='CI') if 'CI' in df_mean.columns else None
    else:
        df, df_mean, df_S, df_CI, df_stats_by_theta = run_s_ci(
            prep_out=prep_out,
            bids_root=bids_root,
            figdir=figdir,
            atlas=atlas,
            sessions=sessions,
            thetas=thetas,
            thetas_fine=thetas_fine,
            mpc_metrics=mpc_metrics,
            compute_ci=compute_ci,
            condition=condition,
            tr=metric_tr,
            onsets=None,
            load_onsets_fn=load_onsets if modality in {"fmri", "eeg"} else None,
            iim_n_parts=iim_n_parts,
            iim_max_timepoints=iim_max_timepoints,
            iim_max_nodes=iim_max_nodes,
            iim_max_mechanism_size=iim_max_mechanism_size,
            iim_max_purview_size=iim_max_purview_size,
            iim_parallel_workers=iim_parallel_workers,
            iim_memory_target_ratio=iim_memory_target_ratio,
            iim_worker_mem_gb_estimate=iim_worker_mem_gb_estimate,
            iim_cpu_oversub_factor=iim_cpu_oversub_factor,
            iim_enable_parallel=iim_enable_parallel,
            iim_checkpoint_dir=iim_checkpoint_dir,
            iim_resume_checkpoint=iim_resume_checkpoint,
            iim_checkpoint_every_cuts=int(iim_checkpoint_every_cuts),
            iim_progress_log_every_cuts=int(iim_progress_log_every_cuts),
            iim_use_shared_memory=iim_use_shared_memory,
            iim_phase1_parallel_workers=iim_phase1_parallel_workers,
            iim_phase1_chunk_size=int(iim_phase1_chunk_size),
            iim_phase1_shared_memory=bool(iim_phase1_shared_memory),
            pdi_params=pdi_params,
            pdi_require_explicit_params=pdi_require_explicit_params,
            pdi_require_strict_baseline=pdi_require_strict_baseline,
            pdi_primary_endpoint=pdi_primary_endpoint,
            nas_params=nas_params,
            srpi_params=srpi_params,
            srpi_require_explicit_params=srpi_require_explicit_params,
            subjects=subjects,
            dataset_id=dataset_id,
            data_origin=provenance.data_origin,
            dataset_role=provenance.dataset_role,
            provenance_label=provenance.provenance_label,
            modality=modality,
            hardware_target=hardware_backend.requested,
        )
        df, df_mean = _apply_metric_subset(
            df,
            df_mean,
            mpc_metrics=mpc_metrics,
            compute_ci=compute_ci,
        )
        df = _ensure_provenance_columns(df, provenance)
        df_mean = _ensure_provenance_columns(df_mean, provenance)
        if 'subject' in df.columns:
            df['subject'] = df['subject'].astype(str)
        if 'subject' in df_mean.columns:
            df_mean['subject'] = df_mean['subject'].astype(str)
        df_S = df_mean.pivot(index='subject', columns='session', values='S')
        df_CI = df_mean.pivot(index='subject', columns='session', values='CI') if 'CI' in df_mean.columns else None
        df.to_csv(step2_df_path, index=False)
        df_mean.to_csv(step2_df_mean_path, index=False)
        df_stats_by_theta.reset_index().to_csv(step2_theta_stats_path, index=False)

    _persist_step2_outputs(
        cache_dir,
        df,
        df_mean,
        df_stats_by_theta,
        persist_primary=not bool(reuse_step2),
    )
    if is_test_object_origin(provenance.data_origin):
        _export_test_object_metric_bank(
            df=df,
            df_mean=df_mean,
            cache_dir=cache_dir,
            provenance=provenance,
            modality=modality,
            atlas=atlas,
            sessions=sessions,
            condition=condition,
        )
    _postprocess_after_step2(
        df=df,
        df_mean=df_mean,
        prep_out=prep_out,
        bids_root=bids_root,
        figdir=figdir,
        atlas=atlas,
        sessions=sessions,
        compute_ci=compute_ci,
        modality=modality,
        dataset_id=dataset_id,
        cfg=cfg,
        subjects=subjects,
        out=out,
        report_doc=report_doc,
        pdi_params=pdi_params,
        pdi_require_explicit_params=pdi_require_explicit_params,
        pdi_require_strict_baseline=pdi_require_strict_baseline,
        pdi_primary_endpoint=pdi_primary_endpoint,
        nas_params=nas_params,
        srpi_params=srpi_params,
        srpi_require_explicit_params=srpi_require_explicit_params,
        run_replication_flag=run_replication_flag,
        df_stats_by_theta=df_stats_by_theta,
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run IMPaCT pipeline")
    parser.add_argument('--out-dir', default='outputs/scratch',
                        help="Root folder for outputs. For non-ds003171 datasets, a dataset subfolder is auto-created.")
    parser.add_argument(
        '--data-origin',
        default=REAL_DATA_ORIGIN,
        choices=('real', 'dummy'),
        help=(
            "Mark the dataset as real study data or synthetic validation data. "
            "Synthetic validation runs are stored separately."
        ),
    )
    parser.add_argument(
        '--dataset-id',
        default='ds003171',
        help=f"Dataset identifier. Known: {', '.join(sorted(DATASET_CONFIGS.keys()))}",
    )
    parser.add_argument(
        '--bids-root',
        default=None,
        help="Optional explicit BIDS root override.",
    )
    parser.add_argument(
        '--subjects', nargs='+',
        help="Optional subject IDs to restrict preprocessing and (if enabled) fMRIPrep."
    )
    parser.add_argument(
        '--run-fmriprep',
        action='store_true',
        help="Run fMRIPrep step (only applicable to fMRI datasets).",
    )
    parser.add_argument(
        '--run-preprocessing',
        action='store_true',
        help="Run preprocessing step for the selected dataset modality.",
    )
    parser.add_argument(
        '--run-replication',
        action='store_true',
        help="Run replication step if configured for selected dataset.",
    )
    parser.add_argument(
        '--reuse-step2',
        action='store_true',
        help="Reuse cached step-2 tables from <out-dir>/cache and skip recomputing core metrics."
    )
    parser.add_argument(
        '--execution-mode',
        default='local',
        choices=('local', 'hunter'),
        help="Execution backend. 'local' keeps the current workstation pipeline; 'hunter' uses distributed IIM campaign stages.",
    )
    parser.add_argument(
        '--hardware-target',
        default='cpu',
        choices=('cpu', 'auto', 'gpu', 'hunter-apu'),
        help=(
            "Compute hardware for MPC kernels. "
            "'cpu' uses NumPy/SciPy, 'auto' uses CuPy when available and otherwise CPU, "
            "'gpu' requires a visible CuPy GPU, and 'hunter-apu' requires ROCm/HIP CuPy."
        ),
    )
    parser.add_argument(
        '--hunter-stage',
        default=None,
        help=(
            "Hunter stage selector. "
            "Supported: build-campaign, phase1-shard, phase1-reduce, cut-shard, cut-reduce, finalize-pipeline."
        ),
    )
    parser.add_argument(
        '--hunter-campaign-dir',
        default=None,
        help="Optional Hunter campaign directory override (default: <out-dir>/cache/hunter_iim_campaign).",
    )
    parser.add_argument(
        '--hunter-task-index',
        type=int,
        default=None,
        help="Array-task index for Hunter shard stages.",
    )
    parser.add_argument(
        '--hunter-run-index',
        type=int,
        default=None,
        help="Run index for Hunter reduction stages.",
    )
    parser.add_argument(
        '--mpc-metrics',
        nargs='+',
        default=None,
        help=(
            "Subset of MPC metrics to compute (choices: RAM PDI NAS IIM SRPI). "
            "If omitted, all MPC metrics are computed."
        ),
    )
    parser.add_argument(
        '--no-ci',
        action='store_true',
        help="Disable CI computation even if all MPC components are available.",
    )
    parser.add_argument(
        '--atlas',
        default=None,
        help="Override atlas/time-series key used in preprocessed filenames.",
    )
    parser.add_argument(
        '--sessions',
        nargs='+',
        default=None,
        help="Override session labels (default from dataset config).",
    )
    parser.add_argument(
        '--condition',
        default=None,
        help="Override condition folder name under preprocessed/<subject>/<session>/<condition>.",
    )
    parser.add_argument(
        '--tr',
        type=float,
        default=None,
        help="Optional TR/sample interval override for metric computation.",
    )
    parser.add_argument('--eeg-target-sfreq', type=float, default=250.0)
    parser.add_argument('--eeg-l-freq', type=float, default=0.5)
    parser.add_argument('--eeg-h-freq', type=float, default=45.0)
    parser.add_argument('--eeg-max-duration-sec', type=float, default=120.0)
    parser.add_argument(
        '--iim-n-parts',
        type=int,
        default=None,
        help="Override number of evaluated system cuts for IIM MIP search (default per dataset; None means exhaustive).",
    )
    parser.add_argument(
        '--iim-max-timepoints',
        type=int,
        default=None,
        help="Override max timepoints used for IIM (uniform decimation; default per dataset).",
    )
    parser.add_argument(
        '--iim-max-nodes',
        type=int,
        default=None,
        help="Override max nodes used for IIM reduced subsystem (default per dataset).",
    )
    parser.add_argument(
        '--iim-max-mechanism-size',
        type=int,
        default=None,
        help="Override maximum mechanism size used in IIM Ψ computation.",
    )
    parser.add_argument(
        '--iim-max-purview-size',
        type=int,
        default=None,
        help="Override maximum purview size used in IIM Ψ computation.",
    )
    parser.add_argument(
        '--iim-parallel-workers',
        type=int,
        default=None,
        help="Override number of parallel IIM workers (default: auto planner).",
    )
    parser.add_argument(
        '--iim-memory-target-ratio',
        type=float,
        default=None,
        help="Target fraction of total system RAM for IIM worker planning (default per dataset).",
    )
    parser.add_argument(
        '--iim-worker-mem-gb-estimate',
        type=float,
        default=None,
        help="Estimated RAM footprint (GB) per IIM worker for auto planning.",
    )
    parser.add_argument(
        '--iim-cpu-oversub-factor',
        type=float,
        default=None,
        help="Soft multiplier on CPU count when auto-planning IIM workers (allows controlled oversubscription).",
    )
    parser.add_argument(
        '--iim-phase1-parallel-workers',
        type=int,
        default=None,
        help=(
            "Intra-task workers per IIM Ψ computation (phase-1 and per-cut Ψ in phase-2). "
            "None/1 disables intra-task parallelization."
        ),
    )
    parser.add_argument(
        '--iim-phase1-chunk-size',
        type=int,
        default=None,
        help="Mechanisms per intra-task work chunk (default from dataset config).",
    )
    parser.add_argument(
        '--iim-checkpoint-dir',
        default=None,
        help="Directory for per-run IIM checkpoint JSON files (default: <out-dir>/cache/iim_checkpoints).",
    )
    parser.add_argument(
        '--no-iim-resume-checkpoint',
        action='store_true',
        help="Disable resuming from existing IIM checkpoint files.",
    )
    parser.add_argument(
        '--iim-checkpoint-every-cuts',
        type=int,
        default=1,
        help="Write IIM checkpoint after every N newly completed cuts.",
    )
    parser.add_argument(
        '--iim-progress-log-every-cuts',
        type=int,
        default=1,
        help="Log IIM cut-level progress every N completed cuts.",
    )
    parser.add_argument(
        '--disable-iim-shared-memory',
        action='store_true',
        help="Disable shared-memory staging of IIM worker input arrays.",
    )
    parser.add_argument(
        '--disable-iim-phase1-shared-memory',
        action='store_true',
        help="Disable shared-memory staging for intra-task phase parallelization.",
    )
    parser.add_argument(
        '--disable-iim-parallel',
        action='store_true',
        help="Disable parallel IIM workers and force sequential IIM computation.",
    )
    args = parser.parse_args()
    main(
        args.out_dir,
        subjects=args.subjects,
        reuse_step2=args.reuse_step2,
        mpc_metrics=args.mpc_metrics,
        compute_ci=not args.no_ci,
        data_origin=args.data_origin,
        dataset_id=args.dataset_id,
        bids_root_override=args.bids_root,
        run_fmriprep=args.run_fmriprep or RUN_FMRIPREP,
        run_preprocessing_flag=args.run_preprocessing or RUN_PREPROCESSING,
        run_replication_flag=args.run_replication or RUN_REPLICATION,
        atlas_override=args.atlas,
        sessions_override=args.sessions,
        condition_override=args.condition,
        tr_override=args.tr,
        eeg_target_sfreq=args.eeg_target_sfreq,
        eeg_l_freq=args.eeg_l_freq,
        eeg_h_freq=args.eeg_h_freq,
        eeg_max_duration_sec=args.eeg_max_duration_sec,
        iim_n_parts_override=args.iim_n_parts,
        iim_max_timepoints_override=args.iim_max_timepoints,
        iim_max_nodes_override=args.iim_max_nodes,
        iim_max_mechanism_size_override=args.iim_max_mechanism_size,
        iim_max_purview_size_override=args.iim_max_purview_size,
        iim_parallel_workers_override=args.iim_parallel_workers,
        iim_memory_target_ratio_override=args.iim_memory_target_ratio,
        iim_worker_mem_gb_estimate_override=args.iim_worker_mem_gb_estimate,
        iim_cpu_oversub_factor_override=args.iim_cpu_oversub_factor,
        iim_phase1_parallel_workers_override=args.iim_phase1_parallel_workers,
        iim_phase1_chunk_size_override=args.iim_phase1_chunk_size,
        iim_checkpoint_dir_override=args.iim_checkpoint_dir,
        disable_iim_checkpoint_resume=args.no_iim_resume_checkpoint,
        iim_checkpoint_every_cuts=args.iim_checkpoint_every_cuts,
        iim_progress_log_every_cuts=args.iim_progress_log_every_cuts,
        disable_iim_shared_memory=args.disable_iim_shared_memory,
        disable_iim_phase1_shared_memory=args.disable_iim_phase1_shared_memory,
        disable_iim_parallel=args.disable_iim_parallel,
        execution_mode=args.execution_mode,
        hardware_target=args.hardware_target,
        hunter_stage=args.hunter_stage,
        hunter_campaign_dir=args.hunter_campaign_dir,
        hunter_task_index=args.hunter_task_index,
        hunter_run_index=args.hunter_run_index,
    )
