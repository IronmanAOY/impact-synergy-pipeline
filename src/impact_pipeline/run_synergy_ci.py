#!/usr/bin/env python
import glob
import json
import logging
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

from bids import BIDSLayout
from impact_pipeline.provenance import (
    PROVENANCE_COLUMNS,
    REAL_DATA_ORIGIN,
)
from impact_pipeline.synergy_ci import compute_synergy_ci
from pathlib import Path

log = logging.getLogger("pipeline")
_SELF_RE = re.compile(
    r"\bself\b|own|myname|subject.?name|participant.?name|\bme\b|\bmy\b",
    re.I,
)
_NONSELF_RE = re.compile(
    r"non[-_ ]?self|other|stranger|another|control|third.?person",
    re.I,
)


def _infer_sample_interval_seconds(bids_root):
    """
    Infer sampling interval from BIDS sidecars.

    Priority:
      1) fMRI RepetitionTime from *_bold.json
      2) EEG SamplingFrequency from *_eeg.json
      3) no fallback
    """
    if bids_root is None:
        return None
    root = Path(bids_root)
    if not root.exists():
        return None

    fmri_sidecars = glob.glob(str(root / "sub-*/func/*_bold.json"))
    if fmri_sidecars:
        with open(fmri_sidecars[0], "r", encoding="utf-8") as f:
            sidecar = json.load(f)
        tr = sidecar.get("RepetitionTime")
        if tr is not None:
            tr = float(tr)
            if np.isfinite(tr) and tr > 0:
                return tr

    eeg_sidecars = glob.glob(str(root / "sub-*/eeg/*_eeg.json"))
    for fn in eeg_sidecars:
        with open(fn, "r", encoding="utf-8") as f:
            sidecar = json.load(f)
        sfreq = sidecar.get("SamplingFrequency")
        if sfreq is None:
            continue
        sfreq = float(sfreq)
        if np.isfinite(sfreq) and sfreq > 0:
            return 1.0 / sfreq

    return None

def _extract_run_id_from_name(fname: str):
    m = re.search(r"_run-0*([0-9]+)", str(fname))
    if not m:
        return None
    return str(int(m.group(1)))


def _resolve_events_file(bids_root, subject, session, condition="audio"):
    """
    Resolve an events.tsv for either fMRI or EEG sessions.

    Search order is fMRI patterns first, then EEG patterns.
    """
    subj_root = Path(bids_root) / f"sub-{subject}"
    session_key = str(session).strip().lower()
    condition_key = str(condition or "").strip().lower()

    fmri_dir = subj_root / "func"
    fmri_patterns = []
    if condition_key and condition_key != "audio":
        fmri_patterns.extend(
            [
                f"sub-{subject}_task-{condition_key}{session_key}*_events.tsv",
                f"sub-{subject}_task-{condition_key}_ses-{session_key}*_events.tsv",
                f"sub-{subject}_task-{condition_key}*_events.tsv",
            ]
        )
    if session_key == "deep":
        fmri_patterns.extend(
            [
                f"sub-{subject}_task-audiodeep*_events.tsv",
                f"sub-{subject}_task-audio*_events.tsv",
            ]
        )
    elif session_key == "awake":
        fmri_patterns.extend(
            [
                f"sub-{subject}_task-audioawake*_events.tsv",
                f"sub-{subject}_task-audio*_events.tsv",
            ]
        )
    else:
        fmri_patterns.extend(
            [
                f"sub-{subject}_task-audio{session_key}*_events.tsv",
                f"sub-{subject}_task-{session_key}*_events.tsv",
                f"sub-{subject}_task-audio*_events.tsv",
            ]
        )

    eeg_dir = subj_root / "eeg"
    eeg_patterns = []
    if session_key == "deep":
        eeg_patterns = [
            f"sub-{subject}_task-sed2_acq-rest*_events.tsv",
            f"sub-{subject}_task-sed_acq-rest*_events.tsv",
        ]
    elif session_key == "awake":
        eeg_patterns = [
            f"sub-{subject}_task-awake_acq-EC*_events.tsv",
            f"sub-{subject}_task-awake_acq-EO*_events.tsv",
            f"sub-{subject}_task-awake*_events.tsv",
        ]
    else:
        eeg_patterns = [
            f"sub-{subject}_task-{session_key}*_events.tsv",
        ]

    for base_dir, patterns in ((fmri_dir, fmri_patterns), (eeg_dir, eeg_patterns)):
        if not base_dir.exists():
            continue
        for pat in patterns:
            matches = sorted(base_dir.glob(pat))
            if matches:
                return matches[0]
    return None


def _events_to_ram_bundle(fn: Path):
    """
    Convert BIDS events.tsv into a structured RAM event bundle.
    """
    df = pd.read_csv(fn, sep="\t")
    if df.empty:
        return {
            "onsets": [],
            "goal_onsets": [],
            "feedback_onsets": [],
            "feedback_values": None,
            "self_onsets": [],
            "nonself_onsets": [],
        }

    # Normalize column labels (handles UTF-8 BOM in some EEG exports).
    df = df.rename(columns={c: str(c).strip().lstrip("\ufeff") for c in df.columns})
    cols_l = {str(c).lower(): c for c in df.columns}

    onset_col = cols_l.get("onset")
    if onset_col is None:
        return {
            "onsets": [],
            "goal_onsets": [],
            "feedback_onsets": [],
            "feedback_values": None,
            "self_onsets": [],
            "nonself_onsets": [],
        }

    onset = pd.to_numeric(df[onset_col], errors="coerce")
    valid_onset = onset.notna()

    trial_col = cols_l.get("trial_type")
    if trial_col is not None:
        trial = df[trial_col].astype(str)
    else:
        trial = pd.Series([""] * len(df), index=df.index, dtype=object)
    trial_l = trial.str.lower()

    stim_mask = trial_l.str.contains(r"audio|stim|tone|target|event", regex=True, na=False)
    goal_mask = trial_l.str.contains(
        r"goal|objective|intent|instruction|cue|self|name",
        regex=True,
        na=False,
    )
    feedback_mask = trial_l.str.contains(
        r"feedback|reward|error|outcome|correct|incorrect|response|choice|result",
        regex=True,
        na=False,
    )

    text_cols = [
        cols_l[k]
        for k in ("trial_type", "condition", "stimulus", "stim_file", "value")
        if k in cols_l
    ]
    if text_cols:
        txt = pd.Series([""] * len(df), index=df.index, dtype=object)
        for c in text_cols:
            txt = txt.str.cat(df[c].astype(str), sep=" ", na_rep="")
        txt = txt.str.lower()
    else:
        txt = pd.Series([""] * len(df), index=df.index, dtype=object)
    self_mask = txt.str.contains(_SELF_RE, regex=True, na=False)
    nonself_mask = txt.str.contains(_NONSELF_RE, regex=True, na=False)
    # prevent "non-self" labels from being counted as both classes
    self_mask = self_mask & (~nonself_mask)

    stim_onsets = onset[valid_onset & stim_mask].astype(float).tolist()
    if not stim_onsets:
        stim_onsets = onset[valid_onset].astype(float).tolist()

    goal_onsets = onset[valid_onset & goal_mask].astype(float).tolist()
    feedback_onsets = onset[valid_onset & feedback_mask].astype(float).tolist()
    self_onsets = onset[valid_onset & self_mask].astype(float).tolist()
    nonself_onsets = onset[valid_onset & nonself_mask].astype(float).tolist()

    # Prefer explicit prediction-error/reward style columns when available.
    feedback_values = None
    feedback_value_candidates = (
        "prediction_error",
        "pe",
        "reward",
        "outcome",
        "accuracy",
        "correct",
        "value",
        "response_time",
    )
    for cand in feedback_value_candidates:
        col = cols_l.get(cand)
        if col is None:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        if feedback_mask.any():
            vals = vals[feedback_mask]
        vals = vals[np.isfinite(vals)]
        if len(vals) >= 2 and float(vals.std(ddof=0)) > 0:
            feedback_values = vals.astype(float).tolist()
            break

    return {
        "onsets": stim_onsets,
        "goal_onsets": goal_onsets,
        "feedback_onsets": feedback_onsets,
        "feedback_values": feedback_values,
        "self_onsets": self_onsets,
        "nonself_onsets": nonself_onsets,
    }


# For each session, read event timings and RAM/SRPI sub-components from BIDS events.tsv.
def load_onsets(bids_root, subject, session, condition="audio"):
    fn = _resolve_events_file(bids_root, subject, session, condition=condition)
    if fn is None:
        log.warning(
            (
                "No events.tsv found for sub-%s session=%s; returning empty event "
                "bundle (RAM/SRPI undefined)."
            ),
            subject,
            session,
        )
        return (
            {
                "onsets": [],
                "goal_onsets": [],
                "feedback_onsets": [],
                "feedback_values": None,
                "self_onsets": [],
                "nonself_onsets": [],
            },
            None,
        )

    bundle = _events_to_ram_bundle(fn)
    run_id = _extract_run_id_from_name(fn.name)
    return bundle, run_id

def run_s_ci(
    prep_out,
    bids_root,
    figdir,
    atlas,
    sessions,
    thetas,
    thetas_fine,
    mpc_metrics=None,
    compute_ci=True,
    condition='audio',
    tr=None,
    onsets=None,
    load_onsets_fn=load_onsets,
    iim_n_parts=None,
    iim_max_timepoints=None,
    iim_max_nodes=None,
    iim_max_mechanism_size=None,
    iim_max_purview_size=None,
    iim_parallel_workers=None,
    iim_memory_target_ratio=0.90,
    iim_worker_mem_gb_estimate=3.0,
    iim_cpu_oversub_factor=3.0,
    iim_enable_parallel=True,
    iim_checkpoint_dir=None,
    iim_resume_checkpoint=True,
    iim_checkpoint_every_cuts=1,
    iim_progress_log_every_cuts=1,
    iim_use_shared_memory=True,
    iim_phase1_parallel_workers=None,
    iim_phase1_chunk_size=8,
    iim_phase1_shared_memory=True,
    pdi_params=None,
    pdi_require_explicit_params=False,
    pdi_require_strict_baseline=False,
    pdi_primary_endpoint="anchor",
    nas_params=None,
    srpi_params=None,
    srpi_require_explicit_params=True,
    subjects=None,
    iim_precomputed_by_path=None,
    dataset_id=None,
    data_origin=REAL_DATA_ORIGIN,
    dataset_role=None,
    provenance_label=None,
    modality=None,
    hardware_target="cpu",
):
    if mpc_metrics is None and compute_ci:
        log.info("2/9 Computing Synergy & Consciousness Index (CI)")
    else:
        metric_label = "all" if mpc_metrics is None else ",".join(mpc_metrics)
        log.info("2/9 Computing Synergy and selected MPC metrics (%s)", metric_label)
    # 1) set up subjects & onsets
    discovered_subjects = []
    if bids_root is not None and Path(bids_root).exists():
        layout = BIDSLayout(bids_root, validate=False)
        discovered_subjects = sorted(layout.get(return_type='id', target='subject'))
    else:
        discovered_subjects = sorted(
            d for d in os.listdir(prep_out)
            if not d.startswith('.') and os.path.isdir(os.path.join(prep_out, d))
        )
    if subjects is not None:
        sel = {str(s).replace("sub-", "").strip() for s in subjects if str(s).strip()}
        discovered_subjects = [s for s in discovered_subjects if s in sel]

    needs_onsets = (mpc_metrics is None) or bool({"RAM", "SRPI"} & set(mpc_metrics))
    if onsets is None and needs_onsets and load_onsets_fn is not None and bids_root is not None:
        def _load_one(subj, ses):
            try:
                return load_onsets_fn(bids_root, subj, ses, condition=condition)
            except TypeError:
                return load_onsets_fn(bids_root, subj, ses)

        onsets = {
            subj: {ses: _load_one(subj, ses) for ses in sessions}
            for subj in discovered_subjects
        }

    # 2) get TR/sample interval (no fallback defaults)
    real_tr = tr if tr is not None else _infer_sample_interval_seconds(bids_root)
    if real_tr is None or (not np.isfinite(real_tr)) or float(real_tr) <= 0:
        raise ValueError(
            "Could not determine a valid sample interval. "
            "Provide explicit --tr (or dataset metadata with RepetitionTime/SamplingFrequency)."
        )
    # 3) compute coarse & fine synergy+CI
    df = compute_synergy_ci(
        str(prep_out),
        atlas,
        thetas,
        sessions,
        condition=condition,
        tr=real_tr,
        stimulus_onsets=onsets,
        mpc_metrics=mpc_metrics,
        compute_ci=compute_ci,
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
        iim_checkpoint_every_cuts=iim_checkpoint_every_cuts,
        iim_progress_log_every_cuts=iim_progress_log_every_cuts,
        iim_use_shared_memory=iim_use_shared_memory,
        iim_phase1_parallel_workers=iim_phase1_parallel_workers,
        iim_phase1_chunk_size=iim_phase1_chunk_size,
        iim_phase1_shared_memory=iim_phase1_shared_memory,
        pdi_params=pdi_params,
        pdi_require_explicit_params=pdi_require_explicit_params,
        pdi_require_strict_baseline=pdi_require_strict_baseline,
        pdi_primary_endpoint=pdi_primary_endpoint,
        nas_params=nas_params,
        srpi_params=srpi_params,
        srpi_require_explicit_params=srpi_require_explicit_params,
        subjects=discovered_subjects,
        iim_precomputed_by_path=iim_precomputed_by_path,
        dataset_id=dataset_id,
        data_origin=data_origin,
        dataset_role=dataset_role,
        provenance_label=provenance_label,
        modality=modality,
        hardware_target=hardware_target,
    )
    df['session']=df['session'].replace({'audioawake':'awake','audiodeep':'deep'})
    meta_cols = [
        c
        for c in (*PROVENANCE_COLUMNS, "hardware_target", "hardware_backend", "hardware_runtime")
        if c in df.columns
    ]
    agg_map = {'S': ('S', 'mean')}
    for meta_col in meta_cols:
        agg_map[meta_col] = (meta_col, 'first')
    for metric in (
        'CI', 'RAM', 'PDI', 'PDI_anchor', 'PDI_task',
        'NAS', 'IIM', 'SRPI', 'IIM_raw', 'IIM_raw_scaled'
    ):
        if metric in df.columns:
            agg_map[metric] = (metric, 'mean')
    df_mean = df.groupby(['subject','session']).agg(**agg_map).reset_index()
    df_S    = df_mean.pivot(index='subject', columns='session', values='S')
    df_CI   = df_mean.pivot(index='subject', columns='session', values='CI') if 'CI' in df_mean.columns else None
    # --- fine grid & supplemental figure ---
    df_fine = compute_synergy_ci(
        str(prep_out),
        atlas,
        thetas_fine,
        sessions,
        condition=condition,
        tr=real_tr,
        stimulus_onsets=onsets,
        compute_mpc=False,
        pdi_params=pdi_params,
        pdi_require_explicit_params=pdi_require_explicit_params,
        pdi_require_strict_baseline=pdi_require_strict_baseline,
        pdi_primary_endpoint=pdi_primary_endpoint,
        nas_params=nas_params,
        srpi_params=srpi_params,
        srpi_require_explicit_params=srpi_require_explicit_params,
        subjects=discovered_subjects,
        dataset_id=dataset_id,
        data_origin=data_origin,
        dataset_role=dataset_role,
        provenance_label=provenance_label,
        modality=modality,
        hardware_target=hardware_target,
    )
    df_fine['session']=df_fine['session'].replace({'audioawake':'awake','audiodeep':'deep'})
    means, sems = [], []
    theta_vals = []
    for theta, subdf in df_fine.groupby('theta'):
        theta_vals.append(float(theta))
        # If multiple runs per subject/session exist, average within each
        # subject/session at fixed theta before paired comparisons.
        sub_mean = (
            subdf.groupby(['subject', 'session'], as_index=False)['S']
            .mean()
        )
        piv = sub_mean.pivot(index='subject', columns='session', values='S')
        paired = piv[['awake', 'deep']].dropna() if {'awake', 'deep'}.issubset(piv.columns) else pd.DataFrame()
        if paired.empty:
            means.append(np.nan)
            sems.append(np.nan)
            continue
        diff = paired['awake'] - paired['deep']
        n = int(diff.notna().sum())
        means.append(float(diff.mean()))
        sems.append(float(diff.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan)
    theta_arr = np.asarray(theta_vals, dtype=float)
    mean_arr = np.asarray(means, dtype=float)
    sem_arr = np.asarray(sems, dtype=float)
    if mean_arr.size and np.isfinite(mean_arr).any():
        i_star = int(np.nanargmax(np.abs(mean_arr)))
        theta_star = float(theta_arr[i_star])
    else:
        theta_star = np.nan
    fig, ax = plt.subplots()
    ax.errorbar(theta_arr, mean_arr, yerr=sem_arr, marker='o')
    if np.isfinite(theta_star):
        ax.axvline(theta_star, linestyle='--')
    ax.set(xlabel='θ', ylabel='Mean S_awake–S_deep'); fig.tight_layout()
    fig.savefig(figdir/'supp_theta_curve.png')
    # --- stats by theta ---
    rows = []
    for theta, subdf in df.groupby('theta'):
        sub_mean = (
            subdf.groupby(['subject', 'session'], as_index=False)['S']
            .mean()
        )
        piv = sub_mean.pivot(index='subject', columns='session', values='S')
        paired = piv[['awake', 'deep']].dropna() if {'awake', 'deep'}.issubset(piv.columns) else pd.DataFrame()
        if len(paired) < 2:
            rows.append(
                {
                    'theta': theta,
                    't_S': np.nan,
                    'p_S': np.nan,
                    'd_S': np.nan,
                    'mean_diff_S': np.nan,
                }
            )
            continue
        t, p = stats.ttest_rel(paired['awake'].values, paired['deep'].values)
        diff = paired['awake'] - paired['deep']
        sd = diff.std(ddof=1)
        d = float(diff.mean() / sd) if np.isfinite(sd) and sd > 0 else np.nan
        rows.append(
            {
                'theta': theta,
                't_S': float(t),
                'p_S': float(p),
                'd_S': d,
                'mean_diff_S': float(diff.mean()),
            }
        )
    df_stats_by_theta = pd.DataFrame(rows).set_index('theta')
    return df, df_mean, df_S, df_CI, df_stats_by_theta
