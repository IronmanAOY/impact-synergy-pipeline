import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


_STIM_RE = re.compile(r"audio|stim|tone|target|event", re.I)
_GOAL_RE = re.compile(r"goal|objective|intent|instruction|cue|self|name", re.I)
_FEEDBACK_RE = re.compile(
    r"feedback|reward|error|outcome|correct|incorrect|response|choice|result",
    re.I,
)
_SELF_RE = re.compile(r"\bself\b|own|myname|subject.?name|participant.?name|me\b|my\b", re.I)
_NONSELF_RE = re.compile(r"non[-_ ]?self|other|stranger|another|control|third.?person", re.I)


def _extract_run_id_from_name(fname: str) -> Optional[str]:
    m = re.search(r"_run-0*([0-9]+)", str(fname))
    if not m:
        return None
    return str(int(m.group(1)))


def _resolve_events_file(
    bids_root: Path,
    subject: str,
    session: str,
    condition: str = "audio",
) -> Optional[Path]:
    subj_root = bids_root / f"sub-{subject}"
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
        eeg_patterns = [f"sub-{subject}_task-{session_key}*_events.tsv"]

    for base_dir, patterns in ((fmri_dir, fmri_patterns), (eeg_dir, eeg_patterns)):
        if not base_dir.exists():
            continue
        for pat in patterns:
            hits = sorted(base_dir.glob(pat))
            if hits:
                return hits[0]
    return None


def _read_events_table(events_file: Path) -> Optional[pd.DataFrame]:
    if events_file is None or (not events_file.exists()):
        return None
    df = pd.read_csv(events_file, sep="\t")
    if df is None:
        return None
    if len(df.columns) > 0:
        df = df.rename(columns={c: str(c).strip().lstrip("\ufeff") for c in df.columns})
    return df


def _events_to_ram_bundle(df: Optional[pd.DataFrame]) -> Dict[str, object]:
    if df is None or df.empty:
        return {
            "onsets": [],
            "goal_onsets": [],
            "feedback_onsets": [],
            "feedback_values": [],
        }
    cols_l = {str(c).lower(): c for c in df.columns}
    onset_col = cols_l.get("onset")
    if onset_col is None:
        return {
            "onsets": [],
            "goal_onsets": [],
            "feedback_onsets": [],
            "feedback_values": [],
        }

    onset = pd.to_numeric(df[onset_col], errors="coerce")
    valid = onset.notna()
    if "trial_type" in cols_l:
        trial = df[cols_l["trial_type"]].astype(str)
    else:
        trial = pd.Series([""] * len(df), index=df.index, dtype=object)

    stim_mask = trial.str.contains(_STIM_RE, regex=True, na=False)
    goal_mask = trial.str.contains(_GOAL_RE, regex=True, na=False)
    fb_mask = trial.str.contains(_FEEDBACK_RE, regex=True, na=False)

    stim_onsets = onset[valid & stim_mask].astype(float).tolist()
    if not stim_onsets:
        stim_onsets = onset[valid].astype(float).tolist()

    goal_onsets = onset[valid & goal_mask].astype(float).tolist()
    feedback_onsets = onset[valid & fb_mask].astype(float).tolist()

    feedback_values = []
    for cand in (
        "prediction_error",
        "pe",
        "reward",
        "outcome",
        "accuracy",
        "correct",
        "value",
        "response_time",
    ):
        c = cols_l.get(cand)
        if c is None:
            continue
        vals = pd.to_numeric(df[c], errors="coerce")
        if fb_mask.any():
            vals = vals[fb_mask]
        vals = vals[np.isfinite(vals)]
        if vals.shape[0] >= 2 and float(vals.std(ddof=0)) > 0:
            feedback_values = vals.astype(float).tolist()
            break

    return {
        "onsets": stim_onsets,
        "goal_onsets": goal_onsets,
        "feedback_onsets": feedback_onsets,
        "feedback_values": feedback_values,
    }


def _events_to_srpi_onsets(df: Optional[pd.DataFrame]) -> Tuple[List[float], List[float]]:
    if df is None or df.empty:
        return [], []
    cols_l = {str(c).lower(): c for c in df.columns}
    onset_col = cols_l.get("onset")
    if onset_col is None:
        return [], []

    onset = pd.to_numeric(df[onset_col], errors="coerce")
    valid = onset.notna()
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
    non_mask = txt.str.contains(_NONSELF_RE, regex=True, na=False)
    self_mask = self_mask & (~non_mask)

    self_onsets = onset[valid & self_mask].astype(float).tolist()
    non_onsets = onset[valid & non_mask].astype(float).tolist()
    return self_onsets, non_onsets


def _pick_session_ts_paths(
    prep_root: Path,
    subject: str,
    session: str,
    condition: str,
    atlas: str,
    run_id: Optional[str],
) -> List[Path]:
    session_dir = prep_root / subject / session / condition
    cands = sorted(session_dir.glob(f"{subject}_run-*_{atlas}_ts.npy"))
    if not cands:
        return []
    if run_id is None:
        return cands

    picked = []
    try:
        run_clean = str(int(run_id))
    except Exception:
        run_clean = str(run_id)
    run_padded = str(run_id)
    for tok in (run_clean, run_padded):
        hits = sorted(session_dir.glob(f"{subject}_run-{tok}_{atlas}_ts.npy"))
        if hits:
            picked.extend(hits)
    if picked:
        return [picked[0]]

    tags = (f"run-{run_clean}_", f"run-{run_padded}_")
    filt = [p for p in cands if any(t in p.name for t in tags)]
    if filt:
        return [filt[0]]
    return cands


def _pdi_state_rest_runs(prep_root: Path, subject: str, session: str, atlas: str) -> List[Path]:
    return sorted(
        (prep_root / subject / session / "rest").glob(f"{subject}_run-*_{atlas}_ts.npy")
    )


def _pdi_deep_rest_runs(prep_root: Path, subject: str, atlas: str) -> List[Path]:
    return sorted(
        (prep_root / subject / "deep" / "rest").glob(f"{subject}_run-*_{atlas}_ts.npy")
    )


def _assess_ram(bundle: Dict[str, object], require_explicit_feedback: bool) -> Tuple[bool, str, int, int, int]:
    stim = np.asarray(bundle.get("onsets", []), dtype=float)
    stim = stim[np.isfinite(stim)]
    n_stim = int(stim.shape[0])
    if n_stim == 0:
        return False, "missing_stimulus_events", 0, 0, 0

    fb = np.asarray(bundle.get("feedback_onsets", []), dtype=float)
    fb = fb[np.isfinite(fb)]
    n_fb = int(fb.shape[0])
    fvals = np.asarray(bundle.get("feedback_values", []), dtype=float)
    fvals = fvals[np.isfinite(fvals)]
    n_fvals = int(fvals.shape[0])

    if require_explicit_feedback:
        if n_fb == 0:
            return False, "missing_feedback_events", n_stim, n_fb, n_fvals
        if n_fvals < 2:
            return False, "missing_or_nonvarying_feedback_values", n_stim, n_fb, n_fvals

    return True, "ok", n_stim, n_fb, n_fvals


def _assess_nas(n_regions: int, n_time: int) -> Tuple[bool, str]:
    if n_regions < 2:
        return False, "insufficient_regions"
    if n_time < 4:
        return False, "insufficient_timepoints"
    return True, "ok"


def _assess_iim(
    n_regions: int,
    n_time: int,
    bins: int,
    lag_trs: int,
    max_state_space: int,
    max_nodes: Optional[int],
) -> Tuple[bool, str, int, int]:
    if n_regions < 2 or (n_time - int(lag_trs)) < 1:
        return False, "insufficient_shape", 0, 0

    eff_bins = int(bins)
    if max_nodes is None:
        n_sel = int(n_regions)
    else:
        n_sel = int(min(int(max_nodes), int(n_regions)))

    while n_sel >= 2 and (eff_bins ** n_sel) > int(max_state_space):
        if eff_bins > 2:
            eff_bins -= 1
        else:
            n_sel -= 1
    if n_sel < 2:
        return False, "state_space_too_large", int(n_sel), int(eff_bins)
    return True, "ok", int(n_sel), int(eff_bins)


def _assess_srpi(
    self_onsets: List[float],
    nonself_onsets: List[float],
    min_events_per_class: int,
) -> Tuple[bool, str, int, int]:
    n_self = int(len(self_onsets))
    n_non = int(len(nonself_onsets))
    m = max(2, int(min_events_per_class))
    if n_self == 0 and n_non == 0:
        return False, "missing_self_and_nonself_events", n_self, n_non
    if n_self == 0:
        return False, "missing_self_events", n_self, n_non
    if n_non == 0:
        return False, "missing_nonself_events", n_self, n_non
    if n_self < m:
        return False, "insufficient_self_events", n_self, n_non
    if n_non < m:
        return False, "insufficient_nonself_events", n_self, n_non
    return True, "ok", n_self, n_non


def _load_ts_shape(ts_path: Path) -> Tuple[int, int, Optional[str]]:
    try:
        arr = np.load(ts_path)
    except Exception as exc:
        return 0, 0, f"load_error:{type(exc).__name__}"
    if arr.ndim != 2:
        return 0, 0, f"invalid_ndim:{arr.ndim}"
    # stored as time x region in preprocessing outputs
    n_time = int(arr.shape[0])
    n_regions = int(arr.shape[1])
    return n_regions, n_time, None


def check_mpc_readiness(
    prep_root: str,
    bids_root: Optional[str],
    atlas: str,
    condition: str,
    sessions: Sequence[str],
    subjects: Optional[Sequence[str]] = None,
    require_explicit_feedback: bool = True,
    require_explicit_srpi: bool = True,
    srpi_min_events_per_class: int = 3,
    iim_bins: int = 3,
    iim_lag_trs: int = 1,
    iim_max_state_space: int = 1500,
    iim_max_nodes: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    if not bool(require_explicit_srpi):
        raise ValueError("Neutral SRPI mode is disabled; explicit SRPI evidence is required.")
    if int(srpi_min_events_per_class) < 2:
        raise ValueError("srpi_min_events_per_class must be >= 2")
    prep = Path(prep_root)
    bids = Path(bids_root) if bids_root is not None else None
    if not prep.exists():
        raise FileNotFoundError(f"Preprocessed root not found: {prep}")

    prep_subjects = sorted(
        p.name for p in prep.iterdir() if p.is_dir() and (not p.name.startswith("."))
    )
    if subjects is None:
        use_subjects = prep_subjects
    else:
        req = [str(s).replace("sub-", "").strip() for s in subjects if str(s).strip()]
        use_subjects = [s for s in req if s in set(prep_subjects)]

    rows = []
    for subj in use_subjects:
        for ses in sessions:
            events_file = (
                _resolve_events_file(bids, subj, ses, condition=condition)
                if bids is not None
                else None
            )
            run_id = _extract_run_id_from_name(events_file.name) if events_file is not None else None
            df_events = _read_events_table(events_file)
            ram_bundle = _events_to_ram_bundle(df_events)
            self_onsets, nonself_onsets = _events_to_srpi_onsets(df_events)
            ts_paths = _pick_session_ts_paths(
                prep_root=prep,
                subject=subj,
                session=ses,
                condition=condition,
                atlas=atlas,
                run_id=run_id,
            )
            if not ts_paths:
                rows.append(
                    {
                        "subject": subj,
                        "session": ses,
                        "ts_path": None,
                        "events_file": None if events_file is None else str(events_file),
                        "run_id": run_id,
                        "ts_ready": False,
                        "ts_reason": "missing_timeseries",
                        "RAM_ready": False,
                        "RAM_reason": "missing_timeseries",
                        "PDI_ready": False,
                        "PDI_reason": "missing_timeseries",
                        "PDI_anchor_ready": False,
                        "PDI_anchor_reason": "missing_timeseries",
                        "PDI_task_ready": False,
                        "PDI_task_reason": "missing_timeseries",
                        "PDI_anchor_baseline_runs": 0,
                        "PDI_task_baseline_runs": 0,
                        "PDI_baseline_mode": "unknown",
                        "NAS_ready": False,
                        "NAS_reason": "missing_timeseries",
                        "IIM_ready": False,
                        "IIM_reason": "missing_timeseries",
                        "IIM_nodes_used": 0,
                        "IIM_bins_used": 0,
                        "SRPI_ready": False,
                        "SRPI_reason": "missing_timeseries",
                        "n_regions": 0,
                        "n_timepoints": 0,
                        "n_stim_onsets": 0,
                        "n_feedback_onsets": 0,
                        "n_feedback_values": 0,
                        "n_self_onsets": int(len(self_onsets)),
                        "n_nonself_onsets": int(len(nonself_onsets)),
                        "CI_ready": False,
                    }
                )
                continue

            for ts_path in ts_paths:
                n_regions, n_time, ts_err = _load_ts_shape(ts_path)
                ts_ready = ts_err is None
                if not ts_ready:
                    rows.append(
                        {
                            "subject": subj,
                            "session": ses,
                            "ts_path": str(ts_path),
                            "events_file": None if events_file is None else str(events_file),
                            "run_id": run_id,
                            "ts_ready": False,
                            "ts_reason": ts_err,
                            "RAM_ready": False,
                            "RAM_reason": ts_err,
                            "PDI_ready": False,
                            "PDI_reason": ts_err,
                            "PDI_anchor_ready": False,
                            "PDI_anchor_reason": ts_err,
                            "PDI_task_ready": False,
                            "PDI_task_reason": ts_err,
                            "PDI_anchor_baseline_runs": 0,
                            "PDI_task_baseline_runs": 0,
                            "PDI_baseline_mode": "unknown",
                            "NAS_ready": False,
                            "NAS_reason": ts_err,
                            "IIM_ready": False,
                            "IIM_reason": ts_err,
                            "IIM_nodes_used": 0,
                            "IIM_bins_used": 0,
                            "SRPI_ready": False,
                            "SRPI_reason": ts_err,
                            "n_regions": 0,
                            "n_timepoints": 0,
                            "n_stim_onsets": 0,
                            "n_feedback_onsets": 0,
                            "n_feedback_values": 0,
                            "n_self_onsets": int(len(self_onsets)),
                            "n_nonself_onsets": int(len(nonself_onsets)),
                            "CI_ready": False,
                        }
                    )
                    continue

                ram_ok, ram_reason, n_stim, n_fb, n_fvals = _assess_ram(
                    ram_bundle,
                    require_explicit_feedback=require_explicit_feedback,
                )
                pdi_anchor_runs = _pdi_deep_rest_runs(prep, subj, atlas)
                pdi_task_runs = _pdi_state_rest_runs(prep, subj, ses, atlas)
                pdi_anchor_ok = bool(len(pdi_anchor_runs) > 0)
                pdi_task_ok = bool(len(pdi_task_runs) > 0)
                pdi_anchor_reason = "ok" if pdi_anchor_ok else "missing_deep_rest_baseline"
                pdi_task_reason = "ok" if pdi_task_ok else "missing_state_rest_baseline"
                pdi_ok = bool(pdi_anchor_ok and pdi_task_ok)
                if pdi_ok:
                    pdi_reason = "ok"
                elif (not pdi_anchor_ok) and (not pdi_task_ok):
                    pdi_reason = "missing_deep_and_state_rest_baselines"
                elif not pdi_anchor_ok:
                    pdi_reason = "missing_deep_rest_baseline"
                else:
                    pdi_reason = "missing_state_rest_baseline"
                pdi_mode = f"anchor_runs={len(pdi_anchor_runs)};task_runs={len(pdi_task_runs)}"
                nas_ok, nas_reason = _assess_nas(n_regions, n_time)
                iim_ok, iim_reason, iim_nodes, iim_bins_used = _assess_iim(
                    n_regions=n_regions,
                    n_time=n_time,
                    bins=iim_bins,
                    lag_trs=iim_lag_trs,
                    max_state_space=iim_max_state_space,
                    max_nodes=iim_max_nodes,
                )
                srpi_ok, srpi_reason, n_self, n_non = _assess_srpi(
                    self_onsets=self_onsets,
                    nonself_onsets=nonself_onsets,
                    min_events_per_class=int(srpi_min_events_per_class),
                )
                ci_ok = bool(ram_ok and pdi_ok and nas_ok and iim_ok and srpi_ok)

                rows.append(
                    {
                        "subject": subj,
                        "session": ses,
                        "ts_path": str(ts_path),
                        "events_file": None if events_file is None else str(events_file),
                        "run_id": run_id,
                        "ts_ready": True,
                        "ts_reason": "ok",
                        "RAM_ready": bool(ram_ok),
                        "RAM_reason": ram_reason,
                        "PDI_ready": bool(pdi_ok),
                        "PDI_reason": pdi_reason,
                        "PDI_anchor_ready": bool(pdi_anchor_ok),
                        "PDI_anchor_reason": pdi_anchor_reason,
                        "PDI_task_ready": bool(pdi_task_ok),
                        "PDI_task_reason": pdi_task_reason,
                        "PDI_anchor_baseline_runs": int(len(pdi_anchor_runs)),
                        "PDI_task_baseline_runs": int(len(pdi_task_runs)),
                        "PDI_baseline_mode": pdi_mode,
                        "NAS_ready": bool(nas_ok),
                        "NAS_reason": nas_reason,
                        "IIM_ready": bool(iim_ok),
                        "IIM_reason": iim_reason,
                        "IIM_nodes_used": int(iim_nodes),
                        "IIM_bins_used": int(iim_bins_used),
                        "SRPI_ready": bool(srpi_ok),
                        "SRPI_reason": srpi_reason,
                        "n_regions": int(n_regions),
                        "n_timepoints": int(n_time),
                        "n_stim_onsets": int(n_stim),
                        "n_feedback_onsets": int(n_fb),
                        "n_feedback_values": int(n_fvals),
                        "n_self_onsets": int(n_self),
                        "n_nonself_onsets": int(n_non),
                        "CI_ready": bool(ci_ok),
                    }
                )

    df = pd.DataFrame.from_records(rows)
    if df.empty:
        summary = {
            "n_rows": 0,
            "metrics": {},
            "settings": {
                "require_explicit_feedback": bool(require_explicit_feedback),
                "require_explicit_srpi": True,
                "srpi_min_events_per_class": int(srpi_min_events_per_class),
                "pdi_baseline_policy": "strict_deep_rest_plus_state_rest",
            },
        }
        return df, summary

    metrics = ["RAM", "PDI", "PDI_anchor", "PDI_task", "NAS", "IIM", "SRPI", "CI"]
    metric_summary = {}
    for m in metrics:
        col = f"{m}_ready"
        metric_summary[m] = {
            "ready": int(df[col].sum()),
            "not_ready": int((~df[col]).sum()),
            "ready_fraction": float(df[col].mean()),
        }
    summary = {
        "n_rows": int(df.shape[0]),
        "n_subjects": int(df["subject"].nunique()),
        "sessions": sorted(df["session"].dropna().unique().tolist()),
        "metrics": metric_summary,
        "settings": {
            "require_explicit_feedback": bool(require_explicit_feedback),
            "require_explicit_srpi": True,
            "srpi_min_events_per_class": int(srpi_min_events_per_class),
            "pdi_baseline_policy": "strict_deep_rest_plus_state_rest",
            "iim_bins": int(iim_bins),
            "iim_lag_trs": int(iim_lag_trs),
            "iim_max_state_space": int(iim_max_state_space),
            "iim_max_nodes": None if iim_max_nodes is None else int(iim_max_nodes),
        },
    }
    return df, summary


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Check dataset readiness for all MPC metrics.")
    p.add_argument("--prep-root", required=True, help="Preprocessed root folder.")
    p.add_argument("--bids-root", required=False, default=None, help="BIDS root for events parsing.")
    p.add_argument("--atlas", default="schaefer400")
    p.add_argument("--condition", default="audio")
    p.add_argument("--sessions", nargs="+", default=["awake", "deep"])
    p.add_argument("--subjects", nargs="+", default=None)
    p.add_argument("--out-csv", default=None, help="Optional output CSV path.")
    p.add_argument("--out-json", default=None, help="Optional summary JSON path.")
    p.add_argument("--allow-implicit-ram-feedback", action="store_true")
    p.add_argument("--srpi-min-events-per-class", type=int, default=3)
    p.add_argument("--iim-bins", type=int, default=3)
    p.add_argument("--iim-lag-trs", type=int, default=1)
    p.add_argument("--iim-max-state-space", type=int, default=1500)
    p.add_argument("--iim-max-nodes", type=int, default=None)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    df, summary = check_mpc_readiness(
        prep_root=args.prep_root,
        bids_root=args.bids_root,
        atlas=args.atlas,
        condition=args.condition,
        sessions=args.sessions,
        subjects=args.subjects,
        require_explicit_feedback=not bool(args.allow_implicit_ram_feedback),
        require_explicit_srpi=True,
        srpi_min_events_per_class=int(args.srpi_min_events_per_class),
        iim_bins=args.iim_bins,
        iim_lag_trs=args.iim_lag_trs,
        iim_max_state_space=args.iim_max_state_space,
        iim_max_nodes=args.iim_max_nodes,
    )

    if args.out_csv:
        out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
    if args.out_json:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
