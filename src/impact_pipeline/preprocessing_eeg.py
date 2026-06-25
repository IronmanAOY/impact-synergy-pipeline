import os
import logging
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Any

import numpy as np
from bids import BIDSLayout

log = logging.getLogger(__name__)


def _zscore_rows(x: np.ndarray) -> np.ndarray:
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True) + 1e-12
    return (x - mu) / sd


def _collect_runs_for_rule(
    layout: BIDSLayout,
    subject: str,
    task: str,
    acq: str,
) -> List[str]:
    files = layout.get(
        subject=subject,
        datatype="eeg",
        suffix="eeg",
        extension=".vhdr",
        task=task,
        acquisition=acq,
        return_type="filename",
    )
    return sorted(files)


def _existing_files(
    files: Sequence[str],
    subject: str,
    session: str,
    task: str,
    acquisition: str,
    missing_files: List[Dict[str, Any]],
) -> List[str]:
    keep: List[str] = []
    for fn in files:
        if os.path.exists(fn):
            keep.append(fn)
        else:
            log.warning("Skipping missing EEG file referenced by BIDS index: %s", fn)
            missing_files.append(
                {
                    "subject": subject,
                    "session": session,
                    "task": task,
                    "acquisition": acquisition,
                    "file": fn,
                    "reason": "missing_on_disk",
                }
            )
    return keep


def _build_session_runs(
    layout: BIDSLayout,
    subject: str,
    session_rules: Dict[str, Sequence[Tuple[str, str]]],
    missing_files: List[Dict[str, Any]],
) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {k: [] for k in session_rules}
    for session, rules in session_rules.items():
        for task, acq in rules:
            hits = _collect_runs_for_rule(layout, subject=subject, task=task, acq=acq)
            hits = _existing_files(
                hits,
                subject=subject,
                session=session,
                task=task,
                acquisition=acq,
                missing_files=missing_files,
            )
            if hits:
                # Use all runs for the first matching rule to avoid discarding
                # valid recordings from the selected session definition.
                out[session].extend(hits)
                break
    return out


def run_preprocessing_eeg(
    bids_root: str,
    out_root: str,
    subjects: Optional[Iterable[str]] = None,
    session_rules: Optional[Dict[str, Sequence[Tuple[str, str]]]] = None,
    condition_label: str = "eeg",
    atlas_key: str = "eeg64",
    target_sfreq: float = 250.0,
    l_freq: float = 0.5,
    h_freq: float = 45.0,
    max_duration_sec: Optional[float] = 120.0,
    min_common_channels: int = 16,
) -> Dict[str, Any]:
    """
    Convert raw EEG-BIDS recordings into standardized time×channel arrays.

    Output layout:
        {out_root}/{subject}/{session}/{condition_label}/{subject}_run-{k}_{atlas_key}_ts.npy
    """
    # Ensure numba has a writable cache location before importing MNE.
    if not os.environ.get("NUMBA_CACHE_DIR", "").strip():
        default_cache = Path(tempfile.gettempdir()) / "numba_cache"
        default_cache.mkdir(parents=True, exist_ok=True)
        os.environ["NUMBA_CACHE_DIR"] = str(default_cache)

    try:
        import mne  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "EEG preprocessing requires MNE-Python. "
            "Install with: conda install -c conda-forge mne"
        ) from exc

    if session_rules is None:
        # Default mapping for ds005620:
        # awake/EC for wake baseline; sed2/rest preferred for deep sedation,
        # fallback to sed/rest when sed2 is unavailable.
        session_rules = {
            "awake": [("awake", "EC")],
            "deep": [("sed2", "rest"), ("sed", "rest")],
        }

    layout = BIDSLayout(bids_root, validate=False)
    all_subjects = sorted(layout.get(return_type="id", target="subject"))
    if not all_subjects:
        raise FileNotFoundError(
            f"No EEG subjects found under '{bids_root}'. "
            "Check --bids-root and ensure data files are present."
        )
    if subjects is None:
        subjects = all_subjects
    else:
        subjects = [s for s in all_subjects if s in set(subjects)]

    summary: Dict[str, Any] = {
        "subjects_requested": list(subjects),
        "missing_files": [],
        "skipped_subjects": [],
        "written_runs": [],
    }

    for subj in subjects:
        run_map = _build_session_runs(
            layout,
            subj,
            session_rules,
            missing_files=summary["missing_files"],
        )

        # Skip subjects without both comparison sessions.
        missing_sessions = [s for s, files in run_map.items() if len(files) == 0]
        if missing_sessions:
            log.warning(
                "Skipping sub-%s: missing EEG runs for sessions=%s",
                subj,
                ",".join(missing_sessions),
            )
            summary["skipped_subjects"].append(
                {
                    "subject": subj,
                    "reason": "missing_sessions",
                    "detail": ",".join(missing_sessions),
                }
            )
            continue

        # Build per-subject common EEG channel set across all selected runs.
        usable_map: Dict[str, List[str]] = {k: [] for k in run_map}
        eeg_sets = []
        for session, files in run_map.items():
            for fn in files:
                try:
                    raw = mne.io.read_raw_brainvision(fn, preload=False, verbose="ERROR")
                except FileNotFoundError:
                    log.warning("Missing EEG file at read time: %s", fn)
                    summary["missing_files"].append(
                        {
                            "subject": subj,
                            "session": session,
                            "task": None,
                            "acquisition": None,
                            "file": fn,
                            "reason": "missing_at_read_time",
                        }
                    )
                    continue
                except Exception as exc:
                    log.warning("Unreadable EEG file %s (%s)", fn, type(exc).__name__)
                    summary["missing_files"].append(
                        {
                            "subject": subj,
                            "session": session,
                            "task": None,
                            "acquisition": None,
                            "file": fn,
                            "reason": f"header_read_error:{type(exc).__name__}",
                        }
                    )
                    continue

                chs = [
                    ch for ch, typ in zip(raw.ch_names, raw.get_channel_types())
                    if typ == "eeg"
                ]
                eeg_sets.append(set(chs))
                usable_map[session].append(fn)
                raw.close()

        missing_sessions = [s for s, files in usable_map.items() if len(files) == 0]
        if missing_sessions:
            log.warning(
                "Skipping sub-%s: no readable EEG runs for sessions=%s",
                subj,
                ",".join(missing_sessions),
            )
            summary["skipped_subjects"].append(
                {
                    "subject": subj,
                    "reason": "no_readable_runs",
                    "detail": ",".join(missing_sessions),
                }
            )
            continue

        common_channels = sorted(set.intersection(*eeg_sets)) if eeg_sets else []
        if len(common_channels) < int(min_common_channels):
            log.warning(
                "Skipping sub-%s: only %d common EEG channels (<%d).",
                subj,
                len(common_channels),
                min_common_channels,
            )
            summary["skipped_subjects"].append(
                {
                    "subject": subj,
                    "reason": "insufficient_common_channels",
                    "detail": str(len(common_channels)),
                }
            )
            continue

        for session, files in usable_map.items():
            out_dir = os.path.join(out_root, subj, session, condition_label)
            os.makedirs(out_dir, exist_ok=True)
            for i, fn in enumerate(sorted(files), start=1):
                try:
                    raw = mne.io.read_raw_brainvision(fn, preload=True, verbose="ERROR")
                except Exception as exc:
                    log.warning("Skipping run due read failure %s (%s)", fn, type(exc).__name__)
                    summary["missing_files"].append(
                        {
                            "subject": subj,
                            "session": session,
                            "task": None,
                            "acquisition": None,
                            "file": fn,
                            "reason": f"run_read_error:{type(exc).__name__}",
                        }
                    )
                    continue

                raw.pick(common_channels)
                if target_sfreq and float(raw.info["sfreq"]) != float(target_sfreq):
                    raw.resample(float(target_sfreq), npad="auto", verbose="ERROR")
                if (l_freq is not None) or (h_freq is not None):
                    raw.filter(
                        l_freq=l_freq,
                        h_freq=h_freq,
                        fir_design="firwin",
                        verbose="ERROR",
                    )
                x = raw.get_data(reject_by_annotation="omit")
                if max_duration_sec is not None:
                    keep = int(float(max_duration_sec) * float(raw.info["sfreq"]))
                    if keep > 0:
                        x = x[:, : min(keep, x.shape[1])]

                x = _zscore_rows(x)
                ts_time_channel = x.T
                out_fn = os.path.join(out_dir, f"{subj}_run-{i}_{atlas_key}_ts.npy")
                np.save(out_fn, ts_time_channel)
                summary["written_runs"].append(
                    {
                        "subject": subj,
                        "session": session,
                        "source_file": fn,
                        "output_file": out_fn,
                        "run_index": int(i),
                        "n_channels": int(ts_time_channel.shape[1]),
                        "n_timepoints": int(ts_time_channel.shape[0]),
                        "sfreq_hz": float(raw.info["sfreq"]),
                    }
                )
                raw.close()

    requested = len(summary["subjects_requested"])
    skipped = len({r["subject"] for r in summary["skipped_subjects"]})
    processed = len({r["subject"] for r in summary["written_runs"]})
    summary["summary"] = {
        "subjects_requested": int(requested),
        "subjects_processed": int(processed),
        "subjects_skipped": int(skipped),
        "missing_file_records": int(len(summary["missing_files"])),
        "written_runs": int(len(summary["written_runs"])),
    }
    return summary
