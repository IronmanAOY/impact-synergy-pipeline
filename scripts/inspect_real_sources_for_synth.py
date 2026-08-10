#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import signal

try:
    import mne
except Exception:  # pragma: no cover - optional at import time
    mne = None


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from impact_pipeline.dataset_catalog import dataset_root_candidates, get_report_dataset


DEFAULT_DATASETS = (
    "ds003171",
    "ds005620",
    "ds002547",
    "ds005479",
    "ds004295",
    "ds002336",
    "ds002685",
    "ds006623",
)

COMPLETE_DATASETS = {
    "ds003171",
    "ds005620",
    "ds002547",
    "ds005479",
    "ds004295",
    "ds002336",
}
INCOMPLETE_DATASETS = {"ds002685", "ds006623"}
DEFAULT_SYNTH_ROOT = Path(os.environ.get("IMPACT_SYNTH_ROOT", "/Volumes/MPW_OT_AOY/impact-synergy-pipeline"))
DEFAULT_SOURCE_ROOT = os.environ.get("IMPACT_SOURCE_ROOT")

FMRI_EXTS = (".nii", ".nii.gz")
EEG_EXTS = (".vhdr", ".set")
RAW_BINARY_EXTS = (".eeg", ".fdt")
TEXT_META_EXTS = (".tsv", ".json", ".bval", ".bvec", ".txt", ".md", ".csv")
ATLAS_NODES = 64


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _unique_paths(paths: list[Path]) -> list[Path]:
    seen = set()
    out = []
    for path in paths:
        expanded = path.expanduser()
        key = str(expanded)
        if key not in seen:
            seen.add(key)
            out.append(expanded)
    return out


def _source_dataset_candidates(dataset_id: str, repo_root: Path, source_root: Path | None) -> list[Path]:
    candidates: list[Path] = []
    if source_root is not None:
        root = source_root.expanduser()
        if dataset_id == "ds005620":
            if root.name in {"ds005620", "ds005620_annex"}:
                candidates.append(root)
            candidates.extend(
                [
                    root / "ds005620_annex",
                    root / "ds005620",
                    root / "data" / "scratch" / "ds005620_annex",
                    root / "data" / "scratch" / "ds005620",
                    root / "data" / "ds005620",
                ]
            )
        else:
            if root.name == dataset_id:
                candidates.append(root)
            candidates.extend(
                [
                    root / dataset_id,
                    root / "data" / "scratch" / dataset_id,
                    root / "data" / dataset_id,
                ]
            )
    candidates.extend(dataset_root_candidates(dataset_id, repo_root))
    return _unique_paths(candidates)


def _resolve_source_dataset_root(dataset_id: str, repo_root: Path, source_root: Path | None) -> Path | None:
    for candidate in _source_dataset_candidates(dataset_id, repo_root, source_root):
        if candidate.exists():
            return candidate.resolve()
    return None


def _is_annex_placeholder(path: Path) -> bool:
    if path.is_symlink():
        try:
            target = path.resolve(strict=True)
        except FileNotFoundError:
            return True
        return not target.exists()
    return False


def _safe_stat(path: Path) -> dict[str, Any]:
    try:
        st = path.stat()
        return {"size": int(st.st_size), "exists": True, "placeholder": _is_annex_placeholder(path)}
    except OSError:
        return {"size": 0, "exists": False, "placeholder": True}


def _hash_file_sample(path: Path, n_bytes: int = 1024 * 1024) -> str | None:
    try:
        h = hashlib.sha256()
        with path.open("rb") as f:
            h.update(f.read(n_bytes))
        return h.hexdigest()
    except OSError:
        return None


def _list_files(root: Path) -> list[Path]:
    skip_dirs = {".git", ".datalad"}
    out: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        base = Path(dirpath)
        for name in filenames:
            p = base / name
            if p.is_file() or p.is_symlink():
                out.append(p)
    return out


def _read_table(path: Path) -> pd.DataFrame | None:
    try:
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        return pd.read_csv(path, sep="\t")
    except Exception:
        return None


def _numeric_summary(values: list[float] | np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p05": float(np.quantile(arr, 0.05)),
        "median": float(np.median(arr)),
        "p95": float(np.quantile(arr, 0.95)),
        "max": float(np.max(arr)),
    }


def _task_from_name(name: str) -> str | None:
    m = re.search(r"_task-([^_]+)", name)
    return m.group(1) if m else None


def _session_from_path(path: Path) -> str | None:
    for part in path.parts:
        if part.startswith("ses-"):
            return part
    return None


def _subject_from_path(path: Path) -> str | None:
    for part in path.parts:
        if part.startswith("sub-"):
            return part
    return None


def _acf1(ts: np.ndarray) -> np.ndarray:
    x = np.asarray(ts, dtype=float)
    if x.ndim != 2 or x.shape[0] < 3:
        return np.asarray([], dtype=float)
    x0 = x[:-1] - x[:-1].mean(axis=0, keepdims=True)
    x1 = x[1:] - x[1:].mean(axis=0, keepdims=True)
    denom = np.sqrt(np.sum(x0 * x0, axis=0) * np.sum(x1 * x1, axis=0))
    out = np.divide(np.sum(x0 * x1, axis=0), denom, out=np.zeros(x.shape[1]), where=denom > 1e-12)
    return out[np.isfinite(out)]


def _corr_summary(ts: np.ndarray) -> dict[str, Any]:
    x = np.asarray(ts, dtype=float)
    if x.ndim != 2 or x.shape[1] < 2 or x.shape[0] < 4:
        return {"n_pairs": 0}
    x = x - x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    keep = sd.reshape(-1) > 1e-8
    if int(keep.sum()) < 2:
        return {"n_pairs": 0}
    x = x[:, keep] / sd[:, keep]
    c = np.corrcoef(x, rowvar=False)
    tri = c[np.triu_indices_from(c, k=1)]
    return _numeric_summary(tri) | {"n_pairs": int(tri.size)}


def _cov_eig_summary(ts: np.ndarray, max_nodes: int = 128) -> dict[str, Any]:
    x = np.asarray(ts, dtype=float)
    if x.ndim != 2 or x.shape[1] < 2 or x.shape[0] < 4:
        return {"n_eigs": 0}
    if x.shape[1] > max_nodes:
        idx = np.linspace(0, x.shape[1] - 1, max_nodes).astype(int)
        x = x[:, idx]
    x = x - x.mean(axis=0, keepdims=True)
    cov = np.cov(x, rowvar=False)
    eig = np.linalg.eigvalsh(cov)
    eig = eig[np.isfinite(eig)]
    out = _numeric_summary(eig)
    out["n_eigs"] = int(eig.size)
    out["condition_estimate"] = float((np.max(eig) + 1e-12) / (np.min(eig[eig > 1e-12]) + 1e-12)) if np.any(eig > 1e-12) else None
    return out


def _nifti_stats(path: Path, dataset_id: str, sample_stride: int = 17) -> dict[str, Any]:
    row: dict[str, Any] = {
        "path": str(path),
        "name": path.name,
        "dataset_id": dataset_id,
        "task": _task_from_name(path.name),
        "subject": _subject_from_path(path),
        "session": _session_from_path(path),
        "kind": "nifti",
        **_safe_stat(path),
    }
    if row["placeholder"]:
        return row
    try:
        img = nib.load(str(path))
        shape = tuple(int(v) for v in img.shape)
        zooms = tuple(float(v) for v in img.header.get_zooms())
        row.update({"shape": shape, "zooms": zooms, "ndim": len(shape)})
        if len(shape) == 4:
            row["n_volumes"] = int(shape[3])
            row["tr"] = float(zooms[3]) if len(zooms) >= 4 else None
            data = np.asanyarray(img.dataobj)
            vol_idx = np.unique(np.linspace(0, shape[3] - 1, min(16, shape[3])).astype(int))
            vals = []
            for vi in vol_idx:
                v = np.asarray(data[..., int(vi)], dtype=np.float32)
                finite = v[np.isfinite(v)]
                if finite.size:
                    vals.append(finite[:: max(1, finite.size // 5000)])
            if vals:
                flat = np.concatenate(vals)
                row["sample_intensity"] = _numeric_summary(flat)
            # Build a coarse grid time series to learn temporal dependence without full atlas extraction.
            if shape[3] >= 8 and all(s > 4 for s in shape[:3]):
                xs = np.linspace(2, shape[0] - 3, 4).astype(int)
                ys = np.linspace(2, shape[1] - 3, 4).astype(int)
                zs = np.linspace(2, shape[2] - 3, 4).astype(int)
                series = []
                for x in xs:
                    for y in ys:
                        for z in zs:
                            ts = np.asarray(data[int(x), int(y), int(z), :], dtype=float)
                            if np.isfinite(ts).all() and float(np.std(ts)) > 1e-6:
                                series.append(ts)
                if series:
                    ts2 = np.vstack(series).T
                    row["sample_time_series"] = {
                        "n_nodes": int(ts2.shape[1]),
                        "n_time": int(ts2.shape[0]),
                        "acf1": _numeric_summary(_acf1(ts2)),
                        "corr": _corr_summary(ts2),
                        "cov_eigs": _cov_eig_summary(ts2),
                    }
        else:
            row["n_volumes"] = None
            row["tr"] = None
    except Exception as exc:
        row["error"] = str(exc)
    return row


def _events_stats(path: Path) -> dict[str, Any]:
    row: dict[str, Any] = {
        "path": str(path),
        "name": path.name,
        "dataset_id": None,
        "task": _task_from_name(path.name),
        "subject": _subject_from_path(path),
        "session": _session_from_path(path),
        "kind": "events",
        **_safe_stat(path),
    }
    if row["placeholder"]:
        return row
    df = _read_table(path)
    if df is None:
        row["error"] = "failed_to_read_table"
        return row
    row["n_rows"] = int(len(df))
    row["columns"] = [str(c) for c in df.columns]
    cols_l = {str(c).lower(): c for c in df.columns}
    if "trial_type" in cols_l:
        counts = df[cols_l["trial_type"]].astype(str).value_counts(dropna=False)
        row["trial_type_counts"] = {str(k): int(v) for k, v in counts.items()}
    for col in ("onset", "duration", "response_time", "reward", "outcome", "value", "feedback", "accuracy", "correct", "prediction_error", "pe"):
        c = cols_l.get(col)
        if c is not None:
            row[f"{col}_summary"] = _numeric_summary(pd.to_numeric(df[c], errors="coerce").to_numpy())
    if "onset" in cols_l:
        onset = pd.to_numeric(df[cols_l["onset"]], errors="coerce").dropna().to_numpy(dtype=float)
        row["onset_isi_summary"] = _numeric_summary(np.diff(np.sort(onset))) if onset.size > 1 else {"n": 0}
        row["run_duration_estimate"] = float(np.nanmax(onset)) if onset.size else None
    txt = " ".join(str(v).lower() for c in df.columns for v in df[c].dropna().astype(str).head(500))
    row["has_self_labels"] = bool(re.search(r"\bself\b|own|me\b|my\b", txt))
    row["has_nonself_labels"] = bool(re.search(r"non[-_ ]?self|other|stranger|another|control", txt))
    row["has_feedback_reward_labels"] = bool(re.search(r"feedback|reward|outcome|correct|incorrect|gain|loss|punish", txt))
    return row


def _json_stats(path: Path) -> dict[str, Any]:
    row = {
        "path": str(path),
        "name": path.name,
        "subject": _subject_from_path(path),
        "session": _session_from_path(path),
        "task": _task_from_name(path.name),
        "kind": "json",
        **_safe_stat(path),
    }
    if row["placeholder"]:
        return row
    obj = _read_json(path)
    row["keys"] = sorted(obj.keys())[:80]
    for key in ("RepetitionTime", "SamplingFrequency", "TaskName", "Manufacturer", "PowerLineFrequency", "EEGReference", "RecordingType"):
        if key in obj:
            row[key] = obj[key]
    return row


def _read_brainvision_segment(vhdr: Path, max_seconds: float = 30.0) -> dict[str, Any]:
    row = {
        "path": str(vhdr),
        "name": vhdr.name,
        "subject": _subject_from_path(vhdr),
        "task": _task_from_name(vhdr.name),
        "kind": "eeg_header",
        **_safe_stat(vhdr),
    }
    if mne is None:
        row["error"] = "mne_unavailable"
        return row
    try:
        raw = mne.io.read_raw_brainvision(str(vhdr), preload=False, verbose="ERROR")
        sfreq = float(raw.info["sfreq"])
        n_ch = int(len(raw.ch_names))
        n_samp = int(raw.n_times)
        stop = min(n_samp, int(round(sfreq * max_seconds)))
        data = raw.get_data(start=0, stop=stop)
        data = np.asarray(data, dtype=float).T
        row.update({
            "sfreq": sfreq,
            "n_channels": n_ch,
            "n_samples": n_samp,
            "duration_sec": float(n_samp / sfreq) if sfreq else None,
            "channels": raw.ch_names[:128],
            "sample_stats": {
                "amplitude": _numeric_summary(data.reshape(-1)),
                "acf1": _numeric_summary(_acf1(data)),
                "corr": _corr_summary(data),
                "cov_eigs": _cov_eig_summary(data),
            },
        })
        bands = {
            "delta": (0.5, 4.0),
            "theta": (4.0, 8.0),
            "alpha": (8.0, 13.0),
            "beta": (13.0, 30.0),
            "gamma": (30.0, 45.0),
        }
        freqs, psd = signal.welch(data, fs=sfreq, axis=0, nperseg=min(data.shape[0], int(sfreq * 4)))
        bandpower = {}
        for name, (lo, hi) in bands.items():
            mask = (freqs >= lo) & (freqs < hi)
            vals = np.trapz(psd[mask], freqs[mask], axis=0) if np.any(mask) else np.asarray([])
            bandpower[name] = _numeric_summary(vals)
        row["bandpower"] = bandpower
    except Exception as exc:
        row["error"] = str(exc)
    return row


def _read_set_header(path: Path) -> dict[str, Any]:
    row = {
        "path": str(path),
        "name": path.name,
        "subject": _subject_from_path(path),
        "task": _task_from_name(path.name),
        "kind": "eeg_set",
        **_safe_stat(path),
    }
    if mne is None:
        row["error"] = "mne_unavailable"
        return row
    try:
        raw = mne.io.read_raw_eeglab(str(path), preload=False, verbose="ERROR")
        sfreq = float(raw.info["sfreq"])
        stop = min(int(raw.n_times), int(round(sfreq * 20)))
        data = np.asarray(raw.get_data(start=0, stop=stop), dtype=float).T
        row.update({
            "sfreq": sfreq,
            "n_channels": int(len(raw.ch_names)),
            "n_samples": int(raw.n_times),
            "duration_sec": float(raw.n_times / sfreq) if sfreq else None,
            "channels": raw.ch_names[:128],
            "sample_stats": {
                "amplitude": _numeric_summary(data.reshape(-1)),
                "acf1": _numeric_summary(_acf1(data)),
                "corr": _corr_summary(data),
                "cov_eigs": _cov_eig_summary(data),
            },
        })
    except Exception as exc:
        row["error"] = str(exc)
    return row


def inspect_dataset(
    dataset_id: str,
    repo_root: Path,
    max_nifti: int,
    max_eeg: int,
    source_root: Path | None = None,
) -> dict[str, Any]:
    entry = get_report_dataset(dataset_id)
    local_root = _resolve_source_dataset_root(dataset_id, repo_root, source_root)
    if dataset_id == "ds005620" and source_root is None and local_root is not None and local_root.name == "ds005620":
        alt = repo_root / "data" / "scratch" / "ds005620_annex"
        if alt.exists():
            local_root = alt.resolve()
    out: dict[str, Any] = {
        "dataset_id": dataset_id,
        "catalog": entry.to_record() if entry is not None else None,
        "configured_source_root": str(source_root) if source_root is not None else None,
        "source_root_candidates": [
            str(path) for path in _source_dataset_candidates(dataset_id, repo_root, source_root)
        ],
        "local_root": str(local_root) if local_root else None,
        "available": bool(local_root and local_root.exists()),
        "marked_incomplete": bool(local_root and (local_root / "DOWNLOAD_INCOMPLETE.txt").exists()),
        "complete_snapshot_expected": dataset_id in COMPLETE_DATASETS,
        "excluded_incomplete": dataset_id in INCOMPLETE_DATASETS,
    }
    if not local_root or not local_root.exists():
        return out
    if dataset_id in INCOMPLETE_DATASETS:
        marker = local_root / "DOWNLOAD_INCOMPLETE.txt"
        status_json = local_root / ".download_incomplete.json"
        out.update(
            {
                "skipped_deep_inspection": True,
                "skip_reason": "incomplete_or_deferred_source",
                "incomplete_marker": str(marker) if marker.exists() else None,
                "incomplete_status_json": str(status_json) if status_json.exists() else None,
            }
        )
        return out
    files = _list_files(local_root)
    out["file_counts"] = {
        "total": len(files),
        "events_tsv": sum(p.name.endswith("_events.tsv") for p in files),
        "scans_tsv": sum(p.name.endswith("_scans.tsv") for p in files),
        "json": sum(p.suffix.lower() == ".json" for p in files),
        "nifti": sum(p.name.endswith(FMRI_EXTS) for p in files),
        "brainvision_vhdr": sum(p.suffix.lower() == ".vhdr" for p in files),
        "eeglab_set": sum(p.suffix.lower() == ".set" for p in files),
        "npy": sum(p.suffix.lower() == ".npy" for p in files),
        "placeholders": sum(_is_annex_placeholder(p) for p in files),
    }
    subjects = sorted({x for p in files if (x := _subject_from_path(p))})
    tasks = sorted({x for p in files if (x := _task_from_name(p.name))})
    out["subjects"] = subjects
    out["n_subjects"] = len(subjects)
    out["tasks"] = tasks
    out["n_tasks"] = len(tasks)

    participants = local_root / "participants.tsv"
    if participants.exists():
        dfp = _read_table(participants)
        if dfp is not None:
            out["participants"] = {
                "rows": int(len(dfp)),
                "columns": [str(c) for c in dfp.columns],
                "path": str(participants),
            }
    scans_rows = []
    for p in sorted(files):
        if p.name.endswith("_scans.tsv"):
            df = _read_table(p)
            if df is not None:
                scans_rows.append({"path": str(p), "rows": int(len(df)), "columns": [str(c) for c in df.columns]})
    out["scans"] = scans_rows

    events = []
    trial_counts = Counter()
    numeric_columns = defaultdict(list)
    for p in sorted(files):
        if p.name.endswith("_events.tsv"):
            row = _events_stats(p)
            row["dataset_id"] = dataset_id
            events.append(row)
            for k, v in row.get("trial_type_counts", {}).items():
                trial_counts[k] += int(v)
            for k, v in row.items():
                if k.endswith("_summary") and isinstance(v, dict) and "mean" in v:
                    numeric_columns[k].append(v["mean"])
    out["events"] = {
        "files": events[:500],
        "n_files": len(events),
        "trial_type_counts_total": dict(trial_counts.most_common(100)),
        "has_self_labels": any(e.get("has_self_labels") for e in events),
        "has_nonself_labels": any(e.get("has_nonself_labels") for e in events),
        "has_feedback_reward_labels": any(e.get("has_feedback_reward_labels") for e in events),
        "numeric_mean_summaries": {k: _numeric_summary(v) for k, v in numeric_columns.items()},
    }

    json_rows = []
    repetition_times = []
    sampling_freqs = []
    for p in sorted(files):
        if p.suffix.lower() == ".json":
            row = _json_stats(p)
            json_rows.append(row)
            if "RepetitionTime" in row:
                try:
                    repetition_times.append(float(row["RepetitionTime"]))
                except Exception:
                    pass
            if "SamplingFrequency" in row:
                try:
                    sampling_freqs.append(float(row["SamplingFrequency"]))
                except Exception:
                    pass
    out["json_sidecars"] = {
        "n_files": len(json_rows),
        "examples": json_rows[:300],
        "repetition_time": _numeric_summary(repetition_times),
        "sampling_frequency": _numeric_summary(sampling_freqs),
    }

    nifti_files = [p for p in sorted(files) if p.name.endswith(FMRI_EXTS)]
    # Prefer functional BOLD and derivatives, then cap the inspection work.
    nifti_priority = sorted(
        nifti_files,
        key=lambda p: (
            0 if "bold" in p.name.lower() else 1,
            0 if "derivatives" in str(p) else 1,
            str(p),
        ),
    )
    nifti_rows = [_nifti_stats(p, dataset_id) for p in nifti_priority[:max_nifti]]
    out["nifti"] = {
        "n_files": len(nifti_files),
        "n_inspected": len(nifti_rows),
        "files": nifti_rows,
        "tr": _numeric_summary([r.get("tr") for r in nifti_rows if r.get("tr")]),
        "n_volumes": _numeric_summary([r.get("n_volumes") for r in nifti_rows if r.get("n_volumes")]),
        "acf1_means": _numeric_summary([
            r.get("sample_time_series", {}).get("acf1", {}).get("mean")
            for r in nifti_rows
            if isinstance(r.get("sample_time_series"), dict)
        ]),
    }

    eeg_rows = []
    eeg_files = [p for p in sorted(files) if p.suffix.lower() in EEG_EXTS]
    for p in eeg_files[:max_eeg]:
        if p.suffix.lower() == ".vhdr":
            eeg_rows.append(_read_brainvision_segment(p))
        elif p.suffix.lower() == ".set":
            eeg_rows.append(_read_set_header(p))
    out["eeg"] = {
        "n_files": len(eeg_files),
        "n_inspected": len(eeg_rows),
        "files": eeg_rows,
        "sampling_frequency": _numeric_summary([r.get("sfreq") for r in eeg_rows if r.get("sfreq")]),
        "n_channels": _numeric_summary([r.get("n_channels") for r in eeg_rows if r.get("n_channels")]),
        "duration_sec": _numeric_summary([r.get("duration_sec") for r in eeg_rows if r.get("duration_sec")]),
    }

    npy_files = [p for p in sorted(files) if p.suffix.lower() == ".npy"]
    npy_rows = []
    for p in npy_files[:200]:
        row = {"path": str(p), "name": p.name, **_safe_stat(p)}
        if not row["placeholder"]:
            try:
                arr = np.load(p, mmap_mode="r")
                row["shape"] = tuple(int(v) for v in arr.shape)
                row["dtype"] = str(arr.dtype)
            except Exception as exc:
                row["error"] = str(exc)
        npy_rows.append(row)
    out["npy_time_series"] = {"n_files": len(npy_files), "examples": npy_rows}
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect real local/OpenNeuro source payloads before synthetic generation.")
    parser.add_argument("--datasets", nargs="*", default=list(DEFAULT_DATASETS))
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_SYNTH_ROOT / "test_objects" / "real_derived_synth_completed" / "reports"),
    )
    parser.add_argument(
        "--source-root",
        default=DEFAULT_SOURCE_ROOT,
        help=(
            "Parent directory containing downloaded OpenNeuro source folders. "
            "If omitted, the script checks the repository data/scratch and data folders."
        ),
    )
    parser.add_argument("--max-nifti", type=int, default=36)
    parser.add_argument("--max-eeg", type=int, default=36)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    source_root = Path(args.source_root).expanduser().resolve() if args.source_root else None
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = {}
    for dataset_id in args.datasets:
        print(f"inspect {dataset_id}", flush=True)
        all_rows[dataset_id] = inspect_dataset(dataset_id, REPO_ROOT, args.max_nifti, args.max_eeg, source_root)
        (output_dir / f"{dataset_id}_source_inspection.json").write_text(
            json.dumps(all_rows[dataset_id], indent=2, sort_keys=True, default=_json_default),
            encoding="utf-8",
        )
    summary = {
        "configured_source_root": str(source_root) if source_root is not None else None,
        "datasets": {
            ds: {
                "available": row.get("available"),
                "marked_incomplete": row.get("marked_incomplete"),
                "complete_snapshot_expected": row.get("complete_snapshot_expected"),
                "excluded_incomplete": row.get("excluded_incomplete"),
                "n_subjects": row.get("n_subjects"),
                "tasks": row.get("tasks"),
                "file_counts": row.get("file_counts"),
            }
            for ds, row in all_rows.items()
        },
        "target_datasets": ["ds003171", "ds002547", "ds005620"],
        "excluded_incomplete": ["ds006623", "ds002685"],
    }
    (output_dir / "source_inspection_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )
    print(f"wrote source inspection reports -> {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
