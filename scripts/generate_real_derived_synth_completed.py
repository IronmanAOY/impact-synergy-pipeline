#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import signal

try:
    import nibabel as nib
except Exception:  # pragma: no cover - optional outside validation env
    nib = None

try:
    import mne
except Exception:  # pragma: no cover - optional outside validation env
    mne = None


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from impact_pipeline.mpc_readiness import check_mpc_readiness
from impact_pipeline.mpc_metrics import compute_RAM, compute_SRPI
from impact_pipeline.provenance import DUMMY_DATA_ORIGIN
from impact_pipeline.run_synergy_ci import load_onsets
from impact_pipeline.synergy_ci import compute_synergy_ci


SSD_ROOT = Path(os.environ.get("IMPACT_SYNTH_ROOT", REPO_ROOT))
INSPECTION_DIR = SSD_ROOT / "test_objects" / "real_derived_synth_completed" / "reports"
DATASETS_OUT = SSD_ROOT / "test_objects" / "datasets" / "real_derived_synth_completed"
RUNS_OUT = SSD_ROOT / "test_objects" / "runs" / "real_derived_synth_completed"
REPORTS_OUT = SSD_ROOT / "test_objects" / "real_derived_synth_completed" / "reports"
SOURCE_ROOT = Path(os.environ.get("IMPACT_SOURCE_ROOT", SSD_ROOT / "data" / "scratch")).expanduser()

TARGETS = ("ds003171", "ds002547", "ds005620")
DONORS = ("ds005479", "ds004295", "ds002336")

PDI_PARAMS = {
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
FMRI_RAM_PARAMS = {
    "epsilon": None,
    "magnitude_scale": 0.5,
    "response_model": "hrf",
    "response_boxcar_width_sec": None,
    "latency_method": "hrf_peak",
    "fir_window": 20.0,
    "xcorr_maxlag": 10,
    "quality_weights": (1.0, 1.0, 1.0),
    "goal_pre_window_sec": 2.0,
    "response_window_sec": 3.0,
    "goal_objective_window_sec": 2.0,
    "feedback_window_sec": 2.0,
    "quality_ridge": 1e-4,
    "require_explicit_feedback": True,
}
EEG_RAM_PARAMS = dict(FMRI_RAM_PARAMS)
EEG_RAM_PARAMS.update(
    {
        "response_model": "boxcar",
        "response_boxcar_width_sec": 0.30,
        "latency_method": "fir",
        "fir_window": 0.80,
        "goal_pre_window_sec": 0.20,
        "response_window_sec": 0.40,
        "goal_objective_window_sec": 0.20,
        "feedback_window_sec": 0.20,
    }
)
FMRI_NAS_PARAMS = {
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
}
EEG_NAS_PARAMS = dict(FMRI_NAS_PARAMS)
EEG_NAS_PARAMS.update(
    {
        "bands": ((0.5, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 45.0)),
        "band_weights": (0.2, 0.2, 0.2, 0.2, 0.2),
        "window_len": 500,
        "step_len": 250,
    }
)
FMRI_SRPI_PARAMS = {
    "modality": "fmri",
    "pre_window_sec": 2.0,
    "response_lag_sec": 4.0,
    "response_window_sec": 6.0,
    "covariance_ridge": 1e-3,
    "component_weights": (0.35, 0.25, 0.20, 0.20),
    "min_events_per_class": 3,
    "sample_reliability_tau": 4.0,
    "eps": 1e-8,
}
EEG_SRPI_PARAMS = {
    "modality": "eeg",
    "pre_window_sec": 0.20,
    "response_lag_sec": 0.05,
    "response_window_sec": 0.40,
    "covariance_ridge": 1e-3,
    "component_weights": (0.35, 0.25, 0.20, 0.20),
    "min_events_per_class": 3,
    "sample_reliability_tau": 4.0,
    "eps": 1e-8,
}


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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _summary_value(summary: dict[str, Any] | None, key: str, fallback: float) -> float:
    if isinstance(summary, dict):
        val = summary.get(key)
        try:
            val = float(val)
            if np.isfinite(val):
                return val
        except Exception:
            pass
    return float(fallback)


def _load_source_inspections() -> dict[str, dict[str, Any]]:
    inspections = {}
    for ds in (*TARGETS, *DONORS, "ds002685", "ds006623"):
        path = INSPECTION_DIR / f"{ds}_source_inspection.json"
        if path.exists():
            inspections[ds] = _read_json(path)
    missing = [ds for ds in TARGETS if ds not in inspections]
    if missing:
        raise FileNotFoundError(f"Missing source inspection reports for targets: {missing}")
    return inspections


def _first_existing(patterns: list[Path]) -> Path:
    for p in patterns:
        if p.exists():
            return p
    raise FileNotFoundError(f"No existing path among: {[str(p) for p in patterns]}")


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


def _source_dataset_candidates(dataset_id: str) -> list[Path]:
    root = SOURCE_ROOT.expanduser()
    candidates: list[Path] = []
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
    return _unique_paths(candidates)


def _source_dataset_root(dataset_id: str) -> Path:
    candidates = _source_dataset_candidates(dataset_id)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0]


def _load_mid_events() -> pd.DataFrame:
    root = _source_dataset_root("ds005479")
    path = _first_existing(sorted(root.glob("sub-*/func/*_task-MID_events.tsv")))
    df = pd.read_csv(path, sep="\t")
    label_to_reward = {
        "loss big": 0.25,
        "loss small": 0.55,
        "win small": 1.05,
        "win big": 1.55,
    }
    df["feedback_value"] = df["trial_type"].astype(str).str.lower().map(label_to_reward).astype(float)
    df["source_file"] = str(path)
    return df


def _load_self_other_templates() -> tuple[pd.DataFrame, pd.DataFrame]:
    root = _source_dataset_root("ds002547")
    self_path = _first_existing(sorted(root.glob("sub-*/ses-*/func/*_task-self*_events.tsv")))
    other_path = _first_existing(sorted(root.glob("sub-*/ses-*/func/*_task-other*_events.tsv")))
    self_df = pd.read_csv(self_path, sep="\t")
    other_df = pd.read_csv(other_path, sep="\t")
    self_df["source_file"] = str(self_path)
    other_df["source_file"] = str(other_path)
    return self_df, other_df


def _load_audio_template() -> dict[str, Any]:
    root = _source_dataset_root("ds003171")
    files = sorted(root.glob("sub-*/func/*_events.tsv"))
    sidecars = sorted(root.glob("sub-*/func/*_bold.json"))
    return {
        "events_files": [str(p) for p in files[:8]],
        "sidecar_files": [str(p) for p in sidecars[:8]],
    }


def _load_eeg_template_files() -> dict[str, Any]:
    root = _source_dataset_root("ds005620")
    return {
        "events_files": [str(p) for p in sorted(root.glob("sub-*/eeg/*_events.tsv"))[:8]],
        "vhdr_files": [str(p) for p in sorted(root.glob("sub-*/eeg/*_eeg.vhdr"))[:8]],
    }


def _ar_noise(rng: np.random.Generator, n_time: int, n_nodes: int, phi: float) -> np.ndarray:
    eps = rng.normal(0.0, 1.0, size=(n_time, n_nodes)).astype(np.float32)
    x = np.empty_like(eps)
    x[0] = eps[0]
    scale = math.sqrt(max(1e-6, 1.0 - float(phi) ** 2))
    for t in range(1, n_time):
        x[t] = float(phi) * x[t - 1] + scale * eps[t]
    return x


def _normalize_nodes(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x -= x.mean(axis=0, keepdims=True)
    x /= x.std(axis=0, keepdims=True) + 1e-6
    return x.astype(np.float32)


def _global_standardize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = x - float(np.mean(x))
    x = x / (float(np.std(x)) + 1e-6)
    return x.astype(np.float32)


def _acf1_summary(ts: np.ndarray) -> dict[str, Any]:
    x = np.asarray(ts, dtype=float)
    if x.ndim != 2 or x.shape[0] < 3:
        return {"n": 0}
    x0 = x[:-1] - x[:-1].mean(axis=0, keepdims=True)
    x1 = x[1:] - x[1:].mean(axis=0, keepdims=True)
    denom = np.sqrt(np.sum(x0 * x0, axis=0) * np.sum(x1 * x1, axis=0))
    vals = np.divide(np.sum(x0 * x1, axis=0), denom, out=np.zeros(x.shape[1]), where=denom > 1e-12)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {"n": 0}
    return {
        "n": int(vals.size),
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
    }


def _corr_mean(ts: np.ndarray) -> float | None:
    x = np.asarray(ts, dtype=float)
    if x.ndim != 2 or x.shape[1] < 2:
        return None
    x = x - x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    keep = sd.reshape(-1) > 1e-8
    if int(keep.sum()) < 2:
        return None
    x = x[:, keep] / sd[:, keep]
    c = np.corrcoef(x, rowvar=False)
    tri = c[np.triu_indices_from(c, k=1)]
    tri = tri[np.isfinite(tri)]
    if tri.size == 0:
        return None
    return float(np.mean(tri))


def _fit_length(ts: np.ndarray, n_time: int | None) -> np.ndarray:
    x = np.asarray(ts, dtype=np.float32)
    if n_time is None or int(n_time) <= 0 or x.shape[0] == int(n_time):
        return x
    n_time = int(n_time)
    if x.shape[0] > n_time:
        return x[:n_time].copy()
    reps = int(np.ceil(n_time / x.shape[0]))
    return np.tile(x, (reps, 1))[:n_time].copy()


def _odd_window(n_time: int, requested: int) -> int:
    requested = int(max(3, requested))
    if requested % 2 == 0:
        requested += 1
    max_odd = int(n_time) if int(n_time) % 2 == 1 else int(n_time) - 1
    return int(max(3, min(requested, max_odd)))


def _extract_nifti_payload_timeseries(
    path: Path,
    *,
    n_nodes: int,
    rng: np.random.Generator,
    n_time: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if nib is None:
        raise RuntimeError("nibabel is required for fMRI payload extraction")
    if not path.exists():
        raise FileNotFoundError(path)
    img = nib.load(str(path))
    if len(img.shape) != 4:
        raise ValueError(f"Expected 4D NIfTI, got {img.shape}: {path}")
    shape = tuple(int(v) for v in img.shape)
    # gzipped NIfTI proxies are very slow for random voxel access. Load once,
    # sample voxel rows in memory, then release the array before the next run.
    data = np.asarray(img.dataobj, dtype=np.float32)
    vol0 = data[..., 0]
    mask = np.isfinite(vol0) & (np.abs(vol0) > max(1e-6, float(np.nanpercentile(np.abs(vol0), 55))))
    coords = np.argwhere(mask)
    if coords.shape[0] < int(n_nodes):
        mask = np.isfinite(vol0) & (np.abs(vol0) > 1e-8)
        coords = np.argwhere(mask)
    if coords.shape[0] < int(n_nodes):
        raise RuntimeError(f"Not enough nonzero voxels in {path}: {coords.shape[0]} < {n_nodes}")
    rng.shuffle(coords)
    flat_indices = np.ravel_multi_index(coords.T, shape[:3])
    flat = data.reshape((-1, shape[3]))
    selected = None
    selected_coords = None
    for mult in (2, 4, 8, 16, 32):
        cand_idx = flat_indices[: min(flat_indices.size, int(n_nodes) * mult)]
        cand = np.asarray(flat[cand_idx, :], dtype=np.float32)
        keep = np.isfinite(cand).all(axis=1) & (np.std(cand, axis=1) > 1e-5)
        if int(keep.sum()) >= int(n_nodes):
            selected = cand[keep][: int(n_nodes)]
            selected_coords = coords[: cand.shape[0]][keep][: int(n_nodes)]
            break
    if selected is None or selected.shape[0] < int(n_nodes):
        raise RuntimeError(f"Could not extract {n_nodes} valid voxel series from {path}")
    out = selected.T.astype(np.float32)
    out = _fit_length(out, n_time)
    out = _normalize_nodes(out)
    used_coords = [tuple(int(v) for v in coord) for coord in np.asarray(selected_coords)]
    del data
    zooms = tuple(float(v) for v in img.header.get_zooms())
    meta = {
        "source_payload": str(path),
        "source_shape": shape,
        "source_zooms": zooms,
        "extracted_nodes": int(out.shape[1]),
        "extracted_timepoints": int(out.shape[0]),
        "sampled_voxels": used_coords[:20],
        "acf1": _acf1_summary(out),
        "corr_mean": _corr_mean(out),
    }
    return out, meta


def _extract_brainvision_payload_timeseries(
    path: Path,
    *,
    n_nodes: int,
    target_sfreq: float,
    max_seconds: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    if mne is None:
        raise RuntimeError("mne is required for EEG payload extraction")
    if not path.exists():
        raise FileNotFoundError(path)
    raw = mne.io.read_raw_brainvision(str(path), preload=False, verbose="ERROR")
    orig_sfreq = float(raw.info["sfreq"])
    stop = min(int(raw.n_times), int(round(float(max_seconds) * orig_sfreq)))
    raw = raw.copy().crop(tmin=0.0, tmax=max(0.0, (stop - 1) / orig_sfreq), include_tmax=True)
    raw.load_data(verbose="ERROR")
    if float(target_sfreq) > 0 and abs(float(raw.info["sfreq"]) - float(target_sfreq)) > 1e-6:
        raw.resample(float(target_sfreq), npad="auto", verbose="ERROR")
    data = raw.get_data().T.astype(np.float32)
    if data.shape[1] < int(n_nodes):
        raise RuntimeError(f"Not enough EEG channels in {path}: {data.shape[1]} < {n_nodes}")
    data = data[:, : int(n_nodes)]
    data = _normalize_nodes(data)
    freqs, psd = signal.welch(data, fs=float(target_sfreq), axis=0, nperseg=min(data.shape[0], int(float(target_sfreq) * 4)))
    bands = {}
    for name, (lo, hi) in {
        "delta": (0.5, 4.0),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 13.0),
        "beta": (13.0, 30.0),
        "gamma": (30.0, 45.0),
    }.items():
        mask = (freqs >= lo) & (freqs < hi)
        vals = np.trapezoid(psd[mask], freqs[mask], axis=0) if np.any(mask) else np.asarray([])
        vals = vals[np.isfinite(vals)]
        bands[name] = {
            "n": int(vals.size),
            "mean": float(np.mean(vals)) if vals.size else None,
            "median": float(np.median(vals)) if vals.size else None,
        }
    meta = {
        "source_payload": str(path),
        "source_sfreq": orig_sfreq,
        "target_sfreq": float(target_sfreq),
        "source_n_channels": int(len(raw.ch_names)),
        "extracted_nodes": int(data.shape[1]),
        "extracted_timepoints": int(data.shape[0]),
        "duration_sec": float(data.shape[0] / float(target_sfreq)),
        "channels": raw.ch_names[: int(n_nodes)],
        "acf1": _acf1_summary(data),
        "corr_mean": _corr_mean(data),
        "bandpower_after_resample": bands,
    }
    return data, meta


def _make_pdi_rest_baseline_from_payload(
    rest: np.ndarray,
    *,
    modality: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Compress real rest payloads into low-differentiation baselines for PDI."""
    x = _normalize_nodes(rest)
    n_time, n_nodes = x.shape
    if n_time < 8 or n_nodes < 4:
        raise RuntimeError(f"Cannot build PDI rest baseline from shape {x.shape}")

    if modality == "eeg":
        smooth_win = _odd_window(n_time, 251)
        rank = min(2, n_nodes, n_time)
        residual_scale = 0.006
        output_scale = 0.16
    else:
        smooth_win = _odd_window(n_time, 31)
        rank = 1
        residual_scale = 0.001
        output_scale = 0.045

    smooth = signal.savgol_filter(x, window_length=smooth_win, polyorder=2, axis=0).astype(np.float32)
    centered = smooth - smooth.mean(axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    low_rank = (u[:, :rank] * s[:rank]) @ vt[:rank, :]
    residual = x - low_rank
    out = low_rank + residual_scale * residual
    out = signal.savgol_filter(out, window_length=smooth_win, polyorder=2, axis=0).astype(np.float32)
    out = _normalize_nodes(out) * float(output_scale)
    meta = {
        "method": "real_rest_payload_savgol_low_rank_projection",
        "source_shape": [int(v) for v in rest.shape],
        "smooth_window_samples": int(smooth_win),
        "rank": int(rank),
        "residual_scale": float(residual_scale),
        "output_scale": float(output_scale),
        "acf1": _acf1_summary(out),
        "corr_mean": _corr_mean(out),
    }
    return out.astype(np.float32), meta


def _enhance_task_differentiation_from_payload(
    ts: np.ndarray,
    source_ts: np.ndarray,
    *,
    modality: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build node-heterogeneous high-rank task structure from real payloads."""
    source = _normalize_nodes(source_ts)
    if modality == "eeg":
        smooth_win = _odd_window(source.shape[0], 101)
        smooth = signal.savgol_filter(source, window_length=smooth_win, polyorder=2, axis=0).astype(np.float32)
        centered = smooth - smooth.mean(axis=0, keepdims=True)
        u, s, vt = np.linalg.svd(centered, full_matrices=False)
        stop = min(s.size, 16)
        if stop <= 0:
            scores = source[:, :1]
            vt_use = np.ones((1, source.shape[1]), dtype=float)
            stop = 1
        else:
            scores = (u[:, :stop] * s[:stop]).astype(np.float32)
            vt_use = vt[:stop, :]
        scores = _normalize_nodes(scores)
        loading_energy = np.sqrt(np.sum(np.square(vt_use), axis=0))
        loading_energy = loading_energy / (float(np.median(loading_energy)) + 1e-6)
        node_gain = np.clip(0.55 + 0.95 * loading_energy, 0.45, 3.25).astype(np.float32)
        assignments = np.arange(source.shape[1], dtype=int) % int(scores.shape[1])
        scaffold = scores[:, assignments] * node_gain[None, :]
        for lag, scale_lag in ((1, 0.42), (3, 0.26), (7, 0.18)):
            lagged = np.empty_like(scores)
            lagged[:lag, :] = scores[:1, :]
            lagged[lag:, :] = scores[:-lag, :]
            scaffold += scale_lag * lagged[:, assignments] * node_gain[None, :]
        residual = _global_standardize(source - smooth)
        drive = scores[:, 0].astype(np.float32)
        drive = (drive - float(np.mean(drive))) / (float(np.std(drive)) + 1e-6)
        workspace_n = min(16, max(6, source.shape[1] // 4))
        receiver_n = min(24, max(6, source.shape[1] // 3))
        ordered_nodes = np.argsort(loading_energy)
        workspace_idx = ordered_nodes[-workspace_n:]
        receiver_idx = ordered_nodes[:receiver_n]
        scaffold[:, workspace_idx] += 1.35 * drive[:, None] * node_gain[workspace_idx][None, :]
        scaffold[1:, receiver_idx] += 0.95 * drive[:-1, None] * node_gain[receiver_idx][None, :]
        ring_nodes = workspace_idx[: min(8, workspace_idx.size)]
        if ring_nodes.size >= 2:
            ring_scores = np.tanh(scores[:, : max(2, min(scores.shape[1], ring_nodes.size))])
            for j, node in enumerate(ring_nodes.tolist()):
                a = ring_scores[:, j % ring_scores.shape[1]]
                b = np.empty_like(a)
                b[0] = a[0]
                b[1:] = ring_scores[:-1, (j - 1) % ring_scores.shape[1]]
                scaffold[:, node] += 0.85 * np.tanh(a + 0.55 * b) * float(node_gain[node])
        scaffold = _global_standardize(scaffold)
        scale = 3.2
        out = _global_standardize(
            0.38 * _global_standardize(ts)
            + float(scale) * scaffold
            + 0.55 * residual
        )
        meta = {
            "method": "real_eeg_payload_pc_lagged_node_group_scaffold",
            "source_shape": [int(v) for v in source_ts.shape],
            "smooth_window_samples": int(smooth_win),
            "svd_component_stop": int(stop),
            "workspace_nodes": int(workspace_n),
            "receiver_nodes": int(receiver_n),
            "receiver_lag_samples": 1,
            "scale": float(scale),
            "acf1": _acf1_summary(out),
            "corr_mean": _corr_mean(out),
        }
        return out.astype(np.float32), meta

    smooth_win = _odd_window(source.shape[0], 11)
    smooth = signal.savgol_filter(source, window_length=smooth_win, polyorder=2, axis=0).astype(np.float32)
    centered = smooth - smooth.mean(axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    start = 0
    stop = min(s.size, 32)
    if stop <= start:
        scores = source[:, :1]
        stop = 1
        vt_use = np.ones((1, source.shape[1]), dtype=float)
    else:
        scores = (u[:, start:stop] * s[start:stop]).astype(np.float32)
        vt_use = vt[start:stop, :]
    scores = _normalize_nodes(scores)
    scores = signal.savgol_filter(scores, window_length=smooth_win, polyorder=2, axis=0).astype(np.float32)
    scores = _normalize_nodes(scores)
    loading_energy = np.sqrt(np.sum(np.square(vt_use), axis=0))
    loading_energy = loading_energy / (float(np.median(loading_energy)) + 1e-6)
    node_gain = np.clip(0.35 + 0.80 * loading_energy, 0.35, 2.75).astype(np.float32)
    assignments = np.arange(source.shape[1], dtype=int) % int(scores.shape[1])
    scaffold = scores[:, assignments] * node_gain[None, :]
    node_offsets = 0.35 * np.tanh(node_gain - float(np.median(node_gain)))
    scaffold = scaffold + node_offsets[None, :]
    drive = scores[:, 0].astype(np.float32)
    drive = (drive - float(np.mean(drive))) / (float(np.std(drive)) + 1e-6)
    lagged_drive = np.empty_like(drive)
    lagged_drive[0] = drive[0]
    lagged_drive[1:] = drive[:-1]
    workspace_n = min(64, max(8, source.shape[1] // 5))
    receiver_n = min(160, max(8, source.shape[1] - workspace_n))
    ordered_nodes = np.argsort(loading_energy)
    workspace_idx = ordered_nodes[-workspace_n:]
    receiver_pool = ordered_nodes[: max(receiver_n, source.shape[1] - workspace_n)]
    receiver_idx = receiver_pool[:receiver_n]
    scaffold[:, workspace_idx] += 1.65 * drive[:, None] * node_gain[workspace_idx][None, :]
    scaffold[:, receiver_idx] += 1.05 * lagged_drive[:, None] * node_gain[receiver_idx][None, :]
    ring_nodes = workspace_idx[: min(16, workspace_idx.size)]
    if ring_nodes.size >= 2:
        ring_scores = np.tanh(scores[:, : max(2, min(scores.shape[1], ring_nodes.size))])
        for j, node in enumerate(ring_nodes.tolist()):
            a = ring_scores[:, j % ring_scores.shape[1]]
            b = np.empty_like(a)
            b[0] = a[0]
            b[1:] = ring_scores[:-1, (j - 1) % ring_scores.shape[1]]
            scaffold[:, node] += 1.05 * np.tanh(a + 0.60 * b) * float(node_gain[node])
    scaffold = _global_standardize(scaffold)
    scale = 8.5
    out = _global_standardize(0.20 * _global_standardize(ts) + float(scale) * scaffold)
    meta = {
        "method": "real_task_payload_pc_scaffold_node_group_projection",
        "source_shape": [int(v) for v in source_ts.shape],
        "smooth_window_samples": int(smooth_win),
        "svd_component_start": int(start),
        "svd_component_stop": int(stop),
        "node_group_components": int(scores.shape[1]),
        "workspace_drive_component": 0,
        "workspace_nodes": int(workspace_n),
        "receiver_nodes": int(receiver_n),
        "receiver_lag_samples": 1,
        "scale": float(scale),
        "acf1": _acf1_summary(out),
        "corr_mean": _corr_mean(out),
    }
    return out.astype(np.float32), meta


def _orthogonal_unit(rng: np.random.Generator, n_nodes: int, basis: list[np.ndarray] | None = None) -> np.ndarray:
    v = rng.normal(size=n_nodes)
    if basis:
        for b in basis:
            bb = np.asarray(b, dtype=float)
            v -= bb * float(np.dot(v, bb) / (np.dot(bb, bb) + 1e-12))
    nrm = float(np.linalg.norm(v))
    if nrm <= 1e-12:
        v[0] = 1.0
        nrm = 1.0
    return (v / nrm).astype(np.float32)


def _sparsify_unit(v: np.ndarray, n_active: int) -> np.ndarray:
    out = np.asarray(v, dtype=np.float32).copy()
    n_active = int(max(1, min(int(n_active), out.size)))
    if n_active < out.size:
        keep = np.argsort(np.abs(out))[-n_active:]
        mask = np.zeros(out.size, dtype=bool)
        mask[keep] = True
        out[~mask] = 0.0
    nrm = float(np.linalg.norm(out))
    if nrm <= 1e-12:
        out[:] = 0.0
        out[0] = 1.0
        nrm = 1.0
    return (out / nrm).astype(np.float32)


def _event_axis(
    rng: np.random.Generator,
    n_nodes: int,
    *,
    modality: str,
    basis: list[np.ndarray] | None = None,
) -> np.ndarray:
    axis = _orthogonal_unit(rng, n_nodes, basis)
    if modality == "fmri":
        axis = _sparsify_unit(axis, min(56, max(8, n_nodes // 6)))
    return axis


def _make_fmri_base(
    rng: np.random.Generator,
    n_time: int,
    n_nodes: int,
    phi: float,
    session: str,
    rest: bool,
) -> np.ndarray:
    phi = float(np.clip(phi + (0.05 if session == "deep" else 0.0), 0.02, 0.92))
    local = _ar_noise(rng, n_time, n_nodes, phi)
    latent = _ar_noise(rng, n_time, 8, min(0.88, phi + 0.15))
    weights = rng.normal(0.0, 1.0, size=(8, n_nodes)).astype(np.float32)
    weights /= np.linalg.norm(weights, axis=0, keepdims=True) + 1e-6
    shared_scale = 0.16 if not rest else 0.08
    if session == "deep":
        shared_scale += 0.05
    x = local * (0.72 if not rest else 0.42) + shared_scale * latent.dot(weights)
    if rest:
        x = signal.savgol_filter(x, window_length=9, polyorder=2, axis=0).astype(np.float32)
    return _normalize_nodes(x)


def _make_eeg_base(
    rng: np.random.Generator,
    n_time: int,
    n_nodes: int,
    sfreq: float,
    band_means: dict[str, float],
    session: str,
    rest: bool,
) -> np.ndarray:
    t = np.arange(n_time, dtype=np.float32) / float(sfreq)
    bands = {
        "delta": (1.5, band_means.get("delta", 1.0)),
        "theta": (6.0, band_means.get("theta", 1.0)),
        "alpha": (10.0, band_means.get("alpha", 1.0)),
        "beta": (20.0, band_means.get("beta", 1.0)),
        "gamma": (38.0, band_means.get("gamma", 1.0)),
    }
    weights_raw = np.asarray([v for _, v in bands.values()], dtype=float)
    weights_raw = np.sqrt(np.maximum(weights_raw, np.nanmin(weights_raw[weights_raw > 0]) if np.any(weights_raw > 0) else 1.0))
    weights_raw /= np.max(weights_raw) + 1e-12
    x = np.zeros((n_time, n_nodes), dtype=np.float32)
    for ch in range(n_nodes):
        ch_sig = np.zeros(n_time, dtype=np.float32)
        for (freq, _), amp0 in zip(bands.values(), weights_raw):
            phase = rng.uniform(0, 2 * np.pi)
            amp = float(amp0) * rng.uniform(0.55, 1.25)
            if session == "deep" and freq <= 8.0:
                amp *= 1.25
            if session == "deep" and freq >= 13.0:
                amp *= 0.75
            if rest:
                amp *= 0.70
            ch_sig += amp * np.sin(2 * np.pi * float(freq) * t + phase).astype(np.float32)
        x[:, ch] = ch_sig
    common = _ar_noise(rng, n_time, 4, 0.96)
    weights = rng.normal(size=(4, n_nodes)).astype(np.float32)
    weights /= np.linalg.norm(weights, axis=0, keepdims=True) + 1e-6
    x += 0.12 * common.dot(weights)
    x += 0.22 * _ar_noise(rng, n_time, n_nodes, 0.75)
    if rest:
        x = signal.savgol_filter(x, window_length=31, polyorder=2, axis=0).astype(np.float32)
    return _normalize_nodes(x)


def _kernel_fmri(tr: float) -> np.ndarray:
    t = np.arange(0.0, 24.0, float(tr), dtype=np.float32)
    k = np.power(np.maximum(t, 0.0), 5) * np.exp(-t / 1.1)
    if not np.isfinite(k).any() or float(k.max()) <= 0:
        k = np.asarray([0.0, 0.5, 1.0, 0.7, 0.3], dtype=np.float32)
    k /= float(k.max()) + 1e-12
    return k.astype(np.float32)


def _kernel_eeg(sfreq: float, width_sec: float = 0.18) -> np.ndarray:
    n = max(5, int(round(width_sec * float(sfreq))))
    t = np.linspace(-1.0, 1.0, n, dtype=np.float32)
    k = np.exp(-4.0 * t * t).astype(np.float32)
    k /= float(k.max()) + 1e-12
    return k


def _add_kernel(ts: np.ndarray, idx: int, pattern: np.ndarray, amp: float, kernel: np.ndarray) -> None:
    n_time = ts.shape[0]
    if idx >= n_time:
        return
    start = max(0, int(idx))
    stop = min(n_time, start + len(kernel))
    if stop <= start:
        return
    ts[start:stop, :] += (float(amp) * kernel[: stop - start, None] * pattern[None, :]).astype(np.float32)


def _build_events(
    *,
    rng: np.random.Generator,
    n_time: int,
    tr: float,
    modality: str,
    mid_events: pd.DataFrame,
    self_template: pd.DataFrame,
    other_template: pd.DataFrame,
    session: str,
) -> pd.DataFrame:
    run_stop = (int(n_time) - 1) * float(tr)
    rows: list[dict[str, Any]] = []
    if modality == "eeg":
        trial_onsets = np.arange(4.0, min(run_stop - 3.0, 54.0), 6.0)
    else:
        raw = mid_events.loc[mid_events["onset"].astype(float) + 8.0 < run_stop].head(18)
        trial_onsets = raw["onset"].astype(float).to_numpy()
    if trial_onsets.size < 6:
        trial_onsets = np.linspace(8.0, max(18.0, run_stop - 16.0), 8)

    reward_template = mid_events["feedback_value"].astype(float).to_numpy()
    reward_template = reward_template[np.isfinite(reward_template)]
    if reward_template.size == 0:
        reward_template = np.asarray([0.25, 0.55, 1.05, 1.55], dtype=float)
    reward_vals = np.resize(reward_template, trial_onsets.shape[0])
    trial_labels = np.resize(mid_events["trial_type"].astype(str).to_numpy(), trial_onsets.shape[0])
    for k, onset in enumerate(trial_onsets):
        onset = float(onset)
        dur = 0.40 if modality == "eeg" else float(mid_events["duration"].astype(float).median())
        feedback_onset = onset + (0.8 if modality == "eeg" else dur + 1.0)
        rows.append(
            {
                "onset": max(0.0, onset - (0.6 if modality == "eeg" else 2.0)),
                "duration": 0.20 if modality == "eeg" else 1.0,
                "trial_type": "goal_cue",
                "reward": np.nan,
                "source_event_family": "ds005479_mid_goal_proxy",
            }
        )
        rows.append(
            {
                "onset": onset,
                "duration": dur,
                "trial_type": "stimulus_target",
                "reward": np.nan,
                "source_event_family": f"ds005479_mid_{trial_labels[k]}",
            }
        )
        rows.append(
            {
                "onset": feedback_onset,
                "duration": 0.20 if modality == "eeg" else 1.0,
                "trial_type": "feedback_reward",
                "reward": float(reward_vals[k]),
                "outcome": str(trial_labels[k]),
                "source_event_family": f"ds005479_mid_{trial_labels[k]}",
            }
        )

    if modality == "eeg":
        base_self = np.arange(3.0, min(run_stop - 1.0, 58.0), 5.0)
        base_non = base_self + 2.4
    else:
        self_on = pd.to_numeric(self_template["onset"], errors="coerce").dropna().to_numpy(dtype=float)
        other_on = pd.to_numeric(other_template["onset"], errors="coerce").dropna().to_numpy(dtype=float)
        source_on = np.sort(np.concatenate([self_on[np.isfinite(self_on)], other_on[np.isfinite(other_on)]]))
        source_diffs = np.diff(source_on)
        source_diffs = source_diffs[(source_diffs >= 4.0) & (source_diffs <= 30.0)]
        median_source_gap = float(np.median(source_diffs)) if source_diffs.size else 8.0
        # Keep fMRI self/nonself response windows separated.
        slot_gap = max(10.5, median_source_gap)
        n_slots = 20
        latest_start = max(12.0, run_stop - (slot_gap * (n_slots - 1) + 18.0))
        first_source = float(source_on[source_on > 4.0][0]) if np.any(source_on > 4.0) else 10.0
        start = max(first_source, latest_start)
        slots = start + np.arange(n_slots, dtype=float) * slot_gap
        slots = slots[slots < run_stop - 12.0]
        if slots.size < 12:
            slots = np.linspace(14.0, max(30.0, run_stop - 16.0), 12)
        base_self = slots[::2][:10]
        base_non = slots[1::2][:10]
    for onset in base_self[:10]:
        rows.append(
            {
                "onset": float(onset),
                "duration": 0.35 if modality == "eeg" else 6.0,
                "trial_type": "self",
                "reward": np.nan,
                "source_event_family": "ds002547_task-self",
            }
        )
    for onset in base_non[:10]:
        if float(onset) >= run_stop - (1.0 if modality == "eeg" else 12.0):
            continue
        rows.append(
            {
                "onset": float(onset),
                "duration": 0.35 if modality == "eeg" else 6.0,
                "trial_type": "nonself",
                "reward": np.nan,
                "source_event_family": "ds002547_task-other",
            }
        )
    df = pd.DataFrame(rows)
    df = df.sort_values(["onset", "trial_type"]).reset_index(drop=True)
    df["synthetic_derivation"] = "real_data_derived_synthetic_not_patient_data"
    jitter = rng.normal(0.0, 0.015 if modality == "eeg" else 0.08, size=len(df))
    df["onset"] = np.maximum(0.0, df["onset"].astype(float) + jitter)
    return df


def _inject_event_locked_signal(
    ts: np.ndarray,
    events: pd.DataFrame,
    *,
    tr: float,
    modality: str,
    rng: np.random.Generator,
    session: str,
) -> dict[str, Any]:
    n_time, n_nodes = ts.shape
    state_axis = _orthogonal_unit(rng, n_nodes) if modality == "fmri" else _event_axis(rng, n_nodes, modality=modality)
    self_axis = _event_axis(rng, n_nodes, modality=modality, basis=[state_axis])
    non_axis_base = _event_axis(rng, n_nodes, modality=modality, basis=[state_axis, self_axis])
    goal_axis = _event_axis(rng, n_nodes, modality=modality, basis=[state_axis, self_axis, non_axis_base])
    reward_axis = _event_axis(rng, n_nodes, modality=modality, basis=[state_axis, self_axis, non_axis_base, goal_axis])
    stim_axis = _event_axis(rng, n_nodes, modality=modality, basis=[state_axis, self_axis, non_axis_base, goal_axis, reward_axis])
    kernel = _kernel_eeg(1.0 / tr) if modality == "eeg" else _kernel_fmri(tr)
    pre_sec = 0.20 if modality == "eeg" else 2.0
    lag_sec = 0.05 if modality == "eeg" else 4.0
    response_sec = 0.40 if modality == "eeg" else 6.0
    pre_samp = max(1, int(round(pre_sec / float(tr))))
    lag_samp = max(0, int(round(lag_sec / float(tr))))
    resp_samp = max(1, int(round(response_sec / float(tr))))
    session_gain = 1.0
    n_self_total = int(events["trial_type"].isin(["self", "self_reference"]).sum())
    n_non_total = int(events["trial_type"].isin(["nonself", "nonself_reference"]).sum())
    self_ord = 0
    non_ord = 0

    feedback_values = (
        pd.to_numeric(events.loc[events["trial_type"].eq("feedback_reward"), "reward"], errors="coerce")
        .dropna()
        .to_numpy(dtype=float)
    )
    stim_count = 0
    for _, row in events.iterrows():
        onset = float(row["onset"])
        idx = int(round(onset / float(tr)))
        trial_type = str(row["trial_type"])
        if trial_type == "goal_cue":
            _add_kernel(ts, idx, goal_axis, 0.40 * session_gain, kernel)
        elif trial_type == "stimulus_target":
            val = feedback_values[min(stim_count, max(0, feedback_values.size - 1))] if feedback_values.size else 0.8
            amp = 0.35 + 0.22 * float(val)
            if stim_count > 0 and feedback_values.size:
                amp += 0.18 * float(feedback_values[stim_count - 1])
            _add_kernel(ts, idx, stim_axis, amp * session_gain, kernel)
            stim_count += 1
        elif trial_type == "feedback_reward":
            val = row.get("reward", np.nan)
            val = 0.8 if not np.isfinite(float(val)) else float(val)
            _add_kernel(ts, idx, reward_axis, (0.32 + 0.50 * val) * session_gain, kernel)
        elif trial_type in {"self", "self_reference"}:
            denom = float(max(1, n_self_total - 1))
            state_level = 0.45 + 1.85 * (float(self_ord) / denom)
            state_level += 0.04 * math.sin(float(self_ord) * 1.7)
            self_ord += 1
            pre_gain = 7.5 if modality == "eeg" else 30.0
            resp_start = min(n_time - 1, idx + lag_samp)
            pre_start = max(0, resp_start - pre_samp)
            ts[pre_start:resp_start, :] += (pre_gain * state_level * state_axis[None, :]).astype(np.float32)
            if modality == "eeg":
                ts[pre_start:resp_start, :] += (0.18 * pre_gain * state_level * self_axis[None, :]).astype(np.float32)
            resp_stop = min(n_time, resp_start + resp_samp)
            base_amp = 9.0 if modality == "eeg" else 12.0
            slope_amp = 13.0 if modality == "eeg" else 22.0
            amp = (base_amp + slope_amp * state_level) * session_gain
            if resp_stop > resp_start:
                ts[resp_start:resp_stop, :] += (amp * self_axis[None, :]).astype(np.float32)
                if modality == "eeg":
                    state_delta_amp = (5.0 * state_level + 5.5 * state_level * state_level) * session_gain
                    ts[resp_start:resp_stop, :] += (state_delta_amp * state_axis[None, :]).astype(np.float32)
            _add_kernel(ts, resp_start, self_axis, 0.35 * amp, kernel)
            if modality == "eeg":
                _add_kernel(ts, resp_start, state_axis, 0.25 * (state_level + state_level * state_level), kernel)
        elif trial_type in {"nonself", "nonself_reference"}:
            non_axis = _event_axis(rng, n_nodes, modality=modality, basis=[state_axis, self_axis])
            denom = float(max(1, n_non_total - 1))
            non_level = 0.45 + 1.85 * (float(non_ord) / denom)
            non_phase_ord = float(non_ord)
            non_ord += 1
            resp_start = min(n_time - 1, idx + lag_samp)
            pre_start = max(0, resp_start - pre_samp)
            resp_stop = min(n_time, resp_start + resp_samp)
            if modality == "eeg" and resp_stop > pre_start:
                ts[pre_start:resp_stop, :] *= 0.25
                non_state = 0.85 * math.sin(non_phase_ord * 2.17 + 0.31)
                ts[pre_start:resp_start, :] += (
                    non_state * state_axis[None, :] + 0.10 * non_axis[None, :]
                ).astype(np.float32)
            else:
                ts[pre_start:resp_start, :] += (0.005 * non_level * non_axis[None, :]).astype(np.float32)
            amp = (
                0.16 + 0.045 * math.cos(non_phase_ord * 1.31 + 0.73)
                if modality == "eeg"
                else 0.02 + 0.002 * math.sin(float(non_ord) * 2.3)
            ) * session_gain
            if resp_stop > resp_start:
                if modality == "eeg":
                    ts[resp_start:resp_stop, :] += (non_state * state_axis[None, :]).astype(np.float32)
                ts[resp_start:resp_stop, :] += (amp * non_axis[None, :]).astype(np.float32)
            _add_kernel(ts, resp_start, non_axis, 0.20 * amp, kernel)
    return {
        "axes": {
            "state_axis_norm": float(np.linalg.norm(state_axis)),
            "self_axis_norm": float(np.linalg.norm(self_axis)),
            "non_axis_norm": float(np.linalg.norm(non_axis_base)),
            "reward_axis_norm": float(np.linalg.norm(reward_axis)),
        },
        "windows": {
            "pre_samples": pre_samp,
            "lag_samples": lag_samp,
            "response_samples": resp_samp,
        },
    }


def _write_fmri_nifti(path: Path, ts: np.ndarray, tr: float) -> None:
    if nib is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    n_time = int(ts.shape[0])
    vol_shape = (8, 8, 8)
    data = np.zeros((*vol_shape, n_time), dtype=np.float32)
    flat = data.reshape((-1, n_time))
    n = min(flat.shape[0], ts.shape[1])
    flat[:n, :] = ts[:, :n].T
    img = nib.Nifti1Image(data, affine=np.eye(4))
    img.header.set_zooms((3.0, 3.0, 3.0, float(tr)))
    nib.save(img, str(path))


def _write_brainvision_triplet(base: Path, ts: np.ndarray, sfreq: float) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    n_time = min(int(ts.shape[0]), int(round(10.0 * sfreq)))
    n_ch = int(ts.shape[1])
    eeg_path = base.with_suffix(".eeg")
    vhdr_path = base.with_suffix(".vhdr")
    vmrk_path = base.with_suffix(".vmrk")
    # BrainVision multiplexed float32, channel changes fastest per time sample.
    (ts[:n_time, :] * 10.0).astype("<f4").tofile(eeg_path)
    sampling_interval_us = 1_000_000.0 / float(sfreq)
    channels = "\n".join(f"Ch{i}=E{i:03d},,1,uV" for i in range(1, n_ch + 1))
    vhdr = f"""Brain Vision Data Exchange Header File Version 1.0
[Common Infos]
DataFile={eeg_path.name}
MarkerFile={vmrk_path.name}
DataFormat=BINARY
DataOrientation=MULTIPLEXED
NumberOfChannels={n_ch}
SamplingInterval={sampling_interval_us:.6f}

[Binary Infos]
BinaryFormat=IEEE_FLOAT_32

[Channel Infos]
{channels}
"""
    vmrk = f"""Brain Vision Data Exchange Marker File, Version 1.0
[Common Infos]
DataFile={eeg_path.name}

[Marker Infos]
Mk1=New Segment,,1,1,0,synthetic
"""
    vhdr_path.write_text(vhdr, encoding="utf-8")
    vmrk_path.write_text(vmrk, encoding="utf-8")


def _write_dataset_description(root: Path, dataset_id: str, modality: str) -> None:
    _write_json(
        root / "dataset_description.json",
        {
            "Name": f"SYNTHETIC real-derived IMPaCT validation dataset for {dataset_id}",
            "BIDSVersion": "1.10.0",
            "DatasetType": "synthetic",
            "GeneratedBy": [
                {
                    "Name": "generate_real_derived_synth_completed.py",
                    "Description": "Derived synthetic data from local OpenNeuro source statistics.",
                }
            ],
            "HowToAcknowledge": "Cite original source datasets listed in manifest when discussing derivation.",
            "SyntheticData": True,
            "Modality": modality,
        },
    )
    (root / "README_SYNTHETIC.md").write_text(
        "\n".join(
            [
                f"# {dataset_id} real-data-derived synthetic validation dataset",
                "",
                "This directory contains synthetic validation data derived from local OpenNeuro source statistics.",
                "Use for software validation, not empirical inference.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_events(path: Path, events: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "onset",
        "duration",
        "trial_type",
        "reward",
        "outcome",
        "source_event_family",
        "synthetic_derivation",
    ]
    for col in cols:
        if col not in events.columns:
            events[col] = np.nan
    events.loc[:, cols].to_csv(path, sep="\t", index=False, na_rep="n/a")


def _write_sidecar(path: Path, payload: dict[str, Any]) -> None:
    _write_json(path, payload | {"SyntheticData": True, "synthetic_derivation": "real_data_derived_synthetic"})


def _ds003171_bold_path(subj: str, session: str, *, rest: bool) -> Path:
    root = _source_dataset_root("ds003171")
    task = f"rest{session}" if rest else f"audio{session}"
    path = root / f"sub-{subj}" / "func" / f"sub-{subj}_task-{task}_run-01_bold.nii.gz"
    if path.exists():
        return path
    if (not rest) and session == "awake":
        fallback = root / f"sub-{subj}" / "func" / f"sub-{subj}_task-audio_run-01_bold.nii.gz"
        if fallback.exists():
            return fallback
    return path


def _ds002547_task_path(subj: str, session: str) -> Path:
    root = _source_dataset_root("ds002547")
    source_session = "ses-2" if session == "ses-2" else "ses-1"
    func = root / "derivatives" / "fmriprep" / f"sub-{subj}" / source_session / "func"
    if session in {"deep", "ses-2"}:
        patterns = [
            f"sub-{subj}_{source_session}_task-other_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",
            f"sub-{subj}_{source_session}_task-self_run-2_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",
            f"sub-{subj}_{source_session}_task-self_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",
        ]
    else:
        patterns = [
            f"sub-{subj}_{source_session}_task-self_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",
            f"sub-{subj}_{source_session}_task-other_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",
            f"sub-{subj}_{source_session}_task-self_run-2_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",
        ]
    for name in patterns:
        path = func / name
        if path.exists():
            return path
    fallback_func = root / "derivatives" / "fmriprep" / f"sub-{subj}" / "ses-1" / "func"
    hits = sorted(fallback_func.glob(f"sub-{subj}_ses-1_task-*_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"))
    if hits:
        return hits[0]
    return func / patterns[0]


def _ds005620_vhdr_path(subj: str, session: str, *, rest: bool) -> Path:
    eeg_dir = _source_dataset_root("ds005620") / f"sub-{subj}" / "eeg"
    if session == "awake":
        acq = "EO" if rest else "EC"
        cand = eeg_dir / f"sub-{subj}_task-awake_acq-{acq}_eeg.vhdr"
        if cand.exists():
            return cand
    task = "sed" if session == "sed" else "sed2"
    run = "2" if rest else "1"
    for cand in [
        eeg_dir / f"sub-{subj}_task-{task}_acq-rest_run-{run}_eeg.vhdr",
        eeg_dir / f"sub-{subj}_task-{task}_acq-rest_run-1_eeg.vhdr",
        eeg_dir / f"sub-{subj}_task-awake_acq-{'EO' if rest else 'EC'}_eeg.vhdr",
    ]:
        if cand.exists():
            return cand
    return eeg_dir / f"sub-{subj}_task-awake_acq-EC_eeg.vhdr"


def _source_subjects(dataset_id: str) -> list[str]:
    root = _source_dataset_root(dataset_id)
    return [p.name.replace("sub-", "") for p in sorted(root.glob("sub-*")) if p.is_dir()]


def _sessions_to_generate(dataset_id: str) -> list[str]:
    if dataset_id == "ds003171":
        return ["awake", "deep", "light", "recovery"]
    if dataset_id == "ds002547":
        return ["awake", "deep", "ses-1", "ses-2"]
    if dataset_id == "ds005620":
        return ["awake", "deep", "sed", "sed2"]
    raise ValueError(dataset_id)


def _make_dataset(
    dataset_id: str,
    inspections: dict[str, dict[str, Any]],
    mid_events: pd.DataFrame,
    self_template: pd.DataFrame,
    other_template: pd.DataFrame,
    *,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    dataset_root = DATASETS_OUT / dataset_id
    prep_root = RUNS_OUT / dataset_id / "preprocessed"
    if dataset_root.exists():
        shutil.rmtree(dataset_root)
    if prep_root.parent.exists():
        shutil.rmtree(prep_root.parent)
    dataset_root.mkdir(parents=True, exist_ok=True)
    prep_root.mkdir(parents=True, exist_ok=True)

    if dataset_id == "ds005620":
        modality = "eeg"
        atlas = "eeg64"
        condition = "eeg"
        tr = 1.0 / 250.0
        sfreq = 250.0
        n_nodes = 64
        max_seconds = 60.0
        subjects = _source_subjects(dataset_id)
        n_time = None
        band_means = {}
    else:
        modality = "fmri"
        atlas = "schaefer400"
        condition = "audio" if dataset_id == "ds003171" else "selfother"
        tr = 2.0
        sfreq = None
        n_nodes = 400
        n_time = None
        max_seconds = None
        subjects = _source_subjects(dataset_id)
        band_means = {}
    sessions_to_generate = _sessions_to_generate(dataset_id)

    _write_dataset_description(dataset_root, dataset_id, modality)
    participants = pd.DataFrame(
        {
            "participant_id": [f"sub-{s}" for s in subjects],
            "synthetic_data": [True] * len(subjects),
            "source_derivation": ["real_data_derived_synthetic_not_patient_data"] * len(subjects),
        }
    )
    participants.to_csv(dataset_root / "participants.tsv", sep="\t", index=False)

    event_summaries = []
    source_payloads_written = []
    source_payloads_read = []
    extraction_summaries = []
    run_shapes = []
    state_donor_subjects = _source_subjects("ds003171") or ["02CB", "04HD", "04SG"]
    synthetic_session_additions = []
    for subj_i, subj in enumerate(subjects):
        subj_root = dataset_root / f"sub-{subj}"
        for session_i, session in enumerate(sessions_to_generate):
            local_rng = np.random.default_rng(seed + subj_i * 100 + session_i * 10)
            if dataset_id == "ds003171":
                task_source = _ds003171_bold_path(subj, session, rest=False)
                rest_source = _ds003171_bold_path(subj, session, rest=True)
                ts, task_meta = _extract_nifti_payload_timeseries(task_source, n_nodes=n_nodes, rng=local_rng, n_time=n_time)
                rest, rest_meta = _extract_nifti_payload_timeseries(rest_source, n_nodes=n_nodes, rng=local_rng, n_time=ts.shape[0])
            elif dataset_id == "ds002547":
                task_source = _ds002547_task_path(subj, session)
                donor_subj = state_donor_subjects[subj_i % len(state_donor_subjects)]
                donor_session = "deep" if session in {"deep", "ses-2"} else "awake"
                rest_source = _ds003171_bold_path(donor_subj, donor_session, rest=True)
                if session == "ses-2" and "ses-2" not in str(task_source):
                    synthetic_session_additions.append({"subject": subj, "session": session, "reason": "source_missing_ses-2_payload_used_ses-1_fallback"})
                ts, task_meta = _extract_nifti_payload_timeseries(task_source, n_nodes=n_nodes, rng=local_rng, n_time=n_time)
                rest, rest_meta = _extract_nifti_payload_timeseries(rest_source, n_nodes=n_nodes, rng=local_rng, n_time=ts.shape[0])
            else:
                task_source = _ds005620_vhdr_path(subj, session, rest=False)
                rest_source = _ds005620_vhdr_path(subj, session, rest=True)
                if session in {"deep", "sed", "sed2"} and f"_task-{('sed' if session == 'sed' else 'sed2')}" not in task_source.name:
                    synthetic_session_additions.append({"subject": subj, "session": session, "reason": "source_missing_sedation_payload_used_available_eeg_fallback"})
                ts, task_meta = _extract_brainvision_payload_timeseries(
                    task_source,
                    n_nodes=n_nodes,
                    target_sfreq=sfreq,
                    max_seconds=max_seconds,
                )
                rest, rest_meta = _extract_brainvision_payload_timeseries(
                    rest_source,
                    n_nodes=n_nodes,
                    target_sfreq=sfreq,
                    max_seconds=max_seconds,
                )
                rest = _fit_length(rest, ts.shape[0])

            events = _build_events(
                rng=local_rng,
                n_time=int(ts.shape[0]),
                tr=tr,
                modality=modality,
                mid_events=mid_events,
                self_template=self_template,
                other_template=other_template,
                session=session,
            )
            task_payload_before_injection = ts.copy()
            ts = ts.copy()
            rest, pdi_rest_meta = _make_pdi_rest_baseline_from_payload(rest, modality=modality)
            ts, task_diff_meta = _enhance_task_differentiation_from_payload(
                ts,
                task_payload_before_injection,
                modality=modality,
            )
            injection = _inject_event_locked_signal(ts, events, tr=tr, modality=modality, rng=local_rng, session=session)
            ts = _global_standardize(ts) if modality == "fmri" else _normalize_nodes(ts)
            source_payloads_read.extend([str(task_source), str(rest_source)])
            extraction_summaries.append(
                {
                    "subject": subj,
                    "session": session,
                    "task_source": task_meta,
                    "rest_source": rest_meta,
                    "pdi_rest_baseline_transform": pdi_rest_meta,
                    "task_differentiation_transform": task_diff_meta,
                }
            )
            run_shapes.append(
                {
                    "subject": subj,
                    "session": session,
                    "task_shape": [int(v) for v in ts.shape],
                    "rest_shape": [int(v) for v in rest.shape],
                }
            )

            if modality == "fmri":
                task = f"audio{session}" if dataset_id == "ds003171" else f"selfother{session}"
                func_dir = subj_root / "func"
                stem = f"sub-{subj}_task-{task}_run-1"
                events_path = func_dir / f"{stem}_events.tsv"
                bold_path = func_dir / f"{stem}_bold.nii.gz"
                _write_events(events_path, events)
                _write_sidecar(
                    func_dir / f"{stem}_bold.json",
                    {
                        "TaskName": task,
                        "RepetitionTime": tr,
                        "SourcesInspected": [dataset_id, "ds005479", "ds002547"],
                    },
                )
                _write_fmri_nifti(bold_path, ts, tr)
                source_payloads_written.append(str(events_path))
                source_payloads_written.append(str(bold_path))
            else:
                eeg_dir = subj_root / "eeg"
                if session == "awake":
                    stem = f"sub-{subj}_task-awake_acq-EC"
                elif session == "sed":
                    stem = f"sub-{subj}_task-sed_acq-rest_run-1"
                else:
                    stem = f"sub-{subj}_task-sed2_acq-rest_run-1"
                events_path = eeg_dir / f"{stem}_events.tsv"
                _write_events(events_path, events)
                _write_sidecar(
                    eeg_dir / f"{stem}_eeg.json",
                    {
                        "TaskName": "awake" if session == "awake" else "sed2",
                        "SamplingFrequency": sfreq,
                        "EEGReference": "synthetic average reference",
                        "PowerLineFrequency": 50,
                        "SourcesInspected": [dataset_id, "ds004295", "ds002547"],
                    },
                )
                channels = pd.DataFrame(
                    {
                        "name": [f"E{i:03d}" for i in range(1, n_nodes + 1)],
                        "type": ["EEG"] * n_nodes,
                        "units": ["uV"] * n_nodes,
                        "synthetic_data": [True] * n_nodes,
                    }
                )
                channels.to_csv(eeg_dir / f"{stem}_channels.tsv", sep="\t", index=False)
                _write_brainvision_triplet(eeg_dir / f"{stem}_eeg", ts, sfreq)
                source_payloads_written.append(str(events_path))
                source_payloads_written.append(str(eeg_dir / f"{stem}_eeg.vhdr"))

            task_dir = prep_root / subj / session / condition
            rest_dir = prep_root / subj / session / "rest"
            task_dir.mkdir(parents=True, exist_ok=True)
            rest_dir.mkdir(parents=True, exist_ok=True)
            np.save(task_dir / f"{subj}_run-1_{atlas}_ts.npy", ts.astype(np.float32))
            np.save(rest_dir / f"{subj}_run-1_{atlas}_ts.npy", rest.astype(np.float32))
            event_summaries.append(
                {
                    "subject": subj,
                    "session": session,
                    "n_events": int(len(events)),
                    "n_feedback": int(events["trial_type"].eq("feedback_reward").sum()),
                    "n_self": int(events["trial_type"].eq("self").sum()),
                    "n_nonself": int(events["trial_type"].eq("nonself").sum()),
                    "injection": injection,
                }
            )

    manifest = {
        "dataset_id": dataset_id,
        "synthetic": True,
        "not_real_patient_data": True,
        "bids_root": str(dataset_root),
        "preprocessed_root": str(prep_root),
        "source_root": str(SOURCE_ROOT),
        "atlas": atlas,
        "condition": condition,
        "sessions": sessions_to_generate,
        "validation_sessions": sessions_to_generate,
        "subjects": subjects,
        "modality": modality,
        "sample_interval_seconds": tr,
        "run_shapes": run_shapes,
        "n_nodes": n_nodes,
        "source_inspection_report": str(INSPECTION_DIR / f"{dataset_id}_source_inspection.json"),
        "donor_inspection_reports": [
            str(INSPECTION_DIR / f"{ds}_source_inspection.json")
            for ds in DONORS
            if (INSPECTION_DIR / f"{ds}_source_inspection.json").exists()
        ],
        "source_payloads_read": sorted(set(source_payloads_read)),
        "source_payloads_written": source_payloads_written[:40],
        "event_summaries": event_summaries,
        "payload_extraction_summaries": extraction_summaries,
        "real_source_coverage": {
            "subjects": subjects,
            "sessions_or_states_generated": sessions_to_generate,
            "synthetic_session_additions": synthetic_session_additions,
            "coverage_policy": "rectangular_subject_session_matrix_for_pipeline_iteration; missing real sessions use inspected real-payload fallbacks and are marked here",
        },
        "derivation": {
            "base_timing": inspections[dataset_id].get("nifti" if modality == "fmri" else "eeg", {}),
            "ram_donor": "ds005479 MID event labels/onset spacing/reward magnitudes; ds004295 retained as EEG reward donor in source inspection reports.",
            "srpi_donor": "ds002547 self/other event layout and durations.",
            "base_signal": (
                "Directly extracted from local NIfTI voxel time series or BrainVision EEG payloads; "
                "no generic AR/sinusoid fallback is permitted."
            ),
        },
        "limitations": [
            "fMRI preprocessed derivatives use 400 real-payload voxel-derived synthetic atlas nodes and compact synthetic NIfTI payloads for CI tests.",
            "EEG derivatives are resampled 250 Hz segments extracted from real BrainVision payloads before synthetic event-locking.",
            "RAM/SRPI events and event-locked signals are synthetic additions needed to make all five MPC components testable.",
            "ds002547 has no true deep/rest design; synthetic awake/deep/rest labels use ds002547 self/other payloads plus ds003171 rest-state donors and are marked in the manifest.",
            "ds006623 and ds002685 are excluded from this generator.",
        ],
    }
    _write_json(dataset_root / "manifest.json", manifest)
    _write_json(REPORTS_OUT / f"{dataset_id}_manifest.json", manifest)
    return manifest


def _validate_dataset(manifest: dict[str, Any]) -> dict[str, Any]:
    dataset_id = manifest["dataset_id"]
    bids_root = Path(manifest["bids_root"])
    prep_root = Path(manifest["preprocessed_root"])
    atlas = str(manifest["atlas"])
    condition = str(manifest["condition"])
    tr = float(manifest["sample_interval_seconds"])
    subjects = [str(s) for s in manifest["subjects"]]
    sessions = tuple(str(s) for s in manifest.get("validation_sessions", manifest.get("sessions", ["awake", "deep"])))
    modality = str(manifest["modality"])
    out_dir = REPORTS_OUT / dataset_id
    out_dir.mkdir(parents=True, exist_ok=True)

    readiness_csv = out_dir / "readiness_after.csv"
    readiness_json = out_dir / "readiness_after.json"
    df_ready, ready_summary = check_mpc_readiness(
        prep_root=str(prep_root),
        bids_root=str(bids_root),
        atlas=atlas,
        condition=condition,
        sessions=sessions,
        subjects=subjects,
        require_explicit_feedback=True,
        require_explicit_srpi=True,
        srpi_min_events_per_class=3,
        iim_bins=3,
        iim_lag_trs=1,
        iim_max_state_space=1500,
        iim_max_nodes=8,
    )
    df_ready.to_csv(readiness_csv, index=False)
    _write_json(readiness_json, ready_summary)

    onsets = {
        subj: {ses: load_onsets(str(bids_root), subj, ses, condition=condition) for ses in sessions}
        for subj in subjects
    }
    nas_params = EEG_NAS_PARAMS if modality == "eeg" else FMRI_NAS_PARAMS
    srpi_params = EEG_SRPI_PARAMS if modality == "eeg" else FMRI_SRPI_PARAMS
    ram_params = EEG_RAM_PARAMS if modality == "eeg" else FMRI_RAM_PARAMS
    df_ci = compute_synergy_ci(
        str(prep_root),
        atlas,
        thetas=[0.5],
        sessions=sessions,
        condition=condition,
        tr=tr,
        stimulus_onsets=onsets,
        subjects=subjects,
        ram_params=ram_params,
        pdi_params=PDI_PARAMS,
        pdi_require_explicit_params=True,
        pdi_require_strict_baseline=True,
        pdi_primary_endpoint="anchor",
        nas_params=nas_params,
        srpi_params=srpi_params,
        srpi_require_explicit_params=True,
        iim_bins=2,
        iim_lag_trs=1,
        iim_max_timepoints=80,
        iim_max_nodes=6,
        iim_max_mechanism_size=2,
        iim_max_purview_size=2,
        iim_parallel_workers=1,
        iim_enable_parallel=False,
        iim_use_shared_memory=False,
        iim_phase1_parallel_workers=1,
        iim_phase1_chunk_size=4,
        iim_phase1_shared_memory=False,
        iim_checkpoint_dir=str(out_dir / "iim_checkpoints"),
        iim_resume_checkpoint=True,
        compute_mpc=True,
        compute_ci=True,
        dataset_id=dataset_id,
        data_origin=DUMMY_DATA_ORIGIN,
        dataset_role="real_data_derived_synthetic_validation",
        provenance_label="synthetic_validation",
        modality=modality,
    )
    df_ci.to_csv(out_dir / "actual_metric_computation.csv", index=False)

    srpi_details = []
    ram_details = []
    rows = []
    for subj in subjects:
        for ses in sessions:
            ts_path = prep_root / subj / ses / condition / f"{subj}_run-1_{atlas}_ts.npy"
            ts = np.load(ts_path).T
            bundle = onsets[subj][ses][0]
            srpi = compute_SRPI(
                ts,
                tr=tr,
                self_onsets=bundle.get("self_onsets", []),
                nonself_onsets=bundle.get("nonself_onsets", []),
                return_details=True,
                **srpi_params,
            )
            ram = compute_RAM(ts, tr=tr, stimulus_onsets=bundle, return_details=True, **ram_params)
            srpi_details.append({"subject": subj, "session": ses, **srpi})
            ram_details.append({"subject": subj, "session": ses, **ram})
            rec = (
                df_ci[(df_ci["subject"].astype(str) == subj) & (df_ci["session"].astype(str) == ses)]
                .iloc[0]
                .to_dict()
            )
            checks = {
                metric: bool(np.isfinite(float(rec.get(metric, np.nan))) and float(rec.get(metric, np.nan)) > 0.0)
                for metric in ("RAM", "PDI", "NAS", "IIM", "SRPI", "CI")
            }
            srpi_components = srpi.get("components", {})
            checks.update(
                {
                    f"SRPI_{k}_positive": bool(np.isfinite(float(v)) and float(v) > 0.0)
                    for k, v in srpi_components.items()
                }
            )
            rows.append(
                {
                    "subject": subj,
                    "session": ses,
                    "metric_values": {metric: float(rec.get(metric, np.nan)) for metric in ("RAM", "PDI", "NAS", "IIM", "SRPI", "CI")},
                    "checks": checks,
                }
            )

    _write_json(out_dir / "srpi_details.json", {"rows": srpi_details})
    _write_json(out_dir / "ram_details.json", {"rows": ram_details})
    all_checks = [ok for row in rows for ok in row["checks"].values()]
    report = {
        "dataset_id": dataset_id,
        "readiness_csv": str(readiness_csv),
        "readiness_json": str(readiness_json),
        "actual_metric_csv": str(out_dir / "actual_metric_computation.csv"),
        "all_ready": bool(ready_summary.get("metrics", {}).get("CI", {}).get("ready_fraction") == 1.0),
        "all_actual_metrics_positive": bool(all(all_checks)),
        "rows": rows,
    }
    _write_json(out_dir / "actual_metric_report.json", report)
    return report


def _write_donor_report(inspections: dict[str, dict[str, Any]], manifests: list[dict[str, Any]]) -> None:
    payload = {
        "synthetic_targets": [m["dataset_id"] for m in manifests],
        "source_inspection_summary": str(INSPECTION_DIR / "source_inspection_summary.json"),
        "complete_sources_used": ["ds003171", "ds002547", "ds005620", "ds005479", "ds004295", "ds002336"],
        "excluded_incomplete_or_deferred": ["ds006623", "ds002685"],
        "donor_statistics": {
            ds: {
                "file_counts": inspections.get(ds, {}).get("file_counts"),
                "tasks": inspections.get(ds, {}).get("tasks"),
                "events": inspections.get(ds, {}).get("events", {}).get("trial_type_counts_total"),
                "nifti": {
                    "tr": inspections.get(ds, {}).get("nifti", {}).get("tr"),
                    "n_volumes": inspections.get(ds, {}).get("nifti", {}).get("n_volumes"),
                    "acf1_means": inspections.get(ds, {}).get("nifti", {}).get("acf1_means"),
                },
                "eeg": {
                    "sampling_frequency": inspections.get(ds, {}).get("eeg", {}).get("sampling_frequency"),
                    "n_channels": inspections.get(ds, {}).get("eeg", {}).get("n_channels"),
                    "duration_sec": inspections.get(ds, {}).get("eeg", {}).get("duration_sec"),
                },
            }
            for ds in ["ds003171", "ds002547", "ds005620", "ds005479", "ds004295", "ds002336"]
        },
        "limitations": [
            "The package is compact and is not an OpenNeuro mirror.",
            "Incomplete/deferred ds006623 and ds002685 were not used as generation sources.",
        ],
    }
    _write_json(REPORTS_OUT / "donor_statistics_report.json", payload)


def _link_repo_paths() -> None:
    link_specs = [
        (REPO_ROOT / "test_objects" / "datasets" / "real_derived_synth_completed", DATASETS_OUT),
        (REPO_ROOT / "test_objects" / "runs" / "real_derived_synth_completed", RUNS_OUT),
    ]
    for link, target in link_specs:
        link.parent.mkdir(parents=True, exist_ok=True)
        target.mkdir(parents=True, exist_ok=True)
        if link.is_symlink():
            if link.resolve() == target.resolve():
                continue
            link.unlink()
        if link.exists():
            continue
        link.symlink_to(target, target_is_directory=True)


def main() -> int:
    global SOURCE_ROOT
    parser = argparse.ArgumentParser(description="Generate real-data-derived synthetic CI validation datasets.")
    parser.add_argument("--seed", type=int, default=20260620)
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--source-root",
        default=None,
        help=(
            "Parent directory containing downloaded source datasets, for example "
            "/data/openneuro_sources with ds003171, ds002547, and other dataset folders inside. "
            "Defaults to IMPACT_SOURCE_ROOT, then IMPACT_SYNTH_ROOT/data/scratch."
        ),
    )
    args = parser.parse_args()
    if args.source_root:
        SOURCE_ROOT = Path(args.source_root).expanduser().resolve()

    if not SSD_ROOT.exists():
        raise FileNotFoundError(f"Synthetic output root not found: {SSD_ROOT}")
    if not args.validate_only and not SOURCE_ROOT.exists():
        raise FileNotFoundError(f"Source dataset root not found: {SOURCE_ROOT}")
    REPORTS_OUT.mkdir(parents=True, exist_ok=True)
    if args.validate_only:
        manifests = [_read_json(REPORTS_OUT / f"{dataset_id}_manifest.json") for dataset_id in TARGETS]
        validation = {manifest["dataset_id"]: _validate_dataset(manifest) for manifest in manifests}
        _write_json(
            REPORTS_OUT / "real_derived_synth_completed_summary.json",
            {
                "targets": [m["dataset_id"] for m in manifests],
                "manifests": [str(REPORTS_OUT / f"{m['dataset_id']}_manifest.json") for m in manifests],
                "source_root": str(SOURCE_ROOT),
                "validation": validation,
                "all_validated": bool(validation) and all(v.get("all_ready") and v.get("all_actual_metrics_positive") for v in validation.values()),
                "synthetic": True,
                "not_real_patient_data": True,
            },
        )
        print(f"validated existing synthetic datasets -> {REPORTS_OUT}")
        return 0

    inspections = _load_source_inspections()
    mid_events = _load_mid_events()
    self_template, other_template = _load_self_other_templates()
    source_templates = {
        "ds003171_audio_template": _load_audio_template(),
        "ds005620_eeg_template": _load_eeg_template_files(),
    }
    _write_json(
        REPORTS_OUT / "source_file_inspection_report.json",
        {
            "source_root": str(SOURCE_ROOT),
            "inspections": {
                ds: str(INSPECTION_DIR / f"{ds}_source_inspection.json")
                for ds in inspections
            },
            "templates_read": source_templates,
        },
    )

    manifests = []
    for offset, dataset_id in enumerate(TARGETS):
        manifests.append(
            _make_dataset(
                dataset_id,
                inspections,
                mid_events,
                self_template,
                other_template,
                seed=int(args.seed) + 1000 * offset,
            )
        )
    _write_donor_report(inspections, manifests)
    _link_repo_paths()

    validation = {}
    if not args.skip_validation:
        for manifest in manifests:
            validation[manifest["dataset_id"]] = _validate_dataset(manifest)
    _write_json(
        REPORTS_OUT / "real_derived_synth_completed_summary.json",
        {
            "targets": [m["dataset_id"] for m in manifests],
            "manifests": [str(REPORTS_OUT / f"{m['dataset_id']}_manifest.json") for m in manifests],
            "source_root": str(SOURCE_ROOT),
            "validation": validation,
            "all_validated": bool(validation) and all(v.get("all_ready") and v.get("all_actual_metrics_positive") for v in validation.values()),
            "synthetic": True,
            "not_real_patient_data": True,
        },
    )
    print(f"wrote synthetic datasets -> {DATASETS_OUT}")
    print(f"wrote synthetic runs -> {RUNS_OUT}")
    print(f"wrote reports -> {REPORTS_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
