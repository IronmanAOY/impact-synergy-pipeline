import os
import glob
import logging
import concurrent.futures
import subprocess
import hashlib
import gc
from collections import deque
from multiprocessing import shared_memory

import numpy as np
import pandas as pd

from impact_pipeline.hardware_backend import (
    backend_summary,
    configure_process_for_hardware,
)
from impact_pipeline.mpc_metrics import (
    compute_RAM,
    compute_PDI,
    compute_NAS,
    compute_IIM,
    compute_SRPI,
    compute_CI,
)
from impact_pipeline.provenance import (
    PROVENANCE_COLUMNS,
    REAL_DATA_ORIGIN,
    dataset_role_for_origin,
    normalize_data_origin,
    provenance_label_for_origin,
)
from impact_pipeline.utils import HypergraphSynergy

# Keep raw IIM display in native (unscaled) units.
IIM_DISPLAY_SCALE_DEFAULT = 1.0
log = logging.getLogger(__name__)

RAM_PARAM_DEFAULTS = {
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
RAM_PARAM_KEYS = tuple(RAM_PARAM_DEFAULTS.keys())

PDI_PARAM_DEFAULTS = {
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
PDI_PARAM_KEYS = tuple(PDI_PARAM_DEFAULTS.keys())

NAS_PARAM_DEFAULTS = {
    "zthr": 1.0,
    "eps": 0.2,
    "tau": None,
    "lambda_phase": 0.5,
    "alpha": 0.20,
    "beta": 0.16,
    "gamma": 0.14,
    "delta": 0.12,
    "eta": 0.16,
    "zeta": 0.12,
    "rho": 0.10,
    "bands": None,
    "band_weights": None,
    "window_len": None,
    "step_len": None,
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
NAS_PARAM_KEYS = tuple(NAS_PARAM_DEFAULTS.keys())

SRPI_PARAM_DEFAULTS = {
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
SRPI_PARAM_KEYS = tuple(SRPI_PARAM_DEFAULTS.keys())


def _parse_vm_stat_pages(vm_stat_text: str) -> dict:
    pages = {}
    page_size = 4096
    for line in vm_stat_text.splitlines():
        line = line.strip()
        if "page size of" in line and "bytes" in line:
            try:
                page_size = int(line.split("page size of", 1)[1].split("bytes", 1)[0].strip())
            except Exception:
                page_size = 4096
            continue
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        val = val.strip().rstrip(".").replace(".", "")
        try:
            pages[key.strip()] = int(val)
        except Exception:
            continue
    pages["_page_size"] = int(page_size)
    return pages


def _memory_snapshot_bytes():
    """
    Best-effort memory snapshot.
    Returns (used_bytes, total_bytes) or (None, None) when unavailable.
    """
    try:
        out = subprocess.check_output(["vm_stat"], text=True)
        p = _parse_vm_stat_pages(out)
        ps = float(p.get("_page_size", 4096))
        free = float(p.get("Pages free", 0))
        active = float(p.get("Pages active", 0))
        inactive = float(p.get("Pages inactive", 0))
        spec = float(p.get("Pages speculative", 0))
        wired = float(p.get("Pages wired down", 0))
        comp = float(p.get("Pages occupied by compressor", 0))
        # Same accounting already used in this project for rough used/free tracking.
        used_pages = active + inactive + spec + wired + comp
        total_pages = used_pages + free
        if total_pages <= 0:
            return None, None
        return int(used_pages * ps), int(total_pages * ps)
    except Exception:
        return None, None


def _iim_worker_from_path(
    ts_source,
    iim_bins,
    iim_lag_trs,
    iim_n_parts,
    iim_max_timepoints,
    iim_max_nodes,
    iim_max_mechanism_size,
    iim_max_purview_size,
    iim_checkpoint_dir,
    iim_resume_checkpoint,
    iim_checkpoint_every_cuts,
    iim_progress_log_every_cuts,
    iim_phase1_parallel_workers,
    iim_phase1_chunk_size,
    iim_phase1_shared_memory,
    hardware_target,
):
    # Keep each worker single-threaded for predictable scaling when many workers are used.
    backend = configure_process_for_hardware(hardware_target)
    if not backend.accelerator:
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    shm = None
    if isinstance(ts_source, dict) and ts_source.get("mode") == "shared":
        ts_path = str(ts_source["ts_path"])
        shm = shared_memory.SharedMemory(name=str(ts_source["shm_name"]))
        ts_time_region = np.ndarray(
            tuple(ts_source["shape"]),
            dtype=np.dtype(ts_source["dtype"]),
            buffer=shm.buf,
        )
    else:
        ts_path = str(ts_source)
        ts_time_region = np.load(ts_path)
    ts_iim = ts_time_region.T
    if iim_max_timepoints is not None and int(iim_max_timepoints) > 0:
        max_tp = int(iim_max_timepoints)
        if ts_iim.shape[1] > max_tp:
            step = int(np.ceil(ts_iim.shape[1] / max_tp))
            ts_iim = ts_iim[:, ::step]

    checkpoint_path = None
    if iim_checkpoint_dir:
        os.makedirs(iim_checkpoint_dir, exist_ok=True)
        stem = os.path.basename(ts_path).replace("_ts.npy", "")
        stem = "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in stem)[:80]
        sig = (
            f"{os.path.abspath(ts_path)}|bins={iim_bins}|lag={iim_lag_trs}|n_parts={iim_n_parts}|"
            f"max_tp={iim_max_timepoints}|max_nodes={iim_max_nodes}|"
            f"max_mech={iim_max_mechanism_size}|max_purv={iim_max_purview_size}|"
            f"ph1_workers={iim_phase1_parallel_workers}|ph1_chunk={iim_phase1_chunk_size}|"
            f"ph1_shm={int(bool(iim_phase1_shared_memory))}"
        )
        digest = hashlib.sha1(sig.encode("utf-8")).hexdigest()[:16]
        checkpoint_path = os.path.join(
            iim_checkpoint_dir,
            f"{stem}_{digest}.iim_checkpoint.json",
        )

    try:
        iim_info = compute_IIM(
            ts_iim,
            bins=iim_bins,
            lag_trs=iim_lag_trs,
            n_parts=iim_n_parts,
            max_nodes=iim_max_nodes,
            max_mechanism_size=iim_max_mechanism_size,
            max_purview_size=iim_max_purview_size,
            partition_mode="all",
            clamp=True,
            return_details=True,
            checkpoint_path=checkpoint_path,
            resume_from_checkpoint=bool(iim_resume_checkpoint),
            checkpoint_every_cuts=int(iim_checkpoint_every_cuts),
            progress_log_every_cuts=int(iim_progress_log_every_cuts),
            progress_label=os.path.basename(ts_path),
            phase1_parallel_workers=iim_phase1_parallel_workers,
            phase1_chunk_size=iim_phase1_chunk_size,
            phase1_shared_memory=bool(iim_phase1_shared_memory),
            hardware_backend=backend,
        )
    finally:
        if shm is not None:
            shm.close()
    return ts_path, iim_info


def _resolve_pdi_kwargs(pdi_params, require_explicit):
    if pdi_params is None:
        if require_explicit:
            raise ValueError(
                "Explicit PDI hyperparameters are required for strict validation runs, "
                "but pdi_params was None."
            )
        params = dict(PDI_PARAM_DEFAULTS)
    else:
        params = {}
        for k in PDI_PARAM_KEYS:
            if k in pdi_params:
                params[k] = pdi_params[k]
        missing = [k for k in PDI_PARAM_KEYS if k not in params]
        if missing and require_explicit:
            raise ValueError(
                "Explicit PDI hyperparameters are required for strict validation runs. "
                f"Missing keys: {missing}"
            )
        for k in missing:
            params[k] = PDI_PARAM_DEFAULTS[k]
    # Ensure normalization is controlled at endpoint level.
    params["normalize"] = False
    return params


def _resolve_ram_kwargs(ram_params):
    if ram_params is None:
        return dict(RAM_PARAM_DEFAULTS)
    params = {}
    for k in RAM_PARAM_KEYS:
        if k in ram_params:
            params[k] = ram_params[k]
    for k in RAM_PARAM_KEYS:
        if k not in params:
            params[k] = RAM_PARAM_DEFAULTS[k]
    return params


def _resolve_nas_kwargs(nas_params):
    if nas_params is None:
        raise ValueError(
            "NAS hyperparameters must be provided explicitly; fallback defaults are disabled."
        )
    params = {}
    for k in NAS_PARAM_KEYS:
        if k not in nas_params:
            raise ValueError(
                "NAS hyperparameters must be fully explicit; "
                f"missing key: '{k}'."
            )
        params[k] = nas_params[k]
    required_non_null = ("tau", "bands", "band_weights", "window_len", "step_len")
    none_keys = [k for k in required_non_null if params.get(k) is None]
    if none_keys:
        raise ValueError(
            "NAS fallback behavior is disabled. "
            f"The following keys cannot be None: {none_keys}"
        )
    # NAS baseline subtraction is handled explicitly when enabled by caller.
    params["baseline_ts"] = None
    return params


def _resolve_srpi_kwargs(srpi_params, require_explicit):
    if srpi_params is None:
        if require_explicit:
            raise ValueError(
                "SRPI hyperparameters must be provided explicitly; "
                "fallback defaults are disabled."
            )
        params = dict(SRPI_PARAM_DEFAULTS)
    else:
        params = {}
        for k in SRPI_PARAM_KEYS:
            if k in srpi_params:
                params[k] = srpi_params[k]
        missing = [k for k in SRPI_PARAM_KEYS if k not in params]
        if missing and require_explicit:
            raise ValueError(
                "SRPI hyperparameters must be fully explicit; "
                f"missing keys: {missing}."
            )
        for k in missing:
            params[k] = SRPI_PARAM_DEFAULTS[k]

    mode = str(params.get("modality", "")).strip().lower()
    if mode not in {"fmri", "eeg"}:
        raise ValueError("SRPI modality must be one of {'fmri','eeg'}.")
    params["modality"] = mode
    return params


def discover_ci_subjects(data_dir, subjects=None):
    subj_filter = None
    subj_order_requested = None
    if subjects is not None:
        subj_order_requested = []
        for s in subjects:
            ss = str(s).replace("sub-", "").strip()
            if ss and ss not in subj_order_requested:
                subj_order_requested.append(ss)
        subj_filter = set(subj_order_requested)

    subj_dirs = [
        subj
        for subj in sorted(os.listdir(data_dir))
        if (not subj.startswith(".")) and os.path.isdir(os.path.join(data_dir, subj))
    ]
    if subj_filter is not None:
        available = set(subj_dirs)
        missing = [s for s in subj_order_requested if s not in available]
        if missing:
            log.warning("Requested subjects not found under %s: %s", data_dir, ",".join(missing))
        return [s for s in subj_order_requested if s in available]
    return subj_dirs


def build_ci_run_specs(
    data_dir,
    atlas,
    sessions,
    condition,
    stimulus_onsets=None,
    subjects=None,
):
    run_specs = []
    subj_iter = discover_ci_subjects(data_dir, subjects=subjects)
    for subj in subj_iter:
        for ses in sessions:
            log.debug("compute_synergy_ci: subject=%s session=%s", subj, ses)
            session_dir = os.path.join(data_dir, subj, ses, condition)
            pattern = os.path.join(session_dir, f"{subj}_run-*_{atlas}_ts.npy")
            all_cands = sorted(glob.glob(pattern))

            subj_onsets, run_id = (None, None)
            if isinstance(stimulus_onsets, dict):
                tup = stimulus_onsets.get(subj, {}).get(ses)
                if tup:
                    subj_onsets, run_id = tup

            session_ts_paths = []
            if run_id is not None:
                run_clean = str(int(run_id))
                run_padded = str(run_id)
                exact_candidates = []
                for run_token in (run_clean, run_padded):
                    exact_candidates.extend(
                        sorted(
                            glob.glob(
                                os.path.join(
                                    session_dir,
                                    f"{subj}_run-{run_token}_{atlas}_ts.npy",
                                )
                            )
                        )
                    )
                if exact_candidates:
                    session_ts_paths = [exact_candidates[0]]
                else:
                    run_tags = (f"run-{run_clean}_", f"run-{run_padded}_")
                    filt = [
                        c for c in all_cands
                        if any(tag in os.path.basename(c) for tag in run_tags)
                    ]
                    if filt:
                        session_ts_paths = [filt[0]]
            if not session_ts_paths:
                if not all_cands:
                    raise FileNotFoundError(
                        f"No time-series for {subj}/{ses} (searched {pattern})"
                    )
                session_ts_paths = all_cands

            for ts_path in session_ts_paths:
                run_specs.append(
                    {
                        "subject": subj,
                        "session": ses,
                        "ts_path": ts_path,
                        "stimulus_onsets": subj_onsets,
                    }
                )
    return run_specs


def compute_synergy_ci(
    data_dir,
    atlas,
    thetas,
    sessions=('awake', 'deep', 'recovery'),
    condition='audio',
    tr=None,
    stimulus_onsets=None,
    ci_human_refs=None,
    ci_weights=None,
    ci_eps=1e-12,
    iim_display_scale=IIM_DISPLAY_SCALE_DEFAULT,
    iim_bins=3,
    iim_lag_trs=1,
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
    compute_mpc=True,
    mpc_metrics=None,
    compute_ci=True,
    subjects=None,
    ram_params=None,
    pdi_params=None,
    pdi_require_explicit_params=False,
    pdi_require_strict_baseline=False,
    pdi_primary_endpoint="anchor",
    nas_params=None,
    srpi_params=None,
    srpi_require_explicit_params=True,
    iim_precomputed_by_path=None,
    dataset_id=None,
    data_origin=REAL_DATA_ORIGIN,
    dataset_role=None,
    provenance_label=None,
    modality=None,
    hardware_target="cpu",
):
    hardware_backend = configure_process_for_hardware(hardware_target)
    log.info("MPC hardware backend: %s", backend_summary(hardware_backend))

    valid_mpc_metrics = ("RAM", "PDI", "NAS", "IIM", "SRPI")
    if not compute_mpc:
        selected_metrics = tuple()
    elif mpc_metrics is None:
        selected_metrics = valid_mpc_metrics
    else:
        selected_metrics = tuple(dict.fromkeys(mpc_metrics))
        unknown = sorted(set(selected_metrics) - set(valid_mpc_metrics))
        if unknown:
            raise ValueError(
                f"Unknown MPC metric(s): {unknown}. "
                f"Expected subset of {valid_mpc_metrics}."
            )
    do_ram = "RAM" in selected_metrics
    do_pdi = "PDI" in selected_metrics
    do_nas = "NAS" in selected_metrics
    do_iim = "IIM" in selected_metrics
    do_srpi = "SRPI" in selected_metrics
    data_origin_norm = normalize_data_origin(data_origin)
    dataset_role_txt = (
        dataset_role_for_origin(data_origin_norm)
        if dataset_role is None
        else str(dataset_role)
    )
    provenance_label_txt = (
        provenance_label_for_origin(data_origin_norm)
        if provenance_label is None
        else str(provenance_label)
    )
    result_metadata = {
        "dataset_id": "" if dataset_id is None else str(dataset_id),
        "data_origin": str(data_origin_norm),
        "dataset_role": str(dataset_role_txt),
        "provenance_label": str(provenance_label_txt),
        "hardware_target": str(hardware_backend.requested),
        "hardware_backend": str(hardware_backend.target),
        "hardware_runtime": str(hardware_backend.runtime),
    }

    pdi_endpoint = str(pdi_primary_endpoint).strip().lower()
    if pdi_endpoint not in {"anchor", "task"}:
        raise ValueError("pdi_primary_endpoint must be one of {'anchor','task'}")
    pdi_kwargs = _resolve_pdi_kwargs(
        pdi_params=pdi_params,
        require_explicit=bool(pdi_require_explicit_params and do_pdi),
    )
    ram_kwargs = _resolve_ram_kwargs(ram_params) if do_ram else {}
    pdi_kwargs_raw = dict(pdi_kwargs)
    pdi_kwargs_raw["clip_negative"] = False
    nas_kwargs = (
        _resolve_nas_kwargs(
            nas_params=nas_params,
        )
        if do_nas
        else {}
    )
    srpi_kwargs = (
        _resolve_srpi_kwargs(
            srpi_params=srpi_params,
            require_explicit=bool(srpi_require_explicit_params and do_srpi),
        )
        if do_srpi
        else {}
    )
    if do_ram or do_nas or do_srpi:
        if tr is None or (not np.isfinite(tr)) or float(tr) <= 0:
            raise ValueError(
                "Explicit positive sample interval 'tr' is required for RAM/NAS/SRPI; "
                "fallback TR resolution is disabled."
            )
        tr = float(tr)

    def _select_pdi_state_rest_runs(subj, session_name):
        subj_root = os.path.join(data_dir, subj)
        return sorted(
            glob.glob(
                os.path.join(
                    subj_root, session_name, 'rest', f"{subj}_run-*_{atlas}_ts.npy"
                )
            )
        )

    def _select_pdi_deep_rest_runs(subj):
        subj_root = os.path.join(data_dir, subj)
        return sorted(
            glob.glob(
                os.path.join(subj_root, "deep", "rest", f"{subj}_run-*_{atlas}_ts.npy")
            )
        )

    def _select_pdi_legacy_baseline_runs(subj, session_name):
        """
        Baseline policy for permissive runs.
        Priority:
          1) same-session rest
          2) any subject-level rest
          3) surrogate baseline inside compute_PDI (None)
        """
        same_session_rest = _select_pdi_state_rest_runs(subj, session_name)
        if same_session_rest:
            return same_session_rest
        subj_root = os.path.join(data_dir, subj)
        any_rest = sorted(
            glob.glob(
                os.path.join(subj_root, '*', 'rest', f"{subj}_run-*_{atlas}_ts.npy")
            )
        )
        if any_rest:
            return any_rest
        return None

    def _load_pdi_baseline_ts(paths, n_regions_expected):
        ts_list = []
        keep_paths = []
        for fn in paths:
            try:
                arr = np.load(fn).T
            except Exception:
                continue
            if arr.ndim != 2:
                continue
            if int(arr.shape[0]) != int(n_regions_expected):
                continue
            ts_list.append(arr)
            keep_paths.append(str(fn))
        return ts_list, keep_paths

    run_specs = build_ci_run_specs(
        data_dir,
        atlas,
        sessions,
        condition,
        stimulus_onsets=stimulus_onsets,
        subjects=subjects,
    )

    records = []
    if not run_specs:
        cols_empty = list(PROVENANCE_COLUMNS) + [
            'hardware_target', 'hardware_backend', 'hardware_runtime',
            'subject', 'session', 'theta', 'S',
        ]
        if compute_mpc:
            cols_empty.extend(list(selected_metrics))
            if do_pdi:
                cols_empty.extend([
                    'PDI_anchor', 'PDI_task',
                    'PDI_anchor_defined', 'PDI_task_defined',
                    'PDI_anchor_reason', 'PDI_task_reason',
                    'PDI_primary_endpoint', 'PDI_primary_source',
                    'PDI_baseline_policy',
                    'PDI_anchor_baseline_n_runs', 'PDI_task_baseline_n_runs',
                    'PDI_anchor_baseline_paths', 'PDI_task_baseline_paths',
                ])
            if do_iim:
                cols_empty.extend([
                    'IIM_raw', 'IIM_raw_scaled',
                    'IIM_defined', 'IIM_undefined_reason',
                ])
            if compute_ci and set(valid_mpc_metrics).issubset(set(selected_metrics)):
                cols_empty.extend([
                    'CI',
                    'RAM_norm', 'PDI_norm', 'NAS_norm', 'IIM_norm', 'SRPI_norm',
                ])
        return pd.DataFrame(columns=cols_empty)

    iim_by_path = {}
    if compute_mpc and do_iim and iim_precomputed_by_path is not None:
        iim_by_path = {
            str(k): v
            for k, v in dict(iim_precomputed_by_path).items()
        }
    elif compute_mpc and do_iim:
        by_subject_paths = {}
        for spec in run_specs:
            subj = str(spec["subject"])
            ts_path = str(spec["ts_path"])
            paths = by_subject_paths.setdefault(subj, [])
            if ts_path not in paths:
                paths.append(ts_path)
        subject_order = list(by_subject_paths.keys())
        unique_paths = [p for subj in subject_order for p in by_subject_paths[subj]]
        ts_source_by_path = {p: p for p in unique_paths}
        shared_handles = []

        if iim_checkpoint_dir:
            os.makedirs(iim_checkpoint_dir, exist_ok=True)
            log.info(
                "IIM checkpointing: dir=%s resume=%s checkpoint_every_cuts=%d progress_every_cuts=%d",
                iim_checkpoint_dir,
                bool(iim_resume_checkpoint),
                int(iim_checkpoint_every_cuts),
                int(iim_progress_log_every_cuts),
            )
        log.info(
            "IIM intra-task config: phase_workers=%s chunk_size=%d phase_shared_memory=%s",
            "auto/off" if iim_phase1_parallel_workers is None else str(int(iim_phase1_parallel_workers)),
            int(iim_phase1_chunk_size),
            bool(iim_phase1_shared_memory),
        )

        if bool(iim_use_shared_memory) and len(unique_paths) > 0:
            shared_total_b = 0
            try:
                for ts_path in unique_paths:
                    arr = np.load(ts_path, mmap_mode="r")
                    arr_c = np.ascontiguousarray(arr)
                    shm = shared_memory.SharedMemory(create=True, size=int(arr_c.nbytes))
                    shm_arr = np.ndarray(arr_c.shape, dtype=arr_c.dtype, buffer=shm.buf)
                    shm_arr[...] = arr_c
                    shared_handles.append(shm)
                    ts_source_by_path[ts_path] = {
                        "mode": "shared",
                        "shm_name": str(shm.name),
                        "shape": tuple(int(x) for x in arr_c.shape),
                        "dtype": str(arr_c.dtype),
                        "ts_path": ts_path,
                    }
                    shared_total_b += int(arr_c.nbytes)
                    del shm_arr
                    del arr_c
                    del arr
                gc.collect()
                log.info(
                    "IIM shared-memory input staging: arrays=%d total=%.3fGB",
                    len(unique_paths),
                    shared_total_b / float(1024 ** 3),
                )
            except Exception as exc:
                log.warning(
                    "IIM shared-memory staging failed (%s). Falling back to direct file loads.",
                    exc,
                )
                for shm in shared_handles:
                    try:
                        shm.close()
                    except Exception:
                        pass
                    try:
                        shm.unlink()
                    except Exception:
                        pass
                shared_handles = []
                ts_source_by_path = {p: p for p in unique_paths}

        cpu_max = max(1, int(os.cpu_count() or 1))
        if not iim_enable_parallel:
            workers = 1
        elif iim_parallel_workers is not None:
            workers = max(1, int(iim_parallel_workers))
        else:
            oversub = max(1.0, float(iim_cpu_oversub_factor))
            soft_cpu_cap = max(1, int(np.ceil(cpu_max * oversub)))
            workers = min(soft_cpu_cap, len(unique_paths))
            used_b, total_b = _memory_snapshot_bytes()
            target_ratio = float(iim_memory_target_ratio)
            target_ratio = min(max(target_ratio, 0.10), 0.99)
            if used_b is not None and total_b is not None and total_b > 0:
                budget_b = max(0, int(total_b * target_ratio) - int(used_b))
                est_b = max(int(float(iim_worker_mem_gb_estimate) * (1024 ** 3)), 256 * 1024 * 1024)
                by_mem = max(1, int(budget_b // est_b))
                workers = min(workers, by_mem)
                used_ratio = float(used_b / total_b)
                log.info(
                    "IIM memory planner: used=%.1f%% target=%.1f%% est_worker_mem=%.2fGB "
                    "budget=%.2fGB oversub=%.2f cpu_cap=%d -> workers=%d",
                    used_ratio * 100.0,
                    target_ratio * 100.0,
                    est_b / float(1024 ** 3),
                    budget_b / float(1024 ** 3),
                    oversub,
                    soft_cpu_cap,
                    workers,
                )
        workers = max(1, min(workers, len(unique_paths)))
        log.info(
            "IIM execution: tasks=%d workers=%d (cpu_max=%d, target_mem_ratio=%.2f)",
            len(unique_paths),
            workers,
            cpu_max,
            float(iim_memory_target_ratio),
        )
        if hardware_backend.accelerator and iim_parallel_workers is None:
            workers_limited = max(1, min(int(workers), max(1, int(hardware_backend.device_count))))
            if workers_limited != workers:
                log.info(
                    "IIM accelerator scheduling: limiting auto workers %d -> %d for %d visible device(s)",
                    int(workers),
                    int(workers_limited),
                    int(hardware_backend.device_count),
                )
                workers = workers_limited
        if subject_order:
            init_slots = int(workers)
            init_alloc = []
            for subj in subject_order:
                if init_slots <= 0:
                    break
                n_take = min(len(by_subject_paths.get(subj, [])), init_slots)
                if n_take > 0:
                    init_alloc.append(f"{subj}:{n_take}")
                    init_slots -= n_take
            log.info(
                "IIM subject-priority scheduling: order=%s initial_allocation=%s",
                ",".join(subject_order),
                ",".join(init_alloc) if init_alloc else "none",
            )

        try:
            if workers == 1:
                for i, ts_path in enumerate(unique_paths, start=1):
                    _, iim_info = _iim_worker_from_path(
                        ts_source_by_path[ts_path],
                        iim_bins,
                        iim_lag_trs,
                        iim_n_parts,
                        iim_max_timepoints,
                        iim_max_nodes,
                        iim_max_mechanism_size,
                        iim_max_purview_size,
                        iim_checkpoint_dir,
                        iim_resume_checkpoint,
                        iim_checkpoint_every_cuts,
                        iim_progress_log_every_cuts,
                        iim_phase1_parallel_workers,
                        iim_phase1_chunk_size,
                        iim_phase1_shared_memory,
                        hardware_backend.requested,
                    )
                    iim_by_path[ts_path] = iim_info
                    if (i == len(unique_paths)) or (i % max(1, len(unique_paths) // 20) == 0):
                        log.info("IIM progress: %d/%d completed", i, len(unique_paths))
            else:
                pending_by_subject = {
                    subj: deque(by_subject_paths.get(subj, []))
                    for subj in subject_order
                }
                with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
                    futures = {}
                    done = 0
                    total = len(unique_paths)
                    while done < total:
                        free_slots = int(workers) - len(futures)
                        if free_slots > 0:
                            for subj in subject_order:
                                q = pending_by_subject[subj]
                                while q and free_slots > 0:
                                    ts_path = q.popleft()
                                    fut = ex.submit(
                                        _iim_worker_from_path,
                                        ts_source_by_path[ts_path],
                                        iim_bins,
                                        iim_lag_trs,
                                        iim_n_parts,
                                        iim_max_timepoints,
                                        iim_max_nodes,
                                        iim_max_mechanism_size,
                                        iim_max_purview_size,
                                        iim_checkpoint_dir,
                                        iim_resume_checkpoint,
                                        iim_checkpoint_every_cuts,
                                        iim_progress_log_every_cuts,
                                        iim_phase1_parallel_workers,
                                        iim_phase1_chunk_size,
                                        iim_phase1_shared_memory,
                                        hardware_backend.requested,
                                    )
                                    futures[fut] = ts_path
                                    free_slots -= 1
                                if free_slots <= 0:
                                    break

                        if not futures:
                            raise RuntimeError(
                                "IIM scheduler stalled with no active futures before completing all tasks."
                            )

                        done_now, _ = concurrent.futures.wait(
                            list(futures.keys()),
                            return_when=concurrent.futures.FIRST_COMPLETED,
                        )
                        for fut in done_now:
                            ts_path = futures.pop(fut)
                            try:
                                _, iim_info = fut.result()
                            except Exception as exc:
                                raise RuntimeError(f"IIM worker failed for {ts_path}: {exc}") from exc
                            iim_by_path[ts_path] = iim_info
                            done += 1
                            if (done == total) or (done % max(1, total // 20) == 0):
                                log.info("IIM progress: %d/%d completed", done, total)
        finally:
            for shm in shared_handles:
                try:
                    shm.close()
                except Exception:
                    pass
                try:
                    shm.unlink()
                except Exception:
                    pass
            if shared_handles:
                log.info("IIM shared-memory staging: cleaned up %d segments", len(shared_handles))

    for spec in run_specs:
        subj = spec["subject"]
        ses = spec["session"]
        ts_path = spec["ts_path"]
        subj_onsets = spec["stimulus_onsets"]

        log.debug("compute_synergy_ci: using TS %s", ts_path)
        ts_time_region = np.load(ts_path)
        ts_region_time = ts_time_region.T

        if compute_mpc:
            ram = (
                compute_RAM(
                    ts_region_time,
                    tr=tr,
                    stimulus_onsets=subj_onsets,
                    **ram_kwargs,
                    hardware_backend=hardware_backend,
                )
                if do_ram
                else np.nan
            )
            if do_pdi:
                n_regions_obs = int(ts_region_time.shape[0])
                deep_rest_cands = _select_pdi_deep_rest_runs(subj)
                state_rest_cands = _select_pdi_state_rest_runs(subj, ses)
                deep_rest_ts, deep_rest_paths = _load_pdi_baseline_ts(
                    deep_rest_cands,
                    n_regions_expected=n_regions_obs,
                )
                state_rest_ts, state_rest_paths = _load_pdi_baseline_ts(
                    state_rest_cands,
                    n_regions_expected=n_regions_obs,
                )

                pdi_anchor_raw = np.nan
                pdi_task_raw = np.nan
                pdi_anchor_reason = "missing_deep_rest_baseline"
                pdi_task_reason = "missing_state_rest_baseline"
                pdi_primary_source = "undefined"
                pdi_baseline_policy = (
                    "strict_dual_baseline"
                    if bool(pdi_require_strict_baseline)
                    else "dual_baseline_with_legacy_fallback"
                )

                if deep_rest_ts:
                    pdi_anchor_raw = compute_PDI(
                        ts_region_time,
                        baseline_ts=deep_rest_ts,
                        hardware_backend=hardware_backend,
                        **pdi_kwargs_raw,
                    )
                    pdi_anchor_reason = "ok"
                if state_rest_ts:
                    pdi_task_raw = compute_PDI(
                        ts_region_time,
                        baseline_ts=state_rest_ts,
                        hardware_backend=hardware_backend,
                        **pdi_kwargs_raw,
                    )
                    pdi_task_reason = "ok"

                if pdi_endpoint == "anchor":
                    pdi_primary_raw = pdi_anchor_raw
                    pdi_primary_source = "anchor"
                else:
                    pdi_primary_raw = pdi_task_raw
                    pdi_primary_source = "task"

                if np.isfinite(pdi_primary_raw):
                    pdi0 = max(float(pdi_primary_raw), 0.0)
                else:
                    pdi0 = np.nan

                if (not np.isfinite(pdi0)) and (not bool(pdi_require_strict_baseline)):
                    legacy_cands = _select_pdi_legacy_baseline_runs(subj, ses)
                    if legacy_cands is None:
                        pdi0 = compute_PDI(
                            ts_region_time,
                            baseline_ts=None,
                            hardware_backend=hardware_backend,
                            **pdi_kwargs,
                        )
                        pdi_primary_source = "legacy_surrogate"
                    else:
                        legacy_ts, legacy_paths = _load_pdi_baseline_ts(
                            legacy_cands,
                            n_regions_expected=n_regions_obs,
                        )
                        if legacy_ts:
                            pdi0 = compute_PDI(
                                ts_region_time,
                                baseline_ts=legacy_ts,
                                hardware_backend=hardware_backend,
                                **pdi_kwargs,
                            )
                            pdi_primary_source = "legacy_rest_pool"
                            if not state_rest_paths:
                                state_rest_paths = legacy_paths
                        else:
                            pdi0 = np.nan
            else:
                pdi0 = np.nan
                pdi_anchor_raw = np.nan
                pdi_task_raw = np.nan
                pdi_anchor_reason = "not_computed"
                pdi_task_reason = "not_computed"
                pdi_primary_source = "not_computed"
                pdi_baseline_policy = "not_computed"
                deep_rest_paths = []
                state_rest_paths = []
            nas = (
                compute_NAS(
                    ts_region_time,
                    tr=tr,
                    hardware_backend=hardware_backend,
                    **nas_kwargs,
                )
                if do_nas
                else np.nan
            )
            if do_iim:
                iim_info = iim_by_path.get(ts_path)
                if iim_info is None:
                    raise RuntimeError(f"Missing precomputed IIM result for {ts_path}")
                iim_defined = bool(iim_info.get("defined", False))
                iim_undefined_reason = iim_info.get("undefined_reason")
                iim = float(iim_info["canonical"]) if iim_defined else np.nan
                iim_raw = float(iim_info["raw"]) if iim_defined else np.nan
                iim_raw_scaled = (
                    iim_raw * float(iim_display_scale) if np.isfinite(iim_raw) else np.nan
                )
            else:
                iim_defined = False
                iim_undefined_reason = "not_computed"
                iim = np.nan
                iim_raw = np.nan
                iim_raw_scaled = np.nan
            if do_srpi:
                srpi_self_onsets = []
                srpi_nonself_onsets = []
                if isinstance(subj_onsets, dict):
                    srpi_self_onsets = subj_onsets.get("self_onsets", [])
                    srpi_nonself_onsets = subj_onsets.get("nonself_onsets", [])
                srpi = compute_SRPI(
                    ts_region_time,
                    tr=tr,
                    self_onsets=srpi_self_onsets,
                    nonself_onsets=srpi_nonself_onsets,
                    hardware_backend=hardware_backend,
                    **srpi_kwargs,
                )
            else:
                srpi = np.nan
        else:
            ram = np.nan
            pdi0 = np.nan
            pdi_anchor_raw = np.nan
            pdi_task_raw = np.nan
            pdi_anchor_reason = "not_computed"
            pdi_task_reason = "not_computed"
            pdi_primary_source = "not_computed"
            pdi_baseline_policy = "not_computed"
            deep_rest_paths = []
            state_rest_paths = []
            nas = np.nan
            iim = np.nan
            iim_raw = np.nan
            iim_raw_scaled = np.nan
            iim_defined = False
            iim_undefined_reason = "not_computed"
            srpi = np.nan

        for theta in thetas:
            S = HypergraphSynergy.compute(ts_time_region, theta)
            rec = dict(result_metadata)
            rec.update({
                'subject': subj,
                'session': ses,
                'theta': theta,
                'S': S,
            })
            if compute_mpc:
                if do_ram:
                    rec['RAM'] = ram
                if do_pdi:
                    rec.update({
                        'PDI': pdi0,
                        'PDI_anchor': pdi_anchor_raw,
                        'PDI_task': pdi_task_raw,
                        'PDI_anchor_defined': bool(np.isfinite(pdi_anchor_raw)),
                        'PDI_task_defined': bool(np.isfinite(pdi_task_raw)),
                        'PDI_anchor_reason': str(pdi_anchor_reason),
                        'PDI_task_reason': str(pdi_task_reason),
                        'PDI_primary_endpoint': str(pdi_endpoint),
                        'PDI_primary_source': str(pdi_primary_source),
                        'PDI_baseline_policy': str(pdi_baseline_policy),
                        'PDI_anchor_baseline_n_runs': int(len(deep_rest_paths)),
                        'PDI_task_baseline_n_runs': int(len(state_rest_paths)),
                        'PDI_anchor_baseline_paths': ";".join(deep_rest_paths),
                        'PDI_task_baseline_paths': ";".join(state_rest_paths),
                    })
                if do_nas:
                    rec['NAS'] = nas
                if do_iim:
                    rec.update({
                        'IIM': iim,
                        'IIM_raw': iim_raw,
                        'IIM_raw_scaled': iim_raw_scaled,
                        'IIM_defined': iim_defined,
                        'IIM_undefined_reason': iim_undefined_reason,
                    })
                if do_srpi:
                    rec['SRPI'] = srpi
            records.append(rec)

    hardware_cols = ['hardware_target', 'hardware_backend', 'hardware_runtime']
    cols_empty = list(PROVENANCE_COLUMNS) + hardware_cols + ['subject', 'session', 'theta', 'S']
    if compute_mpc:
        cols_empty = list(PROVENANCE_COLUMNS) + hardware_cols + ['subject', 'session', 'theta', 'S']
        cols_empty.extend(list(selected_metrics))
        if do_pdi:
            cols_empty.extend([
                'PDI_anchor', 'PDI_task',
                'PDI_anchor_defined', 'PDI_task_defined',
                'PDI_anchor_reason', 'PDI_task_reason',
                'PDI_primary_endpoint', 'PDI_primary_source',
                'PDI_baseline_policy',
                'PDI_anchor_baseline_n_runs', 'PDI_task_baseline_n_runs',
                'PDI_anchor_baseline_paths', 'PDI_task_baseline_paths',
            ])
        if do_iim:
            cols_empty.extend([
                'IIM_raw', 'IIM_raw_scaled',
                'IIM_defined', 'IIM_undefined_reason',
            ])
        if compute_ci and set(valid_mpc_metrics).issubset(set(selected_metrics)):
            cols_empty.extend([
                'CI',
                'RAM_norm', 'PDI_norm', 'NAS_norm', 'IIM_norm', 'SRPI_norm',
            ])
    if not records:
        return pd.DataFrame(columns=cols_empty)

    df = pd.DataFrame.from_records(records)
    if not compute_mpc:
        keep_cols = [
            c
            for c in (*PROVENANCE_COLUMNS, *hardware_cols, 'subject', 'session', 'theta', 'S')
            if c in df.columns
        ]
        return df[keep_cols]

    ci_required = set(valid_mpc_metrics)
    ci_computable = compute_ci and ci_required.issubset(set(df.columns))
    if not ci_computable:
        ordered_cols = [c for c in PROVENANCE_COLUMNS if c in df.columns]
        ordered_cols.extend([c for c in hardware_cols if c in df.columns])
        ordered_cols.extend(['subject', 'session', 'theta', 'S'])
        ordered_cols.extend([m for m in valid_mpc_metrics if m in df.columns])
        for pdi_extra in (
            'PDI_anchor', 'PDI_task',
            'PDI_anchor_defined', 'PDI_task_defined',
            'PDI_anchor_reason', 'PDI_task_reason',
            'PDI_primary_endpoint', 'PDI_primary_source',
            'PDI_baseline_policy',
            'PDI_anchor_baseline_n_runs', 'PDI_task_baseline_n_runs',
            'PDI_anchor_baseline_paths', 'PDI_task_baseline_paths',
        ):
            if pdi_extra in df.columns:
                ordered_cols.append(pdi_extra)
        for extra in ('IIM_raw', 'IIM_raw_scaled', 'IIM_defined', 'IIM_undefined_reason'):
            if extra in df.columns:
                ordered_cols.append(extra)
        return df[ordered_cols]

    # Internal CI integration term:
    # absorb former standalone coupling-broadcast score into NAS via
    #   NAS_tilde = NAS * C_CB(theta), with C_CB(theta)=max(S,0) here.
    # This keeps CI as a five-factor form while embedding the same logic.
    nas_ci_col = "__NAS_CI_INTERNAL__"
    s_nonneg = pd.to_numeric(df['S'], errors='coerce').clip(lower=0.0)
    nas_num = pd.to_numeric(df['NAS'], errors='coerce')
    df[nas_ci_col] = nas_num * s_nonneg

    # If no external human reference is provided, estimate from awake runs.
    if ci_human_refs is None:
        ref_source = df[df['session'] == 'awake']
        if ref_source.empty:
            ref_source = df
        ci_human_refs = {}
        for k in ('RAM', 'PDI', 'NAS', 'IIM', 'SRPI'):
            src_col = nas_ci_col if k == 'NAS' else k
            m = float(ref_source[src_col].mean(skipna=True))
            if not np.isfinite(m) or m <= 0:
                m = 1e-12
            ci_human_refs[k] = m

    ci_vals = []
    ram_norm = []
    pdi_norm = []
    nas_norm = []
    iim_norm = []
    srpi_norm = []

    for _, row in df.iterrows():
        ci_details = compute_CI(
            row['RAM'],
            row['PDI'],
            row[nas_ci_col],
            row['IIM'],
            row['SRPI'],
            references=ci_human_refs,
            weights=ci_weights,
            defined={
                'RAM': np.isfinite(row['RAM']),
                'PDI': np.isfinite(row['PDI']),
                'NAS': np.isfinite(row[nas_ci_col]),
                'IIM': bool(row.get('IIM_defined', np.isfinite(row['IIM']))),
                'SRPI': np.isfinite(row['SRPI']),
            },
            eps=ci_eps,
            return_details=True,
        )
        ci_vals.append(ci_details['value'])
        nrm = ci_details['normalized_components']
        ram_norm.append(nrm['RAM'])
        pdi_norm.append(nrm['PDI'])
        nas_norm.append(nrm['NAS'])
        iim_norm.append(nrm['IIM'])
        srpi_norm.append(nrm['SRPI'])

    df['CI'] = ci_vals
    df['RAM_norm'] = ram_norm
    df['PDI_norm'] = pdi_norm
    df['NAS_norm'] = nas_norm
    df['IIM_norm'] = iim_norm
    df['SRPI_norm'] = srpi_norm
    df = df.drop(columns=[nas_ci_col], errors='ignore')
    ordered_cols = [
        *[c for c in PROVENANCE_COLUMNS if c in df.columns],
        *[c for c in hardware_cols if c in df.columns],
        'subject', 'session', 'theta', 'S', 'CI',
        'RAM', 'PDI', 'NAS', 'IIM', 'SRPI',
        'PDI_anchor', 'PDI_task',
        'PDI_anchor_defined', 'PDI_task_defined',
        'PDI_anchor_reason', 'PDI_task_reason',
        'PDI_primary_endpoint', 'PDI_primary_source',
        'PDI_baseline_policy',
        'PDI_anchor_baseline_n_runs', 'PDI_task_baseline_n_runs',
        'PDI_anchor_baseline_paths', 'PDI_task_baseline_paths',
        'IIM_raw', 'IIM_raw_scaled',
        'IIM_defined', 'IIM_undefined_reason',
        'RAM_norm', 'PDI_norm', 'NAS_norm', 'IIM_norm', 'SRPI_norm',
    ]
    ordered_cols = [c for c in ordered_cols if c in df.columns]
    return df[ordered_cols]
