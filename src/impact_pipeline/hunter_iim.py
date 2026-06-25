from __future__ import annotations

import concurrent.futures
import dataclasses
import hashlib
import json
import math
import os
import shlex
import shutil
import tempfile
import time
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np

from impact_pipeline.execution_profiles import ExecutionProfile
from impact_pipeline.hardware_backend import configure_process_for_hardware
from impact_pipeline.mpc_metrics import (
    _IIMDiskKernelCache,
    _iim_build_cut_tpm,
    _iim_build_phase1_chunks_adaptive,
    _iim_cut_to_key,
    _iim_phase1_chunk_contribution,
    _iim_phase_worker_init_static,
    _iim_phase_worker_run_chunk_for_tpm,
    prepare_iim_problem,
)
from impact_pipeline.synergy_ci import build_ci_run_specs


def _json_dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _json_load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sanitize_token(text: str) -> str:
    raw = str(text)
    return "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in raw)


def _split_evenly(n_items: int, n_shards: int):
    total = int(max(0, n_items))
    if total == 0:
        return []
    shards = int(max(1, n_shards))
    shards = min(shards, max(1, total))
    out = []
    start = 0
    for shard_idx in range(shards):
        width = total // shards
        if shard_idx < (total % shards):
            width += 1
        stop = start + width
        out.append((start, stop))
        start = stop
    return out


def _mk_readonly_array_spec(label, arr, use_shared_memory, tmp_dir, owner_shms, owner_files):
    arr_c = np.ascontiguousarray(arr)
    if bool(use_shared_memory):
        try:
            shm = shared_memory.SharedMemory(create=True, size=int(arr_c.nbytes))
            shm_arr = np.ndarray(arr_c.shape, dtype=arr_c.dtype, buffer=shm.buf)
            shm_arr[...] = arr_c
            owner_shms.append(shm)
            return {
                "mode": "shared_memory",
                "name": str(shm.name),
                "shape": list(arr_c.shape),
                "dtype": str(arr_c.dtype),
            }
        except Exception:
            pass
    path = Path(tmp_dir) / f"{label}.npy"
    np.save(path, arr_c, allow_pickle=False)
    owner_files.append(path)
    return {"mode": "memmap", "path": str(path)}


def _cleanup_specs(owner_shms, owner_files, tmp_dir):
    for shm in owner_shms:
        try:
            shm.close()
        except Exception:
            pass
        try:
            shm.unlink()
        except Exception:
            pass
    for path in owner_files:
        try:
            Path(path).unlink()
        except Exception:
            pass
    if tmp_dir and os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _compute_psi_for_problem(
    *,
    tpm,
    curr_obs,
    states_full,
    base,
    mechanisms,
    purviews,
    cut_mask_a=None,
    phase1_parallel_workers=None,
    phase1_chunk_size=8,
    phase1_shared_memory=True,
    kernel_cache_path=None,
    kernel_cache_memory_entries=300_000,
    kernel_cache_flush_batch=5_000,
    static_cache=None,
    obs_state_cache=None,
):
    mechanisms = tuple(tuple(m) for m in mechanisms)
    purviews = tuple(tuple(z) for z in purviews)
    if not mechanisms or not purviews:
        return 0.0

    workers_eff = 1
    if phase1_parallel_workers is not None:
        workers_eff = max(1, int(phase1_parallel_workers))
    chunk_size_eff = max(1, int(phase1_chunk_size))
    chunks = _iim_build_phase1_chunks_adaptive(
        mechanisms,
        max_chunk_size=chunk_size_eff,
        workers=workers_eff,
    )

    cache = None
    cache_spec = None
    if kernel_cache_path:
        cache_spec = {
            "enabled": True,
            "path": str(kernel_cache_path),
            "memory_entries": int(kernel_cache_memory_entries),
            "flush_batch": int(kernel_cache_flush_batch),
        }
    if workers_eff <= 1 or len(chunks) <= 1:
        if cache_spec is not None:
            cache = _IIMDiskKernelCache(
                str(kernel_cache_path),
                signature=None,
                memory_entries=int(kernel_cache_memory_entries),
                flush_batch=int(kernel_cache_flush_batch),
            )
        try:
            psi = 0.0
            cache_enabled = cache is not None
            for chunk in chunks:
                psi = float(
                    math.fsum(
                        (
                            psi,
                            float(
                                _iim_phase1_chunk_contribution(
                                    chunk,
                                    purviews,
                                    int(base),
                                    tpm,
                                    curr_obs,
                                    states_full,
                                    static_cache=static_cache,
                                    obs_state_cache=obs_state_cache,
                                    kernel_cache=cache,
                                    cut_mask_a=cut_mask_a,
                                    use_induced_partition_cache=bool(cache_enabled),
                                    kernel_cache_lookup_only=False,
                                )
                            ),
                        )
                    )
                )
            return float(psi)
        finally:
            if cache is not None:
                cache.close()

    owner_shms = []
    owner_files = []
    tmp_dir = tempfile.mkdtemp(prefix="hunter_iim_psi_")
    try:
        spec_curr = _mk_readonly_array_spec(
            "curr_obs",
            curr_obs,
            phase1_shared_memory,
            tmp_dir,
            owner_shms,
            owner_files,
        )
        spec_states = _mk_readonly_array_spec(
            "states_full",
            states_full,
            phase1_shared_memory,
            tmp_dir,
            owner_shms,
            owner_files,
        )
        spec_tpm = _mk_readonly_array_spec(
            "tpm",
            tpm,
            phase1_shared_memory,
            tmp_dir,
            owner_shms,
            owner_files,
        )
        psi_terms = []
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=int(workers_eff),
            initializer=_iim_phase_worker_init_static,
            initargs=(spec_curr, spec_states, int(base), purviews),
        ) as ex:
            futures = [
                ex.submit(
                    _iim_phase_worker_run_chunk_for_tpm,
                    spec_tpm,
                    chunk,
                    cache_spec,
                    (None if cut_mask_a is None else int(cut_mask_a)),
                    bool(cache_spec is not None),
                    False,
                )
                for chunk in chunks
            ]
            for fut in concurrent.futures.as_completed(futures):
                psi_chunk, _chunk_len = fut.result()
                psi_terms.append(float(psi_chunk))
        return float(math.fsum(psi_terms)) if psi_terms else 0.0
    finally:
        _cleanup_specs(owner_shms, owner_files, tmp_dir)


def _run_artifact_dir(campaign_dir: Path, run_key: str) -> Path:
    return campaign_dir / "runs" / str(run_key)


def _load_problem(run_dir: Path):
    meta = _json_load(run_dir / "meta.json")
    if not bool(meta.get("defined", False)):
        return meta, None
    problem = {
        "meta": meta,
        "curr_obs": np.load(run_dir / "curr_obs.npy"),
        "tpm_full": np.load(run_dir / "tpm_full.npy"),
        "states_full": np.load(run_dir / "states_full.npy"),
        "mechanisms_all": [tuple(x) for x in _json_load(run_dir / "mechanisms.json")],
        "purviews_all": [tuple(x) for x in _json_load(run_dir / "purviews.json")],
        "cuts_eval": [
            (tuple(item[0]), tuple(item[1]))
            for item in _json_load(run_dir / "cuts.json")
        ],
    }
    return meta, problem


def prepare_hunter_campaign(
    *,
    data_dir,
    atlas,
    sessions,
    condition,
    stimulus_onsets,
    subjects,
    campaign_dir,
    execution_profile: ExecutionProfile,
    iim_bins,
    iim_lag_trs,
    iim_n_parts,
    iim_max_timepoints,
    iim_max_nodes,
    iim_max_mechanism_size,
    iim_max_purview_size,
    step2_context,
    hardware_target="cpu",
):
    campaign_dir = Path(campaign_dir).resolve()
    hardware_backend = configure_process_for_hardware(hardware_target)
    run_specs = build_ci_run_specs(
        str(data_dir),
        atlas,
        tuple(sessions),
        str(condition),
        stimulus_onsets=stimulus_onsets,
        subjects=subjects,
    )
    unique_specs = []
    seen = set()
    for spec in run_specs:
        ts_path = str(Path(spec["ts_path"]).resolve())
        if ts_path in seen:
            continue
        seen.add(ts_path)
        unique_specs.append(
            {
                "subject": str(spec["subject"]),
                "session": str(spec["session"]),
                "ts_path": ts_path,
            }
        )

    runs = []
    phase1_tasks = []
    cut_tasks = []
    for run_index, spec in enumerate(unique_specs):
        ts_path = Path(spec["ts_path"])
        run_digest = hashlib.sha1(str(ts_path).encode("utf-8")).hexdigest()[:12]
        run_key = f"run-{run_index:04d}_{_sanitize_token(spec['subject'])}_{_sanitize_token(spec['session'])}_{run_digest}"
        run_dir = _run_artifact_dir(campaign_dir, run_key)
        run_dir.mkdir(parents=True, exist_ok=True)

        ts_time_region = np.load(ts_path)
        ts_iim = np.asarray(ts_time_region.T, dtype=float)
        if iim_max_timepoints is not None and int(iim_max_timepoints) > 0 and ts_iim.shape[1] > int(iim_max_timepoints):
            step = int(np.ceil(ts_iim.shape[1] / float(int(iim_max_timepoints))))
            ts_iim = ts_iim[:, ::step]

        prep = prepare_iim_problem(
            ts_iim,
            bins=int(iim_bins),
            lag_trs=int(iim_lag_trs),
            n_parts=iim_n_parts,
            rng=0,
            partition_mode="all",
            max_nodes=iim_max_nodes,
            max_mechanism_size=iim_max_mechanism_size,
            max_purview_size=iim_max_purview_size,
            hardware_backend=hardware_backend,
        )

        meta = {
            "run_index": int(run_index),
            "run_key": str(run_key),
            "subject": str(spec["subject"]),
            "session": str(spec["session"]),
            "ts_path": str(ts_path),
            "dataset_id": step2_context.get("dataset_id"),
            "data_origin": step2_context.get("data_origin"),
            "dataset_role": step2_context.get("dataset_role"),
            "provenance_label": step2_context.get("provenance_label"),
            "defined": bool(prep.get("defined", False)),
            "undefined_reason": prep.get("undefined_reason"),
            "created_unix": float(time.time()),
            "iim_bins": int(iim_bins),
            "iim_lag_trs": int(iim_lag_trs),
            "iim_n_parts": (None if iim_n_parts is None else int(iim_n_parts)),
            "iim_max_timepoints": (None if iim_max_timepoints is None else int(iim_max_timepoints)),
            "iim_max_nodes": (None if iim_max_nodes is None else int(iim_max_nodes)),
            "iim_max_mechanism_size": (
                None if iim_max_mechanism_size is None else int(iim_max_mechanism_size)
            ),
            "iim_max_purview_size": (
                None if iim_max_purview_size is None else int(iim_max_purview_size)
            ),
        }
        if not bool(prep.get("defined", False)):
            _json_dump(run_dir / "meta.json", meta)
            final_payload = {
                "value": None,
                "raw": None,
                "canonical": None,
                "clipped": None,
                "iim_plus": None,
                "defined": False,
                "undefined_reason": str(prep.get("undefined_reason")),
                "n_nodes_used": prep.get("n_nodes_used"),
                "bins_used": prep.get("bins_used"),
                "n_cuts_evaluated": 0,
                "mip_cut": None,
                "phase1_parallel_workers": execution_profile.hunter_phase1_workers_per_task,
                "phase1_chunk_size": int(execution_profile.hunter_phase1_chunk_size),
                "phase1_shared_memory": bool(execution_profile.hunter_shared_memory),
            }
            _json_dump(run_dir / "final_result.json", final_payload)
            runs.append(meta)
            continue

        np.save(run_dir / "curr_obs.npy", prep["curr_obs"], allow_pickle=False)
        np.save(run_dir / "tpm_full.npy", prep["tpm_full"], allow_pickle=False)
        np.save(run_dir / "states_full.npy", prep["states_full"], allow_pickle=False)
        _json_dump(run_dir / "mechanisms.json", [list(x) for x in prep["mechanisms_all"]])
        _json_dump(run_dir / "purviews.json", [list(x) for x in prep["purviews_all"]])
        _json_dump(
            run_dir / "cuts.json",
            [[list(A), list(B)] for A, B in prep["cuts_eval"]],
        )

        phase1_ranges = _split_evenly(
            len(prep["mechanisms_all"]),
            int(execution_profile.hunter_phase1_shards_per_run),
        )
        cut_ranges = _split_evenly(
            len(prep["cuts_eval"]),
            int(execution_profile.hunter_cut_shards_per_run),
        )
        meta.update(
            {
                "defined": True,
                "selected_nodes": list(prep["selected_nodes"]),
                "n_regions_input": int(prep["n_regions_input"]),
                "n_time_input": int(prep["n_time_input"]),
                "n_nodes_used": int(prep["n_nodes_used"]),
                "bins_used": int(prep["bins_used"]),
                "max_mechanism_size_used": int(prep["max_mechanism_size_used"]),
                "max_purview_size_used": int(prep["max_purview_size_used"]),
                "n_cuts_evaluated": int(len(prep["cuts_eval"])),
                "n_mechanisms": int(len(prep["mechanisms_all"])),
                "n_purviews": int(len(prep["purviews_all"])),
                "phase1_shards": [
                    {"task_index": int(i), "start": int(a), "stop": int(b)}
                    for i, (a, b) in enumerate(phase1_ranges)
                ],
                "cut_shards": [
                    {"task_index": int(i), "start": int(a), "stop": int(b)}
                    for i, (a, b) in enumerate(cut_ranges)
                ],
            }
        )
        _json_dump(run_dir / "meta.json", meta)
        for shard in meta["phase1_shards"]:
            phase1_tasks.append({"run_key": str(run_key), **shard})
        for shard in meta["cut_shards"]:
            cut_tasks.append({"run_key": str(run_key), **shard})
        runs.append(meta)

    manifest = {
        "created_unix": float(time.time()),
        "campaign_dir": str(campaign_dir),
        "data_dir": str(Path(data_dir).resolve()),
        "dataset_id": step2_context.get("dataset_id"),
        "data_origin": step2_context.get("data_origin"),
        "dataset_role": step2_context.get("dataset_role"),
        "provenance_label": step2_context.get("provenance_label"),
        "atlas": str(atlas),
        "sessions": list(sessions),
        "condition": str(condition),
        "execution_profile": dataclasses.asdict(execution_profile),
        "hardware_backend": hardware_backend.to_dict(),
        "runs": runs,
        "phase1_tasks": phase1_tasks,
        "cut_tasks": cut_tasks,
        "step2_context": step2_context,
    }
    _json_dump(campaign_dir / "campaign_manifest.json", manifest)
    _write_hunter_slurm_scripts(campaign_dir, manifest)
    return manifest


def _write_hunter_slurm_scripts(campaign_dir: Path, manifest):
    profile = manifest["execution_profile"]
    slurm = dict(profile.get("hunter_slurm") or {})
    env_overrides = {
        "IMPACT_HUNTER_APU_PARTITION": "apu_partition",
        "IMPACT_HUNTER_CPU_PARTITION": "cpu_partition",
        "IMPACT_HUNTER_SLURM_ACCOUNT": "account",
        "IMPACT_HUNTER_SLURM_QOS": "qos",
        "IMPACT_HUNTER_PHASE1_TIME": "phase1_time",
        "IMPACT_HUNTER_CUT_TIME": "cut_time",
        "IMPACT_HUNTER_REDUCE_TIME": "reduce_time",
        "IMPACT_HUNTER_CPUS_PER_TASK": "cpus_per_task",
        "IMPACT_HUNTER_MEM_PER_TASK": "mem_per_task",
        "IMPACT_HUNTER_GPUS_PER_TASK": "gpus_per_task",
    }
    for env_name, slurm_key in env_overrides.items():
        env_value = os.environ.get(env_name)
        if env_value is not None and str(env_value).strip():
            slurm[slurm_key] = str(env_value).strip()

    ctx = manifest.get("step2_context") or {}
    hw = manifest.get("hardware_backend") or {}
    hardware_target = str(ctx.get("hardware_target") or hw.get("requested") or "cpu")
    use_accelerator_partition = hardware_target in {"gpu", "hunter-apu"}
    account_line = ""
    if slurm.get("account"):
        account_line += f"#SBATCH --account={slurm['account']}\n"
    if slurm.get("qos"):
        account_line += f"#SBATCH --qos={slurm['qos']}\n"
    repo_root = Path(__file__).resolve().parents[2]

    conda_env = os.environ.get("IMPACT_HUNTER_CONDA_ENV") or os.environ.get("IMPACT_CONDA_ENV")
    if conda_env:
        conda_bin = (
            os.environ.get("IMPACT_HUNTER_CONDA_BIN")
            or os.environ.get("CONDA_BIN")
            or "conda"
        )
        python_launcher = [conda_bin, "run", "-n", conda_env, "python"]
    else:
        python_launcher = [os.environ.get("IMPACT_HUNTER_PYTHON", "python")]

    setup_chunks = []
    setup_file = os.environ.get("IMPACT_HUNTER_SLURM_SETUP_FILE")
    if setup_file:
        setup_path = Path(setup_file).expanduser()
        if not setup_path.exists():
            raise FileNotFoundError(f"Missing IMPACT_HUNTER_SLURM_SETUP_FILE: {setup_path}")
        setup_chunks.append(setup_path.read_text(encoding="utf-8").rstrip())
    inline_setup = os.environ.get("IMPACT_HUNTER_SLURM_SETUP")
    if inline_setup:
        setup_chunks.append(str(inline_setup).rstrip())

    runtime_preamble = "set -euo pipefail\n"
    runtime_preamble += f"cd {shlex.quote(str(repo_root))}\n"
    if setup_chunks:
        runtime_preamble += "\n".join(chunk for chunk in setup_chunks if chunk.strip()) + "\n"

    base_opts = [
        *python_launcher,
        str(repo_root / "run_pipeline.py"),
        "--execution-mode",
        "hunter",
        "--hunter-campaign-dir",
        str(campaign_dir),
        "--hardware-target",
        hardware_target,
    ]
    if ctx.get("dataset_id"):
        base_opts.extend(["--dataset-id", str(ctx["dataset_id"])])
    if ctx.get("data_origin"):
        base_opts.extend(["--data-origin", str(ctx["data_origin"])])
    if ctx.get("out_dir"):
        base_opts.extend(["--out-dir", str(ctx["out_dir"])])
    py_cmd = " ".join(shlex.quote(x) for x in base_opts)

    scripts_dir = campaign_dir / "slurm"
    scripts_dir.mkdir(parents=True, exist_ok=True)

    phase1_tasks = manifest.get("phase1_tasks", [])
    cut_tasks = manifest.get("cut_tasks", [])
    runs = manifest.get("runs", [])

    def _partition(default_kind: str) -> str:
        if use_accelerator_partition:
            return str(slurm.get("apu_partition", "apu"))
        return str(slurm.get("cpu_partition", "cpu"))

    def _gpu_line() -> str:
        if not use_accelerator_partition:
            return ""
        gpus = int(slurm.get("gpus_per_task", 0) or 0)
        if gpus <= 0:
            gpus = 1
        return f"#SBATCH --gpus-per-task={gpus}\n"

    def _write_script(name, text):
        path = scripts_dir / name
        path.write_text(text, encoding="utf-8")
        path.chmod(0o755)

    if phase1_tasks:
        _write_script(
            "01_phase1_shards.sbatch",
            (
                "#!/bin/bash\n"
                f"#SBATCH --job-name=impact_iim_p1\n"
                f"#SBATCH --partition={_partition('cpu')}\n"
                f"#SBATCH --time={slurm.get('phase1_time', '24:00:00')}\n"
                f"#SBATCH --cpus-per-task={int(slurm.get('cpus_per_task', 32))}\n"
                f"#SBATCH --mem={slurm.get('mem_per_task', '0')}\n"
                f"{_gpu_line()}"
                f"#SBATCH --array=0-{len(phase1_tasks) - 1}\n"
                f"{account_line}"
                f"{runtime_preamble}"
                f"{py_cmd} --hunter-stage phase1-shard --hunter-task-index $SLURM_ARRAY_TASK_ID\n"
            ),
        )
    if runs:
        _write_script(
            "02_phase1_reduce.sbatch",
            (
                "#!/bin/bash\n"
                f"#SBATCH --job-name=impact_iim_p1r\n"
                f"#SBATCH --partition={_partition('cpu')}\n"
                f"#SBATCH --time={slurm.get('reduce_time', '02:00:00')}\n"
                f"#SBATCH --cpus-per-task=1\n"
                f"#SBATCH --mem={slurm.get('mem_per_task', '0')}\n"
                f"{_gpu_line()}"
                f"#SBATCH --array=0-{len(runs) - 1}\n"
                f"{account_line}"
                f"{runtime_preamble}"
                f"{py_cmd} --hunter-stage phase1-reduce --hunter-run-index $SLURM_ARRAY_TASK_ID\n"
            ),
        )
    if cut_tasks:
        _write_script(
            "03_cut_shards.sbatch",
            (
                "#!/bin/bash\n"
                f"#SBATCH --job-name=impact_iim_cut\n"
                f"#SBATCH --partition={_partition('apu')}\n"
                f"#SBATCH --time={slurm.get('cut_time', '24:00:00')}\n"
                f"#SBATCH --cpus-per-task={int(slurm.get('cpus_per_task', 32))}\n"
                f"#SBATCH --mem={slurm.get('mem_per_task', '0')}\n"
                f"{_gpu_line()}"
                f"#SBATCH --array=0-{len(cut_tasks) - 1}\n"
                f"{account_line}"
                f"{runtime_preamble}"
                f"{py_cmd} --hunter-stage cut-shard --hunter-task-index $SLURM_ARRAY_TASK_ID\n"
            ),
        )
    if runs:
        _write_script(
            "04_cut_reduce.sbatch",
            (
                "#!/bin/bash\n"
                f"#SBATCH --job-name=impact_iim_red\n"
                f"#SBATCH --partition={_partition('cpu')}\n"
                f"#SBATCH --time={slurm.get('reduce_time', '02:00:00')}\n"
                f"#SBATCH --cpus-per-task=1\n"
                f"#SBATCH --mem={slurm.get('mem_per_task', '0')}\n"
                f"{_gpu_line()}"
                f"#SBATCH --array=0-{len(runs) - 1}\n"
                f"{account_line}"
                f"{runtime_preamble}"
                f"{py_cmd} --hunter-stage cut-reduce --hunter-run-index $SLURM_ARRAY_TASK_ID\n"
            ),
        )
        _write_script(
            "05_finalize_pipeline.sbatch",
            (
                "#!/bin/bash\n"
                f"#SBATCH --job-name=impact_finalize\n"
                f"#SBATCH --partition={_partition('cpu')}\n"
                f"#SBATCH --time={slurm.get('reduce_time', '02:00:00')}\n"
                f"#SBATCH --cpus-per-task=4\n"
                f"#SBATCH --mem={slurm.get('mem_per_task', '0')}\n"
                f"{_gpu_line()}"
                f"{account_line}"
                f"{runtime_preamble}"
                f"{py_cmd} --hunter-stage finalize-pipeline\n"
            ),
        )
        submit_lines = [
            "#!/bin/bash",
            "set -euo pipefail",
            'script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
            'campaign_dir="$(cd "${script_dir}/.." && pwd)"',
            'cd "${campaign_dir}"',
        ]
        if phase1_tasks:
            submit_lines.extend(
                [
                    'jid_p1=$(sbatch --parsable slurm/01_phase1_shards.sbatch)',
                    'jid_p1r=$(sbatch --parsable --dependency=afterok:${jid_p1} slurm/02_phase1_reduce.sbatch)',
                ]
            )
        else:
            submit_lines.append('jid_p1r=$(sbatch --parsable slurm/02_phase1_reduce.sbatch)')
        if cut_tasks:
            submit_lines.extend(
                [
                    'jid_cut=$(sbatch --parsable --dependency=afterok:${jid_p1r} slurm/03_cut_shards.sbatch)',
                    'jid_red=$(sbatch --parsable --dependency=afterok:${jid_cut} slurm/04_cut_reduce.sbatch)',
                ]
            )
        else:
            submit_lines.append('jid_red=$(sbatch --parsable --dependency=afterok:${jid_p1r} slurm/04_cut_reduce.sbatch)')
        submit_lines.extend(
            [
                'jid_fin=$(sbatch --parsable --dependency=afterok:${jid_red} slurm/05_finalize_pipeline.sbatch)',
                'echo "phase1_reduce=${jid_p1r}"',
                'echo "cut_reduce=${jid_red}"',
                'echo "finalize=${jid_fin}"',
            ]
        )
        _write_script("00_submit_all.sh", "\n".join(submit_lines) + "\n")


def run_phase1_shard(campaign_dir, task_index):
    campaign_dir = Path(campaign_dir).resolve()
    manifest = _json_load(campaign_dir / "campaign_manifest.json")
    task = manifest["phase1_tasks"][int(task_index)]
    run_dir = _run_artifact_dir(campaign_dir, task["run_key"])
    meta, problem = _load_problem(run_dir)
    out_path = run_dir / "phase1_shards" / f"shard_{int(task['task_index']):04d}.json"
    if not bool(meta.get("defined", False)):
        payload = {"status": "skipped", "defined": False, "psi_partial": None}
        _json_dump(out_path, payload)
        return payload

    shard_mechanisms = problem["mechanisms_all"][int(task["start"]):int(task["stop"])]
    kernel_cache_path = run_dir / "phase1_shards" / f"kernel_{int(task['task_index']):04d}.sqlite3"
    psi_partial = _compute_psi_for_problem(
        tpm=problem["tpm_full"],
        curr_obs=problem["curr_obs"],
        states_full=problem["states_full"],
        base=int(meta["bins_used"]),
        mechanisms=shard_mechanisms,
        purviews=problem["purviews_all"],
        cut_mask_a=None,
        phase1_parallel_workers=manifest["execution_profile"]["hunter_phase1_workers_per_task"],
        phase1_chunk_size=manifest["execution_profile"]["hunter_phase1_chunk_size"],
        phase1_shared_memory=manifest["execution_profile"]["hunter_shared_memory"],
        kernel_cache_path=str(kernel_cache_path),
    )
    payload = {
        "status": "complete",
        "defined": True,
        "run_key": str(task["run_key"]),
        "task_index": int(task["task_index"]),
        "start": int(task["start"]),
        "stop": int(task["stop"]),
        "n_mechanisms": int(len(shard_mechanisms)),
        "psi_partial": float(psi_partial),
    }
    _json_dump(out_path, payload)
    return payload


def run_phase1_reduce(campaign_dir, run_index):
    campaign_dir = Path(campaign_dir).resolve()
    manifest = _json_load(campaign_dir / "campaign_manifest.json")
    run_meta = manifest["runs"][int(run_index)]
    run_dir = _run_artifact_dir(campaign_dir, run_meta["run_key"])
    if not bool(run_meta.get("defined", False)):
        payload = {
            "status": "skipped",
            "defined": False,
            "undefined_reason": run_meta.get("undefined_reason"),
        }
        _json_dump(run_dir / "phase1_result.json", payload)
        return payload

    partials = []
    for shard in run_meta["phase1_shards"]:
        shard_path = run_dir / "phase1_shards" / f"shard_{int(shard['task_index']):04d}.json"
        if not shard_path.exists():
            raise FileNotFoundError(f"Missing phase1 shard result: {shard_path}")
        rec = _json_load(shard_path)
        if rec.get("status") != "complete":
            raise RuntimeError(f"Phase1 shard did not complete: {shard_path}")
        partials.append(float(rec["psi_partial"]))

    psi_full = float(math.fsum(partials)) if partials else 0.0
    payload = {
        "status": "complete",
        "defined": True,
        "psi_full": float(psi_full),
        "n_shards": int(len(partials)),
    }
    _json_dump(run_dir / "phase1_result.json", payload)
    return payload


def run_cut_shard(campaign_dir, task_index):
    campaign_dir = Path(campaign_dir).resolve()
    manifest = _json_load(campaign_dir / "campaign_manifest.json")
    hardware_backend = configure_process_for_hardware(
        (manifest.get("step2_context") or {}).get("hardware_target")
        or (manifest.get("hardware_backend") or {}).get("requested")
        or "cpu"
    )
    task = manifest["cut_tasks"][int(task_index)]
    run_dir = _run_artifact_dir(campaign_dir, task["run_key"])
    meta, problem = _load_problem(run_dir)
    out_path = run_dir / "cut_shards" / f"shard_{int(task['task_index']):04d}.json"
    if not bool(meta.get("defined", False)):
        payload = {"status": "skipped", "defined": False, "best_cut": None, "best_psi": None}
        _json_dump(out_path, payload)
        return payload

    phase1_path = run_dir / "phase1_result.json"
    if not phase1_path.exists():
        raise FileNotFoundError(f"Missing phase1 reduction result: {phase1_path}")

    shard_cuts = problem["cuts_eval"][int(task["start"]):int(task["stop"])]
    static_cache = {}
    obs_state_cache = {}
    cut_scores = {}
    best_cut = None
    best_psi = -np.inf
    for local_idx, (A, B) in enumerate(shard_cuts):
        cut_mask_a = 0
        for nn in A:
            cut_mask_a |= (1 << int(nn))
        tpm_cut = _iim_build_cut_tpm(
            problem["tpm_full"],
            problem["states_full"],
            int(meta["bins_used"]),
            A,
            B,
            hardware_backend=hardware_backend,
        )
        kernel_cache_path = run_dir / "cut_shards" / f"kernel_{int(task['task_index']):04d}_{int(local_idx):04d}.sqlite3"
        psi_cut = _compute_psi_for_problem(
            tpm=tpm_cut,
            curr_obs=problem["curr_obs"],
            states_full=problem["states_full"],
            base=int(meta["bins_used"]),
            mechanisms=problem["mechanisms_all"],
            purviews=problem["purviews_all"],
            cut_mask_a=int(cut_mask_a),
            phase1_parallel_workers=manifest["execution_profile"]["hunter_phase1_workers_per_task"],
            phase1_chunk_size=manifest["execution_profile"]["hunter_phase1_chunk_size"],
            phase1_shared_memory=manifest["execution_profile"]["hunter_shared_memory"],
            kernel_cache_path=str(kernel_cache_path),
            static_cache=static_cache,
            obs_state_cache=obs_state_cache,
        )
        cut_key = _iim_cut_to_key(A, B)
        cut_scores[cut_key] = float(psi_cut)
        if float(psi_cut) > float(best_psi):
            best_psi = float(psi_cut)
            best_cut = [list(A), list(B)]

    payload = {
        "status": "complete",
        "defined": True,
        "run_key": str(task["run_key"]),
        "task_index": int(task["task_index"]),
        "start": int(task["start"]),
        "stop": int(task["stop"]),
        "cut_scores": cut_scores,
        "best_cut": best_cut,
        "best_psi": (None if not np.isfinite(best_psi) else float(best_psi)),
    }
    _json_dump(out_path, payload)
    return payload


def run_cut_reduce(campaign_dir, run_index, clamp=True, scale=1.0):
    campaign_dir = Path(campaign_dir).resolve()
    manifest = _json_load(campaign_dir / "campaign_manifest.json")
    run_meta = manifest["runs"][int(run_index)]
    run_dir = _run_artifact_dir(campaign_dir, run_meta["run_key"])
    if not bool(run_meta.get("defined", False)):
        final_path = run_dir / "final_result.json"
        if final_path.exists():
            return _json_load(final_path)
        payload = {
            "value": None,
            "raw": None,
            "canonical": None,
            "clipped": None,
            "iim_plus": None,
            "defined": False,
            "undefined_reason": run_meta.get("undefined_reason"),
        }
        _json_dump(final_path, payload)
        return payload

    phase1 = _json_load(run_dir / "phase1_result.json")
    psi_full = float(phase1["psi_full"])
    if not np.isfinite(psi_full) or psi_full <= 0:
        payload = {
            "value": None,
            "raw": None,
            "canonical": None,
            "clipped": None,
            "iim_plus": None,
            "defined": False,
            "undefined_reason": "nonpositive_psi_full",
            "Psi_full": float(psi_full),
            "Psi_mip_preserved": None,
            "n_nodes_used": int(run_meta["n_nodes_used"]),
            "bins_used": int(run_meta["bins_used"]),
            "n_cuts_evaluated": int(run_meta["n_cuts_evaluated"]),
        }
        _json_dump(run_dir / "final_result.json", payload)
        return payload

    best_psi = -np.inf
    best_cut = None
    for shard in run_meta["cut_shards"]:
        shard_path = run_dir / "cut_shards" / f"shard_{int(shard['task_index']):04d}.json"
        if not shard_path.exists():
            raise FileNotFoundError(f"Missing cut shard result: {shard_path}")
        rec = _json_load(shard_path)
        if rec.get("status") != "complete":
            raise RuntimeError(f"Cut shard did not complete: {shard_path}")
        psi = rec.get("best_psi")
        if psi is None:
            continue
        psi = float(psi)
        if psi > best_psi:
            best_psi = psi
            best_cut = rec.get("best_cut")

    if not np.isfinite(best_psi):
        payload = {
            "value": None,
            "raw": None,
            "canonical": None,
            "clipped": None,
            "iim_plus": None,
            "defined": False,
            "undefined_reason": "mip_not_found",
            "Psi_full": float(psi_full),
            "Psi_mip_preserved": None,
            "n_nodes_used": int(run_meta["n_nodes_used"]),
            "bins_used": int(run_meta["bins_used"]),
            "n_cuts_evaluated": int(run_meta["n_cuts_evaluated"]),
        }
        _json_dump(run_dir / "final_result.json", payload)
        return payload

    raw = float((float(psi_full) - float(best_psi)) / (float(psi_full) + 1e-12))
    canonical = float(np.clip(raw, 0.0, 1.0))
    selected = canonical if bool(clamp) else raw
    value = float(float(scale) * float(selected))
    payload = {
        "value": value,
        "raw": raw,
        "canonical": canonical,
        "clipped": canonical,
        "iim_plus": canonical,
        "scale": float(scale),
        "I_full": float(psi_full),
        "min_partition_sum": float(best_psi),
        "Psi_full": float(psi_full),
        "Psi_mip_preserved": float(best_psi),
        "n_nodes_used": int(run_meta["n_nodes_used"]),
        "bins_used": int(run_meta["bins_used"]),
        "max_nodes_requested": run_meta.get("iim_max_nodes"),
        "max_mechanism_size_used": int(run_meta["max_mechanism_size_used"]),
        "max_purview_size_used": int(run_meta["max_purview_size_used"]),
        "n_parts_requested": run_meta.get("iim_n_parts"),
        "n_cuts_evaluated": int(run_meta["n_cuts_evaluated"]),
        "mip_cut": best_cut,
        "checkpoint_path": None,
        "checkpoint_resumed": False,
        "checkpoint_reused_cuts": 0,
        "checkpoint_used_psi_full": False,
        "phase1_resumed_partial": False,
        "phase1_psi_partial": float(psi_full),
        "phase1_mechanisms_done": int(run_meta["n_mechanisms"]),
        "phase1_total_mechanisms": int(run_meta["n_mechanisms"]),
        "phase1_eta_seconds": None,
        "phase1_parallel_workers": manifest["execution_profile"]["hunter_phase1_workers_per_task"],
        "phase1_chunk_size": int(manifest["execution_profile"]["hunter_phase1_chunk_size"]),
        "phase1_shared_memory": bool(manifest["execution_profile"]["hunter_shared_memory"]),
        "induced_partition_cache_enabled": False,
        "induced_partition_cache_path": None,
        "induced_partition_cache_stats": None,
        "defined": True,
        "undefined_reason": None,
    }
    _json_dump(run_dir / "final_result.json", payload)
    return payload


def collect_iim_results_by_path(campaign_dir):
    campaign_dir = Path(campaign_dir).resolve()
    manifest = _json_load(campaign_dir / "campaign_manifest.json")
    out = {}
    for run_meta in manifest.get("runs", []):
        run_dir = _run_artifact_dir(campaign_dir, run_meta["run_key"])
        final_path = run_dir / "final_result.json"
        if not final_path.exists():
            raise FileNotFoundError(f"Missing final IIM result: {final_path}")
        out[str(run_meta["ts_path"])] = _json_load(final_path)
    return out
