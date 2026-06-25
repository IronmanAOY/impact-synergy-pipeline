from __future__ import annotations

import importlib
import os
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


class HardwareBackendError(RuntimeError):
    """Raised when a requested hardware backend is unavailable or invalid."""


@dataclass(frozen=True)
class HardwareBackend:
    requested: str
    target: str
    accelerator: bool
    array_module: str
    runtime: str
    device_count: int
    device_name: str
    strict: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


CPU_BACKEND = HardwareBackend(
    requested="cpu",
    target="cpu",
    accelerator=False,
    array_module="numpy",
    runtime="cpu",
    device_count=0,
    device_name="CPU",
    strict=False,
)


def normalize_hardware_target(target: str | None) -> str:
    key = str(target or "cpu").strip().lower().replace("_", "-")
    aliases = {
        "local": "cpu",
        "host": "cpu",
        "none": "cpu",
        "accelerator": "gpu",
        "rocm": "hunter-apu",
        "hip": "hunter-apu",
        "apu": "hunter-apu",
        "hunter": "hunter-apu",
        "hunter-gpu": "hunter-apu",
        "mi300a": "hunter-apu",
    }
    key = aliases.get(key, key)
    if key not in {"cpu", "auto", "gpu", "hunter-apu"}:
        raise HardwareBackendError(
            f"Unknown hardware target '{target}'. Expected one of: cpu, auto, gpu, hunter-apu."
        )
    return key


def _cupy_runtime_name(cp) -> str:
    is_hip = getattr(cp.cuda.runtime, "is_hip", None)
    try:
        if callable(is_hip):
            is_hip = bool(is_hip())
    except Exception:
        is_hip = None
    if is_hip is True:
        return "rocm"
    if is_hip is False:
        return "cuda"

    # Older/newer CuPy builds may not expose is_hip consistently. Version text
    # is a secondary signal, used only when runtime.is_hip is unavailable.
    version_text = " ".join(
        str(getattr(obj, "__version__", ""))
        for obj in (
            cp,
            getattr(cp, "cuda", None),
            getattr(getattr(cp, "cuda", None), "runtime", None),
        )
    ).lower()
    if "rocm" in version_text or "hip" in version_text:
        return "rocm"
    return "cuda"


def _load_cupy_backend(requested: str, *, require_rocm: bool, strict: bool) -> HardwareBackend:
    try:
        cp = importlib.import_module("cupy")
    except Exception as exc:
        raise HardwareBackendError(
            f"Hardware target '{requested}' requires CuPy, but CuPy could not be imported: {exc}"
        ) from exc

    try:
        device_count = int(cp.cuda.runtime.getDeviceCount())
    except Exception as exc:
        raise HardwareBackendError(
            f"Hardware target '{requested}' requires a visible GPU/APU device, "
            f"but CuPy could not query devices: {exc}"
        ) from exc
    if device_count < 1:
        raise HardwareBackendError(
            f"Hardware target '{requested}' requires a visible GPU/APU device; CuPy reported 0 devices."
        )

    runtime = _cupy_runtime_name(cp)
    if require_rocm and runtime != "rocm":
        raise HardwareBackendError(
            f"Hardware target '{requested}' requires a ROCm/HIP CuPy runtime for Hunter APU, "
            f"but detected runtime='{runtime}'."
        )

    try:
        props = cp.cuda.runtime.getDeviceProperties(0)
        raw_name = props.get("name", b"") if isinstance(props, dict) else b""
        if isinstance(raw_name, bytes):
            device_name = raw_name.decode("utf-8", errors="replace")
        else:
            device_name = str(raw_name or "GPU/APU")
    except Exception:
        device_name = "GPU/APU"

    return HardwareBackend(
        requested=str(requested),
        target="hunter-apu" if require_rocm else "gpu",
        accelerator=True,
        array_module="cupy",
        runtime=runtime,
        device_count=device_count,
        device_name=device_name,
        strict=bool(strict),
    )


def resolve_hardware_backend(target: str | HardwareBackend | dict | None = None) -> HardwareBackend:
    if isinstance(target, HardwareBackend):
        return target
    if isinstance(target, dict):
        raw = target.get("requested") or target.get("target") or "cpu"
    else:
        raw = target
    normalized = normalize_hardware_target(raw)
    if normalized == "cpu":
        return CPU_BACKEND
    if normalized == "auto":
        try:
            return _load_cupy_backend("auto", require_rocm=False, strict=False)
        except HardwareBackendError:
            return HardwareBackend(
                requested="auto",
                target="cpu",
                accelerator=False,
                array_module="numpy",
                runtime="cpu",
                device_count=0,
                device_name="CPU",
                strict=False,
            )
    if normalized == "gpu":
        return _load_cupy_backend("gpu", require_rocm=False, strict=True)
    if normalized == "hunter-apu":
        return _load_cupy_backend("hunter-apu", require_rocm=True, strict=True)
    raise HardwareBackendError(f"Unhandled hardware target '{target}'.")


def configure_process_for_hardware(backend: str | HardwareBackend | dict | None) -> HardwareBackend:
    resolved = resolve_hardware_backend(backend)
    if resolved.accelerator:
        # Keep CPU support libraries conservative when the accelerator is doing
        # the matrix-heavy work. This avoids accidental CPU oversubscription in
        # Slurm array tasks and local multi-worker runs.
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    return resolved


def get_array_module(backend: str | HardwareBackend | dict | None):
    resolved = resolve_hardware_backend(backend)
    if not resolved.accelerator:
        return np
    return importlib.import_module("cupy")


def is_accelerator_backend(backend: str | HardwareBackend | dict | None) -> bool:
    return bool(resolve_hardware_backend(backend).accelerator)


def backend_summary(backend: str | HardwareBackend | dict | None) -> str:
    resolved = resolve_hardware_backend(backend)
    if not resolved.accelerator:
        return f"{resolved.requested}->cpu"
    return (
        f"{resolved.requested}->{resolved.target} "
        f"runtime={resolved.runtime} devices={resolved.device_count} device0={resolved.device_name}"
    )


def to_numpy(value):
    if isinstance(value, np.ndarray):
        return value
    mod = type(value).__module__.split(".", 1)[0]
    if mod == "cupy":
        cp = importlib.import_module("cupy")
        return cp.asnumpy(value)
    return np.asarray(value)


def _xp_array(value, backend):
    xp = get_array_module(backend)
    if xp is np:
        return np.asarray(value)
    return xp.asarray(value)


def accelerated_dot(a, b, backend=None):
    resolved = resolve_hardware_backend(backend)
    if not resolved.accelerator:
        return np.asarray(a).dot(np.asarray(b))
    xp = get_array_module(resolved)
    out = xp.asarray(a).dot(xp.asarray(b))
    return xp.asnumpy(out)


def accelerated_pinv_dot(a, b, backend=None):
    resolved = resolve_hardware_backend(backend)
    if not resolved.accelerator:
        return np.linalg.pinv(np.asarray(a)).dot(np.asarray(b))
    xp = get_array_module(resolved)
    out = xp.linalg.pinv(xp.asarray(a)).dot(xp.asarray(b))
    return xp.asnumpy(out)


def accelerated_corrcoef(a, backend=None):
    resolved = resolve_hardware_backend(backend)
    if not resolved.accelerator:
        return np.corrcoef(np.asarray(a))
    xp = get_array_module(resolved)
    out = xp.corrcoef(xp.asarray(a))
    return xp.asnumpy(out)


def accelerated_svd_values(a, backend=None):
    resolved = resolve_hardware_backend(backend)
    if not resolved.accelerator:
        return np.linalg.svd(np.asarray(a), full_matrices=False, compute_uv=False)
    xp = get_array_module(resolved)
    out = xp.linalg.svd(xp.asarray(a), full_matrices=False, compute_uv=False)
    return xp.asnumpy(out)


def accelerated_solve(a, b, backend=None):
    resolved = resolve_hardware_backend(backend)
    if not resolved.accelerator:
        return np.linalg.solve(np.asarray(a), np.asarray(b))
    xp = get_array_module(resolved)
    out = xp.linalg.solve(xp.asarray(a), xp.asarray(b))
    return xp.asnumpy(out)


def accelerated_zscore(a, axis=0, backend=None, eps=1e-12):
    resolved = resolve_hardware_backend(backend)
    if not resolved.accelerator:
        x = np.asarray(a, dtype=float)
        mean = np.nanmean(x, axis=axis, keepdims=True)
        std = np.nanstd(x, axis=axis, ddof=0, keepdims=True) + float(eps)
        return np.nan_to_num((x - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)
    xp = get_array_module(resolved)
    x = xp.asarray(a, dtype=xp.float64)
    mean = xp.nanmean(x, axis=axis, keepdims=True)
    std = xp.nanstd(x, axis=axis, ddof=0, keepdims=True) + float(eps)
    out = xp.nan_to_num((x - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)
    return xp.asnumpy(out)


def accelerated_row_norm(a, axis=1, backend=None):
    resolved = resolve_hardware_backend(backend)
    if not resolved.accelerator:
        return np.linalg.norm(np.asarray(a), axis=axis)
    xp = get_array_module(resolved)
    out = xp.linalg.norm(xp.asarray(a), axis=axis)
    return xp.asnumpy(out)


def accelerated_psd_invsqrt(mat, eps=1e-10, backend=None):
    resolved = resolve_hardware_backend(backend)
    if not resolved.accelerator:
        sym = 0.5 * (np.asarray(mat) + np.asarray(mat).T)
        vals, vecs = np.linalg.eigh(sym)
        vals = np.maximum(vals, float(eps))
        return (vecs * (1.0 / np.sqrt(vals))) @ vecs.T
    xp = get_array_module(resolved)
    m = xp.asarray(mat, dtype=xp.float64)
    sym = 0.5 * (m + m.T)
    vals, vecs = xp.linalg.eigh(sym)
    vals = xp.maximum(vals, float(eps))
    out = (vecs * (1.0 / xp.sqrt(vals))) @ vecs.T
    return xp.asnumpy(out)
