#!/usr/bin/env python3
"""
Collect high-accuracy thermal telemetry via macOS powermetrics and write cache JSON.

Run with sudo:
  sudo -E conda run -n impact-synergy-clean python scripts/powermetrics_telemetry.py --out-dir outputs/scratch
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any


@functools.lru_cache(maxsize=1)
def _supported_samplers() -> set[str]:
    """
    Return sampler names reported by `powermetrics --help`.
    """
    out = ""
    try:
        out = subprocess.check_output(
            ["powermetrics", "--help"], text=True, stderr=subprocess.STDOUT
        )
    except Exception:
        return set()

    samplers: set[str] = set()
    capture = False
    for ln in out.splitlines():
        s = ln.rstrip()
        if s.strip().startswith("The following samplers are supported by --samplers:"):
            capture = True
            continue
        if capture and s.strip().startswith("and the following sampler groups are supported"):
            break
        if not capture:
            continue
        m = re.match(r"^\s*([a-z_]+)\s{2,}.*$", s)
        if m:
            samplers.add(m.group(1).strip())
    return samplers


def _extract_first_float(text: str, patterns: list[str]) -> float | None:
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE | re.MULTILINE)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                continue
    return None


def _extract_first_text(text: str, patterns: list[str]) -> str | None:
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE | re.MULTILINE)
        if m:
            val = str(m.group(1)).strip()
            if val:
                return val
    return None


def _extract_first_power_w(text: str, patterns: list[str]) -> float | None:
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE | re.MULTILINE)
        if not m:
            continue
        try:
            val = float(m.group(1))
        except Exception:
            continue
        unit = "w"
        if m.lastindex and m.lastindex >= 2:
            unit = str(m.group(2) or "w").strip().lower()
        if unit == "mw":
            val = val / 1000.0
        return val
    return None


def _parse_powermetrics_output(raw: str) -> dict[str, Any]:
    cpu_temp_c = _extract_first_float(
        raw,
        [
            r"CPU(?:\s+die)?\s+temperature\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)\s*°?\s*C?",
            r"CPU temp(?:erature)?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)\s*°?\s*C?",
            r"PECI CPU temperature\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)\s*°?\s*C?",
            r"Die temperature\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)\s*°?\s*C?",
            r"ANE(?:\s+die)?\s+temperature\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)\s*°?\s*C?",
        ],
    )
    cpu_speed_limit = _extract_first_float(
        raw,
        [
            r"CPU(?:[_ ]Speed)?[_ ]?Limit\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)\s*%?",
            r"CPU[_ ]?PLimit\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)\s*%?",
            r"CPU speed limit\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)\s*%?",
            r"CPU power limit(?:\s*\(.*\))?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)\s*%?",
        ],
    )
    if cpu_speed_limit is not None:
        if 0.0 <= cpu_speed_limit <= 1.0:
            cpu_speed_limit = 100.0 * cpu_speed_limit
        cpu_speed_limit = max(0.0, min(100.0, cpu_speed_limit))

    cpu_power_w = _extract_first_power_w(
        raw,
        [
            r"CPU Power\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*(mW|W)",
            r"Package Power\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*(mW|W)",
            r"CPU package.*?:\s*([0-9]+(?:\.[0-9]+)?)\s*(mW|W)",
            r"Combined Power(?:\s*\(.*?\))?\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*(mW|W)",
        ],
    )
    thermal_pressure = _extract_first_text(
        raw,
        [
            r"Thermal pressure(?: level)?\s*:\s*([A-Za-z0-9_\- ]+)",
            r"ThermalLevel\s*[:=]\s*([A-Za-z0-9_\- ]+)",
            r"Current pressure level\s*[:=]\s*([A-Za-z0-9_\- ]+)",
            r"CPU(?:\s+)?thermal(?:\s+)?pressure\s*[:=]\s*([A-Za-z0-9_\- ]+)",
        ],
    )

    return {
        "cpu_temp_c": cpu_temp_c,
        "cpu_speed_limit_pct": (None if cpu_speed_limit is None else round(cpu_speed_limit, 2)),
        "cpu_power_w": cpu_power_w,
        "thermal_pressure": thermal_pressure,
    }


def _run_powermetrics(sample_ms: int, samplers: str) -> dict[str, Any]:
    cmd = [
        "powermetrics",
        "-n",
        "1",
        "-i",
        str(int(sample_ms)),
        "--samplers",
        samplers,
        "--show-plimits",
    ]
    ok = True
    err = ""
    raw = ""
    t0 = time.time()
    try:
        raw = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as exc:
        ok = False
        raw = str(exc.output or "")
        err = f"powermetrics exited with code {exc.returncode}"
    except Exception as exc:
        ok = False
        err = str(exc)
    return {
        "ok": ok,
        "error": err,
        "raw": raw,
        "cmd": cmd,
        "samplers": samplers,
        "duration_s": max(0.0, time.time() - t0),
    }


def _collect_once(sample_ms: int) -> dict[str, Any]:
    started = time.time()
    primary = _run_powermetrics(sample_ms=sample_ms, samplers="thermal,cpu_power")
    ok = bool(primary["ok"])
    err = str(primary["error"] or "")
    raw = str(primary["raw"] or "")
    selected_sampler = str(primary["samplers"])
    selected_cmd = list(primary["cmd"])
    selected_duration = float(primary["duration_s"])
    parsed = _parse_powermetrics_output(raw)

    # On some Apple Silicon models, temperature channels appear under SMC sampler.
    if ok and parsed.get("cpu_temp_c") is None and ("smc" in _supported_samplers()):
        alt = _run_powermetrics(sample_ms=sample_ms, samplers="smc,cpu_power")
        if bool(alt["ok"]):
            alt_raw = str(alt["raw"] or "")
            alt_parsed = _parse_powermetrics_output(alt_raw)
            if any(
                alt_parsed.get(k) is not None
                for k in ("cpu_temp_c", "cpu_speed_limit_pct", "cpu_power_w", "thermal_pressure")
            ):
                raw = alt_raw
                parsed = alt_parsed
                selected_sampler = str(alt["samplers"])
                selected_cmd = list(alt["cmd"])
            selected_duration += float(alt["duration_s"])
        elif not err:
            err = str(alt["error"] or "")

    note = ""
    if not ok:
        lraw = raw.lower()
        if "superuser" in lraw or "root" in lraw:
            note = "powermetrics requires sudo/root privileges."
        elif err:
            note = err
    elif parsed.get("cpu_temp_c") is None:
        if parsed.get("cpu_power_w") is not None or parsed.get("thermal_pressure") is not None:
            note = (
                "No CPU temperature channel exposed; power/thermal telemetry available "
                f"(samplers={selected_sampler})."
            )
        else:
            note = f"No CPU temperature found in powermetrics output (samplers={selected_sampler})."
    else:
        note = f"powermetrics sample captured (samplers={selected_sampler})."

    return {
        "ok": ok,
        "error": (None if ok else err),
        "note": note,
        "timestamp_unix": time.time(),
        "collected_at_unix": started,
        "duration_s": max(0.0, float(selected_duration)),
        "euid": int(getattr(os, "geteuid", lambda: -1)()),
        "samplers": selected_sampler,
        "cmd": " ".join(selected_cmd),
        "raw_excerpt": raw[:12000],
        **parsed,
    }


def _write_cache(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect powermetrics telemetry for dashboard.")
    parser.add_argument("--out-dir", default="outputs/scratch", help="Pipeline output directory.")
    parser.add_argument(
        "--cache-file",
        default=None,
        help="Optional explicit cache file path. Defaults to <out-dir>/cache/powermetrics_telemetry.json",
    )
    parser.add_argument("--interval-sec", type=float, default=5.0, help="Collection interval in seconds.")
    parser.add_argument("--sample-ms", type=int, default=1000, help="powermetrics sample interval in ms.")
    parser.add_argument("--once", action="store_true", help="Collect once and exit.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    cache_file = (
        Path(args.cache_file).resolve()
        if args.cache_file is not None
        else out_dir / "cache" / "powermetrics_telemetry.json"
    )
    interval_s = max(1.0, float(args.interval_sec))
    sample_ms = max(200, int(args.sample_ms))

    print(f"[powermetrics] cache={cache_file}")
    print(f"[powermetrics] interval={interval_s:.1f}s sample={sample_ms}ms")
    if int(getattr(os, "geteuid", lambda: -1)()) != 0:
        print("[powermetrics] warning: not running as root; powermetrics will likely fail.")

    try:
        while True:
            t0 = time.time()
            payload = _collect_once(sample_ms=sample_ms)
            _write_cache(cache_file, payload)
            status = "ok" if payload.get("ok") else "err"
            temp = payload.get("cpu_temp_c")
            speed = payload.get("cpu_speed_limit_pct")
            print(
                f"[powermetrics] {status} temp={temp}C speed_limit={speed}% "
                f"note={payload.get('note')}"
            )
            if args.once:
                break
            dt = time.time() - t0
            time.sleep(max(0.0, interval_s - dt))
    except KeyboardInterrupt:
        print("[powermetrics] stopped by user.")


if __name__ == "__main__":
    main()
