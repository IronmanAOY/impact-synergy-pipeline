#!/usr/bin/env python3
"""
IMPaCT Desktop Launcher

One-click desktop launcher for non-developers:
- Starts/stops the live dashboard backend
- Opens the browser automatically once the server is reachable
- Keeps a visible runtime log for troubleshooting
"""

from __future__ import annotations

import os
import queue
import signal
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path
from tkinter import BOTH, END, LEFT, RIGHT, TOP, Button, Entry, Frame, Label, StringVar, Tk
from tkinter.scrolledtext import ScrolledText


REPO_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_SCRIPT = REPO_ROOT / "scripts" / "live_dashboard.py"


class DesktopLauncher:
    def __init__(self) -> None:
        self.root = Tk()
        self.root.title("IMPaCT Desktop")
        self.root.geometry("980x640")

        self.dataset_id = StringVar(value="ds003171")
        self.out_dir = StringVar(value="outputs/scratch")
        self.host = StringVar(value="127.0.0.1")
        self.port = StringVar(value="8765")
        self.status = StringVar(value="Idle")

        self.proc: subprocess.Popen[str] | None = None
        self.proc_group: int | None = None
        self.log_q: queue.Queue[str] = queue.Queue()

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(150, self._poll_log_queue)

    def _build_ui(self) -> None:
        top = Frame(self.root)
        top.pack(side=TOP, fill="x", padx=10, pady=10)

        row1 = Frame(top)
        row1.pack(fill="x")
        Label(row1, text="Dataset ID", width=14, anchor="w").pack(side=LEFT)
        Entry(row1, textvariable=self.dataset_id, width=18).pack(side=LEFT, padx=(0, 10))
        Label(row1, text="Output Dir", width=14, anchor="w").pack(side=LEFT)
        Entry(row1, textvariable=self.out_dir, width=42).pack(side=LEFT, padx=(0, 10))

        row2 = Frame(top)
        row2.pack(fill="x", pady=(8, 0))
        Label(row2, text="Host", width=14, anchor="w").pack(side=LEFT)
        Entry(row2, textvariable=self.host, width=18).pack(side=LEFT, padx=(0, 10))
        Label(row2, text="Port", width=14, anchor="w").pack(side=LEFT)
        Entry(row2, textvariable=self.port, width=18).pack(side=LEFT, padx=(0, 10))

        controls = Frame(top)
        controls.pack(fill="x", pady=(10, 0))
        Button(controls, text="Start", command=self.start, width=14).pack(side=LEFT)
        Button(controls, text="Stop", command=self.stop, width=14).pack(side=LEFT, padx=(8, 0))
        Button(controls, text="Restart", command=self.restart, width=14).pack(side=LEFT, padx=(8, 0))
        Button(controls, text="Open Dashboard", command=self.open_browser, width=16).pack(side=LEFT, padx=(8, 0))

        status_bar = Frame(self.root)
        status_bar.pack(fill="x", padx=10)
        Label(status_bar, text="Status:", width=8, anchor="w").pack(side=LEFT)
        Label(status_bar, textvariable=self.status, anchor="w").pack(side=LEFT)

        self.log_view = ScrolledText(self.root, wrap="word", font=("Menlo", 11))
        self.log_view.pack(fill=BOTH, expand=True, padx=10, pady=10)
        self._log("IMPaCT Desktop ready. Click Start to launch the dashboard.")

    def _dashboard_url(self) -> str:
        return f"http://{self.host.get().strip()}:{self.port.get().strip()}"

    def _log(self, text: str) -> None:
        ts = time.strftime("%H:%M:%S", time.localtime())
        self.log_view.insert(END, f"[{ts}] {text}\n")
        self.log_view.see(END)

    def _poll_log_queue(self) -> None:
        while True:
            try:
                line = self.log_q.get_nowait()
            except queue.Empty:
                break
            self._log(line.rstrip("\n"))
        self.root.after(150, self._poll_log_queue)

    def _stream_proc(self, proc: subprocess.Popen[str]) -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            self.log_q.put(line)
        rc = proc.wait()
        self.log_q.put(f"Dashboard process exited with code {rc}")
        self.status.set("Stopped")
        self.proc = None
        self.proc_group = None

    def _wait_and_open(self) -> None:
        url = self._dashboard_url()
        deadline = time.time() + 45
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=1.2) as resp:
                    if int(resp.status) == 200:
                        webbrowser.open(url)
                        self.log_q.put(f"Dashboard opened in browser: {url}")
                        return
            except Exception:
                time.sleep(0.5)
        self.log_q.put(f"Dashboard did not respond at {url} within timeout.")

    def start(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.status.set("Already running")
            return

        dataset = self.dataset_id.get().strip() or "ds003171"
        out_dir = self.out_dir.get().strip() or "outputs/scratch"
        host = self.host.get().strip() or "127.0.0.1"
        port = self.port.get().strip() or "8765"

        if not DASHBOARD_SCRIPT.exists():
            self.status.set("Error")
            self._log(f"Missing dashboard script: {DASHBOARD_SCRIPT}")
            return

        cmd = [
            sys.executable,
            str(DASHBOARD_SCRIPT),
            "--dataset-id",
            dataset,
            "--out-dir",
            out_dir,
            "--host",
            host,
            "--port",
            str(port),
        ]

        kwargs: dict[str, object] = {
            "cwd": str(REPO_ROOT),
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
            "text": True,
            "bufsize": 1,
        }

        if os.name == "posix":
            kwargs["preexec_fn"] = os.setsid
        elif os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]

        try:
            proc = subprocess.Popen(cmd, **kwargs)
        except Exception as exc:
            self.status.set("Error")
            self._log(f"Failed to start dashboard: {exc}")
            return

        self.proc = proc
        if os.name == "posix":
            try:
                self.proc_group = os.getpgid(proc.pid)
            except Exception:
                self.proc_group = None
        self.status.set("Running")
        self._log("Dashboard process started.")

        t = threading.Thread(target=self._stream_proc, args=(proc,), daemon=True)
        t.start()

        t_open = threading.Thread(target=self._wait_and_open, daemon=True)
        t_open.start()

    def stop(self) -> None:
        proc = self.proc
        if proc is None or proc.poll() is not None:
            self.status.set("Stopped")
            self._log("No running dashboard process.")
            self.proc = None
            self.proc_group = None
            return

        self._log("Stopping dashboard process...")
        try:
            if os.name == "posix" and self.proc_group is not None:
                os.killpg(self.proc_group, signal.SIGTERM)
            else:
                proc.terminate()
            proc.wait(timeout=8)
        except Exception:
            try:
                if os.name == "posix" and self.proc_group is not None:
                    os.killpg(self.proc_group, signal.SIGKILL)
                else:
                    proc.kill()
            except Exception:
                pass

        self.status.set("Stopped")
        self.proc = None
        self.proc_group = None
        self._log("Dashboard process stopped.")

    def restart(self) -> None:
        self.stop()
        time.sleep(0.4)
        self.start()

    def open_browser(self) -> None:
        url = self._dashboard_url()
        webbrowser.open(url)
        self._log(f"Browser opened at {url}")

    def _on_close(self) -> None:
        try:
            self.stop()
        finally:
            self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    app = DesktopLauncher()
    app.run()


if __name__ == "__main__":
    main()
