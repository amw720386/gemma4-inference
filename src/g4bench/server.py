"""Start / health-check / stop an inference server subprocess, and sample GPU telemetry."""

from __future__ import annotations

import csv
import os
import shutil
import signal
import subprocess
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any


class ServerError(RuntimeError):
    pass


class Server:
    """A server in its own process group so the whole tree can be killed reliably."""

    def __init__(self, argv: list[str], *, env: dict[str, str], log_path: Path, base_url: str,
                 health_path: str = "/health"):
        self.argv, self.env, self.log_path = argv, env, log_path
        self.base_url, self.health_path = base_url.rstrip("/"), health_path
        self.proc: subprocess.Popen | None = None
        self._log = None
        self.start_s: float | None = None

    def start(self, timeout_s: float = 1800.0, poll_s: float = 2.0) -> float:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = open(self.log_path, "w", encoding="utf-8")
        self._log.write("# argv: " + " ".join(self.argv) + "\n")
        self._log.flush()
        t0 = time.monotonic()
        self.proc = subprocess.Popen(self.argv, env=self.env, stdout=self._log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        while True:
            if self.proc.poll() is not None:
                raise ServerError(f"server exited early with code {self.proc.returncode}; see {self.log_path}")
            if self.healthy():
                self.start_s = time.monotonic() - t0
                return self.start_s
            if time.monotonic() - t0 > timeout_s:
                self.stop()
                raise ServerError(f"server not healthy after {timeout_s:.0f}s; see {self.log_path}")
            time.sleep(poll_s)

    def healthy(self) -> bool:
        try:
            with urllib.request.urlopen(self.base_url + self.health_path, timeout=5) as r:
                if r.status != 200:
                    return False
            with urllib.request.urlopen(self.base_url + "/v1/models", timeout=5) as r:
                return r.status == 200
        except Exception:  # noqa: BLE001
            return False

    def stop(self, grace_s: float = 30.0) -> None:
        if self.proc and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
                self.proc.wait(timeout=grace_s)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.proc.wait(timeout=10)
        if self._log:
            self._log.close()
            self._log = None

    def __enter__(self) -> "Server":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()


class GpuSampler(threading.Thread):
    """Poll nvidia-smi every ``interval`` seconds into a CSV; summarize peak mem / mean power."""

    FIELDS = "timestamp,index,memory.used,power.draw,clocks.sm,clocks.mem,utilization.gpu,temperature.gpu"

    def __init__(self, path: Path, interval: float = 1.0):
        super().__init__(daemon=True)
        self.path, self.interval = path, interval
        self._halt = threading.Event()  # NOT _stop: shadows Thread._stop() on Python <= 3.12
        self.available = shutil.which("nvidia-smi") is not None

    def run(self) -> None:
        if not self.available:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            f.write(self.FIELDS + "\n")
            while not self._halt.is_set():
                try:
                    out = subprocess.run(["nvidia-smi", f"--query-gpu={self.FIELDS}", "--format=csv,noheader,nounits"],
                                         capture_output=True, text=True, timeout=10).stdout
                    f.write(out)
                    f.flush()
                except (OSError, subprocess.TimeoutExpired):
                    pass
                self._halt.wait(self.interval)

    def stop(self) -> dict[str, float | None]:
        self._halt.set()
        self.join(timeout=15)
        return summarize_gpu_csv(self.path)


def summarize_gpu_csv(path: Path) -> dict[str, float | None]:
    out: dict[str, float | None] = {"gpu_mem_peak_mib": None, "gpu_power_mean_w": None, "gpu_sm_clock_mean_mhz": None}
    if not path.exists():
        return out
    mem, pw, clk = [], [], []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f, skipinitialspace=True):
            try:
                mem.append(float(row["memory.used"]))
                pw.append(float(row["power.draw"]))
                clk.append(float(row["clocks.sm"]))
            except (KeyError, TypeError, ValueError):
                continue
    if mem:
        out["gpu_mem_peak_mib"] = max(mem)
    if pw:
        out["gpu_power_mean_w"] = sum(pw) / len(pw)
    if clk:
        out["gpu_sm_clock_mean_mhz"] = sum(clk) / len(clk)
    return out
