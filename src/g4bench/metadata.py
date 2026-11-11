"""Capture environment and hardware metadata for every run.

Everything is best-effort: missing tools produce ``None`` plus an entry in
``_errors`` rather than an exception, so CPU-only runs still record what they can.
No environment variable *values* are captured except an explicit allowlist, so
secrets cannot leak into results.
"""

from __future__ import annotations

import datetime as _dt
import os
import platform
import shutil
import subprocess
import sys
from typing import Any

from .paths import REPO

ENV_ALLOWLIST = (
    "CUDA_VISIBLE_DEVICES", "NVIDIA_DRIVER_CAPABILITIES", "VLLM_USE_V1", "VLLM_ATTENTION_BACKEND",
    "SGLANG_VERSION", "HF_HUB_OFFLINE", "MODAL_REGION", "MODAL_CLOUD_PROVIDER", "MODAL_TASK_ID",
    "G4_IMAGE_PATCHES", "G4_IMAGE_PIP", "G4_SPLITK_T_FULL", "G4_SPLITK_T_SLIDING", "G4_DRAFT_HEAD_Q",
    "G4_DRAFT_HEAD_K",  # build-time engine patches baked into the image (src/g4bench/patches)
)

NVSMI_FIELDS = (
    "name,uuid,pci.bus_id,driver_version,memory.total,power.limit,power.default_limit,power.max_limit,"
    "clocks.max.sm,clocks.max.memory,clocks.sm,clocks.mem,pcie.link.gen.current,pcie.link.width.current,"
    "temperature.gpu,persistence_mode,compute_mode,ecc.mode.current,vbios_version"
)


def _run(cmd: list[str], timeout: float = 30) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except (OSError, subprocess.TimeoutExpired) as e:
        return -1, repr(e)


def gpu_info() -> tuple[list[dict[str, str]], str | None]:
    if not shutil.which("nvidia-smi"):
        return [], "nvidia-smi not found"
    rc, out = _run(["nvidia-smi", f"--query-gpu={NVSMI_FIELDS}", "--format=csv,noheader,nounits"])
    if rc != 0:
        return [], out.strip()[:500]
    keys = NVSMI_FIELDS.split(",")
    gpus = []
    for line in out.strip().splitlines():
        vals = [v.strip() for v in line.split(",")]
        gpus.append(dict(zip(keys, vals)))
    return gpus, None


def hardware_class(gpus: list[dict[str, str]]) -> str:
    """Coarse class used for grouping, e.g. 'RTX-PRO-6000x1' or 'cpu'."""
    if not gpus:
        return "cpu"
    name = gpus[0].get("name", "unknown")
    tag = name.replace("NVIDIA ", "").replace(" Blackwell", "").replace(" Server Edition", "-SE")
    tag = tag.replace(" Workstation Edition", "-WS").replace(" Max-Q", "-MaxQ").replace(" ", "-")
    return f"{tag}x{len(gpus)}"


def is_exact_hardware(gpus: list[dict[str, str]]) -> bool:
    """Article hardware: exactly one RTX PRO 6000 Blackwell with ~96 GB."""
    if len(gpus) != 1:
        return False
    g = gpus[0]
    try:
        mem_mib = float(g.get("memory.total", "0"))
    except ValueError:
        mem_mib = 0.0
    return "RTX PRO 6000" in g.get("name", "") and mem_mib > 90_000


def git_info() -> dict[str, Any]:
    rc, sha = _run(["git", "-C", str(REPO), "rev-parse", "HEAD"])
    rc2, st = _run(["git", "-C", str(REPO), "status", "--porcelain"])
    if rc != 0 and os.environ.get("G4_CODE_SHA", "unknown") != "unknown":
        # Inside Modal the repo is uploaded without .git; the launcher passes "<sha>[-dirty]" instead.
        env_sha = os.environ["G4_CODE_SHA"]
        return {"sha": env_sha.removesuffix("-dirty"), "dirty": env_sha.endswith("-dirty"), "source": "G4_CODE_SHA"}
    return {"sha": sha.strip() if rc == 0 else None, "dirty": bool(st.strip()) if rc2 == 0 else None}


def pip_versions(pkgs: tuple[str, ...] = ("vllm", "sglang", "torch", "transformers", "flashinfer-python",
                                          "flash-attn", "triton", "sgl-kernel", "numpy", "aiohttp")) -> dict[str, str | None]:
    from importlib import metadata as md

    out: dict[str, str | None] = {}
    for p in pkgs:
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            out[p] = None
    return out


def capture(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    gpus, gpu_err = gpu_info()
    errors = {}
    if gpu_err:
        errors["gpu"] = gpu_err
    rc, cuda = _run(["nvcc", "--version"]) if shutil.which("nvcc") else (-1, "")
    meta: dict[str, Any] = {
        "captured_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "host": {"platform": platform.platform(), "machine": platform.machine(),
                 "python": sys.version.split()[0], "cpu_count": os.cpu_count(),
                 "cpu_model": _cpu_model(), "mem_total_kib": _mem_total_kib()},
        "gpus": gpus,
        "hardware_class": hardware_class(gpus),
        "exact_hardware": is_exact_hardware(gpus),
        "nvcc": cuda.strip().splitlines()[-1] if rc == 0 and cuda.strip() else None,
        "packages": pip_versions(),
        "git": git_info(),
        "env": {k: os.environ.get(k) for k in ENV_ALLOWLIST if k in os.environ},
        "_errors": errors,
    }
    if extra:
        meta.update(extra)
    return meta


def _cpu_model() -> str | None:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return None


def _mem_total_kib() -> int | None:
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    return int(line.split()[1])
    except OSError:
        pass
    return None
