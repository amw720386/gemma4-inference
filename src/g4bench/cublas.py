"""Per-config cuBLAS selection and a record of which cuBLAS the server process actually mapped (D-061).

A config asks for a cuBLAS build with ``env: {G4_CUBLAS: "13.10"}``. Resolution, in order:

1. the image side-installed that version (``G4_CUBLAS_SIDE="13.10=/opt/cublas-13.10"``, experimental image): the server
   gets ``LD_PRELOAD=<dir>/.../libcublasLt.so.13:<dir>/.../libcublas.so.13`` and that lib dir first on
   ``LD_LIBRARY_PATH``. Preloaded objects come first in the global symbol scope, so torch's ``DT_NEEDED`` on
   ``libcublas.so.13`` binds to them (the image's own copy may still be mapped if torch dlopens it by full path, but
   its symbols are shadowed);
2. the image was built with that version upgraded in place (``G4_IMAGE_PIP`` lists ``nvidia-cublas==<version>.*``,
   production patched image): nothing to do;
3. otherwise the run is refused before any server starts.

No ``G4_CUBLAS``: the image's default cuBLAS (stock 13.1 on the stock and experimental images).

Evidence: after the server is healthy, ``mapped_libs`` scans ``/proc/<pid>/maps`` of the server and all of its
descendants (vLLM's EngineCore is a child process) for ``libcublas*`` and ``verify`` checks the request was honoured.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

ENV_REQUEST = "G4_CUBLAS"
ENV_SIDE = "G4_CUBLAS_SIDE"
LIBS = ("libcublasLt.so.13", "libcublas.so.13")  # preload order: Lt first (libcublas depends on it)


class CublasError(RuntimeError):
    pass


def side_dirs(environ: Mapping[str, str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in (environ.get(ENV_SIDE) or "").split(","):
        ver, sep, d = item.strip().partition("=")
        if sep and ver and d:
            out[ver.strip()] = d.strip()
    return out


def in_image_version(environ: Mapping[str, str]) -> str | None:
    for req in (environ.get("G4_IMAGE_PIP") or "").split(","):
        name, sep, ver = req.strip().partition("==")
        if sep and name.strip() == "nvidia-cublas":
            return ver.strip()
    return None


def _version_matches(have: str, want: str) -> bool:
    return have == want or have.startswith(want + ".")


def find_libs(root: Path) -> dict[str, str]:
    """``{lib name: path}`` for LIBS under a pip ``--target`` tree (layout differs across wheel generations)."""
    out: dict[str, str] = {}
    for name in LIBS:
        hits = sorted(p for p in Path(root).rglob(name) if p.is_file())
        if hits:
            out[name] = str(hits[0])
    return out


def resolve(requested: str | None, environ: Mapping[str, str] | None = None, *, strict: bool = True) -> dict[str, Any]:
    """How to give the server cuBLAS ``requested``: ``{"requested", "mode", "env", ...}``; ``env`` goes to the server.

    ``strict=False`` (dry runs on a laptop) records an unresolvable request instead of raising."""
    environ = os.environ if environ is None else environ
    if not requested:
        return {"requested": None, "mode": "image-default", "env": {}}
    requested = str(requested).strip()
    side = side_dirs(environ)
    if requested in side:
        d = Path(side[requested])
        libs = find_libs(d) if d.is_dir() else {}
        missing = [n for n in LIBS if n not in libs]
        if missing:
            if not strict:
                return {"requested": requested, "mode": "unresolved", "env": {},
                        "error": f"{missing} not found under {d}"}
            raise CublasError(f"G4_CUBLAS={requested}: {missing} not found under side install {d}")
        libdir = str(Path(libs[LIBS[1]]).parent)
        pre = ":".join(libs[n] for n in LIBS)
        old_pre, old_lp = environ.get("LD_PRELOAD", ""), environ.get("LD_LIBRARY_PATH", "")
        env = {"LD_PRELOAD": pre + (":" + old_pre if old_pre else ""),
               "LD_LIBRARY_PATH": libdir + (":" + old_lp if old_lp else "")}
        return {"requested": requested, "mode": "side", "dir": str(d), "libs": libs, "env": env}
    have = in_image_version(environ)
    if have and _version_matches(have, requested):
        return {"requested": requested, "mode": "in-image", "pip": have, "env": {}}
    msg = (f"G4_CUBLAS={requested} is neither side-installed ({ENV_SIDE}={environ.get(ENV_SIDE)!r}) nor the image's "
           f"in-place upgrade (G4_IMAGE_PIP={environ.get('G4_IMAGE_PIP')!r})")
    if not strict:
        return {"requested": requested, "mode": "unresolved", "env": {}, "error": msg}
    raise CublasError(msg)


# ----------------------------------------------------------------------------------------------- evidence
def process_tree(root_pid: int, proc: Path = Path("/proc")) -> list[int]:
    """``root_pid`` and all its descendants, from ``/proc/*/stat`` (empty if /proc is unavailable)."""
    children: dict[int, list[int]] = {}
    try:
        entries = list(proc.iterdir())
    except OSError:
        return []
    for d in entries:
        if not d.name.isdigit():
            continue
        try:
            stat = (d / "stat").read_text(encoding="utf-8", errors="replace")
            ppid = int(stat.rsplit(")", 1)[1].split()[1])  # comm may contain spaces/parens
        except (OSError, IndexError, ValueError):
            continue
        children.setdefault(ppid, []).append(int(d.name))
    if not (proc / str(root_pid)).exists():
        return []
    out, todo = [], [int(root_pid)]
    while todo:
        pid = todo.pop()
        if pid in out:
            continue
        out.append(pid)
        todo.extend(children.get(pid, []))
    return sorted(out)


def mapped_libs(root_pid: int, needle: str = "libcublas", proc: Path = Path("/proc")) -> dict[str, Any]:
    """Shared objects whose path contains ``needle`` mapped by the server process tree."""
    pids = process_tree(root_pid, proc)
    if not pids:
        return {"available": False, "pids": [], "paths": [], "by_pid": {}}
    by_pid: dict[str, list[str]] = {}
    for pid in pids:
        try:
            lines = (proc / str(pid) / "maps").read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        paths = sorted({ln.split(None, 5)[5] for ln in lines if len(ln.split(None, 5)) == 6
                        and needle in ln.split(None, 5)[5]})
        if paths:
            by_pid[str(pid)] = paths
    allp = sorted({p for v in by_pid.values() for p in v})
    return {"available": True, "pids": pids, "paths": allp, "by_pid": by_pid,
            "realpaths": sorted({os.path.realpath(p) for p in allp})}


def verify(resolved: dict[str, Any], probe: dict[str, Any], environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Was the request honoured? ``ok`` True/False, or None when nothing could be observed."""
    environ = os.environ if environ is None else environ
    if not probe.get("available") or not probe.get("paths"):
        return {"ok": None, "why": "no libcublas mapping observed (no /proc, or cuBLAS not loaded)"}
    paths = probe["paths"]
    side = list(side_dirs(environ).values())
    if resolved.get("mode") == "side":
        d = resolved["dir"].rstrip("/") + "/"
        ok = any(p.startswith(d) for p in paths)
        return {"ok": ok, "why": f"side install {d} {'mapped' if ok else 'NOT mapped'}",
                "also_mapped": [p for p in paths if not p.startswith(d)]}
    foreign = [p for p in paths if any(p.startswith(s.rstrip("/") + "/") for s in side)]
    return {"ok": not foreign, "why": "image cuBLAS only" if not foreign else f"side cuBLAS mapped: {foreign}"}
