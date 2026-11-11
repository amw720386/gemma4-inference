"""Job bodies executed inside the Modal workers, kept Modal-free so the exact paid code path runs in CPU tests.

``modal_app.py`` wraps these in Modal functions (adding Volume reload/commit); tests call them directly with the
mock engine, so spec-key mismatches surface before any paid call.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

DEADLINE_HEADROOM_S = 120.0


def guarded(fn: Callable[[dict[str, Any]], dict[str, Any]], spec: dict[str, Any],
            clock: Callable[[], float] = time.time, *, hard_kill: bool = False) -> dict[str, Any]:
    """Run a worker body under the reservation's time bounds.

    * refuse to start after ``start_deadline_unix`` (= reserved_at + startup margin), so start + Modal timeout can
      never pass the reservation's ``deadline_unix`` - this is what makes worst-case settlement a true bound;
    * ordinary exceptions are returned as ``{"error": ...}`` (a finished call with an error payload); cancellation /
      interrupts (BaseException) propagate so Modal can terminate the input;
    * ``hard_kill``: arm a timer that ``os._exit``s the process at ``deadline_unix`` (production only).
    """
    started = clock()
    start_by = spec.get("start_deadline_unix", spec.get("deadline_unix"))
    if start_by and started > float(start_by):
        return {"aborted": "started after reservation start deadline (queue delay/restart/preemption?)",
                "started_unix": started, "ended_unix": clock(), "job_id": spec.get("job_id")}
    timer = None
    if hard_kill and spec.get("deadline_unix"):
        import os
        import threading

        timer = threading.Timer(max(float(spec["deadline_unix"]) - time.time(), 0.0), lambda: os._exit(70))
        timer.daemon = True
        timer.start()
    try:
        out = fn(spec)
    except Exception as e:  # noqa: BLE001
        out = {"error": repr(e)[:4000]}
    finally:
        if timer is not None:
            timer.cancel()
    out.update(started_unix=started, ended_unix=clock(), job_id=spec.get("job_id"))
    return out


def run_job(spec: dict[str, Any], results_root: Path, clock: Callable[[], float] = time.time,
            **runner_overrides: Any) -> dict[str, Any]:
    """Map a job spec onto RunOptions and run it. Soft wall budget never exceeds the deadline."""
    from .runner import RunOptions, run

    remaining = float(spec["deadline_unix"]) - clock() - DEADLINE_HEADROOM_S
    opts = RunOptions(engine=spec["engine"], matrix=spec["matrix"], seeds=spec.get("seeds"),
                      trials=spec.get("trials"), only_keys=spec.get("only"), results_root=Path(results_root),
                      max_wall_s=max(min(float(spec["max_wall_s"]), remaining), 0.0),
                      notes=spec.get("notes", ""), dry_run=bool(spec.get("dry_run")), **runner_overrides)
    run_dir = run(opts)
    rows = []
    f = run_dir / "results.jsonl"
    if f.exists():
        rows = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
    return {"run_id": run_dir.name, "n_points": len(rows), "n_valid": sum(1 for r in rows if r.get("valid")),
            "rows": rows}


MAX_FETCH_BYTES = 20_000_000


def collect_run_files(runs_root: Path, run_id: str) -> dict[str, bytes]:
    """Every regular file of one run (relative POSIX path -> bytes) so the worker can return it to the caller.

    Files over ``MAX_FETCH_BYTES`` are replaced by a short note (they stay on the results Volume)."""
    root = (Path(runs_root) / run_id).resolve()
    if Path(runs_root).resolve() not in root.parents or not root.is_dir():
        return {}
    out: dict[str, bytes] = {}
    for f in sorted(root.rglob("*")):
        if f.is_file():
            rel = f.relative_to(root).as_posix()
            size = f.stat().st_size
            if size > MAX_FETCH_BYTES:
                out[rel + ".SKIPPED.txt"] = f"{size} bytes; on Modal volume g4repro-results".encode()
            else:
                out[rel] = f.read_bytes()
    return out


def write_run_files(dest_root: Path, run_id: str, files: dict[str, bytes]) -> Path:
    """Write files returned by a worker under ``dest_root/run_id`` (refuses traversal and overwrites)."""
    root = (Path(dest_root) / run_id).resolve()
    if Path(dest_root).resolve() not in root.parents:
        raise ValueError(f"bad run id {run_id!r}")
    for rel, data in files.items():
        f = (root / rel).resolve()
        if root not in f.parents:
            raise ValueError(f"path escapes run dir: {rel!r}")
        if f.exists():
            raise FileExistsError(f)
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(data)
    return root
