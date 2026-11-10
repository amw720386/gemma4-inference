"""Job bodies executed by the bridge's workers, kept Modal-free so the exact paid code path runs in CPU tests.

``modal_bridge/app.py`` wraps these in Modal functions (adding Volume reload/commit); tests call them directly
with the mock engine, through the same bridge API, so spec-key mismatches surface before any paid call.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

DEADLINE_HEADROOM_S = 120.0


def guarded(fn: Callable[[dict[str, Any]], dict[str, Any]], spec: dict[str, Any],
            clock: Callable[[], float] = time.time) -> dict[str, Any]:
    """Never raise (so poll errors are only platform-level); refuse to start after the reservation deadline."""
    started = clock()
    if spec.get("deadline_unix") and started > float(spec["deadline_unix"]):
        return {"aborted": "started after reservation deadline (restart/preemption?)", "started_unix": started,
                "ended_unix": clock(), "job_id": spec.get("job_id")}
    try:
        out = fn(spec)
    except BaseException as e:  # noqa: BLE001
        out = {"error": repr(e)[:4000]}
    out.update(started_unix=started, ended_unix=clock(), job_id=spec.get("job_id"))
    return out


def run_job(spec: dict[str, Any], results_root: Path, clock: Callable[[], float] = time.time,
            **runner_overrides: Any) -> dict[str, Any]:
    """Map a bridge job spec onto RunOptions and run it. Soft wall budget never exceeds the deadline."""
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
