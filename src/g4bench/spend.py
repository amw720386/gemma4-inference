"""Project spend control for direct Modal runs (replaces the HTTPS bridge's server-side ledger, D-039).

Model: every paid launch is made from this machine by ``g4bench modal …`` (``modal_run.py``). Before a launch the
*worst case* of that launch is computed from the container's resource **limits** and its Modal ``timeout``; the
launch is refused unless

    committed + worst_case <= project cap          (default $10, ``G4_PROJECT_CAP_USD``)
    worst_case <= per-job cap                       (default $2,  ``G4_PER_JOB_CAP_USD``)

where ``committed`` = max(local ledger total, Modal billing for ``g4repro*`` apps) + volume-storage accrual.
The ledger is an append-only JSONL file; a launch writes ``reserved`` (worst case) *before* the Modal call and
``settled`` (wall-clock estimate, counted from before the container started) afterwards. A reservation without a
settle line (crash, Ctrl-C) stays committed at its worst case until ``reconcile`` replaces it with Modal's billed
figure for that app.

What this is not: a Modal billing hard stop. It gates launches made through this tool only; Modal's invoice is
authoritative, which is why ``reconcile`` reads it back.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .paths import RESULTS

# `modal billing rates` (checked 2026-11-10), USD per second.
PRICE = {"cpu_core_s": 0.0473 / 3600, "mem_gib_s": 0.008 / 3600, "volume_gib_month": 0.09,
         "gpu_s": {"RTX-PRO-6000": 3.03 / 3600, "L40S": 1.95 / 3600, "A100-80GB": 2.50 / 3600, "H100": 3.95 / 3600,
                   "H200": 4.54 / 3600, "B200": 6.25 / 3600}}
STARTUP_MARGIN_S = 900.0     # billed container time before our code can check its deadline (image load etc.)
SCALEDOWN_MARGIN_S = 60.0    # idle time after completion (functions use scaledown_window <= this)
APP_PREFIX = "g4repro"       # every Modal app this project launches is named g4repro*; billing is filtered on it
LEDGER = RESULTS / "budget_ledger.jsonl"
DISCLAIMER = ("Conservative local estimate reconciled against Modal billing for g4repro* apps. Not a Modal billing "
              "hard stop; Modal's invoice is authoritative.")


class BudgetError(RuntimeError):
    """Launch refused: it could exceed the project or per-job cap."""


@dataclass(frozen=True)
class Resources:
    cpu_limit: float
    mem_limit_gib: float
    gpu: str | None = None

    def rate(self, gpu: str | None = None) -> float:
        return resource_rate(gpu if gpu is not None else self.gpu, self.cpu_limit, self.mem_limit_gib)


# Must match the cpu/memory *limits* declared in modal_app.py (tested).
GPU_JOB = Resources(cpu_limit=8.0, mem_limit_gib=64.0, gpu="RTX-PRO-6000")
PREFETCH = Resources(cpu_limit=2.0, mem_limit_gib=4.0)
PREFETCH_TIMEOUT_S = 3600
SELFTEST_TIMEOUT_S = 1200


@dataclass(frozen=True)
class Caps:
    project_usd: float = 10.0
    per_job_usd: float = 2.0

    @classmethod
    def from_env(cls) -> "Caps":
        """Environment may only *tighten* the approved $10 / $2 (never relax it)."""
        p = min(10.0, float(os.environ.get("G4_PROJECT_CAP_USD") or 10.0))
        j = min(2.0, p, float(os.environ.get("G4_PER_JOB_CAP_USD") or 2.0))
        return cls(p, j)


def resource_rate(gpu: str | None, cpu_limit: float, mem_limit_gib: float) -> float:
    """USD/s at the container's resource *limits* (Modal bills max(request, usage) <= limit)."""
    g = PRICE["gpu_s"][gpu.rstrip("!")] if gpu else 0.0
    return g + cpu_limit * PRICE["cpu_core_s"] + mem_limit_gib * PRICE["mem_gib_s"]


def worst_case_usd(rate_usd_s: float, timeout_s: float) -> float:
    return rate_usd_s * (timeout_s + STARTUP_MARGIN_S + SCALEDOWN_MARGIN_S)


def timeout_for_budget(rate_usd_s: float, usd: float) -> int:
    """Largest Modal timeout whose worst case fits in ``usd`` (0 if none)."""
    return max(int(math.floor(usd / rate_usd_s - STARTUP_MARGIN_S - SCALEDOWN_MARGIN_S)), 0)


def _utc(t: float | None = None) -> str:
    return dt.datetime.fromtimestamp(time.time() if t is None else t, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ----------------------------------------------------------------------------- ledger

def read(path: Path = LEDGER) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def append(entry: dict[str, Any], path: Path = LEDGER) -> dict[str, Any]:
    entry = {"utc": _utc(), **entry}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(entry, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())
    return entry


def launches(entries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Fold ledger lines into one record per launch id (lines without ``launch_id`` are history only)."""
    out: dict[str, dict[str, Any]] = {}
    for e in entries:
        lid = e.get("launch_id")
        if lid:
            # "event" lines (and pre-2026-11-11 reconcile lines tagged kind=reconcile) never overwrite the launch kind
            upd = {k: v for k, v in e.items() if v is not None and not (k == "kind" and v == "reconcile")}
            out.setdefault(lid, {}).update(upd)
    return out


def launch_cost(rec: dict[str, Any]) -> float:
    """Open: worst case. Settled: the larger of our estimate and Modal's bill. Settled-from-billing: the bill."""
    billed = float(rec.get("billed_usd") or 0.0)
    worst = float(rec.get("worst_usd") or 0.0)
    status = rec.get("status")
    if status == "settled-from-billing":
        return billed
    if status == "settled":
        return max(float(rec.get("settled_usd") if rec.get("settled_usd") is not None else worst), billed)
    return max(worst, billed)  # reserved / unknown: fail safe


def ledger_total(entries: list[dict[str, Any]]) -> float:
    return sum(launch_cost(r) for r in launches(entries).values())


def storage_accrual_usd(entries: list[dict[str, Any]], now: float | None = None) -> float:
    """Volume storage since each ``storage`` line (GiB stored x months) until a ``storage`` line with 0 GiB."""
    now = time.time() if now is None else now
    total, gib, since = 0.0, 0.0, None
    for e in entries:
        if e.get("kind") != "storage":
            continue
        t = float(e["unix"])
        if since is not None:
            total += gib * (t - since) / (30 * 86400) * PRICE["volume_gib_month"]
        gib, since = float(e["gib"]), t
    if since is not None:
        total += gib * (now - since) / (30 * 86400) * PRICE["volume_gib_month"]
    return total


def committed(entries: list[dict[str, Any]], billed_total: float | None = None, now: float | None = None) -> float:
    """What counts against the cap: the larger of our ledger and Modal's bill for g4repro* apps, plus storage."""
    return max(ledger_total(entries), billed_total or 0.0) + storage_accrual_usd(entries, now)


def check_launch(worst_usd: float, caps: Caps, entries: list[dict[str, Any]],
                 billed_total: float | None = None) -> float:
    """Raise BudgetError unless the launch fits; return the remaining budget after it."""
    if worst_usd > caps.per_job_usd + 1e-9:
        raise BudgetError(f"worst case ${worst_usd:.2f} exceeds the per-job cap ${caps.per_job_usd:.2f}")
    open_ = [lid for lid, r in launches(entries).items() if r.get("status") == "reserved"]
    if open_:
        raise BudgetError(f"open reservation(s) {open_}: settle/reconcile before launching another job")
    c = committed(entries, billed_total)
    if c + worst_usd > caps.project_usd + 1e-9:
        raise BudgetError(f"committed ${c:.2f} + worst case ${worst_usd:.2f} exceeds the project cap "
                          f"${caps.project_usd:.2f}")
    return caps.project_usd - c - worst_usd


# ----------------------------------------------------------------------------- Modal billing (authoritative)

def modal_billing(since: str = "2026-11-01") -> list[dict[str, Any]]:
    """Rows of ``modal billing report`` for this project's apps (daily, per app). Needs Modal credentials.

    ``--start`` alone omits the current (partial) UTC day, so ``--for today`` is queried too; rows are de-duplicated
    by (app, day)."""
    rows: dict[tuple, dict[str, Any]] = {}
    for args in (["--start", since], ["--for", "today"]):
        p = subprocess.run([sys.executable, "-m", "modal", "billing", "report", *args, "--json"], capture_output=True,
                           text=True, timeout=120, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        if p.returncode != 0:
            raise RuntimeError(f"modal billing report failed: {(p.stderr or p.stdout)[-500:]}")
        for r in json.loads(p.stdout or "[]"):
            if str(r.get("description", "")).startswith(APP_PREFIX):
                rows[(r["object_id"], r.get("interval_start"))] = r
    return list(rows.values())


def billed_by_app(rows: list[dict[str, Any]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for r in rows:
        out[r["object_id"]] = out.get(r["object_id"], 0.0) + float(r["cost"])
    return out


def reconcile(entries: list[dict[str, Any]], rows: list[dict[str, Any]], path: Path = LEDGER) -> list[dict]:
    """Attach Modal's billed cost to every launch whose app id appears in the billing rows.

    Billing lags by up to a day; an open reservation stays at its worst case until its app is billed AND at least
    ``SCALEDOWN_MARGIN_S`` has passed since its deadline (so a crashed client cannot under-count a live container).
    """
    by_app = billed_by_app(rows)
    written = []
    for lid, r in launches(entries).items():
        app = r.get("app_id")
        if not app or app not in by_app or abs(float(r.get("billed_usd") or -1) - by_app[app]) < 1e-6:
            continue
        upd: dict[str, Any] = {"launch_id": lid, "event": "reconcile", "billed_usd": round(by_app[app], 6)}
        if r.get("status") == "reserved" and time.time() > float(r.get("deadline_unix", 0)) + SCALEDOWN_MARGIN_S:
            upd["status"] = "settled-from-billing"
        written.append(append(upd, path))
    return written


def summary(caps: Caps, entries: list[dict[str, Any]], billed_rows: list[dict[str, Any]] | None) -> dict[str, Any]:
    billed = sum(float(r["cost"]) for r in billed_rows) if billed_rows is not None else None
    c = committed(entries, billed)
    return {"project_cap_usd": caps.project_usd, "per_job_cap_usd": caps.per_job_usd,
            "ledger_usd": round(ledger_total(entries), 4), "modal_billed_usd": None if billed is None else round(billed, 4),
            "storage_accrual_usd": round(storage_accrual_usd(entries), 4), "committed_usd": round(c, 4),
            "available_usd": round(caps.project_usd - c, 4),
            "open_reservations": [k for k, r in launches(entries).items() if r.get("status") == "reserved"],
            "note": DISCLAIMER}
