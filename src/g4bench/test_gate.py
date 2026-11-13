"""Unit-test policy (owner request, 2026-11-13): the test-suite runs only on Modal, at most once every 2.5 hours.

* ``g4bench modal selftest --execute`` is the only supported way to run the suite (Linux, inside Modal, ~1 cent).
  Before launching it, :func:`check_selftest_interval` refuses if the previous selftest launch recorded in the spend
  ledger started less than :data:`SELFTEST_MIN_INTERVAL_S` ago.
* ``tests/conftest.py`` stops a local ``pytest`` run unless it executes inside a Modal container (``MODAL_TASK_ID``
  is set there) or the owner explicitly sets ``G4_ALLOW_LOCAL_TESTS=1``.
* Between selftests, check edits with ``python -m py_compile`` and small targeted imports, not pytest.
"""

from __future__ import annotations

import datetime as dt
import time
from typing import Any

SELFTEST_MIN_INTERVAL_S = 2.5 * 3600
LOCAL_OVERRIDE_ENV = "G4_ALLOW_LOCAL_TESTS"


class TestGateError(RuntimeError):
    """A selftest was requested before the minimum interval elapsed."""


def last_selftest_unix(entries: list[dict[str, Any]]) -> float | None:
    """Start time of the most recent selftest launch in the spend ledger (failed or not)."""
    times = [float(e["reserved_unix"]) for e in entries
             if e.get("kind") == "selftest" and e.get("reserved_unix") is not None]
    return max(times) if times else None


def next_allowed_unix(entries: list[dict[str, Any]]) -> float:
    last = last_selftest_unix(entries)
    return 0.0 if last is None else last + SELFTEST_MIN_INTERVAL_S


def check_selftest_interval(entries: list[dict[str, Any]], now: float | None = None) -> None:
    """Raise TestGateError unless at least 2.5 h have passed since the last selftest launch."""
    now = time.time() if now is None else now
    nxt = next_allowed_unix(entries)
    if now < nxt:
        when = dt.datetime.fromtimestamp(nxt, dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        raise TestGateError(f"unit tests ran less than 2.5 h ago; next selftest allowed at {when} "
                            f"({(nxt - now) / 60:.0f} min from now)")


def running_inside_modal(environ: dict[str, str]) -> bool:
    return bool(environ.get("MODAL_TASK_ID"))


def local_tests_allowed(environ: dict[str, str]) -> bool:
    return running_inside_modal(environ) or environ.get(LOCAL_OVERRIDE_ENV) == "1"
