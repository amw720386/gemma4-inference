"""Project-wide spend ledger for the Modal bridge (pure logic; storage injected).

What this is: a *conservative estimate* ledger that the bridge consults before launching anything paid.
What this is NOT: a Modal billing hard stop. Modal's invoice is authoritative; this ledger can only refuse
to launch new work through the bridge. It cannot stop spend from anything launched outside the bridge.

Invariants (each tested in tests/test_budget.py):

1. **Reserve before launch.** A job may start only after a reservation of its *worst-case* cost is written.
   Worst case = (cpu+memory+gpu rate at the container's resource *limits*) x (Modal timeout +
   startup margin + scaledown margin). Modal kills the input at ``timeout``, so this is an upper bound
   per attempt; preemption restarts are bounded by the job's ``deadline_unix`` check (see app.py) and
   are still counted, because settlement uses observed wall time, uncapped.
2. **Committed = platform reserve + open reservations + settled estimates <= cap**, checked atomically
   with the write that adds a reservation. Violations raise; nothing is launched.
3. **Release only what provably never ran.** A reservation is released only when the spawn call itself
   failed (no call id). Anything with a call id stays fully reserved until it is confirmed terminal.
4. **Settle conservatively.** A job that finished and reported its own end: cost = rate x (end - reserved_at +
   scaledown margin) - counted from before container start, never capped. Any other terminal outcome
   (timeout, crash, cancel, expired) settles at the worst case ``latest = start_deadline + timeout``: workers
   refuse to start after ``start_deadline = reserved_at + startup margin``, so Modal's timeout ends them by then.
   A reservation that never obtained a call id (ambiguous spawn, lost attach) is settled at its full reservation
   once ``latest`` has passed ("orphan").
7. **The bridge's own spend is metered.** Every ledger op is a bridge request; warm time of the bridge + ledger
   containers is accrued (gaps > warm window count as one window) and committed as
   ``max(platform reserve, metered platform cost)``, so a forgotten poll loop eats into the cap instead of
   escaping it.
8. **Compare-and-swap writes.** Besides the lease lock, each save claims ``ledger/commit/<version>`` with atomic
   put-if-absent; losing that race aborts the op (no lost updates even if two writers overlap).
5. **Serialized, restart-safe writes.** The whole ledger lives under one key and is rewritten in one put.
   Mutations hold a lease lock acquired with atomic put-if-absent. In app.py every mutation additionally
   runs inside a single-container, one-input-at-a-time Modal function, so the only way a lease goes stale
   is its holder crashing; stealing a stale lease is then safe. State is re-read on every operation, so a
   bridge/ledger restart loses nothing.
6. **Fail closed.** Missing, corrupt or unreadable state => refuse to reserve.
"""

from __future__ import annotations

import contextlib
import math
import secrets
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, Protocol

LEDGER_KEY = "ledger/v1"
LOCK_KEY = "ledger/lock"
COMMIT_PREFIX = "ledger/commit/"
DISCLAIMER = ("Conservative estimate maintained by the g4repro bridge. NOT a Modal billing hard stop and not "
              "Modal's invoice; it only gates launches made through this bridge.")

# modal.com/pricing (retrieved 2026-11-10), USD per second.
PRICE = {"cpu_core_s": 0.0000131, "mem_gib_s": 0.00000222,
         "gpu_s": {"RTX-PRO-6000": 0.000842, "L40S": 0.000542, "A100-80GB": 0.000694, "H100": 0.001097,
                   "H200": 0.001261, "B200": 0.001736}}
STARTUP_MARGIN_S = 900.0     # billed container time before our code can check the deadline (image load etc.)
SCALEDOWN_MARGIN_S = 60.0    # idle time after completion (functions use scaledown_window <= this)


class BudgetError(RuntimeError):
    """Reservation refused because it would exceed the cap (HTTP 402)."""


class ConcurrencyError(RuntimeError):
    """Reservation refused because too many jobs are active (HTTP 409)."""


class LedgerStateError(RuntimeError):
    """Ledger unreadable/corrupt or an illegal transition was requested (fail closed)."""


class KV(Protocol):
    def get(self, key: Any, default: Any = None) -> Any: ...
    def put(self, key: Any, value: Any, *, skip_if_exists: bool = False) -> bool: ...
    def pop(self, key: Any, default: Any = ...) -> Any: ...


class MemoryKV:
    """Thread-safe in-memory stand-in for modal.Dict with atomic put-if-absent (tests only)."""

    def __init__(self) -> None:
        self._d: dict[Any, Any] = {}
        self._m = threading.Lock()

    def get(self, key: Any, default: Any = None) -> Any:
        with self._m:
            v = self._d.get(key, default)
            return _copy(v)

    def put(self, key: Any, value: Any, *, skip_if_exists: bool = False) -> bool:
        with self._m:
            if skip_if_exists and key in self._d:
                return False
            self._d[key] = _copy(value)
            return True

    def pop(self, key: Any, default: Any = None) -> Any:
        with self._m:
            return self._d.pop(key, default)


def _copy(v: Any) -> Any:
    import copy

    return copy.deepcopy(v)


def resource_rate(gpu: str | None, cpu_limit: float, mem_limit_gib: float) -> float:
    """USD/s at the container's resource *limits* (Modal bills max(request, usage) <= limit)."""
    g = PRICE["gpu_s"][gpu.rstrip("!")] if gpu else 0.0
    return g + cpu_limit * PRICE["cpu_core_s"] + mem_limit_gib * PRICE["mem_gib_s"]


def worst_case_usd(rate_usd_s: float, timeout_s: float) -> float:
    return rate_usd_s * (timeout_s + STARTUP_MARGIN_S + SCALEDOWN_MARGIN_S)


def timeout_for_budget(rate_usd_s: float, usd: float) -> int:
    """Largest Modal timeout whose worst case fits in ``usd`` (0 if none)."""
    return max(int(math.floor(usd / rate_usd_s - STARTUP_MARGIN_S - SCALEDOWN_MARGIN_S)), 0)


# Bridge web (<= 1 core, 1 GiB) + ledger (<= 0.5 core, 0.5 GiB) containers at their limits.
PLATFORM_RATE_USD_S = 1.5 * PRICE["cpu_core_s"] + 1.5 * PRICE["mem_gib_s"]
PLATFORM_WARM_S = 60.0   # scaledown_window of both functions


@dataclass(frozen=True)
class LedgerConfig:
    cap_usd: float = 10.0
    platform_reserve_usd: float = 0.25   # floor for bridge/ledger container spend; metered usage above it counts too
    per_job_cap_usd: float = 2.0
    max_active: int = 1
    lease_s: float = 60.0
    lock_wait_s: float = 30.0
    platform_rate_usd_s: float = PLATFORM_RATE_USD_S
    platform_warm_s: float = PLATFORM_WARM_S


class Ledger:
    def __init__(self, kv: KV, cfg: LedgerConfig, clock: Callable[[], float] = time.time):
        if cfg.per_job_cap_usd > cfg.cap_usd:
            raise ValueError("per-job cap exceeds project cap")
        self.kv, self.cfg, self.clock = kv, cfg, clock

    # ---------------- locking
    @contextlib.contextmanager
    def _locked(self) -> Iterator[None]:
        token = secrets.token_hex(8)
        deadline = time.monotonic() + self.cfg.lock_wait_s
        while True:
            now = self.clock()
            if self.kv.put(LOCK_KEY, {"token": token, "expires": now + self.cfg.lease_s}, skip_if_exists=True):
                break
            cur = self.kv.get(LOCK_KEY)
            if cur is not None and cur.get("expires", 0) < now:
                # Stale lease: holder crashed (writers are serialized in production, see module doc).
                if self.kv.pop(LOCK_KEY, None) is not None:
                    continue
            if time.monotonic() > deadline:
                raise LedgerStateError("could not acquire ledger lock")
            time.sleep(0.005)
        try:
            yield
        finally:
            cur = self.kv.get(LOCK_KEY)
            if cur is not None and cur.get("token") == token:
                self.kv.pop(LOCK_KEY, None)

    def _load(self) -> dict[str, Any]:
        try:
            st = self.kv.get(LEDGER_KEY)
        except Exception as e:  # noqa: BLE001
            raise LedgerStateError(f"ledger unreadable: {e!r}") from e
        if st is None:
            return {"version": 0, "jobs": {}, "created": self.clock()}
        if not isinstance(st, dict) or not isinstance(st.get("jobs"), dict) or "version" not in st:
            raise LedgerStateError("ledger corrupt")
        return st

    def _save(self, st: dict[str, Any]) -> None:
        new = int(st["version"]) + 1
        key = f"{COMMIT_PREFIX}{new}"
        if not self.kv.put(key, {"at": self.clock()}, skip_if_exists=True):
            claim = self.kv.get(key) or {}
            # A claim with no matching ledger write, older than the lease, is a writer that crashed in between.
            if claim.get("at", self.clock()) < self.clock() - self.cfg.lease_s and self.kv.pop(key, None) is not None \
                    and self.kv.put(key, {"at": self.clock()}, skip_if_exists=True):
                pass
            else:
                raise LedgerStateError(f"lost commit race at version {new}; retry")
        st["version"], st["updated"] = new, self.clock()
        self.kv.put(LEDGER_KEY, st)
        self.kv.pop(f"{COMMIT_PREFIX}{new - 2}", None)  # keep the CAS keyspace bounded

    # ---------------- platform (bridge/ledger containers) metering
    def _touch(self, st: dict[str, Any]) -> None:
        p = st.setdefault("platform", {"last_t": None, "accrued_s": 0.0})
        now = self.clock()
        if p["last_t"] is not None:
            gap = max(now - p["last_t"], 0.0)
            p["accrued_s"] += min(gap, self.cfg.platform_warm_s)
        p["last_t"] = now

    def _platform_usd(self, st: dict[str, Any]) -> float:
        p = st.get("platform") or {}
        tail = self.cfg.platform_warm_s if p.get("last_t") is not None else 0.0  # still-warm window after last op
        return (float(p.get("accrued_s", 0.0)) + tail) * self.cfg.platform_rate_usd_s

    # ---------------- accounting
    @staticmethod
    def _open_usd(j: dict[str, Any]) -> float:
        return float(j["reserved_usd"]) if j["state"] in ("reserved", "running") else 0.0

    @staticmethod
    def _settled_usd(j: dict[str, Any]) -> float:
        return float(j.get("settled_usd") or 0.0) if j["state"] == "settled" else 0.0

    def _totals(self, st: dict[str, Any]) -> dict[str, float]:
        jobs = st["jobs"].values()
        reserved = sum(self._open_usd(j) for j in jobs)
        settled = sum(self._settled_usd(j) for j in jobs)
        metered = self._platform_usd(st)
        platform = max(self.cfg.platform_reserve_usd, metered)
        committed = platform + reserved + settled
        return {"reserved_usd": reserved, "settled_estimate_usd": settled,
                "platform_reserve_usd": self.cfg.platform_reserve_usd, "platform_metered_usd": metered,
                "platform_committed_usd": platform, "committed_usd": committed,
                "available_usd": max(self.cfg.cap_usd - committed, 0.0)}

    def touch(self) -> dict[str, Any]:
        """Record bridge activity (called once per bridge HTTP request)."""
        with self._locked():
            st = self._load()
            self._touch(st)
            self._save(st)
            return {"platform_metered_usd": self._platform_usd(st)}

    def snapshot(self) -> dict[str, Any]:
        st = self._load()
        active = {k: v for k, v in st["jobs"].items() if v["state"] in ("reserved", "running")}
        return {"cap_usd": self.cfg.cap_usd, "per_job_cap_usd": self.cfg.per_job_cap_usd,
                "max_active": self.cfg.max_active, **self._totals(st), "n_jobs": len(st["jobs"]),
                "active": active, "ledger_version": st["version"], "disclaimer": DISCLAIMER,
                "pricing_source": "modal.com/pricing retrieved 2026-11-10",
                "margins_s": {"startup": STARTUP_MARGIN_S, "scaledown": SCALEDOWN_MARGIN_S}}

    def reserve(self, *, kind: str, usd: float, rate_usd_s: float, timeout_s: float,
                meta: dict[str, Any] | None = None) -> dict[str, Any]:
        if not (usd > 0 and math.isfinite(usd)):
            raise BudgetError("reservation must be positive")
        if usd > self.cfg.per_job_cap_usd + 1e-9:
            raise BudgetError(f"reservation ${usd:.2f} exceeds per-job cap ${self.cfg.per_job_cap_usd:.2f}")
        worst = worst_case_usd(rate_usd_s, timeout_s)
        if worst > usd + 1e-9:
            raise BudgetError(f"worst case ${worst:.3f} (rate x timeout+margins) exceeds reservation ${usd:.2f}")
        with self._locked():
            st = self._load()
            active = [j for j in st["jobs"].values() if j["state"] in ("reserved", "running")]
            if len(active) >= self.cfg.max_active:
                raise ConcurrencyError(f"{len(active)} job(s) active; max_active={self.cfg.max_active}")
            t = self._totals(st)
            if t["committed_usd"] + usd > self.cfg.cap_usd + 1e-9:
                raise BudgetError(f"reservation ${usd:.2f} exceeds available ${t['available_usd']:.2f} "
                                  f"(cap ${self.cfg.cap_usd:.2f}, committed ${t['committed_usd']:.2f})")
            jid = f"j{st['version'] + 1:05d}-{secrets.token_hex(3)}"
            now = self.clock()
            self._touch(st)
            st["jobs"][jid] = {"job_id": jid, "kind": kind, "state": "reserved", "reserved_usd": float(usd),
                               "rate_usd_s": float(rate_usd_s), "timeout_s": float(timeout_s),
                               "reserved_at": now, "start_deadline_unix": now + STARTUP_MARGIN_S,
                               "deadline_unix": now + STARTUP_MARGIN_S + timeout_s,
                               "call_id": None, "meta": meta or {}}
            self._save(st)
            return st["jobs"][jid]

    def attach(self, job_id: str, call_id: str) -> dict[str, Any]:
        with self._locked():
            st = self._load()
            j = self._job(st, job_id)
            if j["state"] != "reserved" or j["call_id"]:
                raise LedgerStateError(f"cannot attach to {job_id} in state {j['state']}")
            j["call_id"], j["state"], j["attached_at"] = call_id, "running", self.clock()
            self._save(st)
            return j

    def release(self, job_id: str, reason: str) -> dict[str, Any]:
        """Return a reservation whose spawn provably failed (no call id was ever obtained)."""
        with self._locked():
            st = self._load()
            j = self._job(st, job_id)
            if j["state"] != "reserved" or j["call_id"]:
                raise LedgerStateError(f"refusing to release {job_id}: state {j['state']}, call {j['call_id']}")
            j["state"], j["released_reason"], j["ended_at"] = "released", reason, self.clock()
            self._save(st)
            return j

    def settle(self, job_id: str, *, outcome: str, job_reported_end: float | None = None) -> dict[str, Any]:
        """Close a job confirmed terminal (finished / failed / cancelled / expired)."""
        with self._locked():
            st = self._load()
            j = self._job(st, job_id)
            if j["state"] == "settled":
                return j
            if j["state"] != "running":
                raise LedgerStateError(f"cannot settle {job_id} in state {j['state']}")
            latest = j["deadline_unix"]
            if outcome == "finished" and job_reported_end is not None:
                end, basis = max(job_reported_end, j["reserved_at"]), "job-reported end"
            else:
                end, basis = latest, f"worst case ({outcome}; end unknown)"
            j["settled_usd"] = j["rate_usd_s"] * (end - j["reserved_at"] + SCALEDOWN_MARGIN_S)
            j.update(state="settled", outcome=outcome, ended_at=end, settled_at=self.clock(), settlement_basis=basis)
            self._save(st)
            return j

    def settle_orphan(self, job_id: str) -> dict[str, Any]:
        """Close a reservation that never got a call id, once anything it might have started must have ended."""
        with self._locked():
            st = self._load()
            j = self._job(st, job_id)
            if j["state"] != "reserved" or j["call_id"]:
                raise LedgerStateError(f"{job_id} is not an orphan (state {j['state']})")
            if self.clock() <= j["deadline_unix"] + SCALEDOWN_MARGIN_S:
                raise LedgerStateError(f"{job_id}: too early to settle orphan (deadline not passed)")
            j.update(state="settled", outcome="orphan", settled_usd=float(j["reserved_usd"]), ended_at=j["deadline_unix"],
                     settled_at=self.clock(), settlement_basis="orphan: full reservation (spawn outcome unknown)")
            self._save(st)
            return j

    def job(self, job_id: str) -> dict[str, Any]:
        return self._job(self._load(), job_id)

    @staticmethod
    def _job(st: dict[str, Any], job_id: str) -> dict[str, Any]:
        if job_id not in st["jobs"]:
            raise LedgerStateError(f"unknown job {job_id}")
        return st["jobs"][job_id]

    def op(self, name: str, **kw: Any) -> dict[str, Any]:
        """Single dispatch point (what the serialized Modal function calls)."""
        fn = {"snapshot": self.snapshot, "reserve": self.reserve, "attach": self.attach, "release": self.release,
              "settle": self.settle, "settle_orphan": self.settle_orphan, "touch": self.touch,
              "job": self.job}.get(name)
        if fn is None:
            raise LedgerStateError(f"unknown ledger op {name}")
        return fn(**kw)
