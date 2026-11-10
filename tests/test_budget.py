import threading

import pytest

from g4bench import budget as B


class Clock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


def _ledger(kv=None, clock=None, **cfg):
    c = B.LedgerConfig(**{"cap_usd": 10.0, "platform_reserve_usd": 0.5, "per_job_cap_usd": 2.0, "max_active": 1, **cfg})
    return B.Ledger(kv or B.MemoryKV(), c, clock or Clock())


def _reserve(L, usd=2.0, rate=0.001, timeout=None):
    timeout = B.timeout_for_budget(rate, usd) if timeout is None else timeout
    return L.reserve(kind="gpu:vllm", usd=usd, rate_usd_s=rate, timeout_s=timeout)


def test_rates_and_timeout_math():
    r = B.resource_rate("RTX-PRO-6000", 8, 64)
    assert r == pytest.approx(0.000842 + 8 * 0.0000131 + 64 * 0.00000222)
    t = B.timeout_for_budget(r, 2.0)
    assert B.worst_case_usd(r, t) <= 2.0 < B.worst_case_usd(r, t + 2)
    assert B.timeout_for_budget(r, 0.5) == 0  # $0.50 cannot even cover the startup margin on this GPU


def test_cap_enforced_and_snapshot_accounting():
    L = _ledger(max_active=10)
    for _ in range(4):
        _reserve(L)  # 0.5 + 4*2 = 8.5
    with pytest.raises(B.BudgetError):
        _reserve(L)  # would be 10.5 > 10
    s = L.snapshot()
    assert s["reserved_usd"] == pytest.approx(8.0) and s["committed_usd"] == pytest.approx(8.5)
    assert s["available_usd"] == pytest.approx(1.5) and "NOT a Modal billing hard stop" in s["disclaimer"]
    _reserve(L, usd=1.5, rate=0.0005)  # exactly fills the cap
    assert L.snapshot()["available_usd"] == pytest.approx(0.0)


def test_per_job_cap_and_worst_case_checks():
    L = _ledger()
    with pytest.raises(B.BudgetError):
        _reserve(L, usd=2.5)
    with pytest.raises(B.BudgetError):  # timeout too long for the reservation
        L.reserve(kind="gpu:vllm", usd=1.0, rate_usd_s=0.001, timeout_s=5000)
    with pytest.raises(B.BudgetError):
        L.reserve(kind="x", usd=0.0, rate_usd_s=0.001, timeout_s=1)
    with pytest.raises(ValueError):
        _ledger(per_job_cap_usd=20.0)


def test_concurrency_limit():
    L = _ledger(max_active=1)
    j = _reserve(L)
    with pytest.raises(B.ConcurrencyError):
        _reserve(L)
    L.attach(j["job_id"], "fc-1")
    with pytest.raises(B.ConcurrencyError):
        _reserve(L)
    L.settle(j["job_id"], outcome="finished", job_reported_end=j["reserved_at"] + 100)
    _reserve(L)  # slot freed only after confirmed settlement


def test_parallel_reservations_never_exceed_cap():
    kv = B.MemoryKV()
    clock = Clock()
    ok, refused = [], []

    def worker():
        L = _ledger(kv, clock, max_active=100)  # separate Ledger objects = separate "containers"
        try:
            ok.append(_reserve(L))
        except (B.BudgetError, B.ConcurrencyError):
            refused.append(1)

    ts = [threading.Thread(target=worker) for _ in range(40)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert len(ok) == 4 and len(refused) == 36  # (10 - 0.5) // 2
    assert len({j["job_id"] for j in ok}) == 4
    assert _ledger(kv, clock).snapshot()["committed_usd"] <= 10.0 + 1e-9


def test_state_survives_restart():
    kv, clock = B.MemoryKV(), Clock()
    j = _reserve(_ledger(kv, clock))
    _ledger(kv, clock).attach(j["job_id"], "fc-9")
    fresh = _ledger(kv, clock)  # new process, same Dict
    assert fresh.job(j["job_id"])["state"] == "running" and fresh.snapshot()["reserved_usd"] == 2.0
    with pytest.raises(B.ConcurrencyError):
        _reserve(fresh)


def test_stale_lock_is_recovered_live_lock_is_respected():
    kv, clock = B.MemoryKV(), Clock()
    kv.put(B.LOCK_KEY, {"token": "dead", "expires": clock() - 1})
    _reserve(_ledger(kv, clock))  # crashed holder: lease expired -> proceeds
    kv.put(B.LOCK_KEY, {"token": "alive", "expires": clock() + 3600})
    L = B.Ledger(kv, B.LedgerConfig(lock_wait_s=0.05, max_active=5), clock)
    with pytest.raises(B.LedgerStateError):
        _reserve(L)


def test_corrupt_state_fails_closed():
    kv = B.MemoryKV()
    kv.put(B.LEDGER_KEY, {"jobs": "nonsense"})
    with pytest.raises(B.LedgerStateError):
        _reserve(_ledger(kv))
    with pytest.raises(B.LedgerStateError):
        _ledger(kv).snapshot()


def test_release_only_never_started():
    L = _ledger(max_active=5)
    a = _reserve(L)
    L.release(a["job_id"], "spawn failed")
    assert L.snapshot()["reserved_usd"] == 0
    b = _reserve(L)
    L.attach(b["job_id"], "fc-2")
    with pytest.raises(B.LedgerStateError):
        L.release(b["job_id"], "nope")  # it may be running: never released


def test_settlement_is_conservative_and_uncapped():
    clock = Clock()
    L = _ledger(clock=clock, max_active=5)
    rate = 0.001
    j = _reserve(L, rate=rate)
    L.attach(j["job_id"], "fc")
    s = L.settle(j["job_id"], outcome="finished", job_reported_end=j["reserved_at"] + 600)
    assert s["settled_usd"] == pytest.approx(rate * (600 + B.SCALEDOWN_MARGIN_S))
    assert L.settle(j["job_id"], outcome="finished")["settled_usd"] == s["settled_usd"]  # idempotent
    # expired output => assume it ran as long as Modal allowed
    k = _reserve(L, rate=rate)
    L.attach(k["job_id"], "fc2")
    clock.t += 10
    e = L.settle(k["job_id"], outcome="expired")
    assert e["settled_usd"] == pytest.approx(rate * (k["timeout_s"] + B.STARTUP_MARGIN_S + B.SCALEDOWN_MARGIN_S))
    # a reported overrun is recorded in full, not capped at the reservation
    m = _reserve(L, rate=rate)
    L.attach(m["job_id"], "fc3")
    over = L.settle(m["job_id"], outcome="finished", job_reported_end=m["reserved_at"] + 10 * m["timeout_s"])
    assert over["settled_usd"] > m["reserved_usd"]


def test_unknown_op_and_job():
    L = _ledger()
    with pytest.raises(B.LedgerStateError):
        L.op("drop_table")
    with pytest.raises(B.LedgerStateError):
        L.job("nope")


def test_non_finished_outcomes_settle_at_worst_case_and_bound_holds():
    clock = Clock()
    L = _ledger(clock=clock, max_active=10)
    for outcome in ("failed", "cancelled", "expired"):
        j = _reserve(L)
        L.attach(j["job_id"], "fc")
        clock.t += 5
        s = L.settle(j["job_id"], outcome=outcome)
        assert s["settled_usd"] == pytest.approx(B.worst_case_usd(j["rate_usd_s"], j["timeout_s"]))
        assert s["settled_usd"] <= j["reserved_usd"] + 1e-9
    j = _reserve(L)
    assert j["start_deadline_unix"] == j["reserved_at"] + B.STARTUP_MARGIN_S
    assert j["deadline_unix"] == j["start_deadline_unix"] + j["timeout_s"]


def test_orphan_settles_only_after_deadline_at_full_reservation():
    clock = Clock()
    L = _ledger(clock=clock)
    j = _reserve(L)
    with pytest.raises(B.LedgerStateError):
        L.settle_orphan(j["job_id"])  # too early
    clock.t = j["deadline_unix"] + B.SCALEDOWN_MARGIN_S + 1
    s = L.settle_orphan(j["job_id"])
    assert s["settled_usd"] == j["reserved_usd"] and s["outcome"] == "orphan"
    k = _reserve(L)
    L.attach(k["job_id"], "fc")
    clock.t = k["deadline_unix"] + 1e6
    with pytest.raises(B.LedgerStateError):
        L.settle_orphan(k["job_id"])  # has a call id: must be settled via poll, not as orphan


def test_platform_metering_counts_above_the_reserve():
    clock = Clock()
    L = _ledger(clock=clock, platform_reserve_usd=0.01, platform_rate_usd_s=0.001, platform_warm_s=60)
    L.touch()
    for _ in range(100):  # a poll every 30 s for 50 min keeps the containers warm continuously
        clock.t += 30
        L.touch()
    s = L.snapshot()
    assert s["platform_metered_usd"] == pytest.approx((100 * 30 + 60) * 0.001)
    assert s["committed_usd"] == pytest.approx(s["platform_metered_usd"]) and s["available_usd"] < 10 - 3
    clock.t += 10_000  # long idle gap counts as one warm window, not the whole gap
    L.touch()
    assert L.snapshot()["platform_metered_usd"] == pytest.approx((100 * 30 + 60 + 60) * 0.001)


def test_commit_cas_rejects_concurrent_writer_and_recovers_stale_claim():
    kv, clock = B.MemoryKV(), Clock()
    L = _ledger(kv, clock, max_active=10)
    v = L.snapshot()["ledger_version"]
    kv.put(f"{B.COMMIT_PREFIX}{v + 1}", {"at": clock()})  # another writer claimed the next version just now
    with pytest.raises(B.LedgerStateError):
        _reserve(L)
    clock.t += L.cfg.lease_s + 1                            # ...and then crashed without writing the ledger
    _reserve(L)
    assert L.snapshot()["ledger_version"] == v + 1
