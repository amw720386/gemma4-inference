"""Bridge HTTP API with fake Modal backend: cap, concurrency, settlement, failure handling."""
import threading

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from g4bench import budget as B  # noqa: E402
from g4bench.bridge_api import GPU_JOB, PREFETCH, BridgeConfig, build_api  # noqa: E402


class FakeBackend:
    def __init__(self, cap=10.0, per_job=2.0, max_active=1, reserve=0.5):
        self.clock = [1_000_000.0]
        self.L = B.Ledger(B.MemoryKV(), B.LedgerConfig(cap_usd=cap, per_job_cap_usd=per_job, max_active=max_active,
                                                       platform_reserve_usd=reserve), lambda: self.clock[0])
        self.calls: dict[str, dict] = {}
        self.fail_spawn = False
        self.n = 0
        self.m = threading.Lock()

    def ledger(self, op, **kw):
        return self.L.op(op, **kw)

    def spawn(self, kind, payload, *, gpu, timeout_s):
        if self.fail_spawn:
            raise self.fail_spawn if isinstance(self.fail_spawn, Exception) else RuntimeError("modal down")
        with self.m:
            self.n += 1
            cid = f"fc-{self.n}"
        self.calls[cid] = {"kind": kind, "payload": payload, "gpu": gpu, "timeout_s": timeout_s, "state": "running"}
        return cid

    def poll(self, cid):
        c = self.calls[cid]
        return c["state"], c.get("result")

    def cancel(self, cid):
        self.calls[cid]["cancel_requested"] = True

    def list_runs(self):
        return []

    def list_run_files(self, run_id):
        return None

    def read_run_file(self, run_id, path):
        return None


def _client(be, **kw):
    cfg = BridgeConfig(cap_usd=be.L.cfg.cap_usd, per_job_cap_usd=be.L.cfg.per_job_cap_usd,
                       max_active=be.L.cfg.max_active,
                       estimator=kw.pop("estimator", lambda spec: {"points": 2, "est_gpu_s": 600, "summary": "fake"}))
    return TestClient(build_api(be, cfg))


SPEC = {"engine": "vllm-0.33.0-mtp4", "matrix": "smoke", "max_usd": 2.0}


def test_health_and_budget_shape():
    be = FakeBackend()
    c = _client(be)
    h = c.get("/health").json()
    assert h["ok"] and h["cap_usd"] == 10.0 and h["per_job_cap_usd"] == 2.0 and h["available_usd"] == 9.5
    b = c.get("/budget").json()
    for k in ("cap_usd", "reserved_usd", "settled_estimate_usd", "committed_usd", "available_usd", "disclaimer"):
        assert k in b
    assert "NOT a Modal billing hard stop" in b["disclaimer"]


def test_job_lifecycle_reserve_attach_settle():
    be = FakeBackend()
    c = _client(be)
    r = c.post("/jobs", json=SPEC)
    assert r.status_code == 200, r.text
    j = r.json()
    rate = GPU_JOB.rate()
    assert j["timeout_s"] == B.timeout_for_budget(rate, 2.0) and j["reserved_usd"] <= 2.0
    call = be.calls[j["call_id"]]
    assert call["timeout_s"] == j["timeout_s"] and call["payload"]["deadline_unix"] == j["deadline_unix"]
    assert call["payload"]["max_wall_s"] < j["timeout_s"]
    assert c.get("/budget").json()["reserved_usd"] == j["reserved_usd"]
    assert c.post("/jobs", json=SPEC).status_code == 409  # one active job max
    assert c.post("/prefetch").status_code == 409         # concurrency covers CPU jobs too
    st = c.get(f"/jobs/{j['job_id']}").json()
    assert st["state"] == "running"
    call["state"], call["result"] = "finished", {"ended_unix": be.clock[0] + 400}
    st = c.get(f"/jobs/{j['job_id']}").json()
    assert st["state"] == "finished" and st["job"]["state"] == "settled"
    assert st["job"]["settled_usd"] == pytest.approx(rate * (400 + B.SCALEDOWN_MARGIN_S))
    b = c.get("/budget").json()
    assert b["reserved_usd"] == 0 and b["settled_estimate_usd"] == pytest.approx(st["job"]["settled_usd"])


def test_cap_stops_launches_and_prefetch_selftest_are_charged():
    be = FakeBackend(cap=3.0, reserve=0.5, max_active=5)
    c = _client(be)
    assert c.post("/jobs", json=SPEC).status_code == 200              # ~$2 reserved, $0.5 left
    p = c.post("/prefetch")
    assert p.status_code == 200 and p.json()["reserved_usd"] == pytest.approx(
        min(2.0, -(-B.worst_case_usd(PREFETCH.rate(), PREFETCH.timeout_s) * 100 // 1) / 100))
    assert c.post("/selftest").status_code in (200, 402)
    r = c.post("/jobs", json=SPEC)
    assert r.status_code == 402 and "budget" in r.json()["detail"]
    assert c.get("/budget").json()["committed_usd"] <= 3.0 + 1e-9


def test_job_that_does_not_fit_per_job_reservation_is_refused():
    be = FakeBackend()
    c = _client(be, estimator=lambda s: {"points": 8, "est_gpu_s": 5 * 3600, "summary": "big"})
    r = c.post("/jobs", json=SPEC)
    assert r.status_code == 402 and "split the job" in r.json()["detail"]
    assert c.get("/budget").json()["reserved_usd"] == 0
    assert c.post("/estimate", json=SPEC).json()["fits"] is False


def test_spawn_failures_local_release_ambiguous_hold_then_orphan_settle():
    be = FakeBackend()
    c = _client(be)
    be.fail_spawn = ValueError("unknown kind")          # raised before any RPC: provably never started
    assert c.post("/jobs", json=SPEC).status_code == 400
    assert c.get("/budget").json()["reserved_usd"] == 0
    be.fail_spawn = RuntimeError("deadline exceeded")   # Modal may have accepted it: hold
    r = c.post("/jobs", json=SPEC)
    assert r.status_code == 502 and "stays held" in r.json()["detail"]
    b = c.get("/budget").json()
    held = b["reserved_usd"]
    assert held > 0 and len(b["active"]) == 1
    be.fail_spawn = False
    assert c.post("/jobs", json=SPEC).status_code == 409   # still blocks concurrency
    job = next(iter(b["active"].values()))
    be.clock[0] = job["deadline_unix"] + B.SCALEDOWN_MARGIN_S + 1
    b = c.get("/budget").json()                            # reconcile settles the orphan at full reservation
    assert b["reserved_usd"] == 0 and b["settled_estimate_usd"] == pytest.approx(held)
    assert c.post("/jobs", json=SPEC).status_code == 200


def test_unknown_poll_holds_and_attach_failure_keeps_reservation():
    be = FakeBackend()
    c = _client(be)
    j = c.post("/jobs", json=SPEC).json()
    be.calls[j["call_id"]]["state"] = "unknown"  # transient Modal error: must NOT settle
    assert c.get(f"/jobs/{j['job_id']}").json()["job"]["state"] == "running"
    assert c.get("/budget").json()["reserved_usd"] == j["reserved_usd"]
    be2 = FakeBackend()
    real = be2.ledger

    def flaky(op, **kw):
        if op == "attach":
            raise B.LedgerStateError("dict hiccup")
        return real(op, **kw)

    be2.ledger = flaky
    c2 = _client(be2)
    r = c2.post("/jobs", json=SPEC)
    assert r.status_code == 502 and "fc-1" in r.json()["detail"] and "stays held" in r.json()["detail"]
    be2.ledger = real
    assert c2.get("/budget").json()["reserved_usd"] > 0


def test_failed_and_cancelled_settle_at_worst_case():
    for state in ("failed", "cancelled", "expired"):
        be = FakeBackend()
        c = _client(be)
        j = c.post("/jobs", json=SPEC).json()
        be.calls[j["call_id"]]["state"] = state
        st = c.get(f"/jobs/{j['job_id']}").json()
        rate = GPU_JOB.rate()
        assert st["job"]["settled_usd"] == pytest.approx(B.worst_case_usd(rate, j["timeout_s"]))
        assert st["job"]["settled_usd"] <= j["reserved_usd"] + 1e-9


def test_reservation_never_exceeds_requested_max_usd_and_requests_are_metered():
    be = FakeBackend()
    c = _client(be)
    j = c.post("/jobs", json={**SPEC, "max_usd": 1.95}).json()
    assert j["reserved_usd"] <= 1.95
    v0 = be.L.snapshot()["ledger_version"]
    for _ in range(5):
        c.get("/health")
    assert be.L.snapshot()["ledger_version"] >= v0 + 5  # each request touched the ledger
    assert be.L.snapshot()["platform_metered_usd"] > 0


def test_cancel_holds_reservation_until_confirmed():
    be = FakeBackend()
    c = _client(be)
    j = c.post("/jobs", json=SPEC).json()
    assert c.post(f"/jobs/{j['job_id']}/cancel").status_code == 200
    assert be.calls[j["call_id"]]["cancel_requested"]
    assert c.get("/budget").json()["reserved_usd"] == j["reserved_usd"]
    be.calls[j["call_id"]]["state"] = "cancelled"
    assert c.get("/budget").json()["reserved_usd"] == 0  # /budget reconciles and settles


def test_allowlists_and_validation():
    c = _client(FakeBackend())
    assert c.post("/jobs", json={**SPEC, "engine": "evil"}).status_code == 400
    assert c.post("/jobs", json={**SPEC, "matrix": "x"}).status_code == 400
    assert c.post("/jobs", json={**SPEC, "gpu": "T4"}).status_code == 400
    assert c.post("/jobs", json={**SPEC, "max_usd": 0}).status_code == 422


def test_parallel_http_submissions_respect_cap():
    be = FakeBackend(cap=6.5, max_active=50)
    c = _client(be)
    codes = []

    def go():
        codes.append(c.post("/jobs", json=SPEC).status_code)

    ts = [threading.Thread(target=go) for _ in range(12)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert codes.count(200) == 3 and all(x in (200, 402) for x in codes)  # (6.5-0.5)//2
    assert c.get("/budget").json()["committed_usd"] <= 6.5 + 1e-9


def test_ledger_failure_fails_closed():
    be = FakeBackend()

    def broken(op, **kw):
        raise B.LedgerStateError("dict unreachable")

    be.ledger = broken
    c = _client(be)
    assert c.get("/health").json()["ok"] is False
    assert c.post("/jobs", json=SPEC).status_code == 503 and not be.calls
