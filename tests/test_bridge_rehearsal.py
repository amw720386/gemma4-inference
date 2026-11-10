"""CPU rehearsal of the paid path: HTTP submit -> ledger reserve -> worker (bridge_jobs, same code as the Modal
worker) -> runner on the mock engine -> poll -> settle. Catches spec/RunOptions mismatches before any paid call."""
import threading
import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from g4bench import budget as B  # noqa: E402
from g4bench.bridge_api import BridgeConfig, build_api  # noqa: E402
from g4bench.bridge_jobs import guarded, run_job  # noqa: E402

pytestmark = pytest.mark.slow


class LocalBackend:
    """Runs jobs in threads with the real worker body; ledger in memory."""

    def __init__(self, results_root, tokenizer):
        self.L = B.Ledger(B.MemoryKV(), B.LedgerConfig(cap_usd=10, per_job_cap_usd=2, max_active=1,
                                                       platform_reserve_usd=0.5))
        self.root, self.tok = results_root, tokenizer
        self.threads, self.results, self.specs = {}, {}, {}

    def ledger(self, op, **kw):
        return self.L.op(op, **kw)

    def spawn(self, kind, payload, *, gpu, timeout_s):
        cid = f"local-{len(self.threads) + 1}"
        self.specs[cid] = payload

        def body():
            self.results[cid] = guarded(lambda s: run_job(s, self.root, tokenizer=self.tok, server_timeout_s=60,
                                                           client_timeout_s=300), payload)

        t = threading.Thread(target=body, daemon=True)
        self.threads[cid] = t
        t.start()
        return cid

    def poll(self, cid):
        return ("running", None) if self.threads[cid].is_alive() else ("finished", self.results[cid])

    def cancel(self, cid):
        pass

    def list_runs(self):
        return sorted(p.name for p in self.root.iterdir()) if self.root.exists() else []

    def read_run_file(self, run_id, path):
        f = self.root / run_id / path
        return f.read_text() if f.is_file() else None


def test_submit_run_poll_settle_on_mock(tmp_path, tokenizer_dir):
    be = LocalBackend(tmp_path / "runs", tokenizer_dir)
    c = TestClient(build_api(be, BridgeConfig(cap_usd=10, per_job_cap_usd=2, max_active=1,
                                              allowed_engines={"mock-cpu": "mock"}, allowed_matrices={"mock-ci"})))
    r = c.post("/jobs", json={"engine": "mock-cpu", "matrix": "mock-ci", "seeds": [0], "max_usd": 2,
                              "notes": "rehearsal"})
    assert r.status_code == 200, r.text
    j = r.json()
    spec = be.specs[j["call_id"]]
    assert {"engine", "matrix", "seeds", "only", "trials", "deadline_unix", "max_wall_s", "job_id"} <= set(spec)
    t0 = time.time()
    while True:
        st = c.get(f"/jobs/{j['job_id']}").json()
        if st["state"] != "running":
            break
        assert time.time() - t0 < 300
        time.sleep(1)
    assert st["state"] == "finished" and st["job"]["state"] == "settled"
    res = st["result"]
    assert "error" not in res and "aborted" not in res, res
    assert res["n_points"] == 2 and res["n_valid"] == 2
    assert res["ended_unix"] >= res["started_unix"] and st["job"]["settlement_basis"] == "job-reported end"
    b = c.get("/budget").json()
    assert b["reserved_usd"] == 0 and 0 < b["settled_estimate_usd"] < 2.0 and b["available_usd"] < 9.5
    runs = c.get("/runs").json()
    assert runs == [res["run_id"]]
    assert "MOCK ENGINE" in c.get(f"/runs/{res['run_id']}/REPORT.md").text


def test_worker_refuses_after_deadline_and_wall_budget_is_bounded(tmp_path):
    out = guarded(lambda s: {"ran": True}, {"deadline_unix": time.time() - 1, "job_id": "j"})
    assert "aborted" in out and "ran" not in out
    err = guarded(lambda s: 1 / 0, {"deadline_unix": time.time() + 100})
    assert "ZeroDivisionError" in err["error"] and err["ended_unix"] >= err["started_unix"]
    # deadline already (nearly) reached => runner gets a 0 s wall budget => every point skipped, no crash
    res = run_job({"engine": "mock-cpu", "matrix": "mock-ci", "seeds": [0], "max_wall_s": 9999,
                   "deadline_unix": time.time() + 10}, tmp_path, correctness=False, tokenizer="unused")
    assert res["n_valid"] == 0 and res["n_points"] == 2


def test_guard_uses_start_deadline_and_propagates_cancellation():
    now = time.time()
    out = guarded(lambda s: {"ran": True}, {"start_deadline_unix": now - 1, "deadline_unix": now + 1000})
    assert "aborted" in out and "ran" not in out
    ok = guarded(lambda s: {"ran": True}, {"start_deadline_unix": now + 100, "deadline_unix": now + 1000})
    assert ok["ran"]

    class FakeCancel(BaseException):
        pass

    def body(s):
        raise FakeCancel()

    with pytest.raises(FakeCancel):
        guarded(body, {"start_deadline_unix": now + 100})


def test_modal_poll_classification_with_real_exception_classes(monkeypatch):
    modal = pytest.importorskip("modal")
    import modal.exception as me
    import sys
    from g4bench.paths import REPO

    sys.path.insert(0, str(REPO / "modal_bridge"))
    import app as bridge_app  # noqa: PLC0415

    cases = [(me.TimeoutError(), "running"), (TimeoutError(), "running"), (me.FunctionTimeoutError("t"), "failed"),
             (me.OutputExpiredError(), "expired"), (me.RemoteError("terminated"), "failed"),
             (me.InternalFailure("x"), "failed"), (me.ExecutionError("crash"), "failed"),
             (me.ConnectionError("net"), "unknown"), (OSError("reset"), "unknown")]
    for exc, want in cases:
        class FC:
            def get(self, timeout=None, _e=exc):
                raise _e

        monkeypatch.setattr(modal.FunctionCall, "from_id", staticmethod(lambda cid, _fc=FC(): _fc))
        assert bridge_app.ModalBackend().poll("fc-x")[0] == want, type(exc).__name__
