"""Campaign gates against the real bridge API (fake Modal backend from test_bridge_api)."""
import json

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from g4bench import campaign as C  # noqa: E402
from g4bench.bridge_api import BridgeConfig, build_api  # noqa: E402
from test_bridge_api import FakeBackend  # noqa: E402

E1 = C.load("e1-smoke")
DIGEST = E1["steps"][1]["expect"]["speedbench_coding_digest"]


class Driver:
    """Routes campaign HTTP calls to TestClient; completes each call when it is first polled."""

    def __init__(self, be, results=None, cap=10.0, per_job=2.0):
        self.be = be
        self.c = TestClient(build_api(be, BridgeConfig(cap_usd=cap, per_job_cap_usd=per_job, max_active=1,
                                                       estimator=lambda s: {"points": 2, "est_gpu_s": 600, "summary": ""})))
        self.results = results or {}
        self.posts = []

    def __call__(self, method, path, json=None):
        if method == "POST":
            self.posts.append(path)
        r = self.c.request(method, path, json=json)
        if r.status_code >= 400:
            raise C.CampaignStop(f"HTTP {r.status_code} {path}: {r.text}")
        if method == "GET" and path.startswith("/jobs/"):
            for cid, call in self.be.calls.items():
                if call["state"] == "running":
                    call["state"] = "finished"
                    call["result"] = self.results.get(call["kind"], {"n_points": 2, "n_valid": 2}) | {
                        "ended_unix": self.be.clock[0] + 300}
            r = self.c.request(method, path)
        return r.json()


GOOD = {"cpu:selftest": {"rc": 0}, "cpu:prefetch": {"speedbench": {"coding": DIGEST}}}


def test_plan_only_by_default_spends_nothing(tmp_path):
    d = Driver(FakeBackend(), GOOD)
    outs = C.run(E1, d, ledger=tmp_path / "l.jsonl")
    assert [o.status for o in outs] == ["planned"] * 4 and d.posts == [] and not (tmp_path / "l.jsonl").exists()


def test_full_e1_executes_in_order_and_logs(tmp_path):
    d = Driver(FakeBackend(), GOOD)
    outs = C.run(E1, d, execute=True, sleep=lambda s: None, ledger=tmp_path / "l.jsonl")
    assert [o.status for o in outs] == ["done"] * 4, [o.detail for o in outs]
    assert d.posts == ["/selftest", "/prefetch", "/jobs", "/jobs"]
    lines = [json.loads(x) for x in (tmp_path / "l.jsonl").read_text().splitlines()]
    assert len(lines) == 4 and all(x["budget_after"]["cap_usd"] == 10.0 for x in lines)
    assert lines[-1]["budget_after"]["available_usd"] < 9.5 and all(x["settled_usd"] is not None for x in lines)


def test_stops_on_failed_expectation(tmp_path):
    bad = {**GOOD, "cpu:prefetch": {"speedbench": {"coding": "deadbeef"}}}
    d = Driver(FakeBackend(), bad)
    outs = C.run(E1, d, execute=True, sleep=lambda s: None, ledger=tmp_path / "l.jsonl")
    assert [o.status for o in outs] == ["done", "failed"] and "digest" in outs[1].detail
    assert d.posts == ["/selftest", "/prefetch"]  # no GPU job launched after a failed gate


def test_refuses_when_server_limits_differ_from_approval(tmp_path):
    for cap, per_job in ((100.0, 2.0), (10.0, 5.0)):
        be = FakeBackend(cap=cap, per_job=per_job)
        d = Driver(be, GOOD, cap=cap, per_job=per_job)
        outs = C.run(E1, d, execute=True, sleep=lambda s: None, ledger=tmp_path / "l.jsonl")
        assert outs[0].status == "refused" and d.posts == [], outs[0].detail


def test_refuses_when_remaining_budget_too_small(tmp_path):
    be = FakeBackend(cap=10.0)
    be.L.cfg = type(be.L.cfg)(**{**be.L.cfg.__dict__, "platform_reserve_usd": 8.2})  # leaves $1.80
    d = Driver(be, GOOD)
    outs = C.run(E1, d, execute=True, sleep=lambda s: None, ledger=tmp_path / "l.jsonl")
    assert [o.status for o in outs] == ["done", "done", "refused"] and "available" in outs[2].detail
    assert d.posts == ["/selftest", "/prefetch"]


def test_job_with_invalid_points_stops_campaign(tmp_path):
    d = Driver(FakeBackend(), {**GOOD, "gpu:vllm": {"n_points": 2, "n_valid": 1}})
    outs = C.run(E1, d, execute=True, sleep=lambda s: None, ledger=None)
    assert [o.status for o in outs] == ["done", "done", "failed"] and "1/2" in outs[2].detail


def test_campaign_files_reference_allowlisted_configs():
    from g4bench.bridge_api import ALLOWED_ENGINES, ALLOWED_MATRICES

    for name in ("e1-smoke", "e2-pairs"):
        for st in C.load(name)["steps"]:
            if st["kind"] == "job":
                assert st["spec"]["engine"] in ALLOWED_ENGINES and st["spec"]["matrix"] in ALLOWED_MATRICES
                assert st["spec"]["max_usd"] <= 2.0


def test_campaign_stops_polling_after_deadline_grace(tmp_path):
    be = FakeBackend()
    d = Driver(be, GOOD)
    t = [be.clock[0]]

    def never_finish(method, path, json=None):
        r = d.c.request(method, path, json=json)
        if r.status_code >= 400:
            raise C.CampaignStop(r.text)
        return r.json()

    steps = {"name": "x", "require": E1["require"], "steps": [E1["steps"][2]]}

    def sleep(s):
        t[0] += s

    outs = C.run(steps, never_finish, execute=True, sleep=sleep, ledger=None, now=lambda: t[0], poll_s=600)
    assert outs[-1].status == "failed" and "stopped polling" in outs[-1].detail


def test_environment_approval_only_tightens_limits():
    req = {"cap_usd": 10.0, "per_job_cap_usd_max": 2.0, "max_active": 1}
    assert C.approved_limits(req, {}) == req
    t = C.approved_limits(req, {"G4_PROJECT_BUDGET_USD": "8", "G4_JOB_CAP_USD": "1.5"})
    assert t["cap_usd"] == 8.0 and t["per_job_cap_usd_max"] == 1.5
    loose = C.approved_limits(req, {"G4_PROJECT_BUDGET_USD": "100", "G4_JOB_CAP_USD": "40"})
    assert loose["cap_usd"] == 10.0 and loose["per_job_cap_usd_max"] == 2.0


def test_check_limits_passes_on_fresh_correct_server_and_fails_closed():
    d = Driver(FakeBackend(), GOOD)
    b = C.check_limits(d, {"cap_usd": 10.0, "per_job_cap_usd_max": 2.0, "max_active": 1}, need_usd=0.0)
    assert b["available_usd"] == 9.5 and b["cap_usd"] == 10.0

    def broken(method, path, json=None):
        raise C.CampaignStop("HTTP 404 /health: modal-http: invalid function call")

    with pytest.raises(C.CampaignStop):
        C.check_limits(broken, {"cap_usd": 10.0, "per_job_cap_usd_max": 2.0, "max_active": 1}, need_usd=0.0)


def test_stale_bridge_without_safety_features_is_refused():
    d = Driver(FakeBackend(), GOOD)

    def stale(method, path, json=None):  # an old deploy: /health without the features list
        out = d(method, path, json=json)
        if path == "/health":
            out = {k: v for k, v in out.items() if k != "features"}
        return out

    with pytest.raises(C.CampaignStop, match="lacks safety features"):
        C.check_limits(stale, {"cap_usd": 10.0, "per_job_cap_usd_max": 2.0, "max_active": 1}, need_usd=0.0)
