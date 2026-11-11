"""CPU rehearsal of the paid worker body (jobs.py is exactly what the Modal functions run) on the mock engine."""
import time

import pytest

from g4bench.jobs import collect_run_files, guarded, run_job, write_run_files


@pytest.mark.slow
def test_worker_body_runs_mock_and_returns_full_evidence(tmp_path, tokenizer_dir):
    now = time.time()
    spec = {"engine": "mock-cpu", "matrix": "mock-ci", "seeds": [0], "max_wall_s": 600, "job_id": "j1",
            "start_deadline_unix": now + 60, "deadline_unix": now + 900, "notes": "rehearsal"}
    res = guarded(lambda s: run_job(s, tmp_path / "runs", tokenizer=tokenizer_dir, server_timeout_s=60,
                                    client_timeout_s=300), spec)
    assert "error" not in res and "aborted" not in res, res
    assert res["n_points"] == res["n_valid"] == 2 and res["ended_unix"] >= res["started_unix"]
    files = collect_run_files(tmp_path / "runs", res["run_id"])
    for rel in ("REPORT.md", "throughput.svg", "manifest.json", "metadata.json", "results.jsonl", "results.csv"):
        assert rel in files, rel
    assert any(k.endswith("/inferencex_result.json") for k in files)
    assert any(k.endswith("/server.log") for k in files)
    d = write_run_files(tmp_path / "local", res["run_id"], files)
    assert b"MOCK ENGINE" in (d / "REPORT.md").read_bytes()
    with pytest.raises(FileExistsError):  # never overwrites local evidence
        write_run_files(tmp_path / "local", res["run_id"], files)


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


def test_run_file_helpers_refuse_traversal(tmp_path):
    (tmp_path / "runs" / "r1").mkdir(parents=True)
    (tmp_path / "runs" / "r1" / "a.txt").write_text("x")
    (tmp_path / "secret.txt").write_text("nope")
    assert collect_run_files(tmp_path / "runs", "r1") == {"a.txt": b"x"}
    assert collect_run_files(tmp_path / "runs", "..") == {}
    with pytest.raises(ValueError):
        write_run_files(tmp_path / "out", "r1", {"../../evil.txt": b"x"})
    with pytest.raises(ValueError):
        write_run_files(tmp_path / "out", "..", {"a.txt": b"x"})
