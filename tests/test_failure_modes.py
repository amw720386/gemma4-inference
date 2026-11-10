"""Integration failure cases on CPU (mock server + vendored client). No GPU, no network beyond the tokenizer."""
import json

import pytest

from g4bench import speedbench as sb
from g4bench.runner import DatasetError, RunOptions, run

pytestmark = pytest.mark.slow


def _rows(d):
    return [json.loads(x) for x in (d / "results.jsonl").read_text().splitlines()]


def _opts(tmp_path, tok, **kw):
    base = dict(engine="mock-cpu", matrix="mock-ci", results_root=tmp_path / "runs", tokenizer=tok,
                server_timeout_s=30, client_timeout_s=120, seeds=[0], correctness=False)
    base.update(kw)
    return RunOptions(**base)


def test_server_that_dies_marks_points_skipped(tmp_path, tokenizer_dir):
    d = run(_opts(tmp_path, tokenizer_dir, extra_server_args=["--no-such-flag"]))
    rows = _rows(d)
    assert rows and all(r["status"] == "skipped" and not r["valid"] for r in rows)
    assert all("server failed" in r["problems"][0] for r in rows)
    assert "NOT MEASURED" in (d / "REPORT.md").read_text()


def test_injected_request_failures_fail_the_gate(tmp_path, tokenizer_dir):
    d = run(_opts(tmp_path, tokenizer_dir, extra_server_args=["--fail-every", "2"]))
    rows = _rows(d)
    assert rows and not any(r["valid"] for r in rows)
    assert all(r["failed"] and r["failure_rate"] > 0.05 for r in rows if r["status"] != "no-result")


def test_client_timeout_is_recorded(tmp_path, tokenizer_dir):
    d = run(_opts(tmp_path, tokenizer_dir, only_keys=["1024x1"], client_timeout_s=3,
                  extra_server_args=["--step-ms", "500"]))
    (r,) = _rows(d)
    assert not r["valid"] and any("exit code -9" in p or "no result" in p for p in r["problems"])


def test_wall_budget_exhausted_skips_everything(tmp_path, tokenizer_dir):
    d = run(_opts(tmp_path, tokenizer_dir, max_wall_s=0))
    assert all(r["status"] == "skipped" and "wall budget" in r["notes"] for r in _rows(d))


def test_missing_dataset_fails_before_run_dir(tmp_path, tokenizer_dir, monkeypatch):
    monkeypatch.setenv("G4_DATA_DIR", str(tmp_path / "nodata"))
    with pytest.raises(DatasetError):
        run(_opts(tmp_path, tokenizer_dir, matrix="mock-real"))
    assert not (tmp_path / "runs").exists() or not any((tmp_path / "runs").iterdir())


def test_tampered_dataset_fails_before_run_dir(tmp_path, tokenizer_dir, monkeypatch):
    data = tmp_path / "data"
    f = sb._write(data, "mock-prompts", [{"question_id": "a", "prompt": "hello"}],
                  {"dataset": "mock-prompts", "prompt_digest": sb.digest(["hello"])})
    f.write_text(f.read_text().replace("hello", "HELLO"))
    monkeypatch.setenv("G4_DATA_DIR", str(data))
    with pytest.raises(DatasetError):
        run(_opts(tmp_path, tokenizer_dir, matrix="mock-real"))


def test_real_prompt_arm_end_to_end_on_mock(tmp_path, tokenizer_dir, monkeypatch):
    data = tmp_path / "data"
    prompts = [{"question_id": f"q{i}", "prompt": f"Write function number {i} in Python."} for i in range(5)]
    sb._write(data, "mock-prompts", prompts,
              {"dataset": "mock-prompts", "n_prompts": 5, "prompt_digest": sb.digest([p["prompt"] for p in prompts])})
    monkeypatch.setenv("G4_DATA_DIR", str(data))
    d = run(_opts(tmp_path, tokenizer_dir, matrix="mock-real", correctness=True))
    rows = _rows(d)
    assert len(rows) == 2 and all(r["valid"] for r in rows), [r["problems"] for r in rows]
    for r in rows:
        assert r["dataset"] == "mock-prompts" and r["key"].endswith("@mock-prompts")
        assert r["dataset_digest"] == sb.digest([p["prompt"] for p in prompts])
        assert r["mean_output_len"] == 32  # real-prompt arm + ignore_eos => exact OSL
        assert r["req_state"] == "ok" and r["req_n_ok"] == r["num_prompts"]
    raw = json.loads((d / "points" / rows[1]["point_id"] / "inferencex_result.json").read_text())
    assert raw["dataset"]["n_unique"] == 5 and raw["dataset"]["max_repeats"] == 2  # 8 requests over 5 prompts
    man = json.loads((d / "manifest.json").read_text())
    assert man["datasets"]["mock-prompts"]["n_prompts"] == 5
    md = (d / "REPORT.md").read_text()
    assert "MOCK ENGINE" in md and "mock-prompts" in md and "8/8" in md


def test_slow_point_is_cut_at_wall_budget_and_results_still_written(tmp_path, tokenizer_dir):
    import time as _t

    t0 = _t.monotonic()
    d = run(_opts(tmp_path, tokenizer_dir, only_keys=["1024x1"], max_wall_s=35, client_timeout_s=3600,
                  extra_server_args=["--step-ms", "2000"]))
    elapsed = _t.monotonic() - t0
    (r,) = _rows(d)
    assert not r["valid"] and elapsed < 120  # bounded by the wall budget, not the 3600 s client timeout
    assert (d / "REPORT.md").exists()


def test_bounded_helper():
    import time as _t

    from g4bench.runner import MIN_STEP_TIMEOUT_S, _bounded

    now = _t.monotonic()
    assert _bounded(1800, 600, now) <= 600
    assert _bounded(100, 10_000, now) == 100
    assert _bounded(1800, 5, now - 100) == MIN_STEP_TIMEOUT_S


def test_eos_honouring_real_prompt_arm_on_mock(tmp_path, tokenizer_dir, monkeypatch):
    data = tmp_path / "data"
    prompts = [{"question_id": f"q{i}", "prompt": f"Explain topic {i}."} for i in range(3)]
    sb._write(data, "mock-prompts", prompts,
              {"dataset": "mock-prompts", "prompt_digest": sb.digest([p["prompt"] for p in prompts])})
    monkeypatch.setenv("G4_DATA_DIR", str(data))
    d = run(_opts(tmp_path, tokenizer_dir, matrix="mock-real-eos"))
    rows = _rows(d)
    assert rows and all(r["valid"] for r in rows), [r["problems"] for r in rows]
    for r in rows:
        assert r["ignore_eos"] is False and r["point_id"].endswith("-eos") and r["key"].endswith("+eos")
        assert r["mean_output_len"] == 16  # mock ends early (32 // 2) when EOS is honoured
    argv = json.loads((d / "points" / rows[0]["point_id"] / "client_argv.json").read_text())
    assert "--no-ignore-eos" in argv


def test_eos_validation_accepts_short_but_not_overlong_outputs():
    from g4bench import results

    rec = {"status": "passed", "failed": 0, "mean_output_len": 120.0, "output_throughput": 1.0,
           "mean_ttft_ms": 1.0, "mean_tpot_ms": 1.0, "p99_e2el_ms": 1.0}
    assert results.validate(rec, osl=1024, range_ratio=0.8, ignore_eos=False, real_prompts=True) == []
    assert results.validate(rec, osl=1024, range_ratio=0.8, ignore_eos=True, real_prompts=True)  # must be exactly OSL
    assert results.validate(dict(rec, mean_output_len=2000.0), osl=1024, range_ratio=0.8, ignore_eos=False,
                            real_prompts=True)
