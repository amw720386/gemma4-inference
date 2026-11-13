"""Same-GPU paired benchmarking (D-061): ordering, sizing, ratios, summary math, routing, and a mock end-to-end."""
import json
import math
import socket
import time

import pytest
import yaml

from g4bench import analyze
from g4bench import paired as PR
from g4bench.config import EngineConfig, config_hash, load_pins, resolve_config
from g4bench.matrix import expand, load_matrix


def test_segment_order_counterbalances():
    assert [PR.segment_order("ABBA", r) for r in range(4)] == ["AB", "BA", "AB", "BA"]
    assert [PR.segment_order("ABAB", r) for r in range(3)] == ["AB"] * 3
    assert [PR.segment_order("BAAB", r) for r in range(2)] == ["BA", "AB"]  # continues an ABBA series
    with pytest.raises(ValueError):
        PR.segment_order("AABB", 0)


def test_sizing_against_the_3_dollar_reservation():
    from g4bench import spend as S
    from g4bench.modal_run import SOFT_WALL_HEADROOM_S

    m = load_matrix("backboard-repro-shared")
    wall = S.timeout_for_budget(S.GPU_JOB.rate(), 3.0) - SOFT_WALL_HEADROOM_S
    big = PR.plan_sequence(expand(m, trials=2, only_keys=["8192x64"]), max_wall_s=wall)
    assert not big["fits"] and big["pairs_that_fit"] == 0 and big["n_segments"] == 4
    assert "per-job cap" in big["suggestion"] and big["usd_needed_for_all"] > 3.0
    small = PR.plan_sequence(expand(m, trials=2, only_keys=["1024x1"]), max_wall_s=wall)
    assert small["fits"] and small["pairs_that_fit"] == 2
    mid = PR.plan_sequence(expand(m, trials=2, only_keys=["1024x64"]), max_wall_s=wall)
    assert not mid["fits"] and mid["pairs_that_fit"] == 1 and "BAAB" in mid["suggestion"]
    # Segment model: start + stop + (prompts + warmups) x mean output / article vLLM tok/s.
    (p,) = expand(m, only_keys=["8192x64"])
    assert PR.segment_seconds(p) == pytest.approx(180 + (640 + 128) * 921.6 / 1079)


def test_ratios_and_gpu_segment(tmp_path):
    r = PR.ratios({"output_throughput": 1000.0, "p99_tpot_ms": 20.0, "mean_ttft_ms": None},
                  {"output_throughput": 1100.0, "p99_tpot_ms": 18.0, "mean_ttft_ms": 5.0})
    assert r["output_throughput"] == pytest.approx(1.1) and r["p99_tpot_ms"] == pytest.approx(0.9)
    assert r["mean_ttft_ms"] is None
    f = tmp_path / "gpu_samples.csv"
    f.write_text("timestamp,index,memory.used,power.draw,clocks.sm,clocks.mem,utilization.gpu,temperature.gpu\n"
                 "t, 0, 1000, 500.0, 2370, 13365, 99, 70\n"
                 "t, 0, 1000, 520.0, 2210, 13365, 100, 84\n"
                 "t, 0, [N/A], [N/A], 2000, 13365, 100, 88\n")
    g = PR.gpu_segment(f)
    assert g["n_samples"] == 3 and g["sm_clock_mhz_min"] == 2000 and g["temp_c_max"] == 88
    assert g["power_w_mean"] == pytest.approx(510.0) and g["sm_clock_mhz_mean"] == pytest.approx(2193.33, abs=0.01)
    assert PR.gpu_segment(tmp_path / "missing.csv")["n_samples"] == 0


def _doc(a, b, key, ratios, *, osl=1024, same=True, valid=True, pid="p1"):
    return {"pair_id": pid, "engine_configs": {"A": a, "B": b}, "engine_kinds": {"A": "vllm", "B": "vllm"},
            "hardware_class": "RTX-PRO-6000-SEx1", "exact_hardware": True, "gpu": {"same_gpu": same},
            "pairs": [{"key": key, "osl": osl, "dataset": "random", "valid": valid, "gpu_uuid": [f"GPU-{i}"],
                       "same_gpu_as_job_start": True,
                       "ratios_b_over_a": {"output_throughput": x, "p99_tpot_ms": 1 / x}} for i, x in enumerate(ratios)]}


def test_paired_summary_geomean_ci_and_backboard_estimator():
    targets = {"8192x64": {"output_throughput": {"vllm": 1079.0, "backboard": 1254.0}}}
    xs = [1.12, 1.18, 1.15, 1.20]
    docs = [_doc("vllm-0.33.0-mtp4", "cand", "8192x64", xs[:2], pid="p1"),
            _doc("vllm-0.33.0-mtp4", "cand", "8192x64", xs[2:], pid="p2"),
            _doc("vllm-0.33.0-mtp4", "cand", "8192x64", [9.0], same=False, pid="p3"),  # GPU changed: excluded
            _doc("vllm-0.33.0-mtp4", "cand", "8192x64", [9.0], valid=False, pid="p4"),  # invalid pair: excluded
            _doc("other-ref", "cand", "8192x64", [1.5], pid="p5")]  # A is not the article recipe: no estimator
    rows = analyze.paired_summary(docs, targets)
    main = next(r for r in rows if r["engine_a"] == "vllm-0.33.0-mtp4")
    logs = [math.log(x) for x in xs]
    m = sum(logs) / 4
    sd = math.sqrt(sum((v - m) ** 2 for v in logs) / 3)
    half = 3.182 * sd / 2  # t(0.975, 3)
    assert main["n_pairs"] == 4 and main["n_jobs"] == 2
    assert main["ratio"] == pytest.approx(math.exp(m))
    assert (main["ratio_ci_lo"], main["ratio_ci_hi"]) == (pytest.approx(math.exp(m - half)),
                                                          pytest.approx(math.exp(m + half)))
    assert main["p99_tpot_ms_ratio"] == pytest.approx(1 / math.exp(m))
    f = 1079.0 / 1254.0
    assert main["candidate_vs_backboard"] == pytest.approx(math.exp(m) * f)
    assert main["candidate_vs_backboard_ci_hi"] == pytest.approx(math.exp(m + half) * f)
    other = next(r for r in rows if r["engine_a"] == "other-ref")
    assert other["ratio"] == pytest.approx(1.5) and other["candidate_vs_backboard"] is None
    assert other["ratio_ci_lo"] is None  # n = 1: no interval
    md = "\n".join(analyze.paired_markdown(rows))
    assert "Same-GPU paired" in md and "cand" in md
    # Not the article workload (OSL 512): no Backboard estimate either.
    assert analyze.paired_summary([_doc("vllm-0.33.0-mtp4", "c", "8192x64", [1.1], osl=512)],
                                  targets)[0]["candidate_vs_backboard"] is None


def test_routing_puts_stock_and_candidate_on_one_image():
    from g4bench.modal_run import pair_kind

    assert pair_kind("vllm-0.33.0-mtp4", "vllm-0.33.0-best-e8-pfa-c8k") == ("gpu:vllm-patched-exp", [])
    kind, warn = pair_kind("vllm-0.33.0-mtp4", "vllm-0.33.0-best-e8-all")
    assert kind == "gpu:vllm-patched-exp" and warn and "cuBLAS" in warn[0]  # loses its in-image cuBLAS 13.10
    assert pair_kind("vllm-0.33.0-mtp4", "vllm-0.33.0-mtp6") == ("gpu:vllm", [])
    with pytest.raises(SystemExit):
        pair_kind("vllm-0.33.0-mtp4", "sglang-0.5.23-nextn")


def test_hashes_of_existing_series_are_unchanged_and_foreign_image_is_tagged():
    """The D-061 env gates must not split the E6b/E8 series: these are the hashes in their run ids."""
    pins, m = load_pins(), load_matrix("backboard-repro-shared")
    want = {"vllm-0.33.0-best-e6b": "e1c2c9a3", "vllm-0.33.0-best-e8-all": "deccf806",
            "vllm-0.33.0-best-e8-pfa": "92b787bd", "vllm-0.33.0-best-e8-pfa-c8k": "6fd46fcb",
            "vllm-0.33.0-mtp6-splitk-chunk3072-draftlast": "776cd326", "vllm-0.33.0-mtp4": "28818286"}
    for name, h in want.items():
        e = EngineConfig.load(name)
        assert config_hash(e, m, pins)[:8] == h, name
        assert config_hash(e, m, pins, foreign_image="vllm-patched-exp")[:8] != h, name
    # Gate unset on a config listing the patch = a different (upstream) behaviour = a different hash.
    raw = yaml.safe_load(resolve_config("vllm-0.33.0-best-e6b", "engines").read_text())
    raw["env"].pop("G4_DRAFT_PREFILL_LAST")
    off = EngineConfig(**{**EngineConfig.load("vllm-0.33.0-best-e6b").__dict__, "env": raw["env"] or {}})
    assert config_hash(off, m, pins)[:8] != want["vllm-0.33.0-best-e6b"]


def test_foreign_image_tag():
    from g4bench import images as I

    exp = {"G4_IMAGE_PATCHES": ",".join(I.VLLM_PATCHES_EXP)}
    prod = {"G4_IMAGE_PATCHES": ",".join(I.VLLM_PATCHES)}
    assert I.foreign_image_tag("vllm", [], exp) == "vllm-patched-exp"  # stock recipe on the exp image
    assert I.foreign_image_tag("vllm", ["prefill_attn_g4", "mtp_draft_prefill_last"], exp) is None  # home
    assert I.foreign_image_tag("vllm", ["mtp_draft_prefill_last"], prod) is None  # home
    assert I.foreign_image_tag("vllm", ["mtp_draft_prefill_last"], exp) == "vllm-patched-exp"
    assert I.foreign_image_tag("vllm", [], {}) is None  # stock image / local


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _mock_configs(tmp_path):
    base = yaml.safe_load(resolve_config("mock-cpu", "engines").read_text())
    base["port"] = _free_port()
    (tmp_path / "eng").mkdir()
    a = tmp_path / "eng" / "mock-cpu.yaml"
    a.write_text(yaml.safe_dump(base))
    fast = dict(base, name="mock-cpu-fast",
                args=[x for x in base["args"] if x[0] != "--step-ms"] + [["--step-ms", "0.1"],
                                                                         ["--prefill-ms-per-1k", "1.0"]])
    b = tmp_path / "eng" / "mock-cpu-fast.yaml"
    b.write_text(yaml.safe_dump(fast))
    return str(a), str(b)


def test_wall_budget_never_starts_a_pair_it_cannot_finish(tmp_path):
    a, b = _mock_configs(tmp_path)
    now = time.time()
    spec = {"paired": {"engines": [a, b], "order": "ABBA", "repeats": 2}, "matrix": "mock-ci", "seeds": [0],
            "only": ["1024x1"], "max_wall_s": 5, "deadline_unix": now + 1000, "correctness": False}
    from g4bench.jobs import run_job

    out = run_job(spec, tmp_path / "runs", tokenizer="unused")
    assert out["n_pairs"] == 2 and out["n_valid_pairs"] == 0
    doc = json.loads((tmp_path / "pairs" / out["pair_id"] / "paired.json").read_text())
    assert all("wall budget" in p["skipped"] for p in doc["pairs"])
    for rid in out["run_ids"].values():
        rows = [json.loads(x) for x in (tmp_path / "runs" / rid / "results.jsonl").read_text().splitlines()]
        assert len(rows) == 2 and all(r["status"] == "skipped" for r in rows)
        man = json.loads((tmp_path / "runs" / rid / "manifest.json").read_text())
        assert man["paired"]["pair_id"] == out["pair_id"] and man["paired"]["order"] == "ABBA"


@pytest.mark.slow
def test_paired_mock_end_to_end(tmp_path, tokenizer_dir):
    """Worker body (jobs.run_job) on the mock: 2 repeats ABBA = 4 fresh servers on one 'GPU', both run dirs + paired.json,
    and analyze picks the pairs up."""
    a, b = _mock_configs(tmp_path)
    now = time.time()
    spec = {"paired": {"engines": [a, b], "order": "ABBA", "repeats": 2}, "matrix": "mock-ci", "seeds": [0],
            "only": ["1024x1"], "max_wall_s": 900, "deadline_unix": now + 1200, "correctness": False,
            "notes": "pytest paired"}
    from g4bench.jobs import collect_run_files, guarded, run_job

    res = guarded(lambda s: run_job(s, tmp_path / "runs", tokenizer=tokenizer_dir, server_timeout_s=60,
                                    client_timeout_s=300), spec)
    assert "error" not in res, res
    assert res["n_pairs"] == res["n_valid_pairs"] == 2
    doc = json.loads((tmp_path / "pairs" / res["pair_id"] / "paired.json").read_text())
    assert [p["sequence"] for p in doc["pairs"]] == ["AB", "BA"]
    for p in doc["pairs"]:
        assert [s["slot"] for s in p["segments"]] == list(p["sequence"])
        assert p["segments"][0]["ended_unix"] <= p["segments"][1]["started_unix"]  # strictly sequential
        assert p["ratios_b_over_a"]["output_throughput"] > 1.0  # the faster mock is B
        assert all(s["valid"] and s["server_start_s"] is not None for s in p["segments"])
        assert "gpu" in p["segments"][0] and "container" in p
    assert doc["engines"] == {"A": a, "B": b} and doc["engine_kinds"] == {"A": "mock", "B": "mock"}
    for slot, rid in res["run_ids"].items():
        d = tmp_path / "runs" / rid
        rows = [json.loads(x) for x in (d / "results.jsonl").read_text().splitlines()]
        assert len(rows) == 2 and all(r["valid"] for r in rows) and {r["trial"] for r in rows} == {0, 1}
        assert f"slot {slot}" in rows[0]["notes"] and (d / "REPORT.md").exists()
        assert all((d / "points" / r["point_id"] / "cublas.json").exists() for r in rows)
        assert collect_run_files(tmp_path / "runs", rid)
    out = analyze.run(root=tmp_path / "runs", out=tmp_path / "summary", include_mock=True)
    assert out["paired"] == 1
    assert "Same-GPU paired" in (tmp_path / "summary" / "SUMMARY.md").read_text()
