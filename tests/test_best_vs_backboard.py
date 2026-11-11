"""Best lossless (class a) open-engine config per article workload vs Backboard."""

import math

import pytest

from g4bench import analyze

TARGETS = {"1024x1": {"output_throughput": {"backboard": 200.0}},
           "1024x4": {"output_throughput": {"backboard": 800.0}},
           "8192x4": {"output_throughput": {"backboard": 500.0}}}


def _agg(cfg, isl, conc, mean, n=1, **kw):
    r = {"engine": "vllm", "engine_config": cfg, "hardware_class": "RTX-PRO-6000-SEx1", "exact_hardware": True,
         "dataset": "random", "isl": isl, "osl": 1024, "conc": conc,
         "output_throughput__mean": mean, "output_throughput__n": n}
    r.update(kw)
    return r


ADM = {"vllm-0.33.0-mtp4": "a", "fast-a": "a", "draft-fp8": "b", "quant": "c", "mock-cpu": "c"}


def _by_key(rows):
    return {r["workload"]: r for r in rows}


def test_best_pick_and_baseline_reference():
    agg = [_agg("vllm-0.33.0-mtp4", 1024, 1, 180.0), _agg("fast-a", 1024, 1, 195.0, n=3)]
    r = _by_key(analyze.best_vs_backboard(agg, ADM, targets=TARGETS))["1024x1"]
    assert r["best_config"] == "fast-a" and r["n"] == 3 and r["ratio"] == pytest.approx(0.975)
    assert r["baseline_mean"] == 180.0 and r["baseline_ratio"] == pytest.approx(0.9)


def test_admissibility_mock_and_hardware_filtering():
    agg = [_agg("vllm-0.33.0-mtp4", 1024, 4, 600.0),
           _agg("draft-fp8", 1024, 4, 900.0),                       # class b: excluded
           _agg("quant", 1024, 4, 950.0),                           # class c: excluded
           _agg("unknown-cfg", 1024, 4, 990.0),                     # no admissibility: excluded
           _agg("mock-cpu", 1024, 4, 999.0, engine="mock"),         # mock: excluded
           _agg("fast-a", 1024, 4, 980.0, exact_hardware=False),    # proxy hardware: excluded
           _agg("fast-a", 1024, 4, 970.0, dataset="sharegpt"),      # wrong dataset: excluded
           _agg("fast-a", 1024, 4, 960.0, osl=512)]                 # wrong OSL: excluded
    r = _by_key(analyze.best_vs_backboard(agg, ADM, targets=TARGETS))["1024x4"]
    assert r["best_config"] == "vllm-0.33.0-mtp4" and r["mean"] == 600.0


def test_not_measured_rows_cover_all_eight_workloads():
    rows = analyze.best_vs_backboard([], ADM, targets=TARGETS)
    assert [r["workload"] for r in rows] == [f"{i}x{c}" for i in (1024, 8192) for c in (1, 4, 16, 64)]
    assert not any(r["measured"] for r in rows) and all(r["ratio"] is None for r in rows)
    md = "\n".join(analyze.best_vs_backboard_markdown(rows))
    assert md.count("not measured") == 8 and "**0 / 0**" in md and "**-**" in md


def test_within_5pct_count_and_geomean():
    agg = [_agg("fast-a", 1024, 1, 190.0),      # 0.95 -> within (boundary inclusive)
           _agg("fast-a", 1024, 4, 720.0),      # 0.90 -> not within
           _agg("fast-a", 8192, 4, 520.0)]      # 1.04 -> within
    rows = analyze.best_vs_backboard(agg, ADM, targets=TARGETS)
    s = analyze.best_summary(rows)
    assert s["measured"] == 3 and s["total"] == 8 and s["within_5pct"] == 2
    assert s["geomean_ratio"] == pytest.approx(math.exp((math.log(0.95) + math.log(0.9) + math.log(1.04)) / 3))
    md = "\n".join(analyze.best_vs_backboard_markdown(rows))
    assert "**2 / 3**" in md and "descriptive (n<3)" in md and md.count("not measured") == 5
    assert "Best lossless open-engine config vs Backboard" in md


def test_analyze_writes_section_and_csv(tmp_path):
    import json

    d = tmp_path / "r1"
    d.mkdir()
    rows = [{"engine": "vllm", "engine_config": "vllm-0.33.0-mtp4", "admissibility": "a", "engine_role": "baseline",
             "hardware_class": "RTX-PRO-6000-SEx1", "exact_hardware": True, "dataset": "random", "isl": 1024,
             "osl": 1024, "conc": 1, "seed": 0, "valid": True, "output_throughput": 190.4}]
    (d / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    analyze.run(root=tmp_path, out=tmp_path / "s")
    md = (tmp_path / "s" / "SUMMARY.md").read_text()
    csv = (tmp_path / "s" / "best_vs_backboard.csv").read_text()
    assert "## Best lossless open-engine config vs Backboard" in md and "| 1024x1 | vllm-0.33.0-mtp4 | 190.4 |" in md
    assert csv.startswith("workload,") and csv.count("\n") == 9
