import json

import pytest

from g4bench import analyze
from g4bench.engine_compare import compare_engines

BASE = {"hardware_class": "RTX-PRO-6000x1", "dataset": "random", "dataset_digest": None, "isl": 1024, "osl": 1024,
        "conc": 1, "random_range_ratio": 0.8, "use_chat_template": True, "ignore_eos": True, "num_prompts": 10,
        "valid": True, "pins_target": "t", "pins_draft": "d", "exact_hardware": True}


def _r(engine, seed, thr, p99, run, **kw):
    return {**BASE, "engine": engine, "seed": seed, "output_throughput": thr, "p99_tpot_ms": p99,
            "mean_ttft_ms": 90.0, "p90_ttft_ms": 100.0, "p99_ttft_ms": 110.0, "mean_tpot_ms": p99 * 0.95,
            "mean_e2el_ms": 5000.0, "p99_e2el_ms": 5400.0, "run_id": run, "point_id": f"{engine}-{seed}", **kw}


def _rows():
    return ([_r("vllm", s, v, 5.6, "rv") for s, v in enumerate([186.0, 188.0, 187.0])] +
            [_r("sglang", s, v, 8.4, "rs") for s, v in enumerate([122.0, 123.0, 122.5])])


def _thr(out):
    return next(r for r in out if r["metric"] == "output_throughput")


def test_paired_ratio_ci_and_article_reference():
    out = compare_engines(_rows(), correctness={"rv": True, "rs": True})
    t = _thr(out)
    assert t["n_paired"] == 3 and not t["gated"] and t["evidence"] == "claim (n>=3)"
    assert t["advantage"] == pytest.approx(187.0 / 122.5)
    assert t["boot_ci"][0] < t["advantage"] < t["boot_ci"][1]
    assert t["paired_ci"][0] < t["advantage"] < t["paired_ci"][1]
    assert t["article_advantage"] == pytest.approx(186.8 / 122.6)
    tp = next(r for r in out if r["metric"] == "p99_tpot_ms")
    assert tp["advantage"] == pytest.approx(8.4 / 5.6)  # latency: B/A, >1 = vLLM better


def test_correctness_gate_blocks_ratio():
    out = compare_engines(_rows(), correctness={"rv": True, "rs": False})
    t = _thr(out)
    assert t["gated"] and t["advantage"] is None and any("correctness" in g for g in t["gate_reasons"])
    assert compare_engines(_rows())[0]["gated"]  # unknown correctness also gates


def test_missing_side_and_invalid_rows_are_explicit():
    rows = [r for r in _rows() if r["engine"] == "vllm"] + [_r("sglang", 0, 1.0, 1.0, "rs", valid=False)]
    t = _thr(compare_engines(rows, correctness={"rv": True, "rs": True}))
    assert t["gated"] and "not measured: sglang" in t["gate_reasons"] and t["b_mean"] is None
    assert any("invalid" in g for g in t["gate_reasons"])


def test_unmatched_workloads_never_pair():
    rows = _rows()
    for r in rows:
        if r["engine"] == "sglang":
            r["random_range_ratio"] = 1.0
    out = compare_engines(rows, correctness={"rv": True, "rs": True})
    assert all(r["gated"] for r in out) and len({(r["random_range_ratio"]) for r in out}) == 2


def test_pin_mismatch_and_descriptive_label():
    rows = _rows()[:1] + _rows()[3:4]
    rows[1]["pins_draft"] = "other"
    t = _thr(compare_engines(rows, correctness={"rv": True, "rs": True}))
    assert t["gated"] and any("pins differ" in g for g in t["gate_reasons"])
    rows[1]["pins_draft"] = "d"
    t = _thr(compare_engines(rows, correctness={"rv": True, "rs": True}))
    assert t["evidence"] == "descriptive (n<3)" and t["boot_ci"] is None and t["paired_ci"] is None


def test_analyze_end_to_end_reads_run_context(tmp_path):
    for run, eng in (("rv", "vllm"), ("rs", "sglang")):
        d = tmp_path / run
        d.mkdir()
        (d / "manifest.json").write_text(json.dumps({"pins": {"model": {"target": {"revision": "t"},
                                                                      "draft": {"revision": "d"}}}}))
        (d / "correctness.json").write_text(json.dumps({"probes": {"pass": True}}))
        rows = [r for r in _rows() if r["engine"] == eng]
        for r in rows:
            r.pop("pins_target"), r.pop("pins_draft"), r.pop("run_id")
        (d / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows) + "{corrupt\n")
    res = analyze.run(root=tmp_path, out=tmp_path / "summary")
    assert res["rows"] == 6 and res["engine_pairs"] == 13
    md = (tmp_path / "summary" / "SUMMARY.md").read_text()
    assert "vLLM vs SGLang at matched load" in md and "claim (n>=3)" in md and "1.527" in md


def test_article_implied_ratios_cancel_backboard_for_all_13_metrics():
    from g4bench.engine_compare import METRICS, article_implied

    imp = article_implied("1024x1")
    assert len(imp) == 13
    assert imp["output_throughput"]["est"] == pytest.approx(186.8 / 122.6, rel=0.01)  # matches absolute table
    assert imp["p99_tpot"]["lo"] <= 8.4 / 5.6 * 1.03 and imp["p99_tpot"]["hi"] >= 8.4 / 5.6 * 0.97
    for k, v in imp.items():
        assert v["lo"] < v["est"] < v["hi"]
    out = compare_engines(_rows(), correctness={"rv": True, "rs": True})
    assert {r["metric"] for r in out} == set(METRICS)
    t = _thr(out)
    assert t["consistent_with_article"] is True  # synthetic 187/122.5 = 1.527 lies inside the article bounds
    lo, hi = t["article_bounds"]
    assert (t["boot_ci"][0] <= hi and lo <= t["boot_ci"][1]) == t["consistent_with_article"]
    assert article_implied("0x1") == {}


def test_inconsistency_with_article_is_flagged():
    rows = _rows()
    for r in rows:
        if r["engine"] == "sglang":
            r["output_throughput"] = 180.0  # engines nearly equal: far from the article's ~1.52x
    t = _thr(compare_engines(rows, correctness={"rv": True, "rs": True}))
    assert t["consistent_with_article"] is False and t["advantage"] < 1.1
