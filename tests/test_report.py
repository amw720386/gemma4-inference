"""Report states: not measured, mock, proxy, exact-hardware comparison, real-prompt isolation, corrupt input."""
import json
import xml.etree.ElementTree as ET

from g4bench import report


def _mk(run_dir, *, engine="vllm", exact=True, rows=(), hw="RTX-PRO-6000x1", corr=None):
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(json.dumps({"run_id": run_dir.name, "config_hash": "abc",
                                                       "engine_config": {"name": f"{engine}-cfg", "engine": engine,
                                                                         "version": "x"}}))
    (run_dir / "metadata.json").write_text(json.dumps({"hardware_class": hw, "exact_hardware": exact, "gpus": []}))
    (run_dir / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    if corr is not None:
        (run_dir / "correctness.json").write_text(json.dumps(corr))
    return run_dir


def _row(isl, conc, seed, thr, dataset="random", **kw):
    r = {"valid": True, "engine": "vllm", "dataset": dataset, "isl": isl, "conc": conc, "seed": seed,
         "output_throughput": thr, "mean_ttft_ms": 90.0, "p99_tpot_ms": 5.5, "mean_tpot_ms": 5.3,
         "spec_acceptance_length": None, "point_id": f"isl{isl}-c{conc}-s{seed}"}
    r.update(kw)
    return r


def _svg_ok(d):
    ET.fromstring((d / "throughput.svg").read_text())


def test_empty_run_is_not_measured(tmp_path):
    d = _mk(tmp_path / "r0")
    report.write(d)
    md = (d / "REPORT.md").read_text()
    assert "NOT MEASURED" in md and "not measured" in md and "no correctness.json" in md
    _svg_ok(d)


def test_mock_run_banner_and_no_comparison(tmp_path):
    d = _mk(tmp_path / "r1", engine="mock", exact=False, hw="cpu", rows=[_row(1024, 1, 0, 100.0, engine="mock")])
    md = report.build(d)[0]
    assert "MOCK ENGINE" in md and "Not compared: mock engine." in md
    assert "| 1024 | 1 | 100" not in md.split("Comparison with article")[1]


def test_exact_hardware_comparison_and_acceptance_not_measured(tmp_path):
    rows = [_row(1024, 1, s, v) for s, v in enumerate([186.0, 188.0, 187.0])]
    rows.append(_row(1024, 1, 9, 999.0, valid=False, problems=["2 failed requests"]))
    d = _mk(tmp_path / "r2", rows=rows, corr={"probes": {"n_ok": 8, "n": 8, "pass": True}, "greedy_outputs": ["a"]})
    report.write(d)
    md = (d / "REPORT.md").read_text()
    cmp_section = md.split("Comparison with article")[1]
    assert "| 1024 | 1 | 187 | 3 | 186.8 |" in cmp_section and "consistent" in cmp_section
    al_section = md.split("## Speculative decoding acceptance")[1].split("## Comparison")[0]
    assert "| random | 1024 | 1 | not measured |" in al_section  # AL missing -> explicit, never 0
    assert "8/8 — PASS" in md and "2 failed requests" in md
    assert "(article)" in (d / "throughput.svg").read_text()
    _svg_ok(d)


def test_proxy_hardware_and_real_prompts_not_compared(tmp_path):
    d = _mk(tmp_path / "r3", exact=False, hw="L40Sx1", rows=[_row(1024, 4, 0, 500.0)])
    md = report.build(d)[0]
    assert "PROXY" in md and "Not compared: not the article's hardware" in md
    d2 = _mk(tmp_path / "r4", rows=[_row(1024, 4, 0, 500.0, dataset="speedbench-tp1k-inline")])
    md2 = report.build(d2)[0]
    assert "Real-prompt arms are intentionally not compared" in md2
    assert "| – | – | not measured |" in md2.split("Comparison with article")[1]


def test_corrupt_results_line_is_reported_not_fatal(tmp_path):
    d = _mk(tmp_path / "r5", rows=[_row(1024, 1, 0, 180.0)])
    with open(d / "results.jsonl", "a") as f:
        f.write("{truncated\n")
    md = report.build(d)[0]
    assert "unparseable results line" in md
