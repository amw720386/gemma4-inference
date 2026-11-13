"""Full pipeline on CPU: mock server -> vendored InferenceX client -> normalize -> analyze."""
import json

import pytest

from g4bench import analyze
from g4bench.runner import RunOptions, run


@pytest.mark.slow
def test_mock_end_to_end(tmp_path, tokenizer_dir):
    d = run(RunOptions(engine="mock-cpu", matrix="mock-ci", results_root=tmp_path, tokenizer=tokenizer_dir,
                       server_timeout_s=60, client_timeout_s=300, notes="pytest e2e"))
    rows = [json.loads(x) for x in (d / "results.jsonl").read_text().splitlines()]
    assert len(rows) == 4
    for r in rows:
        assert r["valid"], r["problems"]
        assert r["engine"] == "mock" and r["completed"] == r["num_prompts"]
        assert r["spec_acceptance_length"] == pytest.approx(3.0, rel=0.05)
        assert r["output_throughput"] > 0 and r["p99_e2el_ms"] > r["p99_ttft_ms"]
        assert 51 <= r["mean_output_len"] <= 64  # range ratio 0.8 of osl=64, ignore_eos honoured
    man = json.loads((d / "manifest.json").read_text())
    assert man["config_hash"] and man["points"][0]["point_id"]
    meta = json.loads((d / "metadata.json").read_text())
    assert meta["hardware_class"] == "cpu" and meta["exact_hardware"] is False
    corr = json.loads((d / "correctness.json").read_text())
    assert corr["probes"]["pass"] and corr["probes"]["n_ok"] == 8
    assert len(corr["greedy"]["items"]) == 8 and all(it.get("token_ids") for it in corr["greedy"]["items"])
    out = analyze.run(root=tmp_path, out=tmp_path / "summary", include_mock=True)
    assert out["rows"] == 4 and out["aggregates"] == 2
    assert (tmp_path / "summary" / "SUMMARY.md").exists()
