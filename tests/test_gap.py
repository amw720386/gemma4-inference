import pytest

from g4bench import gap, report
from g4bench.config import EngineConfig


def test_attribution_math():
    r = gap.attribute(tpot_ms=6.0, al=3.0, conc=1, isl=1024, osl=1024, article_tpot_ms=1000 / 204.1, draft_tokens=4)
    assert r["step_ms"] == pytest.approx(18.0)
    assert r["al_needed_for_article"] == pytest.approx(18.0 / (1000 / 204.1))
    assert r["step_speedup_needed"] == pytest.approx(6.0 / (1000 / 204.1))
    assert r["reachable_by_acceptance_alone"] is True and 0 < r["bw_efficiency"] <= 1.0
    hopeless = gap.attribute(tpot_ms=20.0, al=3.0, conc=1, isl=1024, osl=1024, article_tpot_ms=4.9, draft_tokens=4)
    assert hopeless["reachable_by_acceptance_alone"] is False
    assert gap.attribute(tpot_ms=5.0, al=None, conc=1, isl=1, osl=1, article_tpot_ms=1, draft_tokens=4)["state"] != "ok"


def test_draft_tokens_from_engine_configs():
    assert gap.draft_tokens_for(EngineConfig.load("vllm-0.33.0-mtp4").raw) == 4
    assert gap.draft_tokens_for(EngineConfig.load("sglang-0.5.23-nextn").raw) == 5
    assert gap.draft_tokens_for(EngineConfig.load("vllm-0.33.0-nospec").raw) is None


def test_report_gap_section(tmp_path):
    import json

    from test_report import _mk, _row

    eng = EngineConfig.load("vllm-0.33.0-mtp4").raw
    rows = [_row(1024, 1, s, 180.0, mean_tpot_ms=5.4, spec_acceptance_length=3.1) for s in range(3)]
    d = _mk(tmp_path / "g", rows=rows)
    man = json.loads((d / "manifest.json").read_text())
    man["engine_config"] = eng
    (d / "manifest.json").write_text(json.dumps(man))
    md = report.build(d)[0]
    sec = md.split("## Gap attribution")[1].split("## Correctness")[0]
    assert "| 1024 | 1 | 5.4 | 3.1 |" in sec
    assert "Not computed" in report.build(_mk(tmp_path / "m", engine="mock", exact=False, hw="cpu"))[0]
