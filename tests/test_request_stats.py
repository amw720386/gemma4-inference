import sys

import numpy as np
import pytest

from g4bench import request_stats as rs
from g4bench.paths import THIRD_PARTY_INFERENCEX


def _raw(n=20, L=50, fail=(), seed=0):
    rng = np.random.default_rng(seed)
    ttfts, itls, lens, errs = [], [], [], []
    for k in range(n):
        if k in fail:
            ttfts.append(0.0); itls.append([]); lens.append(0); errs.append("boom")
            continue
        ttfts.append(float(rng.uniform(0.05, 0.2)))
        itls.append([float(x) for x in rng.uniform(0.004, 0.006, size=L - 1)])
        lens.append(L); errs.append("")
    return {"ttfts": ttfts, "itls": itls, "output_lens": lens, "errors": errs}


def test_definitions_match_vendored_inferencex_calculate_metrics():
    sys.path.insert(0, str(THIRD_PARTY_INFERENCEX))
    from infx.bench_serving import benchmark_serving as bs
    from infx.bench_serving.backend_request_func import RequestFuncOutput

    raw = _raw(n=30, fail=(3, 17))
    outs = []
    for t, itl, L, e in zip(raw["ttfts"], raw["itls"], raw["output_lens"], raw["errors"]):
        o = RequestFuncOutput()
        o.success, o.ttft, o.itl, o.output_tokens = (not e), t, itl, L
        o.latency = t + sum(itl)
        outs.append(o)
    reqs = [("p", 10, 50, None)] * 30
    m, _ = bs.calculate_metrics(reqs, outs, 10.0, None, [50.0, 90.0, 99.0], {})
    s = rs.summarize(raw, conc=4)
    assert s["n_ok"] == 28 and s["n_failed"] == 2
    got = s["metrics"]
    assert got["tpot_ms"]["mean"]["est"] == pytest.approx(m.mean_tpot_ms, rel=1e-9)
    assert got["ttft_ms"]["p99"]["est"] == pytest.approx(dict(m.percentiles_ttft_ms)[99.0], rel=1e-9)
    assert got["e2el_ms"]["p90"]["est"] == pytest.approx(dict(m.percentiles_e2el_ms)[90.0], rel=1e-9)
    assert got["itl_ms"]["mean"]["est"] == pytest.approx(m.mean_itl_ms, rel=1e-9)


def test_cis_bracket_estimates_and_are_seeded():
    s1 = rs.summarize(_raw(), conc=1, seed=1)
    s2 = rs.summarize(_raw(), conc=1, seed=1)
    assert s1 == s2
    for name in ("ttft_ms", "tpot_ms", "e2el_ms"):
        for st in rs.STATS:
            e = s1["metrics"][name][st]
            assert e["ci"][0] <= e["est"] <= e["ci"][1] + 1e-12


def test_littles_law_at_c1_matches_sequential_throughput():
    raw = _raw(n=10)
    total_L = sum(raw["output_lens"])
    total_e2e = sum(t + sum(i) for t, i in zip(raw["ttfts"], raw["itls"]))
    raw["output_throughput"] = total_L / total_e2e  # c=1, no gaps between requests
    ll = rs.summarize(raw, conc=1)["throughput_littles_law"]
    assert ll["est"] == pytest.approx(raw["output_throughput"]) and abs(ll["rel_gap"]) < 1e-9


def test_missing_inconsistent_and_degenerate_logs():
    assert rs.summarize({"output_throughput": 5.0}, conc=1)["state"] == "not_recorded"
    bad = _raw(n=5)
    bad["ttfts"] = bad["ttfts"][:-1]
    assert rs.summarize(bad, conc=1)["state"] == "inconsistent"
    allfail = _raw(n=4, fail=(0, 1, 2, 3))
    assert rs.summarize(allfail, conc=1)["state"] == "no_successful_requests"
    small = rs.summarize(_raw(n=3), conc=1)
    assert small["state"] == "ok" and small["metrics"]["tpot_ms"]["p99"]["ci"] is None
    assert small["throughput_littles_law"]["ci"] is None
    one_tok = _raw(n=6, L=1)
    s = rs.summarize(one_tok, conc=1)
    assert s["metrics"]["tpot_ms"]["n"] == 0 and s["metrics"]["tpot_ms"]["mean"]["est"] is None
