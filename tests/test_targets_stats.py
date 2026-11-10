import json
import math

import numpy as np
import pytest

from g4bench import compare, stats
from g4bench.inference_checks import check_fixed_length, summary
from g4bench.paths import TARGETS


def test_targets_shape():
    t = compare.load_targets()
    assert sorted(t) == sorted(f"{i}x{c}" for i in (1024, 8192) for c in (1, 4, 16, 64))
    for m in t.values():
        for metric in ("output_throughput", "p99_tpot", "p99_e2el", "p90_ttft", "speed_per_user"):
            assert set(m[metric]) == {"backboard", "vllm", "sglang"}


def test_transcription_matches_article_ratio_tables():
    """Recompute the article's ratio tables from the absolute tables (catches typos in our transcription)."""
    d = json.loads(TARGETS.read_text())
    cols = d["ratio_tables"]["_columns"]
    by_key = {f"{p['isl']}x{p['conc']}": p for p in d["points"]}
    checks = {"output_throughput": True, "p99_tpot": False, "p99_e2el": False, "p90_ttft": False}
    n = 0
    for comp in ("vllm", "sglang"):
        for key, row in d["ratio_tables"][f"vs_{comp}"].items():
            p = by_key[key]
            for metric, hib in checks.items():
                bb, other = p[metric]["backboard"], p[metric][comp]
                ratio = bb / other if hib else other / bb
                pub = row[cols.index(metric)]
                # absolute values are rounded (e.g. TPOT to 0.1 ms) -> allow rounding slack
                tol = 0.011 if metric != "p99_tpot" else 0.03
                assert abs(ratio - pub) <= tol + 1e-9, (comp, key, metric, ratio, pub)
                n += 1
    assert n == 2 * 8 * 4


def test_fixed_length_hypothesis_rejected():
    s = summary()
    assert s["fixed_length_violations"] >= 12  # 16/24 at time of writing
    assert s["c1_implausible"] == 0


def test_summarize_and_ci():
    s = stats.summarize([100.0, 102.0, 98.0])
    assert s.n == 3 and s.mean == pytest.approx(100.0)
    assert s.sd == pytest.approx(2.0)
    assert s.ci_lo == pytest.approx(100 - 4.303 * 2 / math.sqrt(3))
    one = stats.summarize([5.0])
    assert one.n == 1 and math.isnan(one.ci_lo)
    assert stats.summarize([]).n == 0


def test_bootstrap_ratio():
    est, lo, hi = stats.bootstrap_ratio([110, 112, 108], [100, 101, 99], seed=1)
    assert est == pytest.approx(110 / 100)
    assert lo < est < hi and lo > 1.0


def test_classify_and_beats():
    assert compare.classify(103, 99, 107, 100) == "consistent"
    assert compare.classify(80, 79, 81, 100) == "lower"
    assert compare.classify(120, 118, 122, 100) == "higher"
    assert compare.classify(108, 95, 121, 100) == "consistent"  # CI covers target
    assert compare.beats(110, 105, 115, 3, 100, True)
    assert not compare.beats(110, 99, 121, 3, 100, True)
    assert not compare.beats(110, 105, 115, 2, 100, True)  # n<3 never "beats"
    assert compare.beats(9, 8, 9.5, 3, 10, False)


def test_aggregate_groups_runs():
    rows = [{"engine_config": "e", "config_hash": "h", "hardware_class": "g", "isl": 1024, "osl": 1024,
             "conc": 1, "seed": s, "valid": True, "output_throughput": v} for s, v in enumerate([10.0, 12.0, 11.0])]
    rows.append(dict(rows[0], valid=False, output_throughput=999.0))
    agg = stats.aggregate(rows, ["output_throughput"])
    assert len(agg) == 1 and agg[0]["n_runs"] == 3 and agg[0]["n_invalid"] == 1
    assert agg[0]["output_throughput__mean"] == pytest.approx(11.0)


def test_bootstrap_percentile():
    x = np.arange(1000.0)
    est, lo, hi = stats.bootstrap_percentile(x, 50)
    assert lo <= est <= hi
