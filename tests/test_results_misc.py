import json

from g4bench import promscrape, results, roofline
from g4bench.cost import estimate
from g4bench.matrix import expand, load_matrix

RAW = {
    "duration": 10.0, "completed": 10, "total_input_tokens": 9000, "total_output_tokens": 9200,
    "request_throughput": 1.0, "output_throughput": 920.0, "total_token_throughput": 1820.0,
    "input_lens": [900] * 10, "output_lens": [920] * 10,
    "benchmark_outcome": {"status": "passed", "requested": 10, "completed": 10, "failed": 0},
    **{f"{s}_{m}_ms": 5.0 for m in ("ttft", "tpot", "itl", "e2el") for s in ("mean", "median", "p90", "p99")},
}


def test_normalize_validate_roundtrip(tmp_path):
    rec = results.normalize(RAW, context={"run_id": "r", "engine": "vllm", "isl": 1024, "conc": 1})
    assert rec["output_throughput"] == 920.0 and rec["speed_per_user_tok_s"] == 200.0
    assert rec["failure_rate"] == 0 and rec["mean_output_len"] == 920
    assert results.validate(rec, osl=1024, range_ratio=0.8) == []
    bad = dict(rec, mean_output_len=10.0)
    assert any("output len" in p for p in results.validate(bad, osl=1024, range_ratio=0.8))
    results.append_csv(tmp_path / "r.csv", rec)
    results.append_csv(tmp_path / "r.csv", rec)
    back = results.read_csv(tmp_path / "r.csv")
    assert len(back) == 2 and back[0]["output_throughput"] == 920.0 and back[0]["conc"] == 1


def test_failed_requests_flagged():
    raw = dict(RAW, completed=8, benchmark_outcome={"status": "failed", "requested": 10, "completed": 8})
    rec = results.normalize(raw, context={})
    assert rec["failed"] == 2 and rec["failure_rate"] == 0.2
    assert results.validate(rec, osl=1024, range_ratio=0.8)


def test_promscrape_parse_and_acceptance():
    txt = """# HELP x
vllm:spec_decode_num_accepted_tokens_total{model_name="m"} 300
vllm:spec_decode_num_drafts_total{model_name="m"} 100
vllm:spec_decode_num_draft_tokens_total{model_name="m"} 400
"""
    after = promscrape.parse(txt)
    d = promscrape.delta({}, after) or {k: v for k, v in after.items()}
    acc = promscrape.acceptance(d)
    assert acc["acceptance_length"] == 4.0 and acc["draft_acceptance_rate"] == 0.75
    log = "INFO SpecDecoding metrics: Mean acceptance length: 3.10, x\n... Mean acceptance length: 2.90,"
    assert abs(promscrape.acceptance_from_log(log)["acceptance_length_log_mean"] - 3.0) < 1e-9
    sgl = "Decode batch. accept len: 3.50, accept rate: 0.58, x\naccept len: 2.50, accept rate: 0.42,"
    assert promscrape.acceptance_from_log(sgl)["acceptance_length_log_mean"] == 3.0


def test_roofline_kv_math():
    # 40 sliding layers * 1024 * 8 heads * 256 * (K,V) * 2B + 8 global * ctx * 1 * 512 * 1 (k_eq_v) * 2B
    assert roofline.kv_bytes_per_seq(1024) == 40 * 1024 * 8 * 256 * 4 + 8 * 1024 * 512 * 2
    assert roofline.kv_bytes_per_seq(9216) - roofline.kv_bytes_per_seq(1024) == 8 * 8192 * 512 * 2
    g = roofline.GPUS["RTX-PRO-6000"]
    nospec = roofline.max_tokens_per_s(g, 1, 2048)
    assert 50 < nospec < 75  # 1.79 TB/s * 0.85 / ~24.3 GB
    assert roofline.max_tokens_per_s(g, 1, 2048, acceptance_len=3.0, draft_tokens=4) > 2 * nospec


def test_cost_estimate_monotone():
    m = load_matrix("backboard-repro")
    one = estimate(expand(m, seeds=[0]))
    three = estimate(expand(m))
    assert 0 < one.usd < three.usd < 60
