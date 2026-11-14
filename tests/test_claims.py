import csv

from g4bench import claims
from g4bench.config import load_yaml
from g4bench.paths import CONFIGS

KINDS = {"measurable", "verified-source", "verified-analysis", "untestable"}


def test_register_is_well_formed():
    reg = load_yaml(CONFIGS / "claims.yaml")["claims"]
    assert len({c["id"] for c in reg}) == len(reg) >= 10
    for c in reg:
        assert c["kind"] in KINDS
        if c["kind"] == "untestable":
            assert c.get("reason")
        if c["kind"].startswith("verified"):
            assert c.get("evidence")


def test_no_results_means_not_measured_everywhere(tmp_path):
    md = claims.render(comparison_csv=tmp_path / "missing.csv")
    for line in md.splitlines():
        if line.startswith("| C0") and "| measurable |" in line:
            assert line.rstrip().endswith("| not measured |"), line
    assert "untestable (see reason)" in md and "verified (sources)" in md


def test_status_counts_only_exact_hardware_baseline_rows(tmp_path):
    p = tmp_path / "comparison.csv"
    rows = [
        {"reference": "vllm", "metric": "output_throughput", "key": "1024x1", "hardware_class": "RTX-PRO-6000x1",
         "class": "consistent", "n": "1"},
        {"reference": "vllm", "metric": "output_throughput", "key": "8192x4", "hardware_class": "L40Sx1",
         "class": "lower", "n": "3"},  # proxy hardware: ignored
    ]
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    md = claims.render(comparison_csv=p)
    c01 = next(line for line in md.splitlines() if line.startswith("| C01"))
    assert "measured 1/8 (1 consistent) - descriptive, n<3" in c01


def test_engine_ratio_claim_uses_paired_engine_comparison(tmp_path):
    e = tmp_path / "engine.csv"
    rows = [{"metric": "output_throughput", "dataset": "random", "isl": "1024", "osl": "1024", "conc": "1",
             "hardware_class": "RTX-PRO-6000-SEx1", "advantage": "1.39", "article_advantage": "1.52", "n_paired": "1"},
            {"metric": "output_throughput", "dataset": "random", "isl": "1024", "osl": "256", "conc": "4",
             "hardware_class": "RTX-PRO-6000-SEx1", "advantage": "0.9", "article_advantage": "1.4", "n_paired": "1"}]
    with open(e, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    md = claims.render(comparison_csv=tmp_path / "none.csv", engine_csv=e)
    c04 = next(line for line in md.splitlines() if line.startswith("| C04"))
    assert "measured 1/8: vLLM/SGLang 1.39-1.39x (article 1.52-1.52x) - descriptive, n<3" in c04  # OSL 256 ignored
