"""Offline: engine configs vs committed flag snapshots of the pinned engine sources."""
import json

from g4bench import flagcheck as fc
from g4bench.config import EngineConfig, load_pins, placeholder_values


def test_all_engine_configs_use_only_known_flags():
    res = fc.check_all()
    assert set(res) >= {"vllm-0.33.0-mtp4", "sglang-0.5.23-nextn", "vllm-0.33.0-nospec", "vllm-0.33.0-mtp4-textonly"}
    assert all(v == [] for v in res.values()), res


def test_typos_bad_spec_keys_and_negations_are_detected():
    cfg = EngineConfig.load("vllm-0.33.0-mtp4")
    snap = json.loads(fc.snapshot_path("vllm", "0.33.0").read_text())
    argv = cfg.render_args(placeholder_values(load_pins(), 10240, 8000))
    assert fc.check(cfg, snap, argv) == []
    assert fc.check(cfg, snap, argv + ["--gpu-memory-utilisation", "0.9"])  # British spelling
    assert fc.check(cfg, snap, ["--speculative-config", '{"num_spec_tokens": 4}'])
    assert fc.check(cfg, snap, ["--speculative-config", "{not json"])
    assert fc.check(cfg, snap, ["--no-enable-prefix-caching"]) == []
    assert fc.check(cfg, snap, ["--no-such-thing"])


def test_extractors_on_synthetic_sources():
    v = '''
def add_cli_args(parser):
    parser.add_argument("--alpha", type=int)
    g.add_argument("--beta-gamma", "-b")
class FrontendArgs:
    host: str | None = None
class SpeculativeConfig:
    method: str = "mtp"
    num_speculative_tokens: int = 1
'''
    flags, spec = fc.vllm_flags({"vllm/engine/arg_utils.py": v, "vllm/entrypoints/common/cli_args.py": v,
                                 "vllm/config/speculative.py": v})
    assert {"--alpha", "--beta-gamma", "-b", "--host"} <= flags and spec == {"method", "num_speculative_tokens"}
    s = '''
class ModelArgs:
    model_path: A[str, Arg(help="x", aliases=["--model"])]
    tp_size: A[int, Arg(help="y", aliases=["--tensor-parallel-size"])] = 1
    weird: A[int, Arg(help="z", cli_name="--odd-name")] = 0
    disable_radix_cache: A[bool, "help"] = False
'''
    got = fc.sglang_flags({"python/sglang/srt/arg_groups/fields/model.py": s})
    assert got == {"--model-path", "--model", "--tp-size", "--tensor-parallel-size", "--odd-name", "--disable-radix-cache"}


def test_snapshots_record_provenance():
    for eng, ver in (("vllm", "0.33.0"), ("sglang", "0.5.23")):
        d = json.loads(fc.snapshot_path(eng, ver).read_text())
        assert d["tag"] == f"v{ver}" and d["file_sha256"] and len(d["flags"]) > 100


def test_runner_refuses_bad_flag_before_creating_run_dir(tmp_path):
    import pytest

    from g4bench.runner import FlagError, RunOptions, run

    with pytest.raises(FlagError):
        run(RunOptions(engine="vllm-0.33.0-mtp4", matrix="smoke", results_root=tmp_path,
                       extra_server_args=["--gpu-memory-utilisation", "0.9"], dry_run=True))
    assert not any(tmp_path.iterdir())
    d = run(RunOptions(engine="vllm-0.33.0-mtp4", matrix="smoke", results_root=tmp_path, dry_run=True))
    assert (d / "manifest.json").exists()
