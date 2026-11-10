import json
import re

import pytest

from g4bench.config import EngineConfig, config_hash, load_pins, placeholder_values
from g4bench.matrix import expand, load_matrix, new_run_id


def test_engines_render():
    pins = load_pins()
    for name in ("vllm-0.33.0-mtp4", "sglang-0.5.23-nextn", "vllm-0.33.0-nospec", "mock-cpu"):
        e = EngineConfig.load(name)
        argv = e.render_args(placeholder_values(pins, 10240, e.port))
        assert all("{target" not in a and "{draft" not in a for a in argv)
        if e.engine != "mock":
            assert pins["model"]["target"]["revision"] in argv


def test_vllm_args_match_article():
    e = EngineConfig.load("vllm-0.33.0-mtp4")
    argv = e.render_args(placeholder_values(load_pins(), 10240, 8000))
    assert argv[argv.index("--gpu-memory-utilization") + 1] == "0.90"
    assert "--no-enable-prefix-caching" in argv and "--async-scheduling" in argv
    spec = json.loads(argv[argv.index("--speculative-config") + 1])
    assert spec["method"] == "mtp" and spec["num_speculative_tokens"] == 4
    assert spec["model"] == "google/gemma-4-12B-it-assistant"


def test_sglang_args_match_article():
    e = EngineConfig.load("sglang-0.5.23-nextn")
    argv = e.render_args(placeholder_values(load_pins(), 10240, 30000))
    get = lambda f: argv[argv.index(f) + 1]  # noqa: E731
    assert get("--speculative-algorithm") == "NEXTN"
    assert get("--speculative-num-steps") == "5" and get("--speculative-num-draft-tokens") == "6"
    assert get("--mem-fraction-static") == "0.85" and "--disable-radix-cache" in argv


def test_unresolved_placeholder_raises():
    e = EngineConfig.load("vllm-0.33.0-mtp4")
    with pytest.raises(KeyError):
        e.render_args({"target_repo": "x"})


def test_config_hash_sensitivity():
    pins = load_pins()
    m = load_matrix("backboard-repro")
    a = EngineConfig.load("vllm-0.33.0-mtp4")
    b = EngineConfig.load("vllm-0.33.0-nospec")
    assert config_hash(a, m, pins) == config_hash(a, m, pins)
    assert config_hash(a, m, pins) != config_hash(b, m, pins)
    m2 = dict(m, random_range_ratio=1.0)
    assert config_hash(a, m, pins) != config_hash(a, m2, pins)


def test_matrix_expansion():
    m = load_matrix("backboard-repro")
    pts = expand(m)
    assert len(pts) == 2 * 4 * 3
    assert len({p.point_id for p in pts}) == len(pts)
    for p in pts:
        assert p.num_prompts == 10 * p.conc and p.num_warmups == 2 * p.conc  # InferenceX policy
        assert p.osl == 1024 and p.use_chat_template
    assert [p.key for p in expand(m, seeds=[0], only_keys=["8192x4"])] == ["8192x4"]


def test_run_id_format():
    rid = new_run_id("vllm-0.33.0-mtp4", "RTX-PRO-6000x1", "abcdef123456")
    assert re.fullmatch(r"\d{8}T\d{6}Z-vllm-0\.33\.0-mtp4-rtx-pro-6000x1-abcdef12-[0-9a-f]{4}", rid)


def test_every_engine_config_declares_role_and_admissibility():
    from g4bench.bridge_api import ALLOWED_ENGINES
    from g4bench.paths import CONFIGS

    baselines = []
    for f in sorted((CONFIGS / "engines").glob("*.yaml")):
        e = EngineConfig.load(str(f))
        assert e.raw.get("role") in ("baseline", "variant"), f.name
        assert str(e.raw.get("admissibility")) in ("a", "b", "c"), f.name
        if e.raw["role"] == "baseline":
            baselines.append(e.name)
        if e.engine != "mock":
            assert e.name in ALLOWED_ENGINES, f"{e.name} not allowlisted on the bridge"
    assert sorted(baselines) == ["sglang-0.5.23-nextn", "vllm-0.33.0-mtp4"]
    assert EngineConfig.load("vllm-0.33.0-mtp4-draftfp8").raw["admissibility"] == "b"
