"""Our client argv must contain every flag InferenceX's fixed_seq.client_argv sets, with the same values."""
import re
from pathlib import Path

from g4bench.client import client_argv
from g4bench.matrix import Point
from g4bench.paths import REPO


def _upstream_flags() -> set[str]:
    src = (REPO / "docs/sources/inferencex_fixed_seq.py.txt").read_text()
    body = src[src.index("def client_argv"):src.index("if point.tokenizer is not None")]  # unconditional flags
    return set(re.findall(r'"(--[a-z-]+)"', body))


def test_flags_superset_of_inferencex():
    p = Point(isl=1024, osl=1024, conc=4, seed=0, trial=0, num_prompts=40, num_warmups=8,
              random_range_ratio=0.8, use_chat_template=True)
    argv = client_argv(p, base_url="http://x:1", model="m", backend="vllm", result=Path("/tmp/r.json"))
    ours = {a for a in argv if a.startswith("--")}
    missing = _upstream_flags() - ours
    assert not missing, missing
    v = lambda f: argv[argv.index(f) + 1]  # noqa: E731
    assert v("--request-rate") == "inf" and "--ignore-eos" in argv
    assert v("--num-prompts") == "40" and v("--num-warmups") == "8" and v("--max-concurrency") == "4"
    assert v("--dataset-name") == "random" and v("--random-range-ratio") == "0.8"
    assert v("--percentile-metrics") == "ttft,tpot,itl,e2el"
    assert "--use-chat-template" in argv and v("--seed") == "0"


def test_vendored_checksums():
    import hashlib

    base = REPO / "third_party/inferencex"
    for line in (base / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split()
        assert hashlib.sha256((base / name).read_bytes()).hexdigest() == digest, name
