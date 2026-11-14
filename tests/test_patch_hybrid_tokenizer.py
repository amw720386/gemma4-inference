"""Image patch hybrid_tokenizer (src/g4bench/patches): anchors, idempotency, env gate, encode/decode routing of
HybridBackend, DecodeStream unwrapping, copy/pickle behaviour, registration. CPU only."""
import copy
import os
import pathlib
import pickle
import subprocess
import sys

import pytest

from g4bench.config import EngineConfig
from g4bench.images import VLLM_PATCHES, VLLM_PATCHES_EXP
from g4bench.paths import REPO
from g4bench.patches import apply as A
from g4bench.patches import hybrid_tokenizer as P

PATCH_DIR = REPO / "src" / "g4bench" / "patches"
CONFIG = "vllm-0.33.0-best-e8-pfa2-sync-hybtok"
BASE = "vllm-0.33.0-best-e8-pfa2-sync"

# Runnable reduction of vLLM v0.33.0 vllm/tokenizers/registry.py: the VLLM_USE_FASTOKENS block at the top of
# get_tokenizer (lines 218-223) and the ``cached_get_tokenizer`` line (295) are verbatim; the rest is stubbed. Set
# G4_VLLM_TOK_REGISTRY=<path to the pinned upstream file> to also run the anchor tests against the real file.
EXCERPT = '''\
def get_tokenizer(
    tokenizer_name,
    *args,
    **kwargs,
):
    """Gets a tokenizer for the given model name via HuggingFace or ModelScope."""
    if envs.VLLM_USE_FASTOKENS:
        # Process-global, idempotent patch that swaps the Rust BPE backend
        # of any HF fast tokenizer loaded afterwards. No-op for non-HF modes.
        from .fastokens import apply_fastokens_patch

        apply_fastokens_patch()

    return load(tokenizer_name)


cached_get_tokenizer = lru_cache(get_tokenizer)
'''


def _upstream() -> str:
    p = os.environ.get("G4_VLLM_TOK_REGISTRY")
    return pathlib.Path(p).read_text(encoding="utf-8") if p else EXCERPT


def test_anchors_once_idempotent_and_compiles():
    src = _upstream()
    for anchor, new in P.edits():
        assert src.count(anchor) == 1
        assert anchor in new
    out, changed = P.patch_text(src)
    assert changed and out.count(P.MARKER) == 2
    compile(out, "registry", "exec")
    again, changed2 = P.patch_text(out)
    assert not changed2 and again == out


def test_anchor_mismatch_raises():
    with pytest.raises(P.PatchError):
        P.patch_text("nothing here\n")
    out, _ = P.patch_text(EXCERPT)
    with pytest.raises(P.PatchError):
        P.patch_text(out.replace("    mod.install_global()\n", ""))


def _run_patched(monkeypatch, env_value, runtime_path):
    out, _ = P.patch_text(EXCERPT, runtime_path=str(runtime_path))
    calls = []

    class Envs:
        VLLM_USE_FASTOKENS = False

    ns = {"envs": Envs, "lru_cache": lambda f: f, "load": lambda name: calls.append(name) or name}
    exec(compile(out, "registry", "exec"), ns)
    if env_value is None:
        monkeypatch.delenv(P.ENV, raising=False)
    else:
        monkeypatch.setenv(P.ENV, env_value)
    sys.modules.pop("g4patch_runtime_" + P.NAME, None)
    return ns["get_tokenizer"]("tok"), calls


def test_gate_unset_is_upstream(monkeypatch, tmp_path):
    rt = tmp_path / "rt.py"
    rt.write_text("raise RuntimeError('runtime must not load when the gate is unset')\n")
    for v in (None, "0", "", "yes"):
        assert _run_patched(monkeypatch, v, rt) == ("tok", ["tok"])
    assert "g4patch_runtime_" + P.NAME not in sys.modules


def test_gate_on_calls_install_global_once_per_call(monkeypatch, tmp_path):
    rt = tmp_path / "rt.py"
    rt.write_text("N = []\ndef install_global():\n    N.append(1)\n")
    res, calls = _run_patched(monkeypatch, "1", rt)
    assert res == "tok" and calls == ["tok"]
    assert sys.modules["g4patch_runtime_" + P.NAME].N == [1]
    sys.modules.pop("g4patch_runtime_" + P.NAME, None)


# ------------------------------------------------------------------------------------------------- runtime (fakes)
class Enc:
    def __init__(self, src, text, kw):
        self.src, self.text, self.kw = src, text, kw


class FakeHF:
    """Picklable stand-in for tokenizers.Tokenizer."""

    def __init__(self):
        self.truncation = None
        self.padding = None
        self.encode_special_tokens = False
        self.vocab = 10

    def encode(self, seq, pair=None, is_pretokenized=False, add_special_tokens=True):
        return Enc("hf", seq, {"pair": pair, "pre": is_pretokenized})

    def encode_batch(self, input, is_pretokenized=False, add_special_tokens=True):
        return [Enc("hf", x, {}) for x in input]

    def decode(self, ids, skip_special_tokens=True):
        return "hf:" + ",".join(map(str, ids))

    def add_tokens(self, toks):
        self.vocab += len(toks)
        return len(toks)


class FakeShim:
    built = 0

    def __init__(self, hf):
        FakeShim.built += 1
        self.vocab = hf.vocab
        self.truncation = None
        self.padding = None
        self.encode_special_tokens = False
        self.sets = []

    def __setattr__(self, k, v):
        if k in ("truncation", "padding") and hasattr(self, "sets"):
            self.sets.append((k, v))
        object.__setattr__(self, k, v)

    def encode(self, seq, add_special_tokens=True):
        return Enc("fast", seq, {"trunc": self.truncation, "split": self.encode_special_tokens, "vocab": self.vocab})

    def encode_batch(self, inputs, add_special_tokens=True):
        return [self.encode(x) for x in inputs]

    encode_batch_fast = encode_batch


def _hb(hf=None):
    return P.HybridBackend(hf or FakeHF(), shim_cls=FakeShim)


def test_encode_routes_to_fastokens_everything_else_to_hf():
    hb = _hb()
    assert hb.encode("abc").src == "fast"
    assert [e.src for e in hb.encode_batch(["a", "b"])] == ["fast", "fast"]
    assert hb.encode_batch_fast(["a"])[0].src == "fast"
    assert hb.encode("a", "b").src == "hf"                       # pair
    assert hb.encode(["a", "b"], is_pretokenized=True).src == "hf"
    assert hb.encode_batch([("a", "b")])[0].src == "hf"
    assert hb.decode([1, 2]) == "hf:1,2"                         # decode never touches fastokens
    assert hb.vocab == 10                                        # attribute reads are the HF backend's


def test_truncation_padding_and_special_flag_are_synced():
    hf = FakeHF()
    hb = _hb(hf)
    assert hb.encode("x").kw["trunc"] is None
    hf.truncation = {"max_length": 5, "stride": 0, "strategy": "longest_first", "direction": "right"}
    hf.encode_special_tokens = True
    e = hb.encode("x")
    assert e.kw["trunc"] == hf.truncation and e.kw["split"] is True
    n = len(hb._g4_fast.sets)
    hb.encode("y")
    assert len(hb._g4_fast.sets) == n                            # unchanged state is not re-applied
    hf.truncation = None
    assert hb.encode("z").kw["trunc"] is None


def test_setattr_forwards_and_mutations_rebuild_encoder():
    hf = FakeHF()
    hb = _hb(hf)
    hb.added_token_ids = {1: "x"}                                # what vLLM's detokenizer stores
    assert hf.added_token_ids == {1: "x"} and hb.added_token_ids == {1: "x"}
    built = FakeShim.built
    hb.encode("a")
    assert FakeShim.built == built
    assert hb.add_tokens(["t1", "t2"]) == 2
    assert hb.encode("a").kw["vocab"] == 12 and FakeShim.built == built + 1
    hb.normalizer = object()
    hb.encode("a")
    assert FakeShim.built == built + 2


def test_copies_and_pickle():
    hf = FakeHF()
    hb = _hb(hf)
    shallow = copy.copy(hb)
    assert type(shallow) is P.HybridBackend and shallow._g4_hf is hf
    deep = copy.deepcopy(hb)
    assert type(deep) is P.HybridBackend and deep._g4_hf is not hf and deep._g4_fast is not hb._g4_fast
    assert deep.encode("q").src == "fast"
    back = pickle.loads(pickle.dumps(hb))
    assert type(back) is FakeHF                                   # other processes get the plain HF backend


def test_decode_stream_unwraps_hybrid():
    seen = []

    class Orig:
        def __init__(self, ids=None, skip_special_tokens=False):
            self.ids, self.skip = ids, skip_special_tokens

        def step(self, tokenizer, token_id):
            seen.append(tokenizer)
            return str(token_id)

    cls = P.make_decode_stream_cls(Orig)
    hf = FakeHF()
    s = cls(ids=[1, 2], skip_special_tokens=True)
    assert s.ids == [1, 2] and s.skip is True                    # attribute reads reach the original stream
    assert s.step(_hb(hf), 7) == "7" and seen[-1] is hf
    other = FakeHF()
    assert s.step(other, 8) == "8" and seen[-1] is other         # a plain backend passes through
    assert cls._g4_orig is Orig


def test_install_global_skips_when_full_fastokens_is_on(monkeypatch):
    monkeypatch.setenv("VLLM_USE_FASTOKENS", "1")
    monkeypatch.setattr(P, "_installed", False)
    P.install_global()                                           # must not import fastokens/transformers
    assert P._installed is True


def test_real_tokenizer_ids_and_decode_match(tokenizer_dir):
    """With the image's packages (tokenizers, fastokens, transformers v5): hybrid ids/text == HF's."""
    pytest.importorskip("fastokens")
    tokenizers = pytest.importorskip("tokenizers")
    hf = tokenizers.Tokenizer.from_file(str(pathlib.Path(tokenizer_dir) / "tokenizer.json"))
    hb = P.HybridBackend(hf)
    text = "Hello <start_of_turn>user\nwörld 123 \t  code: def f(x): return x**2<end_of_turn>"
    for add in (True, False):
        assert hb.encode(text, add_special_tokens=add).ids == hf.encode(text, add_special_tokens=add).ids
    ids = hf.encode(text).ids
    assert hb.decode(ids) == hf.decode(ids)
    ds_cls = P.make_decode_stream_cls(tokenizers.decoders.DecodeStream)
    a, b = ds_cls(skip_special_tokens=True), tokenizers.decoders.DecodeStream(skip_special_tokens=True)
    assert [a.step(hb, i) for i in ids] == [b.step(hf, i) for i in ids]


# ------------------------------------------------------------------------------------------------------ wiring
def test_registered_on_experimental_image_only():
    assert P.NAME in A.KNOWN
    assert P.NAME in VLLM_PATCHES_EXP and P.NAME not in VLLM_PATCHES


def test_config_is_sync_plus_gate():
    e, base = EngineConfig.load(CONFIG), EngineConfig.load(BASE)
    assert e.raw["image_patches"] == base.raw["image_patches"] + [P.NAME]
    assert e.env == {**base.env, P.ENV: "1"}
    assert "VLLM_USE_FASTOKENS" not in e.env
    assert e.args == base.args
    assert P.NAME not in base.raw["image_patches"]


def test_apply_script_patches_a_package_tree(tmp_path):
    pkg = tmp_path / "vllm"
    f = pkg / P.RELPATH
    f.parent.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    f.write_text(EXCERPT, encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": str(tmp_path)}
    cmd = [sys.executable, str(PATCH_DIR / "apply.py"), P.NAME]
    r = subprocess.run(cmd, cwd=tmp_path, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "patched" in r.stdout and P.MARKER in f.read_text(encoding="utf-8")
    r2 = subprocess.run(cmd, cwd=tmp_path, env=env, capture_output=True, text=True)
    assert r2.returncode == 0 and "already applied" in r2.stdout
    f.write_text("nothing to anchor\n", encoding="utf-8")
    r3 = subprocess.run(cmd, cwd=tmp_path, env=env, capture_output=True, text=True)
    assert r3.returncode != 0 and "FAILED" in r3.stderr
