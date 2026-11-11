"""Image patch mtp_draft_head_q (src/g4bench/patches): anchors, idempotency, env/class guard, wiring, config, and a
pure-torch reference of scan -> top-k -> exact BF16 rescore. CPU only (torch tests skip when torch is missing)."""
import ast
import collections.abc
import json
import os
import pathlib
import subprocess
import sys
import types

import pytest

from g4bench.config import EngineConfig
from g4bench.paths import REPO
from g4bench.patches import apply as A
from g4bench.patches import mtp_draft_head_q as P

PATCH_DIR = REPO / "src" / "g4bench" / "patches"
CONFIG = "vllm-0.33.0-best-e6b-dhq"
RUNTIME_MOD = f"g4patch_runtime_{P.NAME}"

# Runnable reduction of vLLM v0.33.0 vllm/model_executor/models/gemma4_mtp.py: the three patch anchors (end of
# Gemma4MultiTokenPredictor.forward + ``@support_torch_compile class Gemma4MTP`` at 436-440, get_top_tokens 583-588,
# load_weights 594-600) are verbatim; the rest is stubbed. Set G4_VLLM_GEMMA4_MTP=<path to the pinned upstream file>
# to also run the anchor tests against the real 600-line file.
EXCERPT = '''\
def support_torch_compile(cls):
    return cls


class Gemma4MultiTokenPredictor(nn.Module):
    def forward(self, draft_hidden_states, backbone_hidden_states):
        return draft_hidden_states, backbone_hidden_states


@support_torch_compile
class Gemma4MTP(nn.Module):
    """Gemma4 Multi-Token Prediction model for speculative decoding."""

    def __init__(self, suppress_token_ids=None, masked_embedding=None):
        self._suppress_token_ids = suppress_token_ids
        self.masked_embedding = masked_embedding
        self._suppress_idx = None

    def parameters(self):
        return iter([types.SimpleNamespace(device="cpu")])

    def _get_full_lm_head_weight(self):
        return "W"

    def get_top_tokens(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Sparse argmax via centroids masking. Returns token IDs directly."""
        assert self.masked_embedding is not None
        return self.masked_embedding.get_top_tokens(
            hidden_states,
            self._get_full_lm_head_weight(),
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        self._stable_full_lm_head_weight = None
        loaded = {name for name, _ in weights}
        if self._suppress_token_ids:
            self._suppress_idx = torch.tensor(
                self._suppress_token_ids,
                dtype=torch.long,
                device=next(self.parameters()).device,
            )
        return loaded
'''


def _upstream() -> str:
    p = os.environ.get("G4_VLLM_GEMMA4_MTP")
    return pathlib.Path(p).read_text(encoding="utf-8") if p else EXCERPT


def _ns(src: str) -> dict:
    fake_torch = types.SimpleNamespace(Tensor=object, long="long", tensor=lambda x, **kw: ("idx", tuple(x)))
    ns: dict = {"nn": types.SimpleNamespace(Module=object), "torch": fake_torch, "types": types,
                "Iterable": collections.abc.Iterable}
    exec(compile(src, "gemma4_mtp", "exec"), ns)
    return ns


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv(P.ENV_FORMAT, raising=False)
    monkeypatch.delenv(P.ENV_K, raising=False)
    monkeypatch.delitem(sys.modules, RUNTIME_MOD, raising=False)


# ------------------------------------------------------------------------------------------------- text patch
def test_applies_exactly_once_and_is_idempotent():
    src = _upstream()
    out, changed = P.patch_text(src)
    assert changed and out != src
    for _, new in P.edits():
        assert out.count(new) == 1
    assert out.count(P.MARKER) == 3
    assert f"_G4_DHQ_RUNTIME = {P.RUNTIME_PATH!r}" in out
    ast.parse(out)
    compile(out, "patched", "exec")
    again, changed2 = P.patch_text(out)
    assert again == out and not changed2
    # Only insertions: undoing each replacement gives back the upstream file byte for byte.
    back = out
    for anchor, new in P.edits():
        back = back.replace(new, anchor)
    assert back == src


def test_refuses_missing_or_duplicated_anchor():
    with pytest.raises(P.PatchError, match="0 times"):
        P.patch_text(EXCERPT.replace("@support_torch_compile\nclass Gemma4MTP", "class Gemma4MTP"))
    with pytest.raises(P.PatchError, match="0 times"):
        P.patch_text(EXCERPT.replace("        assert self.masked_embedding is not None\n", ""))
    with pytest.raises(P.PatchError, match="0 times"):
        P.patch_text(EXCERPT.replace("        return loaded\n", "        return set(loaded)\n"))
    with pytest.raises(P.PatchError, match="2 times"):
        P.patch_text(EXCERPT + "\n\nclass X:\n    def f(self):\n        x(\n" + P.LOAD_ANCHOR)
    out, _ = P.patch_text(EXCERPT)
    with pytest.raises(P.PatchError, match="marker"):  # marker present but a patched block was edited away
        P.patch_text(out.replace("            return g4_dhq.top_tokens(hidden_states)\n", "            pass\n"))
    with pytest.raises(P.PatchError, match="marker"):  # applied from a different runtime location
        P.patch_text(out, runtime_path="/elsewhere/mtp_draft_head_q.py")


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
    text = f.read_text(encoding="utf-8")
    assert "patched" in r.stdout and P.MARKER in text
    # The runtime path baked in is this patch file as apply.py loaded it (/opt/g4patches/... in the image).
    assert repr(str((PATCH_DIR / f"{P.NAME}.py").resolve())) in text
    r2 = subprocess.run(cmd, cwd=tmp_path, env=env, capture_output=True, text=True)
    assert r2.returncode == 0 and "already applied" in r2.stdout
    f.write_text("nothing to anchor\n", encoding="utf-8")
    r3 = subprocess.run(cmd, cwd=tmp_path, env=env, capture_output=True, text=True)
    assert r3.returncode != 0 and "FAILED" in r3.stderr
    assert P.NAME in A.KNOWN


# ------------------------------------------------------------------------------------------- guard and wiring
def _patched_ns(runtime_path: str = P.RUNTIME_PATH) -> dict:
    return _ns(P.patch_text(EXCERPT, runtime_path=runtime_path)[0])


@pytest.mark.parametrize("value", [None, "", "bf16", "fp16", "nvfp4", "1"])
def test_inactive_without_valid_env_is_upstream(monkeypatch, value):
    if value is not None:
        monkeypatch.setenv(P.ENV_FORMAT, value)
    ns = _patched_ns(runtime_path="/nonexistent/never_imported.py")  # importing it would raise
    G = ns["Gemma4MTP"]
    m = G(suppress_token_ids=[258883, 258882])
    assert m.load_weights([("a", 1)]) == {"a"}
    assert m._g4_dhq is None and m._suppress_idx == ("idx", (258883, 258882))
    assert RUNTIME_MOD not in sys.modules
    with pytest.raises(AssertionError):  # upstream: get_top_tokens only serves the centroid-masked head
        m.get_top_tokens("h")
    masked = types.SimpleNamespace(get_top_tokens=lambda h, w: ("masked", h, w))
    m2 = G(masked_embedding=masked)
    m2.load_weights([])
    assert m2.get_top_tokens("h") == ("masked", "h", "W")
    # Never loaded (e.g. load_format=dummy): no attribute, still upstream.
    assert G(masked_embedding=masked).get_top_tokens("h") == ("masked", "h", "W")


@pytest.mark.parametrize("value", ["fp8", "int4", " INT4 "])
def test_valid_env_routes_get_top_tokens_through_runtime(monkeypatch, value):
    monkeypatch.setenv(P.ENV_FORMAT, value)
    built = []

    class Fake:
        def top_tokens(self, h):
            return ("dhq", h)

    fake_rt = types.SimpleNamespace(build_for=lambda model: built.append(model) or Fake())
    monkeypatch.setitem(sys.modules, RUNTIME_MOD, fake_rt)
    ns = _patched_ns()
    m = ns["Gemma4MTP"](suppress_token_ids=[258883, 258882])
    m.load_weights([])
    assert built == [m] and m.get_top_tokens("h") == ("dhq", "h")
    assert m._suppress_idx is not None  # upstream load_weights body still ran first


def test_valid_env_loads_this_file_as_runtime(monkeypatch):
    monkeypatch.setenv(P.ENV_FORMAT, "fp8")
    ns = _patched_ns()
    other = type("EagleModel", (), {})()  # class guard: only Gemma4MTP is ever served
    assert ns["_g4_dhq_build"](other) is None
    rt = sys.modules[RUNTIME_MOD]
    assert rt.__file__ == P.RUNTIME_PATH and rt.NAME == P.NAME and rt.FORMATS == P.FORMATS


def test_build_for_guards_class_and_env(monkeypatch):
    G = type("Gemma4MTP", (), {})
    assert P.build_for(G()) is None  # env unset
    monkeypatch.setenv(P.ENV_FORMAT, "fp8")
    for name in ("Gemma4MTPSubclass", "DeepSeekMTP", "EagleLlamaForCausalLM"):
        assert P.build_for(type(name, (), {})()) is None


def test_env_parsing(monkeypatch):
    assert P.requested_format({}) is None
    assert P.requested_format({P.ENV_FORMAT: "FP8"}) == "fp8" and P.requested_format({P.ENV_FORMAT: "int4"}) == "int4"
    assert P.requested_format({P.ENV_FORMAT: "int8"}) is None
    assert P.requested_k({}) == 32 and P.requested_k({P.ENV_K: "16"}) == 16
    assert P.requested_k({P.ENV_K: "0"}) == 1 and P.requested_k({P.ENV_K: "1000"}) == P.MAX_K
    with pytest.raises(ValueError):
        P.requested_k({P.ENV_K: "lots"})
    assert [P.block_m(m) for m in (1, 16, 17, 32, 33, 64, 65, 512)] == [16, 16, 32, 32, 64, 64, 64, 64]


@pytest.mark.parametrize("modname", ["x", f"g4patch_{P.NAME}"])
def test_module_loads_without_torch_and_apply_never_imports_it(modname):
    # Loadable without torch/triton (CPU tests, image build); as apply.py loads it, it does not even try.
    code = ("import sys, importlib.util as u; "
            + ("sys.modules['torch'] = None; sys.modules['triton'] = None; " if modname == "x" else "")
            + f"s = u.spec_from_file_location({modname!r}, {P.RUNTIME_PATH!r}); m = u.module_from_spec(s); "
            "s.loader.exec_module(m); assert m.torch is None and m.triton is None; "
            "assert 'torch' not in sys.modules or sys.modules['torch'] is None; print(m.patch_text.__name__)")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0 and "patch_text" in r.stdout, r.stderr


# --------------------------------------------------------------------------------------------------- config
def test_engine_config_is_e6b_plus_patch_and_local_argmax():
    e, base = EngineConfig.load(CONFIG), EngineConfig.load("vllm-0.33.0-best-e6b")
    assert e.raw["image_patches"] == ["mtp_draft_prefill_last", P.NAME]
    assert all(p in A.KNOWN for p in e.raw["image_patches"])
    assert e.env[P.ENV_FORMAT] in P.FORMATS and P.requested_k(e.env) == 32
    assert e.raw["role"] == "variant" and e.raw["admissibility"] == "b"

    def spec(args):
        flat = [x for a in args for x in a]
        return json.loads(flat[flat.index("--speculative-config") + 1])

    s, sb = spec(e.args), spec(base.args)
    assert s.pop("use_local_argmax_reduction") is True and s == sb
    strip = lambda args: [a for a in args if a[0] != "--speculative-config"]  # noqa: E731
    assert strip(e.args) == strip(base.args)
    assert "draft_sample_method" not in s and "enable_adaptive_verification" not in s  # both conflict (vLLM raises)


# ---------------------------------------------------------------------------------------- torch reference
def _torch():  # per-test import so the rest of this module runs without torch
    return pytest.importorskip("torch")


def _head(V=2048, D=256, seed=0):
    t = _torch()
    g = t.Generator().manual_seed(seed)
    return (t.randn(V, D, generator=g) * 0.03).to(t.bfloat16), g


def _easy_h(W, toks, g, noise=0.3):
    """Activations pointing at token rows (clear argmax) plus noise; unit RMS like a post-norm hidden state."""
    t = _torch()
    e = W[toks].float()
    e = e / e.pow(2).mean(1, keepdim=True).sqrt()
    return (e + noise * t.randn(e.shape, generator=g)).to(t.bfloat16)


@pytest.mark.parametrize("fmt", P.FORMATS)
def test_quantize_roundtrip_and_layout(fmt):
    t = _torch()
    W, _ = _head()
    w = P.quantize(fmt, W, chunk=512)  # several chunks
    deq = P.dequantize(fmt, w)
    rel = float((deq - W.float()).norm() / W.float().norm())
    assert rel < {"fp8": 0.05, "int4": 0.2}[fmt], rel
    assert t.allclose(P.dequantize(fmt, w, 100, 300), deq[100:300])
    if fmt == "int4":
        assert w["q"].shape == (2048, 128) and w["q"].dtype == t.uint8 and w["s"].shape == (2, 2048)
        # byte g*64+j of row n: low nibble = k g*128+j, high nibble = k g*128+64+j (offset-binary q+8).
        n, g, j = 7, 1, 5
        b = int(w["q"][n, g * 64 + j])
        s = float(w["s"][g, n])
        assert deq[n, g * 128 + j] == ((b & 15) - 8) * s and deq[n, g * 128 + 64 + j] == ((b >> 4) - 8) * s
    else:
        assert w["q"].dtype == t.float8_e4m3fn and w["s"].shape == (2048,)


@pytest.mark.parametrize("fmt", P.FORMATS)
def test_scan_keeps_block_maxima_and_suppresses(fmt):
    t = _torch()
    W, g = _head()
    w = P.quantize(fmt, W)
    h = t.randn(5, 256, generator=g).to(t.bfloat16)
    sup = (1000, 999)
    ov, oi = P.scan_torch(fmt, w, h, 2048, sup, chunk=384)  # chunk rounds down to a block multiple
    assert ov.shape == oi.shape == (5, 16 * P.TOPT)
    ref = h.float() @ P.dequantize(fmt, w).t()
    ref[:, list(sup)] = float("-inf")
    blk = ref.view(5, 16, 128)
    assert t.allclose(ov.view(5, 16, P.TOPT)[:, :, 0], blk.max(-1).values, atol=1e-5)
    assert t.allclose(t.gather(ref, 1, oi), ov, atol=1e-5)  # ids point at the scored rows
    assert ((oi // 128) == t.arange(16).repeat_interleave(P.TOPT)[None, :]).all()  # T ids per own block
    assert not t.isin(oi, t.tensor(sup)).any()


@pytest.mark.parametrize("fmt", P.FORMATS)
@pytest.mark.parametrize("k", [8, 32])
def test_easy_cases_match_bf16_argmax(fmt, k):
    t = _torch()
    W, g = _head()
    toks = t.randint(0, 2048, (64,), generator=g)
    h = _easy_h(W, toks, g)
    sup = (2047, 2046)
    dhq = P.DraftHeadQ(W, fmt, k=k, suppress=sup)
    assert not dhq.use_triton and dhq.k == k
    got = dhq.top_tokens(h)
    ref = P.reference_argmax(h, W, sup)
    assert got.dtype == t.long and got.shape == (64,)
    assert t.equal(got, ref)
    hit = toks[~t.isin(toks, t.tensor(sup))]
    assert t.equal(got[~t.isin(toks, t.tensor(sup))], hit)  # easy: the pointed-at token wins
    assert t.equal(dhq.top_tokens(h.view(8, 8, 256)), got)  # [.., D] inputs are flattened like logits rows
    assert dhq.top_tokens(h[:0]).shape == (0,)


@pytest.mark.parametrize("fmt", P.FORMATS)
def test_suppressed_ids_never_drafted(fmt):
    t = _torch()
    W, g = _head()
    sup = (1000, 999)
    h = _easy_h(W, t.tensor([1000, 999, 1000, 5]), g, noise=0.05)
    got = P.DraftHeadQ(W, fmt, suppress=sup).top_tokens(h)
    assert not t.isin(got, t.tensor(sup)).any()
    assert t.equal(got, P.reference_argmax(h, W, sup))
    # Without suppression the same rows pick the suppressed ids (the test is not vacuous).
    assert t.equal(P.DraftHeadQ(W, fmt).top_tokens(h)[:3], t.tensor([1000, 999, 1000]))


@pytest.mark.parametrize("fmt", P.FORMATS)
def test_exact_ties_go_to_lowest_token_id(fmt):
    t = _torch()
    W, g = _head()
    W[700] = W[5]   # identical rows in different vocab blocks -> identical exact scores
    W[1500] = W[5]
    h = _easy_h(W, t.tensor([5, 700, 1500]), g, noise=0.0)
    got = P.DraftHeadQ(W, fmt).top_tokens(h)
    assert got.tolist() == [5, 5, 5] and t.equal(got, P.reference_argmax(h, W))


@pytest.mark.parametrize("fmt,min_agree", [("fp8", 0.97), ("int4", 0.9)])
def test_random_activations_mostly_agree(fmt, min_agree):
    t = _torch()
    W, g = _head(V=4096, D=256, seed=3)
    h = t.randn(256, 256, generator=g).to(t.bfloat16)
    got = P.DraftHeadQ(W, fmt, k=32).top_tokens(h)
    ref = P.reference_argmax(h, W)
    assert float((got == ref).float().mean()) >= min_agree
    # Whatever wins is exactly scored: never worse than the BF16 score of the reference token.
    ex = (h.float() @ W.float().t()).to(t.bfloat16).float()
    assert (ex.gather(1, got[:, None]) <= ex.gather(1, ref[:, None])).all()


def test_rescore_is_exact_bf16_on_given_candidates():
    t = _torch()
    W, g = _head()
    h = t.randn(4, 256, generator=g).to(t.bfloat16)
    full = P.reference_argmax(h, W)
    cands = t.stack([t.tensor([3, int(full[i]), 9, 4000 % 2048]) for i in range(4)])
    assert t.equal(P.rescore_torch(h, W, cands), full)


def _fake_model(W, **over):
    Gemma4MTP = type("Gemma4MTP", (), {})
    m = Gemma4MTP()
    m.masked_embedding = None
    m.lm_head = types.SimpleNamespace(weight=W, tp_size=1, org_vocab_size=W.shape[0])
    m.logits_processor = types.SimpleNamespace(soft_cap=None, scale=1.0)
    m._suppress_token_ids = [2047, 2046]
    for k, v in over.items():
        setattr(m, k, v)
    return m


def test_build_for_on_cpu_model(monkeypatch):
    t = _torch()
    W, g = _head()
    monkeypatch.setenv(P.ENV_FORMAT, "int4")
    monkeypatch.setenv(P.ENV_K, "16")
    dhq = P.build_for(_fake_model(W))
    assert dhq is not None and dhq.fmt == "int4" and dhq.k == 16 and dhq.suppress == (2047, 2046)
    assert dhq.weight.data_ptr() == W.data_ptr()  # the BF16 head is kept, not copied (rescoring reads it)
    h = _easy_h(W, t.randint(0, 2046, (16,), generator=g), g)
    assert t.equal(dhq.top_tokens(h), P.reference_argmax(h, W, (2047, 2046)))
    monkeypatch.setenv(P.ENV_FORMAT, "fp8")
    for over, msg in ((dict(masked_embedding=object()), "centroid"),
                      (dict(lm_head=types.SimpleNamespace(weight=W, tp_size=2)), "tensor-parallel"),
                      (dict(logits_processor=types.SimpleNamespace(soft_cap=30.0, scale=1.0)), "soft-cap"),
                      (dict(lm_head=types.SimpleNamespace(weight=W.float(), tp_size=1)), "BF16"),
                      (dict(_suppress_token_ids=[1, 2, 3]), "at most 2")):
        with pytest.raises((RuntimeError, ValueError), match=msg):
            P.build_for(_fake_model(W, **over))
