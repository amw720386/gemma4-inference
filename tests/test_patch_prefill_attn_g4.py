"""Image patch prefill_attn_g4 (src/g4bench/patches): anchors, idempotency, coexistence with splitk_per_group, the
host batch split, and a pure-torch run of the PATCHED forward/build text on a paged mixed batch (custom rows via the
torch fallback, the rest via a torch stand-in for unified_attention). CPU only; torch tests skip without torch."""
import ast
import math
import os
import pathlib
import subprocess
import sys
import types

import pytest

from g4bench.config import EngineConfig
from g4bench.paths import REPO
from g4bench.patches import apply as A
from g4bench.patches import prefill_attn_g4 as P
from g4bench.patches import splitk_per_group as S

PATCH_DIR = REPO / "src" / "g4bench" / "patches"
CONFIG = "vllm-0.33.0-best-e8-pfa"
BASE = "vllm-0.33.0-best-e8-all"

# Runnable reduction of vLLM v0.33.0 vllm/v1/attention/backends/triton_attn.py. Verbatim: line 54 (logger), 62-93
# (TritonAttentionMetadata), 289-300 (end of build()), 683-736 (end of forward(): the unified_attention call); the
# class/def lines around them and the first statement of build() are stubs. Set G4_VLLM_TRITON_ATTN=<path to the
# pinned upstream file> to also run the anchor tests against the real 885-line file.
EXCERPT = '''\
logger = init_logger(__name__)


@dataclass
class TritonAttentionMetadata:
    # NOTE(sang): Definition of context_len, query_len, and seq_len.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ---------------------|
    #                                   |-- query_len ---|

    num_actual_tokens: int  # Number of tokens excluding padding.
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor

    seq_threshold_3D: int
    num_par_softmax_segments: int
    softmax_segm_output: torch.Tensor
    softmax_segm_max: torch.Tensor
    softmax_segm_expsum: torch.Tensor

    causal: bool | torch.Tensor

    # Optional aot scheduling
    scheduler_metadata: torch.Tensor | None = None
    mm_prefix_range: dict[int, list[tuple[int, int]]] | None = None
    mm_prefix_range_tensor: torch.Tensor | None = None
    rswa_prefix_lens: torch.Tensor | None = None
    rswa_window: int | None = None


class TritonAttentionMetadataBuilder:
    rswa_window = None

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        attn_metadata = _stub_metadata(TritonAttentionMetadata, common_attn_metadata)

        rswa_prefix_lens = common_attn_metadata.rswa_prefix_lens
        if self.rswa_window is not None and rswa_prefix_lens is not None:
            assert self.persistent_rswa_prefix_lens is not None
            rswa_prefix_lens = rswa_prefix_lens.to(
                device=self.device, dtype=torch.int32, non_blocking=True
            )
            persistent_prefix_lens = self.persistent_rswa_prefix_lens[:num_reqs]
            persistent_prefix_lens.copy_(rswa_prefix_lens[:num_reqs])
            attn_metadata.rswa_prefix_lens = persistent_prefix_lens
            attn_metadata.rswa_window = self.rswa_window

        return attn_metadata


class TritonAttentionImpl:
    def forward(self, layer, query, key_cache, value_cache, output, attn_metadata, num_actual_tokens, q_descale,
                k_descale, v_descale, k_scale_cache=None, v_scale_cache=None, output_scale=None):
        cu_seqlens_q = attn_metadata.query_start_loc
        seqused_k = attn_metadata.seq_lens
        max_seqlen_q = attn_metadata.max_query_len
        max_seqlen_k = attn_metadata.max_seq_len
        block_table = attn_metadata.block_table

        seq_threshold_3D = attn_metadata.seq_threshold_3D
        num_par_softmax_segments = attn_metadata.num_par_softmax_segments
        softmax_segm_output = attn_metadata.softmax_segm_output
        softmax_segm_max = attn_metadata.softmax_segm_max
        softmax_segm_expsum = attn_metadata.softmax_segm_expsum

        mm_prefix_range_tensor = attn_metadata.mm_prefix_range_tensor

        unified_attention(
            q=query[:num_actual_tokens],
            k=key_cache,
            v=value_cache,
            out=output[:num_actual_tokens],
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            seqused_k=seqused_k,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.scale,
            causal=attn_metadata.causal,
            alibi_slopes=self.alibi_slopes,
            use_alibi_sqrt=self.use_alibi_sqrt,
            window_size=self.sliding_window,
            block_table=block_table,
            softcap=self.logits_soft_cap,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            seq_threshold_3D=seq_threshold_3D,
            num_par_softmax_segments=num_par_softmax_segments,
            softmax_segm_output=softmax_segm_output,
            softmax_segm_max=softmax_segm_max,
            softmax_segm_expsum=softmax_segm_expsum,
            sinks=self.sinks,
            output_scale=output_scale,
            mm_prefix_range=mm_prefix_range_tensor,
            rswa_prefix_lens=attn_metadata.rswa_prefix_lens,
            rswa_window=attn_metadata.rswa_window,
            kv_quant_mode=self._kv_quant_mode,
            k_scale_cache=k_scale_cache,
            v_scale_cache=v_scale_cache,
            chunk_lookback=self.chunk_lookback,
            use_td=self.use_td,
            mm_prefix_clamp_sliding_window=getattr(
                layer, "mm_prefix_clamp_sliding_window", False
            ),
        )

        return output
'''
# Verbatim slices of EXCERPT (checked against the real file when G4_VLLM_TRITON_ATTN is set).
VERBATIM = (
    EXCERPT.split("\n\n\n@dataclass")[0] + "\n",
    "@dataclass" + EXCERPT.split("@dataclass", 1)[1].split("\n\n\nclass TritonAttentionMetadataBuilder")[0] + "\n",
    EXCERPT.split("common_attn_metadata)\n\n", 1)[1].split("\n\n\nclass TritonAttentionImpl")[0] + "\n",
    EXCERPT.split("v_scale_cache=None, output_scale=None):\n", 1)[1],
)


def _upstream() -> str:
    p = os.environ.get("G4_VLLM_TRITON_ATTN")
    return pathlib.Path(p).read_text(encoding="utf-8") if p else EXCERPT


PATCHED, _ = P.patch_text(EXCERPT)


# --- patch mechanics -----------------------------------------------------------------------------------------------


def test_applies_exactly_once_and_is_idempotent():
    src = _upstream()
    out, changed = P.patch_text(src)
    assert changed and out != src
    for anchor, new in P.edits():
        assert src.count(anchor) == 1 and out.count(new) == 1
    ast.parse(out)
    again, changed2 = P.patch_text(out)
    assert again == out and not changed2
    assert out.count(P.MARKER) == 4 and repr(P.RUNTIME_PATH) in out
    if src is not EXCERPT:
        for piece in VERBATIM:
            assert src.count(piece) == 1, piece[:80]


def test_coexists_with_splitk_per_group_in_either_order():
    src = _upstream()
    if src is EXCERPT:  # splitk's anchors live in the builder __init__, which the excerpt leaves out
        pytest.skip("needs G4_VLLM_TRITON_ATTN (the full pinned file)")
    a = S.patch_text(P.patch_text(src)[0])[0]
    b = P.patch_text(S.patch_text(src)[0])[0]
    assert a == b
    ast.parse(a)
    assert P.patch_text(a) == (a, False) and S.patch_text(a) == (a, False)


def test_refuses_missing_or_duplicated_anchor():
    with pytest.raises(P.PatchError, match="0 times"):
        P.patch_text(EXCERPT.replace("            q=query[:num_actual_tokens],", "            q=query,"))
    with pytest.raises(P.PatchError, match="0 times"):
        P.patch_text(EXCERPT.replace("    rswa_window: int | None = None\n", "    rswa_window: int = 0\n"))
    with pytest.raises(P.PatchError, match="0 times"):
        P.patch_text(EXCERPT.replace("            attn_metadata.rswa_window = self.rswa_window\n\n",
                                     "            attn_metadata.rswa_window = self.rswa_window\n"))
    with pytest.raises(P.PatchError, match="2 times"):
        P.patch_text(EXCERPT + "\n" + P.LOGGER_ANCHOR)
    with pytest.raises(P.PatchError, match="marker"):
        P.patch_text(PATCHED.replace("                block_table = block_table[: g4_rest.num_reqs]\n", ""))


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
    assert repr(str(PATCH_DIR / "prefill_attn_g4.py")) in text  # runtime path embedded at apply time
    r2 = subprocess.run(cmd, cwd=tmp_path, env=env, capture_output=True, text=True)
    assert r2.returncode == 0 and "already applied" in r2.stdout
    f.write_text("nothing to anchor\n", encoding="utf-8")
    r3 = subprocess.run(cmd, cwd=tmp_path, env=env, capture_output=True, text=True)
    assert r3.returncode != 0 and "FAILED" in r3.stderr


@pytest.mark.parametrize("modname", ["g4patch_prefill_attn_g4", "x"])
def test_module_loads_without_torch_and_apply_never_imports_it(modname):
    code = ("import sys, importlib.util; "
            + ("sys.modules['torch'] = None; sys.modules['triton'] = None; " if modname == "x" else "")
            + f"s = importlib.util.spec_from_file_location({modname!r}, {str(PATCH_DIR / 'prefill_attn_g4.py')!r}); "
            "m = importlib.util.module_from_spec(s); sys.modules[s.name] = m; "
            "s.loader.exec_module(m); assert m.torch is None and m.triton is None; "
            "assert 'torch' not in sys.modules or sys.modules['torch'] is None; print(m.patch_text.__name__)")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0 and "patch_text" in r.stdout, r.stderr


def test_registered_and_config():
    assert P.NAME in A.KNOWN
    e, base = EngineConfig.load(CONFIG), EngineConfig.load(BASE)
    assert e.raw["image_patches"] == base.raw["image_patches"] + [P.NAME]
    assert list(e.args) == list(base.args)
    extra = {k: v for k, v in e.env.items() if k not in base.env}
    assert {k: e.env[k] for k in base.env} == dict(base.env)
    # + side-installed cuBLAS 13.10 (D-061): the experimental image no longer upgrades cuBLAS in place
    assert extra == {P.ENV: "1", P.ENV_GLOBAL: e.env[P.ENV_GLOBAL], P.ENV_SLIDING: e.env[P.ENV_SLIDING],
                     "G4_CUBLAS": "13.10"}
    assert P.variant_for(0, e.env).name == e.env[P.ENV_GLOBAL]
    assert P.variant_for(1024, e.env).name == e.env[P.ENV_SLIDING]
    assert P.variant_fits(P.variant_for(0, e.env), 512, 16) and P.variant_fits(P.variant_for(1024, e.env), 256, 2)


def test_variant_tables_match_the_kernel_lab():
    # kernels/prefill_attn.py imports triton at module level: compare the _v(...) literals instead of importing it.
    def calls(path, table):
        tree = ast.parse(pathlib.Path(path).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == table for t in node.targets):
                return [tuple(ast.literal_eval(a) for a in c.args) for c in ast.walk(node.value)
                        if isinstance(c, ast.Call) and getattr(c.func, "id", "") == "_v"]
        raise AssertionError(table)
    for table in ("GLOBAL_VARIANTS", "SLIDING_VARIANTS"):
        lab = calls(REPO / "kernels" / "prefill_attn.py", table)
        assert lab and lab == calls(PATCH_DIR / "prefill_attn_g4.py", table)
    assert P.DEFAULT_GLOBAL in P.GLOBAL_VARIANTS and P.DEFAULT_SLIDING in P.SLIDING_VARIANTS


def test_variant_env_and_fit():
    assert P.variant_for(0, {}).name == P.DEFAULT_GLOBAL and P.variant_for(1024, {}).name == P.DEFAULT_SLIDING
    with pytest.raises(ValueError, match=P.ENV_SLIDING):
        P.variant_for(1024, {P.ENV_SLIDING: "nope"})
    for v in P.GLOBAL_VARIANTS.values():
        assert P.variant_fits(v, 512, 16)
    for v in P.SLIDING_VARIANTS.values():
        assert P.variant_fits(v, 256, 2)
    assert not P.variant_fits(P.GLOBAL_VARIANTS["m16_dv512_n32_q"], 512, 32)   # block_m not a multiple of G
    assert not P.variant_fits(P.SLIDING_VARIANTS["m64_dv256_n32_q"], 128, 2)   # head_dim != Q tile
    assert P.requested({P.ENV: "1"}) and not P.requested({}) and not P.requested({P.ENV: "true"})


# --- host split (no torch) -----------------------------------------------------------------------------------------


def _qsl(lens):
    out = [0]
    for n in lens:
        out.append(out[-1] + n)
    return out


@pytest.mark.parametrize("lens, rest, pf", [
    ([7, 7, 3, 1, 20, 9, 300], (4, 18, 7), (4, 5, 6)),      # Model Runner V2 order: decode/verify first
    ([20, 9, 300], (0, 0, 0), (0, 1, 2)),                   # prefill-only batch
    ([7, 1, 50, 0, 0], (2, 8, 7), (2,)),                    # q_len 0 padding after the prefills
    ([0, 0, 50], (0, 0, 0), (2,)),                          # empty prefix -> vLLM's kernel is skipped
])
def test_plan_splits_the_batch(lens, rest, pf):
    q = _qsl(lens)
    p = P.plan_from_query_start_loc(q + [q[-1]] * 3, len(lens), q[-1], max(lens))  # padded qsl tail ignored
    assert (p.rest.num_reqs, p.rest.num_tokens, p.rest.max_query_len) == rest
    assert p.pf_reqs == pf and p.pf_qlen == tuple(lens[i] for i in pf) and p.pf_q0 == tuple(q[i] for i in pf)


@pytest.mark.parametrize("lens", [[7, 7, 8], [1], [20, 7], [9, 3, 30]])
def test_plan_none_for_decode_only_or_unordered(lens):
    q = _qsl(lens)
    assert P.plan_from_query_start_loc(q, len(lens), q[-1], max(lens)) is None


# --- torch: the patched text on a paged mixed batch -------------------------------------------------------------


def _torch():
    return pytest.importorskip("torch")


GLOBAL = dict(H=16, KVH=1, D=512, window=0)        # Gemma 4 global layer shape
SLIDING = dict(H=4, KVH=2, D=256, window=24)       # sliding-layer head_dim/group (G=2), small window


def _namespace(calls):
    torch = _torch()

    def unified_attention(q, k, v, out, cu_seqlens_q, max_seqlen_q, seqused_k, max_seqlen_k, softmax_scale, causal,
                          window_size, block_table, **kw):
        # Stand-in for vLLM's kernel: exact paged attention for every request it is handed.
        n = len(seqused_k)
        assert cu_seqlens_q.shape[0] == n + 1 and block_table.shape[0] == n and causal is True
        lens = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).tolist()
        assert int(cu_seqlens_q[-1]) == q.shape[0] == out.shape[0] and max(lens) <= max_seqlen_q
        if kw["k_descale"] is not None:
            assert kw["k_descale"].shape[0] == n
        calls.append(dict(num_seqs=n, tokens=q.shape[0], max_q=max_seqlen_q))
        window = window_size[0] + 1 if window_size[0] >= 0 else 0
        P.paged_attention_torch(q, k, v, out, block_table, cu_seqlens_q, seqused_k, range(n), softmax_scale, window)

    def _stub_metadata(cls, cm):
        return cls(num_actual_tokens=cm.num_actual_tokens, max_query_len=cm.max_query_len,
                   query_start_loc=cm.query_start_loc, max_seq_len=cm.max_seq_len, seq_lens=cm.seq_lens,
                   block_table=cm.block_table_tensor, slot_mapping=None, seq_threshold_3D=16,
                   num_par_softmax_segments=16, softmax_segm_output=None, softmax_segm_max=None,
                   softmax_segm_expsum=None, causal=cm.causal)

    from dataclasses import dataclass
    return {"torch": torch, "dataclass": dataclass, "init_logger": lambda n: None,
            "unified_attention": unified_attention, "_stub_metadata": _stub_metadata}


def _load(text, calls):
    ns = _namespace(calls)
    exec(compile(text, "triton_attn_excerpt.py", "exec"), ns)  # noqa: S102 - test-only, our own text
    return ns


def _batch(shape, ctx_q, block_size=16, seed=0, dtype=None):
    """Mixed batch over a paged cache: ctx_q = [(context_len, q_len), ...] in batch order."""
    torch = _torch()
    dtype = dtype or torch.float32
    g = torch.Generator().manual_seed(seed)
    H, KVH, D = shape["H"], shape["KVH"], shape["D"]
    seq = [c + q for c, q in ctx_q]
    nblk = [math.ceil(s / block_size) for s in seq]
    num_blocks = sum(nblk) + 5
    perm = torch.randperm(num_blocks, generator=g)
    width = max(nblk) + 2
    bt = torch.zeros(len(seq), width, dtype=torch.int32)
    used = 0
    for i, nb in enumerate(nblk):
        bt[i, :nb] = perm[used:used + nb].int()
        bt[i, nb:] = 10 ** 6  # beyond seq_len: must never be dereferenced
        used += nb
    # vLLM layout: [num_blocks, KVH, block_size, 2*D] -> transpose(1, 2).split(D, -1) (strided K/V views).
    kv = torch.randn(num_blocks, KVH, block_size, 2 * D, generator=g).to(dtype)
    key_cache, value_cache = kv.transpose(1, 2).split(D, dim=-1)
    qlens = [q for _, q in ctx_q]
    qsl = torch.tensor(_qsl(qlens), dtype=torch.int32)
    T = int(qsl[-1])
    query = torch.randn(T + 3, H, D, generator=g).to(dtype)  # + padded token rows
    cm = types.SimpleNamespace(
        query_start_loc=qsl, query_start_loc_cpu=qsl.clone(), seq_lens=torch.tensor(seq, dtype=torch.int32),
        num_reqs=len(seq), num_actual_tokens=T, max_query_len=max(qlens), max_seq_len=max(seq),
        block_table_tensor=bt, causal=True, rswa_prefix_lens=None)
    return cm, query, key_cache, value_cache


def _impl(ns, shape, **over):
    impl = ns["TritonAttentionImpl"]()
    w = shape["window"]
    attrs = dict(scale=shape["D"] ** -0.5, alibi_slopes=None, use_alibi_sqrt=False,
                 sliding_window=(w - 1, 0) if w else (-1, -1), logits_soft_cap=0, sinks=None, _kv_quant_mode=0,
                 chunk_lookback=-1, use_td=False, _is_per_token_head_quant=False)
    attrs.update(over)
    for k, v in attrs.items():
        setattr(impl, k, v)
    return impl


def _run(ns, shape, cm, query, key_cache, value_cache, impl=None, md_over=None):
    torch = _torch()
    md = ns["TritonAttentionMetadataBuilder"]().build(0, cm)
    for k, v in (md_over or {}).items():
        setattr(md, k, v)
    out = torch.full_like(query, float("nan"))
    k_descale = torch.ones(1).expand(cm.num_reqs, shape["KVH"])
    impl = impl or _impl(ns, shape)
    impl.forward(types.SimpleNamespace(), query, key_cache, value_cache, out, md, cm.num_actual_tokens, None,
                 k_descale, k_descale)
    return md, out


def _expected(shape, cm, query, key_cache, value_cache):
    torch = _torch()
    out = torch.full_like(query, float("nan"))
    P.paged_attention_torch(query, key_cache, value_cache, out, cm.block_table_tensor, cm.query_start_loc,
                            cm.seq_lens, range(cm.num_reqs), shape["D"] ** -0.5, shape["window"])
    return out


MIXED = [(40, 7), (3, 7), (100, 1), (0, 30), (50, 9), (70, 45)]   # 3 decode/verify, then 3 prefill chunks


@pytest.mark.parametrize("shape", [GLOBAL, SLIDING], ids=["global", "sliding"])
@pytest.mark.parametrize("block_size", [16, 32])
def test_patched_forward_matches_reference_on_mixed_batch(monkeypatch, shape, block_size):
    torch = _torch()
    monkeypatch.setenv(P.ENV, "1")
    cm, query, kc, vc = _batch(shape, MIXED, block_size=block_size)
    calls = []
    md, out = _run(_load(PATCHED, calls), shape, cm, query, kc, vc)
    assert md.g4_pfa is not None and md.g4_pfa.pf_reqs == (3, 4, 5)
    # vLLM's kernel saw only the 3 leading decode/verify requests; the prefill rows came from the custom path.
    assert calls == [dict(num_seqs=3, tokens=15, max_q=7)]
    T = cm.num_actual_tokens
    torch.testing.assert_close(out[:T], _expected(shape, cm, query, kc, vc)[:T], rtol=1e-5, atol=1e-5)
    assert torch.isnan(out[T:]).all()  # padded rows untouched, as upstream


@pytest.mark.parametrize("shape", [GLOBAL, SLIDING], ids=["global", "sliding"])
def test_patched_forward_prefill_only_skips_vllm_kernel(monkeypatch, shape):
    torch = _torch()
    monkeypatch.setenv(P.ENV, "1")
    cm, query, kc, vc = _batch(shape, [(0, 20), (33, 64)], seed=1)
    calls = []
    _, out = _run(_load(PATCHED, calls), shape, cm, query, kc, vc)
    assert calls == []
    T = cm.num_actual_tokens
    torch.testing.assert_close(out[:T], _expected(shape, cm, query, kc, vc)[:T], rtol=1e-5, atol=1e-5)


def test_inactive_without_env(monkeypatch):
    torch = _torch()
    monkeypatch.delenv(P.ENV, raising=False)
    cm, query, kc, vc = _batch(GLOBAL, MIXED)
    calls_p, calls_u = [], []
    md, out_p = _run(_load(PATCHED, calls_p), GLOBAL, cm, query, kc, vc)
    _, out_u = _run(_load(EXCERPT, calls_u), GLOBAL, cm, query, kc, vc)
    assert md.g4_pfa is None and calls_p == calls_u == [dict(num_seqs=6, tokens=99, max_q=45)]
    assert torch.equal(out_p, out_u) or torch.allclose(out_p.nan_to_num(), out_u.nan_to_num(), atol=0, rtol=0)
    monkeypatch.setenv(P.ENV, "0")
    assert _load(PATCHED, [])["TritonAttentionMetadataBuilder"]().build(0, cm).g4_pfa is None


@pytest.mark.parametrize("ctx_q", [[(5, 7), (9, 3)], [(0, 20), (5, 7)]], ids=["decode-only", "unordered"])
def test_upstream_for_batches_without_a_clean_split(monkeypatch, ctx_q):
    monkeypatch.setenv(P.ENV, "1")
    cm, query, kc, vc = _batch(GLOBAL, ctx_q)
    calls = []
    md, _ = _run(_load(PATCHED, calls), GLOBAL, cm, query, kc, vc)
    assert md.g4_pfa is None and calls == [dict(num_seqs=2, tokens=cm.num_actual_tokens, max_q=cm.max_query_len)]


@pytest.mark.parametrize("over", [
    {"mm_prefix_range_tensor": "images"}, {"causal": "per-seq tensor"}, {"num_actual_tokens": 5}])
def test_per_batch_fallbacks_run_upstream(monkeypatch, over):
    torch = _torch()
    monkeypatch.setenv(P.ENV, "1")
    cm, query, kc, vc = _batch(GLOBAL, MIXED)
    calls = []
    ns = _load(PATCHED, calls)
    md = ns["TritonAttentionMetadataBuilder"]().build(0, cm)
    assert md.g4_pfa is not None
    for k, v in over.items():
        setattr(md, k, v)
    # The stand-in kernel asserts causal is True; here only check that the hook declines.
    assert P.run_prefill(_impl(ns, GLOBAL), query, kc, vc, torch.empty_like(query), md) is None


@pytest.mark.parametrize("over, msg", [
    ({"logits_soft_cap": 30.0}, "soft-cap"), ({"sinks": object()}, "sinks"), ({"alibi_slopes": object()}, "alibi"),
    ({"chunk_lookback": 2}, "chunked"), ({"use_td": True}, "tensor descriptors")])
def test_unsupported_layer_features_raise(monkeypatch, over, msg):
    monkeypatch.setenv(P.ENV, "1")
    cm, query, kc, vc = _batch(GLOBAL, MIXED)
    ns = _load(PATCHED, [])
    with pytest.raises(RuntimeError, match=msg):
        _run(ns, GLOBAL, cm, query, kc, vc, impl=_impl(ns, GLOBAL, **over))


def test_unfit_layer_shape_runs_upstream(monkeypatch):
    torch = _torch()
    monkeypatch.setenv(P.ENV, "1")
    shape = dict(H=4, KVH=1, D=128, window=0)  # head_dim 128: no global variant fits
    cm, query, kc, vc = _batch(shape, MIXED)
    calls = []
    _, out = _run(_load(PATCHED, calls), shape, cm, query, kc, vc)
    assert calls == [dict(num_seqs=6, tokens=99, max_q=45)]
    T = cm.num_actual_tokens
    torch.testing.assert_close(out[:T], _expected(shape, cm, query, kc, vc)[:T], rtol=1e-5, atol=1e-5)


def test_work_table_covers_every_row_latest_first():
    torch = _torch()
    p = P.plan_from_query_start_loc(_qsl([7, 30, 9]), 3, 46, 30)
    for G, bm in ((16, 64), (2, 64), (16, 32)):
        work, n = p.work(G, bm, "cpu")
        items = work.view(-1, 2).tolist()
        assert n == len(items) == sum(-(-T * G // bm) for T in (30, 9))
        assert sorted(map(tuple, items)) == sorted((r, mb) for r, T in ((1, 30), (2, 9))
                                                   for mb in range(-(-T * G // bm)))
        ends = [mb * bm // G for _, mb in items]
        assert ends == sorted(ends, reverse=True)
        assert p.work(G, bm, "cpu")[0] is work  # cached per (G, BLOCK_M, device) for the step's other layers


def test_gather_follows_block_table():
    torch = _torch()
    cm, _, kc, _ = _batch(SLIDING, [(37, 12)], block_size=16)
    k = P.gather_kv(kc, cm.block_table_tensor[0], 49)
    for j in (0, 15, 16, 48):
        blk = int(cm.block_table_tensor[0, j // 16])
        assert torch.equal(k[j], kc[blk, j % 16])


# --- v2 (paged Gluon kernels): selection, load-time check / fallback, config ---------------------------------------
CONFIG_V2 = "vllm-0.33.0-best-e8-pfa2-c8k"
BASE_V2 = "vllm-0.33.0-best-e8-pfa-c8k"


def _gl_calls(path, table):
    """{name: (args, kwargs)} of the _gl(...) literals assigned to ``table`` in ``path`` (no import: triton)."""
    tree = ast.parse(pathlib.Path(path).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == table for t in node.targets):
            out = {}
            for c in ast.walk(node.value):
                if isinstance(c, ast.Call) and getattr(c.func, "id", "") == "_gl":
                    args = tuple(ast.literal_eval(a) for a in c.args)
                    out[args[0]] = (args[1:], {k.arg: ast.literal_eval(k.value) for k in c.keywords})
            return out
    raise AssertionError(table)


@pytest.mark.parametrize("lab_table, table", [("GLOBAL_VARIANTS", "V2_GLOBAL_VARIANTS"),
                                              ("SLIDING_VARIANTS", "V2_SLIDING_VARIANTS")])
def test_v2_tables_match_the_kernel_lab(lab_table, table):
    lab = _gl_calls(REPO / "kernels" / "prefill_attn_v2.py", lab_table)
    ours = _gl_calls(PATCH_DIR / "prefill_attn_g4.py", table)
    assert ours and all(lab[n] == ours[n] for n in ours)                  # same literals
    assert set(lab) == set(ours) | (set(lab) & set(P.V2_EXCLUDED))         # every lab gl variant ported or excluded
    assert not set(ours) & set(P.V2_EXCLUDED)
    assert set(getattr(P, table)) == {P.V2_PREFIX + n for n in ours}


def test_v2_variant_selection_and_fit():
    assert P.variant_for(0, {P.ENV_GLOBAL: "v2:gl_m32_n32_w2x2"}) is P.V2_GLOBAL_VARIANTS["v2:gl_m32_n32_w2x2"]
    assert P.variant_for(1024, {P.ENV_SLIDING: "v2:gl_m64_n32_w4x1"}).name == "v2:gl_m64_n32_w4x1"
    for name, msg in (("v2:gl_m64_n16_w4x2_kw8", "numerics"), ("v2:tl_m64_dv512_n32_d64_tma_s2", "TMA"),
                      ("v2:gl_m64_n16_w4x2_sk2", "split-KV"), ("v2:gl_m64_n32_w4x1", "not a variant")):
        with pytest.raises(ValueError, match=msg):
            P.variant_for(0, {P.ENV_GLOBAL: name})                         # (a sliding name on a global layer)
    with pytest.raises(ValueError, match="not selectable"):
        P.variant_for(1024, {P.ENV_SLIDING: "v2:gl_m128_n32_w8x1_kw8"})
    for v in P.V2_GLOBAL_VARIANTS.values():
        assert P.is_v2(v) and P.variant_fits(v, 512, 16) and P.v2_smem_bytes(v, 512) <= P.SMEM_LIMIT
    for v in P.V2_SLIDING_VARIANTS.values():
        assert P.is_v2(v) and P.variant_fits(v, 256, 2) and P.v2_smem_bytes(v, 256) <= P.SMEM_LIMIT
    assert not P.variant_fits(P.V2_SLIDING_VARIANTS["v2:gl_m128_n32_w8x1"], 512, 2)   # smem at head_dim 512
    assert not P.variant_fits(P.V2_GLOBAL_VARIANTS["v2:gl_m32_n32_w2x2"], 512, 64)    # block_m < group
    assert not P.is_v2(P.fallback_for(0)) and P.variant_fits(P.fallback_for(0), 512, 16)
    assert not P.is_v2(P.fallback_for(1024)) and P.variant_fits(P.fallback_for(1024), 256, 2)
    assert P.variant_for(0, {}).name == P.DEFAULT_GLOBAL                 # v1 stays the default, unchanged


def test_v2_config():
    e, base = EngineConfig.load(CONFIG_V2), EngineConfig.load(BASE_V2)
    assert e.raw["image_patches"] == base.raw["image_patches"] and list(e.args) == list(base.args)
    diff = {k for k in set(e.env) | set(base.env) if e.env.get(k) != base.env.get(k)}
    assert diff == {P.ENV_GLOBAL, P.ENV_SLIDING} and e.env[P.ENV] == "1"
    vg, vs = P.variant_for(0, e.env), P.variant_for(1024, e.env)
    assert (vg.name, vs.name) == ("v2:gl_m32_n32_w2x2", "v2:gl_m64_n32_w4x1")
    assert P.variant_fits(vg, 512, 16) and P.variant_fits(vs, 256, 2)


def test_v2_usable_checks_once_and_caches_the_verdict(monkeypatch):
    monkeypatch.setattr(P, "_V2_STATE", {})
    seen = []
    k_ok, k_bad, k_exc = ("v2:a", 512, 16, 1, 0, 64), ("v2:b", 256, 2, 8, 1024, 64), ("v2:c", 256, 2, 8, 1024, 16)
    assert P.v2_usable(k_ok, lambda: seen.append("ok")) and P.v2_usable(k_ok, lambda: seen.append("again"))
    assert not P.v2_usable(k_bad, lambda: seen.append("bad") or "mismatch")
    assert not P.v2_usable(k_bad, lambda: seen.append("bad again"))
    assert not P.v2_usable(k_exc, lambda: 1 / 0)
    assert seen == ["ok", "bad"] and "ZeroDivisionError" in P._V2_STATE[k_exc]


def _gpu_fakes(monkeypatch, verdict="pass", launch_error=False):
    """Run the GPU branch of run_prefill on CPU: the v1/v2 launchers become recorders that compute the rows with the
    torch reference."""
    monkeypatch.setattr(P, "_V2_STATE", {})
    monkeypatch.setattr(P, "_gpu_path", lambda q: True)
    log = []

    def v1(p, query, kc, vc, out, bt, qsl, seq, scale, window, v):
        log.append(("v1", v.name))
        P.prefill_torch(p, query, kc, vc, out, bt, seq, scale, window)

    def v2(p, query, kc, vc, out, bt, qsl, seq, scale, window, v):
        if launch_error:
            raise RuntimeError("compile failed")
        log.append(("v2", v.name))
        P.prefill_torch(p, query, kc, vc, out, bt, seq, scale, window)

    def check(*a, **k):
        log.append(("check",))
        return None if verdict == "pass" else verdict

    monkeypatch.setattr(P, "prefill_triton", v1)
    monkeypatch.setattr(P, "prefill_gluon", v2)
    monkeypatch.setattr(P, "v2_self_check", check)
    return log


@pytest.mark.parametrize("check, launch_error, used", [
    ("pass", False, "v2"), ("mismatch vs torch reference", False, "v1"), ("pass", True, "v1")])
@pytest.mark.parametrize("shape", [GLOBAL, SLIDING], ids=["global", "sliding"])
def test_v2_dispatch_and_fallback(monkeypatch, shape, check, launch_error, used):
    torch = _torch()
    win = shape["window"]
    name = "v2:gl_m32_n32_w2x2" if win == 0 else "v2:gl_m64_n32_w4x1"
    monkeypatch.setenv(P.ENV, "1")
    monkeypatch.setenv(P.ENV_SLIDING if win else P.ENV_GLOBAL, name)
    log = _gpu_fakes(monkeypatch, check, launch_error)
    cm, query, kc, vc = _batch(shape, MIXED, block_size=16)
    calls = []
    ns = _load(PATCHED, calls)
    for _ in range(2):  # second step: verdict cached, no second check
        _, out = _run(ns, shape, cm, query, kc, vc)
        T = cm.num_actual_tokens
        torch.testing.assert_close(out[:T], _expected(shape, cm, query, kc, vc)[:T], rtol=1e-5, atol=1e-5)
    fb = P.fallback_for(win).name
    want = {"v2": [("check",), ("v2", name), ("v2", name)],
            "v1": [("check",), ("v1", fb), ("v1", fb)]}[used]
    if launch_error:
        want = [("check",), ("v1", fb), ("v1", fb)]  # the failed launch wrote nothing; v1 ran instead, then sticks
    assert log == want
    assert calls == [dict(num_seqs=3, tokens=15, max_q=7)] * 2


def test_v2_env_on_cpu_path_matches_reference(monkeypatch):
    torch = _torch()
    monkeypatch.setenv(P.ENV, "1")
    monkeypatch.setenv(P.ENV_GLOBAL, "v2:gl_m32_n32_w2x2")
    cm, query, kc, vc = _batch(GLOBAL, MIXED)
    _, out = _run(_load(PATCHED, []), GLOBAL, cm, query, kc, vc)
    T = cm.num_actual_tokens
    torch.testing.assert_close(out[:T], _expected(GLOBAL, cm, query, kc, vc)[:T], rtol=1e-5, atol=1e-5)


def test_v2_check_case_exercises_window_and_prefix():
    for window, bs in ((0, 64), (1024, 16), (1024, 64)):
        ctx_q = P.v2_check_case(window, bs)
        assert ctx_q[0][1] <= P.MIN_Q and all(q > P.MIN_Q for _, q in ctx_q[1:])   # decode prefix, then prefills
        c, q = ctx_q[-1]
        assert (c > window if window else True) and (c + q) // bs >= 2 and (c + q) % bs  # spans pages, ragged tail


def _cuda_gluon():
    torch = _torch()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 0):
        pytest.skip("needs an sm_80+ CUDA GPU")
    if P.GLUON_ERROR is not None:
        pytest.skip(f"gluon unavailable: {P.GLUON_ERROR}")
    return torch


@pytest.mark.parametrize("block_size", [16, 64])
@pytest.mark.parametrize("name", sorted(P.V2_GLOBAL_VARIANTS) + sorted(P.V2_SLIDING_VARIANTS))
def test_v2_gluon_paged_kernel_matches_reference_gpu(name, block_size):
    torch = _cuda_gluon()
    glob = name in P.V2_GLOBAL_VARIANTS
    v = (P.V2_GLOBAL_VARIANTS if glob else P.V2_SLIDING_VARIANTS)[name]
    H, KVH, D, window = (16, 1, 512, 0) if glob else (16, 8, 256, 1024)
    err = P.v2_self_check(v, H, KVH, D, window, block_size, torch.bfloat16, torch.device("cuda"), D ** -0.5)
    assert err is None, err


@pytest.mark.parametrize("shape", [dict(H=16, KVH=1, D=512, window=0), dict(H=16, KVH=8, D=256, window=40)],
                         ids=["global", "sliding"])
def test_v2_patched_forward_on_gpu_mixed_batch(monkeypatch, shape):
    torch = _cuda_gluon()
    monkeypatch.setattr(P, "_V2_STATE", {})
    monkeypatch.setenv(P.ENV, "1")
    monkeypatch.setenv(P.ENV_GLOBAL, "v2:gl_m32_n32_w2x2")
    monkeypatch.setenv(P.ENV_SLIDING, "v2:gl_m64_n32_w4x1")
    cm, query, kc, vc = _batch(shape, MIXED + [(130, 200)], block_size=64, dtype=torch.bfloat16)
    dev = torch.device("cuda")
    kc, vc, query = kc.to(dev), vc.to(dev), query.to(dev)
    cm.query_start_loc, cm.seq_lens = cm.query_start_loc.to(dev), cm.seq_lens.to(dev)
    cm.block_table_tensor = cm.block_table_tensor.clamp(max=kc.shape[0] - 1).to(dev)
    calls = []
    _, out = _run(_load(PATCHED, calls), shape, cm, query, kc, vc)
    T = cm.num_actual_tokens
    ref = _expected(shape, cm, query.float(), kc.float(), vc.float())
    assert all(v is True for v in P._V2_STATE.values()) and P._V2_STATE
    torch.testing.assert_close(out[:T].float(), ref[:T], rtol=2e-2, atol=3e-2)
