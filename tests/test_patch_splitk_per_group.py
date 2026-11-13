"""Image patch splitk_per_group (src/g4bench/patches): anchors, idempotency, per-group split-K selection, scratch
sizing and CUDA-graph coverage. CPU only."""
import ast
import enum
import os
import pathlib
import subprocess
import sys
import types
from typing import ClassVar

import pytest

from g4bench.config import EngineConfig
from g4bench.paths import REPO
from g4bench.patches import apply as A
from g4bench.patches import splitk_per_group as P

PATCH_DIR = REPO / "src" / "g4bench" / "patches"
CONFIG = "vllm-0.33.0-best-e6b-splitkpg"
BASE = "vllm-0.33.0-best-e6b"

# Runnable reduction of vLLM v0.33.0 vllm/v1/attention/backends/triton_attn.py: lines 58-59 and 96-210 (the builder
# class line through the softmax-segment scratch allocation, all three patch anchors) are verbatim; imports are
# replaced by the stubs in _namespace(). Set G4_VLLM_TRITON_ATTN=<path to the pinned upstream file> to also run the
# anchor tests against the real 885-line file.
EXCERPT = '''\
MIN_LAUNCH_GRID_SIZE_2D = 128  # Minimum launch grid size of 2D kernel
NUM_PAR_SOFTMAX_SEGMENTS = 16  # Number of parallel tiled softmax segments


class TritonAttentionMetadataBuilder(AttentionMetadataBuilder[TritonAttentionMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS
    # Step-dependent fields reference persistent input buffers directly.
    supports_draft_decode_metadata_update = True

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        self.block_size = kv_cache_spec.block_size

        model_config = vllm_config.model_config
        # Compatible with models with non-uniform per-layer head counts.
        self.num_heads_q = get_num_attention_heads_from_layers(
            vllm_config, layer_names
        ) or model_config.get_num_attention_heads(vllm_config.parallel_config)
        self.num_heads_kv = model_config.get_num_kv_heads(vllm_config.parallel_config)
        self.headdim = model_config.get_head_size()

        # Check if CUDA Graphs are enabled for decode
        self.decode_cudagraph_enabled = (
            self.vllm_config.compilation_config.cudagraph_mode
            in (
                CUDAGraphMode.FULL_AND_PIECEWISE,
                CUDAGraphMode.FULL_DECODE_ONLY,
                CUDAGraphMode.FULL,
            )
        )

        # The launch grid for the 2D kernel is defined as (num_q_blocks, num_heads_kv).
        # A lower bound for num_q_blocks is the number of sequences.
        # To ensure the minimum launch grid size is achieved, the number of sequences
        # must be at least equal to the threshold below.
        # If this threshold is not reached (i.e., the batch size is not large enough),
        # the 3D kernel will be selected instead.
        self.seq_threshold_3D = MIN_LAUNCH_GRID_SIZE_2D // self.num_heads_kv

        # Modify the threshold if needed.
        if self.decode_cudagraph_enabled:
            capture_sizes = self.vllm_config.compilation_config.cudagraph_capture_sizes
            assert capture_sizes, "CUDA Graphs enabled but no capture sizes specified."

            # Select the CUDA Graph capture size closest to self.seq_threshold_3D
            # as threshold. This ensures that each captured graph covers the
            # correct execution path.
            self.seq_threshold_3D = min(
                capture_sizes,
                key=lambda x: abs(x - self.seq_threshold_3D),
            )

        self.num_par_softmax_segments = NUM_PAR_SOFTMAX_SEGMENTS
        # On SM120, batches whose 16-segment decode grid under-fills the SMs use
        # 64. This counts requests, not Q blocks, on purpose: for 1..11 verify
        # requests (K=3/5, 8 q / 1 KV head, 2k-128k context) 64 segments were
        # faster or within 6% at 2k, and 1.1-3.6x faster from 8k up.
        self.max_seqs_64_segments = 0
        if current_platform.is_cuda() and current_platform.is_device_capability(
            (12, 0)
        ):
            self.max_seqs_64_segments = (current_platform.num_compute_units() - 1) // (
                NUM_PAR_SOFTMAX_SEGMENTS * self.num_heads_kv
            )
        headdim_padded = next_power_of_2(self.headdim)
        max_query_len_3d = 1
        speculative_config = vllm_config.speculative_config
        # The INT4 launcher runs multi-query batches in 2D only.
        if (
            speculative_config is not None
            and kv_cache_spec.kv_quant_mode != KVQuantMode.INT4_PER_TOKEN_HEAD
        ):
            num_speculative_tokens = speculative_config.num_speculative_tokens or 0
            query_len = 1 + num_speculative_tokens * (
                2 if speculative_config.parallel_drafting else 1
            )
            max_query_len_3d = min(query_len, MAX_3D_QUERY_LEN)
        # Scratch is indexed by query token, including verification tokens.
        max_num_seqs_3d = min(
            self.seq_threshold_3D, vllm_config.scheduler_config.max_num_seqs
        )
        max_num_tokens_3d = max_num_seqs_3d * max_query_len_3d
        if self.max_seqs_64_segments > 0:
            self.num_par_softmax_segments = 64
            # build() reuses this scratch at 16 segments with proportionally more rows.
            max_num_tokens_3d = max(
                min(self.max_seqs_64_segments, max_num_seqs_3d) * max_query_len_3d,
                cdiv(
                    max_num_tokens_3d,
                    self.num_par_softmax_segments // NUM_PAR_SOFTMAX_SEGMENTS,
                ),
            )
        self.softmax_segm_output = torch.empty(
            (
                max_num_tokens_3d,
                self.num_heads_q,
                self.num_par_softmax_segments,
                headdim_padded,
            ),
            dtype=torch.float32,
            device=device,
        )
        self.softmax_segm_max = torch.empty(
            (max_num_tokens_3d, self.num_heads_q, self.num_par_softmax_segments),
            dtype=torch.float32,
            device=device,
        )
        self.softmax_segm_expsum = torch.empty(
            (max_num_tokens_3d, self.num_heads_q, self.num_par_softmax_segments),
            dtype=torch.float32,
            device=device,
        )
'''

# --- stubs for the excerpt's imports -------------------------------------------------------------------------------


class _CUDAGraphMode(enum.Enum):
    NONE = 0
    PIECEWISE = 1
    FULL = 2
    FULL_DECODE_ONLY = 3
    FULL_AND_PIECEWISE = 4


class _KVQuantMode(enum.Enum):
    NONE = 0
    INT4_PER_TOKEN_HEAD = 1


class _BuilderBase:
    def __class_getitem__(cls, item):
        return cls

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        self.kv_cache_spec, self.layer_names, self.vllm_config, self.device = (
            kv_cache_spec, layer_names, vllm_config, device)


class _Tensor:
    def __init__(self, shape, dtype=None, device=None):
        self.shape = tuple(shape)


class _Platform:
    def __init__(self, cap: int, sms: int):
        self.cap, self.sms = cap, sms

    def is_cuda(self) -> bool:
        return True

    def is_device_capability(self, cap: tuple[int, int]) -> bool:
        return self.cap == cap[0] * 10 + cap[1]

    def num_compute_units(self) -> int:
        return self.sms


class _Logger:
    def __init__(self):
        self.infos = []

    def info(self, msg, *args):
        self.infos.append(msg % args)


def _cdiv(a: int, b: int) -> int:
    return -(a // -b)


def _next_power_of_2(n: int) -> int:
    return 1 << (n - 1).bit_length() if n > 0 else 1


def _namespace(cap: int, sms: int) -> dict:
    return dict(
        ClassVar=ClassVar, AttentionCGSupport=types.SimpleNamespace(ALWAYS="always"),
        AttentionMetadataBuilder=_BuilderBase, TritonAttentionMetadata=object, AttentionSpec=object, VllmConfig=object,
        torch=types.SimpleNamespace(empty=_Tensor, float32="float32", int32="int32", device=object),
        get_num_attention_heads_from_layers=lambda vllm_config, layer_names: vllm_config.layer_heads_q,
        CUDAGraphMode=_CUDAGraphMode, current_platform=_Platform(cap, sms), next_power_of_2=_next_power_of_2,
        cdiv=_cdiv, KVQuantMode=_KVQuantMode, MAX_3D_QUERY_LEN=8, logger=_Logger(),
    )


# Gemma 4 31B (text config): 16 q heads everywhere; global/full layers 1 KV head x 512, sliding 8 KV heads x 256 with a
# 1024-token window. Model level (ModelArchitectureConfig.from_layers takes the max): 8 KV heads, head size 512.
FULL = types.SimpleNamespace(block_size=64, kv_quant_mode=_KVQuantMode.NONE, num_kv_heads=1, head_size=512,
                             sliding_window=None)
SLIDING = types.SimpleNamespace(block_size=16, kv_quant_mode=_KVQuantMode.NONE, num_kv_heads=8, head_size=256,
                                sliding_window=1024)
CHUNKED = types.SimpleNamespace(block_size=16, kv_quant_mode=_KVQuantMode.NONE, num_kv_heads=8, head_size=256,
                                attention_chunk_size=512)  # spec type without a sliding_window attribute -> "full"
D044_SIZES = [1, 2, 28, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184,
              192, 200, 208, 216, 224, 232, 240, 248, 256, 272, 280, 288, 304, 320, 336, 352, 368, 384, 392, 400, 416,
              432, 448, 464, 480, 496, 504, 512]  # vllm-0.33.0-best-e6b --cudagraph-capture-sizes
MAX_NUM_SEQS = 256
TIERS = ((7, 32), (5, MAX_NUM_SEQS))  # dynamic SD: depth 6 (q_len 7) up to 32 requests, depth 4 (q_len 5) above


def _default_capture_sizes(max_size: int = 512, max_tokens: int = 3072) -> list[int]:
    """vLLM v0.33.0 default (config/vllm.py _set_cudagraph_sizes) for this config: token grid + uniform-decode sizes
    of every dynamic-SD tier (request-count grid x query length)."""
    def grid(m: int) -> list[int]:
        return [n for n in (1, 2, 4) if n <= m] + list(range(8, min(m + 1, 256), 8)) + list(range(256, m + 1, 16))

    decode = {n * q for q, m in TIERS for n in sorted(set(grid(min(m, max_size)) + [min(m, max_size)]))
              if n * q <= max_size}
    return sorted(set(grid(max_size)) | {s for s in decode if s <= max_tokens})


def _decode_graph_reqs(capture: list[int], q: int, decode_query_len: int = 7, max_size: int = 512) -> list[int]:
    """Request counts of the uniform-decode FULL graphs (v1/worker/gpu/cudagraph_utils.py _init_candidates)."""
    out = set()
    for s in capture:
        tokens = _cdiv(s, q) * q
        if tokens > MAX_NUM_SEQS * decode_query_len or tokens > max_size or tokens // q > MAX_NUM_SEQS:
            continue
        out.add(tokens // q)
    return sorted(out)


def _builder(src: str, spec, monkeypatch, *, env: dict | None = None, cap: int = 120, sms: int = 188,
             capture=None, cg: bool = True, num_spec: int | None = 6):
    for name in (P.ENV_FULL, P.ENV_SLIDING):
        monkeypatch.delenv(name, raising=False)
    for k, v in (env or {}).items():
        monkeypatch.setenv(k, v)
    ns = _namespace(cap, sms)
    exec(compile(src, "triton_attn", "exec"), ns)
    model = types.SimpleNamespace(get_num_attention_heads=lambda pc: 16, get_num_kv_heads=lambda pc: 8,
                                  get_head_size=lambda: 512)
    vllm_config = types.SimpleNamespace(
        model_config=model, parallel_config=None, layer_heads_q=16,
        compilation_config=types.SimpleNamespace(
            cudagraph_mode=_CUDAGraphMode.FULL_AND_PIECEWISE if cg else _CUDAGraphMode.PIECEWISE,
            cudagraph_capture_sizes=_default_capture_sizes() if capture is None else capture),
        speculative_config=None if num_spec is None else types.SimpleNamespace(num_speculative_tokens=num_spec,
                                                                               parallel_drafting=False),
        scheduler_config=types.SimpleNamespace(max_num_seqs=MAX_NUM_SEQS),
    )
    b = ns["TritonAttentionMetadataBuilder"](spec, ["layer.0", "layer.5"], vllm_config, "cuda")
    b._log = ns["logger"].infos
    return b


def _state(b) -> tuple:
    return (b.seq_threshold_3D, b.num_par_softmax_segments, b.max_seqs_64_segments, b.softmax_segm_output.shape,
            b.softmax_segm_max.shape, b.softmax_segm_expsum.shape)


def _block_q(spec) -> int:
    """BLOCK_Q exactly as unified_attention() (ops/triton_unified_attention.py 955-961) for 16 q heads."""
    per_kv = 16 // spec.num_kv_heads
    block_m = 16 if per_kv <= 16 else _next_power_of_2(per_kv)
    return block_m // per_kv


def _scratch_rows(b, num_seqs: int, tokens: int) -> int:
    """Rows of the scratch view build() hands the kernel (triton_attn.py 264-280)."""
    rows = b.softmax_segm_max.shape[0]
    if b.max_seqs_64_segments > 0 and (b.max_seqs_64_segments < num_seqs or rows < tokens):
        return rows * b.num_par_softmax_segments // 16
    return rows


def _threshold_says_3d(b, spec, num_seqs: int, q_len: int) -> bool:
    """unified_attention()'s use_3d (1070-1088) for a causal decode/verify launch of uniform q_len."""
    return q_len <= 8 and num_seqs * _cdiv(q_len, _block_q(spec)) <= b.seq_threshold_3D


def _use_3d(b, spec, num_seqs: int, q_len: int) -> bool:
    """...including the scratch guard (1089-1091): 3D only if every query token has a scratch row."""
    tokens = num_seqs * q_len
    return _threshold_says_3d(b, spec, num_seqs, q_len) and _scratch_rows(b, num_seqs, tokens) >= tokens


def _upstream() -> str:
    p = os.environ.get("G4_VLLM_TRITON_ATTN")
    return pathlib.Path(p).read_text(encoding="utf-8") if p else EXCERPT


PATCHED, _ = P.patch_text(EXCERPT)
ENV = {P.ENV_FULL: "112", P.ENV_SLIDING: "8"}


# --- patch mechanics -----------------------------------------------------------------------------------------------


def test_applies_exactly_once_and_is_idempotent():
    src = _upstream()
    out, changed = P.patch_text(src)
    assert changed and out != src
    for anchor, new in P.EDITS:
        assert src.count(anchor) == 1 and out.count(new) == 1
    ast.parse(out)
    again, changed2 = P.patch_text(out)
    assert again == out and not changed2
    # Pure insertion: removing the added text gives back the upstream source.
    restored = out.replace(P.HELPERS, "", 1).replace(P.THRESHOLD_ADD, "", 1).replace(P.SCRATCH_ADD, "", 1)
    assert restored == src
    if src is not EXCERPT:  # the excerpt's builder code is verbatim upstream
        assert EXCERPT.split("\n\n\n", 1)[1] in src


def test_refuses_missing_or_duplicated_anchor():
    with pytest.raises(P.PatchError, match="0 times"):
        P.patch_text(EXCERPT.replace("max_num_tokens_3d = max_num_seqs_3d * max_query_len_3d",
                                     "max_num_tokens_3d = max_num_seqs_3d * max_query_len"))
    with pytest.raises(P.PatchError, match="0 times"):
        P.patch_text(EXCERPT.replace("key=lambda x: abs(x - self.seq_threshold_3D),",
                                     "key=lambda x: (abs(x - self.seq_threshold_3D), x),"))
    with pytest.raises(P.PatchError, match="2 times"):
        P.patch_text(EXCERPT + "\n" + P.HELPERS_ANCHOR + "    pass\n")
    with pytest.raises(P.PatchError, match="marker"):  # marker present but a patched block was edited away
        P.patch_text(PATCHED.replace("self.seq_threshold_3D = self._g4_splitk_threshold", "pass"))


def test_apply_script_patches_a_package_tree(tmp_path):
    pkg = tmp_path / "vllm"
    f = pkg / P.RELPATH
    f.parent.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    f.write_text(EXCERPT, encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": str(tmp_path)}  # find_spec('vllm') -> the fake package
    cmd = [sys.executable, str(PATCH_DIR / "apply.py"), P.NAME]
    r = subprocess.run(cmd, cwd=tmp_path, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "patched" in r.stdout and P.MARKER in f.read_text(encoding="utf-8")
    r2 = subprocess.run(cmd, cwd=tmp_path, env=env, capture_output=True, text=True)
    assert r2.returncode == 0 and "already applied" in r2.stdout
    f.write_text("nothing to anchor\n", encoding="utf-8")
    r3 = subprocess.run(cmd, cwd=tmp_path, env=env, capture_output=True, text=True)
    assert r3.returncode != 0 and "FAILED" in r3.stderr


def test_registered_and_config():
    assert P.NAME in A.KNOWN
    e, base = EngineConfig.load(CONFIG), EngineConfig.load(BASE)
    assert e.raw["image_patches"] == ["mtp_draft_prefill_last", P.NAME]
    gate = {"G4_DRAFT_PREFILL_LAST": "1"}  # mtp_draft_prefill_last env gate (D-061)
    assert e.env == {**gate, P.ENV_FULL: "112", P.ENV_SLIDING: "8"} and base.env == gate
    assert e.raw["role"] == "variant" and e.raw["admissibility"] == "a"
    # Same args as E6b except the D-044 capture-size trim, reverted to vLLM's default list.
    assert [a for a in base.args if a[0] != "--cudagraph-capture-sizes"] == list(e.args)
    assert any(a[0] == "--cudagraph-capture-sizes" for a in base.args)


# --- behaviour -----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("cap", [120, 100])
@pytest.mark.parametrize("cg", [True, False])
@pytest.mark.parametrize("capture", [None, D044_SIZES])
def test_inactive_without_env(monkeypatch, cap, cg, capture):
    for spec in (FULL, SLIDING, CHUNKED):
        for num_spec in (6, None):
            kw = dict(cap=cap, cg=cg, capture=capture, num_spec=num_spec)
            patched = _builder(PATCHED, spec, monkeypatch, **kw)
            assert _state(patched) == _state(_builder(EXCERPT, spec, monkeypatch, **kw))
            assert patched._g4_splitk_threshold is None and patched._log == []
    # Upstream reference points (D-044): shared threshold 16 with the default list, 28 with the trimmed one.
    assert _builder(EXCERPT, FULL, monkeypatch).seq_threshold_3D == 16
    assert _builder(EXCERPT, SLIDING, monkeypatch, capture=D044_SIZES).seq_threshold_3D == 28


def test_per_group_thresholds_are_not_snapped(monkeypatch):
    full, sliding = (_builder(PATCHED, s, monkeypatch, env=ENV) for s in (FULL, SLIDING))
    assert (full.seq_threshold_3D, sliding.seq_threshold_3D) == (112, 8)
    assert "full group (2 layers) seq_threshold_3D 16 -> 112" in full._log[0]
    assert "sliding group (2 layers) seq_threshold_3D 16 -> 8" in sliding._log[0]
    # Not snapped: 113 is no capture size in either list; also without CUDA graphs.
    for capture in (None, D044_SIZES):
        assert _builder(PATCHED, FULL, monkeypatch, env={P.ENV_FULL: "113"}, capture=capture).seq_threshold_3D == 113
    assert _builder(PATCHED, SLIDING, monkeypatch, env=ENV, cg=False).seq_threshold_3D == 8
    # Each variable only affects its own kind of group; a spec without sliding_window counts as full.
    only_full = {P.ENV_FULL: "112"}
    assert _builder(PATCHED, SLIDING, monkeypatch, env=only_full).seq_threshold_3D == 16
    assert _builder(PATCHED, CHUNKED, monkeypatch, env=only_full).seq_threshold_3D == 112
    assert _builder(PATCHED, FULL, monkeypatch, env={P.ENV_SLIDING: "8"}).seq_threshold_3D == 16
    assert _builder(PATCHED, FULL, monkeypatch, env={P.ENV_FULL: "0"}).seq_threshold_3D == 0  # 0 = never 3D


@pytest.mark.parametrize("value", ["-1", "abc", "1.5"])
def test_bad_env_value_fails_loudly(monkeypatch, value):
    with pytest.raises(ValueError):
        _builder(PATCHED, FULL, monkeypatch, env={P.ENV_FULL: value})


# Per-call microseconds, q_len 7, RTX PRO 6000 (results/kernels/20261111T172659Z-attn_bench-c6c4db): (3D, 2D).
BENCH = {
    ("global", 8192): {1: (47.2, 578.6), 4: (139.6, 578.7), 16: (386.5, 576.5), 64: (1526.8, 1727.8)},
    ("global", 1024): {1: (15.4, 75.6), 4: (20.8, 75.8), 16: (69.4, 77.8), 64: (407.6, 234.6)},
    ("sliding", 8192): {1: (17.4, 57.4), 4: (38.9, 58.6), 16: (140.6, 94.4), 64: (577.2, 367.7)},
    ("sliding", 1024): {1: (14.0, 57.3), 4: (29.4, 59.3), 16: (106.2, 95.0), 64: (483.9, 367.7)},
}


def test_selection_follows_the_bench(monkeypatch):
    """Patched choice = the faster kernel at every measured point except global c64 at 8K (3D 12 % faster there,
    74 % slower at 1K: 2D is the compromise); never slower in sum than upstream's shared threshold."""
    specs = {"global": FULL, "sliding": SLIDING}
    for (layer, ctx), points in BENCH.items():
        spec = specs[layer]
        new = _builder(PATCHED, spec, monkeypatch, env=ENV)
        old = _builder(EXCERPT, spec, monkeypatch)  # default capture list: shared threshold 16
        old_d044 = _builder(EXCERPT, spec, monkeypatch, capture=D044_SIZES)  # shared threshold 28
        for c, (t3, t2) in points.items():
            picked = _use_3d(new, spec, c, 7)
            if (layer, ctx, c) != ("global", 8192, 64):
                assert picked == (t3 < t2), (layer, ctx, c)
        cost = {name: sum(t3 if _use_3d(b, spec, c, 7) else t2 for c, (t3, t2) in points.items())
                for name, b in (("new", new), ("old", old), ("d044", old_d044))}
        assert cost["new"] <= min(cost["old"], cost["d044"]), (layer, ctx, cost)
    # The headline case: global layers at c4 run 3D (upstream default: 4 x 7 = 28 > 16 -> 2D, 579 vs 140 us).
    assert _use_3d(_builder(PATCHED, FULL, monkeypatch, env=ENV), FULL, 4, 7)
    assert not _use_3d(_builder(EXCERPT, FULL, monkeypatch), FULL, 4, 7)


@pytest.mark.parametrize("capture", [None, D044_SIZES])
def test_cuda_graphs_every_3d_graph_has_scratch(monkeypatch, capture):
    """The 2D/3D branch is frozen per captured graph at its padded request count. For every uniform-decode graph
    (verify q_len 7 / 5, draft q_len 1) the threshold's 3D choice must also pass the scratch guard, i.e. no graph
    silently falls back to 2D for lack of rows."""
    sizes = _default_capture_sizes() if capture is None else capture
    for spec in (FULL, SLIDING):
        for cap in (120, 100):  # sm_120 64-segment layout and the plain 16-segment one
            b = _builder(PATCHED, spec, monkeypatch, env=ENV, capture=capture, cap=cap)
            for q in (7, 5, 1):
                graphs = _decode_graph_reqs(sizes, q)
                assert graphs
                for reqs in graphs:
                    if _threshold_says_3d(b, spec, reqs, q):
                        assert _use_3d(b, spec, reqs, q), (spec.num_kv_heads, cap, q, reqs)


def test_cutovers_land_on_default_capture_sizes(monkeypatch):
    """With vLLM's default list (the config drops the D-044 trim), real batches of n requests replay the graph for
    padded P(n); the 3D -> 2D switch happens exactly at the thresholds: global 16 requests, sliding 8."""
    graphs = _decode_graph_reqs(_default_capture_sizes(), 7)
    assert {8, 16} <= set(graphs)  # 56 and 112 tokens
    full, sliding = (_builder(PATCHED, s, monkeypatch, env=ENV) for s in (FULL, SLIDING))
    for n in range(1, 33):
        padded = min(g for g in graphs if g >= n)
        assert _use_3d(full, FULL, padded, 7) == (n <= 16), n
        assert _use_3d(sliding, SLIDING, padded, 7) == (n <= 8), n
    graphs5 = _decode_graph_reqs(_default_capture_sizes(), 5)
    for n in range(33, MAX_NUM_SEQS + 1):  # depth 4 tier (q_len 5): always 2D; > 102 requests (512 tokens): eager
        padded = min((g for g in graphs5 if g >= n), default=n)
        assert not _use_3d(full, FULL, padded, 5) and not _use_3d(sliding, SLIDING, padded, 5)


def test_scratch_is_sized_for_the_group_not_for_threshold_x_qlen(monkeypatch):
    mib = 1 / (1 << 20)
    for cap, segments in ((100, 16), (120, 64)):
        full = _builder(PATCHED, FULL, monkeypatch, env=ENV, cap=cap)
        sliding = _builder(PATCHED, SLIDING, monkeypatch, env=ENV, cap=cap)
        rows16 = {k: b.softmax_segm_max.shape[0] * b.num_par_softmax_segments // 16
                  for k, b in (("full", full), ("sliding", sliding))}
        assert rows16 == {"full": 112, "sliding": 56}  # exact: 112 x 1 token-blocks; 8 requests x 7 tokens
        assert full.num_par_softmax_segments == segments
        assert full.softmax_segm_output.shape[1:] == (16, segments, 512)  # head_dim: model max (global layers)
        out_mib = 112 * 16 * 16 * 512 * 4 * mib
        assert full.softmax_segm_output.shape[0] * 16 * segments * 512 * 4 * mib == pytest.approx(out_mib)  # 56 MiB
    # Upstream's bound at the same threshold would be min(112, max_num_seqs) x 7 = 784 rows (392 MiB per builder).
    assert 112 * 7 * 16 * 16 * 512 * 4 * mib == 392.0
    # No speculative config (q_len 1): 112 rows. Helpers: BLOCK_Q as the kernel; unknown BLOCK_Q -> upstream bound.
    assert _builder(PATCHED, FULL, monkeypatch, env=ENV, num_spec=None, cap=100).softmax_segm_max.shape[0] == 112
    ns = {"cdiv": _cdiv, "next_power_of_2": _next_power_of_2}
    exec(compile(P.HELPERS, "helpers", "exec"), ns)
    assert ns["_g4_splitk_block_q"](16, 1) == 1 and ns["_g4_splitk_block_q"](16, 8) == 8
    assert ns["_g4_splitk_block_q"](32, 1) == 1 and ns["_g4_splitk_block_q"](16, 3) is None
    assert ns["_g4_splitk_scratch_tokens"](112, None, 7, 256) == 112 * 7
    assert ns["_g4_splitk_scratch_tokens"](112, 1, 7, 10) == 70  # capped by max_num_seqs
    assert ns["_g4_splitk_scratch_tokens"](0, 1, 7, 256) == 0
