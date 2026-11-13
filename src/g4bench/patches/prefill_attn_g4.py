"""Image patch ``prefill_attn_g4``: Gemma 4 prefill attention (q_len > 8 per request) on a packed-head Triton kernel.

Target: ``vllm/v1/attention/backends/triton_attn.py`` of the pinned image (vLLM v0.33.0; byte-identical to
github.com/vllm-project/vllm tag v0.33.0, 885 lines, sha256 53b221f99c54b6bc...cb223; the same file
``splitk_per_group`` patches - the two patches' anchors are disjoint, so they apply in either order). This file is
ALSO the runtime: the image build copies ``src/g4bench/patches`` to ``/opt/g4patches`` before ``apply.py`` runs,
``patch_text`` embeds this file's absolute path, and the patched backend loads it with ``importlib`` by path only
when ``G4_PREFILL_ATTN=1`` (as ``mtp_draft_head_q`` does). The kernel is kernels/prefill_attn.py's
``_prefill_attn_fwd`` extended to vLLM's paged KV cache and a varlen batch (see "Kernel" below).

Why: vLLM's unified attention runs every prefill chunk on its 2D kernel with BLOCK_M = 16 (token, head) rows per
program; for Gemma 4's global layers (16 q : 1 KV head, head_dim 512) that is one query token per program, so each
token streams its whole causal KV prefix alone. The packed kernel loads each K/V tile once for BLOCK_M / G tokens x
G heads (G = q heads per KV head). Gain unmeasured until kernels/prefill_attn_bench.py has run.

vLLM path (v0.33.0, line numbers of the pinned file): ``TritonAttentionImpl.forward`` (579-736) views the cache
``[num_blocks, num_kv_heads, block_size, 2*hs]`` as ``kv_cache.transpose(1, 2).split(hs, -1)`` (655-657): key_cache /
value_cache ``[num_blocks, block_size, num_kv_heads, hs]`` with arbitrary block/slot/head strides, last dim
contiguous. Token j of request r lives at ``block_table[r, j // block_size]``, slot ``j % block_size``
(block_size = ``value_cache.shape[1]``, per KV-cache group). ``query_start_loc`` (cu_seqlens_q, [num_reqs + 1]) and
``seq_lens`` (seqused_k: context + this step's tokens) describe a mixed batch; ``sliding_window`` is
``(window - 1, 0)`` (key j visible to query position p iff p - j < window). ``unified_attention``
(ops/triton_unified_attention.py 828-1230) picks 2D/3D on host: 3D needs max_seqlen_q <= MAX_3D_QUERY_LEN = 8
(1077-1088), so any batch with a q_len > 8 request runs ALL its requests on the 2D kernel today.

Batch split (why sub-batching, not "run vLLM then overwrite"): overwriting would run every prefill row twice and
could never beat upstream. vLLM's kernel cannot skip arbitrary requests (the q-block grid is derived from
``query_start_loc``), but it runs any contiguous request prefix exactly: ``cu_seqlens_q[:n+1]``, ``seq_lens[:n]``,
``block_table[:n]``, ``query[:cu[n]]`` (absolute token offsets from 0, no rebasing). Model Runner V2 sorts every
batch "verification/decode -> short_extend -> prefill" (worker/gpu/model_runner.py 2507-2519,
``sort_batch_req_ids``: requests with draft tokens first, then non-draft requests by ascending num tokens), so the
requests with q_len <= 8 are a prefix and those with q_len > 8 a suffix (q_len 0 padding may follow). Per step,
host only (``plan``, called from ``build()`` with ``query_start_loc_cpu``, no device sync): requests with
q_len > 8 -> custom kernel (rows written in place); the leading q_len <= 8 requests -> the unchanged
``unified_attention`` call on that prefix (now eligible for its 3D split-K path, like a pure decode batch). If a
q_len 1..8 request follows a prefill (another runner/ordering), ``plan`` returns None: upstream for the whole batch.
Adaptive verification may trim draft lengths on device only; vLLM guarantees the decode/prefill split point and
the prefill lengths stay host-correct (v1/attention/backend.py 217-229), and the custom kernel reads q offsets and
seq lens from the device tensors anyway.

Edits (pure insertions; markers ``# g4bench:prefill_attn_g4``):
  1. after ``logger = init_logger(__name__)`` (54): ``_g4_pfa()``: None unless G4_PREFILL_ATTN=1 (cached per
     process); otherwise this module, loaded by path.
  2. ``TritonAttentionMetadata`` (93): field ``g4_pfa = None`` (the step's plan).
  3. ``build()`` before ``return attn_metadata`` (300): ``if max_query_len > 8`` and enabled: ``plan(...)``.
  4. ``forward()`` before ``unified_attention(`` (697): ``run_prefill`` launches the custom kernel for the plan's
     prefill requests and returns the prefix to hand to vLLM (or None = upstream for this layer/batch); the
     upstream call then runs with ``num_actual_tokens``/``cu_seqlens_q``/``seqused_k``/``block_table``/
     ``max_seqlen_q``/``k_descale``/``v_descale`` cut to that prefix (skipped when the prefix is empty).
Unset env: ``_g4_pfa()`` is None, the plan is never built, ``g4_pfa`` stays None, upstream byte-for-byte behaviour.

CUDA graphs: a batch with a prefill is never ``decode_graph_eligible`` in Model Runner V2 (model_runner.py
1357-1368), so with the default FULL_AND_PIECEWISE mode it never replays a FULL decode graph: it runs PIECEWISE
(``mixed_mode``; worker/gpu/cudagraph_utils.py 351-368) or eager, and
``vllm::unified_attention_with_output`` is a splitting op (config/compilation.py 768-769, 1184), i.e. this forward
runs eagerly between graph pieces - the host plan, pinned H2D copy of the work table and the grid are per step.
Uniform decode/verify batches (q_len <= 7 with MTP 6) have max_query_len <= 8: no plan, nothing changes inside the
FULL decode graphs. Safety net for cudagraph_mode FULL (mixed batches captured as FULL graphs) or any other
capture: ``run_prefill`` returns None while ``torch.cuda.is_current_stream_capturing()``, so a captured graph only
ever contains the upstream kernel (replays stay correct; the patch is then simply inactive for those batches).
No ``.item()`` / host sync on the GPU path (work table via pinned ``non_blocking`` copy).

Fallbacks: per batch (logged once each) - stream capture, a per-sequence causal tensor, multimodal prefix ranges
(images), R-SWA, an unordered batch, a stale plan (metadata reused with other lengths). Per layer (logged once per
shape) - head_dim / group size no variant supports. Static layer features this kernel does not implement raise
(quantized KV cache, alibi, sinks, logit soft-cap, chunked local attention, fused output quant, tensor
descriptors): a requested patch never silently degrades on a layer it would compute wrong.

Kernel (``_g4_pfa_fwd``): one program per (work item, dv block) x KV head; a work item is (request, row block) from
a host-built table sorted latest-row-block-first (longest causal prefix first). Row r of a block = (token r // G,
head r % G) of the KV head's group; K/V tiles of BLOCK_N tokens are gathered through the block table per token
(``block_table[req, j // BLOCK_SIZE]``, int64 addressing), so a tile may span cache blocks. Three loops per row
block as in kernels/prefill_attn.py: window edge (masked), fully visible (no masks), causal diagonal + tail
(masked). Online softmax in exp2, fp32 accumulators, P cast to the KV dtype for the PV MMA, as vLLM. Without
Triton/CUDA the same plan runs ``prefill_torch`` (dense gather + softmax per request; CPU tests).

Variants: ``G4_PREFILL_ATTN_GLOBAL`` / ``G4_PREFILL_ATTN_SLIDING`` name a variant (tables below, copied from
kernels/prefill_attn.py); the defaults are placeholders until the bench picks the winners.

v2 (names ``v2:gl_*``, from kernels/prefill_attn_v2.py): the Gluon mma_v2 FA2 kernel (``_g4_pfa_gl_fwd``) on the
same plan / work table / grid ``(n_work, num_kv)`` (no dv split: the warp grid [WM, WN] splits head_dim). Q tile ->
shared memory once (cp.async); K/V tiles of BLOCK_N tokens by cp.async with the paged gather in the address math:
each copied row loads its ``block_table[req, j // BLOCK_SIZE]`` entry (int64 row offset = page * stride_block +
(j % BLOCK_SIZE) * stride_slot), rows past the causal end are zero-filled and never dereference the table, so a
tile may span cache blocks and any BLOCK_SIZE works (vLLM picks it per KV-cache group; the Triton backend needs a
multiple of 16, e.g. 64 in our runs). Masks: causal + window on every tile, as the lab kernel. Not ported: split-KV
(``_sk2``), the ``tl_*`` TMA variants (TMA descriptors need contiguous K/V), ``gl_m64_n16_w4x2_kw8`` (failed the
bench numerics) and ``gl_m128_n32_w8x1_kw8`` (over shared memory); naming them raises. Load-time check (per
process and layer shape: variant, head_dim, group, KV heads, window, block size, dtype, device; outside stream
capture): the kernel runs a small paged mixed batch (a decode prefix, a fresh chunk, a chunk with context past
the window, shuffled cache blocks) and must match ``paged_attention_torch`` (fp32) within atol 3e-2 + rtol 2e-2
without touching rows outside its requests. Any failure - Gluon missing, compile error, mismatch, a later launch
error - is logged once and that layer shape falls back to the v1 bench winner (``V1_FALLBACK``), or to upstream if
that does not fit.

Risks / unverified (CPU-only authoring; nothing here ran on a GPU): kernel correctness on sm_120 (the bench checks
the contiguous-KV kernel; the paged gather is new), speed of the per-token block-table gather vs contiguous loads,
register pressure of the chosen variants, Triton JIT on first prefill per (variant, block size) (no warm-up; the
PIECEWISE capture's eager attention usually triggers it), and the decode prefix newly taking the 3D split-K path in
mixed batches (same math, different reduction order). v2 (authored without a GPU): Gluon API drift (written
against the lab kernel's Triton 3.8 calls; a compile error only disables v2), cp.async vectorisation of the
gathered rows (needs the compiler to prove 16 B alignment of page * stride_block; else slower, still correct), the
synchronous block-table load in front of each tile's cp.async, and an asynchronous device fault (not an exception)
in the check would not be recoverable.
"""

from __future__ import annotations

import logging
import os
import pathlib
from typing import NamedTuple

NAME = "prefill_attn_g4"
MODULE = "vllm"
RELPATH = "v1/attention/backends/triton_attn.py"  # relative to the vllm package directory
MARKER = "# g4bench:prefill_attn_g4"
ENV = "G4_PREFILL_ATTN"
ENV_GLOBAL = "G4_PREFILL_ATTN_GLOBAL"
ENV_SLIDING = "G4_PREFILL_ATTN_SLIDING"
MIN_Q = 8   # q_len > MIN_Q -> custom kernel; == MAX_3D_QUERY_LEN (requests vLLM's split-K path could take)
RUNTIME_PATH = str(pathlib.Path(__file__).resolve())
LOG2E = 1.4426950408889634

# --------------------------------------------------------------------------------------------------- text patch
LOGGER_ANCHOR = """\
logger = init_logger(__name__)
"""
HELPER_TEMPLATE = """\

{marker} (g4inference-optimization-lab image patch): when {env}=1,
# requests with q_len > {min_q} run a packed-head FlashAttention-2-style Triton prefill kernel;
# the leading q_len <= {min_q} requests keep unified_attention. Unset: upstream.
_G4_PFA_RUNTIME = {runtime!r}
_G4_PFA_MOD = None


def _g4_pfa():
    # The runtime module, or None unless {env}=1 (read once per process; nothing imported).
    global _G4_PFA_MOD
    if _G4_PFA_MOD is None:
        import os

        if os.environ.get("{env}", "").strip() != "1":
            _G4_PFA_MOD = False
        else:
            import importlib.util
            import sys

            name = "g4patch_runtime_{name}"
            mod = sys.modules.get(name)
            if mod is None:
                spec = importlib.util.spec_from_file_location(name, _G4_PFA_RUNTIME)
                mod = importlib.util.module_from_spec(spec)
                sys.modules[name] = mod
                spec.loader.exec_module(mod)
            _G4_PFA_MOD = mod
    return _G4_PFA_MOD or None
"""

FIELD_ANCHOR = """\
    rswa_prefix_lens: torch.Tensor | None = None
    rswa_window: int | None = None
"""
FIELD_ADD = f"""\
    g4_pfa: object | None = None  {MARKER}
"""

BUILD_ANCHOR = """\
            attn_metadata.rswa_window = self.rswa_window

        return attn_metadata
"""
BUILD_NEW = f"""\
            attn_metadata.rswa_window = self.rswa_window

        if common_attn_metadata.max_query_len > {MIN_Q}:  {MARKER}
            g4_pfa = _g4_pfa()
            if g4_pfa is not None:
                attn_metadata.g4_pfa = g4_pfa.plan(common_attn_metadata)
        return attn_metadata
"""

FORWARD_ANCHOR = """\
        mm_prefix_range_tensor = attn_metadata.mm_prefix_range_tensor

        unified_attention(
            q=query[:num_actual_tokens],
"""
FORWARD_NEW = f"""\
        mm_prefix_range_tensor = attn_metadata.mm_prefix_range_tensor

        if attn_metadata.g4_pfa is not None:  {MARKER}
            # Requests with q_len > {MIN_Q}: custom kernel (rows written here). The
            # leading q_len <= {MIN_Q} requests: unified_attention below. None: upstream.
            g4_rest = _g4_pfa().run_prefill(
                self, query, key_cache, value_cache, output, attn_metadata, output_scale
            )
            if g4_rest is not None:
                if g4_rest.num_reqs == 0:
                    return output
                num_actual_tokens = g4_rest.num_tokens
                cu_seqlens_q = cu_seqlens_q[: g4_rest.num_reqs + 1]
                seqused_k = seqused_k[: g4_rest.num_reqs]
                block_table = block_table[: g4_rest.num_reqs]
                max_seqlen_q = g4_rest.max_query_len
                if k_descale is not None:
                    k_descale = k_descale[: g4_rest.num_reqs]
                    v_descale = v_descale[: g4_rest.num_reqs]

        unified_attention(
            q=query[:num_actual_tokens],
"""


def edits(runtime_path: str = RUNTIME_PATH) -> tuple[tuple[str, str], ...]:
    """(anchor, replacement) pairs; anchors occur exactly once in the unpatched file and replacements keep them."""
    helper = HELPER_TEMPLATE.format(marker=MARKER, env=ENV, min_q=MIN_Q, runtime=runtime_path, name=NAME)
    return (
        (LOGGER_ANCHOR, LOGGER_ANCHOR + helper),
        (FIELD_ANCHOR, FIELD_ANCHOR + FIELD_ADD),
        (BUILD_ANCHOR, BUILD_NEW),
        (FORWARD_ANCHOR, FORWARD_NEW),
    )


class PatchError(RuntimeError):
    pass


def patch_text(src: str, runtime_path: str = RUNTIME_PATH) -> tuple[str, bool]:
    """Return (patched source, changed). Idempotent; raises PatchError on any anchor mismatch."""
    eds = edits(runtime_path)
    if MARKER in src:
        for _, new in eds:
            if src.count(new) != 1:
                raise PatchError(f"{NAME}: file carries the marker but a patched block is missing or duplicated")
        return src, False
    out = src
    for anchor, new in eds:
        n = out.count(anchor)
        if n != 1:
            raise PatchError(f"{NAME}: anchor found {n} times (expected exactly 1):\n{anchor}")
        out = out.replace(anchor, new, 1)
    return out, True


# ------------------------------------------------------------------------------------------------------ runtime
# apply.py loads this file as ``g4patch_<NAME>`` during the image build and only needs patch_text: skip the
# torch/triton imports there. Everywhere else they are optional (CPU tests without torch/triton).
torch = triton = tl = None
gluon = gl = cp = mma_v2 = None
GLUON_ERROR = "not imported"
if __name__ != "g4patch_" + NAME:
    try:
        import torch
    except Exception:  # noqa: BLE001
        torch = None
    try:
        import triton
        import triton.language as tl
    except Exception:  # noqa: BLE001
        triton = tl = None
    if triton is not None:
        try:  # Gluon (Triton >= 3.5); v2 variants fall back to v1 without it
            from triton.experimental import gluon
            from triton.experimental.gluon import language as gl
            from triton.experimental.gluon.language.nvidia.ampere import async_copy as cp
            from triton.experimental.gluon.language.nvidia.ampere import mma_v2
            GLUON_ERROR = None
        except Exception as e:  # noqa: BLE001
            gluon = gl = cp = mma_v2 = None
            GLUON_ERROR = repr(e)[:300]

logger = logging.getLogger("vllm.g4patch." + NAME)
_LOGGED: set = set()


def _once(level: int, key, msg: str, *args) -> None:
    if key not in _LOGGED:
        _LOGGED.add(key)
        logger.log(level, msg, *args)


# NamedTuple / plain classes, not dataclasses: apply.py execs this file without registering it in sys.modules.
class Variant(NamedTuple):
    name: str
    block_m: int     # (token, head) rows per program, token-major / head-minor; multiple of the group size
    block_n: int     # KV tokens per tile
    block_d: int     # head-dim chunk for Q K^T (== head_dim with q_in_regs)
    block_dv: int    # output/V head-dim columns per program
    num_warps: int
    num_stages: int
    q_in_regs: bool  # load the program's whole Q tile once (needs block_d == head_dim)


def _v(name, bm, bn, bd, bdv, w, s, qr):
    return Variant(name, bm, bn, bd, bdv, w, s, qr)


# Same tables as kernels/prefill_attn.py (keep in sync; the bench names the winners).
GLOBAL_VARIANTS = {v.name: v for v in [
    _v("m32_dv512_n32_q", 32, 32, 512, 512, 8, 1, True),
    _v("m32_dv512_n16_q", 32, 16, 512, 512, 4, 2, True),
    _v("m64_dv512_n32_d64", 64, 32, 64, 512, 8, 2, False),
    _v("m64_dv512_n32_d64_w16", 64, 32, 64, 512, 16, 2, False),
    _v("m64_dv256_n32_q", 64, 32, 512, 256, 8, 1, True),
    _v("m64_dv256_n64_d64", 64, 64, 64, 256, 8, 2, False),
    _v("m128_dv128_n64_d64", 128, 64, 64, 128, 8, 2, False),
    _v("m128_dv256_n32_d64_w16", 128, 32, 64, 256, 16, 2, False),
    _v("m16_dv512_n32_q", 16, 32, 512, 512, 4, 1, True),
]}
SLIDING_VARIANTS = {v.name: v for v in [
    _v("m64_dv256_n32_q", 64, 32, 256, 256, 8, 1, True),
    _v("m64_dv256_n32_q_s2", 64, 32, 256, 256, 8, 2, True),
    _v("m64_dv256_n64_d64", 64, 64, 64, 256, 8, 2, False),
    _v("m128_dv256_n32_d64_w16", 128, 32, 64, 256, 16, 2, False),
    _v("m128_dv256_n32_d64", 128, 32, 64, 256, 8, 2, False),
    _v("m128_dv128_n64_d64", 128, 64, 64, 128, 8, 2, False),
    _v("m32_dv256_n64_q", 32, 64, 256, 256, 4, 1, True),
]}
DEFAULT_GLOBAL = "m64_dv256_n32_q"      # placeholder until the bench
DEFAULT_SLIDING = "m64_dv256_n32_q_s2"  # placeholder until the bench
# v1 bench winners (kernels/prefill_attn_bench.py): where a v2 variant falls back to when it cannot run.
V1_FALLBACK = {"global": "m64_dv512_n32_d64", "sliding": "m64_dv256_n64_d64"}

V2_PREFIX = "v2:"
SMEM_LIMIT = 101376  # sm_120 max dynamic shared memory per block (99 KB)


class V2Variant(NamedTuple):
    name: str        # "v2:" + the kernel lab name
    block_m: int     # (token, head) rows per program == 16 * wm
    block_n: int     # KV tokens per tile
    block_d: int     # Q/K/V shared-memory slice width (cp.async / ldmatrix)
    wm: int          # warp grid rows
    wn: int          # warp grid columns (split of the KV tile in S and of head_dim in O)
    kv_shared: bool  # K and V share one tile buffer
    kw_qk: int       # DotOperand k_width for Q/K
    kw_pv: int       # DotOperand k_width for P/V


def _gl(name, bm, bn, bd, wm, wn, kvs=False, kw_qk=8, kw_pv=2):
    return V2Variant(V2_PREFIX + name, bm, bn, bd, wm, wn, kvs, kw_qk, kw_pv)


# Same _gl(...) literals as kernels/prefill_attn_v2.py (keep in sync), minus V2_EXCLUDED.
V2_GLOBAL_VARIANTS = {v.name: v for v in [
    _gl("gl_m64_n16_w4x2", 64, 16, 128, 4, 2),
    _gl("gl_m64_n16_w4x2_kvs", 64, 16, 128, 4, 2, kvs=True),
    _gl("gl_m32_n32_w2x4", 32, 32, 64, 2, 4),
    _gl("gl_m32_n32_w2x2", 32, 32, 64, 2, 2),                # bench winner: 190.8 TFLOPS (8K chunk)
]}
V2_SLIDING_VARIANTS = {v.name: v for v in [
    _gl("gl_m128_n32_w8x1", 128, 32, 64, 8, 1),
    _gl("gl_m128_n64_w8x1_kvs", 128, 64, 64, 8, 1, kvs=True),
    _gl("gl_m64_n64_w4x1", 64, 64, 64, 4, 1),
    _gl("gl_m64_n32_w4x1", 64, 32, 64, 4, 1),                # bench winner: 190.9 TFLOPS (8K chunk)
]}
V2_EXCLUDED = {
    "gl_m64_n16_w4x2_kw8": "failed the v2 bench numerics (max_abs 8.4)",
    "gl_m128_n32_w8x1_kw8": "exceeds sm_120 shared memory",
    "gl_m64_n16_w4x2_sk2": "split-KV is not ported to the paged path",
    "gl_m128_n32_w8x1_sk2": "split-KV is not ported to the paged path",
}


def is_v2(v) -> bool:
    return v.name.startswith(V2_PREFIX)


def requested(environ=None) -> bool:
    return (os.environ if environ is None else environ).get(ENV, "").strip() == "1"


def variant_for(window: int, environ=None):
    """The configured variant (v1 ``Variant`` or ``V2Variant``) for a global (window 0) or sliding layer; unknown or
    excluded names raise."""
    env = os.environ if environ is None else environ
    sliding = window > 0
    key = ENV_SLIDING if sliding else ENV_GLOBAL
    name = env.get(key, "").strip() or (DEFAULT_SLIDING if sliding else DEFAULT_GLOBAL)
    if name.startswith(V2_PREFIX):
        table = V2_SLIDING_VARIANTS if sliding else V2_GLOBAL_VARIANTS
        base = name[len(V2_PREFIX):]
        if base in V2_EXCLUDED:
            raise ValueError(f"{NAME}: {key}={name!r} is not selectable: {V2_EXCLUDED[base]}")
        if base.startswith("tl_"):
            raise ValueError(f"{NAME}: {key}={name!r}: v2 tl_* variants use TMA descriptors over contiguous K/V and "
                             "are not available on vLLM's paged cache; use a v2:gl_* or a v1 variant")
    else:
        table = SLIDING_VARIANTS if sliding else GLOBAL_VARIANTS
    if name not in table:
        known = list(SLIDING_VARIANTS if sliding else GLOBAL_VARIANTS) + list(
            V2_SLIDING_VARIANTS if sliding else V2_GLOBAL_VARIANTS)
        raise ValueError(f"{NAME}: {key}={name!r} is not a variant; known: {', '.join(known)}")
    return table[name]


def fallback_for(window: int) -> Variant:
    """The v1 variant a v2 variant falls back to."""
    return SLIDING_VARIANTS[V1_FALLBACK["sliding"]] if window > 0 else GLOBAL_VARIANTS[V1_FALLBACK["global"]]


def v2_smem_bytes(v: V2Variant, head_dim: int, elt: int = 2) -> int:
    """Explicit shared-memory buffers of a v2 variant (Q + K + V, + the P exchange tile when wn > 1)."""
    kv = v.block_n * head_dim * elt * (1 if v.kv_shared else 2)
    p_x = v.block_m * v.block_n * elt if v.wn > 1 else 0
    return v.block_m * head_dim * elt + kv + p_x


def variant_fits(v, head_dim: int, group: int) -> bool:
    if (head_dim & (head_dim - 1)) != 0 or v.block_m % group != 0 or head_dim % v.block_d != 0:
        return False
    if is_v2(v):
        tpc = min(32, v.block_d // 8)
        return (v.block_m == 16 * v.wm and v.block_n % (8 * v.wn) == 0 and head_dim % (8 * v.wn) == 0
                and v.block_d % 16 == 0 and v.block_n % 16 == 0 and (32 // tpc) * v.wm * v.wn <= v.block_n
                and v2_smem_bytes(v, head_dim) <= SMEM_LIMIT)
    return head_dim % v.block_dv == 0 and ((not v.q_in_regs) or v.block_d == head_dim)


# ----------------------------------------------------------------------------------------------------- the plan
class Rest(NamedTuple):
    """The leading requests (q_len <= MIN_Q) that keep vLLM's kernel."""
    num_reqs: int
    num_tokens: int
    max_query_len: int


class Plan:
    """One step's split (host values only), shared by every layer of the attention group."""

    def __init__(self, num_actual_tokens: int, max_query_len: int, rest: Rest, pf_reqs: tuple, pf_qlen: tuple,
                 pf_q0: tuple):
        self.num_actual_tokens = num_actual_tokens
        self.max_query_len = max_query_len
        self.rest = rest
        self.pf_reqs = pf_reqs      # batch indices of the q_len > MIN_Q requests
        self.pf_qlen = pf_qlen      # their q_len
        self.pf_q0 = pf_q0          # their first token row
        self.tables: dict = {}      # (G, BLOCK_M, device) -> (work table, n_work)

    def __repr__(self) -> str:
        return f"Plan(rest={self.rest}, pf_reqs={self.pf_reqs}, pf_qlen={self.pf_qlen})"

    def work(self, group: int, block_m: int, device):
        """Work items (request, row block) for this layer shape, latest row block first; one H2D copy per step
        and shape (pinned, non_blocking)."""
        key = (group, block_m, str(device))
        hit = self.tables.get(key)
        if hit is None:
            items = []
            for req, T in zip(self.pf_reqs, self.pf_qlen):
                n_mb = -(-T * group // block_m)
                items.extend((mb * block_m // group, req, mb) for mb in range(n_mb))
            items.sort(key=lambda x: -x[0])
            flat = [v for _, req, mb in items for v in (req, mb)]
            cpu = torch.tensor(flat, dtype=torch.int32)
            if torch.device(device).type == "cuda":
                cpu = cpu.pin_memory()
                hit = (cpu.to(device, non_blocking=True), len(items))
            else:
                hit = (cpu, len(items))
            self.tables[key] = hit
        return hit


def plan_from_query_start_loc(qsl, num_reqs: int, num_actual_tokens: int, max_query_len: int) -> Plan | None:
    """Host-only split of a batch (``qsl`` = query_start_loc_cpu values). None: no q_len > MIN_Q request, or a
    q_len 1..MIN_Q request after one (not the Model Runner V2 order) -> upstream for the whole batch."""
    q = [int(x) for x in qsl[: num_reqs + 1]]
    qlen = [q[i + 1] - q[i] for i in range(num_reqs)]
    pf = [i for i in range(num_reqs) if qlen[i] > MIN_Q]
    if not pf:
        return None
    first = pf[0]
    if any(0 < qlen[i] <= MIN_Q for i in range(first, num_reqs)):
        _once(logging.WARNING, "unordered", "%s: decode request after a prefill in the batch; upstream attention for "
              "such batches", NAME)
        return None
    rest_tokens = q[first] - q[0]
    rest = Rest(num_reqs=first if rest_tokens > 0 else 0, num_tokens=rest_tokens,
                max_query_len=max(qlen[:first], default=0))
    return Plan(num_actual_tokens=int(num_actual_tokens), max_query_len=int(max_query_len), rest=rest,
                pf_reqs=tuple(pf), pf_qlen=tuple(qlen[i] for i in pf), pf_q0=tuple(q[i] for i in pf))


def plan(common_attn_metadata) -> Plan | None:
    """Called from the patched ``TritonAttentionMetadataBuilder.build`` when max_query_len > MIN_Q."""
    cm = common_attn_metadata
    return plan_from_query_start_loc(cm.query_start_loc_cpu.tolist(), cm.num_reqs, cm.num_actual_tokens,
                                     cm.max_query_len)


# ---------------------------------------------------------------------------------------------------- dispatch
def _static_check(impl, query, key_cache, output_scale) -> None:
    bad = []
    if key_cache.dtype != query.dtype or query.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        bad.append(f"KV cache {key_cache.dtype} / query {query.dtype} (quantized KV not supported)")
    if getattr(impl, "_is_per_token_head_quant", False):
        bad.append("per-token-head quantized KV")
    if getattr(impl, "alibi_slopes", None) is not None:
        bad.append("alibi")
    if getattr(impl, "sinks", None) is not None:
        bad.append("sinks")
    if getattr(impl, "logits_soft_cap", 0):
        bad.append(f"logit soft-cap {impl.logits_soft_cap}")
    if getattr(impl, "chunk_lookback", -1) != -1:
        bad.append("chunked local attention")
    if output_scale is not None:
        bad.append("fused output quantization")
    if getattr(impl, "use_td", False):
        bad.append("tensor descriptors")
    if bad:
        raise RuntimeError(f"{NAME}: {ENV}=1 but this attention layer uses " + ", ".join(bad))


def _window(impl) -> int:
    sw = getattr(impl, "sliding_window", (-1, -1))
    return int(sw[0]) + 1 if sw[0] >= 0 else 0


def _gpu_path(query) -> bool:
    """Triton/Gluon kernels (else the torch reference path; CPU tests)."""
    return query.is_cuda and triton is not None


def run_prefill(impl, query, key_cache, value_cache, output, attn_metadata, output_scale=None) -> Rest | None:
    """Patched-forward hook: write the plan's prefill rows of ``output`` and return the leading requests vLLM's kernel
    still has to run, or None (upstream for this layer and batch; nothing written)."""
    p = attn_metadata.g4_pfa
    if query.is_cuda and torch.cuda.is_current_stream_capturing():
        _once(logging.INFO, "capture", "%s: stream capture in progress; upstream attention inside captured graphs",
              NAME)
        return None
    if p.num_actual_tokens != attn_metadata.num_actual_tokens or p.max_query_len != attn_metadata.max_query_len:
        _once(logging.WARNING, "stale", "%s: metadata lengths differ from the plan; upstream for such batches", NAME)
        return None
    if not isinstance(attn_metadata.causal, bool) or not attn_metadata.causal:
        _once(logging.WARNING, "causal", "%s: non-causal / per-sequence causal batch; upstream", NAME)
        return None
    if attn_metadata.mm_prefix_range_tensor is not None or attn_metadata.rswa_prefix_lens is not None:
        _once(logging.INFO, "mm", "%s: multimodal prefix ranges or R-SWA in the batch; upstream for such batches",
              NAME)
        return None
    _static_check(impl, query, key_cache, output_scale)
    window = _window(impl)
    head_dim = query.shape[2]
    num_heads, num_kv = query.shape[1], key_cache.shape[2]
    group = num_heads // num_kv
    v = variant_for(window)
    on_gpu = _gpu_path(query)
    if num_heads % num_kv == 0 and is_v2(v) and on_gpu:
        key = v2_key(v, head_dim, group, num_kv, window, int(value_cache.shape[1]), query)
        fits = variant_fits(v, head_dim, group)
        if not fits:
            _v2_fail(key, "does not fit this layer shape")
        if not fits or not v2_usable(key, lambda: v2_self_check(v, num_heads, num_kv, head_dim, window,
                                                                  int(value_cache.shape[1]), query.dtype,
                                                                  query.device, impl.scale)):
            v = fallback_for(window)
    if num_heads % num_kv or not variant_fits(v, head_dim, group):
        _once(logging.WARNING, ("fit", head_dim, group, v.name), "%s: variant %s does not fit head_dim %d, %d q heads "
              "per KV head; upstream for this layer shape", NAME, v.name, head_dim, group)
        return None
    _once(logging.INFO, ("on", head_dim, group, window, v.name), "%s: prefill rows -> packed kernel %s (head_dim %d, "
          "%d q heads per KV head, window %d)", NAME, v.name, head_dim, group, window)
    args = (p, query, key_cache, value_cache, output, attn_metadata.block_table, attn_metadata.query_start_loc,
            attn_metadata.seq_lens, impl.scale, window)
    if on_gpu:
        if is_v2(v):
            try:
                prefill_gluon(*args, v)
                return p.rest
            except Exception as e:  # noqa: BLE001 - e.g. a recompile for other strides failing; nothing was written
                _v2_fail(key, f"launch failed: {type(e).__name__}: {e}"[:300])
                v = fallback_for(window)
                if not variant_fits(v, head_dim, group):
                    return None
        prefill_triton(*args, v)
    else:
        prefill_torch(p, query, key_cache, value_cache, output, attn_metadata.block_table,
                      attn_metadata.seq_lens, impl.scale, window)
    return p.rest


# ------------------------------------------------------------------------------------------- v2 enable / fallback
_V2_STATE: dict = {}  # v2_key -> True (verified) | str (why not: falls back to v1 for that layer shape)


def v2_key(v, head_dim, group, num_kv, window, block_size, query) -> tuple:
    return (v.name, head_dim, group, num_kv, window, block_size, str(query.dtype), str(query.device))


def _v2_fail(key, why: str) -> None:
    if isinstance(_V2_STATE.get(key), str):
        return  # already failed and logged
    _V2_STATE[key] = why
    _once(logging.WARNING, ("v2fail",) + tuple(key), "%s: v2 variant %s disabled for %s (%s); falling back to v1 %s",
          NAME, key[0], key[1:], why, V1_FALLBACK["sliding" if key[4] > 0 else "global"])


def v2_usable(key, check) -> bool:
    """Run ``check()`` (None = pass, str = why not; exceptions count as failure) once per key; cache the verdict."""
    st = _V2_STATE.get(key)
    if st is None:
        try:
            err = check()
        except Exception as e:  # noqa: BLE001 - Gluon compile errors, OOM, ...
            err = f"{type(e).__name__}: {e}"[:300]
        if err is None:
            _V2_STATE[key] = True
            _once(logging.INFO, ("v2ok",) + tuple(key), "%s: v2 variant %s passed its load-time paged check for %s",
                  NAME, key[0], key[1:])
        else:
            _v2_fail(key, f"load-time check: {err}")
        st = _V2_STATE[key]
    return st is True


def v2_check_case(window: int, block_size: int) -> list:
    """(context_len, q_len) per request of the load-time check batch: a decode/verify prefix (must stay untouched),
    a fresh chunk, and a chunk whose context reaches past the window (or 150 tokens for global layers)."""
    return [(5, 3), (0, 37), ((window + 29) if window > 0 else 150, 70)]


def v2_self_check(v, num_heads, num_kv, head_dim, window, block_size, dtype, device, scale,
                  atol: float = 3e-2, rtol: float = 2e-2) -> str | None:
    """Run v2 variant ``v`` on a small paged batch shaped like the layer and compare with the fp32 torch reference.
    Returns None on a match, else the reason. Same constexprs as the real launches (the compile is reused)."""
    if GLUON_ERROR is not None:
        return f"gluon unavailable: {GLUON_ERROR}"
    if dtype not in (torch.bfloat16, torch.float16):
        return f"dtype {dtype} (v2 needs bf16/fp16)"
    ctx_q = v2_check_case(window, block_size)
    g = torch.Generator().manual_seed(20261113)
    seq = [c + q for c, q in ctx_q]
    nblk = [-(-s // block_size) for s in seq]
    num_blocks = sum(nblk) + 3
    perm = torch.randperm(num_blocks, generator=g)
    bt = torch.zeros(len(seq), max(nblk) + 2, dtype=torch.int32)
    used = 0
    for i, nb in enumerate(nblk):
        bt[i, :nb] = perm[used:used + nb].int()
        used += nb
    kv = torch.randn(num_blocks, num_kv, block_size, 2 * head_dim, generator=g).to(dtype).to(device)
    key_cache, value_cache = kv.transpose(1, 2).split(head_dim, dim=-1)  # vLLM's strided K/V views
    qlens = [q for _, q in ctx_q]
    qsl_l = [0]
    for n in qlens:
        qsl_l.append(qsl_l[-1] + n)
    T = qsl_l[-1]
    query = torch.randn(T + 2, num_heads, head_dim, generator=g).to(dtype).to(device)
    qsl = torch.tensor(qsl_l, dtype=torch.int32, device=device)
    seq_lens = torch.tensor(seq, dtype=torch.int32, device=device)
    bt = bt.to(device)
    p = plan_from_query_start_loc(qsl_l, len(seq), T, max(qlens))
    out = torch.full_like(query, float("nan"))
    prefill_gluon(p, query, key_cache, value_cache, out, bt, qsl, seq_lens, scale, window, v)
    ref = torch.full(query.shape, float("nan"), dtype=torch.float32, device=device)
    paged_attention_torch(query, key_cache, value_cache, ref, bt, qsl, seq_lens, p.pf_reqs, scale, window)
    r0 = p.rest.num_tokens
    if not bool(torch.isnan(out[:r0]).all()) or not bool(torch.isnan(out[T:]).all()):
        return "wrote rows outside its requests"
    o, r = out[r0:T].float(), ref[r0:T]
    if not bool(torch.isfinite(o).all()):
        return "non-finite output"
    err = (o - r).abs()
    if bool((err > atol + rtol * r.abs()).any()):
        return f"mismatch vs torch reference: max_abs {float(err.max()):.4g} (atol {atol}, rtol {rtol})"
    return None


# ------------------------------------------------------------------------------------------------------ kernels
def gather_kv(cache, block_table_row, seq_len: int):
    """[seq_len, num_kv_heads, D] view-gather of one request's keys or values from the paged cache."""
    bs = cache.shape[1]
    idx = torch.arange(seq_len, device=cache.device)
    blocks = block_table_row[idx // bs].long()
    return cache[blocks, idx % bs]


def reference_attention(q, k, v, start: int, scale: float = 1.0, window: int = 0):
    """fp32 causal(/window) attention of q [T, H, D] at positions [start, start+T) over k, v [>= start+T, KVH, D]
    (q head h -> KV head h // G). Returns [T, H, D] fp32."""
    T, H, _ = q.shape
    S = start + T
    g = H // k.shape[1]
    kb = k[:S].float().repeat_interleave(g, dim=1)
    vb = v[:S].float().repeat_interleave(g, dim=1)
    s = torch.einsum("thd,shd->hts", q.float(), kb) * scale
    qpos = start + torch.arange(T, device=q.device)
    kpos = torch.arange(S, device=q.device)
    mask = kpos[None, :] <= qpos[:, None]
    if window > 0:
        mask &= (qpos[:, None] - kpos[None, :]) < window
    s = s.masked_fill(~mask[None], float("-inf"))
    return torch.einsum("hts,shd->thd", torch.softmax(s, dim=-1), vb)


def paged_attention_torch(query, key_cache, value_cache, output, block_table, cu_seqlens_q, seq_lens, reqs,
                          scale: float, window: int = 0) -> None:
    """Reference for requests ``reqs`` of a varlen batch over the paged cache (eager, slow; CPU/test path)."""
    cu = [int(x) for x in cu_seqlens_q.tolist()]
    sl = [int(x) for x in seq_lens.tolist()]
    for r in reqs:
        q0, T = cu[r], cu[r + 1] - cu[r]
        if T <= 0:
            continue
        S = sl[r]
        k = gather_kv(key_cache, block_table[r], S)
        v = gather_kv(value_cache, block_table[r], S)
        output[q0:q0 + T] = reference_attention(query[q0:q0 + T], k, v, S - T, scale, window).to(output.dtype)


def prefill_torch(p: Plan, query, key_cache, value_cache, output, block_table, seq_lens, scale, window) -> None:
    sl = seq_lens.tolist()
    for req, T, q0 in zip(p.pf_reqs, p.pf_qlen, p.pf_q0):
        S = int(sl[req])
        k = gather_kv(key_cache, block_table[req], S)
        v = gather_kv(value_cache, block_table[req], S)
        output[q0:q0 + T] = reference_attention(query[q0:q0 + T], k, v, S - T, scale, window).to(output.dtype)


if triton is not None:
    @triton.jit
    def _g4_pfa_tiles(acc, l_i, m_i, q_all, q_rows, k_base, v_base, bt_row, stride_kb, stride_ks, stride_vb,
                      stride_vs, offs_dv, qpos, lo, hi, kv_hi, qk_scale,
                      BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
                      BLOCK_SIZE: tl.constexpr, Q_IN_REGS: tl.constexpr, MASK: tl.constexpr, WINDOW: tl.constexpr):
        """Online-softmax update over KV tiles [lo, hi) (lo BLOCK_N-aligned), K/V gathered through the block table.
        MASK: causal/window masks and bounds checks against kv_hi; otherwise every key is visible to every row."""
        for n0 in range(lo, hi, BLOCK_N):
            offs_n = n0 + tl.arange(0, BLOCK_N)
            n_ok = offs_n < kv_hi
            if MASK:
                blk = tl.load(bt_row + offs_n // BLOCK_SIZE, mask=n_ok, other=0).to(tl.int64)
            else:
                blk = tl.load(bt_row + offs_n // BLOCK_SIZE).to(tl.int64)
            slot = (offs_n % BLOCK_SIZE).to(tl.int64)
            k_off = blk * stride_kb + slot * stride_ks
            v_off = blk * stride_vb + slot * stride_vs
            if Q_IN_REGS:
                offs_d = tl.arange(0, HEAD_DIM)
                k_ptrs = k_base + k_off[:, None] + offs_d[None, :]
                if MASK:
                    k = tl.load(k_ptrs, mask=n_ok[:, None], other=0.0)
                else:
                    k = tl.load(k_ptrs)
                s = tl.dot(q_all, tl.trans(k))
            else:
                s = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
                for d0 in range(0, HEAD_DIM, BLOCK_D):
                    offs_d = d0 + tl.arange(0, BLOCK_D)
                    qc = tl.load(q_rows + offs_d[None, :])
                    k_ptrs = k_base + k_off[:, None] + offs_d[None, :]
                    if MASK:
                        kc = tl.load(k_ptrs, mask=n_ok[:, None], other=0.0)
                    else:
                        kc = tl.load(k_ptrs)
                    s += tl.dot(qc, tl.trans(kc))
            s = s * qk_scale
            if MASK:
                valid = (offs_n[None, :] <= qpos[:, None]) & n_ok[None, :]
                if WINDOW > 0:
                    valid = valid & ((qpos[:, None] - offs_n[None, :]) < WINDOW)
                s = tl.where(valid, s, float("-inf"))
                m_new = tl.maximum(m_i, tl.max(s, axis=1))
                m_use = tl.where(m_new == float("-inf"), 0.0, m_new)
            else:
                m_new = tl.maximum(m_i, tl.max(s, axis=1))
                m_use = m_new
            alpha = tl.exp2(m_i - m_use)
            p = tl.exp2(s - m_use[:, None])
            l_i = l_i * alpha + tl.sum(p, axis=1)
            v_ptrs = v_base + v_off[:, None] + offs_dv[None, :]
            if MASK:
                vt = tl.load(v_ptrs, mask=n_ok[:, None], other=0.0)
            else:
                vt = tl.load(v_ptrs)
            acc = acc * alpha[:, None] + tl.dot(p.to(vt.dtype), vt)
            m_i = m_new
        return acc, l_i, m_i

    @triton.jit
    def _g4_pfa_fwd(
        Q, KC, VC, Out, BT, CU, SEQ, WORK, qk_scale,
        stride_qt, stride_qh, stride_kb, stride_ks, stride_kh, stride_vb, stride_vs, stride_vh,
        stride_ot, stride_oh, stride_bt,
        G: tl.constexpr, HEAD_DIM: tl.constexpr, WINDOW: tl.constexpr, BLOCK_SIZE: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_DV: tl.constexpr,
        NUM_DV_BLOCKS: tl.constexpr, Q_IN_REGS: tl.constexpr,
    ):
        pid0 = tl.program_id(0)
        kvh = tl.program_id(1)
        w = pid0 // NUM_DV_BLOCKS
        dvb = pid0 % NUM_DV_BLOCKS          # dv programs of one work item adjacent -> share K/V/Q in L2
        req = tl.load(WORK + 2 * w)
        mb = tl.load(WORK + 2 * w + 1)
        q0 = tl.load(CU + req)
        T = tl.load(CU + req + 1) - q0
        start = tl.load(SEQ + req) - T      # context length (tokens before this chunk)

        offs_m = mb * BLOCK_M + tl.arange(0, BLOCK_M)
        row_ok = offs_m < T * G
        t = tl.minimum(offs_m // G, T - 1)  # padded rows duplicate the last token (never stored)
        g = offs_m % G
        qpos = start + t
        q_rows = Q + (q0 + t)[:, None] * stride_qt + (kvh * G + g)[:, None] * stride_qh

        t_lo = (mb * BLOCK_M) // G
        t_hi = tl.minimum((mb * BLOCK_M + BLOCK_M - 1) // G, T - 1)
        hi = start + t_hi + 1
        c_full = ((start + t_lo + 1) // BLOCK_N) * BLOCK_N
        if WINDOW > 0:
            lo = tl.maximum(start + t_lo - WINDOW + 1, 0)
            a = (lo // BLOCK_N) * BLOCK_N
            w_full = tl.maximum(start + t_hi - WINDOW + 1, 0)
            w_full_blk = ((w_full + BLOCK_N - 1) // BLOCK_N) * BLOCK_N
        else:
            a = start * 0
            w_full_blk = start * 0
        b = tl.maximum(tl.minimum(w_full_blk, c_full), a)
        c = tl.maximum(c_full, b)

        bt_row = BT + req.to(tl.int64) * stride_bt
        k_base = KC + kvh * stride_kh
        v_base = VC + kvh * stride_vh
        offs_dv = dvb * BLOCK_DV + tl.arange(0, BLOCK_DV)
        if Q_IN_REGS:
            q_all = tl.load(q_rows + tl.arange(0, HEAD_DIM)[None, :])
        else:
            q_all = tl.zeros([1], dtype=tl.float32)

        m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_DV], dtype=tl.float32)
        if WINDOW > 0:
            acc, l_i, m_i = _g4_pfa_tiles(acc, l_i, m_i, q_all, q_rows, k_base, v_base, bt_row, stride_kb, stride_ks,
                                          stride_vb, stride_vs, offs_dv, qpos, a, b, hi, qk_scale, BLOCK_M, HEAD_DIM,
                                          BLOCK_N, BLOCK_D, BLOCK_SIZE, Q_IN_REGS, True, WINDOW)
        acc, l_i, m_i = _g4_pfa_tiles(acc, l_i, m_i, q_all, q_rows, k_base, v_base, bt_row, stride_kb, stride_ks,
                                      stride_vb, stride_vs, offs_dv, qpos, b, c, hi, qk_scale, BLOCK_M, HEAD_DIM,
                                      BLOCK_N, BLOCK_D, BLOCK_SIZE, Q_IN_REGS, False, WINDOW)
        acc, l_i, m_i = _g4_pfa_tiles(acc, l_i, m_i, q_all, q_rows, k_base, v_base, bt_row, stride_kb, stride_ks,
                                      stride_vb, stride_vs, offs_dv, qpos, c, hi, hi, qk_scale, BLOCK_M, HEAD_DIM,
                                      BLOCK_N, BLOCK_D, BLOCK_SIZE, Q_IN_REGS, True, WINDOW)

        out = acc / tl.where(l_i > 0.0, l_i, 1.0)[:, None]
        o_ptrs = Out + (q0 + t)[:, None] * stride_ot + (kvh * G + g)[:, None] * stride_oh + offs_dv[None, :]
        tl.store(o_ptrs, out.to(Out.dtype.element_ty), mask=row_ok[:, None])


def prefill_triton(p: Plan, query, key_cache, value_cache, output, block_table, query_start_loc, seq_lens, scale,
                   window: int, v: Variant) -> None:
    T, H, D = query.shape
    num_kv = key_cache.shape[2]
    G = H // num_kv
    assert query.stride(2) == 1 and output.stride(2) == 1 and key_cache.stride(3) == 1 and value_cache.stride(3) == 1
    assert max(T * query.stride(0), T * output.stride(0)) < 2 ** 31  # 32-bit q/out address math
    work, n_work = p.work(G, v.block_m, query.device)
    if n_work == 0:
        return
    n_dv = D // v.block_dv
    _g4_pfa_fwd[(n_work * n_dv, num_kv)](
        query, key_cache, value_cache, output, block_table, query_start_loc, seq_lens, work,
        float(scale) * LOG2E,
        query.stride(0), query.stride(1),
        key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
        value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),
        output.stride(0), output.stride(1), block_table.stride(0),
        G=G, HEAD_DIM=D, WINDOW=int(window), BLOCK_SIZE=int(value_cache.shape[1]),
        BLOCK_M=v.block_m, BLOCK_N=v.block_n, BLOCK_D=v.block_d, BLOCK_DV=v.block_dv,
        NUM_DV_BLOCKS=n_dv, Q_IN_REGS=v.q_in_regs,
        num_warps=v.num_warps, num_stages=v.num_stages,
    )


# --------------------------------------------------------------------------------------------- v2 (Gluon, paged)
if gluon is not None:
    @gluon.jit
    def _g4_gl_issue_tile(smem, base, bt_row, n0, n_hi, stride_b, stride_s,
                          ROWS: gl.constexpr, HEAD_DIM: gl.constexpr, BLOCK_D: gl.constexpr,
                          BLOCK_SIZE: gl.constexpr, blk: gl.constexpr):
        """cp.async KV rows [n0, n0 + ROWS) of one request (rows >= n_hi zero-filled, their block-table entries never
        read) from the paged cache into smem: row j lives at page block_table[j // BLOCK_SIZE], slot
        j % BLOCK_SIZE (int64 offsets). BLOCK_D-wide column slices, one commit group."""
        offs_n = n0 + gl.arange(0, ROWS, layout=gl.SliceLayout(1, blk))
        n_ok = offs_n < n_hi
        page = gl.load(bt_row + offs_n // BLOCK_SIZE, mask=n_ok, other=0).to(gl.int64)
        row = page * stride_b + (offs_n % BLOCK_SIZE).to(gl.int64) * stride_s
        offs_d = gl.arange(0, BLOCK_D, layout=gl.SliceLayout(0, blk))
        msk = gl.expand_dims(n_ok, 1) & gl.expand_dims(offs_d < BLOCK_D, 0)
        rows = base + gl.expand_dims(row, 1)
        for i in gl.static_range(HEAD_DIM // BLOCK_D):
            cp.async_load(smem.slice(i * BLOCK_D, BLOCK_D, dim=1), rows + gl.expand_dims(i * BLOCK_D + offs_d, 0),
                          mask=msk)
        cp.commit_group()

    @gluon.jit
    def _g4_pfa_gl_fwd(
        Q, KC, VC, Out, BT, CU, SEQ, WORK, qk_scale,
        stride_qt, stride_qh, stride_kb, stride_ks, stride_kh, stride_vb, stride_vs, stride_vh,
        stride_ot, stride_oh, stride_bt,
        G: gl.constexpr, HEAD_DIM: gl.constexpr, WINDOW: gl.constexpr, BLOCK_SIZE: gl.constexpr,
        BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_D: gl.constexpr,
        WM: gl.constexpr, WN: gl.constexpr, NW: gl.constexpr, TPR: gl.constexpr, TPC: gl.constexpr,
        KW_QK: gl.constexpr, KW_PV: gl.constexpr, KV_SHARED: gl.constexpr,
    ):
        """kernels/prefill_attn_v2.py ``_fwd_gl`` (SPLITS 1) on a varlen batch: the program's (request, row block)
        comes from the work table, q offsets / context lengths from the device metadata, K/V from the paged cache."""
        mma: gl.constexpr = gl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[WM, WN], instr_shape=[16, 8])
        dot_a_qk: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mma, k_width=KW_QK)
        dot_b_qk: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mma, k_width=KW_QK)
        dot_a_pv: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mma, k_width=KW_PV)
        dot_b_pv: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mma, k_width=KW_PV)
        blk: gl.constexpr = gl.BlockedLayout(size_per_thread=[1, 8], threads_per_warp=[TPR, TPC],
                                             warps_per_cta=[NW, 1], order=[1, 0])
        smem_l: gl.constexpr = gl.SwizzledSharedLayout(vec=8, per_phase=1, max_phase=8, order=[1, 0])
        row_l: gl.constexpr = gl.SliceLayout(1, mma)
        col_l: gl.constexpr = gl.SliceLayout(0, mma)

        w = gl.program_id(0)
        kvh = gl.program_id(1)
        req = gl.load(WORK + 2 * w)
        mb = gl.load(WORK + 2 * w + 1)
        q0 = gl.load(CU + req)
        T = gl.load(CU + req + 1) - q0
        start = gl.load(SEQ + req) - T          # context length (tokens before this chunk)

        t_lo = (mb * BLOCK_M) // G
        t_hi = gl.minimum((mb * BLOCK_M + BLOCK_M - 1) // G, T - 1)
        hi = start + t_hi + 1
        if WINDOW > 0:
            j_lo = (gl.maximum(start + t_lo - WINDOW + 1, 0) // BLOCK_N) * BLOCK_N
        else:
            j_lo = start * 0
        j_hi = hi

        q_smem = gl.allocate_shared_memory(Q.dtype.element_ty, [BLOCK_M, HEAD_DIM], smem_l)
        k_smem = gl.allocate_shared_memory(KC.dtype.element_ty, [BLOCK_N, HEAD_DIM], smem_l)
        if KV_SHARED:
            v_smem = k_smem
        else:
            v_smem = gl.allocate_shared_memory(VC.dtype.element_ty, [BLOCK_N, HEAD_DIM], smem_l)

        # Q tile -> smem once (rows past T * G zero-filled, never stored)
        offs_qr = mb * BLOCK_M + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, blk))
        q_row = (q0 + offs_qr // G) * stride_qt + (kvh * G + offs_qr % G) * stride_qh
        offs_qd = gl.arange(0, BLOCK_D, layout=gl.SliceLayout(0, blk))
        q_msk = gl.expand_dims(offs_qr < T * G, 1) & gl.expand_dims(offs_qd < BLOCK_D, 0)
        for i in gl.static_range(HEAD_DIM // BLOCK_D):
            cp.async_load(q_smem.slice(i * BLOCK_D, BLOCK_D, dim=1),
                          Q + gl.expand_dims(q_row, 1) + gl.expand_dims(i * BLOCK_D + offs_qd, 0), mask=q_msk)
        cp.commit_group()

        bt_row = BT + req.to(gl.int64) * stride_bt
        k_base = KC + kvh * stride_kh
        v_base = VC + kvh * stride_vh
        _g4_gl_issue_tile(k_smem, k_base, bt_row, j_lo, j_hi, stride_kb, stride_ks, BLOCK_N, HEAD_DIM, BLOCK_D,
                          BLOCK_SIZE, blk)

        offs_m = mb * BLOCK_M + gl.arange(0, BLOCK_M, layout=row_l)
        qpos = start + gl.minimum(offs_m // G, T - 1)
        m_i = gl.full([BLOCK_M], float("-inf"), gl.float32, row_l)
        l_i = gl.zeros([BLOCK_M], gl.float32, row_l)
        acc = gl.zeros([BLOCK_M, HEAD_DIM], gl.float32, mma)

        for n0 in range(j_lo, j_hi, BLOCK_N):
            if KV_SHARED:
                cp.wait_group(0)
            else:
                _g4_gl_issue_tile(v_smem, v_base, bt_row, n0, j_hi, stride_vb, stride_vs, BLOCK_N, HEAD_DIM,
                                  BLOCK_D, BLOCK_SIZE, blk)                                     # V(j) in flight
                cp.wait_group(1)                                                                # Q, K(j) landed
            gl.barrier()
            s = gl.zeros([BLOCK_M, BLOCK_N], gl.float32, mma)
            for i in gl.static_range(HEAD_DIM // BLOCK_D):
                qa = q_smem.slice(i * BLOCK_D, BLOCK_D, dim=1).load(dot_a_qk)
                kb = k_smem.slice(i * BLOCK_D, BLOCK_D, dim=1).permute((1, 0)).load(dot_b_qk)
                s = mma_v2(qa, kb, s)
            gl.barrier()                                                                        # k_smem free
            if KV_SHARED:
                _g4_gl_issue_tile(v_smem, v_base, bt_row, n0, j_hi, stride_vb, stride_vs, BLOCK_N, HEAD_DIM,
                                  BLOCK_D, BLOCK_SIZE, blk)
            else:
                _g4_gl_issue_tile(k_smem, k_base, bt_row, n0 + BLOCK_N, j_hi, stride_kb, stride_ks, BLOCK_N,
                                  HEAD_DIM, BLOCK_D, BLOCK_SIZE, blk)

            offs_n = n0 + gl.arange(0, BLOCK_N, layout=col_l)
            valid = gl.expand_dims(offs_n, 0) <= gl.expand_dims(qpos, 1)
            if WINDOW > 0:
                valid = valid & ((gl.expand_dims(qpos, 1) - gl.expand_dims(offs_n, 0)) < WINDOW)
            s = gl.where(valid, s, float("-inf"))
            m_new = gl.maximum(m_i, gl.max(s, axis=1) * qk_scale)
            m_use = gl.where(m_new == float("-inf"), 0.0, m_new)
            alpha = gl.exp2(m_i - m_use)
            p = gl.exp2(s * qk_scale - gl.expand_dims(m_use, 1))
            l_i = l_i * alpha + gl.sum(p, axis=1)
            acc = acc * gl.expand_dims(alpha, 1)
            pa = gl.convert_layout(p.to(VC.dtype.element_ty), dot_a_pv)
            m_i = m_new

            if KV_SHARED:
                cp.wait_group(0)
            else:
                cp.wait_group(1)                                                                # V(j) landed
            gl.barrier()
            vb = v_smem.load(dot_b_pv)
            acc = mma_v2(pa, vb, acc)
            gl.barrier()                                                                        # v_smem free
            if KV_SHARED:
                _g4_gl_issue_tile(k_smem, k_base, bt_row, n0 + BLOCK_N, j_hi, stride_kb, stride_ks, BLOCK_N,
                                  HEAD_DIM, BLOCK_D, BLOCK_SIZE, blk)
        cp.wait_group(0)

        offs_d = gl.arange(0, HEAD_DIM, layout=col_l)
        t = gl.minimum(offs_m // G, T - 1)
        head = kvh * G + offs_m % G
        st_msk = gl.expand_dims(offs_m < T * G, 1) & gl.expand_dims(offs_d < HEAD_DIM, 0)
        out = acc / gl.expand_dims(gl.where(l_i > 0.0, l_i, 1.0), 1)
        o_ptrs = Out + gl.expand_dims((q0 + t) * stride_ot + head * stride_oh, 1) + gl.expand_dims(offs_d, 0)
        gl.store(o_ptrs, out.to(Out.dtype.element_ty), mask=st_msk)


def prefill_gluon(p: Plan, query, key_cache, value_cache, output, block_table, query_start_loc, seq_lens, scale,
                  window: int, v: V2Variant) -> None:
    """Launch the paged v2 Gluon kernel for the plan's prefill requests (raises if Gluon is unavailable)."""
    if gluon is None:
        raise RuntimeError(f"gluon unavailable: {GLUON_ERROR}")
    T, H, D = query.shape
    num_kv = key_cache.shape[2]
    G = H // num_kv
    assert query.stride(2) == 1 and output.stride(2) == 1 and key_cache.stride(3) == 1 and value_cache.stride(3) == 1
    assert key_cache.dtype == value_cache.dtype == query.dtype and query.dtype in (torch.bfloat16, torch.float16)
    assert max(T * query.stride(0), T * output.stride(0)) < 2 ** 31  # 32-bit q/out address math
    work, n_work = p.work(G, v.block_m, query.device)
    if n_work == 0:
        return
    tpc = min(32, v.block_d // 8)
    _g4_pfa_gl_fwd[(n_work, num_kv)](
        query, key_cache, value_cache, output, block_table, query_start_loc, seq_lens, work,
        float(scale) * LOG2E,
        query.stride(0), query.stride(1),
        key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
        value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),
        output.stride(0), output.stride(1), block_table.stride(0),
        G=G, HEAD_DIM=D, WINDOW=int(window), BLOCK_SIZE=int(value_cache.shape[1]),
        BLOCK_M=v.block_m, BLOCK_N=v.block_n, BLOCK_D=v.block_d,
        WM=v.wm, WN=v.wn, NW=v.wm * v.wn, TPR=32 // tpc, TPC=tpc, KW_QK=v.kw_qk, KW_PV=v.kw_pv,
        KV_SHARED=v.kv_shared, num_warps=v.wm * v.wn,
    )
