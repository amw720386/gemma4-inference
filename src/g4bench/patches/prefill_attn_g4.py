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

Risks / unverified (CPU-only authoring; nothing here ran on a GPU): kernel correctness on sm_120 (the bench checks
the contiguous-KV kernel; the paged gather is new), speed of the per-token block-table gather vs contiguous loads,
register pressure of the chosen variants, Triton JIT on first prefill per (variant, block size) (no warm-up; the
PIECEWISE capture's eager attention usually triggers it), and the decode prefix newly taking the 3D split-K path in
mixed batches (same math, different reduction order).
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


def requested(environ=None) -> bool:
    return (os.environ if environ is None else environ).get(ENV, "").strip() == "1"


def variant_for(window: int, environ=None) -> Variant:
    """The configured variant for a global (window 0) or sliding layer; unknown names raise."""
    env = os.environ if environ is None else environ
    table, key, default = ((SLIDING_VARIANTS, ENV_SLIDING, DEFAULT_SLIDING) if window > 0
                           else (GLOBAL_VARIANTS, ENV_GLOBAL, DEFAULT_GLOBAL))
    name = env.get(key, "").strip() or default
    if name not in table:
        raise ValueError(f"{NAME}: {key}={name!r} is not a variant; known: {', '.join(table)}")
    return table[name]


def variant_fits(v: Variant, head_dim: int, group: int) -> bool:
    return (head_dim & (head_dim - 1)) == 0 and head_dim % v.block_dv == 0 and head_dim % v.block_d == 0 \
        and ((not v.q_in_regs) or v.block_d == head_dim) and v.block_m % group == 0


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
    if num_heads % num_kv or not variant_fits(v, head_dim, group):
        _once(logging.WARNING, ("fit", head_dim, group, v.name), "%s: variant %s does not fit head_dim %d, %d q heads "
              "per KV head; upstream for this layer shape", NAME, v.name, head_dim, group)
        return None
    _once(logging.INFO, ("on", head_dim, group, window, v.name), "%s: prefill rows -> packed kernel %s (head_dim %d, "
          "%d q heads per KV head, window %d)", NAME, v.name, head_dim, group, window)
    if query.is_cuda and triton is not None:
        prefill_triton(p, query, key_cache, value_cache, output, attn_metadata.block_table,
                       attn_metadata.query_start_loc, attn_metadata.seq_lens, impl.scale, window, v)
    else:
        prefill_torch(p, query, key_cache, value_cache, output, attn_metadata.block_table,
                      attn_metadata.seq_lens, impl.scale, window)
    return p.rest


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
