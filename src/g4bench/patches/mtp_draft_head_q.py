"""Image patch ``mtp_draft_head_q``: Gemma 4 MTP greedy drafting via a quantized head scan + exact BF16 rescore.

Target: ``vllm/model_executor/models/gemma4_mtp.py`` of the pinned image (vLLM v0.33.0; github.com/vllm-project/vllm
tag v0.33.0, 600 lines, sha256 30e68a66fa1c21b1...0bedd05). This file is ALSO the runtime: the image build copies
``src/g4bench/patches`` to ``/opt/g4patches`` (``modal_app._engine_image``, ``add_local_dir(copy=True)``, before
``apply.py`` runs), ``patch_text`` embeds this file's absolute path at apply time, and the patched model loads it with
``importlib`` by path (no ``sys.path`` change, nothing imported unless the env var is set). So the kernels that run
are exactly the ones in the image the patch was applied to.

Why: the Gemma 4 12B MTP drafter scores every draft token with a dense full-vocab BF16 head (``compute_logits``,
gemma4_mtp.py 563-577: ``self.logits_processor(self.lm_head, h)`` with ``lm_head`` tied to the draft's
``model.embed_tokens.weight`` [262144, 1024], 512 MiB) and takes the argmax (speculator.py 478-479, suppressing
``suppress_tokens`` 258883/258882 from generation_config.json via ``index_fill_``). On the RTX PRO 6000 that head is
395 us per draft step at 28 rows; a quantized scan with fused per-block top-T, top-k and exact BF16 rescoring of the
k candidates costs 210 us (FP8) / 133 us (INT4) with 99.2-100 % / 97.5-100 % agreement with the BF16 argmax
(kernels/draft_head_scan.py, results/kernels/20261111T172419Z-draft_head_scan-556087). vLLM's own
``draft_lm_head_quantization`` refuses Gemma 4 MTP (config/speculative.py 1953-1958: no shared lm_head).

Hook (the only place greedy draft tokens are chosen for Gemma 4 MTP on Model Runner V2):
``DraftModelSpeculator.sample_draft`` (v1/worker/gpu/spec_decode/speculator.py 444-481), called from
``TargetDependentARSpeculator._prefill`` (draft step 0, tdar/speculator.py 476) and ``_generate_draft`` (steps 1..,
tdar/speculator.py 659). With ``draft_logits is None`` (draft_sample_method "greedy", the default) it either computes
``compute_draft_logits`` + ``logits.argmax(-1)`` (line 479) or, when the speculative config sets
``use_local_argmax_reduction``, returns ``self.model.get_top_tokens(h)`` (lines 456-457) without materialising logits.
Upstream ``Gemma4MTP.get_top_tokens`` (gemma4_mtp.py 579-588) only serves the centroid-masked head
(``use_ordered_embeddings``; false for gemma-4-12B-it-assistant) and asserts otherwise. The patch therefore edits
gemma4_mtp.py only, and the engine config sets ``"use_local_argmax_reduction": true``:
  1. module level, before ``@support_torch_compile class Gemma4MTP``: ``_g4_dhq_build(model)``: returns None unless
     ``G4_DRAFT_HEAD_Q`` is ``fp8`` or ``int4``; then loads this file and calls ``build_for(model)``.
  2. ``get_top_tokens`` (line 583-584): ``if self._g4_dhq is not None: return self._g4_dhq.top_tokens(h)``; else the
     unchanged upstream body (including its assert).
  3. ``load_weights`` (line 600): ``self._g4_dhq = _g4_dhq_build(self)`` after the weights (and ``_suppress_idx``)
     are loaded.
Env unset/other value: ``_g4_dhq`` is None, nothing is imported, every method behaves exactly as upstream (with
``use_local_argmax_reduction`` and no env var, upstream's assert fires, so the -dhq config cannot silently run BF16).

Algorithm (``DraftHeadQ.top_tokens``, per draft step, M = rows):
  scan:    one Triton program per 128 vocab rows x 64 activation rows (M <= 64 is one BLOCK_M tile, so the quantized
           head is read once): logits_q = h @ deq(Wq)^T in fp32; suppressed ids and padding -> -inf; keep the top T = 2
           (value, token id) per block -> [M, 2048*2] candidates (no [M, V] logits in HBM).
  top-k:   ``torch.topk`` over the candidates, k = ``G4_DRAFT_HEAD_K`` (default 32, clamp 1..64).
  rescore: one Triton program per row gathers the k BF16 head rows, fp32 dot, rounds to BF16 (what cuBLAS's BF16
           output does), argmax with ties -> lowest token id (``torch.argmax`` returns the first maximum).
FP8: e4m3fn with a per-row fp32 scale (amax/448). INT4: symmetric, group 128 along K, fp16 scales [G, V], two
nibbles per byte (byte g*64+j holds k = g*128+j low, g*128+64+j high, offset-binary). Without Triton or CUDA the
same pipeline runs in torch (``scan_torch``/``rescore_torch``; vocab-chunked; slow, eager): that is what the CPU tests
exercise, and the Triton path is checked against the BF16 argmax at load (``_self_check``, raises on a mismatch rate
that only a packing/decode bug produces).

Lossless for the served output (the target verifies every draft token; only draft choices can change):
  * Nothing the target reads changes. The BF16 head is still the tied tensor (kept: it is needed for rescoring; the
    draft's ``model.embed_tokens`` is replaced by the TARGET's embedding in ``Gemma4Speculator._share_embeddings``,
    gemma4/speculator.py 173-187, so this matrix is only used as the head - and is never modified). The quantized copy
    is a new tensor. The draft writes no KV (Q-only, KV-shared; see mtp_draft_prefill_last), so a different draft
    token only changes the draft's own next-step input.
  * Draft tokens are proposals. Greedy requests: a draft token is accepted only if it equals the target's argmax,
    so the emitted sequence is the target's greedy sequence for any proposal. Sampled requests: with greedy drafting
    vLLM's rejection sampler treats the proposal as one-hot q; accept-with-min(1, p/q) plus resampling from
    (p - q)+ yields exactly the target distribution p for ANY proposal that does not depend on the target's random
    draws. The proposal here is a deterministic function of the draft hidden state.
  * Only the acceptance length (hence speed) can change. Bitwise, outputs can differ as with any acceptance change:
    the target's verify batches change shape, and vLLM kernels are not batch-invariant.
  Admissibility: class (b) per D-011 (lower-precision draft shortlist), like ``vllm-0.33.0-mtp4-draftfp8``.

CUDA graphs: draft steps (``_generate_draft``) and draft prefill are captured; ``top_tokens`` is capture-safe: shapes
depend only on M (fixed per captured size); two Triton launches (grid (V/128, ceil(M/BLOCK_M)) and (M,)), one
``torch.topk``, outputs from ``torch.empty`` (graph pool under capture); no ``.item()``, no host sync, no
data-dependent allocation or control flow. Triton compiles per BLOCK_M bucket (16/32/64) and is warmed for every
bucket at load (``_self_check``), never during capture. The torch fallback is eager-oriented (not used on GPU images).

Guards (``build_for``; class guard first, everything else raises so a requested format never silently degrades):
only ``type(model).__name__ == "Gemma4MTP"``; no centroid masking (``masked_embedding is None``); TP = 1; no logit
soft-cap/scale; BF16 head; vocab % 128 == 0, hidden % 128 == 0; <= 2 suppressed ids. Other speculators/models never
reach the code (the edits are inside Gemma4MTP and the speculator only calls get_top_tokens when configured).

Risks / unverified (CPU-only authoring; nothing here ran on a GPU inside vLLM):
  * Acceptance: real hidden states may agree less (or more) than the synthetic recall above; INT4 misses ~2 % of
    BF16 argmaxes on "realistic" activations (FP8 ~0.4 %); each miss can cut an accepted run. Net gain needs an
    end-to-end run (acceptance length vs ``vllm-0.33.0-best-e6b``).
  * Memory: +257 MiB (FP8) / +132 MiB (INT4) allocated during model load, before KV profiling (KV cache shrinks
    by the same amount); a ~128 MiB fp32 transient while quantizing (row-chunked).
  * h is converted to fp16 for the scan (clamped to +-65504); rescoring is exact fp32 -> BF16.
  * ``use_local_argmax_reduction`` is incompatible with adaptive verification / probabilistic drafting (vLLM raises).
  * Kernels were tuned on sm_120 (BLOCK_N 128, BLOCK_K 128, 4 warps, 2 stages); M > 64 runs extra M-tiles (re-reads
    the quantized head per 64 rows). The 2-D grid and the h clamp are new relative to kernels/draft_head_scan.py.
  * Each cold container JIT-compiles 3 kernels x 3 BLOCK_M buckets at load (seconds; Triton cache is not persisted).
"""

from __future__ import annotations

import logging
import os
import pathlib

NAME = "mtp_draft_head_q"
MODULE = "vllm"
RELPATH = "model_executor/models/gemma4_mtp.py"  # relative to the vllm package directory
MARKER = "# g4bench:mtp_draft_head_q"
ENV_FORMAT = "G4_DRAFT_HEAD_Q"
ENV_K = "G4_DRAFT_HEAD_K"
FORMATS = ("fp8", "int4")
DEFAULT_K, MAX_K = 32, 64
TOPT = 2          # candidates kept per vocab block by the fused scan epilogue
BLOCK_N = 128     # vocab rows per scan program
BLOCK_K = 128     # FP8 K step
GROUP = 128       # INT4 group size along K
SCAN_WARPS, SCAN_STAGES = 4, 2   # fastest fused config on sm_120 at M=28 for FP8 and INT4 (draft_head_scan run)
RUNTIME_PATH = str(pathlib.Path(__file__).resolve())

# --------------------------------------------------------------------------------------------------- text patch
HELPER_ANCHOR = """\


@support_torch_compile
class Gemma4MTP(nn.Module):
"""
HELPER_TEMPLATE = """\


{marker} (g4inference-optimization-lab image patch): when {env} is
# fp8 or int4, greedy draft tokens come from a quantized head scan + exact BF16 rescore
# (get_top_tokens; needs speculative_config use_local_argmax_reduction). Unset: upstream.
_G4_DHQ_RUNTIME = {runtime!r}


def _g4_dhq_build(model):
    import os

    if os.environ.get("{env}", "").strip().lower() not in {formats!r}:
        return None
    import importlib.util
    import sys

    name = "g4patch_runtime_{name}"
    mod = sys.modules.get(name)
    if mod is None:
        spec = importlib.util.spec_from_file_location(name, _G4_DHQ_RUNTIME)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return mod.build_for(model)
"""

TOP_ANCHOR = '''\
        """Sparse argmax via centroids masking. Returns token IDs directly."""
        assert self.masked_embedding is not None
'''
TOP_NEW = f'''\
        """Sparse argmax via centroids masking. Returns token IDs directly."""
        g4_dhq = getattr(self, "_g4_dhq", None)  {MARKER}
        if g4_dhq is not None:
            return g4_dhq.top_tokens(hidden_states)
        assert self.masked_embedding is not None
'''

LOAD_ANCHOR = """\
                device=next(self.parameters()).device,
            )
        return loaded
"""
LOAD_NEW = f"""\
                device=next(self.parameters()).device,
            )
        self._g4_dhq = _g4_dhq_build(self)  {MARKER}
        return loaded
"""


def edits(runtime_path: str = RUNTIME_PATH) -> tuple[tuple[str, str], ...]:
    """(anchor, replacement) pairs; anchors occur exactly once in the unpatched file and replacements keep them."""
    helper = HELPER_TEMPLATE.format(marker=MARKER, env=ENV_FORMAT, formats=FORMATS, runtime=runtime_path, name=NAME)
    return ((HELPER_ANCHOR, helper + HELPER_ANCHOR), (TOP_ANCHOR, TOP_NEW), (LOAD_ANCHOR, LOAD_NEW))


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
# torch/triton imports there (seconds per load). Everywhere else they are optional (CPU tests without torch).
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
_BIG = 1 << 30

if triton is not None:
    @triton.jit
    def _g4_dhq_epilogue(acc, rows, offs_m, M, N, OUT_V, OUT_I, pid_n, NCAND, SUP0: tl.constexpr,
                         SUP1: tl.constexpr, TOPT_: tl.constexpr, BLOCK_N_: tl.constexpr):
        bad = (rows == SUP0) | (rows == SUP1) | (rows >= N)
        acc = tl.where(bad[None, :], float("-inf"), acc)
        m_mask = offs_m < M
        cols = tl.arange(0, BLOCK_N_)
        for t in tl.static_range(TOPT_):
            v, i = tl.max(acc, axis=1, return_indices=True)
            base = offs_m * NCAND + pid_n * TOPT_ + t
            tl.store(OUT_V + base, v, mask=m_mask)
            tl.store(OUT_I + base, pid_n * BLOCK_N_ + i, mask=m_mask)
            acc = tl.where(cols[None, :] == i[:, None], float("-inf"), acc)

    @triton.jit(do_not_specialize=["M", "NCAND"])
    def _g4_dhq_fp8_scan(H, Q, S, OUT_V, OUT_I, M, N, NCAND, K: tl.constexpr, SUP0: tl.constexpr,
                         SUP1: tl.constexpr, TOPT_: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N_: tl.constexpr,
                         BLOCK_K_: tl.constexpr):
        pid_n = tl.program_id(0)
        offs_m = tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)
        rows = pid_n * BLOCK_N_ + tl.arange(0, BLOCK_N_)
        m_mask = offs_m < M
        acc = tl.zeros((BLOCK_M, BLOCK_N_), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K_):
            ks = k0 + tl.arange(0, BLOCK_K_)
            h = tl.load(H + offs_m[:, None] * K + ks[None, :], mask=m_mask[:, None], other=0.0).to(tl.float32)
            h = tl.minimum(tl.maximum(h, -65504.0), 65504.0).to(tl.float16)
            q = tl.load(Q + rows[None, :] * K + ks[:, None])           # [BLOCK_K, BLOCK_N] (N % BLOCK_N == 0)
            acc += tl.dot(h, q.to(tl.float16))
        s = tl.load(S + rows)
        acc = acc * s[None, :]
        _g4_dhq_epilogue(acc, rows, offs_m, M, N, OUT_V, OUT_I, pid_n, NCAND, SUP0, SUP1, TOPT_, BLOCK_N_)

    @triton.jit(do_not_specialize=["M", "NCAND"])
    def _g4_dhq_int4_scan(H, P, S, OUT_V, OUT_I, M, N, NCAND, K: tl.constexpr, SUP0: tl.constexpr,
                          SUP1: tl.constexpr, TOPT_: tl.constexpr, GROUP_: tl.constexpr, HALF: tl.constexpr,
                          BLOCK_M: tl.constexpr, BLOCK_N_: tl.constexpr):
        pid_n = tl.program_id(0)
        offs_m = tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)
        rows = pid_n * BLOCK_N_ + tl.arange(0, BLOCK_N_)
        m_mask = offs_m < M
        kk = tl.arange(0, HALF)
        acc = tl.zeros((BLOCK_M, BLOCK_N_), dtype=tl.float32)
        for g in range(0, K // GROUP_):
            k0 = g * GROUP_
            h_lo = tl.load(H + offs_m[:, None] * K + (k0 + kk)[None, :], mask=m_mask[:, None], other=0.0)
            h_hi = tl.load(H + offs_m[:, None] * K + (k0 + HALF + kk)[None, :], mask=m_mask[:, None], other=0.0)
            h_lo = tl.minimum(tl.maximum(h_lo.to(tl.float32), -65504.0), 65504.0).to(tl.float16)
            h_hi = tl.minimum(tl.maximum(h_hi.to(tl.float32), -65504.0), 65504.0).to(tl.float16)
            p = tl.load(P + rows[None, :] * (K // 2) + (g * HALF + kk)[:, None]).to(tl.int32)  # [HALF, BLOCK_N]
            lo = ((p & 15) - 8).to(tl.float16)
            hi = ((p >> 4) - 8).to(tl.float16)
            part = tl.dot(h_lo, lo)
            part = tl.dot(h_hi, hi, acc=part)
            sc = tl.load(S + g * N + rows).to(tl.float32)
            acc += part * sc[None, :]
        _g4_dhq_epilogue(acc, rows, offs_m, M, N, OUT_V, OUT_I, pid_n, NCAND, SUP0, SUP1, TOPT_, BLOCK_N_)

    @triton.jit(do_not_specialize=["KC", "stride_pos", "stride_map"])
    def _g4_dhq_rescore(H, W, POS, MAP, TOK, KC, stride_pos, stride_map, K: tl.constexpr, KC_PAD: tl.constexpr,
                        BLOCK_D: tl.constexpr):
        # One program per row: gather k BF16 head rows, fp32 dot, round to BF16, argmax, ties -> lowest token id.
        m = tl.program_id(0)
        c = tl.arange(0, KC_PAD)
        cmask = c < KC
        p = tl.load(POS + m * stride_pos + c, mask=cmask, other=0)
        idx = tl.load(MAP + m * stride_map + p, mask=cmask, other=0).to(tl.int64)
        acc = tl.zeros((KC_PAD,), dtype=tl.float32)
        for d0 in range(0, K, BLOCK_D):
            ds = d0 + tl.arange(0, BLOCK_D)
            h = tl.load(H + m * K + ds).to(tl.float32)
            w = tl.load(W + idx[:, None] * K + ds[None, :], mask=cmask[:, None], other=0.0).to(tl.float32)
            acc += tl.sum(w * h[None, :], axis=1)
        r = tl.where(cmask, acc.to(tl.bfloat16).to(tl.float32), float("-inf"))
        mx = tl.max(r, axis=0)
        tok = tl.min(tl.where((r == mx) & cmask, idx, 1073741824), axis=0)
        # All-NaN scores (e.g. uninitialised dummy hidden states during CUDA-graph capture) match nothing and would
        # leave the 2**30 sentinel, an out-of-range token id that trips the embedding's index assert. Fall back to the
        # first candidate: it is only a draft token, and the target verifies every draft token anyway.
        first = tl.sum(tl.where(c == 0, idx, 0), axis=0)
        tok = tl.where(tok >= 1073741824, first, tok)
        tl.store(TOK + m, tok)


def requested_format(environ=None) -> str | None:
    v = (os.environ if environ is None else environ).get(ENV_FORMAT, "").strip().lower()
    return v if v in FORMATS else None


def requested_k(environ=None) -> int:
    raw = (os.environ if environ is None else environ).get(ENV_K, "").strip()
    if not raw:
        return DEFAULT_K
    try:
        k = int(raw)
    except ValueError:
        raise ValueError(f"{ENV_K}={raw!r} is not an integer") from None
    return max(1, min(MAX_K, k))


def block_m(m: int) -> int:
    return 16 if m <= 16 else 32 if m <= 32 else 64


# ----------------------------------------------------------------------------------------------- quantization
def quantize(fmt: str, W, chunk: int = 32768) -> dict:
    """Quantize a [V, D] head row-chunked (bounded fp32 transient). FP8: q [V, D] e4m3fn, s [V] fp32.
    INT4: q [V, D/2] uint8 (see module docstring for the nibble layout), s [D/GROUP, V] fp16."""
    V, D = W.shape
    if fmt == "fp8":
        q = torch.empty((V, D), dtype=torch.float8_e4m3fn, device=W.device)
        s = torch.empty((V,), dtype=torch.float32, device=W.device)
        for r0 in range(0, V, chunk):
            wf = W[r0:r0 + chunk].float()
            sc = (wf.abs().amax(1) / 448.0).clamp_min(1e-12)
            q[r0:r0 + chunk] = (wf / sc[:, None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
            s[r0:r0 + chunk] = sc
        return {"q": q, "s": s}
    if fmt == "int4":
        G, half = D // GROUP, GROUP // 2
        q = torch.empty((V, D // 2), dtype=torch.uint8, device=W.device)
        s = torch.empty((G, V), dtype=torch.float16, device=W.device)
        for r0 in range(0, V, chunk):
            wf = W[r0:r0 + chunk].float().view(-1, G, GROUP)
            sc = (wf.abs().amax(-1) / 7.0).clamp_min(2.0 ** -14).half()                 # [r, G]
            qi = torch.round(wf / sc.float()[..., None]).clamp(-8, 7).to(torch.int16) + 8
            u = qi.to(torch.uint8).view(-1, G, 2, half)
            q[r0:r0 + chunk] = (u[:, :, 0, :] | (u[:, :, 1, :] << 4)).reshape(-1, D // 2)
            s[:, r0:r0 + chunk] = sc.t()
        return {"q": q, "s": s}
    raise ValueError(f"unknown format {fmt!r}")


def dequantize(fmt: str, w: dict, r0: int = 0, r1: int | None = None):
    """fp32 [r1 - r0, D] rows of the quantized head (torch reference / fallback)."""
    q = w["q"][r0:r1]
    if fmt == "fp8":
        return q.float() * w["s"][r0:r1].float()[:, None]
    n, d2 = q.shape
    G, half = (2 * d2) // GROUP, GROUP // 2
    p = q.view(n, G, half).to(torch.int16)
    vals = torch.cat([(p & 15) - 8, (p >> 4) - 8], dim=-1).float()               # [n, G, GROUP]
    return (vals * w["s"][:, r0:r1].t().float()[..., None]).reshape(n, 2 * d2)


def scan_torch(fmt: str, w: dict, h, n_vocab: int, suppress=(), topt: int = TOPT, block_n: int = BLOCK_N,
               chunk: int = 16384):
    """Torch twin of the fused Triton scan: per vocab block of ``block_n`` rows, the top ``topt`` (value, id) of
    h @ deq(W)^T with suppressed ids at -inf. Returns ([M, n_vocab/block_n*topt] fp32, same-shape int64 ids)."""
    M = h.shape[0]
    chunk = max(block_n, chunk // block_n * block_n)
    hf = h.float()
    vals, ids = [], []
    for r0 in range(0, n_vocab, chunk):
        r1 = min(n_vocab, r0 + chunk)
        lg = hf @ dequantize(fmt, w, r0, r1).t()                                     # [M, r]
        for sid in suppress:
            if r0 <= sid < r1:
                lg[:, sid - r0] = float("-inf")
        nb = (r1 - r0) // block_n
        v, i = lg.view(M, nb, block_n).topk(topt, dim=-1)
        base = r0 + torch.arange(nb, device=h.device, dtype=torch.long) * block_n
        vals.append(v.reshape(M, nb * topt))
        ids.append((i + base[None, :, None]).reshape(M, nb * topt))
    return torch.cat(vals, 1), torch.cat(ids, 1)


def rescore_torch(h, W, cands):
    """Exact BF16 semantics on candidate ids [M, k]: fp32 dot, round to BF16, max, ties -> lowest id."""
    ex = torch.bmm(W[cands].float(), h.float().unsqueeze(-1)).squeeze(-1)           # [M, k]
    ex = ex.to(torch.bfloat16).float()
    mx = ex.max(1, keepdim=True).values
    return torch.where(ex == mx, cands, torch.full_like(cands, _BIG)).min(1).values


def reference_argmax(h, W, suppress=()):
    """What upstream does: BF16 logits (fp32 accumulate, BF16 out), suppressed ids -> -inf, first-max argmax."""
    lg = torch.nn.functional.linear(h.to(W.dtype), W)
    if suppress:
        lg[:, list(suppress)] = float("-inf")
    return lg.argmax(-1)


# ---------------------------------------------------------------------------------------------------- runtime
class DraftHeadQ:
    """Quantized copy of a [V, D] BF16 head; ``top_tokens(h)`` -> greedy draft ids (int64 [M])."""

    def __init__(self, weight, fmt: str, k: int = DEFAULT_K, suppress=(), use_triton: bool | None = None):
        if torch is None:
            raise RuntimeError(f"{NAME}: torch is not importable")
        if fmt not in FORMATS:
            raise ValueError(f"{NAME}: format {fmt!r} not in {FORMATS}")
        V, D = weight.shape
        if V % BLOCK_N or D % GROUP or D % BLOCK_K:
            raise ValueError(f"{NAME}: head shape {tuple(weight.shape)} needs V % {BLOCK_N} == 0, D % {GROUP} == 0")
        if V * D >= 2 ** 31:
            raise ValueError(f"{NAME}: head too large for int32 offsets")
        suppress = tuple(int(x) for x in suppress)
        if len(suppress) > 2:
            raise ValueError(f"{NAME}: at most 2 suppressed token ids supported, got {suppress}")
        self.fmt, self.V, self.D = fmt, V, D
        self.k = max(1, min(MAX_K, int(k), (V // BLOCK_N) * TOPT))
        self.suppress = suppress
        self.sup = tuple(suppress) + (-1,) * (2 - len(suppress))
        self.weight = weight
        self.nblk = V // BLOCK_N
        self.ncand = self.nblk * TOPT
        if use_triton is None:
            use_triton = triton is not None and weight.is_cuda
        self.use_triton = bool(use_triton)
        with torch.no_grad():
            self.w = quantize(fmt, weight)

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.w.values())

    def top_tokens(self, h):
        h = h.reshape(-1, self.D)
        if h.shape[0] == 0:
            return torch.empty(0, dtype=torch.long, device=h.device)
        h = h.contiguous()
        with torch.no_grad():
            return self._triton(h) if self.use_triton else self._torch(h)

    def _torch(self, h):
        ov, oi = scan_torch(self.fmt, self.w, h, self.V, self.suppress)
        pos = torch.topk(ov, self.k, dim=1, sorted=False).indices
        return rescore_torch(h, self.weight, torch.gather(oi, 1, pos))

    def _triton(self, h):
        M = h.shape[0]
        bm = block_m(M)
        ov = torch.empty((M, self.ncand), device=h.device, dtype=torch.float32)
        oi = torch.empty((M, self.ncand), device=h.device, dtype=torch.int32)
        grid = (self.nblk, triton.cdiv(M, bm))
        common = dict(K=self.D, SUP0=self.sup[0], SUP1=self.sup[1], TOPT_=TOPT, BLOCK_M=bm, BLOCK_N_=BLOCK_N,
                      num_warps=SCAN_WARPS, num_stages=SCAN_STAGES)
        if self.fmt == "fp8":
            _g4_dhq_fp8_scan[grid](h, self.w["q"], self.w["s"], ov, oi, M, self.V, self.ncand, BLOCK_K_=BLOCK_K,
                                   **common)
        else:
            _g4_dhq_int4_scan[grid](h, self.w["q"], self.w["s"], ov, oi, M, self.V, self.ncand, GROUP_=GROUP,
                                    HALF=GROUP // 2, **common)
        pos = torch.topk(ov, self.k, dim=1, sorted=False).indices
        tok = torch.empty((M,), device=h.device, dtype=torch.long)
        _g4_dhq_rescore[(M,)](h, self.weight, pos, oi, tok, self.k, pos.stride(0), oi.stride(0), K=self.D,
                              KC_PAD=max(16, triton.next_power_of_2(self.k)), BLOCK_D=128, num_warps=4)
        return tok

    def _self_check(self, rows: int = 65, min_agree: float = 0.9) -> int:
        """Compile every BLOCK_M bucket outside graph capture and check the kernels against the BF16 argmax on
        random activations (synthetic recall there was 100 %; a packing/decode bug gives ~0 %)."""
        g = torch.Generator(device=self.weight.device).manual_seed(0)
        H = torch.randn(rows, self.D, device=self.weight.device, generator=g).to(self.weight.dtype)
        for m in (1, 17, 33):
            self.top_tokens(H[:m])
        got = self.top_tokens(H)
        ref = torch.cat([reference_argmax(H[i:i + 16], self.weight, self.suppress) for i in range(0, rows, 16)])
        agree = int((got == ref).sum())
        if agree < min_agree * rows:
            raise RuntimeError(f"{NAME}: {self.fmt} self-check failed: {agree}/{rows} rows match the BF16 argmax")
        return agree


def build_for(model) -> DraftHeadQ | None:
    """Called from the patched ``Gemma4MTP.load_weights``. None = inactive (upstream behaviour)."""
    fmt = requested_format()
    if fmt is None:
        return None
    if type(model).__name__ != "Gemma4MTP":
        logger.warning("%s: %s set but model is %s, not Gemma4MTP; inactive", NAME, ENV_FORMAT, type(model).__name__)
        return None
    if getattr(model, "masked_embedding", None) is not None:
        raise RuntimeError(f"{NAME}: centroid-masked head (use_ordered_embeddings) is not supported")
    lm_head = model.lm_head
    if getattr(lm_head, "tp_size", 1) != 1:
        raise RuntimeError(f"{NAME}: tensor-parallel draft head (tp_size={lm_head.tp_size}) is not supported")
    lp = model.logits_processor
    if getattr(lp, "soft_cap", None) is not None or getattr(lp, "scale", 1.0) != 1.0:
        raise RuntimeError(f"{NAME}: logit soft-cap/scale is not supported")
    W = lm_head.weight
    vocab = int(getattr(lm_head, "org_vocab_size", W.shape[0]))
    W = W[:vocab]
    if W.dtype != torch.bfloat16 or W.dim() != 2 or not W.is_contiguous():
        raise RuntimeError(f"{NAME}: expected a contiguous 2-D BF16 head, got {W.dtype} {tuple(W.shape)}")
    k = requested_k()
    dhq = DraftHeadQ(W, fmt, k=k, suppress=getattr(model, "_suppress_token_ids", None) or ())
    agree = None
    if dhq.use_triton:
        agree = dhq._self_check()
    else:
        logger.warning("%s: Triton/CUDA unavailable; using the slow torch reference path", NAME)
    logger.info("%s: Gemma4MTP greedy drafting via %s head scan (k=%d, top-%d per %d-row block, %.1f MiB, "
                "suppress=%s, self-check %s/65) + exact BF16 rescore", NAME, fmt, dhq.k, TOPT, BLOCK_N,
                dhq.nbytes() / 2 ** 20, list(dhq.suppress), agree)
    return dhq
