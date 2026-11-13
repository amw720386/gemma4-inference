"""Kernel lab v2: causal chunked-prefill attention for Gemma 4 12B on sm_120 (RTX PRO 6000 Blackwell), packed rows.

Same problem, layout and semantics as kernels/prefill_attn.py (v1; read its docstring first): one program owns
BLOCK_M (query token, head-in-group) rows that share one KV head (row r = token r // G, head r % G), so every K/V
tile is read once for BLOCK_M / G tokens x G heads. Online softmax in exp2, fp32 accumulation, P cast to the KV dtype
for the PV MMA (as FA2 / vLLM) -> same math up to FP reduction order (lossless).

Why v1 stops at 100-137 TFLOPS (dense BF16 GEMM on this GPU: ~415-440):
  * L2 traffic. GB202 L2 sustains ~8.7 TB/s (Chips and Cheese, RTX PRO 6000 Blackwell), i.e. ~19 B/clk/SM against
    ~890 dense-MMA FLOP/clk/SM: an attention tile must do >= ~46 FLOP per L2 byte for 400 TFLOPS, >= ~23 for 200.
    v1's best global variant re-reads its Q chunk from L2 for every KV tile (Q tile 64 KB + K 32 KB + V 32 KB per
    4.2 MFLOP = 33 FLOP/B); with Q resident only K/V stream from L2 and FLOP/B = BLOCK_M (64-128).
  * Layout heuristics. Triton picks the MMA warp grid itself; for BLOCK_M = 64 and 8 warps the chained-dot layout is
    [4, 2] (Q replicated over warp pairs, S/P exchanged through shared memory), and for a [128, 256] PV accumulator
    it may pick [1, 8]. Only BLOCK_M = 16 x num_warps guarantees "each warp owns 16 full rows" (FA2's layout).
  * No explicit pipelining of the d-chunked QK loop nested inside the KV loop.
sm_120 has only warp-level mma.sync (m16n8k16) - no wgmma / tcgen05 / TMEM - so the Hopper/Blackwell-datacenter
FA3/FA4 techniques (warp-specialised producer/consumer warpgroups, async wgmma ping-pong, TMEM accumulators) do not
apply; what transfers is FA2's structure plus explicit control of shared memory.

Designs (the bench sweeps all; every variant is fail-soft):
  tl  Triton-language FA2, refined v1: BLOCK_M = 16 x num_warps where possible (FA2 warp layout), K/V through host
      TMA tensor descriptors (sm_120 has TMA; no address math / registers for K/V, zero-filled tails), deeper
      num_stages, scale folded into one FMA per logit, optional maxnreg, optional split-KV.
  gl  Gluon (Triton 3.8's explicit-layout dialect) FA2 for mma.sync: Q tile loaded ONCE into shared memory with
      cp.async and re-read per KV tile in BLOCK_D slices through ldmatrix (FA2's default; Q never hits L2 again);
      K and V in separate single-buffered tiles with FA2's interleaved copy schedule (V(j) copy issued before QK(j),
      K(j+1) copy issued before PV(j)); explicit warp grid [WM, WN]: WN = 1 for head_dim 256 (each warp 16 full rows,
      P stays in registers via k_width 2), WN = 2-4 for head_dim 512 (the 512-wide fp32 accumulator split over warp
      pairs: 128 regs/thread instead of 256; row max/sum and P cross warps through shared memory, done by the
      compiler) - no QK recomputation, unlike v1's dv-split programs. kv_shared: K and V share one tile buffer
      (less smem, less overlap).
  split-KV (both): SPLITS > 1 cuts each row block's KV range into SPLITS contiguous tile ranges (more, smaller work
      items -> less wave-quantisation tail on the long-prefix chunks, e.g. 6144:2048 = 2.7 waves at 1 CTA/SM);
      partial unnormalised O (fp32) + running max + sum go to a workspace and _merge_splits combines them
      (LSE merge, fp32: lossless). Costs one fp32 O write+read per split.
Work order: heaviest (latest) row blocks first (longest causal prefix), as in v1.

Not done here (and why): persistent kernels with an atomic work queue (the hardware block scheduler already does
greedy heaviest-first; the remaining tail is granularity, which split-KV addresses); exp2 polynomial emulation
(FA4; at head_dim 256/512 there is 1 exp per 1024-2048 MMA FLOPs, MUFU is not the limit); CuTe DSL (CUTLASS 4.8.0
ships sm_120 GEMM examples - blackwell_geforce - but its attention examples target sm_90/sm_100 wgmma/tcgen05; the
Ampere CuTe-DSL FA2 example would be the starting point for a CuTe port, left as follow-up).

Layout: q [T, H, D], k/v [S, KVH, D] (S >= start + T, last dim contiguous), out [T, H, D]; chunk queries sit at
positions [start, start + T) and see key j iff j <= pos and (window == 0 or pos - j < window).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
import triton
import triton.language as tl

LOG2E = 1.4426950408889634
SMEM_LIMIT = 101376  # sm_120 max dynamic shared memory per block (99 KB)

try:  # Gluon (Triton >= 3.5); the gl variants are skipped without it
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl
    from triton.experimental.gluon.language.nvidia.ampere import async_copy as cp
    from triton.experimental.gluon.language.nvidia.ampere import mma_v2
    GLUON_ERROR = None
except Exception as e:  # noqa: BLE001
    gluon = gl = cp = mma_v2 = None
    GLUON_ERROR = repr(e)[:300]

try:  # host-side TMA descriptors (Triton >= 3.4)
    from triton.tools.tensor_descriptor import TensorDescriptor
    TMA_ERROR = None
except Exception as e:  # noqa: BLE001
    TensorDescriptor = None
    TMA_ERROR = repr(e)[:300]


@dataclass(frozen=True)
class Variant:
    name: str
    kind: str                # "tl" (Triton language) | "gl" (Gluon)
    block_m: int             # (token, head) rows per program; multiple of the group size
    block_n: int             # KV tokens per tile
    block_d: int             # QK head-dim chunk (tl: == head_dim -> Q held in registers; gl: smem slice width)
    block_dv: int = 0        # tl: output columns per program (0 = head_dim; < head_dim recomputes QK)
    num_warps: int = 8       # tl (gl: wm * wn)
    num_stages: int = 2      # tl software-pipelining depth
    tma: bool = False        # tl: K/V via host TMA tensor descriptors
    wm: int = 0              # gl warp grid rows (each warp row = 16 query rows)
    wn: int = 0              # gl warp grid columns (split of the KV tile in S and of head_dim in O)
    kw_qk: int = 8           # gl DotOperand k_width for Q/K (8: vectorised ldmatrix)
    kw_pv: int = 2           # gl k_width for P/V (2: P reuses the S accumulator registers, no shuffle)
    kv_shared: bool = False  # gl: K and V share one tile buffer
    splits: int = 1          # split-KV factor (>1: partial results + LSE merge kernel)
    maxnreg: int = 0         # tl: register cap per thread (0 = compiler default)

    @property
    def warps(self) -> int:
        return self.wm * self.wn if self.kind == "gl" else self.num_warps


def _tl(name, bm, bn, bd, w, s, tma=True, bdv=0, splits=1, maxnreg=0):
    return Variant(name, "tl", bm, bn, bd, bdv, w, s, tma, splits=splits, maxnreg=maxnreg)


def _gl(name, bm, bn, bd, wm, wn, kvs=False, splits=1, kw_qk=8, kw_pv=2):
    return Variant(name, "gl", bm, bn, bd, wm=wm, wn=wn, kv_shared=kvs, splits=splits, kw_qk=kw_qk, kw_pv=kw_pv)


# Global layer: head_dim 512, 16 q : 1 KV head (G = 16, BLOCK_M 32/64/128 = 2/4/8 tokens).
# gl smem = Q (bm x 512 x 2) + K (bn x 512 x 2) + V (same, unless kv_shared); regs/thread for O = 16 x 512 / wn / 32 x 4 B.
GLOBAL_VARIANTS = {v.name: v for v in [
    # (block_d 128 for 16-row KV tiles: the 8-warp cp.async layout then covers exactly 16 rows)
    _gl("gl_m64_n16_w4x2", 64, 16, 128, 4, 2),               # 4 tok; Q 64K + K 16K + V 16K = 96K; O 128 regs
    _gl("gl_m64_n16_w4x2_kvs", 64, 16, 128, 4, 2, kvs=True),  # 80K smem: room for the compiler's P/reduce scratch
    _gl("gl_m32_n32_w2x4", 32, 32, 64, 2, 4),                # 2 tok; Q 32K + K 32K + V 32K; O 64 regs
    _gl("gl_m32_n32_w2x2", 32, 32, 64, 2, 2),                # 4 warps; O 128 regs
    _gl("gl_m64_n16_w4x2_sk2", 64, 16, 128, 4, 2, splits=2),
    _gl("gl_m64_n16_w4x2_kw8", 64, 16, 128, 4, 2, kw_pv=8),   # P/V k_width 8 (vectorised V ldmatrix, P shuffled)
    _tl("tl_m64_dv512_n32_d64_tma_s2", 64, 32, 64, 8, 2),     # v1 best (m64_dv512_n32_d64) + TMA K/V
    _tl("tl_m64_dv512_n16_d64_tma_s3", 64, 16, 64, 8, 3),     # deeper pipeline on smaller KV tiles
    _tl("tl_m64_dv512_n32_d128_tma_s2", 64, 32, 128, 8, 2),   # half the d-chunk iterations
    _tl("tl_m128_dv256_n32_d64_tma_s2", 128, 32, 64, 8, 2, bdv=256),  # FA2 [8,1] warps; QK computed 2x
    _tl("tl_m64_dv512_n32_d64_tma_s2_sk2", 64, 32, 64, 8, 2, splits=2),
]}
# Sliding layer: head_dim 256, 16 q : 8 KV heads (G = 2), window 1024 (BLOCK_M 64/128 = 32/64 tokens).
SLIDING_VARIANTS = {v.name: v for v in [
    _gl("gl_m128_n32_w8x1", 128, 32, 64, 8, 1),              # Q 64K + K 16K + V 16K = 96K; O 128 regs
    _gl("gl_m128_n64_w8x1_kvs", 128, 64, 64, 8, 1, kvs=True),  # Q 64K + KV 32K
    _gl("gl_m64_n64_w4x1", 64, 64, 64, 4, 1),                # Q 32K + K 32K + V 32K
    _gl("gl_m64_n32_w4x1", 64, 32, 64, 4, 1),                # 64K
    _gl("gl_m128_n32_w8x1_sk2", 128, 32, 64, 8, 1, splits=2),
    _gl("gl_m128_n32_w8x1_kw8", 128, 32, 64, 8, 1, kw_pv=8),
    _tl("tl_m128_n32_q_w8_tma_s2", 128, 32, 256, 8, 2),       # Q in regs (64 regs), O 128 regs, [8,1] warps
    _tl("tl_m128_n32_q_w8_s2", 128, 32, 256, 8, 2, tma=False),  # same with pointer loads
    _tl("tl_m64_n32_q_w4_tma_s3", 64, 32, 256, 4, 3),         # FA2 hdim-256 shape: 4 warps, 3 stages (96K)
    _tl("tl_m64_n64_d64_w8_tma_s2", 64, 64, 64, 8, 2),        # v1 best (m64_dv256_n64_d64) + TMA K/V
    _tl("tl_m128_n32_q_w8_tma_s2_r255", 128, 32, 256, 8, 2, maxnreg=255),
    _tl("tl_m128_n32_q_w8_tma_s2_sk2", 128, 32, 256, 8, 2, splits=2),
]}


def variants_for(head_dim: int) -> dict:
    return GLOBAL_VARIANTS if head_dim == 512 else SLIDING_VARIANTS


def smem_bytes(v: Variant, head_dim: int, elt: int = 2) -> int:
    """Lower bound of the shared memory a variant needs (explicit buffers only; gl: + the P exchange tile)."""
    if v.kind == "gl":
        kv = v.block_n * head_dim * elt * (1 if v.kv_shared else 2)
        p_x = v.block_m * v.block_n * elt if v.wn > 1 else 0
        return v.block_m * head_dim * elt + kv + p_x
    dv = v.block_dv or head_dim
    return v.num_stages * v.block_n * (v.block_d + dv) * elt


def split_range(a: int, hi: int, block_n: int, splits: int, split: int) -> tuple[int, int]:
    """KV range [lo, hi) of one split of the tile-aligned range [a, hi) (mirrors the kernels' integer math)."""
    n_t = (hi - a + block_n - 1) // block_n
    per = (n_t + splits - 1) // splits
    lo = a + split * per * block_n
    return lo, min(lo + per * block_n, hi)


def check_variant(v: Variant, head_dim: int, group: int) -> None:
    assert v.block_m % group == 0, f"{v.name}: block_m must be a multiple of the group size {group}"
    assert head_dim % v.block_d == 0, v
    assert v.splits >= 1, v
    if v.kind == "gl":
        assert GLUON_ERROR is None, f"gluon unavailable: {GLUON_ERROR}"
        assert v.block_m == 16 * v.wm, f"{v.name}: gl needs block_m == 16 * wm"
        assert v.block_n % (8 * v.wn) == 0 and head_dim % (8 * v.wn) == 0, v
        assert v.block_d % 16 == 0 and v.block_n % 16 == 0, v
        tpc = min(32, v.block_d // 8)
        assert (32 // tpc) * v.warps <= v.block_n, f"{v.name}: copy layout rows exceed the KV tile (raise block_d)"
    else:
        dv = v.block_dv or head_dim
        assert head_dim % dv == 0, v
        assert not v.tma or TMA_ERROR is None, f"TMA descriptors unavailable: {TMA_ERROR}"
    need = smem_bytes(v, head_dim)
    assert need <= SMEM_LIMIT, f"{v.name}: needs >= {need} B shared memory > {SMEM_LIMIT}"


# ----------------------------------------------------------------------------------------------- tl (Triton language)

@triton.jit
def _tl_tiles(acc, l_i, m_i, q_all, q_rows, K, V, k_base, v_base, stride_ks, stride_vs, kvh, d0v, offs_dv, qpos,
              lo, hi, kv_hi, qk_scale,
              BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
              BLOCK_DV: tl.constexpr, Q_IN_REGS: tl.constexpr, MASK: tl.constexpr, WINDOW: tl.constexpr,
              USE_TMA: tl.constexpr):
    """Online-softmax update over KV tiles [lo, hi). MASK: causal/window masks + bounds against kv_hi. m_i is kept
    in the scaled log2 domain; exp2(s * qk_scale - m) is one FMA per logit."""
    for n0 in range(lo, hi, BLOCK_N):
        n0 = tl.multiple_of(n0, BLOCK_N)
        offs_n = n0 + tl.arange(0, BLOCK_N)
        n_ok = offs_n < kv_hi
        if Q_IN_REGS:
            if USE_TMA:
                k = tl.reshape(K.load([n0, kvh, 0]), (BLOCK_N, HEAD_DIM))
            else:
                k_ptrs = k_base + offs_n[:, None] * stride_ks + tl.arange(0, HEAD_DIM)[None, :]
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
                if USE_TMA:
                    kc = tl.reshape(K.load([n0, kvh, d0]), (BLOCK_N, BLOCK_D))
                else:
                    k_ptrs = k_base + offs_n[:, None] * stride_ks + offs_d[None, :]
                    if MASK:
                        kc = tl.load(k_ptrs, mask=n_ok[:, None], other=0.0)
                    else:
                        kc = tl.load(k_ptrs)
                s += tl.dot(qc, tl.trans(kc))
        if MASK:
            valid = (offs_n[None, :] <= qpos[:, None]) & n_ok[None, :]
            if WINDOW > 0:
                valid = valid & ((qpos[:, None] - offs_n[None, :]) < WINDOW)
            s = tl.where(valid, s, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, axis=1) * qk_scale)
            m_use = tl.where(m_new == float("-inf"), 0.0, m_new)  # row fully masked so far: keep exp2 finite
        else:
            m_new = tl.maximum(m_i, tl.max(s, axis=1) * qk_scale)
            m_use = m_new
        alpha = tl.exp2(m_i - m_use)
        p = tl.exp2(s * qk_scale - m_use[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        if USE_TMA:
            vt = tl.reshape(V.load([n0, kvh, d0v]), (BLOCK_N, BLOCK_DV))
        else:
            v_ptrs = v_base + offs_n[:, None] * stride_vs + offs_dv[None, :]
            if MASK:
                vt = tl.load(v_ptrs, mask=n_ok[:, None], other=0.0)
            else:
                vt = tl.load(v_ptrs)
        acc = acc * alpha[:, None] + tl.dot(p.to(vt.dtype), vt)
        m_i = m_new
    return acc, l_i, m_i


@triton.jit(do_not_specialize=["start", "T"])
def _fwd_tl(
    Q, K, V, Out, Opart, Mpart, Lpart,
    start, T, qk_scale,
    stride_qt, stride_qh, stride_ks, stride_kh, stride_vs, stride_vh, stride_ot, stride_oh,
    G: tl.constexpr, H: tl.constexpr, HEAD_DIM: tl.constexpr, WINDOW: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_DV: tl.constexpr,
    NUM_DV_BLOCKS: tl.constexpr, Q_IN_REGS: tl.constexpr, USE_TMA: tl.constexpr, SPLITS: tl.constexpr,
):
    pid0 = tl.program_id(0)
    kvh = tl.program_id(1)
    n_mb = (T * G + BLOCK_M - 1) // BLOCK_M
    per_mb: tl.constexpr = SPLITS * NUM_DV_BLOCKS
    mb = n_mb - 1 - pid0 // per_mb          # heaviest (latest) row blocks first
    rem = pid0 % per_mb
    split = rem // NUM_DV_BLOCKS
    dvb = rem % NUM_DV_BLOCKS               # dv programs of one row block adjacent -> share K/V/Q in L2

    offs_m = mb * BLOCK_M + tl.arange(0, BLOCK_M)
    row_ok = offs_m < T * G
    t = tl.minimum(offs_m // G, T - 1)      # padded rows duplicate the last token (never stored)
    g = offs_m % G
    qpos = start + t
    head = kvh * G + g
    q_rows = Q + t[:, None] * stride_qt + head[:, None] * stride_qh

    t_lo = (mb * BLOCK_M) // G
    t_hi = tl.minimum((mb * BLOCK_M + BLOCK_M - 1) // G, T - 1)
    hi = start + t_hi + 1
    c_full = ((start + t_lo + 1) // BLOCK_N) * BLOCK_N  # tiles ending <= here are causal-visible to all rows
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
    # split-KV: this program covers tiles [s_lo, s_hi) of [a, hi)
    n_t = (hi - a + BLOCK_N - 1) // BLOCK_N
    per = (n_t + SPLITS - 1) // SPLITS
    s_lo = a + split * per * BLOCK_N
    s_hi = tl.minimum(s_lo + per * BLOCK_N, hi)

    if USE_TMA:  # K, V are tensor descriptors: no pointer math on them
        k_base = Q
        v_base = Q
    else:
        k_base = K + kvh * stride_kh
        v_base = V + kvh * stride_vh
    d0v = dvb * BLOCK_DV
    offs_dv = d0v + tl.arange(0, BLOCK_DV)
    if Q_IN_REGS:
        q_all = tl.load(q_rows + tl.arange(0, HEAD_DIM)[None, :])
    else:
        q_all = tl.zeros([1], dtype=tl.float32)  # unused

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DV], dtype=tl.float32)

    if WINDOW > 0:
        acc, l_i, m_i = _tl_tiles(acc, l_i, m_i, q_all, q_rows, K, V, k_base, v_base, stride_ks, stride_vs, kvh,
                                  d0v, offs_dv, qpos, tl.maximum(a, s_lo), tl.minimum(b, s_hi), hi, qk_scale,
                                  BLOCK_M, HEAD_DIM, BLOCK_N, BLOCK_D, BLOCK_DV, Q_IN_REGS, True, WINDOW, USE_TMA)
    acc, l_i, m_i = _tl_tiles(acc, l_i, m_i, q_all, q_rows, K, V, k_base, v_base, stride_ks, stride_vs, kvh,
                              d0v, offs_dv, qpos, tl.maximum(b, s_lo), tl.minimum(c, s_hi), hi, qk_scale,
                              BLOCK_M, HEAD_DIM, BLOCK_N, BLOCK_D, BLOCK_DV, Q_IN_REGS, False, WINDOW, USE_TMA)
    acc, l_i, m_i = _tl_tiles(acc, l_i, m_i, q_all, q_rows, K, V, k_base, v_base, stride_ks, stride_vs, kvh,
                              d0v, offs_dv, qpos, tl.maximum(c, s_lo), s_hi, hi, qk_scale,
                              BLOCK_M, HEAD_DIM, BLOCK_N, BLOCK_D, BLOCK_DV, Q_IN_REGS, True, WINDOW, USE_TMA)

    if SPLITS == 1:
        out = acc / tl.where(l_i > 0.0, l_i, 1.0)[:, None]
        o_ptrs = Out + t[:, None] * stride_ot + head[:, None] * stride_oh + offs_dv[None, :]
        tl.store(o_ptrs, out.to(Out.dtype.element_ty), mask=row_ok[:, None])
    else:
        prow = (split * T + t) * H + head
        tl.store(Opart + prow[:, None] * HEAD_DIM + offs_dv[None, :], acc, mask=row_ok[:, None])
        if dvb == 0:
            tl.store(Mpart + prow, m_i, mask=row_ok)
            tl.store(Lpart + prow, l_i, mask=row_ok)


@triton.jit
def _merge_splits(Opart, Mpart, Lpart, Out, T, stride_ot, stride_oh,
                  H: tl.constexpr, HEAD_DIM: tl.constexpr, SPLITS: tl.constexpr, BLOCK_R: tl.constexpr):
    """out[row] = sum_s 2^(m_s - m) O_s / sum_s 2^(m_s - m) l_s (fp32; rows = t * H + head)."""
    TH = T * H
    r = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    ok = r < TH
    d = tl.arange(0, HEAD_DIM)
    m = tl.full([BLOCK_R], float("-inf"), dtype=tl.float32)
    for s in tl.static_range(SPLITS):
        m = tl.maximum(m, tl.load(Mpart + s * TH + r, mask=ok, other=float("-inf")))
    m_use = tl.where(m == float("-inf"), 0.0, m)
    acc = tl.zeros([BLOCK_R, HEAD_DIM], dtype=tl.float32)
    l_tot = tl.zeros([BLOCK_R], dtype=tl.float32)
    for s in tl.static_range(SPLITS):
        w = tl.exp2(tl.load(Mpart + s * TH + r, mask=ok, other=float("-inf")) - m_use)
        l_tot += w * tl.load(Lpart + s * TH + r, mask=ok, other=0.0)
        o = tl.load(Opart + (s * TH + r)[:, None] * HEAD_DIM + d[None, :], mask=ok[:, None], other=0.0)
        acc += w[:, None] * o
    out = acc / tl.where(l_tot > 0.0, l_tot, 1.0)[:, None]
    t = r // H
    h = r % H
    tl.store(Out + t[:, None] * stride_ot + h[:, None] * stride_oh + d[None, :], out.to(Out.dtype.element_ty),
             mask=ok[:, None])


# ------------------------------------------------------------------------------------------------------- gl (Gluon)

if gluon is not None:

    @gluon.jit
    def _gl_issue_tile(smem, base, n0, n_hi, stride_s,
                       ROWS: gl.constexpr, HEAD_DIM: gl.constexpr, BLOCK_D: gl.constexpr, blk: gl.constexpr):
        """cp.async rows [n0, n0 + ROWS) (rows >= n_hi zero-filled) of a [*, HEAD_DIM] tensor into smem, in
        BLOCK_D-wide column slices (bounded address registers), as one commit group."""
        offs_n = n0 + gl.arange(0, ROWS, layout=gl.SliceLayout(1, blk))
        offs_d = gl.arange(0, BLOCK_D, layout=gl.SliceLayout(0, blk))
        msk = gl.expand_dims(offs_n < n_hi, 1) & gl.expand_dims(offs_d < BLOCK_D, 0)
        row = gl.expand_dims(offs_n * stride_s, 1)
        for i in gl.static_range(HEAD_DIM // BLOCK_D):
            cp.async_load(smem.slice(i * BLOCK_D, BLOCK_D, dim=1), base + row + gl.expand_dims(i * BLOCK_D + offs_d, 0),
                          mask=msk)
        cp.commit_group()

    @gluon.jit(do_not_specialize=["start", "T"])
    def _fwd_gl(
        Q, K, V, Out, Opart, Mpart, Lpart,
        start, T, qk_scale,
        stride_qt, stride_qh, stride_ks, stride_kh, stride_vs, stride_vh, stride_ot, stride_oh,
        G: gl.constexpr, H: gl.constexpr, HEAD_DIM: gl.constexpr, WINDOW: gl.constexpr,
        BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_D: gl.constexpr,
        WM: gl.constexpr, WN: gl.constexpr, NW: gl.constexpr, TPR: gl.constexpr, TPC: gl.constexpr,
        KW_QK: gl.constexpr, KW_PV: gl.constexpr, KV_SHARED: gl.constexpr, SPLITS: gl.constexpr,
    ):
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

        pid0 = gl.program_id(0)
        kvh = gl.program_id(1)
        n_mb = (T * G + BLOCK_M - 1) // BLOCK_M
        mb = n_mb - 1 - pid0 // SPLITS           # heaviest (latest) row blocks first
        split = pid0 % SPLITS

        t_lo = (mb * BLOCK_M) // G
        t_hi = gl.minimum((mb * BLOCK_M + BLOCK_M - 1) // G, T - 1)
        hi = start + t_hi + 1
        if WINDOW > 0:
            a = (gl.maximum(start + t_lo - WINDOW + 1, 0) // BLOCK_N) * BLOCK_N
        else:
            a = start * 0
        n_t = (hi - a + BLOCK_N - 1) // BLOCK_N
        per = (n_t + SPLITS - 1) // SPLITS
        j_lo = a + split * per * BLOCK_N
        j_hi = gl.minimum(j_lo + per * BLOCK_N, hi)

        q_smem = gl.allocate_shared_memory(Q.dtype.element_ty, [BLOCK_M, HEAD_DIM], smem_l)
        k_smem = gl.allocate_shared_memory(K.dtype.element_ty, [BLOCK_N, HEAD_DIM], smem_l)
        if KV_SHARED:
            v_smem = k_smem
        else:
            v_smem = gl.allocate_shared_memory(V.dtype.element_ty, [BLOCK_N, HEAD_DIM], smem_l)

        # Q tile -> smem once (rows past T * G zero-filled, never stored)
        offs_qr = mb * BLOCK_M + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, blk))
        q_row = (offs_qr // G) * stride_qt + (kvh * G + offs_qr % G) * stride_qh
        offs_qd = gl.arange(0, BLOCK_D, layout=gl.SliceLayout(0, blk))
        q_msk = gl.expand_dims(offs_qr < T * G, 1) & gl.expand_dims(offs_qd < BLOCK_D, 0)
        for i in gl.static_range(HEAD_DIM // BLOCK_D):
            cp.async_load(q_smem.slice(i * BLOCK_D, BLOCK_D, dim=1),
                          Q + gl.expand_dims(q_row, 1) + gl.expand_dims(i * BLOCK_D + offs_qd, 0), mask=q_msk)
        cp.commit_group()

        k_base = K + kvh * stride_kh
        v_base = V + kvh * stride_vh
        _gl_issue_tile(k_smem, k_base, j_lo, j_hi, stride_ks, BLOCK_N, HEAD_DIM, BLOCK_D, blk)

        offs_m = mb * BLOCK_M + gl.arange(0, BLOCK_M, layout=row_l)
        qpos = start + gl.minimum(offs_m // G, T - 1)
        m_i = gl.full([BLOCK_M], float("-inf"), gl.float32, row_l)
        l_i = gl.zeros([BLOCK_M], gl.float32, row_l)
        acc = gl.zeros([BLOCK_M, HEAD_DIM], gl.float32, mma)

        for n0 in range(j_lo, j_hi, BLOCK_N):
            if KV_SHARED:
                cp.wait_group(0)
            else:
                _gl_issue_tile(v_smem, v_base, n0, j_hi, stride_vs, BLOCK_N, HEAD_DIM, BLOCK_D, blk)  # V(j) in flight
                cp.wait_group(1)                                                                       # Q, K(j) landed
            gl.barrier()
            s = gl.zeros([BLOCK_M, BLOCK_N], gl.float32, mma)
            for i in gl.static_range(HEAD_DIM // BLOCK_D):
                qa = q_smem.slice(i * BLOCK_D, BLOCK_D, dim=1).load(dot_a_qk)
                kb = k_smem.slice(i * BLOCK_D, BLOCK_D, dim=1).permute((1, 0)).load(dot_b_qk)
                s = mma_v2(qa, kb, s)
            gl.barrier()                                                                               # k_smem free
            if KV_SHARED:
                _gl_issue_tile(v_smem, v_base, n0, j_hi, stride_vs, BLOCK_N, HEAD_DIM, BLOCK_D, blk)
            else:
                _gl_issue_tile(k_smem, k_base, n0 + BLOCK_N, j_hi, stride_ks, BLOCK_N, HEAD_DIM, BLOCK_D, blk)

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
            pa = gl.convert_layout(p.to(V.dtype.element_ty), dot_a_pv)
            m_i = m_new

            if KV_SHARED:
                cp.wait_group(0)
            else:
                cp.wait_group(1)                                                                       # V(j) landed
            gl.barrier()
            vb = v_smem.load(dot_b_pv)
            acc = mma_v2(pa, vb, acc)
            gl.barrier()                                                                               # v_smem free
            if KV_SHARED:
                _gl_issue_tile(k_smem, k_base, n0 + BLOCK_N, j_hi, stride_ks, BLOCK_N, HEAD_DIM, BLOCK_D, blk)
        cp.wait_group(0)

        offs_d = gl.arange(0, HEAD_DIM, layout=col_l)
        t = gl.minimum(offs_m // G, T - 1)
        head = kvh * G + offs_m % G
        st_msk = gl.expand_dims(offs_m < T * G, 1) & gl.expand_dims(offs_d < HEAD_DIM, 0)
        if SPLITS == 1:
            out = acc / gl.expand_dims(gl.where(l_i > 0.0, l_i, 1.0), 1)
            o_ptrs = Out + gl.expand_dims(t * stride_ot + head * stride_oh, 1) + gl.expand_dims(offs_d, 0)
            gl.store(o_ptrs, out.to(Out.dtype.element_ty), mask=st_msk)
        else:
            prow = (split * T + t) * H + head
            gl.store(Opart + gl.expand_dims(prow * HEAD_DIM, 1) + gl.expand_dims(offs_d, 0), acc, mask=st_msk)
            gl.store(Mpart + prow, m_i, mask=offs_m < T * G)
            gl.store(Lpart + prow, l_i, mask=offs_m < T * G)


# ------------------------------------------------------------------------------------------------------------- host

_WS: dict = {}


def workspace(T: int, H: int, D: int, splits: int, device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split-KV partials (cached per shape: allocate once outside CUDA-graph capture)."""
    key = (str(device), T, H, D, splits)
    if key not in _WS:
        _WS[key] = (torch.empty((splits * T * H, D), dtype=torch.float32, device=device),
                    torch.empty((splits * T * H,), dtype=torch.float32, device=device),
                    torch.empty((splits * T * H,), dtype=torch.float32, device=device))
    return _WS[key]


def prefill_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, start: int, out: torch.Tensor,
                      variant: Variant, window: int = 0, scale: float = 1.0) -> torch.Tensor:
    """Causal (optionally sliding-window) prefill of one chunk: q [T, H, D] at positions [start, start+T) attends
    to keys [0, start+T) of k, v [S, KVH, D]. Writes and returns out [T, H, D]."""
    T, H, D = q.shape
    S, KVH, Dk = k.shape
    assert Dk == D and v.shape[0] == S and v.shape[1] == KVH and H % KVH == 0 and S >= start + T
    assert q.stride(2) == 1 and k.stride(2) == 1 and v.stride(2) == 1 and out.stride(2) == 1
    G = H // KVH
    vt = variant
    check_variant(vt, D, G)
    assert max(T * q.stride(0), T * out.stride(0), S * k.stride(0), S * v.stride(0), vt.splits * T * H * D) < 2**31
    n_mb = triton.cdiv(T * G, vt.block_m)
    qk_scale = float(scale) * LOG2E
    if vt.splits > 1:
        opart, mpart, lpart = workspace(T, H, D, vt.splits, q.device)
    else:
        opart = mpart = lpart = out  # unused
    strides = (q.stride(0), q.stride(1), k.stride(0), k.stride(1), v.stride(0), v.stride(1),
               out.stride(0), out.stride(1))
    if vt.kind == "gl":
        tpc = min(32, vt.block_d // 8)
        _fwd_gl[(n_mb * vt.splits, KVH)](
            q, k, v, out, opart, mpart, lpart, int(start), int(T), qk_scale, *strides,
            G=G, H=H, HEAD_DIM=D, WINDOW=int(window), BLOCK_M=vt.block_m, BLOCK_N=vt.block_n, BLOCK_D=vt.block_d,
            WM=vt.wm, WN=vt.wn, NW=vt.wm * vt.wn, TPR=32 // tpc, TPC=tpc, KW_QK=vt.kw_qk, KW_PV=vt.kw_pv,
            KV_SHARED=vt.kv_shared, SPLITS=vt.splits, num_warps=vt.wm * vt.wn,
        )
    else:
        dv = vt.block_dv or D
        n_dv = D // dv
        q_in_regs = vt.block_d == D
        if vt.tma:
            kd = TensorDescriptor(k, list(k.shape), list(k.stride()), [vt.block_n, 1, vt.block_d])
            vd = TensorDescriptor(v, list(v.shape), list(v.stride()), [vt.block_n, 1, dv])
        else:
            kd, vd = k, v
        extra = {"maxnreg": vt.maxnreg} if vt.maxnreg else {}
        _fwd_tl[(n_mb * vt.splits * n_dv, KVH)](
            q, kd, vd, out, opart, mpart, lpart, int(start), int(T), qk_scale, *strides,
            G=G, H=H, HEAD_DIM=D, WINDOW=int(window), BLOCK_M=vt.block_m, BLOCK_N=vt.block_n, BLOCK_D=vt.block_d,
            BLOCK_DV=dv, NUM_DV_BLOCKS=n_dv, Q_IN_REGS=q_in_regs, USE_TMA=vt.tma, SPLITS=vt.splits,
            num_warps=vt.num_warps, num_stages=vt.num_stages, **extra,
        )
    if vt.splits > 1:
        br = 16 if D <= 256 else 8
        _merge_splits[(triton.cdiv(T * H, br),)](opart, mpart, lpart, out, int(T), out.stride(0), out.stride(1),
                                                 H=H, HEAD_DIM=D, SPLITS=vt.splits, BLOCK_R=br, num_warps=4)
    return out


def with_splits(v: Variant, splits: int) -> Variant:
    return replace(v, name=f"{v.name}_sk{splits}", splits=splits)


def reference_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, start: int, scale: float = 1.0,
                        window: int = 0) -> torch.Tensor:
    """fp32 torch reference with the same semantics. Returns [T, H, D] fp32 (small shapes only)."""
    T, H, _ = q.shape
    S = start + T
    kv_h = k.shape[1]
    kb = k[:S].float().repeat_interleave(H // kv_h, dim=1)
    vb = v[:S].float().repeat_interleave(H // kv_h, dim=1)
    s = torch.einsum("thd,shd->hts", q.float(), kb) * scale
    qpos = start + torch.arange(T, device=q.device)
    kpos = torch.arange(S, device=q.device)
    mask = kpos[None, :] <= qpos[:, None]
    if window > 0:
        mask &= (qpos[:, None] - kpos[None, :]) < window
    s = s.masked_fill(~mask[None], float("-inf"))
    return torch.einsum("hts,shd->thd", torch.softmax(s, dim=-1), vb)


def valid_pairs(start: int, T: int, window: int = 0) -> int:
    """Number of (query, key) pairs a causal(/window) chunk actually needs."""
    if window <= 0:
        return T * start + T * (T + 1) // 2
    return sum(min(p + 1, window) for p in range(start, start + T))
