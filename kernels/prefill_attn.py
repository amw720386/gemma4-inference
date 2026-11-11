"""Kernel lab: FlashAttention-2-style causal prefill attention for Gemma 4 12B's GQA/MQA layers (packed rows).

Idea: vLLM v0.33.0's Triton unified attention (2D path, used for every prefill chunk, q_len > 8) packs BLOCK_M = 16
(token, head) rows per program (sm_120 default). For the 8 global layers (16 q heads : 1 KV head, head_dim 512)
that is ONE query token per program, so every query token streams the whole causal KV prefix separately, and the
16-row MMAs are tiny. Here one program owns BLOCK_M rows of the (query token, head-in-group) pairs that share one
KV head (row r = t * G + g, G = heads per KV head), so each K/V tile is loaded once for BLOCK_M / G tokens x G heads:
  global  G = 16: BLOCK_M 32/64/128 = 2/4/8 tokens x 16 heads
  sliding G = 2:  BLOCK_M 64/128     = 32/64 tokens x 2 heads (window 1024: KV blocks outside every row's window
                  are never visited)
Online softmax (exp2, running max/sum), fp32 accumulators, P cast to the KV dtype for the PV MMA exactly as vLLM
does -> same math up to reduction order. No split-K / merge pass: prefill has enough row blocks to fill 188 SMs.

head_dim 512 makes the O accumulator BLOCK_M x 512 fp32 large. Two ways (variants sweep both):
  * BLOCK_DV < HEAD_DIM: the output/V head dim is split over HEAD_DIM // BLOCK_DV programs (adjacent in the grid, so
    they share K/V and Q through L2); each recomputes S = Q K^T over the full head dim (extra QK FLOPs).
  * BLOCK_DV == HEAD_DIM with a small BLOCK_M (32-64), no recomputation; the accumulator lives in registers.
Q is either held in registers for the whole program (Q_IN_REGS, BLOCK_D == HEAD_DIM) or re-read per KV tile in
BLOCK_D chunks (L1/L2 hits) to free registers for a bigger accumulator.

Causal / window masking (chunked prefill): query token t sits at absolute position start + t and sees key j iff
j <= start + t and (window == 0 or start + t - j < window) (vLLM's window_size = (window - 1, 0) semantics). Per row
block the KV range is split into three loops over BLOCK_N-aligned tiles: [a, b) masked (window edge), [b, c)
unmasked (every key visible to every row: no mask, no bounds checks), [c, hi) masked (causal diagonal + tail).

Layout (micro-benchmark; vLLM's paged cache needs a block-table gather in the K/V address math, a follow-up):
  q    [T, num_q_heads, HEAD_DIM]       the chunk's queries (batch 1), last dim contiguous
  k, v [S, num_kv_heads, HEAD_DIM]      S >= start + T, any token/head strides, last dim contiguous
  out  [T, num_q_heads, HEAD_DIM]
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

LOG2E = 1.4426950408889634


@dataclass(frozen=True)
class Variant:
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


# Global layer: head_dim 512, 16 q : 1 KV head. Shared-memory / register notes (bf16, sm_120 ~99 KB smem per block,
# 64K regs per SM): K or V tile = block_n x 512 x 2 B (32 KB at n32); acc = block_m x block_dv fp32 / (32 x warps).
GLOBAL_VARIANTS = {v.name: v for v in [
    _v("m32_dv512_n32_q", 32, 32, 512, 512, 8, 1, True),        # 2 tok; acc 64 regs/thr, Q 32 KB regs; K+V 64 KB
    _v("m32_dv512_n16_q", 32, 16, 512, 512, 4, 2, True),        # 2 tok; 4 warps; K+V 32 KB x 2 stages
    _v("m64_dv512_n32_d64", 64, 32, 64, 512, 8, 2, False),      # 4 tok; acc 128 regs/thr (risky: spills)
    _v("m64_dv512_n32_d64_w16", 64, 32, 64, 512, 16, 2, False),  # same with 16 warps: acc 64 regs/thr
    _v("m64_dv256_n32_q", 64, 32, 512, 256, 8, 1, True),        # 4 tok; 2 dv programs; Q 64 regs + acc 64 regs
    _v("m64_dv256_n64_d64", 64, 64, 64, 256, 8, 2, False),      # 4 tok; 2 dv programs, Q chunked
    _v("m128_dv128_n64_d64", 128, 64, 64, 128, 8, 2, False),    # 8 tok; 4 dv programs (QK recomputed 4x)
    _v("m128_dv256_n32_d64_w16", 128, 32, 64, 256, 16, 2, False),  # 8 tok; 2 dv programs; acc 64 regs/thr
    _v("m16_dv512_n32_q", 16, 32, 512, 512, 4, 1, True),        # control: 1 token per program (vLLM-like rows)
]}
# Sliding layer: head_dim 256, 16 q : 8 KV heads (G = 2), window 1024. K or V tile = block_n x 256 x 2 B.
SLIDING_VARIANTS = {v.name: v for v in [
    _v("m64_dv256_n32_q", 64, 32, 256, 256, 8, 1, True),        # 32 tok; acc 64 regs/thr, Q 32 KB regs
    _v("m64_dv256_n32_q_s2", 64, 32, 256, 256, 8, 2, True),     # same, 2 stages (K+V 32 KB x 2)
    _v("m64_dv256_n64_d64", 64, 64, 64, 256, 8, 2, False),      # Q chunked, bigger KV tile
    _v("m128_dv256_n32_d64_w16", 128, 32, 64, 256, 16, 2, False),  # 64 tok; acc 64 regs/thr at 16 warps
    _v("m128_dv256_n32_d64", 128, 32, 64, 256, 8, 2, False),    # 64 tok; acc 128 regs/thr (risky)
    _v("m128_dv128_n64_d64", 128, 64, 64, 128, 8, 2, False),    # 2 dv programs (QK recomputed 2x)
    _v("m32_dv256_n64_q", 32, 64, 256, 256, 4, 1, True),        # 16 tok, 4 warps
]}


@triton.jit
def _attn_tiles(acc, l_i, m_i, q_all, q_rows, k_base, v_base, stride_ks, stride_vs, offs_dv, qpos,
                lo, hi, kv_hi, qk_scale,
                BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
                Q_IN_REGS: tl.constexpr, MASK: tl.constexpr, WINDOW: tl.constexpr):
    """Online-softmax update over KV tiles [lo, hi) (lo BLOCK_N-aligned). MASK: apply causal/window masks and
    bounds-check loads against kv_hi; otherwise every key in the range is visible to every row."""
    for n0 in range(lo, hi, BLOCK_N):
        n0 = tl.multiple_of(n0, BLOCK_N)
        offs_n = n0 + tl.arange(0, BLOCK_N)
        n_ok = offs_n < kv_hi
        if Q_IN_REGS:
            offs_d = tl.arange(0, HEAD_DIM)
            k_ptrs = k_base + offs_n[:, None] * stride_ks + offs_d[None, :]
            if MASK:
                k = tl.load(k_ptrs, mask=n_ok[:, None], other=0.0)
            else:
                k = tl.load(k_ptrs)
            s = tl.dot(q_all, tl.trans(k))
        else:
            s = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            for d0 in range(0, HEAD_DIM, BLOCK_D):  # rolled (constexpr bounds, plain range)
                offs_d = d0 + tl.arange(0, BLOCK_D)
                qc = tl.load(q_rows + offs_d[None, :])
                k_ptrs = k_base + offs_n[:, None] * stride_ks + offs_d[None, :]
                if MASK:
                    kc = tl.load(k_ptrs, mask=n_ok[:, None], other=0.0)
                else:
                    kc = tl.load(k_ptrs)
                s += tl.dot(qc, tl.trans(kc))
        s = s * qk_scale  # log2 domain: scale * log2(e)
        if MASK:
            valid = (offs_n[None, :] <= qpos[:, None]) & n_ok[None, :]
            if WINDOW > 0:
                valid = valid & ((qpos[:, None] - offs_n[None, :]) < WINDOW)
            s = tl.where(valid, s, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            m_use = tl.where(m_new == float("-inf"), 0.0, m_new)  # fully masked row so far: keep exp2 finite
        else:
            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            m_use = m_new
        alpha = tl.exp2(m_i - m_use)
        p = tl.exp2(s - m_use[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        v_ptrs = v_base + offs_n[:, None] * stride_vs + offs_dv[None, :]
        if MASK:
            vt = tl.load(v_ptrs, mask=n_ok[:, None], other=0.0)
        else:
            vt = tl.load(v_ptrs)
        acc = acc * alpha[:, None] + tl.dot(p.to(vt.dtype), vt)
        m_i = m_new
    return acc, l_i, m_i


@triton.jit(do_not_specialize=["start", "T"])
def _prefill_attn_fwd(
    Q, K, V, Out,
    start, T, qk_scale,
    stride_qt, stride_qh,
    stride_ks, stride_kh,
    stride_vs, stride_vh,
    stride_ot, stride_oh,
    G: tl.constexpr, HEAD_DIM: tl.constexpr, WINDOW: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_DV: tl.constexpr,
    NUM_DV_BLOCKS: tl.constexpr, Q_IN_REGS: tl.constexpr,
):
    pid0 = tl.program_id(0)
    kvh = tl.program_id(1)
    n_mb = (T * G + BLOCK_M - 1) // BLOCK_M
    mb = n_mb - 1 - pid0 // NUM_DV_BLOCKS  # heaviest (latest) row blocks first: shorter causal tail
    dvb = pid0 % NUM_DV_BLOCKS              # dv programs of one row block adjacent -> share K/V/Q in L2

    offs_m = mb * BLOCK_M + tl.arange(0, BLOCK_M)
    row_ok = offs_m < T * G
    t = tl.minimum(offs_m // G, T - 1)      # padded rows duplicate the last token (never stored)
    g = offs_m % G
    qpos = start + t
    q_rows = Q + t[:, None] * stride_qt + (kvh * G + g)[:, None] * stride_qh

    t_lo = (mb * BLOCK_M) // G
    t_hi = tl.minimum((mb * BLOCK_M + BLOCK_M - 1) // G, T - 1)
    hi = start + t_hi + 1                   # exclusive end of keys any row needs
    c_full = ((start + t_lo + 1) // BLOCK_N) * BLOCK_N  # tiles ending <= here are causal-visible to all rows
    if WINDOW > 0:
        lo = tl.maximum(start + t_lo - WINDOW + 1, 0)
        a = (lo // BLOCK_N) * BLOCK_N
        w_full = tl.maximum(start + t_hi - WINDOW + 1, 0)  # keys >= here are in-window for all rows
        w_full_blk = ((w_full + BLOCK_N - 1) // BLOCK_N) * BLOCK_N
    else:
        a = start * 0
        w_full_blk = start * 0
    b = tl.maximum(tl.minimum(w_full_blk, c_full), a)
    c = tl.maximum(c_full, b)

    k_base = K + kvh * stride_kh
    v_base = V + kvh * stride_vh
    offs_dv = dvb * BLOCK_DV + tl.arange(0, BLOCK_DV)
    if Q_IN_REGS:
        q_all = tl.load(q_rows + tl.arange(0, HEAD_DIM)[None, :])
    else:
        q_all = tl.zeros([1], dtype=tl.float32)  # unused

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DV], dtype=tl.float32)

    if WINDOW > 0:
        acc, l_i, m_i = _attn_tiles(acc, l_i, m_i, q_all, q_rows, k_base, v_base, stride_ks, stride_vs, offs_dv,
                                    qpos, a, b, hi, qk_scale, BLOCK_M, HEAD_DIM, BLOCK_N, BLOCK_D, Q_IN_REGS,
                                    True, WINDOW)
    acc, l_i, m_i = _attn_tiles(acc, l_i, m_i, q_all, q_rows, k_base, v_base, stride_ks, stride_vs, offs_dv,
                                qpos, b, c, hi, qk_scale, BLOCK_M, HEAD_DIM, BLOCK_N, BLOCK_D, Q_IN_REGS, False, WINDOW)
    acc, l_i, m_i = _attn_tiles(acc, l_i, m_i, q_all, q_rows, k_base, v_base, stride_ks, stride_vs, offs_dv,
                                qpos, c, hi, hi, qk_scale, BLOCK_M, HEAD_DIM, BLOCK_N, BLOCK_D, Q_IN_REGS, True, WINDOW)

    out = acc / tl.where(l_i > 0.0, l_i, 1.0)[:, None]
    o_ptrs = Out + t[:, None] * stride_ot + (kvh * G + g)[:, None] * stride_oh + offs_dv[None, :]
    tl.store(o_ptrs, out.to(Out.dtype.element_ty), mask=row_ok[:, None])


def check_variant(v: Variant, head_dim: int, group: int) -> None:
    assert head_dim % v.block_dv == 0 and head_dim % v.block_d == 0, v
    assert (not v.q_in_regs) or v.block_d == head_dim, v
    assert v.block_m % group == 0, f"{v.name}: block_m must be a multiple of the group size {group}"


def prefill_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, start: int, out: torch.Tensor,
                      variant: Variant, window: int = 0, scale: float = 1.0) -> torch.Tensor:
    """Causal (optionally sliding-window) prefill of one chunk: q [T, H, D] at positions [start, start+T) attends
    to keys [0, start+T) of k, v [S, KVH, D]. Writes and returns out [T, H, D]."""
    T, H, D = q.shape
    S, KVH, Dk = k.shape
    assert Dk == D and v.shape[0] == S and v.shape[1] == KVH and H % KVH == 0 and S >= start + T
    assert q.stride(2) == 1 and k.stride(2) == 1 and v.stride(2) == 1 and out.stride(2) == 1
    G = H // KVH
    check_variant(variant, D, G)
    # 32-bit address math in the kernel
    assert max(T * q.stride(0), T * out.stride(0), S * k.stride(0), S * v.stride(0)) < 2**31
    n_dv = D // variant.block_dv
    n_mb = triton.cdiv(T * G, variant.block_m)
    grid = (n_mb * n_dv, KVH)
    _prefill_attn_fwd[grid](
        q, k, v, out,
        int(start), int(T), float(scale) * LOG2E,
        q.stride(0), q.stride(1),
        k.stride(0), k.stride(1),
        v.stride(0), v.stride(1),
        out.stride(0), out.stride(1),
        G=G, HEAD_DIM=D, WINDOW=int(window),
        BLOCK_M=variant.block_m, BLOCK_N=variant.block_n, BLOCK_D=variant.block_d, BLOCK_DV=variant.block_dv,
        NUM_DV_BLOCKS=n_dv, Q_IN_REGS=variant.q_in_regs,
        num_warps=variant.num_warps, num_stages=variant.num_stages,
    )
    return out


def reference_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, start: int, scale: float = 1.0,
                        window: int = 0) -> torch.Tensor:
    """fp32 torch reference with the same semantics. Returns [T, H, D] fp32 (small shapes only)."""
    T, H, _ = q.shape
    S = start + T
    kv_h = k.shape[1]
    kb = k[:S].float().repeat_interleave(H // kv_h, dim=1)  # q head h -> kv head h // G (vLLM mapping)
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
