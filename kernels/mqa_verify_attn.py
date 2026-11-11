"""Kernel lab: split-K flash-decoding attention for speculative verify (q_len query tokens per request).

Idea: vLLM v0.33.0's Triton unified attention packs BLOCK_M = 16 query rows per program, ordered (token, head).
For Gemma 4's global layers (16 q heads : 1 KV head, head_dim 512) that is BLOCK_Q = 16 // 16 = 1 token per
program, so each of the 7 verify tokens of a request streams the whole KV separately. Here one program packs
BLOCK_M of the q_len x (heads per KV head) rows of a request (all 7 x 16 = 112 rows -> BLOCK_M 128 in the packed
variants), so each KV tile is fetched once per row block instead of once per token.

A 128 x 512 fp32 accumulator (256 KB) cannot live on chip, so the V/output head dim is split into
NUM_DV_BLOCKS = HEAD_DIM // BLOCK_DV programs (adjacent in the grid, so they share the KV tile through L2); each
recomputes S = Q K^T over the full head dim. The sequence is split into num_splits ranges (flash-decoding);
a second kernel merges the per-split (acc, max, sum) partials with a log-sum-exp reduction.

Layout (micro-benchmark only, NOT vLLM's paged cache; integrating with block tables is a separate step):
  q    [num_seqs * Q_LEN, num_q_heads, HEAD_DIM]      (vLLM's packed query layout, uniform q_len per request)
  k, v [num_seqs, max_seq, num_kv_heads, HEAD_DIM]    any strides, last dim contiguous
  seq_lens [num_seqs] int32: tokens in the KV including the Q_LEN new ones (vLLM's seqused_k)
Query token t of request b sits at absolute position seq_len - Q_LEN + t and attends causally (and within
`window` if > 0, vLLM's SLIDING_WINDOW = window semantics: key j visible iff 0 <= pos - j < window).
fp32 accumulation, scale passed in (Gemma 4: 1.0), optional tanh softcap (Gemma 4 has no attention softcap).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

LOG2E = 1.4426950408889634


@dataclass(frozen=True)
class Variant:
    name: str
    block_m: int     # query rows (token-major, head-minor) per program
    block_n: int     # KV tokens per tile
    block_d: int     # head-dim chunk for Q K^T (== head_dim with q_in_regs)
    block_dv: int    # output/V head-dim columns per program
    num_warps: int
    num_stages: int
    q_in_regs: bool  # load the program's whole Q tile once (needs block_d == head_dim)


# Global layer (head_dim 512, 16 q : 1 kv, q_len 7 -> 112 rows).
GLOBAL_VARIANTS = {
    v.name: v for v in [
        Variant("m128_dv128_n64", 128, 64, 64, 128, 8, 2, False),   # all 7 tokens in one tile; 4 dv programs
        Variant("m64_dv256_n64", 64, 64, 64, 256, 8, 2, False),     # 4 tokens per tile; 2 row blocks x 2 dv
        Variant("m32_dv512_n32", 32, 32, 512, 512, 8, 1, True),     # 2 tokens per tile; full head dim, Q in regs
        Variant("m16_dv512_n32", 16, 32, 512, 512, 4, 1, True),     # control: 1 token per tile (vLLM-like)
    ]
}
# Sliding layer (head_dim 256, 16 q : 8 kv, q_len 7 -> 14 rows per KV head, window 1024).
SLIDING_VARIANTS = {
    v.name: v for v in [
        Variant("m16_dv256_n32", 16, 32, 256, 256, 4, 2, True),
    ]
}


@triton.jit
def _tanh(x):
    return 2.0 / (1.0 + tl.exp(-2.0 * x)) - 1.0


@triton.jit(do_not_specialize=["num_splits"])
def _mqa_verify_partial(
    Q, K, V, SeqLens, PartAcc, PartM, PartL,
    scale, softcap,
    stride_qt, stride_qh,
    stride_kb, stride_ks, stride_kh,
    stride_vb, stride_vs, stride_vh,
    num_splits,
    stride_pa_bh, stride_pa_s, stride_pa_r,
    stride_pm_bh, stride_pm_s,
    KVH: tl.constexpr, G: tl.constexpr, Q_LEN: tl.constexpr, HEAD_DIM: tl.constexpr,
    WINDOW: tl.constexpr, USE_SOFTCAP: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_DV: tl.constexpr,
    NUM_DV_BLOCKS: tl.constexpr, Q_IN_REGS: tl.constexpr,
):
    pid0 = tl.program_id(0)          # fastest: (row block, dv block) of the same KV range -> co-scheduled, L2 reuse
    split = tl.program_id(1)
    bh = tl.program_id(2)
    mb = pid0 // NUM_DV_BLOCKS
    dvb = pid0 % NUM_DV_BLOCKS
    b = bh // KVH
    kvh = bh % KVH
    ROWS: tl.constexpr = Q_LEN * G

    seq_len = tl.load(SeqLens + b)
    ctx = seq_len - Q_LEN
    offs_m = mb * BLOCK_M + tl.arange(0, BLOCK_M)
    row_ok = offs_m < ROWS
    t = tl.where(row_ok, offs_m // G, 0)
    g = offs_m % G
    qpos = ctx + t
    q_rows = Q + (b * Q_LEN + t).to(tl.int64)[:, None] * stride_qt + (kvh * G + g)[:, None] * stride_qh

    # KV range this row block needs: [lo, hi)
    t_lo = (mb * BLOCK_M) // G
    t_hi = tl.minimum((mb * BLOCK_M + BLOCK_M - 1) // G, Q_LEN - 1)
    hi = ctx + t_hi + 1
    if WINDOW > 0:
        lo = tl.maximum(ctx + t_lo - WINDOW + 1, 0)
    else:
        lo = ctx * 0
    span = hi - lo
    per_split = (span + num_splits - 1) // num_splits
    per_split = ((per_split + BLOCK_N - 1) // BLOCK_N) * BLOCK_N
    start = lo + split * per_split
    end = tl.minimum(start + per_split, hi)

    k_base = K + b.to(tl.int64) * stride_kb + kvh * stride_kh
    v_base = V + b.to(tl.int64) * stride_vb + kvh * stride_vh
    offs_dv = dvb * BLOCK_DV + tl.arange(0, BLOCK_DV)

    if Q_IN_REGS:
        offs_dq = tl.arange(0, HEAD_DIM)
        q_all = tl.load(q_rows + offs_dq[None, :], mask=row_ok[:, None], other=0.0)

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DV], dtype=tl.float32)

    for n0 in range(start, end, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        n_ok = offs_n < end
        offs_n64 = offs_n.to(tl.int64)
        if Q_IN_REGS:
            kt = tl.load(k_base + offs_n64[None, :] * stride_ks + offs_dq[:, None], mask=n_ok[None, :], other=0.0)
            s = tl.dot(q_all, kt)
        else:
            s = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            for d0 in range(0, HEAD_DIM, BLOCK_D):  # rolled: unrolling would multi-buffer 8 chunks in smem
                offs_d = d0 + tl.arange(0, BLOCK_D)
                qc = tl.load(q_rows + offs_d[None, :], mask=row_ok[:, None], other=0.0)
                kc = tl.load(k_base + offs_n64[None, :] * stride_ks + offs_d[:, None], mask=n_ok[None, :], other=0.0)
                s += tl.dot(qc, kc)
        s = s * scale
        if USE_SOFTCAP:
            s = softcap * _tanh(s / softcap)
        valid = row_ok[:, None] & n_ok[None, :] & (offs_n[None, :] <= qpos[:, None])
        if WINDOW > 0:
            valid = valid & ((qpos[:, None] - offs_n[None, :]) < WINDOW)
        s = tl.where(valid, s, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp2((m_i - m_safe) * 1.4426950408889634)
        p = tl.exp2((s - m_safe[:, None]) * 1.4426950408889634)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        vt = tl.load(v_base + offs_n64[:, None] * stride_vs + offs_dv[None, :], mask=n_ok[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(vt.dtype), vt)
        m_i = m_new

    # Unnormalised partials; empty splits store m=-inf, l=0, acc=0 (handled by the merge).
    pa = PartAcc + bh.to(tl.int64) * stride_pa_bh + split * stride_pa_s
    tl.store(pa + offs_m[:, None] * stride_pa_r + offs_dv[None, :], acc, mask=row_ok[:, None])
    pm_off = bh.to(tl.int64) * stride_pm_bh + split * stride_pm_s + offs_m
    ml_mask = row_ok & (dvb == 0)
    tl.store(PartM + pm_off, m_i, mask=ml_mask)
    tl.store(PartL + pm_off, l_i, mask=ml_mask)


@triton.jit(do_not_specialize=["num_splits"])
def _mqa_verify_merge(
    PartAcc, PartM, PartL, Out,
    num_splits,
    stride_pa_bh, stride_pa_s, stride_pa_r,
    stride_pm_bh, stride_pm_s,
    stride_ot, stride_oh,
    KVH: tl.constexpr, G: tl.constexpr, Q_LEN: tl.constexpr,
    BLOCK_R: tl.constexpr, BLOCK_DR: tl.constexpr,
):
    bh = tl.program_id(0)
    rb = tl.program_id(1)
    db = tl.program_id(2)
    b = bh // KVH
    kvh = bh % KVH
    offs_r = rb * BLOCK_R + tl.arange(0, BLOCK_R)
    r_ok = offs_r < Q_LEN * G
    offs_d = db * BLOCK_DR + tl.arange(0, BLOCK_DR)
    pm_base = PartM + bh.to(tl.int64) * stride_pm_bh + offs_r
    pl_base = PartL + bh.to(tl.int64) * stride_pm_bh + offs_r
    pa_base = PartAcc + bh.to(tl.int64) * stride_pa_bh + offs_r[:, None] * stride_pa_r + offs_d[None, :]

    m_max = tl.full([BLOCK_R], float("-inf"), dtype=tl.float32)
    for s in range(0, num_splits):
        m_s = tl.load(pm_base + s * stride_pm_s, mask=r_ok, other=float("-inf"))
        m_max = tl.maximum(m_max, m_s)
    m_safe = tl.where(m_max == float("-inf"), 0.0, m_max)
    l_tot = tl.zeros([BLOCK_R], dtype=tl.float32)
    acc = tl.zeros([BLOCK_R, BLOCK_DR], dtype=tl.float32)
    for s in range(0, num_splits):
        m_s = tl.load(pm_base + s * stride_pm_s, mask=r_ok, other=float("-inf"))
        l_s = tl.load(pl_base + s * stride_pm_s, mask=r_ok, other=0.0)
        w = tl.exp2((m_s - m_safe) * 1.4426950408889634)
        l_tot += l_s * w
        a = tl.load(pa_base + s * stride_pa_s, mask=r_ok[:, None], other=0.0)
        acc += a * w[:, None]
    out = acc / tl.where(l_tot > 0.0, l_tot, 1.0)[:, None]
    t = tl.where(r_ok, offs_r // G, 0)
    g = offs_r % G
    o_ptr = Out + (b * Q_LEN + t).to(tl.int64)[:, None] * stride_ot + (kvh * G + g)[:, None] * stride_oh \
        + offs_d[None, :]
    tl.store(o_ptr, out.to(Out.dtype.element_ty), mask=r_ok[:, None])


def _pow2_floor(x: int) -> int:
    return 1 << (max(int(x), 1).bit_length() - 1)


def rows_padded(q_len: int, group: int, v: Variant) -> int:
    return triton.cdiv(q_len * group, v.block_m) * v.block_m


def choose_num_splits(num_seqs: int, num_kv_heads: int, span: int, q_len: int, group: int, v: Variant,
                      target_programs: int) -> int:
    """Flash-decoding split count: num_seqs * kv_heads * splits >= target (~2 x SMs), as a power of two, but keep
    >= max(BLOCK_N, 2 x padded rows) KV tokens per split so the fp32 partials stay well below the KV bytes."""
    want = 1 << max(0, math.ceil(math.log2(max(1, math.ceil(target_programs / (num_seqs * num_kv_heads))))))
    min_tok = max(v.block_n, 2 * rows_padded(q_len, group, v))
    cap = _pow2_floor(max(1, span // min_tok))
    return max(1, min(want, cap))


def alloc_workspace(num_seqs: int, num_kv_heads: int, num_splits: int, q_len: int, group: int, head_dim: int,
                    v: Variant, device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r = rows_padded(q_len, group, v)
    bh = num_seqs * num_kv_heads
    part_acc = torch.empty((bh, num_splits, r, head_dim), dtype=torch.float32, device=device)
    part_m = torch.empty((bh, num_splits, r), dtype=torch.float32, device=device)
    part_l = torch.empty((bh, num_splits, r), dtype=torch.float32, device=device)
    return part_acc, part_m, part_l


def mqa_verify_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, seq_lens: torch.Tensor, q_len: int,
                         scale: float, out: torch.Tensor, workspace, num_splits: int, variant: Variant,
                         window: int = 0, softcap: float = 0.0) -> torch.Tensor:
    """q [B*q_len, H, D]; k, v [B, S, KVH, D] (last dim contiguous); seq_lens [B] int32; writes out [B*q_len, H, D]."""
    num_tokens, num_heads, head_dim = q.shape
    num_seqs, _, num_kv_heads, _ = k.shape
    assert num_tokens == num_seqs * q_len and num_heads % num_kv_heads == 0
    assert q.stride(2) == 1 and k.stride(3) == 1 and v.stride(3) == 1 and out.stride(2) == 1
    assert head_dim % variant.block_dv == 0 and head_dim % variant.block_d == 0
    assert (not variant.q_in_regs) or variant.block_d == head_dim
    group = num_heads // num_kv_heads
    part_acc, part_m, part_l = workspace
    assert part_acc.shape[1] >= num_splits and part_acc.shape[2] >= rows_padded(q_len, group, variant)
    n_dv = head_dim // variant.block_dv
    n_mb = triton.cdiv(q_len * group, variant.block_m)
    grid = (n_mb * n_dv, num_splits, num_seqs * num_kv_heads)
    _mqa_verify_partial[grid](
        q, k, v, seq_lens, part_acc, part_m, part_l,
        float(scale), float(softcap) if softcap > 0 else 1.0,
        q.stride(0), q.stride(1),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        num_splits,
        part_acc.stride(0), part_acc.stride(1), part_acc.stride(2),
        part_m.stride(0), part_m.stride(1),
        KVH=num_kv_heads, G=group, Q_LEN=q_len, HEAD_DIM=head_dim,
        WINDOW=int(window), USE_SOFTCAP=bool(softcap > 0),
        BLOCK_M=variant.block_m, BLOCK_N=variant.block_n, BLOCK_D=variant.block_d, BLOCK_DV=variant.block_dv,
        NUM_DV_BLOCKS=n_dv, Q_IN_REGS=variant.q_in_regs,
        num_warps=variant.num_warps, num_stages=variant.num_stages,
    )
    block_r, block_dr = 16, min(128, head_dim)
    grid_r = (num_seqs * num_kv_heads, triton.cdiv(q_len * group, block_r), head_dim // block_dr)
    _mqa_verify_merge[grid_r](
        part_acc, part_m, part_l, out,
        num_splits,
        part_acc.stride(0), part_acc.stride(1), part_acc.stride(2),
        part_m.stride(0), part_m.stride(1),
        out.stride(0), out.stride(1),
        KVH=num_kv_heads, G=group, Q_LEN=q_len, BLOCK_R=block_r, BLOCK_DR=block_dr,
        num_warps=4,
    )
    return out


def reference_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, seq_lens: list[int], q_len: int,
                        scale: float, window: int = 0, softcap: float = 0.0) -> torch.Tensor:
    """fp32 torch reference with the same semantics; k, v [B, S, KVH, D]. Returns [B*q_len, H, D] fp32."""
    num_heads = q.shape[1]
    outs = []
    for b, sl in enumerate(seq_lens):
        kb = k[b, :sl].float()
        vb = v[b, :sl].float()
        group = num_heads // kb.shape[1]
        kb = kb.repeat_interleave(group, dim=1)  # q head h -> kv head h // group (vLLM's mapping)
        vb = vb.repeat_interleave(group, dim=1)
        qb = q[b * q_len:(b + 1) * q_len].float()
        s = torch.einsum("thd,shd->hts", qb, kb) * scale
        if softcap > 0:
            s = softcap * torch.tanh(s / softcap)
        qpos = (sl - q_len) + torch.arange(q_len, device=q.device)
        kpos = torch.arange(sl, device=q.device)
        mask = kpos[None, :] <= qpos[:, None]
        if window > 0:
            mask &= (qpos[:, None] - kpos[None, :]) < window
        s = s.masked_fill(~mask[None], float("-inf"))
        outs.append(torch.einsum("hts,shd->thd", torch.softmax(s, dim=-1), vb))
    return torch.cat(outs, dim=0)
