"""Kernel lab: quantized vocab scan + exact BF16 rescore for the MTP draft's dense full-vocabulary head.

The Gemma 4 12B MTP draft (google/gemma-4-12B-it-assistant) computes logits = h @ W^T over all 262144 tokens every
draft step (W = tied embedding [262144, 1024] BF16, 512 MiB; bandwidth bound) and takes a masked argmax (eoa/eoi
suppressed). Here W is scanned in FP8 (per-row scale) or INT4 (group 128 along K), the top-k approximate candidates
are rescored exactly from the BF16 rows, and the greedy token is compared with the BF16 argmax.

Run on Modal inside the pinned vLLM image:
    g4bench modal script --path kernels/draft_head_scan.py --execute [-- --eval-rows 4096 --topt 2]
Writes $G4_OUT_DIR/draft_head_scan.json and prints one JSON line per result. Variants (each fail-soft):
  baseline     F.linear(h, W) (cuBLAS, BF16 out) + index_fill(-inf) + argmax         (what vLLM does)
  fp8_logits   Triton FP8 scan -> fp32 logits [M, V] in HBM -> torch.topk(k) -> Triton BF16 rescore
  fp8_fused    Triton FP8 scan with per-block top-T in the epilogue -> topk over [M, V/BLOCK_N*T] -> rescore
  int4_logits  / int4_fused: same with INT4 (symmetric, group 128, fp16 scales, 2 nibbles/byte)
Recall is measured on two activation sets (random normal h; "realistic" h = unit-RMS embedding row + noise). Rescoring
reproduces BF16 semantics (fp32 dot, round to BF16, lowest token id wins ties) so the reference is the BF16 argmax.
Timing: per call, scan + topk + rescore end to end, cold weights (ring of copies > 3x L2), eager and CUDA graph.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import struct
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    TRITON_ERR = None
except Exception as _e:  # noqa: BLE001
    triton = tl = None
    TRITON_ERR = repr(_e)[:300]

REPO = "google/gemma-4-12B-it-assistant"
REVISION = "3b0b0a4a3b31b5556516329227fb55f275c23c9f"   # configs/pins.yaml
V, D = 262144, 1024
SUP = (258883, 258882)          # eoa_token_index, eoi_token_id: set to -inf before the argmax
GROUP = 128                     # INT4 group size along K
DRAFT_STEPS = 6                 # draft steps per decode step
COLD_BYTES = 384 * 1024**2      # ring of weight copies totalling >= 3x the 128 MB L2
KMAX = 64
BIG = 1 << 30
# Scan configs tried per variant (fail-soft; the fastest at M=28 that compiles for every BLOCK_M bucket is kept).
# sm_120 has ~100 KB shared memory per block, so tiles x stages stay small.
SCAN_CONFIGS = [   # BK applies to FP8 only (INT4 always steps one 128-wide group)
    {"BN": 128, "BK": 128, "warps": 4, "stages": 2},
    {"BN": 256, "BK": 64, "warps": 8, "stages": 3},
    {"BN": 64, "BK": 128, "warps": 4, "stages": 3},
]

if triton is not None:
    @triton.jit
    def _epilogue(acc, rows, offs_m, M, N, OUT_V, OUT_I, pid, NCAND, SUP0: tl.constexpr, SUP1: tl.constexpr,
                  FUSED: tl.constexpr, TOPT: tl.constexpr, BLOCK_N: tl.constexpr):
        bad = (rows == SUP0) | (rows == SUP1) | (rows >= N)
        acc = tl.where(bad[None, :], float("-inf"), acc)
        m_mask = offs_m < M
        if FUSED:
            # Per-block running top-T (T small): repeated max + mask; writes T (value, token) pairs per row per block.
            cols = tl.arange(0, BLOCK_N)
            for t in tl.static_range(TOPT):
                v, i = tl.max(acc, axis=1, return_indices=True)
                base = offs_m * NCAND + pid * TOPT + t
                tl.store(OUT_V + base, v, mask=m_mask)
                tl.store(OUT_I + base, pid * BLOCK_N + i, mask=m_mask)
                acc = tl.where(cols[None, :] == i[:, None], float("-inf"), acc)
        else:
            tl.store(OUT_V + offs_m[:, None] * N + rows[None, :], acc, mask=m_mask[:, None] & (rows < N)[None, :])

    # do_not_specialize: one compile per BLOCK_M bucket instead of per M value (M==1 / M%16 specializations).
    @triton.jit(do_not_specialize=["M", "NCAND"])
    def _fp8_scan_kernel(H, Q, S, OUT_V, OUT_I, M, N, NCAND, K: tl.constexpr, SUP0: tl.constexpr,
                         SUP1: tl.constexpr, FUSED: tl.constexpr, TOPT: tl.constexpr, DECODE: tl.constexpr,
                         BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        # One program per BLOCK_N vocab rows; all M requests in one BLOCK_M tile so W is read exactly once.
        pid = tl.program_id(0)
        offs_m = tl.arange(0, BLOCK_M)
        rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        m_mask = offs_m < M
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            ks = k0 + tl.arange(0, BLOCK_K)
            h = tl.load(H + offs_m[:, None] * K + ks[None, :], mask=m_mask[:, None], other=0.0).to(tl.float16)
            q = tl.load(Q + rows[None, :] * K + ks[:, None])          # [BLOCK_K, BLOCK_N] (N % BLOCK_N == 0)
            if DECODE == 0:
                w = q.to(tl.float16)                                    # native e4m3 -> f16 convert
            else:
                # Bit decode (no fp8 type support needed): e4m3fn bits -> f16 bits; value / 256, folded into S.
                b = q.to(tl.int32)
                w = (((b & 0x80) << 8) | ((b & 0x7F) << 7)).to(tl.int16).to(tl.float16, bitcast=True)
            acc += tl.dot(h, w)
        s = tl.load(S + rows)
        acc = acc * s[None, :]
        _epilogue(acc, rows, offs_m, M, N, OUT_V, OUT_I, pid, NCAND, SUP0, SUP1, FUSED, TOPT, BLOCK_N)

    @triton.jit(do_not_specialize=["M", "NCAND"])
    def _int4_scan_kernel(H, P, S, OUT_V, OUT_I, M, N, NCAND, K: tl.constexpr, SUP0: tl.constexpr,
                          SUP1: tl.constexpr, FUSED: tl.constexpr, TOPT: tl.constexpr, GROUP: tl.constexpr,
                          HALF: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
        # Packing: byte g*HALF+j of row n holds k = g*GROUP+j (low nibble) and g*GROUP+HALF+j (high nibble), stored
        # offset-binary (q+8). Two dots per group avoid any in-register interleave; scale applied per group.
        pid = tl.program_id(0)
        offs_m = tl.arange(0, BLOCK_M)
        rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        m_mask = offs_m < M
        kk = tl.arange(0, HALF)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for g in range(0, K // GROUP):
            k0 = g * GROUP
            h_lo = tl.load(H + offs_m[:, None] * K + (k0 + kk)[None, :], mask=m_mask[:, None], other=0.0)
            h_hi = tl.load(H + offs_m[:, None] * K + (k0 + HALF + kk)[None, :], mask=m_mask[:, None], other=0.0)
            p = tl.load(P + rows[None, :] * (K // 2) + (g * HALF + kk)[:, None]).to(tl.int32)  # [HALF, BLOCK_N]
            lo = ((p & 15) - 8).to(tl.float16)
            hi = ((p >> 4) - 8).to(tl.float16)
            part = tl.dot(h_lo.to(tl.float16), lo)
            part = tl.dot(h_hi.to(tl.float16), hi, acc=part)
            sc = tl.load(S + g * N + rows).to(tl.float32)
            acc += part * sc[None, :]
        _epilogue(acc, rows, offs_m, M, N, OUT_V, OUT_I, pid, NCAND, SUP0, SUP1, FUSED, TOPT, BLOCK_N)

    @triton.jit(do_not_specialize=["KC", "stride_pos", "stride_map"])
    def _rescore_kernel(H, W, POS, MAP, TOK, TOK32, KC, stride_pos, stride_map, K: tl.constexpr,
                        KC_PAD: tl.constexpr, BLOCK_D: tl.constexpr, INDIRECT: tl.constexpr):
        # One program per request: gather k BF16 rows, fp32 dot, round to BF16, argmax with lowest-token tie-break.
        m = tl.program_id(0)
        c = tl.arange(0, KC_PAD)
        cmask = c < KC
        p = tl.load(POS + m * stride_pos + c, mask=cmask, other=0)
        if INDIRECT:
            idx = tl.load(MAP + m * stride_map + p, mask=cmask, other=0).to(tl.int64)
        else:
            idx = p.to(tl.int64)
        acc = tl.zeros((KC_PAD,), dtype=tl.float32)
        for d0 in range(0, K, BLOCK_D):
            ds = d0 + tl.arange(0, BLOCK_D)
            h = tl.load(H + m * K + ds).to(tl.float32)
            w = tl.load(W + idx[:, None] * K + ds[None, :], mask=cmask[:, None], other=0.0).to(tl.float32)
            acc += tl.sum(w * h[None, :], axis=1)
        acc = tl.where(cmask, acc, float("-inf"))
        r = acc.to(tl.bfloat16).to(tl.float32)
        mx = tl.max(r, axis=0)
        tok = tl.min(tl.where(r == mx, idx, 1073741824), axis=0)
        mx32 = tl.max(acc, axis=0)
        tok32 = tl.min(tl.where(acc == mx32, idx, 1073741824), axis=0)
        tl.store(TOK + m, tok)
        tl.store(TOK32 + m, tok32)


# ----------------------------------------------------------------------------------------------------------- helpers
def bench(fn, iters: int, warmup: int = 5) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms


def bench_ring(fns, iters: int) -> float:
    ring = itertools.count()
    return bench(lambda: fns[next(ring) % len(fns)](), iters)


def graph_ring(fns, iters: int) -> float:
    """One CUDA graph per weight copy (same op sequence), replayed round-robin."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for f in fns:
            f()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    graphs = []
    for f in fns:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            f()
        graphs.append(g)
    ring = itertools.count()
    t = bench(lambda: graphs[next(ring) % len(graphs)].replay(), iters)
    del graphs
    return t


def block_m(m: int) -> int:
    return 16 if m <= 16 else 32 if m <= 32 else 64


def emit(rows: list, row: dict) -> None:
    rows.append(row)
    print(json.dumps(row), flush=True)


# ------------------------------------------------------------------------------------------------------------ weights
def _st_header(path: str) -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def _fetch(pattern: str) -> tuple[Path, str]:
    from huggingface_hub import snapshot_download
    try:
        root = Path(snapshot_download(REPO, revision=REVISION, allow_patterns=[pattern], local_files_only=True))
        if any(root.glob(pattern)):
            return root, "cache"
    except Exception:  # noqa: BLE001
        pass
    return Path(snapshot_download(REPO, revision=REVISION, allow_patterns=[pattern])), "download"


def load_head(dev) -> tuple[torch.Tensor, dict]:
    """The draft's output head: lm_head.weight if present, else the tied embedding ([V, D] either way)."""
    from safetensors import safe_open
    meta: dict = {"repo": REPO, "revision": REVISION}
    root, meta["json_source"] = _fetch("*.json")
    try:
        cfg = json.loads((root / "config.json").read_text())
        meta["tie_word_embeddings"] = cfg.get("tie_word_embeddings", cfg.get("text_config", {}).get(
            "tie_word_embeddings"))
        meta["use_ordered_embeddings"] = cfg.get("use_ordered_embeddings")
    except Exception as e:  # noqa: BLE001
        meta["config_error"] = repr(e)[:200]
    idx = root / "model.safetensors.index.json"
    if idx.exists():
        wm = json.loads(idx.read_text())["weight_map"]
        shards = sorted(set(wm.values()))
    else:
        wm, shards = None, None

    def rank(name: str) -> int:
        return 0 if name.endswith("lm_head.weight") else 1 if "embed_tokens.weight" in name else 2

    if wm is not None:
        names = sorted(wm, key=rank)
        order = [wm[n] for n in names if rank(n) < 2] + shards
    else:
        order = ["*.safetensors"]
    seen = set()
    for pat in order:
        if pat in seen:
            continue
        seen.add(pat)
        sroot, src = _fetch(pat)
        for path in sorted(sroot.glob(pat)):
            hdr = _st_header(str(path))
            two_d = {n: v["shape"] for n, v in hdr.items() if n != "__metadata__" and len(v["shape"]) == 2
                     and v["shape"][0] == V}
            meta.setdefault("vocab_tensors", {}).update(two_d)
            hits = sorted([n for n, s in two_d.items() if s == [V, D]], key=rank)
            if hits:
                name = hits[0]
                with safe_open(str(path), framework="pt", device="cpu") as f:
                    t = f.get_tensor(name)
                meta.update({"head_tensor": name, "head_dtype": str(t.dtype), "shard": path.name,
                             "shard_source": src, "tied": not name.endswith("lm_head.weight")})
                return t.to(dev, torch.bfloat16).contiguous(), meta
    raise RuntimeError(f"no [{V}, {D}] tensor found in {REPO}@{REVISION}")


def quant_fp8(W: torch.Tensor):
    Wf = W.float()
    s = (Wf.abs().amax(1) / 448.0).clamp_min(1e-12)
    q = (Wf / s[:, None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    deq = q.float() * s[:, None]
    del Wf
    return {"q": q, "s": s.contiguous()}, deq


def quant_int4(W: torch.Tensor):
    G = D // GROUP
    Wf = W.float().view(V, G, GROUP)
    s = (Wf.abs().amax(-1) / 7.0).clamp_min(2.0**-14).half()                  # [V, G]
    q = torch.round(Wf / s.float()[..., None]).clamp(-8, 7).to(torch.int8)   # [V, G, GROUP]
    deq = (q.float() * s.float()[..., None]).view(V, D)
    u = (q + 8).to(torch.uint8).view(V, G, 2, GROUP // 2)
    packed = (u[:, :, 0, :] | (u[:, :, 1, :] << 4)).contiguous().view(V, D // 2)
    del Wf, q, u
    return {"q": packed, "s": s.t().contiguous()}, deq                         # scales stored [G, V]


def finish_copy(fmt: str, w: dict) -> dict:
    if fmt == "fp8":
        w["q8"] = w["q"].view(torch.uint8)
        w["s256"] = (w["s"] * 256.0).contiguous()
    return w


def ring_copies(fmt: str, w: dict) -> list[dict]:
    nbytes = sum(v.numel() * v.element_size() for k, v in w.items() if k in ("q", "s"))
    n = max(1, -(-COLD_BYTES // nbytes))
    out = [finish_copy(fmt, w)]
    for _ in range(n - 1):
        out.append(finish_copy(fmt, {"q": w["q"].clone(), "s": w["s"].clone()}))
    return out


def weight_bytes(fmt: str, w: dict) -> int:
    return sum(w[k].numel() * w[k].element_size() for k in ("q", "s"))


# --------------------------------------------------------------------------------------------------------- pipelines
_DUMMY: dict = {}


def run_scan(fmt: str, fused: bool, cfg: dict, w: dict, h: torch.Tensor, topt: int, decode: int):
    M = h.shape[0]
    BN = cfg["BN"]
    nblk = V // BN
    ncand = nblk * topt
    if fused:
        ov = torch.empty((M, ncand), device=h.device, dtype=torch.float32)
        oi = torch.empty((M, ncand), device=h.device, dtype=torch.int32)
    else:
        ov = torch.empty((M, V), device=h.device, dtype=torch.float32)
        if "i32" not in _DUMMY:
            _DUMMY["i32"] = torch.empty(1, device=h.device, dtype=torch.int32)
        oi = _DUMMY["i32"]                                                # unused unless fused
    common = dict(K=D, SUP0=SUP[0], SUP1=SUP[1], FUSED=fused, TOPT=topt, BLOCK_M=block_m(M), BLOCK_N=BN,
                  num_warps=cfg["warps"], num_stages=cfg["stages"])
    if fmt == "fp8":
        q, s = (w["q"], w["s"]) if decode == 0 else (w["q8"], w["s256"])
        _fp8_scan_kernel[(nblk,)](h, q, s, ov, oi, M, V, ncand, DECODE=decode, BLOCK_K=cfg["BK"], **common)
    else:
        _int4_scan_kernel[(nblk,)](h, w["q"], w["s"], ov, oi, M, V, ncand, GROUP=GROUP, HALF=GROUP // 2, **common)
    return ov, oi


def rescore(h: torch.Tensor, W: torch.Tensor, pos: torch.Tensor, cmap, k: int):
    M = h.shape[0]
    tok = torch.empty(M, device=h.device, dtype=torch.int64)
    tok32 = torch.empty(M, device=h.device, dtype=torch.int64)
    mp = cmap if cmap is not None else pos
    _rescore_kernel[(M,)](h, W, pos, mp, tok, tok32, k, pos.stride(0), mp.stride(0), K=D,
                          KC_PAD=max(16, triton.next_power_of_2(k)), BLOCK_D=128, INDIRECT=cmap is not None,
                          num_warps=4)
    return tok, tok32


def pipeline(fmt, fused, cfg, w, W, h, k, topt, decode):
    ov, oi = run_scan(fmt, fused, cfg, w, h, topt, decode)
    pos = torch.topk(ov, k, dim=1, sorted=False).indices
    return rescore(h, W, pos, oi if fused else None, k)


def baseline(h, W, sup):
    logits = F.linear(h, W)
    logits.index_fill_(1, sup, float("-inf"))
    return logits.argmax(-1)


# --------------------------------------------------------------------------------------------------------- recall
def exact_scores(H: torch.Tensor, W: torch.Tensor, cands: torch.Tensor) -> torch.Tensor:
    out = []
    for i in range(0, H.shape[0], 512):
        wc = W[cands[i:i + 512]].float()                                    # [r, KMAX, D]
        out.append(torch.bmm(wc, H[i:i + 512].float().unsqueeze(-1)).squeeze(-1))
    return torch.cat(out)


def pick(ex: torch.Tensor, cands: torch.Tensor, k: int, rounded: bool) -> torch.Tensor:
    e, c = ex[:, :k], cands[:, :k]
    if rounded:
        e = e.to(torch.bfloat16).float()
    mx = e.max(1, keepdim=True).values
    return torch.where(e == mx, c, torch.full_like(c, BIG)).min(1).values


def references(H: torch.Tensor, W: torch.Tensor, W32: torch.Tensor, sup) -> dict:
    rb, rf, top, gap = [], [], [], []
    for i in range(0, H.shape[0], 256):
        hc = H[i:i + 256]
        lb = F.linear(hc, W)
        lb.index_fill_(1, sup, float("-inf"))
        rb.append(lb.argmax(1))
        lf = F.linear(hc.float(), W32)
        lf.index_fill_(1, sup, float("-inf"))
        t = lf.topk(KMAX, dim=1)
        rf.append(t.indices[:, 0])
        top.append(t.indices)
        sd = lf[:, :V // 2].std(1)                                            # logit spread (finite half)
        gap.append((t.values[:, 0] - t.values[:, 1]) / sd)
    ref = {"bf16": torch.cat(rb), "fp32": torch.cat(rf), "top": torch.cat(top)}
    g = torch.cat(gap)
    ex = exact_scores(H, W, ref["top"])
    ref["stats"] = {
        "rows": H.shape[0],
        "bf16_vs_fp32_argmax_disagree": round(float((ref["bf16"] != ref["fp32"]).float().mean()), 5),
        "top1_top2_gap_in_logit_sd_p10_p50_p90": [round(float(x), 3) for x in torch.quantile(
            g.float(), torch.tensor([0.1, 0.5, 0.9], device=g.device))],
        # Rescoring the exact fp32 top-k (a perfect scan): residual mismatch vs cuBLAS BF16 = rounding/tie floor.
        "perfect_scan_recall_bf16": {k: round(float((pick(ex, ref["top"], k, True) == ref["bf16"]).float().mean()),
                                              5) for k in (8, 64)},
    }
    return ref


def eval_recall(fmt, fused, cfg, w, W, H, ref, topt, decode, ks) -> tuple[dict, dict]:
    cands = []
    for i in range(0, H.shape[0], 64):
        ov, oi = run_scan(fmt, fused, cfg, w, H[i:i + 64], topt, decode)
        pos = torch.topk(ov, KMAX, dim=1, sorted=True).indices
        cands.append(torch.gather(oi, 1, pos).long() if fused else pos)
    cands = torch.cat(cands)
    ex = exact_scores(H, W, cands)
    res, toks = {}, {}
    for k in ks:
        tb = pick(ex, cands, k, True)
        tf = pick(ex, cands, k, False)
        res[k] = {"recall_bf16": float((tb == ref["bf16"]).float().mean()),
                  "recall_fp32": float((tf == ref["fp32"]).float().mean()),
                  "bf16_argmax_in_topk": float((cands[:, :k] == ref["bf16"][:, None]).any(1).float().mean()),
                  "misses": int((tb != ref["bf16"]).sum())}
        toks[k] = tb[:64].clone()
    return res, toks


def check_scan(fmt, fused, cfg, w, h, ref_logits, topt, decode) -> tuple[float, bool]:
    """Kernel vs torch dequantized reference (same quantized weights): catches packing / decode / layout bugs."""
    ov, oi = run_scan(fmt, fused, cfg, w, h, topt, decode)
    fin = torch.isfinite(ref_logits)
    tol = 5e-3 * float(ref_logits[fin].abs().max()) + 1e-3
    if not fused:
        err = float((ov[fin] - ref_logits[fin]).abs().max())
        ok = err <= tol and bool(torch.isinf(ov[:, list(SUP)]).all())
    else:
        M, BN = h.shape[0], cfg["BN"]
        blk = ref_logits.view(M, V // BN, BN).max(-1).values
        got = ov.view(M, V // BN, topt)[:, :, 0]
        err = float((got - blk).abs().max())
        err = max(err, float((ref_logits.gather(1, oi.long()) - ov).abs()[torch.isfinite(ov)].max()))
        ok = err <= tol
    return err, ok


# ------------------------------------------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-rows", type=int, default=4096, help="activation rows per set for recall")
    ap.add_argument("--ms", default="1,4,7,16,28,64")
    ap.add_argument("--ks", default="8,16,32,64")
    ap.add_argument("--topt", type=int, default=2, help="per-block candidates kept by the fused epilogue")
    ap.add_argument("--signal", default="0.15,0.6", help="realistic h: per-row embedding-signal fraction range")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--variants", default="fp8_logits,fp8_fused,int4_logits,int4_fused")
    ap.add_argument("--no-graph", action="store_true")
    ap.add_argument("--synthetic", action="store_true", help="skip the real head; random normal W")
    ap.add_argument("--budget-s", type=float, default=330.0, help="skip remaining timings past this wall time")
    args = ap.parse_args()
    t_start = time.time()
    ms = [int(x) for x in args.ms.split(",")]
    ks = [int(x) for x in args.ks.split(",")]
    assert max(ks) <= KMAX and max(ms) <= 64
    out_dir = os.environ.get("G4_OUT_DIR", ".")
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False      # fp32 reference must be true fp32
    props = torch.cuda.get_device_properties(dev)
    info = {"gpu": props.name, "sm": f"{props.major}.{props.minor}", "sms": props.multi_processor_count,
            "torch": torch.__version__, "cuda": torch.version.cuda, "triton": getattr(triton, "__version__", None),
            "triton_error": TRITON_ERR, "args": vars(args),
            "bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction}
    print(json.dumps(info), flush=True)
    rows: list = []
    torch.manual_seed(0)

    buf = torch.empty(512 * 1024**2, device=dev, dtype=torch.uint8)
    dst = torch.empty_like(buf)
    info["copy_bandwidth_gbs"] = round(2 * buf.numel() / bench(lambda: dst.copy_(buf), 20) / 1e6, 1)
    del buf, dst

    W, synthetic = None, args.synthetic
    if not synthetic:
        try:
            t0 = time.time()
            W, meta = load_head(dev)
            meta["load_s"] = round(time.time() - t0, 1)
            info["weights"] = meta
        except Exception as e:  # noqa: BLE001
            info["weights_error"] = repr(e)[:500]
            synthetic = True
    if synthetic:
        W = (torch.randn(V, D, device=dev) * 0.03).to(torch.bfloat16)
    info["synthetic"] = synthetic
    info["head_row_rms"] = round(float(W.float().pow(2).mean().sqrt()), 5)
    print(json.dumps({k: info.get(k) for k in ("copy_bandwidth_gbs", "weights", "weights_error", "synthetic",
                                               "head_row_rms")}), flush=True)
    sup = torch.tensor(SUP, device=dev, dtype=torch.long)

    # Activation sets (recall is invariant to the scale of h; RMS ~1 like a post-norm hidden state).
    R = args.eval_rows
    g = torch.Generator(device=dev).manual_seed(1)
    H_rand = torch.randn(R, D, device=dev, generator=g).to(torch.bfloat16)
    lo_s, hi_s = (float(x) for x in args.signal.split(","))
    tok = torch.randint(0, V - 2048, (R,), device=dev, generator=g)          # avoids the suppressed ids
    e = W[tok].float()
    e = e / e.pow(2).mean(1, keepdim=True).sqrt().clamp_min(1e-12)
    a = lo_s + (hi_s - lo_s) * torch.rand(R, 1, device=dev, generator=g)
    H_real = (a * e + (1 - a * a).sqrt() * torch.randn(R, D, device=dev, generator=g)).to(torch.bfloat16)
    sets = {"random": H_rand.contiguous(), "realistic": H_real.contiguous()}
    del e

    W32 = W.float()
    refs = {}
    for name, H in sets.items():
        refs[name] = references(H, W, W32, sup)
        emit(rows, {"kind": "reference", "set": name, "synthetic": synthetic, **refs[name]["stats"]})
    del W32
    torch.cuda.empty_cache()

    # Baseline timing (single 512 MiB copy is already larger than L2).
    base_us: dict = {}
    for m in ms:
        h = H_real[:m].contiguous()
        row = {"kind": "timing", "variant": "baseline", "M": m, "synthetic": synthetic}
        try:
            got = baseline(h, W, sup)
            row["matches_ref"] = int((got == refs["realistic"]["bf16"][:m]).sum())
            row["eager_us"] = round(1e3 * bench(lambda: baseline(h, W, sup), args.iters), 2)
            if not args.no_graph:
                row["graph_us"] = round(1e3 * graph_ring([lambda: baseline(h, W, sup)], args.iters), 2)
            t = row.get("graph_us", row["eager_us"])
            row["gbs"] = round(2 * V * D / t / 1e3, 1)
            base_us[m] = t
        except Exception as ex:  # noqa: BLE001
            row["error"] = repr(ex)[:300]
        emit(rows, row)

    recall: dict = {}
    timing: dict = {}
    variants = [v for v in args.variants.split(",") if v]
    if triton is None:
        emit(rows, {"kind": "error", "stage": "import triton", "error": TRITON_ERR})
        variants = []
    h_chk = H_real[:8].contiguous()
    for fmt in ("fp8", "int4"):
        fvars = [v for v in variants if v.startswith(fmt + "_")]
        if not fvars:
            continue
        try:
            w, deq = (quant_fp8 if fmt == "fp8" else quant_int4)(W)
            ref_chk = F.linear(h_chk.float(), deq)
            ref_chk.index_fill_(1, sup, float("-inf"))
            info[f"{fmt}_rel_weight_err"] = round(float((deq - W.float()).norm() / W.float().norm()), 5)
            del deq
            copies = ring_copies(fmt, w)
            wbytes = weight_bytes(fmt, w)
        except Exception as ex:  # noqa: BLE001
            emit(rows, {"kind": "error", "stage": f"quantize {fmt}", "error": repr(ex)[:300]})
            continue
        torch.cuda.empty_cache()
        decode = 0
        for var in fvars:
            fused = var.endswith("_fused")
            # ---- correctness + config selection (fail-soft per config)
            tune = []
            h28 = H_real[:28].contiguous()

            def attempt(ci, cfg, dec, rec):
                err, ok = check_scan(fmt, fused, cfg, w, h_chk, ref_chk, args.topt, dec)
                rec.update({"decode": dec, "max_abs_err": err, "ok": ok})
                if ok:
                    for mm in (16, 32, 64):                            # compile every BLOCK_M bucket
                        run_scan(fmt, fused, cfg, w, H_real[:mm].contiguous(), args.topt, dec)
                    fns = [lambda c=c: run_scan(fmt, fused, cfg, c, h28, args.topt, dec) for c in copies]
                    rec["scan_us_M28"] = round(1e3 * bench_ring(fns, 50), 2)
                    tune.append((rec["scan_us_M28"], ci))
                return ok

            for ci, cfg in enumerate(SCAN_CONFIGS):
                rec = {"kind": "tune", "variant": var, "cfg": cfg}
                try:
                    if not attempt(ci, cfg, decode, rec) and fmt == "fp8" and decode == 0:
                        rec["native_fp8_err"] = rec["max_abs_err"]
                        decode = 1                                     # wrong numbers: bit decode for all fp8
                        attempt(ci, cfg, decode, rec)
                except Exception as ex:  # noqa: BLE001
                    rec["error"] = repr(ex)[:300]
                    # Native e4m3 conversion unsupported (not a resource limit of this tile): retry with bit decode.
                    if fmt == "fp8" and decode == 0 and "resource" not in repr(ex).lower():
                        try:
                            rec["retried"] = True
                            decode = 1 if attempt(ci, cfg, 1, rec) else 0
                        except Exception as ex2:  # noqa: BLE001
                            rec["retry_error"] = repr(ex2)[:300]
                emit(rows, rec)
            if not tune:
                emit(rows, {"kind": "error", "stage": f"{var} no working config"})
                continue
            cfg = SCAN_CONFIGS[min(tune)[1]]

            # ---- recall on both activation sets (sorted top-64 prefix gives every k at once)
            toks_real = {}
            try:
                for name, H in sets.items():
                    res, toks = eval_recall(fmt, fused, cfg, w, W, H, refs[name], args.topt, decode, ks)
                    if name == "realistic":
                        toks_real = toks
                    for k in ks:
                        recall[(var, k, name)] = res[k]["recall_bf16"]
                        emit(rows, {"kind": "recall", "variant": var, "set": name, "k": k, "cfg": cfg,
                                    "synthetic": synthetic, **res[k]})
            except Exception as ex:  # noqa: BLE001
                emit(rows, {"kind": "error", "stage": f"{var} recall", "error": repr(ex)[:300]})

            # ---- timing: scan alone and scan + topk + rescore, cold weights
            for m in ms:
                h = H_real[:m].contiguous()
                if time.time() - t_start > args.budget_s:
                    emit(rows, {"kind": "timing", "variant": var, "M": m, "skipped": "time budget"})
                    continue
                try:
                    fns = [lambda c=c: run_scan(fmt, fused, cfg, c, h, args.topt, decode) for c in copies]
                    scan_us = round(1e3 * bench_ring(fns, args.iters), 2)
                except Exception as ex:  # noqa: BLE001
                    emit(rows, {"kind": "timing", "variant": var, "M": m, "error": repr(ex)[:300]})
                    continue
                for k in ks:
                    row = {"kind": "timing", "variant": var, "M": m, "k": k, "cfg": cfg, "decode": decode,
                           "synthetic": synthetic, "scan_us_eager": scan_us,
                           "scan_gbs": round(wbytes / scan_us / 1e3, 1)}
                    if not fused:
                        row["logits_bytes"] = m * V * 4 * 2                 # fp32 write + topk read
                    try:
                        tk, _ = pipeline(fmt, fused, cfg, w, W, h, k, args.topt, decode)
                        if k in toks_real:
                            row["triton_vs_eval_mismatch"] = int((tk != toks_real[k][:m]).sum())
                        row["matches_bf16_ref"] = int((tk == refs["realistic"]["bf16"][:m]).sum())
                        fns = [lambda c=c: pipeline(fmt, fused, cfg, c, W, h, k, args.topt, decode) for c in copies]
                        row["eager_us"] = round(1e3 * bench_ring(fns, args.iters), 2)
                        if not args.no_graph:
                            try:
                                row["graph_us"] = round(1e3 * graph_ring(fns, args.iters), 2)
                            except Exception as ex:  # noqa: BLE001
                                row["graph_error"] = repr(ex)[:300]
                        t = row.get("graph_us", row["eager_us"])
                        timing[(var, m, k)] = t
                        if m in base_us:
                            row["speedup"] = round(base_us[m] / t, 3)
                    except Exception as ex:  # noqa: BLE001
                        row["error"] = repr(ex)[:300]
                    emit(rows, row)
                torch.cuda.empty_cache()
        del copies, w
        torch.cuda.empty_cache()

    # ---- summary: fastest (variant, k) with BF16-argmax recall >= 0.999 on both sets
    sets_n = list(sets)
    qual = sorted({(v, k) for (v, k, _) in recall
                   if all(recall.get((v, k, s), 0.0) >= 0.999 for s in sets_n)})
    m_a, m_b = (4 if 4 in ms else ms[0]), (28 if 28 in ms else ms[-1])
    scored = [(timing[(v, m_a, k)] + timing[(v, m_b, k)], v, k) for v, k in qual
              if (v, m_a, k) in timing and (v, m_b, k) in timing]
    summary: dict = {"kind": "summary", "synthetic": synthetic, "criterion": "recall_bf16 >= 0.999 on " +
                     "+".join(sets_n), "timing_basis": "graph_us" if not args.no_graph else "eager_us",
                     "min_k_per_variant": {v: min([k for vv, k in qual if vv == v], default=None) for v in variants}}
    if scored:
        _, v, k = min(scored)
        summary["best"] = {"variant": v, "k": k,
                           "recall": {s: recall[(v, k, s)] for s in sets_n}}
        for m in (m_a, m_b):
            t, b = timing[(v, m, k)], base_us.get(m)
            summary[f"M{m}"] = {"us": t, "baseline_us": b, "speedup": round(b / t, 3) if b else None,
                                "saving_us_per_decode_step": round(DRAFT_STEPS * (b - t), 1) if b else None}
    else:
        summary["best"] = None
    summary["note"] = ("Recall here is vs synthetic activations; true recall must be confirmed by end-to-end draft "
                       "acceptance in serving (real hidden states, real token distribution).")
    summary["wall_s"] = round(time.time() - t_start, 1)
    emit(rows, summary)
    with open(os.path.join(out_dir, "draft_head_scan.json"), "w") as f:
        json.dump({"info": info, "rows": rows}, f, indent=1, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
