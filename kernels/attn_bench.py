"""Kernel lab: Gemma 4 12B speculative-verify attention on the serving GPU (vLLM's Triton kernel vs a packed-row kernel).

Run on Modal inside the pinned vLLM image:  g4bench modal script --path kernels/attn_bench.py --execute [-- ARGS]
Writes $G4_OUT_DIR/attn_bench.json; prints one JSON line per result. Shapes (q_len 7 = depth 6 verify, scale 1.0,
no attention softcap, bf16 KV):
  global  head_dim 512, 16 q : 1 KV head, full causal          (x8 per forward)
  sliding head_dim 256, 16 q : 8 KV heads, window 1024 causal  (x40 per forward)
Implementations:
  vllm_3d / vllm_2d  vllm.v1.attention.ops.triton_unified_attention.unified_attention, called with the same
                     arguments as TritonAttentionImpl.forward (paged cache views split from one K|V tensor, int32
                     block table / seq_lens / query_start_loc). 3D (split-K) is forced with a huge seq_threshold_3D,
                     2D by passing no threshold/scratch; which one the real engine picks at threshold 28 (D-044) is
                     reported as `tuned_path`. The 3D path is verified by checking that its scratch was written.
  mqa:<variant>      kernels/mqa_verify_attn.py (packed rows, split-K + LSE merge) on a contiguous strided view of
                     the SAME K|V memory (needs NHD layout or 1 KV head).
Timing: CUDA graphs of back-to-back calls (CUDA-event timed; eager fallback is flagged). `hot` repeats one KV set
(L2-resident when small: the 128 MB L2 flatters it); `cold` cycles through R KV sets totalling >= --cold-mb so
every call reads data evicted from L2. GB/s = unique KV bytes (K+V actually needed) / time.
Numerics: small heterogeneous case vs an fp32 torch reference; every bench shape: mqa vs vllm output.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

T0 = time.time()
SCALE = 1.0  # Gemma 4: self.scaling = 1.0 (q/k RMSNorm), no attn_logit_softcapping in the config
LAYERS = {
    "global": {"heads": 16, "kv_heads": 1, "head_dim": 512, "window": 0, "count": 8},
    "sliding": {"heads": 16, "kv_heads": 8, "head_dim": 256, "window": 1024, "count": 40},
}
MODEL_KV_HEADS = 8              # TritonAttentionMetadataBuilder uses the model-level KV heads (8) for its rules
NUM_PAR_SOFTMAX_SEGMENTS = 16   # vLLM triton_attn default; 64 on SM120 when num_seqs <= (SMs-1)//(16*8)
TOL = 3e-2


def log(obj) -> None:
    print(json.dumps(obj), flush=True)


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ints = lambda s: [int(x) for x in s.split(",") if x]  # noqa: E731
    p.add_argument("--layers", default="global,sliding")
    p.add_argument("--ctx", type=ints, default=[8192, 1024], help="context tokens before the q_len new ones")
    p.add_argument("--batch", type=ints, default=[4, 1, 16, 64], help="requests")
    p.add_argument("--q-len", type=int, default=7)
    p.add_argument("--kv-layout", choices=["nhd", "hnd"], default="nhd",
                   help="physical order of vLLM's (blocks, kv_heads, block_size, 2*hd) cache: nhd = slot-major")
    p.add_argument("--block-size-global", type=int, default=64, help="hybrid page-size unification guess")
    p.add_argument("--block-size-sliding", type=int, default=16)
    p.add_argument("--segments", default="auto", help="3D split count: auto (vLLM SM120 rule) | 16 | 64")
    p.add_argument("--seq-threshold", type=int, default=28, help="seq_threshold_3D of the tuned engine config")
    p.add_argument("--cold-mb", type=int, default=512, help="min total unique KV bytes cycled in cold mode")
    p.add_argument("--max-pool-gb", type=float, default=24.0)
    p.add_argument("--target-programs", type=int, default=0, help="mqa split rule target (0 = 2 x SMs)")
    p.add_argument("--mqa-variants", default="", help="comma list; default all for the layer")
    p.add_argument("--no-split-sweep", action="store_true", help="time only the rule's split count")
    p.add_argument("--skip-vllm", action="store_true")
    p.add_argument("--skip-mqa", action="store_true")
    p.add_argument("--budget-s", type=float, default=330.0, help="stop starting new cases after this wall time")
    return p.parse_args(argv)


def over_budget(args) -> bool:
    return time.time() - T0 > args.budget_s


def time_calls(calls, min_ms: float = 25.0) -> tuple[float, str]:
    """Mean microseconds per call over a CUDA graph of `calls` (eager loop if capture fails)."""
    for f in calls:
        f()
    torch.cuda.synchronize()
    mode, g = "graph", None
    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for f in calls[:2]:
                f()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for f in calls:
                f()
        run = g.replay
    except Exception:  # noqa: BLE001
        mode, g = "eager", None
        torch.cuda.synchronize()

        def run():
            for f in calls:
                f()
    run()
    torch.cuda.synchronize()
    st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    st.record()
    run()
    en.record()
    torch.cuda.synchronize()
    reps = max(3, min(200, int(min_ms / max(st.elapsed_time(en), 1e-3))))
    st.record()
    for _ in range(reps):
        run()
    en.record()
    torch.cuda.synchronize()
    us = st.elapsed_time(en) * 1000.0 / (reps * len(calls))
    del g
    return us, mode


class KVSet:
    """R rotation instances x B requests in one paged K|V pool, laid out like vLLM's Triton backend cache
    (logical (blocks, kv_heads, block_size, 2*hd), split into K and V views). Each request owns consecutive blocks,
    so the same memory also has an affine [B, tokens, kv_heads, hd] view for the contiguous-layout kernel."""

    def __init__(self, B, seq_lens, kvh, d, bs, layout, R, dev):
        self.nblk = cdiv(max(seq_lens), bs)
        nb = R * B * self.nblk
        if layout == "nhd":
            pool = torch.empty((nb, bs, kvh, 2 * d), dtype=torch.bfloat16, device=dev)
            kv = pool
        else:
            pool = torch.empty((nb, kvh, bs, 2 * d), dtype=torch.bfloat16, device=dev)
            kv = pool.transpose(1, 2)
        pool.normal_()
        self.pool = pool
        self.key_cache, self.value_cache = kv.split(d, dim=-1)  # [blocks, block_size, kv_heads, hd] views
        self.block_tables = [(torch.arange(B * self.nblk, dtype=torch.int32, device=dev) + i * B * self.nblk)
                             .view(B, self.nblk) for i in range(R)]
        self.contig = layout == "nhd" or kvh == 1
        self.k, self.v = [], []
        if self.contig:
            tok = kvh * 2 * d
            hs = 2 * d if layout == "nhd" else bs * 2 * d
            size = (B, self.nblk * bs, kvh, d)
            stride = (self.nblk * bs * tok, tok, hs, 1)
            for i in range(R):
                off = i * B * self.nblk * bs * tok
                self.k.append(torch.as_strided(pool, size, stride, off))
                self.v.append(torch.as_strided(pool, size, stride, off + d))


def vllm_segments(args, B: int, sms: int) -> int:
    if args.segments != "auto":
        return int(args.segments)
    max_seqs_64 = (sms - 1) // (NUM_PAR_SOFTMAX_SEGMENTS * MODEL_KV_HEADS)
    return 64 if B <= max_seqs_64 else NUM_PAR_SOFTMAX_SEGMENTS


def make_vllm_call(ua, q, kvset, inst, out, cu, sl_t, max_k, q_len, window, kd, vd, scratch, segs):
    kc, vc, bt = kvset.key_cache, kvset.value_cache, kvset.block_tables[inst]
    so, sm, se = scratch if scratch is not None else (None, None, None)
    thr = 1 << 30 if scratch is not None else None

    def f():
        ua(q=q, k=kc, v=vc, out=out, cu_seqlens_q=cu, max_seqlen_q=q_len, seqused_k=sl_t, max_seqlen_k=max_k,
           softmax_scale=SCALE, causal=True, window_size=(window - 1, 0) if window else (-1, -1),
           block_table=bt, softcap=0, q_descale=None, k_descale=kd, v_descale=vd,
           seq_threshold_3D=thr, num_par_softmax_segments=segs if scratch is not None else None,
           softmax_segm_output=so, softmax_segm_max=sm, softmax_segm_expsum=se)
    return f


def build_inputs(L, B, seq_lens, q_len, bs, layout, R, dev):
    H, KVH, D = L["heads"], L["kv_heads"], L["head_dim"]
    kvset = KVSet(B, seq_lens, KVH, D, bs, layout, R, dev)
    # unit-RMS K/V (Gemma 4 RMS-norms both); q scaled so logits have std ~2 at scale 1.0
    q = (torch.randn(B * q_len, H, D, device=dev) * (2.0 / math.sqrt(D))).to(torch.bfloat16)
    cu = torch.arange(0, B + 1, dtype=torch.int32, device=dev) * q_len
    sl_t = torch.tensor(seq_lens, dtype=torch.int32, device=dev)
    kd = torch.ones(1, dtype=torch.float32, device=dev).expand(B, KVH)
    return kvset, q, cu, sl_t, kd


def vllm_scratch(B, q_len, L, segs, dev):
    import triton
    dp = triton.next_power_of_2(L["head_dim"])
    so = torch.empty((B * q_len, L["heads"], segs, dp), dtype=torch.float32, device=dev)
    sm = torch.empty((B * q_len, L["heads"], segs), dtype=torch.float32, device=dev)
    se = torch.empty((B * q_len, L["heads"], segs), dtype=torch.float32, device=dev)
    return so, sm, se


def pick_variants(mqa, layer, args):
    pool = mqa.GLOBAL_VARIANTS if layer == "global" else mqa.SLIDING_VARIANTS
    if args.mqa_variants:
        names = [n for n in args.mqa_variants.split(",") if n in pool]
        return [pool[n] for n in names]
    return list(pool.values())


def numerics_check(ua, mqa, layer, args, dev, sms):
    """Two requests with different, non-aligned contexts (sliding: one beyond the window) vs fp32 torch."""
    L = LAYERS[layer]
    q_len, W = args.q_len, L["window"]
    G = L["heads"] // L["kv_heads"]
    seq_lens = [517 + q_len, 1290 + q_len]
    bs = args.block_size_global if layer == "global" else args.block_size_sliding
    kvset, q, cu, sl_t, kd = build_inputs(L, 2, seq_lens, q_len, bs, args.kv_layout, 1, dev)
    rows = []
    if not kvset.contig:
        k_ref = None
    else:
        k_ref, v_ref = kvset.k[0], kvset.v[0]
    if k_ref is None:  # gather through the block table (hnd with several KV heads)
        nb = kvset.nblk
        k_ref = kvset.key_cache[kvset.block_tables[0].long()].reshape(2, nb * bs, L["kv_heads"], L["head_dim"])
        v_ref = kvset.value_cache[kvset.block_tables[0].long()].reshape(2, nb * bs, L["kv_heads"], L["head_dim"])
    ref = mqa.reference_attention(q, k_ref, v_ref, seq_lens, q_len, SCALE, window=W) if mqa else None
    outs = {}
    if ua is not None:
        for path in ("3d", "2d"):
            row = {"check": "numerics", "layer": layer, "seq_lens": seq_lens, "impl": f"vllm_{path}"}
            try:
                out = torch.empty_like(q)
                scratch = vllm_scratch(2, q_len, L, vllm_segments(args, 2, sms), dev) if path == "3d" else None
                if scratch is not None:
                    scratch[1].fill_(float("nan"))
                make_vllm_call(ua, q, kvset, 0, out, cu, sl_t, max(seq_lens), q_len, W, kd, kd, scratch,
                               vllm_segments(args, 2, sms))()
                torch.cuda.synchronize()
                if scratch is not None:
                    row["path_verified"] = bool(not torch.isnan(scratch[1]).all())
                outs[path] = out
                if ref is not None:
                    row["max_abs_vs_ref"] = float((out.float() - ref).abs().max())
                    row["ok"] = row["max_abs_vs_ref"] < TOL
            except Exception as e:  # noqa: BLE001
                row["error"] = repr(e)[:400]
            rows.append(row)
            log(row)
    if mqa is not None and kvset.contig and not args.skip_mqa:
        span = min(max(seq_lens), W + q_len - 1) if W else max(seq_lens)
        for v in pick_variants(mqa, layer, args):
            for ns in sorted({3, mqa.choose_num_splits(2, L["kv_heads"], span, q_len, G, v, 2 * sms)}):
                row = {"check": "numerics", "layer": layer, "seq_lens": seq_lens, "impl": f"mqa:{v.name}",
                       "splits": ns}
                try:
                    out = torch.empty_like(q)
                    ws = mqa.alloc_workspace(2, L["kv_heads"], ns, q_len, G, L["head_dim"], v, dev)
                    mqa.mqa_verify_attention(q, kvset.k[0], kvset.v[0], sl_t, q_len, SCALE, out, ws, ns, v, window=W)
                    torch.cuda.synchronize()
                    row["max_abs_vs_ref"] = float((out.float() - ref).abs().max())
                    row["ok"] = row["max_abs_vs_ref"] < TOL
                    for path, o in outs.items():
                        row[f"max_abs_vs_vllm_{path}"] = float((out.float() - o.float()).abs().max())
                except Exception as e:  # noqa: BLE001
                    row["error"] = repr(e)[:400]
                rows.append(row)
                log(row)
    del kvset
    torch.cuda.empty_cache()
    return rows


def run_case(ua, mqa, layer, ctx, B, args, dev, sms, copy_gbs):
    L = LAYERS[layer]
    H, KVH, D, W = L["heads"], L["kv_heads"], L["head_dim"], L["window"]
    G = H // KVH
    q_len = args.q_len
    seq_len = ctx + q_len
    span = min(seq_len, W + q_len - 1) if W else seq_len
    touched = B * span * KVH * D * 2 * 2  # unique K+V bytes the step needs
    bs = args.block_size_global if layer == "global" else args.block_size_sliding
    inst_bytes = B * cdiv(seq_len, bs) * bs * KVH * 2 * D * 2
    R = max(2, cdiv(args.cold_mb * 2**20, touched))
    R = max(1, min(R, int(args.max_pool_gb * 2**30 // inst_bytes), 512))
    base = {"layer": layer, "ctx": ctx, "batch": B, "q_len": q_len, "seq_len": seq_len, "kv_tokens_needed": span,
            "unique_kv_mb": round(touched / 1e6, 3), "rotate_sets": R, "cold_set_mb": round(R * touched / 1e6, 1),
            "kv_floor_us": round(touched / (copy_gbs * 1e3), 3)}
    flops = 4 * B * q_len * H * span * D  # QK^T + PV over the needed keys (upper bound for causal/window)
    kvset, q, cu, sl_t, kd = build_inputs(L, B, [seq_len] * B, q_len, bs, args.kv_layout, R, dev)
    n_cold = R if R >= 16 else R * cdiv(16, R)
    rows, vref = [], {}

    def finish(row, hot, cold):
        row.update({"hot_us": round(hot[0], 3), "cold_us": round(cold[0], 3), "timing": cold[1],
                    "hot_gbs": round(touched / (hot[0] * 1e3), 1), "cold_gbs": round(touched / (cold[0] * 1e3), 1),
                    "cold_tflops": round(flops / (cold[0] * 1e6), 2),
                    "cold_vs_floor": round(cold[0] / base["kv_floor_us"], 2),
                    "elapsed_s": round(time.time() - T0, 1)})

    if ua is not None and not args.skip_vllm:
        block_m = 16 if G <= 16 else 1 << (G - 1).bit_length()
        q_blocks = B * cdiv(q_len, block_m // G)
        segs = vllm_segments(args, B, sms)
        tuned = "3d" if q_blocks <= args.seq_threshold else "2d"
        stock = "3d" if q_blocks <= 128 // MODEL_KV_HEADS else "2d"
        out = torch.empty_like(q)
        for path in ("3d", "2d"):
            row = {**base, "impl": f"vllm_{path}", "segments": segs if path == "3d" else 1,
                   "block_q": block_m // G, "tuned_path": tuned, "stock_path": stock}
            try:
                scratch = vllm_scratch(B, q_len, L, segs, dev) if path == "3d" else None
                calls = [make_vllm_call(ua, q, kvset, i % R, out, cu, sl_t, seq_len, q_len, W, kd, kd, scratch, segs)
                         for i in range(n_cold)]
                if scratch is not None:
                    scratch[1].fill_(float("nan"))
                calls[0]()
                torch.cuda.synchronize()
                if scratch is not None:
                    row["path_verified"] = bool(not torch.isnan(scratch[1]).all())
                vref[path] = out.clone()
                hot = time_calls([calls[0]] * 16)
                cold = time_calls(calls)
                finish(row, hot, cold)
                del scratch
            except Exception as e:  # noqa: BLE001
                row["error"] = repr(e)[:400]
            rows.append(row)
            log(row)
        if "3d" in vref and "2d" in vref:
            rows[-1]["max_abs_2d_vs_3d"] = float((vref["3d"].float() - vref["2d"].float()).abs().max())
            log({"layer": layer, "ctx": ctx, "batch": B, "max_abs_2d_vs_3d": rows[-1]["max_abs_2d_vs_3d"]})

    if mqa is not None and not args.skip_mqa:
        if not kvset.contig:
            row = {**base, "impl": "mqa", "skipped": "hnd layout with >1 KV head has no affine contiguous view"}
            rows.append(row)
            log(row)
        else:
            target = args.target_programs or 2 * sms
            out_m = torch.empty_like(q)
            for v in pick_variants(mqa, layer, args):
                if over_budget(args):
                    break
                row = {**base, "impl": f"mqa:{v.name}"}
                try:
                    n0 = mqa.choose_num_splits(B, KVH, span, q_len, G, v, target)
                    max_ns = max(1, cdiv(span, v.block_n))
                    cands = [n0] if args.no_split_sweep else sorted({max(1, n0 // 2), n0, min(2 * n0, max_ns)})
                    ws = mqa.alloc_workspace(B, KVH, max(cands), q_len, G, D, v, dev)

                    def mk(i, ns, v=v, ws=ws):
                        return lambda: mqa.mqa_verify_attention(q, kvset.k[i], kvset.v[i], sl_t, q_len, SCALE, out_m,
                                                                ws, ns, v, window=W)
                    mk(0, n0)()
                    torch.cuda.synchronize()
                    for path, o in vref.items():
                        row[f"max_abs_vs_vllm_{path}"] = float((out_m.float() - o.float()).abs().max())
                    tried = {}
                    for ns in cands:
                        tried[ns] = time_calls([mk(i % R, ns) for i in range(n_cold)])
                    best = min(tried, key=lambda k: tried[k][0])
                    row["splits"] = best
                    row["splits_rule"] = n0
                    row["splits_cold_us"] = {str(k): round(t[0], 3) for k, t in tried.items()}
                    row["programs"] = (cdiv(q_len * G, v.block_m) * (D // v.block_dv)) * best * B * KVH
                    hot = time_calls([mk(0, best)] * 16)
                    finish(row, hot, tried[best])
                    del ws
                except Exception as e:  # noqa: BLE001
                    row["error"] = repr(e)[:400]
                rows.append(row)
                log(row)
    del kvset, q
    torch.cuda.empty_cache()
    return rows


def per_forward(rows, args) -> list[dict]:
    """8 global + 40 sliding layers per forward, cold timings."""
    def cold(layer, ctx, B, pred):
        ts = [r["cold_us"] for r in rows if r.get("layer") == layer and r.get("ctx") == ctx and r.get("batch") == B
              and "cold_us" in r and pred(r)]
        return min(ts) if ts else None

    out = []
    for ctx in args.ctx:
        for B in args.batch:
            ent = {"ctx": ctx, "batch": B}
            parts = {}
            for layer in ("global", "sliding"):
                tr = [r for r in rows if r.get("layer") == layer and r.get("ctx") == ctx and r.get("batch") == B
                      and "tuned_path" in r]
                tuned_path = tr[0]["tuned_path"] if tr else None
                parts[layer] = {
                    "vllm_tuned": cold(layer, ctx, B, lambda r, p=tuned_path: r["impl"] == f"vllm_{p}"),
                    "vllm_best": cold(layer, ctx, B, lambda r: r["impl"].startswith("vllm_")),
                    "mqa_best": cold(layer, ctx, B, lambda r: r["impl"].startswith("mqa:")),
                    "floor": next((r["kv_floor_us"] for r in rows if r.get("layer") == layer and r.get("ctx") == ctx
                                   and r.get("batch") == B and "kv_floor_us" in r), None),
                }
            g, s = parts["global"], parts["sliding"]
            ng, ns_ = LAYERS["global"]["count"], LAYERS["sliding"]["count"]
            for key in ("vllm_tuned", "vllm_best", "floor"):
                if g[key] is not None and s[key] is not None:
                    ent[f"{key}_us"] = round(ng * g[key] + ns_ * s[key], 1)
            if g["mqa_best"] is not None and s["vllm_tuned"] is not None:
                s_best = min(x for x in (s["mqa_best"], s["vllm_tuned"]) if x is not None)
                ent["mqa_global_plus_best_sliding_us"] = round(ng * g["mqa_best"] + ns_ * s_best, 1)
            ent["global_us"] = {k: (round(x, 2) if x is not None else None) for k, x in g.items()}
            ent["sliding_us"] = {k: (round(x, 2) if x is not None else None) for k, x in s.items()}
            out.append(ent)
    return out


def main(argv=None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    out_dir = os.environ.get("G4_OUT_DIR", ".")
    dev = torch.device("cuda")
    props = torch.cuda.get_device_properties(dev)
    sms = props.multi_processor_count
    info = {"gpu": props.name, "sm": f"{props.major}.{props.minor}", "sms": sms, "torch": torch.__version__,
            "cuda": torch.version.cuda, "args": vars(args), "scale": SCALE}
    try:
        import triton
        info["triton"] = triton.__version__
    except Exception:  # noqa: BLE001
        pass
    torch.manual_seed(0)
    buf = torch.empty(512 * 1024**2, device=dev, dtype=torch.uint8)
    dst = torch.empty_like(buf)
    for _ in range(3):
        dst.copy_(buf)
    torch.cuda.synchronize()
    st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    st.record()
    for _ in range(20):
        dst.copy_(buf)
    en.record()
    torch.cuda.synchronize()
    copy_gbs = 2 * buf.numel() * 20 / (st.elapsed_time(en) * 1e6)
    info["copy_bandwidth_gbs"] = round(copy_gbs, 1)
    del buf, dst

    ua = None
    if not args.skip_vllm:
        try:
            import vllm
            from vllm.v1.attention.ops.triton_unified_attention import unified_attention as ua
            info["vllm"] = getattr(vllm, "__version__", "?")
            info["unified_attention_params"] = list(inspect.signature(ua).parameters)
        except Exception as e:  # noqa: BLE001
            info["vllm_import_error"] = repr(e)[:400]
    mqa = None
    if not args.skip_mqa:
        try:
            import mqa_verify_attn as mqa
        except Exception as e:  # noqa: BLE001
            info["mqa_import_error"] = repr(e)[:400]
    info["import_s"] = round(time.time() - T0, 1)
    log(info)

    layers = [x for x in args.layers.split(",") if x in LAYERS]
    checks, rows = [], []
    for layer in layers:
        try:
            checks += numerics_check(ua, mqa, layer, args, dev, sms)
        except Exception as e:  # noqa: BLE001
            checks.append({"check": "numerics", "layer": layer, "error": repr(e)[:400]})
            log(checks[-1])
    stopped = None
    for layer in layers:
        for ctx in args.ctx:
            for B in args.batch:
                if over_budget(args):
                    stopped = {"stopped_early": f"budget {args.budget_s}s reached before {layer}/{ctx}/{B}"}
                    break
                try:
                    rows += run_case(ua, mqa, layer, ctx, B, args, dev, sms, copy_gbs)
                except Exception as e:  # noqa: BLE001
                    rows.append({"layer": layer, "ctx": ctx, "batch": B, "error": repr(e)[:400]})
                    log(rows[-1])
                    torch.cuda.empty_cache()
            if stopped:
                break
        if stopped:
            break
    if stopped:
        log(stopped)
    summary = per_forward(rows, args)
    log({"per_forward": summary})
    info["wall_s"] = round(time.time() - T0, 1)
    with open(os.path.join(out_dir, "attn_bench.json"), "w") as f:
        json.dump({"info": info, "numerics": checks, "rows": rows, "per_forward": summary,
                   "stopped": stopped}, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
