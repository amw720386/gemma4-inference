"""Kernel lab: Gemma 4 12B chunked-prefill attention v2 (kernels/prefill_attn_v2.py) vs v1 vs vLLM's Triton kernel.

Run on Modal inside the pinned vLLM image:
  g4bench modal script --path kernels/prefill_attn_v2_bench.py --execute [-- ARGS]
Writes $G4_OUT_DIR/prefill_attn_v2_bench.json (rewritten after every shape: a timeout keeps partial results) and prints
one JSON line per result. Harness copied from kernels/prefill_attn_bench.py (v1; same shapes, timing and checks):
  global  head_dim 512, 16 q : 1 KV head, full causal          (x8 per forward)
  sliding head_dim 256, 16 q : 8 KV heads, window 1024 causal  (x40 per forward)
  chunks  the serving chunks of an 8192-token prompt for every --chunks value: 8192 -> 0:8192 (E8c serving config,
          measured first), 3072 -> 0:3072, 3072:3072, 6144:2048 (E8a/E8b); plus --shapes extras.
Implementations: vllm_2d (vLLM 0.33.0 unified_attention, 2D path, paged cache as TritonAttentionImpl passes it),
v1:<name> (kernels/prefill_attn.py; default its two per-layer winners from D-059 plus one runner-up), v2:<name>
(every kernels/prefill_attn_v2.py variant for the layer, or --variants), optional fi (FlashInfer single prefill,
--flashinfer; JIT may compile for minutes, so off by default).
Timing: CUDA graph of 4 back-to-back calls, CUDA-event timed (eager fallback flagged). TFLOPS counts only
causal(/window)-valid (query, key) pairs: 4 x heads x head_dim x pairs.
Numerics (lossless gate): small non-aligned / window-crossing / split cases vs an fp32 torch reference (max abs <
TOL) and vs vllm_2d; every timed shape vs vllm_2d (and v2 vs v1). A failing or non-compiling variant is recorded
in `broken` and skipped for the rest of that layer (fail-soft); NaN (unwritten rows) fails.
Budget: no new measurement starts after --budget-s (default 330 s) so the run stays < 7 min GPU incl. image start.
Summary per chunk config: per-8K-prompt prefill attention (sum over chunks x 8 global + 40 sliding layers) for
vLLM, best single v1 variant, best single v2 variant; TFLOPS of each per layer.
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
SCALE = 1.0  # Gemma 4: self.scaling = 1.0, no attention logit softcap
LAYERS = {
    "global": {"heads": 16, "kv_heads": 1, "head_dim": 512, "window": 0, "count": 8},
    "sliding": {"heads": 16, "kv_heads": 8, "head_dim": 256, "window": 1024, "count": 40},
}
TOL = 3e-2
NUMERIC_CASES = [(0, 100), (203, 77), (1500, 150), (40, 1100), (2048, 333)]  # (start, T)
V1_DEFAULT = {"global": ["m64_dv512_n32_d64", "m32_dv512_n16_q"],
              "sliding": ["m64_dv256_n64_d64", "m128_dv256_n32_d64"]}


def log(obj) -> None:
    print(json.dumps(obj), flush=True)


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


def parse_shapes(s: str) -> list[tuple[int, int]]:
    out = []
    for item in s.split(","):
        if item:
            a, b = item.split(":")
            out.append((int(a), int(b)))
    return out


def prompt_chunks(prompt_len: int, chunk: int) -> list[tuple[int, int]]:
    return [(s, min(chunk, prompt_len - s)) for s in range(0, prompt_len, chunk)]


def bench_shapes(args) -> list[tuple[int, int]]:
    shapes = []
    for ch in args.chunk_list:
        shapes += [s for s in prompt_chunks(args.prompt_len, ch) if s not in shapes]
    shapes += [s for s in parse_shapes(args.shapes) if s not in shapes]
    return shapes


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--layers", default="global,sliding")
    p.add_argument("--chunks", default="8192,3072", help="serving chunk sizes (--max-num-batched-tokens) to cover")
    p.add_argument("--shapes", default="", help="extra start:T chunks")
    p.add_argument("--prompt-len", type=int, default=8192)
    p.add_argument("--block-size-global", type=int, default=64)
    p.add_argument("--block-size-sliding", type=int, default=16)
    p.add_argument("--kv", choices=["contig", "pool"], default="contig")
    p.add_argument("--variants", default="", help="comma list of v2 variant names; default all for the layer")
    p.add_argument("--v1-variants", default="", help="comma list of v1 variant names; default the D-059 winners")
    p.add_argument("--splits", default="", help="also run these split-KV factors of every listed v2 variant, e.g. 2,3")
    p.add_argument("--skip-vllm", action="store_true")
    p.add_argument("--skip-v1", action="store_true")
    p.add_argument("--flashinfer", action="store_true")
    p.add_argument("--min-ms", type=float, default=30.0)
    p.add_argument("--budget-s", type=float, default=330.0, help="stop starting new measurements after this wall time")
    a = p.parse_args(argv)
    a.chunk_list = [int(x) for x in a.chunks.split(",") if x]
    return a


def over_budget(args) -> bool:
    return time.time() - T0 > args.budget_s


def time_calls(calls, min_ms: float = 30.0) -> tuple[float, str]:
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


class Chunk:
    """One prefill chunk of one layer: queries [T, H, D] at [start, start+T) and a paged K|V cache laid out like
    vLLM's Triton backend, plus [S, KVH, D] K/V (contiguous copies or pool views) for v1/v2."""

    def __init__(self, L, start, T, bs, kv_mode, dev):
        H, KVH, D = L["heads"], L["kv_heads"], L["head_dim"]
        self.start, self.T, self.S = start, T, start + T
        self.nblk = cdiv(self.S, bs)
        pool = torch.empty((self.nblk, bs, KVH, 2 * D), dtype=torch.bfloat16, device=dev)
        pool.normal_()
        self.pool = pool
        self.key_cache, self.value_cache = pool.split(D, dim=-1)
        self.block_table = torch.arange(self.nblk, dtype=torch.int32, device=dev).view(1, self.nblk)
        flat = pool.view(self.nblk * bs, KVH, 2 * D)
        k, v = flat[:, :, :D], flat[:, :, D:]
        if kv_mode == "contig":
            k, v = k[:self.S].contiguous(), v[:self.S].contiguous()
        self.k, self.v = k, v
        self.q = (torch.randn(T, H, D, device=dev) * (2.0 / math.sqrt(D))).to(torch.bfloat16)
        self.cu = torch.tensor([0, T], dtype=torch.int32, device=dev)
        self.seqused = torch.tensor([self.S], dtype=torch.int32, device=dev)
        self.kd = torch.ones(1, dtype=torch.float32, device=dev).expand(1, KVH)


def make_vllm_call(ua, c: Chunk, out, window):
    def f():
        ua(q=c.q, k=c.key_cache, v=c.value_cache, out=out, cu_seqlens_q=c.cu, max_seqlen_q=c.T, seqused_k=c.seqused,
           max_seqlen_k=c.S, softmax_scale=SCALE, causal=True,
           window_size=(window - 1, 0) if window else (-1, -1), block_table=c.block_table, softcap=0,
           q_descale=None, k_descale=c.kd, v_descale=c.kd, seq_threshold_3D=None, num_par_softmax_segments=None,
           softmax_segm_output=None, softmax_segm_max=None, softmax_segm_expsum=None)
    return f


class Impl:
    """One timed implementation: name ("v1:..", "v2:..", "fi") and a call factory (chunk, out, window) -> fn."""

    def __init__(self, name, factory):
        self.name, self.factory = name, factory


def make_impls(mods, layer, args) -> list[Impl]:
    impls = []
    v1, v2, fi = mods.get("v1"), mods.get("v2"), mods.get("fi")
    if v1 is not None and not args.skip_v1:
        pool = v1.GLOBAL_VARIANTS if layer == "global" else v1.SLIDING_VARIANTS
        names = args.v1_variants.split(",") if args.v1_variants else V1_DEFAULT[layer]
        for n in names:
            if n in pool:
                vt = pool[n]
                impls.append(Impl(f"v1:{n}", lambda c, o, w, vt=vt: (
                    lambda: v1.prefill_attention(c.q, c.k, c.v, c.start, o, vt, window=w, scale=SCALE))))
    if v2 is not None:
        pool = v2.GLOBAL_VARIANTS if layer == "global" else v2.SLIDING_VARIANTS
        vts = [pool[n] for n in args.variants.split(",") if n in pool] if args.variants else list(pool.values())
        for sp in [int(x) for x in args.splits.split(",") if x]:
            vts += [v2.with_splits(vt, sp) for vt in list(vts) if vt.splits == 1]
        for vt in vts:
            impls.append(Impl(f"v2:{vt.name}", lambda c, o, w, vt=vt: (
                lambda: v2.prefill_attention(c.q, c.k, c.v, c.start, o, vt, window=w, scale=SCALE))))
    if fi is not None:
        def fi_factory(c, o, w):
            def f():
                r = fi.single_prefill_with_kv_cache(c.q, c.k, c.v, causal=True, kv_layout="NHD", sm_scale=SCALE,
                                                    window_left=(w - 1) if w else -1)
                o.copy_(r)
            return f
        impls.append(Impl("fi", fi_factory))
    return impls


def block_size(layer, args):
    return args.block_size_global if layer == "global" else args.block_size_sliding


def numerics_check(ua, mods, layer, args, dev, broken):
    L = LAYERS[layer]
    W = L["window"]
    ref_mod = mods.get("v2") or mods.get("v1")
    rows = []
    for start, T in NUMERIC_CASES:
        c = Chunk(L, start, T, block_size(layer, args), args.kv, dev)
        ref = ref_mod.reference_attention(c.q, c.k, c.v, start, SCALE, window=W)
        vout = None
        if ua is not None:
            row = {"check": "numerics", "layer": layer, "start": start, "T": T, "impl": "vllm_2d"}
            try:
                vout = torch.empty_like(c.q)
                make_vllm_call(ua, c, vout, W)()
                torch.cuda.synchronize()
                row["max_abs_vs_ref"] = float((vout.float() - ref).abs().max())
                row["ok"] = bool(row["max_abs_vs_ref"] < TOL)
            except Exception as e:  # noqa: BLE001
                vout = None
                row["error"] = repr(e)[:400]
            rows.append(row)
            log(row)
        for im in make_impls(mods, layer, args):
            if im.name in broken:
                continue
            row = {"check": "numerics", "layer": layer, "start": start, "T": T, "impl": im.name}
            try:
                out = torch.full_like(c.q, float("nan"))
                im.factory(c, out, W)()
                torch.cuda.synchronize()
                err = (out.float() - ref).abs().max()
                row["max_abs_vs_ref"] = float(err)  # NaN (unwritten) -> not ok
                row["ok"] = bool(row["max_abs_vs_ref"] < TOL)
                if vout is not None:
                    row["max_abs_vs_vllm"] = float((out.float() - vout.float()).abs().max())
                if not row["ok"]:
                    broken[im.name] = f"numerics {start}:{T} max_abs {row['max_abs_vs_ref']}"
            except Exception as e:  # noqa: BLE001
                row["error"] = repr(e)[:600]
                broken[im.name] = row["error"]
            rows.append(row)
            log(row)
        del c, ref, vout
        torch.cuda.empty_cache()
    return rows


def run_case(ua, mods, layer, start, T, args, dev, broken):
    L = LAYERS[layer]
    H, D, W = L["heads"], L["head_dim"], L["window"]
    ref_mod = mods.get("v2") or mods.get("v1")
    pairs = ref_mod.valid_pairs(start, T, W)
    flops = 4 * H * D * pairs
    base = {"layer": layer, "start": start, "T": T, "ctx": start + T, "valid_pairs": pairs}
    c = Chunk(L, start, T, block_size(layer, args), args.kv, dev)
    rows, vout, v1out = [], None, None

    def finish(row, t):
        row.update({"us": round(t[0], 2), "timing": t[1], "elapsed_s": round(time.time() - T0, 1),
                    "tflops": round(flops / (t[0] * 1e6), 1)})

    if ua is not None and not args.skip_vllm and not over_budget(args):
        row = {**base, "impl": "vllm_2d"}
        try:
            vout = torch.empty_like(c.q)
            make_vllm_call(ua, c, vout, W)()
            torch.cuda.synchronize()
            o2 = torch.empty_like(c.q)
            finish(row, time_calls([make_vllm_call(ua, c, o2, W)] * 4, args.min_ms))
        except Exception as e:  # noqa: BLE001
            vout = None
            row["error"] = repr(e)[:400]
        rows.append(row)
        log(row)

    out = torch.empty_like(c.q)
    for im in make_impls(mods, layer, args):
        if over_budget(args):
            break
        row = {**base, "impl": im.name}
        if im.name in broken:
            row["skipped"] = broken[im.name][:200]
            rows.append(row)
            log(row)
            continue
        try:
            out.fill_(float("nan"))
            f = im.factory(c, out, W)
            f()
            torch.cuda.synchronize()
            if vout is not None:
                row["max_abs_vs_vllm"] = float((out.float() - vout.float()).abs().max())
                row["ok_vs_vllm"] = bool(row["max_abs_vs_vllm"] < TOL)
            if im.name.startswith("v1:") and v1out is None:
                v1out = out.clone()
            elif v1out is not None:
                row["max_abs_vs_v1"] = float((out.float() - v1out.float()).abs().max())
            finish(row, time_calls([f] * 4, args.min_ms))
        except Exception as e:  # noqa: BLE001
            row["error"] = repr(e)[:600]
            broken[im.name] = row["error"]
        rows.append(row)
        log(row)
    vr = next((x for x in rows if x["impl"] == "vllm_2d" and "us" in x), None)
    for r in rows:
        if "us" in r and r["impl"] != "vllm_2d" and vr:
            r["speedup_vs_vllm"] = round(vr["us"] / r["us"], 3)
    del c, vout, v1out, out
    torch.cuda.empty_cache()
    return rows


def usable(r) -> bool:
    return "us" in r and r["impl"] != "vllm_2d" and not r.get("excluded") and r.get("ok_vs_vllm", True)


def per_prompt(rows, args, broken) -> dict:
    """Per chunk config: prefill attention per 8K prompt = sum over its chunks x layer count, for vLLM and the best
    single variant (same variant for every chunk) of each family (v1, v2, fi). Raw times (no fallback to vLLM)."""
    for r in rows:
        if r.get("impl", "vllm_2d") != "vllm_2d" and r["impl"] in broken.get(r.get("layer"), {}):
            r["excluded"] = True
    res = {"prompt_len": args.prompt_len, "configs": {}}
    for ch in args.chunk_list:
        chunks = prompt_chunks(args.prompt_len, ch)
        cfg = {"chunks": chunks, "layers": {}}
        tot: dict[str, float] = {}
        complete = {}
        for layer, L in LAYERS.items():
            sel = lambda r, s, t: r.get("layer") == layer and r.get("start") == s and r.get("T") == t  # noqa: E731
            pairs = sum(rows_pairs(rows, layer, s, t) for s, t in chunks)
            flops = 4 * L["heads"] * L["head_dim"] * pairs
            ent = {"count": L["count"]}
            v = [next((r["us"] for r in rows if sel(r, s, t) and r["impl"] == "vllm_2d" and "us" in r), None)
                 for s, t in chunks]
            fams = {"vllm": {"vllm_2d": v}}
            for r in rows:
                if r.get("layer") == layer and usable(r):
                    fam = r["impl"].split(":")[0]
                    fams.setdefault(fam, {}).setdefault(r["impl"], [None] * len(chunks))
            for fam, cands in fams.items():
                if fam == "vllm":
                    continue
                for n in cands:
                    cands[n] = [next((r["us"] for r in rows if sel(r, s, t) and r["impl"] == n and usable(r)), None)
                                for s, t in chunks]
            for fam, cands in fams.items():
                full = {n: sum(ts) for n, ts in cands.items() if all(x is not None for x in ts)}
                if not full:
                    complete[fam] = False
                    continue
                best = min(full, key=full.get)
                ent[fam] = {"best": best, "us": round(full[best], 1),
                            "tflops": round(flops / (full[best] * 1e6), 1) if pairs else None,
                            "us_per_chunk": cands[best]}
                tot[fam] = tot.get(fam, 0.0) + L["count"] * full[best] / 1e3
                complete.setdefault(fam, True)
            for fam in ("v1", "v2", "fi"):
                if fam in ent and "vllm" in ent:
                    ent[fam]["speedup_vs_vllm"] = round(ent["vllm"]["us"] / ent[fam]["us"], 3)
            if "v1" in ent and "v2" in ent:
                ent["v2"]["speedup_vs_v1"] = round(ent["v1"]["us"] / ent["v2"]["us"], 3)
            cfg["layers"][layer] = ent
        cfg["ms_per_prompt"] = {k: round(x, 2) for k, x in tot.items() if complete.get(k)}
        mp = cfg["ms_per_prompt"]
        if "v2" in mp and "v1" in mp:
            cfg["v2_vs_v1"] = round(mp["v1"] / mp["v2"], 3)
        if "v2" in mp and "vllm" in mp:
            cfg["v2_vs_vllm"] = round(mp["vllm"] / mp["v2"], 3)
        res["configs"][str(ch)] = cfg
    return res


def rows_pairs(rows, layer, s, t) -> int:
    return next((r["valid_pairs"] for r in rows if r.get("layer") == layer and r.get("start") == s
                 and r.get("T") == t and r.get("valid_pairs")), 0)


def main(argv=None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    out_dir = os.environ.get("G4_OUT_DIR", ".")
    dev = torch.device("cuda")
    props = torch.cuda.get_device_properties(dev)
    info = {"gpu": props.name, "sm": f"{props.major}.{props.minor}", "sms": props.multi_processor_count,
            "torch": torch.__version__, "cuda": torch.version.cuda, "args": {k: v for k, v in vars(args).items()},
            "scale": SCALE}
    try:
        import triton
        info["triton"] = triton.__version__
    except Exception:  # noqa: BLE001
        pass
    torch.manual_seed(0)

    ua = None
    if not args.skip_vllm:
        try:
            import vllm
            from vllm.v1.attention.ops.triton_unified_attention import unified_attention as ua
            info["vllm"] = getattr(vllm, "__version__", "?")
            info["unified_attention_params"] = list(inspect.signature(ua).parameters)
        except Exception as e:  # noqa: BLE001
            info["vllm_import_error"] = repr(e)[:400]
    mods = {}
    try:
        import prefill_attn as v1
        mods["v1"] = v1
    except Exception as e:  # noqa: BLE001
        info["v1_import_error"] = repr(e)[:400]
    try:
        import prefill_attn_v2 as v2
        mods["v2"] = v2
        info["gluon_error"] = v2.GLUON_ERROR
        info["tma_error"] = v2.TMA_ERROR
    except Exception as e:  # noqa: BLE001
        info["v2_import_error"] = repr(e)[:400]
    if args.flashinfer:
        try:
            import flashinfer
            mods["fi"] = flashinfer
            info["flashinfer"] = getattr(flashinfer, "__version__", "?")
        except Exception as e:  # noqa: BLE001
            info["flashinfer_import_error"] = repr(e)[:400]
    info["import_s"] = round(time.time() - T0, 1)
    log(info)
    if "v2" not in mods and "v1" not in mods:
        return 1

    layers = [x for x in args.layers.split(",") if x in LAYERS]
    shapes = bench_shapes(args)
    broken = {layer: {} for layer in layers}
    checks, rows, stopped = [], [], None

    def dump():
        summary = per_prompt(rows, args, broken)
        info["wall_s"] = round(time.time() - T0, 1)
        with open(os.path.join(out_dir, "prefill_attn_v2_bench.json"), "w") as f:
            json.dump({"info": info, "numerics": checks, "rows": rows, "per_prompt": summary,
                       "broken": broken, "stopped": stopped}, f, indent=1)
        return summary

    for layer in layers:
        try:
            checks += numerics_check(ua, mods, layer, args, dev, broken[layer])
        except Exception as e:  # noqa: BLE001
            checks.append({"check": "numerics", "layer": layer, "error": repr(e)[:400]})
            log(checks[-1])
        dump()
    for start, T in shapes:  # interleave layers so a budget stop keeps both layer types
        for layer in layers:
            if over_budget(args):
                stopped = {"stopped_early": f"budget {args.budget_s}s reached before {layer} {start}:{T}"}
                break
            try:
                rows += run_case(ua, mods, layer, start, T, args, dev, broken[layer])
            except Exception as e:  # noqa: BLE001
                rows.append({"layer": layer, "start": start, "T": T, "error": repr(e)[:400]})
                log(rows[-1])
                torch.cuda.empty_cache()
            dump()
        if stopped:
            log(stopped)
            break
    summary = dump()
    log({"per_prompt": summary})
    return 0


if __name__ == "__main__":
    sys.exit(main())
