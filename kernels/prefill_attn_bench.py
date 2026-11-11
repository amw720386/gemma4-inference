"""Kernel lab: Gemma 4 12B chunked-prefill attention on the serving GPU (vLLM's Triton 2D kernel vs packed-row FA2).

Run on Modal inside the pinned vLLM image:
  g4bench modal script --path kernels/prefill_attn_bench.py --execute [-- ARGS]
Writes $G4_OUT_DIR/prefill_attn_bench.json (rewritten after every shape, so a timeout keeps partial results) and
prints one JSON line per result. Shapes (batch 1, scale 1.0, no attention softcap, bf16 Q/KV):
  global  head_dim 512, 16 q : 1 KV head, full causal          (x8 per forward)
  sliding head_dim 256, 16 q : 8 KV heads, window 1024 causal  (x40 per forward)
  chunks  start:T = queries at positions [start, start+T) attending to keys [0, start+T); default the serving
          chunks of an 8192-token prompt at --max-num-batched-tokens 3072 (0:3072, 3072:3072, 6144:2048) plus
          6144:3072 and a single 0:8192 chunk.
Implementations:
  vllm_2d     vllm.v1.attention.ops.triton_unified_attention.unified_attention with the arguments
              TritonAttentionImpl.forward passes for a prefill chunk (paged cache views split from one K|V tensor,
              int32 block table / seqused_k / cu_seqlens_q, no 3D threshold -> 2D kernel, as for every q_len > 8).
  pf:<name>   kernels/prefill_attn.py variants on the same K/V values (contiguous [S, KVH, D] copies by default,
              --kv pool uses strided views of the paged pool memory itself).
Timing: CUDA graph of back-to-back calls, CUDA-event timed (eager fallback flagged). The K/V of one chunk (<= 19 MB)
is L2-resident either way: prefill attention is compute-bound, so there is no cold/hot split.
TFLOPS counts only causal(/window)-valid (query, key) pairs: 4 x heads x head_dim x pairs (QK^T + PV).
Numerics: small non-aligned cases (incl. window crossing) vs an fp32 torch reference and vs vllm_2d; every bench
shape vs vllm_2d. Every variant is fail-soft: a compile/launch error (e.g. OutOfResources) is recorded and the variant
skipped for the remaining shapes of that layer.
Summary: per-8K-prompt prefill attention = sum over its chunks x (8 global + 40 sliding layers), vLLM vs best.
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
TOL = 3e-2
NUMERIC_CASES = [(0, 100), (203, 77), (1500, 150), (40, 1100)]  # (start, T): non-aligned, window-crossing


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


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--layers", default="global,sliding")
    p.add_argument("--shapes", default="6144:3072,0:8192", help="extra start:T chunks beyond the prompt's chunks")
    p.add_argument("--prompt-len", type=int, default=8192)
    p.add_argument("--chunk", type=int, default=3072, help="--max-num-batched-tokens of the serving config")
    p.add_argument("--block-size-global", type=int, default=64, help="paged block size (hybrid unification guess)")
    p.add_argument("--block-size-sliding", type=int, default=16)
    p.add_argument("--kv", choices=["contig", "pool"], default="contig",
                   help="K/V for pf kernels: contiguous [S,KVH,D] copies, or strided views of the paged pool")
    p.add_argument("--variants", default="", help="comma list of pf variant names; default all for the layer")
    p.add_argument("--skip-vllm", action="store_true")
    p.add_argument("--skip-pf", action="store_true")
    p.add_argument("--min-ms", type=float, default=30.0, help="timed window per measurement")
    p.add_argument("--budget-s", type=float, default=330.0, help="stop starting new measurements after this wall time")
    return p.parse_args(argv)


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
    """One prefill chunk of one layer: queries [T, H, D] at [start, start+T), and a paged K|V cache laid out like
    vLLM's Triton backend ((blocks, block_size, kv_heads, 2*hd), NHD, split into K and V views; one request owning
    consecutive blocks), plus contiguous [S, KVH, D] K/V (copies or pool views) for the pf kernels."""

    def __init__(self, L, start, T, bs, kv_mode, dev):
        H, KVH, D = L["heads"], L["kv_heads"], L["head_dim"]
        self.start, self.T, self.S = start, T, start + T
        self.nblk = cdiv(self.S, bs)
        pool = torch.empty((self.nblk, bs, KVH, 2 * D), dtype=torch.bfloat16, device=dev)
        pool.normal_()  # unit-RMS K/V (Gemma 4 RMS-norms both)
        self.pool = pool
        self.key_cache, self.value_cache = pool.split(D, dim=-1)
        self.block_table = torch.arange(self.nblk, dtype=torch.int32, device=dev).view(1, self.nblk)
        flat = pool.view(self.nblk * bs, KVH, 2 * D)
        k, v = flat[:, :, :D], flat[:, :, D:]
        if kv_mode == "contig":
            k, v = k[:self.S].contiguous(), v[:self.S].contiguous()
        self.k, self.v = k, v
        # q scaled so logits have std ~2 at scale 1.0
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


def make_pf_call(pf, c: Chunk, out, variant, window):
    return lambda: pf.prefill_attention(c.q, c.k, c.v, c.start, out, variant, window=window, scale=SCALE)


def pick_variants(pf, layer, args):
    pool = pf.GLOBAL_VARIANTS if layer == "global" else pf.SLIDING_VARIANTS
    if args.variants:
        return [pool[n] for n in args.variants.split(",") if n in pool]
    return list(pool.values())


def block_size(layer, args):
    return args.block_size_global if layer == "global" else args.block_size_sliding


def numerics_check(ua, pf, layer, args, dev, broken):
    L = LAYERS[layer]
    W = L["window"]
    rows = []
    for start, T in NUMERIC_CASES:
        c = Chunk(L, start, T, block_size(layer, args), args.kv, dev)
        ref = pf.reference_attention(c.q, c.k, c.v, start, SCALE, window=W)
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
        if not args.skip_pf:
            for v in pick_variants(pf, layer, args):
                if v.name in broken:
                    continue
                row = {"check": "numerics", "layer": layer, "start": start, "T": T, "impl": f"pf:{v.name}"}
                try:
                    out = torch.full_like(c.q, float("nan"))
                    make_pf_call(pf, c, out, v, W)()
                    torch.cuda.synchronize()
                    row["max_abs_vs_ref"] = float((out.float() - ref).abs().max())  # NaN (unwritten) -> not ok
                    row["ok"] = bool(row["max_abs_vs_ref"] < TOL)
                    if vout is not None:
                        row["max_abs_vs_vllm"] = float((out.float() - vout.float()).abs().max())
                    if not row["ok"]:
                        broken[v.name] = f"numerics {start}:{T} max_abs {row['max_abs_vs_ref']}"
                except Exception as e:  # noqa: BLE001
                    row["error"] = repr(e)[:400]
                    broken[v.name] = row["error"]
                rows.append(row)
                log(row)
        del c, ref
        torch.cuda.empty_cache()
    return rows


def run_case(ua, pf, layer, start, T, args, dev, broken):
    L = LAYERS[layer]
    H, D, W = L["heads"], L["head_dim"], L["window"]
    pairs = pf.valid_pairs(start, T, W) if pf else None
    flops = 4 * H * D * pairs if pairs else None
    base = {"layer": layer, "start": start, "T": T, "ctx": start + T, "valid_pairs": pairs}
    c = Chunk(L, start, T, block_size(layer, args), args.kv, dev)
    rows, vout = [], None

    def finish(row, t):
        row.update({"us": round(t[0], 2), "timing": t[1], "elapsed_s": round(time.time() - T0, 1)})
        if flops:
            row["tflops"] = round(flops / (t[0] * 1e6), 1)

    if ua is not None and not args.skip_vllm and not over_budget(args):
        row = {**base, "impl": "vllm_2d"}
        try:
            vout = torch.empty_like(c.q)
            f = make_vllm_call(ua, c, vout, W)
            f()
            torch.cuda.synchronize()
            vout = vout.clone()
            o2 = torch.empty_like(c.q)
            finish(row, time_calls([make_vllm_call(ua, c, o2, W)] * 4, args.min_ms))
        except Exception as e:  # noqa: BLE001
            vout = None
            row["error"] = repr(e)[:400]
        rows.append(row)
        log(row)

    if pf is not None and not args.skip_pf:
        out = torch.empty_like(c.q)
        for v in pick_variants(pf, layer, args):
            if over_budget(args):
                break
            row = {**base, "impl": f"pf:{v.name}"}
            if v.name in broken:
                row["skipped"] = broken[v.name][:200]
                rows.append(row)
                log(row)
                continue
            try:
                out.fill_(float("nan"))
                f = make_pf_call(pf, c, out, v, W)
                f()
                torch.cuda.synchronize()
                if vout is not None:
                    row["max_abs_vs_vllm"] = float((out.float() - vout.float()).abs().max())
                    row["ok_vs_vllm"] = bool(row["max_abs_vs_vllm"] < TOL)
                finish(row, time_calls([f] * 4, args.min_ms))
            except Exception as e:  # noqa: BLE001
                row["error"] = repr(e)[:400]
                broken[v.name] = row["error"]
            rows.append(row)
            log(row)
    for r in rows:
        if "us" in r and r["impl"] != "vllm_2d":
            vr = next((x for x in rows if x["impl"] == "vllm_2d" and "us" in x), None)
            if vr:
                r["speedup_vs_vllm"] = round(vr["us"] / r["us"], 3)
    del c, vout
    torch.cuda.empty_cache()
    return rows


def usable(r) -> bool:
    return "us" in r and r["impl"].startswith("pf:") and not r.get("excluded") and r.get("ok_vs_vllm", True)


def per_prompt(rows, args, broken) -> dict:
    """Prefill attention per prompt: sum over its chunks x layer count, vLLM vs best pf (per chunk, and best single
    variant used for every chunk). Variants in broken[layer] (compile error / failed numerics) are excluded."""
    for r in rows:
        if r.get("impl", "").startswith("pf:") and r["impl"][3:] in broken.get(r.get("layer"), {}):
            r["excluded"] = True
    chunks = prompt_chunks(args.prompt_len, args.chunk)
    out = {"prompt_len": args.prompt_len, "chunks": chunks, "layers": {}}
    tot = {"vllm_ms": 0.0, "pf_best_per_chunk_ms": 0.0, "pf_best_single_ms": 0.0}
    complete = True
    for layer, L in LAYERS.items():
        sel = lambda r, s, t: r.get("layer") == layer and r.get("start") == s and r.get("T") == t  # noqa: E731
        ent = {"count": L["count"]}
        v = [next((r["us"] for r in rows if sel(r, s, t) and r["impl"] == "vllm_2d" and "us" in r), None)
             for s, t in chunks]
        bpc = [min((r["us"] for r in rows if sel(r, s, t) and usable(r)), default=None) for s, t in chunks]
        names = {r["impl"] for r in rows if r.get("layer") == layer and usable(r)}
        single = {}
        for n in names:
            ts = [next((r["us"] for r in rows if sel(r, s, t) and r["impl"] == n and usable(r)), None)
                  for s, t in chunks]
            if all(x is not None for x in ts):
                single[n] = sum(ts)
        ent["vllm_us_per_chunk"] = v
        ent["pf_best_us_per_chunk"] = bpc
        if all(x is not None for x in v):
            ent["vllm_us"] = round(sum(v), 1)
            tot["vllm_ms"] += L["count"] * sum(v) / 1e3
        else:
            complete = False
        if all(x is not None for x in bpc):
            ent["pf_best_per_chunk_us"] = round(sum(bpc), 1)
            tot["pf_best_per_chunk_ms"] += L["count"] * min(sum(bpc), ent.get("vllm_us", 1e30)) / 1e3
        else:
            complete = False
        if single:
            best = min(single, key=single.get)
            ent["pf_best_single"] = best
            ent["pf_best_single_us"] = round(single[best], 1)
            tot["pf_best_single_ms"] += L["count"] * min(single[best], ent.get("vllm_us", 1e30)) / 1e3
        else:
            complete = False
        if "vllm_us" in ent and "pf_best_single_us" in ent:
            ent["speedup_single"] = round(ent["vllm_us"] / ent["pf_best_single_us"], 3)
        out["layers"][layer] = ent
    out.update({k: round(x, 2) for k, x in tot.items()})
    out["complete"] = complete
    if complete:
        out["saved_ms_per_prompt"] = round(tot["vllm_ms"] - tot["pf_best_single_ms"], 2)
    return out


def main(argv=None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    out_dir = os.environ.get("G4_OUT_DIR", ".")
    dev = torch.device("cuda")
    props = torch.cuda.get_device_properties(dev)
    info = {"gpu": props.name, "sm": f"{props.major}.{props.minor}", "sms": props.multi_processor_count,
            "torch": torch.__version__, "cuda": torch.version.cuda, "args": vars(args), "scale": SCALE}
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
    pf = None
    try:
        import prefill_attn as pf
    except Exception as e:  # noqa: BLE001
        info["pf_import_error"] = repr(e)[:400]
    info["import_s"] = round(time.time() - T0, 1)
    log(info)

    layers = [x for x in args.layers.split(",") if x in LAYERS]
    shapes = prompt_chunks(args.prompt_len, args.chunk)
    shapes += [s for s in parse_shapes(args.shapes) if s not in shapes]
    broken = {layer: {} for layer in layers}
    checks, rows, stopped = [], [], None

    def dump():
        summary = per_prompt(rows, args, broken) if pf else None
        info["wall_s"] = round(time.time() - T0, 1)
        with open(os.path.join(out_dir, "prefill_attn_bench.json"), "w") as f:
            json.dump({"info": info, "numerics": checks, "rows": rows, "per_prompt": summary,
                       "broken": broken, "stopped": stopped}, f, indent=1)
        return summary

    if pf is not None:
        for layer in layers:
            try:
                checks += numerics_check(ua, pf, layer, args, dev, broken[layer])
            except Exception as e:  # noqa: BLE001
                checks.append({"check": "numerics", "layer": layer, "error": repr(e)[:400]})
                log(checks[-1])
            dump()
    for start, T in shapes:  # interleave layers so a budget stop keeps both layer types for the prompt chunks
        for layer in layers:
            if over_budget(args):
                stopped = {"stopped_early": f"budget {args.budget_s}s reached before {layer} {start}:{T}"}
                break
            try:
                rows += run_case(ua, pf, layer, start, T, args, dev, broken[layer])
            except Exception as e:  # noqa: BLE001
                rows.append({"layer": layer, "start": start, "T": T, "error": repr(e)[:400]})
                log(rows[-1])
                torch.cuda.empty_cache()
            dump()
        if stopped:
            log(stopped)
            break
    summary = dump()
    if summary is not None:
        log({"per_prompt": summary})
    return 0


if __name__ == "__main__":
    sys.exit(main())
