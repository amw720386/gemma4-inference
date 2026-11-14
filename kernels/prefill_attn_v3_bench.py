"""Kernel lab: paged prefill attention v3 (``_g4_pfa_gl3_fwd`` in src/g4bench/patches/prefill_attn_g4.py) vs the paged
v2 kernel that serves today (``_g4_pfa_gl_fwd``), on vLLM's paged KV-cache layout, at the TTFT-critical shape.

Run on Modal inside the pinned vLLM image:
  g4bench modal script --path kernels/prefill_attn_v3_bench.py --execute [-- ARGS]
Writes $G4_OUT_DIR/prefill_attn_v3_bench.json (rewritten after every row) and prints one JSON line per row.

Unlike kernels/prefill_attn_v2_bench.py (contiguous K/V lab kernels), this times the exact serving launch
(``prefill_gluon``: work table, per-row block-table gather, the serving block sizes 64 / 16) on one fresh prompt
(start 0, T tokens; 8K x1 random prompts average ~7.4K tokens) with shuffled cache blocks:
  global  head_dim 512, 16 q : 1 KV head, full causal          (x8 per forward), block size 64
  sliding head_dim 256, 16 q : 8 KV heads, window 1024 causal  (x40 per forward), block size 16
Numerics (mandatory, fp32 reference): every variant first runs the serving load-time check
(``v2_self_check``: small mixed paged batch, atol 3e-2 + rtol 2e-2 vs an fp32 torch reference); at every timed shape
its output is compared with an fp32 reference on three 48-token query slices (first, middle, last) and with the v2
winner's output on all rows. A failure marks the variant broken (not timed again).
Timing: CUDA graph of 4 back-to-back launches, CUDA-event timed, >= --min-ms per measurement; TFLOPS counts only
causal(/window)-valid (query, key) pairs: 4 x heads x head_dim x pairs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch

T0 = time.time()
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from g4bench.patches import prefill_attn_g4 as P  # noqa: E402

SCALE = 1.0  # Gemma 4: self.scaling = 1.0
LAYERS = {
    "global": {"heads": 16, "kv_heads": 1, "head_dim": 512, "window": 0, "count": 8, "bs": 64},
    "sliding": {"heads": 16, "kv_heads": 8, "head_dim": 256, "window": 1024, "count": 40, "bs": 16},
}
TOL = 3e-2
EXTRA_V2 = {  # ad-hoc v2 kernels (not in the serving tables): K/V share one buffer -> 2 CTAs / SM
    "global": [P.V2Variant("v2:gl_m32_n16_w2x2_kvs", 32, 16, 64, 2, 2, True, 8, 2),
               P.V2Variant("v2:gl_m64_n32_w4x2_kvs", 64, 32, 64, 4, 2, True, 8, 2),     # 100K, 8 warps, 64 FLOP/B
               P.V2Variant("v2:gl_m64_n32_w4x4_kvs", 64, 32, 128, 4, 4, True, 8, 2),    # 16 warps
               P.V2Variant("v2:gl_m64_n32_w4x2_kvs_d128", 64, 32, 128, 4, 2, True, 8, 2)],
    "sliding": [P.V2Variant("v2:gl_m64_n32_w4x1_kvs", 64, 32, 64, 4, 1, True, 8, 2),
                P.V2Variant("v2:gl_m64_n16_w4x1_kvs", 64, 16, 64, 4, 1, True, 8, 2),
                P.V2Variant("v2:gl_m128_n32_w8x1_kvs", 128, 32, 64, 8, 1, True, 8, 2),   # 80K
                P.V2Variant("v2:gl_m64_n64_w4x1_kvs", 64, 64, 64, 4, 1, True, 8, 2),     # 64K
                P.V2Variant("v2:gl_m32_n32_w2x1_kvs", 32, 32, 64, 2, 1, True, 8, 2),     # 32K: 3 CTAs / SM
                P.V2Variant("v2:gl_m32_n64_w2x1_kvs", 32, 64, 64, 2, 1, True, 8, 2),     # 48K: 2 CTAs / SM
                P.V2Variant("v2:gl_m64_n32_w4x1_kvs_d128", 64, 32, 128, 4, 1, True, 8, 2),
                P.V2Variant("v2:gl_m64_n32_w4x1_kvs_kw8", 64, 32, 64, 4, 1, True, 8, 8)],
}


def log(obj) -> None:
    print(json.dumps(obj), flush=True)


def valid_pairs(start: int, T: int, window: int) -> int:
    if window <= 0:
        return T * start + T * (T + 1) // 2
    return sum(min(p + 1, window) for p in range(start, start + T))


def time_calls(f, min_ms: float) -> tuple[float, str]:
    calls = [f] * 4
    for c in calls:
        c()
    torch.cuda.synchronize()
    mode = "graph"
    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for c in calls[:2]:
                c()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for c in calls:
                c()
        run = g.replay
    except Exception:  # noqa: BLE001
        mode = "eager"
        torch.cuda.synchronize()

        def run():
            for c in calls:
                c()
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
    return st.elapsed_time(en) * 1000.0 / (reps * len(calls)), mode


class Case:
    """One fresh prompt of T tokens (start 0) in a paged cache with shuffled blocks, vLLM layout
    ([num_blocks, block_size, num_kv, D] views of a [num_blocks, num_kv, block_size, 2D] tensor)."""

    def __init__(self, L, T, dev, seed=0):
        H, KVH, D, bs = L["heads"], L["kv_heads"], L["head_dim"], L["bs"]
        g = torch.Generator(device="cpu").manual_seed(seed)
        self.T, self.S, self.window = T, T, L["window"]
        nb = -(-T // bs)
        num_blocks = nb + 5
        kv = torch.randn(num_blocks, KVH, bs, 2 * D, generator=g).to(torch.bfloat16).to(dev)
        self.key_cache, self.value_cache = kv.transpose(1, 2).split(D, dim=-1)
        perm = torch.randperm(num_blocks, generator=g)[:nb].int()
        self.bt = torch.zeros(1, nb + 2, dtype=torch.int32)
        self.bt[0, :nb] = perm
        self.bt = self.bt.to(dev)
        self.q = (torch.randn(T, H, D, generator=g) * (2.0 / math.sqrt(D))).to(torch.bfloat16).to(dev)
        self.qsl = torch.tensor([0, T], dtype=torch.int32, device=dev)
        self.seq = torch.tensor([T], dtype=torch.int32, device=dev)
        self.plan = P.plan_from_query_start_loc([0, T], 1, T, T)
        self.k = P.gather_kv(self.key_cache, self.bt[0], T)
        self.v = P.gather_kv(self.value_cache, self.bt[0], T)

    def run(self, v, out):
        P.prefill_gluon(self.plan, self.q, self.key_cache, self.value_cache, out, self.bt, self.qsl, self.seq, SCALE,
                        self.window, v)

    def reference(self, t0: int, n: int) -> torch.Tensor:
        """fp32 reference for query tokens [t0, t0 + n)."""
        H = self.q.shape[1]
        S = t0 + n
        G = H // self.k.shape[1]
        kb = self.k[:S].float().repeat_interleave(G, dim=1)
        vb = self.v[:S].float().repeat_interleave(G, dim=1)
        s = torch.einsum("thd,shd->hts", self.q[t0:S].float(), kb) * SCALE
        qpos = torch.arange(t0, S, device=s.device)
        kpos = torch.arange(S, device=s.device)
        m = kpos[None, :] <= qpos[:, None]
        if self.window > 0:
            m &= (qpos[:, None] - kpos[None, :]) < self.window
        s = s.masked_fill(~m[None], float("-inf"))
        return torch.einsum("hts,shd->thd", torch.softmax(s, dim=-1), vb)


def variants(layer: str, names: str) -> list:
    sliding = layer == "sliding"
    pool = list((P.V2_SLIDING_VARIANTS if sliding else P.V2_GLOBAL_VARIANTS).values())
    pool += EXTRA_V2[layer]
    pool += list((P.V3_SLIDING_VARIANTS if sliding else P.V3_GLOBAL_VARIANTS).values())
    seen = set()
    pool = [v for v in pool if not (v.name in seen or seen.add(v.name))]
    if names:
        want = set(names.split(","))
        pool = [v for v in pool if v.name in want or v.name.split(":", 1)[1] in want]
    return pool


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="global,sliding")
    ap.add_argument("--tokens", default="7424,8192", help="prompt lengths (one fresh chunk each)")
    ap.add_argument("--variants", default="", help="comma list (with or without the v2:/v3: prefix); default all")
    ap.add_argument("--min-ms", type=float, default=40.0)
    ap.add_argument("--budget-s", type=float, default=420.0)
    args = ap.parse_args(sys.argv[1:] if argv is None else argv)
    out_dir = os.environ.get("G4_OUT_DIR", ".")
    dev = torch.device("cuda")
    props = torch.cuda.get_device_properties(dev)
    import triton
    info = {"gpu": props.name, "sm": f"{props.major}.{props.minor}", "sms": props.multi_processor_count,
            "torch": torch.__version__, "triton": triton.__version__, "gluon_error": P.GLUON_ERROR,
            "args": vars(args)}
    log(info)
    rows, checks, broken = [], [], {}

    def dump():
        info["wall_s"] = round(time.time() - T0, 1)
        best = {}
        for r in rows:
            if "us" in r and r["impl"] not in broken.get(r["layer"], {}):
                k = (r["layer"], r["T"])
                if k not in best or r["us"] < best[k]["us"]:
                    best[k] = r
        summ = {}
        for (layer, T), r in best.items():
            base = next((x for x in rows if x["layer"] == layer and x["T"] == T and x["impl"] == P.V2_FALLBACK[layer]
                         and "us" in x), None)
            summ[f"{layer}:{T}"] = {"best": r["impl"], "us": r["us"], "tflops": r["tflops"],
                                    "v2_serving_us": base and base["us"],
                                    "speedup_vs_v2": base and round(base["us"] / r["us"], 3),
                                    "ms_per_prompt_saved": base and round(
                                        (base["us"] - r["us"]) * LAYERS[layer]["count"] / 1e3, 2)}
        with open(os.path.join(out_dir, "prefill_attn_v3_bench.json"), "w") as f:
            json.dump({"info": info, "checks": checks, "rows": rows, "broken": broken, "summary": summ}, f, indent=1)
        return summ

    layers = [x for x in args.layers.split(",") if x in LAYERS]
    for layer in layers:  # serving load-time check first (compiles every variant once)
        L = LAYERS[layer]
        broken[layer] = {}
        for v in variants(layer, args.variants):
            row = {"check": "load_time", "layer": layer, "impl": v.name}
            if not P.variant_fits(v, L["head_dim"], L["heads"] // L["kv_heads"]):
                row["error"] = "does not fit"
            else:
                t = time.time()
                try:
                    err = P.v2_self_check(v, L["heads"], L["kv_heads"], L["head_dim"], L["window"], L["bs"],
                                          torch.bfloat16, dev, SCALE)
                except Exception as e:  # noqa: BLE001
                    err = f"{type(e).__name__}: {e}"[-3000:]
                row["ok"] = err is None
                row["s"] = round(time.time() - t, 1)
                if err:
                    row["error"] = err
            if "error" in row:
                broken[layer][v.name] = row["error"][:300]
            checks.append(row)
            log(row)
        dump()

    for T in [int(x) for x in args.tokens.split(",") if x]:
        for layer in layers:
            L = LAYERS[layer]
            c = Case(L, T, dev)
            pairs = valid_pairs(0, T, L["window"])
            flops = 4 * L["heads"] * L["head_dim"] * pairs
            slices = [(0, 48), (T // 2, 48), (T - 48, 48)]
            refs = [c.reference(a, n) for a, n in slices]
            base_out = None
            vs = variants(layer, args.variants)
            vs.sort(key=lambda v: v.name != P.V2_FALLBACK[layer])  # serving v2 winner first (comparison base)
            for v in vs:
                if v.name in broken[layer]:
                    continue
                if time.time() - T0 > args.budget_s:
                    log({"stopped": f"budget at {layer} T={T} {v.name}"})
                    dump()
                    return 0
                row = {"layer": layer, "T": T, "impl": v.name, "valid_pairs": pairs}
                try:
                    out = torch.full_like(c.q, float("nan"))
                    c.run(v, out)
                    torch.cuda.synchronize()
                    errs = [float((out[a:a + n].float() - r).abs().max()) for (a, n), r in zip(slices, refs)]
                    row["max_abs_vs_ref"] = max(errs)
                    row["finite"] = bool(torch.isfinite(out).all())
                    if base_out is None and v.name == P.V2_FALLBACK[layer]:
                        base_out = out.clone()
                    elif base_out is not None:
                        row["max_abs_vs_v2"] = float((out.float() - base_out.float()).abs().max())
                    ok = row["finite"] and row["max_abs_vs_ref"] < TOL and row.get("max_abs_vs_v2", 0.0) < TOL
                    row["ok"] = ok
                    if not ok:
                        broken[layer][v.name] = f"numerics T={T}: {row}"[:300]
                    else:
                        o2 = torch.empty_like(c.q)
                        us, mode = time_calls(lambda v=v, o2=o2: c.run(v, o2), args.min_ms)
                        row.update({"us": round(us, 1), "timing": mode, "tflops": round(flops / (us * 1e6), 1),
                                    "ms_per_prompt": round(us * L["count"] / 1e3, 2)})
                except Exception as e:  # noqa: BLE001
                    row["error"] = f"{type(e).__name__}: {e}"[:600]
                    broken[layer][v.name] = row["error"][:300]
                row["elapsed_s"] = round(time.time() - T0, 1)
                rows.append(row)
                log(row)
                dump()
            del c, refs, base_out
            torch.cuda.empty_cache()
    log({"summary": dump()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
