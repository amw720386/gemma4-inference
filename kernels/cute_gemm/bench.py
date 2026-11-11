"""Kernel lab: CUTLASS CuTe DSL sm_120 dense GEMM vs cuBLAS at Gemma 4 12B's BF16 linear shapes.

Run on Modal inside the pinned vLLM image:  g4bench modal script --path kernels/cute_gemm/bench.py --execute
Writes $G4_OUT_DIR/cute_gemm_bench.json (rewritten after every shape, so a timeout keeps partial results) and prints
JSON lines.

Kernel: kernels/cute_gemm/dense_gemm.py, vendored unmodified (BSD-3 header intact) from NVIDIA/cutlass tag v4.8.0
(commit 098de2a652cf8f00fd70b2df54051c7eccbb855a), examples/python/CuTeDSL/cute/blackwell_geforce/kernel/dense_gemm/
dense_gemm.py. It is a persistent, warp-specialised kernel (8 mma.sync warps in a 4x2 layout + a TMA-load warp, TMA
store epilogue, cluster 1x1x1); it has no pingpong/cooperative switch (only the block-scaled FP4/FP8 examples do), so
the sweep is over CTA tile shapes and A/B operand swap. Its API, used here without the example's own run():
    kern = Sm120GemmKernel(cutlass.Float32, tile_shape_mnk, epi_stage)
    fn = cute.compile(kern, a, b, c, max_active_clusters, stream)      # a: (M,K,L)  b: (N,K,L)  c: (M,N,L)
    fn(a, b, c, stream)                                                # C = A @ B^T per batch l
Operands are cute tensors from DLPack with dynamic layouts, so each config compiles ONCE and serves every shape/M.
  normal  A=x (M,K,1) K-major, B=W (N,K,1) K-major, C=y (M,N,1) N-major     -> y = x @ W^T
  swap    A=W (N,K,1),         B=x (M,K,1),         C=y^T (N,M,1) M-major  -> same y, tile M runs over out-features
Every output is checked against an fp32 F.linear reference (max abs error; output pre-filled with NaN so unwritten
tiles are caught) next to cuBLAS's own error. Timing: CUDA events, hot (one weight copy) and cold (a ring of weight
copies > L2, as kernels/gemm_bench.py). Every config is fail-soft: an error is recorded and that config skipped.
Env: G4_CUTE_BUDGET_S (default 420) stops starting new work after that many seconds.
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.metadata as md
import io
import json
import math
import os
import re
import site
import subprocess
import sys
import threading
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.environ.get("G4_OUT_DIR", ".")
CUTLASS_PIN = "4.8.0"
CUTLASS_TARGET = "/tmp/cutlass_dsl"

# (name, N=out_features, K=in_features, count per forward pass of the 48-layer target) -- as kernels/gemm_bench.py
SHAPES = [
    ("sliding.qkv", 8192, 3840, 40),
    ("sliding.o", 3840, 4096, 40),
    ("global.qkv", 8704, 3840, 8),
    ("global.o", 3840, 8192, 8),
    ("mlp.gate_up", 30720, 3840, 48),
    ("mlp.down", 3840, 15360, 48),
]
M_LIST = [3072, 1024, 3100, 28]  # headline first; 3100 = not a tile multiple; 28 = one decode (spec-verify) size
PREFILL_M = 3072
# cuBLAS TFLOPS at M=3072 from the earlier gemm_bench run (for cross-run comparison only)
PRIOR_CUBLAS_TFLOPS = {"sliding.qkv": 341, "sliding.o": 408, "global.qkv": 387, "global.o": 427, "mlp.gate_up": 395,
                       "mlp.down": 411}
# (label, CTA tile (M,N,K), swap A/B). <= 6 configs = 6 JIT compiles total. ab_stage from the example's smem heuristic
# at ~99 KB smem, epi_stage 4: 128x128x64 -> 2, 256x128x32 -> 3, 128x256x64 -> 1, 64x128x64 -> 3, 128x64x64 -> 3.
CONFIGS = [
    ("t128x128x64", (128, 128, 64), False),
    ("t256x128x32", (256, 128, 32), False),
    ("t64x128x64", (64, 128, 64), False),
    ("t128x256x64", (128, 256, 64), False),
    ("swap.t128x128x64", (128, 128, 64), True),
    ("swap.t128x64x64", (128, 64, 64), True),  # small-M friendly: tokens on the 64-wide N tile
]
EPI_STAGE = 4
BUDGET_S = float(os.environ.get("G4_CUTE_BUDGET_S", "420"))
STEP_TIMEOUT_S = 180.0  # watchdog: a single compile/launch stuck this long (hung kernel) -> dump results and exit
COLD_BYTES = 768 * 1024**2  # > 5x the 128 MB L2
PIP_TIMEOUT_S = 180

T0 = time.time()
STATE = {"stage": "start", "deadline": float("inf")}
RESULT: dict = {"info": {}, "compiles": [], "rows": [], "summary": {}}


def emit(obj) -> None:
    print(json.dumps(obj), flush=True)


def dump() -> None:
    try:
        with open(os.path.join(OUT_DIR, "cute_gemm_bench.json"), "w") as f:
            json.dump(RESULT, f, indent=1)
    except Exception as e:  # noqa: BLE001
        print(f"dump failed: {e!r}", file=sys.stderr, flush=True)


def watchdog() -> None:
    while True:
        time.sleep(5)
        if time.time() > STATE["deadline"]:
            RESULT["info"]["watchdog"] = f"step '{STATE['stage']}' exceeded {STEP_TIMEOUT_S}s; aborted"
            emit({"watchdog": RESULT["info"]["watchdog"]})
            dump()
            os._exit(3)


def step(stage: str, timeout_s: float = STEP_TIMEOUT_S) -> None:
    STATE["stage"], STATE["deadline"] = stage, time.time() + timeout_s


def over_budget() -> bool:
    return time.time() - T0 > BUDGET_S


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


def cutlass_dists() -> dict:
    out = {}
    for d in md.distributions():
        name = d.metadata.get("Name") or ""
        if "cutlass" in name.lower():
            out[name] = d.version
    return out


def ensure_cutlass(info: dict):
    """Import the CuTe DSL from the image; if absent, pip-install the pinned wheel into /tmp (image untouched)."""
    info["cutlass_dists_image"] = cutlass_dists()
    try:
        import cutlass  # noqa: F401
        info["cutlass_source"] = "image"
    except Exception as e:  # noqa: BLE001
        info["cutlass_image_import_error"] = repr(e)[:300]
        info["pip"] = []
        for spec in (f"nvidia-cutlass-dsl[cu13]=={CUTLASS_PIN}", f"nvidia-cutlass-dsl=={CUTLASS_PIN}"):
            p = subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--target", CUTLASS_TARGET, spec],
                               capture_output=True, text=True, timeout=PIP_TIMEOUT_S)
            info["pip"].append({"spec": spec, "rc": p.returncode, "stderr_tail": p.stderr[-600:]})
            if p.returncode == 0:
                break
        site.addsitedir(CUTLASS_TARGET)  # appended (image packages keep priority); also processes the wheel's .pth
        importlib.invalidate_caches()
        for mod in [m for m in sys.modules if m == "cutlass" or m.startswith("cutlass.")]:
            del sys.modules[mod]
        import cutlass  # noqa: F401
        info["cutlass_source"] = CUTLASS_TARGET
    import cutlass

    info["cutlass_file"] = getattr(cutlass, "__file__", None)
    info["cutlass_version_attr"] = str(getattr(cutlass, "__version__", None))
    info["cutlass_dists"] = cutlass_dists()
    vers = [v for k, v in info["cutlass_dists"].items() if k.lower().replace("_", "-") == "nvidia-cutlass-dsl"]
    info["cutlass_dsl_version"] = vers[0] if vers else None
    info["cutlass_matches_vendored_example"] = (vers[0] == CUTLASS_PIN) if vers else None
    info["CUTE_DSL_ARCH"] = os.environ.get("CUTE_DSL_ARCH", "not set (auto-detect)")
    return cutlass


def make_ct(t: torch.Tensor, leading_dim: int):
    from cutlass.cute.runtime import from_dlpack

    return from_dlpack(t, assumed_align=16).mark_layout_dynamic(leading_dim=leading_dim)


def operands(x: torch.Tensor, w: torch.Tensor, y: torch.Tensor, swap: bool):
    """Explicit-stride (.,.,L=1) views matching cutlass.torch.matrix's layouts; no copies."""
    m, k = x.shape
    n = w.shape[0]
    xa = x.as_strided((m, k, 1), (k, 1, m * k))
    wb = w.as_strided((n, k, 1), (k, 1, n * k))
    if not swap:
        return make_ct(xa, 1), make_ct(wb, 1), make_ct(y.as_strided((m, n, 1), (n, 1, m * n)), 1)
    return make_ct(wb, 1), make_ct(xa, 1), make_ct(y.as_strided((n, m, 1), (1, n, m * n)), 0)


def short_err(e: BaseException) -> str:
    return (type(e).__name__ + ": " + str(e)).replace("\n", " | ")[:600]


def compile_configs(cutlass, dg, stream, dev) -> dict:
    import cutlass.cute as cute

    compiled = {}
    try:
        max_active = cutlass.utils.HardwareInfo().get_max_active_clusters(1)
    except Exception as e:  # noqa: BLE001
        RESULT["info"]["max_active_clusters_error"] = short_err(e)
        max_active = torch.cuda.get_device_properties(dev).multi_processor_count
    RESULT["info"]["max_active_clusters"] = int(max_active)
    # Sample operands for tracing only (layouts are dynamic, so the shape here does not specialise the kernel).
    xs = torch.randn(1024, 3840, device=dev, dtype=torch.bfloat16)
    wsamp = torch.randn(8192, 3840, device=dev, dtype=torch.bfloat16)
    ys = torch.empty(1024, 8192, device=dev, dtype=torch.bfloat16)
    for label, tile, swap in CONFIGS:
        rec = {"config": label, "tile_mnk": list(tile), "swap_ab": swap, "epi_stage": EPI_STAGE}
        if over_budget():
            rec["skipped"] = "budget"
        else:
            step(f"compile {label}")
            buf = io.StringIO()
            t0 = time.time()
            try:
                kern = dg.Sm120GemmKernel(cutlass.Float32, tile, EPI_STAGE)
                a, b, c = operands(xs, wsamp, ys, swap)
                with contextlib.redirect_stdout(buf):  # the example prints its stage counts
                    fn = cute.compile(kern, a, b, c, max_active, stream)
                compiled[label] = (fn, swap)
                rec["ok"] = True
            except (Exception, SystemExit) as e:  # noqa: BLE001  (the example sys.exit(0)s when smem is too small)
                rec["ok"] = False
                rec["error"] = short_err(e)
            rec["compile_s"] = round(time.time() - t0, 2)
            printed = buf.getvalue()
            mt = re.search(r"ab_stage=(\d+), epi_stage=(\d+)", printed)
            if mt:
                rec["ab_stage"], rec["epi_stage_used"] = int(mt.group(1)), int(mt.group(2))
            if printed and not mt:
                rec["printed"] = printed[-300:]
        RESULT["compiles"].append(rec)
        emit({"compile": rec})
        dump()
    del xs, wsamp, ys
    return compiled


def max_abs_err(y: torch.Tensor, ref32: torch.Tensor) -> tuple[float, int]:
    nan = int(torch.isnan(y).sum())
    return float((y.float() - ref32).abs().nan_to_num(nan=float("inf")).max()), nan


def run_shape(name, n, k, count, compiled, cstream, dev) -> None:
    w = torch.randn(n, k, device=dev, dtype=torch.bfloat16) * 0.02
    copies = max(2, -(-COLD_BYTES // (2 * n * k)))
    ws = [w.clone() for _ in range(copies)]
    w32 = w.float()
    for m in M_LIST:
        if over_budget():
            emit({"skipped": f"{name} M={m}", "reason": "budget"})
            continue
        step(f"{name} M={m} cublas")
        x = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
        ref = F.linear(x.float(), w32)
        ref_max = float(ref.abs().max())
        err_cub, _ = max_abs_err(F.linear(x, w), ref)
        iters = 200 if m <= 448 else 30
        t_hot = bench(lambda: F.linear(x, w), iters)
        ring = iter(range(10**9))
        t_cold = bench(lambda: F.linear(x, ws[next(ring) % copies]), iters)
        flops = 2 * m * n * k
        row = {"shape": name, "M": m, "N": n, "K": k, "count": count, "cublas_ms": round(t_cold, 5),
               "cublas_hot_ms": round(t_hot, 5), "cublas_tflops": round(flops / t_cold / 1e9, 1),
               "cublas_max_abs_err": err_cub, "ref_max_abs": round(ref_max, 3), "variants": {}}
        if m == PREFILL_M:
            row["prior_cublas_tflops"] = PRIOR_CUBLAS_TFLOPS.get(name)
        tol = 4 * err_cub + 1e-2  # rounding-order differences are ~1 BF16 ulp of the largest output
        for label, (fn, swap) in compiled.items():
            v: dict = {}
            if over_budget():
                v["skipped"] = "budget"
                row["variants"][label] = v
                continue
            step(f"{name} M={m} {label}")
            try:
                y = torch.full((m, n), float("nan"), device=dev, dtype=torch.bfloat16)
                a, b, c = operands(x, w, y, swap)
                fn(a, b, c, cstream)
                torch.cuda.synchronize()
                err, nans = max_abs_err(y, ref)
                v.update(max_abs_err=err, nan_count=nans, ok=bool(nans == 0 and err <= tol))
                hot_ops = (a, b, c)
                t_h = bench(lambda: fn(*hot_ops, cstream), iters)
                ring_ops = [operands(x, wi, y, swap) for wi in ws]
                ring = iter(range(10**9))
                t_c = bench(lambda: fn(*ring_ops[next(ring) % copies], cstream), iters)
                v.update(ms=round(t_c, 5), hot_ms=round(t_h, 5), tflops=round(flops / t_c / 1e9, 1),
                         speedup=round(t_cold / t_c, 3), speedup_hot=round(t_hot / t_h, 3))
                del y, ring_ops
            except (Exception, SystemExit) as e:  # noqa: BLE001
                v["error"] = short_err(e)
                v["ok"] = False
                try:
                    torch.cuda.synchronize()
                except Exception as e2:  # noqa: BLE001  (sticky CUDA error: later configs will fail too)
                    v["sync_error"] = short_err(e2)
            row["variants"][label] = v
        good = {lb: v for lb, v in row["variants"].items() if v.get("ok") and "ms" in v}
        if good:
            best = min(good, key=lambda lb: good[lb]["ms"])
            row.update(best=best, best_ms=good[best]["ms"], best_hot_ms=good[best]["hot_ms"],
                       best_tflops=good[best]["tflops"], best_speedup=good[best]["speedup"],
                       best_speedup_hot=good[best]["speedup_hot"])
        RESULT["rows"].append(row)
        emit(row)
        del x, ref
    del w, ws, w32
    torch.cuda.empty_cache()


def summarize() -> None:
    rows = RESULT["rows"]
    per_shape = []
    for r in rows:
        per_shape.append({k: r.get(k) for k in ("shape", "M", "cublas_tflops", "best", "best_tflops", "best_speedup",
                                                "best_speedup_hot")})
    fwd = {}
    pre = [r for r in rows if r["M"] == PREFILL_M]
    if pre:
        cub = sum(r["cublas_ms"] * r["count"] for r in pre)
        cub_hot = sum(r["cublas_hot_ms"] * r["count"] for r in pre)
        cute_ms = sum((r.get("best_ms") or r["cublas_ms"]) * r["count"] for r in pre)
        cute_hot = sum((r.get("best_hot_ms") or r["cublas_hot_ms"]) * r["count"] for r in pre)
        mixed = sum(min(r.get("best_ms") or math.inf, r["cublas_ms"]) * r["count"] for r in pre)
        fwd = {"M": PREFILL_M, "shapes_measured": len(pre),
               "shapes_without_valid_cute": [r["shape"] for r in pre if "best_ms" not in r],
               "cublas_ms": round(cub, 3), "best_cute_ms": round(cute_ms, 3),
               "best_of_both_ms": round(mixed, 3), "speedup_cute_vs_cublas": round(cub / cute_ms, 3),
               "cublas_hot_ms": round(cub_hot, 3), "best_cute_hot_ms": round(cute_hot, 3),
               "note": "cold-weight times x per-forward counts; a shape with no valid CuTe config counts at cuBLAS time"}
    RESULT["summary"] = {"per_shape": per_shape, "prefill_linear_per_forward": fwd,
                         "elapsed_s": round(time.time() - T0, 1)}
    emit({"summary": RESULT["summary"]})


def main() -> int:
    threading.Thread(target=watchdog, daemon=True).start()
    os.makedirs(OUT_DIR, exist_ok=True)
    dev = torch.device("cuda")
    props = torch.cuda.get_device_properties(dev)
    info = RESULT["info"]
    info.update(gpu=props.name, sm=f"{props.major}.{props.minor}", sms=props.multi_processor_count,
                torch=torch.__version__, cuda=torch.version.cuda, configs=[c[0] for c in CONFIGS], m_list=M_LIST,
                budget_s=BUDGET_S, vendored_example="NVIDIA/cutlass v4.8.0 098de2a6 "
                "examples/python/CuTeDSL/cute/blackwell_geforce/kernel/dense_gemm/dense_gemm.py")
    try:
        import triton
        info["triton"] = triton.__version__
    except Exception:  # noqa: BLE001
        pass
    torch.backends.cuda.matmul.allow_tf32 = False  # fp32 reference must be true fp32
    torch.manual_seed(0)

    compiled: dict = {}
    cstream = None
    step("import / pip-install cutlass", 2 * PIP_TIMEOUT_S + 60)
    try:
        cutlass = ensure_cutlass(info)
        emit({k: v for k, v in info.items() if k.startswith(("cutlass", "pip", "CUTE"))})
        sys.path.insert(0, HERE)
        import dense_gemm as dg
        import cutlass.torch as cutlass_torch

        cstream = cutlass_torch.current_stream()
        info["import_s"] = round(time.time() - T0, 1)
        compiled = compile_configs(cutlass, dg, cstream, dev)
    except (Exception, SystemExit) as e:  # noqa: BLE001
        info["cutlass_setup_error"] = short_err(e)
    emit({"info": info})
    dump()

    for name, n, k, count in SHAPES:
        try:
            run_shape(name, n, k, count, compiled, cstream, dev)
        except Exception as e:  # noqa: BLE001
            err = {"shape": name, "error": short_err(e)}
            RESULT.setdefault("shape_errors", []).append(err)
            emit(err)
        dump()
    STATE["deadline"] = float("inf")
    summarize()
    dump()
    return 0


if __name__ == "__main__":
    sys.exit(main())
