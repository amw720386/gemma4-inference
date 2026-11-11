"""Kernel lab: BF16 GEMM efficiency on the serving GPU at Gemma 4 12B's exact linear-layer shapes.

Run on Modal inside the pinned vLLM image:  g4bench modal script --path kernels/gemm_bench.py --execute
Writes $G4_OUT_DIR/gemm_bench.json and prints a table. Backends:
  cublas    torch.nn.functional.linear (what vLLM uses for unquantized BF16 linears)
  autotune  torch.compile(mode="max-autotune-no-cudagraphs") of the same linear (Inductor Triton templates vs aten)
All outputs are checked against cublas (fp32-accumulated BF16; max abs error reported).
"""

from __future__ import annotations

import json
import os
import sys
import time

import torch
import torch.nn.functional as F

# (name, N=out_features, K=in_features, count per forward pass of the 48-layer target)
SHAPES = [
    ("sliding.qkv", 8192, 3840, 40),   # q 16x256 + k 8x256 + v 8x256
    ("sliding.o", 3840, 4096, 40),
    ("global.qkv", 8704, 3840, 8),     # q 16x512 + k 1x512 (attention_k_eq_v: no separate v)
    ("global.o", 3840, 8192, 8),
    ("mlp.gate_up", 30720, 3840, 48),
    ("mlp.down", 3840, 15360, 48),
]
DECODE_M = [4, 7, 14, 28, 56, 112, 448]   # spec-decode verify: requests x (depth+1)
PREFILL_M = [1024, 3072]
AUTOTUNE_M: list[int] = []  # run 1 (20261111T170723Z): max-autotune never beat cuBLAS (0.33-1.01x); skipped
COLD_BYTES = 768 * 1024**2  # > 5x the 128 MB L2


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


def main() -> int:
    out_dir = os.environ.get("G4_OUT_DIR", ".")
    dev = torch.device("cuda")
    props = torch.cuda.get_device_properties(dev)
    info = {"gpu": props.name, "sm": f"{props.major}.{props.minor}", "sms": props.multi_processor_count,
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "cublaslt_preferred": str(torch.backends.cuda.preferred_blas_library())}
    try:
        import triton
        info["triton"] = triton.__version__
    except Exception:  # noqa: BLE001
        pass
    print(json.dumps(info), flush=True)
    torch.manual_seed(0)
    rows = []

    # Peak dense BF16 throughput (large square GEMM) and copy bandwidth, for reference.
    a = torch.randn(8192, 8192, device=dev, dtype=torch.bfloat16)
    b = torch.randn(8192, 8192, device=dev, dtype=torch.bfloat16)
    t = bench(lambda: a @ b, 20)
    info["peak_bf16_tflops_measured"] = round(2 * 8192**3 / t / 1e9, 1)
    buf = torch.empty(512 * 1024**2, device=dev, dtype=torch.uint8)
    dst = torch.empty_like(buf)
    t = bench(lambda: dst.copy_(buf), 20)
    info["copy_bandwidth_gbs"] = round(2 * buf.numel() / t / 1e6, 1)
    del a, b, buf, dst
    print(json.dumps({k: info[k] for k in ("peak_bf16_tflops_measured", "copy_bandwidth_gbs")}), flush=True)

    for name, n, k, count in SHAPES:
        w = torch.randn(n, k, device=dev, dtype=torch.bfloat16) * 0.02
        # Cold-weight variant: cycle through enough copies to exceed the 128 MB L2, as real decode does (each layer's
        # weights are read once per step from DRAM). The hot variant (one copy) is what the first run measured.
        copies = max(2, -(-COLD_BYTES // (2 * n * k)))
        ws = [w.clone() for _ in range(copies)]
        compiled = None
        for m in DECODE_M + PREFILL_M:
            x = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
            iters = 200 if m <= 448 else 50
            t_cub = bench(lambda: F.linear(x, w), iters)
            ring = iter(range(10**9))
            t_cold = bench(lambda: F.linear(x, ws[next(ring) % copies]), iters)
            flops = 2 * m * n * k
            bytes_ = 2 * (n * k + m * k + m * n)
            row = {"shape": name, "M": m, "N": n, "K": k, "count": count, "cublas_ms": round(t_cold, 5),
                   "cublas_hot_ms": round(t_cub, 5), "cold_gbs": round(bytes_ / t_cold / 1e6, 1),
                   "cublas_tflops": round(flops / t_cold / 1e9, 1),
                   "hot_gbs": round(bytes_ / t_cub / 1e6, 1)}
            if m in AUTOTUNE_M:
                try:
                    if compiled is None:
                        compiled = torch.compile(lambda xx: F.linear(xx, w), mode="max-autotune-no-cudagraphs",
                                                 dynamic=False)
                    t0 = time.time()
                    ref = F.linear(x, w)
                    got = compiled(x)
                    row["autotune_compile_s"] = round(time.time() - t0, 1)
                    row["autotune_max_abs_err"] = float((got.float() - ref.float()).abs().max())
                    t_at = bench(lambda: compiled(x), iters)
                    row["autotune_ms"] = round(t_at, 5)
                    row["autotune_speedup_hot"] = round(t_cub / t_at, 3)
                except Exception as e:  # noqa: BLE001
                    row["autotune_error"] = repr(e)[:300]
            rows.append(row)
            print(json.dumps(row), flush=True)
        del w, ws
        torch.cuda.empty_cache()

    # Per-forward-pass linear time at decode batch sizes (all 48 layers), vs a weight-read roofline.
    weight_bytes = sum(2 * n * k * c for _, n, k, c in SHAPES)
    summary = []
    for m in DECODE_M + PREFILL_M:
        tot = sum(r["cublas_ms"] * r["count"] for r in rows if r["M"] == m)
        summary.append({"M": m, "linear_ms_per_forward": round(tot, 3),
                        "weight_read_floor_ms": round(weight_bytes / (info["copy_bandwidth_gbs"] * 1e6), 3)})
    info["per_forward"] = summary
    print(json.dumps({"per_forward": summary}), flush=True)
    with open(os.path.join(out_dir, "gemm_bench.json"), "w") as f:
        json.dump({"info": info, "rows": rows}, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
