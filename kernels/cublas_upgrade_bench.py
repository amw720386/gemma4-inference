"""Kernel lab: does a newer cuBLAS speed up Gemma 4 12B's BF16 GEMMs on sm_120?

The pinned vLLM v0.33.0 image ships cuBLAS 13.1; NVIDIA's cuBLAS 13.2 release notes claim up to 20 % faster
FP16/BF16 on RTX PRO 6000. This script installs a newer `nvidia-cublas` wheel into /tmp (no change to the image),
then runs kernels/gemm_bench.py twice in fresh processes: stock, and with the new libcublas/libcublasLt preloaded.
Each run records the cuBLAS version actually loaded. Outputs land in $G4_OUT_DIR/{stock,new}/.

Run:  g4bench modal script --path kernels/cublas_upgrade_bench.py --execute [-- <nvidia-cublas version>]
"""

from __future__ import annotations

import glob
import json
import os
import subprocess
import sys

NEW_VERSION = sys.argv[1] if len(sys.argv) > 1 else "13.10.0.20"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("G4_OUT_DIR", ".")
VERSION_PROBE = (
    "import ctypes, torch\n"
    "torch.ones(1, device='cuda') @ torch.ones(1, 1, device='cuda')\n"
    "lib = [l for l in open('/proc/self/maps').read().split() if 'libcublasLt.so' in l]\n"
    "lt = ctypes.CDLL(lib[0]) if lib else None\n"
    "print('cublasLt', lt.cublasLtGetVersion() if lt else None, lib[:1])\n"
)


def run(label: str, env: dict) -> dict:
    out = os.path.join(OUT, label)
    os.makedirs(out, exist_ok=True)
    env = {**env, "G4_OUT_DIR": out, "G4_GEMM_QUICK": "1"}
    probe = subprocess.run([sys.executable, "-c", VERSION_PROBE], env=env, capture_output=True, text=True)
    p = subprocess.run([sys.executable, os.path.join(HERE, "gemm_bench.py")], env=env, capture_output=True, text=True)
    res = {"label": label, "probe": (probe.stdout + probe.stderr).strip()[-600:], "rc": p.returncode,
           "stderr_tail": p.stderr[-1500:]}
    print(json.dumps(res), flush=True)
    return res


def main() -> int:
    base = dict(os.environ)
    results = [run("stock", base)]
    target = "/tmp/newcublas"
    pip = subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--quiet", "--target", target,
                          f"nvidia-cublas=={NEW_VERSION}"], capture_output=True, text=True)
    libs = sorted(glob.glob(f"{target}/nvidia/cu*/lib/libcublas*.so*") + glob.glob(f"{target}/nvidia/cublas/lib/libcublas*.so*"))
    print(json.dumps({"pip_rc": pip.returncode, "pip_err": pip.stderr[-800:], "libs": libs}), flush=True)
    lt = [x for x in libs if "libcublasLt.so" in x]
    blas = [x for x in libs if "libcublas.so" in x]
    if pip.returncode == 0 and lt and blas:
        libdir = os.path.dirname(lt[0])
        env = {**base, "LD_PRELOAD": f"{lt[0]}:{blas[0]}",
               "LD_LIBRARY_PATH": f"{libdir}:{base.get('LD_LIBRARY_PATH', '')}"}
        results.append(run(f"cublas-{NEW_VERSION}", env))
    with open(os.path.join(OUT, "cublas_upgrade.json"), "w") as f:
        json.dump({"new_version": NEW_VERSION, "runs": results}, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
