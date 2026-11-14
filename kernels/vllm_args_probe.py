"""Which TTFT-relevant serving options does vLLM 0.33.0 expose? (no model load; prints JSON)

Checks the CLI for frontend scale-out, streaming, scheduling and chunked-prefill knobs, and whether the v1
request-timing histograms (queue / prefill / TTFT) are registered, so TTFT can be split server-side.
"""
import json
import pathlib
import re
import subprocess

import vllm

help_text = subprocess.run(["vllm", "serve", "--help=all"], capture_output=True, text=True, timeout=300).stdout
flags = sorted(set(re.findall(r"--[a-z0-9][a-z0-9-]+", help_text)))
want = ("api-server", "stream", "schedul", "prefill", "chunk", "batched", "num-seqs", "async", "priority",
        "cuda-graph", "cudagraph", "compilation", "frontend", "detok", "tokeniz", "log-stats", "disable-log")
root = pathlib.Path(vllm.__file__).parent
src = (root / "v1" / "metrics" / "loggers.py").read_text()
print(json.dumps({
    "vllm": vllm.__version__,
    "flags": [f for f in flags if any(w in f for w in want)],
    "timing_histograms": sorted(set(re.findall(r'"(vllm:[a-z_]*(?:time|latency)[a-z_]*)"', src))),
}, indent=1))
