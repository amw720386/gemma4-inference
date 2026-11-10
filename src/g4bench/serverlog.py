"""Extract the *effective* engine configuration from a server log.

The article states only a few flags; most performance-relevant settings are chosen by the engine at
startup (attention backend, KV-cache capacity, CUDA-graph sizes, the effective speculative algorithm).
These must be recorded per run to explain gaps against the article. Patterns are taken verbatim from the
pinned sources (vLLM v0.33.0, SGLang v0.5.23); every field is ``None`` when its line is absent, never guessed.

vLLM sources: model_executor/models/config.py (Gemma4Config head-dim backend choice), platforms/cuda.py
("Using %s attention backend out of potential backends"), v1/core/kv_cache_utils.py ("%s KV cache size: %s
tokens, Maximum concurrency for %s tokens per request: %.2fx"), v1/worker/gpu_model_runner.py ("Graph capturing
finished in %.0f secs, took %.2f GiB").
SGLang sources: entrypoints/engine.py (``server_args=...``), managers/scheduler.py (``max_total_num_tokens=``,
``max_running_requests=``).
"""

from __future__ import annotations

import re
from typing import Any

_V_GEMMA = re.compile(r"Gemma4 model has heterogeneous head dimensions .*?(Using FA4 for all layers|"
                      r"per-layer FA3/FA4 selection|forcing TRITON_ATTN backend|Leaving the attention backend)")
_V_BACKEND = re.compile(r"Using (\S+) attention backend out of potential backends")
_V_BACKEND2 = re.compile(r"Using (\S+) backend\.")
_V_KV = re.compile(r"(\w+) KV cache size: ([\d,]+) tokens, Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x")
_V_GRAPH = re.compile(r"Graph capturing finished in (\d+) secs, took ([\d.]+) GiB")
_S_ARGS_LINE = re.compile(r"server_args=(.*)")
_S_KEYS = ("attention_backend", "decode_attention_backend", "prefill_attention_backend", "speculative_algorithm",
           "speculative_num_steps", "speculative_num_draft_tokens", "speculative_eagle_topk",
           "speculative_draft_attention_backend", "mem_fraction_static", "cuda_graph_max_bs", "max_running_requests",
           "chunked_prefill_size", "context_length", "disable_radix_cache", "disable_cuda_graph", "dtype",
           "kv_cache_dtype", "sampling_backend")
_S_TOTAL = re.compile(r"max_total_num_tokens=(\d+)")
_S_RUNNING = re.compile(r"max_running_requests=(\d+)")

GEMMA_MAP = {"Using FA4 for all layers": "FA4 (forced by Gemma4Config)",
             "per-layer FA3/FA4 selection": "per-layer FA3/FA4",
             "forcing TRITON_ATTN backend": "TRITON_ATTN (FA4 unavailable)",
             "Leaving the attention backend": "platform default (XPU)"}


def _kv(text: str, key: str) -> str | None:
    m = re.search(r"(?:'%s'\s*:\s*|\b%s=)('([^']*)'|\"([^\"]*)\"|[^,)}\s]+)" % (key, key), text)
    if not m:
        return None
    return m.group(2) if m.group(2) is not None else (m.group(3) if m.group(3) is not None else m.group(1))


def parse(log: str, engine: str | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"engine": engine}
    if engine in (None, "vllm"):
        g = _V_GEMMA.search(log)
        b = _V_BACKEND.findall(log) or _V_BACKEND2.findall(log)
        kv = _V_KV.search(log)
        gr = _V_GRAPH.search(log)
        out.update({
            "gemma4_backend_decision": GEMMA_MAP.get(g.group(1)) if g else None,
            "attention_backend": b[0] if b else None,
            "attention_backends_seen": sorted(set(b)) or None,
            "kv_cache_tokens": int(kv.group(2).replace(",", "")) if kv else None,
            "max_model_len": int(kv.group(3).replace(",", "")) if kv else None,
            "max_concurrency_at_max_len": float(kv.group(4)) if kv else None,
            "cuda_graph_capture_s": int(gr.group(1)) if gr else None,
            "cuda_graph_mem_gib": float(gr.group(2)) if gr else None,
        })
    if engine in (None, "sglang"):
        line = _S_ARGS_LINE.search(log)
        args = {k: _kv(line.group(1), k) for k in _S_KEYS} if line else {}
        t, r = _S_TOTAL.search(log), _S_RUNNING.search(log)
        out.update({"sglang_server_args": {k: v for k, v in args.items() if v is not None} or None,
                    "kv_cache_tokens": int(t.group(1)) if t else out.get("kv_cache_tokens"),
                    "max_running_requests": int(r.group(1)) if r else None})
        if args.get("attention_backend"):
            out["attention_backend"] = args["attention_backend"]
        if args.get("speculative_algorithm"):
            out["speculative_algorithm"] = args["speculative_algorithm"]
    return out
