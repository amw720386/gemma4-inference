"""A tiny OpenAI-compatible streaming server for CPU-only harness tests.

It is NOT a performance model of any real engine. It exists so the full pipeline
(server lifecycle -> InferenceX client -> result normalization -> stats) can be
exercised without a GPU. Timing knobs are deliberately simple:

* TTFT = ``prefill_ms_per_1k`` * prompt_chars/4/1000 (+ queueing)
* each decode step takes ``step_ms`` * (1 + ``batch_penalty`` * (active-1))
  and emits ``tokens_per_step`` tokens (simulating speculative acceptance)

It exposes /health, /v1/models, /v1/completions (stream + usage) and /metrics with
vLLM-style spec-decode counters so the Prometheus path is exercised too.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time

from aiohttp import web

STATE = {"active": 0, "accepted": 0, "drafts": 0, "draft_tokens": 0, "requests": 0}


def make_app(model: str, step_ms: float, tokens_per_step: int, prefill_ms_per_1k: float,
             batch_penalty: float, num_spec: int, fail_every: int = 0,
             profile_dir: str | None = None) -> web.Application:
    app = web.Application()

    async def health(_: web.Request) -> web.Response:
        return web.Response(text="ok")

    async def models(_: web.Request) -> web.Response:
        return web.json_response({"object": "list", "data": [{"id": model, "object": "model"}]})

    async def metrics(_: web.Request) -> web.Response:
        lines = [
            "# TYPE vllm:spec_decode_num_accepted_tokens_total counter",
            f'vllm:spec_decode_num_accepted_tokens_total{{model_name="{model}"}} {STATE["accepted"]}',
            f'vllm:spec_decode_num_drafts_total{{model_name="{model}"}} {STATE["drafts"]}',
            f'vllm:spec_decode_num_draft_tokens_total{{model_name="{model}"}} {STATE["draft_tokens"]}',
            f'vllm:num_requests_running{{model_name="{model}"}} {STATE["active"]}',
        ]
        return web.Response(text="\n".join(lines) + "\n")

    async def completions(req: web.Request) -> web.StreamResponse:
        body = await req.json()
        STATE["requests"] += 1
        if fail_every and STATE["requests"] % fail_every == 0:
            return web.json_response({"error": {"message": "injected failure"}}, status=500)
        max_tokens = int(body.get("max_tokens", 16))
        if not body.get("ignore_eos"):
            max_tokens = max(max_tokens // 2, 1)  # simulated natural EOS when the client honours EOS
        prompt = body.get("prompt", "")
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(req)
        STATE["active"] += 1
        try:
            await asyncio.sleep(prefill_ms_per_1k * (len(prompt) / 4) / 1000 / 1000)
            sent = 0
            created = int(time.time())
            while sent < max_tokens:
                k = min(tokens_per_step, max_tokens - sent)
                await asyncio.sleep(step_ms * (1 + batch_penalty * (STATE["active"] - 1)) / 1000)
                chunk = {"id": "cmpl-mock", "object": "text_completion", "created": created, "model": model,
                         "choices": [{"index": 0, "text": " tok" * k, "finish_reason": None}]}
                await resp.write(f"data: {json.dumps(chunk)}\n\n".encode())
                sent += k
                if num_spec:
                    STATE["drafts"] += 1
                    STATE["draft_tokens"] += num_spec
                    STATE["accepted"] += max(k - 1, 0)
            usage = {"id": "cmpl-mock", "object": "text_completion", "created": created, "model": model,
                     "choices": [], "usage": {"prompt_tokens": len(prompt) // 4, "completion_tokens": sent,
                                              "total_tokens": len(prompt) // 4 + sent}}
            await resp.write(f"data: {json.dumps(usage)}\n\n".encode())
            await resp.write(b"data: [DONE]\n\n")
        finally:
            STATE["active"] -= 1
        return resp

    async def chat(req: web.Request) -> web.Response:
        body = await req.json()
        q = body["messages"][-1]["content"]
        # Canned answers so the correctness probes can be exercised end-to-end on CPU.
        canned = {"17 + 25": "42", "capital of France": "Paris", "backwards": "tac", "9 times 8": "72",
                  "blue and yellow": "Green", "leap year": "366", "gold": "Au", "thank you": "Gracias"}
        ans = next((v for k, v in canned.items() if k in q), "mock answer " * 8)
        return web.json_response({"choices": [{"index": 0, "message": {"role": "assistant", "content": ans}}]})

    async def start_profile(_: web.Request) -> web.Response:
        STATE["profile_start"] = time.time()
        return web.Response(status=200)

    async def stop_profile(_: web.Request) -> web.Response:
        # Stand-in for vLLM's /stop_profile: with --profile-dir, write a tiny synthetic Kineto-style trace so the
        # harness's trace -> summary path runs on CPU. The content is fake by construction.
        if profile_dir and STATE.get("profile_start"):
            import gzip
            from pathlib import Path

            d = Path(profile_dir)
            d.mkdir(parents=True, exist_ok=True)
            ev = [{"ph": "X", "cat": "user_annotation", "name": "execute_context_0(0)_generation_1(4)",
                   "pid": 1, "tid": 1, "ts": 0, "dur": 100},
                  {"ph": "X", "cat": "cuda_runtime", "name": "cudaGraphLaunch", "pid": 1, "tid": 1, "ts": 10,
                   "dur": 5, "args": {"correlation": 1}},
                  {"ph": "X", "cat": "kernel", "name": "mock_gemm_kernel", "pid": 0, "tid": 7, "ts": 20, "dur": 40,
                   "args": {"correlation": 1, "device": 0}}]
            with gzip.open(d / f"mock_rank0.{int(time.time() * 1000)}.pt.trace.json.gz", "wt") as f:
                json.dump({"traceEvents": ev}, f)
        STATE.pop("profile_start", None)
        return web.Response(status=200)

    app.router.add_post("/start_profile", start_profile)
    app.router.add_post("/stop_profile", stop_profile)
    app.router.add_post("/v1/chat/completions", chat)
    app.router.add_get("/health", health)
    app.router.add_get("/v1/models", models)
    app.router.add_get("/metrics", metrics)
    app.router.add_post("/v1/completions", completions)
    return app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mock/model")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18000)
    ap.add_argument("--step-ms", type=float, default=2.0)
    ap.add_argument("--tokens-per-step", type=int, default=3)
    ap.add_argument("--prefill-ms-per-1k", type=float, default=20.0)
    ap.add_argument("--batch-penalty", type=float, default=0.02)
    ap.add_argument("--num-spec", type=int, default=4)
    ap.add_argument("--fail-every", type=int, default=0)
    ap.add_argument("--profile-dir", default=None, help="/stop_profile writes a synthetic trace here")
    a = ap.parse_args()
    print(f"mockserver listening on {a.host}:{a.port} (NOT a real engine)", flush=True)
    web.run_app(make_app(a.model, a.step_ms, a.tokens_per_step, a.prefill_ms_per_1k, a.batch_penalty,
                         a.num_spec, a.fail_every, a.profile_dir), host=a.host, port=a.port, print=None)


if __name__ == "__main__":
    main()
