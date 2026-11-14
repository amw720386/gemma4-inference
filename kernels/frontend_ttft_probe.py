"""Kernel lab: where does the non-GPU part of TTFT go in vLLM 0.33.0's /v1/completions path? (CPU only, no model load)

Server TTFT (vllm:time_to_first_token_seconds) runs from `arrival_time`, which BaseRenderer.render_cmpl_async stamps
BEFORE tokenization, to the frontend's processing of the first EngineCoreOutputs. The engine-core prefill histogram
(scheduled -> first token) excludes everything in the frontend. This probe times every per-request CPU step of that
path for InferenceX-style random-token prompts (8K and 1K workloads, Gemma chat template applied client-side):

  hf_encode          tokenizer(prompt) as vLLM calls it (renderer encode kwargs), and the raw Rust encode
  render             renderer.render_cmpl_async (arrival stamp -> tokenize in the renderer pool -> engine input)
  process_inputs     InputProcessor.process_inputs (validation, SamplingParams clone, EngineCoreRequest)
  msgpack            MsgpackEncoder/Decoder round trip of the EngineCoreRequest (frontend -> engine core IPC)
  core_request       Request.from_engine_core_request (engine-core input thread, no prefix caching)
  detok_init         IncrementalDetokenizer.from_new_request (OutputProcessor.add_request, before the IPC send)
  detok_first        first-token detokenizer update
  body_parse         CompletionRequest.model_validate_json of the HTTP body (before arrival_time: client TTFT only)

Variants run in separate subprocesses (VLLM_USE_FASTOKENS patches process-globally): default engine args, text-only
--limit-mm-per-prompt, --language-model-only, and VLLM_USE_FASTOKENS=1 (fail-soft if the package is unavailable).
Token ids of every variant are compared with the default renderer's ids (identical ids = identical outputs).

    g4bench modal script --path kernels/frontend_ttft_probe.py --max-usd 1.5 --execute
Writes $G4_OUT_DIR/frontend_ttft_probe.json.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

TARGET_REPO = "google/gemma-4-12B-it"
TARGET_REV = "43c9d014b231474fa7212dbb48a762566c2b255d"
OUT = Path(os.environ.get("G4_OUT_DIR", "/tmp/frontend_ttft_probe"))
REPS = 25
WARM = 3

VARIANTS = {
    "default": {"engine": {}, "env": {}},
    "mm0": {"engine": {"limit_mm_per_prompt": {"image": 0, "audio": 0, "video": 0}}, "env": {}},
    "lmo": {"engine": {"language_model_only": True}, "env": {}},
    "fastokens": {"engine": {}, "env": {"VLLM_USE_FASTOKENS": "1"}},
}


def stats(xs: list[float]) -> dict:
    xs = sorted(xs)
    return {"med_ms": round(statistics.median(xs) * 1e3, 3), "p90_ms": round(xs[int(0.9 * (len(xs) - 1))] * 1e3, 3),
            "min_ms": round(xs[0] * 1e3, 3)}


def timeit(fn, reps: int = REPS) -> dict:
    for _ in range(WARM):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return stats(ts)


def agg(rows: list[dict]) -> dict:
    """Mean over prompts of each per-prompt median/p90."""
    keys = rows[0].keys()
    return {k: round(sum(r[k] for r in rows) / len(rows), 3) for k in keys}


# ------------------------------------------------------------------------------------------------ prompts (orchestrator)
def build_prompts(path: Path) -> dict:
    """InferenceX sample_random_requests (serial path, seed 0, range ratio 0.8, --use-chat-template)."""
    import numpy as np
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TARGET_REPO, revision=TARGET_REV)
    out = {}
    for name, isl, n in (("isl8192", 8192, 8), ("isl1024", 1024, 4)):
        np.random.seed(0)
        vocab = tok.vocab_size
        np.random.randint(0, vocab, size=0)  # prefix_len 0
        dummy = tok.apply_chat_template([{"role": "user", "content": "a"}], add_generation_prompt=True, tokenize=False)
        input_len = isl - (len(tok.encode(dummy, add_special_tokens=False)) - 1)
        lens = np.random.randint(int(input_len * 0.8), input_len + 1, size=n).tolist()
        np.random.randint(0, 1024, size=n)  # output lens (unused)
        offsets = np.random.randint(0, vocab, size=n).tolist()
        rng = np.random.RandomState(np.random.get_state()[1][:4].tolist())
        prompts = []
        for i in range(n):
            ids = [(offsets[i] + i + j) % vocab for j in range(lens[i])]
            text = tok.decode(ids)
            for _ in range(10):
                ids = tok.encode(text, add_special_tokens=False)
                if len(ids) < lens[i]:
                    ids.extend(rng.randint(0, vocab, size=lens[i] - len(ids)).tolist())
                elif len(ids) > lens[i]:
                    ids = ids[:lens[i]]
                else:
                    break
                text = tok.decode(ids)
            text = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                           tokenize=False)
            prompts.append(text)
        out[name] = prompts
    path.write_text(json.dumps(out))
    return {k: [len(p) for p in v] for k, v in out.items()}


# ---------------------------------------------------------------------------------------------------------- worker
def worker(variant: str, prompts_path: Path) -> dict:
    cfg = VARIANTS[variant]
    res: dict = {"variant": variant, "env": cfg["env"], "engine": cfg["engine"]}
    t0 = time.perf_counter()
    import vllm
    from vllm.engine.arg_utils import EngineArgs
    from vllm.renderers import renderer_from_config
    from vllm.sampling_params import RequestOutputKind, SamplingParams
    from vllm.v1.engine.input_processor import InputProcessor
    res["vllm"] = vllm.__version__
    ea = EngineArgs(model=TARGET_REPO, revision=TARGET_REV, dtype="bfloat16", enable_prefix_caching=False,
                    max_num_batched_tokens=8192, **cfg["engine"])
    vcfg = ea.create_engine_config()
    mc = vcfg.model_config
    res["config_s"] = round(time.perf_counter() - t0, 2)
    t1 = time.perf_counter()
    renderer = renderer_from_config(vcfg)
    res["renderer_init_s"] = round(time.perf_counter() - t1, 2)
    res["renderer_cls"] = type(renderer).__name__
    res["mm_processor"] = type(renderer.mm_processor).__name__ if getattr(renderer, "mm_processor", None) else None
    res["supports_mm_inputs"] = bool(getattr(mc, "supports_multimodal_inputs", None))
    tok = renderer.tokenizer
    res["tokenizer_cls"] = type(tok).__name__
    inner = getattr(tok, "_tokenizer", None)
    res["inner_tokenizer_cls"] = f"{type(inner).__module__}.{type(inner).__name__}" if inner is not None else None
    ip = InputProcessor(vcfg, renderer)

    try:
        from vllm.entrypoints.openai.completion.protocol import CompletionRequest
    except Exception:  # noqa: BLE001
        CompletionRequest = None
    from vllm.renderers.inputs.preprocess import parse_model_prompt
    from vllm.v1.engine import EngineCoreRequest
    from vllm.v1.engine.detokenizer import IncrementalDetokenizer
    from vllm.v1.request import Request
    from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

    enc, dec = MsgpackEncoder(), MsgpackDecoder(EngineCoreRequest)
    loop = asyncio.new_event_loop()
    prompts = json.loads(prompts_path.read_text())
    res["workloads"] = {}
    for wl, texts in prompts.items():
        per: dict[str, list] = {}
        ids_out = []
        for pi, text in enumerate(texts):
            body = json.dumps({"model": TARGET_REPO, "prompt": text, "max_tokens": 1024, "temperature": 0.0,
                               "stream": True, "ignore_eos": True, "stream_options": {"include_usage": True}})
            if CompletionRequest is not None:
                req = CompletionRequest.model_validate_json(body)
                tok_params = req.build_tok_params(mc)
                per.setdefault("body_parse", []).append(timeit(lambda: CompletionRequest.model_validate_json(body)))
            else:
                tok_params = renderer.default_cmpl_tok_params
            kw = tok_params.get_encode_kwargs()
            res.setdefault("encode_kwargs", {k: v for k, v in kw.items()})
            per.setdefault("hf_encode", []).append(timeit(lambda: tok(text, **kw)))
            if inner is not None and hasattr(inner, "encode"):
                per.setdefault("rust_encode", []).append(
                    timeit(lambda: inner.encode(text, add_special_tokens=kw.get("add_special_tokens", True))))

            async def render():
                return await renderer.render_cmpl_async([parse_model_prompt(mc, {"prompt": text})], tok_params)
            per.setdefault("render", []).append(timeit(lambda: loop.run_until_complete(render())))
            (ein,) = loop.run_until_complete(render())
            ids = list(ein["prompt_token_ids"])
            ids_out.append(ids)
            sp = SamplingParams(max_tokens=1024, temperature=0.0, ignore_eos=True,
                                output_kind=RequestOutputKind.DELTA)
            n = [0]

            def proc():
                n[0] += 1
                e = dict(ein)
                return ip.process_inputs(f"r{n[0]}", e, sp, supported_tasks=("generate",))
            per.setdefault("process_inputs", []).append(timeit(proc))
            ecr = proc()
            ecr.request_id = "probe-0"
            per.setdefault("msgpack_encode", []).append(timeit(lambda: enc.encode(ecr)))
            frames = enc.encode(ecr)
            per.setdefault("msgpack_decode", []).append(timeit(lambda: dec.decode(frames)))
            res.setdefault("msgpack_bytes", {})[wl] = sum(len(f) for f in frames)
            per.setdefault("core_request", []).append(timeit(lambda: Request.from_engine_core_request(ecr, None)))
            per.setdefault("detok_init", []).append(timeit(lambda: IncrementalDetokenizer.from_new_request(tok, ecr)))

            def first():
                d = IncrementalDetokenizer.from_new_request(tok, ecr)
                t = time.perf_counter()
                d.update([236761], False)
                d.get_next_output_text(False, True)
                return time.perf_counter() - t
            for _ in range(WARM):
                first()
            per.setdefault("detok_first", []).append(stats([first() for _ in range(REPS)]))
            if pi == 0:
                res.setdefault("detok_cls", type(IncrementalDetokenizer.from_new_request(tok, ecr)).__name__)
        res["workloads"][wl] = {"n_tokens": [len(x) for x in ids_out],
                                "steps": {k: agg(v) for k, v in per.items()}}
        (OUT / f"ids_{variant}_{wl}.json").write_text(json.dumps(ids_out))
    loop.close()
    return res


def ensure_fastokens() -> dict:
    try:
        import importlib.metadata as md
        return {"installed": md.version("fastokens"), "source": "image"}
    except Exception:  # noqa: BLE001
        pass
    tgt = "/tmp/fastokens_pkgs"
    p = subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", "--target", tgt,
                        "fastokens>=0.2.0"], capture_output=True, text=True, timeout=300)
    return {"installed": None if p.returncode else "pip --target", "pythonpath": tgt, "rc": p.returncode,
            "err": p.stderr[-800:]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker")
    ap.add_argument("--prompts")
    a = ap.parse_args()
    if a.worker:
        try:
            r = worker(a.worker, Path(a.prompts))
        except Exception:  # noqa: BLE001
            r = {"variant": a.worker, "error": traceback.format_exc()[-3000:]}
        print("RESULT " + json.dumps(r), flush=True)
        return

    OUT.mkdir(parents=True, exist_ok=True)
    pp = OUT / "prompts.json"
    t0 = time.perf_counter()
    summary: dict = {"prompt_chars": build_prompts(pp), "prompt_build_s": round(time.perf_counter() - t0, 1),
                     "variants": {}}
    print(json.dumps(summary), flush=True)
    for v, cfg in VARIANTS.items():
        env = {**os.environ, **cfg["env"]}
        if v == "fastokens":
            fk = ensure_fastokens()
            summary["fastokens"] = fk
            print(json.dumps({"fastokens": fk}), flush=True)
            if not fk.get("installed"):
                continue
            if fk.get("pythonpath"):
                env["PYTHONPATH"] = fk["pythonpath"] + os.pathsep + env.get("PYTHONPATH", "")
        p = subprocess.run([sys.executable, __file__, "--worker", v, "--prompts", str(pp)], capture_output=True,
                           text=True, env=env, timeout=900)
        line = next((ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")), None)
        r = json.loads(line[7:]) if line else {"error": f"rc={p.returncode}", "stderr": p.stderr[-3000:]}
        summary["variants"][v] = r
        print(json.dumps(r), flush=True)
    # token-id identity vs the default renderer
    ident = {}
    for v in summary["variants"]:
        for wl in ("isl8192", "isl1024"):
            a_, b_ = OUT / f"ids_default_{wl}.json", OUT / f"ids_{v}_{wl}.json"
            if a_.exists() and b_.exists():
                ident[f"{v}/{wl}"] = json.loads(a_.read_text()) == json.loads(b_.read_text())
    summary["ids_identical_to_default"] = ident
    (OUT / "frontend_ttft_probe.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({"ids_identical_to_default": ident}), flush=True)


if __name__ == "__main__":
    main()
