"""Kernel lab: hybrid tokenizer (patch hybrid_tokenizer) vs HF tokenizers vs VLLM_USE_FASTOKENS in vLLM 0.33.0 (CPU only).

The 8K c=1 pair best-e8-pfa2-sync(-ftok) (results/pairs/20261113T232853Z-*) showed fastokens cutting TTFT (encode 13.5
-> 1.1 ms) but costing 4 % output throughput / +1.2 ms p99 TPOT. This probe times, per backend, in a fresh process each
(fastokens patches process-globally):

  encode         tok(prompt, **renderer encode kwargs) and renderer.render_cmpl_async, InferenceX-style random prompts
                 (8K and 1K, Gemma chat template applied client-side, as kernels/frontend_ttft_probe.py builds them)
  detok_init     IncrementalDetokenizer.from_new_request (DecodeStream primed with the 7.4K prompt ids)
  detok_first    first update() + get_next_output_text (prompt-seeded first step)
  detok_step     every later per-token update() + get_next_output_text, 1024 generated-like tokens
  decode_1024    tok.decode of the 1024 tokens

Backends: ``hf`` (stock), ``fastokens`` (VLLM_USE_FASTOKENS=1), ``hybrid`` (vllm/tokenizers/registry.py patched by
src/g4bench/patches/hybrid_tokenizer.py, loaded through an import hook, G4_HYBRID_TOKENIZER=1). Checks: prompt ids and
the streamed text of 3 token streams (random-prompt slice, English/code text, text with special tokens) equal hf's.

    g4bench modal script --path kernels/hybrid_tok_probe.py --max-usd 1.5 --execute
Writes $G4_OUT_DIR/hybrid_tok_probe.json.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
TARGET_REPO = "google/gemma-4-12B-it"
TARGET_REV = "43c9d014b231474fa7212dbb48a762566c2b255d"
OUT = Path(os.environ.get("G4_OUT_DIR", "/tmp/hybrid_tok_probe"))
FTK_DIR = "/tmp/fastokens_pkgs"
PATCHED_DIR = Path("/tmp/g4_hybtok")
REPS, WARM = 25, 3
N_GEN = 1024
DETOK_PROMPTS = 4   # 8K prompts used for the detokenizer runs
DETOK_REPS = 3

VARIANTS = {
    "hf": {},
    "fastokens": {"VLLM_USE_FASTOKENS": "1"},
    "hybrid": {"G4_HYBRID_TOKENIZER": "1"},
}


def stats(xs: list[float]) -> dict:
    xs = sorted(xs)
    q = lambda f: xs[min(len(xs) - 1, int(f * (len(xs) - 1)))]  # noqa: E731
    return {"med_us": round(statistics.median(xs) * 1e6, 2), "mean_us": round(statistics.fmean(xs) * 1e6, 2),
            "p99_us": round(q(0.99) * 1e6, 2), "min_us": round(xs[0] * 1e6, 2)}


def timeit(fn, reps: int = REPS) -> dict:
    for _ in range(WARM):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return stats(ts)


def mean_rows(rows: list[dict]) -> dict:
    return {k: round(sum(r[k] for r in rows) / len(rows), 2) for k in rows[0]}


def load_frontend_probe():
    spec = importlib.util.spec_from_file_location("frontend_ttft_probe", HERE / "frontend_ttft_probe.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------------------------------------------------- orchestrator
def build_streams(prompts: dict, path: Path) -> dict:
    """Three 1024-token 'generated-like' streams, from the stock HF tokenizer."""
    import inspect
    import textwrap

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TARGET_REPO, revision=TARGET_REV)
    rand = tok.encode(prompts["isl8192"][0], add_special_tokens=False)[200:200 + N_GEN]
    src = textwrap.dedent(inspect.getsource(argparse)) + inspect.getsource(json)
    text = tok.encode(src, add_special_tokens=False)[:N_GEN]
    specials = sorted(set(tok.all_special_ids))
    mixed = list(text)
    for k, i in enumerate(range(50, N_GEN, 97)):
        mixed[i] = specials[k % len(specials)]
    streams = {"rand": rand, "text": text, "mixed": mixed}
    path.write_text(json.dumps(streams))
    return {k: len(v) for k, v in streams.items()} | {"n_specials": len(specials)}


def ensure_fastokens() -> dict:
    try:
        import importlib.metadata as md
        return {"installed": md.version("fastokens"), "source": "image"}
    except Exception:  # noqa: BLE001
        pass
    p = subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", "--target", FTK_DIR,
                        "fastokens==0.3.6"], capture_output=True, text=True, timeout=300)
    return {"installed": None if p.returncode else "0.3.6 (pip --target)", "pythonpath": FTK_DIR, "rc": p.returncode,
            "err": p.stderr[-800:]}


def write_patched_registry() -> dict:
    """Apply the image patch's text edits to the installed registry.py (anchor check) into PATCHED_DIR."""
    sys.path.insert(0, str(REPO / "src"))
    from g4bench.patches import hybrid_tokenizer as P

    spec = importlib.util.find_spec("vllm")
    path = Path(list(spec.submodule_search_locations)[0]) / P.RELPATH
    src = path.read_text(encoding="utf-8")
    out, changed = P.patch_text(src, runtime_path=str(REPO / "src" / "g4bench" / "patches" / "hybrid_tokenizer.py"))
    again, changed2 = P.patch_text(out)
    PATCHED_DIR.mkdir(parents=True, exist_ok=True)
    (PATCHED_DIR / "registry.py").write_text(out, encoding="utf-8")
    compile(out, str(PATCHED_DIR / "registry.py"), "exec")
    return {"target": str(path), "changed": changed, "idempotent": (not changed2) and again == out,
            "lines": [len(src.splitlines()), len(out.splitlines())]}


# ---------------------------------------------------------------------------------------------------------- worker
def install_registry_hook() -> None:
    import importlib.abc

    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name == "vllm.tokenizers.registry":
                return importlib.util.spec_from_file_location(name, PATCHED_DIR / "registry.py")
            return None

    sys.meta_path.insert(0, Finder())


def worker(variant: str, prompts_path: Path, streams_path: Path) -> dict:
    if variant == "hybrid":
        install_registry_hook()
    res: dict = {"variant": variant, "env": VARIANTS[variant]}
    import copy
    import pickle

    import tokenizers.decoders
    import vllm
    from vllm.engine.arg_utils import EngineArgs
    from vllm.renderers import renderer_from_config
    from vllm.sampling_params import RequestOutputKind, SamplingParams
    from vllm.v1.engine.input_processor import InputProcessor
    res["vllm"] = vllm.__version__
    ea = EngineArgs(model=TARGET_REPO, revision=TARGET_REV, dtype="bfloat16", enable_prefix_caching=False,
                    max_num_batched_tokens=8192)
    vcfg = ea.create_engine_config()
    mc = vcfg.model_config
    renderer = renderer_from_config(vcfg)
    tok = renderer.tokenizer
    inner = tok._tokenizer
    cls = lambda o: f"{type(o).__module__}.{type(o).__name__}"  # noqa: E731
    res["classes"] = {"tokenizer": cls(tok), "inner": cls(inner),
                      "inner_deepcopy": cls(copy.deepcopy(tok)._tokenizer),
                      "inner_pickled": cls(pickle.loads(pickle.dumps(inner))) if variant != "fastokens" else None,
                      "DecodeStream": cls(tokenizers.decoders.DecodeStream) + "/" + tokenizers.decoders.DecodeStream.__name__}
    if variant == "hybrid":
        res["hybrid_installed"] = type(inner).__name__ == "HybridBackend"
    ip = InputProcessor(vcfg, renderer)
    from vllm.entrypoints.openai.completion.protocol import CompletionRequest
    from vllm.renderers.inputs.preprocess import parse_model_prompt
    from vllm.v1.engine.detokenizer import IncrementalDetokenizer

    loop = asyncio.new_event_loop()
    prompts = json.loads(prompts_path.read_text())
    streams = json.loads(streams_path.read_text())
    res["encode"], ids_all, eins = {}, {}, {}
    for wl, texts in prompts.items():
        per: dict[str, list] = {}
        ids_all[wl] = []
        for pi, text in enumerate(texts):
            body = json.dumps({"model": TARGET_REPO, "prompt": text, "max_tokens": N_GEN, "temperature": 0.0,
                               "stream": True, "ignore_eos": True})
            tok_params = CompletionRequest.model_validate_json(body).build_tok_params(mc)
            kw = tok_params.get_encode_kwargs()
            res.setdefault("encode_kwargs", {k: v for k, v in kw.items()})
            per.setdefault("tok_call", []).append(timeit(lambda: tok(text, **kw)))
            per.setdefault("inner_encode", []).append(
                timeit(lambda: inner.encode(text, add_special_tokens=kw.get("add_special_tokens", True))))

            async def render():
                return await renderer.render_cmpl_async([parse_model_prompt(mc, {"prompt": text})], tok_params)
            per.setdefault("render", []).append(timeit(lambda: loop.run_until_complete(render())))
            (ein,) = loop.run_until_complete(render())
            ids_all[wl].append(list(ein["prompt_token_ids"]))
            if wl == "isl8192" and pi < DETOK_PROMPTS:
                eins[pi] = ein
        res["encode"][wl] = {"n_tokens": [len(x) for x in ids_all[wl]], "mean": {k: mean_rows(v) for k, v in per.items()}}
    loop.close()
    # other encode paths (truncation sync, batch, special-token text, chat template) must match hf too
    t8, t1 = prompts["isl8192"][0], prompts["isl1024"][0]
    calls = {
        "trunc512": lambda: tok(t8, truncation=True, max_length=512)["input_ids"],
        "notrunc_after": lambda: tok(t8)["input_ids"],
        "batch": lambda: tok([t1, prompts["isl1024"][1]])["input_ids"],
        "special_text": lambda: tok.encode("hi <start_of_turn>user\nyo<end_of_turn>\n<eos> x", add_special_tokens=True),
        "chat": lambda: list(tok.apply_chat_template([{"role": "user", "content": t1[:2000]}],
                                                     add_generation_prompt=True, tokenize=True, return_dict=False)),
        "convert_ids_to_tokens": lambda: tok.convert_ids_to_tokens(streams["mixed"][:300]),
        "batch_decode": lambda: tok.batch_decode([streams["text"][:200], streams["mixed"][:200]]),
    }
    extra = {}
    for k, fn in calls.items():
        try:
            extra[k] = fn()
        except Exception as e:  # noqa: BLE001
            extra[k] = f"ERROR {type(e).__name__}: {e}"[:300]
    ids_all["extra"] = extra
    (OUT / f"ids_{variant}.json").write_text(json.dumps(ids_all))

    # detokenizer: vLLM's own IncrementalDetokenizer (FastIncrementalDetokenizer for a TokenizersBackend)
    sp = SamplingParams(max_tokens=N_GEN, temperature=0.0, ignore_eos=True, output_kind=RequestOutputKind.DELTA)
    texts_out: dict[str, str] = {}
    det: dict[str, dict[str, list]] = {}
    n = [0]
    for pi, ein in eins.items():
        n[0] += 1
        ecr = ip.process_inputs(f"r{n[0]}", dict(ein), sp, supported_tasks=("generate",))
        for sname, gen in streams.items():
            rows = det.setdefault(sname, {"init": [], "first": [], "step": [], "decode_1024": []})
            for rep in range(DETOK_REPS + 1):
                t0 = time.perf_counter()
                d = IncrementalDetokenizer.from_new_request(tok, ecr)
                t_init = time.perf_counter() - t0
                pieces, steps = [], []
                for t in gen:
                    t0 = time.perf_counter()
                    d.update([t], False)
                    pieces.append(d.get_next_output_text(False, True))
                    steps.append(time.perf_counter() - t0)
                pieces.append(d.get_next_output_text(True, True))
                if rep == 0:  # warm-up rep; its text is the one compared
                    res.setdefault("detok_cls", type(d).__name__)
                    texts_out[f"p{pi}/{sname}"] = "".join(pieces)
                    continue
                rows["init"].append(t_init)
                rows["first"].append(steps[0])
                rows["step"].extend(steps[1:])
            rows["decode_1024"].append(timeit(lambda: tok.decode(gen, skip_special_tokens=True), reps=10)["med_us"] / 1e6)
    res["detok"] = {s: {k: stats(v) for k, v in rows.items()} for s, rows in det.items()}
    (OUT / f"texts_{variant}.json").write_text(json.dumps(texts_out))
    res["decode_text_vs_stream"] = {s: (tok.decode(g, skip_special_tokens=True) == texts_out["p0/" + s])
                                    for s, g in streams.items()}
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker")
    ap.add_argument("--prompts")
    ap.add_argument("--streams")
    a = ap.parse_args()
    if a.worker:
        try:
            r = worker(a.worker, Path(a.prompts), Path(a.streams))
        except Exception:  # noqa: BLE001
            r = {"variant": a.worker, "error": traceback.format_exc()[-4000:]}
        print("RESULT " + json.dumps(r), flush=True)
        return

    OUT.mkdir(parents=True, exist_ok=True)
    summary: dict = {}
    fk = ensure_fastokens()
    summary["fastokens"] = fk
    try:
        summary["patch"] = write_patched_registry()
    except Exception:  # noqa: BLE001
        summary["patch"] = {"error": traceback.format_exc()[-2000:]}
    print(json.dumps(summary), flush=True)
    fp = load_frontend_probe()
    pp, sp_ = OUT / "prompts.json", OUT / "streams.json"
    fp.build_prompts(pp)
    allp = json.loads(pp.read_text())
    pp.write_text(json.dumps({"isl8192": allp["isl8192"][:DETOK_PROMPTS], "isl1024": allp["isl1024"][:2]}))
    summary["streams"] = build_streams(allp, sp_)
    summary["variants"] = {}
    for v, extra in VARIANTS.items():
        if v == "hybrid" and "error" in summary["patch"]:
            continue
        env = {**os.environ, **extra}
        if fk.get("pythonpath"):
            env["PYTHONPATH"] = fk["pythonpath"] + os.pathsep + env.get("PYTHONPATH", "")
        p = subprocess.run([sys.executable, __file__, "--worker", v, "--prompts", str(pp), "--streams", str(sp_)],
                           capture_output=True, text=True, env=env, timeout=1200)
        line = next((ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")), None)
        r = json.loads(line[7:]) if line else {"error": f"rc={p.returncode}", "stderr": p.stderr[-3000:]}
        summary["variants"][v] = r
        print(json.dumps(r), flush=True)
    eq: dict = {}
    for v in ("fastokens", "hybrid"):
        for kind in ("ids", "texts"):
            a_, b_ = OUT / f"{kind}_hf.json", OUT / f"{kind}_{v}.json"
            if a_.exists() and b_.exists():
                A, B = json.loads(a_.read_text()), json.loads(b_.read_text())
                eq[f"{v}/{kind}"] = A == B
                if A != B:
                    eq[f"{v}/{kind}_diff_keys"] = [k for k in A if A.get(k) != B.get(k)][:10]
                if kind == "ids":
                    eq[f"{v}/extra"] = {k: A["extra"][k] == B["extra"].get(k) for k in A.get("extra", {})}
    summary["equal_to_hf"] = eq
    (OUT / "hybrid_tok_probe.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({"equal_to_hf": eq}), flush=True)


if __name__ == "__main__":
    main()
