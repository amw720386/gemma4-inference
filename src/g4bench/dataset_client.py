"""Benchmark client for prepared real-prompt sets (SPEED-Bench arms).

The vendored InferenceX CLI only accepts ``--dataset-name random``. Rather than modifying the
checksummed vendored code, this module imports its ``benchmark()`` coroutine and
``benchmark_outcome`` gate and feeds them our request list, so request dispatch, timing, warmups,
metric definitions and the 5 % failure gate are byte-for-byte InferenceX's. The saved JSON has the
same keys as InferenceX's ``--save-result --save-detailed`` output plus a ``dataset`` block with
provenance and selection metadata.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .datasets import build_requests
from .paths import THIRD_PARTY_INFERENCEX
from .speedbench import load_prepared


def _bs():
    p = str(THIRD_PARTY_INFERENCEX)
    if p not in sys.path:
        sys.path.insert(0, p)
    from infx.bench_serving import benchmark_outcome, benchmark_serving  # type: ignore

    return benchmark_serving, benchmark_outcome


def run(a: argparse.Namespace) -> dict[str, Any]:
    bs, outcome_mod = _bs()
    random.seed(a.seed)
    np.random.seed(a.seed)  # noqa: NPY002 - mirrors benchmark_serving.main
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    rows, prov = load_prepared(Path(a.dataset_file))
    reqs, sel = build_requests(rows, tok, num_prompts=a.num_prompts, seed=a.seed, osl=a.osl,
                               use_chat_template=a.use_chat_template, apply_template=bs._apply_chat_template)
    gc.collect()
    gc.freeze()
    res = asyncio.run(bs.benchmark(
        backend=a.backend, api_url=a.base_url + a.endpoint, base_url=a.base_url, model_id=a.model,
        model_name=None, tokenizer=tok, input_requests=reqs, logprobs=None, best_of=1,
        request_rate=float("inf"), burstiness=1.0, disable_tqdm=True, num_warmups=a.num_warmups, profile=False,
        selected_percentile_metrics=["ttft", "tpot", "itl", "e2el"], selected_percentiles=[50.0, 90.0, 99.0, 99.9],
        ignore_eos=a.ignore_eos, goodput_config_dict={}, max_concurrency=a.max_concurrency, lora_modules=None,
    ))
    try:
        outcome = outcome_mod.benchmark_outcome(a.num_prompts, res["completed"])
    except ValueError as exc:
        outcome = {"status": "failed", "requested": a.num_prompts, "completed": res["completed"], "error": str(exc)}
    res["benchmark_outcome"] = outcome
    out = {"date": datetime.now().strftime("%Y%m%d-%H%M%S"), "backend": a.backend, "model_id": a.model,  # noqa: DTZ005
           "tokenizer_id": a.tokenizer, "best_of": 1, "num_prompts": a.num_prompts, "request_rate": "inf",
           "burstiness": 1.0, "max_concurrency": a.max_concurrency, "ignore_eos": a.ignore_eos,
           "dataset": {"name": prov.get("dataset"), "prompt_digest": prov.get("prompt_digest"),
                       "file_sha256": prov.get("file_sha256"),
                       "matches_inferencex_digest": prov.get("matches_inferencex_digest"),
                       "subset_warning": prov.get("SUBSET_WARNING"), **sel},
           **res}
    if not a.save_detailed:
        for k in ("ttfts", "itls", "generated_texts", "errors"):
            out.pop(k, None)
    Path(a.result_dir).mkdir(parents=True, exist_ok=True)
    (Path(a.result_dir) / a.result_filename).write_text(json.dumps(out))
    return out


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m g4bench.dataset_client")
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--endpoint", default="/v1/completions")
    ap.add_argument("--backend", default="vllm")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--dataset-file", required=True)
    ap.add_argument("--num-prompts", type=int, required=True)
    ap.add_argument("--max-concurrency", type=int, required=True)
    ap.add_argument("--num-warmups", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--osl", type=int, required=True)
    ap.add_argument("--ignore-eos", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--use-chat-template", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--save-detailed", action="store_true")
    ap.add_argument("--result-dir", required=True)
    ap.add_argument("--result-filename", required=True)
    return ap


def main(argv: list[str] | None = None) -> int:
    a = parser().parse_args(argv)
    out = run(a)
    o = out["benchmark_outcome"]
    if o.get("status") == "failed":
        print(f"FAIL: {o}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
