"""Regenerate the exact InferenceX random-prompt set on CPU and fingerprint it.

This lets us (a) prove our client produces the same request stream as InferenceX
for a given seed, (b) measure chat-template overhead and actual token lengths, and
(c) detect hidden nondeterminism (e.g. dependence on CPU count) before paying for GPUs.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import random
import sys
from typing import Any

import numpy as np

from .paths import THIRD_PARTY_INFERENCEX


def _bench_module():
    p = str(THIRD_PARTY_INFERENCEX)
    if p not in sys.path:
        sys.path.insert(0, p)
    from infx.bench_serving import benchmark_serving  # type: ignore

    return benchmark_serving


def generate(tokenizer_path: str, *, isl: int, osl: int, num_prompts: int, seed: int,
             range_ratio: float, use_chat_template: bool = True, num_workers: int = 1) -> list[tuple]:
    bs = _bench_module()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_path)
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - mirrors benchmark_serving.main
    with contextlib.redirect_stdout(sys.stderr):  # the upstream sampler prints stats to stdout
        return bs.sample_random_requests(prefix_len=0, input_len=isl, output_len=osl, num_prompts=num_prompts,
                                         range_ratio=range_ratio, tokenizer=tok,
                                         use_chat_template=use_chat_template, tokenizer_id=tokenizer_path,
                                         num_workers=num_workers)


def fingerprint(reqs: list[tuple]) -> dict[str, Any]:
    h = hashlib.sha256()
    for r in reqs:
        h.update(r[0].encode("utf-8"))
        h.update(f"|{r[1]}|{r[2]}\n".encode())
    il = [r[1] for r in reqs]
    ol = [r[2] for r in reqs]
    return {"sha256": h.hexdigest(), "n": len(reqs), "input_len_min": min(il), "input_len_max": max(il),
            "input_len_mean": float(np.mean(il)), "output_len_min": min(ol), "output_len_max": max(ol),
            "output_len_mean": float(np.mean(ol)), "first_prompt_head": reqs[0][0][:120]}


def chat_template_overhead(tokenizer_path: str) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_path)
    s = tok.apply_chat_template([{"role": "user", "content": "a"}], add_generation_prompt=True, tokenize=False)
    ids = tok.encode(s, add_special_tokens=False)
    return {"template_rendered": s, "template_tokens_incl_a": len(ids), "overhead_tokens": len(ids) - 1}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("tokenizer")
    ap.add_argument("--isl", type=int, default=1024)
    ap.add_argument("--osl", type=int, default=1024)
    ap.add_argument("--conc", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--range-ratio", type=float, default=0.8)
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    reqs = generate(a.tokenizer, isl=a.isl, osl=a.osl, num_prompts=10 * a.conc, seed=a.seed,
                    range_ratio=a.range_ratio, num_workers=a.workers)
    print(json.dumps(fingerprint(reqs), indent=2))
