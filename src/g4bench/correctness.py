"""Correctness gates run against a live server before/after a benchmark.

Two checks:

1. **Known-answer probes** - short prompts with unambiguous answers. Catches broken
   weight loading, wrong chat template, or a mis-wired draft that corrupts output.
   Gate: >= 7/8 correct (one allowed miss for formatting quirks).
2. **Greedy equivalence** - the same greedy (temperature 0) requests sent to two
   servers (e.g. spec-decode vs no-spec, or optimized vs baseline). Speculative
   decoding with exact verification should be output-preserving up to floating-point
   non-determinism of batched BF16 kernels, so we report the token-prefix agreement
   distribution rather than requiring 100% bit-identity. Gate (pre-registered):
   median common-prefix >= 64 tokens and >= 50% of requests identical over 128 tokens
   at concurrency 1. Any optimization failing this is reported as NOT output-preserving.
"""

from __future__ import annotations

import json
from typing import Any

import requests

PROBES: list[tuple[str, list[str]]] = [
    ("What is 17 + 25? Answer with just the number.", ["42"]),
    ("What is the capital of France? Answer with one word.", ["paris"]),
    ("Spell the word 'cat' backwards. Answer with just the result.", ["tac"]),
    ("What is 9 times 8? Answer with just the number.", ["72"]),
    ("What color do you get by mixing blue and yellow? One word.", ["green"]),
    ("How many days are in a leap year? Answer with just the number.", ["366"]),
    ("What is the chemical symbol for gold? Answer with just the symbol.", ["au"]),
    ("Translate 'thank you' into Spanish. Answer with just the translation.", ["gracias"]),
]

EQUIV_PROMPTS = [
    "Write a detailed explanation of how a hash map works, including collision handling.",
    "List the first 30 prime numbers separated by commas, then explain the sieve of Eratosthenes.",
    "Write a Python function that parses ISO-8601 dates without using datetime, with tests.",
    "Summarize the causes and consequences of the French Revolution in five paragraphs.",
    "Explain the difference between TCP and UDP, with examples of protocols built on each.",
    "Write a short story about a lighthouse keeper who discovers a message in a bottle.",
    "Describe the steps of the Krebs cycle and the role of each enzyme.",
    "Derive the quadratic formula step by step.",
]


def _chat(prompt: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": prompt}]


def complete(base_url: str, model: str, prompt: str, max_tokens: int, timeout: float = 300) -> str:
    r = requests.post(f"{base_url.rstrip('/')}/v1/chat/completions", timeout=timeout,
                      json={"model": model, "messages": _chat(prompt), "temperature": 0.0,
                            "max_tokens": max_tokens, "seed": 0})
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"] or ""


def probes(base_url: str, model: str) -> dict[str, Any]:
    out = []
    for q, answers in PROBES:
        try:
            a = complete(base_url, model, q, 32)
            ok = any(x in a.strip().lower() for x in answers)
        except Exception as e:  # noqa: BLE001
            a, ok = f"ERROR {e!r}", False
        out.append({"q": q, "a": a, "ok": ok})
    n_ok = sum(o["ok"] for o in out)
    return {"n_ok": n_ok, "n": len(out), "pass": n_ok >= len(out) - 1, "items": out}


def greedy_outputs(base_url: str, model: str, max_tokens: int = 256) -> list[str]:
    return [complete(base_url, model, p, max_tokens) for p in EQUIV_PROMPTS]


def common_prefix_words(a: str, b: str) -> int:
    """Whitespace-token common prefix length (tokenizer-free proxy, conservative)."""
    n = 0
    for x, y in zip(a.split(), b.split()):
        if x != y:
            break
        n += 1
    return n


def compare_outputs(ref: list[str], test: list[str], min_words: int = 96) -> dict[str, Any]:
    prefixes = [common_prefix_words(a, b) for a, b in zip(ref, test)]
    identical = sum(1 for a, b, p in zip(ref, test, prefixes) if a == b or p >= min_words)
    srt = sorted(prefixes)
    median = srt[len(srt) // 2] if srt else 0
    return {"prefix_words": prefixes, "median_prefix_words": median, "n_identical_or_long": identical,
            "n": len(prefixes), "pass": median >= 48 and identical >= len(prefixes) / 2}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {"probes": probes(a.base_url, a.model), "greedy": greedy_outputs(a.base_url, a.model)}
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(json.dumps({"probes_pass": res["probes"]["pass"], "n_ok": res["probes"]["n_ok"]}))
