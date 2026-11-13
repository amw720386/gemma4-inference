"""Correctness gates run against a live server during a benchmark job.

Two checks:

1. **Known-answer probes** - short prompts with unambiguous answers. Catches broken
   weight loading, wrong chat template, or a mis-wired draft that corrupts output.
   Gate: >= 7/8 correct (one allowed miss for formatting quirks).
2. **Greedy equivalence** (``greedy_run`` / ``compare_greedy``) - a fixed, deterministic prompt set
   (``greedy_prompts``: 4 short, 2 ~3K-token and 2 ~8K-token prompts, so long prefill, chunked prefill and
   the custom prefill-attention / draft-head paths are exercised) sent at concurrency 1, temperature 0,
   ``max_tokens`` 256 to the engine under test. The generated **token ids** (plus top-2 logprobs when the
   server returns them) are stored in ``correctness.json``. A candidate is compared with a reference produced
   by the BF16 no-spec config (``vllm-0.33.0-nospec``; ``data/greedy_reference/``):

   * gate: every prompt's token ids are **exactly equal** to the reference (``pass``);
   * diagnostics: per-prompt first-divergence position, identical-prefix fraction and positional match
     fraction; for each divergence, whether it is a *near tie* (the two competing tokens' logprobs differ by
     <= ``NEAR_TIE_NATS`` at the divergence position). A change of floating-point reduction order (batched
     kernels, split-K, a different attention kernel, spec-decode verification shapes) can legitimately flip a
     near-tie argmax; such divergences are reported separately, but they still fail the exact gate.

   The older word-prefix metric (``compare_outputs``; median common prefix >= 48 words and >= 50 % of
   requests identical or sharing >= 96 words) is still reported alongside for continuity.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import requests

from .paths import REPO

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

# ------------------------------------------------------------------ greedy-equivalence prompt set
GREEDY_SET_VERSION = "g1"
GREEDY_MAX_TOKENS = 256
GREEDY_TOP_LOGPROBS = 2
# A divergence is a "near tie" when the two competing tokens' logprobs differ by at most this much at the
# divergence position. Gemma 4 logits are soft-capped at 30, where one BF16 ulp is 0.125 (|x| in [16, 32)), so
# ~2 ulps of logit noise from a different reduction order is the scale of a legitimate flip.
NEAR_TIE_NATS = 0.25
REFERENCE_DIR = REPO / "data" / "greedy_reference"
REFERENCE_ENGINE = "vllm-0.33.0-nospec"

# Synthetic, deterministic long-context material (no external text, nothing copyrighted): an inventory log
# whose entries are chosen by SHA-256 of (seed, index). Record counts were calibrated with the pinned Gemma 4
# tokenizer + chat template (tests/test_greedy_gate.py checks the token lengths when the tokenizer is cached).
_ADJ = ["amber", "brass", "cedar", "dusty", "early", "faded", "giant", "hollow", "ivory", "jade", "keen",
        "lunar", "mossy", "narrow", "olive", "pale", "quiet", "rusty", "silver", "tidal", "upper", "velvet",
        "woven", "young"]
_NOUN = ["anchor", "barrel", "candle", "drum", "engine", "flask", "gear", "harness", "ingot", "jar", "kettle",
         "lantern", "mallet", "needle", "oar", "pulley", "quill", "rope", "sail", "trunk", "valve", "wheel"]
_PLACE = ["Arden", "Briar", "Calder", "Dunmore", "Elstow", "Fenwick", "Glenrock", "Harlow", "Ivel", "Kestrel",
          "Larkspur", "Marrow", "Northgate", "Oakhurst", "Pellam", "Redcliff", "Stourby", "Thornwick"]
_STATE = ["sealed", "cracked", "polished", "repaired", "unopened", "inspected", "damp", "labelled"]

LONG_SPECS = (  # (id, records, seed, question)
    ("ctx3k-a", 96, 1, "Which entries come from Calder? List every matching entry number, then state the "
                        "total weight of those entries and describe the trend you see across the log."),
    ("ctx3k-b", 96, 2, "Report the bay, weight and condition of entry 0007, entry 0051 and entry 0090, then "
                        "summarise which item types appear most often in the log."),
    ("ctx8k-a", 254, 3, "Which entries come from Thornwick? List every matching entry number, then state the "
                        "heaviest of them and describe how the log is organised."),
    ("ctx8k-b", 254, 4, "Report the bay, weight and condition of entry 0003, entry 0140 and entry 0250, then "
                        "list five places that occur in the log and how many entries each has."),
)


def _pick(seed: int, i: int, field: str, options: list[str] | int) -> Any:
    h = int.from_bytes(hashlib.sha256(f"{seed}:{i}:{field}".encode()).digest()[:8], "big")
    return options[h % len(options)] if isinstance(options, list) else h % options


def inventory_log(n_records: int, seed: int) -> str:
    lines = []
    for i in range(n_records):
        state = _pick(seed, i, "a", _STATE)
        lines.append(f"Entry {i:04d}: {'an' if state[0] in 'aeiou' else 'a'} {state} {_pick(seed, i, 'j', _ADJ)} "
                     f"{_pick(seed, i, 'n', _NOUN)} from {_pick(seed, i, 'p', _PLACE)}, weight "
                     f"{100 + _pick(seed, i, 'w', 9900)} grams, stored in bay {_pick(seed, i, 'b', 64)}.")
    return "\n".join(lines)


def greedy_prompts() -> list[dict[str, str]]:
    """The fixed prompt set (ids are stable; changing any text changes ``prompt_set_sha256``)."""
    out = [{"id": f"short-{k}", "kind": "short", "prompt": p} for k, p in enumerate(EQUIV_PROMPTS[:4])]
    for pid, n, seed, q in LONG_SPECS:
        text = ("The following is an inventory log. Read it carefully.\n\n" + inventory_log(n, seed)
                + f"\n\nEnd of log.\n\nQuestion: {q}")
        out.append({"id": pid, "kind": pid.split("-")[0], "prompt": text})
    return out


def prompt_set_sha256(prompts: list[dict[str, str]] | None = None, max_tokens: int = GREEDY_MAX_TOKENS) -> str:
    prompts = prompts if prompts is not None else greedy_prompts()
    blob = json.dumps({"v": GREEDY_SET_VERSION, "max_tokens": max_tokens,
                       "prompts": [[p["id"], p["prompt"]] for p in prompts]}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


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
    """Legacy text-only capture (kept for callers of the old API)."""
    return [complete(base_url, model, p, max_tokens) for p in EQUIV_PROMPTS]


# ------------------------------------------------------------------ token-level greedy capture
def _token_id(tok: Any) -> int | None:
    """``return_tokens_as_token_ids`` renders logprob tokens as ``"token_id:123"``."""
    if isinstance(tok, int):
        return tok
    if isinstance(tok, str) and tok.startswith("token_id:"):
        try:
            return int(tok.split(":", 1)[1])
        except ValueError:
            return None
    return None


def parse_choice(choice: dict[str, Any]) -> dict[str, Any]:
    """token ids + per-position top logprobs from one OpenAI-compatible chat choice (vLLM extensions)."""
    content = ((choice.get("logprobs") or {}).get("content")) or []
    ids = choice.get("token_ids")
    if ids is None and content:
        ids = [_token_id(c.get("token")) for c in content]
        if any(x is None for x in ids):
            ids = None
    top = None
    if content:
        top = []
        for c in content:
            alts = [[_token_id(t.get("token")), t.get("logprob")] for t in (c.get("top_logprobs") or [])]
            chosen = [_token_id(c.get("token")), c.get("logprob")]
            if chosen[0] is not None and all(a[0] != chosen[0] for a in alts):
                alts.insert(0, chosen)
            top.append(alts)
    return {"token_ids": ids, "top_logprobs": top, "text": (choice.get("message") or {}).get("content") or "",
            "finish_reason": choice.get("finish_reason")}


def greedy_request(base_url: str, model: str, prompt: str, max_tokens: int = GREEDY_MAX_TOKENS,
                   top_logprobs: int = GREEDY_TOP_LOGPROBS, timeout: float = 600) -> dict[str, Any]:
    body: dict[str, Any] = {"model": model, "messages": _chat(prompt), "temperature": 0.0, "max_tokens": max_tokens,
                            "seed": 0, "return_token_ids": True, "return_tokens_as_token_ids": True}
    if top_logprobs:
        body.update(logprobs=True, top_logprobs=top_logprobs)
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    r = requests.post(url, json=body, timeout=timeout)
    mode = "token_ids+logprobs" if top_logprobs else "token_ids"
    if r.status_code == 400 and top_logprobs:  # engine rejects logprobs (e.g. with spec decode): ids only
        body.pop("logprobs"), body.pop("top_logprobs")
        r = requests.post(url, json=body, timeout=timeout)
        mode = "token_ids"
    r.raise_for_status()
    j = r.json()
    out = parse_choice(j["choices"][0])
    out["prompt_tokens"] = (j.get("usage") or {}).get("prompt_tokens")
    out["completion_tokens"] = (j.get("usage") or {}).get("completion_tokens")
    out["mode"] = mode if out["token_ids"] is not None else "text-only"
    return out


def greedy_run(base_url: str, model: str, prompts: list[dict[str, str]] | None = None,
               max_tokens: int = GREEDY_MAX_TOKENS, top_logprobs: int = GREEDY_TOP_LOGPROBS) -> dict[str, Any]:
    """Send the prompt set sequentially (concurrency 1) and record token ids for each prompt."""
    prompts = prompts if prompts is not None else greedy_prompts()
    t0 = time.monotonic()
    items = []
    for p in prompts:
        rec: dict[str, Any] = {"id": p["id"], "kind": p["kind"]}
        try:
            rec.update(greedy_request(base_url, model, p["prompt"], max_tokens, top_logprobs))
        except Exception as e:  # noqa: BLE001 - recorded; an errored item fails the gate
            rec["error"] = repr(e)[:1000]
        items.append(rec)
    return {"set_version": GREEDY_SET_VERSION, "prompt_set_sha256": prompt_set_sha256(prompts, max_tokens),
            "max_tokens": max_tokens, "temperature": 0.0, "concurrency": 1, "top_logprobs": top_logprobs,
            "elapsed_s": round(time.monotonic() - t0, 2), "items": items}


# ------------------------------------------------------------------ comparison
def first_divergence(a: list[int], b: list[int]) -> int | None:
    """Index of the first differing token (len of the shorter one if one is a prefix); None if equal."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def _gap(top: list[list[Any]] | None, i: int, chosen: int, other: int) -> float | None:
    """logprob(chosen) - logprob(other) at position i, if both are among the recorded top tokens."""
    if not top or i >= len(top):
        return None
    lp = {t[0]: t[1] for t in top[i] if t[1] is not None}
    if chosen in lp and other in lp:
        return float(lp[chosen]) - float(lp[other])
    return None


def _outside_top(top: list[list[Any]] | None, i: int, other: int) -> bool:
    """The competing token is not in this side's recorded top-k: the gap exceeds that of the k-th token."""
    return bool(top) and i < len(top) and len(top[i]) >= GREEDY_TOP_LOGPROBS and all(t[0] != other for t in top[i])


def compare_item(ref: dict[str, Any], cand: dict[str, Any], near_tie_nats: float = NEAR_TIE_NATS) -> dict[str, Any]:
    out: dict[str, Any] = {"id": ref.get("id")}
    if ref.get("error") or cand.get("error"):
        return {**out, "identical": False, "status": "error",
                "error": cand.get("error") or f"reference: {ref.get('error')}"}
    a, b = ref.get("token_ids"), cand.get("token_ids")
    if a is None or b is None:  # text-only fallback: no token-level verdict possible
        same = (ref.get("text") or "") == (cand.get("text") or "")
        return {**out, "identical": same, "status": "text-only", "text_identical": same,
                "common_prefix_words": common_prefix_words(ref.get("text") or "", cand.get("text") or "")}
    n = max(len(a), len(b), 1)
    d = first_divergence(a, b)
    out.update(identical=d is None, ref_len=len(a), cand_len=len(b),
               first_divergence=d, prefix_fraction=round((n if d is None else d) / n, 4),
               positional_match_fraction=round(sum(x == y for x, y in zip(a, b)) / n, 4))
    if d is None:
        return {**out, "status": "identical"}
    if d >= min(len(a), len(b)):  # one stopped early (EOS vs continue): compare at the EOS position
        out["status"] = "length"
        return out
    ra, cb = a[d], b[d]
    gaps = [g for g in (_gap(ref.get("top_logprobs"), d, ra, cb), _gap(cand.get("top_logprobs"), d, cb, ra))
            if g is not None]
    out.update(ref_token=ra, cand_token=cb, logprob_gap=(round(min(abs(g) for g in gaps), 5) if gaps else None))
    if gaps:
        out["near_tie"] = min(abs(g) for g in gaps) <= near_tie_nats
    elif _outside_top(ref.get("top_logprobs"), d, cb) and _outside_top(cand.get("top_logprobs"), d, ra):
        out["near_tie"] = False
    else:
        out["near_tie"] = None  # no logprobs recorded
    out["status"] = "divergent"
    return out


def compare_greedy(ref: dict[str, Any], cand: dict[str, Any], near_tie_nats: float = NEAR_TIE_NATS) -> dict[str, Any]:
    """Exact token-equality gate of a candidate ``greedy_run`` against a reference ``greedy_run``."""
    if ref.get("prompt_set_sha256") != cand.get("prompt_set_sha256"):
        return {"pass": False, "error": "prompt set differs from the reference "
                f"({cand.get('prompt_set_sha256')} vs {ref.get('prompt_set_sha256')})"}
    by_id = {it["id"]: it for it in cand.get("items") or []}
    per = [compare_item(r, by_id.get(r["id"], {"id": r["id"], "error": "missing"}), near_tie_nats)
           for r in ref.get("items") or []]
    tok = [p for p in per if p["status"] in ("identical", "divergent", "length")]
    ref_tokens = sum(p["ref_len"] for p in tok)
    div = [p for p in per if p["status"] == "divergent"]
    words = compare_outputs([r.get("text") or "" for r in ref.get("items") or []],
                            [by_id.get(r["id"], {}).get("text") or "" for r in ref.get("items") or []])
    return {
        "pass": bool(per) and all(p["identical"] and p["status"] != "text-only" for p in per),
        "criterion": "exact token-id equality on every prompt (temperature 0, concurrency 1)",
        "n": len(per), "n_identical": sum(p["identical"] for p in per),
        "n_divergent": len(div), "n_length_mismatch": sum(p["status"] == "length" for p in per),
        "n_near_tie": sum(p.get("near_tie") is True for p in div),
        "n_not_near_tie": sum(p.get("near_tie") is False for p in div),
        "n_tie_unknown": sum(p.get("near_tie") is None for p in div),
        "n_error": sum(p["status"] == "error" for p in per), "n_text_only": sum(p["status"] == "text-only" for p in per),
        "near_tie_nats": near_tie_nats,
        "identical_prefix_token_fraction": (round(sum((p["ref_len"] if p["first_divergence"] is None
                                                       else p["first_divergence"]) for p in tok) / ref_tokens, 4)
                                            if ref_tokens else None),
        "legacy_word_gate": words,
        "items": per,
    }


def reference_path(engine: str = REFERENCE_ENGINE) -> Path:
    return REFERENCE_DIR / f"{engine}.json"


def load_reference(path: str | Path | None = None) -> dict[str, Any] | None:
    p = Path(path) if path else reference_path()
    if not p.is_absolute() and not p.exists():
        p = REPO / p  # repo-relative (inside Modal the repo is /repo, cwd may differ)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def promote(run_dir: Path, dest: Path | None = None) -> Path:
    """Turn a reference-config run's greedy capture into ``data/greedy_reference/<engine>.json``."""
    run_dir = Path(run_dir)
    corr = json.loads((run_dir / "correctness.json").read_text(encoding="utf-8"))
    g = corr.get("greedy")
    if not g or any(it.get("error") or it.get("token_ids") is None for it in g.get("items") or []):
        raise SystemExit(f"{run_dir}: no complete token-level greedy capture to promote")
    man = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    meta = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    engine = man["engine_config"]["name"]
    if engine != REFERENCE_ENGINE:
        print(f"[g4bench] warning: promoting {engine}, not the pre-registered reference {REFERENCE_ENGINE}")
    ref = {**g, "reference": {"engine_config": engine, "run_id": run_dir.name, "git": meta.get("git"),
                              "hardware_class": meta.get("hardware_class"),
                              "packages": {k: v for k, v in (meta.get("packages") or {}).items() if v}}}
    dest = Path(dest) if dest else reference_path(engine)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(ref, indent=1), encoding="utf-8")
    return dest


def compare_run(run_dir: Path, reference: str | Path | None = None) -> dict[str, Any]:
    """Post-hoc: compare a run's stored greedy capture with a reference file; writes greedy_compare.json."""
    run_dir = Path(run_dir)
    ref = load_reference(reference)
    if ref is None:
        raise SystemExit(f"no reference at {reference or reference_path()}")
    corr = json.loads((run_dir / "correctness.json").read_text(encoding="utf-8"))
    if not corr.get("greedy"):
        raise SystemExit(f"{run_dir}: correctness.json has no token-level greedy capture")
    res = {"reference": ref.get("reference"), **compare_greedy(ref, corr["greedy"])}
    (run_dir / "greedy_compare.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


def summary_line(cmp: dict[str, Any]) -> str:
    if "error" in cmp:
        return f"greedy equivalence: ERROR {cmp['error']}"
    return (f"greedy equivalence: {'PASS' if cmp['pass'] else 'FAIL'} {cmp['n_identical']}/{cmp['n']} identical; "
            f"{cmp['n_divergent']} divergent ({cmp['n_near_tie']} near-tie <= {cmp['near_tie_nats']} nats, "
            f"{cmp['n_not_near_tie']} not, {cmp['n_tie_unknown']} unknown), {cmp['n_length_mismatch']} length, "
            f"{cmp['n_error']} error; identical-prefix tokens {cmp['identical_prefix_token_fraction']}")


# ------------------------------------------------------------------ legacy word-level metric
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
    res = {"probes": probes(a.base_url, a.model), "greedy": greedy_run(a.base_url, a.model)}
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(json.dumps({"probes_pass": res["probes"]["pass"], "n_ok": res["probes"]["n_ok"]}))
