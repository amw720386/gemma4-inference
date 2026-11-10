"""Turn a prepared real-prompt set into a seeded, reproducible request list.

Selection policy (pre-registered, DECISIONS D-014):
* ``rng = numpy.random.default_rng(seed)``; ``order = rng.permutation(n_prompts_in_set)``;
* the request list is ``order`` cycled until ``num_prompts`` requests exist. With 80 coding
  prompts and num_prompts = 10*conc, conc >= 16 repeats prompts. Prefix caching / radix cache
  are disabled in every baseline, so repeats are recomputed, not cache hits (recorded as
  ``n_unique`` / ``max_repeats`` in the result for transparency).
* prompts are rendered with the tokenizer's chat template exactly as InferenceX does for its
  random dataset (``_apply_chat_template``), and ``prompt_len`` is the token count of the rendered
  prompt without extra special tokens (same definition as InferenceX).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from .paths import REPO

DEFAULT_DATA_DIR = REPO / "data" / "speedbench"


def data_dir() -> Path:
    return Path(os.environ.get("G4_DATA_DIR", str(DEFAULT_DATA_DIR)))


def dataset_path(name: str) -> Path:
    return data_dir() / f"{name}.jsonl"


def select(n_available: int, num_prompts: int, seed: int) -> list[int]:
    if n_available <= 0:
        raise ValueError("empty prompt set")
    if num_prompts <= 0:
        raise ValueError("num_prompts must be positive")
    order = np.random.default_rng(seed).permutation(n_available).tolist()
    return [order[i % n_available] for i in range(num_prompts)]


def build_requests(rows: list[dict[str, Any]], tokenizer: Any, *, num_prompts: int, seed: int, osl: int,
                   use_chat_template: bool = True, apply_template=None) -> tuple[list[tuple], dict[str, Any]]:
    """Return (InferenceX-style request tuples, selection metadata)."""
    idx = select(len(rows), num_prompts, seed)
    reqs = []
    for i in idx:
        p = rows[i]["prompt"]
        if use_chat_template:
            p = apply_template(p, tokenizer, False) if apply_template else tokenizer.apply_chat_template(
                [{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False)
        reqs.append((p, len(tokenizer.encode(p, add_special_tokens=False)), int(osl), None))
    counts: dict[int, int] = {}
    for i in idx:
        counts[i] = counts.get(i, 0) + 1
    lens = [r[1] for r in reqs]
    meta = {"selection_seed": seed, "num_prompts": num_prompts, "n_unique": len(counts),
            "max_repeats": max(counts.values()), "question_ids": [rows[i].get("question_id") for i in idx],
            "input_len_min": min(lens), "input_len_max": max(lens), "input_len_mean": float(np.mean(lens))}
    return reqs, meta
