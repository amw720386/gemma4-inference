"""Build the InferenceX benchmark_serving argv for a point.

Mirrors ``infx.bench.fixed_seq.client_argv`` at InferenceX commit 0b991451
(docs/sources/inferencex_fixed_seq.py.txt) with three explicit, documented additions:

* ``--seed`` (InferenceX relies on the default 0; the article varies it at 8K/c4),
* ``--tokenizer`` pinned to the same HF revision as the served model,
* ``--save-detailed`` so per-request TTFT/ITL/E2E are kept for bootstrap CIs.

None of these change the request stream for seed 0 relative to InferenceX.
"""

from __future__ import annotations

import sys
from pathlib import Path

from .matrix import Point
from .paths import THIRD_PARTY_INFERENCEX


def client_argv(point: Point, *, base_url: str, model: str, backend: str, result: Path,
                tokenizer: str | None = None, python: str | None = None,
                save_detailed: bool = True, endpoint: str = "/v1/completions",
                dataset_file: Path | None = None) -> list[str]:
    if point.dataset != "random":
        return dataset_client_argv(point, base_url=base_url, model=model, backend=backend, result=result,
                                   tokenizer=tokenizer, python=python, save_detailed=save_detailed,
                                   endpoint=endpoint, dataset_file=dataset_file)
    if not point.ignore_eos:
        raise ValueError("random dataset arm always uses --ignore-eos (InferenceX policy)")
    argv = [
        python or sys.executable, "-m", "infx.bench_serving.benchmark_serving",
        "--model", model,
        "--backend", backend,
        "--base-url", base_url,
        "--endpoint", endpoint,
        "--dataset-name", "random",
        "--random-input-len", str(point.isl),
        "--random-output-len", str(point.osl),
        "--random-range-ratio", str(point.random_range_ratio),
        "--num-prompts", str(point.num_prompts),
        "--max-concurrency", str(point.conc),
        "--request-rate", "inf",
        "--ignore-eos",
        "--save-result",
        "--num-warmups", str(point.num_warmups),
        "--percentile-metrics", "ttft,tpot,itl,e2el",
        "--metric-percentiles", "50,90,99,99.9",
        "--seed", str(point.seed),
        "--result-dir", str(result.parent),
        "--result-filename", result.name,
    ]
    if tokenizer:
        argv += ["--tokenizer", tokenizer]
    if point.use_chat_template:
        argv.append("--use-chat-template")
    if save_detailed:
        argv.append("--save-detailed")
    return argv


def client_env(base_env: dict[str, str]) -> dict[str, str]:
    """PYTHONPATH that resolves ``infx`` to the vendored, checksummed copy."""
    env = dict(base_env)
    # vendored infx + our src (dataset_client runs as a subprocess, possibly outside a venv)
    pp = str(THIRD_PARTY_INFERENCEX) + ":" + str(THIRD_PARTY_INFERENCEX.parents[1] / "src")
    env["PYTHONPATH"] = pp + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    return env


def dataset_client_argv(point: Point, *, base_url: str, model: str, backend: str, result: Path,
                        tokenizer: str | None, python: str | None, save_detailed: bool, endpoint: str,
                        dataset_file: Path | None) -> list[str]:
    """Real-prompt arm: same request/timing/metric code (InferenceX ``benchmark()``), our request list."""
    if dataset_file is None or tokenizer is None:
        raise ValueError("real-prompt arm needs dataset_file and a tokenizer")
    argv = [
        python or sys.executable, "-m", "g4bench.dataset_client",
        "--base-url", base_url, "--endpoint", endpoint, "--backend", backend, "--model", model,
        "--tokenizer", tokenizer, "--dataset-file", str(dataset_file),
        "--num-prompts", str(point.num_prompts), "--max-concurrency", str(point.conc),
        "--num-warmups", str(point.num_warmups), "--seed", str(point.seed), "--osl", str(point.osl),
        "--ignore-eos" if point.ignore_eos else "--no-ignore-eos",
        "--use-chat-template" if point.use_chat_template else "--no-use-chat-template",
        "--result-dir", str(result.parent), "--result-filename", result.name,
    ]
    if save_detailed:
        argv.append("--save-detailed")
    return argv
