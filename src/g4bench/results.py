"""Normalize InferenceX result JSON into a flat, versioned record; append to CSV/JSONL.

Schema version bumps whenever a column's meaning changes. Columns are fixed so the
CSV stays machine-readable across runs.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = 2  # v2 (2026-11-10): + dataset, ignore_eos, dataset_digest, request-level CI columns

LAT_METRICS = ("ttft", "tpot", "itl", "e2el")
LAT_STATS = ("mean", "median", "p90", "p99")

COLUMNS: list[str] = [
    "schema_version", "run_id", "config_hash", "engine", "engine_config", "engine_version",
    "hardware_class", "gpu_name", "gpu_count", "exact_hardware", "point_id", "key",
    "isl", "osl", "conc", "seed", "trial", "num_prompts", "random_range_ratio", "use_chat_template",
    "dataset", "ignore_eos", "dataset_digest",
    "status", "completed", "failed", "failure_rate", "duration_s",
    "total_input_tokens", "total_output_tokens", "mean_input_len", "mean_output_len",
    "output_throughput", "total_token_throughput", "request_throughput",
] + [f"{s}_{m}_ms" for m in LAT_METRICS for s in LAT_STATS] + [
    "speed_per_user_tok_s",
    "req_n_ok", "req_state", "req_p99_tpot_ci_lo", "req_p99_tpot_ci_hi", "req_p99_ttft_ci_lo", "req_p99_ttft_ci_hi",
    "req_littles_law_tok_s", "req_littles_law_rel_gap",
    "spec_acceptance_length", "spec_draft_acceptance_rate",
    "gpu_mem_peak_mib", "gpu_power_mean_w", "gpu_sm_clock_mean_mhz",
    "server_start_s", "started_utc", "notes",
]


def normalize(raw: dict[str, Any], *, context: dict[str, Any]) -> dict[str, Any]:
    """Map one InferenceX result JSON (+ run context) to a COLUMNS-keyed record."""
    rec: dict[str, Any] = {c: None for c in COLUMNS}
    rec["schema_version"] = SCHEMA_VERSION
    for k in ("run_id", "config_hash", "engine", "engine_config", "engine_version", "hardware_class",
              "gpu_name", "gpu_count", "exact_hardware", "point_id", "key", "isl", "osl", "conc",
              "seed", "trial", "num_prompts", "random_range_ratio", "use_chat_template", "dataset", "ignore_eos",
              "spec_acceptance_length", "spec_draft_acceptance_rate", "gpu_mem_peak_mib",
              "gpu_power_mean_w", "gpu_sm_clock_mean_mhz", "server_start_s", "started_utc", "notes"):
        if k in context:
            rec[k] = context[k]
    outcome = raw.get("benchmark_outcome") or {}
    requested = int(outcome.get("requested", raw.get("num_prompts") or 0) or 0)
    completed = int(raw.get("completed", 0) or 0)
    rec["completed"] = completed
    rec["failed"] = max(requested - completed, 0)
    rec["failure_rate"] = (rec["failed"] / requested) if requested else None
    rec["status"] = outcome.get("status", "unknown")
    rec["duration_s"] = raw.get("duration")
    for k in ("total_input_tokens", "total_output_tokens", "output_throughput",
              "total_token_throughput", "request_throughput"):
        rec[k] = raw.get(k)
    if raw.get("input_lens"):
        rec["mean_input_len"] = float(np.mean(raw["input_lens"]))
    if raw.get("output_lens"):
        rec["mean_output_len"] = float(np.mean(raw["output_lens"]))
    for m in LAT_METRICS:
        for s in LAT_STATS:
            rec[f"{s}_{m}_ms"] = raw.get(f"{s}_{m}_ms")
    mt = rec.get("mean_tpot_ms")
    rec["speed_per_user_tok_s"] = (1000.0 / mt) if mt else None
    ds = raw.get("dataset")
    if isinstance(ds, dict):
        rec["dataset_digest"] = ds.get("prompt_digest")
    if raw:
        from .request_stats import summarize

        rs = summarize(raw, conc=int(context.get("conc") or raw.get("max_concurrency") or 1))
        rec["req_state"] = rs["state"]
        rec["req_n_ok"] = rs.get("n_ok")
        m = rs.get("metrics", {})
        for metric, col in (("tpot_ms", "tpot"), ("ttft_ms", "ttft")):
            ci = (m.get(metric, {}).get("p99") or {}).get("ci")
            rec[f"req_p99_{col}_ci_lo"], rec[f"req_p99_{col}_ci_hi"] = (ci or (None, None))
        ll = rs.get("throughput_littles_law") or {}
        rec["req_littles_law_tok_s"] = ll.get("est")
        rec["req_littles_law_rel_gap"] = ll.get("rel_gap")
    return rec


def validate(rec: dict[str, Any], *, osl: int, range_ratio: float, ignore_eos: bool = True,
             real_prompts: bool = False) -> list[str]:
    """Return a list of problems (empty == valid). Invalid points are kept but flagged.

    Random arm: output lengths drawn from [range_ratio*osl, osl]. Real-prompt arm with ignore_eos:
    every request must return exactly osl tokens. Real-prompt arm honouring EOS: only an upper bound.
    """
    problems: list[str] = []
    if real_prompts:
        range_ratio = 1.0 if ignore_eos else 0.0
    if rec.get("status") != "passed":
        problems.append(f"client outcome {rec.get('status')!r}")
    if rec.get("failed"):
        problems.append(f"{rec['failed']} failed requests")
    mol = rec.get("mean_output_len")
    if mol is not None:
        lo = max(int(osl * range_ratio), 1)
        if not (lo - 1 <= mol <= osl + 1):
            problems.append(f"mean output len {mol:.1f} outside [{lo},{osl}] (ignore_eos not honoured?)")
    for k in ("output_throughput", "mean_ttft_ms", "mean_tpot_ms", "p99_e2el_ms"):
        v = rec.get(k)
        if v is None or not np.isfinite(v) or v <= 0:
            problems.append(f"{k}={v!r} not positive/finite")
    return problems


def append_csv(path: Path, rec: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow({k: _fmt(rec.get(k)) for k in COLUMNS})


def append_jsonl(path: Path, rec: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, sort_keys=True, default=str) + "\n")


def read_csv(path: Path) -> list[dict[str, Any]]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [{k: _parse(v) for k, v in r.items()} for r in rows]


def _fmt(v: Any) -> Any:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.6g}"
    return v


def _parse(v: str) -> Any:
    if v == "":
        return None
    if v in ("True", "False"):
        return v == "True"
    try:
        if v.lstrip("-").isdigit():
            return int(v)
        return float(v)
    except ValueError:
        return v
