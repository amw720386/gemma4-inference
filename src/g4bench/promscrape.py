"""Scrape Prometheus /metrics and compute speculative-decoding acceptance.

Acceptance length (AL) = 1 + accepted draft tokens / verification steps, the same
definition InferenceX's al_gate uses. Metric names differ by engine and version,
so we match by pattern and always store the raw before/after deltas of every
spec-related counter for later re-analysis.
"""

from __future__ import annotations

import re
import urllib.request
from typing import Any

_LINE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([-+0-9.eEinfNa]+)")

# (accepted_tokens_counter, verification_steps_counter, drafted_tokens_counter) candidates
VLLM_NAMES = ("vllm:spec_decode_num_accepted_tokens_total", "vllm:spec_decode_num_drafts_total",
              "vllm:spec_decode_num_draft_tokens_total")


def parse(text: str) -> dict[str, float]:
    """Sum samples per metric name (labels collapsed)."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = _LINE.match(line)
        if not m:
            continue
        try:
            v = float(m.group(3))
        except ValueError:
            continue
        out[m.group(1)] = out.get(m.group(1), 0.0) + v
    return out


def scrape(base_url: str, timeout: float = 10.0) -> dict[str, float] | None:
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/metrics", timeout=timeout) as r:
            return parse(r.read().decode("utf-8", "replace"))
    except Exception:  # noqa: BLE001 - metrics are optional diagnostics
        return None


def delta(before: dict[str, float] | None, after: dict[str, float] | None) -> dict[str, float]:
    if not before or not after:
        return {}
    return {k: after[k] - before.get(k, 0.0) for k in after if ("spec" in k or "accept" in k or "draft" in k)}


def acceptance(d: dict[str, float]) -> dict[str, Any]:
    """Best-effort AL and per-draft acceptance rate from counter deltas."""
    acc, steps, drafted = (d.get(n) for n in VLLM_NAMES)
    if acc is not None and steps:
        return {"acceptance_length": 1.0 + acc / steps,
                "draft_acceptance_rate": (acc / drafted) if drafted else None, "source": "vllm"}
    # SGLang exposes e.g. sglang:spec_accept_length (gauge/histogram) in recent versions; match loosely.
    for k, v in d.items():
        if k.startswith("sglang") and "accept_length" in k and k.endswith("_sum"):
            cnt = d.get(k[:-4] + "_count")
            if cnt:
                return {"acceptance_length": v / cnt, "draft_acceptance_rate": None, "source": k}
    return {"acceptance_length": None, "draft_acceptance_rate": None, "source": None}


# --- log-based fallback (server logs are always captured) ---
# vLLM v0.33.0 vllm/v1/spec_decode/metrics.py: "Mean acceptance length: %.2f, " (includes bonus token)
# SGLang v0.5.23 metrics_reporter.py: "accept len: %.2f, accept rate: %.2f, "
_VLLM_LOG = re.compile(r"Mean acceptance length:\s*([0-9.]+)")
_SGL_LOG = re.compile(r"accept len:\s*([0-9.]+),\s*accept rate:\s*([0-9.]+)")


def acceptance_from_log(text: str) -> dict[str, Any]:
    """Mean of logged interval acceptance lengths (unweighted; diagnostic only)."""
    v = [float(x) for x in _VLLM_LOG.findall(text)]
    if v:
        return {"acceptance_length_log_mean": sum(v) / len(v), "n_log_samples": len(v), "source": "vllm-log"}
    s = [(float(a), float(r)) for a, r in _SGL_LOG.findall(text)]
    s = [x for x in s if x[0] > 0]
    if s:
        return {"acceptance_length_log_mean": sum(a for a, _ in s) / len(s),
                "accept_rate_log_mean": sum(r for _, r in s) / len(s),
                "n_log_samples": len(s), "source": "sglang-log"}
    return {"acceptance_length_log_mean": None, "n_log_samples": 0, "source": None}
