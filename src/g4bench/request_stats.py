"""Within-run, per-request statistics with bootstrap CIs from InferenceX ``--save-detailed`` output.

Definitions match InferenceX ``calculate_metrics`` exactly, using only successful requests:
  TTFT_k  = ttfts[k]
  E2EL_k  = ttfts[k] + sum(itls[k])        (== output.latency: timestamps chain, so this is exact)
  TPOT_k  = sum(itls[k]) / (L_k - 1)       for L_k > 1 (L_k = output_lens[k])
  ITL     = pooled itls over successful requests

Throughput has no per-request definition. We report a Little's-law estimator for a closed loop at
fixed concurrency c, ``c * sum(L) / sum(E2EL)``, bootstrapped over requests, next to the measured
value; their gap shows how much ramp-up/tail effects matter. It is a **within-run diagnostic**,
not a substitute for between-run replication (stats.py), and is labelled as such everywhere.

Missing or inconsistent logs are reported, never imputed:
* no per-request arrays (run without --save-detailed) -> state ``not_recorded``;
* arrays with mismatched lengths -> state ``inconsistent`` (nothing computed);
* failed requests (non-empty ``errors[k]`` or ``output_lens[k] == 0``) are excluded and counted;
* fewer than ``MIN_N`` successful requests -> estimates given, CIs omitted (``ci: null``).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

MIN_N = 5
STATS = ("mean", "p50", "p90", "p99")


def _stat(x: np.ndarray, s: str) -> float:
    return float(x.mean()) if s == "mean" else float(np.percentile(x, float(s[1:])))


def _boot(x: np.ndarray, s: str, iters: int, rng: np.random.Generator) -> tuple[float, float]:
    idx = rng.integers(0, x.size, size=(iters, x.size))
    xs = x[idx]
    vals = xs.mean(axis=1) if s == "mean" else np.percentile(xs, float(s[1:]), axis=1)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def extract(raw: dict[str, Any]) -> dict[str, Any]:
    """Per-request arrays (seconds) for successful requests, plus a data-quality state."""
    ttfts, itls, lens = raw.get("ttfts"), raw.get("itls"), raw.get("output_lens")
    errors = raw.get("errors")
    if ttfts is None or itls is None or lens is None:
        return {"state": "not_recorded", "reason": "run lacks per-request arrays (no --save-detailed?)"}
    n = len(lens)
    if len(ttfts) != n or len(itls) != n or (errors is not None and len(errors) != n):
        return {"state": "inconsistent",
                "reason": f"array lengths differ: ttfts={len(ttfts)} itls={len(itls)} output_lens={n}"
                          + (f" errors={len(errors)}" if errors is not None else "")}
    ok = [k for k in range(n) if lens[k] and lens[k] > 0 and not (errors and errors[k])]
    ttft = np.array([ttfts[k] for k in ok], dtype=float)
    itl_sum = np.array([float(sum(itls[k] or [])) for k in ok], dtype=float)
    L = np.array([lens[k] for k in ok], dtype=float)
    e2el = ttft + itl_sum
    m = L > 1
    tpot = itl_sum[m] / (L[m] - 1)
    pooled = np.array([v for k in ok for v in (itls[k] or [])], dtype=float)
    bad_ttft = int(np.sum(~np.isfinite(ttft) | (ttft <= 0))) if ttft.size else 0
    return {"state": "ok", "n_total": n, "n_ok": len(ok), "n_failed": n - len(ok), "n_bad_ttft": bad_ttft,
            "ttft": ttft, "tpot": tpot, "e2el": e2el, "itl": pooled, "L": L}


def summarize(raw: dict[str, Any], *, conc: int, iters: int = 4000, seed: int = 0) -> dict[str, Any]:
    ex = extract(raw)
    if ex["state"] != "ok":
        return {"state": ex["state"], "reason": ex["reason"]}
    rng = np.random.default_rng(seed)
    out: dict[str, Any] = {"state": "ok", "n_total": ex["n_total"], "n_ok": ex["n_ok"], "n_failed": ex["n_failed"],
                           "n_bad_ttft": ex["n_bad_ttft"], "metrics": {}}
    if ex["n_ok"] == 0:
        out["state"] = "no_successful_requests"
        return out
    for name in ("ttft", "tpot", "e2el", "itl"):
        x = ex[name] * 1000.0  # ms
        rec = {}
        for s in STATS:
            if x.size == 0:
                rec[s] = {"est": None, "ci": None}
                continue
            est = _stat(x, s)
            ci = _boot(x, s, iters, rng) if x.size >= MIN_N else None
            rec[s] = {"est": est, "ci": list(ci) if ci else None}
        rec["n"] = int(x.size)
        out["metrics"][f"{name}_ms"] = rec
    L, E = ex["L"], ex["e2el"]
    est = conc * L.sum() / E.sum() if E.sum() > 0 else math.nan
    ci = None
    if L.size >= MIN_N:
        idx = rng.integers(0, L.size, size=(iters, L.size))
        vals = conc * L[idx].sum(axis=1) / E[idx].sum(axis=1)
        ci = [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
    measured = raw.get("output_throughput")
    out["throughput_littles_law"] = {
        "est": float(est), "ci": ci, "measured": measured,
        "rel_gap": (float(est) / measured - 1) if measured else None,
        "label": "within-run diagnostic (Little's law, closed loop at fixed concurrency)",
    }
    return out
