"""Statistics for repeated benchmark trials.

Two levels of uncertainty are handled separately and never conflated:

1. **Between-run** (seeds x independent server restarts): the unit of replication is a
   whole benchmark point. We report mean, sample SD, coefficient of variation, and a
   Student-t 95% CI on the mean (n is small, typically 3-6), plus min/max.
2. **Within-run** (per-request latency distribution): percentile CIs by bootstrap over
   requests. These describe sampling noise of a percentile *within one run* and are
   reported only as a diagnostic (e.g. p99 over 10 requests at conc=1 is ~the max).

Ratios between engines use a bootstrap over run-level values (resampling each engine's
runs independently), which makes no normality assumption about the ratio.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

# Two-sided 97.5% Student-t quantiles for df = 1..30 (avoids a scipy dependency).
_T975 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179,
         2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064,
         2.060, 2.056, 2.052, 2.048, 2.045, 2.042]


def t975(df: int) -> float:
    if df < 1:
        return float("nan")
    return _T975[df - 1] if df <= 30 else 1.96


@dataclass(frozen=True)
class Summary:
    n: int
    mean: float
    sd: float
    cv: float
    ci_lo: float
    ci_hi: float
    min: float
    max: float

    def as_dict(self) -> dict[str, float]:
        return self.__dict__.copy()


def summarize(values: Iterable[float]) -> Summary:
    x = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    n = int(x.size)
    if n == 0:
        nan = float("nan")
        return Summary(0, nan, nan, nan, nan, nan, nan, nan)
    mean = float(x.mean())
    if n == 1:
        return Summary(1, mean, float("nan"), float("nan"), float("nan"), float("nan"), mean, mean)
    sd = float(x.std(ddof=1))
    half = t975(n - 1) * sd / math.sqrt(n)
    return Summary(n, mean, sd, sd / mean if mean else float("nan"), mean - half, mean + half,
                   float(x.min()), float(x.max()))


def bootstrap_ratio(num: Iterable[float], den: Iterable[float], *, iters: int = 10000,
                    seed: int = 0) -> tuple[float, float, float]:
    """Point estimate and 95% percentile-bootstrap CI for mean(num)/mean(den)."""
    a = np.asarray(list(num), dtype=float)
    b = np.asarray(list(den), dtype=float)
    if a.size == 0 or b.size == 0:
        return (float("nan"),) * 3
    est = float(a.mean() / b.mean())
    if a.size == 1 and b.size == 1:
        return est, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    ra = a[rng.integers(0, a.size, size=(iters, a.size))].mean(axis=1)
    rb = b[rng.integers(0, b.size, size=(iters, b.size))].mean(axis=1)
    r = ra / rb
    return est, float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def bootstrap_percentile(samples: Iterable[float], q: float, *, iters: int = 5000,
                         seed: int = 0) -> tuple[float, float, float]:
    """Within-run percentile with bootstrap 95% CI (diagnostic only)."""
    x = np.asarray(list(samples), dtype=float)
    if x.size == 0:
        return (float("nan"),) * 3
    est = float(np.percentile(x, q))
    rng = np.random.default_rng(seed)
    boots = np.percentile(x[rng.integers(0, x.size, size=(iters, x.size))], q, axis=1)
    return est, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def group_by(rows: list[dict[str, Any]], *keys: str) -> dict[tuple, list[dict[str, Any]]]:
    out: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        out[tuple(r.get(k) for k in keys)].append(r)
    return dict(out)


def aggregate(rows: list[dict[str, Any]], metrics: list[str],
              keys: tuple[str, ...] = ("engine_config", "config_hash", "hardware_class", "isl", "osl", "conc"),
              valid_only: bool = True) -> list[dict[str, Any]]:
    """Run-level aggregation: one output row per (engine config, hw, workload)."""
    out = []
    for gk, grp in sorted(group_by(rows, *keys).items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        use = [r for r in grp if (not valid_only or r.get("valid", True))]
        row: dict[str, Any] = dict(zip(keys, gk))
        row["n_runs"] = len(use)
        row["n_invalid"] = len(grp) - len(use)
        row["seeds"] = sorted({r.get("seed") for r in use})
        for m in metrics:
            s = summarize(r.get(m) for r in use)
            for f, v in s.as_dict().items():
                row[f"{m}__{f}"] = v
        out.append(row)
    return out
