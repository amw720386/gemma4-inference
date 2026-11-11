"""Compare measured aggregates against the article's reported numbers.

Classification per (workload, metric), using run-level mean m, 95% CI [lo, hi] and target T:

* ``consistent``  - |m - T| / T <= tol, or T inside [lo, hi]
* ``higher`` / ``lower`` - otherwise (direction of *our* value relative to T)
* ``n/a``         - no measurement or no target

``tol`` (default 5%) absorbs systematic differences we cannot control (RTX PRO 6000
edition / power limit, driver, host CPU). It is a pre-registered constant: see
docs/EXPERIMENTS.md "Acceptance criteria". Do not tune it after seeing data.

For "beats target" claims we require the *lower* CI bound of throughput to exceed T
(or the *upper* CI bound of a latency to be below T), with n >= 3 runs.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .paths import TARGETS

# our column -> (target metric name, higher_is_better)
METRIC_MAP: dict[str, tuple[str, bool]] = {
    "output_throughput": ("output_throughput", True),
    "p99_tpot_ms": ("p99_tpot", False),
    "p99_e2el_ms": ("p99_e2el", False),
    "p90_ttft_ms": ("p90_ttft", False),
    "speed_per_user_tok_s": ("speed_per_user", True),
}

DEFAULT_TOL = 0.05


def load_targets(path: Path | None = None) -> dict[str, dict[str, dict[str, float]]]:
    """Return {"{isl}x{conc}": {target_metric: {engine: value}}}."""
    d = json.loads(Path(path or TARGETS).read_text())
    out: dict[str, dict[str, dict[str, float]]] = {}
    for p in d["points"]:
        key = f"{p['isl']}x{p['conc']}"
        out[key] = {m: v for m, v in p.items() if isinstance(v, dict)}
    return out


def classify(mean: float, lo: float, hi: float, target: float, tol: float = DEFAULT_TOL) -> str:
    if mean is None or target is None or not math.isfinite(mean):
        return "n/a"
    if abs(mean - target) / target <= tol:
        return "consistent"
    if math.isfinite(lo) and math.isfinite(hi) and lo <= target <= hi:
        return "consistent"
    return "higher" if mean > target else "lower"


def beats(mean: float, lo: float, hi: float, n: int, target: float, higher_is_better: bool) -> bool:
    if n < 3 or not (math.isfinite(lo) and math.isfinite(hi)):
        return False
    return lo > target if higher_is_better else hi < target


def compare(agg_rows: list[dict[str, Any]], reference_engine: str, *, tol: float = DEFAULT_TOL,
            targets: dict | None = None) -> list[dict[str, Any]]:
    """One row per (aggregate row, metric) against ``reference_engine`` in the article."""
    targets = targets or load_targets()
    out = []
    for r in agg_rows:
        key = f"{r['isl']}x{r['conc']}"
        tgt_point = targets.get(key, {})
        for col, (tname, hib) in METRIC_MAP.items():
            mean = r.get(f"{col}__mean")
            if mean is None:
                continue
            lo, hi, n = r.get(f"{col}__ci_lo"), r.get(f"{col}__ci_hi"), int(r.get(f"{col}__n") or 0)
            target = tgt_point.get(tname, {}).get(reference_engine)
            ratio = (mean / target) if (target and mean is not None) else None
            out.append({
                "engine_config": r.get("engine_config"), "hardware_class": r.get("hardware_class"),
                "key": key, "metric": col, "reference": reference_engine, "target": target,
                "mean": mean, "ci_lo": lo, "ci_hi": hi, "n": n, "ratio_to_target": ratio,
                "class": classify(mean, lo if lo is not None else float("nan"),
                                  hi if hi is not None else float("nan"), target, tol) if target else "n/a",
                "beats": beats(mean, lo or float("nan"), hi or float("nan"), n, target, hib) if target else False,
            })
    return out


def markdown_table(rows: list[dict[str, Any]], metric: str = "output_throughput") -> str:
    sel = [r for r in rows if r["metric"] == metric]
    lines = [f"| config | hw | workload | target ({sel[0]['reference'] if sel else '?'}) | ours mean | 95% CI | n | ours/target | class |",
             "|---|---|---|---:|---:|---|---:|---:|---|"]
    for r in sel:
        ci = (f"[{r['ci_lo']:.4g}, {r['ci_hi']:.4g}]"
              if r["ci_lo"] is not None and math.isfinite(r["ci_lo"]) else "n<2")
        ratio = f"{r['ratio_to_target']:.3f}" if r["ratio_to_target"] else "-"
        lines.append(f"| {r['engine_config']} | {r['hardware_class']} | {r['key']} | {r['target']} | "
                     f"{r['mean']:.4g} | {ci} | {r['n']} | {ratio} | {r['class']} |")
    return "\n".join(lines)
