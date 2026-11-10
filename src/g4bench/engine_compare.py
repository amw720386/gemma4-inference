"""Matched-load comparison of two engines (default vLLM vs SGLang) from measured runs.

"Matched load" means every client-side variable that shapes the request stream is identical:
dataset + prompt digest, ISL, OSL, concurrency, range ratio, chat template, ignore_eos, num_prompts,
and the same hardware class (exact article hardware for anything labelled a reproduction). Rows are
paired by seed (same seed => same request stream, see configs/fingerprints-rr0.8.json).

Gates (a pair is reported but marked ``gated`` instead of producing a ratio when any fails):
* both sides valid point-runs;
* both runs' correctness probes passed (``correctness.json`` of the run);
* same model pins (target + draft revisions).

Ratios are "A advantage" in the article's convention: throughput A/B, latency B/A (>1 = A better).
CIs: percentile bootstrap of the ratio of means over runs (stats.bootstrap_ratio) and, when >= 2 seeds are
paired, a Student-t CI on the mean of per-seed log ratios. n < 3 is labelled descriptive.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np

from . import compare, stats

METRICS: dict[str, bool] = {  # column -> higher is better (the article's 13 metrics)
    "output_throughput": True,
    "mean_ttft_ms": False, "median_ttft_ms": False, "p90_ttft_ms": False, "p99_ttft_ms": False,
    "mean_tpot_ms": False, "median_tpot_ms": False, "p90_tpot_ms": False, "p99_tpot_ms": False,
    "mean_e2el_ms": False, "median_e2el_ms": False, "p90_e2el_ms": False, "p99_e2el_ms": False,
}
RATIO_COL = {m: ("output_throughput" if m == "output_throughput" else m.removesuffix("_ms")) for m in METRICS}


def article_implied(key: str) -> dict[str, dict[str, float]]:
    """Article-implied vLLM-vs-SGLang advantage for all 13 metrics at ``{isl}x{conc}``, Backboard cancelled out.

    vs_sglang/vs_vllm = (BB/S)/(BB/V) = V/S for throughput and (S/BB)/(V/BB) = S/V for latencies, i.e. the vLLM
    advantage in the article's convention. Published ratios have 2 decimals, so bounds propagate +-0.005 on each.
    """
    import json

    from .paths import TARGETS

    rt = json.loads(TARGETS.read_text())["ratio_tables"]
    cols = rt["_columns"]
    if key not in rt["vs_vllm"]:
        return {}
    v, sg = rt["vs_vllm"][key], rt["vs_sglang"][key]
    out = {}
    for i, c in enumerate(cols):
        out[c] = {"est": sg[i] / v[i], "lo": (sg[i] - 0.005) / (v[i] + 0.005), "hi": (sg[i] + 0.005) / (v[i] - 0.005)}
    return out
TARGET_NAME = {"output_throughput": "output_throughput", "p99_tpot_ms": "p99_tpot", "p99_e2el_ms": "p99_e2el",
               "p90_ttft_ms": "p90_ttft"}
WORKLOAD_FIELDS = ("hardware_class", "dataset", "dataset_digest", "isl", "osl", "conc", "random_range_ratio",
                   "use_chat_template", "ignore_eos", "num_prompts")


def workload_key(r: dict[str, Any]) -> tuple:
    return tuple(r.get(f) if f != "dataset" else (r.get(f) or "random") for f in WORKLOAD_FIELDS)


def _paired_log_ci(a: list[float], b: list[float], higher: bool) -> tuple[float, float] | None:
    if len(a) < 2:
        return None
    lr = np.log(np.asarray(a) / np.asarray(b)) * (1 if higher else -1)
    s = stats.summarize(lr.tolist())
    return (math.exp(s.ci_lo), math.exp(s.ci_hi)) if math.isfinite(s.ci_lo) else None


def compare_engines(rows: list[dict[str, Any]], *, a: str = "vllm", b: str = "sglang",
                    correctness: dict[str, bool] | None = None) -> list[dict[str, Any]]:
    """One output row per (workload, metric). ``correctness`` maps run_id -> probes passed."""
    correctness = correctness or {}
    by: dict[tuple, dict[str, dict[int, dict]]] = defaultdict(lambda: {a: {}, b: {}})
    invalid: dict[tuple, list[str]] = defaultdict(list)
    for r in rows:
        eng = r.get("engine")
        if eng not in (a, b):
            continue
        k = workload_key(r)
        if not r.get("valid"):
            invalid[k].append(f"{eng} {r.get('point_id')} invalid")
            continue
        by[k][eng].setdefault(int(r.get("seed") or 0), r)  # first valid run per seed
    targets = compare.load_targets()
    out = []
    for k in sorted(by, key=lambda t: tuple(str(x) for x in t)):
        sides = by[k]
        wl = dict(zip(WORKLOAD_FIELDS, k))
        seeds = sorted(set(sides[a]) & set(sides[b]))
        gate: list[str] = list(invalid.get(k, []))
        if not sides[a] or not sides[b]:
            gate.append(f"not measured: {a if not sides[a] else b}")
        for s in seeds:
            for eng in (a, b):
                rr = sides[eng][s]
                if correctness.get(rr.get("run_id")) is not True:
                    gate.append(f"correctness not passed for {eng} run {rr.get('run_id')}")
            if (sides[a][s].get("pins_target"), sides[a][s].get("pins_draft")) != \
                    (sides[b][s].get("pins_target"), sides[b][s].get("pins_draft")):
                gate.append(f"model pins differ at seed {s}")
        gate = sorted(set(gate))
        tkey = f"{wl['isl']}x{wl['conc']}" if wl["dataset"] == "random" else None
        for m, higher in METRICS.items():
            va = [sides[a][s].get(m) for s in seeds]
            vb = [sides[b][s].get(m) for s in seeds]
            ok = seeds and all(v is not None and v > 0 for v in va + vb)
            row = {**wl, "metric": m, "engine_a": a, "engine_b": b, "n_paired": len(seeds), "seeds": seeds,
                   "a_mean": float(np.mean(va)) if ok else None, "b_mean": float(np.mean(vb)) if ok else None,
                   "advantage": None, "boot_ci": None, "paired_ci": None, "gated": bool(gate), "gate_reasons": gate,
                   "evidence": "none"}
            if ok and not gate:
                num, den = (va, vb) if higher else (vb, va)
                est, lo, hi = stats.bootstrap_ratio(num, den)
                row.update(advantage=est, boot_ci=None if math.isnan(lo) else [lo, hi],
                           paired_ci=_paired_log_ci(va, vb, higher),
                           evidence="claim (n>=3)" if len(seeds) >= 3 else "descriptive (n<3)")
            tn = TARGET_NAME.get(m)
            if tkey and tn and tkey in targets and (a, b) == ("vllm", "sglang"):
                t = targets[tkey][tn]
                row["article_advantage"] = (t[a] / t[b]) if higher else (t[b] / t[a])  # from absolute values
            if tkey and (a, b) == ("vllm", "sglang"):
                imp = article_implied(tkey).get(RATIO_COL[m])
                if imp:
                    row["article_advantage_from_ratios"] = imp["est"]
                    row["article_bounds"] = [imp["lo"], imp["hi"]]
                    row.setdefault("article_advantage", imp["est"])
                    ci = row.get("boot_ci")
                    row["consistent_with_article"] = (None if not ci else bool(ci[0] <= imp["hi"] and imp["lo"] <= ci[1]))
            out.append(row)
    return out


def markdown(rows: list[dict[str, Any]], metric: str = "output_throughput") -> str:
    sel = [r for r in rows if r["metric"] == metric]
    lines = [f"| hw | dataset | ISL | conc | n | {sel[0]['engine_a'] if sel else 'A'} | {sel[0]['engine_b'] if sel else 'B'} | "
             "advantage | 95% CI (boot) | article (bounds) | consistent? | evidence / gate |",
             "|---|---|---:|---:|---:|---:|---:|---:|---|---|---|---|"]
    for r in sel:
        f = (lambda v: "not measured" if v is None else f"{v:.4g}")  # noqa: E731
        ci = f"[{r['boot_ci'][0]:.3f}, {r['boot_ci'][1]:.3f}]" if r["boot_ci"] else "–"
        art = f"{r['article_advantage']:.3f}" if r.get("article_advantage") else "–"
        if r.get("article_bounds"):
            art += f" [{r['article_bounds'][0]:.3f}, {r['article_bounds'][1]:.3f}]"
        cons = {True: "yes", False: "NO", None: "–"}[r.get("consistent_with_article")]
        ev = r["evidence"] if not r["gated"] else "GATED: " + "; ".join(r["gate_reasons"])
        lines.append(f"| {r['hardware_class']} | {r['dataset']} | {r['isl'] or 'natural'} | {r['conc']} | {r['n_paired']} | "
                     f"{f(r['a_mean'])} | {f(r['b_mean'])} | {f(r['advantage'])} | {ci} | {art} | {cons} | {ev} |")
    return "\n".join(lines)
