"""Collect all runs, aggregate across seeds/trials, compare to article targets, write reports."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from . import compare, stats
from .paths import RESULTS

AGG_METRICS = ["output_throughput", "mean_ttft_ms", "median_ttft_ms", "p90_ttft_ms", "p99_ttft_ms",
               "mean_tpot_ms", "median_tpot_ms", "p90_tpot_ms", "p99_tpot_ms", "mean_e2el_ms",
               "median_e2el_ms", "p90_e2el_ms", "p99_e2el_ms", "speed_per_user_tok_s",
               "spec_acceptance_length", "failure_rate", "gpu_mem_peak_mib", "gpu_power_mean_w",
               "req_littles_law_rel_gap"]


def _cfg_raw(name: str | None) -> dict[str, Any]:
    from .config import EngineConfig

    try:
        return EngineConfig.load(name).raw if name else {}
    except (FileNotFoundError, ValueError):
        return {}


def _role_from_config(name: str | None) -> str:
    return _cfg_raw(name).get("role", "variant")


def _adm_from_config(name: str | None) -> str:
    return str(_cfg_raw(name).get("admissibility", "c"))


def run_context(run_dir: Path) -> dict[str, Any]:
    """Run-level facts every row inherits: model pins (manifest) and correctness gate (correctness.json)."""
    ctx: dict[str, Any] = {"pins_target": None, "pins_draft": None, "correctness_pass": None,
                           "greedy_equivalent": None}
    man = run_dir / "manifest.json"
    if man.exists():
        try:
            pins = (json.loads(man.read_text()).get("pins") or {}).get("model") or {}
            ctx["pins_target"] = (pins.get("target") or {}).get("revision")
            ctx["pins_draft"] = (pins.get("draft") or {}).get("revision")
        except json.JSONDecodeError:
            pass
    corr = run_dir / "correctness.json"
    if corr.exists():
        try:
            cj = json.loads(corr.read_text())
            ctx["correctness_pass"] = bool((cj.get("probes") or {}).get("pass"))
            # Token-level greedy gate vs the BF16 no-spec reference: True/False, None = not measured.
            gc = cj.get("greedy_compare") or {}
            ctx["greedy_equivalent"] = gc.get("pass") if isinstance(gc.get("pass"), bool) else None
        except json.JSONDecodeError:
            ctx["correctness_pass"] = False
    return ctx


def load_all(root: Path = RESULTS / "runs", include_mock: bool = False) -> list[dict[str, Any]]:
    rows = []
    for f in sorted(root.glob("*/results.jsonl")):
        ctx = run_context(f.parent)
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue  # reported by the per-run report; never aggregated
            if r.get("engine") == "mock" and not include_mock:
                continue
            if r.get("profiled"):  # E5g torch-profiler runs: profiler overhead, never aggregated or compared
                continue
            for k, v in ctx.items():
                r.setdefault(k, v)
            r.setdefault("run_id", f.parent.name)
            rows.append(r)
    return rows


def reference_engine(engine: str | None) -> str | None:
    return {"vllm": "vllm", "sglang": "sglang"}.get(engine or "")


def run(root: Path = RESULTS / "runs", out: Path = RESULTS / "summary", include_mock: bool = False,
        pairs_root: Path | None = None) -> dict[str, Any]:
    rows = load_all(root, include_mock)
    pairs = paired_summary(load_pairs(pairs_root or Path(root).parent / "pairs", include_mock))
    out.mkdir(parents=True, exist_ok=True)
    for r in rows:  # schema v1 rows predate the dataset column
        r.setdefault("dataset", "random")
    agg = stats.aggregate(rows, AGG_METRICS, keys=("engine", "engine_config", "config_hash", "hardware_class",
                                                    "exact_hardware", "dataset", "isl", "osl", "conc"))
    _write_csv(out / "aggregate.csv", agg)
    comps: list[dict[str, Any]] = []
    roles = {r.get("engine_config"): (r.get("engine_role") or _role_from_config(r.get("engine_config")),
                                      r.get("admissibility") or _adm_from_config(r.get("engine_config"))) for r in rows}
    for r in agg:
        role, adm = roles.get(r.get("engine_config"), ("variant", "c"))
        # Only the article's own workload (random tokens, OSL 1024, exact hardware) is comparable at all.
        if r.get("osl") != 1024 or r.get("dataset") != "random" or not r.get("exact_hardware") or adm == "c":
            continue
        refs = ["backboard"] + ([reference_engine(r.get("engine"))] if role == "baseline" else [])
        for ref in [x for x in refs if x]:
            for c in compare.compare([r], ref):
                c.update(engine_role=role, admissibility=adm)
                comps.append(c)
    _write_csv(out / "comparison.csv", comps)
    best = best_vs_backboard(agg, {k: v[1] for k, v in roles.items()})
    _write_csv(out / "best_vs_backboard.csv", best)
    md = ["# Results summary (auto-generated by `g4bench analyze`)", "",
          f"Valid point-runs: {sum(1 for r in rows if r.get('valid'))} / {len(rows)} total.", ""]
    if not rows:
        md.append("**No GPU measurements yet.** Nothing below is a measured result.")
    for ref in ("vllm", "sglang", "backboard"):
        sel = [c for c in comps if c["reference"] == ref]
        if sel:
            md += [f"## Output throughput vs article `{ref}`", "", compare.markdown_table(sel), ""]
    md += best_vs_backboard_markdown(best)
    _write_csv(out / "paired.csv", pairs)
    md += paired_markdown(pairs)
    from .engine_compare import compare_engines, markdown as ec_md

    corr = {r.get("run_id"): r.get("correctness_pass") for r in rows}
    ec = compare_engines([r for r in rows if roles.get(r.get("engine_config"), ("variant",))[0] == "baseline"],
                         correctness=corr)
    _write_csv(out / "engine_comparison.csv", ec)
    md += ["## vLLM vs SGLang at matched load (measured; advantage > 1 = vLLM better)", ""]
    md += [ec_md(ec) if ec else "not measured: no vLLM/SGLang pairs yet.", ""]
    (out / "SUMMARY.md").write_text("\n".join(md) + "\n")
    return {"rows": len(rows), "aggregates": len(agg), "comparisons": len(comps), "engine_pairs": len(ec),
            "paired": len(pairs)}


# ------------------------------------------------------------------------------- same-GPU paired runs (D-061)
PAIR_METRICS = ("output_throughput", "mean_ttft_ms", "p90_ttft_ms", "p99_ttft_ms", "mean_tpot_ms", "p99_tpot_ms",
                "mean_e2el_ms", "p99_e2el_ms")


def load_pairs(root: Path = RESULTS / "pairs", include_mock: bool = False) -> list[dict[str, Any]]:
    docs = []
    for f in sorted(Path(root).glob("*/paired.json")):
        try:
            d = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not include_mock and "mock" in (d.get("engine_kinds") or {}).values():
            continue
        docs.append(d)
    return docs


def _log_ratio_stats(xs: list[float]) -> dict[str, Any]:
    """Geometric mean of ratios and a paired-t 95 % CI: Student t on log(B/A) per pair, back-transformed."""
    logs = [math.log(x) for x in xs if x and x > 0 and math.isfinite(x)]
    n = len(logs)
    if not n:
        return {"n": 0, "geomean": None, "ci_lo": None, "ci_hi": None, "sd_log": None}
    m = sum(logs) / n
    if n < 2:
        return {"n": 1, "geomean": math.exp(m), "ci_lo": None, "ci_hi": None, "sd_log": None}
    sd = math.sqrt(sum((x - m) ** 2 for x in logs) / (n - 1))
    half = stats.t975(n - 1) * sd / math.sqrt(n)
    return {"n": n, "geomean": math.exp(m), "ci_lo": math.exp(m - half), "ci_hi": math.exp(m + half), "sd_log": sd}


def paired_summary(docs: list[dict[str, Any]], targets: dict | None = None) -> list[dict[str, Any]]:
    """One row per (A, B, workload, hardware class) over every valid same-GPU pair in ``results/pairs``.

    ``ratio``: B/A output throughput, geometric mean over pairs, with a paired-t 95 % CI (t on the per-pair log
    ratios; each pair shares one GPU, so the GPU-to-GPU spread cancels and only within-GPU noise remains). Latency
    metrics are summarized the same way (``<metric>_ratio``; < 1 = B faster).

    ``candidate_vs_backboard`` (only when A is the article's stock recipe ``BASELINE_CONFIG`` on the article's own
    workload: random tokens, OSL 1024)::

        candidate_vs_backboard = (B/A)_paired x (article_vLLM / Backboard)

    i.e. our measured speedup of the candidate over the stock vLLM recipe on identical hardware, times the article's
    own published ratio of that same stock recipe to Backboard on Backboard's hardware. The article's two numbers
    come from one machine, so their ratio carries no GPU-to-GPU term from our side; ours carries none either. What it
    assumes: our stock-recipe run behaves like the article's (same image digest, flags, workload - checked separately by
    the unpaired baseline comparison), and the speedup B/A transfers to the article's GPU. The CI is ours only (the
    article publishes n=1 point estimates without error bars), so it understates the total uncertainty.
    """
    targets = targets or compare.load_targets()
    groups: dict[tuple, list[tuple[dict, dict]]] = {}
    for d in docs:
        if (d.get("gpu") or {}).get("same_gpu") is False:
            continue
        a, b = d["engine_configs"]["A"], d["engine_configs"]["B"]
        for p in d.get("pairs") or []:
            r = p.get("ratios_b_over_a") or {}
            if not p.get("valid") or p.get("same_gpu_as_job_start") is False or not r.get("output_throughput"):
                continue
            k = (a, b, p.get("key"), p.get("osl"), p.get("dataset"), d.get("hardware_class"),
                 bool(d.get("exact_hardware")))
            groups.setdefault(k, []).append((d, p))
    out = []
    for (a, b, key, osl, ds, hw, exact), items in sorted(groups.items(), key=lambda kv: tuple(map(str, kv[0]))):
        row: dict[str, Any] = {"engine_a": a, "engine_b": b, "workload": key, "osl": osl, "dataset": ds,
                               "hardware_class": hw, "exact_hardware": exact,
                               "n_pairs": len(items), "n_jobs": len({d["pair_id"] for d, _ in items}),
                               "n_gpus": len({tuple(p.get("gpu_uuid") or []) for _, p in items})}
        s = _log_ratio_stats([p["ratios_b_over_a"]["output_throughput"] for _, p in items])
        row.update(ratio=s["geomean"], ratio_ci_lo=s["ci_lo"], ratio_ci_hi=s["ci_hi"], ratio_sd_log=s["sd_log"])
        for m in PAIR_METRICS[1:]:
            row[f"{m}_ratio"] = _log_ratio_stats([(p["ratios_b_over_a"] or {}).get(m) for _, p in items])["geomean"]
        tgt = targets.get(key or "", {}).get("output_throughput", {})
        vllm, bb = tgt.get("vllm"), tgt.get("backboard")
        eligible = a == BASELINE_CONFIG and osl == 1024 and ds == "random" and vllm and bb
        f = (vllm / bb) if eligible else None
        row.update(article_vllm=vllm if eligible else None, backboard=bb if eligible else None,
                   article_vllm_over_backboard=f,
                   candidate_vs_backboard=(s["geomean"] * f) if f and s["geomean"] else None,
                   candidate_vs_backboard_ci_lo=(s["ci_lo"] * f) if f and s["ci_lo"] else None,
                   candidate_vs_backboard_ci_hi=(s["ci_hi"] * f) if f and s["ci_hi"] else None)
        out.append(row)
    return out


def paired_markdown(rows: list[dict[str, Any]]) -> list[str]:
    def num(v: float | None, fmt: str = ".3f") -> str:
        return format(v, fmt) if v is not None else "-"

    md = ["## Same-GPU paired comparisons (D-061)", ""]
    if not rows:
        return md + ["not measured: no paired runs yet.", ""]
    md += ["B/A = candidate / reference on the same GPU in one container (geometric mean over pairs, paired-t 95 % "
           "CI on log ratios). B vs Backboard = (B/A) x (article vLLM / Backboard), only when A is the article's "
           f"recipe `{BASELINE_CONFIG}`; its CI covers our pairs only. n < 3 pairs: descriptive.", "",
           "| A | B | workload | pairs | GPUs | B/A tok/s | 95 % CI | B/A p99 TPOT | B/A mean TTFT | "
           "est. B vs Backboard | 95 % CI |", "|---|---|---|---:|---:|---:|---|---:|---:|---:|---|"]
    for r in rows:
        ci = f"{num(r['ratio_ci_lo'])}-{num(r['ratio_ci_hi'])}" if r["ratio_ci_lo"] is not None else "-"
        cb = (f"{num(r['candidate_vs_backboard_ci_lo'])}-{num(r['candidate_vs_backboard_ci_hi'])}"
              if r["candidate_vs_backboard_ci_lo"] is not None else "-")
        md.append(f"| {r['engine_a']} | {r['engine_b']} | {r['workload']} | {r['n_pairs']} | {r['n_gpus']} | "
                  f"{num(r['ratio'])} | {ci} | {num(r.get('p99_tpot_ms_ratio'))} | {num(r.get('mean_ttft_ms_ratio'))} | "
                  f"{num(r['candidate_vs_backboard'])} | {cb} |")
    return md + [""]


ARTICLE_WORKLOADS = [(isl, conc) for isl in (1024, 8192) for conc in (1, 4, 16, 64)]
BASELINE_CONFIG = "vllm-0.33.0-mtp4"
WITHIN_TOL = 0.95  # "within 5 % of Backboard": ours / Backboard >= 0.95


def _article_point(r: dict[str, Any], isl: int, conc: int) -> bool:
    """True when an aggregate row is the article's workload (random, OSL 1024) on exact hardware."""
    return (r.get("isl") == isl and r.get("conc") == conc and r.get("osl") == 1024
            and r.get("dataset") == "random" and r.get("exact_hardware") in (True, "True", "true")
            and r.get("engine") != "mock" and r.get("output_throughput__mean") is not None)


def _ratio(mean: float | None, target: float | None) -> float | None:
    return mean / target if (mean is not None and target) else None


def best_vs_backboard(agg: list[dict[str, Any]], adm_of: dict[str | None, str],
                      targets: dict | None = None) -> list[dict[str, Any]]:
    """Per article workload: the class-(a) config with the highest mean output throughput vs Backboard.

    ``adm_of`` maps engine_config -> admissibility (D-011); only "a" is eligible. The baseline
    (``BASELINE_CONFIG``) row is reported for reference whether or not it is the best.
    """
    targets = targets or compare.load_targets()
    out = []
    for isl, conc in ARTICLE_WORKLOADS:
        key = f"{isl}x{conc}"
        bb = targets.get(key, {}).get("output_throughput", {}).get("backboard")
        pts = [r for r in agg if _article_point(r, isl, conc) and adm_of.get(r.get("engine_config")) == "a"]
        best = max(pts, key=lambda r: r["output_throughput__mean"], default=None)
        base = max((r for r in pts if r.get("engine_config") == BASELINE_CONFIG),
                   key=lambda r: (int(r.get("output_throughput__n") or 0), r["output_throughput__mean"]), default=None)
        b_mean = best["output_throughput__mean"] if best else None
        base_mean = base["output_throughput__mean"] if base else None
        out.append({"workload": key, "isl": isl, "conc": conc,
                    "best_config": best.get("engine_config") if best else None,
                    "hardware_class": best.get("hardware_class") if best else None,
                    "mean": b_mean, "n": int(best.get("output_throughput__n") or 0) if best else 0,
                    "backboard": bb, "ratio": _ratio(b_mean, bb),
                    "baseline_mean": base_mean, "baseline_ratio": _ratio(base_mean, bb),
                    "measured": best is not None})
    return out


def best_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Count of measured workloads within 5 % of Backboard and the geometric-mean ratio over them."""
    ratios = [r["ratio"] for r in rows if r.get("ratio")]
    geo = math.exp(sum(math.log(x) for x in ratios) / len(ratios)) if ratios else None
    return {"measured": len(ratios), "total": len(rows),
            "within_5pct": sum(1 for x in ratios if x >= WITHIN_TOL), "geomean_ratio": geo}


def best_vs_backboard_markdown(rows: list[dict[str, Any]]) -> list[str]:
    """SUMMARY.md section lines for :func:`best_vs_backboard`."""
    def num(v: float | None, fmt: str) -> str:
        return format(v, fmt) if v is not None else "-"

    md = ["## Best lossless open-engine config vs Backboard", "",
          "Highest mean output tok/s per article workload among admissibility-(a) configs (D-011) on exact "
          "hardware; random dataset, OSL 1024. Baseline = article recipe `" + BASELINE_CONFIG + "`. "
          "Rows with n < 3 are **descriptive only**, not claims.", "",
          "| workload | best config | ours tok/s | n | Backboard tok/s | ours/Backboard | baseline tok/s | "
          "baseline/Backboard | evidence |",
          "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for r in rows:
        if not r["measured"]:
            md.append(f"| {r['workload']} | not measured | - | 0 | {r['backboard']} | - | - | - | - |")
            continue
        ev = "claim-eligible (n>=3)" if r["n"] >= 3 else "descriptive (n<3)"
        md.append(f"| {r['workload']} | {r['best_config']} | {r['mean']:.1f} | {r['n']} | {r['backboard']} | "
                  f"{num(r['ratio'], '.3f')} | {num(r['baseline_mean'], '.1f')} | "
                  f"{num(r['baseline_ratio'], '.3f')} | {ev} |")
    s = best_summary(rows)
    geo = num(s["geomean_ratio"], ".3f")
    md += ["", f"Within 5 % of Backboard (ratio >= {WITHIN_TOL}): **{s['within_5pct']} / {s['measured']}** measured "
               f"workloads ({s['total']} total); geometric-mean ratio over measured workloads: **{geo}**.", ""]
    return md


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if (isinstance(v, float) and math.isnan(v)) else v) for k, v in r.items()})
