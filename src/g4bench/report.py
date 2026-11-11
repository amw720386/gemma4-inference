"""Per-run report: REPORT.md + throughput.svg written into the run directory.

Design rules (enforced in code, tested):
* every number shown is read from this run's results.jsonl; nothing is interpolated or invented;
* a metric that was not captured is printed as ``not measured`` (never 0 or blank);
* mock-engine runs get a banner saying they are NOT measurements and get no target comparison;
* non-exact hardware gets a PROXY banner and no target comparison;
* real-prompt arms (keys with ``@dataset``) are never compared to the article's random-token targets.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from . import compare, stats

NM = "not measured"


ARTICLE_OSL = 1024

def _f(v: Any, fmt: str = "{:.4g}") -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return NM
    return fmt.format(v)


def load_run(run_dir: Path) -> tuple[dict, dict, list[dict], dict | None]:
    man = json.loads((run_dir / "manifest.json").read_text()) if (run_dir / "manifest.json").exists() else {}
    meta = json.loads((run_dir / "metadata.json").read_text()) if (run_dir / "metadata.json").exists() else {}
    rows: list[dict] = []
    f = run_dir / "results.jsonl"
    if f.exists():
        for line in f.read_text().splitlines():
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    rows.append({"status": "corrupt-line", "valid": False, "problems": ["unparseable results line"]})
    corr = json.loads((run_dir / "correctness.json").read_text()) if (run_dir / "correctness.json").exists() else None
    return man, meta, rows, corr


def _group(rows: list[dict]) -> dict[tuple, list[dict]]:
    g: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        if r.get("valid"):
            g[(r.get("dataset") or "random", r.get("isl"), r.get("conc"))].append(r)
    return g


def _svg(series: dict[str, list[tuple[int, float, float | None, float | None]]],
         targets: dict[str, list[tuple[int, float]]], title: str) -> str:
    """Log2-x line chart; series name -> [(conc, mean, ci_lo, ci_hi)]; dashed target lines."""
    W, H, L, R, T, B = 640, 360, 70, 160, 40, 50
    pts = [p for s in series.values() for p in s] + [(c, v, None, None) for t in targets.values() for c, v in t]
    if not pts:
        return ""
    xs = sorted({p[0] for p in pts})
    ymax = max([p[1] for p in pts] + [p[3] for p in pts if p[3] is not None]) * 1.1 or 1.0
    lx = [math.log2(x) for x in xs]
    x0, x1 = min(lx), max(lx) if max(lx) > min(lx) else min(lx) + 1

    def X(c: float) -> float:
        return L + (math.log2(c) - x0) / (x1 - x0) * (W - L - R)

    def Y(v: float) -> float:
        return H - B - v / ymax * (H - T - B)

    pal = ["#2a6fdb", "#d9480f", "#2f9e44", "#7048e8", "#c2255c", "#0b7285"]
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="sans-serif" font-size="11">',
         f'<rect width="{W}" height="{H}" fill="white"/>',
         f'<text x="{L}" y="20" font-size="13">{title}</text>',
         f'<line x1="{L}" y1="{H-B}" x2="{W-R}" y2="{H-B}" stroke="#333"/>',
         f'<line x1="{L}" y1="{T}" x2="{L}" y2="{H-B}" stroke="#333"/>']
    for c in xs:
        o.append(f'<text x="{X(c):.1f}" y="{H-B+15}" text-anchor="middle">{c}</text>')
    for i in range(5):
        v = ymax * i / 4
        o.append(f'<text x="{L-6}" y="{Y(v)+4:.1f}" text-anchor="end">{v:.0f}</text>')
        o.append(f'<line x1="{L}" y1="{Y(v):.1f}" x2="{W-R}" y2="{Y(v):.1f}" stroke="#eee"/>')
    o.append(f'<text x="{(L+W-R)/2:.0f}" y="{H-12}" text-anchor="middle">concurrency (log2)</text>')
    o.append(f'<text x="16" y="{(T+H-B)/2:.0f}" transform="rotate(-90 16 {(T+H-B)/2:.0f})" text-anchor="middle">output tok/s</text>')
    ly = T
    for i, (name, s) in enumerate(sorted(series.items())):
        col = pal[i % len(pal)]
        s = sorted(s)
        o.append('<polyline fill="none" stroke="{}" stroke-width="2" points="{}"/>'.format(
            col, " ".join(f"{X(c):.1f},{Y(v):.1f}" for c, v, _, _ in s)))
        for c, v, lo, hi in s:
            if lo is not None and hi is not None:
                o.append(f'<line x1="{X(c):.1f}" y1="{Y(lo):.1f}" x2="{X(c):.1f}" y2="{Y(hi):.1f}" stroke="{col}"/>')
            o.append(f'<circle cx="{X(c):.1f}" cy="{Y(v):.1f}" r="3" fill="{col}"/>')
        o.append(f'<text x="{W-R+10}" y="{ly+10}" fill="{col}">{name}</text>')
        ly += 16
    for i, (name, t) in enumerate(sorted(targets.items())):
        col = "#868e96"
        t = sorted(t)
        o.append('<polyline fill="none" stroke="{}" stroke-dasharray="5,4" points="{}"/>'.format(
            col, " ".join(f"{X(c):.1f},{Y(v):.1f}" for c, v in t)))
        o.append(f'<text x="{W-R+10}" y="{ly+10}" fill="{col}">{name} (article)</text>')
        ly += 16
    o.append("</svg>")
    return "\n".join(o)


def build(run_dir: Path) -> tuple[str, str]:
    man, meta, rows, corr = load_run(run_dir)
    eng = (man.get("engine_config") or {})
    engine_kind = eng.get("engine") or (rows[0].get("engine") if rows else None)
    exact = bool(meta.get("exact_hardware"))
    is_mock = engine_kind == "mock"
    profiled = bool(man.get("profile")) or any(r.get("profiled") for r in rows)
    valid = [r for r in rows if r.get("valid")]
    lines = [f"# Run report: `{man.get('run_id', run_dir.name)}`", ""]
    if profiled:
        lines += ["> **PROFILED RUN (torch profiler enabled).** Benchmark numbers include profiler overhead; "
                  "see `points/*/profile_summary.md`. Not compared with the article.", ""]
    if is_mock:
        lines += ["> **MOCK ENGINE — NOT A MEASUREMENT.** CPU fake server used to test the harness. "
                  "Numbers below describe the mock, not any GPU or inference engine.", ""]
    elif not exact:
        lines += [f"> **PROXY / NON-TARGET HARDWARE** (`{meta.get('hardware_class', 'unknown')}`). "
                  "Not comparable to the article (1× RTX PRO 6000 96 GB).", ""]
    if not valid:
        lines += ["> **NOT MEASURED:** this run has no valid point-runs.", ""]
    lines += ["## Identity", "",
              f"- engine config: `{eng.get('name', NM)}` ({engine_kind or NM} {eng.get('version', '')})",
              f"- config hash: `{man.get('config_hash', NM)}`",
              f"- hardware: `{meta.get('hardware_class', NM)}`; exact article hardware: **{exact}**",
              f"- GPU: {', '.join(g.get('name', '?') + ' / power limit ' + g.get('power.limit', '?') + ' W / driver ' + g.get('driver_version', '?') for g in meta.get('gpus', [])) or 'none (CPU)'}",
              f"- git: `{(meta.get('git') or {}).get('sha', NM)}` dirty={(meta.get('git') or {}).get('dirty')}",
              f"- point-runs: {len(rows)} total, {len(valid)} valid", ""]
    if man.get("server_argv"):
        lines += ["<details><summary>server argv</summary>", "", "```", " ".join(man["server_argv"]), "```", "</details>", ""]
    if man.get("datasets"):
        lines += ["## Prompt sets", ""]
        for name, p in man["datasets"].items():
            lines.append(f"- `{name}`: {p.get('n_prompts')} prompts, digest `{str(p.get('prompt_digest'))[:16]}…`, "
                         f"InferenceX-identical: {p.get('matches_inferencex_digest')}"
                         + (f" — **{p['SUBSET_WARNING']}**" if p.get("SUBSET_WARNING") else ""))
        lines.append("")

    g = _group(rows)
    keys = sorted(g, key=lambda k: (str(k[0]), k[1] or 0, k[2] or 0))
    # The article's workload is OSL 1024; other output lengths (e.g. the smoke matrix's 256) are never compared.
    off_osl = {k for k in keys if any(r.get("osl") not in (None, ARTICLE_OSL) for r in g[k])}
    agg: dict[tuple, dict[str, Any]] = {}
    for k in keys:
        rs = g[k]
        a = {"n": len(rs), "seeds": sorted({r.get("seed") for r in rs})}
        for m in ("output_throughput", "mean_ttft_ms", "median_ttft_ms", "p90_ttft_ms", "p99_ttft_ms",
                  "mean_tpot_ms", "p50_tpot_ms", "median_tpot_ms", "p90_tpot_ms", "p99_tpot_ms", "mean_e2el_ms",
                  "p99_e2el_ms", "spec_acceptance_length", "gpu_mem_peak_mib", "gpu_power_mean_w",
                  "req_littles_law_tok_s", "req_littles_law_rel_gap", "speed_per_user_tok_s"):
            vals = [r.get(m) for r in rs if r.get(m) is not None]
            a[m] = stats.summarize(vals) if vals else None
        agg[k] = a

    def mean(a: dict, m: str) -> Any:
        s = a.get(m)
        return s.mean if s else None

    def ci(a: dict, m: str) -> str:
        s = a.get(m)
        if not s or s.n < 2 or not math.isfinite(s.ci_lo):
            return "n<2"
        return f"[{s.ci_lo:.4g}, {s.ci_hi:.4g}]"

    lines += ["## Throughput vs concurrency", "",
              "Mean over valid seeds/trials in this run; CI = Student-t 95 % across runs (needs n ≥ 2). "
              "Little's-law column is a within-run diagnostic from per-request logs.", "",
              "| dataset | ISL | conc | n | output tok/s | 95% CI | tok/s/user | Little's-law tok/s (gap) |",
              "|---|---:|---:|---:|---:|---|---:|---|"]
    for k in keys:
        a = agg[k]
        ll, gap = mean(a, "req_littles_law_tok_s"), mean(a, "req_littles_law_rel_gap")
        lines.append(f"| {k[0]} | {k[1] if k[1] else 'natural'} | {k[2]} | {a['n']} | {_f(mean(a, 'output_throughput'))} | "
                     f"{ci(a, 'output_throughput')} | {_f(mean(a, 'speed_per_user_tok_s'))} | "
                     f"{_f(ll)}{'' if gap is None else f' ({gap:+.1%})'} |")
    if not keys:
        lines.append(f"| – | – | – | 0 | {NM} | – | {NM} | {NM} |")
    lines += ["", "![throughput vs concurrency](throughput.svg)", "",
              "## Latency (ms, mean over runs of each run's statistic)", "",
              "| dataset | ISL | conc | TTFT mean | TTFT p90 | TTFT p99 | TPOT mean | TPOT p99 | E2EL mean | E2EL p99 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for k in keys:
        a = agg[k]
        lines.append(f"| {k[0]} | {k[1] or 'natural'} | {k[2]} | " + " | ".join(
            _f(mean(a, m)) for m in ("mean_ttft_ms", "p90_ttft_ms", "p99_ttft_ms", "mean_tpot_ms", "p99_tpot_ms",
                                     "mean_e2el_ms", "p99_e2el_ms")) + " |")
    lines += ["", "## Speculative decoding acceptance & resources", "",
              "| dataset | ISL | conc | acceptance length | peak GPU MiB | mean GPU W |", "|---|---:|---:|---:|---:|---:|"]
    for k in keys:
        a = agg[k]
        lines.append(f"| {k[0]} | {k[1] or 'natural'} | {k[2]} | {_f(mean(a, 'spec_acceptance_length'), '{:.3f}')} | "
                     f"{_f(mean(a, 'gpu_mem_peak_mib'), '{:.0f}')} | {_f(mean(a, 'gpu_power_mean_w'), '{:.0f}')} |")

    lines += ["", "## Effective engine configuration (parsed from server logs)", "",
              "| attention backend | spec algorithm | KV-cache tokens | max running requests | seen in |",
              "|---|---|---:|---:|---|"]
    effs = sorted({(r.get("eff_attention_backend"), r.get("eff_spec_algorithm"), r.get("eff_kv_cache_tokens"),
                    r.get("eff_max_running_requests")) for r in rows if r.get("status") != "skipped"},
                  key=lambda t: tuple(str(x) for x in t))
    for e in effs:
        n = sum(1 for r in rows if (r.get("eff_attention_backend"), r.get("eff_spec_algorithm"),
                                    r.get("eff_kv_cache_tokens"), r.get("eff_max_running_requests")) == e)
        lines.append(f"| {e[0] or NM} | {e[1] or NM} | {_f(e[2], '{:,}')} | {_f(e[3], '{:d}')} | {n} point-run(s) |")
    if not effs:
        lines.append(f"| {NM} | {NM} | {NM} | {NM} | – |")

    # ---- comparison to the article
    lines += ["", "## Comparison with article baselines", ""]
    role, adm = eng.get("role", "variant"), str(eng.get("admissibility", "c"))
    ref = {"vllm": "vllm", "sglang": "sglang"}.get(engine_kind or "") if role == "baseline" else None
    lines.append(f"Config role: **{role}**, admissibility class **{adm}** (DECISIONS D-011)."
                 + ("" if role == "baseline" else " Variants are not compared with the article's open baselines.") + "")
    targets_svg: dict[str, list[tuple[int, float]]] = {}
    if profiled:
        lines.append("Not compared: profiled run (torch profiler overhead).")
    elif adm == "c" and not is_mock:
        lines.append("Not compared: admissibility class c (not comparable).")
    elif is_mock:
        lines.append("Not compared: mock engine.")
    elif not exact:
        lines.append("Not compared: not the article's hardware (proxy run).")
    elif ref is None and role == "baseline":
        lines.append("Not compared: engine has no article counterpart.")
    elif ref is None:
        lines.append("Variant run: see `g4bench analyze` (results/summary/comparison.csv) for the comparison with "
                     "Backboard's reported numbers, labelled by admissibility class.")
    else:
        tg = compare.load_targets()
        lines += [f"Reference: article `{ref}` (and Backboard) numbers for random-token prompts. Classification uses the "
                  "pre-registered 5 % tolerance (docs/EXPERIMENTS.md); n<3 results are descriptive only.", "",
                  f"| ISL | conc | ours | n | article {ref} | ours/article | class | article Backboard | ours/Backboard |",
                  "|---:|---:|---:|---:|---:|---:|---|---:|---:|"]
        any_cmp = False
        for k in keys:
            if k[0] != "random" or k in off_osl:
                continue
            t = tg.get(f"{k[1]}x{k[2]}", {}).get("output_throughput", {})
            a = agg[k]
            s = a.get("output_throughput")
            if not s or not t:
                continue
            any_cmp = True
            cls = compare.classify(s.mean, s.ci_lo, s.ci_hi, t[ref])
            lines.append(f"| {k[1]} | {k[2]} | {s.mean:.4g} | {s.n} | {t[ref]} | {s.mean / t[ref]:.3f} | {cls} | "
                         f"{t['backboard']} | {s.mean / t['backboard']:.3f} |")
            targets_svg.setdefault(f"{ref} ISL {k[1]}", []).append((k[2], t[ref]))
        if not any_cmp and not off_osl:
            lines.append(f"| – | – | {NM} | 0 | – | – | – | – | – |")
        if any(k[0] != "random" for k in keys):
            lines += ["", "Real-prompt arms are intentionally not compared with the article (different workload)."]
        if off_osl:
            lines += ["", f"Not compared: output length differs from the article's OSL {ARTICLE_OSL} "
                          f"({len(off_osl)} workload(s), e.g. the smoke matrix)."]

    lines += ["", "## Gap attribution (first-order; acceptance vs step speed)", ""]
    if is_mock or profiled or not exact or ref is None:
        lines.append("Not computed: needs exact article hardware and a vLLM/SGLang run.")
    else:
        from . import gap

        tg = compare.load_targets()
        dt = gap.draft_tokens_for(eng)
        lines += [f"Assumptions: {gap.ASSUMPTIONS}. Article mean TPOT = 1000 / its 'speed per user'.", "",
                  f"| ISL | conc | our TPOT ms | our AL | step ms | BW eff. | AL needed for article {ref} | for Backboard | "
                  "step speed-up needed (Backboard) |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        rows_added = 0
        for k in keys:
            if k[0] != "random" or dt is None or k in off_osl:
                continue
            a = agg[k]
            spu = tg.get(f"{k[1]}x{k[2]}", {}).get("speed_per_user", {})
            r_ref = gap.attribute(tpot_ms=mean(a, "mean_tpot_ms"), al=mean(a, "spec_acceptance_length"), conc=k[2],
                                  isl=k[1], osl=1024, article_tpot_ms=1000 / spu[ref] if spu.get(ref) else None,
                                  draft_tokens=dt)
            r_bb = gap.attribute(tpot_ms=mean(a, "mean_tpot_ms"), al=mean(a, "spec_acceptance_length"), conc=k[2],
                                 isl=k[1], osl=1024, article_tpot_ms=1000 / spu["backboard"] if spu.get("backboard") else None,
                                 draft_tokens=dt)
            if r_ref["state"] != "ok":
                lines.append(f"| {k[1]} | {k[2]} | {_f(mean(a, 'mean_tpot_ms'))} | {NM} | – | – | – | – | – |")
                continue
            rows_added += 1
            flag = lambda r: "" if r.get("reachable_by_acceptance_alone", True) else " (> max AL)"  # noqa: E731
            lines.append(f"| {k[1]} | {k[2]} | {mean(a, 'mean_tpot_ms'):.3g} | {mean(a, 'spec_acceptance_length'):.3g} | "
                         f"{r_ref['step_ms']:.3g} | {r_ref['bw_efficiency']:.2f} | "
                         f"{_f(r_ref.get('al_needed_for_article'), '{:.2f}')}{flag(r_ref)} | "
                         f"{_f(r_bb.get('al_needed_for_article'), '{:.2f}')}{flag(r_bb)} | "
                         f"{_f(r_bb.get('step_speedup_needed'), '{:.2f}x')} |")
        if not rows_added and not any(True for k in keys if k[0] == "random"):
            lines.append(f"| – | – | {NM} | {NM} | – | – | – | – | – |")

    lines += ["", "## Correctness gate", ""]
    if corr is None:
        lines.append(f"{NM} (no correctness.json)")
    elif "error" in corr:
        lines.append(f"ERROR: `{corr['error'][:300]}`")
    else:
        pr = corr.get("probes", {})
        lines.append(f"Known-answer probes: {pr.get('n_ok')}/{pr.get('n')} — {'PASS' if pr.get('pass') else 'FAIL'}; "
                     f"greedy outputs captured: {len(corr.get('greedy_outputs') or [])}")

    bad = [r for r in rows if not r.get("valid")]
    lines += ["", "## Invalid / skipped point-runs", ""]
    if not bad:
        lines.append("None.")
    for r in bad:
        lines.append(f"- `{r.get('point_id', '?')}` status={r.get('status')}: {'; '.join(r.get('problems') or []) or r.get('notes')}")
    lines.append("")

    series: dict[str, list[tuple[int, float, float | None, float | None]]] = {}
    for k in keys:
        s = agg[k].get("output_throughput")
        if s:
            lo, hi = (s.ci_lo, s.ci_hi) if s.n >= 2 and math.isfinite(s.ci_lo) else (None, None)
            series.setdefault(f"{k[0]} ISL {k[1] or 'nat'}", []).append((k[2], s.mean, lo, hi))
    title = ("MOCK (not a measurement) — " if is_mock else "") + f"{eng.get('name', '')} output throughput"
    svg = _svg(series, targets_svg, title)
    if not svg:
        svg = ('<svg xmlns="http://www.w3.org/2000/svg" width="400" height="60"><rect width="400" height="60" fill="white"/>'
               '<text x="10" y="35" font-family="sans-serif">not measured: no valid point-runs</text></svg>')
    return "\n".join(lines), svg


def write(run_dir: Path) -> Path:
    md, svg = build(run_dir)
    (run_dir / "throughput.svg").write_text(svg)
    out = run_dir / "REPORT.md"
    out.write_text(md)
    return out
