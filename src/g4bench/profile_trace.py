"""Summarize a torch-profiler chrome trace written by vLLM's ``--profiler-config`` (E5g).

vLLM v0.33.0 (``vllm/profiler/wrapper.py::TorchProfilerWrapper``) writes one trace per worker with
``torch.profiler.tensorboard_trace_handler(dir, worker_name=<rank suffix>, use_gzip=...)``, i.e.
``<dir>/<worker_name>.<timestamp_ms>.pt.trace.json[.gz]``: Kineto chrome-trace JSON
(``{"traceEvents": [...]}``; GPU work has ``cat`` ``kernel`` / ``gpu_memcpy`` / ``gpu_memset``, CPU launches
``cuda_runtime`` / ``cuda_driver`` with ``args.correlation``; ``record_function`` ranges are ``user_annotation``).

Attribution used here:
* kernel class: name-based buckets (``BUCKETS``; first match wins);
* model-runner phase: with ``VLLM_CUSTOM_SCOPES_FOR_PROFILING=1`` the runner wraps its work in
  ``gpu_model_runner: <phase>`` ranges (forward / sample / draft / ...). A kernel inherits the innermost phase
  range enclosing the CPU launch (runtime event) with the same correlation id; CUDA-graph replays carry the
  ``cudaGraphLaunch`` correlation id;
* steps: the worker annotates every engine step as ``execute_context_<n>(<tok>)_generation_<m>(<tok>)``
  (``vllm/v1/worker/gpu_worker.py::annotate_profile``).

Everything is descriptive of the profiled window only (profiler overhead included); never a benchmark number.
"""

from __future__ import annotations

import gzip
import json
import re
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path
from typing import Any

TRACE_GLOBS = ("*.pt.trace.json.gz", "*.pt.trace.json")
GPU_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}
LAUNCH_CATS = {"cuda_runtime", "cuda_driver"}
PHASE_PREFIX = "gpu_model_runner: "
STEP_RE = re.compile(r"^execute_context_(\d+)\((\d+)\)_generation_(\d+)\((\d+)\)")

# (bucket, regex on the lower-cased kernel name); order matters, first match wins.
BUCKETS: list[tuple[str, re.Pattern[str]]] = [
    ("nccl", re.compile(r"nccl|all_?reduce|all_?gather|reduce_?scatter|cross_device")),
    ("memcpy", re.compile(r"memcpy|memset")),
    ("attention", re.compile(r"attention|attn|flash|fmha|flashinfer|paged|reshape_and_cache|kv_cache|"
                             r"merge_states")),
    ("spec_decode", re.compile(r"draft|eagle|mtp|rejection|spec_?decode|speculat|propose")),
    ("sampling", re.compile(r"sampl|top_?k|top_?p|argmax|multinomial|gumbel|radix_?sort|penalt|"
                            r"logits|softmax|log_softmax")),
    ("gemm", re.compile(r"gemm|gemv|cutlass|cublas|xmma|nvjet|matmul|split_?k|wgmma|hmma|scaled_mm|marlin|"
                        r"\bbmm|_mm_|tensorop")),
    ("norm_act", re.compile(r"rms_?norm|layer_?norm|norm_kernel|silu|gelu|act_and_mul|fused_add|rotary|rope")),
]
BUCKET_NAMES = [b for b, _ in BUCKETS] + ["other"]


def classify(name: str, cat: str = "kernel") -> str:
    if cat in ("gpu_memcpy", "gpu_memset"):
        return "memcpy"
    n = name.lower()
    for bucket, rx in BUCKETS:
        if rx.search(n):
            return bucket
    return "other"


def find_traces(trace_dir: Path) -> list[Path]:
    d = Path(trace_dir)
    if not d.is_dir():
        return []
    return sorted({f for g in TRACE_GLOBS for f in d.rglob(g) if f.is_file()})


def load_trace(path: Path) -> dict[str, Any]:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    return {"traceEvents": data} if isinstance(data, list) else data


def _union_us(intervals: list[tuple[float, float]]) -> float:
    total, cur_s, cur_e = 0.0, None, None
    for s, e in sorted(intervals):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        elif e > cur_e:
            cur_e = e
    if cur_e is not None:
        total += cur_e - cur_s
    return total


class _Ranges:
    """Per-thread CPU ranges; ``innermost(ts)`` = the latest-starting range that contains ``ts``."""

    def __init__(self, ranges: list[tuple[float, float, Any]]):
        self.r = sorted(ranges, key=lambda x: x[0])
        self.starts = [x[0] for x in self.r]

    def innermost(self, ts: float, max_back: int = 64) -> Any:
        i = bisect_right(self.starts, ts) - 1
        for j in range(i, max(i - max_back, -1), -1):
            s, e, v = self.r[j]
            if s <= ts <= e:
                return v
        return None


def _pct(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    return s[min(int(round(q * (len(s) - 1))), len(s) - 1)]


def summarize(trace: dict[str, Any], top_n: int = 25) -> dict[str, Any]:
    events = [e for e in trace.get("traceEvents", []) if isinstance(e, dict) and e.get("ph") == "X"]
    gpu = [e for e in events if str(e.get("cat", "")).lower() in GPU_CATS and e.get("dur") is not None]

    # CPU-side phase ranges and step ranges, per (pid, tid)
    phase_r: dict[tuple, list] = defaultdict(list)
    step_r: dict[tuple, list] = defaultdict(list)
    steps = []
    for e in events:
        if str(e.get("cat", "")).lower() != "user_annotation":
            continue
        name, ts, dur = str(e.get("name", "")), float(e["ts"]), float(e.get("dur") or 0)
        tid = (e.get("pid"), e.get("tid"))
        if name.startswith(PHASE_PREFIX):
            phase_r[tid].append((ts, ts + dur, name[len(PHASE_PREFIX):].strip()))
        m = STEP_RE.match(name)
        if m:
            st = {"i": len(steps), "ts": ts, "cpu_us": dur, "ctx_reqs": int(m[1]), "ctx_tokens": int(m[2]),
                  "gen_reqs": int(m[3]), "gen_tokens": int(m[4]), "gpu_us": 0.0}
            steps.append(st)
            step_r[tid].append((ts, ts + dur, st["i"]))
    phase_idx = {k: _Ranges(v) for k, v in phase_r.items()}
    step_idx = {k: _Ranges(v) for k, v in step_r.items()}
    corr_phase: dict[Any, str] = {}
    corr_step: dict[Any, int] = {}
    if phase_idx or step_idx:
        for e in events:
            if str(e.get("cat", "")).lower() not in LAUNCH_CATS:
                continue
            c = (e.get("args") or {}).get("correlation")
            if c is None:
                continue
            tid = (e.get("pid"), e.get("tid"))
            ts = float(e["ts"])
            if tid in phase_idx:
                ph = phase_idx[tid].innermost(ts)
                if ph is not None:
                    corr_phase[c] = ph
            if tid in step_idx:
                si = step_idx[tid].innermost(ts)
                if si is not None:
                    corr_step[c] = si

    by_name: dict[str, dict[str, Any]] = {}
    buckets: dict[str, float] = defaultdict(float)
    phases: dict[str, float] = defaultdict(float)
    phase_buckets: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    per_dev: dict[Any, list[tuple[float, float]]] = defaultdict(list)
    total = 0.0
    for e in gpu:
        cat = str(e.get("cat", "")).lower()
        name = str(e.get("name", "?"))
        dur, ts = float(e["dur"]), float(e["ts"])
        args = e.get("args") or {}
        b = classify(name, cat)
        c = args.get("correlation")
        ph = corr_phase.get(c, "unattributed") if phase_idx else "unattributed"
        total += dur
        buckets[b] += dur
        phases[ph] += dur
        phase_buckets[ph][b] += dur
        per_dev[args.get("device", e.get("pid"))].append((ts, ts + dur))
        if c in corr_step:
            steps[corr_step[c]]["gpu_us"] += dur
        k = by_name.setdefault(name, {"name": name, "bucket": b, "count": 0, "total_us": 0.0})
        k["count"] += 1
        k["total_us"] += dur

    span = sum(max(e for _, e in iv) - min(s for s, _ in iv) for iv in per_dev.values())
    busy = sum(_union_us(iv) for iv in per_dev.values())
    share = (lambda us: round(us / total, 4) if total else None)
    top = sorted(by_name.values(), key=lambda k: -k["total_us"])[:top_n]
    for k in top:
        k["mean_us"] = round(k["total_us"] / k["count"], 3)
        k["share"] = share(k["total_us"])
        k["total_us"] = round(k["total_us"], 1)

    def _step_stats(sel: list[dict]) -> dict[str, Any]:
        cpu = [s["cpu_us"] for s in sel]
        g = [s["gpu_us"] for s in sel]
        return {"n": len(sel), "cpu_ms_mean": round(sum(cpu) / len(cpu) / 1e3, 4) if cpu else None,
                "cpu_ms_p50": round(_pct(cpu, .5) / 1e3, 4) if cpu else None,
                "cpu_ms_p90": round(_pct(cpu, .9) / 1e3, 4) if cpu else None,
                "gpu_kernel_ms_mean": round(sum(g) / len(g) / 1e3, 4) if g else None,
                "gen_reqs_mean": round(sum(s["gen_reqs"] for s in sel) / len(sel), 3) if sel else None,
                "gen_tokens_mean": round(sum(s["gen_tokens"] for s in sel) / len(sel), 3) if sel else None}

    return {
        "n_events": len(events), "n_gpu_events": len(gpu), "n_kernel_names": len(by_name),
        "gpu_time_ms": round(total / 1e3, 3), "gpu_span_ms": round(span / 1e3, 3),
        "gpu_busy_ms": round(busy / 1e3, 3),
        "gpu_idle_fraction": round(1 - busy / span, 4) if span > 0 else None,
        "buckets": {b: {"ms": round(buckets[b] / 1e3, 3), "share": share(buckets[b])}
                    for b in BUCKET_NAMES if b in buckets},
        "phases_available": bool(phase_idx),
        "phases": {p: {"ms": round(us / 1e3, 3), "share": share(us)}
                   for p, us in sorted(phases.items(), key=lambda x: -x[1])},
        "phase_buckets": {p: {b: round(us / 1e3, 3) for b, us in sorted(d.items(), key=lambda x: -x[1])}
                          for p, d in phase_buckets.items()},
        "steps": {"all": _step_stats(steps),
                  "decode_only": _step_stats([s for s in steps if s["ctx_reqs"] == 0]),
                  "with_prefill": _step_stats([s for s in steps if s["ctx_reqs"] > 0])} if steps else None,
        "top_kernels": top,
    }


def markdown(s: dict[str, Any], title: str = "Profile summary") -> str:
    def f(v: Any) -> str:
        return "–" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))

    L = [f"# {title}", "",
         "> **PROFILED (torch profiler on).** Descriptive of the profiled window only; profiler overhead included. "
         "Never compare these timings, or this point's benchmark row, with the article.", ""]
    for t in s.get("traces", []):
        L.append(f"- trace `{t['file']}` ({t['bytes']:,} bytes)")
    L += ["", f"GPU span {f(s['gpu_span_ms'])} ms, busy {f(s['gpu_busy_ms'])} ms, "
              f"**idle fraction {f(s['gpu_idle_fraction'])}**; summed GPU op time {f(s['gpu_time_ms'])} ms "
              f"({s['n_gpu_events']} GPU events, {s['n_kernel_names']} distinct names).", "",
          "## By kernel class", "", "| bucket | ms | share |", "|---|---:|---:|"]
    L += [f"| {b} | {f(v['ms'])} | {f(v['share'])} |" for b, v in s["buckets"].items()]
    L += ["", "## By model-runner phase", ""]
    if s["phases_available"]:
        L += ["| phase | ms | share | top classes (ms) |", "|---|---:|---:|---|"]
        for p, v in s["phases"].items():
            top3 = ", ".join(f"{b} {ms:.1f}" for b, ms in list(s["phase_buckets"].get(p, {}).items())[:3])
            L.append(f"| {p} | {f(v['ms'])} | {f(v['share'])} | {top3} |")
    else:
        L.append("not available: no `gpu_model_runner: *` ranges (set VLLM_CUSTOM_SCOPES_FOR_PROFILING=1).")
    L += ["", "## Engine steps (`execute_context_*` annotations)", ""]
    if s.get("steps"):
        L += ["| steps | n | CPU ms mean | p50 | p90 | GPU kernel ms mean | gen reqs mean | gen tokens mean |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for k, v in s["steps"].items():
            L.append(f"| {k} | {v['n']} | {f(v['cpu_ms_mean'])} | {f(v['cpu_ms_p50'])} | {f(v['cpu_ms_p90'])} | "
                     f"{f(v['gpu_kernel_ms_mean'])} | {f(v['gen_reqs_mean'])} | {f(v['gen_tokens_mean'])} |")
    else:
        L.append("not available: no step annotations in the trace.")
    L += ["", f"## Top {len(s['top_kernels'])} GPU ops by total time", "",
          "| # | name | class | count | total ms | mean µs | share |", "|---:|---|---|---:|---:|---:|---:|"]
    for i, k in enumerate(s["top_kernels"], 1):
        n = k["name"].replace("|", "\\|")
        n = n if len(n) <= 100 else n[:97] + "..."
        L.append(f"| {i} | `{n}` | {k['bucket']} | {k['count']} | {k['total_us'] / 1e3:.3f} | "
                 f"{k['mean_us']:.1f} | {f(k['share'])} |")
    return "\n".join(L) + "\n"


def write_summary(trace_dir: Path, out_dir: Path, top_n: int = 25) -> dict[str, Any] | None:
    """Summarize the worker trace(s) under ``trace_dir`` into ``out_dir/profile_summary.{json,md}``.

    With several files (e.g. one per rank, or a CPU-only frontend trace) the one with the most GPU time is
    summarized; all are listed. Returns the summary, or None when no trace was found."""
    files = find_traces(trace_dir)
    if not files:
        return None
    best: dict[str, Any] | None = None
    listing = []
    for f in files:
        s = summarize(load_trace(f), top_n=top_n)
        listing.append({"file": f.relative_to(Path(out_dir)).as_posix() if Path(out_dir) in f.parents else f.name,
                        "bytes": f.stat().st_size, "gpu_time_ms": s["gpu_time_ms"]})
        if best is None or s["gpu_time_ms"] > best["gpu_time_ms"]:
            best = s
            best["summarized"] = listing[-1]["file"]
    assert best is not None
    best["traces"] = listing
    out = Path(out_dir)
    (out / "profile_summary.json").write_text(json.dumps(best, indent=2))
    (out / "profile_summary.md").write_text(markdown(best), encoding="utf-8")
    return best
