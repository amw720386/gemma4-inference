"""Run a matrix of benchmark points against one engine config, end to end.

Layout written per run::

    results/runs/<run_id>/
        manifest.json        engine config, matrix, pins, config hash, cost estimate, argv
        metadata.json        host / GPU / package metadata
        results.csv          one normalized row per point (results.COLUMNS)
        results.jsonl        same rows + validation problems + spec-decode deltas
        points/<point_id>/   raw InferenceX JSON, client log, server log, metrics, gpu samples

Safety: a wall-clock budget (``max_wall_s``) stops the run between points and the
server is always killed in ``finally``. Nothing here talks to a billing system.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import metadata as md
from . import promscrape, results
from .client import client_argv, client_env
from .config import EngineConfig, config_hash, load_pins, placeholder_values
from .cost import estimate
from .matrix import Point, expand, load_matrix, new_run_id
from .paths import RESULTS
from .server import GpuSampler, Server, ServerError


@dataclass
class RunOptions:
    engine: str
    matrix: str
    seeds: list[int] | None = None
    trials: int | None = None
    only_keys: list[str] | None = None
    results_root: Path = RESULTS / "runs"
    max_wall_s: float = 4 * 3600
    server_timeout_s: float = 1800
    client_timeout_s: float = 3600
    tokenizer: str | None = None          # local path or HF id; default: pinned target snapshot
    model_name: str | None = None         # served model name (default: target repo id)
    extra_server_args: list[str] | None = None
    dry_run: bool = False
    correctness: bool = True
    greedy: bool = True                   # token-level greedy-equivalence capture (correctness.greedy_run)
    greedy_reference: str | None = None   # reference JSON; default data/greedy_reference/vllm-0.33.0-nospec.json
    correctness_only: bool = False        # start the server for the first point, run the gates, no benchmark
    notes: str = ""
    profile: dict[str, Any] | None = None  # E5g: {"key": "8192x4", "warmup_s": 30, "duration_s": 4}


def resolve_tokenizer(pins: dict[str, Any]) -> str:
    """Download only tokenizer files at the pinned revision; return local dir."""
    from huggingface_hub import snapshot_download

    t = pins["model"]["target"]
    return snapshot_download(t["repo"], revision=t["revision"],
                             allow_patterns=["tokenizer*", "chat_template.jinja", "config.json",
                                             "generation_config.json", "processor_config.json"])


@dataclass
class Prepared:
    """Everything a run needs once its run dir exists (shared by ``run`` and paired runs, paired.py)."""
    opts: RunOptions
    pins: dict[str, Any]
    eng: EngineConfig
    mat: dict[str, Any]
    points: list[Point]
    meta: dict[str, Any]
    chash: str
    run_id: str
    run_dir: Path
    fresh: bool
    base_url: str
    server_argv: list[str]
    model_name: str
    prof: dict[str, Any] | None
    ds_files: dict[str, tuple[Path, dict[str, Any]]]
    env: dict[str, str]
    cublas: dict[str, Any]
    tokenizer: str | None = None
    cublas_recorded: bool = False


def prepare(opts: RunOptions, *, manifest_extra: dict[str, Any] | None = None,
            fresh_override: bool | None = None) -> Prepared:
    """Validate, create the run dir, write manifest + metadata. Raises before any server/GPU time on a bad config."""
    from . import cublas, images

    pins = load_pins()
    eng = EngineConfig.load(opts.engine)
    mat = load_matrix(opts.matrix)
    points = expand(mat, seeds=opts.seeds, trials=opts.trials, only_keys=opts.only_keys)
    if not points:
        raise SystemExit("no points selected")
    meta = md.capture()
    # A config on an image other than its home image (paired runs) gets its own hash; never set for a home-image run.
    foreign = images.foreign_image_tag(eng.engine, eng.raw.get("image_patches"), os.environ) \
        if eng.engine in ("vllm", "sglang") else None
    chash = config_hash(eng, mat, pins, foreign_image=foreign)
    run_id = new_run_id(eng.name, meta["hardware_class"], chash)
    run_dir = opts.results_root / run_id
    fresh = bool(mat.get("fresh_server_per_point", True)) if fresh_override is None else fresh_override
    est = estimate(points, fresh_server_per_point=fresh, tok_s_ref="sglang" if eng.engine == "sglang" else "vllm")
    port = eng.port
    base_url = f"http://127.0.0.1:{port}"
    values = placeholder_values(pins, int(mat.get("max_model_len", 10240)), port)
    server_argv = eng.launcher() + eng.render_args(values) + list(opts.extra_server_args or [])
    model_name = opts.model_name or pins["model"]["target"]["repo"]
    prof = profile_plan(opts.profile, eng, points, run_dir) if opts.profile else None
    if prof:
        server_argv += prof["server_argv"]

    # Fail fast (before any run dir, server or GPU time): flags must exist in the pinned engine source,
    # every real-prompt set must exist and re-verify, and a requested cuBLAS must be available (D-061).
    check_flags(eng, server_argv[len(eng.launcher()):])
    ds_files = check_datasets(points)
    cub = cublas.resolve(eng.env.get(cublas.ENV_REQUEST), os.environ, strict=not opts.dry_run)
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "datasets": {k: v[1] for k, v in ds_files.items()},
        "run_id": run_id, "config_hash": chash, "engine_config": eng.raw, "matrix": mat,
        "pins": pins, "server_argv": server_argv, "points": [p.to_dict() for p in points],
        "cost_estimate": {"usd": est.usd, "gpu_seconds": est.gpu_seconds, "summary": est.summary()},
        "options": {k: (str(v) if isinstance(v, Path) else v) for k, v in opts.__dict__.items()},
        "profile": prof,
        "runtime_image": images.runtime_image(os.environ), "foreign_image": foreign,
        "cublas_request": {k: v for k, v in cub.items() if k != "env"},
        "argv": sys.argv,
        **(manifest_extra or {}),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    (run_dir / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"[g4bench] run {run_id}: {len(points)} points; {est.summary()}", flush=True)
    env = dict(os.environ)
    env.update(eng.env)
    env.update(cub["env"])
    if prof:
        env.update(prof["server_env"])
    return Prepared(opts=opts, pins=pins, eng=eng, mat=mat, points=points, meta=meta, chash=chash, run_id=run_id,
                    run_dir=run_dir, fresh=fresh, base_url=base_url, server_argv=server_argv, model_name=model_name,
                    prof=prof, ds_files=ds_files, env=env, cublas=cub)


def start_server(prep: Prepared, pdir: Path, timeout_s: float) -> tuple[Server, float, list[str]]:
    """Start a fresh server (log in ``pdir``); record which cuBLAS its process tree mapped.

    Returns (server, start seconds, problems); problems are attached to every point measured on this server.
    Raises ServerError (the server is stopped) if it never gets healthy."""
    from . import cublas

    server = Server(prep.server_argv, env=prep.env, log_path=pdir / "server.log", base_url=prep.base_url,
                    health_path=prep.eng.health_path)
    start_s = server.start(timeout_s=timeout_s)
    problems: list[str] = []
    try:
        probe = cublas.mapped_libs(server.proc.pid) if server.proc else {"available": False, "paths": []}
        check = cublas.verify(prep.cublas, probe)
    except Exception as e:  # noqa: BLE001 - evidence collection must never kill a run
        probe, check = {"available": False, "error": repr(e)[:500], "paths": []}, {"ok": None, "why": "probe failed"}
    rec = {"request": {k: v for k, v in prep.cublas.items() if k != "env"}, "server_env_added": prep.cublas["env"],
           "probe": probe, "check": check}
    (pdir / "cublas.json").write_text(json.dumps(rec, indent=1))
    if check.get("ok") is False:
        problems.append(f"cuBLAS request not honoured: {check.get('why')}")
    if not prep.cublas_recorded:  # first server of the run: into the run metadata
        try:
            meta_path = prep.run_dir / "metadata.json"
            meta = json.loads(meta_path.read_text())
            meta["cublas"] = {"request": rec["request"], "loaded": probe.get("realpaths") or probe.get("paths"),
                              "check": check}
            meta_path.write_text(json.dumps(meta, indent=2, default=str))
            prep.cublas_recorded = True
        except (OSError, ValueError):
            pass
    return server, start_s, problems


def run_point(prep: Prepared, pdir: Path, p: Point, *, server: Server, start_s: float | None, timeout_s: float,
              problems: list[str] | None = None) -> dict[str, Any]:
    prof = prep.prof
    return _run_point(prep.run_dir, pdir, p, eng=prep.eng, base_url=prep.base_url, model=prep.model_name,
                      tokenizer=prep.tokenizer, chash=prep.chash, run_id=prep.run_id, meta=prep.meta, start_s=start_s,
                      server=server, timeout_s=timeout_s, notes=prep.opts.notes,
                      dataset_file=prep.ds_files.get(p.dataset, (None,))[0],
                      profile=prof if prof and p.point_id == prof["point_id"] else None, profiled=bool(prof),
                      extra_problems=problems)


def correctness_gate(prep: Prepared) -> None:
    _correctness(prep.run_dir, prep.base_url, prep.model_name, greedy=prep.opts.greedy,
                 reference=prep.opts.greedy_reference)


def finish(prep: Prepared) -> None:
    try:
        from . import report

        report.write(prep.run_dir)
    except Exception as e:  # noqa: BLE001 - a report bug must never lose results
        print(f"[g4bench] report generation failed: {e!r}", flush=True)


def run(opts: RunOptions) -> Path:
    prep = prepare(opts)
    run_dir, points = prep.run_dir, prep.points
    if opts.dry_run:
        print("[g4bench] dry run: server argv =", " ".join(prep.server_argv))
        return run_dir

    prep.tokenizer = opts.tokenizer or resolve_tokenizer(prep.pins)
    t_start = time.monotonic()
    server: Server | None = None
    server_problems: list[str] = []
    try:
        for i, p in enumerate(points):
            if time.monotonic() - t_start > opts.max_wall_s:
                print(f"[g4bench] wall budget {opts.max_wall_s:.0f}s exhausted; stopping before {p.point_id}")
                _record_skip(run_dir, p, "wall budget exhausted")
                continue
            pdir = run_dir / "points" / p.point_id
            pdir.mkdir(parents=True, exist_ok=True)
            start_s = None
            if server is None or prep.fresh:
                if server is not None:
                    server.stop()
                    server = None
                try:
                    server, start_s, server_problems = start_server(
                        prep, pdir, _bounded(opts.server_timeout_s, opts.max_wall_s, t_start))
                except ServerError as e:
                    _record_skip(run_dir, p, f"server failed: {e}")
                    server = None
                    continue
            if opts.correctness_only:
                correctness_gate(prep)
                break
            run_point(prep, pdir, p, server=server, start_s=start_s,
                      timeout_s=_bounded(opts.client_timeout_s, opts.max_wall_s, t_start), problems=server_problems)
            if i == 0 and opts.correctness:
                # After the first measured point, so the gate never perturbs a measurement.
                correctness_gate(prep)
            print(f"[g4bench] {i+1}/{len(points)} done {p.point_id} ({time.monotonic()-t_start:.0f}s elapsed)", flush=True)
    finally:
        if server is not None:
            server.stop()
        finish(prep)
    return run_dir


MIN_STEP_TIMEOUT_S = 30.0


def _bounded(step_timeout_s: float, max_wall_s: float, t_start: float) -> float:
    """Never let one server start / client run outlive the run's wall budget.

    On the bridge, max_wall_s ends before the Modal timeout (bridge_jobs), so a slow point is cut by us, recorded
    as invalid, and the results + report are still written - instead of Modal killing the container mid-point.
    """
    remaining = max_wall_s - (time.monotonic() - t_start)
    return max(min(step_timeout_s, remaining), MIN_STEP_TIMEOUT_S)


class DatasetError(SystemExit):
    pass


class FlagError(SystemExit):
    pass


def check_flags(eng: EngineConfig, argv: list[str]) -> None:
    if eng.engine == "mock":
        return
    from . import flagcheck

    p = flagcheck.snapshot_path(eng.engine, eng.version)
    if not p.exists():
        raise FlagError(f"no flag snapshot for {eng.engine} {eng.version}; run `g4bench flagcheck --refresh`")
    problems = flagcheck.check(eng, json.loads(p.read_text()), argv)
    if problems:
        raise FlagError("refusing to start server: " + "; ".join(problems))


def check_datasets(points: list[Point]) -> dict[str, tuple[Path, dict[str, Any]]]:
    """Map dataset name -> (verified file, provenance summary). Raises before any server starts."""
    from .datasets import dataset_path
    from .speedbench import ProvenanceError, load_prepared

    out: dict[str, tuple[Path, dict[str, Any]]] = {}
    for name in sorted({p.dataset for p in points} - {"random"}):
        f = dataset_path(name)
        if not f.exists():
            raise DatasetError(f"dataset {name!r} not prepared at {f}; run `g4bench speedbench prepare`")
        try:
            rows, prov = load_prepared(f)
        except ProvenanceError as e:
            raise DatasetError(f"dataset {name!r} failed verification: {e}") from e
        if name == "speedbench-coding" and not prov.get("matches_inferencex_digest"):
            raise DatasetError("speedbench-coding does not match InferenceX's PROMPTS_SHA256")
        out[name] = (f, {k: prov.get(k) for k in ("dataset", "n_prompts", "prompt_digest", "file_sha256",
                                                  "matches_inferencex_digest", "SUBSET_WARNING")})
    return out


def _run_point(run_dir: Path, pdir: Path, p: Point, *, eng: EngineConfig, base_url: str, model: str,
               tokenizer: str, chash: str, run_id: str, meta: dict, start_s: float | None,
               server: Server, timeout_s: float, notes: str, dataset_file: Path | None = None,
               profile: dict[str, Any] | None = None, profiled: bool = False,
               extra_problems: list[str] | None = None) -> dict[str, Any]:
    result_path = pdir / "inferencex_result.json"
    argv = client_argv(p, base_url=base_url, model=model, backend=eng.client_backend, result=result_path,
                       tokenizer=tokenizer, endpoint=eng.endpoint, dataset_file=dataset_file)
    (pdir / "client_argv.json").write_text(json.dumps(argv))
    before = promscrape.scrape(base_url)
    sampler = GpuSampler(pdir / "gpu_samples.csv")
    sampler.start()
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    prof_rec: dict[str, Any] | None = None
    with open(pdir / "client.log", "w", encoding="utf-8") as log:
        if profile is None:
            try:
                rc = subprocess.run(argv, env=client_env(dict(os.environ)), stdout=log, stderr=subprocess.STDOUT,
                                    timeout=timeout_s).returncode
            except subprocess.TimeoutExpired:
                rc = -9
        else:
            proc = subprocess.Popen(argv, env=client_env(dict(os.environ)), stdout=log, stderr=subprocess.STDOUT)
            t0 = time.monotonic()
            try:
                prof_rec = profile_window(proc, base_url, profile, deadline=t0 + timeout_s)
                rc = proc.wait(timeout=max(t0 + timeout_s - time.monotonic(), 1.0))
            except subprocess.TimeoutExpired:
                rc = -9
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
    gpu = sampler.stop()
    if prof_rec is not None:
        prof_rec["summary"] = _summarize_profile(Path(profile["trace_dir"]), pdir) if "stop" in prof_rec else None
        (pdir / "profile.json").write_text(json.dumps(prof_rec, indent=2))
    after = promscrape.scrape(base_url)
    d = promscrape.delta(before, after)
    acc = promscrape.acceptance(d)
    log_text = server.log_path.read_text(encoding="utf-8", errors="replace") if server.log_path.exists() else ""
    acc_log = promscrape.acceptance_from_log(log_text)
    (pdir / "spec_metrics.json").write_text(json.dumps({"delta": d, "acceptance": acc, "log": acc_log,
                                                         "timing": promscrape.timing(d)}, indent=2))
    from . import serverlog

    eff = serverlog.parse(log_text, eng.engine if eng.engine in ("vllm", "sglang") else None)
    (pdir / "effective_config.json").write_text(json.dumps(eff, indent=2))
    ctx = {
        "run_id": run_id, "config_hash": chash, "engine": eng.engine, "engine_config": eng.name,
        "engine_version": eng.version, "engine_role": eng.raw.get("role", "variant"),
        "admissibility": str(eng.raw.get("admissibility", "c")), "hardware_class": meta["hardware_class"],
        "gpu_name": (meta["gpus"][0]["name"] if meta["gpus"] else None), "gpu_count": len(meta["gpus"]),
        "exact_hardware": meta["exact_hardware"], **{k: v for k, v in p.to_dict().items()},
        "spec_acceptance_length": acc.get("acceptance_length") or acc_log.get("acceptance_length_log_mean"),
        "spec_draft_acceptance_rate": acc.get("draft_acceptance_rate"),
        "server_start_s": start_s, "started_utc": started, "notes": notes, **gpu,
        "eff_attention_backend": eff.get("attention_backend") or eff.get("gemma4_backend_decision"),
        "eff_spec_algorithm": eff.get("speculative_algorithm"),
        "eff_kv_cache_tokens": eff.get("kv_cache_tokens"),
        "eff_max_running_requests": eff.get("max_running_requests"),
    }
    if profiled:
        ctx["notes"] = (notes + " " if notes else "") + "[PROFILED: torch profiler enabled; not comparable]"
    if result_path.exists():
        raw = json.loads(result_path.read_text())
        rec = results.normalize(raw, context=ctx)
        problems = results.validate(rec, osl=p.osl, range_ratio=p.random_range_ratio, ignore_eos=p.ignore_eos,
                                    real_prompts=p.dataset != "random")
    else:
        rec = results.normalize({}, context=ctx)
        rec["status"] = "no-result"
        problems = [f"client rc={rc}, no result file"]
    if rc != 0:
        problems.append(f"client exit code {rc}")
    problems.extend(extra_problems or [])
    rec["valid"] = not problems
    rec["problems"] = problems
    if profiled:
        rec["profiled"] = True  # analyze/report never compare these rows (profiler overhead)
    results.append_csv(run_dir / "results.csv", rec)
    results.append_jsonl(run_dir / "results.jsonl", rec)
    return rec


def _correctness(run_dir: Path, base_url: str, model: str, *, greedy: bool = True,
                 reference: str | None = None) -> None:
    """Probes + token-level greedy capture; compared in-job with the reference file when one is present."""
    from . import correctness

    out: dict[str, Any] = {}
    try:
        out["probes"] = correctness.probes(base_url, model)
        if greedy:
            out["greedy"] = correctness.greedy_run(base_url, model)
            ref = correctness.load_reference(reference)
            if ref is not None:
                out["greedy_compare"] = {"reference": ref.get("reference"),
                                         **correctness.compare_greedy(ref, out["greedy"])}
            else:
                out["greedy_compare"] = {"pass": None, "error": f"no reference at "
                                         f"{reference or correctness.reference_path()}"}
    except Exception as e:  # noqa: BLE001 - recorded, and the run is flagged by analysis
        out["error"] = repr(e)[:2000]
    (run_dir / "correctness.json").write_text(json.dumps(out, indent=1))
    msg = f"[g4bench] correctness: {out.get('probes', {}).get('n_ok')}/{len(correctness.PROBES)} probes ok"
    if "greedy_compare" in out:
        msg += "; " + correctness.summary_line(out["greedy_compare"])
    print(msg + (f" ERROR {out['error'][:200]}" if "error" in out else ""), flush=True)


def _record_skip(run_dir: Path, p: Point, why: str) -> None:
    rec = {c: None for c in results.COLUMNS}
    rec.update(p.to_dict())
    rec.update({"schema_version": results.SCHEMA_VERSION, "status": "skipped", "notes": why,
                "valid": False, "problems": [why]})
    results.append_csv(run_dir / "results.csv", rec)
    results.append_jsonl(run_dir / "results.jsonl", rec)


# ---------------------------------------------------------------- E5g torch-profiler window
PROFILE_DEFAULTS = {"warmup_s": 30.0, "duration_s": 4.0, "stop_timeout_s": 600.0, "top_n": 25}


def vllm_profiler_config(trace_dir: Path) -> dict[str, Any]:
    """``--profiler-config`` for vLLM v0.33.0 (``vllm/config/profiler.py::ProfilerConfig``).

    torch profiler, CPU+CUDA activities (CPU is needed for launch->kernel correlation and step annotations),
    no Python stacks (trace size), gzip, no frontend (AsyncLLM CPU) trace, no key_averages table on stop."""
    return {"profiler": "torch", "torch_profiler_dir": str(Path(trace_dir).resolve()),
            "torch_profiler_activities": ["CPU", "CUDA"], "torch_profiler_with_stack": False,
            "torch_profiler_use_gzip": True, "torch_profiler_dump_cuda_time_total": False,
            "ignore_frontend": True}


def profile_plan(spec: dict[str, Any], eng: EngineConfig, points: list[Point], run_dir: Path) -> dict[str, Any]:
    """Resolve a profile spec: which point is profiled, and the exact server argv / env additions."""
    spec = {**PROFILE_DEFAULTS, **{k: v for k, v in spec.items() if v is not None}}
    key = spec.get("key")
    match = [p for p in points if p.key == key]
    if not match:
        raise SystemExit(f"profile key {key!r} is not among the selected points "
                         f"({sorted({p.key for p in points})})")
    if float(spec["duration_s"]) <= 0 or float(spec["warmup_s"]) < 0:
        raise SystemExit("profile duration_s must be > 0 and warmup_s >= 0")
    pid = match[0].point_id
    trace_dir = (Path(run_dir) / "points" / pid / "torch_trace").resolve()
    if eng.engine == "vllm":
        argv = ["--profiler-config", json.dumps(vllm_profiler_config(trace_dir), separators=(",", ":"))]
        env = {"VLLM_CUSTOM_SCOPES_FOR_PROFILING": "1"}  # gpu_model_runner: forward/sample/draft ranges
    elif eng.engine == "mock":
        argv, env = ["--profile-dir", str(trace_dir)], {}
    else:
        raise SystemExit(f"profiling is implemented for vLLM (and the mock) only, not {eng.engine!r}")
    return {"key": key, "point_id": pid, "trace_dir": str(trace_dir), "server_argv": argv, "server_env": env,
            **{k: (int(spec[k]) if k == "top_n" else float(spec[k])) for k in PROFILE_DEFAULTS}}


def _post(url: str, timeout_s: float) -> dict[str, Any]:
    import urllib.error
    import urllib.request

    t = time.monotonic()
    req = urllib.request.Request(url, data=b"{}", method="POST", headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            status, err = r.status, None
    except urllib.error.HTTPError as e:
        status, err = e.code, repr(e)[:500]
    except Exception as e:  # noqa: BLE001 - recorded in profile.json
        status, err = None, repr(e)[:500]
    return {"status": status, "error": err, "latency_s": round(time.monotonic() - t, 3)}


def _sleep_while_running(proc: subprocess.Popen, seconds: float, deadline: float) -> bool:
    """Sleep up to ``seconds`` (never past ``deadline``); False if the client exited or the deadline hit."""
    end = min(time.monotonic() + seconds, deadline)
    while time.monotonic() < end:
        if proc.poll() is not None:
            return False
        time.sleep(min(0.2, max(end - time.monotonic(), 0)))
    return proc.poll() is None and time.monotonic() < deadline


def profile_window(proc: subprocess.Popen, base_url: str, prof: dict[str, Any], deadline: float) -> dict[str, Any]:
    """While the client runs: wait ``warmup_s``, POST /start_profile, wait ``duration_s``, POST /stop_profile.

    /stop_profile is always sent once /start_profile was (even if the client finished early)."""
    rec: dict[str, Any] = {"key": prof["key"], "point_id": prof["point_id"], "warmup_s": prof["warmup_s"],
                           "duration_s": prof["duration_s"], "server_argv_added": prof["server_argv"],
                           "server_env_added": prof["server_env"]}
    t0 = time.monotonic()
    if not _sleep_while_running(proc, prof["warmup_s"], deadline):
        rec["skipped"] = "client finished (or deadline) before the warmup ended; no profile taken"
        return rec
    rec["start_at_s"] = round(time.monotonic() - t0, 3)
    rec["start"] = _post(base_url + "/start_profile", 120)
    rec["client_running_at_stop"] = _sleep_while_running(proc, prof["duration_s"], deadline)
    rec["stop_at_s"] = round(time.monotonic() - t0, 3)
    rec["stop"] = _post(base_url + "/stop_profile", prof["stop_timeout_s"])
    rec["stop_returned_at_s"] = round(time.monotonic() - t0, 3)
    return rec


def _summarize_profile(trace_dir: Path, pdir: Path, wait_s: float = 60.0, poll_s: float = 2.0) -> dict[str, Any]:
    """Wait (bounded) for the trace file(s) to exist and stop growing, then write profile_summary.{json,md}."""
    from . import profile_trace

    end = time.monotonic() + wait_s
    last = None
    while time.monotonic() < end:
        files = profile_trace.find_traces(trace_dir)
        sizes = tuple(f.stat().st_size for f in files)
        if files and sizes == last:
            break
        last = sizes
        time.sleep(poll_s)
    try:
        s = profile_trace.write_summary(trace_dir, pdir, top_n=int(PROFILE_DEFAULTS["top_n"]))
    except Exception as e:  # noqa: BLE001 - a parser bug must never lose the run
        return {"error": repr(e)[:2000]}
    if s is None:
        return {"error": f"no *.pt.trace.json[.gz] under {trace_dir}"}
    return {"traces": s["traces"], "gpu_idle_fraction": s["gpu_idle_fraction"], "buckets": s["buckets"]}
