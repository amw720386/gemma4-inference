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
    notes: str = ""


def resolve_tokenizer(pins: dict[str, Any]) -> str:
    """Download only tokenizer files at the pinned revision; return local dir."""
    from huggingface_hub import snapshot_download

    t = pins["model"]["target"]
    return snapshot_download(t["repo"], revision=t["revision"],
                             allow_patterns=["tokenizer*", "chat_template.jinja", "config.json",
                                             "generation_config.json", "processor_config.json"])


def run(opts: RunOptions) -> Path:
    pins = load_pins()
    eng = EngineConfig.load(opts.engine)
    mat = load_matrix(opts.matrix)
    points = expand(mat, seeds=opts.seeds, trials=opts.trials, only_keys=opts.only_keys)
    if not points:
        raise SystemExit("no points selected")
    meta = md.capture()
    chash = config_hash(eng, mat, pins)
    run_id = new_run_id(eng.name, meta["hardware_class"], chash)
    run_dir = opts.results_root / run_id
    fresh = bool(mat.get("fresh_server_per_point", True))
    est = estimate(points, fresh_server_per_point=fresh)
    port = eng.port
    base_url = f"http://127.0.0.1:{port}"
    values = placeholder_values(pins, int(mat.get("max_model_len", 10240)), port)
    server_argv = eng.launcher() + eng.render_args(values) + list(opts.extra_server_args or [])
    model_name = opts.model_name or pins["model"]["target"]["repo"]

    run_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "run_id": run_id, "config_hash": chash, "engine_config": eng.raw, "matrix": mat,
        "pins": pins, "server_argv": server_argv, "points": [p.to_dict() for p in points],
        "cost_estimate": {"usd": est.usd, "gpu_seconds": est.gpu_seconds, "summary": est.summary()},
        "options": {k: (str(v) if isinstance(v, Path) else v) for k, v in opts.__dict__.items()},
        "argv": sys.argv,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    (run_dir / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"[g4bench] run {run_id}: {len(points)} points; {est.summary()}", flush=True)
    if opts.dry_run:
        print("[g4bench] dry run: server argv =", " ".join(server_argv))
        return run_dir

    tokenizer = opts.tokenizer or resolve_tokenizer(pins)
    env = dict(os.environ)
    env.update(eng.env)
    t_start = time.monotonic()
    server: Server | None = None
    try:
        for i, p in enumerate(points):
            if time.monotonic() - t_start > opts.max_wall_s:
                print(f"[g4bench] wall budget {opts.max_wall_s:.0f}s exhausted; stopping before {p.point_id}")
                _record_skip(run_dir, p, "wall budget exhausted")
                continue
            pdir = run_dir / "points" / p.point_id
            pdir.mkdir(parents=True, exist_ok=True)
            start_s = None
            if server is None or fresh:
                if server is not None:
                    server.stop()
                server = Server(server_argv, env=env, log_path=pdir / "server.log", base_url=base_url,
                                health_path=eng.health_path)
                try:
                    start_s = server.start(timeout_s=opts.server_timeout_s)
                except ServerError as e:
                    _record_skip(run_dir, p, f"server failed: {e}")
                    server = None
                    continue
            _run_point(run_dir, pdir, p, eng=eng, base_url=base_url, model=model_name, tokenizer=tokenizer,
                       chash=chash, run_id=run_id, meta=meta, start_s=start_s, server=server,
                       timeout_s=opts.client_timeout_s, notes=opts.notes)
            print(f"[g4bench] {i+1}/{len(points)} done {p.point_id} ({time.monotonic()-t_start:.0f}s elapsed)", flush=True)
    finally:
        if server is not None:
            server.stop()
    return run_dir


def _run_point(run_dir: Path, pdir: Path, p: Point, *, eng: EngineConfig, base_url: str, model: str,
               tokenizer: str, chash: str, run_id: str, meta: dict, start_s: float | None,
               server: Server, timeout_s: float, notes: str) -> dict[str, Any]:
    result_path = pdir / "inferencex_result.json"
    argv = client_argv(p, base_url=base_url, model=model, backend=eng.client_backend, result=result_path,
                       tokenizer=tokenizer, endpoint=eng.endpoint)
    (pdir / "client_argv.json").write_text(json.dumps(argv))
    before = promscrape.scrape(base_url)
    sampler = GpuSampler(pdir / "gpu_samples.csv")
    sampler.start()
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(pdir / "client.log", "w", encoding="utf-8") as log:
        try:
            rc = subprocess.run(argv, env=client_env(dict(os.environ)), stdout=log, stderr=subprocess.STDOUT,
                                timeout=timeout_s).returncode
        except subprocess.TimeoutExpired:
            rc = -9
    gpu = sampler.stop()
    after = promscrape.scrape(base_url)
    d = promscrape.delta(before, after)
    acc = promscrape.acceptance(d)
    log_text = server.log_path.read_text(encoding="utf-8", errors="replace") if server.log_path.exists() else ""
    acc_log = promscrape.acceptance_from_log(log_text)
    (pdir / "spec_metrics.json").write_text(json.dumps({"delta": d, "acceptance": acc, "log": acc_log}, indent=2))
    ctx = {
        "run_id": run_id, "config_hash": chash, "engine": eng.engine, "engine_config": eng.name,
        "engine_version": eng.version, "hardware_class": meta["hardware_class"],
        "gpu_name": (meta["gpus"][0]["name"] if meta["gpus"] else None), "gpu_count": len(meta["gpus"]),
        "exact_hardware": meta["exact_hardware"], **{k: v for k, v in p.to_dict().items()},
        "spec_acceptance_length": acc.get("acceptance_length") or acc_log.get("acceptance_length_log_mean"),
        "spec_draft_acceptance_rate": acc.get("draft_acceptance_rate"),
        "server_start_s": start_s, "started_utc": started, "notes": notes, **gpu,
    }
    if result_path.exists():
        raw = json.loads(result_path.read_text())
        rec = results.normalize(raw, context=ctx)
        problems = results.validate(rec, osl=p.osl, range_ratio=p.random_range_ratio)
    else:
        rec = results.normalize({}, context=ctx)
        rec["status"] = "no-result"
        problems = [f"client rc={rc}, no result file"]
    if rc != 0:
        problems.append(f"client exit code {rc}")
    rec["valid"] = not problems
    rec["problems"] = problems
    results.append_csv(run_dir / "results.csv", rec)
    results.append_jsonl(run_dir / "results.jsonl", rec)
    return rec


def _record_skip(run_dir: Path, p: Point, why: str) -> None:
    rec = {c: None for c in results.COLUMNS}
    rec.update(p.to_dict())
    rec.update({"schema_version": results.SCHEMA_VERSION, "status": "skipped", "notes": why,
                "valid": False, "problems": [why]})
    results.append_csv(run_dir / "results.csv", rec)
    results.append_jsonl(run_dir / "results.jsonl", rec)
