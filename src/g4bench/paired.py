"""Same-GPU paired benchmarking (D-061): engine A and engine B interleaved in ONE container on ONE GPU.

Why: Modal assigns a different physical GPU per job, and machine-to-machine spread at 8K is 5-15 % (some GPUs
throttle at 83 C; D-060) - larger than the effects we compare. Running both engines back to back on the same GPU
makes the hardware term (mostly) cancel in the ratio B/A.

Sequence, for each selected workload point and repeat ``r`` (= the point's ``trial``):

    pair r:  fresh server for the first engine of ``segment_order(order, r)`` -> run the point -> stop the server,
             then the same for the other engine.

``order``: "ABBA" alternates AB, BA, AB, ... over repeats (counterbalances drift and warm-up order); "ABAB" always
AB; "BAAB" / "BABA" are the mirror images (to continue an ABBA series in a second job). Every segment starts its own
server (the matrix's ``fresh_server_per_point`` is ignored), so both engines see the same start-up/cool-down pattern.
The correctness gate runs once per engine, after its first measured segment (never inside a measurement).

Outputs: one ordinary run dir per engine under ``results/runs`` (so analyze / report keep working; the manifest
carries ``paired``), plus ``results/pairs/<pair_id>/paired.json`` with, per pair: both segments' metrics, B/A ratios,
GPU clock / power / temperature per segment (from that segment's nvidia-smi samples), the GPU UUID read at the start
of the pair and the container id.

Wall budget: a pair is started only if it is expected to finish (first pair: ``plan_sequence`` estimate; later
pairs: the slowest pair so far); a pair is never left half-measured on purpose.
"""

from __future__ import annotations

import csv
import json
import math
import os
import secrets
import socket
import time
from pathlib import Path
from typing import Any, Callable

ORDERS: dict[str, tuple[str, ...]] = {"ABBA": ("AB", "BA"), "ABAB": ("AB",), "BAAB": ("BA", "AB"), "BABA": ("BA",)}
# B/A for every metric; for latencies < 1 means B is faster, for throughputs > 1 means B is faster.
RATIO_METRICS = ("output_throughput", "total_token_throughput", "request_throughput",
                 "mean_ttft_ms", "median_ttft_ms", "p90_ttft_ms", "p99_ttft_ms",
                 "mean_tpot_ms", "median_tpot_ms", "p90_tpot_ms", "p99_tpot_ms",
                 "mean_e2el_ms", "median_e2el_ms", "p90_e2el_ms", "p99_e2el_ms", "spec_acceptance_length")

# Sizing model for one segment (calibrated on the D-056..D-060 runs, pinned vLLM image, warm compile caches):
# server start 98-226 s (mean ~125 s), plus SIGTERM/teardown; requests at the stock recipe's speed, which the article's
# vLLM number matches within a few % (our measured mtp4 vs article). B is the faster engine, so A bounds the pair.
SEG_SERVER_START_S = 150.0
SEG_STOP_S = 30.0
GATE_S = 60.0             # probes + 8 greedy prompts x 256 tokens, once per engine
JOB_OVERHEAD_S = 60.0     # tokenizer, metadata, reports, returning files
SAFETY = 1.10


def segment_order(order: str, repeat: int) -> str:
    if order not in ORDERS:
        raise ValueError(f"order must be one of {sorted(ORDERS)}, got {order!r}")
    cyc = ORDERS[order]
    return cyc[repeat % len(cyc)]


def _tok_s_guess(p, targets: dict[str, Any]) -> float:
    near = 8192 if p.isl > 4096 else 1024
    ref = targets.get(f"{near}x{p.conc}", {}).get("output_throughput", {}).get("vllm")
    return float(ref) if ref else 50.0


def segment_seconds(p, targets: dict[str, Any] | None = None) -> float:
    """Expected wall seconds of one segment (server start + warmups + measured requests + stop)."""
    from .compare import load_targets

    targets = load_targets() if targets is None else targets
    mean_out = p.osl * (1.0 + float(p.random_range_ratio)) / 2.0 if p.ignore_eos else p.osl
    toks = (p.num_prompts + p.num_warmups) * mean_out
    return SEG_SERVER_START_S + SEG_STOP_S + toks / _tok_s_guess(p, targets)


def plan_sequence(points: list, *, max_wall_s: float, correctness: bool = True,
                  targets: dict[str, Any] | None = None, rate_usd_s: float | None = None) -> dict[str, Any]:
    """Will the whole paired sequence fit the job's wall budget? (``points`` already expanded with trials=repeats.)"""
    from . import spend as S
    from .compare import load_targets

    targets = load_targets() if targets is None else targets
    pairs = [2 * segment_seconds(p, targets) for p in points]
    fixed = JOB_OVERHEAD_S + (2 * GATE_S if correctness else 0.0)
    total = (fixed + sum(pairs)) * SAFETY
    fit, acc = 0, fixed
    for s in pairs:
        if (acc + s) * SAFETY > max_wall_s:
            break
        acc += s
        fit += 1
    rate = S.GPU_JOB.rate() if rate_usd_s is None else rate_usd_s
    # Reservation that would hold the whole sequence: timeout = need + soft-wall headroom (modal_run).
    from .modal_run import SOFT_WALL_HEADROOM_S

    usd_needed = S.worst_case_usd(rate, total + SOFT_WALL_HEADROOM_S)
    out = {"n_pairs": len(pairs), "n_segments": 2 * len(pairs), "est_pair_s": [round(s) for s in pairs],
           "est_total_s": round(total), "max_wall_s": round(max_wall_s), "fits": total <= max_wall_s,
           "pairs_that_fit": fit, "usd_needed_for_all": round(usd_needed, 2),
           "model": {"server_start_s": SEG_SERVER_START_S, "stop_s": SEG_STOP_S, "gate_s": GATE_S if correctness else 0,
                     "job_overhead_s": JOB_OVERHEAD_S, "safety": SAFETY,
                     "tok_s": "article vLLM output tok/s for the workload (stock recipe = the slower engine)"}}
    if not out["fits"]:
        if fit:
            out["suggestion"] = (f"only {fit} of {len(pairs)} pairs fit in {max_wall_s:.0f} s; split into jobs of "
                                 f"<= {fit} pair(s) (continue an ABBA series with --order BAAB in the next job)")
        else:
            one = (fixed + pairs[0]) * SAFETY
            bare = (JOB_OVERHEAD_S + pairs[0]) * SAFETY
            alt = (" or --no-correctness (fits)" if correctness and bare <= max_wall_s else "")
            out["suggestion"] = (f"even one pair needs ~{round(one)} s > {max_wall_s:.0f} s: it needs a "
                                 f"${S.worst_case_usd(rate, one + SOFT_WALL_HEADROOM_S):.2f} reservation (above the "
                                 f"per-job cap = owner decision){alt}, or a smaller workload")
    return out


# ------------------------------------------------------------------------------------------------ GPU evidence
def gpu_segment(path: Path) -> dict[str, Any]:
    """Clock / power / temperature over one segment's nvidia-smi samples (server.GpuSampler CSV)."""
    cols = {"clocks.sm": "sm_clock_mhz", "clocks.mem": "mem_clock_mhz", "power.draw": "power_w",
            "temperature.gpu": "temp_c", "utilization.gpu": "util_pct"}
    vals: dict[str, list[float]] = {v: [] for v in cols.values()}
    n = 0
    if Path(path).exists():
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f, skipinitialspace=True):
                n += 1
                for k, v in cols.items():
                    try:
                        vals[v].append(float(row[k]))
                    except (KeyError, TypeError, ValueError):
                        pass
    out: dict[str, Any] = {"n_samples": n}
    for k, xs in vals.items():
        out[f"{k}_mean"] = round(sum(xs) / len(xs), 2) if xs else None
        out[f"{k}_min"] = min(xs) if xs else None
        out[f"{k}_max"] = max(xs) if xs else None
    return out


def gpu_uuids() -> list[str]:
    from . import metadata as md

    gpus, _ = md.gpu_info()
    return [g.get("uuid") for g in gpus if g.get("uuid")]


def container_id() -> dict[str, Any]:
    return {"modal_task_id": os.environ.get("MODAL_TASK_ID"), "hostname": socket.gethostname()}


def ratios(a: dict[str, Any], b: dict[str, Any]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for m in RATIO_METRICS:
        x, y = a.get(m), b.get(m)
        try:
            out[m] = float(y) / float(x) if x and y is not None and math.isfinite(float(x)) else None
        except (TypeError, ValueError, ZeroDivisionError):
            out[m] = None
    return out


# ------------------------------------------------------------------------------------------------ the run
def _segment(prep, p, slot: str, *, t_start: float, max_wall_s: float, gate: bool) -> dict[str, Any]:
    from .runner import (ServerError, _bounded, _record_skip, correctness_gate, run_point, start_server)

    pdir = prep.run_dir / "points" / p.point_id
    pdir.mkdir(parents=True, exist_ok=True)
    seg: dict[str, Any] = {"slot": slot, "engine": prep.eng.name, "run_id": prep.run_id, "point_id": p.point_id,
                           "started_unix": time.time()}
    t0 = time.monotonic()
    try:
        server, start_s, problems = start_server(prep, pdir, _bounded(prep.opts.server_timeout_s, max_wall_s, t_start))
    except ServerError as e:
        _record_skip(prep.run_dir, p, f"server failed: {e}")
        seg.update(valid=False, problems=[f"server failed: {e}"], wall_s=round(time.monotonic() - t0, 2),
                   ended_unix=time.time(), metrics={}, gpu=gpu_segment(pdir / "gpu_samples.csv"))
        return seg
    try:
        rec = run_point(prep, pdir, p, server=server, start_s=start_s,
                        timeout_s=_bounded(prep.opts.client_timeout_s, max_wall_s, t_start), problems=problems)
        if gate:
            correctness_gate(prep)  # after the measurement, on the same server
    finally:
        server.stop()
    try:
        cub = json.loads((pdir / "cublas.json").read_text())
    except (OSError, ValueError):
        cub = None
    seg.update(valid=bool(rec.get("valid")), problems=rec.get("problems") or [], server_start_s=start_s,
               wall_s=round(time.monotonic() - t0, 2), ended_unix=time.time(),
               metrics={m: rec.get(m) for m in RATIO_METRICS + ("duration_s", "completed", "failed")},
               gpu=gpu_segment(pdir / "gpu_samples.csv"),
               cublas=(cub or {}).get("check"), cublas_loaded=((cub or {}).get("probe") or {}).get("realpaths"))
    return seg


def run_paired(engines: list[str], *, matrix: str, results_root: Path, pairs_root: Path, order: str = "ABBA",
               repeats: int = 1, seeds: list[int] | None = None, only: list[str] | None = None,
               max_wall_s: float = 3600.0, notes: str = "", **run_kw: Any) -> dict[str, Any]:
    """Run the interleaved sequence; returns {pair_id, run_ids, n_pairs, n_valid_pairs, paired_json}."""
    from . import runner
    from .compare import load_targets

    if len(engines) != 2:
        raise ValueError(f"paired needs exactly two engines, got {engines}")
    if order not in ORDERS:
        raise ValueError(f"order must be one of {sorted(ORDERS)}, got {order!r}")
    repeats = int(repeats)
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    safe = lambda s: "".join(ch if ch.isalnum() or ch in ".-" else "-" for ch in Path(s).stem)  # noqa: E731
    pair_id = f"{stamp}-pair-{safe(engines[0])}-vs-{safe(engines[1])}-{secrets.token_hex(2)}"
    preps: dict[str, Any] = {}
    for slot, name in zip("AB", engines):
        opts = runner.RunOptions(engine=name, matrix=matrix, seeds=seeds, trials=repeats, only_keys=only,
                                 results_root=Path(results_root), max_wall_s=max_wall_s,
                                 notes=(notes + " " if notes else "") + f"[paired {pair_id} slot {slot}]", **run_kw)
        preps[slot] = runner.prepare(opts, fresh_override=True, manifest_extra={"paired": {
            "pair_id": pair_id, "slot": slot, "engines": {"A": engines[0], "B": engines[1]}, "order": order,
            "repeats": repeats}})
    pa, pb = preps["A"], preps["B"]
    if [p.point_id for p in pa.points] != [p.point_id for p in pb.points]:
        raise SystemExit("paired engines expanded to different points (same matrix/only/seeds expected)")
    tok = run_kw.get("tokenizer") or runner.resolve_tokenizer(pa.pins)
    for prep in preps.values():
        prep.tokenizer = tok
    targets = load_targets()
    t_start = time.monotonic()
    gated: set[str] = set()
    pair_s: list[float] = []
    pairs: list[dict[str, Any]] = []
    uuids_start = gpu_uuids()
    container = container_id()
    try:
        for p in pa.points:
            seq = segment_order(order, p.trial)
            elapsed = time.monotonic() - t_start
            need = max(pair_s) if pair_s else 2 * segment_seconds(p, targets)
            if elapsed + need > max_wall_s:
                why = f"wall budget: {elapsed:.0f}s used + ~{need:.0f}s for this pair > {max_wall_s:.0f}s"
                print(f"[g4bench] paired: skipping {p.point_id} ({why})", flush=True)
                for prep in preps.values():
                    runner._record_skip(prep.run_dir, p, why)
                pairs.append({"key": p.key, "point_id": p.point_id, "trial": p.trial, "seed": p.seed,
                              "sequence": seq, "valid": False, "skipped": why})
                continue
            t_pair = time.monotonic()
            uuid_now = gpu_uuids()
            segs = {}
            for slot in seq:
                prep = preps[slot]
                gate = bool(prep.opts.correctness) and slot not in gated
                segs[slot] = _segment(prep, p, slot, t_start=t_start, max_wall_s=max_wall_s, gate=gate)
                gated.add(slot)
            pair_s.append(time.monotonic() - t_pair)
            valid = segs["A"]["valid"] and segs["B"]["valid"]
            pairs.append({"key": p.key, "point_id": p.point_id, "trial": p.trial, "seed": p.seed, "isl": p.isl,
                          "osl": p.osl, "conc": p.conc, "dataset": p.dataset, "sequence": seq, "valid": valid,
                          "gpu_uuid": uuid_now, "same_gpu_as_job_start": uuid_now == uuids_start,
                          "container": container, "wall_s": round(pair_s[-1], 2),
                          "segments": [segs[s] for s in seq],
                          "ratios_b_over_a": ratios(segs["A"]["metrics"], segs["B"]["metrics"]) if valid else None})
            print(f"[g4bench] paired {p.point_id} {seq}: B/A tok/s = "
                  f"{(pairs[-1]['ratios_b_over_a'] or {}).get('output_throughput')}", flush=True)
    finally:
        for prep in preps.values():
            runner.finish(prep)
        uuids_end = gpu_uuids()
        doc = {
            "schema": 1, "pair_id": pair_id, "engines": {"A": engines[0], "B": engines[1]},
            "engine_configs": {"A": pa.eng.name, "B": pb.eng.name},
            "engine_kinds": {"A": pa.eng.engine, "B": pb.eng.engine},
            "run_ids": {"A": pa.run_id, "B": pb.run_id}, "config_hashes": {"A": pa.chash, "B": pb.chash},
            "matrix": matrix, "order": order, "repeats": repeats,
            "container": container, "hardware_class": pa.meta.get("hardware_class"),
            "exact_hardware": pa.meta.get("exact_hardware"),
            "gpu": {"uuid_start": uuids_start, "uuid_end": uuids_end, "same_gpu": uuids_start == uuids_end,
                    "name": (pa.meta.get("gpus") or [{}])[0].get("name")},
            "runtime_image": pa.meta.get("env", {}).get("G4_IMAGE_PATCHES"),
            "ratio_definition": "B/A per metric (throughput > 1 = B faster; latency < 1 = B faster)",
            "pairs": pairs, "notes": notes,
        }
        out_dir = Path(pairs_root) / pair_id
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "paired.json").write_text(json.dumps(doc, indent=1, default=str))
    return {"pair_id": pair_id, "run_id": pair_id, "run_ids": {"A": pa.run_id, "B": pb.run_id},
            "n_pairs": len(pairs), "n_valid_pairs": sum(1 for x in pairs if x.get("valid")),
            "paired_json": str(out_dir / "paired.json"),
            "ratios_output_throughput": [(x.get("ratios_b_over_a") or {}).get("output_throughput") for x in pairs]}


def run_paired_job(spec: dict[str, Any], results_root: Path, clock: Callable[[], float] = time.time,
                   **runner_overrides: Any) -> dict[str, Any]:
    """jobs.run_job for a spec with ``paired`` (Modal worker body; CPU-tested with the mock engine)."""
    from .jobs import DEADLINE_HEADROOM_S

    if spec.get("code_sha"):
        os.environ["G4_CODE_SHA"] = str(spec["code_sha"])
    for k in ("greedy", "greedy_reference", "correctness"):
        if k in spec and spec[k] is not None:
            runner_overrides.setdefault(k, spec[k])
    pr = spec["paired"]
    remaining = float(spec["deadline_unix"]) - clock() - DEADLINE_HEADROOM_S
    max_wall = max(min(float(spec["max_wall_s"]), remaining), 0.0)
    results_root = Path(results_root)
    return run_paired(list(pr["engines"]), matrix=spec["matrix"], results_root=results_root,
                      pairs_root=results_root.parent / "pairs", order=pr.get("order", "ABBA"),
                      repeats=int(pr.get("repeats", 1)), seeds=spec.get("seeds"), only=spec.get("only"),
                      max_wall_s=max_wall, notes=spec.get("notes", ""), **runner_overrides)
