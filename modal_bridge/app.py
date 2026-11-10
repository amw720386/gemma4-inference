"""One-time-deployable Modal app: GPU benchmark workers + an authenticated HTTPS bridge.

Why a bridge: this project's agent runs in an environment whose egress allows HTTPS
but not Modal's gRPC control plane. A human deploys this file ONCE with the Modal
CLI from their own machine (see modal_bridge/BOOTSTRAP.md); afterwards every job is
submitted, polled and cancelled over documented HTTPS web endpoints protected by
Modal proxy auth (``requires_proxy_auth=True`` => Modal-Key/Modal-Secret headers or
``Authorization: Bearer <id>.<secret>``).

Guard rails (enforced server-side, not just in the client):
* engine/matrix/GPU allowlists; no arbitrary commands, images, or server args
* pre-launch cost estimate (g4bench.cost); jobs above ``max_usd`` or the hard cap
  ``G4_BRIDGE_HARD_CAP_USD`` (default $40/job) are refused
* soft wall budget passed to the runner + hard Modal ``timeout`` per GPU function
* at most ``G4_BRIDGE_MAX_ACTIVE`` concurrent GPU jobs (default 1)
* weights are prefetched to a Volume by a CPU-only function, so GPU time is not
  spent downloading 24 GB

Nothing here touches billing settings or other apps.
"""

from __future__ import annotations

import json
import os
import pathlib
import time

import modal

APP_NAME = "g4repro-bridge"
REPO_LOCAL = pathlib.Path(__file__).resolve().parents[1]
REPO_REMOTE = "/repo"
HF_HOME = "/cache/hf"
RESULTS_MOUNT = "/results"

# Pinned by digest (configs/pins.yaml). linux/amd64 manifests.
VLLM_IMAGE = "vllm/vllm-openai@sha256:91ee972ab751fb7180bc71c175865e36b091686600dca783d52b2511399810b3"
SGLANG_IMAGE = "lmsysorg/sglang@sha256:f7876bef696ccf9cf75a2bf7ccb82970f5fac577f082fb8ca95c9a59d2160eb5"

IGNORE = [".git", ".venv", "**/__pycache__", "results/runs", "results/scratch", "*.pyc"]
ENV = {"PYTHONPATH": f"{REPO_REMOTE}/src", "HF_HOME": HF_HOME, "HF_HUB_ENABLE_HF_TRANSFER": "0",
       "TOKENIZERS_PARALLELISM": "false", "G4_DATA_DIR": f"{HF_HOME}/g4data/speedbench"}

ALLOWED_ENGINES = {"vllm-0.33.0-mtp4": "vllm", "vllm-0.33.0-mtp4-textonly": "vllm", "vllm-0.33.0-nospec": "vllm",
                   "sglang-0.5.23-nextn": "sglang"}
ALLOWED_MATRICES = {"smoke", "backboard-repro", "speedbench-coding", "speedbench-tp"}
ALLOWED_GPUS = {"RTX-PRO-6000", "L40S", "A100-80GB", "H100!", "H200", "B200"}
HARD_TIMEOUT_S = 6 * 3600


def _engine_image(ref: str) -> modal.Image:
    # Engine images already ship python3, pyyaml, numpy, requests, aiohttp, transformers, jinja2,
    # huggingface_hub and tqdm (all are engine dependencies). We add NOTHING via pip so the engine's
    # dependency set stays exactly the published image.
    return (modal.Image.from_registry(ref).entrypoint([]).env(ENV)
            .add_local_dir(str(REPO_LOCAL), REPO_REMOTE, ignore=IGNORE))


vllm_image = _engine_image(VLLM_IMAGE)
sglang_image = _engine_image(SGLANG_IMAGE)
bridge_image = (modal.Image.debian_slim(python_version="3.13")
                # Exactly the versions the CPU test-suite passed with (see STATUS.md).
                .pip_install("fastapi[standard]==0.162.4", "pyyaml==6.0.3", "numpy==2.5.4", "requests==2.34.2",
                             "huggingface_hub==2.4.0", "aiohttp==3.14.5", "transformers==5.22.2", "tqdm==4.70.1",
                             "jinja2==3.1.6", "pytest==9.2.0", "pyarrow==26.0.1")
                .env(ENV).add_local_dir(str(REPO_LOCAL), REPO_REMOTE, ignore=IGNORE))

hf_vol = modal.Volume.from_name("g4repro-hf-cache", create_if_missing=True)
res_vol = modal.Volume.from_name("g4repro-results", create_if_missing=True)
jobs = modal.Dict.from_name("g4repro-jobs", create_if_missing=True)

app = modal.App(APP_NAME)


def _commit_loop(stop: "threading.Event") -> None:  # noqa: F821
    while not stop.wait(60):
        try:
            res_vol.commit()
        except Exception:  # noqa: BLE001
            pass


def _run_job(spec: dict) -> dict:
    import threading

    from g4bench.runner import RunOptions, run

    hf_vol.reload()
    stop = threading.Event()
    t = threading.Thread(target=_commit_loop, args=(stop,), daemon=True)
    t.start()
    try:
        run_dir = run(RunOptions(
            engine=spec["engine"], matrix=spec["matrix"], seeds=spec.get("seeds"), trials=spec.get("trials"),
            only_keys=spec.get("only"), results_root=pathlib.Path(RESULTS_MOUNT) / "runs",
            max_wall_s=float(spec["max_wall_s"]), notes=spec.get("notes", ""), dry_run=bool(spec.get("dry_run")),
        ))
    finally:
        stop.set()
        res_vol.commit()
    rows = []
    f = run_dir / "results.jsonl"
    if f.exists():
        rows = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
    return {"run_id": run_dir.name, "n_points": len(rows), "n_valid": sum(1 for r in rows if r.get("valid")),
            "rows": rows}


_gpu_kw = dict(gpu="RTX-PRO-6000", timeout=HARD_TIMEOUT_S, volumes={HF_HOME: hf_vol, RESULTS_MOUNT: res_vol},
               max_containers=1)


@app.function(image=vllm_image, **_gpu_kw)
def run_vllm(spec: dict) -> dict:
    return _run_job(spec)


@app.function(image=sglang_image, **_gpu_kw)
def run_sglang(spec: dict) -> dict:
    return _run_job(spec)


@app.function(image=bridge_image, timeout=3600, volumes={HF_HOME: hf_vol}, cpu=2.0, memory=4096)
def prefetch_weights() -> dict:
    """CPU-only: download pinned target + draft snapshots, and prepare + verify the SPEED-Bench
    prompt sets, into the HF cache Volume (so GPU containers never download anything big)."""
    from huggingface_hub import snapshot_download

    from g4bench.config import load_pins

    pins = load_pins()
    out = {}
    for role in ("target", "draft"):
        m = pins["model"][role]
        t0 = time.time()
        path = snapshot_download(m["repo"], revision=m["revision"])
        out[role] = {"repo": m["repo"], "revision": m["revision"], "path": path, "seconds": round(time.time() - t0, 1)}
    from g4bench import speedbench

    data, cache = pathlib.Path(os.environ["G4_DATA_DIR"]), pathlib.Path(HF_HOME) / "g4data" / "cache"
    out["speedbench"] = {
        "tp1k": speedbench.prepare_throughput_inline("throughput_1k", data, cache)["prompt_digest"],
        "tp8k": speedbench.prepare_throughput_inline("throughput_8k", data, cache)["prompt_digest"],
        "coding": speedbench.prepare_coding(data, cache)["prompt_digest"],  # raises unless InferenceX digest
    }
    hf_vol.commit()
    return out


@app.function(image=bridge_image, timeout=1800, cpu=2.0, memory=4096)
def selftest() -> dict:
    """CPU-only: run the repo's test-suite inside Modal (validates image + code, costs cents)."""
    import subprocess

    p = subprocess.run(["python", "-m", "pytest", "-q", f"{REPO_REMOTE}/tests"], capture_output=True, text=True,
                       cwd=REPO_REMOTE)
    return {"rc": p.returncode, "tail": (p.stdout + p.stderr)[-4000:]}


@app.function(image=bridge_image, volumes={RESULTS_MOUNT: res_vol}, timeout=600)
@modal.concurrent(max_inputs=16)
@modal.asgi_app(label="g4repro-bridge", requires_proxy_auth=True)
def bridge():
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import PlainTextResponse
    from pydantic import BaseModel, Field

    from g4bench.cost import estimate
    from g4bench.matrix import expand, load_matrix

    api = FastAPI(title="g4repro bridge")
    hard_cap = float(os.environ.get("G4_BRIDGE_HARD_CAP_USD", "40"))
    max_active = int(os.environ.get("G4_BRIDGE_MAX_ACTIVE", "1"))

    class JobSpec(BaseModel):
        engine: str
        matrix: str = "smoke"
        seeds: list[int] | None = None
        trials: int | None = Field(default=None, ge=1, le=5)
        only: list[str] | None = None
        gpu: str = "RTX-PRO-6000"
        max_usd: float = Field(gt=0)
        notes: str = ""
        dry_run: bool = False

    def _active() -> list[str]:
        act = []
        for cid, meta in list(jobs.items()):
            if meta.get("state") in ("submitted", "running"):
                try:
                    modal.FunctionCall.from_id(cid).get(timeout=0)
                    meta["state"] = "finished"
                    jobs[cid] = meta
                except TimeoutError:
                    act.append(cid)
                except Exception:  # noqa: BLE001 - failed/cancelled/expired
                    meta["state"] = "ended"
                    jobs[cid] = meta
        return act

    @api.get("/health")
    def health():
        return {"ok": True, "app": APP_NAME, "hard_cap_usd": hard_cap, "max_active": max_active,
                "engines": sorted(ALLOWED_ENGINES), "matrices": sorted(ALLOWED_MATRICES), "gpus": sorted(ALLOWED_GPUS)}

    @api.post("/estimate")
    def est(spec: JobSpec):
        return _estimate(spec)

    def _estimate(spec: JobSpec) -> dict:
        if spec.engine not in ALLOWED_ENGINES:
            raise HTTPException(400, f"engine not allowed: {spec.engine}")
        if spec.matrix not in ALLOWED_MATRICES:
            raise HTTPException(400, f"matrix not allowed: {spec.matrix}")
        if spec.gpu not in ALLOWED_GPUS:
            raise HTTPException(400, f"gpu not allowed: {spec.gpu}")
        mat = load_matrix(spec.matrix)
        pts = expand(mat, seeds=spec.seeds, trials=spec.trials, only_keys=spec.only)
        if not pts:
            raise HTTPException(400, "no points selected")
        gpu_key = spec.gpu.rstrip("!")
        kind = ALLOWED_ENGINES[spec.engine]
        e = estimate(pts, gpu_key,
                     fresh_server_per_point=bool(mat.get("fresh_server_per_point", True)),
                     tok_s_ref="sglang" if kind == "sglang" else "vllm")
        return {"points": len(pts), "est_usd": round(e.usd, 2), "est_gpu_s": round(e.gpu_seconds), "summary": e.summary()}

    @api.post("/jobs")
    def submit(spec: JobSpec):
        e = _estimate(spec)
        if e["est_usd"] > min(spec.max_usd, hard_cap):
            raise HTTPException(402, f"estimate ${e['est_usd']} exceeds max_usd={spec.max_usd} / hard cap {hard_cap}")
        if len(_active()) >= max_active:
            raise HTTPException(409, "another GPU job is active; poll or cancel it first")
        # Soft wall budget: 1.5x the (already pessimistic) estimate, below the hard Modal timeout.
        payload = spec.model_dump()
        payload["max_wall_s"] = min(1.5 * e["est_gpu_s"], HARD_TIMEOUT_S - 600)
        fn = run_vllm if ALLOWED_ENGINES[spec.engine] == "vllm" else run_sglang
        if spec.gpu != "RTX-PRO-6000":
            fn = fn.with_options(gpu=spec.gpu)
        call = fn.spawn(payload)
        jobs[call.object_id] = {"state": "submitted", "spec": payload, "estimate": e, "submitted": time.time()}
        return {"call_id": call.object_id, "estimate": e, "max_wall_s": payload["max_wall_s"]}

    @api.get("/jobs")
    def list_jobs():
        _active()
        return {cid: {k: v for k, v in m.items() if k != "result"} for cid, m in jobs.items()}

    @api.get("/jobs/{call_id}")
    def status(call_id: str):
        meta = jobs.get(call_id)
        if meta is None:
            raise HTTPException(404, "unknown call id")
        try:
            res = modal.FunctionCall.from_id(call_id).get(timeout=0)
        except TimeoutError:
            return {"state": "running", "elapsed_s": round(time.time() - meta["submitted"]), "estimate": meta["estimate"]}
        except Exception as ex:  # noqa: BLE001
            meta["state"] = "ended"
            jobs[call_id] = meta
            return {"state": "ended", "error": repr(ex)[:2000]}
        meta["state"] = "finished"
        meta["finished"] = time.time()
        jobs[call_id] = meta
        return {"state": "finished", "result": res, "wall_s": round(meta["finished"] - meta["submitted"])}

    @api.post("/jobs/{call_id}/cancel")
    def cancel(call_id: str):
        if call_id not in jobs:
            raise HTTPException(404, "unknown call id")
        modal.FunctionCall.from_id(call_id).cancel()
        meta = jobs[call_id]
        meta["state"] = "cancelled"
        jobs[call_id] = meta
        return {"cancelled": call_id}

    @api.post("/prefetch")
    def prefetch():
        call = prefetch_weights.spawn()
        jobs[call.object_id] = {"state": "submitted", "spec": {"kind": "prefetch"}, "estimate": {"est_usd": 0.05},
                                "submitted": time.time()}
        return {"call_id": call.object_id}

    @api.post("/selftest")
    def run_selftest():
        call = selftest.spawn()
        jobs[call.object_id] = {"state": "submitted", "spec": {"kind": "selftest"}, "estimate": {"est_usd": 0.02},
                                "submitted": time.time()}
        return {"call_id": call.object_id}

    @api.get("/runs")
    def runs():
        res_vol.reload()
        root = pathlib.Path(RESULTS_MOUNT) / "runs"
        return sorted(p.name for p in root.iterdir()) if root.exists() else []

    @api.get("/runs/{run_id}/{path:path}", response_class=PlainTextResponse)
    def run_file(run_id: str, path: str):
        res_vol.reload()
        root = (pathlib.Path(RESULTS_MOUNT) / "runs" / run_id).resolve()
        f = (root / path).resolve()
        if root not in f.parents or not f.is_file():
            raise HTTPException(404, "not found")
        if f.stat().st_size > 20_000_000:
            raise HTTPException(413, "file too large")
        return f.read_text(errors="replace")

    return api
