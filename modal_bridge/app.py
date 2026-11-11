"""One-time-deployable Modal app: GPU benchmark workers + an authenticated, budget-gated HTTPS bridge.

Why a bridge: this project's agent runs where egress allows HTTPS but not Modal's gRPC control plane.
A human deploys this file ONCE with the Modal CLI (modal_bridge/BOOTSTRAP.md); afterwards jobs are
submitted, polled and cancelled over documented HTTPS web endpoints protected by Modal proxy auth
(``requires_proxy_auth=True`` => ``Modal-Key``/``Modal-Secret`` or ``Authorization: Bearer <id>.<secret>``).

Spend control (server-side; see src/g4bench/budget.py for the invariants and proofs-by-test):
* **Project-wide cap** ``G4_PROJECT_CAP_USD`` (default **$10**), persisted in the Modal Dict
  ``g4repro-budget``. Every paid launch - GPU jobs, CPU ``/prefetch``, CPU ``/selftest`` - first reserves its
  worst-case cost; the reservation is refused if committed spend would exceed the cap.
* **Per-job cap** ``G4_PER_JOB_CAP_USD`` (default **$2**): a job's Modal ``timeout`` is derived from its
  reservation (``budget.timeout_for_budget``) so Modal itself kills the input before the worst case is exceeded.
* **Race-free / restart-safe**: all ledger mutations run in ``ledger_op`` - one container
  (``max_containers=1``), one input at a time (no ``@modal.concurrent``) - and additionally take a lease lock
  in the Dict via atomic put-if-absent. State is re-read from the Dict on every operation.
* **Concurrency**: ``G4_BRIDGE_MAX_ACTIVE`` (default 1) open reservations across all job kinds.
* **Conservative settlement**: reservations are held until Modal confirms the call terminal; cost is then
  settled from wall time since reservation (before container start) + margins. Unknown poll errors hold the
  reservation. Every GPU job aborts immediately if (re)started after its deadline (preemption restarts).
* This ledger is an **estimate that gates launches through this bridge only. It is NOT a Modal billing hard
  stop** and cannot see spend from anything launched outside the bridge. Modal's invoice is authoritative.

Also: engine/matrix/GPU allowlists; exact digest-pinned engine images; explicit cpu/memory *limits* on every
function (the ledger prices containers at their limits); ``retries=0``. Nothing here touches billing settings.
"""

from __future__ import annotations

import json
import os
import pathlib
import time

import modal

from g4bench import budget as B  # noqa: E402  (deployer runs `modal deploy` from the repo root with src on PYTHONPATH)
from g4bench.image_setup import python_setup_commands  # noqa: E402
from g4bench.bridge_api import (ALLOWED_ENGINES, ALLOWED_GPUS, ALLOWED_MATRICES, GPU_JOB, PREFETCH,  # noqa: F401
                                SELFTEST, BridgeConfig, build_api)

APP_NAME = "g4repro-bridge"
REPO_LOCAL = pathlib.Path(__file__).resolve().parents[1]
REPO_REMOTE = "/repo"
HF_HOME = "/cache/hf"
RESULTS_MOUNT = "/results"

# Pinned by digest (configs/pins.yaml). linux/amd64 manifests.
VLLM_IMAGE = "vllm/vllm-openai@sha256:91ee972ab751fb7180bc71c175865e36b091686600dca783d52b2511399810b3"
SGLANG_IMAGE = "lmsysorg/sglang@sha256:f7876bef696ccf9cf75a2bf7ccb82970f5fac577f082fb8ca95c9a59d2160eb5"

# Budget configuration is baked into the deployed image env: changing it requires a (human) redeploy.
# G4_PROJECT_BUDGET_USD / G4_JOB_CAP_USD are accepted as aliases (names used in the agent's cloud environment).
PROJECT_CAP_USD = float(os.environ.get("G4_PROJECT_CAP_USD") or os.environ.get("G4_PROJECT_BUDGET_USD") or "10")
PER_JOB_CAP_USD = min(float(os.environ.get("G4_PER_JOB_CAP_USD") or os.environ.get("G4_JOB_CAP_USD") or "2"),
                      PROJECT_CAP_USD)
MAX_ACTIVE = int(os.environ.get("G4_BRIDGE_MAX_ACTIVE", "1"))
PLATFORM_RESERVE_USD = float(os.environ.get("G4_PLATFORM_RESERVE_USD", "0.50"))
LEDGER_CFG = B.LedgerConfig(cap_usd=PROJECT_CAP_USD, platform_reserve_usd=PLATFORM_RESERVE_USD,
                            per_job_cap_usd=PER_JOB_CAP_USD, max_active=MAX_ACTIVE)

def _code_sha() -> str:
    if os.environ.get("G4_CODE_SHA"):
        return os.environ["G4_CODE_SHA"]
    import subprocess

    try:
        sha = subprocess.run(["git", "-C", str(REPO_LOCAL), "rev-parse", "HEAD"], capture_output=True, text=True,
                             timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(REPO_LOCAL), "status", "--porcelain"], capture_output=True, text=True,
                               timeout=10).stdout.strip()
        return (sha + ("-dirty" if dirty else "")) if sha else "unknown"
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"


CODE_SHA = _code_sha()  # baked into the image env at deploy time; reported by /health

IGNORE = [".git", ".venv", "**/__pycache__", "results/runs", "results/scratch", "*.pyc", ".cache", "data/speedbench/*.jsonl"]
ENV = {"PYTHONPATH": f"{REPO_REMOTE}/src", "HF_HOME": HF_HOME, "HF_HUB_ENABLE_HF_TRANSFER": "0",
       "TOKENIZERS_PARALLELISM": "false", "G4_DATA_DIR": f"{HF_HOME}/g4data/speedbench",
       "G4_PROJECT_CAP_USD": str(PROJECT_CAP_USD), "G4_PER_JOB_CAP_USD": str(PER_JOB_CAP_USD),
       "G4_BRIDGE_MAX_ACTIVE": str(MAX_ACTIVE), "G4_PLATFORM_RESERVE_USD": str(PLATFORM_RESERVE_USD),
       "G4_CODE_SHA": CODE_SHA}

# GPU job default timeout = the longest a per-job-cap reservation can buy (spawns pass a tighter one).
GPU_DEFAULT_TIMEOUT_S = B.timeout_for_budget(GPU_JOB.rate(), PER_JOB_CAP_USD)
SCALEDOWN_WINDOW_S = 10  # <= budget.SCALEDOWN_MARGIN_S


def _engine_image(ref: str, engine_module: str) -> modal.Image:
    # Engine images already ship python3, pyyaml, numpy, requests, aiohttp, transformers, jinja2,
    # huggingface_hub and tqdm (all engine dependencies). We add NOTHING via pip, so the engine's
    # dependency set is exactly the published image. The setup line only aliases `python` to the image's own
    # python3 when missing (vLLM image) and verifies it can see the engine (D-038, g4bench.image_setup).
    return (modal.Image.from_registry(ref, setup_dockerfile_commands=python_setup_commands(engine_module))
            .entrypoint([]).env(ENV).add_local_dir(str(REPO_LOCAL), REPO_REMOTE, ignore=IGNORE))


vllm_image = _engine_image(VLLM_IMAGE, "vllm")
sglang_image = _engine_image(SGLANG_IMAGE, "sglang")
bridge_image = (modal.Image.debian_slim(python_version="3.13")
                # Exactly the versions the CPU test-suite passed with (see STATUS.md).
                .pip_install("fastapi[standard]==0.162.4", "pyyaml==6.0.3", "numpy==2.5.4", "requests==2.34.2",
                             "huggingface_hub==2.4.0", "aiohttp==3.14.5", "transformers==5.22.2", "tqdm==4.70.1",
                             "jinja2==3.1.6", "pytest==9.2.0", "pyarrow==26.0.1", "httpx==0.28.1")
                .env(ENV).add_local_dir(str(REPO_LOCAL), REPO_REMOTE, ignore=IGNORE))

hf_vol = modal.Volume.from_name("g4repro-hf-cache", create_if_missing=True)
res_vol = modal.Volume.from_name("g4repro-results", create_if_missing=True)
budget_dict = modal.Dict.from_name("g4repro-budget", create_if_missing=True)

app = modal.App(APP_NAME)


# ----------------------------------------------------------------------------- ledger (serialized)

@app.function(image=bridge_image, max_containers=1, timeout=60, retries=0, cpu=(0.25, 0.5), memory=(256, 512),
              scaledown_window=60)
def ledger_op(op: str, kw: dict) -> dict:
    """The ONLY writer of the budget ledger: one container, one input at a time."""
    try:
        return {"ok": True, "value": B.Ledger(budget_dict, LEDGER_CFG).op(op, **kw)}
    except (B.BudgetError, B.ConcurrencyError, B.LedgerStateError) as e:
        return {"ok": False, "error": type(e).__name__, "message": str(e)}
    except Exception as e:  # noqa: BLE001 - fail closed
        return {"ok": False, "error": "LedgerStateError", "message": f"unexpected: {e!r}"}


# ----------------------------------------------------------------------------- workers

def _commit_loop(stop: "threading.Event") -> None:  # noqa: F821
    while not stop.wait(60):
        try:
            res_vol.commit()
        except Exception:  # noqa: BLE001
            pass


def _guarded(fn, spec: dict) -> dict:
    from g4bench.bridge_jobs import guarded

    return guarded(fn, spec, hard_kill=True)  # os._exit at the reservation's end deadline (D-030)


def _run_job(spec: dict) -> dict:
    import threading

    from g4bench.bridge_jobs import run_job

    hf_vol.reload()
    stop = threading.Event()
    threading.Thread(target=_commit_loop, args=(stop,), daemon=True).start()
    try:
        return run_job(spec, pathlib.Path(RESULTS_MOUNT) / "runs")
    finally:
        stop.set()
        res_vol.commit()


_gpu_kw = dict(gpu=GPU_JOB.gpu, timeout=GPU_DEFAULT_TIMEOUT_S, startup_timeout=int(B.STARTUP_MARGIN_S), volumes={HF_HOME: hf_vol, RESULTS_MOUNT: res_vol},
               max_containers=1, retries=0, scaledown_window=SCALEDOWN_WINDOW_S,
               cpu=(4.0, GPU_JOB.cpu_limit), memory=(32768, int(GPU_JOB.mem_limit_gib * 1024)))


@app.function(image=vllm_image, **_gpu_kw)
def run_vllm(spec: dict) -> dict:
    return _guarded(_run_job, spec)


@app.function(image=sglang_image, **_gpu_kw)
def run_sglang(spec: dict) -> dict:
    return _guarded(_run_job, spec)


def _prefetch(spec: dict) -> dict:
    from huggingface_hub import snapshot_download

    from g4bench import speedbench
    from g4bench.config import load_pins

    pins = load_pins()
    out: dict = {}
    for role in ("target", "draft"):
        m = pins["model"][role]
        t0 = time.time()
        path = snapshot_download(m["repo"], revision=m["revision"])
        out[role] = {"repo": m["repo"], "revision": m["revision"], "path": path, "seconds": round(time.time() - t0, 1)}
    data, cache = pathlib.Path(os.environ["G4_DATA_DIR"]), pathlib.Path(HF_HOME) / "g4data" / "cache"
    out["speedbench"] = {
        "tp1k": speedbench.prepare_throughput_inline("throughput_1k", data, cache)["prompt_digest"],
        "tp8k": speedbench.prepare_throughput_inline("throughput_8k", data, cache)["prompt_digest"],
        "coding": speedbench.prepare_coding(data, cache)["prompt_digest"],  # raises unless InferenceX digest
    }
    hf_vol.commit()
    return out


@app.function(image=bridge_image, timeout=PREFETCH.timeout_s, startup_timeout=600, volumes={HF_HOME: hf_vol},
              retries=0, max_containers=1,
              scaledown_window=SCALEDOWN_WINDOW_S, cpu=(PREFETCH.cpu_limit, PREFETCH.cpu_limit),
              memory=(int(PREFETCH.mem_limit_gib * 1024), int(PREFETCH.mem_limit_gib * 1024)))
def prefetch_weights(spec: dict) -> dict:
    """CPU-only: pinned target + draft snapshots and verified SPEED-Bench sets into the HF cache Volume."""
    return _guarded(_prefetch, spec)


def _selftest(spec: dict) -> dict:
    import subprocess

    p = subprocess.run(["python", "-m", "pytest", "-q", f"{REPO_REMOTE}/tests"], capture_output=True, text=True,
                       cwd=REPO_REMOTE)
    return {"rc": p.returncode, "tail": (p.stdout + p.stderr)[-4000:]}


@app.function(image=bridge_image, timeout=SELFTEST.timeout_s, startup_timeout=600, retries=0, max_containers=1,
              scaledown_window=SCALEDOWN_WINDOW_S, cpu=(SELFTEST.cpu_limit, SELFTEST.cpu_limit),
              memory=(int(SELFTEST.mem_limit_gib * 1024), int(SELFTEST.mem_limit_gib * 1024)))
def selftest(spec: dict) -> dict:
    """CPU-only: run the repo's test-suite inside Modal (validates image + code)."""
    return _guarded(_selftest, spec)


# ----------------------------------------------------------------------------- Modal backend for the API

class ModalBackend:
    _ERR = {"BudgetError": B.BudgetError, "ConcurrencyError": B.ConcurrencyError,
            "LedgerStateError": B.LedgerStateError}

    def ledger(self, op: str, **kw):
        try:
            r = ledger_op.remote(op, kw)
        except Exception as e:  # noqa: BLE001 - transport/Modal errors: fail closed (HTTP 503)
            raise B.LedgerStateError(f"ledger call failed: {e!r}"[:500]) from e
        if not r.get("ok"):
            raise self._ERR.get(r.get("error"), B.LedgerStateError)(r.get("message"))
        return r["value"]

    def spawn(self, kind: str, payload: dict, *, gpu: str | None, timeout_s: int) -> str:
        if kind == "cpu:prefetch":
            fn = prefetch_weights.with_options(timeout=timeout_s)
        elif kind == "cpu:selftest":
            fn = selftest.with_options(timeout=timeout_s)
        elif kind in ("gpu:vllm", "gpu:sglang"):
            fn = (run_vllm if kind == "gpu:vllm" else run_sglang).with_options(timeout=timeout_s, gpu=gpu)
        else:
            raise ValueError(f"unknown job kind {kind}")
        return fn.spawn(payload).object_id

    def poll(self, call_id: str):
        import modal.exception as me

        try:
            return "finished", modal.FunctionCall.from_id(call_id).get(timeout=0)
        except me.FunctionTimeoutError:     # subclass of modal TimeoutError: must be caught first (D-030)
            return "failed", {"error": "Modal function timeout (worst case reached)"}
        except me.OutputExpiredError:       # subclass of modal TimeoutError too
            return "expired", None
        except (me.TimeoutError, TimeoutError):
            return "running", None          # plain "no output yet" (modal's TimeoutError is NOT builtins')
        except (me.RemoteError, me.ExecutionError, me.InternalFailure) as e:
            # A non-success *result* came back (cancelled/terminated, crashed, OOM, startup failure): terminal.
            return "failed", {"error": f"{type(e).__name__}: {e!r}"[:500]}
        except BaseException as e:  # noqa: BLE001 - transport errors etc.: NOT terminal; reservation stays held
            if isinstance(e, (KeyboardInterrupt, SystemExit)):
                raise
            return "unknown", {"error": repr(e)[:500]}

    def cancel(self, call_id: str) -> None:
        modal.FunctionCall.from_id(call_id).cancel(terminate_containers=True)

    def list_runs(self) -> list[str]:
        res_vol.reload()
        root = pathlib.Path(RESULTS_MOUNT) / "runs"
        return sorted(p.name for p in root.iterdir()) if root.exists() else []

    def list_run_files(self, run_id: str):
        from g4bench.bridge_jobs import list_run_files

        res_vol.reload()
        return list_run_files(pathlib.Path(RESULTS_MOUNT) / "runs", run_id)

    def read_run_file(self, run_id: str, path: str) -> str | None:
        from g4bench.bridge_jobs import read_run_file

        res_vol.reload()
        return read_run_file(pathlib.Path(RESULTS_MOUNT) / "runs", run_id, path)


@app.function(image=bridge_image, volumes={RESULTS_MOUNT: res_vol}, timeout=600, cpu=(0.25, 1.0),
              memory=(512, 1024), scaledown_window=60, max_containers=1)  # metered as platform spend (D-030)
@modal.concurrent(max_inputs=16)
@modal.asgi_app(label="g4repro-bridge", requires_proxy_auth=True)
def bridge():
    return build_api(ModalBackend(), BridgeConfig(
        cap_usd=PROJECT_CAP_USD, per_job_cap_usd=PER_JOB_CAP_USD, max_active=MAX_ACTIVE, app_name=APP_NAME,
        extra={"code_sha": CODE_SHA, "platform_reserve_usd": PLATFORM_RESERVE_USD, "gpu_default_timeout_s": GPU_DEFAULT_TIMEOUT_S,
               "billing_note": B.DISCLAIMER}))
