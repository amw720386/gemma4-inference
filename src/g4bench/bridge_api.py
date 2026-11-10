"""HTTP API of the g4repro Modal bridge, independent of Modal so it can be unit-tested.

``modal_bridge/app.py`` supplies a ``Backend`` that talks to Modal (spawn / poll / cancel and a serialized
ledger function); tests supply fakes. Every paid action (GPU jobs, CPU prefetch, CPU selftest) goes through
the same path: validate -> reconcile -> reserve worst case in the ledger -> spawn with a Modal timeout that
bounds that worst case -> attach call id; a failed spawn releases the reservation, anything else is held
until a poll confirms the call is terminal and the ledger settles it conservatively.
"""


import math
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from . import budget as B

ALLOWED_ENGINES = {"vllm-0.33.0-mtp4": "vllm", "vllm-0.33.0-mtp4-textonly": "vllm", "vllm-0.33.0-nospec": "vllm",
                   "sglang-0.5.23-nextn": "sglang"}
ALLOWED_MATRICES = {"smoke", "backboard-repro", "speedbench-coding", "speedbench-tp"}
ALLOWED_GPUS = {"RTX-PRO-6000", "L40S", "A100-80GB", "H100!", "H200", "B200"}
TERMINAL = {"finished", "failed", "cancelled", "expired"}


@dataclass(frozen=True)
class Resources:
    cpu_limit: float
    mem_limit_gib: float
    gpu: str | None = None
    timeout_s: int | None = None  # fixed timeout (CPU jobs); GPU jobs derive it from the reservation

    def rate(self, gpu: str | None = None) -> float:
        return B.resource_rate(gpu if gpu is not None else self.gpu, self.cpu_limit, self.mem_limit_gib)


# Must match the cpu/memory *limits* declared in modal_bridge/app.py (tested).
GPU_JOB = Resources(cpu_limit=8.0, mem_limit_gib=64.0, gpu="RTX-PRO-6000")
PREFETCH = Resources(cpu_limit=2.0, mem_limit_gib=4.0, timeout_s=3600)
SELFTEST = Resources(cpu_limit=2.0, mem_limit_gib=4.0, timeout_s=1800)
SOFT_WALL_HEADROOM_S = 180  # runner stops starting new points this long before the Modal timeout


class Backend(Protocol):
    def ledger(self, op: str, **kw: Any) -> dict[str, Any]: ...
    def spawn(self, kind: str, payload: dict[str, Any], *, gpu: str | None, timeout_s: int) -> str: ...
    def poll(self, call_id: str) -> tuple[str, Any]: ...
    def cancel(self, call_id: str) -> None: ...
    def list_runs(self) -> list[str]: ...
    def read_run_file(self, run_id: str, path: str) -> str | None: ...


@dataclass
class BridgeConfig:
    cap_usd: float
    per_job_cap_usd: float
    max_active: int
    app_name: str = "g4repro-bridge"
    estimator: Callable[[dict[str, Any]], dict[str, Any]] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def default_estimator(spec: dict[str, Any]) -> dict[str, Any]:
    from .cost import estimate
    from .matrix import expand, load_matrix

    mat = load_matrix(spec["matrix"])
    pts = expand(mat, seeds=spec.get("seeds"), trials=spec.get("trials"), only_keys=spec.get("only"))
    if not pts:
        raise ValueError("no points selected")
    kind = ALLOWED_ENGINES[spec["engine"]]
    e = estimate(pts, spec["gpu"].rstrip("!"), fresh_server_per_point=bool(mat.get("fresh_server_per_point", True)),
                 tok_s_ref="sglang" if kind == "sglang" else "vllm")
    return {"points": len(pts), "est_gpu_s": round(e.gpu_seconds), "summary": e.summary()}


def build_api(backend: Backend, cfg: BridgeConfig):
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import PlainTextResponse
    from pydantic import BaseModel, Field

    api = FastAPI(title="g4repro bridge")
    estimator = cfg.estimator or default_estimator

    class JobSpec(BaseModel):
        engine: str
        matrix: str = "smoke"
        seeds: list[int] | None = None
        trials: int | None = Field(default=None, ge=1, le=5)
        only: list[str] | None = None
        gpu: str = "RTX-PRO-6000"
        max_usd: float = Field(gt=0)
        notes: str = Field(default="", max_length=500)
        dry_run: bool = False

    def _ledger(op: str, **kw: Any) -> dict[str, Any]:
        try:
            return backend.ledger(op, **kw)
        except B.BudgetError as e:
            raise HTTPException(402, f"budget: {e}") from e
        except B.ConcurrencyError as e:
            raise HTTPException(409, f"concurrency: {e}") from e
        except B.LedgerStateError as e:
            raise HTTPException(503, f"ledger unavailable/inconsistent (failing closed): {e}") from e

    def _refresh(job: dict[str, Any]) -> dict[str, Any]:
        """Poll one open job; settle it if Modal reports it terminal."""
        if job["state"] != "running":
            return {"job": job, "state": job["state"]}
        state, payload = backend.poll(job["call_id"])
        if state not in TERMINAL:
            return {"job": job, "state": state}
        end = payload.get("ended_unix") if state == "finished" and isinstance(payload, dict) else None
        job = _ledger("settle", job_id=job["job_id"], outcome=state, job_reported_end=end)
        return {"job": job, "state": state, "result": payload}

    def _reconcile() -> None:
        for j in list(_ledger("snapshot")["active"].values()):
            _refresh(j)

    def _launch(kind: str, res: Resources, payload: dict[str, Any], *, gpu: str | None, timeout_s: int,
                meta: dict[str, Any]) -> dict[str, Any]:
        rate = res.rate(gpu)
        worst = B.worst_case_usd(rate, timeout_s)
        _reconcile()
        usd = min(cfg.per_job_cap_usd, math.ceil(worst * 100) / 100)  # >= worst because timeout_s was floored
        job = _ledger("reserve", kind=kind, usd=usd, rate_usd_s=rate, timeout_s=timeout_s, meta=meta)
        payload = dict(payload, job_id=job["job_id"], deadline_unix=job["deadline_unix"],
                       max_wall_s=max(timeout_s - SOFT_WALL_HEADROOM_S, 60))
        try:
            call_id = backend.spawn(kind, payload, gpu=gpu, timeout_s=timeout_s)
        except Exception as e:  # noqa: BLE001 - spawn never happened: give the reservation back
            _ledger("release", job_id=job["job_id"], reason=f"spawn failed: {e!r}"[:300])
            raise HTTPException(502, f"spawn failed; reservation released: {e!r}"[:500]) from e
        job = _ledger("attach", job_id=job["job_id"], call_id=call_id)
        return {"job_id": job["job_id"], "call_id": call_id, "reserved_usd": job["reserved_usd"],
                "timeout_s": timeout_s, "deadline_unix": job["deadline_unix"]}

    def _plan(spec: JobSpec) -> dict[str, Any]:
        if spec.engine not in ALLOWED_ENGINES:
            raise HTTPException(400, f"engine not allowed: {spec.engine}")
        if spec.matrix not in ALLOWED_MATRICES:
            raise HTTPException(400, f"matrix not allowed: {spec.matrix}")
        if spec.gpu not in ALLOWED_GPUS:
            raise HTTPException(400, f"gpu not allowed: {spec.gpu}")
        try:
            e = estimator(spec.model_dump())
        except ValueError as ex:
            raise HTTPException(400, str(ex)) from ex
        usd = min(spec.max_usd, cfg.per_job_cap_usd)
        rate = GPU_JOB.rate(spec.gpu)
        timeout_s = B.timeout_for_budget(rate, usd)
        fits = timeout_s >= e["est_gpu_s"]
        return {**e, "rate_usd_s": rate, "reservation_cap_usd": usd, "timeout_s": timeout_s,
                "worst_case_usd": round(B.worst_case_usd(rate, timeout_s), 4) if timeout_s else None,
                "fits": fits, "per_job_cap_usd": cfg.per_job_cap_usd}

    @api.get("/health")
    def health():
        try:
            snap = backend.ledger("snapshot")
            ledger_ok = True
        except Exception:  # noqa: BLE001
            snap, ledger_ok = {}, False
        return {"ok": ledger_ok, "app": cfg.app_name, "ledger_ok": ledger_ok, "cap_usd": cfg.cap_usd,
                "per_job_cap_usd": cfg.per_job_cap_usd, "max_active": cfg.max_active,
                "available_usd": snap.get("available_usd"), "engines": sorted(ALLOWED_ENGINES),
                "matrices": sorted(ALLOWED_MATRICES), "gpus": sorted(ALLOWED_GPUS), **cfg.extra}

    @api.get("/budget")
    def get_budget():
        _reconcile()
        return _ledger("snapshot")

    @api.post("/estimate")
    def est(spec: JobSpec):
        return _plan(spec)

    @api.post("/jobs")
    def submit(spec: JobSpec):
        p = _plan(spec)
        if not p["fits"]:
            raise HTTPException(402, f"estimated {p['est_gpu_s']} s does not fit the {p['timeout_s']} s that a "
                                     f"${p['reservation_cap_usd']:.2f} reservation buys at ${p['rate_usd_s']*3600:.2f}/h; "
                                     "split the job with `only` or seeds")
        kind = "gpu:" + ALLOWED_ENGINES[spec.engine]
        out = _launch(kind, GPU_JOB, spec.model_dump(), gpu=spec.gpu, timeout_s=p["timeout_s"],
                      meta={"engine": spec.engine, "matrix": spec.matrix, "gpu": spec.gpu, "points": p["points"],
                            "est_gpu_s": p["est_gpu_s"], "notes": spec.notes})
        return {**out, "estimate": p}

    @api.post("/prefetch")
    def prefetch():
        return _launch("cpu:prefetch", PREFETCH, {}, gpu=None, timeout_s=PREFETCH.timeout_s, meta={})

    @api.post("/selftest")
    def selftest():
        return _launch("cpu:selftest", SELFTEST, {}, gpu=None, timeout_s=SELFTEST.timeout_s, meta={})

    @api.get("/jobs")
    def list_jobs():
        _reconcile()
        return _ledger("snapshot")["active"]

    @api.get("/jobs/{job_id}")
    def status(job_id: str):
        return _refresh(_ledger("job", job_id=job_id))

    @api.post("/jobs/{job_id}/cancel")
    def cancel(job_id: str):
        j = _ledger("job", job_id=job_id)
        if j["state"] != "running":
            raise HTTPException(409, f"job is {j['state']}")
        backend.cancel(j["call_id"])
        return {"cancel_requested": job_id, "note": "reservation held until a poll confirms termination"}

    @api.get("/runs")
    def runs():
        return backend.list_runs()

    @api.get("/runs/{run_id}/{path:path}", response_class=PlainTextResponse)
    def run_file(run_id: str, path: str):
        txt = backend.read_run_file(run_id, path)
        if txt is None:
            raise HTTPException(404, "not found")
        return txt

    return api
