"""Launch paid Modal work directly from this machine, gated by the local spend ledger (D-039).

    g4bench modal spend [--reconcile]                       free: ledger + Modal billing for g4repro* apps
    g4bench modal selftest --execute                        CPU: the test-suite on Linux inside Modal (~1 cent)
    g4bench modal prefetch [--speedbench] --execute         CPU: weights (+ SPEED-Bench) into the cache Volume
    g4bench modal job --engine E --matrix M [--only K] [--seeds 0] [--max-usd 2] --execute

Without ``--execute`` nothing is launched: the command prints the reservation it would make. Every launch:
reconcile with Modal billing -> refuse unless worst case fits the per-job and remaining project budget ->
append ``reserved`` -> ephemeral ``app.run()`` with a Modal timeout derived from the reservation -> append
``settled`` (wall time since reservation, before container start) -> write the returned run files locally.
Credentials: ``MODAL_TOKEN_ID`` / ``MODAL_TOKEN_SECRET`` from the environment or ``.env`` (never committed).
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

from . import spend as S
from .paths import REPO, RESULTS

SOFT_WALL_HEADROOM_S = 180  # runner stops starting new points this long before the Modal timeout


def load_dotenv(path: Path = REPO / ".env") -> None:
    """Export MODAL_TOKEN_* from .env unless already set (values are never printed)."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        k, sep, v = line.strip().partition("=")
        if sep and k.startswith(("MODAL_", "G4_")) and not os.environ.get(k):
            os.environ[k] = v.strip().strip('"').strip("'")


def allowed_engines() -> dict[str, str]:
    from .config import EngineConfig
    from .paths import CONFIGS

    out = {}
    for f in sorted((CONFIGS / "engines").glob("*.yaml")):
        e = EngineConfig.load(str(f))
        if e.engine in ("vllm", "sglang"):
            out[e.name] = e.engine
    return out


def _code_sha() -> str:
    import subprocess

    try:
        sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True,
                             timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, timeout=10).stdout.strip()
        return (sha + ("-dirty" if dirty else "")) if sha else "unknown"
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"


def plan(kind: str, max_usd: float) -> dict[str, Any]:
    if kind in ("prefetch", "selftest"):
        rate = S.PREFETCH.rate()
        cap = S.PREFETCH_TIMEOUT_S if kind == "prefetch" else S.SELFTEST_TIMEOUT_S
        timeout = min(cap, S.timeout_for_budget(rate, max_usd))
    else:
        rate = S.GPU_JOB.rate()
        timeout = S.timeout_for_budget(rate, max_usd)
    if timeout <= 0:
        raise S.BudgetError(f"${max_usd:.2f} buys no runtime after the {S.STARTUP_MARGIN_S:.0f}s startup margin")
    return {"kind": kind, "rate_usd_s": rate, "rate_usd_h": round(rate * 3600, 4), "timeout_s": timeout,
            "worst_usd": round(S.worst_case_usd(rate, timeout), 4)}


def preflight(p: dict[str, Any], *, ledger: Path = S.LEDGER, offline: bool = False) -> dict[str, Any]:
    """Reconcile with Modal billing, then check caps. Fails closed if billing is unreadable (unless offline)."""
    caps = S.Caps.from_env()
    rows = None
    if not offline:
        rows = S.modal_billing()
        S.reconcile(S.read(ledger), rows, ledger)
    entries = S.read(ledger)
    billed = sum(float(r["cost"]) for r in rows) if rows is not None else None
    remaining = S.check_launch(p["worst_usd"], caps, entries, billed)
    return {**S.summary(caps, entries, rows), "remaining_after_worst_case_usd": round(remaining, 4)}


def launch(kind: str, spec: dict[str, Any], max_usd: float, *, ledger: Path = S.LEDGER,
           dest: Path = RESULTS / "runs") -> dict[str, Any]:
    import modal

    from . import modal_app as M

    p = plan(kind, max_usd)
    pre = preflight(p, ledger=ledger)
    lid = f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{uuid.uuid4().hex[:6]}"
    now = time.time()
    spec = {**spec, "job_id": lid, "start_deadline_unix": now + S.STARTUP_MARGIN_S,
            "deadline_unix": now + S.STARTUP_MARGIN_S + p["timeout_s"]}
    if kind.startswith("gpu:"):
        spec["max_wall_s"] = max(p["timeout_s"] - SOFT_WALL_HEADROOM_S, 0)
    S.append({"launch_id": lid, "kind": kind, "status": "reserved", "worst_usd": p["worst_usd"],
              "rate_usd_s": p["rate_usd_s"], "timeout_s": p["timeout_s"], "reserved_unix": now,
              "deadline_unix": spec["deadline_unix"], "max_usd": max_usd,
              "spec": {k: v for k, v in spec.items() if k not in ("start_deadline_unix", "deadline_unix")},
              "committed_before_usd": pre["committed_usd"]}, ledger)
    app, fn = {"prefetch": (M.cpu_app, M.prefetch_weights), "selftest": (M.cpu_app, M.selftest),
               "gpu:vllm": (M.vllm_app, M.run_vllm), "gpu:sglang": (M.sglang_app, M.run_sglang)}[kind]
    result: dict[str, Any] | None = None
    err = None
    try:
        with modal.enable_output(), app.run() as running:
            S.append({"launch_id": lid, "app_id": running.app_id}, ledger)
            result = fn.with_options(timeout=p["timeout_s"]).remote(spec)
    except BaseException as e:  # noqa: BLE001 - record, settle, re-raise interrupts
        err = e
    end = time.time()
    # Settle from wall time since reservation (covers container start); ephemeral app is stopped by now.
    settled = round(p["rate_usd_s"] * (end - now + S.SCALEDOWN_MARGIN_S), 4)
    files = (result or {}).pop("files", {}) if isinstance(result, dict) else {}
    S.append({"launch_id": lid, "status": "settled", "settled_usd": settled, "end_unix": end,
              "outcome": "error" if err or (result or {}).get("error") or (result or {}).get("aborted") else "ok",
              "error": repr(err)[:500] if err else (result or {}).get("error") or (result or {}).get("aborted"),
              "run_id": (result or {}).get("run_id")}, ledger)
    if err is not None:
        raise err
    out = {"launch_id": lid, "settled_estimate_usd": settled, **(result or {})}
    if files and out.get("run_id"):
        from .jobs import write_run_files

        out["run_dir"] = str(write_run_files(dest, out["run_id"], files))
    if kind == "prefetch" and out.get("volume_gib") is not None:
        S.append({"kind": "storage", "gib": out["volume_gib"], "unix": time.time(),
                  "volume": "g4repro-hf-cache"}, ledger)
    out.pop("rows", None)
    return out


def main(args) -> int:
    load_dotenv()
    if args.modal_cmd == "spend":
        caps = S.Caps.from_env()
        rows = S.modal_billing() if not args.offline else None
        if rows is not None and args.reconcile:
            S.reconcile(S.read(), rows)
        print(json.dumps(S.summary(caps, S.read(), rows), indent=2))
        return 0
    if args.modal_cmd == "prefetch":
        kind, spec = "prefetch", {"speedbench": bool(args.speedbench)}
    elif args.modal_cmd == "selftest":
        kind, spec = "selftest", {}
    else:
        eng = allowed_engines().get(args.engine)
        if not eng:
            raise SystemExit(f"unknown/non-GPU engine config {args.engine!r}")
        kind = f"gpu:{eng}"
        spec = {"engine": args.engine, "matrix": args.matrix, "notes": args.notes,
                "seeds": [int(x) for x in args.seeds.split(",")] if args.seeds else None,
                "only": args.only.split(",") if args.only else None}
    p = plan(kind, args.max_usd)
    if not args.execute:
        print(json.dumps({"plan_only": True, **p, "spec": spec,
                          "budget": preflight(p)}, indent=2))
        return 0
    os.environ["G4_CODE_SHA"] = _code_sha()
    out = launch(kind, spec, args.max_usd)
    print(json.dumps(out, indent=2, default=str))
    return 0 if not out.get("error") and not out.get("aborted") and out.get("rc", 0) == 0 else 1


def add_parser(sub) -> None:
    m = sub.add_parser("modal", help="paid Modal runs launched from this machine (budget-gated)")
    ms = m.add_subparsers(dest="modal_cmd", required=True)
    s = ms.add_parser("spend", help="free: committed spend vs the $10 cap")
    s.add_argument("--reconcile", action="store_true", help="record Modal's billed cost per launch in the ledger")
    s.add_argument("--offline", action="store_true", help="skip Modal billing (ledger only)")
    pf = ms.add_parser("prefetch", help="CPU: download pinned weights into the cache Volume")
    pf.add_argument("--speedbench", action="store_true")
    pf.add_argument("--max-usd", type=float, default=0.2)
    st = ms.add_parser("selftest", help="CPU: run the test-suite on Linux inside Modal (~1 cent)")
    st.add_argument("--max-usd", type=float, default=0.1)
    j = ms.add_parser("job", help="GPU: run one engine config over a matrix")
    j.add_argument("--engine", required=True)
    j.add_argument("--matrix", required=True)
    j.add_argument("--seeds")
    j.add_argument("--only")
    j.add_argument("--notes", default="")
    j.add_argument("--max-usd", type=float, default=2.0)
    for x in (pf, st, j):
        x.add_argument("--execute", action="store_true", help="actually launch (default: plan only)")
