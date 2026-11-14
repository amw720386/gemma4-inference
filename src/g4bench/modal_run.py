"""Launch paid Modal work directly from this machine, gated by the local spend ledger (D-039).

    g4bench modal spend [--reconcile]                       free: ledger + Modal billing for g4repro* apps
    g4bench modal selftest --execute                        CPU: the test-suite on Linux inside Modal (~1 cent)
    g4bench modal prefetch [--speedbench] --execute         CPU: weights (+ SPEED-Bench) into the cache Volume
    g4bench modal job --engine E --matrix M [--only K] [--seeds 0] [--max-usd 2] --execute
    g4bench modal job --engine E --matrix M --profile-key 8192x4 [--profile-seconds 4]   E5g torch profile
    g4bench modal job --engine E --matrix M --only 1024x1 --seeds 0 --correctness-only --execute   gates only
    g4bench modal pair --a A --b B --matrix M --only 8192x4 --repeats 2 --order ABBA [--execute]   same-GPU pairs

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


def gpu_kind(engine_name: str, engine: str) -> str:
    """``gpu:<engine>``, or ``gpu:vllm-patched`` when the config lists image_patches (must equal VLLM_PATCHES)."""
    from .config import EngineConfig

    patches = tuple(EngineConfig.load(engine_name).raw.get("image_patches") or ())
    if not patches:
        return f"gpu:{engine}"
    from .images import ALWAYS_ON_PATCHES, VLLM_PATCHES, VLLM_PATCHES_EXP

    if engine == "vllm" and not set(patches) <= set(VLLM_PATCHES) and set(patches) <= set(VLLM_PATCHES_EXP) \
            and ALWAYS_ON_PATCHES <= set(patches):
        return "gpu:vllm-patched-exp"
    if engine != "vllm" or not set(patches) <= set(VLLM_PATCHES) or not ALWAYS_ON_PATCHES <= set(patches):
        raise SystemExit(f"{engine_name}: image_patches {list(patches)} unsupported; the patched image is "
                         f"vllm + {list(VLLM_PATCHES)} (configs must include {sorted(ALWAYS_ON_PATCHES)})")
    return "gpu:vllm-patched"


def pair_kind(a: str, b: str) -> tuple[str, list[str]]:
    """One image for both engines of a paired job, plus warnings about what that image changes for either config.

    Same home image -> that image. A stock config (no image_patches) with a patched one -> the experimental image:
    every patch there is env-gated and its default cuBLAS is the stock image's, so the stock recipe runs unchanged;
    the production patched image cannot host it (cuBLAS 13.10 upgraded in place)."""
    from .config import EngineConfig
    from .images import CUBLAS_SIDE, VLLM_PATCH_PIP

    eng = allowed_engines()
    for n in (a, b):
        if n not in eng:
            raise SystemExit(f"unknown/non-GPU engine config {n!r}")
    ka, kb = gpu_kind(a, eng[a]), gpu_kind(b, eng[b])
    if ka == kb:
        return ka, []
    if "gpu:sglang" in (ka, kb):
        raise SystemExit(f"cannot pair {a} ({ka}) with {b} ({kb}): different engine images")
    kind, warn = "gpu:vllm-patched-exp", []
    for n, k in ((a, ka), (b, kb)):
        if k == "gpu:vllm-patched":
            cfg = EngineConfig.load(n)
            if not cfg.env.get("G4_CUBLAS"):
                warn.append(f"{n} normally runs on the production patched image with {list(VLLM_PATCH_PIP)} upgraded "
                            f"in place; on the experimental image it gets that image's default cuBLAS (13.1) unless "
                            f"its config sets G4_CUBLAS (available: {sorted(CUBLAS_SIDE)})")
    return kind, warn


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
    app, fn = {"prefetch": (M.cpu_app, M.prefetch_weights), "selftest": (M.cpu_app, M.selftest),
               "gpu:vllm": (M.vllm_app, M.run_vllm), "gpu:sglang": (M.sglang_app, M.run_sglang),
               "gpu:vllm-patched": (M.vllm_patched_app, M.run_vllm_patched),
               "gpu:vllm-patched-exp": (M.vllm_patched_exp_app, M.run_vllm_patched_exp),
               "gpu:script": (M.vllm_app, M.run_script_vllm)}[kind]
    result: dict[str, Any] | None = None
    err = None
    now: float | None = None
    try:
        # app.run() builds/pulls images first (no function container runs yet); the reservation and its start
        # deadline begin only after that, immediately before the one paid call.
        with modal.enable_output(), app.run() as running:
            now = time.time()
            spec = {**spec, "job_id": lid, "start_deadline_unix": now + S.STARTUP_MARGIN_S,
                    "deadline_unix": now + S.STARTUP_MARGIN_S + p["timeout_s"]}
            if kind.startswith("gpu:"):
                spec["max_wall_s"] = max(p["timeout_s"] - SOFT_WALL_HEADROOM_S, 0)
            S.append({"launch_id": lid, "kind": kind, "status": "reserved", "worst_usd": p["worst_usd"],
                      "rate_usd_s": p["rate_usd_s"], "timeout_s": p["timeout_s"], "reserved_unix": now,
                      "deadline_unix": spec["deadline_unix"], "max_usd": max_usd, "app_id": running.app_id,
                      "spec": {k: v for k, v in spec.items() if k not in ("start_deadline_unix", "deadline_unix")},
                      "committed_before_usd": pre["committed_usd"]}, ledger)
            result = fn.with_options(timeout=p["timeout_s"]).remote(spec)
    except BaseException as e:  # noqa: BLE001 - record, settle, re-raise interrupts
        err = e
    if now is None:  # failed while creating the app / building images: no paid function call was made
        raise err if err is not None else RuntimeError("app.run() exited without a call")
    end = time.time()
    # Settle from wall time since reservation (covers container start); ephemeral app is stopped by now.
    settled = round(p["rate_usd_s"] * (end - now + S.SCALEDOWN_MARGIN_S), 4)
    files = (result or {}).pop("files", {}) if isinstance(result, dict) else {}
    multi = (result or {}).pop("files_multi", None) if isinstance(result, dict) else None
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
    if multi:  # paired job: {"runs": {run_id: files}, "pairs": {pair_id: files}} -> results/<sub>/<id>
        out["run_dirs"] = write_multi(dest.parent, multi)
    if kind == "prefetch" and out.get("volume_gib") is not None:
        S.append({"kind": "storage", "gib": out["volume_gib"], "unix": time.time(),
                  "volume": "g4repro-hf-cache"}, ledger)
    out.pop("rows", None)
    return out


def write_multi(results_root: Path, multi: dict[str, dict[str, dict[str, bytes]]]) -> list[str]:
    """Write a paired job's returned files: only the ``runs`` and ``pairs`` subtrees are accepted."""
    from .jobs import write_run_files

    written = []
    for sub, by_id in multi.items():
        if sub not in ("runs", "pairs"):
            raise ValueError(f"unexpected result subtree {sub!r}")
        for rid, files in (by_id or {}).items():
            if files:
                written.append(str(write_run_files(results_root / sub, rid, files)))
    return written


def pair_spec(args, code_sha: str) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """(kind, spec, sizing) for ``g4bench modal pair``; sizing says whether the sequence fits the reservation."""
    from .matrix import expand, load_matrix
    from .paired import ORDERS, plan_sequence

    if args.order not in ORDERS:
        raise SystemExit(f"--order must be one of {sorted(ORDERS)}")
    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1")
    kind, warnings = pair_kind(args.a, args.b)
    seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else None
    only = args.only.split(",") if args.only else None
    pts = expand(load_matrix(args.matrix), seeds=seeds, trials=args.repeats, only_keys=only)
    if not pts:
        raise SystemExit("no points selected")
    p = plan(kind, args.max_usd)
    sizing = plan_sequence(pts, max_wall_s=max(p["timeout_s"] - SOFT_WALL_HEADROOM_S, 0),
                           correctness=not args.no_correctness, rate_usd_s=p["rate_usd_s"])
    sizing["warnings"] = warnings
    sizing["sequence"] = [f"{q.point_id}:{''.join(_seq(args.order, q.trial))}" for q in pts]
    spec = {"paired": {"engines": [args.a, args.b], "order": args.order, "repeats": args.repeats},
            "engine": None, "matrix": args.matrix, "notes": args.notes, "seeds": seeds, "only": only,
            "code_sha": code_sha, "greedy": not args.no_greedy}
    if args.no_correctness:
        spec["correctness"] = False
    return kind, spec, sizing


def _seq(order: str, trial: int) -> str:
    from .paired import segment_order

    return segment_order(order, trial)


def main(args) -> int:
    load_dotenv()
    # Before anything imports modal_app: its image ENV captures G4_CODE_SHA at import time, and gpu_kind() imports
    # modal_app for patched configs. Setting it later (as before) left every patched-image run with sha=null.
    code_sha = os.environ["G4_CODE_SHA"] = _code_sha()
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
        from .test_gate import TestGateError, check_selftest_interval

        try:  # owner policy: the suite runs on Modal at most once every 2.5 h (src/g4bench/test_gate.py)
            check_selftest_interval(S.read())
        except TestGateError as e:
            raise SystemExit(f"selftest refused: {e}") from None
        kind, spec = "selftest", {}
    elif args.modal_cmd == "pair":
        kind, spec, sizing = pair_spec(args, code_sha)
        p = plan(kind, args.max_usd)
        if not args.execute:
            print(json.dumps({"plan_only": True, **p, "spec": spec, "sizing": sizing, "budget": preflight(p)},
                             indent=2))
            return 0
        if not sizing["fits"]:
            raise SystemExit("paired sequence does not fit this reservation: " + sizing.get("suggestion", ""))
        out = launch(kind, spec, args.max_usd, dest=RESULTS / "runs")
        print(json.dumps(out, indent=2, default=str))
        return 0 if not out.get("error") and not out.get("aborted") else 1
    elif args.modal_cmd == "script":
        if not (REPO / args.path).is_file():
            raise SystemExit(f"no such script {args.path}")
        kind, spec = "gpu:script", {"script": Path(args.path).as_posix(), "args": args.args or []}
    else:
        eng = allowed_engines().get(args.engine)
        if not eng:
            raise SystemExit(f"unknown/non-GPU engine config {args.engine!r}")
        kind = gpu_kind(args.engine, eng)
        spec = {"engine": args.engine, "matrix": args.matrix, "notes": args.notes,
                "seeds": [int(x) for x in args.seeds.split(",")] if args.seeds else None,
                "only": args.only.split(",") if args.only else None, "code_sha": code_sha,
                "greedy": not args.no_greedy, "correctness_only": bool(args.correctness_only)}
        if args.greedy_reference:  # repo-relative path; the repo (minus results/) is uploaded to /repo
            ref = Path(args.greedy_reference)
            if ref.is_absolute() or not (REPO / ref).is_file() or ref.parts[0] == "results":
                raise SystemExit(f"--greedy-reference must be a repo-relative file outside results/: {ref}")
            spec["greedy_reference"] = ref.as_posix()
        if args.profile_key:  # E5g: torch-profiler window on one point (rows flagged ``profiled``)
            spec["profile"] = {"key": args.profile_key, "duration_s": args.profile_seconds,
                               "warmup_s": args.profile_warmup_s}
            spec["only"] = spec["only"] or [args.profile_key]
            if not spec["seeds"]:
                spec["seeds"] = [0]
    p = plan(kind, args.max_usd)
    if not args.execute:
        print(json.dumps({"plan_only": True, **p, "spec": spec,
                          "budget": preflight(p)}, indent=2))
        return 0
    out = launch(kind, spec, args.max_usd, dest=RESULTS / ("kernels" if kind == "gpu:script" else "runs"))
    print(json.dumps(out, indent=2, default=str))
    return 0 if not out.get("error") and not out.get("aborted") and out.get("rc", 0) == 0 else 1


def add_parser(sub) -> None:
    m = sub.add_parser("modal", help="paid Modal runs launched from this machine (budget-gated)")
    ms = m.add_subparsers(dest="modal_cmd", required=True)
    s = ms.add_parser("spend", help="free: committed spend vs the project cap")
    s.add_argument("--reconcile", action="store_true", help="record Modal's billed cost per launch in the ledger")
    s.add_argument("--offline", action="store_true", help="skip Modal billing (ledger only)")
    pf = ms.add_parser("prefetch", help="CPU: download pinned weights into the cache Volume")
    pf.add_argument("--speedbench", action="store_true")
    pf.add_argument("--max-usd", type=float, default=0.2)
    st = ms.add_parser("selftest", help="CPU: the unit-test suite on Linux inside Modal (~1 cent; at most every 2.5 h)")
    st.add_argument("--max-usd", type=float, default=0.1)
    sc = ms.add_parser("script", help="GPU kernel lab: run a repo script inside the pinned vLLM image")
    sc.add_argument("--path", required=True, help="repo-relative script, e.g. kernels/gemm_bench.py")
    sc.add_argument("--max-usd", type=float, default=1.75)
    sc.add_argument("args", nargs="*", help="arguments passed to the script (after --)")
    j = ms.add_parser("job", help="GPU: run one engine config over a matrix")
    j.add_argument("--engine", required=True)
    j.add_argument("--matrix", required=True)
    j.add_argument("--seeds")
    j.add_argument("--only")
    j.add_argument("--notes", default="")
    j.add_argument("--max-usd", type=float, default=2.0)
    j.add_argument("--profile-key", help="E5g: torch-profile this workload key (e.g. 8192x4); rows flagged profiled")
    j.add_argument("--profile-seconds", type=float, default=4.0, help="profiler window length (keep short)")
    j.add_argument("--profile-warmup-s", type=float, default=30.0,
                   help="seconds after the server first counts prompt tokens (the client's first request; "
                        "client launch if the metric is absent) before /start_profile")
    j.add_argument("--no-greedy", action="store_true", help="skip the token-level greedy-equivalence capture")
    j.add_argument("--correctness-only", action="store_true",
                   help="start the server for the first selected point, run probes + greedy gate, no benchmark")
    j.add_argument("--greedy-reference", help="repo-relative reference JSON (default "
                   "data/greedy_reference/vllm-0.33.0-nospec.json)")
    pr = ms.add_parser("pair", help="GPU: engines A and B interleaved on ONE GPU in one container (D-061)")
    pr.add_argument("--a", required=True, help="reference engine config, e.g. vllm-0.33.0-mtp4 (article recipe)")
    pr.add_argument("--b", required=True, help="candidate engine config")
    pr.add_argument("--matrix", required=True)
    pr.add_argument("--only")
    pr.add_argument("--seeds")
    pr.add_argument("--repeats", type=int, default=2, help="pairs per point (each repeat = one A and one B segment)")
    pr.add_argument("--order", default="ABBA", help="ABBA (AB, BA, ...), ABAB, or the mirrors BAAB / BABA")
    pr.add_argument("--notes", default="")
    pr.add_argument("--max-usd", type=float, default=3.0)
    pr.add_argument("--no-greedy", action="store_true", help="probes only in the correctness gate")
    pr.add_argument("--no-correctness", action="store_true", help="skip the correctness gate entirely")
    for x in (pf, st, sc, j, pr):
        x.add_argument("--execute", action="store_true", help="actually launch (default: plan only)")
