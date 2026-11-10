"""HTTPS client for the deployed bridge. Uses only documented Modal web-endpoint auth.

Configuration (never printed):
  G4_BRIDGE_URL        e.g. https://<workspace>--g4repro-bridge.us-east.modal.run (default)
  Auth, one of:
    - nothing, when an egress proxy injects ``Authorization: Bearer <id>.<secret>`` for the host
    - MODAL_PROXY_TOKEN_ID + MODAL_PROXY_TOKEN_SECRET  (sent as Modal-Key / Modal-Secret)

Usage:
  python modal_bridge/client.py health
  python modal_bridge/client.py budget
  python modal_bridge/client.py estimate --engine vllm-0.33.0-mtp4 --matrix smoke --max-usd 2
  python modal_bridge/client.py submit   --engine vllm-0.33.0-mtp4 --matrix smoke --max-usd 2
  python modal_bridge/client.py status <job_id>
  python modal_bridge/client.py wait <job_id> [--poll 120] [--max-s 3600]
  python modal_bridge/client.py cancel <job_id>

``submit``/``prefetch``/``selftest`` first read /budget and refuse locally if the remaining estimate cannot
cover the per-job cap; the server enforces the project cap regardless (its ledger is authoritative for the
bridge, and is an estimate, not a Modal billing hard stop).
  python modal_bridge/client.py runs | fetch <run_id> <dest_dir>
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

import requests

# The cloud environment's network secret is bound to the regional host (checked 2026-11-10 via the environment's
# settings); G4_BRIDGE_URL, when present in the process environment, always wins.
DEFAULT_URL = "https://<workspace>--g4repro-bridge.us-east.modal.run"


def _session() -> tuple[requests.Session, str]:
    url = os.environ.get("G4_BRIDGE_URL", DEFAULT_URL).rstrip("/")
    s = requests.Session()
    tid, tsec = os.environ.get("MODAL_PROXY_TOKEN_ID"), os.environ.get("MODAL_PROXY_TOKEN_SECRET")
    if tid and tsec:
        s.headers.update({"Modal-Key": tid, "Modal-Secret": tsec})
    return s, url


def _call(method: str, path: str, **kw) -> dict | list | str:
    s, url = _session()
    r = s.request(method, url + path, timeout=kw.pop("timeout", 60), **kw)
    if r.status_code >= 400:
        # Never echo request headers; body only.
        raise SystemExit(f"HTTP {r.status_code} {path}: {r.text[:1000]}")
    ct = r.headers.get("content-type", "")
    return r.json() if "json" in ct else r.text


def _precheck(max_usd: float | None) -> None:
    b = _call("GET", "/budget")
    need = min(max_usd if max_usd is not None else 0.25, b["per_job_cap_usd"])
    if b["active"]:
        raise SystemExit(f"refusing: {len(b['active'])} job(s) still active per /budget")
    if b["available_usd"] < need:
        raise SystemExit(f"refusing: available ${b['available_usd']:.2f} < ${need:.2f} needed (cap ${b['cap_usd']:.2f})")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("health")
    sub.add_parser("budget")
    sub.add_parser("verify", help="free: check /health + /budget show the approved $10 / <= $2 / 1-active limits")
    sub.add_parser("runs")
    sub.add_parser("jobs")
    sub.add_parser("prefetch")
    sub.add_parser("selftest")
    for name in ("estimate", "submit"):
        p = sub.add_parser(name)
        p.add_argument("--engine", required=True)
        p.add_argument("--matrix", default="smoke")
        p.add_argument("--seeds", type=lambda s: [int(x) for x in s.split(",")])
        p.add_argument("--trials", type=int)
        p.add_argument("--only", type=lambda s: s.split(","))
        p.add_argument("--gpu", default="RTX-PRO-6000")
        p.add_argument("--max-usd", type=float, required=True)
        p.add_argument("--notes", default="")
        p.add_argument("--dry-run", action="store_true")
    for name in ("status", "cancel"):
        sub.add_parser(name).add_argument("job_id")
    w = sub.add_parser("wait")
    w.add_argument("job_id")
    w.add_argument("--poll", type=float, default=120)  # polling keeps the bridge container warm (platform reserve)
    w.add_argument("--max-s", type=float, default=3600)
    f = sub.add_parser("fetch")
    f.add_argument("run_id")
    f.add_argument("dest")
    a = ap.parse_args(argv)

    if a.cmd == "verify":
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
        from g4bench import campaign as C

        req = C.approved_limits({"cap_usd": 10.0, "per_job_cap_usd_max": 2.0, "max_active": 1})

        def call(method, path, json=None):
            try:
                return _call(method, path)
            except SystemExit as e:
                raise C.CampaignStop(str(e)) from None

        try:
            b = C.check_limits(call, req, need_usd=0.0)
            out = {"verified": True, "required": req, "available_usd": b["available_usd"],
                   "committed_usd": b["committed_usd"], "note": "free check; no job launched"}
        except C.CampaignStop as e:
            print(json.dumps({"verified": False, "required": req, "reason": str(e)}, indent=2))
            return 1
    elif a.cmd == "health":
        out = _call("GET", "/health")
    elif a.cmd == "budget":
        out = _call("GET", "/budget")
    elif a.cmd == "runs":
        out = _call("GET", "/runs")
    elif a.cmd == "jobs":
        out = _call("GET", "/jobs")
    elif a.cmd in ("prefetch", "selftest"):
        _precheck(None)
        out = _call("POST", f"/{a.cmd}")
    elif a.cmd in ("estimate", "submit"):
        spec = {k: v for k, v in {"engine": a.engine, "matrix": a.matrix, "seeds": a.seeds, "trials": a.trials,
                                  "only": a.only, "gpu": a.gpu, "max_usd": a.max_usd, "notes": a.notes,
                                  "dry_run": a.dry_run}.items() if v is not None}
        if a.cmd == "submit":
            _precheck(a.max_usd)
        out = _call("POST", "/estimate" if a.cmd == "estimate" else "/jobs", json=spec)
    elif a.cmd == "status":
        out = _call("GET", f"/jobs/{a.job_id}")
    elif a.cmd == "cancel":
        out = _call("POST", f"/jobs/{a.job_id}/cancel")
    elif a.cmd == "wait":
        t0 = time.time()
        while True:
            out = _call("GET", f"/jobs/{a.job_id}")
            if out.get("state") not in ("running", "unknown"):
                break
            if time.time() - t0 > a.max_s:
                print(f"max wait exceeded; cancelling {a.job_id}", file=sys.stderr)
                _call("POST", f"/jobs/{a.job_id}/cancel")
                a.max_s = float("inf")  # keep polling until termination is confirmed (settles the ledger)
            print(f"[{time.time()-t0:.0f}s] running", file=sys.stderr, flush=True)
            time.sleep(a.poll)
    elif a.cmd == "fetch":
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
        from g4bench.campaign import fetch_run

        out = fetch_run(lambda m, p, json=None: _call(m, p), a.run_id, pathlib.Path(a.dest))
    print(json.dumps(out, indent=2) if not isinstance(out, str) else out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
