"""HTTPS client for the deployed bridge. Uses only documented Modal web-endpoint auth.

Configuration (never printed):
  G4_BRIDGE_URL        e.g. https://arnold-17059--g4repro-bridge.modal.run
  Auth, one of:
    - nothing, when an egress proxy injects ``Authorization: Bearer <id>.<secret>`` for the host
    - MODAL_PROXY_TOKEN_ID + MODAL_PROXY_TOKEN_SECRET  (sent as Modal-Key / Modal-Secret)

Usage:
  python modal_bridge/client.py health
  python modal_bridge/client.py estimate --engine vllm-0.33.0-mtp4 --matrix smoke --max-usd 2
  python modal_bridge/client.py submit   --engine vllm-0.33.0-mtp4 --matrix smoke --max-usd 2
  python modal_bridge/client.py status <call_id>
  python modal_bridge/client.py wait <call_id> [--poll 60] [--max-s 7200]
  python modal_bridge/client.py cancel <call_id>
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

DEFAULT_URL = "https://arnold-17059--g4repro-bridge.modal.run"


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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("health")
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
        sub.add_parser(name).add_argument("call_id")
    w = sub.add_parser("wait")
    w.add_argument("call_id")
    w.add_argument("--poll", type=float, default=60)
    w.add_argument("--max-s", type=float, default=4 * 3600)
    f = sub.add_parser("fetch")
    f.add_argument("run_id")
    f.add_argument("dest")
    a = ap.parse_args(argv)

    if a.cmd == "health":
        out = _call("GET", "/health")
    elif a.cmd == "runs":
        out = _call("GET", "/runs")
    elif a.cmd == "jobs":
        out = _call("GET", "/jobs")
    elif a.cmd in ("prefetch", "selftest"):
        out = _call("POST", f"/{a.cmd}")
    elif a.cmd in ("estimate", "submit"):
        spec = {k: v for k, v in {"engine": a.engine, "matrix": a.matrix, "seeds": a.seeds, "trials": a.trials,
                                  "only": a.only, "gpu": a.gpu, "max_usd": a.max_usd, "notes": a.notes,
                                  "dry_run": a.dry_run}.items() if v is not None}
        out = _call("POST", "/estimate" if a.cmd == "estimate" else "/jobs", json=spec)
    elif a.cmd == "status":
        out = _call("GET", f"/jobs/{a.call_id}")
    elif a.cmd == "cancel":
        out = _call("POST", f"/jobs/{a.call_id}/cancel")
    elif a.cmd == "wait":
        t0 = time.time()
        while True:
            out = _call("GET", f"/jobs/{a.call_id}")
            if out.get("state") != "running":
                break
            if time.time() - t0 > a.max_s:
                print(f"max wait exceeded; cancelling {a.call_id}", file=sys.stderr)
                out = _call("POST", f"/jobs/{a.call_id}/cancel")
                break
            print(f"[{time.time()-t0:.0f}s] running", file=sys.stderr, flush=True)
            time.sleep(a.poll)
    elif a.cmd == "fetch":
        dest = pathlib.Path(a.dest) / a.run_id
        dest.mkdir(parents=True, exist_ok=True)
        for name in ("manifest.json", "metadata.json", "results.csv", "results.jsonl"):
            try:
                (dest / name).write_text(_call("GET", f"/runs/{a.run_id}/{name}"))
            except SystemExit as e:
                print(f"skip {name}: {e}", file=sys.stderr)
        out = {"fetched_to": str(dest)}
    print(json.dumps(out, indent=2) if not isinstance(out, str) else out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
