"""Model-availability test: are the pinned target and draft revisions publicly fetchable?

Uses only documented HF Hub HTTPS endpoints (model info API + resolve HEAD). Does not
download weights. Never sends or prints tokens; works anonymously for ungated repos.
"""

from __future__ import annotations

from typing import Any

import requests

from .config import load_pins

HF = "https://huggingface.co"


def check_repo(repo: str, revision: str, expect_files: tuple[str, ...], timeout: float = 30) -> dict[str, Any]:
    out: dict[str, Any] = {"repo": repo, "revision": revision, "ok": False, "problems": []}
    r = requests.get(f"{HF}/api/models/{repo}/revision/{revision}", timeout=timeout)
    if r.status_code != 200:
        out["problems"].append(f"model info HTTP {r.status_code}")
        return out
    info = r.json()
    out["sha"] = info.get("sha")
    out["gated"] = info.get("gated")
    out["license"] = (info.get("cardData") or {}).get("license")
    out["architectures"] = (info.get("config") or {}).get("architectures")
    names = {s["rfilename"] for s in info.get("siblings", [])}
    missing = [f for f in expect_files if f not in names]
    if missing:
        out["problems"].append(f"missing files {missing}")
    if info.get("sha") != revision:
        out["problems"].append(f"resolved sha {info.get('sha')} != pinned {revision}")
    if info.get("gated") not in (False, None):
        out["problems"].append(f"repo is gated ({info.get('gated')}); needs HF token + license acceptance")
    # HEAD the weights at the pinned revision (follows to CDN; no body downloaded).
    h = requests.head(f"{HF}/{repo}/resolve/{revision}/model.safetensors", allow_redirects=False, timeout=timeout)
    out["weights_head_status"] = h.status_code
    size = h.headers.get("x-linked-size") or h.headers.get("content-length")
    out["weights_bytes"] = int(size) if size and size.isdigit() else None
    if h.status_code not in (200, 302, 307):
        out["problems"].append(f"weights HEAD HTTP {h.status_code}")
    out["ok"] = not out["problems"]
    return out


def check_all() -> list[dict[str, Any]]:
    pins = load_pins()
    t, d = pins["model"]["target"], pins["model"]["draft"]
    res = [check_repo(t["repo"], t["revision"], ("config.json", "model.safetensors", "tokenizer.json", "chat_template.jinja")),
           check_repo(d["repo"], d["revision"], ("config.json", "model.safetensors"))]
    if res[0].get("weights_bytes") and res[0]["weights_bytes"] != t["weights_bytes"]:
        res[0]["problems"].append(f"weights size {res[0]['weights_bytes']} != pinned {t['weights_bytes']}")
        res[0]["ok"] = False
    for r, pin in ((res[0], t), (res[1], d)):
        if r.get("architectures") and pin["architecture"] not in r["architectures"]:
            r["problems"].append(f"architecture {r['architectures']} != pinned {pin['architecture']}")
            r["ok"] = False
    return res
