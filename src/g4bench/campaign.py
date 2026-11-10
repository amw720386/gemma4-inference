"""Budget-gated, stop-on-first-failure execution of a planned sequence of bridge jobs.

Safety properties (tested in tests/test_campaign.py):
* default is PLAN ONLY; spending requires ``execute=True`` (CLI ``--execute``);
* before every paid step: ``/health`` must be ok with the ledger reachable and must report exactly the approved
  limits (``require``), ``/budget`` must report the same cap, no active jobs, and ``available_usd`` >= the step's
  reservation ceiling (per-job cap, or ``max_usd`` if smaller). Otherwise the campaign stops before spending;
* steps run strictly one at a time; each is polled to a terminal state (a cancel is issued if ``max_wait_s`` is
  exceeded, and polling continues until termination is confirmed so the ledger settles);
* each step's result must meet its ``expect`` block, or the campaign stops;
* every step (including refusals) is appended to a JSONL ledger with the bridge's budget snapshot. That file is the
  agent's copy; the bridge's ``/budget`` is authoritative for the bridge, and neither is Modal's invoice.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .config import load_yaml
from .paths import CONFIGS, RESULTS

Call = Callable[..., Any]   # call(method, path, json=None) -> parsed JSON; raises on HTTP error
LEDGER = RESULTS / "budget_ledger.jsonl"
POLL_GRACE_S = 1800.0


class CampaignStop(RuntimeError):
    pass


@dataclass
class StepOutcome:
    id: str
    status: str                       # planned | refused | done | failed
    detail: str = ""
    job_id: str | None = None
    reserved_usd: float | None = None
    settled_usd: float | None = None
    result: Any = None
    budget_after: dict[str, Any] = field(default_factory=dict)


def load(name_or_path: str) -> dict[str, Any]:
    p = Path(name_or_path)
    if not p.exists():
        p = CONFIGS / "campaigns" / f"{name_or_path}.yaml"
    c = load_yaml(p)
    for k in ("name", "require", "steps"):
        if k not in c:
            raise ValueError(f"campaign missing {k!r}")
    return c


def approved_limits(require: dict[str, Any], env: dict[str, str] | None = None) -> dict[str, Any]:
    """Campaign ``require`` tightened by the approval recorded in the agent's environment, if present.

    ``G4_PROJECT_BUDGET_USD`` / ``G4_JOB_CAP_USD`` (set by the user in the cloud environment) win when stricter; a
    looser environment value never relaxes the campaign file.
    """
    import os

    env = dict(os.environ) if env is None else env
    out = dict(require)
    if env.get("G4_PROJECT_BUDGET_USD"):
        out["cap_usd"] = min(float(out["cap_usd"]), float(env["G4_PROJECT_BUDGET_USD"]))
    if env.get("G4_JOB_CAP_USD"):
        out["per_job_cap_usd_max"] = min(float(out["per_job_cap_usd_max"]), float(env["G4_JOB_CAP_USD"]))
    return out


def check_limits(call: Call, require: dict[str, Any], need_usd: float) -> dict[str, Any]:
    h = call("GET", "/health")
    if not (h.get("ok") and h.get("ledger_ok")):
        raise CampaignStop(f"/health not ok: {h}")
    b = call("GET", "/budget")
    if float(h.get("cap_usd", -1)) != float(require["cap_usd"]) or float(b.get("cap_usd", -1)) != float(require["cap_usd"]):
        raise CampaignStop(f"server cap {h.get('cap_usd')}/{b.get('cap_usd')} != approved {require['cap_usd']}")
    per_job = float(b.get("per_job_cap_usd", h.get("per_job_cap_usd", 1e9)))
    if per_job > float(require["per_job_cap_usd_max"]) + 1e-9:
        raise CampaignStop(f"server per-job cap ${per_job} exceeds approved ${require['per_job_cap_usd_max']}")
    if int(h.get("max_active", -1)) != int(require["max_active"]):
        raise CampaignStop(f"server max_active {h.get('max_active')} != {require['max_active']}")
    if "NOT a Modal billing hard stop" not in str(b.get("disclaimer", "")):
        raise CampaignStop("budget endpoint lacks the expected ledger disclaimer (wrong server version?)")
    if b.get("active"):
        raise CampaignStop(f"{len(b['active'])} job(s) still active")
    if float(b["available_usd"]) < need_usd - 1e-9:
        raise CampaignStop(f"available ${b['available_usd']:.2f} < ${need_usd:.2f} needed for the next step")
    return b


def _expect(step: dict[str, Any], state: str, res: Any) -> str | None:
    if state != "finished":
        return f"terminal state {state}"
    if not isinstance(res, dict):
        return "no result payload"
    if res.get("error") or res.get("aborted"):
        return f"worker reported {res.get('error') or res.get('aborted')}"
    e = step.get("expect") or {}
    if "rc" in e and res.get("rc") != e["rc"]:
        return f"rc {res.get('rc')} != {e['rc']}: {str(res.get('tail', ''))[-500:]}"
    if "speedbench_coding_digest" in e and (res.get("speedbench") or {}).get("coding") != e["speedbench_coding_digest"]:
        return f"SPEED-Bench coding digest {(res.get('speedbench') or {}).get('coding')} != expected"
    if e.get("all_points_valid") and not (res.get("n_points") and res.get("n_valid") == res.get("n_points")):
        return f"{res.get('n_valid')}/{res.get('n_points')} points valid"
    return None


def run(campaign: dict[str, Any], call: Call, *, execute: bool = False, poll_s: float = 120.0,
        max_wait_s: float = 3600.0, sleep: Callable[[float], None] = time.sleep,
        ledger: Path | None = LEDGER, now: Callable[[], float] = time.time) -> list[StepOutcome]:
    out: list[StepOutcome] = []
    req = approved_limits(campaign["require"])
    for step in campaign["steps"]:
        spec = step.get("spec") or {}
        need = min(float(spec.get("max_usd", req["per_job_cap_usd_max"])), float(req["per_job_cap_usd_max"]))
        if step["kind"] != "job":
            need = min(need, 0.25)  # CPU steps reserve well under $0.25 (BUDGET.md)
        if not execute:
            out.append(StepOutcome(step["id"], "planned", f"would reserve <= ${need:.2f}"))
            continue
        try:
            check_limits(call, req, need)
        except CampaignStop as e:
            o = StepOutcome(step["id"], "refused", str(e))
            _log(ledger, campaign, o, now)
            out.append(o)
            break
        path = {"selftest": "/selftest", "prefetch": "/prefetch", "job": "/jobs"}[step["kind"]]
        sub = call("POST", path, json=spec if step["kind"] == "job" else None)
        jid = sub["job_id"]
        t0, cancelled = now(), False
        # Hard stop for polling: the server settles at worst case anyway; never poll (and keep the bridge warm) forever.
        poll_until = float(sub.get("deadline_unix") or (t0 + max_wait_s)) + POLL_GRACE_S
        errors = 0
        while True:
            try:
                st = call("GET", f"/jobs/{jid}")
                errors = 0
            except Exception as e:  # noqa: BLE001 - transient network/HTTP errors while polling
                errors += 1
                if errors > 5:
                    o = StepOutcome(step["id"], "failed", f"lost contact while polling ({e}); job may still be "
                                    "running - its reservation stays held server-side; re-poll with client.py status",
                                    jid, sub.get("reserved_usd"))
                    _log(ledger, campaign, o, now)
                    out.append(o)
                    return out
                sleep(poll_s)
                continue
            if st["state"] not in ("running", "unknown"):
                break
            if now() > poll_until:
                o = StepOutcome(step["id"], "failed", f"still {st['state']} {POLL_GRACE_S:.0f}s after its deadline; "
                                "stopped polling - the reservation stays held and settles at worst case server-side",
                                jid, sub.get("reserved_usd"))
                _log(ledger, campaign, o, now)
                out.append(o)
                return out
            if not cancelled and now() - t0 > max_wait_s:
                call("POST", f"/jobs/{jid}/cancel")
                cancelled = True
            sleep(poll_s)
        problem = _expect(step, st["state"], st.get("result"))
        b = call("GET", "/budget")
        o = StepOutcome(step["id"], "failed" if problem else "done", problem or "ok", jid, sub.get("reserved_usd"),
                        (st.get("job") or {}).get("settled_usd"), _slim(st.get("result")),
                        {k: b.get(k) for k in ("cap_usd", "committed_usd", "settled_estimate_usd", "available_usd")})
        _log(ledger, campaign, o, now)
        out.append(o)
        if problem:
            break
    return out


def _slim(res: Any) -> Any:
    if isinstance(res, dict) and "rows" in res:
        return {k: v for k, v in res.items() if k != "rows"} | {"n_rows": len(res["rows"])}
    return res


def _log(ledger: Path | None, campaign: dict[str, Any], o: StepOutcome, now: Callable[[], float]) -> None:
    if ledger is None:
        return
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with open(ledger, "a", encoding="utf-8") as f:
        f.write(json.dumps({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now())), "campaign": campaign["name"],
                            **o.__dict__}, default=str) + "\n")
