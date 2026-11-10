"""Run a budget-gated campaign against the deployed bridge over HTTPS.

  python modal_bridge/campaign.py e1-smoke            # plan only (default; spends nothing)
  python modal_bridge/campaign.py e1-smoke --execute  # spend, one step at a time, stop at first failed gate

See src/g4bench/campaign.py for the gates. Never run with --execute unless the user has approved the budget and
GET /health and /budget show the server-side $10 cumulative and <= $2/job limits (the campaign re-checks both).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from client import _call  # noqa: E402

from g4bench import campaign as C  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("campaign")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--poll", type=float, default=120.0)
    ap.add_argument("--max-wait-s", type=float, default=3600.0)
    ap.add_argument("--ledger", type=Path, default=C.LEDGER,
                    help="JSONL ledger to append to (default results/budget_ledger.jsonl; use another path for rehearsals)")
    ap.add_argument("--fetch-dir", type=Path, default=Path(__file__).resolve().parents[1] / "results" / "runs",
                    help="where finished runs are downloaded (all files via /runfiles)")
    a = ap.parse_args(argv)
    def call(method, path, json=None):
        try:
            return _call(method, path, json=json) if json is not None else _call(method, path)
        except SystemExit as e:  # client.py reports HTTP errors as SystemExit
            raise C.CampaignStop(str(e)) from None

    try:
        outs = C.run(C.load(a.campaign), call, execute=a.execute, poll_s=a.poll, max_wait_s=a.max_wait_s,
                     ledger=a.ledger, fetch_dir=a.fetch_dir)
    except C.CampaignStop as e:
        print(json.dumps({"stopped": str(e)}))
        return 1
    print(json.dumps([o.__dict__ for o in outs], indent=2, default=str))
    return 0 if all(o.status in ("planned", "done") for o in outs) else 1


if __name__ == "__main__":
    sys.exit(main())
