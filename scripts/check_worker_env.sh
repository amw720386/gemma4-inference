#!/usr/bin/env bash
# Run the Modal worker body on the mock engine under the engine images' conditions:
# Python 3.12 (both pinned images, from their registry configs) and ONLY packages the engine images ship.
# Catches stray imports (fastapi/pyarrow/httpx/modal/pytest) and Python-version-specific bugs before paid runs.
# Usage: scripts/check_worker_env.sh <tokenizer_dir>
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
tok="${1:?tokenizer dir}"
work="$(mktemp -d)"
uv venv -q -p 3.12 "$work/venv"
VIRTUAL_ENV="$work/venv" uv pip install -q numpy pyyaml requests aiohttp "transformers>=5.10" huggingface_hub jinja2 tqdm
cat > "$work/check.py" <<'PY'
import json, pathlib, sys, time
sys.path.insert(0, sys.argv[1] + "/src")
import g4bench.jobs as bj
spec = {"engine": "mock-cpu", "matrix": "mock-ci", "seeds": [0], "max_wall_s": 300, "job_id": "envcheck",
        "deadline_unix": time.time() + 900, "start_deadline_unix": time.time() + 60}
out = bj.guarded(lambda s: bj.run_job(s, pathlib.Path(sys.argv[2]), tokenizer=sys.argv[3]), spec)
bad = sorted({m.split(".")[0] for m in sys.modules} & {"fastapi", "pyarrow", "httpx", "modal", "pydantic", "pytest"})
ok = not out.get("error") and not out.get("aborted") and out.get("n_valid") == out.get("n_points") == 2 and not bad
print(json.dumps({"python": sys.version.split()[0], "error": out.get("error"), "n_valid": out.get("n_valid"),
                  "n_points": out.get("n_points"), "forbidden_imports": bad, "ok": ok}))
sys.exit(0 if ok else 1)
PY
(cd "$work" && PYTHONPATH="$repo/src" "$work/venv/bin/python" -I check.py "$repo" "$work/runs" "$tok")
