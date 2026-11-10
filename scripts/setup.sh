#!/usr/bin/env bash
# One-command setup + verification for the Gemma 4 reproduction harness (CPU only; spends nothing).
#   scripts/setup.sh            # venv, install, model availability, flag preflight, tests
#   scripts/setup.sh --data     # also prepare SPEED-Bench prompt sets (~4.6 GB streamed, digest-gated)
#   scripts/setup.sh --no-tests # skip the test suite
# Requires Python >= 3.11 and network access to huggingface.co / raw.githubusercontent.com.
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo"
data=0; tests=1
for a in "$@"; do
  case "$a" in
    --data) data=1 ;;
    --no-tests) tests=0 ;;
    *) echo "unknown option $a" >&2; exit 2 ;;
  esac
done
py="${PYTHON:-python3}"
"$py" -c 'import sys; assert sys.version_info >= (3, 11), sys.version' || { echo "need Python >= 3.11" >&2; exit 1; }
if [ ! -x .venv/bin/python ]; then
  if command -v uv >/dev/null 2>&1; then uv venv -q -p "$py" .venv; else "$py" -m venv .venv; fi
fi
# Exact versions the test-suite was validated with (requirements-lock.txt), then the package itself without deps.
if command -v uv >/dev/null 2>&1; then
  VIRTUAL_ENV="$repo/.venv" uv pip install -q -r requirements-lock.txt
  VIRTUAL_ENV="$repo/.venv" uv pip install -q --no-deps -e .
else
  .venv/bin/python -m pip install -q -r requirements-lock.txt
  .venv/bin/python -m pip install -q --no-deps -e .
fi
step() { printf '\n== %s\n' "$1"; }
step "model availability (pinned HF revisions, ungated)"
.venv/bin/g4bench sources | .venv/bin/python -c 'import json,sys; r=json.load(sys.stdin); [print(x["repo"], "ok" if x["ok"] else x["problems"]) for x in r]; sys.exit(0 if all(x["ok"] for x in r) else 1)'
step "engine flag preflight (pinned vLLM / SGLang sources)"
.venv/bin/g4bench flagcheck >/dev/null && echo "all engine configs: no unknown flags"
if [ "$data" = 1 ]; then
  step "SPEED-Bench prompt sets (digest-gated)"
  .venv/bin/g4bench speedbench prepare
fi
if [ "$tests" = 1 ]; then
  step "tests (CPU only)"
  .venv/bin/python -m pytest -q
fi
step "done"
echo "Next: python modal_bridge/client.py verify   (free; needs the deployed bridge)"
