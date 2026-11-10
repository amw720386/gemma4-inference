# Next actions (executable checkpoint 3)

## A. User (in progress, per user 2026-11-10) — the agent waits; it does not deploy or request credentials

1. Deploy the bridge once: `modal_bridge/BOOTSTRAP.md` §1 (defaults already = approved $10 / $2 / 1 active).
2. Provision a separate scoped proxy token as a network secret for host `arnold-17059--g4repro-bridge.us-east.modal.run`
   (BOOTSTRAP §2–3).

## B. Agent, after A — strictly in this order, stop at the first failure or when `/budget` cannot cover the next step

Mechanised (D-026): `python modal_bridge/campaign.py e1-smoke` (plan) → `… e1-smoke --execute` (steps 1–4) →
review + commit results → `… e2-pairs --execute` (step 5). The table below is what the campaigns do.

Before every paid step: run `budget`, write the planned reservation in BUDGET.md, then launch; after it settles,
record the settled estimate. Never run two jobs at once (the bridge refuses anyway).

| # | Command (`python modal_bridge/client.py …`) | Reserved | Expected settle | Gate to continue |
|---|---|---:|---:|---|
| 0 | `g4bench flagcheck` + `scripts/check_worker_env.sh <tok>` (local, free), then `client.py verify` (free: /health + /budget must show $10 / ≤ $2 / 1 active) | 0 | 0 | `ok`, `ledger_ok`, cap 10, per-job ≤ 2, available 9.50, no active jobs |
| 1 | `selftest` | 0.10 | ≤ 0.10 | tests pass inside Modal |
| 2 | `prefetch` | 0.16 | ≤ 0.16 | weights cached; SPEED-Bench coding digest = InferenceX |
| 3 | `submit --engine vllm-0.33.0-mtp4 --matrix smoke --max-usd 2`, then `wait <job_id>` | 2.00 | ~0.7 | 0 failures, probes ≥ 7/8, acceptance length captured, `exact_hardware=true`, effective attention backend recorded (REPORT.md table); `fetch` + commit `results/runs/<run_id>/` |
| 4 | same with `--engine sglang-0.5.23-nextn` | 2.00 | ~0.7 | same; record effective spec algorithm; check `driver_version` supports CUDA 13 (images are CUDA 13.0.x) |
| 5 | single-point article jobs, seed 0: `--matrix backboard-repro --seeds 0 --only <key> --max-usd 2`, alternating engines at the same key — start with `1024x1`, then `8192x4` (largest vLLM gap in the article) | 2.00 each | 0.76–0.86 each | each point valid; stop when `available_usd` < 2.0 |

With ~$7.5 left after steps 1–4, step 5 affords ≈ 8 single-point jobs (≈ 4 matched vLLM/SGLang pairs). These are
n = 1 results: **descriptive only**, never claims (EXPERIMENTS criterion 2). More seeds or the full matrix need a
new, explicit budget approval.

## C. Agent, CPU-side work available now

1. ~~vLLM-vs-SGLang matched-load comparison~~ ✅ done (cycle 1: `engine_compare.py`, D-022).
2. ~~Static flag preflight~~ ✅ done (cycle 2, D-023).
3. ~~EOS-honouring real-prompt variant~~ ✅ done (cycle 8, D-029).
4. Torch-profiler trace parsing (for E5g) against a recorded fixture.
5. Optional: licence choice for this repository (owner decision; none added).
