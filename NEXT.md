# Next actions (executable checkpoint 5)

## A. Paid steps (direct Modal from the owner's machine; D-039) — strictly in this order, stop at the first failure

Each command without `--execute` prints the reservation it would make and the budget after it. With `--execute` it
reconciles with Modal billing, refuses if the worst case does not fit, launches one ephemeral app, settles, and writes
`results/runs/<run_id>/` locally.

| # | Command (`g4bench modal …`) | Worst case | Expected settle | Gate to continue |
|---|---|---:|---:|---|
| 0 | `spend` (free), then `selftest --execute` (✅ 2026-11-11: 98 passed, 1 skipped on Linux; $0.012 est.) | 0.08 | ≈ 0.01 | suite passes in Modal |
| 1 | `prefetch --execute` | 0.13 | ≤ 0.05 | target + draft snapshots on `g4repro-hf-cache` |
| 2 | `job --engine vllm-0.33.0-mtp4 --matrix smoke --execute` | 2.00 | ~0.5–0.7 | 0 failures, probes ≥ 7/8, acceptance length captured, `exact_hardware=true`, effective attention backend recorded |
| 3 | same with `--engine sglang-0.5.23-nextn` | 2.00 | ~0.5–0.7 | same; record effective spec algorithm |
| 4 | single-point article jobs, seed 0: `job --matrix backboard-repro --seeds 0 --only <key> --execute`, alternating engines — start with `1024x1`, then `8192x4` | 2.00 each | 0.6–0.9 each | each point valid; stop when the remaining budget < $2 |

These are n = 1 results: **descriptive only**, never claims (EXPERIMENTS criterion 2). More seeds or the full matrix
need a new, explicit budget approval. Delete the `g4repro-hf-cache` Volume when GPU work pauses for more than a few days
(≈ $0.09/GiB-month; tracked by the `storage` ledger lines).

## C. Agent, CPU-side work available now (each as its own `feat/…` branch + PR off `main`)

1. ~~vLLM-vs-SGLang matched-load comparison~~ ✅ done (cycle 1: `engine_compare.py`, D-022).
2. ~~Static flag preflight~~ ✅ done (D-023). ~~Free /estimate preflight~~ (#5), ~~one-command setup~~ (#6),
   ~~claims register~~ (#7) ✅ done.
3. ~~EOS-honouring real-prompt variant~~ ✅ done (cycle 8, D-029).
4. Torch-profiler trace parsing (for E5g) against a recorded fixture.
5. Optional: licence choice for this repository (owner decision; none added).

## D. After each measured job

`g4bench analyze && g4bench claims` → commit `results/runs/<run_id>/`, `results/summary/`, `docs/CLAIMS.md` on a
`feat/results-…` branch with the run's REPORT.md quoted in the PR (real numbers only, n and CIs stated).

## E. Repository housekeeping (needs the GitHub UI; ref deletion pushes are 403 from this environment)

Delete merged branches `feat/smoke-readiness`, `feat/one-command-setup`, `feat/claims-register`.
