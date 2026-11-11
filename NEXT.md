# Next actions (executable checkpoint 6)

## A. Paid steps (direct Modal; D-039, D-040)

Done 2026-11-11: selftest, prefetch, vLLM smoke, and all 16 seed-0 article points (6 vLLM + 6 SGLang jobs on
`backboard-repro-shared`; 8192×64 with the $3 per-job cap, D-041). Remaining within the $10 approval: ≈ $3.4 by the
conservative ledger before E5b; ≈ $2.2 after it.

1. ~~E5b depth sweep at 8192×4~~ ✅ (D-042). Remaining ≈ $2.2 (ledger): enough for ≈ 2 more single-point repeats,
   e.g. depth 4 and depth 6 at 8192×4 again to tell the +9 % from noise — or stop and delete the cache Volume.
2. Seeds 1, 2 for n = 3 claims (EXPERIMENTS criterion 2) need ≈ $10–15: a new budget approval.

Every command: `g4bench modal job … ` (plan only) → `… --execute`; one launch at a time; `g4bench modal spend --reconcile`
between launches. Delete the `g4repro-hf-cache` Volume when GPU work pauses for more than a few days.

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
