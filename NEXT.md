# Next actions (executable checkpoint 6)

## A. Paid steps (direct Modal; D-039, D-040)

Done 2026-11-11: selftest, prefetch, vLLM smoke, and 14/16 seed-0 article points (5 vLLM + 5 SGLang jobs on
`backboard-repro-shared`). Remaining within the $10 approval (≈ $5.7 by the conservative ledger, more by Modal's bill):

1. **8192×64, both engines** — needs one job of ≈ 14 min (vLLM) / ≈ 20 min (SGLang) of runtime. A $2 reservation buys
   876 s, so this needs the owner to raise the per-job cap (≈ $2.6 / $3.0 worst case; expected settle ≈ $1.0 / $1.4),
   or a reduced-request variant labelled as such.
2. Then, if budget remains: E5b depth sweep at the largest vLLM-vs-Backboard gap (8192×4), e.g.
   `job --engine vllm-0.33.0-mtp5 --matrix backboard-repro-shared --only 8192x4 --execute` (≈ $0.5 each).

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
