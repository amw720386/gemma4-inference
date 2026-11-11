# Next actions (executable checkpoint 6)

## A. Paid steps (direct Modal; D-039, D-040)

Done 2026-11-11: selftest, prefetch, vLLM smoke, and all 16 seed-0 article points (6 vLLM + 6 SGLang jobs on
`backboard-repro-shared`; 8192×64 with the $3 per-job cap, D-041), then E5b, E5e and E6a/E6b runs (EXPERIMENTS run log).

1. ~~E5b depth sweep at 8192×4 + repeats~~ ✅ (D-042): depth 6 ≈ +8 %. ~~E5e chunked prefill + split-K~~ ✅ (D-043,
   D-044). ~~E6a depth by batch size, image patches~~ ✅ (D-046, D-047).
   Project cap is now $20 / $3 per job (D-045); check `g4bench modal spend --reconcile` before each launch.
2. E6b (`vllm-0.33.0-best-e6b`) on the remaining 6 workloads (1024×1/4/16, 8192×1/4/16) — running; fill the D-047 table.
3. E6b profile at 8192×4 (`--profile-key 8192x4`, D-049).
4. Then decode/prefill levers chosen from that profile (D-047 model: ≈ 10–15 % faster prefill is the biggest lever).
5. Lower priority: `sglang-0.5.23-nextn-fair64` at 1024×64 / 8192×64 for a like-for-like c64 comparison (D-048).
6. Seeds 1, 2 for n = 3 claims (EXPERIMENTS criterion 2) need ≈ $10–15: a new budget approval.

Every command: `g4bench modal job … ` (plan only) → `… --execute`; one launch at a time; `g4bench modal spend --reconcile`
between launches. Delete the `g4repro-hf-cache` Volume when GPU work pauses for more than a few days.

## C. Agent, CPU-side work available now (each as its own `feat/…` branch + PR off `main`)

1. ~~vLLM-vs-SGLang matched-load comparison~~ ✅ done (cycle 1: `engine_compare.py`, D-022).
2. ~~Static flag preflight~~ ✅ done (D-023). ~~Free /estimate preflight~~ (#5), ~~one-command setup~~ (#6),
   ~~claims register~~ (#7) ✅ done.
3. ~~EOS-honouring real-prompt variant~~ ✅ done (cycle 8, D-029).
4. ~~Torch-profiler trace parsing (for E5g)~~ ✅ done (D-049); validate it on the first real trace.
5. Optional: licence choice for this repository (owner decision; none added).

## D. After each measured job

`g4bench analyze && g4bench claims` → commit `results/runs/<run_id>/`, `results/summary/`, `docs/CLAIMS.md` on a
`feat/results-…` branch with the run's REPORT.md quoted in the PR (real numbers only, n and CIs stated).

## E. Repository housekeeping (needs the GitHub UI; ref deletion pushes are 403 from this environment)

Delete merged branches `feat/smoke-readiness`, `feat/one-command-setup`, `feat/claims-register`.
