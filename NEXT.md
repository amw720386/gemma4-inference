# Next actions

## Needs a human (blocking all GPU work)

1. **Deploy the bridge once** — `modal_bridge/BOOTSTRAP.md` §1 (workspace `arnold-17059`).
2. **Create a Modal proxy-auth token** and add it as a network secret on this cloud environment
   (environment settings → network secrets, host `arnold-17059--g4repro-bridge.modal.run`, header
   `Authorization`, value `Bearer <TokenID>.<TokenSecret>`). The in-session request was declined on
   2026-11-10, so the agent won't ask again; add it in settings when ready.
   *Alternative without giving the agent a token:* run the matrix yourself on any 1× RTX PRO 6000 host
   (`g4bench run --engine vllm-0.33.0-mtp4 --matrix smoke`), or call the bridge with
   `modal_bridge/client.py` from your own machine, then commit `results/runs/<run_id>/`. The agent
   analyzes committed results with `g4bench analyze`.
3. Optional: confirm or change the budget caps in BUDGET.md (proposed project cap $100).
4. Optional: choose a licence for this repository (none added; the vendored client stays Apache-2.0).

## Agent, as soon as the bridge answers `/health`

1. `client.py selftest` (CPU) → `client.py prefetch` (CPU) → log both in BUDGET.md.
2. E1: `submit --engine vllm-0.33.0-mtp4 --matrix smoke --max-usd 2`; `wait`; `fetch`; inspect server log for
   effective attention backend, CUDA-graph sizes, acceptance length, max-model-len default; same for SGLang.
3. E2/E3: seed-0 full matrix per engine (estimates $4.70 / $6.04), gate at ±25 % before E4.
4. E4 repeats → first claims with CIs; update README table with measured columns.
5. E5 ablations in the order listed in EXPERIMENTS.md, each with a fresh estimate.

## Agent, CPU-side work that can continue meanwhile

- Add a SPEED-Bench (real-prompt) dataset arm to quantify acceptance-length sensitivity (E5c), CPU-validated
  against the mock.
- Add a per-request bootstrap report (from `--save-detailed` per-request TTFT/ITL) to `g4bench analyze`.
- Add a per-run HTML/markdown report with throughput-by-concurrency and latency curves.
