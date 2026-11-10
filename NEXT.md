# Next actions (executable checkpoint)

## A. Human-only unblockers (GPU work cannot start without all three)

1. **Deploy the bridge once** from a machine with the Modal CLI — `modal_bridge/BOOTSTRAP.md` §1
   (workspace `arnold-17059`). Expected endpoint: `https://arnold-17059--g4repro-bridge.modal.run`.
2. **Provision access for this environment**: add a Modal proxy-auth token as a network secret on this
   cloud environment (settings → network secrets; host `arnold-17059--g4repro-bridge.modal.run`, header
   `Authorization`, value `Bearer <TokenID>.<TokenSecret>`). The agent will not ask for it in chat.
   *Alternative:* run jobs yourself (`modal_bridge/client.py` or `g4bench run` on any 1× RTX PRO 6000 host)
   and commit `results/runs/<run_id>/`; the agent analyzes committed results.
3. **Approve a budget** in writing (BUDGET.md caps are a proposal only, D-013). Minimum useful approval:
   E1 smoke ≈ $1.05 (+ ≈ $0.07 CPU selftest/prefetch).

## B. Agent, once A1–A3 are done (in this order; log each estimate + actual in BUDGET.md first)

1. `python modal_bridge/client.py health` — must return `ok: true` and list the speedbench matrices.
2. `client.py selftest` (CPU) → `client.py prefetch` (CPU; downloads weights **and** prepares + digest-verifies
   SPEED-Bench into the volume; fails loudly if the coding digest ≠ InferenceX).
3. **E1 smoke** (`--matrix smoke`, `--max-usd 2`), vLLM then SGLang. Pass = 0 failures, probes ≥ 7/8,
   acceptance length captured, `exact_hardware=true`, GPU edition/power limit recorded. Read the server
   log for effective attention backend, CUDA-graph sizes, max-model-len default, SGLang's effective spec
   algorithm (NEXTN vs FROZEN_KV_MTP). `client.py fetch` the run; commit `results/runs/<run_id>/` incl. REPORT.md.
4. **E2/E3** seed-0 full matrix per engine ($4.70 / $6.04 est.). Gate: any point > 25 % off the article ⇒
   stop and diagnose before E4.
5. **E4** seeds 1,2 ($21.07 est.) ⇒ first claims (n = 3, t-CIs) via `g4bench analyze`; update README table.
6. **E5c** real prompts at matched engine configs: `speedbench-coding` then `speedbench-tp`, seed 0 first
   ($4.47 / $10.74 est. for both engines). Compare acceptance length and throughput vs the random arm at the
   same conc — this quantifies how much the article's random-token workload flatters or penalises MTP.
7. Remaining E5 ablations in EXPERIMENTS.md order, each with a fresh estimate.

## C. Agent, CPU-side work available now (no blocker)

1. EOS-honouring real-prompt variant (`ignore_eos: false`, D-017) — matrix + validation already supported;
   add a CPU test and a cost estimate.
2. Cross-run comparison report (vLLM vs SGLang at matched load) in `g4bench analyze`: per-workload ratio with
   bootstrap CI from `stats.bootstrap_ratio`, gated on both engines passing correctness.
3. Torch-profiler trace parsing (for E5g) against a recorded fixture.
4. Optional: licence choice for this repository (owner decision; none added).
