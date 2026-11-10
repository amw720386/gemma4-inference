# Status — 2026-11-10 (checkpoint 3)

**One-line state:** harness, real-prompt arm, reporting and a **server-side $10 project spend limit** in the bridge
are built and CPU-tested (96 tests); **zero GPU measurements exist**. Budget: **$10 total approved** (per-job ≤ $2).
GPU work waits only on the user's one-time bridge deploy + scoped proxy token. Spend to date: **$0.00**.

## Summary

| Area | State | Evidence |
|---|---|---|
| Source verification | ✅ | docs/SOURCES.md |
| Harness: runner, InferenceX client, stats, compare, telemetry, spec-decode capture | ✅ CPU-tested | tests/ |
| Random-token arm (article workload), request-stream fingerprints | ✅ | configs/fingerprints-rr0.8.json |
| **Real-prompt arm: SPEED-Bench coding (InferenceX golden-AL set)** | ✅ prepared, **digest-identical to InferenceX** (`6f184545…`) | data/speedbench/speedbench-coding.provenance.json |
| Real-prompt arm: SPEED-Bench throughput_1k/8k inline subsets | ✅ prepared, labelled SUBSET | data/speedbench/*-inline.provenance.json, DECISIONS D-016 |
| **Per-request bootstrap CIs** (TTFT/TPOT/E2EL/ITL + Little's-law throughput diagnostic) | ✅ in results schema v2 | src/g4bench/request_stats.py |
| **Per-run report** (REPORT.md + throughput.svg, not-measured/mock/proxy states) | ✅ auto-written by every run | src/g4bench/report.py, docs/examples/ (MOCK) |
| Failure-mode integration tests (dead server, failures, timeout, budget, bad datasets) | ✅ | tests/test_failure_modes.py |
| **Bridge budget ledger** ($10 project cap, $2/job, 1 active, reserve-before-launch, race/restart-safe, `/budget`) | ✅ CPU-tested | src/g4bench/budget.py, src/g4bench/bridge_api.py, tests/test_budget.py, tests/test_bridge_api.py |
| Modal HTTPS bridge | 🟨 written + definition-validated (modal 1.6.3); **not deployed** | modal_bridge/ |
| vLLM vs SGLang matched-load comparison (paired by seed, correctness-gated, CIs) | ✅ CPU-tested | src/g4bench/engine_compare.py, results/summary/engine_comparison.csv |
| Static engine-flag preflight vs pinned vLLM/SGLang source (runner refuses unknown flags) | ✅ all 4 configs clean | src/g4bench/flagcheck.py, configs/engine_flags/ |
| CPU rehearsal of the paid bridge path (HTTP → ledger → worker → runner(mock) → settle) | ✅ | src/g4bench/bridge_jobs.py, tests/test_bridge_rehearsal.py |
| Effective engine config from server logs (attention backend, KV tokens, CUDA graphs, SGLang effective spec algorithm) | ✅ schema v3 | src/g4bench/serverlog.py |
| Budget-gated campaign runner (plan-only default; re-checks /health+/budget limits before every paid step) | ✅ verified refusing against undeployed bridge ($0) | src/g4bench/campaign.py, configs/campaigns/, results/budget_ledger.jsonl |
| GPU experiments E1–E6 | ⛔ blocked (see below) | — |

## Measured results

**None on GPU.** Every executed run used the CPU mock engine (`engine=mock`), which `g4bench analyze`
excludes and every report banners as "MOCK ENGINE — NOT A MEASUREMENT". `results/runs/` is empty.

## Exact blocker

1. **Bridge not deployed** (user is arranging it). `https://arnold-17059--g4repro-bridge.modal.run` answers
   `modal-http: invalid function call` (last checked 2026-11-10, cycle 1; `G4_BRIDGE_URL` not yet visible in this
   container's environment — it applies on the next session restart; the client falls back to the expected URL). This session must not deploy or use the Modal CLI/SDK.
2. **No scoped proxy token** in this environment yet (user is arranging it separately; the agent does not request it).
3. ~~No approved budget~~ → **resolved: $10 total approved** (BUDGET.md, DECISIONS D-019). Enforced by the bridge
   ledger once deployed; it is an estimate gating bridge launches, **not** a Modal billing hard stop.

## Key findings (CPU-side only)

- Article internals: 16/24 published (p99 E2EL, p99 TPOT) pairs are impossible with fixed 1024-token outputs
  ⇒ variable output lengths (consistent with InferenceX range ratio 0.8).
- Roofline: no-spec single-stream ≈ 63 tok/s on RTX PRO 6000 ⇒ the article's ~185–204 tok/s/user implies
  speculative acceptance length ≳ 3.1; throughput is acceptance-dominated, hence prompt-content-sensitive.
- The InferenceX golden-AL prompt set (80 SPEED-Bench coding prompts) can be reproduced byte-identically
  with ~4.6 GB of streamed downloads instead of ~35 GB; Gemma-tokenized lengths 27–1471 (mean 251).
- vLLM v0.33.0 source: Gemma 4's mixed head dims (256/512) make vLLM use FA4 if available, else force
  TRITON_ATTN; which happens on sm_120 is an E1 observable (D-025).
- The article implies vLLM leads SGLang 1.4–2.1× on throughput/TPOT but is at parity on p99 TTFT at 8K
  (D-027, docs/analysis/article_vllm_vs_sglang.md) — a Backboard-independent target for our baselines.
- SPEED-Bench throughput "1k"/"8k" are ~1210 / ~9372 Gemma tokens (sized for a different tokenizer), so
  they are real-text proxies near, not at, the article's ISLs.

## Tooling versions used for CPU checks

Python 3.13.18; numpy 2.5.4, aiohttp 3.14.5, transformers 5.22.2, tokenizers 0.23.3, huggingface_hub 2.4.0,
jinja2 3.1.6, pyarrow 26.0.1, requests 2.34.2, fastapi 0.162.4, httpx 0.28.1, pytest 9.2.0, modal 1.6.3 (definition import only).

## Verify this checkpoint

```bash
python3 -m venv .venv && . .venv/bin/activate && pip install -e ".[dev]"
```

```bash
g4bench speedbench prepare && python -m pytest -q
```

Expected: SPEED-Bench coding digest `6f1845452f1cfbc1930ea779ba6c3d5b437a9d5853eca199dd9b7619b20a0c18`,
`matches_inferencex_digest: true`; 96 tests pass (without `speedbench prepare`, the one network-marked
digest test is skipped; without `modal` installed, the bridge-definition test is skipped).
