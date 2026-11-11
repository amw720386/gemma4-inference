# Status — 2026-11-10 (checkpoint 5)

**One-line state:** CPU side complete on `main`. The Modal HTTPS bridge is **removed** (D-039): paid runs are launched
directly from the owner's machine with `g4bench modal …`, gated locally by a $10 project / $2 per-job spend check that
is reconciled against Modal's per-app billing. **Zero GPU measurements exist yet**; E1 smoke is next.

Workflow (CLAUDE.md): Git author Ahamed Mohamed Wajibu + `Co-Authored-By: Claude Opus 5.5`; one semantic branch per
change (`feat/…`, `fix/…`, `docs/…`) off `main`; focused PR with executed test output and NOT-RUN limits; merge
commit after verification. No GitHub Actions; `agent/*` branches are retired.

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
| Spend gate ($10 project, $2/job, 1 open launch, worst case reserved before launch, reconciled with `modal billing report`) | ✅ CPU-tested | src/g4bench/spend.py, tests/test_spend.py |
| Direct Modal launches (ephemeral app, digest-pinned images, no deployed service) | ✅ definition-tested | src/g4bench/modal_app.py, src/g4bench/modal_run.py, tests/test_modal_app.py |
| vLLM vs SGLang matched-load comparison (paired by seed, correctness-gated, CIs) | ✅ CPU-tested | src/g4bench/engine_compare.py, results/summary/engine_comparison.csv |
| Static engine-flag preflight vs pinned vLLM/SGLang source (runner refuses unknown flags) | ✅ all 4 configs clean | src/g4bench/flagcheck.py, configs/engine_flags/ |
| CPU rehearsal of the paid worker body (worker → runner(mock) → returned run files) | ✅ | src/g4bench/jobs.py, tests/test_jobs.py |
| Effective engine config from server logs (attention backend, KV tokens, CUDA graphs, SGLang effective spec algorithm) | ✅ schema v3 | src/g4bench/serverlog.py |
| Worker verified on Python 3.12 with engine-image-only deps (`scripts/check_worker_env.sh`); fixed a bug that would have crashed every GPU job | ✅ | DECISIONS D-031 |
| Gap attribution (acceptance length vs step speed vs roofline) in per-run reports | ✅ | src/g4bench/gap.py |
| E5/E6 optimization arms (depth sweep, depth-by-batch-size, draft LM-head FP8, SGLang adaptive) as flag-checked configs; role/admissibility gating in analysis | ✅ prepared, not run | configs/engines/, D-034 |
| Free `/estimate` preflight in campaigns; one-command setup + dependency lock; claims register (`docs/CLAIMS.md`) | ✅ | PRs #5, #6, #7 |
| GPU experiments E1–E6 | ⬜ unblocked (direct Modal) | — |

## Measured results

**None on GPU.** Every executed run used the CPU mock engine (`engine=mock`), which `g4bench analyze`
excludes and every report banners as "MOCK ENGINE — NOT A MEASUREMENT". `results/runs/` is empty.

## Blockers

None for E1. Budget: **$10 total / ≤ $2 per job** (BUDGET.md); `g4bench modal spend` shows committed spend.
Note: the `.env` token secret was missing its `as-` prefix's leading `a` and was corrected locally on 2026-11-10.

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

Python 3.13.18 (cloud container) and 3.12 (owner's Windows machine); numpy 2.5.4, aiohttp 3.14.5, transformers 5.22.2, tokenizers 0.23.3, huggingface_hub 2.4.0,
jinja2 3.1.6, pyarrow 26.0.1, requests 2.34.2, fastapi 0.162.4, httpx 0.28.1, pytest 9.2.0, modal 1.6.3.

## Verify this checkpoint

One command: `scripts/setup.sh` (add `--data` for SPEED-Bench). Manual equivalent:

```bash
python3 -m venv .venv && . .venv/bin/activate && pip install -e ".[dev]"
```

```bash
g4bench speedbench prepare && python -m pytest -q
```

Expected: SPEED-Bench coding digest `6f1845452f1cfbc1930ea779ba6c3d5b437a9d5853eca199dd9b7619b20a0c18`,
`matches_inferencex_digest: true`; all tests pass (without `speedbench prepare`, the one network-marked digest test
is skipped; without `modal` installed, the Modal-definition test is skipped).
