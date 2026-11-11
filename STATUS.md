# Status — 2026-11-11 (checkpoint 6)

**One-line state:** all 16 of the article's seed-0 points measured on 1× RTX PRO 6000 Blackwell Server Edition (Modal),
all valid, correctness 8/8 on every server. **vLLM 0.33.0 reproduces the article at 8/8 points (+0.9 … +3.1 %); our
SGLang 0.5.23 is 7–17 % faster than the article's at 8/8.** n = 1 each: descriptive, not claims. Spend ≈ $5.1 billed by
Modal (≈ $6.6 on the conservative ledger) of the $10 cap; per-job cap raised to $3 by the owner for 8192×64 (D-041).

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
| E1 smoke + E2/E3 seed-0 baselines (16/16 points) | ✅ n=1 | results/runs/, results/summary/SUMMARY.md |
| E4 repeats, E5/E6 | ⬜ beyond the $10 approval | — |

## Measured results (output tok/s, seed 0, OSL 1024, n = 1, `results/summary/SUMMARY.md`)

| Workload (ISL×conc) | vLLM ours | article vLLM | ratio | SGLang ours | article SGLang | ratio | Backboard |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1024×1 | 190.4 | 186.8 | 1.019 | 136.9 | 122.6 | 1.117 | 203.6 |
| 1024×4 | 655.0 | 649.0 | 1.009 | 499.1 | 456.8 | 1.093 | 779.4 |
| 1024×16 | 1905 | 1861 | 1.024 | 1299 | 1122 | 1.157 | 2082 |
| 1024×64 | 3668 | 3634 | 1.009 | 2039 | 1750 | 1.165 | 3822 |
| 8192×1 | 164.6 | 159.6 | 1.031 | 89.4 | 82.5 | 1.084 | 179.1 |
| 8192×4 | 402.6 | 395.7 | 1.018 | 287.9 | 269.5 | 1.068 | 529.6 |
| 8192×16 | 801.9 | 794.5 | 1.009 | 587.5 | 533.7 | 1.101 | 961.8 |
| 8192×64 | 1095 | 1079 | 1.015 | 728.6 | 673.6 | 1.082 | 1254 |

Readouts (descriptive): speculative acceptance length vLLM 4.06–4.21, SGLang 4.47–4.86 (effective algorithm
`FROZEN_KV_MTP`); attention backends vLLM `TRITON_ATTN` (no FA4 on sm_120), SGLang `triton`. vLLM/SGLang throughput
ratio 1.31–1.84× vs the article's implied 1.42–2.08× (C04). Backboard stays ahead of both open engines: vLLM reaches
76–96 % of it (largest gap 8192×4), SGLang 50–67 %. Throughput at 8192×64 uses one 13–18 min job (D-041). Points of one job share a server start (D-040).

## Blockers

No technical blockers. Repeats (seeds 1, 2) for real claims need ≈ $10–15 more: a new budget approval. Budget: **$10 total / ≤ $3 per job** (BUDGET.md); `g4bench modal spend` shows committed spend.
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
