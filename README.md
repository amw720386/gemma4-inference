# gemma4-inference-repro

An independent, open-source attempt to **reproduce** the Gemma 4 12B serving benchmark that Backboard published on
2026-11-09 ([article](https://backboard.io/blog/gemma-4-12b-vllm-sglang-inference)), and then to **close the gap** to
Backboard's proprietary engine using only open engines and lossless changes.

Setup throughout: 1× NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB, 600 W) on Modal, `google/gemma-4-12B-it`
BF16 with the `google/gemma-4-12B-it-assistant` MTP draft, random-token prompts (InferenceX client, range ratio 0.8,
chat template), 1024 output tokens, input length 1K / 8K × concurrency 1 / 4 / 16 / 64.

> Every number below is a single run (n = 1, seed 0) unless stated, so it is **descriptive, not a claim**: the
> pre-registered criteria in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) require n ≥ 3. Repeats we did run agreed within ≈ 1 %.

## Results at a glance

![Output throughput as % of Backboard](docs/results-vs-backboard.svg)

| | Result |
|---|---|
| **Article baselines reproduced** | vLLM 0.33.0 within **+0.9 … +3.1 %** of the article at **8/8** workloads; SGLang 0.5.23 runs **7–17 % faster** than the article's |
| **Best single lossless config** (`vllm-0.33.0-best-e6b`) | **≥ 95 % of Backboard at 5/8** workloads, **≥ 90 % at 7/8**, geometric mean **97.7 %** (article vLLM recipe: 88.1 %) |
| **Biggest single win** | split-K attention on Gemma 4's head-dim-512 layers via CUDA-graph capture sizes: **+8 %** at 8K/c4 |
| **What is left** | 8K workloads (89–94 %): profile shows the GPU 99.5 % busy and 74 % of time in BF16 GEMMs — kernel work, not configuration |
| **Cost** | ≈ **$12.9** of Modal compute in total (every launch in [`results/budget_ledger.jsonl`](results/budget_ledger.jsonl)) |

### 1. Reproducing the article's open-engine baselines

Output throughput (tok/s), our run vs the article, same recipes (vLLM MTP 4 tokens; SGLang NEXTN 5 steps / 6 tokens):

| Workload | vLLM ours | vLLM article | Δ | SGLang ours | SGLang article | Δ |
|---|---:|---:|---:|---:|---:|---:|
| 1K × 1 | 190.4 | 186.8 | +1.9 % | 136.9 | 122.6 | +11.7 % |
| 1K × 4 | 655.0 | 649.0 | +0.9 % | 499.1 | 456.8 | +9.3 % |
| 1K × 16 | 1905 | 1861 | +2.4 % | 1299 | 1122 | +15.8 % |
| 1K × 64 | 3668 | 3634 | +0.9 % | 2039 | 1750 | +16.5 % ¹ |
| 8K × 1 | 164.6 | 159.6 | +3.1 % | 89.4 | 82.5 | +8.4 % |
| 8K × 4 | 402.6 | 395.7 | +1.8 % | 287.9 | 269.5 | +6.8 % |
| 8K × 16 | 801.9 | 794.5 | +0.9 % | 587.5 | 533.7 | +10.1 % |
| 8K × 64 | 1095 | 1079 | +1.5 % | 728.6 | 673.6 | +8.2 % ¹ |

vLLM's latencies (p99 TPOT, p99 E2EL, p90 TTFT) land at 0.82–1.11× the article's. ¹ SGLang auto-capped itself
at 48 running requests, so its c64 points queue (TTFT 5–20 s); a like-for-like config (`sglang-0.5.23-nextn-fair64`)
is prepared but not run.

### 2. Closing the gap to Backboard

One configuration for all workloads (as Backboard did), every change output-preserving for the BF16 target:

| Workload | Backboard | Article vLLM recipe | **This repo** (`best-e6b`) | % of Backboard |
|---|---:|---:|---:|---:|
| 1K × 1 | 203.6 | 190.4 | **223.9** | **110 %** |
| 1K × 4 | 779.4 | 655.0 | **768.1** | **99 %** |
| 1K × 16 | 2082 | 1905 | **2106** | **101 %** |
| 1K × 64 | 3822 | 3668 | **3710** | **97 %** |
| 8K × 1 | 179.1 | 164.6 | **183.2** | **102 %** |
| 8K × 4 | 529.6 | 402.6 | **497.8** | 94 % |
| 8K × 16 | 961.8 | 801.9 | **853.9** | 89 % ² |
| 8K × 64 | 1254 | 1095 | **1149** | 92 % |

² The same settings without the depth-by-batch schedule reached 885.3 (92 %) at 8K × 16.

**What moved the needle** (details and run IDs in [docs/DECISIONS.md](docs/DECISIONS.md)):

| Change | Effect | Ref |
|---|---|---|
| Speculative depth 6 instead of 4 (acceptance 4.05 → 4.99) | +8 % at 8K/c4 (n = 2, ±1 %) | D-042 |
| CUDA-graph capture sizes `1 2 28 32 …` keep vLLM's Triton **split-K** attention on for 4 requests × 7 tokens (Gemma 4's 8 global layers: 1 KV head, head dim 512) | +8.4 % at 8K/c4, median token gap −12 % | D-044 |
| Patched vLLM: Gemma 4 MTP **draft prefill on the last token only** (the draft is KV-shared and row-wise) | +2.6 % at 8K/c16, −3.6 % 8K TTFT | D-046 |
| Depth by batch size: 6 up to 32 running requests, 4 above | +3 % at 1K/c64; with the draft patch, 8K/c64 1098 → 1149 | D-047 |
| 3072-token prefill chunks | p99 inter-token gap 584 → 274 ms; +1.4 % throughput | D-043 |

**What did not:** patched Triton prefill tiles for sm_120 (no change), depth 5 (never best), depth 7+ (breaks split-K
at c4). Ruled out from source without a GPU run: FlashInfer (no fast path for head dim 512 on sm_120), draft LM-head
quantization (vLLM rejects it for Gemma 4 MTP), partial-prefill flags (absent in v0.33.0).

**Where the rest of the gap is.** A torch-profiler trace of `best-e6b` at 8K/c4 shows the GPU busy 99.5 % of the time:
74 % BF16 GEMMs (CUTLASS sm_80 `mma.sync` kernels — sm_120 has no wgmma/tcgen05), 20 % attention, < 1 % sampling and
speculative bookkeeping. A step-level scheduler model calibrated on our runs (RMS error 0.35 %) says ≈ 10–15 % faster
prefill would bring every 8K workload above 95 %; Backboard's TTFTs are consistently 14–17 % below the article's vLLM,
consistent with faster kernels. Backboard's only disclosed technique — a 4-bit draft shortlist rescored in BF16 —
raises acceptance, which our depth results show is not the main lever at 8K.

## How it works

```
g4bench modal spend                       # free: committed spend vs the cap, reconciled with Modal billing
g4bench modal prefetch --execute          # CPU (~$0.01): pinned weights into a Modal Volume
g4bench modal job --engine vllm-0.33.0-best-e6b --matrix backboard-repro-shared --only 8192x4            # plan only
g4bench modal job --engine vllm-0.33.0-best-e6b --matrix backboard-repro-shared --only 8192x4 --execute  # run it
g4bench modal job ... --profile-key 8192x4 --execute   # torch-profiler summary of one point
g4bench analyze && g4bench claims         # aggregate results/runs → results/summary, docs/CLAIMS.md
```

- **Paid runs go straight to Modal** from your machine (credentials in an untracked `.env`), one ephemeral app per
  launch, nothing left deployed. Engine images are digest-pinned (vLLM v0.33.0, SGLang v0.5.23) and receive no extra
  packages; optional build-time engine patches are anchored, idempotent and recorded with each run.
- **Spend gate** (`src/g4bench/spend.py`): before every launch the worst case (container limits × timeout + margins)
  must fit a per-job and a project cap; the Modal timeout is derived from the reservation, and every launch is
  reconciled against Modal's per-app bill.
- **Every run** writes `results/runs/<run_id>/` with raw InferenceX output, server logs, effective engine config,
  speculative-decoding counters, GPU samples, correctness probes and a `REPORT.md`. Mock, proxy-hardware, profiled
  and non-article-length runs are never compared with the article.

## Quick start (CPU)

```bash
scripts/setup.sh            # venv from requirements-lock.txt, pinned-revision check, engine-flag preflight, tests
```

The harness targets Linux (the GPU images run Python 3.12). On Windows a few process-group tests fail by design;
`g4bench modal selftest --execute` runs the full suite on Linux inside Modal for about a cent.

## Repository layout

```
configs/engines/        engine configs: article baselines (role: baseline) and every experiment (role: variant)
configs/matrix/         workload matrices (article matrix, shared-server variant, smoke, real-prompt arms)
configs/targets/        the article's published numbers, transcribed and test-checked
src/g4bench/            harness: runner, InferenceX client, stats, analysis, Modal app + launcher, spend gate,
                        engine patches (patches/), profiler summaries
results/runs/           every measured run, raw and normalized
results/summary/        aggregated tables (SUMMARY.md, best_vs_backboard.csv, …)
docs/                   DECISIONS.md (decision log D-001…D-050), EXPERIMENTS.md (pre-registered plan and run log),
                        CLAIMS.md, SOURCES.md, analysis/
third_party/inferencex/ vendored InferenceX bench_serving @ 0b991451 (Apache-2.0)
```

## Caveats

- Backboard's engine is proprietary; we reproduce its *numbers* as targets, not its engine.
- n = 1 per point (seed 0); repeats for claims (n = 3) were not run within the budget.
- Points in one job share a server start (`backboard-repro-shared`), unlike the article's fresh server per point.
- Random-token prompts only; real-prompt arms (SPEED-Bench) are prepared but not measured.

Licence: not yet chosen. `third_party/inferencex` keeps its upstream Apache-2.0 licence and notice.
