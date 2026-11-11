# gemma4-inference-repro

An independent, open-source attempt to **reproduce** the Gemma 4 12B serving benchmark that Backboard published on
2026-11-09 ([article](https://backboard.io/blog/gemma-4-12b-vllm-sglang-inference)), and then to **match and beat**
Backboard's proprietary engine with an open engine (vLLM 0.33.0 plus small, documented patches) without changing the
target model's outputs.

Setup throughout: 1× NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB, 600 W) on Modal, `google/gemma-4-12B-it`
BF16 with the `google/gemma-4-12B-it-assistant` MTP draft, random-token prompts (InferenceX client, range ratio 0.8,
chat template), 1024 output tokens, input length 1K / 8K × concurrency 1 / 4 / 16 / 64.

> The headline table uses **n = 3 runs per point** (seeds 0/1/2) with Student-t 95 % confidence intervals, as the
> pre-registered criteria in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) require. Other tables are single runs and are
> descriptive only. Modal assigns a different physical GPU per run; at 8K inputs that machine-to-machine spread is
> ~5–7 % (one run throttled at 83 °C), larger than seed noise.

## Results at a glance

![Output throughput as % of Backboard](docs/results-vs-backboard.svg)

| | Result |
|---|---|
| **Article baselines reproduced** | vLLM 0.33.0 within **+0.9 … +3.1 %** of the article at **8/8** workloads; SGLang 0.5.23 runs **7–17 % faster** than the article's |
| **Beats Backboard (n = 3)** | at **all four 1K workloads**, with every 95 % CI above 100 % (102–115 %) |
| **Ties Backboard (n = 2–3)** | at 8K: 96–108 % of Backboard; confidence intervals span 100 %, dominated by GPU-to-GPU variance |
| **Best configs** | `vllm-0.33.0-best-e8-all` (1K) and `vllm-0.33.0-best-e8-pfa-c8k` (8K: + custom prefill attention, 8192-token chunks) |
| **Configuration only** (`vllm-0.33.0-best-e6b`, no engine patches) | ≥ 95 % of Backboard at 5/8, geometric mean 97.7 % |
| **Biggest wins** | split-K attention for Gemma 4's head-dim-512 layers (+8 % at 8K/c4); FP8 draft-head scan with exact BF16 rescoring (Backboard's disclosed idea); per-layer-group split-K thresholds; a custom prefill-attention kernel (−40 % prefill attention, +5 % at 8K × 64) |
| **What is left** | statistical separation at 8K (more runs, or pinned hardware); BF16 GEMMs on this GPU already run at its practical limit |
| **Cost** | ≈ **$24.1** of Modal compute in total (every launch in [`results/budget_ledger.jsonl`](results/budget_ledger.jsonl)) |

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

### 2. Beating Backboard

Every change keeps the target model's outputs unchanged: the BF16 target verifies every drafted token. The draft head
runs as an FP8 scan with exact BF16 rescoring, which lowers *draft* precision (class b, D-011) — the same category as
Backboard's own disclosed "4-bit shortlist".

**Headline (repeated runs):**

| Workload | Backboard | Ours, mean of n runs | n | % of Backboard | 95 % CI | Config |
|---|---:|---:|---:|---:|---|---|
| 1K × 1 | 203.6 | **234.8** | 3 | **115 %** | 112–119 % | best-e8-all |
| 1K × 4 | 779.4 | **815.2** | 3 | **105 %** | 104–106 % | best-e8-all |
| 1K × 16 | 2082 | **2207** | 3 | **106 %** | 103–109 % | best-e8-all |
| 1K × 64 | 3822 | **3917** | 3 | **102 %** | 102–103 % | best-e8-all |
| 8K × 1 | 179.1 | **193.5** | 3 | 108 % | 97–119 % | best-e8-all |
| 8K × 4 | 529.6 | 521.6 | 2 | 98.5 % | wide (n = 2) | best-e8-pfa-c8k |
| 8K × 16 | 961.8 | **987.9** | 2 | 103 % | wide (n = 2) | best-e8-pfa-c8k |
| 8K × 64 | 1254 | 1201 | 3 | 95.8 % | 85–106 % | best-e8-pfa-c8k |

The 8K rows use the configuration with the custom prefill-attention kernel; it was not run at 1K (it should help
there slightly too). Per-seed values and run IDs are in [docs/DECISIONS.md](docs/DECISIONS.md) (D-059, D-060).

**Single-run progression** (seed 0, one config across all eight workloads, tok/s):

| Workload | Backboard | Article vLLM recipe | Config only (`best-e6b`) | + engine patches (`best-e8-all`) | + prefill kernel (`best-e8-pfa`) |
|---|---:|---:|---:|---:|---:|
| 1K × 1 | 203.6 | 190.4 | 223.9 | 238.1 | – |
| 1K × 4 | 779.4 | 655.0 | 768.1 | 813.1 | – |
| 1K × 16 | 2082 | 1905 | 2106 | 2235 | – |
| 1K × 64 | 3822 | 3668 | 3710 | 3916 | – |
| 8K × 1 | 179.1 | 164.6 | 183.2 | 197.5 | – |
| 8K × 4 | 529.6 | 402.6 | 497.8 | 528.0 | 525.2 |
| 8K × 16 | 961.8 | 801.9 | 853.9 | 945.8 | 986.0 |
| 8K × 64 | 1254 | 1095 | 1149 | 1170 | 1226 |

**What moved the needle** (details and run IDs in [docs/DECISIONS.md](docs/DECISIONS.md)):

| Change | Effect | Ref |
|---|---|---|
| Speculative depth 6 instead of 4 (acceptance 4.05 → 4.99) | +8 % at 8K/c4 (n = 2, ±1 %) | D-042 |
| CUDA-graph capture sizes `1 2 28 32 …` keep vLLM's Triton **split-K** attention on for 4 requests × 7 tokens (Gemma 4's 8 global layers: 1 KV head, head dim 512) | +8.4 % at 8K/c4, median token gap −12 % | D-044 |
| Patched vLLM: Gemma 4 MTP **draft prefill on the last token only** (the draft is KV-shared and row-wise) | +2.6 % at 8K/c16, −3.6 % 8K TTFT | D-046 |
| Depth by batch size: 6 up to 32 running requests, 4 above | +3 % at 1K/c64; with the draft patch, 8K/c64 1098 → 1149 | D-047 |
| 3072-token prefill chunks | p99 inter-token gap 584 → 274 ms; +1.4 % throughput | D-043 |
| Patched vLLM: **FP8 draft-head scan + exact BF16 rescore** (the 12B drafter scores all 262 144 tokens in BF16 every draft step) | head 395 → 210 µs per draft step; acceptance unchanged (5.25 at 8K/c4) | D-054 |
| Patched vLLM: **split-K threshold per attention group** (global and sliding layers want opposite kernels) | attention −3.4 ms/forward at 8K/c16 | D-053 |
| cuBLAS 13.10 instead of the image's 13.1 | −2 to −3 % per forward on every GEMM size | D-052 |
| Patched vLLM: **custom prefill attention** (FlashAttention-2-style, packs the 16 query heads that share Gemma 4's single global KV head; sliding-window variant) | prefill attention per 8K prompt 190 → 114 ms; 8K × 16 +4 %, 8K × 64 +5 % | D-059 |

**What did not:** patched Triton prefill tiles for sm_120 (no change), depth 5 (never best), depth 7+ (breaks split-K
at c4). Ruled out from source without a GPU run: FlashInfer (no fast path for head dim 512 on sm_120), draft LM-head
quantization (vLLM rejects it for Gemma 4 MTP), partial-prefill flags (absent in v0.33.0).

### 3. Kernel lab: where the time goes on sm_120

`g4bench modal script --path kernels/<bench>.py --execute` runs micro-benchmarks on the same GPU inside the pinned
image (results in [`results/kernels/`](results/kernels/), cents per run):

- **GEMMs are at the hardware limit.** With cold weights, decode linears take 15.8 ms per forward at 28 tokens vs a
  14.9 ms weight-read floor (1.46 TB/s); prefill shapes run at 341–427 TFLOPS vs a measured ~415–440 peak.
  `torch.compile` max-autotune and the CUTLASS CuTe-DSL sm_120 GEMM never beat cuBLAS; cuBLAS 13.10 is 2–3 % faster.
- **Attention:** for decode, a packed-row verify kernel saves only ~0.5 ms at 8K/c4 (repeat KV reads hit the 128 MB
  L2), but vLLM's single split-K threshold picks the wrong kernel for one of Gemma 4's two layer types at c16/c64.
  For prefill, vLLM's generic Triton kernel ran at 57–90 TFLOPS; our packed-head kernel reaches 100–137 TFLOPS.
- **Draft head:** the quantized scan reads 4× (INT4) or 2× (FP8) fewer bytes; FP8 is shipped for its higher recall.
- **Stronger drafters:** DeepSeek's DSpark block-7 drafter is slower on random-token prompts (no acceptance gain).

**Where the rest of the gap is.** At 8K the remaining difference to Backboard is smaller than the run-to-run spread
between Modal GPUs. Prefill GEMMs are already at peak; prefill attention still has headroom (≈ 130 TFLOPS vs ≈ 400 for
GEMMs), which is the most promising next kernel.

## How it works

```
g4bench modal spend                       # free: committed spend vs the cap, reconciled with Modal billing
g4bench modal prefetch --execute          # CPU (~$0.01): pinned weights into a Modal Volume
g4bench modal job --engine vllm-0.33.0-best-e8-all --matrix backboard-repro-shared --only 8192x4            # plan only
g4bench modal job --engine vllm-0.33.0-best-e8-all --matrix backboard-repro-shared --only 8192x4 --execute  # run it
g4bench modal script --path kernels/gemm_bench.py --execute   # kernel micro-benchmark on the GPU
g4bench modal job ... --profile-key 8192x4 --execute   # torch-profiler summary of one point
g4bench analyze && g4bench claims         # aggregate results/runs → results/summary, docs/CLAIMS.md
```

- **Paid runs go straight to Modal** from your machine (credentials in an untracked `.env`), one ephemeral app per
  launch, nothing left deployed. Engine images are digest-pinned (vLLM v0.33.0, SGLang v0.5.23). A separate patched
  vLLM image carries build-time patches (`src/g4bench/patches/`: anchored, idempotent, opt-in per config) plus
  cuBLAS 13.10; every run records which patches and settings were active.
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
kernels/                GPU micro-benchmarks (GEMM, cuBLAS versions, CuTe DSL, attention, draft-head scan)
results/runs/           every measured run, raw and normalized
results/kernels/        micro-benchmark outputs
results/summary/        aggregated tables (SUMMARY.md, best_vs_backboard.csv, …)
docs/                   DECISIONS.md (decision log D-001…D-056), EXPERIMENTS.md (pre-registered plan and run log),
                        CLAIMS.md, SOURCES.md, analysis/
third_party/inferencex/ vendored InferenceX bench_serving @ 0b991451 (Apache-2.0)
```

## Caveats

- Backboard's engine is proprietary; we reproduce its *numbers* as targets, not its engine.
- The headline uses n = 3 at seven points and n = 2 at two 8K points; the other tables are single runs. GPU-to-GPU
  variance on Modal (~5–7 % at 8K) dominates the 8K intervals.
- Points in one job share a server start (`backboard-repro-shared`), unlike the article's fresh server per point.
- Random-token prompts only; real-prompt arms (SPEED-Bench) are prepared but not measured. Acceptance, and therefore
  the speculative-decoding gains, depends on prompt content.
- `best-e8-all` lowers draft-head precision (FP8 scan, class b); target outputs are unchanged.

Licence: not yet chosen. `third_party/inferencex` keeps its upstream Apache-2.0 licence and notice.
