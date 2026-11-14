# gemma4-inference-repro

An independent, open-source attempt to **reproduce** the Gemma 4 12B serving benchmark that Backboard published on
2026-11-09 ([article](https://backboard.io/blog/gemma-4-12b-vllm-sglang-inference)), and then to get as close as
possible to Backboard's proprietary engine with an open engine (vLLM 0.33.0 plus small, documented patches).

Setup throughout: 1× NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB, 600 W) on Modal, `google/gemma-4-12B-it`
BF16 with the `google/gemma-4-12B-it-assistant` MTP draft, random-token prompts (InferenceX client, range ratio 0.8,
chat template), 1024 output tokens, input length 1K / 8K × concurrency 1 / 4 / 16 / 64.

> **Read this first.** Backboard's figures are single published values from unstated hardware, so every comparison
> with Backboard here is **descriptive**, not a pre-registered "beats". The strongest evidence is the **same-GPU paired**
> measurement below: the article's own vLLM recipe and our configuration run interleaved on one GPU, which cancels
> Modal's 5–15 % GPU-to-GPU spread, and the speed-up is then scaled by the article's published vLLM numbers. An
> independent adversarial review of these claims is summarised in [Caveats](#caveats).

## Results at a glance

![Throughput and p90 TTFT vs Backboard, same-GPU paired runs](docs/results-vs-backboard.svg)

| | Result |
|---|---|
| **Article baselines reproduced** | vLLM 0.33.0 within **+0.9 … +3.1 %** of the article at **8/8** workloads; SGLang 0.5.23 runs **7–17 % faster** than the article's |
| **Throughput, same GPU, one frozen config** (`vllm-0.33.0-best-e8-pfa2-sync-hybtok` vs the article recipe, interleaved) | **1.07–1.33× the article's vLLM recipe** at every workload. Scaled to Backboard: **ahead at concurrency 1 (+14 %)**, ahead by +2–6 % at 1K × 4/16/64 and 8K × 16, **level at 8K × 4 and 8K × 64** (geometric mean 1.05) |
| **Latency, same runs** | p90 TTFT below Backboard's at **7/8** workloads (8K × 1 a tie: 552 vs 549 ms over 4 pairs); median TTFT 16–26 % below the article recipe everywhere. p99 E2EL better at 7/8, p99 TPOT at 4/8 (behind at 8K × 4 by 9 %, 8K × 64 by 4 %, 8K × 16 by 4 %, 1K × 16 tie) |
| **Earlier, across GPUs** (`vllm-0.33.0-best-e8-all`, n = 3) | **102.1 %** of Backboard's published throughput (geometric mean): 102–115 % at 1K, 89–108 % at 8K; **8K × 64 is a significant shortfall (89 %, CI 80–98 %)** |
| **Outputs** | greedy equivalence vs a BF16 no-speculation reference (8 prompts up to 7.9K tokens): every divergence of every config, the article's own recipe included, is a near-tie (top-2 within 0.25 nats); bitwise identity is impossible with speculative decoding in vLLM (D-063) |
| **Configuration only** (`vllm-0.33.0-best-e6b`, no engine patches) | ≥ 95 % of Backboard at 5/8, geometric mean 97.7 % |
| **Biggest wins** | for TTFT, turning off async scheduling (median −22 % at 1K × 16/64, throughput unchanged) and fast prompt tokenization (−16 ms at 8K); split-K attention for Gemma 4's head-dim-512 layers (+8 % at 8K/c4); FP8 draft-head scan with exact BF16 rescoring (Backboard's disclosed idea); per-layer-group split-K thresholds; a custom prefill-attention kernel (−40 % prefill attention, +5 % at 8K × 64) |
| **What is left** | 8K × 4 p99 TPOT (a TTFT/TPOT trade-off with vLLM's scheduler options, D-069); 8K prefill (78 % GEMMs already at 400–430 TFLOPS); real-text prompts |

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

### 2. Comparison with Backboard's published numbers

What we did and did not establish:

- **Same-GPU paired runs** (D-061, D-067, D-068, `results/pairs/`): in one container on one GPU, a fresh server for the
  article's recipe (`vllm-0.33.0-mtp4`, A) and one for ours (`vllm-0.33.0-best-e8-pfa2-sync-hybtok`, B), interleaved.
  Backboard estimate = (B / A) × (article vLLM / Backboard). One frozen configuration for all eight workloads, seed 0
  (8K × 4 also checked on held-out seeds 1 and 2 with an earlier config: B / A 1.320 and 1.335 vs 1.326). Latency cells
  are means over pairs of each run's percentile; bold = better than Backboard.

| Workload | Pairs (GPUs) | A tok/s | B tok/s | B / A | **Est. vs Backboard** | p90 TTFT ms B / A / Backboard | median TTFT ms B / A | p99 TPOT ms B / Backboard | p99 E2EL ms B / Backboard |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1K × 1 | 2 (1) | 190 | 235 | 1.238 | **1.14** | **81** / 104 / 85 | 75 / 96 | **4.75** / 5.0 | **4861** / 5074 |
| 1K × 4 | 2 (1) | 651 | 819 | 1.258 | **1.05** | **152** / 190 / 176 | 95 / 126 | **5.70** / 6.1 | **5162** / 5846 |
| 1K × 16 | 2 (1) | 1890 | 2248 | 1.190 | **1.06** | **218** / 261 / 245 | 111 / 145 | 8.73 / 8.7 | **8381** / 8452 |
| 1K × 64 | 1 (1) | 3672 | 3942 | 1.073 | **1.02** | **510** / 686 / 547 | 213 / 289 | **18.97** / 19.4 | **19237** / 20123 |
| 8K × 1 | 4 (2) | 162 | 207 | 1.274 | **1.14** | 552 / 658 / 549 | 522 / 625 | **5.53** / 6.0 | **5833** / 6276 |
| 8K × 4 | 4 (2) | 401 | 535 | 1.334 | **1.00** | **656** / 1031 / 724 | 538 / 673 | 11.25 / 10.3 | 11637 / 10414 |
| 8K × 16 | 3 (3) | 800 | 1007 | 1.257 | **1.04** | **1327** / 1637 / 1537 | 584 / 708 | 24.52 / 23.6 | **23288** / 23806 |
| 8K × 64 | 1 (1) | 1086 | 1264 | 1.164 | **1.00** | **3005** / 5799 / 3102 | 1069 / 1447 | 76.96 / 73.8 | **77953** / 78977 |

  Geometric mean of the throughput estimates **1.054**. Previous configs on the same harness agree: with async
  scheduling off but the stock tokenizer (`-sync`), 8K × 64 gave B / A 1.164 on three GPUs and p90 TTFT 2973–2989 ms,
  and 1K × 64 gave p90 TTFT 437–512 ms on two GPUs. How to read it:
  - **Throughput:** pairs at one point agree within 0.1–1.4 %, so the uncertainty is the estimator, which trusts the
    article's single vLLM and Backboard figure per point; ±2–3 % there would erase the smaller margins. Read 1.02–1.06
    as small wins at best and 8K × 4 / 8K × 64 as **parity**.
  - **p90 TTFT at c ≥ 4 mostly measures one cold burst:** the client sends `10 × c` requests with all `c` arriving at
    once on an idle server, so that burst is exactly the slowest 10 %. Repeats on one GPU differ by up to 60 % at 8K × 4
    (633–1014 ms with the D-067 config); median TTFT is the steadier number. With this config 8K × 4 p90 stayed at
    610–677 ms over 4 pairs, and 8K × 16 gave 1171 / 1195 / 1614 ms (2 of 3 below Backboard).
  - **Scaled TTFT** ((B/A) × article vLLM / Backboard) agrees at most points but swings at 8K × 4 (our article-recipe
    runs had lower p90 than the article's 1262 ms) and 8K × 64 (higher than its 3612 ms): both are indeterminate.
  - **TPOT at 8K × 4:** set by the few requests that sit through other requests' whole 8K prefill steps; median TPOT
    is 6.15 vs 8.5 ms for the article recipe. Splitting prefills under load (`--long-prefill-token-threshold 2048`)
    brings p99 TPOT to 10.4–10.8 ms but raises p90 TTFT to 0.97–1.13 s; we kept the TTFT side (D-069).
  - **Costs of async scheduling off:** p99 inter-token latency +1–5 % at 1K × 64 on the same GPU; throughput unchanged.

- **Earlier, across GPUs** (`vllm-0.33.0-best-e8-all`, n = 3 runs on different Modal GPUs,
  Student-t 95 % CI of *our* run-to-run spread — Backboard's own variance and hardware are unknown):

| Workload | Backboard | Ours (mean, n = 3) | % of Backboard | 95 % CI (ours) |
|---|---:|---:|---:|---|
| 1K × 1 | 203.6 | 234.8 | 115 % | 112–119 % |
| 1K × 4 | 779.4 | 815.2 | 105 % | 104–106 % |
| 1K × 16 | 2082 | 2207 | 106 % | 103–109 % |
| 1K × 64 | 3822 | 3917 | 102 % | 102–103 % |
| 8K × 1 | 179.1 | 193.5 | 108 % | 97–119 % |
| 8K × 4 | 529.6 | 507.9 | 96 % | 87–105 % |
| 8K × 16 | 961.8 | 941.0 | 98 % | 95–101 % |
| 8K × 64 | 1254 | 1115.6 | **89 %** | **80–98 %** |

  Geometric mean **102.1 %**. The 1K margins (+2.5 … +15 %) are of the same order as our environment's offset on the
  article's own vLLM recipe (+0.9 … +3.1 %), so the smallest of them may not survive a hardware-matched comparison.
- **Custom prefill attention** (`best-e8-pfa-c8k`, added later and chosen per workload): 8K × 4 98.5 % (n = 2,
  CI 66–131 %), 8K × 16 103 % (n = 2, CI 87–119 %), 8K × 64 95.8 % (n = 3, CI 85–106 %; 94.1 % without the seed-0
  run used to select it). Not run at 1K or 8K × 1.
- **Latency** (in the table above): Backboard reports more metrics than the three compared here.
- **Output equivalence: near-ties only** (D-063). Against a BF16 no-speculation greedy reference (8 prompts, 4 of them
  3K–8K tokens, 256 output tokens, concurrency 1), the article's own recipe matches 4/8 and ours 3/8 (all 16 runs of
  the final config); every first
  divergence, for every config, is at a position where the reference's top-2 tokens are within 0.25 nats. This is a
  weaker guarantee than Backboard's stated "identical to BF16" output. The custom kernels, cuBLAS
  13.10 and chunk size change the target's arithmetic only at that level; the tokenizer change produces identical ids. The draft head runs as an FP8 scan with
  BF16 rescoring (class b, D-011), the same category as Backboard's disclosed "4-bit shortlist".

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
| Prefill attention **v2** in Gluon for sm_120's `mma.sync` (~190 TFLOPS) plus 8192-token chunks | prefill attention per 8K prompt 114 → 73 ms | D-064 |

**What did not:** patched Triton prefill tiles for sm_120 (no change), depth 5 (never best), depth 7+ (breaks split-K
at c4). Ruled out from source without a GPU run: FlashInfer (no fast path for head dim 512 on sm_120), draft LM-head
quantization (vLLM rejects it for Gemma 4 MTP), partial-prefill flags (absent in v0.33.0).

### 3. Kernel lab: where the time goes on sm_120

`g4bench modal script --path kernels/<bench>.py --execute` runs micro-benchmarks on the same GPU inside the pinned
image (results in [`results/kernels/`](results/kernels/)):

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
g4bench modal prefetch --execute          # CPU: pinned weights into a Modal Volume
g4bench modal job --engine vllm-0.33.0-best-e8-all --matrix backboard-repro-shared --only 8192x4            # plan only
g4bench modal job --engine vllm-0.33.0-best-e8-all --matrix backboard-repro-shared --only 8192x4 --execute  # run it
g4bench modal script --path kernels/gemm_bench.py --execute   # kernel micro-benchmark on the GPU
g4bench modal job ... --profile-key 8192x4 --execute   # torch-profiler summary of one point
g4bench analyze && g4bench claims         # aggregate results/runs → results/summary, docs/CLAIMS.md
```

- **GPU runs go straight to Modal** from your machine (credentials in an untracked `.env`), one ephemeral app per
  launch, nothing left deployed. Engine images are digest-pinned (vLLM v0.33.0, SGLang v0.5.23). A separate patched
  vLLM image carries build-time patches (`src/g4bench/patches/`: anchored, idempotent, opt-in per config) plus
  cuBLAS 13.10; every run records which patches and settings were active.
- **Every run** writes `results/runs/<run_id>/` with raw InferenceX output, server logs, effective engine config,
  speculative-decoding counters, GPU samples, correctness probes and a `REPORT.md`. Mock, proxy-hardware, profiled
  and non-article-length runs are never compared with the article.
- **Not committed:** the per-request InferenceX dumps (`points/*/inferencex_result.json`), profiler traces and
  tokenizer-probe id dumps stay local (`.gitignore`). Everything the tables are built from is committed. The dumps
  from runs up to 2026-11-14 (≈ 400 MB) are in git history:
  `git checkout "$(git log -1 --format=%H --grep='Merge pull request #25')^1" -- results/runs/<run_id>/points/<point>/inferencex_result.json`.

## Quick start (CPU)

```bash
scripts/setup.sh            # venv from requirements-lock.txt, pinned-revision check, engine-flag preflight, tests
```

The harness targets Linux (the GPU images run Python 3.12). On Windows a few process-group tests fail by design;
`g4bench modal selftest --execute` runs the full suite on Linux inside Modal.

## Repository layout

```
configs/engines/        engine configs: article baselines (role: baseline) and every experiment (role: variant)
configs/matrix/         workload matrices (article matrix, shared-server variant, smoke, real-prompt arms)
configs/targets/        the article's published numbers, transcribed and test-checked
src/g4bench/            harness: runner, InferenceX client, stats, analysis, Modal app + launcher,
                        engine patches (patches/), profiler summaries
kernels/                GPU micro-benchmarks (GEMM, cuBLAS versions, CuTe DSL, attention, draft-head scan)
results/runs/           every measured run: normalized results, logs, configs (raw per-request dumps stay local)
results/pairs/          same-GPU paired comparisons (paired.json per pair job)
results/kernels/        micro-benchmark outputs
results/summary/        aggregated tables (SUMMARY.md, best_vs_backboard.csv, …)
docs/                   DECISIONS.md (decision log D-001…D-069), EXPERIMENTS.md (pre-registered plan and run log),
                        CLAIMS.md, SOURCES.md, analysis/
third_party/inferencex/ vendored InferenceX bench_serving @ 0b991451 (Apache-2.0)
```

## Caveats

From two independent adversarial reviews of the comparison (2026-11-13; the second covered the latency claims):

- **Output equivalence is weaker than pre-registered.** The greedy gate now runs (D-063), but exact token identity is
  impossible for any speculative config, the article's included; the amended near-tie criterion was set after seeing
  that, and 8 prompts is a small set.
- **Selection after the fact.** About 20 engine configurations were tuned on seed-0 prompts, the same seed the paired
  table uses. Held-out seeds 1 and 2 at 8K × 4 reproduce the seed-0 speed-up (1.320 / 1.335 vs 1.326), but other
  points were not re-checked on held-out seeds.
- **Synthetic prompts.** Most of the low-concurrency gain comes from deeper speculation (acceptance length ≈ 5.1 vs
  4.1 for the article recipe), which random-token prompts may favour; the article does not state its prompt method.
- **Hardware not matched.** The article does not state its GPU edition, power limit or driver. Modal GPUs vary by
  7–15 % at 8K (runs at ~2,200 MHz / ~520 W vs ~2,370 MHz / ~570 W); the confidence intervals mix GPUs and cover only
  our spread, not Backboard's. The paired runs cancel this for ours vs the article recipe, but the step to Backboard
  still assumes the article's vLLM and Backboard figures came from comparable hardware. Our environment also runs the article's SGLang recipe 7–17 % faster than published.
- **Workload assumptions the article does not state:** random-token prompts, range ratio 0.8, ignore-EOS, 10 requests
  per unit of concurrency (only 10 measured requests at c1), 8 warm-ups. Random tokens may inflate draft acceptance.
  The paired runs start a fresh server per point and engine, as the article does; earlier single-engine jobs did not.
- **Provenance:** about half of the runs before 2026-11-13 lack the code SHA in their metadata; later runs record it.
- **p90 latency is fragile at c ≥ 4** (one cold burst is the slowest 10 %, see section 2); 1K × 64 and 8K × 64 have 1
  pair with this exact config (2–3 with the near-identical `-sync` config).
- **To settle it:** ≥ 5 same-GPU pairs per workload rotated across GPUs; real-text prompts; more requests per point so
  that p90/p99 are not set by a handful of requests. A same-GPU run of Backboard's engine is not possible (proprietary).

Licence: not yet chosen. `third_party/inferencex` keeps its upstream Apache-2.0 licence and notice.
