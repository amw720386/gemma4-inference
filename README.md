# gemma4-inference-repro

Backboard published a Gemma 4 12B serving benchmark on 2026-11-09
([article](https://backboard.io/blog/gemma-4-12b-vllm-sglang-inference)) in which its proprietary engine beat vLLM
and SGLang. This repository reproduces that benchmark with open tooling, then goes past it: **open-source vLLM
0.33.0 with a handful of small, documented patches beats Backboard's published throughput on average and its
published p90 time-to-first-token at 7 of 8 workloads.**

Setup: 1× NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB) on Modal, `google/gemma-4-12B-it` in BF16 with the
`google/gemma-4-12B-it-assistant` MTP draft, the article's workload (InferenceX client, random-token prompts,
1024 output tokens, input 1K / 8K × concurrency 1 / 4 / 16 / 64).

## Results

![Throughput and p90 TTFT vs Backboard, same-GPU paired runs](docs/results-vs-backboard.svg)

One frozen configuration, [`vllm-0.33.0-best-e8-pfa2-sync-hybtok`](configs/engines/vllm-0.33.0-best-e8-pfa2-sync-hybtok.yaml),
for all eight workloads, measured head-to-head against the article's own vLLM recipe on the same GPU:

- **Throughput 1.07–1.33× the article's vLLM recipe** at every workload.
- **Ahead of Backboard's published throughput** at 6 of 8 workloads (up to +14 %), level at 8K × 4 and 8K × 64;
  geometric mean **1.054×**.
- **p90 TTFT below Backboard's at 7 of 8 workloads** (8K × 1 is a tie) and 16–48 % below the article recipe's;
  median TTFT 16–26 % below the article recipe's everywhere.
- **p99 end-to-end latency better than Backboard's at 7 of 8**, p99 TPOT better or level at 5 of 8.

| Workload | Pairs | Throughput vs article recipe (same GPU) | **Throughput vs Backboard (est.)** | p90 TTFT ms: ours / article recipe / Backboard | p99 TPOT ms: ours / Backboard | p99 E2EL ms: ours / Backboard |
|---|---:|---:|---:|---:|---:|---:|
| 1K × 1 | 2 | 1.238 | **1.14** | **81** / 104 / 85 | **4.75** / 5.0 | **4861** / 5074 |
| 1K × 4 | 2 | 1.258 | **1.05** | **152** / 190 / 176 | **5.70** / 6.1 | **5162** / 5846 |
| 1K × 16 | 2 | 1.190 | **1.06** | **218** / 261 / 245 | 8.73 / 8.7 | **8381** / 8452 |
| 1K × 64 | 1 | 1.073 | **1.02** | **510** / 686 / 547 | **18.97** / 19.4 | **19237** / 20123 |
| 8K × 1 | 4 | 1.274 | **1.14** | 552 / 658 / 549 | **5.53** / 6.0 | **5833** / 6276 |
| 8K × 4 | 4 | 1.334 | 1.00 | **656** / 1031 / 724 | 11.25 / 10.3 | 11637 / 10414 |
| 8K × 16 | 3 | 1.257 | **1.04** | **1327** / 1637 / 1537 | 24.52 / 23.6 | **23288** / 23806 |
| 8K × 64 | 1 | 1.164 | 1.00 | **3005** / 5799 / 3102 | 76.96 / 73.8 | **77953** / 78977 |

Bold = better than Backboard. "Throughput vs Backboard" = (ours / article recipe, same GPU) × (article's vLLM /
Backboard, as published). Latencies are means over pairs. Raw data: [`results/pairs/`](results/pairs/), aggregated in
[`results/summary/SUMMARY.md`](results/summary/SUMMARY.md).

**The baselines reproduce.** The article's vLLM recipe lands within +0.9 … +3.1 % of its published throughput at
8/8 workloads (+0.5 … +1.6 % over the 6–10 runs per workload it got as the reference side of the pairs), and its
SGLang recipe runs 7–17 % faster than published:

| Workload | vLLM ours | vLLM article | Δ | SGLang ours | SGLang article | Δ |
|---|---:|---:|---:|---:|---:|---:|
| 1K × 1 | 190.4 | 186.8 | +1.9 % | 136.9 | 122.6 | +11.7 % |
| 1K × 4 | 655.0 | 649.0 | +0.9 % | 499.1 | 456.8 | +9.3 % |
| 1K × 16 | 1905 | 1861 | +2.4 % | 1299 | 1122 | +15.8 % |
| 1K × 64 | 3668 | 3634 | +0.9 % | 2039 | 1750 | +16.5 % |
| 8K × 1 | 164.6 | 159.6 | +3.1 % | 89.4 | 82.5 | +8.4 % |
| 8K × 4 | 402.6 | 395.7 | +1.8 % | 287.9 | 269.5 | +6.8 % |
| 8K × 16 | 801.9 | 794.5 | +0.9 % | 587.5 | 533.7 | +10.1 % |
| 8K × 64 | 1095 | 1079 | +1.5 % | 728.6 | 673.6 | +8.2 % |

Output tok/s, seed 0, same recipes as the article (vLLM MTP 4 tokens; SGLang NEXTN 5 steps / 6 tokens).

## What made the difference

Configuration changes alone (`vllm-0.33.0-best-e6b`, no patches) already reach 97.7 % of Backboard (geometric
mean). The rest comes from engine patches in [`src/g4bench/patches/`](src/g4bench/patches/): anchored, idempotent,
enabled per config, applied when the image is built.

| Change | Effect | Ref |
|---|---|---|
| **Async scheduling off** (`--no-async-scheduling`): vLLM 0.33's async mode admits a new request one engine step late | p90 TTFT 302 → 219 ms at 1K × 16, 600 → 448 ms at 1K × 64; throughput unchanged | D-067 |
| **Hybrid tokenizer** (patch): fastokens for prompt encoding, HF tokenizers for decode; identical ids and text | prompt encode 13.5 → 1.1 ms per 7.4K tokens; 8K × 1 median TTFT −17 ms, throughput unchanged | D-068 |
| **Split-K attention for Gemma 4's head-dim-512 global layers**, kept on by choosing CUDA-graph capture sizes | +8.4 % throughput at 8K × 4 | D-044 |
| **FP8 draft-head scan with exact BF16 rescoring** of the top 32 (patch; the drafter otherwise scores all 262 144 tokens in BF16 every step) | draft head 395 → 210 µs per step, acceptance unchanged | D-054 |
| **Split-K thresholds per attention layer group** (patch; global and sliding layers want opposite kernels) | attention −3.4 ms per forward at 8K × 16 | D-053 |
| **Custom prefill attention** (patch): FlashAttention-2-style kernel in Gluon for sm_120 that packs the 16 query heads sharing the single global KV head, plus a sliding-window variant | prefill attention per 8K prompt 191 → 73 ms; +4–5 % throughput at 8K × 16 / 64 | D-059, D-064 |
| Deeper speculation: 6 draft tokens up to 32 running requests, 4 above | +8 % at 8K × 4, +3 % at 1K × 64 | D-042, D-047 |
| Draft prefill on the last token only (patch; the Gemma 4 draft is KV-shared) | +2.6 % at 8K × 16 | D-046 |
| cuBLAS 13.10 instead of the image's 13.1 | −2 to −3 % per forward on every GEMM size | D-052 |

Micro-benchmarks on the same GPU ([`kernels/`](kernels/), results in [`results/kernels/`](results/kernels/)) show
the GEMMs already at the hardware limit (decode near the weight-read floor, prefill at 400–430 TFLOPS), so further
gains would have to come from attention and fusion. The full decision log is [docs/DECISIONS.md](docs/DECISIONS.md)
(D-001 … D-071); the pre-registered plan and run log is [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md); the article's
claims and their status are in [docs/CLAIMS.md](docs/CLAIMS.md).

## Reproduce

```bash
scripts/setup.sh                    # CPU: venv from requirements-lock.txt, pinned-revision check, flag preflight, tests
g4bench modal prefetch --execute    # pinned weights into a Modal Volume

# Same-GPU pair: the article recipe (A) against the final config (B), interleaved in one container on one GPU
g4bench modal pair --a vllm-0.33.0-mtp4 --b vllm-0.33.0-best-e8-pfa2-sync-hybtok \
    --matrix backboard-repro-shared --only 8192x4 --repeats 2              # prints the plan
g4bench modal pair ... --execute                                          # runs it

g4bench modal job --engine vllm-0.33.0-mtp4 --matrix backboard-repro-shared --only 1024x1 --execute   # one config
g4bench modal script --path kernels/gemm_bench.py --execute                                         # a kernel micro-benchmark
g4bench analyze && g4bench claims   # results/ → results/summary/, docs/CLAIMS.md
```

Modal credentials go in an untracked `.env` (`MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`). Every launch is an ephemeral
Modal app; nothing stays deployed. Engine images are digest-pinned (vLLM v0.33.0, SGLang v0.5.23). Every run writes
`results/runs/<run_id>/` with normalized results, server and client logs, the effective engine config, active
patches, speculative-decoding counters, GPU samples and correctness probes; per-request client dumps and profiler
traces stay local. The harness targets Linux (the images run Python 3.12); `g4bench modal selftest --execute` runs
the test suite on Linux inside Modal.

## Repository layout

```
configs/engines/        engine configs: the article's recipes (role: baseline) and every experiment (role: variant)
configs/matrix/         workload matrices
configs/targets/        the article's published numbers, transcribed and test-checked
src/g4bench/            harness: runner, InferenceX client, stats, analysis, Modal launcher, engine patches (patches/)
kernels/                GPU micro-benchmarks and the prefill-attention kernels
results/runs/           every measured run
results/pairs/          same-GPU paired comparisons (paired.json per pair job)
results/kernels/        micro-benchmark outputs
results/summary/        aggregated tables (SUMMARY.md, paired.csv, …)
docs/                   DECISIONS.md, EXPERIMENTS.md, CLAIMS.md, SOURCES.md, analysis/
third_party/inferencex/ vendored InferenceX bench_serving @ 0b991451 (Apache-2.0)
```

## Notes on method

- **Estimator.** Backboard's engine is proprietary and its GPU edition, power limit and driver are not stated, so
  it cannot be run here. The comparison takes our measured speed-up over the article's vLLM recipe on one GPU and
  scales it by the article's own vLLM-to-Backboard ratio. Pairs at one workload agree within 0.1–1 % (3 % at
  8K × 4), so the uncertainty sits mostly in the article's single published figures: ±2–3 % there would cover the 1.02–1.06
  margins. Pairing cancels Modal's 5–15 % GPU-to-GPU spread; 1K × 64 and 8K × 64 have one pair with this exact
  config (three with the near-identical `-sync` config, same result).
- **Where we are behind.** p99 TPOT trails Backboard at 8K × 4 (+9 %), 8K × 16 (+4 %) and 8K × 64 (+4 %), and so
  does p99 E2EL at 8K × 4. At 8K × 4 it is set by the few requests that wait through other requests' 8K prefill
  steps (median TPOT 6.15 vs 8.5 ms for the article recipe); splitting prefills brings p99 TPOT to 10.4–10.8 ms but
  pushes p90 TTFT to 0.97–1.13 s, so the frozen config keeps the TTFT side (D-069).
- **p90 TTFT at concurrency ≥ 4** largely measures the first burst of `c` simultaneous arrivals on an idle server
  and varies between repeats (8K × 4: 610–677 ms over 4 pairs; 8K × 16: 1171–1614 ms over 3); medians are steadier.
- **Workload assumptions** the article leaves open: random-token prompts, range ratio 0.8, ignore-EOS, 10 requests
  per unit of concurrency, 8 warm-ups, a fresh server per point. Part of the low-concurrency gain is deeper
  speculation (acceptance length ≈ 5.1 vs 4.1), which random-token prompts may favour; real-text prompts were not
  run. Configurations were tuned on seed 0, the seed the table uses; held-out seeds 1 and 2 at 8K × 4 reproduce the
  speed-up (1.320 / 1.335 vs 1.326).
- **Outputs.** The target model stays BF16. Against a BF16 no-speculation greedy reference (8 prompts up to 7.9K
  tokens), every divergence of every configuration, the article's recipe included, falls where the reference's top
  two tokens are within 0.25 nats; bitwise identity is not attainable with speculative decoding in vLLM (D-063).
  The FP8 draft-head scan is the same category of change as Backboard's disclosed 4-bit shortlist.

## License

Not yet chosen. `third_party/inferencex` keeps its upstream Apache-2.0 licence and notice.
