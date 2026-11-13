# Experiment plan

Goal: an **independent, open-source, honestly-reported** reproduction of the vLLM and SGLang
baselines in Backboard's Gemma 4 12B article, followed by measured optimization research that
tries to match or beat Backboard's *reported* numbers with open tooling. We do **not** reproduce
Backboard's proprietary engine; its numbers are treated as external targets only.

Status legend: ⬜ not started · 🟨 in progress · ✅ done · ⛔ blocked

## Pre-registered acceptance criteria

Fixed before any GPU data exists (do not edit after seeing data; add a dated amendment instead).

1. **Validity of a point-run**: client outcome `passed`, 0 failed requests, mean output length in
   `[0.8·OSL, OSL]` (ignore-EOS honoured), all metrics finite and positive, server started fresh
   (main matrix), correctness probes ≥ 7/8 on that server build.
2. **Uncertainty**: every reported number is a run-level mean over n ≥ 3 runs (seeds 0,1,2; ≥ 1 trial)
   with a Student-t 95 % CI and CV. Single runs are labelled `n=1` and never used for claims.
   Per-request bootstrap CIs (request_stats.py) are within-run diagnostics only (D-018).
   *(Amendment 2026-11-10, before any data: clarifies the role of within-run CIs; no threshold changed.)*
3. **Baseline reproduced** (per engine, per workload, output throughput): article value lies within our
   95 % CI, or |ours − article| / article ≤ **5 %** (systematic tolerance for GPU edition / driver / host).
   Engine baseline "reproduced" overall if ≥ 6 / 8 workloads pass; otherwise we report the gap and
   diagnose it (acceptance length, TTFT, scheduler) before any optimization claim.
4. **"Matches Backboard"**: our mean output throughput within 5 % of Backboard's reported value,
   same exact hardware class (1× RTX PRO 6000 96 GB), BF16 target weights, output-preserving gate passed.
   **"Beats Backboard"**: lower 95 % CI bound > Backboard's value (n ≥ 3), with p99 TPOT and p99 E2EL
   reported alongside (a throughput win with a latency regression is reported as such, not as a win).
5. **Output preservation** for any optimization (correctness.py): known-answer probes ≥ 7/8 and greedy
   equivalence gate vs the BF16 no-spec reference. Anything that lowers draft precision is reported in a
   separate "InferenceX-inadmissible but output-preserving" category (see DECISIONS D-011).
6. **Exact vs proxy hardware**: results from any GPU other than 1× RTX PRO 6000 96 GB are labelled
   `proxy` (`exact_hardware=false` in CSV) and never compared to article numbers as a reproduction.

## Cost model

`g4bench plan` (src/g4bench/cost.py) – pessimistic: assumes we run 1.5× slower than the article's
same-engine numbers, 6 min per fresh server start, 4 min container overhead, Modal RTX-PRO-6000 at
GPU + CPU/memory *limits* ($0.001089/s = $3.92/h; D-021). **Approved budget: $10 total, ≤ $2 per job**
(D-019): only E1 plus ≈ 8 single-point seed-0 jobs fit; everything beyond needs a new approval. `vllm-nospec` estimates are ~3× too low (it uses the spec-decode throughput
guess); budget it at ~$9 per seed.

## Experiments

### E0 – CPU harness validation (✅, $0; 51 tests as of checkpoint 2)
- Vendored InferenceX client (checksummed), pinned configs, 24 CPU tests incl. full mock end-to-end.
- Model-availability test passes (both repos ungated, pinned revisions resolvable, weights size matches).
- Prompt-stream fingerprints for all 24 (ISL × conc × seed) points (`configs/fingerprints-rr0.8.json`):
  deterministic and independent of client CPU count.
- Consistency analysis of the article (docs/analysis/consistency.json): fixed 1024-token outputs rejected.
- Roofline (docs/analysis/roofline.md): no-spec single-stream ceiling ≈ 63 tok/s on RTX PRO 6000 ⇒ the
  reported 185–204 tok/s/user implies mean acceptance length ≳ 3.1 tokens per verification step.

### E1 – GPU smoke (🟨 direct Modal, D-039; est. ≈ $1.10)
1. `g4bench modal prefetch --execute` (CPU, ≈ $0.05) – weights to Volume so GPU time is not spent downloading.
2. `g4bench modal job --engine vllm-0.33.0-mtp4 --matrix smoke --execute` (≈ $0.52), then SGLang (≈ $0.53).
- Pass: server healthy, 0 failures, probes ≥ 7/8, acceptance length captured, GPU name/power recorded.
- Also answers: is the client CPU-bound at c=64? (client.log timings vs server metrics).

### E2 – vLLM baseline, seed 0, exact hardware (✅ 8/8 points, n=1, 2026-11-11)
Full 8-point matrix, fresh server per point. First comparison against article vLLM numbers (n=1, so
descriptive only). **Gate**: if any point is > 25 % off, stop and diagnose (acceptance length,
max-model-len default, CUDA-graph capture, attention backend) before E4.

### E3 – SGLang baseline, seed 0 (✅ 8/8 points, n=1, 2026-11-11; effective algorithm FROZEN_KV_MTP)
Same as E2. Record the effective speculative algorithm from logs (community reports NEXTN → FROZEN_KV_MTP).

### E4 – Repeats for uncertainty (⬜, est. $21.07)
Seeds 1,2 for both engines (completing n=3 per point); optionally a second trial (new container) at
1K/c1, 8K/c4, 1K/c64 to separate seed variance from machine variance. Produces the first claims.

### E5 – Profiling & ablations (🟨 2026-11-11: b, e done at 8192×4 and combined in E6; g tooling ready; each sub-item estimated before launch)
Ordered by expected information per dollar; run only those the E2–E4 data justify:
- a. **No-spec control** (`vllm-0.33.0-nospec`, seed 0) – spec-decode gain and the acceptance-free floor.
- b. ✅ (8192×4 only, D-042: depth 6 +8.3 % over depth 4, n=2 each, runs agree within ≈ 1 %; depth 3/5 +5 %, n=1) **Speculative depth sweep** `num_speculative_tokens ∈ {2,3,5,6}` at 1K/c1, 8K/c4, 1K/c64 – the
  article's largest vLLM gap is 8K/c4 (1.34×); depth is the most likely lever. Configs ready:
  `vllm-0.33.0-mtp{2,3,5,6}`, `sglang-0.5.23-nextn{3,4}` (D-034).
- c. **Prompt-content sensitivity** (🟨 CPU side done; GPU ⛔) – random tokens vs real prompts at the same
  engine config, OSL 1024, conc {1,4,16,64}, seeds {0,1,2}:
  - `speedbench-coding`: the 80 InferenceX golden-AL coding prompts, digest-identical to InferenceX
    (`6f184545…`); natural lengths 27–1471 Gemma tokens (mean 251). Est. $4.47 seed 0, both engines.
  - `speedbench-tp`: SPEED-Bench throughput_1k / _8k **inline-only subsets** (897 / 808 rows; no `mixed`
    category); Gemma lengths ≈ 1210 / 9372 tokens. Est. $10.74 seed 0, both engines.
  - Readouts: acceptance length, throughput, p99 TPOT per conc vs the random arm; never compared to the
    article (keys carry `@dataset`). Pre-registered expectation: none — direction unknown a priori.
- d. **Text-only variant** (`vllm-0.33.0-mtp4-textonly`) – skip MM profiling / bounded context.
- e. ✅ (vLLM, 8192×4, n=1; D-043, D-044) **Scheduler/graph knobs** – `max-num-seqs`, cudagraph capture sizes, `max-num-batched-tokens`
  (8K TTFT), SGLang `cuda-graph-max-bs`, `chunked-prefill-size`. 3072-token chunks: +1.4 %, p99 ITL 584 → 274 ms
  (2048 refused by v0.33.0 for Gemma 4); split-K capture sizes: +8.4 %; both combined: 488.3 tok/s (92 % of Backboard).
  SGLang `fair64` / `fast` prepared, not run (D-048).
- f. **Attention backend** (FlashInfer / Triton / FA) on sm_120. Related: sm_120 Triton prefill-tile patch gave no
  gain (D-046, n=1).
- g. 🟨 (tooling ready, D-049; first run E6b at 8192×4 pending) **Torch profiler trace** at 1K/c1 and 8K/c4 to split verify / draft / sampling / CPU overhead.

### E6 – Optimization candidates (🟨 2026-11-11, n=1 descriptive; E4 not run; configs ready: `vllm-0.33.0-mtp-bsched`, `vllm-0.33.0-mtp4-draftfp8` (class b), `sglang-0.5.23-nextn-adaptive`)
- a. ✅ depth by batch size `vllm-0.33.0-dyn-splitk-chunk3072` (D-047): depth 4 helps at 1K/c64 (3674 vs 3603 tok/s),
  depth 5 hurts at 8K/c16 (848 vs 863).
- Draft-prefill patch `mtp_draft_prefill_last` (D-046): 8K/c16 +2.6 %, 8K/c1 mean TTFT −3.6 %; acceptance unchanged.
- b. 🟨 single-config candidate `vllm-0.33.0-best-e6b` (D-047): 1K/c64 3710 tok/s (0.971 of Backboard), 8K/c64 1149
  (0.916); remaining 6 workloads pending.
Candidates justified by E5 profiles, e.g. adaptive speculation depth by batch size, reduced draft
overhead (the 12B draft head is dense, not centroid-sparse: D-054; draft-step CUDA graphs), KV/attention kernel choice for 8K,
and – separately labelled – lower-precision draft shortlisting (SGLang `--speculative-draft-model-quantization`),
the closest open analogue to Backboard's 4-bit shortlist. FP8/NVFP4 *target* weights are a different
model numerically and are reported only in a "not BF16 – not comparable" table.

### E8 – Kernel lab (🟨 2026-11-11; `g4bench modal script`, results in `results/kernels/`, cap $50, D-051; ≈ $24.1 billed)
- GEMM efficiency ✅ (D-052): cold-weight decode linears 15.8 ms/forward at M=28 vs 14.9 ms weight-read floor; prefill
  341–427 TFLOPS at M=3072; max-autotune never wins; CuTe DSL sm_120 GEMM 169.4 vs 168.0 ms/forward (no gain).
- cuBLAS 13.10 vs 13.1 ✅ (D-052, interleaved n=2): −2.0 / −3.1 / −1.9 % at M=28/1024/3072; shipped as an image build step.
- Verify attention ✅ (D-053): shared split-K threshold misroutes the two layer types (8K/c16 10.2 vs 6.9 ms/forward);
  patch `splitk_per_group`; packed-row kernel only ≈ 0.5 ms at 8K/c4, not shipped.
- Draft head ✅ (D-054): dense 262144×1024 BF16 head; FP8 / INT4 scan + BF16 rescore 1.86× / 2.9× at M=28; patch
  `mtp_draft_head_q` (class b).
- E7a third-party drafter DSpark ✅ (D-055): slower than the official MTP (1K/c1 211.0, 8K/c4 443.4 tok/s). DLoop-MTP
  prepared, not run.
- a. ✅ combined config `vllm-0.33.0-best-e8-all` (D-056, D-060): all 8 workloads n=3 (seeds 0/1/2), geomean of means
  1.021; 1K 1.025–1.153 with every 95 % CI above 100 %; 8K 0.890–1.080, CIs span 100 % (GPU-to-GPU spread, throttling).
- Prefill attention kernel ✅ (D-059, `…-prefill_attn_bench-01f4bf`): packed-head FA2-style, 189.8 → 114.0 ms per 8K
  prompt (global 1.40×, sliding 2.15×), within BF16 rounding; patch `prefill_attn_g4` on the experimental image (D-058).
- b. ✅ `vllm-0.33.0-best-e8-pfa` (E8a + `prefill_attn_g4`, D-059), n=1: 8K/c4/c16/c64 525.2 / 986.0 / 1226 tok/s
  (0.992 / 1.025 / 0.978).
- c. 🟨 `vllm-0.33.0-best-e8-pfa-c8k` (E8b + 8192-token chunks, D-059): 8K/c64 n=3 mean 1201.0 (0.958); 8K/c4 / c16
  n=2 means 521.6 / 987.9 (0.985 / 1.027). Not run: 1K workloads, third seed at 8K/c4 and 8K/c16.

- d. ✅ prefill attention v2 (D-064): Gluon `mma_v2` kernels ≈ 190 TFLOPS; per 8K prompt 108.0 → 73.1 ms; config
  `vllm-0.33.0-best-e8-pfa2-c8k`.
- e. ✅ same-GPU paired series (D-061, D-065): `best-e8-pfa2-c8k` vs `mtp4` on one GPU per job, all 8 workloads,
  B/A 1.07–1.33; scaled estimate vs Backboard 0.99–1.14, geometric mean 1.053.

- f. ✅ E9 TTFT (D-067): `--no-async-scheduling` is the lever (−25 % p90 TTFT at 1K×16/64, throughput unchanged);
  config `vllm-0.33.0-best-e8-pfa2-sync` paired at all 8 workloads: throughput geomean 1.053 of Backboard (est.),
  p90 TTFT below Backboard's at 6/8.

**Amendment (2026-11-13) to criterion 5, output preservation (D-063).** Exact greedy token identity against the BF16
no-speculation reference is unattainable for any speculative config in vLLM, the article's unmodified recipe included
(4/8 identical). The criterion applied from now on, identically to the article recipe and every candidate: every first
divergence is a near-tie (reference top-2 within 0.25 nats) and the identical-prefix fraction is not materially below
the article recipe's. This amendment was made after seeing the data and is weaker than the original criterion.

### E7 – Hardware necessity check (⬜, optional, ≤ $10)
Only if E6 finds something hardware-specific. The roofline says H100/H200/B200 would be faster but are
**not** needed for a valid reproduction; L40S cannot hold 64×8K KV at BF16 within 48 GB. RTX PRO 6000
is both the exact target and the cheapest GPU that fits the whole matrix.

## Run log

| Date | Exp | Run ID | Engine | Points | Valid | Notes |
|---|---|---|---|---|---|---|
| 2026-11-10 | E0 | (CPU, mock) | mock-cpu | 4 | 4 | pytest e2e only; numbers meaningless by design |
| 2026-11-10 | E5c-prep | (CPU, data) | – | – | – | SPEED-Bench prepared; coding digest = InferenceX; tp subsets labelled |
| 2026-11-10 | E0 | docs/examples/mock-runs/* | mock-cpu | 4 | 4 | example REPORT.md layout; MOCK, not a measurement |
| 2026-11-11 | E5b | `…-mtp4-…-13d9`, `…-mtp6-…-5c27` | vllm mtp4/mtp6 repeats | 2 (8192×4) | 2 | 402.0 / 434.2 tok/s; depth-6 gain confirmed (≈ +8 %) |
| 2026-11-11 | E5b | `…-mtp3-…-1a1a`, `…-mtp5-…-1982`, `…-mtp6-…-cc24` | vllm mtp3/5/6 | 3 (8192×4) | 3 | 424.4 / 423.9 / 437.4 tok/s vs baseline 402.6; ≈ $1.17 est. |
| 2026-11-11 | E2/E3 | `…-ad4e` (vLLM), `…-3b1e` (SGLang) | both baselines | 2 (8192×64) | 2 | vLLM +1.5 %, SGLang +8.2 % vs article; $3 per-job cap (D-041) |
| 2026-11-11 | E2/E3 | 7 runs `20261111T01…`–`T023134Z…` (see STATUS) | vllm-0.33.0-mtp4, sglang-0.5.23-nextn | 14 (seed 0, OSL 1024, shared server per job) | 14 | vLLM 7/7 within +0.9…+3.1 % of the article; SGLang +6.8…+16.5 %; ≈ $3.3 billed |
| 2026-11-11 | E1 | 20261111T012903Z-vllm-0.33.0-mtp4-rtx-pro-6000-sex1-d6761acd-a22c | vllm-0.33.0-mtp4 | 2 (smoke, OSL 256) | 2 | RTX PRO 6000 Blackwell Server Edition 600 W; probes 8/8; AL 3.07 / 3.20 (c1 / c4); TRITON_ATTN; ≈ $0.34 |
| 2026-11-11 | E5e | `…-mtp6-chunk2048-…-ad8c` | vllm-0.33.0-mtp6-chunk2048 | 1 (8192×4) | 0 | server refused `--max-num-batched-tokens 2048` for Gemma 4 (max_tokens_per_mm_item 2496); point skipped |
| 2026-11-11 | E5e | `…-mtp6-chunk3072-…-1c0b` | vllm-0.33.0-mtp6-chunk3072 | 1 (8192×4) | 1 | 441.7 tok/s (+1.4 % vs depth 6), p99 ITL 584 → 274 ms (D-043) |
| 2026-11-11 | E5e | `…-mtp6-splitk-…-92f4` | vllm-0.33.0-mtp6-splitk | 1 (8192×4) | 1 | 472.4 tok/s (+8.4 % vs depth 6) (D-044) |
| 2026-11-11 | E5e | `…-mtp6-splitk-chunk3072-…-ccdb` | vllm-0.33.0-mtp6-splitk-chunk3072 | 2 (1024×4, 8192×4) | 2 | 765.1 / 488.3 tok/s (0.98 / 0.92 of Backboard) (D-044) |
| 2026-11-11 | E5e | `…-2d15`, `…-d1cf`, `…-aee1` | vllm-0.33.0-mtp6-splitk-chunk3072 | 6 (1024×1/16/64, 8192×1/16/64) | 6 | 223.3 / 2096 / 3603 and 182.3 / 863.1 / 1098 tok/s |
| 2026-11-11 | E6a | `…-dyn-splitk-chunk3072-…-85e7` | vllm-0.33.0-dyn-splitk-chunk3072 | 3 (1024×64, 8192×4, 8192×16) | 3 | 3674 / 481.0 / 848.3 tok/s; depth 4 helps c64, depth 5 hurts c16 (D-047) |
| 2026-11-11 | E5f | `…-tiles-…-110f` | vllm-0.33.0-mtp6-splitk-chunk3072-tiles (patched image) | 2 (8192×1, 8192×16) | 2 | 182.5 / 860.0 tok/s; no gain, patch retired (D-046) |
| 2026-11-11 | E6 | `…-draftlast-…-930d` | vllm-0.33.0-mtp6-splitk-chunk3072-draftlast (patched image) | 2 (8192×1, 8192×16) | 2 | 183.3 / 885.3 tok/s (+2.6 % at c16), 8K/c1 mean TTFT 638 ms (D-046) |
| 2026-11-11 | E6b | `…-best-e6b-…-d129` | vllm-0.33.0-best-e6b (patched image) | 2 (1024×64, 8192×64) | 2 | 3710 / 1149 tok/s (0.971 / 0.916 of Backboard) (D-047) |
| 2026-11-11 | E8 | `20261111T170723Z-gemm_bench-c46eac` | kernels/gemm_bench.py | – | – | decode linears below the weight floor (11.2 ms at M=28): L2-hot artefact, withdrawn; ≈ $0.12 |
| 2026-11-11 | E8 | `20261111T171035Z-gemm_bench-025434` | kernels/gemm_bench.py (cold weights) | – | – | 15.8 ms/forward at M=28 vs 14.9 ms floor; prefill 341–427 TFLOPS (D-052); ≈ $0.08 |
| 2026-11-11 | E8 | `20261111T171428Z-cublas_upgrade_bench-7d63ed` | kernels/cublas_upgrade_bench.py | – | – | first 13.10 vs 13.1 comparison, not interleaved; superseded; ≈ $0.09 |
| 2026-11-11 | E8 | `20261111T171610Z-cublas_upgrade_bench-af4bfc` | kernels/cublas_upgrade_bench.py | – | – | interleaved n=2: −2.0 / −3.1 / −1.9 % at M=28/1024/3072 (D-052); ≈ $0.09 |
| 2026-11-11 | E8 | `20261111T172120Z-bench-916b9a` | kernels/cute_gemm/bench.py (CuTe DSL 4.8.0) | – | – | 169.4 vs 168.0 ms/forward at M=3072, no gain (D-052); ≈ $0.09 |
| 2026-11-11 | E8 | `20261111T172419Z-draft_head_scan-556087` | kernels/draft_head_scan.py | – | – | BF16 395 µs → FP8 212 / INT4 137 µs at M=28, k=32; recall FP8 0.996, INT4 0.981 (D-054); ≈ $0.11 |
| 2026-11-11 | E8 | `20261111T172659Z-attn_bench-c6c4db` | kernels/attn_bench.py | – | – | 8K/c16 10.2 vs 6.9 ms/forward with per-group split-K; packed-row ≈ 0.5 ms at 8K/c4 (D-053); ≈ $0.10 |
| 2026-11-11 | E7a | `…-dspark-block7-…-cc48` | vllm-0.33.0-dspark-block7 | 2 (1024×1, 8192×4) | 2 | 211.0 / 443.4 tok/s vs E6b 223.9 / 497.8; AL 5.06 / 4.55; probes 8/8 (D-055); ≈ $0.42 billed |
| 2026-11-11 | E8a | `…-best-e8-all-…-4b53` | vllm-0.33.0-best-e8-all (patched image) | 2 (8192×4, 8192×16) | 0 | server exited during speculator CUDA-graph capture: all-NaN draft scores → sentinel token; fixed in `ab3fc93` (D-054) |
| 2026-11-11 | E8a | `…-best-e8-all-…-0b3c` | vllm-0.33.0-best-e8-all (patched image) | 2 (8192×4, 8192×16) | 2 | 528.0 / 945.8 tok/s (0.997 / 0.983 of Backboard); AL 5.25 / 5.09 (D-056) |
| 2026-11-11 | E8a | `…-best-e8-all-…-d1a0` | vllm-0.33.0-best-e8-all (patched image) | 4 (1024×1/4/16, 8192×1) | 4 | 238.1 / 813.1 / 2235 / 197.5 tok/s (1.17 / 1.04 / 1.07 / 1.10) (D-056) |
| 2026-11-11 | E8a | `…-best-e8-all-…-400b` | vllm-0.33.0-best-e8-all (patched image) | 1 (1024×64) | 1 | 3916 tok/s (1.02) (D-056) |
| 2026-11-11 | E8a | `…-best-e8-all-…-e87a` | vllm-0.33.0-best-e8-all (patched image) | 1 (8192×64) | 1 | 1170 tok/s (0.933); prefill-bound; geomean over 8 workloads 1.039, n=1 (D-056) |
| 2026-11-11 | E8 | `20261111T202443Z-prefill_attn_bench-01f4bf` | kernels/prefill_attn.py | – | – | 189.8 → 114.0 ms per 8K prompt; global 1.40×, sliding 2.15×; max abs vs vLLM 0.016 (D-059) |
| 2026-11-11 | E8a | `…-best-e8-all-…-a764` | vllm-0.33.0-best-e8-all (patched image) | 2 (8192×4, 8192×16), seed 1 | 2 | 488.4 / 948.5 tok/s (D-060) |
| 2026-11-11 | E8a | `…-best-e8-all-…-1628` | vllm-0.33.0-best-e8-all (patched image) | 1 (8192×64), seed 1 | 1 | 1083.9 tok/s; GPU 83 °C, SM 2220 MHz mean (throttled) (D-060) |
| 2026-11-11 | E8a | `…-best-e8-all-…-ca1f` | vllm-0.33.0-best-e8-all (patched image) | 4 (1024×1/4/16, 8192×1), seed 1 | 4 | 233.6 / 814.1 / 2178 / 184.6 tok/s (D-060) |
| 2026-11-11 | E8a | `…-best-e8-all-…-9efd` | vllm-0.33.0-best-e8-all (patched image) | 1 (1024×64), seed 1 | 1 | 3906 tok/s (D-060) |
| 2026-11-11 | E8b | `…-best-e8-pfa-…-e5f7` | vllm-0.33.0-best-e8-pfa (exp. image) | 1 (8192×64) | 1 | 1226 tok/s (0.978); AL 4.09 (D-059) |
| 2026-11-11 | E8a | `…-best-e8-all-…-6106` | vllm-0.33.0-best-e8-all (patched image) | 2 (8192×4, 8192×16), seed 2 | 2 | 507.3 / 928.7 tok/s (D-060) |
| 2026-11-11 | E8a | `…-best-e8-all-…-416c` | vllm-0.33.0-best-e8-all (patched image) | 1 (8192×64), seed 2 | 1 | 1092.6 tok/s; GPU 83 °C, SM 2217 MHz mean (throttled) (D-060) |
| 2026-11-11 | E8b | `…-best-e8-pfa-…-0460` | vllm-0.33.0-best-e8-pfa (exp. image) | 2 (8192×4, 8192×16) | 2 | 525.2 / 986.0 tok/s (0.992 / 1.025); AL 5.03 / 5.09 (D-059) |
| 2026-11-11 | E8c | `…-best-e8-pfa-c8k-…-6959` | vllm-0.33.0-best-e8-pfa-c8k (exp. image) | 1 (8192×64) | 1 | 1243.4 tok/s (0.992) (D-059) |
| 2026-11-11 | E8a | `…-best-e8-all-…-2ef5` | vllm-0.33.0-best-e8-all (patched image) | 4 (1024×1/4/16, 8192×1), seed 2 | 4 | 232.7 / 818.5 / 2207 / 198.3 tok/s (D-060) |
| 2026-11-11 | E8a | `…-best-e8-all-…-4654` | vllm-0.33.0-best-e8-all (patched image) | 1 (1024×64), seed 2 | 1 | 3929 tok/s; E8a n=3 complete, geomean of means 1.021 (D-060) |
| 2026-11-11 | E8c | `…-best-e8-pfa-c8k-…-c061` | vllm-0.33.0-best-e8-pfa-c8k (exp. image) | 1 (8192×64), seed 1 | 1 | 1219.0 tok/s (D-059) |
| 2026-11-11 | E8c | `…-best-e8-pfa-c8k-…-deb6` | vllm-0.33.0-best-e8-pfa-c8k (exp. image) | 2 (8192×4, 8192×16), seed 1 | 2 | 508.0 / 1000.0 tok/s (D-059) |
| 2026-11-11 | E8c | `…-best-e8-pfa-c8k-…-47e9` | vllm-0.33.0-best-e8-pfa-c8k (exp. image) | 1 (8192×64), seed 2 | 1 | 1140.6 tok/s; GPU 83 °C, SM 2199 MHz mean (throttled); n=3 mean 1201.0 (0.958) (D-059, D-060) |
| 2026-11-11 | E8c | `…-best-e8-pfa-c8k-…-684f` | vllm-0.33.0-best-e8-pfa-c8k (exp. image) | 2 (8192×4, 8192×16), seed 2 | 2 | 535.3 / 975.7 tok/s (D-059) |
