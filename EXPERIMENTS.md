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
overhead (centroid head already sparse; draft-step CUDA graphs), KV/attention kernel choice for 8K,
and – separately labelled – lower-precision draft shortlisting (SGLang `--speculative-draft-model-quantization`),
the closest open analogue to Backboard's 4-bit shortlist. FP8/NVFP4 *target* weights are a different
model numerically and are reported only in a "not BF16 – not comparable" table.

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
