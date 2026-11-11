# Run report: `20261111T015610Z-vllm-0.33.0-mtp4-rtx-pro-6000-sex1-28818286-9527`

## Identity

- engine config: `vllm-0.33.0-mtp4` (vllm 0.33.0)
- config hash: `288182863c19`
- hardware: `RTX-PRO-6000-SEx1`; exact article hardware: **True**
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition / power limit 600.00 W / driver 610.57.04
- git: `8b11b0be166535ed92ccecab4ce123c103ef33a5` dirty=True
- point-runs: 3 total, 3 valid

<details><summary>server argv</summary>

```
vllm serve google/gemma-4-12B-it --revision 43c9d014b231474fa7212dbb48a762566c2b255d --tensor-parallel-size 1 --dtype bfloat16 --gpu-memory-utilization 0.90 --no-enable-prefix-caching --async-scheduling --speculative-config {"method":"mtp","model":"google/gemma-4-12B-it-assistant","revision":"3b0b0a4a3b31b5556516329227fb55f275c23c9f","num_speculative_tokens":4} --host 0.0.0.0 --port 8000
```
</details>

## Throughput vs concurrency

Mean over valid seeds/trials in this run; CI = Student-t 95 % across runs (needs n ≥ 2). Little's-law column is a within-run diagnostic from per-request logs.

| dataset | ISL | conc | n | output tok/s | 95% CI | tok/s/user | Little's-law tok/s (gap) |
|---|---:|---:|---:|---:|---|---:|---|
| random | 1024 | 64 | 1 | 3668 | n<2 | 60.65 | 3754 (+2.3%) |
| random | 8192 | 1 | 1 | 164.6 | n<2 | 184.9 | 164.6 (+0.0%) |
| random | 8192 | 4 | 1 | 402.6 | n<2 | 111.7 | 409.1 (+1.6%) |

![throughput vs concurrency](throughput.svg)

## Latency (ms, mean over runs of each run's statistic)

| dataset | ISL | conc | TTFT mean | TTFT p90 | TTFT p99 | TPOT mean | TPOT p99 | E2EL mean | E2EL p99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random | 1024 | 64 | 529.4 | 632.5 | 4120 | 16.49 | 19.88 | 1.571e+04 | 2.051e+04 |
| random | 8192 | 1 | 624.7 | 656.6 | 685.4 | 5.407 | 6.606 | 5651 | 6935 |
| random | 8192 | 4 | 767 | 827.2 | 2244 | 8.955 | 12.99 | 8998 | 1.297e+04 |

## Speculative decoding acceptance & resources

| dataset | ISL | conc | acceptance length | peak GPU MiB | mean GPU W |
|---|---:|---:|---:|---:|---:|
| random | 1024 | 64 | 4.191 | 87097 | 517 |
| random | 8192 | 1 | 4.101 | 87097 | 362 |
| random | 8192 | 4 | 4.064 | 87097 | 389 |

## Effective engine configuration (parsed from server logs)

| attention backend | spec algorithm | KV-cache tokens | max running requests | seen in |
|---|---|---:|---:|---|
| AttentionBackendEnum.TRITON_ATTN | not measured | 1,568,055 | not measured | 3 point-run(s) |

## Comparison with article baselines

Config role: **baseline**, admissibility class **a** (DECISIONS D-011).
Reference: article `vllm` (and Backboard) numbers for random-token prompts. Classification uses the pre-registered 5 % tolerance (EXPERIMENTS.md); n<3 results are descriptive only.

| ISL | conc | ours | n | article vllm | ours/article | class | article Backboard | ours/Backboard |
|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 1024 | 64 | 3668 | 1 | 3634 | 1.009 | consistent | 3822 | 0.960 |
| 8192 | 1 | 164.6 | 1 | 159.6 | 1.031 | consistent | 179.1 | 0.919 |
| 8192 | 4 | 402.6 | 1 | 395.7 | 1.018 | consistent | 529.6 | 0.760 |

## Gap attribution (first-order; acceptance vs step speed)

Assumptions: first-order: TPOT = step/AL; mean TPOT; AL incl. bonus token; batch = conc; ctx = ISL + OSL/2; roofline at 85% of peak HBM/GDDR bandwidth. Article mean TPOT = 1000 / its 'speed per user'.

| ISL | conc | our TPOT ms | our AL | step ms | BW eff. | AL needed for article vllm | for Backboard | step speed-up needed (Backboard) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 64 | 16.5 | 4.19 | 69.1 | 0.52 | 4.11 | 4.35 | 1.04x |
| 8192 | 1 | 5.41 | 4.1 | 22.2 | 0.76 | 4.11 | 4.53 | 1.10x |
| 8192 | 4 | 8.96 | 4.06 | 36.4 | 0.49 | 4.18 | 5.78 (> max AL) | 1.42x |

## Correctness gate

Known-answer probes: 8/8 — PASS; greedy outputs captured: 8

## Invalid / skipped point-runs

None.
