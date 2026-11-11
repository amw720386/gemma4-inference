# Run report: `20261111T014900Z-vllm-0.33.0-mtp4-rtx-pro-6000-sex1-28818286-7226`

## Identity

- engine config: `vllm-0.33.0-mtp4` (vllm 0.33.0)
- config hash: `288182863c19`
- hardware: `RTX-PRO-6000-SEx1`; exact article hardware: **True**
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition / power limit 600.00 W / driver 610.57.04
- git: `8b11b0be166535ed92ccecab4ce123c103ef33a5` dirty=False
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
| random | 1024 | 1 | 1 | 190.4 | n<2 | 193.8 | 190.5 (+0.0%) |
| random | 1024 | 4 | 1 | 655 | n<2 | 171 | 667.9 (+2.0%) |
| random | 1024 | 16 | 1 | 1905 | n<2 | 125.8 | 1956 (+2.7%) |

![throughput vs concurrency](throughput.svg)

## Latency (ms, mean over runs of each run's statistic)

| dataset | ISL | conc | TTFT mean | TTFT p90 | TTFT p99 | TPOT mean | TPOT p99 | E2EL mean | E2EL p99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random | 1024 | 1 | 89.84 | 100.3 | 100.7 | 5.16 | 5.406 | 4838 | 5454 |
| random | 1024 | 4 | 139.3 | 170.9 | 296 | 5.849 | 7.333 | 5496 | 6917 |
| random | 1024 | 16 | 219.5 | 242.9 | 1067 | 7.951 | 9.196 | 7499 | 9122 |

## Speculative decoding acceptance & resources

| dataset | ISL | conc | acceptance length | peak GPU MiB | mean GPU W |
|---|---:|---:|---:|---:|---:|
| random | 1024 | 1 | 4.213 | 85914 | 322 |
| random | 1024 | 4 | 4.154 | 86266 | 344 |
| random | 1024 | 16 | 4.156 | 87226 | 418 |

## Effective engine configuration (parsed from server logs)

| attention backend | spec algorithm | KV-cache tokens | max running requests | seen in |
|---|---|---:|---:|---|
| AttentionBackendEnum.TRITON_ATTN | not measured | 1,568,055 | not measured | 3 point-run(s) |

## Comparison with article baselines

Config role: **baseline**, admissibility class **a** (DECISIONS D-011).
Reference: article `vllm` (and Backboard) numbers for random-token prompts. Classification uses the pre-registered 5 % tolerance (EXPERIMENTS.md); n<3 results are descriptive only.

| ISL | conc | ours | n | article vllm | ours/article | class | article Backboard | ours/Backboard |
|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 1024 | 1 | 190.4 | 1 | 186.8 | 1.019 | consistent | 203.6 | 0.935 |
| 1024 | 4 | 655 | 1 | 649.0 | 1.009 | consistent | 779.4 | 0.840 |
| 1024 | 16 | 1905 | 1 | 1861 | 1.024 | consistent | 2082 | 0.915 |

## Gap attribution (first-order; acceptance vs step speed)

Assumptions: first-order: TPOT = step/AL; mean TPOT; AL incl. bonus token; batch = conc; ctx = ISL + OSL/2; roofline at 85% of peak HBM/GDDR bandwidth. Article mean TPOT = 1000 / its 'speed per user'.

| ISL | conc | our TPOT ms | our AL | step ms | BW eff. | AL needed for article vllm | for Backboard | step speed-up needed (Backboard) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 1 | 5.16 | 4.21 | 21.7 | 0.77 | 4.10 | 4.44 | 1.05x |
| 1024 | 4 | 5.85 | 4.15 | 24.3 | 0.73 | 4.12 | 4.96 | 1.19x |
| 1024 | 16 | 7.95 | 4.16 | 33 | 0.65 | 4.08 | 4.53 | 1.09x |

## Correctness gate

Known-answer probes: 8/8 — PASS; greedy outputs captured: 8

## Invalid / skipped point-runs

None.
