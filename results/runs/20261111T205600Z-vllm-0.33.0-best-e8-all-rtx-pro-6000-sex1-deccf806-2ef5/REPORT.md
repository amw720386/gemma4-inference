# Run report: `20261111T205600Z-vllm-0.33.0-best-e8-all-rtx-pro-6000-sex1-deccf806-2ef5`

## Identity

- engine config: `vllm-0.33.0-best-e8-all` (vllm 0.33.0)
- config hash: `deccf80607ad`
- hardware: `RTX-PRO-6000-SEx1`; exact article hardware: **True**
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition / power limit 600.00 W / driver 610.57.04
- git: `None` dirty=None
- point-runs: 4 total, 4 valid

<details><summary>server argv</summary>

```
vllm serve google/gemma-4-12B-it --revision 43c9d014b231474fa7212dbb48a762566c2b255d --tensor-parallel-size 1 --dtype bfloat16 --gpu-memory-utilization 0.90 --no-enable-prefix-caching --async-scheduling --speculative-config {"method":"mtp","model":"google/gemma-4-12B-it-assistant","revision":"3b0b0a4a3b31b5556516329227fb55f275c23c9f","num_speculative_tokens":6,"num_speculative_tokens_per_batch_size":[[1,32,6],[33,4096,4]],"use_local_argmax_reduction":true} --max-num-batched-tokens 3072 --host 0.0.0.0 --port 8000
```
</details>

## Throughput vs concurrency

Mean over valid seeds/trials in this run; CI = Student-t 95 % across runs (needs n ≥ 2). Little's-law column is a within-run diagnostic from per-request logs.

| dataset | ISL | conc | n | output tok/s | 95% CI | tok/s/user | Little's-law tok/s (gap) |
|---|---:|---:|---:|---:|---|---:|---|
| random | 1024 | 1 | 1 | 232.7 | n<2 | 237.6 | 232.8 (+0.0%) |
| random | 1024 | 4 | 1 | 818.5 | n<2 | 217.3 | 844.1 (+3.1%) |
| random | 1024 | 16 | 1 | 2207 | n<2 | 147.5 | 2289 (+3.7%) |
| random | 8192 | 1 | 1 | 198.3 | n<2 | 228.2 | 198.4 (+0.0%) |

![throughput vs concurrency](throughput.svg)

## Latency (ms, mean over runs of each run's statistic)

| dataset | ISL | conc | TTFT mean | TTFT p90 | TTFT p99 | TPOT mean | TPOT p99 | E2EL mean | E2EL p99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random | 1024 | 1 | 87.26 | 97.84 | 97.98 | 4.209 | 4.972 | 3906 | 4711 |
| random | 1024 | 4 | 132 | 172.8 | 252.6 | 4.602 | 5.182 | 4359 | 5279 |
| random | 1024 | 16 | 214.9 | 308 | 1032 | 6.778 | 8.874 | 6421 | 8380 |
| random | 8192 | 1 | 636.1 | 703.1 | 703.8 | 4.381 | 7.756 | 4652 | 7442 |

## Speculative decoding acceptance & resources

| dataset | ISL | conc | acceptance length | peak GPU MiB | mean GPU W |
|---|---:|---:|---:|---:|---:|
| random | 1024 | 1 | 5.139 | 86128 | 277 |
| random | 1024 | 4 | 5.276 | 86572 | 326 |
| random | 1024 | 16 | 5.112 | 86572 | 416 |
| random | 8192 | 1 | 5.116 | 86572 | 313 |

## Effective engine configuration (parsed from server logs)

| attention backend | spec algorithm | KV-cache tokens | max running requests | seen in |
|---|---|---:|---:|---|
| AttentionBackendEnum.TRITON_ATTN | not measured | 2,357,394 | not measured | 4 point-run(s) |

## Comparison with article baselines

Config role: **variant**, admissibility class **b** (DECISIONS D-011). Variants are not compared with the article's open baselines.
Variant run: see `g4bench analyze` (results/summary/comparison.csv) for the comparison with Backboard's reported numbers, labelled by admissibility class.

## Gap attribution (first-order; acceptance vs step speed)

Not computed: needs exact article hardware and a vLLM/SGLang run.

## Correctness gate

Known-answer probes: 8/8 — PASS; greedy outputs captured: 8

## Invalid / skipped point-runs

None.
