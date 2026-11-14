# Run report: `20261113T212447Z-vllm-0.33.0-best-e8-pfa2-sync-rtx-pro-6000-sex1-8229db0c-7610`

> **PROFILED RUN (torch profiler enabled).** Benchmark numbers include profiler overhead; see `points/*/profile_summary.md`. Not compared with the article.

## Identity

- engine config: `vllm-0.33.0-best-e8-pfa2-sync` (vllm 0.33.0)
- config hash: `8229db0c32b0`
- hardware: `RTX-PRO-6000-SEx1`; exact article hardware: **True**
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition / power limit 600.00 W / driver 610.57.04
- git: `9de00c4fe32822d04753555849afa4fe18464d7d` dirty=True
- point-runs: 1 total, 1 valid

<details><summary>server argv</summary>

```
vllm serve google/gemma-4-12B-it --revision 43c9d014b231474fa7212dbb48a762566c2b255d --tensor-parallel-size 1 --dtype bfloat16 --gpu-memory-utilization 0.90 --no-enable-prefix-caching --no-async-scheduling --speculative-config {"method":"mtp","model":"google/gemma-4-12B-it-assistant","revision":"3b0b0a4a3b31b5556516329227fb55f275c23c9f","num_speculative_tokens":6,"num_speculative_tokens_per_batch_size":[[1,32,6],[33,4096,4]],"use_local_argmax_reduction":true} --max-num-batched-tokens 8192 --host 0.0.0.0 --port 8000 --profiler-config {"profiler":"torch","torch_profiler_dir":"/__modal/volumes/vo-vj6F1MxmD3X0PsdDlgXrq0/runs/20261113T212447Z-vllm-0.33.0-best-e8-pfa2-sync-rtx-pro-6000-sex1-8229db0c-7610/points/isl1024-osl1024-c1-s0-t0/torch_trace","torch_profiler_activities":["CPU","CUDA"],"torch_profiler_with_stack":false,"torch_profiler_use_gzip":true,"torch_profiler_dump_cuda_time_total":false,"ignore_frontend":true}
```
</details>

## Throughput vs concurrency

Mean over valid seeds/trials in this run; CI = Student-t 95 % across runs (needs n ≥ 2). Little's-law column is a within-run diagnostic from per-request logs.

| dataset | ISL | conc | n | output tok/s | 95% CI | tok/s/user | Little's-law tok/s (gap) |
|---|---:|---:|---:|---:|---|---:|---|
| random | 1024 | 1 | 1 | 232.1 | n<2 | 237 | 232.2 (+0.0%) |

![throughput vs concurrency](throughput.svg)

## Latency (ms, mean over runs of each run's statistic)

| dataset | ISL | conc | TTFT mean | TTFT p90 | TTFT p99 | TPOT mean | TPOT p99 | E2EL mean | E2EL p99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random | 1024 | 1 | 77.72 | 87.07 | 89.56 | 4.22 | 4.78 | 3969 | 4894 |

## Speculative decoding acceptance & resources

| dataset | ISL | conc | acceptance length | peak GPU MiB | mean GPU W |
|---|---:|---:|---:|---:|---:|
| random | 1024 | 1 | 5.268 | 89243 | 280 |

## Effective engine configuration (parsed from server logs)

| attention backend | spec algorithm | KV-cache tokens | max running requests | seen in |
|---|---|---:|---:|---|
| AttentionBackendEnum.TRITON_ATTN | not measured | 2,136,054 | not measured | 1 point-run(s) |

## Comparison with article baselines

Config role: **variant**, admissibility class **b** (DECISIONS D-011). Variants are not compared with the article's open baselines.
Not compared: profiled run (torch profiler overhead).

## Gap attribution (first-order; acceptance vs step speed)

Not computed: needs exact article hardware and a vLLM/SGLang run.

## Correctness gate

Known-answer probes: 8/8 — PASS
Greedy equivalence: not measured

## Invalid / skipped point-runs

None.
