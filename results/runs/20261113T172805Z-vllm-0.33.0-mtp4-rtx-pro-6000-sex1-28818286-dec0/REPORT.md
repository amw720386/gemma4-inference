# Run report: `20261113T172805Z-vllm-0.33.0-mtp4-rtx-pro-6000-sex1-28818286-dec0`

> **NOT MEASURED:** this run has no valid point-runs.

## Identity

- engine config: `vllm-0.33.0-mtp4` (vllm 0.33.0)
- config hash: `288182863c19`
- hardware: `RTX-PRO-6000-SEx1`; exact article hardware: **True**
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition / power limit 600.00 W / driver 610.57.04
- git: `7ba7edeaccfa22cce7aa0ff3de88c2a087b143ac` dirty=False
- point-runs: 0 total, 0 valid

<details><summary>server argv</summary>

```
vllm serve google/gemma-4-12B-it --revision 43c9d014b231474fa7212dbb48a762566c2b255d --tensor-parallel-size 1 --dtype bfloat16 --gpu-memory-utilization 0.90 --no-enable-prefix-caching --async-scheduling --speculative-config {"method":"mtp","model":"google/gemma-4-12B-it-assistant","revision":"3b0b0a4a3b31b5556516329227fb55f275c23c9f","num_speculative_tokens":4} --host 0.0.0.0 --port 8000
```
</details>

## Throughput vs concurrency

Mean over valid seeds/trials in this run; CI = Student-t 95 % across runs (needs n ≥ 2). Little's-law column is a within-run diagnostic from per-request logs.

| dataset | ISL | conc | n | output tok/s | 95% CI | tok/s/user | Little's-law tok/s (gap) |
|---|---:|---:|---:|---:|---|---:|---|
| – | – | – | 0 | not measured | – | not measured | not measured |

![throughput vs concurrency](throughput.svg)

## Latency (ms, mean over runs of each run's statistic)

| dataset | ISL | conc | TTFT mean | TTFT p90 | TTFT p99 | TPOT mean | TPOT p99 | E2EL mean | E2EL p99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|

## Speculative decoding acceptance & resources

| dataset | ISL | conc | acceptance length | peak GPU MiB | mean GPU W |
|---|---:|---:|---:|---:|---:|

## Effective engine configuration (parsed from server logs)

| attention backend | spec algorithm | KV-cache tokens | max running requests | seen in |
|---|---|---:|---:|---|
| not measured | not measured | not measured | not measured | – |

## Comparison with article baselines

Config role: **baseline**, admissibility class **a** (DECISIONS D-011).
Reference: article `vllm` (and Backboard) numbers for random-token prompts. Classification uses the pre-registered 5 % tolerance (docs/EXPERIMENTS.md); n<3 results are descriptive only.

| ISL | conc | ours | n | article vllm | ours/article | class | article Backboard | ours/Backboard |
|---:|---:|---:|---:|---:|---:|---|---:|---:|
| – | – | not measured | 0 | – | – | – | – | – |

## Gap attribution (first-order; acceptance vs step speed)

Assumptions: first-order: TPOT = step/AL; mean TPOT; AL incl. bonus token; batch = conc; ctx = ISL + OSL/2; roofline at 85% of peak HBM/GDDR bandwidth. Article mean TPOT = 1000 / its 'speed per user'.

| ISL | conc | our TPOT ms | our AL | step ms | BW eff. | AL needed for article vllm | for Backboard | step speed-up needed (Backboard) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| – | – | not measured | not measured | – | – | – | – | – |

## Correctness gate

Known-answer probes: 8/8 — PASS
Greedy capture: 8 prompts (set b17f5663089c, 256 max tokens, 14.79 s)
greedy equivalence: FAIL 4/8 identical; 4 divergent (4 near-tie <= 0.25 nats, 0 not, 0 unknown), 0 length, 0 error; identical-prefix tokens 0.7174

## Invalid / skipped point-runs

None.
