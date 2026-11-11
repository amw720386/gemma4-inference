# Run report: `20261111T022301Z-sglang-0.5.23-nextn-rtx-pro-6000-sex1-7cbf3a7a-cbaf`

## Identity

- engine config: `sglang-0.5.23-nextn` (sglang 0.5.23)
- config hash: `7cbf3a7a62fe`
- hardware: `RTX-PRO-6000-SEx1`; exact article hardware: **True**
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition / power limit 600.00 W / driver 610.57.04
- git: `8b11b0be166535ed92ccecab4ce123c103ef33a5` dirty=True
- point-runs: 1 total, 1 valid

<details><summary>server argv</summary>

```
python3 -m sglang.launch_server --model-path google/gemma-4-12B-it --revision 43c9d014b231474fa7212dbb48a762566c2b255d --tp-size 1 --dtype bfloat16 --speculative-algorithm NEXTN --speculative-draft-model-path google/gemma-4-12B-it-assistant --speculative-draft-model-revision 3b0b0a4a3b31b5556516329227fb55f275c23c9f --speculative-num-steps 5 --speculative-num-draft-tokens 6 --speculative-eagle-topk 1 --mem-fraction-static 0.85 --disable-radix-cache --host 0.0.0.0 --port 30000
```
</details>

## Throughput vs concurrency

Mean over valid seeds/trials in this run; CI = Student-t 95 % across runs (needs n ≥ 2). Little's-law column is a within-run diagnostic from per-request logs.

| dataset | ISL | conc | n | output tok/s | 95% CI | tok/s/user | Little's-law tok/s (gap) |
|---|---:|---:|---:|---:|---|---:|---|
| random | 1024 | 64 | 1 | 2039 | n<2 | 44.72 | 2130 (+4.4%) |

![throughput vs concurrency](throughput.svg)

## Latency (ms, mean over runs of each run's statistic)

| dataset | ISL | conc | TTFT mean | TTFT p90 | TTFT p99 | TPOT mean | TPOT p99 | E2EL mean | E2EL p99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random | 1024 | 64 | 7114 | 1.351e+04 | 2.008e+04 | 22.36 | 26.93 | 2.77e+04 | 4.213e+04 |

## Speculative decoding acceptance & resources

| dataset | ISL | conc | acceptance length | peak GPU MiB | mean GPU W |
|---|---:|---:|---:|---:|---:|
| random | 1024 | 64 | 4.728 | 78155 | 458 |

## Effective engine configuration (parsed from server logs)

| attention backend | spec algorithm | KV-cache tokens | max running requests | seen in |
|---|---|---:|---:|---|
| triton | FROZEN_KV_MTP | 1,074,431 | 48 | 1 point-run(s) |

## Comparison with article baselines

Config role: **baseline**, admissibility class **a** (DECISIONS D-011).
Reference: article `sglang` (and Backboard) numbers for random-token prompts. Classification uses the pre-registered 5 % tolerance (EXPERIMENTS.md); n<3 results are descriptive only.

| ISL | conc | ours | n | article sglang | ours/article | class | article Backboard | ours/Backboard |
|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 1024 | 64 | 2039 | 1 | 1750 | 1.165 | higher | 3822 | 0.534 |

## Gap attribution (first-order; acceptance vs step speed)

Assumptions: first-order: TPOT = step/AL; mean TPOT; AL incl. bonus token; batch = conc; ctx = ISL + OSL/2; roofline at 85% of peak HBM/GDDR bandwidth. Article mean TPOT = 1000 / its 'speed per user'.

| ISL | conc | our TPOT ms | our AL | step ms | BW eff. | AL needed for article sglang | for Backboard | step speed-up needed (Backboard) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 64 | 22.4 | 4.73 | 106 | 0.35 | 2.99 | 6.65 (> max AL) | 1.41x |

## Correctness gate

Known-answer probes: 8/8 — PASS; greedy outputs captured: 8

## Invalid / skipped point-runs

None.
