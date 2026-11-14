# gemma4-inference-repro

**Stock vLLM 0.33.0, a tuned config and five patches on one GPU: faster than Backboard's proprietary engine on
Gemma 4 12B.**

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![vLLM 0.33.0](https://img.shields.io/badge/vLLM-0.33.0-30a2ff.svg)](https://github.com/vllm-project/vllm/releases/tag/v0.33.0)
[![GPU: RTX PRO 6000 Blackwell](https://img.shields.io/badge/GPU-RTX%20PRO%206000%20Blackwell-76b900.svg)](#setup)

[Results](#results) · [What we changed](#what-we-changed) · [Reproduce](#reproduce) · [How we measured](#how-we-measured) · [Decision log](docs/DECISIONS.md)

![Throughput and p90 TTFT speed-up over Backboard, per workload](docs/results-vs-backboard.svg)

- **+5.4% throughput vs Backboard** (geometric mean over 8 workloads), up to **+14%** at concurrency 1.
- **Lower p90 time-to-first-token at 7 of 8 workloads.**
- **1.07–1.33× the article's own vLLM recipe**, side by side on the same GPU.
- **Same model, same precision:** BF16 target, outputs checked against a no-speculation BF16 reference.

Backboard's [Gemma 4 12B benchmark](https://backboard.io/blog/gemma-4-12b-vllm-sglang-inference) (2026-11-09)
showed its engine beating vLLM and SGLang everywhere. We reproduced the article's vLLM and SGLang baselines, then
tuned and patched stock vLLM until it beat Backboard's own numbers.

## Results

One frozen config, [`vllm-0.33.0-best-e8-pfa2-sync-hybtok`](configs/engines/vllm-0.33.0-best-e8-pfa2-sync-hybtok.yaml),
for all eight workloads.

| Workload<br><sub>input × conc.</sub> | Backboard<br><sub>tok/s</sub> | Ours¹<br><sub>tok/s</sub> | Throughput<br><sub>speed-up</sub> | Backboard<br><sub>p90 TTFT ms</sub> | Ours<br><sub>p90 TTFT ms</sub> | TTFT<br><sub>speed-up</sub> |
|---|---:|---:|---:|---:|---:|---:|
| 1K × 1  | 204  | 231  | **1.14×** | 85   | 81   | **1.05×** |
| 1K × 4  | 779  | 816  | **1.05×** | 176  | 152  | **1.16×** |
| 1K × 16 | 2082 | 2214 | **1.06×** | 245  | 218  | **1.12×** |
| 1K × 64 | 3822 | 3901 | **1.02×** | 547  | 510  | **1.07×** |
| 8K × 1  | 179  | 203  | **1.14×** | 549  | 552  | 0.99×     |
| 8K × 4  | 530  | 528  | 1.00×     | 724  | 656  | **1.10×** |
| 8K × 16 | 962  | 999  | **1.04×** | 1537 | 1327 | **1.16×** |
| 8K × 64 | 1254 | 1256 | 1.00×     | 3102 | 3005 | **1.03×** |

Gemma 4 12B-it BF16 + MTP draft, 1024 output tokens, InferenceX client, as in the article. p99 TPOT and p99
end-to-end latency per workload: [`results/summary/SUMMARY.md`](results/summary/SUMMARY.md).

¹ Same-GPU speed-up over the article's vLLM recipe × the article's published vLLM figure; see [Notes](#notes).

## What we changed

Configuration alone reaches 97.7% of Backboard ([config only, no patches](configs/engines/vllm-0.33.0-best-e6b.yaml)).
The patches live in [`src/g4bench/patches/`](src/g4bench/patches/), are applied at image build and each sits behind
its own switch.

| Change | Kind | Effect |
|---|---|---|
| [Custom prefill attention](src/g4bench/patches/prefill_attn_g4.py): a Gluon FlashAttention-2 kernel for sm_120 that packs the 16 query heads sharing each KV head | patch | 8K prefill attention 191 → 73 ms; +4–5% at 8K × 16 / 64 |
| Synchronous scheduling (`--no-async-scheduling`): new requests start one step sooner | config | p90 TTFT −27% at 1K × 16, −25% at 1K × 64 |
| Split-K decode attention for the head-dim-512 layers, [thresholds per layer group](src/g4bench/patches/splitk_per_group.py) | patch | +8.4% at 8K × 4 |
| Deeper speculation: 6 draft tokens up to 32 running requests, 4 above | config | +8% at 8K × 4, +3% at 1K × 64 |
| [FP8 draft-head scan with exact BF16 rescoring](src/g4bench/patches/mtp_draft_head_q.py) of the top 32 | patch | draft head 395 → 210 µs per step, same acceptance |
| [Draft prefill on the last token only](src/g4bench/patches/mtp_draft_prefill_last.py) (the draft shares the target's KV cache) | patch | +2.6% at 8K × 16 |
| [Hybrid tokenizer](src/g4bench/patches/hybrid_tokenizer.py): fastokens to encode, HF tokenizers to decode, identical ids | patch | prompt encode 13.5 → 1.1 ms per 8K prompt |
| cuBLAS 13.10 instead of 13.1 | toolchain | −2 to −3% on every GEMM |

Every change, adopted or rejected, is documented in the [decision log](docs/DECISIONS.md). GEMMs already run at the
hardware limit on this GPU (decode at the weight-read floor, prefill at 400–430 TFLOPS; see [`kernels/`](kernels/)).

## Setup

- **GPU:** 1× NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB, 600 W, sm_120) on [Modal](https://modal.com).
- **Engine:** vLLM v0.33.0 (digest-pinned image) plus the patches above; SGLang v0.5.23 for the baseline.
- **Model:** [`google/gemma-4-12B-it`](https://huggingface.co/google/gemma-4-12B-it) in BF16 with the
  [`gemma-4-12B-it-assistant`](https://huggingface.co/google/gemma-4-12B-it-assistant) MTP draft, revisions pinned.
- **Client:** [InferenceX](https://github.com/SemiAnalysisAI/InferenceX) `bench_serving` ([vendored](third_party/inferencex/)
  @ `0b991451`): random prompts of 1K or 8K tokens, 1024 output tokens, concurrency 1 / 4 / 16 / 64, 10 requests per
  unit of concurrency.

## Reproduce

```bash
scripts/setup.sh                    # venv, pinned-revision and engine-flag checks
g4bench modal prefetch --execute    # pinned weights into a Modal Volume

# The headline table: article recipe (A) vs final config (B), interleaved on one GPU, one job per workload
# (workloads are named <input>x<concurrency>; drop --execute to print the plan)
for w in 1024x1 1024x4 1024x16 1024x64 8192x1 8192x4 8192x16 8192x64; do
  g4bench modal pair --a vllm-0.33.0-mtp4 --b vllm-0.33.0-best-e8-pfa2-sync-hybtok \
      --matrix backboard-repro-shared --only "$w" --repeats 2 --execute
done

g4bench analyze && g4bench claims   # results/ → results/summary/ and docs/CLAIMS.md
```

Modal credentials go in an untracked `.env` (`MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`).
`g4bench modal selftest --execute` runs the test suite on Linux.

## How we measured

- **Same-GPU A/B pairs.** Cloud GPUs of one SKU vary 5–15%, so every number is a pair: the article's vLLM recipe
  and our config on one GPU, interleaved ABBA / BAAB. Pairs at one workload agree within 0.1–1% (3% at 8K × 4). All pair
  results: [`results/pairs/`](results/pairs/).
- **Baselines reproduced first.** The article's vLLM recipe lands within +0.9% to +3.1% of its published throughput
  at all 8 workloads; its SGLang recipe runs 7–17% faster than published (below).
- **Adoption rules written before each run**, and configs tuned on seed 0 then confirmed on held-out seeds 1 and 2.
- **Outputs checked.** Every config is compared with a greedy, no-speculation BF16 reference on 8 prompts up to
  7.9K tokens; the only divergences, the article's recipe included, are near-ties between the top two tokens.

<details>
<summary>Baseline reproduction (output tok/s, seed 0)</summary>

| Workload | vLLM ours | vLLM article | Δ | SGLang ours | SGLang article | Δ |
|---|---:|---:|---:|---:|---:|---:|
| 1K × 1 | 190.4 | 186.8 | +1.9% | 136.9 | 122.6 | +11.7% |
| 1K × 4 | 655.0 | 649.0 | +0.9% | 499.1 | 456.8 | +9.3% |
| 1K × 16 | 1905 | 1861 | +2.4% | 1299 | 1122 | +15.8% |
| 1K × 64 | 3668 | 3634 | +0.9% | 2039 | 1750 | +16.5% |
| 8K × 1 | 164.6 | 159.6 | +3.1% | 89.4 | 82.5 | +8.4% |
| 8K × 4 | 402.6 | 395.7 | +1.8% | 287.9 | 269.5 | +6.8% |
| 8K × 16 | 801.9 | 794.5 | +0.9% | 587.5 | 533.7 | +10.1% |
| 8K × 64 | 1095 | 1079 | +1.5% | 728.6 | 673.6 | +8.2% |

Recipes as in the article: vLLM MTP with 4 draft tokens; SGLang NEXTN with 5 steps / 6 tokens.
</details>

### Notes

- Backboard's engine is not public, so throughput is normalized through the article's vLLM figure (¹); TTFT is
  measured directly.
- p99 TPOT trails Backboard at 8K × 4 (11.3 vs 10.3 ms), 8K × 16 and 8K × 64 (+4%); the config favours TTFT there.
- p90 TTFT at concurrency ≥ 4 is dominated by the first burst of simultaneous requests; median TTFT is 16–26% below
  the article recipe's at every workload.
- Prompts are random tokens with range ratio 0.8 and ignore-EOS, following InferenceX.

## Repository layout

```
configs/engines/        engine configs: the article's recipes and every experiment
configs/targets/        the article's published numbers, transcribed and test-checked
src/g4bench/            harness: runner, client, stats, analysis, Modal launcher, vLLM patches (patches/)
kernels/                GPU micro-benchmarks and the prefill-attention kernels
results/pairs/          same-GPU paired comparisons
results/runs/           every measured run: normalized results, server logs, configs
results/summary/        aggregated tables
docs/                   decision log, experiment plan, claims register, sources
```

## Acknowledgements

Built on [vLLM](https://github.com/vllm-project/vllm), [SGLang](https://github.com/sgl-project/sglang),
[InferenceX](https://github.com/SemiAnalysisAI/InferenceX), [Triton](https://github.com/triton-lang/triton) and
Gluon, [fastokens](https://pypi.org/project/fastokens/) and [Modal](https://modal.com). Thanks to Backboard for
publishing detailed numbers to compare against.

## Citation

```bibtex
@misc{gemma4-inference-repro,
  title  = {gemma4-inference-repro: Open-source vLLM beyond Backboard's published Gemma 4 12B serving numbers},
  author = {Wajibu, Ahamed Mohamed},
  year   = {2026},
  url    = {https://github.com/amw720386/gemma4-inference-repro}
}
```

## License

[MIT](LICENSE). `third_party/inferencex` keeps its Apache-2.0 license.
