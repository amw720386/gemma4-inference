# Roofline sanity report (analytic, NOT measured)

Target weights read/step: 23.92 GB; draft ~0.297 GB/step
KV/seq at ctx 2048: SWA-aware 352 MB; no-SWA 688 MB
KV/seq at ctx 9216: SWA-aware 411 MB; no-SWA 3095 MB

| GPU | $/hr | no-spec bs1 tok/s bound | 64x9K KV (GB) fits? | AL needed for article vLLM 8K/c1 speed-per-user 185.2 tok/s (4 drafts, 85% BW eff.) |
|---|---:|---:|---|---:|
| RTX-PRO-6000 | 3.03 | 63 | 26.3 yes | 3.12 |
| L40S | 1.95 | 30 | 26.3 NO | 6.47 |
| A100-80GB | 2.50 | 71 | 26.3 yes | 2.74 |
| H100 | 3.95 | 117 | 26.3 yes | 1.67 |
| H200 | 4.54 | 168 | 26.3 yes | 1.16 |
| B200 | 6.25 | 280 | 26.3 yes | 0.70 |

Interpretation: on RTX PRO 6000 the no-spec single-stream ceiling is ~60-65 tok/s, so the reported ~185-204 tok/s/user requires mean acceptance length ~3+ tokens/step from MTP. Throughput is therefore dominated by draft acceptance, which depends on the prompt content (random tokens + chat template) - the key reproduction risk.
