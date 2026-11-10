# GPU budget and spend ledger

Rules: estimate with `g4bench plan` (or bridge `/estimate`) **before** every launch; record estimate and
actual here; actual = Modal per-second GPU price × wall time reported by the bridge (Modal's invoice is
authoritative – we never read or change billing settings). Stop and re-plan if any job exceeds its
estimate by > 50 %.

## Caps (proposed; change only with the user's approval)

| Scope | Cap |
|---|---:|
| Per job (bridge hard cap `G4_BRIDGE_HARD_CAP_USD`) | $40 |
| E1 smoke | $2 |
| E2 + E3 baselines (seed 0) | $15 |
| E4 repeats | $25 |
| E5 + E6 ablations / optimization | $40 |
| **Project total without further approval** | **$100** |

## Prices used (modal.com/pricing, retrieved 2026-11-10)

RTX-PRO-6000 $0.000842/s ($3.03/h) · L40S $0.000542/s · A100-80GB $0.000694/s · H100 $0.001097/s ·
H200 $0.001261/s · B200 $0.001736/s · CPU $0.0000131/core/s · memory $0.00000222/GiB/s.

## Estimates (pessimistic, from `g4bench plan`)

| Experiment | Est. GPU-h | Est. $ |
|---|---:|---:|
| E1 smoke vLLM + SGLang | 0.35 | 1.05 |
| E2 vLLM seed 0 (8 pts) | 1.55 | 4.70 |
| E3 SGLang seed 0 (8 pts) | 1.99 | 6.04 |
| E4 seeds 1,2 both engines | 6.95 | 21.07 |
| E5a no-spec control seed 0 | ~3 | ~9 |
| **Planned through E5a** | **~13.8** | **~42** |

## Ledger

| Date (UTC) | Job / call id | Experiment | GPU | Est. $ | Actual wall | Actual $ | Notes |
|---|---|---|---|---:|---:|---:|---|
| 2026-11-10 | – | E0 CPU-only | none | 0 | – | **0.00** | no paid compute used |

**Total spent to date: $0.00**
