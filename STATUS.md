# Status — 2026-11-10

## Summary

| Area | State |
|---|---|
| Source verification (article, model, draft, engines, client, recipes) | ✅ done — docs/SOURCES.md |
| Harness (runner, client, stats, compare, metadata, telemetry, spec-decode capture) | ✅ built, CPU-tested |
| CPU checks (24 tests incl. mock end-to-end) | ✅ passing |
| Model-availability test | ✅ both pinned repos reachable, ungated |
| Request-stream fingerprints (24 points) | ✅ configs/fingerprints-rr0.8.json |
| Article consistency analysis | ✅ fixed-length outputs ruled out (16/24) |
| Modal HTTPS bridge | 🟨 written + definition-validated (modal 1.6.3); **not deployed** |
| GPU baselines (E1–E4) | ⛔ blocked on bridge deploy + proxy-auth credential |
| Optimization research (E5–E6) | ⬜ waits on baselines |
| GPU spend | **$0.00** |

## Measured results

**None on GPU.** The only executed runs are CPU mock runs inside pytest (engine `mock`, excluded from all
analysis by default). Nothing in this repo should be read as a reproduction result yet.

## Blocker (external, needs a human once)

1. Deploy `modal_bridge/app.py` to workspace `arnold-17059` — exact steps in `modal_bridge/BOOTSTRAP.md`.
2. Provide a Modal proxy-auth token for host `arnold-17059--g4repro-bridge.modal.run` through the
   environment's credential form (Bearer `<id>.<secret>`). No other secret is required (models ungated).

Checked 2026-11-10: the host is reachable via HTTPS from this environment and returns
`modal-http: invalid function call` (nothing deployed). No existing bridge for this project exists; we did not
look at or use any other project's bridge or credentials.

## Key findings so far (CPU-side, no GPU)

- `google/gemma-4-12B-it` @ `43c9d014`: encoder-free unified 12B, 48 layers (40 SWA-1024 + 8 global with
  K=V sharing). KV per sequence at 9K context is only ~411 MB *if* the engine is SWA-aware (vs ~3.1 GB if
  not) — 64×8K fits in 96 GB only with SWA-aware allocation.
- Draft `gemma-4-12B-it-assistant` @ `3b0b0a4a`: 4-layer, KV-sharing MTP drafter with a centroid-masked
  LM head (top-32 of 2048 clusters ≈ 4K of 262K vocab). Backboard's "4-bit shortlist, BF16 rescore" reads like
  a quantized variant of this shortlist step; SGLang 0.5.23 has `--speculative-draft-model-quantization`.
- Roofline: no-spec single-stream ≈ 63 tok/s bound on RTX PRO 6000 ⇒ reported ~185–204 tok/s/user needs
  acceptance length ≳ 3.1. Throughput is acceptance-dominated, and acceptance depends on prompt content.
- Range ratio: InferenceX default 0.8 ⇒ "1K" ≈ 917–948 input / 910–939 output tokens on average.
- InferenceX rules forbid lowering draft precision below shipped precision; Backboard's method would be
  inadmissible there (class (b) in DECISIONS D-011), though it is stated to be output-preserving.

## Tooling versions used for CPU checks

Python 3.13.18; numpy 2.5.4, aiohttp 3.14.5, transformers 5.22.2, tokenizers 0.23.3, huggingface_hub 2.4.0,
jinja2 3.1.6, fastapi 0.162.4, pytest 9.2.0, modal 1.6.3 (definition import only).
