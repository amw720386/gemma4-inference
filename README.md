# gemma4-inference-repro

An **independent, open-source** attempt to reproduce — and then measurably improve on — the
Gemma 4 12B serving numbers reported by Backboard on 2026-11-09
([article](https://backboard.io/blog/gemma-4-12b-vllm-sglang-inference)).

> **What this is not.** Backboard's inference engine is proprietary. This project does not
> reimplement or claim to reproduce it. We reproduce the article's *open* baselines
> (vLLM 0.33.0, SGLang 0.5.23) with an InferenceX-style harness, report our own measurements
> with uncertainty, and treat Backboard's numbers purely as external targets.

**Current state (2026-11-10):** harness, real-prompt (SPEED-Bench) arm, per-run reports, vLLM-vs-SGLang
matched-load comparison, gap attribution, static engine-flag preflight, prepared E5/E6 optimization arms, and a
server-side **$10 project / $2 per-job spend ledger** in the Modal bridge (independently reviewed, D-030) are complete
and CPU-validated (120 tests, incl. a Python-3.12 engine-image rehearsal of the worker). **No GPU measurements yet** —
the bridge is not deployed. See [STATUS.md](STATUS.md), [NEXT.md](NEXT.md), [BUDGET.md](BUDGET.md).

Once deployed, the paid path is: `python modal_bridge/client.py verify` (free; checks the server shows $10 / ≤ $2 /
1 active) → `python modal_bridge/campaign.py e1-smoke` (plan) → `… --execute` (stops at the first failed gate). No number in this repo is a measured GPU
result unless it appears in `results/runs/*/results.csv` with a run ID.

## What the article reports (targets, not our results)

1× RTX PRO 6000 Blackwell 96 GB, `google/gemma-4-12B-it` BF16, draft `google/gemma-4-12B-it-assistant`,
1K output tokens, output throughput (tok/s) at concurrency 1 / 4 / 16 / 64:

| ISL | Backboard | vLLM 0.33.0 (MTP 4) | SGLang 0.5.23 (NEXTN 5/6) |
|---|---|---|---|
| 1K | 203.6 / 779.4 / 2082 / 3822 | 186.8 / 649.0 / 1861 / 3634 | 122.6 / 456.8 / 1122 / 1750 |
| 8K | 179.1 / 529.6 / 961.8 / 1254 | 159.6 / 395.7 / 794.5 / 1079 | 82.5 / 269.5 / 533.7 / 673.6 |

All tables (p99 TPOT/E2EL, p90 TTFT, per-user speed, 13-metric ratio tables) are in
[`configs/targets/backboard-2026-11-09.json`](configs/targets/backboard-2026-11-09.json).

## What we verified before writing any code

See [docs/SOURCES.md](docs/SOURCES.md). Highlights:
- Model, draft, licences (Apache-2.0, **ungated**), architecture, and exact HF revisions confirmed.
- vLLM 0.33.0 / SGLang 0.5.23 exist, are pinned by image digest, and contain Gemma 4 Unified + MTP code.
- Baseline flags match the vLLM recipe and SGLang cookbook.
- **Unstated by the article:** InferenceX commit, range ratio, GPU edition/power, driver, server limits.
- **Inferred:** the article's latency tables are impossible under fixed 1024-token outputs
  (16/24 violations of p99 E2EL ≥ 1023·p99 TPOT) ⇒ variable lengths, consistent with InferenceX's 0.8.
- Roofline: no-spec decode on RTX PRO 6000 is capped near ~63 tok/s/stream, so the reported
  ~185–204 tok/s/user requires speculative acceptance length ≳ 3.1 — the key reproduction variable.

## Layout

```
configs/pins.yaml             pinned model revisions, image digests, client commit
configs/engines/*.yaml        server configs (baselines, controls, variants, CPU mock)
configs/matrix/*.yaml         workload matrices (article matrix, smoke, mock)
configs/targets/*.json        article numbers (transcribed, test-checked)
configs/fingerprints-*.json   CPU-generated fingerprints of the exact request streams
src/g4bench/                  harness (runner, client, stats, compare, roofline, cost, ...)
third_party/inferencex/       vendored InferenceX bench_serving @ 0b991451 (Apache-2.0)
modal_bridge/                 one-time-deployable Modal app + HTTPS client + BOOTSTRAP.md
results/runs/<run_id>/        raw + normalized results per run (CSV/JSONL/manifest/metadata)
results/summary/              aggregated tables (g4bench analyze)
docs/                         sources, analysis
```

## Quick start (CPU, one command)

```bash
scripts/setup.sh
```

Creates `.venv`, installs the exact tested versions from `requirements-lock.txt`, checks the pinned model revisions
are reachable, statically checks every engine config's flags against the pinned vLLM/SGLang sources, and runs the
CPU test-suite. Add `--data` to also prepare the SPEED-Bench prompt sets (digest-gated against InferenceX, ~4.6 GB
streamed). Verified from a fresh clone on 2026-11-10: 120 passed, 1 skipped (the `--data` digest test), ~3 min.

Other free checks:

```bash
.venv/bin/g4bench consistency && .venv/bin/g4bench roofline && .venv/bin/g4bench plan
```

On a GPU host with the engine installed (or via the Modal bridge):

```bash
g4bench run --engine vllm-0.33.0-mtp4 --matrix backboard-repro --seeds 0
```

```bash
g4bench analyze
```

Every run writes `results/runs/<run_id>/REPORT.md` + `throughput.svg` (regenerate with
`g4bench report <run_dir>`). See `docs/examples/` for the layout (MOCK engine — not a measurement).

## Project docs

[STATUS.md](STATUS.md) · [NEXT.md](NEXT.md) · [EXPERIMENTS.md](EXPERIMENTS.md) (plan + pre-registered
acceptance criteria) · [DECISIONS.md](DECISIONS.md) · [BUDGET.md](BUDGET.md) · [docs/SOURCES.md](docs/SOURCES.md)

Licence: not yet chosen by the repository owner (see NEXT.md). `third_party/inferencex` keeps its upstream Apache-2.0 licence and notice.
