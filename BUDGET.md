# GPU budget and spend ledger

## Authorization (2026-11-10)

**Approved: an INITIAL project-wide cap of USD $10 total** (by Ahamed). This supersedes the earlier $100
*proposal*, which was never approved. Any increase needs a new explicit approval and a code change to `spend.Caps`
(the `G4_PROJECT_CAP_USD` / `G4_PER_JOB_CAP_USD` environment variables can only tighten the caps).

Paid runs are launched **directly from the owner's machine** with `g4bench modal …` (D-039; the HTTPS bridge was
removed). Credentials come from `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` in the untracked `.env`.

## How the cap is enforced (and what it is not)

Locally, before every launch (`src/g4bench/spend.py`, tested in `tests/test_spend.py`):

| Control | Value | Mechanism |
|---|---|---|
| Project cap | **$10.00** | refuse unless committed + worst case ≤ $10. Committed = max(local ledger, **Modal billing for `g4repro*` apps**) + Volume storage accrual |
| Per-job cap | **$2.00** | the call's Modal `timeout` is derived from its reservation, so Modal kills it before the worst case |
| Concurrency | 1 | refuse while any reservation is unsettled; functions have `max_containers=1`, `retries=0` |
| Nothing left running | ephemeral `app.run()` | no deployed app; the app stops when the launching process exits |
| Reconciliation | every launch | `modal billing report` per app id; Modal's figure replaces ours when larger; a crashed launch stays at its worst case until billed **and** past its deadline |
| Fail closed | – | billing unreadable ⇒ no launch |

Worst case = rate at the container's **limits** × (Modal timeout + 900 s startup margin + 60 s scaledown margin).
Rates (`modal billing rates`, 2026-11-10): GPU job (RTX PRO 6000 + 8 cores + 64 GiB limits) **$3.92/h**;
CPU prefetch (2 cores, 4 GiB) $0.13/h; Volumes $0.09/GiB-month. A $2 reservation buys an **876 s** GPU timeout.
Settled estimate = rate × (end − reservation time + 60 s), counted from before the container started.

**This is a conservative estimate that gates launches made by this tool. It is NOT a Modal billing hard stop**;
Modal's invoice is authoritative, which is why every launch first reconciles against it.

## What $10 buys (pessimistic `g4bench plan`, same full-container pricing as the ledger)

| Step | Est. GPU-h | Est. $ (settled) | Reserved while running |
|---|---:|---:|---:|
| `modal selftest` (CPU; test-suite on Linux) | – | ≈ 0.01 | 0.08 |
| `modal prefetch` (CPU; weights) | – | ≤ 0.05 | 0.13 |
| E1 smoke vLLM (2 pts, 1K/OSL 256) | 0.17 | 0.68 | 2.00 |
| E1 smoke SGLang | 0.18 | 0.69 | 2.00 |
| one article point, seed 0 (e.g. 1K/c1 or 8K/c4), per engine | ~0.2 | 0.76–0.86 | 2.00 |

So after E1 (~$1.4 incl. prefetch), roughly **$8.5 remains ≈ 8–9 single-point jobs**. The full
seed-0 matrix (≈ $6.08 vLLM + $7.81 SGLang) and the repeats (≈ $27) do **not** fit in $10; jobs must also be split
so each fits the 876 s timeout (`--only <key>`). The plan within $10 is in NEXT.md.

## Ledger

Machine-readable: `results/budget_ledger.jsonl` (written by `g4bench modal`; `g4bench modal spend` prints the
totals including Modal's billed figure).

| Date (UTC) | Job id | Experiment | GPU | Reserved $ | Settled est. $ | Notes |
|---|---|---|---|---:|---:|---|
| 2026-11-10 | – | E0 CPU-only (this container) | none | 0 | **0.00** | no paid compute |
| 2026-11-10 | – | E5c data prep (this container) | none | 0 | **0.00** | SPEED-Bench prepared locally |
| 2026-11-10 | – | e1-smoke `--execute` gate test | none | 0 | **0.00** | refused at /health (404, not deployed) before any POST |
| 2026-11-11 | 20261111T005351Z-2c8cb0 | selftest (direct Modal) | none | 0.076 | 0.003 | failed at app creation: vLLM image had no `python` (fixed by D-038) |
| 2026-11-11 | 20261111T005544Z-a782f7 | selftest (direct Modal) | none (2 CPU) | 0.076 | 0.012 | 98 passed, 1 skipped on Linux |

**Total spent to date: ≈ $0.02 (estimate)**; `g4bench modal spend --reconcile` replaces estimates with Modal's billed figures.

## Prices (`modal billing rates`, retrieved 2026-11-10)

RTX-PRO-6000 $0.000842/s · L40S $0.000542/s · A100-80GB $0.000694/s · H100 $0.001097/s · H200 $0.001261/s ·
B200 $0.001736/s · CPU $0.0000131/core/s · memory $0.00000222/GiB/s. Hard-coded in `src/g4bench/spend.py`;
re-check before launches and update both if Modal changes prices.
