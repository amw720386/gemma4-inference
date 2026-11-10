# GPU budget and spend ledger

## Authorization (2026-11-10)

**Approved: an INITIAL project-wide cap of USD $10 total** (by Ahamed). This supersedes the earlier $100
*proposal*, which was never approved. Any increase needs a new explicit approval and a redeploy of the bridge
with a new `G4_PROJECT_CAP_USD`.

Preconditions before any paid launch (none are met yet):
1. the bridge is deployed by the user (not from this cloud session);
2. a separate, scoped proxy-auth token is provisioned by the user (the agent does not request it);
3. `GET /health` returns `ok: true`, `cap_usd: 10`, `per_job_cap_usd <= 2`, and `GET /budget` shows
   `available_usd` covering the next job. Then **smoke first**, one job at a time.

## How the cap is enforced (and what it is not)

Server-side in the deployed bridge (`modal_bridge/app.py` + `src/g4bench/budget.py`, tested in
`tests/test_budget.py`, `tests/test_bridge_api.py`):

| Control | Value | Mechanism |
|---|---|---|
| Project cap | **$10.00** (`G4_PROJECT_CAP_USD`) | ledger in Modal Dict `g4repro-budget`; reservation refused if platform reserve + open reservations + settled estimates would exceed it |
| Platform reserve | $0.50 (`G4_PLATFORM_RESERVE_USD`) | set aside, not metered, for the bridge (≤ 1 core / 1 GiB) + ledger (≤ 0.5 core / 0.5 GiB) containers: ≈ $0.08/h while warm ⇒ ≈ 6 h of continuous polling; poll every ≥ 120 s → **usable for jobs: $9.50** |
| Per-job cap | **$2.00** (`G4_PER_JOB_CAP_USD`) | each job's Modal `timeout` is derived from its reservation so Modal kills it before the worst case |
| Concurrency | 1 open reservation (`G4_BRIDGE_MAX_ACTIVE`) | covers `/jobs`, `/prefetch` and `/selftest` alike |
| Race safety | single-container, one-input ledger function + Dict lease lock (atomic put-if-absent) | 40-thread and 12-request parallel tests never exceed the cap |
| Restart safety | whole ledger in one Dict key, re-read on every op | restart test |
| Fail closed | unreadable/corrupt ledger ⇒ 503, nothing launched | test |

Worst-case reservation = rate at the container's **limits** × (Modal timeout + 900 s startup margin + 60 s
scaledown margin). Rates: GPU job (RTX PRO 6000 + 8 cores + 64 GiB limits) **$0.001089/s = $3.92/h**; CPU
prefetch/selftest (2 cores, 4 GiB) $0.126/h. A $2 reservation buys a **876 s** GPU timeout.

Settlement: held at the full reservation until Modal confirms the call terminal; then cost = rate × (job's
reported end − reservation time + 60 s), i.e. counted from *before* the container started, never capped (overruns
are recorded). Unknown/transient poll errors keep the reservation; expired outputs settle at the worst case;
a spawn that failed (no call id) is the only thing ever released. Jobs abort if (re)started after their deadline
(preemption restarts); a restart already in flight can overrun by at most one attempt, which settlement records.

**This is a conservative *estimate* that gates launches made through this bridge. It is NOT a Modal billing hard
stop**, it cannot see anything launched outside the bridge, and Modal's invoice is authoritative. Estimated
settled amounts are labelled "estimate" everywhere (`/budget` includes a disclaimer).

## What $10 buys (pessimistic `g4bench plan`, same full-container pricing as the ledger)

| Step | Est. GPU-h | Est. $ (settled) | Reserved while running |
|---|---:|---:|---:|
| `/selftest` (CPU) | – | ≤ 0.10 | 0.10 |
| `/prefetch` (CPU; weights + SPEED-Bench) | – | ≤ 0.16 | 0.16 |
| E1 smoke vLLM (2 pts, 1K/OSL 256) | 0.17 | 0.68 | 2.00 |
| E1 smoke SGLang | 0.18 | 0.69 | 2.00 |
| one article point, seed 0 (e.g. 1K/c1 or 8K/c4), per engine | ~0.2 | 0.76–0.86 | 2.00 |

So after E1 (~$1.6 incl. CPU steps + $0.50 reserve), roughly **$7.5 remains ≈ 8–9 single-point jobs**. The full
seed-0 matrix (≈ $6.08 vLLM + $7.81 SGLang) and the repeats (≈ $27) do **not** fit in $10; jobs must also be split
so each fits the 876 s timeout (`--only <key>`). The plan within $10 is in NEXT.md.

## Ledger (agent's copy; the bridge's `/budget` is the source of truth once deployed)

Machine-readable per-step log: `results/budget_ledger.jsonl` (written by the campaign runner).

| Date (UTC) | Job id | Experiment | GPU | Reserved $ | Settled est. $ | Notes |
|---|---|---|---|---:|---:|---|
| 2026-11-10 | – | E0 CPU-only (this container) | none | 0 | **0.00** | no paid compute |
| 2026-11-10 | – | E5c data prep (this container) | none | 0 | **0.00** | SPEED-Bench prepared locally |
| 2026-11-10 | – | e1-smoke `--execute` gate test | none | 0 | **0.00** | refused at /health (404, not deployed) before any POST |

**Total spent to date: $0.00.** Bridge not deployed; no Modal compute has run.

## Prices (modal.com/pricing, retrieved 2026-11-10)

RTX-PRO-6000 $0.000842/s · L40S $0.000542/s · A100-80GB $0.000694/s · H100 $0.001097/s · H200 $0.001261/s ·
B200 $0.001736/s · CPU $0.0000131/core/s · memory $0.00000222/GiB/s. Hard-coded in `src/g4bench/budget.py`;
re-check before launches and update both if Modal changes prices.
