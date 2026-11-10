# One-time Modal bridge bootstrap (human step, ~5 minutes)

The agent environment can reach `*.modal.run` over HTTPS but must not use Modal's gRPC control plane
(CLI/SDK). So **you** deploy the bridge once from your own machine; afterwards the agent submits, polls and
cancels jobs over HTTPS only, and every paid launch is gated by the bridge's budget ledger.

Target workspace: **Team Arnold (`arnold-17059`)**. Expected URL:
`https://arnold-17059--g4repro-bridge.us-east.modal.run` (regional host the environment's network secret is bound to; the unregional `…--g4repro-bridge.modal.run` also answered 404) (as of 2026-11-10 it answers
`modal-http: invalid function call`, i.e. nothing deployed).

## What gets created

One app `g4repro-bridge` with functions `bridge` (web, proxy-auth), `ledger_op` (budget ledger, 1 container),
`run_vllm`, `run_sglang` (GPU, RTX-PRO-6000), `prefetch_weights`, `selftest` (CPU); two Volumes
(`g4repro-hf-cache`, `g4repro-results`); one Dict **`g4repro-budget`** (the spend ledger — do not delete it while
the project is active, or the spend history is lost and the cap restarts from $0). No secrets (both models are
ungated); no billing settings touched. Idle cost ≈ $0 (nothing runs until a request arrives).

## Budget configuration (baked into the deployment)

| Env var at deploy time | Default | Meaning |
|---|---|---|
| `G4_PROJECT_CAP_USD` | **10** | approved project-wide cap (2026-11-10) |
| `G4_PER_JOB_CAP_USD` | **2** | max reservation per job; also bounds each job's Modal timeout (876 s GPU) |
| `G4_BRIDGE_MAX_ACTIVE` | 1 | open reservations at once, across GPU jobs, prefetch and selftest |
| `G4_PLATFORM_RESERVE_USD` | 0.50 | floor for bridge/ledger container spend (metered usage above it also counts) |

Aliases accepted at deploy time: `G4_PROJECT_BUDGET_USD` (= `G4_PROJECT_CAP_USD`), `G4_JOB_CAP_USD` (= `G4_PER_JOB_CAP_USD`).
On the agent side, the same two names in the cloud environment can only *tighten* the limits the campaign requires
the server to report.

The defaults equal the approved values, so **no env vars are needed**. To change a value, redeploy with the variable
set, e.g. `G4_PROJECT_CAP_USD=10 modal deploy …`; the agent cannot change it. Lowering the cap is always safe;
spend already in the ledger stays counted. The ledger is a conservative estimate that gates launches through the
bridge — **it is not a Modal billing hard stop**. If you want an account-level guard as well, set one yourself in
the Modal dashboard (the agent does not touch billing settings).

## 1. Deploy (on your own machine)

```bash
git clone -b agent/gemma4-repro https://github.com/amw720386/gemma4-inference-repro
```

```bash
cd gemma4-inference-repro && python3 -m venv .venv && . .venv/bin/activate && pip install -e ".[bridge]"
```

```bash
modal profile current
```

Ensure the active profile is the `arnold-17059` workspace (otherwise `modal profile activate <profile>`, or
`modal token new` and choose that workspace), then:

```bash
modal deploy modal_bridge/app.py
```

The output lists the web endpoint URL; it should be the URL above. If it differs, note it.

## 2. Create a scoped proxy-auth token

Modal dashboard → workspace `arnold-17059` → Settings → **Proxy Auth Tokens** → create a token used only for this
bridge (or `modal workspace proxy-tokens` per Modal docs). Per Modal's docs it authenticates requests to
proxy-auth web endpoints; it is distinct from the Modal API token used for `modal deploy`, which stays on your machine.

## 3. Give this cloud environment access

Add it as a **network secret** on the cloud environment (environment settings → network secrets): host
`arnold-17059--g4repro-bridge.modal.run`, header `Authorization`, value `Bearer <TokenID>.<TokenSecret>`
(Modal documents this as equivalent to the `Modal-Key`/`Modal-Secret` headers). The token then never enters the
repo, logs or chat. If the URL differs, also set the non-secret env var `G4_BRIDGE_URL=<url>`.

## 4. Verification the agent performs before any paid work (HTTPS only, free)

```bash
python modal_bridge/client.py health
```

Must show `ok: true`, `ledger_ok: true`, `cap_usd: 10.0`, `per_job_cap_usd: 2.0`, `max_active: 1`.

```bash
python modal_bridge/client.py budget
```

Must show `cap_usd: 10.0`, `committed_usd: 0.5` (platform reserve only), `available_usd: 9.5`, no active jobs.
Only then, one at a time, each logged in BUDGET.md first and stopping if `/budget` cannot cover the next job:
`selftest` (≤ $0.10) → `prefetch` (≤ $0.16) → `submit --engine vllm-0.33.0-mtp4 --matrix smoke --max-usd 2`.

## Stopping things

- Cancel a job: `python modal_bridge/client.py cancel <job_id>` (reservation stays held until termination is confirmed),
  or the Modal dashboard.
- Remove everything: `modal app stop g4repro-bridge`; delete the Volumes / the Dict in the dashboard only when the
  project is finished (the Dict holds the spend ledger).
