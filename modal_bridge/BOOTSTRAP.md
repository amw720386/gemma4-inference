# One-time Modal bridge bootstrap (human step, ~5 minutes)

The agent environment can reach `*.modal.run` over HTTPS but must not use Modal's gRPC control plane
(CLI/SDK). So **you** deploy the bridge once from your own machine; afterwards the agent submits, polls and
cancels jobs over HTTPS only, and every paid launch is gated by the bridge's budget ledger.

Target workspace: **`<workspace>`**. The cloud environment's network secret is currently bound to
`https://<workspace>--g4repro-bridge.us-east.modal.run`, and the client defaults to that host. As of 2026-11-10 both it
and the unregional `https://<workspace>--g4repro-bridge.modal.run` answer `modal-http: invalid function call` (HTTP
404), i.e. nothing is deployed. Whatever URL `modal deploy` prints is authoritative (step 3).

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

## 1. Deploy (on your own machine or a Colab/VM runtime with the Modal CLI; Python >= 3.11)

Deploy from `main` (all bridge code is merged there; there is no separate agent branch):

```bash
git clone https://github.com/amw720386/gemma4-inference-repro
```

```bash
cd gemma4-inference-repro && git checkout main && git log -1 --oneline
```

```bash
python3 -m venv .venv && . .venv/bin/activate && pip install -e ".[bridge]"
```

The last command installs the package plus `fastapi` and `modal==1.6.3` (the version the bridge was validated
with). Deploy from a clean checkout of the latest `main` (if you deployed earlier, redeploy: the pre-spend check refuses a
bridge missing any required feature, e.g. `run-file-listing`). `/health` reports the deployed git SHA (with `-dirty` if there were local
edits), and the agent's pre-spend check refuses a bridge that lacks the current safety features.

```bash
modal profile current
```

Ensure the active profile is the `<workspace>` workspace (otherwise `modal profile activate <profile>`, or
`modal token new` and choose that workspace), then:

```bash
modal deploy modal_bridge/app.py
```

The output lists the web endpoint URL (label `g4repro-bridge`); note it exactly for step 3. The vLLM image build adds
one setup step that aliases `python` to the image's own `python3` and checks it can import `vllm` (D-038); if that step
fails, the deploy stops at image build with the reason, before any GPU is used.

## 2. Create a scoped proxy-auth token

Modal dashboard → workspace `<workspace>` → Settings → **Proxy Auth Tokens** → create a token used only for this
bridge (CLI equivalent: the `modal workspace proxy-tokens` command group). Proxy tokens have a `wk-…` ID and a `ws-…`
secret and authenticate requests to proxy-auth web endpoints only; they cannot be used as Modal API tokens (`ak-`/
`as-`), so the API token used for `modal deploy` stays on your machine.

## 3. Give this cloud environment access

Add it as a **network secret** on the cloud environment (environment settings → network secrets): header
`Authorization`, value `Bearer <TokenID>.<TokenSecret>` (Modal documents this as equivalent to the
`Modal-Key`/`Modal-Secret` headers), and **host = exactly the host printed by `modal deploy`**. The environment
currently has such a secret for `<workspace>--g4repro-bridge.us-east.modal.run`; if the deployed host differs (e.g. the
unregional `<workspace>--g4repro-bridge.modal.run`), update the secret's host and set the non-secret env var
`G4_BRIDGE_URL=https://<that host>`. The token never enters the repo, logs or chat. Environment changes reach the
agent's container on its next restart.

## 4. Verification the agent performs before any paid work (HTTPS only, free)

```bash
python modal_bridge/client.py verify
```

Passes only if `/health` is ok with the ledger reachable and the current safety features, and both `/health` and
`/budget` report cap $10.00, per-job cap ≤ $2.00, max_active 1, the ledger disclaimer and no active jobs (a fresh
ledger shows `committed_usd` ≈ 0.50 platform reserve, `available_usd` ≈ 9.50). Then the agent runs:

```bash
python modal_bridge/campaign.py e1-smoke
```

```bash
python modal_bridge/campaign.py e1-smoke --execute
```

The first only prints the plan. The second repeats the free checks, sends `/estimate` for both smoke jobs, and only
then runs, one at a time: CPU selftest (≤ $0.10) → CPU prefetch (≤ $0.16) → vLLM smoke → SGLang smoke (≤ $2.00
reserved each), stopping at the first failed gate or when `/budget` cannot cover the next reservation. Every step is
logged to `results/budget_ledger.jsonl`.

## Stopping things

- Cancel a job: `python modal_bridge/client.py cancel <job_id>` (reservation stays held until termination is confirmed),
  or the Modal dashboard.
- Remove everything: `modal app stop g4repro-bridge`; delete the Volumes / the Dict in the dashboard only when the
  project is finished (the Dict holds the spend ledger).
