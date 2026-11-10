# One-time Modal bridge bootstrap (human step, ~5 minutes, ~$0)

The agent environment can reach `*.modal.run` over HTTPS but cannot use Modal's gRPC
control plane (CLI/SDK). So a human deploys the bridge **once** from a normal machine;
afterwards the agent submits, polls and cancels jobs over HTTPS only.

Target workspace: **Team Arnold (`arnold-17059`)**. Expected bridge URL:
`https://arnold-17059--g4repro-bridge.modal.run` (as of 2026-11-10 that host answers
`modal-http: invalid function call`, i.e. nothing deployed yet).

Deploying creates only: one app `g4repro-bridge`, two Volumes (`g4repro-hf-cache`,
`g4repro-results`) and one Dict (`g4repro-jobs`). It touches no other app, secret, or
billing setting, and needs **no secrets** (both models are ungated on Hugging Face).
Idle cost is $0: GPU functions run only when a job is submitted.

## 1. Deploy (on your own machine)

```bash
git clone -b agent/gemma4-repro https://github.com/amw720386/gemma4-inference-repro
```

```bash
cd gemma4-inference-repro && python3 -m pip install "modal==1.6.3"
```

```bash
modal profile current
```

Make sure the active profile is the `arnold-17059` workspace (otherwise `modal profile activate <profile>`,
or `modal token new` and pick that workspace), then:

```bash
modal deploy modal_bridge/app.py
```

The output lists the web endpoint URL; it should be the URL above. If it differs, note it.

Optional, recommended guard rails (defaults shown; set before deploy if you want different values):
`G4_BRIDGE_HARD_CAP_USD=40` (max estimated $ per job) and `G4_BRIDGE_MAX_ACTIVE=1` are read from the
bridge container environment; to change them, edit the `ENV` dict in `modal_bridge/app.py` and redeploy.

## 2. Create a proxy-auth token

Modal dashboard → workspace `arnold-17059` → Settings → **Proxy Auth Tokens** → create one
(or `modal workspace proxy-tokens` per Modal docs). You get a *Token ID* (`wk-…`) and a *Token Secret* (`ws-…`).

## 3. Give the agent access (pick one; never paste secrets into chat)

**Preferred:** when the agent asks via the secure credential form for host
`arnold-17059--g4repro-bridge.modal.run`, enter **`<TokenID>.<TokenSecret>`** (joined by a period) as a
Bearer token. Modal documents `Authorization: Bearer <id>.<secret>` as equivalent to the
`Modal-Key`/`Modal-Secret` headers. The environment's proxy then injects it; the token never enters the repo or logs.

If the URL differs from the expected one, also set the non-secret environment variable
`G4_BRIDGE_URL=<url>` in the cloud environment settings.

## 4. Verify (agent side, HTTPS only)

```bash
python modal_bridge/client.py health
```

```bash
python modal_bridge/client.py selftest
```

Then the first paid step (≈ $0.05 CPU, then ≈ $0.52 GPU), each logged in BUDGET.md first:

```bash
python modal_bridge/client.py prefetch
```

```bash
python modal_bridge/client.py submit --engine vllm-0.33.0-mtp4 --matrix smoke --max-usd 2
```

## Stopping things

- Cancel a job: `python modal_bridge/client.py cancel <call_id>` (agent) or the Modal dashboard.
- Remove everything: `modal app stop g4repro-bridge` (and delete the two Volumes / the Dict in the dashboard if desired).
