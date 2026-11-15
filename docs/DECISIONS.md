# Decision log

Each entry: context → decision → consequence. Newest last. Amend, never rewrite.

**D-001 (2026-11-10) Scope: open-source reproduction, not a clone.** Backboard's engine, kernels and
4-bit draft shortlist are proprietary and unpublished. We reproduce only the *open baselines* (vLLM,
SGLang) and treat Backboard's numbers as external targets. Nothing in this repo claims to implement
Backboard's method.

**D-002 (2026-11-10) Branch.** The user explicitly asked for `agent/gemma4-repro` (not `main`). The
session tooling suggested `claude/trusting-allen-aa9i7x`; the user's explicit instruction wins. Only
`agent/gemma4-repro` is pushed.

**D-003 (2026-11-10) Vendor the InferenceX client** (`third_party/inferencex`, Apache-2.0, commit
`0b991451`, SHA256SUMS). Reason: the article names no commit; vendoring freezes our client. Our argv is a
strict superset of `fixed_seq.client_argv` (test-enforced) adding only `--seed`, `--tokenizer` (pinned
local snapshot), `--save-detailed`, `--metric-percentiles 50,90,99,99.9` – none alter the request stream.

**D-004 (2026-11-10) Range ratio 0.8 is primary; 1.0 is a sensitivity arm.** The article does not state
it. InferenceX uses `RANDOM_RANGE_RATIO=0.8`. With fixed 1024-token outputs, p99 E2EL ≥ 1023·p99 TPOT
must hold (proof in `src/g4bench/inference_checks.py`); 16/24 published pairs violate it, including
vLLM 1K/c1 (5432 ms < 5678 ms bound). So outputs varied in length ⇒ ratio < 1, consistent with 0.8.

**D-005 (2026-11-10) More replication than the article.** Article: 1 run/point (3 seeds at 8K/c4). Ours:
seeds 0,1,2 at every point, plus restart trials where variance matters. Seeds are interleaved across the
matrix (not back-to-back) to spread drift.

**D-006 (2026-11-10) Hardware: go straight to exact hardware (RTX PRO 6000) for the smoke test.** Modal
offers it at $3.03/h. Proxies are not cheaper enough to justify a separate debug pass (L40S $1.95/h, and
cannot fit the 8K/c64 point); H100/H200/B200 cost more and are not the article's hardware. The CPU mock
already exercises the harness for $0.

**D-007 (2026-11-10) Baseline configs carry no extra flags.** Article: "published recipes with default
settings". So no `--max-model-len`, `--limit-mm-per-prompt`, `--context-length` in baselines; those live in
`vllm-0.33.0-mtp4-textonly` (variant, E5d). Explicit `--dtype bfloat16` equals the default for this
checkpoint; `--revision` pins have no performance effect.

**D-008 (2026-11-10) Greedy decoding** is what the InferenceX client sends (`temperature: 0.0`). We keep it.
Acceptance length under greedy + random-token prompts is the dominant performance factor (roofline:
≥ 3.1 needed for the article's single-stream numbers), so we record it for every point.

**D-009 (2026-11-10) 5 % systematic tolerance** for "reproduced" (EXPERIMENTS criteria 3–4), pre-registered.

**D-010 (2026-11-10) Modal only through a deployed HTTPS bridge with proxy auth.** This environment's
egress cannot use Modal's gRPC control plane; we do not retry the CLI or use undocumented APIs. A human
deploys `modal_bridge/app.py` once (BOOTSTRAP.md). Bridge enforces allowlists, cost estimate vs cap,
single active GPU job, soft wall budget + hard timeout. Verified 2026-11-10: `https://<workspace>--g4repro-bridge.modal.run`
is reachable over HTTPS and returns `modal-http: invalid function call` (= not deployed yet).

**D-011 (2026-11-10) Admissibility categories.** (a) like-for-like: BF16 target, shipped-precision draft;
(b) output-preserving but InferenceX-inadmissible: e.g. lower-precision draft (the class Backboard's
4-bit shortlist falls in); (c) not comparable: lower-precision target weights/KV. Only (a) can "match/beat"
the article's baselines; Backboard's own result is in class (b) by its own description.

**D-012 (2026-11-10) No GitHub Actions for GPU work; no billing changes.** All paid compute goes through the
bridge with a pre-launch estimate logged in BUDGET.md.

**D-013 (2026-11-10) Budget is a proposal, not an authorization.** The $100 cap in BUDGET.md is proposed
only. No paid GPU job is launched until the user explicitly confirms a budget **and** the HTTPS bridge
answers `/health`. CPU-side Modal steps (selftest, prefetch) are also treated as paid and wait for approval.

**D-014 (2026-11-10) Real-prompt selection policy.** `numpy.random.default_rng(seed).permutation(n)` cycled to
`num_prompts = 10·conc` requests (InferenceX request count). With 80 coding prompts, conc ≥ 16 repeats
prompts; prefix/radix caching is off in every baseline, so repeats are recomputed. `n_unique`/`max_repeats`
are stored per run.

**D-015 (2026-11-10) SPEED-Bench coding set prepared by re-implementation + digest gate.** Instead of running
NeMo-Skills' prepare.py (needs `datasets` and ~35 GB), `speedbench.py` re-implements its two coding branches
(code_contests, livecodebench), fetches only the 10 rows needed from the same pinned revisions (every file
sha256-verified; LiveCodeBench stream-hashed), and refuses to write unless the digest equals InferenceX's
`PROMPTS_SHA256`. Result: **match** (`6f184545…`), i.e. byte-identical to InferenceX's golden-AL prompt set.
Prompt text is not committed (SPEED-Bench ships under an NVIDIA licence, sources under various licences);
provenance sidecars with all hashes are.

**D-016 (2026-11-10) Throughput splits used as a labelled inline-only subset, not truncated.** Rows needing
external resolution (HLE is gated; AdaLEval/BAMBOO need fetches) are excluded → 897/1536 (1k) and 808/1536
(8k) rows; the `mixed` category is absent. Under the Gemma tokenizer + chat template the "1k" prompts average
1210 tokens (994–3076) and "8k" 9372 (7638–12256): SPEED-Bench sized them for another tokenizer. We keep the
text unmodified (fidelity to the dataset) and report actual lengths; these arms are never compared to the
article. `speedbench-tp` matrix sets max_model_len 14336 for bounded-context variants; baselines use the
engine default context, so they are unaffected.

**D-017 (2026-11-10) Real-prompt arms use ignore_eos=True first** (fixed 1024-token outputs, matching the
throughput matrix's decode load). Post-EOS text can distort acceptance either way, so an EOS-honouring
variant (`ignore_eos: false`, point IDs suffixed `-eos`) is the planned follow-up, not a replacement.

**D-018 (2026-11-10) Within-run request statistics are diagnostics.** Per-request bootstrap CIs (TTFT, TPOT,
E2EL, ITL) and the Little's-law throughput estimator describe one run's sampling noise only; claims use
between-run CIs (n ≥ 3). Missing/inconsistent per-request logs are reported as states, never imputed.

**D-019 (2026-11-10) Approved budget: $10 total project cap, enforced server-side.** Supersedes the $100 proposal
(D-013 stays: nothing paid runs without explicit approval). The bridge now keeps a project-wide ledger in Modal Dict
`g4repro-budget`; every paid launch (GPU jobs, CPU prefetch, CPU selftest) reserves its worst case at the container's
cpu/memory/GPU *limits* × (Modal timeout + 900 s startup + 60 s scaledown) before spawning; per-job cap $2 bounds the
Modal timeout (876 s on RTX PRO 6000 at $3.92/h incl. CPU/memory limits); one open reservation at a time; $0.50
platform reserve for the bridge/ledger containers. Mutations are serialized through a single-container, one-input
`ledger_op` function plus a Dict lease lock (atomic put-if-absent); the ledger is one key re-read per op, so restarts
lose nothing; corrupt/unreachable state fails closed. Reservations are released only if the spawn itself failed;
otherwise settled from wall time since reservation once Modal confirms termination (uncapped; expired ⇒ worst case;
unknown poll errors ⇒ keep holding). `G4_BRIDGE_HARD_CAP_USD` is removed (per-job cap replaces it). The ledger is an
estimate gating this bridge only — explicitly **not** a Modal billing hard stop.

**D-020 (2026-11-10) Bug fixed before first deploy: Modal's poll timeout exception.** `FunctionCall.get(timeout=0)`
raises `modal.exception.TimeoutError`, which is not a subclass of builtins `TimeoutError` (verified on modal 1.6.3).
The previous bridge caught only the builtin, so a running job would have been marked ended, defeating the concurrency
limit. The new backend catches it explicitly and maps only Modal's specific terminal exceptions (InputCancellation,
FunctionTimeoutError, OutputExpiredError) to terminal states; anything else is "unknown" and keeps the reservation.
A definition test also caught that web functions cannot take `retries` (deploy would have failed).

**D-021 (2026-11-10) Estimates priced like the ledger.** `g4bench plan` now prices GPU jobs at GPU + CPU/memory limits
($3.92/h on RTX PRO 6000, +29 % vs GPU-only), so plans and ledger reservations agree. Under $10 the full seed-0 matrix
for both engines (~$13.9) does not fit; work is planned as single-point jobs, smoke first (NEXT.md).

**D-022 (2026-11-10) Engine comparisons are paired by seed at identical workload.** `engine_compare.py` pairs vLLM and
SGLang rows only when hardware class, dataset + prompt digest, ISL, OSL, concurrency, range ratio, chat template,
ignore_eos and num_prompts all match, and the same seed (=> identical request stream) exists on both sides. A pair
is GATED (no ratio printed) unless both point-runs are valid, both runs' correctness probes passed, and model pins
match. Ratios use the article's "advantage" convention; CIs are bootstrap-of-means and paired log-ratio t-CIs;
n < 3 is labelled descriptive. The article's implied vLLM/SGLang ratio is shown as a reference column only.

**D-023 (2026-11-10) Static flag preflight against pinned engine sources.** With $10 total, a server dying on a
mistyped flag is the cheapest avoidable loss. `flagcheck.py` extracts flag names with `ast` from vLLM v0.33.0
(`arg_utils.py`, `entrypoints/common/cli_args.py`; `SpeculativeConfig` fields for `--speculative-config` JSON) and
SGLang v0.5.23 (`arg_groups/fields/*.py` field names, `cli_name`, `aliases`), commits snapshots with per-file sha256
(`configs/engine_flags/`), and the runner refuses to start a server whose argv (incl. `--extra-server-arg`) has an
unknown flag. Result on 2026-11-10: all four real engine configs clean (344 vLLM / 562 SGLang flags known). Limits:
static existence only — it cannot catch invalid *values* or runtime incompatibilities (that is what E1 smoke is for).

**D-024 (2026-11-10) The paid worker body is CPU-rehearsed.** `_guarded`/`_run_job` moved from `modal_bridge/app.py` into
`g4bench/bridge_jobs.py` (app.py keeps only Volume reload/commit). `tests/test_bridge_rehearsal.py` drives the real
HTTP API → ledger → worker body → runner on the mock engine → poll → settle path, so job-spec/RunOptions mismatches
and deadline/wall-budget handling are exercised for free. Test-only allowlist overrides live in `BridgeConfig`;
production allowlists are unchanged.

**D-025 (2026-11-10) Record the engine's effective configuration per point.** Source reading (vLLM v0.33.0
`Gemma4Config.verify_and_update_config`): because Gemma 4 mixes head_dim 256 (sliding) and 512 (global), vLLM uses
FA4 for all layers when FA4 is available, otherwise **forces TRITON_ATTN**; the MTP proposer carries the target's
backend into draft layers. Which branch fires on sm_120 (RTX PRO 6000) is unknown until E1, and the backend
plausibly dominates the vLLM-vs-Backboard gap, so `serverlog.py` parses (verbatim format strings from the pinned
sources) the backend decision, selected attention backend, KV-cache tokens and max concurrency, CUDA-graph capture
time/memory, and SGLang's effective `server_args` (attention backend, effective speculative algorithm — e.g. NEXTN
vs FROZEN_KV_MTP — cuda_graph_max_bs, max_running_requests). Results schema v3 adds `eff_*` columns; each point
writes `effective_config.json`; reports show an "Effective engine configuration" table. Absent lines stay `None`.
No flags were changed: the article says default recipes, and both engines select backends automatically.

**D-026 (2026-11-10) Paid steps run only through a gated campaign.** `g4bench/campaign.py` +
`modal_bridge/campaign.py` execute `configs/campaigns/*.yaml` (e1-smoke, e2-pairs) strictly sequentially. Default is
plan-only; `--execute` re-checks before *every* step that `/health` is ok with the ledger reachable, that both
`/health` and `/budget` report exactly the approved limits (cap $10, per-job ≤ $2, max_active 1, ledger disclaimer
present), that no job is active and that `available_usd` covers the step; stops at the first failed gate or failed
expectation (selftest rc 0; prefetch coding digest = InferenceX; all points valid); cancels past `max_wait_s` but keeps
polling until termination is confirmed; logs every step to `results/budget_ledger.jsonl`. Verified live on
2026-11-10: against the undeployed bridge it refused at `/health` (404) before any POST — $0 spent.

**D-027 (2026-11-10) A Backboard-independent target for the open baselines.** Dividing the article's Backboard/SGLang
ratio table by its Backboard/vLLM table cancels Backboard and yields the article's implied vLLM-vs-SGLang advantage
for all 13 metrics at all 8 points (incl. mean/median metrics never published in absolute form), with ±0.005
rounding bounds per published ratio. Matches the absolute table where both exist (1K/c1 throughput 1.523 vs
186.8/122.6 = 1.524). `engine_compare` now reports it per metric plus a descriptive `consistent_with_article` flag
(our bootstrap CI overlaps the article bounds). Table: docs/analysis/article_vllm_vs_sglang.md. Finding: the
article's SGLang gap is a decode (TPOT) gap; p99 TTFT is at parity (0.97–1.06) at 8K.

**D-028 (2026-11-10) Step timeouts never outlive the wall budget.** Server-start (1800 s) and per-point client
(3600 s) timeouts exceeded the 876 s Modal timeout a $2 reservation buys, so a slow point would have been killed by
Modal before results/report were written (paid data lost). The runner now bounds each step by the remaining wall
budget (min 30 s); on the bridge that budget ends ≥ 180 s before the Modal timeout, so slow points are cut, recorded
invalid, and the run still returns its results.

**D-029 (2026-11-10) EOS-honouring real-prompt arm implemented.** `speedbench-coding-eos` (same 80 prompts, `ignore_eos:
false`, max 1024 tokens). Keys get a `+eos` suffix and point ids `-eos`, so it can never pool with the ignore-EOS
arm; validation accepts outputs in [1, OSL]. Throughput is not comparable across the two arms (different decode work);
the readout is acceptance length and TPOT. Planned only — not in the $10 campaigns (est. $5.78 seed 0, both engines,
pessimistic since outputs are assumed full length).

**D-030 (2026-11-10) Independent review of the budget path; 8 findings fixed before any deploy.** A fresh reviewer
(not anchored on my reasoning) read the ledger, API, Modal wiring and campaign against the modal 1.6.3 source. I
verified each claim in the SDK source, then fixed:
1. *High*: `FunctionTimeoutError` and `OutputExpiredError` subclass `modal.exception.TimeoutError`, so my poll mapped
   timed-out/expired jobs to "running" forever (holding $2 and blocking all launches). Now most-specific first.
2. *High*: cancelled/crashed/OOM calls return a non-success result raised as RemoteError/ExecutionError/
   InternalFailure — now terminal ("failed"); cancel uses `terminate_containers=True`; the worker no longer swallows
   BaseException (cancellation propagates).
3. *High*: bridge/ledger container spend was an unmetered $0.50 set-aside. Every request now touches the ledger; warm
   time is metered (gaps > 60 s count as one window) and committed as max(reserve, metered); the web function has
   `max_containers=1`; the campaign stops polling 30 min after a job's deadline.
4. *Med-high*: any spawn exception released the reservation although Modal may have accepted the call. Now only
   pre-RPC `ValueError` releases; anything else stays held and becomes an orphan.
5. *Med*: a failed attach left a running job without call id. Attach is retried; failures return the call id; orphans
   (reserved, no call id) settle at their full reservation once their deadline has passed.
6. *Med*: Modal's timeout starts at input start, so queueing/preemption could push real cost past the reservation,
   and non-finished outcomes settled at min(now, latest). Now workers refuse to start after
   `start_deadline = reserved_at + 900 s` (so start + timeout ≤ `deadline_unix`), arm a hard `os._exit` at
   `deadline_unix`, GPU `startup_timeout=900`, CPU `startup_timeout=600`; every non-finished outcome settles at the
   worst case.
7. *Low*: stale-lock stealing was not compare-and-swap. Each save now claims `ledger/commit/<version>` with atomic
   put-if-absent (stale claims older than the lease are recoverable).
8. *Low*: reservations could exceed `max_usd` by < 1 cent through rounding — now capped at the requested ceiling.
Not fixable in this design and documented: workers can still be invoked directly with a workspace token, bypassing the
ledger; the ledger remains an estimate that gates bridge launches only.

**D-031 (2026-11-10) Worker verified under engine-image conditions; a crash-every-job bug fixed.** Both pinned engine
images run Python 3.12 (read from their registry image configs: vLLM CUDA 13.0.2 with `TORCH_CUDA_ARCH_LIST` incl.
12.0 = sm_120; SGLang CUDA 13.0.3). Our dev venv is 3.13. Running the bridge worker body on Python 3.12 with only
engine-shipped packages (`scripts/check_worker_env.sh`) exposed that `GpuSampler(threading.Thread)` stored its event as
`self._stop`, shadowing `Thread._stop()` used by `join()` on ≤ 3.12 → `TypeError` on every point, i.e. every paid GPU
job would have failed. Fixed (`_halt`) with an AST guard test that fails on the old code. The same check confirms no
fastapi/pyarrow/httpx/modal/pydantic/pytest import on the worker path. New E1 risk recorded: Modal's host driver must
support CUDA 13 (metadata captures `driver_version`).

**D-032 (2026-11-10) First-order gap attribution in every exact-hardware report.** `gap.py`: TPOT ≈ step/AL ⇒ step =
TPOT × AL; bandwidth efficiency = roofline step ÷ measured step; AL needed to hit the article's mean TPOT (= 1000 /
its speed-per-user) at our step time, flagged if above draft_tokens + 1; or the step speed-up needed at our AL. Shown
against the same engine's article numbers and Backboard's. Purpose: decide after E1 whether optimization effort
belongs in drafting/acceptance or in kernels/scheduling. Diagnostic only; assumptions printed with every table.

**D-033 (2026-11-10) Bridge URL and approval variables aligned with the configured environment.** The cloud
environment's network secret is bound to `<workspace>--g4repro-bridge.us-east.modal.run` (regional host), not the
unregional host the client defaulted to; both return 404 today (not deployed). The client now defaults to the
regional host (`G4_BRIDGE_URL` still wins once visible after a session restart). The environment also defines
`G4_PROJECT_BUDGET_USD` and `G4_JOB_CAP_USD`: the campaign uses them to *tighten* (never relax) the limits it requires
the server to report, and the bridge accepts them as deploy-time aliases of `G4_PROJECT_CAP_USD` / `G4_PER_JOB_CAP_USD`.

**D-034 (2026-11-10) Optimization arms prepared; every config carries role + admissibility.** From the pinned sources:
vLLM v0.33.0 `SpeculativeConfig` exposes `num_speculative_tokens_per_batch_size` (depth by batch size) and
`draft_lm_head_quantization` ("fp8"/"nvfp4", MRv2 only; "the target verifies with its own head, so only the acceptance
length can change") — the closest open analogue of Backboard's 4-bit shortlist; SGLang v0.5.23 exposes
`--speculative-adaptive` and draft-model quantization. New flag-checked configs: vLLM mtp2/3/5/6 (E5b depth sweep),
mtp-bsched (E6, schedule to be re-derived from E5b), mtp4-draftfp8 (E6, class **b**); SGLang nextn3/nextn4 (E5b),
nextn-adaptive (E6). All configs now declare `role` (only the two article recipes are `baseline`) and
`admissibility` (a/b/c). Analysis compares only baselines with the article's open-engine numbers and pairs only
baselines in vLLM-vs-SGLang; variants are compared with Backboard only, labelled by class; class c never. All are on
the bridge allowlist; none is scheduled under the current $10 approval (each single-point job ≈ $0.8–1.0).

**D-035 (2026-11-10) Free `/estimate` preflight before any spend.** `campaign.run(execute=True)` now begins with a
free preflight: `/health` + `/budget` must show the approved limits, then every job step is sent to `/estimate`; the
campaign refuses unless each fits one per-job reservation, its worst case ≤ the approved per-job cap, and the largest
worst case ≤ `available_usd`. Order on a live bridge: verify → estimate → CPU selftest → prefetch → vLLM smoke →
SGLang smoke (same `smoke` matrix for both engines = comparable pair).

**D-036 (2026-11-10) `agent/gemma4-repro` retired; deploy from `main`.** The working branch was replayed with
attribution into PRs #2–#4 (all 23 commits patch-equivalent), merged, and deleted by the owner; it must never be
recreated. BOOTSTRAP.md now clones `main` (D-002 above is historical). Verified offline: a plain clone lands on `main`,
`pip install -e ".[bridge]"` succeeds, `modal_bridge/app.py` loads with cap $10 / per-job $2 / max_active 1 / GPU
timeout 876 s, and `modal workspace proxy-tokens` exists in modal 1.6.3.

**D-037 (2026-11-10) Paid runs must come back complete.** A real-HTTP rehearsal (uvicorn-served bridge API, fake Modal
backend, the exact `client.py verify` / `campaign.py e1-smoke --execute` commands) showed that `client.py fetch` copied
only manifest/metadata/results.csv/results.jsonl and the campaign fetched nothing — REPORT.md, correctness.json and
per-point server logs / effective config / raw InferenceX JSON would have stayed on the Modal volume. The bridge now
lists a run's files (`GET /runfiles/{run_id}`, path-traversal guarded), `fetch` downloads all of them (never
overwriting local files; > 20 MB skipped and reported), and the campaign fetches every finished job's run
automatically. `run-file-listing` is a required `/health` feature, so a bridge deployed before this change is refused
until redeployed. The campaign CLI also gains `--ledger` / `--fetch-dir`, so rehearsals no longer write to the real
`results/budget_ledger.jsonl`.

**D-038 (2026-11-11) Make the pinned engine images deployable without changing their Python.** The first `modal deploy`
(modal 1.6.3, Colab) failed for `run_vllm`: "unable to determine the version of Python installed in the Image".
`from_registry` without `add_python` requires `python` + `pip` on PATH. Source check of the pinned images: vLLM
v0.33.0's runtime stage installs vLLM into system Python 3.12 (deadsnakes; `python3` via update-alternatives, pip via
get-pip) and ships no bare `python`; SGLang v0.5.23 already has `python`/`pip` in its venv `/opt/sglang/bin` (first on
PATH). Fix: `g4bench.image_setup.python_setup_commands(engine)` is passed as `setup_dockerfile_commands` (inserted right
after `FROM`, before Modal's setup): if `python` is missing it is symlinked to the image's own `python3`; an existing
`python` is never touched; the step then asserts Python 3.12, `find_spec(engine)` and `python -m pip`, so a broken
interpreter fails at image build, not on a paid GPU start. `add_python` was rejected: it would put a second interpreter
first on PATH that cannot import the engine. Tests run the exact RUN line under /bin/sh with an isolated PATH.

**D-039 (2026-11-11) The HTTPS bridge is removed; paid runs launch directly from the owner's machine.** The bridge
existed only because the earlier cloud agent could reach Modal over HTTPS but not its gRPC control plane (D-010,
now superseded). The owner now runs the agent locally with Modal credentials in the untracked `.env` and asked for the
bridge to be eliminated. Replacement: `modal_app.py` (same digest-pinned images, Volumes, resource limits,
`retries=0`, `max_containers=1`, start-deadline + hard-kill guard, the D-038 image setup) run as an **ephemeral** `app.run()` per launch, so
nothing stays deployed; `spend.py` keeps the $10 / $2 caps locally — worst case reserved in
`results/budget_ledger.jsonl` before the call, the Modal timeout derived from it, wall-clock settlement afterwards —
and, new, reconciles against Modal's authoritative per-app billing (`modal billing report`, filtered to `g4repro*`)
before every launch, counting the larger figure, plus Volume storage accrual. Removed: `modal_bridge/`,
`bridge_api.py`, `budget.py` (Dict ledger), `campaign.py` and `configs/campaigns/` (HTTP campaign runner), their tests.
Run evidence now comes back as the worker's return value (all files ≤ 20 MB) instead of a fetch endpoint. Platform
reserve/metering (D-030 #3) is obsolete: there is no always-on web container. Trade-off: the per-launch billing lag
means a just-finished launch is counted at our estimate until Modal bills it; the gate takes the larger of the two.

**D-040 (2026-11-11) Budget plan: one server start per job, no separate SGLang smoke.** The E1 vLLM smoke
(`20261111T012903Z-…-d6761acd-a22c`, settled ≈ $0.34) showed the cost structure on RTX PRO 6000: ≈ 6.5¢/GPU-minute
at container limits, ≈ 3.5 min for server start + correctness probes, and an engine image pull (22 min for vLLM) that
happens on Modal's builder before any GPU container (≈ $0.06 billed to the app). With fresh servers per point the
16-point seed-0 matrix does not fit the remaining ≈ $9.5. New matrix `backboard-repro-shared` = `backboard-repro`
with `seeds: [0]` and `fresh_server_per_point: false`; each job selects 2–3 points with `--only` so it fits the 876 s
timeout of a $2 reservation. Points sharing a server are not independent restarts; all results are n = 1,
descriptive only (EXPERIMENTS criterion 2, unchanged). SGLang skips the smoke matrix and starts directly on
`1024x1,1024x4`: a broken engine fails in its first minutes either way, saving ≈ $0.5. Also fixed from the smoke
run: the per-run report compared the smoke's OSL-256 throughput with the article's OSL-1024 numbers (now excluded),
and Modal runs recorded no code SHA (now `G4_CODE_SHA`).

**D-041 (2026-11-11) Per-job cap raised from $2 to $3 (owner approval) for 8192×64.** 768 requests at 8K input need
≈ 13 min (vLLM) / ≈ 18 min (SGLang) of runtime plus ≈ 3 min of server start — more than the 876 s timeout a $2
reservation buys. Ahamed approved a $3 per-job cap; the $10 project cap is unchanged. Ran vLLM at `--max-usd 2.5`
(1335 s timeout; used 796 s, settled ≈ $0.94) and SGLang at `--max-usd 3` (1794 s; used 1109 s, ≈ $1.29). Other jobs
keep the $2 default.

**D-042 (2026-11-11) E5b depth sweep at 8192×4: a small, noisy gain at most.** Approved by the owner (≈ $1.5).
vLLM 0.33.0, seed 0, one shared-server job per depth, otherwise identical to the baseline recipe:

| draft tokens | config | output tok/s | vs baseline (MTP 4) | acceptance length | p99 TPOT ms | vs Backboard (529.6) |
|---:|---|---:|---:|---:|---:|---:|
| 3 | vllm-0.33.0-mtp3 | 424.4 | +5.4 % | 3.50 | – | 0.801 |
| 4 | vllm-0.33.0-mtp4 (baseline) | 402.6 | – | 4.06 | – | 0.760 |
| 5 | vllm-0.33.0-mtp5 | 423.9 | +5.3 % | 4.56 | 12.7 | 0.800 |
| 6 | vllm-0.33.0-mtp6 | 437.4 | +8.6 % | 4.99 | 13.2 | 0.826 |

Acceptance length rises with depth as expected (3.50 → 4.99), but throughput is non-monotonic: depth 3 and 5 both beat
the baseline depth 4 by ≈ 5 %. With n = 1 per config that pattern is best explained by ≈ 5 % run-to-run variation
(the baseline run may have been low), so the only defensible reading is "depth 6 is not worse and may help by up to
≈ 9 %". It closes at most a quarter of the gap to Backboard (76 % → 83 %); speculation depth alone does not explain
Backboard's lead. Not a claim; separating the effect from noise needs ≥ 3 runs of depth 4 and 6 (≈ $2.5).

*Amendment to D-042 (2026-11-11, after a second run of depth 4 and depth 6, approved by the owner, ≈ $0.70):*
depth 4 = 402.6 / 402.0 tok/s, depth 6 = 437.4 / 434.2 tok/s (acceptance length 4.06 / 4.03 and 4.99 / 4.98). Runs
repeat within ≈ 1 %, so the earlier "within noise" reading is withdrawn: **depth 6 is ≈ 8 % faster than the article's
depth 4 at 8192×4** (82 % of Backboard instead of 76 %). n = 2 per config — still below the n = 3 claim threshold,
and only one workload; the depth-3/5 +5 % results remain n = 1. The budget gate refused the depth-6 repeat at the
default $2 reservation (committed $8.18 + $2 > $10); it ran at `--max-usd 1.75` (647 s timeout, used ≈ 270 s).

**D-043 (2026-11-11) Chunked prefill: smoother tokens, little throughput.** Run-data analysis (ITL distributions) showed
≈ 22 % of vLLM's decode wall time at 8K/c4 is spent stalled behind 8K prompts prefilled in one ≈ 600 ms step
(`max_num_batched_tokens=8192`); 52 % at 8K/c16, 71 % at 8K/c64. `--max-num-batched-tokens 2048` is refused by
v0.33.0 for Gemma 4 ("Chunked MM input disabled but max_tokens_per_mm_item (2496) …"), so 3072 was used. Depth 6 at
8K/c4: 441.7 vs 435.8 tok/s (+1.4 %), p99 ITL 584 → 274 ms, stall share 23 → 21 %: the prefill work is the same,
only spread out. Kept in the combined candidate for its latency effect.

**D-044 (2026-11-11) Split-K attention via CUDA-graph capture sizes: the first step-speed win.** From the pinned
vLLM v0.33.0 source (two independent agent reads): Triton unified attention uses its split-K ("3D") kernel only
when `num_seqs * cdiv(q_len, BLOCK_Q) <= seq_threshold_3D` and `q_len <= 8`; the threshold is `128 // num_kv_heads`
(= 16 from Gemma 4's model-level 8 KV heads) snapped to the nearest CUDA-graph capture size (ties → smaller). Gemma
4's 8 global layers have 1 KV head and head_dim 512, so BLOCK_Q = 1 and 4 requests × (k+1) > 16 tokens fall back to
the 2D kernel (≈ 24 CTAs on ≈ 188 SMs over 8K×512 KV) — which explains the depth 3 → 4 step-time cliff. Capture
sizes `1 2 28 32 …` (all of 3..27 dropped) raise the threshold to 28 = 4 × 7. Lossless (same math, FP reduction order
only). Depth 6 at 8K/c4: **472.4 vs 435.8 tok/s (+8.4 %)**, median ITL 30.1 → 26.6 ms. Depth 7 cannot benefit at c4
(threshold 32 would require dropping size 1). Combined candidate `vllm-0.33.0-mtp6-splitk-chunk3072`: **1K/c4
765.1 tok/s (98 % of Backboard's 779.4; +18 % over the article's vLLM)**, **8K/c4 488.3 (92 % of 529.6; +23 %)**. n = 1.

**D-045 (2026-11-11) Project cap raised to $20 by the agent under the owner's open authorization.** The owner wrote
that more Modal credits beyond the $10 allowance may be used if necessary, without waste, and started an
unbounded iteration loop. An unbounded loop needs a hard stop, so `spend.Caps` is set to $20 total / $3 per job;
the environment can only tighten it. Past $20 the loop does CPU-side work only and asks for a new figure.

**D-046 (2026-11-11) Engine-image patches; the sm_120 tile patch is retired, the draft-prefill patch is kept.**
Build-time source patches live in `src/g4bench/patches/`; `modal_app.py` applies those named in `VLLM_PATCHES` to a
separate image `g4repro-vllm-patched` (the digest-pinned base stays untouched), a config selects it with
`image_patches: [...]`, and the applied list is recorded as `G4_IMAGE_PATCHES` in run metadata. vLLM v0.33.0, seed 0,
n = 1, compared with the unpatched `vllm-0.33.0-mtp6-splitk-chunk3072` runs (`…-2d15`, `…-d1cf`):
- `sm120_attn_tiles` (Triton prefill tiles for head_dim 256/512 on sm_120, sized to fit ≈ 99 KB shared memory;
  `…-110f`): 8K/c1 182.5 vs 182.3 tok/s, mean TTFT 662 vs 662 ms; 8K/c16 860.0 vs 863.1 tok/s. No measurable gain:
  prefill is GEMM-dominated, not attention-tile bound. Retired.
- `mtp_draft_prefill_last` (the Gemma 4 MTP draft runs prefill only on each request's last prefill token; lossless
  because the Gemma 4 draft is KV-shared with the target and row-wise, and the target verifies every token;
  config `vllm-0.33.0-mtp6-splitk-chunk3072-draftlast`, `…-930d`): 8K/c16 **885.3 vs 863.1 tok/s (+2.6 %)**, 8K/c1
  mean TTFT 638 vs 662 ms (−3.6 %), 183.3 vs 182.3 tok/s; acceptance length 5.09 vs 5.10 (unchanged). Kept. Descriptive (n=1).

**D-047 (2026-11-11) Depth by batch size (E6a) and the single-config candidate E6b.** `vllm-0.33.0-dyn-splitk-chunk3072`
(`…-85e7`) = the D-044 combined config with `num_speculative_tokens_per_batch_size` [[1,8,6],[9,24,5],[25,4096,4]]:
1K/c64 3674 tok/s (fixed depth 6: 3603; baseline depth 4: 3668), 8K/c4 481.0 (488.3), 8K/c16 848.3 (863.1). Depth 4
helps at c64; depth 5 hurts at c16 (acceptance length 4.63 vs 5.10). E6b therefore uses [[1,32,6],[33,4096,4]].
A throughput model (step-level replay of the chunked-prefill scheduler, calibrated on the measured runs; RMS error
0.35 %, validation max 2.4 %) attributes 31 / 55 / 69 % of wall time to prefill at 8K c4 / c16 / c64 and
predicts that ≈ 10–15 % faster prefill is the single biggest remaining lever. E6b = `vllm-0.33.0-best-e6b`: depth
schedule + split-K capture sizes + 3072-token chunks + `mtp_draft_prefill_last`. Descriptive (n=1, seed 0):

| workload | E6b tok/s | vs Backboard | previous best / baseline | run |
|---|---:|---:|---|---|
| 1024×1 | pending | pending | | |
| 1024×4 | pending | pending | | |
| 1024×16 | pending | pending | | |
| 1024×64 | 3710 | 0.971 | 3603 (D-044 combined), 3674 (E6a) | `…-d129` |
| 8192×1 | pending | pending | | |
| 8192×4 | pending | pending | | |
| 8192×16 | pending | pending | | |
| 8192×64 | 1149 | 0.916 | 1098 (D-044 combined, `…-aee1`) | `…-d129` |

**D-048 (2026-11-11) SGLang 0.5.23 slowness diagnosed from source; no GPU run yet.** SGLang's steps are ≈ 1.6× slower
than vLLM's because target-verify attention on sm_120 uses the Triton extend kernel without KV splitting (split-KV
verify is AMD-only in v0.5.23), so the 8 global layers (head_dim 512, 1 KV head) scan the 8K prefix serially — the
same problem vLLM's split-K kernel fixes (D-044). No server flag changes this. Prepared, not run:
`sglang-0.5.23-nextn-fair64` (`--max-running-requests 64`, `--cuda-graph-max-bs-decode 64`; makes c64 like-for-like
after SGLang's automatic cap of 48) and `sglang-0.5.23-nextn-fast` (+ `--triton-attention-num-kv-splits 32`, which
reaches the draft decode kernel only).

**D-049 (2026-11-11) Profiling mode for E5g.** `g4bench modal job ... --profile-key <ISLxconc>` starts vLLM v0.33.0
with `--profiler-config` (torch profiler, CPU + CUDA, gzip) and `VLLM_CUSTOM_SCOPES_FOR_PROFILING=1`, captures a short
steady-state window via `/start_profile` / `/stop_profile`, and writes `points/<id>/profile_summary.{json,md}`
(kernel classes, per-step summary). Profiled rows carry profiler overhead and are excluded from analysis and
comparisons. First run (E6b at 8192×4) pending.

**D-050 (2026-11-11) Single config E6b measured at all 8 workloads; profile says the rest is GEMM kernels.**
`vllm-0.33.0-best-e6b` (vLLM v0.33.0 + `mtp_draft_prefill_last` patch, split-K capture sizes, 3072-token chunks,
depth 6 up to 32 running requests then 4), seed 0, n = 1, output tok/s vs Backboard: 1K c1/c4/c16/c64 223.9 / 768.1 /
2106 / 3710 (1.100 / 0.986 / 1.012 / 0.971); 8K c1/c4/c16/c64 183.2 / 497.8 / 853.9 / 1149 (1.023 / 0.940 / 0.888 /
0.916). 5/8 within 5 %, geometric mean 0.977 (article vLLM recipe: 0.881). 8K/c16 is 3.5 % below the same settings
without a depth schedule (885.3, `…-930d`) — schedule cost (partial graphs) or noise; untested. Torch profile of E6b at
8K/c4 (4 s window, `…-0d48`): GPU idle 0.5 %; GEMM 74 %, attention 20 %, norm/activation 5 %, sampling and
spec-decode bookkeeping < 1 %; GEMMs are CUTLASS sm_80 mma.sync kernels from cuBLAS (decode-sized ones as 32×32 WMMA
tiles, 16 % of GPU time); decode step ≈ 19.8 ms GPU, prefill-carrying step ≈ 211 ms. sm_120 has no wgmma/tcgen05, so
no flag selects faster BF16 GEMMs. Conclusion: the remaining 8K gap (6–11 %) is GEMM kernel efficiency — custom kernel
work, outside what configuration can reach. Paid experimentation paused here (≈ $13 billed of the $20 cap).


**D-051 (2026-11-11) Kernel lab; project cap raised to $50 by the agent.** The owner's go-ahead to "use a lot more"
Modal credits was open-ended, so the agent set `spend.Caps` to $50 total / $3 per job (the environment can only
tighten it). New command `g4bench modal script --path <repo script>` runs a repo script on the RTX PRO 6000 inside the
digest-pinned vLLM image, under the same spend gate (default reservation $1.75), and copies whatever the script writes to
`$G4_OUT_DIR` into `results/kernels/<run id>/` (JSON + stdout/stderr). Each kernel bench below cost ≈ $0.08–0.12.

**D-052 (2026-11-11) BF16 GEMMs are already at the hardware limit; cuBLAS 13.10 ships as an image build step.**
`kernels/gemm_bench.py`, Gemma 4 12B linear shapes (sliding/global qkv and o, MLP gate_up/down) × per-forward counts:
- First run (`…-gemm_bench-c46eac`) reported decode linears *below* the weight-read floor (11.2 ms/forward at M=28 vs
  14.9 ms): the 128 MB L2 held the weights between repeats. Withdrawn as an artefact.
- Cold-weight rerun (`…-gemm_bench-025434`, weights rotated through > L2): decode linears 15.3 / 15.8 / 17.1 ms/forward
  at M = 4 / 28 / 112 vs the **14.9 ms weight-read floor at the measured 1.46 TB/s** copy bandwidth (94 % at M=28);
  prefill per-shape 341–427 TFLOPS at M=3072 vs ≈ 415–440 TFLOPS measured peak (peak probe 406.5 / 415.1 in the
  two runs). `torch.compile` max-autotune never won (0.51–1.01× of cuBLAS). n=1 per run.
- CuTe DSL sm_120 dense GEMM (NVIDIA/cutlass v4.8.0 `blackwell_geforce` example, 6 tile configs, `…-bench-916b9a`):
  prefill linears at M=3072 **169.4 vs 168.0 ms/forward** (0.992×); per shape 0.94–1.06× at M=3072, 0.76–1.05× at
  M=1024, 0.71–1.02× at M=28. Best-of-both 166.7 ms (−0.8 %): not worth a custom path.
- cuBLAS 13.10.0.20 vs the image's 13.1 (cublasLt 131000 vs 130101), interleaved stock/new/stock/new, n=2 each
  (`…-cublas_upgrade_bench-af4bfc`; the first run `…-7d63ed` was not interleaved): linear ms/forward 15.77 → 15.46
  (**−2.0 %**, M=28), 61.98 → 60.07 (**−3.1 %**, M=1024), 170.18 → 166.92 (**−1.9 %**, M=3072). Shipped as a build
  step of the patched image (`g4repro-vllm-patched`); lossless (same math, different kernels).
- Spec correction (research): the RTX PRO 6000 Blackwell **Server Edition** is rated **1597 GB/s**; the measured 1.46 TB/s
  copy is 91 % of that. Conclusion: there is no GEMM headroom
  left that a kernel can buy on this GPU at BF16.

**D-053 (2026-11-11) Attention: one shared split-K threshold misroutes Gemma 4's two layer types.**
`kernels/attn_bench.py` (`…-attn_bench-c6c4db`, vLLM 0.33.0 Triton unified attention, verify q_len 7, cold KV, CUDA
graphs; numerics vs reference ≤ 0.008 abs) per forward (8 global + 40 sliding layers): the threshold that suits the
1-KV-head global layers is wrong for the sliding layers and vice versa. 8K/c16: **10.2 ms with vLLM's routing vs 6.9 ms
with the best path per group** (floor 5.2 ms; global 577 → 387 µs/layer, sliding 141 → 94 µs/layer); 8K/c64 28.5 vs 26.9
ms; 1K/c16 4.87 vs 4.35 ms; c1/c4 already optimal. Packed-row (MQA-style) split-K kernels for the global layers save
only **≈ 0.5 ms/forward at 8K/c4** (2.67 → 2.16 ms): repeat reads of the single KV head are absorbed by L2. Patch
`splitk_per_group` gives each attention group its own `seq_threshold_3D` (`G4_SPLITK_T_FULL=112`,
`G4_SPLITK_T_SLIDING=8`); lossless (FP reduction order only). The packed-row kernel is not shipped.

**D-054 (2026-11-11) Draft head: the 12B assistant has no centroid shortlist; quantized scan + exact BF16 rescore.**
The downloaded `google/gemma-4-12B-it-assistant` config has `use_ordered_embeddings: false`: the head is the dense,
tied 262144 × 1024 BF16 embedding (SOURCES corrected; only the E2B/E4B assistants use ordered embeddings), i.e. 512 MB
read per draft token: 369 µs at M=4, 395 µs at M=28, i.e. **≈ 2.2 ms per depth-6 draft step** at c4. `kernels/draft_head_scan.py`
(`…-draft_head_scan-556087`, real weights, 4096 synthetic activation rows, CUDA graphs) scans a quantized copy, keeps the
top k and rescores those k exactly in BF16: at M=28, k=32, BF16 head 395 µs → **FP8 fused 212 µs (1.86×)**, **INT4
fused 137 µs (2.9×)**. Top-1 recall vs the BF16 argmax ("realistic" set, k=8/32/64): FP8 0.992 / 0.996 / 0.996, INT4
0.975 / 0.981 / 0.984 (random set: FP8 1.000). Neither meets the 0.999 bench criterion, so acceptance in serving is
the real test; a miss only lowers acceptance, never target output. Patch `mtp_draft_head_q` (`G4_DRAFT_HEAD_Q=fp8|int4`,
`G4_DRAFT_HEAD_K=32`) replaces the head + argmax; draft precision lowered → **class b** (D-011). Bug: during CUDA-graph
capture the draft sees uninitialised hidden states, all scores were NaN and the 2^30 "no candidate" sentinel became the
draft token, tripping the embedding index assert (E8a run `…-4b53`, 0/2 valid). Fix `ab3fc93`: fall back to a valid
token when all scores are NaN.

**D-055 (2026-11-11) Third-party drafters: DSpark is slower; the rest are not run.** `vllm-0.33.0-dspark-block7`
(`deepseek-ai/dspark_gemma4_12b_block7`, 3.43 B params, block 7, own non-causal KV cache, no image patch;
`…-dspark-block7-…-cc48`, seed 0, n=1, probes 8/8): **1K/c1 211.0 vs 223.9 tok/s** (E6b), **8K/c4 443.4 vs 497.8**,
acceptance length 5.06 / 4.55 — at least as long as Google's assistant on random-token prompts, but (likely) its 5 full-width
layers cost more per step than the extra acceptance saves; ≈ $0.42 billed. `vllm-0.33.0-dloopmtp-e6b` (`naver-ai/DLoop-MTP-gemma-4-12B-it`,
same architecture as Google's assistant) prepared, not run. DFlash and Eagle3 rejected without a run: the DSpark paper
(same authors) reports both below DSpark in accepted length on Gemma 4 12B (e.g. 5.45 / 5.87 vs 6.05 on GSM8K), and DSpark
already loses. Neither the DSpark nor the DLoop HF repo states a licence (no model card / licence tag); both are labelled
"third-party drafter", never a like-for-like recipe change.

**D-056 (2026-11-11) Combined config E8a beats Backboard on the geometric mean (n=1).** `vllm-0.33.0-best-e8-all` =
E6b + `splitk_per_group` (D-053) + `mtp_draft_head_q` FP8, k=32 (D-054) + cuBLAS 13.10 (D-052), patched image, class b,
seed 0, n=1, probes 8/8 on every run. Output tok/s (vs Backboard; vs E6b):

| workload | E8a tok/s | vs Backboard | vs E6b | acceptance length | run |
|---|---:|---:|---:|---:|---|
| 1024×1 | 238.1 | 1.17 | +6.3 % | 5.32 | `…-d1a0` |
| 1024×4 | 813.1 | 1.04 | +5.9 % | 5.11 | `…-d1a0` |
| 1024×16 | 2235 | 1.07 | +6.1 % | 5.15 | `…-d1a0` |
| 1024×64 | 3916 | 1.02 | +5.6 % | 4.20 | `…-400b` |
| 8192×1 | 197.5 | 1.10 | +7.8 % | 5.03 | `…-d1a0` |
| 8192×4 | 528.0 | 0.997 | +6.1 % | 5.25 | `…-0b3c` |
| 8192×16 | 945.8 | 0.983 | +10.8 % | 5.09 | `…-0b3c` |
| 8192×64 | 1170 | 0.933 | +1.8 % | 4.12 | `…-e87a` |

**Geometric mean 1.039** (E6b 0.977); 7/8 within 5 % or above, 5/8 above. Not a "beats Backboard" claim under the
pre-registered criteria (n=1, class b). 8K/c64 remains prefill-bound (D-047 model: ≈ 69 % of wall time), which no change
here touches. Spend ≈ $15.7 billed in total of the $50 cap.

**D-057 (2026-11-11) Two concurrent paid launches.** The spend gate (`spend.MAX_OPEN_LAUNCHES = 2`, `6093a88`) now
admits a second launch while one is open. Every open reservation already counts at its worst case in `committed`, so
the $50 total / $3 per-job cap is unchanged; the environment can still only tighten the limit. Used to run repeats
and the E8b/E8c variants side by side.

**D-058 (2026-11-11) Experimental patched image on its own app.** Patches under evaluation go into
`VLLM_PATCHES_EXP` = `VLLM_PATCHES` + `prefill_attn_g4`, built as a separate image on the Modal app
`g4repro-vllm-patched-exp` (`7f76c3d`). A config whose patches are a subset of `VLLM_PATCHES_EXP` but not of
`VLLM_PATCHES` is routed there. The production patched image (`g4repro-vllm-patched`) stays byte-identical, so the
E8a repeat series (D-060) runs on one image throughout.

**D-059 (2026-11-11) Packed-head prefill attention lifts 8K; 8K/c64 still just below Backboard.**
`kernels/prefill_attn.py`: FlashAttention-2-style prefill kernel that packs the (token, head) rows sharing one KV head
into one tile, so each K/V block is read once per group. Bench (`…-prefill_attn_bench-01f4bf`, 8192-token prompt in
3072/3072/2048 chunks, 8 global + 40 sliding layers, CUDA graphs): prefill attention **189.8 → 114.0 ms per prompt**
(global 12.9 → 9.2 ms per layer, **1.40×**, `m64_dv512_n32_d64`; sliding 2.17 → 1.01 ms per layer, **2.15×**, `m64_dv256_n64_d64`); max abs
difference vs vLLM ≤ 0.016, i.e. within BF16 rounding. Patch `prefill_attn_g4` (`G4_PREFILL_ATTN=1`, experimental
image, D-058) sub-batches each attention call: the decode/verify prefix (q_len ≤ 8) goes to vLLM's kernel, prefill
rows to the new kernel. Lossless (FP reduction order only); class unchanged (b, from the draft head). Serving, seed 0,
n=1, probes 8/8, acceptance unchanged:
- E8b `vllm-0.33.0-best-e8-pfa` (E8a + `prefill_attn_g4`): 8K/c4 **525.2**, 8K/c16 **986.0**, 8K/c64 **1226** tok/s
  (0.992 / 1.025 / 0.978 of Backboard; `…-0460`, `…-e5f7`).
- E8c `vllm-0.33.0-best-e8-pfa-c8k` (E8b + 8192-token prefill chunks): 8K/c64 seeds 0/1/2 **1243.4 / 1219.0 /
  1140.6** (mean 1201.0, 0.958, n=3); 8K/c4 seeds 1/2 508.0 / 535.3 (mean 521.6, 0.985, n=2); 8K/c16 seeds 1/2
  1000.0 / 975.7 (mean 987.9, 1.027, n=2).
A step-level model predicted 1.017 (E8b) / 1.045 (E8c) of Backboard at 8K/c64; measured 0.978 (n=1) / 0.958 (n=3).
The model over-predicts by 4–9 %; part of that is hardware spread (D-060).

**D-060 (2026-11-11) Repeats: 1K beats Backboard; 8K is a statistical tie, not a proven win.** E8a
(`vllm-0.33.0-best-e8-all`, production patched image) at all 8 workloads, seeds 0/1/2, n=3, 24/24 points valid.
Mean output tok/s, ratio to Backboard and Student-t 95 % CI of the ratio (t₂ = 4.303):

| workload | seeds 0 / 1 / 2 tok/s | mean | vs Backboard | 95 % CI |
|---|---|---:|---:|---|
| 1024×1 | 238.1 / 233.6 / 232.7 | 234.8 | 1.153 | 111.8–118.8 % |
| 1024×4 | 813.1 / 814.1 / 818.5 | 815.2 | 1.046 | 103.7–105.5 % |
| 1024×16 | 2235 / 2178 / 2207 | 2207 | 1.060 | 102.6–109.4 % |
| 1024×64 | 3916 / 3906 / 3929 | 3917 | 1.025 | 101.7–103.2 % |
| 8192×1 | 197.5 / 184.6 / 198.3 | 193.5 | 1.080 | 97.3–118.7 % |
| 8192×4 | 528.0 / 488.4 / 507.3 | 507.9 | 0.959 | 86.6–105.2 % |
| 8192×16 | 945.8 / 948.5 / 928.7 | 941.0 | 0.978 | 95.1–100.6 % |
| 8192×64 | 1170 / 1083.9 / 1092.6 | 1115.6 | 0.890 | 79.5–98.4 % |

Geometric mean of the means 1.021. All four 1K CIs lie above 100 % of Backboard. At 8K the best measured config per
workload has means 0.958–1.080 (E8a at c1, E8c at c4/c16/c64), with CIs spanning 100 % — and E8c at c4/c16 is n=2.
Cause of the width: Modal assigns a different physical GPU per launch (19 distinct GPU UUIDs over 20 E8 runs), and some
run hot. All three slow 8K/c64 runs are throttled: E8a seed 1 **1083.9** (mean SM clock 2220 MHz, min 2040; GPU
83 °C mean, 88 °C max), E8a seed 2 1092.6 (2217 MHz, 83 °C), E8c seed 2 **1140.6** (2199 MHz, min 1995; 83 °C, 89 °C
max) vs 2369–2386 MHz at 68–70 °C for E8a seed 0 (1170.2) and E8c seed 0 (1243.4). Machine-to-machine spread at 8K is
≈ 5–8 % (also 7.5 % at 8K/c4 without throttling: 528.0 vs 488.4), larger than the remaining 8K gap, so 8K vs Backboard is
a tie within noise. Not run: E8c at 1K, a third E8c run at 8K/c4 and 8K/c16. Spend ≈ $24.1 billed in total of the $50 cap.

**D-061 (2026-11-13) Same-GPU paired benchmarking; every image patch env-gated; cuBLAS chosen per config.**
An independent review found the Backboard comparison confounded by hardware: Modal gives each job a different GPU
(5-15 % spread at 8K, throttling at 83 °C; D-060). Fix: run the article's stock recipe (A) and a candidate (B) in ONE
container on ONE GPU, interleaved, so the GPU term cancels in B/A.
- `mtp_draft_prefill_last` is now gated by `G4_DRAFT_PREFILL_LAST=1` (unset = upstream); `images.ALWAYS_ON_PATCHES` is
  empty. Every config that lists the patch sets the gate; `config.LEGACY_ENV` drops gate values that reproduce the
  pre-gate behaviour from the config hash, so the E6b/E8 series keep their hashes (`e1c2c9a3`, `deccf806`,
  `92b787bd`, `6fd46fcb`, pinned by a test). The production patched image's build steps are unchanged.
- Experimental image (`g4repro-vllm-patched-exp`): cuBLAS 13.10 is no longer upgraded in place but installed with
  `pip --target /opt/cublas-13.10` (`G4_CUBLAS_SIDE`); a config selects it with `G4_CUBLAS: "13.10"` (server gets
  `LD_PRELOAD` of `libcublasLt.so.13` + `libcublas.so.13` and that lib dir on `LD_LIBRARY_PATH`), otherwise it gets the
  stock 13.1. `best-e8-pfa` / `best-e8-pfa-c8k` set it (they ran with the in-place 13.10 before; hash unchanged).
  `best-e8-all` ran on the production image with the in-place 13.10 and does not set it; paired on the experimental
  image it would get 13.1 (the `pair` plan warns). After each server start the runner scans `/proc/<pid>/maps` of the
  server tree for `libcublas*` (`points/*/cublas.json`, `metadata.json` -> `cublas`); a request that was not honoured
  invalidates the point. A stock config is refused on an image with in-place upgrades.
- Paired mode: `g4bench modal pair --a vllm-0.33.0-mtp4 --b <candidate> --matrix backboard-repro-shared --only K
  --repeats n --order ABBA` (plan only unless `--execute`). Per point and repeat: fresh server A, point, stop; fresh
  server B, point, stop (ABBA alternates AB/BA). One run dir per engine plus `results/pairs/<id>/paired.json` (B/A per
  metric, per-segment GPU clock/power/temperature, GPU UUID per pair, container). A config on a non-home image gets
  a distinct config hash (`runtime_image`), so stock runs on the experimental image never pool with stock-image runs.
- `analyze`: geometric-mean B/A with a paired-t 95 % CI on log ratios, and
  `candidate_vs_backboard = (B/A)_paired x (article vLLM / Backboard)` when A is the article recipe.
- Sizing ($3 per-job cap -> 1614 s of wall after headroom; model: 150 s server start + 30 s stop + requests at the
  article's vLLM speed, ×1.10): 8K×64 one pair ≈ 1.67 ks + gates → does NOT fit (needs ≈ $3.46; owner decision);
  8K×16 and 1K×64: 1 pair per job; 8K×1, 8K×4, 1K×1, 1K×4, 1K×16: 2 pairs (ABBA) per job.

**D-062 (2026-11-13) Per-job cap raised from $3 to $4 (owner approval).**
One 8K×64 same-GPU pair needs ≈ 2040 s of wall (two fresh servers, two 640-request points at ~1.1–1.3 ktok/s, the
greedy gate), ≈ $3.5 worst case, which the $3 cap refused (D-061 sizing). The owner approved $4 per job; the $50
project cap and the 2-launch concurrency limit are unchanged. `Caps.from_env` can still only tighten either cap.

**D-063 (2026-11-13) Greedy output-equivalence gate: amended criterion.**
The pre-registered gate (EXPERIMENTS.md, criterion 5) asked for identical greedy token ids against a BF16
no-speculation reference. It was never wired until now (`correctness.py`, `g4bench greedy promote|compare`;
reference `data/greedy_reference/vllm-0.33.0-nospec.json`: 8 prompts, 4 short and 4 synthetic 3K/8K-token, 256
tokens each, token ids plus top-2 logprobs). Result: the unmodified article recipe `vllm-0.33.0-mtp4` already differs
from the reference on 4/8 prompts, because the verify forward runs a different batch shape and kernel path than plain
decoding; exact equality is unattainable for any speculative config in vLLM. Amended criterion, applied identically to
the article recipe and to every candidate: (i) every first divergence is a **near-tie** (reference top-2 logprobs
within 0.25 nats at that position), and (ii) the identical-prefix fraction is not materially below the article
recipe's. Results: `mtp4` 4/8 identical, prefix 0.72; `best-e8-all` 4/8, 0.79; `best-e8-pfa-c8k` 4/8, 0.82;
`best-e8-pfa2-c8k` (inside the paired jobs) 3/8; every divergence of every config is a near-tie. This shows the
custom kernels and FP8 draft scan change outputs only where the target model is itself undecided; it is weaker than
bitwise identity and is reported as such.

**D-064 (2026-11-13) Prefill attention v2 (Gluon `mma_v2`), config `best-e8-pfa2-c8k`.**
v1 (D-059) left prefill attention at ~120 TFLOPS. v2 rewrites both kernels in Gluon for sm_120's `mma.sync`
(FA2 layout, explicit shared-memory staging) and a bench sweep chose `gl_m32_n32_w2x2` for global layers (head_dim
512, 1 KV head) and `gl_m64_n32_w4x1` for sliding layers (head_dim 256, window 1024): ≈ 190.8 / 190.9 TFLOPS; per 8K
prompt (8192-token chunk) prefill attention 191.2 ms upstream → 108.0 ms v1 → 73.1 ms v2. `gl_m64_n16_w4x2_kw8` failed
numerics and is excluded. The patch reads vLLM's paged KV cache directly, runs a per-shape startup check against an
fp32 reference and falls back to v1, then to upstream, with a log line. Config `vllm-0.33.0-best-e8-pfa2-c8k` =
`best-e8-pfa-c8k` with the v2 kernels selected (`G4_PREFILL_ATTN_GLOBAL` / `_SLIDING`).

**D-065 (2026-11-13) Same-GPU paired verdict: one frozen config vs the article recipe at all eight workloads.**
`g4bench modal pair --a vllm-0.33.0-mtp4 --b vllm-0.33.0-best-e8-pfa2-c8k` (D-061), 2 pairs per workload (1 at 8K×64,
which needs the $4 cap, D-062), ABBA, except that 8K×16 and 1K×64 each ran as two one-pair jobs (one AB, one BA). B/A output throughput: 1K×1 1.239, 1K×4 1.267,
1K×16 1.182, 1K×64 1.074, 8K×1 1.272, 8K×4 1.326, 8K×16 1.257, 8K×64 1.162; the two pairs per point agree within
0.1–0.8 %. Scaled by article-vLLM / Backboard: 1.14 / 1.05 / 1.06 / 1.02 / 1.13 / 0.99 / 1.04 / 1.00, geometric mean
1.053. Reading: ahead of Backboard at 6/8 workloads by this estimator, parity at 8K×4 and 8K×64. The estimator's error
is dominated by the single published figures, not by our measurement. Greedy gate inside the pair jobs: candidate
3/8 identical, every divergence a near-tie (D-063). Spend after this series ≈ $36 of the $50 cap.

**D-066 (2026-11-13) Project cap raised from $50 to $100 (owner approval) for TTFT work.**
An independent critic (2026-11-13) accepted the paired arithmetic but flagged one GPU per point, seed-0-only tuning
and evaluation, and worse p90 TTFT than Backboard at 5/8 points. The owner asked to push TTFT and approved $100;
per-job $4 and 2 concurrent launches unchanged. Spend after D-067 ≈ $50.

**D-067 (2026-11-13) TTFT: async scheduling costs one engine step of admission; config `best-e8-pfa2-sync`.**
Per-request data showed the TTFT gap was queueing, not prefill speed: against the article recipe on the same GPU our
median TTFT was already lower; p90 depends on prefill collisions. vLLM 0.33 with `--async-scheduling` runs two batches
in flight, so a request arriving while the engine waits on batch N enters batch N+2 (engine/core.py
`step_with_batch_queue`). Screens on one GPU each vs `best-e8-pfa2-c8k` (E9, `results/pairs/`):
`--no-async-scheduling` p90 TTFT 302 → 219 ms (1K×16) and 600 → 448 ms (1K×64) at unchanged throughput;
`--api-server-count 2` −5 to −7 % p90 TTFT, −0.5 % throughput (not adopted); `--max-num-batched-tokens 16384`
−2 % TTFT, −1 % throughput at 8K×4 (not adopted); `--max-cudagraph-capture-size 1280` no effect. An early-token-copy
patch was not written: the V2 model runner (used by our servers) already copies sampled tokens before drafting.
Config `vllm-0.33.0-best-e8-pfa2-sync` = `best-e8-pfa2-c8k` + `--no-async-scheduling`, paired vs `mtp4` at all 8
workloads (1–2 pairs each): throughput B/A 1.08–1.32 (estimate vs Backboard 0.99–1.14, geomean 1.053, same as
before); p90 TTFT 15–46 % below the article recipe and below Backboard's at 6/8 (behind at 8K×1 +3 %, 8K×4 +14 %).
Held-out seeds 1/2 at 8K×4 (previous config) gave B/A 1.320 / 1.335 vs 1.326 on seed 0; 8K×64 on a second GPU with B
first gave 1.165 vs 1.162. Remaining 8K TTFT is prefill compute (≈ 530 ms per 8K prompt) and collisions.

**D-068 (2026-11-14) 8K TTFT: tokenizer and prefill breakdown; config `best-e8-pfa2-sync-hybtok`.**
Server histograms at 8K×1 (D-067 config): TTFT 547 ms = engine prefill 518 ms + ~29 ms outside the engine; queue 0.
- Frontend (`kernels/frontend_ttft_probe.py`): arrival is stamped before tokenization; HF tokenization of a 7.4K-token
  prompt 13.5–14.1 ms, first-token detokenization 5.7 ms (client-side only); multimodal limits / text-only mode change
  nothing per request (no processor runs for text prompts). ~10 ms of the 29 stays unattributed (IPC, GIL).
- `VLLM_USE_FASTOKENS=1` (fastokens 0.3.6 in the experimental image): identical ids, 8K×1 median TTFT −16 to −28 ms,
  but output throughput −3.6 % and p99 TPOT +22 % at 8K×1, reproduced on two GPUs in both orders (cause not found;
  decode itself is not slower in isolation). Not adopted.
- Patch `hybrid_tokenizer` (`G4_HYBRID_TOKENIZER=1`): fastokens for prompt encoding only, HF `tokenizers` for every
  decode path; ids and streamed text identical to HF (`kernels/hybrid_tok_probe.py`). Same GPU vs the D-067 config:
  8K×1 p90 TTFT 561 → 545 ms, median 533 → 516, throughput +0.3 %, p99 TPOT unchanged. **Adopted.**
- Prefill profile of one 7,772-token step (profiler window now anchored on the first request, runner fix): 527 ms GPU
  = GEMMs 409 ms (cuBLAS 13.10, 399–428 TFLOPS), attention 68 ms (global 43, sliding 26; ~185–191 TFLOPS), GELU·mul
  22 ms (memory-bound), norms/RoPE/residual 20 ms, idle 4 ms; logits and drafter < 2 ms. Attention v3 (multi-stage
  cp.async) was slower; a shared K/V tile sliding variant (`v2:gl_m64_n32_w4x1_kvs`) is 1.12–1.16× in the bench (~3 ms
  per prompt) but unmeasurable in serving (config `-skvs`, not adopted). Remaining exact-math levers: a GEMM epilogue
  fusing GELU·mul (≤ 22 ms) or a > 250 TFLOPS global-attention kernel (~12 ms); neither attempted.

**D-069 (2026-11-14) TTFT vs TPOT at 8K×4: keep full 8K prefill steps; project cap $120.**
8K×4 p99 TPOT (ours 11.3 vs Backboard 10.3) is set by the same few requests in every run, ours and the article
recipe's: those that sit through other requests' whole-prompt prefill steps (~530 ms each, one decode step per
prefill). Median TPOT is 6.15 vs 8.5 ms for the article recipe. `--long-prefill-token-threshold` caps a chunk only
when more than one request is eligible (vLLM scheduler.py), so c=1 is untouched. Rule fixed before the result: adopt
the threshold with the lowest 8K×4 p99 TPOT if 8K×4 p90 TTFT stays under Backboard's 724 ms and 8K×64 throughput
drops < 0.5 %. Same GPU vs `-hybtok`: 4096: 8K×4 p99 TPOT −3 %, 8K×16 slightly worse, 8K×64 flat (noise level);
2048: 8K×4 p99 TPOT 11.5–11.7 → 10.4–10.8 ms, p99 E2EL → 10.7–11.0 s, but p90 TTFT 969–1126 ms (all four burst
prompts prefill in parallel at 2048 tokens per step) and median TTFT +45–55 ms. Neither meets the rule: **rejected**.
At 8K×4 these scheduler options trade one metric against the other; the frozen config takes the TTFT side (p90 TTFT
610–677 ms over 4 pairs on 2 GPUs). The owner left a further spend increase to the agent; the cap is $120 for the
confirmation pairs (spend ≈ $85 after D-069).

**D-070 (2026-11-14) Repository cleanup: raw dumps out of the tree, workspace name redacted.** The tracked tree held
456 MB of results, 394 MB of it per-request InferenceX dumps (`inferencex_result.json`, 241 files) that no analysis
reads (`analyze`, `report`, `claims` and the greedy gate use `results.jsonl`, `manifest.json`, `metadata.json`,
`correctness.json`). Those dumps, two profiler traces and the tokenizer-probe id/prompt dumps are removed from the
tree and ignored from now on; they remain in history (first parent of the PR #25 merge). Server and client logs stay (≈ 19 MB; the logs are
the evidence for active kernels and patches). The Modal workspace name in D-010/D-033 is replaced by `<workspace>`.
On the owner's instruction the history was then rewritten (D-071), which also removed the name from older commits. No secrets,
tokens, personal paths or e-mail addresses were found in tracked files; Modal resource ids (`vo-…`, `ap-…`) in run
metadata are ephemeral and kept.

**D-071 (2026-11-14) History rewrite on the owner's instruction.** The owner asked that agent-instruction files and
budget talk never appear in the published repository, including its history. Every commit was rewritten with
git-filter-repo: `CLAUDE.md`, `BUDGET.md`, the old handoff docs `STATUS.md` / `NEXT.md` and the launch ledger
`results/budget_ledger.jsonl` are removed from all commits (the ledger stays local and untracked), budget and spend
passages are removed from every version of the README files, and the Modal workspace name is redacted everywhere.
Merged topic branches were deleted. Commit SHAs before this point changed; a full bundle of the old history is kept
off-repo by the owner.

**D-072 (2026-11-15) Held-out concurrencies: the final config generalizes.** Asked whether the config only works at
the tuned batch sizes, we ran same-GPU pairs (article recipe A vs final config B, n = 2 pairs each) at concurrencies
never used for tuning: 8, 32 (the speculation-depth switch) and 48 (above it), at 1K and 8K input
(`configs/matrix/untuned-conc.yaml`). Throughput B/A: 1K × 8 1.18, 1K × 32 1.13, 1K × 48 1.07, 8K × 8 1.28,
8K × 32 1.16, 8K × 48 1.16; the two pairs at each point agree within 0.4%, and every value lies on the curve through
the tuned neighbours (1K: 1.26 @4, 1.19 @16, 1.07 @64; 8K: 1.33 @4, 1.26 @16, 1.16 @64). p99 TPOT is lower at every
pair; median TTFT is 0.49–0.84× the recipe's; p90 TTFT is lower in 11 of 12 pairs (one 1K × 32 pair 1.08×, a
first-burst outlier; its twin 0.75×). Above 32 running requests the config drafts 4 tokens like the recipe
(acceptance ratio ≈ 1.00 vs 1.25 below), so 1K × 48 gets the patches' gain but not deeper speculation; whether 6
drafts pay up to 48 at 1K is untested. The pair launcher's cost estimate now interpolates the article's throughput
between published concurrencies instead of assuming 50 tok/s for unpublished ones.
