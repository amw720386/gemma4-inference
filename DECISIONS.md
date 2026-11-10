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
single active GPU job, soft wall budget + hard timeout. Verified 2026-11-10: `https://arnold-17059--g4repro-bridge.modal.run`
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
