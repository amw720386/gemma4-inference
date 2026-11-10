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
