"""Attribute a measured-vs-article decode gap to speculative acceptance vs verification-step speed.

Per sequence, speculative decoding emits on average AL tokens per verification step, so
    TPOT ≈ step_time / AL      ⇒      step_time = TPOT × AL.
From one measured point (TPOT_ours, AL_ours) and the article's TPOT for the same workload:
* ``step_ms``              = TPOT_ours × AL_ours (our cost of one target-verify + draft cycle);
* ``bw_efficiency``        = roofline step time at that batch/context ÷ measured step time (≤ 1; low ⇒ kernels/overhead);
* ``al_needed_for_article`` = TPOT_ours × AL_ours ÷ TPOT_article: the AL that, at OUR step time, would reproduce the
  article's TPOT. If it exceeds draft_tokens + 1 the article's number cannot be reached by acceptance alone.
* ``step_speedup_needed`` = TPOT_ours ÷ TPOT_article at OUR AL: how much faster steps must get instead.

Assumptions (stated in every output): mean TPOT, steady-state decode, AL counts the bonus token (vLLM's
definition), batch = concurrency, context = ISL + OSL/2. It is a first-order decomposition for diagnosis, not a model.
"""

from __future__ import annotations

from typing import Any

from . import roofline

ASSUMPTIONS = ("first-order: TPOT = step/AL; mean TPOT; AL incl. bonus token; batch = conc; ctx = ISL + OSL/2; "
               "roofline at 85% of peak HBM/GDDR bandwidth")


def attribute(*, tpot_ms: float | None, al: float | None, conc: int, isl: int, osl: int,
              article_tpot_ms: float | None, draft_tokens: int, gpu: str = "RTX-PRO-6000") -> dict[str, Any]:
    out: dict[str, Any] = {"assumptions": ASSUMPTIONS, "state": "ok"}
    if not tpot_ms or not al:
        out["state"] = "not measured (needs mean TPOT and acceptance length)"
        return out
    g = roofline.GPUS[gpu]
    ctx = isl + osl // 2
    step_ms = tpot_ms * al
    roof_tok_s = roofline.max_tokens_per_s(g, conc, ctx, acceptance_len=1.0, draft_tokens=draft_tokens)
    roof_step_ms = 1000.0 * conc / roof_tok_s  # one verify+draft cycle for the whole batch
    out.update(step_ms=step_ms, roofline_step_ms=roof_step_ms, bw_efficiency=min(roof_step_ms / step_ms, 9.99),
               max_al=draft_tokens + 1)
    if article_tpot_ms:
        need = step_ms / article_tpot_ms
        out.update(article_tpot_ms=article_tpot_ms, al_needed_for_article=need,
                   step_speedup_needed=tpot_ms / article_tpot_ms,
                   reachable_by_acceptance_alone=need <= draft_tokens + 1)
    return out


def draft_tokens_for(engine_config: dict[str, Any]) -> int | None:
    """num_speculative_tokens (vLLM) / speculative_num_steps (SGLang chain, topk 1) from an engine config."""
    import json

    args = [t for g in engine_config.get("args", []) for t in g]
    for i, t in enumerate(args):
        if t == "--speculative-config" and i + 1 < len(args):
            try:
                return int(json.loads(args[i + 1]).get("num_speculative_tokens"))
            except (ValueError, TypeError, json.JSONDecodeError):
                return None
        if t == "--speculative-num-steps" and i + 1 < len(args):
            return int(args[i + 1])
    return None
