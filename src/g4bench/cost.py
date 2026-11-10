"""Pre-launch time and dollar estimates. Always run before any paid GPU job.

Point wall time = server start (if fresh) + warmup + measured requests, where the
request phase is estimated from a throughput guess (by default the article's vLLM
number for that point; conservative factor applied). Estimates are deliberately
pessimistic; BUDGET.md records estimate vs actual for every launch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .compare import load_targets
from .matrix import Point
from .roofline import GPUS

DEFAULT_SERVER_START_S = 360.0   # weights load (24 GB from Volume) + torch.compile + CUDA graphs + draft
CONTAINER_OVERHEAD_S = 240.0     # image pull / cold start / metadata capture, once per job
SLOWDOWN_FACTOR = 1.5            # assume we are 1.5x slower than the article's vLLM until measured


@dataclass(frozen=True)
class Estimate:
    points: int
    gpu_seconds: float
    usd: float
    detail: list[dict[str, Any]]

    def summary(self) -> str:
        return (f"{self.points} points, ~{self.gpu_seconds/3600:.2f} GPU-h, ~${self.usd:.2f} "
                f"(pessimistic; slowdown x{SLOWDOWN_FACTOR}, start {DEFAULT_SERVER_START_S:.0f}s/server)")


def point_seconds(p: Point, tok_s: float) -> float:
    toks = (p.num_prompts + p.num_warmups) * p.osl
    return toks / max(tok_s, 1e-6)


def estimate(points: list[Point], gpu: str = "RTX-PRO-6000", *, fresh_server_per_point: bool = True,
             tok_s_ref: str = "vllm", server_start_s: float = DEFAULT_SERVER_START_S,
             engines: int = 1) -> Estimate:
    g = GPUS[gpu]
    targets = load_targets()
    detail = []
    total = CONTAINER_OVERHEAD_S
    if not fresh_server_per_point:
        total += server_start_s
    for p in points:
        # Real-prompt arms (key "...@dataset", isl 0 = natural length) borrow the nearest article workload's
        # throughput as the guess; the 1.5x slowdown factor still applies.
        near = 8192 if p.isl > 4096 else 1024
        ref = targets.get(f"{near}x{p.conc}", {}).get("output_throughput", {}).get(tok_s_ref)
        tok_s = (ref / SLOWDOWN_FACTOR) if ref else 50.0
        s = point_seconds(p, tok_s) + (server_start_s if fresh_server_per_point else 0.0)
        detail.append({"point_id": p.point_id, "est_tok_s": tok_s, "est_s": s})
        total += s
    total *= engines
    # Same pricing as the bridge ledger: GPU + CPU/memory at the GPU job's container *limits*.
    from .bridge_api import GPU_JOB
    from .budget import resource_rate

    usd = total * resource_rate(gpu, GPU_JOB.cpu_limit, GPU_JOB.mem_limit_gib) if g.usd_per_hr_modal else 0.0
    return Estimate(len(points) * engines, total, usd, detail)
