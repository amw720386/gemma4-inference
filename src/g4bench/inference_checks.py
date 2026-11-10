"""Internal-consistency checks on the article's published numbers (no GPU needed).

Check A - output-length policy (infers the unstated ``--random-range-ratio``)
---------------------------------------------------------------------------
With ``--ignore-eos`` every request k returns L_k tokens and
E2EL_k = TTFT_k + (L_k - 1) * TPOT_k  >=  (L_k - 1) * TPOT_k  (InferenceX TPOT definition).
If all L_k = L (range ratio 1.0) then E2EL_k >= (L-1) * TPOT_k element-wise, so every
order statistic, hence every linearly-interpolated percentile, satisfies
    pXX(E2EL) >= (L-1) * pXX(TPOT).
A published (p99 E2EL, p99 TPOT) pair violating this is impossible under fixed
1024-token outputs, i.e. evidence that output lengths varied (range ratio < 1, as in
InferenceX's RANDOM_RANGE_RATIO=0.8). TPOT is published rounded to 0.1 ms, so we use
TPOT - 0.05 ms (most favourable to the fixed-length hypothesis).

Check B - throughput vs speed-per-user at concurrency 1
-------------------------------------------------------
At conc=1 requests are sequential, so throughput ~= sum(L) / sum(E2EL) must not
exceed 1000/mean(TPOT) (speed-per-user) by more than rounding.
"""

from __future__ import annotations

from typing import Any

from .compare import load_targets


def check_fixed_length(osl: int = 1024, targets: dict | None = None) -> list[dict[str, Any]]:
    t = targets or load_targets()
    out = []
    for key, m in t.items():
        for eng in ("backboard", "vllm", "sglang"):
            p99_tpot = m["p99_tpot"][eng]
            p99_e2el = m["p99_e2el"][eng]
            bound = (osl - 1) * (p99_tpot - 0.05)
            out.append({"key": key, "engine": eng, "p99_tpot": p99_tpot, "p99_e2el": p99_e2el,
                        "fixed_len_lower_bound_ms": round(bound, 1),
                        "violates_fixed_length": p99_e2el < bound})
    return out


def check_c1_throughput(targets: dict | None = None, rel_tol: float = 0.01) -> list[dict[str, Any]]:
    t = targets or load_targets()
    out = []
    for key, m in t.items():
        if not key.endswith("x1"):
            continue
        for eng in ("backboard", "vllm", "sglang"):
            thr = m["output_throughput"][eng]
            spu = m["speed_per_user"][eng]
            out.append({"key": key, "engine": eng, "throughput": thr, "speed_per_user": spu,
                        "ratio": thr / spu, "plausible": thr <= spu * (1 + rel_tol)})
    return out


def summary() -> dict[str, Any]:
    a = check_fixed_length()
    b = check_c1_throughput()
    return {"fixed_length_violations": sum(r["violates_fixed_length"] for r in a), "fixed_length_checked": len(a),
            "fixed_length_rows": a, "c1_rows": b,
            "c1_implausible": sum(not r["plausible"] for r in b)}


if __name__ == "__main__":
    import json

    print(json.dumps(summary(), indent=1))
