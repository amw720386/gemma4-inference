"""The deployable bridge must only reference configs that exist (definition-level check; no network)."""
import sys

import pytest

from g4bench.config import resolve_config
from g4bench.paths import REPO
from g4bench.roofline import GPUS


def test_bridge_allowlists_resolve():
    pytest.importorskip("modal")
    sys.path.insert(0, str(REPO / "modal_bridge"))
    import app as bridge_app  # noqa: PLC0415

    for e in bridge_app.ALLOWED_ENGINES:
        resolve_config(e, "engines")
    for m in bridge_app.ALLOWED_MATRICES:
        resolve_config(m, "matrix")
    for g in bridge_app.ALLOWED_GPUS:
        assert g.rstrip("!") in GPUS, g
    assert "sha256:" in bridge_app.VLLM_IMAGE and "sha256:" in bridge_app.SGLANG_IMAGE
    # budget wiring: defaults are the approved caps, and the GPU function's default timeout fits the per-job cap
    from g4bench import budget as B
    from g4bench.bridge_api import GPU_JOB

    assert bridge_app.PROJECT_CAP_USD == 10.0 and bridge_app.PER_JOB_CAP_USD <= 2.0
    assert B.worst_case_usd(GPU_JOB.rate(), bridge_app.GPU_DEFAULT_TIMEOUT_S) <= bridge_app.PER_JOB_CAP_USD
    assert bridge_app._gpu_kw["retries"] == 0 and bridge_app._gpu_kw["cpu"][1] == GPU_JOB.cpu_limit
    assert bridge_app._gpu_kw["memory"][1] == int(GPU_JOB.mem_limit_gib * 1024)
    assert bridge_app.SCALEDOWN_WINDOW_S <= B.SCALEDOWN_MARGIN_S
    for name in ("ledger_op", "run_vllm", "run_sglang", "prefetch_weights", "selftest", "bridge"):
        assert hasattr(bridge_app, name)
    pins = (REPO / "configs/pins.yaml").read_text()
    assert bridge_app.VLLM_IMAGE.split("@")[1] in pins and bridge_app.SGLANG_IMAGE.split("@")[1] in pins
