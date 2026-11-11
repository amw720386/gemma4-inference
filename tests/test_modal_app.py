"""Definition-level checks of the Modal app (no network, no spend)."""
import pytest

from g4bench import spend as S
from g4bench.paths import REPO


def test_modal_app_wiring_matches_spend_model():
    pytest.importorskip("modal")
    from g4bench import modal_app as M

    assert "sha256:" in M.VLLM_IMAGE and "sha256:" in M.SGLANG_IMAGE
    pins = (REPO / "configs/pins.yaml").read_text()
    assert M.VLLM_IMAGE.split("@")[1] in pins and M.SGLANG_IMAGE.split("@")[1] in pins
    assert S.worst_case_usd(S.GPU_JOB.rate(), M.GPU_DEFAULT_TIMEOUT_S) <= 2.0
    assert M._gpu_kw["retries"] == 0 and M._gpu_kw["max_containers"] == 1
    assert M._gpu_kw["cpu"][1] == S.GPU_JOB.cpu_limit
    assert M._gpu_kw["memory"][1] == int(S.GPU_JOB.mem_limit_gib * 1024)
    assert M._gpu_kw["gpu"] == S.GPU_JOB.gpu and M.SCALEDOWN_WINDOW_S <= S.SCALEDOWN_MARGIN_S
    for a in (M.cpu_app, M.vllm_app, M.sglang_app):  # billing reconciliation filters on this prefix
        assert a.name.startswith(S.APP_PREFIX)
    # each launch builds only its own app's image: engine images never ride along with CPU work
    assert M.run_vllm.app is M.vllm_app and M.run_sglang.app is M.sglang_app
    assert M.selftest.app is M.cpu_app and M.prefetch_weights.app is M.cpu_app
    assert ".env" in M.IGNORE and "*.secret" in M.IGNORE  # credentials never enter an image
    for name in ("run_vllm", "run_sglang", "prefetch_weights", "selftest"):
        assert hasattr(M, name)


def test_plan_only_derives_timeout_from_reservation():
    from g4bench.modal_run import allowed_engines, plan

    p = plan("gpu:vllm", 2.0)
    assert p["worst_usd"] <= 2.0 and p["timeout_s"] > 600
    assert plan("prefetch", 0.2)["worst_usd"] <= 0.2
    assert plan("selftest", 0.1)["timeout_s"] == S.SELFTEST_TIMEOUT_S and plan("selftest", 0.1)["worst_usd"] <= 0.1
    with pytest.raises(S.BudgetError):
        plan("gpu:vllm", 0.5)
    eng = allowed_engines()
    assert eng["vllm-0.33.0-mtp4"] == "vllm" and eng["sglang-0.5.23-nextn"] == "sglang" and "mock-cpu" not in eng
