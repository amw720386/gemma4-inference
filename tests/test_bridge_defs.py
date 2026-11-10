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
    pins = (REPO / "configs/pins.yaml").read_text()
    assert bridge_app.VLLM_IMAGE.split("@")[1] in pins and bridge_app.SGLANG_IMAGE.split("@")[1] in pins
