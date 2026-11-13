import os
from pathlib import Path

import pytest

from g4bench.test_gate import LOCAL_OVERRIDE_ENV, local_tests_allowed


def pytest_configure(config):
    """Owner policy (2026-11-13): unit tests run on Modal only (`g4bench modal selftest --execute`, at most every
    2.5 h). Inside a Modal container MODAL_TASK_ID is set; locally only an explicit owner override runs tests."""
    if not local_tests_allowed(dict(os.environ)):
        pytest.exit("unit tests run on Modal only: use `g4bench modal selftest --execute` (at most once every "
                    f"2.5 h); local runs need {LOCAL_OVERRIDE_ENV}=1 set by the owner", returncode=3)


TOK = Path(os.environ.get("G4_TOKENIZER", "/root/.cache/huggingface/hub/models--google--gemma-4-12B-it/snapshots/43c9d014b231474fa7212dbb48a762566c2b255d"))


@pytest.fixture(scope="session")
def tokenizer_dir() -> str:
    if (TOK / "tokenizer.json").exists():
        return str(TOK)
    try:
        from g4bench.config import load_pins
        from g4bench.runner import resolve_tokenizer

        return resolve_tokenizer(load_pins())
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"tokenizer unavailable (network?): {e}")
