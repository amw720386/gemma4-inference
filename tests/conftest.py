import os
from pathlib import Path

import pytest

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
