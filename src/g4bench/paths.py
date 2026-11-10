"""Repository-relative paths."""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CONFIGS = REPO / "configs"
RESULTS = REPO / "results"
THIRD_PARTY_INFERENCEX = REPO / "third_party" / "inferencex"
TARGETS = CONFIGS / "targets" / "backboard-2026-11-09.json"
