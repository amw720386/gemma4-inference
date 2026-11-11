"""Local spend ledger for direct Modal launches (spend.py)."""
import json
import subprocess
import time

import pytest

from g4bench import spend as S

CAPS = S.Caps(10.0, 2.0)


def test_rates_match_modal_billing_rates():
    # `modal billing rates` 2026-11-10: RTX PRO 6000 $3.03/h, CPU $0.0473/core/h, memory $0.008/GiB/h
    assert S.GPU_JOB.rate() * 3600 == pytest.approx(3.03 + 8 * 0.0473 + 64 * 0.008)
    t = S.timeout_for_budget(S.GPU_JOB.rate(), 2.0)
    assert S.worst_case_usd(S.GPU_JOB.rate(), t) <= 2.0 < S.worst_case_usd(S.GPU_JOB.rate(), t + 2)
    assert S.timeout_for_budget(S.GPU_JOB.rate(), 0.5) == 0  # startup margin alone exceeds $0.50


def test_env_only_tightens(monkeypatch):
    monkeypatch.setenv("G4_PROJECT_CAP_USD", "50")
    monkeypatch.setenv("G4_PER_JOB_CAP_USD", "5")
    assert S.Caps.from_env() == S.Caps(10.0, 2.0)
    monkeypatch.setenv("G4_PROJECT_CAP_USD", "4")
    monkeypatch.setenv("G4_PER_JOB_CAP_USD", "1.5")
    assert S.Caps.from_env() == S.Caps(4.0, 1.5)


def test_reserve_settle_reconcile_cycle(tmp_path):
    L = tmp_path / "ledger.jsonl"
    S.append({"note": "history without launch id is ignored"}, L)
    assert S.check_launch(2.0, CAPS, S.read(L)) == pytest.approx(8.0)
    S.append({"launch_id": "a", "status": "reserved", "worst_usd": 2.0, "deadline_unix": time.time() - 100}, L)
    with pytest.raises(S.BudgetError, match="open reservation"):
        S.check_launch(0.1, CAPS, S.read(L))
    assert S.ledger_total(S.read(L)) == 2.0           # open = worst case
    S.append({"launch_id": "a", "app_id": "ap-1"}, L)
    S.append({"launch_id": "a", "status": "settled", "settled_usd": 0.5}, L)
    assert S.ledger_total(S.read(L)) == 0.5
    rows = [{"object_id": "ap-1", "description": "g4repro", "cost": "0.30"},
            {"object_id": "ap-1", "description": "g4repro", "cost": "0.40"}]
    S.reconcile(S.read(L), rows, L)
    assert S.ledger_total(S.read(L)) == pytest.approx(0.7)   # bill above our estimate wins
    assert S.reconcile(S.read(L), rows, L) == []             # idempotent


def test_crashed_launch_settles_from_billing_only_after_deadline(tmp_path):
    L = tmp_path / "ledger.jsonl"
    rows = [{"object_id": "ap-2", "description": "g4repro", "cost": "0.2"}]
    S.append({"launch_id": "b", "status": "reserved", "worst_usd": 2.0, "app_id": "ap-2",
              "deadline_unix": time.time() + 3600}, L)
    S.reconcile(S.read(L), rows, L)
    assert S.ledger_total(S.read(L)) == 2.0  # still possibly running: keep worst case
    S.append({"launch_id": "c", "status": "reserved", "worst_usd": 2.0, "app_id": "ap-3",
              "deadline_unix": time.time() - 3600}, L)
    S.reconcile(S.read(L), rows + [{"object_id": "ap-3", "description": "g4repro", "cost": "0.9"}], L)
    assert S.launches(S.read(L))["c"]["status"] == "settled-from-billing"
    assert S.ledger_total(S.read(L)) == pytest.approx(2.9)


def test_project_cap_uses_larger_of_ledger_and_bill_plus_storage(tmp_path):
    L = tmp_path / "ledger.jsonl"
    S.append({"launch_id": "x", "status": "settled", "worst_usd": 2, "settled_usd": 1.0}, L)
    assert S.check_launch(2.0, CAPS, S.read(L), billed_total=0.2) == pytest.approx(7.0)
    with pytest.raises(S.BudgetError, match="project cap"):
        S.check_launch(2.0, CAPS, S.read(L), billed_total=8.5)
    with pytest.raises(S.BudgetError, match="per-job"):
        S.check_launch(2.01, CAPS, [])
    now = time.time()
    entries = [{"kind": "storage", "gib": 30, "unix": now - 30 * 86400}]
    assert S.storage_accrual_usd(entries, now) == pytest.approx(30 * 0.09)
    entries.append({"kind": "storage", "gib": 0, "unix": now - 15 * 86400})
    assert S.storage_accrual_usd(entries, now) == pytest.approx(15 * 0.09)


def test_billing_rows_filtered_to_project_apps(monkeypatch):
    rows = [{"object_id": "ap-1", "description": "g4repro", "cost": "1"},
            {"object_id": "ap-2", "description": "someone-else", "cost": "50"}]
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, json.dumps(rows), ""))
    assert [r["object_id"] for r in S.modal_billing()] == ["ap-1"]
