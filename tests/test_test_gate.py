"""The unit-test policy: Modal only, at most once every 2.5 hours (src/g4bench/test_gate.py)."""
import pytest

from g4bench import test_gate as G


def test_interval_counts_from_the_last_selftest_launch():
    now = 1_000_000.0
    assert G.last_selftest_unix([]) is None
    G.check_selftest_interval([], now)  # never ran: allowed
    entries = [{"kind": "selftest", "reserved_unix": now - 3 * 3600},
               {"kind": "gpu:vllm", "reserved_unix": now - 60},          # other launches do not count
               {"kind": "selftest", "reserved_unix": now - 2 * 3600}]
    assert G.last_selftest_unix(entries) == now - 2 * 3600
    with pytest.raises(G.TestGateError, match="2.5 h"):
        G.check_selftest_interval(entries, now)
    G.check_selftest_interval(entries, now - 2 * 3600 + G.SELFTEST_MIN_INTERVAL_S)  # exactly 2.5 h later: allowed


def test_local_runs_need_modal_or_explicit_override():
    assert not G.local_tests_allowed({})
    assert G.local_tests_allowed({"MODAL_TASK_ID": "ta-123"})
    assert G.local_tests_allowed({G.LOCAL_OVERRIDE_ENV: "1"})
    assert not G.local_tests_allowed({G.LOCAL_OVERRIDE_ENV: "yes"})
