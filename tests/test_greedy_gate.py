"""Token-level greedy-equivalence gate (correctness.greedy_run / compare_greedy) on CPU with the mock server."""
import asyncio
import copy
import json
import os
import threading
from pathlib import Path

import pytest

from g4bench import correctness as C


@pytest.fixture()
def mock_url():
    """The mock server's aiohttp app in a background thread (no subprocess, works on any OS)."""
    from aiohttp import web

    from g4bench.mockserver import make_app

    loop = asyncio.new_event_loop()
    runner = web.AppRunner(make_app("mock/model", 1.0, 3, 0.0, 0.0, 4))
    started = threading.Event()
    port: dict = {}

    def serve():
        asyncio.set_event_loop(loop)
        loop.run_until_complete(runner.setup())
        site = web.TCPSite(runner, "127.0.0.1", 0)
        loop.run_until_complete(site.start())
        port["p"] = site._server.sockets[0].getsockname()[1]
        started.set()
        loop.run_forever()

    t = threading.Thread(target=serve, daemon=True)
    t.start()
    assert started.wait(10)
    yield f"http://127.0.0.1:{port['p']}"
    loop.call_soon_threadsafe(loop.stop)
    t.join(5)


def test_prompt_set_is_fixed_and_deterministic():
    p1, p2 = C.greedy_prompts(), C.greedy_prompts()
    assert p1 == p2 and len(p1) == 8
    assert [p["kind"] for p in p1] == ["short"] * 4 + ["ctx3k"] * 2 + ["ctx8k"] * 2
    assert C.prompt_set_sha256(p1) == C.prompt_set_sha256(p2)
    assert C.prompt_set_sha256(p1) != C.prompt_set_sha256(p1, max_tokens=128)
    assert len(p1[6]["prompt"]) > 2.4 * len(p1[4]["prompt"])


def test_long_prompts_hit_target_token_lengths(tokenizer_dir):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_dir)
    for p in C.greedy_prompts():
        s = tok.apply_chat_template([{"role": "user", "content": p["prompt"]}], add_generation_prompt=True,
                                    tokenize=False)
        n = len(tok.encode(s, add_special_tokens=False))
        lo, hi = {"short": (5, 100), "ctx3k": (2800, 3300), "ctx8k": (7600, 8200)}[p["kind"]]
        assert lo <= n <= hi, (p["id"], n)  # + 256 new tokens must fit max_model_len 10240


def test_greedy_run_captures_token_ids_and_logprobs(mock_url):
    g = C.greedy_run(mock_url, "mock/model")
    assert g["prompt_set_sha256"] == C.prompt_set_sha256() and len(g["items"]) == 8
    for it in g["items"]:
        assert "error" not in it, it
        assert it["mode"] == "token_ids+logprobs" and len(it["token_ids"]) == C.GREEDY_MAX_TOKENS
        assert len(it["top_logprobs"]) == C.GREEDY_MAX_TOKENS and len(it["top_logprobs"][0]) == 2
    cmp = C.compare_greedy(g, C.greedy_run(mock_url, "mock/model"))
    assert cmp["pass"] and cmp["n_identical"] == 8 and cmp["identical_prefix_token_fraction"] == 1.0


def _fake(ids_by_prompt, gap=2.4):
    items = []
    for pid, ids in ids_by_prompt.items():
        top = [[[t, -0.1], [t + 1, -0.1 - gap]] for t in ids]
        items.append({"id": pid, "kind": "short", "token_ids": ids, "top_logprobs": top, "text": " ".join(map(str, ids))})
    return {"prompt_set_sha256": "x", "items": items}


def test_compare_reports_first_divergence_and_near_ties():
    ref = _fake({"a": list(range(100)), "b": list(range(100)), "c": list(range(100)), "d": list(range(50))})
    cand = copy.deepcopy(ref)
    # a: flip to the rank-2 token at 40, a near tie on both sides
    cand["items"][0]["token_ids"][40] = 41
    ref["items"][0]["top_logprobs"][40] = [[40, -0.69], [41, -0.70]]
    cand["items"][0]["top_logprobs"][40] = [[41, -0.69], [40, -0.71]]
    # b: flip at 10 to a token outside both top-2 lists -> not a near tie
    cand["items"][1]["token_ids"][10] = 999
    cand["items"][1]["top_logprobs"][10] = [[999, -0.1], [1000, -3.0]]
    # d: candidate stops early (EOS) at 30
    cand["items"][3]["token_ids"] = cand["items"][3]["token_ids"][:30]
    r = C.compare_greedy(ref, cand)
    assert not r["pass"]
    by = {it["id"]: it for it in r["items"]}
    assert by["a"]["first_divergence"] == 40 and by["a"]["near_tie"] is True and by["a"]["prefix_fraction"] == 0.4
    assert by["a"]["positional_match_fraction"] == 0.99
    assert by["b"]["first_divergence"] == 10 and by["b"]["near_tie"] is False
    assert by["c"]["identical"] and by["c"]["status"] == "identical"
    assert by["d"]["status"] == "length" and by["d"]["first_divergence"] == 30
    assert (r["n_identical"], r["n_divergent"], r["n_near_tie"], r["n_not_near_tie"], r["n_length_mismatch"]) \
        == (1, 2, 1, 1, 1)
    assert r["identical_prefix_token_fraction"] == round((40 + 10 + 100 + 30) / 350, 4)


def test_compare_refuses_different_prompt_set_and_flags_errors():
    ref = _fake({"a": [1, 2, 3]})
    other = {**copy.deepcopy(ref), "prompt_set_sha256": "y"}
    assert C.compare_greedy(ref, other)["pass"] is False and "prompt set" in C.compare_greedy(ref, other)["error"]
    bad = copy.deepcopy(ref)
    bad["items"][0] = {"id": "a", "error": "HTTP 400"}
    r = C.compare_greedy(ref, bad)
    assert not r["pass"] and r["n_error"] == 1
    no_lp = copy.deepcopy(ref)
    no_lp["items"][0]["token_ids"] = [1, 9, 3]
    no_lp["items"][0]["top_logprobs"] = None
    ref["items"][0]["top_logprobs"] = None
    assert C.compare_greedy(ref, no_lp)["n_tie_unknown"] == 1


def test_parse_choice_from_logprob_tokens_without_token_ids():
    ch = {"message": {"content": "hi"}, "finish_reason": "stop",
          "logprobs": {"content": [{"token": "token_id:5", "logprob": -0.2,
                                    "top_logprobs": [{"token": "token_id:5", "logprob": -0.2},
                                                     {"token": "token_id:7", "logprob": -1.9}]}]}}
    out = C.parse_choice(ch)
    assert out["token_ids"] == [5] and out["top_logprobs"] == [[[5, -0.2], [7, -1.9]]]


def test_promote_then_compare_run(tmp_path, mock_url):
    g = C.greedy_run(mock_url, "mock/model")
    for name, eng in (("ref", "vllm-0.33.0-nospec"), ("cand", "vllm-0.33.0-best-e8-all")):
        d = tmp_path / name
        d.mkdir()
        (d / "correctness.json").write_text(json.dumps({"probes": {"pass": True}, "greedy": g}))
        (d / "manifest.json").write_text(json.dumps({"engine_config": {"name": eng}}))
        (d / "metadata.json").write_text(json.dumps({"git": {"sha": "abc"}, "hardware_class": "cpu"}))
    ref = C.promote(tmp_path / "ref", tmp_path / "refs" / "nospec.json")
    assert json.loads(ref.read_text())["reference"]["engine_config"] == "vllm-0.33.0-nospec"
    res = C.compare_run(tmp_path / "cand", ref)
    assert res["pass"] and (tmp_path / "cand" / "greedy_compare.json").exists()


def test_runner_correctness_writes_greedy_and_compare(tmp_path, mock_url):
    from g4bench.runner import _correctness

    refdir = tmp_path / "r"
    refdir.mkdir()
    _correctness(refdir, mock_url, "mock/model", reference=str(tmp_path / "missing.json"))
    corr = json.loads((refdir / "correctness.json").read_text())
    assert corr["probes"]["pass"] and len(corr["greedy"]["items"]) == 8
    assert corr["greedy_compare"]["pass"] is None  # no reference yet: not measured, never a pass
    reffile = tmp_path / "ref.json"
    reffile.write_text(json.dumps(corr["greedy"]))
    cand = tmp_path / "c"
    cand.mkdir()
    _correctness(cand, mock_url, "mock/model", reference=str(reffile))
    assert json.loads((cand / "correctness.json").read_text())["greedy_compare"]["pass"] is True


def test_job_spec_code_sha_reaches_metadata(monkeypatch, tmp_path):
    """Patched-image runs had git.sha null: modal_app froze G4_CODE_SHA=unknown at import. The spec now carries it."""
    from g4bench import metadata

    monkeypatch.setattr(metadata, "REPO", tmp_path)  # no .git here, like /repo inside Modal
    monkeypatch.setenv("G4_CODE_SHA", "unknown")
    g = metadata.git_info()
    assert g["sha"] is None and "G4_CODE_SHA='unknown'" in g["error"]
    monkeypatch.setenv("G4_CODE_SHA", "0123abcd-dirty")
    assert metadata.git_info() == {"sha": "0123abcd", "dirty": True, "source": "G4_CODE_SHA"}


def test_run_job_exports_code_sha_and_greedy_options(monkeypatch, tmp_path):
    import time

    from g4bench import jobs, runner

    seen = {}

    def fake_run(opts):
        seen.update(opts=opts, sha=os.environ.get("G4_CODE_SHA"))
        d = Path(tmp_path) / "run1"
        d.mkdir()
        return d

    monkeypatch.setattr(runner, "run", fake_run)
    monkeypatch.setenv("G4_CODE_SHA", "unknown")
    jobs.run_job({"engine": "mock-cpu", "matrix": "mock-ci", "max_wall_s": 100, "deadline_unix": time.time() + 1000,
                  "code_sha": "feedbeef", "correctness_only": True, "greedy": True,
                  "greedy_reference": "data/greedy_reference/x.json"}, tmp_path)
    assert seen["sha"] == "feedbeef"
    o = seen["opts"]
    assert o.correctness_only and o.greedy and o.greedy_reference == "data/greedy_reference/x.json"


def test_modal_main_sets_code_sha_before_gpu_kind(monkeypatch):
    """gpu_kind() imports modal_app (which snapshots G4_CODE_SHA); the sha must already be set by then."""
    import argparse

    from g4bench import modal_run

    order = []
    monkeypatch.setattr(modal_run, "_code_sha", lambda: "c0ffee")
    monkeypatch.setattr(modal_run, "allowed_engines", lambda: {"vllm-0.33.0-best-e8-all": "vllm"})
    monkeypatch.setattr(modal_run, "gpu_kind", lambda *a: order.append(os.environ.get("G4_CODE_SHA")) or "gpu:vllm")
    monkeypatch.setattr(modal_run, "plan", lambda kind, usd: {"kind": kind})
    monkeypatch.setattr(modal_run, "preflight", lambda p: {})
    monkeypatch.setattr(modal_run, "load_dotenv", lambda: None)
    printed = []
    monkeypatch.setattr("builtins.print", lambda *a, **k: printed.append(a[0]))
    args = argparse.Namespace(modal_cmd="job", engine="vllm-0.33.0-best-e8-all", matrix="backboard-repro", notes="",
                              seeds="0", only="1024x1", profile_key=None, max_usd=1.0, execute=False,
                              no_greedy=False, correctness_only=True, greedy_reference=None)
    assert modal_run.main(args) == 0
    assert order == ["c0ffee"]
    spec = json.loads(printed[0])["spec"]
    assert spec["code_sha"] == "c0ffee" and spec["correctness_only"] and spec["greedy"]
