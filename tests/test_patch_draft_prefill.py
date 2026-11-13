"""Image patch mtp_draft_prefill_last (src/g4bench/patches): anchors, idempotency, guard, index invariants. CPU only."""
import ast
import enum
import os
import pathlib
import subprocess
import sys
import types

import pytest

from g4bench.paths import REPO
from g4bench.patches import apply as A
from g4bench.patches import mtp_draft_prefill_last as P

PATCH_DIR = REPO / "src" / "g4bench" / "patches"

# Runnable reduction of vLLM v0.33.0 vllm/v1/worker/gpu/spec_decode/target_dependent_ar/speculator.py: both patch
# anchors (propose lines 276-277, ``def _multi_step_decode(`` at line 492) are verbatim; the rest is stubbed. Set
# G4_VLLM_TDAR=<path to the pinned upstream file> to also run the anchor tests against the real 994-line file.
EXCERPT = '''\
class TargetDependentARSpeculator:
    @property
    def advance_draft_positions(self) -> bool:
        return True

    def propose(self, input_batch, num_rejected, prefill_batch_desc, max_query_len,
                num_reqs=2, dummy_run=False, is_profile=False, num_speculative_tokens=None):
        if num_speculative_tokens is None:
            num_speculative_tokens = self.num_speculative_steps
        self.calls = []
        self.on_prefill_begin(num_reqs)
        if prefill_batch_desc.cg_mode == CUDAGraphMode.FULL:
            # Replay the full graph for draft prefill.
            self.calls.append("fullgraph")
        else:
            self.calls.append(("upstream_prefill", num_speculative_tokens == 0))
        self.on_prefill_end(num_reqs)
        return self.calls

    def on_prefill_begin(self, num_reqs): ...

    def on_prefill_end(self, num_reqs): ...

    def _multi_step_decode(
        self,
        num_reqs,
    ):
        pass


class Gemma4Speculator(TargetDependentARSpeculator):
    @property
    def advance_draft_positions(self) -> bool:
        return False


class EagleSpeculator(TargetDependentARSpeculator):
    pass
'''


class CUDAGraphMode(enum.Enum):
    NONE = 0
    PIECEWISE = 1
    FULL = 2


def _upstream() -> str:
    p = os.environ.get("G4_VLLM_TDAR")
    return pathlib.Path(p).read_text(encoding="utf-8") if p else EXCERPT


def _classes(src: str) -> dict:
    ns: dict = {"CUDAGraphMode": CUDAGraphMode}
    exec(compile(src, "tdar", "exec"), ns)
    return ns


def _spec(cls, **over):
    s = cls.__new__(cls)
    attrs = dict(prefill_cudagraph_manager=object(), pcp_manager=None, supports_mm_inputs=False, dp_size=1,
                 dcp_size=1, num_speculative_steps=6)
    attrs.update(over)
    for k, v in attrs.items():
        setattr(s, k, v)
    return s


def _desc(mode: CUDAGraphMode):
    return types.SimpleNamespace(cg_mode=mode)


def test_applies_exactly_once_and_is_idempotent():
    src = _upstream()
    out, changed = P.patch_text(src)
    assert changed and out != src
    assert out.count(P.PROPOSE_NEW) == 1 and out.count(P.METHODS + P.METHODS_ANCHOR) == 1
    assert out.count(P.MARKER) == 2
    ast.parse(out)
    compile(out, "patched", "exec")
    again, changed2 = P.patch_text(out)
    assert again == out and not changed2


def test_refuses_missing_or_duplicated_anchor():
    with pytest.raises(P.PatchError, match="0 times"):
        P.patch_text(EXCERPT.replace("        self.on_prefill_begin(num_reqs)\n", "        pass\n"))
    with pytest.raises(P.PatchError, match="0 times"):
        P.patch_text(EXCERPT.replace("def _multi_step_decode(", "def _multi_step_decode_v2("))
    with pytest.raises(P.PatchError, match="2 times"):
        P.patch_text(EXCERPT + "\n\nclass X:\n" + P.METHODS_ANCHOR + "        pass\n")
    out, _ = P.patch_text(EXCERPT)
    with pytest.raises(P.PatchError, match="marker"):  # marker present but a patched block was edited away
        P.patch_text(out.replace("int(max_query_len) > self.num_speculative_steps + 1", "True"))


def test_env_gate_off_means_upstream(monkeypatch):
    """D-061: without G4_DRAFT_PREFILL_LAST=1 the guard is False and propose takes the upstream branches."""
    ns = _classes(P.patch_text(EXCERPT)[0])
    g4 = ns["Gemma4Speculator"]
    pw = _desc(CUDAGraphMode.PIECEWISE)
    g4._g4_prefill_last_tokens = lambda self, *a: (_ for _ in ()).throw(AssertionError("patched path taken"))
    for val in (None, "0", "", "true", "yes"):
        if val is None:
            monkeypatch.delenv(P.ENV, raising=False)
        else:
            monkeypatch.setenv(P.ENV, val)
        s = _spec(g4)
        assert not s._g4_use_last_token_prefill(pw, 3072, False, False), val
        assert _spec(g4).propose("ib", "rej", pw, 3072) == [("upstream_prefill", False)], val
    monkeypatch.setenv(P.ENV, "1")
    s = _spec(g4)
    assert s._g4_use_last_token_prefill(pw, 3072, False, False)
    monkeypatch.setenv(P.ENV, "0")  # read once per speculator instance (constructed at server start)
    assert s._g4_use_last_token_prefill(pw, 3072, False, False)
    assert P.ENV in P.METHODS


def test_every_config_listing_the_patch_turns_it_on():
    from g4bench.config import EngineConfig
    from g4bench.paths import CONFIGS

    listed = [EngineConfig.load(str(f)) for f in sorted((CONFIGS / "engines").glob("*.yaml"))]
    listed = [e for e in listed if P.NAME in (e.raw.get("image_patches") or [])]
    assert len(listed) >= 8
    for e in listed:
        assert e.env.get(P.ENV) == "1", e.name


def test_guard_selects_only_gemma4_prefill_batches(monkeypatch):
    monkeypatch.setenv(P.ENV, "1")
    ns = _classes(P.patch_text(EXCERPT)[0])
    g4, eagle = ns["Gemma4Speculator"], ns["EagleSpeculator"]
    pw, full = _desc(CUDAGraphMode.PIECEWISE), _desc(CUDAGraphMode.FULL)
    use = lambda s, desc=pw, mql=3072, **kw: s._g4_use_last_token_prefill(desc, mql, kw.get("dummy", False),  # noqa: E731
                                                                         kw.get("profile", False))
    assert use(_spec(g4)) and use(_spec(g4), _desc(CUDAGraphMode.NONE))
    assert use(_spec(g4), mql=8) and not use(_spec(g4), mql=7)  # 7 = verify window of 6 spec tokens
    assert not use(_spec(eagle))  # other speculators untouched
    assert not use(_spec(g4), full)  # FULL graphs (uniform decode) keep upstream replay
    assert not use(_spec(g4), dummy=True) and not use(_spec(g4), profile=True)
    for over in (dict(pcp_manager=object()), dict(supports_mm_inputs=True), dict(dp_size=2), dict(dcp_size=2),
                 dict(prefill_cudagraph_manager=None)):
        assert not use(_spec(g4, **over)), over


def test_propose_routes_to_last_token_path_and_keeps_upstream_branches(monkeypatch):
    monkeypatch.setenv(P.ENV, "1")
    ns = _classes(P.patch_text(EXCERPT)[0])
    g4, eagle = ns["Gemma4Speculator"], ns["EagleSpeculator"]
    seen = []
    g4._g4_prefill_last_tokens = lambda self, n, ib, rej, skip: seen.append((n, ib, rej, skip))
    pw, full = _desc(CUDAGraphMode.PIECEWISE), _desc(CUDAGraphMode.FULL)
    assert _spec(g4).propose("ib", "rej", pw, 3072) == []
    assert _spec(g4).propose("ib", "rej", pw, 3072, num_speculative_tokens=0) == []
    assert seen == [(2, "ib", "rej", False), (2, "ib", "rej", True)]
    assert _spec(g4).propose("ib", "rej", full, 7) == ["fullgraph"]
    assert _spec(g4).propose("ib", "rej", pw, 7) == [("upstream_prefill", False)]
    assert _spec(eagle).propose("ib", "rej", pw, 3072) == [("upstream_prefill", False)]


def _prepare_prefill(query_start_loc, seq_lens, num_rejected):
    """Pure-Python model of upstream _prepare_prefill_inputs_kernel (tdar lines 705-747): per request, the kept
    row and its position, where target positions run up to seq_len - 1 across the request's query."""
    positions, last = {}, []
    for r in range(len(seq_lens)):
        qs, qe = query_start_loc[r], query_start_loc[r + 1]
        for t in range(qs, qe):
            positions[t] = seq_lens[r] - (qe - qs) + (t - qs)
        last.append(qs + (qe - qs - num_rejected[r]) - 1)
    return positions, last


def _compact(query_start_loc, seq_lens, num_rejected, max_num_reqs):
    """Pure-Python model of _g4_prefill_last_tokens' index updates."""
    positions, last = _prepare_prefill(query_start_loc, seq_lens, num_rejected)
    n = len(seq_lens)
    return dict(
        positions=[positions[i] for i in last],
        query_start_loc=[min(i, n) for i in range(max_num_reqs + 1)],
        seq_lens=[seq_lens[r] - num_rejected[r] for r in range(n)] + [0] * (max_num_reqs - n),
        last_token_indices=list(range(n)),
    )


@pytest.mark.parametrize("qlens,seq_lens,rejected", [
    ([3072], [3072], [0]),                       # first chunk of a long prompt (non-final chunk)
    ([3072, 7, 7], [6144, 900, 41], [0, 2, 6]),   # final chunk mixed with verify rows (some rejections)
    ([1, 513, 7], [1, 513, 70], [0, 0, 0]),
])
def test_compacted_rows_match_kept_rows_and_causal_lengths(qlens, seq_lens, rejected):
    qsl = [0]
    for q in qlens:
        qsl.append(qsl[-1] + q)
    positions, last = _prepare_prefill(qsl, seq_lens, rejected)
    c = _compact(qsl, seq_lens, rejected, max_num_reqs=8)
    assert c["positions"] == [positions[i] for i in last]  # same rows _prefill would have kept
    # One query per request at the kept position attends to keys [0, pos] - the same causal row as the full pass,
    # and the same seq_len the upstream Gemma 4 decode steps use (target_seq_len - num_rejected).
    assert c["seq_lens"][: len(qlens)] == [p + 1 for p in c["positions"]]
    assert c["query_start_loc"][: len(qlens) + 1] == list(range(len(qlens) + 1))
    assert c["query_start_loc"][len(qlens):] == [len(qlens)] * (9 - len(qlens)) and c["seq_lens"][len(qlens):] == [
        0] * (8 - len(qlens))
    assert c["last_token_indices"] == list(range(len(qlens)))
    draft_rows_before, draft_rows_after = qsl[-1], len(qlens)
    assert draft_rows_after <= draft_rows_before


def test_apply_script_patches_a_package_tree(tmp_path):
    pkg = tmp_path / "vllm"
    f = pkg / P.RELPATH
    f.parent.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    f.write_text(EXCERPT, encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": str(tmp_path)}
    cmd = [sys.executable, str(PATCH_DIR / "apply.py"), P.NAME]
    r = subprocess.run(cmd, cwd=tmp_path, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "patched" in r.stdout and P.MARKER in f.read_text(encoding="utf-8")
    r2 = subprocess.run(cmd, cwd=tmp_path, env=env, capture_output=True, text=True)
    assert r2.returncode == 0 and "already applied" in r2.stdout
    f.write_text("nothing to anchor\n", encoding="utf-8")
    r3 = subprocess.run(cmd, cwd=tmp_path, env=env, capture_output=True, text=True)
    assert r3.returncode != 0 and "FAILED" in r3.stderr
    assert P.NAME in A.KNOWN
