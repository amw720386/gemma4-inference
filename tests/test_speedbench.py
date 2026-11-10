import hashlib
import json

import pytest

from g4bench import datasets, speedbench as sb

PH = sb.TURNS_PLACEHOLDER


def _rows():
    return [
        {"question_id": "q1", "category": "coding", "source": "https://huggingface.co/datasets/bigcode/humanevalpack/tree/x",
         "src_id": "a", "turns": ["def f(): pass"]},
        {"question_id": "q2", "category": "coding",
         "source": "https://huggingface.co/datasets/livecodebench/code_generation_lite/tree/0fe8", "src_id": "lc1",
         "turns": [f"{PH}\n\nSolve: {{question}}\nStart: {{starter_code}}"]},
        {"question_id": "q3", "category": "math", "source": "x", "src_id": "m", "turns": ["ignored"]},
        {"question_id": "q4", "category": "coding",
         "source": "https://huggingface.co/datasets/deepmind/code_contests/tree/8024", "src_id": "cc1",
         "turns": [f"{PH}\n\nProblem: {{question}}"]},
    ]


def test_resolve_coding_mirrors_prepare_py():
    out = sb.resolve_coding(_rows(), {"cc1": "add two ints"}.__getitem__, {"lc1": ("sum it", "def s():")}.__getitem__)
    assert [r["question_id"] for r in out] == ["q1", "q2", "q4"]  # coding only, dataset order kept
    assert out[1]["prompt"] == "Solve: sum it\nStart: def s():"
    assert out[2]["prompt"] == "Problem: add two ints"
    assert out[0]["prompt"] == "def f(): pass"


def test_resolve_rejects_unsupported_external_and_unresolved():
    bad = [{"question_id": "x", "category": "coding", "source": "https://huggingface.co/datasets/cais/hle/tree/1",
            "src_id": "h", "turns": ["t"]}]
    with pytest.raises(sb.ProvenanceError):
        sb.resolve_coding(bad, dict().__getitem__, dict().__getitem__)
    still = [{"question_id": "y", "category": "coding", "source": "inline", "src_id": "", "turns": [PH + " left"]}]
    with pytest.raises(sb.ProvenanceError):
        sb.resolve_coding(still, dict().__getitem__, dict().__getitem__)


def test_inline_rows_counts_exclusions():
    rows = [{"question_id": "a", "category": "high_entropy", "source": "https://www.gutenberg.org", "turns": ["t1"]},
            {"question_id": "b", "category": "mixed", "source": "https://huggingface.co/datasets/cais/hle/tree/1",
             "turns": ["t2"]}]
    keep, excl = sb.inline_rows(rows)
    assert [r["question_id"] for r in keep] == ["a"]
    assert sum(excl.values()) == 1 and any("cais/hle" in k for k in excl)


def test_digest_definition_matches_inferencex():
    prompts = ["héllo", "wörld"]
    assert sb.digest(prompts) == hashlib.sha256(json.dumps(prompts, ensure_ascii=False).encode()).hexdigest()


def test_stream_find_first_occurrence_wins():
    lines = [b"", b'{"question_id": "a", "question_content": "first", "starter_code": ""}',
             b'{"question_id": "b", "question_content": "bee", "starter_code": null}',
             b'{"question_id": "a", "question_content": "second", "starter_code": "x"}', b"not json but no match"]
    got = sb.stream_jsonl_find(lines, {"a", "b"})
    assert got == {"a": ("first", ""), "b": ("bee", "")}


def test_prepared_roundtrip_and_tamper_detection(tmp_path):
    rows = [{"question_id": "q", "prompt": "p1"}, {"question_id": "r", "prompt": "p2"}]
    path = sb._write(tmp_path, "unit-set", rows, {"dataset": "unit-set", "prompt_digest": sb.digest(["p1", "p2"])})
    got, prov = sb.load_prepared(path)
    assert got == rows and prov["file_sha256"]
    path.write_text(path.read_text().replace("p2", "pX"))
    with pytest.raises(sb.ProvenanceError):
        sb.load_prepared(path)
    (tmp_path / "unit-set.provenance.json").unlink()
    with pytest.raises(sb.ProvenanceError):
        sb.load_prepared(path)


def test_selection_is_seeded_and_cycles():
    a = datasets.select(80, 640, seed=0)
    assert a == datasets.select(80, 640, seed=0) and a != datasets.select(80, 640, seed=1)
    assert len(a) == 640 and len(set(a)) == 80 and all(a.count(i) == 8 for i in range(80))
    assert len(set(datasets.select(80, 40, seed=3))) == 40  # no repeats while num <= n
    with pytest.raises(ValueError):
        datasets.select(0, 10, 0)


class _Tok:
    def apply_chat_template(self, msgs, add_generation_prompt, tokenize):
        return "<u>" + msgs[0]["content"] + "</u>"

    def encode(self, s, add_special_tokens=False):
        return s.split()


def test_build_requests_metadata():
    rows = [{"question_id": f"q{i}", "prompt": "w " * (i + 1)} for i in range(5)]
    reqs, meta = datasets.build_requests(rows, _Tok(), num_prompts=12, seed=7, osl=64)
    assert len(reqs) == 12 and all(r[2] == 64 and r[3] is None for r in reqs)
    assert all(r[0].startswith("<u>") for r in reqs)
    assert meta["n_unique"] == 5 and meta["max_repeats"] == 3 and len(meta["question_ids"]) == 12


@pytest.mark.network
def test_prepared_coding_set_is_inferencex_identical():
    f = datasets.data_dir() / "speedbench-coding.jsonl"
    if not f.exists():
        pytest.skip("run `g4bench speedbench prepare --which coding` first")
    rows, prov = sb.load_prepared(f)
    assert len(rows) == 80 and prov["matches_inferencex_digest"] is True
    assert sb.digest([r["prompt"] for r in rows]) == sb.INFERENCEX_PROMPTS_SHA256
