"""Prepare SPEED-Bench real-prompt sets with verifiable provenance.

Two kinds of prompt set are produced:

``speedbench-coding`` (primary, InferenceX-standard)
    The 80 first-user-turns of the ``coding`` category of nvidia/SPEED-Bench's
    qualitative split, in dataset order, resolved exactly as NeMo-Skills' pinned
    ``prepare.py`` does. This is the set InferenceX uses for its golden
    acceptance-length curves (inferencex-e2e/infx/bench/speedbench.py @ 0b991451).
    We do NOT run prepare.py (it needs ``datasets`` and ~35 GB of downloads); we
    re-implement its two coding branches (code_contests, livecodebench) as pure
    functions, fetch only the 10 rows they need from the same pinned revisions,
    and refuse to write anything unless the result's digest equals InferenceX's
    ``PROMPTS_SHA256``. A match proves byte-identity with the InferenceX set.

``speedbench-tp1k-inline`` / ``speedbench-tp8k-inline`` (secondary, labelled SUBSET)
    Rows of the ``throughput_1k`` / ``throughput_8k`` splits whose text is stored
    inline (no external resolution). This drops the ``mixed`` category and some
    ``high_entropy`` rows (HLE is gated; AdaLEval/BAMBOO need external fetches), so
    it is a biased subset of the official throughput split and is reported as such.
    Its value is matched input length (≈1K / ≈8K) with real text.

Every prepared file gets a sidecar ``*.provenance.json`` with dataset revisions,
per-file sha256 (verified against the Hub's LFS hashes), the prompt digest, row
counts, excluded sources and tool versions. Prompt text itself is NOT committed
(third-party licences; see docs/SOURCES.md) - it is regenerated deterministically.
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ---- pins (InferenceX speedbench.py @ 0b991451 + HF Hub LFS metadata, verified 2026-11-10) ----
SPEED_BENCH_REPO = "nvidia/SPEED-Bench"
SPEED_BENCH_REVISION = "454f88454792dfa3ccfd7ef15fff248efde44cd1"
PREPARE_PY = ("https://raw.githubusercontent.com/NVIDIA-NeMo/Skills/5ac8609a56ac941540b10c92e68d556e6343cd4c/"
              "nemo_skills/dataset/speed-bench/prepare.py")
PREPARE_PY_SHA256 = "a551be4df541474e54e21b480022b0cbb66c2da068fda61b2a64bd3223bbbed2"
INFERENCEX_PROMPTS_SHA256 = "6f1845452f1cfbc1930ea779ba6c3d5b437a9d5853eca199dd9b7619b20a0c18"
INFERENCEX_COMMIT = "0b991451d71d7dca4b478fad09a28f14c43fe804"
TURNS_PLACEHOLDER = "FULL BENCHMARK DATA SHOULD BE FETCHED FROM THE SOURCE USING SPECDEC_BENCH"

SPLIT_FILES = {  # split -> (path in repo, LFS sha256)
    "qualitative": ("qualitative/test-00000-of-00001.parquet",
                    "4f76bc45bdffb38712a5c26d7a7c9a9a791484b392a7b64a6ffb9abe48e02e26"),
    "throughput_1k": ("throughput_1k/test-00000-of-00001.parquet",
                      "edd654704de36b4d01d03da00d7ea138422b46507e1d0a2b9c48c14abd565f2f"),
    "throughput_8k": ("throughput_8k/test-00000-of-00001.parquet",
                      "11fb562ad82bab6af8139135ac18453230bf3233dee3228235634e5fc4939a78"),
}
CODE_CONTESTS = ("deepmind/code_contests", "802411c3010cb00d1b05bad57ca77365a3c699d6",
                 "data/test-00000-of-00001-9c49eeff30aacaa8.parquet",
                 "aa426cbdb202bf8703b658bcb31fd1878ca7cfd33ca07d3b703dc94ca6a2b651")
LIVECODEBENCH_REPO = "livecodebench/code_generation_lite"
LIVECODEBENCH_REVISION = "0fe84c3912ea0c4d4a78037083943e8f0c4dd505"
LIVECODEBENCH_FILES = [  # prepare.py concatenates these in this order; first match wins (.iloc[0])
    ("test.jsonl", "2bd02b38beb48e8c46b5b9987095d999ff38cd8efc255ea5d58974317c48f63f"),
    ("test2.jsonl", "095df7c5daf15f882c51a9deb84085cff1e073495a5dbcf95015a564d485f3a3"),
    ("test3.jsonl", "28ed26cc83363ce3f1fe2d5fad9f8393077beb1907b167a31bd3b32f80801b79"),
    ("test4.jsonl", "d711138ddaebfcf5f8ec6a4283ee677298c0f5c5d374a235af92aaf0584510da"),
    ("test5.jsonl", "7f77571c2a6df0c2a72a3277650309f67e01e0008e18117e624633df53f81214"),
    ("test6.jsonl", "bb4c364f71921c4495a6ad15abe1a927350b720009f4933e2e71f8af0f6fd1f5"),
]
# Source substrings that prepare.py's _fetch_all_turns_data resolves externally (BenchmarkDataset values).
EXTERNAL_SOURCE_KEYS = (
    "RUCAIBox/BAMBOO", "abisee/cnn_dailymail", "cais/hle", "livecodebench/code_generation_lite",
    "deepmind/code_contests", "mtbench101/mt-bench-101", "Helsinki-NLP/opus-100", "nvidia/ChatRAG-Bench",
    "TIGER-Lab/MMLU-Pro", "AdaLEval/stackselect", "AdaLEval/textsort", "ZenMoore/RoleBench", "Neph0s/CoSER",
)

HF = "https://huggingface.co"
DATASETS = ("speedbench-coding", "speedbench-tp1k-inline", "speedbench-tp8k-inline")


class ProvenanceError(RuntimeError):
    """A downloaded artefact or the resolved prompt set does not match its pin."""


def digest(prompts: list[str]) -> str:
    """InferenceX's digest: sha256 of the prompts as one UTF-8 JSON array (ensure_ascii=False)."""
    return hashlib.sha256(json.dumps(prompts, ensure_ascii=False).encode()).hexdigest()


def is_external(source: str) -> bool:
    return any(k in source for k in EXTERNAL_SOURCE_KEYS)


# ---------------------------------------------------------------- pure resolution logic

def resolve_coding(rows: list[dict[str, Any]], code_contests: Callable[[str], str],
                   livecodebench: Callable[[str], tuple[str, str]]) -> list[dict[str, str]]:
    """Mirror prepare.py for the qualitative ``coding`` rows (dataset order preserved).

    ``code_contests(name) -> description``; ``livecodebench(question_id) -> (question_content, starter_code)``.
    Rows from any other external source raise, so a silent partial resolution is impossible.
    """
    out = []
    for row in rows:
        if row["category"] != "coding":
            continue
        src, turn0 = row["source"], row["turns"][0]
        if "livecodebench/code_generation_lite" in src:
            q, starter = livecodebench(row["src_id"])
            turn0 = turn0.removeprefix(f"{TURNS_PLACEHOLDER}\n\n").format(question=q, starter_code=starter)
        elif "deepmind/code_contests" in src:
            turn0 = turn0.removeprefix(f"{TURNS_PLACEHOLDER}\n\n").format(question=code_contests(row["src_id"]))
        elif is_external(src):
            raise ProvenanceError(f"coding row {row['question_id']} needs unsupported source {src}")
        if TURNS_PLACEHOLDER in turn0:
            raise ProvenanceError(f"unresolved placeholder in {row['question_id']}")
        out.append({"question_id": row["question_id"], "prompt": turn0})
    return out


def inline_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, str]], dict[str, int]]:
    """Rows usable without external data; plus counts of excluded rows by (category, source)."""
    keep, excluded = [], {}
    for row in rows:
        if is_external(row["source"]) or TURNS_PLACEHOLDER in row["turns"][0]:
            k = f"{row['category']} | {row['source']}"
            excluded[k] = excluded.get(k, 0) + 1
            continue
        keep.append({"question_id": row["question_id"], "prompt": row["turns"][0], "category": row["category"]})
    return keep, excluded


# ---------------------------------------------------------------- fetching with verification

def _resolve_url(repo: str, revision: str, path: str) -> str:
    return f"{HF}/datasets/{repo}/resolve/{revision}/{path}"


def fetch_verified(repo: str, revision: str, path: str, sha256: str, cache: Path, timeout: float = 120) -> Path:
    """Download one file (cached by sha256) and verify it."""
    import requests

    cache.mkdir(parents=True, exist_ok=True)
    dest = cache / f"{sha256}-{Path(path).name}"
    if dest.exists() and _sha256_file(dest) == sha256:
        return dest
    h = hashlib.sha256()
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(_resolve_url(repo, revision, path), stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                h.update(chunk)
                f.write(chunk)
    if h.hexdigest() != sha256:
        tmp.unlink(missing_ok=True)
        raise ProvenanceError(f"{repo}@{revision}/{path}: sha256 {h.hexdigest()} != pinned {sha256}")
    tmp.replace(dest)
    return dest


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_parquet_rows(path: Path, columns: list[str] | None = None) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    return pq.read_table(path, columns=columns).to_pylist()


def stream_jsonl_find(lines: Iterable[bytes], wanted: set[str], key: str = "question_id",
                      fields: tuple[str, ...] = ("question_content", "starter_code")) -> dict[str, tuple]:
    """First occurrence of each wanted key in a JSONL byte-line stream (cheap substring pre-filter)."""
    found: dict[str, tuple] = {}
    for raw in lines:
        if not raw.strip():
            continue
        if not any(w.encode() in raw for w in wanted - set(found)):
            continue
        rec = json.loads(raw)
        k = rec.get(key)
        if k in wanted and k not in found:
            found[k] = tuple(rec.get(f) or "" for f in fields)
    return found


@dataclass
class StreamReport:
    file: str
    complete: bool
    sha256: str | None
    verified: bool | None


def find_livecodebench(ids: set[str], timeout: float = 600) -> tuple[dict[str, tuple[str, str]], list[StreamReport]]:
    """Stream the six pinned LiveCodeBench files in prepare.py order; stop once all ids are found.

    Files streamed to the end are hash-verified; a file abandoned early cannot be hashed, but by
    construction it can only contain later (ignored) duplicates, so the result is unaffected.
    """
    import requests

    found: dict[str, tuple[str, str]] = {}
    reports: list[StreamReport] = []
    for name, pin in LIVECODEBENCH_FILES:
        if ids <= set(found):
            reports.append(StreamReport(name, False, None, None))
            continue
        h = hashlib.sha256()
        with requests.get(_resolve_url(LIVECODEBENCH_REPO, LIVECODEBENCH_REVISION, name), stream=True,
                          timeout=timeout) as r:
            r.raise_for_status()

            def lines() -> Iterator[bytes]:
                buf = b""
                for chunk in r.iter_content(1 << 20):
                    h.update(chunk)
                    buf += chunk
                    *full, buf = buf.split(b"\n")
                    yield from full
                if buf:
                    yield buf

            for k, v in stream_jsonl_find(lines(), ids - set(found)).items():
                found.setdefault(k, v)  # type: ignore[arg-type]
        got = h.hexdigest()
        reports.append(StreamReport(name, True, got, got == pin))
        if got != pin:
            raise ProvenanceError(f"{LIVECODEBENCH_REPO}/{name}: sha256 {got} != pinned {pin}")
    return found, reports


# ---------------------------------------------------------------- prepare + write

def _versions() -> dict[str, str | None]:
    from importlib import metadata as md

    out: dict[str, str | None] = {"python": platform.python_version()}
    for p in ("pyarrow", "requests", "g4bench"):
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            out[p] = None
    return out


def _write(out_dir: Path, name: str, rows: list[dict[str, Any]], prov: dict[str, Any]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    prov["file"] = path.name
    prov["file_sha256"] = _sha256_file(path)
    (out_dir / f"{name}.provenance.json").write_text(json.dumps(prov, indent=2, ensure_ascii=False))
    return path


def prepare_coding(out_dir: Path, cache: Path, *, require_digest: bool = True) -> dict[str, Any]:
    t0 = time.time()
    qpath, qsha = SPLIT_FILES["qualitative"]
    qfile = fetch_verified(SPEED_BENCH_REPO, SPEED_BENCH_REVISION, qpath, qsha, cache)
    rows = [r for r in read_parquet_rows(qfile) if r["category"] == "coding"]
    cc_names = {r["src_id"] for r in rows if "deepmind/code_contests" in r["source"]}
    lcb_ids = {r["src_id"] for r in rows if "livecodebench/code_generation_lite" in r["source"]}
    cc_file = fetch_verified(CODE_CONTESTS[0], CODE_CONTESTS[1], CODE_CONTESTS[2], CODE_CONTESTS[3], cache)
    cc: dict[str, str] = {}
    for r in read_parquet_rows(cc_file, columns=["name", "description"]):  # first match, like .iloc[0]
        if r["name"] in cc_names and r["name"] not in cc:
            cc[r["name"]] = r["description"]
    lcb, reports = find_livecodebench(lcb_ids)
    missing = (cc_names - set(cc)) | (lcb_ids - set(lcb))
    if missing:
        raise ProvenanceError(f"external rows not found: {sorted(missing)}")
    resolved = resolve_coding(rows, cc.__getitem__, lcb.__getitem__)
    dg = digest([r["prompt"] for r in resolved])
    ok = dg == INFERENCEX_PROMPTS_SHA256
    if require_digest and not ok:
        raise ProvenanceError(f"{len(resolved)} prompts digest {dg} != InferenceX {INFERENCEX_PROMPTS_SHA256}")
    prov = {
        "dataset": "speedbench-coding", "n_prompts": len(resolved), "prompt_digest": dg,
        "matches_inferencex_digest": ok, "inferencex_prompts_sha256": INFERENCEX_PROMPTS_SHA256,
        "inferencex_commit": INFERENCEX_COMMIT, "method": "re-implemented prepare.py coding branches; digest-gated",
        "prepare_py": {"url": PREPARE_PY, "sha256": PREPARE_PY_SHA256},
        "sources": {
            "speed_bench": {"repo": SPEED_BENCH_REPO, "revision": SPEED_BENCH_REVISION, "file": qpath, "sha256": qsha},
            "code_contests": {"repo": CODE_CONTESTS[0], "revision": CODE_CONTESTS[1], "file": CODE_CONTESTS[2],
                              "sha256": CODE_CONTESTS[3], "rows_used": sorted(cc_names)},
            "livecodebench": {"repo": LIVECODEBENCH_REPO, "revision": LIVECODEBENCH_REVISION,
                              "rows_used": sorted(lcb_ids), "files": [r.__dict__ for r in reports]},
        },
        "source_counts": _count(rows), "prepared_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seconds": round(time.time() - t0, 1), "versions": _versions(),
    }
    _write(out_dir, "speedbench-coding", resolved, prov)
    return prov


def prepare_throughput_inline(split: str, out_dir: Path, cache: Path) -> dict[str, Any]:
    path, sha = SPLIT_FILES[split]
    f = fetch_verified(SPEED_BENCH_REPO, SPEED_BENCH_REVISION, path, sha, cache)
    rows = read_parquet_rows(f)
    keep, excluded = inline_rows(rows)
    name = {"throughput_1k": "speedbench-tp1k-inline", "throughput_8k": "speedbench-tp8k-inline"}[split]
    prov = {
        "dataset": name, "n_prompts": len(keep), "n_rows_in_split": len(rows),
        "prompt_digest": digest([r["prompt"] for r in keep]), "matches_inferencex_digest": None,
        "SUBSET_WARNING": ("Inline-only subset of the official split: externally-resolved rows are excluded, "
                           "so category mix differs from SPEED-Bench throughput. Not an official SPEED-Bench result."),
        "kept_by_category": _cat(keep), "excluded_by_source": excluded,
        "sources": {"speed_bench": {"repo": SPEED_BENCH_REPO, "revision": SPEED_BENCH_REVISION, "file": path, "sha256": sha}},
        "prepared_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "versions": _versions(),
    }
    _write(out_dir, name, keep, prov)
    return prov


def _count(rows: list[dict[str, Any]]) -> dict[str, int]:
    c: dict[str, int] = {}
    for r in rows:
        c[r["source"]] = c.get(r["source"], 0) + 1
    return c


def _cat(rows: list[dict[str, Any]]) -> dict[str, int]:
    c: dict[str, int] = {}
    for r in rows:
        c[r["category"]] = c.get(r["category"], 0) + 1
    return c


def load_prepared(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load a prepared JSONL + provenance, re-verifying the file hash and prompt digest."""
    prov_path = path.with_name(path.name.removesuffix(".jsonl") + ".provenance.json")
    if not prov_path.exists():
        raise ProvenanceError(f"missing provenance sidecar {prov_path.name}")
    prov = json.loads(prov_path.read_text())
    if _sha256_file(path) != prov.get("file_sha256"):
        raise ProvenanceError(f"{path.name}: file hash differs from provenance (edited after preparation?)")
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if digest([r["prompt"] for r in rows]) != prov.get("prompt_digest"):
        raise ProvenanceError(f"{path.name}: prompt digest mismatch")
    return rows, prov


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/speedbench"))
    ap.add_argument("--cache", type=Path, default=Path(".cache/speedbench"))
    ap.add_argument("--which", default="all", choices=["all", "coding", "tp1k", "tp8k"])
    a = ap.parse_args()
    res = {}
    if a.which in ("all", "tp1k"):
        res["tp1k"] = prepare_throughput_inline("throughput_1k", a.out, a.cache)
    if a.which in ("all", "tp8k"):
        res["tp8k"] = prepare_throughput_inline("throughput_8k", a.out, a.cache)
    if a.which in ("all", "coding"):
        res["coding"] = prepare_coding(a.out, a.cache)
    print(json.dumps({k: {kk: v.get(kk) for kk in ("n_prompts", "prompt_digest", "matches_inferencex_digest")}
                      for k, v in res.items()}, indent=2))
