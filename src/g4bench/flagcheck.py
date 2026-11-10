"""Static preflight: every server flag in configs/engines/*.yaml must exist in the pinned engine source.

Why: a typo'd flag kills the server at startup on a paid GPU. This check costs nothing.

How: fetch the argument-definition sources at the pinned git tags (raw.githubusercontent.com, HTTPS),
extract flag names with ``ast`` (no engine import, no GPU), and save a snapshot under
``configs/engine_flags/<engine>-<version>.json`` with per-file sha256. Tests check configs against the
committed snapshot offline; ``g4bench flagcheck --refresh`` regenerates it.

Extraction rules (mirroring each engine's own CLI construction at the pinned tag):
* vLLM v0.33.0: every ``"--..."`` string literal passed to ``add_argument`` in ``vllm/engine/arg_utils.py``
  and ``vllm/entrypoints/common/cli_args.py``, plus ``--<field>`` for dataclass fields of the frontend arg
  classes in cli_args.py; bool options use ``BooleanOptionalAction`` so ``--no-<flag>`` is accepted when
  ``--<flag>`` exists. ``--speculative-config`` JSON keys are checked against ``SpeculativeConfig`` fields.
* SGLang v0.5.23: for every annotated field in ``python/sglang/srt/arg_groups/fields/*.py``:
  ``Arg(cli_name=...)`` or ``--<field-with-dashes>`` plus ``Arg(aliases=[...])``
  (``arg_groups/arg_utils.py::_field_to_cli_name``).
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any

from .config import EngineConfig
from .paths import CONFIGS

RAW = "https://raw.githubusercontent.com"
SOURCES = {
    "vllm": {"repo": "vllm-project/vllm", "tag": "v0.33.0",
             "files": ["vllm/engine/arg_utils.py", "vllm/entrypoints/common/cli_args.py", "vllm/config/speculative.py"]},
    "sglang": {"repo": "sgl-project/sglang", "tag": "v0.5.23",
               "files": [f"python/sglang/srt/arg_groups/fields/{n}.py" for n in
                         ("device", "disagg", "exec_", "lora", "memory", "mm", "model", "observability", "parallel",
                          "schedule", "serving", "spec")] + ["python/sglang/srt/arg_groups/arg_utils.py"]},
}
SNAP_DIR = CONFIGS / "engine_flags"


def _str_list(node: ast.AST) -> list[str]:
    if isinstance(node, (ast.List, ast.Tuple)):
        return [e.value for e in node.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)]
    return []


def vllm_flags(files: dict[str, str]) -> tuple[set[str], set[str]]:
    flags: set[str] = set()
    spec_fields: set[str] = set()
    for name, src in files.items():
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "add_argument":
                for a in node.args:
                    if isinstance(a, ast.Constant) and isinstance(a.value, str) and a.value.startswith("-"):
                        flags.add(a.value)
            if name.endswith("cli_args.py") and isinstance(node, ast.ClassDef) and "Args" in node.name:
                for st in node.body:
                    if isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name):
                        flags.add("--" + st.target.id.replace("_", "-"))
            if name.endswith("speculative.py") and isinstance(node, ast.ClassDef) and node.name == "SpeculativeConfig":
                for st in node.body:
                    if isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name):
                        spec_fields.add(st.target.id)
    return flags, spec_fields


def sglang_flags(files: dict[str, str]) -> set[str]:
    flags: set[str] = set()
    for name, src in files.items():
        if "/fields/" not in name:
            continue
        for node in ast.walk(ast.parse(src)):
            if not isinstance(node, ast.ClassDef):
                continue
            for st in node.body:
                if not (isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name)):
                    continue
                cli, aliases = None, []
                for sub in ast.walk(st.annotation):
                    if isinstance(sub, ast.Call) and getattr(sub.func, "id", None) == "Arg":
                        for kw in sub.keywords:
                            if kw.arg == "cli_name" and isinstance(kw.value, ast.Constant):
                                cli = kw.value.value
                            if kw.arg == "aliases":
                                aliases = _str_list(kw.value)
                flags.add(cli or "--" + st.target.id.replace("_", "-"))
                flags.update(aliases)
    return flags


def fetch(engine: str, timeout: float = 60) -> dict[str, Any]:
    import requests

    s = SOURCES[engine]
    files, shas = {}, {}
    for f in s["files"]:
        r = requests.get(f"{RAW}/{s['repo']}/{s['tag']}/{f}", timeout=timeout)
        r.raise_for_status()
        files[f] = r.text
        shas[f] = hashlib.sha256(r.content).hexdigest()
    if engine == "vllm":
        flags, spec = vllm_flags(files)
        snap = {"flags": sorted(flags), "speculative_config_fields": sorted(spec)}
    else:
        snap = {"flags": sorted(sglang_flags(files))}
    return {"engine": engine, "repo": s["repo"], "tag": s["tag"], "file_sha256": shas, **snap}


def snapshot_path(engine: str, version: str) -> Path:
    return SNAP_DIR / f"{engine}-{version}.json"


def refresh(engine: str) -> Path:
    snap = fetch(engine)
    version = snap["tag"].lstrip("v")
    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    p = snapshot_path(engine, version)
    p.write_text(json.dumps(snap, indent=1) + "\n")
    return p


def check(cfg: EngineConfig, snap: dict[str, Any], rendered: list[str]) -> list[str]:
    """Problems with a rendered argv (empty = ok)."""
    known = set(snap["flags"])
    problems = []
    for i, tok in enumerate(rendered):
        if not tok.startswith("--"):
            continue
        name = tok.split("=", 1)[0]
        ok = name in known or (name.startswith("--no-") and "--" + name[5:] in known)
        if not ok:
            problems.append(f"unknown flag {name} for {cfg.engine} {cfg.version}")
        if name == "--speculative-config" and "speculative_config_fields" in snap:
            try:
                spec = json.loads(rendered[i + 1])
            except (IndexError, json.JSONDecodeError) as e:
                problems.append(f"--speculative-config is not valid JSON: {e}")
                continue
            bad = sorted(set(spec) - set(snap["speculative_config_fields"]))
            if bad:
                problems.append(f"unknown SpeculativeConfig keys {bad}")
    return problems


def check_all() -> dict[str, list[str]]:
    from .config import load_pins, placeholder_values

    pins = load_pins()
    out: dict[str, list[str]] = {}
    for f in sorted((CONFIGS / "engines").glob("*.yaml")):
        cfg = EngineConfig.load(str(f))
        if cfg.engine == "mock":
            continue
        p = snapshot_path(cfg.engine, cfg.version)
        if not p.exists():
            out[cfg.name] = [f"no flag snapshot for {cfg.engine} {cfg.version}; run `g4bench flagcheck --refresh`"]
            continue
        out[cfg.name] = check(cfg, json.loads(p.read_text()), cfg.render_args(placeholder_values(pins, 10240, cfg.port)))
    return out
