"""Load pinned configs, render engine command lines, and hash configurations.

A *config hash* identifies everything that should change measured performance
(pins + engine args + matrix client policy). Results are only pooled across
runs that share the same hash.
"""

from __future__ import annotations

import hashlib
import re
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .paths import CONFIGS


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping at top level")
    return data


def load_pins(path: str | Path | None = None) -> dict[str, Any]:
    return load_yaml(path or CONFIGS / "pins.yaml")


def resolve_config(name_or_path: str, kind: str) -> Path:
    """Accept either a path or a bare name under configs/<kind>/."""
    p = Path(name_or_path)
    if p.exists():
        return p
    cand = CONFIGS / kind / f"{name_or_path}.yaml"
    if cand.exists():
        return cand
    raise FileNotFoundError(f"no {kind} config named {name_or_path!r} (looked for {cand})")


@dataclass(frozen=True)
class EngineConfig:
    name: str
    engine: str
    version: str
    client_backend: str
    endpoint: str
    port: int
    health_path: str
    args: tuple[tuple[str, ...], ...]
    env: dict[str, str] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict, compare=False, hash=False)

    @classmethod
    def load(cls, name_or_path: str) -> "EngineConfig":
        path = resolve_config(name_or_path, "engines")
        d = load_yaml(path)
        for key in ("name", "engine", "version", "client_backend", "endpoint", "port", "args"):
            if key not in d:
                raise ValueError(f"{path}: missing required key {key!r}")
        if d["engine"] not in ("vllm", "sglang", "mock"):
            raise ValueError(f"{path}: unsupported engine {d['engine']!r}")
        args = tuple(tuple(str(x) for x in a) for a in d["args"])
        return cls(
            name=d["name"],
            engine=d["engine"],
            version=str(d["version"]),
            client_backend=d["client_backend"],
            endpoint=d["endpoint"],
            port=int(d["port"]),
            health_path=d.get("health_path", "/health"),
            args=args,
            env={str(k): str(v) for k, v in (d.get("env") or {}).items()},
            raw=d,
        )

    def render_args(self, values: dict[str, Any]) -> list[str]:
        """Substitute ``{placeholders}`` and flatten to argv (without the launcher prefix)."""
        out: list[str] = []
        for group in self.args:
            for tok in group:
                out.append(_render(tok, values))
        return out

    def launcher(self) -> list[str]:
        if self.engine == "vllm":
            return ["vllm", "serve"]
        if self.engine == "sglang":
            return ["python3", "-m", "sglang.launch_server"]
        import sys

        return [sys.executable, "-m", "g4bench.mockserver"]


def _render(tok: str, values: dict[str, Any]) -> str:
    # Only replace known {name} placeholders; JSON braces ({"key":...}) never match [a-z_]+.
    out = tok
    for k, v in values.items():
        out = out.replace("{" + k + "}", str(v))
    m = _PLACEHOLDER.search(out)
    if m:
        raise KeyError(f"unresolved placeholder {m.group(0)} in {tok!r}")
    return out


_PLACEHOLDER = re.compile(r"\{([a-z_]+)\}")


def placeholder_values(pins: dict[str, Any], max_model_len: int, port: int) -> dict[str, Any]:
    return {
        "target_repo": pins["model"]["target"]["repo"],
        "target_rev": pins["model"]["target"]["revision"],
        "draft_repo": pins["model"]["draft"]["repo"],
        "draft_rev": pins["model"]["draft"]["revision"],
        "max_model_len": max_model_len,
        "port": port,
    }


def stable_hash(obj: Any, n: int = 12) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:n]


def config_hash(engine: EngineConfig, matrix: dict[str, Any], pins: dict[str, Any]) -> str:
    client_policy = {k: matrix.get(k) for k in (
        "random_range_ratio", "use_chat_template", "num_prompts_per_conc", "num_warmups_per_conc",
        "max_model_len",
    )}
    return stable_hash({
        "engine": {"name": engine.name, "args": engine.args, "env": engine.env, "version": engine.version},
        "client_policy": client_policy,
        "pins": pins,
    })
