"""Expand a matrix config into benchmark points; generate run IDs."""

from __future__ import annotations

import datetime as _dt
import secrets
from dataclasses import asdict, dataclass
from typing import Any

from .config import load_yaml, resolve_config


@dataclass(frozen=True)
class Point:
    isl: int
    osl: int
    conc: int
    seed: int
    trial: int
    num_prompts: int
    num_warmups: int
    random_range_ratio: float
    use_chat_template: bool
    dataset: str = "random"       # "random" (InferenceX synthetic) or a prepared real-prompt set
    ignore_eos: bool = True

    @property
    def key(self) -> str:
        """Workload key shared across seeds/trials (matches target JSON ``{isl}x{conc}``).

        Real-prompt arms get a dataset suffix so they can never be pooled with, or compared
        against, the article's random-token targets.
        """
        base = f"{self.isl}x{self.conc}"
        return base if self.dataset == "random" else f"{base}@{self.dataset}"

    @property
    def point_id(self) -> str:
        pid = f"isl{self.isl}-osl{self.osl}-c{self.conc}-s{self.seed}-t{self.trial}"
        if self.dataset != "random":
            pid += f"-{self.dataset}" + ("" if self.ignore_eos else "-eos")
        return pid

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["key"] = self.key
        d["point_id"] = self.point_id
        return d


def load_matrix(name_or_path: str) -> dict[str, Any]:
    m = load_yaml(resolve_config(name_or_path, "matrix"))
    for k in ("name", "isl", "osl", "conc", "seeds"):
        if k not in m:
            raise ValueError(f"matrix missing {k!r}")
    return m


def expand(matrix: dict[str, Any], seeds: list[int] | None = None, trials: int | None = None,
           only_keys: list[str] | None = None) -> list[Point]:
    """Points ordered trial-major then seed, then ISL, then conc (cheap points first).

    Ordering matters: interleaving seeds/trials across the whole matrix (rather than
    repeating one point back-to-back) spreads slow drift (thermal, neighbours) over
    all points instead of biasing one.
    """
    seeds = list(matrix["seeds"] if seeds is None else seeds)
    trials = int(matrix.get("trials_per_seed", 1) if trials is None else trials)
    npc = int(matrix.get("num_prompts_per_conc", 10))
    nwc = int(matrix.get("num_warmups_per_conc", 2))
    rr = float(matrix.get("random_range_ratio", 1.0))
    chat = bool(matrix.get("use_chat_template", True))
    ign = bool(matrix.get("ignore_eos", True))
    by_isl = {int(k): v for k, v in (matrix.get("dataset_by_isl") or {}).items()}
    default_ds = matrix.get("dataset", "random")
    pts: list[Point] = []
    for t in range(trials):
        for s in seeds:
            for isl in matrix["isl"]:
                for osl in matrix["osl"]:
                    for c in matrix["conc"]:
                        p = Point(isl=int(isl), osl=int(osl), conc=int(c), seed=int(s), trial=t,
                                  num_prompts=npc * int(c), num_warmups=nwc * int(c),
                                  random_range_ratio=rr, use_chat_template=chat,
                                  dataset=by_isl.get(int(isl), default_ds), ignore_eos=ign)
                        if only_keys and p.key not in only_keys:
                            continue
                        pts.append(p)
    return pts


def new_run_id(engine_name: str, hw_tag: str, cfg_hash: str, now: _dt.datetime | None = None) -> str:
    now = now or _dt.datetime.now(_dt.timezone.utc)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    safe_hw = "".join(ch if ch.isalnum() else "-" for ch in hw_tag.lower()).strip("-") or "unknown"
    return f"{stamp}-{engine_name}-{safe_hw}-{cfg_hash[:8]}-{secrets.token_hex(2)}"
