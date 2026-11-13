"""Engine-image identities, kept Modal-free so routing, hashing and the worker checks share one definition.

``modal_app.py`` builds the images from these constants; ``modal_run.py`` routes configs with them; ``runner.py``
uses them to tag a config that runs on an image other than its home image (paired runs, D-061).

Images (all from the digest-pinned vLLM image, configs/pins.yaml):

* ``vllm``              unmodified.
* ``vllm-patched``      VLLM_PATCHES applied + ``VLLM_PATCH_PIP`` (cuBLAS 13.10) upgraded IN the image (D-052). Kept
                        byte-for-byte in its build steps so the E8a series (D-060) stays reproducible.
* ``vllm-patched-exp``  VLLM_PATCHES_EXP applied; the image's own cuBLAS (13.1) is left in place and cuBLAS 13.10 is
                        side-installed under ``CUBLAS_SIDE`` for configs that ask for it (``G4_CUBLAS``, D-061). Every
                        patch is env-gated, so the article's stock recipe runs here unchanged (paired mode).
"""

from __future__ import annotations

from typing import Iterable, Mapping

# The one patch set the production patched image is built with. Configs opt in by listing patches in
# ``image_patches`` (a subset) AND setting the patch's env gate; with no gate set, every patch is upstream behaviour.
# sm120_attn_tiles retired: no gain (D-046).
VLLM_PATCHES: tuple[str, ...] = ("mtp_draft_prefill_last", "splitk_per_group", "mtp_draft_head_q")
# Patches active without an env gate. Empty since D-061 (mtp_draft_prefill_last is gated by G4_DRAFT_PREFILL_LAST).
ALWAYS_ON_PATCHES: frozenset[str] = frozenset()
# Library upgrades installed INTO the production patched image (lossless; results/kernels/*cublas_upgrade*).
VLLM_PATCH_PIP: tuple[str, ...] = ("nvidia-cublas==13.10.0.20",)
# Experimental image: VLLM_PATCHES + patches still under evaluation (D-058).
VLLM_PATCHES_EXP: tuple[str, ...] = VLLM_PATCHES + ("prefill_attn_g4",)
# Experimental image only: cuBLAS builds installed next to (not over) the image's own, selected per config through
# ``G4_CUBLAS: "<version>"`` (runner -> cublas.resolve -> LD_PRELOAD). version -> (pip requirement, target dir).
CUBLAS_SIDE: dict[str, tuple[str, str]] = {"13.10": ("nvidia-cublas==13.10.0.20", "/opt/cublas-13.10")}


def side_env(side: Mapping[str, tuple[str, str]] = CUBLAS_SIDE) -> str:
    """Value of the image env ``G4_CUBLAS_SIDE``: ``ver=dir[,ver=dir]``."""
    return ",".join(f"{v}={d}" for v, (_, d) in sorted(side.items()))


def home_image(engine: str, patches: Iterable[str] | None) -> str | None:
    """The image a config was designed for: unpatched -> its engine image; else the smallest patched image."""
    p = set(patches or ())
    if not p:
        return engine
    if engine != "vllm":
        return None
    if p <= set(VLLM_PATCHES):
        return "vllm-patched"
    if p <= set(VLLM_PATCHES_EXP):
        return "vllm-patched-exp"
    return None


def runtime_image(environ: Mapping[str, str]) -> str | None:
    """The image this process runs in, from the env the images set; None outside a patched image (or unknown)."""
    if "G4_IMAGE_PATCHES" not in environ:
        return None
    have = tuple(p for p in environ["G4_IMAGE_PATCHES"].split(",") if p)
    if have == VLLM_PATCHES:
        return "vllm-patched"
    if have == VLLM_PATCHES_EXP:
        return "vllm-patched-exp"
    return "vllm-patched:" + ",".join(have)


def foreign_image_tag(engine: str, patches: Iterable[str] | None, environ: Mapping[str, str]) -> str | None:
    """Name of the runtime image when it differs from the config's home image, else None.

    Used in the config hash so a config measured on a foreign image (e.g. the stock recipe on the experimental image
    in a paired run) never pools with its runs on the home image. None for every historical run (each ran at home)."""
    rt = runtime_image(environ)
    if rt is None:
        return None
    return None if rt == home_image(engine, patches) else rt
