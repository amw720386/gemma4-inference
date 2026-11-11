"""Apply named image patches to the installed engine: ``python apply.py <name>...`` (an image build step).

Standalone (stdlib only, no g4bench import) because it runs during the Modal image build, before the repo is
mounted. Each patch module in this directory defines NAME, MODULE, RELPATH and ``patch_text(src) -> (src, changed)``.
Any failure (unknown patch, engine not importable, anchor not found exactly once, result does not compile) exits
non-zero, which fails the image build.
"""

from __future__ import annotations

import importlib.util
import pathlib
import py_compile
import sys

HERE = pathlib.Path(__file__).resolve().parent
KNOWN = ("sm120_attn_tiles",)


def load_patch(name: str):
    if name not in KNOWN:
        raise SystemExit(f"unknown image patch {name!r}; known: {', '.join(KNOWN)}")
    spec = importlib.util.spec_from_file_location(f"g4patch_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def target_file(mod, package_dir: pathlib.Path | None = None) -> pathlib.Path:
    if package_dir is None:
        spec = importlib.util.find_spec(mod.MODULE)
        if spec is None or not spec.submodule_search_locations:
            raise SystemExit(f"{mod.NAME}: package {mod.MODULE!r} not importable by {sys.executable}")
        package_dir = pathlib.Path(list(spec.submodule_search_locations)[0])
    return package_dir / mod.RELPATH


def apply(name: str, package_dir: pathlib.Path | None = None) -> str:
    mod = load_patch(name)
    path = target_file(mod, package_dir)
    src = path.read_text(encoding="utf-8")
    out, changed = mod.patch_text(src)
    if changed:
        path.write_text(out, encoding="utf-8")
        py_compile.compile(str(path), doraise=True)
        # Drop a stale bytecode cache so the patched source is what gets imported.
        for pyc in path.parent.glob(f"__pycache__/{path.stem}.*.pyc"):
            pyc.unlink()
    return f"{name}: {'patched' if changed else 'already applied'} {path}"


def main(argv: list[str]) -> int:
    if not argv:
        raise SystemExit("usage: apply.py <patch-name>...")
    for name in argv:
        try:
            print(apply(name), flush=True)
        except Exception as e:  # noqa: BLE001 - any failure must fail the image build
            print(f"image patch {name} FAILED: {e}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
