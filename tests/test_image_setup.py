"""Regression for the failed deploy 'unable to determine the version of Python installed in the Image' (D-038)."""
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from g4bench.image_setup import python_setup_commands
from g4bench.paths import REPO

HERE_VER = "%d.%d" % sys.version_info[:2]


def _run(cmd: str, path_dirs: list[Path]) -> subprocess.CompletedProcess:
    """Run a Dockerfile RUN line under /bin/sh with an isolated PATH (no host python can leak in)."""
    assert cmd.startswith("RUN ")
    tools = path_dirs[0].parent / "tools"
    tools.mkdir(exist_ok=True)
    if not (tools / "ln").exists():
        (tools / "ln").symlink_to(shutil.which("ln"))
    env = {"PATH": ":".join(str(p) for p in [*path_dirs, tools]), "HOME": os.environ.get("HOME", "/tmp")}
    return subprocess.run(["/bin/sh", "-c", cmd[4:]], env=env, capture_output=True, text=True)


def _fake_image(tmp_path: Path, *, with_python: bool = False) -> tuple[Path, Path]:
    """A PATH dir with python3 (this interpreter, like the vLLM image) and an empty bin dir for the alias."""
    img_bin, alias_dir = tmp_path / "img_bin", tmp_path / "usr_local_bin"
    img_bin.mkdir()
    alias_dir.mkdir()
    # exec wrapper (keeps this venv's context; a bare symlink would lose pyvenv.cfg and its packages)
    for name in ("python3",) + (("python",) if with_python else ()):
        w = img_bin / name
        w.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
        w.chmod(0o755)
    return img_bin, alias_dir


def test_aliases_python_to_the_images_own_python3(tmp_path):
    img_bin, alias_dir = _fake_image(tmp_path)
    (cmd,) = python_setup_commands("pytest", HERE_VER, bin_dir=str(alias_dir))
    r = _run(cmd, [img_bin, alias_dir])
    assert r.returncode == 0, r.stderr
    assert (alias_dir / "python").resolve() == (img_bin / "python3").resolve()  # the image's own python3, not a new one
    assert "pip" in r.stdout


def test_existing_python_is_never_replaced(tmp_path):
    img_bin, alias_dir = _fake_image(tmp_path, with_python=True)
    (cmd,) = python_setup_commands("pytest", HERE_VER, bin_dir=str(alias_dir))
    assert _run(cmd, [img_bin, alias_dir]).returncode == 0
    assert not (alias_dir / "python").exists()


@pytest.mark.parametrize("module,version,why", [("no_such_engine_xyz", HERE_VER, "not importable"),
                                                ("pytest", "2.7", "!=")])
def test_fails_fast_on_wrong_interpreter(tmp_path, module, version, why):
    img_bin, alias_dir = _fake_image(tmp_path)
    (cmd,) = python_setup_commands(module, version, bin_dir=str(alias_dir))
    r = _run(cmd, [img_bin, alias_dir])
    assert r.returncode != 0 and why in r.stderr


def test_fails_when_image_has_no_python_at_all(tmp_path):
    empty, alias_dir = tmp_path / "empty", tmp_path / "alias"
    empty.mkdir()
    alias_dir.mkdir()
    (cmd,) = python_setup_commands("vllm", bin_dir=str(alias_dir))
    r = _run(cmd, [empty, alias_dir])  # isolated PATH: no python, no python3
    assert r.returncode != 0 and not (alias_dir / "python").exists()


def test_engine_images_use_setup_commands_not_add_python(monkeypatch):
    modal = pytest.importorskip("modal")
    calls = []
    real = modal.Image.from_registry

    def spy(tag, *a, **kw):
        calls.append((tag, kw.get("setup_dockerfile_commands"), kw.get("add_python")))
        return real(tag, *a, **kw)

    monkeypatch.setattr(modal.Image, "from_registry", staticmethod(spy))
    spec = importlib.util.spec_from_file_location("app_under_test", REPO / "modal_bridge" / "app.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    by_tag = {t: (cmds, py) for t, cmds, py in calls}
    assert by_tag[mod.VLLM_IMAGE] == (python_setup_commands("vllm"), None)
    assert by_tag[mod.SGLANG_IMAGE] == (python_setup_commands("sglang"), None)
