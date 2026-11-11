"""Dockerfile setup lines that make the pinned engine images usable by ``modal.Image.from_registry``.

Modal requires a registry image (without ``add_python``) to have ``python`` and ``pip`` on PATH. The pinned
vLLM v0.33.0 image installs vLLM into the system Python 3.12 (deadsnakes; ``python3`` via update-alternatives, pip via
get-pip) but ships no bare ``python``, so ``modal deploy`` fails with "unable to determine the version of Python
installed in the Image". SGLang v0.5.23 already exposes ``python`` (its venv /opt/sglang/bin is first on PATH).

Fix (D-038): if ``python`` is missing, alias it to the image's own ``python3`` - the interpreter the engine is
installed into - and never touch an existing ``python``. ``add_python`` is deliberately NOT used: it would put a
different interpreter first that cannot import the engine. The same build step then fails fast unless that
``python`` is the expected version, can locate the engine package, and has ``pip``.
"""

from __future__ import annotations


def python_setup_commands(engine_module: str, python_version: str = "3.12", bin_dir: str = "/usr/local/bin") -> list[str]:
    script = (
        "set -e; "
        f'if ! command -v python >/dev/null 2>&1; then ln -s "$(command -v python3)" {bin_dir}/python; fi; '
        "python -c \"import importlib.util, sys; "
        f"v = '%d.%d' % sys.version_info[:2]; assert v == '{python_version}', 'python ' + v + ' != {python_version}'; "
        f"assert importlib.util.find_spec('{engine_module}'), '{engine_module} not importable by python'\"; "
        "python -m pip --version"
    )
    return [f"RUN {script}"]
