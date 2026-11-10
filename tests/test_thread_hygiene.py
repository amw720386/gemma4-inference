"""Regression for a bug that would have crashed every GPU job on Python 3.12 (engine images):
a threading.Thread subclass assigning self._stop shadows Thread._stop() and breaks join()."""
import ast
import threading
from pathlib import Path

from g4bench.paths import REPO


def test_thread_subclasses_do_not_shadow_thread_internals():
    reserved = {n for n in dir(threading.Thread)} | {"_stop", "_started", "_target", "_args", "_kwargs", "_name",
                                                     "_tstate_lock", "_is_stopped", "_ident", "_handle"}
    bad = []
    for f in (REPO / "src").rglob("*.py"):
        for cls in ast.walk(ast.parse(f.read_text())):
            if not isinstance(cls, ast.ClassDef) or not any("Thread" in ast.unparse(b) for b in cls.bases):
                continue
            for node in ast.walk(cls):
                if (isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store)
                        and isinstance(node.value, ast.Name) and node.value.id == "self" and node.attr in reserved):
                    bad.append(f"{f.relative_to(REPO)}:{node.lineno} self.{node.attr}")
    assert not bad, bad


def test_gpu_sampler_start_stop_without_nvidia_smi(tmp_path):
    from g4bench.server import GpuSampler

    s = GpuSampler(Path(tmp_path) / "g.csv", interval=0.01)
    s.start()
    assert s.stop()["gpu_mem_peak_mib"] is None
