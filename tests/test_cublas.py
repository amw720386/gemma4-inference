"""Per-config cuBLAS selection (side install via LD_PRELOAD) and the /proc evidence of what was loaded (D-061)."""
import pytest

from g4bench import cublas as C
from g4bench.images import CUBLAS_SIDE, VLLM_PATCH_PIP, side_env


def _side_tree(tmp_path, layout=("nvidia", "cu13", "lib")):
    root = tmp_path / "opt" / "cublas-13.10"
    lib = root.joinpath(*layout)
    lib.mkdir(parents=True)
    for n in C.LIBS:
        (lib / n).write_bytes(b"\x7fELF")
    return root, lib


def test_no_request_is_image_default():
    r = C.resolve(None, {})
    assert r == {"requested": None, "mode": "image-default", "env": {}}


def test_side_install_preloads_both_libs_first(tmp_path):
    root, lib = _side_tree(tmp_path)
    env = {C.ENV_SIDE: f"13.10={root}", "LD_PRELOAD": "/x/libfoo.so", "LD_LIBRARY_PATH": "/usr/lib"}
    r = C.resolve("13.10", env)
    assert r["mode"] == "side" and r["dir"] == str(root)
    pre = r["env"]["LD_PRELOAD"].split(":")
    assert pre == [str(lib / "libcublasLt.so.13"), str(lib / "libcublas.so.13"), "/x/libfoo.so"]
    assert r["env"]["LD_LIBRARY_PATH"] == f"{lib}:/usr/lib"
    root2, lib2 = _side_tree(tmp_path / "b", layout=("nvidia", "cublas", "lib"))  # older wheel layout
    assert C.resolve("13.10", {C.ENV_SIDE: f"13.10={root2}"})["libs"]["libcublas.so.13"] == str(lib2 / "libcublas.so.13")


def test_in_image_upgrade_and_refusals(tmp_path):
    prod = {"G4_IMAGE_PIP": ",".join(VLLM_PATCH_PIP)}
    r = C.resolve("13.10", prod)
    assert r["mode"] == "in-image" and r["env"] == {} and r["pip"] == "13.10.0.20"
    with pytest.raises(C.CublasError):
        C.resolve("13.1", prod)  # 13.1 is not 13.10.x
    with pytest.raises(C.CublasError):
        C.resolve("13.10", {})
    with pytest.raises(C.CublasError, match="not found"):  # declared but empty side dir
        C.resolve("13.10", {C.ENV_SIDE: f"13.10={tmp_path / 'empty'}"})
    assert C.resolve("13.10", {}, strict=False)["mode"] == "unresolved"


def test_image_side_env_round_trips():
    assert C.side_dirs({C.ENV_SIDE: side_env()}) == {v: d for v, (_, d) in CUBLAS_SIDE.items()}
    assert C.side_dirs({C.ENV_SIDE: "13.10=/a, 13.11=/b,junk"}) == {"13.10": "/a", "13.11": "/b"}


def _fake_proc(tmp_path, procs):
    """procs: pid -> (ppid, comm, [mapped paths])."""
    proc = tmp_path / "proc"
    for pid, (ppid, comm, maps) in procs.items():
        d = proc / str(pid)
        d.mkdir(parents=True)
        (d / "stat").write_text(f"{pid} ({comm}) S {ppid} 1 1 0 -1\n")
        (d / "maps").write_text("".join(
            f"7f00000{i}000-7f00000{i}fff r-xp 00000000 08:01 {100 + i}                      {p}\n"
            for i, p in enumerate(maps)) + "7ffd0000-7ffd1000 rw-p 00000000 00:00 0                          [stack]\n")
    (proc / "self").mkdir()
    return proc


def test_probe_walks_the_server_tree_and_verifies(tmp_path):
    side = "/opt/cublas-13.10/nvidia/cu13/lib"
    img = "/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib"
    proc = _fake_proc(tmp_path, {
        100: (1, "vllm", ["/usr/lib/libc.so.6"]),
        101: (100, "VLLM::EngineCore) x (", [f"{side}/libcublasLt.so.13", f"{side}/libcublas.so.13",
                                            f"{img}/libcublas.so.13"]),
        102: (101, "pt_main", []),
        200: (1, "other", [f"{img}/libcublas.so.13"]),  # not in the server tree
    })
    assert C.process_tree(100, proc) == [100, 101, 102]
    assert C.process_tree(999, proc) == []
    probe = C.mapped_libs(100, proc=proc)
    assert probe["available"] and probe["by_pid"] == {"101": sorted([f"{side}/libcublasLt.so.13",
                                                                     f"{side}/libcublas.so.13",
                                                                     f"{img}/libcublas.so.13"])}
    env = {C.ENV_SIDE: "13.10=/opt/cublas-13.10"}
    side_req = {"mode": "side", "dir": "/opt/cublas-13.10"}
    v = C.verify(side_req, probe, env)
    assert v["ok"] is True and v["also_mapped"] == [f"{img}/libcublas.so.13"]
    assert C.verify({"mode": "image-default"}, probe, env)["ok"] is False  # side lib leaked into a stock run
    stock = C.mapped_libs(200, proc=proc)
    assert C.verify({"mode": "image-default"}, stock, env)["ok"] is True
    assert C.verify(side_req, stock, env)["ok"] is False
    assert C.verify(side_req, {"available": False, "paths": []}, env)["ok"] is None
    assert C.mapped_libs(1, proc=tmp_path / "noproc")["available"] is False
