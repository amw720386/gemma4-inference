"""Build-time engine image patches (src/g4bench/patches) and their Modal wiring. CPU only."""
import ast
import importlib.util
import os
import pathlib
import subprocess
import sys
import warnings

import pytest

from g4bench.config import EngineConfig
from g4bench.paths import REPO
from g4bench.patches import apply as A
from g4bench.patches import sm120_attn_tiles as P

PATCH_DIR = REPO / "src" / "g4bench" / "patches"
TILES = "vllm-0.33.0-mtp6-splitk-chunk3072-tiles"
ACTIVE = "vllm-0.33.0-mtp6-splitk-chunk3072-draftlast"  # config matching the currently built patched image

# Runnable reduction of vLLM v0.33.0 vllm/v1/attention/ops/triton_unified_attention.py lines 956-1022: the
# block/tile selection lines (and both patch anchors) are verbatim; tensors/platform are stubbed. Set
# G4_VLLM_TUA=<path to the pinned upstream file> to also run the anchor tests against the real 1230-line file.
EXCERPT = '''\
MAX_3D_QUERY_LEN = 8


def unified_attention(q, max_seqlen_q, current_platform, triton, use_td=False, block_size=16):
    num_query_heads = q.shape[1]
    num_kv_heads = q.shape[3]
    num_queries_per_kv = num_query_heads // num_kv_heads
    head_size = q.shape[2]

    BLOCK_M = (
        16 if num_queries_per_kv <= 16 else triton.next_power_of_2(num_queries_per_kv)
    )
    BLOCK_Q = BLOCK_M // num_queries_per_kv

    # Tuned launch parameters; ``None`` lets Triton pick its defaults.
    launch_num_warps: int | None = None
    launch_num_stages: int | None = None

    tuned_large_head = (
        head_size == 256
        and max_seqlen_q > 1
        and num_queries_per_kv <= 16
        and current_platform.is_device_capability_family(100)
    )
    if tuned_large_head:
        BLOCK_M = 32
        BLOCK_Q = BLOCK_M // num_queries_per_kv
        launch_num_warps = 8
        launch_num_stages = 2

    TILE_SIZE_PREFILL = 32
    TILE_SIZE_DECODE = 16

    # Wider KV tile for the tuned large-head path (see above). Only the 2D
    # path (used when max_seqlen_q > 1) reads TILE_SIZE_PREFILL.
    if tuned_large_head:
        TILE_SIZE_PREFILL = 128

    if use_td:
        TILE_SIZE_PREFILL = min(TILE_SIZE_PREFILL, block_size)
        TILE_SIZE_DECODE = min(TILE_SIZE_DECODE, block_size)
    return dict(BLOCK_M=BLOCK_M, BLOCK_Q=BLOCK_Q, warps=launch_num_warps, stages=launch_num_stages,
                tile_prefill=TILE_SIZE_PREFILL, tile_decode=TILE_SIZE_DECODE, tuned=tuned_large_head)
'''


class _Platform:
    def __init__(self, cap: int):
        self.cap = cap

    def is_device_capability_family(self, capability: int) -> bool:
        return self.cap // 10 == capability // 10


class _Triton:
    @staticmethod
    def next_power_of_2(n: int) -> int:
        return 1 << (n - 1).bit_length()


class _Q:
    def __init__(self, q_heads, head, kv_heads):
        self.shape = (0, q_heads, head, kv_heads)


def _launch(src: str, cap: int, head: int, kv_heads: int, q_len: int, **kw) -> dict:
    ns: dict = {}
    exec(compile(src, "tua", "exec"), ns)
    return ns["unified_attention"](_Q(16, head, kv_heads), q_len, _Platform(cap), _Triton, **kw)


def _upstream() -> str:
    p = os.environ.get("G4_VLLM_TUA")
    return pathlib.Path(p).read_text(encoding="utf-8") if p else EXCERPT


def test_applies_exactly_once_and_is_idempotent():
    src = _upstream()
    out, changed = P.patch_text(src)
    assert changed and out != src
    for anchor, add in P.EDITS:
        assert out.count(anchor + add) == 1
    ast.parse(out)
    again, changed2 = P.patch_text(out)
    assert again == out and not changed2


def test_refuses_missing_or_duplicated_anchor():
    with pytest.raises(P.PatchError, match="0 times"):
        P.patch_text(EXCERPT.replace("        TILE_SIZE_PREFILL = 128\n", "        TILE_SIZE_PREFILL = 256\n"))
    with pytest.raises(P.PatchError, match="2 times"):
        P.patch_text(EXCERPT + "\n" + P.PATCH1_ANCHOR)
    out, _ = P.patch_text(EXCERPT)
    with pytest.raises(P.PatchError, match="marker"):  # marker present but a patched block was edited away
        P.patch_text(out.replace("TILE_SIZE_PREFILL = _g4_sm120_tile", "TILE_SIZE_PREFILL = 64"))


def test_launch_shapes_change_only_for_sm120_prefill():
    out, _ = P.patch_text(EXCERPT)
    # sm_120 prefill: sliding (head 256, 8 KV heads) and global (head 512, 1 KV head) layers.
    assert _launch(out, 120, 256, 8, 3072) == dict(BLOCK_M=32, BLOCK_Q=16, warps=8, stages=2, tile_prefill=32,
                                                   tile_decode=16, tuned=False)
    assert _launch(out, 120, 512, 1, 3072) == dict(BLOCK_M=32, BLOCK_Q=2, warps=8, stages=1, tile_prefill=16,
                                                   tile_decode=16, tuned=False)
    assert _launch(out, 121, 512, 1, 100)["BLOCK_Q"] == 2  # 12.x family
    # USE_TD clamp still applies after the patch.
    assert _launch(out, 120, 256, 8, 3072, use_td=True)["tile_prefill"] == 16
    assert _launch(out, 120, 512, 1, 3072, use_td=True, block_size=8)["tile_prefill"] == 8
    # Unchanged: spec-decode / decode (q_len <= 8, may use 3D), other GPUs, other head shapes.
    for args in [(120, 256, 8, 7), (120, 512, 1, 8), (120, 512, 1, 1), (100, 256, 8, 3072), (100, 512, 1, 3072),
                 (90, 256, 8, 3072), (120, 128, 8, 3072), (120, 512, 2, 3072)]:
        assert _launch(out, *args) == _launch(EXCERPT, *args), args


def _smem_kib(L: dict, head: int) -> float:
    """Conservative model from the patch docstring: Q once + (K+V) per stage + S (fp32) and P (bf16); bf16 tiles.
    Unset warps/stages mean Triton's CUDA defaults (4 warps, 3 stages)."""
    stages = L["stages"] or 3
    q = L["BLOCK_M"] * head * 2
    kv = 2 * L["tile_prefill"] * head * 2 * stages
    sp = L["BLOCK_M"] * L["tile_prefill"] * (4 + 2)
    return (q + kv + sp) / 1024


def test_sm120_prefill_tiles_fit_shared_memory_with_margin():
    out, _ = P.patch_text(EXCERPT)
    budget_kib, margin_kib = 99, 12  # sm_120 opt-in max per block
    got = {h: _smem_kib(_launch(out, 120, h, kv, 3072), h) for h, kv in ((256, 8), (512, 1))}
    default = {h: _smem_kib(_launch(EXCERPT, 120, h, kv, 3072), h) for h, kv in ((256, 8), (512, 1))}
    assert got == {256: 86.0, 512: 67.0} and default == {256: 107.0, 512: 211.0}  # docstring table
    assert all(v <= budget_kib - margin_kib for v in got.values())
    # Larger launch than default where it matters: more query rows per program, and 8 warps.
    for h, kv in ((256, 8), (512, 1)):
        new, old = _launch(out, 120, h, kv, 3072), _launch(EXCERPT, 120, h, kv, 3072)
        assert new["BLOCK_Q"] == 2 * old["BLOCK_Q"] and new["warps"] == 8 and old["warps"] is None
        assert (new["stages"], old["stages"]) == ({256: 2, 512: 1}[h], None)


def test_apply_script_patches_a_package_tree(tmp_path):
    pkg = tmp_path / "vllm"
    f = pkg / P.RELPATH
    f.parent.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    f.write_text(EXCERPT, encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": str(tmp_path)}  # find_spec('vllm') -> the fake package
    cmd = [sys.executable, str(PATCH_DIR / "apply.py"), P.NAME]
    r = subprocess.run(cmd, cwd=tmp_path, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "patched" in r.stdout and P.MARKER in f.read_text(encoding="utf-8")
    r2 = subprocess.run(cmd, cwd=tmp_path, env=env, capture_output=True, text=True)
    assert r2.returncode == 0 and "already applied" in r2.stdout
    f.write_text("nothing to anchor\n", encoding="utf-8")
    r3 = subprocess.run(cmd, cwd=tmp_path, env=env, capture_output=True, text=True)
    assert r3.returncode != 0 and "FAILED" in r3.stderr
    r4 = subprocess.run([sys.executable, str(PATCH_DIR / "apply.py"), "no_such_patch"], capture_output=True, text=True)
    assert r4.returncode != 0


def test_apply_is_stdlib_only():
    tree = ast.parse((PATCH_DIR / "apply.py").read_text(encoding="utf-8"))
    mods = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | {
        a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert mods <= {"__future__", "importlib.util", "pathlib", "py_compile", "sys"}
    assert P.NAME in A.KNOWN


def test_config_loads_with_image_patches():
    e = EngineConfig.load(TILES)
    base = EngineConfig.load("vllm-0.33.0-mtp6-splitk-chunk3072")
    assert e.raw["image_patches"] == [P.NAME] and e.args == base.args and e.env == base.env
    assert e.raw["role"] == "variant" and e.raw["admissibility"] == "a"
    assert "image_patches" not in base.raw


def _modal_app():
    pytest.importorskip("modal")
    spec = importlib.util.spec_from_file_location("app_under_test_patches", REPO / "src" / "g4bench" / "modal_app.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_modal_app_defines_patched_function_and_keeps_unpatched():
    M = _modal_app()
    assert M.vllm_patched_app.name == "g4repro-vllm-patched"
    assert M.vllm_app.name == "g4repro-vllm" and M.sglang_app.name == "g4repro-sglang"
    assert M.VLLM_PATCHES and all(p in A.KNOWN for p in M.VLLM_PATCHES)
    assert M.vllm_patched_image is not M.vllm_image
    with warnings.catch_warnings():  # registered_functions is deprecated (removed in modal 1.7; pinned 1.6.3)
        warnings.simplefilter("ignore")
        fns = {a.name: set(a.registered_functions) for a in (M.vllm_patched_app, M.vllm_app, M.sglang_app,
                                                               M.vllm_patched_exp_app)}
    assert fns == {"g4repro-vllm-patched-exp": {"run_vllm_patched_exp"}, "g4repro-vllm-patched": {"run_vllm_patched"}, "g4repro-vllm": {"run_vllm", "run_script_vllm"},
                   "g4repro-sglang": {"run_sglang"}}


def test_patched_image_build_steps(monkeypatch):
    modal = pytest.importorskip("modal")
    calls = []
    real_dir, real_run, real_env = modal.Image.add_local_dir, modal.Image.run_commands, modal.Image.env

    def spy_dir(self, local, remote, **kw):
        calls.append(("dir", str(local), remote, kw.get("copy", False)))
        return real_dir(self, local, remote, **kw)

    def spy_run(self, *cmds, **kw):
        calls.append(("run", cmds))
        return real_run(self, *cmds, **kw)

    def spy_env(self, env):
        calls.append(("env", env.get("G4_IMAGE_PATCHES")))
        return real_env(self, env)

    monkeypatch.setattr(modal.Image, "add_local_dir", spy_dir)
    monkeypatch.setattr(modal.Image, "run_commands", spy_run)
    monkeypatch.setattr(modal.Image, "env", spy_env)
    M = _modal_app()
    img = M._engine_image(M.VLLM_IMAGE, "vllm", M.VLLM_PATCHES)
    assert img is not None
    tail = calls[-5:]
    assert tail[0] == ("dir", str(PATCH_DIR), M.PATCHES_REMOTE, True)  # copied: visible to the build step
    names = " ".join(M.VLLM_PATCHES)
    assert tail[1] == ("run", (f"python {M.PATCHES_REMOTE}/apply.py {names}",))
    assert tail[2] == ("run", (f"python -m pip install --no-deps --upgrade {' '.join(M.VLLM_PATCH_PIP)}",))
    assert tail[3] == ("env", ",".join(M.VLLM_PATCHES))
    assert tail[4][:3] == ("dir", str(M.REPO_LOCAL), M.REPO_REMOTE) and tail[4][3] is False  # repo mount last
    calls.clear()
    M._engine_image(M.VLLM_IMAGE, "vllm")
    assert [c[0] for c in calls] == ["env", "dir"] and calls[0][1] is None  # unpatched image: no patch steps
    # Experimental image (D-061): no in-place upgrade; cuBLAS 13.10 side-installed with pip --target.
    calls.clear()
    M._engine_image(M.VLLM_IMAGE, "vllm", M.VLLM_PATCHES_EXP, pip=(), side=M.CUBLAS_SIDE)
    runs = [c[1][0] for c in calls if c[0] == "run"]
    assert not any("--upgrade" in r for r in runs)
    req, target = M.CUBLAS_SIDE["13.10"]
    assert runs[-1] == f"python -m pip install --no-deps --target {target} {req}" and target == "/opt/cublas-13.10"
    assert ("env", ",".join(M.VLLM_PATCHES_EXP)) in calls


def test_modal_run_routes_patched_configs():
    from g4bench.modal_run import allowed_engines, gpu_kind

    pytest.importorskip("modal")
    assert allowed_engines()[ACTIVE] == "vllm"
    assert gpu_kind(ACTIVE, "vllm") == "gpu:vllm-patched"
    with pytest.raises(SystemExit):  # a config whose patches differ from the built image is refused
        gpu_kind(TILES, "vllm")
    assert gpu_kind("vllm-0.33.0-mtp6-splitk-chunk3072", "vllm") == "gpu:vllm"
    assert gpu_kind("sglang-0.5.23-nextn", "sglang") == "gpu:sglang"
    assert gpu_kind("vllm-0.33.0-best-e8-pfa-c8k", "vllm") == "gpu:vllm-patched-exp"


def test_worker_refuses_mismatched_image(monkeypatch):
    M = _modal_app()
    monkeypatch.setenv("G4_IMAGE_PATCHES", "")
    with pytest.raises(RuntimeError, match="sm120_attn_tiles"):  # patch the image lacks
        M._check_image_patches({"engine": TILES})
    M._check_image_patches({"engine": "vllm-0.33.0-mtp6-splitk-chunk3072"})  # unpatched image, unpatched config
    monkeypatch.setenv("G4_IMAGE_PATCHES", ",".join(M.VLLM_PATCHES))
    monkeypatch.setenv("G4_IMAGE_PIP", ",".join(M.VLLM_PATCH_PIP))  # production patched image
    M._check_image_patches({"engine": ACTIVE})  # subset: accepted
    with pytest.raises(RuntimeError):  # TILES is not in the current image
        M._check_image_patches({"engine": TILES})
    with pytest.raises(RuntimeError):  # unpatched config on the production image would silently get cuBLAS 13.10
        M._check_image_patches({"engine": "vllm-0.33.0-mtp6-splitk-chunk3072"})
    # Experimental image (D-061): every patch env-gated, image cuBLAS = stock -> the stock recipe runs unchanged.
    monkeypatch.setenv("G4_IMAGE_PATCHES", ",".join(M.VLLM_PATCHES_EXP))
    monkeypatch.setenv("G4_IMAGE_PIP", "")
    M._check_image_patches({"engine": "vllm-0.33.0-mtp4"})
    M._check_image_patches({"paired": {"engines": ["vllm-0.33.0-mtp4", ACTIVE]}, "engine": None})
    monkeypatch.delenv("G4_CUBLAS_SIDE", raising=False)
    with pytest.raises(Exception, match="G4_CUBLAS"):  # pfa asks for cuBLAS 13.10, not side-installed here
        M._check_image_patches({"paired": {"engines": ["vllm-0.33.0-mtp4", "vllm-0.33.0-best-e8-pfa-c8k"]}})
