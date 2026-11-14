"""Modal app for paid GPU runs, launched directly from this machine (``g4bench modal …``; D-039).

There is no deployed service: every launch runs inside an *ephemeral* ``app.run()`` owned by the local process, so
nothing stays up between jobs, and if the local process dies Modal stops the app. Spend control is local
(``spend.py``): each call gets a Modal ``timeout`` derived from its reservation, ``retries=0`` and
``max_containers=1``, and is priced at its cpu/memory *limits*.

Engine images are digest-pinned (configs/pins.yaml) and receive NO extra pip packages, so the engine's dependency
set is exactly the published image. Weights, the HF tokenizer and vLLM/Triton compile caches live on the
``g4repro-hf-cache`` Volume; run directories are written to ``g4repro-results`` and also returned to the caller.
"""

from __future__ import annotations

import os
import pathlib
import time

import modal

from g4bench import spend as S
from g4bench.image_setup import python_setup_commands
from g4bench.images import (  # noqa: F401 - re-exported: tests and older callers read them from here
    ALWAYS_ON_PATCHES, CUBLAS_SIDE, VLLM_PATCH_PIP, VLLM_PATCHES, VLLM_PATCHES_EXP, side_env,
)

APP_NAME = "g4repro"
REPO_LOCAL = pathlib.Path(__file__).resolve().parents[2]
REPO_REMOTE = "/repo"
HF_HOME = "/cache/hf"
CACHE_ROOT = "/cache"
RESULTS_MOUNT = "/results"

# Pinned by digest (configs/pins.yaml). linux/amd64 manifests.
VLLM_IMAGE = "vllm/vllm-openai@sha256:91ee972ab751fb7180bc71c175865e36b091686600dca783d52b2511399810b3"
SGLANG_IMAGE = "lmsysorg/sglang@sha256:f7876bef696ccf9cf75a2bf7ccb82970f5fac577f082fb8ca95c9a59d2160eb5"

# Never upload secrets, venvs, local results or bulky data into an image.
IGNORE = [".git", ".venv", ".env", "*.secret", "**/__pycache__", "*.pyc", ".pytest_cache", ".cache", "results",
          "data/speedbench/*.jsonl"]
ENV = {"PYTHONPATH": f"{REPO_REMOTE}/src", "HF_HOME": HF_HOME, "HF_HUB_ENABLE_HF_TRANSFER": "0",
       "TOKENIZERS_PARALLELISM": "false", "G4_DATA_DIR": f"{HF_HOME}/g4data/speedbench",
       # Compile caches persist across jobs (startup time only; no effect on measured serving).
       "VLLM_CACHE_ROOT": f"{CACHE_ROOT}/vllm", "TRITON_CACHE_DIR": f"{CACHE_ROOT}/triton",
       "G4_CODE_SHA": os.environ.get("G4_CODE_SHA", "unknown")}

GPU_DEFAULT_TIMEOUT_S = S.timeout_for_budget(S.GPU_JOB.rate(), 2.0)
SCALEDOWN_WINDOW_S = 10  # <= spend.SCALEDOWN_MARGIN_S


PATCHES_REMOTE = "/opt/g4patches"
# Patch sets / library choices per image live in images.py (Modal-free; routing and hashing use them too).
# Configs opt in to a patch by listing it in `image_patches` (a subset of the image's) AND setting its env gate; since
# D-061 every patch is env-gated (ALWAYS_ON_PATCHES is empty). sm120_attn_tiles retired: no gain (D-046).


def _engine_image(ref: str, engine_module: str, patches: tuple[str, ...] = (), pip: tuple[str, ...] | None = None,
                  side: dict[str, tuple[str, str]] | None = None, extra: tuple[str, ...] = ()) -> modal.Image:
    """``pip``: upgraded IN the image (default VLLM_PATCH_PIP for a patched image); ``side``: version ->
    (requirement, dir) installed with ``pip --target`` next to the image's own libs, selected per config (D-061)."""
    # Only aliases `python` to the image's own python3 when missing (vLLM image) and checks it sees the engine (D-038).
    img = modal.Image.from_registry(ref, setup_dockerfile_commands=python_setup_commands(engine_module)).entrypoint([])
    pip = (VLLM_PATCH_PIP if patches else ()) if pip is None else tuple(pip)
    side = dict(side or {})
    if patches:
        # Build-time text patches of the installed engine (src/g4bench/patches). copy=True: a non-copied local dir is
        # only mounted at container start, so a build step could not see it; the repo mount stays the last layer.
        img = (img.add_local_dir(str(REPO_LOCAL / "src" / "g4bench" / "patches"), PATCHES_REMOTE, copy=True,
                                 ignore=["**/__pycache__", "*.pyc"])
               .run_commands(f"python {PATCHES_REMOTE}/apply.py {' '.join(patches)}"))
        if pip:
            img = img.run_commands(f"python -m pip install --no-deps --upgrade {' '.join(pip)}")
        for _, (req, target) in sorted(side.items()):
            img = img.run_commands(f"python -m pip install --no-deps --target {target} {req}")
        for req in extra:  # optional packages the engine only uses when a config's env enables them (D-068)
            mod = req.split("==")[0].replace("-", "_")
            img = img.run_commands(f"python -m pip install --no-deps {req}", f"python -c 'import {mod}'")
    env = dict(ENV)
    if patches:
        env.update({"G4_IMAGE_PATCHES": ",".join(patches), "G4_IMAGE_PIP": ",".join(pip)})
        if side:
            env["G4_CUBLAS_SIDE"] = side_env(side)
        if extra:
            env["G4_IMAGE_EXTRA"] = ",".join(extra)
    return img.env(env).add_local_dir(str(REPO_LOCAL), REPO_REMOTE, ignore=IGNORE)


vllm_image = _engine_image(VLLM_IMAGE, "vllm")
# Production patched image: build steps unchanged (patches + in-image cuBLAS 13.10), so the E8a series (D-060) stays
# reproducible; only env gates decide what a config gets.
vllm_patched_image = _engine_image(VLLM_IMAGE, "vllm", VLLM_PATCHES)
# Second patched image: VLLM_PATCHES + experimental patches still under evaluation, on its own app so the first
# image (and any run series on it) stays byte-identical while they are tested (D-058). Since D-061 its cuBLAS is the
# stock image's (13.1); 13.10 is side-installed and chosen per config (G4_CUBLAS), so the article's stock recipe runs
# here unchanged and can be paired with a candidate on the same GPU.
# fastokens: vLLM 0.33's optional Rust tokenizer (VLLM_USE_FASTOKENS=1; same ids as the HF tokenizer, 13.5 -> 1.1 ms per
# 7.4K-token prompt, results/kernels/*frontend_ttft_probe*); unused unless a config sets the env var (D-068).
vllm_patched_exp_image = _engine_image(VLLM_IMAGE, "vllm", VLLM_PATCHES_EXP, pip=(), side=CUBLAS_SIDE,
                                       extra=("fastokens==0.3.6",))
sglang_image = _engine_image(SGLANG_IMAGE, "sglang")
cpu_image = (modal.Image.debian_slim(python_version="3.12")
             .pip_install("pyyaml==6.0.3", "numpy==2.5.4", "requests==2.34.2", "huggingface_hub==2.4.0",
                          "transformers==5.22.2", "tqdm==4.70.1", "jinja2==3.1.6", "pyarrow==26.0.1", "aiohttp==3.14.5",
                          "pytest==9.2.0")
             .env(ENV).add_local_dir(str(REPO_LOCAL), REPO_REMOTE, ignore=IGNORE))

hf_vol = modal.Volume.from_name("g4repro-hf-cache", create_if_missing=True)
res_vol = modal.Volume.from_name("g4repro-results", create_if_missing=True)

# One app per image, so a launch only builds/pulls the image it uses (a CPU selftest never pulls a 10+ GB engine image).
# include_source=False: the code comes only from /repo (add_local_dir), so paths.REPO resolves to /repo.
cpu_app = modal.App(f"{APP_NAME}-cpu", include_source=False)
vllm_app = modal.App(f"{APP_NAME}-vllm", include_source=False)
vllm_patched_app = modal.App(f"{APP_NAME}-vllm-patched", include_source=False)
vllm_patched_exp_app = modal.App(f"{APP_NAME}-vllm-patched-exp", include_source=False)
sglang_app = modal.App(f"{APP_NAME}-sglang", include_source=False)


def _commit_loop(stop) -> None:
    while not stop.wait(60):
        try:
            res_vol.commit()
        except Exception:  # noqa: BLE001
            pass


def check_engine_on_image(engine: str, environ=None) -> None:
    """Refuse an engine config this image cannot run as declared.

    * every patch it lists must be in the image, and every always-on patch of the image must be listed (none since
      D-061: patches are env-gated, so a config that does not opt in gets upstream code);
    * a config listing no patches (a stock recipe) is refused on an image with in-place library upgrades
      (G4_IMAGE_PIP, the production patched image): it would silently not be stock;
    * its G4_CUBLAS request must be satisfiable here (cublas.resolve raises otherwise)."""
    from g4bench import cublas
    from g4bench.config import EngineConfig

    environ = os.environ if environ is None else environ
    cfg = EngineConfig.load(engine)
    want = set(cfg.raw.get("image_patches") or [])
    have = {p for p in environ.get("G4_IMAGE_PATCHES", "").split(",") if p}
    if not want <= have or (have and not ALWAYS_ON_PATCHES & have <= want):
        raise RuntimeError(f"{engine}: engine config wants image_patches {sorted(want)} but this image has "
                           f"{sorted(have)} (always-on: {sorted(ALWAYS_ON_PATCHES & have)})")
    if have and not want and environ.get("G4_IMAGE_PIP"):
        raise RuntimeError(f"{engine}: a config without image_patches would not be stock on this image (in-place "
                           f"upgrades {environ.get('G4_IMAGE_PIP')}); use the experimental image")
    cublas.resolve(cfg.env.get(cublas.ENV_REQUEST), environ)


def _check_image_patches(spec: dict) -> None:
    """Refuse a job whose engine config(s) this image cannot run as declared (check_engine_on_image)."""
    for engine in ((spec.get("paired") or {}).get("engines") or [spec["engine"]]):
        check_engine_on_image(engine)


def _run_job(spec: dict) -> dict:
    import threading

    from g4bench.jobs import collect_run_files, run_job

    _check_image_patches(spec)
    hf_vol.reload()
    stop = threading.Event()
    threading.Thread(target=_commit_loop, args=(stop,), daemon=True).start()
    runs = pathlib.Path(RESULTS_MOUNT) / "runs"
    try:
        out = run_job(spec, runs)
        if spec.get("paired"):  # one run dir per engine + the pair record (jobs.run_job -> paired.run_paired_job)
            out["files_multi"] = {
                "runs": {rid: collect_run_files(runs, rid) for rid in (out.get("run_ids") or {}).values() if rid},
                "pairs": {out["pair_id"]: collect_run_files(runs.parent / "pairs", out["pair_id"])},
            }
        else:
            out["files"] = collect_run_files(runs, out["run_id"])
        return out
    finally:
        stop.set()
        res_vol.commit()
        try:
            hf_vol.commit()  # compile caches
        except Exception:  # noqa: BLE001
            pass


def _guarded(fn, spec: dict) -> dict:
    from g4bench.jobs import guarded

    return guarded(fn, spec, hard_kill=True)  # os._exit at the reservation's deadline (D-030)


_gpu_kw = dict(gpu=S.GPU_JOB.gpu, timeout=GPU_DEFAULT_TIMEOUT_S, startup_timeout=int(S.STARTUP_MARGIN_S),
               volumes={CACHE_ROOT: hf_vol, RESULTS_MOUNT: res_vol}, max_containers=1, retries=0,
               scaledown_window=SCALEDOWN_WINDOW_S,
               cpu=(4.0, S.GPU_JOB.cpu_limit), memory=(32768, int(S.GPU_JOB.mem_limit_gib * 1024)))


@vllm_app.function(image=vllm_image, **_gpu_kw)
def run_vllm(spec: dict) -> dict:
    return _guarded(_run_job, spec)


def _run_script(spec: dict) -> dict:
    """Kernel lab: run one repo script on the GPU inside the pinned vLLM image; it writes files to $G4_OUT_DIR."""
    import subprocess

    from g4bench.jobs import collect_run_files

    hf_vol.reload()
    root = pathlib.Path(RESULTS_MOUNT) / "scripts"
    run_id = f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{pathlib.Path(spec['script']).stem}-{spec['job_id'][-6:]}"
    out = root / run_id
    out.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "G4_OUT_DIR": str(out), "PYTHONUNBUFFERED": "1"}
    try:
        p = subprocess.run(["python", f"{REPO_REMOTE}/{spec['script']}", *spec.get("args", [])], capture_output=True,
                           text=True, cwd=REPO_REMOTE, env=env, timeout=max(float(spec.get("max_wall_s", 600)), 30))
        rc, so, se = p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired as e:
        rc, so, se = 124, (e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or ""), "timeout"
    (out / "stdout.txt").write_text(so, encoding="utf-8")
    (out / "stderr.txt").write_text(se, encoding="utf-8")
    res_vol.commit()
    return {"run_id": run_id, "rc": rc, "stdout_tail": so[-6000:], "stderr_tail": se[-3000:],
            "files": collect_run_files(root, run_id)}


@vllm_app.function(image=vllm_image, **_gpu_kw)
def run_script_vllm(spec: dict) -> dict:
    return _guarded(_run_script, spec)


@vllm_patched_exp_app.function(image=vllm_patched_exp_image, **_gpu_kw)
def run_vllm_patched_exp(spec: dict) -> dict:
    """vLLM on the experimental patched image (VLLM_PATCHES_EXP)."""
    return _guarded(_run_job, spec)


@vllm_patched_app.function(image=vllm_patched_image, **_gpu_kw)
def run_vllm_patched(spec: dict) -> dict:
    """vLLM on the image with VLLM_PATCHES applied; only for engine configs whose image_patches equal them."""
    return _guarded(_run_job, spec)


@sglang_app.function(image=sglang_image, **_gpu_kw)
def run_sglang(spec: dict) -> dict:
    return _guarded(_run_job, spec)


def _prefetch(spec: dict) -> dict:
    from huggingface_hub import snapshot_download

    from g4bench import speedbench
    from g4bench.config import load_pins

    pins = load_pins()
    out: dict = {}
    for role in ("target", "draft"):
        m = pins["model"][role]
        t0 = time.time()
        path = snapshot_download(m["repo"], revision=m["revision"])
        out[role] = {"repo": m["repo"], "revision": m["revision"], "path": path, "seconds": round(time.time() - t0, 1)}
    if spec.get("speedbench"):
        data, cache = pathlib.Path(os.environ["G4_DATA_DIR"]), pathlib.Path(HF_HOME) / "g4data" / "cache"
        out["speedbench"] = {
            "tp1k": speedbench.prepare_throughput_inline("throughput_1k", data, cache)["prompt_digest"],
            "tp8k": speedbench.prepare_throughput_inline("throughput_8k", data, cache)["prompt_digest"],
            "coding": speedbench.prepare_coding(data, cache)["prompt_digest"],  # raises unless InferenceX digest
        }
    hf_vol.commit()
    # lstat: HF snapshot entries are symlinks into blobs/xet; following them would double-count the weights.
    total = sum(os.lstat(os.path.join(d, f)).st_size for d, _, fs in os.walk(CACHE_ROOT) for f in fs)
    out["volume_gib"] = round(total / 2**30, 2)
    return out


@cpu_app.function(image=cpu_image, timeout=S.PREFETCH_TIMEOUT_S, startup_timeout=600, volumes={CACHE_ROOT: hf_vol},
              retries=0, max_containers=1, scaledown_window=SCALEDOWN_WINDOW_S,
              cpu=(S.PREFETCH.cpu_limit, S.PREFETCH.cpu_limit),
              memory=(int(S.PREFETCH.mem_limit_gib * 1024), int(S.PREFETCH.mem_limit_gib * 1024)))
def prefetch_weights(spec: dict) -> dict:
    """CPU-only: pinned target + draft snapshots (and optionally SPEED-Bench sets) into the cache Volume."""
    return _guarded(_prefetch, spec)


def _selftest(spec: dict) -> dict:
    import subprocess

    p = subprocess.run(["python", "-m", "pytest", "-q", "-p", "no:cacheprovider", f"{REPO_REMOTE}/tests"],
                       capture_output=True, text=True, cwd=REPO_REMOTE)
    return {"rc": p.returncode, "tail": (p.stdout + p.stderr)[-4000:]}


@cpu_app.function(image=cpu_image, timeout=S.SELFTEST_TIMEOUT_S, startup_timeout=600, retries=0, max_containers=1,
              scaledown_window=SCALEDOWN_WINDOW_S, cpu=(S.PREFETCH.cpu_limit, S.PREFETCH.cpu_limit),
              memory=(int(S.PREFETCH.mem_limit_gib * 1024), int(S.PREFETCH.mem_limit_gib * 1024)))
def selftest(spec: dict) -> dict:
    """CPU-only: the repo's test-suite on Linux inside Modal (validates image, code and the launch path)."""
    return _guarded(_selftest, spec)
