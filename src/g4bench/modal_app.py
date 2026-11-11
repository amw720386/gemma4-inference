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
# The one patch set the patched vLLM function is built with (an engine config's `image_patches` must equal it).
VLLM_PATCHES: tuple[str, ...] = ("mtp_draft_prefill_last",)  # sm120_attn_tiles retired: no gain (D-046)


def _engine_image(ref: str, engine_module: str, patches: tuple[str, ...] = ()) -> modal.Image:
    # Only aliases `python` to the image's own python3 when missing (vLLM image) and checks it sees the engine (D-038).
    img = modal.Image.from_registry(ref, setup_dockerfile_commands=python_setup_commands(engine_module)).entrypoint([])
    if patches:
        # Build-time text patches of the installed engine (src/g4bench/patches). copy=True: a non-copied local dir is
        # only mounted at container start, so a build step could not see it; the repo mount stays the last layer.
        img = (img.add_local_dir(str(REPO_LOCAL / "src" / "g4bench" / "patches"), PATCHES_REMOTE, copy=True,
                                 ignore=["**/__pycache__", "*.pyc"])
               .run_commands(f"python {PATCHES_REMOTE}/apply.py {' '.join(patches)}"))
    env = {**ENV, "G4_IMAGE_PATCHES": ",".join(patches)} if patches else ENV
    return img.env(env).add_local_dir(str(REPO_LOCAL), REPO_REMOTE, ignore=IGNORE)


vllm_image = _engine_image(VLLM_IMAGE, "vllm")
vllm_patched_image = _engine_image(VLLM_IMAGE, "vllm", VLLM_PATCHES)
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
sglang_app = modal.App(f"{APP_NAME}-sglang", include_source=False)


def _commit_loop(stop) -> None:
    while not stop.wait(60):
        try:
            res_vol.commit()
        except Exception:  # noqa: BLE001
            pass


def _check_image_patches(spec: dict) -> None:
    """Refuse a job whose engine config's image_patches differ from what this image was built with."""
    from g4bench.config import EngineConfig

    want = ",".join(EngineConfig.load(spec["engine"]).raw.get("image_patches") or [])
    have = os.environ.get("G4_IMAGE_PATCHES", "")
    if want != have:
        raise RuntimeError(f"engine config wants image_patches [{want}] but this image has [{have}]")


def _run_job(spec: dict) -> dict:
    import threading

    from g4bench.jobs import collect_run_files, run_job

    _check_image_patches(spec)
    hf_vol.reload()
    stop = threading.Event()
    threading.Thread(target=_commit_loop, args=(stop,), daemon=True).start()
    try:
        out = run_job(spec, pathlib.Path(RESULTS_MOUNT) / "runs")
        out["files"] = collect_run_files(pathlib.Path(RESULTS_MOUNT) / "runs", out["run_id"])
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
