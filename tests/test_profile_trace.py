"""E5g torch-profiler job mode: trace summary, bucket classification, runner path (mock), exclusion from analysis."""
import gzip
import json
import subprocess
import sys
import time

import pytest

from g4bench import analyze, profile_trace as pt
from g4bench.config import EngineConfig
from g4bench.matrix import expand, load_matrix
from g4bench.runner import RunOptions, check_flags, profile_plan, profile_window, run


def _trace() -> dict:
    """Two engine steps; forward (attention + GEMM via one graph launch), sample, draft; one memcpy; a gap."""
    ev = []

    def cpu(name, ts, dur, cat="user_annotation", corr=None):
        e = {"ph": "X", "cat": cat, "name": name, "pid": 10, "tid": 11, "ts": ts, "dur": dur}
        if corr is not None:
            e["args"] = {"correlation": corr}
        ev.append(e)

    def gpu(name, ts, dur, corr, cat="kernel"):
        ev.append({"ph": "X", "cat": cat, "name": name, "pid": 0, "tid": 7, "ts": ts, "dur": dur,
                   "args": {"correlation": corr, "device": 0}})

    for step, base in ((0, 0), (1, 1000)):
        cpu(f"execute_context_{1 - step}({512 * (1 - step)})_generation_4(20)", base, 900)
        cpu("gpu_model_runner: forward", base + 10, 300)
        cpu("cudaGraphLaunch", base + 20, 5, cat="cuda_runtime", corr=100 + step)
        cpu("gpu_model_runner: sample", base + 320, 50)
        cpu("cudaLaunchKernel", base + 330, 5, cat="cuda_runtime", corr=200 + step)
        cpu("gpu_model_runner: draft", base + 400, 200)
        cpu("cudaLaunchKernel", base + 410, 5, cat="cuda_runtime", corr=300 + step)
        cpu("cudaMemcpyAsync", base + 700, 5, cat="cuda_runtime", corr=400 + step)
        gpu("kernel_unified_attention_2d", base + 30, 100, 100 + step)
        gpu("nvjet_tst_128x256_64x4_1x2_h_bz_coopB_TNT", base + 130, 200, 100 + step)  # cuBLAS GEMM
        gpu("_topk_topp_sampler_kernel", base + 340, 20, 200 + step)
        gpu("cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64", base + 420, 60, 300 + step)  # draft GEMM
        gpu("Memcpy DtoH (Device -> Pinned)", base + 710, 10, 400 + step, cat="gpu_memcpy")
    gpu("triton_red_fused_rms_norm_0", 1800, 10, 999)  # unattributed (no launch in a phase)
    ev.append({"ph": "i", "name": "instant", "ts": 0})  # ignored
    return {"traceEvents": ev, "displayTimeUnit": "ms"}


@pytest.mark.parametrize("name,cat,bucket", [
    ("kernel_unified_attention_2d", "kernel", "attention"),
    ("_fwd_kernel_flash_decode_stage1", "kernel", "attention"),
    ("reshape_and_cache_flash_kernel", "kernel", "attention"),
    ("nvjet_tst_128x256_64x4_1x2_h_bz_coopB_TNT", "kernel", "gemm"),
    ("cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64", "kernel", "gemm"),
    ("sm90_xmma_gemm_bf16bf16_bf16f32_f32_tn_n_tilesize128x128x64", "kernel", "gemm"),
    ("ncclDevKernel_AllReduce_Sum_bf16_RING_LL", "kernel", "nccl"),
    ("Memcpy HtoD (Pinned -> Device)", "gpu_memcpy", "memcpy"),
    ("Memset (Device)", "gpu_memset", "memcpy"),
    ("_topk_topp_sampler_kernel", "kernel", "sampling"),
    ("rejection_greedy_sample_kernel", "kernel", "spec_decode"),
    ("eagle_prepare_inputs_padded_kernel", "kernel", "spec_decode"),
    ("triton_red_fused_rms_norm_0", "kernel", "norm_act"),
    ("void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>>", "kernel", "other"),
])
def test_bucket_classification(name, cat, bucket):
    assert pt.classify(name, cat) == bucket


def test_summary_buckets_phases_steps_idle():
    s = pt.summarize(_trace(), top_n=3)
    assert s["n_gpu_events"] == 11
    assert s["gpu_time_ms"] == pytest.approx((100 + 200 + 20 + 60 + 10) * 2 / 1e3 + 0.01)
    b = s["buckets"]
    assert b["attention"]["ms"] == pytest.approx(0.2) and b["gemm"]["ms"] == pytest.approx(0.52)
    assert b["sampling"]["ms"] == pytest.approx(0.04) and b["memcpy"]["ms"] == pytest.approx(0.02)
    assert b["norm_act"]["ms"] == pytest.approx(0.01)
    assert sum(v["share"] for v in b.values()) == pytest.approx(1.0, abs=1e-3)
    # GPU span 30..1810 us, busy = union of kernels (attention+gemm are back-to-back) = 790 us
    assert s["gpu_span_ms"] == pytest.approx(1.78) and s["gpu_busy_ms"] == pytest.approx(0.79)
    assert s["gpu_idle_fraction"] == pytest.approx(1 - 790 / 1780, abs=1e-4)
    ph = s["phases"]
    assert s["phases_available"]
    assert ph["forward"]["ms"] == pytest.approx(0.6) and ph["draft"]["ms"] == pytest.approx(0.12)
    assert ph["sample"]["ms"] == pytest.approx(0.04)
    assert ph["unattributed"]["ms"] == pytest.approx(0.03)  # memcpy launched outside phases + rms_norm
    assert s["phase_buckets"]["draft"] == {"gemm": pytest.approx(0.12)}
    st = s["steps"]
    assert st["all"]["n"] == 2 and st["decode_only"]["n"] == 1 and st["with_prefill"]["n"] == 1
    assert st["all"]["cpu_ms_mean"] == pytest.approx(0.9)
    assert st["all"]["gpu_kernel_ms_mean"] == pytest.approx(0.39)  # rms_norm (corr 999) is in no step
    top = s["top_kernels"]
    assert [k["name"][:5] for k in top] == ["nvjet", "kerne", "cutla"] and top[0]["count"] == 2
    assert top[0]["mean_us"] == pytest.approx(200)
    md = pt.markdown(s)
    assert "PROFILED" in md and "idle fraction" in md and "| forward |" in md and "nvjet_tst" in md


def test_summary_without_annotations_degrades_gracefully():
    tr = {"traceEvents": [{"ph": "X", "cat": "Kernel", "name": "gemm_x", "ts": 0, "dur": 5, "pid": 0, "tid": 1}]}
    s = pt.summarize(tr)
    assert s["buckets"] == {"gemm": {"ms": 0.005, "share": 1.0}} and s["steps"] is None
    assert not s["phases_available"] and s["gpu_idle_fraction"] == 0.0
    assert "not available" in pt.markdown(s)
    assert pt.summarize({"traceEvents": []})["gpu_idle_fraction"] is None


def test_write_summary_reads_gz_and_plain_and_picks_gpu_trace(tmp_path):
    out = tmp_path / "points" / "p1"
    tdir = out / "torch_trace"
    tdir.mkdir(parents=True)
    with gzip.open(tdir / "dp0_pp0_tp0_dcp0_ep0_rank0.1700000000000.pt.trace.json.gz", "wt") as f:
        json.dump(_trace(), f)
    # a CPU-only trace (e.g. the frontend) in plain JSON, list form
    (tdir / "async_llm.1700000000001.pt.trace.json").write_text(json.dumps(
        [{"ph": "X", "cat": "cpu_op", "name": "aten::add", "ts": 0, "dur": 3, "pid": 1, "tid": 1}]))
    assert len(pt.find_traces(tdir)) == 2
    assert pt.load_trace(tdir / "async_llm.1700000000001.pt.trace.json")["traceEvents"][0]["name"] == "aten::add"
    s = pt.write_summary(tdir, out)
    assert s["summarized"].endswith(".pt.trace.json.gz") and s["summarized"].startswith("torch_trace/")
    assert len(s["traces"]) == 2
    js = json.loads((out / "profile_summary.json").read_text())
    assert js["buckets"]["gemm"]["ms"] == pytest.approx(0.52)
    assert "nvjet" in (out / "profile_summary.md").read_text(encoding="utf-8")
    assert pt.write_summary(tmp_path / "missing", out) is None


def _points(keys):
    return expand(load_matrix("backboard-repro"), seeds=[0], only_keys=keys)


def test_profile_plan_vllm_uses_pinned_profiler_config(tmp_path):
    eng = EngineConfig.load("vllm-0.33.0-mtp4")
    pr = profile_plan({"key": "8192x4", "duration_s": 3}, eng, _points(["8192x4", "1024x1"]), tmp_path / "r")
    assert pr["point_id"] == "isl8192-osl1024-c4-s0-t0" and pr["duration_s"] == 3.0 and pr["warmup_s"] == 30.0
    flag, cfg = pr["server_argv"]
    assert flag == "--profiler-config"
    cfg = json.loads(cfg)
    # Field names of vllm/config/profiler.py::ProfilerConfig at v0.33.0
    assert set(cfg) <= {"profiler", "torch_profiler_dir", "torch_profiler_activities", "torch_profiler_with_stack",
                        "torch_profiler_use_gzip", "torch_profiler_dump_cuda_time_total", "ignore_frontend"}
    assert cfg["profiler"] == "torch" and cfg["torch_profiler_dir"].endswith("torch_trace")
    assert pr["trace_dir"] == cfg["torch_profiler_dir"]
    assert pr["server_env"] == {"VLLM_CUSTOM_SCOPES_FOR_PROFILING": "1"}
    # the flag exists in the pinned vLLM source snapshot (the runner's fail-fast check passes)
    from g4bench.config import load_pins, placeholder_values

    argv = eng.render_args(placeholder_values(load_pins(), 10240, eng.port)) + pr["server_argv"]
    check_flags(eng, argv)
    with pytest.raises(SystemExit, match="not among"):
        profile_plan({"key": "8192x64"}, eng, _points(["8192x4"]), tmp_path / "r")
    with pytest.raises(SystemExit, match="duration_s"):
        profile_plan({"key": "8192x4", "duration_s": 0}, eng, _points(["8192x4"]), tmp_path / "r")
    with pytest.raises(SystemExit, match="vLLM"):
        profile_plan({"key": "8192x4"}, EngineConfig.load("sglang-0.5.23-nextn"), _points(["8192x4"]), tmp_path)


def test_profile_window_skips_when_client_finishes_before_warmup():
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    prof = {"key": "k", "point_id": "p", "warmup_s": 30.0, "duration_s": 4.0, "stop_timeout_s": 5.0,
            "server_argv": [], "server_env": {}}
    t = time.monotonic()
    rec = profile_window(proc, "http://127.0.0.1:9", prof, deadline=t + 60)
    assert "skipped" in rec and "start" not in rec and time.monotonic() - t < 20


def test_profiled_rows_excluded_from_analysis(tmp_path):
    base = {"engine": "vllm", "engine_config": "vllm-0.33.0-mtp4", "config_hash": "h", "hardware_class": "x",
            "exact_hardware": True, "dataset": "random", "isl": 8192, "osl": 1024, "conc": 4, "valid": True,
            "output_throughput": 100.0}
    for rid, extra in (("r-plain", {}), ("r-prof", {"profiled": True})):
        d = tmp_path / rid
        d.mkdir()
        (d / "results.jsonl").write_text(json.dumps({**base, **extra}) + "\n")
    rows = analyze.load_all(tmp_path)
    assert [r["run_id"] for r in rows] == ["r-plain"]
    assert [r["run_id"] for r in analyze.load_all(tmp_path, include_mock=True)] == ["r-plain"]


def test_modal_job_cli_adds_profile_to_spec(monkeypatch, capsys):
    from g4bench import cli, modal_run

    monkeypatch.setattr(modal_run, "preflight", lambda p: {"stub": True})
    monkeypatch.setattr(modal_run, "load_dotenv", lambda: None)
    assert cli.main(["modal", "job", "--engine", "vllm-0.33.0-mtp4", "--matrix", "backboard-repro",
                     "--profile-key", "8192x4", "--profile-seconds", "3"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["plan_only"] and out["spec"]["profile"] == {"key": "8192x4", "duration_s": 3.0, "warmup_s": 30.0}
    assert out["spec"]["only"] == ["8192x4"] and out["spec"]["seeds"] == [0]


def test_job_spec_profile_reaches_runner_dry_run(tmp_path):
    from g4bench.jobs import run_job

    spec = {"engine": "mock-cpu", "matrix": "mock-ci", "seeds": [0], "max_wall_s": 600, "dry_run": True,
            "deadline_unix": time.time() + 900, "profile": {"key": "1024x4", "duration_s": 1}}
    res = run_job(spec, tmp_path, tokenizer="unused")
    man = json.loads((tmp_path / res["run_id"] / "manifest.json").read_text())
    assert [p["key"] for p in man["points"]] == ["1024x4"]
    assert man["profile"]["point_id"] == "isl1024-osl64-c4-s0-t0"
    assert man["server_argv"][-2] == "--profile-dir"


def _free_port() -> int:
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.slow
def test_mock_run_with_profile_window(tmp_path, tokenizer_dir, monkeypatch):
    import os

    import yaml

    from g4bench import runner
    from g4bench.client import THIRD_PARTY_INFERENCEX

    # Own port (never talk to some other mock left on 18000); OS path separator for the vendored client.
    from g4bench.config import resolve_config

    cfg = yaml.safe_load(resolve_config("mock-cpu", "engines").read_text())
    cfg["port"] = _free_port()
    (tmp_path / "eng").mkdir()
    eng_path = tmp_path / "eng" / "mock-cpu.yaml"
    eng_path.write_text(yaml.safe_dump(cfg))
    pp = os.pathsep.join([str(THIRD_PARTY_INFERENCEX), str(THIRD_PARTY_INFERENCEX.parents[1] / "src")])
    monkeypatch.setattr(runner, "client_env", lambda e: {**e, "PYTHONPATH": pp, "PYTHONDONTWRITEBYTECODE": "1"})
    if sys.platform == "win32":  # Server.stop uses os.killpg (POSIX); kill the venv-launcher tree instead
        def _stop(self, grace_s=30.0):
            if self.proc and self.proc.poll() is None:
                subprocess.run(["taskkill", "/T", "/F", "/PID", str(self.proc.pid)], capture_output=True)
                self.proc.wait(timeout=30)
            if self._log:
                self._log.close()
                self._log = None

        monkeypatch.setattr(runner.Server, "stop", _stop)
    d = run(RunOptions(engine=str(eng_path), matrix="mock-ci", results_root=tmp_path / "runs", tokenizer=tokenizer_dir,
                       seeds=[0], only_keys=["1024x4"], correctness=False, server_timeout_s=60,
                       client_timeout_s=300, profile={"key": "1024x4", "warmup_s": 0, "duration_s": 0.2}))
    rows = [json.loads(x) for x in (d / "results.jsonl").read_text().splitlines() if x.strip()]
    assert len(rows) == 1 and rows[0]["profiled"] is True and rows[0]["valid"], rows[0].get("problems")
    assert "PROFILED" in rows[0]["notes"]
    pdir = d / "points" / rows[0]["point_id"]
    prof = json.loads((pdir / "profile.json").read_text())
    assert prof["start"]["status"] == 200 and prof["stop"]["status"] == 200, prof
    assert prof["summary"]["buckets"] == {"gemm": {"ms": 0.04, "share": 1.0}}
    assert (pdir / "profile_summary.md").exists() and (pdir / "profile_summary.json").exists()
    assert "PROFILED RUN" in (d / "REPORT.md").read_text(encoding="utf-8")
    assert analyze.run(root=tmp_path / "runs", out=tmp_path / "summary", include_mock=True)["rows"] == 0
