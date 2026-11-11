"""Command-line entry point: ``g4bench <command>``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _csv_ints(s: str | None) -> list[int] | None:
    return [int(x) for x in s.split(",")] if s else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="g4bench")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("sources", help="model-availability test against pinned HF revisions")
    sub.add_parser("roofline", help="print analytic bandwidth roofline (not a measurement)")
    sub.add_parser("consistency", help="internal-consistency checks on the article's numbers")

    p = sub.add_parser("plan", help="expand a matrix and estimate GPU time / dollars")
    p.add_argument("--matrix", default="backboard-repro")
    p.add_argument("--engines", default="vllm-0.33.0-mtp4,sglang-0.5.23-nextn")
    p.add_argument("--gpu", default="RTX-PRO-6000")
    p.add_argument("--seeds")
    p.add_argument("--trials", type=int)
    p.add_argument("--only", help="comma list of workload keys, e.g. 1024x1,8192x4")

    r = sub.add_parser("run", help="run a matrix against one engine config (on the GPU host)")
    r.add_argument("--engine", required=True)
    r.add_argument("--matrix", required=True)
    r.add_argument("--seeds")
    r.add_argument("--trials", type=int)
    r.add_argument("--only")
    r.add_argument("--results-root", type=Path)
    r.add_argument("--max-wall-s", type=float, default=4 * 3600)
    r.add_argument("--tokenizer")
    r.add_argument("--model-name")
    r.add_argument("--notes", default="")
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--extra-server-arg", action="append", default=[],
                   help="append one raw server argv token (repeatable); changes config hash via notes only, "
                        "so ALSO set --notes")

    fcp = sub.add_parser("flagcheck", help="check engine configs' flags against pinned engine sources (static)")
    fcp.add_argument("--refresh", action="store_true", help="re-fetch sources at the pinned tags (HTTPS)")

    sub.add_parser("claims", help="render docs/CLAIMS.md (status from measured results only)")

    rp = sub.add_parser("report", help="(re)generate REPORT.md + throughput.svg for one run directory")
    rp.add_argument("run_dir", type=Path)

    sb = sub.add_parser("speedbench", help="prepare/verify SPEED-Bench real-prompt sets (CPU, network)")
    sb.add_argument("action", choices=["prepare", "verify"])
    sb.add_argument("--which", default="all", choices=["all", "coding", "tp1k", "tp8k"])
    sb.add_argument("--cache", type=Path, default=Path(".cache/speedbench"))

    a = sub.add_parser("analyze", help="aggregate all runs and compare to article targets")
    a.add_argument("--root", type=Path)
    a.add_argument("--out", type=Path)
    a.add_argument("--include-mock", action="store_true")

    from .modal_run import add_parser as _add_modal

    _add_modal(sub)

    args = ap.parse_args(argv)

    if args.cmd == "modal":
        from .modal_run import main as modal_main

        return modal_main(args)

    if args.cmd == "sources":
        from .availability import check_all

        res = check_all()
        print(json.dumps(res, indent=2))
        return 0 if all(x["ok"] for x in res) else 1

    if args.cmd == "roofline":
        from .roofline import report

        print(report())
        return 0

    if args.cmd == "consistency":
        from .inference_checks import summary

        print(json.dumps(summary(), indent=1))
        return 0

    if args.cmd == "plan":
        from .config import load_yaml, resolve_config
        from .cost import estimate
        from .matrix import expand, load_matrix

        mat = load_matrix(args.matrix)
        pts = expand(mat, seeds=_csv_ints(args.seeds), trials=args.trials,
                     only_keys=args.only.split(",") if args.only else None)
        engines = [e for e in args.engines.split(",") if e]
        per_engine = {}
        for e in engines:
            kind = load_yaml(resolve_config(e, "engines"))["engine"]
            # Throughput guess: the article's number for the same engine family (SGLang is slower).
            est = estimate(pts, args.gpu, fresh_server_per_point=bool(mat.get("fresh_server_per_point", True)),
                           tok_s_ref="sglang" if kind == "sglang" else "vllm")
            per_engine[e] = {"usd": round(est.usd, 2), "gpu_hours": round(est.gpu_seconds / 3600, 3),
                             "summary": est.summary()}
        print(json.dumps({"matrix": mat["name"], "gpu": args.gpu, "points_per_engine": len(pts),
                          "per_engine": per_engine,
                          "total_usd": round(sum(v["usd"] for v in per_engine.values()), 2),
                          "total_gpu_hours": round(sum(v["gpu_hours"] for v in per_engine.values()), 3)}, indent=2))
        return 0

    if args.cmd == "run":
        from .runner import RunOptions, run

        kw = {}
        if args.results_root:
            kw["results_root"] = args.results_root
        d = run(RunOptions(engine=args.engine, matrix=args.matrix, seeds=_csv_ints(args.seeds),
                           trials=args.trials, only_keys=args.only.split(",") if args.only else None,
                           max_wall_s=args.max_wall_s, tokenizer=args.tokenizer, model_name=args.model_name,
                           extra_server_args=args.extra_server_arg, dry_run=args.dry_run, notes=args.notes, **kw))
        print(str(d))
        return 0

    if args.cmd == "flagcheck":
        from . import flagcheck

        if args.refresh:
            for e in ("vllm", "sglang"):
                print("refreshed", flagcheck.refresh(e))
        res = flagcheck.check_all()
        print(json.dumps(res, indent=1))
        return 0 if all(not v for v in res.values()) else 1

    if args.cmd == "claims":
        from . import claims

        print(claims.write())
        return 0

    if args.cmd == "report":
        from . import report

        print(report.write(args.run_dir))
        return 0

    if args.cmd == "speedbench":
        from . import speedbench
        from .datasets import data_dir, dataset_path

        out = data_dir()
        names = {"coding": "speedbench-coding", "tp1k": "speedbench-tp1k-inline", "tp8k": "speedbench-tp8k-inline"}
        which = list(names) if args.which == "all" else [args.which]
        res: dict = {}
        for w in which:
            if args.action == "prepare":
                prov = (speedbench.prepare_coding(out, args.cache) if w == "coding" else
                        speedbench.prepare_throughput_inline({"tp1k": "throughput_1k", "tp8k": "throughput_8k"}[w],
                                                             out, args.cache))
            else:
                _, prov = speedbench.load_prepared(dataset_path(names[w]))
            res[names[w]] = {k: prov.get(k) for k in ("n_prompts", "prompt_digest", "matches_inferencex_digest")}
        print(json.dumps(res, indent=2))
        return 0

    if args.cmd == "analyze":
        from . import analyze

        kw = {}
        if args.root:
            kw["root"] = args.root
        if args.out:
            kw["out"] = args.out
        print(json.dumps(analyze.run(include_mock=args.include_mock, **kw)))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
