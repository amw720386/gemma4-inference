"""Generate CPU-side fingerprints of the exact InferenceX request stream for each matrix point.

Usage: python scripts/make_fingerprints.py <tokenizer_dir> <out.json> [--range-ratio 0.8]
"""
import argparse
import hashlib
import json
import sys
import time

from g4bench.prompts import fingerprint, generate

ap = argparse.ArgumentParser()
ap.add_argument("tokenizer")
ap.add_argument("out")
ap.add_argument("--range-ratio", type=float, default=0.8)
a = ap.parse_args()
pts = [(isl, c, s) for isl in (1024, 8192) for c in (1, 4, 16, 64) for s in (0, 1, 2)]
res = {"range_ratio": a.range_ratio, "osl": 1024, "use_chat_template": True, "points": {}}
for isl, c, s in pts:
    t0 = time.time()
    # warmups: benchmark_serving warms up with the first request(s) of the same list (see benchmark()).
    reqs = generate(a.tokenizer, isl=isl, osl=1024, num_prompts=10 * c, seed=s, range_ratio=a.range_ratio,
                    num_workers=4)
    fp = fingerprint(reqs)
    fp["input_lens_sha256"] = hashlib.sha256(json.dumps([r[1] for r in reqs]).encode()).hexdigest()
    fp["output_lens_sha256"] = hashlib.sha256(json.dumps([r[2] for r in reqs]).encode()).hexdigest()
    fp["output_lens"] = [r[2] for r in reqs] if c <= 4 else None
    fp.pop("first_prompt_head", None)
    res["points"][f"{isl}x{c}-s{s}"] = fp
    print(f"{isl}x{c}-s{s} {fp['sha256'][:12]} in={fp['input_len_mean']:.1f} out={fp['output_len_mean']:.1f} ({time.time()-t0:.0f}s)",
          file=sys.stderr, flush=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
