"""Image patch ``splitk_per_group``: per-attention-group split-K ("3D") thresholds for vLLM v0.33.0 Triton attention.

Target: ``vllm/v1/attention/backends/triton_attn.py`` of the pinned image (vllm/vllm-openai v0.33.0; byte-identical to
github.com/vllm-project/vllm tag v0.33.0, 885 lines, sha256 53b221f99c54b6bc...cb223).

Why: ``unified_attention`` (ops/triton_unified_attention.py 1070-1091) runs the split-K kernel only when
``num_seqs * cdiv(max_seqlen_q, BLOCK_Q) <= seq_threshold_3D`` (and q_len <= 8). The builder sets that threshold once
from MODEL-level shapes (triton_attn.py 136-149: ``128 // get_num_kv_heads()`` = 128 // 8 = 16 for Gemma 4, snapped to
the nearest CUDA-graph capture size), so it is the same for every attention group. Gemma 4 has two very different
groups: full/global (head_dim 512, 1 KV head, 16 q heads -> BLOCK_Q 1) and sliding (head_dim 256, 8 KV heads ->
BLOCK_Q 8). Measured per call on the RTX PRO 6000, q_len 7 (results/kernels/20261111T172659Z-attn_bench-c6c4db),
3D vs 2D us: global 8K c1/c4/c16/c64 47/579, 140/579, 387/577, 1527/1728; 1K 15/76, 21/76, 69/78, 408/235. Sliding 8K
17/57, 39/59, 141/94, 577/368; 1K 14/57, 29/59, 106/95, 484/368. Global wants 3D up to 16 x 7 = 112 q-blocks, sliding
only up to ~8 requests. One shared value cannot express both.

What: builders are per attention group (vllm/v1/worker/gpu/attn_utils.py ``init_attn_backend`` 197-265 keys one
builder per (backend, per-layer kv_cache_spec, num_heads_q)), and the builder receives that group's spec. A
sliding-window group's spec carries ``sliding_window`` (``SlidingWindowSpec``, or ``FullAttentionSpec.sliding_window``
when the hybrid allocator is off); a full-attention group's is ``None``. If ``G4_SPLITK_T_SLIDING`` /
``G4_SPLITK_T_FULL`` is set (int >= 0) it replaces ``seq_threshold_3D`` for that kind of group, compared exactly as
upstream (q-blocks: num_seqs * cdiv(q_len, BLOCK_Q)), WITHOUT snapping. Unset: upstream value, byte-for-byte
behaviour. The kernel file is not touched; same math, the 2D/3D choice only changes FP reduction order.

CUDA graphs: the 2D/3D branch is host Python inside ``unified_attention``, evaluated while capturing each graph with
that graph's padded shapes (num_seqs = padded num_reqs, max_seqlen_q = the uniform decode length) and frozen into it;
replay never re-evaluates it. Both kernels compute the same attention for any batch, so ANY threshold yields a
correct graph; the threshold only decides which correct kernel each captured size bakes in. Upstream snaps to "cover
the correct execution path", i.e. so that the cut-over falls on a captured size: a perf heuristic, not a correctness
requirement. Unsnapped, a real batch of n requests simply gets the choice made for its padded size P(n) (3D iff
P(n) * cdiv(q, BLOCK_Q) <= T); eager batches decide on their own n. The only hard constraint is scratch capacity at
the padded size, which ``unified_attention`` itself enforces (1089-1091: 3D -> 2D if scratch rows < q tokens), and
which edit 3 sizes for. With vLLM's default capture list the cut-overs land exactly on captured sizes anyway
(uniform-decode graphs at 8 and 16 requests exist: 56 and 112 tokens).

Scratch (softmax segments, 136-210): rows = max query tokens of any 3D launch. Upstream sizes
``min(threshold, max_num_seqs) * max_query_len_3d`` (i.e. assumes BLOCK_Q >= q_len), which at threshold 112 would be
784 rows x 16 q heads x 16 segments x 512 (model-max head_dim, ModelArchitectureConfig.from_layers takes the max)
x fp32 = 392 MiB per full-attention builder (target and draft each have one). With an override the patch uses the
exact bound for the group's BLOCK_Q: max over q <= max_query_len_3d of min(T // cdiv(q, BLOCK_Q), max_num_seqs) * q
(full T=112, BLOCK_Q 1: 112 rows = 56 MiB; sliding T=8, BLOCK_Q 8, q 7: 56 rows, same as upstream). The sm_120
64-segment rule that follows (151-162, 181-190) is unchanged and sees the overridden threshold. head_dim is already
sized for the larger (512) group by upstream.

Edit 1, before ``class TritonAttentionMetadataBuilder(`` (line 96): three module-level helpers (HELPERS).
Edit 2, after the capture-size snapping (lines 146-149):

    self.seq_threshold_3D = min(                    (kept)
        capture_sizes,
        key=lambda x: abs(x - self.seq_threshold_3D),
    )
                                                    self._g4_splitk_threshold = _g4_splitk_threshold_override(spec)
                                                    if self._g4_splitk_threshold is not None:
                                                        logger.info(...); self.seq_threshold_3D = <override>

Edit 3, after ``max_num_tokens_3d = max_num_seqs_3d * max_query_len_3d`` (line 180): with an override, replace it by
the exact bound above.

Unverified (CPU-only authoring): GPU execution and the end-to-end gain. Draft (MTP) decode/prefill launches are q_len
1 and use the same per-group thresholds (global: 3D up to 112 requests; sliding: up to 8); 3D vs 2D at q_len 1 above
16 requests was not measured.
"""

from __future__ import annotations

NAME = "splitk_per_group"
MODULE = "vllm"
RELPATH = "v1/attention/backends/triton_attn.py"  # relative to the vllm package directory
MARKER = "# g4bench:splitk_per_group"
ENV_FULL = "G4_SPLITK_T_FULL"
ENV_SLIDING = "G4_SPLITK_T_SLIDING"

HELPERS_ANCHOR = """\
class TritonAttentionMetadataBuilder(AttentionMetadataBuilder[TritonAttentionMetadata]):
"""
HELPERS = f"""\
{MARKER} (g4inference-optimization-lab image patch): per-attention-group split-K
# ("3D") thresholds. {ENV_FULL} / {ENV_SLIDING} (int >= 0) replace seq_threshold_3D for
# full-attention / sliding-window groups, compared as upstream against
# num_seqs * cdiv(max_seqlen_q, BLOCK_Q), without capture-size snapping. Unset: upstream.
def _g4_splitk_threshold_override(kv_cache_spec) -> int | None:
    import os

    sliding = getattr(kv_cache_spec, "sliding_window", None) is not None
    name = "{ENV_SLIDING}" if sliding else "{ENV_FULL}"
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    value = int(raw)
    if value < 0:
        raise ValueError(f"{{name}} must be >= 0, got {{raw!r}}")
    return value


def _g4_splitk_block_q(num_heads_q: int, num_kv_heads: int) -> int | None:
    # BLOCK_Q as unified_attention() derives it (2D-only tuned paths aside); None if unknown.
    if num_heads_q <= 0 or num_kv_heads <= 0 or num_heads_q % num_kv_heads:
        return None
    num_queries_per_kv = num_heads_q // num_kv_heads
    block_m = 16 if num_queries_per_kv <= 16 else next_power_of_2(num_queries_per_kv)
    return max(1, block_m // num_queries_per_kv)


def _g4_splitk_scratch_tokens(
    threshold: int, block_q: int | None, max_query_len_3d: int, max_num_seqs: int
) -> int:
    # Most query tokens a 3D launch can carry: num_seqs * cdiv(q, BLOCK_Q) <= threshold,
    # num_seqs <= max_num_seqs, q <= max_query_len_3d. Unknown BLOCK_Q: upstream bound.
    if block_q is None:
        block_q = max_query_len_3d
    return max(
        min(threshold // cdiv(q, block_q), max_num_seqs) * q
        for q in range(1, max_query_len_3d + 1)
    )


"""

THRESHOLD_ANCHOR = """\
            self.seq_threshold_3D = min(
                capture_sizes,
                key=lambda x: abs(x - self.seq_threshold_3D),
            )
"""
THRESHOLD_ADD = f"""\

        {MARKER}: per-group override, not snapped (graph-safe:
        # the 2D/3D choice is frozen per captured size and both kernels are exact).
        self._g4_splitk_threshold = _g4_splitk_threshold_override(kv_cache_spec)
        if self._g4_splitk_threshold is not None:
            logger.info(
                "g4bench splitk_per_group: %s group (%d layers) seq_threshold_3D %d -> %d",
                "sliding"
                if getattr(kv_cache_spec, "sliding_window", None) is not None
                else "full",
                len(layer_names),
                self.seq_threshold_3D,
                self._g4_splitk_threshold,
            )
            self.seq_threshold_3D = self._g4_splitk_threshold
"""

SCRATCH_ANCHOR = """\
        max_num_tokens_3d = max_num_seqs_3d * max_query_len_3d
"""
SCRATCH_ADD = f"""\
        if self._g4_splitk_threshold is not None:  {MARKER}
            # Exact bound for this group's BLOCK_Q; unified_attention() still runs 2D
            # if a launch ever has more query tokens than scratch rows.
            max_num_tokens_3d = _g4_splitk_scratch_tokens(
                self.seq_threshold_3D,
                _g4_splitk_block_q(self.num_heads_q, kv_cache_spec.num_kv_heads),
                max_query_len_3d,
                vllm_config.scheduler_config.max_num_seqs,
            )
"""

# (anchor, replacement). Anchors must occur exactly once in the unpatched file; every replacement keeps its anchor.
EDITS = (
    (HELPERS_ANCHOR, HELPERS + HELPERS_ANCHOR),
    (THRESHOLD_ANCHOR, THRESHOLD_ANCHOR + THRESHOLD_ADD),
    (SCRATCH_ANCHOR, SCRATCH_ANCHOR + SCRATCH_ADD),
)


class PatchError(RuntimeError):
    pass


def patch_text(src: str) -> tuple[str, bool]:
    """Return (patched source, changed). Idempotent; raises PatchError on any anchor mismatch."""
    if MARKER in src:
        # Already patched: require the exact patched form of every edit, once each.
        for _, new in EDITS:
            if src.count(new) != 1:
                raise PatchError(f"{NAME}: file carries the marker but a patched block is missing or duplicated")
        return src, False
    out = src
    for anchor, new in EDITS:
        n = out.count(anchor)
        if n != 1:
            raise PatchError(f"{NAME}: anchor found {n} times (expected exactly 1):\n{anchor}")
        out = out.replace(anchor, new, 1)
    return out, True
