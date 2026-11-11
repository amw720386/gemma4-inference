"""Image patch ``sm120_attn_tiles``: sm_120 prefill tiles for vLLM v0.33.0 Triton unified attention.

Target: ``vllm/v1/attention/ops/triton_unified_attention.py`` of the pinned image (vllm/vllm-openai v0.33.0;
byte-identical to github.com/vllm-project/vllm tag v0.33.0, sha256 c1cf2300c99f5dc1...6215da).

Why: v0.33.0 applies its tuned prefill launch for head_dim 256 (BLOCK_M 32, 128-token KV tile, 8 warps, 2 stages)
only on ``is_device_capability_family(100)`` (B200). On sm_120 (RTX PRO 6000 Blackwell, ~100 KB shared memory per
SM) Gemma 4's 40 sliding layers (head_dim 256, 16 q / 8 KV heads) get BLOCK_M 16 / 32-token tiles / Triton default
warps, and the 8 global layers (head_dim 512, 16 q heads per KV head) get BLOCK_M 16, i.e. ONE query token per
program, so a prefill chunk re-reads the whole KV once per token.

Scope: prefill only. The new launch applies only when ``max_seqlen_q > MAX_3D_QUERY_LEN`` (8), a condition that
already forces the 2D kernel; ``use_3d``, ``seq_threshold_3D``, the decode tile (TILE_SIZE_DECODE) and every batch
with only decode / spec-decode rows (q_len <= 8) are unchanged. The kernel source is not touched: accumulation stays
fp32 (``acc``/``L``/``M`` fp32); only the launch shape changes, so results differ by FP reduction order only.

Patch 1, after source lines 977-981 (block selection):

    BEFORE (v0.33.0 lines 977-981)              AFTER (5 lines kept, then appended)
    if tuned_large_head:                        if tuned_large_head:
        BLOCK_M = 32                                BLOCK_M = 32
        BLOCK_Q = BLOCK_M // num_queries_per_kv     BLOCK_Q = BLOCK_M // num_queries_per_kv
        launch_num_warps = 8                        launch_num_warps = 8
        launch_num_stages = 2                       launch_num_stages = 2
                                                # g4bench:sm120_attn_tiles ... (see PATCH1_ADD below)
                                                _g4_sm120_tile = None
                                                if (max_seqlen_q > MAX_3D_QUERY_LEN and not tuned_large_head
                                                        and current_platform.is_device_capability_family(120)):
                                                    if head_size == 256 and num_queries_per_kv <= 16:
                                                        BLOCK_M = 32; BLOCK_Q = BLOCK_M // num_queries_per_kv
                                                        launch_num_warps = 8; launch_num_stages = 2
                                                        _g4_sm120_tile = 32
                                                    elif head_size == 512 and num_queries_per_kv == 16:
                                                        BLOCK_M = 32; BLOCK_Q = BLOCK_M // num_queries_per_kv
                                                        launch_num_warps = 8; launch_num_stages = 1
                                                        _g4_sm120_tile = 16

Patch 2, after source lines 1013-1014 (KV tile; runs before the USE_TD clamp at 1020-1022, which still applies):

    BEFORE (v0.33.0 lines 1013-1014)            AFTER
    if tuned_large_head:                        if tuned_large_head:
        TILE_SIZE_PREFILL = 128                     TILE_SIZE_PREFILL = 128
                                                if _g4_sm120_tile is not None:  # g4bench:sm120_attn_tiles
                                                    TILE_SIZE_PREFILL = _g4_sm120_tile

Shared-memory budget (sm_120: 99 KiB opt-in per block, ~100 KiB per SM). Conservative model, bf16 (2 B):
Q tile BLOCK_M x D once + (K + V) tiles 2 x TILE x D per pipeline stage + S (BLOCK_M x TILE fp32) and P
(BLOCK_M x TILE bf16) in case the MMA layout conversion goes through shared memory. The fp32 accumulator
BLOCK_M x D lives in registers (shown separately, per program, spread over 32 x warps threads).

    Gemma 4 layer            launch                       Q      K+V/stage x stages  S+P   total    acc (regs)
    sliding D=256 DEFAULT    BM 16, TILE 32, 4 warps, 3*  8 KiB  32 KiB x 3 = 96   3     107 KiB    16 KiB
    sliding D=256 PATCHED    BM 32, TILE 32, 8 warps, 2   16     32 KiB x 2 = 64   6     86 KiB     32 KiB (32/thr)
    global  D=512 DEFAULT    BM 16, TILE 32, 4 warps, 3*  16     64 KiB x 3 = 192  3     211 KiB    32 KiB
    global  D=512 PATCHED    BM 32, TILE 16, 8 warps, 1   32     32 KiB x 1 = 32   3     67 KiB     64 KiB (64/thr)

    * v0.33.0 passes no num_warps/num_stages on sm_120, so Triton's CUDA defaults apply (4 warps, 3 stages);
      TILE_SIZE_PREFILL is _get_tile_size(..., is_prefill=True) = 32 for both layer types.

The DEFAULT rows exceed 99 KiB in this model yet run today (E5 runs), so Triton's pipeliner evidently does not
multi-buffer every K/V tile; the model over-counts and the PATCHED rows are upper bounds. Head_dim 256 keeps the
default 32-token KV tile with 2 stages (rather than TILE 64 with 1 stage, also 16 + 64 = 80 KiB + S/P) because it
never needs more K/V staging than the default already allocates (2 vs 3 stages of the same tiles) and keeps
copy/compute overlap; only the Q tile grows (8 -> 16 KiB). Head_dim 512 doubles query rows per program (1 -> 2
query tokens) and halves the KV tile so one stage fits with ~32 KiB margin; TILE 16 is the decode tile vLLM already
uses for bf16, so tl.dot sees K = 16, its bf16 minimum.

Unverified (CPU-only authoring): the actual Triton allocation on sm_120 (only measurable by compiling there). If it
is still too large, the first prefill raises ``triton.runtime.errors.OutOfResources`` (loud, not silent).
"""

from __future__ import annotations

NAME = "sm120_attn_tiles"
MODULE = "vllm"
RELPATH = "v1/attention/ops/triton_unified_attention.py"  # relative to the vllm package directory
MARKER = "# g4bench:sm120_attn_tiles"

PATCH1_ANCHOR = """\
    if tuned_large_head:
        BLOCK_M = 32
        BLOCK_Q = BLOCK_M // num_queries_per_kv
        launch_num_warps = 8
        launch_num_stages = 2
"""
PATCH1_ADD = f"""\
    {MARKER} (g4inference-optimization-lab image patch): sm_120 prefill-only
    # launch shape. max_seqlen_q > MAX_3D_QUERY_LEN already forces the 2D kernel, so the
    # 3D split-K / decode path and its thresholds are unchanged. fp32 accumulation kept.
    _g4_sm120_tile = None
    if (
        max_seqlen_q > MAX_3D_QUERY_LEN
        and not tuned_large_head
        and current_platform.is_device_capability_family(120)
    ):
        if head_size == 256 and num_queries_per_kv <= 16:
            # 2x query rows, 8 warps, 2 stages of the default 32-token KV tile:
            # <= 86 KiB smem upper bound (default: 3 stages of the same tile).
            BLOCK_M = 32
            BLOCK_Q = BLOCK_M // num_queries_per_kv
            launch_num_warps = 8
            launch_num_stages = 2
            _g4_sm120_tile = 32
        elif head_size == 512 and num_queries_per_kv == 16:
            # 2 query tokens per program instead of 1; 16-token KV tile, 1 stage:
            # Q 32 KiB + K/V 32 KiB (+ S/P 3 KiB) <= 67 KiB smem upper bound.
            BLOCK_M = 32
            BLOCK_Q = BLOCK_M // num_queries_per_kv
            launch_num_warps = 8
            launch_num_stages = 1
            _g4_sm120_tile = 16
"""
PATCH2_ANCHOR = """\
    if tuned_large_head:
        TILE_SIZE_PREFILL = 128
"""
PATCH2_ADD = f"""\
    if _g4_sm120_tile is not None:  {MARKER}
        TILE_SIZE_PREFILL = _g4_sm120_tile
"""

# (anchor, text inserted right after it). Anchors must occur exactly once in the unpatched file.
EDITS = ((PATCH1_ANCHOR, PATCH1_ADD), (PATCH2_ANCHOR, PATCH2_ADD))


class PatchError(RuntimeError):
    pass


def patch_text(src: str) -> tuple[str, bool]:
    """Return (patched source, changed). Idempotent; raises PatchError on any anchor mismatch."""
    patched_forms = [a + add for a, add in EDITS]
    if MARKER in src:
        # Already patched: require the exact patched form of every edit, once each.
        for form in patched_forms:
            if src.count(form) != 1:
                raise PatchError(f"{NAME}: file carries the marker but a patched block is missing or duplicated")
        return src, False
    out = src
    for anchor, add in EDITS:
        n = out.count(anchor)
        if n != 1:
            raise PatchError(f"{NAME}: anchor found {n} times (expected exactly 1):\n{anchor}")
        out = out.replace(anchor, anchor + add, 1)
    return out, True
