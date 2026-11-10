"""Analytic memory-bandwidth roofline for Gemma 4 12B decode.

Purpose: sanity-check reported numbers and pick hardware *before* spending money.
This is a first-order model, not a prediction: it ignores kernel efficiency,
launch overhead, sampling, the draft model's own cost beyond weight reads, and
CPU/scheduler overhead. Every hardware constant is labelled with its source and
confidence; treat "spec-sheet" bandwidths as upper bounds (achievable is typically
75-90% of peak on GDDR7/HBM).

Architecture facts are taken from the pinned config.json (docs/sources/):
48 layers; 40 sliding (window 1024, 8 KV heads x 256) + 8 global (1 KV head x 512,
attention_k_eq_v=True => K and V share one tensor). hidden 3840, vocab 262144 (tied).
"""

from __future__ import annotations

from dataclasses import dataclass

BF16 = 2

# --- model (from configs; verified) ---
WEIGHT_BYTES_TARGET = 23_919_549_408      # model.safetensors size (includes vision/audio projections)
N_LAYERS = 48
N_SLIDING = 40
N_GLOBAL = 8
SLIDING_WINDOW = 1024
KV_HEADS_SLIDING, HEAD_DIM_SLIDING = 8, 256
KV_HEADS_GLOBAL, HEAD_DIM_GLOBAL = 1, 512
GLOBAL_K_EQ_V = True
PARAMS_TEXT = 11.95e9                      # model card "Total Parameters 11.95B"

# --- draft (gemma-4-12B-it-assistant config.json) ---
DRAFT_HIDDEN = 1024
DRAFT_LAYERS = 4
DRAFT_INTERMEDIATE = 8192
DRAFT_VOCAB = 262144
DRAFT_NUM_CENTROIDS = 2048
DRAFT_CENTROID_TOPK = 32


@dataclass(frozen=True)
class GPU:
    name: str
    mem_gb: float
    bw_tbs: float            # peak DRAM bandwidth, TB/s
    bf16_dense_tflops: float  # dense BF16 tensor TFLOPS (no sparsity)
    usd_per_hr_modal: float | None
    source: str


# Bandwidth / TFLOPS: vendor spec sheets (peak). Prices: modal.com/pricing retrieved 2026-11-10
# (per-second price x 3600). Confidence: bandwidth high; TFLOPS medium (vendors quote sparse
# numbers; dense = sparse/2); RTX PRO 6000 edition (Server vs Workstation vs Max-Q) changes
# power limit and sustained clocks, not bandwidth.
GPUS: dict[str, GPU] = {
    "RTX-PRO-6000": GPU("RTX PRO 6000 Blackwell 96GB", 96, 1.792, 500, 0.000842 * 3600,
                        "NVIDIA datasheet: 1792 GB/s GDDR7; ~1 PFLOPS FP16 sparse => ~0.5 dense (medium conf.)"),
    "L40S": GPU("L40S 48GB", 48, 0.864, 362, 0.000542 * 3600, "NVIDIA datasheet: 864 GB/s; 733 sparse BF16"),
    "A100-80GB": GPU("A100 80GB", 80, 2.039, 312, 0.000694 * 3600, "NVIDIA datasheet SXM: 2039 GB/s"),
    "H100": GPU("H100 SXM 80GB", 80, 3.35, 989, 0.001097 * 3600, "NVIDIA datasheet: 3.35 TB/s"),
    "H200": GPU("H200 SXM 141GB", 141, 4.8, 989, 0.001261 * 3600, "NVIDIA datasheet: 4.8 TB/s"),
    "B200": GPU("B200 180GB", 180, 8.0, 2250, 0.001736 * 3600, "NVIDIA datasheet: 8 TB/s (medium conf. on dense TFLOPS)"),
}


def kv_bytes_per_seq(context_len: int) -> int:
    """KV cache bytes for one sequence at ``context_len`` tokens (BF16, SWA-aware)."""
    sliding = N_SLIDING * min(context_len, SLIDING_WINDOW) * KV_HEADS_SLIDING * HEAD_DIM_SLIDING * 2 * BF16
    kv_mult = 1 if GLOBAL_K_EQ_V else 2
    glob = N_GLOBAL * context_len * KV_HEADS_GLOBAL * HEAD_DIM_GLOBAL * kv_mult * BF16
    return sliding + glob


def kv_bytes_per_seq_no_swa(context_len: int) -> int:
    """Upper bound if an engine allocates full-length KV for sliding layers too."""
    sliding = N_SLIDING * context_len * KV_HEADS_SLIDING * HEAD_DIM_SLIDING * 2 * BF16
    kv_mult = 1 if GLOBAL_K_EQ_V else 2
    return sliding + N_GLOBAL * context_len * KV_HEADS_GLOBAL * HEAD_DIM_GLOBAL * kv_mult * BF16


def draft_weight_bytes() -> int:
    """Approximate draft weights read per draft step (dense layers + centroid head, not full vocab)."""
    # attn: q (16*256) + o; KV shared with target (num_kv_shared_layers=4) => no k/v proj assumed.
    attn = DRAFT_LAYERS * (DRAFT_HIDDEN * 16 * 256 * 2)
    mlp = DRAFT_LAYERS * (3 * DRAFT_HIDDEN * DRAFT_INTERMEDIATE)
    centroid = DRAFT_HIDDEN * DRAFT_NUM_CENTROIDS
    sparse_vocab = DRAFT_CENTROID_TOPK * (DRAFT_VOCAB // DRAFT_NUM_CENTROIDS) * DRAFT_HIDDEN
    proj = 2 * 3840 * DRAFT_HIDDEN  # backbone<->draft projections (approx.)
    return (attn + mlp + centroid + sparse_vocab + proj) * BF16


def decode_step_bytes(batch: int, context_len: int, swa_aware: bool = True) -> float:
    kv = kv_bytes_per_seq(context_len) if swa_aware else kv_bytes_per_seq_no_swa(context_len)
    return WEIGHT_BYTES_TARGET + batch * kv


def max_tokens_per_s(gpu: GPU, batch: int, context_len: int, *, acceptance_len: float = 1.0,
                     draft_tokens: int = 0, bw_eff: float = 0.85, swa_aware: bool = True) -> float:
    """Upper bound on aggregate decode tokens/s.

    One verification step reads target weights + KV once and yields ``acceptance_len``
    tokens per sequence; each of ``draft_tokens`` draft steps reads draft weights + KV of
    the shared global/sliding layers it attends to (approximated as 4/48 of target KV).
    """
    bw = gpu.bw_tbs * 1e12 * bw_eff
    t_verify = decode_step_bytes(batch, context_len, swa_aware) / bw
    kv = kv_bytes_per_seq(context_len) if swa_aware else kv_bytes_per_seq_no_swa(context_len)
    t_draft = draft_tokens * (draft_weight_bytes() + batch * kv * DRAFT_LAYERS / N_LAYERS) / bw
    return batch * acceptance_len / (t_verify + t_draft)


def implied_acceptance_len(gpu: GPU, tok_s_per_user: float, batch: int, context_len: int,
                           draft_tokens: int, bw_eff: float = 0.85) -> float:
    """Acceptance length needed to reach an observed per-user decode rate under the roofline.

    If this exceeds draft_tokens+1, the observation is impossible under the model's
    assumptions => some assumption (bandwidth, bytes read) is wrong.
    """
    base = max_tokens_per_s(gpu, batch, context_len, acceptance_len=1.0, draft_tokens=draft_tokens,
                            bw_eff=bw_eff) / batch
    return tok_s_per_user / base


def prefill_seconds(gpu: GPU, tokens: int, mfu: float = 0.5) -> float:
    """Compute-bound prefill estimate: 2 * params * tokens FLOPs (+attention ignored)."""
    return 2 * PARAMS_TEXT * tokens / (gpu.bf16_dense_tflops * 1e12 * mfu)


def report() -> str:
    lines = ["# Roofline sanity report (analytic, NOT measured)", ""]
    lines.append(f"Target weights read/step: {WEIGHT_BYTES_TARGET/1e9:.2f} GB; draft ~{draft_weight_bytes()/1e9:.3f} GB/step")
    for ctx in (1024 + 1024, 8192 + 1024):
        lines.append(f"KV/seq at ctx {ctx}: SWA-aware {kv_bytes_per_seq(ctx)/1e6:.0f} MB; "
                     f"no-SWA {kv_bytes_per_seq_no_swa(ctx)/1e6:.0f} MB")
    lines.append("")
    lines.append("| GPU | $/hr | no-spec bs1 tok/s bound | 64x9K KV (GB) fits? | AL needed for article vLLM 8K/c1 speed-per-user 185.2 tok/s (4 drafts, 85% BW eff.) |")
    lines.append("|---|---:|---:|---|---:|")
    for k, g in GPUS.items():
        nospec = max_tokens_per_s(g, 1, 2048)
        kv64 = 64 * kv_bytes_per_seq(9216) / 1e9
        fits = (WEIGHT_BYTES_TARGET / 1e9 + kv64) < g.mem_gb * 0.9
        al = implied_acceptance_len(g, 185.2, 1, 9216, 4)
        lines.append(f"| {k} | {g.usd_per_hr_modal:.2f} | {nospec:.0f} | {kv64:.1f} {'yes' if fits else 'NO'} | {al:.2f} |")
    lines.append("")
    lines.append("Interpretation: on RTX PRO 6000 the no-spec single-stream ceiling is ~60-65 tok/s, so the "
                 "reported ~185-204 tok/s/user requires mean acceptance length ~3+ tokens/step from MTP. "
                 "Throughput is therefore dominated by draft acceptance, which depends on the prompt "
                 "content (random tokens + chat template) - the key reproduction risk.")
    return "\n".join(lines)


if __name__ == "__main__":
    print(report())
