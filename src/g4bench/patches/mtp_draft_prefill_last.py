"""Image patch ``mtp_draft_prefill_last``: Gemma 4 MTP draft prefill on the last token per request only.

Target: ``vllm/v1/worker/gpu/spec_decode/target_dependent_ar/speculator.py`` of the pinned image (vLLM v0.33.0;
github.com/vllm-project/vllm tag v0.33.0, 994 lines, sha256 511b337a201d4ef8...2ce318).

Why: ``TargetDependentARSpeculator.propose`` runs the draft "prefill" (draft position 0) over EVERY token the target
just processed (lines 282-294: ``self._prefill(num_reqs, prefill_batch_desc.num_tokens, attn_metadata, ...)``,
reusing the target's metadata), then keeps one row per request (``_prefill`` lines 451-452, 475:
``last_hidden_states[last_token_indices]``). For a 3072-token prefill chunk the 4-layer draft runs 3072 rows to use 1.

Why this is lossless for Gemma 4 MTP (vllm/model_executor/models/gemma4_mtp.py and gemma4/speculator.py @ v0.33.0):
  * The draft writes no state. Every draft attention layer is KV-shared and Q-only (gemma4_mtp.py 149-252:
    ``self.attn(q, None, None)``; ``kv_sharing_target_layer_name`` set in gemma4/speculator.py 153-154), and
    ``Attention.forward`` skips the cache update when ``kv_sharing_target_layer_name`` is set
    (model_executor/layers/attention/attention.py 546-555). ``Gemma4Speculator.advance_draft_positions`` is False
    (gemma4/speculator.py 49-53) for exactly this reason.
  * Every other op is row-wise: embed(input_id) || target hidden -> pre_projection -> per-layer RMSNorm/q_proj/
    q_norm/RoPE(position)/o_proj/MLP -> norm -> post_projection (gemma4_mtp.py 404-436). The only cross-row op is
    attention, and it reads the TARGET's KV (already written for the whole chunk by the target forward), not draft
    rows. So row i's output depends only on (input_ids[i], hidden_states[i], positions[i], target KV[0..pos_i]).
  * The draft decode steps already run this exact 1-query-per-request shape for Gemma 4: same positions (not
    advanced), ``seq_len = target_seq_len - num_rejected`` (``_prepare_decode_inputs_kernel`` 848-859), metadata from
    ``_build_uniform_attn_metadata`` over the draft's own attention groups (``_multi_step_decode`` 507-526). The
    patched prefill reuses that path with step-0 inputs, so seq_len == position + 1 of the kept row: causal
    attention over the same keys (and the same sliding window) as row ``last_token_index`` of the full pass.
  * Sampling is unchanged: ``_prefill`` still samples with ``positions[last]+1`` keys and ``idx_mapping``, and still
    writes ``hidden_states[:num_reqs]`` / ``positions[:num_reqs]`` for the decode steps.
  Outputs are mathematically identical; bitwise they may differ (GEMMs / attention see M = num_reqs instead of
  num_tokens, decode-style attention kernel). That only perturbs DRAFT proposals; target verification/rejection
  sampling keeps the served output distribution exact either way.
  * Chunked prefill: a non-final chunk keeps one row per request too (its draft token for ``next_prefill_tokens``
    is discarded upstream exactly as before), so it costs one draft row instead of the whole chunk.
  * ``num_speculative_tokens == 0`` (dynamic SD): upstream runs the full pass with ``skip_sample`` "only to keep the
    drafter KV in sync" (line 472); Gemma 4 has no drafter KV, so the patched path returns without a forward.

Env gate (D-061): the patched path runs only when ``G4_DRAFT_PREFILL_LAST=1`` is set in the server environment (read
once per speculator instance); unset, the guard is False and ``propose`` takes the upstream branches unchanged, so an
image carrying this patch also serves the stock recipe. Before D-061 the patch was always on once applied; every
config that lists it sets the gate, and ``config.LEGACY_ENV`` keeps those configs' hashes unchanged.

Scope / guards (otherwise upstream code runs unchanged): only ``type(self).__name__ == "Gemma4Speculator"`` with
``advance_draft_positions`` False; not dummy/profile runs; the target-shaped prefill descriptor is not FULL (FULL
is only uniform decode batches); no PCP, DCP, DP (>1), or multimodal draft inputs; and the batch contains a request
whose query is longer than a verify window (``max_query_len > num_speculative_steps + 1``), i.e. a prefill chunk.
CUDA graphs: the compacted batch is dispatched through ``prefill_cudagraph_manager`` with no uniform token count
(PIECEWISE at the padded capture size when captured, else eager); a FULL result is downgraded to eager. FULL graph
capture/replay of ``_prefill`` is untouched (the method itself is not modified).

Edit 1, ``propose`` (v0.33.0 lines 276-277):

    BEFORE                                                  AFTER
    self.on_prefill_begin(num_reqs)                         self.on_prefill_begin(num_reqs)
    if prefill_batch_desc.cg_mode == CUDAGraphMode.FULL:    if self._g4_use_last_token_prefill(  # marker
                                                                prefill_batch_desc, max_query_len, dummy_run, is_profile
                                                            ):
                                                                self._g4_prefill_last_tokens(
                                                                    num_reqs, input_batch, num_rejected,
                                                                    num_speculative_tokens == 0)
                                                            elif prefill_batch_desc.cg_mode == CUDAGraphMode.FULL:

Edit 2, before ``def _multi_step_decode(`` (v0.33.0 line 492): two new methods (METHODS below).
``_g4_prefill_last_tokens`` gathers rows ``last_token_indices`` of input_ids/positions/hidden_states into rows
[0, num_reqs), sets query_start_loc = arange (padded with num_reqs), seq_lens = target_seq_len - num_rejected,
last_token_indices = arange, builds slot mappings + draft attention metadata like ``_multi_step_decode``, and calls
the unmodified ``_prefill`` on num_reqs rows.

Unverified (CPU-only authoring): GPU execution and the latency gain (estimated ~10-25 ms per 8K-token prompt);
acceptance-rate parity vs. the unpatched image (expected equal up to bf16 reduction-order noise).
"""

from __future__ import annotations

NAME = "mtp_draft_prefill_last"
MODULE = "vllm"
RELPATH = "v1/worker/gpu/spec_decode/target_dependent_ar/speculator.py"  # relative to the vllm package directory
MARKER = "# g4bench:mtp_draft_prefill_last"
ENV = "G4_DRAFT_PREFILL_LAST"  # "1" = patched path; unset/anything else = upstream (D-061)

PROPOSE_ANCHOR = """\
        self.on_prefill_begin(num_reqs)
        if prefill_batch_desc.cg_mode == CUDAGraphMode.FULL:
"""
PROPOSE_NEW = f"""\
        self.on_prefill_begin(num_reqs)
        if self._g4_use_last_token_prefill(  {MARKER}
            prefill_batch_desc, max_query_len, dummy_run, is_profile
        ):
            self._g4_prefill_last_tokens(
                num_reqs, input_batch, num_rejected, num_speculative_tokens == 0
            )
        elif prefill_batch_desc.cg_mode == CUDAGraphMode.FULL:
"""

METHODS_ANCHOR = """\
    def _multi_step_decode(
        self,
"""
METHODS = f"""\
    {MARKER} (g4inference-optimization-lab image patch): Gemma 4 MTP is Q-only
    # over the target's KV and writes no draft KV, so draft-prefill row i depends only on
    # row i's inputs. Run the draft prefill on the kept row (last token) of each request.
    def _g4_use_last_token_prefill(
        self, prefill_batch_desc, max_query_len, dummy_run, is_profile
    ) -> bool:
        on = getattr(self, "_g4_prefill_last_on", None)
        if on is None:  # env gate, read once per speculator: unset/other = upstream
            import os

            on = os.environ.get("{ENV}", "").strip() == "1"
            self._g4_prefill_last_on = on
        return (
            on
            and type(self).__name__ == "Gemma4Speculator"
            and not self.advance_draft_positions
            and not dummy_run
            and not is_profile
            and prefill_batch_desc.cg_mode != CUDAGraphMode.FULL
            and self.prefill_cudagraph_manager is not None
            and self.pcp_manager is None
            and not self.supports_mm_inputs
            and self.dp_size == 1
            and self.dcp_size == 1
            and int(max_query_len) > self.num_speculative_steps + 1
        )

    def _g4_prefill_last_tokens(
        self, num_reqs, input_batch, num_rejected, skip_sample
    ) -> None:
        if skip_sample:
            # No drafts needed and Gemma 4 MTP keeps no drafter KV to sync.
            return
        buffers = self.input_buffers
        idx = self.last_token_indices[:num_reqs]
        # Advanced indexing copies, so the gathers cannot alias the writes.
        buffers.input_ids[:num_reqs] = buffers.input_ids[idx]
        buffers.positions[:num_reqs] = buffers.positions[idx]
        self.hidden_states[:num_reqs] = self.hidden_states[idx]
        qsl = buffers.query_start_loc
        qsl.copy_(
            torch.arange(qsl.shape[0], dtype=qsl.dtype, device=qsl.device).clamp_(
                max=num_reqs
            )
        )
        buffers.seq_lens[:num_reqs] = (
            input_batch.seq_lens[:num_reqs] - num_rejected[:num_reqs]
        ).to(buffers.seq_lens.dtype)
        buffers.seq_lens[num_reqs:] = 0
        self.last_token_indices[:num_reqs] = torch.arange(
            num_reqs, dtype=self.last_token_indices.dtype, device=idx.device
        )
        batch_desc, _ = dispatch_cg_and_sync_dp(
            self.prefill_cudagraph_manager,
            num_reqs,
            num_reqs,
            None,
            dp_size=1,
            dp_rank=0,
        )
        if batch_desc.cg_mode == CUDAGraphMode.FULL:
            batch_desc = BatchExecutionDescriptor(
                cg_mode=CUDAGraphMode.NONE, num_tokens=num_reqs, num_reqs=num_reqs
            )
        idx_mapping = self.idx_mapping[:num_reqs]
        slot_mappings = self.block_tables.compute_slot_mappings(
            idx_mapping,
            qsl[: num_reqs + 1],
            buffers.positions[:num_reqs],
            batch_desc.num_tokens,
        )
        slot_mappings_by_layer = build_slot_mappings_by_layer(
            slot_mappings, self.kv_cache_config
        )
        attn_metadata = self._build_uniform_attn_metadata(
            num_reqs=num_reqs,
            batch_desc=batch_desc,
            num_query_per_req=1,
            seq_lens_cpu_upper_bound=input_batch.seq_lens_cpu_upper_bound,
            step=0,
        )
        self._prefill(
            num_reqs,
            batch_desc.num_tokens,
            attn_metadata,
            slot_mappings_by_layer,
            num_tokens_across_dp=None,
            cudagraph_runtime_mode=batch_desc.cg_mode,
        )

"""

# (anchor, replacement). Anchors must occur exactly once in the unpatched file; each replacement keeps its anchor
# except PROPOSE_NEW, which turns the anchor's ``if`` into ``elif`` behind the new branch.
EDITS = ((PROPOSE_ANCHOR, PROPOSE_NEW), (METHODS_ANCHOR, METHODS + METHODS_ANCHOR))


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
