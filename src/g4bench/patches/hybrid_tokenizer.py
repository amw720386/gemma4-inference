"""Image patch ``hybrid_tokenizer``: fastokens for prompt ENCODING only; every decode path stays on HF ``tokenizers``.

Target: ``vllm/tokenizers/registry.py`` of the pinned image (vllm/vllm-openai v0.33.0, github tag v0.33.0).

Why: ``VLLM_USE_FASTOKENS=1`` (registry.get_tokenizer -> vllm/tokenizers/fastokens.py -> fastokens.patch_transformers)
replaces the whole Rust backend ``TokenizersBackend._tokenizer`` with fastokens' ``_TokenizerShim`` AND rebinds
``tokenizers.decoders.DecodeStream`` to fastokens' own stream, so encode, decode, convert_ids_to_tokens and vLLM's
streaming detokenizer (v1/engine/detokenizer.py FastIncrementalDetokenizer: ``tokenizers.decoders.DecodeStream(
ids=prompt_token_ids)`` then ``stream.step(tokenizer._tokenizer, id)`` per output token) all move to fastokens. Encode
gets ~12x faster (13.5 -> 1.1 ms for a 7.4K-token prompt), but the 8K c=1 pair (results/pairs/20261113T232853Z-*) lost
4 % output throughput and p99 TPOT went 5.53 -> 6.75 ms: the decode side is suspected.

What: with ``G4_HYBRID_TOKENIZER=1`` (and VLLM_USE_FASTOKENS unset), get_tokenizer installs (process-global,
idempotent, at the same point and the same way as the fastokens patch) a wrapper of ``TokenizersBackend.from_pretrained``
that loads the tokenizer exactly as upstream (HF Rust backend) and then replaces ``tokenizer._tokenizer`` with
``HybridBackend``. The swap has to happen inside from_pretrained, before hf.get_cached_tokenizer: the CachedTokenizer /
renderer / TokenizerPool copies are rebuilt through ``__reduce__`` from that original object, so a swap on the returned
copy would be lost. HybridBackend: ``encode`` / ``encode_batch``
/ ``encode_batch_fast`` of plain strings go to a fastokens ``_TokenizerShim`` built from the same backend JSON (its
truncation / padding / encode_special_tokens are synced from the HF backend before each call, so transformers'
``set_truncation_and_padding`` keeps working); EVERY other attribute (decode, decode_batch, id_to_token, get_vocab,
to_str, setters, ...) is forwarded to the untouched HF ``tokenizers.Tokenizer``. ``tokenizers.decoders.DecodeStream``
is rebound to a thin wrapper that unwraps a HybridBackend to its HF tokenizer before ``step`` (vLLM looks the class up
on the module per request precisely so backend patches are honoured), so detokenization runs on HF's DecodeStream.
Pair / pre-tokenized inputs and mutations (add_tokens, normalizer, ...) use HF / rebuild the encoder. Deep copies
(vLLM's renderer thread pool, hf.maybe_make_thread_pool) copy both halves; pickling yields the plain HF backend
(upstream behaviour). If the backend is not a ``tokenizers.Tokenizer`` (slow tokenizer, or VLLM_USE_FASTOKENS already
swapped it) the patch logs and leaves it alone. Unset gate: nothing imported, byte-for-byte upstream.

Edit 1, after ``apply_fastokens_patch()`` (top of get_tokenizer, end of the VLLM_USE_FASTOKENS block, dedented so it
runs unconditionally): ``_g4_hybrid_tokenizer_install()``.
Edit 2, before ``cached_get_tokenizer = lru_cache(get_tokenizer)``: that helper (env check, loads this file's runtime).
"""

from __future__ import annotations

import pathlib

NAME = "hybrid_tokenizer"
MODULE = "vllm"
RELPATH = "tokenizers/registry.py"  # relative to the vllm package directory
MARKER = "# g4bench:hybrid_tokenizer"
ENV = "G4_HYBRID_TOKENIZER"  # "1" = hybrid; unset/anything else = upstream
RUNTIME_PATH = str(pathlib.Path(__file__).resolve())

# --------------------------------------------------------------------------------------------------- text patch
INSTALL_ANCHOR = """\
        apply_fastokens_patch()
"""
INSTALL_NEW = f"""\
        apply_fastokens_patch()
    _g4_hybrid_tokenizer_install()  {MARKER}
"""

HELPER_ANCHOR = """\
cached_get_tokenizer = lru_cache(get_tokenizer)
"""
HELPER_TEMPLATE = """\
{marker} (g4inference-optimization-lab image patch): with {env}=1 prompt
# encoding uses fastokens while decode / DecodeStream stay on HF tokenizers. Unset: upstream.
_G4_HYBTOK_RUNTIME = {runtime!r}


def _g4_hybrid_tokenizer_install() -> None:
    import os

    if os.environ.get("{env}", "").strip() != "1":
        return
    import importlib.util
    import sys

    name = "g4patch_runtime_{name}"
    mod = sys.modules.get(name)
    if mod is None:
        spec = importlib.util.spec_from_file_location(name, _G4_HYBTOK_RUNTIME)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    mod.install_global()


"""


def edits(runtime_path: str = RUNTIME_PATH) -> tuple[tuple[str, str], ...]:
    """(anchor, replacement) pairs; anchors occur exactly once in the unpatched file and replacements keep them."""
    helper = HELPER_TEMPLATE.format(marker=MARKER, env=ENV, runtime=runtime_path, name=NAME)
    return ((INSTALL_ANCHOR, INSTALL_NEW), (HELPER_ANCHOR, helper + HELPER_ANCHOR))


class PatchError(RuntimeError):
    pass


def patch_text(src: str, runtime_path: str = RUNTIME_PATH) -> tuple[str, bool]:
    """Return (patched source, changed). Idempotent; raises PatchError on any anchor mismatch."""
    eds = edits(runtime_path)
    if MARKER in src:
        for _, new in eds:
            if src.count(new) != 1:
                raise PatchError(f"{NAME}: file carries the marker but a patched block is missing or duplicated")
        return src, False
    out = src
    for anchor, new in eds:
        n = out.count(anchor)
        if n != 1:
            raise PatchError(f"{NAME}: anchor found {n} times (expected exactly 1):\n{anchor}")
        out = out.replace(anchor, new, 1)
    return out, True


# ------------------------------------------------------------------------------------------------------ runtime
# Loaded by the patched registry only when the gate is on. Nothing heavy is imported at module level (apply.py loads
# this file at image build time and CPU tests import it without tokenizers / fastokens).
_UNSET = object()
_ENCODE_ONLY_RESET = frozenset({"normalizer", "pre_tokenizer", "post_processor", "model"})


class HybridBackend:
    """``tokenizers.Tokenizer`` stand-in: plain-string encodes on fastokens, everything else on the HF backend."""

    __slots__ = ("_g4_hf", "_g4_fast", "_g4_trunc", "_g4_pad", "_g4_shim_cls")

    def __init__(self, hf, fast=None, shim_cls=None):
        if shim_cls is None:
            from fastokens._compat import _TokenizerShim as shim_cls
        object.__setattr__(self, "_g4_hf", hf)
        object.__setattr__(self, "_g4_shim_cls", shim_cls)
        object.__setattr__(self, "_g4_fast", fast if fast is not None else shim_cls(hf))
        object.__setattr__(self, "_g4_trunc", _UNSET)
        object.__setattr__(self, "_g4_pad", _UNSET)

    # -- encode routing ------------------------------------------------------------------------------------------
    def _g4_ready(self):
        hf, fast = self._g4_hf, self._g4_fast
        if fast is None:  # a mutation invalidated the encoder: rebuild from the live HF backend
            fast = self._g4_shim_cls(hf)
            object.__setattr__(self, "_g4_fast", fast)
            object.__setattr__(self, "_g4_trunc", _UNSET)
            object.__setattr__(self, "_g4_pad", _UNSET)
        t = hf.truncation
        if t != self._g4_trunc:
            fast.truncation = t
            object.__setattr__(self, "_g4_trunc", t)
        p = hf.padding
        if p != self._g4_pad:
            fast.padding = p
            object.__setattr__(self, "_g4_pad", p)
        fast.encode_special_tokens = hf.encode_special_tokens
        return fast

    def encode(self, sequence, pair=None, is_pretokenized=False, add_special_tokens=True):
        if pair is None and not is_pretokenized and isinstance(sequence, str):
            return self._g4_ready().encode(sequence, add_special_tokens=add_special_tokens)
        return self._g4_hf.encode(sequence, pair, is_pretokenized=is_pretokenized,
                                  add_special_tokens=add_special_tokens)

    def encode_batch(self, input, is_pretokenized=False, add_special_tokens=True):  # noqa: A002 - HF signature
        if not is_pretokenized and all(isinstance(x, str) for x in input):
            return self._g4_ready().encode_batch(list(input), add_special_tokens=add_special_tokens)
        return self._g4_hf.encode_batch(input, is_pretokenized=is_pretokenized, add_special_tokens=add_special_tokens)

    def encode_batch_fast(self, input, is_pretokenized=False, add_special_tokens=True):  # noqa: A002
        if not is_pretokenized and all(isinstance(x, str) for x in input):
            return self._g4_ready().encode_batch_fast(list(input), add_special_tokens=add_special_tokens)
        return self._g4_hf.encode_batch_fast(input, is_pretokenized=is_pretokenized,
                                             add_special_tokens=add_special_tokens)

    # -- mutations: apply to HF, rebuild the encoder lazily ------------------------------------------------------
    def add_tokens(self, tokens):
        n = self._g4_hf.add_tokens(tokens)
        object.__setattr__(self, "_g4_fast", None)
        return n

    def add_special_tokens(self, tokens):
        n = self._g4_hf.add_special_tokens(tokens)
        object.__setattr__(self, "_g4_fast", None)
        return n

    # -- everything else is the HF backend -----------------------------------------------------------------------
    def __getattr__(self, name):
        if name.startswith("_g4_"):
            raise AttributeError(name)
        return getattr(self._g4_hf, name)

    def __setattr__(self, name, value):
        setattr(self._g4_hf, name, value)
        if name in _ENCODE_ONLY_RESET:
            object.__setattr__(self, "_g4_fast", None)

    def __copy__(self):
        return HybridBackend(self._g4_hf, self._g4_fast, self._g4_shim_cls)

    def __deepcopy__(self, memo):
        import copy

        fast = copy.deepcopy(self._g4_fast, memo) if self._g4_fast is not None else None
        new = HybridBackend(copy.deepcopy(self._g4_hf, memo), fast, self._g4_shim_cls)
        memo[id(self)] = new
        return new

    def __reduce__(self):
        # Pickles as the plain HF backend: another process gets upstream behaviour, never a broken object.
        import operator

        return operator.itemgetter(0), ((self._g4_hf,),)

    def __repr__(self):
        return f"HybridBackend(hf={self._g4_hf!r})"


def make_decode_stream_cls(orig):
    class HybridDecodeStream:
        """``tokenizers.decoders.DecodeStream`` that steps HF's stream with HF's tokenizer (unwrapping HybridBackend)."""

        __slots__ = ("_g4_stream",)
        _g4_orig = orig

        def __init__(self, *args, **kwargs):
            self._g4_stream = orig(*args, **kwargs)

        def step(self, tokenizer, id):  # noqa: A002 - HF signature
            return self._g4_stream.step(tokenizer._g4_hf if type(tokenizer) is HybridBackend else tokenizer, id)

        def __getattr__(self, name):
            if name == "_g4_stream":
                raise AttributeError(name)
            return getattr(self._g4_stream, name)

    return HybridDecodeStream


def _rebind_decode_stream() -> None:
    import tokenizers.decoders as td

    cur = td.DecodeStream
    if getattr(cur, "_g4_orig", None) is not None:
        return  # already rebound
    td.DecodeStream = make_decode_stream_cls(cur)


def _log():
    import logging

    return logging.getLogger("vllm.g4bench.hybrid_tokenizer")


def install(tokenizer):
    """Swap ``tokenizer._tokenizer`` for a HybridBackend (in place) and rebind DecodeStream; returns ``tokenizer``."""
    import tokenizers

    backend = getattr(tokenizer, "_tokenizer", None)
    if type(backend) is HybridBackend:
        return tokenizer
    if not isinstance(backend, tokenizers.Tokenizer):
        _log().warning("g4bench hybrid_tokenizer: backend %s is not a tokenizers.Tokenizer (slow tokenizer, or "
                       "VLLM_USE_FASTOKENS set); left unchanged", type(backend).__name__)
        return tokenizer
    tokenizer._tokenizer = HybridBackend(backend)
    _rebind_decode_stream()
    _log().warning("g4bench hybrid_tokenizer: %s prompt encode -> fastokens, decode/DecodeStream -> HF tokenizers",
                   type(tokenizer).__name__)
    return tokenizer


_installed = False


def install_global() -> None:
    """Wrap ``TokenizersBackend.from_pretrained`` (as fastokens.patch_transformers does) so every HF fast tokenizer
    loaded afterwards gets a HybridBackend before vLLM copies it. Idempotent; never stacks on VLLM_USE_FASTOKENS."""
    global _installed
    import os

    if _installed:
        return
    _installed = True
    if os.environ.get("VLLM_USE_FASTOKENS", "0").strip() not in ("", "0"):
        _log().warning("g4bench hybrid_tokenizer: VLLM_USE_FASTOKENS is set (full fastokens backend); not installed")
        return
    from fastokens._compat import _TokenizerShim  # noqa: F401 - fail loudly at startup if fastokens is missing
    from transformers.tokenization_utils_tokenizers import TokenizersBackend

    orig = TokenizersBackend.from_pretrained

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return install(orig.__func__(cls, *args, **kwargs))

    TokenizersBackend.from_pretrained = from_pretrained
