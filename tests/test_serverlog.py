"""Server-log parsing, with lines constructed from the pinned engines' own format strings."""
from g4bench import serverlog

VLLM = """INFO 11-10 [config.py:351] Gemma4 model has heterogeneous head dimensions {'sliding_attention': 256, 'full_attention': 512}. FA4 not available, forcing TRITON_ATTN backend.
INFO 11-10 [cuda.py:509] Using TRITON_ATTN attention backend out of potential backends: ['TRITON_ATTN', 'FLEX_ATTENTION'].
INFO 11-10 [kv_cache_utils.py:1147] GPU KV cache size: 1,234,567 tokens, Maximum concurrency for 131,072 tokens per request: 9.42x
INFO 11-10 [gpu_model_runner.py:6517] Graph capturing finished in 31 secs, took 1.25 GiB
"""
SGL = """[2026-11-10] server_args=ServerArgs(model_path='google/gemma-4-12B-it', attention_backend='triton', speculative_algorithm='FROZEN_KV_MTP', speculative_num_steps=5, speculative_num_draft_tokens=6, mem_fraction_static=0.85, cuda_graph_max_bs=160, disable_radix_cache=True)
[2026-11-10] max_total_num_tokens=812345, chunked_prefill_size=8192, max_prefill_tokens=16384, max_running_requests=96, context_len=262144
"""


def test_vllm_effective_config():
    e = serverlog.parse(VLLM, "vllm")
    assert e["gemma4_backend_decision"] == "TRITON_ATTN (FA4 unavailable)"
    assert e["attention_backend"] == "TRITON_ATTN" and e["kv_cache_tokens"] == 1234567
    assert e["max_model_len"] == 131072 and e["max_concurrency_at_max_len"] == 9.42
    assert e["cuda_graph_capture_s"] == 31 and e["cuda_graph_mem_gib"] == 1.25


def test_sglang_effective_config_repr_and_dict_styles():
    e = serverlog.parse(SGL, "sglang")
    a = e["sglang_server_args"]
    assert a["attention_backend"] == "triton" and a["speculative_algorithm"] == "FROZEN_KV_MTP"
    assert a["speculative_num_steps"] == "5" and a["disable_radix_cache"] == "True"
    assert e["speculative_algorithm"] == "FROZEN_KV_MTP" and e["kv_cache_tokens"] == 812345
    assert e["max_running_requests"] == 96
    d = serverlog.parse("server_args={'attention_backend': 'fa3', 'speculative_algorithm': 'NEXTN'}", "sglang")
    assert d["attention_backend"] == "fa3" and d["speculative_algorithm"] == "NEXTN"


def test_absent_lines_are_none_never_guessed():
    for eng in ("vllm", "sglang", None):
        e = serverlog.parse("nothing useful here\n", eng)
        assert all(v is None for k, v in e.items() if k != "engine")
