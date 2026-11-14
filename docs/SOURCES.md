# Source verification log

Everything below was checked against a primary source on **2026-11-10** by this
project. "Claim" = what the Backboard article says; "Verified" = what the primary
source says; "Status" = ✅ confirmed / ⚠️ partly / ❓ unverifiable / ❌ contradicted.

## The article

- URL: <https://backboard.io/blog/gemma-4-12b-vllm-sglang-inference>
- Title: "Gemma 4: Up to 2.2× faster throughput than vLLM and SGLang"; author Rob Imbeault; dated **2026-11-09**.
- HTML sha256 at retrieval: `a3c2af43b5b4fe20f8487a6cb42f42a6a6ac06d92e23a4968593ff3fd4c7a964`.
- All numeric tables transcribed to [`configs/targets/backboard-2026-11-09.json`](../configs/targets/backboard-2026-11-09.json);
  `tests/test_targets_stats.py::test_transcription_matches_article_ratio_tables` recomputes the article's
  ratio tables from its absolute tables (64 checks) to catch transcription errors.
- The article publishes absolute values only for: output throughput, p99 TPOT, p99 E2EL, p90 TTFT and
  "speed per user". Mean/median TTFT/TPOT/E2EL appear **only as ratios**.
- It links no code, no raw results, no client commit, and no server command lines.

## Model

| Item | Claim | Verified (primary source) | Status |
|---|---|---|---|
| Target model ID | `google/gemma-4-12B-it` | HF API: exists, `sha=43c9d014b231474fa7212dbb48a762566c2b255d` (lastModified 2026-10-30), created 2026-05-23 | ✅ |
| Architecture | "Gemma 4 12B" | `Gemma4UnifiedForConditionalGeneration`, `model_type=gemma4_unified`; encoder-free unified multimodal (text/image/audio); 48 layers (40 sliding-window 1024 + 8 global, final layer global); hidden 3840, 16 heads, 8 KV heads × 256 (sliding), 1 KV head × 512 (global) with `attention_k_eq_v=true`; vocab 262144, tied embeddings; final logit softcap 30; model card: 11.95B params, 256K context (vLLM recipe pins 128K from config) | ✅ |
| Precision | BF16 | `config.dtype=bfloat16`; single `model.safetensors` = 23,919,549,408 bytes | ✅ |
| Licence / access | – | `license: apache-2.0` (card links Gemma 4 licence page), **`gated: false`** → no HF token needed | ✅ |
| Draft model | "official Gemma assistant model" | `google/gemma-4-12B-it-assistant`, `sha=3b0b0a4a3b31b5556516329227fb55f275c23c9f`, `Gemma4UnifiedAssistantForCausalLM`; 4 layers, hidden 1024, `num_kv_shared_layers=4` (reads target KV), ~~centroid-masked LM head, only ~4096 logits per step~~ *corrected 2026-11-11 (D-054)*: config has `use_ordered_embeddings: false`, so the head is the **dense, tied 262144 × 1024 BF16 embedding** (all 262144 logits per draft token); the centroid/ordered-embedding shortlist in vLLM/SGLang `gemma4_mtp.py` applies only to the E2B/E4B assistants; 845,719,296 bytes; Apache-2.0, ungated | ✅ |

Model-availability test: `g4bench sources` (HF API + HEAD on weights at the pinned revision) — passes for both repos.

## Hardware

| Item | Claim | Verified | Status |
|---|---|---|---|
| GPU | 1× RTX PRO 6000 Blackwell 96 GB, "two identical machines, default clocks and power" | Edition (Workstation 600 W / Server / Max-Q 300 W), driver, CUDA, host CPU **not stated** | ⚠️ |
| Memory bandwidth | – | RTX PRO 6000 Blackwell **Server Edition** (the edition Modal serves, per `nvidia-smi`): rated **1597 GB/s** (added 2026-11-11, D-052); measured device copy 1.46 TB/s | ✅ |
| Availability for us | – | Modal offers `gpu="RTX-PRO-6000"`; edition not stated by Modal either → we record `nvidia-smi` name, power limits, clocks per run | ✅ (exact model), ❓ (edition) |

## Software versions

| Item | Claim | Verified | Status |
|---|---|---|---|
| vLLM | 0.33.0 | PyPI upload 2026-11-02; Docker `vllm/vllm-openai:v0.33.0` index `sha256:d825baf9…`, amd64 `sha256:91ee972a…`. Tag `v0.33.0` contains `gemma4_unified.py`, `gemma4_mtp.py` (centroid masking), `v1/spec_decode/gemma4.py`. `SpeculativeConfig` has `revision`, `method`, `num_speculative_tokens`. | ✅ |
| SGLang | 0.5.23 | PyPI upload 2026-10-30; Docker `lmsysorg/sglang:v0.5.23` index `sha256:f832b5c3…`, amd64 `sha256:f7876bef…`. Tag contains `models/gemma4_unified.py`, `models/gemma4_mtp.py` (Frozen-KV MTP with centroid head). Flags `--speculative-draft-model-revision`, `--disable-radix-cache`, `--mem-fraction-static` exist. **Also has `--speculative-draft-model-quantization`.** | ✅ |
| Benchmark client | "SemiAnalysis InferenceX … client and chat template" | InferenceX `main` @ `0b991451d71d7dca4b478fad09a28f14c43fe804` (2026-11-10). The article names **no commit**; the client may have changed between their run and this commit. | ⚠️ |

## Baseline recipes

| Item | Claim | Verified | Status |
|---|---|---|---|
| vLLM recipe | MTP 4 tokens, async scheduling, gpu-mem 0.90, prefix caching off, "published recipe, default settings" | vllm-project/recipes `models/Google/gemma-4-12B-it.yaml` @ `fa1173a1`: spec_decoding opt-in `{"model":"google/gemma-4-12B-it-assistant","num_speculative_tokens":4}`, recommends 4–8; `--async-scheduling`; `--gpu-memory-utilization 0.90`. Prefix caching off is the InferenceX convention, not in the NVIDIA recipe. vLLM docs: use `"method":"mtp"` for Gemma 4 assistants. | ✅ (flags) / ⚠️ (max-model-len, max-num-seqs etc. unstated) |
| SGLang recipe | NEXTN 5 steps / 6 draft tokens, mem 0.85 static, radix off | docs.sglang.io cookbook Gemma4: 12B MTP command `--speculative-algorithm NEXTN --speculative-draft-model-path google/gemma-4-12B-it-assistant --speculative-num-steps 5 --speculative-num-draft-tokens 6 --speculative-eagle-topk 1 --mem-fraction-static 0.85` | ✅ |

## Benchmark methodology (InferenceX conventions at commit 0b991451)

From `inferencex-e2e/infx/bench/fixed_seq.py` (copied to `docs/sources/inferencex_fixed_seq.py.txt`):

- `--dataset-name random`, `--request-rate inf`, `--ignore-eos`, `--num-prompts = 10 × conc`,
  `--num-warmups = 2 × conc` (warmups repeat the first request), `--percentile-metrics ttft,tpot,itl,e2el`,
  `--use-chat-template` for spec-decode runs (InferenceX PR checklist requires chat templates for spec decoding).
- `RANDOM_RANGE_RATIO = "0.8"` in `infx/srt_slurm/generate.py`: input and output lengths are drawn
  uniformly from `[0.8·L, L]`. **The article does not state the range ratio.** See DECISIONS D-004:
  16/24 published (p99 E2EL, p99 TPOT) pairs are mathematically impossible with fixed 1024-token outputs.
- With the Gemma 4 chat template (13 template tokens incl. an empty thought channel) and ratio 0.8,
  "1K" prompts average ~917–948 tokens and "8K" ~7.33–7.45K tokens; outputs average ~910–939 tokens
  (`configs/fingerprints-rr0.8.json`).
- Sampling: the client sends `temperature: 0.0` (greedy) for the vLLM/OpenAI-completions backend.
- InferenceX's own docs warn **random-token prompts distort speculative acceptance** (golden-AL README) and
  its PR checklist **forbids lowering draft-model precision below the shipped precision**. Backboard's
  "draft head shortlists candidates in 4-bit before rescoring in BF16" would not be an admissible
  InferenceX submission under that rule, although Backboard states target outputs are unchanged.
- `KNOWN_LIMITATION.md`: the single-process client becomes client-bound for small models at very high QPS.
  Our c≤64 burst regime is unlikely to hit it but we record client CPU and check it (EXPERIMENTS E1).

## Things that cannot be verified

- Backboard's engine, its kernels, its 4-bit shortlist method, and its raw results (proprietary / unpublished).
- Exact server command lines, max-model-len, scheduler limits, attention backends, CUDA graph settings of the baselines.
- InferenceX commit, client host CPU, GPU edition / power limit, driver/CUDA versions.
- Whether "1K/8K" used range ratio 0.8 (likely, see above) or 1.0.

## Real-prompt data (added 2026-11-10)

| Artefact | Pin | Verified |
|---|---|---|
| InferenceX SPEED-Bench prep | `inferencex-e2e/infx/bench/speedbench.py` @ `0b991451`: `SPEED_BENCH_REVISION=454f8845…`, `PREPARE_PY_SHA256=a551be4d…`, `PROMPTS_SHA256=6f184545…` | ✅ read |
| NeMo-Skills prepare.py | Skills @ `5ac8609a`, sha256 `a551be4d…` | ✅ downloaded, hash matches |
| nvidia/SPEED-Bench | rev `454f88454792dfa3ccfd7ef15fff248efde44cd1`, ungated, licence "other" (NVIDIA License.pdf); qualitative / throughput_1k / throughput_8k parquet LFS sha256 pinned in `src/g4bench/speedbench.py` | ✅ |
| deepmind/code_contests | rev `802411c3…`, test parquet sha256 `aa426cbd…` | ✅ |
| livecodebench/code_generation_lite | rev `0fe84c39…`, test..test6.jsonl sha256 pinned; all six stream-hashed | ✅ |
| Resulting coding set | 80 prompts, digest `6f1845452f1cfbc1930ea779ba6c3d5b437a9d5853eca199dd9b7619b20a0c18` | ✅ equals InferenceX |

Coding sources: 61 humanevalpack + 9 Spec-Bench (inline), 5 code_contests + 5 LiveCodeBench (resolved).
