# Profile summary

> **PROFILED (torch profiler on).** Descriptive of the profiled window only; profiler overhead included. Never compare these timings, or this point's benchmark row, with the article.

- trace `rank0.1794612341383347343.pt.trace.json.gz` (24,530,154 bytes)

GPU span 13011.549 ms, busy 12822.513 ms, **idle fraction 0.015**; summed GPU op time 12845.279 ms (597722 GPU events, 82 distinct names).

## By kernel class

| bucket | ms | share |
|---|---:|---:|
| memcpy | 8.686 | 0.001 |
| attention | 719.466 | 0.056 |
| spec_decode | 12.860 | 0.001 |
| sampling | 62.664 | 0.005 |
| gemm | 10479.450 | 0.816 |
| norm_act | 619.448 | 0.048 |
| other | 942.705 | 0.073 |

## By model-runner phase

not available: no `gpu_model_runner: *` ranges (set VLLM_CUSTOM_SCOPES_FOR_PROFILING=1).

## Engine steps (`execute_context_*` annotations)

| steps | n | CPU ms mean | p50 | p90 | GPU kernel ms mean | gen reqs mean | gen tokens mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| all | 538 | 2.339 | 2.099 | 2.468 | 19.050 | 0.993 | 6.948 |
| decode_only | 536 | 2.201 | 2.099 | 2.454 | 17.260 | 0.996 | 6.974 |
| with_prefill | 2 | 39.406 | 39.310 | 39.502 | 498.663 | 0.000 | 0.000 |

## Top 25 GPU ops by total time

| # | name | class | count | total ms | mean µs | share |
|---:|---|---|---:|---:|---:|---:|
| 1 | `nvjet_sm120_tst_mma_64x8x128_5_16x8x128_tmaAB_alignCD4_bz_TNNN` | gemm | 36360 | 4289.912 | 118.0 | 0.334 |
| 2 | `nvjet_sm120_tst_mma_128x8x64_5_32x8x64_tmaAB_alignCD4_splitK_TNNN` | gemm | 54042 | 2499.139 | 46.2 | 0.195 |
| 3 | `nvjet_sm120_tst_mma_64x8x64_10_16x8x64_tmaAB_alignCD4_bz_TNNN` | gemm | 28302 | 1176.876 | 41.6 | 0.092 |
| 4 | `void cutlass::Kernel2<cutlass_75_wmma_tensorop_bf16_s161616gemm_bf16_32x32_64x2_tn_align1>(cutlas...` | gemm | 534 | 725.928 | 1359.4 | 0.057 |
| 5 | `kernel_unified_attention` | attention | 38496 | 676.632 | 17.6 | 0.053 |
| 6 | `_g4_dhq_fp8_scan` | other | 3216 | 589.150 | 183.2 | 0.046 |
| 7 | `nvjet_sm120_tst_mma_256x128x64_2_64x64x64_tmaAB_alignCD4_bz_TNNN` | gemm | 192 | 506.657 | 2638.8 | 0.039 |
| 8 | `nvjet_sm120_tst_mma_256x8x64_2_64x8x64_tmaAB_alignCD4_splitK_TNNN` | gemm | 21360 | 485.950 | 22.8 | 0.038 |
| 9 | `nvjet_sm120_tst_mma_128x8x64_5_32x8x64_tmaAB_alignCD4_bz_TNNN` | gemm | 15546 | 346.735 | 22.3 | 0.027 |
| 10 | `reduce_segments` | other | 38496 | 162.767 | 4.2 | 0.013 |
| 11 | `triton_red_fused_add_mul_rms_norm_2` | norm_act | 25728 | 153.992 | 6.0 | 0.012 |
| 12 | `nvjet_sm120_tst_mma_128x160x64_2_64x40x64_tmaAB_alignCD4_bz_TNNN` | gemm | 96 | 144.204 | 1502.1 | 0.011 |
| 13 | `_g4_pfa_gl_fwd` | other | 96 | 128.109 | 1334.5 | 0.010 |
| 14 | `void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, float, __nv_bfloat16, false...` | gemm | 80208 | 112.886 | 1.4 | 0.009 |
| 15 | `triton_red_fused_add_rms_norm_0` | norm_act | 25728 | 112.635 | 4.4 | 0.009 |
| 16 | `triton_poi_fused_gelu_mul_slice_1` | norm_act | 38592 | 108.728 | 2.8 | 0.009 |
| 17 | `nvjet_sm120_tst_mma_128x176x64_2_32x88x64_tmaAB_alignCD4_bz_TNNN` | gemm | 48 | 103.487 | 2156.0 | 0.008 |
| 18 | `triton_per_fused_add_cat_index_select_mul_rms_norm_split_split_with_sizes_sub_unsqueeze_view_4` | norm_act | 25192 | 63.403 | 2.5 | 0.005 |
| 19 | `triton_per_fused_add_cat_index_select_mul_rms_norm_split_split_with_sizes_sub_unsqueeze_view_3` | norm_act | 25192 | 62.516 | 2.5 | 0.005 |
| 20 | `nvjet_sm120_tst_mma_64x8x128_5_16x8x128_tmaAB_alignCD4_splitK_TNNN` | gemm | 4806 | 53.831 | 11.2 | 0.004 |
| 21 | `void at::native::sbtopk::gatherTopK<float, unsigned int, 1, false>(at::cuda::detail::TensorInfo<f...` | sampling | 3216 | 48.541 | 15.1 | 0.004 |
| 22 | `reshape_and_cache_kernel_flash` | attention | 25728 | 42.834 | 1.7 | 0.003 |
| 23 | `triton_per_fused_rms_norm_split_with_sizes_view_5` | norm_act | 25192 | 31.437 | 1.2 | 0.002 |
| 24 | `triton_per_fused_add_mul_rms_norm_2` | norm_act | 12864 | 31.290 | 2.4 | 0.002 |
| 25 | `nvjet_sm120_tst_mma_96x256x64_2_48x64x64_tmaAB_alignCD4_bz_TNNN` | gemm | 48 | 31.109 | 648.1 | 0.002 |
