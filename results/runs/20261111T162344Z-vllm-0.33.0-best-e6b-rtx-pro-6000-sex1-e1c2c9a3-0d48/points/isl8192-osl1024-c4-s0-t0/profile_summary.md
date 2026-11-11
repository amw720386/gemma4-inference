# Profile summary

> **PROFILED (torch profiler on).** Descriptive of the profiled window only; profiler overhead included. Never compare these timings, or this point's benchmark row, with the article.

- trace `rank0.1794414385813238416.pt.trace.json.gz` (4,770,195 bytes)

GPU span 4035.846 ms, busy 4015.025 ms, **idle fraction 0.005**; summed GPU op time 4016.221 ms (95865 GPU events, 76 distinct names).

## By kernel class

| bucket | ms | share |
|---|---:|---:|
| memcpy | 1.916 | 0.001 |
| attention | 805.389 | 0.201 |
| spec_decode | 2.158 | 0.001 |
| sampling | 11.826 | 0.003 |
| gemm | 2986.109 | 0.744 |
| norm_act | 183.182 | 0.046 |
| other | 25.642 | 0.006 |

## By model-runner phase

not available: no `gpu_model_runner: *` ranges (set VLLM_CUSTOM_SCOPES_FOR_PROFILING=1).

## Engine steps (`execute_context_*` annotations)

| steps | n | CPU ms mean | p50 | p90 | GPU kernel ms mean | gen reqs mean | gen tokens mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| all | 86 | 10.874 | 2.447 | 4.638 | 39.805 | 3.826 | 26.779 |
| decode_only | 77 | 2.259 | 2.308 | 3.038 | 19.779 | 3.948 | 27.636 |
| with_prefill | 9 | 84.584 | 108.373 | 124.963 | 211.136 | 2.778 | 19.444 |

## Top 25 GPU ops by total time

| # | name | class | count | total ms | mean µs | share |
|---:|---|---|---:|---:|---:|---:|
| 1 | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x128_32x3_tn_align8>(cutla...` | gemm | 896 | 873.709 | 975.1 | 0.217 |
| 2 | `kernel_unified_attention` | attention | 6192 | 794.933 | 128.4 | 0.198 |
| 3 | `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_64x1_tn_align8>(cutlas...` | gemm | 3696 | 647.976 | 175.3 | 0.161 |
| 4 | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x64_64x4_tn_align8>(cutlas...` | gemm | 7392 | 415.580 | 56.2 | 0.103 |
| 5 | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_tn_align8>(cutla...` | gemm | 752 | 408.052 | 542.6 | 0.102 |
| 6 | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x64_32x4_tn_align8>(cutlas...` | gemm | 554 | 277.140 | 500.3 | 0.069 |
| 7 | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_64x6_tn_align8>(cutlass...` | gemm | 3696 | 173.269 | 46.9 | 0.043 |
| 8 | `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_128x1_tn_align8>(cutla...` | gemm | 2668 | 93.113 | 34.9 | 0.023 |
| 9 | `triton_poi_fused_gelu_mul_slice_1` | norm_act | 6192 | 70.984 | 11.5 | 0.018 |
| 10 | `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_tn_align8>(cutlass...` | gemm | 4644 | 57.162 | 12.3 | 0.014 |
| 11 | `triton_red_fused_add_mul_rms_norm_2` | norm_act | 4128 | 36.337 | 8.8 | 0.009 |
| 12 | `triton_red_fused_add_rms_norm_0` | norm_act | 4128 | 26.073 | 6.3 | 0.006 |
| 13 | `void cublasLt::splitKreduce_kernel<32, 16, int, __nv_bfloat16, __nv_bfloat16, float, __nv_bfloat1...` | gemm | 12036 | 21.764 | 1.8 | 0.005 |
| 14 | `reduce_segments` | other | 5760 | 17.489 | 3.0 | 0.004 |
| 15 | `triton_per_fused_add_cat_index_select_mul_rms_norm_split_split_with_sizes_sub_unsqueeze_view_3` | norm_act | 4042 | 17.301 | 4.3 | 0.004 |
| 16 | `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_128x2_tn_align8>(cutla...` | gemm | 2064 | 16.849 | 8.2 | 0.004 |
| 17 | `triton_per_fused_add_cat_index_select_mul_rms_norm_split_split_with_sizes_sub_unsqueeze_view_4` | norm_act | 4042 | 12.761 | 3.2 | 0.003 |
| 18 | `reshape_and_cache_kernel_flash` | attention | 4128 | 10.456 | 2.5 | 0.003 |
| 19 | `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<c10::BFloat16, at::native::ArgMaxOps<...` | sampling | 516 | 7.799 | 15.1 | 0.002 |
| 20 | `triton_per_fused_rms_norm_split_with_sizes_view_5` | norm_act | 4042 | 5.966 | 1.5 | 0.002 |
| 21 | `triton_per_fused_add_mul_rms_norm_2` | norm_act | 2064 | 4.700 | 2.3 | 0.001 |
| 22 | `triton_per_fused_add_rms_norm_0` | norm_act | 2064 | 3.227 | 1.6 | 0.001 |
| 23 | `triton_per_fused_add_cat_index_select_mul_rms_norm_split_sub_unsqueeze_view_3` | norm_act | 1548 | 3.045 | 2.0 | 0.001 |
| 24 | `_compute_local_logits_stats_kernel` | sampling | 86 | 1.875 | 21.8 | 0.001 |
| 25 | `_resample_kernel` | sampling | 86 | 1.776 | 20.7 | 0.000 |
