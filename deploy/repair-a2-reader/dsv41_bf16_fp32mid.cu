// Isolated SM120 DSV4.1 BF16 split-K decode with FP32 partial accumulation.
// Calls FlashInfer's own run_bf16_tiles<EXTRA_FP4, MT> (2x pipelined PTX gather
// + exact BF16xBF16->FP32 MMA) per split and keeps mid_out in FP32 until merge.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include <flashinfer/attention/sparse_mla_sm120/common/lse.cuh>
#include <flashinfer/attention/sparse_mla_sm120/execution/attention_plan.h>
#include <flashinfer/attention/sparse_mla_sm120/kernels/dsv41_bf16/prefill.cuh>

#include "tvm_ffi_utils.h"

using tvm::ffi::Optional;

namespace flashinfer::sparse_mla_sm120::repair_a2 {

using bf16 = __nv_bfloat16;
using kernels::dsv41_bf16::Dsv41Bf16Resources;
using kernels::dsv41_bf16::Dsv41Bf16Smem;
using kernels::dsv41_bf16::run_bf16_tiles;

template <int NUM_HEADS, bool EXTRA_FP4, ModelType MT = ModelType::DSV4_1_INLINE>
__global__ void __launch_bounds__(Dsv41Bf16Resources::BLOCK_THREADS, 1)
    sparse_mla_decode_dsv41_bf16_fp32mid_kernel(
        const bf16* __restrict__ Q, const uint8_t* __restrict__ KV_cache,
        const int32_t* __restrict__ indices, float* __restrict__ mid_out,
        float* __restrict__ mid_lse, const int* __restrict__ topk_length_ptr,
        const uint8_t* __restrict__ extra_KV_cache, const int32_t* __restrict__ extra_indices,
        const int* __restrict__ extra_topk_length_ptr, int extra_topk, int pbs_extra,
        size_t extra_page_stride_bytes, int num_tokens, int num_heads, int topk,
        int scratch_split_stride, int chunks_per_block, float sm_scale, size_t page_stride_bytes,
        size_t indices_stride_elems, size_t extra_indices_stride_elems, int main_page_block_size) {
  using R = Dsv41Bf16Resources;
  using Geometry = R::Geometry;
  const int t = blockIdx.x, hs = blockIdx.y * R::HEADS_PER_CTA, split = blockIdx.z;
  const int qh = NUM_HEADS ? NUM_HEADS : num_heads;
  const int mh = NUM_HEADS ? NUM_HEADS : int(gridDim.y) * R::HEADS_PER_CTA;
  const int vh = min(R::HEADS_PER_CTA, qh - hs);
  const int tidx = threadIdx.x, warp = tidx / R::WARP_THREADS, lane = tidx % R::WARP_THREADS;
  const int gid = lane / 4, tid = lane % 4;
  if (t >= num_tokens) return;
  int len = topk_length_ptr ? topk_length_ptr[t] : topk;
  len = max(0, min(topk, len));
  int elen = extra_KV_cache ? (extra_topk_length_ptr ? extra_topk_length_ptr[t] : extra_topk) : 0;
  elen = max(0, min(extra_topk, elen));
  const int main_chunks = (len + R::CANDIDATES - 1) / R::CANDIDATES;
  const int chunks = main_chunks + (elen + R::CANDIDATES - 1) / R::CANDIDATES;
  const int lo = split * chunks_per_block, hi = min(lo + chunks_per_block, chunks);
  if (lo >= chunks) {
    if (tidx < vh) mid_lse[((size_t)t * mh + hs + tidx) * scratch_split_stride + split] = -1e30f;
    return;
  }
  extern __shared__ __align__(16) unsigned char storage[];
  auto& sm = *reinterpret_cast<Dsv41Bf16Smem*>(storage);
  for (int i = tidx; i < R::HEADS_PER_CTA * R::QK_VECTORS; i += R::BLOCK_THREADS) {
    const int h = i / R::QK_VECTORS, d = i % R::QK_VECTORS * R::VECTOR_ELEMS;
    uint4 data =
        h < vh ? *reinterpret_cast<const uint4*>(Q + ((size_t)t * qh + hs + h) * Geometry::D_QK + d)
               : make_uint4(0, 0, 0, 0);
    *reinterpret_cast<uint4*>(&sm.q[h][d]) = data;
  }
  float acc[R::PV_TILES_PER_WARP][4] = {};
  float m[2] = {-1e30f, -1e30f}, sum[2] = {0.f, 0.f};
  run_bf16_tiles<EXTRA_FP4, MT>(
      sm, KV_cache, indices, extra_KV_cache, extra_indices, t, lo, hi, len, elen, main_chunks,
      main_page_block_size, pbs_extra, page_stride_bytes, extra_page_stride_bytes,
      indices_stride_elems, extra_indices_stride_elems, sm_scale, acc, m, sum);
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const int head = hs + gid + h * 8;
    if (head < qh) {
      const float inv = sum[h] > 0.f ? 1.f / sum[h] : 0.f;
      const size_t base =
          ((size_t)t * mh + head) * scratch_split_stride * Geometry::D_V + split * Geometry::D_V;
#pragma unroll
      for (int v = 0; v < R::PV_TILES_PER_WARP; ++v) {
        const int d = v * (R::WARPS * R::MMA_N) + warp * R::MMA_N + tid * 2;
        *reinterpret_cast<float2*>(mid_out + base + d) =
            make_float2(acc[v][h * 2] * inv, acc[v][h * 2 + 1] * inv);
      }
      if (warp == 0 && tid == 0)
        mid_lse[((size_t)t * mh + head) * scratch_split_stride + split] =
            sum[h] > 0.f ? log2f(sum[h]) + m[h] : -1e30f;
    }
  }
}

template <int NUM_HEADS, int D_V_VAL, int BLOCK_THREADS, int DIMS_PER_THREAD>
static __global__ void __launch_bounds__(BLOCK_THREADS, 8)
    sparse_mla_decode_dsv4_fp32mid_merge_kernel(
        const float* __restrict__ mid_out, const float* __restrict__ mid_lse,
        bf16* __restrict__ output, float* __restrict__ out_lse,
        const float* __restrict__ attn_sink, int num_tokens, int num_splits,
        int num_heads, int mid_heads, size_t stride_out_lse, float lse_scale) {
  static_assert(BLOCK_THREADS % 32 == 0, "BLOCK_THREADS must be multiple of 32");
  static_assert(DIMS_PER_THREAD % 8 == 0, "DIMS_PER_THREAD must be multiple of 8");
  static_assert(BLOCK_THREADS * DIMS_PER_THREAD == D_V_VAL, "block must cover full D_V");
  constexpr int VECS_PER_THREAD = DIMS_PER_THREAD / 8;
  const int q_heads = (NUM_HEADS == 0) ? num_heads : NUM_HEADS;
  const int m_heads = (NUM_HEADS == 0) ? mid_heads : NUM_HEADS;

  const int t_idx = blockIdx.x;
  const int h = blockIdx.y;
  if (t_idx >= num_tokens || h >= q_heads) return;
  const int tid = threadIdx.x;

  extern __shared__ __align__(16) unsigned char merge_smem_raw[];
  float* sm_lse = reinterpret_cast<float*>(merge_smem_raw);
  __shared__ float sm_gmax;
  __shared__ float sm_inv_gsum;
  __shared__ float sm_glse;

  const float* lse_ptr = mid_lse + (size_t)t_idx * m_heads * num_splits + (size_t)h * num_splits;
  for (int sp = tid; sp < num_splits; sp += BLOCK_THREADS) {
    sm_lse[sp] = lse_ptr[sp];
  }
  __syncthreads();

  if (tid < 32) {
    float local_max = -1e30f;
    for (int sp = tid; sp < num_splits; sp += 32) {
      local_max = fmaxf(local_max, sm_lse[sp]);
    }
#pragma unroll
    for (int s = 16; s >= 1; s >>= 1) {
      local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, s));
    }
    float gmax = (local_max > -1e29f) ? local_max : 0.f;
    float sink_log2 = 0.f;
    if (attn_sink != nullptr) {
      sink_log2 = __ldg(attn_sink + h) * LOG2E;
      gmax = local_max > -1e29f ? fmaxf(gmax, sink_log2) : sink_log2;
    }

    float local_sum = 0.f;
    for (int sp = tid; sp < num_splits; sp += 32) {
      float lse_sp = sm_lse[sp];
      if (lse_sp > -1e29f) local_sum += exp2f(lse_sp - gmax);
    }
#pragma unroll
    for (int s = 16; s >= 1; s >>= 1) {
      local_sum += __shfl_xor_sync(0xffffffff, local_sum, s);
    }
    if (tid == 0) {
      float total_sum = local_sum;
      if (attn_sink != nullptr) total_sum += exp2f(sink_log2 - gmax);
      sm_gmax = gmax;
      sm_inv_gsum = (total_sum > 0.f) ? (1.f / total_sum) : 0.f;
      sm_glse = (total_sum > 0.f) ? (log2f(total_sum) + gmax) : -INFINITY;
    }
  }
  __syncthreads();
  const float global_max = sm_gmax;
  const float inv_global_sum = sm_inv_gsum;

  const float* mid_base = mid_out + ((size_t)t_idx * m_heads + h) * (size_t)num_splits * D_V_VAL;
  bf16* out_ptr = output + ((size_t)t_idx * q_heads + h) * D_V_VAL;
  const int dim_base = tid * DIMS_PER_THREAD;

  float acc[DIMS_PER_THREAD];
#pragma unroll
  for (int d = 0; d < DIMS_PER_THREAD; d++) acc[d] = 0.f;

  for (int sp = 0; sp < num_splits; sp++) {
    float lse_sp = sm_lse[sp];
    if (lse_sp <= -1e29f) continue;
    const float weight = exp2f(lse_sp - global_max);
    const float* row_base = mid_base + (size_t)sp * D_V_VAL + dim_base;
#pragma unroll
    for (int v = 0; v < VECS_PER_THREAD; v++) {
      const float4 f0 = *reinterpret_cast<const float4*>(row_base + v * 8);
      const float4 f1 = *reinterpret_cast<const float4*>(row_base + v * 8 + 4);
      acc[v * 8 + 0] += weight * f0.x;
      acc[v * 8 + 1] += weight * f0.y;
      acc[v * 8 + 2] += weight * f0.z;
      acc[v * 8 + 3] += weight * f0.w;
      acc[v * 8 + 4] += weight * f1.x;
      acc[v * 8 + 5] += weight * f1.y;
      acc[v * 8 + 6] += weight * f1.z;
      acc[v * 8 + 7] += weight * f1.w;
    }
  }

#pragma unroll
  for (int v = 0; v < VECS_PER_THREAD; v++) {
    uint4 packed;
    __nv_bfloat162* pairs = reinterpret_cast<__nv_bfloat162*>(&packed);
#pragma unroll
    for (int p = 0; p < 4; p++) {
      pairs[p] = __floats2bfloat162_rn(acc[v * 8 + p * 2 + 0] * inv_global_sum,
                                       acc[v * 8 + p * 2 + 1] * inv_global_sum);
    }
    *reinterpret_cast<uint4*>(out_ptr + dim_base + v * 8) = packed;
  }
  if (out_lse != nullptr && tid == 0) {
    out_lse[(size_t)t_idx * stride_out_lse + h] = scale_output_lse(sm_glse, lse_scale);
  }
}

template <ModelType MT, int NUM_HEADS>
cudaError_t ensure_smem_attr(int dev_id, bool extra_fp4, int smem_bytes) {
  static uint64_t configured_true = 0;
  static uint64_t configured_false = 0;
  uint64_t& mask = extra_fp4 ? configured_true : configured_false;
  const uint64_t bit = uint64_t{1} << (dev_id & 63);
  if (mask & bit) return cudaSuccess;
  auto kernel = extra_fp4 ? sparse_mla_decode_dsv41_bf16_fp32mid_kernel<NUM_HEADS, true, MT>
                          : sparse_mla_decode_dsv41_bf16_fp32mid_kernel<NUM_HEADS, false, MT>;
  cudaError_t err =
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
  if (err == cudaSuccess) mask |= bit;
  return err;
}

template <ModelType MT, int NUM_HEADS>
cudaError_t launch_fp32mid_impl(
    int dev_id, const bf16* q, const uint8_t* kv, const int32_t* indices, float* mid_out,
    float* mid_lse, bf16* output, float* out_lse, const int* topk_length, const float* attn_sink,
    const uint8_t* extra_kv, const int32_t* extra_indices, const int* extra_topk_length,
    int num_tokens, int num_heads, int topk, int extra_topk, int page_size, int extra_page_size,
    size_t main_stride, size_t extra_stride, size_t indices_stride, size_t extra_indices_stride,
    size_t lse_stride, int active_splits, int cpb, int smem_bytes, bool extra_fp4, float sm_scale,
    float lse_scale, cudaStream_t stream) {
  cudaError_t err = ensure_smem_attr<MT, NUM_HEADS>(dev_id, extra_fp4, smem_bytes);
  if (err != cudaSuccess) return err;
  const int h_blocks = (num_heads + 15) / 16;
  const int q_heads = (NUM_HEADS == 0) ? num_heads : NUM_HEADS;
  auto kernel = extra_fp4 ? sparse_mla_decode_dsv41_bf16_fp32mid_kernel<NUM_HEADS, true, MT>
                          : sparse_mla_decode_dsv41_bf16_fp32mid_kernel<NUM_HEADS, false, MT>;
  kernel<<<dim3(num_tokens, h_blocks, active_splits), dim3(Dsv41Bf16Resources::BLOCK_THREADS),
           smem_bytes, stream>>>(
      q, kv, indices, mid_out, mid_lse, topk_length, extra_kv, extra_indices, extra_topk_length,
      extra_topk, extra_page_size, extra_stride, num_tokens, q_heads, topk, active_splits, cpb,
      sm_scale, main_stride, indices_stride, extra_indices_stride, page_size);
  err = cudaGetLastError();
  if (err != cudaSuccess) return err;
  constexpr int MERGE_THREADS = 64;
  constexpr int D_V = 512;
  auto merge = sparse_mla_decode_dsv4_fp32mid_merge_kernel<NUM_HEADS, D_V, MERGE_THREADS,
                                                           D_V / MERGE_THREADS>;
  merge<<<dim3(num_tokens, q_heads), dim3(MERGE_THREADS),
          size_t(active_splits) * sizeof(float), stream>>>(
      mid_out, mid_lse, output, out_lse, attn_sink, num_tokens, active_splits, q_heads,
      h_blocks * 16, lse_stride, lse_scale);
  return cudaGetLastError();
}

void InitDsv41Bf16Fp32MidSmem(int64_t dev_id, int64_t model_type, int64_t specialized_heads,
                              bool extra_fp4, int64_t smem_bytes) {
  cudaSetDevice(static_cast<int>(dev_id));
  const auto mt = static_cast<ModelType>(model_type);
  cudaError_t err = cudaErrorInvalidValue;
  if (mt == ModelType::DSV4_1_INLINE) {
    err = execution::visit_decode_heads<ModelType::DSV4_1_INLINE>(
        static_cast<int>(specialized_heads), [&](auto head) {
          return ensure_smem_attr<ModelType::DSV4_1_INLINE, decltype(head)::value>(
              static_cast<int>(dev_id), extra_fp4, static_cast<int>(smem_bytes));
        });
  } else if (mt == ModelType::DSV4_1) {
    err = execution::visit_decode_heads<ModelType::DSV4_1>(
        static_cast<int>(specialized_heads), [&](auto head) {
          return ensure_smem_attr<ModelType::DSV4_1, decltype(head)::value>(
              static_cast<int>(dev_id), extra_fp4, static_cast<int>(smem_bytes));
        });
  }
  TVM_FFI_ICHECK_EQ(err, cudaSuccess) << "InitDsv41Bf16Fp32MidSmem: " << cudaGetErrorString(err);
}

void ExecuteDsv41Bf16Fp32Mid(
    TensorView q, TensorView kv_cache, TensorView indices, TensorView mid_out, TensorView mid_lse,
    TensorView output, TensorView out_lse, Optional<TensorView> topk_length,
    Optional<TensorView> attn_sink, Optional<TensorView> extra_kv_cache,
    Optional<TensorView> extra_indices, Optional<TensorView> extra_topk_length,
    int64_t model_type, int64_t specialized_heads, int64_t active_splits, int64_t cpb,
    int64_t smem_bytes, int64_t page_size, int64_t extra_page_size, int64_t main_stride,
    int64_t extra_stride, bool extra_fp4, double sm_scale, double lse_scale) {
  const int dev_id = q.device().device_id;
  ffi::CUDADeviceGuard guard(dev_id);
  cudaStream_t stream = get_stream(q.device());

  const int num_tokens = static_cast<int>(q.size(0));
  const int num_heads = static_cast<int>(q.size(1));
  const int topk = static_cast<int>(indices.size(-1));
  const size_t indices_stride = static_cast<size_t>(indices.stride(0));
  const size_t lse_stride = static_cast<size_t>(out_lse.stride(0));

  const int* topk_len_ptr =
      topk_length.has_value() ? static_cast<const int*>(topk_length.value().data_ptr()) : nullptr;
  const float* attn_sink_ptr =
      attn_sink.has_value() ? static_cast<const float*>(attn_sink.value().data_ptr()) : nullptr;
  const uint8_t* extra_kv_ptr = extra_kv_cache.has_value()
                                    ? static_cast<const uint8_t*>(extra_kv_cache.value().data_ptr())
                                    : nullptr;
  const int32_t* extra_idx_ptr = extra_indices.has_value()
                                     ? static_cast<const int32_t*>(extra_indices.value().data_ptr())
                                     : nullptr;
  const int* extra_len_ptr =
      extra_topk_length.has_value()
          ? static_cast<const int*>(extra_topk_length.value().data_ptr())
          : nullptr;
  const int extra_topk = extra_indices.has_value() ? static_cast<int>(extra_indices.value().size(-1)) : 0;
  const size_t extra_indices_stride =
      extra_indices.has_value() ? static_cast<size_t>(extra_indices.value().stride(0)) : 0;

  const auto mt = static_cast<ModelType>(model_type);
  cudaError_t err = cudaErrorInvalidValue;
  if (mt == ModelType::DSV4_1_INLINE) {
    err = execution::visit_decode_heads<ModelType::DSV4_1_INLINE>(
        static_cast<int>(specialized_heads), [&](auto head) {
          return launch_fp32mid_impl<ModelType::DSV4_1_INLINE, decltype(head)::value>(
              dev_id, static_cast<const bf16*>(q.data_ptr()),
              static_cast<const uint8_t*>(kv_cache.data_ptr()),
              static_cast<const int32_t*>(indices.data_ptr()),
              static_cast<float*>(mid_out.data_ptr()), static_cast<float*>(mid_lse.data_ptr()),
              static_cast<bf16*>(output.data_ptr()), static_cast<float*>(out_lse.data_ptr()),
              topk_len_ptr, attn_sink_ptr, extra_kv_ptr, extra_idx_ptr, extra_len_ptr, num_tokens,
              num_heads, topk, extra_topk, static_cast<int>(page_size),
              static_cast<int>(extra_page_size), static_cast<size_t>(main_stride),
              static_cast<size_t>(extra_stride), indices_stride, extra_indices_stride, lse_stride,
              static_cast<int>(active_splits), static_cast<int>(cpb), static_cast<int>(smem_bytes),
              extra_fp4, static_cast<float>(sm_scale), static_cast<float>(lse_scale), stream);
        });
  } else if (mt == ModelType::DSV4_1) {
    err = execution::visit_decode_heads<ModelType::DSV4_1>(
        static_cast<int>(specialized_heads), [&](auto head) {
          return launch_fp32mid_impl<ModelType::DSV4_1, decltype(head)::value>(
              dev_id, static_cast<const bf16*>(q.data_ptr()),
              static_cast<const uint8_t*>(kv_cache.data_ptr()),
              static_cast<const int32_t*>(indices.data_ptr()),
              static_cast<float*>(mid_out.data_ptr()), static_cast<float*>(mid_lse.data_ptr()),
              static_cast<bf16*>(output.data_ptr()), static_cast<float*>(out_lse.data_ptr()),
              topk_len_ptr, attn_sink_ptr, extra_kv_ptr, extra_idx_ptr, extra_len_ptr, num_tokens,
              num_heads, topk, extra_topk, static_cast<int>(page_size),
              static_cast<int>(extra_page_size), static_cast<size_t>(main_stride),
              static_cast<size_t>(extra_stride), indices_stride, extra_indices_stride, lse_stride,
              static_cast<int>(active_splits), static_cast<int>(cpb), static_cast<int>(smem_bytes),
              extra_fp4, static_cast<float>(sm_scale), static_cast<float>(lse_scale), stream);
        });
  }
  TVM_FFI_ICHECK_EQ(err, cudaSuccess) << "ExecuteDsv41Bf16Fp32Mid: " << cudaGetErrorString(err);
}

}  // namespace flashinfer::sparse_mla_sm120::repair_a2

TVM_FFI_DLL_EXPORT_TYPED_FUNC(init_dsv41_bf16_fp32mid_smem,
                              flashinfer::sparse_mla_sm120::repair_a2::InitDsv41Bf16Fp32MidSmem);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(execute_dsv41_bf16_fp32mid,
                              flashinfer::sparse_mla_sm120::repair_a2::ExecuteDsv41Bf16Fp32Mid);
