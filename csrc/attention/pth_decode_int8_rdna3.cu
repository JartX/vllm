// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// HIP split-KV decode attention for INT8 per-token-head KV cache on RDNA3.
// Single-wave (32 threads), zero __syncthreads, pure __shfl_xor reduction.
// Mirrors the INT4 v3 kernel but without nibble unpack (1 byte = 1 value).

#include <cstdint>
#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#if defined(USE_ROCM)
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

// Templated load/store helpers for fp16 and bf16 query/output dtypes.
// Both __half and __hip_bfloat16 are implicitly convertible to/from float.
template <typename T> __device__ __forceinline__ float to_float(T x) {
  return (float)x;
}
template <> __device__ __forceinline__ float to_float<half>(half x) {
  return __half2float(x);
}
template <typename T> __device__ __forceinline__ T from_float(float x) {
  return (T)x;
}
template <> __device__ __forceinline__ half from_float<half>(float x) {
  return __float2half(x);
}

// ---------------------------------------------------------------------------
// Stage 1 v3: 32 threads (1 wave), 8 dims/thread. ZERO __syncthreads.
// K/V are int8 with per-token-head float32 scale (symmetric, no zp).
// Grid: (num_q, num_q_heads, NUM_SPLITS), Block: 32 threads.
// ---------------------------------------------------------------------------

template <int HEAD_SIZE, typename QT>
__global__ void decode_int8_stage1_v3(
    const QT* __restrict__ Q,
    const int8_t* __restrict__ K_cache,
    const int8_t* __restrict__ V_cache,
    const float* __restrict__ K_scale,
    const float* __restrict__ V_scale,
    const int* __restrict__ block_table,
    const int* __restrict__ q_to_req,
    const int* __restrict__ q_to_klen,
    float* __restrict__ mid_o,
    float sm_scale,
    int num_q_heads, int num_kv_heads, int block_size, int max_blocks,
    int num_splits,
    int64_t sq0, int64_t sq1,
    int64_t skb, int64_t sks, int64_t skh,
    int64_t svb, int64_t svs, int64_t svh,
    int64_t ssb, int64_t sss, int64_t ssh,
    int64_t svsb, int64_t svss, int64_t svsh,
    int64_t smo, int64_t smh, int64_t sms) {

  constexpr int DIMS_PER_THREAD = HEAD_SIZE / 32;  // 8 for HS=256
  constexpr int BYTES_PER_THREAD = DIMS_PER_THREAD; // 1 byte per dim for int8

  const int qi = blockIdx.x;
  const int hi = blockIdx.y;
  const int si = blockIdx.z;
  const int tid = threadIdx.x;  // 0..31

  const int req = q_to_req[qi];
  // Defensive clamp on kv_len. A garbage value here (e.g. q_to_klen not yet
  // landed from an async H2D copy, or a stale persistent buffer) would make the
  // KV loop below run for billions of iterations (GPU hang -> 300s client
  // timeout) or index block_table far out of bounds (page fault). Bound it to
  // the allocated KV capacity; for a valid kv_len this is a no-op.
  int kv_len = q_to_klen[qi];
  const int max_kv = max_blocks * block_size;
  if (kv_len < 0) kv_len = 0;
  if (kv_len > max_kv) kv_len = max_kv;
  const int kvh = hi / (num_q_heads / num_kv_heads);

  const int tps = (kv_len + num_splits - 1) / num_splits;
  const int start = si * tps;
  const int end = min(start + tps, kv_len);
  if (start >= kv_len) {
    float* out_ptr = mid_o + qi * smo + hi * smh + si * sms;
    for (int d = 0; d < DIMS_PER_THREAD; ++d)
      out_ptr[tid * DIMS_PER_THREAD + d] = 0.0f;
    if (tid == 0) {
      out_ptr[HEAD_SIZE] = -INFINITY;
      out_ptr[HEAD_SIZE + 1] = 0.0f;
    }
    return;
  }

  // Load Q values for this thread's dims
  const QT* qr = Q + qi * sq0 + hi * sq1;
  float q_vals[DIMS_PER_THREAD];
  #pragma unroll
  for (int d = 0; d < DIMS_PER_THREAD; ++d)
    q_vals[d] = to_float<QT>(qr[tid * DIMS_PER_THREAD + d]);

  // Pre-scale to log2 space for exp2f
  const float sm_scale_log2 = sm_scale * 1.4426950408889634f;

  float m_state = -INFINITY;
  float l_state = 0.0f;
  float o_vals[DIMS_PER_THREAD] = {};

  int prev_lb = -1;
  int pb = 0;
  for (int kv = start; kv < end; ++kv) {
    const int lb = kv / block_size;
    const int slot = kv - lb * block_size;
    if (lb != prev_lb) { pb = block_table[req * max_blocks + lb]; prev_lb = lb; }

    // Vectorized load: 2 dwords = 8 int8 values for K
    // (8 bytes per thread, loaded as two uint32)
    const int8_t* k_base = K_cache + pb * skb + slot * sks + kvh * skh
                           + tid * BYTES_PER_THREAD;
    float partial = 0.0f;
    #pragma unroll
    for (int d = 0; d < DIMS_PER_THREAD; ++d)
      partial += q_vals[d] * (float)k_base[d];

    // Wave-wide reduction (no sync — single wave)
    #pragma unroll
    for (int s = 16; s > 0; s >>= 1)
      partial += __shfl_xor(partial, s);

    float k_sc = K_scale[pb * ssb + slot * sss + kvh * ssh];
    float score = partial * k_sc * sm_scale_log2;

    // Online softmax — branchless, log2 space
    float m_new = fmaxf(m_state, score);
    float alpha = exp2f(m_state - m_new);
    float p = exp2f(score - m_new);
    l_state = l_state * alpha + p;
    #pragma unroll
    for (int d = 0; d < DIMS_PER_THREAD; ++d)
      o_vals[d] *= alpha;
    m_state = m_new;

    // V accumulation
    const int8_t* v_base = V_cache + pb * svb + slot * svs + kvh * svh
                           + tid * BYTES_PER_THREAD;
    float v_sc = V_scale[pb * svsb + slot * svss + kvh * svsh];
    float p_vs = p * v_sc;
    #pragma unroll
    for (int d = 0; d < DIMS_PER_THREAD; ++d)
      o_vals[d] += p_vs * (float)v_base[d];
  }

  // Write output
  float* out_ptr = mid_o + qi * smo + hi * smh + si * sms;
  #pragma unroll
  for (int d = 0; d < DIMS_PER_THREAD; ++d)
    out_ptr[tid * DIMS_PER_THREAD + d] = o_vals[d];
  if (tid == 0) {
    out_ptr[HEAD_SIZE] = m_state;
    out_ptr[HEAD_SIZE + 1] = l_state;
  }
}

// ---------------------------------------------------------------------------
// Stage 2: reduce across splits → final output.
// 128 threads, 2 dims per thread (reuses v2 reduce pattern).
// ---------------------------------------------------------------------------

template <int HEAD_SIZE, typename OT>
__global__ void decode_int8_reduce(
    const float* __restrict__ mid_o,
    OT* __restrict__ out,
    int num_splits,
    int64_t smo, int64_t smh, int64_t sms,
    int64_t soo, int64_t soh) {

  const int qi = blockIdx.x;
  const int hi = blockIdx.y;
  const int tid = threadIdx.x;  // 0..HEAD_SIZE/2-1

  float m_global = -INFINITY;
  for (int s = 0; s < num_splits; ++s) {
    float ms = mid_o[qi * smo + hi * smh + s * sms + HEAD_SIZE];
    m_global = fmaxf(m_global, ms);
  }

  float o0 = 0.0f, o1 = 0.0f, l_global = 0.0f;
  for (int s = 0; s < num_splits; ++s) {
    const float* sp = mid_o + qi * smo + hi * smh + s * sms;
    float ms = sp[HEAD_SIZE];
    float ls = sp[HEAD_SIZE + 1];
    // ms / m_global are stored in log2 space (stage1 pre-scaled by log2(e)),
    // so exp2(ms - m_global) is the correct rescaling factor. Multiplying the
    // delta by log2(e) again would compute e^delta instead of 2^delta and
    // systematically underweight non-max splits, drifting attention toward the
    // max-only split and inducing repetition loops over long generations.
    float a = (ms == -INFINITY) ? 0.0f : exp2f(ms - m_global);
    o0 += sp[2 * tid] * a;
    o1 += sp[2 * tid + 1] * a;
    l_global += ls * a;
  }
  float inv_l = 1.0f / (l_global + 1e-10f);
  out[qi * soo + hi * soh + 2 * tid] = from_float<OT>(o0 * inv_l);
  out[qi * soo + hi * soh + 2 * tid + 1] = from_float<OT>(o1 * inv_l);
}

// ---------------------------------------------------------------------------
// Launcher
// ---------------------------------------------------------------------------

void pth_decode_int8_rdna3(
    torch::Tensor out,
    torch::Tensor query,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor k_scale_cache,
    torch::Tensor v_scale_cache,
    torch::Tensor block_table,
    torch::Tensor q_to_req,
    torch::Tensor q_to_klen,
    torch::Tensor mid_o_buf,
    double sm_scale,
    int64_t num_kv_splits) {

  const int num_q = query.size(0);
  const int num_q_heads = query.size(1);
  const int head_size = query.size(2);
  const int num_kv_heads = k_scale_cache.size(2);
  const int block_size = key_cache.size(1);
  const int max_blocks = block_table.size(1);
  auto stream = at::cuda::getCurrentCUDAStream().stream();

  TORCH_CHECK(head_size == 256,
              "pth_decode_int8_rdna3: only head_size=256 supported, got ", head_size);
  TORCH_CHECK(query.dtype() == at::kHalf || query.dtype() == at::kBFloat16,
              "pth_decode_int8_rdna3: query must be fp16 or bf16");
  TORCH_CHECK(out.dtype() == query.dtype(),
              "pth_decode_int8_rdna3: out must match query dtype");
  TORCH_CHECK(key_cache.dtype() == at::kChar);  // int8

  int ns = (int)num_kv_splits;
  constexpr int HS = 256;
  constexpr int TH = HS / 2;

  dim3 grid1(num_q, num_q_heads, ns);
  dim3 grid2(num_q, num_q_heads);

#define LAUNCH_INT8_V3(QT, OT)                                                 \
  decode_int8_stage1_v3<HS, QT><<<grid1, dim3(32), 0, stream>>>(               \
      (const QT*)query.data_ptr(),                                             \
      (const int8_t*)key_cache.data_ptr(),                                     \
      (const int8_t*)value_cache.data_ptr(),                                   \
      (const float*)k_scale_cache.data_ptr(),                                  \
      (const float*)v_scale_cache.data_ptr(),                                  \
      (const int*)block_table.data_ptr(),                                      \
      (const int*)q_to_req.data_ptr(),                                         \
      (const int*)q_to_klen.data_ptr(),                                        \
      (float*)mid_o_buf.data_ptr(),                                            \
      (float)sm_scale,                                                         \
      num_q_heads, num_kv_heads, block_size, max_blocks, ns,                   \
      query.stride(0), query.stride(1),                                        \
      key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),           \
      value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),     \
      k_scale_cache.stride(0), k_scale_cache.stride(1), k_scale_cache.stride(2),\
      v_scale_cache.stride(0), v_scale_cache.stride(1), v_scale_cache.stride(2),\
      mid_o_buf.stride(0), mid_o_buf.stride(1), mid_o_buf.stride(2));          \
  decode_int8_reduce<HS, OT><<<grid2, dim3(TH), 0, stream>>>(                  \
      (const float*)mid_o_buf.data_ptr(),                                      \
      (OT*)out.data_ptr(),                                                     \
      ns,                                                                      \
      mid_o_buf.stride(0), mid_o_buf.stride(1), mid_o_buf.stride(2),           \
      out.stride(0), out.stride(1));

  if (query.dtype() == at::kHalf) {
    LAUNCH_INT8_V3(half, half);
  } else {
    LAUNCH_INT8_V3(__hip_bfloat16, __hip_bfloat16);
  }
#undef LAUNCH_INT8_V3
}

#else
void pth_decode_int8_rdna3(
    torch::Tensor out, torch::Tensor query,
    torch::Tensor key_cache, torch::Tensor value_cache,
    torch::Tensor k_scale_cache, torch::Tensor v_scale_cache,
    torch::Tensor block_table, torch::Tensor q_to_req,
    torch::Tensor q_to_klen, torch::Tensor mid_o_buf,
    double sm_scale, int64_t num_kv_splits) {
  TORCH_CHECK(false, "pth_decode_int8_rdna3 requires ROCm");
}
#endif
