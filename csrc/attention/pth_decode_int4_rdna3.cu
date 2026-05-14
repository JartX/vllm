// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// HIP split-KV decode attention for INT4 per-token-head KV cache on RDNA3.
// Replaces the Triton packed split-KV kernel to eliminate ~40 µs/call of
// Python/JIT dispatch overhead. Fuses RHT butterfly on Q + inverse on output.
//
// Stage 1: each block handles one (query, head, kv_split) — scalar dot
//          product with wave reduction, online softmax, V accumulation.
// Stage 2: reduce partial outputs across splits.

#include <cstdint>
#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#if defined(USE_ROCM)
#include <hip/hip_runtime.h>

// ---------------------------------------------------------------------------
// Stage 1: per-split attention
// Grid: (num_q, num_q_heads, NUM_SPLITS)
// Block: HEAD_SIZE threads. Each thread owns one head dimension.
// ---------------------------------------------------------------------------

template <int HEAD_SIZE, int BLOCK_KV>
__global__ void decode_int4_stage1(
    const half* __restrict__ Q,          // [num_q, num_q_heads, HEAD_SIZE]
    const uint8_t* __restrict__ K_cache, // [blocks, block_size, kv_heads, HEAD_SIZE/2]
    const uint8_t* __restrict__ V_cache,
    const float* __restrict__ K_scale,   // [blocks, block_size, kv_heads] stego
    const float* __restrict__ V_scale,
    const float* __restrict__ rht_signs, // [HEAD_SIZE] D₁ signs
    const int* __restrict__ block_table, // [num_reqs, max_blocks]
    const int* __restrict__ q_to_req,
    const int* __restrict__ q_to_klen,
    float* __restrict__ mid_o,           // [num_q, num_q_heads, NUM_SPLITS, HEAD_SIZE+2]
    float sm_scale,
    int num_q_heads, int num_kv_heads, int block_size, int max_blocks,
    int num_splits,
    int64_t sq0, int64_t sq1,
    int64_t skb, int64_t sks, int64_t skh,
    int64_t svb, int64_t svs, int64_t svh,
    int64_t ssb, int64_t sss, int64_t ssh,
    int64_t svsb, int64_t svss, int64_t svsh,
    int64_t smo, int64_t smh, int64_t sms) {

  const int qi = blockIdx.x;
  const int hi = blockIdx.y;
  const int si = blockIdx.z;
  const int tid = threadIdx.x;  // 0..HEAD_SIZE-1

  const int req = q_to_req[qi];
  const int kv_len = q_to_klen[qi];
  const int kvh = hi / (num_q_heads / num_kv_heads);

  // Split range
  const int tps = (kv_len + num_splits - 1) / num_splits;
  const int start = si * tps;
  const int end = min(start + tps, kv_len);
  if (start >= kv_len) {
    // Empty split — write zeros
    mid_o[qi * smo + hi * smh + si * sms + tid] = 0.0f;
    if (tid == 0) {
      mid_o[qi * smo + hi * smh + si * sms + HEAD_SIZE] = -INFINITY;
      mid_o[qi * smo + hi * smh + si * sms + HEAD_SIZE + 1] = 0.0f;
    }
    return;
  }

  // ---- Load Q + RHT butterfly ----
  __shared__ float qs[HEAD_SIZE];
  float qv = __half2float(Q[qi * sq0 + hi * sq1 + tid]);
  // D₁ sign flip
  qs[tid] = qv * rht_signs[tid];
  __syncthreads();
  // Parallel Hadamard butterfly
  #pragma unroll
  for (int h = 1; h < HEAD_SIZE; h *= 2) {
    int partner = tid ^ h;
    float mine = qs[tid], other = qs[partner];
    __syncthreads();
    qs[tid] = (tid & h) ? (other - mine) : (mine + other);
    __syncthreads();
  }
  float q_rot = qs[tid];

  // ---- Online softmax + V accumulation ----
  __shared__ float dot_lds[4];  // partial sums from 4 waves
  const int wave_id = tid / 32;
  const int lane = tid & 31;

  float m_state = -INFINITY;
  float l_state = 0.0f;
  float o_acc = 0.0f;

  for (int kv = start; kv < end; ++kv) {
    const int lb = kv / block_size;
    const int slot = kv - lb * block_size;
    const int pb = block_table[req * max_blocks + lb];

    // Load K nibble for my dimension
    const int packed_idx = tid / 2;
    uint8_t k_byte = K_cache[pb * skb + slot * sks + kvh * skh + packed_idx];
    int k_nib = (tid & 1) ? ((k_byte >> 4) & 0xF) : (k_byte & 0xF);

    // Extract scale + zp from steganographed float
    float k_raw = K_scale[pb * ssb + slot * sss + kvh * ssh];
    int k_bits; __builtin_memcpy(&k_bits, &k_raw, 4);
    int k_zp = k_bits & 0xF;
    int k_sb = k_bits & ~0xF;
    float k_sc; __builtin_memcpy(&k_sc, &k_sb, 4);

    float k_val = (float)(k_nib - k_zp);

    // Dot product: Q_rot[d] * K[d] summed over all d
    float partial = q_rot * k_val;
    // Wave-level reduction (32 lanes)
    partial += __shfl_xor(partial, 1);
    partial += __shfl_xor(partial, 2);
    partial += __shfl_xor(partial, 4);
    partial += __shfl_xor(partial, 8);
    partial += __shfl_xor(partial, 16);
    // Lane 0 of each wave has sum of 32 dims
    if (lane == 0) dot_lds[wave_id] = partial;
    __syncthreads();

    // Combine 4 waves (only need first 4 threads)
    float score = 0;
    if (tid < 4) score = dot_lds[tid];
    if (tid < 4) {
      score += __shfl_xor(score, 1);
      score += __shfl_xor(score, 2);
    }
    // Broadcast score to all threads via LDS
    if (tid == 0) dot_lds[0] = score * k_sc * sm_scale;
    __syncthreads();
    score = dot_lds[0];

    // Online softmax
    float m_new = fmaxf(m_state, score);
    float alpha = (m_state == -INFINITY) ? 0.0f : __expf(m_state - m_new);
    float p = (m_new == -INFINITY) ? 0.0f : __expf(score - m_new);
    l_state = l_state * alpha + p;
    o_acc = o_acc * alpha;
    m_state = m_new;

    // Load V nibble and accumulate
    uint8_t v_byte = V_cache[pb * svb + slot * svs + kvh * svh + packed_idx];
    int v_nib = (tid & 1) ? ((v_byte >> 4) & 0xF) : (v_byte & 0xF);
    float v_raw = V_scale[pb * svsb + slot * svss + kvh * svsh];
    int v_bits; __builtin_memcpy(&v_bits, &v_raw, 4);
    int v_zp = v_bits & 0xF;
    int v_sb = v_bits & ~0xF;
    float v_sc; __builtin_memcpy(&v_sc, &v_sb, 4);
    float v_val = (float)(v_nib - v_zp) * v_sc;

    o_acc += p * v_val;
  }

  // Store partial output + (m, l) for reduce stage
  float* out_ptr = mid_o + qi * smo + hi * smh + si * sms;
  out_ptr[tid] = o_acc;
  if (tid == 0) {
    out_ptr[HEAD_SIZE] = m_state;
    out_ptr[HEAD_SIZE + 1] = l_state;
  }
}

// ---------------------------------------------------------------------------
// Stage 2: reduce across splits → final output + inverse RHT
// Grid: (num_q, num_q_heads)
// Block: HEAD_SIZE threads
// ---------------------------------------------------------------------------

template <int HEAD_SIZE>
__global__ void decode_int4_reduce(
    const float* __restrict__ mid_o,     // [num_q, num_q_heads, NUM_SPLITS, HEAD_SIZE+2]
    half* __restrict__ out,              // [num_q, num_q_heads, HEAD_SIZE]
    const float* __restrict__ rht_signs,
    int num_splits, float inv_head_size,
    int64_t smo, int64_t smh, int64_t sms,
    int64_t soo, int64_t soh) {

  const int qi = blockIdx.x;
  const int hi = blockIdx.y;
  const int tid = threadIdx.x;

  // Find global max across splits
  float m_global = -INFINITY;
  for (int s = 0; s < num_splits; ++s) {
    float ms = mid_o[qi * smo + hi * smh + s * sms + HEAD_SIZE];
    m_global = fmaxf(m_global, ms);
  }

  // Combine outputs with rescaling
  float o_final = 0.0f;
  float l_global = 0.0f;
  for (int s = 0; s < num_splits; ++s) {
    const float* sp = mid_o + qi * smo + hi * smh + s * sms;
    float ms = sp[HEAD_SIZE];
    float ls = sp[HEAD_SIZE + 1];
    float os = sp[tid];
    float alpha = (ms == -INFINITY) ? 0.0f : __expf(ms - m_global);
    o_final += os * alpha;
    l_global += ls * alpha;
  }
  o_final /= (l_global + 1e-10f);

  // ---- Inverse RHT butterfly + scale ----
  __shared__ float rs[HEAD_SIZE];
  rs[tid] = o_final;
  __syncthreads();

  #pragma unroll
  for (int h = 1; h < HEAD_SIZE; h *= 2) {
    int partner = tid ^ h;
    float mine = rs[tid], other = rs[partner];
    __syncthreads();
    rs[tid] = (tid & h) ? (other - mine) : (mine + other);
    __syncthreads();
  }

  // D₁ sign flip + /head_size
  out[qi * soo + hi * soh + tid] =
      __float2half(rs[tid] * rht_signs[tid] * inv_head_size);
}

// ---------------------------------------------------------------------------
// Launcher
// ---------------------------------------------------------------------------

void pth_decode_int4_rdna3(
    torch::Tensor out,        // [num_q, num_q_heads, HEAD_SIZE]
    torch::Tensor query,      // [num_q, num_q_heads, HEAD_SIZE]
    torch::Tensor key_cache,  // [blocks, block_size, kv_heads, HEAD_SIZE/2]
    torch::Tensor value_cache,
    torch::Tensor k_scale_cache,
    torch::Tensor v_scale_cache,
    torch::Tensor rht_signs,  // [HEAD_SIZE]
    torch::Tensor block_table,
    torch::Tensor q_to_req,
    torch::Tensor q_to_klen,
    torch::Tensor mid_o_buf,  // [num_q, num_q_heads, NUM_SPLITS, HEAD_SIZE+2]
    double sm_scale,
    int64_t num_kv_splits) {

  const int num_q = query.size(0);
  const int num_q_heads = query.size(1);
  const int head_size = query.size(2);
  const int num_kv_heads = k_scale_cache.size(2);
  const int block_size = key_cache.size(1);
  const int max_blocks = block_table.size(1);
  auto stream = at::cuda::getCurrentCUDAStream().stream();

  TORCH_CHECK(head_size == 128);
  TORCH_CHECK(query.dtype() == at::kHalf);

  constexpr int HS = 128;
  constexpr int BKV = 1;  // process 1 KV token per iteration (simple)
  int ns = (int)num_kv_splits;

  // Ensure mid_o is big enough
  TORCH_CHECK(mid_o_buf.size(2) >= ns);
  TORCH_CHECK(mid_o_buf.size(3) >= HS + 2);

  // Stage 1
  dim3 grid1(num_q, num_q_heads, ns);
  decode_int4_stage1<HS, BKV><<<grid1, dim3(HS), 0, stream>>>(
      (const half*)query.data_ptr(),
      (const uint8_t*)key_cache.data_ptr(),
      (const uint8_t*)value_cache.data_ptr(),
      (const float*)k_scale_cache.data_ptr(),
      (const float*)v_scale_cache.data_ptr(),
      (const float*)rht_signs.data_ptr(),
      (const int*)block_table.data_ptr(),
      (const int*)q_to_req.data_ptr(),
      (const int*)q_to_klen.data_ptr(),
      (float*)mid_o_buf.data_ptr(),
      (float)sm_scale,
      num_q_heads, num_kv_heads, block_size, max_blocks, ns,
      query.stride(0), query.stride(1),
      key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
      value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),
      k_scale_cache.stride(0), k_scale_cache.stride(1), k_scale_cache.stride(2),
      v_scale_cache.stride(0), v_scale_cache.stride(1), v_scale_cache.stride(2),
      mid_o_buf.stride(0), mid_o_buf.stride(1), mid_o_buf.stride(2));

  // Stage 2: reduce + inverse RHT
  dim3 grid2(num_q, num_q_heads);
  constexpr float inv_hs = 1.0f / (float)HS;
  decode_int4_reduce<HS><<<grid2, dim3(HS), 0, stream>>>(
      (const float*)mid_o_buf.data_ptr(),
      (half*)out.data_ptr(),
      (const float*)rht_signs.data_ptr(),
      ns, inv_hs,
      mid_o_buf.stride(0), mid_o_buf.stride(1), mid_o_buf.stride(2),
      out.stride(0), out.stride(1));
}

#else
void pth_decode_int4_rdna3(
    torch::Tensor out, torch::Tensor query,
    torch::Tensor key_cache, torch::Tensor value_cache,
    torch::Tensor k_scale_cache, torch::Tensor v_scale_cache,
    torch::Tensor rht_signs, torch::Tensor block_table,
    torch::Tensor q_to_req, torch::Tensor q_to_klen,
    torch::Tensor mid_o_buf, double sm_scale, int64_t num_kv_splits) {
  TORCH_CHECK(false, "requires ROCm");
}
#endif
