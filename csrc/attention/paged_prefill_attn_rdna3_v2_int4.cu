// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Paged prefill attention v2 INT4 per-token-head kernel for AMD RDNA3
// (gfx1100). Extends v2 (4 waves, BLOCK_M=64) for INT4 KV cache with
// per-token-head asymmetric dequantization (zero-point + scale).
//
// KEY OPTIMIZATION: Uses v_wmma_i32_16x16x16_iu8 for Q×K matmul.
// - Q is quantized to int8 per-row on-the-fly (amortized over all K tiles).
// - K nibbles (0..15) stored as uint8 in LDS (no fp16 conversion needed!).
// - i32 result rescaled: S = (i32_dot - Q_int_sum * zp) * q_scale * k_scale * sm
// - Eliminates ALL nibble→fp16 conversions in the K path.
//
// For P×V (second matmul): V nibbles dequanted to fp16, bf16 WMMA used
// (P is softmax probability in fp16, no int8 quantization viable).
//
// Layout:
//   K/V cache (uint8): [num_blocks, block_size, num_kv_heads, head_size//2]
//     (nibble-packed: low nibble = even index, high nibble = odd index)
//   k_scale_cache (fp32): [num_blocks, block_size, num_kv_heads]
//     (bits 0-3 = zero-point, bits 4-31 = float32 scale)
//   v_scale_cache (fp32): same layout

#include <cstdint>

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include "paged_prefill_attn_rdna3.cuh"

namespace vllm {
namespace prefill_attn_rdna3_v2_int4 {

#if defined(USE_ROCM)

using vllm::prefill_attn_rdna3::bf16_t;
using vllm::prefill_attn_rdna3::bitcast_elem;
using vllm::prefill_attn_rdna3::to_T;
using vllm::prefill_attn_rdna3::to_f;
using vllm::prefill_attn_rdna3::v16bf16;
using vllm::prefill_attn_rdna3::v16fp16;
using vllm::prefill_attn_rdna3::v8fp32;
using vllm::prefill_attn_rdna3::v8i32;
using vllm::prefill_attn_rdna3::v16i8;
using vllm::prefill_attn_rdna3::v16u8;
using vllm::prefill_attn_rdna3::wmma_mma;
using vllm::prefill_attn_rdna3::wmma_mma_iu8;
using vllm::prefill_attn_rdna3::WmmaNative;

constexpr int K_TILE = 16;
constexpr int NUM_WAVES = 4;
constexpr int M_PER_WAVE = 16;
constexpr int BLOCK_M_V2 = NUM_WAVES * M_PER_WAVE;  // = 64
constexpr int THREADS = NUM_WAVES * 32;             // = 128

__device__ __forceinline__ float wave16_max(float v) {
  v = fmaxf(v, __shfl_xor(v, 1));
  v = fmaxf(v, __shfl_xor(v, 2));
  v = fmaxf(v, __shfl_xor(v, 4));
  v = fmaxf(v, __shfl_xor(v, 8));
  return v;
}

__device__ __forceinline__ float wave16_sum(float v) {
  v += __shfl_xor(v, 1);
  v += __shfl_xor(v, 2);
  v += __shfl_xor(v, 4);
  v += __shfl_xor(v, 8);
  return v;
}

// ---------------------------------------------------------------------------
// Extract scale and zero-point from steganographed float32.
// Bits 0-3: 4-bit zero-point (unsigned, 0..15)
// Bits 4-31: float32 scale (with low 4 mantissa bits cleared)
// ---------------------------------------------------------------------------
struct ScaleZP {
  float scale;
  float zp;
};

__device__ __forceinline__ ScaleZP extract_scale_zp(float raw) {
  int bits;
  __builtin_memcpy(&bits, &raw, 4);
  int zp_int = bits & 0xF;
  int scale_bits = bits & ~0xF;
  float scale;
  __builtin_memcpy(&scale, &scale_bits, 4);
  return {scale, (float)zp_int};
}

// ---------------------------------------------------------------------------
// INT4 K cache load into LDS as UINT8 (for v_wmma_i32_16x16x16_iu8).
// K cache layout: [blocks, slots, heads, dim//2] (nibble-packed).
//
// Strategy: 128 threads, slot=tid/8, d_chunk=tid%8. Each chunk = 8 packed
// bytes = 16 nibble values → stored as 16 uint8 in K_lds_u8.
//
// K_lds_u8 layout: [D_HIGH=8][K_TILE=16][16] where 16 is the WMMA K-dim.
// Same as K_lds_raw in bf16 kernel but uint8 per element (half the LDS!).
// ---------------------------------------------------------------------------

template <int HEAD_SIZE>
__device__ __forceinline__ void load_k_tile_paged_int4_coop(
    uint8_t* __restrict__ K_lds_u8,  // [D_HIGH][K_TILE][16]
    const uint8_t* __restrict__ k_cache_u8,
    const float* __restrict__ k_scale_cache,
    const int* __restrict__ block_table,
    int seq_idx, int kv_head_idx, int start_n, int seq_ctx_len,
    int block_size, int max_blocks_per_seq,
    int64_t stride_kc_block, int64_t stride_kc_slot, int64_t stride_kc_head,
    int64_t stride_ks_blk, int64_t stride_ks_slot, int64_t stride_ks_head,
    float* __restrict__ k_scale_lds,  // [K_TILE] scales
    float* __restrict__ k_zp_lds,     // [K_TILE] zero-points
    int tid) {
  constexpr int BYTES_PER_CHUNK = 8;  // 16 nibbles packed in 8 bytes
  constexpr int D_HIGH [[maybe_unused]] = HEAD_SIZE / 16;  // = 8

  // 128 threads: slot = tid / 8 (0..15), d_chunk = tid % 8 (0..7)
  const int my_k_idx = tid >> 3;   // slot within tile
  const int my_dh = tid & 7;       // dim chunk

  const int abs_k = start_n + my_k_idx;
  const bool valid_k = abs_k < seq_ctx_len;
  const int log_block = abs_k / block_size;
  const int slot = abs_k - log_block * block_size;
  const int p_block =
      valid_k ? block_table[seq_idx * max_blocks_per_seq + log_block] : 0;

  // Load scale + zp (thread with d_chunk==0 loads it)
  if (my_dh == 0 && valid_k) {
    float raw = k_scale_cache[p_block * stride_ks_blk +
                              slot * stride_ks_slot +
                              kv_head_idx * stride_ks_head];
    ScaleZP sz = extract_scale_zp(raw);
    k_scale_lds[my_k_idx] = sz.scale;
    k_zp_lds[my_k_idx] = sz.zp;
  } else if (my_dh == 0 && !valid_k) {
    k_scale_lds[my_k_idx] = 0.0f;
    k_zp_lds[my_k_idx] = 0.0f;
  }

  // Load 8 packed bytes = 16 nibble values
  const int d_base_packed = my_dh * BYTES_PER_CHUNK;
  const uint8_t* src = k_cache_u8 +
                       (int64_t)p_block * stride_kc_block +
                       (int64_t)slot * stride_kc_slot +
                       (int64_t)kv_head_idx * stride_kc_head +
                       (int64_t)d_base_packed;

  uint8_t k_packed[BYTES_PER_CHUNK];
  if (valid_k) {
    *(uint64_t*)k_packed = *(const uint64_t*)src;
  } else {
    #pragma unroll
    for (int i = 0; i < BYTES_PER_CHUNK; ++i) k_packed[i] = 0;
  }

  // Unpack nibbles to uint8 [0..15] and store to K_lds_u8[dh][k_idx][0..15]
  uint8_t unpacked[16];
  #pragma unroll
  for (int i = 0; i < BYTES_PER_CHUNK; ++i) {
    unpacked[2 * i] = k_packed[i] & 0xF;
    unpacked[2 * i + 1] = (k_packed[i] >> 4) & 0xF;
  }

  // Store: K_lds_u8[my_dh][my_k_idx][0..15] = unpacked[0..15]
  // Layout: contiguous 16 uint8 per (dh, k_idx) entry for WMMA B-fragment load
  *(int4*)&K_lds_u8[my_dh * (K_TILE * 16) + my_k_idx * 16] = *(int4*)unpacked;
}

// ---------------------------------------------------------------------------
// INT4 V cache load + unpack nibbles to LDS as FP16 (for bf16 WMMA on PV).
// Same packed layout as K. V_lds target: [HEAD_SIZE][K_TILE] (transpose).
// ---------------------------------------------------------------------------

template <typename T, int HEAD_SIZE>
__device__ __forceinline__ void load_v_tile_paged_int4_coop(
    T* __restrict__ V_lds,
    const uint8_t* __restrict__ v_cache_u8,
    const float* __restrict__ v_scale_cache,
    const int* __restrict__ block_table,
    int seq_idx, int kv_head_idx, int start_n, int seq_ctx_len,
    int block_size, int max_blocks_per_seq,
    int64_t stride_vc_block, int64_t stride_vc_slot, int64_t stride_vc_head,
    int64_t stride_vs_blk, int64_t stride_vs_slot, int64_t stride_vs_head,
    float* __restrict__ v_scale_lds,  // [K_TILE] v_scales
    float* __restrict__ v_zp_lds,     // [K_TILE] v_zero_points
    int tid) {
  constexpr int BYTES_PER_CHUNK = 8;

  const int log_block = start_n / block_size;
  const int slot_base = start_n - log_block * block_size;
  const int p_block = block_table[seq_idx * max_blocks_per_seq + log_block];
  const int valid_k_count = max(0, min(K_TILE, seq_ctx_len - start_n));

  // Load V scales+zp for this tile (threads 0..15)
  if (tid < K_TILE) {
    const int slot = slot_base + tid;
    if (tid < valid_k_count) {
      float raw = v_scale_cache[p_block * stride_vs_blk +
                                slot * stride_vs_slot +
                                kv_head_idx * stride_vs_head];
      ScaleZP sz = extract_scale_zp(raw);
      v_scale_lds[tid] = sz.scale;
      v_zp_lds[tid] = sz.zp;
    } else {
      v_scale_lds[tid] = 0.0f;
      v_zp_lds[tid] = 0.0f;
    }
  }

  // 128 threads: slot = tid / 8, d_chunk = tid % 8
  const int my_slot_offset = tid >> 3;
  const int my_d_chunk = tid & 7;
  const int d_base_packed = my_d_chunk * BYTES_PER_CHUNK;
  const int abs_slot = slot_base + my_slot_offset;
  const bool valid = my_slot_offset < valid_k_count;

  uint8_t v_packed[BYTES_PER_CHUNK];
  if (valid) {
    const uint8_t* src = v_cache_u8 +
                         (int64_t)p_block * stride_vc_block +
                         (int64_t)abs_slot * stride_vc_slot +
                         (int64_t)kv_head_idx * stride_vc_head +
                         (int64_t)d_base_packed;
    *(uint64_t*)v_packed = *(const uint64_t*)src;
  } else {
    #pragma unroll
    for (int i = 0; i < BYTES_PER_CHUNK; ++i) v_packed[i] = 0;
  }

  // Unpack to fp16 and transpose-store to V_lds[d][k]
  const int d_base = my_d_chunk * 16;
  #pragma unroll
  for (int i = 0; i < BYTES_PER_CHUNK; ++i) {
    uint8_t byte = v_packed[i];
    uint8_t lo = byte & 0xF;
    uint8_t hi = (byte >> 4) & 0xF;
    V_lds[(d_base + 2 * i) * K_TILE + my_slot_offset] = to_T<T>((float)lo);
    V_lds[(d_base + 2 * i + 1) * K_TILE + my_slot_offset] = to_T<T>((float)hi);
  }
}

// ---------------------------------------------------------------------------
// attn_step for INT4 per-token-head using INT8 WMMA for Q×K.
//
// Q×K path:
//   Uses v_wmma_i32_16x16x16_iu8 with Q in int8 (pre-quantized) and
//   K in uint8 (raw nibbles). i32 result rescaled post-matmul:
//   S[m][n] = (i32_dot[m][n] - Q_int_sum[m] * k_zp[n])
//             * q_scale[m] * k_scale[n] * sm_scale
//
// P×V path:
//   Uses bf16 WMMA (same as INT8 kernel). V nibbles dequanted to fp16.
//   P_scaled = P * v_scale, then: O -= Pv_zp_sum (per-row correction).
// ---------------------------------------------------------------------------

template <typename T, int HEAD_SIZE, int X, bool CAUSAL_MASK>
__device__ __forceinline__ void attn_step_wave_int4(
    const uint8_t* __restrict__ K_lds_u8,  // [D_HIGH][K_TILE][16] uint8
    const T* __restrict__ V_lds,           // [HEAD_SIZE][K_TILE] fp16
    T* __restrict__ P_lds_wave,
    const float* __restrict__ k_scale_lds,
    const float* __restrict__ k_zp_lds,
    const float* __restrict__ v_scale_lds,
    const float* __restrict__ v_zp_lds,
    v16i8 (&q_int8_frags)[HEAD_SIZE / 16],  // Q quantized to int8
    typename WmmaNative<T>::v16 (&q_fp_frags)[HEAD_SIZE / 16],  // Q in fp16 (for PV)
    v8fp32 (&out_acc)[HEAD_SIZE / 16],
    float (&m_state)[8], float (&l_state)[8],
    float q_scale,       // per-row Q quantization scale
    float q_int_sum,     // sum of Q_int8 elements for this lane's row
    int wave_q_tile_start, int start_n, int valid_q_count, int valid_k_count,
    float sm_scale, int lane, int lane_lo, int lane_hi) {
  using V16 = typename WmmaNative<T>::v16;
  constexpr int FRAGS = HEAD_SIZE / 16;

  // ---- Q × K via INT8 WMMA (8 WMMAs into i32 accumulator) ----
  v8i32 s_i32 = {0, 0, 0, 0, 0, 0, 0, 0};
  #pragma unroll
  for (int dh = 0; dh < FRAGS; ++dh) {
    // Load K fragment: 16 uint8 values for this lane_lo (B col-major in N)
    v16u8 b_frag;
    *(int4*)&b_frag = *(const int4*)&K_lds_u8[dh * (K_TILE * 16) + lane_lo * 16];
    s_i32 = wmma_mma_iu8(q_int8_frags[dh], b_frag, s_i32);
  }

  // ---- Rescale i32 → float32 + zero-point correction + mask ----
  // S[m][n] = (i32_dot - Q_int_sum[m] * k_zp[n]) * q_scale[m] * k_scale[n] * sm
  const float k_sc = k_scale_lds[lane_lo];
  const float k_zp = k_zp_lds[lane_lo];
  const int abs_k = start_n + lane_lo;
  const bool k_in_seg = (lane_lo < valid_k_count);

  float s_acc[8];
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int m_row = 2 * i + lane_hi;
    const bool m_in_q = (m_row < valid_q_count);
    bool keep = m_in_q && k_in_seg;
    if constexpr (CAUSAL_MASK) {
      const int abs_q = wave_q_tile_start + m_row;
      keep = keep && (abs_k <= abs_q);
    }
    // Get Q_int_sum and q_scale for the actual M row via shuffle
    float qi_sum_row = __shfl(q_int_sum, m_row);
    float qs_row = __shfl(q_scale, m_row);
    // Correction: (i32_dot - Q_int_sum * zp) * q_scale * k_scale * sm_scale
    float raw = (float)s_i32[i] - qi_sum_row * k_zp;
    s_acc[i] = keep ? (raw * qs_row * k_sc * sm_scale) : -INFINITY;
  }

  // ---- Online softmax ----
  float m_ij[8], m_new[8], alpha[8], p_ij[8], l_ij[8];
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    m_ij[i] = wave16_max(s_acc[i]);
    m_new[i] = fmaxf(m_state[i], m_ij[i]);
    alpha[i] = (m_state[i] == -INFINITY) ? 0.0f : __expf(m_state[i] - m_new[i]);
    p_ij[i] = (m_new[i] == -INFINITY) ? 0.0f : __expf(s_acc[i] - m_new[i]);
    l_ij[i] = wave16_sum(p_ij[i]);
    l_state[i] = l_state[i] * alpha[i] + l_ij[i];
    m_state[i] = m_new[i];
  }
  #pragma unroll
  for (int dh = 0; dh < FRAGS; ++dh) {
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      out_acc[dh][i] *= alpha[i];
    }
  }

  // ---- Fuse v_scale into P + compute Pv_zp_sum ----
  const float v_sc = v_scale_lds[lane_lo];
  const float v_zp = v_zp_lds[lane_lo];
  float pv_zp_sum[8];
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    float p_scaled = p_ij[i] * v_sc;
    pv_zp_sum[i] = wave16_sum(p_scaled * v_zp);
    p_ij[i] = p_scaled;
  }

  // ---- Transpose P → p_frag via WAVE-LOCAL P_lds ----
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int m_row = 2 * i + lane_hi;
    P_lds_wave[m_row * K_TILE + lane_lo] = to_T<T>(p_ij[i]);
  }

  V16 p_frag;
  int4 p_lo = *(const int4*)&P_lds_wave[lane_lo * K_TILE + 0];
  int4 p_hi = *(const int4*)&P_lds_wave[lane_lo * K_TILE + 8];
  __builtin_memcpy(&p_frag, &p_lo, 16);
  __builtin_memcpy(((char*)&p_frag) + 16, &p_hi, 16);

  // ---- P × V via bf16 WMMA (V nibbles as fp16) + zp correction ----
  #pragma unroll
  for (int dh = 0; dh < FRAGS; ++dh) {
    V16 v_frag;
    int4 v_lo = *(const int4*)&V_lds[(dh * 16 + lane_lo) * K_TILE + 0];
    int4 v_hi = *(const int4*)&V_lds[(dh * 16 + lane_lo) * K_TILE + 8];
    __builtin_memcpy(&v_frag, &v_lo, 16);
    __builtin_memcpy(((char*)&v_frag) + 16, &v_hi, 16);
    out_acc[dh] = wmma_mma(p_frag, v_frag, out_acc[dh]);
  }

  // Subtract Pv_zp_sum (same per row, broadcast across all dims)
  #pragma unroll
  for (int dh = 0; dh < FRAGS; ++dh) {
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      out_acc[dh][i] -= pv_zp_sum[i];
    }
  }
}

// ---------------------------------------------------------------------------
// Main INT4 per-token-head kernel
// ---------------------------------------------------------------------------

template <typename T, int HEAD_SIZE>
__global__ void paged_prefill_attn_kernel_v2_int4(
    T* __restrict__ out,
    const T* __restrict__ q,
    const T* __restrict__ k_chunk,        // current chunk K (fp16/bf16, RHT-rotated)
    const T* __restrict__ v_chunk,        // current chunk V (fp16/bf16, RHT-rotated)
    const uint8_t* __restrict__ k_cache,  // paged K cache (packed nibbles)
    const uint8_t* __restrict__ v_cache,  // paged V cache (packed nibbles)
    const float* __restrict__ k_scale_cache,
    const float* __restrict__ v_scale_cache,
    const int* __restrict__ block_table,
    const int* __restrict__ cu_seqlens_q,
    const int* __restrict__ seq_lens,
    const int num_query_heads, const int num_kv_heads,
    const int block_size, const int max_blocks_per_seq,
    const float sm_scale, const bool causal,
    const int64_t stride_q_token, const int64_t stride_q_head,
    const int64_t stride_kc_token, const int64_t stride_kc_head,
    const int64_t stride_vc_token, const int64_t stride_vc_head,
    const int64_t stride_kcache_block, const int64_t stride_kcache_slot,
    const int64_t stride_kcache_head,
    const int64_t stride_vcache_block, const int64_t stride_vcache_slot,
    const int64_t stride_vcache_head,
    const int64_t stride_ks_blk, const int64_t stride_ks_slot,
    const int64_t stride_ks_head,
    const int64_t stride_vs_blk, const int64_t stride_vs_slot,
    const int64_t stride_vs_head,
    const int64_t stride_o_token, const int64_t stride_o_head) {
  using V16 = typename WmmaNative<T>::v16;
  using E = typename WmmaNative<T>::elem;
  constexpr int FRAGS = HEAD_SIZE / 16;
  constexpr int X_FP16 = 16 / sizeof(T);  // = 8

  const int seq_idx = blockIdx.x;
  const int head_idx = blockIdx.y;
  const int q_tile_idx = blockIdx.z;

  const int tid = threadIdx.x;
  const int wave_id = tid >> 5;
  const int lane = tid & 31;
  const int lane_lo = lane & 15;
  const int lane_hi = lane >> 4;

  const int q_start_token = cu_seqlens_q[seq_idx];
  const int q_end_token = cu_seqlens_q[seq_idx + 1];
  const int query_len = q_end_token - q_start_token;
  const int seq_len = seq_lens[seq_idx];
  const int ctx_len = seq_len - query_len;

  const int q_tile_start = q_tile_idx * BLOCK_M_V2;
  if (q_tile_start >= query_len) return;

  const int wave_q_offset = wave_id * M_PER_WAVE;
  const int wave_q_tile_start = q_tile_start + wave_q_offset;
  const int my_q_pos = wave_q_tile_start + lane_lo;
  const bool valid_q = my_q_pos < query_len;
  const int valid_q_count_for_wave =
      max(0, min(M_PER_WAVE, query_len - wave_q_tile_start));

  const int num_queries_per_kv = num_query_heads / num_kv_heads;
  const int kv_head_idx = head_idx / num_queries_per_kv;

  // ---- Load Q in fp16 (for P×V path) ----
  V16 q_fp_frags[FRAGS];
  if (valid_q) {
    const T* q_row = q + (int64_t)(q_start_token + my_q_pos) * stride_q_token +
                     (int64_t)head_idx * stride_q_head;
    #pragma unroll
    for (int dh = 0; dh < FRAGS; ++dh) {
      __builtin_memcpy(&q_fp_frags[dh], q_row + dh * 16, sizeof(V16));
    }
  } else {
    #pragma unroll
    for (int dh = 0; dh < FRAGS; ++dh) {
      #pragma unroll
      for (int k = 0; k < 16; ++k) q_fp_frags[dh][k] = (E)0;
    }
  }

  // ---- Quantize Q to int8 per-row (for Q×K INT8 WMMA path) ----
  // q_scale = absmax(Q_row) / 127. Q_int8 = round(Q / q_scale).
  // Also compute Q_int_sum = sum(Q_int8) for zero-point correction.
  float q_absmax = 0.0f;
  #pragma unroll
  for (int dh = 0; dh < FRAGS; ++dh) {
    #pragma unroll
    for (int k = 0; k < 16; ++k) {
      q_absmax = fmaxf(q_absmax, fabsf((float)q_fp_frags[dh][k]));
    }
  }
  const float q_scale = (q_absmax > 0.0f) ? (q_absmax / 127.0f) : 1.0f;
  const float q_scale_inv = 1.0f / q_scale;

  v16i8 q_int8_frags[FRAGS];
  float q_int_sum = 0.0f;
  #pragma unroll
  for (int dh = 0; dh < FRAGS; ++dh) {
    #pragma unroll
    for (int k = 0; k < 16; ++k) {
      float qf = (float)q_fp_frags[dh][k];
      int qi = __float2int_rn(qf * q_scale_inv);
      qi = max(-128, min(127, qi));
      q_int8_frags[dh][k] = (int8_t)qi;
      q_int_sum += (float)qi;
    }
  }

  // Per-wave online-softmax state.
  float m_state[8], l_state[8];
  v8fp32 out_acc[FRAGS];
  #pragma unroll
  for (int i = 0; i < 8; ++i) { m_state[i] = -INFINITY; l_state[i] = 0.0f; }
  #pragma unroll
  for (int dh = 0; dh < FRAGS; ++dh) {
    out_acc[dh] = (v8fp32){0, 0, 0, 0, 0, 0, 0, 0};
  }

  // ---- Shared LDS ----
  // K_lds_u8: uint8 [D_HIGH=8][K_TILE=16][16] = 2 KB (half of bf16!)
  // V_lds: fp16 [HEAD_SIZE][K_TILE] = 4 KB
  // P_lds: fp16 [4 waves][16 rows][16 cols] = 2 KB
  // scales: float [K_TILE] × 4 = 256 B
  __shared__ uint8_t K_lds_u8[FRAGS * K_TILE * 16];     // 2 KB
  __shared__ T V_lds[HEAD_SIZE * K_TILE];                 // 4 KB
  __shared__ T P_lds[NUM_WAVES][M_PER_WAVE * K_TILE];     // 2 KB
  __shared__ float k_scale_lds[K_TILE];                   // 64 B
  __shared__ float k_zp_lds[K_TILE];                      // 64 B
  __shared__ float v_scale_lds[K_TILE];                   // 64 B
  __shared__ float v_zp_lds[K_TILE];                      // 64 B
  T* P_lds_wave = &P_lds[wave_id][0];

  // ---- PHASE 1: Cached prefix (INT4 paged cache, no causal) ----
  for (int start_n = 0; start_n < ctx_len; start_n += K_TILE) {
    load_k_tile_paged_int4_coop<HEAD_SIZE>(
        K_lds_u8, k_cache, k_scale_cache, block_table, seq_idx, kv_head_idx,
        start_n, ctx_len, block_size, max_blocks_per_seq,
        stride_kcache_block, stride_kcache_slot, stride_kcache_head,
        stride_ks_blk, stride_ks_slot, stride_ks_head,
        k_scale_lds, k_zp_lds, tid);
    load_v_tile_paged_int4_coop<T, HEAD_SIZE>(
        V_lds, v_cache, v_scale_cache, block_table, seq_idx, kv_head_idx,
        start_n, ctx_len, block_size, max_blocks_per_seq,
        stride_vcache_block, stride_vcache_slot, stride_vcache_head,
        stride_vs_blk, stride_vs_slot, stride_vs_head,
        v_scale_lds, v_zp_lds, tid);
    __syncthreads();

    const int valid_k_count = min(K_TILE, ctx_len - start_n);
    attn_step_wave_int4<T, HEAD_SIZE, X_FP16, /*CAUSAL_MASK=*/false>(
        K_lds_u8, V_lds, P_lds_wave, k_scale_lds, k_zp_lds,
        v_scale_lds, v_zp_lds,
        q_int8_frags, q_fp_frags, out_acc, m_state, l_state,
        q_scale, q_int_sum,
        wave_q_tile_start, start_n, valid_q_count_for_wave, valid_k_count,
        sm_scale, lane, lane_lo, lane_hi);
    __syncthreads();
  }

  // ---- PHASE 2: Current chunk (fp16, RHT-rotated, causal) ----
  // Current chunk tokens are passed pre-rotated but not quantized.
  // For Q×K: quantize k_chunk to uint8 on-the-fly (nibble range 0..15
  // won't work — fp16 values are arbitrary). Instead, use the bf16 WMMA path
  // for Phase 2 (same as INT8 kernel does — set scale=1, zp=0).
  // Actually: just store fp16 K to K_lds as bf16 WMMA fragments and use
  // the fp16 WMMA path for Q×K on the current chunk. This is simpler and
  // Phase 2 is typically small (≤ max_num_batched_tokens, often 2048).
  //
  // We use a separate fp16 K_lds for Phase 2 (reusing V_lds space is messy).
  // Easier: just overlay K_lds_u8 with fp16 data since Phase 1 is done.

  const int valid_q_count_for_block =
      max(0, min(BLOCK_M_V2, query_len - q_tile_start));
  const int causal_k_upper =
      causal ? (q_tile_start + valid_q_count_for_block) : query_len;
  const int phase2_k_end = min(query_len, causal_k_upper);

  // For Phase 2, reinterpret K_lds as fp16 (we have 2KB = enough for 4KB? No.)
  // Actually K_lds_u8 is only 2KB but fp16 K needs 4KB. Use a union or just
  // use the V_lds-style approach with a separate __shared__ for P2.
  // Simplest: use the same bf16 WMMA layout as v2 fp16 kernel.
  // K_lds_u8 has 2KB. We need FRAGS*2 * K_TILE * X_FP16 * sizeof(T) = 8*2*16*8*2 = 4KB.
  // Won't fit in K_lds_u8. Let's just declare a separate shared array.
  // Actually we can reuse K_lds_u8 + V_lds combined (6KB total) but that's complex.
  // Best approach: use __shared__ union for Phase 1 vs Phase 2.
  // For now: Phase 2 uses same K_lds_u8 but we actually DO have space because
  // we don't need V_lds at the same time (we load V into V_lds fresh each tile).
  //
  // Actually simplest: Cast K_lds_u8 to T* for Phase 2 fp16 data. 2KB holds
  // at most 1024 fp16 values. We need FRAGS*2*K_TILE*X_FP16 = 8*2*16*8 = 2048 fp16
  // = 4KB. Doesn't fit.
  //
  // Real solution: For Phase 2, do inline fp16 Q×K without WMMA (scalar dot
  // product on fp16). Phase 2 is short and compute isn't the bottleneck there.
  //
  // ACTUALLY: simplest correct solution — reuse the full bf16 WMMA attn_step
  // from the INT8 kernel for Phase 2. We have enough shared memory if we
  // overlay K_lds_u8 with a larger array. Let's use a union.

  // Phase 2 K_lds in fp16 format — overlaps with K_lds_u8 (not used anymore)
  // We need 4KB but K_lds_u8 is only 2KB. So we declare K_lds_fp16 overlapping
  // with K_lds_u8 + first half of V_lds. But V_lds is needed too.
  // Final approach: add separate __shared__ for Phase 2. Total LDS budget on
  // gfx1100 is 64KB per workgroup, we're using ~8.5KB, so plenty of room.
  __shared__ T K_lds_fp16[FRAGS * 2 * K_TILE * X_FP16];  // 4 KB (Phase 2 only)

  for (int start_n = 0; start_n < phase2_k_end; start_n += K_TILE) {
    // Load fp16 K chunk into K_lds_fp16 (same as v2 fp16 kernel)
    {
      constexpr int X = X_FP16;
      const int my_k_idx = tid >> 3;
      const int my_dh_base = (tid & 7) * 2;
      const int abs_k = start_n + my_k_idx;
      const bool valid_k = abs_k < query_len;
      const T* row = valid_k
          ? (k_chunk + (int64_t)(q_start_token + abs_k) * stride_kc_token +
             (int64_t)kv_head_idx * stride_kc_head)
          : nullptr;
      #pragma unroll
      for (int dh = 0; dh < 2; ++dh) {
        const int d_high = my_dh_base + dh;
        int4 vec;
        if (valid_k) {
          vec = *(const int4*)(row + d_high * X);
        } else {
          vec.x = vec.y = vec.z = vec.w = 0;
        }
        *(int4*)&K_lds_fp16[d_high * (K_TILE * X) + my_k_idx * X] = vec;
      }
    }
    // Load fp16 V chunk into V_lds (transposed)
    {
      #pragma unroll
      for (int p = 0; p < 2; ++p) {
        const int my_k = tid >> 3;
        const int my_dc = (tid & 7) + p * 8;
        const int d_base = my_dc * 8;
        const int abs_k = start_n + my_k;
        const bool valid = abs_k < query_len;
        int4 vec;
        if (valid) {
          const T* src = v_chunk +
                         (int64_t)(q_start_token + abs_k) * stride_vc_token +
                         (int64_t)kv_head_idx * stride_vc_head + (int64_t)d_base;
          vec = *(const int4*)src;
        } else {
          vec.x = vec.y = vec.z = vec.w = 0;
        }
        T tmp[8];
        __builtin_memcpy(tmp, &vec, 16);
        #pragma unroll
        for (int e = 0; e < 8; ++e) {
          V_lds[(d_base + e) * K_TILE + my_k] = tmp[e];
        }
      }
    }
    // Scales = identity for chunk (not quantized)
    if (tid < K_TILE) {
      k_scale_lds[tid] = 1.0f;
      k_zp_lds[tid] = 0.0f;
      v_scale_lds[tid] = 1.0f;
      v_zp_lds[tid] = 0.0f;
    }
    __syncthreads();

    // Phase 2 uses bf16 WMMA for Q×K (fp16 data, not int4).
    // Since scale=1 and zp=0, the INT4 attn_step degenerates to:
    //   S = i32_dot * q_scale * 1.0 * sm_scale - Q_int_sum * 0 = correct
    // BUT: K_lds_fp16 is in fp16 format, not uint8. We need to use
    // the Q_fp_frags (fp16 Q) + bf16 WMMA for Phase 2.
    // Inline a simplified bf16 attn_step here.
    {
      const int valid_k_count = min(K_TILE, query_len - start_n);

      // Q × K via bf16 WMMA (using fp16 Q and fp16 K)
      v8fp32 s_acc_p2 = {0, 0, 0, 0, 0, 0, 0, 0};
      #pragma unroll
      for (int dh = 0; dh < FRAGS; ++dh) {
        V16 b_frag;
        int4 lo = *(const int4*)&K_lds_fp16[(dh * 2 + 0) * (K_TILE * X_FP16) + lane_lo * X_FP16];
        int4 hi = *(const int4*)&K_lds_fp16[(dh * 2 + 1) * (K_TILE * X_FP16) + lane_lo * X_FP16];
        __builtin_memcpy(&b_frag, &lo, 16);
        __builtin_memcpy(((char*)&b_frag) + 16, &hi, 16);
        s_acc_p2 = wmma_mma(q_fp_frags[dh], b_frag, s_acc_p2);
      }

      // Apply sm_scale + mask (no zp correction, scale=1)
      const bool k_in_seg = (lane_lo < valid_k_count);
      const int abs_k = start_n + lane_lo;
      #pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int m_row = 2 * i + lane_hi;
        const bool m_in_q = (m_row < valid_q_count_for_wave);
        bool keep = m_in_q && k_in_seg;
        if (causal) {
          const int abs_q = wave_q_tile_start + m_row;
          keep = keep && (abs_k <= abs_q);
        }
        s_acc_p2[i] = keep ? (s_acc_p2[i] * sm_scale) : -INFINITY;
      }

      // Online softmax
      float m_ij2[8], m_new2[8], alpha2[8], p_ij2[8], l_ij2[8];
      #pragma unroll
      for (int i = 0; i < 8; ++i) {
        m_ij2[i] = wave16_max(s_acc_p2[i]);
        m_new2[i] = fmaxf(m_state[i], m_ij2[i]);
        alpha2[i] = (m_state[i] == -INFINITY) ? 0.0f : __expf(m_state[i] - m_new2[i]);
        p_ij2[i] = (m_new2[i] == -INFINITY) ? 0.0f : __expf(s_acc_p2[i] - m_new2[i]);
        l_ij2[i] = wave16_sum(p_ij2[i]);
        l_state[i] = l_state[i] * alpha2[i] + l_ij2[i];
        m_state[i] = m_new2[i];
      }
      #pragma unroll
      for (int dh = 0; dh < FRAGS; ++dh) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
          out_acc[dh][i] *= alpha2[i];
        }
      }

      // P × V via bf16 WMMA (V in fp16, no zp correction needed)
      #pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int m_row = 2 * i + lane_hi;
        P_lds_wave[m_row * K_TILE + lane_lo] = to_T<T>(p_ij2[i]);
      }
      V16 p_frag;
      int4 p_lo2 = *(const int4*)&P_lds_wave[lane_lo * K_TILE + 0];
      int4 p_hi2 = *(const int4*)&P_lds_wave[lane_lo * K_TILE + 8];
      __builtin_memcpy(&p_frag, &p_lo2, 16);
      __builtin_memcpy(((char*)&p_frag) + 16, &p_hi2, 16);

      #pragma unroll
      for (int dh = 0; dh < FRAGS; ++dh) {
        V16 v_frag;
        int4 v_lo = *(const int4*)&V_lds[(dh * 16 + lane_lo) * K_TILE + 0];
        int4 v_hi = *(const int4*)&V_lds[(dh * 16 + lane_lo) * K_TILE + 8];
        __builtin_memcpy(&v_frag, &v_lo, 16);
        __builtin_memcpy(((char*)&v_frag) + 16, &v_hi, 16);
        out_acc[dh] = wmma_mma(p_frag, v_frag, out_acc[dh]);
      }
    }
    __syncthreads();
  }

  // ---- Epilogue: divide by L, write output ----
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int m_row = 2 * i + lane_hi;
    const int abs_m_row = wave_q_offset + m_row;
    const int abs_q_pos = q_tile_start + abs_m_row;
    if (abs_q_pos >= query_len) continue;
    const float l_inv = 1.0f / (l_state[i] + 1e-10f);
    T* out_row = out + (int64_t)(q_start_token + abs_q_pos) * stride_o_token +
                 (int64_t)head_idx * stride_o_head;
    #pragma unroll
    for (int dh = 0; dh < FRAGS; ++dh) {
      const int out_col = dh * 16 + lane_lo;
      out_row[out_col] = to_T<T>(out_acc[dh][i] * l_inv);
    }
  }
}

// ---------------------------------------------------------------------------
// Launcher
// ---------------------------------------------------------------------------

template <typename T, int HEAD_SIZE>
void launch_paged_prefill_attn_v2_int4(
    T* out, const T* q, const T* k_chunk, const T* v_chunk,
    const uint8_t* k_cache, const uint8_t* v_cache,
    const float* k_scale_cache, const float* v_scale_cache,
    const int* block_table, const int* cu_seqlens_q, const int* seq_lens,
    int num_seqs, int num_query_heads, int num_kv_heads,
    int block_size, int max_blocks_per_seq, int max_query_len,
    float sm_scale, bool causal,
    int64_t stride_q_token, int64_t stride_q_head,
    int64_t stride_kc_token, int64_t stride_kc_head,
    int64_t stride_vc_token, int64_t stride_vc_head,
    int64_t stride_kcache_block, int64_t stride_kcache_slot,
    int64_t stride_kcache_head,
    int64_t stride_vcache_block, int64_t stride_vcache_slot,
    int64_t stride_vcache_head,
    int64_t stride_ks_blk, int64_t stride_ks_slot, int64_t stride_ks_head,
    int64_t stride_vs_blk, int64_t stride_vs_slot, int64_t stride_vs_head,
    int64_t stride_o_token, int64_t stride_o_head,
    cudaStream_t stream) {
  const int q_blocks = (max_query_len + BLOCK_M_V2 - 1) / BLOCK_M_V2;
  dim3 block(THREADS);
  dim3 grid(num_seqs, num_query_heads, q_blocks);
  paged_prefill_attn_kernel_v2_int4<T, HEAD_SIZE><<<grid, block, 0, stream>>>(
      out, q, k_chunk, v_chunk, k_cache, v_cache,
      k_scale_cache, v_scale_cache, block_table, cu_seqlens_q, seq_lens,
      num_query_heads, num_kv_heads, block_size, max_blocks_per_seq,
      sm_scale, causal,
      stride_q_token, stride_q_head,
      stride_kc_token, stride_kc_head, stride_vc_token, stride_vc_head,
      stride_kcache_block, stride_kcache_slot, stride_kcache_head,
      stride_vcache_block, stride_vcache_slot, stride_vcache_head,
      stride_ks_blk, stride_ks_slot, stride_ks_head,
      stride_vs_blk, stride_vs_slot, stride_vs_head,
      stride_o_token, stride_o_head);
}

// Explicit instantiations
template void launch_paged_prefill_attn_v2_int4<half, 128>(
    half*, const half*, const half*, const half*,
    const uint8_t*, const uint8_t*, const float*, const float*,
    const int*, const int*, const int*,
    int, int, int, int, int, int, float, bool,
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t, int64_t,
    int64_t, int64_t, int64_t,
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t, cudaStream_t);

template void launch_paged_prefill_attn_v2_int4<bf16_t, 128>(
    bf16_t*, const bf16_t*, const bf16_t*, const bf16_t*,
    const uint8_t*, const uint8_t*, const float*, const float*,
    const int*, const int*, const int*,
    int, int, int, int, int, int, float, bool,
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t, int64_t,
    int64_t, int64_t, int64_t,
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t, cudaStream_t);

#endif  // USE_ROCM

}  // namespace prefill_attn_rdna3_v2_int4
}  // namespace vllm

// ---------------------------------------------------------------------------
// Torch-callable entry point (registered in torch_bindings.cpp)
// ---------------------------------------------------------------------------

#if defined(USE_ROCM)
void paged_prefill_attn_rdna3_int4(
    torch::Tensor& out, torch::Tensor q, torch::Tensor k_chunk,
    torch::Tensor v_chunk, torch::Tensor k_cache, torch::Tensor v_cache,
    torch::Tensor k_scale_cache, torch::Tensor v_scale_cache,
    torch::Tensor block_table, torch::Tensor cu_seqlens_q,
    torch::Tensor seq_lens, int64_t max_query_len, double sm_scale,
    bool causal) {
  using namespace vllm::prefill_attn_rdna3_v2_int4;

  const int num_seqs = seq_lens.size(0);
  const int num_query_heads = q.size(1);
  const int num_kv_heads = k_scale_cache.size(2);  // [blocks, slots, heads]
  // k_cache: [blocks, slots, heads, dim//2]
  const int block_size = k_cache.size(1);
  const int max_blocks_per_seq = block_table.size(1);
  auto stream = at::cuda::getCurrentCUDAStream().stream();

  TORCH_CHECK(q.dtype() == at::kHalf || q.dtype() == at::kBFloat16,
              "paged_prefill_attn_rdna3_int4: only fp16/bf16 supported");
  TORCH_CHECK(k_cache.dtype() == at::kByte, "k_cache must be uint8 (packed int4)");

  if (q.dtype() == at::kHalf) {
    using T = half;
    launch_paged_prefill_attn_v2_int4<T, 128>(
        (T*)out.data_ptr(), (const T*)q.data_ptr(),
        (const T*)k_chunk.data_ptr(), (const T*)v_chunk.data_ptr(),
        (const uint8_t*)k_cache.data_ptr(), (const uint8_t*)v_cache.data_ptr(),
        (const float*)k_scale_cache.data_ptr(),
        (const float*)v_scale_cache.data_ptr(),
        (const int*)block_table.data_ptr(),
        (const int*)cu_seqlens_q.data_ptr(),
        (const int*)seq_lens.data_ptr(),
        num_seqs, num_query_heads, num_kv_heads,
        block_size, max_blocks_per_seq, (int)max_query_len,
        (float)sm_scale, causal,
        q.stride(0), q.stride(1),
        k_chunk.stride(0), k_chunk.stride(1),
        v_chunk.stride(0), v_chunk.stride(1),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
        k_scale_cache.stride(0), k_scale_cache.stride(1),
        k_scale_cache.stride(2),
        v_scale_cache.stride(0), v_scale_cache.stride(1),
        v_scale_cache.stride(2),
        out.stride(0), out.stride(1),
        stream);
  } else {
    using T = vllm::prefill_attn_rdna3::bf16_t;
    launch_paged_prefill_attn_v2_int4<T, 128>(
        (T*)out.data_ptr(), (const T*)q.data_ptr(),
        (const T*)k_chunk.data_ptr(), (const T*)v_chunk.data_ptr(),
        (const uint8_t*)k_cache.data_ptr(), (const uint8_t*)v_cache.data_ptr(),
        (const float*)k_scale_cache.data_ptr(),
        (const float*)v_scale_cache.data_ptr(),
        (const int*)block_table.data_ptr(),
        (const int*)cu_seqlens_q.data_ptr(),
        (const int*)seq_lens.data_ptr(),
        num_seqs, num_query_heads, num_kv_heads,
        block_size, max_blocks_per_seq, (int)max_query_len,
        (float)sm_scale, causal,
        q.stride(0), q.stride(1),
        k_chunk.stride(0), k_chunk.stride(1),
        v_chunk.stride(0), v_chunk.stride(1),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
        k_scale_cache.stride(0), k_scale_cache.stride(1),
        k_scale_cache.stride(2),
        v_scale_cache.stride(0), v_scale_cache.stride(1),
        v_scale_cache.stride(2),
        out.stride(0), out.stride(1),
        stream);
  }
}
#else
void paged_prefill_attn_rdna3_int4(
    torch::Tensor& out, torch::Tensor q, torch::Tensor k_chunk,
    torch::Tensor v_chunk, torch::Tensor k_cache, torch::Tensor v_cache,
    torch::Tensor k_scale_cache, torch::Tensor v_scale_cache,
    torch::Tensor block_table, torch::Tensor cu_seqlens_q,
    torch::Tensor seq_lens, int64_t max_query_len, double sm_scale,
    bool causal) {
  TORCH_CHECK(false, "paged_prefill_attn_rdna3_int4 requires ROCm");
}
#endif
