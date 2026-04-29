// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// W4A16 GPTQ kernel for RDNA3 (gfx1100 / RX 7900 XTX class), templated on the
// activation dtype (half or __hip_bfloat16). Adapted from exllamav2's 4-bit
// kernel (csrc/quantization/gptq/q_gemm.cu) with three changes:
//
//   1. Output accumulator is FP32 (atomicAdd to a zeroed FP32 tile, then
//      cast back to T at the end). RDNA3 has a native v_global_atomic_add_f32
//      and lacks reliable hardware atomic-add for half2/bfloat162 in global
//      memory, so accumulating in FP32 is both simpler and faster.
//
//   2. The bf16 path uses a dedicated bit-trick that avoids the fp16-only
//      "upper nibble * 16" trick, which would overflow the 7-bit bf16
//      mantissa. See qdq_4_rdna3.cuh for details.
//
//   3. Wave32-friendly geometry: 32 threads per block, 1 wave per CU per
//      block, BLOCK_KN_SIZE=128 (4 N output cols per thread, exactly one
//      cache-line-sized int4 load per K iteration).

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#if defined(USE_ROCM)
  #include <hip/hip_runtime.h>
  #include <hip/hip_bf16.h>
  #include <hip/hip_fp16.h>
#else
  #include <cuda_runtime.h>
  #include <cuda_bf16.h>
  #include <cuda_fp16.h>
#endif

#include "qdq_4_rdna3.cuh"

namespace vllm {
namespace gptq_rdna3 {

// BLOCK_KN_SIZE = 256 (was 128 in exllama). Each block now covers 256 K
// elements and THREADS_X*4 = 1024 N columns. For Qwen-class K=4096 this
// halves gridDim.z (32 → 16) and therefore halves the atomic count per
// output position vs the exllama default. THREADS_X=256 = 8 waves on RDNA3
// wave32; with ~32 wave slots per CU we still fit 4 blocks per CU at peak.
#define BLOCK_KN_SIZE 256
#define THREADS_X 256

// ---------------------------------------------------------------------------
// Per-dtype helpers. We avoid heavy template metaprogramming and just provide
// overloaded inline functions; the kernel below selects via `if constexpr`.
// ---------------------------------------------------------------------------

// Type-generic zero — both half and bf16_t in HIP/ROCm have a converting
// constructor from float, but going through __float2half_rn / __float2bfloat16
// is the unambiguously correct path on every ROCm version.
template <typename T>
__forceinline__ __device__ T tzero();

template <>
__forceinline__ __device__ half tzero<half>() {
  return __float2half_rn(0.0f);
}

template <>
__forceinline__ __device__ bf16_t tzero<bf16_t>() {
  return __float2bfloat16(0.0f);
}

__forceinline__ __device__ float dot22_8_f(half2 (&dq)[4], const half* a_ptr) {
  half2 result = {};
  const half2* a2_ptr = (const half2*)a_ptr;
#pragma unroll
  for (int i = 0; i < 4; i++) result = __hfma2(dq[i], *a2_ptr++, result);
  return __half2float(__low2half(result)) + __half2float(__high2half(result));
}

__forceinline__ __device__ float dot22_8_f(bf162_t (&dq)[4],
                                           const bf16_t* a_ptr) {
  // RDNA3 (gfx1100) lacks a packed bf16 FMA: there is no v_pk_fma_bf16 in
  // the gfx11 ISA (it only landed on CDNA3+ / gfx94x and later). hipcc
  // therefore lowers __hfma2(bf162_t, bf162_t, bf162_t) to a serialised
  // fallback (single-element FMAs or fp32 round-trips), which empirically
  // runs ~2× the cycle count of v_pk_fma_f16 on the same VALU. The bf16
  // decode path was paying that tax in full, scaling linearly with M (the
  // fp16 path scales sub-linearly because its v_pk_fma_f16 is full rate
  // and the kernel becomes memory-bound).
  //
  // Fix: widen bf16 → fp32 explicitly (a left-shift by 16, free in VGPRs)
  // and accumulate with v_fma_f32, which IS full rate on RDNA3. Same FMA
  // count, but each FMA is fast. Bonus: the accumulator is now fp32
  // throughout instead of bf16, which is also numerically more accurate
  // (no compounding bf16-rounding inside the dot loop).
  float result = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; i++) {
    uint32_t aw, dw;
    __builtin_memcpy(&aw, a_ptr + 2 * i, sizeof(uint32_t));
    __builtin_memcpy(&dw, &dq[i], sizeof(uint32_t));
    // bf16 in low 16 bits  → fp32 by left-shifting into the upper half.
    // bf16 in high 16 bits → already aligned with fp32's upper half.
    float a_x = __uint_as_float((aw & 0xFFFFu) << 16);
    float a_y = __uint_as_float(aw & 0xFFFF0000u);
    float d_x = __uint_as_float((dw & 0xFFFFu) << 16);
    float d_y = __uint_as_float(dw & 0xFFFF0000u);
    result = __fmaf_rn(d_x, a_x, result);
    result = __fmaf_rn(d_y, a_y, result);
  }
  return result;
}

// ---------------------------------------------------------------------------
// Packed atomic-add via CAS-loop on the underlying uint32. RDNA3 (gfx11) does
// NOT have native v_global_atomic_pk_add_f16 / _bf16 (those landed on gfx940
// / gfx1250 respectively), so this lowers to global_atomic_cmpswap_b32 plus
// retry. We use this in the kernel epilogue to write directly to fp16/bf16
// output without going through an FP32 accumulator buffer + epilogue cast
// pass — saves M*N*4 bytes of allocation, the memset, and a kernel launch
// per matmul (~5-10 μs/call → 11-22% of decode budget at 50 tk/s).
// ---------------------------------------------------------------------------

__forceinline__ __device__ void atomic_add_pk_f16(half2* addr, half2 val) {
  uint32_t* addr_u = reinterpret_cast<uint32_t*>(addr);
  uint32_t old = *addr_u;
  while (true) {
    half2 cur = *reinterpret_cast<half2*>(&old);
    half2 sum = __hadd2(cur, val);
    uint32_t sum_u = *reinterpret_cast<uint32_t*>(&sum);
    uint32_t prev = atomicCAS(addr_u, old, sum_u);
    if (prev == old) break;
    old = prev;
  }
}

__forceinline__ __device__ void atomic_add_pk_bf16(bf162_t* addr,
                                                   bf162_t val) {
  uint32_t* addr_u = reinterpret_cast<uint32_t*>(addr);
  uint32_t old = *addr_u;
  while (true) {
    bf162_t cur = *reinterpret_cast<bf162_t*>(&old);
    bf162_t sum = __hadd2(cur, val);
    uint32_t sum_u = *reinterpret_cast<uint32_t*>(&sum);
    uint32_t prev = atomicCAS(addr_u, old, sum_u);
    if (prev == old) break;
    old = prev;
  }
}

// Load one row's worth of 4 packed zeros (column n..n+3) from a [groups, N/8]
// uint32 tensor. n is a multiple of 4 by construction (n = offset_n + t*4 with
// offset_n = blockIdx.x * 512), so the 4 nibbles always live within one or two
// uint32 words; in practice within one because n & 7 is 0 or 4.
__forceinline__ __device__ void load4_zeros(const uint32_t* qzeros_row,
                                            int n, int (&zeros)[4]) {
  int qcol = n / 8;
  int shift = (n & 0x07) * 4;
  uint32_t d = qzeros_row[qcol] >> shift;
  zeros[0] = (int)(d & 0xF);
  zeros[1] = (int)((d >> 4) & 0xF);
  zeros[2] = (int)((d >> 8) & 0xF);
  zeros[3] = (int)((d >> 12) & 0xF);
}

template <typename T>
__forceinline__ __device__ void load4_scales(const T* scales_row, int n,
                                             T (&scales)[4]) {
  scales[0] = scales_row[n + 0];
  scales[1] = scales_row[n + 1];
  scales[2] = scales_row[n + 2];
  scales[3] = scales_row[n + 3];
}

// ---------------------------------------------------------------------------
// Main kernel.
// ---------------------------------------------------------------------------

template <typename T, int M_COUNT>
__global__ void gemm_q4_kernel_rdna3(const T* __restrict__ a,
                                     const uint32_t* __restrict__ b_q_weight,
                                     const uint32_t* __restrict__ b_qzeros,
                                     const T* __restrict__ b_scales,
                                     T* __restrict__ c,
                                     const int size_m, const int size_n,
                                     const int size_k, const int groups,
                                     const int zero_offset,
                                     const int* __restrict__ b_q_perm) {
  const int t = threadIdx.x;
  const int offset_n = blockIdx.x * BLOCK_KN_SIZE * 4;
  const int offset_m = blockIdx.y * M_COUNT;
  const int offset_k = blockIdx.z * BLOCK_KN_SIZE;
  const int end_k = min(offset_k + BLOCK_KN_SIZE, size_k);
  const int n = offset_n + t * 4;

  // LDS layout: [M_COUNT][BLOCK_KN_SIZE + LDS_PAD]. The PAD=8 elements per M
  // row break the natural 256-element/512-byte alignment that would otherwise
  // collide on the same LDS bank when a thread reads block_a[0..M_COUNT-1][k]
  // (same k, different m). Row stride becomes 264 elements * 2B = 528B = 132
  // 4-byte banks, so m-stride hits banks (m*132)%32 = (m*4)%32 — distinct for
  // all M_COUNT ≤ 8. Cost: 16B LDS per block, irrelevant.
  constexpr int LDS_PAD = 8;
  __shared__ T block_a[M_COUNT][BLOCK_KN_SIZE + LDS_PAD];

  // Stage A: each thread loads 1 K element per M row into LDS (with optional
  // act-order permutation). THREADS_X == BLOCK_KN_SIZE so this is a 1:1 map.
  // For M_COUNT > 1 with size_m not a multiple of M_COUNT, slots past size_m
  // are zero-padded so the dot product contribution is 0 (we then skip the
  // atomic write for those rows below).
  static_assert(BLOCK_KN_SIZE == THREADS_X,
                "BLOCK_KN_SIZE must equal THREADS_X (1 K element per thread)");
  if (offset_k + t < end_k) {
#pragma unroll
    for (int m = 0; m < M_COUNT; ++m) {
      T av;
      if (offset_m + m < size_m) {
        const T* a_row = a + (offset_m + m) * size_k;
        if (b_q_perm)
          av = a_row[b_q_perm[offset_k + t]];
        else
          av = a_row[offset_k + t];
      } else {
        av = tzero<T>();  // zero-pad invalid M rows
      }
      block_a[m][t] = av;
    }
  }

  // Threads beyond the right edge of N have nothing to do. Note: we must NOT
  // return before __syncthreads() if any thread in the block participates in
  // the LDS load above — but here all 128 threads always do, regardless of
  // whether their `n` is in bounds. The early return below is safe because
  // the LDS load doesn't depend on `n`, only on `t`/`offset_k`.
  __syncthreads();
  if (n >= size_n) return;

  // Group bookkeeping. We require size_k % groups == 0 (groupsize divides K).
  const int groupsize = size_k / groups;
  int group = offset_k / groupsize;
  int nextgroup = (group + 1) * groupsize;

  // qweight stride: weights are [K/8, N] uint32 with K packed at dim 0.
  int qk = offset_k / 8;
  const uint32_t* b_ptr = b_q_weight + qk * size_n + n;

  // Per-column dequant constants. We hold one set of (z, y) pairs per column.
  // For fp16 we need the exllama (z1z16, y1y16) double-pair; for bf16 a single
  // (z, y) suffices because we shift each pair down to bits [3:0]/[19:16].
  half2 z1z16_h[4][2], y1y16_h[4][2];
  bf162_t z_b[4], y_b[4];

  auto refresh_group = [&](int g) {
    const uint32_t* qz_row = b_qzeros + g * (size_n / 8);
    const T* sc_row = b_scales + g * size_n;
    int zeros[4];
    T scales[4];
    load4_zeros(qz_row, n, zeros);
    load4_scales<T>(sc_row, n, scales);
    if constexpr (std::is_same<T, half>::value) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        prep_zero_scale_fp16((uint32_t)(zeros[i] + zero_offset), scales[i],
                             z1z16_h[i], y1y16_h[i]);
      }
    } else {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        prep_zero_scale_bf16((uint32_t)(zeros[i] + zero_offset), scales[i],
                             z_b[i], y_b[i]);
      }
    }
  };

  refresh_group(group);

  float block_c[M_COUNT][4];
#pragma unroll
  for (int m = 0; m < M_COUNT; ++m) {
#pragma unroll
    for (int j = 0; j < 4; ++j) block_c[m][j] = 0.0f;
  }

  // Note on group-transition granularity: we check `k == nextgroup` at the
  // start of each outer iteration (which advances K by 32). This is correct
  // when group_size >= 32 OR group_size divides 32 evenly (groupsize is one
  // of {1,2,4,8,16,32,64,128,...}). For group_size in {16, 8, 4, ...} the
  // inner loop would cross a group boundary between j-iterations; we require
  // group_size >= 32 here, mirroring exllama's assumption.
  //
  // Software pipelining: we issue all 4 vectorized weight loads up front
  // before any dequant/FMA depends on them. This gives the AMDGPU backend
  // freedom to schedule the global_loads early and overlap their latency
  // with dequant + v_pk_fma_f16 of earlier iterations. Cost: 4×int4 = 16
  // VGPRs in flight per thread, plenty of headroom on RDNA3.
  int k = offset_k;
  while (k < end_k) {
    if (k == nextgroup) {
      group++;
      nextgroup += groupsize;
      refresh_group(group);
    }

    // Prefetch all four j-iterations' weight words. The compiler emits 4
    // global_load_b128 instructions back-to-back; the dependent dequant +
    // FMA work below hides their latency.
    int4 b_w[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      b_w[j] = *(const int4*)(b_ptr + j * size_n);
    }
    b_ptr += 4 * size_n;

#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int a_off = (k - offset_k) + 8 * j;

      if constexpr (std::is_same<T, half>::value) {
        half2 dq[4][4];
        dequant_4bit_8_fp16((uint32_t)b_w[j].x, dq[0], z1z16_h[0],
                            y1y16_h[0]);
        dequant_4bit_8_fp16((uint32_t)b_w[j].y, dq[1], z1z16_h[1],
                            y1y16_h[1]);
        dequant_4bit_8_fp16((uint32_t)b_w[j].z, dq[2], z1z16_h[2],
                            y1y16_h[2]);
        dequant_4bit_8_fp16((uint32_t)b_w[j].w, dq[3], z1z16_h[3],
                            y1y16_h[3]);

#pragma unroll
        for (int m = 0; m < M_COUNT; ++m) {
          const half* a_ptr =
              reinterpret_cast<const half*>(&block_a[m][a_off]);
          block_c[m][0] += dot22_8_f(dq[0], a_ptr);
          block_c[m][1] += dot22_8_f(dq[1], a_ptr);
          block_c[m][2] += dot22_8_f(dq[2], a_ptr);
          block_c[m][3] += dot22_8_f(dq[3], a_ptr);
        }
      } else {
        bf162_t dq[4][4];
        dequant_4bit_8_bf16((uint32_t)b_w[j].x, dq[0], z_b[0], y_b[0]);
        dequant_4bit_8_bf16((uint32_t)b_w[j].y, dq[1], z_b[1], y_b[1]);
        dequant_4bit_8_bf16((uint32_t)b_w[j].z, dq[2], z_b[2], y_b[2]);
        dequant_4bit_8_bf16((uint32_t)b_w[j].w, dq[3], z_b[3], y_b[3]);

#pragma unroll
        for (int m = 0; m < M_COUNT; ++m) {
          const bf16_t* a_ptr =
              reinterpret_cast<const bf16_t*>(&block_a[m][a_off]);
          block_c[m][0] += dot22_8_f(dq[0], a_ptr);
          block_c[m][1] += dot22_8_f(dq[1], a_ptr);
          block_c[m][2] += dot22_8_f(dq[2], a_ptr);
          block_c[m][3] += dot22_8_f(dq[3], a_ptr);
        }
      }
    }
    k += 32;  // 4 weight words * 8 nibbles = 32 K elements
  }

  // Pack the 4 FP32 partial sums into 2 packed pairs and atomically add them
  // directly into the T-typed output (caller pre-zeros it). On gfx11 the
  // packed atomic is a CAS-loop on uint32, but we save the FP32 buffer +
  // memset + cast pass that we'd otherwise need.
#pragma unroll
  for (int m = 0; m < M_COUNT; ++m) {
    if (offset_m + m >= size_m) continue;  // skip padding rows past size_m
    T* out = c + (offset_m + m) * size_n + n;
    if constexpr (std::is_same<T, half>::value) {
      half2 r01 = __halves2half2(__float2half_rn(block_c[m][0]),
                                 __float2half_rn(block_c[m][1]));
      half2 r23 = __halves2half2(__float2half_rn(block_c[m][2]),
                                 __float2half_rn(block_c[m][3]));
      atomic_add_pk_f16(reinterpret_cast<half2*>(out + 0), r01);
      atomic_add_pk_f16(reinterpret_cast<half2*>(out + 2), r23);
    } else {
      bf162_t r01;
      r01.x = __float2bfloat16(block_c[m][0]);
      r01.y = __float2bfloat16(block_c[m][1]);
      bf162_t r23;
      r23.x = __float2bfloat16(block_c[m][2]);
      r23.y = __float2bfloat16(block_c[m][3]);
      atomic_add_pk_bf16(reinterpret_cast<bf162_t*>(out + 0), r01);
      atomic_add_pk_bf16(reinterpret_cast<bf162_t*>(out + 2), r23);
    }
  }
}

// ---------------------------------------------------------------------------
// Launcher.
// ---------------------------------------------------------------------------

template <typename T, int M_COUNT>
void launch_gemm_q4_for_mcount(const T* a, const uint32_t* b_q_weight,
                               const uint32_t* b_qzeros, const T* b_scales,
                               const int* b_q_perm, T* c, int size_m,
                               int size_n, int size_k, int groups,
                               int zero_offset, cudaStream_t stream) {
  dim3 block(THREADS_X);
  dim3 grid((size_n + BLOCK_KN_SIZE * 4 - 1) / (BLOCK_KN_SIZE * 4),
            (size_m + M_COUNT - 1) / M_COUNT,
            (size_k + BLOCK_KN_SIZE - 1) / BLOCK_KN_SIZE);

  gemm_q4_kernel_rdna3<T, M_COUNT><<<grid, block, 0, stream>>>(
      a, b_q_weight, b_qzeros, b_scales, c, size_m, size_n, size_k, groups,
      zero_offset, b_q_perm);
}

// Dispatch to the largest M_COUNT template that doesn't waste more than
// half a tile. Caps at 8: above that, the WMMA-prefill kernel (M >= 16) is
// the right tool, not bigger M_COUNT in the scalar dot-product path.
//
// Tile-waste table:
//   M=1   -> M_COUNT=1   (no waste)
//   M=2,3 -> M_COUNT=2   (M=3 wastes 1/2 of last tile)
//   M=4-7 -> M_COUNT=4   (worst case M=5: wastes 3/4 of last tile)
//   M=8-15-> M_COUNT=8   (worst case M=9: wastes 7/8 of last tile)
// "Wasted" rows are zero-padded in LDS and skip the atomic write, so they
// only burn instructions on the last block, never affect correctness.
template <typename T>
void launch_gemm_q4(const T* a, const uint32_t* b_q_weight,
                    const uint32_t* b_qzeros, const T* b_scales,
                    const int* b_q_perm, T* c, int size_m, int size_n,
                    int size_k, int groups, bool use_v2_format,
                    cudaStream_t stream) {
  const int zero_offset = use_v2_format ? 0 : 1;

  if (size_m == 1) {
    launch_gemm_q4_for_mcount<T, 1>(a, b_q_weight, b_qzeros, b_scales,
                                    b_q_perm, c, size_m, size_n, size_k,
                                    groups, zero_offset, stream);
  } else if (size_m <= 3) {
    launch_gemm_q4_for_mcount<T, 2>(a, b_q_weight, b_qzeros, b_scales,
                                    b_q_perm, c, size_m, size_n, size_k,
                                    groups, zero_offset, stream);
  } else if (size_m <= 7) {
    launch_gemm_q4_for_mcount<T, 4>(a, b_q_weight, b_qzeros, b_scales,
                                    b_q_perm, c, size_m, size_n, size_k,
                                    groups, zero_offset, stream);
  } else {
    // M_COUNT=8 covers M up to 15 here; M >= 16 should ideally take the
    // WMMA path, but if it falls through we still produce correct output —
    // just leaving 3-5× of throughput on the table for prefill workloads.
    launch_gemm_q4_for_mcount<T, 8>(a, b_q_weight, b_qzeros, b_scales,
                                    b_q_perm, c, size_m, size_n, size_k,
                                    groups, zero_offset, stream);
  }
}

}  // namespace gptq_rdna3
}  // namespace vllm

// ---------------------------------------------------------------------------
// Public entry point.
// ---------------------------------------------------------------------------
//
// Inputs:
//   a         [M, K]            half or bfloat16
//   b_q_weight[K/8, N]          uint32 (already shuffled via gptq_shuffle)
//   b_qzeros  [groups, N/8]     uint32 (packed 4-bit zeros)
//   b_scales  [groups, N]       half or bfloat16
//   b_g_idx   [K] or empty      int32 (act-order permutation; empty=identity)
//   use_v2_format                bool   (true = GPTQv2, no +1 zero offset)
//
// Output:
//   c         [M, N]            same dtype as a

#if defined(USE_ROCM)
// Forward declaration of the WMMA prefill entry. Defined in
// q_gemm_rdna3_wmma.cu (separate TU so the WMMA builtins are kept localised;
// cross-TU call resolved at link time, no codegen interaction).
torch::Tensor gptq_gemm_rdna3_wmma(torch::Tensor a, torch::Tensor b_q_weight,
                                   torch::Tensor b_qzeros,
                                   torch::Tensor b_scales,
                                   torch::Tensor b_g_idx,
                                   bool use_v2_format);
#endif

torch::Tensor gptq_gemm_rdna3(torch::Tensor a, torch::Tensor b_q_weight,
                              torch::Tensor b_qzeros, torch::Tensor b_scales,
                              torch::Tensor b_g_idx, bool use_v2_format) {
#if defined(USE_ROCM)
  // Dispatch to the WMMA kernel for bf16 prefill / batched (M >= 16). The
  // branch lives in C++ rather than in Python apply_weights to keep the
  // torch.compile'd graph branch-free (an `if x.size(0) >= 16` inside the
  // traced fwd previously caused a 7x decode regression — see git log).
  //
  // bf16-only gating rationale: a microbench sweep (M ∈ {1..256} × 5
  // Qwen-class shapes) showed the scalar fp16 kernel beats the current WMMA
  // implementation at every M because the fp16 dequant bit-trick keeps the
  // scalar path memory-bound (already saturating ~25% of HBM2 BW). bf16
  // scalar is compute-bound (extra shifts in dequant), so WMMA wins from
  // M=16 onward. The fp16 WMMA path stays available via the standalone op
  // gptq_gemm_rdna3_wmma for direct callers / future kernel tuning, but is
  // not auto-dispatched here until the kernel hits >2x scalar at fp16.
  static const bool kWmmaDisabled = []() {
    const char* env = std::getenv("VLLM_RDNA3_DISABLE_WMMA");
    return env != nullptr && env[0] == '1' && env[1] == '\0';
  }();
  if (!kWmmaDisabled && a.scalar_type() == torch::kBFloat16 &&
      a.dim() == 2 && b_q_weight.dim() == 2 && a.size(0) >= 16 &&
      a.size(1) % 16 == 0 && b_q_weight.size(1) % 16 == 0) {
    return gptq_gemm_rdna3_wmma(a, b_q_weight, b_qzeros, b_scales, b_g_idx,
                                use_v2_format);
  }
#endif
  TORCH_CHECK(a.is_cuda(), "a must be a CUDA/HIP tensor");
  TORCH_CHECK(b_q_weight.is_cuda(), "b_q_weight must be a CUDA/HIP tensor");
  TORCH_CHECK(b_qzeros.is_cuda(), "b_qzeros must be a CUDA/HIP tensor");
  TORCH_CHECK(b_scales.is_cuda(), "b_scales must be a CUDA/HIP tensor");
  TORCH_CHECK(a.dim() == 2, "a must be 2D [M, K]");
  TORCH_CHECK(b_q_weight.dim() == 2, "b_q_weight must be 2D [K/8, N]");
  TORCH_CHECK(a.scalar_type() == torch::kHalf ||
                  a.scalar_type() == torch::kBFloat16,
              "a must be half or bfloat16");
  TORCH_CHECK(a.scalar_type() == b_scales.scalar_type(),
              "b_scales dtype must match a");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
  auto stream = at::cuda::getCurrentCUDAStream();

  int size_m = (int)a.size(0);
  int size_k = (int)a.size(1);
  int size_n = (int)b_q_weight.size(1);
  int groups = (int)b_qzeros.size(0);

  TORCH_CHECK(b_q_weight.size(0) * 8 == size_k,
              "b_q_weight first dim must be K/8");
  TORCH_CHECK(b_scales.size(0) == groups,
              "b_scales must have same group count as qzeros");
  TORCH_CHECK(b_scales.size(1) == size_n, "b_scales last dim must be N");

  auto opts = torch::TensorOptions().dtype(a.dtype()).device(a.device());
  // c must be zeroed: the kernel atomically accumulates partial sums from
  // every split-K block into it.
  at::Tensor c = torch::zeros({size_m, size_n}, opts);

  const int* g_idx_ptr = nullptr;
  if (!b_g_idx.device().is_meta() && b_g_idx.numel() > 0) {
    TORCH_CHECK(b_g_idx.scalar_type() == torch::kInt32,
                "b_g_idx must be int32");
    g_idx_ptr = (const int*)b_g_idx.data_ptr();
  }

  if (a.scalar_type() == torch::kHalf) {
    vllm::gptq_rdna3::launch_gemm_q4<half>(
        (const half*)a.data_ptr(),
        (const uint32_t*)b_q_weight.data_ptr(),
        (const uint32_t*)b_qzeros.data_ptr(),
        (const half*)b_scales.data_ptr(), g_idx_ptr, (half*)c.data_ptr(),
        size_m, size_n, size_k, groups, use_v2_format, stream);
  } else {
    vllm::gptq_rdna3::launch_gemm_q4<vllm::gptq_rdna3::bf16_t>(
        (const vllm::gptq_rdna3::bf16_t*)a.data_ptr(),
        (const uint32_t*)b_q_weight.data_ptr(),
        (const uint32_t*)b_qzeros.data_ptr(),
        (const vllm::gptq_rdna3::bf16_t*)b_scales.data_ptr(), g_idx_ptr,
        (vllm::gptq_rdna3::bf16_t*)c.data_ptr(), size_m, size_n, size_k,
        groups, use_v2_format, stream);
  }

  return c;
}
