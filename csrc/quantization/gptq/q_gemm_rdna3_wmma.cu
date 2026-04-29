// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// W4A16 GPTQ WMMA prefill kernel for AMD RDNA3 (gfx1100). This is the
// matrix-instruction path for M >= 16; small-M decode lives in the sibling
// file q_gemm_rdna3.cu and is exposed via a different op
// (`gptq_gemm_rdna3`). Keeping the two paths in separate translation units
// is intentional: an earlier attempt at putting WMMA in the same TU as the
// scalar dot-product kernel introduced a compile-time interaction that
// silently miscompiled the M=1 path even though the WMMA template was never
// instantiated for M=1. Hipcc's optimizer appears to scope some decisions
// at the TU level (likely register file / SGPR pressure heuristics across
// all kernels in the TU), so we isolate.
//
// Hardware notes (RDNA3 / gfx1100 / RX 7900 XTX):
//   * v_wmma_f32_16x16x16_{f16,bf16}_w32 — 16×16×16 GEMM in one instruction
//     (~16 cycles per WMMA on wave32). Accumulator dtype is FP32; inputs
//     are fp16 or bf16. There is NO 16x16x16 with fp16/bf16 accumulator on
//     gfx11 that we'd want here (we always need fp32 accum to avoid loss
//     across many K iterations).
//   * Wave32 fragment storage is "doubled" — lanes 16..31 hold a copy of
//     lanes 0..15 for the input fragments. The output C fragment splits
//     across the wave: lanes 0..15 hold columns 0..7 of one row each,
//     lanes 16..31 hold columns 8..15.
//   * No native v_global_atomic_pk_add_{f16,bf16} on gfx11; we sidestep
//     this entirely by giving each block exclusive ownership of its
//     16M × 16N output tile (no split-K, no atomic).

#include <cstdint>
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
namespace gptq_rdna3_wmma {

#if defined(USE_ROCM)

// Pull dequant types from the sibling namespace.
using vllm::gptq_rdna3::bf16_t;
using vllm::gptq_rdna3::bf162_t;

// PRECISE dequant variants live HERE, not in the shared qdq_4_rdna3.cuh
// header. Reason: hipcc takes different register/scheduling decisions when
// the shared header grows, which caused a measurable decode-tk/s regression
// in the scalar kernel even when these new functions were never called from
// it. Keeping them in the WMMA TU only restores scalar's binary identity
// to its tuned baseline.
//
// Numerics: the classic fp16 bit-trick (FMA form: q*scale + (-(1024+zero)*scale))
// loses up to ~0.025 per cell at scale=0.1 because fp16(scale) ≠ scale and
// the FMA amplifies that by 1024× without cancelling against the precomputed
// (1024+zero)*scale (which is rounded to fp16 BEFORE the FMA).
//
// Fix: subtract (1024+zero) as an integer FIRST — exact in fp16 because
// integers in [1024, 2047] are exactly representable — then multiply by
// scale, incurring at most one half-ULP rounding. Costs one extra
// instruction per dequant pair (sub+mul vs single FMA), worth it for the
// WMMA path because K can be small (16) and errors don't average.
__forceinline__ __device__ void prep_zero_scale_fp16_precise(uint32_t zero,
                                                             half scale,
                                                             half2& z_prep,
                                                             half2& y_prep) {
  union { uint16_t u; half h; } zu;
  zu.u = (uint16_t)(0x6400 | zero);
  z_prep = __half2half2(zu.h);
  y_prep = __half2half2(scale);
}

__forceinline__ __device__ void dequant_4bit_8_fp16_precise(uint32_t qa,
                                                            half2 (&dq)[4],
                                                            half2 z_prep,
                                                            half2 y_prep) {
  const uint32_t c0 = 0x64006400;
  union { uint32_t u; half2 h2; } q0, q1, q2, q3;
  q0.u = ((qa >>  0) & 0x000F000F) | c0;
  q1.u = ((qa >>  4) & 0x000F000F) | c0;
  q2.u = ((qa >>  8) & 0x000F000F) | c0;
  q3.u = ((qa >> 12) & 0x000F000F) | c0;
  dq[0] = __hmul2(__hsub2(q0.h2, z_prep), y_prep);
  dq[1] = __hmul2(__hsub2(q1.h2, z_prep), y_prep);
  dq[2] = __hmul2(__hsub2(q2.h2, z_prep), y_prep);
  dq[3] = __hmul2(__hsub2(q3.h2, z_prep), y_prep);
}

__forceinline__ __device__ void prep_zero_scale_bf16_precise(uint32_t zero,
                                                             bf16_t scale,
                                                             bf162_t& z_prep,
                                                             bf162_t& y_prep) {
  union { uint16_t u; bf16_t h; } zu;
  zu.u = (uint16_t)(0x4300 | zero);
  z_prep = __bfloat162bfloat162(zu.h);
  y_prep = __bfloat162bfloat162(scale);
}

__forceinline__ __device__ void dequant_4bit_8_bf16_precise(uint32_t qa,
                                                            bf162_t (&dq)[4],
                                                            bf162_t z_prep,
                                                            bf162_t y_prep) {
  const uint32_t c0 = 0x43004300;
  union { uint32_t u; bf162_t b2; } q0, q1, q2, q3;
  q0.u = ((qa >>  0) & 0x000F000F) | c0;
  q1.u = ((qa >>  4) & 0x000F000F) | c0;
  q2.u = ((qa >>  8) & 0x000F000F) | c0;
  q3.u = ((qa >> 12) & 0x000F000F) | c0;
  dq[0] = __hmul2(__hsub2(q0.b2, z_prep), y_prep);
  dq[1] = __hmul2(__hsub2(q1.b2, z_prep), y_prep);
  dq[2] = __hmul2(__hsub2(q2.b2, z_prep), y_prep);
  dq[3] = __hmul2(__hsub2(q3.b2, z_prep), y_prep);
}

// Native AMDGPU vector types expected by the WMMA built-ins.
using v16fp16 = _Float16 __attribute__((ext_vector_type(16)));
using v16bf16 = __bf16 __attribute__((ext_vector_type(16)));
using v8fp32 = float __attribute__((ext_vector_type(8)));

__device__ __forceinline__ v8fp32 wmma_mma(v16fp16 a, v16fp16 b, v8fp32 c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}
__device__ __forceinline__ v8fp32 wmma_mma(v16bf16 a, v16bf16 b, v8fp32 c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c);
}

// Map HIP wrapper types (half, __hip_bfloat16) to native compiler types
// (_Float16, __bf16) used by the WMMA built-ins. Bitcast is a register
// reinterpret in practice.
template <typename T>
struct WmmaNative;
template <>
struct WmmaNative<half> {
  using elem = _Float16;
  using v16 = v16fp16;
};
template <>
struct WmmaNative<bf16_t> {
  using elem = __bf16;
  using v16 = v16bf16;
};

template <typename FROM, typename TO>
__device__ __forceinline__ TO bitcast_elem(FROM x) {
  static_assert(sizeof(FROM) == sizeof(TO),
                "bitcast_elem requires equal-sized types");
  TO r;
  __builtin_memcpy(&r, &x, sizeof(TO));
  return r;
}

// Per-T tzero (matches the helper in the scalar TU).
template <typename T>
__device__ __forceinline__ T tzero();
template <>
__device__ __forceinline__ half tzero<half>() {
  return __float2half_rn(0.0f);
}
template <>
__device__ __forceinline__ bf16_t tzero<bf16_t>() {
  return __float2bfloat16(0.0f);
}

// ===========================================================================
// WMMA kernel: 16M × 16N tile per block, 1 wave, full K traversal.
//
// Wave32 fragment layout (verified empirically with all-modes diagnostic
// against a random A and B and the eight candidate output mappings):
//
//   * A frag (row-major in M, K in slot):
//       lane t, slot i → A[lane_lo][k = i]
//       Lane axis encodes M (A's row), slot encodes K.
//   * B frag (col-major in N, K in slot):
//       lane t, slot i → B[k = i][lane_lo]
//       Lane axis encodes N (B's column), slot encodes K. K-axis aligns
//       with A's K-axis (same slot index).
//   * C frag (output, lane = N, slot = M with hi-bit interleave):
//       lane t, slot i → C[m = 2*i + lane_hi][n = lane_lo]
//       Lane axis encodes N (C's column). Each lane holds 8 elements of
//       its output column, alternating rows: lanes 0..15 (hi=0) hold even
//       rows m=0,2,4,...,14; lanes 16..31 (hi=1) hold odd rows
//       m=1,3,5,...,15.
//   * Both halves of the wave (lanes 0..15 and 16..31) hold IDENTICAL input
//     fragments (AMD's "doubled" wave32 input layout). Output is split
//     between halves via lane_hi.
//
// History note: an earlier version of this kernel loaded B row-major in K
// and assumed the output was C[lane_lo][2*i+lane_hi]. That layout passes
// all-A=identity tests because A=I makes the K-axis sum collapse, but
// implements C = A @ B^T for non-trivial A — the bug only shows up against
// random A. The probe op gptq_gemm_rdna3_wmma_probe iterates all four
// {row,col} × {row,col} loadings and identifies mode 1 (A row, B col) with
// output [m=2*i+hi][n=lane_lo] as the unique mapping that yields A @ B.
// ===========================================================================

template <typename T>
__global__ void gemm_q4_wmma_kernel(const T* __restrict__ a,
                                    const uint32_t* __restrict__ b_q,
                                    const uint32_t* __restrict__ b_qzeros,
                                    const T* __restrict__ b_scales,
                                    T* __restrict__ c, const int size_m,
                                    const int size_n, const int size_k,
                                    const int groups, const int zero_offset,
                                    const int* __restrict__ b_q_perm) {
  using E = typename WmmaNative<T>::elem;
  using V16 = typename WmmaNative<T>::v16;

  const int m_tile = blockIdx.y * 16;
  const int n_tile = blockIdx.x * 16;
  if (m_tile >= size_m || n_tile >= size_n) return;

  const int lane = threadIdx.x;   // 0..31
  const int lane_lo = lane & 15;  // row index within fragment
  const int lane_hi = lane >> 4;  // 0 or 1

  v8fp32 c_acc = {0, 0, 0, 0, 0, 0, 0, 0};

  const int groupsize = size_k / groups;

  // LDS tile of dequantized B. 16 K rows × 16 N cols.
  __shared__ T b_lds[16][16];

  for (int k_tile = 0; k_tile < size_k; k_tile += 16) {
    // ---- Dequant 16x16 B tile into LDS ----
    // 32 lanes split 16 N cols × 2 K-octets per col = 32 dequant tasks.
    const int my_n = lane_lo;
    const int my_k_octet = lane_hi;  // 0 → K[0..7], 1 → K[8..15]
    const int actual_n = n_tile + my_n;

    if (actual_n < size_n) {
      const int qk_row = (k_tile / 8) + my_k_octet;
      const uint32_t qa = b_q[qk_row * size_n + actual_n];

      const int g = k_tile / groupsize;
      const int qz_idx = g * (size_n / 8) + actual_n / 8;
      const int qz_shift = (actual_n & 7) * 4;
      const uint32_t zero_v =
          ((b_qzeros[qz_idx] >> qz_shift) & 0xF) + (uint32_t)zero_offset;
      const T scale_t = b_scales[g * size_n + actual_n];

      const int k_base = my_k_octet * 8;

      if constexpr (std::is_same<T, half>::value) {
        half2 z_prep, y_prep;
        prep_zero_scale_fp16_precise(zero_v, scale_t, z_prep, y_prep);
        half2 dq[4];
        dequant_4bit_8_fp16_precise(qa, dq, z_prep, y_prep);
        b_lds[k_base + 0][my_n] = __low2half(dq[0]);
        b_lds[k_base + 1][my_n] = __high2half(dq[0]);
        b_lds[k_base + 2][my_n] = __low2half(dq[1]);
        b_lds[k_base + 3][my_n] = __high2half(dq[1]);
        b_lds[k_base + 4][my_n] = __low2half(dq[2]);
        b_lds[k_base + 5][my_n] = __high2half(dq[2]);
        b_lds[k_base + 6][my_n] = __low2half(dq[3]);
        b_lds[k_base + 7][my_n] = __high2half(dq[3]);
      } else {
        bf162_t z_b, y_b;
        prep_zero_scale_bf16_precise(zero_v, scale_t, z_b, y_b);
        bf162_t dq[4];
        dequant_4bit_8_bf16_precise(qa, dq, z_b, y_b);
        b_lds[k_base + 0][my_n] = dq[0].x;
        b_lds[k_base + 1][my_n] = dq[0].y;
        b_lds[k_base + 2][my_n] = dq[1].x;
        b_lds[k_base + 3][my_n] = dq[1].y;
        b_lds[k_base + 4][my_n] = dq[2].x;
        b_lds[k_base + 5][my_n] = dq[2].y;
        b_lds[k_base + 6][my_n] = dq[3].x;
        b_lds[k_base + 7][my_n] = dq[3].y;
      }
    }

    __syncthreads();

    // ---- Build A and B fragments, run WMMA ----
    V16 a_frag, b_frag;
    const int m_row = m_tile + lane_lo;

    if (m_row < size_m) {
      const T* a_row = a + m_row * size_k;
#pragma unroll
      for (int i = 0; i < 16; i++) {
        T v;
        if (k_tile + i < size_k) {
          v = b_q_perm ? a_row[b_q_perm[k_tile + i]] : a_row[k_tile + i];
        } else {
          v = tzero<T>();
        }
        a_frag[i] = bitcast_elem<T, E>(v);
      }
    } else {
#pragma unroll
      for (int i = 0; i < 16; i++) a_frag[i] = (E)0;
    }

    // B fragment: lane t holds COLUMN n=lane_lo of the B tile (K-axis in
    // slot, N-axis in lane). This is the AMD WMMA convention for the right
    // operand of a matrix multiply — K-axis aligns with A's K-axis (also
    // in slot), enabling per-lane inner products.
#pragma unroll
    for (int i = 0; i < 16; i++) {
      b_frag[i] = bitcast_elem<T, E>(b_lds[i][lane_lo]);
    }

#ifdef VLLM_WMMA_LAYOUT_DEBUG
    // Diagnostic: skip WMMA, force c_acc to encode (lane, slot) so the
    // store pattern reveals the C-output lane→matrix mapping. Compile with
    // -DVLLM_WMMA_LAYOUT_DEBUG to enable. Output: c[m][n] = lane + slot/16.
    (void)a_frag;
    (void)b_frag;
#pragma unroll
    for (int i = 0; i < 8; i++) {
      c_acc[i] = (float)lane + (float)i / 16.0f;
    }
    // Run only one K iteration in debug mode so c_acc isn't overwritten.
    if (k_tile == 0) {
      k_tile = size_k;  // exit loop on next check
    }
#else
    c_acc = wmma_mma(a_frag, b_frag, c_acc);
#endif

    __syncthreads();  // before next iter overwrites b_lds
  }

  // ---- Store C ----
  // Lane t holds column n=lane_lo of the output tile, 8 rows determined by
  // slot i and lane_hi:
  //   lane_hi == 0  →  rows 0, 2, 4, ..., 14   (even rows)
  //   lane_hi == 1  →  rows 1, 3, 5, ..., 15   (odd rows)
  // c_acc[i] corresponds to actual row m = 2*i + lane_hi at column lane_lo.
  const int out_n = n_tile + lane_lo;

  if (out_n < size_n) {
#pragma unroll
    for (int i = 0; i < 8; i++) {
      const int out_m = m_tile + 2 * i + lane_hi;
      if (out_m < size_m) {
        T* dst = c + out_m * size_n + out_n;
        if constexpr (std::is_same<T, half>::value) {
          *dst = __float2half_rn(c_acc[i]);
        } else {
          *dst = __float2bfloat16(c_acc[i]);
        }
      }
    }
  }
}

// ===========================================================================
// V2 kernel: 4 waves per block, 16M × 64N output tile.
//
// Same WMMA fragment contract as v1 (mode 1 layout: A row-major, B col-major,
// output [m=2*i+lane_hi][n=lane_lo]). Difference is purely organisational:
//   * 4 waves cooperate on a single block, each owning a 16-col N stripe.
//   * B dequant is split block-wide: 128 threads × 1 task each cover the
//     64 N cols × 2 K-octets = 128 dequant tasks per K iter.
//   * A is reloaded by each wave from global memory; with all 4 waves on
//     the same WGP (and therefore the same L0 cache), 3 of 4 reads should
//     hit L0 → ~1.25x effective A bandwidth, not 4x.
//
// Why this should beat v1: v1 launches 1 wave per block, so for the
// typical Qwen prefill shape (M=16, N=4096) the grid is only 256 blocks,
// running 2-3 waves per CU on a 96-CU GPU — well below the ~8 waves/CU
// needed to hide LDS / global latency. v2 quadruples the work per block
// without changing block count significantly (it does drop blocks 4× in
// the N direction, but each does 4× the work; net is more useful waves
// in flight per CU at small M).
//
// Selectable at runtime via VLLM_RDNA3_WMMA_V2=1; default keeps v1 so
// production bf16 prefill is unaffected until v2 is validated.
// ===========================================================================
template <typename T>
__global__ void gemm_q4_wmma_kernel_v2(
    const T* __restrict__ a, const uint32_t* __restrict__ b_q,
    const uint32_t* __restrict__ b_qzeros, const T* __restrict__ b_scales,
    T* __restrict__ c, const int size_m, const int size_n, const int size_k,
    const int groups, const int zero_offset,
    const int* __restrict__ b_q_perm) {
  using E = typename WmmaNative<T>::elem;
  using V16 = typename WmmaNative<T>::v16;

  const int wave_id = threadIdx.x >> 5;  // 0..3
  const int lane = threadIdx.x & 31;
  const int lane_lo = lane & 15;
  const int lane_hi = lane >> 4;

  const int m_tile = blockIdx.y * 16;
  const int n_tile_block = blockIdx.x * 64;
  // Block-level early-out: if BOTH dims out of range, every wave bails out
  // before touching __syncthreads(). Per-wave/per-lane bounds (n_tile + ...
  // < size_n) are handled below without divergent syncs.
  if (m_tile >= size_m || n_tile_block >= size_n) return;

  const int wave_n_base = wave_id * 16;        // wave's offset within block tile
  const int n_tile_wave = n_tile_block + wave_n_base;

  v8fp32 c_acc = {0, 0, 0, 0, 0, 0, 0, 0};

  const int groupsize = size_k / groups;

  // 16K × 64N B tile in LDS. Each wave writes its own 16-col stripe.
  __shared__ T b_lds[16][64];

  for (int k_tile = 0; k_tile < size_k; k_tile += 16) {
    // ---- Cooperative B dequant: 128 threads, 1 task each ----
    // Per-block: 64 N cols × 2 K-octets = 128 tasks. Each wave's 32 lanes
    // dequant 16 cols × 2 octets within its N stripe.
    const int my_n_local = wave_n_base + lane_lo;  // 0..63 inside block
    const int my_k_octet = lane_hi;
    const int actual_n = n_tile_block + my_n_local;

    if (actual_n < size_n) {
      const int qk_row = (k_tile / 8) + my_k_octet;
      const uint32_t qa = b_q[qk_row * size_n + actual_n];

      const int g = k_tile / groupsize;
      const int qz_idx = g * (size_n / 8) + actual_n / 8;
      const int qz_shift = (actual_n & 7) * 4;
      const uint32_t zero_v =
          ((b_qzeros[qz_idx] >> qz_shift) & 0xF) + (uint32_t)zero_offset;
      const T scale_t = b_scales[g * size_n + actual_n];

      const int k_base = my_k_octet * 8;

      if constexpr (std::is_same<T, half>::value) {
        half2 z_prep, y_prep;
        prep_zero_scale_fp16_precise(zero_v, scale_t, z_prep, y_prep);
        half2 dq[4];
        dequant_4bit_8_fp16_precise(qa, dq, z_prep, y_prep);
        b_lds[k_base + 0][my_n_local] = __low2half(dq[0]);
        b_lds[k_base + 1][my_n_local] = __high2half(dq[0]);
        b_lds[k_base + 2][my_n_local] = __low2half(dq[1]);
        b_lds[k_base + 3][my_n_local] = __high2half(dq[1]);
        b_lds[k_base + 4][my_n_local] = __low2half(dq[2]);
        b_lds[k_base + 5][my_n_local] = __high2half(dq[2]);
        b_lds[k_base + 6][my_n_local] = __low2half(dq[3]);
        b_lds[k_base + 7][my_n_local] = __high2half(dq[3]);
      } else {
        bf162_t z_b, y_b;
        prep_zero_scale_bf16_precise(zero_v, scale_t, z_b, y_b);
        bf162_t dq[4];
        dequant_4bit_8_bf16_precise(qa, dq, z_b, y_b);
        b_lds[k_base + 0][my_n_local] = dq[0].x;
        b_lds[k_base + 1][my_n_local] = dq[0].y;
        b_lds[k_base + 2][my_n_local] = dq[1].x;
        b_lds[k_base + 3][my_n_local] = dq[1].y;
        b_lds[k_base + 4][my_n_local] = dq[2].x;
        b_lds[k_base + 5][my_n_local] = dq[2].y;
        b_lds[k_base + 6][my_n_local] = dq[3].x;
        b_lds[k_base + 7][my_n_local] = dq[3].y;
      }
    }

    __syncthreads();

    // ---- Build A frag (per-wave from global; L0 cache on the WGP serves
    // 3 of 4 waves' duplicate reads) and B frag (per-wave from LDS stripe).
    V16 a_frag, b_frag;
    const int m_row = m_tile + lane_lo;

    if (m_row < size_m) {
      const T* a_row = a + m_row * size_k;
#pragma unroll
      for (int i = 0; i < 16; i++) {
        T v;
        if (k_tile + i < size_k) {
          v = b_q_perm ? a_row[b_q_perm[k_tile + i]] : a_row[k_tile + i];
        } else {
          v = tzero<T>();
        }
        a_frag[i] = bitcast_elem<T, E>(v);
      }
    } else {
#pragma unroll
      for (int i = 0; i < 16; i++) a_frag[i] = (E)0;
    }

#pragma unroll
    for (int i = 0; i < 16; i++) {
      b_frag[i] = bitcast_elem<T, E>(b_lds[i][wave_n_base + lane_lo]);
    }

    c_acc = wmma_mma(a_frag, b_frag, c_acc);

    __syncthreads();  // before next iter overwrites b_lds
  }

  // ---- Store ----
  const int out_n = n_tile_wave + lane_lo;
  if (out_n < size_n) {
#pragma unroll
    for (int i = 0; i < 8; i++) {
      const int out_m = m_tile + 2 * i + lane_hi;
      if (out_m < size_m) {
        T* dst = c + out_m * size_n + out_n;
        if constexpr (std::is_same<T, half>::value) {
          *dst = __float2half_rn(c_acc[i]);
        } else {
          *dst = __float2bfloat16(c_acc[i]);
        }
      }
    }
  }
}

template <typename T>
void launch_gemm_q4_wmma(const T* a, const uint32_t* b_q_weight,
                         const uint32_t* b_qzeros, const T* b_scales,
                         const int* b_q_perm, T* c, int size_m, int size_n,
                         int size_k, int groups, int zero_offset,
                         cudaStream_t stream) {
  // VLLM_RDNA3_WMMA_V2=1 selects the 4-wave-per-block kernel. Default = v1
  // so production bf16 prefill keeps the validated path until v2 is
  // confirmed bit-exact + faster end-to-end.
  static const bool use_v2 = []() {
    const char* env = std::getenv("VLLM_RDNA3_WMMA_V2");
    return env != nullptr && env[0] == '1' && env[1] == '\0';
  }();

  if (use_v2) {
    dim3 block(128);  // 4 waves
    dim3 grid((size_n + 63) / 64, (size_m + 15) / 16, 1);
    gemm_q4_wmma_kernel_v2<T><<<grid, block, 0, stream>>>(
        a, b_q_weight, b_qzeros, b_scales, c, size_m, size_n, size_k, groups,
        zero_offset, b_q_perm);
  } else {
    dim3 block(32);  // 1 wave
    dim3 grid((size_n + 15) / 16, (size_m + 15) / 16, 1);
    gemm_q4_wmma_kernel<T><<<grid, block, 0, stream>>>(
        a, b_q_weight, b_qzeros, b_scales, c, size_m, size_n, size_k, groups,
        zero_offset, b_q_perm);
  }
}

// ===========================================================================
// Layout probe kernel — diagnostic only.
// Takes fp16 A[16,16] and B[16,16] directly (no dequant). Loads them into
// the WMMA fragments under one of several layout hypotheses, runs WMMA, and
// dumps c_acc[i] per lane to a flat fp32 output[32 * 8].
// Mode selects how A and B are loaded:
//   0: A row-major (a_frag[i] = A[lane_lo][i]),
//      B row-major (b_frag[i] = B[lane_lo][i])
//   1: A row-major,
//      B col-major (b_frag[i] = B[i][lane_lo])
//   2: A col-major (a_frag[i] = A[i][lane_lo]),
//      B row-major
//   3: A col-major,
//      B col-major
// With A = identity (16x16) and B with unique per-cell values, decoding the
// dump tells us which C cell each c_acc[lane][i] holds.
// ===========================================================================

// Full-kernel dump probe: same code path as gemm_q4_wmma_kernel for ONE 16x16
// output tile (block (0,0)) — dequant + LDS + WMMA — but writes c_acc[lane][i]
// directly to a flat fp32 [32 * 8] dump instead of doing the interleaved
// row store. Used to isolate whether c_acc is correct (store-mapping bug)
// vs c_acc itself wrong (dequant/LDS/fragment-load bug).
__global__ void gemm_q4_wmma_dump_kernel(
    const half* __restrict__ a, const uint32_t* __restrict__ b_q,
    const uint32_t* __restrict__ b_qzeros, const half* __restrict__ b_scales,
    float* __restrict__ dump_out, const int size_m, const int size_n,
    const int size_k, const int groups, const int zero_offset) {
  const int lane = threadIdx.x;
  const int lane_lo = lane & 15;
  const int lane_hi = lane >> 4;

  v8fp32 c_acc = {0, 0, 0, 0, 0, 0, 0, 0};
  const int groupsize = size_k / groups;
  __shared__ half b_lds[16][16];

  for (int k_tile = 0; k_tile < size_k; k_tile += 16) {
    const int my_n = lane_lo;
    const int my_k_octet = lane_hi;
    const int actual_n = my_n;

    if (actual_n < size_n) {
      const int qk_row = (k_tile / 8) + my_k_octet;
      const uint32_t qa = b_q[qk_row * size_n + actual_n];

      const int g = k_tile / groupsize;
      const int qz_idx = g * (size_n / 8) + actual_n / 8;
      const int qz_shift = (actual_n & 7) * 4;
      const uint32_t zero_v =
          ((b_qzeros[qz_idx] >> qz_shift) & 0xF) + (uint32_t)zero_offset;
      const half scale_t = b_scales[g * size_n + actual_n];

      const int k_base = my_k_octet * 8;

      half2 z_prep, y_prep;
      prep_zero_scale_fp16_precise(zero_v, scale_t, z_prep, y_prep);
      half2 dq[4];
      dequant_4bit_8_fp16_precise(qa, dq, z_prep, y_prep);
      b_lds[k_base + 0][my_n] = __low2half(dq[0]);
      b_lds[k_base + 1][my_n] = __high2half(dq[0]);
      b_lds[k_base + 2][my_n] = __low2half(dq[1]);
      b_lds[k_base + 3][my_n] = __high2half(dq[1]);
      b_lds[k_base + 4][my_n] = __low2half(dq[2]);
      b_lds[k_base + 5][my_n] = __high2half(dq[2]);
      b_lds[k_base + 6][my_n] = __low2half(dq[3]);
      b_lds[k_base + 7][my_n] = __high2half(dq[3]);
    }

    __syncthreads();

    v16fp16 a_frag, b_frag;
    const int m_row = lane_lo;

    if (m_row < size_m) {
      const half* a_row = a + m_row * size_k;
#pragma unroll
      for (int i = 0; i < 16; i++) {
        half v = (k_tile + i < size_k) ? a_row[k_tile + i] : __float2half_rn(0.0f);
        a_frag[i] = bitcast_elem<half, _Float16>(v);
      }
    } else {
#pragma unroll
      for (int i = 0; i < 16; i++) a_frag[i] = (_Float16)0;
    }

#pragma unroll
    for (int i = 0; i < 16; i++) {
      b_frag[i] = bitcast_elem<half, _Float16>(b_lds[i][lane_lo]);
    }

    c_acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_frag, b_frag, c_acc);

    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < 8; i++) {
    dump_out[lane * 8 + i] = c_acc[i];
  }
}

// LDS-content probe. Runs the same dequant + LDS-write path as the GEMM
// kernel for a single 16x16 B tile, then has lane 0 dump all 256 b_lds cells
// to global memory after __syncthreads. If the dump matches the expected
// dequantized B[k][n], then dequant + LDS-write are correct and the bug is
// in fragment load (b_frag = b_lds[lane_lo][i]) or the WMMA instruction.
__global__ void wmma_lds_check_kernel(const uint32_t* __restrict__ b_q,
                                      const uint32_t* __restrict__ b_qzeros,
                                      const half* __restrict__ b_scales,
                                      half* __restrict__ lds_dump,
                                      const int size_n, const int size_k,
                                      const int groups,
                                      const int zero_offset) {
  const int lane = threadIdx.x;
  const int lane_lo = lane & 15;
  const int lane_hi = lane >> 4;

  const int groupsize = size_k / groups;
  __shared__ half b_lds[16][16];

  const int k_tile = 0;
  const int my_n = lane_lo;
  const int my_k_octet = lane_hi;
  const int actual_n = my_n;

  if (actual_n < size_n) {
    const int qk_row = (k_tile / 8) + my_k_octet;
    const uint32_t qa = b_q[qk_row * size_n + actual_n];

    const int g = k_tile / groupsize;
    const int qz_idx = g * (size_n / 8) + actual_n / 8;
    const int qz_shift = (actual_n & 7) * 4;
    const uint32_t zero_v =
        ((b_qzeros[qz_idx] >> qz_shift) & 0xF) + (uint32_t)zero_offset;
    const half scale_t = b_scales[g * size_n + actual_n];

    const int k_base = my_k_octet * 8;

    half2 z_prep, y_prep;
    prep_zero_scale_fp16_precise(zero_v, scale_t, z_prep, y_prep);
    half2 dq[4];
    dequant_4bit_8_fp16_precise(qa, dq, z_prep, y_prep);
    b_lds[k_base + 0][my_n] = __low2half(dq[0]);
    b_lds[k_base + 1][my_n] = __high2half(dq[0]);
    b_lds[k_base + 2][my_n] = __low2half(dq[1]);
    b_lds[k_base + 3][my_n] = __high2half(dq[1]);
    b_lds[k_base + 4][my_n] = __low2half(dq[2]);
    b_lds[k_base + 5][my_n] = __high2half(dq[2]);
    b_lds[k_base + 6][my_n] = __low2half(dq[3]);
    b_lds[k_base + 7][my_n] = __high2half(dq[3]);
  }

  __syncthreads();

  if (lane == 0) {
    for (int k = 0; k < 16; k++) {
      for (int n = 0; n < 16; n++) {
        lds_dump[k * 16 + n] = b_lds[k][n];
      }
    }
  }
}

__global__ void wmma_layout_probe_kernel(const half* a_in, const half* b_in,
                                         float* dump_out, int mode) {
  const int lane = threadIdx.x;
  const int lane_lo = lane & 15;

  v16fp16 a_frag, b_frag;
#pragma unroll
  for (int i = 0; i < 16; i++) {
    half av, bv;
    if (mode == 0) {
      av = a_in[lane_lo * 16 + i];
      bv = b_in[lane_lo * 16 + i];
    } else if (mode == 1) {
      av = a_in[lane_lo * 16 + i];
      bv = b_in[i * 16 + lane_lo];
    } else if (mode == 2) {
      av = a_in[i * 16 + lane_lo];
      bv = b_in[lane_lo * 16 + i];
    } else {
      av = a_in[i * 16 + lane_lo];
      bv = b_in[i * 16 + lane_lo];
    }
    a_frag[i] = bitcast_elem<half, _Float16>(av);
    b_frag[i] = bitcast_elem<half, _Float16>(bv);
  }

  v8fp32 c = {0, 0, 0, 0, 0, 0, 0, 0};
  c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_frag, b_frag, c);

#pragma unroll
  for (int i = 0; i < 8; i++) {
    dump_out[lane * 8 + i] = c[i];
  }
}

#endif  // USE_ROCM

}  // namespace gptq_rdna3_wmma
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
//   use_v2_format               bool   (true = GPTQv2, no +1 zero offset)
//
// Output:
//   c         [M, N]            same dtype as a
//
// Requirements:
//   * size_m >= 16 (otherwise prefer the scalar gptq_gemm_rdna3 op)
//   * size_n % 16 == 0 (WMMA tile size)
//   * size_k % 16 == 0 (WMMA tile size)

torch::Tensor gptq_gemm_rdna3_wmma(torch::Tensor a, torch::Tensor b_q_weight,
                                   torch::Tensor b_qzeros,
                                   torch::Tensor b_scales,
                                   torch::Tensor b_g_idx,
                                   bool use_v2_format) {
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
  TORCH_CHECK(size_n % 16 == 0, "WMMA path requires N % 16 == 0");
  TORCH_CHECK(size_k % 16 == 0, "WMMA path requires K % 16 == 0");

  auto opts = torch::TensorOptions().dtype(a.dtype()).device(a.device());
  // No atomics, so no zero-init needed.
  at::Tensor c = torch::empty({size_m, size_n}, opts);

  const int* g_idx_ptr = nullptr;
  if (!b_g_idx.device().is_meta() && b_g_idx.numel() > 0) {
    TORCH_CHECK(b_g_idx.scalar_type() == torch::kInt32,
                "b_g_idx must be int32");
    g_idx_ptr = (const int*)b_g_idx.data_ptr();
  }

  const int zero_offset = use_v2_format ? 0 : 1;

#if defined(USE_ROCM)
  if (a.scalar_type() == torch::kHalf) {
    vllm::gptq_rdna3_wmma::launch_gemm_q4_wmma<half>(
        (const half*)a.data_ptr(),
        (const uint32_t*)b_q_weight.data_ptr(),
        (const uint32_t*)b_qzeros.data_ptr(),
        (const half*)b_scales.data_ptr(), g_idx_ptr, (half*)c.data_ptr(),
        size_m, size_n, size_k, groups, zero_offset, stream);
  } else {
    vllm::gptq_rdna3_wmma::launch_gemm_q4_wmma<
        vllm::gptq_rdna3_wmma::bf16_t>(
        (const vllm::gptq_rdna3_wmma::bf16_t*)a.data_ptr(),
        (const uint32_t*)b_q_weight.data_ptr(),
        (const uint32_t*)b_qzeros.data_ptr(),
        (const vllm::gptq_rdna3_wmma::bf16_t*)b_scales.data_ptr(), g_idx_ptr,
        (vllm::gptq_rdna3_wmma::bf16_t*)c.data_ptr(), size_m, size_n, size_k,
        groups, zero_offset, stream);
  }
#else
  TORCH_CHECK(false,
              "gptq_gemm_rdna3_wmma is only available on ROCm (gfx11)");
#endif

  return c;
}

// Diagnostic probe: takes raw fp16 A[16,16] and B[16,16], runs a single WMMA
// under the requested fragment-load hypothesis, returns fp32 [32, 8] dump of
// c_acc[lane][i] per lane. Used to empirically identify the wave32 fragment
// layout. Not for production use.
// Diagnostic: runs the FULL gemm_q4_wmma path (dequant + LDS + WMMA) for a
// single 16x16 output tile, dumps c_acc per lane to fp32 [32, 8]. Tells us
// whether c_acc itself is correct or whether the bug is in the store mapping.
torch::Tensor gptq_gemm_rdna3_wmma_dump(torch::Tensor a, torch::Tensor b_q_weight,
                                        torch::Tensor b_qzeros,
                                        torch::Tensor b_scales,
                                        bool use_v2_format) {
  TORCH_CHECK(a.is_cuda() && b_q_weight.is_cuda() && b_qzeros.is_cuda() &&
              b_scales.is_cuda(), "all tensors must be CUDA");
  TORCH_CHECK(a.scalar_type() == torch::kHalf, "a must be fp16");
  TORCH_CHECK(b_scales.scalar_type() == torch::kHalf, "b_scales must be fp16");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
  auto stream = at::cuda::getCurrentCUDAStream();

  int size_m = (int)a.size(0);
  int size_k = (int)a.size(1);
  int size_n = (int)b_q_weight.size(1);
  int groups = (int)b_qzeros.size(0);
  const int zero_offset = use_v2_format ? 0 : 1;

  auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(a.device());
  at::Tensor dump = torch::zeros({32, 8}, opts);

#if defined(USE_ROCM)
  vllm::gptq_rdna3_wmma::gemm_q4_wmma_dump_kernel<<<1, 32, 0, stream>>>(
      (const half*)a.data_ptr(),
      (const uint32_t*)b_q_weight.data_ptr(),
      (const uint32_t*)b_qzeros.data_ptr(),
      (const half*)b_scales.data_ptr(),
      (float*)dump.data_ptr(),
      size_m, size_n, size_k, groups, zero_offset);
#else
  TORCH_CHECK(false, "gptq_gemm_rdna3_wmma_dump is ROCm-only");
#endif

  return dump;
}

// Diagnostic: runs the dequant + LDS-write path, then dumps the entire
// 16x16 b_lds tile to a fp16 [16, 16] tensor. Lets us verify dequant + LDS
// produce correct B values before the WMMA reads them.
torch::Tensor gptq_gemm_rdna3_wmma_lds_check(torch::Tensor b_q_weight,
                                             torch::Tensor b_qzeros,
                                             torch::Tensor b_scales,
                                             bool use_v2_format) {
  TORCH_CHECK(b_q_weight.is_cuda() && b_qzeros.is_cuda() && b_scales.is_cuda(),
              "all tensors must be CUDA");
  TORCH_CHECK(b_scales.scalar_type() == torch::kHalf,
              "b_scales must be fp16");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(b_q_weight));
  auto stream = at::cuda::getCurrentCUDAStream();

  int size_k = (int)b_q_weight.size(0) * 8;
  int size_n = (int)b_q_weight.size(1);
  int groups = (int)b_qzeros.size(0);
  const int zero_offset = use_v2_format ? 0 : 1;

  auto opts = torch::TensorOptions().dtype(torch::kHalf).device(b_q_weight.device());
  at::Tensor dump = torch::zeros({16, 16}, opts);

#if defined(USE_ROCM)
  vllm::gptq_rdna3_wmma::wmma_lds_check_kernel<<<1, 32, 0, stream>>>(
      (const uint32_t*)b_q_weight.data_ptr(),
      (const uint32_t*)b_qzeros.data_ptr(),
      (const half*)b_scales.data_ptr(),
      (half*)dump.data_ptr(),
      size_n, size_k, groups, zero_offset);
#else
  TORCH_CHECK(false, "gptq_gemm_rdna3_wmma_lds_check is ROCm-only");
#endif

  return dump;
}

torch::Tensor gptq_gemm_rdna3_wmma_probe(torch::Tensor a, torch::Tensor b,
                                         int64_t mode) {
  TORCH_CHECK(a.is_cuda() && b.is_cuda(), "a/b must be CUDA tensors");
  TORCH_CHECK(a.scalar_type() == torch::kHalf, "a must be fp16");
  TORCH_CHECK(b.scalar_type() == torch::kHalf, "b must be fp16");
  TORCH_CHECK(a.dim() == 2 && a.size(0) == 16 && a.size(1) == 16,
              "a must be 16x16");
  TORCH_CHECK(b.dim() == 2 && b.size(0) == 16 && b.size(1) == 16,
              "b must be 16x16");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
  auto stream = at::cuda::getCurrentCUDAStream();

  auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(a.device());
  at::Tensor dump = torch::zeros({32, 8}, opts);

#if defined(USE_ROCM)
  vllm::gptq_rdna3_wmma::wmma_layout_probe_kernel<<<1, 32, 0, stream>>>(
      (const half*)a.data_ptr(), (const half*)b.data_ptr(),
      (float*)dump.data_ptr(), (int)mode);
#else
  TORCH_CHECK(false, "gptq_gemm_rdna3_wmma_probe is ROCm-only");
#endif

  return dump;
}
