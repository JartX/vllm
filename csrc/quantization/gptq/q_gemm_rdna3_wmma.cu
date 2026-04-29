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

// Pull dequant primitives from the sibling namespace.
using vllm::gptq_rdna3::bf16_t;
using vllm::gptq_rdna3::bf162_t;
using vllm::gptq_rdna3::dequant_4bit_8_bf16;
using vllm::gptq_rdna3::dequant_4bit_8_fp16;
using vllm::gptq_rdna3::prep_zero_scale_bf16;
using vllm::gptq_rdna3::prep_zero_scale_fp16;

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
// Wave32 fragment layout (RDNA3 ISA + AMD's WMMA samples):
//   * A frag: lane t holds A[t & 15][0..15] — row (t & 15), all 16 cols.
//   * B frag: lane t holds B[t & 15][0..15] — row (t & 15), all 16 cols.
//     IMPORTANT: B is row-major per-lane just like A. The MMA hardware
//     internally does the per-output column gather; the kernel does NOT
//     load "column t of B" into lane t. (An earlier version of this file
//     assumed column-major and produced numerically wrong results.)
//   * C frag: lane t holds row (t & 15), 8 contiguous columns starting at
//     column (lane >> 4) * 8. So lanes 0..15 → columns 0..7;
//     lanes 16..31 → columns 8..15.
//   * Lanes 16..31 duplicate lanes 0..15 for the input fragments
//     (AMD's "doubled" fragment).
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
        half2 z1z16[2], y1y16[2];
        prep_zero_scale_fp16(zero_v, scale_t, z1z16, y1y16);
        half2 dq[4];
        dequant_4bit_8_fp16(qa, dq, z1z16, y1y16);
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
        prep_zero_scale_bf16(zero_v, scale_t, z_b, y_b);
        bf162_t dq[4];
        dequant_4bit_8_bf16(qa, dq, z_b, y_b);
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

    // B fragment: lane t holds row (t & 15) of the B tile, all 16 columns.
    // Same row-major convention as A. The WMMA hardware does the per-output
    // column gather internally; we must NOT pre-transpose to "column of B".
#pragma unroll
    for (int i = 0; i < 16; i++) {
      b_frag[i] = bitcast_elem<T, E>(b_lds[lane_lo][i]);
    }

    c_acc = wmma_mma(a_frag, b_frag, c_acc);

    __syncthreads();  // before next iter overwrites b_lds
  }

  // ---- Store C ----
  // C frag: lane t holds row (t & 15) at columns ((t >> 4) * 8) + 0..7.
  const int out_m = m_tile + lane_lo;
  const int out_n_base = n_tile + lane_hi * 8;

  if (out_m < size_m) {
    T* out = c + out_m * size_n + out_n_base;
#pragma unroll
    for (int i = 0; i < 8; i++) {
      const int out_n = out_n_base + i;
      if (out_n < size_n) {
        if constexpr (std::is_same<T, half>::value) {
          out[i] = __float2half_rn(c_acc[i]);
        } else {
          out[i] = __float2bfloat16(c_acc[i]);
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
  // 1 wave per block (32 lanes), 16x16 C tile per block.
  dim3 block(32);
  dim3 grid((size_n + 15) / 16, (size_m + 15) / 16, 1);
  gemm_q4_wmma_kernel<T><<<grid, block, 0, stream>>>(
      a, b_q_weight, b_qzeros, b_scales, c, size_m, size_n, size_k, groups,
      zero_offset, b_q_perm);
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
