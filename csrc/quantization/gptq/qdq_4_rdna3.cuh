// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// W4A16 dequant primitives for RDNA3 (gfx1100/gfx1101/gfx1102), templated on
// the activation/scale dtype (half or __hip_bfloat16). The fp16 path reuses
// the classic exllamav2 bit-trick:
//
//   (qa & 0x000F000F) | 0x64006400  ->  half2(1024+q_lo, 1024+q_hi)
//   (qa & 0x00F000F0) | 0x64006400  ->  half2(1024+q_lo*16, 1024+q_hi*16)
//
// The "*16 then divide by 16 in the FMA" trick for the upper-nibble pairs
// works in fp16 because the mantissa (10 bits) is wide enough to hold a value
// shifted by 4 bits. In bf16 the mantissa is only 7 bits, so shifting an upper
// nibble into bits [7:4] would spill into the exponent. To avoid that, the
// bf16 path shifts each pair of nibbles down to bits [3:0]/[19:16] with a
// single right-shift before the OR with 0x43004300 (= bf162(128, 128)).

#ifndef _qdq_4_rdna3_cuh
#define _qdq_4_rdna3_cuh

#include <cstdint>

#if defined(USE_ROCM)
  #include <hip/hip_bf16.h>
  #include <hip/hip_fp16.h>
#else
  #include <cuda_bf16.h>
  #include <cuda_fp16.h>
#endif

namespace vllm {
namespace gptq_rdna3 {

#if defined(USE_ROCM)
using bf16_t = __hip_bfloat16;
using bf162_t = __hip_bfloat162;
#else
using bf16_t = __nv_bfloat16;
using bf162_t = __nv_bfloat162;
#endif

// Bit-shuffle for an int32 holding 8 sequential 4-bit weights q[0..7]:
//   in:  q[7] q[6] q[5] q[4] q[3] q[2] q[1] q[0]   (LSB first)
//   out: q[7] q[5] q[3] q[1] q[6] q[4] q[2] q[0]   (even/odd interleaved)
//
// After shuffle, q[2k]   sits at bits [4k   : 4k+3]   (lower 16)
//                q[2k+1] sits at bits [16+4k: 16+4k+3] (upper 16)
// so a single mask 0x000F000F selects the matching even/odd pair, ready to
// bitcast to half2 / bfloat162 after OR-ing with the magic constant.
__forceinline__ __device__ void shuffle_4bit_8(uint32_t* q) {
  uint32_t qa = q[0];
  uint32_t qb = 0;
#pragma unroll
  for (int i = 0; i < 4; i++) {
    uint32_t qa0 = qa & 0x0F;
    uint32_t qa1 = (qa & 0xF0) >> 4;
    qa >>= 8;
    qb |= (qa1 << (i * 4 + 16));
    qb |= (qa0 << (i * 4));
  }
  q[0] = qb;
}

// ---------------------------------------------------------------------------
// fp16 path
// ---------------------------------------------------------------------------

// Precompute scale-baked constants for a single zero/scale pair.
//   z1z16[0] = scale * (-1024 - zero)            (used for "low" pairs)
//   z1z16[1] = scale * (-64   - zero)            (used for "high" pairs)
//   y1y16[0] = scale * 1                          (low pairs are q + 1024)
//   y1y16[1] = scale * (1/16)                     (high pairs are q*16 + 1024)
__forceinline__ __device__ void prep_zero_scale_fp16(uint32_t zero, half scale,
                                                     half2 (&z1z16)[2],
                                                     half2 (&y1y16)[2]) {
  // half(-1024 - zero) via the exllamav2 bit-trick:
  //   half bits 0xE400 == -1024.0 ; ORing the zero into mantissa subtracts it.
  union {
    uint16_t u;
    half h;
  } z1u;
  z1u.u = (uint16_t)(0xE400 | zero);
  half z1 = z1u.h;
  half z16 = __hsub(__int2half_rn(-64), __int2half_rn((int)zero));

  half2 scale2 = __half2half2(scale);
  z1z16[0] = __hmul2(scale2, __half2half2(z1));
  z1z16[1] = __hmul2(scale2, __half2half2(z16));

  half y1 = __float2half_rn(1.0f);
  half y16 = __float2half_rn(1.0f / 16.0f);
  y1y16[0] = __hmul2(scale2, __half2half2(y1));
  y1y16[1] = __hmul2(scale2, __half2half2(y16));
}

// Dequantize one int32 (8 shuffled 4-bit weights) into 4 half2 pairs:
//   dq[0] = (q[0], q[1]) * scale - zero*scale
//   dq[1] = (q[2], q[3]) * scale - zero*scale
//   dq[2] = (q[4], q[5]) * scale - zero*scale
//   dq[3] = (q[6], q[7]) * scale - zero*scale
__forceinline__ __device__ void dequant_4bit_8_fp16(uint32_t qa,
                                                    half2 (&dq)[4],
                                                    half2 (&z1z16)[2],
                                                    half2 (&y1y16)[2]) {
  const uint32_t c0 = 0x64006400;

  union { uint32_t u; half2 h2; } q0, q1, q2, q3;
  q0.u = (qa & 0x000F000F) | c0;        // half2(q[0]+1024, q[1]+1024)
  q1.u = (qa & 0x00F000F0) | c0;        // half2(q[2]*16+1024, q[3]*16+1024)
  uint32_t qa_hi = qa >> 8;
  q2.u = (qa_hi & 0x000F000F) | c0;     // half2(q[4]+1024, q[5]+1024)
  q3.u = (qa_hi & 0x00F000F0) | c0;     // half2(q[6]*16+1024, q[7]*16+1024)

  dq[0] = __hfma2(q0.h2, y1y16[0], z1z16[0]);
  dq[1] = __hfma2(q1.h2, y1y16[1], z1z16[1]);
  dq[2] = __hfma2(q2.h2, y1y16[0], z1z16[0]);
  dq[3] = __hfma2(q3.h2, y1y16[1], z1z16[1]);
}

// ---------------------------------------------------------------------------
// fp16 PRECISE path: subtract-first instead of FMA.
//
// The classic bit-trick above (FMA form: q*scale + (-(1024+zero)*scale)) loses
// up to ~0.025 per cell at scale=0.1 because fp16(scale) ≠ scale and the FMA
// amplifies that by 1024x without cancelling against the precomputed
// (1024+zero)*scale (which is rounded to fp16 BEFORE the FMA).
//
// The fix: subtract (1024+zero) as an integer first — this is EXACT in fp16
// because integers in [1024, 2047] are exactly representable — then multiply
// by scale, incurring at most one half-ULP rounding. This costs an extra
// instruction per dequant pair (sub+mul vs single FMA), so we keep the fast
// FMA path above for the scalar dot-product kernel where small per-cell error
// averages out across thousands of K iterations, and use this precise variant
// only in the WMMA path where K can be small (16) and errors don't average.
__forceinline__ __device__ void prep_zero_scale_fp16_precise(uint32_t zero,
                                                             half scale,
                                                             half2& z_prep,
                                                             half2& y_prep) {
  // z = half(1024 + zero), exact for zero in [0, 15].
  union {
    uint16_t u;
    half h;
  } zu;
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

  // (q + 1024) - (1024 + zero) = q - zero, exact. Then * scale (one rounding).
  dq[0] = __hmul2(__hsub2(q0.h2, z_prep), y_prep);
  dq[1] = __hmul2(__hsub2(q1.h2, z_prep), y_prep);
  dq[2] = __hmul2(__hsub2(q2.h2, z_prep), y_prep);
  dq[3] = __hmul2(__hsub2(q3.h2, z_prep), y_prep);
}

// ---------------------------------------------------------------------------
// bf16 path
// ---------------------------------------------------------------------------

// Bit-trick magic for bf16:
//   bf16(128) == 0x4300 (sign 0, exp 134, mantissa 0).
//   For nibble n in [0..15], bits [3:0] of mantissa hold n exactly because
//   bf16's ULP at 128 is 1 (mantissa step = 2^(7-7) = 1). So
//   ((qa & 0x000F000F) | 0x43004300) bitcasts to bfloat162(128+n_lo, 128+n_hi).
//
// Because bf16's mantissa is only 7 bits, we cannot use the fp16 "upper nibble
// * 16" trick. Instead each pair of nibbles is shifted down to [3:0]/[19:16]
// via a single 4/8/12-bit right-shift before the OR. That costs one extra
// shift per pair vs fp16, but keeps the FMA structure identical.
__forceinline__ __device__ void prep_zero_scale_bf16(uint32_t zero,
                                                     bf16_t scale,
                                                     bf162_t& z_prep,
                                                     bf162_t& y_prep) {
  // z = scale * -(128 + zero); y = scale.
  float scale_f = __bfloat162float(scale);
  float zf = -(128.0f + (float)zero) * scale_f;
  bf16_t zb = __float2bfloat16(zf);
  z_prep = __bfloat162bfloat162(zb);
  y_prep = __bfloat162bfloat162(scale);
}

__forceinline__ __device__ void dequant_4bit_8_bf16(uint32_t qa,
                                                    bf162_t (&dq)[4],
                                                    bf162_t z_prep,
                                                    bf162_t y_prep) {
  const uint32_t c0 = 0x43004300;

  union { uint32_t u; bf162_t b2; } q0, q1, q2, q3;
  q0.u = ((qa >>  0) & 0x000F000F) | c0;  // bf162(128+q[0], 128+q[1])
  q1.u = ((qa >>  4) & 0x000F000F) | c0;  // bf162(128+q[2], 128+q[3])
  q2.u = ((qa >>  8) & 0x000F000F) | c0;  // bf162(128+q[4], 128+q[5])
  q3.u = ((qa >> 12) & 0x000F000F) | c0;  // bf162(128+q[6], 128+q[7])

  // dq = q_b * scale + (-(128+zero)*scale) = (q - zero) * scale
  dq[0] = __hfma2(q0.b2, y_prep, z_prep);
  dq[1] = __hfma2(q1.b2, y_prep, z_prep);
  dq[2] = __hfma2(q2.b2, y_prep, z_prep);
  dq[3] = __hfma2(q3.b2, y_prep, z_prep);
}

// PRECISE bf16 variant. Same motivation as fp16_precise: avoid amplifying the
// fp16(scale)≠scale rounding error by 128x in the FMA. Used by the WMMA
// kernel; the scalar kernel keeps the FMA form for speed.
__forceinline__ __device__ void prep_zero_scale_bf16_precise(uint32_t zero,
                                                             bf16_t scale,
                                                             bf162_t& z_prep,
                                                             bf162_t& y_prep) {
  // z = bf16(128 + zero), exact for zero in [0, 15] (bf16 ULP=1 at 128).
  union {
    uint16_t u;
    bf16_t h;
  } zu;
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

  // (128 + q) - (128 + zero) = q - zero, exact. Then * scale.
  dq[0] = __hmul2(__hsub2(q0.b2, z_prep), y_prep);
  dq[1] = __hmul2(__hsub2(q1.b2, z_prep), y_prep);
  dq[2] = __hmul2(__hsub2(q2.b2, z_prep), y_prep);
  dq[3] = __hmul2(__hsub2(q3.b2, z_prep), y_prep);
}

}  // namespace gptq_rdna3
}  // namespace vllm

#endif  // _qdq_4_rdna3_cuh
