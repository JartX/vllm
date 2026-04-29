# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Triton-based W4A16 GEMM kernel for ROCm MI300.

Implements fused int4-weight dequantization + fp16 GEMM in a single kernel,
using GPTQ sequential packing (8 int4 values per int32, shifts [0,4,...,28]).
Plugs into the MPLinearKernel selection system and is preferred over
MarlinLinearKernel/ExllamaLinearKernel on ROCm.

Weight layout expected by this kernel (post-process_weights_after_loading):
  qweight: [K, N//8]  int32  — rows=K (input), cols=N//8 (N is packed)
  scales:  [K//G, N]  fp16/bf16
  qzeros:  [K//G, N//8]  int32  (optional; None for symmetric uint4b8)

Checkpoint layout from compressed_tensors_wNa16 create_weights:
  weight_packed:     [N, K//8]  int32  (output_dim=0, input_dim=1, packed_dim=1)
  weight_scale:      [N, K//G]  fp16   (output_dim=0, input_dim=1)
  weight_zero_point: [N//8, K//G]  int32 (output_dim=0, packed_dim=0)
"""

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.model_executor.parameter import BasevLLMParameter, permute_param_layout_
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types
from vllm.triton_utils import tl, triton

from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig

logger = init_logger(__name__)

# One-shot diagnostic so we can verify the gfx11 KN GEMV path is being
# entered. Logged on first call from any layer; cleared after.
_GFX11_KN_FIRST_CALL_LOGGED = {"done": False}

TRITON_W4A16_SUPPORTED_GROUP_SIZES = [-1, 32, 64, 128, 256]
TRITON_W4A16_SUPPORTED_QUANT_TYPES = [
    scalar_types.uint4b8,  # symmetric GPTQ (bias=8)
    scalar_types.uint4,  # asymmetric with explicit zeros
]


@triton.jit
def triton_w4a16_gemm_kernel(
    # Pointers
    a_ptr,  # [M, K]  fp16/bf16 activations
    b_ptr,  # [K, N//8]  int32 packed 4-bit weights (N is the packed dim)
    scales_ptr,  # [K//G, N]  fp16/bf16 scales
    zeros_ptr,  # [K//G, N//8]  int32 packed zeros (unused when HAS_ZP=False)
    c_ptr,  # [M, N]  fp16/bf16 output
    # Dimensions
    M,
    N,
    K,
    # Strides
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,  # stride in b along the packed N//8 dim
    stride_cm,
    stride_cn,
    # Quantization parameters
    group_size,
    # Whether explicit zero points are provided
    HAS_ZP: tl.constexpr,
    # Zero bias used when HAS_ZP is False (e.g. 8 for uint4b8)
    ZP_BIAS: tl.constexpr,
    # Block sizes (tuned for MI300 wavefront=64)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused W4A16 GEMM: C[M,N] = A[M,K] @ dequant(B)[K,N]

    B is stored as [K, N//8] int32 using GPTQ sequential packing:
      each int32 packs 8 consecutive N-values at bit offsets [0,4,8,12,16,20,24,28].

    Dequant: w_fp = (w_int4 - zero) * scale
      HAS_ZP=True:  zero is loaded from zeros_ptr and unpacked
      HAS_ZP=False: zero = ZP_BIAS constant (e.g. 8 for uint4b8 symmetric)
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Row/col offsets for this tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # b/zeros are stored with N packed: N//8 int32 columns per K row
    offs_bn = pid_n * (BLOCK_N // 8) + tl.arange(0, BLOCK_N // 8)

    # GPTQ sequential shifts: each int32 packs 8 nibbles at offsets
    # [0, 4, 8, ..., 28]. The unpack uses a 3D broadcast against this.
    shifts_row = tl.arange(0, 8) * 4  # [8]

    # Scales column offsets: full N-width (one scale per output neuron)
    offs_sn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k_start * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # ---- Load activations A: [BLOCK_M, BLOCK_K] ----
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        mask_a = (offs_m[:, None] < M) & mask_k[None, :]
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)

        # ---- Load packed weights B: [BLOCK_K, BLOCK_N//8] int32 ----
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
        mask_b = mask_k[:, None] & (offs_bn[None, :] < N // 8)
        b_packed = tl.load(b_ptrs, mask=mask_b, other=0)

        # ---- Unpack int4 weights → [BLOCK_K, BLOCK_N] ----
        # Broadcast variant: shift each int32 by [0,4,...,28] in a 3D
        # op then reshape. Same data layout as the interleave path
        # (column j -> nibble (j%8)*4 of int32 j//8). Fewer register
        # shuffles than the 3x tl.interleave chain on RDNA3.
        b = (b_packed[:, :, None] >> shifts_row[None, None, :]) & 0xF
        b = tl.reshape(b, (BLOCK_K, BLOCK_N))

        # ---- Compute scale/zero group row index ----
        g_idx = (k_start * BLOCK_K) // group_size

        # ---- Load scales: [BLOCK_N] → broadcast to [BLOCK_K, BLOCK_N] ----
        scale_offset = g_idx * N + offs_sn
        scale_mask = offs_sn < N
        scales = tl.load(scales_ptr + scale_offset, mask=scale_mask, other=1.0)
        scales = tl.broadcast_to(scales[None, :], (BLOCK_K, BLOCK_N))

        # ---- Load / compute zeros ----
        if HAS_ZP:
            # Load packed zeros row: [BLOCK_N//8] int32
            zero_offset = g_idx * (N // 8) + offs_bn
            zero_mask = offs_bn < N // 8
            z_packed = tl.load(zeros_ptr + zero_offset, mask=zero_mask, other=0)
            # Unpack to [BLOCK_N] via broadcast (same trick as B)
            z = (z_packed[:, None] >> shifts_row[None, :]) & 0xF
            z = tl.reshape(z, (BLOCK_N,))
            z = tl.broadcast_to(z[None, :], (BLOCK_K, BLOCK_N))
        else:
            z = tl.full((BLOCK_K, BLOCK_N), ZP_BIAS, dtype=tl.int32)

        # ---- Dequantize via bit-trick (avoids slow int32→fp16 cast) ----
        # The int4 nibble n (0..15) is encoded as float(C + n) by ORing
        # the bit pattern of float(C) with the nibble in the low 4 bits,
        # then bitcasting via int16. This skips the explicit int→float
        # conversion that takes multiple instructions on RDNA3. The C
        # offset cancels in the (b - z) subtraction.
        #
        # For fp16 (10-bit mantissa), need exp such that LSB of mantissa
        # is worth 1: exp = 25 (=15+10) → C = 1024, magic = 0x6400.
        # For bf16 (7-bit mantissa), need exp such that LSB worth 1:
        # exp = 134 (=127+7) → C = 128, magic = 0x4300.
        if a.dtype == tl.float16:
            b_fp16 = ((b | 0x6400).to(tl.int16)).to(tl.float16, bitcast=True)
            z_fp16 = ((z | 0x6400).to(tl.int16)).to(tl.float16, bitcast=True)
            b_fp = (b_fp16 - z_fp16) * scales
        elif a.dtype == tl.bfloat16:
            b_bf16 = ((b | 0x4300).to(tl.int16)).to(tl.bfloat16, bitcast=True)
            z_bf16 = ((z | 0x4300).to(tl.int16)).to(tl.bfloat16, bitcast=True)
            b_fp = (b_bf16 - z_bf16) * scales
        else:
            # Fallback: explicit cast.
            b_fp = (b - z).to(a.dtype) * scales

        # ---- Accumulate ----
        accumulator += tl.dot(a, b_fp, out_dtype=tl.float32)

    # ---- Store output C: [BLOCK_M, BLOCK_N] ----
    c = accumulator.to(c_ptr.type.element_ty)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask_c)


@triton.jit
def triton_w4a16_gemv_kernel(
    # Pointers
    a_ptr,  # [M, K]  fp16/bf16 activations (M small, typically 1..8)
    b_ptr,  # [K, N//8]  int32 packed 4-bit weights
    scales_ptr,  # [K//G, N]  fp16/bf16 scales
    zeros_ptr,  # [K//G, N//8]  int32 packed zeros (unused if HAS_ZP=False)
    c_ptr,  # [M, N]  fp16/bf16 output
    # Dimensions
    M,
    N,
    K,
    # Strides
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    # Quantization parameters
    group_size,
    HAS_ZP: tl.constexpr,
    ZP_BIAS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    GEMV W4A16 kernel for small M (decode path).

    The GEMM kernel uses `tl.dot`, which on RDNA3 lowers to WMMA with a
    minimum M tile of 16. At M=1..8 most of those rows are masked-zero
    padding, so 8/16..15/16 of the WMMA cycles compute discarded zeros.
    This kernel uses an explicit elementwise multiply + `tl.sum`
    K-reduction — Triton lowers it to scalar FMA with no WMMA, the
    same shape as exllama's HIP kernel for batch=1.

    Grid: (M, cdiv(N, BLOCK_N)) — one program per (row, N-tile).
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_bn = pid_n * (BLOCK_N // 8) + tl.arange(0, BLOCK_N // 8)
    shifts_row = tl.arange(0, 8) * 4  # [8]

    accumulator = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for k_start in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k_start * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # ---- Load activations: [BLOCK_K] (single M row per program) ----
        a = tl.load(
            a_ptr + pid_m * stride_am + offs_k * stride_ak,
            mask=mask_k,
            other=0.0,
        )

        # ---- Load packed weights: [BLOCK_K, BLOCK_N//8] int32 ----
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
        mask_b = mask_k[:, None] & (offs_bn[None, :] < N // 8)
        b_packed = tl.load(b_ptrs, mask=mask_b, other=0)

        # ---- Unpack int4 weights → [BLOCK_K, BLOCK_N] ----
        b = (b_packed[:, :, None] >> shifts_row[None, None, :]) & 0xF
        b = tl.reshape(b, (BLOCK_K, BLOCK_N))

        # ---- Group index for scales/zeros ----
        g_idx = (k_start * BLOCK_K) // group_size

        # ---- Scales: [BLOCK_N] ----
        scales = tl.load(
            scales_ptr + g_idx * N + offs_n,
            mask=offs_n < N,
            other=1.0,
        )

        # ---- Zeros: [BLOCK_N] ----
        if HAS_ZP:
            z_packed = tl.load(
                zeros_ptr + g_idx * (N // 8) + offs_bn,
                mask=offs_bn < N // 8,
                other=0,
            )
            z = (z_packed[:, None] >> shifts_row[None, :]) & 0xF
            z = tl.reshape(z, (BLOCK_N,))
        else:
            z = tl.full((BLOCK_N,), ZP_BIAS, dtype=tl.int32)

        # ---- Bit-trick dequant (see GEMM kernel for derivation) ----
        # Defer the scale multiply: scales are constant across the
        # entire K tile (one group), so applying them per-K-position
        # wastes BLOCK_K-1 multiplies per output. Reduce the unscaled
        # product first, scale once per K tile.
        if a.dtype == tl.float16:
            b_fp = ((b | 0x6400).to(tl.int16)).to(tl.float16, bitcast=True)
            z_fp = ((z | 0x6400).to(tl.int16)).to(tl.float16, bitcast=True)
            b_minus_z = b_fp - z_fp[None, :]
        elif a.dtype == tl.bfloat16:
            b_fp = ((b | 0x4300).to(tl.int16)).to(tl.bfloat16, bitcast=True)
            z_fp = ((z | 0x4300).to(tl.int16)).to(tl.bfloat16, bitcast=True)
            b_minus_z = b_fp - z_fp[None, :]
        else:
            b_minus_z = (b - z[None, :]).to(a.dtype)

        # ---- GEMV reduction along K — no `tl.dot`, no WMMA ----
        # Inner mul stays in act dtype so AMD can fuse with v_pk_fma_f16
        # (half2). Apply scales once per K tile, then promote to fp32
        # for the cross-tile accumulator.
        prod = a[:, None] * b_minus_z
        partial = tl.sum(prod, axis=0) * scales
        accumulator += partial.to(tl.float32)

    # ---- Store output [BLOCK_N] ----
    c_ptrs = c_ptr + pid_m * stride_cm + offs_n * stride_cn
    tl.store(c_ptrs, accumulator.to(c_ptr.type.element_ty), mask=offs_n < N)


@triton.jit
def triton_w4a16_gemv_kn_kernel(
    # Pointers
    a_ptr,  # [M, K] fp16/bf16 activations (M small, decode path)
    b_ptr,  # [K//8, N] int32 -- K packed at dim 0, N row-major at dim 1
    scales_ptr,  # [K//G, N] fp16/bf16
    zeros_ptr,  # [K//G, N//8] int32 (unused if HAS_ZP=False)
    c_ptr,  # [K_SPLITS, M, N] fp32 -- partial sums, one slice per K-split
    # Dimensions
    M,
    N,
    K,
    # Strides
    stride_am,
    stride_ak,
    stride_bk,  # row stride of B (= N for contiguous [K//8, N])
    stride_bn,  # col stride of B (= 1)
    stride_csplit,  # = M * N (one full output plane per K-split)
    stride_cm,
    stride_cn,
    # Quantization (GROUP_SIZE constexpr → enables compile-time
    # GROUPS_PER_TILE; one Triton compile per (group_size, BLOCK_K) pair)
    GROUP_SIZE: tl.constexpr,
    HAS_ZP: tl.constexpr,
    ZP_BIAS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Decode GEMV for [K//8, N] layout with split-K + multi-group inner.

    BLOCK_K can span MULTIPLE quant groups (BLOCK_K % GROUP_SIZE == 0).
    The inner static loop processes one group per iteration so scales/
    zeros are correct, but the OUTER per-program work amortizes setup
    over GROUPS_PER_TILE × BLOCK_N × GROUP_SIZE FMAs — typically 8x
    more than the prior single-group kernel and on par with exllama's
    hand-tuned 128×128 tile.

    Each K-split writes a unique slice c_ptr[pid_k, m, n_tile]; the
    caller reduces over the K-split dim with torch.sum (no atomics).
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    GROUPS_PER_TILE: tl.constexpr = BLOCK_K // GROUP_SIZE

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_zn = pid_n * (BLOCK_N // 8) + tl.arange(0, BLOCK_N // 8)
    shifts = tl.arange(0, 8) * 4  # [8]

    accumulator = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Inner loop over groups within this K-tile (unrolled at compile time)
    for g_local in tl.static_range(GROUPS_PER_TILE):
        k_g_start = pid_k * BLOCK_K + g_local * GROUP_SIZE
        offs_k = k_g_start + tl.arange(0, GROUP_SIZE)
        offs_bk = (k_g_start // 8) + tl.arange(0, GROUP_SIZE // 8)
        mask_k = offs_k < K

        # A: [GROUP_SIZE]
        # Force fp16 internal compute regardless of model dtype. Triton's
        # AMD backend has much better fp16 codegen than bf16 (more mature
        # v_pk_fma_f16 / v_pk_mul_f16 lowering). bf16 inputs are downcast
        # for the duration of the matmul; the fp32 accumulator preserves
        # accuracy across groups, and the caller casts back to bf16. This
        # is the same approach that lets exllama (fp16-only) saturate
        # RDNA3's W4A16 BW.
        a_raw = tl.load(
            a_ptr + pid_m * stride_am + offs_k * stride_ak,
            mask=mask_k,
            other=0.0,
        )
        a = a_raw.to(tl.float16)

        # B: [GROUP_SIZE//8, BLOCK_N] int32 -- coalesced N-row reads
        b_ptrs = b_ptr + offs_bk[:, None] * stride_bk + offs_n[None, :] * stride_bn
        mask_b = (offs_bk[:, None] < K // 8) & (offs_n[None, :] < N)
        b_packed = tl.load(b_ptrs, mask=mask_b, other=0)

        # Unpack to [GROUP_SIZE, BLOCK_N] without permute (8-shift dim
        # placed BETWEEN k_row and n_col so reshape flattens correctly).
        b3d = (b_packed[:, None, :] >> shifts[None, :, None]) & 0xF
        b = tl.reshape(b3d, (GROUP_SIZE, BLOCK_N))

        # Scales/zeros for this group (cast scales to fp16 to match)
        g_idx = k_g_start // GROUP_SIZE
        scales_raw = tl.load(
            scales_ptr + g_idx * N + offs_n, mask=offs_n < N, other=1.0
        )
        scales = scales_raw.to(tl.float16)

        if HAS_ZP:
            z_packed = tl.load(
                zeros_ptr + g_idx * (N // 8) + offs_zn,
                mask=offs_zn < N // 8,
                other=0,
            )
            z = (z_packed[:, None] >> shifts[None, :]) & 0xF
            z = tl.reshape(z, (BLOCK_N,))
        else:
            z = tl.full((BLOCK_N,), ZP_BIAS, dtype=tl.int32)

        # Bit-trick dequant — fp16 only (0x6400 = half(1024.0))
        b_fp = ((b | 0x6400).to(tl.int16)).to(tl.float16, bitcast=True)
        z_fp = ((z | 0x6400).to(tl.int16)).to(tl.float16, bitcast=True)
        b_minus_z = b_fp - z_fp[None, :]

        # GEMV reduction within this group, all fp16 → fp32 accumulator
        prod = a[:, None] * b_minus_z  # [GROUP_SIZE, BLOCK_N] fp16
        partial_g = tl.sum(prod, axis=0) * scales  # [BLOCK_N] fp16
        accumulator += partial_g.to(tl.float32)

    # Plain store to a unique slice [pid_k, m, n_tile] — no atomic
    c_ptrs = (
        c_ptr
        + pid_k * stride_csplit
        + pid_m * stride_cm
        + offs_n * stride_cn
    )
    tl.store(c_ptrs, accumulator, mask=offs_n < N)


@triton.jit
def triton_w4a16_gemm_kn_kernel(
    # Pointers
    a_ptr,  # [M, K] fp16/bf16
    b_ptr,  # [K//8, N] int32
    scales_ptr,  # [K//G, N]
    zeros_ptr,  # [K//G, N//8]
    c_ptr,  # [M, N]
    # Dimensions
    M,
    N,
    K,
    # Strides
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    # Quantization
    group_size,
    HAS_ZP: tl.constexpr,
    ZP_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Prefill GEMM for [K//8, N] layout. Uses tl.dot (WMMA on RDNA3).

    No split-K: at M>=16 the M*N grid already saturates the GPU,
    and atomic_add cost outweighs the parallelism gain.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_zn = pid_n * (BLOCK_N // 8) + tl.arange(0, BLOCK_N // 8)

    shifts = tl.arange(0, 8) * 4
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k_start * BLOCK_K + tl.arange(0, BLOCK_K)
        offs_bk = k_start * (BLOCK_K // 8) + tl.arange(0, BLOCK_K // 8)
        mask_k = offs_k < K

        # A: [BLOCK_M, BLOCK_K]
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        a = tl.load(
            a_ptrs, mask=(offs_m[:, None] < M) & mask_k[None, :], other=0.0
        )

        # B: [BLOCK_K//8, BLOCK_N] int32  -- coalesced row-major load
        b_ptrs = b_ptr + offs_bk[:, None] * stride_bk + offs_n[None, :] * stride_bn
        mask_b = (offs_bk[:, None] < K // 8) & (offs_n[None, :] < N)
        b_packed = tl.load(b_ptrs, mask=mask_b, other=0)

        # Unpack to [BLOCK_K, BLOCK_N] (K dim contiguous after reshape)
        b3d = (b_packed[:, None, :] >> shifts[None, :, None]) & 0xF
        b = tl.reshape(b3d, (BLOCK_K, BLOCK_N))

        g_idx = (k_start * BLOCK_K) // group_size
        scales = tl.load(
            scales_ptr + g_idx * N + offs_n, mask=offs_n < N, other=1.0
        )

        if HAS_ZP:
            z_packed = tl.load(
                zeros_ptr + g_idx * (N // 8) + offs_zn,
                mask=offs_zn < N // 8,
                other=0,
            )
            z = (z_packed[:, None] >> shifts[None, :]) & 0xF
            z = tl.reshape(z, (BLOCK_N,))
        else:
            z = tl.full((BLOCK_N,), ZP_BIAS, dtype=tl.int32)

        # Bit-trick dequant
        if a.dtype == tl.float16:
            b_fp = ((b | 0x6400).to(tl.int16)).to(tl.float16, bitcast=True)
            z_fp = ((z | 0x6400).to(tl.int16)).to(tl.float16, bitcast=True)
            b_minus_z = b_fp - z_fp[None, :]
        elif a.dtype == tl.bfloat16:
            b_fp = ((b | 0x4300).to(tl.int16)).to(tl.bfloat16, bitcast=True)
            z_fp = ((z | 0x4300).to(tl.int16)).to(tl.bfloat16, bitcast=True)
            b_minus_z = b_fp - z_fp[None, :]
        else:
            b_minus_z = (b - z[None, :]).to(a.dtype)

        b_dequant = b_minus_z * scales[None, :]  # [BLOCK_K, BLOCK_N]
        accumulator += tl.dot(a, b_dequant, out_dtype=tl.float32)

    c = accumulator.to(c_ptr.type.element_ty)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask_c)


def triton_w4a16_gemm_kn(
    a: torch.Tensor,  # [M, K] fp16/bf16
    b_q: torch.Tensor,  # [K//8, N] int32 -- exllama-style layout
    scales: torch.Tensor,  # [K//G, N]
    qzeros: torch.Tensor | None,  # [K//G, N//8] or None
    group_size: int,
    zp_bias: int = 8,
) -> torch.Tensor:
    """
    W4A16 GEMM with [K//8, N] weight layout (the layout exllama uses).

    Routes to a split-K GEMV kernel for M<=16 (decode) and a regular
    GEMM kernel for M>16 (prefill). Used on gfx11 (RDNA3/3.5) where
    coalesced N-row reads close the gap with exllama.
    """
    assert a.is_contiguous(), "Activation matrix must be contiguous"
    assert b_q.is_contiguous(), "Weight matrix must be contiguous"
    assert scales.is_contiguous(), "Scales must be contiguous"

    M, K = a.shape
    N = b_q.shape[1]

    assert b_q.shape == (K // 8, N), (
        f"b_q shape mismatch: {b_q.shape} vs ({K // 8}, {N})"
    )
    assert scales.shape == (K // group_size, N), (
        f"scales shape mismatch: {scales.shape} vs ({K // group_size}, {N})"
    )
    if qzeros is not None:
        assert qzeros.shape == (K // group_size, N // 8), (
            f"qzeros shape mismatch: {qzeros.shape}"
        )

    has_zp = qzeros is not None
    zeros_ptr = qzeros if has_zp else b_q

    # ---- Decode path: split-K multi-group GEMV ----
    if M <= 16:
        # BLOCK_K spans multiple groups for amortization. For G=32 and
        # BLOCK_K=128 each program does 4 inner-group iterations, lifting
        # per-program FMAs from ~1k to ~8k — closer to exllama's 16k tile.
        BLOCK_N = 64
        BLOCK_K = max(128, group_size)
        # Round BLOCK_K to a multiple of group_size and clamp to K
        BLOCK_K = (BLOCK_K // group_size) * group_size
        BLOCK_K = min(BLOCK_K, K)
        # Avoid degenerate single-group tile if K < BLOCK_K
        if BLOCK_K == 0:
            BLOCK_K = group_size

        K_SPLITS = triton.cdiv(K, BLOCK_K)

        # Per-K-split partial sums in fp32. Plain stores to unique slots,
        # no atomics. torch.sum reduces along the K-split dim.
        c_temp = torch.empty(
            (K_SPLITS, M, N), dtype=torch.float32, device=a.device
        )

        grid = (M, triton.cdiv(N, BLOCK_N), K_SPLITS)
        triton_w4a16_gemv_kn_kernel[grid](
            a,
            b_q,
            scales,
            zeros_ptr,
            c_temp,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            b_q.stride(0),
            b_q.stride(1),
            c_temp.stride(0),
            c_temp.stride(1),
            c_temp.stride(2),
            GROUP_SIZE=group_size,
            HAS_ZP=has_zp,
            ZP_BIAS=zp_bias,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            num_warps=2,
            num_stages=2,
        )
        # Sum across K-splits → [M, N] fp32 → cast to activation dtype
        return c_temp.sum(dim=0).to(a.dtype)

    # ---- Prefill path: regular GEMM, no split-K ----
    c = torch.empty((M, N), dtype=a.dtype, device=a.device)

    if M <= 32:
        BLOCK_M, BLOCK_N, BLOCK_K = 32, 64, 64
    elif M <= 64:
        BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    else:
        BLOCK_M, BLOCK_N, BLOCK_K = 128, 32, 64

    if group_size < BLOCK_K:
        BLOCK_K = group_size

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    triton_w4a16_gemm_kn_kernel[grid](
        a,
        b_q,
        scales,
        zeros_ptr,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b_q.stride(0),
        b_q.stride(1),
        c.stride(0),
        c.stride(1),
        group_size=group_size,
        HAS_ZP=has_zp,
        ZP_BIAS=zp_bias,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=2,
    )
    return c


def triton_w4a16_gemm(
    a: torch.Tensor,  # [M, K] fp16/bf16
    b_q: torch.Tensor,  # [K, N//8] int32
    scales: torch.Tensor,  # [K//G, N] fp16/bf16
    qzeros: torch.Tensor | None,  # [K//G, N//8] int32, or None
    group_size: int,
    zp_bias: int = 8,  # bias for uint4b8 when qzeros is None
) -> torch.Tensor:
    """
    Fused W4A16 GEMM using GPTQ-packed int4 weights.

    Args:
        a:          Activation matrix [M, K], float16 or bfloat16.
        b_q:        Packed weight matrix [K, N//8], int32 (GPTQ sequential).
        scales:     Per-group scales [K//G, N], same dtype as a.
        qzeros:     Per-group packed zero points [K//G, N//8] int32, or None
                    for symmetric quantization (uses zp_bias instead).
        group_size: Quantization group size (resolved from -1 to K by caller).
        zp_bias:    Constant zero used when qzeros is None (default 8 for uint4b8).

    Returns:
        Output matrix [M, N], same dtype as a.
    """
    assert a.is_contiguous(), "Activation matrix must be contiguous"
    assert b_q.is_contiguous(), "Weight matrix must be contiguous"
    assert scales.is_contiguous(), "Scales must be contiguous"

    M, K = a.shape
    N = b_q.shape[1] * 8

    assert b_q.shape == (K, N // 8), (
        f"b_q shape mismatch: {b_q.shape} vs ({K}, {N // 8})"
    )
    assert scales.shape == (K // group_size, N), (
        f"scales shape mismatch: {scales.shape} vs ({K // group_size}, {N})"
    )
    if qzeros is not None:
        assert qzeros.shape == (K // group_size, N // 8), (
            f"qzeros shape mismatch: {qzeros.shape}"
        )

    c = torch.empty((M, N), dtype=a.dtype, device=a.device)

    has_zp = qzeros is not None
    # Provide a dummy pointer when HAS_ZP=False (Triton requires a valid ptr)
    zeros_ptr = qzeros if has_zp else b_q

    num_warps: int | None = None
    num_stages: int | None = None

    is_gfx11 = False
    if current_platform.is_rocm():
        from vllm.platforms.rocm import on_gfx1x, on_gfx11

        is_gfx11 = on_gfx11()

        # ---- gfx11 small-M decode path: bypass WMMA via GEMV kernel ----
        # tl.dot on RDNA3 forces a 16-row WMMA tile. At M<=16 most of
        # those rows are zero-padded, so the WMMA kernel computes
        # 0/16..15/16 discarded work per cycle. The GEMV kernel uses
        # `tl.sum` over an elementwise product instead of `tl.dot`,
        # lowering to scalar FMA — same shape as exllama's HIP kernel
        # at batch=1. Restricted to gfx11 (RDNA3/3.5) where the WMMA
        # tile waste is most pronounced.
        #
        # Tile choice: M=1 decode is occupancy-bound — RDNA3 needs many
        # active warps to hide global-memory latency. Use BLOCK_N=32
        # with num_warps=1 (one wave32 per program, 1 lane per output)
        # so we get ~N/32 programs. For typical Qwen3.5 dims that's
        # 160..1728 warps, vs 64 with BLOCK_N=64 num_warps=2 — well
        # past the ~96-CU saturation point so memory latency hides.
        if is_gfx11 and M <= 16:
            BLOCK_N_GEMV = 32
            BLOCK_K_GEMV = 128
            if group_size < BLOCK_K_GEMV:
                BLOCK_K_GEMV = group_size

            grid_gemv = (M, triton.cdiv(N, BLOCK_N_GEMV))
            triton_w4a16_gemv_kernel[grid_gemv](
                a,
                b_q,
                scales,
                zeros_ptr,
                c,
                M,
                N,
                K,
                a.stride(0),
                a.stride(1),
                b_q.stride(0),
                b_q.stride(1),
                c.stride(0),
                c.stride(1),
                group_size=group_size,
                HAS_ZP=has_zp,
                ZP_BIAS=zp_bias,
                BLOCK_N=BLOCK_N_GEMV,
                BLOCK_K=BLOCK_K_GEMV,
                num_warps=1,
                num_stages=3,
            )
            return c

        if is_gfx11:
            # WMMA forces a 16-row M tile minimum. With BLOCK_M=32 and
            # M=1 we burn 31/32 of the WMMA cycles on masked-zero rows.
            # Drop to BLOCK_M=16 for M<=16 to halve that waste.
            if M <= 16:
                BLOCK_M, BLOCK_N, BLOCK_K = 16, 64, 64
            elif M <= 32:
                BLOCK_M, BLOCK_N, BLOCK_K = 32, 32, 64
            elif M <= 64:
                BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
            else:
                BLOCK_M, BLOCK_N, BLOCK_K = 128, 32, 64
            num_warps = 4
            num_stages = 2
        elif on_gfx1x():
            # gfx12 (RDNA4): keep previous tuning, no explicit warp/stage
            # override yet.
            if M <= 32:
                BLOCK_M, BLOCK_N, BLOCK_K = 32, 32, 64
            elif M <= 64:
                BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
            else:
                BLOCK_M, BLOCK_N, BLOCK_K = 128, 32, 64
        else:
            # Tuned for MI300 (gfx942, 304 CUs, 64-wide wavefronts).
            if M <= 32:
                BLOCK_M, BLOCK_N, BLOCK_K = 32, 64, 32
            elif M <= 64:
                BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
            else:
                BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 32
    else:
        if M <= 32:
            BLOCK_M, BLOCK_N, BLOCK_K = 32, 64, 32
        elif M <= 64:
            BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
        else:
            BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 32

    # The kernel loads scales/zeros for a single group per BLOCK_K tile
    # (one g_idx per iteration). If BLOCK_K > group_size, rows at the tail
    # of the tile dequantize with the wrong group's scales, silently
    # corrupting the output. Clamp BLOCK_K to group_size to keep one
    # scale group per tile.
    if group_size < BLOCK_K:
        BLOCK_K = group_size

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    launch_kwargs: dict = dict(
        group_size=group_size,
        HAS_ZP=has_zp,
        ZP_BIAS=zp_bias,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    if num_warps is not None:
        launch_kwargs["num_warps"] = num_warps
    if num_stages is not None:
        launch_kwargs["num_stages"] = num_stages

    triton_w4a16_gemm_kernel[grid](
        a,
        b_q,
        scales,
        zeros_ptr,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b_q.stride(0),
        b_q.stride(1),
        c.stride(0),
        c.stride(1),
        **launch_kwargs,
    )
    return c


class TritonW4A16LinearKernel(MPLinearKernel):
    """
    Triton-based W4A16 GEMM kernel for ROCm (MI300 and newer).

    Supports GPTQ-format int4 weights (uint4b8 symmetric, uint4 asymmetric)
    with grouped quantization. Weight tensors are transposed from the
    compressed-tensors checkpoint layout to the kernel's [K, N//8] layout.
    """

    SUPPORTED_QUANT_TYPES = TRITON_W4A16_SUPPORTED_QUANT_TYPES

    @classmethod
    def get_min_capability(cls) -> int:
        # Triton handles capability checks itself
        return 0

    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        if not current_platform.is_rocm():
            return False, "TritonW4A16LinearKernel only targets ROCm"

        if c.weight_type not in cls.SUPPORTED_QUANT_TYPES:
            return (
                False,
                f"Quant type {c.weight_type} not supported; "
                f"supported: {cls.SUPPORTED_QUANT_TYPES}",
            )

        if c.act_type not in (torch.float16, torch.bfloat16):
            return False, "Only float16/bfloat16 activations are supported"

        N = c.partition_weight_shape[1]
        if N % 8 != 0:
            return (
                False,
                f"Output features ({N}) must be divisible by 8 "
                "(8 int4 values packed per int32)",
            )

        if c.has_g_idx:
            return (
                False,
                "Activation reordering (g_idx) is not supported by "
                "TritonW4A16LinearKernel",
            )

        gs = c.group_size
        if (
            gs not in TRITON_W4A16_SUPPORTED_GROUP_SIZES
            and gs != c.full_weight_shape[0]
        ):
            return (
                False,
                f"Group size {gs} not supported; "
                f"supported: {TRITON_W4A16_SUPPORTED_GROUP_SIZES} "
                f"or full K ({c.full_weight_shape[0]})",
            )

        K = c.partition_weight_shape[0]
        eff_gs = gs if gs != -1 else K
        if K % eff_gs != 0:
            return (False, f"Input features {K} not divisible by group size {eff_gs}")

        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """
        Convert compressed-tensors checkpoint layout to kernel layout.

        Checkpoint (from compressed_tensors_wNa16.create_weights):
          weight_packed:     [N, K//8]  int32   input_dim=1, output_dim=0, packed_dim=1
          weight_scale:      [N, K//G]  fp16    input_dim=1, output_dim=0
          weight_zero_point: [N//8, K//G] int32  output_dim=0, packed_dim=0

        On gfx11 (RDNA3/3.5) we transpose to [K//8, N] — the same
        layout exllama's HIP kernel uses. With N row-major, a 32-lane
        warp loads one full cache line per cycle (32 * 4 B = 128 B,
        fully coalesced). On other platforms we keep the [K, N//8]
        repack for the original WMMA GEMM kernel.

        Scales / zeros are still transposed to [K//G, N] and
        [K//G, N//8] for both layouts (group-major access).
        """

        import sys

        layout_kn = False
        if current_platform.is_rocm():
            from vllm.platforms.rocm import on_gfx11

            layout_kn = on_gfx11()
        layer._w4a16_layout_kn = layout_kn

        # Loud one-shot diagnostic: this runs at model load (not in any
        # torch.compile / CUDA-graph capture context) so it should always
        # print. Use both logger.warning AND raw stderr to guarantee
        # visibility.
        if not _GFX11_KN_FIRST_CALL_LOGGED["done"]:
            msg = (
                f"[Triton W4A16] process_weights_after_loading: "
                f"is_rocm={current_platform.is_rocm()} "
                f"layout_kn={layout_kn} (gfx11={layout_kn}) "
                f"→ {'NEW [K//8,N] split-K path' if layout_kn else 'OLD [K,N//8] path'}"
            )
            logger.warning(msg)
            print(msg, file=sys.stderr, flush=True)
            _GFX11_KN_FIRST_CALL_LOGGED["done"] = True

        if layout_kn:
            # ---- gfx11: transpose to [K//8, N] (exllama-style) ----
            def repack_w_q(x: BasevLLMParameter) -> BasevLLMParameter:
                # Checkpoint is [N, K//8] with K packed at dim 1.
                # Transpose to [K//8, N] keeps K packed (now at dim 0)
                # and puts N as a contiguous row-major dim. No bit-level
                # repack needed — each int32 still stores 8 K-values.
                permute_param_layout_(x, input_dim=1, output_dim=0, packed_dim=1)
                x.data = x.data.t().contiguous()
                return x
        else:
            # ---- Other platforms: full repack to [K, N//8] ----
            # Original packing: K packed into K//8 (8 K-values per int32)
            # Kernel packing:   N packed into N//8 (8 N-values per int32)
            # Requires a full unpack→transpose→repack (CPU-side, one-time).
            def repack_w_q(x: BasevLLMParameter) -> BasevLLMParameter:
                permute_param_layout_(x, input_dim=1, output_dim=0, packed_dim=1)
                w = x.data  # [N, K//8] int32

                N_dim, K8 = w.shape
                K_dim = K8 * 8
                shifts = torch.arange(8, device=w.device, dtype=torch.int32) * 4
                w_unpacked = ((w.unsqueeze(-1) >> shifts) & 0xF).reshape(N_dim, K_dim)
                w_KN = w_unpacked.t().contiguous()
                N8 = N_dim // 8
                w_repacked = torch.sum(
                    (w_KN.view(K_dim, N8, 8) & 0xF) << shifts,
                    dim=2,
                    dtype=torch.int32,
                )
                x.data = w_repacked.contiguous()
                return x

        def repack_w_s(x: BasevLLMParameter) -> BasevLLMParameter:
            # x.data is [N, K//G] fp16, bring to [K//G, N]
            permute_param_layout_(x, input_dim=1, output_dim=0)
            x.data = x.data.t().contiguous()
            return x

        self._transform_param(layer, self.w_q_name, repack_w_q)
        self._transform_param(layer, self.w_s_name, repack_w_s)

        if self.w_zp_name is not None:
            zp = getattr(layer, self.w_zp_name, None)
            if zp is not None:
                # Checkpoint: [N//8, K//G] int32 (N packed at dim 0, K//G at dim 1)
                # Kernel needs: [K//G, N//8] — just transpose
                replace_parameter(
                    layer,
                    self.w_zp_name,
                    torch.nn.Parameter(zp.data.t().contiguous(), requires_grad=False),
                )

    def apply_weights(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        c = self.config
        w_q, w_s, w_zp, _ = self._get_weight_params(layer)

        x_2d = x.reshape(-1, x.shape[-1]).contiguous()
        out_shape = x.shape[:-1] + (c.partition_weight_shape[1],)

        K = c.partition_weight_shape[0]
        group_size = c.group_size if c.group_size != -1 else K

        # For symmetric types (uint4b8), use the scalar bias; no zeros tensor
        zp_bias = c.weight_type.bias if c.weight_type.has_bias() else 0

        if getattr(layer, "_w4a16_layout_kn", False):
            output = triton_w4a16_gemm_kn(
                a=x_2d,
                b_q=w_q,
                scales=w_s,
                qzeros=w_zp,
                group_size=group_size,
                zp_bias=zp_bias,
            )
        else:
            output = triton_w4a16_gemm(
                a=x_2d,
                b_q=w_q,
                scales=w_s,
                qzeros=w_zp,
                group_size=group_size,
                zp_bias=zp_bias,
            )

        if bias is not None:
            output.add_(bias)

        return output.reshape(out_shape)
