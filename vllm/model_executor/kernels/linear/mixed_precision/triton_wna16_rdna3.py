# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""WNA16 dense linear Triton kernel for AMD RDNA3 (gfx11xx).

The default WNA16 backend on ROCm is :class:`ExllamaLinearKernel`, which
calls into a HIP/CUDA C++ GPTQ GEMM that on RDNA3 falls back to
``hipBLAS`` and never lights up the WMMA pipeline. This file provides a
pure-Triton replacement that compiles to RDNA3 WMMA fragments via
``tl.dot``. It registers itself ahead of Exllama in the ROCm linear
kernel chain when the GPU is gfx11xx, and falls back to Exllama for
shapes/quant-types it cannot handle.

Layout (after :meth:`process_weights_after_loading`):
- ``w_q``: ``[K // 2, N]`` ``uint8``  (int4-packed: low nibble = even K)
          ``[K, N]``      ``uint8``  (int8 stored directly)
- ``w_s``: ``[K // group_size, N]`` activation dtype (fp16/bf16)
- ``w_zp``: ``[K // group_size, N]`` ``uint8`` (only when asymmetric)

Symmetric quantization uses the type bias (8 for ``uint4b8``,
128 for ``uint8b128``) so the kernel only needs a single integer
constant; asymmetric types load a per-group ``zp`` tensor instead.
"""

from typing import Final

import torch

import vllm.envs as envs
from vllm.model_executor.parameter import BasevLLMParameter, permute_param_layout_
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types
from vllm.triton_utils import tl, triton

from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig

_TRITON_RDNA3_SUPPORTED_TYPES: Final = [
    scalar_types.uint4b8,
    scalar_types.uint8b128,
    scalar_types.uint4,
    scalar_types.uint8,
]


@triton.jit
def _wna16_rdna3_gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    s_ptr,
    zp_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_sk,
    stride_sn,
    stride_zk,
    stride_zn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    group_size: tl.constexpr,
    weight_bias: tl.constexpr,
    HAS_ZP: tl.constexpr,
    USE_INT4: tl.constexpr,
    compute_type: tl.constexpr,
):
    """RDNA3-tuned WNA16 dense GEMM (one expert).

    Constraints (asserted at the launch site):
      - ``BLOCK_SIZE_K`` must be a multiple of ``group_size``.
      - ``K`` must be a multiple of ``BLOCK_SIZE_K``.
      - ``N`` may be padded by the launcher when not a multiple of
        ``BLOCK_SIZE_N``; the kernel masks the trailing N tile.
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_mask = offs_am[:, None] < M
    n_mask = offs_bn < N

    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak

    if USE_INT4:
        b_ptrs = (
            b_ptr + (offs_k[:, None] // 2) * stride_bk + offs_bn[None, :] * stride_bn
        )
        b_shifter = (offs_k[:, None] % 2) * 4
    else:
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
        b_shifter = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=tl.int32)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    num_k_iters: tl.constexpr = K // BLOCK_SIZE_K

    for k in range(0, num_k_iters):
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b_packed = tl.load(b_ptrs, mask=n_mask[None, :], other=0)
        # constexpr branch: int4 path produces int32, int8 path keeps uint8;
        # a ternary would force a common dtype that Triton cannot infer.
        if USE_INT4:  # noqa: SIM108
            b_int = (b_packed >> b_shifter) & 0xF
        else:
            b_int = b_packed

        # BLOCK_SIZE_K == group_size: exactly one scale (and zp) per K iter.
        b_scale = tl.load(
            s_ptr + k * stride_sk + offs_bn * stride_sn,
            mask=n_mask,
            other=0.0,
        ).to(tl.float32)

        if HAS_ZP:
            b_zp = tl.load(
                zp_ptr + k * stride_zk + offs_bn * stride_zn,
                mask=n_mask,
                other=0,
            ).to(tl.float32)
            b_f = (b_int.to(tl.float32) - b_zp[None, :]) * b_scale[None, :]
        else:
            b_f = (b_int.to(tl.float32) - weight_bias) * b_scale[None, :]

        accumulator = tl.dot(a, b_f.to(compute_type), acc=accumulator)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        if USE_INT4:
            b_ptrs += (BLOCK_SIZE_K // 2) * stride_bk
        else:
            b_ptrs += BLOCK_SIZE_K * stride_bk

    accumulator = accumulator.to(compute_type)
    c_ptrs = c_ptr + offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn
    c_mask = a_mask & n_mask[None, :]
    tl.store(c_ptrs, accumulator, mask=c_mask)


def _select_block_sizes(
    M: int, N: int, K: int, group_size: int, size_bits: int
) -> dict[str, int]:
    """Tile selector for the RDNA3 WNA16 dense kernel.

    BLOCK_SIZE_K is locked to ``group_size`` so a single scale vector
    feeds the whole K iteration; BLOCK_SIZE_N scales with M.
    """
    block_k = group_size

    if M <= 16:
        block_m = 16
        block_n = 64
        group_m = 1
    elif M <= 64:
        block_m = 32
        block_n = 128
        group_m = 4
    else:
        block_m = 64
        block_n = 128
        group_m = 8

    return {
        "BLOCK_SIZE_M": block_m,
        "BLOCK_SIZE_N": block_n,
        "BLOCK_SIZE_K": block_k,
        "GROUP_SIZE_M": group_m,
        "num_warps": 4,
        "num_stages": 2,
        "waves_per_eu": 2 if M <= 64 else 1,
    }


def _wna16_rdna3_gemm(
    x: torch.Tensor,
    w_q: torch.Tensor,
    w_s: torch.Tensor,
    w_zp: torch.Tensor | None,
    weight_bias: int,
    size_bits: int,
    group_size: int,
) -> torch.Tensor:
    """Eager-mode launcher for the WNA16 RDNA3 GEMM Triton kernel."""
    assert x.is_contiguous(), "Input must be contiguous"
    assert w_q.is_contiguous() and w_s.is_contiguous()
    assert size_bits in (4, 8)

    M, K = x.shape
    if size_bits == 4:
        K_packed, N = w_q.shape
        assert K_packed * 2 == K, f"int4 weight K mismatch: w_q[{K_packed}] vs x[{K}]"
    else:
        K_w, N = w_q.shape
        assert K_w == K, f"int8 weight K mismatch: w_q[{K_w}] vs x[{K}]"

    out = torch.empty((M, N), dtype=x.dtype, device=x.device)

    block = _select_block_sizes(M, N, K, group_size, size_bits)
    assert K % block["BLOCK_SIZE_K"] == 0, (
        "RDNA3 wna16 dense kernel requires K to be divisible by BLOCK_SIZE_K"
    )

    if x.dtype == torch.bfloat16:
        compute_type = tl.bfloat16
    elif x.dtype == torch.float16:
        compute_type = tl.float16
    elif x.dtype == torch.float32:
        compute_type = tl.float32
    else:
        raise ValueError(f"Unsupported activation dtype: {x.dtype}")

    use_int4 = size_bits == 4
    has_zp = w_zp is not None

    grid = lambda META: (  # noqa: E731
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    if has_zp:
        zp_stride_k = w_zp.stride(0)
        zp_stride_n = w_zp.stride(1)
        zp_ptr = w_zp
    else:
        zp_stride_k = 0
        zp_stride_n = 0
        zp_ptr = w_q  # unused; placeholder pointer

    _wna16_rdna3_gemm_kernel[grid](
        x,
        w_q,
        out,
        w_s,
        zp_ptr,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        w_q.stride(0),
        w_q.stride(1),
        out.stride(0),
        out.stride(1),
        w_s.stride(0),
        w_s.stride(1),
        zp_stride_k,
        zp_stride_n,
        group_size=group_size,
        weight_bias=weight_bias,
        HAS_ZP=has_zp,
        USE_INT4=use_int4,
        compute_type=compute_type,
        **block,
    )
    return out


class TritonWNA16RDNA3LinearKernel(MPLinearKernel):
    """RDNA3-only Triton WNA16 linear kernel.

    Selected ahead of :class:`ExllamaLinearKernel` on gfx11xx so we hit
    WMMA via Triton instead of falling back to ``hipBLAS``. The kernel
    is opt-out via ``VLLM_DISABLED_KERNELS=TritonWNA16RDNA3LinearKernel``
    if it ever causes issues — Exllama then takes over as before.
    """

    @classmethod
    def get_min_capability(cls) -> int:
        # gfx1100 reports compute capability (11, 0) -> 110.
        return 110

    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        if not current_platform.is_rocm():
            return False, "RDNA3 Triton wna16 kernel only runs on ROCm"

        try:
            from vllm.platforms.rocm import on_gfx1x
        except ImportError:
            return False, "ROCm platform helpers unavailable"

        if not on_gfx1x():
            return False, "RDNA3 Triton wna16 kernel requires gfx11xx"

        if c.weight_type not in _TRITON_RDNA3_SUPPORTED_TYPES:
            return False, (
                f"Weight type ({c.weight_type}) not supported, "
                f"supported: {_TRITON_RDNA3_SUPPORTED_TYPES}"
            )

        if c.act_type not in (torch.float16, torch.bfloat16):
            return False, "Activation dtype must be fp16 or bf16"

        if c.has_g_idx:
            return False, "Activation reordering (g_idx) not supported"

        if c.group_size <= 0:
            return False, "Channelwise quantization not supported"

        if c.partition_weight_shape[0] % c.group_size != 0:
            return False, (
                f"Group size ({c.group_size}) does not evenly divide "
                f"input features ({c.partition_weight_shape[0]})"
            )

        # BLOCK_SIZE_K is locked to group_size; Triton's WMMA lowering
        # needs at least 16 elements per K step.
        if c.group_size < 16:
            return False, (
                f"Group size ({c.group_size}) is below the WMMA minimum (16)"
            )

        if envs.VLLM_DISABLED_KERNELS and cls.__name__ in envs.VLLM_DISABLED_KERNELS:
            return False, "Disabled by VLLM_DISABLED_KERNELS"

        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        c = self.config
        size_bits = c.weight_type.size_bits
        pack_factor = 32 // size_bits

        def transform_w_q(x):
            assert isinstance(x, BasevLLMParameter)
            permute_param_layout_(x, input_dim=0, output_dim=1, packed_dim=0)
            x_cont = x.data.contiguous()
            # int32 nibble-packed -> uint8 byte-packed (low nibble = even K)
            x.data = x_cont.view(torch.uint8).contiguous()
            return x

        def transform_w_s(x):
            assert isinstance(x, BasevLLMParameter)
            permute_param_layout_(x, input_dim=0, output_dim=1)
            x.data = x.data.contiguous()
            return x.to(dtype=c.act_type)

        def transform_w_zp(x):
            # Same unpack/permute as the Conch kernel: stored as
            # [N//pack_factor, K//G] int32 -> [K//G, N] uint8.
            assert isinstance(x, BasevLLMParameter)
            packed = x.data
            mask = (1 << size_bits) - 1
            n_packed, k_groups = packed.shape
            n_full = n_packed * pack_factor

            shifts = torch.arange(
                0, 32, size_bits, dtype=torch.int32, device=packed.device
            )
            unpacked = (packed.unsqueeze(-1) >> shifts) & mask
            unpacked = unpacked.permute(1, 0, 2).reshape(k_groups, n_full)

            x.data = unpacked.to(torch.uint8).contiguous()
            if hasattr(x, "_input_dim"):
                x._input_dim = 0
            if hasattr(x, "_output_dim"):
                x._output_dim = 1
            if hasattr(x, "_packed_factor"):
                x._packed_factor = 1
            return x

        self._transform_param(layer, self.w_q_name, transform_w_q)
        self._transform_param(layer, self.w_s_name, transform_w_s)
        if c.zero_points:
            self._transform_param(layer, self.w_zp_name, transform_w_zp)
        elif self.w_zp_name is not None:
            layer.register_parameter(self.w_zp_name, None)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        c = self.config
        w_q, w_s, w_zp, _ = self._get_weight_params(layer)

        x_2d = x.reshape(-1, x.shape[-1]).contiguous()
        out_shape = x.shape[:-1] + (c.partition_weight_shape[1],)

        # Symmetric types embed the bias in the type definition; asymmetric
        # types carry an explicit zp tensor and ignore the constant bias.
        weight_bias = c.weight_type.bias if c.weight_type.has_bias() else 0

        out = _wna16_rdna3_gemm(
            x_2d,
            w_q.data,
            w_s.data,
            w_zp.data if w_zp is not None else None,
            weight_bias=weight_bias,
            size_bits=c.weight_type.size_bits,
            group_size=c.group_size,
        )

        if bias is not None:
            out.add_(bias)

        return out.reshape(out_shape)
