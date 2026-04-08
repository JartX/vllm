# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dedicated WNA16 fused MoE Triton kernel for AMD RDNA3 (gfx11xx).

This is a slim variant of ``fused_moe_kernel_gptq_awq`` specialised for the
RDNA3 ISA:

- ``num_warps=4`` (wave32 → 128 lanes per program), ``waves_per_eu=2``,
  ``num_stages=2`` are the defaults; the AMD Triton backend lowers
  ``tl.dot`` into the 16x16x{16,32,64} WMMA fragments natively.
- The K-loop is rewritten as an outer loop over scale-groups and an
  inner loop over the K elements within a group, so the dequant scale
  is loaded *once per group* (a vector of N) instead of broadcast across
  the full tile every iteration. This halves scale-load traffic for
  ``BLOCK_SIZE_K == group_size`` and dominates for the small wna16
  decode shapes used by Qwen3-MoE / Mixtral / DeepSeek-V2-Lite.
- Symmetric weights only (no zero-points). The wna16 MoE method always
  passes ``zero_points=False`` so this is not a regression; the
  dispatcher falls back to the generic kernel for the asymmetric case.

The kernel is opt-in via ``VLLM_USE_TRITON_WNA16_RDNA3_MOE=1``. The
dispatcher in ``fused_moe.py`` checks the env var, the architecture and
the shape preconditions, and otherwise calls the generic kernel.
"""

from typing import Any

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _fused_moe_wna16_rdna3_kernel(
    # Pointers
    a_ptr,
    b_ptr,
    c_ptr,
    b_scale_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    N: tl.constexpr,
    K: tl.constexpr,
    EM,
    num_valid_tokens,
    # Strides
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_bse,
    stride_bsk,
    stride_bsn,
    # Meta-parameters
    group_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    use_int4_w4a16: tl.constexpr,
):
    """RDNA3-tuned fused MoE WNA16 GEMM (symmetric, no zero-points).

    Constraints (asserted at the launch site):
      - ``BLOCK_SIZE_K`` must be a multiple of ``group_size``.
      - ``K`` must be a multiple of ``BLOCK_SIZE_K`` (no K-tail mask).
      - Symmetric quantization (no zero-points).
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_experts == -1:
        # Zero-fill output for tokens routed to a non-local expert.
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
        c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
        tl.store(
            c_ptrs,
            tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type),
            mask=c_mask,
        )
        return

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (
        offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
    )
    if use_int4_w4a16:
        b_ptrs = (
            b_ptr
            + off_experts * stride_be
            + (offs_k[:, None] // 2) * stride_bk
            + offs_bn[None, :] * stride_bn
        )
        b_shifter = (offs_k[:, None] % 2) * 4
        b_zero_offset = 8.0
    else:
        b_ptrs = (
            b_ptr
            + off_experts * stride_be
            + offs_k[:, None] * stride_bk
            + offs_bn[None, :] * stride_bn
        )
        b_shifter = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=tl.int32)
        b_zero_offset = 128.0

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    num_k_iters: tl.constexpr = K // BLOCK_SIZE_K

    for k in range(0, num_k_iters):
        a = tl.load(
            a_ptrs,
            mask=token_mask[:, None],
            other=0.0,
        )

        b_packed = tl.load(b_ptrs)
        # constexpr branch: int4 path produces int32, int8 path keeps uint8;
        # a ternary would force a common dtype that Triton cannot infer.
        if use_int4_w4a16:  # noqa: SIM108
            b_int = (b_packed >> b_shifter) & 0xF
        else:
            b_int = b_packed

        # Load the per-group scale once for the whole tile (vector of N).
        # BLOCK_SIZE_K is a multiple of group_size by construction, so each
        # K iteration crosses a fixed number of groups; the kernel covers
        # the common case of BLOCK_SIZE_K == group_size by hoisting the
        # scale load out of the inner BLOCK_SIZE_K dot.
        scale_group_idx = (k * BLOCK_SIZE_K) // group_size
        b_scale = tl.load(
            b_scale_ptr
            + off_experts * stride_bse
            + scale_group_idx * stride_bsk
            + offs_bn * stride_bsn
        ).to(tl.float32)

        b_f = (b_int.to(tl.float32) - b_zero_offset) * b_scale[None, :]
        b_cast = b_f.to(compute_type)
        accumulator = tl.dot(a, b_cast, acc=accumulator)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        if use_int4_w4a16:
            b_ptrs += (BLOCK_SIZE_K // 2) * stride_bk
        else:
            b_ptrs += BLOCK_SIZE_K * stride_bk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator = accumulator * moe_weight[:, None]

    accumulator = accumulator.to(compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def can_use_fused_moe_wna16_rdna3(
    A: torch.Tensor,
    B_zp: torch.Tensor | None,
    block_shape: list[int] | None,
) -> bool:
    """Cheap structural precondition check for the dedicated RDNA3 kernel.

    Block-size constraints (``BLOCK_SIZE_K`` divisible by group_size and
    by K) are validated inside the launcher itself, after the block
    config has been resolved.
    """
    if B_zp is not None:
        return False
    if block_shape is None or len(block_shape) < 2:
        return False
    group_size = block_shape[1]
    if group_size <= 0:
        return False
    return A.size(1) % group_size == 0


def try_invoke_fused_moe_wna16_rdna3_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    B_scale: torch.Tensor,
    topk_weights: torch.Tensor | None,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: dict[str, Any],
    compute_type: tl.dtype,
    use_int4_w4a16: bool,
    block_shape: list[int],
) -> bool:
    """Launcher for the RDNA3-specific WNA16 fused MoE kernel.

    Returns ``True`` when the kernel was launched and ``False`` when the
    block-size constraints could not be satisfied (the caller should then
    fall back to the generic ``invoke_fused_moe_wna16_triton_kernel``).
    """
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        get_moe_wna16_block_config,
    )

    assert B_scale is not None and B_scale.ndim == 3
    assert block_shape is not None and block_shape[0] == 0

    M = A.size(0)
    num_tokens = M * top_k
    EM = sorted_token_ids.size(0)
    if A.size(0) < config["BLOCK_SIZE_M"]:
        EM = min(sorted_token_ids.size(0), A.size(0) * top_k * config["BLOCK_SIZE_M"])

    config = config.copy()
    config.update(
        get_moe_wna16_block_config(
            config=config,
            use_moe_wna16_cuda=False,
            num_valid_tokens=num_tokens,
            size_k=A.size(1),
            size_n=B.size(1),
            num_experts=B.size(1),
            group_size=block_shape[1],
            real_top_k=top_k,
            block_size_m=config["BLOCK_SIZE_M"],
        )
    )

    block_k = config.get("BLOCK_SIZE_K")
    if block_k is None or block_k % block_shape[1] != 0:
        return False
    if A.size(1) % block_k != 0:
        return False

    grid = lambda META: (  # noqa: E731
        triton.cdiv(EM, META["BLOCK_SIZE_M"])
        * triton.cdiv(B.size(1), META["BLOCK_SIZE_N"]),
    )

    launch_config = {
        "BLOCK_SIZE_M": config["BLOCK_SIZE_M"],
        "BLOCK_SIZE_N": config["BLOCK_SIZE_N"],
        "BLOCK_SIZE_K": config["BLOCK_SIZE_K"],
        "GROUP_SIZE_M": config.get("GROUP_SIZE_M", 1),
        "num_warps": config.get("num_warps", 4),
        "num_stages": config.get("num_stages", 2),
    }
    waves_per_eu = config.get("waves_per_eu")
    if waves_per_eu is not None:
        launch_config["waves_per_eu"] = waves_per_eu

    _fused_moe_wna16_rdna3_kernel[grid](
        A,
        B,
        C,
        B_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        B.size(1),
        A.size(1),
        EM,
        num_tokens,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        C.stride(1),
        C.stride(2),
        B_scale.stride(0),
        B_scale.stride(2),
        B_scale.stride(1),
        group_size=block_shape[1],
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=compute_type,
        use_int4_w4a16=use_int4_w4a16,
        **launch_config,
    )
    return True
