# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""INT2 per-token-head KV cache quantization plugin.

Owns the INT2-specific Lloyd-Max centroid lookup + reshape kernel
(4 x int2 per byte, full Hadamard pre-rotation, ``norm / d^1.5`` scale).
The shared sub-byte attention kernel + launcher + factory plumbing
lives in :mod:`_packed_core`.

Adding a new sub-byte packed mode = copy this file, change the name +
packing factor + reshape kernel (+ rotation hooks if needed).  No
other file needs editing.
"""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.triton_quant_kv import register
from vllm.v1.attention.ops.triton_quant_kv._hadamard import (
    fast_hadamard_transform,
)
from vllm.v1.attention.ops.triton_quant_kv._pack_unpack import pack_int2_quartet
from vllm.v1.attention.ops.triton_quant_kv._packed_core import _PackedFactory
from vllm.v1.attention.ops.triton_quant_kv.base import QuantKVSpec


@triton.jit
def _lloyd_max_quantize_4(z):
    """Quantize N(0,1) values to 4 Lloyd-Max centroids (INT2).

    Returns index in [0, 3].  Boundaries: [-0.9816, 0, 0.9816].
    """
    return tl.where(
        z < 0.0,
        tl.where(z < -0.9816, 0, 1).to(tl.uint8),
        tl.where(z < 0.9816, 2, 3).to(tl.uint8),
    )


@triton.jit
def _lloyd_max_dequant_4(idx):
    """Look up INT2 Lloyd-Max centroid for N(0,1).  idx in [0..3]."""
    return tl.where(
        idx < 2,
        tl.where(idx == 0, -1.5104, -0.4528),
        tl.where(idx == 2, 0.4528, 1.5104),
    )


@triton.jit
def _reshape_cache_int2_kernel(
    key_ptr,
    value_ptr,
    key_cache_ptr,
    value_cache_ptr,
    k_scale_cache_ptr,
    v_scale_cache_ptr,
    slot_mapping_ptr,
    stride_key_tok: tl.int64,
    stride_key_head: tl.int64,
    stride_val_tok: tl.int64,
    stride_val_head: tl.int64,
    stride_kc_blk: tl.int64,
    stride_kc_slot: tl.int64,
    stride_kc_head: tl.int64,
    stride_vc_blk: tl.int64,
    stride_vc_slot: tl.int64,
    stride_vc_head: tl.int64,
    stride_ks_blk: tl.int64,
    stride_ks_slot: tl.int64,
    stride_ks_head: tl.int64,
    stride_vs_blk: tl.int64,
    stride_vs_slot: tl.int64,
    stride_vs_head: tl.int64,
    block_size: tl.constexpr,
    head_size: tl.constexpr,
    head_size_v: tl.constexpr,
    PACKED_HEAD_PADDED: tl.constexpr,
):
    """INT2 Hadamard + Lloyd-Max 4-centroid quantization.

    Packs 4 × 2-bit indices per byte → head_size/4 bytes per head.
    """
    tok = tl.program_id(0)
    head = tl.program_id(1)

    slot = tl.load(slot_mapping_ptr + tok).to(tl.int64)
    if slot < 0:
        return

    blk = slot // block_size
    slot_in_blk = slot % block_size

    qtr_offs = tl.arange(0, PACKED_HEAD_PADDED)
    offs_0 = qtr_offs * 4
    offs_1 = qtr_offs * 4 + 1
    offs_2 = qtr_offs * 4 + 2
    offs_3 = qtr_offs * 4 + 3

    qtr_k = head_size // 4
    mask_0k = offs_0 < head_size
    mask_1k = offs_1 < head_size
    mask_2k = offs_2 < head_size
    mask_3k = offs_3 < head_size
    key_base = key_ptr + tok * stride_key_tok + head * stride_key_head

    k0 = tl.load(key_base + offs_0, mask=mask_0k, other=0.0).to(tl.float32)
    k1 = tl.load(key_base + offs_1, mask=mask_1k, other=0.0).to(tl.float32)
    k2 = tl.load(key_base + offs_2, mask=mask_2k, other=0.0).to(tl.float32)
    k3 = tl.load(key_base + offs_3, mask=mask_3k, other=0.0).to(tl.float32)

    k_sq = (
        tl.sum(tl.where(mask_0k, k0 * k0, 0.0))
        + tl.sum(tl.where(mask_1k, k1 * k1, 0.0))
        + tl.sum(tl.where(mask_2k, k2 * k2, 0.0))
        + tl.sum(tl.where(mask_3k, k3 * k3, 0.0))
    )
    k_norm = tl.sqrt(k_sq + 1e-12)

    k_inv_sigma = tl.sqrt(float(head_size)) / k_norm
    q0 = _lloyd_max_quantize_4(k0 * k_inv_sigma)
    q1 = _lloyd_max_quantize_4(k1 * k_inv_sigma)
    q2 = _lloyd_max_quantize_4(k2 * k_inv_sigma)
    q3 = _lloyd_max_quantize_4(k3 * k_inv_sigma)

    k_packed = pack_int2_quartet(q0, q1, q2, q3)
    tl.store(
        key_cache_ptr
        + blk * stride_kc_blk
        + slot_in_blk * stride_kc_slot
        + head * stride_kc_head
        + qtr_offs,
        k_packed,
        mask=qtr_offs < qtr_k,
    )

    # Store norm/d^1.5 as scale; see module docstring for the math.
    k_scale = k_norm / float(head_size**1.5)
    tl.store(
        k_scale_cache_ptr
        + blk * stride_ks_blk
        + slot_in_blk * stride_ks_slot
        + head * stride_ks_head,
        k_scale,
    )

    qtr_v = head_size_v // 4
    mask_0v = offs_0 < head_size_v
    mask_1v = offs_1 < head_size_v
    mask_2v = offs_2 < head_size_v
    mask_3v = offs_3 < head_size_v
    val_base = value_ptr + tok * stride_val_tok + head * stride_val_head

    v0 = tl.load(val_base + offs_0, mask=mask_0v, other=0.0).to(tl.float32)
    v1 = tl.load(val_base + offs_1, mask=mask_1v, other=0.0).to(tl.float32)
    v2 = tl.load(val_base + offs_2, mask=mask_2v, other=0.0).to(tl.float32)
    v3 = tl.load(val_base + offs_3, mask=mask_3v, other=0.0).to(tl.float32)

    v_sq = (
        tl.sum(tl.where(mask_0v, v0 * v0, 0.0))
        + tl.sum(tl.where(mask_1v, v1 * v1, 0.0))
        + tl.sum(tl.where(mask_2v, v2 * v2, 0.0))
        + tl.sum(tl.where(mask_3v, v3 * v3, 0.0))
    )
    v_norm = tl.sqrt(v_sq + 1e-12)
    v_inv_sigma = tl.sqrt(float(head_size_v)) / v_norm
    vq0 = _lloyd_max_quantize_4(v0 * v_inv_sigma)
    vq1 = _lloyd_max_quantize_4(v1 * v_inv_sigma)
    vq2 = _lloyd_max_quantize_4(v2 * v_inv_sigma)
    vq3 = _lloyd_max_quantize_4(v3 * v_inv_sigma)

    v_packed = pack_int2_quartet(vq0, vq1, vq2, vq3)
    tl.store(
        value_cache_ptr
        + blk * stride_vc_blk
        + slot_in_blk * stride_vc_slot
        + head * stride_vc_head
        + qtr_offs,
        v_packed,
        mask=qtr_offs < qtr_v,
    )

    v_scale = v_norm / float(head_size_v**1.5)
    tl.store(
        v_scale_cache_ptr
        + blk * stride_vs_blk
        + slot_in_blk * stride_vs_slot
        + head * stride_vs_head,
        v_scale,
    )


class Int2PerTokenHeadFactory(_PackedFactory):
    """KV cache plugin for ``int2_per_token_head``.

    Full (non-random) Hadamard pre-rotation gaussianizes K/V.  Because
    the Hadamard is its own inverse, the output rotation reuses the
    same transform; no ``softmax_scale`` adjustment is needed — the
    ``d^1.5`` factor is absorbed into the stored scale at write time.
    """

    spec = QuantKVSpec(
        name="int2_per_token_head",
        storage_dtype=torch.uint8,
        packing_factor=4,  # 4 x int2 per byte
        needs_per_token_head_scales=True,
        description="INT2 Lloyd-Max centroids + full Hadamard rotation",
    )

    _reshape_kernel = _reshape_cache_int2_kernel

    @staticmethod
    def _rotate_kv(x: torch.Tensor) -> torch.Tensor:
        return fast_hadamard_transform(x)

    @staticmethod
    def _rotate_q(q: torch.Tensor) -> torch.Tensor:
        return fast_hadamard_transform(q)

    @staticmethod
    def _unrotate_out(out: torch.Tensor, head_size: int) -> torch.Tensor:
        return fast_hadamard_transform(out.float())


register(Int2PerTokenHeadFactory())
