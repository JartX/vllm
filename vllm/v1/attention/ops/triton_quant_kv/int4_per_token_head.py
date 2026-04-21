# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""INT4 per-token-head KV cache quantization plugin.

Owns only the INT4-specific write kernel (2 x int4 per byte,
asymmetric min/max quantization with a 4-bit zero-point steganographed
into the low nibble of the float32 scale).  The shared sub-byte
attention kernel + launcher + factory plumbing lives in
:mod:`_packed_core`; this file only declares the mode-specific pieces.

Adding a new sub-byte packed mode = copy this file, change the name +
packing factor + reshape kernel (+ rotation hooks if needed).  No
other file needs editing — the loader auto-discovers new plugins.
"""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.triton_quant_kv import register
from vllm.v1.attention.ops.triton_quant_kv._hadamard import single_rht
from vllm.v1.attention.ops.triton_quant_kv._pack_unpack import pack_int4_nibbles
from vllm.v1.attention.ops.triton_quant_kv._packed_core import _PackedFactory
from vllm.v1.attention.ops.triton_quant_kv.base import QuantKVSpec


@triton.jit
def _reshape_cache_int4_kernel(
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
    """INT4 asymmetric quantization with zero-point steganography."""
    tok = tl.program_id(0)
    head = tl.program_id(1)

    slot = tl.load(slot_mapping_ptr + tok).to(tl.int64)
    if slot < 0:
        return

    blk = slot // block_size
    slot_in_blk = slot % block_size

    half_offs = tl.arange(0, PACKED_HEAD_PADDED)
    even_offs = half_offs * 2
    odd_offs = half_offs * 2 + 1

    half_k = head_size // 2
    even_k_mask = even_offs < head_size
    odd_k_mask = odd_offs < head_size
    key_base = key_ptr + tok * stride_key_tok + head * stride_key_head

    k_even = tl.load(key_base + even_offs, mask=even_k_mask, other=0.0).to(tl.float32)
    k_odd = tl.load(key_base + odd_offs, mask=odd_k_mask, other=0.0).to(tl.float32)

    k_min = tl.minimum(
        tl.min(tl.where(even_k_mask, k_even, float("inf"))),
        tl.min(tl.where(odd_k_mask, k_odd, float("inf"))),
    )
    k_max = tl.maximum(
        tl.max(tl.where(even_k_mask, k_even, float("-inf"))),
        tl.max(tl.where(odd_k_mask, k_odd, float("-inf"))),
    )
    k_scale = tl.maximum((k_max - k_min) / 15.0, 1e-6)
    k_zp_f = tl.clamp(
        tl.where(
            -k_min / k_scale >= 0,
            (-k_min / k_scale + 0.5).to(tl.int32),
            (-k_min / k_scale - 0.5).to(tl.int32),
        ).to(tl.float32),
        0.0,
        15.0,
    )

    inv_k = 1.0 / k_scale
    k_even_s = k_even * inv_k + k_zp_f
    k_odd_s = k_odd * inv_k + k_zp_f
    k_even_q = tl.clamp(
        tl.where(
            k_even_s >= 0,
            (k_even_s + 0.5).to(tl.int32),
            (k_even_s - 0.5).to(tl.int32),
        ).to(tl.float32),
        0.0,
        15.0,
    )
    k_odd_q = tl.clamp(
        tl.where(
            k_odd_s >= 0,
            (k_odd_s + 0.5).to(tl.int32),
            (k_odd_s - 0.5).to(tl.int32),
        ).to(tl.float32),
        0.0,
        15.0,
    )

    k_zp_int = k_zp_f.to(tl.int32)
    k_scale_bits = k_scale.to(tl.int32, bitcast=True)
    k_scale_packed = ((k_scale_bits & -16) | (k_zp_int & 0xF)).to(
        tl.float32, bitcast=True
    )

    tl.store(
        k_scale_cache_ptr
        + blk * stride_ks_blk
        + slot_in_blk * stride_ks_slot
        + head * stride_ks_head,
        k_scale_packed,
    )

    k_packed = pack_int4_nibbles(k_even_q.to(tl.uint8), k_odd_q.to(tl.uint8))
    tl.store(
        key_cache_ptr
        + blk * stride_kc_blk
        + slot_in_blk * stride_kc_slot
        + head * stride_kc_head
        + half_offs,
        k_packed,
        mask=half_offs < half_k,
    )

    half_v = head_size_v // 2
    even_v_mask = even_offs < head_size_v
    odd_v_mask = odd_offs < head_size_v
    val_base = value_ptr + tok * stride_val_tok + head * stride_val_head

    v_even = tl.load(val_base + even_offs, mask=even_v_mask, other=0.0).to(tl.float32)
    v_odd = tl.load(val_base + odd_offs, mask=odd_v_mask, other=0.0).to(tl.float32)

    v_min = tl.minimum(
        tl.min(tl.where(even_v_mask, v_even, float("inf"))),
        tl.min(tl.where(odd_v_mask, v_odd, float("inf"))),
    )
    v_max = tl.maximum(
        tl.max(tl.where(even_v_mask, v_even, float("-inf"))),
        tl.max(tl.where(odd_v_mask, v_odd, float("-inf"))),
    )
    v_scale = tl.maximum((v_max - v_min) / 15.0, 1e-6)
    v_zp_f = tl.clamp(
        tl.where(
            -v_min / v_scale >= 0,
            (-v_min / v_scale + 0.5).to(tl.int32),
            (-v_min / v_scale - 0.5).to(tl.int32),
        ).to(tl.float32),
        0.0,
        15.0,
    )

    inv_v = 1.0 / v_scale
    v_even_s = v_even * inv_v + v_zp_f
    v_odd_s = v_odd * inv_v + v_zp_f
    v_even_q = tl.clamp(
        tl.where(
            v_even_s >= 0,
            (v_even_s + 0.5).to(tl.int32),
            (v_even_s - 0.5).to(tl.int32),
        ).to(tl.float32),
        0.0,
        15.0,
    )
    v_odd_q = tl.clamp(
        tl.where(
            v_odd_s >= 0,
            (v_odd_s + 0.5).to(tl.int32),
            (v_odd_s - 0.5).to(tl.int32),
        ).to(tl.float32),
        0.0,
        15.0,
    )

    v_zp_int = v_zp_f.to(tl.int32)
    v_scale_bits = v_scale.to(tl.int32, bitcast=True)
    v_scale_packed = ((v_scale_bits & -16) | (v_zp_int & 0xF)).to(
        tl.float32, bitcast=True
    )

    tl.store(
        v_scale_cache_ptr
        + blk * stride_vs_blk
        + slot_in_blk * stride_vs_slot
        + head * stride_vs_head,
        v_scale_packed,
    )

    v_packed = pack_int4_nibbles(v_even_q.to(tl.uint8), v_odd_q.to(tl.uint8))
    tl.store(
        value_cache_ptr
        + blk * stride_vc_blk
        + slot_in_blk * stride_vc_slot
        + head * stride_vc_head
        + half_offs,
        v_packed,
        mask=half_offs < half_v,
    )


class Int4PerTokenHeadFactory(_PackedFactory):
    """KV cache plugin for ``int4_per_token_head``.

    RHT pre-rotation gaussianizes K/V so the asymmetric quantization
    has a useful dynamic range.  The forward RHT has norm
    ``sqrt(head_size)``, so ``softmax_scale`` is divided by
    ``head_size`` and the inverse RHT on the attention output divides
    by ``head_size`` as well.
    """

    spec = QuantKVSpec(
        name="int4_per_token_head",
        storage_dtype=torch.uint8,
        packing_factor=2,  # 2 x int4 per byte
        needs_per_token_head_scales=True,
        description="INT4 asymmetric per-(token, head) + RHT rotation",
    )

    _reshape_kernel = _reshape_cache_int4_kernel

    @staticmethod
    def _rotate_kv(x: torch.Tensor) -> torch.Tensor:
        return single_rht(x)

    @staticmethod
    def _rotate_q(q: torch.Tensor) -> torch.Tensor:
        return single_rht(q)

    @staticmethod
    def _unrotate_out(out: torch.Tensor, head_size: int) -> torch.Tensor:
        return single_rht(out.float(), inverse=True) / head_size

    @staticmethod
    def _transform_softmax_scale(scale: float, head_size: int) -> float:
        return scale / head_size


register(Int4PerTokenHeadFactory())
