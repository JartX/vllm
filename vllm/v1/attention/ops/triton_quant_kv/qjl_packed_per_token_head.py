# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""INT2 Lloyd-Max + 1-bit QJL residual sign (``INT2_QJL_PER_TOKEN_HEAD``).

Builds on :mod:`packed_per_token_head` (plain INT2) by storing one extra
bit per coord: ``sign(qjl_project(z - centroid))`` where ``qjl_project``
is a single Hadamard-JL rotation (seed-2) of the Lloyd-Max residual in
the unit-Gaussian space ``z = x_rotated · sqrt(d) / norm``.

At read time the attention kernel adds a residual-correction term:

    correction_per_score = softmax_scale · stored_scale_K
                           · (sqrt(D) · sqrt(2/π) / sqrt(d)) · <Q_jl, b_K>

where ``D = 0.1175`` is the per-coord Lloyd-Max-4 MSE on ``N(0, 1)``,
``b_K ∈ {-1,+1}^d`` are the stored sign bits, and
``Q_jl = qjl_project(double_rht(Q))`` is precomputed once per attention
call.  Full derivation in :meth:`Int2QJLPerTokenHeadBackend._qjl_correction_const`.

Per-head cache layout:

  +----------------------------+--------------------+
  | LM4 indices (packed ×4/B)  | QJL signs (1 bit)  |
  | ``head_size / 4``          | ``head_size / 8``  |
  +----------------------------+--------------------+

Total bytes/head: ``head_size · (1/4 + 1/8)``.  For d=128: 48 B
(vs 32 B for plain INT2 — +50% memory, 1 extra bit/coord of score
accuracy).  V-side also carries the QJL bytes so the cache layout is
uniform across K/V, but only the K-side signs are consumed by the
attention kernel.  V QJL bits are currently dead storage — removing
them is a possible follow-up once an asymmetric-K/V page layout lands.
"""

from __future__ import annotations

import math

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.triton_quant_kv import register
from vllm.v1.attention.ops.triton_quant_kv._hadamard import (
    double_rht,
    qjl_project,
)
from vllm.v1.attention.ops.triton_quant_kv.base import QuantKVBackend
from vllm.v1.attention.ops.triton_quant_kv.packed_per_token_head import (
    _launch_packed_attn,
)
from vllm.v1.kv_cache_interface import KVQuantMode

# Per-coord Lloyd-Max-4 MSE on N(0, 1).  Max (1960), Table I.
_LM4_MSE = 0.1175

# sqrt(2/π) — E[|X|] for X ~ N(0, 1).  The 1-bit QJL estimator uses this
# as the reconstruction factor: E[X | sign(X) = +1] = σ · sqrt(2/π).
_SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)


# ---------------------------------------------------------------------------
# Python-side quantization.  Closed-form ``torch.where`` expressions — no
# constant-tensor allocations, so this is safe inside a CUDA-graph capture
# (``torch.tensor(...)`` on a captured stream is forbidden).  Thresholds
# and centroids must match the Lloyd-Max-4 LUTs baked into the Triton
# attention kernel (Max 1960 Table I, uncorrected — same convention as
# the plain INT2 backend).
# ---------------------------------------------------------------------------


def _lm4_quantize(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Lloyd-Max 4-level quantizer for N(0, 1).

    Returns ``(indices [0..3], dequantized centroid)``.  Boundaries
    ``±0.9816``; centroids ``±0.4528, ±1.5104``.
    """
    abs_z = z.abs()
    is_outer = abs_z > 0.9816
    is_pos = z >= 0
    # idx layout: 0: < -0.9816, 1: [-0.9816, 0), 2: [0, 0.9816), 3: ≥ 0.9816
    indices = torch.where(
        is_pos,
        2 + is_outer.to(torch.long),
        1 - is_outer.to(torch.long),
    )
    mag = torch.where(is_outer, 1.5104, 0.4528)
    centroid = torch.where(is_pos, mag, -mag)
    return indices, centroid


def _quantize_packed_qjl(
    x_rotated: torch.Tensor,
    *,
    head_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Lloyd-Max quantize + pack + QJL-project + sign-pack.

    Returns ``(packed_combined, scale)`` where:

      * ``packed_combined`` has shape ``(num_tokens, num_heads,
        head_size/4 + head_size/8)``: packed LM4 nibbles followed by
        QJL sign bytes (little-endian within each byte).
      * ``scale`` is ``norm_rotated / d^2.5`` — matches plain INT2 so
        the attention kernel can reuse the same ``scale × k_scale`` fuse.

    ``x_rotated`` is the input *after* :func:`double_rht` with shape
    ``(num_tokens, num_kv_heads, head_size)``.
    """
    num_tokens, num_heads, _ = x_rotated.shape
    x = x_rotated.float()

    # Per-(token, head) norm in the rotated domain.
    norm = x.norm(dim=-1, keepdim=True).clamp(min=1e-6)  # [N, H, 1]
    inv_sigma = math.sqrt(head_size) / norm
    z = x * inv_sigma  # unit-variance Gaussian per coord

    indices, centroid_dequant = _lm4_quantize(z)

    # Pack 4 × 2-bit indices per byte.
    i0 = indices[..., 0::4]
    i1 = indices[..., 1::4]
    i2 = indices[..., 2::4]
    i3 = indices[..., 3::4]
    packed_data = (
        (i0 & 0x3) | ((i1 & 0x3) << 2) | ((i2 & 0x3) << 4) | ((i3 & 0x3) << 6)
    ).to(torch.uint8)

    # Residual r = z - dequant(z) (unit-variance Lloyd-Max residue).
    residual = z - centroid_dequant

    # Hadamard-JL projection (seed-2) decorrelates the sign bits from the
    # double-RHT data rotation.  Only ``sign`` is retained.
    s = qjl_project(residual)
    sign_bits = (s > 0).to(torch.uint8)  # ∈ {0, 1}

    # Pack 8 sign bits per byte, little-endian (bit 0 = coord 8*b + 0).
    sb = sign_bits.view(num_tokens, num_heads, head_size // 8, 8)
    packed_signs = (
        sb[..., 0]
        | (sb[..., 1] << 1)
        | (sb[..., 2] << 2)
        | (sb[..., 3] << 3)
        | (sb[..., 4] << 4)
        | (sb[..., 5] << 5)
        | (sb[..., 6] << 6)
        | (sb[..., 7] << 7)
    )

    # Concatenate: data section first, signs section second.
    packed_combined = torch.cat([packed_data, packed_signs], dim=-1)  # [N, H, hsz']

    scale = (norm.squeeze(-1) / (head_size**2.5)).to(torch.float32)
    return packed_combined, scale


# ---------------------------------------------------------------------------
# Triton store-only scatter kernel.  The quantization math runs in PyTorch
# above; this kernel just writes the per-(token, head) byte-string into
# the paged cache at ``slot_mapping``.
# ---------------------------------------------------------------------------


@triton.jit
def _qjl_store_kernel(
    packed_k_ptr,
    packed_v_ptr,
    scale_k_ptr,
    scale_v_ptr,
    key_cache_ptr,
    value_cache_ptr,
    k_scale_cache_ptr,
    v_scale_cache_ptr,
    slot_mapping_ptr,
    stride_pkg_tok: tl.int64,
    stride_pkg_head: tl.int64,
    stride_pkg_byte: tl.constexpr,
    stride_pkv_tok: tl.int64,
    stride_pkv_head: tl.int64,
    stride_pkv_byte: tl.constexpr,
    stride_kc_blk: tl.int64,
    stride_kc_slot: tl.int64,
    stride_kc_head: tl.int64,
    stride_kc_byte: tl.constexpr,
    stride_vc_blk: tl.int64,
    stride_vc_slot: tl.int64,
    stride_vc_head: tl.int64,
    stride_vc_byte: tl.constexpr,
    stride_ks_blk: tl.int64,
    stride_ks_slot: tl.int64,
    stride_ks_head: tl.int64,
    stride_vs_blk: tl.int64,
    stride_vs_slot: tl.int64,
    stride_vs_head: tl.int64,
    stride_sk_tok: tl.int64,
    stride_sk_head: tl.int64,
    stride_sv_tok: tl.int64,
    stride_sv_head: tl.int64,
    block_size: tl.constexpr,
    bytes_per_head_k: tl.constexpr,
    bytes_per_head_v: tl.constexpr,
    BYTES_PADDED: tl.constexpr,
):
    tok = tl.program_id(0)
    head = tl.program_id(1)

    slot = tl.load(slot_mapping_ptr + tok).to(tl.int64)
    if slot < 0:
        return

    blk = slot // block_size
    slot_in_blk = slot % block_size

    byte_offs = tl.arange(0, BYTES_PADDED)
    k_mask = byte_offs < bytes_per_head_k
    v_mask = byte_offs < bytes_per_head_v

    k_bytes = tl.load(
        packed_k_ptr
        + tok * stride_pkg_tok
        + head * stride_pkg_head
        + byte_offs * stride_pkg_byte,
        mask=k_mask,
        other=0,
    )
    tl.store(
        key_cache_ptr
        + blk * stride_kc_blk
        + slot_in_blk * stride_kc_slot
        + head * stride_kc_head
        + byte_offs * stride_kc_byte,
        k_bytes,
        mask=k_mask,
    )

    v_bytes = tl.load(
        packed_v_ptr
        + tok * stride_pkv_tok
        + head * stride_pkv_head
        + byte_offs * stride_pkv_byte,
        mask=v_mask,
        other=0,
    )
    tl.store(
        value_cache_ptr
        + blk * stride_vc_blk
        + slot_in_blk * stride_vc_slot
        + head * stride_vc_head
        + byte_offs * stride_vc_byte,
        v_bytes,
        mask=v_mask,
    )

    k_scale = tl.load(scale_k_ptr + tok * stride_sk_tok + head * stride_sk_head)
    tl.store(
        k_scale_cache_ptr
        + blk * stride_ks_blk
        + slot_in_blk * stride_ks_slot
        + head * stride_ks_head,
        k_scale,
    )
    v_scale = tl.load(scale_v_ptr + tok * stride_sv_tok + head * stride_sv_head)
    tl.store(
        v_scale_cache_ptr
        + blk * stride_vs_blk
        + slot_in_blk * stride_vs_slot
        + head * stride_vs_head,
        v_scale,
    )


def _store_packed(
    *,
    packed_k: torch.Tensor,  # [N, H, bytes_k]
    packed_v: torch.Tensor,  # [N, H, bytes_v]
    scale_k: torch.Tensor,  # [N, H]
    scale_v: torch.Tensor,  # [N, H]
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    k_scale_cache: torch.Tensor,
    v_scale_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    num_tokens, num_kv_heads, bytes_k = packed_k.shape
    bytes_v = packed_v.shape[-1]
    bytes_padded = triton.next_power_of_2(max(bytes_k, bytes_v))
    if current_platform.is_rocm() or current_platform.is_xpu():
        num_warps = 4
    else:
        num_warps = min(16, max(1, bytes_padded // 32))

    _qjl_store_kernel[(num_tokens, num_kv_heads)](
        packed_k_ptr=packed_k,
        packed_v_ptr=packed_v,
        scale_k_ptr=scale_k,
        scale_v_ptr=scale_v,
        key_cache_ptr=key_cache,
        value_cache_ptr=value_cache,
        k_scale_cache_ptr=k_scale_cache,
        v_scale_cache_ptr=v_scale_cache,
        slot_mapping_ptr=slot_mapping,
        stride_pkg_tok=packed_k.stride(0),
        stride_pkg_head=packed_k.stride(1),
        stride_pkg_byte=packed_k.stride(2),
        stride_pkv_tok=packed_v.stride(0),
        stride_pkv_head=packed_v.stride(1),
        stride_pkv_byte=packed_v.stride(2),
        stride_kc_blk=key_cache.stride(0),
        stride_kc_slot=key_cache.stride(1),
        stride_kc_head=key_cache.stride(2),
        stride_kc_byte=key_cache.stride(3),
        stride_vc_blk=value_cache.stride(0),
        stride_vc_slot=value_cache.stride(1),
        stride_vc_head=value_cache.stride(2),
        stride_vc_byte=value_cache.stride(3),
        stride_ks_blk=k_scale_cache.stride(0),
        stride_ks_slot=k_scale_cache.stride(1),
        stride_ks_head=k_scale_cache.stride(2),
        stride_vs_blk=v_scale_cache.stride(0),
        stride_vs_slot=v_scale_cache.stride(1),
        stride_vs_head=v_scale_cache.stride(2),
        stride_sk_tok=scale_k.stride(0),
        stride_sk_head=scale_k.stride(1),
        stride_sv_tok=scale_v.stride(0),
        stride_sv_head=scale_v.stride(1),
        block_size=key_cache.shape[1],
        bytes_per_head_k=bytes_k,
        bytes_per_head_v=bytes_v,
        BYTES_PADDED=bytes_padded,
        num_warps=num_warps,
    )


# ---------------------------------------------------------------------------
# Backend class
# ---------------------------------------------------------------------------


class Int2QJLPerTokenHeadBackend(QuantKVBackend):
    """KV cache backend for ``KVQuantMode.INT2_QJL_PER_TOKEN_HEAD``.

    Same double-RHT + Lloyd-Max-4 skeleton as plain INT2, plus a 1-bit
    QJL sign per coord that lets the attention kernel fold a bias-free
    residual correction into the raw score.
    """

    mode = KVQuantMode.INT2_QJL_PER_TOKEN_HEAD
    packing_factor = 4
    needs_scale_caches = True

    def packed_head_size(self, head_size: int) -> int:
        # Data (LM4 packed) + QJL signs sections.
        assert head_size % 8 == 0, (
            f"head_size={head_size} must be a multiple of 8 for INT2_QJL"
        )
        return head_size // 4 + head_size // 8

    @staticmethod
    def _qjl_correction_const(head_size: int) -> float:
        # Score-level residual correction:
        #   raw_score = <Q_rot, centroid_K> + <Q_rot, r_K_rot>
        # where r_K_rot is the K residual in the rotated domain.  The
        # stored QJL signs ``b_K = sign(H·D_jl·r_K_rot)`` give the
        # 1-bit JL estimator
        #   <Q_rot, r_K_rot> ≈ sqrt(D·d)·sqrt(2/π) · <Q_jl, b_K> / d
        #                    = sqrt(D)·sqrt(2/π)/sqrt(d) · <Q_jl, b_K>
        # which absorbs cleanly into the fused ``softmax_scale · scale_K``
        # multiplier:
        #   correction = softmax_scale · scale_K
        #                · sqrt(D)·sqrt(2/π)/sqrt(d) · <Q_jl, b_K>
        # (D = per-coord Lloyd-Max MSE on N(0,1)).
        return math.sqrt(_LM4_MSE) * _SQRT_2_OVER_PI / math.sqrt(head_size)

    def reshape_and_cache(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        *,
        k_scale_cache: torch.Tensor | None = None,
        v_scale_cache: torch.Tensor | None = None,
    ) -> None:
        assert k_scale_cache is not None and v_scale_cache is not None, (
            f"{self.mode.name} requires k_scale_cache / v_scale_cache"
        )
        head_size = key.shape[-1]
        head_size_v = value.shape[-1]

        # Double-RHT matches plain INT2 so the centroids + scale math
        # carry over unchanged.
        key_rot = double_rht(key.float())
        value_rot = double_rht(value.float())

        packed_k, scale_k = _quantize_packed_qjl(key_rot, head_size=head_size)
        packed_v, scale_v = _quantize_packed_qjl(value_rot, head_size=head_size_v)

        _store_packed(
            packed_k=packed_k,
            packed_v=packed_v,
            scale_k=scale_k,
            scale_v=scale_v,
            key_cache=key_cache,
            value_cache=value_cache,
            k_scale_cache=k_scale_cache,
            v_scale_cache=v_scale_cache,
            slot_mapping=slot_mapping,
        )

    def unified_attention(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        out: torch.Tensor,
        *,
        cu_seqlens_q: torch.Tensor,
        max_seqlen_q: int,
        seqused_k: torch.Tensor,
        max_seqlen_k: int,
        softmax_scale: float,
        window_size: tuple[int, int],
        block_table: torch.Tensor,
        softcap: float,
        sinks: torch.Tensor | None,
        alibi_slopes: torch.Tensor | None,
        use_alibi_sqrt: bool,
        qq_bias: torch.Tensor | None,
        output_scale: torch.Tensor | None,
        mm_prefix_range: torch.Tensor | None,
        k_scale_cache: torch.Tensor | None = None,
        v_scale_cache: torch.Tensor | None = None,
        seq_threshold_3D: int | None = None,
        num_par_softmax_segments: int | None = None,
        softmax_segm_output: torch.Tensor | None = None,
        softmax_segm_max: torch.Tensor | None = None,
        softmax_segm_expsum: torch.Tensor | None = None,
    ) -> None:
        assert k_scale_cache is not None and v_scale_cache is not None
        q_orig_dtype = q.dtype
        head_size = q.shape[2]

        # Data-side rotation matches plain INT2.
        q_rot_f = double_rht(q.float())
        # JL projection of the rotated query for the QJL correction term.
        # Stored alongside Q so the kernel can load the two in parallel.
        q_jl = qjl_project(q_rot_f).contiguous().to(q_orig_dtype)
        q_rot = q_rot_f.to(q_orig_dtype)

        _launch_packed_attn(
            q=q_rot,
            k_cache=k_cache,
            v_cache=v_cache,
            out=out,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            seqused_k=seqused_k,
            softmax_scale=softmax_scale,
            window_size=window_size,
            block_table=block_table,
            softcap=softcap,
            sinks=sinks,
            alibi_slopes=alibi_slopes,
            use_alibi_sqrt=use_alibi_sqrt,
            qq_bias=qq_bias,
            output_scale=output_scale,
            mm_prefix_range=mm_prefix_range,
            k_scale_cache=k_scale_cache,
            v_scale_cache=v_scale_cache,
            seq_threshold_3D=seq_threshold_3D,
            num_par_softmax_segments=num_par_softmax_segments,
            softmax_segm_output=softmax_segm_output,
            softmax_segm_max=softmax_segm_max,
            softmax_segm_expsum=softmax_segm_expsum,
            packing_factor=self.packing_factor,
            q_jl=q_jl,
            qjl_correction_const=self._qjl_correction_const(head_size),
            qjl_byte_offset=head_size // self.packing_factor,
        )

        out_f = double_rht(out.float(), inverse=True)
        out.copy_(out_f.to(q_orig_dtype))


register(Int2QJLPerTokenHeadBackend())
