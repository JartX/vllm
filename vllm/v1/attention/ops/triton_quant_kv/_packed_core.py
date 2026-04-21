# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared core for sub-byte packed per-token-head KV quantization.

The two concrete modes live in their own plugin files:
:mod:`int4_per_token_head` (2×int4/byte, RHT + asymmetric zp) and
:mod:`int2_per_token_head` (4×int2/byte, Hadamard + Lloyd-Max).  This
module provides the three pieces that both modes reuse:

* :func:`_attn_packed` — a single Triton attention kernel parameterised
  by ``PACKING_FACTOR: tl.constexpr`` (2 → INT4, 4 → INT2).  All
  mode-specific sections (unpack, score correction, dequant) are gated
  by constexpr branches that Triton prunes at compile time, so each
  concrete launch is as tight as a bespoke per-mode kernel.

* :func:`_launch_packed_attn` — the Python-side launcher with tile /
  warp heuristics and 2D vs 3D decode dispatch.

* :class:`_PackedFactory` — the common base class.  Concrete plugins
  override the ``_rotate_kv`` / ``_rotate_q`` / ``_unrotate_out`` /
  ``_transform_softmax_scale`` hooks plus bind ``_reshape_kernel`` to
  their mode-specific reshape (write) kernel.

The leading underscore in the filename keeps this module out of the
external plugin discovery scan — see
:func:`vllm.v1.attention.ops.triton_quant_kv._ensure_external_loaded`.
"""

from __future__ import annotations

from typing import Any

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.triton_attention_helpers import (
    apply_alibi_to_score,
    apply_softcap,
    cdiv_fn,
    compute_kv_seq_mask,
    compute_tile_loop_bounds,
    init_softmax_M,
    load_qq_bias_tile,
    resolve_seq_and_query_len,
    softmax_step,
    store_segm_reduce_scalars,
)
from vllm.v1.attention.ops.triton_quant_kv._pack_unpack import (
    unpack_int2_quartet,
    unpack_int4_nibbles,
)
from vllm.v1.attention.ops.triton_quant_kv.base import QuantKVPlugin
from vllm.v1.attention.ops.triton_unified_attention import reduce_segments

float8_info = torch.finfo(current_platform.fp8_dtype())


# ---------------------------------------------------------------------------
# INT2 centroid dequant (used by the shared attention kernel's
# ``PACKING_FACTOR == 4`` constexpr branch).  The quantize-side
# companion (``_lloyd_max_quantize_4``) lives in ``int2_per_token_head``
# because it's only used by the INT2 reshape kernel at write time; the
# dequant lookup must live here so Triton resolves the name when
# compiling ``_attn_packed``.
# ---------------------------------------------------------------------------
@triton.jit
def _lloyd_max_dequant_4(idx):
    """Look up INT2 Lloyd-Max centroid for N(0, 1) inputs.  idx in [0..3]."""
    return tl.where(
        idx < 2,
        tl.where(idx == 0, -1.5104, -0.4528),
        tl.where(idx == 2, 0.4528, 1.5104),
    )


@triton.jit
def _attn_packed(
    # Output destinations.  In 2D mode the final result is written into
    # ``output_ptr``; in 3D mode per-segment partials go into the three
    # ``segm_*`` tensors and ``output_ptr`` is unused.
    output_ptr,
    segm_output_ptr,
    segm_max_ptr,
    segm_expsum_ptr,
    query_ptr,
    key_cache_ptr,
    value_cache_ptr,
    sink_ptr,
    block_tables_ptr,
    seq_lens_ptr,
    alibi_slopes_ptr,
    qq_bias_ptr,
    scale,
    out_scale,
    softcap,
    k_scale_cache_ptr,
    v_scale_cache_ptr,
    num_query_heads: tl.constexpr,
    num_queries_per_kv: tl.constexpr,
    block_table_stride: tl.int64,
    query_stride_0: tl.int64,
    query_stride_1: tl.int64,
    output_stride_0: tl.int64,
    output_stride_1: tl.int64,
    qq_bias_stride_0: tl.int64,
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
    PACKED_HEAD_PADDED: tl.constexpr,  # HEAD_SIZE / PACKING_FACTOR, rounded up
    USE_ALIBI_SLOPES: tl.constexpr,
    USE_ALIBI_SQRT: tl.constexpr,
    USE_QQ_BIAS: tl.constexpr,
    USE_SOFTCAP: tl.constexpr,
    USE_SINKS: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    USE_MM_PREFIX: tl.constexpr,
    MAX_MM_RANGES: tl.constexpr,
    mm_prefix_range_ptr,
    stride_k_cache_0: tl.int64,
    stride_k_cache_1: tl.int64,
    stride_k_cache_2: tl.int64,
    stride_k_cache_3: tl.constexpr,
    stride_v_cache_0: tl.int64,
    stride_v_cache_1: tl.int64,
    stride_v_cache_2: tl.int64,
    stride_v_cache_3: tl.constexpr,
    stride_ks_blk: tl.int64,
    stride_ks_slot: tl.int64,
    stride_ks_head: tl.int64,
    stride_vs_blk: tl.int64,
    stride_vs_slot: tl.int64,
    stride_vs_head: tl.int64,
    query_start_len_ptr,
    BLOCK_Q: tl.constexpr,
    num_seqs: tl.int32,
    BLOCK_M: tl.constexpr,
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,
    USE_FP8: tl.constexpr,
    IS_3D: tl.constexpr,
    # 2 → INT4 nibble pair (asymmetric + zp); 4 → INT2 quartet
    # (Lloyd-Max centroids).  All mode-specific branches below gate on
    # this value.
    PACKING_FACTOR: tl.constexpr,
    FP8_MIN: tl.constexpr = float8_info.min,
    FP8_MAX: tl.constexpr = float8_info.max,
):
    # -----------------------------------------------------------------
    # Shared prologue: sequence lookup, q-block bounds, early returns.
    # -----------------------------------------------------------------
    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    segm_idx = tl.program_id(2) if IS_3D else 0

    (
        seq_idx,
        q_block_local_idx,
        cur_batch_in_all_start_index,
        cur_batch_query_len,
        seq_len,
    ) = resolve_seq_and_query_len(
        query_start_len_ptr, seq_lens_ptr, q_block_global_idx, num_seqs, BLOCK_Q
    )

    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return

    if IS_3D:
        tiles_per_segment = cdiv_fn(seq_len, NUM_SEGMENTS_PER_SEQ * TILE_SIZE)
        if segm_idx * tiles_per_segment * TILE_SIZE >= seq_len:
            return
    else:
        tiles_per_segment = 0

    offs_m = tl.arange(0, BLOCK_M)
    offs_t = tl.arange(0, TILE_SIZE)
    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv

    query_mask_0 = tl.where(query_pos < cur_batch_query_len, 1, 0).to(tl.int1)
    query_mask_1 = tl.where(query_offset_1 < num_query_heads, 1, 0).to(tl.int1)

    # -----------------------------------------------------------------
    # Split-Q prologue: PACKING_FACTOR interleaved streams of Q.
    # INT4 uses 2 streams (even / odd); INT2 uses 4.  The packed KV
    # cache stores one byte per ``packed_offs``, which holds
    # PACKING_FACTOR values.
    # -----------------------------------------------------------------
    packed_offs = tl.arange(0, PACKED_HEAD_PADDED)
    offs_s0 = packed_offs * PACKING_FACTOR
    offs_s1 = packed_offs * PACKING_FACTOR + 1
    mask_s0 = tl.where(offs_s0 < HEAD_SIZE, 1, 0).to(tl.int1)
    mask_s1 = tl.where(offs_s1 < HEAD_SIZE, 1, 0).to(tl.int1)
    packed_dim_mask = tl.where(packed_offs < HEAD_SIZE // PACKING_FACTOR, 1, 0).to(
        tl.int1
    )
    q_base = (
        query_offset_0[:, None] * query_stride_0
        + query_offset_1[:, None] * query_stride_1
    )
    q_mask = query_mask_0[:, None] & query_mask_1[:, None]
    Q_s0 = tl.load(
        query_ptr + q_base + offs_s0[None, :],
        mask=mask_s0[None, :] & q_mask,
        other=0.0,
    ).to(tl.float32)
    Q_s1 = tl.load(
        query_ptr + q_base + offs_s1[None, :],
        mask=mask_s1[None, :] & q_mask,
        other=0.0,
    ).to(tl.float32)
    if PACKING_FACTOR == 4:
        offs_s2 = packed_offs * 4 + 2
        offs_s3 = packed_offs * 4 + 3
        mask_s2 = tl.where(offs_s2 < HEAD_SIZE, 1, 0).to(tl.int1)
        mask_s3 = tl.where(offs_s3 < HEAD_SIZE, 1, 0).to(tl.int1)
        Q_s2 = tl.load(
            query_ptr + q_base + offs_s2[None, :],
            mask=mask_s2[None, :] & q_mask,
            other=0.0,
        ).to(tl.float32)
        Q_s3 = tl.load(
            query_ptr + q_base + offs_s3[None, :],
            mask=mask_s3[None, :] & q_mask,
            other=0.0,
        ).to(tl.float32)

    # INT4 asymmetric correction needs sum(Q) per row.
    if PACKING_FACTOR == 2:
        Q_sum = tl.sum(Q_s0, axis=1) + tl.sum(Q_s1, axis=1)

    block_table_offset = seq_idx * block_table_stride

    # -----------------------------------------------------------------
    # Online-softmax state + optional feature loads.
    # -----------------------------------------------------------------
    M = init_softmax_M(
        sink_ptr, query_offset_1, query_mask_1, segm_idx, BLOCK_M, USE_SINKS, IS_3D
    )
    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc_s0 = tl.zeros([BLOCK_M, PACKED_HEAD_PADDED], dtype=tl.float32)
    acc_s1 = tl.zeros([BLOCK_M, PACKED_HEAD_PADDED], dtype=tl.float32)
    if PACKING_FACTOR == 4:
        acc_s2 = tl.zeros([BLOCK_M, PACKED_HEAD_PADDED], dtype=tl.float32)
        acc_s3 = tl.zeros([BLOCK_M, PACKED_HEAD_PADDED], dtype=tl.float32)

    context_len = seq_len - cur_batch_query_len

    if USE_ALIBI_SLOPES:
        alibi_slope = tl.load(
            alibi_slopes_ptr + query_offset_1, mask=query_mask_1, other=0.0
        )

    if USE_QQ_BIAS:
        qq_bias_row_ptrs = qq_bias_ptr + query_pos[:, None] * qq_bias_stride_0

    loop_lo, loop_hi, max_seq_prefix_len = compute_tile_loop_bounds(
        context_len,
        seq_len,
        cur_batch_query_len,
        q_block_local_idx,
        segm_idx,
        tiles_per_segment,
        TILE_SIZE,
        BLOCK_M,
        BLOCK_Q,
        num_queries_per_kv,
        SLIDING_WINDOW,
        USE_MM_PREFIX,
        IS_3D,
    )

    # -----------------------------------------------------------------
    # Tile loop.  Per-tile: load packed KV + scales, dequantize into
    # PACKING_FACTOR streams, compute the split dot, run the shared
    # softmax step, accumulate per stream.
    # -----------------------------------------------------------------
    for j in range(loop_lo, loop_hi):
        seq_offset = j * TILE_SIZE + offs_t
        tile_mask = seq_offset < max_seq_prefix_len

        physical_block_idx = tl.load(
            block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE
        ).to(tl.int64)

        slot_in_blk = seq_offset % BLOCK_SIZE
        k_off = (
            physical_block_idx[None, :] * stride_k_cache_0
            + kv_head_idx * stride_k_cache_2
            + packed_offs[:, None] * stride_k_cache_3
            + slot_in_blk[None, :] * stride_k_cache_1
        )
        K_packed = tl.load(
            key_cache_ptr + k_off,
            mask=packed_dim_mask[:, None] & tile_mask[None, :],
            other=0,
        )
        v_off = (
            physical_block_idx[:, None] * stride_v_cache_0
            + kv_head_idx * stride_v_cache_2
            + packed_offs[None, :] * stride_v_cache_3
            + slot_in_blk[:, None] * stride_v_cache_1
        )
        V_packed = tl.load(
            value_cache_ptr + v_off,
            mask=packed_dim_mask[None, :] & tile_mask[:, None],
            other=0,
        )
        # Dequantize KV.  INT4 unpacks nibbles as plain uint [0..15];
        # the zero-point is applied on the score side.  INT2 unpacks
        # quartet indices and looks up Lloyd-Max centroids (N(0, 1)).
        if PACKING_FACTOR == 2:
            K_s0_u, K_s1_u = unpack_int4_nibbles(K_packed)
            K_s0 = K_s0_u.to(tl.float32)
            K_s1 = K_s1_u.to(tl.float32)
            V_s0_u, V_s1_u = unpack_int4_nibbles(V_packed)
            V_s0 = V_s0_u.to(tl.float32)
            V_s1 = V_s1_u.to(tl.float32)
        else:
            kc0_u, kc1_u, kc2_u, kc3_u = unpack_int2_quartet(K_packed)
            K_s0 = _lloyd_max_dequant_4(kc0_u).to(tl.float32)
            K_s1 = _lloyd_max_dequant_4(kc1_u).to(tl.float32)
            K_s2 = _lloyd_max_dequant_4(kc2_u).to(tl.float32)
            K_s3 = _lloyd_max_dequant_4(kc3_u).to(tl.float32)
            vc0_u, vc1_u, vc2_u, vc3_u = unpack_int2_quartet(V_packed)
            V_s0 = _lloyd_max_dequant_4(vc0_u).to(tl.float32)
            V_s1 = _lloyd_max_dequant_4(vc1_u).to(tl.float32)
            V_s2 = _lloyd_max_dequant_4(vc2_u).to(tl.float32)
            V_s3 = _lloyd_max_dequant_4(vc3_u).to(tl.float32)

        ks_idx = (
            physical_block_idx * stride_ks_blk
            + slot_in_blk * stride_ks_slot
            + kv_head_idx * stride_ks_head
        )
        ks_raw = tl.load(k_scale_cache_ptr + ks_idx, mask=tile_mask, other=0)
        vs_idx = (
            physical_block_idx * stride_vs_blk
            + slot_in_blk * stride_vs_slot
            + kv_head_idx * stride_vs_head
        )
        vs_raw = tl.load(v_scale_cache_ptr + vs_idx, mask=tile_mask, other=0)

        # INT4 steganographs the 4-bit zero-point in the low 4 bits of
        # the float32 scale's mantissa.  INT2 stores a plain scale.
        if PACKING_FACTOR == 2:
            ks_bits = ks_raw.to(tl.int32, bitcast=True)
            k_zp = (ks_bits & 0xF).to(tl.float32)
            k_token_head_scales = (ks_bits & -16).to(tl.float32, bitcast=True)
            vs_bits = vs_raw.to(tl.int32, bitcast=True)
            v_zp = (vs_bits & 0xF).to(tl.float32)
            v_token_head_scales = (vs_bits & -16).to(tl.float32, bitcast=True)
        else:
            k_token_head_scales = ks_raw
            v_token_head_scales = vs_raw

        query_abs_pos = context_len + query_pos[:, None]
        seq_mask = compute_kv_seq_mask(
            query_abs_pos,
            seq_offset,
            seq_idx,
            mm_prefix_range_ptr,
            SLIDING_WINDOW,
            USE_MM_PREFIX,
            MAX_MM_RANGES,
        )

        # Score: split-dot across PACKING_FACTOR streams; fused
        # softmax_scale * per-(token, head) k_scale in one mul.  INT4
        # subtracts the ``zp * sum(Q)`` correction term; INT2 doesn't.
        S = tl.zeros(shape=(BLOCK_M, TILE_SIZE), dtype=tl.float32)
        if PACKING_FACTOR == 2:
            raw_dot = tl.dot(Q_s0, K_s0) + tl.dot(Q_s1, K_s1)
            S += (raw_dot - Q_sum[:, None] * k_zp[None, :]) * (
                scale * k_token_head_scales[None, :]
            )
        else:
            raw_dot = (
                tl.dot(Q_s0, K_s0)
                + tl.dot(Q_s1, K_s1)
                + tl.dot(Q_s2, K_s2)
                + tl.dot(Q_s3, K_s3)
            )
            S += raw_dot * (scale * k_token_head_scales[None, :])

        if USE_SOFTCAP:
            S = apply_softcap(S, softcap)

        S = tl.where(
            query_mask_1[:, None] & query_mask_0[:, None] & seq_mask, S, float("-inf")
        )

        if USE_ALIBI_SLOPES:
            S = apply_alibi_to_score(
                S, alibi_slope, seq_offset, context_len, query_pos, USE_ALIBI_SQRT
            )

        if USE_QQ_BIAS:
            S += load_qq_bias_tile(
                qq_bias_row_ptrs, seq_offset, context_len, qq_bias_stride_0
            )

        M, L, P, alpha = softmax_step(S, M, L)
        acc_s0 = acc_s0 * alpha[:, None]
        acc_s1 = acc_s1 * alpha[:, None]
        if PACKING_FACTOR == 4:
            acc_s2 = acc_s2 * alpha[:, None]
            acc_s3 = acc_s3 * alpha[:, None]

        if SLIDING_WINDOW:
            qpos_lo = q_block_local_idx * BLOCK_Q
            sw_mask = (context_len + qpos_lo - seq_offset) < SLIDING_WINDOW
            V_s0 = tl.where(sw_mask[:, None], V_s0, 0.0)
            V_s1 = tl.where(sw_mask[:, None], V_s1, 0.0)
            if PACKING_FACTOR == 4:
                V_s2 = tl.where(sw_mask[:, None], V_s2, 0.0)
                V_s3 = tl.where(sw_mask[:, None], V_s3, 0.0)

        # Fuse v per-(token, head) scale into P.  INT4 also subtracts
        # the v-zero-point contribution from each stream once.
        P_v = (P * v_token_head_scales[None, :]).to(tl.float32)
        if PACKING_FACTOR == 2:
            Pv_zp_sum = tl.sum(P_v * v_zp[None, :], axis=1)
            acc_s0 += tl.dot(P_v, V_s0) - Pv_zp_sum[:, None]
            acc_s1 += tl.dot(P_v, V_s1) - Pv_zp_sum[:, None]
        else:
            acc_s0 += tl.dot(P_v, V_s0)
            acc_s1 += tl.dot(P_v, V_s1)
            acc_s2 += tl.dot(P_v, V_s2)
            acc_s3 += tl.dot(P_v, V_s3)

    # -----------------------------------------------------------------
    # Epilogue.  2D writes the final output with optional FP8 clamp;
    # 3D writes the per-segment partials (output / max / expsum) for
    # ``reduce_segments`` to finalize.  Each stream writes its own
    # stripe in the output layout.
    # -----------------------------------------------------------------
    out_mask = query_mask_0[:, None] & query_mask_1[:, None]
    if IS_3D:
        segm_base = (
            query_offset_0[:, None].to(tl.int64)
            * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
            + query_offset_1[:, None] * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
            + segm_idx * HEAD_SIZE_PADDED
        )
        tl.store(
            segm_output_ptr + segm_base + offs_s0[None, :],
            acc_s0,
            mask=mask_s0[None, :] & out_mask,
        )
        tl.store(
            segm_output_ptr + segm_base + offs_s1[None, :],
            acc_s1,
            mask=mask_s1[None, :] & out_mask,
        )
        if PACKING_FACTOR == 4:
            tl.store(
                segm_output_ptr + segm_base + offs_s2[None, :],
                acc_s2,
                mask=mask_s2[None, :] & out_mask,
            )
            tl.store(
                segm_output_ptr + segm_base + offs_s3[None, :],
                acc_s3,
                mask=mask_s3[None, :] & out_mask,
            )
        store_segm_reduce_scalars(
            segm_max_ptr,
            segm_expsum_ptr,
            query_offset_0,
            query_offset_1,
            segm_idx,
            M,
            L,
            query_mask_0,
            query_mask_1,
            num_query_heads,
            NUM_SEGMENTS_PER_SEQ,
        )
    else:
        acc_s0 = acc_s0 / L[:, None]
        acc_s1 = acc_s1 / L[:, None]
        if PACKING_FACTOR == 4:
            acc_s2 = acc_s2 / L[:, None]
            acc_s3 = acc_s3 / L[:, None]
        if USE_FP8:
            out_s = tl.load(out_scale)
            acc_s0 = tl.clamp(acc_s0 * out_s, FP8_MIN, FP8_MAX)
            acc_s1 = tl.clamp(acc_s1 * out_s, FP8_MIN, FP8_MAX)
            if PACKING_FACTOR == 4:
                acc_s2 = tl.clamp(acc_s2 * out_s, FP8_MIN, FP8_MAX)
                acc_s3 = tl.clamp(acc_s3 * out_s, FP8_MIN, FP8_MAX)
        out_base = (
            query_offset_0[:, None] * output_stride_0
            + query_offset_1[:, None] * output_stride_1
        )
        tl.store(
            output_ptr + out_base + offs_s0[None, :],
            acc_s0,
            mask=mask_s0[None, :] & out_mask,
        )
        tl.store(
            output_ptr + out_base + offs_s1[None, :],
            acc_s1,
            mask=mask_s1[None, :] & out_mask,
        )
        if PACKING_FACTOR == 4:
            tl.store(
                output_ptr + out_base + offs_s2[None, :],
                acc_s2,
                mask=mask_s2[None, :] & out_mask,
            )
            tl.store(
                output_ptr + out_base + offs_s3[None, :],
                acc_s3,
                mask=mask_s3[None, :] & out_mask,
            )


# ===========================================================================
# Python launchers + Factory classes.
# ===========================================================================


def _launch_packed_attn(
    *,
    q,
    k_cache,
    v_cache,
    out,
    cu_seqlens_q,
    max_seqlen_q,
    seqused_k,
    softmax_scale,
    window_size,
    block_table,
    softcap,
    sinks,
    alibi_slopes,
    use_alibi_sqrt,
    qq_bias,
    output_scale,
    mm_prefix_range,
    k_scale_cache,
    v_scale_cache,
    seq_threshold_3D,
    num_par_softmax_segments,
    softmax_segm_output,
    softmax_segm_max,
    softmax_segm_expsum,
    packing_factor: int,
):
    """Launch ``_attn_packed`` for one of the sub-byte modes.

    Handles 2D-vs-3D dispatch, placeholder pointers for the unused side
    of that split, and the trailing ``reduce_segments`` pass.  Writes
    into ``out`` (directly for 2D; via the segm buffers for 3D).
    """
    import vllm.envs as envs
    from vllm.v1.attention.ops.triton_unified_attention import _get_tile_size

    is_batch_invariant = envs.VLLM_BATCH_INVARIANT

    use_mm_prefix = False
    max_mm_ranges = 0
    if mm_prefix_range is not None:
        assert mm_prefix_range.ndim == 3, (
            f"Unsupported mm_prefix_range shape: {mm_prefix_range.shape}"
        )
        use_mm_prefix = True
        max_mm_ranges = mm_prefix_range.shape[1]

    block_size = v_cache.shape[1]
    num_seqs = len(seqused_k)
    num_query_heads = q.shape[1]
    num_kv_heads = k_cache.shape[2]
    num_queries_per_kv = num_query_heads // num_kv_heads
    head_size = q.shape[2]

    BLOCK_M = (
        16 if num_queries_per_kv <= 16 else triton.next_power_of_2(num_queries_per_kv)
    )
    BLOCK_Q = BLOCK_M // num_queries_per_kv
    total_num_q_blocks = q.shape[0] // BLOCK_Q + num_seqs
    sliding_window_val = 1 + window_size[0] if window_size[0] >= 0 else 0
    TILE_SIZE_PREFILL = _get_tile_size(
        head_size, sliding_window_val, q.element_size(), is_prefill=True
    )
    TILE_SIZE_DECODE = _get_tile_size(
        head_size, sliding_window_val, q.element_size(), is_prefill=False
    )

    use_3d = not (
        seq_threshold_3D is None
        or num_par_softmax_segments is None
        or softmax_segm_output is None
        or softmax_segm_max is None
        or softmax_segm_expsum is None
        or max_seqlen_q > 1
        or num_seqs > seq_threshold_3D
        or is_batch_invariant
    )

    # 3D never reads ``output_ptr`` and 2D never reads the segm tensors,
    # but Triton needs a non-null pointer everywhere; reuse ``out`` as
    # the placeholder for the unused side.
    segm_output_ptr = softmax_segm_output if use_3d else out
    segm_max_ptr = softmax_segm_max if use_3d else out
    segm_expsum_ptr = softmax_segm_expsum if use_3d else out
    num_segments = num_par_softmax_segments if use_3d else 1

    grid: tuple[Any, ...]
    if use_3d:
        grid = (total_num_q_blocks, num_kv_heads, num_par_softmax_segments)
        tile_size = TILE_SIZE_DECODE
    else:
        grid = (total_num_q_blocks, num_kv_heads)
        tile_size = TILE_SIZE_PREFILL

    _attn_packed[grid](
        output_ptr=out,
        segm_output_ptr=segm_output_ptr,
        segm_max_ptr=segm_max_ptr,
        segm_expsum_ptr=segm_expsum_ptr,
        query_ptr=q,
        key_cache_ptr=k_cache,
        value_cache_ptr=v_cache,
        sink_ptr=sinks,
        block_tables_ptr=block_table,
        seq_lens_ptr=seqused_k,
        alibi_slopes_ptr=alibi_slopes,
        qq_bias_ptr=qq_bias,
        scale=softmax_scale,
        out_scale=1 / output_scale if output_scale is not None else 1.0,
        softcap=softcap,
        k_scale_cache_ptr=k_scale_cache,
        v_scale_cache_ptr=v_scale_cache,
        num_query_heads=num_query_heads,
        num_queries_per_kv=num_queries_per_kv,
        block_table_stride=block_table.stride(0),
        query_stride_0=q.stride(0),
        query_stride_1=q.stride(1),
        output_stride_0=out.stride(0),
        output_stride_1=out.stride(1),
        qq_bias_stride_0=qq_bias.stride(0) if qq_bias is not None else 0,
        BLOCK_SIZE=block_size,
        TILE_SIZE=tile_size,
        HEAD_SIZE=head_size,
        HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
        PACKED_HEAD_PADDED=triton.next_power_of_2(head_size) // packing_factor,
        USE_ALIBI_SLOPES=alibi_slopes is not None,
        USE_ALIBI_SQRT=use_alibi_sqrt,
        USE_QQ_BIAS=qq_bias is not None,
        USE_SOFTCAP=(softcap > 0),
        USE_SINKS=(sinks is not None),
        SLIDING_WINDOW=(1 + window_size[0]),
        USE_MM_PREFIX=use_mm_prefix,
        MAX_MM_RANGES=max_mm_ranges,
        mm_prefix_range_ptr=mm_prefix_range,
        stride_k_cache_0=k_cache.stride(0),
        stride_k_cache_1=k_cache.stride(1),
        stride_k_cache_2=k_cache.stride(2),
        stride_k_cache_3=k_cache.stride(3),
        stride_v_cache_0=v_cache.stride(0),
        stride_v_cache_1=v_cache.stride(1),
        stride_v_cache_2=v_cache.stride(2),
        stride_v_cache_3=v_cache.stride(3),
        stride_ks_blk=k_scale_cache.stride(0),
        stride_ks_slot=k_scale_cache.stride(1),
        stride_ks_head=k_scale_cache.stride(2),
        stride_vs_blk=v_scale_cache.stride(0),
        stride_vs_slot=v_scale_cache.stride(1),
        stride_vs_head=v_scale_cache.stride(2),
        query_start_len_ptr=cu_seqlens_q,
        BLOCK_Q=BLOCK_Q,
        num_seqs=num_seqs,
        BLOCK_M=BLOCK_M,
        NUM_SEGMENTS_PER_SEQ=num_segments,
        USE_FP8=output_scale is not None,
        IS_3D=use_3d,
        PACKING_FACTOR=packing_factor,
    )

    if use_3d:
        reduce_segments[(q.shape[0], num_query_heads)](
            output_ptr=out,
            segm_output_ptr=softmax_segm_output,
            segm_max_ptr=softmax_segm_max,
            segm_expsum_ptr=softmax_segm_expsum,
            seq_lens_ptr=seqused_k,
            num_seqs=num_seqs,
            num_query_heads=num_query_heads,
            out_scale_inv=1 / output_scale if output_scale is not None else 1.0,
            output_stride_0=out.stride(0),
            output_stride_1=out.stride(1),
            block_table_stride=block_table.stride(0),
            TILE_SIZE=TILE_SIZE_DECODE,
            HEAD_SIZE=head_size,
            HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
            query_start_len_ptr=cu_seqlens_q,
            BLOCK_Q=BLOCK_Q,
            NUM_SEGMENTS_PER_SEQ=num_par_softmax_segments,
            USE_FP8=output_scale is not None,
        )


# ===========================================================================
# Flash-attention-shape prefill kernel for sub-byte packed modes.
#
# ``_attn_packed`` above is grid-tuned for decode (small BLOCK_M, split-KV
# via IS_3D when num_seqs fits).  For long chunked prefill with cached
# context, its BLOCK_M defaults to 16 for MHA / small-group GQA, which
# wastes the K-reuse opportunity across BLOCK_M queries of the same
# request.  ``_attn_packed_prefill`` uses the flash-attention layout —
# grid ``(num_reqs, Hq, cdiv(max_q_len, BLOCK_M))`` with BLOCK_M × BLOCK_N
# tiles — so one K tile feeds 32 or 64 queries instead of 16.  Scope is
# intentionally narrow (no alibi / sinks / softcap / sliding window /
# mm_prefix / qq_bias); the factory routes those cases back to
# ``_launch_packed_attn``.  Same split-dot across PACKING_FACTOR streams
# as ``_attn_packed``; INT4 applies the asymmetric zero-point correction
# (packed in the low 4 mantissa bits of the fp32 scale), INT2 does the
# Lloyd-Max centroid lookup.
# ===========================================================================


@triton.jit
def _attn_packed_prefill(
    Q_ptr,
    K_cache_ptr,
    V_cache_ptr,
    K_scale_ptr,
    V_scale_ptr,
    Block_table_ptr,
    Query_start_loc_ptr,
    Seq_lens_ptr,
    Out_ptr,
    stride_q_tok: tl.int64,
    stride_q_h: tl.int64,
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
    stride_bt_r: tl.int64,
    stride_o_tok: tl.int64,
    stride_o_h: tl.int64,
    SM_SCALE: tl.constexpr,
    KV_GROUP: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    PACKED_HEAD_PADDED: tl.constexpr,
    # 2 → INT4 nibbles (asymmetric + zp); 4 → INT2 quartets (Lloyd-Max).
    PACKING_FACTOR: tl.constexpr,
):
    req_id = tl.program_id(0)
    head_id = tl.program_id(1)
    start_m = tl.program_id(2)

    kv_head = head_id // KV_GROUP

    q_start = tl.load(Query_start_loc_ptr + req_id)
    q_end = tl.load(Query_start_loc_ptr + req_id + 1)
    q_len = q_end - q_start

    block_start = start_m * BLOCK_M
    if block_start >= q_len:
        return

    seq_len = tl.load(Seq_lens_ptr + req_id)
    cached_len = seq_len - q_len

    offs_m = block_start + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    packed_offs = tl.arange(0, PACKED_HEAD_PADDED)
    m_mask = offs_m < q_len

    # Split-Q prologue: PACKING_FACTOR interleaved streams.  K is stored
    # as 1 byte per packed_offs holding PACKING_FACTOR values at indices
    # (packed_offs*PF + 0..PF-1).  We load matching Q stripes so the
    # split dot ``sum_i Q_si·K_si`` reconstructs the full Q·Kᵀ.
    offs_s0 = packed_offs * PACKING_FACTOR
    offs_s1 = packed_offs * PACKING_FACTOR + 1
    mask_s0 = offs_s0 < HEAD_SIZE
    mask_s1 = offs_s1 < HEAD_SIZE
    packed_dim_mask = packed_offs < HEAD_SIZE // PACKING_FACTOR

    q_row_off = (q_start + offs_m)[:, None] * stride_q_tok + head_id * stride_q_h
    q_mask = m_mask[:, None]

    Q_s0 = tl.load(
        Q_ptr + q_row_off + offs_s0[None, :],
        mask=mask_s0[None, :] & q_mask,
        other=0.0,
    ).to(tl.float32)
    Q_s1 = tl.load(
        Q_ptr + q_row_off + offs_s1[None, :],
        mask=mask_s1[None, :] & q_mask,
        other=0.0,
    ).to(tl.float32)
    if PACKING_FACTOR == 4:
        offs_s2 = packed_offs * 4 + 2
        offs_s3 = packed_offs * 4 + 3
        mask_s2 = offs_s2 < HEAD_SIZE
        mask_s3 = offs_s3 < HEAD_SIZE
        Q_s2 = tl.load(
            Q_ptr + q_row_off + offs_s2[None, :],
            mask=mask_s2[None, :] & q_mask,
            other=0.0,
        ).to(tl.float32)
        Q_s3 = tl.load(
            Q_ptr + q_row_off + offs_s3[None, :],
            mask=mask_s3[None, :] & q_mask,
            other=0.0,
        ).to(tl.float32)

    # INT4 asymmetric correction needs sum(Q) per row.
    if PACKING_FACTOR == 2:
        Q_sum = tl.sum(Q_s0, axis=1) + tl.sum(Q_s1, axis=1)

    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc_s0 = tl.zeros([BLOCK_M, PACKED_HEAD_PADDED], dtype=tl.float32)
    acc_s1 = tl.zeros([BLOCK_M, PACKED_HEAD_PADDED], dtype=tl.float32)
    if PACKING_FACTOR == 4:
        acc_s2 = tl.zeros([BLOCK_M, PACKED_HEAD_PADDED], dtype=tl.float32)
        acc_s3 = tl.zeros([BLOCK_M, PACKED_HEAD_PADDED], dtype=tl.float32)

    q_pos = cached_len + offs_m
    end_n = tl.minimum(seq_len, cached_len + block_start + BLOCK_M)

    bt_base = req_id * stride_bt_r

    for start_n in range(0, end_n, BLOCK_N):
        k_pos = start_n + offs_n
        valid_k = k_pos < seq_len

        page_idx = k_pos // BLOCK_SIZE
        page_off = k_pos % BLOCK_SIZE
        block_nums = tl.load(
            Block_table_ptr + bt_base + page_idx, mask=valid_k, other=0
        )

        # K : [PACKED_HEAD_PADDED, BLOCK_N] — packed bytes, 1 byte holds
        # PACKING_FACTOR values.  Stride within a head is 1 byte.
        k_addrs = (
            block_nums[None, :] * stride_kc_blk
            + page_off[None, :] * stride_kc_slot
            + kv_head * stride_kc_head
            + packed_offs[:, None]
        )
        K_packed = tl.load(
            K_cache_ptr + k_addrs,
            mask=valid_k[None, :] & packed_dim_mask[:, None],
            other=0,
        )

        # V : [BLOCK_N, PACKED_HEAD_PADDED]
        v_addrs = (
            block_nums[:, None] * stride_vc_blk
            + page_off[:, None] * stride_vc_slot
            + kv_head * stride_vc_head
            + packed_offs[None, :]
        )
        V_packed = tl.load(
            V_cache_ptr + v_addrs,
            mask=valid_k[:, None] & packed_dim_mask[None, :],
            other=0,
        )

        # Dequantize KV.  Layouts mirror ``_attn_packed``:
        # INT4 → nibble pair, zp on the score side.
        # INT2 → quartet indices, Lloyd-Max centroids.
        if PACKING_FACTOR == 2:
            K_s0_u, K_s1_u = unpack_int4_nibbles(K_packed)
            K_s0 = K_s0_u.to(tl.float32)
            K_s1 = K_s1_u.to(tl.float32)
            V_s0_u, V_s1_u = unpack_int4_nibbles(V_packed)
            V_s0 = V_s0_u.to(tl.float32)
            V_s1 = V_s1_u.to(tl.float32)
        else:
            kc0_u, kc1_u, kc2_u, kc3_u = unpack_int2_quartet(K_packed)
            K_s0 = _lloyd_max_dequant_4(kc0_u).to(tl.float32)
            K_s1 = _lloyd_max_dequant_4(kc1_u).to(tl.float32)
            K_s2 = _lloyd_max_dequant_4(kc2_u).to(tl.float32)
            K_s3 = _lloyd_max_dequant_4(kc3_u).to(tl.float32)
            vc0_u, vc1_u, vc2_u, vc3_u = unpack_int2_quartet(V_packed)
            V_s0 = _lloyd_max_dequant_4(vc0_u).to(tl.float32)
            V_s1 = _lloyd_max_dequant_4(vc1_u).to(tl.float32)
            V_s2 = _lloyd_max_dequant_4(vc2_u).to(tl.float32)
            V_s3 = _lloyd_max_dequant_4(vc3_u).to(tl.float32)

        ks_idx = (
            block_nums * stride_ks_blk
            + page_off * stride_ks_slot
            + kv_head * stride_ks_head
        )
        ks_raw = tl.load(K_scale_ptr + ks_idx, mask=valid_k, other=0.0)
        vs_idx = (
            block_nums * stride_vs_blk
            + page_off * stride_vs_slot
            + kv_head * stride_vs_head
        )
        vs_raw = tl.load(V_scale_ptr + vs_idx, mask=valid_k, other=0.0)

        # INT4 steganographs the 4-bit zero-point in the low 4 bits of
        # the float32 scale's mantissa.  INT2 stores a plain scale.
        if PACKING_FACTOR == 2:
            ks_bits = ks_raw.to(tl.int32, bitcast=True)
            k_zp = (ks_bits & 0xF).to(tl.float32)
            k_tok_head_scales = (ks_bits & -16).to(tl.float32, bitcast=True)
            vs_bits = vs_raw.to(tl.int32, bitcast=True)
            v_zp = (vs_bits & 0xF).to(tl.float32)
            v_tok_head_scales = (vs_bits & -16).to(tl.float32, bitcast=True)
        else:
            k_tok_head_scales = ks_raw
            v_tok_head_scales = vs_raw

        # S = Q @ Kᵀ (split across PACKING_FACTOR streams).  Fused
        # softmax_scale * per-(token, head) k_scale in one mul.
        if PACKING_FACTOR == 2:
            raw_dot = tl.dot(Q_s0, K_s0) + tl.dot(Q_s1, K_s1)
            qk = (raw_dot - Q_sum[:, None] * k_zp[None, :]) * (
                SM_SCALE * k_tok_head_scales[None, :]
            )
        else:
            raw_dot = (
                tl.dot(Q_s0, K_s0)
                + tl.dot(Q_s1, K_s1)
                + tl.dot(Q_s2, K_s2)
                + tl.dot(Q_s3, K_s3)
            )
            qk = raw_dot * (SM_SCALE * k_tok_head_scales[None, :])

        # Causal mask: absolute positions within the sequence.
        causal = k_pos[None, :] <= q_pos[:, None]
        full_mask = causal & valid_k[None, :]
        qk = tl.where(full_mask, qk, -float("inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk = qk - m_ij[:, None]
        p = tl.exp(qk)
        l_ij = tl.sum(p, 1)

        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc_s0 = acc_s0 * alpha[:, None]
        acc_s1 = acc_s1 * alpha[:, None]
        if PACKING_FACTOR == 4:
            acc_s2 = acc_s2 * alpha[:, None]
            acc_s3 = acc_s3 * alpha[:, None]

        # P @ V (split across streams).  INT4 also subtracts the
        # v-zero-point contribution from each stream once.
        P_v = p * v_tok_head_scales[None, :]
        if PACKING_FACTOR == 2:
            Pv_zp_sum = tl.sum(P_v * v_zp[None, :], axis=1)
            acc_s0 += tl.dot(P_v, V_s0) - Pv_zp_sum[:, None]
            acc_s1 += tl.dot(P_v, V_s1) - Pv_zp_sum[:, None]
        else:
            acc_s0 += tl.dot(P_v, V_s0)
            acc_s1 += tl.dot(P_v, V_s1)
            acc_s2 += tl.dot(P_v, V_s2)
            acc_s3 += tl.dot(P_v, V_s3)

        m_i = m_ij

    safe_l = tl.where(l_i > 0.0, l_i, 1.0)
    acc_s0 = acc_s0 / safe_l[:, None]
    acc_s1 = acc_s1 / safe_l[:, None]
    if PACKING_FACTOR == 4:
        acc_s2 = acc_s2 / safe_l[:, None]
        acc_s3 = acc_s3 / safe_l[:, None]

    out_row_off = (q_start + offs_m)[:, None] * stride_o_tok + head_id * stride_o_h
    out_mask = m_mask[:, None]
    tl.store(
        Out_ptr + out_row_off + offs_s0[None, :],
        acc_s0,
        mask=mask_s0[None, :] & out_mask,
    )
    tl.store(
        Out_ptr + out_row_off + offs_s1[None, :],
        acc_s1,
        mask=mask_s1[None, :] & out_mask,
    )
    if PACKING_FACTOR == 4:
        tl.store(
            Out_ptr + out_row_off + offs_s2[None, :],
            acc_s2,
            mask=mask_s2[None, :] & out_mask,
        )
        tl.store(
            Out_ptr + out_row_off + offs_s3[None, :],
            acc_s3,
            mask=mask_s3[None, :] & out_mask,
        )


def launch_packed_prefill(
    *,
    q: torch.Tensor,
    out: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale_cache: torch.Tensor,
    v_scale_cache: torch.Tensor,
    block_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    softmax_scale: float,
    num_reqs: int,
    max_query_len: int,
    packing_factor: int,
) -> None:
    """Launch ``_attn_packed_prefill`` for INT4 or INT2.

    Scope: causal attention only, no alibi / sinks / softcap / sliding
    window / mm_prefix / qq_bias.  Callers that need those features
    should route back to ``_launch_packed_attn``.
    """
    total_q, Hq, head_size = q.shape
    Hkv = k_cache.shape[2]
    kv_group = Hq // Hkv
    block_size = k_cache.shape[1]

    if total_q == 0 or num_reqs == 0:
        return

    # BLOCK_M × PACKING_FACTOR streams dominates register pressure;
    # back off on large heads.  Values match the tuning of the
    # byte-aligned prefill kernel once you account for the extra streams.
    if head_size <= 64:
        BLOCK_M = 64
    else:
        BLOCK_M = 32
    BLOCK_N = 64
    packed_head_padded = triton.next_power_of_2(head_size) // packing_factor

    grid = (num_reqs, Hq, triton.cdiv(max_query_len, BLOCK_M))

    _attn_packed_prefill[grid](
        q,
        k_cache,
        v_cache,
        k_scale_cache,
        v_scale_cache,
        block_table,
        query_start_loc,
        seq_lens,
        out,
        q.stride(0),
        q.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        k_scale_cache.stride(0),
        k_scale_cache.stride(1),
        k_scale_cache.stride(2),
        v_scale_cache.stride(0),
        v_scale_cache.stride(1),
        v_scale_cache.stride(2),
        block_table.stride(0),
        out.stride(0),
        out.stride(1),
        SM_SCALE=softmax_scale,
        KV_GROUP=kv_group,
        BLOCK_SIZE=block_size,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        HEAD_SIZE=head_size,
        PACKED_HEAD_PADDED=packed_head_padded,
        PACKING_FACTOR=packing_factor,
        num_warps=4 if head_size <= 64 else 8,
        num_stages=2,
    )


def _run_reshape_kernel(
    kernel,
    *,
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    k_scale_cache: torch.Tensor,
    v_scale_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    packing_factor: int,
) -> None:
    """Launch a packed reshape kernel (INT4 or INT2)."""
    num_tokens, num_kv_heads, head_size = key.shape
    head_size_v = value.shape[2]
    assert head_size % packing_factor == 0 and head_size_v % packing_factor == 0
    packed_padded = triton.next_power_of_2(
        max(head_size, head_size_v) // packing_factor
    )
    if current_platform.is_rocm() or current_platform.is_xpu():
        num_warps = 4
    else:
        num_warps = min(16, max(1, packed_padded // 32))

    kernel[(num_tokens, num_kv_heads)](
        key_ptr=key,
        value_ptr=value,
        key_cache_ptr=key_cache,
        value_cache_ptr=value_cache,
        k_scale_cache_ptr=k_scale_cache,
        v_scale_cache_ptr=v_scale_cache,
        slot_mapping_ptr=slot_mapping,
        stride_key_tok=key.stride(0),
        stride_key_head=key.stride(1),
        stride_val_tok=value.stride(0),
        stride_val_head=value.stride(1),
        stride_kc_blk=key_cache.stride(0),
        stride_kc_slot=key_cache.stride(1),
        stride_kc_head=key_cache.stride(2),
        stride_vc_blk=value_cache.stride(0),
        stride_vc_slot=value_cache.stride(1),
        stride_vc_head=value_cache.stride(2),
        stride_ks_blk=k_scale_cache.stride(0),
        stride_ks_slot=k_scale_cache.stride(1),
        stride_ks_head=k_scale_cache.stride(2),
        stride_vs_blk=v_scale_cache.stride(0),
        stride_vs_slot=v_scale_cache.stride(1),
        stride_vs_head=v_scale_cache.stride(2),
        block_size=key_cache.shape[1],
        head_size=head_size,
        head_size_v=head_size_v,
        PACKED_HEAD_PADDED=packed_padded,
        num_warps=num_warps,
    )


class _PackedFactory(QuantKVPlugin):
    """Shared base for sub-byte packed per-token-head plugins.

    Subclasses declare the mode-specific pieces as class attributes /
    static methods; the ``reshape_and_cache`` / ``unified_attention``
    bodies are identical and live here.  No :class:`KVQuantMode` enum
    value is referenced — each subclass identifies itself via its
    :class:`QuantKVSpec` (``spec.name`` is the string the dispatcher
    uses end-to-end), so adding a new sub-byte mode requires only
    creating a subclass of this class.

    Mode-specific hooks (subclass must set / override)
    --------------------------------------------------
    ``spec``
        :class:`QuantKVSpec` with at least ``name`` and
        ``packing_factor`` populated.
        ``needs_per_token_head_scales`` must be ``True``.
    ``_reshape_kernel``
        The ``@triton.jit`` reshape kernel for this mode.
    ``_rotate_kv(x)``
        Pre-rotation applied to K / V before packing (RHT for INT4,
        full Hadamard for INT2).
    ``_rotate_q(q)``
        Pre-rotation applied to Q before attention.  Typically the same
        rotation as ``_rotate_kv`` so the dot product is preserved.
    ``_unrotate_out(out, head_size)``
        Inverse rotation on the kernel output, written back in-place.
    ``_transform_softmax_scale(scale, head_size)``
        Optional rescaling of ``softmax_scale`` before the kernel (INT4
        divides by ``head_size`` to absorb the RHT scale; INT2 is a
        no-op).
    """

    # Filled in by subclasses.
    _reshape_kernel: object

    @staticmethod
    def _rotate_kv(x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def _rotate_q(q: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def _unrotate_out(out: torch.Tensor, head_size: int) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def _transform_softmax_scale(scale: float, head_size: int) -> float:
        return scale

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
            f"{self.spec.name!r} requires k_scale_cache / v_scale_cache"
        )
        key = self._rotate_kv(key.float()).to(key.dtype)
        value = self._rotate_kv(value.float()).to(value.dtype)
        _run_reshape_kernel(
            self._reshape_kernel,
            key=key,
            value=value,
            key_cache=key_cache,
            value_cache=value_cache,
            k_scale_cache=k_scale_cache,
            v_scale_cache=v_scale_cache,
            slot_mapping=slot_mapping,
            packing_factor=self.spec.packing_factor,
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
        q = self._rotate_q(q.float()).to(q_orig_dtype)
        head_size = q.shape[2]
        softmax_scale = self._transform_softmax_scale(softmax_scale, head_size)

        _launch_packed_attn(
            q=q,
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
            packing_factor=self.spec.packing_factor,
        )

        out_f = self._unrotate_out(out, head_size)
        out.copy_(out_f.to(q_orig_dtype))

    def unified_attention_prefill(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        out: torch.Tensor,
        *,
        query_start_loc: torch.Tensor,
        seq_lens: torch.Tensor,
        block_table: torch.Tensor,
        softmax_scale: float,
        num_reqs: int,
        max_query_len: int,
        k_scale_cache: torch.Tensor,
        v_scale_cache: torch.Tensor,
    ) -> None:
        """Flash-attention-shape prefill for this packed mode.

        Scope matches :func:`launch_packed_prefill`: causal attention
        only, no alibi / sinks / softcap / sliding window / mm_prefix /
        qq_bias.  Caller gates on those being unset; when they're set,
        route to :meth:`unified_attention` which forwards to
        ``_launch_packed_attn`` (handles all of the above).
        """
        q_orig_dtype = q.dtype
        q = self._rotate_q(q.float()).to(q_orig_dtype)
        head_size = q.shape[2]
        softmax_scale = self._transform_softmax_scale(softmax_scale, head_size)

        launch_packed_prefill(
            q=q,
            out=out,
            k_cache=k_cache,
            v_cache=v_cache,
            k_scale_cache=k_scale_cache,
            v_scale_cache=v_scale_cache,
            block_table=block_table,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            softmax_scale=softmax_scale,
            num_reqs=num_reqs,
            max_query_len=max_query_len,
            packing_factor=self.spec.packing_factor,
        )

        out_f = self._unrotate_out(out, head_size)
        out.copy_(out_f.to(q_orig_dtype))

