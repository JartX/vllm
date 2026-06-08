# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-token-head (int8/int4) KV cache attention path for ROCM_ATTN.

Split out of :mod:`rocm_attn` as a mixin so the backend file stays focused
on the native PagedAttention path. :class:`RocmAttentionImpl` inherits these
methods; they rely on the per-token-head attributes set in its ``__init__``
(``_kv_quant_mode``, ``_rdna3_*_ready``, ``_int4_scale``, ``max_num_kv_splits``,
``_max_cudagraph_capture_size`` and the ``scale`` / ``head_size`` basics).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from vllm.platforms import current_platform
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.attention.ops.triton_per_token_head_attention import (
    triton_per_token_head_attention,
    triton_per_token_head_prefill,
)
from vllm.v1.attention.ops.triton_prefill_attention import context_attention_fwd
from vllm.v1.attention.ops.triton_quant_kv import get_quant_kv_factory
from vllm.v1.attention.ops.triton_unified_attention import unified_attention
from vllm.v1.kv_cache_interface import KVQuantMode

if TYPE_CHECKING:
    from vllm.v1.attention.backends.rocm_attn import RocmAttentionMetadata

_CONTINUATION_DECODE_THRESHOLD = 128


class RocmPerTokenHeadMixin:
    """Per-token-head forward + scale-cache helpers for RocmAttentionImpl."""

    # Per-token-head scale caches (float32 strided views over KV cache bytes).
    _k_scale_cache: torch.Tensor | None = None
    _v_scale_cache: torch.Tensor | None = None
    _rht_signs: torch.Tensor | None = None

    def _get_rht_signs(self, device: torch.device) -> torch.Tensor:
        """Cached RHT D_1 signs [head_size] for the fused INT4 kernel."""
        if self._rht_signs is None:
            from vllm.v1.attention.ops.triton_quant_kv._hadamard import (
                _get_rht_signs,
            )

            self._rht_signs = _get_rht_signs(self.head_size, 0, device)
        return self._rht_signs

    def _ensure_scale_caches(self, kv_cache: torch.Tensor) -> None:
        """Strided float32 views over the per-head scale tail of the cache.

        Cache shape ``(num_blocks, 2, block_size, nkv, hs+pad)``; the last
        ``pad`` head elements hold one float32 scale. Scale view shape
        ``(num_blocks, block_size, nkv)``.
        """
        if self._k_scale_cache is not None:
            return

        num_blocks, _, block_size, nkv, padded_hs = kv_cache.shape
        dtype_sz = kv_cache.element_size()
        scale_pad = get_dtype_size(torch.float32) // dtype_sz
        hs = padded_hs - scale_pad

        raw = kv_cache.untyped_storage()
        base_f32 = torch.tensor([], dtype=torch.float32, device=kv_cache.device).set_(
            raw
        )

        kv_half_bytes = block_size * nkv * padded_hs * dtype_sz
        full_block_f32 = 2 * kv_half_bytes // 4
        slot_f32 = nkv * padded_hs * dtype_sz // 4
        head_f32 = padded_hs * dtype_sz // 4
        scale_off_f32 = hs * dtype_sz // 4

        self._k_scale_cache = torch.as_strided(
            base_f32,
            size=(num_blocks, block_size, nkv),
            stride=(full_block_f32, slot_f32, head_f32),
            storage_offset=scale_off_f32,
        )
        self._k_scale_cache.fill_(1.0)

        v_base_f32 = kv_half_bytes // 4
        self._v_scale_cache = torch.as_strided(
            base_f32,
            size=(num_blocks, block_size, nkv),
            stride=(full_block_f32, slot_f32, head_f32),
            storage_offset=v_base_f32 + scale_off_f32,
        )
        self._v_scale_cache.fill_(1.0)

    def _pth_unified_attention(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        max_seqlen_q: int,
        seqused_k: torch.Tensor,
        max_seqlen_k: int,
        block_table: torch.Tensor,
        causal: bool,
        alibi_slopes: torch.Tensor | None,
        window_size: tuple[int, int],
        softcap: float,
        sinks: torch.Tensor | None,
        output_scale: torch.Tensor | None,
        mm_prefix_range: torch.Tensor | None,
        seq_threshold_3D: int,
        num_par_softmax_segments: int,
        softmax_segm_output: torch.Tensor | None,
        softmax_segm_max: torch.Tensor | None,
        softmax_segm_expsum: torch.Tensor | None,
        k_scale_cache: torch.Tensor | None,
        v_scale_cache: torch.Tensor | None,
    ) -> None:
        """INT4 -> quant-kv factory packed kernel; INT8/FP8 -> shared kernel."""
        if self._kv_quant_mode == KVQuantMode.INT4_PER_TOKEN_HEAD:
            get_quant_kv_factory(self._kv_quant_mode).unified_attention(
                q=q,
                k_cache=k,
                v_cache=v,
                out=out,
                cu_seqlens_q=cu_seqlens_q,
                max_seqlen_q=max_seqlen_q,
                seqused_k=seqused_k,
                max_seqlen_k=max_seqlen_k,
                softmax_scale=self.scale,
                window_size=window_size,
                block_table=block_table,
                softcap=softcap,
                sinks=sinks,
                alibi_slopes=alibi_slopes,
                use_alibi_sqrt=self.use_alibi_sqrt,
                qq_bias=None,
                output_scale=output_scale,
                mm_prefix_range=mm_prefix_range,
                k_scale_cache=k_scale_cache,
                v_scale_cache=v_scale_cache,
                seq_threshold_3D=seq_threshold_3D,
                num_par_softmax_segments=num_par_softmax_segments,
                softmax_segm_output=softmax_segm_output,
                softmax_segm_max=softmax_segm_max,
                softmax_segm_expsum=softmax_segm_expsum,
            )
            return
        unified_attention(
            q=q,
            k=k,
            v=v,
            out=out,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            seqused_k=seqused_k,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.scale,
            causal=causal,
            alibi_slopes=alibi_slopes,
            use_alibi_sqrt=self.use_alibi_sqrt,
            window_size=window_size,
            block_table=block_table,
            softcap=softcap,
            q_descale=None,
            k_descale=None,
            v_descale=None,
            seq_threshold_3D=seq_threshold_3D,
            num_par_softmax_segments=num_par_softmax_segments,
            softmax_segm_output=softmax_segm_output,
            softmax_segm_max=softmax_segm_max,
            softmax_segm_expsum=softmax_segm_expsum,
            sinks=sinks,
            output_scale=output_scale,
            mm_prefix_range=mm_prefix_range,
            kv_quant_mode=self._kv_quant_mode,
            k_scale_cache=k_scale_cache,
            v_scale_cache=v_scale_cache,
        )

    def _forward_per_token_head(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: RocmAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None,
    ) -> torch.Tensor:
        """Per-token-head (int8/int4) attention: RDNA3 HIP fast-paths with
        Triton split-KV / flash / unified_attention fallback."""
        num_actual_tokens = attn_metadata.num_actual_tokens

        # RDNA3 INT8 decode (HS=256).
        if (
            self._rdna3_int8_decode_ready
            and self._k_scale_cache is not None
            and attn_metadata.max_query_len <= _CONTINUATION_DECODE_THRESHOLD
            and attn_metadata.q_to_req is not None
        ):
            self._ensure_scale_caches(kv_cache)
            key_cache, value_cache = kv_cache.unbind(1)
            mid_o_buf = getattr(layer, "_pth_mid_o_buf", None)
            if mid_o_buf is None or mid_o_buf.shape[0] < query.size(0):
                buf_size = max(query.size(0), self._max_cudagraph_capture_size)
                mid_o_buf = torch.zeros(
                    buf_size,
                    self.num_heads,
                    self.max_num_kv_splits,
                    self.head_size + 2,
                    dtype=torch.float32,
                    device=query.device,
                )
                layer._pth_mid_o_buf = mid_o_buf
            torch.ops._C.pth_decode_int8_rdna3(
                output[:num_actual_tokens],
                query[:num_actual_tokens],
                key_cache,
                value_cache,
                self._k_scale_cache,
                self._v_scale_cache,
                attn_metadata.block_table,
                attn_metadata.q_to_req,
                attn_metadata.q_to_klen,
                mid_o_buf,
                self.scale,
                self.max_num_kv_splits,
            )
            return output

        # RDNA3 INT4 decode (HS=128/256).
        if (
            self._rdna3_int4_decode_ready
            and self._rht_signs is not None
            and self._k_scale_cache is not None
            and attn_metadata.max_query_len <= _CONTINUATION_DECODE_THRESHOLD
            and attn_metadata.q_to_req is not None
        ):
            self._ensure_scale_caches(kv_cache)
            key_cache, value_cache = kv_cache.unbind(1)
            mid_o_buf = getattr(layer, "_pth_mid_o_buf", None)
            if mid_o_buf is None or mid_o_buf.shape[0] < query.size(0):
                buf_size = max(query.size(0), self._max_cudagraph_capture_size)
                mid_o_buf = torch.zeros(
                    buf_size,
                    self.num_heads,
                    self.max_num_kv_splits,
                    self.head_size + 2,
                    dtype=torch.float32,
                    device=query.device,
                )
                layer._pth_mid_o_buf = mid_o_buf
            q_slice = query[:num_actual_tokens]
            o_slice = output[:num_actual_tokens]
            if self.head_size > 128:  # HS=256: Q rotated in, output rotated out
                q_rot = getattr(layer, "_pth_q_rot_buf", None)
                if q_rot is None or q_rot.shape[0] < query.size(0):
                    buf_size = max(query.size(0), self._max_cudagraph_capture_size)
                    q_rot = torch.empty(
                        buf_size,
                        self.num_heads,
                        self.head_size,
                        dtype=query.dtype,
                        device=query.device,
                    )
                    layer._pth_q_rot_buf = q_rot
                q_rot[:num_actual_tokens].copy_(q_slice)
                q_slice = q_rot[:num_actual_tokens]
                torch.ops._C.rht_rotate_inplace_rdna3(
                    q_slice, self._rht_signs, False, 1.0
                )
            torch.ops._C.pth_decode_int4_rdna3(
                o_slice,
                q_slice,
                key_cache,
                value_cache,
                self._k_scale_cache,
                self._v_scale_cache,
                self._rht_signs,
                attn_metadata.block_table,
                attn_metadata.q_to_req,
                attn_metadata.q_to_klen,
                mid_o_buf,
                self._int4_scale,
                self.max_num_kv_splits,
            )
            if self.head_size > 128:
                torch.ops._C.rht_rotate_inplace_rdna3(
                    o_slice, self._rht_signs, True, 1.0 / self.head_size
                )
            return output

        # RDNA3 INT4 continuation prefill.
        if (
            self._rdna3_int4_prefill_ready
            and attn_metadata.num_decodes == 0
            and not attn_metadata.all_pure_first_prefill
        ):
            self._ensure_scale_caches(kv_cache)
            key_cache, value_cache = kv_cache.unbind(1)
            rht_signs = self._get_rht_signs(query.device)
            torch.ops._C.paged_prefill_attn_rdna3_int4(
                output[:num_actual_tokens],
                query[:num_actual_tokens],
                key_cache,
                value_cache,
                self._k_scale_cache,
                self._v_scale_cache,
                rht_signs,
                attn_metadata.block_table,
                attn_metadata.query_start_loc,
                attn_metadata.seq_lens,
                attn_metadata.max_query_len,
                self._int4_scale,
                True,
            )
            return output

        # Batches with a prefill chunk; decode-only falls through below.
        if (
            self.alibi_slopes is None
            and not self.use_alibi_sqrt
            and self.sinks is None
            and not self.logits_soft_cap
            and attn_metadata.mm_prefix_range_tensor is None
            and output_scale is None
            and self.kv_sharing_target_layer_name is None
            and key is not None
            and value is not None
        ):
            num_dec = attn_metadata.num_decodes
            num_dec_tok = attn_metadata.num_decode_tokens
            pref_first_chunk = attn_metadata.prefill_is_first_chunk
            all_first_chunk = attn_metadata.all_pure_first_prefill

            # Pure first-chunk prefill.
            if num_dec == 0 and all_first_chunk:
                context_attention_fwd(
                    q=query[:num_actual_tokens],
                    k=key[:num_actual_tokens],
                    v=value[:num_actual_tokens],
                    o=output[:num_actual_tokens],
                    b_start_loc=attn_metadata.query_start_loc,
                    b_seq_len=attn_metadata.seq_lens,
                    max_input_len=attn_metadata.max_query_len,
                    is_causal=True,
                    softmax_scale=self.scale,
                    sliding_window_q=self.sliding_window[0],
                    sliding_window_k=self.sliding_window[1],
                )
                return output

            # Mixed decode + first-chunk prefill.
            if num_dec > 0 and num_dec_tok < num_actual_tokens and pref_first_chunk:
                self._ensure_scale_caches(kv_cache)
                key_cache, value_cache = kv_cache.unbind(1)
                self._pth_unified_attention(
                    q=query[:num_dec_tok],
                    k=key_cache,
                    v=value_cache,
                    out=output[:num_dec_tok],
                    cu_seqlens_q=attn_metadata.query_start_loc[: num_dec + 1],
                    max_seqlen_q=1,
                    seqused_k=attn_metadata.seq_lens[:num_dec],
                    max_seqlen_k=attn_metadata.max_seq_len,
                    block_table=attn_metadata.block_table[:num_dec],
                    causal=True,
                    alibi_slopes=None,
                    window_size=self.sliding_window,
                    softcap=0,
                    sinks=None,
                    output_scale=None,
                    mm_prefix_range=None,
                    seq_threshold_3D=attn_metadata.seq_threshold_3D,
                    num_par_softmax_segments=attn_metadata.num_par_softmax_segments,
                    softmax_segm_output=attn_metadata.softmax_segm_output,
                    softmax_segm_max=attn_metadata.softmax_segm_max,
                    softmax_segm_expsum=attn_metadata.softmax_segm_expsum,
                    k_scale_cache=self._k_scale_cache,
                    v_scale_cache=self._v_scale_cache,
                )
                pref_qsl = attn_metadata.query_start_loc[num_dec:] - num_dec_tok
                context_attention_fwd(
                    q=query[num_dec_tok:num_actual_tokens],
                    k=key[num_dec_tok:num_actual_tokens],
                    v=value[num_dec_tok:num_actual_tokens],
                    o=output[num_dec_tok:num_actual_tokens],
                    b_start_loc=pref_qsl,
                    b_seq_len=attn_metadata.seq_lens[num_dec:],
                    max_input_len=attn_metadata.max_query_len,
                    is_causal=True,
                    softmax_scale=self.scale,
                    sliding_window_q=self.sliding_window[0],
                    sliding_window_k=self.sliding_window[1],
                )
                return output

            # Pure prefill with a continuation chunk.
            if (
                num_dec == 0
                and num_actual_tokens > 0
                and self.sliding_window == (-1, -1)
            ):
                self._ensure_scale_caches(kv_cache)
                key_cache, value_cache = kv_cache.unbind(1)
                k_scale_cache = self._k_scale_cache
                v_scale_cache = self._v_scale_cache
                use_qk_int8_wmma = (
                    key_cache.dtype == torch.int8 and current_platform.is_rocm()
                )
                _head_size = query.shape[2]
                if (
                    use_qk_int8_wmma
                    and _head_size in (64, 128, 256)
                    and hasattr(torch.ops, "_C")
                    and hasattr(torch.ops._C, "paged_prefill_attn_rdna3_int8")
                ):
                    torch.ops._C.paged_prefill_attn_rdna3_int8(
                        output[:num_actual_tokens],
                        query[:num_actual_tokens],
                        key[:num_actual_tokens],
                        value[:num_actual_tokens],
                        key_cache,
                        value_cache,
                        k_scale_cache,
                        v_scale_cache,
                        attn_metadata.block_table,
                        attn_metadata.query_start_loc,
                        attn_metadata.seq_lens,
                        attn_metadata.max_query_len,
                        self.scale,
                        True,
                    )
                    return output

                if self._rdna3_int4_prefill_ready:
                    torch.ops._C.paged_prefill_attn_rdna3_int4(
                        output[:num_actual_tokens],
                        query[:num_actual_tokens],
                        key_cache,
                        value_cache,
                        k_scale_cache,
                        v_scale_cache,
                        self._get_rht_signs(query.device),
                        attn_metadata.block_table,
                        attn_metadata.query_start_loc,
                        attn_metadata.seq_lens,
                        attn_metadata.max_query_len,
                        self._int4_scale,
                        True,
                    )
                    return output

                num_reqs_pref = attn_metadata.query_start_loc.shape[0] - 1
                triton_per_token_head_prefill(
                    query=query[:num_actual_tokens],
                    output=output[:num_actual_tokens],
                    key_cache=key_cache,
                    value_cache=value_cache,
                    k_scale_cache=k_scale_cache,
                    v_scale_cache=v_scale_cache,
                    block_table=attn_metadata.block_table,
                    query_start_loc=attn_metadata.query_start_loc,
                    seq_lens=attn_metadata.seq_lens,
                    softmax_scale=self.scale,
                    num_reqs=num_reqs_pref,
                    max_query_len=attn_metadata.max_query_len,
                    use_qk_int8_wmma=use_qk_int8_wmma,
                    kv_quant_mode=self._kv_quant_mode,
                )
                return output

            # Mixed decode + continuation prefill.
            if (
                num_dec > 0
                and num_dec_tok < num_actual_tokens
                and self.sliding_window == (-1, -1)
            ):
                self._ensure_scale_caches(kv_cache)
                key_cache, value_cache = kv_cache.unbind(1)
                k_scale_cache = self._k_scale_cache
                v_scale_cache = self._v_scale_cache
                self._pth_unified_attention(
                    q=query[:num_dec_tok],
                    k=key_cache,
                    v=value_cache,
                    out=output[:num_dec_tok],
                    cu_seqlens_q=attn_metadata.query_start_loc[: num_dec + 1],
                    max_seqlen_q=1,
                    seqused_k=attn_metadata.seq_lens[:num_dec],
                    max_seqlen_k=attn_metadata.max_seq_len,
                    block_table=attn_metadata.block_table[:num_dec],
                    causal=True,
                    alibi_slopes=None,
                    window_size=self.sliding_window,
                    softcap=0,
                    sinks=None,
                    output_scale=None,
                    mm_prefix_range=None,
                    seq_threshold_3D=attn_metadata.seq_threshold_3D,
                    num_par_softmax_segments=attn_metadata.num_par_softmax_segments,
                    softmax_segm_output=attn_metadata.softmax_segm_output,
                    softmax_segm_max=attn_metadata.softmax_segm_max,
                    softmax_segm_expsum=attn_metadata.softmax_segm_expsum,
                    k_scale_cache=k_scale_cache,
                    v_scale_cache=v_scale_cache,
                )
                pref_qsl = attn_metadata.query_start_loc[num_dec:] - num_dec_tok
                if self._rdna3_int4_prefill_ready:
                    torch.ops._C.paged_prefill_attn_rdna3_int4(
                        output[num_dec_tok:num_actual_tokens],
                        query[num_dec_tok:num_actual_tokens],
                        key_cache,
                        value_cache,
                        k_scale_cache,
                        v_scale_cache,
                        self._get_rht_signs(query.device),
                        attn_metadata.block_table[num_dec:],
                        pref_qsl,
                        attn_metadata.seq_lens[num_dec:],
                        attn_metadata.max_query_len,
                        self._int4_scale,
                        True,
                    )
                else:
                    num_reqs_pref = attn_metadata.query_start_loc.shape[0] - 1 - num_dec
                    use_qk_int8_wmma = (
                        key_cache.dtype == torch.int8 and current_platform.is_rocm()
                    )
                    if (
                        use_qk_int8_wmma
                        and query.shape[2] in (64, 128, 256)
                        and hasattr(torch.ops, "_C")
                        and hasattr(torch.ops._C, "paged_prefill_attn_rdna3_int8")
                    ):
                        torch.ops._C.paged_prefill_attn_rdna3_int8(
                            output[num_dec_tok:num_actual_tokens],
                            query[num_dec_tok:num_actual_tokens],
                            key[num_dec_tok:num_actual_tokens],
                            value[num_dec_tok:num_actual_tokens],
                            key_cache,
                            value_cache,
                            k_scale_cache,
                            v_scale_cache,
                            attn_metadata.block_table[num_dec:],
                            pref_qsl,
                            attn_metadata.seq_lens[num_dec:],
                            attn_metadata.max_query_len,
                            self.scale,
                            True,
                        )
                    else:
                        triton_per_token_head_prefill(
                            query=query[num_dec_tok:num_actual_tokens],
                            output=output[num_dec_tok:num_actual_tokens],
                            key_cache=key_cache,
                            value_cache=value_cache,
                            k_scale_cache=k_scale_cache,
                            v_scale_cache=v_scale_cache,
                            block_table=attn_metadata.block_table[num_dec:],
                            query_start_loc=pref_qsl,
                            seq_lens=attn_metadata.seq_lens[num_dec:],
                            softmax_scale=self.scale,
                            num_reqs=num_reqs_pref,
                            max_query_len=attn_metadata.max_query_len,
                            use_qk_int8_wmma=use_qk_int8_wmma,
                            kv_quant_mode=self._kv_quant_mode,
                        )
                return output

        # Decode / small continuation via split-KV; larger falls through below.
        self._ensure_scale_caches(kv_cache)
        key_cache, value_cache = kv_cache.unbind(1)
        k_scale_cache = self._k_scale_cache
        v_scale_cache = self._v_scale_cache

        if (
            attn_metadata.max_query_len <= _CONTINUATION_DECODE_THRESHOLD
            and attn_metadata.q_to_req is not None
            and attn_metadata.q_to_klen is not None
            and self.alibi_slopes is None
            and not self.use_alibi_sqrt
            and self.sinks is None
            and not self.logits_soft_cap
            and self.sliding_window == (-1, -1)
            and attn_metadata.mm_prefix_range_tensor is None
            and output_scale is None
        ):
            if self._rdna3_int4_decode_ready:
                mid_o_buf = getattr(layer, "_pth_mid_o_buf", None)
                if mid_o_buf is None:
                    mid_o_buf = torch.zeros(
                        query.size(0),
                        self.num_heads,
                        self.max_num_kv_splits,
                        self.head_size + 2,
                        dtype=torch.float32,
                        device=query.device,
                    )
                    layer._pth_mid_o_buf = mid_o_buf
                rht_signs = self._get_rht_signs(query.device)
                q_slice = query[:num_actual_tokens]
                o_slice = output[:num_actual_tokens]
                if self.head_size > 128:
                    q_rot = getattr(layer, "_pth_q_rot_buf", None)
                    if q_rot is None or q_rot.shape[0] < num_actual_tokens:
                        q_rot = torch.empty(
                            query.size(0),
                            self.num_heads,
                            self.head_size,
                            dtype=query.dtype,
                            device=query.device,
                        )
                        layer._pth_q_rot_buf = q_rot
                    q_rot[:num_actual_tokens].copy_(q_slice)
                    q_slice = q_rot[:num_actual_tokens]
                    torch.ops._C.rht_rotate_inplace_rdna3(
                        q_slice, rht_signs, False, 1.0
                    )
                torch.ops._C.pth_decode_int4_rdna3(
                    o_slice,
                    q_slice,
                    key_cache,
                    value_cache,
                    k_scale_cache,
                    v_scale_cache,
                    rht_signs,
                    attn_metadata.block_table,
                    attn_metadata.q_to_req,
                    attn_metadata.q_to_klen,
                    mid_o_buf,
                    self._int4_scale,
                    self.max_num_kv_splits,
                )
                if self.head_size > 128:
                    torch.ops._C.rht_rotate_inplace_rdna3(
                        o_slice, rht_signs, True, 1.0 / self.head_size
                    )
                return output

            mid_o_buf = getattr(layer, "_pth_mid_o_buf", None)
            output_buf = getattr(layer, "_pth_output_buf", None)
            lse_buf = getattr(layer, "_pth_lse_buf", None)
            use_qk_int8_wmma = (
                key_cache.dtype == torch.int8 and current_platform.is_rocm()
            )
            triton_per_token_head_attention(
                query=query[:num_actual_tokens],
                key_cache=key_cache,
                value_cache=value_cache,
                k_scale_cache=k_scale_cache,
                v_scale_cache=v_scale_cache,
                block_table=attn_metadata.block_table,
                q_to_req=attn_metadata.q_to_req,
                q_to_klen=attn_metadata.q_to_klen,
                scale=self.scale,
                max_num_kv_splits=self.max_num_kv_splits,
                block_kv=(
                    32 if self._kv_quant_mode == KVQuantMode.INT4_PER_TOKEN_HEAD else 16
                ),
                output=output[:num_actual_tokens],
                mid_o_buf=mid_o_buf,
                output_buf=output_buf,
                lse_buf=lse_buf,
                buf_holder=layer,
                use_qk_int8_wmma=use_qk_int8_wmma,
                kv_quant_mode=self._kv_quant_mode,
            )
            return output

        # Fallback: large continuation or special features.
        self._pth_unified_attention(
            q=query[:num_actual_tokens],
            k=key_cache,
            v=value_cache,
            out=output[:num_actual_tokens],
            cu_seqlens_q=attn_metadata.query_start_loc,
            max_seqlen_q=attn_metadata.max_query_len,
            seqused_k=attn_metadata.seq_lens,
            max_seqlen_k=attn_metadata.max_seq_len,
            block_table=attn_metadata.block_table,
            causal=True,
            alibi_slopes=self.alibi_slopes,
            window_size=self.sliding_window,
            softcap=self.logits_soft_cap,
            sinks=self.sinks,
            output_scale=output_scale,
            mm_prefix_range=attn_metadata.mm_prefix_range_tensor,
            seq_threshold_3D=attn_metadata.seq_threshold_3D,
            num_par_softmax_segments=attn_metadata.num_par_softmax_segments,
            softmax_segm_output=attn_metadata.softmax_segm_output,
            softmax_segm_max=attn_metadata.softmax_segm_max,
            softmax_segm_expsum=attn_metadata.softmax_segm_expsum,
            k_scale_cache=k_scale_cache,
            v_scale_cache=v_scale_cache,
        )
        return output
