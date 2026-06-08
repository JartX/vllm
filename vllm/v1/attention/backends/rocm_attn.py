# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention layer with PagedAttention and Triton prefix prefill."""

from dataclasses import dataclass
from typing import ClassVar

import torch

from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8StaticTensorSym,
)
from vllm.platforms import current_platform
from vllm.utils.math_utils import next_power_of_2
from vllm.utils.torch_utils import get_dtype_size, is_quantized_kv_cache
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.ops.chunked_prefill_paged_decode import (
    chunked_prefill_paged_decode,
    has_native_kv_cache_layout,
)
from vllm.v1.attention.ops.paged_attn import PagedAttention
from vllm.v1.attention.ops.triton_per_token_head_attention import (
    triton_per_token_head_attention,
    triton_per_token_head_prefill,
)
from vllm.v1.attention.ops.triton_prefill_attention import context_attention_fwd
from vllm.v1.attention.ops.triton_quant_kv import get_quant_kv_factory
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    triton_reshape_and_cache_flash,
)
from vllm.v1.attention.ops.triton_unified_attention import unified_attention
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVQuantMode,
    get_kv_quant_mode,
    kv_cache_uses_per_token_head_scales,
)

logger = init_logger(__name__)

_CONTINUATION_DECODE_THRESHOLD = 128
MIN_LAUNCH_GRID_SIZE_2D = 128
NUM_PAR_SOFTMAX_SEGMENTS = 16


@dataclass
class RocmAttentionMetadata:
    # NOTE(sang): Definition of context_len, query_len, and seq_len.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ---------------------|
    #                                   |-- query_len ---|

    num_actual_tokens: int  # Number of tokens excluding padding.
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor

    # For cascade attention.
    use_cascade: bool
    common_prefix_len: int
    cu_prefix_query_lens: torch.Tensor | None
    prefix_kv_lens: torch.Tensor | None
    suffix_kv_lens: torch.Tensor | None

    # Optional aot scheduling
    scheduler_metadata: torch.Tensor | None = None
    prefix_scheduler_metadata: torch.Tensor | None = None

    # DFlash drafting sets this to False via CommonAttentionMetadata.
    causal: bool = True

    # Per-token-head quantization (int8/int4); defaults for the non-quant path.
    num_decodes: int = 0
    num_decode_tokens: int = 0
    prefill_is_first_chunk: bool = False
    all_pure_first_prefill: bool = False
    q_to_req: torch.Tensor | None = None
    q_to_klen: torch.Tensor | None = None
    seq_threshold_3D: int = 0
    num_par_softmax_segments: int = 0
    softmax_segm_output: torch.Tensor | None = None
    softmax_segm_max: torch.Tensor | None = None
    softmax_segm_expsum: torch.Tensor | None = None
    mm_prefix_range_tensor: torch.Tensor | None = None


class RocmAttentionMetadataBuilder(AttentionMetadataBuilder[RocmAttentionMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        self.block_size = kv_cache_spec.block_size

        model_config = vllm_config.model_config
        self.num_heads_q = model_config.get_num_attention_heads(
            vllm_config.parallel_config
        )
        self.num_heads_kv = model_config.get_num_kv_heads(vllm_config.parallel_config)
        self.headdim = model_config.get_head_size()

        self._is_per_token_head = kv_cache_spec.kv_quant_mode.is_per_token_head
        if self._is_per_token_head:
            self._init_reorder_batch_threshold(1, supports_spec_as_decode=False)
            # Persistent buffers; stable pointers across CUDA graph replay.
            max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
            self._q_to_req_buf = torch.empty(
                max_tokens, dtype=torch.int32, device=device
            )
            self._q_to_klen_buf = torch.empty(
                max_tokens, dtype=torch.int32, device=device
            )

            self.decode_cudagraph_enabled = (
                self.vllm_config.compilation_config.cudagraph_mode
                in (
                    CUDAGraphMode.FULL_AND_PIECEWISE,
                    CUDAGraphMode.FULL_DECODE_ONLY,
                    CUDAGraphMode.FULL,
                )
            )
            self.seq_threshold_3D = MIN_LAUNCH_GRID_SIZE_2D // self.num_heads_kv
            if self.decode_cudagraph_enabled:
                capture_sizes = (
                    self.vllm_config.compilation_config.cudagraph_capture_sizes
                )
                assert capture_sizes, (
                    "CUDA Graphs enabled but no capture sizes specified."
                )
                self.seq_threshold_3D = min(
                    capture_sizes,
                    key=lambda x: abs(x - self.seq_threshold_3D),
                )

            self.num_par_softmax_segments = NUM_PAR_SOFTMAX_SEGMENTS
            headdim_padded = next_power_of_2(self.headdim)
            self.softmax_segm_output = torch.empty(
                (
                    self.seq_threshold_3D,
                    self.num_heads_q,
                    self.num_par_softmax_segments,
                    headdim_padded,
                ),
                dtype=torch.float32,
                device=device,
            )
            self.softmax_segm_max = torch.empty(
                (
                    self.seq_threshold_3D,
                    self.num_heads_q,
                    self.num_par_softmax_segments,
                ),
                dtype=torch.float32,
                device=device,
            )
            self.softmax_segm_expsum = torch.empty(
                (
                    self.seq_threshold_3D,
                    self.num_heads_q,
                    self.num_par_softmax_segments,
                ),
                dtype=torch.float32,
                device=device,
            )

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> RocmAttentionMetadata:
        attn_metadata = self.build(0, common_attn_metadata)
        # When doing full graph capture, setting seq_lens to
        # max_model_len will cause graph capture to be extremely
        # slow, so here we set it to 1.
        attn_metadata.seq_lens.fill_(1)
        if self._is_per_token_head:
            attn_metadata.all_pure_first_prefill = False
            attn_metadata.prefill_is_first_chunk = False

        # Here we set the query start locs to 0. This is to
        # cover up an invalid memory access in the prefix_prefil kernel
        # that we run into during graph capture (#25985)
        common_attn_metadata.query_start_loc.zero_()
        common_attn_metadata.query_start_loc_cpu.zero_()

        return attn_metadata

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> RocmAttentionMetadata:
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        max_query_len = common_attn_metadata.max_query_len

        max_seq_len = common_attn_metadata.max_seq_len
        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens
        block_table_tensor = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping

        use_cascade = common_prefix_len > 0

        # Per-token-head: decode/prefill split + per-query maps. Gated on the
        # CPU seq_lens copy being ready to avoid a D2H sync.
        num_decodes = 0
        num_decode_tokens = 0
        prefill_is_first_chunk = False
        all_pure_first_prefill = False
        q_to_req = None
        q_to_klen = None
        if self._is_per_token_head:
            seq_lens_cpu = common_attn_metadata._seq_lens_cpu
            if seq_lens_cpu is not None:
                qsl_cpu = common_attn_metadata.query_start_loc_cpu
                query_lens_cpu = qsl_cpu[1:] - qsl_cpu[:-1]
                all_pure_first_prefill = bool(
                    torch.equal(query_lens_cpu, seq_lens_cpu.to(query_lens_cpu.dtype))
                )
                decode_mask = query_lens_cpu <= 1
                if bool(decode_mask.all()):
                    num_decodes = int(query_lens_cpu.shape[0])
                elif not bool(decode_mask[0]):
                    num_decodes = 0
                else:
                    num_decodes = int(decode_mask.to(torch.int32).sum().item())
                num_decode_tokens = int(qsl_cpu[num_decodes].item())
                if num_decodes < query_lens_cpu.shape[0]:
                    ql_pref = query_lens_cpu[num_decodes:]
                    sl_pref = seq_lens_cpu[num_decodes:].to(ql_pref.dtype)
                    prefill_is_first_chunk = bool(torch.equal(ql_pref, sl_pref))

                q_lens_i32 = query_lens_cpu.to(torch.int32)
                num_reqs_total = q_lens_i32.shape[0]
                total_q = int(qsl_cpu[-1].item())
                if total_q > 0:
                    if num_reqs_total == total_q:
                        # Pure decode: q_to_req = arange, q_to_klen = seq_lens.
                        q_to_req_cpu = torch.arange(num_reqs_total, dtype=torch.int32)
                        q_to_klen_cpu = seq_lens_cpu.to(torch.int32)
                    else:
                        qsl_i32 = qsl_cpu[:-1].to(torch.int32)
                        seq_lens_i32 = seq_lens_cpu.to(torch.int32)
                        q_to_req_cpu = torch.repeat_interleave(
                            torch.arange(num_reqs_total, dtype=torch.int32),
                            q_lens_i32,
                        )
                        cached_len_per_req = seq_lens_i32 - q_lens_i32
                        pos_in_req = (
                            torch.arange(total_q, dtype=torch.int32)
                            - qsl_i32[q_to_req_cpu.long()]
                        )
                        q_to_klen_cpu = (
                            cached_len_per_req[q_to_req_cpu.long()] + pos_in_req + 1
                        )
                    self._q_to_req_buf[:total_q].copy_(q_to_req_cpu, non_blocking=True)
                    self._q_to_klen_buf[:total_q].copy_(
                        q_to_klen_cpu, non_blocking=True
                    )
                    q_to_req = self._q_to_req_buf[:num_actual_tokens]
                    q_to_klen = self._q_to_klen_buf[:num_actual_tokens]

        if use_cascade:
            cu_prefix_query_lens = torch.tensor(
                [0, num_actual_tokens], dtype=torch.int32, device=self.device
            )
            prefix_kv_lens = torch.tensor(
                [common_prefix_len], dtype=torch.int32, device=self.device
            )
            suffix_kv_lens = common_attn_metadata.seq_lens.cpu() - common_prefix_len
            suffix_kv_lens = suffix_kv_lens.to(self.device)
        else:
            cu_prefix_query_lens = None
            prefix_kv_lens = None
            suffix_kv_lens = None
            prefix_scheduler_metadata = None

        attn_metadata = RocmAttentionMetadata(
            num_actual_tokens=num_actual_tokens,
            max_query_len=max_query_len,
            query_start_loc=query_start_loc,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            block_table=block_table_tensor,
            slot_mapping=slot_mapping,
            use_cascade=use_cascade,
            common_prefix_len=common_prefix_len,
            cu_prefix_query_lens=cu_prefix_query_lens,
            prefix_kv_lens=prefix_kv_lens,
            suffix_kv_lens=suffix_kv_lens,
            prefix_scheduler_metadata=prefix_scheduler_metadata,
            causal=common_attn_metadata.causal,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            prefill_is_first_chunk=prefill_is_first_chunk,
            all_pure_first_prefill=all_pure_first_prefill,
            q_to_req=q_to_req,
            q_to_klen=q_to_klen,
            seq_threshold_3D=getattr(self, "seq_threshold_3D", 0),
            num_par_softmax_segments=getattr(self, "num_par_softmax_segments", 0),
            softmax_segm_output=getattr(self, "softmax_segm_output", None),
            softmax_segm_max=getattr(self, "softmax_segm_max", None),
            softmax_segm_expsum=getattr(self, "softmax_segm_expsum", None),
        )
        return attn_metadata


class RocmAttentionBackend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
        "fp8_e5m2",
        "int8_per_token_head",
        "int4_per_token_head",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        # ROCM paged attention native C++ kernel only supports block sizes 16 and 32
        # due to shared memory (LDS) constraints on AMD GPUs.
        # See csrc/rocm/attention.cu CALL_CUSTOM_LAUNCHER_BLK macro.
        # However, vLLM allows support for any multiple of 16 via the Triton path.
        # As addressed in PR: https://github.com/vllm-project/vllm/pull/31380,
        # non-standard models (like qwen3-next with block_size 544, or qwen3_5
        # with 784 and 1056) are dynamically routed to our optimized Triton kernel
        # in `do_kv_cache_update`.
        return [MultipleOf(16)]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [32, 64, 80, 96, 128, 160, 192, 224, 256]

    @classmethod
    def supports_mm_prefix(cls) -> bool:
        return True

    @classmethod
    def supports_sink(cls) -> bool:
        # ROCM custom attention kernel does not support sinks.
        # Callink this backend with sinks will cause it to fall back to the Triton
        # kernel, which is less efficient than the proper triton backends.
        return False

    @classmethod
    def supports_non_causal(cls) -> bool:
        return True

    @classmethod
    def supports_kv_connector(cls) -> bool:
        # ROCM_ATTN uses (2, num_blocks, ...) KV cache layout which is
        # incompatible with KV connectors that require blocks-first layout.
        return False

    forward_includes_kv_cache_update: bool = False

    @staticmethod
    def get_name() -> str:
        return "ROCM_ATTN"

    @staticmethod
    def get_impl_cls() -> type["RocmAttentionImpl"]:
        return RocmAttentionImpl

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        """ENCODER_DECODER is not supported because
        chunked_prefill_paged_decode's prefill kernel (context_attention_fwd)
        assumes self-attention semantics: it treats passed K/V as new tokens
        to mix with cached K/V. For cross-attention layers the encoder K/V
        are already fully cached, so mixing them again produces incorrect
        results when max_query_len > 1 (e.g. beam search).
        """
        return attn_type in (
            AttentionType.DECODER,
            AttentionType.ENCODER,
            AttentionType.ENCODER_ONLY,
        )

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        if kv_cache_uses_per_token_head_scales(cache_dtype_str):
            # Blocks-first padded layout (as TRITON_ATTN): a float32 scale lives
            # in the padded tail of each head. No stride_order override, so the
            # physical layout matches this contiguous logical shape.
            from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE

            cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_dtype_str]
            scale_pad = get_dtype_size(torch.float32) // get_dtype_size(cache_dtype)
            data_head_size = get_kv_quant_mode(cache_dtype_str).packed_head_size(
                head_size
            )
            return (num_blocks, 2, block_size, num_kv_heads, data_head_size + scale_pad)
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def use_cascade_attention(*args, **kwargs) -> bool:
        return False

    @staticmethod
    def get_builder_cls() -> type["RocmAttentionMetadataBuilder"]:
        return RocmAttentionMetadataBuilder


class RocmAttentionImpl(AttentionImpl):
    # Per-token-head scale caches (float32 strided views over KV cache bytes).
    _k_scale_cache: torch.Tensor | None = None
    _v_scale_cache: torch.Tensor | None = None
    _rht_signs: torch.Tensor | None = None

    def fused_output_quant_supported(self, quant_key: QuantKey):
        return quant_key == kFp8StaticTensorSym

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: int | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None:
        self.attn_type = attn_type
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        if sliding_window is None:
            self.sliding_window = (-1, -1)
        else:
            self.sliding_window = (sliding_window - 1, 0)
        self.kv_cache_dtype = kv_cache_dtype
        if logits_soft_cap is None:
            # In flash-attn, setting logits_soft_cap as 0 means no soft cap.
            logits_soft_cap = 0
        self.logits_soft_cap = logits_soft_cap
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        self.fp8_dtype = current_platform.fp8_dtype()

        self.sinks = sinks
        if sinks is not None:
            assert sinks.shape[0] == num_heads, (
                "Sinks must have the same number of heads as the number of "
                f"heads in the layer. Sinks shape: {sinks.shape}, "
                f"num_heads: {num_heads}."
            )

        # No alibi-sqrt path on ROCM_ATTN; kept so the forward gates evaluate.
        self.use_alibi_sqrt = False

        # Per-token-head quantization (int8/int4) setup.
        self._kv_quant_mode = get_kv_quant_mode(kv_cache_dtype)
        self._is_per_token_head_quant = self._kv_quant_mode.is_per_token_head
        if self._is_per_token_head_quant:
            from vllm.config import get_current_vllm_config

            vllm_config = get_current_vllm_config()
            self.max_num_kv_splits = (
                vllm_config.attention_config.tq_max_kv_splits_for_cuda_graph
            )
            self._max_cudagraph_capture_size = (
                vllm_config.compilation_config.max_cudagraph_capture_size or 4
            )

        _is_rdna3_int4 = (
            self._kv_quant_mode == KVQuantMode.INT4_PER_TOKEN_HEAD
            and head_size in (128, 256)
            and current_platform.is_rocm()
            and hasattr(torch.ops, "_C")
        )
        _decode_hs_ok = head_size in (128, 256)
        _no_special_features = (
            alibi_slopes is None
            and not self.use_alibi_sqrt
            and sinks is None
            and not self.logits_soft_cap
            and sliding_window is None
            and kv_sharing_target_layer_name is None
        )
        self._rdna3_int4_decode_ready = (
            _is_rdna3_int4
            and _decode_hs_ok
            and _no_special_features
            and hasattr(torch.ops._C, "pth_decode_int4_rdna3")
        )
        self._rdna3_int4_prefill_ready = (
            _is_rdna3_int4
            and _no_special_features
            and hasattr(torch.ops._C, "paged_prefill_attn_rdna3_int4")
        )
        # INT8 HIP decode is HS=256 only.
        _is_rdna3_int8 = (
            self._kv_quant_mode == KVQuantMode.INT8_PER_TOKEN_HEAD
            and head_size == 256
            and current_platform.is_rocm()
            and hasattr(torch.ops, "_C")
        )
        self._rdna3_int8_decode_ready = (
            _is_rdna3_int8
            and _no_special_features
            and hasattr(torch.ops._C, "pth_decode_int8_rdna3")
        )
        self._int4_scale = self.scale / head_size if _is_rdna3_int4 else 0.0

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

    def _forward_encoder_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: RocmAttentionMetadata,
        layer: torch.nn.Module,
    ) -> torch.Tensor:
        """Forward pass for encoder attention without KV cache.

        Args:
            query: shape = [num_encoder_tokens, num_heads, head_size]
            key: shape = [num_encoder_tokens, num_kv_heads, head_size]
            value: shape = [num_encoder_tokens, num_kv_heads, head_size]
            output: shape = [num_encoder_tokens, num_heads, head_size]
            attn_metadata: Encoder attention metadata
            layer: The attention layer
        """
        # For encoder attention, process FP8 quantization if needed
        if is_quantized_kv_cache(self.kv_cache_dtype):
            raise NotImplementedError(
                "quantization is not supported for encoder attention"
            )

        # Use encoder-specific metadata for sequence information
        query_start_loc = attn_metadata.query_start_loc
        seq_lens = attn_metadata.seq_lens
        max_query_len = attn_metadata.max_query_len

        # Call flash attention directly on Q, K, V tensors
        from vllm.v1.attention.ops.triton_prefill_attention import context_attention_fwd

        context_attention_fwd(
            q=query,
            k=key,
            v=value,
            o=output,
            b_start_loc=query_start_loc,
            b_seq_len=seq_lens,
            max_input_len=max_query_len,
            is_causal=False,
            softmax_scale=self.scale,
            sliding_window_q=self.sliding_window[0],
            sliding_window_k=self.sliding_window[1],
        )
        return output

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: RocmAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with FlashAttention.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: shape =
                [2, num_blocks, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        if output_block_scale is not None:
            raise NotImplementedError(
                "fused block_scale output quantization is not yet supported"
                " for RocmAttentionImpl"
            )

        if attn_metadata is None:
            # Profiling run.
            return output.fill_(0)

        assert attn_metadata.use_cascade is False

        # IMPORTANT!
        # NOTE(woosuk): With piece-wise CUDA graphs, this method is executed in
        # eager-mode PyTorch. Thus, we need to be careful about any CPU overhead
        # in this method. For example, `view` and `slice` (or `[:n]`) operations
        # are surprisingly slow even in the case they do not invoke any GPU ops.
        # Minimize the PyTorch ops in this method as much as possible.
        # Whenever making a change in this method, please benchmark the
        # performance to make sure it does not introduce any overhead.

        num_actual_tokens = attn_metadata.num_actual_tokens

        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            return self._forward_encoder_attention(
                query[:num_actual_tokens],
                key[:num_actual_tokens],
                value[:num_actual_tokens],
                output[:num_actual_tokens],
                attn_metadata,
                layer,
            )

        # Per-token-head bypasses the native chunked_prefill_paged_decode path.
        if self._is_per_token_head_quant:
            return self._forward_per_token_head(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
            )

        key_cache, value_cache = PagedAttention.split_kv_cache(
            kv_cache, self.num_kv_heads, self.head_size
        )

        if is_quantized_kv_cache(self.kv_cache_dtype):
            key_cache = key_cache.view(self.fp8_dtype)
            value_cache = value_cache.view(self.fp8_dtype)
            assert layer._q_scale_float == 1.0, (
                "A non 1.0 q_scale is not currently supported."
            )

        cu_seqlens_q = attn_metadata.query_start_loc
        seqused_k = attn_metadata.seq_lens
        max_seqlen_q = attn_metadata.max_query_len
        max_seqlen_k = attn_metadata.max_seq_len
        block_table = attn_metadata.block_table

        # Compute attention and update output up to `num_actual_tokens`.
        chunked_prefill_paged_decode(
            query=query[:num_actual_tokens],
            key=key[:num_actual_tokens] if key is not None else None,
            value=value[:num_actual_tokens] if value is not None else None,
            output=output[:num_actual_tokens],
            kv_cache_dtype=self.kv_cache_dtype,
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=block_table,
            query_start_loc=cu_seqlens_q,
            seq_lens=seqused_k,
            max_seq_len=max_seqlen_k,
            max_query_len=max_seqlen_q,
            k_scale=layer._k_scale,
            v_scale=layer._v_scale,
            alibi_slopes=self.alibi_slopes,
            sliding_window=self.sliding_window[0],
            sm_scale=self.scale,
            output_scale=output_scale,
            sinks=self.sinks,
            causal=attn_metadata.causal,
        )

        return output

    def do_kv_cache_update(
        self,
        layer: AttentionLayer,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ):
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            return

        # Per-token-head: quantize/pack into the padded cache via the quant-kv
        # factory (HIP int4 reshape when available, else Triton).
        if self._is_per_token_head_quant:
            self._ensure_scale_caches(kv_cache)
            key_cache, value_cache = kv_cache.unbind(1)
            k_scale_cache = self._k_scale_cache
            v_scale_cache = self._v_scale_cache
            if (
                self._kv_quant_mode == KVQuantMode.FP8_PER_TOKEN_HEAD
                and key_cache.dtype == torch.uint8
            ):
                key_cache = key_cache.view(self.fp8_dtype)
                value_cache = value_cache.view(self.fp8_dtype)
            get_quant_kv_factory(self._kv_quant_mode).reshape_and_cache(
                key,
                value,
                key_cache,
                value_cache,
                slot_mapping,
                k_scale_cache=k_scale_cache,
                v_scale_cache=v_scale_cache,
            )
            return

        key_cache, value_cache = PagedAttention.split_kv_cache(
            kv_cache, self.num_kv_heads, self.head_size
        )

        # Reshape the input keys and values and store them in the cache.
        # Get the actual block_size from value_cache
        # value_cache shape: [num_blocks, num_heads, head_size, block_size]
        block_size = value_cache.shape[3]
        has_native_layout = has_native_kv_cache_layout(key_cache, value_cache)

        if block_size in (16, 32) and has_native_layout:
            # Normal 16, 32 with contiguous blocks: use vLLM native HIP C++ logic.
            PagedAttention.write_to_paged_cache(
                key,
                value,
                key_cache,
                value_cache,
                slot_mapping,
                self.kv_cache_dtype,
                layer._k_scale,
                layer._v_scale,
            )
        else:
            # Non-standard blocks and hybrid attention/Mamba layouts need the
            # stride-aware Triton writer. The native reshape_and_cache kernel
            # assumes contiguous block storage and writes to the wrong hybrid
            # cache blocks.
            triton_reshape_and_cache_flash(
                key,
                value,
                key_cache,
                value_cache,
                slot_mapping,
                self.kv_cache_dtype,
                layer._k_scale,
                layer._v_scale,
            )

    def fused_rope_kvcache_supported(self):
        # Fused aiter rope+cache can't quantize per-token-head; use the
        # separate do_kv_cache_update path instead.
        if self._is_per_token_head_quant:
            return False
        return rocm_aiter_ops.is_enabled()

    def do_rope_and_kv_cache_update(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        is_neox: bool,
        kv_cache: torch.Tensor,
        layer_slot_mapping: torch.Tensor,
    ):
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            return
        key_cache, value_cache = PagedAttention.split_kv_cache(
            kv_cache,
            layer.num_kv_heads,  # type: ignore[attr-defined]
            layer.head_size,  # type: ignore[attr-defined]
        )
        flash_layout = False

        is_fp8_kv_cache = is_quantized_kv_cache(self.kv_cache_dtype)
        if is_fp8_kv_cache:
            key_cache = key_cache.view(self.fp8_dtype)
            value_cache = value_cache.view(self.fp8_dtype)

        rocm_aiter_ops.triton_rope_and_cache(
            query,
            key,
            value,
            positions,
            cos_sin_cache,
            is_neox,
            key_cache,
            value_cache,
            layer_slot_mapping,
            layer._k_scale,
            layer._v_scale,
            flash_layout,
            is_fp8_kv_cache,
        )
