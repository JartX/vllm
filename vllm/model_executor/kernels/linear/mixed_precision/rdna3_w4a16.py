# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""W4A16 GPTQ kernel for AMD RDNA3 (gfx1100) — fp16 + bf16.

Drop-in replacement for ExllamaLinearKernel on RDNA3 that adds native bf16
support. Two HIP kernels:

  * ``csrc/quantization/gptq/q_gemm_rdna3.cu`` — scalar dot-product, used
    for decode (M=1) and small batches. Exposed as
    ``torch.ops._C.gptq_gemm_rdna3``.
  * ``csrc/quantization/gptq/q_gemm_rdna3_wmma.cu`` — matrix-instruction
    path (v_wmma_f32_16x16x16_*_w32) used for prefill (M >= 16). Exposed
    as ``torch.ops._C.gptq_gemm_rdna3_wmma``. Optional: if the C++
    extension wasn't built with this op, we silently keep using the
    scalar path for all M.

Registered ahead of TritonW4A16LinearKernel for the ROCm-RDNA3 path; falls
through to the Triton kernel on non-RDNA3 ROCm devices (e.g. CDNA/MI300).
"""

import os

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    pack_quantized_values_into_int32,
)
from vllm.model_executor.parameter import BasevLLMParameter, permute_param_layout_
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig


def _resolve_wmma_op():
    """Resolve the WMMA op once. Called from process_weights_after_loading,
    BEFORE vLLM/Dynamo traces the forward pass (so the result is captured
    as a constant during torch.compile rather than triggering an untrace
    able setattr inside apply_weights)."""
    if os.environ.get("VLLM_RDNA3_DISABLE_WMMA", "") == "1":
        return None
    return getattr(torch.ops._C, "gptq_gemm_rdna3_wmma", None)


# Module-level cache: shared across all instances. Populated on first
# process_weights_after_loading call. Sentinel "_uninitialized" → not yet
# resolved. None → resolved and either env-disabled or the op doesn't
# exist (use scalar). Op → use WMMA for M >= 16.
_WMMA_OP_CACHE: object = "_uninitialized"


class RDNA3W4A16LinearKernel(MPLinearKernel):
    SUPPORTED_QUANT_TYPES = [scalar_types.uint4b8]

    @classmethod
    def get_min_capability(cls) -> int:
        # ROCm doesn't use compute capabilities the same way; gating happens
        # in can_implement via current_platform.on_gfx11().
        return 60

    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        if not current_platform.is_rocm():
            return False, "RDNA3 W4A16 kernel is ROCm-only"

        # `on_gfx11` is a module-level function in vllm.platforms.rocm, not
        # a method on the platform instance, so we import it lazily here
        # (the import is guarded by is_rocm() above).
        try:
            from vllm.platforms.rocm import on_gfx11
        except ImportError:
            return False, "vllm.platforms.rocm.on_gfx11 not available"
        if not on_gfx11():
            return False, "RDNA3 W4A16 kernel requires gfx11 (RDNA3)"

        # The HIP op is registered by the C++ extension; if a user is running
        # against a vLLM build that doesn't include it (e.g. partial rebuild),
        # fall through gracefully to the next kernel in the registry.
        if not hasattr(torch.ops._C, "gptq_gemm_rdna3"):
            return (
                False,
                "torch.ops._C.gptq_gemm_rdna3 missing — rebuild C++ extension",
            )

        if c.act_type not in (torch.float16, torch.bfloat16):
            return False, "RDNA3 W4A16 kernel only supports fp16 and bf16"

        if c.weight_type not in cls.SUPPORTED_QUANT_TYPES:
            return (
                False,
                f"Quant type ({c.weight_type}) not supported by "
                f"RDNA3 W4A16 kernel; supported: {cls.SUPPORTED_QUANT_TYPES}",
            )

        if c.group_size <= 0:
            return (
                False,
                "RDNA3 W4A16 kernel does not support channelwise quantization",
            )

        if c.full_weight_shape[0] % c.group_size != 0:
            return (
                False,
                f"Group size ({c.group_size}) does not evenly divide K "
                f"({c.full_weight_shape[0]})",
            )

        # Output features must be a multiple of the pack factor (8 nibbles per
        # int32) and of 8 so that qzeros (packed 4-bit per col) align cleanly
        # against the BLOCK_KN_SIZE*4 = 512 N-stride and per-thread 4 columns.
        if c.partition_weight_shape[1] % 8 != 0:
            return (
                False,
                "Output features must be a multiple of 8 for the RDNA3 "
                "W4A16 kernel (qzeros packing)",
            )

        if c.has_g_idx and c.partition_weight_shape[0] != c.full_weight_shape[0]:
            return (
                False,
                "Act-order with TP-partitioned input features is not "
                "supported by the RDNA3 W4A16 kernel",
            )

        return True, None

    # ----- Weight prep (identical layout/shuffle as ExllamaLinearKernel) -----

    def process_weights_after_loading(self, layer: torch.nn.Module):
        c = self.config
        device = getattr(layer, self.w_q_name).device

        # Synthesize zero points if the checkpoint doesn't carry them.
        if not c.zero_points:
            self.w_zp_name = "qzeros"
            groups = c.partition_weight_shape[0] // c.group_size
            out_features = c.partition_weight_shape[1]

            if c.weight_type.has_bias():
                # GPTQv1 quirk: the kernel adds 1 to the stored zero, so we
                # encode (bias - 1) here. See exllama.py for the link to the
                # documentation of this checkpoint-format wart.
                zeros = torch.full(
                    (groups, out_features),
                    c.weight_type.bias - 1,
                    dtype=torch.int32,
                    device=device,
                )
            else:
                raise NotImplementedError(
                    "RDNA3 W4A16 kernel: zero-bias 4-bit quant requires "
                    "explicit zero points (GPTQv1 +1 quirk)."
                )
            zeros = pack_quantized_values_into_int32(
                zeros, c.weight_type, packed_dim=1
            )
            setattr(
                layer, self.w_zp_name, torch.nn.Parameter(zeros, requires_grad=False)
            )

        # Act-order: convert g_idx to the inverse permutation array exllama
        # expects (kernel reads a[perm[k]] instead of using groups indirected
        # by g_idx[k]).
        if c.has_g_idx:

            def transform_w_g_idx(x):
                return torch.argsort(x).to(torch.int)

            self._transform_param(layer, self.w_gidx_name, transform_w_g_idx)  # type: ignore
        else:
            self.w_gidx_name = "g_idx"
            empty_g_idx = torch.nn.Parameter(
                torch.empty((0,), dtype=torch.int, device=device),
                requires_grad=False,
            )
            setattr(layer, self.w_gidx_name, empty_g_idx)

        def transform_w_q(x):
            assert isinstance(x, BasevLLMParameter)
            assert self.w_gidx_name is not None
            g_idx = getattr(layer, self.w_gidx_name)

            permute_param_layout_(x, input_dim=0, output_dim=1, packed_dim=0)
            x_cont = x.data.contiguous()
            # Same 4-bit shuffle as exllama. The RDNA3 kernel reads weights in
            # the same shuffled int32 layout and uses the (qa & 0x000F000F)
            # bit-trick on top.
            ops.gptq_shuffle(x_cont, g_idx, c.weight_type.size_bits)
            return x_cont

        def transform_w_s(x):
            assert isinstance(x, BasevLLMParameter)
            permute_param_layout_(x, input_dim=0, output_dim=1)
            x.data = x.data.contiguous()
            # Keep scales in the activation dtype (fp16 OR bf16) — the kernel
            # branches on dtype internally.
            return x.to(dtype=c.act_type)

        self._transform_param(layer, self.w_q_name, transform_w_q)
        self._transform_param(layer, self.w_s_name, transform_w_s)

    # ----- Forward --------------------------------------------------------

    # Sentinel: dispatch cache hasn't been populated yet on this instance.
    # We resolve torch.ops._C.gptq_gemm_rdna3_wmma lazily on the FIRST
    # apply_weights call (rather than at module import) because vLLM's
    # kernel registry imports this module before the C++ extension is
    # guaranteed to be fully attribute-registered. Touching torch.ops._C
    # at import time can fail and silently cause vLLM to drop this kernel
    # from the registry, falling back to TritonW4A16LinearKernel — which
    # is dramatically slower on RDNA3 (was the cause of an earlier 55→8
    # tk/s decode regression).
    _wmma_op = "_uninitialized"

    def _resolve_wmma_op(self):
        if os.environ.get("VLLM_RDNA3_DISABLE_WMMA", "") == "1":
            return None
        return getattr(torch.ops._C, "gptq_gemm_rdna3_wmma", None)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        c = self.config

        x_2d = x.reshape(-1, x.shape[-1])
        out_shape = x.shape[:-1] + (c.partition_weight_shape[1],)

        w_q, w_s, w_zp, w_g_idx = self._get_weight_params(layer)

        # Same GPTQv1 vs v2 distinction as ExllamaLinearKernel — the
        # MPLinearLayerConfig doesn't carry format info, so we hardcode v1
        # to mirror the exllama path's behavior.
        use_v2_format = False

        assert w_zp is not None, "Zero points are required by RDNA3 W4A16"
        assert w_g_idx is not None, "g_idx tensor (possibly empty) required"

        # Dispatch:
        #   * M >= 16 → WMMA matrix-instruction path (prefill / batched).
        #   * Else   → scalar dot-product (decode + small batch).
        # WMMA op is resolved once per instance (lazy) and cached. The
        # `m >= 16` check is the only per-call cost beyond the original
        # decode path.
        wmma_op = self._wmma_op
        if wmma_op == "_uninitialized":
            wmma_op = self._resolve_wmma_op()
            type(self)._wmma_op = wmma_op  # cache on the class — shared
            # One-shot diagnostic: print exactly once when this kernel is
            # first invoked, so you can confirm in logs that vLLM picked
            # RDNA3W4A16LinearKernel and not a fallback. Remove this
            # block after validating perf is back to baseline.
            print(
                f"[RDNA3W4A16] FIRST DISPATCH — wmma_op={wmma_op}, "
                f"m={x_2d.size(0)}, k={x_2d.size(1)}, "
                f"n={w_q.size(1)}",
                flush=True,
            )

        m = x_2d.size(0)
        if wmma_op is not None and m >= 16:
            output = wmma_op(
                x_2d, w_q, w_zp, w_s, w_g_idx, use_v2_format
            )
        else:
            output = ops.gptq_gemm_rdna3(
                x_2d, w_q, w_zp, w_s, w_g_idx, use_v2_format
            )

        if bias is not None:
            output.add_(bias)
        return output.reshape(out_shape)
