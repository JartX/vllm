# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP8 per-tensor KV-cache plugin — metadata marker only.

``fp8_per_tensor`` is the classic FP8 KV cache path: a single
process-wide ``k_scale`` / ``v_scale`` pair multiplies every read.
The runtime work for this mode lives inside the core kernel
(:func:`kernel_unified_attention`) under the
``KV_QUANT_MODE == 1`` constexpr branch, and the write path is
:func:`triton_reshape_and_cache_flash` with a per-tensor scalar
(``layer._k_scale`` / ``layer._v_scale``).  Neither needs a plugin
hook.

This file exists so the registry can answer metadata questions about
the mode — storage dtype, packing factor, scale topology — uniformly
with every other mode.  The :attr:`QuantKVSpec.is_metadata_marker`
flag documents that this plugin is inspected, not invoked.

Storage dtype is ``torch.float8_e4m3fn`` on CUDA / ``torch.float8_e4m3fnuz``
on ROCm; the concrete value is resolved via the platform helper at
registration time.
"""

from __future__ import annotations

from vllm.platforms import current_platform
from vllm.v1.attention.ops.triton_quant_kv import register
from vllm.v1.attention.ops.triton_quant_kv.base import (
    QuantKVPlugin,
    QuantKVSpec,
)


class Fp8PerTensorPlugin(QuantKVPlugin):
    """Metadata marker for the ``KVQuantMode.FP8_PER_TENSOR`` mode."""

    spec = QuantKVSpec(
        name="fp8_per_tensor",
        storage_dtype=current_platform.fp8_dtype(),
        packing_factor=1,
        needs_per_token_head_scales=False,
        needs_per_tensor_scale=True,
        is_metadata_marker=True,
        description=(
            "FP8 per-tensor KV cache; scales live on the attention layer "
            "and the write/read paths are handled by the core kernel"
        ),
    )

    def reshape_and_cache(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "Fp8PerTensorPlugin is a metadata marker.  The FP8 per-tensor "
            "write path is handled by `triton_reshape_and_cache_flash` "
            "directly and must not be routed through the plugin dispatcher."
        )


register(Fp8PerTensorPlugin())
