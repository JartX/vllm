# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unquantized (pass-through) KV-cache plugin — metadata marker only.

``none`` is the default mode: KV caches are stored in the model's
compute dtype (fp16, bf16, …) with no quantization, packing, or
per-(token, head) scales.  There is nothing for a plugin to do at
runtime — both the write path (:func:`triton_reshape_and_cache_flash`)
and the read path (:func:`kernel_unified_attention`) handle this mode
without any dispatch, and the attention backend bypasses the plugin
interface entirely for ``NONE``.

This file exists so the registry is the single source of truth for
*metadata*: the cache allocator can query the plugin for the storage
dtype and packing factor instead of hard-coding cases.  The
:attr:`QuantKVSpec.is_metadata_marker` flag documents that this plugin
is not meant to be *invoked* — only inspected.

.. note::
   Calling :meth:`reshape_and_cache` or :meth:`unified_attention` on
   this plugin raises :class:`NotImplementedError` on purpose.  If
   you hit that exception the dispatcher is mis-configured —
   unquantized paths must bypass the plugin interface.
"""

from __future__ import annotations

from vllm.v1.attention.ops.triton_quant_kv import register
from vllm.v1.attention.ops.triton_quant_kv.base import (
    QuantKVPlugin,
    QuantKVSpec,
)


class NonePlugin(QuantKVPlugin):
    """Metadata marker for the unquantized cache mode.

    ``storage_dtype`` is ``None`` because the concrete dtype is the
    model compute dtype and is not known at plugin registration.
    Consumers needing the concrete dtype read it from the allocated
    cache tensor (or from :attr:`vllm.config.ModelConfig.dtype`).
    """

    spec = QuantKVSpec(
        name="none",
        storage_dtype=None,
        packing_factor=1,
        needs_per_token_head_scales=False,
        needs_per_tensor_scale=False,
        is_metadata_marker=True,
        description="unquantized pass-through; handled by the core kernel",
    )

    def reshape_and_cache(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "NonePlugin is a metadata marker.  The unquantized write path "
            "is handled by `triton_reshape_and_cache_flash` directly and "
            "must not be routed through the plugin dispatcher."
        )


register(NonePlugin())
