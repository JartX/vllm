# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP8 per-token-head KV cache quantization plugin.

Thin subclass of :class:`PerTokenHeadFactoryBase` (in
:mod:`_per_token_head_core`) that pins the symmetric clamp range to
the platform-appropriate FP8 interval.  The write path is shared
with every other byte-aligned per-token-head plugin; the attention
read path lives in the core kernel via the
``USE_PER_TOKEN_HEAD_SCALES`` constexpr branch.

Adding a new byte-aligned symmetric per-token-head mode = copy this
file, change the name + clamp range + storage dtype.  No other file
needs editing.
"""

from __future__ import annotations

from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_fp8_min_max,
)
from vllm.platforms import current_platform
from vllm.v1.attention.ops.triton_quant_kv import register
from vllm.v1.attention.ops.triton_quant_kv._per_token_head_core import (
    PerTokenHeadFactoryBase,
)
from vllm.v1.attention.ops.triton_quant_kv.base import QuantKVSpec

_FP8_QUANT_MIN, _FP8_QUANT_MAX = get_fp8_min_max()


class Fp8PerTokenHeadFactory(PerTokenHeadFactoryBase):
    """KV cache plugin for ``fp8_per_token_head``."""

    spec = QuantKVSpec(
        name="fp8_per_token_head",
        storage_dtype=current_platform.fp8_dtype(),
        packing_factor=1,
        needs_per_token_head_scales=True,
        description="FP8 per-(token, head) symmetric absmax quantization",
    )

    _quant_max = _FP8_QUANT_MAX
    _quant_min = _FP8_QUANT_MIN


register(Fp8PerTokenHeadFactory())
