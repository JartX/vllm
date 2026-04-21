# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP8 per-token-head KV cache quantization plugin.

Drives the symmetric-absmax write path shared with INT8 per-token-head
(see :mod:`_per_token_head_core`) with the platform-appropriate FP8
clamp range.  The attention read path lives in the core kernel, gated
by the ``USE_PER_TOKEN_HEAD_SCALES`` constexpr branch.
"""

from __future__ import annotations

from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_fp8_min_max,
)
from vllm.v1.attention.ops.triton_quant_kv import register
from vllm.v1.attention.ops.triton_quant_kv._per_token_head_core import (
    PerTokenHeadFactoryBase,
)
from vllm.v1.kv_cache_interface import KVQuantMode

_FP8_QUANT_MIN, _FP8_QUANT_MAX = get_fp8_min_max()


class Fp8PerTokenHeadFactory(PerTokenHeadFactoryBase):
    """KV cache factory for ``KVQuantMode.FP8_PER_TOKEN_HEAD``."""

    mode = KVQuantMode.FP8_PER_TOKEN_HEAD
    _quant_max = _FP8_QUANT_MAX
    _quant_min = _FP8_QUANT_MIN


register(Fp8PerTokenHeadFactory())
