# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""INT8 per-token-head KV cache quantization plugin.

Drives the symmetric-absmax write path shared with FP8 per-token-head
(see :mod:`_per_token_head_core`) with the INT8 clamp range.  The
attention read path lives in the core kernel, gated by the
``USE_PER_TOKEN_HEAD_SCALES`` constexpr branch.
"""

from __future__ import annotations

from vllm.v1.attention.ops.triton_quant_kv import register
from vllm.v1.attention.ops.triton_quant_kv._per_token_head_core import (
    PerTokenHeadFactoryBase,
)
from vllm.v1.kv_cache_interface import KVQuantMode

_INT8_QUANT_MAX = 127.0
_INT8_QUANT_MIN = -128.0


class Int8PerTokenHeadFactory(PerTokenHeadFactoryBase):
    """KV cache factory for ``KVQuantMode.INT8_PER_TOKEN_HEAD``."""

    mode = KVQuantMode.INT8_PER_TOKEN_HEAD
    _quant_max = _INT8_QUANT_MAX
    _quant_min = _INT8_QUANT_MIN


register(Int8PerTokenHeadFactory())
