# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""INT8 per-token-head KV cache quantization plugin.

Thin subclass of :class:`PerTokenHeadFactoryBase` (in
:mod:`_per_token_head_core`) that pins the symmetric clamp range to
the INT8 interval.  The write path is shared with every other
byte-aligned per-token-head plugin; the attention read path lives in
the core kernel via the ``USE_PER_TOKEN_HEAD_SCALES`` constexpr
branch.

Adding a new byte-aligned symmetric per-token-head mode = copy this
file, change the name + clamp range + storage dtype.  No other file
needs editing.
"""

from __future__ import annotations

import torch

from vllm.v1.attention.ops.triton_quant_kv import register
from vllm.v1.attention.ops.triton_quant_kv._per_token_head_core import (
    PerTokenHeadFactoryBase,
)
from vllm.v1.attention.ops.triton_quant_kv.base import QuantKVSpec


class Int8PerTokenHeadFactory(PerTokenHeadFactoryBase):
    """KV cache plugin for ``int8_per_token_head``."""

    spec = QuantKVSpec(
        name="int8_per_token_head",
        storage_dtype=torch.int8,
        packing_factor=1,
        needs_per_token_head_scales=True,
        description="INT8 per-(token, head) symmetric absmax quantization",
    )

    _quant_max = 127.0
    _quant_min = -128.0


register(Int8PerTokenHeadFactory())
