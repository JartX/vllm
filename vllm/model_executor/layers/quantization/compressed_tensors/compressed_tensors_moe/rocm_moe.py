# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm MoE kernel dispatcher.

Selects architecture-specific native HIP MoE kernels in priority order.
Returns None to fall back to the Triton WNA16 path.
"""

from vllm.platforms import current_platform


def try_make(weight_quant, input_quant, moe_config):
    """Return a native ROCm MoE method if available, else None."""
    if not current_platform.is_rocm():
        return None

    # RDNA3 (gfx1100)
    from .compressed_tensors_moe_wna16_rdna3 import (
        try_make as _try_rdna3,
    )

    result = _try_rdna3(weight_quant, input_quant, moe_config)
    if result is not None:
        return result

    # Future: RDNA4 (gfx12x), CDNA (gfx94x), etc.

    return None
