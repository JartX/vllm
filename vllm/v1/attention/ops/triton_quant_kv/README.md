# KV-cache Quantization Plugin System

Drop-in plugin architecture for KV-cache quantization modes in vLLM's
Triton attention backend.  Adding a new mode takes **one new Python
file** — no edits to enums, dispatchers, config validators, or any
other file in the repo.

- [1. What this is](#1-what-this-is)
- [2. Package layout](#2-package-layout)
- [3. The contract](#3-the-contract)
- [4. Discovery](#4-discovery)
- [5. Adding a new mode — decide the category](#5-adding-a-new-mode--decide-the-category)
- [6. Category A — byte-aligned symmetric absmax](#6-category-a--byte-aligned-symmetric-absmax)
- [7. Category C — sub-byte packed compatible with the shared kernel](#7-category-c--sub-byte-packed-compatible-with-the-shared-kernel)
  - [7.1 Where to put `@triton.jit` helpers](#71-where-to-put-tritonjit-helpers)
- [8. Category D — custom total (new reshape AND new attention kernel)](#8-category-d--custom-total-new-reshape-and-new-attention-kernel)
- [9. What you must NEVER touch](#9-what-you-must-never-touch)
- [10. Testing](#10-testing)
- [11. Using your plugin](#11-using-your-plugin)
- [12. Debugging](#12-debugging)

---

## 1. What this is

Every KV-cache quantization mode is a *plugin* that lives in its own
`.py` file.  Each plugin exposes:

- **Static metadata** (`QuantKVSpec`): public name, storage dtype,
  packing factor, whether it needs per-(token, head) scale buffers.
- **Runtime entry points**: a `reshape_and_cache` method (write path)
  and, when the mode needs a bespoke attention kernel, a
  `unified_attention` method (read path).

The loader auto-discovers every `.py` file sitting next to this
`__init__.py` and imports it once per process.  External plugins
(anywhere on disk) are discovered via the `VLLM_QUANT_KV_PATH`
environment variable.

## 2. Package layout

```
triton_quant_kv/
|-- __init__.py               registry + auto-discovery + loader
|-- base.py                   QuantKVSpec, QuantKVPlugin, QuantKVFactory shim
|
|-- _packed_core.py           shared attention kernel + _PackedFactory
|                             base (for sub-byte packed plugins)
|-- _per_token_head_core.py   shared reshape kernel + PerTokenHeadFactoryBase
|                             (for byte-aligned symmetric plugins)
|-- _hadamard.py              RHT / full Hadamard helpers
|-- _pack_unpack.py           nibble / quartet bit-packing helpers
|
|-- none.py                   metadata marker for the unquantized path
|-- fp8_per_tensor.py         metadata marker for classic FP8 per-tensor
|-- int8_per_token_head.py    INT8 plugin (uses PerTokenHeadFactoryBase)
|-- fp8_per_token_head.py     FP8 plugin (uses PerTokenHeadFactoryBase)
|-- int4_per_token_head.py    INT4 plugin (uses _PackedFactory)
|-- int2_per_token_head.py    INT2 plugin (uses _PackedFactory)
```

Files whose stem starts with `_` are private helpers and are skipped
by both the builtin scanner and the external scanner.

## 3. The contract

### `QuantKVSpec`

Immutable metadata.  Every plugin defines one.

```python
@dataclass(frozen=True)
class QuantKVSpec:
    name: str                                # public name (must match file stem)
    storage_dtype: torch.dtype | None        # cache storage dtype
    packing_factor: int = 1                  # logical values per cache byte
    needs_per_token_head_scales: bool = False
    needs_per_tensor_scale: bool = False
    is_metadata_marker: bool = False         # True for 'none' / 'fp8_per_tensor'
    description: str = ""
```

### `QuantKVPlugin`

Abstract base.  Subclasses set `spec` and implement
`reshape_and_cache`; optionally override `unified_attention` to take
over the attention read path.

```python
class QuantKVPlugin(ABC):
    spec: QuantKVSpec

    @property
    def has_bespoke_attention(self) -> bool:
        """Auto: True iff unified_attention is overridden in the subclass."""

    def packed_head_size(self, head_size: int) -> int:
        """head_size // spec.packing_factor (with divisibility assert)."""

    def allocate_scale_caches(
        self, num_blocks, block_size, num_kv_heads, device,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Default: fp32 (num_blocks, block_size, num_kv_heads) for K and V
        when spec.needs_per_token_head_scales is True, else (None, None).
        Override for a different scale layout."""

    @abstractmethod
    def reshape_and_cache(
        self, key, value, key_cache, value_cache, slot_mapping, *,
        k_scale_cache=None, v_scale_cache=None,
    ) -> None:
        """Write path — always required."""

    def unified_attention(
        self, q, k_cache, v_cache, out, *,
        cu_seqlens_q, max_seqlen_q, seqused_k, max_seqlen_k,
        softmax_scale, window_size, block_table, softcap,
        sinks, alibi_slopes, use_alibi_sqrt,
        qq_bias, output_scale, mm_prefix_range,
        k_scale_cache=None, v_scale_cache=None,
        seq_threshold_3D=None, num_par_softmax_segments=None,
        softmax_segm_output=None, softmax_segm_max=None,
        softmax_segm_expsum=None,
    ) -> None:
        """Read path — override only when your mode needs a custom kernel.
        Default raises; modes that can ride the core kernel leave this
        alone and the dispatcher falls through."""

    @property
    def has_bespoke_prefill(self) -> bool:
        """Auto: True iff unified_attention_prefill is overridden."""

    def unified_attention_prefill(
        self, q, k_cache, v_cache, out, *,
        query_start_loc, seq_lens, block_table,
        softmax_scale, num_reqs, max_query_len,
        k_scale_cache=None, v_scale_cache=None,
    ) -> None:
        """Optional flash-attention-shape prefill entry point.

        The triton backend routes continuation-chunk prefill slices
        (q_len > 1 with cached context) here when ``has_bespoke_prefill``
        is True.  Narrow contract: causal attention only, no alibi /
        sinks / softcap / sliding window / mm_prefix / qq_bias /
        output_scale.  Plugins that need any of those features leave
        this unoverridden; the slice falls back to ``unified_attention``
        which carries the full feature set at decode-tuned shape."""
```

### `register(plugin)`

Called at module load by every plugin.  First-registration wins: a
later registration under the same `spec.name` logs a warning and is
ignored.  External plugins load *before* builtins, so an external
file with the same name as a builtin transparently overrides it.

## 4. Discovery

At the first lookup of any plugin name, the loader runs both scans:

1. **External scan** — every directory in `$VLLM_QUANT_KV_PATH`
   (`:`-separated).  Each `*.py` (no `_` prefix) is imported under a
   synthetic module name.  Errors are logged and skipped.
2. **Builtin scan** — every `*.py` next to this `__init__.py`.
   `__init__.py` and `base.py` are skipped explicitly; anything
   starting with `_` is skipped.  Errors logged and skipped.

External is first so it can override a builtin via the first-wins
rule.  Both scans run once per process.

Public API:

```python
from vllm.v1.attention.ops.triton_quant_kv import (
    register,
    get_quant_kv_plugin,     # lookup by name
    get_plugin_for_dtype,    # lookup by --kv-cache-dtype string (returns None on miss)
    has_quant_kv_plugin,
    list_registered_plugins,
)
```

## 5. Adding a new mode — decide the category

```
Is your cache storage byte-aligned (1 logical value per byte)?
|
|-- YES
|   |
|   |-- Is your math symmetric absmax with a clamp interval?
|   |   |-- YES -> Category A  (inherit PerTokenHeadFactoryBase, ~40 lines)
|   |   '-- NO  -> Category B  (inherit QuantKVPlugin + custom reshape kernel,
|   |                           core attention kernel handles read)
|
'-- NO (sub-byte packed)
    |
    |-- Does your packing factor + unpack logic fit the existing
    |   _attn_packed shared kernel?
    |   |-- YES -> Category C  (inherit _PackedFactory, ~260 lines)
    |   '-- NO  -> Category D  (inherit QuantKVPlugin + custom reshape
                               AND custom attention kernel, ~900 lines)
```

Categories A, B, C have short plugin files because they reuse shared
infrastructure.  Category D writes the full stack — that's the flow
we document in full below.

## 6. Category A — byte-aligned symmetric absmax

The kernel is already in `_per_token_head_core.py`.  You only supply
the clamp range, storage dtype, and name.

Template: [`int8_per_token_head.py`](int8_per_token_head.py) /
[`fp8_per_token_head.py`](fp8_per_token_head.py).

```python
# my_mode_per_token_head.py
from __future__ import annotations
import torch

from vllm.v1.attention.ops.triton_quant_kv import register
from vllm.v1.attention.ops.triton_quant_kv._per_token_head_core import (
    PerTokenHeadFactoryBase,
)
from vllm.v1.attention.ops.triton_quant_kv.base import QuantKVSpec


class MyModePerTokenHeadFactory(PerTokenHeadFactoryBase):
    spec = QuantKVSpec(
        name="my_mode_per_token_head",   # MUST match the file stem
        storage_dtype=torch.int8,
        packing_factor=1,
        needs_per_token_head_scales=True,
        description="...",
    )
    _quant_max = 31.0    # your clamp bounds
    _quant_min = -32.0


register(MyModePerTokenHeadFactory())
```

Done.  ~40 lines total.

## 7. Category C — sub-byte packed compatible with the shared kernel

The shared kernel `_attn_packed` in `_packed_core.py` currently
dispatches on `PACKING_FACTOR == 2` (INT4: nibble unpack +
asymmetric-zp score correction) and `PACKING_FACTOR == 4` (INT2:
quartet unpack + Lloyd-Max centroid lookup).

If your mode reuses one of those exact unpack/dequant paths and only
differs in the reshape (write) math, you're in Category C.

Template: [`int4_per_token_head.py`](int4_per_token_head.py) /
[`int2_per_token_head.py`](int2_per_token_head.py).

Your file needs:

- A `@triton.jit` reshape kernel for the write path.
- A factory class inheriting from `_PackedFactory` with:
  - `spec = QuantKVSpec(...)`
  - `_reshape_kernel = <your_reshape_kernel>`
  - Static methods `_rotate_kv`, `_rotate_q`, `_unrotate_out`,
    `_transform_softmax_scale` (identity versions are fine).

~260 lines total.

**If your packing factor is new** (e.g. 8 for int1, or some exotic
layout) the shared `_attn_packed` won't know how to unpack it.
Either:

1. Add an `elif PACKING_FACTOR == N:` branch to `_attn_packed` in
   `_packed_core.py` (you're editing shared code — cheap if the
   delta is small), OR
2. Jump to Category D (write your own attention kernel).

### 7.1 Where to put `@triton.jit` helpers

When you add a new centroid / dequant helper for your Category C
plugin, ask which kernel actually calls it:

| Helper is called by...         | Where it lives                    |
|---|---|
| **Shared** `_attn_packed` (read path) | `_packed_core.py`           |
| Only your **plugin's** reshape kernel (write path) | Your plugin `.py` file |

Triton's `@triton.jit` resolves referenced names from the **module
globals where the calling kernel is defined**.  `_attn_packed` lives
in `_packed_core.py`, so any helper it references — for example the
`PACKING_FACTOR == 4` branch dequant — must be importable as a
global in `_packed_core.py` itself.  Putting that helper in the
plugin file will blow up at launch with:

```
NameError('<helper> is not defined')
```

Rule of thumb: if `grep -n <helper_name> _packed_core.py` returns a
hit, the helper belongs in `_packed_core.py`.  Write-path helpers
(the ones only your reshape kernel calls) stay local to the plugin
so the plugin file remains the single source of truth for that
mode's encode math.

Concrete split, using INT2 as the worked example:

- `_lloyd_max_dequant_4` (read-path centroid lookup, used by the
  shared kernel) → lives in `_packed_core.py`.
- `_lloyd_max_quantize_4` (write-path quantizer, used only by the
  INT2 reshape kernel) → lives in `int2_per_token_head.py`.

## 8. Category D — custom total (new reshape AND new attention kernel)

This is the end-to-end flow for a mode whose attention kernel must be
written from scratch — for example, a new packing layout or a
fundamentally different quantization scheme.  The resulting file is
~800-1000 lines and requires roughly 2-3 days of engineering for an
author comfortable with Triton.

### Step 1 — Decide the layout (30 min to 2h)

On paper, decide:

| Question | Answer example |
|---|---|
| Storage dtype | `torch.uint8` |
| Packing factor | 8 (8 × int1 per byte) |
| Quantization math | Lloyd-Max 2 centroids |
| Rotation, if any | Full Hadamard |
| Scale layout | 1 × float32 per (token, head) for K and V |
| Scale buffer allocation | default from `QuantKVPlugin.allocate_scale_caches` |

### Step 2 — Create the file and spec (10 min)

```bash
touch vllm/v1/attention/ops/triton_quant_kv/my_mode_per_token_head.py
```

```python
# SPDX-License-Identifier: Apache-2.0
"""MY_MODE per-token-head KV cache quantization plugin."""

from __future__ import annotations
from typing import Any

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

# Reusable attention-loop helpers.  Use the ones you need.
from vllm.v1.attention.ops.triton_attention_helpers import (
    apply_alibi_to_score, apply_softcap, cdiv_fn,
    compute_kv_seq_mask, compute_tile_loop_bounds,
    init_softmax_M, load_qq_bias_tile,
    resolve_seq_and_query_len, softmax_step,
    store_segm_reduce_scalars,
)
from vllm.v1.attention.ops.triton_unified_attention import reduce_segments
from vllm.v1.attention.ops.triton_quant_kv._hadamard import fast_hadamard_transform
from vllm.v1.attention.ops.triton_quant_kv import register
from vllm.v1.attention.ops.triton_quant_kv.base import QuantKVPlugin, QuantKVSpec

float8_info = torch.finfo(current_platform.fp8_dtype())
```

### Step 3 — Write the reshape kernel (2-4h)

Reference: the `@triton.jit` reshape kernel in any of the packed
plugin files.

```python
@triton.jit
def _reshape_cache_my_mode_kernel(
    # pointers
    key_ptr, value_ptr, key_cache_ptr, value_cache_ptr,
    k_scale_cache_ptr, v_scale_cache_ptr,
    slot_mapping_ptr,
    # strides (caller passes key.stride(0), etc.)
    stride_key_tok: tl.int64, stride_key_head: tl.int64,
    stride_val_tok: tl.int64, stride_val_head: tl.int64,
    stride_kc_blk: tl.int64, stride_kc_slot: tl.int64, stride_kc_head: tl.int64,
    stride_vc_blk: tl.int64, stride_vc_slot: tl.int64, stride_vc_head: tl.int64,
    stride_ks_blk: tl.int64, stride_ks_slot: tl.int64, stride_ks_head: tl.int64,
    stride_vs_blk: tl.int64, stride_vs_slot: tl.int64, stride_vs_head: tl.int64,
    # shapes
    block_size: tl.constexpr,
    head_size: tl.constexpr,
    head_size_v: tl.constexpr,
    PACKED_HEAD_PADDED: tl.constexpr,   # head_size // packing_factor, pow-2 padded
):
    """Grid: (num_tokens, num_kv_heads).

    Each program_id processes one (token, head):
      1. Load the head in fp32
      2. Compute scale (norm / absmax / whatever)
      3. Quantize to N-bit indices
      4. Pack and store to cache
      5. Store the scale
    """
    tok = tl.program_id(0)
    head = tl.program_id(1)

    slot = tl.load(slot_mapping_ptr + tok).to(tl.int64)
    if slot < 0:
        return

    blk = slot // block_size
    slot_in_blk = slot % block_size

    # ... your math ...
```

And the Python-side launcher:

```python
def _launch_reshape_my_mode(
    key, value, key_cache, value_cache,
    k_scale_cache, v_scale_cache, slot_mapping,
    packing_factor,
):
    num_tokens, num_kv_heads, head_size = key.shape
    head_size_v = value.shape[2]
    block_size = key_cache.shape[1]
    packed_head = triton.next_power_of_2(head_size // packing_factor)

    num_warps = (
        4 if (current_platform.is_rocm() or current_platform.is_xpu())
        else min(16, max(1, packed_head // 32))
    )

    _reshape_cache_my_mode_kernel[(num_tokens, num_kv_heads)](
        # pass every kwarg by name — Triton is unforgiving with positional
        key_ptr=key, value_ptr=value,
        key_cache_ptr=key_cache, value_cache_ptr=value_cache,
        k_scale_cache_ptr=k_scale_cache, v_scale_cache_ptr=v_scale_cache,
        slot_mapping_ptr=slot_mapping,
        stride_key_tok=key.stride(0), stride_key_head=key.stride(1),
        stride_val_tok=value.stride(0), stride_val_head=value.stride(1),
        stride_kc_blk=key_cache.stride(0), stride_kc_slot=key_cache.stride(1),
        stride_kc_head=key_cache.stride(2),
        stride_vc_blk=value_cache.stride(0), stride_vc_slot=value_cache.stride(1),
        stride_vc_head=value_cache.stride(2),
        stride_ks_blk=k_scale_cache.stride(0), stride_ks_slot=k_scale_cache.stride(1),
        stride_ks_head=k_scale_cache.stride(2),
        stride_vs_blk=v_scale_cache.stride(0), stride_vs_slot=v_scale_cache.stride(1),
        stride_vs_head=v_scale_cache.stride(2),
        block_size=block_size,
        head_size=head_size, head_size_v=head_size_v,
        PACKED_HEAD_PADDED=packed_head,
        num_warps=num_warps,
    )
```

### Step 4 — Write the attention kernel (1-2 days)

Reference: `_packed_core.py::_attn_packed` (~450 lines).  The
skeleton is the same for any paged-attention kernel — the parts you
change are the unpack, dequant, and score correction inside the tile
loop.

```python
@triton.jit
def _attn_my_mode(
    # outputs (2D direct / 3D via segm buffers)
    output_ptr, segm_output_ptr, segm_max_ptr, segm_expsum_ptr,
    query_ptr, key_cache_ptr, value_cache_ptr,
    sink_ptr, block_tables_ptr, seq_lens_ptr,
    alibi_slopes_ptr, qq_bias_ptr, scale,
    k_scale_cache_ptr, v_scale_cache_ptr,
    # ... strides, problem shape ...
    BLOCK_M: tl.constexpr, BLOCK_Q: tl.constexpr,
    HEAD_SIZE: tl.constexpr, HEAD_SIZE_PADDED: tl.constexpr,
    PACKED_HEAD_SIZE: tl.constexpr,
    USE_ALIBI_SLOPES: tl.constexpr, USE_ALIBI_SQRT: tl.constexpr,
    USE_SOFTCAP: tl.constexpr, USE_QQ_BIAS: tl.constexpr,
    MAX_MM_RANGES: tl.constexpr, USE_MM_PREFIX: tl.constexpr,
    USE_SINKS: tl.constexpr, USE_WINDOW: tl.constexpr,
    IS_3D: tl.constexpr, PARA_SOFTMAX_SEGMENTS: tl.constexpr,
    PACKING_FACTOR: tl.constexpr,
):
    """Paged attention for MY_MODE.  Same structure as _attn_packed:
      1. resolve_seq_and_query_len
      2. init_softmax_M(BLOCK_M)
      3. Load Q tile, rotate if needed
      4. For each KV tile in the loop bounds:
         a. Load packed K, unpack (YOUR unpack), dequant (YOUR math)
         b. Score = Q @ K + softmax_scale + mask/alibi/softcap/qq_bias
         c. softmax_step -> P
         d. Load packed V, unpack, dequant, P @ V -> acc
      5. Epilogue:
         - 3D: store per-segment partials
         - 2D: write output directly
    """
    # ... implementation ...
```

Launcher:

```python
def _launch_attn_my_mode(
    *, q, k_cache, v_cache, out,
    cu_seqlens_q, max_seqlen_q, seqused_k,
    softmax_scale, window_size, block_table, softcap,
    sinks, alibi_slopes, use_alibi_sqrt,
    qq_bias, output_scale, mm_prefix_range,
    k_scale_cache, v_scale_cache,
    seq_threshold_3D, num_par_softmax_segments,
    softmax_segm_output, softmax_segm_max, softmax_segm_expsum,
    packing_factor,
):
    """Copy of _launch_packed_attn's scaffolding, swapping the jit
    function.  Computes tile sizes + num_warps, decides 2D-vs-3D
    decode dispatch, launches the kernel, and runs reduce_segments
    after the 3D path.
    """
    # ... direct copy from _packed_core._launch_packed_attn,
    # replacing _attn_packed with _attn_my_mode ...
```

### Step 5 — Factory class

The `reshape_and_cache` and `unified_attention` signatures are
**fixed** by `QuantKVPlugin`.  Do not add or remove parameters.  Pass
mode-specific parameters through `self.spec` or class attributes.

```python
class MyModePerTokenHeadFactory(QuantKVPlugin):
    spec = QuantKVSpec(
        name="my_mode_per_token_head",   # MUST match the file stem
        storage_dtype=torch.uint8,
        packing_factor=8,
        needs_per_token_head_scales=True,
        description="MY_MODE: ...",
    )

    # Optional: override if your scale layout is not the default
    # (num_blocks, block_size, num_kv_heads) fp32 per K and V.
    # def allocate_scale_caches(self, num_blocks, block_size, num_kv_heads, device):
    #     ...

    def reshape_and_cache(
        self, key, value, key_cache, value_cache, slot_mapping, *,
        k_scale_cache=None, v_scale_cache=None,
    ) -> None:
        assert k_scale_cache is not None and v_scale_cache is not None, (
            f"{self.spec.name!r} requires k_scale_cache / v_scale_cache"
        )
        # Optional pre-rotation
        key = fast_hadamard_transform(key.float()).to(key.dtype)
        value = fast_hadamard_transform(value.float()).to(value.dtype)
        _launch_reshape_my_mode(
            key, value, key_cache, value_cache,
            k_scale_cache, v_scale_cache, slot_mapping,
            packing_factor=self.spec.packing_factor,
        )

    def unified_attention(
        self, q, k_cache, v_cache, out, *,
        cu_seqlens_q, max_seqlen_q, seqused_k, max_seqlen_k,
        softmax_scale, window_size, block_table, softcap,
        sinks, alibi_slopes, use_alibi_sqrt,
        qq_bias, output_scale, mm_prefix_range,
        k_scale_cache=None, v_scale_cache=None,
        seq_threshold_3D=None, num_par_softmax_segments=None,
        softmax_segm_output=None, softmax_segm_max=None,
        softmax_segm_expsum=None,
    ) -> None:
        q_orig = q.dtype
        q = fast_hadamard_transform(q.float()).to(q_orig)

        _launch_attn_my_mode(
            q=q, k_cache=k_cache, v_cache=v_cache, out=out,
            cu_seqlens_q=cu_seqlens_q, max_seqlen_q=max_seqlen_q,
            seqused_k=seqused_k,
            softmax_scale=softmax_scale,
            window_size=window_size, block_table=block_table, softcap=softcap,
            sinks=sinks, alibi_slopes=alibi_slopes,
            use_alibi_sqrt=use_alibi_sqrt,
            qq_bias=qq_bias, output_scale=output_scale,
            mm_prefix_range=mm_prefix_range,
            k_scale_cache=k_scale_cache, v_scale_cache=v_scale_cache,
            seq_threshold_3D=seq_threshold_3D,
            num_par_softmax_segments=num_par_softmax_segments,
            softmax_segm_output=softmax_segm_output,
            softmax_segm_max=softmax_segm_max,
            softmax_segm_expsum=softmax_segm_expsum,
            packing_factor=self.spec.packing_factor,
        )

        # Inverse rotation (if your rotation is not self-inverse, use the
        # correct inverse here).
        out_f = fast_hadamard_transform(out.float())
        out.copy_(out_f.to(q_orig))


register(MyModePerTokenHeadFactory())
```

Because `unified_attention` is overridden, `has_bespoke_attention`
becomes True automatically and the dispatcher routes attention calls
to your plugin.

### Step 6 — Smoke test before running vLLM

```python
from vllm.v1.attention.ops.triton_quant_kv import get_plugin_for_dtype

p = get_plugin_for_dtype("my_mode_per_token_head")
assert p is not None
print(p.spec)
print("has_bespoke_attention =", p.has_bespoke_attention)   # expect True
```

If this fails, either the file has a Python syntax error, an import
is broken, or `register()` was never called at module level.

## 9. What you must NEVER touch

Adding a new plugin requires **zero** edits to any of these files:

- `vllm/v1/kv_cache_interface.py` — the `KVQuantMode` enum.  New
  modes with a bespoke attention kernel don't need an enum value;
  the dispatcher uses the string.
- `vllm/v1/attention/ops/triton_quant_kv/__init__.py` — the
  registry.  Auto-discovery finds your file.
- `vllm/config/cache.py` — the `CacheDType` literal.  The field
  validator accepts any name the registry knows.  The literal is
  kept for autocomplete only and is optional to update.
- `vllm/utils/torch_utils.py` — the `STR_DTYPE_TO_TORCH_DTYPE`
  mapping.  Your `spec.storage_dtype` is the source of truth.
- `vllm/v1/attention/ops/triton_unified_attention.py` and
  `vllm/v1/attention/ops/triton_reshape_and_cache_flash.py` — the
  dispatchers.  Plugin-first routing already handles your mode.
- `vllm/v1/attention/backends/triton_attn.py` — the backend.  It
  propagates `self.kv_cache_dtype` through automatically.

Touching any of these to add a plugin is a sign that something in
your plugin file is wrong — re-check the spec name, the file stem,
and the `register()` call.

## 10. Testing

Copy the pattern in
[`tests/quantization/test_per_token_kv_cache.py`](/tests/quantization/test_per_token_kv_cache.py):

```python
MY_MODE_CONFIG = QuantConfig(
    cache_dtype=torch.uint8,
    kv_cache_dtype_str="my_mode_per_token_head",
    quant_max=...,
    quant_min=...,
    kv_quant_mode=None,     # plugin-only; not in the enum
    uses_trunc=False,
)
QUANT_CONFIGS = [INT2_CONFIG, INT4_CONFIG, MY_MODE_CONFIG, INT8_CONFIG, FP8_CONFIG]
```

The parametric fixture picks it up automatically.  Where the
existing tests compare to a PyTorch reference (INT8 / FP8), add
a branch so your mode is either tolerated with a wider `rt_atol` or
excluded from the comparison:

```python
is_my_mode = qcfg.kv_cache_dtype_str == "my_mode_per_token_head"
if is_my_mode:
    rt_atol = 1.0   # whatever your math supports
```

Discovery-only tests (no GPU, no kernels) live in
[`tests/quantization/test_kv_quant_plugin_discovery.py`](/tests/quantization/test_kv_quant_plugin_discovery.py).

## 11. Using your plugin

### In-tree (committed to the repo)

```bash
vllm serve <model> --kv-cache-dtype my_mode_per_token_head
```

### Out-of-tree (user plugin in any directory)

1. Drop the file anywhere:
   ```bash
   mkdir -p /opt/my_vllm_plugins
   cp my_mode_per_token_head.py /opt/my_vllm_plugins/
   ```
2. Point the loader at the directory:
   ```bash
   export VLLM_QUANT_KV_PATH=/opt/my_vllm_plugins
   ```
   Multiple directories are separated by `os.pathsep` (`:` on Linux
   and macOS, `;` on Windows).
3. Select at runtime:
   ```bash
   vllm serve <model> --kv-cache-dtype my_mode_per_token_head
   ```

No edits to vLLM sources are needed for external plugins.

## 12. Debugging

### Plugin not found

```python
KeyError: No KV-quant plugin named 'my_mode_per_token_head'.
Registered: [...].  External plugins are loaded from directories
listed in VLLM_QUANT_KV_PATH.
```

Checklist:

1. `spec.name` matches the file stem exactly (`my_mode_per_token_head`
   in both).
2. The file calls `register(MyModePerTokenHeadFactory())` at module
   level (not inside a function).
3. The file is in the correct directory
   (`triton_quant_kv/` for builtins, or on `$VLLM_QUANT_KV_PATH`).
4. The file stem does not start with `_` (those are skipped).

### Plugin fails to import

Look for a line like this in the logs:

```
ERROR ... Failed to load builtin KV-quant plugin <path>: <reason>
```

The reason is a Python exception from `exec_module`: usually a
`NameError`, `ImportError`, or `SyntaxError`.  Run
`.venv/bin/python -m py_compile <path>` on the plugin file to get a
clean syntax-check first, then import it in a REPL to see the full
traceback.

### Duplicate registration warnings

```
INFO ... KV-quant plugin 'my_mode_per_token_head' already
registered as X; ignoring new registration of Y
(first-registration-wins; external plugins override builtins when
loaded first).
```

Two common causes:

1. Stale files in the installed vLLM directory (old `.py` files that
   still register under the same name).  Clean up the installed
   path or reinstall with `--force-reinstall`.
2. Two plugin files claim the same `spec.name`.  Decide which wins,
   rename the other.
