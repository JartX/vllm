# KV-Quant Architecture and Flow

Companion to [README.md](README.md).  The README is the reference for
plugin internals (spec, registry, discovery, testing).  This document
describes the **end-to-end flow** of a forward pass and summarizes
**how to add a new plugin** at a higher level, so you can see how all
the pieces connect before diving into implementation details.

- [1. High-level picture](#1-high-level-picture)
- [2. The forward-pass flow](#2-the-forward-pass-flow)
- [3. Cache layout and strided views](#3-cache-layout-and-strided-views)
- [4. Fast paths in the backend](#4-fast-paths-in-the-backend)
- [5. The plugin contract (3 hooks)](#5-the-plugin-contract-3-hooks)
- [6. Adding a new plugin — walkthrough](#6-adding-a-new-plugin--walkthrough)
- [7. Which path did my request take?](#7-which-path-did-my-request-take)

---

## 1. High-level picture

A KV-quant mode owns two operations on the paged KV cache:

- **Write path** (`reshape_and_cache`): convert raw `K`/`V` from the
  model's projection into the mode's storage format (packed bytes,
  int8 values, fp8 values, …) and place them at the right slot in the
  paged cache, along with any per-(token, head) scales.
- **Read path** (`unified_attention`, optionally
  `unified_attention_prefill`): run paged attention by reading those
  stored bytes + scales and producing the attention output.

Both paths must agree on the storage format — scales location, packing
layout, dtype — so a plugin bundles them together.  Everything else
(the scheduler, the attention `nn.Module`, the compile pipeline, the
CUDA graph capture) is mode-agnostic.

The backend ([`triton_attn.py`](../../backends/triton_attn.py)) sits
between the framework-level attention op and the plugin:

```
Layer (attention.py)
        │   unified_attention_with_output op
        ▼
TritonAttentionImpl.forward
        │   builds / reads paged KV cache
        │   picks a specialized kernel based on batch composition
        ▼
┌──────────────────────────────────────────────────────────────┐
│  Fast paths in triton_attn.forward (per-token-head modes):   │
│                                                              │
│  FP1  pure first prefill     → context_attention_fwd         │
│  FP2  decode + first-chunk   → unified_attention + ctx_fwd   │
│  FP3  pure continuation      → *_prefill kernel              │
│  FP4  mixed continuation     → unified_attention + *_prefill │
│  split-KV decode (q≤128)     → triton_per_token_head_*       │
└──────────────────────────────────────────────────────────────┘
        │   fall-through when no fast path matches
        ▼
unified_attention (triton_unified_attention.py)
        │   plugin-first dispatch by kv_cache_dtype string
        │   (has_bespoke_attention → plugin kernel; else core kernel)
        ▼
plugin.unified_attention  /  kernel_unified_attention
        │
        ▼
  Triton-compiled kernel on GPU
```

---

## 2. The forward-pass flow

Follow a token's journey through one attention layer end-to-end.

### Stage 1 — Metadata builder (once per batch)

[`TritonAttentionMetadataBuilder.build`](../../backends/triton_attn.py)
runs on the model-runner side before `forward`.  For per-token-head
modes it:

1. Reads `seq_lens_cpu` and `query_start_loc_cpu` (already on CPU — no
   extra D2H sync).
2. Computes fast-path predicates:
   - `all_pure_first_prefill`: every request has `q_len == seq_len`
     (no cached context).
   - `num_decodes`: leading contiguous run of requests with `q_len ≤ 1`.
   - `prefill_is_first_chunk`: the non-decode tail all have
     `q_len == seq_len`.
3. Builds per-query maps (`q_to_req`, `q_to_klen`) on CPU and stages
   them into pre-allocated GPU buffers via `.copy_(non_blocking=True)`.
   **Pointers stay stable across CUDA graph capture/replay** — graphs
   reference the buffer, only its contents change each iteration.

The predicates and maps land on the `TritonAttentionMetadata` dataclass
consumed by `forward`.

### Stage 2 — The layer calls the op

The attention `nn.Module` invokes
`torch.ops.vllm.unified_attention_with_output`, which calls
`self.impl.forward(query, key, value, kv_cache, attn_metadata, output)`
on a `TritonAttentionImpl`.

### Stage 3 — `forward` picks a path

The backend tries fast paths in order.  They are pre-gated on
"clean" attention shape (no alibi, no sinks, no softcap, no
`mm_prefix_range`, no `output_scale`, no kv-sharing, no sliding window
for the prefill kernels).  If any feature is on, we fall through to
the generic `unified_attention`.

Fast-path matrix (for per-token-head modes):

| FP | When | What runs |
|---|---|---|
| FP1 | `num_decodes == 0 && all_pure_first_prefill` | `context_attention_fwd(q, k_raw, v_raw)` — attends over the raw K/V from the model projection.  Cache is not read. |
| FP2 | `num_decodes > 0 && prefill_is_first_chunk && num_dec < total` | Decode slice through `unified_attention` (plugin-aware); prefill slice through `context_attention_fwd` with raw K/V. |
| FP3 | `num_decodes == 0 && some_requests_have_cached_context` | Flash-shape prefill kernel reading the paged cache with inline dequant: `triton_per_token_head_prefill` (int8/fp8) or `_attn_packed_prefill` (int4/int2 via `plugin.unified_attention_prefill`). |
| FP4 | Mixed decode + continuation prefill | Decode slice through `unified_attention`; prefill slice through the same prefill kernel as FP3. |
| split-KV | `max_query_len ≤ 128`, per-token-head, byte-aligned (int8/fp8) | `triton_per_token_head_attention` — split-KV stage1 + stage2 with per-query causal `k_len`. |

If none fires, the backend calls `unified_attention(...)` which is the
dispatcher.

### Stage 4 — Dispatcher

[`unified_attention`](../ops/triton_unified_attention.py) does
plugin-first dispatch:

1. **By `kv_cache_dtype` string** — preferred.  Looks up the plugin
   registry; if the plugin has `has_bespoke_attention`, routes to
   `plugin.unified_attention(...)`.  Covers external plugins whose
   names aren't in `KVQuantMode`.
2. **By `kv_quant_mode` enum** — legacy fallback for call sites that
   haven't threaded the string through.  Skipped when the mode is
   `NONE` (contract: `has_quant_kv_factory(NONE) == False`).
3. **Fall-through** — the mode is one of NONE / FP8_PER_TENSOR /
   INT8_PER_TOKEN_HEAD / FP8_PER_TOKEN_HEAD, all handled by the core
   `kernel_unified_attention` via its `KV_QUANT_MODE` constexpr
   branches.

### Stage 5 — Kernel executes

Either the plugin's bespoke Triton kernel or the shared
`kernel_unified_attention`.  Both consume the same arguments (paged
cache strides, scale cache strides, block table, query layout) —
what differs is the per-tile math (unpack / dequant / score
correction).

---

## 3. Cache layout and strided views

Per-token-head modes use a **fused paged-cache layout** allocated by
the backend's `get_kv_cache_shape`:

```
shape = (num_blocks, 1, block_size, num_kv_heads, 2 * (head_bytes + scale_pad))

per-slot bytes = [ K_data | K_scale | V_data | V_scale ]
                 ^ hs      ^ sp      ^ hs     ^ sp
```

where

- `hs = head_size // packing_factor` (bytes of data per head).
- `sp = sizeof(float32) // sizeof(cache_dtype)` (padding needed to
  fit one fp32 scale inline).
- The size-1 dim keeps the tensor 5-D for `stride_order` compat with
  backends that assume a 5-D shape.

`TritonAttentionImpl._ensure_fused_cache_views` carves **four
zero-copy strided views** out of this one tensor:

- `key_cache_view`: `(num_blocks, block_size, nkv, hs)` at offset `0`.
- `value_cache_view`: same shape, offset `padded_hs = hs + sp`.
- `k_scale_cache`: fp32 view `(num_blocks, block_size, nkv)` at byte
  offset `hs * dtype_sz`, reinterpreting the scale-pad region as
  `float32`.
- `v_scale_cache`: fp32 view, offset `(padded_hs + hs) * dtype_sz`.

Existing Triton kernels consume these via `.stride()` — no kernel
code changes are needed for the new layout.  Spatial locality is the
win: per-slot accesses to (K, K_scale, V, V_scale) land in **one
cacheline** instead of two HBM regions ~16 KB apart.

Non-per-token-head modes (`auto`, fp8 per-tensor) keep the original
`(num_blocks, 2, block_size, nkv, head_size)` layout — the backend
branches on the mode.

---

## 4. Fast paths in the backend

Two orthogonal gates:

```python
is_per_token_head = plugin.spec.needs_per_token_head_scales  # any PTH mode
is_byte_aligned_pth = kv_quant_mode in (INT8_PER_TOKEN_HEAD,
                                         FP8_PER_TOKEN_HEAD)
has_prefill_kernel = is_byte_aligned_pth or plugin.has_bespoke_prefill
```

- **FP1 and FP2** gate on `is_per_token_head`.  They don't dequant
  from the cache — FP1 skips it, FP2 delegates decode to the plugin
  via `unified_attention`.  Packed plugins (int4/int2) benefit
  transparently.

- **FP3 and FP4** gate on `has_prefill_kernel`.  Byte-aligned modes
  use `triton_per_token_head_prefill`; packed modes that implement
  `unified_attention_prefill` use the plugin's kernel
  (in-tree int4/int2 via `_attn_packed_prefill`).  Plugins without a
  prefill override fall through to the decode-shaped
  `unified_attention`.

- **split-KV decode** gates on `is_byte_aligned_pth` — the kernel
  does byte-aligned dequant and has no unpack logic for packed bytes.

The exact condition set for each FP lives in
[`triton_attn.py::TritonAttentionImpl.forward`](../../backends/triton_attn.py);
read it there when in doubt.

### Why the FP1/FP2 split matters

A pure prefill of 2 K tokens for a 32-layer MHA model avoids, per
layer, `~2 × 2048 × Hk × head_size` bytes of HBM reads from the just-
written paged cache by attending over raw K/V in registers.  That's
on the order of 100 MB of HBM traffic skipped across all layers on a
single forward — the biggest TTFT win in this stack.

---

## 5. The plugin contract (3 hooks)

Defined in [`base.py::QuantKVPlugin`](base.py).

### 5.1 `reshape_and_cache` — required

```python
def reshape_and_cache(
    self,
    key: torch.Tensor,        # [num_tokens, num_kv_heads, head_size]
    value: torch.Tensor,      # [num_tokens, num_kv_heads, head_size_v]
    key_cache: torch.Tensor,  # strided view from _ensure_fused_cache_views
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    *,
    k_scale_cache: torch.Tensor | None = None,
    v_scale_cache: torch.Tensor | None = None,
) -> None:
```

Writes one token per `slot_mapping[i]`.  Per-token-head modes compute
a scale per (token, head) and write it to `k_scale_cache`/
`v_scale_cache` alongside the data.

### 5.2 `unified_attention` — bespoke read path (optional for core modes)

```python
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
```

Full-feature paged attention for this mode.  Override when the core
`kernel_unified_attention` cannot handle your layout (packed bytes,
centroid lookups, exotic quantization).  The
`has_bespoke_attention` property auto-detects the override; the
dispatcher checks it.

### 5.3 `unified_attention_prefill` — optional flash-shape prefill

```python
def unified_attention_prefill(
    self, q, k_cache, v_cache, out, *,
    query_start_loc, seq_lens, block_table,
    softmax_scale, num_reqs, max_query_len,
    k_scale_cache=None, v_scale_cache=None,
) -> None:
```

Narrower contract: causal attention only, no alibi / sinks / softcap
/ sliding window / mm_prefix / qq_bias / output_scale.  Override only
if you can implement a flash-attention-shape kernel that beats your
decode-shape `unified_attention` for long prefill chunks with cached
context.

The backend calls this from the FP3/FP4 fast paths when the plugin
reports `has_bespoke_prefill == True`.  The property auto-detects the
override — same pattern as `has_bespoke_attention`.

---

## 6. Adding a new plugin — walkthrough

See [README.md § 5](README.md#5-adding-a-new-mode--decide-the-category)
for the full decision tree.  Summary:

| Your mode is… | Category | Template | Plugin file size |
|---|---|---|---|
| Byte-aligned + symmetric absmax (int8, fp8) | **A** | [`int8_per_token_head.py`](int8_per_token_head.py) | ~40 lines |
| Byte-aligned + custom quant math | **B** | (write reshape kernel + inherit `QuantKVPlugin`) | ~150 lines |
| Sub-byte packed (factor 2 or 4), compatible with shared `_attn_packed` | **C** | [`int4_per_token_head.py`](int4_per_token_head.py) | ~260 lines |
| Sub-byte packed with unique attention math | **D** | (write everything) | ~800-1000 lines |

### The one-screen recipe (Category A)

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
        description="my mode — 40-line description",
    )
    _quant_max = 31.0
    _quant_min = -32.0


register(MyModePerTokenHeadFactory())
```

Drop this file into `vllm/v1/attention/ops/triton_quant_kv/` (builtin)
or anywhere on `$VLLM_QUANT_KV_PATH` (external).  Use it:

```bash
vllm serve <model> --kv-cache-dtype my_mode_per_token_head
```

### What you do NOT touch

- `KVQuantMode` enum.  Plugin-only modes have no enum entry; the
  dispatcher keys off `kv_cache_dtype` strings.
- Backend or dispatcher wiring.  `register()` + discovery does the
  plumbing.
- Scheduler, compile pipeline, CUDA graph capture.  All mode-agnostic.

See [README.md § 9](README.md#9-what-you-must-never-touch).

### Opting into the prefill fast path (optional)

If your mode has a flash-attention-shape kernel that can outperform
its decode-shape `unified_attention` for long prefill with cached
context, override `unified_attention_prefill` in your plugin class.
The `has_bespoke_prefill` property flips to True automatically and
the backend routes FP3/FP4 prefill slices through it.

Reference implementation: [`_packed_core.py`](_packed_core.py)'s
`_PackedFactory.unified_attention_prefill`, which launches the
`_attn_packed_prefill` kernel.  Contract scope is intentionally
narrow (no alibi/sinks/softcap/…) — the backend gates on those
features being unset.

---

## 7. Which path did my request took?

### Eyeball test

Set up a dummy plugin or add logging at key forks.  Suggested probe
points in order of resolution:

1. **Fast path entry**: log at the top of each FP1-FP4 gate in
   `TritonAttentionImpl.forward` with the batch composition
   (`num_decodes`, `num_dec_tokens`, `num_actual_tokens`,
   `max_query_len`, `prefill_is_first_chunk`,
   `all_pure_first_prefill`).
2. **Plugin dispatch**: log in `unified_attention` whether `factory`
   was resolved and whether it had `has_bespoke_attention`.
3. **Which kernel**: the launcher functions (`triton_per_token_head_*`,
   `_launch_packed_attn`, `launch_packed_prefill`) are the final
   hop — a print there tells you definitively.

### When it didn't take the fast path you expected

Usually one of:

- `attn_metadata.seq_lens_cpu is None` — the scheduler didn't
  materialize the CPU copy this iteration, so
  `TritonAttentionMetadataBuilder.build` skipped the predicates and
  left all fast-path flags at their defaults (False / 0).
- A "clean" check failed — alibi, sinks, softcap,
  `mm_prefix_range_tensor`, `output_scale`, `kv_sharing_target_layer_name`,
  or sliding window is active.  All fast paths require these to be
  unset.
- `is_byte_aligned_pth = False` and the plugin doesn't advertise
  `has_bespoke_prefill`.  FP3/FP4 skip; request falls through to
  `unified_attention` (plugin's decode-shape kernel).
- `max_query_len > _CONTINUATION_DECODE_THRESHOLD` (128) — the
  split-KV decode kernel only fires for short continuation prefills;
  longer shapes go to the prefill kernel or fall through.

### When it crashes in the kernel

- `cannot cast intN to fp8eXnv` at a `tl.load` — the load has
  `other=0` (int) for a cache whose dtype is fp8.  Use `other=0.0`
  for float caches; reserve `other=0` for integer caches (int8 WMMA
  path only).
- `NameError(<helper> is not defined)` — a `@triton.jit` helper used
  by the shared `_attn_packed` kernel lives in the wrong module.
  See [README.md § 7.1](README.md#71-where-to-put-tritonjit-helpers).
- `ValueError: KVQuantMode.NONE is the unquantized path and has no
  plugin` — `unified_attention` tried to look up a plugin for
  `NONE`.  Fixed at the dispatcher + `has_quant_kv_factory(NONE)`
  level; if you see it, one of those guards regressed.
