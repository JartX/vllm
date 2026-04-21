# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Plugin contract for KV-cache quantization modes.

A *plugin* owns the paged-cache write path (``reshape_and_cache``) and,
when the mode's attention loop is structurally different from the core
kernel, the paged-attention read path (``unified_attention``).  Each
plugin declares static metadata via :class:`QuantKVSpec` so the cache
allocator and dispatcher can size pages and pick the right kernel
without any per-mode hard-coding.

Two flavours of plugin coexist:

* **New-style** — subclass :class:`QuantKVPlugin` and set ``spec`` to a
  :class:`QuantKVSpec` instance.  Required for external plugins loaded
  via ``VLLM_QUANT_KV_PATH``.

* **Legacy factory** — subclass :class:`QuantKVFactory` with
  ``mode: KVQuantMode`` plus ``packing_factor`` / ``needs_scale_caches``
  class attributes.  The shim synthesises ``spec`` on demand so these
  factories appear as plugins to the loader without any change to
  their implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVQuantMode


# ---------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QuantKVSpec:
    """Static metadata describing a KV-cache quantization plugin.

    Populated once at plugin registration and read by the loader to
    decide cache layout (packing, scale buffers, storage dtype) before
    any kernel is invoked.  Immutable by design — if a plugin needs to
    vary behaviour at runtime it should do so inside its own kernel
    launchers, not by mutating the spec.
    """

    #: Public plugin name.  Used by ``--kv-cache-dtype`` and by the
    #: discovery loader.  Must be unique across all registered plugins
    #: (builtin + external).  Conventionally matches the plugin file
    #: stem, e.g. ``"int4_per_token_head"``.
    name: str

    #: torch dtype of the stored cache bytes.  ``torch.uint8`` for
    #: sub-byte packed modes, ``torch.float8_e4m3fn`` for fp8 variants,
    #: ``torch.int8`` for int8.  ``None`` means "inherit from the
    #: model compute dtype" and is used by the unquantized marker
    #: plugin (:mod:`builtin.none`) — the concrete dtype is not known
    #: at registration and the cache allocator falls back to the
    #: model dtype in that case.
    storage_dtype: torch.dtype | None

    #: Number of logical KV elements stored per cache byte.  1 for
    #: plain storage, 2 for INT4 (nibbles), 4 for INT2 (quartets).
    #: The attention-spec page-size calculation divides ``head_size``
    #: by this factor.
    packing_factor: int = 1

    #: True when the plugin allocates per-(token, head) scale buffers
    #: next to the cache.  Triggers the extra
    #: ``2 * block_size * num_kv_heads * sizeof(fp32)`` budget in the
    #: page-size computation.
    needs_per_token_head_scales: bool = False

    #: True when the plugin is driven by a single per-tensor scalar
    #: (e.g. the existing FP8 per-tensor path).  Mutually exclusive
    #: with ``needs_per_token_head_scales``.
    needs_per_tensor_scale: bool = False

    #: True when this plugin is a *metadata marker* — it exists so
    #: callers can query cache layout (``storage_dtype``, packing
    #: factor, scale budget) from the registry, but the write and
    #: read runtime paths are handled elsewhere (typically inside the
    #: core ``kernel_unified_attention`` and the plain
    #: ``triton_reshape_and_cache_flash`` launcher).  Markers must
    #: not be dispatched to — the loader uses them for allocation /
    #: validation only.
    is_metadata_marker: bool = False

    #: Free-form human description surfaced in errors and logs.
    description: str = ""


# ---------------------------------------------------------------------------
# New-style plugin base
# ---------------------------------------------------------------------------


class QuantKVPlugin(ABC):
    """Base class for KV-cache quantization plugins.

    Subclass, set the class attribute ``spec`` to a
    :class:`QuantKVSpec` instance, implement :meth:`reshape_and_cache`,
    and optionally override :meth:`unified_attention` when the mode
    needs a bespoke attention kernel (sub-byte packed INT4 / INT2,
    centroid-based INT2, etc.).  Modes that can use the core
    ``kernel_unified_attention`` via its constexpr branches leave the
    default :meth:`unified_attention` in place.

    Plugin modules register themselves at import by calling
    :func:`vllm.v1.attention.ops.triton_quant_kv.register` on an
    instance.  The loader auto-discovers them: every ``*.py`` file
    sitting next to the ``__init__.py`` of the plugin package (and
    not starting with ``_``) is treated as a builtin plugin, and
    every ``*.py`` file in directories listed in the
    ``VLLM_QUANT_KV_PATH`` environment variable is treated as an
    external plugin.  No registry file needs editing to add a new
    mode — drop the ``.py`` and it shows up on next process start.
    """

    #: Static metadata for this plugin.  Must be set by subclasses.
    spec: QuantKVSpec

    # ----- Capability introspection ----------------------------------------
    @property
    def has_bespoke_attention(self) -> bool:
        """True if this plugin owns a custom paged-attention kernel.

        Checked by the dispatcher: when True, the attention call is
        routed to :meth:`unified_attention` on the plugin; when False
        the core ``kernel_unified_attention`` handles it via its
        constexpr branches (only possible for modes already known to
        that kernel: ``NONE``, ``FP8_PER_TENSOR``, ``INT8_PER_TOKEN_HEAD``,
        ``FP8_PER_TOKEN_HEAD``).  External plugins with data layouts
        not recognised by the core kernel must override
        :meth:`unified_attention` so this returns True.
        """
        return type(self).unified_attention is not QuantKVPlugin.unified_attention

    # ----- Cache shape introspection ---------------------------------------
    def packed_head_size(self, head_size: int) -> int:
        """Storage head size after packing: ``head_size // packing_factor``."""
        pf = self.spec.packing_factor
        assert head_size % pf == 0, (
            f"head_size={head_size} is not divisible by packing factor "
            f"{pf} required by plugin {self.spec.name!r}"
        )
        return head_size // pf

    def allocate_scale_caches(
        self,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        device: torch.device,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Allocate aux per-(token, head) scale buffers.

        Default: when ``spec.needs_per_token_head_scales`` is True,
        allocate one ``float32`` per (block, slot, kv_head) for both
        K and V — the layout shared by every per-token-head mode.
        Plugins needing a different shape or dtype override this.
        """
        if not self.spec.needs_per_token_head_scales:
            return (None, None)
        shape = (num_blocks, block_size, num_kv_heads)
        return (
            torch.zeros(shape, dtype=torch.float32, device=device),
            torch.zeros(shape, dtype=torch.float32, device=device),
        )

    # ----- Cache write path ------------------------------------------------
    @abstractmethod
    def reshape_and_cache(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        *,
        k_scale_cache: torch.Tensor | None = None,
        v_scale_cache: torch.Tensor | None = None,
    ) -> None:
        """Write *key*/*value* into the paged cache for this mode.

        Per-token-head modes also write scales into the supplied
        ``k_scale_cache`` / ``v_scale_cache``.
        """

    # ----- Attention read path ---------------------------------------------
    def unified_attention(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        out: torch.Tensor,
        *,
        cu_seqlens_q: torch.Tensor,
        max_seqlen_q: int,
        seqused_k: torch.Tensor,
        max_seqlen_k: int,
        softmax_scale: float,
        window_size: tuple[int, int],
        block_table: torch.Tensor,
        softcap: float,
        sinks: torch.Tensor | None,
        alibi_slopes: torch.Tensor | None,
        use_alibi_sqrt: bool,
        qq_bias: torch.Tensor | None,
        output_scale: torch.Tensor | None,
        mm_prefix_range: torch.Tensor | None,
        k_scale_cache: torch.Tensor | None = None,
        v_scale_cache: torch.Tensor | None = None,
        seq_threshold_3D: int | None = None,
        num_par_softmax_segments: int | None = None,
        softmax_segm_output: torch.Tensor | None = None,
        softmax_segm_max: torch.Tensor | None = None,
        softmax_segm_expsum: torch.Tensor | None = None,
    ) -> None:
        """Run paged attention with this plugin's KV layout, writing into *out*.

        The default raises: plugins that do not override this are
        expected to be handled by the core ``kernel_unified_attention``
        via its constexpr dispatch, and the call should never reach
        here for such modes.
        """
        raise NotImplementedError(
            f"Plugin {self.spec.name!r} does not implement a bespoke "
            f"attention kernel.  Modes without a bespoke kernel are "
            f"expected to be handled by the core unified_attention "
            f"kernel via its constexpr dispatch."
        )


# ---------------------------------------------------------------------------
# Legacy factory shim
# ---------------------------------------------------------------------------


class QuantKVFactory(QuantKVPlugin):
    """Legacy base class for the four in-tree quantization factories.

    Predates :class:`QuantKVSpec`.  Subclasses declare
    ``mode: KVQuantMode`` plus ``packing_factor`` / ``needs_scale_caches``
    as class attributes; the shim synthesises a :class:`QuantKVSpec` on
    first access so the loader can treat factories and new-style
    plugins uniformly.  New plugins — especially external ones loaded
    via ``VLLM_QUANT_KV_PATH`` — should subclass :class:`QuantKVPlugin`
    directly and set ``spec`` explicitly; that path carries richer
    metadata (e.g. ``storage_dtype``) that legacy factories only
    approximate.
    """

    #: KV quant mode this factory implements.  Subclasses must set this.
    mode: "KVQuantMode"
    #: Logical values per cache byte.  Override in subclasses as needed.
    packing_factor: int = 1  # type: ignore[assignment]
    #: True when this factory needs per-(token, head) scale buffers.
    needs_scale_caches: bool = False

    @cached_property
    def spec(self) -> QuantKVSpec:  # type: ignore[override]
        return QuantKVSpec(
            name=self.mode.name.lower(),
            storage_dtype=self._legacy_storage_dtype(),
            packing_factor=self.packing_factor,
            needs_per_token_head_scales=self.needs_scale_caches,
            description=f"legacy factory for {self.mode.name}",
        )

    def _legacy_storage_dtype(self) -> torch.dtype:
        """Best-effort storage dtype from the legacy ``mode`` enum.

        Approximate on purpose — legacy call sites read the dtype
        directly from allocated cache tensors rather than from the
        spec, so this field exists only for parity with new-style
        plugins that populate it explicitly.  Covers the handful of
        modes still addressed by :class:`KVQuantMode`; packed modes
        are plugin-only and set ``spec.storage_dtype`` directly.
        """
        from vllm.v1.kv_cache_interface import KVQuantMode as _M

        if self.mode == _M.INT8_PER_TOKEN_HEAD:
            return torch.int8
        if self.mode in (_M.FP8_PER_TOKEN_HEAD, _M.FP8_PER_TENSOR):
            return torch.float8_e4m3fn
        return torch.uint8
