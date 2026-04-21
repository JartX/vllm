# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV-cache quantization plugin registry.

Each quantization mode is a *plugin* exposing :class:`QuantKVSpec`
static metadata plus runtime entry points for the cache write and
paged-attention read paths.  Plugins live in one of two places:

* **Builtin** — a file under ``triton_quant_kv/``, listed in
  :data:`_BUILTIN_MODULES` below.  Lazy-imported on first lookup so
  unused modes pay zero import or Triton compile cost.

* **External** — any ``*.py`` file in a directory listed in the
  ``VLLM_QUANT_KV_PATH`` environment variable (``:``-separated on
  Linux/macOS, ``;``-separated on Windows; follows ``os.pathsep``).
  Scanned once per process on the first plugin lookup.  External
  plugins self-register by calling :func:`register` at import.

Adding a new builtin mode
-------------------------
1. Create a new file under ``triton_quant_kv/`` that defines a
   subclass of :class:`QuantKVPlugin`, sets ``spec``, implements
   :meth:`QuantKVPlugin.reshape_and_cache` (and optionally
   :meth:`QuantKVPlugin.unified_attention`), and calls
   :func:`register` at module level.
2. Add one line to :data:`_BUILTIN_MODULES` below mapping the
   plugin name to the module path.

Adding an external plugin
-------------------------
1. Drop the same kind of file in any directory on your machine.
2. ``export VLLM_QUANT_KV_PATH=/path/to/dir`` before launching vLLM.
3. Select it with ``--kv-cache-dtype <plugin_name>``.

No edits to vLLM sources are required for external plugins — the
plugin name is the lookup key end-to-end.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.v1.attention.ops.triton_quant_kv.base import (
    QuantKVFactory,
    QuantKVPlugin,
    QuantKVSpec,
)

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVQuantMode

logger = init_logger(__name__)

__all__ = [
    "QuantKVFactory",
    "QuantKVPlugin",
    "QuantKVSpec",
    "register",
    "get_quant_kv_factory",
    "get_quant_kv_plugin",
    "get_plugin_for_dtype",
    "has_quant_kv_factory",
    "has_quant_kv_plugin",
    "list_registered_plugins",
]


# Registry keyed by public plugin name (string).  Keying by name
# rather than by ``KVQuantMode`` lets external plugins declare names
# that are not in the closed enum.
_REGISTRY: dict[str, QuantKVPlugin] = {}

# Builtin modes: name -> dotted module path, lazy-imported on first
# lookup.  By convention each plugin's file stem matches its public
# name (``int4_per_token_head.py`` registers ``int4_per_token_head``),
# so adding a builtin is one new file plus one line here.  Shared
# helpers live in ``_``-prefixed modules that the external discovery
# scanner skips.
_BUILTIN_MODULES: dict[str, str] = {
    "none": (
        "vllm.v1.attention.ops.triton_quant_kv.none"
    ),
    "fp8_per_tensor": (
        "vllm.v1.attention.ops.triton_quant_kv.fp8_per_tensor"
    ),
    "int8_per_token_head": (
        "vllm.v1.attention.ops.triton_quant_kv.int8_per_token_head"
    ),
    "fp8_per_token_head": (
        "vllm.v1.attention.ops.triton_quant_kv.fp8_per_token_head"
    ),
    "int4_per_token_head": (
        "vllm.v1.attention.ops.triton_quant_kv.int4_per_token_head"
    ),
    "int2_per_token_head": (
        "vllm.v1.attention.ops.triton_quant_kv.int2_per_token_head"
    ),
}

# Tracks whether the one-time external plugin scan has run.  External
# plugins are loaded lazily on the first lookup that misses the
# registry, not at package import, so a process that uses no KV
# quantization never touches the env var.
_EXTERNAL_LOADED = False


def register(plugin: QuantKVPlugin) -> None:
    """Register a plugin instance under its ``spec.name``.

    Called at module import by each builtin factory and by external
    plugin modules.  First-registration wins: repeated registrations
    of the same concrete class are idempotent (benign double-import);
    registrations of a *different* class under a name already held
    are logged and ignored, so the lookup chain (which loads external
    plugins before falling back to builtins) can give users a working
    override of a builtin simply by shipping a plugin with the same
    name in ``VLLM_QUANT_KV_PATH``.
    """
    name = plugin.spec.name
    existing = _REGISTRY.get(name)
    if existing is not None:
        if type(existing) is type(plugin):
            return
        logger.info(
            "KV-quant plugin %r already registered as %s; ignoring new "
            "registration of %s (first-registration-wins; external "
            "plugins override builtins when loaded first).",
            name,
            type(existing).__name__,
            type(plugin).__name__,
        )
        return
    _REGISTRY[name] = plugin


def _load_builtin(name: str) -> None:
    """Lazy-import the builtin module that provides *name*, if any."""
    module_path = _BUILTIN_MODULES.get(name)
    if module_path is None:
        return
    importlib.import_module(module_path)


def _ensure_external_loaded() -> None:
    """Discover and import every plugin in ``VLLM_QUANT_KV_PATH``.

    Runs at most once per process.  Each directory on the path is
    scanned for top-level ``*.py`` files (names starting with ``_``
    are skipped as private helpers); every file is imported under a
    synthetic module name so import-time ``register()`` calls take
    effect.  Bad plugins are logged and skipped — a broken external
    plugin does not tank vLLM startup.
    """
    global _EXTERNAL_LOADED
    if _EXTERNAL_LOADED:
        return
    _EXTERNAL_LOADED = True

    raw = os.environ.get("VLLM_QUANT_KV_PATH", "")
    if not raw:
        return

    for dir_str in raw.split(os.pathsep):
        dir_str = dir_str.strip()
        if not dir_str:
            continue
        path = Path(dir_str)
        if not path.is_dir():
            logger.warning(
                "VLLM_QUANT_KV_PATH entry %r is not a directory; skipping",
                dir_str,
            )
            continue
        for py in sorted(path.glob("*.py")):
            if py.name.startswith("_"):
                continue
            module_name = f"_vllm_quant_kv_ext_{py.stem}"
            if module_name in sys.modules:
                continue
            try:
                mod_spec = importlib.util.spec_from_file_location(
                    module_name, py
                )
                if mod_spec is None or mod_spec.loader is None:
                    raise ImportError(f"could not build import spec for {py}")
                module = importlib.util.module_from_spec(mod_spec)
                sys.modules[module_name] = module
                mod_spec.loader.exec_module(module)
                logger.info(
                    "Loaded external KV-quant plugin %s (%s)", py, module_name
                )
            except Exception as exc:  # noqa: BLE001
                sys.modules.pop(module_name, None)
                logger.error(
                    "Failed to load external KV-quant plugin %s: %s", py, exc
                )


def get_quant_kv_plugin(name: str) -> QuantKVPlugin:
    """Look up a plugin by its public name.

    Resolution order: in-memory registry → external scan
    (``VLLM_QUANT_KV_PATH``) → lazy-import the builtin that claims
    *name*.  External is tried before builtin so that a user-supplied
    plugin with the same name transparently overrides the builtin
    (see :func:`register` for the first-wins semantics).  Raises
    :class:`KeyError` when the name does not resolve anywhere.
    """
    if name in _REGISTRY:
        return _REGISTRY[name]
    _ensure_external_loaded()
    if name in _REGISTRY:
        return _REGISTRY[name]
    _load_builtin(name)
    if name in _REGISTRY:
        return _REGISTRY[name]
    raise KeyError(
        f"No KV-quant plugin named {name!r}.  Builtin names: "
        f"{sorted(_BUILTIN_MODULES)}.  External plugins are loaded "
        f"from directories listed in VLLM_QUANT_KV_PATH."
    )


def get_quant_kv_factory(mode: "KVQuantMode") -> QuantKVFactory:
    """Legacy lookup keyed by :class:`KVQuantMode`.

    Resolves ``mode`` to a plugin name via ``mode.name.lower()`` and
    delegates to :func:`get_quant_kv_plugin`.  Kept for call sites
    that still thread ``KVQuantMode`` through the stack; new code
    should use :func:`get_quant_kv_plugin`.
    """
    from vllm.v1.kv_cache_interface import KVQuantMode

    if mode == KVQuantMode.NONE:
        raise ValueError(
            "KVQuantMode.NONE is the unquantized path and has no plugin"
        )
    plugin = get_quant_kv_plugin(mode.name.lower())
    # All in-tree factories subclass QuantKVFactory; preserve the
    # narrower return type for legacy callers that introspect it.
    assert isinstance(plugin, QuantKVFactory), (
        f"Plugin {mode.name!r} resolved to {type(plugin).__name__}, "
        f"which is not a legacy QuantKVFactory"
    )
    return plugin


def has_quant_kv_plugin(name: str) -> bool:
    """Return True if *name* resolves to a plugin (loaded or lazy).

    Does not trigger the external scan; a user-supplied plugin becomes
    visible to this predicate only after it has been registered (via
    :func:`register`) or after :func:`list_registered_plugins` or
    :func:`get_plugin_for_dtype` has forced discovery.
    """
    return name in _REGISTRY or name in _BUILTIN_MODULES


def get_plugin_for_dtype(kv_cache_dtype: str) -> QuantKVPlugin | None:
    """Resolve a ``--kv-cache-dtype`` string to a plugin, if any.

    Returns the matching plugin for dtype strings that name a builtin
    or registered external plugin, or ``None`` for the unquantized
    path and for dtype strings handled inside the core kernel without
    a plugin (``fp8``, ``fp8_e4m3``, ``nvfp4``, …).  External plugin
    discovery is triggered on miss, so a user-supplied plugin named
    ``"mymode"`` becomes selectable via ``--kv-cache-dtype mymode``
    without any edit to vLLM sources.
    """
    if not kv_cache_dtype:
        return None
    # Fast path: already-loaded plugin.
    if kv_cache_dtype in _REGISTRY:
        return _REGISTRY[kv_cache_dtype]
    # External first — gives VLLM_QUANT_KV_PATH precedence over
    # builtins when the same name is provided by both.
    _ensure_external_loaded()
    if kv_cache_dtype in _REGISTRY:
        return _REGISTRY[kv_cache_dtype]
    # Fall back to a builtin module that claims the name.
    if kv_cache_dtype in _BUILTIN_MODULES:
        _load_builtin(kv_cache_dtype)
        return _REGISTRY.get(kv_cache_dtype)
    return None


def has_quant_kv_factory(mode: "KVQuantMode") -> bool:
    """Legacy predicate keyed by :class:`KVQuantMode`."""
    return has_quant_kv_plugin(mode.name.lower())


def list_registered_plugins() -> list[str]:
    """Return the sorted list of currently-registered plugin names.

    Triggers external discovery so user-supplied plugins show up even
    before first use.  Does *not* eagerly import builtins that have
    not been requested yet — those names appear in the returned list
    only after their first lookup.
    """
    _ensure_external_loaded()
    return sorted(_REGISTRY)
