# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV-cache quantization plugin registry.

Each quantization mode is a *plugin* exposing :class:`QuantKVSpec`
static metadata plus runtime entry points for the cache write and
paged-attention read paths.  Plugins live in one of two places:

* **Builtin** — any ``*.py`` file sitting next to this ``__init__``.
  Auto-discovered and imported once per process; files whose stem
  starts with ``_`` (private helpers like ``_packed_core`` or
  ``_hadamard``) are skipped, as are ``__init__.py`` and ``base.py``.

* **External** — any ``*.py`` file in a directory listed in the
  ``VLLM_QUANT_KV_PATH`` environment variable (``:``-separated on
  Linux/macOS, ``;``-separated on Windows; follows ``os.pathsep``).
  Scanned once per process on the first plugin lookup.  External
  plugins are loaded *before* builtins so a user can transparently
  override an in-tree plugin by shipping a file with the same name.

Adding a new mode
-----------------
Drop a single ``.py`` file that defines a :class:`QuantKVPlugin`
subclass with a :class:`QuantKVSpec` and calls :func:`register` at
module level.  For a builtin: place it next to this file — no other
edit required, auto-discovery picks it up.  For an external plugin:
place it in a directory on ``VLLM_QUANT_KV_PATH``.  Select it at
runtime with ``--kv-cache-dtype <spec.name>``.
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

# Registry keyed by public plugin name (string).
_REGISTRY: dict[str, QuantKVPlugin] = {}

# Filenames whose stem is not a plugin even though they sit in this
# package.  Everything else matching ``*.py`` and not starting with
# ``_`` is treated as a plugin and imported at package init.
_NON_PLUGIN_STEMS: frozenset[str] = frozenset({"__init__", "base"})

# Tracks one-shot state.
_BUILTIN_LOADED = False
_EXTERNAL_LOADED = False


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register(plugin: QuantKVPlugin) -> None:
    """Register a plugin instance under its ``spec.name``.

    Called at module import by each plugin file.  First-registration
    wins: a second call with the same concrete class is an idempotent
    no-op (benign double-import); a call with a *different* class on
    a name already held is logged and ignored — this is how an
    external plugin transparently overrides an in-tree one with the
    same name.  The loader runs external discovery *before* the
    builtin scan so the external wins by order of arrival.
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


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def _ensure_builtin_loaded() -> None:
    """Import every plugin file sitting next to this ``__init__`` once.

    Runs at most once per process, the first time a lookup is made.
    Walks the package directory, imports every ``*.py`` file whose
    stem is not ``_``-prefixed and not in :data:`_NON_PLUGIN_STEMS`,
    and lets each module self-register via :func:`register` at its
    module level.  Failures are logged and skipped — a broken builtin
    never prevents the other plugins from loading.
    """
    global _BUILTIN_LOADED
    if _BUILTIN_LOADED:
        return
    _BUILTIN_LOADED = True

    pkg_dir = Path(__file__).parent
    pkg_name = __name__
    for py in sorted(pkg_dir.glob("*.py")):
        if py.name.startswith("_"):
            continue
        if py.stem in _NON_PLUGIN_STEMS:
            continue
        module_path = f"{pkg_name}.{py.stem}"
        try:
            importlib.import_module(module_path)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load builtin KV-quant plugin %s: %s", py, exc)


def _ensure_external_loaded() -> None:
    """Discover and import every plugin in ``VLLM_QUANT_KV_PATH``.

    Runs at most once per process.  Each directory on the path is
    scanned for top-level ``*.py`` files (names starting with ``_``
    are skipped as private helpers); every file is imported under a
    synthetic module name so import-time :func:`register` calls take
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


def _ensure_all_loaded() -> None:
    """Trigger external discovery first, then builtin scan.

    External-first so a user-supplied file with the same ``spec.name``
    as a builtin wins via the first-registration rule in
    :func:`register`.
    """
    _ensure_external_loaded()
    _ensure_builtin_loaded()


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------


def get_quant_kv_plugin(name: str) -> QuantKVPlugin:
    """Look up a plugin by its public name.

    Resolution order: already-loaded registry → external scan
    (``VLLM_QUANT_KV_PATH``) → builtin scan (``*.py`` next to this
    file).  Raises :class:`KeyError` when the name does not resolve.
    """
    if name in _REGISTRY:
        return _REGISTRY[name]
    _ensure_all_loaded()
    if name in _REGISTRY:
        return _REGISTRY[name]
    raise KeyError(
        f"No KV-quant plugin named {name!r}.  Registered: "
        f"{sorted(_REGISTRY)}.  External plugins are loaded from "
        f"directories listed in VLLM_QUANT_KV_PATH."
    )


def get_plugin_for_dtype(kv_cache_dtype: str) -> QuantKVPlugin | None:
    """Resolve a ``--kv-cache-dtype`` string to a plugin, if any.

    Returns the matching plugin for dtype strings that name a builtin
    or registered external plugin, or ``None`` for dtype strings not
    handled by any plugin (``auto``, ``float16``, ``fp8_inc``, …) —
    those go through non-plugin code paths in vLLM.
    """
    if not kv_cache_dtype:
        return None
    if kv_cache_dtype in _REGISTRY:
        return _REGISTRY[kv_cache_dtype]
    _ensure_all_loaded()
    return _REGISTRY.get(kv_cache_dtype)


def get_quant_kv_factory(mode: "KVQuantMode") -> QuantKVFactory:
    """Legacy lookup keyed by :class:`KVQuantMode`.

    Resolves ``mode`` to a plugin name via ``mode.name.lower()`` and
    delegates to :func:`get_quant_kv_plugin`.  Kept for call sites
    that still thread ``KVQuantMode`` through the stack; new code
    should use :func:`get_quant_kv_plugin` or
    :func:`get_plugin_for_dtype` directly.
    """
    from vllm.v1.kv_cache_interface import KVQuantMode

    if mode == KVQuantMode.NONE:
        raise ValueError(
            "KVQuantMode.NONE is the unquantized path and has no plugin"
        )
    return get_quant_kv_plugin(mode.name.lower())


def has_quant_kv_plugin(name: str) -> bool:
    """Return True if *name* resolves to a plugin.

    Forces discovery so that callers get a correct answer before the
    first lookup — useful for config validation.
    """
    if name in _REGISTRY:
        return True
    _ensure_all_loaded()
    return name in _REGISTRY


def has_quant_kv_factory(mode: "KVQuantMode") -> bool:
    """Legacy predicate keyed by :class:`KVQuantMode`.

    Returns False for :attr:`KVQuantMode.NONE` to match the contract of
    :func:`get_quant_kv_factory` (which raises for NONE).  This keeps
    the common ``if has_*: get_*`` pattern safe — a caller that passes
    NONE (e.g. because the call site uses
    ``if mode != NONE or kv_cache_dtype``) shortcuts out cleanly instead
    of hitting the raise from the follow-up ``get_*`` call.
    """
    from vllm.v1.kv_cache_interface import KVQuantMode

    if mode == KVQuantMode.NONE:
        return False
    return has_quant_kv_plugin(mode.name.lower())


def list_registered_plugins() -> list[str]:
    """Return the sorted list of all discoverable plugin names.

    Triggers external + builtin discovery so every installed plugin
    (user-supplied and in-tree) appears in the result.
    """
    _ensure_all_loaded()
    return sorted(_REGISTRY)
