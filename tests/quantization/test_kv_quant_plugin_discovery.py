# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the KV-cache quantization plugin registry.

Covers the pure-Python discovery layer that sits above the Triton
kernels:

* in-tree builtin plugins load via ``_BUILTIN_MODULES``;
* user-supplied plugins load by scanning every directory listed in
  ``VLLM_QUANT_KV_PATH`` and importing each ``*.py`` file;
* first-registration wins, so an external plugin with the same name
  as a builtin transparently overrides it;
* malformed external plugins are logged and skipped without crashing
  the registry.

These tests do not touch a GPU — the registry is a Python-side
concern and is exercised by constructing plugins whose runtime
methods raise on call.  End-to-end kernel integration lives in
``test_per_token_kv_cache.py``.

Run: ``pytest tests/quantization/test_kv_quant_plugin_discovery.py -v``
"""

from __future__ import annotations

import textwrap

import pytest
import torch

import vllm.v1.attention.ops.triton_quant_kv as _qkv
from vllm.v1.attention.ops.triton_quant_kv import (
    QuantKVPlugin,
    QuantKVSpec,
    get_plugin_for_dtype,
    get_quant_kv_plugin,
    has_quant_kv_plugin,
    list_registered_plugins,
)


@pytest.fixture
def fresh_registry():
    """Yield an empty registry; restore the original on teardown.

    The plugin registry is a process-global singleton, so isolating
    tests requires snapshot/clear/restore around each test body.
    """
    original_registry = dict(_qkv._REGISTRY)
    original_loaded = _qkv._EXTERNAL_LOADED
    _qkv._REGISTRY.clear()
    _qkv._EXTERNAL_LOADED = False
    try:
        yield
    finally:
        _qkv._REGISTRY.clear()
        _qkv._REGISTRY.update(original_registry)
        _qkv._EXTERNAL_LOADED = original_loaded


def _write_plugin(
    dir_path,
    file_name: str,
    name: str,
    *,
    packing_factor: int = 1,
    needs_per_token_head_scales: bool = False,
    bespoke_attention: bool = True,
    extra_class_body: str = "",
) -> None:
    """Write a minimal external plugin file into *dir_path*."""
    attn_impl = (
        "def unified_attention(self, *args, **kwargs):\n"
        "        raise NotImplementedError('test stub')"
        if bespoke_attention
        else ""
    )
    code = textwrap.dedent(
        f"""
        import torch
        from vllm.v1.attention.ops.triton_quant_kv import (
            QuantKVPlugin, QuantKVSpec, register,
        )

        class _TestPlugin_{name}(QuantKVPlugin):
            spec = QuantKVSpec(
                name={name!r},
                storage_dtype=torch.uint8,
                packing_factor={packing_factor},
                needs_per_token_head_scales={needs_per_token_head_scales},
                description="discovery-test plugin",
            )
            def reshape_and_cache(self, *args, **kwargs):
                raise NotImplementedError('test stub')
            {attn_impl}
            {extra_class_body}

        register(_TestPlugin_{name}())
        """
    )
    (dir_path / file_name).write_text(code)


def test_external_plugin_discovered_by_dtype(
    tmp_path, monkeypatch, fresh_registry
):
    _write_plugin(tmp_path, "my_mode.py", "my_mode", packing_factor=8)
    monkeypatch.setenv("VLLM_QUANT_KV_PATH", str(tmp_path))

    plugin = get_plugin_for_dtype("my_mode")

    assert plugin is not None
    assert plugin.spec.name == "my_mode"
    assert plugin.spec.packing_factor == 8
    assert plugin.spec.storage_dtype is torch.uint8
    assert plugin.has_bespoke_attention is True


def test_external_plugin_visible_via_has_and_list(
    tmp_path, monkeypatch, fresh_registry
):
    _write_plugin(tmp_path, "visible.py", "visible_mode")
    monkeypatch.setenv("VLLM_QUANT_KV_PATH", str(tmp_path))

    # Before any call, external scan has not happened yet and the
    # plugin is invisible.
    assert has_quant_kv_plugin("visible_mode") is False

    # list_registered_plugins triggers the external scan.
    names = list_registered_plugins()
    assert "visible_mode" in names
    assert has_quant_kv_plugin("visible_mode") is True


def test_external_plugin_overrides_builtin_name(
    tmp_path, monkeypatch, fresh_registry
):
    """External plugin claiming a builtin name wins by first-registration.

    The lookup chain loads external first, so when the user then
    requests ``int4_per_token_head`` the builtin import is skipped.
    """
    _write_plugin(
        tmp_path,
        "int4_override.py",
        "int4_per_token_head",
        packing_factor=2,
    )
    monkeypatch.setenv("VLLM_QUANT_KV_PATH", str(tmp_path))

    plugin = get_plugin_for_dtype("int4_per_token_head")

    assert plugin is not None
    assert type(plugin).__name__.startswith("_TestPlugin_")


def test_malformed_external_plugin_is_skipped(
    tmp_path, monkeypatch, fresh_registry, caplog
):
    """A syntactically broken plugin file should not crash discovery."""
    (tmp_path / "broken.py").write_text("this is @!= not valid python")
    _write_plugin(tmp_path, "ok.py", "ok_mode")
    monkeypatch.setenv("VLLM_QUANT_KV_PATH", str(tmp_path))

    # The good plugin still loads; the broken one is logged.
    assert get_plugin_for_dtype("ok_mode") is not None
    assert get_plugin_for_dtype("broken") is None


def test_private_files_are_ignored(
    tmp_path, monkeypatch, fresh_registry
):
    """Files starting with ``_`` are treated as private helpers."""
    _write_plugin(tmp_path, "_private.py", "private_mode")
    _write_plugin(tmp_path, "public.py", "public_mode")
    monkeypatch.setenv("VLLM_QUANT_KV_PATH", str(tmp_path))

    assert get_plugin_for_dtype("public_mode") is not None
    assert get_plugin_for_dtype("private_mode") is None


def test_nonexistent_directory_is_tolerated(
    tmp_path, monkeypatch, fresh_registry
):
    monkeypatch.setenv("VLLM_QUANT_KV_PATH", str(tmp_path / "does_not_exist"))
    # Must not raise.
    assert get_plugin_for_dtype("anything") is None


def test_empty_env_var_noop(monkeypatch, fresh_registry):
    monkeypatch.delenv("VLLM_QUANT_KV_PATH", raising=False)
    assert get_plugin_for_dtype("nothing_here") is None


def test_multiple_directories_via_pathsep(
    tmp_path, monkeypatch, fresh_registry
):
    import os

    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    _write_plugin(dir_a, "mode_a.py", "mode_a")
    _write_plugin(dir_b, "mode_b.py", "mode_b")

    monkeypatch.setenv(
        "VLLM_QUANT_KV_PATH",
        f"{dir_a}{os.pathsep}{dir_b}",
    )

    assert get_plugin_for_dtype("mode_a") is not None
    assert get_plugin_for_dtype("mode_b") is not None


def test_missing_plugin_raises_keyerror(fresh_registry):
    with pytest.raises(KeyError):
        get_quant_kv_plugin("definitely_not_a_real_mode_zzz")


def test_has_bespoke_attention_false_when_not_overridden(fresh_registry):
    """Plugin that relies on the core kernel (no override) reports False."""

    class CoreKernelPlugin(QuantKVPlugin):
        spec = QuantKVSpec(name="core_only", storage_dtype=torch.int8)

        def reshape_and_cache(self, *args, **kwargs):
            pass

    plugin = CoreKernelPlugin()
    assert plugin.has_bespoke_attention is False


def test_spec_is_frozen():
    """QuantKVSpec must be immutable so plugins can't drift at runtime."""
    spec = QuantKVSpec(name="x", storage_dtype=torch.uint8)
    with pytest.raises((AttributeError, Exception)):  # dataclass FrozenInstanceError
        spec.name = "y"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Builtin marker plugins (``none`` and ``fp8_per_tensor``)
# ---------------------------------------------------------------------------
# These tests *do not* use the ``fresh_registry`` fixture because the
# builtin plugin modules are imported once per process (Python module
# cache) — clearing the registry after import would leave them invisible
# on subsequent lookup since the module import is a no-op the second
# time.


def test_none_plugin_registered_as_marker():
    plugin = get_plugin_for_dtype("none")
    assert plugin is not None, "builtin 'none' plugin should register"
    assert plugin.spec.name == "none"
    assert plugin.spec.is_metadata_marker is True
    assert plugin.spec.storage_dtype is None, (
        "'none' inherits dtype from the model; storage_dtype should be None"
    )
    assert plugin.spec.packing_factor == 1
    assert plugin.spec.needs_per_token_head_scales is False


def test_fp8_per_tensor_plugin_registered_as_marker():
    plugin = get_plugin_for_dtype("fp8_per_tensor")
    assert plugin is not None, "builtin 'fp8_per_tensor' plugin should register"
    assert plugin.spec.name == "fp8_per_tensor"
    assert plugin.spec.is_metadata_marker is True
    assert plugin.spec.needs_per_tensor_scale is True
    assert plugin.spec.needs_per_token_head_scales is False
    assert plugin.spec.packing_factor == 1


def test_marker_plugin_reshape_and_cache_raises():
    """Calling runtime methods on a marker must fail loudly — the
    dispatcher should never route to a marker."""
    plugin = get_plugin_for_dtype("none")
    assert plugin is not None
    with pytest.raises(NotImplementedError, match="metadata marker"):
        plugin.reshape_and_cache()


def test_marker_plugin_has_no_bespoke_attention():
    """Markers must report False so the dispatcher falls through to the
    core kernel."""
    for name in ("none", "fp8_per_tensor"):
        plugin = get_plugin_for_dtype(name)
        assert plugin is not None
        assert plugin.has_bespoke_attention is False, (
            f"marker {name!r} must not claim a bespoke kernel"
        )
