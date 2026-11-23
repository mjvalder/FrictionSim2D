"""Every module under frictionsim2d/ must import, and core's lazy registry must resolve.

This guards the core reorganisation: a move that breaks a lazy re-export or
leaves a dangling import fails here, not later inside a builder. It also
exercises the __getattr__ registries in frictionsim2d/__init__.py and
frictionsim2d/core/__init__.py, which grep-based checks cannot see.
"""
import importlib
import pkgutil

import pytest

import frictionsim2d


def _all_modules():
    return sorted(m.name for m in pkgutil.walk_packages(frictionsim2d.__path__, "frictionsim2d."))


@pytest.mark.parametrize("name", _all_modules())
def test_module_imports(name):
    importlib.import_module(name)


def test_core_lazy_exports_resolve():
    from frictionsim2d import core

    for name in core.__all__:
        assert getattr(core, name) is not None, f"frictionsim2d.core.{name} did not resolve"


def test_top_level_lazy_exports_resolve():
    assert "run_with_aiida" not in frictionsim2d.__all__  # AiiDA submission is gone
    for name in frictionsim2d.__all__:
        assert getattr(frictionsim2d, name) is not None, f"frictionsim2d.{name} did not resolve"


def test_builders_lazy_exports_resolve():
    """Every name in builders.__all__ resolves through its __getattr__ registry.

    pylint reports these names as undefined-all-variable because the
    registry is dynamic; this test is what makes that pragma honest.
    """
    from frictionsim2d import builders

    for name in builders.__all__:
        assert getattr(builders, name) is not None, f"frictionsim2d.builders.{name} did not resolve"
