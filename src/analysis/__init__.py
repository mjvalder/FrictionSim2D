"""On-demand analyses of a dataset in the local AiiDA store.

An analysis (:class:`~src.analysis.base.Analysis`) reads the integrity-pass
points of one dataset (:class:`~src.analysis.source.RunSource`) and returns
one table, stored as a result (:mod:`src.analysis.store`) and reused while
its name, version, parameters and dataset are unchanged.
"""
# pylint: disable=import-outside-toplevel
# Deliberate: like src/__init__.py, nothing is imported until it is asked for.
from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .base import ANALYSES, Analysis, get_analysis, register
    from .source import RunSource
    from .store import find_result, run_analysis, store_result

#: Public name -> the submodule that defines it.
_EXPORTS = {
    'ANALYSES': 'base', 'Analysis': 'base', 'get_analysis': 'base', 'register': 'base',
    'RunSource': 'source',
    'find_result': 'store', 'run_analysis': 'store', 'store_result': 'store',
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    """Import the submodule that defines ``name`` on first use."""
    if name in _EXPORTS:
        return getattr(import_module(f'.{_EXPORTS[name]}', __name__), name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
