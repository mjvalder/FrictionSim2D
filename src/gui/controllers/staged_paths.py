"""Absolutising relative paths before a config is staged elsewhere.

``ConfigDocument.apply_material_directories`` deliberately stores relative
paths (see ``ConfigDocument._relative_to_config``): a saved config is meant
to travel with its run directory, so a folder under the config's own
directory is written as e.g. ``cif/{mat}.cif`` rather than an absolute path.

But ``PreviewController`` and ``GenerateController`` both copy the config
into a *different* directory before building it -- a preview cache directory
keyed by content hash, or a ``_generate`` staging directory -- and hand that
copy's path to a build subprocess. ``src.core.utils.search_paths`` resolves a
relative name against the config file's own directory (and its parents), so
once the config has been copied, a relative ``cif_path``/``pot_path`` that
used to resolve next to the user's original config no longer resolves at
all: it is now being searched for next to the staging directory instead.

This module fixes that at the staging boundary, not at the source: it takes
a config mapping plus the *original* document's directory and returns a copy
with path-like values resolved to absolute paths, using the exact same
search (:func:`~src.core.utils.search_paths` / ``_find_existing``) the
builders use to resolve them at build time. A value that already resolves
absolutely, or that cannot be resolved at all, is left untouched -- in the
latter case so that the existing "missing file" validation still reports it,
rather than this module mangling it into something more confusing.

The document's own saved copy on disk is never touched by this: it only ever
operates on an in-memory mapping about to be written to a *different* file.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

from ..config_validation import PATH_KEYS
from ...core.utils import _find_existing  # noqa: PLC2701  (intentional internal reuse)

__all__ = ['absolutize_staged_paths']

#: [2D] key naming a materials-list file. Not part of ``PATH_KEYS`` --
#: that constant is what ``preflight_paths`` checks as a direct file
#: reference -- but a relative ``materials_list`` has exactly the same
#: staging problem as a relative ``cif_path``/``pot_path``.
_MATERIALS_LIST_KEY = 'materials_list'

#: Every config key this module resolves.
_PATH_LIKE_KEYS = frozenset(PATH_KEYS) | {_MATERIALS_LIST_KEY}

#: Matches a ``{mat}``, ``{mat1}``, ``{mat2}``, ... placeholder left for
#: ``expand_config_sweeps`` to substitute later. Not anchored to a specific
#: name -- any future per-sheet placeholder needs no change here -- and
#: deliberately broader than ``run.placeholders.TEMPLATE_MARKERS`` (a fixed
#: ``('{mat}', '{mat1}', '{mat2}')`` tuple) for the same reason: this module
#: only needs to know "is this value templated at all", not which specific
#: marker it holds, so narrowing it to the fixed set would silently stop
#: resolving a not-yet-invented per-sheet placeholder.
_TEMPLATE_TOKEN = re.compile(r'\{[^{}]+\}')


def _resolve_plain(value: str, base_dir: Optional[Union[str, Path]]) -> str:
    """Resolve a value that carries no template placeholder."""
    found = _find_existing(value, base_dir)
    return str(found) if found is not None else value


def _resolve_templated(value: str, base_dir: Optional[Union[str, Path]]) -> str:
    """Resolve a value like ``cif/{mat}.cif``.

    The value as a whole cannot be located on disk -- the placeholder is not
    substituted yet -- so only the directory part is searched for, using the
    same resolution as a plain value, and the templated basename is rejoined
    unchanged.
    """
    as_path = Path(value)
    directory, basename = as_path.parent, as_path.name

    if str(directory) in ('.', ''):
        # No directory component: the templated file is meant to sit
        # directly in base_dir once substituted.
        if base_dir is None:
            return value
        resolved_dir = Path(base_dir)
        return str(resolved_dir / basename) if resolved_dir.is_dir() else value

    found_dir = _find_existing(str(directory), base_dir)
    if found_dir is None or not found_dir.is_dir():
        return value
    return str(found_dir / basename)


def _resolve_value(value: str, base_dir: Optional[Union[str, Path]]) -> str:
    if not value or Path(value).is_absolute():
        return value
    if _TEMPLATE_TOKEN.search(value):
        return _resolve_templated(value, base_dir)
    return _resolve_plain(value, base_dir)


def absolutize_staged_paths(config: Mapping[str, Any],
                            base_dir: Optional[Union[str, Path]]) -> Dict[str, Any]:
    """A copy of ``config`` with path-like values resolved to absolute paths.

    Args:
        config: A config mapping in the ``{section: {key: value}}`` shape
            ``ConfigDocument.to_dict()``/``parse_config`` produce.
        base_dir: The *original* document's directory (``ConfigDocument.base_dir``)
            -- i.e. where the config used to live, not where it is about to be
            staged. ``None`` (an unsaved document) leaves every relative value
            untouched, same as the builders' own resolution would.

    Returns:
        A new mapping; ``config`` itself is not mutated. Sections that are not
        mappings (there should not be any, but this makes no assumption) pass
        through unchanged.
    """
    resolved: Dict[str, Any] = {}
    for section, values in config.items():
        if not isinstance(values, Mapping):
            resolved[section] = values
            continue

        new_values = dict(values)
        for key in _PATH_LIKE_KEYS:
            value = new_values.get(key)
            if isinstance(value, str) and value:
                new_values[key] = _resolve_value(value, base_dir)
        resolved[section] = new_values

    return resolved
