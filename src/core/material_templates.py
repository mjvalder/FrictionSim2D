"""Placeholders substituted into a config section during material-sweep
expansion.

A templated config holds ``{mat}``/``{mat1}``/``{mat2}`` in place of a real
material name (``cif_path``, ``pot_path``, ``mat`` itself) until
``expand_config_sweeps`` substitutes it. Two very different callers need to
agree on exactly what a marker looks like:

* ``gui.config_validation.preflight_paths`` must recognise a still-templated path
  as its own issue kind (``unexpanded_template``), not a missing file -- the
  file genuinely does not exist yet, but for a reason the user cannot fix by
  finding a file, unlike a real typo. Missing this distinction for ``{mat1}``/
  ``{mat2}`` (a literal check for ``{mat}`` alone) turns every heterostructure
  pair config into a hard validation error that blocks Render and Generate.
* ``src.run.expand`` substitutes the markers once real material names are
  known, both for a single-material sweep (``{mat}`` and ``{mat1}`` are
  equivalent there) and for an ordered pair (``{mat1}`` / ``{mat2}``).

Kept in its own module, not in either of the two, so ``config_validation``
(imported by GUI validation on every keystroke) does not have to pull in
``src.run.expand``'s builder imports to get it.
"""

from __future__ import annotations

from typing import Optional

__all__ = ['TEMPLATE_MARKERS', 'has_template_marker', 'substitute_materials']

#: Every placeholder a config section may hold. ``{mat}`` is the original,
#: single-material spelling; ``{mat1}``/``{mat2}`` name the first and second
#: sheet of an ordered heterostructure pair. ``{mat}`` and ``{mat1}`` are
#: substituted identically -- see :func:`substitute_materials`.
TEMPLATE_MARKERS = ('{mat}', '{mat1}', '{mat2}')


def has_template_marker(text: str) -> bool:
    """Whether ``text`` still holds an unsubstituted material placeholder."""
    return isinstance(text, str) and any(marker in text for marker in TEMPLATE_MARKERS)


def substitute_materials(text: str, mat1: str, mat2: Optional[str] = None) -> str:
    """Replace material placeholders in ``text`` with real names.

    ``{mat}`` and ``{mat1}`` are the same placeholder under two spellings --
    the former is what a single-material config has always used, the latter
    is how an ordered pair's primary sheet spells it -- so both take
    ``mat1``. ``{mat2}`` is only replaced when ``mat2`` is given; a
    single-material template has no second material to substitute, so a
    stray ``{mat2}`` (there should not be one) is left as-is rather than
    raising.
    """
    result = text.replace('{mat}', mat1).replace('{mat1}', mat1)
    if mat2 is not None:
        result = result.replace('{mat2}', mat2)
    return result
