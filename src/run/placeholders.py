"""Placeholders substituted into a config section during sweep expansion.

A templated config holds ``{mat}``/``{mat1}``/``{mat2}`` in place of a real
material name (``cif_path``, ``pot_path``, ``mat`` itself) until
:func:`~src.run.expand.expand_config_sweeps` substitutes it. Markers exist
only *before* expansion, which is what decides who needs this module:

* :mod:`src.run.expand` does the substituting, and is the only consumer that
  destroys markers.
* ``gui.config_validation.preflight_paths`` validates a config the user is
  still editing -- so it sees markers, and must report a still-templated path
  as its own issue kind (``unexpanded_template``) rather than a missing file.
  The file genuinely does not exist yet, but for a reason the user cannot fix
  by finding a file, unlike a real typo. Missing this distinction for
  ``{mat1}``/``{mat2}`` (a literal check for ``{mat}`` alone) turns every
  heterostructure pair config into a hard validation error that blocks Render
  and Generate.
* ``src.core.config``'s pydantic validators run strictly *after* expansion and
  never see a marker. They cannot tell ``{mat}.cif`` from a typo -- both raise
  the same ``FileNotFoundError`` -- which is correct, because they should
  never be handed one.

Kept beside :mod:`src.run.expand` rather than inside it: ``expand`` imports the
builders, and GUI validation would otherwise pull that whole tree in to read
three constants. ``src/run/__init__.py`` is deliberately empty, so importing
this module costs nothing.

Named ``placeholders`` rather than ``templates`` to avoid reading as a second
``src/templates/``, which holds the Jinja2 LAMMPS and HPC scripts -- an
unrelated sense of the word.
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
