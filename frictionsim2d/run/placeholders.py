"""Material placeholders (``{mat}``/``{mat1}``/``{mat2}``) in a config section.

They exist only before :func:`~frictionsim2d.run.expand.expand_config_sweeps`
substitutes them. ``gui.config_validation.preflight_paths`` also sees them, since
it validates a config still being edited, and reports a templated path as
``unexpanded_template`` rather than a missing file. Kept apart from ``expand``
because ``expand`` imports the builders. Not related to the Jinja templates in
``frictionsim2d/templates/``.
"""

from __future__ import annotations

from typing import Optional

__all__ = ['TEMPLATE_MARKERS', 'has_template_marker', 'substitute_materials']

#: Every placeholder a config section may hold: ``{mat}`` (single material),
#: ``{mat1}``/``{mat2}`` (first and second sheet of an ordered pair).
TEMPLATE_MARKERS = ('{mat}', '{mat1}', '{mat2}')


def has_template_marker(text: str) -> bool:
    """Whether ``text`` still holds an unsubstituted material placeholder."""
    return isinstance(text, str) and any(marker in text for marker in TEMPLATE_MARKERS)


def substitute_materials(text: str, mat1: str, mat2: Optional[str] = None) -> str:
    """Replace material placeholders in ``text`` with real names.

    ``{mat}`` and ``{mat1}`` both take ``mat1``; ``{mat2}`` is replaced only
    when ``mat2`` is given.
    """
    result = text.replace('{mat}', mat1).replace('{mat1}', mat1)
    if mat2 is not None:
        result = result.replace('{mat2}', mat2)
    return result
