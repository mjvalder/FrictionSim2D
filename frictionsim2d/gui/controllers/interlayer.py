"""What the form says about the interlayer style, without Qt.

The rules are the backend's (`frictionsim2d.potentials.interlayer`): which bundled file
a style defaults to, and which Gaussian strength a material carries. This
module only asks them of the document being edited, so a blank field can show
what will be used and a choice that cannot build is reported on the field
that has to change -- not as an unlocated message from the model validator.

Element detection reads the sheet CIFs with the same cheap text scan the LJ
pair picker uses (`controllers.elements`), never ASE. It fails soft: any sheet
whose CIF cannot be read (missing, templated by a sweep, unparseable) makes
the answer "unknown", and unknown means no placeholder and no error -- the
build-time check still runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Mapping, Optional, Set

from ...core.utils.paths import _find_existing  # noqa: PLC2701
from ...potentials.interlayer import default_interlayer_file, gaussian_g
from ...run.placeholders import has_template_marker
from ..config_validation import ConfigIssue
from ..schema.overrides import DOC_PREDICATES, interlayer_styles, sheet_sections
from .elements import _parse_cif_text  # noqa: PLC2701

__all__ = ['sheet_elements', 'interlayer_file_placeholder',
           'interlayer_g_placeholder', 'interlayer_issues']


def sheet_elements(values: Mapping[str, Any],
                   base_dir: Optional[Path]) -> Optional[Set[str]]:
    """Union of every sheet CIF's elements; None if any cannot be read."""
    found: Set[str] = set()
    sections = sheet_sections(values)
    if not sections:
        return None
    for key in sections:
        cif = values[key].get('cif_path')
        if not isinstance(cif, str) or not cif or has_template_marker(cif):
            return None
        path = _find_existing(cif, base_dir)
        if path is None:
            return None
        try:
            elements = _parse_cif_text(path.read_text(encoding='utf-8', errors='ignore'))
        except OSError:
            return None
        if not elements:
            return None
        found |= elements
    return found


def _file_styles(values: Mapping[str, Any]) -> List[str]:
    return [s for s in interlayer_styles(values) if s in ('ilp', 'kc')]


def interlayer_file_placeholder(values: Mapping[str, Any],
                                base_dir: Optional[Path]) -> str:
    """``default: <bundled file>``, ``required``, or '' when unknown."""
    styles = _file_styles(values)
    elements = sheet_elements(values, base_dir) if styles else None
    if elements is None:
        return ''
    names = [default_interlayer_file(style, elements) for style in styles]
    if any(name is None for name in names):
        return 'required'
    return 'default: ' + ', '.join(dict.fromkeys(names))


def _g_value(values: Mapping[str, Any], section: str) -> Optional[float]:
    """The Gaussian strength ``section`` would run with, or None."""
    sheet = values.get(section) or {}
    override = sheet.get('interlayer_g')
    if override in (None, ''):
        override = None
    else:
        try:
            override = float(override)
        except (TypeError, ValueError):
            return None
    return gaussian_g(str(sheet.get('mat') or ''), override)


def interlayer_g_placeholder(values: Mapping[str, Any], section: str) -> str:
    """``default: <table g>`` for ``section``'s material, else ``required``."""
    sheet = values.get(section) or {}
    table = gaussian_g(str(sheet.get('mat') or ''), None)
    return 'required' if table is None else f'default: {table:g}'


def interlayer_issues(values: Mapping[str, Any],
                      base_dir: Optional[Path]) -> List[ConfigIssue]:
    """Blocking issues for an interlayer choice that cannot build, located.

    Mirrors two of `check_interlayer_choice`'s rules -- a ``kc``/``ilp`` style
    with no file and no bundled default, and ``lj_g`` on a material with no
    built-in g and none given -- on the field the user has to fill in. Only
    while those fields are shown: a hidden field is not in the document.
    """
    issues: List[ConfigIssue] = []
    if DOC_PREDICATES['interlayer_is_lj_g'](values):
        for section in sheet_sections(values):
            if _g_value(values, section) is None:
                mat = (values.get(section) or {}).get('mat')
                issues.append(ConfigIssue(
                    message=(f"LJ + Gaussian needs a Gaussian strength g (meV) "
                             f"for {mat!r}: it has no built-in value."),
                    section=section, key='interlayer_g'))
    if DOC_PREDICATES['interlayer_needs_file'](values):
        general = values.get('general') or {}
        if general.get('interlayer_file') in (None, ''):
            elements = sheet_elements(values, base_dir)
            if elements is not None:
                missing = [s for s in _file_styles(values)
                           if default_interlayer_file(s, elements) is None]
                if missing:
                    issues.append(ConfigIssue(
                        message=(f"interlayer_style {'/'.join(missing)!s} has no "
                                 f"bundled parameter file for {sorted(elements)}: "
                                 "choose one."),
                        section='general', key='interlayer_file'))
    return issues
