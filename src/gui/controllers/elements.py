"""Scanning which elements a configured simulation actually uses.

The LJ override table's keys are element pairs the user must type by hand --
"Mo-S", not a material name -- so there is nothing in the schema to validate
them against. This module answers a narrower, answerable question: which
elements does *this* config actually place atoms of? That is enough to turn
the pair column into dropdowns and to flag a pair scanning never saw.

Scanning is advisory, never authoritative: an unreadable or missing CIF is
skipped rather than raised, the same way an in-progress edit leaves most of
the form incomplete. Qt-free, so it is exercised directly, without a display
-- callers decide when it is cheap enough to run (see `gui.widgets.form`,
which only calls this on a settled edit, never per keystroke).

Parses CIF text directly with a handful of regexes rather than going through
`cifread` in `core.utils` (the ASE CIF reader): a scan for element symbols
does not need a full structure parse. `ase.data.atomic_numbers` is reused for
the known-symbols table (already a project dependency; see
`potentials.manager`), not a new one.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any, List, Optional, Set, Tuple


from ...run.placeholders import has_template_marker, substitute_materials
from ...core.utils.paths import _find_existing  # noqa: PLC2701

__all__ = ['scan_config_elements']

#: Component sections that may carry a `cif_path`/`mat` pair, whichever the
#: document actually has. Most models only populate a few of these (`sub`
#: and `tip` for AFM-like models, just `2D` for sheet-on-sheet); `2D-2` and
#: `flake` are included for forward compatibility with configs this GUI
#: version does not itself build but may still hold in `document.values`.
_COMPONENT_SECTIONS = ('2D', '2D-2', 'tip', 'sub', 'flake')

#: Sections a material sweep templates. Only the sheets: the list names sheet
#: materials, so unioning a templated tip or substrate over it would scan
#: paths the sweep never builds.
_SWEEPING_SECTIONS = ('2D', '2D-2')

@lru_cache(maxsize=1)
def _known_symbols() -> frozenset:
    """Every element symbol ASE knows, loaded on first use.

    ``ase.data`` costs ~74 ms and this module is on the GUI startup path
    (``widgets/form.py`` imports it), while the symbol table is only needed
    once a config is actually scanned. Cached: the set is built once.
    """
    from ase.data import atomic_numbers  # pylint: disable=import-outside-toplevel

    return frozenset(atomic_numbers)

_ELEMENT_TOKEN_RE = re.compile(r'[A-Z][a-z]?')
_ATOM_SITE_SYMBOL_TAG = '_atom_site_type_symbol'
_FORMULA_SUM_RE = re.compile(r'_chemical_formula_sum\s+["\']?([^"\'\n]+)')


def scan_config_elements(document: Any) -> List[str]:
    """Element symbols placed by every component section the document has.

    For a templated sheet section with a configured material sweep, the union
    is taken over every material the sweep names (that sheet's own member of
    each entry), not just the one literal path in the section -- a shared
    override file is meant to cover a whole sweep, so the scan has to see
    every element the sweep could actually place.

    That covers a heterostructure pair too: `[2D]` templates on `{mat1}` and
    `[2D-2]` on `{mat2}`, and each takes its own column of the pair list.
    Testing for the literal `'{mat}'` instead left a pair config falling
    through to a path that cannot exist (`.../{mat1}.cif`) with no element
    tokens in `'{mat1}'` to fall back on, so the LJ pair picker came back
    empty for exactly the configs that need it most.
    """
    base_dir = getattr(document, 'base_dir', None)
    elements: Set[str] = set()

    for section_key in _COMPONENT_SECTIONS:
        if not document.has_section(section_key):
            continue
        section_values = document.values.get(section_key) or {}
        cif_template = section_values.get('cif_path')
        mat_template = section_values.get('mat')
        if not isinstance(cif_template, str):
            cif_template = ''
        # A missing/blank cif_path must not skip the section outright: the
        # third-tier `mat`-token fallback (_scan_component, below) can still
        # find something in `mat` on its own, and only continue-ing here
        # when *both* are unusable kept that fallback unreachable.
        if not cif_template and not isinstance(mat_template, str):
            continue

        entries: List[Tuple[str, ...]] = []
        if section_key in _SWEEPING_SECTIONS and has_template_marker(cif_template):
            entries = _sweep_entries(document)

        if entries:
            for entry in entries:
                # Every section is substituted with the whole entry, so each
                # takes whichever member its own placeholder names: `{mat1}`
                # in `[2D]`, `{mat2}` in `[2D-2]`. A single-material entry
                # answers for both.
                mat1 = entry[0]
                mat2 = entry[1] if len(entry) > 1 else entry[0]
                mat_text = (substitute_materials(mat_template, mat1, mat2)
                            if isinstance(mat_template, str) else mat1)
                elements |= _scan_component(
                    substitute_materials(cif_template, mat1, mat2),
                    mat_text, base_dir)
        else:
            elements |= _scan_component(cif_template, mat_template, base_dir)

    return sorted(elements)


def _sweep_entries(document: Any) -> List[Tuple[str, ...]]:
    """The sweep's entries -- singles or ordered pairs -- or [] when unreadable.

    Asked of the document rather than re-read here, so the scan sees exactly
    the list the run would. `material_entries` is the pair-aware reader, and
    already returns 1-tuples for a single-material list.
    """
    return list(document.material_entries())


def _scan_component(cif_path: str, mat_text: Optional[str],
                    base_dir: Optional[Path]) -> Set[str]:
    """Elements for one component: its CIF if readable, else its `mat` text."""
    found = _find_existing(cif_path, base_dir) if cif_path else None
    if found is not None:
        try:
            text = found.read_text(encoding='utf-8', errors='ignore')
        except OSError:
            text = None
        if text is not None:
            elements = _parse_cif_text(text)
            if elements:
                return elements

    return _elements_from_text(mat_text)


def _parse_cif_text(text: str) -> Set[str]:
    elements = _parse_atom_site_loop(text)
    if elements:
        return elements
    return _parse_formula_sum(text)


def _parse_atom_site_loop(text: str) -> Set[str]:
    """Element symbols from the `_atom_site_type_symbol` loop column.

    A minimal CIF ``loop_`` reader: a tag block (lines starting with `_`)
    followed by whitespace-separated data rows until the next tag, loop, or
    blank line. Good enough for the CIFs this project writes (ASE's own CIF
    writer, see examples/materials/*.cif) without pulling in a real parser.
    """
    lines = text.splitlines()
    i, n = 0, len(lines)
    while i < n:
        if lines[i].strip() != 'loop_':
            i += 1
            continue
        i += 1
        tags = []
        while i < n and lines[i].strip().startswith('_'):
            tags.append(lines[i].strip())
            i += 1
        if _ATOM_SITE_SYMBOL_TAG not in tags:
            continue
        index = tags.index(_ATOM_SITE_SYMBOL_TAG)
        elements: Set[str] = set()
        while i < n:
            row = lines[i].strip()
            if not row or row.startswith('_') or row == 'loop_' or row.startswith('data_'):
                break
            tokens = row.split()
            if len(tokens) > index:
                match = re.match(r'[A-Za-z]+', tokens[index])
                if match:
                    symbol = match.group(0).capitalize()
                    if symbol in _known_symbols():
                        elements.add(symbol)
            i += 1
        if elements:
            return elements
    return set()


def _parse_formula_sum(text: str) -> Set[str]:
    match = _FORMULA_SUM_RE.search(text)
    if not match:
        return set()
    return _elements_from_text(match.group(1))


def _elements_from_text(text: Optional[str]) -> Set[str]:
    """Element-like tokens in free text, filtered against known symbols.

    Used both for `_chemical_formula_sum` ("Mo1 S2") and as the final
    fallback against a `mat` field ("h_MoS2") when no CIF could be read.
    """
    if not isinstance(text, str) or not text:
        return set()
    return {tok for tok in _ELEMENT_TOKEN_RE.findall(text) if tok in _known_symbols()}
