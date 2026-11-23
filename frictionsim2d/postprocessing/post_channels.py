"""Post-stream observable channels, derived from the retained final state.

The registry declares four channels with ``stream='post'`` and no
``lammps_compute``: ``gap_min``/``deflect_max`` (afm, indent) and
``gap_min_il``/``corrugation_max`` (sheetonsheet, hetero, flake). They are
DERIVED, not emitted -- and until now nothing derived them, so ``role_key``,
``units`` and the whole post stream had no consumer at all.

The ingredient is the ``data/final_*.data`` file the slide templates write
(``frictionsim2d/templates/afm/slide.lmp``, ``frictionsim2d/templates/sheetonsheet/slide.lmp``)
plus the run's ``provenance.json``: ``layer_type_map`` says which atom types
are which sheet layer, ``unloaded_reference_z`` says where each layer sat
before the load was applied.

The atom reader lives here rather than in ``builders/utils/lammps_data.py``:
that module holds style conversion and box dimensions, not per-atom
coordinates, and it is a BUILDER utility -- ``frictionsim2d/postprocessing/`` reaching
into ``frictionsim2d/builders/`` for one twenty-line parser is a worse edge than owning
the parser.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

__all__ = ['AFM_LIKE_FAMILIES', 'read_atom_z', 'final_data_path',
           'post_channels', 'post_channels_for']

#: Families whose final state holds a tip above a sheet.
AFM_LIKE_FAMILIES = ('afm', 'indent', 'pes_tip')

#: `final_f30N_a0.data` / `final_p1.5GPa_a-15.data`, as the templates spell them.
_FINAL_RE = re.compile(
    r'^final_(?P<kind>[fp])(?P<load>-?\d+\.?\d*(?:[eE][+-]?\d+)?)'
    r'(?P<unit>N|GPa)_a(?P<angle>-?\d+\.?\d*(?:[eE][+-]?\d+)?)\.data$')

#: The static PES scans write ONE `final_pes.data` at the end of the grid.
_FINAL_PES = 'final_pes.data'

_SECTION_RE = re.compile(r'^[A-Z][A-Za-z ]*$')


def read_atom_z(path: Path) -> Dict[int, List[float]]:
    """`{atom type: [z, ...]}` from a LAMMPS data file's Atoms section.

    Style-agnostic: the z coordinate is the LAST of the three trailing floats
    on the line, and the atom type is the second integer field. Both hold for
    `atomic`, `charge` and `molecular` styles, which is every style
    `builders/utils/lammps_data.py` converts between.
    """
    by_type: Dict[int, List[float]] = {}
    in_atoms = False
    for raw in Path(path).read_text(encoding='utf-8',
                                    errors='ignore').splitlines():
        line = raw.split('#')[0].strip()
        if not line:
            continue
        if _SECTION_RE.match(line):
            in_atoms = line.startswith('Atoms')
            continue
        if not in_atoms:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            atom_type = int(parts[1])
            z = float(parts[-1])
        except ValueError:
            continue
        by_type.setdefault(atom_type, []).append(z)
    return by_type


def final_data_path(data_dir: Path, load_kind: str, load: float,
                    angle: float) -> Optional[Path]:
    """The `final_*.data` file for one load point, or None.

    Matched on parsed VALUES, not on a formatted filename: LAMMPS writes
    `$(v_find)` with its own `%.10g`, so 30.0 lands as `30` and re-formatting
    the number here would miss the file for exactly the runs that have one.
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        return None
    if load_kind == 'none':
        # PES point: one end-of-scan file, load and angle do not apply.
        pes = data_dir / _FINAL_PES
        return pes if pes.is_file() else None
    for path in sorted(data_dir.iterdir()):
        match = _FINAL_RE.match(path.name)
        if match is None:
            continue
        kind = 'force' if match.group('unit') == 'N' else 'pressure'
        if kind != load_kind:
            continue
        if abs(float(match.group('load')) - float(load)) > 1e-9:
            continue
        if abs(float(match.group('angle')) - float(angle)) > 1e-9:
            continue
        return path
    return None


def _layer_bands(by_type: Dict[int, List[float]],
                 layer_type_map) -> List[List[float]]:
    """Each layer's z values, ordered bottom to top by mean z."""
    bands = []
    for types in layer_type_map.values():
        zs = [z for t in types for z in by_type.get(int(t), [])]
        if zs:
            bands.append(zs)
    return sorted(bands, key=lambda zs: sum(zs) / len(zs))


def post_channels(data_file: Path, record, family: str) -> Dict[str, float]:
    """The post-stream channels for one final state, in Angstrom.

    Returns `{}` rather than guessing whenever an ingredient is missing: an
    empty `layer_type_map` (hetero and flake partition by z-band, not by atom
    type, and honestly write `{}`), an unreadable file, or -- for
    `deflect_max` -- no `unloaded_reference_z`, which is enriched post-run
    from the log and absent on a run whose log was not parsed.
    """
    layer_type_map = dict(getattr(record, 'layer_type_map', {}) or {})
    if not layer_type_map:
        return {}
    by_type = read_atom_z(data_file)
    if not by_type:
        return {}
    bands = _layer_bands(by_type, layer_type_map)
    if not bands:
        return {}

    out: Dict[str, float] = {}
    top = bands[-1]

    if family in AFM_LIKE_FAMILIES:
        sheet_z = [z for band in bands for z in band]
        sheet_top = max(sheet_z)
        layer_types = {int(t) for types in layer_type_map.values()
                       for t in types}
        tip_z = [z for t, zs in by_type.items() if t not in layer_types
                 for z in zs if z > sheet_top]
        if tip_z:
            out['gap_min'] = float(min(tip_z) - sheet_top)
        refs = dict(getattr(record, 'unloaded_reference_z', None) or {})
        if refs:
            out['deflect_max'] = float(max(refs.values()) - min(top))
        return out

    if len(bands) >= 2:
        out['gap_min_il'] = float(min(
            min(upper) - max(lower)
            for lower, upper in zip(bands, bands[1:])))
    out['corrugation_max'] = float(max(top) - min(top))
    return out


def post_channels_for(locator, load_kind: str, load: float,
                      angle: float) -> Dict[str, float]:
    """`post_channels` for one load point of one `RunLocator`."""
    from ..data.provenance import read_provenance  # noqa: PLC0415  # pylint: disable=import-outside-toplevel

    path = final_data_path(locator.data_dir, load_kind, load, angle)
    if path is None:
        return {}
    tokens = locator.tokens
    try:
        record = read_provenance(
            locator.provenance_file, family=tokens.family, model=tokens.model,
            material_name=tokens.materials[0] if tokens.materials else '')
    except (OSError, ValueError) as exc:
        logger.warning("No provenance for %s (%s); post channels skipped",
                       tokens.rel_path, exc)
        return {}
    return post_channels(path, record, tokens.family)
