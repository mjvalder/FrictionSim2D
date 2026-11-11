"""Interlayer-interaction vocabulary: styles, parameter files and validation.

The interaction BETWEEN the layers of a 2D stack can be the run's normal
Lennard-Jones (``lj``), LJ plus the Jiang-Park Gaussian (``lj_g``), or a
registered many-body style (``ilp``, ``kc``). This module holds the vocabulary
and the checks; emission lives in :mod:`src.potentials.manager`.

Must not import ``ase`` (``tests/repo/test_import_weight.py``): the CIF read in
:func:`check_interlayer_choice` is a deferred import.
"""
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Set, Tuple

from ..core.utils.paths import get_material_path, get_potential_path

InterlayerStyle = Literal['lj', 'lj_g', 'ilp', 'kc']
INTERLAYER_STYLES: Tuple[str, ...] = ('lj', 'lj_g', 'ilp', 'kc')

#: Jiang-Park Gaussian strength g (meV) by material, structure prefix stripped.
GAUSSIAN_G_MEV: Dict[str, float] = {
    'MoS2': 175.68, 'graphene': 94.87, 'C': 94.87, 'BP': 123.0, 'P': 123.0,
}

ILP_CUTOFF = 16.0
KC_FULL_CUTOFF = 16.0
TAPER_FLAG = 1

EXAMPLES_POTENTIALS_DIR = Path(__file__).resolve().parents[2] / 'examples' / 'potentials'

_ILP_TMD = {'Mo', 'W', 'S', 'Se'}
_ILP_BNCH = {'B', 'N', 'C', 'H'}


def _strip_prefix(mat: str) -> str:
    """Drop the structure prefix (``h_``, ``t-``, ...) from a material name."""
    if len(mat) > 2 and mat[0] in 'htbp' and mat[1] in '_-':
        return mat[2:]
    return mat


def gaussian_g(mat: str, override: Optional[float]) -> Optional[float]:
    """Gaussian strength (meV) for ``mat``: the override, else the table, else None."""
    if override is not None:
        return float(override)
    return GAUSSIAN_G_MEV.get(_strip_prefix(mat))


def mixed_g(g1: float, g2: float) -> float:
    """Geometric-mean Gaussian strength for a heterostructure interface."""
    return (g1 * g2) ** 0.5


def default_interlayer_file(style: str, elements: Set[str]) -> Optional[str]:
    """Bundled parameter-file name for ``style`` over ``elements``, or None."""
    elems = set(elements)
    if style == 'ilp':
        if elems <= _ILP_TMD:
            return 'TMD.ILP'
        if elems <= _ILP_BNCH:
            return 'BNCH.ILP'
    elif style == 'kc':
        if elems <= {'C'}:
            return 'CC.KC-full'
        if elems <= {'C', 'H'}:
            return 'CH.KC'
    return None


def ilp_substyle(elements: Set[str]) -> str:
    """LAMMPS ILP sub-style for ``elements``."""
    return 'ilp/tmd' if set(elements) <= _ILP_TMD else 'ilp/graphene/hbn'


def resolve_interlayer_file(name: str, style: str,
                            base_dir: Optional[Path]) -> Path:
    """Find an interlayer parameter file like any other potential.

    Looks relative to ``base_dir``/working directory first, then in the bundled
    ``examples/potentials/<ilp|kc>/``.

    Raises:
        FileNotFoundError: Naming both places searched.
    """
    try:
        return get_potential_path(name, base_dir)
    except FileNotFoundError:
        pass
    bundled_dir = EXAMPLES_POTENTIALS_DIR / ('ilp' if style == 'ilp' else 'kc')
    bundled = bundled_dir / Path(name).name
    if bundled.exists():
        return bundled
    searched = base_dir if base_dir is not None else Path.cwd()
    raise FileNotFoundError(
        f"Interlayer file '{name}' not found in {searched} (working directory / "
        f"config directory) or in the bundled {bundled_dir}.")


def _sheet_elements(sheets, base_dir: Optional[Path]) -> Optional[Set[str]]:
    """Union of the CIF elements of ``sheets``; None if a CIF cannot be read."""
    from ..core.utils.structures import cifread  # pylint: disable=import-outside-toplevel
    found: Set[str] = set()
    for sheet in sheets:
        try:
            path = Path(sheet.cif_path)
            if not path.exists():
                path = get_material_path(sheet.cif_path, base_dir)
            found |= set(cifread(path)['elements'])
        except Exception:  # pylint: disable=broad-exception-caught
            return None
    return found


def check_interlayer_choice(config: Any, base_dir: Optional[Path] = None) -> None:
    """Reject interlayer settings that cannot produce a valid run.

    Raises:
        ValueError: On a non-``lj`` style combined with ``constraint_mode``,
            an internal-LJ potential, a missing Gaussian ``g`` or a missing
            parameter file that cannot be defaulted.
    """
    # pylint: disable=import-outside-toplevel
    from .utils import POTENTIALS_WITH_INTERNAL_LJ
    general = config.general
    style = general.interlayer_style
    if style == 'lj':
        return

    mode = config.settings.simulation.constraint_mode
    if mode != 'none':
        raise ValueError(
            f"interlayer_style '{style}' is incompatible with constraint_mode "
            f"'{mode}': the many-body/Gaussian layers need the f_fsbot reaction "
            "force, not the atom_bonds/com_spring constraint. Set "
            "constraint_mode = none or interlayer_style = lj.")

    sheets = list(config.sheets) or [config.sheet]
    sheets = sheets + list(config.flake_sheets)
    for sheet in sheets:
        if sheet.pot_type in POTENTIALS_WITH_INTERNAL_LJ:
            raise ValueError(
                f"interlayer_style '{style}' cannot be combined with pot_type "
                f"'{sheet.pot_type}' ({sheet.mat}): it carries its own interlayer "
                "term, so the choice would be a silent no-op.")

    if style == 'lj_g':
        missing = [s.mat for s in sheets
                   if gaussian_g(s.mat, s.interlayer_g) is None]
        if missing:
            raise ValueError(
                f"interlayer_style 'lj_g' needs a Gaussian strength g (meV) for "
                f"{sorted(set(missing))}: set interlayer_g in the sheet section "
                f"(built in: {sorted(GAUSSIAN_G_MEV)}).")
        return

    if general.interlayer_file is None:
        elements = _sheet_elements(sheets, base_dir)
        if elements is not None and default_interlayer_file(style, elements) is None:
            raise ValueError(
                f"interlayer_style '{style}' has no bundled parameter file for "
                f"elements {sorted(elements)}: set general interlayer_file.")
