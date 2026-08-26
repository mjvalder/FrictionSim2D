"""Shared vocabulary and helpers for classical potential files.

Holds three things that no single consumer owns: the parsing/lookup helpers
split out of src/core/utils.py, the potential-type vocabulary formerly in
potential_manager.py, and the sweep type resolver formerly in
src/core/potential_resolve.py.

Keeping them here rather than in manager.py is what makes the package acyclic:
manager.py imports this module, so anything this module needed from manager
would close a loop. Direction of dependency is potentials -> core, never the
reverse.
"""
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from ..core.utils.paths import _find_existing  # noqa: PLC2701
from ..core.utils.structures import _normalize_element_symbol, cifread, count_atomtypes  # noqa: PLC2701

from . import uff as lj

logger = logging.getLogger(__name__)


def lj_params(atom_type_1: str, atom_type_2: str) -> Tuple[float, float]:
    """Calculate LJ parameters using Lorentz-Berthelot mixing rules.

    Pulls UFF parameters and applies mixing rules to determine interaction
    parameters between two atom types. Atom types are normalized to canonical
    element symbol case (e.g., 'nb'/'NB'/'nB' -> 'Nb').

    Args:
        atom_type_1: Symbol of the first atom type (e.g., 'C', 'c', 'H').
        atom_type_2: Symbol of the second atom type (e.g., 'H', 'h').

    Returns:
        Tuple of (epsilon, sigma) - potential well depth and zero-potential distance.
        
    Raises:
        KeyError: If atom type is not found in LJ parameters database.
    """
    # Normalize to canonical symbol case so two-letter elements map correctly
    key1 = _normalize_element_symbol(atom_type_1)
    key2 = _normalize_element_symbol(atom_type_2)

    try:
        e1 = lj.lj_params[key1][1]
        e2 = lj.lj_params[key2][1]
        s1 = lj.lj_params[key1][0]
        s2 = lj.lj_params[key2][0]
    except KeyError as e:
        raise KeyError(
            f"Atom type '{e.args[0]}' not found in LJ parameters database. "
            f"Available types: {', '.join(lj.lj_params.keys())}"
        ) from e

    epsilon = math.sqrt(e1 * e2)
    sigma = (s1 + s2) / 2
    return epsilon, sigma

def get_potential_element_order(
    potential_filepath: Union[str, Path],
    pot_type: str,
    cif_elements: List[str]
) -> Tuple[List[str], Dict[str, int]]:
    """Parse potential file to determine element order and validate CIF compatibility.

    Uses both CIF and potential metadata to build a robust, validated mapping.
    Fails early if CIF and potential have mismatched elements.

    Args:
        potential_filepath: Path to potential file.
        pot_type: Potential type (sw, tersoff, rebo, reaxff, etc.).
        cif_elements: Elements from CIF file (source of truth for identity).

    Returns:
        Tuple of (pot_element_order, pot_element_counts) where:
        - pot_element_order: Ordered list of elements expected by potential
        - pot_element_counts: {element: num_types} for each element

    Raises:
        ValueError: If CIF elements don't match potential elements or validation fails.
    """
    pot_lower = pot_type.lower()
    single_type_pots = {'reaxff', 'reax/c', 'rebo', 'rebomos', 'airebo', 'meam'}
    single_type_exts = ('.rebo', '.rebomos', '.airebo', '.meam', '.reaxff')

    if pot_lower in single_type_pots or str(potential_filepath).lower().endswith(single_type_exts):
        pot_element_counts = {el: 1 for el in cif_elements}
        return cif_elements, pot_element_counts

    pattern = re.compile(r'([A-Za-z]+)(\d*)')
    pot_element_counts = {el: 0 for el in cif_elements}
    cif_set = set(cif_elements)

    with open(potential_filepath, 'r', encoding='utf-8') as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith('#') or not stripped:
                continue

            parts = stripped.split()
            for element_token in parts[:3]:
                match = pattern.match(element_token)
                if match:
                    element_name = match.group(1)
                    element_number = int(match.group(2)) if match.group(2) else 1

                    if element_name not in cif_set:
                        raise ValueError(
                            f"Potential contains element '{element_name}' not in CIF (CIF has {cif_elements})"
                        )

                    pot_element_counts[element_name] = max(
                        pot_element_counts[element_name], element_number
                    )

    missing = [el for el, count in pot_element_counts.items() if count == 0]
    if missing:
        raise ValueError(
            f"CIF has element(s) {missing} but potential has no definition for them."
        )

    pot_element_order = list(cif_elements)
    return pot_element_order, pot_element_counts

def check_potential_cif_compatibility(cif_path: Union[str, Path],
                                        pot_path: Union[str, Path]) -> float:
    """Check if the number of atom types per element is compatible.

    Compares the number of atom types for each element defined in a CIF file
    versus a potential file. Returns a multiplier if the potential defines
    a consistent multiple of atom types compared to the CIF.

    Args:
        cif_path: Path to the CIF file.
        pot_path: Path to the potential file.

    Returns:
        The consistent ratio of atom types in the potential vs. the CIF.

    Raises:
        ValueError: If the ratio of atom types is not consistent across all elements.
    """
    cif_data = cifread(cif_path)
    potentials_count = count_atomtypes(pot_path, cif_data['elements'])

    multiples = {
        element: (potentials_count.get(element, 0) / cif_count
                    if cif_count > 0 and potentials_count.get(element, 0) != 1 else 1)
        for element, cif_count in cif_data['elem_comp'].items()
    }

    unique_multiples = set(multiples.values())
    if len(unique_multiples) > 1:
        raise ValueError(
            f'Inconsistent atom type multiplier between potential and CIF file. '
            f'Ratios found: {unique_multiples}'
        )

    return unique_multiples.pop()

def normalize_potential_type(potential_type: str) -> str:
    """Normalize a potential type string for comparison and validation.
    
    Converts to lowercase and strips whitespace for case-insensitive comparison.
    
    Args:
        potential_type: Potential type string (e.g., 'SW', 'reaxff', 'ReaxFF')
    
    Returns:
        Normalized lowercase potential type string.
    
    Example:
        >>> normalize_potential_type('ReaxFF')
        'reaxff'
        >>> normalize_potential_type('  SW  ')
        'sw'
    """
    return potential_type.strip().lower()


# ---------------------------------------------------------------------------
# Potential-type vocabulary
# ---------------------------------------------------------------------------
#: Lives here rather than in manager.py because three modules need it and none
#: owns it: manager.py, the resolver below, and src/builders/sheetonsheet.py --
#: which previously imported the whole 1,000-line manager to read a six-element
#: set.
POTENTIALS_WITH_INTERNAL_LJ = {'airebo', 'rebomos', 'comb', 'comb3', 'reaxff', 'reax/c'}
POTENTIALS_REQUIRING_LJ = {'sw', 'tersoff', 'rebo', 'edip', 'meam', 'eam', 'bop',
                            'morse',  'sw/mod', 'extep', 'vashishta'}
REAXFF_TYPES = {'reaxff', 'reax/c'}
KNOWN_TYPES = POTENTIALS_REQUIRING_LJ | POTENTIALS_WITH_INTERNAL_LJ


# ---------------------------------------------------------------------------
# Potential-sweep type resolution (merged in from src/core/potential_resolve.py)
# ---------------------------------------------------------------------------

#: Extension -> BASE family only. No entry ever yields a type with '/'.
EXTENSION_MAP = {
    '.sw': 'sw',
    '.tersoff': 'tersoff',
    '.rebo': 'rebo',
    '.rebomos': 'rebomos',
    '.airebo': 'airebo',
    '.meam': 'meam',
    '.extep': 'extep',
    '.vashishta': 'vashishta',
}


class PotentialTypeError(ValueError):
    """A sweep entry's potential type could not be resolved unambiguously."""


@dataclass(frozen=True)
class ResolvedPotential:
    path: str
    pot_type: str
    route: str   # 'explicit' | 'directory' | 'extension'


# Potential-sweep type resolution (spec §13.3).
#
# Neither the file extension nor the directory determines the pair style,
# because LAMMPS variants share both: ``.sw`` serves ``sw`` AND ``sw/mod``;
# ``.tersoff`` serves ``tersoff``/``tersoff/mod``/...; ``.reax.*`` serves
# ``reaxff`` and legacy ``reax/c``. Resolution order per entry:
#
#     1. explicit ``pot_type`` on the entry (always wins)
#     2. parent directory under the potentials root, if it names a known type
#     3. file extension, through EXTENSION_MAP
#     4. otherwise FAIL THE BUILD
#
# Variant styles (anything containing ``/``) are NEVER inferred. This matters
# because the legacy fallback is not an error: ``is_sheet_lj`` logs "Unknown
# potential type, assuming LJ is required" and proceeds — under a sweep that
# would silently emit the wrong pair_style for an entire arm.
def resolve_pot_type(entry: Union[str, dict],
                     potentials_root: Path) -> ResolvedPotential:
    if isinstance(entry, dict):
        path = entry['path']
        explicit = entry.get('pot_type')
    else:
        path = entry
        explicit = None

    if explicit:
        pot_type = explicit.strip().lower()
        if pot_type not in KNOWN_TYPES:
            raise PotentialTypeError(
                f"{path}: explicit pot_type '{explicit}' is not a known "
                f"LAMMPS style (known: {sorted(KNOWN_TYPES)})")
        return ResolvedPotential(path, pot_type, 'explicit')

    parts = Path(path).parts
    for part in parts[:-1]:
        candidate = part.strip().lower()
        if candidate in KNOWN_TYPES and '/' not in candidate:
            return ResolvedPotential(path, candidate, 'directory')

    suffix = Path(path).suffix.lower()
    name = Path(path).name.lower()
    if '.reax.' in name or name.startswith('ffield'):
        raise PotentialTypeError(
            f"{path}: ReaxFF files serve both 'reaxff' and legacy 'reax/c' — "
            "state pot_type explicitly")
    if suffix in EXTENSION_MAP:
        return ResolvedPotential(path, EXTENSION_MAP[suffix], 'extension')

    raise PotentialTypeError(
        f"{path}: cannot resolve potential type from directory or extension "
        f"'{suffix}'. State pot_type explicitly.")


def pot_dir_token(pot_type: str, path: Union[str, Path]) -> str:
    """Output-directory segment: pot_<type-slug>_<stem> (spec §13.4).

    '/' becomes '-' (sw/mod -> sw-mod); the stem is sanitised so the token
    is always a single path segment.
    """
    slug = pot_type.strip().lower().replace('/', '-')
    stem = Path(path).name
    for ext in sorted(EXTENSION_MAP, key=len, reverse=True):
        if stem.lower().endswith(ext):
            stem = stem[:-len(ext)]
            break
    # No known extension (e.g. ffield_ost.reax.MoS): keep the full name —
    # for ReaxFF files the trailing token is identity, not an extension.
    stem = ''.join(c if (c.isalnum() or c in '-_.') else '-' for c in stem)
    return f'pot_{slug}_{stem}'
