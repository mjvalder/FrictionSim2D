"""Shared vocabulary and helpers for classical potential files.

Holds three things that no single consumer owns: the parsing/lookup helpers
split out of frictionsim2d/core/utils.py, the potential-type vocabulary formerly in
potential_manager.py, and the sweep type resolver formerly in
frictionsim2d/core/potential_resolve.py.

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
from typing import Dict, List, Optional, Tuple, Union

from ..core.mlip import MLIP_MODEL_EXTENSIONS, MLIP_REGISTRY, is_mlip
from ..core.utils.structures import (
    cifread, count_atomtypes, SINGLE_TYPE_EXTENSIONS, SINGLE_TYPE_POTENTIALS,
)
from ..core.utils.tokens import normalize_element_symbol

from ..interfaces import lammps as _lammps
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
    key1 = normalize_element_symbol(atom_type_1)
    key2 = normalize_element_symbol(atom_type_2)

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
    # Shared with count_atomtypes: MLIP model files are binary (or a symmetrix
    # .json) and must never be parsed as a classical potential file.
    if (pot_lower in SINGLE_TYPE_POTENTIALS
            or str(potential_filepath).lower().endswith(SINGLE_TYPE_EXTENSIONS)):
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
                            f"Potential contains element '{element_name}' not in CIF "
                            f"(CIF has {cif_elements})"
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
#: owns it: manager.py, the resolver below, and frictionsim2d/builders/sheetonsheet.py --
#: which previously imported the whole 1,000-line manager to read a six-element
#: set.
POTENTIALS_WITH_INTERNAL_LJ = {'airebo', 'rebomos', 'comb', 'comb3', 'reaxff', 'reax/c'}
POTENTIALS_REQUIRING_LJ = {'sw', 'tersoff', 'rebo', 'edip', 'meam', 'eam', 'bop',
                            'morse',  'sw/mod', 'extep', 'vashishta'}
REAXFF_TYPES = {'reaxff', 'reax/c'}
#: Machine-learned interatomic potentials (MACE, DeePMD), derived from the
#: frictionsim2d.core.mlip registry. A third category: neither "requires LJ" nor
#: "internal LJ" -- an MLIP cell is full-cell only (documentation/mlip.md).
MLIP_TYPES = frozenset(MLIP_REGISTRY)
KNOWN_TYPES = POTENTIALS_REQUIRING_LJ | POTENTIALS_WITH_INTERNAL_LJ | MLIP_TYPES


# ---------------------------------------------------------------------------
# Potential-sweep type resolution (merged in from frictionsim2d/core/potential_resolve.py)
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

#: MLIP model-file extensions (MLIPSpec.file_extensions). Deliberately NOT in
#: EXTENSION_MAP: an MLIP is never inferred from a file. '.pth' is also an
#: older NequIP/Allegro deployment and '.json' could become another code's
#: format, so a guess could pick the wrong framework; an MLIP needs an explicit
#: pot_type. Used only to trim the extension from the pot_ directory token.
MLIP_FILE_EXTENSIONS = frozenset(MLIP_MODEL_EXTENSIONS)


class PotentialTypeError(ValueError):
    """A sweep entry's potential type could not be resolved unambiguously."""


@dataclass(frozen=True)
class ResolvedPotential:
    """A sweep entry's potential file with the pair style it resolved to and how."""
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
#        (an MLIP is never inferred by 2 or 3: it must be explicit)
#     4. otherwise FAIL THE BUILD
#
# Variant styles (anything containing ``/``) are NEVER inferred. This matters
# because the legacy fallback is not an error: ``is_sheet_lj`` logs "Unknown
# potential type, assuming LJ is required" and proceeds — under a sweep that
# would silently emit the wrong pair_style for an entire arm.
def _accept_custom_type(normalised: str, raw: str, source: str) -> None:
    """Allow a pair style outside KNOWN_TYPES iff the installed LAMMPS lists it.

    With no LAMMPS binary to ask, the style is accepted with a warning. Either
    way it is treated as an unknown-LJ style downstream.

    Raises:
        PotentialTypeError: If LAMMPS is available and does not list it.
    """
    installed = _lammps.lammps_pair_styles()
    if installed is None:
        logger.warning(
            "%s '%s' is not a built-in style and no LAMMPS binary was found to "
            "check it against; accepting it as written.", source, raw)
        return
    if normalised not in installed:  # pylint: disable=unsupported-membership-test
        raise PotentialTypeError(
            f"{source} '{raw}' is not a known LAMMPS pair style "
            f"(known: {sorted(KNOWN_TYPES)}) nor one of the installed "
            f"LAMMPS pair styles. It would be written straight into pair_style and the run "
            f"would fail at LAMMPS start-up."
        )
    logger.warning(
        "%s '%s' is a custom pair style (installed in LAMMPS, not built in); "
        "it is not validated further.", source, raw)


def validate_pot_type(value: str, *, source: str) -> str:
    """Return ``value`` normalised, or raise if it is not a known LAMMPS style.

    The same :data:`KNOWN_TYPES` gate :func:`resolve_pot_type` applies to a sweep
    entry, exposed for the plain ``[section] pot_type`` route. Until 2026-09-09
    only the sweep route was checked, so an unknown style on the plain route
    reached ``PotentialManager.is_sheet_lj``, which logged a warning, guessed
    "LJ required", and let the string through into
    ``pair_style hybrid sw sw sw <typo> lj/cut 11.0``. The build succeeded and
    LAMMPS rejected the run with "Unknown pair style" -- after the job had been
    transferred, queued and started.

    Args:
        value: The raw ``pot_type`` as written in the config.
        source: What to name in the error, e.g. ``"[2D] pot_type"``.

    Returns:
        The normalised (stripped, lowercased) type.

    Raises:
        PotentialTypeError: If it is not in :data:`KNOWN_TYPES` and is not
            listed by the installed LAMMPS either. A type outside
            :data:`KNOWN_TYPES` is accepted when the installed ``lmp`` lists
            it, or with a warning when no LAMMPS binary is found.
    """
    normalised = value.strip().lower()
    if normalised not in KNOWN_TYPES:
        _accept_custom_type(normalised, value, source)
    return normalised


def resolve_pot_type(entry: Union[str, dict],
                     potentials_root: Path) -> ResolvedPotential:  # pylint: disable=unused-argument
    # potentials_root is accepted for call-site symmetry with the sweep expander; the
    # directory route deliberately matches any parent segment (spec §13.3), not only
    # children of the root.
    if isinstance(entry, dict):
        path = entry['path']
        explicit = entry.get('pot_type')
    else:
        path = entry
        explicit = None

    if explicit:
        pot_type = explicit.strip().lower()
        if pot_type not in KNOWN_TYPES:
            _accept_custom_type(pot_type, explicit, f"{path}: explicit pot_type")
        return ResolvedPotential(path, pot_type, 'explicit')

    parts = Path(path).parts
    suffix = Path(path).suffix.lower()
    mlip_dir = next((p.strip().lower() for p in parts[:-1]
                     if p.strip().lower() in MLIP_TYPES), None)
    if suffix in MLIP_FILE_EXTENSIONS or mlip_dir:
        raise PotentialTypeError(
            f"{path}: looks like an MLIP model; MLIPs are never inferred from a "
            f"folder or extension. State pot_type explicitly "
            f"({' or '.join(sorted(MLIP_TYPES))}).")
    for part in parts[:-1]:
        candidate = part.strip().lower()
        if candidate in KNOWN_TYPES and '/' not in candidate:
            return ResolvedPotential(path, candidate, 'directory')

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


#: Slug suffix marking an MLIP run with the D3 dispersion overlay.
MLIP_D3_SLUG_SUFFIX = '-d3'


def pot_dir_token(pot_type: str, path: Union[str, Path],
                  dispersion: Optional[str] = None) -> str:
    """Output-directory segment: pot_<type-slug>_<stem> (spec §13.4).

    '/' becomes '-' (sw/mod -> sw-mod); the stem is sanitised so the token
    is always a single path segment. For an MLIP ``pot_type`` the slug also
    carries the dispersion choice -- ``mace`` (bare model) or ``mace-d3``
    (``dispersion = d3``) -- so two runs of one model with and without D3
    never share a directory. ``dispersion`` is ignored for classical types.
    """
    slug = pot_type.strip().lower().replace('/', '-')
    if slug in MLIP_TYPES and (dispersion or '').strip().lower() == 'd3':
        slug += MLIP_D3_SLUG_SUFFIX
    stem = Path(path).name
    for ext in sorted({*EXTENSION_MAP, *MLIP_FILE_EXTENSIONS}, key=len, reverse=True):
        if stem.lower().endswith(ext):
            stem = stem[:-len(ext)]
            break
    # No known extension (e.g. ffield_ost.reax.MoS): keep the full name —
    # for ReaxFF files the trailing token is identity, not an extension.
    stem = ''.join(c if (c.isalnum() or c in '-_.') else '-' for c in stem)
    return f'pot_{slug}_{stem}'


# ---------------------------------------------------------------------------
# Pair cutoffs a potential file carries (for the periodic-z vacuum)
# ---------------------------------------------------------------------------
#: ``fix qeq/reaxff 1 0.0 10.0 ...``: the charge-equilibration cutoff every
#: ReaxFF template emits; it reaches across the periodic boundary too.
REAXFF_QEQ_CUTOFF = 10.0

#: LAMMPS ``pair rebomos`` cuts its LJ term at 2.5 sigma.
REBOMOS_LJ_SIGMA_FACTOR = 2.5

#: ``pair_style airebo 3.0``: LJ to 3 sigma_CC (3.4 Angstrom).
AIREBO_LJ_CUTOFF = 3.0 * 3.4


def reaxff_cutoff(path: Union[str, Path]) -> float:
    """Longest ReaxFF range: the upper taper radius (``swb``) or the QEq cutoff.

    ``swb`` is general parameter 13 of the ``ffield`` file (the non-bonded
    vdW + Coulomb taper). Falls back to :data:`REAXFF_QEQ_CUTOFF` when the
    file cannot be parsed.
    """
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as handle:
            lines = handle.readlines()
        swb = float(lines[1 + 13].split()[0])
    except (OSError, IndexError, ValueError):
        logger.warning("Could not read swb from %s; using %.1f A",
                       path, REAXFF_QEQ_CUTOFF)
        swb = 0.0
    return max(swb, REAXFF_QEQ_CUTOFF)


def rebomos_cutoff(path: Union[str, Path]) -> float:
    """REBOMoS LJ range: 2.5 x the largest ``sigma_*`` in the parameter file."""
    sigmas: List[float] = []
    with open(path, 'r', encoding='utf-8', errors='replace') as handle:
        for raw in handle:
            parts = raw.split()
            if len(parts) >= 2 and parts[1].lower().startswith('sigma'):
                try:
                    sigmas.append(float(parts[0]))
                except ValueError:
                    continue
    if not sigmas:
        raise ValueError(f"{path}: no sigma_* lines; cannot size the z vacuum")
    return REBOMOS_LJ_SIGMA_FACTOR * max(sigmas)


def internal_pair_cutoff(pot_type: str, path: Union[str, Path],
                         fallback: float) -> float:
    """The long-range cutoff a many-body potential with its own LJ applies.

    ``fallback`` (the run's ``lj_cutoff``) is used for internal-LJ types
    without a parser here (``comb``, ``comb3``) and for MLIPs (a model's
    cutoff, ~5-6 A for the shipped MACE/DeePMD models, is not read from the
    file); types that need an external LJ are short-ranged and return 0.
    """
    if is_mlip(pot_type):
        return fallback
    ptype = normalize_potential_type(pot_type)
    if ptype in REAXFF_TYPES:
        return reaxff_cutoff(path)
    if ptype == 'rebomos':
        return rebomos_cutoff(path)
    if ptype == 'airebo':
        return AIREBO_LJ_CUTOFF
    if ptype in POTENTIALS_WITH_INTERNAL_LJ:
        return fallback
    return 0.0
