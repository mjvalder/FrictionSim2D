"""Parsing and lookup for classical potential files.

Split out of src/core/utils.py: these five helpers were the only part of that
module that knew about potential file formats. Direction of dependency is
potentials -> core, never the reverse.
"""
import logging
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from ..core.utils import (  # noqa: PLC2701
    cifread,
    count_atomtypes,
    _find_existing,
    _normalize_element_symbol,
)

# UFF_params does not move until Task 8, which repoints this to
# `from . import uff as lj`.
from ..core import UFF_params as lj

logger = logging.getLogger(__name__)

def get_potential_path(pot_name: str,
                       base_dir: Optional[Union[str, Path]] = None) -> Path:
    """Find potential files.

    Args:
        pot_name: Potential name or file path.
        base_dir: Directory to resolve relative paths against (normally the
            directory holding the config file). Optional; when omitted the
            behaviour is the historical current-working-directory lookup.

    Returns:
        Path to the potential file.

    Raises:
        FileNotFoundError: If the potential file is not found.
    """
    found = _find_existing(pot_name, base_dir)
    if found is not None:
        return found

    raise FileNotFoundError(
        f"Potential file '{pot_name}' not found. "
        f"Please provide a valid path to a potential file. "
        f"For a collection of potential files (sw, tersoff, rebo, reaxff, etc.), "
        f"check the 'examples/potentials' folder in the GitHub repository: "
        f"https://github.com/Mclat16/FrictionSim2D"
    )

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
