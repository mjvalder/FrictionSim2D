"""Reading structure files and deriving model dimensions from them.

The only module here that reaches ASE, through
:mod:`frictionsim2d.interfaces.ase`. Keep it that way: ``paths``, ``tokens`` and
``lammps_data`` are imported by code that must stay cheap.
"""
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ...interfaces.ase import read_structure
from ..mlip import MLIP_REGISTRY

logger = logging.getLogger(__name__)

#: Model-file extensions of every registered MLIP framework ('.json' symmetrix
#: MACE, '.pb'/'.pth' DeePMD), plus the legacy ACEsuit '.pt'. These files are
#: serialised tensors (or a symmetrix JSON), never a classical potential file.
MLIP_MODEL_EXTENSIONS = tuple(sorted(
    {ext.lower() for spec in MLIP_REGISTRY.values() for ext in spec.file_extensions}
    | {'.pt'}
))

#: Potentials that take exactly one LAMMPS atom type per element. Their files
#: are never parsed for per-element multiplicity -- for an MLIP that would
#: open a binary model as UTF-8 text and crash. Shared with
#: frictionsim2d.potentials.utils.get_potential_element_order.
SINGLE_TYPE_POTENTIALS = frozenset(
    {'reaxff', 'reax/c', 'rebo', 'rebomos', 'airebo', 'meam'} | set(MLIP_REGISTRY))
#: Extension fallback for callers that pass no ``pot_type`` (e.g.
#: check_potential_cif_compatibility): the extension is then the only guard.
SINGLE_TYPE_EXTENSIONS = (
    ('.rebo', '.rebomos', '.airebo', '.meam', '.reaxff') + MLIP_MODEL_EXTENSIONS)

def cifread(cif_path: Union[str, Path]) -> Dict[str, Any]:
    """Read a CIF file and extract crystal structure information using ASE.

    Args:
        cif_path: Path to the CIF file.

    Returns:
        Dictionary containing lattice constants ('lat_a', 'lat_b', 'lat_c'),
        cell angles, chemical formula, and list of elements.
    """
    cif_path = Path(cif_path)
    filename = cif_path.stem

    struct = read_structure(cif_path)
    cell = struct['cellpar']

    symbols = struct['symbols']
    elements = list(dict.fromkeys(symbols))
    elem_comp = {el: symbols.count(el) for el in elements}

    return {
        'lat_a': float(cell[0]),
        'lat_b': float(cell[1]),
        'lat_c': float(cell[2]),
        'ang_a': float(cell[3]),
        'ang_b': float(cell[4]),
        'ang_g': float(cell[5]),
        'formula': struct['formula'],
        'elements': elements,
        'elem_comp': elem_comp,
        'nelements': len(elements),
        'filename': filename
    }


def count_atomtypes(
    potential_filepath: Union[str, Path],
    elements: List[str],
    pot_type: Optional[str] = None
) -> Dict[str, int]:
    """Count the number of different atom types per element in a potential file.

    Args:
        potential_filepath: Path to the LAMMPS potential file.
        elements: List of element symbols to look for.
        pot_type: Potential type string (e.g., 'sw', 'reaxff'). Used as
            fallback when file extension is ambiguous.

    Returns:
        Dictionary where keys are element names and values are the count of
        unique atom types for that element (e.g., {'C': 2} for C1, C2).
    """
    elem_type = {el: 0 for el in elements}
    str_path = str(potential_filepath)
    pot_lower = (pot_type or '').lower()

    if (pot_lower in SINGLE_TYPE_POTENTIALS
            or str_path.lower().endswith(SINGLE_TYPE_EXTENSIONS)):
        return {el: 1 for el in elements}

    pattern = re.compile(r'([A-Za-z]+)(\d*)')

    with open(potential_filepath, 'r', encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        stripped_line = line.strip()
        if stripped_line.startswith('#') or not stripped_line:
            continue

        parts = stripped_line.split()
        for element in parts[:3]:
            match = pattern.match(element)
            if match:
                element_name = match.group(1)
                element_number = int(match.group(2)) if match.group(2) else 1
                if element_name in elem_type:
                    elem_type[element_name] = max(elem_type[element_name], element_number)
    return elem_type


def _normalize_element_symbol(value: str) -> str:
    """Normalize element-like tokens to canonical symbol case."""
    token = value.strip()
    if not token:
        return token
    return token[0].upper() + token[1:].lower()
