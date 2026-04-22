"""A collection of utility functions for file processing and calculations.

This module provides tools for reading, writing, and modifying simulation
files (e.g., CIF, LAMMPS data), parsing configuration files, and calculating
physical parameters like Lennard-Jones coefficients.
"""
import configparser
import json
import re
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Union, Any, Optional
import math
from ase.io import read as ase_read

from . import UFF_params as lj

logger = logging.getLogger(__name__)

# LAMMPS file section markers
LAMPS_SECTION_ATOMS = 'Atoms'
LAMMPS_SECTION_MASSES = 'Masses'
LAMMPS_SECTION_VELOCITIES = 'Velocities'

# LAMMPS atom style markers
LAMMPS_STYLE_ATOMIC = '# atomic'
LAMMPS_STYLE_CHARGE = '# charge'
LAMMPS_STYLE_MOLECULAR = '# molecular'


def _is_float(value: str) -> bool:
    """Check if a string represents a valid float, including scientific notation.
    
    Handles integer floats, decimal floats, and scientific notation (e.g., 1e-5, -3.2e+4).
    
    Args:
        value: String to check.
    
    Returns:
        True if value can be parsed as a float, False otherwise.
    """
    try:
        float(value)
        return True
    except ValueError:
        return False


def get_material_path(mat_name: str) -> Path:
    """Find material files.

    Args:
        mat_name: Material identifier or file path.

    Returns:
        Path to the material file.

    Raises:
        FileNotFoundError: If the material file is not found.
    """
    user_path = Path(mat_name)
    if user_path.exists():
        return user_path

    raise FileNotFoundError(
        f"Material file '{mat_name}' not found. "
        f"Please provide a valid path to a CIF file."
        f"For a collection of 2D material CIF files, check the 'examples/materials' "
        f"folder in the GitHub repository: https://github.com/Mclat16/FrictionSim2D"
    )

def get_potential_path(pot_name: str) -> Path:
    """Find potential files.

    Args:
        pot_name: Potential name or file path.

    Returns:
        Path to the potential file.

    Raises:
        FileNotFoundError: If the potential file is not found.
    """
    pot_path = Path(pot_name)
    if pot_path.exists():
        return pot_path

    raise FileNotFoundError(
        f"Potential file '{pot_name}' not found. "
        f"Please provide a valid path to a potential file. "
        f"For a collection of potential files (sw, tersoff, rebo, reaxff, etc.), "
        f"check the 'examples/potentials' folder in the GitHub repository: "
        f"https://github.com/Mclat16/FrictionSim2D"
    )

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

    atoms = ase_read(str(cif_path))
    if isinstance(atoms, list):
        atoms = atoms[0]
    cell = atoms.cell.cellpar()

    symbols = atoms.get_chemical_symbols()
    elements = list(dict.fromkeys(symbols))
    elem_comp = {el: symbols.count(el) for el in elements}

    return {
        'lat_a': float(cell[0]),
        'lat_b': float(cell[1]),
        'lat_c': float(cell[2]),
        'ang_a': float(cell[3]),
        'ang_b': float(cell[4]),
        'ang_g': float(cell[5]),
        'formula': atoms.get_chemical_formula(mode='hill'),
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

    single_type_pots = {'reaxff', 'reax/c', 'rebo', 'rebomos', 'airebo', 'meam'}
    single_type_exts = ('.rebo', '.rebomos', '.airebo', '.meam', '.reaxff')

    if pot_lower in single_type_pots or str_path.lower().endswith(single_type_exts):
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

def get_model_dimensions(lmp_path: Union[str, Path]) -> Dict[str, Optional[float]]:
    """Read a LAMMPS data file and extract the simulation box dimensions.

    Args:
        lmp_path: Path to the LAMMPS data file.

    Returns:
        Dictionary containing the box dimensions with keys 'xlo', 'xhi',
        'ylo', 'yhi', 'zlo', 'zhi'. Returns None for any dimension that
        cannot be parsed from the file.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is missing critical dimension lines.
    """
    lmp_path = Path(lmp_path)
    if not lmp_path.exists():
        raise FileNotFoundError(f"LAMMPS data file not found: {lmp_path}")

    dims: Dict[str, Optional[float]] = {k: None for k in ["xlo", "xhi", "ylo", "yhi", "zlo", "zhi"]}
    found_dims = set()

    with open(lmp_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            try:
                if "xlo xhi" in line:
                    dims["xlo"], dims["xhi"] = map(float, line.split()[0:2])
                    found_dims.add("xlo")
                elif "ylo yhi" in line:
                    dims["ylo"], dims["yhi"] = map(float, line.split()[0:2])
                    found_dims.add("ylo")
                elif "zlo zhi" in line:
                    dims["zlo"], dims["zhi"] = map(float, line.split()[0:2])
                    found_dims.add("zlo")
            except (ValueError, IndexError) as e:
                logger.warning("Failed to parse dimension line '%s': %s", line.strip(), e)

    expected_dims = {"xlo", "ylo", "zlo"}
    missing = expected_dims - found_dims
    if missing:
        logger.warning(
            "LAMMPS file %s missing dimension markers: %s. Some dimension values are None.",
            lmp_path, missing
        )

    return dims

def get_num_atom_types(lmp_path: Union[str, Path]) -> int:
    """Read a LAMMPS data file and extract the number of atom types.

    Args:
        lmp_path: Path to the LAMMPS data file.

    Returns:
        Number of atom types in the file.
    """
    with open(lmp_path, "r", encoding="utf-8") as f:
        for line in f:
            if "atom types" in line:
                return int(line.split()[0])
    return 1

def _normalize_element_symbol(value: str) -> str:
    """Normalize element-like tokens to canonical symbol case."""
    token = value.strip()
    if not token:
        return token
    return token[0].upper() + token[1:].lower()


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

def _remove_inline_comments(config: configparser.ConfigParser) -> configparser.ConfigParser:
    """Remove inline comments from a ConfigParser object.

    Args:
        config: The ConfigParser object to process.

    Returns:
        The object with inline comments removed.
    """
    for section in config.sections():
        for item in config.items(section):
            config.set(section, item[0], item[1].split("#")[0].strip())
    return config

def read_config(filepath: Union[str, Path]) -> Dict[str, Dict[str, Any]]:
    """Read a configuration file and return a dictionary with parsed values.

    Args:
        filepath: Path to the configuration file.

    Returns:
        Dictionary containing the parsed configuration parameters.
    """
    config = configparser.ConfigParser()
    config.read(filepath)
    config = _remove_inline_comments(config)

    params = {}
    for section in config.sections():
        params[section] = {}
        for key in config[section]:
            value = config.get(section, key)

            if value == '':
                params[section][key] = None
                continue

            if value.endswith(']'):
                try:
                    params[section][key] = json.loads(value)
                except json.JSONDecodeError:
                    cleaned = value.strip('[]')
                    items = [item.strip() for item in cleaned.split(',')]
                    parsed_items = []
                    for item in items:
                        if item.isdigit():
                            parsed_items.append(int(item))
                        elif _is_float(item):
                            parsed_items.append(float(item))
                        else:
                            parsed_items.append(item)
                    params[section][key] = parsed_items
            elif value.isdigit():
                params[section][key] = int(value)
            elif _is_float(value):
                params[section][key] = float(value)
            else:
                params[section][key] = value
    return params

def atomic2charge(filepath: Union[str, Path]) -> None:
    """Convert a LAMMPS data file from atomic to charge format in-place.

    Modifies the "Atoms" section of a LAMMPS data file, changing the style
    from 'atomic' to 'charge' and inserting a zero charge field between
    atom type and position. No-op if the file is already in charge format.

    Args:
        filepath: Path to the LAMMPS data file to be modified.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    atoms_section = False
    modified_lines = []

    for line in lines:
        line = line.strip()

        if line.startswith(LAMMPS_SECTION_VELOCITIES):
            break

        if line == f"{LAMPS_SECTION_ATOMS} {LAMMPS_STYLE_ATOMIC}":
            modified_lines.append(f"{LAMPS_SECTION_ATOMS} {LAMMPS_STYLE_CHARGE}")
            atoms_section = True
            continue

        if atoms_section and line:
            parts = line.split()
            if len(parts) >= 5 and all(c in '0123456789.-+eE' for c in parts[0]):
                atom_id = parts[0]
                atom_type = parts[1]
                x, y, z = parts[2:5]
                modified_lines.append(f"{atom_id} {atom_type} 0.0 {x} {y} {z}")
                continue

        modified_lines.append(line)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("\n".join(modified_lines) + "\n")


def atomic2molecular(filepath: Union[str, Path]) -> None:
    """Convert a LAMMPS data file from atomic to molecular format in-place.

    Modifies the "Atoms" section of a LAMMPS data file, changing the style
    from 'atomic' to 'molecular' and adding the required molecule ID and
    charge/dipole fields.

    Args:
        filepath: Path to the LAMMPS data file to be modified.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    atoms_section = False
    modified_lines = []

    for line in lines:
        line = line.strip()

        if line.startswith(LAMMPS_SECTION_VELOCITIES):
            break

        if line == f"{LAMPS_SECTION_ATOMS} {LAMMPS_STYLE_ATOMIC}":
            modified_lines.append(f"{LAMPS_SECTION_ATOMS} {LAMMPS_STYLE_MOLECULAR}")
            atoms_section = True
            continue

        if atoms_section and line:
            parts = line.split()
            if len(parts) >= 4 and all(c in '0123456789.-+eE' for c in parts[0]):
                atom_id = parts[0]
                atom_type = parts[1]
                x, y, z = parts[2:5]
                new_line = f"{atom_id} 0 {atom_type} {x} {y} {z} 0 0 0"
                modified_lines.append(new_line)
                continue

        modified_lines.append(line)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("\n".join(modified_lines) + "\n")


def renumber_atom_types(filename: Union[str, Path], pot: Optional[List[str]] = None) -> None:
    """Renumber atom types in a LAMMPS data file to sequential order.

    Modifies a LAMMPS data file in-place to ensure atom types are numbered
    sequentially from 1. If a potential `pot` is provided, renumbers the types
    to match the order of elements in that list.
    
    Algorithm:
    1. Parse the Masses section to build a map of type_id -> (element_name, mass).
    2. Scan Atoms section and renumber each atom's type field to sequential order.
    3. If pot list is provided, reorder atoms by element to match pot order.
    4. Rebuild Masses section with new sequential type IDs.

    Args:
        filename: Path to the LAMMPS data file to be modified.
        pot: List of element symbols in the desired order for renumbering.
    """
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    masses_idx = next((i for i, line in enumerate(lines) if line.strip() == LAMMPS_SECTION_MASSES), None)
    atoms_idx = next((i for i, line in enumerate(lines) if LAMPS_SECTION_ATOMS in line), None)
    if masses_idx is None or atoms_idx is None or masses_idx >= atoms_idx:
        return

    # Parse Masses section entries: old_type_id -> (element_name, mass)
    atom_types: Dict[int, Tuple[str, float]] = {}
    mass_line_indices: List[int] = []
    for i in range(masses_idx + 1, atoms_idx):
        stripped = lines[i].strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) < 2 or not parts[0].isdigit():
            continue
        type_id = int(parts[0])
        mass = float(parts[1])
        element_name = lines[i].split('#', maxsplit=1)[1].strip() if '#' in lines[i] else f'Unknown_{type_id}'
        atom_types[type_id] = (element_name, mass)
        mass_line_indices.append(i)

    if not atom_types:
        return

    # Build old_type -> new_type mapping, optionally following potential element order.
    old_type_ids = sorted(atom_types.keys())
    old_to_new: Dict[int, int] = {}
    next_type = 1

    if pot:
        remaining = list(old_type_ids)
        for element in pot:
            matches = [
                old_id for old_id in remaining
                if atom_types[old_id][0].upper() == element.upper()
            ]
            for old_id in matches:
                old_to_new[old_id] = next_type
                next_type += 1
                remaining.remove(old_id)

        for old_id in remaining:
            old_to_new[old_id] = next_type
            next_type += 1
    else:
        for old_id in old_type_ids:
            old_to_new[old_id] = next_type
            next_type += 1

    # Renumber atom types in Atoms section without touching atom IDs.
    section_headers = {
        LAMMPS_SECTION_MASSES,
        LAMPS_SECTION_ATOMS,
        LAMMPS_SECTION_VELOCITIES,
        'Bonds',
        'Angles',
        'Dihedrals',
        'Impropers',
        'Pair Coeffs',
        'Bond Coeffs',
        'Angle Coeffs',
        'Dihedral Coeffs',
        'Improper Coeffs',
    }

    for i in range(atoms_idx + 1, len(lines)):
        stripped = lines[i].strip()
        if not stripped:
            continue
        if stripped in section_headers or any(stripped.startswith(h) for h in section_headers):
            break

        parts = stripped.split()
        if len(parts) < 2 or not parts[0].isdigit() or not parts[1].isdigit():
            continue

        old_type = int(parts[1])
        if old_type in old_to_new:
            parts[1] = str(old_to_new[old_type])
            lines[i] = '  '.join(parts) + '\n'

    # Update atom types header count.
    n_types = len(old_to_new)
    for i, line in enumerate(lines):
        if re.match(r'^\s*\d+\s+atom types\s*$', line.strip()):
            lines[i] = f"  {n_types}  atom types\n"
            break

    # Rewrite Masses section to match new type ordering.
    new_type_to_elem_mass = {new_id: atom_types[old_id] for old_id, new_id in old_to_new.items()}
    mass_lines = [f"{new_id} {new_type_to_elem_mass[new_id][1]}  #{new_type_to_elem_mass[new_id][0]}\n"
                  for new_id in sorted(new_type_to_elem_mass.keys())]

    if mass_line_indices:
        insert_at = mass_line_indices[0]
        for idx in reversed(mass_line_indices):
            del lines[idx]
        for offset, mass_line in enumerate(mass_lines):
            lines.insert(insert_at + offset, mass_line)
    else:
        lines[masses_idx + 1:masses_idx + 1] = mass_lines

    with open(filename, 'w', encoding='utf-8') as f:
        f.writelines(lines)

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
