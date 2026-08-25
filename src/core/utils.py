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


#: How far above ``base_dir`` to look for a config-relative file. Configs in
#: ``examples/`` name ``examples/materials/...``, i.e. paths relative to the
#: project root rather than to themselves, so the search has to climb.
PATH_SEARCH_MAX_PARENTS = 4


def search_paths(name: str, base_dir: Optional[Union[str, Path]] = None) -> List[Path]:
    """Candidate locations for a config-supplied file, in resolution order.

    Config files reference potentials and structures by relative path, and those
    paths are relative to wherever the user ran from -- in practice the project
    root, not the config's own directory. ``examples/indent_config.ini`` names
    ``examples/materials/h-MoS2.cif``, which resolves one level *above* itself.

    So the search tries the name as given (absolute, or relative to the current
    directory), then ``base_dir``, then each of its parents up to
    :data:`PATH_SEARCH_MAX_PARENTS`. That accommodates both conventions without
    asking anyone to rewrite existing configs, and it is what lets the GUI open
    one from an arbitrary working directory.

    Args:
        name: Path or filename from a config.
        base_dir: Directory to resolve relative names against, normally the
            directory containing the config file.

    Returns:
        Candidate paths, most specific first. Never empty.
    """
    candidates = [Path(name)]
    if base_dir is None or Path(name).is_absolute():
        return candidates

    root = Path(base_dir)
    for depth, directory in enumerate([root, *root.parents]):
        if depth > PATH_SEARCH_MAX_PARENTS:
            break
        candidates.append(directory / name)
    return candidates


def _find_existing(name: str, base_dir: Optional[Union[str, Path]]) -> Optional[Path]:
    """First existing candidate for ``name``, or None."""
    for candidate in search_paths(name, base_dir):
        if candidate.exists():
            return candidate
    return None


def get_material_path(mat_name: str,
                      base_dir: Optional[Union[str, Path]] = None) -> Path:
    """Find material files.

    Args:
        mat_name: Material identifier or file path.
        base_dir: Directory to resolve relative paths against (normally the
            directory holding the config file). Optional; when omitted the
            behaviour is the historical current-working-directory lookup.

    Returns:
        Path to the material file.

    Raises:
        FileNotFoundError: If the material file is not found.
    """
    found = _find_existing(mat_name, base_dir)
    if found is not None:
        return found

    raise FileNotFoundError(
        f"Material file '{mat_name}' not found. "
        f"Please provide a valid path to a CIF file."
        f"For a collection of 2D material CIF files, check the 'examples/materials' "
        f"folder in the GitHub repository: https://github.com/Mclat16/FrictionSim2D"
    )

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
    # interpolation=None: configparser's default BasicInterpolation raises on a
    # bare '%' in any value. No config in this project uses '%(name)s'
    # substitution, so the feature only ever cost us a crash on values a user
    # could reasonably type.
    config = configparser.ConfigParser(interpolation=None)
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


def _is_int_token(value: str) -> bool:
    """Return True if token represents an integer (optional sign allowed)."""
    return bool(re.fullmatch(r'[+-]?\d+', value))


def _is_lammps_section_header(line: str) -> bool:
    """Detect a section header line while scanning an Atoms block."""
    if not line:
        return False
    first = line[0]
    if first == '#':
        return False
    if first.isdigit() or first in ['-', '+', '.']:
        return False
    return True


def _convert_atoms_style(
    filepath: Union[str, Path],
    source_style: str,
    target_style: str,
    transform_atom_fields
) -> None:
    """Convert Atoms section style in-place using a per-line transformation callback."""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    in_atoms_section = False
    converted_lines = []

    for raw_line in lines:
        stripped = raw_line.strip()

        if stripped == f"{LAMPS_SECTION_ATOMS} {source_style}":
            converted_lines.append(f"{LAMPS_SECTION_ATOMS} {target_style}\n")
            in_atoms_section = True
            continue

        if in_atoms_section:
            if not stripped:
                converted_lines.append(raw_line)
                continue

            if _is_lammps_section_header(stripped):
                in_atoms_section = False
                converted_lines.append(raw_line)
                continue

            parts = stripped.split()
            transformed = transform_atom_fields(parts)
            if transformed is None:
                converted_lines.append(raw_line)
            else:
                converted_lines.append(f"{transformed}\n")
            continue

        converted_lines.append(raw_line)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(converted_lines)

def atomic2charge(filepath: Union[str, Path]) -> None:
    """Convert a LAMMPS data file from atomic to charge format in-place.

    Modifies the "Atoms" section of a LAMMPS data file, changing the style
    from 'atomic' to 'charge' and inserting a zero charge field between
    atom type and position. No-op if the file is already in charge format.

    Args:
        filepath: Path to the LAMMPS data file to be modified.
    """
    def _to_charge(parts: List[str]) -> Optional[str]:
        if len(parts) < 5 or not _is_int_token(parts[0]) or not _is_int_token(parts[1]):
            return None
        return ' '.join([parts[0], parts[1], '0.0'] + parts[2:])

    _convert_atoms_style(filepath, LAMMPS_STYLE_ATOMIC, LAMMPS_STYLE_CHARGE, _to_charge)


def charge2atom(filepath: Union[str, Path]) -> None:
    """Convert a LAMMPS data file from charge to atomic format in-place.

    Modifies the "Atoms" section by switching header style from charge to
    atomic and removing the charge column. If the file is not in charge style,
    this function is a no-op.

    Args:
        filepath: Path to the LAMMPS data file to be modified.
    """
    def _to_atomic(parts: List[str]) -> Optional[str]:
        if len(parts) < 6 or not _is_int_token(parts[0]) or not _is_int_token(parts[1]):
            return None
        return ' '.join([parts[0], parts[1]] + parts[3:])

    _convert_atoms_style(filepath, LAMMPS_STYLE_CHARGE, LAMMPS_STYLE_ATOMIC, _to_atomic)


def atomic2molecular(filepath: Union[str, Path]) -> None:
    """Convert a LAMMPS data file from atomic to molecular format in-place.

    Modifies the "Atoms" section of a LAMMPS data file, changing the style
    from 'atomic' to 'molecular' and adding the required molecule ID and
    charge/dipole fields.

    Args:
        filepath: Path to the LAMMPS data file to be modified.
    """
    def _to_molecular(parts: List[str]) -> Optional[str]:
        if len(parts) < 5 or not _is_int_token(parts[0]) or not _is_int_token(parts[1]):
            return None
        # Preserve image flags (nx, ny, nz) and any trailing atom metadata.
        return ' '.join([parts[0], '0', parts[1]] + parts[2:])

    _convert_atoms_style(filepath, LAMMPS_STYLE_ATOMIC, LAMMPS_STYLE_MOLECULAR, _to_molecular)


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


def shift_atoms_to_z_zero(filename: Union[str, Path]) -> None:
    """Shift all atom z-coordinates so the minimum z is 0, and update the box bounds.

    Modifies the LAMMPS data file in-place. After this operation, the lowest
    atom z-coordinate will be 0.0, and zlo/zhi are updated accordingly.

    Args:
        filename: Path to the LAMMPS data file.
    """
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Find minimum z across all atom lines
    atoms_section = False
    z_values: List[float] = []
    for line in lines:
        if LAMPS_SECTION_ATOMS in line:
            atoms_section = True
            continue
        if atoms_section and line.strip():
            parts = line.strip().split()
            if len(parts) >= 5 and parts[0].lstrip('-').isdigit():
                try:
                    z_values.append(float(parts[4]))
                except ValueError:
                    pass

    if not z_values:
        return

    z_min = min(z_values)
    if abs(z_min) < 1e-10:
        return  # Already at zero, nothing to do

    # Shift atom z-coordinates and update zlo/zhi box bounds
    atoms_section = False
    for i, line in enumerate(lines):
        if LAMPS_SECTION_ATOMS in line:
            atoms_section = True
            continue
        if atoms_section and line.strip():
            parts = line.strip().split()
            if len(parts) >= 5 and parts[0].lstrip('-').isdigit():
                try:
                    parts[4] = f"{float(parts[4]) - z_min:.15f}"
                    lines[i] = '  '.join(parts) + '\n'
                except ValueError:
                    pass
        elif 'zlo zhi' in line:
            zlo, zhi = map(float, line.split()[:2])
            lines[i] = f"      {zlo - z_min:.15f}      {zhi - z_min:.15f}  zlo zhi\n"

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


def format_numeric_token(value: float) -> str:
    """Format a numeric value into a filename-safe token string.
    
    Converts numbers to strings suitable for use in filenames and directory names.
    Replaces problematic characters: minus signs become 'm', decimals become 'p'.
    
    Args:
        value: Numeric value to format (int, float, or scientific notation)
    
    Returns:
        Alphanumeric token string (e.g., '1p5' for 1.5, 'm2p3' for -2.3)
    
    Example:
        >>> format_numeric_token(1.5)
        '1p5'
        >>> format_numeric_token(-2.3)
        'm2p3'
        >>> format_numeric_token(1e-5)
        '1e-05'
    """
    token = f"{value:g}"
    token = token.replace('-', 'm').replace('.', 'p')
    return re.sub(r'[^A-Za-z0-9_]+', '_', token)
