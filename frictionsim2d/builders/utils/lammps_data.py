"""Reading and rewriting LAMMPS data files (atom styles, section surgery).

Lives under builders/ rather than core/utils/ because every consumer is a
builder: core, run, potentials, lattice and gui never touch a LAMMPS data
file directly. get_model_dimensions and get_num_atom_types were previously
filed under core/utils/structures.py next to cifread -- they read LAMMPS data
files, not structure files, and use no ASE.
"""
import logging
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

from ...core.utils.tokens import is_int_token

logger = logging.getLogger(__name__)

# LAMMPS file section markers
LAMMPS_SECTION_ATOMS = 'Atoms'


# LAMMPS atom style markers
LAMMPS_STYLE_ATOMIC = '# atomic'
LAMMPS_STYLE_CHARGE = '# charge'
LAMMPS_STYLE_MOLECULAR = '# molecular'


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

        if stripped == f"{LAMMPS_SECTION_ATOMS} {source_style}":
            converted_lines.append(f"{LAMMPS_SECTION_ATOMS} {target_style}\n")
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
        if len(parts) < 5 or not is_int_token(parts[0]) or not is_int_token(parts[1]):
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
        if len(parts) < 6 or not is_int_token(parts[0]) or not is_int_token(parts[1]):
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
        if len(parts) < 5 or not is_int_token(parts[0]) or not is_int_token(parts[1]):
            return None
        # Preserve image flags (nx, ny, nz) and any trailing atom metadata.
        return ' '.join([parts[0], '0', parts[1]] + parts[2:])

    _convert_atoms_style(filepath, LAMMPS_STYLE_ATOMIC, LAMMPS_STYLE_MOLECULAR, _to_molecular)


def _atom_lines(lines: List[str]) -> Iterator[Tuple[int, List[str]]]:
    """``(index, fields)`` of every atom line in the ``Atoms`` section."""
    in_atoms = False
    for i, line in enumerate(lines):
        if LAMMPS_SECTION_ATOMS in line:
            in_atoms = True
            continue
        if not in_atoms or not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 5 and parts[0].lstrip('-').isdigit():
            yield i, parts


def shift_atoms_to_z_zero(filename: Union[str, Path]) -> None:
    """Shift all atom z-coordinates so the minimum z is 0, and update the box bounds.

    Modifies the LAMMPS data file in-place. After this operation, the lowest
    atom z-coordinate will be 0.0, and zlo/zhi are updated accordingly.

    Args:
        filename: Path to the LAMMPS data file.
    """
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    z_values: List[float] = []
    for _, parts in _atom_lines(lines):
        try:
            z_values.append(float(parts[4]))
        except ValueError:
            pass

    if not z_values:
        return

    z_min = min(z_values)
    if abs(z_min) < 1e-10:
        return  # Already at zero, nothing to do

    for i, parts in _atom_lines(lines):
        try:
            parts[4] = f"{float(parts[4]) - z_min:.15f}"
            lines[i] = '  '.join(parts) + '\n'
        except ValueError:
            pass
    for i, line in enumerate(lines):
        if 'zlo zhi' in line:
            zlo, zhi = map(float, line.split()[:2])
            lines[i] = f"      {zlo - z_min:.15f}      {zhi - z_min:.15f}  zlo zhi\n"

    with open(filename, 'w', encoding='utf-8') as f:
        f.writelines(lines)


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


def _atom_z_column(n_columns: int) -> int:
    """Index of the z column in an Atoms line of ``n_columns`` fields.

    atomic is ``id type x y z`` (5, or 8 with image flags); charge is
    ``id type q x y z`` (6, or 9 with image flags).
    """
    return 5 if n_columns in (6, 9) else 4


def atom_count_and_z_extent(path: Union[str, Path]) -> Tuple[int, float]:
    """Atom count in the ``Atoms`` section and ``max(z) - min(z)``.

    Handles ``atomic`` (``id type x y z``) and ``charge``
    (``id type q x y z``) styles: the last three numeric columns before any
    image flags are the coordinates, so the z column is the one found by
    :func:`_atom_z_column`.
    """
    n_atoms = 0
    z_values: List[float] = []
    for _, parts in _atom_lines(Path(path).read_text(encoding='utf-8').splitlines()):
        try:
            z_values.append(float(parts[_atom_z_column(len(parts))]))
        except ValueError:
            continue
        n_atoms += 1
    if not z_values:
        return 0, 0.0
    return n_atoms, max(z_values) - min(z_values)
