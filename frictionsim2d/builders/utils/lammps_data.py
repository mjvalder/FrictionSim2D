"""Reading and rewriting LAMMPS data files (atom styles, section surgery).

Lives under builders/ rather than core/utils/ because every consumer is a
builder: core, run, potentials and gui never touch a LAMMPS data file
directly. Uses no ASE.
"""
import logging
import re
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

import numpy as np

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


def atoms_section_start(lines: List[str]) -> Optional[Tuple[int, str]]:
    """``(index of the first atom line, atom style)`` of a data file, or None without Atoms.

    The style is ``'charge'`` when the ``Atoms`` header comment says so, else ``'atomic'``.
    """
    for i, line in enumerate(lines):
        if re.match(r'^\s*Atoms\s*(#|$)', line):
            return i + 2, 'charge' if 'charge' in line.lower() else 'atomic'
    return None


def parse_lmp_atoms(
    lmp_path: Path,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Parse atom types and Cartesian positions from a LAMMPS data file.

    Supports both 'atomic' and 'charge' atom styles (auto-detected from the
    ``Atoms`` header comment).

    Returns:
        Tuple of (types, xs, ys, zs) numpy arrays, or None if no atoms found.
    """
    lines = lmp_path.read_text(encoding='utf-8').splitlines()

    section = atoms_section_start(lines)
    if section is None:
        return None
    atoms_start, atom_style = section

    types: List[int] = []
    xs: List[float] = []
    ys: List[float] = []
    zs: List[float] = []

    for line in lines[atoms_start:]:
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            if types:
                break
            continue
        parts = stripped.split()
        try:
            if atom_style == 'charge':
                t = int(parts[1])
                x, y, z = float(parts[3]), float(parts[4]), float(parts[5])
            else:
                t = int(parts[1])
                x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
            types.append(t)
            xs.append(x)
            ys.append(y)
            zs.append(z)
        except (ValueError, IndexError):
            break

    if not types:
        return None
    return np.array(types), np.array(xs), np.array(ys), np.array(zs)


def parse_cell_atoms_masses(content: str):
    """Parse a LAMMPS ``atomic``-style data file string.

    Returns:
        Tuple of (cell 3x3 list, atoms [(type, x, y, z)], masses {type: (mass, element)}).
        Atoms are returned in file order; the cell includes any xy/xz/yz tilt.
    """
    bounds = {}
    tilt = {'xy': 0.0, 'xz': 0.0, 'yz': 0.0}
    masses: Dict[int, tuple] = {}
    atoms = []
    section = None
    for raw in content.splitlines():
        line = raw.strip()
        if not line:
            continue
        low = line.split('#', 1)[0].strip()
        if low.endswith(('xlo xhi', 'ylo yhi', 'zlo zhi')):
            parts = low.split()
            key = parts[2][0]  # 'x' | 'y' | 'z'
            bounds[f'{key}lo'] = float(parts[0])
            bounds[f'{key}hi'] = float(parts[1])
            continue
        if low.endswith('xy xz yz'):
            parts = low.split()
            tilt['xy'], tilt['xz'], tilt['yz'] = (float(parts[0]), float(parts[1]), float(parts[2]))
            continue
        if line in ('Masses', 'Atoms') or line.startswith('Atoms'):
            section = 'Masses' if line == 'Masses' else 'Atoms'
            continue
        if line in ('Velocities', 'Bonds', 'Angles'):
            section = None
            continue
        if section == 'Masses':
            parts = line.split()
            if parts[0].lstrip('-').isdigit():
                elem = line.split('#')[-1].strip().split()[0] if '#' in line else 'X'
                masses[int(parts[0])] = (float(parts[1]), elem)
        elif section == 'Atoms':
            parts = low.split()
            if len(parts) >= 5 and parts[0].lstrip('-').isdigit():
                atoms.append((int(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])))

    lx = bounds['xhi'] - bounds['xlo']
    ly = bounds['yhi'] - bounds['ylo']
    lz = bounds['zhi'] - bounds['zlo']
    cell = [[lx, 0.0, 0.0], [tilt['xy'], ly, 0.0], [tilt['xz'], tilt['yz'], lz]]
    return cell, atoms, masses


def write_typed_atomic(output_path: Path, cell, positions, types,
                        masses: Dict[int, tuple]) -> None:
    """Write a LAMMPS ``atomic`` data file preserving explicit per-atom types.

    Args:
        output_path: Destination path.
        cell: 3x3 lower-triangular cell (rows are a, b, c vectors).
        positions: Nx3 atom coordinates.
        types: length-N per-atom integer type ids.
        masses: mapping type-id -> (mass, element) for the Masses section.
    """
    n_types = max(masses) if masses else max(types)
    lx, ly, lz = cell[0][0], cell[1][1], cell[2][2]
    xy, xz, yz = cell[1][0], cell[2][0], cell[2][1]
    lines = [
        "LAMMPS data file via FrictionSim2D (type-preserving orthogonalization)\n\n",
        f"{len(positions)} atoms\n",
        f"{n_types} atom types\n\n",
        f"0.0 {lx:.12f} xlo xhi\n",
        f"0.0 {ly:.12f} ylo yhi\n",
        f"0.0 {lz:.12f} zlo zhi\n",
    ]
    if any(abs(v) > 1e-10 for v in (xy, xz, yz)):
        lines.append(f"{xy:.12f} {xz:.12f} {yz:.12f} xy xz yz\n")
    lines.append("\nMasses\n\n")
    for t in range(1, n_types + 1):
        mass, elem = masses.get(t, (1.0, 'X'))
        lines.append(f"{t} {mass:.6f}  # {elem}\n")
    lines.append("\nAtoms # atomic\n\n")
    for i, (pos, t) in enumerate(zip(positions, types), start=1):
        lines.append(f"{i} {int(t)} {pos[0]:.12f} {pos[1]:.12f} {pos[2]:.12f}\n")
    output_path.write_text(''.join(lines), encoding='utf-8')


def snap_small_tilts(path: Path, tol: float = 0.05) -> None:
    """Drop a negligible xy/xz/yz tilt line so the box is treated as orthogonal.

    Near-hexagonal cells orthogonalize to a box with a sub-0.01 Å residual tilt
    (float non-commensurability). LAMMPS then refuses to stack such a box onto an
    orthogonal one ("cannot switch to restricted triclinic"), so we snap tiny
    tilts to zero (atom displacement over the box is far below tol).
    """
    lines = path.read_text(encoding='utf-8').splitlines(keepends=True)
    out = []
    for line in lines:
        low = line.split('#', 1)[0].strip()
        if low.endswith('xy xz yz'):
            parts = low.split()
            if all(abs(float(parts[i])) < tol for i in range(3)):
                continue  # drop the tilt line -> orthogonal box
        out.append(line)
    path.write_text(''.join(out), encoding='utf-8')
