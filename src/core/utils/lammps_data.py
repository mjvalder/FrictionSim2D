"""Reading and rewriting LAMMPS data files (atom styles, section surgery)."""
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .tokens import _is_int_token  # noqa: PLC2701

logger = logging.getLogger(__name__)

# LAMMPS file section markers
LAMPS_SECTION_ATOMS = 'Atoms'
LAMMPS_SECTION_MASSES = 'Masses'
LAMMPS_SECTION_VELOCITIES = 'Velocities'


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