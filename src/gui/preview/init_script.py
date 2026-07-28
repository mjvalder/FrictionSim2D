"""Parsing a generated LAMMPS assembly script into a cell and placements.

The preview replays what the generated script tells LAMMPS to do rather than
recomputing component positions from the config. That is the whole point: the
placement arithmetic already happened in the builder, and a second
implementation of it would be free to disagree.

Two grammars occur, both produced by this project's templates:

*Placed* -- an explicit box plus one ``read_data`` per component::

    region box block 0.0 60.49 0.0 60.66 -5 118.5
    create_box 9 box
    read_data <root>/build/sub.lmp add append group sub
    read_data <root>/build/tip.lmp add append shift 30.2 30.3 68.5 group tip offset 3 0 0 0 0

*Pre-assembled* -- a single already-assembled data file, no box line::

    read_data <root>/build/h-MoS2_2.lmp

In the second case the cell comes from the data file's own header.

Paths inside these scripts are relative to the **simulation root**, not to the
script's directory.

Standard library only: no Qt, no ovito.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

__all__ = ['Placement', 'Assembly', 'parse_assembly', 'read_data_file_box']

_REGION_BOX = re.compile(
    r'^\s*region\s+\S+\s+block\s+(?P<bounds>.+?)(?:\s+units\s+\w+)?\s*$'
)
_READ_DATA = re.compile(r'^\s*read_data\s+(?P<rest>.+?)\s*$')

#: Box bounds in a LAMMPS data file header, e.g. ``0.0 28.65 xlo xhi``.
_DATA_BOX = re.compile(
    r'^\s*(?P<lo>-?[\d.eE+-]+)\s+(?P<hi>-?[\d.eE+-]+)\s+(?P<axis>[xyz])lo\s+(?P=axis)hi'
)

#: A LAMMPS variable reference such as ``$(v_find)``. Files named this way are
#: run-time state, not components, and cannot be resolved at preview time.
_LAMMPS_VARIABLE = re.compile(r'[$&]')


@dataclass(frozen=True)
class Placement:
    """One component read into the system."""

    file: str
    shift: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    group: Optional[str] = None
    type_offset: int = 0

    @property
    def is_runtime_state(self) -> bool:
        """True for files named by an unexpanded LAMMPS variable."""
        return bool(_LAMMPS_VARIABLE.search(self.file))

    def resolve(self, simulation_root: Path) -> Optional[Path]:
        """Absolute path of this component, or None if it does not exist.

        Resolution is against the simulation root: that is what the generated
        scripts are written relative to.
        """
        if self.is_runtime_state:
            return None
        candidate = Path(simulation_root) / self.file
        return candidate if candidate.exists() else None


@dataclass
class Assembly:
    """What a generated assembly script builds."""

    source: Path
    simulation_root: Path
    cell: Optional[Tuple[float, float, float, float, float, float]] = None
    placements: List[Placement] = field(default_factory=list)

    @property
    def is_preassembled(self) -> bool:
        """True when the script reads one already-assembled file and has no box."""
        return self.cell is None and len(self.placements) == 1

    def resolved_placements(self) -> List[Tuple[Placement, Path]]:
        """Placements whose component file exists, with its absolute path."""
        out = []
        for placement in self.placements:
            path = placement.resolve(self.simulation_root)
            if path is not None:
                out.append((placement, path))
        return out

    def effective_cell(self) -> Optional[Tuple[float, ...]]:
        """The simulation cell, from the script or from the data file header."""
        if self.cell is not None:
            return self.cell
        resolved = self.resolved_placements()
        if len(resolved) == 1:
            return read_data_file_box(resolved[0][1])
        return None


def _floats(tokens: Sequence[str]) -> List[float]:
    values = []
    for token in tokens:
        try:
            values.append(float(token))
        except ValueError:
            return []
    return values


def _parse_read_data(rest: str) -> Optional[Placement]:
    """Parse the tail of a ``read_data`` line."""
    tokens = rest.split()
    if not tokens:
        return None

    filename = tokens[0]
    shift = (0.0, 0.0, 0.0)
    group = None
    type_offset = 0

    index = 1
    while index < len(tokens):
        token = tokens[index]
        if token == 'shift' and index + 3 < len(tokens):
            values = _floats(tokens[index + 1:index + 4])
            if len(values) == 3:
                shift = (values[0], values[1], values[2])
                index += 4
                continue
        elif token == 'group' and index + 1 < len(tokens):
            group = tokens[index + 1]
            index += 2
            continue
        elif token == 'offset' and index + 1 < len(tokens):
            try:
                type_offset = int(tokens[index + 1])
            except ValueError:
                type_offset = 0
            index += 2
            continue
        index += 1

    return Placement(file=filename, shift=shift, group=group, type_offset=type_offset)


def parse_assembly(script: Path, simulation_root: Path) -> Assembly:
    """Parse a generated assembly script.

    Args:
        script: The script identified as the assembly step.
        simulation_root: Root that ``read_data`` paths are relative to.

    Returns:
        An :class:`Assembly`. Components naming run-time state (an unexpanded
        ``$(v_find)`` variable) are kept as placements but resolve to None.
    """
    script = Path(script)
    assembly = Assembly(source=script, simulation_root=Path(simulation_root))

    for line in script.read_text(encoding='utf-8').splitlines():
        stripped = line.split('#', 1)[0]
        if not stripped.strip():
            continue

        match = _REGION_BOX.match(stripped)
        if match and assembly.cell is None:
            bounds = _floats(match.group('bounds').split()[:6])
            if len(bounds) == 6:
                assembly.cell = (bounds[0], bounds[1], bounds[2],
                                 bounds[3], bounds[4], bounds[5])
            continue

        match = _READ_DATA.match(stripped)
        if match:
            placement = _parse_read_data(match.group('rest'))
            if placement is not None:
                assembly.placements.append(placement)

    return assembly


def read_data_file_box(data_file: Path) -> Optional[Tuple[float, ...]]:
    """Box bounds from a LAMMPS data file header, as ``(xlo, xhi, ...)``.

    Used for pre-assembled scripts, which carry no ``region`` line of their own.

    Returns:
        Six bounds, or None if the header does not define all three axes.
    """
    bounds = {}
    with open(data_file, 'r', encoding='utf-8') as handle:
        for line in handle:
            if line.strip().startswith('Atoms'):
                break
            match = _DATA_BOX.match(line)
            if match:
                bounds[match.group('axis')] = (
                    float(match.group('lo')), float(match.group('hi'))
                )

    if not all(axis in bounds for axis in 'xyz'):
        return None
    return (bounds['x'][0], bounds['x'][1],
            bounds['y'][0], bounds['y'][1],
            bounds['z'][0], bounds['z'][1])
